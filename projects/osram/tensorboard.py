"""Additional TensorBoard logging for OSRAM segmentation experiments."""

from typing import Optional, Sequence

import cv2
import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import HOOKS, METRICS, VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@HOOKS.register_module()
class BestMetricsVisualizationHook(Hook):
    """Log the best validation value reached for selected metrics."""

    def __init__(
        self,
        metrics=("mDice", "mIoU", "fg_mDice", "fg_mIoU"),
        classwise_metric_prefixes=("Dice/", "IoU/"),
        excluded_classes=("background",),
    ):
        self.metrics = tuple(metrics)
        self.classwise_metric_prefixes = tuple(classwise_metric_prefixes)
        self.excluded_classes = set(excluded_classes)

    def after_val_epoch(
        self,
        runner: Runner,
        metrics: Optional[dict] = None,
    ) -> None:
        if not metrics:
            return

        metric_names = set(self.metrics)
        for metric_name in metrics:
            if not metric_name.startswith(self.classwise_metric_prefixes):
                continue
            class_name = metric_name.rsplit("/", maxsplit=1)[-1]
            if class_name not in self.excluded_classes:
                metric_names.add(metric_name)

        for metric_name in sorted(metric_names):
            if metric_name not in metrics:
                continue

            current_value = float(metrics[metric_name])
            message_hub_key = f"best_metrics/{metric_name}"
            best_value = runner.message_hub.get_info(message_hub_key)

            if best_value is None or current_value > best_value:
                best_value = current_value
                runner.message_hub.update_info(message_hub_key, best_value)

            runner.visualizer.add_scalar(
                f"{metric_name}_best", best_value, step=runner.iter
            )


@HOOKS.register_module()
class EMAEarlyStoppingHook(Hook):
    """Stop when an exponentially smoothed validation metric plateaus.

    The default smoothing matches nnU-Net:
    ``ema = 0.9 * previous_ema + 0.1 * current_value``.
    """

    def __init__(
        self,
        monitor: str,
        rule: str = "greater",
        min_delta: float = 0.0,
        patience: int = 10,
        momentum: float = 0.9,
        check_finite: bool = True,
    ):
        if rule not in ("greater", "less"):
            raise ValueError('rule must be either "greater" or "less"')
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")

        self.monitor = monitor
        self.rule = rule
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.momentum = float(momentum)
        self.check_finite = check_finite
        self.ema = None
        self.best = None
        self.bad_count = 0

    def after_val_epoch(
        self,
        runner: Runner,
        metrics: Optional[dict] = None,
    ) -> None:
        if not metrics or self.monitor not in metrics:
            runner.logger.warning(
                f"Early stopping metric {self.monitor!r} is unavailable."
            )
            return

        current = float(metrics[self.monitor])
        if self.check_finite and not np.isfinite(current):
            runner.logger.warning(
                f"Early stopping metric {self.monitor!r} is not finite; "
                "training will be stopped."
            )
            runner.train_loop.stop_training = True
            return

        if self.ema is None:
            self.ema = current
        else:
            self.ema = (
                self.momentum * self.ema
                + (1.0 - self.momentum) * current
            )

        runner.visualizer.add_scalar(
            f"{self.monitor}_ema", self.ema, step=runner.iter
        )

        if self.best is None:
            improved = True
        elif self.rule == "greater":
            improved = self.ema > self.best + self.min_delta
        else:
            improved = self.ema < self.best - self.min_delta

        if improved:
            self.best = self.ema
            self.bad_count = 0
        else:
            self.bad_count += 1

        runner.logger.info(
            f"EMA early stopping: {self.monitor}={current:.4f}, "
            f"ema={self.ema:.4f}, best_ema={self.best:.4f}, "
            f"bad validations={self.bad_count}/{self.patience}"
        )

        if self.bad_count >= self.patience:
            runner.logger.info(
                f"Early stopping: EMA of {self.monitor!r} did not improve "
                f"by at least {self.min_delta} for "
                f"{self.patience} validations."
            )
            runner.train_loop.stop_training = True


@METRICS.register_module()
class ClasswiseIoUMetric(IoUMetric):
    """Add foreground and per-class metrics to the standard MMSeg metrics.

    ``fg_mIoU`` and ``fg_mDice`` are the respective means over all foreground
    classes; class index 0 (background) is excluded.
    """

    def compute_metrics(self, results):
        # Behält die bisherigen MMSeg-Metriken wie mIoU, mDice und
        # mFscore sowie die Ausgabe der Klassentabelle bei.
        metrics = super().compute_metrics(results)

        if self.format_only:
            return metrics

        transposed = tuple(zip(*results))

        per_class_metrics = self.total_area_to_metrics(
            total_area_intersect=sum(transposed[0]),
            total_area_union=sum(transposed[1]),
            total_area_pred_label=sum(transposed[2]),
            total_area_label=sum(transposed[3]),
            metrics=self.metrics,
            nan_to_num=self.nan_to_num,
            beta=self.beta,
        )

        # aAcc ist eine globale Metrik und keine Klassenmetrik.
        per_class_metrics.pop("aAcc", None)

        # Mean of every requested class metric over foreground classes only.
        # For mFscore MMSeg also returns Precision and Recall, so their
        # foreground means are exposed as fg_mPrecision and fg_mRecall.
        for metric_name, values in per_class_metrics.items():
            foreground_values = np.asarray(values[1:])
            metrics[f"fg_m{metric_name}"] = round(
                float(np.nanmean(foreground_values)) * 100.0, 2
            )

        # fg_mDice is always calculated even when mDice was not explicitly
        # requested, because checkpointing and early stopping depend on it.
        total_area_intersect = sum(transposed[0])
        total_area_pred_label = sum(transposed[2])
        total_area_label = sum(transposed[3])
        foreground_dice = (
            2 * total_area_intersect[1:]
            / (total_area_pred_label[1:] + total_area_label[1:])
        ).numpy()
        if self.nan_to_num is not None:
            foreground_dice = np.nan_to_num(
                foreground_dice, nan=self.nan_to_num
            )
        if "Dice" not in per_class_metrics:
            metrics["fg_mDice"] = round(
                float(np.nanmean(foreground_dice)) * 100.0, 2
            )

        for metric_name, values in per_class_metrics.items():
            for class_name, value in zip(
                self.dataset_meta["classes"], values
            ):
                # Verhindert, dass "/" im Klassennamen eine zusätzliche
                # TensorBoard-Gruppe erzeugt.
                safe_class_name = str(class_name).replace("/", "_")

                # MMSeg gibt seine Metriken standardmäßig in Prozent aus.
                metrics[f"{metric_name}/{safe_class_name}"] = (
                    float(value) * 100.0
                )

        return metrics


@VISUALIZERS.register_module()
class ThreePanelSegVisualizer(SegLocalVisualizer):
    """Visualize input image, ground-truth mask and prediction."""

    @staticmethod
    def _colorize_mask(segmentation, palette, image_shape):
        labels = segmentation.cpu().data.squeeze().numpy()

        color_mask = np.zeros(image_shape, dtype=np.uint8)

        for class_id, color in enumerate(palette):
            color_mask[labels == class_id] = color

        return color_mask

    @staticmethod
    def _add_title(image, title):
        title_height = 28

        canvas = np.zeros(
            (
                image.shape[0] + title_height,
                image.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        canvas[title_height:] = image

        cv2.putText(
            canvas,
            title,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample=None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        step: int = 0,
        with_labels: Optional[bool] = True,
    ) -> None:
        palette = self.dataset_meta["palette"]

        panels = [
            self._add_title(image, "Real image"),
        ]

        if (
            draw_gt
            and data_sample is not None
            and "gt_sem_seg" in data_sample
        ):
            ground_truth = self._colorize_mask(
                data_sample.gt_sem_seg,
                palette,
                image.shape,
            )

            panels.append(
                self._add_title(ground_truth, "Ground truth")
            )

        if (
            draw_pred
            and data_sample is not None
            and "pred_sem_seg" in data_sample
        ):
            prediction = self._colorize_mask(
                data_sample.pred_sem_seg,
                palette,
                image.shape,
            )

            panels.append(
                self._add_title(prediction, "Prediction")
            )

        drawn_image = np.concatenate(panels, axis=1)

        if show:
            self.show(
                drawn_image,
                win_name=name,
                wait_time=wait_time,
            )

        if out_file is not None:
            mmcv.imwrite(
                mmcv.rgb2bgr(drawn_image),
                out_file,
            )
        else:
            # Übergibt das Dreierbild unter Beibehaltung aller konfigurierten
            # Backends an LocalVisBackend und TensorboardVisBackend.
            self.add_image(
                name,
                drawn_image,
                step,
            )


@HOOKS.register_module()
class ClassBalancedSegVisualizationHook(Hook):
    """Log one fixed validation image for every requested anomaly class.

    During the first validation, the first image containing each requested
    ground-truth class is selected. The corresponding validation batch indices
    are retained, so all later validations visualize the same images.

    An image may contain multiple anomaly classes and may therefore be selected
    for more than one class.
    """

    def __init__(
        self,
        class_ids=(1, 2, 3, 4),
        backend_args=None,
    ):
        self.class_ids = tuple(class_ids)
        self.backend_args = (
            backend_args.copy() if backend_args is not None else None
        )

        # Zuordnung: class_id -> batch_idx
        self.selected_batches = {}

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
    ) -> None:
        if not outputs:
            return

        data_sample = outputs[0]

        if "gt_sem_seg" not in data_sample:
            return

        gt_mask = data_sample.gt_sem_seg.data.squeeze()

        classes_to_visualize = []

        # Ein Validierungsbild darf nur einer Anomalieklasse zugeordnet werden.
        batch_is_already_selected = (
            batch_idx in self.selected_batches.values()
        )

        for class_id in self.class_ids:
            if class_id in self.selected_batches:
                # Bei späteren Validierungen wieder dasselbe Bild darstellen.
                if self.selected_batches[class_id] == batch_idx:
                    classes_to_visualize.append(class_id)

                continue

            # Dieses Bild wurde bereits für eine andere Klasse ausgewählt.
            if batch_is_already_selected:
                continue

            contains_class = bool(
                (gt_mask == class_id).any().item()
            )

            if contains_class:
                self.selected_batches[class_id] = batch_idx
                classes_to_visualize.append(class_id)

                # Dieses Bild ist jetzt vergeben. Keine weitere Klasse für
                # dasselbe Bild auswählen.
                batch_is_already_selected = True
                break

        if not classes_to_visualize:
            return

        img_path = data_sample.img_path
        img_bytes = get(
            img_path,
            backend_args=self.backend_args,
        )
        image = mmcv.imfrombytes(
            img_bytes,
            channel_order="rgb",
        )

        class_names = runner.val_dataloader.dataset.metainfo["classes"]

        # runner.epoch ist nullbasiert.
        step = runner.epoch + 1

        for class_id in classes_to_visualize:
            class_name = class_names[class_id]

            runner.visualizer.add_datasample(
                name=f"validation/{class_name}",
                image=image,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=True,
                show=False,
                step=step,
            )

    def after_val_epoch(
        self,
        runner: Runner,
        metrics=None,
    ) -> None:
        logger = MMLogger.get_current_instance()

        missing_classes = [
            class_id
            for class_id in self.class_ids
            if class_id not in self.selected_batches
        ]

        if missing_classes:
            class_names = runner.val_dataloader.dataset.metainfo["classes"]

            missing_names = [
                class_names[class_id]
                for class_id in missing_classes
            ]

            logger.warning(
                "No validation images were found for these classes: "
                + ", ".join(missing_names)
            )
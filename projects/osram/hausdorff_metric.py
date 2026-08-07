"""Hausdorff-distance metrics for OSRAM segmentation experiments."""

from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable
from scipy.ndimage import binary_erosion, distance_transform_edt

from mmseg.registry import METRICS


@METRICS.register_module()
class HausdorffDistanceMetric(BaseMetric):
    """Compute per-class symmetric HD95 in pixels.

    Background is excluded by default. Samples for which prediction and
    ground truth are both empty are ignored for that class. If only one mask
    is empty, the image diagonal is used as a finite worst-case distance.
    Dataset values are macro averages over the eligible images.
    """

    default_prefix = "hd"

    def __init__(
        self,
        percentile: float = 95.0,
        include_background: bool = False,
        ignore_index: int = 255,
        collect_device: str = "cpu",
        prefix: str = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must satisfy 0 < percentile <= 100")
        self.percentile = float(percentile)
        self.include_background = include_background
        self.ignore_index = ignore_index

    @staticmethod
    def _surface(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask, border_value=0)
        return np.logical_and(mask, np.logical_not(eroded))

    @classmethod
    def _hd_percentile(
        cls,
        prediction: np.ndarray,
        target: np.ndarray,
        percentile: float,
    ) -> float:
        prediction_nonempty = bool(prediction.any())
        target_nonempty = bool(target.any())
        if not prediction_nonempty and not target_nonempty:
            return np.nan
        if prediction_nonempty != target_nonempty:
            height, width = prediction.shape
            return float(np.hypot(max(height - 1, 0), max(width - 1, 0)))

        prediction_surface = cls._surface(prediction)
        target_surface = cls._surface(target)
        distance_to_prediction = distance_transform_edt(~prediction_surface)
        distance_to_target = distance_transform_edt(~target_surface)
        prediction_to_target = distance_to_target[prediction_surface]
        target_to_prediction = distance_to_prediction[target_surface]
        return float(
            max(
                np.percentile(prediction_to_target, percentile),
                np.percentile(target_to_prediction, percentile),
            )
        )

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta["classes"])
        first_class = 0 if self.include_background else 1

        for data_sample in data_samples:
            prediction = (
                data_sample["pred_sem_seg"]["data"].squeeze().cpu().numpy()
            )
            target = data_sample["gt_sem_seg"]["data"].squeeze().cpu().numpy()
            valid = target != self.ignore_index

            class_distances = np.full(num_classes, np.nan, dtype=np.float64)
            for class_id in range(first_class, num_classes):
                class_distances[class_id] = self._hd_percentile(
                    np.logical_and(prediction == class_id, valid),
                    np.logical_and(target == class_id, valid),
                    self.percentile,
                )
            self.results.append(class_distances)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        class_names = self.dataset_meta["classes"]
        first_class = 0 if self.include_background else 1
        values = np.asarray(results, dtype=np.float64)
        per_class = np.full(len(class_names), np.nan, dtype=np.float64)
        for class_id in range(first_class, len(class_names)):
            eligible = values[:, class_id]
            if np.isfinite(eligible).any():
                per_class[class_id] = np.nanmean(eligible)

        metric_name = f"HD{self.percentile:g}"
        metrics = OrderedDict()
        for class_id in range(first_class, len(class_names)):
            safe_name = str(class_names[class_id]).replace("/", "_")
            metrics[f"{metric_name}/{safe_name}"] = float(per_class[class_id])
        metrics[f"fg_m{metric_name}"] = float(
            np.nanmean(per_class[first_class:])
        )

        table = PrettyTable()
        table.add_column("Class", class_names[first_class:])
        table.add_column(
            f"{metric_name} (px)",
            [
                round(float(value), 3) if np.isfinite(value) else "nan"
                for value in per_class[first_class:]
            ],
        )
        logger = MMLogger.get_current_instance()
        print_log(
            f"per-class {metric_name} results (pixels):", logger=logger
        )
        print_log("\n" + table.get_string(), logger=logger)
        return metrics

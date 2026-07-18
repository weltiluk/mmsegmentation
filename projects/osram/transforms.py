"""Dataset transforms for the OSRAM anomaly segmentation data."""

import mmcv
import mmengine.fileio as fileio
import numpy as np

from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadOsramAnnotations(LoadAnnotations):
    """Load a class-index mask and accept grayscale-encoded RGB PNGs.

    Most masks in the dataset are single-channel PNGs. Some synthetic masks
    contain the same class index in all three RGB channels. MMSegmentation's
    standard ``LoadAnnotations`` keeps these masks three-dimensional, so this
    transform collapses identical RGB channels to one label plane.
    """

    def _load_seg_map(self, results: dict) -> None:
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_seg_map = mmcv.imfrombytes(
            img_bytes, flag='unchanged', backend=self.imdecode_backend)

        if gt_seg_map.ndim == 3:
            if not np.all(gt_seg_map == gt_seg_map[..., :1]):
                raise ValueError(
                    'Expected identical channels in RGB segmentation mask: '
                    f"{results['seg_map_path']}")
            gt_seg_map = gt_seg_map[..., 0]

        gt_seg_map = gt_seg_map.squeeze().astype(np.uint8)
        if gt_seg_map.ndim != 2:
            raise ValueError(
                'Expected a 2D segmentation mask, got shape '
                f"{gt_seg_map.shape}: {results['seg_map_path']}")

        reduce_zero_label = results['reduce_zero_label']
        if self.reduce_zero_label is None:
            self.reduce_zero_label = reduce_zero_label
        assert self.reduce_zero_label == reduce_zero_label

        if reduce_zero_label:
            gt_seg_map[gt_seg_map == 0] = 255
            gt_seg_map = gt_seg_map - 1
            gt_seg_map[gt_seg_map == 254] = 255

        if results.get('label_map') is not None:
            original = gt_seg_map.copy()
            for old_id, new_id in results['label_map'].items():
                gt_seg_map[original == old_id] = new_id

        results['gt_seg_map'] = gt_seg_map
        results['seg_fields'].append('gt_seg_map')

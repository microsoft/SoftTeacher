import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import Collect

from ssod.core import TrimapMasks


@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


@PIPELINES.register_module()
class ExtraCollect(Collect):
    def __init__(self, *args, extra_meta_keys=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = self.meta_keys + tuple(extra_meta_keys)


@PIPELINES.register_module()
class PseudoSamples(object):
    def __init__(
        self, with_bbox=False, with_mask=False, with_seg=False, fill_value=255
    ):
        """
        Replacing gt labels in original data with fake labels or adding extra fake labels for unlabeled data.
        This is to remove the effect of labeled data and keep its elements aligned with other sample.
        Args:
            with_bbox:
            with_mask:
            with_seg:
            fill_value:
        """
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.fill_value = fill_value

    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 4))
            results["gt_labels"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            num_inst = len(results["gt_bboxes"])
            h, w = results["img"].shape[:2]
            results["gt_masks"] = TrimapMasks(
                [
                    self.fill_value * np.ones((h, w), dtype=np.uint8)
                    for _ in range(num_inst)
                ],
                h,
                w,
            )

            if "mask_fields" not in results:
                results["mask_fields"] = []
            if "gt_masks" not in results["mask_fields"]:
                results["mask_fields"].append("gt_masks")
        if self.with_seg:
            results["gt_semantic_seg"] = self.fill_value * np.ones(
                results["img"].shape[:2], dtype=np.uint8
            )
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results

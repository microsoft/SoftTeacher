"""
Designed for pseudo masks.
In a `TrimapMasks`, it allow some part of the mask is ignored when computing loss.
"""
import numpy as np
import torch
from mmcv.ops.roi_align import roi_align
from mmdet.core import BitmapMasks


class TrimapMasks(BitmapMasks):
    def __init__(self, masks, height, width, ignore_value=255):
        """
        Args:
            ignore_value: flag to ignore in loss computation.
        See `mmdet.core.BitmapMasks` for more information
        """
        super().__init__(masks, height, width)
        self.ignore_value = ignore_value

    def crop_and_resize(
        self, bboxes, out_shape, inds, device="cpu", interpolation="bilinear"
    ):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[
            :, None
        ]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = (
                torch.from_numpy(self.masks)
                .to(device)
                .index_select(0, inds)
                .to(dtype=rois.dtype)
            )
            targets = roi_align(
                gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, "avg", True
            ).squeeze(1)
            # for a mask:
            # value<0.5 -> background,
            # 0.5<=value<=1 -> foreground
            # value>1 -> ignored area
            resized_masks = (targets >= 0.5).float()
            resized_masks[targets > 1] = self.ignore_value
            resized_masks = resized_masks.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

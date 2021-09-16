# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device="cuda:0", cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.train_cfg = None

    if hasattr(config.model, "model"):
        config.model.model.pretrained = None
        config.model.model.train_cfg = None
    else:
        config.model.pretrained = None

    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        map_loc = "cpu" if device == "cpu" else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "Class names are not saved in the checkpoint's "
                "meta data, use COCO classes by default."
            )
            model.CLASSES = get_classes("coco")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def save_result(model, img, result, score_thr=0.3, out_file="res.png"):
    """Save the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str): Specifies where to save the visualization result
    """
    if hasattr(model, "module"):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        out_file=out_file,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
    )

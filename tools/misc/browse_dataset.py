import argparse
import os
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes

from ssod.datasets import build_dataset
from ssod.models.utils import Transform2D


def parse_args():
    parser = argparse.ArgumentParser(description="Browse a dataset")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--skip-type",
        type=str,
        nargs="+",
        default=["DefaultFormatBundle", "Normalize", "Collect"],
        help="skip some useless pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="If there is no display interface, you can save it",
    )
    parser.add_argument("--not-show", default=False, action="store_true")
    parser.add_argument(
        "--show-interval", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def remove_pipe(pipelines, skip_type):
    if isinstance(pipelines, list):
        new_pipelines = []
        for pipe in pipelines:
            pipe = remove_pipe(pipe, skip_type)
            if pipe is not None:
                new_pipelines.append(pipe)
        return new_pipelines
    elif isinstance(pipelines, dict):
        if pipelines["type"] in skip_type:
            return None
        elif pipelines["type"] == "MultiBranch":
            new_pipelines = {}
            for k, v in pipelines.items():
                if k != "type":
                    new_pipelines[k] = remove_pipe(v, skip_type)
                else:
                    new_pipelines[k] = v
            return new_pipelines
        else:
            return pipelines
    else:
        raise NotImplementedError()


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    train_data_cfg = cfg.data.train
    while "dataset" in train_data_cfg:
        train_data_cfg = train_data_cfg["dataset"]
    train_data_cfg["pipeline"] = remove_pipe(train_data_cfg["pipeline"], skip_type)
    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    for item in dataset:
        if not isinstance(item, list):
            item = [item]
        bboxes = []
        labels = []
        tran_mats = []
        out_shapes = []
        for it in item:
            trans_matrix = it["transform_matrix"]
            bbox = it["gt_bboxes"]
            tran_mats.append(trans_matrix)
            bboxes.append(bbox)
            labels.append(it["gt_labels"])
            out_shapes.append(it["img_shape"])

            filename = (
                os.path.join(args.output_dir, Path(it["filename"]).name)
                if args.output_dir is not None
                else None
            )

            gt_masks = it.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = mask2ndarray(gt_masks)

            imshow_det_bboxes(
                it["img"],
                it["gt_bboxes"],
                it["gt_labels"],
                gt_masks,
                class_names=dataset.CLASSES,
                show=not args.not_show,
                wait_time=args.show_interval,
                out_file=filename,
                bbox_color=(255, 102, 61),
                text_color=(255, 102, 61),
            )

        if len(tran_mats) == 2:
            # check equality between different augmentation
            transed_bboxes = Transform2D.transform_bboxes(
                torch.from_numpy(bboxes[1]).float(),
                torch.from_numpy(tran_mats[0]).float()
                @ torch.from_numpy(tran_mats[1]).float().inverse(),
                out_shapes[0],
            )
            img = imshow_det_bboxes(
                item[0]["img"],
                item[0]["gt_bboxes"],
                item[0]["gt_labels"],
                class_names=dataset.CLASSES,
                show=False,
                wait_time=args.show_interval,
                out_file=None,
                bbox_color=(255, 102, 61),
                text_color=(255, 102, 61),
            )
            imshow_det_bboxes(
                img,
                transed_bboxes.numpy(),
                labels[1],
                class_names=dataset.CLASSES,
                show=True,
                wait_time=args.show_interval,
                out_file=None,
                bbox_color=(0, 0, 255),
                text_color=(0, 0, 255),
                thickness=5,
            )

        progress_bar.update()


if __name__ == "__main__":
    main()

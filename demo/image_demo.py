# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import asyncio
import glob
import os
from argparse import ArgumentParser

from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot

from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="bbox score threshold"
    )
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    imgs = glob.glob(args.img)
    for img in imgs:
        # test a single image
        result = inference_detector(model, img)
        # show the results
        if args.output is None:
            show_result_pyplot(model, img, result, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, result, score_thr=args.score_thr, out_file=out_file_path
            )


async def async_main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    # test a single image
    args.img = glob.glob(args.img)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    for img, pred in zip(args.img, result):
        if args.output is None:
            show_result_pyplot(model, img, pred, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, pred, score_thr=args.score_thr, out_file=out_file_path
            )


if __name__ == "__main__":
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)

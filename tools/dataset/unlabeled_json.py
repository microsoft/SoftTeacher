"""Generate unlabeled coco dataset json annotations from a folder of images.
Uses imagesize for significant speedup over reading images into memory.

Example:
python tools/unlabeled_json.py --img-dir <img/path/> --json-out <json/save/path.json>
"""

import argparse
import glob
import imagesize
import json


def folder_to_json(img_dir, json_out_path):

    ext = ('*.jpg', '*.jpeg', '*.png')
    paths = [p for paths in [glob.glob(img_dir + e) for e in ext]
        for p in paths]
    assert len(paths) > 0

    images = []
    for i, p in enumerate(paths):
        w, h = imagesize.get(p)
        name = p.split('/')[-1]

        per_image_dict = dict(
            id=i,
            file_name=name,
            width=w,
            height=h
            )

        images.append(per_image_dict)

    data = dict(categories=[])
    data['images'] = images
    with open(json_out_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    folder_to_json(args.img_dir, args.json_out)

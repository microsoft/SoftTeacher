_base_ = "base.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint="open-mmlab://detectron2/resnet101_caffe"),
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        sup=dict(
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            ann_file="data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="data/coco/unlabeled2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 6, 160000 * 6])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 6)

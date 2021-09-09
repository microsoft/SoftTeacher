_base_ = "base.py"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
    ),
)

optimizer = dict(lr=0.02)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

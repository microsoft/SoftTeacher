_base_ = "soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint="open-mmlab://detectron2/resnet101_caffe"),
    )
)

lr_config = dict(step=[120000 * 6, 160000 * 6])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 6)

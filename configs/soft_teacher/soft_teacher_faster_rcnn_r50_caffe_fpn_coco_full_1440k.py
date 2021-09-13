_base_ = "soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py"


lr_config = dict(step=[120000 * 8, 160000 * 8])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 8)

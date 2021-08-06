# Self-Supervised Learning with Swin Transformers

This repo is the official implementation of ["End-to-End Semi-Supervised Object Detection with Soft Teacher"](https://arxiv.org/abs/2106.09018).

**Code and models will be released soon.**

## Introduction

This paper presents an end-to-end semi-supervised object detection approach, in contrast to previous more complex multi-stage methods. The end-to-end training gradually improves pseudo label qualities during the curriculum, and the more and more accurate pseudo labels in turn benefit object detection training. We also propose two simple yet effective techniques within this framework: a soft teacher mechanism where the classification loss of each unlabeled bounding box is weighed by the classification score produced by the teacher network; a box jittering approach to select reliable pseudo boxes for the learning of box regression. On COCO benchmark, the proposed approach outperforms previous methods by a large margin under various labelling ratios, i.e. 1\%, 5\% and 10\%. Moreover, our approach proves to perform also well when the amount of labeled data is relatively large. For example, it can improve a 40.9 mAP baseline detector trained using the full COCO training set by +3.6 mAP, reaching 44.5 mAP, by leveraging the 123K unlabeled images of COCO. On the state-of-the-art Swin Transformer based object detector (58.9 mAP on test-dev), it can still significantly improve the detection accuracy by +1.5 mAP, reaching 60.4 mAP, and improve the instance segmentation accuracy by +1.2 mAP, reaching 52.4 mAP. Further incorporating with the Object365 pre-trained model, the detection accuracy reaches 61.3 mAP and the instance segmentation accuracy reaches 53.0 mAP, pushing the new state-of-the-art. 

In this repository, we provide model implementation (with Pytorch) as well as data preparation, training and evaluation
scripts on MS-COCO.

## Citation

```
@article{xu2021end,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={arXiv preprint arXiv:2106.09018},
  year={2021}
}
```

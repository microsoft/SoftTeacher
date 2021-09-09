import copy
import json

from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class PseudoCocoDataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pseudo_ann_file,
        pipeline,
        confidence_threshold=0.9,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
    ):
        self.confidence_threshold = confidence_threshold
        self.pseudo_ann_file = pseudo_ann_file

        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )

    def load_pesudo_targets(self, pseudo_ann_file):
        with open(pseudo_ann_file) as f:
            pesudo_anns = json.load(f)
        print(f"loading {len(pesudo_anns)} results")

        def _add_attr(dict_terms, **kwargs):
            new_dict = copy.copy(dict_terms)
            new_dict.update(**kwargs)
            return new_dict

        def _compute_area(bbox):
            _, _, w, h = bbox
            return w * h

        pesudo_anns = [
            _add_attr(ann, id=i, area=_compute_area(ann["bbox"]))
            for i, ann in enumerate(pesudo_anns)
            if ann["score"] > self.confidence_threshold
        ]
        print(
            f"With {len(pesudo_anns)} results over threshold {self.confidence_threshold}"
        )

        return pesudo_anns

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        pesudo_anns = self.load_pesudo_targets(self.pseudo_ann_file)
        self.coco = COCO(ann_file)
        self.coco.dataset["annotations"] = pesudo_anns
        self.coco.createIndex()

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)

        return data_infos

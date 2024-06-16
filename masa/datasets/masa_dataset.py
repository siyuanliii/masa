# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
import pickle
from typing import List, Union

import h5py
import tqdm
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS
from mmengine.fileio import get_local_path
from mmengine.logging import print_log


@DATASETS.register_module()
class MASADataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        "classes": ("object"),
        # palette is a list of color tuples, which is used for visualization.
        "palette": [(220, 20, 60)],
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self, anno_hdf5_path=None, img_prefix=None, *args, **kwargs):

        self.anno_hdf5_path = anno_hdf5_path
        self.img_prefix = img_prefix
        super().__init__(*args, **kwargs)

    def read_dicts_from_hdf5(self, hdf5_file_path, pkl_file_path):
        with h5py.File(hdf5_file_path, "r") as hf:
            # Retrieve the dataset corresponding to the specified .pkl file path
            dataset = hf[pkl_file_path]
            binary_data = dataset[()]
        # Deserialize the binary data and load the list of dictionaries
        list_of_dicts = pickle.loads(binary_data)
        return list_of_dicts

    def get_ann_info(self, img_info):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        if self.anno_hdf5_path is not None:
            try:
                ann_info = self.read_dicts_from_hdf5(
                    self.anno_hdf5_path, img_info["file_name"].replace(".jpg", ".pkl")
                )
                return ann_info
            except:
                print(self.anno_hdf5_path)
                print(img_info["file_name"].replace(".jpg", ".pkl"))
                return None
        else:
            img_id = img_info["id"]
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
            ann_info = self.coco.load_anns(ann_ids)
            return ann_info

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                "Please call `full_init()` method manually to accelerate " "the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception(
                    "Test time pipline should not get `None` " "data_sample"
                )
            return data

        for _ in range(self.max_refetch + 1):
            try:
                data = self.prepare_data(idx)
            except Exception as e:
                data = None
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(
            f"Cannot find valid image after {self.max_refetch}! "
            "Please check your image path and pipeline"
        )

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo["classes"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        print("Loading data list...")
        for img_id in tqdm.tqdm(img_ids):
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)

            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix["img"], img_info["file_name"])
        if self.data_prefix.get("seg", None):
            seg_map_path = osp.join(
                self.data_prefix["seg"],
                img_info["file_name"].rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None
        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["img_id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        if self.return_classes:
            data_info["text"] = self.metainfo["classes"]
            data_info["caption_prompt"] = self.caption_prompt
            data_info["custom_entities"] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if "category_id" not in ann:
                ann["category_id"] = 1
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[ann["category_id"]]

            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]

            if "instance_id" in ann:
                instance["instance_id"] = ann["instance_id"]
            else:
                instance["instance_id"] = ann["id"]

            instances.append(instance)
        data_info["instances"] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get("filter_empty_gt", False)
        min_size = self.filter_cfg.get("min_size", 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info["img_id"] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info["img_id"]
            width = data_info["width"]
            height = data_info["height"]
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

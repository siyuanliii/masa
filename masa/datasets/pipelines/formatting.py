from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData


@TRANSFORMS.register_module(force=True)
class PackMatchInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation. All the information of images are packed to ``inputs``. All
    the information except images are packed to ``data_samples``. In order to
    get the original annotaiton and meta info, we add `instances` key into meta
    keys.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "gt_instances_ids": "instances_ids",
    }

    def __init__(
        self,
        meta_keys: Optional[dict] = None,
        default_meta_keys: tuple = (
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "frame_id",
            "video_id",
            "video_length",
            "ori_video_length",
            "instances",
        ),
    ):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys,)
            else:
                assert isinstance(meta_keys, tuple), "meta_keys must be str or tuple"
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results["inputs"] = dict()

        # 1. Pack images
        if "img" in results:
            imgs = results["img"]
            imgs = np.stack(imgs, axis=0)
            # imgs = imgs.transpose(0, 3, 1, 2)
            if not imgs.flags.c_contiguous:
                imgs = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))
                imgs = to_tensor(imgs)
            else:
                imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
            packed_results["inputs"] = imgs

        # 2. Pack InstanceData
        if "gt_ignore_flags" in results:
            gt_ignore_flags_list = results["gt_ignore_flags"]
            valid_idx_list, ignore_idx_list = [], []
            for gt_ignore_flags in gt_ignore_flags_list:
                valid_idx = np.where(gt_ignore_flags == 0)[0]
                ignore_idx = np.where(gt_ignore_flags == 1)[0]
                valid_idx_list.append(valid_idx)
                ignore_idx_list.append(ignore_idx)

        assert "img_id" in results, "'img_id' must contained in the results "
        "for counting the number of images"

        num_imgs = len(results["img_id"])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]
        ignore_instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == "gt_masks":
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                if "gt_ignore_flags" in results:
                    for i, gt_mask in enumerate(gt_masks_list):
                        valid_idx, ignore_idx = valid_idx_list[i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_mask[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_mask[ignore_idx]

                else:
                    for i, gt_mask in enumerate(gt_masks_list):
                        instance_data_list[i][mapped_key] = gt_mask

            elif isinstance(results[key][0], BaseBoxes):
                mapped_key = self.mapping_table[key]
                gt_bboxes_list = results[key]
                if "gt_ignore_flags" in results:
                    for i, gt_bbox in enumerate(gt_bboxes_list):
                        gt_bbox = gt_bbox.tensor
                        valid_idx, ignore_idx = valid_idx_list[i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_bbox[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_bbox[ignore_idx]

            else:
                anns_list = results[key]
                if "gt_ignore_flags" in results:
                    for i, ann in enumerate(anns_list):
                        valid_idx, ignore_idx = valid_idx_list[i], ignore_idx_list[i]
                        instance_data_list[i][self.mapping_table[key]] = to_tensor(
                            ann[valid_idx]
                        )
                        ignore_instance_data_list[i][
                            self.mapping_table[key]
                        ] = to_tensor(ann[ignore_idx])
                else:
                    for i, ann in enumerate(anns_list):
                        instance_data_list[i][self.mapping_table[key]] = to_tensor(ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_sample.ignored_instances = ignore_instance_data_list[i]
            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f"{key}": img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        if "key_frame_flags" in results:
            key_frame_flags = np.asarray(results["key_frame_flags"])
            key_frames_inds = np.where(key_frame_flags)[0].tolist()
            ref_frames_inds = np.where(~key_frame_flags)[0].tolist()
            track_data_sample.set_metainfo(dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(dict(ref_frames_inds=ref_frames_inds))

        packed_results["data_samples"] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"meta_keys={self.meta_keys}, "
        repr_str += f"default_meta_keys={self.default_meta_keys})"
        return repr_str

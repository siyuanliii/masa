# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import log_img_scale
from mmengine.dataset import BaseDataset
from numpy import random

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]


def _fixed_scale_size(
    size: Tuple[int, int], scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don't need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(
    old_size: tuple, scale: Union[float, int, tuple], return_scale: bool = False
) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f"Scale must be a number or tuple of int, but got {type(scale)}"
        )
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


@TRANSFORMS.register_module(force=True)
class SeqMosaic(BaseTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_instances_ids (options, only used in MOT/VIS)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        prob: float = 1.0,
    ) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, (
            "The probability should be in range [0,1]. " f"got {prob}."
        )

        log_img_scale(img_scale, skip_square=True, shape_order="wh")
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert "mosaic_mix_results" in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_instances_ids = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mosaic_mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i, self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_bboxes_labels_i = results_patch["gt_bboxes_labels"]
            gt_ignore_flags_i = results_patch["gt_ignore_flags"]
            gt_instances_ids_i = results_patch.get("gt_instances_ids", None)

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            mosaic_instances_ids.append(gt_instances_ids_i)

        if len(mosaic_bboxes_labels) > 0:
            mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
            mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
            mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)
            mosaic_instances_ids = np.concatenate(mosaic_instances_ids, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]
        ).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
        mosaic_instances_ids = mosaic_instances_ids[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape[:2]
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_bboxes_labels"] = mosaic_bboxes_labels
        results["gt_ignore_flags"] = mosaic_ignore_flags
        results["gt_instances_ids"] = mosaic_instances_ids

        return results

    def _mosaic_combine(
        self, loc: str, center_position_xy: Sequence[float], img_shape_wh: Sequence[int]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                img_shape_wh[1] - (y2 - y1),
                img_shape_wh[0],
                img_shape_wh[1],
            )

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                center_position_xy[1],
            )
            crop_coord = (
                0,
                img_shape_wh[1] - (y2 - y1),
                min(img_shape_wh[0], x2 - x1),
                img_shape_wh[1],
            )

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                0,
                img_shape_wh[0],
                min(y2 - y1, img_shape_wh[1]),
            )

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                0,
                0,
                min(img_shape_wh[0], x2 - x1),
                min(y2 - y1, img_shape_wh[1]),
            )

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob})"
        return repr_str


@TRANSFORMS.register_module(force=True)
class SeqMixUp(BaseTransform):
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
    ) -> None:
        assert isinstance(img_scale, tuple)
        log_img_scale(img_scale, skip_square=True, shape_order="wh")
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            gt_bboxes_i = dataset[index]["gt_bboxes"]
            if len(gt_bboxes_i) != 0:
                break

        return index

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mixup_mix_results" in results
        assert (
            len(results["mixup_mix_results"]) == 1
        ), "MixUp only support 2 images now !"

        if results["mixup_mix_results"][0]["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = copy.deepcopy(results["mixup_mix_results"][0])
        retrieve_img = retrieve_results["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones(
                    (self.dynamic_scale[1], self.dynamic_scale[0], 3),
                    dtype=retrieve_img.dtype,
                )
                * self.pad_val
            )
        else:
            out_img = (
                np.ones(self.dynamic_scale[::-1], dtype=retrieve_img.dtype)
                * self.pad_val
            )

        # 1. keep_ratio resize
        scale_ratio = min(
            self.dynamic_scale[1] / retrieve_img.shape[0],
            self.dynamic_scale[0] / retrieve_img.shape[1],
        )
        retrieve_img = mmcv.imresize(
            retrieve_img,
            (
                int(retrieve_img.shape[1] * scale_ratio),
                int(retrieve_img.shape[0] * scale_ratio),
            ),
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
        )

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = (
            np.ones((max(origin_h, target_h), max(origin_w, target_w), 3))
            * self.pad_val
        )
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes.flip_([origin_h, origin_w], direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results["gt_bboxes_labels"]
        retrieve_gt_ignore_flags = retrieve_results["gt_ignore_flags"]
        retrieve_gt_instances_ids = retrieve_results["gt_instances_ids"]

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results["gt_bboxes"], cp_retrieve_gt_bboxes), dim=0
        )
        mixup_gt_bboxes_labels = np.concatenate(
            (results["gt_bboxes_labels"], retrieve_gt_bboxes_labels), axis=0
        )
        mixup_gt_ignore_flags = np.concatenate(
            (results["gt_ignore_flags"], retrieve_gt_ignore_flags), axis=0
        )
        mixup_gt_instances_ids = np.concatenate(
            (results["gt_instances_ids"], retrieve_gt_instances_ids), axis=0
        )

        # remove outside bbox
        inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]
        mixup_gt_instances_ids = mixup_gt_instances_ids[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape[:2]
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_bboxes_labels"] = mixup_gt_bboxes_labels
        results["gt_ignore_flags"] = mixup_gt_ignore_flags
        results["gt_instances_ids"] = mixup_gt_instances_ids

        assert len(results["gt_bboxes"]) == len(results["gt_instances_ids"])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


@TRANSFORMS.register_module(force=True)
class FilterMatchAnnotations(BaseTransform):
    """Filter invalid annotations.

    Required Keys:

    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground truth
            boxes. Default: (1., 1.)
        min_gt_mask_area (int): Minimum foreground area of ground truth masks.
            Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: True
        by_mask (bool): Filter instances with masks not meeting
            min_gt_mask_area threshold. Default: False
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(
        self,
        min_gt_bbox_wh: Tuple[int, int] = (1, 1),
        min_gt_mask_area: int = 1,
        by_box: bool = True,
        by_mask: bool = False,
        keep_empty: bool = True,
    ) -> None:
        # TODO: add more filter options
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert "gt_bboxes" in results
        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results

        tests = []
        if self.by_box:
            tests.append(
                (
                    (gt_bboxes.widths > self.min_gt_bbox_wh[0])
                    & (gt_bboxes.heights > self.min_gt_bbox_wh[1])
                ).numpy()
            )
        if self.by_mask:
            assert "gt_masks" in results
            gt_masks = results["gt_masks"]
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = (
            "gt_bboxes",
            "gt_bboxes_labels",
            "gt_masks",
            "gt_instances_ids",
            "gt_ignore_flags",
        )
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        return results

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(min_gt_bbox_wh={self.min_gt_bbox_wh}, "
            f"keep_empty={self.keep_empty})"
        )


@TRANSFORMS.register_module(force=True)
class SeqCopyPaste(BaseTransform):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Defaults to True.
        paste_by_box (bool): Whether use boxes as masks when masks are not
            available.
            Defaults to False.
    """

    def __init__(
        self,
        max_num_pasted: int = 100,
        bbox_occluded_thr: int = 10,
        mask_occluded_thr: int = 300,
        selected: bool = True,
        paste_by_box: bool = False,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = paste_by_box

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert "copypaste_mix_results" in results
        num_images = len(results["copypaste_mix_results"])
        assert (
            num_images == 1
        ), f"CopyPaste only supports processing 2 images, got {num_images}"
        if self.selected:
            selected_results = copy.deepcopy(
                self._select_object(results["copypaste_mix_results"][0])
            )
        else:
            selected_results = copy.deepcopy(results["copypaste_mix_results"][0])

        return self._copy_paste(results, selected_results)

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get("gt_masks", None) is not None:
            if self.paste_by_box:
                warnings.warn(
                    "gt_masks is already contained in results, "
                    "so paste_by_box is disabled."
                )
            return results["gt_masks"]
        else:
            if not self.paste_by_box:
                raise RuntimeError("results does not contain masks.")
            return results["gt_bboxes"].create_masks(results["img"].shape[:2])

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results["gt_bboxes"]
        labels = results["gt_bboxes_labels"]
        masks = self.get_gt_masks(results)
        ignore_flags = results["gt_ignore_flags"]
        gt_instances_ids = results.get("gt_instances_ids", None)

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]
        selected_gt_instances_ids = gt_instances_ids[selected_inds]

        results["gt_bboxes"] = selected_bboxes
        results["gt_bboxes_labels"] = selected_labels
        results["gt_masks"] = selected_masks
        results["gt_ignore_flags"] = selected_ignore_flags
        results["gt_instances_ids"] = selected_gt_instances_ids
        return results

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results["img"]
        dst_bboxes = dst_results["gt_bboxes"]
        dst_labels = dst_results["gt_bboxes_labels"]
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results["gt_ignore_flags"]
        dst_instances_ids = dst_results.get("gt_instances_ids", None)

        src_img = src_results["img"]
        src_bboxes = src_results["gt_bboxes"]
        src_labels = src_results["gt_bboxes_labels"]
        src_masks = src_results["gt_masks"]
        src_ignore_flags = src_results["gt_ignore_flags"]
        src_instances_ids = src_results.get("gt_instances_ids", None)

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = (
            dst_img * (1 - composed_mask[..., np.newaxis])
            + src_img * composed_mask[..., np.newaxis]
        )
        bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])
        instances_ids = np.concatenate(
            [dst_instances_ids[valid_inds], src_instances_ids]
        )

        dst_results["img"] = img
        dst_results["gt_bboxes"] = bboxes
        dst_results["gt_bboxes_labels"] = labels
        dst_results["gt_masks"] = BitmapMasks(masks, masks.shape[1], masks.shape[2])
        dst_results["gt_ignore_flags"] = ignore_flags
        dst_results["gt_instances_ids"] = instances_ids

        return dst_results

    def _get_updated_masks(
        self, masks: BitmapMasks, composed_mask: np.ndarray
    ) -> BitmapMasks:
        """Update masks with composed mask."""
        assert (
            masks.masks.shape[-2:] == composed_mask.shape[-2:]
        ), "Cannot compare two arrays of different size"
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_num_pasted={self.max_num_pasted}, "
        repr_str += f"bbox_occluded_thr={self.bbox_occluded_thr}, "
        repr_str += f"mask_occluded_thr={self.mask_occluded_thr}, "
        repr_str += f"selected={self.selected}), "
        repr_str += f"paste_by_box={self.paste_by_box})"
        return repr_str


@TRANSFORMS.register_module(force=True)
class SeqRandomAffine(BaseTransform):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: Tuple[int, int] = (0, 0),
        border_val: Tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
    ) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        return warp_matrix

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results["img"]
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]

        bboxes = results["gt_bboxes"]
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results["gt_bboxes"] = bboxes[valid_index]
            results["gt_bboxes_labels"] = results["gt_bboxes_labels"][valid_index]
            results["gt_ignore_flags"] = results["gt_ignore_flags"][valid_index]
            results["gt_instances_ids"] = results["gt_instances_ids"][valid_index]
            assert len(results["gt_bboxes"]) == len(results["gt_instances_ids"])
            if "gt_masks" in results:
                raise NotImplementedError("RandomAffine only supports bbox.")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_rotate_degree={self.max_rotate_degree}, "
        repr_str += f"max_translate_ratio={self.max_translate_ratio}, "
        repr_str += f"scaling_ratio_range={self.scaling_ratio_range}, "
        repr_str += f"max_shear_degree={self.max_shear_degree}, "
        repr_str += f"border={self.border}, "
        repr_str += f"border_val={self.border_val}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian), 0.0],
                [np.sin(radian), np.cos(radian), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array(
            [[1, np.tan(x_radian), 0.0], [np.tan(y_radian), 1, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array(
            [[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return translation_matrix

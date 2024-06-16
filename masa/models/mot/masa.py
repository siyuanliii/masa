"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

import copy
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.models.mot.base import BaseMOTModel
from mmdet.registry import MODELS
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData
from torch import Tensor


@MODELS.register_module()
class MASA(BaseMOTModel):

    """Matching Anything By Segmenting Anything.

    This multi object tracker is the implementation of `MASA
    https://arxiv.org/abs/2406.04221`.

    Args:
        backbone (dict, optional): Configuration of backbone. Defaults to None.
        detector (dict, optional): Configuration of detector. Defaults to None.
        masa_adapter (dict, optional): Configuration of MASA adapter. Defaults to None.
        rpn_head (dict, optional): Configuration of RPN head. Defaults to None.
        roi_head (dict, optional): Configuration of RoI head. Defaults to None.
        track_head (dict, optional): Configuration of track head. Defaults to None.
        tracker (dict, optional): Configuration of tracker. Defaults to None.
        freeze_detector (bool): If True, freeze the detector weights. Defaults to False.
        freeze_masa_backbone (bool): If True, freeze the MASA backbone weights. Defaults to False.
        freeze_masa_adapter (bool): If True, freeze the MASA adapter weights. Defaults to False.
        freeze_object_prior_distillation (bool): If True, freeze the object prior distillation. Defaults to False.
        data_preprocessor (dict or ConfigDict, optional): The pre-process config of :class:`TrackDataPreprocessor`.
            It usually includes, ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``. Defaults to None.
        train_cfg (dict or ConfigDict, optional): Training configuration. Defaults to None.
        test_cfg (dict or ConfigDict, optional): Testing configuration. Defaults to None.
        init_cfg (dict or list[dict], optional): Configuration of initialization. Defaults to None.
        load_public_dets (bool): If True, load public detections. Defaults to False.
        public_det_path (str, optional): Path to public detections. Required if load_public_dets is True. Defaults to None.
        given_dets (bool): If True, detections are given. Defaults to False.
        with_segm (bool): If True, segmentation masks are included. Defaults to False.
        end_pkl_name (str): Suffix for pickle file names. Defaults to '.pth'.
        unified_backbone (bool): If True, use a unified backbone. Defaults to False.
        use_masa_backbone (bool): If True, use the MASA backbone. Defaults to False.
        benchmark (str): Benchmark for evaluation. Defaults to 'tao'.
    """

    def __init__(
        self,
        backbone: Optional[dict] = None,
        detector: Optional[dict] = None,
        masa_adapter: Optional[dict] = None,
        rpn_head: Optional[dict] = None,
        roi_head: Optional[dict] = None,
        track_head: Optional[dict] = None,
        tracker: Optional[dict] = None,
        freeze_detector: bool = False,
        freeze_masa_backbone: bool = False,
        freeze_masa_adapter: bool = False,
        freeze_object_prior_distillation: bool = False,
        data_preprocessor: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        load_public_dets=False,
        public_det_path=None,
        given_dets=False,
        with_segm=False,
        end_pkl_name=".pth",
        unified_backbone=False,
        use_masa_backbone=False,
        benchmark="tao",
    ) -> None:
        super().__init__(data_preprocessor, init_cfg)

        self.use_masa_backbone = use_masa_backbone
        if use_masa_backbone:
            assert (
                backbone is not None
            ), "backbone must be set when using MASA backbone."

        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if masa_adapter is not None:
            self.masa_adapter = MODELS.build(masa_adapter)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get("num_classes", None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        "The `num_classes` should be 1 in RPN, but get "
                        f"{rpn_head_num_classes}, please set "
                        "rpn_head.num_classes = 1 in your config file."
                    )
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.freeze_detector = freeze_detector
        self.freeze_masa_adapter = freeze_masa_adapter
        self.freeze_object_prior_distillation = freeze_object_prior_distillation
        self.freeze_masa_backbone = freeze_masa_backbone

        def set_to_eval(module, input):
            module.eval()

        if self.freeze_detector:
            assert (
                detector is not None
            ), "detector must be set when freeze_detector is True."
            self.freeze_module("detector")
            # self.detector.backbone.register_forward_pre_hook(set_to_eval)

        if self.freeze_masa_adapter:
            assert (
                masa_adapter is not None
            ), "masa_adapter must be set when freeze_masa_adapter is True."
            self.freeze_module("masa_adapter")

            self.masa_adapter.register_forward_pre_hook(set_to_eval)

        if self.freeze_object_prior_distillation:
            assert (
                roi_head is not None
            ), "roi_head must be set when freeze_object_prior_distillation is True."
            assert (
                rpn_head is not None
            ), "rpn_head must be set when freeze_object_prior_distillation is True."
            self.freeze_module("roi_head")
            self.freeze_module("rpn_head")

        if self.freeze_masa_backbone:
            assert (
                backbone is not None
            ), "backbone must be set when freeze_masa_backbone is True."
            self.freeze_module("backbone")
            self.backbone.register_forward_pre_hook(set_to_eval)

        if load_public_dets:
            assert (
                public_det_path is not None
            ), "load_public_dets and public_det_path must be set together."
            self.benchmark = benchmark
        self.load_public_dets = load_public_dets
        self.public_det_path = public_det_path
        self.with_segm = with_segm
        self.end_pkl_name = end_pkl_name
        self.given_dets = given_dets

        self.unified_backbone = unified_backbone

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, "roi_head") and self.roi_head is not None

    def predict(
        self,
        inputs: Tensor,
        data_samples: TrackSampleList,
        rescale: bool = True,
        **kwargs,
    ) -> TrackSampleList:
        """Predict results from a video and data samples with post- processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            TrackSampleList: Tracking results of the inputs.
        """
        assert inputs.dim() == 5, "The img must be 5D Tensor (N, T, C, H, W)."
        assert (
            inputs.size(0) == 1
        ), "MASA inference only support 1 batch size per gpu for now."

        assert len(data_samples) == 1, "MASA only support 1 batch size per gpu for now."

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)
        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            if self.load_public_dets:
                img_name = img_data_sample.img_path
                if img_name is not None:
                    if self.benchmark == "bdd":
                        pickle_name = img_name.replace(
                            "data/bdd/bdd100k/images/track/val/", ""
                        ).replace(".jpg", self.end_pkl_name)
                    elif self.benchmark == "tao":
                        pickle_name = img_name.replace("data/tao/frames/", "").replace(
                            ".jpg", self.end_pkl_name
                        )

                path = os.path.join(self.public_det_path, pickle_name)
                pickle_res = pickle.load(open(path, "rb"))
                det_labels = torch.tensor(pickle_res["det_labels"]).to("cuda")
                det_bboxes = (
                    torch.tensor(pickle_res["det_bboxes"]).to("cuda").to(torch.float32)
                )
                if len(det_bboxes) != 0:
                    if det_bboxes.size(1) == 4:
                        det_bboxes = torch.cat(
                            [
                                det_bboxes,
                                torch.ones(det_bboxes.size(0), 1).to(det_bboxes.device),
                            ],
                            dim=1,
                        )

                det_results = InstanceData()
                det_results.labels = det_labels
                det_results.bboxes = det_bboxes[:, :4]
                det_results.scores = det_bboxes[:, 4]

                if self.with_segm:
                    segm_results = pickle_res["det_masks"]
                    det_results.masks = segm_results

                img_data_sample.pred_instances = det_results

                if self.unified_backbone:
                    if hasattr(self.detector.backbone, "with_text_model"):
                        x = self.detector.backbone.forward_image(single_img)
                    elif self.detector.__class__.__name__ == "SamMasa":
                        x = self.detector.backbone.forward_base_multi_level(single_img)
                    else:
                        x = self.detector.backbone(single_img)
                elif self.use_masa_backbone:
                    x = self.backbone.forward(single_img)
                x_m = self.masa_adapter(x)

            elif self.given_dets:
                assert (
                    "det_bboxes" in img_data_sample
                ), "det_bboxes must be given when given_dets is True."
                assert (
                    "det_labels" in img_data_sample
                ), "det_labels must be given when given_dets is True."
                det_labels = img_data_sample.det_labels
                det_bboxes = img_data_sample.det_bboxes
                if len(det_bboxes) != 0:
                    if det_bboxes.size(1) == 4:
                        det_bboxes = torch.cat(
                            [
                                det_bboxes,
                                torch.ones(det_bboxes.size(0), 1).to(det_bboxes.device),
                            ],
                            dim=1,
                        )
                det_results = InstanceData()
                det_results.labels = det_labels
                det_results.bboxes = det_bboxes[:, :4]
                det_results.scores = det_bboxes[:, 4]

                img_data_sample.pred_instances = det_results

                if self.unified_backbone:
                    if hasattr(self.detector.backbone, "with_text_model"):
                        x = self.detector.backbone.forward_image(single_img)
                    elif self.detector.__class__.__name__ == "SamMasa":
                        x = self.detector.backbone.forward_base_multi_level(single_img)
                    else:
                        x = self.detector.backbone(single_img)
                elif self.use_masa_backbone:
                    x = self.backbone.forward(single_img)
                x_m = self.masa_adapter(x)
            else:
                if self.unified_backbone:
                    if hasattr(self.detector.backbone, "with_text_model"):
                        texts = img_data_sample.texts
                        ## fix some inconsistency caused by the implementation of yolo-world and mmdet
                        if type(texts[0]) == list:
                            new_texts = [text[0] for text in texts]
                            del img_data_sample.texts
                            img_data_sample.set_field(
                                new_texts, "texts", field_type="metainfo"
                            )
                        (
                            backbone_feats,
                            img_feats,
                            text_feats,
                        ) = self.detector.extract_feat(single_img, [img_data_sample])
                        x_m = self.masa_adapter(backbone_feats)
                        img_data_sample = self.detector.predict(
                            single_img,
                            (img_feats, text_feats),
                            [img_data_sample],
                            rescale=rescale,
                        )[0]
                    else:
                        x = self.detector.backbone(single_img)
                        x_m = self.masa_adapter(x)
                        if self.detector.with_neck:
                            x = self.detector.neck(x)

                        img_data_sample = self.detector.predict(
                            single_img, x, [img_data_sample], rescale=rescale
                        )[0]
                else:
                    raise NotImplementedError

            frame_pred_track_instances = self.tracker.track(
                model=self,
                img=single_img,
                feats=x_m,
                data_sample=img_data_sample,
                with_segm=self.with_segm,
                **kwargs,
            )
            if self.with_segm:
                if frame_pred_track_instances.mask_inds is not None:
                    frame_pred_track_instances.masks = [
                        img_data_sample.pred_instances.masks[i]
                        for i in frame_pred_track_instances.mask_inds
                    ]

            img_data_sample.pred_track_instances = frame_pred_track_instances

        return [track_data_sample]

    def parse_tensors(self, tensor_tuple, key_ids, ref_ids):
        key_tensors = []
        ref_tensors = []
        device = tensor_tuple[0].device
        for tensor in tensor_tuple:
            key_tensors.append(
                tensor.index_select(
                    0, torch.tensor(key_ids, dtype=torch.long, device=device)
                )
            )
            ref_tensors.append(
                tensor.index_select(
                    0, torch.tensor(ref_ids, dtype=torch.long, device=device)
                )
            )

        return list(key_tensors), list(ref_tensors)

    def loss(
        self, inputs: Tensor, data_samples: TrackSampleList, **kwargs
    ) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                frames.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        assert inputs.dim() == 5, "The img must be 5D Tensor (N, T, C, H, W)."
        assert (
            inputs.size(1) == 2
        ), "MASA can only have 1 key frame and 1 reference frame."
        if self.detector is not None:
            self.detector.eval()
        # split the data_samples into two aspects: key frames and reference
        # frames
        ref_data_samples, key_data_samples = [], []
        key_frame_inds, ref_frame_inds = [], []
        # set cat_id of gt_labels to 0 in RPN
        for track_data_sample in data_samples:
            key_frame_inds.append(track_data_sample.key_frames_inds[0])
            ref_frame_inds.append(track_data_sample.ref_frames_inds[0])
            key_data_sample = track_data_sample.get_key_frames()[0]
            key_data_sample.gt_instances.labels = torch.zeros_like(
                key_data_sample.gt_instances.labels
            )
            key_data_samples.append(key_data_sample)
            ref_data_sample = track_data_sample.get_ref_frames()[0]
            ref_data_samples.append(ref_data_sample)

        key_frame_inds = torch.tensor(key_frame_inds, dtype=torch.int64)
        ref_frame_inds = torch.tensor(ref_frame_inds, dtype=torch.int64)
        batch_inds = torch.arange(len(inputs))
        key_imgs = inputs[batch_inds, key_frame_inds].contiguous()
        ref_imgs = inputs[batch_inds, ref_frame_inds].contiguous()

        if self.use_masa_backbone:
            x = self.backbone.forward(key_imgs)
            ref_x = self.backbone.forward(ref_imgs)

        else:
            if hasattr(self.detector.backbone, "with_text_model"):
                x = self.detector.backbone.forward_image(key_imgs)
                ref_x = self.detector.backbone.forward_image(ref_imgs)
            elif self.detector.__class__.__name__ == "SamMasa":
                x = self.detector.backbone.forward_base_multi_level(key_imgs)
                ref_x = self.detector.backbone.forward_base_multi_level(ref_imgs)
            else:
                x = self.detector.backbone.forward(key_imgs)
                ref_x = self.detector.backbone.forward(ref_imgs)

        x_m = self.masa_adapter(x)
        ref_x_m = self.masa_adapter(ref_x)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            key_rpn_data_samples = copy.deepcopy(key_data_samples)
            ref_rpn_data_samples = copy.deepcopy(ref_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in key_rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels
                )
            for data_sample in ref_rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels
                )

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x_m, key_rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs
            )
            ref_rpn_results_list = self.rpn_head.predict(
                ref_x_m, ref_rpn_data_samples, **kwargs
            )

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if "loss" in key and "rpn" not in key:
                    rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            raise NotImplementedError("MASA  only support with_rpn for now.")

        # roi_head loss
        losses_detect = self.roi_head.loss(
            x_m, rpn_results_list, key_data_samples, **kwargs
        )
        losses.update(losses_detect)

        # tracking head loss
        losses_track = self.track_head.loss(
            x_m, ref_x_m, rpn_results_list, ref_rpn_results_list, data_samples, **kwargs
        )
        losses.update(losses_track)

        return losses

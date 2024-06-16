import copy
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.evaluation import get_classes
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import autocast, load_checkpoint

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def init_masa(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = "none",
    device: str = "cuda:0",
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a unified masa detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )

    with_backbone = config.model.get("backbone", False)
    if with_backbone:
        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        elif "init_cfg" in config.model.backbone:
            config.model.backbone.init_cfg = None
    else:
        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        elif "init_cfg" in config.model.detector.backbone:
            config.model.detector.backbone.init_cfg = None

    scope = config.get("default_scope", "mmdet")
    if scope is not None:
        init_default_scope(config.get("default_scope", "mmdet"))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter("once")
        warnings.warn("checkpoint is None, use COCO classes by default.")
        model.dataset_meta = {"classes": get_classes("coco")}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get("meta", {})

        # save the dataset_meta in the model for convenience
        if "dataset_meta" in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()
            }
        elif "CLASSES" in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta["CLASSES"]
            model.dataset_meta = {"classes": classes}
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "dataset_meta or class names are not saved in the "
                "checkpoint's meta data, use COCO classes by default."
            )
            model.dataset_meta = {"classes": get_classes("coco")}

    # Priority:  args.palette -> config -> checkpoint
    if palette != "none":
        model.dataset_meta["palette"] = palette
    else:
        if "palette" not in model.dataset_meta:
            warnings.warn(
                "palette does not exist, random is used by default. "
                "You can also set the palette to customize."
            )
            model.dataset_meta["palette"] = "random"

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
    fp16: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == "cpu":
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."

    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_["inputs"] = [data_["inputs"]]
        data_["data_samples"] = [data_["data_samples"]]

        # forward the model
        with torch.no_grad():
            with autocast(enabled=fp16):
                results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def inference_masa(
    model: nn.Module,
    img: np.ndarray,
    frame_id: int,
    video_len: int,
    test_pipeline: Optional[Compose] = None,
    text_prompt=None,
    custom_entities: bool = False,
    det_bboxes=None,
    det_labels=None,
    fp16=False,
    detector_type="mmdet",
    show_fps=False,
) -> SampleList:
    """Inference image(s) with the masa model.

    Args:
        model (nn.Module): The loaded mot model.
        img (np.ndarray): Loaded image.
        frame_id (int): frame id.
        video_len (int): demo video length
    Returns:
        SampleList: The tracking data samples.
    """
    data = dict(
        img=[img.astype(np.float32)],
        # img=[img.astype(np.uint8)],
        frame_id=[frame_id],
        ori_shape=[img.shape[:2]],
        img_id=[frame_id + 1],
        ori_video_length=[video_len],
    )

    if text_prompt is not None:
        if detector_type == "mmdet":
            data["text"] = [text_prompt]
            data["custom_entities"] = [custom_entities]
        elif detector_type == "yolo-world":
            data["texts"] = [text_prompt]
            data["custom_entities"] = [custom_entities]

    data = test_pipeline(data)

    # forward the model
    with torch.no_grad():
        data = default_collate([data])
        if det_bboxes is not None:
            data["data_samples"][0].video_data_samples[0].det_bboxes = det_bboxes
            data["data_samples"][0].video_data_samples[0].det_labels = det_labels
        # measure FPS ##
        if show_fps:
            start = time.time()
            with autocast(enabled=fp16):
                result = model.test_step(data)[0]
            end = time.time()
            fps = 1 / (end - start)
            return result, fps

        else:
            with autocast(enabled=fp16):
                result = model.test_step(data)[0]
            return result


def build_test_pipeline(
    cfg: ConfigType, with_text=False, detector_type="mmdet"
) -> ConfigType:
    """Build test_pipeline for mot/vis demo. In mot/vis infer, original
    test_pipeline should remove the "LoadImageFromFile" and
    "LoadTrackAnnotations".

    Args:
         cfg (ConfigDict): The loaded config.
    Returns:
         ConfigType: new test_pipeline
    """
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    transform_broadcaster = cfg.inference_pipeline[0].copy()
    if detector_type == "yolo-world":
        kept_transform = []
        for transform in transform_broadcaster["transforms"]:
            if (
                transform["type"] == "mmyolo.YOLOv5KeepRatioResize"
                or transform["type"] == "mmyolo.LetterResize"
            ):
                kept_transform.append(transform)
        transform_broadcaster["transforms"] = kept_transform
        pack_track_inputs = cfg.test_dataloader.dataset.pipeline[-1].copy()
        test_pipeline = Compose([transform_broadcaster, pack_track_inputs])
    else:
        for transform in transform_broadcaster["transforms"]:
            if "Resize" in transform["type"]:
                transform_broadcaster["transforms"] = transform
        pack_track_inputs = cfg.inference_pipeline[-1].copy()
        if with_text:
            pack_track_inputs["meta_keys"] = ("text", "custom_entities")
        test_pipeline = Compose([transform_broadcaster, pack_track_inputs])

    return test_pipeline

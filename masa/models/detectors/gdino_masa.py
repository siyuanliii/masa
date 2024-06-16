"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

import copy
import logging
import re
import warnings

from mmdet.registry import MODELS
from mmengine.logging import MMLogger, print_log
from mmengine.model.weight_init import (PretrainedInit, initialize,
                                        update_init_info)

from .grounding_dino import GroundingDINO


def clean_label_name(name: str) -> str:
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i : i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert counter == len(lst)

    return all_


@MODELS.register_module()
class GroundingDINOMasa(GroundingDINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.track_text_prompt = None
        self.track_text_dict = None
        self.token_positive_maps = None
        self.track_entities = None

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        if self.init_cfg:
            print_log(
                f"initialize {self.__class__.__name__} with init_cfg {self.init_cfg}",
                logger="current",
                level=logging.DEBUG,
            )

            init_cfgs = self.init_cfg
            if isinstance(self.init_cfg, dict):
                init_cfgs = [self.init_cfg]

            # PretrainedInit has higher priority than any other init_cfg.
            # Therefore we initialize `pretrained_cfg` last to overwrite
            # the previous initialized weights.
            # See details in https://github.com/open-mmlab/mmengine/issues/691 # noqa E501
            other_cfgs = []
            pretrained_cfg = []
            for init_cfg in init_cfgs:
                assert isinstance(init_cfg, dict)
                if (
                    init_cfg["type"] == "Pretrained"
                    or init_cfg["type"] is PretrainedInit
                ):
                    pretrained_cfg.append(init_cfg)
                else:
                    other_cfgs.append(init_cfg)

            initialize(self, other_cfgs)

        else:
            super().init_weights()

        initialize(self, pretrained_cfg)

    def predict(
        self, batch_inputs, detection_features, batch_data_samples, rescale: bool = True
    ):

        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if "caption_prompt" in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get("tokens_positive", None))

        if "custom_entities" in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        if self.track_text_dict is not None and self.track_text_prompt == text_prompts:
            # text feature map layer

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if self.token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = self.token_positive_maps[i]

            visual_feats = detection_features

            head_inputs_dict = self.forward_transformer(
                visual_feats, self.track_text_dict, batch_data_samples
            )
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples,
            )

            entities = self.track_entities

        else:
            self.track_text_prompt = text_prompts

            if len(text_prompts) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                _positive_maps_and_prompts = [
                    self.get_tokens_positive_and_prompts(
                        text_prompts[0],
                        custom_entities,
                        enhanced_text_prompts[0],
                        tokens_positives[0],
                    )
                ] * len(batch_inputs)
            else:
                _positive_maps_and_prompts = [
                    self.get_tokens_positive_and_prompts(
                        text_prompt,
                        custom_entities,
                        enhanced_text_prompt,
                        tokens_positive,
                    )
                    for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                        text_prompts, enhanced_text_prompts, tokens_positives
                    )
                ]
            token_positive_maps, text_prompts, _, entities = zip(
                *_positive_maps_and_prompts
            )

            self.token_positive_maps = token_positive_maps
            self.track_entities = entities

            # image feature extraction
            visual_feats = detection_features

            if isinstance(text_prompts[0], list):
                # chunked text prompts, only bs=1 is supported
                assert len(batch_inputs) == 1
                count = 0
                results_list = []

                entities = [[item for lst in entities[0] for item in lst]]

                for b in range(len(text_prompts[0])):
                    text_prompts_once = [text_prompts[0][b]]
                    token_positive_maps_once = token_positive_maps[0][b]
                    text_dict = self.language_model(text_prompts_once)
                    # text feature map layer
                    if self.text_feat_map is not None:
                        text_dict["embedded"] = self.text_feat_map(
                            text_dict["embedded"]
                        )

                    batch_data_samples[0].token_positive_map = token_positive_maps_once

                    head_inputs_dict = self.forward_transformer(
                        copy.deepcopy(visual_feats), text_dict, batch_data_samples
                    )
                    pred_instances = self.bbox_head.predict(
                        **head_inputs_dict,
                        rescale=rescale,
                        batch_data_samples=batch_data_samples,
                    )[0]

                    if len(pred_instances) > 0:
                        pred_instances.labels += count
                    count += len(token_positive_maps_once)
                    results_list.append(pred_instances)
                results_list = [results_list[0].cat(results_list)]
                is_rec_tasks = [False] * len(results_list)
            else:
                # extract text feats
                text_dict = self.language_model(list(text_prompts))
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

                is_rec_tasks = []
                for i, data_samples in enumerate(batch_data_samples):
                    if token_positive_maps[i] is not None:
                        is_rec_tasks.append(False)
                    else:
                        is_rec_tasks.append(True)
                    data_samples.token_positive_map = token_positive_maps[i]

                if self.track_text_dict is None:
                    self.track_text_dict = text_dict

                head_inputs_dict = self.forward_transformer(
                    visual_feats, text_dict, batch_data_samples
                )
                results_list = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples,
                )

        for data_sample, pred_instances, entity, is_rec_task in zip(
            batch_data_samples, results_list, entities, is_rec_tasks
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            "The unexpected output indicates an issue with "
                            "named entity recognition. You can try "
                            "setting custom_entities=True and running "
                            "again to see if it helps."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

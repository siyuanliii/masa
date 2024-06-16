import collections
import copy
import random
from typing import List, Sequence, Union

import numpy as np
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.base_video_dataset import BaseVideoDataset
from mmdet.registry import DATASETS, TRANSFORMS
from mmengine.dataset import BaseDataset, force_full_init

from .rsconcat_dataset import RandomSampleJointVideoConcatDataset


@DATASETS.register_module(force=True)
class SeqMultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(
        self,
        dataset: Union[BaseDataset, dict],
        pipeline: Sequence[str],
        skip_type_keys: Union[Sequence[str], None] = None,
        max_refetch: int = 15,
        lazy_init: bool = False,
    ) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all(
                [isinstance(skip_type_key, str) for skip_type_key in skip_type_keys]
            )
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform["type"])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError("pipeline must be a dict")

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                "elements in datasets sequence should be config or "
                f"`BaseDataset` instance, but got {type(dataset)}"
            )

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, "flag"):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

        self.generate_indices()

    def generate_indices(self):
        cat_datasets = self.dataset.datasets
        for dataset in cat_datasets:
            self.test_mode = dataset.test_mode
            assert not self.test_mode, "'ConcatDataset' should not exist in "
            "test mode"
            video_indices = []
            img_indices = []
            if isinstance(dataset, BaseVideoDataset):
                num_videos = len(dataset)
                for video_ind in range(num_videos):
                    video_indices.extend(
                        [
                            (video_ind, frame_ind)
                            for frame_ind in range(dataset.get_len_per_video(video_ind))
                        ]
                    )
            elif isinstance(dataset, BaseDetDataset):
                num_imgs = len(dataset)
                for img_ind in range(num_imgs):
                    img_indices.extend([img_ind])

        ###### special process to make debug task easier #####
        def alternate_merge(list1, list2):
            # Create a new list to hold the merged elements
            merged_list = []

            # Get the length of the shorter list
            min_length = min(len(list1), len(list2))

            # Append elements alternately from both lists
            for i in range(min_length):
                merged_list.append(list1[i])
                merged_list.append(list2[i])

            # Append the remaining elements from the longer list
            if len(list1) > len(list2):
                merged_list.extend(list1[min_length:])
            else:
                merged_list.extend(list2[min_length:])

            return merged_list

        self.indices = alternate_merge(img_indices, video_indices)

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def get_transform_indexes(self, transform, results, t_type="SeqMosaic"):
        num_samples = len(results["img_id"])
        for i in range(self.max_refetch):
            # Make sure the results passed the loading pipeline
            # of the original dataset is not None.
            indexes = transform.get_indexes(self.dataset)
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]
            mix_results = [copy.deepcopy(self.dataset[index]) for index in indexes]
            if None not in mix_results:
                if t_type == "SeqMosaic":
                    results["mosaic_mix_results"] = [mix_results] * num_samples
                elif t_type == "SeqMixUp":
                    results["mixup_mix_results"] = [mix_results] * num_samples
                elif t_type == "SeqCopyPaste":
                    results["copypaste_mix_results"] = [mix_results] * num_samples
                return results
        else:
            raise RuntimeError(
                "The loading pipeline of the original dataset"
                " always return None. Please check the correctness "
                "of the dataset and its pipeline."
            )

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            results = copy.deepcopy(self.dataset[idx])

            for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
                if (
                    self._skip_type_keys is not None
                    and transform_type in self._skip_type_keys
                ):
                    continue
                if transform_type == "MasaTransformBroadcaster":
                    for sub_transform in transform.transforms:
                        if hasattr(sub_transform, "get_indexes"):
                            sub_transform_type = type(sub_transform).__name__
                            results = self.get_transform_indexes(
                                sub_transform, results, sub_transform_type
                            )

                elif hasattr(transform, "get_indexes"):
                    for i in range(self.max_refetch):
                        # Make sure the results passed the loading pipeline
                        # of the original dataset is not None.
                        indexes = transform.get_indexes(self.dataset)
                        if not isinstance(indexes, collections.abc.Sequence):
                            indexes = [indexes]
                        mix_results = [
                            copy.deepcopy(self.dataset[index]) for index in indexes
                        ]
                        if None not in mix_results:
                            results["mix_results"] = mix_results
                            break
                    else:
                        raise RuntimeError(
                            "The loading pipeline of the original dataset"
                            " always return None. Please check the correctness "
                            "of the dataset and its pipeline."
                        )

                for i in range(self.max_refetch):
                    # To confirm the results passed the training pipeline
                    # of the wrapper is not None.
                    try:
                        updated_results = transform(copy.deepcopy(results))
                    except Exception as e:
                        print(
                            "Error occurred while running pipeline",
                            f"{transform} with error: {e}",
                        )
                        # print('Empty instances due to augmentation, re-sampling...')
                        idx = self._rand_another(idx)
                        continue
                    if updated_results is not None:
                        results = updated_results
                        break
                else:
                    raise RuntimeError(
                        "The training pipeline of the dataset wrapper"
                        " always return None.Please check the correctness "
                        "of the dataset and its pipeline."
                    )

                if "mosaic_mix_results" in results:
                    results.pop("mosaic_mix_results")

                if "mixup_mix_results" in results:
                    results.pop("mixup_mix_results")

                if "copypaste_mix_results" in results:
                    results.pop("copypaste_mix_results")

            return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([isinstance(skip_type_key, str) for skip_type_key in skip_type_keys])
        self._skip_type_keys = skip_type_keys

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        return np.random.choice(self.indices)


@DATASETS.register_module()
class SeqRandomMultiImageVideoMixDataset(SeqMultiImageMixDataset):
    def __init__(
        self, video_pipeline: Sequence[str], video_sample_ratio=0.5, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.video_pipeline = []
        self.video_pipeline_types = []
        for transform in video_pipeline:
            if isinstance(transform, dict):
                self.video_pipeline_types.append(transform["type"])
                transform = TRANSFORMS.build(transform)
                self.video_pipeline.append(transform)
            else:
                raise TypeError("pipeline must be a dict")

        self.video_sample_ratio = video_sample_ratio
        assert isinstance(self.dataset, RandomSampleJointVideoConcatDataset)

    @force_full_init
    def get_transform_indexes(
        self, transform, results, sample_video, t_type="SeqMosaic"
    ):
        num_samples = len(results["img_id"])
        for i in range(self.max_refetch):
            # Make sure the results passed the loading pipeline
            # of the original dataset is not None.

            indexes = transform.get_indexes(self.dataset.datasets[0])
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]
            if sample_video:
                mix_results = [copy.deepcopy(self.dataset[0]) for index in indexes]
            else:
                mix_results = [copy.deepcopy(self.dataset[1]) for index in indexes]

            if None not in mix_results:
                if t_type == "SeqMosaic":
                    results["mosaic_mix_results"] = [mix_results] * num_samples
                elif t_type == "SeqMixUp":
                    results["mixup_mix_results"] = [mix_results] * num_samples
                elif t_type == "SeqCopyPaste":
                    results["copypaste_mix_results"] = [mix_results] * num_samples
                return results
        else:
            raise RuntimeError(
                "The loading pipeline of the original dataset"
                " always return None. Please check the correctness "
                "of the dataset and its pipeline."
            )

    def __getitem__(self, idx):

        while True:
            if random.random() < self.video_sample_ratio:
                sample_video = True
            else:
                sample_video = False
            if sample_video:
                results = copy.deepcopy(self.dataset[0])
                pipeline = self.video_pipeline
                pipeline_type = self.video_pipeline_types

            else:
                results = copy.deepcopy(self.dataset[1])
                pipeline = self.pipeline
                pipeline_type = self.pipeline_types
                # if results['img_id'][0] != results['img_id'][1]:
                #     self.update_skip_type_keys(['SeqMosaic', 'SeqMixUp'])
                # else:
                #     self._skip_type_keys = None

            for (transform, transform_type) in zip(pipeline, pipeline_type):
                if (
                    self._skip_type_keys is not None
                    and transform_type in self._skip_type_keys
                ):
                    continue
                if transform_type == "MasaTransformBroadcaster":
                    for sub_transform in transform.transforms:
                        if hasattr(sub_transform, "get_indexes"):
                            sub_transform_type = type(sub_transform).__name__
                            results = self.get_transform_indexes(
                                sub_transform, results, sample_video, sub_transform_type
                            )

                elif hasattr(transform, "get_indexes"):
                    for i in range(self.max_refetch):
                        # Make sure the results passed the loading pipeline
                        # of the original dataset is not None.
                        indexes = transform.get_indexes(self.dataset)
                        if not isinstance(indexes, collections.abc.Sequence):
                            indexes = [indexes]
                        mix_results = [
                            copy.deepcopy(self.dataset[index]) for index in indexes
                        ]
                        if None not in mix_results:
                            results["mix_results"] = mix_results
                            break
                    else:
                        raise RuntimeError(
                            "The loading pipeline of the original dataset"
                            " always return None. Please check the correctness "
                            "of the dataset and its pipeline."
                        )

                for i in range(self.max_refetch):
                    # To confirm the results passed the training pipeline
                    # of the wrapper is not None.
                    try:
                        updated_results = transform(copy.deepcopy(results))
                    except Exception as e:
                        print(
                            "Error occurred while running pipeline",
                            f"{transform} with error: {e}",
                        )
                        # print('Empty instances due to augmentation, re-sampling...')
                        # idx = self._rand_another(idx)
                        continue
                    if updated_results is not None:
                        results = updated_results
                        break
                else:
                    raise RuntimeError(
                        "The training pipeline of the dataset wrapper"
                        " always return None.Please check the correctness "
                        "of the dataset and its pipeline."
                    )

                if "mosaic_mix_results" in results:
                    results.pop("mosaic_mix_results")

                if "mixup_mix_results" in results:
                    results.pop("mixup_mix_results")

                if "copypaste_mix_results" in results:
                    results.pop("copypaste_mix_results")

            return results

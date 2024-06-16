import random
from typing import Iterable, List

import numpy as np
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.base_video_dataset import BaseVideoDataset
from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


@DATASETS.register_module()
class RandomSampleConcatDataset(_ConcatDataset):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        sampling_probs: List[float],
        fixed_length: int,
        lazy_init: bool = False,
    ):
        super(RandomSampleConcatDataset, self).__init__(datasets)
        assert len(sampling_probs) == len(
            datasets
        ), "Number of sampling probabilities must match the number of datasets"
        assert sum(sampling_probs) == 1.0, "Sum of sampling probabilities must be 1.0"

        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    "elements in datasets sequence should be config or "
                    f"`BaseDataset` instance, but got {type(dataset)}"
                )
        self.sampling_probs = sampling_probs
        self.fixed_length = fixed_length

        self.metainfo = self.datasets[0].metainfo
        total_datasets_length = sum([len(dataset) for dataset in self.datasets])
        assert (
            self.fixed_length <= total_datasets_length
        ), "the length of the concatenated dataset must be less than the sum of the lengths of the individual datasets"
        self.flag = np.zeros(self.fixed_length, dtype=np.uint8)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for i, dataset in enumerate(self.datasets):
            dataset.full_init()
        self._ori_len = self.fixed_length
        self._fully_initialized = True

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    def __len__(self):
        return self.fixed_length

    def __getitem__(self, idx):
        # Choose a dataset based on the sampling probabilities
        chosen_dataset_idx = random.choices(
            range(len(self.datasets)), weights=self.sampling_probs, k=1
        )[0]
        chosen_dataset = self.datasets[chosen_dataset_idx]

        # Sample a random item from the chosen dataset
        sample_idx = random.randrange(0, len(chosen_dataset))
        return chosen_dataset[sample_idx]


@DATASETS.register_module()
class RandomSampleJointVideoConcatDataset(_ConcatDataset):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        fixed_length: int,
        lazy_init: bool = False,
        video_sampling_probs: List[float] = [],
        img_sampling_probs: List[float] = [],
        *args,
        **kwargs,
    ):
        super(RandomSampleJointVideoConcatDataset, self).__init__(datasets)

        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    "elements in datasets sequence should be config or "
                    f"`BaseDataset` instance, but got {type(dataset)}"
                )

        self.video_dataset_idx = []
        self.img_dataset_idx = []
        self.datasets_indices_mapping = {}
        for i, dataset in enumerate(self.datasets):
            if isinstance(dataset, BaseVideoDataset):
                self.video_dataset_idx.append(i)
                num_videos = len(dataset)
                video_indices = []
                for video_ind in range(num_videos):
                    video_indices.extend(
                        [
                            (video_ind, frame_ind)
                            for frame_ind in range(dataset.get_len_per_video(video_ind))
                        ]
                    )
                self.datasets_indices_mapping[i] = video_indices

            elif isinstance(dataset, BaseDetDataset):
                self.img_dataset_idx.append(i)
                img_indices = []
                num_imgs = len(dataset)
                for img_ind in range(num_imgs):
                    img_indices.extend([img_ind])
                self.datasets_indices_mapping[i] = img_indices

            else:
                raise TypeError(
                    "elements in datasets sequence should be config or "
                    f"`BaseDataset` instance, but got {type(dataset)}"
                )

        self.fixed_length = fixed_length
        self.metainfo = self.datasets[0].metainfo
        total_datasets_length = sum(
            [len(indices) for key, indices in self.datasets_indices_mapping.items()]
        )
        assert (
            self.fixed_length <= total_datasets_length
        ), "the length of the concatenated dataset must be less than the sum of the lengths of the individual datasets"
        self.flag = np.zeros(self.fixed_length, dtype=np.uint8)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

        self.video_sampling_probs = video_sampling_probs
        self.img_sampling_probs = img_sampling_probs
        if self.video_sampling_probs:
            assert (
                sum(self.video_sampling_probs) == 1.0
            ), "Sum of video sampling probabilities must be 1.0"
        if self.img_sampling_probs:
            assert (
                sum(self.img_sampling_probs) == 1.0
            ), "Sum of image sampling probabilities must be 1.0"

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for i, dataset in enumerate(self.datasets):
            dataset.full_init()
        self._ori_len = self.fixed_length
        self._fully_initialized = True

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    def __len__(self):
        return self.fixed_length

    def __getitem__(self, idx):
        # idx ==0 means samples from video dataset, idx == 1 means samples from image dataset
        # Choose a dataset based on the sampling probabilities
        if idx == 0:
            chosen_dataset_idx = random.choices(
                self.video_dataset_idx, weights=self.video_sampling_probs, k=1
            )[0]
        elif idx == 1:
            chosen_dataset_idx = random.choices(
                self.img_dataset_idx, weights=self.img_sampling_probs, k=1
            )[0]

        chosen_dataset = self.datasets[chosen_dataset_idx]
        # Sample a random item from the chosen dataset
        sample_idx = random.choice(self.datasets_indices_mapping[chosen_dataset_idx])

        return chosen_dataset[sample_idx]

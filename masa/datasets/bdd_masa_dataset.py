from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
from mmdet.datasets import BaseVideoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class BDDVideoDataset(BaseVideoDataset):
    """Dataset for TAO benchmark.
    """

    METAINFO = {
        "classes": (
            "pedestrian",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ),
        "palette": None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_data(self, idx) -> Any:
        """Get date processed by ``self.pipeline``. Note that ``idx`` is a
        video index in default since the base element of video dataset is a
        video. However, in some cases, we need to specific both the video index
        and frame index. For example, in traing mode, we may want to sample the
        specific frames and all the frames must be sampled once in a epoch; in
        test mode, we may want to output data of a single image rather than the
        whole video for saving memory.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if isinstance(idx, tuple):
            assert len(idx) == 2, "The length of idx must be 2: "
            "(video_index, frame_index)"
            video_idx, frame_idx = idx[0], idx[1]
        else:
            video_idx, frame_idx = idx, None

        data_info = self.get_data_info(video_idx)
        if self.test_mode:
            # Support two test_mode: frame-level and video-level
            final_data_info = defaultdict(list)
            if frame_idx is None:
                frames_idx_list = list(range(data_info["video_length"]))
            else:
                frames_idx_list = [frame_idx]
            for index in frames_idx_list:
                frame_ann = data_info["images"][index]
                frame_ann["video_id"] = data_info["video_id"]
                # Collate data_list (list of dict to dict of list)
                for key, value in frame_ann.items():
                    final_data_info[key].append(value)
                # copy the info in video-level into img-level
                # TODO: the value of this key is the same as that of
                # `video_length` in test mode
                final_data_info["ori_video_length"].append(data_info["video_length"])

            final_data_info["video_length"] = [len(frames_idx_list)] * len(
                frames_idx_list
            )
            return self.pipeline(final_data_info)
        else:
            # Specify `key_frame_id` for the frame sampling in the pipeline
            if frame_idx is not None:
                data_info["key_frame_id"] = frame_idx

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                try:

                    data = self.pipeline(data_info)
                except Exception as e:
                    print("Error occurred while running pipeline", f" with error: {e}")
                    # print('Empty instances due to augmentation, re-sampling...')
                    video_idx = self._rand_another(video_idx)
                    data_info = self.get_data_info(video_idx)
                    continue

                if data is not None:
                    break
            return data

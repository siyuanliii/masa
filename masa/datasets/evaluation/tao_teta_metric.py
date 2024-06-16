import os
import os.path as osp
import pickle
import shutil
import tempfile
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import tqdm

try:
    import teta
except ImportError:
    teta = None

import mmengine
import mmengine.fileio as fileio
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric
from mmdet.registry import METRICS, TASK_UTILS
from mmengine.dist import (all_gather_object, barrier, broadcast,
                           broadcast_object_list, get_dist_info,
                           is_main_process)
from mmengine.logging import MMLogger


def get_tmpdir() -> str:
    """return the same tmpdir for all processes."""
    rank, world_size = get_dist_info()
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8)
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
        tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
        dir_tensor[: len(tmpdir)] = tmpdir
    broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    return tmpdir


@METRICS.register_module()
class TaoTETAMetric(BaseVideoMetric):
    """Evaluation metrics for TAO TETA and open-vocabulary MOT benchmark.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'TETA'
            Defaults to ['TETA'].
        outfile_prefix (str, optional): Path to save the formatted results.
            Defaults to None.
        track_iou_thr (float): IoU threshold for tracking evaluation.
            Defaults to 0.5.
        benchmark (str): Benchmark to be evaluated. Defaults to 'MOT17'.
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
        postprocess_tracklet_cfg (List[dict], optional): configs for tracklets
            postprocessing methods. `InterpolateTracklets` is supported.
            Defaults to []
            - InterpolateTracklets:
                - min_num_frames (int, optional): The minimum length of a
                    track that will be interpolated. Defaults to 5.
                - max_num_frames (int, optional): The maximum disconnected
                    length in a track. Defaults to 20.
                - use_gsi (bool, optional): Whether to use the GSI (Gaussian-
                    smoothed interpolation) method. Defaults to False.
                - smooth_tau (int, optional): smoothing parameter in GSI.
                    Defaults to 10.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    Returns:
    """

    TRACKER = "masa-tracker"
    allowed_metrics = ["TETA"]
    default_prefix: Optional[str] = "tao_teta_metric"

    def __init__(
        self,
        metric: Union[str, List[str]] = ["TETA"],
        outfile_prefix: Optional[str] = None,
        track_iou_thr: float = 0.5,
        format_only: bool = False,
        ann_file: Optional[str] = None,
        dataset_type: str = "Taov1Dataset",
        use_postprocess: bool = False,
        postprocess_tracklet_cfg: Optional[List[dict]] = [],
        collect_device: str = "cpu",
        tcc: bool = True,
        open_vocabulary=False,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if teta is None:
            raise RuntimeError(
                "teta is not installed,"
                "please install it by:  python -m pip install git+https://github.com/SysCV/tet.git/#subdirectory=teta "
            )

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str.")
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")
        self.metrics = metrics
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, "outfile_prefix must be not"
            "None when format_only is True, otherwise the result files will"
            "be saved to a temp directory which will be cleaned up at the end."
        self.use_postprocess = use_postprocess
        self.postprocess_tracklet_cfg = postprocess_tracklet_cfg.copy()
        self.postprocess_tracklet_methods = [
            TASK_UTILS.build(cfg) for cfg in self.postprocess_tracklet_cfg
        ]
        self.track_iou_thr = track_iou_thr
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir.name = get_tmpdir()
        self.seq_pred = defaultdict(lambda: [])
        self.gt_dir = self._get_gt_dir()
        self.pred_dir = self._get_pred_dir(outfile_prefix)
        self.outfile_prefix = outfile_prefix

        self.ann_file = ann_file
        self.tcc = tcc
        self.open_vocabulary = open_vocabulary

        with fileio.get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)

        # get the class list according to the dataset type
        assert dataset_type in ["Taov05Dataset", "Taov1Dataset"]
        if dataset_type == "Taov05Dataset":
            from masa.datasets import Taov05Dataset

            self.class_list = Taov05Dataset.METAINFO["classes"]
        if dataset_type == "Taov1Dataset":
            from masa.datasets import Taov1Dataset

            self.class_list = Taov1Dataset.METAINFO["classes"]
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.class_list)

    def __del__(self):
        # To avoid tmpdir being cleaned up too early, because in multiple
        # consecutive ValLoops, the value of `self.tmp_dir.name` is unchanged,
        # and calling `tmp_dir.cleanup()` in compute_metrics will cause errors.
        self.tmp_dir.cleanup()

    def _get_pred_dir(self, outfile_prefix):
        """Get directory to save the prediction results."""
        logger: MMLogger = MMLogger.get_current_instance()

        if outfile_prefix is None:
            outfile_prefix = self.tmp_dir.name
        else:
            if osp.exists(outfile_prefix) and is_main_process():
                logger.info("remove previous results.")
                shutil.rmtree(outfile_prefix)
        pred_dir = osp.join(outfile_prefix, self.TRACKER)
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir

    def _get_gt_dir(self):
        """Get directory to save the gt files."""
        output_dir = osp.join(self.tmp_dir.name, "gt")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def transform_gt_and_pred(self, img_data_sample):

        # load predictions
        assert "pred_track_instances" in img_data_sample
        pred_instances = img_data_sample["pred_track_instances"]

        pred_instances_list = []

        for i in range(len(pred_instances["instances_id"])):
            data_dict = dict()
            data_dict["image_id"] = img_data_sample["img_id"]
            data_dict["track_id"] = int(pred_instances["instances_id"][i])
            data_dict["bbox"] = self.xyxy2xywh(pred_instances["bboxes"][i])
            data_dict["score"] = float(pred_instances["scores"][i])
            data_dict["category_id"] = self.cat_ids[pred_instances["labels"][i]]
            data_dict["video_id"] = img_data_sample["video_id"]
            pred_instances_list.append(data_dict)

        return pred_instances_list

    def process_image(self, data_samples, video_len):

        img_data_sample = data_samples[0].to_dict()
        video_id = img_data_sample["video_id"]
        pred_instances_list = self.transform_gt_and_pred(img_data_sample)
        self.seq_pred[video_id].extend(pred_instances_list)

    def process_video(self, data_samples):

        video_len = len(data_samples)
        for frame_id in range(video_len):
            img_data_sample = data_samples[frame_id].to_dict()
            # load basic info
            video_id = img_data_sample["video_id"]
            pred_instances_list = self.transform_gt_and_pred(img_data_sample)
            self.seq_pred[video_id].extend(pred_instances_list)

    def compute_metrics(self, results: list = None) -> dict:

        logger: MMLogger = MMLogger.get_current_instance()

        eval_results = dict()

        if self.format_only:
            logger.info("Only formatting results to the official format.")
            return eval_results

        resfile_path = self.outfile_prefix

        # Command line interface:
        default_eval_config = teta.config.get_default_eval_config()
        # print only combined since TrackMAP is undefined for per sequence breakdowns
        default_eval_config["PRINT_ONLY_COMBINED"] = True
        default_eval_config["DISPLAY_LESS_PROGRESS"] = True
        default_eval_config["OUTPUT_TEM_RAW_DATA"] = True
        default_eval_config["NUM_PARALLEL_CORES"] = 8
        default_dataset_config = teta.config.get_default_dataset_config()
        default_dataset_config["TRACKERS_TO_EVAL"] = ["MASA"]
        default_dataset_config["GT_FOLDER"] = self.ann_file
        default_dataset_config["OUTPUT_FOLDER"] = resfile_path
        default_dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(
            resfile_path, "tao_track.json"
        )

        evaluator = teta.Evaluator(default_eval_config)
        dataset_list = [teta.datasets.TAO(default_dataset_config)]
        print("Overall classes performance")
        eval_results, _ = evaluator.evaluate(dataset_list, [teta.metrics.TETA()])

        if self.open_vocabulary:
            eval_results_path = os.path.join(
                resfile_path, "MASA", "teta_summary_results.pth"
            )
            eval_res = pickle.load(open(eval_results_path, "rb"))

            base_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] != "r"
                ]
            )
            novel_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] == "r"
                ]
            )

            self.compute_teta_on_ovsetup(
                eval_res, base_class_synset, novel_class_synset
            )

        return eval_results

    def evaluate(self, size: int = 1) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.
                Defaults to None.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.

        """
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info(f"Wait for all processes to complete prediction.")
        # wait for all processes to complete prediction.
        barrier()

        logger.info(f"Start gathering tracking results.")

        # gather seq_info and convert the list of dict to a dict.
        # convert self.seq_info to dict first to make it picklable.
        gathered_seq_info = all_gather_object(dict(self.seq_pred))

        if is_main_process():

            all_seq_pred = dict()
            for _seq_info in gathered_seq_info:
                all_seq_pred.update(_seq_info)
            all_seq_pred = self.compute_global_track_id(all_seq_pred)

            # merge all the values (list of pred in each videos) into a single long list
            all_seq_pred_json = list(chain.from_iterable(all_seq_pred.values()))

            if self.tcc and all_seq_pred_json:
                all_seq_pred_json = self.majority_vote(all_seq_pred_json)

            result_files_path = f"{self.outfile_prefix}/tao_track.json"

            logger.info(f"Saving json pred file into {result_files_path}")
            mmengine.dump(all_seq_pred_json, result_files_path)

            logger.info(f"Start evaluation")

            _metrics = self.compute_metrics()

            # Add prefix to metric names
            if self.prefix:
                _metrics = {"/".join((self.prefix, k)): v for k, v in _metrics.items()}
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)
        self.seq_pred.clear()

        return metrics[0]

    def compute_global_track_id(self, all_seq_pred):

        max_track_id = 0

        for video_id, seq_pred in all_seq_pred.items():
            track_ids = []

            for frame_pred in seq_pred:
                track_ids.append(frame_pred["track_id"])
                frame_pred["track_id"] += max_track_id
            track_ids = list(set(track_ids))

            if track_ids:
                max_track_id += max(track_ids) + 1

        return all_seq_pred

    def majority_vote(self, prediction):

        tid_res_mapping = {}
        for res in prediction:
            tid = res["track_id"]
            if tid not in tid_res_mapping:
                tid_res_mapping[tid] = [res]
            else:
                tid_res_mapping[tid].append(res)
        # change the results to data frame
        df_pred_res = pd.DataFrame(prediction)
        # group the results by track_id
        # df_pred_res = df_pred_res.apply(changebbox, axis=1)
        groued_df_pred_res = df_pred_res.groupby("track_id")

        # change the majority
        class_by_majority_count_res = []
        for tid, group in tqdm.tqdm(groued_df_pred_res):
            cid = group["category_id"].mode()[0]
            group["category_id"] = cid
            dict_list = group.to_dict("records")
            class_by_majority_count_res += dict_list
        return class_by_majority_count_res

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def compute_teta_on_ovsetup(self, teta_res, base_class_names, novel_class_names):
        if "COMBINED_SEQ" in teta_res:
            teta_res = teta_res["COMBINED_SEQ"]

        frequent_teta = []
        rare_teta = []
        for key in teta_res:
            if key in base_class_names:
                frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
            elif key in novel_class_names:
                rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

        print("Base and Novel classes performance")

        # print the header
        print(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "TETA50:",
                "TETA",
                "LocA",
                "AssocA",
                "ClsA",
                "LocRe",
                "LocPr",
                "AssocRe",
                "AssocPr",
                "ClsRe",
                "ClsPr",
            )
        )

        if frequent_teta:
            freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

            # print the frequent teta mean
            print("{:<10} ".format("Base"), end="")
            print(*["{:<10.3f}".format(num) for num in freq_teta_mean])

        else:
            print("No Base classes to evaluate!")
            freq_teta_mean = None
        if rare_teta:
            rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

            # print the rare teta mean
            print("{:<10} ".format("Novel"), end="")
            print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
        else:
            print("No Novel classes to evaluate!")
            rare_teta_mean = None

        return freq_teta_mean, rare_teta_mean

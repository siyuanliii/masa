import json
import os
import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict
from itertools import chain
from multiprocessing import Pool
from typing import List, Optional, Sequence, Union

import pandas as pd
import torch
import tqdm

try:
    import teta
except ImportError:
    teta = None
from pathlib import Path

import mmengine
import mmengine.fileio as fileio
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric
from mmdet.registry import METRICS, TASK_UTILS
from mmengine.dist import (all_gather_object, barrier, broadcast,
                           broadcast_object_list, get_dist_info,
                           is_main_process)
from mmengine.logging import MMLogger
from scalabel.eval.box_track import BoxTrackResult, bdd100k_to_scalabel
from scalabel.eval.hota import HOTAResult, evaluate_track_hota
from scalabel.eval.hotas import evaluate_seg_track_hota
from scalabel.eval.mot import TrackResult, acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots, evaluate_seg_track
from scalabel.eval.teta import TETAResult, evaluate_track_teta
from scalabel.eval.tetas import evaluate_seg_track_teta
from scalabel.label.io import group_and_sort, load, load_label_config

from .utils import mask_postprocess, mask_prepare

cpu_num = os.cpu_count()
NPROC: int = min(4, cpu_num if cpu_num else 1)

MOT_CFG_FILE = os.path.join(
    str(Path(__file__).parent.absolute()), "dataset_configs/box_track.toml"
)
MOTS_CFG_FILE = os.path.join(
    str(Path(__file__).parent.absolute()), "dataset_configs/seg_track.toml"
)


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
class BDDTETAMetric(BaseVideoMetric):
    """Evaluation metrics for MOT Challenge.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Options are
            'HOTA', 'CLEAR', 'Identity'.
            Defaults to ['HOTA', 'CLEAR', 'Identity'].
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
    allowed_metrics = ["TETA", "HOTA", "CLEAR"]
    default_prefix: Optional[str] = "tao_teta_metric"

    def __init__(
        self,
        metric: Union[str, List[str]] = ["TETA", "HOTA", "CLEAR"],
        outfile_prefix: Optional[str] = None,
        track_iou_thr: float = 0.5,
        format_only: bool = False,
        ann_file: Optional[str] = None,
        scalabel_gt: Optional[str] = None,
        dataset_type: str = "BDDVideoDataset",
        use_postprocess: bool = False,
        postprocess_tracklet_cfg: Optional[List[dict]] = [],
        collect_device: str = "cpu",
        tcc: bool = True,
        scalabel_format=True,
        with_mask=False,
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
        self.scalabel_format = scalabel_format
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
        self.scalabel_gt = scalabel_gt
        self.tcc = tcc
        self.with_mask = with_mask

        with fileio.get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)

        # get the class list according to the dataset type
        assert dataset_type in ["BDDVideoDataset"]
        if dataset_type == "BDDVideoDataset":
            from masa.datasets import BDDVideoDataset

            self.class_list = BDDVideoDataset.METAINFO["classes"]

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
            if self.with_mask:
                if isinstance(pred_instances["masks"][i]["counts"], bytes):
                    pred_instances["masks"][i]["counts"] = pred_instances["masks"][i][
                        "counts"
                    ].decode()
                data_dict["segmentation"] = pred_instances["masks"][i]
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

        resfile_path = os.path.join(
            self.outfile_prefix, "bdd_track_scalabel_format.json"
        )

        bdd100k_config = load_label_config(MOT_CFG_FILE)
        print("Start loading.")

        gts = group_and_sort(load(self.scalabel_gt).frames)
        results = group_and_sort(load(resfile_path).frames)
        print("gt_len", len(gts), "results", len(results))
        print("Finish loading.")
        print("Start evaluation")
        print("Ignore unknown cats")

        logger.info("Tracking evaluation.")
        t = time.time()
        gts = [bdd100k_to_scalabel(gt, bdd100k_config) for gt in gts]
        results = [bdd100k_to_scalabel(result, bdd100k_config) for result in results]

        if "CLEAR" in self.metrics:
            if self.with_mask:
                mot_result = evaluate_seg_track(
                    acc_single_video_mots,
                    gts,
                    results,
                    bdd100k_config,
                    ignore_unknown_cats=True,
                    nproc=NPROC,
                )

            else:

                mot_result = evaluate_track(
                    acc_single_video_mot,
                    gts,
                    results,
                    bdd100k_config,
                    ignore_unknown_cats=True,
                    nproc=NPROC,
                )
            print("CLEAR and IDF1 results :")
            print(mot_result)
            print(mot_result.summary())

        if "HOTA" in self.metrics:
            if self.with_mask:
                hota_result = evaluate_seg_track_hota(
                    gts, results, bdd100k_config, NPROC
                )
            else:
                hota_result = evaluate_track_hota(gts, results, bdd100k_config, NPROC)
            print("HOTA results :")
            print(hota_result)
            print(hota_result.summary())

        if "TETA" in self.metrics:
            if self.with_mask:
                teta_result = evaluate_seg_track_teta(
                    gts, results, bdd100k_config, NPROC
                )
            else:
                teta_result = evaluate_track_teta(gts, results, bdd100k_config, NPROC)

            print("TETA results :")
            print(teta_result)
            print(teta_result.summary())

        if (
            "CLEAR" in self.metrics
            and "HOTA" in self.metrics
            and "TETA" in self.metrics
        ):
            print("Aggregated results: ")
            combined_result = BoxTrackResult(
                **{**mot_result.dict(), **hota_result.dict(), **teta_result.dict()}
            )
            print(combined_result)
            print(combined_result.summary())

        t = time.time() - t
        logger.info("evaluation finishes with %.1f s.", t)

        print("Completed evaluation")
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

            if self.scalabel_format:
                all_seq_pred_json = self.format_scalabel_pred(all_seq_pred_json)

                result_files_path = (
                    f"{self.outfile_prefix}/bdd_track_scalabel_format.json"
                )

                logger.info(f"Saving json pred file into {result_files_path}")
                mmengine.dump(all_seq_pred_json, result_files_path)
            else:
                if self.tcc and all_seq_pred_json:
                    all_seq_pred_json = self.majority_vote(all_seq_pred_json)
                else:
                    all_seq_pred_json = all_seq_pred_json

                result_files_path = f"{self.outfile_prefix}/bdd_track_cocofmt.json"

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

    def format_scalabel_pred(self, all_seq_pred_json):
        """Convert the prediction results to the format of Scalabel.

        Args:
            all_seq_pred_json (list): The prediction results.

        Returns:
            list: The formatted prediction results.
        """

        bdd_scalabel_gt = json.load(open(self.ann_file))
        bdd_cid_cinfo_mapping = {}
        for c in bdd_scalabel_gt["categories"]:
            if c["id"] not in bdd_cid_cinfo_mapping:
                bdd_cid_cinfo_mapping[c["id"]] = c
        # imid info mapping
        imid_iminfo_mapping = {}
        for i in bdd_scalabel_gt["images"]:
            if i["id"] not in imid_iminfo_mapping:
                imid_iminfo_mapping[i["id"]] = i
        # vidid info mapping
        vid_vinfo_mapping = {}
        for i in bdd_scalabel_gt["videos"]:
            if i["id"] not in vid_vinfo_mapping:
                vid_vinfo_mapping[i["id"]] = i

        if self.tcc and all_seq_pred_json:
            mc_res = self.majority_vote(all_seq_pred_json)
        else:
            mc_res = all_seq_pred_json

        imid_results_mapping = self.convert_coco_result_to_bdd(
            mc_res, bdd_cid_cinfo_mapping, imid_iminfo_mapping, vid_vinfo_mapping,
        )

        if self.with_mask:
            scalabel_results = self.overlapping_masks_removal(imid_results_mapping)
        else:
            scalabel_results = list(imid_results_mapping.values())

        return scalabel_results

    def overlapping_masks_removal(self, imid_results_mapping, nproc=NPROC):

        with Pool(nproc) as pool:
            print("\nCollecting mask information")
            mask_infors = pool.map(
                mask_prepare, tqdm.tqdm(list(imid_results_mapping.values()))
            )

            print("\nRemoving overlaps and retrieving valid masks and indexes.")
            results = pool.starmap(mask_postprocess, mask_infors)

        return results

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

    def convert_pred_to_label_format(self, coco_pred, bdd_cid_cinfo_mapping):
        """
        convert the single prediction result to label format for bdd

        coco_pred:
            'image_id': 1,
             'bbox': [998.872802734375,
              379.5665283203125,
              35.427490234375,
              59.21759033203125],
             'score': 0.9133418202400208,
             'category_id': 1,
             'video_id': 1,
             'track_id': 16

        - labels [ ]: list of dicts
            - id: string
            - category: string
            - box2d:
               - x1: float
               - y1: float
               - x2: float
               - y2: float
        Args:
            coco_pred: coco_pred dict.
            bdd_cid_cinfo_mapping: bdd category id to category infomation mapping.
        Return:
            a new dict in bdd format.
        """
        new_label = {}
        new_label["id"] = coco_pred["track_id"]
        new_label["score"] = coco_pred["score"]
        new_label["category"] = bdd_cid_cinfo_mapping[coco_pred["category_id"]]["name"]
        new_label["box2d"] = {
            "x1": coco_pred["bbox"][0],
            "y1": coco_pred["bbox"][1],
            "x2": coco_pred["bbox"][0] + coco_pred["bbox"][2],
            "y2": coco_pred["bbox"][1] + coco_pred["bbox"][3],
        }
        if "segmentation" in coco_pred:
            new_label["rle"] = coco_pred["segmentation"]

        return new_label

    def convert_coco_result_to_bdd(
        self, new_pred, bdd_cid_cinfo_mapping, imid_iminfo_mapping, vid_vinfo_mapping
    ):
        """
        Args:
            new_pred: list of coco predictions
            bdd_cid_cinfo_mapping: bdd category id to category infomation mapping.
        Return:
            submitable result for bdd eval
        """

        imid_new_dict_mapping = {}
        for item in tqdm.tqdm(new_pred):
            imid = item["image_id"]
            if imid not in imid_new_dict_mapping:
                new_dict = {}
                new_dict["name"] = imid_iminfo_mapping[imid]["file_name"]
                new_dict["videoName"] = vid_vinfo_mapping[
                    imid_iminfo_mapping[imid]["video_id"]
                ]["name"]
                new_dict["frameIndex"] = imid_iminfo_mapping[imid]["frame_id"]
                new_dict["labels"] = [
                    self.convert_pred_to_label_format(item, bdd_cid_cinfo_mapping)
                ]
                imid_new_dict_mapping[imid] = new_dict
            else:
                imid_new_dict_mapping[imid]["labels"].append(
                    self.convert_pred_to_label_format(item, bdd_cid_cinfo_mapping)
                )
        for key in imid_iminfo_mapping:
            if key not in imid_new_dict_mapping:
                new_dict = {}
                new_dict["name"] = imid_iminfo_mapping[key]["file_name"]
                new_dict["videoName"] = vid_vinfo_mapping[
                    imid_iminfo_mapping[key]["video_id"]
                ]["name"]
                new_dict["frameIndex"] = imid_iminfo_mapping[key]["frame_id"]
                new_dict["labels"] = []
                imid_new_dict_mapping[key] = new_dict

        return imid_new_dict_mapping

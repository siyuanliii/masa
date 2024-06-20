import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import json
import gc
#import resource
import argparse
import cv2
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry
from utils import filter_and_update_tracks

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--save_dir', type=str, help='Output for video frames')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path', type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument('--json_output', type=str, default='output.json', help='Output path for JSON file')
    args = parser.parse_args()
    return args

def tensor_to_list(tensor):
    return tensor.cpu().numpy().tolist()

def track_result_to_dict(track_result):
    track_instances = track_result[0].pred_track_instances
    data_dict = {
        "bboxes": tensor_to_list(track_instances.bboxes),
        "scores": tensor_to_list(track_instances.scores),
        "labels": tensor_to_list(track_instances.labels),
        "masks": [tensor_to_list(mask) for mask in track_instances.masks] if 'masks' in track_instances else None,
        "instances_id": tensor_to_list(track_instances.instances_id)
    }
    return data_dict

def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        # build test pipeline
        det_model.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    if args.sam_mask:
        print('Loading SAM model...')
        device = args.device
        sam_model = sam_model_registry[args.sam_type](args.sam_path)
        sam_predictor = SamPredictor(sam_model.to(device))

    video_reader = mmcv.VideoReader(args.video)

    #### parsing the text input
    texts = args.texts
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)

    if texts is not None:
        masa_model.cfg.visualizer['texts'] = texts
    else:
        masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']

    frame_idx = 0
    instances_list = []
    for frame in track_iter_progress((video_reader, len(video_reader))):

        # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
        if args.unified:
            track_result = inference_masa(masa_model, frame,
                                          frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          text_prompt=texts,
                                          fp16=args.fp16,
                                          detector_type=args.detector_type)
        else:
            if args.detector_type == 'mmdet':
                result = inference_detector(det_model, frame,
                                            text_prompt=texts,
                                            test_pipeline=test_pipeline,
                                            fp16=args.fp16)

            # Perfom inter-class NMS to remove nosiy detections
            det_bboxes, keep_idx = batched_nms(boxes=result.pred_instances.bboxes,
                                               scores=result.pred_instances.scores,
                                               idxs=result.pred_instances.labels,
                                               class_agnostic=True,
                                               nms_cfg=dict(type='nms',
                                                             iou_threshold=0.5,
                                                             class_agnostic=True,
                                                             split_thr=100000))

            det_bboxes = torch.cat([det_bboxes,
                                    result.pred_instances.scores[keep_idx].unsqueeze(1)],
                                    dim=1)
            det_labels = result.pred_instances.labels[keep_idx]

            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          det_bboxes=det_bboxes,
                                          det_labels=det_labels,
                                          fp16=args.fp16)

        frame_idx += 1
        if 'masks' in track_result[0].pred_track_instances:
            if len(track_result[0].pred_track_instances.masks) > 0:
                track_result[0].pred_track_instances.masks = torch.stack(track_result[0].pred_track_instances.masks, dim=0)
                track_result[0].pred_track_instances.masks = track_result[0].pred_track_instances.masks.cpu().numpy()

        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        gc.collect()

    if not args.no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))

    # 将 instances_list 转换为字典列表
    instances_dict_list = [track_result_to_dict(result) for result in instances_list]

    # 将结果保存为 JSON 文件
    json_output_path = args.json_output
    with open(json_output_path, 'w') as json_file:
        json.dump(instances_dict_list, json_file)

    print('Done')

if __name__ == '__main__':
    main()

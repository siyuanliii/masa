import torch
import numpy as np
from collections import defaultdict

from mmdet.models.task_modules.assigners import BboxOverlaps2D
from mmengine.structures import InstanceData
def average_score_filter(instances_list):
    # Extract instance IDs and their scores
    instance_id_to_frames = defaultdict(list)
    instance_id_to_scores = defaultdict(list)
    for frame_idx, instances in enumerate(instances_list):
        for i, instance_id in enumerate(instances[0].pred_track_instances.instances_id):
            instance_id_to_frames[instance_id.item()].append(frame_idx)
            instance_id_to_scores[instance_id.item()].append(instances[0].pred_track_instances.scores[i].cpu().numpy())

    # Compute average scores for each segment of each instance ID
    for instance_id, frames in instance_id_to_frames.items():
        scores = np.array(instance_id_to_scores[instance_id])

        # Identify segments
        segments = []
        segment = [frames[0]]
        for idx in range(1, len(frames)):
            if frames[idx] == frames[idx - 1] + 1:
                segment.append(frames[idx])
            else:
                segments.append(segment)
                segment = [frames[idx]]
        segments.append(segment)

        # Compute average score for each segment
        avg_scores = np.copy(scores)
        for segment in segments:
            segment_scores = scores[frames.index(segment[0]):frames.index(segment[-1]) + 1]
            avg_score = np.mean(segment_scores)
            avg_scores[frames.index(segment[0]):frames.index(segment[-1]) + 1] = avg_score

        # Update instances_list with average scores
        for frame_idx, avg_score in zip(frames, avg_scores):
            instances_list[frame_idx][0].pred_track_instances.scores[
                instances_list[frame_idx][0].pred_track_instances.instances_id == instance_id] = torch.tensor(avg_score, dtype=instances_list[frame_idx][0].pred_track_instances.scores.dtype)

    return instances_list


def moving_average_filter(instances_list, window_size=5):
    # Helper function to compute the moving average
    def smooth_bbox(bboxes, window_size):
        smoothed_bboxes = np.copy(bboxes)
        half_window = window_size // 2
        for i in range(4):
            padded_bboxes = np.pad(bboxes[:, i], (half_window, half_window), mode='edge')
            smoothed_bboxes[:, i] = np.convolve(padded_bboxes, np.ones(window_size) / window_size, mode='valid')
        return smoothed_bboxes

    # Extract bounding boxes and instance IDs
    instance_id_to_frames = defaultdict(list)
    instance_id_to_bboxes = defaultdict(list)
    for frame_idx, instances in enumerate(instances_list):
        for i, instance_id in enumerate(instances[0].pred_track_instances.instances_id):
            instance_id_to_frames[instance_id.item()].append(frame_idx)
            instance_id_to_bboxes[instance_id.item()].append(instances[0].pred_track_instances.bboxes[i].cpu().numpy())

    # Apply moving average filter to each segment
    for instance_id, frames in instance_id_to_frames.items():
        bboxes = np.array(instance_id_to_bboxes[instance_id])

        # Identify segments
        segments = []
        segment = [frames[0]]
        for idx in range(1, len(frames)):
            if frames[idx] == frames[idx - 1] + 1:
                segment.append(frames[idx])
            else:
                segments.append(segment)
                segment = [frames[idx]]
        segments.append(segment)

        # Smooth bounding boxes for each segment
        smoothed_bboxes = np.copy(bboxes)
        for segment in segments:
            if len(segment) >= window_size:
                segment_bboxes = bboxes[frames.index(segment[0]):frames.index(segment[-1]) + 1]
                smoothed_segment_bboxes = smooth_bbox(segment_bboxes, window_size)
                smoothed_bboxes[frames.index(segment[0]):frames.index(segment[-1]) + 1] = smoothed_segment_bboxes

        # Update instances_list with smoothed bounding boxes
        for frame_idx, smoothed_bbox in zip(frames, smoothed_bboxes):
            instances_list[frame_idx][0].pred_track_instances.bboxes[
                instances_list[frame_idx][0].pred_track_instances.instances_id == instance_id] = torch.tensor(smoothed_bbox, dtype=instances_list[frame_idx][0].pred_track_instances.bboxes.dtype).to(instances_list[frame_idx][0].pred_track_instances.bboxes.device)

    return instances_list


def identify_and_remove_giant_bounding_boxes(instances_list, image_size, size_threshold, confidence_threshold,
                                             coverage_threshold, object_num_thr=4, max_objects_in_box=6):
    # Initialize BboxOverlaps2D with 'iof' mode
    bbox_overlaps_calculator = BboxOverlaps2D()

    # Initialize data structures
    invalid_instance_ids = set()

    image_width, image_height = image_size
    two_thirds_image_area = (2 / 3) * (image_width * image_height)

    # Step 1: Identify giant bounding boxes and record their instance_ids
    for frame_idx, instances in enumerate(instances_list):
        bounding_boxes = instances[0].pred_track_instances.bboxes
        confidence_scores = instances[0].pred_track_instances.scores
        instance_ids = instances[0].pred_track_instances.instances_id

        N = bounding_boxes.size(0)

        for i in range(N):
            current_box = bounding_boxes[i]
            box_size = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])

            if box_size < size_threshold:
                continue

            other_boxes = torch.cat([bounding_boxes[:i], bounding_boxes[i + 1:]])
            other_confidences = torch.cat([confidence_scores[:i], confidence_scores[i + 1:]])
            iofs = bbox_overlaps_calculator(other_boxes, current_box.unsqueeze(0), mode='iof', is_aligned=False)

            if iofs.numel() == 0:
                continue

            high_conf_mask = other_confidences > confidence_threshold

            if high_conf_mask.numel() == 0 or torch.sum(high_conf_mask) == 0:
                continue

            high_conf_masked_iofs = iofs[high_conf_mask]

            covered_high_conf_boxes_count = torch.sum(high_conf_masked_iofs > coverage_threshold)

            if covered_high_conf_boxes_count >= object_num_thr and torch.all(
                    confidence_scores[i] < other_confidences[high_conf_mask]):
                invalid_instance_ids.add(instance_ids[i].item())
                continue

            if box_size > two_thirds_image_area:
                invalid_instance_ids.add(instance_ids[i].item())
                continue

            # New condition: if the bounding box contains more than 6 objects
            if covered_high_conf_boxes_count > max_objects_in_box:
                invalid_instance_ids.add(instance_ids[i].item())
                continue

    # Remove invalid tracks
    for frame_idx, instances in enumerate(instances_list):
        valid_mask = torch.tensor(
            [instance_id.item() not in invalid_instance_ids for instance_id in
             instances[0].pred_track_instances.instances_id])
        if len(valid_mask) == 0:
            continue
        new_instance_data = InstanceData()
        new_instance_data.bboxes = instances[0].pred_track_instances.bboxes[valid_mask]
        new_instance_data.scores = instances[0].pred_track_instances.scores[valid_mask]
        new_instance_data.instances_id = instances[0].pred_track_instances.instances_id[valid_mask]
        new_instance_data.labels = instances[0].pred_track_instances.labels[valid_mask]
        if 'masks' in instances[0].pred_track_instances:
            new_instance_data.masks = instances[0].pred_track_instances.masks[valid_mask]
        instances[0].pred_track_instances = new_instance_data

    return instances_list


def filter_and_update_tracks(instances_list, image_size, size_threshold=10000, coverage_threshold=0.75,
                             confidence_threshold=0.2, smoothing_window_size=5):

    # Step 1: Identify and remove giant bounding boxes
    instances_list = identify_and_remove_giant_bounding_boxes(instances_list, image_size, size_threshold, confidence_threshold, coverage_threshold)

     # Step 2: Smooth interpolated bounding boxes
    instances_list = moving_average_filter(instances_list, window_size=smoothing_window_size)

    # Step 3: compute the track average score
    instances_list = average_score_filter(instances_list)


    return instances_list
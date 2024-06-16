import numpy as np
from pycocotools import mask as mask_utils

SHAPE = [720, 1280]


def mask_prepare(track_dict):
    scores, masks = [], []
    labels = track_dict["labels"]
    for instance in labels:
        masks.append(mask_utils.decode(instance["rle"]))
        scores.append(instance["score"])
    return scores, masks, track_dict


def mask_postprocess(scores, masks, track_dict):
    sorted_idxs = np.argsort(scores)[::-1]  # Sort indices in descending order of scores
    processed_area = np.zeros(
        SHAPE, dtype=np.uint8
    )  # Empty mask to record processed areas

    for idx in sorted_idxs:
        current_mask = masks[idx]
        # Remove overlapping parts with already processed areas
        current_mask = np.where(processed_area, 0, current_mask)
        if current_mask.sum() > 0:  # Only keep non-empty masks
            # Update processed area
            processed_area = np.maximum(processed_area, current_mask)

        masks[idx] = current_mask

    valid_rle_masks = [
        mask_utils.encode(np.asfortranarray(masks[idx]))
        for idx in sorted_idxs
        if masks[idx].sum() > 0
    ]
    valid_idxs = [idx for idx in sorted_idxs if masks[idx].sum() > 0]

    valid_track_dicts = track_dict.copy()

    valid_labels = []
    for i in range(len(valid_idxs)):
        vidx = valid_idxs[i]
        if isinstance(valid_rle_masks[i]["counts"], bytes):
            valid_rle_masks[i]["counts"] = valid_rle_masks[i]["counts"].decode()
        valid_track_dicts["labels"][vidx]["rle"] = valid_rle_masks[i]
        valid_labels.append(valid_track_dicts["labels"][vidx])

    valid_track_dicts["labels"] = valid_labels

    return valid_track_dicts

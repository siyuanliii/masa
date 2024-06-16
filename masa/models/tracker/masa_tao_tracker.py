"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.trackers.base_tracker import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor


@MODELS.register_module()
class MasaTaoTracker(BaseTracker):
    """Tracker for MASA on TAO benchmark.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        distractor_score_thr (float): The score threshold to consider an object as a distractor.
            Defaults to 0.5.
        distractor_nms_thr (float): The NMS threshold for filtering out distractors.
            Defaults to 0.3.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Can be 'bisoftmax', 'softmax', or 'cosine'. Defaults to 'bisoftmax'.
        max_distance (float): Maximum distance for considering matches. Defaults to -1.
        fps (int): Frames per second of the input video. Used for calculating growth factor. Defaults to 1.
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_score_thr: float = 0.5,
        memo_tracklet_frames: int = 10,
        memo_momentum: float = 0.8,
        distractor_score_thr: float = 0.5,
        distractor_nms_thr=0.3,
        with_cats: bool = True,
        max_distance: float = -1,
        fps=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_momentum = memo_momentum
        self.distractor_score_thr = distractor_score_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.with_cats = with_cats

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.max_distance = max_distance  # Maximum distance for considering matches
        self.fps = fps
        self.growth_factor = self.fps / 6  # Growth factor for the distance mask
        self.distance_smoothing_factor = 100 / self.fps

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        for id, bbox, embed, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                self.tracks[id]["bbox"] = bbox
                self.tracks[id]["embed"] = (1 - self.memo_momentum) * self.tracks[id][
                    "embed"
                ] + self.memo_momentum * embed
                self.tracks[id]["last_frame"] = frame_id
                self.tracks[id]["label"] = label
                self.tracks[id]["score"] = score
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                )

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_frame_ids = []

        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_frame_ids.append(v["last_frame"])

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_frame_ids = torch.tensor(memo_frame_ids, dtype=torch.long).view(1, -1)

        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_ids.squeeze(0),
            memo_frame_ids.squeeze(0),
        )

    def compute_distance_mask(self, bboxes1, bboxes2, frame_ids1, frame_ids2):
        """Compute a mask based on the pairwise center distances and frame IDs with piecewise soft-weighting."""
        centers1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
        centers2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0
        distances = torch.cdist(centers1, centers2)

        frame_id_diff = torch.abs(frame_ids1[:, None] - frame_ids2[None, :]).to(
            distances.device
        )

        # Define a scaling factor for the distance based on frame difference (exponential growth)
        scaling_factor = torch.exp(frame_id_diff.float() / self.growth_factor)

        # Apply the scaling factor to max_distance
        adaptive_max_distance = self.max_distance * scaling_factor

        # Create a piecewise function for soft gating
        soft_distance_mask = torch.where(
            distances <= adaptive_max_distance,
            torch.ones_like(distances),
            torch.exp(
                -(distances - adaptive_max_distance) / self.distance_smoothing_factor
            ),
        )

        return soft_distance_mask

    def track(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        feats: List[torch.Tensor],
        data_sample: TrackDataSample,
        rescale=True,
        with_segm=False,
        **kwargs
    ) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        frame_id = metainfo.get("frame_id", -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            pred_track_instances.mask_inds = torch.zeros_like(labels)
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(metainfo["scale_factor"]).repeat(
                (1, 2)
            )
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])
        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = track_feats[inds, :]
        if with_segm:
            mask_inds = torch.arange(bboxes.size(0)).to(embeds.device)
            mask_inds = mask_inds[inds]
        else:
            mask_inds = []

        bboxes, labels, scores, embeds, mask_inds = self.remove_distractor(
            bboxes,
            labels,
            scores,
            track_feats=embeds,
            mask_inds=mask_inds,
            nms="inter",
            distractor_score_thr=self.distractor_score_thr,
            distractor_nms_thr=self.distractor_nms_thr,
        )

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_ids,
                memo_frame_ids,
            ) = self.memo

            feats = torch.mm(embeds, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            match_scores_bisoftmax = (d2t_scores + t2d_scores) / 2

            match_scores_cosine = torch.mm(
                F.normalize(embeds, p=2, dim=1),
                F.normalize(memo_embeds, p=2, dim=1).t(),
            )

            match_scores = (match_scores_bisoftmax + match_scores_cosine) / 2

            if self.max_distance != -1:

                # Compute the mask based on spatial proximity
                current_frame_ids = torch.full(
                    (bboxes.size(0),), frame_id, dtype=torch.long
                )
                distance_mask = self.compute_distance_mask(
                    bboxes, memo_bboxes, current_frame_ids, memo_frame_ids
                )

                # Apply the mask to the match scores
                match_scores = match_scores * distance_mask

            # track according to match_scores
            for i in range(bboxes.size(0)):
                conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        # keep bboxes with high object score
                        # and remove background bboxes
                        if scores[i] > self.obj_score_thr:
                            ids[i] = id
                            match_scores[:i, memo_ind] = 0
                            match_scores[i + 1 :, memo_ind] = 0

        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id)
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]
        if with_segm:
            pred_track_instances.mask_inds = mask_inds[tracklet_inds]

        return pred_track_instances

    def remove_distractor(
        self,
        bboxes,
        labels,
        scores,
        track_feats,
        mask_inds=[],
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        nms="inter",
    ):
        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(scores < distractor_score_thr, as_tuple=False).squeeze(
            1
        )
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes[:, :])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes)
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        if track_feats is not None:
            track_feats = track_feats[valid_inds]

        if len(mask_inds) > 0:
            mask_inds = mask_inds[valid_inds]

        return bboxes, labels, scores, track_feats, mask_inds

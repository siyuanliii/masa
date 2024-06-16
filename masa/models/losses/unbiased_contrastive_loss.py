"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

import torch
import torch.nn as nn
from mmdet.models import weight_reduce_loss
from mmdet.registry import MODELS


def multi_pos_cross_entropy(
    pred, label, weight=None, reduction="mean", avg_factor=None, pos_normalize=True,
):

    valid_mask = label.sum(1) != 0
    pred = pred[valid_mask]
    label = label[valid_mask]
    weight = weight[valid_mask]
    if min(pred.shape) != 0:
        logits_max, _ = torch.max(pred, dim=1, keepdim=True)
        logits = pred - logits_max.detach()
    else:
        logits = pred

    if pos_normalize:
        pos_norm = torch.div(label, label.sum(1).reshape(-1, 1))
        exp_logits = (torch.exp(logits)) * pos_norm + (
            torch.exp(logits)
        ) * torch.logical_not(label)
    else:
        exp_logits = torch.exp(logits)
    exp_logits_input = exp_logits.sum(1, keepdim=True)
    log_prob = logits - torch.log(exp_logits_input)

    mean_log_prob_pos = (label * log_prob).sum(1) / label.sum(1)
    loss = -mean_log_prob_pos

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


@MODELS.register_module()
class UnbiasedContrastLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(UnbiasedContrastLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_cls

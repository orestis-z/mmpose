# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_overlaps


@MODELS.register_module()
class IoULoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 reduction='mean',
                 mode='log',
                 eps: float = 1e-16,
                 loss_weight=1.):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        assert mode in ('linear', 'square', 'log'), f'the argument ' \
            f'`reduction` should be either \'linear\', \'square\' or ' \
            f'\'log\', but got {mode}'

        self.reduction = reduction
        self.criterion = partial(F.cross_entropy, reduction='none')
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
        """
        ious = bbox_overlaps(
            output, target, is_aligned=True).clamp(min=self.eps)

        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious.pow(2)
        elif self.mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError

        if target_weight is not None:
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight

@MODELS.register_module()
class CIoULoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 mode='log',
                 eps: float = 1e-16,
                 loss_weight=1.):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'The argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        assert mode in ('linear', 'square', 'log'), f'The argument ' \
            f'`mode` should be either \'linear\', \'square\' or ' \
            f'\'log\', but got {mode}'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, pred_bbox, target_bbox, target_weight=None):
        ious = bbox_overlaps(
            pred_bbox, target_bbox, is_aligned=True).clamp(min=self.eps)

        pred_center_x = (pred_bbox[:, 0] + pred_bbox[:, 2]) / 2
        pred_center_y = (pred_bbox[:, 1] + pred_bbox[:, 3]) / 2
        target_center_x = (target_bbox[:, 0] + target_bbox[:, 2]) / 2
        target_center_y = (target_bbox[:, 1] + target_bbox[:, 3]) / 2

        center_dist = (pred_center_x - target_center_x).pow(2) + \
                      (pred_center_y - target_center_y).pow(2)

        enclose_x1 = torch.min(pred_bbox[:, 0], target_bbox[:, 0])
        enclose_y1 = torch.min(pred_bbox[:, 1], target_bbox[:, 1])
        enclose_x2 = torch.max(pred_bbox[:, 2], target_bbox[:, 2])
        enclose_y2 = torch.max(pred_bbox[:, 3], target_bbox[:, 3])

        enclose_diag = (enclose_x2 - enclose_x1).pow(2) + \
                       (enclose_y2 - enclose_y1).pow(2) + self.eps

        pred_w = pred_bbox[:, 2] - pred_bbox[:, 0] + self.eps
        pred_h = pred_bbox[:, 3] - pred_bbox[:, 1] + self.eps
        target_w = target_bbox[:, 2] - target_bbox[:, 0] + self.eps
        target_h = target_bbox[:, 3] - target_bbox[:, 1] + self.eps

        v = (4 / (torch.pi ** 2)) * \
            (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)).pow(2)

        alpha = v / (1 - ious + v + self.eps)

        cious = ious - (center_dist / enclose_diag) - (alpha * v)

        if self.mode == 'linear':
            loss = 1 - cious
        elif self.mode == 'square':
            loss = 1 - cious.pow(2)
        elif self.mode == 'log':
            loss = -cious.log()
        else:
            raise NotImplementedError

        if target_weight is not None:
            for _ in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        intersection = (
            pred * target
        ).sum()

        dice = (
            2.0 * intersection + 1
        ) / (
            pred.sum()
            + target.sum()
            + 1
        )

        return 1 - dice
    
class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred,
        target
    ):

        pred = torch.sigmoid(pred)

        pred_edge = (
            pred[:, :, 1:, :]
            - pred[:, :, :-1, :]
        ).abs()

        target_edge = (
            target[:, :, 1:, :]
            - target[:, :, :-1, :]
        ).abs()

        return F.l1_loss(
            pred_edge,
            target_edge
        )
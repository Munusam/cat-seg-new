import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.fuse = nn.Sequential(

            nn.Conv2d(
                2048,
                512,
                kernel_size=1
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        res3,
        res4,
        res5
    ):

        res4_up = F.interpolate(
            res4,
            size=res3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        res5_up = F.interpolate(
            res5,
            size=res3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        fused = torch.cat(
            [
                res3,
                res4_up,
                res5_up
            ],
            dim=1
        )

        fused = self.fuse(
            fused
        )

        return fused
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


class UNetPPBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

    def load_pretrained(
        self,
        ckpt_path
    ):

        state_dict = torch.load(
            ckpt_path,
            map_location="cpu"
        )

        self.model.load_state_dict(
            state_dict
        )

        print(
            f"Loaded UNet++ weights from {ckpt_path}"
        )

    def forward(
        self,
        x
    ):

        return self.model(x)
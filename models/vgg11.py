"""VGG11 encoder

Architecture follows the official VGG11 paper (Simonyan & Zisserman, 2015).
We modernise it with:
  - BatchNorm2d after every Conv2d.

Input: 224x224 images (hardcoded per VGG paper).
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d -> BatchNorm2d -> ReLU block (core VGG building block)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional encoder with optional skip-feature returns.

    The 8 convolutional layers of VGG11 are grouped into 5 blocks separated
    by MaxPool2d (stride 2).  Spatial sizes for 224x224 input:
        block1 -> 112x112   (64 ch)
        block2 -> 56x56     (128 ch)
        block3 -> 28x28     (256 ch)
        block4 -> 14x14     (512 ch)
        block5 -> 7x7       (512 ch)  <- bottleneck after AdaptiveAvgPool

    When return_features=True the output of each block *before* pooling is
    returned so that the U-Net decoder can use them as skip connections.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # ---- Convolutional blocks (VGG11 topology) ----
        # Block 1: 1 conv, 64 filters
        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 1 conv, 128 filters
        self.block2 = nn.Sequential(_conv_bn_relu(64, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 2 convs, 256 filters
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 2 convs, 512 filters
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 2 convs, 512 filters
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pool -> fixed 7x7 bottleneck (standard VGG)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where
              feature_dict keys are 'block1'..'block5' (pre-pool feature maps).
        """
        f1 = self.block1(x)  # [B,  64, 224, 224]
        x = self.pool1(f1)  # [B,  64, 112, 112]

        f2 = self.block2(x)  # [B, 128, 112, 112]
        x = self.pool2(f2)  # [B, 128,  56,  56]

        f3 = self.block3(x)  # [B, 256,  56,  56]
        x = self.pool3(f3)  # [B, 256,  28,  28]

        f4 = self.block4(x)  # [B, 512,  28,  28]
        x = self.pool4(f4)  # [B, 512,  14,  14]

        f5 = self.block5(x)  # [B, 512,  14,  14]
        x = self.pool5(f5)  # [B, 512,   7,   7]

        bottleneck = self.avgpool(x)  # [B, 512,   7,   7]

        if return_features:
            features = {
                "block1": f1,
                "block2": f2,
                "block3": f3,
                "block4": f4,
                "block5": f5,
            }
            return bottleneck, features

        return bottleneck


# Alias required by autograder: from models.vgg11 import VGG11
VGG11 = VGG11Encoder

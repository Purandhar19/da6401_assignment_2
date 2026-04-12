"""Segmentation model - U-Net style with VGG11 encoder"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """Single U-Net decoder stage:
    TransposedConv (upsample 2x) → concat skip → Conv-BN-ReLU x2
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Learnable upsampling — transposed conv, mandatory per assignment
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After concat with skip: channels = in_ch//2 + skip_ch
        self.conv = nn.Sequential(
            _conv_bn_relu(in_ch // 2 + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network using VGG11 as contracting path.

    Encoder (VGG11):
        block1: 64  ch, 224x224
        block2: 128 ch, 112x112
        block3: 256 ch,  56x56
        block4: 512 ch,  28x28
        block5: 512 ch,  14x14
        bottleneck: 512 ch, 7x7

    Decoder (symmetric, transposed convolutions):
        up5: 7x7   → 14x14,  concat block5 skip
        up4: 14x14 → 28x28,  concat block4 skip
        up3: 28x28 → 56x56,  concat block3 skip
        up2: 56x56 → 112x112, concat block2 skip
        up1: 112x112 → 224x224, concat block1 skip

    Loss: CrossEntropyLoss (3 classes: bg, fg, boundary).
    Justification: CE is standard for multi-class pixel classification.
    Dice loss alone can be unstable early in training; CE provides stable
    gradients from the start and handles the 3-class trimap naturally.

    Dropout is placed after the bottleneck projection only — adding dropout
    in every decoder block was found empirically to slow convergence without
    meaningful regularisation benefit at the pixel level.
    """

    def __init__(
        self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5
    ):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes (3 for trimap).
            in_channels: Number of input channels.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Bottleneck projection (7x7 → prepare for decoder)
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(512, 512),
            CustomDropout(p=dropout_p),
        )

        # Decoder blocks (in_ch, skip_ch, out_ch)
        self.dec5 = DecoderBlock(512, 512, 256)  # 7→14,  skip=block5(512)
        self.dec4 = DecoderBlock(256, 512, 256)  # 14→28, skip=block4(512)
        self.dec3 = DecoderBlock(256, 256, 128)  # 28→56, skip=block3(256)
        self.dec2 = DecoderBlock(128, 128, 64)  # 56→112,skip=block2(128)
        self.dec1 = DecoderBlock(64, 64, 64)  # 112→224,skip=block1(64)

        # Final 1x1 conv → class logits
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor [B, in_channels, H, W].
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder with skip connections
        bottleneck, feats = self.encoder(x, return_features=True)

        # Bottleneck
        z = self.bottleneck(bottleneck)  # [B, 512, 7, 7]

        # Decoder
        z = self.dec5(z, feats["block5"])  # [B, 256, 14, 14]
        z = self.dec4(z, feats["block4"])  # [B, 256, 28, 28]
        z = self.dec3(z, feats["block3"])  # [B, 128, 56, 56]
        z = self.dec2(z, feats["block2"])  # [B,  64, 112,112]
        z = self.dec1(z, feats["block1"])  # [B,  64, 224,224]

        return self.head(z)  # [B, num_classes, 224, 224]

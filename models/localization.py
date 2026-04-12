"""Localization modules"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Encoder: pretrained VGG11 convolutional backbone (fine-tuned).
    Decoder: regression head outputting [x_center, y_center, width, height]
             in pixel space (not normalised) for a 224x224 input image.

    Design choices:
    - Fine-tuning the backbone (not freezing): the pet bounding boxes
      correlate strongly with high-level semantic features (fur texture,
      face shape) that are encoded in deeper conv layers. Freezing would
      discard this signal.
    - ReLU activation on the output is NOT used; instead we use Sigmoid
      scaled to image size so the output is bounded and smooth.
    - Loss: MSE + IoULoss (as required by README).
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),  # output in [0, 1]; scale to pixel space below
        )

        # Fixed image size per VGG paper
        self.image_size = 224.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalised).
        """
        features = self.encoder(x, return_features=False)  # [B, 512, 7, 7]
        out = self.regressor(features)  # [B, 4] in [0,1]
        # Scale to pixel coordinates
        return out * self.image_size  # [B, 4] pixel space

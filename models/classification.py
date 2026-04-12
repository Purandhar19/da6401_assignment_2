"""Classification components"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    The classification head mirrors the original VGG paper:
        FC(512*7*7 -> 4096) -> BN -> ReLU -> Dropout
        FC(4096 -> 4096)    -> BN -> ReLU -> Dropout
        FC(4096 -> num_classes)

    Design justification:
    - BatchNorm1d is placed *before* ReLU in the FC layers so normalisation
      acts on pre-activations, which keeps them centred and avoids the
      dead-ReLU problem at scale.
    - CustomDropout is placed *after* the non-linearity (post-ReLU) so that
      only active units are randomly silenced; this matches the canonical
      dropout paper by Srivastava et al. (2014) and gives the clearest
      regularisation signal on the 4096-dim representations.
    - Dropout is NOT placed inside the convolutional blocks because BatchNorm
      already regularises those layers, and combining both has been shown
      empirically to destabilise training (Li et al., 2019).
    """

    def __init__(
        self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5
    ):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head (VGG-style 3-layer FC)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x, return_features=False)  # [B, 512, 7, 7]
        logits = self.classifier(features)  # [B, num_classes]
        return logits

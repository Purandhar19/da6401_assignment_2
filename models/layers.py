"""Reusable custom layers"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implemented from scratch.

    Uses inverted dropout scaling so that the expected value of activations
    is preserved at both train and test time. During inference (self.training=False)
    the input is returned unchanged.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        # p=0.0 is valid: forward() will return input unchanged
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        At training time:
          1. Sample a binary mask from Bernoulli(1 - p).
          2. Multiply input by the mask.
          3. Scale by 1/(1-p)  <-- inverted dropout so no scaling needed at test time.

        At eval time:
          Return x unchanged.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor same shape as x.
        """
        if not self.training or self.p == 0.0:
            return x

        # Bernoulli mask: 1 with prob (1-p), 0 with prob p
        keep_prob = 1.0 - self.p
        # Create mask on same device/dtype as input
        mask = torch.bernoulli(
            torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        # Inverted dropout scaling
        return x * mask / keep_prob

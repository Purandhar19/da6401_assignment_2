"""Inference and evaluation script for the multi-task perception pipeline."""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from multitask import MultiTaskPerceptionModel


def predict(image_path: str, device: str = "cpu"):
    """Run full pipeline inference on a single image.

    Args:
        image_path: Path to input image.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with keys 'classification', 'localization', 'segmentation'.
    """
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    img = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0)

    model = MultiTaskPerceptionModel()
    model.eval()

    with torch.no_grad():
        output = model(tensor.to(device))

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = predict(sys.argv[1])
        print("Classification logits shape:", result["classification"].shape)
        print("Localization bbox:", result["localization"])
        print("Segmentation mask shape:", result["segmentation"].shape)

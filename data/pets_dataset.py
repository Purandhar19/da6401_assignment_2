"""Dataset skeleton for Oxford-IIIT Pet."""

import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Oxford-IIIT Pet: 37 breeds. Class index = (species_id - 1).
# The annotation file lists: Image CLASS-ID SPECIES BREED-ID
# CLASS-ID is 1-indexed (1..37).


def get_transforms(image_size: int = 224, augment: bool = True):
    """Return albumentations transform pipeline."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augment:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Args:
        root: Path to the dataset root (contains 'images/' and 'annotations/').
        split: 'train' or 'val' or 'test'.
        image_size: Resize target (default 224 for VGG).
        augment: Apply augmentations (train only).
        task: 'classification' | 'localization' | 'segmentation' | 'all'.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
        task: str = "classification",
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.task = task
        self.transform = get_transforms(
            image_size, augment=(augment and split == "train")
        )

        # Load annotation list file
        ann_file = (
            self.root
            / "annotations"
            / ("trainval.txt" if split in ("train", "val") else "test.txt")
        )
        entries = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name, class_id = parts[0], int(parts[1])
                entries.append((name, class_id - 1))  # 0-indexed

        # 90/10 train/val split from trainval
        if split in ("train", "val"):
            np.random.seed(42)
            idx = np.random.permutation(len(entries))
            cut = int(0.9 * len(entries))
            if split == "train":
                entries = [entries[i] for i in idx[:cut]]
            else:
                entries = [entries[i] for i in idx[cut:]]

        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _load_image(self, name: str) -> np.ndarray:
        path = self.root / "images" / f"{name}.jpg"
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _load_bbox(self, name: str, orig_w: int, orig_h: int):
        """Load bounding box in pixel (x_center, y_center, width, height) format."""
        bbox_file = self.root / "annotations" / "xmls" / f"{name}.xml"
        if not bbox_file.exists():
            return None
        import xml.etree.ElementTree as ET

        tree = ET.parse(bbox_file)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        # Scale to resized image coordinates
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        xmin, xmax = xmin * scale_x, xmax * scale_x
        ymin, ymax = ymin * scale_y, ymax * scale_y
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def _load_mask(self, name: str) -> np.ndarray:
        """Load trimap mask. Values: 1=foreground, 2=background, 3=not classified."""
        path = self.root / "annotations" / "trimaps" / f"{name}.png"
        mask = np.array(Image.open(path))
        # Convert to 0-indexed: 0=bg, 1=fg, 2=boundary
        mask = mask - 1
        mask = np.clip(mask, 0, 2)
        return mask

    def __getitem__(self, idx):
        name, class_id = self.entries[idx]
        img_np = self._load_image(name)
        orig_h, orig_w = img_np.shape[:2]

        transformed = self.transform(image=img_np)
        image = transformed["image"]  # [3, H, W] float tensor

        sample = {"image": image, "label": torch.tensor(class_id, dtype=torch.long)}

        if self.task in ("localization", "all"):
            bbox = self._load_bbox(name, orig_w, orig_h)
            sample["bbox"] = bbox if bbox is not None else torch.zeros(4)

        if self.task in ("segmentation", "all"):
            mask_np = self._load_mask(name)
            mask_resized = np.array(
                Image.fromarray(mask_np.astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                )
            )
            sample["mask"] = torch.from_numpy(mask_resized).long()

        return sample

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Loads weights from classifier.pth, localizer.pth, unet.pth and
    initialises a single shared VGG11 backbone + three task heads.
    A single forward pass yields classification logits, bbox, and seg mask.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super().__init__()

        import gdown

        gdown.download(
            id="1ifL6yMyZoG1p7gQvpHy8WjSRIw9URd4Q", output=classifier_path, quiet=False
        )
        gdown.download(
            id="1x4nsxpeqqFQeD0hFRt-gBTt5bYPe9JxO", output=localizer_path, quiet=False
        )
        gdown.download(
            id="17KEFeWRrFl6fdaX4rH8zQYZQSrnxqkU8", output=unet_path, quiet=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load individual trained models to extract weights
        clf = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _load(model, path):
            ckpt = torch.load(path, map_location=device)
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)

        _load(clf, classifier_path)
        _load(loc, localizer_path)
        _load(unet, unet_path)

        # Shared backbone from classifier
        self.encoder = clf.encoder

        # Classification head
        self.cls_head = clf.classifier

        # Localization head
        self.loc_head = loc.regressor
        self.image_size = 224.0

        # Segmentation head
        self.bottleneck = unet.bottleneck
        self.dec5 = unet.dec5
        self.dec4 = unet.dec4
        self.dec3 = unet.dec3
        self.dec2 = unet.dec2
        self.dec1 = unet.dec1
        self.seg_head = unet.head

    def forward(self, x: torch.Tensor):
        """Single forward pass through shared backbone → three heads.

        Args:
            x: [B, in_channels, H, W] normalised input tensor.

        Returns:
            dict with keys:
                'classification': [B, num_breeds] logits
                'localization':   [B, 4] bbox in pixel space
                'segmentation':   [B, seg_classes, H, W] logits
        """
        bottleneck, feats = self.encoder(x, return_features=True)

        # Classification
        cls_out = self.cls_head(bottleneck)

        # Localization
        loc_out = self.loc_head(bottleneck) * self.image_size

        # Segmentation
        z = self.bottleneck(bottleneck)
        z = self.dec5(z, feats["block5"])
        z = self.dec4(z, feats["block4"])
        z = self.dec3(z, feats["block3"])
        z = self.dec2(z, feats["block2"])
        z = self.dec1(z, feats["block1"])
        seg_out = self.seg_head(z)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }

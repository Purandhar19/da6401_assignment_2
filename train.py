"""train.py — DA6401 Assignment 2 — Single training entrypoint for ALL tasks.

Covers:
  Task 1  : VGG11 classification
  Task 2  : Bounding box localization
  Task 3  : U-Net segmentation (3 transfer learning strategies)
  W&B 2.1 : BatchNorm activation distribution study
  W&B 2.2 : Dropout generalisation gap study
  W&B 2.3 : Transfer learning showdown (frozen / partial / full finetune)
  W&B 2.4 : Feature map visualisation
  W&B 2.5 : BBox prediction table with GT vs predicted boxes
  W&B 2.6 : Segmentation sample logging (pixel acc vs dice)
  W&B 2.7 : Wild image inference pipeline showcase

Usage:
    python train.py --data_root /path/to/pets --run all        # everything
    python train.py --data_root /path/to/pets --run 2.1
    python train.py --data_root /path/to/pets --run 2.2
    python train.py --data_root /path/to/pets --run task1
    python train.py --data_root /path/to/pets --run task2
    python train.py --data_root /path/to/pets --run task3
    python train.py --data_root /path/to/pets --run 2.4
    python train.py --data_root /path/to/pets --run 2.7 --wild_image_dir ./wild_images
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.layers import CustomDropout
from losses.iou_loss import IoULoss


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cls_loaders(data_root, batch_size=32):
    tr = OxfordIIITPetDataset(
        data_root, split="train", augment=True, task="classification"
    )
    vl = OxfordIIITPetDataset(
        data_root, split="val", augment=False, task="classification"
    )
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(vl, batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def get_loc_loaders(data_root, batch_size=32):
    tr = OxfordIIITPetDataset(
        data_root, split="train", augment=True, task="localization"
    )
    vl = OxfordIIITPetDataset(
        data_root, split="val", augment=False, task="localization"
    )
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(vl, batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def get_seg_loaders(data_root, batch_size=16):
    tr = OxfordIIITPetDataset(
        data_root, split="train", augment=True, task="segmentation"
    )
    vl = OxfordIIITPetDataset(
        data_root, split="val", augment=False, task="segmentation"
    )
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(vl, batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def load_encoder_weights(model, classifier_ckpt, device):
    if os.path.exists(classifier_ckpt):
        ckpt = torch.load(classifier_ckpt, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        enc_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        model.encoder.load_state_dict(enc_sd, strict=False)
        print(f"  Loaded encoder weights from {classifier_ckpt}")


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════


def compute_iou_batch(pred, target, eps=1e-6):
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)
    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    inter = iw * ih
    union = (
        (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        + (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        - inter
        + eps
    )
    return inter / union


def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    dice = 0.0
    for c in range(num_classes):
        p = (pred_mask == c).float()
        t = (true_mask == c).float()
        dice += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return (dice / num_classes).item()


def pixel_accuracy(pred_mask, true_mask):
    return (pred_mask == true_mask).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
#  NO-BN MODEL (experiment 2.1 only)
# ══════════════════════════════════════════════════════════════════════════════


def _conv_relu(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))


class VGG11EncoderNoBN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = nn.Sequential(_conv_relu(in_channels, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(_conv_relu(64, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = nn.Sequential(_conv_relu(128, 256), _conv_relu(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(_conv_relu(256, 512), _conv_relu(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = nn.Sequential(_conv_relu(512, 512), _conv_relu(512, 512))
        self.pool5 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        return self.avgpool(x)


class VGG11ClassifierNoBN(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5):
        super().__init__()
        self.encoder = VGG11EncoderNoBN(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def cls_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def cls_eval(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    all_preds = []
    all_labels = []
    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, total_correct / total, f1


def run_cls_experiment(
    run_name,
    data_root,
    model,
    lr,
    epochs,
    batch_size,
    wandb_project,
    config,
    save_path=None,
    log_activations=False,
):
    device = get_device()
    model = model.to(device)
    train_dl, val_dl = get_cls_loaders(data_root, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    wandb.init(project=wandb_project, name=run_name, config=config, reinit=True)

    hook_outputs = []
    handle = None
    if log_activations:

        def _hook(m, i, o):
            hook_outputs.append(o.detach().cpu())

        try:
            hook_layer = model.encoder.block3[0][0]
        except:
            hook_layer = model.encoder.block3[0]
        handle = hook_layer.register_forward_hook(_hook)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = cls_train_epoch(model, train_dl, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1 = cls_eval(model, val_dl, criterion, device)
        scheduler.step()
        log = {
            "epoch": epoch,
            "train/loss": tr_loss,
            "train/accuracy": tr_acc,
            "val/loss": vl_loss,
            "val/accuracy": vl_acc,
            "val/macro_f1": vl_f1,
            "lr": scheduler.get_last_lr()[0],
        }
        if log_activations and hook_outputs:
            acts = hook_outputs[-1].float().numpy().ravel()
            log["activations/block3_conv1"] = wandb.Histogram(acts)
            hook_outputs.clear()
        wandb.log(log)
        print(
            f"[{run_name}] Epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
            f"| vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} vl_f1={vl_f1:.3f}"
        )
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_f1,
                    },
                    save_path,
                )
    if handle:
        handle.remove()
    wandb.finish()
    return best_f1


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — LOCALIZATION
# ══════════════════════════════════════════════════════════════════════════════


def loc_train_epoch(model, loader, mse_fn, iou_fn, optimizer, device):
    model.train()
    total_loss = total_iou = n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)
        valid = bboxes.sum(dim=1) > 0
        if valid.sum() == 0:
            continue
        imgs, bboxes = imgs[valid], bboxes[valid]
        optimizer.zero_grad()
        pred = model(imgs)
        loss = mse_fn(pred, bboxes) + iou_fn(pred, bboxes)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_iou += compute_iou_batch(pred.detach(), bboxes).sum().item()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(n, 1), total_iou / max(n, 1)


@torch.no_grad()
def loc_eval(model, loader, mse_fn, iou_fn, device):
    model.eval()
    total_loss = total_iou = n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)
        valid = bboxes.sum(dim=1) > 0
        if valid.sum() == 0:
            continue
        imgs, bboxes = imgs[valid], bboxes[valid]
        pred = model(imgs)
        loss = mse_fn(pred, bboxes) + iou_fn(pred, bboxes)
        total_iou += compute_iou_batch(pred, bboxes).sum().item()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(n, 1), total_iou / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════


def seg_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_dice = total_acc = n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        pred = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_score(pred, masks) * imgs.size(0)
        total_acc += pixel_accuracy(pred, masks) * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n, total_dice / n, total_acc / n


@torch.no_grad()
def seg_eval(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_acc = n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        pred = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_score(pred, masks) * imgs.size(0)
        total_acc += pixel_accuracy(pred, masks) * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n, total_dice / n, total_acc / n


def freeze_encoder(model, mode):
    for p in model.encoder.parameters():
        p.requires_grad = False
    if mode == "partial":
        for part in [
            model.encoder.block4,
            model.encoder.pool4,
            model.encoder.block5,
            model.encoder.pool5,
            model.encoder.avgpool,
        ]:
            for p in part.parameters():
                p.requires_grad = True
    elif mode == "full_finetune":
        for p in model.encoder.parameters():
            p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
#  W&B LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def log_bbox_table(model, val_dl, device, wandb_project, n_samples=15):
    print("  Logging W&B bbox table (section 2.5)...")
    wandb.init(project=wandb_project, name="2.5_bbox_predictions", reinit=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    table = wandb.Table(columns=["image", "iou", "result"])
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            valid = bboxes.sum(dim=1) > 0
            if valid.sum() == 0:
                continue
            imgs, bboxes = imgs[valid], bboxes[valid]
            preds = model(imgs)
            ious = compute_iou_batch(preds, bboxes)
            for i in range(min(len(imgs), n_samples - count)):
                img_np = np.clip(
                    imgs[i].cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1
                )
                gt = bboxes[i].cpu().numpy()
                pr = preds[i].cpu().numpy()
                iou = ious[i].item()
                fig, ax = plt.subplots(1, figsize=(4, 4))
                ax.imshow(img_np)
                ax.add_patch(
                    patches.Rectangle(
                        (gt[0] - gt[2] / 2, gt[1] - gt[3] / 2),
                        gt[2],
                        gt[3],
                        linewidth=2,
                        edgecolor="green",
                        facecolor="none",
                    )
                )
                ax.add_patch(
                    patches.Rectangle(
                        (pr[0] - pr[2] / 2, pr[1] - pr[3] / 2),
                        pr[2],
                        pr[3],
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
                ax.set_title(f"IoU={iou:.3f}")
                ax.axis("off")
                result = "good" if iou > 0.5 else ("miss" if iou < 0.1 else "partial")
                table.add_data(wandb.Image(fig), f"{iou:.3f}", result)
                plt.close(fig)
                count += 1
            if count >= n_samples:
                break
    wandb.log({"2.5_bbox_table": table})
    wandb.finish()


def log_seg_samples(model, val_dl, device, wandb_project, n=5):
    print("  Logging W&B seg samples (section 2.6)...")
    wandb.init(project=wandb_project, name="2.6_seg_samples", reinit=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    colors = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128]], dtype=np.uint8)
    table = wandb.Table(
        columns=["original", "ground_truth", "prediction", "pixel_acc", "dice"]
    )
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(imgs).argmax(1)
            for i in range(min(len(imgs), n - count)):
                img_np = np.clip(
                    imgs[i].cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1
                )
                table.add_data(
                    wandb.Image(img_np),
                    wandb.Image(colors[masks[i].cpu().numpy()]),
                    wandb.Image(colors[preds[i].cpu().numpy()]),
                    f"{pixel_accuracy(preds[i:i+1],masks[i:i+1]):.3f}",
                    f"{dice_score(preds[i:i+1],masks[i:i+1]):.3f}",
                )
                count += 1
            if count >= n:
                break
    wandb.log({"2.6_seg_samples": table})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNERS
# ══════════════════════════════════════════════════════════════════════════════


def experiment_2_1(data_root, wandb_project, epochs, batch_size):
    print("\n" + "=" * 60 + "\nEXPERIMENT 2.1: BatchNorm Activation Study\n" + "=" * 60)
    base = dict(task="2.1", lr=1e-3, epochs=epochs, batch_size=batch_size)
    run_cls_experiment(
        "2.1_with_batchnorm",
        data_root,
        VGG11Classifier(37, dropout_p=0.5),
        1e-3,
        epochs,
        batch_size,
        wandb_project,
        {**base, "batchnorm": True},
        log_activations=True,
    )
    run_cls_experiment(
        "2.1_without_batchnorm",
        data_root,
        VGG11ClassifierNoBN(37, dropout_p=0.5),
        1e-3,
        epochs,
        batch_size,
        wandb_project,
        {**base, "batchnorm": False},
        log_activations=True,
    )


def experiment_2_2(data_root, wandb_project, epochs, batch_size):
    print(
        "\n" + "=" * 60 + "\nEXPERIMENT 2.2: Dropout Generalisation Study\n" + "=" * 60
    )
    for run_name, dp in [
        ("2.2_no_dropout", 0.0),
        ("2.2_dropout_p0.2", 0.2),
        ("2.2_dropout_p0.5", 0.5),
    ]:
        run_cls_experiment(
            run_name,
            data_root,
            VGG11Classifier(37, dropout_p=dp),
            1e-3,
            epochs,
            batch_size,
            wandb_project,
            dict(task="2.2", dropout_p=dp, epochs=epochs),
        )


def train_task1(data_root, wandb_project, epochs, batch_size):
    print("\n" + "=" * 60 + "\nTASK 1: Final classifier\n" + "=" * 60)
    run_cls_experiment(
        "task1_final_classifier",
        data_root,
        VGG11Classifier(37, dropout_p=0.5),
        1e-3,
        epochs,
        batch_size,
        wandb_project,
        dict(task="task1", dropout_p=0.5, epochs=epochs),
        save_path="checkpoints/classifier.pth",
    )


def train_task2(data_root, wandb_project, epochs, batch_size, classifier_ckpt):
    print("\n" + "=" * 60 + "\nTASK 2: Localization\n" + "=" * 60)
    device = get_device()
    model = VGG11Localizer(in_channels=3, dropout_p=0.5).to(device)
    load_encoder_weights(model, classifier_ckpt, device)
    mse_fn = nn.MSELoss()
    iou_fn = IoULoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_dl, val_dl = get_loc_loaders(data_root, batch_size)
    wandb.init(
        project=wandb_project,
        name="task2_localizer",
        config=dict(task="task2", epochs=epochs, lr=1e-4),
        reinit=True,
    )
    best_iou = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_iou = loc_train_epoch(
            model, train_dl, mse_fn, iou_fn, optimizer, device
        )
        vl_loss, vl_iou = loc_eval(model, val_dl, mse_fn, iou_fn, device)
        scheduler.step()
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/iou": tr_iou,
                "val/loss": vl_loss,
                "val/iou": vl_iou,
            }
        )
        print(
            f"[task2] Epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_iou={tr_iou:.3f} "
            f"| vl_loss={vl_loss:.4f} vl_iou={vl_iou:.3f}"
        )
        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_iou,
                },
                "checkpoints/localizer.pth",
            )
    wandb.finish()
    log_bbox_table(model, val_dl, device, wandb_project)
    print(f"Best val IoU: {best_iou:.4f}")


def experiment_2_4(data_root, wandb_project, classifier_ckpt):
    print("\n" + "=" * 60 + "\nSECTION 2.4: Feature Map Visualisation\n" + "=" * 60)
    device = get_device()
    model = VGG11Classifier(num_classes=37).to(device)
    if os.path.exists(classifier_ckpt):
        ckpt = torch.load(classifier_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    ds = OxfordIIITPetDataset(
        data_root, split="val", augment=False, task="classification"
    )
    img = ds[0]["image"].unsqueeze(0).to(device)
    activations = {}

    def make_hook(name):
        def hook(m, i, o):
            activations[name] = o.detach().cpu()

        return hook

    h1 = model.encoder.block1[0][0].register_forward_hook(make_hook("first_conv"))
    h2 = model.encoder.block5[1][0].register_forward_hook(make_hook("last_conv"))
    with torch.no_grad():
        model(img)
    h1.remove()
    h2.remove()
    wandb.init(project=wandb_project, name="2.4_feature_maps", reinit=True)
    for layer_name, feat in activations.items():
        feat = feat[0]
        n_show = min(16, feat.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f"Feature maps — {layer_name}", fontsize=13)
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                fm = feat[idx].numpy()
                fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)
                ax.imshow(fm, cmap="viridis")
            ax.axis("off")
        plt.tight_layout()
        wandb.log({f"2.4_{layer_name}": wandb.Image(fig)})
        plt.close(fig)
        print(f"  Logged {n_show} maps for {layer_name}")
    wandb.finish()


def run_seg_strategy(
    run_name,
    data_root,
    wandb_project,
    epochs,
    batch_size,
    freeze_mode,
    classifier_ckpt,
    save_path,
):
    device = get_device()
    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=0.5).to(device)
    load_encoder_weights(model, classifier_ckpt, device)
    freeze_encoder(model, freeze_mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{run_name}] Trainable: {trainable:,}/{total:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_dl, val_dl = get_seg_loaders(data_root, batch_size)
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=dict(task="2.3", freeze_mode=freeze_mode, epochs=epochs),
        reinit=True,
    )
    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice, tr_acc = seg_train_epoch(
            model, train_dl, criterion, optimizer, device
        )
        vl_loss, vl_dice, vl_acc = seg_eval(model, val_dl, criterion, device)
        scheduler.step()
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/dice": tr_dice,
                "train/pixel_acc": tr_acc,
                "val/loss": vl_loss,
                "val/dice": vl_dice,
                "val/pixel_acc": vl_acc,
            }
        )
        print(
            f"  [{run_name}] Epoch {epoch:3d} | "
            f"tr_loss={tr_loss:.4f} tr_dice={tr_dice:.3f} "
            f"| vl_loss={vl_loss:.4f} vl_dice={vl_dice:.3f} vl_acc={vl_acc:.3f}"
        )
        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_dice,
                },
                save_path,
            )
    wandb.finish()
    return best_dice, model, val_dl


def train_task3(data_root, wandb_project, epochs, batch_size, classifier_ckpt):
    print(
        "\n"
        + "=" * 60
        + "\nTASK 3 / SECTION 2.3: Segmentation + Transfer Learning\n"
        + "=" * 60
    )
    strategies = [
        ("2.3_frozen_backbone", "full", "checkpoints/unet_frozen.pth"),
        ("2.3_partial_finetune", "partial", "checkpoints/unet_partial.pth"),
        ("2.3_full_finetune", "full_finetune", "checkpoints/unet.pth"),
    ]
    best_model = None
    best_val_dl = None
    for run_name, freeze_mode, save_path in strategies:
        _, model, val_dl = run_seg_strategy(
            run_name,
            data_root,
            wandb_project,
            epochs,
            batch_size,
            freeze_mode,
            classifier_ckpt,
            save_path,
        )
        if freeze_mode == "full_finetune":
            best_model = model
            best_val_dl = val_dl
    if best_model is not None:
        log_seg_samples(best_model, best_val_dl, get_device(), wandb_project)


def experiment_2_7(
    wild_image_dir, wandb_project, classifier_ckpt, localizer_ckpt, unet_ckpt
):
    print("\n" + "=" * 60 + "\nSECTION 2.7: Wild Image Inference\n" + "=" * 60)
    from PIL import Image as PILImage
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    BREED_NAMES = [
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
        "Egyptian_Mau",
        "Maine_Coon",
        "Persian",
        "Ragdoll",
        "Russian_Blue",
        "Siamese",
        "Sphynx",
        "american_bulldog",
        "american_pit_bull_terrier",
        "basset_hound",
        "beagle",
        "boxer",
        "chihuahua",
        "english_cocker_spaniel",
        "english_setter",
        "german_shorthaired",
        "great_pyrenees",
        "havanese",
        "japanese_chin",
        "keeshond",
        "leonberger",
        "miniature_pinscher",
        "newfoundland",
        "pomeranian",
        "pug",
        "saint_bernard",
        "samoyed",
        "scottish_terrier",
        "shiba_inu",
        "staffordshire_bull_terrier",
        "wheaten_terrier",
        "yorkshire_terrier",
    ]
    SEG_COLORS = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128]], dtype=np.uint8)

    device = get_device()
    clf = VGG11Classifier(37).to(device)
    loc = VGG11Localizer().to(device)
    unet = VGG11UNet(3).to(device)

    def _load(model, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    _load(clf, classifier_ckpt)
    _load(loc, localizer_ckpt)
    _load(unet, unet_ckpt)
    clf.eval()
    loc.eval()
    unet.eval()

    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    image_paths = [
        str(p)
        for p in Path(wild_image_dir).glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    if not image_paths:
        print(f"  No images found in {wild_image_dir}. Add pet images there first.")
        return

    wandb.init(project=wandb_project, name="2.7_wild_images", reinit=True)
    table = wandb.Table(columns=["filename", "pipeline_output", "breed", "confidence"])

    with torch.no_grad():
        for img_path in image_paths:
            img_np = np.array(PILImage.open(img_path).convert("RGB"))
            tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
            bn, feats = clf.encoder(tensor, return_features=True)
            cls_out = clf.classifier(bn)
            loc_out = (loc.regressor(bn) * 224.0)[0].cpu().numpy()
            z = unet.bottleneck(bn)
            for dec, sk in zip(
                [unet.dec5, unet.dec4, unet.dec3, unet.dec2, unet.dec1],
                ["block5", "block4", "block3", "block2", "block1"],
            ):
                z = dec(z, feats[sk])
            seg_mask = unet.head(z).argmax(1)[0].cpu().numpy()
            probs = torch.softmax(cls_out[0], 0).cpu()
            breed = BREED_NAMES[probs.argmax().item()]
            conf = probs.max().item()
            orig = np.array(PILImage.fromarray(img_np).resize((224, 224)))
            b = loc_out
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"{Path(img_path).name}  |  {breed}  ({conf:.2f})")
            axes[0].imshow(orig)
            axes[0].add_patch(
                patches.Rectangle(
                    (b[0] - b[2] / 2, b[1] - b[3] / 2),
                    b[2],
                    b[3],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            axes[0].set_title("Detection")
            axes[0].axis("off")
            axes[1].imshow(SEG_COLORS[seg_mask])
            axes[1].set_title("Segmentation")
            axes[1].axis("off")
            axes[2].imshow((orig * 0.6 + SEG_COLORS[seg_mask] * 0.4).astype(np.uint8))
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.tight_layout()
            table.add_data(Path(img_path).name, wandb.Image(fig), breed, f"{conf:.3f}")
            plt.close(fig)
            print(f"  {Path(img_path).name}: {breed} ({conf:.3f})")

    wandb.log({"2.7_wild_images": table})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment 2")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--localizer_ckpt", type=str, default="checkpoints/localizer.pth")
    p.add_argument("--unet_ckpt", type=str, default="checkpoints/unet.pth")
    p.add_argument("--wild_image_dir", type=str, default="wild_images")
    p.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "2.1", "2.2", "task1", "task2", "2.4", "task3", "2.7"],
    )
    return p.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    if args.run in ("2.1", "all"):
        experiment_2_1(args.data_root, args.wandb_project, args.epochs, args.batch_size)
    if args.run in ("2.2", "all"):
        experiment_2_2(args.data_root, args.wandb_project, args.epochs, args.batch_size)
    if args.run in ("task1", "all"):
        train_task1(args.data_root, args.wandb_project, args.epochs, args.batch_size)
    if args.run in ("task2", "all"):
        train_task2(
            args.data_root,
            args.wandb_project,
            args.epochs,
            args.batch_size,
            args.classifier_ckpt,
        )
    if args.run in ("2.4", "all"):
        experiment_2_4(args.data_root, args.wandb_project, args.classifier_ckpt)
    if args.run in ("task3", "all"):
        train_task3(
            args.data_root,
            args.wandb_project,
            args.epochs,
            args.batch_size,
            args.classifier_ckpt,
        )
    if args.run in ("2.7", "all"):
        experiment_2_7(
            args.wild_image_dir,
            args.wandb_project,
            args.classifier_ckpt,
            args.localizer_ckpt,
            args.unet_ckpt,
        )

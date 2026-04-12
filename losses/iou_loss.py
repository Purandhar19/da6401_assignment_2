"""Custom IoU loss"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Loss = 1 - IoU, so range is [0, 1] as required.

    Boxes are in (x_center, y_center, width, height) format (pixel space).
    Internally converted to (x1, y1, x2, y2) for area calculations.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            eps: Small value to avoid division by zero.
            reduction: 'mean' | 'sum' | 'none'. Default is 'mean'.
        """
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes (x_center, y_center, width, height).
            target_boxes: [B, 4] target boxes   (x_center, y_center, width, height).

        Returns:
            Scalar loss if reduction is 'mean' or 'sum', else [B] tensor.
        """

        # ── convert cx,cy,w,h  →  x1,y1,x2,y2 ──────────────────────────────
        def to_xyxy(boxes):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return x1, y1, x2, y2

        px1, py1, px2, py2 = to_xyxy(pred_boxes)
        tx1, ty1, tx2, ty2 = to_xyxy(target_boxes)

        # ── intersection ─────────────────────────────────────────────────────
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # ── union ─────────────────────────────────────────────────────────────
        pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union_area = pred_area + target_area - inter_area + self.eps

        # ── IoU and loss ──────────────────────────────────────────────────────
        iou = inter_area / union_area  # [B]  in [0, 1]
        loss = 1.0 - iou  # [B]  in [0, 1]  ← required range

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss

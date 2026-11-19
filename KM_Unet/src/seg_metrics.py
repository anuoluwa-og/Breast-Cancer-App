import torch


@torch.no_grad()
def per_image_dice_iou(logits, target, thr=0.5, eps=1e-6):
    """Per-image Dice and IoU for a batch. logits/target: (B, 1, H, W). Returns two (B,) tensors."""
    pred = (torch.sigmoid(logits.float()) > thr).float().flatten(1)
    tgt = (target > 0.5).float().flatten(1)
    inter = (pred * tgt).sum(1)
    psum, tsum = pred.sum(1), tgt.sum(1)
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice, iou

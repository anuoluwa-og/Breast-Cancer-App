"""Augmentation configs for the BUSI ablation.

--aug baseline : RandomRotate90 + HorizontalFlip  (what the existing recipe uses)
--aug trad     : baseline + mild scaling, brightness/contrast, gamma
--aug cutmix   : trad + CutMix-style patch mixing applied on each batch (on GPU)

The configs are cumulative on purpose (each run adds one thing to the previous one).
"""
import math
import random

import albumentations as A
import numpy as np
import torch

AUG_CHOICES = ['baseline', 'trad', 'cutmix']


def build_transform(aug, size, train=True):
    resize = A.Resize(size, size)
    norm = A.Normalize()  # ImageNet mean/std on 0-255 input; dataset.py must NOT divide by 255 again
    if not train:
        return A.Compose([resize, norm])

    ts = [A.RandomRotate90(), A.HorizontalFlip()]
    if aug in ('trad', 'cutmix'):
        ts += [
            A.Affine(scale=(0.9, 1.1), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ]
    ts += [resize, norm]
    return A.Compose(ts)


def cutmix_batch(x, y, p=0.5):
    """Mask-consistent CutMix for segmentation.

    For each sample (with probability p) a random box is copied from another
    sample in the batch into the same location, for the image AND the mask.
    Box area ratio is 1 - lambda with lambda ~ Beta(1, 1), as in standard CutMix.
    """
    B, _, H, W = x.shape
    src = torch.randperm(B, device=x.device)
    x_out, y_out = x.clone(), y.clone()
    for i in range(B):
        j = int(src[i])
        if j == i or random.random() >= p:
            continue
        lam = np.random.beta(1.0, 1.0)
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        top, bot = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        left, right = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        if bot <= top or right <= left:
            continue
        x_out[i, :, top:bot, left:right] = x[j, :, top:bot, left:right]
        y_out[i, :, top:bot, left:right] = y[j, :, top:bot, left:right]
    return x_out, y_out

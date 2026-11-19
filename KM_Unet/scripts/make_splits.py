"""Build the BUSI-only dataset for the KMP-UNet ablation and a fixed 7:1:2 split.

Steps
  1. Read raw BUSI from KM_Unet/BUSI/{benign,malignant,normal}.
  2. Keep only images whose mask contains a lesion (this drops the 'normal'
     class, whose masks are blank). Images with several masks
     (e.g. 'benign (100)_mask.png' + 'benign (100)_mask_1.png') get the union.
  3. Find near-duplicate images with a perceptual hash and group them, so a
     group never straddles train/val/test.
  4. Stratify by class (benign/malignant) and split ~7:1:2 at group level.
  5. Write data/busi_lesion/{images,masks/0} and splits/busi_seed<seed>.json.

Usage (from KM_Unet/):
    python scripts/make_splits.py --seed 0
    python scripts/make_splits.py --seed 0 --hash_thresh 6
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def find_pairs(raw_dir):
    """Return {stem: dict(cls, image_path, mask_paths)} for every image that has >=1 mask file."""
    items = {}
    for cls in ['benign', 'malignant', 'normal']:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob('*.png')):
            if '_mask' in p.name:
                continue
            masks = sorted(cls_dir.glob(p.stem + '_mask*.png'))
            if masks:
                items[p.stem] = dict(cls=cls, image=p, masks=masks)
    return items


def union_mask(mask_paths):
    out = None
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if out is None:
            out = np.zeros_like(m)
        if m.shape != out.shape:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 0).astype(np.uint8) * 255)
    return out


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(ROOT / 'BUSI'))
    ap.add_argument('--out_data', default=str(ROOT / 'data' / 'busi_lesion'))
    ap.add_argument('--out_split_dir', default=str(ROOT / 'splits'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hash_thresh', type=int, default=4,
                    help='max pHash Hamming distance (64-bit) to call two images near-duplicates')
    args = ap.parse_args()

    raw_dir, out_data, split_dir = Path(args.raw_dir), Path(args.out_data), Path(args.out_split_dir)
    (out_data / 'images').mkdir(parents=True, exist_ok=True)
    (out_data / 'masks' / '0').mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    items = find_pairs(raw_dir)
    print(f'raw images with a mask file: {len(items)}')

    # 1) keep lesion images only, merge multi-masks
    keep, dropped_empty, multi = [], [], 0
    for stem, it in items.items():
        m = union_mask(it['masks'])
        if m.max() == 0:
            dropped_empty.append(stem)
            continue
        if len(it['masks']) > 1:
            multi += 1
        # filenames like 'benign (1).png' are awkward; use benign_0001
        num = int(re.search(r'\((\d+)\)', stem).group(1))
        new_id = f"{it['cls']}_{num:04d}"
        cv2.imwrite(str(out_data / 'images' / f'{new_id}.png'), cv2.imread(str(it['image'])))
        cv2.imwrite(str(out_data / 'masks' / '0' / f'{new_id}.png'), m)
        keep.append(dict(id=new_id, cls=it['cls'], src=str(it['image'])))
    print(f'kept {len(keep)} lesion images ({multi} had multiple masks merged); '
          f'dropped {len(dropped_empty)} with empty masks')

    # 2) near-duplicate groups (pHash on the image, plus ANDed with same class)
    hashes = []
    for k in keep:
        img = Image.open(out_data / 'images' / f"{k['id']}.png").convert('L')
        hashes.append(imagehash.phash(img, hash_size=8))
    uf = UnionFind(len(keep))
    pairs = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            d = hashes[i] - hashes[j]
            if d <= args.hash_thresh:
                uf.union(i, j)
                pairs.append((keep[i]['id'], keep[j]['id'], int(d)))
    groups = np.array([uf.find(i) for i in range(len(keep))])
    gsize = Counter(groups.tolist())
    n_multi_groups = sum(1 for v in gsize.values() if v > 1)
    print(f'near-duplicate pairs (pHash dist <= {args.hash_thresh}): {len(pairs)}; '
          f'groups with >1 image: {n_multi_groups}')

    # 3) stratified group split into 10 folds -> 7 train / 1 val / 2 test
    y = np.array([k['cls'] for k in keep])
    ids = np.array([k['id'] for k in keep])
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    folds = [idx for _, idx in sgkf.split(ids, y, groups)]
    test_idx = np.concatenate(folds[0:2])
    val_idx = folds[2]
    train_idx = np.concatenate(folds[3:])
    split = dict(train=sorted(ids[train_idx].tolist()),
                 val=sorted(ids[val_idx].tolist()),
                 test=sorted(ids[test_idx].tolist()))

    # sanity checks: disjoint, complete, no group straddles splits
    allids = split['train'] + split['val'] + split['test']
    assert len(allids) == len(set(allids)) == len(keep)
    where = {}
    for name, lst in split.items():
        for i in lst:
            where[i] = name
    for g in set(groups.tolist()):
        members = {where[ids[i]] for i in np.where(groups == g)[0]}
        assert len(members) == 1, f'group {g} leaks across {members}'

    report = dict(
        seed=args.seed, hash_thresh=args.hash_thresh,
        counts={k: len(v) for k, v in split.items()},
        class_counts={name: dict(Counter(i.split('_')[0] for i in lst)) for name, lst in split.items()},
        n_dropped_empty_mask=len(dropped_empty), n_multi_mask_merged=multi,
        near_duplicate_pairs=pairs,
    )
    split['meta'] = report
    out = split_dir / f'busi_seed{args.seed}.json'
    out.write_text(json.dumps(split, indent=1))
    print(json.dumps({k: report[k] for k in ['counts', 'class_counts']}, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

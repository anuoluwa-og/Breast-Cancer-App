"""Build the BUS-BRA dataset for KMP-UNet pretraining and a patient-level train/val split.

Mirrors scripts/make_splits.py so train.py can use the output unchanged:
  data/busbra/images/<id>.png
  data/busbra/masks/0/<id>.png      (binary 0/255)
  splits/busbra_seed<seed>.json     ({"train": [...], "val": [...], "meta": {...}})

There is no test split on purpose: BUS-BRA is only used for pretraining here, and the
real test set is the untouched BUSI test split from splits/busi_seed0.json.

All images from the same patient go to the same split (no patient leakage), and the
split is stratified by histology (benign/malignant).

Expected raw layout (adjust with the flags if your copy differs):
    <raw_dir>/Images/bus_0001-l.png
    <raw_dir>/Masks/mask_0001-l.png
    <raw_dir>/bus_data.csv        (needs an id column, a histology column, a patient/case column)

Usage (from KM_Unet/):
    python scripts/make_busbra.py --raw_dir /kaggle/input/<bus-bra-dataset>/BUS-BRA --seed 0
If a column name is wrong the script prints the columns it found and stops.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]


def find_mask(mask_dir, img_id):
    """BUS-BRA names masks mask_XXXX-l.png for bus_XXXX-l.png; fall back to the same name."""
    for name in (img_id.replace('bus_', 'mask_', 1), img_id):
        p = mask_dir / f'{name}.png'
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--csv', default='bus_data.csv', help='metadata CSV, relative to --raw_dir')
    ap.add_argument('--img_subdir', default='Images')
    ap.add_argument('--mask_subdir', default='Masks')
    ap.add_argument('--id_col', default='ID')
    ap.add_argument('--label_col', default='Histology')
    ap.add_argument('--patient_col', default='Case', help='column that identifies the patient')
    ap.add_argument('--out_data', default=str(ROOT / 'data' / 'busbra'))
    ap.add_argument('--out_split_dir', default=str(ROOT / 'splits'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val_folds', type=int, default=1, help='out of 10 folds (1 -> ~90/10 train/val)')
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    img_dir, mask_dir = raw / args.img_subdir, raw / args.mask_subdir
    out_data, split_dir = Path(args.out_data), Path(args.out_split_dir)
    (out_data / 'images').mkdir(parents=True, exist_ok=True)
    (out_data / 'masks' / '0').mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw / args.csv)
    need = [args.id_col, args.label_col, args.patient_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f'Columns {missing} not found in {args.csv}. Columns present: {list(df.columns)}. '
                         f'Pass --id_col / --label_col / --patient_col.')

    keep, skipped = [], Counter()
    for _, row in df.iterrows():
        img_id = str(row[args.id_col])
        ip = img_dir / f'{img_id}.png'
        mp = find_mask(mask_dir, img_id)
        if not ip.exists():
            skipped['no_image'] += 1
            continue
        if mp is None:
            skipped['no_mask'] += 1
            continue
        img = cv2.imread(str(ip))
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            skipped['unreadable'] += 1
            continue
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            skipped['mask_resized'] += 1
        m = (m > 0).astype(np.uint8) * 255
        if m.max() == 0:
            skipped['empty_mask'] += 1
            continue
        cv2.imwrite(str(out_data / 'images' / f'{img_id}.png'), img)
        cv2.imwrite(str(out_data / 'masks' / '0' / f'{img_id}.png'), m)
        keep.append(dict(id=img_id, label=str(row[args.label_col]), patient=str(row[args.patient_col])))
    print(f'kept {len(keep)} of {len(df)} rows; notes: {dict(skipped)}')
    if not keep:
        raise SystemExit('nothing kept - check --raw_dir, --img_subdir and --mask_subdir')

    ids = np.array([k['id'] for k in keep])
    y = np.array([k['label'] for k in keep])
    groups = np.array([k['patient'] for k in keep])

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    folds = [idx for _, idx in sgkf.split(ids, y, groups)]
    val_idx = np.concatenate(folds[:args.val_folds])
    train_idx = np.concatenate(folds[args.val_folds:])
    split = dict(train=sorted(ids[train_idx].tolist()), val=sorted(ids[val_idx].tolist()))

    # sanity checks: disjoint, complete, no patient in both splits
    assert not set(split['train']) & set(split['val'])
    assert len(split['train']) + len(split['val']) == len(keep)
    assert not set(groups[train_idx]) & set(groups[val_idx]), 'a patient appears in both train and val'

    label_of = dict(zip(ids.tolist(), y.tolist()))
    split['meta'] = dict(
        seed=args.seed,
        counts={k: len(v) for k, v in split.items()},
        class_counts={k: dict(Counter(label_of[i] for i in split[k])) for k in ['train', 'val']},
        patients={'train': len(set(groups[train_idx])), 'val': len(set(groups[val_idx]))},
        skipped=dict(skipped),
    )
    out = split_dir / f'busbra_seed{args.seed}.json'
    out.write_text(json.dumps(split, indent=1))
    print(json.dumps(split['meta'], indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

"""Evaluate a finished run's model_best.pth. Defaults to the VAL split.

The TEST split is for the single final report only, so it needs --confirm_test:
    python evaluate.py --run outputs/final_seed0 --split_file splits/busi_seed0.json --split test --confirm_test
"""
import argparse
import json
import os
import sys

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
sys.path.insert(0, ROOT_DIR)
from src import archs
from src.augment import build_transform
from src.dataset import Dataset
from src.seg_metrics import per_image_dice_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir containing config.yml and model_best.pth')
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--confirm_test', action='store_true', help='required to touch the test split')
    ap.add_argument('--weights', default='model_best.pth')
    ap.add_argument('--batch_size', default=8, type=int)
    ap.add_argument('--num_workers', default=2, type=int)
    a = ap.parse_args()
    if a.split == 'test' and not a.confirm_test:
        sys.exit('Refusing to evaluate on the test split without --confirm_test (test set is used once).')

    cfg = yaml.safe_load(open(os.path.join(a.run, 'config.yml')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = archs.__dict__[cfg['arch']](cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
                                        base_channels=cfg['base_channels'], groups=cfg['groups'],
                                        d_state=cfg['d_state'], embed_dims=cfg['input_list'],
                                        no_kan=cfg['no_kan']).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, a.weights), map_location=device))
    model.eval()

    ids = json.load(open(a.split_file))[a.split]
    base = os.path.join(cfg['data_dir'], cfg['dataset'])
    ds = Dataset(ids, os.path.join(base, 'images'), os.path.join(base, 'masks'), '.png', '.png',
                 cfg['num_classes'], transform=build_transform('baseline', cfg['input_h'], train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    rows = []
    with torch.no_grad():
        for x, y, meta in loader:
            out = model(x.to(device))
            out = out[-1] if cfg['deep_supervision'] else out
            d, i = per_image_dice_iou(out, y.to(device))
            rows += [dict(id=n, dice=float(dd), iou=float(ii)) for n, dd, ii in zip(meta['img_id'], d.cpu(), i.cpu())]
    dice = torch.tensor([r['dice'] for r in rows])
    iou = torch.tensor([r['iou'] for r in rows])
    res = dict(split=a.split, n=len(rows), weights=a.weights,
               dice_mean=dice.mean().item(), dice_std=dice.std().item(),
               iou_mean=iou.mean().item(), iou_std=iou.std().item(), per_image=rows)
    out_path = os.path.join(a.run, f'eval_{a.split}.json')
    json.dump(res, open(out_path, 'w'), indent=1)
    print(f"{a.split}: n={res['n']}  Dice {res['dice_mean']:.4f} +- {res['dice_std']:.4f}  "
          f"IoU {res['iou_mean']:.4f} +- {res['iou_std']:.4f}  -> {out_path}")


if __name__ == '__main__':
    main()

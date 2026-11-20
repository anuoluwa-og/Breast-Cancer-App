import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import time

def push_checkpoint_to_kaggle(checkpoint_path, dataset_slug, message):
    """
    Uploads the given checkpoint file as a new version of a Kaggle Dataset,
    so training progress survives even if the notebook session dies without
    a graceful stop. Silently no-ops (with a printed warning) if the Kaggle
    API isn't configured or the push fails for any reason -- this must never
    crash the training loop.
    """
    import os
    import shutil
    import subprocess
    import json

    if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
        print('=> [kaggle push] skipped: KAGGLE_USERNAME/KAGGLE_KEY not set')
        return

    upload_dir = '/kaggle/working/_checkpoint_push'
    os.makedirs(upload_dir, exist_ok=True)
    shutil.copy(checkpoint_path, os.path.join(upload_dir, 'checkpoint_latest.pth'))

    metadata_path = os.path.join(upload_dir, 'dataset-metadata.json')
    if not os.path.exists(metadata_path):
        metadata = {
            "title": dataset_slug.split('/')[-1],
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    try:
        # First push ever needs `create`; afterwards `version` updates it.
        result = subprocess.run(
            ['kaggle', 'datasets', 'version', '-p', upload_dir, '-m', message],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0 and 'not found' in (result.stderr or '').lower():
            result = subprocess.run(
                ['kaggle', 'datasets', 'create', '-p', upload_dir],
                capture_output=True, text=True, timeout=180,
            )
        if result.returncode == 0:
            print(f'=> [kaggle push] checkpoint uploaded ({message})')
        else:
            print(f'=> [kaggle push] failed: {result.stderr.strip()[:300]}')
    except Exception as e:
        print(f'=> [kaggle push] failed with exception: {e}')

import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import archs
from src import losses
from src.dataset import Dataset
from src.augment import AUG_CHOICES, build_transform, cutmix_batch
from src.seg_metrics import per_image_dice_iou
from src.utils import AverageMeter, str2bool

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

LOSS_NAMES = list(losses.__all__) + ['BCEWithLogitsLoss']


def list_type(s):
    return [int(a) for a in s.split(',')] if s else []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='run name (output subfolder)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')
    parser.add_argument('--seed', default=1029, type=int, help='training seed (init, shuffling, augmentation)')
    parser.add_argument('--dataseed', default=2981, type=int, help='only used when no --split_file is given')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='KMPUNet')
    parser.add_argument('--base_channels', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)
    parser.add_argument('--d_state', default=16, type=int)

    parser.add_argument('--resume_from', default=None, type=str, help='path to checkpoint_latest.pth')
    parser.add_argument('--amp', action='store_true', help='mixed precision (fp16)')
    parser.add_argument('--kaggle_push_every', default=0, type=int)
    parser.add_argument('--kaggle_dataset_slug', default=None, type=str)

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int, help='final / evaluation width')
    parser.add_argument('--input_h', default=256, type=int, help='final / evaluation height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
    parser.add_argument('--prog_sizes', type=list_type, default=[],
                        help="progressive resizing, e.g. 128,192,256: training size steps through the "
                             "list in equal fractions of --epochs. Validation always uses input_h. "
                             "Empty = fixed input_h.")

    # experiment config
    parser.add_argument('--aug', default='baseline', choices=AUG_CHOICES,
                        help='baseline | trad (+scale/brightness/contrast/gamma) | cutmix (trad + CutMix)')
    parser.add_argument('--cutmix_p', default=0.5, type=float)
    parser.add_argument('--split_file', default=None, type=str,
                        help='JSON from scripts/make_splits.py. Only train+val ids are read here; '
                             'test ids are never loaded by this script.')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--dataset', default='busi_lesion')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='outputs')

    # optimizer
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N',
                        help='stop when val Dice has not improved for N epochs (counted only once the '
                             'final training size is reached). -1 = off.')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    return parser.parse_args()


def train(config, train_loader, model, criterion, optimizer, device, scaler):
    meters = {'loss': AverageMeter(), 'dice': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    data_t, comp_t = 0.0, 0.0
    pbar = tqdm(total=len(train_loader))
    t_end = time.perf_counter()
    for input, target, _ in train_loader:
        data_t += time.perf_counter() - t_end
        t0 = time.perf_counter()

        input, target = input.to(device), target.to(device)
        if config['aug'] == 'cutmix':
            input, target = cutmix_batch(input, target, p=config['cutmix_p'])

        optimizer.zero_grad()
        with autocast(enabled=config['amp']):
            if config['deep_supervision']:
                outputs = model(input)
                loss = sum(criterion(o, target) for o in outputs) / len(outputs)
                final_out = outputs[-1]
            else:
                final_out = model(input)
                loss = criterion(final_out, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice, iou = per_image_dice_iou(final_out.detach(), target)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        comp_t += time.perf_counter() - t0

        n = input.size(0)
        meters['loss'].update(loss.item(), n)
        meters['dice'].update(dice.mean().item(), n)
        meters['iou'].update(iou.mean().item(), n)
        pbar.set_postfix(OrderedDict([('loss', meters['loss'].avg), ('dice', meters['dice'].avg),
                                      ('data_s', round(data_t, 1)), ('compute_s', round(comp_t, 1))]))
        pbar.update(1)
        t_end = time.perf_counter()
    pbar.close()
    print(f"[TIMING] data {data_t:.1f}s | compute {comp_t:.1f}s")
    return OrderedDict([(k, m.avg) for k, m in meters.items()])


def validate(config, loader, model, criterion, device):
    """Per-image mean Dice / IoU (+ std) over the whole loader, in fp32."""
    model.eval()
    losses_, dices, ious = [], [], []
    with torch.no_grad():
        for input, target, _ in loader:
            input, target = input.to(device), target.to(device)
            out = model(input)
            out = out[-1] if config['deep_supervision'] else out
            losses_.append(criterion(out, target).item() * input.size(0))
            d, i = per_image_dice_iou(out, target)
            dices.append(d.cpu())
            ious.append(i.cpu())
    dices, ious = torch.cat(dices), torch.cat(ious)
    return OrderedDict([('loss', sum(losses_) / len(dices)),
                        ('dice', dices.mean().item()), ('dice_std', dices.std().item()),
                        ('iou', ious.mean().item()), ('iou_std', ious.std().item())])


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def size_schedule(config):
    sizes = config['prog_sizes'] or [config['input_h']]
    if sizes[-1] != config['input_h']:
        raise ValueError(f"last --prog_sizes entry ({sizes[-1]}) must equal --input_h ({config['input_h']})")
    return sizes


def size_for_epoch(sizes, epoch, total_epochs):
    return sizes[min(int(epoch / total_epochs * len(sizes)), len(sizes) - 1)]


def main():
    config = vars(parse_args())
    seed_torch(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_ext = mask_ext = '.png'
    config['mask_ext'] = mask_ext

    if config['name'] is None:
        config['name'] = '%s_%s_%s_seed%d' % (config['dataset'], config['arch'], config['aug'], config['seed'])
    exp_name, output_dir = config['name'], config['output_dir']
    run_dir = f'{output_dir}/{exp_name}'
    os.makedirs(run_dir, exist_ok=True)
    my_writer = SummaryWriter(run_dir)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    with open(f'{run_dir}/config.yml', 'w') as f:
        yaml.dump(config, f)

    cudnn.benchmark = True
    print(f'Using device: {device}')

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    model = archs.__dict__[config['arch']](
        config['num_classes'], config['input_channels'], config['deep_supervision'],
        base_channels=config['base_channels'], groups=config['groups'], d_state=config['d_state'],
        embed_dims=config['input_list'], no_kan=config['no_kan'],
    ).to(device)
    print('params: %.3fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():  # UKAN-style KAN lr (no match in KMPUNet)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    start_epoch = 0
    scaler = GradScaler(enabled=config['amp'])
    log = OrderedDict([(k, []) for k in ['epoch', 'size', 'lr', 'loss', 'dice', 'iou',
                                         'val_loss', 'val_dice', 'val_iou']])
    best_dice, best_epoch, bad_epochs = -1.0, -1, 0
    if config['resume_from']:
        print(f"=> resuming from checkpoint: {config['resume_from']}")
        ckpt = torch.load(config['resume_from'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        log = ckpt.get('log', log)
        best_dice = ckpt.get('best_dice', best_dice)
        best_epoch = ckpt.get('best_epoch', best_epoch)
        bad_epochs = ckpt.get('bad_epochs', bad_epochs)
        # restore RNG so a resumed run continues the same stream where possible
        print(f"=> resuming at epoch {start_epoch}/{config['epochs']} (best val dice so far {best_dice:.4f})")

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],
                                                   eta_min=config['min_lr'], last_epoch=start_epoch - 1)
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'], last_epoch=start_epoch - 1)
    else:
        scheduler = None

    shutil.copy2('train.py', f'{run_dir}/')
    for f in ['src/archs.py', 'src/augment.py']:
        shutil.copy2(f, f'{run_dir}/')

    # ---------------- data (train + val only; test ids are never read here) ----------------
    img_dir = os.path.join(config['data_dir'], config['dataset'], 'images')
    mask_dir = os.path.join(config['data_dir'], config['dataset'], 'masks')
    if config['split_file']:
        split = json.load(open(config['split_file']))
        train_ids, val_ids = split['train'], split['val']
        print(f"split file {config['split_file']}: train={len(train_ids)} val={len(val_ids)}")
    else:
        all_ids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob(os.path.join(img_dir, '*' + img_ext)))]
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=config['dataseed'])
        print(f'WARNING: no --split_file; using a random 80/20 train/val split of {len(all_ids)} images')

    sizes = size_schedule(config)
    final_stage_start = next(e for e in range(config['epochs'])
                             if size_for_epoch(sizes, e, config['epochs']) == sizes[-1])

    def make_ds(ids, transform):
        return Dataset(img_ids=ids, img_dir=img_dir, mask_dir=mask_dir, img_ext=img_ext, mask_ext=mask_ext,
                       num_classes=config['num_classes'], transform=transform)

    train_dataset = make_ds(train_ids, build_transform(config['aug'], sizes[0], train=True))
    val_dataset = make_ds(val_ids, build_transform(config['aug'], config['input_h'], train=False))
    # seeded generator so the shuffle order depends on --seed (and is reproducible)
    g = torch.Generator()
    g.manual_seed(config['seed'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=config['num_workers'], drop_last=True, generator=g)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'], drop_last=False)

    for epoch in range(start_epoch, config['epochs']):
        size = size_for_epoch(sizes, epoch, config['epochs'])
        # workers are re-created every epoch (persistent_workers=False), so the new transform is picked up
        train_dataset.transform = build_transform(config['aug'], size, train=True)
        print('Epoch [%d/%d]  train size %dx%d' % (epoch, config['epochs'], size, size))

        train_log = train(config, train_loader, model, criterion, optimizer, device, scaler)
        if scheduler is not None:
            scheduler.step()
        val_log = validate(config, val_loader, model, criterion, device)
        print('train loss %.4f dice %.4f | val loss %.4f dice %.4f iou %.4f' %
              (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice'], val_log['iou']))

        if val_log['dice'] > best_dice:
            best_dice, best_epoch, bad_epochs = val_log['dice'], epoch, 0
            torch.save(model.state_dict(), f'{run_dir}/model_best.pth')
            print(f'=> new best val dice {best_dice:.4f} (epoch {epoch}), saved model_best.pth')
        elif epoch >= final_stage_start:
            bad_epochs += 1

        for k, v in [('epoch', epoch), ('size', size), ('lr', [pg['lr'] for pg in optimizer.param_groups][0]),
                     ('loss', train_log['loss']), ('dice', train_log['dice']), ('iou', train_log['iou']),
                     ('val_loss', val_log['loss']), ('val_dice', val_log['dice']), ('val_iou', val_log['iou'])]:
            log.setdefault(k, []).append(v)
        pd.DataFrame(dict(log)).to_csv(f'{run_dir}/log.csv', index=False)
        for k in ['loss', 'dice', 'iou']:
            my_writer.add_scalar('train/' + k, train_log[k], global_step=epoch)
        for k in ['loss', 'dice', 'iou']:
            my_writer.add_scalar('val/' + k, val_log[k], global_step=epoch)

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'config': config,
                    'scaler_state_dict': scaler.state_dict(), 'log': dict(log),
                    'best_dice': best_dice, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs},
                   f'{run_dir}/checkpoint_latest.pth')
        if config['kaggle_push_every'] and config['kaggle_dataset_slug'] and (epoch + 1) % config['kaggle_push_every'] == 0:
            push_checkpoint_to_kaggle(f'{run_dir}/checkpoint_latest.pth', config['kaggle_dataset_slug'],
                                      message=f"{exp_name} epoch {epoch + 1}")
        torch.cuda.empty_cache()

        if config['early_stopping'] > 0 and bad_epochs >= config['early_stopping']:
            print(f"=> early stopping at epoch {epoch}: no val dice gain for {bad_epochs} epochs")
            break

    # final model = best-val-Dice checkpoint; also keep the last-epoch weights
    torch.save(model.state_dict(), f'{run_dir}/model_last.pth')
    model.load_state_dict(torch.load(f'{run_dir}/model_best.pth', map_location=device))
    final_val = validate(config, val_loader, model, criterion, device)
    final_val.update(best_epoch=best_epoch, epochs_run=len(log['epoch']))
    json.dump(final_val, open(f'{run_dir}/final_val.json', 'w'), indent=1)
    print('Best-val checkpoint (epoch %d): val dice %.4f +- %.4f  iou %.4f' %
          (best_epoch, final_val['dice'], final_val['dice_std'], final_val['iou']))
    my_writer.close()


if __name__ == '__main__':
    main()

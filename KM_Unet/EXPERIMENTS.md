# BUSI-only KMP-UNet ablation: how to run

Run from KM_Unet/ (Kaggle: after cloning the repo, `pip install -r requirements.txt` if needed).

## 0. One-time: build the dataset and split
    python scripts/make_splits.py --seed 0
Creates data/busi_lesion/ (647 lesion images, multi-masks merged) and splits/busi_seed0.json
(7:1:2, stratified benign/malignant, near-duplicates kept in the same split).

## 1. The four runs (same split, seed, epoch budget; only --aug changes)
COMMON="--dataset busi_lesion --data_dir data --split_file splits/busi_seed0.json \
  --input_h 256 --input_w 256 --prog_sizes 128,224,256 -b 32 --amp \
  --optimizer AdamW --lr 1e-3 --scheduler CosineAnnealingLR --epochs 100 --seed 0 \
  --kaggle_push_every 5 --kaggle_dataset_slug <YOUR_REAL_KAGGLE_USERNAME>/kmpunet-ckpt"

    python train.py --name run1_baseline_s0 --aug baseline $COMMON
    python train.py --name run2_trad_s0     --aug trad     $COMMON
    python train.py --name run3_cutmix_s0   --aug cutmix   $COMMON
    # run 4 (GAN pairs) not implemented yet

Resume after a Kaggle cut-off: add --resume_from outputs/<name>/checkpoint_latest.pth

## 2. Compare on VAL only
Each run writes outputs/<name>/log.csv, model_best.pth and final_val.json (per-image mean Dice/IoU on val).

## 3. Test: once, at the very end, for the final config
    python evaluate.py --run outputs/<final_name> --split_file splits/busi_seed0.json --split test --confirm_test

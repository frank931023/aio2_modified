"""
錯標檢測推論腳本（Image-level）
- 從 checkpoint 載入指定模型（預設 EMA-MEP）
- 對 val/test 資料集做推論（避免 train 的 overfitting 問題）
- 比較模型預測 mask 與 GT label 的差異
- 若差異區域中最大連通區域面積 > 圖片總像素 * threshold，判定該圖有錯標
- 輸出 CSV：每張圖的像素統計與 flagged 結果
- 輸出每張圖的預測 mask（tif 格式）
- 若提供 clean_seg_dir，額外計算 precision/recall/f1（以 ns vs clean seg 為 true label）
"""

import os
import sys
import logging
import argparse
import datetime
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append('./')

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from scipy import ndimage
import cv2

from dataset.data_loading_Building import BuildingDataset
from models import UNet


# ──────────────────────────────────────────────
# Logger 設定
# ──────────────────────────────────────────────

def setup_logger(log_path):
    logger = logging.getLogger('mislabel_detection')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # 輸出到 console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 輸出到 log 檔案
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────
# 載入模型
# ──────────────────────────────────────────────

def load_model(ckpt_path, device, model_key='model_state_dict_mep', logger=None):
    """
    從 checkpoint 載入模型權重。
    model_key 選項：
        'model_state_dict'     -> Student model
        'model_state_dict_mit' -> EMA-MIT（每個 iteration 更新的 teacher）
        'model_state_dict_mep' -> EMA-MEP（每個 epoch 更新的 teacher）← 預設
    """
    if logger:
        logger.info(f"Loading checkpoint: {ckpt_path}")
        logger.info(f"Using model key: {model_key}")

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 列出 checkpoint 內可用的模型 key
    available_keys = [k for k in ckpt.keys() if 'state_dict' in k]
    if logger:
        logger.info(f"Available model keys in checkpoint: {available_keys}")

    if model_key not in ckpt:
        raise KeyError(f"Key '{model_key}' not found. Available: {available_keys}")

    net = UNet(n_channels=3, n_classes=1)

    # 取出 state dict（mep 在原始程式碼中有時被存成 tuple，需處理）
    state_dict = ckpt[model_key]
    if isinstance(state_dict, tuple):
        state_dict = state_dict[0]

    # 移除 DataParallel 的 'module.' 前綴
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_sd)
    net.to(device)
    net.eval()

    epoch = ckpt.get('epoch', 'unknown')
    if logger:
        logger.info(f"Model loaded successfully. Checkpoint epoch: {epoch}")

    return net


# ──────────────────────────────────────────────
# 計算最大連通區域佔比
# ──────────────────────────────────────────────

def largest_cc_ratio(diff_mask):
    """
    輸入二值差異 mask（H x W，值為 0/1），
    回傳最大連通區域面積佔圖片總像素的比例。
    """
    total_pixels = diff_mask.size
    if diff_mask.sum() == 0:
        return 0.0

    labeled, num_features = ndimage.label(diff_mask)
    if num_features == 0:
        return 0.0

    cc_sizes = ndimage.sum(diff_mask, labeled, range(1, num_features + 1))
    return max(cc_sizes) / total_pixels


# ──────────────────────────────────────────────
# 主推論流程
# ──────────────────────────────────────────────

def run_inference(ckpt_path, data_path, noise_dir_name, splits,
                  output_csv, pred_save_dir,
                  threshold=0.50,
                  batch_size=25, num_workers=0,
                  model_key='model_state_dict_mep',
                  log_path=None, device=None):

    if log_path is None:
        log_path = output_csv.replace('.csv', '.log')

    # 避免重複加 handler（多次呼叫時）
    logger = logging.getLogger('mislabel_detection')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger = setup_logger(log_path)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Splits: {splits} (merged into one dataset)")
    logger.info(f"Mislabel threshold (largest CC ratio): {threshold}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Prediction tif save dir: {pred_save_dir}")

    # 建立預測 tif 輸出目錄
    os.makedirs(pred_save_dir, exist_ok=True)

    # 載入模型
    net = load_model(ckpt_path, device, model_key=model_key, logger=logger)

    # 合併多個 split 的資料集（val + test 放在一起，aug=False 避免 flip）
    datasets = []
    for split in splits:
        ds = BuildingDataset(data_path, noise_dir_name=noise_dir_name,
                             split=split, aug=False)
        logger.info(f"  {split} dataset size: {len(ds)} images")
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, drop_last=False)

    logger.info(f"Total images: {len(combined_dataset)}, batches: {len(loader)}")

    # CSV 欄位定義
    csv_columns = [
        'fname',                  # 檔案名稱
        'pred_positive_pixels',   # 模型預測為建築物（1）的像素數
        'pred_negative_pixels',   # 模型預測為背景（0）的像素數
        'gt_positive_pixels',     # GT 中建築物像素數
        'gt_negative_pixels',     # GT 中背景像素數
        'diff_pixels',            # pred 與 GT 差異的像素總數
        'diff_ratio',             # diff_pixels / 圖片總像素
        'largest_cc_ratio',       # 差異區域中最大連通區域佔比
        'flagged_mislabel',       # 是否被判定為有錯標（1=有，0=無）
    ]

    rows = []
    total_pixels_per_image = None
    total_flagged = 0
    t0 = datetime.datetime.now()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit='batch'):
            images   = batch['img'].to(device, dtype=torch.float32)
            gt_masks = torch.squeeze(batch['gt'], 1)   # (B, H, W)
            fnames   = batch['fname']

            # 前向推論
            logits     = net(images).squeeze(1)                        # (B, H, W)
            pred_masks = (torch.sigmoid(logits) > 0.5).float().cpu()  # 二值化

            B, H, W = pred_masks.shape[0], pred_masks.shape[1], pred_masks.shape[2]
            if total_pixels_per_image is None:
                total_pixels_per_image = H * W
                logger.info(f"Image size: {H}x{W} = {total_pixels_per_image} pixels/image")

            for i in range(B):
                pred  = pred_masks[i].numpy().astype(np.uint8)   # (H, W) 值為 0/1
                gt    = gt_masks[i].numpy().astype(np.uint8)
                fname = fnames[i]

                # 儲存預測 mask 為 tif（值 0/1，與原始 mask 格式一致）
                pred_tif_name = os.path.splitext(fname)[0] + '.tif'
                pred_tif_path = os.path.join(pred_save_dir, pred_tif_name)
                cv2.imwrite(pred_tif_path, pred)

                # 各類像素統計
                pred_pos = int(pred.sum())
                pred_neg = total_pixels_per_image - pred_pos
                gt_pos   = int(gt.sum())
                gt_neg   = total_pixels_per_image - gt_pos

                # pred 與 GT 的差異
                diff        = (pred != gt).astype(np.uint8)
                diff_pixels = int(diff.sum())
                diff_ratio  = diff_pixels / total_pixels_per_image

                # 最大連通區域佔比
                lcc_ratio = largest_cc_ratio(diff)

                # image-level 錯標判定
                flagged = int(lcc_ratio > threshold)
                total_flagged += flagged

                logger.debug(
                    f"{fname} | pred_pos={pred_pos} | gt_pos={gt_pos} | "
                    f"diff={diff_pixels} ({diff_ratio:.3f}) | "
                    f"lcc={lcc_ratio:.3f} | flagged={flagged}"
                )

                rows.append({
                    'fname':                fname,
                    'pred_positive_pixels': pred_pos,
                    'pred_negative_pixels': pred_neg,
                    'gt_positive_pixels':   gt_pos,
                    'gt_negative_pixels':   gt_neg,
                    'diff_pixels':          diff_pixels,
                    'diff_ratio':           round(diff_ratio, 6),
                    'largest_cc_ratio':     round(lcc_ratio, 6),
                    'flagged_mislabel':     flagged,
                })

    # 寫出 CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    total   = len(rows)
    elapsed = datetime.datetime.now() - t0

    logger.info("=" * 55)
    logger.info(f"Summary | Checkpoint: {os.path.basename(ckpt_path)}")
    logger.info(f"Total images    : {total}")
    logger.info(f"Flagged         : {total_flagged} ({total_flagged/total*100:.1f}%)")
    logger.info(f"Not flagged     : {total - total_flagged} ({(total-total_flagged)/total*100:.1f}%)")
    logger.info(f"Threshold used  : {threshold}")
    logger.info(f"Elapsed time    : {elapsed}")
    logger.info(f"CSV saved to    : {output_csv}")
    logger.info(f"Pred tifs saved : {pred_save_dir}")
    logger.info("=" * 55)


# ──────────────────────────────────────────────
# 參數設定
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='錯標檢測推論腳本')
    parser.add_argument('--data_path', type=str, required=True,
                        help='資料集根目錄（與訓練時相同）')
    parser.add_argument('--noise_dir_name', type=str, default='ns_seg_1',
                        help='noisy label 的資料夾名稱（例如 ns_seg_1）')
    parser.add_argument('--ckpt_correct', type=str,
                    default='D:/aio2_results_2/unet_ns_ns_seg/tdobj/mcr_1_seed_42/checkpoints/checkpoint_correct_mcr_1_epoch_50.pth',
                    help='correction 階段的 checkpoint 路徑')
    parser.add_argument('--ckpt_warmup', type=str,
                        default='D:/aio2_results_2/unet_ns_ns_seg/tdobj/mcr_1_seed_42/checkpoints/checkpoint_mcr_1_epoch_30.pth',
                        help='warm-up 階段的 checkpoint 路徑')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='CSV、log、tif 的輸出根目錄')
    parser.add_argument('--splits', type=str, nargs='+', default=['val', 'test'],
                        help='要合併推論的 split（預設 val + test）')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='最大連通區域佔比閾值，超過則判定為錯標')
    parser.add_argument('--model_key', type=str, default='model_state_dict_mep',
                        choices=['model_state_dict', 'model_state_dict_mit', 'model_state_dict_mep'],
                        help='要從 checkpoint 載入哪個模型')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


# ──────────────────────────────────────────────
# 執行入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 兩個 checkpoint 都跑
    checkpoints = {
        'epoch50_correct': args.ckpt_correct,
        'epoch30_warmup':  args.ckpt_warmup,
    }

    for ckpt_tag, ckpt_path in checkpoints.items():
        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        # warm-up checkpoint 沒有 mep key，自動改用 mit
        model_key = args.model_key
        if ckpt_tag == 'epoch30_warmup' and model_key == 'model_state_dict_mep':
            print(f"[WARNING] warm-up checkpoint has no mep key, switching to model_state_dict_mit")
            model_key = 'model_state_dict_mit'

        output_csv   = os.path.join(args.output_dir, f"mislabel_{ckpt_tag}.csv")
        log_path     = os.path.join(args.output_dir, f"mislabel_{ckpt_tag}.log")
        pred_save_dir = os.path.join(args.output_dir, f"pred_masks_{ckpt_tag}")

        print(f"\n{'='*60}")
        print(f"Running: {ckpt_tag} | splits={args.splits} | model_key={model_key}")
        print(f"{'='*60}")

        run_inference(
            ckpt_path=ckpt_path,
            data_path=args.data_path,
            noise_dir_name=args.noise_dir_name,
            splits=args.splits,
            output_csv=output_csv,
            pred_save_dir=pred_save_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_key=model_key,
            log_path=log_path,
            device=device,
        )

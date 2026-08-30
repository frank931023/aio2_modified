"""
評估 pred mask vs clean seg 的 image-level 指標
- 每張圖各自算 largest CC ratio，> threshold 就當作 1（有建築物/有大面積區域），否則 0
- pred_label  = pred  的 CC ratio > threshold
- true_label  = clean seg 的 CC ratio > threshold
- 輸出：precision, recall, accuracy, f1, confusion matrix（image-level）

用法：
    python py_scripts/evaluate_mislabel_detection.py \
        --pred_dir      inference_results/pred_masks_epoch50_correct \
                        inference_results/pred_masks_epoch30_warmup \
        --clean_seg_dir Massachusetts/tiff_256/val/seg \
                        Massachusetts/tiff_256/test/seg \
        --threshold 0.50
"""

import argparse
import os
import numpy as np
import cv2
from scipy import ndimage
from tqdm import tqdm


def largest_cc_ratio(mask):
    """mask: H x W, 值為 0/1，回傳 (最大連通區域佔整張圖比例, 最大連通區域像素數)"""
    total_pixels = mask.size
    if mask.sum() == 0:
        return 0.0, 0
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return 0.0, 0
    cc_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    max_cc = int(max(cc_sizes))
    return max_cc / total_pixels, max_cc


def build_lookup(dirs):
    lookup = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARNING] Directory not found: {d}")
            continue
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            lookup[stem] = os.path.join(d, f)
    return lookup


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, nargs='+', required=False,
                        default=['inference_results/pred_masks_epoch50_correct',
                                 'inference_results/pred_masks_epoch30_warmup'],
                        help='pred mask 資料夾（可多個）')
    parser.add_argument('--clean_seg_dir', type=str, nargs='+', required=False,
                        default=['bananan_images_converted/tiff_256/val/seg',
                                 'bananan_images_converted/tiff_256/test/seg'],
                        help='clean seg 資料夾（可多個 split）')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='largest CC ratio 閾值')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    clean_lookup = build_lookup(args.clean_seg_dir)
    print(f"Clean seg files : {len(clean_lookup)}")
    print(f"Threshold       : {args.threshold}\n")

    for pred_dir in args.pred_dir:
        if not os.path.isdir(pred_dir):
            print(f"[SKIP] Not found: {pred_dir}")
            continue

        pred_files = [f for f in os.listdir(pred_dir)
                      if f.lower().endswith(('.tif', '.tiff', '.png'))]

        y_pred, y_true = [], []
        rows = []
        skipped = 0

        for fname in tqdm(pred_files, desc=os.path.basename(pred_dir)):
            stem = os.path.splitext(fname)[0]
            clean_path = clean_lookup.get(stem)
            if clean_path is None:
                skipped += 1
                continue

            pred  = cv2.imread(os.path.join(pred_dir, fname), cv2.IMREAD_GRAYSCALE)
            clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

            if pred is None or clean is None:
                skipped += 1
                continue

            pred  = (pred  > 0  ).astype(np.uint8)
            clean = (clean > 127).astype(np.uint8)

            pred_lcc, pred_lcc_pixels   = largest_cc_ratio(pred)
            clean_lcc, clean_lcc_pixels = largest_cc_ratio(clean)
            pred_label  = int(pred_lcc  > args.threshold)
            true_label  = int(clean_lcc > args.threshold)

            pred_pos_pixels = int(pred.sum())
            pred_neg_pixels = int(pred.size - pred_pos_pixels)
            seg_pos_pixels  = int(clean.sum())
            seg_neg_pixels  = int(clean.size - seg_pos_pixels)

            y_pred.append(pred_label)
            y_true.append(true_label)

            rows.append({
                'fname':                fname,
                'pred_label':           pred_label,
                'true_label':           true_label,
                'pred_pos_pixels':      pred_pos_pixels,
                'pred_pos_lcc_pixels':  pred_lcc_pixels,
                'pred_neg_pixels':      pred_neg_pixels,
                'pred_lcc_ratio':       round(pred_lcc, 6),
                'seg_pos_pixels':       seg_pos_pixels,
                'seg_pos_lcc_pixels':   clean_lcc_pixels,
                'seg_neg_pixels':       seg_neg_pixels,
                'seg_lcc_ratio':        round(clean_lcc, 6),
            })

        if skipped > 0:
            print(f"  [WARNING] Skipped {skipped} files")

        if not y_true:
            print("  No valid samples.\n")
            continue

        # 寫出 CSV
        import csv
        csv_path = os.path.join(os.path.dirname(pred_dir),
                                f"eval_{os.path.basename(pred_dir)}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV saved: {csv_path}")

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        total     = len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy  = (tp + tn) / total if total > 0 else 0.0

        print(f"\n{'='*55}")
        print(f"  {os.path.basename(pred_dir)}")
        print(f"  Total images : {total}")
        print(f"  Confusion Matrix (image-level):")
        print(f"    TP={tp}  FP={fp}")
        print(f"    FN={fn}  TN={tn}")
        print()
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print()

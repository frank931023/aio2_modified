"""
分析 evaluate_mislabel_detection.py 產生的 CSV

功能：
  1. pred/gt 被判為 1 和 0 的個數
  2. 非零 pred_lcc_ratio / seg_lcc_ratio 的分布（histogram）
  3. 不同 threshold 下的 precision / recall / f1 / accuracy 曲線

用法：
    python py_scripts/analyze_eval_csv.py \
        --csv inference_results/eval_pred_masks_epoch50_correct.csv \
              inference_results/eval_pred_masks_epoch30_warm.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── helpers ──────────────────────────────────────────────────────────────────

def metrics_at(df, threshold, ratio_col='pred_lcc_ratio', gt_col='seg_lcc_ratio'):
    y_pred = (df[ratio_col] > threshold).astype(int)
    y_true = (df[gt_col]   > threshold).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / len(df) if len(df) > 0 else 0.0
    return prec, rec, f1, acc, tp, fp, fn, tn


def analyze(csv_path, thresholds, out_dir):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    df   = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"  {name}  (N={len(df)})")
    print(f"{'='*60}")

    # ── 1. label counts ──────────────────────────────────────────
    print("\n[Label counts (from stored pred_label / true_label columns)]")
    for col, tag in [('pred_label', 'pred'), ('true_label', 'gt')]:
        if col in df.columns:
            vc = df[col].value_counts().sort_index()
            print(f"  {tag:6s}  0: {vc.get(0, 0):>7,}   1: {vc.get(1, 0):>7,}")

    # ── 2. threshold sweep ───────────────────────────────────────
    print(f"\n[Threshold sweep]")
    print(f"  {'thresh':>7}  {'prec':>7}  {'rec':>7}  {'f1':>7}  {'acc':>7}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}")
    records = []
    for thr in thresholds:
        prec, rec, f1, acc, tp, fp, fn, tn = metrics_at(df, thr)
        print(f"  {thr:>7.3f}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}  {acc:>7.4f}  "
              f"{tp:>6}  {fp:>6}  {fn:>6}  {tn:>6}")
        records.append((thr, prec, rec, f1, acc))

    arr = np.array(records)   # (T, 5)

    # ── 3. plots ─────────────────────────────────────────────────
    BIN_EDGES = np.arange(0, 1.01, 0.1)   # 0~0.1, 0.1~0.2, ..., 0.9~1.0
    bin_labels = [f'{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}'
                  for i in range(len(BIN_EDGES) - 1)]

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(name, fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── row 0: pred histogram + pred bar chart + scatter ──────────
    nz_pred = df.loc[df['pred_lcc_ratio'] > 0, 'pred_lcc_ratio']
    nz_seg  = df.loc[df['seg_lcc_ratio']  > 0, 'seg_lcc_ratio']

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(nz_pred, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    ax0.set_title('pred_lcc_ratio histogram (non-zero)')
    ax0.set_xlabel('ratio'); ax0.set_ylabel('count')

    ax1 = fig.add_subplot(gs[0, 1])
    pred_counts, _ = np.histogram(nz_pred, bins=BIN_EDGES)
    bars1 = ax1.bar(range(len(bin_labels)), pred_counts, color='steelblue',
                    edgecolor='white', width=0.7)
    ax1.bar_label(bars1, fmt='%d', fontsize=7, rotation=45)
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_title('pred_lcc_ratio bar chart (non-zero)')
    ax1.set_xlabel('ratio bin'); ax1.set_ylabel('count')

    ax2 = fig.add_subplot(gs[0, 2])
    sample = df.sample(min(5000, len(df)), random_state=0)
    ax2.scatter(sample['seg_lcc_ratio'], sample['pred_lcc_ratio'],
                alpha=0.3, s=4, color='purple')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=0.8)
    ax2.set_title('pred vs seg lcc_ratio')
    ax2.set_xlabel('seg_lcc_ratio'); ax2.set_ylabel('pred_lcc_ratio')

    # ── row 1: seg histogram + seg bar chart + label count bar ────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(nz_seg, bins=50, color='darkorange', edgecolor='white', linewidth=0.3)
    ax3.set_title('seg_lcc_ratio histogram (non-zero)')
    ax3.set_xlabel('ratio'); ax3.set_ylabel('count')

    ax4 = fig.add_subplot(gs[1, 1])
    seg_counts, _ = np.histogram(nz_seg, bins=BIN_EDGES)
    bars4 = ax4.bar(range(len(bin_labels)), seg_counts, color='darkorange',
                    edgecolor='white', width=0.7)
    ax4.bar_label(bars4, fmt='%d', fontsize=7, rotation=45)
    ax4.set_xticks(range(len(bin_labels)))
    ax4.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_title('seg_lcc_ratio bar chart (non-zero)')
    ax4.set_xlabel('ratio bin'); ax4.set_ylabel('count')

    ax5 = fig.add_subplot(gs[1, 2])
    cats = ['pred=0', 'pred=1', 'gt=0', 'gt=1']
    vals = [
        (df['pred_label'] == 0).sum() if 'pred_label' in df else 0,
        (df['pred_label'] == 1).sum() if 'pred_label' in df else 0,
        (df['true_label'] == 0).sum() if 'true_label' in df else 0,
        (df['true_label'] == 1).sum() if 'true_label' in df else 0,
    ]
    colors = ['#4c72b0', '#4c72b0', '#dd8452', '#dd8452']
    bars5 = ax5.bar(cats, vals, color=colors, edgecolor='white')
    ax5.bar_label(bars5, fmt='%d', fontsize=8)
    ax5.set_title('Label counts'); ax5.set_ylabel('count')

    # ── row 2: precision/recall/f1 + accuracy ─────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(arr[:, 0], arr[:, 1], label='precision', marker='.')
    ax6.plot(arr[:, 0], arr[:, 2], label='recall',    marker='.')
    ax6.plot(arr[:, 0], arr[:, 3], label='f1',        marker='.')
    ax6.set_title('Precision / Recall / F1 vs Threshold')
    ax6.set_xlabel('threshold'); ax6.set_ylabel('score')
    ax6.legend(); ax6.set_ylim(0, 1)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(arr[:, 0], arr[:, 4], color='green', marker='.')
    ax7.set_title('Accuracy vs Threshold')
    ax7.set_xlabel('threshold'); ax7.set_ylabel('accuracy')
    ax7.set_ylim(0, 1)

    print(f"\n[pred_lcc_ratio non-zero]  n={len(nz_pred):,}  "
          f"mean={nz_pred.mean():.4f}  median={nz_pred.median():.4f}  "
          f"p25={nz_pred.quantile(0.25):.4f}  p75={nz_pred.quantile(0.75):.4f}")
    print(f"[seg_lcc_ratio  non-zero]  n={len(nz_seg):,}  "
          f"mean={nz_seg.mean():.4f}  median={nz_seg.median():.4f}  "
          f"p25={nz_seg.quantile(0.25):.4f}  p75={nz_seg.quantile(0.75):.4f}")

    out_path = os.path.join(out_dir, f"analysis_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='+', required=False,
                        default=['inference_results/eval_pred_masks_epoch50_correct.csv',
                                 'inference_results/eval_pred_masks_epoch30_warm.csv'],
                        help='eval CSV 檔案（可多個）')
    parser.add_argument('--out_dir', type=str, default='inference_results',
                        help='圖片輸出資料夾')
    parser.add_argument('--thr_min',  type=float, default=0.05)
    parser.add_argument('--thr_max',  type=float, default=0.95)
    parser.add_argument('--thr_step', type=float, default=0.05)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    thresholds = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    os.makedirs(args.out_dir, exist_ok=True)
    for csv_path in args.csv:
        if not os.path.isfile(csv_path):
            print(f"[SKIP] Not found: {csv_path}")
            continue
        analyze(csv_path, thresholds, args.out_dir)

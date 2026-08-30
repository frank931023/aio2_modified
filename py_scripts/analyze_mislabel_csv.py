"""
分析 inference_mislabel_detection.py 產生的 CSV 結果
1. pred_positive_pixels 不為 0 的分布長條圖
2. 不同 threshold 下 flagged 數量變化折線圖
用法：
    python py_scripts/analyze_mislabel_csv.py \
        --csv inference_results/mislabel_epoch50_correct.csv \
        --csv inference_results/mislabel_epoch30_warmup.csv \
        --output_dir ./analysis_results
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────
# 1. pred_positive_pixels 分布長條圖
# ──────────────────────────────────────────────

def plot_pred_positive_hist(df, label, output_dir):
    nonzero = df[df['pred_positive_pixels'] > 0]['pred_positive_pixels']
    total   = len(df)
    nonzero_count = len(nonzero)

    if nonzero_count == 0:
        print(f"[{label}] No images with pred_positive_pixels > 0, skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # 自動決定 bin 數（最多 40 個區間）
    bins = min(40, max(10, nonzero_count // 20))
    ax.hist(nonzero, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('pred_positive_pixels', fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.set_title(
        f'[{label}] Distribution of pred_positive_pixels (nonzero only)\n'
        f'nonzero: {nonzero_count} / {total} images ({nonzero_count/total*100:.1f}%)',
        fontsize=12
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'hist_pred_positive_{label}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────
# 2. 不同 threshold 下 flagged 數量折線圖
# ──────────────────────────────────────────────

def plot_threshold_sweep(dfs_labels, output_dir,
                         thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_count, ax_pct, (df, label) in zip(
        [axes[0]] * len(dfs_labels),
        [axes[1]] * len(dfs_labels),
        dfs_labels
    ):
        total = len(df)
        flagged_counts = [
            (df['largest_cc_ratio'] > t).sum() for t in thresholds
        ]
        flagged_pcts = [c / total * 100 for c in flagged_counts]

        axes[0].plot(thresholds, flagged_counts, marker='', linewidth=1.5, label=label)
        axes[1].plot(thresholds, flagged_pcts,   marker='', linewidth=1.5, label=label)

    for ax, ylabel, title in zip(
        axes,
        ['Flagged count', 'Flagged (%)'],
        ['Flagged count vs threshold', 'Flagged % vs threshold']
    ):
        ax.set_xlabel('Threshold (largest_cc_ratio)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='default=0.5')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'threshold_sweep.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # 同時印出幾個關鍵 threshold 的數字
    key_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print("\n--- Flagged count at key thresholds ---")
    header = f"{'threshold':>10}" + "".join(f"  {label:>25}" for _, label in dfs_labels)
    print(header)
    for t in key_thresholds:
        row = f"{t:>10.2f}"
        for df, _ in dfs_labels:
            cnt = (df['largest_cc_ratio'] > t).sum()
            pct = cnt / len(df) * 100
            row += f"  {cnt:>10,} ({pct:5.1f}%)"
        print(row)


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                        help='一或多個 CSV 路徑')
    parser.add_argument('--output_dir', type=str, default='./analysis_results')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dfs_labels = []
    for csv_path in args.csv:
        if not os.path.isfile(csv_path):
            print(f"[SKIP] File not found: {csv_path}")
            continue
        label = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        print(f"Loaded [{label}]: {len(df)} rows")
        dfs_labels.append((df, label))

        # 各自畫 pred_positive 分布圖
        plot_pred_positive_hist(df, label, args.output_dir)

    if dfs_labels:
        # 所有 CSV 一起畫 threshold sweep
        plot_threshold_sweep(dfs_labels, args.output_dir)

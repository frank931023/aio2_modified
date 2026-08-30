"""
Visualize training loss curves from test.txt

This script extracts 'current training loss' and 'average training loss' 
from the log file and creates visualization plots with moving average smoothing.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv


def parse_log_file(log_file):
    """
    Parse training log file and extract losses
    
    Args:
        log_file: Path to log file
    
    Returns:
        tuple: (batch_nums, current_losses, avg_losses)
    """
    batch_nums = []
    current_losses = []
    avg_losses = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract batch number
            batch_match = re.search(r'Batch-(\d+)', line)
            if not batch_match:
                continue
            
            batch_num = int(batch_match.group(1))
            
            # Extract current training loss
            current_match = re.search(r'current training loss: ([\d.]+)', line)
            if not current_match:
                continue
            current_loss = float(current_match.group(1))
            
            # Extract average training loss
            avg_match = re.search(r'average training loss: ([\d.]+)', line)
            if not avg_match:
                continue
            avg_loss = float(avg_match.group(1))
            
            batch_nums.append(batch_num)
            current_losses.append(current_loss)
            avg_losses.append(avg_loss)
    
    return batch_nums, current_losses, avg_losses


def moving_average(data, window_size=50):
    """
    Calculate moving average
    
    Args:
        data: List of values
        window_size: Size of the moving window
    
    Returns:
        List of moving average values
    """
    if window_size <= 0 or window_size > len(data):
        return data
    
    ma = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Pad with NaN at the beginning to maintain same length
    ma = np.concatenate([np.full(window_size-1, np.nan), ma])
    return ma


def save_losses_to_csv(log_file, csv_file):
    """Parse `log_file` and write losses to `csv_file`.

    CSV columns: batch,current_loss,average_loss
    """
    batch_nums, current_losses, avg_losses = parse_log_file(log_file)
    if not batch_nums:
        print(f"❌ No entries parsed from {log_file}; CSV not written")
        return False

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'current_loss', 'average_loss'])
        for b, c, a in zip(batch_nums, current_losses, avg_losses):
            writer.writerow([b, f"{c:.8f}", f"{a:.8f}"])

    print(f"✓ CSV written: {csv_file} ({len(batch_nums)} rows)")
    return True


def load_losses_from_csv(csv_file):
    """Load batch, current, average from CSV file."""
    batch_nums = []
    current_losses = []
    avg_losses = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                batch_nums.append(int(row['batch']))
                current_losses.append(float(row['current_loss']))
                avg_losses.append(float(row['average_loss']))
            except Exception:
                continue
    return batch_nums, current_losses, avg_losses


# def plot_losses(batch_nums, current_losses, avg_losses, output_file=None, ma_window=50):
#     """
#     Create visualization plots for losses with moving average
    
#     Args:
#         batch_nums: List of batch numbers
#         current_losses: List of current training losses
#         avg_losses: List of average training losses
#         output_file: Path to save figure (optional)
#         ma_window: Window size for moving average (default: 50)
#     """
    
#     # Calculate moving averages
#     current_ma = moving_average(current_losses, ma_window)
#     avg_ma = moving_average(avg_losses, ma_window)
    
#     # Create figure with single plot showing only EMA results
#     fig, ax = plt.subplots(figsize=(14, 6))
    
#     # Plot EMA curves (only two lines) with clearly distinct colors and styles
#     ax.plot(
#         batch_nums,
#         current_ma,
#         label=f'Current Loss (MA-{ma_window})',
#         color='#e41a1c',
#         linewidth=2.8,
#         linestyle='-',
#         marker='o',
#         markersize=4,
#         markevery=max(1, len(batch_nums)//20),
#     )
#     ax.plot(
#         batch_nums,
#         avg_ma,
#         label=f'Average Loss (MA-{ma_window})',
#         color='#377eb8',
#         linewidth=2.8,
#         linestyle='--',
#         marker=None,
#     )
#     ax.set_xlabel('Batch Number', fontsize=12)
#     ax.set_ylabel('Loss (Moving Average)', fontsize=12)
#     ax.set_title(f'Training Loss - EMA (Window={ma_window})', fontsize=14, fontweight='bold')
#     ax.legend(fontsize=12)
    
#     plt.tight_layout()
    
#     # Save or show
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"✓ Figure saved to: {output_file}")
#     else:
#         plt.show()

def plot_losses(batch_nums, current_losses, avg_losses, output_file=None, ma_window=50):
    """
    Create visualization plots for losses with moving average
    """
    # Calculate moving averages
    current_ma = moving_average(current_losses, ma_window)
    avg_ma = moving_average(avg_losses, ma_window)
    
    # 💡 修正 1: 建立連續的 X 軸，防止因為 Batch 號碼重置 (Epoch 重新開始) 而畫出折返的橫線
    steps = np.arange(1, len(batch_nums) + 1)
    
    # Create figure with single plot showing only EMA results
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 💡 修正 2: 調整線條粗細與透明度。因為數值完全一樣，我們讓 Current 粗且半透明，Average 細且虛線
    ax.plot(
        steps, # 改用連續步數
        current_ma,
        label=f'Current Loss (MA-{ma_window})',
        color='#e41a1c', # 紅色
        linewidth=5.0,   # 加粗當底色
        alpha=0.4,       # 設定半透明
        linestyle='-',
    )
    ax.plot(
        steps, # 改用連續步數
        avg_ma,
        label=f'Average Loss (MA-{ma_window})',
        color='#377eb8', # 藍色
        linewidth=2.0,   # 較細的線
        linestyle='--',  # 虛線，讓底下的紅色透出來
    )
    
    # X 軸標籤改為 Global Steps 會比較符合實際狀況
    ax.set_xlabel('Global Training Steps (Cumulative)', fontsize=12)
    ax.set_ylabel('Loss (Moving Average)', fontsize=12)
    ax.set_title(f'Training Loss - EMA (Window={ma_window})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_file}")
    else:
        plt.show()

def print_statistics(current_losses, avg_losses):
    """
    Print statistics about the losses
    """
    print("\n" + "="*60)
    print("TRAINING LOSS STATISTICS")
    print("="*60)
    
    print("\nCurrent Training Loss:")
    print(f"  Min:     {min(current_losses):.6f}")
    print(f"  Max:     {max(current_losses):.6f}")
    print(f"  Mean:    {np.mean(current_losses):.6f}")
    print(f"  Std:     {np.std(current_losses):.6f}")
    print(f"  Median:  {np.median(current_losses):.6f}")
    
    print("\nAverage Training Loss:")
    print(f"  Min:     {min(avg_losses):.6f}")
    print(f"  Max:     {max(avg_losses):.6f}")
    print(f"  Mean:    {np.mean(avg_losses):.6f}")
    print(f"  Std:     {np.std(avg_losses):.6f}")
    print(f"  Median:  {np.median(avg_losses):.6f}")
    
    # Calculate trend
    first_10_avg = np.mean(current_losses[:10])
    last_10_avg = np.mean(current_losses[-10:])
    trend = ((last_10_avg - first_10_avg) / first_10_avg) * 100
    print(f"\nTrend (first 10 vs last 10 batches): {trend:+.2f}%")
    print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training loss curves from log file')
    parser.add_argument('--log_file', type=str, default='test.txt',
                        help='Path to log file (default: test.txt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save figure (default: show in window)')
    parser.add_argument('--no_stats', action='store_true',
                        help='Do not print statistics')
    parser.add_argument('--ma_window', type=int, default=1000,
                        help='Moving average window size (default: 50)')
    parser.add_argument('--csv_file', type=str, default='losses.csv',
                        help='Path to CSV file to write/read (default: losses.csv)')
    parser.add_argument('--only_csv', action='store_true',
                        help='Only extract losses to CSV and exit')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}")
        return
    
    # Step 1: extract losses from log and write CSV
    print(f"📖 Parsing and writing CSV from: {args.log_file} -> {args.csv_file}")
    written = save_losses_to_csv(args.log_file, args.csv_file)
    if not written:
        return

    if args.only_csv:
        print("✓ CSV export complete; exiting as requested (--only_csv)")
        return

    # Step 2: load losses from CSV and plot
    print(f"📥 Loading losses from CSV: {args.csv_file}")
    batch_nums, current_losses, avg_losses = load_losses_from_csv(args.csv_file)
    if not batch_nums:
        print(f"❌ Error: No data found in CSV: {args.csv_file}")
        return

    print(f"✓ Loaded {len(batch_nums)} entries from CSV")

    # Print statistics (from CSV-loaded data)
    if not args.no_stats:
        print_statistics(current_losses, avg_losses)

    # Create plots from CSV data
    print("📊 Creating plots from CSV data...")
    plot_losses(batch_nums, current_losses, avg_losses, args.output, args.ma_window)

    if args.output:
        print(f"\n✓ Visualization complete! Saved to: {args.output}")
    else:
        print("\n✓ Close the plot window to exit")


if __name__ == '__main__':
    main()

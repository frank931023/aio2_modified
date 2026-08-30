import re
import numpy as np
import matplotlib.pyplot as plt

# ── 解析 test2.txt ──────────────────────────────────────
pattern = re.compile(
    r"Epoch-(\d+)\|Batch-(\d+).*?current training loss:\s*([\d.]+)"
)

epochs, batches, losses = [], [], []
with open("test2.txt", "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            batches.append(int(m.group(2)))
            losses.append(float(m.group(3)))

if not losses:
    print("No loss data found in test2.txt")
    exit()

losses = np.array(losses)
x = np.arange(len(losses))  # global step index

# ── EMA 平滑 ────────────────────────────────────────────
def ema(values, alpha=0.05):
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

smoothed = ema(losses, alpha=0.05)

# ── 畫圖：左原始 / 右平滑 ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Training Loss Curve", fontsize=14)

# 左：原始
axes[0].plot(x, losses, linewidth=0.6, color="steelblue", alpha=0.85)
axes[0].set_title("Raw Loss")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# 右：平滑（底層保留原始淡色）
axes[1].plot(x, losses,   linewidth=0.4, color="steelblue", alpha=0.25, label="raw")
axes[1].plot(x, smoothed, linewidth=1.8, color="tomato",    label="EMA (α=0.05)")
axes[1].set_title("Smoothed Loss (EMA)")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
plt.show()
print(f"Parsed {len(losses)} steps | Saved → loss_plot.png")

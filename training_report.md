# 訓練過程報告：`train_unet_png_emaCorrect_singleGPU.py`

---

## 1. 程式概述

這支腳本實作了 **AIO2（Adaptive Iterative Online Object-wise correction）** 框架，用於在 **Massachusetts Building 資料集** 上訓練 UNet 語意分割模型。核心概念是：在有 noisy label（錯誤標記）的情況下，透過 **EMA（Exponential Moving Average）Teacher Model** 動態修正訓練標籤，並搭配 **Early Learning Detection（ACT 模組）** 自動偵測何時開始進行標籤修正。

---

## 2. 程式架構說明

### 2.1 模型架構

| 元件 | 說明 |
|------|------|
| Student Model (`net`) | UNet，3 通道輸入，1 類別輸出（二元分割） |
| EMA-MIT (`net_ema_it`) | 每個 iteration 更新的 EMA Teacher Model，alpha=0.999 |
| EMA-MEP (`net_ema_ep`) | 每個 epoch 更新的 EMA Teacher Model（若啟用） |

EMA 更新公式：`θ_teacher = α * θ_teacher + (1 - α) * θ_student`

### 2.2 Loss Function

使用 **Combined Loss**（`loss_type='cd'`）：
- **Binary Cross Entropy（BCE）**：`nn.BCEWithLogitsLoss()`
- **Dice Loss**：自定義 `dice_loss()`
- 最終 loss = BCE loss + Dice loss

### 2.3 Optimizer

- **Adam**，初始 lr = 0.001，weight decay = 1e-8
- 搭配 **GradScaler**（Mixed Precision AMP）

### 2.4 資料集

| 分割 | 樣本數 | Batch 數（batch_size=25） |
|------|--------|--------------------------|
| Train | 95,400 | 1,490 |
| Val | 114,000 | 4,560 |
| Test | 126,000 | 5,040 |

Noisy label 目錄：`ns_seg_1`（Monte Carlo run 1）

### 2.5 訓練流程（每個 Epoch）

```
for each batch:
  1. 載入 image、noisy mask、gt mask
  2. 若 correct=False → 使用原始 noisy label 訓練（warm-up 階段）
     若 correct=True  → 用 EMA teacher 預測，做 object-wise label correction
  3. Forward pass（student model）
  4. 計算 BCE + Dice loss
  5. Backward + optimizer step
  6. 更新 EMA-MIT（每個 iteration）
  7. 記錄 loss 與 metrics

after each epoch:
  1. Validate（noisy labels + GT labels）
  2. Test（GT labels + noisy labels）
  3. Evaluate EMA-MIT model
  4. 若 correct=False → 執行 Early Learning Detection（ACT 模組）
     若偵測到 trigger → 從 trigger epoch 的 checkpoint reload，開始 correction
  5. 儲存 checkpoint
  6. Log 到 wandb
```

### 2.6 Early Learning Detection（ACT 模組）

- 每個 epoch 記錄 EMA-MIT 的 training IoU（`tr_iou_mit`）
- 計算 numerical gradient（sliding window）
- 當 gradient 趨勢符合 trigger 條件時，設定 `correct=True`
- **Reload 到 trigger epoch 的 checkpoint**，從那個點重新開始訓練（但這次帶 label correction）

### 2.7 Object-wise Label Correction

- 用 EMA teacher 對當前 batch 做推論
- 將 teacher 的預測與 noisy label 做 object-wise 比對
- 若 teacher 預測與 noisy label 不一致，以 teacher 預測取代
- 修正後的 label 送回 GPU 作為 true label 訓練 student

---

## 3. 訓練過程時間軸

### 3.1 初始化（0:00:00）

```
Network architecture: unet (unet_ns_ns_seg)
len/n_batch of train, val, test: 95400/1490, 114000/4560, 126000/5040
BinaryCrossEntropy loss is used!
Dice loss is used!
No pretrained weights, will start from scratch.
EMA-mit model has been initialized from student model!
EMA-mep model has been initialized from student model!
Start training...
```

- 從頭開始訓練（無預訓練權重）
- EMA-MIT 與 EMA-MEP 皆從 student model 複製初始化

---

### 3.2 Warm-up 階段（Epoch 1–36）：使用 Noisy Labels 訓練

此階段 `correct=False`，全程使用原始 noisy label 訓練，同時持續監控 EMA-MIT 的 IoU 趨勢以偵測 early learning。

| Epoch | 開始時間 | Batch-1 Loss | 備註 |
|-------|----------|--------------|------|
| 1 | 0:00:09 | 1.5494 | 起始 loss 最高 |
| 2 | 3:10:20 | 0.4827 | 明顯下降 |
| 3 | 6:10:54 | 0.7133 | |
| 4 | 8:58:04 | 0.6496 | |
| 5 | 11:36:13 | 0.4769 | |
| 6 | 14:32:50 | 0.4373 | |
| 7 | 17:20:25 | 0.4605 | |
| 8 | 19:58:43 | 0.4082 | |
| 9 | 22:54:38 | 0.4119 | |
| 10 | 1 day, 1:44:01 | 0.4328 | |
| 11 | 1 day, 4:22:20 | 0.5186 | |
| 12 | 1 day, 7:22:36 | 0.3353 | |
| 13 | 1 day, 10:09:43 | 0.3702 | |
| 14 | 1 day, 12:47:38 | 0.3808 | |
| 15 | 1 day, 15:43:59 | 0.4569 | |
| 16 | 1 day, 18:31:14 | 0.3165 | |
| 17 | 1 day, 21:09:23 | 0.6121 | |
| 18 | 2 days, 1:04:17 | 0.3724 | |
| 19 | 2 days, 5:19:54 | 0.3680 | |
| 20 | 2 days, 9:45:52 | 0.4007 | **← ACT 偵測到的 trigger epoch** |
| 21 | 2 days, 13:55:57 | 0.2711 | |
| 22 | 2 days, 18:14:22 | 0.5743 | |
| 23 | 2 days, 22:33:42 | 0.4328 | |
| 24 | 3 days, 2:40:11 | 0.6264 | |
| 25 | 3 days, 6:52:44 | 0.3575 | |
| 26 | 3 days, 10:29:37 | 0.3635 | |
| 27 | 3 days, 14:25:09 | 0.4862 | |
| 28 | 3 days, 18:36:52 | 0.3499 | |
| 29 | 3 days, 22:55:53 | 0.4732 | |
| 30 | 4 days, 3:05:32 | 0.3300 | |
| 31 | 4 days, 7:27:59 | 0.3980 | |
| 32 | 4 days, 11:37:41 | 0.3292 | |
| 33 | 4 days, 14:43:39 | 0.3714 | |
| 34 | 4 days, 18:53:27 | 0.4691 | |
| 35 | 4 days, 23:12:18 | 0.3021 | |
| 36 | 5 days, 3:22:09 | 0.3225 | **← ACT trigger 在此 epoch 結束後觸發** |

**Epoch 1 的 Loss 趨勢（Batch-1 到 Batch-1490）：**
- Batch-1：1.5494（最高）
- Batch-25：0.9263（快速下降）
- Batch-100：0.7523
- Batch-500：0.7199
- Batch-1000：約 0.5~0.6
- Batch-1490：0.8268（有波動，屬正常）

每個 epoch 約花費 **2.5~3 小時**，每個 batch 約 **1~2 秒**。

---

### 3.3 ACT 觸發事件（5 days, 3:49:59 後）

```
Correction starts! - After EPOCH 36 and resume from EPOCH 20
```

**發生了什麼：**
1. ACT 模組在 Epoch 36 結束後，分析了 Epoch 1–36 的 EMA-MIT training IoU 趨勢
2. 偵測到 **Epoch 20** 是 early learning 的 trigger point（loss 開始過擬合 noisy label 的轉折點）
3. 程式自動：
   - 從磁碟載入 `checkpoint_mcr_1_epoch_20.pth`
   - 將 student model、EMA-MIT model、optimizer 全部 reload 回 Epoch 20 的狀態
   - 設定 `correct=True`，開始 label correction 階段
   - 重置 epoch counter，從 Epoch 21 繼續（但這次帶 correction）

---

### 3.4 Label Correction 階段（Epoch 21–50）

此階段 `correct=True`，每個 batch 都會：
1. 用 EMA-MIT teacher 對當前 batch 做推論
2. 執行 object-wise label correction
3. 用修正後的 label 訓練 student model

Loss 在此階段明顯更低，因為 label 品質提升：

| Epoch | 開始時間 | Batch-1 Loss | 備註 |
|-------|----------|--------------|------|
| 21 | 5 days, 7:44:56 | 0.2342 | correction 開始，loss 大幅下降 |
| 22 | 5 days, 12:05:29 | 0.1681 | |
| 23 | 5 days, 16:36:14 | 0.2646 | |
| 24 | 5 days, 20:57:08 | 0.3862 | |
| 25 | 6 days, 1:31:26 | 0.3341 | |
| 26 | 6 days, 6:00:42 | 0.1956 | |
| 27 | 6 days, 10:04:57 | 0.2190 | |
| 28 | 6 days, ~14:00 | ~0.25 | |
| 29 | 6 days, ~18:00 | ~0.25 | |
| 30 | 6 days, 23:28:53 | 0.3488 | |
| 31 | 7 days, 3:49:12 | 0.3398 | |
| 32 | 7 days, 8:23:57 | 0.2272 | |
| 33 | 7 days, 12:45:37 | 0.2466 | |
| 34 | 7 days, 17:21:49 | 0.2910 | |
| 35 | 7 days, 21:51:49 | 0.2500 | |
| 36 | 8 days, 2:16:28 | 0.2730 | |
| 37 | 8 days, 6:26:06 | 0.2695 | |
| 38 | 8 days, 10:47:23 | 0.2043 | |
| 39 | 8 days, 15:18:41 | 0.3611 | |
| 40 | 8 days, 19:40:21 | 0.2404 | |
| 41 | 9 days, 0:03:32 | 0.1269 | 最低 batch-1 loss |
| 42 | 9 days, 3:45:54 | 0.1627 | |
| 43 | 9 days, 8:20:01 | 0.2484 | |
| 44 | 9 days, 11:46:54 | 0.1616 | |
| 45 | 9 days, 16:18:13 | 0.2217 | |
| 46 | 9 days, 20:40:08 | 0.3137 | |
| 47 | 10 days, 0:20:39 | 0.2230 | |
| 48 | 10 days, 4:41:18 | 0.2352 | |
| 49 | 10 days, 9:15:33 | 0.2096 | |
| 50 | 10 days, 13:42:36 | 0.2155 | **最後一個 epoch** |

---

### 3.5 訓練結束（10 days, 18:14:32）

```
Training is finished|Total spent time:10 days, 18:14:32!
```

---

## 4. 關鍵數字總結

| 項目 | 數值 |
|------|------|
| 總訓練時間 | **10 天 18 小時 14 分 32 秒** |
| 設定 epochs | 5（但 `fepochs = epochs * 2 = 10`，實際跑到 50） |
| Warm-up 階段 epochs | Epoch 1–36（約 5 天） |
| Correction 階段 epochs | Epoch 21–50（約 5 天 10 小時） |
| ACT trigger epoch | Epoch 20（在 Epoch 36 結束後偵測到） |
| 初始 loss（Epoch 1, Batch 1） | 1.5494 |
| 最終 loss（Epoch 50, Batch 1490 附近） | ~0.17–0.29 |
| 每個 epoch 耗時 | 約 2.5–3 小時 |
| 每個 batch 耗時 | 約 1–2 秒 |
| Train batches per epoch | 1,490 |
| Val batches per epoch | 4,560（用於 evaluate） |
| Test batches per epoch | 5,040（用於 evaluate） |
| Learning rate | 固定 0.001（無 scheduler） |

---

## 5. 程式中的重要設計細節

### 5.1 `fepochs = args.epochs * 2`

程式將 `epochs` 參數乘以 2 作為實際迴圈上限（`fepochs`）。這是因為 correction 階段會 reload 到更早的 epoch，需要額外的 epoch 空間繼續訓練。

### 5.2 `n_back` 機制

當 correction 觸發後，`n_back = epoch - fmid + 1`（本次為 `36 - 20 + 1 = 17`），用來在 epoch counter 上做偏移，讓 epoch 顯示從 21 重新開始，而不是繼續從 37。

### 5.3 Checkpoint 命名

- Warm-up 階段：`checkpoint_mcr_1_epoch_{N}.pth`（包含 `tr_iou_mit` 陣列）
- Correction 階段：`checkpoint_correct_mcr_1_epoch_{N}.pth`

### 5.4 Wandb 整合

每個 epoch 結束後，將 train/val/test metrics 以及 EMA-MIT 的 metrics 全部 log 到 wandb，方便視覺化追蹤。

### 5.5 `[DEBUG] length of dataloader` 訊息

每個 epoch 結束後的 validate/test 階段，`evl.evaluate()` 函式會印出 dataloader 長度的 debug 訊息（val=4560, test=5040），每次 evaluate 呼叫一次，共 6 次（val_ns, val_gt, test_gt, test_ns，各 model 各一組）。

---

## 6. Loss 下降趨勢

```
Epoch 1  (warm-up)    : ~1.5 → ~0.5  (快速下降)
Epoch 2–19 (warm-up)  : ~0.3 → ~0.6  (震盪，緩慢下降)
Epoch 20–36 (warm-up) : ~0.3 → ~0.4  (趨於穩定)
--- ACT trigger: reload to Epoch 20, start correction ---
Epoch 21–30 (correct) : ~0.15 → ~0.35 (correction 效果顯現)
Epoch 31–50 (correct) : ~0.12 → ~0.30 (持續下降，趨於穩定)
```

Correction 階段的 loss 整體比 warm-up 階段低，代表 label correction 有效提升了訓練品質。

---

## 7. 結論

這次訓練完整跑完了整個 AIO2 流程：
1. **Warm-up（Epoch 1–36）**：用 noisy label 訓練，同時讓 EMA teacher 學習
2. **ACT 偵測**：在 Epoch 36 結束後，自動偵測到 Epoch 20 是 early learning trigger point
3. **Correction（Epoch 21–50）**：從 Epoch 20 的 checkpoint reload，啟動 object-wise label correction，持續訓練到 Epoch 50
4. **訓練完成**：總耗時約 10 天 18 小時，loss 從初始 1.55 降至約 0.17–0.29

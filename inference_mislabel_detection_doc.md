# inference_mislabel_detection.py 說明文件

---

## 1. 腳本目的

這支腳本的目標是做 **Image-level 錯標檢測**。

原本的 AIO2 論文做的是 pixel-level 的 label correction（修正每個像素的標記），這支腳本則是在此基礎上往上抽象一層：**只要一張圖裡面有可疑的錯標區域，就把整張圖 flag 起來**。

---

## 2. 核心概念

### 為什麼用 val/test 而不用 train？

模型是用 train dataset 訓練的，在 train 上的預測會有 overfitting 偏差（可能記住了 noisy label 的錯誤模式）。val/test 是模型從未見過的資料，預測結果更能反映模型真正學到的「建築物長什麼樣子」，比較結果更可信。

### 錯標判定邏輯

```
預測 mask（model output）
        ↓
與 GT label 做差異比較 → diff mask（哪些像素不一樣）
        ↓
對 diff mask 做連通區域分析（Connected Component Analysis）
        ↓
取最大連通區域面積 / 圖片總像素 = largest_cc_ratio
        ↓
largest_cc_ratio > threshold（預設 0.50）→ 判定為有錯標
```

用連通區域而不是單純差異像素數的原因：模型預測在邊界處幾乎一定有零散誤差，這些誤差是分散的小點。真正的錯標（例如整棟建築被漏標）會形成一大塊連續的差異區域，連通區域分析可以有效區分這兩種情況。

---

## 3. 函式說明

### `setup_logger(log_path)`

設定 logging 系統，同時輸出到：
- **console**：顯示 INFO 以上的訊息
- **log 檔案**：記錄 DEBUG 以上的所有訊息（更詳細）

每次呼叫 `run_inference` 前會先清掉舊的 handler，避免多次執行時訊息重複輸出。

---

### `load_model(ckpt_path, device, model_key, logger)`

從 `.pth` checkpoint 載入模型。

checkpoint 內包含三個模型的權重：

| `model_key` | 對應模型 | 說明 |
|---|---|---|
| `model_state_dict` | Student model | 直接做 backprop 訓練的主模型 |
| `model_state_dict_mit` | EMA-MIT | 每個 iteration 更新的 EMA teacher（alpha=0.999） |
| `model_state_dict_mep` | EMA-MEP | 每個 epoch 更新的 EMA teacher（alpha=0.99） |

預設使用 `model_state_dict_mep`，因為根據 wandb 結果它在 val GT 上 Dice 最高（0.906）。

注意事項：
- `model_state_dict_mep` 在原始程式碼中有時被存成 `tuple`（原始碼的小 bug），這裡會自動處理
- 會自動移除 DataParallel 的 `module.` 前綴

---

### `largest_cc_ratio(diff_mask)`

輸入：二值差異 mask（H × W，值為 0/1）

流程：
1. 用 `scipy.ndimage.label` 對差異區域做連通區域標記
2. 計算每個連通區域的面積
3. 取最大連通區域面積 / 圖片總像素數

回傳值範圍：0.0 ~ 1.0

---

### `run_inference(...)`

主推論流程，完整步驟如下：

```
1. 初始化 logger
2. 載入模型（load_model）
3. 建立 BuildingDataset（val + test 合併，aug=False）
4. 建立 DataLoader
5. 對每個 batch：
   a. 模型 forward（torch.no_grad）
   b. sigmoid > 0.5 二值化得到 pred_mask
   c. 儲存 pred_mask 為 tif（值 0/1）
   d. 計算 pred vs GT 的 diff mask
   e. 計算 largest_cc_ratio
   f. 判定 flagged_mislabel
   g. 記錄到 rows
6. 寫出 CSV
7. 計算並 log 整體 Precision / Recall / F1
```

---

## 4. 輸出檔案

每次執行會產生兩組輸出（對應兩個 checkpoint）：

```
inference_results/
├── mislabel_epoch50_correct.csv   ← correction 階段 checkpoint 的結果
├── mislabel_epoch50_correct.log
├── pred_masks_epoch50_correct/    ← 每張圖的預測 mask tif（值 0/1）
├── mislabel_epoch30_warmup.csv    ← warm-up 階段 checkpoint 的結果
├── mislabel_epoch30_warmup.log
└── pred_masks_epoch30_warmup/
```

### CSV 欄位說明

| 欄位 | 說明 |
|------|------|
| `fname` | 圖片檔案名稱 |
| `pred_positive_pixels` | 模型預測為建築物（1）的像素數 |
| `pred_negative_pixels` | 模型預測為背景（0）的像素數 |
| `gt_positive_pixels` | GT 中建築物像素數 |
| `gt_negative_pixels` | GT 中背景像素數 |
| `diff_pixels` | pred 與 GT 差異的像素總數 |
| `diff_ratio` | diff_pixels / 圖片總像素（0~1） |
| `largest_cc_ratio` | 差異區域最大連通區域佔比（0~1） |
| `flagged_mislabel` | 是否被判定為錯標（1=有，0=無） |

---

## 5. 執行指令

```bash
python py_scripts/inference_mislabel_detection.py \
  --data_path    "你的資料集路徑/Massachusetts/png" \
  --noise_dir_name "ns_seg_1" \
  --ckpt_correct "路徑/checkpoint_correct_mcr_1_epoch_50.pth" \
  --ckpt_warmup  "路徑/checkpoint_mcr_1_epoch_30.pth" \
  --output_dir   "./inference_results" \
  --splits val test \
  --threshold    0.50 \
  --model_key    model_state_dict_mep \
  --batch_size   25 \
  --num_workers  0
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data_path` | 必填 | 資料集根目錄，需指到包含 `/val`、`/test` 的那一層 |
| `--noise_dir_name` | `ns_seg_1` | noisy label 資料夾名稱 |
| `--ckpt_correct` | epoch50 路徑 | correction 階段 checkpoint |
| `--ckpt_warmup` | epoch30 路徑 | warm-up 階段 checkpoint |
| `--output_dir` | `./inference_results` | 輸出目錄 |
| `--splits` | `val test` | 要推論的 split，可以只填一個 |
| `--threshold` | `0.50` | 最大連通區域佔比閾值 |
| `--model_key` | `model_state_dict_mep` | 要載入哪個模型 |
| `--batch_size` | `25` | 不影響結果，只影響速度 |
| `--num_workers` | `0` | DataLoader 的 worker 數 |

---

## 6. 注意事項

**warm-up checkpoint 的 model_key 自動切換**

`checkpoint_mcr_1_epoch_30.pth` 是 warm-up 階段存的，裡面沒有 `model_state_dict_mep`（因為 MEP 在 warm-up 早期幾乎沒有收斂）。腳本會自動偵測並切換為 `model_state_dict_mit`，不需要手動修改。

**threshold 的選擇**

預設 0.50 代表「差異區域中最大的連通塊必須佔整張圖 50% 以上才算錯標」，這是一個相對嚴格的標準。可以根據實驗結果調整：
- 調低（如 0.20）→ 更容易 flag
- 調高（如 0.70）→ 更難 flag

**資料夾結構要求**

```
data_path/
├── val/
│   ├── data/       ← 原始圖像
│   ├── seg/        ← GT label
│   └── ns_seg_1/   ← noisy label
└── test/
    ├── data/
    ├── seg/
    └── ns_seg_1/
```

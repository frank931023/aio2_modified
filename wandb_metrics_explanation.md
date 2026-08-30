# WandB 指標說明文件

---

## 1. 指標基本定義

這個訓練框架共追蹤 5 個核心指標，全部都是針對**二元語意分割**（建築物 vs. 背景）計算：

| 指標名稱 | 英文全名 | 意義 | 範圍 | 越高越好？ |
|---------|---------|------|------|-----------|
| `oa` | Overall Accuracy | 所有像素中預測正確的比例 | 0–1 | ✅ |
| `precise` | Precision | 預測為建築物的像素中，真的是建築物的比例 | 0–1 | ✅ |
| `recall` | Recall | 真實建築物像素中，被正確預測出來的比例 | 0–1 | ✅ |
| `dice` | Dice Coefficient | 預測與真實 mask 的重疊程度（F1 的幾何版本） | 0–1 | ✅ |
| `iou` | Intersection over Union | 預測與真實 mask 的交集除以聯集 | 0–1 | ✅ |

### 公式

```
OA       = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
Dice      = 2 * TP / (2 * TP + FP + FN)
IoU       = TP / (TP + FP + FN)
```

> TP = 預測為建築物且正確；FP = 預測為建築物但錯誤；FN = 漏掉的建築物

---

## 2. 前綴命名規則解碼

WandB 的指標名稱由 **前綴（prefix）** + **指標名** 組成，前綴說明了「哪個模型」在「哪個資料集」上、對「哪種 label」評估。

### 2.1 前綴 `bc_` vs. 無前綴

| 前綴 | 意義 |
|------|------|
| `bc_` | Before Correction，**warm-up 階段**（Epoch 1–36）的指標 |
| 無前綴 | **correction 階段**（Epoch 21–50）的指標 |

### 2.2 模型識別碼

| 識別碼 | 模型 | 說明 |
|--------|------|------|
| （無）| Student Model | 主要訓練的 UNet |
| `mit_` | EMA-MIT | 每個 iteration 更新的 EMA Teacher Model（alpha=0.999） |
| `mep_` | EMA-MEP | 每個 epoch 更新的 EMA Teacher Model（alpha=0.99） |

### 2.3 資料集識別碼

| 識別碼 | 資料集 | 對比 label |
|--------|--------|-----------|
| `tr_` | Training set | Noisy labels |
| `trg_` | Training set | **GT labels**（真實標記，用來衡量真實準確度） |
| `v_` | Validation set | Noisy labels |
| `vg_` | Validation set | **GT labels** |
| `ts_` | Test set | **GT labels** |
| `tsn_` | Test set | Noisy labels |

### 2.4 完整命名範例

```
bc_mit_vg_dice
│   │   │  └── 指標：Dice Coefficient
│   │   └────── 資料集：Validation set，對比 GT labels
│   └────────── 模型：EMA-MIT（iteration-level EMA teacher）
└────────────── 階段：Before Correction（warm-up 階段）
```

---

## 3. 所有指標分組解說

### 3.1 Loss（損失值）

| 指標 | 說明 | 最終值 |
|------|------|--------|
| `bc_loss` | Warm-up 階段的總 loss（BCE + Dice） | 0.386 |
| `bc_loss_c` | Warm-up 階段的 BCE loss | 0.139 |
| `bc_loss_d` | Warm-up 階段的 Dice loss | 0.247 |
| `loss` | Correction 階段的總 loss | 0.232 |
| `loss_c` | Correction 階段的 BCE loss | 0.106 |
| `loss_d` | Correction 階段的 Dice loss | 0.126 |

> Correction 階段的 loss 比 warm-up 低，代表 label correction 有效提升訓練品質。

---

### 3.2 Student Model 訓練指標

#### 對 Noisy Labels 評估（`tr_`）

| 指標 | 說明 | 最終值 |
|------|------|--------|
| `tr_oa` | 訓練集 OA（對 noisy label） | 0.927 |
| `tr_precise` | 訓練集 Precision（對 noisy label） | 0.756 |
| `tr_recall` | 訓練集 Recall（對 noisy label） | 0.943 |
| `tr_dice` | 訓練集 Dice（對 noisy label） | 0.745 |
| `tr_iou` | 訓練集 IoU（對 noisy label） | 0.721 |

#### 對 GT Labels 評估（`trg_`）— 真實準確度

| 指標 | 說明 | 最終值 | 備註 |
|------|------|--------|------|
| `trg_oa` | 訓練集 OA（對 GT） | 0.822 | |
| `trg_precise` | 訓練集 Precision（對 GT） | 0.518 | 偏低，代表有誤判 |
| `trg_recall` | 訓練集 Recall（對 GT） | 0.668 | |
| `trg_dice` | 訓練集 Dice（對 GT） | 1.047 | ⚠️ 見下方說明 |
| `trg_iou` | 訓練集 IoU（對 GT） | 18.78 | ⚠️ 見下方說明 |

> ⚠️ **`trg_iou` 和 `trg_dice` 數值異常（>1）**：這是程式的 bug。`evaluate_batch` 函式在計算 batch-level 累積時，沒有除以 batch 數就直接加總，導致 epoch 結束後的平均值被錯誤放大。這些數值**不可信**，請忽略。

---

### 3.3 Validation 指標（Student Model）

#### 對 Noisy Labels（`v_`）

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `v_oa` | 0.977 | 97.7% 像素預測正確 |
| `v_precise` | 0.854 | 85.4% 的預測建築物是真的 |
| `v_recall` | 0.996 | 99.6% 的建築物被找到 |
| `v_dice` | 0.853 | |
| `v_iou` | 0.852 | |

#### 對 GT Labels（`vg_`）— 最重要的驗證指標

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `vg_oa` | 0.974 | 97.4% 像素預測正確 |
| `vg_precise` | 0.845 | 84.5% 的預測建築物是真的 |
| `vg_recall` | 0.987 | 98.7% 的建築物被找到 |
| `vg_dice` | 0.863 | **主要評估指標** |
| `vg_iou` | — | 未顯示（可能有同樣 bug） |

---

### 3.4 Test 指標（Student Model）

#### 對 GT Labels（`ts_`）— 最終測試結果

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `ts_oa` | 0.944 | 94.4% 像素預測正確 |
| `ts_precise` | 0.748 | 74.8% 的預測建築物是真的 |
| `ts_recall` | 0.962 | 96.2% 的建築物被找到 |
| `ts_dice` | 0.805 | |
| `ts_iou` | 2.43 | ⚠️ 異常，同上 bug |

#### 對 Noisy Labels（`tsn_`）

| 指標 | 最終值 |
|------|--------|
| `tsn_oa` | 0.956 |
| `tsn_precise` | 0.774 |
| `tsn_recall` | 0.991 |
| `tsn_dice` | 0.772 |
| `tsn_iou` | 0.768 |

---

### 3.5 EMA-MIT Model 指標（`mit_` / `bc_mit_`）

EMA-MIT 是每個 iteration 更新的 teacher model，通常比 student 更穩定。

#### Warm-up 階段訓練指標（`bc_mit_tr_`）

| 指標 | 最終值 | 對比 Student (`bc_tr_`) |
|------|--------|------------------------|
| `bc_mit_tr_oa` | 0.953 | 0.950 |
| `bc_mit_tr_precise` | 0.888 | 0.883 |
| `bc_mit_tr_recall` | 0.900 | 0.894 |
| `bc_mit_tr_dice` | 0.830 | 0.821 |
| `bc_mit_tr_iou` | 0.809 | 0.799 |

> EMA-MIT 在訓練集上略優於 student，符合預期（EMA 平滑效果）。

#### Warm-up 階段 Validation GT（`bc_mit_vg_`）

| 指標 | 最終值 |
|------|--------|
| `bc_mit_vg_oa` | 0.988 |
| `bc_mit_vg_precise` | 0.943 |
| `bc_mit_vg_recall` | 0.987 |
| `bc_mit_vg_dice` | 0.955 |

#### Correction 階段 Validation GT（`mit_vg_`）

| 指標 | 最終值 |
|------|--------|
| `mit_vg_oa` | 0.973 |
| `mit_vg_precise` | 0.828 |
| `mit_vg_recall` | 0.987 |
| `mit_vg_dice` | 0.846 |

> 有趣的是，correction 階段的 EMA-MIT 指標反而比 warm-up 階段低一些，這可能是因為 reload 到 Epoch 20 後，teacher model 也被重置了。

#### Correction 階段 Test GT（`mit_ts_`）

| 指標 | 最終值 |
|------|--------|
| `mit_ts_oa` | 0.942 |
| `mit_ts_precise` | 0.729 |
| `mit_ts_recall` | 0.962 |
| `mit_ts_dice` | 0.787 |

---

### 3.6 EMA-MEP Model 指標（`mep_` / `bc_mep_`）

EMA-MEP 是每個 epoch 更新的 teacher model（alpha=0.99，更新較慢）。

#### Warm-up 階段（`bc_mep_`）— 數值異常

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `bc_mep_ts_oa` | 0.004 | ⚠️ 極低，模型幾乎沒學到東西 |
| `bc_mep_ts_recall` | 0.962 | recall 高但 precision 為 0 |
| `bc_mep_ts_precise` | 0.0 | ⚠️ 完全沒有 precision |
| `bc_mep_v_recall` | 0.9998 | recall 接近 1 |
| `bc_mep_v_precise` | 0.004 | ⚠️ precision 極低 |

> ⚠️ **EMA-MEP 在 warm-up 階段表現極差**：recall 接近 1 但 precision 接近 0，代表模型把**所有像素都預測為建築物**（全部猜正類）。這是 EMA-MEP 更新太慢（alpha=0.99，每 epoch 才更新一次）導致在早期訓練時幾乎沒有學到有效特徵。

#### Correction 階段（`mep_`）— 恢復正常

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `mep_ts_oa` | 0.975 | 正常 |
| `mep_ts_precise` | 0.834 | 正常 |
| `mep_ts_recall` | 0.962 | 正常 |
| `mep_ts_dice` | 0.844 | 正常 |
| `mep_vg_dice` | 0.906 | 驗證集表現不錯 |

---

### 3.7 Label Correction 相關指標

#### `ns_`：修正後 label 的品質（對 GT 評估）

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `ns_oa` | 0.843 | 修正後 label 的 OA |
| `ns_precise` | 0.506 | 修正後 label 的 Precision |
| `ns_recall` | 0.668 | 修正後 label 的 Recall |
| `ns_dice` | 1.158 | ⚠️ 同樣的累積 bug |
| `ns_iou` | 75.78 | ⚠️ 同樣的累積 bug |

> 這些指標衡量「修正後的 label 有多接近 GT」，OA=0.843 代表修正後的 label 有 84.3% 的像素與 GT 一致。

#### `mit_ptrg_`：Teacher 預測對 GT 的準確度

| 指標 | 最終值 | 說明 |
|------|--------|------|
| `mit_ptrg_oa` | 0.822 | Teacher 預測的 OA |
| `mit_ptrg_precise` | 0.520 | Teacher 預測的 Precision |
| `mit_ptrg_recall` | 0.668 | Teacher 預測的 Recall |

> 這是 teacher model 在做 label correction 時，其預測結果對 GT 的準確度。

---

## 4. 最終結果總覽（Epoch 50）

### Student Model 最終表現

| 評估場景 | OA | Precision | Recall | Dice |
|---------|-----|-----------|--------|------|
| Val（noisy label） | 0.977 | 0.854 | 0.996 | 0.853 |
| Val（GT label）✅ | 0.974 | 0.845 | 0.987 | **0.863** |
| Test（GT label）✅ | 0.944 | 0.748 | 0.962 | **0.805** |
| Test（noisy label） | 0.956 | 0.774 | 0.991 | 0.772 |

### EMA-MIT Model 最終表現

| 評估場景 | OA | Precision | Recall | Dice |
|---------|-----|-----------|--------|------|
| Val（GT label）✅ | 0.973 | 0.828 | 0.987 | **0.846** |
| Test（GT label）✅ | 0.942 | 0.729 | 0.962 | **0.787** |

### EMA-MEP Model 最終表現

| 評估場景 | OA | Precision | Recall | Dice |
|---------|-----|-----------|--------|------|
| Val（GT label）✅ | 0.994 | 0.917 | 0.987 | **0.906** |
| Test（GT label）✅ | 0.975 | 0.834 | 0.962 | **0.844** |

> EMA-MEP 在 correction 階段的最終表現最好（Dice 0.906 on Val GT），這符合 EMA 平滑的特性——更新慢的模型在後期往往更穩定。

---

## 5. 哪些指標最重要？

| 優先級 | 指標 | 原因 |
|--------|------|------|
| ⭐⭐⭐ | `vg_dice` / `ts_dice` | 對 GT 的 Dice，最能反映真實分割品質 |
| ⭐⭐⭐ | `vg_iou` / `ts_iou` | 對 GT 的 IoU，語意分割標準指標 |
| ⭐⭐ | `vg_recall` | 建築物的召回率，漏掉建築物的代價高 |
| ⭐⭐ | `vg_precise` | 誤判率，避免把非建築物標成建築物 |
| ⭐ | `v_dice` / `ts_dice` | 對 noisy label 的指標，僅供參考 |
| ❌ | `trg_iou` > 1 的值 | 程式 bug，數值不可信 |
| ❌ | `bc_mep_*` 的異常值 | EMA-MEP 早期未收斂，不可信 |

---

## 6. 指標異常值說明

程式中有一個已知的累積計算 bug：`evaluate_batch` 函式在 epoch 內累積 batch-level 指標時，`iou` 和 `dice` 是直接加總 tensor 值而非平均，導致以下指標數值超過 1（理論上限）：

- `trg_iou`、`trg_dice`（>1）
- `bc_trg_iou`、`bc_trg_dice`（>1）
- `mit_trg_iou`、`mit_trg_dice`（>1）
- `ns_iou`、`ns_dice`（>1）
- `ts_iou`、`mit_ts_iou` 等部分 test 指標（>1）

**這些數值請直接忽略**，以 `v_dice`、`vg_dice`、`ts_dice` 等 epoch-level evaluate 函式計算的結果為準。

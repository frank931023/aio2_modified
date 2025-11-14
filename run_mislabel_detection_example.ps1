# AIO2 錯標檢測功能使用範例腳本 (PowerShell)

Write-Host "=== AIO2 錯標檢測功能測試 ===" -ForegroundColor Green

# 設定基本參數
$DATA_PATH = "path\to\your\massachusetts\dataset"
$SAVE_DIR = "Results_with_mislabel_detection"
$EPOCHS = 10
$BATCH_SIZE = 16

Write-Host "開始使用 emaCorrect 方法進行訓練並啟用錯標檢測..." -ForegroundColor Yellow

# 範例 1: 基本錯標檢測
$command = @"
python py_scripts\train_unet_png_emaCorrect_singleGPU.py `
    --data_path "$DATA_PATH" `
    --save_dir "$SAVE_DIR" `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --enable_mislabel_detection `
    --detection_confidence_threshold 0.8 `
    --detection_agreement_threshold 0.7 `
    --detection_save_interval 5 `
    --cal_tr_acc `
    --batch_to_wandb `
    --wandb_mode offline
"@

Write-Host "執行命令:" -ForegroundColor Cyan
Write-Host $command -ForegroundColor White

# 執行訓練 (註解掉實際執行，避免在沒有資料的情況下出錯)
# Invoke-Expression $command

Write-Host "訓練完成！檢查結果..." -ForegroundColor Yellow

# 檢查輸出結果
Write-Host "=== 檢測結果文件 ===" -ForegroundColor Green

if (Test-Path $SAVE_DIR) {
    Write-Host "找到保存目錄: $SAVE_DIR" -ForegroundColor Green
    
    # 查找錯標檢測結果目錄
    $detectionDirs = Get-ChildItem -Path $SAVE_DIR -Recurse -Directory -Name "mislabel_detection" -ErrorAction SilentlyContinue
    
    foreach ($dir in $detectionDirs) {
        $fullPath = Join-Path $SAVE_DIR $dir
        Write-Host "檢測結果目錄: $fullPath" -ForegroundColor White
        
        # 列出批次檢測檔案
        $batchFiles = Get-ChildItem -Path $fullPath -Filter "detection_epoch_*_batch_*.json" -ErrorAction SilentlyContinue
        Write-Host "  批次檢測檔案數: $($batchFiles.Count)" -ForegroundColor White
        
        # 列出 epoch 匯總檔案
        $epochFiles = Get-ChildItem -Path $fullPath -Filter "epoch_*_summary.json" -ErrorAction SilentlyContinue
        Write-Host "  Epoch 匯總檔案數: $($epochFiles.Count)" -ForegroundColor White
        
        # 顯示最新的匯總結果
        if ($epochFiles.Count -gt 0) {
            $latestSummary = $epochFiles | Sort-Object Name | Select-Object -Last 1
            Write-Host "  最新匯總結果: $($latestSummary.FullName)" -ForegroundColor White
            Write-Host "  內容預覽:" -ForegroundColor Gray
            
            try {
                $content = Get-Content $latestSummary.FullName -ErrorAction SilentlyContinue | Select-Object -First 20
                foreach ($line in $content) {
                    Write-Host "    $line" -ForegroundColor Gray
                }
            } catch {
                Write-Host "    無法讀取檔案內容" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "未找到保存目錄: $SAVE_DIR" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 可用的訓練檔案與錯標檢測支援狀態 ===" -ForegroundColor Green
Write-Host "✅ train_unet_png_emaCorrect_singleGPU.py - 已完整支援" -ForegroundColor Green
Write-Host "🔄 train_unet_png_pixelCorrect_singleGPU.py - 部分支援" -ForegroundColor Yellow
Write-Host "🔄 train_unet_h5_smp_emaCorrect_singleGPU.py - 部分支援" -ForegroundColor Yellow
Write-Host "⏳ 其他檔案 - 待添加支援" -ForegroundColor Red

Write-Host ""
Write-Host "=== 使用說明 ===" -ForegroundColor Green
Write-Host "1. 修改 `$DATA_PATH 變數指向您的資料集路徑"
Write-Host "2. 確保資料集包含 gt (真實標籤) 和 noisy labels"
Write-Host "3. 取消註解 Invoke-Expression 行以實際執行訓練"
Write-Host "4. 運行腳本後檢查 Results_with_mislabel_detection 目錄"
Write-Host "5. 查看 mislabel_detection 子目錄中的 JSON 結果檔案"

Write-Host ""
Write-Host "=== 參數說明 ===" -ForegroundColor Green
Write-Host "--enable_mislabel_detection: 啟用錯標檢測功能"
Write-Host "--detection_confidence_threshold: 預測信心度閾值 (0.0-1.0)"
Write-Host "--detection_agreement_threshold: 教師學生一致性閾值 (0.0-1.0)"
Write-Host "--detection_save_interval: 檢測間隔 (每N個批次檢測一次)"
Write-Host "--enable_detection_visualization: 啟用視覺化保存 (可選)"

Write-Host ""
Write-Host "=== 範例輸出結果 ===" -ForegroundColor Green
Write-Host "批次檢測結果檔案: detection_epoch_5_batch_10.json"
Write-Host "Epoch 匯總檔案: epoch_5_summary.json"
Write-Host "終端輸出: 錯標檢測 - Epoch 6, Batch 15: Precision=0.752, Recall=0.681, F1=0.714"

Write-Host ""
Write-Host "腳本執行完成！" -ForegroundColor Green
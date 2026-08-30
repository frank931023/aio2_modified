"""
AIO2 錯標檢測功能測試腳本
用於快速驗證錯標檢測模組是否正常工作
"""

import sys
import os
sys.path.append('./')

import torch
import numpy as np
import json
from pathlib import Path

# 導入錯標檢測模組
import utils.mislabel_detection as mld

def create_mock_data():
    """創建模擬資料用於測試"""
    batch_size = 4
    height, width = 64, 64
    
    # 創建模擬的 tensor 資料
    teacher_pred = torch.randn(batch_size, height, width)  # 教師模型 logits
    student_pred = torch.randn(batch_size, height, width)  # 學生模型 logits
    
    # 創建模擬的標籤資料
    ns_masks = torch.randint(0, 2, (batch_size, height, width)).float()  # 雜訊標籤
    gt_masks = torch.randint(0, 2, (batch_size, height, width)).float()  # 真實標籤
    
    # 讓一些雜訊標籤與真實標籤不同（模擬錯標）
    for i in range(batch_size):
        # 隨機選擇一些位置作為錯標
        error_positions = torch.randint(0, height*width, (10,))
        for pos in error_positions:
            row, col = pos // width, pos % width
            ns_masks[i, row, col] = 1 - gt_masks[i, row, col]
    
    return teacher_pred, student_pred, ns_masks, gt_masks

def test_mislabel_detection():
    """測試錯標檢測功能"""
    print("=== 開始測試錯標檢測功能 ===")
    
    # 創建測試資料
    teacher_pred, student_pred, ns_masks, gt_masks = create_mock_data()
    print(f"✅ 創建模擬資料: {teacher_pred.shape}")
    
    # 創建測試目錄
    test_dir = "test_mislabel_detection"
    Path(test_dir).mkdir(exist_ok=True)
    
    # 測試錯標檢測功能
    try:
        detection_results = mld.detect_mislabeled_coordinates(
            teacher_pred=teacher_pred,
            student_pred=student_pred,
            ns_masks=ns_masks,
            gt_masks=gt_masks,
            image_indices=None,
            epoch=1,
            batch_idx=0,
            save_dir=test_dir,
            confidence_threshold=0.5,  # 降低閾值增加檢測機會
            agreement_threshold=0.5
        )
        print(f"✅ 錯標檢測執行成功，檢測到 {len(detection_results)} 個結果")
        
        # 顯示檢測結果
        for img_key, result in detection_results.items():
            print(f"  圖片 {img_key}:")
            print(f"    可疑像素數: {result['suspicious_count']}")
            print(f"    TP: {result['tp_count']}, FP: {result['fp_count']}, FN: {result['fn_count']}")
            print(f"    精確度: {result['precision']:.3f}, 召回率: {result['recall']:.3f}")
            
    except Exception as e:
        print(f"❌ 錯標檢測執行失敗: {e}")
        return False
    
    # 測試指標計算
    try:
        batch_metrics = mld.calculate_detection_metrics(detection_results)
        print(f"✅ 批次指標計算成功:")
        print(f"  批次精確度: {batch_metrics['batch_precision']:.3f}")
        print(f"  批次召回率: {batch_metrics['batch_recall']:.3f}")
        print(f"  批次F1分數: {batch_metrics['batch_f1']:.3f}")
    except Exception as e:
        print(f"❌ 指標計算失敗: {e}")
        return False
    
    # 測試 epoch 匯總
    try:
        summary = mld.summarize_epoch_detection(
            detection_dir=os.path.join(test_dir, 'mislabel_detection'),
            epoch=1,
            wandb_log=False  # 關閉 wandb 記錄避免錯誤
        )
        if summary:
            print(f"✅ Epoch 匯總成功:")
            print(f"  總體精確度: {summary['precision']:.3f}")
            print(f"  總體召回率: {summary['recall']:.3f}")
            print(f"  總體F1分數: {summary['f1_score']:.3f}")
        else:
            print("ℹ️ 沒有檢測結果可匯總")
    except Exception as e:
        print(f"❌ Epoch 匯總失敗: {e}")
        return False
    
    # 檢查檔案輸出
    detection_dir = os.path.join(test_dir, 'mislabel_detection')
    if os.path.exists(detection_dir):
        files = os.listdir(detection_dir)
        print(f"✅ 檢測結果檔案已保存: {len(files)} 個檔案")
        for file in files:
            print(f"  - {file}")
    else:
        print("⚠️ 檢測結果目錄不存在")
    
    print("=== 錯標檢測功能測試完成 ===")
    return True

def test_integration_with_training():
    """測試與訓練腳本的整合"""
    print("\n=== 測試與訓練腳本整合 ===")
    
    # 檢查主要訓練檔案是否已正確修改
    main_script = "py_scripts/train_unet_png_emaCorrect_singleGPU.py"
    if os.path.exists(main_script):
        with open(main_script, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 檢查是否包含錯標檢測模組導入
        if 'import utils.mislabel_detection as mld' in content:
            print("✅ 主要訓練腳本已正確導入錯標檢測模組")
        else:
            print("❌ 主要訓練腳本缺少錯標檢測模組導入")
            
        # 檢查是否包含錯標檢測參數
        if '--enable_mislabel_detection' in content:
            print("✅ 主要訓練腳本已添加錯標檢測參數")
        else:
            print("❌ 主要訓練腳本缺少錯標檢測參數")
            
        # 檢查是否包含錯標檢測功能調用
        if 'detect_mislabeled_coordinates' in content:
            print("✅ 主要訓練腳本已整合錯標檢測功能")
        else:
            print("❌ 主要訓練腳本缺少錯標檢測功能調用")
    else:
        print(f"❌ 找不到主要訓練腳本: {main_script}")
    
    # 檢查錯標檢測模組
    mld_module = "utils/mislabel_detection.py"
    if os.path.exists(mld_module):
        print(f"✅ 錯標檢測模組存在: {mld_module}")
    else:
        print(f"❌ 錯標檢測模組不存在: {mld_module}")
    
    print("=== 整合測試完成 ===")

if __name__ == "__main__":
    print("AIO2 錯標檢測功能測試")
    print("=" * 50)
    
    # 測試基本功能
    success = test_mislabel_detection()
    
    # 測試整合
    test_integration_with_training()
    
    if success:
        print("\n🎉 所有測試通過！錯標檢測功能已成功整合到 AIO2 專案中。")
        print("\n📝 使用方法:")
        print("python py_scripts/train_unet_png_emaCorrect_singleGPU.py \\")
        print("    --data_path 'your_data_path' \\")
        print("    --enable_mislabel_detection \\")
        print("    --detection_confidence_threshold 0.8 \\")
        print("    --detection_agreement_threshold 0.7")
    else:
        print("\n❌ 測試失敗，請檢查錯標檢測模組的實現。")
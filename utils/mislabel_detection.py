# -*- coding: utf-8 -*-
"""
錯標檢測工具模組
用於檢測語義分割中的錯誤標籤並輸出座標位置
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def detect_mislabeled_coordinates(teacher_pred: torch.Tensor, 
                                student_pred: torch.Tensor, 
                                ns_masks: torch.Tensor, 
                                gt_masks: Optional[torch.Tensor], 
                                image_indices: Optional[List[str]], 
                                epoch: int, 
                                batch_idx: int, 
                                save_dir: str,
                                confidence_threshold: float = 0.8, 
                                agreement_threshold: float = 0.7) -> Dict:
    """
    檢測錯標並輸出座標位置
    
    參數說明:
        teacher_pred: 教師模型預測結果 (B, H, W) - logits
        student_pred: 學生模型預測結果 (B, H, W) - logits  
        ns_masks: 雜訊標籤 (B, H, W)
        gt_masks: 真實標籤 (B, H, W), 可選
        image_indices: 圖片索引列表
        epoch: 當前訓練輪數
        batch_idx: 當前批次索引
        save_dir: 保存目錄
        confidence_threshold: 預測信心度閾值
        agreement_threshold: 模型間一致性閾值
        
    返回:
        detection_results: 檢測結果字典
    """
    detection_results = {}
    batch_size = teacher_pred.shape[0]
    
    # 建立保存目錄
    detection_dir = os.path.join(save_dir, 'mislabel_detection')
    os.makedirs(detection_dir, exist_ok=True)
    
    for b in range(batch_size):
        # 設定圖片標識符
        img_idx = image_indices[b] if image_indices else f"epoch_{epoch}_batch_{batch_idx}_img_{b}"
        
        # 將 logits 轉為機率
        teacher_prob = torch.sigmoid(teacher_pred[b])
        student_prob = torch.sigmoid(student_pred[b])
        
        # 二值化預測
        teacher_binary = (teacher_prob > 0.5).float()
        student_binary = (student_prob > 0.5).float()
        
        # 計算模型間一致性
        model_agreement = (teacher_binary == student_binary).float()
        
        # 計算預測信心度
        teacher_confidence = torch.abs(teacher_prob - 0.5) * 2
        student_confidence = torch.abs(student_prob - 0.5) * 2
        avg_confidence = (teacher_confidence + student_confidence) / 2
        
        # 檢測可疑區域的條件:
        # 1. 高信心度預測
        high_confidence_mask = avg_confidence > confidence_threshold
        # 2. 教師學生模型一致
        high_agreement_mask = model_agreement > agreement_threshold
        # 3. 模型預測與雜訊標籤不一致
        disagree_with_noisy = (teacher_binary != ns_masks[b].float())
        
        # 可疑錯標區域 = 三個條件同時滿足
        suspicious_mask = high_confidence_mask & high_agreement_mask & disagree_with_noisy
        
        # 提取可疑像素座標
        suspicious_coords = torch.nonzero(suspicious_mask, as_tuple=False)
        
        if len(suspicious_coords) > 0:
            coords_list = suspicious_coords.cpu().numpy().tolist()
            
            # 如果有真實標籤，計算 TP/FP/FN
            tp_coords, fp_coords, fn_coords = [], [], []
            tp_count = fp_count = fn_count = 0
            
            if gt_masks is not None:
                # 真實錯標位置：雜訊標籤與真實標籤不一致的地方
                true_mislabels = (ns_masks[b] != gt_masks[b])
                
                # TP: 正確檢測到的錯標
                tp_mask = suspicious_mask & true_mislabels
                tp_coords = torch.nonzero(tp_mask, as_tuple=False).cpu().numpy().tolist()
                tp_count = len(tp_coords)
                
                # FP: 誤檢的正確標籤
                fp_mask = suspicious_mask & (~true_mislabels)
                fp_coords = torch.nonzero(fp_mask, as_tuple=False).cpu().numpy().tolist()
                fp_count = len(fp_coords)
                
                # FN: 未檢測到的錯標
                fn_mask = (~suspicious_mask) & true_mislabels
                fn_coords = torch.nonzero(fn_mask, as_tuple=False).cpu().numpy().tolist()
                fn_count = len(fn_coords)
            
            # 保存檢測結果
            detection_results[img_idx] = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'suspicious_coords': coords_list,
                'suspicious_count': len(coords_list),
                'tp_coords': tp_coords,
                'fp_coords': fp_coords, 
                'fn_coords': fn_coords,
                'tp_count': tp_count,
                'fp_count': fp_count,
                'fn_count': fn_count,
                'precision': tp_count / max(tp_count + fp_count, 1),
                'recall': tp_count / max(tp_count + fn_count, 1),
                'confidence_threshold': confidence_threshold,
                'agreement_threshold': agreement_threshold
            }
    
    # 保存批次檢測結果到檔案
    if detection_results:
        save_path = os.path.join(detection_dir, f'detection_epoch_{epoch}_batch_{batch_idx}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, indent=2, ensure_ascii=False)
    
    return detection_results


def summarize_epoch_detection(detection_dir: str, epoch: int, wandb_log: bool = True) -> Optional[Dict]:
    """
    匯總單個 epoch 的錯標檢測結果
    
    參數說明:
        detection_dir: 檢測結果目錄
        epoch: 輪數
        wandb_log: 是否記錄到 wandb
        
    返回:
        summary: 匯總統計結果
    """
    import glob
    
    # 找到該 epoch 的所有檢測檔案
    pattern = os.path.join(detection_dir, f'detection_epoch_{epoch}_batch_*.json')
    detection_files = glob.glob(pattern)
    
    if not detection_files:
        return None
    
    # 匯總統計
    total_tp = total_fp = total_fn = 0
    total_suspicious = 0
    all_detections = {}
    
    for file_path in detection_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
            
        for img_key, result in batch_results.items():
            all_detections[img_key] = result
            total_tp += result['tp_count']
            total_fp += result['fp_count'] 
            total_fn += result['fn_count']
            total_suspicious += result['suspicious_count']
    
    # 計算 epoch 級別指標
    epoch_precision = total_tp / max(total_tp + total_fp, 1)
    epoch_recall = total_tp / max(total_tp + total_fn, 1)
    epoch_f1 = 2 * epoch_precision * epoch_recall / max(epoch_precision + epoch_recall, 1e-6)
    
    summary = {
        'epoch': epoch,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_suspicious': total_suspicious,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1_score': epoch_f1,
        'num_images_detected': len(all_detections)
    }
    
    # 保存 epoch 匯總
    summary_path = os.path.join(detection_dir, f'epoch_{epoch}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 記錄到 wandb
    if wandb_log:
        try:
            import wandb
            wandb.log({
                'mislabel_detection/epoch_precision': epoch_precision,
                'mislabel_detection/epoch_recall': epoch_recall,
                'mislabel_detection/epoch_f1': epoch_f1,
                'mislabel_detection/epoch_tp': total_tp,
                'mislabel_detection/epoch_fp': total_fp,
                'mislabel_detection/epoch_fn': total_fn,
                'epoch': epoch
            })
        except:
            pass  # 如果 wandb 不可用就跳過
    
    return summary


def save_detection_visualization(teacher_pred: torch.Tensor,
                               student_pred: torch.Tensor,
                               ns_masks: torch.Tensor,
                               gt_masks: Optional[torch.Tensor],
                               suspicious_coords: List[List[int]],
                               tp_coords: List[List[int]],
                               fp_coords: List[List[int]],
                               save_path: str) -> None:
    """
    保存錯標檢測視覺化結果
    
    參數說明:
        teacher_pred: 教師模型預測
        student_pred: 學生模型預測  
        ns_masks: 雜訊標籤
        gt_masks: 真實標籤
        suspicious_coords: 可疑座標列表
        tp_coords: 真正例座標
        fp_coords: 假正例座標
        save_path: 保存路徑
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 將 tensor 轉為 numpy
    teacher_prob = torch.sigmoid(teacher_pred).cpu().numpy()
    student_prob = torch.sigmoid(student_pred).cpu().numpy()
    ns_mask = ns_masks.cpu().numpy()
    
    # 創建視覺化圖表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 顯示教師模型預測
    axes[0, 0].imshow(teacher_prob, cmap='gray')
    axes[0, 0].set_title('教師模型預測')
    axes[0, 0].axis('off')
    
    # 顯示學生模型預測
    axes[0, 1].imshow(student_prob, cmap='gray')
    axes[0, 1].set_title('學生模型預測')
    axes[0, 1].axis('off')
    
    # 顯示雜訊標籤
    axes[0, 2].imshow(ns_mask, cmap='gray')
    axes[0, 2].set_title('雜訊標籤')
    axes[0, 2].axis('off')
    
    # 顯示真實標籤（如果有的話）
    if gt_masks is not None:
        gt_mask = gt_masks.cpu().numpy()
        axes[1, 0].imshow(gt_mask, cmap='gray')
        axes[1, 0].set_title('真實標籤')
    else:
        axes[1, 0].text(0.5, 0.5, '無真實標籤', ha='center', va='center')
        axes[1, 0].set_title('真實標籤')
    axes[1, 0].axis('off')
    
    # 顯示檢測結果疊加圖
    overlay = np.zeros((*teacher_prob.shape, 3))  # RGB 圖像
    overlay[:, :, 0] = teacher_prob  # 紅色通道顯示預測
    
    # 標記不同類型的檢測結果
    for coord in suspicious_coords:
        y, x = coord
        overlay[max(0, y-2):min(overlay.shape[0], y+3), 
                max(0, x-2):min(overlay.shape[1], x+3), 1] = 1  # 可疑點用綠色
    
    for coord in tp_coords:
        y, x = coord
        overlay[max(0, y-1):min(overlay.shape[0], y+2), 
                max(0, x-1):min(overlay.shape[1], x+2), 2] = 1  # TP 用藍色
    
    for coord in fp_coords:
        y, x = coord
        overlay[max(0, y-1):min(overlay.shape[0], y+2), 
                max(0, x-1):min(overlay.shape[1], x+2), 0] = 1  # FP 用紅色
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('檢測結果疊加圖\n(綠:可疑, 藍:TP, 紅:FP)')
    axes[1, 1].axis('off')
    
    # 統計資訊
    stats_text = f"""檢測統計:
可疑像素數: {len(suspicious_coords)}
真正例 (TP): {len(tp_coords)}
假正例 (FP): {len(fp_coords)}
精確度: {len(tp_coords) / max(len(tp_coords) + len(fp_coords), 1):.3f}
召回率: {len(tp_coords) / max(len(tp_coords) + len(fp_coords), 1):.3f}"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('統計資訊')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_detection_metrics(detection_results: Dict) -> Dict:
    """
    計算批次級別的檢測指標
    
    參數說明:
        detection_results: 檢測結果字典
        
    返回:
        metrics: 指標字典
    """
    if not detection_results:
        return {
            'batch_precision': 0.0,
            'batch_recall': 0.0,
            'batch_f1': 0.0,
            'batch_tp': 0,
            'batch_fp': 0,
            'batch_fn': 0
        }
    
    # 匯總所有圖片的結果
    total_tp = sum([v['tp_count'] for v in detection_results.values()])
    total_fp = sum([v['fp_count'] for v in detection_results.values()])
    total_fn = sum([v['fn_count'] for v in detection_results.values()])
    
    # 計算指標
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'batch_precision': precision,
        'batch_recall': recall,
        'batch_f1': f1,
        'batch_tp': total_tp,
        'batch_fp': total_fp,
        'batch_fn': total_fn
    }
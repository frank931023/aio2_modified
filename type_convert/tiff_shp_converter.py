import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
import os
import imageio
import random


def convert_tif_shp_to_image(tif_path, shp_path, output_dir="images", partition="train", output_format="png"):
    """
    將 TIFF 影像 + SHP 標註轉成：
    1. RGB 衛星圖 (保留原始 geospatial info 若為 tif)
    2. Ground truth mask (0=背景, 1=建築物)
    
    Args:
        output_format: 'png' or 'tif'
        partition: 'train', 'val', or 'test'
    """
    
    # 檢查格式支援
    if output_format not in ['png', 'tif']:
        raise ValueError("output_format must be 'png' or 'tif'")

    # 建立輸出資料夾 (包含 partition)
    # 結構: output_dir / train / data
    data_dir = os.path.join(output_dir, partition, "data")
    # 結構: output_dir / train / seg
    seg_dir = os.path.join(output_dir, partition, "seg")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # 取得檔案名稱
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"[{partition}] 正在處理：{base_name}")

    # ------------------ 讀取 TIFF ------------------
    with rasterio.open(tif_path) as src:
        count = src.count
        transform = src.transform
        width = src.width
        height = src.height
        profile = src.profile.copy()
        
        # 讀取影像資料 (確保至少有3通道用於RGB)
        if count >= 3:
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
            # 組合為 (Height, Width, 3) 供 imageio 使用 (若是 tif 則需轉回 (3, H, W))
            rgb = np.stack([r, g, b], axis=-1) 
        else:
            band = src.read(1)
            rgb = np.stack([band] * 3, axis=-1)

    # Normalize to 0–255 (視覺化用)
    rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)

    # ------------------ 讀取 SHP ------------------
    gdf = gpd.read_file(shp_path)
    # print(f"SHP 包含 {len(gdf)} 個 polygons")

    # ------------------ 產生 segmentation mask ------------------
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # ------------------ 儲存 ------------------
    img_out_path = os.path.join(data_dir, f"{base_name}.{output_format}")
    mask_out_path = os.path.join(seg_dir, f"{base_name}.{output_format}")

    if output_format == 'png':
        imageio.imwrite(img_out_path, rgb_norm)
        imageio.imwrite(mask_out_path, mask * 255) # PNG 需要可視化 (0, 255)
    
    elif output_format == 'tif':
        # 更新 Profile 準備寫入 RGB Tiff
        profile.update(
            dtype=rasterio.uint8,
            count=3,
            driver='GTiff',
            compress='lzw'
        )
        
        # 1. 寫入 RGB 影像 (需轉回 (Count, Height, Width))
        # rgb_norm shape is (H, W, 3) -> (3, H, W)
        rgb_to_write = np.moveaxis(rgb_norm, -1, 0)
        
        with rasterio.open(img_out_path, 'w', **profile) as dst:
            dst.write(rgb_to_write)
            
        # 2. 寫入 Mask (單通道)
        profile.update(count=1)
        # 注意：為了維持與 AIO2 pipeline 的一致性 (0 與 255)，這裡存 255。
        with rasterio.open(mask_out_path, 'w', **profile) as dst:
            dst.write(mask * 255, 1)

    return img_out_path, mask_out_path

def get_common_names(tif_dir, shp_dir):
    tif_files = {os.path.splitext(f)[0] for f in os.listdir(tif_dir)
                 if f.lower().endswith(".tif")}
    shp_files = {os.path.splitext(f)[0] for f in os.listdir(shp_dir)
                 if f.lower().endswith(".shp")}

    common = sorted(list(tif_files & shp_files))
    return common  # return List

if __name__ == "__main__":
    # 設定亂數種子，確保每次分割結果一致
    random.seed(42)
    
    # 使用者指定的路徑
    banana_tiff_dir = r"D:\DMCIII_1220_banana_tif"
    banana_shp_dir = r"E:\crop\bananas\SHP"
    output_folder_dir = "bananan_images_converted"

    common_list = get_common_names(banana_tiff_dir, banana_shp_dir)
    print(f"找到共同檔案：{len(common_list)} 個")

    # 隨機打亂
    random.shuffle(common_list)
    
    # 計算分割數量 (8:1:1)
    total_count = len(common_list)
    n_train = int(total_count * 0.8)
    n_val = int(total_count * 0.1)
    # 剩下的都給 test，確保總數不變
    n_test = total_count - n_train - n_val
    
    print(f"分割計畫 -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    train_files = common_list[:n_train]
    val_files = common_list[n_train : n_train + n_val]
    test_files = common_list[n_train + n_val :]
    
    # 處理 Train
    for name in train_files:
        tif_path = os.path.join(banana_tiff_dir, name + ".tif")
        shp_path = os.path.join(banana_shp_dir, name + ".shp")
        convert_tif_shp_to_image(tif_path, shp_path, output_folder_dir, partition='train', output_format='tif')
        
    # 處理 Val
    for name in val_files:
        tif_path = os.path.join(banana_tiff_dir, name + ".tif")
        shp_path = os.path.join(banana_shp_dir, name + ".shp")
        convert_tif_shp_to_image(tif_path, shp_path, output_folder_dir, partition='val', output_format='tif')
        
    # 處理 Test
    for name in test_files:
        tif_path = os.path.join(banana_tiff_dir, name + ".tif")
        shp_path = os.path.join(banana_shp_dir, name + ".shp")
        convert_tif_shp_to_image(tif_path, shp_path, output_folder_dir, partition='test', output_format='tif')
        
    print("\n全部轉換並分割完成！")

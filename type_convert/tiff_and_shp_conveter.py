import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
import os
import imageio


def convert_tif_shp_to_png(tif_path, shp_path, output_dir="images"):
    """
    將 TIFF 影像 + SHP 標註轉成：
    1. RGB 衛星圖 PNG
    2. Ground truth mask PNG (0=背景, 1=建築物)
    """

    # 建立輸出資料夾
    data_dir = os.path.join(output_dir, "data")
    seg_dir = os.path.join(output_dir, "seg")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # 取得檔案名稱
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"正在處理：{base_name}")

    # ------------------ 讀取 TIFF ------------------
    with rasterio.open(tif_path) as src:
        count = src.count
        transform = src.transform
        width = src.width
        height = src.height

        if count >= 3:
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
            rgb = np.dstack([r, g, b])
        else:
            band = src.read(1)
            rgb = np.stack([band] * 3, axis=-1)

    # Normalize to 0–255
    rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)

    # ------------------ 讀取 SHP ------------------
    gdf = gpd.read_file(shp_path)
    print(f"SHP 包含 {len(gdf)} 個 polygons")

    # ------------------ 產生 segmentation mask ------------------
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # ------------------ 儲存 PNG ------------------
    img_out_path = os.path.join(data_dir, f"{base_name}.png")
    mask_out_path = os.path.join(seg_dir, f"{base_name}.png")

    imageio.imwrite(img_out_path, rgb_norm)
    imageio.imwrite(mask_out_path, mask * 255)

    print(f"衛星影像已儲存: {img_out_path}")
    print(f"ground truth 已儲存: {mask_out_path}")
    print("-------------------------------------------------\n")

    return img_out_path, mask_out_path

def get_common_names(tif_dir, shp_dir):
    tif_files = {os.path.splitext(f)[0] for f in os.listdir(tif_dir)
                 if f.lower().endswith(".tif")}
    shp_files = {os.path.splitext(f)[0] for f in os.listdir(shp_dir)
                 if f.lower().endswith(".shp")}

    common = sorted(list(tif_files & shp_files))
    return common  # return List

if __name__ == "__main__":
    banana_tiff_dir = r"D:\DMCIII_1220_banana_tif"
    banana_shp_dir = r"E:\crop\bananas\SHP"
    output_folder_dir = "bananan_images"

    common_list = get_common_names(banana_tiff_dir, banana_shp_dir)
    print("共同檔案：", common_list)

    for name in common_list:
        tif_path = os.path.join(banana_tiff_dir, name + ".tif")
        shp_path = os.path.join(banana_shp_dir, name + ".shp")
        convert_tif_shp_to_png(tif_path, shp_path, output_folder_dir)


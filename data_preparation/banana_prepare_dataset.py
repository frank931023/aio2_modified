# -*- coding: utf-8 -*-
"""
作物資料集前處理腳本（通用版）
功能：
  1. 從來源資料夾複製 TIF / TFW / AUX / NPY / SHP 等檔案
  2. 將 SHP 燒錄成與 TIF 對齊的二值 GeoTIFF mask（值 0/1）
  3. 將大張 TIF 裁切成 256×256 patches，data 與 seg 均存為 GeoTIFF
  4. 統計 mask 分布，進行下採樣：保留全部正樣本，負樣本補滿至總量 10%
  5. 依 8:1:1 切割 train / val / test
  6. 輸出符合 Massachusetts/png 格式的資料集
  7. 全程視覺化各階段統計資訊

使用 --crop 指定作物名稱（預設 banana），輸出目錄、暫存目錄、
視覺化目錄、CSV log 檔名均自動帶入作物名稱，換作物只需改此參數。

目標結構:
  Massachusetts/png/
    ├── train/
    │   ├── data/   (256×256 RGB GeoTIFF patches)
    │   └── seg/    (256×256 單波段 GeoTIFF，值 0/1)
    ├── val/
    │   ├── data/
    │   └── seg/
    └── test/
        ├── data/
        └── seg/

所有路徑均透過命令列參數指定，無硬編碼路徑。

依賴套件: rasterio, geopandas, numpy, opencv-python, matplotlib, tqdm
         （rasterio/geopandas 不可用時自動 fallback 至 GDAL）
"""

import os
import sys
import shutil
import argparse
import random
import math
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── 可選依賴（GDAL / rasterio / geopandas）──────────────────────────────────
try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False


# ── 常數 ────────────────────────────────────────────────────────────────────
PATCH_SIZE   = 256
KEEP_RATIO   = 0.10
SPLIT_RATIO  = (0.8, 0.1, 0.1)   # train : val : test
RANDOM_SEED  = 42

# 要複製的副檔名（TIF 資料夾）
TIF_EXTS = {".tfw", ".tif", ".aux.xml", ".ari2"}

# ── 輸出路徑預設值 ───────────────────────────────────────────────────────────
DEFAULT_OUT_DIR = "Massachusetts/png"


# ════════════════════════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════════════════════════

def get_args():
    # 先解析 --crop，讓其他預設值可以動態帶入作物名稱
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--crop", default="banana",
                     help="作物名稱，用於命名輸出目錄與檔案（預設: banana）")
    pre_args, _ = pre.parse_known_args()
    crop = pre_args.crop

    parser = argparse.ArgumentParser(description=f"{crop} 資料集前處理腳本")
    # ── 作物名稱 ────────────────────────────────────────────────────────────
    parser.add_argument("--crop", default="banana",
                        help="作物名稱，用於命名輸出目錄與檔案（預設: banana）")
    # ── 來源路徑（必填，無預設值）──────────────────────────────────────────
    parser.add_argument("--src_tif", required=True,
                        help="TIF 來源資料夾")
    parser.add_argument("--src_npy", required=True,
                        help="NPY 來源資料夾")
    parser.add_argument("--src_shp", required=True,
                        help="SHP 來源資料夾")
    # ── 輸出路徑（預設值帶入 crop 名稱）────────────────────────────────────
    parser.add_argument("--out_dir",
                        default=DEFAULT_OUT_DIR,
                        help=f"輸出根目錄（預設: {DEFAULT_OUT_DIR}）")
    parser.add_argument("--tmp_dir",
                        default=None,
                        help="暫存目錄（預設: {{out_dir}}/{crop}_tmp）")
    parser.add_argument("--viz_dir",
                        default=None,
                        help="視覺化輸出目錄（預設: {{out_dir}}/{crop}_viz）")
    # ── 裁切與採樣參數 ───────────────────────────────────────────────────────
    parser.add_argument("--patch_size",  type=int,   default=PATCH_SIZE,
                        help=f"Patch 大小（預設: {PATCH_SIZE}）")
    parser.add_argument("--keep_ratio",  type=float, default=KEEP_RATIO,
                        help=f"下採樣保留比例（預設: {KEEP_RATIO}，即 10%%）")
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED,
                        help=f"隨機種子（預設: {RANDOM_SEED}）")
    # ── 跳過旗標 ────────────────────────────────────────────────────────────
    parser.add_argument("--skip_copy",  action="store_true",
                        help="跳過複製步驟（已複製過時使用）")
    parser.add_argument("--skip_patch", action="store_true",
                        help="跳過裁切步驟（已裁切過時使用）")
    return parser.parse_args()


def print_section(title: str):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def collect_source_files(src_dir: str, exts: set) -> list:
    """遞迴收集符合副檔名的檔案"""
    found = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            # 支援複合副檔名如 .tif.aux.xml
            lower = f.lower()
            for ext in exts:
                if lower.endswith(ext.lower()):
                    found.append(os.path.join(root, f))
                    break
    return found


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 ── 複製來源檔案
# ════════════════════════════════════════════════════════════════════════════

def step1_copy_sources(args) -> dict:
    """
    複製 TIF / NPY / SHP 到暫存目錄，保留原始子目錄結構。
    回傳各類別檔案清單。
    """
    print_section("STEP 1 ── 複製來源檔案")

    tmp = Path(args.tmp_dir)
    tif_dst = tmp / "tif"
    npy_dst = tmp / "npy"
    shp_dst = tmp / "shp"
    ensure_dirs(tif_dst, npy_dst, shp_dst)

    stats = {}

    # ── TIF 資料夾 ──────────────────────────────────────────────────────────
    print(f"\n[TIF] 來源: {args.src_tif}")
    tif_files = collect_source_files(args.src_tif, TIF_EXTS)
    print(f"      找到 {len(tif_files)} 個檔案")
    copied_tif = []
    for fp in tqdm(tif_files, desc="  複製 TIF"):
        dst = tif_dst / Path(fp).name
        if not dst.exists():
            shutil.copy2(fp, dst)
        copied_tif.append(str(dst))
    stats["tif"] = copied_tif
    print(f"  ✓ TIF 複製完成：{len(copied_tif)} 個")

    # ── NPY 資料夾 ──────────────────────────────────────────────────────────
    print(f"\n[NPY] 來源: {args.src_npy}")
    npy_files = collect_source_files(args.src_npy, {".npy"})
    print(f"      找到 {len(npy_files)} 個檔案")
    copied_npy = []
    for fp in tqdm(npy_files, desc="  複製 NPY"):
        dst = npy_dst / Path(fp).name
        if not dst.exists():
            shutil.copy2(fp, dst)
        copied_npy.append(str(dst))
    stats["npy"] = copied_npy
    print(f"  ✓ NPY 複製完成：{len(copied_npy)} 個")

    # ── SHP 資料夾 ──────────────────────────────────────────────────────────
    print(f"\n[SHP] 來源: {args.src_shp}")
    shp_exts = {".cpg", ".dbf", ".prj", ".sbn", ".sbx", ".shp", ".shp.xml", ".shx"}
    shp_files = collect_source_files(args.src_shp, shp_exts)
    print(f"      找到 {len(shp_files)} 個檔案")
    copied_shp = []
    for fp in tqdm(shp_files, desc="  複製 SHP"):
        dst = shp_dst / Path(fp).name
        if not dst.exists():
            shutil.copy2(fp, dst)
        copied_shp.append(str(dst))
    stats["shp"] = copied_shp
    print(f"  ✓ SHP 複製完成：{len(copied_shp)} 個")

    # ── 視覺化：各類別檔案數量 ───────────────────────────────────────────────
    _viz_copy_summary(stats, args.viz_dir)

    return stats


def _viz_copy_summary(stats: dict, viz_dir: str):
    ensure_dirs(viz_dir)
    labels = ["TIF related", "NPY", "SHP related"]
    counts = [len(stats["tif"]), len(stats["npy"]), len(stats["shp"])]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("STEP 1 -- Files Copied by Category", fontsize=13)
    ax.set_ylabel("File Count")
    ax.set_ylim(0, max(counts) * 1.2 + 1)
    plt.tight_layout()
    out = os.path.join(viz_dir, "step1_copy_summary.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 ── SHP → 二值 Mask（與 TIF 對齊）
# ════════════════════════════════════════════════════════════════════════════

def shp_to_mask_rasterio(shp_path: str, tif_path: str, out_tif_path: str) -> np.ndarray:
    """
    使用 rasterio 將 SHP 向量圖層燒錄成與 TIF 同解析度的二值 GeoTIFF mask。
    - 儲存為 GeoTIFF（保留地理座標 transform / CRS）
    - 像素值：香蕉區域 = 1，背景 = 0
    回傳 uint8 numpy array，shape: (H, W)，values: 0 or 1。
    """
    if not HAS_RASTERIO or not HAS_GPD:
        raise ImportError("需要 rasterio 與 geopandas：pip install rasterio geopandas")

    gdf = gpd.read_file(shp_path)

    with rasterio.open(tif_path) as src:
        height    = src.height
        width     = src.width
        transform = src.transform
        crs       = src.crs
        meta      = src.meta.copy()

    # 確保 CRS 一致
    if gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # 燒錄向量到 raster（值為 0/1）
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None)
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )  # values: 0 or 1

    # 儲存為 GeoTIFF（單波段，uint8，保留地理資訊）
    meta.update({"count": 1, "dtype": "uint8", "driver": "GTiff"})
    with rasterio.open(out_tif_path, "w", **meta) as dst:
        dst.write(mask, 1)

    return mask  # shape: (H, W), values: 0 or 1


def step2_shp_to_masks(args, stats: dict) -> dict:
    """
    對每個 TIF 找對應的 SHP，產生全圖 mask 並存成 PNG。
    回傳 {stem: {"tif": path, "mask": path}} 的字典。
    """
    print_section("STEP 2 ── SHP → Segmentation Mask")

    if not HAS_RASTERIO or not HAS_GPD:
        print("  ⚠  未安裝 rasterio / geopandas，嘗試使用 GDAL fallback …")
        return step2_shp_to_masks_gdal(args, stats)

    tmp      = Path(args.tmp_dir)
    mask_dir = tmp / "full_masks"
    ensure_dirs(mask_dir)

    # 找所有主 TIF（排除 .aux.xml / .ari2）
    tif_files = [f for f in stats["tif"] if f.lower().endswith(".tif")
                 and not f.lower().endswith(".aux.xml")]
    shp_files = [f for f in stats["shp"] if f.lower().endswith(".shp")]

    print(f"  主 TIF 數量: {len(tif_files)}")
    print(f"  SHP 數量:    {len(shp_files)}")

    # 建立 stem → shp 對應表（以主檔名前綴比對）
    shp_map = {}
    for sp in shp_files:
        stem = Path(sp).stem
        shp_map[stem] = sp

    pairs = []
    for tp in tif_files:
        stem = Path(tp).stem
        if stem in shp_map:
            pairs.append((tp, shp_map[stem]))
        else:
            # 嘗試模糊比對（TIF stem 包含 SHP stem）
            matched = [s for s in shp_map if stem.startswith(s) or s.startswith(stem)]
            if matched:
                pairs.append((tp, shp_map[matched[0]]))
            else:
                print(f"  ⚠  找不到對應 SHP：{Path(tp).name}")

    print(f"  成功配對: {len(pairs)} 組")

    result = {}
    mask_stats = {"total_px": 0, "pos_px": 0}

    for tif_path, shp_path in tqdm(pairs, desc="  燒錄 mask"):
        stem = Path(tif_path).stem
        # 全圖 mask 存成 GeoTIFF（保留地理座標資訊，值為 0/1）
        mask_path = str(mask_dir / f"{stem}_mask.tif")

        if not os.path.exists(mask_path):
            try:
                mask = shp_to_mask_rasterio(shp_path, tif_path, mask_path)
            except Exception as e:
                print(f"  ✗ 燒錄失敗 {stem}: {e}")
                continue
        else:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # 讀回 0/1 array

        # 統計（mask 值為 0/1）
        mask_stats["total_px"] += mask.size
        mask_stats["pos_px"]   += int((mask > 0).sum())

        result[stem] = {"tif": tif_path, "mask": mask_path}

    neg_px = mask_stats["total_px"] - mask_stats["pos_px"]
    pos_ratio = mask_stats["pos_px"] / max(mask_stats["total_px"], 1) * 100
    print(f"\n  全圖像素統計:")
    print(f"    總像素:   {mask_stats['total_px']:,}")
    print(f"    正樣本:   {mask_stats['pos_px']:,}  ({pos_ratio:.2f}%)")
    print(f"    負樣本:   {neg_px:,}  ({100-pos_ratio:.2f}%)")

    _viz_mask_pixel_dist(mask_stats, args.crop, args.viz_dir)
    return result


def step2_shp_to_masks_gdal(args, stats: dict) -> dict:
    """GDAL fallback（當 rasterio/geopandas 不可用時）"""
    try:
        from osgeo import gdal, ogr, osr
    except ImportError:
        print("  ✗ 也找不到 GDAL，請安裝 rasterio+geopandas 或 gdal")
        sys.exit(1)

    tmp      = Path(args.tmp_dir)
    mask_dir = tmp / "full_masks"
    ensure_dirs(mask_dir)

    tif_files = [f for f in stats["tif"] if f.lower().endswith(".tif")
                 and not f.lower().endswith(".aux.xml")]
    shp_files = [f for f in stats["shp"] if f.lower().endswith(".shp")]
    shp_map   = {Path(s).stem: s for s in shp_files}

    result = {}
    for tp in tqdm(tif_files, desc="  GDAL 燒錄 mask"):
        stem = Path(tp).stem
        if stem not in shp_map:
            continue
        sp = shp_map[stem]
        # 全圖 mask 存成 GeoTIFF（值為 0/1）
        mask_path = str(mask_dir / f"{stem}_mask.tif")

        if not os.path.exists(mask_path):
            ds = gdal.Open(tp)
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            cols, rows = ds.RasterXSize, ds.RasterYSize

            # 建立 GeoTIFF 輸出（值為 0/1）
            drv = gdal.GetDriverByName("GTiff")
            out_ds = drv.Create(mask_path, cols, rows, 1, gdal.GDT_Byte)
            out_ds.SetGeoTransform(gt)
            out_ds.SetProjection(proj)
            out_ds.GetRasterBand(1).Fill(0)

            shp_ds = ogr.Open(sp)
            lyr = shp_ds.GetLayer()
            gdal.RasterizeLayer(out_ds, [1], lyr, burn_values=[1])  # 值為 1，不是 255
            out_ds.FlushCache()
            del ds, out_ds, shp_ds

        result[stem] = {"tif": tp, "mask": mask_path}

    return result


def _viz_mask_pixel_dist(stats: dict, crop: str, viz_dir: str):
    ensure_dirs(viz_dir)
    pos = stats["pos_px"]
    neg = stats["total_px"] - pos
    labels = ["Background (mask=0)", f"{crop} (mask=1)"]
    sizes  = [neg, pos]
    colors = ["#AEC6CF", "#FFB347"]
    explode = (0, 0.08)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
                autopct="%1.2f%%", startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Full-image Pixel Distribution (pos/neg)", fontsize=12)

    axes[1].bar(labels, sizes, color=colors, edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Pixel Count (log scale)", fontsize=12)
    axes[1].set_ylabel("Pixel Count")
    for i, v in enumerate(sizes):
        axes[1].text(i, v * 1.1, f"{v:,}", ha="center", fontsize=9)

    plt.suptitle("STEP 2 -- Full-image Mask Pixel Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(viz_dir, "step2_mask_pixel_dist.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 ── 裁切 256×256 Patches
# ════════════════════════════════════════════════════════════════════════════

def read_tif_as_rgb(tif_path: str) -> np.ndarray:
    """
    讀取多波段 TIF，取前三波段轉成 uint8 RGB。
    若只有單波段則複製成三通道。
    """
    if HAS_RASTERIO:
        with rasterio.open(tif_path) as src:
            n_bands = src.count
            if n_bands >= 3:
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
            else:
                ch = src.read(1).astype(np.float32)
                r = g = b = ch
            # 線性拉伸到 0-255
            def stretch(arr):
                lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
                return arr.astype(np.uint8)
            rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
    else:
        # fallback: cv2 直接讀（適用單波段 GeoTIFF）
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"無法讀取 TIF: {tif_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        # 拉伸
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        img = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


def crop_to_patches(img: np.ndarray, mask: np.ndarray, patch_size: int):
    """
    將影像與 mask 裁切成不重疊的 patch_size × patch_size 方塊。
    回傳 list of (img_patch, mask_patch)。
    """
    H, W = img.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            ip = img[y:y+patch_size, x:x+patch_size]
            mp = mask[y:y+patch_size, x:x+patch_size]
            patches.append((ip, mp))
    return patches


def step3_crop_patches(args, pair_dict: dict) -> dict:
    """
    對每組 (TIF, mask) 裁切成 256×256 patches，存到暫存目錄。
    回傳 {"data": [img_paths], "seg": [mask_paths], "has_pos": [bool]}
    """
    print_section("STEP 3 ── 裁切 256×256 Patches")

    tmp       = Path(args.tmp_dir)
    patch_img = tmp / "patches" / "data"
    patch_seg = tmp / "patches" / "seg"
    ensure_dirs(patch_img, patch_seg)

    psz = args.patch_size
    all_img_paths, all_seg_paths, all_has_pos = [], [], []
    total_patches = 0
    pos_patches   = 0

    for stem, paths in tqdm(pair_dict.items(), desc="  裁切 patches"):
        tif_path  = paths["tif"]
        mask_path = paths["mask"]

        try:
            img  = read_tif_as_rgb(tif_path)
            # 全圖 mask 是 GeoTIFF，值為 0/1
            if HAS_RASTERIO:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.uint8)  # 0 or 1
            else:
                mask = cv2.imread(mask_path, 0)          # GDAL 存的也是 0/1
            if mask is None:
                print(f"  ⚠  無法讀取 mask: {mask_path}")
                continue
            # 確保尺寸一致
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"  ✗ 讀取失敗 {stem}: {e}")
            continue

        patches = crop_to_patches(img, mask, psz)

        for idx, (ip, mp) in enumerate(patches):
            # 全部存成 .tif
            fname    = f"{stem}_{idx}.tif"
            img_out  = str(patch_img / fname)
            mask_out = str(patch_seg / fname)

            if not os.path.exists(img_out):
                # 影像 patch：RGB uint8，存成 3 波段 GeoTIFF
                if HAS_RASTERIO:
                    with rasterio.open(
                        img_out, "w", driver="GTiff",
                        height=psz, width=psz, count=3, dtype=np.uint8
                    ) as dst:
                        dst.write(ip[:, :, 0], 1)
                        dst.write(ip[:, :, 1], 2)
                        dst.write(ip[:, :, 2], 3)
                else:
                    cv2.imwrite(img_out, cv2.cvtColor(ip, cv2.COLOR_RGB2BGR))

            if not os.path.exists(mask_out):
                # mask patch：單波段 uint8，值 0/1，存成 GeoTIFF
                if HAS_RASTERIO:
                    with rasterio.open(
                        mask_out, "w", driver="GTiff",
                        height=psz, width=psz, count=1, dtype=np.uint8
                    ) as dst:
                        dst.write(mp.astype(np.uint8), 1)
                else:
                    cv2.imwrite(mask_out, mp.astype(np.uint8))

            has_pos = bool((mp > 0).any())
            all_img_paths.append(img_out)
            all_seg_paths.append(mask_out)
            all_has_pos.append(has_pos)

        total_patches += len(patches)
        pos_patches   += sum(1 for _, mp in patches if (mp > 0).any())

    neg_patches = total_patches - pos_patches
    print(f"\n  裁切結果:")
    print(f"    總 patches:   {total_patches:,}")
    print(f"    正樣本 (mask>0): {pos_patches:,}  ({pos_patches/max(total_patches,1)*100:.2f}%)")
    print(f"    負樣本 (mask=0): {neg_patches:,}  ({neg_patches/max(total_patches,1)*100:.2f}%)")

    result = {
        "data":    all_img_paths,
        "seg":     all_seg_paths,
        "has_pos": all_has_pos,
    }
    _viz_patch_dist(total_patches, pos_patches, neg_patches, args.viz_dir)
    return result


def _viz_patch_dist(total, pos, neg, viz_dir):
    ensure_dirs(viz_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = [f"Negative\n{neg:,}", f"Positive\n{pos:,}"]
    sizes  = [neg, pos]
    colors = ["#AEC6CF", "#FFB347"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.2f%%",
                startangle=90, explode=(0, 0.08))
    axes[0].set_title("Patch Distribution (pos/neg)", fontsize=12)

    axes[1].bar(["Negative", "Positive"], [neg, pos], color=colors, edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Patch Count (total {total:,})", fontsize=12)
    axes[1].set_ylabel("Count (log scale)")
    for i, v in enumerate([neg, pos]):
        axes[1].text(i, v * 1.15, f"{v:,}", ha="center", fontsize=9)

    plt.suptitle("STEP 3 -- Patch Statistics after Cropping", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(viz_dir, "step3_patch_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 ── 下採樣（保留 10%，正:負 ≈ 1:2）
# ════════════════════════════════════════════════════════════════════════════

def step4_subsample(args, patch_dict: dict) -> dict:
    """
    下採樣策略：
      - 目標總量 = 原始總量 × keep_ratio（預設 10%）
      - 保留全部正樣本（mask 含有 1 的 patch）
      - 剩餘名額全部用負樣本（全黑 mask）補滿
      - 若正樣本數已超過目標總量，則只保留正樣本（不截斷）
    """
    print_section("STEP 4 ── 下採樣（解決類別不平衡）")

    rng = random.Random(args.seed)

    data_paths = patch_dict["data"]
    seg_paths  = patch_dict["seg"]
    has_pos    = patch_dict["has_pos"]

    pos_idx = [i for i, h in enumerate(has_pos) if h]
    neg_idx = [i for i, h in enumerate(has_pos) if not h]

    total_raw    = len(data_paths)
    n_pos        = len(pos_idx)
    n_neg_raw    = len(neg_idx)
    target_total = int(total_raw * args.keep_ratio)

    # 負樣本補到剛好湊滿 target_total
    n_neg_target = max(target_total - n_pos, 0)
    n_neg_target = min(n_neg_target, n_neg_raw)   # 不超過現有負樣本數

    # 隨機抽取負樣本
    rng.shuffle(neg_idx)
    kept_neg_idx = neg_idx[:n_neg_target]
    kept_idx     = sorted(pos_idx + kept_neg_idx)
    final_total  = len(kept_idx)

    print(f"\n  原始統計:")
    print(f"    總 patches:      {total_raw:,}")
    print(f"    正樣本 (mask>0): {n_pos:,}  ({n_pos/total_raw*100:.2f}%)")
    print(f"    負樣本 (mask=0): {n_neg_raw:,}  ({n_neg_raw/total_raw*100:.2f}%)")
    print(f"\n  下採樣後:")
    print(f"    目標總量 (10%):  {target_total:,}")
    print(f"    正樣本保留:      {n_pos:,}  (全部保留)")
    print(f"    負樣本保留:      {n_neg_target:,}  (補滿至目標總量)")
    print(f"    最終保留:        {final_total:,}  ({final_total/total_raw*100:.2f}%)")

    kept_data = [data_paths[i] for i in kept_idx]
    kept_seg  = [seg_paths[i]  for i in kept_idx]
    kept_pos  = [has_pos[i]    for i in kept_idx]

    _viz_subsample(total_raw, n_pos, n_neg_raw, n_neg_target, target_total, args.viz_dir)

    return {"data": kept_data, "seg": kept_seg, "has_pos": kept_pos}


def _viz_subsample(total, n_pos, n_neg_raw, n_neg_kept, target_total, viz_dir):
    ensure_dirs(viz_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    categories = ["Positive", "Negative (raw)", "Negative (kept)"]
    values     = [n_pos, n_neg_raw, n_neg_kept]
    colors     = ["#FFB347", "#AEC6CF", "#77B5FE"]
    bars = axes[0].bar(categories, values, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                     f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_title("Before vs After Subsampling (log scale)", fontsize=12)
    axes[0].set_ylabel("Count")

    final_total = n_pos + n_neg_kept
    sizes  = [n_pos, n_neg_kept]
    labels = [f"Positive\n{n_pos:,}", f"Negative\n{n_neg_kept:,}"]
    axes[1].pie(sizes, labels=labels, colors=["#FFB347", "#77B5FE"],
                autopct="%1.1f%%", startangle=90, explode=(0.05, 0))
    axes[1].set_title(
        f"Final Dataset Composition ({final_total:,} patches)\nTarget 10% = {target_total:,}",
        fontsize=12)

    plt.suptitle("STEP 4 -- Subsampling (keep all positives, fill negatives to 10%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(viz_dir, "step4_subsample.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 ── 8:1:1 切割並複製到目標結構
# ════════════════════════════════════════════════════════════════════════════

def step5_split_and_copy(args, kept_dict: dict):
    """
    將保留的 patches 依 8:1:1 切割，複製到 Massachusetts/png 目標結構。
    同時輸出 CSV log：{viz_dir}/dataset_log.csv
    """
    print_section("STEP 5 ── 8:1:1 切割 & 複製到目標結構")

    import csv

    rng = random.Random(args.seed)

    data_paths = kept_dict["data"]
    seg_paths  = kept_dict["seg"]
    has_pos    = kept_dict["has_pos"]
    n = len(data_paths)

    # shuffle
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * SPLIT_RATIO[0])
    n_val   = int(n * SPLIT_RATIO[1])

    splits = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train+n_val],
        "test":  indices[n_train+n_val:],
    }

    out_root  = Path(args.out_dir)
    log_path  = os.path.join(args.viz_dir, f"{args.crop}_dataset_log.csv")
    ensure_dirs(args.viz_dir)
    split_counts = {}

    with open(log_path, "w", newline="", encoding="utf-8") as log_f:
        writer = csv.writer(log_f)
        writer.writerow(["fname", "split", "has_pos", "src_data", "src_seg"])

        for split_name, idxs in splits.items():
            data_dst = out_root / split_name / "data"
            seg_dst  = out_root / split_name / "seg"
            ensure_dirs(data_dst, seg_dst)

            pos_cnt = 0
            for i in tqdm(idxs, desc=f"  複製 {split_name:5s}", unit="patch"):
                src_img  = data_paths[i]
                src_seg  = seg_paths[i]
                hp       = has_pos[i]
                fname    = Path(src_img).name
                dst_img  = data_dst / fname
                dst_seg  = seg_dst  / fname

                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                if not dst_seg.exists():
                    shutil.copy2(src_seg, dst_seg)

                writer.writerow([fname, split_name, int(hp), src_img, src_seg])

            # 統計正樣本（複製完後統一掃描，避免每張都開關 tif）
            seg_files = list(seg_dst.glob("*.tif"))
            for seg_f in tqdm(seg_files, desc=f"  統計 {split_name:5s} mask",
                              unit="patch", leave=False):
                try:
                    if HAS_RASTERIO:
                        with rasterio.open(str(seg_f)) as src:
                            m = src.read(1)
                    else:
                        m = cv2.imread(str(seg_f), 0)
                    if m is not None and (m > 0).any():
                        pos_cnt += 1
                except Exception:
                    pass

            split_counts[split_name] = {"total": len(idxs), "pos": pos_cnt,
                                        "neg": len(idxs) - pos_cnt}
            print(f"  {split_name:5s}: {len(idxs):6,} 張  "
                  f"(正: {pos_cnt:,}  負: {len(idxs)-pos_cnt:,})")

    print(f"  [csv log] saved -> {log_path}")
    _viz_split_summary(split_counts, args.viz_dir)
    return split_counts


def _viz_split_summary(split_counts: dict, viz_dir: str):
    ensure_dirs(viz_dir)

    splits = list(split_counts.keys())
    totals = [split_counts[s]["total"] for s in splits]
    pos    = [split_counts[s]["pos"]   for s in splits]
    neg    = [split_counts[s]["neg"]   for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 堆疊長條
    axes[0].bar(x, neg, width, label="Negative", color="#AEC6CF")
    axes[0].bar(x, pos, width, bottom=neg, label="Positive", color="#FFB347")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.upper() for s in splits])
    axes[0].set_title("Positive/Negative Composition per Split", fontsize=12)
    axes[0].set_ylabel("Count")
    axes[0].legend()
    for xi, (t, p, ng) in enumerate(zip(totals, pos, neg)):
        axes[0].text(xi, t + t*0.01, f"{t:,}", ha="center", fontsize=9)

    axes[1].pie(totals,
                labels=[f"{s.upper()}\n{t:,}" for s, t in zip(splits, totals)],
                colors=["#4C72B0", "#DD8452", "#55A868"],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Train / Val / Test Ratio", fontsize=12)

    plt.suptitle("STEP 5 -- Dataset Split Result", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(viz_dir, "step5_split_summary.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 ── 視覺化樣本預覽
# ════════════════════════════════════════════════════════════════════════════

def step6_preview_samples(args, n_samples: int = 8):
    """
    從 train/data 與 train/seg 隨機抽取樣本，並排顯示影像與 mask。
    """
    print_section("STEP 6 ── 樣本預覽")

    out_root = Path(args.out_dir)
    img_dir  = out_root / "train" / "data"
    seg_dir  = out_root / "train" / "seg"

    if not img_dir.exists():
        print("  ⚠  train/data 不存在，跳過預覽")
        return

    fnames = sorted(os.listdir(img_dir))
    if len(fnames) == 0:
        print("  ⚠  train/data 為空，跳過預覽")
        return

    rng = random.Random(args.seed)

    # 分成正/負各半（讀 tif seg）
    pos_fnames, neg_fnames = [], []
    for fn in tqdm(fnames, desc="  掃描 train seg", unit="patch", leave=False):
        seg_path = seg_dir / fn
        try:
            if HAS_RASTERIO:
                with rasterio.open(str(seg_path)) as src:
                    m = src.read(1)
            else:
                m = cv2.imread(str(seg_path), 0)
            if m is not None and (m > 0).any():
                pos_fnames.append(fn)
            else:
                neg_fnames.append(fn)
        except Exception:
            neg_fnames.append(fn)

    n_each = n_samples // 2
    preview = (rng.sample(pos_fnames, min(n_each, len(pos_fnames))) +
               rng.sample(neg_fnames, min(n_each, len(neg_fnames))))

    cols = 4
    rows = math.ceil(len(preview) / cols) * 2  # 每個樣本佔 2 行（img + seg）
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    ax_idx = 0
    for fn in preview:
        img_path = img_dir / fn
        seg_path = seg_dir / fn
        # 讀影像（3 波段 tif）
        try:
            if HAS_RASTERIO:
                with rasterio.open(str(img_path)) as src:
                    img = src.read([1, 2, 3]).transpose(1, 2, 0)
            else:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
        # 讀 mask（單波段 tif，值 0/1）
        try:
            if HAS_RASTERIO:
                with rasterio.open(str(seg_path)) as src:
                    seg = src.read(1)
            else:
                seg = cv2.imread(str(seg_path), 0)
        except Exception:
            continue
        if img is None or seg is None:
            continue
        has_pos = (seg > 0).any()
        label = "Positive" if has_pos else "Negative"

        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(f"{fn[:20]}\n{label}", fontsize=7)
        axes[ax_idx].axis("off")
        ax_idx += 1

        axes[ax_idx].imshow(seg, cmap="gray", vmin=0, vmax=1)
        axes[ax_idx].set_title("mask", fontsize=7)
        axes[ax_idx].axis("off")
        ax_idx += 1

    for i in range(ax_idx, len(axes)):
        axes[i].axis("off")

    plt.suptitle("STEP 6 -- Sample Preview (train set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(args.viz_dir, "step6_sample_preview.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 ── 最終統計報告
# ════════════════════════════════════════════════════════════════════════════

def step7_final_report(args, split_counts: dict):
    print_section("STEP 7 ── 最終統計報告")

    total = sum(v["total"] for v in split_counts.values())
    total_pos = sum(v["pos"] for v in split_counts.values())
    total_neg = sum(v["neg"] for v in split_counts.values())

    print(f"\n  {'Split':<8} {'Total':>8} {'Positive':>10} {'Negative':>10} {'Pos%':>7}")
    print(f"  {'-'*45}")
    for split, cnt in split_counts.items():
        pct = cnt["pos"] / max(cnt["total"], 1) * 100
        print(f"  {split:<8} {cnt['total']:>8,} {cnt['pos']:>10,} {cnt['neg']:>10,} {pct:>6.2f}%")
    print(f"  {'-'*45}")
    pct_all = total_pos / max(total, 1) * 100
    print(f"  {'TOTAL':<8} {total:>8,} {total_pos:>10,} {total_neg:>10,} {pct_all:>6.2f}%")

    print(f"\n  輸出目錄: {args.out_dir}")
    print(f"  視覺化:   {args.viz_dir}")
    print(f"\n  ✓ 資料集準備完成，可直接接上 data_preparation/ 腳本使用。")
    print(f"\n  後續步驟建議:")
    print(f"    1. python data_preparation/png_remove_no_data.py \\")
    print(f"         --data_dir {args.out_dir} --check_ref_dir train/data")
    print(f"    2. python data_preparation/png_count_and_index_buildings.py")
    print(f"       (修改 data_dir 為 {args.out_dir})")
    print(f"    3. python data_preparation/png_insert_label_noises.py \\")
    print(f"         --data_dir {args.out_dir} --partition train \\")
    print(f"         --save_dir_name ns_seg_1")

    # 最終視覺化：完整 pipeline 流程圖
    _viz_pipeline_summary(split_counts, args.crop, args.viz_dir)


def _viz_pipeline_summary(split_counts: dict, crop: str, viz_dir: str):
    ensure_dirs(viz_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    steps = [
        "STEP 1\n複製來源檔案\n(TIF/NPY/SHP)",
        "STEP 2\nSHP → Mask\n(rasterize)",
        "STEP 3\n裁切 256×256\nPatches",
        "STEP 4\n下採樣\n(10%, 1:2)",
        "STEP 5\n8:1:1 切割\n& 複製",
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    n = len(steps)
    for i, (step, color) in enumerate(zip(steps, colors)):
        x = i / (n - 1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.08, 0.2), 0.16, 0.6,
            boxstyle="round,pad=0.02", facecolor=color, alpha=0.85,
            edgecolor="white", linewidth=2, transform=ax.transAxes))
        ax.text(x, 0.5, step, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
                transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=((i+1)/(n-1) - 0.09, 0.5),
                        xytext=(x + 0.09, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    total = sum(v["total"] for v in split_counts.values())
    ax.set_title(f"{crop} 資料集前處理 Pipeline  ─  最終 {total:,} 張 patches",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    out = os.path.join(viz_dir, "step7_pipeline_summary.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [viz] saved -> {out}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # tmp_dir / viz_dir 若未指定，預設放在 out_dir 底下
    if args.tmp_dir is None:
        args.tmp_dir = os.path.join(args.out_dir, f"{args.crop}_tmp")
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.out_dir, f"{args.crop}_viz")

    ensure_dirs(args.tmp_dir, args.viz_dir, args.out_dir)

    print("\n" + "█" * 60)
    print(f"  {args.crop} 資料集前處理腳本")
    print("█" * 60)
    print(f"  作物名稱:  {args.crop}")
    print(f"  TIF 來源:  {args.src_tif}")
    print(f"  NPY 來源:  {args.src_npy}")
    print(f"  SHP 來源:  {args.src_shp}")
    print(f"  輸出目錄:  {args.out_dir}")
    print(f"  Patch 大小: {args.patch_size}×{args.patch_size}")
    print(f"  保留比例:  {args.keep_ratio*100:.0f}%")
    print(f"  切割比例:  train={SPLIT_RATIO[0]:.0%} val={SPLIT_RATIO[1]:.0%} test={SPLIT_RATIO[2]:.0%}")
    print(f"  隨機種子:  {args.seed}")

    # ── STEP 1: 複製來源 ────────────────────────────────────────────────────
    if not args.skip_copy:
        stats = step1_copy_sources(args)
    else:
        print_section("STEP 1 ── 跳過（--skip_copy）")
        tmp = Path(args.tmp_dir)
        stats = {
            "tif": [str(p) for p in (tmp / "tif").glob("*") if p.is_file()],
            "npy": [str(p) for p in (tmp / "npy").glob("*") if p.is_file()],
            "shp": [str(p) for p in (tmp / "shp").glob("*") if p.is_file()],
        }
        print(f"  TIF: {len(stats['tif'])}  NPY: {len(stats['npy'])}  SHP: {len(stats['shp'])}")

    # ── STEP 2: SHP → Mask ──────────────────────────────────────────────────
    pair_dict = step2_shp_to_masks(args, stats)

    if len(pair_dict) == 0:
        print("\n  ✗ 沒有成功配對的 TIF/SHP，請確認來源路徑。")
        sys.exit(1)

    # ── STEP 3: 裁切 Patches ────────────────────────────────────────────────
    if not args.skip_patch:
        patch_dict = step3_crop_patches(args, pair_dict)
    else:
        print_section("STEP 3 ── 跳過（--skip_patch）")
        tmp = Path(args.tmp_dir)
        img_paths = sorted(str(p) for p in (tmp / "patches" / "data").glob("*.tif"))
        seg_paths = sorted(str(p) for p in (tmp / "patches" / "seg").glob("*.tif"))
        has_pos = []
        for sp in tqdm(seg_paths, desc="  重新統計 mask"):
            try:
                if HAS_RASTERIO:
                    with rasterio.open(sp) as src:
                        m = src.read(1)
                else:
                    m = cv2.imread(sp, 0)
                has_pos.append(bool(m is not None and (m > 0).any()))
            except Exception:
                has_pos.append(False)
        patch_dict = {"data": img_paths, "seg": seg_paths, "has_pos": has_pos}
        n_pos = sum(has_pos)
        print(f"  載入 {len(img_paths):,} 個 patches，正樣本 {n_pos:,}")

    # ── STEP 4: 下採樣 ──────────────────────────────────────────────────────
    kept_dict = step4_subsample(args, patch_dict)

    # ── STEP 5: 切割 & 複製 ─────────────────────────────────────────────────
    split_counts = step5_split_and_copy(args, kept_dict)

    # ── STEP 6: 樣本預覽 ────────────────────────────────────────────────────
    step6_preview_samples(args)

    # ── STEP 7: 最終報告 ────────────────────────────────────────────────────
    step7_final_report(args, split_counts)


if __name__ == "__main__":
    main()

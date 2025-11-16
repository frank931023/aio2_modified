import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

tif_path = r"D:\DMCIII_1220_banana_tif\94171006_240406a_16~4819_hr4.tif"
shp_path = r"E:\crop\bananas\SHP\94171006_240406a_16~4819_hr4.shp"

# === 讀 TIFF ===
with rasterio.open(tif_path) as src:
    count = src.count              # 總 band 數
    transform = src.transform
    crs = src.crs

    # 讀取前3個 band（假設 1,2,3 是 RGB）
    if count >= 3:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        rgb = np.dstack([r, g, b])
    else:
        # 單 band（灰階）
        rgb = src.read(1)

# 正規化 0~1
rgb = rgb.astype(float)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

print("Bands:", count)
print("RGB shape:", rgb.shape)
print("CRS:", crs)
print("Transform:", transform)

# === 讀 SHP ===
gdf = gpd.read_file(shp_path)
print("SHP CRS:", gdf.crs)

# === 計算 geospatial extent ===
height, width = rgb.shape[:2]

left   = transform.c
right  = transform.c + transform.a * width
top    = transform.f
bottom = transform.f + transform.e * height

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(rgb, extent=[left, right, bottom, top])
gdf.boundary.plot(ax=ax, color='red', linewidth=1.5)

ax.set_title("Satellite RGB + SHP Boundary")
plt.savefig("images/test_3.png", dpi=200)

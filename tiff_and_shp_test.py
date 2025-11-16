import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

tif_path = r"D:\DMCIII_1220_banana_tif\94171006_240406a_16~4819_hr4.tif"
shp_path = r"E:\crop\bananas\SHP\94171006_240406a_16~4819_hr4.shp"

# TIFF
with rasterio.open(tif_path) as src:
    img = src.read(1)
    transform = src.transform
    crs = src.crs

print("TIFF shape:", img.shape)
print("Transform:", transform)
print("CRS:", crs)

# SHP
gdf = gpd.read_file(shp_path)

print(gdf.head())
print("CRS:", gdf.crs)

# plot
plt.imshow(img, cmap='gray')
gdf.boundary.plot(ax=plt.gca(), color='red', linewidth=1)
plt.show()

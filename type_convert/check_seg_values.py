
import os
import glob
from PIL import Image
import numpy as np

def check_pixel_values(directory):
    files = glob.glob(os.path.join(directory, "*.tif"))
    print(f"Found {len(files)} tif files in {directory}")
    
    # Check first 5 files
    for f in files[:5]:
        try:
            img = Image.open(f)
            data = np.array(img)
            unique_values = np.unique(data)
            print(f"File: {os.path.basename(f)}")
            print(f"  Shape: {data.shape}")
            print(f"  Unique Values: {unique_values}")
            print(f"  Dtype: {data.dtype}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    target_dir = r"c:\Users\frank\Desktop\code\AIO2\Massachusetts\tiff_256\train\seg"
    check_pixel_values(target_dir)

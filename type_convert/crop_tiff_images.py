import os
from PIL import Image
from tqdm import tqdm
import pathlib

def get_crop_coordinates(image_size, patch_size):
    """
    計算裁切座標，處理無法整除的情況 (使用重疊法)
    image_size: 原始圖片大小 (e.g., 1500)
    patch_size: 裁切大小 (e.g., 256)
    """
    coords = []
    
    # 一般的整除切法
    for i in range(0, image_size, patch_size):
        # 如果這一塊切下去會超出邊界
        if i + patch_size > image_size:
            # 就強行設為「最後一段」：從 (總長 - patch_size) 開始切
            coords.append(image_size - patch_size)
        else:
            coords.append(i)
            
    # 去除重複並排序
    return sorted(list(set(coords)))

def crop_images(input_dir, output_dir, patch_size=256):
    """
    遍歷 input_dir 下的所有 tiff 檔案，裁切後存到 output_dir
    保持原本的目錄結構
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

    # 搜尋所有 tiff 檔案 (包含 .tif)
    all_images = list(input_path.rglob("*.tiff")) + list(input_path.rglob("*.tif"))
    
    # 過濾: 只保留路徑中包含 'data' 或 'seg' 的檔案
    image_files = [
        p for p in all_images 
        if 'data' in p.parts or 'seg' in p.parts
    ]
    
    print(f"在 'data' 和 'seg' 資料夾中找到 {len(image_files)} 張圖片，準備開始裁切...")

    for img_path in tqdm(image_files, desc="Cropping"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                
                # 取得 x 和 y 的裁切座標
                x_coords = get_crop_coordinates(w, patch_size)
                y_coords = get_crop_coordinates(h, patch_size)
                
                # 建構輸出路徑 (保持相對目錄結構)
                rel_path = img_path.relative_to(input_path)
                save_dir = output_path / rel_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                base_name = img_path.stem
                ext = img_path.suffix
                
                for y in y_coords:
                    for x in x_coords:
                        box = (x, y, x + patch_size, y + patch_size)
                        crop_img = img.crop(box)
                        
                        # 檔名加入座標資訊，避免重複
                        new_filename = f"{base_name}_{x}_{y}{ext}"
                        save_path = save_dir / new_filename
                        
                        crop_img.save(save_path)
                        
        except Exception as e:
            print(f"處理 {img_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    # 設定輸入與輸出目錄
    # 根據您的描述，目標是在 /tiff 底下，這裡假設是專案根目錄下的 Massachusetts/tiff
    # 您可以根據實際路徑修改這裡
    BASE_DIR = pathlib.Path(__file__).parent.parent 
    INPUT_DIR = BASE_DIR / "Massachusetts" / "tiff"
    OUTPUT_DIR = BASE_DIR / "Massachusetts" / "tiff_256"
    
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not INPUT_DIR.exists():
        print(f"錯誤: 找不到輸入目錄 {INPUT_DIR}")
    else:
        crop_images(INPUT_DIR, OUTPUT_DIR, patch_size=256)
        print("裁切完成！")

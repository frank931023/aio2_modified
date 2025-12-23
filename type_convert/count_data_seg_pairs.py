import pathlib
import os

def count_pairs(base_dir):
    """
    Counts matching pairs of images between 'data' and 'seg' subdirectories
    for train, test, and val splits.
    """
    base_path = pathlib.Path(base_dir)
    splits = ['train', 'test', 'val']
    
    print(f"Checking pairs in {base_path}...\n")
    
    total_pairs = 0
    
    for split in splits:
        split_dir = base_path / split
        data_dir = split_dir / 'data'
        seg_dir = split_dir / 'seg'
        
        if not split_dir.exists():
            print(f"[{split}] Directory not found: {split_dir}")
            continue
            
        # Get sets of filenames without extensions (stems)
        # Assuming files are images (tiff/tif)
        data_files = set()
        if data_dir.exists():
            data_files = {p.stem for p in data_dir.glob('*') if p.suffix.lower() in ['.tif', '.tiff']}
            
        seg_files = set()
        if seg_dir.exists():
            seg_files = {p.stem for p in seg_dir.glob('*') if p.suffix.lower() in ['.tif', '.tiff']}
            
        # Find intersection
        pairs = data_files.intersection(seg_files)
        count = len(pairs)
        total_pairs += count
        
        print(f"[{split.upper()}]")
        print(f"  Data files: {len(data_files)}")
        print(f"  Seg files : {len(seg_files)}")
        print(f"  Pairs     : {count}")
        
        # Optional: Show warnings for mismatches
        only_in_data = len(data_files - seg_files)
        only_in_seg = len(seg_files - data_files)
        if only_in_data > 0 or only_in_seg > 0:
            print(f"  (Unmatched: {only_in_data} in data only, {only_in_seg} in seg only)")
        print("-" * 30)

    print(f"\nTotal Pairs across all splits: {total_pairs}")

if __name__ == "__main__":
    # Base directory is assumed to be ../Massachusetts/tiff relative to this script
    # Adjust this path if the script is moved or the data is elsewhere
    current_dir = pathlib.Path(__file__).parent
    # Looking for Massachusetts/tiff 
    # Structure:
    # root/
    #   type_convert/ (this script)
    #   Massachusetts/
    #     tiff/
    target_base = current_dir.parent / "Massachusetts" / "tiff_256"
    
    if target_base.exists():
        count_pairs(target_base)
    else:
        print(f"Error: Could not find target directory at {target_base}")

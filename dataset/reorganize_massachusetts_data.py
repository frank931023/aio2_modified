"""
Reorganize Massachusetts dataset structure for data_loading_Building.py

Original structure:
    Massachusetts/png/
        ├── train/
        ├── train_labels/
        ├── test/
        ├── test_labels/
        ├── val/
        └── val_labels/

Target structure:
    Massachusetts/png/
        ├── train/
        │   ├── data/      (images)
        │   └── seg/       (labels)
        ├── test/
        │   ├── data/
        │   └── seg/
        └── val/
            ├── data/
            └── seg/

Usage:
    python reorganize_massachusetts_data.py --source_dir <path_to_Massachusetts/png> [--mode copy]
"""

import os
import shutil
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='Reorganize Massachusetts dataset structure')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to Massachusetts/png directory')
    parser.add_argument('--mode', type=str, choices=['copy', 'move'], default='copy',
                        help='Copy or move files (default: copy, safer option)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview operations without actually moving/copying files')
    return parser.parse_args()


def reorganize_split(source_dir, split_name, mode='copy', dry_run=False):
    """
    Reorganize one split (train/test/val)
    
    Args:
        source_dir: Path to Massachusetts/png directory
        split_name: 'train', 'test', or 'val'
        mode: 'copy' or 'move'
        dry_run: If True, only print operations without executing
    """
    source_dir = Path(source_dir)
    
    # Source directories
    img_src = source_dir / split_name
    label_src = source_dir / f"{split_name}_labels"
    
    # Check if source directories exist
    if not img_src.exists():
        print(f"❌ Warning: {img_src} does not exist, skipping {split_name}")
        return False
    if not label_src.exists():
        print(f"❌ Warning: {label_src} does not exist, skipping {split_name}")
        return False
    
    # Target directories
    img_dst = source_dir / split_name / "data"
    label_dst = source_dir / split_name / "seg"
    
    # Get list of files
    img_files = sorted([f for f in os.listdir(img_src) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_src) if f.endswith('.png')])
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split:")
    print(f"  Images found: {len(img_files)}")
    print(f"  Labels found: {len(label_files)}")
    
    # Verify matching files
    if set(img_files) != set(label_files):
        print(f"⚠️  Warning: Image and label files don't match perfectly!")
        missing_in_labels = set(img_files) - set(label_files)
        missing_in_images = set(label_files) - set(img_files)
        if missing_in_labels:
            print(f"  Missing labels for: {missing_in_labels}")
        if missing_in_images:
            print(f"  Missing images for: {missing_in_images}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would create:")
        print(f"  {img_dst}")
        print(f"  {label_dst}")
        print(f"[DRY RUN] Would {mode} {len(img_files)} image files")
        print(f"[DRY RUN] Would {mode} {len(label_files)} label files")
        return True
    
    # Create target directories
    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories:")
    print(f"  {img_dst}")
    print(f"  {label_dst}")
    
    # Move/Copy image files
    print(f"\n{'Copy' if mode == 'copy' else 'Move'}ing images...")
    for i, fname in enumerate(img_files, 1):
        src_path = img_src / fname
        dst_path = img_dst / fname
        
        if mode == 'copy':
            shutil.copy2(src_path, dst_path)
        else:  # move
            shutil.move(str(src_path), str(dst_path))
        
        if i % 50 == 0 or i == len(img_files):
            print(f"  Progress: {i}/{len(img_files)} images", end='\r')
    print(f"\n✓ {mode.capitalize()}d {len(img_files)} images")
    
    # Move/Copy label files
    print(f"\n{'Copy' if mode == 'copy' else 'Move'}ing labels...")
    for i, fname in enumerate(label_files, 1):
        src_path = label_src / fname
        dst_path = label_dst / fname
        
        if mode == 'copy':
            shutil.copy2(src_path, dst_path)
        else:  # move
            shutil.move(str(src_path), str(dst_path))
        
        if i % 50 == 0 or i == len(label_files):
            print(f"  Progress: {i}/{len(label_files)} labels", end='\r')
    print(f"\n✓ {mode.capitalize()}d {len(label_files)} labels")
    
    # Remove old directories if moving
    if mode == 'move':
        # Only remove if empty or only contains data/seg subdirs
        try:
            remaining_img = [f for f in os.listdir(img_src) if f not in ['data', 'seg']]
            remaining_label = [f for f in os.listdir(label_src) if f not in ['data', 'seg']]
            
            if not remaining_img and img_src != img_dst.parent:
                # Don't remove if it's the parent directory
                pass
            if not remaining_label:
                shutil.rmtree(label_src)
                print(f"✓ Removed old directory: {label_src}")
        except Exception as e:
            print(f"⚠️  Could not remove old directories: {e}")
    
    return True


def main():
    args = get_args()
    
    source_dir = Path(args.source_dir)
    
    # Validate source directory
    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        return
    
    print("="*60)
    print("Massachusetts Dataset Reorganization Tool")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Dry run: {'YES (no changes will be made)' if args.dry_run else 'NO'}")
    
    if not args.dry_run:
        response = input("\nProceed with reorganization? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Process each split
    splits = ['train', 'val', 'test']
    success_count = 0
    
    for split in splits:
        if reorganize_split(source_dir, split, args.mode, args.dry_run):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Reorganization complete!")
    print(f"Successfully processed {success_count}/{len(splits)} splits")
    print("="*60)
    
    if not args.dry_run:
        print("\nNew structure:")
        print(f"{source_dir}/")
        for split in splits:
            split_dir = source_dir / split
            if split_dir.exists():
                print(f"├── {split}/")
                print(f"│   ├── data/      ({len(list((split_dir / 'data').glob('*.png')))} files)")
                print(f"│   └── seg/       ({len(list((split_dir / 'seg').glob('*.png')))} files)")
        
        print("\n✓ Ready to use with data_loading_Building.py!")
        print(f"\nUsage example:")
        print(f"  BuildingDataset(data_path='{source_dir}', split='train')")


if __name__ == '__main__':
    main()

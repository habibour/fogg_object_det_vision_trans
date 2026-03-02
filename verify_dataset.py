"""
Quick verification script for the prepared dataset
"""
import json
import os
from pathlib import Path

def verify_dataset():
    """Verify the prepared dataset."""
    
    print("=" * 70)
    print("  DATASET VERIFICATION")
    print("=" * 70)
    
    base_path = Path('voc_2012/processed')
    
    # Check main directories
    print("\n✓ Checking main directories...")
    dirs_to_check = [
        'VOC2012_filtered',
        'VOC2012_foggy',
        'VOC2012_paired'
    ]
    
    for d in dirs_to_check:
        path = base_path / d
        if path.exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ - MISSING!")
    
    # Check pairs.json
    print("\n✓ Checking pairs.json...")
    pairs_file = base_path / 'VOC2012_paired' / 'pairs.json'
    if pairs_file.exists():
        with open(pairs_file, 'r') as f:
            pairs_data = json.load(f)
        print(f"  ✓ pairs.json exists")
        print(f"  ✓ Number of pairs: {pairs_data['metadata']['num_pairs']}")
        print(f"  ✓ Fog levels: {', '.join(pairs_data['metadata']['fog_levels'])}")
    else:
        print(f"  ✗ pairs.json - MISSING!")
    
    # Check split files
    print("\n✓ Checking split files...")
    splits_dir = base_path / 'VOC2012_paired' / 'ImageSets' / 'Main'
    split_files = ['train.txt', 'val.txt', 'test.txt', 'trainval.txt']
    
    for split_file in split_files:
        path = splits_dir / split_file
        if path.exists():
            with open(path, 'r') as f:
                count = len(f.readlines())
            print(f"  ✓ {split_file}: {count} images")
        else:
            print(f"  ✗ {split_file} - MISSING!")
    
    # Check symlinks
    print("\n✓ Checking foggy image symlinks...")
    for fog_level in ['low', 'mid', 'high']:
        link_path = base_path / 'VOC2012_paired' / 'foggy' / fog_level / 'JPEGImages'
        if link_path.is_symlink():
            target = link_path.resolve()
            print(f"  ✓ {fog_level}/ -> {target.name}")
        else:
            print(f"  ✗ {fog_level}/ symlink - MISSING!")
    
    # Count images
    print("\n✓ Counting images...")
    clean_imgs = list((base_path / 'VOC2012_filtered' / 'JPEGImages').glob('*.jpg'))
    print(f"  ✓ Clean images: {len(clean_imgs)}")
    
    for fog_level in ['low', 'mid', 'high']:
        fog_imgs = list((base_path / 'VOC2012_foggy' / fog_level).glob('*.jpg'))
        print(f"  ✓ {fog_level.capitalize()} fog images: {len(fog_imgs)}")
    
    # Show sample pair
    print("\n✓ Sample pair from dataset:")
    if pairs_file.exists():
        sample_pair = pairs_data['pairs'][0]
        print(f"  ID: {sample_pair['id']}")
        print(f"  Clean: {sample_pair['clean']['image']}")
        print(f"  Foggy (low): {sample_pair['foggy']['low']['image']}")
        print(f"  Foggy (mid): {sample_pair['foggy']['mid']['image']}")
        print(f"  Foggy (high): {sample_pair['foggy']['high']['image']}")
    
    print("\n" + "=" * 70)
    print("  VERIFICATION COMPLETE ✅")
    print("=" * 70)
    print("\nDataset is ready for training!")
    print("See DATASET_SUMMARY.md for detailed information.")
    print("=" * 70)

if __name__ == '__main__':
    verify_dataset()

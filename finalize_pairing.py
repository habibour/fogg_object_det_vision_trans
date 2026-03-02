"""
Quick finalization script for dataset pairing using symbolic links
"""
import os
import json
from pathlib import Path
import shutil

def create_symlinks_for_foggy(foggy_source_root, paired_output_root, fog_levels):
    """Create symlinks instead of copying foggy images."""
    for fog_level in fog_levels:
        source_dir = Path(foggy_source_root) / fog_level
        target_dir = Path(paired_output_root) / 'foggy' / fog_level / 'JPEGImages'
        
        # Remove existing directory if it exists
        if target_dir.exists():
            print(f"Removing existing {target_dir}")
            shutil.rmtree(target_dir)
        
        # Create parent directory
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink
        print(f"Creating symlink: {target_dir} -> {source_dir}")
        os.symlink(source_dir.absolute(), target_dir)

def copy_annotations(clean_ann_dir, paired_output_root, fog_levels):
    """Copy annotations to all foggy directories."""
    clean_ann_path = Path(clean_ann_dir)
    output_path = Path(paired_output_root)
    
    ann_files = list(clean_ann_path.glob('*.xml'))
    print(f"Copying {len(ann_files)} annotations...")
    
    # Copy to clean
    clean_target = output_path / 'clean' / 'Annotations'
    clean_target.mkdir(parents=True, exist_ok=True)
    for ann_file in ann_files:
        if not (clean_target / ann_file.name).exists():
            shutil.copy2(ann_file, clean_target / ann_file.name)
    
    # Copy to foggy levels
    for fog_level in fog_levels:
        target_dir = output_path / 'foggy' / fog_level / 'Annotations'
        target_dir.mkdir(parents=True, exist_ok=True)
        for ann_file in ann_files:
            if not (target_dir / ann_file.name).exists():
                shutil.copy2(ann_file, target_dir / ann_file.name)

def create_pairs_json(paired_output_root, fog_levels):
    """Create pairs.json mapping file."""
    output_path = Path(paired_output_root)
    clean_images_dir = output_path / 'clean' / 'JPEGImages'
    
    # Get all clean images
    clean_images = list(clean_images_dir.glob('*.jpg'))
    clean_images.extend(clean_images_dir.glob('*.png'))
    
    pairs = {
        'metadata': {
            'num_pairs': len(clean_images),
            'fog_levels': fog_levels,
        },
        'pairs': []
    }
    
    for clean_img in clean_images:
        img_name = clean_img.name
        img_id = clean_img.stem
        
        pair_info = {
            'id': img_id,
            'clean': {
                'image': f'clean/JPEGImages/{img_name}',
                'annotation': f'clean/Annotations/{img_id}.xml'
            },
            'foggy': {}
        }
        
        for fog_level in fog_levels:
            pair_info['foggy'][fog_level] = {
                'image': f'foggy/{fog_level}/JPEGImages/{img_name}',
                'annotation': f'foggy/{fog_level}/Annotations/{img_id}.xml'
            }
        
        pairs['pairs'].append(pair_info)
    
    # Save to JSON
    pairs_file = output_path / 'pairs.json'
    with open(pairs_file, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"Created pairs.json with {len(clean_images)} pairs")

def create_splits(paired_output_root):
    """Create train/val/test splits."""
    import random
    
    output_path = Path(paired_output_root)
    
    # Load pairs
    with open(output_path / 'pairs.json', 'r') as f:
        pairs = json.load(f)
    
    image_ids = [pair['id'] for pair in pairs['pairs']]
    n_total = len(image_ids)
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_ids)
    
    # Split 70/15/15
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    # Create directory
    splits_dir = output_path / 'ImageSets' / 'Main'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'trainval': train_ids + val_ids
    }
    
    for split_name, ids in splits.items():
        with open(splits_dir / f'{split_name}.txt', 'w') as f:
            f.write('\n'.join(ids))
        print(f"{split_name}: {len(ids)} images")

if __name__ == '__main__':
    FILTERED_ROOT = 'voc_2012/processed/VOC2012_filtered'
    FOGGY_ROOT = 'voc_2012/processed/VOC2012_foggy'
    PAIRED_ROOT = 'voc_2012/processed/VOC2012_paired'
    FOG_LEVELS = ['low', 'mid', 'high']
    
    print("=" * 60)
    print("Finalizing Paired Dataset with Symlinks")
    print("=" * 60)
    
    # Step 1: Create symlinks for foggy images (faster than copying)
    print("\n[1/4] Creating symlinks for foggy images...")
    create_symlinks_for_foggy(FOGGY_ROOT, PAIRED_ROOT, FOG_LEVELS)
    
    # Step 2: Copy annotations
    print("\n[2/4] Copying annotations...")
    copy_annotations(
        f'{FILTERED_ROOT}/Annotations',
        PAIRED_ROOT,
        FOG_LEVELS
    )
    
    # Step 3: Create pairs.json
    print("\n[3/4] Creating pairs.json...")
    create_pairs_json(PAIRED_ROOT, FOG_LEVELS)
    
    # Step 4: Create splits
    print("\n[4/4] Creating train/val/test splits...")
    create_splits(PAIRED_ROOT)
    
    print("\n" + "=" * 60)
    print("✅ Paired dataset finalized!")
    print("=" * 60)

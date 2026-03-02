"""
Image Pairing Utilities for Clean and Foggy Images
Based on the paper "Weather-Aware Object Detection Transformer for Domain Adaptation"

This module provides utilities to:
1. Create paired datasets of clean and foggy images
2. Generate paired data loaders for training
3. Maintain annotation consistency across clean/foggy pairs
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

class ImagePairManager:
    """Manage paired clean and foggy image datasets."""
    
    def __init__(self, clean_root: str, foggy_root: str, fog_levels: List[str] = None):
        """
        Initialize image pair manager.
        
        Args:
            clean_root: Root directory of clean images
            foggy_root: Root directory of foggy images
            fog_levels: List of fog levels (e.g., ['low', 'mid', 'high'])
        """
        self.clean_root = Path(clean_root)
        self.foggy_root = Path(foggy_root)
        self.fog_levels = fog_levels or ['low', 'mid', 'high']
        
    def create_paired_structure(self, output_root: str) -> None:
        """
        Create a directory structure with paired clean/foggy images.
        
        Output structure:
            output_root/
                clean/
                    JPEGImages/
                    Annotations/
                foggy/
                    low/
                        JPEGImages/
                        Annotations/
                    mid/
                        JPEGImages/
                        Annotations/
                    high/
                        JPEGImages/
                        Annotations/
                pairs.json  # Mapping of clean to foggy images
        """
        output_path = Path(output_root)
        
        # Create directory structure
        (output_path / 'clean' / 'JPEGImages').mkdir(parents=True, exist_ok=True)
        (output_path / 'clean' / 'Annotations').mkdir(parents=True, exist_ok=True)
        
        for fog_level in self.fog_levels:
            (output_path / 'foggy' / fog_level / 'JPEGImages').mkdir(parents=True, exist_ok=True)
            (output_path / 'foggy' / fog_level / 'Annotations').mkdir(parents=True, exist_ok=True)
        
        print(f"Created paired dataset structure at {output_root}")
        
    def copy_annotations_to_foggy(self, clean_annotations_dir: str, output_root: str) -> None:
        """
        Copy annotations from clean dataset to all foggy versions.
        Annotations remain the same since bounding boxes don't change with fog.
        
        Args:
            clean_annotations_dir: Directory containing clean annotations
            output_root: Root of paired dataset
        """
        clean_ann_path = Path(clean_annotations_dir)
        output_path = Path(output_root)
        
        # Get all annotation files
        ann_files = list(clean_ann_path.glob('*.xml'))
        
        print(f"Copying {len(ann_files)} annotations to foggy directories...")
        
        # Copy to clean directory
        for ann_file in ann_files:
            shutil.copy2(ann_file, output_path / 'clean' / 'Annotations' / ann_file.name)
        
        # Copy to all foggy directories
        for fog_level in self.fog_levels:
            for ann_file in ann_files:
                shutil.copy2(ann_file, output_path / 'foggy' / fog_level / 'Annotations' / ann_file.name)
        
        print("Annotations copied successfully!")
        
    def create_pairs_mapping(self, output_root: str) -> Dict:
        """
        Create a JSON file mapping clean images to their foggy counterparts.
        
        Args:
            output_root: Root of paired dataset
            
        Returns:
            Dictionary with pairing information
        """
        output_path = Path(output_root)
        clean_images_dir = output_path / 'clean' / 'JPEGImages'
        
        # Get all clean images
        image_extensions = ['.jpg', '.jpeg', '.png']
        clean_images = []
        for ext in image_extensions:
            clean_images.extend(clean_images_dir.glob(f'*{ext}'))
            clean_images.extend(clean_images_dir.glob(f'*{ext.upper()}'))
        
        pairs = {
            'metadata': {
                'num_pairs': len(clean_images),
                'fog_levels': self.fog_levels,
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
            
            # Add foggy versions for each level
            for fog_level in self.fog_levels:
                pair_info['foggy'][fog_level] = {
                    'image': f'foggy/{fog_level}/JPEGImages/{img_name}',
                    'annotation': f'foggy/{fog_level}/Annotations/{img_id}.xml'
                }
            
            pairs['pairs'].append(pair_info)
        
        # Save to JSON
        pairs_file = output_path / 'pairs.json'
        with open(pairs_file, 'w') as f:
            json.dump(pairs, f, indent=2)
        
        print(f"Created pairs mapping with {len(clean_images)} image pairs")
        print(f"Saved to {pairs_file}")
        
        return pairs
    
    def create_split_files(self, output_root: str, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
        """
        Create train/val/test split files for paired dataset.
        
        Args:
            output_root: Root of paired dataset
            split_ratio: Tuple of (train, val, test) ratios
        """
        output_path = Path(output_root)
        
        # Load pairs
        with open(output_path / 'pairs.json', 'r') as f:
            pairs = json.load(f)
        
        image_ids = [pair['id'] for pair in pairs['pairs']]
        n_total = len(image_ids)
        
        # Shuffle and split
        import random
        random.shuffle(image_ids)
        
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]
        
        # Create ImageSets directories
        (output_path / 'ImageSets' / 'Main').mkdir(parents=True, exist_ok=True)
        
        # Save split files
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'trainval': train_ids + val_ids
        }
        
        for split_name, ids in splits.items():
            with open(output_path / 'ImageSets' / 'Main' / f'{split_name}.txt', 'w') as f:
                f.write('\n'.join(ids))
            print(f"{split_name}: {len(ids)} images")
        
        print(f"\nSplit files created in {output_path / 'ImageSets' / 'Main'}")
    
    def get_random_pair(self, pairs_json: str, fog_level: Optional[str] = None) -> Dict:
        """
        Get a random clean-foggy image pair.
        
        Args:
            pairs_json: Path to pairs.json file
            fog_level: Specific fog level, or None for random
            
        Returns:
            Dictionary with paths to clean and foggy images
        """
        import random
        
        with open(pairs_json, 'r') as f:
            pairs = json.load(f)
        
        pair = random.choice(pairs['pairs'])
        
        if fog_level is None:
            fog_level = random.choice(self.fog_levels)
        
        return {
            'id': pair['id'],
            'clean_image': pair['clean']['image'],
            'clean_annotation': pair['clean']['annotation'],
            'foggy_image': pair['foggy'][fog_level]['image'],
            'foggy_annotation': pair['foggy'][fog_level]['annotation'],
            'fog_level': fog_level
        }
    
    def verify_pairs(self, output_root: str) -> bool:
        """
        Verify that all pairs exist and are consistent.
        
        Args:
            output_root: Root of paired dataset
            
        Returns:
            True if all pairs are valid
        """
        output_path = Path(output_root)
        
        # Load pairs
        with open(output_path / 'pairs.json', 'r') as f:
            pairs_data = json.load(f)
        
        print("Verifying image pairs...")
        
        errors = []
        for pair in pairs_data['pairs']:
            # Check clean image and annotation
            clean_img = output_path / pair['clean']['image']
            clean_ann = output_path / pair['clean']['annotation']
            
            if not clean_img.exists():
                errors.append(f"Missing clean image: {clean_img}")
            if not clean_ann.exists():
                errors.append(f"Missing clean annotation: {clean_ann}")
            
            # Check all foggy versions
            for fog_level in self.fog_levels:
                foggy_img = output_path / pair['foggy'][fog_level]['image']
                foggy_ann = output_path / pair['foggy'][fog_level]['annotation']
                
                if not foggy_img.exists():
                    errors.append(f"Missing foggy image ({fog_level}): {foggy_img}")
                if not foggy_ann.exists():
                    errors.append(f"Missing foggy annotation ({fog_level}): {foggy_ann}")
        
        if errors:
            print(f"Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return False
        else:
            print(f"✓ All {len(pairs_data['pairs'])} pairs verified successfully!")
            return True

def create_paired_dataset(clean_images_dir: str,
                         clean_annotations_dir: str,
                         foggy_images_root: str,
                         output_root: str,
                         fog_levels: List[str] = None) -> None:
    """
    High-level function to create a complete paired dataset.
    
    Args:
        clean_images_dir: Directory with clean images
        clean_annotations_dir: Directory with clean annotations
        foggy_images_root: Root directory with foggy images (containing fog level subdirs)
        output_root: Output directory for paired dataset
        fog_levels: List of fog levels
    """
    fog_levels = fog_levels or ['low', 'mid', 'high']
    
    print("=" * 60)
    print("Creating Paired Clean-Foggy Dataset")
    print("=" * 60)
    
    manager = ImagePairManager(clean_images_dir, foggy_images_root, fog_levels)
    
    # Step 1: Create directory structure
    print("\n[1/5] Creating directory structure...")
    manager.create_paired_structure(output_root)
    
    # Step 2: Copy clean images
    print("\n[2/5] Copying clean images...")
    clean_imgs = list(Path(clean_images_dir).glob('*.*'))
    for img in clean_imgs:
        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy2(img, Path(output_root) / 'clean' / 'JPEGImages' / img.name)
    print(f"Copied {len(clean_imgs)} clean images")
    
    # Step 3: Copy foggy images
    print("\n[3/5] Copying foggy images...")
    for fog_level in fog_levels:
        foggy_dir = Path(foggy_images_root) / fog_level
        if foggy_dir.exists():
            foggy_imgs = list(foggy_dir.glob('*.*'))
            for img in foggy_imgs:
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, Path(output_root) / 'foggy' / fog_level / 'JPEGImages' / img.name)
            print(f"  {fog_level}: {len(foggy_imgs)} images")
    
    # Step 4: Copy annotations
    print("\n[4/5] Copying annotations...")
    manager.copy_annotations_to_foggy(clean_annotations_dir, output_root)
    
    # Step 5: Create pairs mapping
    print("\n[5/5] Creating pairs mapping...")
    manager.create_pairs_mapping(output_root)
    
    # Verify
    print("\n" + "=" * 60)
    manager.verify_pairs(output_root)
    
    # Create splits
    print("\nCreating train/val/test splits...")
    manager.create_split_files(output_root)
    
    print("\n" + "=" * 60)
    print("Paired dataset creation complete!")
    print(f"Output directory: {output_root}")
    print("=" * 60)

if __name__ == '__main__':
    # Example usage
    CLEAN_IMAGES = 'voc_2012/VOC2012_filtered/JPEGImages'
    CLEAN_ANNOTATIONS = 'voc_2012/VOC2012_filtered/Annotations'
    FOGGY_IMAGES = 'voc_2012/VOC2012_foggy'
    OUTPUT_ROOT = 'voc_2012/VOC2012_paired'
    
    if os.path.exists(CLEAN_IMAGES) and os.path.exists(FOGGY_IMAGES):
        create_paired_dataset(
            clean_images_dir=CLEAN_IMAGES,
            clean_annotations_dir=CLEAN_ANNOTATIONS,
            foggy_images_root=FOGGY_IMAGES,
            output_root=OUTPUT_ROOT,
            fog_levels=['low', 'mid', 'high']
        )
    else:
        print("Required directories not found!")
        print(f"Clean images: {CLEAN_IMAGES} - {'✓' if os.path.exists(CLEAN_IMAGES) else '✗'}")
        print(f"Foggy images: {FOGGY_IMAGES} - {'✓' if os.path.exists(FOGGY_IMAGES) else '✗'}")
        print("\nPlease run the previous scripts first:")
        print("  1. filter_voc_classes.py")
        print("  2. synthetic_fog.py")

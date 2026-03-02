"""
Filter Pascal VOC dataset to keep only specific classes.
Based on the paper "Weather-Aware Object Detection Transformer for Domain Adaptation"
Filters VOC dataset to keep only: bicycle, bus, car, motorbike, person
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from tqdm import tqdm

# Classes to keep (matching RTTS dataset)
TARGET_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']

# VOC class mapping (original 20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Create mapping from original VOC to filtered indices
CLASS_MAPPING = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

def parse_voc_annotation(xml_path):
    """Parse VOC XML annotation file and extract object information."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name in TARGET_CLASSES:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            objects.append({
                'class': cls_name,
                'class_id': CLASS_MAPPING[cls_name],
                'bbox': (xmin, ymin, xmax, ymax),
                'difficult': int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            })
    
    # Get image info
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    filename = root.find('filename').text
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

def filter_voc_dataset(voc_root, output_root, split='train'):
    """
    Filter VOC dataset to keep only target classes.
    
    Args:
        voc_root: Path to VOC dataset root (e.g., voc_2012/VOC2012_train_val/VOC2012_train_val)
        output_root: Path to output filtered dataset
        split: 'train', 'val', or 'trainval'
    """
    voc_path = Path(voc_root)
    output_path = Path(output_root)
    
    # Create output directories
    (output_path / 'Annotations').mkdir(parents=True, exist_ok=True)
    (output_path / 'JPEGImages').mkdir(parents=True, exist_ok=True)
    (output_path / 'ImageSets' / 'Main').mkdir(parents=True, exist_ok=True)
    
    # Get image IDs from split file
    if split == 'train':
        split_file = voc_path / 'ImageSets' / 'Main' / 'train.txt'
    elif split == 'val':
        split_file = voc_path / 'ImageSets' / 'Main' / 'val.txt'
    elif split == 'trainval':
        split_file = voc_path / 'ImageSets' / 'Main' / 'trainval.txt'
    else:
        raise ValueError(f"Unknown split: {split}")
    
    if not split_file.exists():
        print(f"Warning: Split file {split_file} not found, using all annotations")
        # Get all annotation files
        annotation_files = list((voc_path / 'Annotations').glob('*.xml'))
        image_ids = [f.stem for f in annotation_files]
    else:
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
    
    filtered_ids = []
    stats = {cls: 0 for cls in TARGET_CLASSES}
    total_objects = 0
    
    print(f"Filtering {len(image_ids)} images from {split} split...")
    
    for img_id in tqdm(image_ids):
        xml_path = voc_path / 'Annotations' / f'{img_id}.xml'
        
        if not xml_path.exists():
            continue
        
        # Parse annotation
        ann_data = parse_voc_annotation(xml_path)
        
        # Skip images without target classes
        if len(ann_data['objects']) == 0:
            continue
        
        # Copy image and annotation
        img_src = voc_path / 'JPEGImages' / ann_data['filename']
        img_dst = output_path / 'JPEGImages' / ann_data['filename']
        
        if img_src.exists():
            shutil.copy2(img_src, img_dst)
            shutil.copy2(xml_path, output_path / 'Annotations' / f'{img_id}.xml')
            
            filtered_ids.append(img_id)
            
            # Update statistics
            for obj in ann_data['objects']:
                stats[obj['class']] += 1
                total_objects += 1
    
    # Save filtered image IDs
    with open(output_path / 'ImageSets' / 'Main' / f'{split}.txt', 'w') as f:
        f.write('\n'.join(filtered_ids))
    
    # Save class-specific lists
    for cls in TARGET_CLASSES:
        cls_ids = []
        for img_id in filtered_ids:
            xml_path = output_path / 'Annotations' / f'{img_id}.xml'
            ann_data = parse_voc_annotation(xml_path)
            has_class = any(obj['class'] == cls for obj in ann_data['objects'])
            cls_ids.append(f"{img_id} {1 if has_class else -1}")
        
        with open(output_path / 'ImageSets' / 'Main' / f'{cls}_{split}.txt', 'w') as f:
            f.write('\n'.join(cls_ids))
    
    # Print statistics
    print(f"\nFiltering complete!")
    print(f"Retained {len(filtered_ids)} images with {total_objects} objects")
    print("\nClass distribution:")
    for cls, count in stats.items():
        print(f"  {cls}: {count}")
    
    return filtered_ids, stats

def create_class_mapping_file(output_root):
    """Create a file with class names and IDs for reference."""
    output_path = Path(output_root)
    
    with open(output_path / 'classes.txt', 'w') as f:
        for cls in TARGET_CLASSES:
            f.write(f"{cls}\n")
    
    with open(output_path / 'class_mapping.txt', 'w') as f:
        for cls, idx in CLASS_MAPPING.items():
            f.write(f"{idx}: {cls}\n")
    
    print(f"Saved class mapping to {output_path / 'class_mapping.txt'}")

if __name__ == '__main__':
    # Example usage
    VOC_ROOT = 'voc_2012/VOC2012_train_val/VOC2012_train_val'
    OUTPUT_ROOT = 'voc_2012/VOC2012_filtered'
    
    print("=" * 60)
    print("VOC Dataset Class Filtering")
    print("Target classes:", ', '.join(TARGET_CLASSES))
    print("=" * 60)
    
    # Filter train and val splits
    for split in ['train', 'val']:
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} split")
        print('=' * 60)
        
        try:
            filter_voc_dataset(VOC_ROOT, OUTPUT_ROOT, split=split)
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
    
    # Create class mapping file
    create_class_mapping_file(OUTPUT_ROOT)
    
    print("\n" + "=" * 60)
    print("Filtering complete! Filtered dataset saved to:", OUTPUT_ROOT)
    print("=" * 60)

"""
Check Dataset Issues Before Full Training
Diagnose class imbalance, annotation quality, and dataset statistics
"""

import json
import os
from collections import Counter
import xml.etree.ElementTree as ET

def analyze_dataset_issues(pairs_json, annotations_dir):
    """Analyze potential dataset issues affecting performance."""
    
    print("="*80)
    print("DATASET DIAGNOSIS - Identifying Issues")
    print("="*80)
    
    # Load pairs
    with open(pairs_json, 'r') as f:
        pairs_data = json.load(f)
    
    # Class mapping
    classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    class_counts_train = Counter()
    class_counts_val = Counter()
    class_counts_test = Counter()
    
    # Object size statistics
    small_objects = {cls: 0 for cls in classes}
    medium_objects = {cls: 0 for cls in classes}
    large_objects = {cls: 0 for cls in classes}
    
    # Occlusion/truncation stats
    difficult_objects = {cls: 0 for cls in classes}
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        split_file = f"{annotations_dir}/../ImageSets/Main/{split}.txt"
        if not os.path.exists(split_file):
            continue
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]
        
        for img_id in image_ids:
            xml_file = f"{annotations_dir}/{img_id}.xml"
            if not os.path.exists(xml_file):
                continue
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            img_area = img_width * img_height
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    continue
                
                # Count instances
                if split == 'train':
                    class_counts_train[class_name] += 1
                elif split == 'val':
                    class_counts_val[class_name] += 1
                else:
                    class_counts_test[class_name] += 1
                
                # Check difficulty
                difficult = obj.find('difficult')
                truncated = obj.find('truncated')
                if (difficult is not None and int(difficult.text) == 1) or \
                   (truncated is not None and int(truncated.text) == 1):
                    difficult_objects[class_name] += 1
                
                # Get bbox
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Calculate relative size
                obj_area = (xmax - xmin) * (ymax - ymin)
                relative_size = obj_area / img_area
                
                # Classify by size
                if relative_size < 0.01:  # < 1% of image
                    small_objects[class_name] += 1
                elif relative_size < 0.1:  # 1-10% of image
                    medium_objects[class_name] += 1
                else:  # > 10% of image
                    large_objects[class_name] += 1
    
    # Print results
    print("\n📊 CLASS DISTRIBUTION:")
    print(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'% of Total':<12}")
    print("-" * 80)
    
    total_all = sum(class_counts_train.values()) + sum(class_counts_val.values()) + sum(class_counts_test.values())
    
    for cls in classes:
        train_count = class_counts_train[cls]
        val_count = class_counts_val[cls]
        test_count = class_counts_test[cls]
        total = train_count + val_count + test_count
        percentage = (total / total_all * 100) if total_all > 0 else 0
        
        print(f"{cls:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10} {percentage:<12.2f}%")
    
    print("\n⚠️  CLASS IMBALANCE ANALYSIS:")
    max_count = max([class_counts_train[cls] for cls in classes])
    for cls in classes:
        ratio = class_counts_train[cls] / max_count if max_count > 0 else 0
        status = "❌ SEVERE" if ratio < 0.1 else "⚠️  MODERATE" if ratio < 0.5 else "✅ OK"
        print(f"  {cls:<15} {status:<12} (ratio: {ratio:.3f})")
    
    print("\n📏 OBJECT SIZE DISTRIBUTION:")
    print(f"{'Class':<15} {'Small (<1%)':<15} {'Medium (1-10%)':<18} {'Large (>10%)':<15}")
    print("-" * 80)
    
    for cls in classes:
        small = small_objects[cls]
        medium = medium_objects[cls]
        large = large_objects[cls]
        total = small + medium + large
        
        small_pct = (small / total * 100) if total > 0 else 0
        medium_pct = (medium / total * 100) if total > 0 else 0
        large_pct = (large / total * 100) if total > 0 else 0
        
        print(f"{cls:<15} {small} ({small_pct:.1f}%){'':<6} {medium} ({medium_pct:.1f}%){'':<6} {large} ({large_pct:.1f}%)")
    
    print("\n🚫 DIFFICULT OBJECTS (truncated/occluded):")
    print(f"{'Class':<15} {'Difficult Count':<20} {'% of Total':<15}")
    print("-" * 80)
    
    for cls in classes:
        difficult = difficult_objects[cls]
        total = class_counts_train[cls] + class_counts_val[cls] + class_counts_test[cls]
        pct = (difficult / total * 100) if total > 0 else 0
        status = "⚠️" if pct > 20 else ""
        print(f"{cls:<15} {difficult:<20} {pct:<15.2f}% {status}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    # Generate recommendations
    recommendations = []
    
    # Check class imbalance
    for cls in classes:
        ratio = class_counts_train[cls] / max_count if max_count > 0 else 0
        if ratio < 0.1:
            recommendations.append(f"❌ CRITICAL: '{cls}' has severe class imbalance (ratio: {ratio:.3f})")
            recommendations.append(f"   → Use class weights in loss: weight[{cls}] = {1/ratio:.2f}")
        elif ratio < 0.5:
            recommendations.append(f"⚠️  WARNING: '{cls}' has moderate imbalance (ratio: {ratio:.3f})")
            recommendations.append(f"   → Consider oversampling or class weight = {1/ratio:.2f}")
    
    # Check small objects
    for cls in classes:
        total = small_objects[cls] + medium_objects[cls] + large_objects[cls]
        if total > 0:
            small_ratio = small_objects[cls] / total
            if small_ratio > 0.5:
                recommendations.append(f"⚠️  '{cls}' has {small_ratio*100:.1f}% small objects")
                recommendations.append(f"   → Use multi-scale training and augmentation")
    
    # Check difficult objects
    for cls in classes:
        total = class_counts_train[cls] + class_counts_val[cls] + class_counts_test[cls]
        if total > 0:
            difficult_ratio = difficult_objects[cls] / total
            if difficult_ratio > 0.3:
                recommendations.append(f"⚠️  '{cls}' has {difficult_ratio*100:.1f}% difficult objects")
                recommendations.append(f"   → Review annotation quality and training strategy")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return {
        'class_counts_train': dict(class_counts_train),
        'class_counts_val': dict(class_counts_val),
        'class_counts_test': dict(class_counts_test),
        'small_objects': small_objects,
        'medium_objects': medium_objects,
        'large_objects': large_objects,
        'difficult_objects': difficult_objects
    }


if __name__ == '__main__':
    # Update these paths based on your dataset
    PAIRS_JSON = 'voc_2012/processed/VOC2012_paired/pairs.json'
    ANNOTATIONS_DIR = 'voc_2012/processed/VOC2012_filtered/Annotations'
    
    stats = analyze_dataset_issues(PAIRS_JSON, ANNOTATIONS_DIR)

"""
Convert VOC XML annotations to YOLO format for Ultralytics RT-DETR

YOLO format: Each image has a corresponding .txt file with one line per object:
<class_id> <x_center> <y_center> <width> <height>
All values are normalized to [0, 1]
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm


def convert_voc_box_to_yolo(size, box):
    """
    Convert VOC bounding box to YOLO format.
    
    Args:
        size: (width, height) of the image
        box: (xmin, ymin, xmax, ymax) in pixels
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    
    return (x_center, y_center, width, height)


def convert_annotation(xml_path, output_path, classes):
    """
    Convert a single VOC XML annotation to YOLO format.
    
    Args:
        xml_path: Path to VOC XML file
        output_path: Path to output YOLO .txt file
        classes: List of class names
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # Convert each object
    yolo_annotations = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        
        cls_id = classes.index(cls_name)
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format
        x_center, y_center, width, height = convert_voc_box_to_yolo(
            (w, h), (xmin, ymin, xmax, ymax)
        )
        
        yolo_annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.writelines(yolo_annotations)


def convert_dataset(voc_root, output_root, dataset_type='clean'):
    """
    Convert entire VOC dataset to YOLO format.
    
    Args:
        voc_root: Root directory of VOC dataset (contains Annotations, JPEGImages)
        output_root: Output directory for YOLO format dataset
        dataset_type: 'clean', 'foggy_low', 'foggy_mid', or 'foggy_high'
    """
    voc_root = Path(voc_root)
    output_root = Path(output_root)
    
    # Class names (matching our filtered VOC)
    classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    
    # Create output structure
    images_dir = output_root / 'images' / dataset_type
    labels_dir = output_root / 'labels' / dataset_type
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    annotations_dir = voc_root / 'Annotations'
    xml_files = list(annotations_dir.glob('*.xml'))
    
    print(f"\nConverting {len(xml_files)} annotations for {dataset_type}...")
    
    for xml_file in tqdm(xml_files):
        image_id = xml_file.stem
        
        # Convert annotation
        yolo_file = labels_dir / f'{image_id}.txt'
        convert_annotation(xml_file, yolo_file, classes)
        
        # Copy image (or create symlink)
        src_image = voc_root / 'JPEGImages' / f'{image_id}.jpg'
        dst_image = images_dir / f'{image_id}.jpg'
        
        if not dst_image.exists():
            if src_image.exists():
                # Use symlink to save space
                dst_image.symlink_to(src_image.resolve())
            else:
                print(f"Warning: Image not found: {src_image}")
    
    print(f"✅ Converted {dataset_type}: {len(xml_files)} images")
    return len(xml_files)


def create_yaml_config(output_root, dataset_name='voc_custom'):
    """
    Create YAML configuration file for Ultralytics.
    
    Args:
        output_root: Root directory containing images/ and labels/
        dataset_name: Name for the YAML file
    """
    output_root = Path(output_root)
    
    yaml_content = f"""# VOC Custom Dataset for RT-DETR
# Auto-generated for Weather-Aware Object Detection

path: {output_root.resolve()}  # dataset root dir
train: images/clean  # train images (relative to 'path')
val: images/clean    # val images (relative to 'path')

# Classes
names:
  0: bicycle
  1: bus
  2: car
  3: motorbike
  4: person

# Number of classes
nc: 5

# Additional dataset info
download: false
"""
    
    yaml_path = output_root / f'{dataset_name}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created dataset config: {yaml_path}")
    return yaml_path


def main():
    """Convert VOC dataset to YOLO format."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert VOC XML to YOLO format')
    parser.add_argument(
        '--voc_root',
        type=str,
        default='voc_2012/processed/VOC2012_paired',
        help='Root directory of VOC dataset'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='voc_2012/yolo_format',
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--include_foggy',
        action='store_true',
        help='Also convert foggy images (low/mid/high fog)'
    )
    
    args = parser.parse_args()
    
    voc_root = Path(args.voc_root)
    output_root = Path(args.output_root)
    
    print("="*60)
    print("VOC to YOLO Format Converter")
    print("="*60)
    
    # Convert clean dataset
    clean_dir = voc_root / 'clean'
    if clean_dir.exists():
        convert_dataset(clean_dir, output_root, 'clean')
    else:
        print(f"Warning: Clean directory not found: {clean_dir}")
    
    # Convert foggy datasets if requested
    if args.include_foggy:
        foggy_dir = voc_root / 'foggy'
        
        for fog_level in ['low', 'mid', 'high']:
            fog_dir = foggy_dir / fog_level
            if fog_dir.exists():
                # For foggy, we still use clean annotations
                # but link to foggy images
                convert_dataset_foggy(
                    clean_dir,  # annotations
                    fog_dir,    # images
                    output_root,
                    f'foggy_{fog_level}'
                )
            else:
                print(f"Warning: Foggy directory not found: {fog_dir}")
    
    # Create YAML config
    yaml_path = create_yaml_config(output_root)
    
    print("\n" + "="*60)
    print("✅ Conversion Complete!")
    print("="*60)
    print(f"\nDataset ready at: {output_root.resolve()}")
    print(f"Config file: {yaml_path}")
    print("\nYou can now train RT-DETR with:")
    print(f"  from ultralytics import RTDETR")
    print(f"  model = RTDETR('rtdetr-l.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=100)")


def convert_dataset_foggy(voc_annotations_root, foggy_images_root, output_root, dataset_type):
    """
    Convert foggy dataset using clean annotations but foggy images.
    
    Args:
        voc_annotations_root: Root with Annotations folder (clean)
        foggy_images_root: Root with JPEGImages folder (foggy)
        output_root: Output directory
        dataset_type: 'foggy_low', 'foggy_mid', or 'foggy_high'
    """
    voc_annotations_root = Path(voc_annotations_root)
    foggy_images_root = Path(foggy_images_root)
    output_root = Path(output_root)
    
    classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    
    # Create output structure
    images_dir = output_root / 'images' / dataset_type
    labels_dir = output_root / 'labels' / dataset_type
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files from clean annotations
    annotations_dir = voc_annotations_root / 'Annotations'
    xml_files = list(annotations_dir.glob('*.xml'))
    
    print(f"\nConverting {len(xml_files)} annotations for {dataset_type}...")
    
    for xml_file in tqdm(xml_files):
        image_id = xml_file.stem
        
        # Convert annotation (from clean)
        yolo_file = labels_dir / f'{image_id}.txt'
        convert_annotation(xml_file, yolo_file, classes)
        
        # Link foggy image
        src_image = foggy_images_root / 'JPEGImages' / f'{image_id}.jpg'
        dst_image = images_dir / f'{image_id}.jpg'
        
        if not dst_image.exists():
            if src_image.exists():
                dst_image.symlink_to(src_image.resolve())
            else:
                print(f"Warning: Foggy image not found: {src_image}")
    
    print(f"✅ Converted {dataset_type}: {len(xml_files)} images")


if __name__ == '__main__':
    main()

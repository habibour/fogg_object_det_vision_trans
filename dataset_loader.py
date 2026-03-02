"""
Dataset Loader for Weather-Aware Object Detection
Implements paired clean-foggy image loading for PL-RT-DETR training
Based on: "Weather-Aware Object Detection Transformer for Domain Adaptation"
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


class VOCPairedDataset(Dataset):
    """
    Dataset for paired clean-foggy images from VOC.
    Returns both clean and foggy versions of the same image for teacher-student training.
    """
    
    def __init__(
        self,
        pairs_json_path: str,
        dataset_root: str,
        split: str = 'train',
        fog_level: Optional[str] = None,
        random_fog: bool = True,
        transform=None,
        return_clean: bool = True,
        return_foggy: bool = True
    ):
        """
        Args:
            pairs_json_path: Path to pairs.json
            dataset_root: Root directory containing the paired dataset
            split: 'train', 'val', or 'test'
            fog_level: Specific fog level ('low', 'mid', 'high') or None for random
            random_fog: If True, randomly select fog level during training
            transform: Albumentations or torchvision transforms
            return_clean: Whether to return clean image
            return_foggy: Whether to return foggy image
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.fog_level = fog_level
        self.random_fog = random_fog
        self.transform = transform
        self.return_clean = return_clean
        self.return_foggy = return_foggy
        
        # Load pairs data
        with open(pairs_json_path, 'r') as f:
            pairs_data = json.load(f)
        
        self.fog_levels = pairs_data['metadata']['fog_levels']
        self.all_pairs = pairs_data['pairs']
        
        # Load split IDs
        split_file = self.dataset_root / 'ImageSets' / 'Main' / f'{split}.txt'
        with open(split_file, 'r') as f:
            split_ids = set(line.strip() for line in f)
        
        # Filter pairs for this split
        self.pairs = [p for p in self.all_pairs if p['id'] in split_ids]
        
        print(f"Loaded {len(self.pairs)} image pairs for {split} split")
        
        # Class mapping
        self.classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.pairs)
    
    def parse_voc_xml(self, xml_path: str) -> Dict:
        """Parse VOC XML annotation."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in self.class_to_idx:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        return {
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64),
            'image_size': (height, width)
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing:
        - clean_image: Clean image tensor
        - foggy_image: Foggy image tensor
        - targets: Bounding boxes and labels
        - metadata: Additional info (image_id, fog_level, etc.)
        """
        pair = self.pairs[idx]
        image_id = pair['id']
        
        # Select fog level
        if self.random_fog and self.split == 'train':
            selected_fog_level = random.choice(self.fog_levels)
        elif self.fog_level:
            selected_fog_level = self.fog_level
        else:
            selected_fog_level = 'mid'  # default
        
        result = {'image_id': image_id, 'fog_level': selected_fog_level}
        
        # Load clean image
        if self.return_clean:
            # Clean images might also be in VOC2012_filtered, not VOC2012_paired
            # Try VOC2012_paired first, fallback to VOC2012_filtered
            clean_img_path = self.dataset_root / pair['clean']['image']
            if not clean_img_path.exists():
                # Fallback to VOC2012_filtered
                filtered_base = self.dataset_root.parent / 'VOC2012_filtered'
                clean_img_path = filtered_base / 'JPEGImages' / f'{image_id}.jpg'
            clean_image = cv2.imread(str(clean_img_path))
            if clean_image is None:
                raise FileNotFoundError(f"Clean image not found: {clean_img_path}")
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
            result['clean_image'] = clean_image
        
        # Load foggy image
        if self.return_foggy:
            # Foggy images are actually in VOC2012_foggy directory, not VOC2012_paired
            # Construct path to VOC2012_foggy/{level}/{image_id}.jpg
            foggy_base = self.dataset_root.parent / 'VOC2012_foggy'
            foggy_img_path = foggy_base / selected_fog_level / f'{image_id}.jpg'
            foggy_image = cv2.imread(str(foggy_img_path))
            if foggy_image is None:
                raise FileNotFoundError(f"Foggy image not found: {foggy_img_path}")
            foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)
            result['foggy_image'] = foggy_image
        
        # Load annotations (same for clean and foggy)
        # Try VOC2012_paired first, fallback to VOC2012_filtered
        ann_path = self.dataset_root / pair['clean']['annotation']
        if not ann_path.exists():
            # Fallback to VOC2012_filtered
            filtered_base = self.dataset_root.parent / 'VOC2012_filtered'
            ann_path = filtered_base / 'Annotations' / f'{image_id}.xml'
        annotations = self.parse_voc_xml(str(ann_path))
        
        result['boxes'] = annotations['boxes']
        result['labels'] = annotations['labels']
        result['image_size'] = annotations['image_size']
        
        # Apply transforms if provided
        if self.transform:
            result = self.transform(result)
        
        return result


class RTDETRDataset(Dataset):
    """
    Dataset wrapper for RT-DETR format.
    Converts VOC paired data to RT-DETR input format.
    Can return both clean and foggy images for knowledge distillation.
    """
    
    def __init__(
        self,
        pairs_json_path: str,
        dataset_root: str,
        split: str = 'train',
        img_size: int = 640,
        use_foggy: bool = False,
        fog_level: Optional[str] = None,
        random_fog: bool = True,
        return_both: bool = False  # Return both clean and foggy
    ):
        """
        Args:
            pairs_json_path: Path to pairs.json
            dataset_root: Root directory
            split: 'train', 'val', or 'test'
            img_size: Target image size for RT-DETR
            use_foggy: If True, use foggy images; if False, use clean images
            fog_level: Specific fog level or None for random
            random_fog: Random fog level selection during training
            return_both: If True, return both clean and foggy (for student training)
        """
        self.img_size = img_size
        self.use_foggy = use_foggy
        self.return_both = return_both
        
        # Use VOCPairedDataset as base
        self.base_dataset = VOCPairedDataset(
            pairs_json_path=pairs_json_path,
            dataset_root=dataset_root,
            split=split,
            fog_level=fog_level,
            random_fog=random_fog,
            return_clean=True if return_both else not use_foggy,
            return_foggy=True if return_both else use_foggy
        )
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Preprocessed image tensor (C, H, W)  OR dict with clean_image and foggy_image
            target: Dictionary with boxes, labels, image_id
        """
        data = self.base_dataset[idx]
        
        def process_image(img):
            """Process a single image."""
            # Resize image
            h, w = img.shape[:2]
            scale = self.img_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Pad to square
            pad_h = self.img_size - new_h
            pad_w = self.img_size - new_w
            img_padded = cv2.copyMakeBorder(
                img_resized, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            
            # Convert to tensor and normalize
            img_tensor = img_padded.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)  # HWC -> CHW
            
            return img_tensor, scale
        
        # Process images
        if self.return_both:
            # Return both clean and foggy for student training
            clean_tensor, scale = process_image(data['clean_image'])
            foggy_tensor, _ = process_image(data['foggy_image'])
        else:
            # Return only the requested type
            if self.use_foggy:
                image_tensor, scale = process_image(data['foggy_image'])
            else:
                image_tensor, scale = process_image(data['clean_image'])
        
        # Scale bounding boxes
        boxes = data['boxes'].copy()
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale  # x coordinates
            boxes[:, [1, 3]] *= scale  # y coordinates
        
        # Prepare target
        target = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(data['labels']),
            'image_id': data['image_id'],
            'orig_size': torch.tensor(data['image_size']),
            'size': torch.tensor([self.img_size, self.img_size])
        }
        
        if self.return_both:
            # Return dict with both images
            return {'clean_image': clean_tensor, 'foggy_image': foggy_tensor}, target
        else:
            return image_tensor, target


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    targets = []
    clean_images = []
    foggy_images = []
    has_both = False
    
    for item in batch:
        image, target = item
        
        # Check if image is a dict (both clean and foggy) or a tensor
        if isinstance(image, dict):
            has_both = True
            clean_images.append(image['clean_image'])
            foggy_images.append(image['foggy_image'])
        else:
            images.append(image)
        
        targets.append(target)
    
    # Return appropriate format
    if has_both:
        return {
            'clean_image': torch.stack(clean_images, dim=0),
            'foggy_image': torch.stack(foggy_images, dim=0),
            'images': torch.stack(foggy_images, dim=0),  # Default to foggy for compatibility
            'targets': targets
        }
    else:
        return {
            'images': torch.stack(images, dim=0),
            'targets': targets  
        }


def create_dataloaders(
    pairs_json_path: str,
    dataset_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 640,
    use_foggy_train: bool = False,
    use_foggy_val: bool = False
):
    """
    Create train and validation dataloaders.
    
    Args:
        pairs_json_path: Path to pairs.json
        dataset_root: Root directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Image size for RT-DETR
        use_foggy_train: Use foggy images for training
        use_foggy_val: Use foggy images for validation
        
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = RTDETRDataset(
        pairs_json_path=pairs_json_path,
        dataset_root=dataset_root,
        split='train',
        img_size=img_size,
        use_foggy=use_foggy_train,
        random_fog=True
    )
    
    val_dataset = RTDETRDataset(
        pairs_json_path=pairs_json_path,
        dataset_root=dataset_root,
        split='val',
        img_size=img_size,
        use_foggy=use_foggy_val,
        random_fog=False,
        fog_level='mid'  # Use mid fog for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset loader
    pairs_json = 'voc_2012/processed/VOC2012_paired/pairs.json'
    dataset_root = 'voc_2012/processed/VOC2012_paired'
    
    print("Testing VOCPairedDataset...")
    dataset = VOCPairedDataset(
        pairs_json_path=pairs_json,
        dataset_root=dataset_root,
        split='train',
        random_fog=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Fog level: {sample['fog_level']}")
    print(f"  Clean image shape: {sample['clean_image'].shape}")
    print(f"  Foggy image shape: {sample['foggy_image'].shape}")
    print(f"  Boxes: {sample['boxes'].shape}")
    print(f"  Labels: {sample['labels']}")
    
    print("\n" + "="*60)
    print("Testing RTDETRDataset...")
    rtdetr_dataset = RTDETRDataset(
        pairs_json_path=pairs_json,
        dataset_root=dataset_root,
        split='train',
        img_size=640,
        use_foggy=True
    )
    
    image, target = rtdetr_dataset[0]
    print(f"Image tensor shape: {image.shape}")
    print(f"Target keys: {target.keys()}")
    print(f"Boxes shape: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
    
    print("\n" + "="*60)
    print("Testing DataLoader...")
    train_loader, val_loader = create_dataloaders(
        pairs_json_path=pairs_json,
        dataset_root=dataset_root,
        batch_size=4,
        num_workers=0,
        use_foggy_train=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    images, targets = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch size: {len(targets)}")
    
    print("\n✅ Dataset loader test complete!")

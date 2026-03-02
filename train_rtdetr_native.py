"""
Simplified RT-DETR Training using Ultralytics Native API
For Kaggle notebook - uses Ultralytics .train() method directly
"""

import os
from pathlib import Path
from ultralytics import RTDETR
import yaml

def create_yaml_config(dataset_root, output_dir):
    """Create YAML config for Ultralytics training."""
    
    # Define paths
    train_images = f"{dataset_root}/clean/JPEGImages"
    val_images = f"{dataset_root}/clean/JPEGImages"
    
    # YOLO format config
    config = {
        'path': dataset_root,
        'train': 'clean/JPEGImages',
        'val': 'clean/JPEGImages',
        'names': {
            0: 'bicycle',
            1: 'bus', 
            2: 'car',
            3: 'motorbike',
            4: 'person'
        },
        'nc': 5  # number of classes
    }
    
    # Save config
    yaml_path = f"{output_dir}/voc_rtdetr.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    return yaml_path


def train_teacher_native(
    dataset_root,
    output_dir,
    epochs=20,
    batch_size=8,
    img_size=640,
    device='cuda',
    resume=False
):
    """
    Train teacher model using Ultralytics native training.
    
    Args:
        dataset_root: Path to VOC2012_paired directory
        output_dir: Output directory for checkpoints and logs
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device ('cuda' or 'cpu')
        resume: Resume from last checkpoint
    """
    
    print("="*60)
    print("NATIVE ULTRALYTICS RT-DETR TRAINING")
    print("="*60)
    
    # Create YAML config
    print("\n📝 Creating dataset config...")
    yaml_path = create_yaml_config(dataset_root, output_dir)
    print(f"   Config saved to: {yaml_path}")
    
    # Load RT-DETR model
    print("\n🤖 Loading RT-DETR-L model...")
    model = RTDETR('rtdetr-l.pt')
    
    # Train
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print(f"   Device: {device}")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name='teacher',
        save=True,
        save_period=2,  # Save every 2 epochs
        val=True,
        plots=True,
        verbose=True,
        resume=resume
    )
    
    print("\n✅ Training completed!")
    print(f"   Results saved to: {output_dir}/teacher")
    
    return results, model


def train_student_native(
    dataset_root,
    teacher_weights,
    output_dir,
    epochs=20,
    batch_size=8,
    img_size=640,
    device='cuda',
    use_foggy=True
):
    """
    Train student model on foggy images.
    
    Args:
        dataset_root: Path to VOC2012_paired directory
        teacher_weights: Path to teacher checkpoint
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Image size
        device: Device
        use_foggy: Use foggy images for training
    """
    
    print("="*60)
    print("STUDENT TRAINING ON FOGGY IMAGES")
    print("="*60)
    
    # Create YAML config for foggy images
    print("\n📝 Creating foggy dataset config...")
    
    config = {
        'path': dataset_root,
        'train': 'foggy/mid' if use_foggy else 'clean/JPEGImages',
        'val': 'foggy/mid' if use_foggy else 'clean/JPEGImages',
        'names': {
            0: 'bicycle',
            1: 'bus',
            2: 'car',
            3: 'motorbike',
            4: 'person'
        },
        'nc': 5
    }
    
    yaml_path = f"{output_dir}/voc_foggy_rtdetr.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"   Config saved to: {yaml_path}")
    
    # Load model
    print(f"\n🤖 Loading student model (initialized from teacher)...")
    if os.path.exists(teacher_weights):
        print(f"   Loading teacher weights from: {teacher_weights}")
        model = RTDETR(teacher_weights)
    else:
        print(f"   Teacher weights not found, using pretrained RT-DETR-L")
        model = RTDETR('rtdetr-l.pt')
    
    # Train
    print(f"\n🚀 Starting student training for {epochs} epochs...")
    print(f"   Training on: {'foggy images' if use_foggy else 'clean images'}")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name='student',
        save=True,
        save_period=2,
        val=True,
        plots=True,
        verbose=True
    )
    
    print("\n✅ Student training completed!")
    print(f"   Results saved to: {output_dir}/student")
    
    return results, model


if __name__ == "__main__":
    # Example usage
    DATASET_ROOT = "/kaggle/input/datasets/mdhabibourrahman/voc-2012-filtered/voc_2012/processed/VOC2012_paired"
    OUTPUT_DIR = "/kaggle/working/logs"
    
    # Train teacher
    teacher_results, teacher_model = train_teacher_native(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        epochs=20,
        batch_size=8
    )
    
    # Get teacher weights
    teacher_weights = f"{OUTPUT_DIR}/teacher/weights/last.pt"
    
    # Train student
    student_results, student_model = train_student_native(
        dataset_root=DATASET_ROOT,
        teacher_weights=teacher_weights,
        output_dir=OUTPUT_DIR,
        epochs=20,
        batch_size=8,
        use_foggy=True
    )

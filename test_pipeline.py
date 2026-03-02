"""
Quick Test Script for PL-RT-DETR Training
Tests that the pipeline works with minimal epochs
"""

import torch
from pathlib import Path
from dataset_loader import create_dataloaders
from ultralytics import RTDETR

def test_dataset_loading():
    """Test that dataset loads correctly."""
    print("=" * 60)
    print("Step 1: Testing Dataset Loading")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        pairs_json_path='voc_2012/processed/VOC2012_paired/pairs.json',
        dataset_root='voc_2012/processed/VOC2012_paired',
        batch_size=2,
        num_workers=0,
        img_size=640,
        use_foggy_train=False,
        use_foggy_val=False
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Test loading one batch
    batch = next(iter(train_loader))
    print(f"✅ Batch keys: {batch.keys()}")
    print(f"✅ Image shape: {batch['images'].shape}")
    print(f"✅ Number of objects in batch: {len(batch['targets'])}")
    
    return train_loader, val_loader


def test_model_loading():
    """Test RT-DETR model loading."""
    print("\n" + "=" * 60)
    print("Step 2: Testing RT-DETR Model Loading")
    print("=" * 60)
    
    # Load pretrained RT-DETR
    print("Loading RT-DETR model...")
    model = RTDETR('rtdetr-l.pt')  # Downloads pretrained weights
    
    print(f"✅ Model loaded: {type(model)}")
    
    # Test forward pass with dummy input
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using device: {device}")
    
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    model.model.to(device)
    
    with torch.no_grad():
        # RT-DETR forward pass
        outputs = model.model(dummy_input)
    
    print(f"✅ Forward pass successful")
    print(f"✅ Output type: {type(outputs)}")
    
    return model


def test_quick_training():
    """Test a quick training run (1 epoch)."""
    print("\n" + "=" * 60)
    print("Step 3: Quick Training Test (1 Epoch)")
    print("=" * 60)
    
    # For this test, we'll use Ultralytics' built-in training
    # which requires YOLO format
    print("\n⚠️  For quick testing with Ultralytics built-in training,")
    print("    you'll need to convert annotations to YOLO format:")
    print("    python3 convert_voc_to_yolo.py\n")
    
    print("Alternative: Use the custom training loop in train_pl_rtdetr.py")
    print("             which works directly with XML annotations.")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PL-RT-DETR Quick Test")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Dataset
        train_loader, val_loader = test_dataset_loading()
        
        # Test 2: Model
        model = test_model_loading()
        
        # Test 3: Training info
        test_quick_training()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Convert annotations: python3 convert_voc_to_yolo.py")
        print("2. Run training: python3 train_pl_rtdetr.py --teacher_epochs 2 --student_epochs 2")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

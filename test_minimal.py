#!/usr/bin/env python3  
"""
Minimal test - just validates data loading without running full training
"""

def test_imports():
    """Test that basic modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch not installed: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ❌ OpenCV not installed")
        return False
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
        return False
    
    return True

def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        from dataset_loader import VOCPairedDataset, RTDETRDataset
        
        # Load dataset
        dataset = VOCPairedDataset(
            pairs_json_path='voc_2012/processed/VOC2012_paired/pairs.json',
            dataset_root='voc_2012/processed/VOC2012_paired',
            split='train',
            random_fog=True
        )
        
        print(f"  ✅ Dataset loaded: {len(dataset)} samples")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"  ✅ Sample {sample['image_id']}: {sample['clean_image'].shape}")
        print(f"  ✅ Boxes: {len(sample['boxes'])}, Labels: {sample['labels']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """Test PyTorch DataLoader."""
    print("\nTest RTDETRDataset and DataLoader...")
    
    try:
        from dataset_loader import RTDETRDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = RTDETRDataset(
            pairs_json_path='voc_2012/processed/VOC2012_paired/pairs.json',
            dataset_root='voc_2012/processed/VOC2012_paired',
            split='train',
            img_size=640,
            use_foggy=False,
            return_both=False
        )
        
        print(f"  ✅ RT-DETR Dataset: {len(dataset)} samples")
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        print(f"  ✅ DataLoader: {len(loader)} batches")
        
        # Load one batch
        batch = next(iter(loader))
        print(f"  ✅ Batch loaded:")
        print(f"     - Images shape: {batch['images'].shape}")
        print(f"     - Num targets: {len(batch['targets'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_paired_dataloader():
    """Test dataloader with both clean and foggy images."""
    print("\nTesting paired (clean + foggy) dataloader...")
    
    try:
        from dataset_loader import RTDETRDataset, collate_fn
        from torch.utils.data import DataLoader
        
        dataset = RTDETRDataset(
            pairs_json_path='voc_2012/processed/VOC2012_paired/pairs.json',
            dataset_root='voc_2012/processed/VOC2012_paired',
            split='train',
            img_size=640,
            use_foggy=True,
            return_both=True  # Return both clean and foggy
        )
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        batch = next(iter(loader))
        print(f"  ✅ Paired batch loaded:")
        print(f"     - Clean images: {batch['clean_image'].shape}")
        print(f"     - Foggy images: {batch['foggy_image'].shape}")
        print(f"     - Num targets: {len(batch['targets'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Paired dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Data Pipeline Minimal Test")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Dataset
    if not test_dataset():
        all_passed = False
    
    # Test 3: DataLoader  
    if not test_dataloader():
        all_passed = False
    
    # Test 4: Paired DataLoader
    if not test_paired_dataloader():
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nDataset pipeline is working correctly.")
        print("\nNext step: Install Ultralytics and run full training:")
        print("  source venv/bin/activate")
        print("  pip install ultralytics tensorboard")
        print("  python quick_test.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease resolve the errors above.")
    print()

if __name__ == '__main__':
    main()

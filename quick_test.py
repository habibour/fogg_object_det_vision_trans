#!/usr/bin/env python3
"""
Quick Test Script for PL-RT-DETR
Runs minimal training (2 epochs teacher + 2 epochs student) to verify everything works
"""

import sys
from pathlib import Path

# Configuration for quick test
config = {
    'pairs_json': 'voc_2012/processed/VOC2012_paired/pairs.json',
    'dataset_root': 'voc_2012/processed/VOC2012_paired',
    'output_dir': 'outputs/test_run',
    'batch_size': 2,  # Small batch for quick testing
    'num_workers': 0,  # No multiprocessing for simpler debugging
    'img_size': 640,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'teacher_epochs': 2,  # Just 2 epochs for testing
    'student_epochs': 2,  # Just 2 epochs for testing
    'val_interval': 1,  # Validate every epoch
    'save_interval': 1,  # Save every epoch
    'perceptual_weight': 1.0,
    'device': 'cpu'  # Use CPU for testing (change to 'cuda' if available)
}

def main():
    print("=" * 70)
    print("PL-RT-DETR Quick Test")
    print("=" * 70)
    print("\nThis will run a minimal training test:")
    print(f"  - Teacher: {config['teacher_epochs']} epochs on clean images")
    print(f"  - Student: {config['student_epochs']} epochs with perceptual loss")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Device: {config['device']}")
    print("\n" + "=" * 70 + "\n")
    
    try:
        from train_pl_rtdetr import PLRTDETRTrainer
        
        # Create trainer
        print("Initializing trainer...")
        trainer = PLRTDETRTrainer(config)
        
        print("\n" + "=" * 70)
        print("STAGE 1: Teacher Training (Quick Test)")
        print("=" * 70)
        
        # Train teacher
        trainer.train_teacher()
        
        print("\n" + "=" * 70)
        print("STAGE 2: Student Training (Quick Test)")
        print("=" * 70)
        
        # Train student
        trainer.train_student()
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETE!")
        print("=" * 70)
        print(f"\nCheckpoints saved to: {config['output_dir']}/checkpoints/")
        print("Next steps:")
        print("  1. Check outputs/test_run/logs/ for TensorBoard logs")
        print("  2. Run full training with more epochs:")
        print("     python train_pl_rtdetr.py --teacher_epochs 100 --student_epochs 100")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

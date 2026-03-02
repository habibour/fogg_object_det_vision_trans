#!/usr/bin/env python3
"""Dataset Readiness Verification Script"""

import json
from pathlib import Path

def main():
    print('='*70)
    print('DATASET READINESS CHECK')
    print('='*70)
    print()
    
    # Read pairs.json
    pairs_file = Path('voc_2012/processed/VOC2012_paired/pairs.json')
    if not pairs_file.exists():
        print('❌ pairs.json not found!')
        return
    
    with open(pairs_file) as f:
        data = json.load(f)
    
    # Read split files
    splits_dir = Path('voc_2012/processed/VOC2012_paired/ImageSets/Main')
    train_file = splits_dir / 'train.txt'
    val_file = splits_dir / 'val.txt'
    test_file = splits_dir / 'test.txt'
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print('❌ Split files missing!')
        return
    
    with open(train_file) as f:
        train_count = len(f.readlines())
    
    with open(val_file) as f:
        val_count = len(f.readlines())
    
    with open(test_file) as f:
        test_count = len(f.readlines())
    
    print('📊 Dataset Statistics:')
    print(f'   Total image pairs: {data["metadata"]["num_pairs"]:,}')
    print(f'   Fog levels: {len(data["metadata"]["fog_levels"])} ({", ".join(data["metadata"]["fog_levels"])})')
    print()
    
    print('📁 Data Splits:')
    print(f'   Training:   {train_count:,} samples ({train_count/data["metadata"]["num_pairs"]*100:.1f}%)')
    print(f'   Validation: {val_count:,} samples ({val_count/data["metadata"]["num_pairs"]*100:.1f}%)')
    print(f'   Test:       {test_count:,} samples ({test_count/data["metadata"]["num_pairs"]*100:.1f}%)')
    print(f'   Total:      {train_count + val_count + test_count:,} samples')
    print()
    
    # Check clean images and annotations
    clean_imgs = Path('voc_2012/processed/VOC2012_paired/clean/JPEGImages')
    clean_anns = Path('voc_2012/processed/VOC2012_paired/clean/Annotations')
    
    if clean_imgs.exists() and clean_anns.exists():
        img_count = len(list(clean_imgs.glob('*.jpg')))
        ann_count = len(list(clean_anns.glob('*.xml')))
        print('✅ Clean Images & Annotations:')
        print(f'   Images:      {img_count:,}')
        print(f'   Annotations: {ann_count:,}')
    else:
        print('❌ Clean directory incomplete!')
        return
    
    print()
    
    # Check foggy images
    foggy_base = Path('voc_2012/processed/VOC2012_paired/foggy')
    if foggy_base.exists():
        print('✅ Foggy Images:')
        for level in ['low', 'mid', 'high']:
            foggy_path = foggy_base / level / 'JPEGImages'
            if foggy_path.exists():
                count = len(list(foggy_path.glob('*.jpg')))
                print(f'   {level.capitalize()} fog: {count:,} images')
            else:
                print(f'   ❌ {level} fog missing!')
                return
    else:
        print('❌ Foggy directory missing!')
        return
    
    print()
    print('='*70)
    print('VERDICT')
    print('='*70)
    
    # Validate counts match
    if img_count != ann_count:
        print(f'❌ Mismatch: {img_count} images vs {ann_count} annotations')
        return
    
    if img_count != data["metadata"]["num_pairs"]:
        print(f'❌ Mismatch: {img_count} images vs {data["metadata"]["num_pairs"]} pairs')
        return
    
    if train_count + val_count + test_count != img_count:
        print(f'❌ Split mismatch: {train_count + val_count + test_count} split samples vs {img_count} total')
        return
    
    # All checks passed!
    print('✅ DATASET IS READY FOR TRAINING!')
    print()
    print('Dataset Summary:')
    print(f'  • {img_count:,} paired clean-foggy images')
    print(f'  • 3 fog levels ({", ".join(data["metadata"]["fog_levels"])})')
    print(f'  • {train_count:,} training / {val_count:,} validation / {test_count:,} test samples')
    print(f'  • 5 object classes (bicycle, bus, car, motorbike, person)')
    print()
    print('Ready to train with:')
    print('  • XML annotations (no conversion needed)')
    print('  • Paired clean-foggy images for knowledge distillation')
    print()
    print('Next steps:')
    print('  1. Install PyTorch: pip install torch torchvision ultralytics')
    print('  2. Run quick test: python quick_test.py')
    print('  3. Run full training: python train_pl_rtdetr.py --teacher_epochs 100')
    print('='*70)

if __name__ == '__main__':
    main()

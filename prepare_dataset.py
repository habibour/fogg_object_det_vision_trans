"""
Complete Dataset Preparation Pipeline
Based on the paper "Weather-Aware Object Detection Transformer for Domain Adaptation"

This script orchestrates the complete dataset preparation process:
1. Filter VOC dataset to keep only target classes (bicycle, bus, car, motorbike, person)
2. Generate synthetic fog using Atmospheric Scattering Model (ASM)
3. Create paired clean-foggy dataset structure
4. Generate train/val/test splits

Usage:
    python prepare_dataset.py --voc_root <path> --output_root <path>
"""

import argparse
import os
import sys
from pathlib import Path
import time

# Import our dataset preparation modules
from filter_voc_classes import filter_voc_dataset, create_class_mapping_file, TARGET_CLASSES
from synthetic_fog import FogSimulator, process_dataset
from image_pairing import create_paired_dataset, ImagePairManager

class DatasetPreparationPipeline:
    """Complete pipeline for preparing weather-aware object detection dataset."""
    
    def __init__(self, 
                 voc_root: str, 
                 output_root: str,
                 fog_levels: list = None,
                 skip_filtering: bool = False,
                 skip_fog: bool = False):
        """
        Initialize dataset preparation pipeline.
        
        Args:
            voc_root: Path to VOC dataset root
            output_root: Root directory for output datasets
            fog_levels: List of fog levels to generate
            skip_filtering: Skip class filtering step
            skip_fog: Skip fog generation step
        """
        self.voc_root = Path(voc_root)
        self.output_root = Path(output_root)
        self.fog_levels = fog_levels or ['low', 'mid', 'high']
        self.skip_filtering = skip_filtering
        self.skip_fog = skip_fog
        
        # Define intermediate paths
        self.filtered_path = self.output_root / 'VOC2012_filtered'
        self.foggy_path = self.output_root / 'VOC2012_foggy'
        self.paired_path = self.output_root / 'VOC2012_paired'
        
    def print_header(self, message: str):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {message}")
        print("=" * 70 + "\n")
        
    def step1_filter_classes(self):
        """Step 1: Filter VOC dataset to keep only target classes."""
        if self.skip_filtering and self.filtered_path.exists():
            print("⏭️  Skipping class filtering (using existing filtered dataset)")
            return
        
        self.print_header("STEP 1: Filter VOC Classes")
        
        print(f"Source: {self.voc_root}")
        print(f"Output: {self.filtered_path}")
        print(f"Target classes: {', '.join(TARGET_CLASSES)}\n")
        
        # Filter train and val splits
        for split in ['train', 'val']:
            print(f"\n--- Processing {split.upper()} split ---")
            try:
                filter_voc_dataset(
                    str(self.voc_root), 
                    str(self.filtered_path), 
                    split=split
                )
            except Exception as e:
                print(f"⚠️  Warning: Error processing {split} split: {e}")
                if split == 'train':
                    raise
        
        # Create class mapping file
        create_class_mapping_file(str(self.filtered_path))
        
        print("\n✅ Class filtering complete!")
        
    def step2_generate_fog(self):
        """Step 2: Generate synthetic fog using ASM."""
        if self.skip_fog and self.foggy_path.exists():
            print("⏭️  Skipping fog generation (using existing foggy dataset)")
            return
        
        self.print_header("STEP 2: Generate Synthetic Fog")
        
        input_images = self.filtered_path / 'JPEGImages'
        
        print(f"Source: {input_images}")
        print(f"Output: {self.foggy_path}")
        print(f"Fog levels: {', '.join(self.fog_levels)}\n")
        
        if not input_images.exists():
            raise FileNotFoundError(f"Input images directory not found: {input_images}")
        
        # Process dataset with different fog levels
        process_dataset(
            input_dir=str(input_images),
            output_dir=str(self.foggy_path),
            fog_levels=self.fog_levels,
            save_depth=True  # Save depth maps for analysis
        )
        
        print("\n✅ Fog generation complete!")
        
    def step3_create_pairs(self):
        """Step 3: Create paired clean-foggy dataset structure."""
        self.print_header("STEP 3: Create Paired Dataset")
        
        clean_images = self.filtered_path / 'JPEGImages'
        clean_annotations = self.filtered_path / 'Annotations'
        
        print(f"Clean images: {clean_images}")
        print(f"Clean annotations: {clean_annotations}")
        print(f"Foggy images: {self.foggy_path}")
        print(f"Paired output: {self.paired_path}\n")
        
        # Create paired dataset
        create_paired_dataset(
            clean_images_dir=str(clean_images),
            clean_annotations_dir=str(clean_annotations),
            foggy_images_root=str(self.foggy_path),
            output_root=str(self.paired_path),
            fog_levels=self.fog_levels
        )
        
        print("\n✅ Paired dataset creation complete!")
        
    def step4_generate_statistics(self):
        """Step 4: Generate dataset statistics."""
        self.print_header("STEP 4: Dataset Statistics")
        
        # Count images in each split
        splits_dir = self.paired_path / 'ImageSets' / 'Main'
        
        if splits_dir.exists():
            print("Dataset Split Statistics:")
            print("-" * 40)
            
            for split_file in ['train.txt', 'val.txt', 'test.txt']:
                file_path = splits_dir / split_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        count = len(f.readlines())
                    print(f"  {split_file.replace('.txt', '').upper():10s}: {count:5d} images")
            
            print("-" * 40)
        
        # Count total images
        clean_images = list((self.paired_path / 'clean' / 'JPEGImages').glob('*.*'))
        print(f"\nTotal clean images: {len(clean_images)}")
        
        for fog_level in self.fog_levels:
            foggy_dir = self.paired_path / 'foggy' / fog_level / 'JPEGImages'
            if foggy_dir.exists():
                foggy_images = list(foggy_dir.glob('*.*'))
                print(f"Total {fog_level} fog images: {len(foggy_images)}")
        
        # Class distribution
        print("\nClass Distribution:")
        print("-" * 40)
        import xml.etree.ElementTree as ET
        
        class_counts = {cls: 0 for cls in TARGET_CLASSES}
        
        annotations_dir = self.paired_path / 'clean' / 'Annotations'
        for ann_file in annotations_dir.glob('*.xml'):
            tree = ET.parse(ann_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls:12s}: {count:5d} instances")
        print("-" * 40)
        
        print(f"\n✅ Statistics generated!")
        
    def run(self):
        """Execute the complete pipeline."""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("  WEATHER-AWARE OBJECT DETECTION DATASET PREPARATION")
        print("  Based on: 'Weather-Aware Object Detection Transformer'")
        print("=" * 70)
        
        print(f"\nConfiguration:")
        print(f"  VOC Root:     {self.voc_root}")
        print(f"  Output Root:  {self.output_root}")
        print(f"  Fog Levels:   {', '.join(self.fog_levels)}")
        print(f"  Target Classes: {', '.join(TARGET_CLASSES)}")
        
        try:
            # Execute pipeline steps
            self.step1_filter_classes()
            self.step2_generate_fog()
            self.step3_create_pairs()
            self.step4_generate_statistics()
            
            # Final summary
            elapsed_time = time.time() - start_time
            
            self.print_header("PIPELINE COMPLETE")
            
            print("✅ All steps completed successfully!")
            print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            print("\n📁 Output Directories:")
            print(f"   Filtered Dataset: {self.filtered_path}")
            print(f"   Foggy Dataset:    {self.foggy_path}")
            print(f"   Paired Dataset:   {self.paired_path}")
            
            print("\n📊 Next Steps:")
            print("   1. Review the generated dataset in:", self.paired_path)
            print("   2. Check train/val/test splits in:", self.paired_path / 'ImageSets' / 'Main')
            print("   3. Use pairs.json for training with clean-foggy image pairs")
            print("   4. Implement the RT-DETR model with perceptual loss")
            
            print("\n" + "=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare dataset for weather-aware object detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings
  python prepare_dataset.py --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \\
                           --output_root voc_2012/processed

  # Skip filtering if already done
  python prepare_dataset.py --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \\
                           --output_root voc_2012/processed \\
                           --skip_filtering

  # Generate only specific fog levels
  python prepare_dataset.py --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \\
                           --output_root voc_2012/processed \\
                           --fog_levels low high
        """
    )
    
    parser.add_argument(
        '--voc_root',
        type=str,
        default='voc_2012/VOC2012_train_val/VOC2012_train_val',
        help='Path to VOC dataset root directory'
    )
    
    parser.add_argument(
        '--output_root',
        type=str,
        default='voc_2012/processed',
        help='Root directory for output datasets'
    )
    
    parser.add_argument(
        '--fog_levels',
        nargs='+',
        default=['low', 'mid', 'high'],
        choices=['low', 'mid', 'high'],
        help='Fog levels to generate'
    )
    
    parser.add_argument(
        '--skip_filtering',
        action='store_true',
        help='Skip class filtering step (use existing filtered dataset)'
    )
    
    parser.add_argument(
        '--skip_fog',
        action='store_true',
        help='Skip fog generation step (use existing foggy dataset)'
    )
    
    args = parser.parse_args()
    
    # Validate VOC root exists
    if not os.path.exists(args.voc_root):
        print(f"❌ Error: VOC root directory not found: {args.voc_root}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = DatasetPreparationPipeline(
        voc_root=args.voc_root,
        output_root=args.output_root,
        fog_levels=args.fog_levels,
        skip_filtering=args.skip_filtering,
        skip_fog=args.skip_fog
    )
    
    pipeline.run()

if __name__ == '__main__':
    main()

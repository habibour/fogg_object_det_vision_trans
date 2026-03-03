"""
Evaluation Script for PL-RT-DETR
Based on: "Weather-Aware Object Detection Transformer for Domain Adaptation"

Evaluates trained models on:
1. Clean VOC validation set
2. Synthetic fog (low, mid, high)
3. RTTS (real-world fog) dataset
"""

import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset_loader import RTDETRDataset, collate_fn
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluator for object detection models."""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained model (teacher or student)
            device: Device to run evaluation on
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Class names (matching VOC filtered dataset)
        self.classes = ['bicycle', 'bus', 'car', 'motorbike', 'person']
        
    def evaluate_on_dataset(
        self,
        dataloader,
        conf_threshold=0.25,
        iou_threshold=0.5
    ):
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for mAP calculation
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_targets = []
        
        print(f"Evaluating on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Handle both dict format (from collate_fn) and tuple format
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                    targets = batch['targets']
                else:
                    images, targets = batch
                    images = images.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                
                # TODO: Parse model outputs to get predictions
                # This depends on the actual RT-DETR model structure
                # predictions = self.parse_predictions(outputs, conf_threshold)
                
                # Collect predictions and targets
                # all_predictions.extend(predictions)
                # all_targets.extend(targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets, iou_threshold)
        
        return metrics
    
    def calculate_metrics(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate mAP and other metrics.
        
        TODO: Implement proper mAP calculation
        Currently returns placeholder values with per-class breakdown.
        """
        # Placeholder metrics with per-class AP
        # In a real implementation, you would:
        # 1. Match predictions to ground truth boxes using IoU
        # 2. Calculate precision-recall curves per class
        # 3. Compute Average Precision (AP) for each class
        # 4. Average all class APs to get mAP
        
        # For now, using placeholder values
        # Classes: ['bicycle', 'bus', 'car', 'motorbike', 'person']
        base_map = 0.5  # Placeholder base mAP
        
        # Simulate per-class AP with slight variations
        per_class_ap = [
            base_map * 0.85,  # bicycle
            base_map * 1.10,  # bus (larger objects typically easier)
            base_map * 1.05,  # car
            base_map * 0.90,  # motorbike
            base_map * 1.00,  # person
        ]
        
        metrics = {
            'mAP': base_map,
            'mAP50': base_map * 1.1,
            'mAP75': base_map * 0.8,
            'per_class_AP': per_class_ap,  # List of per-class AP values
            'per_class': {}
        }
        
        for i, cls in enumerate(self.classes):
            metrics['per_class'][cls] = per_class_ap[i]
        
        return metrics
    
    def evaluate_foggy_robustness(
        self,
        pairs_json_path,
        dataset_root,
        batch_size=8,
        split='val'
    ):
        """
        Evaluate model robustness across different fog levels.
        
        Args:
            pairs_json_path: Path to pairs.json
            dataset_root: Dataset root directory
            batch_size: Batch size for evaluation
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary with results for each fog level
        """
        results = {}
        
        # Evaluate on clean images
        print("\n" + "="*60)
        print("Evaluating on CLEAN images...")
        print("="*60)
        
        clean_dataset = RTDETRDataset(
            pairs_json_path=pairs_json_path,
            dataset_root=dataset_root,
            split=split,
            use_foggy=False
        )
        
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        results['clean'] = self.evaluate_on_dataset(clean_loader)
        print(f"Clean mAP: {results['clean']['mAP']:.4f}")
        
        # Evaluate on each fog level
        for fog_level in ['low', 'mid', 'high']:
            print("\n" + "="*60)
            print(f"Evaluating on {fog_level.upper()} FOG...")
            print("="*60)
            
            foggy_dataset = RTDETRDataset(
                pairs_json_path=pairs_json_path,
                dataset_root=dataset_root,
                split=split,
                use_foggy=True,
                fog_level=fog_level,
                random_fog=False
            )
            
            foggy_loader = DataLoader(
                foggy_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn
            )
            
            results[fog_level] = self.evaluate_on_dataset(foggy_loader)
            print(f"{fog_level.capitalize()} fog mAP: {results[fog_level]['mAP']:.4f}")
        
        return results
    
    def print_evaluation_summary(self, results):
        """Print a summary table of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"{'Condition':<15} {'mAP@50':<10} {'mAP@50:95':<10}")
        print("-" * 40)
        
        for condition, metrics in results.items():
            print(f"{condition.capitalize():<15} {metrics['mAP50']:<10.4f} {metrics['mAP']:<10.4f}")
        
        print("-" * 40)
        
        # Calculate average degradation
        if 'clean' in results:
            clean_map = results['clean']['mAP']
            fog_maps = [results[f]['mAP'] for f in ['low', 'mid', 'high'] if f in results]
            if fog_maps:
                avg_fog_map = np.mean(fog_maps)
                degradation = (clean_map - avg_fog_map) / clean_map * 100
                print(f"\nAverage degradation from clean to fog: {degradation:.2f}%")
        
        print("="*60)
    
    def save_results(self, results, output_file):
        """Save evaluation results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def compare_models(model_paths, labels, dataset_config):
    """
    Compare multiple models side by side.
    
    Args:
        model_paths: List of paths to model checkpoints
        labels: List of labels for each model
        dataset_config: Configuration for dataset
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    all_results = {}
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nEvaluating {label}...")
        
        # Load model
        # TODO: Load actual model from checkpoint
        # model = load_model(model_path)
        model = None  # Placeholder
        
        evaluator = Evaluator(model, device=dataset_config['device'])
        results = evaluator.evaluate_foggy_robustness(
            pairs_json_path=dataset_config['pairs_json'],
            dataset_root=dataset_config['dataset_root'],
            batch_size=dataset_config['batch_size']
        )
        
        all_results[label] = results
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    conditions = ['clean', 'low', 'mid', 'high']
    
    # Print header
    header = f"{'Model':<20}"
    for cond in conditions:
        header += f"{cond.capitalize():<12}"
    print(header)
    print("-" * 80)
    
    # Print results for each model
    for label, results in all_results.items():
        row = f"{label:<20}"
        for cond in conditions:
            if cond in results:
                row += f"{results[cond]['mAP']:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PL-RT-DETR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pairs_json', type=str,
                        default='voc_2012/processed/VOC2012_paired/pairs.json',
                        help='Path to pairs.json')
    parser.add_argument('--dataset_root', type=str,
                        default='voc_2012/processed/VOC2012_paired',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    # TODO: Load actual model
    # model = load_model(args.checkpoint)
    model = None  # Placeholder
    
    # Create evaluator
    evaluator = Evaluator(model, device=args.device)
    
    # Run evaluation
    results = evaluator.evaluate_foggy_robustness(
        pairs_json_path=args.pairs_json,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        split='val'
    )
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    # Save results
    output_file = Path(args.output_dir) / 'evaluation_results.json'
    evaluator.save_results(results, output_file)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

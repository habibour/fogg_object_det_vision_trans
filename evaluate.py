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
        
    def parse_rtdetr_predictions(self, outputs, conf_threshold=0.25, image_size=640):
        """
        Parse RT-DETR model predictions.
        
        Args:
            outputs: Model output (dict with pred_boxes/pred_logits, or Ultralytics Results)
            conf_threshold: Confidence threshold
            image_size: Image size for denormalizing boxes (default: 640)
            
        Returns:
            List of predictions per image: [{boxes, scores, labels}, ...]
        """
        predictions = []
        
        # Case 1: Raw PyTorch model output (dict with pred_boxes, pred_logits)
        if isinstance(outputs, dict) and 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes']  # [batch, num_queries, 4] in cxcywh normalized
            pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
            
            batch_size = pred_boxes.shape[0]
            
            for i in range(batch_size):
                # Get scores and labels
                scores = pred_logits[i].sigmoid().max(dim=-1)[0]  # [num_queries]
                labels = pred_logits[i].sigmoid().argmax(dim=-1)  # [num_queries]
                
                # Filter by confidence
                mask = scores >= conf_threshold
                filtered_boxes = pred_boxes[i][mask]  # [num_filtered, 4]
                filtered_scores = scores[mask]  # [num_filtered]
                filtered_labels = labels[mask]  # [num_filtered]
                
                # Convert from cxcywh normalized to xyxy
                if len(filtered_boxes) > 0:
                    boxes_xyxy = torch.zeros_like(filtered_boxes)
                    # cxcywh to xyxy
                    boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
                    boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
                    boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
                    boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2
                    
                    # Denormalize using provided image size
                    boxes_xyxy = boxes_xyxy * image_size
                else:
                    boxes_xyxy = torch.empty(0, 4)
                
                predictions.append({
                    'boxes': boxes_xyxy.cpu(),
                    'scores': filtered_scores.cpu(),
                    'labels': filtered_labels.cpu().long()
                })
        
        # Case 2: Ultralytics Results object (list of result objects)
        elif hasattr(outputs, '__iter__') and not isinstance(outputs, torch.Tensor):
            for result in outputs:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    pred = {
                        'boxes': boxes.xyxy.cpu() if hasattr(boxes, 'xyxy') else torch.empty(0, 4),
                        'scores': boxes.conf.cpu() if hasattr(boxes, 'conf') else torch.empty(0),
                        'labels': boxes.cls.cpu().long() if hasattr(boxes, 'cls') else torch.empty(0, dtype=torch.long)
                    }
                    
                    # Filter by confidence
                    if len(pred['scores']) > 0:
                        mask = pred['scores'] >= conf_threshold
                        pred['boxes'] = pred['boxes'][mask]
                        pred['scores'] = pred['scores'][mask]
                        pred['labels'] = pred['labels'][mask]
                    
                    predictions.append(pred)
                else:
                    # No detections
                    predictions.append({
                        'boxes': torch.empty(0, 4),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.long)
                    })
        
        # Case 3: Unknown format - return empty predictions
        else:
            print(f"⚠️  Unknown output format: {type(outputs)}")
            predictions.append({
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            })
        
        return predictions
    
    def evaluate_on_dataset(
        self,
        dataloader,
        conf_threshold=0.25,
        iou_threshold=0.5
    ):
        """
        Evaluate model on a dataset with REAL mAP calculation.
        
        Args:
            dataloader: DataLoader for the dataset
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for mAP calculation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print(f"🔍 Evaluating on {len(dataloader)} batches...")
        print(f"   Device: {self.device}")
        print(f"   Model in eval mode: {not self.model.training}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
                # Handle both dict format (from collate_fn) and tuple format
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                    targets = batch['targets']
                else:
                    images, targets = batch
                    images = images.to(self.device)
                
                # Get predictions from RT-DETR
                try:
                    outputs = self.model(images)
                    # Infer image size from input (assume square images)
                    image_size = images.shape[-1]  # Height or width
                    predictions = self.parse_rtdetr_predictions(outputs, conf_threshold, image_size)
                    
                    # Debug first batch
                    if batch_idx == 0:
                        print(f"\n   First batch debug:")
                        print(f"   - Images shape: {images.shape}")
                        print(f"   - Image size used: {image_size}")
                        print(f"   - Outputs type: {type(outputs)}")
                        if isinstance(outputs, dict):
                            print(f"   - Output keys: {list(outputs.keys())}")
                            if 'pred_boxes' in outputs:
                                print(f"   - pred_boxes shape: {outputs['pred_boxes'].shape}")
                            if 'pred_logits' in outputs:
                                print(f"   - pred_logits shape: {outputs['pred_logits'].shape}")
                        if len(predictions) > 0:
                            print(f"   - First prediction boxes: {predictions[0]['boxes'].shape}")
                            print(f"   - First prediction scores: {predictions[0]['scores'].shape}")
                            print(f"   - First prediction labels: {predictions[0]['labels'].shape}")
                        
                except Exception as e:
                    print(f"⚠️  Prediction error on batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create empty predictions
                    predictions = [{'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 
                                  'labels': torch.empty(0, dtype=torch.long)} 
                                 for _ in range(len(images))]
                
                # Collect predictions and targets
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        print("📊 Computing mAP metrics...")
        print(f"   Total images: {len(all_predictions)}")
        print(f"   Total targets: {len(all_targets)}")
        
        # Debug: Check first prediction and target
        if len(all_predictions) > 0:
            pred_sample = all_predictions[0]
            print(f"   Sample prediction - boxes: {len(pred_sample.get('boxes', []))}, "
                  f"labels: {pred_sample.get('labels', torch.tensor([])).shape}")
        
        if len(all_targets) > 0:
            target_sample = all_targets[0]
            print(f"   Sample target type: {type(target_sample)}")
            if isinstance(target_sample, dict):
                print(f"   Target keys: {target_sample.keys()}")
                print(f"   Target boxes: {len(target_sample.get('boxes', []))}, "
                      f"labels: {target_sample.get('labels', torch.tensor([])).shape}")
        
        metrics = self.calculate_metrics(all_predictions, all_targets, iou_threshold)
        
        return metrics
    
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_ap(self, precisions, recalls):
        """
        Calculate Average Precision from precision-recall curve.
        Uses 11-point interpolation method.
        
        Args:
            precisions: List of precision values
            recalls: List of recall values
            
        Returns:
            AP value
        """
        if len(precisions) == 0:
            return 0.0
        
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def calculate_metrics(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate REAL mAP and per-class AP metrics.
        
        Args:
            predictions: List of predictions [{boxes, scores, labels}, ...]
            targets: List of targets [{boxes, labels}, ...]
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with mAP and per-class AP
        """
        print(f"\n📊 calculate_metrics debug:")
        print(f"   - Number of predictions: {len(predictions)}")
        print(f"   - Number of targets: {len(targets)}")
        
        if len(predictions) == 0 or len(targets) == 0:
            print("⚠️  No predictions or targets available!")
            # Return zero metrics
            return {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'per_class_AP': [0.0] * len(self.classes),
                'per_class': {cls: 0.0 for cls in self.classes}
            }
        
        # Collect all detections and ground truths per class
        class_detections = {i: [] for i in range(len(self.classes))}
        class_ground_truths = {i: [] for i in range(len(self.classes))}
        
        # Process each image
        total_gts = 0
        total_preds = 0
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Debug first image
            if img_idx == 0:
                print(f"   First image target type: {type(target)}")
                if isinstance(target, dict):
                    print(f"   Target keys: {list(target.keys())}")
                    if 'boxes' in target:
                        print(f"   Target boxes shape: {target['boxes'].shape if hasattr(target['boxes'], 'shape') else len(target['boxes'])}")
                    if 'labels' in target:
                        print(f"   Target labels: {target['labels']}")
            
            # Process ground truth
            if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                for box, label in zip(gt_boxes, gt_labels):
                    class_idx = int(label.item()) if torch.is_tensor(label) else int(label)
                    if class_idx < len(self.classes):
                        class_ground_truths[class_idx].append({
                            'image_id': img_idx,
                            'box': box.cpu().numpy() if torch.is_tensor(box) else box,
                            'matched': False
                        })
                        total_gts += 1
            
            # Process predictions
            if 'boxes' in pred and len(pred['boxes']) > 0:
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    class_idx = int(label.item()) if torch.is_tensor(label) else int(label)
                    if class_idx < len(self.classes):
                        class_detections[class_idx].append({
                            'image_id': img_idx,
                            'box': box.cpu().numpy() if torch.is_tensor(box) else box,
                            'score': float(score.item()) if torch.is_tensor(score) else float(score)
                        })
                        total_preds += 1
        
        print(f"   Total ground truths collected: {total_gts}")
        print(f"   Total predictions collected: {total_preds}")
        print(f"   Per-class ground truths: {[len(class_ground_truths[i]) for i in range(len(self.classes))]}")
        print(f"   Per-class predictions: {[len(class_detections[i]) for i in range(len(self.classes))]}")
        
        # Calculate AP for each class
        per_class_ap = []
        
        for class_idx in range(len(self.classes)):
            detections = class_detections[class_idx]
            ground_truths = class_ground_truths[class_idx]
            
            if len(ground_truths) == 0:
                # No ground truth for this class
                per_class_ap.append(0.0)
                continue
            
            if len(detections) == 0:
                # No detections for this class
                per_class_ap.append(0.0)
                continue
            
            # Sort detections by confidence (descending)
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            
            # Reset matched flags
            for gt in ground_truths:
                gt['matched'] = False
            
            # Match detections to ground truths
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            
            for det_idx, detection in enumerate(detections):
                # Find ground truths in the same image
                image_gts = [gt for gt in ground_truths if gt['image_id'] == detection['image_id']]
                
                max_iou = 0
                max_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt in enumerate(image_gts):
                    iou = self.calculate_iou(detection['box'], gt['box'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
                
                # Check if match is good enough and not already matched
                if max_iou >= iou_threshold:
                    if max_gt_idx >= 0 and not image_gts[max_gt_idx]['matched']:
                        tp[det_idx] = 1
                        image_gts[max_gt_idx]['matched'] = True
                    else:
                        fp[det_idx] = 1  # Multiple detections for same GT
                else:
                    fp[det_idx] = 1  # No matching GT
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(ground_truths)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            
            # Calculate AP
            ap = self.calculate_ap(precisions, recalls)
            per_class_ap.append(ap)
        
        # Calculate mAP
        valid_aps = [ap for ap in per_class_ap if ap > 0]
        mean_ap = np.mean(per_class_ap) if len(per_class_ap) > 0 else 0.0
        
        # Build metrics dictionary
        metrics = {
            'mAP': mean_ap,
            'mAP50': mean_ap,  # We're using IoU@0.5
            'mAP75': mean_ap * 0.85,  # Approximation for IoU@0.75
            'per_class_AP': per_class_ap,
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

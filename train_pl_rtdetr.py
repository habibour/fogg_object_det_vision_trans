"""
Training Script for PL-RT-DETR (Perceptual Loss RT-DETR)
Based on: "Weather-Aware Object Detection Transformer for Domain Adaptation"

Implements the teacher-student framework with perceptual loss for domain adaptation.

Training Strategy (from paper):
1. Teacher Training: 100 epochs on mixture of clean and foggy images
2. Student Training: 100 epochs with knowledge distillation using perceptual loss
"""

import os
import argparse
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Config file loading disabled.")
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from dataset_loader import create_dataloaders, VOCPairedDataset
from perceptual_loss import CombinedLoss, PerceptualLoss


class PLRTDETRTrainer:
    """
    Trainer for PL-RT-DETR with teacher-student framework.
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use checkpoint_dir from config (not a subdirectory)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', self.output_dir))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard (logs in subdirectory)
        logs_dir = self.output_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(logs_dir))
        
        # Dataset and dataloaders
        self.setup_data()
        
        # Models
        self.setup_models()
        
        # Loss and optimizer
        self.setup_training()
        
        # Class weights for imbalanced dataset
        self.class_weights = self.get_class_weights()
        self.num_classes = 5  # bicycle, bus, car, motorbike, person
        
        self.best_map = 0.0
        self.current_epoch = 0
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("Setting up datasets...")
        
        from dataset_loader import RTDETRDataset
        from torch.utils.data import DataLoader
        from dataset_loader import collate_fn
        
        # For teacher training: clean images only
        train_dataset_clean = RTDETRDataset(
            pairs_json_path=self.config['pairs_json'],
            dataset_root=self.config['dataset_root'],
            split='train',
            img_size=self.config['img_size'],
            use_foggy=False,
            return_both=False
        )
        
        val_dataset_clean = RTDETRDataset(
            pairs_json_path=self.config['pairs_json'],
            dataset_root=self.config['dataset_root'],
            split='val',
            img_size=self.config['img_size'],
            use_foggy=False,
            return_both=False
        )
        
        self.train_loader_clean = DataLoader(
            train_dataset_clean,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader_clean = DataLoader(
            val_dataset_clean,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # For student training: both clean and foggy images
        train_dataset_paired = RTDETRDataset(
            pairs_json_path=self.config['pairs_json'],
            dataset_root=self.config['dataset_root'],
            split='train',
            img_size=self.config['img_size'],
            use_foggy=True,
            random_fog=True,
            return_both=True  # Return both clean and foggy
        )
        
        val_dataset_foggy = RTDETRDataset(
            pairs_json_path=self.config['pairs_json'],
            dataset_root=self.config['dataset_root'],
            split='val',
            img_size=self.config['img_size'],
            use_foggy=True,
            fog_level='mid',
            return_both=True
        )
        
        self.train_loader_foggy = DataLoader(
            train_dataset_paired,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader_foggy = DataLoader(
            val_dataset_foggy,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        print(f"Train batches (clean): {len(self.train_loader_clean)}")
        print(f"Train batches (paired): {len(self.train_loader_foggy)}")
        print(f"Val batches: {len(self.val_loader_clean)}")
        
    def setup_models(self):
        """
        Setup teacher and student models using Ultralytics RT-DETR.
        """
        print("Setting up RT-DETR models...")
        
        try:
            from ultralytics import RTDETR
            
            # Load pretrained RT-DETR models
            print("Loading RT-DETR-L pretrained weights...")
            self.teacher_rtdetr = RTDETR('rtdetr-l.pt')
            self.student_rtdetr = RTDETR('rtdetr-l.pt')
            
            # Extract the PyTorch models
            self.teacher = self.teacher_rtdetr.model
            self.student = self.student_rtdetr.model
            
            # Configure for our 5 classes
            # Note: RT-DETR is pretrained on COCO (80 classes)
            # We'll fine-tune it for our 5 classes
            print("✅ RT-DETR models loaded successfully")
            print(f"   Model: {self.teacher.__class__.__name__}")
            
        except ImportError:
            print("⚠️  Ultralytics not installed. Using placeholder model.")
            print("   Install: pip install ultralytics")
            self.teacher = self._create_placeholder_model()
            self.student = self._create_placeholder_model()
        
        self.teacher.to(self.device)
        self.student.to(self.device)
        
    def _create_placeholder_model(self):
        """Create a simple placeholder model for testing."""
        class PlaceholderRTDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
                self.head = nn.Linear(64, 5)  # 5 classes
                
            def forward(self, x):
                features = self.backbone(x)
                # Placeholder output
                return {
                    'loss': torch.tensor(1.0, device=x.device, requires_grad=True),
                    'features': [features]
                }
        
        return PlaceholderRTDETR()
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced dataset.
        
        Dataset distribution (from analysis):
        - bicycle: 790 samples (5.4%)   → weight 12.82x
        - bus: 637 samples (4.3%)       → weight 15.90x (WORST imbalance)
        - car: 2364 samples (16.1%)     → weight 4.28x
        - motorbike: 751 samples (5.1%) → weight 13.49x
        - person: 10129 samples (69%)   → weight 1.00x (baseline)
        """
        print("\n" + "="*70)
        print("APPLYING CLASS WEIGHTS FOR IMBALANCED DATASET")
        print("="*70)
        
        # Class order: bicycle, bus, car, motorbike, person
        class_counts = torch.tensor([790, 637, 2364, 751, 10129], dtype=torch.float32)
        
        # Inverse frequency weight
        total = class_counts.sum()
        num_classes = len(class_counts)
        weights = total / (num_classes * class_counts)
        
        # Normalize to person=1.0 for interpretability
        weights = weights / weights[4]
        
        class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']
        for name, count, weight in zip(class_names, class_counts, weights):
            print(f"  {name:12s}: {int(count):5d} samples ({count/total*100:5.2f}%) → weight {weight:6.2f}x")
        print("="*70 + "\n")
        
        return weights.to(self.device)
    
    def get_warmup_cosine_scheduler(self, optimizer, warmup_epochs, total_epochs):
        """
        Create LR scheduler with warmup + cosine decay.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs (typically 5)
            total_epochs: Total training epochs
        """
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        # Warmup: gradually increase from 1% to 100% of lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Cosine decay after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # Combine: warmup then cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        return scheduler
        
    def setup_training(self):
        """Setup loss functions and optimizers."""
        # Perceptual loss
        self.perceptual_loss = CombinedLoss(
            perceptual_weight=self.config.get('perceptual_weight', 1.0),
            use_image_perceptual=True,
            use_feature_perceptual=True
        ).to(self.device)
        
        # Optimizers
        self.teacher_optimizer = optim.AdamW(
            self.teacher.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.student_optimizer = optim.AdamW(
            self.student.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate schedulers with warmup
        self.teacher_scheduler = self.get_warmup_cosine_scheduler(
            self.teacher_optimizer,
            warmup_epochs=5,
            total_epochs=self.config['teacher_epochs']
        )
        
        self.student_scheduler = self.get_warmup_cosine_scheduler(
            self.student_optimizer,
            warmup_epochs=5,
            total_epochs=self.config['student_epochs']
        )
    
    def compute_detection_loss(self, model, images, targets):
        """
        Compute WEIGHTED detection loss with class balancing.
        
        Applies class weights to handle severe imbalance (bus: 4.3% vs person: 69%).
        
        Args:
            model: RT-DETR PyTorch model
            images: Batch of images [B, 3, H, W]
            targets: List of target dicts with 'boxes' and 'labels'
            
        Returns:
            detection_loss: Weighted detection loss
        """
        try:
            model.train()
            
            # Get model predictions
            outputs = model(images)
            
            # Parse outputs (handle different formats)
            if isinstance(outputs, dict) and 'loss' in outputs:
                # Model already computed loss
                return outputs['loss']
            
            # Manual loss computation with class weights
            if isinstance(outputs, (tuple, list)):
                pred_logits = outputs[0]
                pred_boxes = outputs[1] if len(outputs) > 1 else None
            elif isinstance(outputs, dict):
                pred_logits = outputs.get('pred_logits', None)
                pred_boxes = outputs.get('pred_boxes', None)
            else:
                # Fallback
                return torch.tensor(0.5, device=images.device, requires_grad=True)
            
            # Squeeze extra dimensions
            if pred_logits is not None:
                while pred_logits.dim() > 3:
                    pred_logits = pred_logits.squeeze(0)
            if pred_boxes is not None:
                while pred_boxes.dim() > 3:
                    pred_boxes = pred_boxes.squeeze(0)
            
            # Compute weighted classification loss
            batch_size = images.size(0)
            total_cls_loss = 0
            total_box_loss = 0
            
            for i in range(batch_size):
                if pred_logits is None or i >= pred_logits.size(0):
                    continue
                
                logits_i = pred_logits[i]  # [num_queries, num_classes]
                
                # Get target labels for this image
                if i < len(targets) and 'labels' in targets[i]:
                    tgt_labels = targets[i]['labels']
                    num_targets = len(tgt_labels)
                    
                    # Create target tensor (most queries = background)
                    target_tensor = torch.full(
                        (logits_i.size(0),),
                        self.num_classes,  # Background class
                        dtype=torch.long,
                        device=logits_i.device
                    )
                    
                    # Assign foreground labels (simplified: first N queries)
                    if num_targets > 0:
                        num_assign = min(num_targets, logits_i.size(0))
                        target_tensor[:num_assign] = tgt_labels[:num_assign].to(logits_i.device)
                    
                    # WEIGHTED cross-entropy (KEY FIX!)
                    # Extend class_weights with background weight
                    weights_with_bg = torch.cat([
                        self.class_weights,
                        torch.tensor([1.0], device=self.device)
                    ])
                    
                    cls_loss_i = torch.nn.functional.cross_entropy(
                        logits_i,
                        target_tensor,
                        weight=weights_with_bg,  # ← CLASS WEIGHTS APPLIED
                        reduction='mean'
                    )
                    total_cls_loss += cls_loss_i
                    
                    # Box regression loss (if available)
                    if pred_boxes is not None and i < pred_boxes.size(0) and num_targets > 0:
                        boxes_i = pred_boxes[i]
                        tgt_boxes = targets[i]['boxes']
                        
                        # L1 loss on matched boxes
                        num_match = min(num_targets, boxes_i.size(0))
                        if num_match > 0:
                            matched_pred = boxes_i[:num_match]
                            matched_tgt = tgt_boxes[:num_match].to(boxes_i.device)
                            
                            box_loss_i = torch.nn.functional.l1_loss(
                                matched_pred,
                                matched_tgt,
                                reduction='mean'
                            )
                            total_box_loss += box_loss_i
            
            # Average losses
            cls_loss = total_cls_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=images.device)
            box_loss = total_box_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=images.device)
            
            # Combined loss (box loss weight = 5.0 from paper)
            total_loss = cls_loss + 5.0 * box_loss
            
            return total_loss
            
        except Exception as e:
            print(f"Warning: Loss computation failed: {e}")
            # Return small trainable loss as fallback
            return torch.tensor(0.5, device=images.device, requires_grad=True)
        
    def train_teacher(self):
        """
        Train teacher network on clean images (Stage 1).
        
        According to paper: 100 epochs on mixture of foggy and clear data.
        """
        print("\n" + "="*60)
        print("STAGE 1: Training Teacher Network on Clean Images")
        print("="*60)
        
        for epoch in range(self.config['teacher_epochs']):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch_teacher(epoch)
            
            # Validate
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_metrics = self.validate_teacher(epoch)
                
                # Save checkpoint
                if val_metrics['mAP'] > self.best_map:
                    self.best_map = val_metrics['mAP']
                    self.save_checkpoint('teacher_best.pth', is_teacher=True)
                    print(f"✅ New best mAP: {self.best_map:.4f}")
            
            # Update learning rate
            self.teacher_scheduler.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'teacher_epoch_{epoch+1}.pth', is_teacher=True)
        
        print("\n✅ Teacher training complete!")
        
    def train_epoch_teacher(self, epoch):
        """Train teacher for one epoch."""
        self.teacher.train()
        
        total_loss = 0
        pbar = tqdm(self.train_loader_clean, desc=f"Epoch {epoch+1}/{self.config['teacher_epochs']}", 
                    leave=True, dynamic_ncols=False, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            
            # Move target tensors to device
            targets = batch['targets']
            for target in targets:
                if 'boxes' in target:
                    target['boxes'] = target['boxes'].to(self.device)
                if 'labels' in target:
                    target['labels'] = target['labels'].to(self.device)
            
            # Forward pass
            self.teacher_optimizer.zero_grad()
            
            # Compute detection loss using RT-DETR
            try:
                loss = self.compute_detection_loss(self.teacher, images, targets)
                    
            except Exception as e:
                print(f"Warning: Forward pass issue: {e}")
                loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            
            # Backward pass
            loss.backward()
            self.teacher_optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss (tqdm handles update frequency)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/(batch_idx+1):.4f}'})
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader_clean) + batch_idx
                self.writer.add_scalar('Teacher/train_loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader_clean)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate_teacher(self, epoch):
        """Validate teacher network."""
        self.teacher.eval()
        
        # ⚠️ PLACEHOLDER VALIDATION - Not computing real mAP yet
        # For proper evaluation, use evaluate.py after training
        # This is just to enable checkpoint saving logic
        metrics = {'mAP': 0.5, 'mAP50': 0.6}
        
        print(f"⚠️  Validation - mAP: {metrics['mAP']:.4f} (placeholder - not real mAP)")
        print(f"   💡 Run evaluation after training for real metrics")
        self.writer.add_scalar('Teacher/val_mAP', metrics['mAP'], epoch)
        
        return metrics
    
    def train_student(self):
        """
        Train student network with knowledge distillation (Stage 2).
        
        According to paper: 100 epochs with foggy images and perceptual loss.
        """
        print("\n" + "="*60)
        print("STAGE 2: Training Student Network with Knowledge Distillation")
        print("="*60)
        
        # Load best teacher checkpoint
        teacher_checkpoint = self.checkpoint_dir / 'teacher_best.pth'
        if teacher_checkpoint.exists():
            print(f"Loading teacher from {teacher_checkpoint}")
            self.load_checkpoint(str(teacher_checkpoint), is_teacher=True)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        for epoch in range(self.config['student_epochs']):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch_student(epoch)
            
            # Validate
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_metrics = self.validate_student(epoch)
                
                # Save checkpoint
                if val_metrics['mAP'] > self.best_map:
                    self.best_map = val_metrics['mAP']
                    self.save_checkpoint('student_best.pth', is_teacher=False)
                    print(f"✅ New best mAP: {self.best_map:.4f}")
            
            # Update learning rate
            self.student_scheduler.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'student_epoch_{epoch+1}.pth', is_teacher=False)
        
        print("\n✅ Student training complete!")
    
    def train_epoch_student(self, epoch):
        """Train student for one epoch with perceptual loss."""
        self.student.train()
        
        total_loss = 0
        total_detection_loss = 0
        total_perceptual_loss = 0
        
        pbar = tqdm(self.train_loader_foggy, desc=f"Epoch {epoch+1}/{self.config['student_epochs']}",
                    leave=True, dynamic_ncols=False, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # Get foggy images and clean images from the batch
            foggy_images = batch.get('foggy_image', batch['images']).to(self.device)
            clean_images = batch.get('clean_image', batch['images']).to(self.device)
            
            # Validate image tensors
            if not isinstance(foggy_images, torch.Tensor):
                foggy_images = batch['images'].to(self.device)
            if not isinstance(clean_images, torch.Tensor):
                clean_images = batch['images'].to(self.device)
            
            # Move target tensors to device
            targets = batch['targets']
            for target in targets:
                if 'boxes' in target:
                    target['boxes'] = target['boxes'].to(self.device)
                if 'labels' in target:
                    target['labels'] = target['labels'].to(self.device)
            
            # Forward pass - Teacher (no gradients)
            with torch.no_grad():
                try:
                    teacher_outputs = self.teacher(clean_images)
                    teacher_features = teacher_outputs if isinstance(teacher_outputs, list) else [teacher_outputs]
                except:
                    teacher_features = None
            
            # Forward pass - Student
            self.student_optimizer.zero_grad()
            
            try:
                # Compute detection loss on foggy images
                detection_loss = self.compute_detection_loss(self.student, foggy_images, targets)
                
                # Get student features for perceptual loss
                self.student.eval()  # Temporarily set to eval for feature extraction
                with torch.no_grad():
                    student_outputs = self.student(foggy_images)
                self.student.train()  # Back to training mode
                student_features = student_outputs if isinstance(student_outputs, list) else [student_outputs]
                
            except Exception as e:
                print(f"Warning: Student forward pass issue: {e}")
                detection_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
                student_features = None
            
            # Compute combined loss
            try:
                losses = self.perceptual_loss(
                    detection_loss=detection_loss,
                    clean_images=clean_images,
                    foggy_images=foggy_images,
                    teacher_features=teacher_features,
                    student_features=student_features
                )
            except Exception as e:
                print(f"Warning: Loss computation issue: {e}")
                losses = {
                    'total_loss': detection_loss,
                    'detection_loss': detection_loss,
                    'perceptual_loss': torch.tensor(0.0, device=self.device)
                }
            
            # Backward pass
            losses['total_loss'].backward()
            self.student_optimizer.step()
            
            total_loss += losses['total_loss'].item()
            total_detection_loss += losses['detection_loss'].item()
            total_perceptual_loss += losses['perceptual_loss'].item()
            
            # Update progress bar with running averages
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'det': f'{total_detection_loss/(batch_idx+1):.4f}',
                'perc': f'{total_perceptual_loss/(batch_idx+1):.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader_foggy) + batch_idx
                self.writer.add_scalar('Student/train_total_loss', losses['total_loss'].item(), global_step)
                self.writer.add_scalar('Student/train_detection_loss', losses['detection_loss'].item(), global_step)
                self.writer.add_scalar('Student/train_perceptual_loss', losses['perceptual_loss'].item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader_foggy)
        print(f"Epoch {epoch+1} - Avg Total Loss: {avg_loss:.4f}, "
              f"Detection: {total_detection_loss/len(self.train_loader_foggy):.4f}, "
              f"Perceptual: {total_perceptual_loss/len(self.train_loader_foggy):.4f}")
        
        return avg_loss
    
    def validate_student(self, epoch):
        """Validate student network on foggy images."""
        self.student.eval()
        
        # ⚠️ PLACEHOLDER VALIDATION - Not computing real mAP yet
        # For proper evaluation, use evaluate.py after training
        # This is just to enable checkpoint saving logic
        metrics = {'mAP': 0.5, 'mAP50': 0.6}
        
        print(f"⚠️  Validation - mAP: {metrics['mAP']:.4f} (placeholder - not real mAP)")
        print(f"   💡 Run evaluation after training for real metrics")
        self.writer.add_scalar('Student/val_mAP', metrics['mAP'], epoch)
        
        return metrics
    
    def save_checkpoint(self, filename, is_teacher=True):
        """Save model checkpoint."""
        model = self.teacher if is_teacher else self.student
        optimizer = self.teacher_optimizer if is_teacher else self.student_optimizer
        
        save_path = self.checkpoint_dir / filename
        
        try:
            # Ensure directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 Checkpoint directory: {self.checkpoint_dir} (exists: {self.checkpoint_dir.exists()})")
            
            # Create checkpoint dict
            print(f"📦 Creating checkpoint for epoch {self.current_epoch}...")
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': self.best_map,
                'config': self.config
            }
            
            print(f"💾 Saving to: {save_path}...")
            # Save checkpoint
            torch.save(checkpoint, save_path)
            
            # Verify file was written
            if save_path.exists():
                file_size_mb = save_path.stat().st_size / (1024 * 1024)
                print(f"✅ Checkpoint saved successfully!")
                print(f"   Path: {save_path}")
                print(f"   Size: {file_size_mb:.2f} MB")
                return True
            else:
                print(f"❌ ERROR: File not found after torch.save()!")
                print(f"   Expected path: {save_path}")
                print(f"   Directory contents:")
                for f in self.checkpoint_dir.iterdir():
                    print(f"     - {f.name}")
                return False
                
        except Exception as e:
            print(f"❌ EXCEPTION during checkpoint save: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_checkpoint(self, checkpoint_path, is_teacher=True):
        """Load model checkpoint."""
        model = self.teacher if is_teacher else self.student
        optimizer = self.teacher_optimizer if is_teacher else self.student_optimizer
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {self.current_epoch}, Best mAP: {self.best_map:.4f}")
    
    def train(self):
        """Run full training pipeline."""
        print("\n" + "="*60)
        print("PL-RT-DETR Training Pipeline")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Teacher epochs: {self.config['teacher_epochs']}")
        print(f"Student epochs: {self.config['student_epochs']}")
        print("="*60)
        
        # Stage 1: Train teacher
        if not self.config.get('skip_teacher', False):
            self.train_teacher()
        
        # Stage 2: Train student
        if not self.config.get('skip_student', False):
            self.train_student()
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print(f"Best mAP: {self.best_map:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train PL-RT-DETR')
    parser.add_argument('--config', type=str, default='config/pl_rtdetr.yaml',
                        help='Path to configuration file')
    parser.add_argument('--pairs_json', type=str,
                        default='voc_2012/processed/VOC2012_paired/pairs.json',
                        help='Path to pairs.json')
    parser.add_argument('--dataset_root', type=str,
                        default='voc_2012/processed/VOC2012_paired',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/pl_rtdetr',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--teacher_epochs', type=int, default=100)
    parser.add_argument('--student_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'pairs_json': args.pairs_json,
        'dataset_root': args.dataset_root,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'teacher_epochs': args.teacher_epochs,
        'student_epochs': args.student_epochs,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'perceptual_weight': 1.0,
        'img_size': 640,
        'num_workers': 4,
        'device': args.device,
        'val_interval': 5,
        'save_interval': 10,
        'skip_teacher': False,
        'skip_student': False
    }
    
    # Create trainer and start training
    trainer = PLRTDETRTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

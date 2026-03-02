"""
Kaggle Quick-Start Training Script for PL-RT-DETR
Vision Transformer Object Detection on Foggy Weather Dataset

Copy and paste this entire script into a Kaggle notebook code cell.
"""

# ===========================
# SECTION 1: Setup & Clone
# ===========================
import os
import sys

# Clone repository
print("📥 Cloning repository...")
!git clone https://github.com/habibour/fogg_object_det_vision_trans.git
os.chdir('/kaggle/working/fogg_object_det_vision_trans')

# To pull updates: !git pull origin main

print("✅ Repository cloned!")

# ===========================
# SECTION 2: Install Dependencies
# ===========================
print("\n📦 Installing dependencies...")
!pip install -q ultralytics>=8.0.0 opencv-python-headless albumentations pycocotools tensorboard

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ===========================
# SECTION 3: Configure Paths
# ===========================
# Kaggle dataset paths
KAGGLE_VOC_PREFIX = "/kaggle/input/datasets/mdhabibourrahman/voc-2012-filtered"
# Dataset structure - point to VOC2012_paired directory
DATASET_ROOT = f"{KAGGLE_VOC_PREFIX}/voc_2012/processed/VOC2012_paired"
PAIRS_JSON = f"{DATASET_ROOT}/pairs.json"

# Output directories
OUTPUT_DIR = "/kaggle/working/logs"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n📁 Dataset: {DATASET_ROOT}")
print(f"📁 Output: {OUTPUT_DIR}")

# ===========================
# SECTION 4: Training Config
# ===========================
config = {
    'pairs_json': PAIRS_JSON,
    'dataset_root': DATASET_ROOT,
    'model_name': 'rtdetr-l',
    'num_classes': 5,
    'img_size': 640,
    'batch_size': 8,
    'num_workers': 2,
    'device': 'cuda',
    'teacher_epochs': 20,
    'student_epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'perceptual_weight': 1.0,
    'save_interval': 2,  # Save every 2 epochs
    'val_interval': 2,   # Validate every 2 epochs
    'output_dir': OUTPUT_DIR,
    'checkpoint_dir': CHECKPOINT_DIR,
}

print("\n⚙️  Configuration:")
for k, v in config.items():
    print(f"   {k}: {v}")

# ===========================
# SECTION 5: Import Modules
# ===========================
sys.path.append('/kaggle/working/fogg_object_det_vision_trans')

from dataset_loader import RTDETRDataset, collate_fn
from perceptual_loss import CombinedLoss
from train_pl_rtdetr import PLRTDETRTrainer
from evaluate import Evaluator

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

print("\n✅ Modules imported!")

# ===========================
# SECTION 6: Load Data
# ===========================
print("\n📦 Loading datasets...")

with open(PAIRS_JSON, 'r') as f:
    pairs_data = json.load(f)

# Define classes (VOC filtered dataset)
CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    
print(f"Total pairs: {len(pairs_data['pairs'])}")
print(f"Fog levels: {pairs_data['metadata']['fog_levels']}")
print(f"Classes: {CLASSES}")
print(f"Number of classes: {len(CLASSES)}")

# Create dataloaders (handled by PLRTDETRTrainer)

# ===========================
# SECTION 7: Initialize Trainer
# ===========================
print("\n🤖 Initializing trainer...")
trainer = PLRTDETRTrainer(config)

# ===========================
# SECTION 8: Train Teacher
# ===========================
print("\n" + "="*60)
print("STAGE 1: Training Teacher Network")
print("="*60)

trainer.train_teacher()

print("\n✅ Teacher training completed!")

# ===========================
# SECTION 9: Train Student
# ===========================
print("\n" + "="*60)
print("STAGE 2: Training Student Network")
print("="*60)

trainer.train_student()

print("\n✅ Student training completed!")

# ===========================
# SECTION 10: Evaluate
# ===========================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Create test datasets
test_conditions = {
    'clean': RTDETRDataset(PAIRS_JSON, DATASET_ROOT, 'val', config['img_size'], use_foggy=False),
    'fog_low': RTDETRDataset(PAIRS_JSON, DATASET_ROOT, 'val', config['img_size'], use_foggy=True, fog_level='low'),
    'fog_mid': RTDETRDataset(PAIRS_JSON, DATASET_ROOT, 'val', config['img_size'], use_foggy=True, fog_level='mid'),
    'fog_high': RTDETRDataset(PAIRS_JSON, DATASET_ROOT, 'val', config['img_size'], use_foggy=True, fog_level='high'),
}

results = {}
teacher_evaluator = Evaluator(trainer.teacher, device=config['device'])
student_evaluator = Evaluator(trainer.student, device=config['device'])

print(f"\n{'Condition':<15} {'Teacher mAP':<15} {'Student mAP':<15}")
print("-" * 50)

for condition_name, dataset in test_conditions.items():
    test_loader = DataLoader(dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    teacher_metrics = teacher_evaluator.evaluate_on_dataset(test_loader)
    student_metrics = student_evaluator.evaluate_on_dataset(test_loader)
    
    results[condition_name] = {'teacher': teacher_metrics, 'student': student_metrics}
    print(f"{condition_name:<15} {teacher_metrics['mAP']:<15.4f} {student_metrics['mAP']:<15.4f}")

# Save results
with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# ===========================
# SECTION 11: Visualize Results
# ===========================
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

conditions = list(results.keys())
teacher_maps = [results[c]['teacher']['mAP'] for c in conditions]
student_maps = [results[c]['student']['mAP'] for c in conditions]

x = np.arange(len(conditions))
width = 0.35

ax1.bar(x - width/2, teacher_maps, width, label='Teacher', color='steelblue')
ax1.bar(x + width/2, student_maps, width, label='Student', color='coral')
ax1.set_xlabel('Test Condition')
ax1.set_ylabel('mAP')
ax1.set_title('Model Performance Across Fog Levels')
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, rotation=45)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

improvements = [(student_maps[i] - teacher_maps[i]) / max(teacher_maps[i], 0.001) * 100 
                for i in range(len(conditions))]
colors = ['green' if imp > 0 else 'red' for imp in improvements]

ax2.barh(conditions, improvements, color=colors, alpha=0.7)
ax2.set_xlabel('Improvement (%)')
ax2.set_title('Student Improvement Over Teacher')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results.png", dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# SECTION 12: Summary
# ===========================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\n📦 Saved to: {OUTPUT_DIR}")
print(f"   - Teacher model: checkpoints/teacher_best.pth")
print(f"   - Student model: checkpoints/student_best.pth")
print(f"   - Results: evaluation_results.json")
print(f"   - Plots: results.png")
print("\n✅ All files available for download from Kaggle output!")

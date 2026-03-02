# PL-RT-DETR Implementation Guide

## 📚 Overview

This guide explains how to implement and train **PL-RT-DETR (Perceptual Loss RT-DETR)** to reproduce the results from the paper "Weather-Aware Object Detection Transformer for Domain Adaptation".

---

## 🎯 Paper Results to Reproduce

| Model          | VOC (Clean) | RTTS (Real Fog) | Low Fog   | Mid Fog   | High Fog  |
| -------------- | ----------- | --------------- | --------- | --------- | --------- |
| YOLOv8         | 0.903       | 0.315           | 0.896     | 0.895     | 0.890     |
| RT-DETR        | 0.909       | 0.403           | 0.800     | 0.800     | 0.800     |
| **PL-RT-DETR** | **0.909**   | **0.422**       | **0.871** | **0.872** | **0.871** |

**Goal**: Achieve or exceed PL-RT-DETR's performance, especially on foggy conditions.

---

## 📋 What Has Been Implemented

### ✅ Completed Components

1. **Dataset Preparation** (`prepare_dataset.py`)
   - VOC class filtering (5 classes)
   - Synthetic fog generation using ASM
   - Clean-foggy image pairing
   - Train/val/test splits

2. **Data Loading** (`dataset_loader.py`)
   - `VOCPairedDataset`: Loads paired clean-foggy images
   - `RTDETRDataset`: Formats data for RT-DETR
   - Custom collate function for batching

3. **Perceptual Loss** (`perceptual_loss.py`)
   - Image-level perceptual loss (VGG16/VGG19/ResNet50)
   - Feature-level perceptual loss (backbone features)
   - Combined loss function

4. **Training Framework** (`train_pl_rtdetr.py`)
   - Teacher-student training pipeline
   - Stage 1: Teacher training on clean images
   - Stage 2: Student training with knowledge distillation
   - Tensorboard logging

5. **Evaluation** (`evaluate.py`)
   - Multi-condition evaluation (clean, low/mid/high fog)
   - Model comparison utilities
   - Results export to JSON

---

## 🔧 Integration with RT-DETR

### Option 1: Ultralytics RT-DETR (Recommended)

The paper mentions using Ultralytics implementation of RT-DETR. Here's how to integrate:

#### Step 1: Install Ultralytics

```bash
pip install ultralytics
```

#### Step 2: Modify `train_pl_rtdetr.py`

Replace the `setup_models()` method:

```python
def setup_models(self):
    """Setup teacher and student models using Ultralytics RT-DETR."""
    from ultralytics import RTDETR

    print("Loading pretrained RT-DETR models...")

    # Load pretrained RT-DETR
    # Options: rtdetr-l.pt, rtdetr-x.pt
    self.teacher = RTDETR('rtdetr-l.pt')
    self.student = RTDETR('rtdetr-l.pt')

    # Configure for VOC classes (5 classes)
    self.teacher.model.nc = 5
    self.student.model.nc = 5

    self.teacher.to(self.device)
    self.student.to(self.device)
```

#### Step 3: Update Training Loop

For Ultralytics, the training approach needs modification:

```python
# In train_epoch_teacher()
def train_epoch_teacher(self, epoch):
    # Ultralytics has its own training loop
    # Use their API:
    results = self.teacher.train(
        data='voc_custom.yaml',
        epochs=1,
        imgsz=640,
        batch=self.config['batch_size']
    )
    return results.box.map  # mAP
```

#### Step 4: Create VOC Data Config

Create `voc_custom.yaml`:

```yaml
# VOC Custom dataset config for RT-DETR
path: voc_2012/processed/VOC2012_paired/clean
train: JPEGImages
val: JPEGImages

# Classes
names:
  0: bicycle
  1: bus
  2: car
  3: motorbike
  4: person

# Number of classes
nc: 5
```

### Option 2: Hugging Face Transformers RT-DETR

Alternatively, use Hugging Face's implementation:

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

def setup_models(self):
    """Setup models using Hugging Face."""
    self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

    self.teacher = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd",
        num_labels=5,  # Our 5 classes
        ignore_mismatched_sizes=True
    )

    self.student = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd",
        num_labels=5,
        ignore_mismatched_sizes=True
    )

    self.teacher.to(self.device)
    self.student.to(self.device)
```

---

## 🚀 Training Pipeline

### Step 1: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install PyTorch (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify Dataset

```bash
# Test the dataset loader
python dataset_loader.py

# Should output:
# ✅ Dataset loader test complete!
# Train batches: 490
# Val batches: 105
```

### Step 3: Test Perceptual Loss

```bash
python perceptual_loss.py

# Should output:
# ✅ Perceptual Loss Module test complete!
```

### Step 4: Train Teacher Network (Stage 1)

```bash
python train_pl_rtdetr.py \
    --pairs_json voc_2012/processed/VOC2012_paired/pairs.json \
    --dataset_root voc_2012/processed/VOC2012_paired \
    --output_dir outputs/pl_rtdetr \
    --batch_size 8 \
    --teacher_epochs 100 \
    --student_epochs 100 \
    --device cuda
```

**Training Details:**

- **Duration**: ~100 epochs for teacher (paper specification)
- **Data**: Clean images from VOC
- **Objective**: Train robust baseline on clean data
- **Output**: `outputs/pl_rtdetr/checkpoints/teacher_best.pth`

### Step 5: Train Student Network (Stage 2)

The training script automatically proceeds to student training after teacher training completes.

**Training Details:**

- **Duration**: ~100 epochs for student (paper specification)
- **Data**: Foggy images (random fog level selection)
- **Objective**: Learn from teacher using perceptual loss
- **Loss**: Detection loss + Perceptual loss
- **Output**: `outputs/pl_rtdetr/checkpoints/student_best.pth`

### Step 6: Evaluate Model

```bash
python evaluate.py \
    --checkpoint outputs/pl_rtdetr/checkpoints/student_best.pth \
    --pairs_json voc_2012/processed/VOC2012_paired/pairs.json \
    --dataset_root voc_2012/processed/VOC2012_paired \
    --output_dir outputs/evaluation
```

---

## 📊 Monitoring Training

### Tensorboard

View training progress in real-time:

```bash
tensorboard --logdir outputs/pl_rtdetr/logs
```

**Available Metrics:**

- `Teacher/train_loss`: Teacher training loss
- `Teacher/val_mAP`: Teacher validation mAP
- `Student/train_total_loss`: Combined student loss
- `Student/train_detection_loss`: Detection loss only
- `Student/train_perceptual_loss`: Perceptual loss only
- `Student/val_mAP`: Student validation mAP

### Weights & Biases (Optional)

For better experiment tracking:

```bash
pip install wandb
wandb login
```

Add to training script:

```python
import wandb

wandb.init(
    project="pl-rtdetr",
    config=config,
    name=f"teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
```

---

## 🔬 Hyperparameter Tuning

### Key Hyperparameters

Based on the paper and best practices:

| Parameter             | Paper Value | Recommended Range | Notes                           |
| --------------------- | ----------- | ----------------- | ------------------------------- |
| Teacher Epochs        | 100         | 50-150            | Depends on dataset size         |
| Student Epochs        | 100         | 50-150            | Match or exceed teacher         |
| Batch Size            | 8-16        | 4-32              | GPU memory dependent            |
| Learning Rate         | 1e-4        | 1e-5 to 1e-3      | Use cosine annealing            |
| Perceptual Weight (α) | 1.0         | 0.1-2.0           | Balance detection vs perceptual |
| Image Size            | 640         | 640-1024          | RT-DETR default is 640          |

### Tuning Perceptual Loss Weight

The perceptual loss weight α balances detection and domain adaptation:

```bash
# Low weight (emphasize detection)
python train_pl_rtdetr.py --perceptual_weight 0.5

# High weight (emphasize domain adaptation)
python train_pl_rtdetr.py --perceptual_weight 2.0
```

**Recommendation**: Start with α=1.0, then experiment.

---

## 🎨 Advanced Techniques

### 1. Mixed Fog Training (Paper Approach)

The paper mentions training teacher on "mixture of foggy and clear data":

Modify dataloader to randomly select clean or foggy:

```python
def get_mixed_dataloader():
    """Create dataloader that randomly samples clean or foggy."""
    # Combine clean and foggy datasets with random sampling
    pass
```

### 2. Progressive Fog Curriculum

Train with increasing fog difficulty:

```python
# Epochs 1-33: Low fog
# Epochs 34-66: Mid fog
# Epochs 67-100: High fog
current_fog_level = get_fog_level_by_epoch(epoch)
```

### 3. Exponential Moving Average (EMA)

Use EMA for more stable student training:

```python
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(student.parameters(), decay=0.9999)

# During training
loss.backward()
optimizer.step()
ema.update()
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python train_pl_rtdetr.py --batch_size 4

# Or use gradient accumulation
--gradient_accumulation_steps 2
```

#### 2. Slow Training

- Reduce image size: `--img_size 512`
- Use fewer workers: `--num_workers 2`
- Enable mixed precision: Add AMP support

#### 3. Poor Performance on Foggy Images

- Increase perceptual loss weight
- Train teacher longer on mixed data
- Use stronger data augmentation
- Verify fog generation parameters (β, A)

#### 4. Model Not Loading

Ensure checkpoint paths are correct:

```bash
ls -lh outputs/pl_rtdetr/checkpoints/
```

---

## 📈 Expected Results

### Training Time (Estimated)

**Hardware**: NVIDIA RTX 3090 (24GB VRAM)

| Stage     | Epochs | Time per Epoch | Total Time    |
| --------- | ------ | -------------- | ------------- |
| Teacher   | 100    | ~15 min        | ~25 hours     |
| Student   | 100    | ~20 min        | ~33 hours     |
| **Total** | 200    | -              | **~58 hours** |

### Performance Benchmarks

Target performance after full training:

| Dataset     | Expected mAP@50 |
| ----------- | --------------- |
| VOC Clean   | 0.85-0.91       |
| Low Fog     | 0.80-0.87       |
| Mid Fog     | 0.80-0.87       |
| High Fog    | 0.75-0.87       |
| RTTS (Real) | 0.35-0.42       |

---

## 🎯 Quick Start (Minimal Example)

For a quick test run with reduced epochs:

```bash
# Quick test (10 epochs each)
python train_pl_rtdetr.py \
    --batch_size 4 \
    --teacher_epochs 10 \
    --student_epochs 10 \
    --output_dir outputs/test_run
```

---

## 📦 Deliverables

After successful training, you should have:

1. ✅ Trained teacher model: `teacher_best.pth`
2. ✅ Trained student model: `student_best.pth`
3. ✅ Training logs: `tensorboard logs/`
4. ✅ Evaluation results: `evaluation_results.json`
5. ✅ Checkpoints: Periodic saves every 10 epochs

---

## 🔄 Iterative Improvement

### Baseline Experiment

1. Train with default settings
2. Evaluate on all fog conditions
3. Identify weak points

### Iteration 1: Optimize Perceptual Loss

- Try different α values: `[0.5, 1.0, 2.0]`
- Compare results

### Iteration 2: Architecture Tuning

- Try different RT-DETR variants (S, M, L, X)
- Larger models may perform better

### Iteration 3: Data Augmentation

- Add augmentations to training pipeline
- Color jitter, rotation, scaling

---

## 📝 Next Steps

1. **Complete RT-DETR Integration**
   - Choose Ultralytics or Hugging Face
   - Update model loading code
   - Test forward pass

2. **Run Baseline Training**
   - Start with clean RT-DETR (no fog)
   - Establish baseline performance

3. **Train PL-RT-DETR**
   - Follow full 200-epoch protocol
   - Monitor perceptual loss convergence

4. **Evaluate and Compare**
   - Compare with baseline RT-DETR
   - Replicate paper's Table 1 results

5. **Tune and Optimize**
   - Experiment with hyperparameters
   - Try to beat paper's results!

---

## 📚 References

- **Paper**: "Weather-Aware Object Detection Transformer for Domain Adaptation" (arXiv:2504.10877)
- **RT-DETR**: https://github.com/ultralytics/ultralytics
- **Hugging Face RT-DETR**: https://huggingface.co/PekingU/rtdetr_r50vd
- **Pascal VOC**: http://host.robots.ox.ac.uk/pascal/VOC/

---

## 💡 Tips for Better Results

1. **Use pretrained weights**: Don't train from scratch
2. **Monitor both losses**: Detection and perceptual should both decrease
3. **Save often**: Checkpoints every 5-10 epochs
4. **Validate regularly**: Check mAP every 5 epochs
5. **Use EMA**: Smooths model weights for better generalization
6. **Mixed precision**: Faster training with AMP
7. **Data augmentation**: Helps both teacher and student
8. **Learning rate warmup**: Start with low LR for first few epochs

---

**Good luck reproducing (and beating!) the paper's results! 🚀**

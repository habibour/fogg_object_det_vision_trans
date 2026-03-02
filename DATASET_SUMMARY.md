# Dataset Preparation Summary

## ✅ COMPLETION STATUS: SUCCESS

The weather-aware object detection dataset has been successfully prepared according to the paper methodology.

---

## 📊 Dataset Statistics

| Metric             | Value                                    |
| ------------------ | ---------------------------------------- |
| **Total Images**   | 5,607                                    |
| **Target Classes** | 5 (bicycle, bus, car, motorbike, person) |
| **Fog Levels**     | 3 (low, mid, high)                       |
| **Train Split**    | 3,924 images (70%)                       |
| **Val Split**      | 841 images (15%)                         |
| **Test Split**     | 842 images (15%)                         |

---

## 📁 Generated Directory Structure

```
voc_2012/processed/
├── VOC2012_filtered/              # Step 1: Filtered Dataset
│   ├── Annotations/               # 5,607 XML annotation files
│   ├── JPEGImages/                # 5,607 clean images
│   ├── ImageSets/Main/            # train.txt, val.txt
│   ├── classes.txt                # List of 5 target classes
│   └── class_mapping.txt          # Class ID mapping
│
├── VOC2012_foggy/                 # Step 2: Synthetic Fog
│   ├── low/                       # 5,607 images (β=0.08, A=0.85)
│   ├── mid/                       # 5,607 images (β=0.12, A=0.90)
│   └── high/                      # 5,607 images (β=0.16, A=0.95)
│
└── VOC2012_paired/                # Step 3: Paired Dataset
    ├── clean/
    │   ├── JPEGImages/            # 5,607 clean images
    │   └── Annotations/           # 5,607 XML files
    ├── foggy/
    │   ├── low/
    │   │   ├── JPEGImages/        # Symlink to VOC2012_foggy/low
    │   │   └── Annotations/       # 5,607 XML files
    │   ├── mid/
    │   │   ├── JPEGImages/        # Symlink to VOC2012_foggy/mid
    │   │   └── Annotations/       # 5,607 XML files
    │   └── high/
    │       ├── JPEGImages/        # Symlink to VOC2012_foggy/high
    │       └── Annotations/       # 5,607 XML files
    ├── ImageSets/Main/
    │   ├── train.txt              # 3,924 image IDs
    │   ├── val.txt                # 841 image IDs
    │   ├── test.txt               # 842 image IDs
    │   └── trainval.txt           # 4,765 image IDs
    └── pairs.json                 # Clean-foggy mapping (117,759 lines)
```

---

## 🔬 Atmospheric Scattering Model Parameters

According to the paper, synthetic fog was generated using:

**Formula:** `It(x) = Is(x) * e^(-β*d(x)) + A * (1 - e^(-β*d(x)))`

| Fog Level | β (Scattering Coefficient) | A (Atmospheric Light) |
| --------- | -------------------------- | --------------------- |
| Low       | 0.08                       | 0.85                  |
| Mid       | 0.12                       | 0.90                  |
| High      | 0.16                       | 0.95                  |

---

## 📋 Class Distribution

From the filtered dataset:

**Train Split (2,766 images, 7,312 objects):**

- bicycle: 410 instances
- bus: 317 instances
- car: 1,191 instances
- motorbike: 375 instances
- person: 5,019 instances

**Val Split (2,841 images, 7,359 objects):**

- bicycle: 380 instances
- bus: 320 instances
- car: 1,173 instances
- motorbike: 376 instances
- person: 5,110 instances

---

## 🗂️ Key Files

### pairs.json

Maps each clean image to its foggy counterparts across all fog levels.

- **Total pairs:** 5,607
- **Structure:** Each pair contains paths to clean image/annotation and foggy versions (low/mid/high)

Example entry:

```json
{
  "id": "2008_008336",
  "clean": {
    "image": "clean/JPEGImages/2008_008336.jpg",
    "annotation": "clean/Annotations/2008_008336.xml"
  },
  "foggy": {
    "low": { "image": "...", "annotation": "..." },
    "mid": { "image": "...", "annotation": "..." },
    "high": { "image": "...", "annotation": "..." }
  }
}
```

### Split Files

- `train.txt`: 3,924 image IDs for training
- `val.txt`: 841 image IDs for validation
- `test.txt`: 842 image IDs for testing
- `trainval.txt`: 4,765 image IDs (train + val combined)

### class_mapping.txt

```
0: bicycle
1: bus
2: car
3: motorbike
4: person
```

---

## 🎯 Next Steps for Implementation

### 1. Data Loading

Use the `pairs.json` file to create data loaders that:

- Load clean-foggy image pairs during training
- Apply random fog level selection during training (as mentioned in the paper)
- Maintain consistent splits using the ImageSets files

### 2. Model Implementation

According to the paper, implement these three approaches:

#### a) PL-RT-DETR (Perceptual Loss RT-DETR) ⭐ Best Performance

- **Architecture:** Teacher-student framework
- **Teacher:** Pre-trained on clean images
- **Student:** Trained on foggy images
- **Loss:** Original detection loss + perceptual loss
- **Results:** mAP 0.422 on RTTS, ~0.871 on synthetic fog

#### b) WAA-RT-DETR (Weather Adaptive Attention)

- **Modification:** Fog-aware attention mechanism
- **Method:** Uses depth/fog density to adjust attention weights
- **Status:** Required further refinement (convergence issues)

#### c) WFE-RT-DETR (Weather Fusion Encoder)

- **Architecture:** Dual-stream encoder
- **Method:** Process clean and foggy images in parallel, fuse with cross-attention
- **Results:** On par with baseline

### 3. Training Strategy

Based on the paper:

1. **Teacher Training (100 epochs):**
   - Mix of foggy and clear data
   - Random fog severity selection

2. **Student Training (100 epochs):**
   - Knowledge distillation from teacher
   - Perceptual loss for feature alignment
   - Foggy image input

### 4. Evaluation

- Test on synthetic fog (low/mid/high)
- Test on RTTS (real-world foggy dataset)
- Compare with baseline RT-DETR and YOLOv8

---

## 🚀 Quick Usage Examples

### Load a random pair

```python
import json
import random

with open('voc_2012/processed/VOC2012_paired/pairs.json', 'r') as f:
    data = json.load(f)

# Get random pair
pair = random.choice(data['pairs'])
print(f"Clean image: {pair['clean']['image']}")
print(f"Foggy (mid): {pair['foggy']['mid']['image']}")
```

### Load train split

```python
with open('voc_2012/processed/VOC2012_paired/ImageSets/Main/train.txt', 'r') as f:
    train_ids = [line.strip() for line in f]

print(f"Training on {len(train_ids)} images")
```

---

## ✅ Verification Checklist

- [x] VOC dataset filtered to 5 classes
- [x] 5,607 images retained with annotations
- [x] Synthetic fog generated at 3 levels (low, mid, high)
- [x] Paired dataset structure created
- [x] Annotations copied to all foggy directories
- [x] pairs.json created with 5,607 mappings
- [x] Train/val/test splits generated (70/15/15)
- [x] Class mapping file created
- [x] Symlinks used for efficient storage

---

## 📖 Paper Reference

**Title:** Weather-Aware Object Detection Transformer for Domain Adaptation  
**Authors:** Soheil Gharatappeh, Salimeh Yasaei Sekeh, Vikas Dhiman  
**arXiv:** 2504.10877v1  
**Year:** 2025

**Key Finding:** PL-RT-DETR (Perceptual Loss approach) achieved the best performance with consistent improvements over baseline RT-DETR across both synthetic and real-world foggy conditions.

---

## 💾 Storage Information

**Optimization Used:** Symbolic links for foggy images

- Foggy images are stored once in `VOC2012_foggy/`
- `VOC2012_paired/foggy/` uses symlinks to avoid duplication
- Saves ~10GB of storage space

**Total Storage:**

- Filtered dataset: ~1.5GB
- Foggy dataset: ~10GB (all 3 levels)
- Paired dataset: ~1.5GB (excluding symlinked images)

---

**Generated:** February 26, 2026  
**Status:** ✅ Ready for model training

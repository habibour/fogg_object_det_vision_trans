# Weather-Aware Object Detection Dataset Preparation

This repository implements the dataset preparation pipeline for the paper **"Weather-Aware Object Detection Transformer for Domain Adaptation"** (arXiv:2504.10877v1).

## Overview

The pipeline prepares the Pascal VOC dataset for weather-aware object detection by:

1. **Filtering classes** to match the RTTS dataset (5 classes)
2. **Generating synthetic fog** using the Atmospheric Scattering Model (ASM)
3. **Creating paired datasets** of clean and foggy images
4. **Generating train/val/test splits**

## Target Classes

Following the paper's methodology, we filter the Pascal VOC dataset to include only these 5 classes:

- `bicycle`
- `bus`
- `car`
- `motorbike`
- `person`

## Atmospheric Scattering Model (ASM)

Synthetic fog is generated using the ASM formula:

```
It(x) = Is(x) * e^(-β*d(x)) + A * (1 - e^(-β*d(x)))
```

Where:

- `It(x)` = foggy image intensity
- `Is(x)` = clean image intensity
- `A` = atmospheric light (global illumination)
- `β` = scattering coefficient (controls fog density)
- `d(x)` = scene depth

### Fog Levels

Three fog density levels are generated:

| Level | β (Scattering Coeff) | A (Atmospheric Light) | Description |
| ----- | -------------------- | --------------------- | ----------- |
| Low   | 0.08                 | 0.85                  | Light fog   |
| Mid   | 0.12                 | 0.90                  | Medium fog  |
| High  | 0.16                 | 0.95                  | Dense fog   |

## Installation

### Prerequisites

- Python 3.7+
- Pascal VOC 2012 dataset

### Setup

1. Clone or download this repository:

```bash
cd vision_transformer_od
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline with default settings:

```bash
python prepare_dataset.py \
    --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \
    --output_root voc_2012/processed
```

### Step-by-Step Execution

You can also run individual steps:

#### 1. Filter VOC Classes

```bash
python filter_voc_classes.py
```

This will:

- Filter the VOC dataset to keep only the 5 target classes
- Create `voc_2012/VOC2012_filtered/` with filtered images and annotations
- Generate class mapping files

#### 2. Generate Synthetic Fog

```bash
python synthetic_fog.py
```

This will:

- Generate foggy versions of all filtered images
- Create three fog levels: low, mid, high
- Save results to `voc_2012/VOC2012_foggy/`
- Optionally save depth maps for analysis

#### 3. Create Paired Dataset

```bash
python image_pairing.py
```

This will:

- Create a structured paired dataset
- Copy clean and foggy images to organized directories
- Generate `pairs.json` mapping file
- Create train/val/test splits

### Advanced Options

```bash
# Skip filtering if already done
python prepare_dataset.py \
    --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \
    --output_root voc_2012/processed \
    --skip_filtering

# Generate only specific fog levels
python prepare_dataset.py \
    --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \
    --output_root voc_2012/processed \
    --fog_levels low high

# Skip fog generation (use existing)
python prepare_dataset.py \
    --voc_root voc_2012/VOC2012_train_val/VOC2012_train_val \
    --output_root voc_2012/processed \
    --skip_fog
```

## Output Structure

The pipeline creates the following directory structure:

```
voc_2012/processed/
├── VOC2012_filtered/                # Step 1: Filtered dataset
│   ├── Annotations/
│   ├── JPEGImages/
│   ├── ImageSets/Main/
│   ├── classes.txt
│   └── class_mapping.txt
│
├── VOC2012_foggy/                   # Step 2: Foggy images
│   ├── low/JPEGImages/
│   ├── mid/JPEGImages/
│   └── high/JPEGImages/
│
└── VOC2012_paired/                  # Step 3: Paired dataset
    ├── clean/
    │   ├── JPEGImages/
    │   └── Annotations/
    ├── foggy/
    │   ├── low/
    │   │   ├── JPEGImages/
    │   │   └── Annotations/
    │   ├── mid/
    │   │   ├── JPEGImages/
    │   │   └── Annotations/
    │   └── high/
    │       ├── JPEGImages/
    │       └── Annotations/
    ├── ImageSets/Main/
    │   ├── train.txt
    │   ├── val.txt
    │   ├── test.txt
    │   └── trainval.txt
    └── pairs.json                   # Clean-foggy image mapping
```

## Dataset Files

### pairs.json

The `pairs.json` file maps each clean image to its foggy counterparts:

```json
{
  "metadata": {
    "num_pairs": 1500,
    "fog_levels": ["low", "mid", "high"]
  },
  "pairs": [
    {
      "id": "2008_000008",
      "clean": {
        "image": "clean/JPEGImages/2008_000008.jpg",
        "annotation": "clean/Annotations/2008_000008.xml"
      },
      "foggy": {
        "low": {
          "image": "foggy/low/JPEGImages/2008_000008.jpg",
          "annotation": "foggy/low/Annotations/2008_000008.xml"
        },
        "mid": { ... },
        "high": { ... }
      }
    }
  ]
}
```

### Split Files

Train/val/test splits are stored in `ImageSets/Main/*.txt` with one image ID per line.

## Scripts Overview

| Script                  | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `prepare_dataset.py`    | **Main pipeline** - Orchestrates all steps |
| `filter_voc_classes.py` | Filters VOC dataset to target classes      |
| `synthetic_fog.py`      | Generates synthetic fog using ASM          |
| `image_pairing.py`      | Creates paired clean-foggy dataset         |
| `extract_pdf.py`        | Utility to extract text from paper PDF     |

## Dataset Statistics

After running the pipeline, you'll see statistics like:

```
Dataset Split Statistics:
----------------------------------------
  TRAIN     :  1050 images
  VAL       :   225 images
  TEST      :   225 images
----------------------------------------

Total clean images: 1500
Total low fog images: 1500
Total mid fog images: 1500
Total high fog images: 1500

Class Distribution:
----------------------------------------
  bicycle     :   523 instances
  bus         :   312 instances
  car         :  1876 instances
  motorbike   :   421 instances
  person      :  2347 instances
----------------------------------------
```

## Next Steps

After preparing the dataset:

1. **Verify the dataset**:

   ```bash
   # Check a few random pairs
   python -c "from image_pairing import ImagePairManager; \
              mgr = ImagePairManager('.', '.'); \
              print(mgr.get_random_pair('voc_2012/processed/VOC2012_paired/pairs.json'))"
   ```

2. **Implement RT-DETR variants** from the paper:
   - PL-RT-DETR (Perceptual Loss)
   - WAA-RT-DETR (Weather Adaptive Attention)
   - WFE-RT-DETR (Weather Fusion Encoder)

3. **Training**:
   - Use the paired dataset for domain adaptation
   - Train teacher network on clean images
   - Transfer knowledge to student network using foggy images
   - Apply perceptual loss between teacher and student features

## Paper Reference

```bibtex
@article{gharatappeh2025weather,
  title={Weather-Aware Object Detection Transformer for Domain Adaptation},
  author={Gharatappeh, Soheil and Yasaei Sekeh, Salimeh and Dhiman, Vikas},
  journal={arXiv preprint arXiv:2504.10877},
  year={2025}
}
```

## Key Findings from the Paper

- **PL-RT-DETR** achieved the best performance with mAP of 0.422 on RTTS (real-world foggy dataset)
- On synthetic fog, PL-RT-DETR achieved ~0.871 mAP across all fog levels
- Perceptual loss helps preserve semantic features despite fog degradation
- Weather Fusion Encoder performed on par with baseline
- Weather Adaptive Attention requires further refinement

## Troubleshooting

### Common Issues

1. **Missing VOC dataset**:
   - Download Pascal VOC 2012 from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
   - Extract to `voc_2012/VOC2012_train_val/`

2. **Memory issues during fog generation**:
   - Process images in batches
   - Reduce image resolution if needed

3. **Module import errors**:
   - Ensure all scripts are in the same directory
   - Check that `requirements.txt` packages are installed

## License

This implementation is for research and educational purposes. Please cite the original paper if you use this code.

## Contact

For questions or issues related to the paper implementation, please refer to the original paper authors.

---

**Note**: This is an implementation of the dataset preparation methodology described in the paper. The actual RT-DETR model training code is not included in this repository.

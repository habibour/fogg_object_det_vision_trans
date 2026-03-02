# Kaggle Training Guide for PL-RT-DETR

## 🚀 Quick Start

### Option 1: Use the Jupyter Notebook (Recommended)

1. Upload `kaggle_training.ipynb` to your Kaggle notebook
2. Add the dataset: `mdhabibourrahman/voc-2012-filtered`
3. Enable GPU in notebook settings
4. Run all cells sequentially

### Option 2: Copy-Paste Single Script

1. Create a new Kaggle notebook
2. Copy entire content from `kaggle_quickstart.py`
3. Paste into a single code cell
4. Add the dataset: `mdhabibourrahman/voc-2012-filtered`
5. Enable GPU
6. Run the cell

## 📋 Prerequisites

### Kaggle Dataset

Add this dataset to your kernel:

- **Dataset**: `mdhabibourrahman/voc-2012-filtered`
- Path: `/kaggle/input/datasets/mdhabibourrahman/voc-2012-filtered`

### Kaggle Settings

- **Accelerator**: GPU (T4 or P100)
- **Internet**: ON (to clone GitHub repo and install packages)
- **Environment**: Python 3.10+

## 🔄 Updating Code

To pull the latest code from GitHub within Kaggle:

```python
# In a code cell
!cd /kaggle/working/fogg_object_det_vision_trans && git pull origin main
```

Or re-run the clone command (it will skip if already exists):

```python
!git clone https://github.com/habibour/fogg_object_det_vision_trans.git
```

## 📊 Training Details

### Stage 1: Teacher Network

- **Dataset**: Clean VOC 2012 images
- **Epochs**: 20 (configurable)
- **Checkpoints**: Saved every 2 epochs to `/kaggle/working/logs/checkpoints/`

### Stage 2: Student Network

- **Dataset**: Foggy VOC 2012 images (synthetic fog at 3 levels)
- **Method**: Knowledge distillation with perceptual loss
- **Teacher**: Frozen, provides guidance
- **Epochs**: 20 (configurable)
- **Checkpoints**: Saved every 2 epochs

### Evaluation

Models are evaluated on:

- Clean images
- Low fog
- Mid fog
- High fog

## 📁 Output Files

All outputs saved to `/kaggle/working/logs/`:

```
/kaggle/working/logs/
├── checkpoints/
│   ├── teacher_best.pth          # Best teacher model
│   ├── teacher_epoch_2.pth       # Teacher checkpoint at epoch 2
│   ├── teacher_epoch_4.pth       # Teacher checkpoint at epoch 4
│   ├── ...
│   ├── student_best.pth          # Best student model
│   ├── student_epoch_2.pth       # Student checkpoint at epoch 2
│   └── ...
├── teacher_logs/                 # TensorBoard logs for teacher
├── student_logs/                 # TensorBoard logs for student
├── evaluation_results.json       # Evaluation metrics
├── results.png                   # Performance comparison plots
└── training_summary.json         # Complete training summary
```

## 🔧 Configuration

Modify training parameters in the config dictionary:

```python
config = {
    'batch_size': 8,              # Adjust based on GPU memory
    'teacher_epochs': 20,         # Number of teacher training epochs
    'student_epochs': 20,         # Number of student training epochs
    'learning_rate': 1e-4,        # Learning rate
    'save_interval': 2,           # Save checkpoint every N epochs
    'val_interval': 2,            # Validate every N epochs
    'img_size': 640,              # Input image size
    'perceptual_weight': 1.0,     # Weight for perceptual loss
}
```

## 📥 Downloading Results

From Kaggle notebook:

1. Click "Output" tab
2. Download entire `/kaggle/working/logs/` folder
3. Or download individual checkpoint files

## 🐛 Troubleshooting

### Dataset Not Found

```
⚠️ Dataset not found!
```

**Solution**: Add dataset `mdhabibourrahman/voc-2012-filtered` to your Kaggle kernel

### GPU Memory Error

```
CUDA out of memory
```

**Solution**: Reduce batch_size in config (try 4 or 2)

### Git Clone Fails

```
fatal: could not create work tree
```

**Solution**: Repository already exists. Navigate to it:

```python
os.chdir('/kaggle/working/fogg_object_det_vision_trans')
```

## 📖 Using Trained Models

### Load Checkpoint for Inference

```python
import torch
from ultralytics import RTDETR

# Load model
model = RTDETR('rtdetr-l.pt')
checkpoint = torch.load('/kaggle/working/logs/checkpoints/student_best.pth')
model.model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
results = model.predict('path/to/foggy/image.jpg', conf=0.25)
```

### Resume Training from Checkpoint

Modify the trainer to load existing checkpoint before calling `train_teacher()` or `train_student()`.

## 📚 Classes

The model detects 5 object classes from VOC 2012:

1. bicycle
2. bus
3. car
4. motorbike
5. person

## 🔗 Links

- **GitHub Repository**: https://github.com/habibour/fogg_object_det_vision_trans
- **Paper**: Weather-Aware Object Detection Transformer for Domain Adaptation

## 💡 Tips

1. **Training Time**:
   - Teacher: ~2-3 hours on T4 GPU (20 epochs)
   - Student: ~2-3 hours on T4 GPU (20 epochs)
   - Total: ~4-6 hours

2. **Monitoring**: Use TensorBoard to monitor training:

   ```python
   %load_ext tensorboard
   %tensorboard --logdir /kaggle/working/logs/teacher_logs
   ```

3. **Checkpoints**: Saved every 2 epochs - you can download intermediate checkpoints if training is interrupted

4. **Memory**: If GPU memory is tight, reduce batch_size or img_size

## ✅ Success Checklist

- [ ] Dataset added to Kaggle kernel
- [ ] GPU enabled in notebook settings
- [ ] Internet enabled (for git clone)
- [ ] Code cloned/pasted successfully
- [ ] Teacher training completed
- [ ] Student training completed
- [ ] Evaluation completed
- [ ] Checkpoints saved to `/kaggle/working/logs/`
- [ ] Results downloaded

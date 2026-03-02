#!/usr/bin/env python3
"""
Update the training configuration cell in kaggle_training.ipynb
"""
import json

NOTEBOOK_PATH = 'kaggle_training.ipynb'

# New configuration code
NEW_CONFIG_CODE = """# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# 
# Choose your training mode:
# 
# 🚀 QUICK TEST (Recommended for first run)
#    - teacher_epochs: 10
#    - student_epochs: 10
#    - Time: ~2-3 hours
#    - Purpose: Verify pipeline, check for errors
#    - Expected mAP: 0.60-0.75 (lower due to limited training)
#
# 🎯 FULL TRAINING (For paper-level results)
#    - teacher_epochs: 100
#    - student_epochs: 100
#    - Time: ~58 hours (RTX 3090) or ~80+ hours (T4/P100)
#    - Purpose: Reproduce paper results
#    - Expected mAP: 0.75-0.871 (baseline to paper-level)
#
# ⚙️ MONITORING:
#    - Loss should DECREASE steadily (not stay constant)
#    - If loss stuck ~0.1 or constant, check warnings in output
#    - Progress bar updates every 5% of batches
#
# 🔧 TUNING PERCEPTUAL WEIGHT:
#    - Default: 0.1 (balanced between detection and perceptual loss)
#    - If fog performance is low: increase to 0.2-0.3
#    - If clean performance drops: decrease to 0.05
#
# ============================================================================

config = {
    # Dataset
    'pairs_json': PAIRS_JSON,
    'dataset_root': DATASET_ROOT,
    
    # Model
    'model_name': 'rtdetr-l',  # RT-DETR Large
    'num_classes': 5,  # bicycle, bus, car, motorbike, person
    'img_size': 640,
    
    # Training
    'batch_size': 8,  # Reduce to 4 if OOM errors occur
    'num_workers': 2,
    'device': 'cuda',
    
    # ⚠️ CHOOSE YOUR TRAINING MODE (uncomment one set):
    
    # === QUICK TEST MODE (2-3 hours) ===
    'teacher_epochs': 10,
    'student_epochs': 10,
    
    # === FULL TRAINING MODE (58+ hours) === 
    # Uncomment below for paper-level results:
    # 'teacher_epochs': 100,
    # 'student_epochs': 100,
    
    # Optimization
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    # Loss weighting (tune based on results)
    # Increase if fog performance low, decrease if clean performance drops
    'perceptual_weight': 0.1,  # Default: 0.1 | Low fog perf: 0.2-0.3 | High clean perf: 0.05
    'perceptual_backbone': 'vgg19',  # Options: vgg16, vgg19, resnet50
    
    # Checkpointing
    'save_interval': 2,  # Save checkpoint every 2 epochs
    'val_interval': 2,   # Validate every 2 epochs
    
    # Paths
    'output_dir': OUTPUT_DIR,
    'checkpoint_dir': CHECKPOINT_DIR,
}

print("⚙️  Training Configuration:")
print(f"   Mode: {'QUICK TEST' if config['teacher_epochs'] <= 20 else 'FULL TRAINING'}")
print(f"   Total epochs: {config['teacher_epochs'] + config['student_epochs']}")
print(f"   Estimated time: {'~2-3 hours' if config['teacher_epochs'] <= 20 else '~58-80 hours'}")
print(f"\\n📊 Key Settings:")
for key, value in config.items():
    if key in ['teacher_epochs', 'student_epochs', 'batch_size', 'perceptual_weight', 'learning_rate']:
        print(f"   {key}: {value}")
print("\\n💡 Tip: Monitor loss - it should DECREASE steadily, not stay constant!")"""

# Load notebook
with open(NOTEBOOK_PATH, 'r') as f:
    notebook = json.load(f)

# Find and update the configuration cell
# Look for cell with 'Training configuration' comment
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '# Training configuration' in source and 'config = {' in source:
            print(f"Found configuration cell at index {i}")
            # Update the cell source
            cell['source'] = NEW_CONFIG_CODE.split('\n')
            # Add newline to each line except the last
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            print("✅ Updated configuration cell")
            break
else:
    print("⚠️  Configuration cell not found")
    exit(1)

# Save notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook saved: {NOTEBOOK_PATH}")

# Periodontal Bone Loss Evaluation using Deep Learning-Based Image Segmentation

## 📦 Project Structure

```
📁 scripts
├── 📄 predict.py           # Inference script for generating segmentation results
├── 📄 prepare_kfold.py     # Data splitting for K-Fold cross-validation
├── 📄 run_experiment.py    # Main script to run complete experimental workflows
└── 📄 train.py             # Entry point for single model training
📁 src
├── 📂 models
│   ├── 📄 ...              # Additional model implementation
│   └── 📄 unet.py          # U-Net model architecture definition
├── 📄 config.py            # Configuration settings and hyperparameters
├── 📄 dataset.py           # Custom Dataset and DataLoader implementation
├── 📄 losses.py            # Custom loss functions
├── 📄 metrics.py           # Evaluation metrics (e.g., mIoU)
├── 📄 trainer.py           # Handles training and validation loops
└── 📄 utils.py             # Utility functions
```

## 📁 Dataset Preparation

Your dataset must follow:
```
📁 bone_tooth_mask
├── 📂 image
│   ├── 📂 data_1
│   │   ├── 📄 91.png
│   │   └── 📄 ...
│   └── 📂 ...
└── 📂 mask
    ├── 📂 data_1
    └── 📂 ...
```
**Requirements**
* Images and masks must share identical folder/file names.
* Masks should contain pixel labels `{0, 1, 2}` for 3 classes.

## ⚙️ Configuration (`src/config.py`)

```python=
class Config:
    # System and Experiment
    EXPERIMENT = 'UNet_baseline'
    SEED = 42
    NUM_WORKERS = 4

    # Data Configuration
    DATASET = 'bone_tooth_mask'
    NUM_FOLDS = 4
    BATCH_SIZE = 16

    # Training Settings
    NUM_EPOCHS = 50

    # Model Architecture
    MODEL_NAME = 'UNet'
    MODEL_PARAMETERS = {
        'in_channels': 1,
        'num_classes': 3
    }

    # Optimizer
    OPTIMIZER_NAME = 'Adam'
    OPTIMIZER_PARAMETERS = {
        'lr': 1e-4
    }

    # Loss Function
    LOSS_NAME = 'MultipleLoss'
    MAIN_LOSS = 'Total Loss'
    LOSS_PARAMETERS = {
        'num_classes': 3
    }

    # Metric
    METRIC_NAME = 'mIoU'
    METRIC_PARAMETERS = {
        'num_classes': 3
    }
```
| You can modify this file or override parameters inside the scripts if needed.

## 🔀 Generate K-Fold Split

```
python -m scripts.prepare_kfold
```
Generates:
```
splits/bone_tooth_mask.json
{
    "1": ["data_1", ...],
    "2": ["data_2", ...],
    "3": ["data_3", ...],
    "4": ["data_4", ...]
}
```

## 🏋️ Train Model

**Train a single fold**

```
python -m scripts.train --fold 1
```

**Run full experiment (all folds)**

```
python -m scripts.run_experiment
```

Results saved as:
```
📁 logs/<EXPERIMENT_NAME>
├── 📂 Fold_1
│   ├── 📄 best.pth
│   └── 📄 last.pth
├── 📂 Fold_2
├── 📂 Fold_3
├── 📂 Fold_4
├── 📄 bone_tooth_mask.json
└── 📄 config.json
```

## 🔍 Inference

```
python -m scripts.predict <EXPERIMENT_NAME>
```
Outputs:
```
📁 outputs/<EXPERIMENT_NAME>
├── 📂 Fold_1
│   ├── 📂 data_1
│   │   ├── 📄 91.png
│   │   ├── 📄 ...
│   │   ├── 📄 ground_truth.npy
│   │   └── 📄 volume.npy
│   └── 📂 ...
├── 📂 Fold_2
├── 📂 Fold_3
└── 📂 Fold_4
```

## 📝 Notes

* Ensure the dataset follows the required structure.
* Modify `Config` to customize model, loss, optimizer, and metrics.
* Support for additional models/losses can be added under `src/models` or `src/losses`.

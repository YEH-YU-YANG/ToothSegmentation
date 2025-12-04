import os
import torch

class Config:
    # System and Experiment
    EXPERIMENT = 'UNet_DeepSupervision_MultiLoss'
    SEED = 42
    NUM_WORKERS = 4

    # Data Configuration
    DATASET = 'bone_tooth_mask'
    NUM_FOLDS = 4
    BATCH_SIZE = 16

    # Training Settings
    NUM_EPOCHS = 50

    # Model Architecture
    MODEL_NAME = 'DeepUNet'
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
    LOSS_NAME = 'DeepSupervisionLoss'
    MAIN_LOSS = 'Total Loss'
    LOSS_PARAMETERS = {
        'num_classes': 3
    }

    # Metric
    METRIC_NAME = 'mIoU'
    METRIC_PARAMETERS = {
        'num_classes': 3,
        'predict_index': 0
    }

    def __init__(self):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.SPLIT_FILENAME = os.path.join('splits', f'{self.DATASET}.json')

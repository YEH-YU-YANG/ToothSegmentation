import json
import os

from sklearn.model_selection import GroupKFold
from src.utils import Table
from src.config import load_config

config = load_config('configs/config.toml')

image_dir = os.path.join('datasets', config.dataset, 'image')

patients = os.listdir(image_dir)
patients.sort(key=lambda x: int(x[5:]))

groups = []
for patient in patients:
    images = os.listdir(os.path.join(image_dir, patient))
    groups.extend([patient] * len(images))

Table(
    ['Item', 'Count'],
    ['Patient', len(patients)],
    ['Image', len(groups)]
).display()

k_fold = GroupKFold(config.num_folds)

folds = {}

table = Table(['Fold', 'Train', 'Val'])
for fold, (train_indices, val_indices) in enumerate(k_fold.split(range(len(groups)), groups=groups), start=1):
    val_patients = set(groups[index] for index in val_indices)
    val_patients = sorted(val_patients, key=lambda x: int(x[5:]))
    folds[str(fold)] = val_patients
    table.add_row([f'Fold {fold}', len(train_indices), len(val_indices)])
table.display()

for fold, patients in folds.items():
    print(f'Fold {fold}: {patients}')

os.makedirs(os.path.dirname(config.split_filename), exist_ok=True)
with open(config.split_filename, 'w') as file:
    json.dump(folds, file)

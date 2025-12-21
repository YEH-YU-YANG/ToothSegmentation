import cv2
import json
import os
import torch

from torch.utils.data import Dataset, DataLoader

def get_fold(split_file, fold_id):
    with open(split_file) as file:
        folds = json.load(file)
    train_patients = []
    val_patients = []
    for fold, patients in folds.items():
        patient_list = val_patients if str(fold_id) == fold else train_patients
        patient_list.extend(patients)
    train_patients.sort(key=lambda x: int(x[5:]))
    val_patients.sort(key=lambda x: int(x[5:]))
    return train_patients, val_patients

class CBCTDataset(Dataset):
    def __init__(self, dataset_dir, patients, binary=False):
        self.dataset_dir = dataset_dir
        self.patients = patients
        self.binary = binary

        self.filenames = []
        for patient in patients:
            filenames = os.listdir(f'{dataset_dir}/image/{patient}')
            filenames.sort(key=lambda x: int(x[:-4]))
            filenames = [f'{patient}/{filename}' for filename in filenames]
            self.filenames.extend(filenames)
        
        self.length = len(self.filenames)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        filename = self.filenames[index]

        image = cv2.imread(f'{self.dataset_dir}/image/{filename}', cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image)
        image = image / 255
        image = image.unsqueeze(0)

        mask = cv2.imread(f'{self.dataset_dir}/mask/{filename}', cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask)
        if self.binary:
            mask = mask == 255
        mask = mask.long()

        return image, mask, filename

def get_loader(config):
    train_patients, val_patients = get_fold(config.split_filename, config.fold)

    dataset_dir = os.path.join('datasets', config.dataset)

    train_dataset = CBCTDataset(dataset_dir, train_patients)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    val_dataset = CBCTDataset(dataset_dir, val_patients)
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader

if __name__ == '__main__':
    from src.utils import Table
    from src.config import load_config

    config = load_config('configs/unet.toml')
    config.fold = 1

    train_loader, val_loader = get_loader(config)

    for images, masks, filenames in train_loader:
        break

    print('  Train Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['mask', masks.shape, f'{masks.min()} ~ {masks.max()}', masks.dtype],
        ['filename', len(filenames), '', f'{type(filenames).__name__}[{type(filenames[0]).__name__}]']
    ).display()

    for images, masks, filenames in val_loader:
        break

    print('  Val Dataset')
    Table(
        ['Item', 'Shape', 'Range', 'Type'],
        ['image', images.shape, f'{images.min():.4f} ~ {images.max():.4f}', images.dtype],
        ['mask', masks.shape, f'{masks.min()} ~ {masks.max()}', masks.dtype],
        ['filename', len(filenames), '', f'{type(filenames).__name__}[{type(filenames[0]).__name__}]']
    ).display()

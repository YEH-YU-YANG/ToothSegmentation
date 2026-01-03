import torch
import os
import numpy

from torchvision.transforms import ToPILImage
from PIL import ImageFont
from PIL.ImageDraw import Draw
from src.dataset import CBCTDataset
from src.console import track
from torch.utils.data import DataLoader

def save_image(image, output_dir, filename):
    to_image = ToPILImage()
    image = to_image(image)
    image.save(os.path.join(output_dir, filename))

def save_compare(image, predict, mask, output_dir, filename):
    to_image = ToPILImage()
    font = ImageFont.load_default(20)

    width = image.size(1)

    image = torch.cat([image, predict, mask], dim=1)
    image = to_image(image)
    draw = Draw(image)
    for i, text in enumerate([filename, 'predict', 'mask']):
        draw.text((width * i, 0), text, 'white', font)
    
    image.save(os.path.join(output_dir, filename))

def predict_patient(model, dataset, patient, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = os.path.join('datasets', dataset)
    dataset = CBCTDataset(dataset_dir, [patient])
    loader = DataLoader(dataset, config.batch_size, shuffle=False)

    volume = []
    ground_truth = []
    for images, masks, filenames in track(loader, desc=f'{patient:7}'):
        images = images.to(config.device)

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(images) # (B, C, H, W) or (N, B, C, H, W)
            if isinstance(predicts, tuple):
                predicts = predicts[0] # (B, C, H, W)
        predicts = predicts.argmax(1).cpu() # (B, H, W)

        volume.append(predicts)
        ground_truth.append(masks)

        images = images.squeeze(1).cpu()

        for filename, image, predict, mask in zip(filenames, images, predicts, masks):
            image_filename = os.path.basename(filename)
            save_image(predict.to(torch.uint8), os.path.join(output_dir, 'predict'), image_filename)
            predict = predict / 2
            mask = mask / 2
            save_compare(image, predict, mask, os.path.join(output_dir, 'compare'), image_filename)

    volume = torch.cat(volume)
    numpy.save(os.path.join(output_dir, 'volume.npy'), volume.numpy())

    ground_truth = torch.cat(ground_truth)
    numpy.save(os.path.join(output_dir, 'ground_truth.npy'), ground_truth.numpy())

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config
    from src.dataset import get_fold
    from src.models import load_model
    from src.downloader import ensure_experiment_exists

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    ensure_experiment_exists(experiment_name)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))

    for fold in range(1, config.num_folds + 1):
        config.fold = fold

        model = load_model(config)

        _, val_dataset_patients = get_fold(config.split_filename, fold)
        for dataset, patients in val_dataset_patients.items():
            for patient in patients:
                output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient)
                os.makedirs(os.path.join(output_dir, 'predict'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'compare'), exist_ok=True)

                predict_patient(model, dataset, patient, output_dir, config)

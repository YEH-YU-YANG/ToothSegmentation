import torch
import os
import numpy

from torchvision.transforms import ToPILImage
from PIL import ImageFont
from PIL.ImageDraw import Draw
from src.dataset import CBCTDataset
from src.utils import track
from src.models import get_model
from torch.utils.data import DataLoader

def save_image(image, predict, mask, output_dir, filename):
    to_image = ToPILImage()
    font = ImageFont.load_default(20)

    width = image.size(1)

    image = torch.cat([image, predict, mask], dim=1)
    image = to_image(image)
    draw = Draw(image)
    for i, text in enumerate([filename, 'predict', 'mask']):
        draw.text((width * i, 0), text, 'white', font)
    
    image.save(os.path.join(output_dir, filename))

def predict_patient(patient, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    dataset = CBCTDataset(config.DATASET, [patient])
    loader = DataLoader(dataset, config.BATCH_SIZE, shuffle=False)

    volume = []
    ground_truth = []
    for images, masks, filenames in track(loader, desc=f'{patient:7}'):
        images = images.to(config.DEVICE)

        with torch.no_grad(), torch.autocast(config.DEVICE):
            predicts = model(images) # (B, C, H, W)
        predicts = predicts.argmax(1).cpu() # (B, H, W)

        volume.append(predicts)
        ground_truth.append(masks)

        images = images.squeeze(1).cpu()
        predicts = predicts / 2
        masks = masks / 2

        for filename, image, predict, mask in zip(filenames, images, predicts, masks):
            image_filename = os.path.basename(filename)
            save_image(image, predict, mask, output_dir, image_filename)
    
    volume = torch.cat(volume)
    numpy.save(os.path.join(output_dir, 'volume.npy'), volume.numpy())

    ground_truth = torch.cat(ground_truth)
    numpy.save(os.path.join(output_dir, 'ground_truth.npy'), ground_truth.numpy())

def load_model(config):
    model = get_model(config).to(config.DEVICE)
    state_dict = torch.load(os.path.join('logs', config.EXPERIMENT, f'Fold_{config.FOLD}', 'best.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.utils import load_config
    from src.dataset import get_fold

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.json'))

    for fold in range(1, config.NUM_FOLDS + 1):
        config.FOLD = fold

        model = load_model(config)

        _, val_patients = get_fold(config.SPLIT_FILENAME, fold)
        for patient in val_patients:
            output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', patient)
            os.makedirs(output_dir, exist_ok=True)

            predict_patient(patient, output_dir, config)

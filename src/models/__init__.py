import os
import torch

def get_model(config):
    match config.model.name:
        case 'UNet':
            from .unet import UNet
            return UNet(**config.model.parameters)
        case 'U2Net':
            from .u2net import U2Net
            return U2Net(**config.model.parameters)
        case 'DeepUNet':
            from .deep_unet import DeepUNet
            return DeepUNet(**config.model.parameters)

def load_model(config):
    model = get_model(config).to(config.device)
    state_dict = torch.load(os.path.join('logs', config.experiment, f'Fold_{config.fold}', 'best.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

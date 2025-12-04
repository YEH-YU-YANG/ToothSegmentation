from .unet import UNet
from .u2net import U2Net
from .deep_unet import DeepUNet

MODELS = {
    'UNet': UNet,
    'U2Net': U2Net,
    'DeepUNet': DeepUNet
}

def get_model(config):
    return MODELS[config.MODEL_NAME](**config.MODEL_PARAMETERS)

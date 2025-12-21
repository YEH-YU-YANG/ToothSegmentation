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

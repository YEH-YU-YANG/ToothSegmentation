import os
import torch
import tomllib

from omegaconf import OmegaConf, MISSING

def load_config(load_path):
    with open(load_path, 'rb') as file:
        content = tomllib.load(file)

    config = OmegaConf.create(content)

    config.fold = MISSING
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.split_filename = os.path.join('splits', f'{config.dataset}.json')

    OmegaConf.set_struct(config, True)

    return config

if __name__ == '__main__':
    from src.utils import Table
    from omegaconf.dictconfig import DictConfig

    def get_items(config, prefix=''):
        for key, value in config.items():
            key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, DictConfig):
                yield from get_items(value, key)
            else:
                yield key, value

    config = load_config('configs/unet.toml')
    config.fold = 1

    table = Table(['Parameter', 'Value'])
    for key, value in get_items(config):
        table.add_row([key, value])
    table.display()

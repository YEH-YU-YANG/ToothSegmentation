import torch
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11
from omegaconf import OmegaConf, MISSING

def load_config(load_path):
    with open(load_path, 'rb') as file:
        content = tomllib.load(file)

    config = OmegaConf.create(content)

    config.fold = MISSING
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    OmegaConf.set_struct(config, True)

    return config

if __name__ == '__main__':
    from src.console import Table
    from omegaconf.dictconfig import DictConfig

    def get_items(config, prefix=''):
        for key, value in config.items():
            key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, DictConfig):
                yield from get_items(value, key)
            else:
                yield key, value

    config = load_config('configs/config.toml')
    config.fold = 1

    Table(
        ['Parameter', 'Value'],
        *get_items(config)
    ).display()

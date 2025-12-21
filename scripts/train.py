from src.trainer import Trainer
from src.utils import Table

def train(config):
    Table(
        ['Argument', 'Value'],
        ['fold', config.fold]
    ).display()

    trainer = Trainer(config)
    trainer.fit(config.num_epochs)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config

    config = load_config('configs/config.toml')

    parser = ArgumentParser()
    parser.add_argument('--fold', type=int, choices=list(range(1, config.num_folds + 1)), required=True)
    args = parser.parse_args()

    config.fold = args.fold

    train(config)

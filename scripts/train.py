from src.trainer import Trainer
from src.utils import Table

def train(config):
    Table(
        ['Argument', 'Value'],
        ['fold', config.FOLD]
    ).display()

    trainer = Trainer(config)
    trainer.fit(config.NUM_EPOCHS)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import Config

    config = Config()

    parser = ArgumentParser()
    parser.add_argument('--fold', type=int, choices=list(range(1, config.NUM_FOLDS + 1)), required=True)
    args = parser.parse_args()

    fold = args.fold
    config.FOLD = fold

    train(config)

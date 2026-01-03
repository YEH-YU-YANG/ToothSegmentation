import os
import torch

from src.models import load_model

def evaluate_fold(config, criterion, metric_fn):
    model = load_model(config)
    _, loader = get_loader(config)

    logs = defaultdict(float)
    data_size = 0

    for images, masks, _ in track(loader, desc=f'Fold {config.fold}'):
        images = images.to(config.device)
        masks = masks.to(config.device)

        zero_mask = (masks == 0).all((1, 2))
        images = images[~zero_mask]
        masks = masks[~zero_mask]

        batch_size = images.size(0)
        data_size += batch_size
        if batch_size == 0:
            continue

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(images)
            loss_dict = criterion(predicts, masks)

        for name, loss in loss_dict.items():
            logs[name] += loss.item() * batch_size
        metric_fn.update(predicts, masks)

    logs = {
        name: loss / data_size
        for name, loss in logs.items()
    }
    logs[config.metric.name] = metric_fn.compute_reset().item()

    return logs

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track, Table
    from src.metrics import get_metric
    from src.losses import get_loss
    from src.dataset import get_loader
    from collections import defaultdict
    from src.downloader import ensure_experiment_exists

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    ensure_experiment_exists(experiment_name)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))

    metric_fn = get_metric(config).to(config.device)
    criterion = get_loss(config).to(config.device)

    logs = {}
    for fold in range(1, config.num_folds + 1):
        config.fold = fold
        logs[f'Fold {fold}'] = evaluate_fold(config, criterion, metric_fn)

    titles = [*criterion.components, config.metric.name]
    Table(
        ['', *titles],
        *[
            [fold, *[log[title] for title in titles]]
            for fold, log in logs.items()
        ]
    ).display()

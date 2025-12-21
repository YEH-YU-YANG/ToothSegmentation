from torch.nn import Module
from torchmetrics.segmentation import MeanIoU

class mIoU(Module):
    def __init__(self, num_classes, predict_index=None):
        super().__init__()
        self.metric_fn = MeanIoU(num_classes, include_background=False, input_format='index')
        self.predict_index = predict_index
    def update(self, predicts, targets):
        if self.predict_index is not None:
            predicts = predicts[self.predict_index]
        predicts = predicts.argmax(1)
        self.metric_fn.update(predicts, targets)
    def compute_reset(self):
        metric = self.metric_fn.compute()
        self.metric_fn.reset()
        return metric

METRICS = {
    'mIoU': mIoU
}

def get_metric(config):
    return METRICS[config.metric.name](**config.metric.parameters)

if __name__ == '__main__':
    import torch

    from src.dataset import get_loader
    from src.models import get_model
    from src.utils import Table
    from src.config import load_config

    config = load_config('configs/unet.toml')
    config.fold = 1

    loader, _ = get_loader(config)
    model = get_model(config).to(config.device)

    for images, masks, _ in loader:
        images = images.to(config.device)
        masks = masks.to(config.device)
        break

    with torch.autocast(config.device):
        predicts = model(images)
    metric_fn = get_metric(config).to(config.device)
    metric_fn.update(predicts, masks)
    metric = metric_fn.compute_reset()

    Table(
        ['Metric', 'Value'],
        ['mIoU', metric]
    ).display()

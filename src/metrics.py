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
    return METRICS[config.METRIC_NAME](**config.METRIC_PARAMETERS)

if __name__ == '__main__':
    from .dataset import get_loader
    from .models import get_model
    from .utils import Table
    from .config import Config

    config = Config()
    config.MODEL_NAME = 'DeepUNet'
    config.MODEL_PARAMETERS = {
        'in_channels': 1,
        'num_classes': 3
    }
    config.METRIC_NAME = 'mIoU'
    config.METRIC_PARAMETERS = {
        'num_classes': 3,
        'predict_index': 0
    }
    config.BATCH_SIZE = 4
    config.FOLD = 1

    loader, _ = get_loader(config)
    model = get_model(config).to(config.DEVICE)

    for images, masks, _ in loader:
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        break

    predicts = model(images)
    metric_fn = get_metric(config).to(config.DEVICE)
    metric_fn.update(predicts, masks)
    metric = metric_fn.compute_reset()

    Table(
        ['Metric', 'Value'],
        ['mIoU', metric]
    ).display()

from torch import nn
from torch.nn.functional import one_hot

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, predicts, targets):
        predicts = predicts.softmax(1) # (B, C, H, W)
        predicts = predicts.permute(1, 0, 2, 3) # (C, B, H, W)
        predicts = predicts.reshape(self.num_classes, -1) # (C, BHW)

        targets = one_hot(targets, self.num_classes) # (B, H, W, C)
        targets = targets.permute(3, 0, 1, 2) # (C, B, H, W)
        targets = targets.reshape(self.num_classes, -1) # (C, BHW)

        # ignore background
        predicts = predicts[1:]
        targets = targets[1:]

        intersection = (predicts * targets).sum(1) # (C)
        predict_area = predicts.sum(1) # (C)
        target_area = targets.sum(1) # (C)

        dice_loss = 1 - (2 * intersection + 1e-6) / (predict_area + target_area + 1e-6)
        dice_loss = dice_loss.mean()
        return dice_loss

class MultipleLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, cross_entropy_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

        self.dice_loss = DiceLoss(num_classes)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.components = ['Total Loss', 'Dice Loss', 'Cross Entropy Loss']
    def forward(self, predicts, targets):
        dice_loss = self.dice_loss(predicts, targets)
        cross_entropy_loss = self.cross_entropy_loss(predicts, targets)
        loss = self.dice_weight * dice_loss + self.cross_entropy_weight * cross_entropy_loss
        return {
            'Total Loss': loss,
            'Dice Loss': dice_loss,
            'Cross Entropy Loss': cross_entropy_loss
        }

class DeepSupervisionLoss(MultipleLoss):
    def forward(self, predicts, targets):
        loss = {
            'Total Loss': 0,
            'Dice Loss': 0,
            'Cross Entropy Loss': 0
        }
        for side_predicts in predicts:
            loss_dict = super().forward(side_predicts, targets)
            for name, value in loss_dict.items():
                loss[name] += value
        return loss

LOSSES = {
    'MultipleLoss': MultipleLoss,
    'DeepSupervisionLoss': DeepSupervisionLoss
}

def get_loss(config):
    return LOSSES[config.loss.name](**config.loss.parameters)

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
    criterion = get_loss(config).to(config.device)

    for images, masks, _ in loader:
        images = images.to(config.device)
        masks = masks.to(config.device)
        break

    with torch.autocast(config.device):
        predicts = model(images)
        loss_dict = criterion(predicts, masks)

    Table(
        ['Loss', 'Value'],
        *[
            [name, loss]
            for name, loss in loss_dict.items()
        ]
    ).display()

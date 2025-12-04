import torch

from torch import nn
from torch.nn.functional import interpolate

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(in_channels, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512)
        ])
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.downsample(x)
        return x, features

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2)
        ])
        self.blocks = nn.ModuleList([
            Block(1024, 512),
            Block(512, 256),
            Block(256, 128),
            Block(128, 64)
        ])
    def forward(self, x, encoder_features):
        features = encoder_features[::-1]
        output_features = []
        for feature, upsample, block in zip(features, self.upsample, self.blocks):
            x = upsample(x)
            x = torch.cat([feature, x], dim=1)
            x = block(x)
            output_features.append(x)
        return output_features

class DeepSupervisionHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.side_classification = nn.ModuleList([
            nn.Conv2d(channel, num_classes, 1)
            for channel in channels
        ])
        self.fuse_classification = nn.Conv2d(num_classes * len(channels), num_classes, 1)
    def forward(self, *xs):
        side_outputs = []
        target_size = xs[-1].shape[2:]
        for x, side_classification in zip(xs, self.side_classification):
            x = side_classification(x)
            x = interpolate(x, target_size, mode='bilinear')
            side_outputs.append(x)
        side_outputs = side_outputs[::-1]
        x = torch.cat(side_outputs, dim=1)
        fuse_output = self.fuse_classification(x)
        return fuse_output, *side_outputs

class DeepUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Block(512, 1024)
        self.decoder = Decoder()
        self.head = DeepSupervisionHead([1024, 512, 256, 128, 64], num_classes)
    def forward(self, x):
        x, encoder_features = self.encoder(x)
        x = self.bottleneck(x)
        outputs = self.decoder(x, encoder_features)
        outputs = self.head(x, *outputs)
        return outputs

if __name__ == '__main__':
    from src.dataset import get_loader
    from src.config import Config
    from src.utils import Table

    config = Config()
    config.FOLD = 1
    config.BATCH_SIZE = 4

    loader, _ = get_loader(config)

    model = DeepUNet(in_channels=1, num_classes=3).to(config.DEVICE)

    for images, _, _ in loader:
        images = images.to(config.DEVICE)
        break

    predicts = model(images)

    Table(
        ['Item', 'Shape'],
        ['Input', images.shape],
        ['Output Fusion', predicts[0].shape],
        *[
            [f'Output {i}', predict.shape]
            for i, predict in enumerate(predicts[1:], start=1)
        ]
    ).display()

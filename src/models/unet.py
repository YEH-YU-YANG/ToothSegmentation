import torch

from torch import nn

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
        for feature, upsample, block in zip(features, self.upsample, self.blocks):
            x = upsample(x)
            x = torch.cat([feature, x], dim=1)
            x = block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.bottleneck = Block(512, 1024)
        self.decoder = Decoder()
        self.classification = nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x, encoder_features = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, encoder_features)
        x = self.classification(x)
        return x

if __name__ == '__main__':
    from src.dataset import get_loader
    from src.config import load_config

    config = load_config('configs/unet.toml')
    config.fold = 1

    loader, _ = get_loader(config)

    model = UNet(**config.model.parameters).to(config.device)

    for images, _, _ in loader:
        images = images.to(config.device)
        break

    with torch.autocast(config.device):
        predicts = model(images)

    print(f'{tuple(images.shape)} --{model.__class__.__name__}-> {tuple(predicts.shape)}')

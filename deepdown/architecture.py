import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        # Resnet part
        self.resnet1 = nn.Sequential(
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(256),
        )
        self.resnet2 = nn.Sequential(
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(256),
        )
        # Decoder part
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def forward(self, x):
        x = self.encoder(x)

        identity = x
        x = self.resnet1(x)
        x = x + identity
        identity = x
        x = self.resnet2(x)
        x = x + identity

        x = self.decoder(x)
        return x

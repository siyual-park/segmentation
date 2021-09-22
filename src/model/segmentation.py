import torch
from torch import nn

from src.model.cbam import CBAM
from src.model.common import Conv, Shortcut, Bottleneck, C3, ConvTranspose


class ResBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        down_channels = max(int(channels * expansion), 1)

        kernel_size = 3
        stride = 1

        self.block = Shortcut(
            module=Bottleneck(
                in_channels=channels,
                out_channels=channels,
                down_channels=down_channels,
                module=nn.Sequential(
                    Conv(
                        in_channels=down_channels,
                        out_channels=down_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout_prob=dropout_prob
                    ),
                    Conv(
                        in_channels=down_channels,
                        out_channels=down_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout_prob=dropout_prob
                    )
                )
            ),
            activate=True
        )

    def forward(self, x):
        x_out = self.block(x)

        return x_out


class Encoder(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        c3s = []
        current_channels = channels
        for i in range(deep):
            in_channels = current_channels
            out_channels = current_channels * 2

            c3 = C3(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=expansion,
                dropout_prob=dropout_prob
            )
            pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )

            c3s.append(nn.Sequential(
                c3,
                pool
            ))

            current_channels = out_channels

        self.c3s = nn.ModuleList(c3s)

    def forward(self, x):
        x_out = x
        x_outs = []
        for i, block in enumerate(self.c3s):
            x_out = block(x_out)
            x_outs.append(x_out)

        x_outs.reverse()
        return x_outs


class Decoder(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        c3s = []
        current_channels = channels * (2 ** deep)
        for i in range(deep):
            in_channels = current_channels
            out_channels = current_channels // 2

            c3 = C3(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=expansion,
                dropout_prob=dropout_prob
            )
            upsample = ConvTranspose(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0
            )

            c3s.append(nn.Sequential(
                c3,
                upsample
            ))

            current_channels = out_channels

        self.c3s = nn.ModuleList(c3s)

    def forward(self, x):
        x_out = x[0]
        for i, block in enumerate(self.c3s):
            x_out = block(x_out)
            if i < len(self.c3s) - 1:
                x_out += x[i + 1]

        return x_out


class Mask(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.attention = Shortcut(
            module=CBAM(
                gate_channels=3,
                dropout_prob=dropout_prob
            ),
            activate=True
        )
        self.up_scaling = Conv(
            in_channels=3,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dropout_prob=dropout_prob
        )
        self.down_scaling = Conv(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

        self.encoder = Encoder(
            channels=channels,
            deep=deep,
            expansion=expansion,
            dropout_prob=dropout_prob
        )
        self.decoder = Decoder(
            channels=channels,
            deep=deep,
            expansion=expansion,
            dropout_prob=dropout_prob
        )

    def forward(self, x):
        x_out = self.attention(x)
        x_out = self.up_scaling(x_out)

        x_out = self.encoder(x_out)
        x_out = self.decoder(x_out)

        x_out = self.down_scaling(x_out)

        x_out = torch.sigmoid(x_out)
        return x_out

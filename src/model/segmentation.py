import torch
from torch import nn

from src.model.cbam import CBAM
from src.model.common import Conv, Shortcut, Bottleneck


class Encoder(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        down_channels = max(int(channels * expansion), 1)

        kernel_size = 3
        stride = 1

        self.up_scaling = Conv(
            in_channels=3,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dropout_prob=dropout_prob
        )
        self.block = nn.Sequential(*[
            Shortcut(
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
                )
            ) for _ in range(deep)
        ])

    def forward(self, x):
        x_out = self.up_scaling(x)
        x_out = self.block(x_out)

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

        self.attention = CBAM(
            gate_channels=3,
            dropout_prob=dropout_prob
        )
        self.encoder = Encoder(
            channels=channels,
            deep=deep,
            expansion=expansion,
            dropout_prob=dropout_prob
        )
        self.decoder = Conv(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        x_out = self.attention(x)
        x_out = self.encoder(x_out)
        x_out = self.decoder(x_out)

        x_out = torch.sigmoid(x_out)
        return x_out

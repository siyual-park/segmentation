import torch
from torch import nn

from src.model.common import Conv, C3, ConvTranspose


class Encoder(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        blocks = []
        current_channels = channels
        for i in range(deep):
            in_channels = current_channels
            out_channels = current_channels * 2

            conv = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                dropout_prob=dropout_prob,
            )
            c3 = C3(
                in_channels=out_channels,
                out_channels=out_channels,
                expansion=expansion,
                dropout_prob=dropout_prob
            )

            blocks.append(nn.Sequential(
                conv,
                c3,
            ))

            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x_out = x
        x_outs = []
        for i, block in enumerate(self.blocks):
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

        blocks = []
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
            conv_transpose = ConvTranspose(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0
            )

            blocks.append(nn.Sequential(
                c3,
                conv_transpose
            ))

            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.adjust = C3(
            in_channels=channels,
            out_channels=channels,
            expansion=expansion,
            dropout_prob=dropout_prob
        )

    def forward(self, x):
        x_out = x[0]
        for i, block in enumerate(self.blocks):
            x_out = block(x_out)
            if i < len(self.blocks) - 1:
                x_out += x[i + 1]

        return self.adjust(x_out)


class Mask(nn.Module):
    def __init__(
            self,
            channels: int,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

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
        x_out = self.up_scaling(x)

        x_out = self.encoder(x_out)
        x_out = self.decoder(x_out)

        x_out = self.down_scaling(x_out)

        x_out = torch.sigmoid(x_out)
        return x_out

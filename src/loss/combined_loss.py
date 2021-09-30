from typing import Tuple

from torch import nn, Tensor

from src.loss.dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, weights: Tuple[float, float] = (0.5, 1.0), smooth: float = 0.0, eps: float = 1e-7):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth, eps)

        self.weights = weights

    def forward(self, input: Tensor, target: Tensor):
        bce_loss = self.bce_loss(input, target)
        dice_loss = self.dice_loss(input, target)

        bce_loss = bce_loss * self.weights[0]
        dice_loss = dice_loss * self.weights[1]

        return bce_loss + dice_loss

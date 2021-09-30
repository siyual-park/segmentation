from torch import nn, Tensor


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 0.0, eps: float = 1e-7):
        super().__init__()

        self.smooth = smooth
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor):
        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
            pred: tensor with first dimension as batch
            target: tensor with first dimension as batch
            """

        # have to use contiguous since they may from a torch.view op
        batch_size = target.size()[0]

        iflat = input.contiguous().view(batch_size, -1)
        tflat = target.contiguous().view(batch_size, -1)

        intersection = (iflat * tflat).sum(dim=1)
        cardinality = (iflat + tflat).sum(dim=1)

        loss = 1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps))

        return loss.mean()

def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    batch_size = target.size(0)

    iflat = pred.contiguous().view(batch_size, -1)
    tflat = target.contiguous().view(batch_size, -1)
    intersection = (iflat * tflat).sum(dim=1)

    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth))

    return loss.sum()

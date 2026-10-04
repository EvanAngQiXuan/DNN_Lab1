'''Evaluation metrics for binary segmentation.'''
import torch


def dice_coefficient(probs, mask, tau=0.5, eps=1e-7):
    '''Per-image Dice between thresholded predictions and ground truth.

    probs: [B, 1, H, W] probabilities in [0, 1] (i.e. sigmoid(logits))
    mask:  [B, 1, H, W] ground truth in {0, 1}
    returns: [B] tensor of Dice coefficients
    '''
    pred = (probs >= tau).float()
    mask = mask.float()
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * mask).sum(dim=dims)
    denom = pred.sum(dim=dims) + mask.sum(dim=dims)
    return (2. * intersection + eps) / (denom + eps)

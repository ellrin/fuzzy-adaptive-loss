import torch
import torch.nn.functional as F

from config import LINEAR_LAMBDA
from utils.fuzzy import compute_fuzzy_weight, compute_linear_weight


def dice_loss_fn(pred, target, eps=1e-5):
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1 - (2 * intersection + eps) / (union + eps)


def bce_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction='none').mean(dim=(1, 2, 3))


def base_loss_fn(pred, target, loss_type='dice'):
    if loss_type == 'dice':
        return dice_loss_fn(pred, target)
    if loss_type == 'bce':
        return bce_loss_fn(pred, target)
    raise ValueError(f'unknown loss_type: {loss_type}')


def adaptive_weights(target, quality_probs, weight_mode='choquet'):
    if weight_mode == 'choquet':
        return [compute_fuzzy_weight(quality_probs[b], target[b, 0]) for b in range(target.shape[0])]
    if weight_mode == 'linear':
        return [compute_linear_weight(quality_probs[b], target[b, 0], LINEAR_LAMBDA) for b in range(target.shape[0])]
    raise ValueError(f'unknown weight_mode: {weight_mode}')


def loss_fn(pred, target, quality_probs, loss_type='dice', weight_mode='choquet'):
    per_image_loss = base_loss_fn(pred, target, loss_type)

    if weight_mode == 'none':
        return per_image_loss.mean()

    weights = torch.tensor(adaptive_weights(target, quality_probs, weight_mode),
                           dtype=per_image_loss.dtype, device=per_image_loss.device)
    return (per_image_loss * weights).mean()

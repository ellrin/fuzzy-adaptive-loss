import numpy as np


def IOU(pred, label, epsilon=1e-6):
    intersection = (pred * label).sum()
    union = (pred + label).sum() - intersection
    return (intersection + epsilon) / (union + epsilon)


def DICE(pred, label, epsilon=1e-6):
    intersection = (pred * label).sum()
    return (2 * intersection + epsilon) / ((pred + label).sum() + epsilon)


def PRECISION(pred, label, epsilon=1e-6):
    intersection = (pred * label).sum()
    return (intersection + epsilon) / (pred.sum() + epsilon)


def RECALL(pred, label, epsilon=1e-6):
    intersection = (pred * label).sum()
    return (intersection + epsilon) / (label.sum() + epsilon)


def get_results(preds, labels):
    iou, dice, precision, recall = [], [], [], []
    for pred, label in zip(preds, labels):
        iou.append(IOU(pred, label))
        dice.append(DICE(pred, label))
        precision.append(PRECISION(pred, label))
        recall.append(RECALL(pred, label))
    return np.mean(iou), np.mean(dice), np.mean(precision), np.mean(recall)

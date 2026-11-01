"""Metrics with explicit image-level and dataset-level aggregation."""

import numpy as np
import torch


def binary_counts(prediction, target):
    return {
        "TP": int((prediction & target).sum()),
        "FP": int((prediction & ~target).sum()),
        "FN": int((~prediction & target).sum()),
        "TN": int((~prediction & ~target).sum()),
    }


def binary_metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))
    return {
        "precision_percent": 100 * tp / max(tp + fp, 1),
        "recall_percent": 100 * tp / max(tp + fn, 1),
        "f1_percent": 200 * tp / max(2 * tp + fp + fn, 1),
        "iou_percent": 100 * tp / max(tp + fp + fn, 1),
        "accuracy_percent": 100 * (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def normalized_probability(logits):
    probability = logits.sigmoid()[0, 0].cpu().numpy()
    return (probability - probability.min()) / (np.ptp(probability) + 1e-8)


def nyu_depth_metrics(prediction, target):
    valid = (target > 0.001) & (target < 10.0)
    crop = torch.zeros_like(valid)
    crop[45:471, 41:601] = True
    valid &= crop
    p, y = prediction[valid].clamp(0.001, 10.0), target[valid]
    threshold = torch.maximum(y / p, p / y)
    difference, log_difference = p - y, torch.log(p) - torch.log(y)
    return {
        "d1": (threshold < 1.25).float().mean().item(),
        "d2": (threshold < 1.25**2).float().mean().item(),
        "d3": (threshold < 1.25**3).float().mean().item(),
        "abs_rel": (difference.abs() / y).mean().item(),
        "sq_rel": (difference.square() / y).mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "rmse_log": log_difference.square().mean().sqrt().item(),
        "log10": (p.log10() - y.log10()).abs().mean().item(),
        "silog": (log_difference.square().mean() - 0.5 * log_difference.mean().square())
        .sqrt()
        .item(),
    }

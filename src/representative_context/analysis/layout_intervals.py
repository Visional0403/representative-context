"""Paired uncertainty for fixed spatial-layout comparisons."""

from pathlib import PurePosixPath
import re
import numpy as np


def sufficient(counts):
    counts = np.asarray(counts, dtype=np.float64)
    true_positive = np.diagonal(counts, axis1=-2, axis2=-1)
    union = counts.sum(-2) + counts.sum(-1) - true_positive
    return np.concatenate((true_positive, union), axis=-1)


def miou(statistics):
    true_positive, union = np.split(np.asarray(statistics, dtype=np.float64), 2, axis=-1)
    return 100 * np.nanmean(np.divide(true_positive, union,
                                     out=np.full_like(true_positive, np.nan), where=union > 0), axis=-1)


def paired_interval(reference, candidate, images, grouping, repeats=5000, seed=42,
                    draw_method="multinomial"):
    """Recalculate pooled mIoU after paired image or source-group resampling."""
    if draw_method not in {"multinomial", "integers"}:
        raise ValueError("Choose multinomial counts or integer-index draws.")
    first, second = sufficient(reference), sufficient(candidate)
    if len(first) != len(second) or len(first) != len(images):
        raise ValueError("Paired counts and image identities must have equal lengths.")
    rng = np.random.default_rng(seed)

    def sample_counts(count):
        if draw_method == "multinomial":
            return rng.multinomial(count, np.full(count, 1 / count), size=repeats)
        indices = rng.integers(0, count, size=(repeats, count))
        offsets = count * np.arange(repeats)[:, None]
        return np.bincount((indices + offsets).ravel(),
                           minlength=repeats * count).reshape(repeats, count)
    if grouping == "images":
        count = len(first)
        draws = sample_counts(count)
        changes = miou(draws @ second) - miou(draws @ first)
    else:
        normalized = [str(image).replace("\\", "/") for image in images]
        if grouping == "sequences":
            units = [re.search(r"(?:^|/)(seq[^/]+)(?:/|$)", image)[1] for image in normalized]
        elif grouping == "cities":
            units = [PurePosixPath(image).stem.split("_")[0] for image in normalized]
        else:
            raise ValueError("Choose images, sequences or cities for paired resampling.")
        labels = sorted(set(units))
        units = np.asarray(units)
        if grouping == "sequences":
            a = np.stack([first[units == value].sum(0) for value in labels])
            b = np.stack([second[units == value].sum(0) for value in labels])
            count = len(labels)
            draws = sample_counts(count)
            changes = miou(draws @ b) - miou(draws @ a)
        else:
            a = np.zeros((repeats, first.shape[1]))
            b = np.zeros_like(a)
            for value in labels:
                selected = units == value
                count = int(selected.sum())
                draws = sample_counts(count)
                a += draws @ first[selected]
                b += draws @ second[selected]
            changes = miou(b) - miou(a)
    return dict(change_points=float(miou(second.sum(0)) - miou(first.sum(0))),
                interval95=np.quantile(changes, (0.025, 0.975)).tolist(),
                repeats=repeats, seed=seed, grouping=grouping, draw_method=draw_method)

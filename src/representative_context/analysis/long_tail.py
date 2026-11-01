"""Paired prediction transitions and fixed difficult-region statistics."""

import math
import numpy as np

GAP_BOUNDARIES = (0.05, 0.10, 0.20, 0.50)
FLOW_TOLERANCE = 0.001
FLOW_RAW_KEYS = (
    "pixels",
    "dense_epe_sum",
    "sparse_epe_sum",
    "increase_sum",
    "decrease_sum",
    "strictly_worse_pixels",
    "strictly_better_pixels",
    "unchanged_pixels",
    "worse_beyond_tolerance",
    "better_beyond_tolerance",
    "within_tolerance",
    "dense_bad_pixels",
    "sparse_bad_pixels",
    "good_to_bad",
    "bad_to_good",
)


def gap_pair_counts(labels, reference, candidate, bins):
    labels, reference, candidate, bins = [
        np.asarray(value).reshape(-1) for value in (labels, reference, candidate, bins)
    ]
    valid = (labels >= 0) & (labels < 8)
    encoded = (labels[valid].astype(np.int64) * 8 + reference[valid]) * 8 + candidate[
        valid
    ]
    encoded += 8**3 * bins[valid].astype(np.int64)
    return np.bincount(encoded, minlength=5 * 8**3).reshape(5, 8, 8, 8)


def confusion_metrics(hist):
    hist = np.asarray(hist, dtype=np.int64)
    truth, predicted = hist.sum(1), hist.sum(0)
    tp = hist.diagonal()
    union = truth + predicted - tp
    iou = np.divide(tp, union, out=np.full(8, np.nan), where=union > 0)
    total = int(hist.sum())
    ratio = lambda a, b: float(100 * a / b) if b else None
    return dict(
        valid_pixels=total,
        miou_percent=float(np.nanmean(iou) * 100) if total else None,
        per_class_iou_percent=[
            float(value * 100) if np.isfinite(value) else None for value in iou
        ],
        human_iou_percent=float(iou[7] * 100) if union[7] else None,
        human_recall_percent=ratio(tp[7], truth[7]),
        human_precision_percent=ratio(tp[7], predicted[7]),
        human_gt_pixels=int(truth[7]),
        human_gt_percent=ratio(truth[7], total),
        human_tp=int(tp[7]),
        human_fp=int(predicted[7] - tp[7]),
        human_fn=int(truth[7] - tp[7]),
    )


def classification(cube):
    cube = np.asarray(cube, dtype=np.int64)
    dense, sparse = confusion_metrics(cube.sum(2)), confusion_metrics(cube.sum(1))
    gt, left, right = np.indices(cube.shape)
    new = int(cube[(gt == left) & (gt != right)].sum())
    corrected = int(cube[(gt != left) & (gt == right)].sum())
    changed = int(cube[left != right].sum())
    total = dense["valid_pixels"]
    return dict(
        dense=dense,
        sparse=sparse,
        new_errors=new,
        corrections=corrected,
        wrong_class_switches=changed - new - corrected,
        changed_labels=changed,
        accuracy_change_pp=100 * (corrected - new) / total if total else None,
        human_iou_change_pp=(
            sparse["human_iou_percent"] - dense["human_iou_percent"]
            if dense["human_iou_percent"] is not None
            and sparse["human_iou_percent"] is not None
            else None
        ),
    )


def accuracy_interval(counters, groups, repeats=10000, seed=20261006):
    grouped = {}
    for row, group in zip(counters, groups):
        grouped.setdefault(group, np.zeros(3, dtype=np.int64))
        grouped[group] += row
    values = np.stack(list(grouped.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for begin in range(0, repeats, 256):
        draw = rng.integers(0, len(values), (min(256, repeats - begin), len(values)))
        sums = values[draw].sum(1)
        good = sums[:, 0] > 0
        samples.extend((100 * (sums[good, 2] - sums[good, 1]) / sums[good, 0]).tolist())
    return dict(
        groups=len(values),
        repeats=repeats,
        seed=seed,
        ci95=np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
    )


def human_iou_interval(cubes, groups, indices, gap=None, repeats=10000, seed=20261006):
    sequences = sorted({groups[index] for index in indices})
    if len(sequences) < 2:
        return dict(groups=len(sequences), ci95=None)
    totals = []
    for sequence in sequences:
        values = cubes[[index for index in indices if groups[index] == sequence]]
        totals.append(values.sum((0, 1)) if gap is None else values[:, gap].sum(0))
    totals = np.stack(totals)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sequences), size=(repeats, len(sequences)))
    dense = totals.sum(3)[draws].sum(1)
    sparse = totals.sum(2)[draws].sum(1)

    def iou(hist):
        tp = hist[:, 7, 7]
        union = hist[:, 7, :].sum(1) + hist[:, :, 7].sum(1) - tp
        return np.divide(
            100.0 * tp, union, out=np.full(repeats, np.nan), where=union > 0
        )

    changes = iou(sparse) - iou(dense)
    valid = changes[np.isfinite(changes)]
    return dict(
        groups=len(sequences),
        sequences=sequences,
        repeats=repeats,
        seed=seed,
        ci95=np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
        valid_repeats=len(valid),
    )


def human_quintiles(values, correct_only=False):
    margins, dense = values["dense_margin"], values["dense_prediction"]
    selected = dense == 7 if correct_only else np.ones(len(dense), dtype=bool)
    margins, dense = margins[selected], dense[selected]
    candidates = {
        key: values[key][selected] for key in ("k14_prediction", "k16_prediction")
    }
    if not len(dense):
        return dict(pixels=0, bins=[])
    thresholds = np.quantile(margins, [0.2, 0.4, 0.6, 0.8])
    bins = np.searchsorted(thresholds, margins, side="right")
    rows = []
    for group in range(5):
        chosen = bins == group
        n = int(chosen.sum())
        reference = dense[chosen] == 7
        row = dict(
            bin=group + 1,
            pixels=n,
            dense_recall_percent=float(reference.mean() * 100) if n else None,
            min_margin=float(margins[chosen].min()) if n else None,
            max_margin=float(margins[chosen].max()) if n else None,
            sparse={},
        )
        for key, value in candidates.items():
            prediction = value[chosen] == 7
            lost = int((reference & ~prediction).sum())
            recovered = int((~reference & prediction).sum())
            row["sparse"][key.removesuffix("_prediction")] = dict(
                recall_percent=float(prediction.mean() * 100) if n else None,
                recall_change_pp=(
                    float((prediction.mean() - reference.mean()) * 100) if n else None
                ),
                new_misses=lost,
                recovered_detections=recovered,
                new_miss_percent=100 * lost / n if n else None,
                recovered_percent=100 * recovered / n if n else None,
            )
        rows.append(row)
    return dict(
        pixels=len(dense),
        thresholds=thresholds.tolist(),
        bins=rows,
        correct_only=correct_only,
    )


def flow_derive(raw):
    result = dict(raw)
    n = int(raw["pixels"])
    fields = (
        "dense_epe",
        "sparse_epe",
        "epe_change",
        "mean_increase",
        "mean_decrease",
        "worse_percent",
        "better_percent",
        "within_tolerance_percent",
        "dense_bad_percent",
        "sparse_bad_percent",
        "bad_rate_change_pp",
        "decrease_over_increase_percent",
        "relative_epe_change_percent",
    )
    if not n:
        result.update({key: None for key in fields})
        return result
    dense, sparse = raw["dense_epe_sum"] / n, raw["sparse_epe_sum"] / n
    result.update(
        dense_epe=dense,
        sparse_epe=sparse,
        epe_change=sparse - dense,
        mean_increase=raw["increase_sum"] / n,
        mean_decrease=raw["decrease_sum"] / n,
        worse_percent=100 * raw["worse_beyond_tolerance"] / n,
        better_percent=100 * raw["better_beyond_tolerance"] / n,
        within_tolerance_percent=100 * raw["within_tolerance"] / n,
        dense_bad_percent=100 * raw["dense_bad_pixels"] / n,
        sparse_bad_percent=100 * raw["sparse_bad_pixels"] / n,
        bad_rate_change_pp=100
        * (raw["sparse_bad_pixels"] - raw["dense_bad_pixels"])
        / n,
        decrease_over_increase_percent=(
            100 * raw["decrease_sum"] / raw["increase_sum"]
            if raw["increase_sum"]
            else None
        ),
        relative_epe_change_percent=100 * (sparse - dense) / dense if dense else None,
    )
    assert np.isclose(
        sparse - dense,
        (raw["increase_sum"] - raw["decrease_sum"]) / n,
        rtol=1e-9,
        atol=1e-10,
    )
    assert (
        raw["good_to_bad"] - raw["bad_to_good"]
        == raw["sparse_bad_pixels"] - raw["dense_bad_pixels"]
    )
    return result


def paired_flow(dense_error, sparse_error, magnitude, mask):
    dense, sparse, magnitude = [
        np.asarray(value, dtype=float)[mask]
        for value in (dense_error, sparse_error, magnitude)
    ]
    assert all(np.isfinite(value).all() for value in (dense, sparse, magnitude))
    delta = sparse - dense
    dense_bad = (dense > 3) & (dense > 0.05 * magnitude)
    sparse_bad = (sparse > 3) & (sparse > 0.05 * magnitude)
    return flow_derive(
        dict(
            pixels=len(delta),
            dense_epe_sum=float(dense.sum()),
            sparse_epe_sum=float(sparse.sum()),
            increase_sum=float(np.maximum(delta, 0).sum()),
            decrease_sum=float(np.maximum(-delta, 0).sum()),
            strictly_worse_pixels=int((delta > 0).sum()),
            strictly_better_pixels=int((delta < 0).sum()),
            unchanged_pixels=int((delta == 0).sum()),
            worse_beyond_tolerance=int((delta > FLOW_TOLERANCE).sum()),
            better_beyond_tolerance=int((delta < -FLOW_TOLERANCE).sum()),
            within_tolerance=int((np.abs(delta) <= FLOW_TOLERANCE).sum()),
            dense_bad_pixels=int(dense_bad.sum()),
            sparse_bad_pixels=int(sparse_bad.sum()),
            good_to_bad=int((~dense_bad & sparse_bad).sum()),
            bad_to_good=int((dense_bad & ~sparse_bad).sum()),
        )
    )


def flow_masks(dense, target, occlusion):
    valid = np.isfinite(target).all(-1) & (np.abs(target) < 1e9).all(-1)
    baseline = np.linalg.norm(dense.astype(float) - target.astype(float), axis=-1)
    magnitude = np.linalg.norm(target.astype(float), axis=-1)
    threshold = float(np.quantile(baseline[valid], 0.8))
    masks = dict(
        all_pixels=valid,
        top_dense_error_quintile=valid & (baseline >= threshold),
        occluded=valid & occlusion,
        unoccluded=valid & ~occlusion,
        motion_at_least40=valid & (magnitude >= 40),
        motion10_to40=valid & (magnitude >= 10) & (magnitude < 40),
        motion_below10=valid & (magnitude < 10),
    )
    return baseline, magnitude, masks, threshold


def flow_combine(rows):
    return flow_derive({key: sum(row[key] for row in rows) for key in FLOW_RAW_KEYS})


def flow_interval(rows, groups, repeats=10000, seed=20261007):
    grouped = {}
    for row, group in zip(rows, groups):
        grouped.setdefault(group, np.zeros(3, dtype=float))
        grouped[group] += (row["pixels"], row["dense_epe_sum"], row["sparse_epe_sum"])
    values = np.stack(list(grouped.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for begin in range(0, repeats, 256):
        draw = rng.integers(0, len(values), (min(256, repeats - begin), len(values)))
        sums = values[draw].sum(1)
        good = sums[:, 0] > 0
        samples.extend(((sums[good, 2] - sums[good, 1]) / sums[good, 0]).tolist())
    return dict(
        groups=len(values),
        repeats=repeats,
        seed=seed,
        ci95=np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
    )

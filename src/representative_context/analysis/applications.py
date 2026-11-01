"""Source-group paired bootstrap and fitted application-readout relationships."""

import numpy as np


def fit_readout(records):
    truth = np.array([row["gt_readout"] for row in records], dtype=np.float64)
    result = {}
    for label, field in (("dense", "dense_readout"), ("sparse", "sparse_readout")):
        prediction = np.array([row[field] for row in records], dtype=np.float64)
        centered = truth - truth.mean()
        slope = float(
            np.dot(centered, prediction - prediction.mean())
            / np.dot(centered, centered)
        )
        intercept = float(prediction.mean() - slope * truth.mean())
        fitted = slope * truth + intercept
        sse = float(np.square(prediction - fitted).sum())
        sst = float(np.square(prediction - prediction.mean()).sum())
        result[label] = dict(
            slope=slope,
            intercept=intercept,
            r_squared=1 - sse / sst,
            sse=sse,
            sst=sst,
            mae=float(np.abs(prediction - truth).mean()),
        )
    return result


def paired_statistics(application):
    rows = application["samples"]
    settings = application["statistics"]
    gt, dense, sparse = (
        np.array([row[name] for row in rows], dtype=float)
        for name in ("gt_readout", "dense_readout", "sparse_readout")
    )
    if not np.isfinite(np.stack([gt, dense, sparse])).all():
        raise ValueError("Application readouts must be finite.")
    groups = {str(row["id"]): str(row["group"]) for row in settings["sample_groups"]}
    labels = [groups[str(row["id"])] for row in rows]
    unique, inverse = np.unique(labels, return_inverse=True)
    delta = np.abs(sparse - gt) - np.abs(dense - gt)
    moments = np.stack(
        [
            np.ones(len(gt)),
            gt,
            gt * gt,
            dense,
            sparse,
            gt * dense,
            gt * sparse,
            delta,
            sparse - dense,
        ],
        axis=1,
    )
    sums = np.zeros((len(unique), 9))
    np.add.at(sums, inverse, moments)
    rng = np.random.default_rng(settings["seed"])
    repeats = settings["bootstrap_repeats"]
    if repeats % 250:
        raise ValueError("The paired bootstrap uses complete batches of 250 draws.")
    boot = np.concatenate(
        [
            sums[rng.integers(0, len(unique), (250, len(unique)))].sum(axis=1)
            for _ in range(repeats // 250)
        ]
    )
    n, sx, sxx = boot[:, 0], boot[:, 1], boot[:, 2]
    denominator = sxx - sx * sx / n
    if np.any(denominator <= 0):
        raise ValueError("A resampled regression has zero reference-readout variance.")
    xs = np.linspace(gt.min(), gt.max(), 200)
    bands = {}
    for label, iy, ixy in (("dense", 3, 5), ("sparse", 4, 6)):
        slope = (boot[:, ixy] - sx * boot[:, iy] / n) / denominator
        intercept = boot[:, iy] / n - slope * sx / n
        bands[label] = np.percentile(
            slope[:, None] * xs + intercept[:, None], [2.5, 97.5], axis=0
        ).tolist()
    return dict(
        samples=len(rows),
        source_groups=len(unique),
        bootstrap_repeats=repeats,
        seed=settings["seed"],
        gt=gt.tolist(),
        dense=dense.tolist(),
        sparse=sparse.tolist(),
        sparse_minus_dense=(sparse - dense).tolist(),
        absolute_error_change=delta.tolist(),
        mean_absolute_error_change=float(delta.mean()),
        mean_absolute_error_change_ci95=np.percentile(
            boot[:, 7] / n, [2.5, 97.5]
        ).tolist(),
        mean_readout_change_ci95=np.percentile(boot[:, 8] / n, [2.5, 97.5]).tolist(),
        readout_change_central95=np.percentile(sparse - dense, [2.5, 97.5]).tolist(),
        dense_mae=float(np.abs(dense - gt).mean()),
        sparse_mae=float(np.abs(sparse - gt).mean()),
        mean_absolute_readout_change=float(np.abs(sparse - dense).mean()),
        fit_x=xs.tolist(),
        fit_mean_pointwise_ci95=bands,
        sample_groups=settings["sample_groups"],
        fit=fit_readout(rows),
    )

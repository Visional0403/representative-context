"""Class-margin certificates calculated from paired complete-window scores."""

import numpy as np


def certificate_fields(dense, sparse, labels):
    """Return the strict per-pixel predicate inputs using float64 arithmetic."""
    dense, sparse = np.asarray(dense), np.asarray(sparse)
    if dense.shape != sparse.shape or dense.shape[:-1] != np.asarray(labels).shape:
        raise ValueError("Paired scores and reference labels must share their spatial grid.")
    classes = dense.shape[-1]
    z = dense.reshape(-1, classes).astype(np.float64)
    error = sparse.reshape(-1, classes).astype(np.float64) - z
    error -= error.mean(axis=1, keepdims=True)
    energy = np.einsum("ij,ij->i", error, error)
    sorted_scores = np.partition(z, -2, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    valid = ((np.asarray(labels) >= 0) & (np.asarray(labels) < classes)).reshape(-1)
    if not np.isfinite(energy[valid]).all() or not np.isfinite(margin[valid]).all():
        raise ValueError("Certificate scores must be finite at valid reference pixels.")
    return dict(centered_error_energy=energy, minimum_dense_margin=margin, valid_mask=valid)


def certificate_counts(centered_error_energy, minimum_dense_margin, valid_mask):
    """Pool the exact predicate 2*energy < margin**2 on the fixed valid mask."""
    energy, margin, valid = (np.asarray(x) for x in
                            (centered_error_energy, minimum_dense_margin, valid_mask))
    if energy.shape != margin.shape or energy.shape != valid.shape or valid.dtype != np.bool_:
        raise ValueError("Certificate inputs require equal shapes and a Boolean valid mask.")
    if np.any(energy[valid] < 0) or np.any(margin[valid] < 0):
        raise ValueError("Squared energy and winning margins must be nonnegative.")
    certified = valid & (2 * energy < np.square(margin))
    count, total = int(certified.sum()), int(valid.sum())
    if total == 0:
        raise ValueError("The certificate requires valid reference pixels.")
    return dict(certified_pixels=count, valid_pixels=total,
                certified_percent=100 * count / total)

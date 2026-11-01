"""Class-conditioned attention content and fixed-norm restoration interventions."""

import torch
from torch.nn import functional as F

from representative_context.adapters.selection import extract_qkv
from representative_context.layouts import make_ranks

PATTERNS = ("dispersed32", "clustered16")
ARMS = (
    "sparse",
    "class_mass",
    "class_content",
    "content_norm_matched",
    "restore_aggregation",
)


def conditional(probabilities, values, labels, classes=8):
    """Return class probability mass and its value-weighted numerator per query."""
    indicator = F.one_hot(labels.long(), classes).to(values.dtype)
    mass = probabilities @ indicator
    numerator = torch.stack(
        [
            (probabilities * indicator[:, category]) @ values
            for category in range(classes)
        ],
        dim=2,
    )
    return mass, numerator


def conditional_mean(mass, numerator, fallback):
    denominator = torch.where(mass > 0, mass, 1)
    mean = numerator / denominator.unsqueeze(-1)
    return torch.where((mass > 0).unsqueeze(-1), mean, fallback)


def project_heads(value, projection):
    """Project a difference; the affine bias cancels in paired differences."""
    heads, queries, dim = value.shape
    return F.linear(
        value.permute(1, 0, 2).reshape(1, queries, heads * dim),
        projection.weight,
        bias=None,
    )


def match_total_norm(value, reference):
    """Match the full tensor Frobenius norm while retaining its direction."""
    denominator = torch.linalg.vector_norm(value.double())
    target = torch.linalg.vector_norm(reference.double())
    if float(denominator) == 0:
        if float(target) != 0:
            raise ValueError("A zero direction cannot be scaled to a nonzero error.")
        return value.clone(), 1.0
    scale = float(target / denominator)
    return value * scale, scale


def factors(core, x, labels, side, classes, patterns=PATTERNS):
    q, k, v, kh, kw = extract_qkv(core, x, side, side)
    q, k, v = q[0], k[0], v[0]
    heads, count, dim = q.shape
    if (kh, kw) != (side, side):
        raise ValueError("Conditional restoration uses equal query and KV grids.")
    old_dense, old_pattern = core.dense, core.pattern

    def actual(pattern):
        core.dense = pattern is None
        if pattern is not None:
            core.pattern = pattern
        return core(x, side, side)

    try:
        originals = {pattern: actual(pattern) for pattern in patterns}
        dense = actual(None)
    finally:
        core.dense, core.pattern = old_dense, old_pattern
    plan = core.last_plan
    ids = {
        pattern: torch.tensor(
            make_ranks(side, side, plan.capacity_limit, pattern, core.seed),
            device=x.device,
            dtype=torch.int64,
        ).remainder(plan.groups)
        for pattern in patterns
    }
    parts = {pattern: [[], []] for pattern in ("dense", *patterns)}
    for start in range(0, count, 128):
        scores = (q[:, start : start + 128] @ k.transpose(1, 2)) * dim**-0.5
        for pattern in parts:
            mask = (
                None
                if pattern == "dense"
                else ids[pattern][start : start + 128, None] == ids[pattern][None, :]
            )
            probability = (
                scores if mask is None else scores.masked_fill(~mask[None], -torch.inf)
            ).softmax(-1)
            mass, numerator = conditional(probability, v, labels, classes)
            parts[pattern][0].append(mass)
            parts[pattern][1].append(numerator)
    parts = {
        pattern: [torch.cat(a, 1), torch.cat(b, 1)] for pattern, (a, b) in parts.items()
    }
    dm, dn = parts["dense"]
    dense_mean = conditional_mean(dm, dn, torch.zeros_like(dn))
    reconstructed = project_heads(dn.sum(2), core.out_proj) + core.out_proj.bias
    assert float((reconstructed - dense).abs().max()) < 2e-4
    common = torch.ones_like(dm, dtype=torch.bool)
    for pattern in patterns:
        common &= parts[pattern][0] > 0
    weights = dm * common
    candidates, details = {}, {
        "common_support_mass": float(weights.sum(-1).mean()),
        "layouts": {},
    }
    for pattern in patterns:
        mass, numerator = parts[pattern]
        mean = conditional_mean(mass, numerator, dense_mean)
        quota = originals[pattern] + project_heads(
            torch.einsum("hqc,hqcd->hqd", dm, mean) - numerator.sum(2), core.out_proj
        )
        content = originals[pattern] + project_heads(
            torch.einsum("hqc,hqcd->hqd", mass, dense_mean) - numerator.sum(2),
            core.out_proj,
        )
        normalized, scale = match_total_norm(
            content - dense, originals[pattern] - dense
        )
        candidates[pattern] = dict(
            sparse=originals[pattern],
            class_mass=quota,
            class_content=content,
            content_norm_matched=dense + normalized,
            restore_aggregation=dense,
        )
        error = (mean.double() - dense_mean.double()).square().sum(-1)
        original_norm = float((originals[pattern] - dense).double().norm())
        matched_norm = float(
            (candidates[pattern]["content_norm_matched"] - dense).double().norm()
        )
        assert abs(matched_norm - original_norm) / max(original_norm, 1e-12) < 2e-6
        direct = project_heads(numerator.sum(2), core.out_proj) + core.out_proj.bias
        reconstruction_error = float((direct - originals[pattern]).abs().max())
        assert reconstruction_error < 2e-4
        details["layouts"][pattern] = dict(
            within_class_rms=float(
                (error * weights.double()).sum() / weights.double().sum() / dim
            )
            ** 0.5,
            class_mass_tv=float((mass - dm).abs().sum(-1).mean() * 0.5),
            reconstruction_max_abs_error=reconstruction_error,
            sparse_error_norm=original_norm,
            matched_error_norm=matched_norm,
            matching_scale=scale,
        )
    return candidates, details


def task_counts(gt, dense, current, classes):
    import numpy as np

    valid = gt < classes
    code = (
        classes**2 * gt[valid].astype(np.int64)
        + classes * dense[valid].astype(np.int64)
        + current[valid]
    )
    return np.bincount(code, minlength=classes**3).reshape(classes, classes, classes)


def metrics(cube):
    import numpy as np

    classes = len(cube)
    gt, dense, current = np.indices(cube.shape)
    total = int(cube.sum())
    confusion = cube.sum(1)
    tp = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - tp
    return dict(
        pixels=total,
        label_changes=int(cube[dense != current].sum()),
        new_errors=int(cube[(gt == dense) & (gt != current)].sum()),
        corrections=int(cube[(gt != dense) & (gt == current)].sum()),
        wrong_class_switches=int(
            cube[(gt != dense) & (gt != current) & (dense != current)].sum()
        ),
        correct_pixels=int(cube[gt == current].sum()),
        miou_percent=float(np.mean(tp[union > 0] / union[union > 0]) * 100),
        counts=cube.tolist(),
    )

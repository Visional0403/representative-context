"""Motion-context components and recurrent content-restoration interventions."""

import torch
from torch import nn
from torch.nn import functional as F
from representative_context.layouts.spatial import (
    _balanced_ranks,
    _morton,
    coverage_ranks,
    point_coverage_ranks,
)
from representative_context.adapters.motion import (
    MotionAggregation,
    MotionAttention,
    SparseAttentionMap,
)

ARMS = (
    "sparse",
    "restore_motion",
    "restore_matching",
    "matching_norm_matched",
    "restore_aggregation",
)
PATTERNS = ("dispersed32", "clustered16")


def native_cover_ranks(height, width, groups, seed=42):
    """Balance contiguous two-by-two tiles within sixteen spatial regions."""
    if height % 8 == 0 and width % 8 == 0:
        return coverage_ranks(height, width, groups, 16, seed)
    if min(height, width) < 4 or groups < 1:
        raise ValueError(
            "The rectangular layout requires a valid four-by-four partition."
        )
    members = [[] for _ in range(groups)]
    for ry in range(4):
        y0, y1 = (ry * height // 4, (ry + 1) * height // 4)
        for rx in range(4):
            x0, x1 = (rx * width // 4, (rx + 1) * width // 4)
            tiles = sorted(
                ((y, x) for y in range(0, y1 - y0, 2) for x in range(0, x1 - x0, 2)),
                key=lambda item: _morton(item[0] // 2, item[1] // 2),
            )
            common, remainder = divmod(len(tiles), groups)
            if common == 0:
                raise ValueError(
                    "Each region must contain at least one tile per group."
                )
            cursor = 0
            rotation = (ry * 4 + rx) * (seed % groups + 1) % groups
            for segment in range(groups):
                count = common + (segment < remainder)
                target = (segment + rotation) % groups
                for ly, lx in tiles[cursor : cursor + count]:
                    members[target].extend(
                        (
                            y * width + x
                            for y in range(y0 + ly, min(y0 + ly + 2, y1))
                            for x in range(x0 + lx, min(x0 + lx + 2, x1))
                        )
                    )
                cursor += count
    return _balanced_ranks(members, height * width)


class NativeCoverAttention(MotionAttention):

    def _order(self, height, width, groups, device):
        cache_key = (height, width, groups, str(device))
        if cache_key not in self.cache:
            count = height * width
            if count % groups:
                raise ValueError(
                    "The declared native grid must be divisible by the group count."
                )
            ranks = native_cover_ranks(height, width, groups, self.seed)
            order = sorted(
                range(count),
                key=lambda index: (ranks[index] % groups, ranks[index] // groups),
            )
            self.cache[cache_key] = torch.tensor(order, dtype=torch.long, device=device)
        return self.cache[cache_key]


def sparse_product(attention, value):
    """Aggregate B,H,N,D values in the declared grouped matrix-product order."""
    batch, heads, count, dim = value.shape
    order = attention.order
    if order.numel() != count:
        raise ValueError("The declared 7040-key grid requires no padding.")
    groups, per_group = (attention.groups, attention.weights.shape[-1])
    ordered = value.index_select(2, order).reshape(batch, heads, groups, per_group, dim)
    pooled = (attention.weights @ ordered).reshape(batch, heads, count, dim)
    output = torch.empty_like(pooled)
    output.index_copy_(2, order, pooled)
    return output


def heads(value):
    batch, channels, height, width = value.shape
    return value.flatten(2).transpose(1, 2)[:, None]


def feature(value, shape):
    batch, channels, height, width = shape
    return value[:, 0].transpose(1, 2).reshape(batch, channels, height, width)


def components(original, fmap):
    if original.heads != 1 or original.project is not None or fmap.shape[1] != 128:
        raise ValueError(
            "Component analysis requires one 128-channel head without an extra output projection."
        )
    if original.to_v.bias is not None:
        raise ValueError("Value projection must have no bias.")
    total = original.to_v(fmap)
    motion = F.conv2d(fmap[:, -2:], original.to_v.weight[:, -2:])
    learned = total - motion
    return (total, motion, learned)


def comparisons(original, dense, maps, fmap):
    total, motion, learned = components(original, fmap)
    values = {
        "full": heads(total),
        "explicit_motion": heads(motion),
        "learned_matching": heads(learned),
        "motion_pixels": heads(fmap[:, -2:]) * 8,
    }
    references = {name: dense @ value for name, value in values.items()}
    stats, arrays = ({}, {})
    for pattern, attention in maps.items():
        detail = {}
        for name, value in values.items():
            error = sparse_product(attention, value) - references[name]
            if name != "motion_pixels":
                error = error * original.gamma
            squared = error.double().square().sum(-1)[0, 0]
            detail[name + "_rms"] = float((squared.mean() / value.shape[-1]).sqrt())
            detail[name + "_dimension"] = value.shape[-1]
            arrays[pattern + "_" + name + "_squared_error"] = (
                squared.cpu().numpy().copy()
            )
        baseline = MotionAggregation(original)(attention, fmap)
        reference = original(dense, fmap)
        direct = fmap + original.gamma * feature(
            sparse_product(attention, values["full"]), fmap.shape
        )
        detail["native_vs_direct_max_abs_error"] = float(
            (baseline - direct).abs().max()
        )
        detail["actual_aggregation_rms"] = float(
            (baseline - reference).double().square().mean().sqrt()
        )
        stats[pattern] = detail
    return (stats, arrays)


class RestoreAggregate(nn.Module):
    """Restore current-input components during the final recurrent updates."""

    def __init__(self, original, dense, arm):
        super().__init__()
        self.original, self.dense, self.arm = (original, dense, arm)
        self.records = []
        if arm not in ARMS:
            raise ValueError("Unknown intervention.")

    def forward(self, attention, fmap):
        baseline = MotionAggregation(self.original)(attention, fmap)
        if self.arm == ARMS[0]:
            return baseline
        if self.arm == ARMS[4]:
            return self.original(self.dense, fmap)
        _, motion, learned = components(self.original, fmap)
        value = motion if self.arm == ARMS[1] else learned
        value = heads(value)
        delta = feature(
            self.dense @ value - sparse_product(attention, value), fmap.shape
        )
        restored = baseline + self.original.gamma * delta
        dense_actual = self.original(self.dense, fmap)
        scale = 1.0
        if self.arm == ARMS[3]:
            error, initial = (restored - dense_actual, baseline - dense_actual)
            actual_norm, initial_norm = (error.double().norm(), initial.double().norm())
            if float(actual_norm) == 0 and float(initial_norm) != 0:
                raise RuntimeError(
                    "Zero residual cannot match a nonzero reference norm."
                )
            scale = float(initial_norm / actual_norm) if float(actual_norm) else 1.0
            restored = dense_actual + error * scale
        self.records.append(
            {
                "restored_vs_dense_rms": float(
                    (restored - dense_actual).double().square().mean().sqrt()
                ),
                "sparse_vs_dense_rms": float(
                    (baseline - dense_actual).double().square().mean().sqrt()
                ),
                "norm_matching_scale": scale,
            }
        )
        return restored


@torch.no_grad()
def check_math():
    torch.manual_seed(20261005)
    for shape in ((8, 16), (64, 64)):
        assert native_cover_ranks(*shape, 4) == coverage_ranks(*shape, 4, 16, 42)
    native = native_cover_ranks(55, 128, 4)
    assert sorted(native) == list(range(7040))
    assert torch.bincount(torch.tensor(native).remainder(4)).tolist() == [1760] * 4
    from types import SimpleNamespace
    from gma import Aggregate

    original = Aggregate(
        args=SimpleNamespace(), dim=128, heads=1, dim_head=128
    ).double()
    original.gamma.data.fill_(0.7)
    fmap = torch.randn(1, 128, 8, 16, dtype=torch.float64)
    dense = torch.randn(1, 1, 128, 128, dtype=torch.float64).softmax(-1)
    group = torch.tensor(point_coverage_ranks(8, 16, 4, 32, 42)).remainder(4)
    order = torch.argsort(group, stable=True)
    sparse = dense.masked_fill(group[:, None] != group[None], 0)
    sparse = sparse / sparse.sum(-1, keepdim=True)
    weights = torch.stack(
        [
            sparse[:, :, order[g * 32 : (g + 1) * 32], :][
                :, :, :, order[g * 32 : (g + 1) * 32]
            ]
            for g in range(4)
        ],
        dim=2,
    )
    attention = SparseAttentionMap(weights, order, 4)
    total, motion, learned = components(original, fmap)
    assert torch.allclose(total, motion + learned, rtol=0, atol=2e-15)
    actual = MotionAggregation(original)(attention, fmap)
    target = original(dense, fmap)
    restored_motion = RestoreAggregate(original, dense, ARMS[1])(attention, fmap)
    restored_learned = RestoreAggregate(original, dense, ARMS[2])(attention, fmap)
    assert torch.allclose(
        restored_motion + restored_learned - actual, target, atol=2e-14, rtol=0
    )
    control = RestoreAggregate(original, dense, ARMS[3])(attention, fmap)
    assert abs(float((control - target).norm() / (actual - target).norm()) - 1) < 1e-13
    return {
        "divisible_grid_layout_equal": "passed",
        "rectangular_balanced_permutation": "passed",
        "component_reconstruction": "passed",
        "additive_restoration_identity": "passed",
        "norm_matching_identity": "passed",
    }


import numpy as np


def numpy(value):
    return value.detach().cpu().numpy().copy()


def prediction(value, padder):
    return numpy(padder.unpad(value[0]).permute(1, 2, 0).float())


def measures(current, reference, truth):
    valid = np.isfinite(truth).all(-1) & (np.abs(truth) < 1000000000.0).all(-1)
    if (
        not np.isfinite(current).all()
        or not np.isfinite(reference).all()
        or (not valid.any())
    ):
        raise RuntimeError(
            "Predictions must be finite and labeled flow must contain valid pixels."
        )
    actual = np.linalg.norm(
        current.astype(np.float64) - truth.astype(np.float64), axis=-1
    )[valid]
    change = np.linalg.norm(
        current.astype(np.float64) - reference.astype(np.float64), axis=-1
    )[valid]
    original = np.linalg.norm(
        reference.astype(np.float64) - truth.astype(np.float64), axis=-1
    )[valid]
    return {
        "valid_pixels": int(valid.sum()),
        "epe_sum": float(actual.sum()),
        "GT_EPE": float(actual.mean()),
        "flow_change_sum": float(change.sum()),
        "mean_flow_change": float(change.mean()),
        "flow_change_squared_sum": float(np.square(change).sum()),
        "rms_flow_change": float(np.square(change).mean() ** 0.5),
        "improved_pixels": int((actual < original).sum()),
        "degraded_pixels": int((actual > original).sum()),
    }


class ObservedCorrelation:

    def __init__(self, target):
        self.target, self.last_coordinates = (target, None)

    def __call__(self, coordinates):
        self.last_coordinates = coordinates.detach().clone()
        return self.target(coordinates)


def capture_forward(
    model,
    original_attention,
    original_aggregate,
    first,
    second,
    padder,
    pattern,
    probes,
):
    import network

    model.att = original_attention if pattern == "dense" else probes[pattern]
    model.update_block.aggregator = (
        original_aggregate
        if pattern == "dense"
        else MotionAggregation(original_aggregate).eval()
    )
    captured = {"step": 0, "motion_inputs": {}, "aggregation_outputs": {}}
    correlation_class = network.CorrBlock

    def factory(*args, **kwargs):
        value = ObservedCorrelation(correlation_class(*args, **kwargs))
        captured["correlation"] = value
        return value

    def attention_hook(module, inputs, output):
        captured["qk_input"] = inputs[0].detach().clone()
        captured["attention"] = output

    def update_hook(module, inputs):
        captured["step"] += 1
        if captured["step"] == 11:
            captured["update11_input"] = {
                "net": inputs[0].detach().clone(),
                "inp": inputs[1].detach().clone(),
                "coordinates": captured["correlation"]
                .last_coordinates.detach()
                .clone(),
            }
        captured["current_flow"] = inputs[3]

    def aggregate_hook(module, inputs, output):
        if not torch.equal(inputs[1][:, -2:], captured["current_flow"]):
            raise RuntimeError(
                "The last two motion channels must equal the explicit flow."
            )
        step = captured["step"]
        if step in (6, 12):
            captured["motion_inputs"][step] = inputs[1].detach().clone()
            captured["aggregation_outputs"][step] = output.detach().clone()

    handles = [
        model.att.register_forward_hook(attention_hook),
        model.update_block.register_forward_pre_hook(update_hook),
        model.update_block.aggregator.register_forward_hook(aggregate_hook),
    ]
    network.CorrBlock = factory
    try:
        low, high = model(first, second, iters=12, test_mode=True)
    finally:
        network.CorrBlock = correlation_class
        for handle in handles:
            handle.remove()
    if captured["step"] != 12 or set(captured["motion_inputs"]) != {6, 12}:
        raise RuntimeError(
            "Expected all twelve updates and both declared diagnostic inputs."
        )
    captured.pop("current_flow")
    return (captured, low.detach().clone(), prediction(high, padder))


def replay(model, original_aggregate, dense, captured, padder, first, arm):
    controller = RestoreAggregate(original_aggregate, dense, arm).eval()
    model.update_block.aggregator = controller
    initial = captured["update11_input"]
    net, coordinates = (initial["net"].clone(), initial["coordinates"].clone())
    coords0, _ = model.initialize_flow(first)
    for _ in range(2):
        coordinates = coordinates.detach()
        corr = captured["correlation"](coordinates)
        flow = coordinates - coords0
        net, mask, delta = model.update_block(
            net, initial["inp"], corr, flow, captured["attention"]
        )
        coordinates = coordinates + delta
        if mask is None:
            raise RuntimeError(
                "The released configuration requires convex flow upsampling."
            )
        high = model.upsample_flow(coordinates - coords0, mask)
    return (coordinates - coords0, prediction(high, padder), controller.records)

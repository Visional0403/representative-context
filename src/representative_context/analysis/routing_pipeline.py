"""Matched-input routing costs and actual selected-context coverage."""

import gc
import math
import statistics
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from representative_context.analysis.routing import (
    StageTimer,
    dense_attention,
    projection,
    qkv,
    region_attention,
    region_prepare,
    routed_topk,
)
from representative_context.kernels.indexed import build_layout, indexed_attention
from representative_context.layouts.spatial import coverage_ranks, point_coverage_ranks
from representative_context.controls.regions import bilevel_region_attention
from representative_context.controls.topk import topk_attention_optimized
from representative_context.controls import topk as deployed_topk

COVER_METHODS = ("point32", "proxy8", "exact", "region", "cover1", "cover16")


def timing(call, warmup=5, repeats=20):
    for _ in range(warmup):
        out = call()
        del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for _ in range(repeats):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        out = call()
        b.record()
        b.synchronize()
        records.append(a.elapsed_time(b))
        del out
    return {
        "timing_ms": records,
        "median_ms": statistics.median(records),
        "mean_ms": statistics.mean(records),
        "stdev_ms": statistics.pstdev(records),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "warmup": warmup,
        "repeats": repeats,
    }


def make_fixed(spec, pattern="point32"):
    s, i, core, side, ks = spec
    groups = math.ceil(ks * ks / core.diagnostic_capacity)
    rank = (
        point_coverage_ranks(ks, ks, groups, 32, 42 + 100 * s + i)
        if pattern == "point32"
        else coverage_ranks(ks, ks, groups, int(pattern[5:]), 42 + 100 * s + i)
    )
    ranks = torch.tensor(rank, dtype=torch.int64, device="cuda")
    return (
        ranks,
        build_layout(ranks, (side, side), (ks, ks), core.diagnostic_capacity),
    )


def setup(model, specs, method, dataset):
    """Reuse the same model parameters while selecting a routing execution path."""
    for spec in specs:
        s, i, core, side, ks = spec
        core._diagnostic_projection = projection(
            core.in_proj_weight.shape[1] // core.num_heads, 42 + 100 * s + i, "cuda"
        )
        core._diagnostic_layout = make_fixed(spec)[1]
        core._diagnostic_method = method

        def forward(self, x, height, width):
            q, k, v, kh, kw = qkv(self, x, height, width)
            mode = self._diagnostic_method
            budget = math.ceil(kh * kw / math.ceil(kh * kw / self.diagnostic_capacity))
            if mode == "dense":
                output = dense_attention(q, k, v)
            elif mode == "point32":
                output = indexed_attention(q, k, v, self._diagnostic_layout)
            elif mode.startswith("region"):
                output = (
                    bilevel_region_attention(q, k, v, (height, width), (kh, kw), budget)
                    if mode == "region_ref"
                    else region_attention(q, k, v, (height, width), (kh, kw), budget)
                )
            else:
                proxy = self._diagnostic_projection if mode == "proxy8" else None
                rq, rk = (None, None) if proxy is None else (q @ proxy, k @ proxy)
                output = topk_attention_optimized(
                    q,
                    k,
                    v,
                    budget,
                    rq,
                    rk,
                    score_elements=(4 if dataset == "uavid" else 16) * 1024**2,
                    selector="torch" if dataset == "uavid" else "triton",
                )
            return self.out_proj(output.transpose(1, 2).reshape(x.shape))

        core.forward = MethodType(forward, core)


def capture(specs, call, window):
    result, handles = ({}, [])
    for s, i, core, side, ks in specs:

        def hook(module, inputs, key=(s, i)):
            result[key] = inputs[0].detach().cpu()

        handles.append(core.register_forward_pre_hook(hook))
    out = call(window)
    del out
    for handle in handles:
        handle.remove()
    assert len(result) == len(specs)
    return result


def whole(args, model, specs, get_window, call):
    setup(model, specs, args.method, args.dataset)
    window = get_window(0)
    verification = None
    if args.method == "region":
        for *_, core, side, ks in specs:
            core._diagnostic_method = "region_ref"
        expected = call(window)
        for *_, core, side, ks in specs:
            core._diagnostic_method = "region"
        actual = call(window)
        verification = {
            "region_logits_relative_l2": float(
                (actual - expected).norm() / expected.norm()
            ),
            "max_abs_error": float((actual - expected).abs().max()),
            "native_label_change_fraction": float(
                (actual.argmax(1) != expected.argmax(1)).float().mean()
            ),
        }
        assert all((math.isfinite(value) for value in verification.values()))
        del expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "scope": "First fixed top-left GPU window through model logits with dynamic selection per forward.",
        "window_shape": list(window.shape),
        "region_numerical_check": verification,
    }
    report["whole_model_timing"] = timing(lambda: call(window))
    report["recheck_timing"] = timing(lambda: call(window), 3, 10)
    return report


def cost(args, specs, get_window, call):
    window = get_window(0)
    inputs = capture(specs, call, window)
    rows = []
    for spec in specs:
        s, i, core, side, ks = spec
        q, k, v, kh, kw = qkv(core, inputs.pop((s, i)).cuda(), side, side)
        assert (kh, kw) == (ks, ks)
        keep = core.diagnostic_capacity
        proxy = projection(q.shape[-1], 42 + 100 * s + i, "cuda")
        layout = make_fixed(spec)[1]
        checks = []
        expected = bilevel_region_attention(q, k, v, (side, side), (ks, ks), keep)
        actual = region_attention(q, k, v, (side, side), (ks, ks), keep)
        checks.append(
            {
                "method": "region_sdpa_execution",
                "relative_l2": float((actual - expected).norm() / expected.norm()),
                "max_abs_error": float((actual - expected).abs().max()),
            }
        )
        assert checks[-1]["relative_l2"] < 0.0001
        del actual, expected
        for mode in ("dense", "point32", "proxy8", "exact", "region"):

            def op(timer=None):
                timer = timer or StageTimer()
                if mode in ("dense", "point32"):
                    with timer.stage(
                        "dense_attention"
                        if mode == "dense"
                        else "fixed_indexed_attention"
                    ):
                        return (
                            dense_attention(q, k, v)
                            if mode == "dense"
                            else indexed_attention(q, k, v, layout)
                        )
                if mode in ("proxy8", "exact"):
                    return routed_topk(
                        q,
                        k,
                        v,
                        keep,
                        proxy if mode == "proxy8" else None,
                        (4 if args.dataset == "uavid" else 16) * 1024**2,
                        "torch" if args.dataset == "uavid" else "triton",
                        timer,
                    )
                return region_attention(
                    q, k, v, (side, side), (ks, ks), keep, timer=timer
                )

            if mode in ("proxy8", "exact"):
                rq, rk = (q @ proxy, k @ proxy) if mode == "proxy8" else (None, None)
                expected = topk_attention_optimized(
                    q,
                    k,
                    v,
                    keep,
                    rq,
                    rk,
                    score_elements=(4 if args.dataset == "uavid" else 16) * 1024**2,
                    selector="torch" if args.dataset == "uavid" else "triton",
                )
                actual = op()
                check = {
                    "method": mode,
                    "relative_l2": float((actual - expected).norm() / expected.norm()),
                    "max_abs_error": float((actual - expected).abs().max()),
                }
                assert check["max_abs_error"] == 0
                checks.append(check)
                del actual, expected, rq, rk
            measured = timing(op, 3, 10)
            decomposed, instrumented = ([], [])
            for _ in range(5):
                timer = StageTimer(True)
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                out = op(timer)
                b.record()
                b.synchronize()
                decomposed.append(timer.result())
                instrumented.append(a.elapsed_time(b))
                del out
            names = sorted({key for rep in decomposed for key in rep})
            components = {
                name: statistics.median((rep.get(name, 0.0) for rep in decomposed))
                for name in names
            }
            rows.append(
                {
                    "stage": s + 1,
                    "block": i,
                    "method": mode,
                    "Q": side**2,
                    "KV": ks**2,
                    "capacity": keep,
                    "heads": core.num_heads,
                    "head_dimension": q.shape[-1],
                    "uninstrumented": measured,
                    "component_median_ms": components,
                    "component_observations_ms": decomposed,
                    "instrumented_total_ms": instrumented,
                    "numerical_checks": checks.copy(),
                }
            )
            print(
                f"Cost {args.dataset} s{s + 1} b{i} {mode}: {measured['median_ms']:.4f}ms",
                flush=True,
            )
        del q, k, v, layout, proxy
    return {
        "scope": "Shared dense-trajectory QKV; per-block aggregation with separate uninstrumented and staged timings.",
        "blocks": rows,
    }


def sample_ids(side):
    y = torch.linspace(0, side - 1, 8, device="cuda").round().long()
    return (y[:, None] * side + y[None, :]).reshape(-1)


def actual_topk_indices(q, k, v, keep, qids, proxy, dataset):
    """Capture actual selected indices with the declared GEMM chunks and tie handling."""
    captured = []
    original = deployed_topk.selected_attention

    def tap(q, k, v, indices, scores, start, output, *args):
        selected = qids[(qids >= start) & (qids < start + indices.shape[-2])]
        if selected.numel():
            captured.append(indices[:, :, selected - start].clone())
        return output

    rq, rk = (None, None) if proxy is None else (q @ proxy, k @ proxy)
    deployed_topk.selected_attention = tap
    try:
        ignored = topk_attention_optimized(
            q,
            k,
            v,
            keep,
            rq,
            rk,
            score_elements=(4 if dataset == "uavid" else 16) * 1024**2,
            selector="torch" if dataset == "uavid" else "triton",
        )
        del ignored
    finally:
        deployed_topk.selected_attention = original
    result = torch.cat(captured, dim=2)
    assert result.shape[-2:] == (qids.numel(), keep)
    return result


def geometry(ids, side, query_cell):
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.zeros((side, side), dtype=bool)
    mask.reshape(-1)[ids] = True
    distance = distance_transform_edt(~mask)
    diag = math.sqrt(2) * (side - 1)
    occupied = np.unique(ids // side * 4 // side * 8 + ids % side * 8 // side).size
    adjacency = np.count_nonzero(mask[:-1] & mask[1:]) + np.count_nonzero(
        mask[:, :-1] & mask[:, 1:]
    )
    ys, xs = (ids // side, ids % side)
    cy, cx = divmod(int(query_cell), side)
    local = np.count_nonzero((abs(ys - cy) <= 1) & (abs(xs - cx) <= 1))
    return [
        distance.mean() / diag,
        distance.max() / diag,
        occupied / 32,
        adjacency / len(ids),
        local,
        (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / side**2,
    ]


def coverage(args, specs, images, indexes, get_window, call):
    records, arrays, archive_rows = ([], {}, [])
    for index in indexes:
        window = get_window(index)
        inputs = capture(specs, call, window)
        image_row = {
            "index": index,
            "image": str(Path(images[index]).relative_to(args.data_root)),
            "window_shape": list(window.shape),
            "top_left": [0, 0],
        }
        archive_rows.append(image_row)
        for spec in specs:
            s, i, core, side, ks = spec
            q, k, v, kh, kw = qkv(core, inputs.pop((s, i)).cuda(), side, side)
            assert kh == kw == ks
            qids = sample_ids(side)
            cells = qids // side // (side // ks) * ks + qids % side // (side // ks)
            qs = q[:, :, qids]
            scores = qs @ k.transpose(-1, -2) * q.shape[-1] ** (-0.5)
            probs = scores.softmax(-1)
            dense_out = probs @ v
            keep, heads = (core.diagnostic_capacity, core.num_heads)
            masks = {}
            for name in ("point32", "cover1", "cover16"):
                ranks, layout = make_fixed(spec, name)
                groups = ranks % layout.groups
                selected = groups[cells, None] == groups[None, :]
                assert bool((selected.sum(-1) == keep).all())
                masks[name] = (
                    selected.nonzero()[:, 1]
                    .reshape(64, keep)[None, None]
                    .expand(1, heads, 64, keep)
                )
            masks["exact"] = actual_topk_indices(
                q, k, v, keep, qids, None, args.dataset
            )
            proxy = projection(q.shape[-1], 42 + 100 * s + i, "cuda")
            masks["proxy8"] = actual_topk_indices(
                q, k, v, keep, qids, proxy, args.dataset
            )
            qr, kr, vr, qm, km, geo = region_prepare(q, k, v, (side, side), (ks, ks))
            rr, rc, qth, qtw, kth, ktw = geo
            routes = (qm @ km).topk(keep // (kth * ktw), dim=-1, sorted=False).indices
            region_cells = (
                torch.arange(ks**2, device="cuda")
                .reshape(rr, kth, rc, ktw)
                .permute(0, 2, 1, 3)
                .reshape(rr * rc, kth * ktw)
            )
            qregions = qids // side // qth * rc + qids % side // qtw
            ids = region_cells[routes[:, qregions]].reshape(1, 1, 64, keep)
            masks["region"] = ids.expand(1, heads, 64, keep)
            metrics = {}
            for mode in COVER_METHODS:
                ids = masks[mode]
                assert ids.shape == (1, heads, 64, keep)
                assert bool((ids.sort(-1).values.diff(dim=-1) > 0).all())
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                    -1, ids, True
                )
                out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
                chosen_mass = probs.gather(-1, ids).sum(-1)
                top1 = mask.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
                err2 = (out - dense_out).double().square().sum(-1)
                ref2 = dense_out.double().square().sum(-1)
                id_cpu = ids[0].cpu().numpy().astype(np.uint16)
                cells_cpu = cells.cpu().numpy()
                spatial = np.asarray(
                    [
                        [geometry(row, ks, cell) for row, cell in zip(head, cells_cpu)]
                        for head in id_cpu
                    ]
                )
                vals = np.concatenate(
                    (
                        spatial,
                        chosen_mass[0].cpu().numpy()[..., None],
                        top1[0].cpu().numpy()[..., None],
                        err2[0].cpu().numpy()[..., None],
                        ref2[0].cpu().numpy()[..., None],
                    ),
                    axis=-1,
                )
                prefix = f"image{index}_s{s}_b{i}_{mode}"
                arrays[prefix + "_ids"] = id_cpu
                arrays[prefix + "_metrics"] = vals
                metrics[mode] = {
                    "indices_key": prefix + "_ids",
                    "metrics_key": prefix + "_metrics",
                    "mean_normalized_nearest_distance": float(spatial[..., 0].mean()),
                    "mean_normalized_largest_gap": float(spatial[..., 1].mean()),
                    "mean_macroregion_coverage": float(spatial[..., 2].mean()),
                    "mean_adjacent_pairs_per_key": float(spatial[..., 3].mean()),
                    "mean_local_3x3_keys": float(spatial[..., 4].mean()),
                    "mean_bounding_box_fraction": float(spatial[..., 5].mean()),
                    "retained_dense_probability_mass": float(chosen_mass.mean()),
                    "dense_top1_retention": float(top1.float().mean()),
                    "aggregation_relative_l2": float((err2.sum() / ref2.sum()).sqrt()),
                    "aggregation_error_squared_sum": float(err2.sum()),
                    "dense_aggregation_squared_sum": float(ref2.sum()),
                }
            exact_mass = probs.gather(-1, masks["exact"]).sum(-1)
            assert all(
                (
                    bool((exact_mass + 1e-05 >= probs.gather(-1, ids).sum(-1)).all())
                    for ids in masks.values()
                )
            )
            key = f"image{index}_s{s}_b{i}"
            arrays[key + "_qids"] = qids.cpu().numpy().astype(np.int32)
            records.append(
                {
                    "image_index": index,
                    "stage": s + 1,
                    "block": i,
                    "Q": side**2,
                    "KV": ks**2,
                    "capacity": keep,
                    "heads": heads,
                    "sampled_queries": 64,
                    "exact_topk_has_maximum_mass": True,
                    "method": metrics,
                }
            )
            print(
                f"Coverage {args.dataset} image {index} s{s + 1} b{i} complete",
                flush=True,
            )
            del (
                q,
                k,
                v,
                qs,
                scores,
                probs,
                dense_out,
                masks,
                proxy,
                qr,
                kr,
                vr,
                qm,
                km,
                routes,
                out,
            )
        del window
    path = args.output.with_suffix(".npz")
    np.savez_compressed(path, **arrays)
    return {
        "scope": "Shared dense inputs, fixed 8-by-8 queries, all heads and equal retained capacity.",
        "inputs": archive_rows,
        "blocks": records,
        "arrays": str(path.name),
        "metric_columns": [
            "mean_nearest_distance_over_diagonal",
            "largest_gap_over_diagonal",
            "macroregion_coverage_4x8",
            "adjacent_pairs_per_key",
            "local_3x3_keys",
            "bounding_box_fraction",
            "dense_probability_mass",
            "dense_top1_hit",
            "aggregation_error_squared",
            "dense_aggregation_squared",
        ],
    }

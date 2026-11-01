"""Fixed-capacity spatial-support controls for segmentation attention."""

import math
import numpy as np
import torch
from representative_context.adapters.selection import extract_qkv
from representative_context.layouts import make_ranks
from representative_context.layouts.extent import spatial_statistics


def difference(reference, candidate):
    first = reference.detach().float().reshape(-1)
    second = candidate.detach().float().reshape(-1)
    ref_norm = torch.linalg.vector_norm(first)
    cand_norm = torch.linalg.vector_norm(second)
    delta_norm = torch.linalg.vector_norm(second - first)
    n = first.numel() ** 0.5
    return dict(
        reference_rms=float(ref_norm / n),
        difference_rms=float(delta_norm / n),
        relative_l2=float(delta_norm / ref_norm.clamp_min(1e-12)),
        cosine_similarity=float(
            torch.dot(first, second) / (ref_norm * cand_norm).clamp_min(1e-12)
        ),
        candidate_reference_norm_ratio=float(cand_norm / ref_norm.clamp_min(1e-12)),
    )


def geometry(case):
    native = case == "segformer_uavid"
    side = 2048 if native else 1024
    block_ids = ((0, 1),) * 4 if native else ((1,), (1,), (1, 3, 5, 7, 9), (1, 3))
    heads = (1, 2, 5, 8) if native else (2, 4, 8, 16)
    patterns = [f"extent{i}" for i in range(7 if native else 5)] + ["dispersed32"]
    result = {}
    for pattern in patterns:
        rows = []
        for stage, blocks in enumerate(block_ids):
            qside = side // (4 * 2**stage)
            kside = side // 32
            capacity = 64 * 2**stage
            for block in blocks:
                seed = 42 + 100 * stage + block
                ranks = make_ranks(kside, kside, capacity, pattern, seed)
                row = spatial_statistics(ranks, kside, kside, kside**2 // capacity)
                row.update(
                    stage=stage + 1,
                    block=block,
                    seed=seed,
                    Q=qside**2,
                    KV=kside**2,
                    heads=heads[stage],
                    retained_edges=heads[stage] * qside**2 * capacity,
                    attention_matmul_flops=4 * 32 * heads[stage] * qside**2 * capacity,
                )
                rows.append(row)
        weighted = sum(row["retained_edges"] for row in rows)
        gap = (
            sum(row["mean_nearest_distance"] * row["retained_edges"] for row in rows)
            / weighted
        )
        result[pattern] = dict(
            blocks=rows,
            coverage_score=1 - gap,
            mean_nearest_distance=gap,
            retained_edges=sum(row["retained_edges"] for row in rows),
            attention_matmul_flops=sum(row["attention_matmul_flops"] for row in rows),
        )
    assert len({row["retained_edges"] for row in result.values()}) == 1
    return dict(case=case, k=14, patterns=result)


def probe(adapter, metadata, limit=None):
    captures, handles = {}, []
    for stage, index, _, core in adapter.specs:

        def input_hook(module, inputs, key=(stage, index)):
            captures[key] = dict(
                x=inputs[0].detach(), shape=(int(inputs[1]), int(inputs[2]))
            )

        def output_hook(module, inputs, key=(stage, index)):
            captures[key]["raw"] = inputs[0].detach()

        handles.append(core.register_forward_pre_hook(input_hook))
        handles.append(core.out_proj.register_forward_pre_hook(output_hook))
    rows = []
    declared = metadata["windows"] if limit is None else metadata["windows"][:limit]
    adapter.mode(None)
    try:
        for expected in declared:
            window, _, identity = adapter.window(expected["index"])
            assert identity["top_left"] == expected["top_left"]
            captures.clear()
            features = adapter.model.backbone(window)
            _, output = adapter.decode(features)
            blocks = []
            for stage, index, _, core in adapter.specs:
                captured = captures[stage, index]
                height, width = captured["shape"]
                y = (
                    torch.linspace(0, height - 1, 16, device=window.device)
                    .round()
                    .long()
                )
                x = (
                    torch.linspace(0, width - 1, 16, device=window.device)
                    .round()
                    .long()
                )
                queries = (y[:, None] * width + x[None]).reshape(-1)
                q, k, v, kh, kw = extract_qkv(core, captured["x"], height, width)
                q = q[:, :, queries]
                scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
                probabilities = scores.softmax(-1)
                dense = probabilities @ v
                actual = (
                    captured["raw"]
                    .reshape(1, height * width, core.num_heads, q.shape[-1])
                    .permute(0, 2, 1, 3)[:, :, queries]
                )
                check = difference(actual, dense)
                assert check["relative_l2"] < 1e-5
                capacity = 64 * 2**stage
                groups = math.ceil(kh * kw / capacity)
                cells = (
                    queries // width * kh // height
                ) * kw + queries % width * kw // width
                layouts = {}
                for pattern in metadata["patterns"]:
                    ids = (
                        torch.tensor(
                            make_ranks(kh, kw, capacity, pattern, core.seed),
                            device=window.device,
                        )
                        % groups
                    )
                    selected = ids[cells, None] == ids[None]
                    assert torch.all(selected.sum(-1) == capacity)
                    sparse = (
                        scores.masked_fill(
                            ~selected[None, None], -float("inf")
                        ).softmax(-1)
                        @ v
                    )
                    layouts[pattern] = dict(
                        aggregation_relative_l2=difference(dense, sparse)[
                            "relative_l2"
                        ],
                        aggregation_error_squared_sum=float(
                            (dense - sparse).square().sum()
                        ),
                        dense_aggregation_squared_sum=float(dense.square().sum()),
                        retained_dense_probability_mass=float(
                            probabilities.masked_fill(~selected[None, None], 0)
                            .sum(-1)
                            .mean()
                        ),
                    )
                blocks.append(
                    dict(
                        stage=stage + 1,
                        block=index,
                        queries=len(queries),
                        dense_check=check,
                        layouts=layouts,
                    )
                )
            rows.append(dict(identity, blocks=blocks))
            print(f"{len(rows)}/{len(declared)} {identity['image']}", flush=True)
            captures.clear()
            del features, output, window
    finally:
        for handle in handles:
            handle.remove()
    return dict(
        case=adapter.case,
        k=14,
        complete_windows=len(rows) == len(metadata["windows"]),
        windows=rows,
    )

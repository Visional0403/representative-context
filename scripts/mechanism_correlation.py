"""Measure aggregation advantages while preserving local context and global anchors."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from _bootstrap import ROOT
from representative_context.adapters.selection import extract_qkv
from representative_context.analysis.correlation import protected_permutation
from representative_context.analysis.segmentation import SegmentationAdapter
from representative_context.controls.context_layout import construct
from representative_context.runtime import environment, set_seed, write_json

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("segformer_uavid", "twins_uavid", "twins_cityscapes"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose an output directory that does not already exist.")
    if args.limit is not None and args.limit < 1:
        parser.error("A restricted run requires at least one window.")
    set_seed(42)
    torch.backends.cudnn.allow_tf32 = False
    adapter = SegmentationAdapter(args.case, ROOT, args.data_root, args.checkpoint)
    indices = adapter.indices if args.limit is None else adapter.indices[: args.limit]
    args.output_dir.mkdir(parents=True)
    report = dict(
        case=args.case,
        task="spatial_correlation_intervention",
        inference_seed=42,
        window_size=adapter.side,
        declared_indices=adapter.indices,
        evaluated_indices=indices,
        complete_windows=len(indices) == len(adapter.indices),
        environment=environment(),
        windows=[],
    )
    layout_cache = {}
    with torch.inference_mode():
        for index in indices:
            window, gt, item = adapter.window(index)
            captured, handles = {}, []
            adapter.mode(None)
            for stage, block_index, block, core in adapter.specs:

                def capture(module, inputs, key=(stage, block_index)):
                    captured[key] = (inputs[0].detach(), int(inputs[1]), int(inputs[2]))

                handles.append(core.register_forward_pre_hook(capture))
            try:
                adapter.model.backbone(window)
            finally:
                for handle in handles:
                    handle.remove()
            item["blocks"] = []
            stored = {}
            for stage, block_index, block, core in adapter.specs:
                x, qh, qw = captured[stage, block_index]
                axis = torch.tensor(
                    [int((a + 0.5) * qh / 4) for a in range(4)], device=x.device
                )
                qids = (axis[:, None] * qw + axis[None, :]).flatten()
                q, k, v, kh, kw = extract_qkv(core, x, qh, qw)
                q = q[:, :, qids, :]
                assert kh == kw == adapter.grid
                q, k, v = q[0], k[0], v[0]
                budget, seed = core.last_plan.capacity_limit, core.seed
                identity = kh, budget, seed
                if identity not in layout_cache:
                    layout_cache[identity] = {
                        cluster: construct(kh, budget, cluster, seed)[0]
                        for cluster in (4, 1)
                    }
                layouts = layout_cache[identity]
                scores = (q @ k.transpose(1, 2)) * q.shape[-1] ** -0.5
                dense = scores.softmax(-1) @ v
                values = F.normalize(v - v.mean(1, keepdim=True), dim=-1)
                y, xpos = np.divmod(np.arange(kh * kh), kh)
                quadrant = (y >= kh // 2).astype(int) * 2 + (xpos >= kh // 2)
                cells = np.arange(kh * kh).reshape(kh, kh)
                neighbors = np.concatenate(
                    (
                        np.stack((cells[:, :-1].ravel(), cells[:, 1:].ravel()), 1),
                        np.stack((cells[:-1].ravel(), cells[1:].ravel()), 1),
                    )
                )
                probes = []
                for fraction in FRACTIONS:
                    for permutation_seed in (
                        (20261022,) if fraction == 0 else (20261022, 20261023)
                    ):
                        errors = {cluster: [] for cluster in (4, 1)}
                        correlations, reference, dense_errors = [], [], []
                        for sample, qid in enumerate(qids.cpu().tolist()):
                            cell = (qid // qw // (qh // kh)) * kh + (
                                qid % qw // (qw // kw)
                            )
                            permutation, fixed = protected_permutation(
                                kh,
                                cell,
                                fraction,
                                permutation_seed + 100 * stage + block_index,
                            )
                            permutation = torch.as_tensor(
                                permutation, device=x.device, dtype=torch.long
                            )
                            ps, pv = scores[:, sample, permutation], v[:, permutation]
                            pd = torch.einsum("hk,hkd->hd", ps.softmax(-1), pv)
                            dense_errors.append(
                                float(
                                    (pd - dense[:, sample]).norm()
                                    / dense[:, sample].norm().clamp_min(1e-12)
                                )
                            )
                            reference.append(
                                dense[:, sample].double().square().sum().item()
                            )
                            pairs = neighbors[
                                (~fixed[neighbors]).all(1)
                                & (
                                    quadrant[neighbors[:, 0]]
                                    == quadrant[neighbors[:, 1]]
                                )
                            ]
                            pairs = torch.as_tensor(
                                pairs, device=x.device, dtype=torch.long
                            )
                            normalized = values[:, permutation]
                            correlations.append(
                                float(
                                    (
                                        normalized[:, pairs[:, 0]]
                                        * normalized[:, pairs[:, 1]]
                                    )
                                    .sum(-1)
                                    .mean()
                                )
                            )
                            for cluster in (4, 1):
                                ids = torch.as_tensor(
                                    layouts[cluster][cell],
                                    device=x.device,
                                    dtype=torch.long,
                                )
                                output = torch.einsum(
                                    "hk,hkd->hd", ps[:, ids].softmax(-1), pv[:, ids]
                                )
                                errors[cluster].append(
                                    float(
                                        (output - dense[:, sample])
                                        .double()
                                        .square()
                                        .sum()
                                    )
                                )
                        assert max(dense_errors) < 1e-4
                        probes.append(
                            dict(
                                reassignment_fraction=fraction,
                                seed=permutation_seed,
                                neighbor_residual_cosine=float(np.mean(correlations)),
                                dense_max_relative_change=max(dense_errors),
                                aggregation_relative_l2={
                                    str(cluster): (
                                        sum(errors[cluster]) / sum(reference)
                                    )
                                    ** 0.5
                                    for cluster in (4, 1)
                                },
                            )
                        )
                        stored[
                            f"s{stage}b{block_index}_f{fraction}_seed{permutation_seed}"
                        ] = np.array([reference, errors[4], errors[1], correlations])
                item["blocks"].append(
                    dict(
                        stage=stage,
                        block=block_index,
                        capacity=budget,
                        sampled_queries=len(qids),
                        heads=q.shape[0],
                        interventions=probes,
                    )
                )
                del q, k, v, scores, dense, values
            filename = f"gradient_{index:03d}.npz"
            np.savez_compressed(args.output_dir / filename, **stored)
            item["arrays"] = filename
            report["windows"].append(item)
            del captured, window
            torch.cuda.empty_cache()
            print(f"Measured {args.case} correlation window {index}", flush=True)
    write_json(args.output_dir / "measurements.json", report)


if __name__ == "__main__":
    main()

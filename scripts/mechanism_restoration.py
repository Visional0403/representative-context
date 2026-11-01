"""Measure class-mass and class-content restoration in fixed segmentation windows."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from _bootstrap import ROOT
from representative_context.analysis.context import (
    ARMS,
    PATTERNS,
    factors,
    metrics,
    task_counts,
)
from representative_context.analysis.segmentation import SegmentationAdapter
from representative_context.runtime import environment, set_seed, write_json


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
        task="class_conditioned_restoration",
        seed=42,
        window_size=adapter.side,
        classes=adapter.classes,
        declared_indices=adapter.indices,
        evaluated_indices=indices,
        complete_windows=len(indices) == len(adapter.indices),
        environment=environment(),
        windows=[],
    )
    with torch.inference_mode():
        for index in indices:
            window, gt, item = adapter.window(index)
            stored = {"target": gt}
            adapter.mode(None)
            features = adapter.model.backbone(window)
            native, logits = adapter.decode(features)
            dense = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            labels = (
                F.interpolate(
                    native.argmax(1, keepdim=True).float(),
                    size=(adapter.grid, adapter.grid),
                    mode="nearest",
                )
                .long()
                .flatten()
            )
            stored["dense_prediction"] = dense
            item["layouts"] = {}
            del features, native, logits
            for pattern in PATTERNS:
                adapter.mode(pattern)
                captured, handles = {}, []
                for _, block_index, block, core in adapter.targets:

                    def pre(module, inputs, key=block_index):
                        captured.setdefault(key, {})["block"] = (
                            inputs[0].detach().clone()
                        )

                    def post(module, inputs, branch, key=block_index):
                        captured.setdefault(key, {})["x"] = inputs[0].detach().clone()
                        captured[key]["branch"] = branch.detach().clone()

                    handles.extend(
                        (
                            block.register_forward_pre_hook(pre),
                            core.register_forward_hook(post),
                        )
                    )
                try:
                    features = adapter.model.backbone(window)
                finally:
                    for handle in handles:
                        handle.remove()
                native, logits = adapter.decode(features)
                baseline = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                stored[f"{pattern}_sparse"] = baseline
                row = {
                    "tasks": {
                        "sparse": metrics(
                            task_counts(gt, dense, baseline, adapter.classes)
                        )
                    },
                    "same_input_decomposition": [],
                }
                item["layouts"][pattern] = row
                for _, block_index, block, core in adapter.targets:
                    _, detail = factors(
                        core,
                        captured[block_index]["x"],
                        labels,
                        adapter.grid,
                        adapter.classes,
                    )
                    row["same_input_decomposition"].append(detail)
                identity, identity_logits, _ = adapter.tail(
                    features, captured, pattern, "identity", labels
                )
                difference = float((identity - native).abs().max())
                assert difference < 1e-5 and np.array_equal(
                    identity_logits.argmax(1)[0].cpu().numpy(), baseline
                )
                row["identity_max_abs_error"] = difference
                del identity, identity_logits, native, logits
                for arm in ARMS[1:]:
                    tail, logits, details = adapter.tail(
                        features, captured, pattern, arm, labels
                    )
                    prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    row["tasks"][arm] = metrics(
                        task_counts(gt, dense, prediction, adapter.classes)
                    )
                    row.setdefault("block_interventions", {})[arm] = details
                    stored[f"{pattern}_{arm}"] = prediction
                    del tail, logits
                del features, captured
                torch.cuda.empty_cache()
            filename = f"window_{index:03d}.npz"
            np.savez_compressed(args.output_dir / filename, **stored)
            item["arrays"] = filename
            report["windows"].append(item)
            del stored, window, labels
            torch.cuda.empty_cache()
            print(f"Measured {args.case} window {index}", flush=True)
    write_json(args.output_dir / "measurements.json", report)


if __name__ == "__main__":
    main()

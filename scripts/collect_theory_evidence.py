"""Reconstruct fixed QKV or paired budget-competition evidence from task models."""

import argparse
from pathlib import Path
import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.adapters.selection import extract_qkv
from representative_context.analysis.context import task_counts
from representative_context.analysis.framework import competition, pool_classification
from representative_context.analysis.segmentation import SegmentationAdapter
from representative_context.data.uavid import normalize_image, read_rgb
from representative_context.evaluation.mmseg import read_label
from representative_context.runtime import (
    environment,
    read_config,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument("--task", choices=("spatial", "competition"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    if args.limit is not None and args.limit < 1:
        parser.error("A restricted collection requires at least one window.")
    kind = "spatial" if args.task == "spatial" else "budget_competition"
    directory = ROOT / f"results/03_theory/{kind}/{args.case}"
    manifest = read_config(directory / "measurements.json")
    rows = (
        manifest["windows"] if args.limit is None else manifest["windows"][: args.limit]
    )
    set_seed(manifest["inference_seed"])
    torch.backends.cudnn.allow_tf32 = manifest["cudnn_allow_tf32"]
    adapter = SegmentationAdapter(args.case, ROOT, args.data_root, args.checkpoint)
    targets = [
        next(value for value in adapter.specs if value[0] == stage)
        for stage in range(4)
    ]
    args.output_dir.mkdir(parents=True)
    result = {
        **manifest,
        "complete_windows": len(rows) == len(manifest["windows"]),
        "environment": environment(),
        "windows": [],
    }
    with torch.inference_mode():
        for row in rows:
            path = adapter.root / row["image"]
            label = path.parent.parent / "Labels" / path.name
            rgb = read_rgb(path)
            top, left = row["top_left"]
            gt = read_label(label, "UAVid")[top : top + 2048, left : left + 2048].copy()
            if adapter.native:
                image = normalize_image(rgb)
                window = image[:, top : top + 2048, left : left + 2048][None].cuda()
            else:
                image = (
                    torch.from_numpy(
                        rgb[top : top + 2048, left : left + 2048]
                        .transpose(2, 0, 1)
                        .copy()
                    )
                    .float()[None]
                    .cuda()
                )
                window = (
                    image - adapter.model.data_preprocessor.mean
                ) / adapter.model.data_preprocessor.std
            item = {
                key: value
                for key, value in row.items()
                if key not in ("full_statistics", "confusion_cubes")
            }
            adapter.mode(None)
            captures, handles = {}, []
            if args.task == "spatial":
                for stage, _, _, core in targets:

                    def capture(module, values, current=stage):
                        captures[current] = (
                            values[0].detach(),
                            int(values[1]),
                            int(values[2]),
                        )

                    handles.append(core.register_forward_pre_hook(capture))
            try:
                features = adapter.model.backbone(window)
            finally:
                for handle in handles:
                    handle.remove()
            if args.task == "spatial":
                stored = {}
                for stage, _, _, core in targets:
                    x, qh, qw = captures.pop(stage)
                    ys = torch.tensor(
                        [int((index + 0.5) * qh / 4) for index in range(4)],
                        device=x.device,
                    )
                    xs = torch.tensor(
                        [int((index + 0.5) * qw / 4) for index in range(4)],
                        device=x.device,
                    )
                    indices = (ys[:, None] * qw + xs[None]).flatten()
                    q, key, value, kh, kw = extract_qkv(core, x, qh, qw)
                    assert kh == kw == 64
                    for name, tensor in (
                        ("Q", q[:, :, indices]),
                        ("K", key),
                        ("V", value),
                    ):
                        stored[f"stage{stage}_{name}"] = (
                            tensor[0].detach().float().cpu().numpy().copy()
                        )
                    stored[f"stage{stage}_query_indices"] = indices.cpu().numpy()
                np.savez_compressed(args.output_dir / item["qkv_arrays"], **stored)
                del features, q, key, value, x
            else:
                with np.load(directory / row["arrays"], allow_pickle=False) as frozen:
                    indices = frozen["pixel_indices"].copy()
                _, logits = adapter.decode(features)
                dense = (
                    logits[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                    .transpose(1, 2, 0)
                    .reshape(-1, 8)
                )
                stored = dict(
                    pixel_indices=indices,
                    sampled_labels=gt.ravel()[indices],
                    labels=gt,
                    dense_logits=dense[indices],
                    dense_prediction=dense.argmax(1).reshape(gt.shape).astype(np.uint8),
                )
                item["full_statistics"], item["confusion_cubes"] = {}, {}
                del logits, features
                for exponent in manifest["resource_exponents"]:
                    adapter.mode("dispersed32")
                    for _, _, _, core in adapter.specs:
                        core.k = exponent
                        core.cache.clear()
                    _, logits = adapter.decode(adapter.model.backbone(window))
                    sparse = (
                        logits[0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .copy()
                        .transpose(1, 2, 0)
                        .reshape(-1, 8)
                    )
                    stored[f"k{exponent}_logits"] = sparse[indices]
                    stored[f"k{exponent}_prediction"] = (
                        sparse.argmax(1).reshape(gt.shape).astype(np.uint8)
                    )
                    chunks = []
                    flatgt = gt.ravel()
                    for start in range(0, len(flatgt), 32768):
                        sl = slice(start, start + 32768)
                        valid = flatgt[sl] < 8
                        if valid.any():
                            chunks.append(
                                competition(
                                    dense[sl][valid],
                                    sparse[sl][valid],
                                    flatgt[sl][valid],
                                )[0]
                            )
                    item["full_statistics"][f"k{exponent}"] = pool_classification(
                        chunks
                    )
                    item["confusion_cubes"][f"k{exponent}"] = task_counts(
                        gt,
                        stored["dense_prediction"],
                        stored[f"k{exponent}_prediction"],
                        8,
                    ).tolist()
                    del logits, sparse
                np.savez_compressed(args.output_dir / item["arrays"], **stored)
                del dense
            result["windows"].append(item)
            print(f"Collected {args.task} window {row['index']}", flush=True)
            del image, window, gt, stored
            torch.cuda.empty_cache()
    write_json(args.output_dir / "measurements.json", result)


if __name__ == "__main__":
    main()

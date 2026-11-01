"""Collect fixed-window classification scores with the published sampling IDs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.adapters.sra import install_sra
from representative_context.data.uavid import read_pair
from representative_context.models.segformer import load_segformer
from representative_context.runtime import read_config, set_seed, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/03_theory/classification/manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/UAVid")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "weights/segformer/segformer_b0_uavid.pth",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose an output directory that does not already exist.")
    manifest = read_config(args.manifest)
    rows = (
        manifest["windows"] if args.limit is None else manifest["windows"][: args.limit]
    )
    if not rows:
        parser.error("Select at least one window.")
    set_seed(manifest["inference_seed"])
    torch.backends.cudnn.allow_tf32 = manifest["cudnn_allow_tf32"]
    model = load_segformer(args.checkpoint)
    controls = install_sra(
        model, k=manifest["k"], pattern=manifest["layout"], dense=True
    )
    args.output_dir.mkdir(parents=True)
    size = manifest["window_size"]
    with torch.inference_mode():
        for row in rows:
            image_path = args.data_root / row["image"]
            label_path = image_path.parent.parent / "Labels" / image_path.name
            image, label = read_pair(image_path, label_path)
            top, left = row["top_left"]
            window = (
                image[:, top : top + size, left : left + size][None].contiguous().cuda()
            )
            pixels = np.sort(
                np.random.default_rng(row["sampling_seed"]).choice(
                    size * size, size=row["sampled_pixels"], replace=False
                )
            )
            values = {
                "pixel_indices": pixels,
                "labels": label[top : top + size, left : left + size]
                .numpy()
                .reshape(-1)[pixels],
            }
            for dense, field in ((True, "dense_logits"), (False, "sparse_logits")):
                for control in controls:
                    control.dense = dense
                output = model(window)
                values[field] = (
                    output[0]
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .reshape(-1, manifest["classes"])[pixels]
                    .copy()
                )
                del output
            np.savez_compressed(args.output_dir / row["path"], **values)
            del image, label, window
            print(f"Collected window {row['window_index']}", flush=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            **manifest,
            "windows": rows,
            "total_sampled_pixels": sum(row["sampled_pixels"] for row in rows),
            "complete_windows": len(rows) == len(manifest["windows"]),
        },
    )


if __name__ == "__main__":
    main()

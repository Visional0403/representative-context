"""Recompute complete-window class certificates from the frozen model and labels."""

import argparse
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.adapters.sra import install_sra
from representative_context.analysis.certificates import certificate_counts, certificate_fields
from representative_context.data.uavid import read_pair
from representative_context.models.segformer import load_segformer
from representative_context.runtime import environment, evaluation_inputs, read_config, set_seed, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/03_theory/classification/manifest.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/UAVid")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/segformer/segformer_b0_uavid.pth")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    root, checkpoint = args.data_root.resolve(), args.checkpoint.resolve()
    manifest = read_config(args.manifest)
    rows = manifest["windows"] if args.limit is None else manifest["windows"][:args.limit]
    if not rows:
        parser.error("Select at least one complete window.")
    pairs = [(root / row["image"], root / row["image"].replace("/Images/", "/Labels/")) for row in rows]
    inputs = evaluation_inputs(checkpoint, root, pairs)
    set_seed(manifest["inference_seed"])
    torch.backends.cudnn.allow_tf32 = manifest["cudnn_allow_tf32"]
    model = load_segformer(checkpoint)
    controls = install_sra(model, k=manifest["k"], pattern=manifest["layout"], dense=True)
    args.output_dir.mkdir(parents=True)
    result = dict(kind="new_model_reproduction", inputs=inputs, environment=environment(),
                  k=manifest["k"], layout=manifest["layout"], precision=manifest["precision"],
                  seed=manifest["inference_seed"], windows=[],
                  predicate="2 * centered_error_energy < minimum_dense_margin ** 2")
    size = manifest["window_size"]
    with torch.inference_mode():
        for row, (image_path, label_path) in zip(rows, pairs):
            image, label = read_pair(image_path, label_path)
            top, left = row["top_left"]
            window = image[:, top:top+size, left:left+size][None].contiguous().cuda()
            labels = label[top:top+size, left:left+size].numpy()
            values = []
            for dense in (True, False):
                for control in controls:
                    control.dense = dense
                output = model(window)
                values.append(output[0].permute(1, 2, 0).contiguous().cpu().numpy())
                del output
            fields = certificate_fields(values[0], values[1], labels)
            entry = dict(validation_index=row["window_index"], image=row["image"],
                         top_left=row["top_left"], shape=list(labels.shape), **certificate_counts(**fields))
            if args.save_fields:
                filename = f"window_{row['window_index']:03d}.npz"
                np.savez_compressed(args.output_dir / filename, **fields)
                entry["fields"] = filename
            result["windows"].append(entry)
            del image, label, window, values, fields
            print(f"Completed window {row['window_index']}", flush=True)
    result.update(complete_windows=len(rows) == len(manifest["windows"]),
                  certified_pixels=sum(row["certified_pixels"] for row in result["windows"]),
                  valid_pixels=sum(row["valid_pixels"] for row in result["windows"]))
    result["certified_percent"] = 100 * result["certified_pixels"] / result["valid_pixels"]
    write_json(args.output_dir / "manifest.json", result)


if __name__ == "__main__":
    main()

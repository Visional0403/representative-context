"""Evaluate SegFormer-B0 on the complete labeled UAVid validation split."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
import numpy as np
import torch

from representative_context.adapters.sra import install_sra
from representative_context.data.uavid import CLASS_NAMES, evaluation_pairs, read_pair
from representative_context.evaluation.segmentation import (
    confusion_matrix,
    predict_image,
    segmentation_metrics,
)
from representative_context.models.segformer import load_segformer
from representative_context.runtime import (
    environment,
    file_info,
    evaluation_inputs,
    read_config,
    resource_path,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/segformer_uavid/validation.json"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pattern")
    parser.add_argument(
        "--method",
        choices=(
            "topk_exact",
            "topk_proxy8",
            "pooled_regions",
            "sparge",
            "sparge_fp32",
            "sparge_quant_full",
        ),
    )
    parser.add_argument("--keep-scale", type=float, default=1.0)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--native", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate the first N sorted images for a smoke check.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    config = read_config(args.config)
    for key in ("k", "pattern", "window_size", "stride"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    checkpoint = (args.checkpoint or resource_path(config["checkpoint"], ROOT)).resolve()
    data_root = (args.data_root or resource_path(config["data_root"], ROOT)).resolve()
    config.update(checkpoint=str(checkpoint), data_root=str(data_root))
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.limit is not None and args.limit < 1:
        parser.error("The image limit must be positive.")
    set_seed(config["seed"], config["threads"])
    pairs = evaluation_pairs(data_root, config["split"])
    total = len(pairs)
    if total != config["expected_split_size"]:
        parser.error(
            f"Expected {config['expected_split_size']} labeled images, found {total}."
        )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    inputs = evaluation_inputs(checkpoint, data_root, pairs)
    inputs["configuration_file"] = file_info(args.config)
    model = load_segformer(checkpoint)
    install_sra(
        model,
        config["k"],
        config["pattern"],
        config["seed"],
        args.mode == "dense",
        args.method,
        args.keep_scale,
    )
    hist = np.zeros((8, 8), dtype=np.float64)
    rows = []
    for image_path, label_path in pairs:
        image, target = read_pair(image_path, label_path)
        logits = predict_image(
            model,
            image.unsqueeze(0).cuda(),
            window_size=config["window_size"],
            stride=config["stride"],
            native=args.native,
        )
        prediction = logits.argmax(1)[0].cpu().numpy()
        counts = confusion_matrix(prediction, target.numpy())
        hist += counts
        row = {
            "image": image_path.relative_to(data_root).as_posix(),
            "shape": list(target.shape),
            "confusion_matrix": counts.tolist(),
            **segmentation_metrics(counts),
        }
        rows.append(row)
        print(
            f"{len(rows)}/{len(pairs)} {row['image']} mIoU={row['miou_percent']:.4f}",
            flush=True,
        )
        del logits, image, target
    write_json(
        args.output,
        {
            "model": "segformer_b0",
            "dataset": "UAVid",
            "mode": args.mode,
            "complete_split": len(pairs) == total,
            "split_size": total,
            "evaluated_images": len(pairs),
            "native": args.native,
            "config": config,
            "inputs": inputs,
            "class_names": CLASS_NAMES,
            "method": args.method,
            "keep_scale": args.keep_scale,
            "environment": environment(),
            "confusion_matrix": hist.tolist(),
            **segmentation_metrics(hist),
            "images": rows,
        },
    )


if __name__ == "__main__":
    main()

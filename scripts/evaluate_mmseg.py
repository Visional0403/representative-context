"""Evaluate the task checkpoints of SegFormer/Cityscapes and Twins/UAVid or Cityscapes."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
import numpy as np
import torch
from PIL import Image

from representative_context.adapters.mmseg import install_mmseg
from representative_context.evaluation.mmseg import (
    image_pairs,
    metrics,
    read_label,
    timing_input,
)
from representative_context.models.mmseg import load_mmseg
from representative_context.runtime import (
    environment,
    file_info,
    evaluation_inputs,
    read_config,
    resource_path,
    set_seed,
    write_json,
)
from representative_context.timing import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--pattern", default="dispersed32")
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
    parser.add_argument(
        "--protocol", choices=("slide1024", "slide2048", "native"), default="slide1024"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    for setting in ("warmup", "repeats", "rounds"):
        parser.add_argument("--" + setting, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    config = read_config(args.config)
    for setting in ("warmup", "repeats", "rounds"):
        if getattr(args, setting) is not None:
            if getattr(args, setting) < 1:
                parser.error("Timing counts must be positive.")
            config["timing"][setting] = getattr(args, setting)
    if args.output.exists():
        parser.error("Choose a new output file.")
    if config["dataset"] == "Cityscapes" and args.protocol == "slide2048":
        parser.error(
            "Use native evaluation for Cityscapes images wider than the 1K window."
        )
    set_seed(config["seed"], config["threads"])
    root = (args.data_root or resource_path(config["data_root"], ROOT)).resolve()
    pairs = image_pairs(root, config["dataset"])
    if len(pairs) != config["expected_images"]:
        parser.error(
            f"Expected {config['expected_images']} images, found {len(pairs)}."
        )
    checkpoint = (args.checkpoint or resource_path(config["checkpoint"], ROOT)).resolve()
    model = load_mmseg(
        config,
        checkpoint,
        ROOT / "third_party/mmsegmentation",
        protocol=args.protocol,
    )
    wrappers = install_mmseg(
        model,
        config["architecture"],
        args.k,
        args.pattern,
        config["seed"],
        args.mode == "dense",
        args.method,
        args.keep_scale,
    )
    report = {
        "architecture": config["architecture"],
        "dataset": config["dataset"],
        "mode": args.mode,
        "k": args.k,
        "pattern": args.pattern,
        "method": args.method,
        "keep_scale": args.keep_scale,
        "protocol": args.protocol,
        "global_attention_blocks": len(wrappers),
        "environment": environment(),
    }
    report["configuration"] = {**config, "protocol": args.protocol, "mode": args.mode,
        "k": args.k, "pattern": args.pattern, "method": args.method, "keep_scale": args.keep_scale,
        "limit": args.limit, "benchmark": args.benchmark, "timing_shape": [args.height, args.width]}
    report["inputs"] = evaluation_inputs(checkpoint, root, pairs)
    report["inputs"]["configuration_file"] = file_info(args.config)
    if args.benchmark:
        sample = None
        for candidate, _ in pairs:
            with Image.open(candidate) as source:
                width, height = source.size
            if height >= args.height and width >= args.width:
                sample = candidate
                break
        if sample is None:
            parser.error("No image contains the requested timing window.")
        image = timing_input(sample, args.height, args.width)
        metadata = [
            {
                "img_shape": (args.height, args.width),
                "ori_shape": (args.height, args.width),
            }
        ]
        timing = config["timing"]
        report.update(
            measure(
                {args.mode: lambda: model.encode_decode(image, metadata)}, **timing
            )[args.mode]
        )
        report.update(
            input_shape=list(image.shape),
            image=sample.relative_to(root).as_posix(),
            timing_scope="GPU-resident input through complete model logits",
            timing=timing,
        )
    else:
        from mmseg.apis import inference_model

        selected = pairs[: args.limit] if args.limit else pairs
        classes = config["classes"]
        hist = np.zeros((classes, classes), dtype=np.int64)
        rows = []
        with torch.no_grad():
            for image, label in selected:
                prediction = (
                    inference_model(model, str(image))
                    .pred_sem_seg.data.squeeze()
                    .cpu()
                    .numpy()
                )
                target = read_label(label, config["dataset"])
                valid = (target >= 0) & (target < classes)
                counts = np.bincount(
                    target[valid].astype(np.int64) * classes + prediction[valid],
                    minlength=classes * classes,
                ).reshape(classes, classes)
                hist += counts
                rows.append(
                    {
                        "image": image.relative_to(root).as_posix(),
                        "confusion_matrix": counts.tolist(),
                        **metrics(counts),
                    }
                )
                print(f"{len(rows)}/{len(selected)} {image.name}", flush=True)
        report.update(
            complete_split=len(selected) == len(pairs),
            images=rows,
            confusion_matrix=hist.tolist(),
            **metrics(hist),
        )
    write_json(args.output, report)


if __name__ == "__main__":
    main()

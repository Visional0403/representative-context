"""Measure GPU-resident single-window inference independently of validation."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
import torch

from representative_context.adapters.sra import install_sra
from representative_context.data.uavid import normalize_image, read_rgb
from representative_context.models.segformer import load_segformer
from representative_context.timing import measure
from representative_context.runtime import (
    environment,
    file_info,
    read_config,
    resource_path,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/segformer_uavid/benchmark.json"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--k", type=int)
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
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    for setting in ("warmup", "repeats", "rounds"):
        parser.add_argument("--" + setting, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config, args.image, args.output = args.config.resolve(), args.image.resolve(), args.output.resolve()
    config = read_config(args.config)
    for key in ("k", "height", "width", "warmup", "repeats", "rounds"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output.exists():
        parser.error("Choose a new output file.")
    if min(config[x] for x in ("warmup", "rounds", "repeats", "height", "width")) < 1:
        parser.error("Timing counts and input dimensions must be positive.")
    set_seed(config["seed"], config["threads"])
    image = normalize_image(read_rgb(args.image))
    h, w = config["height"], config["width"]
    if h > image.shape[-2] or w > image.shape[-1]:
        parser.error(
            "The image must contain the requested window; resizing is not applied."
        )
    image = image[:, :h, :w].unsqueeze(0).contiguous().cuda()
    checkpoint = (args.checkpoint or resource_path(config["checkpoint"], ROOT)).resolve()
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
    timing = measure({args.mode: lambda: model(image)}, warmup=config["warmup"],
                     repeats=config["repeats"], rounds=config["rounds"])[args.mode]
    write_json(
        args.output,
        {
            "model": "segformer_b0",
            "mode": args.mode,
            "config": config,
            "inputs": dict(checkpoint=file_info(checkpoint), image=file_info(args.image),
                           configuration_file=file_info(args.config)),
            "method": args.method,
            "keep_scale": args.keep_scale,
            "input_shape": list(image.shape),
            "environment": environment(),
            "scope": "GPU-resident input through all model logits; CUDA events",
            **timing,
        },
    )
    print(f"{args.mode}: {timing['median_ms']:.4f} ms")


if __name__ == "__main__":
    main()

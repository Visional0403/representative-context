"""Measure matched routing overheads and representative-context selection."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import torch

from _bootstrap import ROOT
from representative_context.analysis.routing_models import load
from representative_context.analysis.routing_pipeline import cost, coverage, whole
from representative_context.runtime import environment, set_seed, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_cityscapes"), required=True
    )
    parser.add_argument("--phase", choices=("whole", "cost", "coverage"), required=True)
    parser.add_argument(
        "--method",
        choices=("dense", "point32", "proxy8", "exact", "region_ref", "region"),
        default="dense",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.phase == "coverage"
        and args.output.with_suffix(".npz").exists()
    ):
        parser.error("Choose a new output path.")
    if (
        args.limit is not None
        and args.limit < 1
        or args.max_blocks is not None
        and args.max_blocks < 1
    ):
        parser.error("Restricted checks require positive counts.")
    if args.phase == "whole" and args.max_blocks is not None:
        parser.error("Whole-model timing uses all global attention blocks.")
    set_seed(42)
    torch.backends.cudnn.allow_tf32 = True
    model, specs, images, indices, get_window, call, data_root = load(
        ROOT, args.case, args.data_root, args.checkpoint
    )
    full_blocks = len(specs)
    full_indices = indices
    if args.max_blocks is not None:
        specs = specs[: args.max_blocks]
    if args.limit is not None:
        indices = indices[: args.limit]
    params = SimpleNamespace(
        dataset="uavid" if args.case == "segformer_uavid" else "twins",
        method=args.method,
        output=args.output,
        data_root=data_root,
    )
    with torch.inference_mode():
        if args.phase == "whole":
            report = whole(params, model, specs, get_window, call)
        elif args.phase == "cost":
            report = cost(params, specs, get_window, call)
        else:
            report = coverage(params, specs, images, indices, get_window, call)
    report.update(
        case=args.case,
        phase=args.phase,
        method=args.method if args.phase == "whole" else None,
        k=14,
        seed=42,
        selector="torch" if args.case == "segformer_uavid" else "triton",
        score_elements=(4 if args.case == "segformer_uavid" else 16) * 1024**2,
        complete_blocks=len(specs) == full_blocks,
        complete_inputs=len(indices) == len(full_indices),
        environment=environment(),
    )
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

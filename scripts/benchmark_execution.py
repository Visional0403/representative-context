"""Measure pure aggregation, identical-layout execution or complete-model costs."""

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace
import torch

from _bootstrap import ROOT
from representative_context.analysis.execution import (
    grouped_sdpa,
    paired_measure,
    profile,
)
from representative_context.analysis.execution_models import (
    capture_entries,
    check_entries,
    load_input,
    load_model,
    select_image,
)
from representative_context.kernels.indexed import indexed_attention
from representative_context.runtime import (
    environment,
    resource_path,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument(
        "--phase", choices=("pure", "compare", "profile"), required=True
    )
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if (args.height, args.width) not in (
        (1024, 1024),
        (2048, 2048),
        (2160, 3840),
        (2160, 4096),
    ):
        parser.error("Use a declared square or native UAVid shape.")
    if args.check_only and args.phase == "profile":
        parser.error("Use compare or pure for numerical checks.")
    set_seed(42)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = args.case == "segformer_uavid"
    model, blocks, config = load_model(ROOT, args.case, args.k, args.checkpoint)
    data_root = args.data_root or resource_path(config["data_root"], ROOT)
    image = select_image(data_root, args.height, args.width, args.image)
    inputs = load_input(image, args.height, args.width, args.case == "segformer_uavid")
    report = dict(
        case=args.case,
        phase=args.phase,
        k=args.k,
        seed=42,
        image=(
            image.relative_to(data_root).as_posix()
            if image.is_relative_to(data_root)
            else image.name
        ),
        input_shape=list(inputs.shape),
        environment=environment(),
        timing_enabled=not args.check_only,
    )
    with torch.inference_mode():
        if args.phase == "profile":
            params = SimpleNamespace(
                model="segformer" if args.case == "segformer_uavid" else "twins",
                height=args.height,
                width=args.width,
            )
            profile(params, model, blocks, inputs, report)
        else:
            entries, records = capture_entries(
                model, blocks, inputs, args.k, with_buckets=args.phase == "compare"
            )
            del model, blocks, inputs
            gc.collect()
            torch.cuda.empty_cache()
            check_entries(entries, records)
            report["blocks"] = records
            if not args.check_only:

                def replay(method):
                    for q, key, value, layout, plans in entries:
                        out = (
                            grouped_sdpa(q, key, value, None)
                            if method == "dense" or layout is None
                            else (
                                grouped_sdpa(q, key, value, plans)
                                if method == "reordered"
                                else indexed_attention(q, key, value, layout)
                            )
                        )
                        del out

                names = (
                    ("dense", "indexed")
                    if args.phase == "pure"
                    else ("dense", "reordered", "indexed")
                )
                report["timing_protocol"] = dict(
                    warmup=10 if args.phase == "pure" else 5,
                    rounds=3,
                    repeats=20,
                    timer="cuda_event",
                    scope="all captured global QKV aggregations replayed sequentially",
                )
                report["timing"] = paired_measure(
                    {name: lambda n=name: replay(n) for name in names},
                    warmup=report["timing_protocol"]["warmup"],
                )
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

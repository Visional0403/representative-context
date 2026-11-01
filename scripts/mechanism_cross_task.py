"""Reconstruct paired score-column interventions in three visual tasks."""

import argparse
from pathlib import Path
import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.analysis.cross_task_models import (
    GMAEvaluator,
    FusformerEvaluator,
    IMLEvaluator,
)
from representative_context.runtime import (
    environment,
    read_config,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gma", "fusformer", "iml"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose an output directory that does not already exist.")
    if args.limit is not None and args.limit < 1:
        parser.error("A restricted run requires at least one source group.")
    protocol = read_config(
        ROOT / "results/02_mechanisms/cross_task" / args.case / "measurements.json"
    )
    declared = protocol["declared_scenes"]
    rows = declared if args.limit is None else declared[: args.limit]
    set_seed(42)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    evaluator = {
        "gma": GMAEvaluator,
        "fusformer": FusformerEvaluator,
        "iml": IMLEvaluator,
    }[args.case](ROOT, args.data_root, args.weights, args.source)
    parameters = {
        name: value.data_ptr() for name, value in evaluator.model.state_dict().items()
    }
    report = dict(
        case=args.case,
        status="running",
        settings=protocol["settings"],
        complete_scenes=len(rows) == len(declared),
        declared_scene_count=len(declared),
        declared_scenes=declared,
        scenes=[],
        environment={**environment(), "precision": "FP32", "TF32": False},
        bootstrap_repeats=10000,
        bootstrap_seed=20261025,
    )
    args.output_dir.mkdir(parents=True)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            blocks, arrays, check = evaluator(row, index)
            filename = f"scene_{index:03d}.npz"
            np.savez_compressed(args.output_dir / filename, **arrays)
            report["scenes"].append(
                dict(
                    **row, blocks=blocks, dense_trajectory_check=check, arrays=filename
                )
            )
            print(f"{index+1}/{len(rows)} {row['id']}", flush=True)
    report["parameters_preserved"] = parameters == {
        name: value.data_ptr() for name, value in evaluator.model.state_dict().items()
    }
    assert report["parameters_preserved"]
    report["status"] = "complete"
    write_json(args.output_dir / "measurements.json", report)


if __name__ == "__main__":
    main()

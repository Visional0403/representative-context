"""Recompute context interventions and scene-level uncertainty from raw records."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.analysis.context_statistics import correlation, restoration
from representative_context.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("restoration", "correlation"), required=True)
    parser.add_argument(
        "--case",
        choices=("segformer_uavid", "twins_uavid", "twins_cityscapes"),
        required=True,
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    directory = args.input_dir or ROOT / "results/02_mechanisms" / args.task / args.case
    report = read_config(directory / "measurements.json")
    result = (restoration if args.task == "restoration" else correlation)(
        directory, report
    )
    write_json(
        args.output,
        {
            "case": args.case,
            "task": args.task,
            "complete_windows": report.get("complete_windows", True),
            "evaluated_windows": len(report["windows"]),
            **result,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

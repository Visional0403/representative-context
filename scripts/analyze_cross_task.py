"""Recompute cross-task mechanism curves and paired source-group intervals."""

import argparse
from pathlib import Path
from _bootstrap import ROOT
from representative_context.analysis.cross_task_statistics import analyze
from representative_context.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gma", "fusformer", "iml"), required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    directory = args.input_dir or ROOT / "results/02_mechanisms/cross_task" / args.case
    report = read_config(directory / "measurements.json")
    write_json(args.output, analyze(args.case, directory, report))
    print(args.output)


if __name__ == "__main__":
    main()

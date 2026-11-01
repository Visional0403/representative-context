"""Recompute fitted relationships and paired confidence intervals from readouts."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.analysis.applications import paired_statistics
from representative_context.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/06_generalization/application_readouts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    data = read_config(args.input)
    write_json(
        args.output,
        {case: paired_statistics(application) for case, application in data.items()},
    )
    print(args.output)


if __name__ == "__main__":
    main()

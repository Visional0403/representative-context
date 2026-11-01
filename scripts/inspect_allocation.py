"""Construct matched-compute stage capacities and count real sparse interactions."""

import argparse
from pathlib import Path
from _bootstrap import ROOT
from representative_context.analysis.allocation import allocation, layout_cost
from representative_context.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument("--k", type=int, choices=(14, 16), required=True)
    parser.add_argument(
        "--policy", choices=("structural", "uniform", "no_density"), required=True
    )
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if min(args.height, args.width) < 32:
        parser.error("Each image dimension must be at least 32.")
    plan = allocation(args.case, args.k, args.policy)
    write_json(
        args.output,
        dict(
            plan=plan,
            input_shape=[args.height, args.width],
            actual_layout=layout_cost(
                args.case, args.height, args.width, plan["capacities"]
            ),
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()

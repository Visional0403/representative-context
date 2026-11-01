"""Recompute routing, coverage and controlled-layout results on CPU."""

import argparse
from pathlib import Path
from _bootstrap import ROOT
from representative_context.analysis.routing_statistics import controlled, summarize
from representative_context.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements", type=Path, default=ROOT / "results/01_context"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for case in ("segformer_uavid", "twins_cityscapes"):
        report[case] = dict(
            routing=summarize(args.measurements / "routing" / case),
            controlled=controlled(args.measurements / "controlled" / case),
        )
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

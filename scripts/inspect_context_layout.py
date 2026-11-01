"""Inspect complete per-query capacity, local-context and global-anchor invariants."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.controls.context_layout import audit, construct
from representative_context.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-side", type=int, default=64)
    parser.add_argument("--query-side", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--cluster", type=int, choices=(4, 2, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    layout = construct(args.kv_side, args.capacity, args.cluster, args.seed)
    report = audit(*layout, args.kv_side, args.query_side)
    write_json(
        args.output,
        {
            "kv_side": args.kv_side,
            "query_side": args.query_side,
            "cluster": args.cluster,
            "seed": args.seed,
            **report,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

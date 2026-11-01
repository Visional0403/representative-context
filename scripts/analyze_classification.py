"""Recompute classification competition and signed accuracy changes from scores."""

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.framework import competition, pool_classification
from representative_context.runtime import read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/03_theory/classification"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    manifest = read_config(args.input / "manifest.json")
    rows = []
    for window in manifest["windows"]:
        with np.load(args.input / window["path"], allow_pickle=False) as values:
            statistics, _ = competition(
                values["dense_logits"], values["sparse_logits"], values["labels"]
            )
        rows.append({"window_index": window["window_index"], "statistics": statistics})
    write_json(
        args.output,
        {
            "statistical_unit": manifest["statistical_unit"],
            "windows": rows,
            "pooled": pool_classification([row["statistics"] for row in rows]),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

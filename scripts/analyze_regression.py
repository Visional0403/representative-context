"""Recompute signed regression-loss changes from complete paired flow outputs."""

import argparse
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.framework import regression_decomposition
from representative_context.runtime import read_config, write_json


def analyze(directory, manifest):
    rows = []
    for pair in manifest["pairs"]:
        with np.load(directory / pair["arrays"], allow_pickle=False) as values:
            for pattern in manifest["layouts"]:
                for arm in manifest["interventions"]:
                    stats, _ = regression_decomposition(
                        values["dense_prediction"],
                        values[f"{pattern}_{arm}"],
                        values["target"],
                    )
                    rows.append(
                        dict(
                            scene=pair["scene"],
                            layout=pattern,
                            intervention=arm,
                            **stats,
                        )
                    )
    return dict(
        case=manifest["case"],
        statistical_unit=manifest["statistical_unit"],
        vector_mse=manifest["vector_mse"],
        records=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results/03_theory/regression/gma_sintel",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    write_json(
        args.output,
        analyze(args.input_dir, read_config(args.input_dir / "measurements.json")),
    )
    print(args.output)


if __name__ == "__main__":
    main()

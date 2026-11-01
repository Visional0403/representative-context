"""Recompute paired classification competition across retained-capacity levels."""

import argparse
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.context import task_counts
from representative_context.analysis.framework import competition, pool_classification
from representative_context.runtime import read_config, write_json


def analyze(directory, manifest):
    rows = []
    for window in manifest["windows"]:
        with np.load(directory / window["arrays"], allow_pickle=False) as values:
            for exponent in manifest["resource_exponents"]:
                stats, _ = competition(
                    values["dense_logits"],
                    values[f"k{exponent}_logits"],
                    values["sampled_labels"],
                )
                full = window["full_statistics"][f"k{exponent}"]
                cube = (
                    np.asarray(window["confusion_cubes"][f"k{exponent}"])
                    if "confusion_cubes" in window
                    else task_counts(
                        values["labels"],
                        values["dense_prediction"],
                        values[f"k{exponent}_prediction"],
                        8,
                    )
                )
                gt, dense, current = np.indices(cube.shape)
                counts = dict(
                    pixels=int(cube.sum()),
                    label_changes=int(cube[dense != current].sum()),
                    new_errors=int(cube[(gt == dense) & (gt != current)].sum()),
                    corrections=int(cube[(gt != dense) & (gt == current)].sum()),
                    wrong_class_switches=int(
                        cube[(gt != dense) & (gt != current) & (dense != current)].sum()
                    ),
                )
                for name, value in counts.items():
                    assert full[name] == value, name
                assert full["label_changes"] <= full["truncated_competition_sum"] + 1e-9
                rows.append(
                    dict(
                        index=window["index"],
                        scene=window["scene"],
                        k=exponent,
                        sampled=stats,
                        full=full,
                    )
                )
    pooled = []
    for exponent in manifest["resource_exponents"]:
        selected = [row for row in rows if row["k"] == exponent]
        pooled.append(
            dict(
                k=exponent,
                windows=len(selected),
                sampled=pool_classification([row["sampled"] for row in selected]),
                full=pool_classification([row["full"] for row in selected]),
            )
        )
    return dict(
        case=manifest["case"],
        complete_windows=manifest["complete_windows"],
        per_window=rows,
        pooled=pooled,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    directory = (
        args.input_dir or ROOT / f"results/03_theory/budget_competition/{args.case}"
    )
    write_json(
        args.output, analyze(directory, read_config(directory / "measurements.json"))
    )
    print(args.output)


if __name__ == "__main__":
    main()

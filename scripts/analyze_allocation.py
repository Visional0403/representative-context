"""Recompute matched-compute comparisons and source-group budget responses."""

import argparse
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.allocation import miou, paired_bootstrap, summarize
from representative_context.runtime import read_config, write_json


def compare(report):
    arms = list(report["miou_percent"])
    scenes = sorted({row["scene"] for row in report["images"]})
    hist = np.zeros((len(scenes), len(arms), 8, 8), dtype=np.int64)
    for row in report["images"]:
        hist[scenes.index(row["scene"])] += np.asarray(
            [row["configurations"][arm]["confusion_matrix"] for arm in arms]
        )
    point, samples = paired_bootstrap(hist)
    assert (
        np.max(np.abs(point - np.array([report["miou_percent"][arm] for arm in arms])))
        < 1e-10
    )
    comparisons = []
    for plan in report["plans"]:
        reference = arms.index(f"k{plan['k']}_structural")
        current = arms.index(f"k{plan['k']}_{plan['policy']}")
        comparisons.append(
            dict(
                **plan,
                miou_percent=float(point[current]),
                dense_change_pp=float(point[current] - point[0]),
                structural_change_pp=float(point[current] - point[reference]),
                structural_change_interval_pp=np.quantile(
                    samples[:, current] - samples[:, reference], [0.025, 0.975]
                ).tolist(),
                per_group_miou_percent=miou(hist[:, current]).tolist(),
                per_group_structural_change_pp=(
                    miou(hist[:, current]) - miou(hist[:, reference])
                ).tolist(),
            )
        )
    return dict(
        case=report["case"],
        phase=report["phase"],
        complete_scope=report["complete_scope"],
        evaluated_images=len(report["images"]),
        source_groups=scenes,
        configurations=arms,
        per_group_confusion=hist.tolist(),
        pooled_miou_percent=point.tolist(),
        comparisons=comparisons,
        bootstrap_repeats=10000,
        bootstrap_seed=20261022,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("allocation", "curves"), required=True)
    parser.add_argument(
        "--case",
        choices=("segformer_uavid", "twins_uavid", "segformer_cityscapes"),
        required=True,
    )
    parser.add_argument("--phase", choices=("full", "diagnostic"), default="full")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.task == "allocation":
        if args.case == "segformer_cityscapes":
            parser.error("Matched-compute allocation uses the two UAVid models.")
        path = (
            args.input
            or ROOT / f"results/04_budget/allocation/{args.case}_{args.phase}.json"
        )
        result = compare(read_config(path))
    else:
        directory = args.input or ROOT / f"results/04_budget/grouped_curves/{args.case}"
        manifest = read_config(directory / "measurements.json")
        with np.load(directory / manifest["arrays"], allow_pickle=False) as arrays:
            result = summarize(
                arrays["confusion"],
                manifest["configurations"],
                manifest["source_groups"],
            )
        result.update(
            case=args.case,
            evaluation_scope=manifest["evaluation_scope"],
            bootstrap_repeats=10000,
            bootstrap_seed=20261022,
        )
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

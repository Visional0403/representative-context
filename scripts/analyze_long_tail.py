"""Recompute native-image classification and optical-flow difficulty analyses."""

import argparse
import math
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.long_tail import (
    accuracy_interval,
    classification,
    flow_combine,
    flow_derive,
    flow_interval,
    flow_masks,
    human_iou_interval,
    human_quintiles,
    paired_flow,
)
from representative_context.runtime import read_config, write_json


def uavid(directory):
    metadata = read_config(directory / "measurements.json")
    rows = metadata["images"]
    groups = [row["scene"] for row in rows]
    result = {}
    with np.load(directory / metadata["arrays"], allow_pickle=False) as values:
        baseline = [classification(cube.sum(0))["dense"] for cube in values["k14"]]
        eligible = [
            index for index, row in enumerate(baseline) if row["human_gt_pixels"] > 0
        ]
        eligible.sort(key=lambda index: (baseline[index]["human_iou_percent"], index))
        first, second = math.ceil(0.2 * len(eligible)), math.ceil(0.6 * len(eligible))
        subsets = dict(
            all_images=list(range(len(rows))),
            hardest20=eligible[:first],
            middle40=eligible[first:second],
            easiest40=eligible[second:],
        )
        for name in ("k14", "k16"):
            cubes = values[name]
            image_results = {}
            for label, indices in subsets.items():
                pooled = classification(cubes[indices].sum((0, 1)))
                pooled.update(
                    images=len(indices),
                    image_indices=indices,
                    human_iou_ci=human_iou_interval(cubes, groups, indices),
                )
                image_results[label] = pooled
            bins = []
            for selected in ([0], [1], [2], [3], [4], [0, 1]):
                per_image = [classification(cube[selected].sum(0)) for cube in cubes]
                pooled = classification(cubes[:, selected].sum((0, 1)))
                pooled["accuracy_ci"] = accuracy_interval(
                    [
                        (
                            row["dense"]["valid_pixels"],
                            row["new_errors"],
                            row["corrections"],
                        )
                        for row in per_image
                    ],
                    groups,
                )
                pooled.update(margin_bins=selected)
                bins.append(pooled)
            result[name] = dict(image_subsets=image_results, margin_subsets=bins)
    with np.load(directory / metadata["human_arrays"], allow_pickle=False) as values:
        result["human_quintiles"] = dict(
            all=human_quintiles(values), initially_correct=human_quintiles(values, True)
        )
    return dict(case="segformer_uavid", complete_images=len(rows), results=result)


def gma(directory):
    metadata = read_config(directory / "measurements.json")
    rows = metadata["pairs"]
    ordered = sorted(
        rows,
        key=lambda row: (
            -flow_derive(row["subsets"]["all_pixels"])["dense_epe"],
            row["scene"],
            row["frame"],
        ),
    )
    cut = math.ceil(0.2 * len(rows))
    results = {}
    for label, selected in (
        ("all_pairs", rows),
        ("hardest20", ordered[:cut]),
        ("remaining80", ordered[cut:]),
    ):
        subsets = {}
        for name in rows[0]["subsets"]:
            items = [row["subsets"][name] for row in selected]
            summary = flow_combine(items)
            changes = [flow_derive(row)["epe_change"] for row in items if row["pixels"]]
            summary.update(
                scene_ci=flow_interval(items, [row["scene"] for row in selected]),
                pair_directions=dict(
                    nonempty_pairs=len(changes),
                    increased=sum(value > 0 for value in changes),
                    decreased=sum(value < 0 for value in changes),
                    unchanged=sum(value == 0 for value in changes),
                ),
            )
            subsets[name] = summary
        results[label] = dict(
            pairs=len(selected),
            pair_indices=[row["index"] for row in selected],
            subsets=subsets,
        )
    array_checks = 0
    for row in rows:
        if "arrays" not in row:
            continue
        with np.load(directory / row["arrays"], allow_pickle=False) as values:
            baseline, magnitude, masks, threshold = flow_masks(
                values["dense_prediction"], values["target"], values["occlusion"]
            )
            assert threshold == row["dense_error_quantile80"]
            current = np.linalg.norm(
                values["sparse_prediction"].astype(float)
                - values["target"].astype(float),
                axis=-1,
            )
            for name, mask in masks.items():
                measured = paired_flow(baseline, current, magnitude, mask)
                for key, value in row["subsets"][name].items():
                    assert np.isclose(measured[key], value, rtol=1e-12, atol=1e-9), (
                        row["index"],
                        name,
                        key,
                    )
                array_checks += 1
    return dict(
        case="gma_sintel",
        complete_pairs=len(rows),
        sample_subset_checks=array_checks,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results/02_mechanisms/long_tail"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        dict(
            segformer_uavid=uavid(args.input_dir / "segformer_uavid"),
            gma_sintel=gma(args.input_dir / "gma_sintel"),
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()

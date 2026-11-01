"""Recompute complete extent-control metrics and fixed-input aggregation summaries."""

import argparse
import math
from pathlib import Path
import statistics
import numpy as np
from _bootstrap import ROOT
from representative_context.evaluation.mmseg import metrics as mmseg_metrics
from representative_context.evaluation.segmentation import segmentation_metrics
from representative_context.analysis.layout_intervals import paired_interval
from representative_context.runtime import read_config, write_json


def analyze(directory):
    metadata = read_config(directory / "measurements.json")
    native = metadata["case"] == "segformer_uavid"
    rows = []
    paired = []
    with np.load(directory / metadata["arrays"], allow_pickle=False) as arrays:
        for layout in metadata["layouts"]:
            counts = arrays[layout["pattern"]]
            assert len(counts) == len(metadata["images"])
            result = (
                segmentation_metrics(counts.sum(0))
                if native
                else mmseg_metrics(counts.sum(0))
            )
            assert abs(result["miou_percent"] - layout["miou_percent"]) < 1e-10
            timing = layout["timing"]
            assert (
                abs(statistics.median(timing["timing_ms"]) - timing["median_ms"])
                < 1e-10
            )
            assert (
                abs(statistics.pstdev(timing["timing_ms"]) - timing["stdev_ms"]) < 1e-10
            )
            rows.append(dict(pattern=layout["pattern"], **result, timing=timing))
            if not native and layout["pattern"].startswith("extent"):
                paired.append(dict(pattern=layout["pattern"], **paired_interval(
                    arrays["extent0"], counts, metadata["images"], "images", repeats=2000)))
    summaries = {}
    for pattern in metadata["patterns"]:
        stages = {}
        for stage in range(1, 5):
            values = [
                block["layouts"][pattern]
                for window in metadata["windows"]
                for block in window["blocks"]
                if block["stage"] == stage
            ]
            stages[str(stage)] = dict(
                mean_aggregation_relative_l2=statistics.mean(
                    row["aggregation_relative_l2"] for row in values
                ),
                pooled_aggregation_relative_l2=math.sqrt(
                    sum(row["aggregation_error_squared_sum"] for row in values)
                    / sum(row["dense_aggregation_squared_sum"] for row in values)
                ),
                mean_retained_probability_mass=statistics.mean(
                    row["retained_dense_probability_mass"] for row in values
                ),
            )
        summaries[pattern] = stages
    return dict(
        case=metadata["case"],
        complete_validation=True,
        evaluated_images=len(metadata["images"]),
        layouts=rows,
        paired_intervals=paired,
        aggregation_by_stage=summaries,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results/01_context/extent"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        {
            case: analyze(args.input_dir / case)
            for case in ("segformer_uavid", "twins_cityscapes")
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()

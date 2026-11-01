"""Check complete segmentation capacity curves against their scientific records."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from _bootstrap import ROOT


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9):
        raise ValueError(f"Measurement mismatch: {actual} versus {expected}")


def metric(counts):
    counts = np.asarray(counts, dtype=np.float64)
    intersection = counts.diagonal()
    union = counts.sum(0) + counts.sum(1) - intersection
    return float(100 * np.mean(intersection[union > 0] / union[union > 0]))


def analyze(data):
    result = []
    calls = 0
    for point in data["points"]:
        accuracy = point["accuracy"]
        summary = point["summary"]
        expected_images = 500 if "cityscapes" in point["case"] else 70
        assert accuracy["complete_split"] and accuracy["images"] == expected_images
        for mode, row in accuracy["methods"].items():
            close(row["miou_percent"], summary[f"{mode}_miou_percent"])
            if "confusion_matrix" in row:
                close(metric(row["confusion_matrix"]), row["miou_percent"])
            if "per_class_iou_percent" in row:
                values = [
                    v
                    for v in row["per_class_iou_percent"]
                    if v is not None and math.isfinite(v)
                ]
                close(statistics.mean(values), row["miou_percent"])
        if "per_image" in accuracy:
            rows = accuracy["per_image"]
            assert len(rows) == expected_images
            close(
                metric(np.sum([row["confusion_matrix"] for row in rows], axis=0)),
                accuracy["methods"]["sparse"]["miou_percent"],
            )
        timed = {}
        available = {}
        for mode in ("dense", "sparse"):
            runs = [row["methods"][mode] for row in point["timing"]]
            samples = [number for row in runs for number in row.get("times_ms", [])]
            available[mode] = bool(samples)
            if samples:
                timed[mode] = statistics.median(samples)
                calls += len(samples)
                for row in runs:
                    close(statistics.median(row["times_ms"]), row["median_ms"])
            else:
                assert len(runs) == 1
                timed[mode] = runs[0]["median_ms"]
            close(timed[mode], summary[f"{mode}_median_ms"])
        close(timed["dense"] / timed["sparse"], summary["speedup"])
        result.append(
            dict(
                case=point["case"],
                k=point["k"],
                timing_width=point["timing_width"],
                complete_images=expected_images,
                raw_calls_available=available,
                **summary,
            )
        )
    return dict(points=result, point_count=len(result), retained_timing_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "results/04_budget/measurements/capacity_curves.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.data.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['point_count']} complete capacity-curve points.")


if __name__ == "__main__":
    main()

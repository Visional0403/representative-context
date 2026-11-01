"""Recompute complete Cityscapes selector counts and matched selector timings."""

import argparse
from pathlib import Path
import statistics
import numpy as np
from _bootstrap import ROOT
from representative_context.analysis.layout_intervals import miou, sufficient, paired_interval
from representative_context.runtime import read_config, write_json


def analyze(directory):
    document = read_config(directory / "measurements.json")
    scores = []
    comparisons = []
    with np.load(directory / document["arrays"], allow_pickle=False) as arrays:
        reference = next(row for row in document["configurations"] if row["method"] == "dispersed32")
        for row in document["configurations"]:
            counts = arrays[row["array"]]
            assert counts.shape == (500, 19, 19)
            score = float(miou(sufficient(counts).sum(0)))
            assert abs(score - row["miou_percent"]) < 1e-10
            scores.append(dict(row, recomputed_miou_percent=score))
            if row["method"] in ("topk_exact", "topk_proxy8") and row["k"] == 14:
                result = paired_interval(arrays[reference["array"]], counts, document["images"], "cities")
                recorded = next(value for value in document["selector_intervals"] if value["method"] == row["method"])
                np.testing.assert_allclose(result["interval95"], recorded["interval95"], atol=1e-12, rtol=0)
                comparisons.append(dict(case="twins_cityscapes", method=row["method"], **result))
    uavid = ROOT / "results/segformer_uavid/evaluations"
    reference = read_config(uavid / "dispersed32_2048_k14.json")
    images = [row["image"] for row in reference["images"]]
    first = np.asarray([row["confusion_matrix"] for row in reference["images"]])
    for method in ("exact", "proxy8"):
        candidate = read_config(uavid / f"topk_{method}_2048_k14.json")
        assert images == [row["image"] for row in candidate["images"]]
        second = np.asarray([row["confusion_matrix"] for row in candidate["images"]])
        comparisons.append(dict(case="segformer_uavid", method="topk_" + method,
                                **paired_interval(first, second, images, "sequences")))
    timing_records = read_config(directory / "timings.json")["records"]
    for record in timing_records:
        for methods in record["measurements"].values():
            for values in methods.values():
                assert len(values["timing_ms"]) == record["rounds"] * record["repeats_per_round"]
                assert abs(statistics.median(values["timing_ms"]) - values["median_ms"]) < 1e-10
                assert abs(statistics.mean(values["timing_ms"]) - values["mean_ms"]) < 1e-10
                assert abs(statistics.pstdev(values["timing_ms"]) - values["stdev_ms"]) < 1e-10
    return dict(accuracy=scores, paired_selector_comparisons=comparisons, timing_records=timing_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/01_context/selection_evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.input_dir))
    print(args.output)


if __name__ == "__main__":
    main()

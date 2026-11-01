"""Recompute the published task curves from complete paired scientific records."""

import argparse
import math
from pathlib import Path
import statistics
import numpy as np
from _bootstrap import ROOT
from representative_context.runtime import read_config, write_json
from representative_context.analysis.timing_records import validate_report
from representative_context.analysis.prediction_records import primary_metric


def accuracy(case, document, mode):
    if document.get("schema") == "paired_prediction_v1":
        return primary_metric(document["case"], [row[mode] for row in document["records"]])
    if "counts" in document:
        row = document["counts"]["dense_sdpa" if mode == "dense" else "sparse_point32"]
        return 200 * row["TP"] / (2 * row["TP"] + row["FP"] + row["FN"])
    records = document["records"]
    if case == "gma_sintel":
        return sum(row["epe"] * row["pixels"] for row in records) / sum(
            row["pixels"] for row in records
        )
    if case == "glpn_nyu":
        return statistics.mean(row["abs_rel"] for row in records)
    if case in ("polyp_kvasir", "tcd_tiled"):
        label = "dense_sdpa" if mode == "dense" else "sparse_point32"
        values = [row[label] for row in records]
        if case == "polyp_kvasir":
            return (
                statistics.mean(
                    row["author_normalized_binary"]["Dice"] for row in values
                )
                * 100
            )
        hist = np.sum([row["confusion_matrix"] for row in values], axis=0)
        return float(
            100 * hist[1, 1] / (hist[1, :].sum() + hist[:, 1].sum() - hist[1, 1])
        )
    if case in ("hitnet", "iml", "trufor"):
        key = "Dice" if case == "hitnet" else "F1"
        return statistics.mean(row["metrics"][mode][key] for row in records) * 100
    values = [row[mode] for row in records]
    if case in ("div2k", "gopro", "cave", "qb"):
        return statistics.mean(row["PSNRdB"] for row in values)
    if case == "geco":
        return statistics.mean(row["absolute_error"] for row in values)
    if case == "gamus":
        return math.sqrt(
            sum(row["squared_error_sum"] for row in values)
            / sum(row["valid_pixels"] for row in values)
        )
    if case == "pi3":
        return statistics.mean(
            frame["relative_point_error"] for row in values for frame in row["frames"]
        )
    raise ValueError(case)


def latency(document, mode):
    if document.get("task") == "timing":
        return validate_report(document)["latency_ms"]["dense" if mode == "dense" else "sparse"]
    if "timings" in document:
        return statistics.median(
            row["milliseconds"] for row in document["timings"] if row["mode"] == mode
        )
    timing = document["timing"]
    if "stratum_rounds" in timing:
        rounds = {}
        for row in timing["stratum_rounds"]:
            alias = row["execution_aliases"].get(mode, mode)
            rounds.setdefault(row["round"], 0.0)
            rounds[row["round"]] += row["weight"] * statistics.median(
                row["timings"][alias]["timing_ms"]
            )
        return statistics.median(rounds.values())
    if "round" in timing:
        return statistics.median(
            statistics.median(row[mode]["timing_ms"]) for row in timing["round"]
        )
    if "whole_model" in timing:
        return statistics.median(
            timing["whole_model"][
                "dense_sdpa" if mode == "dense" else "sparse_point32"
            ]["timing_ms"]
        )
    if "raw_ms" in timing:
        raw = timing["raw_ms"]
        if "complete_model" in raw:
            rows = raw["complete_model"]
        else:
            rows = raw["full_model"]
        key = (
            "dense"
            if mode == "dense" and "dense" in rows
            else (
                "dense_sdpa"
                if mode == "dense"
                else "sparse" if "sparse" in rows else "sparse_point32"
            )
        )
        values = rows[key]
        return (
            statistics.median(value for row in values for value in row)
            if isinstance(values[0], list)
            else statistics.median(values)
        )
    raise ValueError("Unrecognized timing record.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=ROOT / "results/06_generalization/measurements",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=ROOT / "results/06_generalization/budget_curves.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for case, curve in read_config(args.curves).items():
        folder = args.measurements / ("geco2" if case == "geco" else case)
        index = read_config(folder / "index.json")
        records = [(row, read_config(folder / row["file"])) for row in index["files"]]
        points = []
        for point in curve["points"]:
            k = point["k"]
            candidates = [
                (row, document)
                for row, document in records
                if k in row["resource_exponents"]
                and ("records" in document or "counts" in document)
            ]
            if not candidates and point.get("identical_execution"):
                candidates = [
                    (row, document)
                    for row, document in records
                    if "records" in document or "counts" in document
                ]
            values = []
            for row, document in candidates:
                try:
                    if (case in ("gma_sintel", "glpn_nyu")
                            and document.get("schema") != "paired_prediction_v1"):
                        baseline = [
                            item
                            for _, item in records
                            if str(item.get("execution_mode", "")).startswith("dense")
                            and "records" in item
                        ]
                        assert len(baseline) == 1, (case, "dense baseline")
                        dense = accuracy(case, baseline[0], "dense")
                    else:
                        dense = accuracy(case, document, "dense")
                    sparse = (
                        dense
                        if point.get("identical_execution")
                        else accuracy(case, document, f"k{k}")
                    )
                    values.append((row, dense, sparse))
                except KeyError:
                    continue
            assert values, (case, k, "accuracy")
            if (
                len(values) > 1
                and point.get("identical_execution")
                and not any(k in row["resource_exponents"] for row, _ in records)
            ):
                values = values[:1]
            assert len(values) == 1, (
                case,
                k,
                "unique declared accuracy source",
                [value[0] for value in values],
            )
            row, dense, sparse = values[0]
            tolerance = 2e-7 if case == "gma_sintel" else 1e-10
            assert np.isclose(
                dense, point["dense_metric"], atol=tolerance, rtol=1e-10
            ), (case, k, dense, point["dense_metric"])
            assert np.isclose(
                sparse, point["sparse_metric"], atol=tolerance, rtol=1e-10
            ), (case, k, sparse, point["sparse_metric"])
            if "dense_median_ms" not in point:
                assert point["identical_execution"] and point["speedup"] == 1
                points.append(
                    dict(
                        k=k,
                        dense_metric=dense,
                        sparse_metric=sparse,
                        identical_execution=True,
                        latency_source="dense_baseline_reuse", speedup=1.0,
                        accuracy_file=row["file"],
                    )
                )
                continue
            timing = []
            for record, document in records:
                if k not in record["resource_exponents"] or not (
                    "timing" in document or "timings" in document or document.get("task") == "timing"
                ):
                    continue
                timing.append(
                    (record, latency(document, "dense"), latency(document, f"k{k}"))
                )
            assert timing, (case, k, "timing")
            assert len(timing) == 1, (case, k, "unique declared timing source")
            timing_row, dense_ms, sparse_ms = timing[0]
            measured_sparse_ms = sparse_ms
            reused = point.get("latency_source") == "dense_baseline_reuse"
            if reused:
                assert point["identical_execution"] and point["speedup"] == 1
                observed = point.get("observed_sparse_median_ms")
                if observed is not None:
                    assert np.isclose(observed, measured_sparse_ms, atol=1e-8, rtol=1e-10)
                sparse_ms = dense_ms
            assert np.isclose(
                dense_ms, point["dense_median_ms"], atol=1e-8, rtol=1e-10
            ), (case, k, dense_ms, point["dense_median_ms"])
            assert np.isclose(
                sparse_ms, point["sparse_median_ms"], atol=1e-8, rtol=1e-10
            ), (case, k, sparse_ms, point["sparse_median_ms"])
            measured_ratio = dense_ms / measured_sparse_ms
            if point.get("identical_execution"):
                assert point["speedup"] == 1.0, (case, k, "identical execution endpoint")
                assert abs(sparse - dense) < tolerance
            elif point.get("round_speedups"):
                assert np.isclose(statistics.median(point["round_speedups"]), point["speedup"], rtol=1e-10)
            else:
                assert np.isclose(measured_ratio, point["speedup"], rtol=1e-10), (case, k, "speedup")
            higher_is_better = case in {"polyp_kvasir", "hitnet", "tcd_tiled", "changeformer", "iml", "trufor", "div2k", "gopro", "cave", "qb"}
            relative_loss = 100 * (dense - sparse if higher_is_better else sparse - dense) / abs(dense)
            assert abs(relative_loss - point["relative_metric_loss_percent"]) < 1e-5, (case, k, "relative metric change")
            points.append(
                dict(
                    k=k,
                    dense_metric=dense,
                    sparse_metric=sparse,
                    dense_median_ms=dense_ms,
                    sparse_median_ms=sparse_ms,
                    latency_source=point.get("latency_source", "paired_measurement"),
                    observed_sparse_median_ms=point.get("observed_sparse_median_ms",
                        None if reused else measured_sparse_ms),
                    accuracy_file=row["file"],
                    timing_file=timing_row["file"],
                )
            )
        result[case] = points
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

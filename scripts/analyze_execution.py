"""Recompute exact interactions, timing distributions and complete-model costs."""

import argparse
from pathlib import Path
import statistics
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.execution import distribution
from representative_context.runtime import read_config, write_json


def summarize(raw):
    pure = raw["pure"]
    comparison = raw["comparison"]
    block_count = len(pure["blocks"])
    assert (
        block_count
        == len(comparison["blocks"])
        == (8 if raw["case"] == "segformer_uavid" else 9)
    )
    for before, after in zip(pure["blocks"], comparison["blocks"]):
        for key in (
            "Q",
            "KV",
            "groups",
            "sparse_edges_per_head",
            "heads",
            "head_dimension",
        ):
            assert before[key] == after[key]
        check = after["numerical_check"]
        assert check["full_relative_l2"] <= 1e-4
        assert (
            max(
                check["full_max_abs_error"],
                check["fp64_reordered_max_abs_error"],
                check["fp64_indexed_max_abs_error"],
            )
            <= 1e-3
        )
    counts = {
        name: sum(row[name] for row in pure["blocks"])
        for name in ("dense_flops", "sparse_flops", "index_bytes")
    }
    counts.update(
        dense_head_weighted_edges=sum(
            row["dense_edges_per_head"] * row["heads"] for row in pure["blocks"]
        ),
        sparse_head_weighted_edges=sum(
            row["sparse_edges_per_head"] * row["heads"] for row in pure["blocks"]
        ),
    )
    pure_times = {
        name: distribution(pure["timing"][name]["timing_ms"])
        for name in ("dense", "indexed")
    }
    compare_times = {
        name: distribution(value["timing_ms"])
        for name, value in comparison["timing"]["methods"].items()
    }
    assert all(
        value["count"] == 60
        for value in list(pure_times.values()) + list(compare_times.values())
    )
    execution = dict(
        distributions=compare_times,
        same_layout_speedup=compare_times["reordered"]["median_ms"]
        / compare_times["indexed"]["median_ms"],
        per_round_same_layout_speedup=[
            statistics.median(row["timing_ms"]["reordered"])
            / statistics.median(row["timing_ms"]["indexed"])
            for row in comparison["timing"]["rounds"]
        ],
        attention_speedup=compare_times["dense"]["median_ms"]
        / compare_times["indexed"]["median_ms"],
    )
    profile = raw["profile"]
    costs = {}
    for mode in ("dense", "indexed"):
        rows = profile["profile_observations"][mode]
        assert len(rows) == 12
        labels = sorted({key for row in rows for key in row["parts_ms"]})
        parts = {
            key: statistics.mean(row["parts_ms"].get(key, 0) for row in rows)
            for key in labels
        }
        total = statistics.mean(row["total_ms"] for row in rows)
        assert abs(sum(parts.values()) - total) < 0.03
        for row in rows:
            assert abs(sum(row["parts_ms"].values()) - row["total_ms"]) < 0.03
        expected = profile["profile_summary"][mode]
        assert abs(expected["mean_total_ms"] - total) < 1e-10
        assert (
            max(abs(expected["mean_parts_ms"][name] - parts[name]) for name in labels)
            < 1e-10
        )
        costs[mode] = dict(
            mean_total_ms=total,
            mean_parts_ms=parts,
            global_aggregation_percent=100 * parts["global_aggregation"] / total,
            other_time_ms=total - parts["global_aggregation"],
            total_distribution=distribution([row["total_ms"] for row in rows]),
        )
    unprofiled = {
        mode: distribution(value["timing_ms"])
        for mode, value in profile["unprofiled_timing"]["methods"].items()
    }
    assert all(value["count"] == 30 for value in unprofiled.values())
    return dict(
        case=raw["case"],
        input_shape=raw["input_shape"],
        k=raw["k"],
        counts=counts,
        pure_attention=dict(
            distributions=pure_times,
            speedup=pure_times["dense"]["median_ms"]
            / pure_times["indexed"]["median_ms"],
        ),
        execution=execution,
        profile=costs,
        unprofiled=unprofiled,
        model_speedup=unprofiled["dense"]["median_ms"]
        / unprofiled["indexed"]["median_ms"],
        correctness=dict(
            blocks=block_count,
            fp64_queries=sum(
                row["numerical_check"]["fp64_queries"] for row in comparison["blocks"]
            ),
            max_relative_l2=max(
                row["numerical_check"]["full_relative_l2"]
                for row in comparison["blocks"]
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    paths = args.input or sorted(
        (ROOT / "results/05_execution/measurements").glob("*.json")
    )
    results = [summarize(read_config(path)) for path in paths]
    write_json(
        args.output,
        dict(records=results, statistical_unit="timing repeat on the stated device"),
    )
    print(args.output)


if __name__ == "__main__":
    main()

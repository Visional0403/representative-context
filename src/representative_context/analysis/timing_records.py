"""Recalculate timing summaries from ordered, explicitly grouped measurements."""

import math
import statistics

import numpy as np


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"Saved {label} disagrees with the original timing calls.")


def call_statistics(values):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing calls must be finite and positive.")
    return dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                stdev_ms=statistics.pstdev(values),
                percentiles_ms=dict(zip(("p10", "p25", "p75", "p90"),
                                       map(float, np.percentile(values, [10, 25, 75, 90])))))


def round_medians(record):
    """Recompute each round from raw calls and verify any saved method summaries."""
    values = record.get("timing_ms")
    if not values:
        raise ValueError("Raw per-call timings are required; this record is summary-only.")
    stats = call_statistics(values)
    for key in ("median_ms", "mean_ms", "stdev_ms"):
        if key in record:
            close(stats[key], record[key], key)
    for key, value in record.get("percentiles_ms", {}).items():
        close(stats["percentiles_ms"][key], value, key)
    calls = record.get("calls")
    if calls is not None:
        if [row["milliseconds"] for row in calls] != values:
            raise ValueError("Call identities and the raw timing array disagree.")
        ids = [(row["round_id"], row["call_id"]) for row in calls]
        if len(set(ids)) != len(ids):
            raise ValueError("Timing calls have duplicate round/call identities.")
        groups = {}
        for row in calls:
            groups.setdefault(row["round_id"], []).append(row)
        if sorted(groups) != list(range(len(groups))):
            raise ValueError("Round IDs must be contiguous and start at zero.")
        for rows in groups.values():
            if [row["call_id"] for row in rows] != list(range(len(rows))):
                raise ValueError("Call IDs must preserve the order within each round.")
        calculated = [statistics.median(row["milliseconds"] for row in groups[i]) for i in sorted(groups)]
    else:
        saved = record.get("round_median_ms")
        if not saved or len(values) % len(saved):
            raise ValueError("Round boundaries cannot be recovered from this timing record.")
        width = len(values) // len(saved)
        calculated = [statistics.median(values[i:i+width]) for i in range(0, len(values), width)]
    if "round_median_ms" in record:
        if len(calculated) != len(record["round_median_ms"]):
            raise ValueError("Saved round count disagrees with call identities.")
        for a, b in zip(calculated, record["round_median_ms"]):
            close(a, b, "round median")
    return calculated


def summarize_strata(strata, timing_summary, speedup_summary):
    """Apply the declared latency and speedup estimators to raw call sequences."""
    modes = list(strata[0]["timings"])
    if not math.isclose(sum(part["weight"] for part in strata), 1.0, abs_tol=1e-9):
        raise ValueError("Input-shape weights must sum to one.")
    recalculated = [{name: round_medians(part["timings"][name]) for name in modes} for part in strata]
    rounds = len(recalculated[0][modes[0]])
    if any(len(row[name]) != rounds for row in recalculated for name in modes):
        raise ValueError("All methods and shape strata must have the same round count.")
    weighted = {name: [sum(part["weight"] * row[name][i] for part, row in zip(strata, recalculated))
                       for i in range(rounds)] for name in modes}
    if timing_summary == "pooled_median":
        if len(strata) != 1:
            raise ValueError("Pooled-call timing requires one declared input shape.")
        latency = {name: statistics.median(strata[0]["timings"][name]["timing_ms"]) for name in modes}
    elif timing_summary == "median_weighted_round_medians":
        latency = {name: statistics.median(values) for name, values in weighted.items()}
    else:
        raise ValueError("Unknown latency summary.")
    result = dict(summary_statistic=timing_summary, speedup_statistic=speedup_summary,
                  latency_ms=latency, weighted_round_median_ms=weighted)
    if set(modes) == {"dense", "sparse"}:
        ratios = [a/b for a, b in zip(weighted["dense"], weighted["sparse"])]
        if speedup_summary == "median_paired_round_ratios":
            speedup = statistics.median(ratios)
        elif speedup_summary == "ratio_of_reported_latencies":
            speedup = latency["dense"] / latency["sparse"]
        else:
            raise ValueError("Unknown speedup summary.")
        result.update(speedup=speedup, paired_round_speedups=ratios)
    return result


def validate_report(document):
    """Check a newly generated timing JSON against its original call records."""
    result = summarize_strata(document["strata"], document["summary_statistic"],
                              document["speedup_statistic"])
    for mode, value in result["latency_ms"].items():
        close(value, document["latency_ms"][mode], f"{mode} latency")
    if "speedup" in result:
        close(result["speedup"], document["speedup"], "speedup")
        saved = document.get("paired_round_speedups")
        if saved is not None:
            if len(saved) != len(result["paired_round_speedups"]):
                raise ValueError("Saved speed ratios have the wrong round count.")
            for a, b in zip(result["paired_round_speedups"], saved):
                close(a, b, "paired round speedup")
    return result

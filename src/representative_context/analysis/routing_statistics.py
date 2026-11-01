"""Recompute matched routing costs and paired spatial-coverage summaries."""

from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from representative_context.analysis.layout_intervals import paired_interval

METHODS = ("point32", "proxy8", "exact", "region", "cover1", "cover16")
METRICS = (
    "mean_normalized_nearest_distance",
    "mean_normalized_largest_gap",
    "mean_macroregion_coverage",
    "mean_adjacent_pairs_per_key",
    "mean_local_3x3_keys",
    "mean_bounding_box_fraction",
    "retained_dense_probability_mass",
    "dense_top1_retention",
)


def read(path):
    return json.loads(Path(path).read_text())


def summarize(directory):
    directory = Path(directory)
    whole = read(directory / "whole.json")
    cost = read(directory / "cost.json")
    coverage = read(directory / "coverage.json")
    report = dict(
        case=whole["case"], whole_model={}, aggregation={}, coverage={}, paired={}
    )
    for method, row in whole["observations"].items():
        timing = row["whole_model_timing"]
        assert abs(np.median(timing["timing_ms"]) - timing["median_ms"]) < 1e-10
        report["whole_model"][method] = dict(
            median_ms=timing["median_ms"],
            recheck_median_ms=row["recheck_timing"]["median_ms"],
            stdev_ms=timing["stdev_ms"],
            peak_allocated_mib=timing["peak_allocated_mib"],
            region_numerical_check=row["region_numerical_check"],
        )
    start, end = [
        report["whole_model"][name]["median_ms"]
        for name in ("dense_start", "dense_end")
    ]
    report["dense_bookend_change_percent"] = (end / start - 1) * 100
    for method in ("dense", "point32", "proxy8", "exact", "region"):
        rows = [row for row in cost["blocks"] if row["method"] == method]
        names = sorted({name for row in rows for name in row["component_median_ms"]})
        components = {
            name: sum(row["component_median_ms"].get(name, 0.0) for row in rows)
            for name in names
        }
        selection = sum(
            value
            for name, value in components.items()
            if name
            in (
                "proxy_projection",
                "pairing_scores",
                "topk_and_indices",
                "region_organization_and_pooling",
            )
        )
        organization = sum(
            value
            for name, value in components.items()
            if name in ("candidate_kv_copy", "restore_query_order", "output_allocation")
        )
        total = sum(components.values())
        report["aggregation"][method] = dict(
            sum_block_medians_ms=sum(
                row["uninstrumented"]["median_ms"] for row in rows
            ),
            component_sum_ms=components,
            instrumented_component_total_ms=total,
            selection_ms=selection,
            explicit_organization_ms=organization,
            selection_fraction=selection / total,
            selection_and_organization_fraction=(selection + organization) / total,
            numerical_checks=[
                row["numerical_checks"] for row in rows if method == "exact"
            ],
        )
    blocks = coverage["blocks"]
    weights = [row["heads"] * row["Q"] for row in blocks]
    for method in METHODS:
        rows = [row["method"][method] for row in blocks]
        values = {
            name: float(np.average([row[name] for row in rows], weights=weights))
            for name in METRICS
        }
        error = sum(
            row["method"][method]["aggregation_error_squared_sum"] * row["Q"] / 64
            for row in blocks
        )
        norm = sum(
            row["method"][method]["dense_aggregation_squared_sum"] * row["Q"] / 64
            for row in blocks
        )
        values["weighted_aggregation_relative_l2"] = math.sqrt(error / norm)
        values["mean_block_aggregation_relative_l2"] = float(
            np.mean([row["aggregation_relative_l2"] for row in rows])
        )
        report["coverage"][method] = values
    with np.load(directory / coverage["arrays"], allow_pickle=False) as arrays:
        for method in ("proxy8", "exact", "region"):
            paired, total, lower, broader = [], 0.0, 0.0, 0.0
            for row in blocks:
                fixed = arrays[row["method"]["point32"]["metrics_key"]]
                routed = arrays[row["method"][method]["metrics_key"]]
                weight = row["Q"] / 64
                total += fixed.shape[0] * fixed.shape[1] * weight
                lower += np.count_nonzero(fixed[..., 8] < routed[..., 8]) * weight
                broader += np.count_nonzero(fixed[..., 0] < routed[..., 0]) * weight
                paired.append(
                    dict(
                        image_index=row["image_index"],
                        stage=row["stage"],
                        block=row["block"],
                        point32_aggregation_relative_l2=row["method"]["point32"][
                            "aggregation_relative_l2"
                        ],
                        routed_aggregation_relative_l2=row["method"][method][
                            "aggregation_relative_l2"
                        ],
                        point32_broader_query_heads=int(
                            np.count_nonzero(fixed[..., 0] < routed[..., 0])
                        ),
                        point32_lower_error_query_heads=int(
                            np.count_nonzero(fixed[..., 8] < routed[..., 8])
                        ),
                        sampled_query_heads=fixed.shape[0] * fixed.shape[1],
                    )
                )
            report["paired"][method] = dict(
                blocks=paired,
                point32_broader_weighted_fraction=broader / total,
                point32_lower_error_weighted_fraction=lower / total,
                point32_lower_error_blocks=sum(
                    row["point32_aggregation_relative_l2"]
                    < row["routed_aggregation_relative_l2"]
                    for row in paired
                ),
                total_blocks=len(paired),
            )
        correlations, query_weights = [], []
        for row in blocks:
            values = np.stack(
                [arrays[row["method"][name]["metrics_key"]] for name in METHODS],
                axis=-2,
            )
            x, y = rankdata(values[..., 0], axis=-1), rankdata(values[..., 8], axis=-1)
            x, y = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
            denominator = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
            corr = np.divide(
                (x * y).sum(-1),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            correlations.extend(corr.reshape(-1).tolist())
            query_weights.extend([row["Q"] / 64] * corr.size)
        report["within_query_head_rank_correlation"] = dict(
            weighted_mean=float(np.average(correlations, weights=query_weights)),
            positive_weighted_fraction=float(
                np.average(np.array(correlations) > 0, weights=query_weights)
            ),
            unit="Within the same dense query and head, rank six methods; descriptive statistic.",
        )
    report["inputs"] = coverage["inputs"]
    return report


def controlled(directory):
    directory = Path(directory)
    metadata = read(directory / "measurements.json")
    rows = []
    with np.load(directory / metadata["arrays"], allow_pickle=False) as arrays:
        for layout in metadata["layouts"]:
            counts = arrays[f"cluster{layout['cluster']}"]
            assert counts.shape[0] == len(metadata["images"])
            hist = counts.sum(0)
            tp = np.diag(hist)
            union = hist.sum(0) + hist.sum(1) - tp
            values = (
                np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
                * 100
            )
            assert abs(np.nanmean(values) - layout["miou_percent"]) < 1e-10
            assert np.allclose(
                values, layout["per_class_iou_percent"], atol=1e-10, rtol=0
            )
            rows.append(dict(layout, evaluated_images=len(counts)))
    comparisons = []
    grouping = "sequences" if metadata["case"] == "segformer_uavid" else "cities"
    with np.load(directory / metadata["arrays"], allow_pickle=False) as arrays:
        for first, second in combinations(metadata["layouts"], 2):
            a, b = first["cluster"], second["cluster"]
            comparisons.append(dict(reference_cluster=a, candidate_cluster=b,
                **paired_interval(arrays[f"cluster{a}"], arrays[f"cluster{b}"],
                                  metadata["images"], grouping, **{key: value for key, value in
                                      metadata.get("bootstrap", {}).items() if key != "grouping"})))
    return dict(case=metadata["case"], complete_validation=True, layouts=rows,
                paired_comparisons=comparisons)

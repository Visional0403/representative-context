"""Recompute cross-task aggregation evidence by independent source group."""

import math
import numpy as np
from representative_context.analysis.cross_task import FRACTIONS

EXPECTED = {
    "gma": (23, ["u6", "u12"]),
    "fusformer": (11, ["b1", "b3", "b6"]),
    "iml": (24, ["b3", "b6", "b9", "b12"]),
}


def ci(values, draws):
    return np.quantile(np.asarray(values)[draws].mean(1), [0.025, 0.975]).tolist()


def analyze(case, directory, raw):
    path = directory / "measurements.json"
    declared = raw["declared_scenes"]
    count, expected_blocks = EXPECTED[case]
    assert (
        raw["status"] == "complete"
        and raw["complete_scenes"]
        and (raw["declared_scene_count"] == count)
    )
    assert [r["id"] for r in raw["scenes"]] == [r["id"] for r in declared]
    assert len({r["source_group"] for r in declared}) == count
    assert (
        raw["parameters_preserved"]
        and raw["environment"]["precision"] == "FP32"
        and (not raw["environment"]["TF32"])
    )
    scenes, checked_outputs, maxabs, maxrel, native_max, cpu_checks = (
        [],
        0,
        0.0,
        0.0,
        0.0,
        [],
    )
    for scene in raw["scenes"]:
        archive = path.parent / scene["arrays"]
        assert [b["block"] for b in scene["blocks"]] == expected_blocks
        blocks = []
        with np.load(archive, allow_pickle=False) as arrays:
            for block in scene["blocks"]:
                assert block["capacity"] == raw["settings"]["capacity"]
                ref = arrays[block["block"] + "_dense"].astype(np.float64)
                assert ref.shape == (
                    block["heads"],
                    len(block["query_indices"]),
                    block["head_dimension"],
                )
                assert np.isfinite(ref).all()
                denominator = np.square(ref).sum()
                assert denominator > 0
                per_fraction = {f: [] for f in FRACTIONS}
                for row in block["interventions"]:
                    errors = {}
                    for cluster in (1, 4):
                        output = arrays[row["array_prefix"] + f"_c{cluster}"].astype(
                            np.float64
                        )
                        assert output.shape == ref.shape and np.isfinite(output).all()
                        errors[str(cluster)] = math.sqrt(
                            np.square(output - ref).sum() / denominator
                        )
                        assert math.isclose(
                            errors[str(cluster)],
                            row["aggregation_relative_l2"][str(cluster)],
                            rel_tol=1e-10,
                            abs_tol=1e-11,
                        )
                        checked_outputs += 1
                    sums = arrays[row["array_prefix"] + "_correlation_sums"]
                    counts = arrays[row["array_prefix"] + "_correlation_counts"]
                    assert np.all(counts > 0) and len(sums) == len(
                        block["query_indices"]
                    )
                    correlation = float((sums / counts).mean())
                    assert (
                        abs(correlation - row["neighbor_content_correlation"]) < 1e-11
                    )
                    gap = errors["4"] - errors["1"]
                    assert abs(gap - row["excess_error"]) < 1e-11
                    maxabs = max(maxabs, row["dense_max_abs_change"])
                    maxrel = max(maxrel, row["dense_max_relative_change"])
                    per_fraction[row["reassignment_fraction"]].append(
                        {
                            "correlation": correlation,
                            "excess_error": gap,
                            "dispersed_error": errors["1"],
                            "clustered_error": errors["4"],
                        }
                    )
                assert [len(per_fraction[f]) for f in FRACTIONS] == [1, 2, 2, 2, 2]
                native_max = max(native_max, block["native_relative_error"])
                if block["cpu_reconstruction_max_abs_error"] is not None:
                    cpu_checks.append(block["cpu_reconstruction_max_abs_error"])
                curve = [
                    {
                        "reassignment_fraction": f,
                        **{
                            key: float(np.mean([r[key] for r in per_fraction[f]]))
                            for key in (
                                "correlation",
                                "excess_error",
                                "dispersed_error",
                                "clustered_error",
                            )
                        },
                    }
                    for f in FRACTIONS
                ]
                blocks.append({"block": block["block"], "interventions": curve})
        combined = [
            {
                "reassignment_fraction": f,
                **{
                    key: float(np.mean([b["interventions"][j][key] for b in blocks]))
                    for key in (
                        "correlation",
                        "excess_error",
                        "dispersed_error",
                        "clustered_error",
                    )
                },
            }
            for j, f in enumerate(FRACTIONS)
        ]
        scenes.append(
            {
                "scenes": scene["source_group"],
                "id": scene["id"],
                "interventions": combined,
                "per_block": blocks,
            }
        )
    assert len(cpu_checks) == 1
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, count, size=(10000, count))
    summary, block_summary = ([], [])
    for j, fraction in enumerate(FRACTIONS):
        values = {
            key: np.array([s["interventions"][j][key] for s in scenes])
            for key in (
                "correlation",
                "excess_error",
                "dispersed_error",
                "clustered_error",
            )
        }
        summary.append(
            {
                "reassignment_fraction": fraction,
                **{k: float(a.mean()) for k, a in values.items()},
                "correlation_interval": ci(values["correlation"], draws),
                "excess_error_interval": ci(values["excess_error"], draws),
            }
        )
    for block_index, label in enumerate(expected_blocks):
        curve = []
        for j, fraction in enumerate(FRACTIONS):
            values = {
                key: np.array(
                    [
                        s["per_block"][block_index]["interventions"][j][key]
                        for s in scenes
                    ]
                )
                for key in (
                    "correlation",
                    "excess_error",
                    "dispersed_error",
                    "clustered_error",
                )
            }
            curve.append(
                {
                    "reassignment_fraction": fraction,
                    **{k: float(a.mean()) for k, a in values.items()},
                    "correlation_interval": ci(values["correlation"], draws),
                    "excess_error_interval": ci(values["excess_error"], draws),
                }
            )
        block_summary.append({"block": label, "interventions": curve})
    initial = np.array([s["interventions"][0]["excess_error"] for s in scenes])
    final = np.array([s["interventions"][-1]["excess_error"] for s in scenes])
    changes = initial - final
    corr_changes = np.array(
        [
            s["interventions"][0]["correlation"] - s["interventions"][-1]["correlation"]
            for s in scenes
        ]
    )
    shrink = 100 * (1 - final.mean() / initial.mean()) if initial.mean() > 0 else None
    return {
        "settings": raw["settings"],
        "correlation_gradient": {
            "scenes": scenes,
            "pooled": summary,
            "per_block": block_summary,
            "source_groups": count,
            "measured_blocks": count * len(expected_blocks),
            "initial_excess_error": float(initial.mean()),
            "final_excess_error": float(final.mean()),
            "excess_error_reduction": float(changes.mean()),
            "excess_error_reduction_interval": ci(changes, draws),
            "relative_excess_error_reduction_percent": (
                float(shrink) if shrink is not None else None
            ),
            "inputs_with_smaller_excess_error": int((changes > 0).sum()),
            "inputs_with_initial_dispersed_advantage": int((initial > 0).sum()),
            "correlation_reduction_interval": ci(corr_changes, draws),
            "monotone_mean_correlation": bool(
                np.all(np.diff([r["correlation"] for r in summary]) <= 1e-12)
            ),
            "monotone_mean_excess_error": bool(
                np.all(np.diff([r["excess_error"] for r in summary]) <= 1e-12)
            ),
            "dense_max_abs_change": maxabs,
            "dense_max_relative_change": maxrel,
            "native_max_relative_error": native_max,
            "cpu_reconstruction_max_abs_error": max(cpu_checks),
            "recomputed_aggregations": checked_outputs,
        },
    }

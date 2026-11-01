"""Recompute spatial representation bounds from frozen QKV and real layouts."""

import argparse
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.analysis.framework import (
    attention_diagnostics,
    layout_geometry,
    prepare_attention,
)
from representative_context.runtime import read_config, write_json


def analyze(directory, manifest, rows, output, check_saved=False):
    result = dict(
        case=manifest["case"],
        complete_windows=len(rows) == len(manifest["windows"]),
        windows=[],
        max_identity_error=0.0,
        max_error_bound_ratio=0.0,
        head_query_comparisons=0,
    )
    output.mkdir(parents=True)
    for row in rows:
        records = []
        stored = {}
        with np.load(directory / row["qkv_arrays"], allow_pickle=False) as arrays:
            for block in row["blocks"]:
                stage = block["stage"]
                q, key, value = [
                    arrays[f"stage{stage}_{name}"] for name in ("Q", "K", "V")
                ]
                indices = arrays[f"stage{stage}_query_indices"]
                prepared = prepare_attention(q, key, value, 64)
                for capacity in manifest["capacities"]:
                    for pattern in manifest["layouts"]:
                        values = attention_diagnostics(
                            q,
                            key,
                            value,
                            indices,
                            block["query_grid"][0],
                            64,
                            capacity,
                            pattern,
                            block["seed"],
                            prepared,
                        )
                        prefix = f"stage{stage}_{pattern}_{capacity}"
                        stored.update(
                            {prefix + "_" + name: val for name, val in values.items()}
                        )
                        error = values["aggregation_l2_error"]
                        dense = values["dense_aggregation_l2"]
                        ratios = np.divide(
                            error,
                            values["aggregation_error_bound"],
                            out=np.zeros_like(error),
                            where=values["aggregation_error_bound"] > 0,
                        )
                        result["max_identity_error"] = max(
                            result["max_identity_error"],
                            values["identity_max_abs_error"],
                        )
                        result["max_error_bound_ratio"] = max(
                            result["max_error_bound_ratio"], float(ratios.max())
                        )
                        result["head_query_comparisons"] += error.size
                        records.append(
                            dict(
                                stage=stage,
                                capacity=capacity,
                                layout=pattern,
                                aggregation_error_squared_sum=float(
                                    np.square(error).sum()
                                ),
                                dense_aggregation_squared_sum=float(
                                    np.square(dense).sum()
                                ),
                                relative_l2=float(
                                    np.linalg.norm(error) / np.linalg.norm(dense)
                                ),
                                min_normalization=float(values["D_S"].min()),
                                median_normalization=float(np.median(values["D_S"])),
                                max_normalization=float(values["D_S"].max()),
                                max_error_bound_ratio=float(ratios.max()),
                            )
                        )
                full = attention_diagnostics(
                    q,
                    key,
                    value,
                    indices,
                    block["query_grid"][0],
                    64,
                    4096,
                    "point32",
                    block["seed"],
                    prepared,
                )
                assert full["aggregation_l2_error"].max() < 1e-10
                assert np.max(np.abs(full["D_S"] - 1)) < 1e-12
        if check_saved:
            with np.load(
                directory / row["diagnostic_arrays"], allow_pickle=False
            ) as expected:
                assert set(expected.files) == set(stored)
                for name, val in stored.items():
                    assert np.allclose(
                        val, expected[name], rtol=1e-12, atol=1e-12
                    ), name
        filename = f"diagnostics_{row['index']:03d}.npz"
        np.savez_compressed(output / filename, **stored)
        result["windows"].append(
            dict(
                index=row["index"], scene=row["scene"], records=records, arrays=filename
            )
        )
        print(f"Analyzed window {row['index']}", flush=True)
    result["geometry"] = [
        dict(
            capacity=capacity,
            layout=pattern,
            transport_distances=layout_geometry(64, capacity, pattern, 42)[
                "transport_distances"
            ].tolist(),
            analytic_upper_bound=(
                float(np.sqrt(2.5 / capacity)) if pattern == "point32" else None
            ),
            strip_distance=(1 - capacity / 4096) / 2 if pattern == "strip" else None,
        )
        for capacity in (*manifest["capacities"], 4096)
        for pattern in manifest["layouts"]
    ]
    write_json(output / "statistics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-saved", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    if args.limit is not None and args.limit < 1:
        parser.error("Select at least one window.")
    directory = args.input_dir or ROOT / f"results/03_theory/spatial/{args.case}"
    manifest = read_config(directory / "measurements.json")
    rows = (
        manifest["windows"] if args.limit is None else manifest["windows"][: args.limit]
    )
    analyze(directory, manifest, rows, args.output_dir, args.check_saved)


if __name__ == "__main__":
    main()

"""Recompute recurrent context errors and signed flow-restoration statistics."""

import argparse
from pathlib import Path
import numpy as np
from _bootstrap import ROOT
from representative_context.runtime import read_config, write_json


def task(current, reference, target):
    valid = np.isfinite(target).all(-1) & (np.abs(target) < 1e9).all(-1)
    cu, cv = current[..., 0].astype(float), current[..., 1].astype(float)
    ru, rv = reference[..., 0].astype(float), reference[..., 1].astype(float)
    gu, gv = target[..., 0].astype(float), target[..., 1].astype(float)
    error, baseline, change = (
        np.hypot(cu - gu, cv - gv)[valid],
        np.hypot(ru - gu, rv - gv)[valid],
        np.hypot(cu - ru, cv - rv)[valid],
    )
    return dict(
        valid_pixels=int(valid.sum()),
        GT_EPE=float(error.mean()),
        epe_sum=float(error.sum()),
        flow_change_sum=float(change.sum()),
        mean_flow_change=float(change.mean()),
        flow_change_squared_sum=float((change**2).sum()),
        rms_flow_change=float((change**2).mean() ** 0.5),
        improved_pixels=int((error < baseline).sum()),
        degraded_pixels=int((error > baseline).sum()),
    )


def combine(rows):
    pixels = sum(row["valid_pixels"] for row in rows)
    epe = sum(row["epe_sum"] for row in rows)
    change = sum(row["flow_change_sum"] for row in rows)
    energy = sum(row["flow_change_squared_sum"] for row in rows)
    return dict(
        valid_pixels=pixels,
        GT_EPE=epe / pixels,
        epe_sum=epe,
        flow_change_sum=change,
        mean_flow_change=change / pixels,
        flow_change_squared_sum=energy,
        rms_flow_change=(energy / pixels) ** 0.5,
        improved_pixels=sum(row["improved_pixels"] for row in rows),
        degraded_pixels=sum(row["degraded_pixels"] for row in rows),
    )


def verify(actual, expected):
    for key, value in actual.items():
        if isinstance(value, int):
            assert value == expected[key], key
        else:
            assert np.isclose(value, expected[key], rtol=1e-12, atol=1e-10), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results/03_theory/regression/gma_sintel",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_config(args.input_dir / "measurements.json")
    rows, comparisons, dense_rows = [], [], []
    pooled = {
        f"{pattern}_{arm}": []
        for pattern in report["layouts"]
        for arm in report["interventions"]
    }
    for pair in report["pairs"]:
        with (
            np.load(args.input_dir / pair["arrays"], allow_pickle=False) as flows,
            np.load(
                args.input_dir / pair["context_arrays"], allow_pickle=False
            ) as contexts,
        ):
            dense = task(
                flows["dense_prediction"], flows["dense_prediction"], flows["target"]
            )
            verify(dense, pair["dense_task"])
            dense_rows.append(dense)
            for pattern in report["layouts"]:
                for arm in report["interventions"]:
                    values = task(
                        flows[f"{pattern}_{arm}"],
                        flows["dense_prediction"],
                        flows["target"],
                    )
                    verify(
                        values, pair["layouts"][pattern]["interventions"][arm]["task"]
                    )
                    rows.append(
                        dict(
                            scene=pair["scene"],
                            layout=pattern,
                            intervention=arm,
                            **values,
                        )
                    )
                    pooled[f"{pattern}_{arm}"].append(values)
                for step in report["diagnostic_updates"]:
                    values = {}
                    for component in (
                        "motion_pixels",
                        "explicit_motion",
                        "learned_matching",
                        "full",
                    ):
                        rms = {}
                        for candidate in report["layouts"]:
                            error = contexts[
                                f"{pattern}_update{step}_{candidate}_{component}_squared_error"
                            ]
                            recorded = pair["layouts"][pattern]["comparisons"][
                                str(step)
                            ][candidate]
                            dimension = recorded[component + "_dimension"]
                            value = float(
                                np.sqrt(
                                    error.sum(dtype=float) / (error.size * dimension)
                                )
                            )
                            assert np.isclose(
                                value,
                                recorded[component + "_rms"],
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            rms[candidate] = value
                        values[component] = dict(
                            rms,
                            relative_reduction_percent=100
                            * (1 - rms["dispersed32"] / rms["clustered16"]),
                        )
                    comparisons.append(
                        dict(
                            scene=pair["scene"],
                            upstream=pattern,
                            update=step,
                            components=values,
                        )
                    )
    write_json(
        args.output,
        dict(
            case="gma_sintel",
            pairs=len(report["pairs"]),
            dense=combine(dense_rows),
            tasks=rows,
            comparisons=comparisons,
            pooled={name: combine(values) for name, values in pooled.items()},
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()

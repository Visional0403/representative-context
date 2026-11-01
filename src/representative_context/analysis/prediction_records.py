"""Pair inference reports and pool task metrics at their declared statistical units."""

import math
import statistics


HIGHER_IS_BETTER = {"polyp_kvasir", "hitnet", "tcd_tiled", "changeformer", "iml",
                    "trufor", "div2k", "gopro", "cave", "qb"}


def primary_metric(case, metrics):
    """Recompute the primary score from per-image counts or sufficient statistics."""
    if case in ("tcd_tiled", "changeformer"):
        tp, fp, fn = [sum(row["counts"][name] for row in metrics) for name in ("TP", "FP", "FN")]
        return (100 * tp / max(1, tp + fp + fn) if case == "tcd_tiled" else
                200 * tp / max(1, 2 * tp + fp + fn))
    if case == "gma_sintel":
        return sum(row["epe_sum"] for row in metrics) / sum(row["pixels"] for row in metrics)
    if case == "gamus":
        return math.sqrt(sum(row["squared_error_sum"] for row in metrics) /
                         sum(row["valid_pixels"] for row in metrics))
    if case == "pi3":
        return statistics.mean(frame["point_relative_error"] for row in metrics for frame in row["frames"])
    key, scale = {
        "polyp_kvasir": ("Dice", 100), "hitnet": ("Dice", 100),
        "iml": ("F1", 100), "trufor": ("F1", 100), "glpn_nyu": ("abs_rel", 1),
        "div2k": ("psnr_db", 1), "gopro": ("psnr_db", 1), "cave": ("psnr_db", 1),
        "qb": ("psnr_db", 1), "geco2": ("absolute_error", 1),
    }[case]
    return scale * statistics.mean(row[key] for row in metrics)


def input_identity(report):
    inputs = report["inputs"]
    return dict(weights=inputs["weights"], dataset=inputs["data_root"],
                sample_manifest=inputs.get("sample_manifest"),
                source=inputs.get("source_specification"),
                configuration={key: value for key, value in inputs["configuration"].items()
                               if key not in {"mode", "k", "benchmark", "limit", "sample_index", "timing"}})


def pair_reports(dense, sparse):
    if dense["mode"] != "dense" or sparse["mode"] != "sparse":
        raise ValueError("Provide one dense and one sparse evaluation report.")
    if dense["case"] != sparse["case"] or input_identity(dense) != input_identity(sparse):
        raise ValueError("Paired reports must share model, data, split and evaluation settings.")
    a, b = dense["samples"], sparse["samples"]
    if [str(row["id"]) for row in a] != [str(row["id"]) for row in b]:
        raise ValueError("Paired reports must have the same ordered sample identities.")
    for report in (dense, sparse):
        if (report.get("task") != "evaluation" or not report.get("complete_split") or
                len(report["samples"]) != report["evaluation_units"]):
            raise ValueError("Curve collection requires complete evaluation reports.")
    return dict(schema="paired_prediction_v1", case=dense["case"],
                inputs=dense["inputs"], resource_exponents=[sparse["k"]],
                records=[dict(id=str(x["id"]), dense=x["metrics"],
                              **{f"k{sparse['k']}": y["metrics"]}) for x, y in zip(a, b)])


def application_records(dense, sparse, profile):
    if sparse["k"] != profile["k"]:
        raise ValueError("The application profile specifies a different capacity exponent.")
    pair_reports(dense, sparse)
    values = []
    for report in (dense, sparse):
        rows = [row for sample in report["samples"] for row in sample.get("application_readouts", [])]
        mapping = {str(row["id"]): row for row in rows}
        expected = [row["input_id"] for row in profile["samples"]]
        if len(mapping) != len(rows) or set(mapping) != set(expected):
            raise ValueError("Application readouts must cover every declared input exactly once.")
        values.append(mapping)
    samples = []
    for identity in profile["samples"]:
        a, b = [mapping[identity["input_id"]] for mapping in values]
        if a["target"] != b["target"] or a["valid_pixels"] != b["valid_pixels"]:
            raise ValueError("Paired readouts must use identical reference values and masks.")
        samples.append(dict(id=identity["id"], gt_readout=a["target"],
                            dense_readout=a["prediction"], sparse_readout=b["prediction"]))
    return dict(name=profile["name"], unit=profile["unit"], k=profile["k"],
                sampling_unit=profile["sampling_unit"], samples=samples,
                statistics=dict(seed=profile["seed"], bootstrap_repeats=profile["bootstrap_repeats"],
                    sample_groups=[dict(id=row["id"], group=row["group"]) for row in profile["samples"]]))

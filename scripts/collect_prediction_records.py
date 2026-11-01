"""Collect new inference reports for the task-curve and application analyses."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.analysis.prediction_records import (
    HIGHER_IS_BETTER, application_records, input_identity, pair_reports, primary_metric,
)
from representative_context.runtime import read_config, write_json
from representative_context.analysis.timing_records import validate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, nargs="+", required=True)
    parser.add_argument("--timing", type=Path, nargs="+", required=True)
    parser.add_argument("--applications", type=Path, default=ROOT / "configs/applications/readouts.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    dense = read_config(args.dense)
    case = dense["case"]
    curve_case = "geco" if case == "geco2" else case
    folder = args.output_dir / ("geco2" if case == "geco2" else case)
    timings = {}
    for path in args.timing:
        document = read_config(path)
        if (document.get("task") != "timing" or document.get("mode") != "paired" or
                document.get("case") != case or input_identity(document) != input_identity(dense)):
            raise ValueError("Timing must use the same model and inputs in paired mode.")
        if document["k"] in timings:
            raise ValueError("Provide one declared timing report per capacity exponent.")
        validate_report(document)
        timings[document["k"]] = document
    index, points, applications = [], [], {}
    profiles = read_config(args.applications)
    seen = set()
    for path in args.sparse:
        sparse = read_config(path)
        k = sparse["k"]
        if k in seen:
            raise ValueError("Provide one sparse evaluation report per capacity exponent.")
        seen.add(k)
        paired = pair_reports(dense, sparse)
        timing = timings[k]
        score_d = primary_metric(case, [row["metrics"] for row in dense["samples"]])
        score_s = primary_metric(case, [row["metrics"] for row in sparse["samples"]])
        for name, document, units in ((f"accuracy_k{k}.json", paired, len(paired["records"])),
                                      (f"timing_k{k}.json", timing, None)):
            write_json(folder / name, document)
            index.append(dict(file=name, resource_exponents=[k], accuracy_units=units))
        points.append(dict(k=k, dense_metric=score_d, sparse_metric=score_s,
            relative_metric_loss_percent=100 * (score_d - score_s if case in HIGHER_IS_BETTER else score_s - score_d) / abs(score_d),
            dense_median_ms=timing["latency_ms"]["dense"], sparse_median_ms=timing["latency_ms"]["sparse"],
            speedup=timing["speedup"], round_speedups=(timing.get("paired_round_speedups")
                if timing["speedup_statistic"] == "median_paired_round_ratios" else None)))
        if case in profiles and k == profiles[case]["k"]:
            applications[case] = application_records(dense, sparse, profiles[case])
    write_json(folder / "index.json", dict(case=curve_case, files=index))
    write_json(args.output_dir / "budget_curves.json", {curve_case: dict(points=sorted(points, key=lambda p:p["k"]))})
    if applications:
        write_json(args.output_dir / "application_readouts.json", applications)
    print(args.output_dir)


if __name__ == "__main__":
    main()

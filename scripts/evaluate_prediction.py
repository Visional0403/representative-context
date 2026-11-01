"""Evaluate a pretrained visual prediction model or time it independently."""

import argparse
from pathlib import Path

import torch

from _bootstrap import ROOT
from representative_context.runtime import (
    environment,
    file_info,
    input_shapes,
    prediction_inputs,
    read_config,
    resource_path,
    set_seed,
    write_json,
)
from representative_context.tasks.registry import create_task, load_task_model
from representative_context.tasks.timing import measure_task
from representative_context.tasks.readouts import collect_readouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("dense", "sparse", "paired"), required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("A restricted evaluation requires a positive sample count.")
    if args.output.exists():
        parser.error("Choose an output file that does not already exist.")
    if args.mode == "paired" and not args.benchmark:
        parser.error(
            "Paired execution is available for independent timing; evaluate each mode separately."
        )
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    cfg = read_config(args.config)
    set_seed(cfg["seed"], cfg["threads"])
    torch.backends.cudnn.allow_tf32 = cfg["execution"]["cudnn_allow_tf32"]
    torch.backends.cudnn.deterministic = cfg["execution"]["cudnn_deterministic"]
    data_root = args.data_root or resource_path(cfg["data_root"], ROOT)
    weights = args.weights or resource_path(cfg["weights"], ROOT)
    source = args.source or (
        resource_path(cfg["source"], ROOT) if cfg.get("source") else None
    )
    data_root, weights = data_root.resolve(), weights.resolve()
    source = source.resolve() if source is not None else None
    samples = resource_path(cfg["samples"], ROOT) if cfg.get("samples") else None
    task = create_task(cfg["case"], data_root, source, samples=samples,
                       timing_sample_id=cfg.get("timing_sample_id"))
    if task.count != cfg["expected_samples"]:
        raise ValueError(
            f"Expected {cfg['expected_samples']} evaluation units, found {task.count}."
        )
    model, controls = load_task_model(
        cfg["case"],
        weights,
        source,
        k=args.k,
        dense=args.mode == "dense",
        dense_backend=cfg.get("dense_backend", "sdpa"),
    )
    report = {
        "case": cfg["case"],
        "mode": args.mode,
        "k": args.k,
        "precision": cfg.get("precision", "FP32"),
        "batch_size": 1,
        "seed": cfg["seed"],
        "evaluation_units": task.count,
        "attention_modules": len(controls),
        "environment": environment(),
    }
    resolved = dict(cfg)
    resolved.update(mode=args.mode, k=args.k, limit=args.limit,
                    benchmark=args.benchmark, sample_index=args.sample_index)
    resolved["timing"] = {
        key: getattr(args, key) or cfg["timing"][key]
        for key in ("warmup", "repeats", "rounds")
    }
    report["inputs"] = prediction_inputs(weights, data_root, samples, source, resolved)
    report["inputs"]["configuration_file"] = file_info(args.config)
    with torch.inference_mode():
        if args.benchmark:
            settings = {
                key: getattr(args, key) or cfg["timing"][key]
                for key in ("warmup", "repeats", "rounds")
            }
            representatives = (
                task.timing_representatives()
                if args.sample_index is None
                else [
                    (
                        next(
                            row
                            for index, row in enumerate(task.iter_records())
                            if index == args.sample_index
                        ),
                        1.0,
                    )
                ]
            )
            strata = []
            for record, weight in representatives:
                inputs, _ = task.prepare(record)

                def forward(dense):
                    for control in controls:
                        control.dense = dense or getattr(control, "force_dense", False)
                    return task.forward(model, inputs)

                def before_method(name):
                    if cfg.get("release_timing_cache", False):
                        for control in controls:
                            control.cache.clear()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                methods = {
                    name: (lambda dense=name == "dense": forward(dense))
                    for name in (
                        ("dense", "sparse") if args.mode == "paired" else (args.mode,)
                    )
                }
                strata.append(
                    {
                        "sample_id": record["id"],
                        "weight": weight,
                        "input_tensors": input_shapes(inputs),
                        "timings": measure_task(
                            methods,
                            **settings,
                            timer=cfg["timer"],
                            before_method=before_method,
                        ),
                    }
                )
                del inputs
            import statistics

            modes = strata[0]["timings"]
            weighted = {
                name: [
                    sum(
                        part["weight"] * part["timings"][name]["round_median_ms"][index]
                        for part in strata
                    )
                    for index in range(settings["rounds"])
                ]
                for name in modes
            }
            from representative_context.tasks.timing import summarize_strata

            report.update(
                task="timing",
                scope=cfg["timing_scope"],
                settings=settings,
                timer=cfg["timer"],
                synchronization=("before_each_method_and_after_event_batch"
                    if cfg["timer"] == "cuda_event" else "before_and_after_each_call"),
                strata=strata,
                weighted_round_median_ms=weighted,
                weighted_median_ms={
                    name: statistics.median(values) for name, values in weighted.items()
                },
            )
            report.update(
                summarize_strata(strata, cfg["timing_summary"], cfg["speedup_summary"])
            )
        else:
            rows = []
            for record in task.iter_records(args.limit):
                inputs, target = task.prepare(record)
                output = task.forward(model, inputs)
                metrics = task.score(output, target, inputs)
                rows.append(dict(id=record["id"], metrics=metrics,
                    application_readouts=collect_readouts(
                        cfg["case"], record, output, target, metrics)))
                if len(rows) % 10 == 0:
                    print(f"Evaluated {len(rows)} samples", flush=True)
                del inputs, target, output
                if cfg.get("release_evaluation_cache", False):
                    torch.cuda.empty_cache()
            report.update(
                task="evaluation",
                evaluated_samples=len(rows),
                complete_split=len(rows) == task.count,
                metrics=task.aggregate(rows),
                samples=rows,
            )
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()

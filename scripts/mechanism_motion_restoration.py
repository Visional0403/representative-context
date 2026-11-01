"""Reconstruct motion-context restoration on four declared Sintel image pairs."""

import argparse
from pathlib import Path
import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.adapters.motion import MotionAttention
from representative_context.analysis.cross_task_models import decode_flow, decode_image
from representative_context.analysis.motion_context import (
    ARMS,
    PATTERNS,
    NativeCoverAttention,
    capture_forward,
    check_math,
    comparisons,
    measures,
    numpy,
    replay,
)
from representative_context.runtime import (
    environment,
    read_config,
    resource_path,
    write_json,
)
from representative_context.tasks.registry import create_task, load_task_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    if args.limit is not None and not 1 <= args.limit <= 4:
        parser.error("Select one to four declared image pairs.")
    cfg = read_config(ROOT / "configs/gma_sintel/validation.json")
    reference = read_config(
        ROOT / "results/03_theory/regression/gma_sintel/measurements.json"
    )
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    source = args.source or resource_path(cfg["source"], ROOT)
    model, controls = load_task_model(
        "gma_sintel",
        args.checkpoint or resource_path(cfg["weights"], ROOT),
        source,
        k=14,
        dense=True,
    )
    original_attention = controls[0].original
    original_aggregate = model.update_block.aggregator.original
    model.att = original_attention
    model.update_block.aggregator = original_aggregate
    model.args.mixed_precision = False
    checks = check_math()
    task = create_task(
        "gma_sintel", args.data_root or resource_path(cfg["data_root"], ROOT), source
    )
    scenes = sorted({row["scene"] for row in task.rows})
    assert len(scenes) == 23
    assert [scenes[index] for index in (0, 7, 14, 22)] == [
        row["scene"] for row in reference["pairs"]
    ]
    declared = (
        reference["pairs"] if args.limit is None else reference["pairs"][: args.limit]
    )
    probes = {
        "dispersed32": MotionAttention(original_attention, "sparse", 2048, 42).eval(),
        "clustered16": NativeCoverAttention(
            original_attention, "sparse", 2048, 42
        ).eval(),
    }
    args.output_dir.mkdir(parents=True)
    rows = []
    from utils.utils import InputPadder

    with torch.inference_mode():
        for expected in declared:
            candidates = [row for row in task.rows if row["scene"] == expected["scene"]]
            pair = candidates[len(candidates) // 2]
            assert Path(pair["first"]).name == expected["first_frame"]
            first = decode_image(task.images.read(pair["first"]), torch.device("cuda"))
            second = decode_image(
                task.images.read(pair["second"]), torch.device("cuda")
            )
            target = decode_flow(task.extras.read(pair["flow"])).copy()
            assert tuple(first.shape[-2:]) == (436, 1024) and target.shape == (
                436,
                1024,
                2,
            )
            padder = InputPadder(first.shape)
            first, second = padder.pad(first, second)
            dense, _, dense_prediction = capture_forward(
                model,
                original_attention,
                original_aggregate,
                first,
                second,
                padder,
                "dense",
                probes,
            )
            attention, context = dense["attention"], dense["qk_input"]
            flows = dict(target=target, dense_prediction=dense_prediction)
            contexts = dict(qk_input=numpy(context))
            maps = {name: probe(context) for name, probe in probes.items()}
            assert tuple(context.shape[-2:]) == (55, 128)
            assert all(
                item.groups == 4 and item.weights.shape[-1] == 1760
                for item in maps.values()
            )
            info = {}
            del dense
            for pattern in PATTERNS:
                captured, original_low, current = capture_forward(
                    model,
                    original_attention,
                    original_aggregate,
                    first,
                    second,
                    padder,
                    pattern,
                    probes,
                )
                assert torch.equal(captured["qk_input"], context)
                assert torch.equal(captured["attention"].weights, maps[pattern].weights)
                details = dict(comparisons={}, interventions={})
                for step in (6, 12):
                    values, per_query = comparisons(
                        original_aggregate,
                        attention,
                        maps,
                        captured["motion_inputs"][step],
                    )
                    details["comparisons"][str(step)] = values
                    contexts[f"{pattern}_update{step}_motion_input"] = numpy(
                        captured["motion_inputs"][step]
                    )
                    contexts[f"{pattern}_update{step}_aggregation"] = numpy(
                        captured["aggregation_outputs"][step]
                    )
                    contexts.update(
                        {
                            f"{pattern}_update{step}_{name}": value
                            for name, value in per_query.items()
                        }
                    )
                contexts[f"{pattern}_update11_net"] = numpy(
                    captured["update11_input"]["net"]
                )
                contexts[f"{pattern}_update11_coordinates"] = numpy(
                    captured["update11_input"]["coordinates"]
                )
                for arm in ARMS:
                    low, predicted, records = replay(
                        model,
                        original_aggregate,
                        attention,
                        captured,
                        padder,
                        first,
                        arm,
                    )
                    if arm == "sparse":
                        details["replay_max_abs_error"] = dict(
                            low_resolution=float((low - original_low).abs().max()),
                            prediction=float(np.abs(predicted - current).max()),
                        )
                        assert max(details["replay_max_abs_error"].values()) <= 1e-5
                        predicted = current
                    flows[f"{pattern}_{arm}"] = predicted
                    details["interventions"][arm] = dict(
                        task=measures(predicted, dense_prediction, target),
                        local_restoration=records,
                    )
                info[pattern] = details
                del captured, original_low
            scene = pair["scene"]
            flow_file = f"{scene}.npz"
            context_file = f"contexts_{scene}.npz"
            np.savez_compressed(args.output_dir / flow_file, **flows)
            np.savez_compressed(args.output_dir / context_file, **contexts)
            rows.append(
                dict(
                    expected,
                    arrays=flow_file,
                    context_arrays=context_file,
                    first=pair["first"],
                    second=pair["second"],
                    flow=pair["flow"],
                    dense_task=measures(dense_prediction, dense_prediction, target),
                    layouts=info,
                )
            )
            print(f"{len(rows)}/{len(declared)} {scene}", flush=True)
            del maps, attention, context, first, second, flows, contexts
    write_json(
        args.output_dir / "measurements.json",
        dict(
            reference,
            pairs=rows,
            complete_declared_pairs=len(rows) == 4,
            seed=42,
            k=14,
            total_updates=12,
            restored_updates=[11, 12],
            diagnostic_updates=[6, 12],
            capacity_limit=2048,
            actual_capacity=1760,
            groups=4,
            algebra_checks=checks,
            environment=environment(),
        ),
    )


if __name__ == "__main__":
    main()

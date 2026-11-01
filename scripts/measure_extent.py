"""Reproduce fixed-capacity spatial-support geometry, accuracy and diagnostics."""

import argparse
from pathlib import Path
import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.analysis.extent import geometry, probe
from representative_context.analysis.segmentation import SegmentationAdapter
from representative_context.data.uavid import read_pair
from representative_context.evaluation.mmseg import read_label, metrics as mmseg_metrics
from representative_context.evaluation.segmentation import (
    predict_image,
    segmentation_metrics,
)
from representative_context.runtime import (
    environment,
    read_config,
    set_seed,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_cityscapes"), required=True
    )
    parser.add_argument(
        "--phase", choices=("geometry", "evaluate", "probe", "speed"), required=True
    )
    parser.add_argument("--pattern", default="extent0")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.limit is not None and args.limit < 1:
        parser.error("Select at least one input.")
    metadata = read_config(
        ROOT / f"results/01_context/extent/{args.case}/measurements.json"
    )
    if args.pattern not in metadata["patterns"] + (
        ["dense"] if args.case == "twins_cityscapes" else []
    ):
        parser.error("Choose a declared layout.")
    if args.phase == "geometry":
        report = geometry(args.case)
    else:
        set_seed(42)
        torch.backends.cudnn.allow_tf32 = True
        if args.phase == "speed":
            from representative_context.analysis.routing_models import load
            from representative_context.analysis.routing_pipeline import timing

            model, specs, _, _, window, call, _ = load(
                ROOT, args.case, args.data_root, args.checkpoint
            )
            for _, _, core, _, _ in specs:
                core.dense = args.pattern == "dense"
                core.pattern = args.pattern
            value = window(0)
            with torch.inference_mode():
                report = timing(lambda: call(value), 5, 20)
        else:
            adapter = SegmentationAdapter(
                args.case, ROOT, args.data_root, args.checkpoint
            )
            with torch.inference_mode():
                if args.phase == "probe":
                    report = probe(adapter, metadata, args.limit)
                else:
                    adapter.mode(None if args.pattern == "dense" else args.pattern)
                    selected = (
                        adapter.pairs
                        if args.limit is None
                        else adapter.pairs[: args.limit]
                    )
                    total = np.zeros((adapter.classes, adapter.classes), dtype=np.int64)
                    rows = []
                    for image, label in selected:
                        if adapter.native:
                            value, target = read_pair(image, label)
                            output = predict_image(
                                adapter.model,
                                value[None].cuda(),
                                window_size=2048,
                                stride=1024,
                            )
                            predicted = output.argmax(1)[0].cpu().numpy()
                            target = target.numpy()
                            del output, value
                        else:
                            from mmseg.apis import inference_model

                            predicted = (
                                inference_model(adapter.model, str(image))
                                .pred_sem_seg.data.squeeze()
                                .cpu()
                                .numpy()
                            )
                            target = read_label(label, "Cityscapes")
                        valid = (target >= 0) & (target < adapter.classes)
                        hist = np.bincount(
                            target[valid].astype(np.int64) * adapter.classes
                            + predicted[valid],
                            minlength=adapter.classes**2,
                        ).reshape(adapter.classes, adapter.classes)
                        total += hist
                        rows.append(
                            dict(
                                image=image.relative_to(adapter.root).as_posix(),
                                confusion_matrix=hist.tolist(),
                            )
                        )
                        print(f"{len(rows)}/{len(selected)} {image.name}", flush=True)
                    report = dict(
                        complete_validation=len(rows) == len(adapter.pairs),
                        images=rows,
                        confusion_matrix=total.tolist(),
                        **(
                            segmentation_metrics(total)
                            if adapter.native
                            else mmseg_metrics(total)
                        ),
                    )
        report.update(environment=environment())
    report.update(
        case=args.case,
        phase=args.phase,
        pattern=args.pattern if args.phase in ("evaluate", "speed") else None,
        k=14,
    )
    write_json(args.output, report)


if __name__ == "__main__":
    main()

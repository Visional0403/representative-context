"""Evaluate clustered-to-dispersed controls with identical per-query contexts."""

import argparse
from pathlib import Path
import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.analysis.execution_models import load_model
from representative_context.data.uavid import evaluation_pairs, read_pair
from representative_context.evaluation.mmseg import image_pairs, read_label
from representative_context.evaluation.segmentation import predict_image
from representative_context.kernels.context import install
from representative_context.runtime import (
    environment,
    read_config,
    resource_path,
    set_seed,
    write_json,
)


def metrics(hist):
    tp = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0) * 100
    return dict(
        miou_percent=float(np.nanmean(iou)),
        per_class_iou_percent=[
            float(value) if np.isfinite(value) else None for value in iou
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cluster", type=int, choices=(4, 2, 1), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.limit is not None and args.limit < 1:
        parser.error("Select at least one image.")
    manifest = read_config(args.prepared / "manifest.json")
    assert manifest["all_queries_audited"]
    assert manifest["identical_local_anchors_and_quotas"]
    case = manifest["case"]
    native = case == "segformer_uavid"
    set_seed(42)
    torch.backends.cudnn.allow_tf32 = True
    model, blocks, config = load_model(ROOT, case, k=14, checkpoint=args.checkpoint)
    cores = {(stage - 1, block - 1): core for stage, block, core in blocks}
    for row in manifest["blocks"]:
        if row["cluster"] != args.cluster:
            continue
        core = cores[row["stage"], row["block"]]
        assert (
            core.num_heads == row["heads"]
            and core.in_proj_weight.shape[1] // core.num_heads == row["head_dimension"]
        )
        with np.load(args.prepared / row["arrays"], allow_pickle=False) as values:
            indices = values["indices"]
        assert indices.shape == (row["kv_side"] ** 2, row["capacity"])
        install(core, indices, row["kv_side"], row["query_side"])
    data_root = args.data_root or resource_path(config["data_root"], ROOT)
    pairs = (
        evaluation_pairs(data_root) if native else image_pairs(data_root, "Cityscapes")
    )
    assert len(pairs) == (70 if native else 500)
    selected = pairs if args.limit is None else pairs[: args.limit]
    classes = 8 if native else 19
    records = []
    total = np.zeros((classes, classes), dtype=np.int64)
    with torch.inference_mode():
        for image, label in selected:
            if native:
                inputs, target = read_pair(image, label)
                output = predict_image(
                    model, inputs[None].cuda(), window_size=2048, stride=1024
                )
                prediction = output.argmax(1)[0].cpu().numpy()
                target = target.numpy()
                del inputs, output
            else:
                from mmseg.apis import inference_model

                prediction = (
                    inference_model(model, str(image))
                    .pred_sem_seg.data.squeeze()
                    .cpu()
                    .numpy()
                )
                target = read_label(label, "Cityscapes")
            valid = (target >= 0) & (target < classes)
            hist = np.bincount(
                target[valid].astype(np.int64) * classes + prediction[valid],
                minlength=classes**2,
            ).reshape(classes, classes)
            total += hist
            records.append(
                dict(
                    image=image.relative_to(data_root).as_posix(),
                    confusion_matrix=hist.tolist(),
                    **metrics(hist),
                )
            )
            print(f"{len(records)}/{len(selected)} {records[-1]['image']}", flush=True)
    assert [core.call_count for core in cores.values()] == [
        (6 if native else 3) * len(selected)
    ] * len(cores)
    write_json(
        args.output,
        dict(
            case=case,
            k=14,
            cluster=args.cluster,
            complete_validation=len(selected) == len(pairs),
            evaluated_images=len(records),
            images=records,
            confusion_matrix=total.tolist(),
            **metrics(total),
            environment=environment(),
        ),
    )


if __name__ == "__main__":
    main()

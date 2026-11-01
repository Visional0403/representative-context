"""Evaluate matched-compute stage allocations on complete UAVid images or fixed windows."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from _bootstrap import ROOT
from representative_context.analysis.allocation import allocation, miou
from representative_context.analysis.segmentation import SegmentationAdapter
from representative_context.controls.allocation import set_stage_capacities
from representative_context.data.uavid import read_pair
from representative_context.evaluation.mmseg import read_label
from representative_context.evaluation.segmentation import gaussian_weight
from representative_context.runtime import environment, set_seed, write_json


def starts(length):
    values = list(range(0, length - 2048 + 1, 1024))
    if values[-1] != length - 2048:
        values.append(length - 2048)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_uavid"), required=True
    )
    parser.add_argument("--phase", choices=("full", "diagnostic"), default="full")
    parser.add_argument("--k", type=int, nargs="+", default=[14, 16], choices=(14, 16))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.limit is not None and args.limit < 1:
        parser.error("A restricted run requires at least one image.")
    set_seed(42)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = args.case == "segformer_uavid"
    adapter = SegmentationAdapter(args.case, ROOT, args.data_root, args.checkpoint)
    plans = [
        allocation(args.case, k, policy)
        for k in args.k
        for policy in ("structural", "uniform", "no_density")
    ]
    arms = [("dense", None)] + [
        (f"k{plan['k']}_{plan['policy']}", plan) for plan in plans
    ]
    declared = (
        adapter.indices
        if args.phase == "diagnostic"
        else list(range(len(adapter.pairs)))
    )
    indices = declared if args.limit is None else declared[: args.limit]
    report = dict(
        case=args.case,
        phase=args.phase,
        complete_scope=len(indices) == len(declared),
        declared_indices=declared,
        plans=plans,
        seed=42,
        window_size=2048,
        stride=1024,
        environment=environment(),
        images=[],
    )

    def forward(x):
        return (
            adapter.model(x)
            if adapter.native
            else adapter.model.encode_decode(x, [{"img_shape": x.shape[-2:]}])
        )

    with torch.inference_mode():
        for index in indices:
            if args.phase == "diagnostic":
                image, gt, item = adapter.window(index)
            else:
                path, label = adapter.pairs[index]
                gt = read_label(label, "UAVid")
                if adapter.native:
                    image, gt_tensor = read_pair(path, label)
                    image = image[None].cuda()
                    gt = gt_tensor.numpy().astype(np.uint8)
                else:
                    rgb = np.array(Image.open(path).convert("RGB"))
                    image = (
                        torch.from_numpy(rgb.transpose(2, 0, 1).copy())
                        .cuda()
                        .float()[None]
                    )
                    image = (
                        image - adapter.model.data_preprocessor.mean
                    ) / adapter.model.data_preprocessor.std
                item = dict(
                    index=index,
                    image=path.relative_to(adapter.root).as_posix(),
                    scene=path.parent.parent.name,
                )
            item["configurations"] = {}
            for name, plan in arms:
                set_stage_capacities(
                    adapter.specs, None if plan is None else plan["capacities"]
                )
                if args.phase == "diagnostic":
                    logits = forward(image)
                else:
                    height, width = gt.shape
                    logits = torch.zeros((1, 8, height, width), device="cuda")
                    weights = torch.zeros((1, 1, height, width), device="cuda")
                    weight = 1.0
                    if adapter.native:
                        weight = gaussian_weight(2048, "cuda")
                    for y in starts(height):
                        for x in starts(width):
                            patch = forward(image[..., y : y + 2048, x : x + 2048])
                            logits[..., y : y + 2048, x : x + 2048] += patch * weight
                            weights[..., y : y + 2048, x : x + 2048] += weight
                            del patch
                    logits /= weights
                    del weights
                prediction = logits.argmax(1)[0].cpu().numpy()
                valid = gt < 8
                counts = np.bincount(
                    gt[valid].astype(np.int64) * 8 + prediction[valid], minlength=64
                ).reshape(8, 8)
                item["configurations"][name] = dict(
                    confusion_matrix=counts.tolist(), miou_percent=float(miou(counts))
                )
                if plan is not None:
                    assert [
                        core.last_plan.groups for _, _, _, core in adapter.specs
                    ] == [plan["groups"][stage] for stage, _, _, _ in adapter.specs]
                del logits, prediction
            report["images"].append(item)
            print(f"{len(report['images'])}/{len(indices)} {item['image']}", flush=True)
            del image, gt
            torch.cuda.empty_cache()
    totals = {
        name: np.sum(
            [
                row["configurations"][name]["confusion_matrix"]
                for row in report["images"]
            ],
            axis=0,
        )
        for name, _ in arms
    }
    report["total_confusion"] = {name: value.tolist() for name, value in totals.items()}
    report["miou_percent"] = {
        name: float(miou(value)) for name, value in totals.items()
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()

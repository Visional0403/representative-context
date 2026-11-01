"""Collect paired native-image classification or optical-flow difficulty counts."""

import argparse
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from _bootstrap import ROOT
from representative_context.analysis.long_tail import (
    FLOW_RAW_KEYS,
    GAP_BOUNDARIES,
    flow_masks,
    gap_pair_counts,
    paired_flow,
)
from representative_context.runtime import (
    environment,
    read_config,
    resource_path,
    set_seed,
    write_json,
)


def reference_details(logits, human):
    dense = logits.argmax(1)[0]
    top = torch.topk(logits[0], 2, dim=0).values
    normalizer = torch.logsumexp(logits[0], dim=0)
    gap = torch.exp(top[0] - normalizer) - torch.exp(top[1] - normalizer)
    bins = (
        torch.bucketize(gap.contiguous(), gap.new_tensor(GAP_BOUNDARIES), right=True)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    competitor = torch.where(dense == 7, top[1], top[0])
    margin = (
        (torch.exp(logits[0, 7] - normalizer) - torch.exp(competitor - normalizer))[
            human
        ]
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return dense.to(torch.uint8).cpu().numpy(), bins, margin


def uavid(args, metadata):
    from representative_context.adapters.sra import install_sra
    from representative_context.data.uavid import COLORS, evaluation_pairs, read_rgb
    from representative_context.models.segformer import load_segformer

    cfg = read_config(ROOT / "configs/segformer_uavid/validation.json")
    data = args.data_root or resource_path(cfg["data_root"], ROOT)
    model = load_segformer(args.checkpoint or resource_path(cfg["checkpoint"], ROOT))
    cores = install_sra(model, k=14, pattern="dispersed32", dense=True)
    pairs = evaluation_pairs(data)
    assert len(pairs) == 70
    selected = pairs if args.limit is None else pairs[: args.limit]
    records = []
    cubes = {name: [] for name in ("k14", "k16")}
    samples = {
        name: []
        for name in (
            "dense_margin",
            "dense_prediction",
            "k14_prediction",
            "k16_prediction",
        )
    }
    offsets = [0]
    for index, (image, label) in enumerate(selected):
        rgb = read_rgb(image)
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
        value = (
            torch.from_numpy((rgb.astype(np.float32) / 255.0 - mean) / std)
            .permute(2, 0, 1)
            .float()
        )
        mask = read_rgb(label)
        labels = np.full(mask.shape[:2], 255, dtype=np.uint8)
        for category, color in enumerate(COLORS):
            labels[np.all(mask == color, axis=-1)] = category
        human = labels == 7
        value = value[None].cuda()
        for core in cores:
            core.dense = True
        logits = model(value)
        dense, bins, margin = reference_details(logits, torch.from_numpy(human).cuda())
        del logits
        torch.cuda.empty_cache()
        samples["dense_margin"].append(margin)
        samples["dense_prediction"].append(dense[human])
        for k in (14, 16):
            for core in cores:
                core.dense = False
                core.k = k
            logits = model(value)
            candidate = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
            del logits
            torch.cuda.empty_cache()
            cubes[f"k{k}"].append(gap_pair_counts(labels, dense, candidate, bins))
            samples[f"k{k}_prediction"].append(candidate[human])
        records.append(
            dict(
                index=index,
                image=image.relative_to(data).as_posix(),
                label=label.relative_to(data).as_posix(),
                scene=image.parent.parent.name,
                shape=list(labels.shape),
            )
        )
        offsets.append(offsets[-1] + len(margin))
        print(f"{len(records)}/{len(selected)} {image.name}", flush=True)
        del value
    np.savez_compressed(
        args.output_dir / "confusion.npz",
        **{key: np.asarray(values, dtype=np.int64) for key, values in cubes.items()},
    )
    np.savez_compressed(
        args.output_dir / "human_pixels.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        **{key: np.concatenate(values) for key, values in samples.items()},
    )
    return dict(
        metadata,
        images=records,
        complete_split=len(records) == 70,
        environment=environment(),
    )


def gma(args, metadata):
    from representative_context.tasks.registry import create_task, load_task_model
    from representative_context.tasks.motion import archive_index

    cfg = read_config(ROOT / "configs/gma_sintel/validation.json")
    source = args.source or resource_path(cfg["source"], ROOT)
    model, controls = load_task_model(
        "gma_sintel",
        args.checkpoint or resource_path(cfg["weights"], ROOT),
        source,
        k=14,
        dense=True,
    )
    control = controls[0]
    task = create_task(
        "gma_sintel", args.data_root or resource_path(cfg["data_root"], ROOT), source
    )
    assert len(task.rows) == 1041
    occlusions = archive_index(task.extras, "occlusions")
    selected = task.rows if args.limit is None else task.rows[: args.limit]
    records = []
    for index, row in enumerate(selected):
        inputs, target = task.prepare(row)
        control.dense = True
        dense = (
            target["padder"]
            .unpad(task.forward(model, inputs)[0])
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )
        control.dense = False
        sparse = (
            target["padder"]
            .unpad(task.forward(model, inputs)[0])
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )
        assert control.last_shape == (55, 128) and control.last_groups == 4
        mask_path = occlusions[row["scene"], Path(row["first"]).name]
        occlusion = np.asarray(Image.open(io.BytesIO(task.extras.read(mask_path)))) > 0
        truth = target["flow"]
        baseline, magnitude, masks, threshold = flow_masks(dense, truth, occlusion)
        current = np.linalg.norm(sparse.astype(float) - truth.astype(float), axis=-1)
        details = {
            name: {
                key: value
                for key, value in paired_flow(
                    baseline, current, magnitude, mask
                ).items()
                if key in FLOW_RAW_KEYS
            }
            for name, mask in masks.items()
        }
        record = dict(
            index=index,
            scene=row["scene"],
            frame=Path(row["first"]).name,
            dense_error_quantile80=threshold,
            groups=4,
            subsets=details,
        )
        if index in (0, 260, 520, 780, 1040):
            filename = f"pair_{index:04d}.npz"
            np.savez_compressed(
                args.output_dir / filename,
                target=truth,
                dense_prediction=dense,
                sparse_prediction=sparse,
                occlusion=occlusion,
            )
            record["arrays"] = filename
        records.append(record)
        print(f"{len(records)}/{len(selected)} {row['id']}", flush=True)
        del inputs, dense, sparse, target, truth
    return dict(
        metadata,
        pairs=records,
        complete_split=len(records) == 1041,
        environment=environment(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "gma_sintel"), required=True
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    if args.limit is not None and args.limit < 1:
        parser.error("Select at least one evaluation unit.")
    metadata = read_config(
        ROOT / f"results/02_mechanisms/long_tail/{args.case}/measurements.json"
    )
    if args.limit is not None and args.limit > metadata.get(
        "expected_images", metadata.get("expected_pairs")
    ):
        parser.error("The limit exceeds the evaluation split.")
    set_seed(metadata["seed"])
    torch.backends.cudnn.allow_tf32 = metadata["cudnn_allow_tf32"]
    torch.backends.cudnn.deterministic = metadata["cudnn_deterministic"]
    args.output_dir.mkdir(parents=True)
    with torch.inference_mode():
        report = (
            uavid(args, metadata)
            if args.case == "segformer_uavid"
            else gma(args, metadata)
        )
    write_json(args.output_dir / "measurements.json", report)


if __name__ == "__main__":
    main()

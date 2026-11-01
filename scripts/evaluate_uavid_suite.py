"""Evaluate the complete SegFormer/UAVid comparison suite serially."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/segformer_uavid/evaluation_suite.json")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--configuration")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/uavid_suite")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared or args.output_dir / "prepared_context"
    if args.all:
        if not prepared.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/prepare_context_controls.py"),
                            "--case", "segformer_uavid", "--output-dir", str(prepared)], check=True)
        for row in config["configurations"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                       "--configuration", row["id"], "--output-dir", str(args.output_dir), "--prepared", str(prepared)]
            for option, value in (("--data-root", args.data_root), ("--checkpoint", args.checkpoint)):
                if value is not None:
                    command += [option, str(value)]
            subprocess.run(command, check=True)
        return
    candidates = [row for row in config["configurations"] if row["id"] == args.configuration]
    if len(candidates) != 1:
        parser.error("Select a configuration ID from the suite configuration.")
    evaluate(config, candidates[0], args, prepared)


def evaluate(config, job, args, prepared):
    import numpy as np
    import torch

    from representative_context.adapters.sra import install_sra
    from representative_context.data.uavid import CLASS_NAMES, evaluation_pairs, read_pair
    from representative_context.evaluation.segmentation import confusion_matrix, predict_image, segmentation_metrics
    from representative_context.models.segformer import load_segformer
    from representative_context.runtime import (
        environment, evaluation_inputs, file_info, resource_path, set_seed, update_json,
    )

    protocol = config["protocol"]
    set_seed(protocol["seed"], protocol["threads"])
    torch.backends.cudnn.allow_tf32 = True
    data_root = (args.data_root or resource_path(config["data_root"], ROOT)).resolve()
    checkpoint = (args.checkpoint or resource_path(config["checkpoint"], ROOT)).resolve()
    pairs = evaluation_pairs(data_root)
    assert len(pairs) == config["expected_images"]
    configuration = {key: value for key, value in job.items() if key != "result"}
    inputs = evaluation_inputs(checkpoint, data_root, pairs)
    if "cluster" in job:
        layout = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        inputs["prepared"] = [file_info(prepared / "manifest.json")] + [
            file_info(prepared / row["arrays"]) for row in layout["blocks"]
            if row["cluster"] == job["cluster"]
        ]
    output = args.output_dir / f"{job['id']}.json"
    report = dict(configuration=configuration, protocol=protocol, environment=environment(),
                  inputs=inputs, class_names=CLASS_NAMES, complete_split=False,
                  evaluated_images=0, images=[])
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["configuration"] != configuration or report["protocol"] != protocol:
            raise ValueError("Saved configuration or protocol differs. Choose a new output directory.")
        if report.get("inputs") != inputs:
            raise ValueError("Saved checkpoint or data identity differs or is missing. Choose a new output directory.")
        if report["complete_split"]:
            print(f"Complete: {job['id']} ({report['miou_percent']:.6f}% mIoU)", flush=True)
            return
    model = load_segformer(checkpoint)
    install_sra(model, k=job.get("k", 14), pattern=job.get("pattern", "dispersed32"),
                seed=protocol["seed"], dense=job.get("dense", False), method=job.get("method"),
                keep_scale=job.get("keep_scale", 1.0))
    specs = [(stage, block, layer, layer.attn) for stage, layers in enumerate(model.backbone.layers)
             for block, layer in enumerate(layers[1])]
    if "capacities" in job:
        from representative_context.controls.allocation import set_stage_capacities

        set_stage_capacities(specs, job["capacities"])
    if "cluster" in job:
        from representative_context.kernels.context import install

        assert layout["all_queries_audited"] and layout["identical_local_anchors_and_quotas"]
        cores = {(stage, block): core for stage, block, _, core in specs}
        for row in layout["blocks"]:
            if row["cluster"] == job["cluster"]:
                with np.load(prepared / row["arrays"], allow_pickle=False) as data:
                    install(cores[row["stage"], row["block"]], data["indices"], row["kv_side"], row["query_side"])
    total = sum((np.asarray(row["confusion_matrix"], dtype=np.int64) for row in report["images"]),
                start=np.zeros((8, 8), dtype=np.int64))
    for index, (image_path, label_path) in enumerate(pairs):
        image_id = image_path.relative_to(data_root).as_posix()
        if index < len(report["images"]):
            assert report["images"][index]["image"] == image_id
            continue
        image, target = read_pair(image_path, label_path)
        logits = predict_image(model, image[None].cuda(), window_size=job["window_size"], stride=job["window_size"] // 2)
        prediction = logits.argmax(1)[0].cpu().numpy()
        counts = confusion_matrix(prediction, target.numpy())
        total += counts
        report["images"].append(dict(image=image_id, scene=image_path.parent.parent.name,
                                     shape=list(target.shape), confusion_matrix=counts.tolist(),
                                     **segmentation_metrics(counts)))
        report.update(complete_split=index + 1 == len(pairs), evaluated_images=index + 1,
                      confusion_matrix=total.tolist(), **segmentation_metrics(total))
        update_json(output, report)
        print(f"{job['id']}: {index + 1}/{len(pairs)}, mIoU={report['miou_percent']:.6f}%", flush=True)


if __name__ == "__main__":
    main()

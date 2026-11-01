"""Recompute complete-split UAVid metrics from paired per-image confusion counts."""

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT


def metrics(counts):
    tp = np.diag(counts)
    union = counts.sum(0) + counts.sum(1) - tp
    iou = 100 * tp / (union + 1e-8)
    return dict(miou_percent=float(iou.mean()), class_iou_percent=iou.tolist(),
                pixel_accuracy_percent=float(100 * tp.sum() / (counts.sum() + 1e-8)))


def analyze(config, root):
    rows = []
    for job in config["configurations"]:
        record = json.loads((root / job["result"]).read_text(encoding="utf-8"))
        assert record["complete_split"] and record["evaluated_images"] == config["expected_images"]
        assert record["protocol"] == config["protocol"]
        assert len(record["images"]) == config["expected_images"]
        assert len({row["image"] for row in record["images"]}) == config["expected_images"]
        counts = np.sum([row["confusion_matrix"] for row in record["images"]], axis=0, dtype=np.int64)
        np.testing.assert_array_equal(counts, record["confusion_matrix"])
        recomputed = metrics(counts)
        for name, value in recomputed.items():
            np.testing.assert_allclose(value, record[name], rtol=0, atol=1e-10)
        for image in record["images"]:
            per_image = metrics(np.asarray(image["confusion_matrix"], dtype=np.int64))
            np.testing.assert_allclose(per_image["miou_percent"], image["miou_percent"], rtol=0, atol=1e-10)
        rows.append(dict(configuration=job["id"], images=len(record["images"]), **recomputed))
    return dict(configurations=len(rows), image_configuration_pairs=sum(row["images"] for row in rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/segformer_uavid/evaluation_suite.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.config.read_text(encoding="utf-8")), ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {result['configurations']} configurations and {result['image_configuration_pairs']} image-configuration pairs.")


if __name__ == "__main__":
    main()

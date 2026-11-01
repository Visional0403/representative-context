"""Convert official Cityscapes labelIds to labelTrainIds without changing originals."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from _bootstrap import ROOT
from representative_context.data.labels import cityscapes_train_ids


def convert(source):
    source = Path(source).resolve()
    target = source.with_name(source.name.replace("_labelIds.png", "_labelTrainIds.png"))
    if target == source:
        raise ValueError("Expected an official *_labelIds.png filename.")
    with Image.open(source) as image:
        labels = cityscapes_train_ids(np.asarray(image))
    if target.exists():
        with Image.open(target) as image:
            if not np.array_equal(np.asarray(image), labels):
                raise ValueError(f"Existing train IDs disagree with the official mapping: {target}")
        return False
    Image.fromarray(labels).save(target)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/Cityscapes")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    args = parser.parse_args()
    root = args.data_root.resolve()
    sources = sorted(path for split in args.splits for path in (root / "gtFine" / split).glob("*/*_labelIds.png"))
    if not sources:
        parser.error("No official Cityscapes labelIds files were found.")
    created = sum(convert(path) for path in sources)
    print(f"Checked {len(sources)} labels; created {created} train-ID files.")


if __name__ == "__main__":
    main()

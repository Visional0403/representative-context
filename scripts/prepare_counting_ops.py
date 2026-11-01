"""Fetch the pinned Linux PyTorch 2.10 / CUDA 12.8 deformable-attention package."""

import argparse
from pathlib import Path
import urllib.request

from _bootstrap import ROOT

REVISION = "a4431ff2d5767c279f0ff2335a87a6828b77c7a0"
BASE = f"https://huggingface.co/kernels-community/deformable-detr/resolve/{REVISION}/build/torch210-cxx11-cu128-x86_64-linux/"
FILES = (
    "__init__.py",
    "_ops.py",
    "layers.py",
    "metadata.json",
    "deformable_detr/__init__.py",
    "_deformable_detr_cuda_5129df0.abi3.so",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "third_party/sources/deformable_ops"
    )
    args = parser.parse_args()
    for name in FILES:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(BASE + name, target)
    print(args.output)


if __name__ == "__main__":
    main()

"""Create fixed RGB training tiles using the declared image and label canvases.

Label canvases retain their RGB encoding. The SegFormer and Twins training
readers apply the model-specific class mapping documented in docs/datasets.md.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image

from _bootstrap import ROOT


def create_tiles(data_root, output, manifest, limit=None):
    rows = manifest["tiles"] if limit is None else manifest["tiles"][:limit]
    groups = defaultdict(list)
    for sequence, frame, top, left in rows:
        groups[sequence, frame].append((top, left))
    size = manifest["crop_size"]
    written = skipped = 0
    for (sequence, frame), coordinates in groups.items():
        source = data_root / manifest["source_split"] / sequence
        with Image.open(source / "Images" / frame) as handle:
            image = handle.convert("RGB")
        with Image.open(source / "Labels" / frame) as handle:
            label = handle.convert("RGB")
        if image.size != label.size:
            raise ValueError(f"Image/label dimensions differ: {sequence}/{frame}")
        width, height = image.size
        for top, left in coordinates:
            name = f"{Path(frame).stem}_{top}_{left}.png"
            targets = [output / sequence / kind / name for kind in ("Images", "Labels")]
            if all(path.exists() for path in targets):
                skipped += 1
                continue
            if any(path.exists() for path in targets):
                raise FileExistsError(f"Only one member of a tile pair exists: {sequence}/{name}")
            if not (0 <= top < height and 0 <= left < width):
                raise ValueError(f"Invalid tile origin: {sequence}/{name}")
            box = (left, top, min(left + size, width), min(top + size, height))
            for source_image, target, padding in zip(
                (image, label), targets, (manifest["rgb_padding"], manifest["label_padding"])
            ):
                tile = Image.new("RGB", (size, size), tuple(padding))
                tile.paste(source_image.crop(box), (0, 0))
                target.parent.mkdir(parents=True, exist_ok=True)
                tile.save(target, format="PNG")
            written += 1
    return dict(requested_tiles=len(rows), written_pairs=written, existing_pairs=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/UAVid")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/UAVidCrops/train")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/segformer_uavid/training_crops.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(create_tiles(args.data_root, args.output, manifest, args.limit)))


if __name__ == "__main__":
    main()

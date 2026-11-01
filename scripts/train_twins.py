"""Train a dense Twins-SVT-S segmentation baseline from ImageNet initialization."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from representative_context.models.mmseg import prepare_mmseg
from representative_context.runtime import read_config, resource_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--crop-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare_mmseg(ROOT / "third_party/mmsegmentation")
    from mmengine import Config
    from mmengine.runner import Runner

    cfg = Config(read_config(args.config))
    if cfg.get("custom_imports"):
        from mmengine.utils import import_modules_from_strings

        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.backbone.init_cfg.checkpoint = str(
        args.pretrained or resource_path(cfg.checkpoint, ROOT)
    )
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        data = cfg[name].dataset
        override = (
            args.crop_root if data.data_root.endswith("UAVidCrops") else args.data_root
        )
        data.data_root = str(override or resource_path(data.data_root, ROOT))
        for transform in data.pipeline:
            for field in ("crop_size", "scale"):
                if field in transform:
                    transform[field] = tuple(transform[field])
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.resume = args.resume
    cfg.launcher = "none"
    runner = Runner.from_cfg(cfg)
    if args.dry_run:
        print(type(runner.model).__name__)
        return
    runner.train()


if __name__ == "__main__":
    main()

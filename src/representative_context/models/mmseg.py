"""MMSegmentation model construction and strict task-checkpoint loading."""

import sys
from pathlib import Path

import torch


def prepare_mmseg(source):
    """Import the selected segmentation models and training components."""
    sys.path.insert(0, str(Path(source).resolve()))
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)


def load_mmseg(configuration, checkpoint, source, device="cuda", protocol="slide1024"):
    prepare_mmseg(source)
    from mmengine import Config
    from mmseg.registry import MODELS
    cfg = Config(configuration)
    if protocol == "native":
        cfg.model.test_cfg = dict(mode="whole")
    else:
        size = int(protocol.removeprefix("slide"))
        stride = 768 if configuration["dataset"] == "Cityscapes" else size // 2
        cfg.model.test_cfg = dict(
            mode="slide", crop_size=(size, size), stride=(stride, stride)
        )
    cfg.test_pipeline = [dict(type="LoadImageFromFile"), dict(type="PackSegInputs")]
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.cfg = cfg
    return model.to(device).eval()

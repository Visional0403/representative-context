"""Task models with relative positions and globally retained special tokens."""

from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

from representative_context.adapters.dino import install_dino_point32
from representative_context.adapters.relative import install_relative


def load_structured_prediction_model(case, weights, source, k=16, dense=False):
    source = Path(source)
    sys.path.insert(0, str(source))
    if case == "iml":
        namespace = types.ModuleType("modules")
        namespace.__path__ = [str(source / "modules")]
        sys.modules["modules"] = namespace
        from iml_vit_model import iml_vit_model
        from modules.window_attention_ViT import get_rel_pos

        model = iml_vit_model()
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        controls = install_relative(model.encoder_net, get_rel_pos)
    elif case == "gamus":
        original_hub = torch.hub.load

        def local_hub(repository, name, *args, **kwargs):
            if repository != "facebookresearch/dinov2":
                raise ValueError(f"Unexpected backbone repository: {repository}")
            return original_hub(
                str(source.parent / "dinov2"),
                name,
                *args,
                **dict(kwargs, source="local"),
            )

        with patch("torch.hub.load", side_effect=local_hub):
            from models.dpt import DPT_DINOv2

            model = DPT_DINOv2(
                encoder="vitl",
                head_configs=[
                    {"name": "regression", "nclass": 1},
                    {"name": "segmentation", "nclass": 8},
                ],
                pretrained=False,
            )
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        controls = install_dino_point32(model.pretrained, special=1)
    else:
        raise ValueError(f"Unsupported structured task: {case}")
    for control in controls:
        control.k = k
        control.dense = dense
    return model.cuda().eval(), controls

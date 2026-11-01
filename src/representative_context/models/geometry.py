"""The complete Pi3 model with its published mixed-precision execution."""

from pathlib import Path
import sys
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from representative_context.adapters.multiview import install_multiview_attention


def load_geometry_model(weights, source, k=16, dense=False):
    sys.path.insert(0, str(Path(source)))
    from pi3.models.pi3 import Pi3

    linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return linspace(*args, **kwargs)

    with patch("torch.linspace", side_effect=cpu_linspace):
        with torch.device("meta"):
            model = Pi3()
    state = load_file(str(weights), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    for name, module in model.named_modules():
        if isinstance(
            module, (torch.nn.Linear, torch.nn.Conv2d)
        ) and not name.startswith(("point_head", "conf_head", "camera_head")):
            module.to(dtype=torch.bfloat16)
    model = model.cuda().eval()
    model.image_mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(
        1, 3, 1, 1
    )
    model.image_std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(
        1, 3, 1, 1
    )
    controls = install_multiview_attention(model, k, dense)
    model.context_controls = controls
    return model, controls

"""Complete GECO2 counting architecture with its fixed CUDA dependency."""

import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

from representative_context.adapters.hiera import install_hiera_attention


def load_counting_model(weights, source, k=16, dense=False):
    source = Path(source)
    sys.path.insert(0, str(source))
    package = types.ModuleType("models")
    package.__path__ = [str(source / "models"), str(source / "Deformable-DETR/models")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "models", loader=None, is_package=True
    )
    sys.modules["models"] = package
    directory = source.parent / "deformable_ops"
    kernel = sys.modules.get("representative_context_deformable")
    if kernel is None:
        spec = importlib.util.spec_from_file_location(
            "representative_context_deformable",
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        kernel = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel
        spec.loader.exec_module(kernel)
    sys.modules["MultiScaleDeformableAttention"] = kernel
    module = importlib.import_module("models.counter_infer")
    from hydra import compose, initialize_config_dir

    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    sam = {}
    for name, value in state.items():
        if name.startswith("sam_mask.mask_decoder."):
            sam["sam_mask_decoder." + name.removeprefix("sam_mask.mask_decoder.")] = (
                value
            )
        elif name.startswith("sam_mask.prompt_encoder_sam."):
            sam[
                "sam_prompt_encoder."
                + name.removeprefix("sam_mask.prompt_encoder_sam.")
            ] = value
        elif name == "sam_mask.no_mem_embed":
            sam["no_mem_embed"] = value

    def source_weights(url, *args, **kwargs):
        if not url.endswith("/sam2_hiera_base_plus.pt"):
            raise ValueError(f"Unexpected initialization checkpoint: {url}")
        return {"model": sam}

    def local_compose(config_name, *args, **kwargs):
        return compose(config_name=Path(config_name).name, *args, **kwargs)

    with (
        patch("torch.hub.load_state_dict_from_url", side_effect=source_weights),
        patch.object(module, "compose", side_effect=local_compose),
    ):
        with initialize_config_dir(
            config_dir=str(source / "configs"), version_base=None
        ):
            model = module.CNT(
                image_size=1024,
                num_objects=3,
                emb_dim=256,
                kernel_dim=3,
                reduction=16,
                zero_shot=False,
                training=False,
            )
    model.load_state_dict(state, strict=True)
    controls = install_hiera_attention(model, k, dense)
    return model.cuda().eval(), controls

"""Strict construction of pretrained spatial-reduction prediction models."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

from representative_context.adapters.huggingface import install_huggingface
from representative_context.adapters.pvt import PVTSpatialAttention, install_pvt


def load_prediction_model(case, weights, source=None, k=16, dense=False):
    """Reuse the full task state and replace global aggregation operations."""
    weights = Path(weights)
    if case in ("tcd_tiled", "glpn_nyu"):
        if case == "tcd_tiled":
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
            from safetensors.torch import load_file

            model = SegformerForSemanticSegmentation(
                SegformerConfig.from_json_file(weights / "config.json")
            )
            model.load_state_dict(
                load_file(weights / "model.safetensors", device="cpu"), strict=True
            )
        else:
            from transformers import GLPNForDepthEstimation

            model = GLPNForDepthEstimation.from_pretrained(
                str(weights), local_files_only=True
            )
        controls = install_huggingface(model, k=k, dense=dense)
    else:
        if source is None or not Path(source).is_dir():
            raise FileNotFoundError(
                "Prepare the pinned model source with scripts/prepare_sources.py."
            )
        sys.path.insert(0, str(Path(source).resolve()))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if case in ("polyp_kvasir", "hitnet"):
            if case == "polyp_kvasir":
                from lib.pvt import PolypPVT as constructor
            else:
                from lib.pvt import Hitnet as constructor
            # The complete task state supplies both backbone and prediction weights.
            with patch("torch.load", return_value={}):
                model = constructor()
        elif case == "changeformer":
            from models.ChangeFormer import ChangeFormerV6

            model = ChangeFormerV6(embed_dim=256)
        elif case == "trufor":
            from lib.config import config
            from lib.utils import get_model

            cfg = config.clone()
            cfg.defrost()
            cfg.merge_from_file(str(Path(source) / "lib/config/trufor_ph3.yaml"))
            cfg.freeze()
            model = get_model(cfg)
        else:
            raise ValueError(f"Unsupported prediction model: {case}")
        model.load_state_dict(state, strict=True)
        del state
        if case == "trufor":
            controls = []
            for stage, stride in enumerate((4, 8, 16, 32), 1):
                for prefix in ("block", "extra_block"):
                    for block in getattr(model.backbone, f"{prefix}{stage}"):
                        block.attn = PVTSpatialAttention(
                            block.attn, stride, k, 42 + len(controls), dense
                        ).eval()
                        controls.append(block.attn)
        else:
            backbone = model.Tenc_x2 if case == "changeformer" else model.backbone
            controls = install_pvt(
                backbone,
                k=k,
                dense=dense,
                seed_scheme="linear" if case == "hitnet" else "stage",
            )
    return model.cuda().eval(), controls

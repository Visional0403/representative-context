"""Released restoration, spectral-fusion and optical-flow task architectures."""

import argparse
import ast
import importlib.util
from pathlib import Path
import sys
import types

import torch

from representative_context.adapters.pvt import PVTSpatialAttention
from representative_context.adapters.fusion import install_fusion_attention
from representative_context.adapters.motion import install_motion_attention
from representative_context.budget import plan_capacity


def load_extra_prediction_model(
    case, weights, source, k=16, dense=False, dense_backend="sdpa"
):
    source = Path(source)
    if case == "gma_sintel":
        sys.path.insert(0, str(source / "core"))
        from network import RAFTGMA

        args = argparse.Namespace(
            num_heads=1,
            position_only=False,
            position_and_content=False,
            mixed_precision=True,
            replace=False,
            no_alpha=False,
            no_residual=False,
        )
        model = RAFTGMA(args)
    elif case in ("cave", "qb"):
        path = source / ("model.py" if case == "cave" else "Networks/Fusformer/net.py")
        spec = importlib.util.spec_from_file_location("fusion_task_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = (
            module.MainNet()
            if case == "cave"
            else module.MainNet(channel=4, msichannel=1, sf=4)
        )
    elif case in ("div2k", "gopro"):
        sys.path.insert(0, str(source))
        path = source / "transweather_model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # The architecture uses base_networks; ConvModule is an unused import.
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "mmcv.cnn")
        ]
        module = types.ModuleType("transweather_mean_prediction")
        exec(compile(tree, str(path), "exec"), module.__dict__)
        model = module.Transweather()
        # The released mean-prediction state defines these inference components.
        del model.Tenc.mini_patch_embed4
        del model.convtail.conv_output
    else:
        raise ValueError(f"Unsupported task model: {case}")
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    if case == "gma_sintel":
        capacity = plan_capacity(8, 1, 1, k).capacity_limit
        controls = [
            install_motion_attention(
                model, "dense" if dense else "sparse", capacity, 42
            )
        ]
    elif case in ("cave", "qb"):
        controls = install_fusion_attention(
            model, module.Attention, k, dense, dense_backend
        )
    else:
        controls = []
        for stage, stride in enumerate((4, 8, 16, 32), 1):
            branches = [(getattr(model.Tenc, f"block{stage}"), stride)]
            if stage < 4:
                branches.append(
                    (getattr(model.Tenc, f"patch_block{stage}"), stride * 2)
                )
            for blocks, actual_stride in branches:
                for block in blocks:
                    force_dense = (
                        block.attn.q.out_features // block.attn.num_heads > 128
                    )
                    control = PVTSpatialAttention(
                        block.attn,
                        actual_stride,
                        k,
                        42 + len(controls),
                        dense or force_dense,
                    )
                    control.force_dense = force_dense
                    block.attn = control.eval()
                    controls.append(control)
    return model.cuda().eval(), controls

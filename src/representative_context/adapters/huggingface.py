"""Inference adapters for Hugging Face SegFormer and GLPN encoders."""

import types
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from representative_context.budget import plan_capacity
from representative_context.layouts import make_ranks
from representative_context.kernels.indexed import build_layout, indexed_attention


def spatial_forward(self, hidden_states, height, width, output_attentions=False):
    if self.training or output_attentions:
        raise ValueError("Use inference without requesting a full attention matrix.")
    q = self.transpose_for_scores(self.query(hidden_states))
    reduced, kh, kw = hidden_states, height, width
    if self.sr_ratio > 1:
        batch, _, channels = hidden_states.shape
        reduced = self.sr(
            hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
        )
        kh, kw = reduced.shape[-2:]
        reduced = self.layer_norm(reduced.flatten(2).transpose(1, 2))
    key, value = self.transpose_for_scores(
        self.key(reduced)
    ), self.transpose_for_scores(self.value(reduced))
    plan = plan_capacity(self.query_stride, self.sr_ratio, kh * kw, self.capacity_k)
    self.last_plan = plan
    if self.capture_qkv:
        self.captured_qkv = (
            q.detach(),
            key.detach(),
            value.detach(),
            height,
            width,
            kh,
            kw,
        )
    if self.dense or plan.groups == 1:
        backend = (
            SDPBackend.EFFICIENT_ATTENTION
            if q.dtype == torch.float32
            else SDPBackend.FLASH_ATTENTION
        )
        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(q, key, value)
    else:
        shape = (height, width, kh, kw, plan.capacity_limit, q.device)
        if shape not in self.layout_cache:
            ranks = torch.tensor(
                make_ranks(kh, kw, plan.capacity_limit, seed=self.layout_seed),
                dtype=torch.int64,
                device=q.device,
            )
            self.layout_cache[shape] = build_layout(
                ranks, (height, width), (kh, kw), plan.capacity_limit
            )
        output = indexed_attention(q, key, value, self.layout_cache[shape])
    output = output.permute(0, 2, 1, 3).contiguous()
    return (output.reshape(*output.shape[:-2], self.all_head_size),)


def install_huggingface(model, k=16, seed=42, dense=False):
    encoder = model.glpn.encoder if hasattr(model, "glpn") else model.segformer.encoder
    staged_seeds = hasattr(model, "glpn")
    modules = []
    for stage, blocks in enumerate(encoder.block):
        for index, block in enumerate(blocks):
            module = block.attention.self
            module.query_stride, module.capacity_k, module.dense = (
                4 * 2**stage,
                k,
                dense,
            )
            module.layout_seed = seed + (
                100 * stage + index if staged_seeds else len(modules)
            )
            module.layout_cache, module.last_plan = {}, None
            module.capture_qkv, module.captured_qkv = False, None
            module.forward = types.MethodType(spatial_forward, module)
            modules.append(module)
    return modules

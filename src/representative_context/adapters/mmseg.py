"""Global-attention replacement for MMSegmentation SegFormer and Twins."""

from torch import nn

from representative_context.adapters.sra import SpatialReductionAttention


class ResidualSRA(nn.Module):
    """Preserve projection dropout and the original attention residual."""

    def __init__(
        self, original, stride, k, pattern, seed, dense, method=None, keep_scale=1.0
    ):
        super().__init__()
        self.core = SpatialReductionAttention(original, stride, k, pattern, seed, dense)
        if method is not None and not dense:
            from .selection import SelectionSRA

            self.core = SelectionSRA(original, stride, k, method, seed, keep_scale)
        self.proj_drop = original.proj_drop
        self.dropout_layer = original.dropout_layer

    def forward(self, x, hw_shape, identity=None):
        identity = x if identity is None else identity
        return identity + self.dropout_layer(self.proj_drop(self.core(x, *hw_shape)))


def install_mmseg(
    model,
    architecture,
    k=16,
    pattern="dispersed32",
    seed=42,
    dense=False,
    method=None,
    keep_scale=1.0,
):
    """Use indexed execution for global attention; preserve local modules."""
    wrappers, stride = [], 1
    if architecture == "twins_svt_s":
        from mmseg.models.backbones.twins import GSAEncoderLayer, LSAEncoderLayer

        stages = model.backbone.layers
        patches = model.backbone.patch_embeds
    elif architecture == "segformer_b0":
        stages = [layer[1] for layer in model.backbone.layers]
        patches = [layer[0] for layer in model.backbone.layers]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    for stage, (blocks, patch) in enumerate(zip(stages, patches)):
        step = patch.projection.stride
        if isinstance(step, int):
            step = (step, step)
        if step[0] != step[1]:
            raise ValueError("The capacity rule requires isotropic stage strides.")
        stride *= step[0]
        for index, block in enumerate(blocks):
            if architecture == "twins_svt_s":
                if isinstance(block, LSAEncoderLayer):
                    continue
                if not isinstance(block, GSAEncoderLayer):
                    raise TypeError(type(block))
            block.attn = ResidualSRA(
                block.attn,
                stride,
                k,
                pattern,
                seed + 100 * stage + index,
                dense,
                method,
                keep_scale,
            ).eval()
            wrappers.append(block.attn.core)
    return wrappers

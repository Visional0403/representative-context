"""SegFormer-B0 and its UAVid segmentation head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Drop residual paths independently per training example."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, padding, in_channels, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.permute(0, 2, 3, 1).contiguous())
        return x.permute(0, 3, 1, 2).contiguous(), height, width


class EfficientAttention(nn.Module):
    """Checkpoint-compatible SRA projections and native dense attention."""

    def __init__(self, embed_dim, num_heads, sr_ratio=1, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.embed_dim = embed_dim
        self.attn = nn.Module()
        self.attn.in_proj_weight = nn.Parameter(torch.randn(3 * embed_dim, embed_dim))
        self.attn.in_proj_bias = nn.Parameter(torch.randn(3 * embed_dim))
        self.attn.out_proj = nn.Linear(embed_dim, embed_dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, key, value = qkv.unbind(0)
        if self.sr_ratio > 1:
            reduced = self.sr(
                x.permute(0, 2, 1).reshape(batch, channels, height, width)
            )
            reduced = self.norm(reduced.reshape(batch, channels, -1).permute(0, 2, 1))
            kv = F.linear(reduced, self.attn.in_proj_weight, self.attn.in_proj_bias)
            kv = kv.reshape(batch, -1, 3, self.num_heads, self.head_dim).permute(
                2, 0, 3, 1, 4
            )
            _, key, value = kv.unbind(0)
        probabilities = self.attn_drop(
            ((q @ key.transpose(-2, -1)) * self.scale).softmax(-1)
        )
        output = (
            (probabilities @ value).transpose(1, 2).reshape(batch, tokens, channels)
        )
        return self.proj_drop(self.attn.out_proj(output))


class MixFFN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(embed_dim, hidden, 1),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Conv2d(hidden, embed_dim, 1),
            ]
        )

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = self.layers[0](x.transpose(1, 2).view(batch, channels, height, width))
        x = self.layers[2](self.layers[1](x))
        x = self.layers[4](self.layers[3](x))
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, sr_ratio, drop_path):
        super().__init__()
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.ffn(self.norm2(x), height, width))


class MixVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        dims, heads, ratios = (32, 64, 160, 256), (1, 2, 5, 8), (8, 4, 2, 1)
        self.layers = nn.ModuleList()
        rates = [x.item() for x in torch.linspace(0, 0.1, 8)]
        for stage, (dim, head, ratio) in enumerate(zip(dims, heads, ratios)):
            patch = OverlapPatchEmbed(
                7 if stage == 0 else 3,
                4 if stage == 0 else 2,
                3 if stage == 0 else 1,
                3 if stage == 0 else dims[stage - 1],
                dim,
            )
            blocks = nn.ModuleList(
                [Block(dim, head, ratio, rates[stage * 2 + i]) for i in range(2)]
            )
            self.layers.append(nn.Sequential(patch, blocks, nn.LayerNorm(dim)))

    def forward(self, x):
        features = []
        batch = x.shape[0]
        for patch, blocks, norm in self.layers:
            x, height, width = patch(x)
            x = x.flatten(2).transpose(1, 2)
            for block in blocks:
                x = block(x, height, width)
            x = (
                norm(x)
                .reshape(batch, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            features.append(x)
        return features


def conv_norm_relu(in_channels, out_channels):
    block = nn.Module()
    block.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    block.bn = nn.BatchNorm2d(out_channels)
    block.relu = nn.ReLU(inplace=True)
    return block


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, num_classes, 1)
        self.convs = nn.ModuleList(
            [conv_norm_relu(dim, 256) for dim in (32, 64, 160, 256)]
        )
        self.fusion_conv = conv_norm_relu(4 * 256, 256)

    def forward(self, features):
        outputs = []
        for block, feature in zip(self.convs, features):
            x = block.relu(block.bn(block.conv(feature)))
            outputs.append(
                F.interpolate(
                    x, size=features[0].shape[2:], mode="bilinear", align_corners=False
                )
            )
        x = torch.cat(outputs, dim=1)
        x = self.fusion_conv.relu(self.fusion_conv.bn(self.fusion_conv.conv(x)))
        return F.interpolate(
            self.conv_seg(self.dropout(x)),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )


class SegFormerB0(nn.Module):
    """Eight-class segmentation model with strict pure-weight loading."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = SegFormerHead(num_classes)

    def forward(self, image):
        return self.decode_head(self.backbone(image))


def load_segformer(checkpoint, device="cuda"):
    model = SegFormerB0()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

"""Count actual retained interactions for a SegFormer-B0 input using CPU only."""

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT
from representative_context.budget import plan_capacity
from representative_context.layouts import make_ranks
from representative_context.runtime import write_json


def count_stages(height, width, k, pattern, seed):
    rows = []
    qh, qw = height, width
    for stage, (stride, ratio, channels) in enumerate(
        zip((4, 8, 16, 32), (8, 4, 2, 1), (32, 64, 160, 256))
    ):
        patch_stride = 4 if stage == 0 else 2
        qh, qw = (qh + patch_stride - 1) // patch_stride, (
            qw + patch_stride - 1
        ) // patch_stride
        kh, kw = qh // ratio, qw // ratio
        plan = plan_capacity(stride, ratio, kh * kw, k)
        blocks = []
        for block in range(2):
            ranks = make_ranks(
                kh, kw, plan.capacity_limit, pattern, seed + stage * 100 + block
            )
            key_counts = Counter(rank % plan.groups for rank in ranks)
            query_counts = Counter()
            for token, rank in enumerate(ranks):
                y, x = divmod(token, kw)
                copies_h = ((y + 1) * qh + kh - 1) // kh - (y * qh + kh - 1) // kh
                copies_w = ((x + 1) * qw + kw - 1) // kw - (x * qw + kw - 1) // kw
                query_counts[rank % plan.groups] += copies_h * copies_w
            edges = sum(query_counts[g] * key_counts[g] for g in range(plan.groups))
            dense_edges = qh * qw * kh * kw
            blocks.append(
                {
                    "block": block,
                    "retained_edges_per_head": edges,
                    "dense_edges_per_head": dense_edges,
                    "retained_fraction": edges / dense_edges,
                    "attention_matmul_flops": 4 * edges * channels,
                }
            )
        rows.append(
            {
                "stage": stage,
                "query_shape": [qh, qw],
                "kv_shape": [kh, kw],
                "plan": dataclasses.asdict(plan),
                "blocks": blocks,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--pattern", default="dispersed32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.height, args.width) < 32:
        parser.error("Both input dimensions must be at least 32 pixels.")
    report = {
        "input_shape": [args.height, args.width],
        "k": args.k,
        "stages": count_stages(
            args.height, args.width, args.k, args.pattern, args.seed
        ),
        "flops_scope": "QK and attention-value products; two FLOPs per multiply-add",
    }
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Prepare and audit all fixed-local-context layout controls on CPU."""

import argparse
from pathlib import Path
import numpy as np

from _bootstrap import ROOT
from representative_context.budget import plan_capacity
from representative_context.controls.context_layout import audit, construct
from representative_context.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("segformer_uavid", "twins_cityscapes"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    native = args.case == "segformer_uavid"
    side = 2048 if native else 1024
    indices = ((0, 1),) * 4 if native else ((1,), (1,), (1, 3, 5, 7, 9), (1, 3))
    heads = (1, 2, 5, 8) if native else (2, 4, 8, 16)
    args.output_dir.mkdir(parents=True)
    rows = []
    for stage, blocks in enumerate(indices):
        stride = 4 * 2**stage
        sr = 8 // 2**stage
        qside = side // stride
        kside = qside // sr
        capacity = plan_capacity(stride, sr, kside**2, 14).capacity_limit
        for block in blocks:
            fixed = None
            for cluster in (4, 2, 1):
                seed = 42 + 100 * stage + block
                layout = construct(kside, capacity, cluster, seed)
                if fixed is None:
                    fixed = [part.copy() for part in layout[1:]]
                else:
                    for before, after in zip(fixed, layout[1:]):
                        assert np.array_equal(before, after)
                checked = audit(*layout, kside, qside)
                filename = f"stage{stage}_block{block}_cluster{cluster}.npz"
                np.savez_compressed(args.output_dir / filename, indices=layout[0])
                rows.append(
                    dict(
                        stage=stage,
                        block=block,
                        heads=heads[stage],
                        head_dimension=32,
                        query_side=qside,
                        kv_side=kside,
                        capacity=capacity,
                        seed=seed,
                        cluster=cluster,
                        arrays=filename,
                        audit=checked,
                    )
                )
                print(filename, flush=True)
    write_json(
        args.output_dir / "manifest.json",
        dict(
            case=args.case,
            k=14,
            input_side=side,
            clusters=[4, 2, 1],
            blocks=rows,
            all_queries_audited=True,
            identical_local_anchors_and_quotas=True,
        ),
    )


if __name__ == "__main__":
    main()

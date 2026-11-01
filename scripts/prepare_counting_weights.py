"""Assemble the complete GECO2 inference state from task and SAM2 checkpoints."""

import argparse
from pathlib import Path

import torch

from _bootstrap import ROOT


def assemble(task, sam):
    state = {
        name.removeprefix("module."): value
        for name, value in task["model"].items()
        if not name.removeprefix("module.").startswith(
            ("class_embed_aux.", "bbox_embed_aux.")
        )
    }
    for name, value in sam["model"].items():
        if name.startswith("sam_mask_decoder."):
            target = "sam_mask.mask_decoder." + name.removeprefix("sam_mask_decoder.")
        elif name.startswith("sam_prompt_encoder."):
            target = "sam_mask.prompt_encoder_sam." + name.removeprefix(
                "sam_prompt_encoder."
            )
        elif name == "no_mem_embed":
            target = "sam_mask.no_mem_embed"
        else:
            continue
        state.setdefault(target, value)
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected named inference tensors.")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "weights/geco2/geco2_fsc147.pth"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an output checkpoint that does not already exist.")
    state = assemble(
        torch.load(args.task_checkpoint, map_location="cpu", weights_only=True),
        torch.load(args.sam2_checkpoint, map_location="cpu", weights_only=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Wrote {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()

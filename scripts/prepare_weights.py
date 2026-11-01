"""Convert an author checkpoint into the task-only tensor format used here."""

import argparse
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from representative_context.runtime import read_config, resource_path

KEYS = {
    "segformer_cityscapes": "state_dict",
    "twins_cityscapes": "state_dict",
    "twins_uavid": "state_dict",
    "trufor": "state_dict",
    "gamus": "model",
    "changeformer": "model_G_state_dict",
    "cave": "net",
    "qb": "net",
    "div2k": "params",
    "gopro": "params",
}


def load_tensor_checkpoint(path):
    globals_used = set(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
    numpy_globals = {
        "numpy.dtype",
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
    }
    if globals_used <= numpy_globals:
        allowed = [
            np.dtype,
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (np._core.multiarray.scalar, "numpy._core.multiarray.scalar"),
            type(np.dtype("float64")),
            type(np.dtype("int64")),
        ]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config)
    case = cfg.get("case")
    if case is None:
        case = args.config.parent.name
    target = args.output or resource_path(
        cfg.get("weights", cfg.get("checkpoint")), ROOT
    )
    if target.exists():
        parser.error("Choose an output checkpoint that does not already exist.")
    if case == "geco2":
        parser.error(
            "Use prepare_counting_weights.py with both the task and SAM2 checkpoints."
        )
    state = load_tensor_checkpoint(args.input)
    key = KEYS.get(case)
    if key is not None and key in state:
        state = state[key]
    if case in ("cave", "gma_sintel"):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    if case in ("div2k", "gopro"):
        state = {
            name: value
            for name, value in state.items()
            if not name.startswith("var_conv.")
        }
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError(
            "The selected checkpoint field must contain only named tensors."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    print(f"Wrote {len(state)} tensors to {target}")


if __name__ == "__main__":
    main()

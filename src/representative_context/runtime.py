"""Configuration, reproducibility, and JSON output helpers."""

import importlib.metadata
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, threads: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resource_path(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def update_json(path: Path, data):
    """Atomically replace a progress report after a complete JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_info(path: Path) -> dict:
    """Record the selected file and its basic filesystem properties."""
    path = Path(path).resolve()
    info = path.stat()
    return dict(path=str(path), size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def input_shapes(value):
    """Describe tensor inputs used for evaluation or timing."""
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, (tuple, list)):
        return [input_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): input_shapes(item) for key, item in value.items()}
    return type(value).__name__


def evaluation_inputs(checkpoint: Path, data_root: Path, pairs) -> dict:
    """Record the checkpoint, dataset location and ordered evaluation pairs."""
    root = Path(data_root).resolve()
    return dict(checkpoint=file_info(checkpoint), data_root=str(root),
                samples=[dict(image=Path(image).resolve().relative_to(root).as_posix(),
                              label=Path(label).resolve().relative_to(root).as_posix())
                         for image, label in pairs])


def prediction_inputs(weights: Path, data_root: Path, samples: Path | None,
                      source: Path | None, configuration: dict) -> dict:
    """Record model files, evaluation settings and the selected data source."""
    weights, root = Path(weights).resolve(), Path(data_root).resolve()
    weight_files = [weights] if weights.is_file() else sorted(
        path for path in weights.rglob("*")
        if path.is_file() and path.suffix in {".pt", ".pth", ".bin", ".safetensors", ".json"}
    )
    result = dict(weights=[file_info(path) for path in weight_files],
                  configuration=configuration, data_root=str(root))
    if samples is not None:
        result["sample_manifest"] = file_info(samples)
    if source is not None and (Path(source).resolve() / ".source.json").is_file():
        result["source_specification"] = read_config(Path(source).resolve() / ".source.json")
    return result


def environment():
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    for package in ("triton", "torchvision", "mmsegmentation", "mmengine"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(
            gpu=torch.cuda.get_device_name(), gpu_free_bytes=free, gpu_total_bytes=total
        )
    return report

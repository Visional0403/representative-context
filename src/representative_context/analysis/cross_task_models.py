"""Cross-task model hooks preserving the declared dense trajectories."""

import io
import math
import struct
import zipfile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from representative_context.analysis.cross_task import (
    Diagnostic,
    geometry,
    relative_difference,
)
from representative_context.kernels.prefix import dense_attention
from representative_context.runtime import read_config
from representative_context.tasks.registry import load_task_model


def decode_image(data: bytes, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None].to(device)


def decode_flow(data: bytes) -> np.ndarray:
    if data[:4] != b"PIEH":
        raise ValueError("Invalid Sintel flow header.")
    width, height = struct.unpack_from("<ii", data, 4)
    flow = np.frombuffer(data, dtype="<f4", offset=12)
    if flow.size != width * height * 2:
        raise ValueError("Invalid Sintel flow length.")
    return flow.reshape(height, width, 2)


def stamp_arrays(target, prefix, new):
    for name, value in new.items():
        key = prefix + name
        if key in target:
            raise RuntimeError("Diagnostic array names must be unique.")
        target[key] = value


class GMAEvaluator:

    def __init__(self, root, data_root=None, weights=None, source=None):
        cfg = read_config(root / "configs/gma_sintel/validation.json")
        self.weight = weights or root / cfg["weights"]
        self.model, controls = load_task_model(
            "gma_sintel", self.weight, source or root / cfg["source"], dense=True
        )
        self.model.att = controls[0].original
        self.model.update_block.aggregator = self.model.update_block.aggregator.original
        self.model.args.mixed_precision = False
        data_root = data_root or root / cfg["data_root"]
        self.images = zipfile.ZipFile(data_root / "MPI-Sintel-training_images.zip")
        self.extras = zipfile.ZipFile(data_root / "MPI-Sintel-training_extras.zip")
        self.diag = Diagnostic()
        self.geom = geometry(55, 128, 55, 128, 1760)

    def __call__(self, row, index):
        from utils.utils import InputPadder

        _, _, first_path, second_path, gt_path = row["pair"]
        first = decode_image(self.images.read(first_path), torch.device("cuda"))
        second = decode_image(self.images.read(second_path), torch.device("cuda"))
        gt = decode_flow(self.extras.read(gt_path))
        padder = InputPadder(first.shape)
        first, second = padder.pad(first, second)
        entries, arrays, state = ([], {}, {"step": 0})
        att, agg = (self.model.att, self.model.update_block.aggregator)
        assert not att.args.position_only and (not att.args.position_and_content)
        assert agg.heads == 1 and agg.project is None

        def attention_hook(module, inputs, output):
            q, k = module.to_qk(inputs[0]).chunk(2, 1)
            q = q.flatten(2).transpose(1, 2)[:, None]
            k = k.flatten(2).transpose(1, 2)[:, None]
            qs = torch.as_tensor(self.geom["queries"], device=q.device)
            state["scores"] = q[0, :, qs] * module.scale @ k[0].transpose(-1, -2)

        def before_aggregate(module, inputs):
            state["step"] += 1
            if state["step"] not in (6, 12):
                return
            fmap = inputs[1]
            value = module.to_v(fmap).flatten(2).transpose(1, 2)
            label = f"u{state['step']}"
            detail, raw, dense = self.diag(
                state["scores"],
                value,
                self.geom,
                label,
                cpu_spot=index == 0 and state["step"] == 6,
            )
            entries.append(detail)
            stamp_arrays(arrays, "", raw)
            state["reference"] = (
                fmap.flatten(2)[0, :, self.geom["queries"]].T + module.gamma * dense[0]
            )

        def after_aggregate(module, inputs, output):
            if state["step"] not in (6, 12):
                return
            actual = output.flatten(2)[0, :, self.geom["queries"]].T
            absolute, relative = relative_difference(actual, state["reference"])
            entries[-1]["native_max_abs_error"] = absolute
            entries[-1]["native_relative_error"] = relative

        handles = [
            att.register_forward_hook(attention_hook),
            agg.register_forward_pre_hook(before_aggregate),
            agg.register_forward_hook(after_aggregate),
        ]
        try:
            _, prediction = self.model(first, second, iters=12, test_mode=True)
        finally:
            for handle in handles:
                handle.remove()
        assert state["step"] == 12 and [r["block"] for r in entries] == ["u6", "u12"]
        current = padder.unpad(prediction[0]).permute(1, 2, 0).float().cpu().numpy()
        valid = np.isfinite(gt).all(-1) & (abs(gt) < 1000000000.0).all(-1)
        epe = float(
            np.linalg.norm(current.astype(float) - gt.astype(float), axis=-1)[
                valid
            ].mean()
        )
        return (entries, arrays, {"dense_epe": epe})


class FusformerEvaluator:

    def __init__(self, root, data_root=None, weights=None, source=None):
        import h5py

        cfg = read_config(root / "configs/cave/validation.json")
        self.weight = weights or root / cfg["weights"]
        self.model, controls = load_task_model(
            "cave",
            self.weight,
            source or root / cfg["source"],
            dense=True,
            dense_backend="sdpa",
        )
        self.data = h5py.File(
            (data_root or root / cfg["data_root"]) / "Test_CAVE.h5", "r"
        )
        self.geom = geometry(512, 512, 512, 512, 65536)
        self.diag = Diagnostic()
        self.entries, self.arrays = ([], {})
        attentions = [control.original for control in controls]
        assert len(attentions) == 6
        for index, a in enumerate(attentions):

            def replacement(x, mask=None, a=a, layer=index):
                if mask is not None:
                    raise RuntimeError(
                        "The declared fusion inputs require no attention mask."
                    )
                batch, count, channels = x.shape
                q, k, v = [
                    t.reshape(batch, count, a.heads, -1).permute(0, 2, 1, 3)
                    for t in a.to_qkv(x).chunk(3, -1)
                ]
                if layer in (0, 2, 5):
                    qs = torch.as_tensor(self.geom["queries"], device=x.device)
                    scores = q[0, :, qs] * a.scale @ k[0].transpose(-1, -2)
                    detail, raw, ref = self.diag(
                        scores,
                        v[0],
                        self.geom,
                        f"b{layer + 1}",
                        cpu_spot=self.scene_index == 0 and layer == 0,
                    )
                    self.entries.append(detail)
                    stamp_arrays(self.arrays, "", raw)
                output = dense_attention(q, k, v, a.scale)
                if layer in (0, 2, 5):
                    absolute, relative = relative_difference(output[0, :, qs], ref)
                    self.entries[-1]["native_max_abs_error"] = absolute
                    self.entries[-1]["native_relative_error"] = relative
                return a.to_out(output.transpose(1, 2).reshape(batch, count, channels))

            a.forward = replacement

    def __call__(self, row, index):
        self.entries, self.arrays, self.scene_index = ([], {}, index)
        i = row["index"]
        low = torch.from_numpy(self.data["LRHSI"][i].astype(np.float32))[None].cuda()
        rgb = torch.from_numpy(self.data["RGB"][i].astype(np.float32))[None].cuda()
        pred = self.model(low, rgb)[0][0].cpu().numpy()
        error = pred.astype(float) - self.data["GT"][i].astype(float)
        psnr = -10 * math.log10(max(float(np.mean(error**2)), 1e-16))
        assert [r["block"] for r in self.entries] == ["b1", "b3", "b6"]
        return (self.entries, self.arrays, {"dense_psnr_db": psnr})


class IMLEvaluator:

    def __init__(self, root, data_root=None, weights=None, source=None):
        cfg = read_config(root / "configs/iml/validation.json")
        self.weight = weights or root / cfg["weights"]
        self.model, self.controls = load_task_model(
            "iml", self.weight, source or root / cfg["source"], dense=True
        )
        self.data_root = data_root or root / cfg["data_root"]
        self.diag = Diagnostic()
        self.entries, self.arrays, self.refs = ([], {}, {})
        self.Image, self.get_rel_pos = (Image, self.controls[0].get_rel_pos)
        get_rel_pos = self.get_rel_pos
        for index, control in enumerate(self.controls):
            if not control.is_global:
                continue
            a = control.a

            def before(module, inputs, layer=index):
                x = inputs[0]
                b, h, w, c = x.shape
                q, k, v = (
                    module.qkv(x)
                    .reshape(b, h * w, 3, module.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )
                qs = torch.as_tensor(self.geom["queries"], device=x.device)
                query = q[0, :, qs]
                scores = query * module.scale @ k[0].transpose(-1, -2)
                rh = self.get_rel_pos(h, h, module.rel_pos_h)[qs // w]
                rw = self.get_rel_pos(w, w, module.rel_pos_w)[qs % w]
                rows = torch.einsum("hqd,qkd->hqk", query, rh)
                cols = torch.einsum("hqd,qkd->hqk", query, rw)
                scores += (rows[:, :, :, None] + cols[:, :, None, :]).flatten(-2)
                detail, raw, ref = self.diag(
                    scores,
                    v[0],
                    self.geom,
                    f"b{layer + 1}",
                    cpu_spot=self.scene_index == 0 and layer == 2,
                )
                detail["includes_relative_positions"] = True
                self.entries.append(detail)
                stamp_arrays(self.arrays, "", raw)
                self.refs[layer] = module.proj(
                    ref.transpose(0, 1).reshape(1, len(qs), c)
                )

            def after(module, inputs, output, layer=index):
                actual = output.flatten(1, 2)[:, self.geom["queries"]]
                absolute, relative = relative_difference(actual, self.refs.pop(layer))
                self.entries[-1]["native_max_abs_error"] = absolute
                self.entries[-1]["native_relative_error"] = relative

            a.register_forward_pre_hook(before)
            a.register_forward_hook(after)

    def __call__(self, row, index):
        raw = np.asarray(self.Image.open(self.data_root / row["image"]).convert("RGB"))[
            :1024, :1024
        ]
        vh, vw = row["valid_grid"]
        self.geom = geometry(64, 64, vh, vw, 1024)
        self.entries, self.arrays, self.refs, self.scene_index = ([], {}, {}, index)
        x = torch.from_numpy(raw.transpose(2, 0, 1).copy()).float()[None].cuda() / 255
        x = F.pad(x, (0, 1024 - x.shape[-1], 0, 1024 - x.shape[-2]))
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        features = self.model.encoder_net((x - mean) / std)
        output = self.model.predict_head(
            list(self.model.featurePyramid_net(features).values())
        )
        assert [r["block"] for r in self.entries] == ["b3", "b6", "b9", "b12"]
        return (
            self.entries,
            self.arrays,
            {"dense_output_mean": float(output.mean()), "valid_patch_grid": [vh, vw]},
        )

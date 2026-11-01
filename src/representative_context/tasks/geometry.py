"""Full-sequence multiview point-map evaluation with one similarity per group."""

import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F


def fit_similarity(source, target):
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    covariance = y.T @ x / len(x)
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vh))
    rotation = u @ np.diag(sign) @ vh
    variance = (x * x).sum() / len(x)
    scale = float((singular * sign).sum() / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Point-map alignment requires a finite positive scale.")
    return scale, rotation, b - scale * (rotation @ a)


def relative_depth_relief(values):
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    if not np.isfinite(quantiles).all() or quantiles[1] <= 0:
        raise ValueError("Relative depth relief requires a finite positive median.")
    return float((quantiles[2] - quantiles[0]) / quantiles[1]), quantiles.tolist()


class GeometryTask:
    def __init__(self, data_root, samples):
        self.root = Path(data_root)
        self.rows = json.loads(Path(samples).read_text(encoding="utf-8"))
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        return [(self.rows[len(self.rows) // 2], 1.0)]

    def prepare(self, row):
        images = [
            Image.open(self.root / frame["rgb"]).convert("RGB")
            for frame in row["frames"]
        ]
        width, height = images[0].size
        scale = math.sqrt(255000 / (width * height))
        tw, th = round(width * scale / 14), round(height * scale / 14)
        while tw * th * 196 > 255000:
            if tw / th > width / height:
                tw -= 1
            else:
                th -= 1
        inputs = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        image.resize((tw * 14, th * 14), Image.Resampling.LANCZOS)
                    ).copy()
                )
                .permute(2, 0, 1)
                .float()
                / 255
                for image in images
            ]
        )[None].cuda()
        yy, xx = np.mgrid[:height, :width]
        first = row["frames"][0]["pose_t_xyzw"]
        first_pose = np.eye(4)
        first_pose[:3, :3] = Rotation.from_quat(first[3:]).as_matrix()
        first_pose[:3, 3] = first[:3]
        target = []
        for frame in row["frames"]:
            z = (
                np.asarray(Image.open(self.root / frame["depth"])).astype(np.float64)
                / 5000
            )
            valid = np.isfinite(z) & (z > 0)
            local = np.stack(((xx - 319.5) * z / 525, (yy - 239.5) * z / 525, z), -1)
            pose = np.eye(4)
            values = frame["pose_t_xyzw"]
            pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
            pose[:3, 3] = values[:3]
            relative = np.linalg.inv(first_pose) @ pose
            target.append(
                dict(
                    points=local @ relative[:3, :3].T + relative[:3, 3],
                    local_points=local,
                    valid=valid,
                    pose=relative,
                )
            )
        return inputs, target

    def forward(self, model, inputs):
        control = model.context_controls[0]
        self.collect_depth_relief = control.dense or control.k == 17
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(inputs)

    def score(self, output, target, inputs):
        prediction = output["points"][0].float().permute(0, 3, 1, 2)
        height, width = target[0]["valid"].shape
        prediction = (
            F.interpolate(
                prediction, (height, width), mode="bilinear", align_corners=False
            )
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        local_depth = (
            F.interpolate(
                output["local_points"][0, ..., 2].unsqueeze(1).float(),
                (height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            .cpu()
            .numpy()
            .astype(np.float64)
            if self.collect_depth_relief
            else None
        )
        ys = np.linspace(0, height - 1, 64).round().astype(int)
        xs = np.linspace(0, width - 1, 64).round().astype(int)
        sub = ys[:, None], xs[None, :]
        fit_a, fit_b = [], []
        for pred, gt in zip(prediction, target):
            if not np.isfinite(pred[gt["valid"]]).all():
                raise ValueError(
                    "Nonfinite point prediction at a valid reference location."
                )
            valid = gt["valid"][sub]
            fit_a.append(pred[sub][valid])
            fit_b.append(gt["points"][sub][valid])
        scale, rotation, translation = fit_similarity(
            np.concatenate(fit_a), np.concatenate(fit_b)
        )
        frames = []
        for index, (pred, gt) in enumerate(zip(prediction, target)):
            valid = gt["valid"]
            aligned = scale * pred[valid] @ rotation.T + translation
            distance = np.linalg.norm(aligned - gt["points"][valid], axis=-1)
            denominator = np.linalg.norm(gt["local_points"][valid], axis=-1)
            readout = {}
            if local_depth is not None:
                predicted_relief, pq = relative_depth_relief(local_depth[index][valid])
                truth_relief, gq = relative_depth_relief(
                    gt["local_points"][..., 2][valid]
                )
                readout = dict(
                    predicted_depth_relief=predicted_relief,
                    target_depth_relief=truth_relief,
                    predicted_depth_quantiles=pq,
                    target_depth_quantiles=gq,
                )
            frames.append(
                dict(
                    point_relative_error=float((distance / denominator).mean()),
                    point_rmse_m=float(np.sqrt((distance * distance).mean())),
                    point_delta1=float((distance < 0.25 * denominator).mean()),
                    valid_points=int(valid.sum()),
                    **readout,
                )
            )
        poses = output["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        re, te = [], []
        for index in range(1, len(target)):
            predicted = np.linalg.inv(poses[index - 1]) @ poses[index]
            gt = np.linalg.inv(target[index - 1]["pose"]) @ target[index]["pose"]
            re.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (np.trace(predicted[:3, :3] @ gt[:3, :3].T) - 1) / 2,
                                -1,
                                1,
                            )
                        )
                    )
                )
            )
            a, b = predicted[:3, 3], gt[:3, 3]
            if np.linalg.norm(b) > 1e-6:
                te.append(
                    180.0
                    if np.linalg.norm(a) <= 1e-12
                    else float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(a, b)
                                    / np.linalg.norm(a)
                                    / np.linalg.norm(b),
                                    -1,
                                    1,
                                )
                            )
                        )
                    )
                )
        return dict(
            frames=frames,
            rotation_errors_deg=re,
            translation_direction_errors_deg=te,
            alignment=dict(
                scale=scale,
                rotation=rotation.tolist(),
                translation=translation.tolist(),
            ),
        )

    def aggregate(self, records):
        frames = [frame for row in records for frame in row["metrics"]["frames"]]
        rotations = [
            value for row in records for value in row["metrics"]["rotation_errors_deg"]
        ]
        translations = [
            value
            for row in records
            for value in row["metrics"]["translation_direction_errors_deg"]
        ]
        return dict(
            point_relative_error=statistics.mean(
                frame["point_relative_error"] for frame in frames
            ),
            point_rmse_m=statistics.mean(frame["point_rmse_m"] for frame in frames),
            point_delta1=statistics.mean(frame["point_delta1"] for frame in frames),
            rotation_rpe_deg=statistics.mean(rotations),
            translation_direction_error_deg=statistics.mean(translations),
            evaluated_frames=len(frames),
            evaluated_rotation_pairs=len(rotations),
            evaluated_translation_pairs=len(translations),
        )

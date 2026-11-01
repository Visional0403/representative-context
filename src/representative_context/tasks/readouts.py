"""Application quantities collected from the same predictions and reference masks."""

import numpy as np
from torch.nn import functional as F


def collect_readouts(case, record, output, target, metrics):
    """Return application values without changing task scores or model execution."""
    if case in ("tcd_tiled", "changeformer"):
        counts = metrics["counts"]
        pixels = sum(counts.values())
        return [dict(id=str(record["id"]), valid_pixels=pixels,
                     prediction=100 * (counts["TP"] + counts["FP"]) / pixels,
                     target=100 * (counts["TP"] + counts["FN"]) / pixels)]
    if case == "cave":
        prediction = output[0].detach().cpu().numpy().astype(np.float64)
        truth = np.asarray(target, dtype=np.float64)
        wavelengths = np.arange(400, 701, 10, dtype=np.float64)[:, None, None]
        if truth.shape != prediction.shape or truth.shape[0] != 31:
            raise ValueError("CAVE centroid requires paired 31-band spectral cubes.")
        valid = truth.sum(axis=0) > 1e-4

        def centroid(cube):
            values = (wavelengths * cube).sum(axis=0) / np.maximum(cube.sum(axis=0), 1e-12)
            return float(values[valid].mean())

        return [dict(id=str(record["id"]), valid_pixels=int(valid.sum()),
                     prediction=centroid(prediction), target=centroid(truth))]
    if case == "gma_sintel":
        prediction = target["padder"].unpad(output[0]).float().permute(1, 2, 0)
        prediction = prediction.detach().cpu().numpy().astype(np.float64)
        truth = np.asarray(target["flow"], dtype=np.float64)
        return [dict(id=str(record["id"]), valid_pixels=truth.shape[0] * truth.shape[1],
                     prediction=float(np.linalg.norm(prediction, axis=-1).mean()),
                     target=float(np.linalg.norm(truth, axis=-1).mean()))]
    if case == "glpn_nyu":
        prediction = output
        if prediction.shape != target.shape:
            prediction = F.interpolate(prediction[None, None], size=target.shape,
                                       mode="bilinear", align_corners=False)[0, 0]
        valid = (target > 0.001) & (target < 10.0)
        valid[:45] = False
        valid[471:] = False
        valid[:, :41] = False
        valid[:, 601:] = False
        p = prediction[valid].clamp(0.001, 10.0).detach().double()
        y = target[valid].detach().double()
        return [dict(id=str(record["id"]), valid_pixels=p.numel(),
                     prediction=float(p.mean()), target=float(y.mean()),
                     predicted_depth_sum_m=float(p.sum()), gt_depth_sum_m=float(y.sum()))]
    if case == "pi3":
        return [dict(id=f"{record['id']}:{i:02d}", valid_pixels=frame["valid_points"],
                     prediction=100 * frame["predicted_depth_relief"],
                     target=100 * frame["target_depth_relief"])
                for i, frame in enumerate(metrics["frames"])
                if "predicted_depth_relief" in frame]
    return []

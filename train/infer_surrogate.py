"""Load a trained surrogate checkpoint and run forward passes on normalized inputs."""

from __future__ import annotations

from pathlib import Path

import torch

from train.field_dataset import params_to_grid_tensor
from train.hdf5_loader import pooled_features
from train.surrogate_models import TinyConv3dSurrogate, build_field_surrogate


def load_checkpoint(path: Path, *, prefer_tfno: bool, device: torch.device) -> tuple[torch.nn.Module, dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        ckpt = torch.load(path, map_location=device)

    arch = ckpt.get("arch")
    if arch == "tiny_conv3d":
        model = TinyConv3dSurrogate().to(device)
    elif arch == "tfno":
        model = build_field_surrogate(prefer_tfno=True).to(device)
    else:
        model = build_field_surrogate(prefer_tfno=prefer_tfno).to(device)
        try:
            model.load_state_dict(ckpt["model_state"], strict=True)
        except Exception:
            model = TinyConv3dSurrogate().to(device)
            model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        return model, ckpt

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def forward_pooled(
    model: torch.nn.Module,
    ckpt: dict,
    inp: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, float]:
    rpm_lo, rpm_hi = float(ckpt["rpm_bounds"][0]), float(ckpt["rpm_bounds"][1])
    grid_shape = tuple(int(x) for x in ckpt["spatial_shape"])
    x = params_to_grid_tensor(inp, rpm_lo=rpm_lo, rpm_hi=rpm_hi, grid_shape=grid_shape).unsqueeze(0).to(device)
    mean = ckpt["norm"]["mean"].to(device)
    std = ckpt["norm"]["std"].to(device)
    with torch.no_grad():
        yn = model(x)[0]
        y = yn * std + mean
    arr = y.cpu().numpy()
    vel = arr[:3]
    pr = arr[3, ...]
    pools = pooled_features({"velocity": vel, "pressure": pr})
    return {"mean_speed": float(pools[0]), "mean_pressure": float(pools[1]), "max_speed": float(pools[2])}

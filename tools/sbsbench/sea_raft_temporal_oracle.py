#!/usr/bin/env python3
"""Optional SEA-RAFT temporal artifact oracle for rendered SBS sequences.

The Princeton-VL implementation and checkpoint stay outside this repository.  This wrapper
loads an official checkout at runtime, estimates bidirectional source flow, rejects cuts,
forward/backward-inconsistent pixels, disocclusions, and high-uncertainty matches, then measures
only stereo-output change left after the matched mono-source change has been removed.

The output is deliberately eval-only.  A single-frame DA-V2 augmentation model cannot observe
sequence history, so none of these values may be exported as a per-frame training label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


SCHEMA = 2
DEFAULT_MIN_SUPPORT_PCT = 2.0
DEFAULT_MIN_SUPPORT_PIXELS = 256
DEFAULT_SOURCE_RESIDUAL_LIMIT = 20.0 / 255.0
DEFAULT_CUT_RESIDUAL = 0.18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="ascii").strip()
    if not value.startswith("ref: "):
        return value or None
    ref = repo / ".git" / value[5:]
    return ref.read_text(encoding="ascii").strip() if ref.is_file() else None


def _rgb_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(f"expected HxWx3 image, got {value.shape}")
    value = value[..., :3].astype(np.float32)
    if value.size and float(np.nanmax(value)) > 1.5:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def _luma(image: np.ndarray) -> np.ndarray:
    rgb = _rgb_float(image)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float32)


def _split_sbs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = _rgb_float(image)
    if value.shape[1] % 2:
        raise ValueError(f"SBS image width must be even, got {value.shape[1]}")
    width = value.shape[1] // 2
    return value[:, :width], value[:, width:]


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    value = (_rgb_float(image) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(value).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _sample(array: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample an earlier-frame array at current coordinates plus current->earlier flow."""
    values = np.asarray(array, dtype=np.float32)
    vectors = np.asarray(flow, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[2] != 2 or vectors.shape[:2] != values.shape[:2]:
        raise ValueError(f"flow {vectors.shape} cannot sample {values.shape}")
    height, width = values.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    sx = xx + vectors[..., 0]
    sy = yy + vectors[..., 1]
    valid = (np.isfinite(sx) & np.isfinite(sy) & (sx >= 0.0) & (sx <= width - 1)
             & (sy >= 0.0) & (sy <= height - 1))
    safe_x = np.clip(np.nan_to_num(sx), 0.0, width - 1)
    safe_y = np.clip(np.nan_to_num(sy), 0.0, height - 1)
    x0 = np.floor(safe_x).astype(np.int32)
    y0 = np.floor(safe_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = safe_x - x0
    wy = safe_y - y0
    if values.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    result = ((1.0 - wy) * ((1.0 - wx) * values[y0, x0] + wx * values[y0, x1])
              + wy * ((1.0 - wx) * values[y1, x0] + wx * values[y1, x1]))
    return result.astype(np.float32), valid


def _erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce([
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3) for dx in range(3)
        ])
    return result


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce([
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3) for dx in range(3)
        ])
    return result


def _gradient_vector(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(value)
    gy = np.zeros_like(value)
    gx[:, 1:-1] = 0.5 * (value[:, 2:] - value[:, :-2])
    gy[1:-1] = 0.5 * (value[2:] - value[:-2])
    return np.stack((gx, gy), axis=2).astype(np.float32)


def _gradient_magnitude(vectors: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 2:
        raise ValueError(f"expected HxWx2 gradient vectors, got {value.shape}")
    return np.linalg.norm(value, axis=2).astype(np.float32)


def _laplacian(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    padded = np.pad(value, 1, mode="edge")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
            - 4.0 * value).astype(np.float32)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _histogram_distance(before: np.ndarray, after: np.ndarray) -> float:
    first, _ = np.histogram(before, bins=32, range=(0.0, 1.0), density=False)
    second, _ = np.histogram(after, bins=32, range=(0.0, 1.0), density=False)
    first = first.astype(np.float64) / max(1, int(first.sum()))
    second = second.astype(np.float64) / max(1, int(second.sum()))
    return float(0.5 * np.abs(first - second).sum())


def flow_consistency_mask(forward: np.ndarray, backward: np.ndarray,
                          forward_uncertainty: np.ndarray | None = None,
                          backward_uncertainty: np.ndarray | None = None
                          ) -> tuple[np.ndarray, dict]:
    """Return current-grid non-occluded support using the standard RAFT cycle criterion."""
    forward = np.asarray(forward, dtype=np.float32)
    backward = np.asarray(backward, dtype=np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError(f"incompatible forward/backward flow: {forward.shape}, {backward.shape}")
    sampled_forward, in_bounds = _sample(forward, backward)
    cycle = np.linalg.norm(sampled_forward + backward, axis=2)
    magnitude_sq = np.sum(sampled_forward ** 2 + backward ** 2, axis=2)
    # Widely used forward/backward occlusion test: an absolute allowance plus a term that scales
    # with motion. It is intentionally independent from image residual, which is measured later.
    consistent = in_bounds & (cycle ** 2 <= 0.5 + 0.01 * magnitude_sq)
    uncertainty = None
    if forward_uncertainty is not None and backward_uncertainty is not None:
        forward_u, uncertainty_valid = _sample(
            np.asarray(forward_uncertainty, dtype=np.float32), backward)
        uncertainty = np.maximum(forward_u, np.asarray(backward_uncertainty, dtype=np.float32))
        finite = np.isfinite(uncertainty) & uncertainty_valid
        if np.any(consistent & finite):
            # The mixture-Laplace scale is not calibrated as a probability. Remove only its
            # worst decile; cycle consistency and source photometry remain the primary tests.
            limit = max(2.0, _percentile(uncertainty[consistent & finite], 90))
            consistent &= finite & (uncertainty <= limit)
        else:
            consistent &= False
    details = {
        "cycle_p95_px": _percentile(cycle[in_bounds], 95) if np.any(in_bounds) else None,
        "uncertainty_p95_px": (_percentile(uncertainty[consistent], 95)
                               if uncertainty is not None and np.any(consistent) else None),
    }
    return _erode(consistent, 1), details


def _eye_residual_maps(previous: np.ndarray, current: np.ndarray, warped_source: np.ndarray,
                       current_source: np.ndarray, backward_flow: np.ndarray,
                       reliable: np.ndarray) -> dict[str, np.ndarray]:
    old = _luma(previous)
    now = _luma(current)
    old_warped, eye_valid = _sample(old, backward_flow)
    source_before = _luma(warped_source)
    source_now = _luma(current_source)

    # Preserve the sign of each registered temporal change until the source change has been
    # subtracted.  Subtracting magnitudes instead would call equal-magnitude, opposite-direction
    # changes perfect: an eye that darkens while the source brightens (or moves an edge in the
    # opposite direction) would report zero residual.
    luma = np.abs((now - old_warped) - (source_now - source_before))
    grad_now = _gradient_vector(now)
    grad_old = _gradient_vector(old_warped)
    grad_source_now = _gradient_vector(source_now)
    grad_source_old = _gradient_vector(source_before)
    gradient = _gradient_magnitude(
        (grad_now - grad_old) - (grad_source_now - grad_source_old))
    log_now = _laplacian(now)
    log_old = _laplacian(old_warped)
    log_source_now = _laplacian(source_now)
    log_source_old = _laplacian(source_before)
    log = np.abs((log_now - log_old) - (log_source_now - log_source_old))
    edge = _dilate(np.maximum.reduce((
        _gradient_magnitude(grad_now),
        _gradient_magnitude(grad_old),
        _gradient_magnitude(grad_source_now),
        _gradient_magnitude(grad_source_old),
    )) >= 4.0 / 255.0, 2)
    valid = reliable & eye_valid
    return {"luma": luma, "gradient": gradient, "log": log,
            "edge": edge, "valid": valid}


def _motion_mismatch(source_backward: np.ndarray, eye_forward: np.ndarray | None,
                     eye_backward: np.ndarray | None, support: np.ndarray,
                     forward_uncertainty: np.ndarray | None = None,
                     backward_uncertainty: np.ndarray | None = None
                     ) -> tuple[float | None, float | None]:
    if eye_forward is None or eye_backward is None:
        return None, None
    eye = np.asarray(eye_backward, dtype=np.float32)
    if eye.shape != source_backward.shape or np.asarray(eye_forward).shape != eye.shape:
        return None, None
    eye_support, _ = flow_consistency_mask(
        eye_forward, eye_backward, forward_uncertainty, backward_uncertainty)
    finite = support & eye_support & np.isfinite(eye).all(axis=2)
    support_pct = float(np.mean(finite) * 100.0)
    if np.count_nonzero(finite) < 64:
        return None, support_pct
    return (_percentile(np.linalg.norm(eye - source_backward, axis=2)[finite], 95),
            support_pct)


def temporal_artifact_metrics(
        previous_source: np.ndarray, current_source: np.ndarray,
        previous_left: np.ndarray, current_left: np.ndarray,
        previous_right: np.ndarray, current_right: np.ndarray,
        source_forward: np.ndarray, source_backward: np.ndarray,
        source_forward_uncertainty: np.ndarray | None = None,
        source_backward_uncertainty: np.ndarray | None = None,
        left_forward: np.ndarray | None = None, left_backward: np.ndarray | None = None,
        right_forward: np.ndarray | None = None, right_backward: np.ndarray | None = None,
        left_forward_uncertainty: np.ndarray | None = None,
        left_backward_uncertainty: np.ndarray | None = None,
        right_forward_uncertainty: np.ndarray | None = None,
        right_backward_uncertainty: np.ndarray | None = None,
        min_support_pct: float = DEFAULT_MIN_SUPPORT_PCT,
        min_support_pixels: int = DEFAULT_MIN_SUPPORT_PIXELS,
        source_residual_limit: float = DEFAULT_SOURCE_RESIDUAL_LIMIT,
        return_maps: bool = False) -> dict:
    """Measure flow-compensated stereo-only temporal artifacts for one adjacent frame pair."""
    now_source = _rgb_float(current_source)
    old_source = _rgb_float(previous_source)
    height, width = now_source.shape[:2]
    images = (old_source, _rgb_float(previous_left), _rgb_float(current_left),
              _rgb_float(previous_right), _rgb_float(current_right))
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("source and both SBS eyes must share one evaluation geometry")
    source_backward = np.asarray(source_backward, dtype=np.float32)
    source_forward = np.asarray(source_forward, dtype=np.float32)
    if source_backward.shape != (height, width, 2):
        raise ValueError("source flow must use the evaluation image geometry")

    cycle_support, flow_details = flow_consistency_mask(
        source_forward, source_backward,
        source_forward_uncertainty, source_backward_uncertainty)
    warped_source, source_in_bounds = _sample(old_source, source_backward)
    before_luma = _luma(warped_source)
    now_luma = _luma(now_source)
    source_residual = np.abs(now_luma - before_luma)
    histogram_distance = _histogram_distance(_luma(old_source), now_luma)

    geometric = cycle_support & source_in_bounds
    geometric_pct = float(np.mean(geometric) * 100.0)
    residual_p50 = (_percentile(source_residual[geometric], 50)
                    if np.any(geometric) else _percentile(np.abs(now_luma - _luma(old_source)), 50))
    # A cut must not be converted into a giant ghost score. Require both appearance disagreement
    # and histogram replacement so fast camera motion does not trigger merely from large flow.
    is_cut = residual_p50 >= DEFAULT_CUT_RESIDUAL and histogram_distance >= 0.22
    base = {
        "status": "cut" if is_cut else "abstained",
        "eval_only": True,
        "training_label_eligible": False,
        "reason": "scene_cut" if is_cut else "insufficient_reliable_flow_support",
        "sea_flow_geometric_support_pct": geometric_pct,
        "sea_source_residual_p50": float(residual_p50 * 255.0),
        "sea_source_histogram_distance": histogram_distance,
        "sea_flow_cycle_p95_px": flow_details["cycle_p95_px"],
        "sea_flow_uncertainty_p95_px": flow_details["uncertainty_p95_px"],
    }
    if is_cut:
        return base

    reliable = geometric & (source_residual <= float(source_residual_limit))
    support_pixels = int(np.count_nonzero(reliable))
    support_pct = support_pixels / float(height * width) * 100.0
    base["sea_flow_support_pct"] = support_pct
    base["support_pixels"] = support_pixels
    if support_pixels < max(int(min_support_pixels),
                            int(math.ceil(height * width * min_support_pct / 100.0))):
        return base

    left = _eye_residual_maps(previous_left, current_left, warped_source, now_source,
                              source_backward, reliable)
    right = _eye_residual_maps(previous_right, current_right, warped_source, now_source,
                               source_backward, reliable)
    valid = left["valid"] & right["valid"]
    edge_support = valid & (left["edge"] | right["edge"])
    if np.count_nonzero(edge_support) < max(64, int(0.001 * height * width)):
        base["reason"] = "insufficient_temporal_edge_support"
        base["sea_flow_edge_support_pct"] = float(np.mean(edge_support) * 100.0)
        return base

    luma_excess = np.maximum(left["luma"], right["luma"])
    gradient_excess = np.maximum(left["gradient"], right["gradient"])
    log_excess = np.maximum(left["log"], right["log"])
    edge_ghost = np.maximum(gradient_excess, 0.5 * log_excess)
    source_motion = np.linalg.norm(source_backward, axis=2)
    static = valid & (source_motion <= 0.25) & (source_residual <= 2.0 / 255.0)

    left_mismatch, left_motion_support = _motion_mismatch(
        source_backward, left_forward, left_backward, valid,
        left_forward_uncertainty, left_backward_uncertainty)
    right_mismatch, right_motion_support = _motion_mismatch(
        source_backward, right_forward, right_backward, valid,
        right_forward_uncertainty, right_backward_uncertainty)
    result = dict(base)
    result.update({
        "status": "ok",
        "reason": None,
        "sea_flow_support_pct": float(np.mean(valid) * 100.0),
        "sea_flow_edge_support_pct": float(np.mean(edge_support) * 100.0),
        "sea_flow_static_support_pct": float(np.mean(static) * 100.0),
        "sea_flow_source_motion_p95_px": _percentile(source_motion[valid], 95),
        "sea_flow_flicker_p95": _percentile(luma_excess[valid] * 255.0, 95),
        "sea_flow_edge_ghost_p95": _percentile(edge_ghost[edge_support] * 255.0, 95),
        "sea_flow_gradient_ghost_p95": _percentile(
            gradient_excess[edge_support] * 255.0, 95),
        "sea_flow_log_ghost_p95": _percentile(log_excess[edge_support] * 255.0, 95),
        "sea_static_jitter_p95": (_percentile(luma_excess[static] * 255.0, 95)
                                  if np.count_nonzero(static) >= 64 else None),
        "sea_left_motion_mismatch_p95_px": left_mismatch,
        "sea_right_motion_mismatch_p95_px": right_mismatch,
        "sea_left_motion_support_pct": left_motion_support,
        "sea_right_motion_support_pct": right_motion_support,
    })
    if return_maps:
        result["_maps"] = {
            "support": valid,
            "edge_support": edge_support,
            "flicker": luma_excess,
            "edge_ghost": edge_ghost,
        }
    return result


class SeaRaftModel:
    """Thin adapter around an external official SEA-RAFT checkout."""

    def __init__(self, repo: Path, checkpoint: Path, config: Path, device: str | None = None):
        self.repo = repo.resolve()
        self.checkpoint = checkpoint.resolve()
        self.config = config.resolve()
        if not (self.repo / "core" / "raft.py").is_file():
            raise RuntimeError(f"not a SEA-RAFT checkout: {self.repo}")
        if not self.config.is_file():
            raise RuntimeError(f"SEA-RAFT config missing: {self.config}")
        if not self.checkpoint.exists():
            raise RuntimeError(f"SEA-RAFT checkpoint missing: {self.checkpoint}")
        import torch
        from types import SimpleNamespace

        self.torch = torch
        args = SimpleNamespace(**json.loads(self.config.read_text(encoding="utf-8")))
        sys.path.insert(0, str(self.repo / "core"))
        sys.path.insert(0, str(self.repo))
        module_spec = importlib.util.spec_from_file_location(
            "apollo_external_sea_raft", self.repo / "core" / "raft.py")
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("cannot load external SEA-RAFT module")
        raft_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(raft_module)
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The official constructor initializes from ImageNet before loading the complete optical-
        # flow checkpoint. Avoid an otherwise surprising network download: every initialized
        # tensor is immediately replaced by the official SEA-RAFT state below.
        import torchvision.models as vision_models
        original_resnet18 = vision_models.resnet18
        original_resnet34 = vision_models.resnet34
        vision_models.resnet18 = lambda *unused_args, **unused_kwargs: original_resnet18(
            weights=None)
        vision_models.resnet34 = lambda *unused_args, **unused_kwargs: original_resnet34(
            weights=None)
        try:
            model = raft_module.RAFT(args)
        finally:
            vision_models.resnet18 = original_resnet18
            vision_models.resnet34 = original_resnet34
        if self.checkpoint.is_dir():
            from safetensors.torch import load_file
            state = load_file(str(self.checkpoint / "model.safetensors"), device="cpu")
        else:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        # The published safetensors snapshot predates BatchNorm layers in four downsample paths.
        # Their constructor values are exact identity transforms. Match the official load_ckpt
        # (strict=False), but fail closed if a learned tensor outside that known ABI gap is
        # absent.
        allowed_missing = re.compile(
            r"^(cnet|fnet)\.layer[23]\.0\.downsample\.1\."
            r"(weight|bias|running_mean|running_var)$")
        unexpected_missing = [key for key in incompatible.missing_keys
                              if not allowed_missing.match(key)]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "SEA-RAFT checkpoint/model mismatch: "
                f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}")
        self.model = model.to(target).requires_grad_(False).eval()
        self.args = args
        self.device = target

    def infer(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        import torch.nn.functional as functional

        one = torch.from_numpy((_rgb_float(first) * 255.0).transpose(2, 0, 1))[None].to(
            self.device)
        two = torch.from_numpy((_rgb_float(second) * 255.0).transpose(2, 0, 1))[None].to(
            self.device)
        scale = int(getattr(self.args, "scale", 0))
        with torch.inference_mode():
            scaled_one = functional.interpolate(
                one, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
            scaled_two = functional.interpolate(
                two, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
            output = self.model(scaled_one, scaled_two,
                                iters=int(self.args.iters), test_mode=True)
            flow = output["flow"][-1]
            info = output["info"][-1]
            if scale:
                factor = 0.5 ** scale
                flow = functional.interpolate(
                    flow, size=one.shape[-2:], mode="bilinear", align_corners=False) * factor
                info = functional.interpolate(info, size=one.shape[-2:], mode="area")
            logits = info[:, :2].softmax(dim=1)
            raw = info[:, 2:]
            log_b = torch.empty_like(raw)
            log_b[:, 0] = torch.clamp(
                raw[:, 0], min=0.0, max=float(getattr(self.args, "var_max", 10.0)))
            log_b[:, 1] = torch.clamp(
                raw[:, 1], min=float(getattr(self.args, "var_min", 0.0)), max=0.0)
            uncertainty = torch.sum(logits * torch.exp(log_b), dim=1)
        return (flow[0].permute(1, 2, 0).float().cpu().numpy(),
                uncertainty[0].float().cpu().numpy())


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"file name has no numeric frame id: {path}")
    return int(match.group(1))


def _indexed_files(directory: Path, pattern: str) -> dict[int, Path]:
    return {_frame_id(path): path for path in sorted(directory.glob(pattern))}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _colorize(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    finite = support & np.isfinite(data)
    normalized = np.zeros_like(data)
    if np.any(finite):
        limit = max(1e-6, _percentile(data[finite], 99))
        normalized[finite] = np.clip(data[finite] / limit, 0.0, 1.0)
    color = np.zeros(data.shape + (3,), dtype=np.uint8)
    color[..., 0] = (255.0 * normalized).astype(np.uint8)
    color[..., 1] = (64.0 * np.sqrt(normalized)).astype(np.uint8)
    color[~support] = np.array((12, 16, 22), dtype=np.uint8)
    return color


def save_evidence(path: Path, source: np.ndarray, maps: dict[str, np.ndarray]) -> None:
    source_rgb = (_rgb_float(source) * 255.0 + 0.5).astype(np.uint8)
    support = np.repeat((maps["support"] * 255)[..., None], 3, axis=2).astype(np.uint8)
    ghost = _colorize(maps["edge_ghost"], maps["edge_support"])
    flicker = _colorize(maps["flicker"], maps["support"])
    canvas = np.concatenate((source_rgb, support, ghost, flicker), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def evaluate_sequence(source_dir: Path, sbs_dir: Path, model: SeaRaftModel,
                      source_pattern: str = "frame_*.png", sbs_pattern: str = "sbs_*.png",
                      evidence_dir: Path | None = None, eye_flow: bool = True) -> dict:
    sources = _indexed_files(source_dir, source_pattern)
    outputs = _indexed_files(sbs_dir, sbs_pattern)
    frame_ids = sorted(set(sources) & set(outputs))
    if len(frame_ids) < 2:
        raise RuntimeError("need at least two matching source/SBS frame ids")
    if set(frame_ids) != set(sources) or set(frame_ids) != set(outputs):
        raise RuntimeError("source/SBS frame ids do not match exactly")

    rows = []
    for previous_id, current_id in zip(frame_ids, frame_ids[1:]):
        previous_source = _load_rgb(sources[previous_id])
        current_source = _load_rgb(sources[current_id])
        previous_left, previous_right = _split_sbs(_load_rgb(outputs[previous_id]))
        current_left, current_right = _split_sbs(_load_rgb(outputs[current_id]))
        height, width = current_left.shape[:2]
        previous_source = _resize_rgb(previous_source, width, height)
        current_source = _resize_rgb(current_source, width, height)

        source_forward, forward_uncertainty = model.infer(previous_source, current_source)
        source_backward, backward_uncertainty = model.infer(current_source, previous_source)
        left_forward = left_backward = right_forward = right_backward = None
        left_forward_u = left_backward_u = right_forward_u = right_backward_u = None
        if eye_flow:
            left_forward, left_forward_u = model.infer(previous_left, current_left)
            left_backward, left_backward_u = model.infer(current_left, previous_left)
            right_forward, right_forward_u = model.infer(previous_right, current_right)
            right_backward, right_backward_u = model.infer(current_right, previous_right)
        result = temporal_artifact_metrics(
            previous_source, current_source,
            previous_left, current_left, previous_right, current_right,
            source_forward, source_backward, forward_uncertainty, backward_uncertainty,
            left_forward, left_backward, right_forward, right_backward,
            left_forward_u, left_backward_u, right_forward_u, right_backward_u,
            return_maps=evidence_dir is not None)
        maps = result.pop("_maps", None)
        result["previous_frame"] = previous_id
        result["frame"] = current_id
        rows.append(result)
        if evidence_dir is not None and maps is not None:
            save_evidence(evidence_dir / f"temporal_{current_id:05d}.png", current_source, maps)

    metric_names = (
        "sea_flow_flicker_p95", "sea_flow_edge_ghost_p95",
        "sea_flow_gradient_ghost_p95", "sea_flow_log_ghost_p95",
        "sea_static_jitter_p95", "sea_left_motion_mismatch_p95_px",
        "sea_right_motion_mismatch_p95_px")
    aggregate = {}
    accepted = [row for row in rows if row["status"] == "ok"]
    for metric in metric_names:
        values = [float(row[metric]) for row in accepted
                  if row.get(metric) is not None and np.isfinite(row[metric])]
        if values:
            aggregate[f"{metric}_p50"] = float(np.percentile(values, 50))
            aggregate[f"{metric}_p95"] = float(np.percentile(values, 95))
    return {
        "schema": SCHEMA,
        "oracle": "SEA-RAFT",
        "role": "optional_eval_only_temporal_artifact_oracle",
        "training_label_eligible": False,
        "flow_model": {
            "repo_revision": _git_revision(model.repo),
            "config": str(model.config),
            "config_sha256": _sha256(model.config),
            "checkpoint": str(model.checkpoint),
            "checkpoint_sha256": (_sha256(model.checkpoint)
                                  if model.checkpoint.is_file()
                                  else _sha256(model.checkpoint / "model.safetensors")),
            "device": model.device,
        },
        "pairs_total": len(rows),
        "pairs_measured": len(accepted),
        "pairs_cut": sum(row["status"] == "cut" for row in rows),
        "pairs_abstained": sum(row["status"] == "abstained" for row in rows),
        "aggregate": aggregate,
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--sbs-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path,
                        help="external official Princeton-VL/SEA-RAFT checkout")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="official .pth file or local Hugging Face snapshot directory")
    parser.add_argument("--config", type=Path,
                        help="SEA-RAFT JSON config (default: repo Spring-M config)")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--source-pattern", default="frame_*.png")
    parser.add_argument("--sbs-pattern", default="sbs_*.png")
    parser.add_argument("--source-only-flow", action="store_true",
                        help="skip diagnostic per-eye motion estimates")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = args.config or args.repo / "config" / "eval" / "spring-M.json"
    model = SeaRaftModel(args.repo, args.checkpoint, config, args.device)
    result = evaluate_sequence(
        args.source_dir, args.sbs_dir, model, args.source_pattern, args.sbs_pattern,
        args.evidence_dir, not args.source_only_flow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

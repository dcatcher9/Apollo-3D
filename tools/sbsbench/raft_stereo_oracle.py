#!/usr/bin/env python3
"""Optional RAFT-Stereo correspondence oracle for rendered Apollo SBS frames.

The Princeton-VL implementation and checkpoints remain outside this repository.  This wrapper
loads an official checkout at runtime, estimates left-to-right horizontal displacement, and uses
a flipped/swapped second inference for left-right consistency.  RAFT-Stereo has no confidence
head and explicitly forces its vertical flow to zero, so this module does not pretend otherwise:
validity comes from finite/in-bounds/left-right-consistent matches, while a small independent
vertical patch search measures epipolar misalignment.

All outputs are experimental diagnostics.  They are not gates or training labels until the
controlled-corruption and benign-invariance qualification suite establishes their behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from PIL import Image


SCHEMA = 1
MIN_SUPPORT_PIXELS = 64
DEFAULT_LR_LIMIT_PX = 1.0
DEFAULT_MIN_SUPPORT_PCT = 0.5
DEFAULT_MIN_TEXTURE_COVERAGE_PCT = 20.0
DEFAULT_TEXTURE_THRESHOLD = 2.0 / 255.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str | None:
    head_path = repo / ".git" / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head or None
    ref_path = repo / ".git" / head[5:]
    if ref_path.is_file():
        return ref_path.read_text(encoding="ascii").strip() or None
    return None


def _split_sbs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[1] % 2:
        raise ValueError(f"expected even-width RGB SBS image, got {image.shape}")
    width = image.shape[1] // 2
    return image[:, :width, :3], image[:, width:, :3]


def _luma(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]).astype(np.float32)


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    size = 2 * radius + 1
    summed = (integral[size:, size:] - integral[:-size, size:]
              - integral[size:, :-size] + integral[:-size, :-size])
    return (summed / float(size * size)).astype(np.float32)


def _horizontal_structure(gray: np.ndarray, radius: int = 2) -> np.ndarray:
    delta = np.zeros_like(gray, dtype=np.float32)
    adjacent = np.abs(gray[:, 1:] - gray[:, :-1])
    delta[:, 1:] = np.maximum(delta[:, 1:], adjacent)
    delta[:, :-1] = np.maximum(delta[:, :-1], adjacent)
    return _box_mean(delta, radius)


def _sample_horizontal(field: np.ndarray, displacement: np.ndarray,
                       vertical_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``field[y + vertical_offset, x + displacement[y,x]]`` bilinearly in x."""
    field = np.asarray(field, dtype=np.float32)
    displacement = np.asarray(displacement, dtype=np.float32)
    if field.shape != displacement.shape:
        raise ValueError(f"field/displacement shape mismatch: {field.shape} vs {displacement.shape}")
    height, width = field.shape
    xx = np.arange(width, dtype=np.float32)[None, :]
    sx = xx + displacement
    sy = np.arange(height, dtype=np.int32)[:, None] + int(vertical_offset)
    valid = (np.isfinite(sx) & (sx >= 0.0) & (sx <= width - 1)
             & (sy >= 0) & (sy < height))
    safe_x = np.clip(np.nan_to_num(sx, nan=0.0), 0.0, width - 1)
    x0 = np.floor(safe_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    weight = safe_x - x0
    safe_y = np.clip(sy, 0, height - 1)
    rows = np.broadcast_to(safe_y, (height, width))
    sampled = (field[rows, x0] * (1.0 - weight) + field[rows, x1] * weight)
    return sampled.astype(np.float32), valid


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _vertical_correspondence(left: np.ndarray, right: np.ndarray, disparity: np.ndarray,
                             support: np.ndarray, max_offset: int = 4) -> dict:
    left_grad = np.zeros_like(left)
    right_grad = np.zeros_like(right)
    left_grad[:, 1:] = left[:, 1:] - left[:, :-1]
    right_grad[:, 1:] = right[:, 1:] - right[:, :-1]
    offsets = [0]
    for amount in range(1, max(0, int(max_offset)) + 1):
        offsets.extend((-amount, amount))

    costs = []
    residuals = []
    for offset in offsets:
        sampled, valid = _sample_horizontal(right, disparity, offset)
        sampled_grad, gradient_valid = _sample_horizontal(right_grad, disparity, offset)
        residual = np.abs(left - sampled)
        cost = _box_mean(residual + 0.35 * np.abs(left_grad - sampled_grad), 2)
        valid &= gradient_valid
        cost = np.where(valid, cost + abs(offset) * 1e-6, np.inf)
        costs.append(cost)
        residuals.append(residual)

    stack = np.stack(costs)
    best_index = np.argmin(stack, axis=0)
    base = stack[0]
    best = np.take_along_axis(stack, best_index[None], axis=0)[0]
    if stack.shape[0] > 1:
        second_best = np.partition(stack, 1, axis=0)[1]
    else:
        second_best = np.full_like(best, np.inf)
    # A non-zero vertical estimate must win by both an absolute and a relative margin.  This
    # prevents texture noise or a flat-cost tie from being mislabeled as epipolar misalignment.
    finite_cost = np.isfinite(base) & np.isfinite(best)
    improvement = np.zeros_like(base)
    np.subtract(base, best, out=improvement, where=finite_cost)
    separation = np.zeros_like(best)
    finite_pair = np.isfinite(best) & np.isfinite(second_best)
    np.subtract(second_best, best, out=separation, where=finite_pair)
    unique = finite_pair & (separation >= 1.0 / 255.0) & (best <= 0.90 * second_best)
    meaningful = ((best_index != 0) & finite_cost & unique
                  & (improvement >= 1.5 / 255.0) & (best <= 0.90 * base))
    chosen = np.zeros(left.shape, dtype=np.float32)
    offset_values = np.asarray(offsets, dtype=np.float32)
    chosen[meaningful] = offset_values[best_index[meaningful]]
    common = support & np.isfinite(base)
    if not common.any():
        return {}
    vertical = np.abs(chosen[common])
    raw_residual = residuals[0][common] * 255.0
    best_residual_stack = np.stack(residuals)
    best_residual = np.take_along_axis(
        best_residual_stack, best_index[None], axis=0)[0][common] * 255.0
    vertical_p50 = _percentile(vertical, 50)
    vertical_p95 = _percentile(vertical, 95)
    vertical_p99 = _percentile(vertical, 99)
    return {
        "raft_vertical_abs_p50_px": vertical_p50,
        "raft_vertical_abs_p95_px": vertical_p95,
        "raft_vertical_abs_p99_px": vertical_p99,
        "raft_vertical_abs_p95_pct": vertical_p95 / left.shape[0] * 100.0,
        "raft_vertical_nonzero_pct": float(np.mean(vertical >= 0.5) * 100.0),
        "raft_correspondence_residual_p50": _percentile(raw_residual, 50),
        "raft_correspondence_residual_p95": _percentile(raw_residual, 95),
        "raft_vertical_aligned_residual_p95": _percentile(best_residual, 95),
    }


def exact_left_to_right_disparity(mapping: np.ndarray, shape: dict,
                                  monotonic_tolerance_pct: float = 1.0
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Invert the exact right-eye source-U map onto left-eye output coordinates.

    Only rows whose rendered right-eye map is overwhelmingly monotonic are used.  Clamped,
    folded, repeated, or off-source mappings are intentionally invalid: in those regions the
    rendered image has no unique correspondence and Apollo's exact topology metrics are the
    authoritative diagnosis.
    """
    mapping = np.asarray(mapping, dtype=np.float32)
    height = int(shape.get("height", 0))
    eye_width = int(shape.get("eye_width", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if mapping.shape != (height, 2 * eye_width):
        raise ValueError(f"warp map {mapping.shape} does not match {(height, 2 * eye_width)}")
    if min(height, eye_width, scale_x, scale_y) <= 0:
        raise ValueError("warp-map shape contract is missing positive dimensions/scales")

    left = mapping[:, :eye_width]
    right = mapping[:, eye_width:]
    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / eye_width
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / height
    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    content_x = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    content_y = (output_v >= lo_y) & (output_v <= lo_y + scale_y)
    x_coordinates = np.arange(eye_width, dtype=np.float32)
    disparity = np.full((height, eye_width), np.nan, dtype=np.float32)
    valid = np.zeros((height, eye_width), dtype=bool)

    for y in np.flatnonzero(content_y):
        right_valid = content_x & np.isfinite(right[y]) & (right[y] >= 0.0) & (right[y] <= 1.0)
        indices = np.flatnonzero(right_valid)
        if indices.size < 4:
            continue
        steps = np.diff(right[y, indices])
        bad_step_pct = float(np.mean(steps <= 1e-7) * 100.0)
        if bad_step_pct > monotonic_tolerance_pct:
            continue
        # Drop the rare locally invalid samples, then require the remaining inverse points to be
        # strictly ordered.  This is fail-closed rather than sorting a folded map into plausibility.
        point_good = np.ones(indices.size, dtype=bool)
        point_good[1:] &= steps > 1e-7
        point_good[:-1] &= steps > 1e-7
        indices = indices[point_good]
        if indices.size < 4 or np.any(np.diff(right[y, indices]) <= 0.0):
            continue
        left_valid = content_x & np.isfinite(left[y])
        targets = np.flatnonzero(left_valid)
        target_u = left[y, targets]
        within = ((target_u >= right[y, indices[0]])
                  & (target_u <= right[y, indices[-1]])
                  & (target_u >= 0.0) & (target_u <= 1.0))
        targets = targets[within]
        if not targets.size:
            continue
        matched_x = np.interp(left[y, targets], right[y, indices], x_coordinates[indices])
        disparity[y, targets] = matched_x - x_coordinates[targets]
        valid[y, targets] = True
    return disparity, valid


def correspondence_metrics(left_rgb: np.ndarray, right_rgb: np.ndarray,
                           left_to_right: np.ndarray, right_to_left: np.ndarray,
                           exact_disparity: np.ndarray | None = None,
                           exact_valid: np.ndarray | None = None,
                           lr_limit_px: float = DEFAULT_LR_LIMIT_PX,
                           min_support_pct: float = DEFAULT_MIN_SUPPORT_PCT,
                           min_texture_coverage_pct: float = DEFAULT_MIN_TEXTURE_COVERAGE_PCT,
                           texture_threshold: float = DEFAULT_TEXTURE_THRESHOLD,
                           max_vertical_px: int | None = None) -> dict:
    """Validate signed dense correspondence and optionally compare it to Apollo's exact map."""
    left = _luma(left_rgb)
    right = _luma(right_rgb)
    disparity = np.asarray(left_to_right, dtype=np.float32)
    reverse = np.asarray(right_to_left, dtype=np.float32)
    if not (left.shape == right.shape == disparity.shape == reverse.shape):
        raise ValueError("eye images and correspondence fields must share one HxW shape")
    reverse_at_match, reverse_valid = _sample_horizontal(reverse, disparity)
    _, image_valid = _sample_horizontal(right, disparity)
    lr_error = np.abs(disparity + reverse_at_match)
    finite = np.isfinite(disparity) & np.isfinite(reverse_at_match)
    geometric = (finite & reverse_valid & image_valid & (lr_error <= float(lr_limit_px)))
    texture = _horizontal_structure(left) >= float(texture_threshold)
    support = geometric & texture
    support_count = int(np.count_nonzero(support))
    support_pct = float(support_count / support.size * 100.0)
    result = {
        "status": "ok",
        "role": "diagnostic_experimental",
        "raft_support_pct": support_pct,
        "raft_geometric_support_pct": float(np.mean(geometric) * 100.0),
        "raft_texture_support_pct": float(np.mean(texture) * 100.0),
        "raft_supported_texture_pct": float(
            support_count / max(np.count_nonzero(texture), 1) * 100.0),
    }
    required = max(MIN_SUPPORT_PIXELS,
                   int(math.ceil(support.size * float(min_support_pct) / 100.0)),
                   int(math.ceil(np.count_nonzero(texture)
                                 * float(min_texture_coverage_pct) / 100.0)))
    if support_count < required:
        result.update({
            "status": "abstained",
            "reason": "insufficient_left_right_consistent_textured_support",
            "required_support_pixels": required,
            "support_pixels": support_count,
        })
        return result

    signed = disparity[support]
    result.update({
        # These are raw left-to-right pixel displacements.  A conventional positive stereo
        # disparity appears negative here; retaining the sign makes polarity inversions visible.
        "raft_signed_disparity_p01_px": _percentile(signed, 1),
        "raft_signed_disparity_p50_px": _percentile(signed, 50),
        "raft_signed_disparity_p99_px": _percentile(signed, 99),
        "raft_lr_consistency_p50_px": _percentile(lr_error[support], 50),
        "raft_lr_consistency_p95_px": _percentile(lr_error[support], 95),
    })
    vertical_limit = (min(12, max(2, round(4.0 * left.shape[0] / 720.0)))
                      if max_vertical_px is None else int(max_vertical_px))
    result.update(_vertical_correspondence(
        left, right, disparity, support, max_offset=vertical_limit))

    if exact_disparity is None:
        result["exact_comparison_status"] = "not_available"
        return result
    exact = np.asarray(exact_disparity, dtype=np.float32)
    if exact.shape != disparity.shape:
        raise ValueError(f"exact disparity shape mismatch: {exact.shape} vs {disparity.shape}")
    exact_mask = np.isfinite(exact)
    if exact_valid is not None:
        if np.asarray(exact_valid).shape != exact.shape:
            raise ValueError("exact validity mask shape mismatch")
        exact_mask &= np.asarray(exact_valid, dtype=bool)
    comparison = support & exact_mask
    comparison_count = int(np.count_nonzero(comparison))
    exact_count = int(np.count_nonzero(exact_mask))
    result["raft_exact_coverage_pct"] = float(
        comparison_count / max(exact_count, 1) * 100.0)
    if comparison_count < required:
        result["exact_comparison_status"] = "abstained_insufficient_support"
        return result

    signed_error = disparity[comparison] - exact[comparison]
    absolute_error = np.abs(signed_error)
    result.update({
        "exact_comparison_status": "ok",
        "raft_exact_signed_bias_px": float(np.mean(signed_error)),
        "raft_exact_mae_px": float(np.mean(absolute_error)),
        "raft_exact_p95_px": _percentile(absolute_error, 95),
        "raft_exact_mae_pct": float(np.mean(absolute_error) / left.shape[1] * 100.0),
        "raft_exact_p95_pct": _percentile(absolute_error, 95) / left.shape[1] * 100.0,
    })
    for threshold in (0.5, 1.0, 2.0, 4.0):
        name = str(threshold).replace(".", "_").replace("_0", "")
        result[f"raft_exact_bad_{name}px_pct"] = float(
            np.mean(absolute_error > threshold) * 100.0)
    polarity = comparison & (np.abs(exact) >= 0.25)
    if np.count_nonzero(polarity) >= MIN_SUPPORT_PIXELS:
        result["raft_exact_polarity_agreement_pct"] = float(
            np.mean(np.sign(disparity[polarity]) == np.sign(exact[polarity])) * 100.0)
    return result


def _load_model(repo: Path, checkpoint: Path, device: str, valid_iters: int):
    if not (repo / "core" / "raft_stereo.py").is_file():
        raise RuntimeError(f"not an official RAFT-Stereo checkout: {repo}")
    if not checkpoint.is_file():
        raise RuntimeError(f"RAFT-Stereo checkpoint is missing: {checkpoint}")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "core"))
    import torch
    from core.raft_stereo import RAFTStereo

    options = SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="alt",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=(device.startswith("cuda")),
        valid_iters=int(valid_iters),
    )
    model = RAFTStereo(options)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).requires_grad_(False).eval()
    return model, options


def _resize_for_inference(left: np.ndarray, right: np.ndarray,
                          max_eye_width: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    height, width = left.shape[:2]
    if max_eye_width <= 0 or width <= max_eye_width:
        return left, right, 1.0, 1.0
    scale = max_eye_width / float(width)
    size = (max_eye_width, max(1, round(height * scale)))
    left_small = np.asarray(Image.fromarray(left).resize(size, Image.Resampling.LANCZOS))
    right_small = np.asarray(Image.fromarray(right).resize(size, Image.Resampling.LANCZOS))
    return left_small, right_small, width / size[0], height / size[1]


def _resize_field(field: np.ndarray, shape: tuple[int, int], scale_x: float) -> np.ndarray:
    height, width = shape
    if field.shape == shape:
        return field.astype(np.float32)
    resized = Image.fromarray(field.astype(np.float32), mode="F").resize(
        (width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) * float(scale_x)


def _infer(model, options, left: np.ndarray, right: np.ndarray, device: str) -> np.ndarray:
    import torch
    from core.utils.utils import InputPadder

    left_tensor = torch.from_numpy(np.array(left, copy=True, order="C")).permute(2, 0, 1)
    right_tensor = torch.from_numpy(np.array(right, copy=True, order="C")).permute(2, 0, 1)
    left_tensor = left_tensor.float().unsqueeze(0).to(device)
    right_tensor = right_tensor.float().unsqueeze(0).to(device)
    padder = InputPadder(left_tensor.shape, divis_by=32)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    with torch.inference_mode():
        _, flow = model(left_tensor, right_tensor,
                        iters=options.valid_iters, test_mode=True)
    return padder.unpad(flow)[0, 0].float().cpu().numpy()


def _mapping_for_sbs(path: Path) -> tuple[np.ndarray, dict] | tuple[None, None]:
    suffix = path.stem.removeprefix("sbs_")
    mapping_path = path.parent / f"warp_map_{suffix}.f32"
    shape_path = path.parent / "warp_map_shape.json"
    if not mapping_path.is_file() or not shape_path.is_file():
        return None, None
    shape = json.loads(shape_path.read_text(encoding="utf-8"))
    expected = int(shape["height"]) * int(shape["width"])
    mapping = np.fromfile(mapping_path, dtype="<f4")
    if mapping.size != expected:
        raise RuntimeError(
            f"warp map has {mapping.size} floats, expected {expected}: {mapping_path}")
    return mapping.reshape(int(shape["height"]), int(shape["width"])), shape


def _iter_inputs(inputs: Iterable[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_dir():
            found = sorted(item.glob("sbs_*.png"))
            # Controlled-corruption exports use semantic case names instead of harness frame
            # names.  Fall back to every PNG only when no harness-shaped artifact is present.
            paths.extend(found or sorted(item.glob("*.png")))
        elif item.is_file():
            paths.append(item)
        else:
            raise RuntimeError(f"input does not exist: {item}")
    if not paths:
        raise RuntimeError("no SBS PNG inputs found")
    return paths


def _orientation_is_ambiguous(selected: dict, alternate: dict) -> bool:
    selected_support = float(selected.get("raft_supported_texture_pct", 0.0))
    alternate_support = float(alternate.get("raft_supported_texture_pct", 0.0))
    selected_residual = float(selected.get("raft_correspondence_residual_p50", 255.0))
    alternate_residual = float(alternate.get("raft_correspondence_residual_p50", 255.0))
    return bool(
        selected.get("status") == "ok" and selected_support > 0.0
        and alternate_support >= 0.8 * selected_support
        and alternate_residual <= 1.2 * max(selected_residual, 1e-6))


def evaluate_paths(paths: Iterable[Path], repo: Path, checkpoint: Path,
                   device: str | None = None, valid_iters: int = 32,
                   max_eye_width: int = 0, save_fields: Path | None = None) -> dict:
    import torch

    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, options = _load_model(repo, checkpoint, target, valid_iters)
    frames = []
    if save_fields:
        save_fields.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        left, right = _split_sbs(image)
        infer_left, infer_right, scale_x, _ = _resize_for_inference(
            left, right, int(max_eye_width))
        # Official RAFT-Stereo is trained for conventional negative left-to-right displacement.
        # Apollo can pack either physical eye first.  Evaluate both signs without consulting the
        # exact map: flipping both eyes maps a positive-displacement pair into the learned negative
        # contract.  The candidate with stronger independent left-right consistency wins; an
        # ambiguous choice abstains below instead of silently forcing the exact-map polarity.
        negative_forward_small = _infer(
            model, options, infer_left, infer_right, target)
        positive_forward_small = -np.flip(_infer(
            model, options, np.flip(infer_left, axis=1),
            np.flip(infer_right, axis=1), target), axis=1).copy()
        negative_reverse_small = _infer(
            model, options, infer_right, infer_left, target)
        positive_reverse_small = -np.flip(_infer(
            model, options, np.flip(infer_right, axis=1),
            np.flip(infer_left, axis=1), target), axis=1).copy()
        negative_forward = _resize_field(
            negative_forward_small, left.shape[:2], scale_x)
        positive_forward = _resize_field(
            positive_forward_small, left.shape[:2], scale_x)
        negative_reverse = _resize_field(
            negative_reverse_small, left.shape[:2], scale_x)
        positive_reverse = _resize_field(
            positive_reverse_small, left.shape[:2], scale_x)
        lr_limit = max(DEFAULT_LR_LIMIT_PX, float(scale_x))
        candidates = {
            "negative_left_to_right": (negative_forward, positive_reverse),
            "positive_left_to_right": (positive_forward, negative_reverse),
        }
        candidate_metrics = {
            name: correspondence_metrics(
                left, right, fields[0], fields[1], lr_limit_px=lr_limit)
            for name, fields in candidates.items()
        }

        def candidate_key(item):
            data = item[1]
            residual = float(data.get("raft_correspondence_residual_p50", 255.0))
            return (float(data.get("raft_supported_texture_pct", 0.0)), -residual)

        ranked = sorted(candidate_metrics.items(), key=candidate_key, reverse=True)
        selected_name, selected_preview = ranked[0]
        alternate_name, alternate_preview = ranked[1]
        forward, reverse = candidates[selected_name]
        mapping, shape = _mapping_for_sbs(path)
        exact = exact_valid = None
        if mapping is not None:
            exact, exact_valid = exact_left_to_right_disparity(mapping, shape)
        orientation_ambiguous = _orientation_is_ambiguous(
            selected_preview, alternate_preview)
        metrics = correspondence_metrics(
            left, right, forward, reverse,
            None if orientation_ambiguous else exact,
            None if orientation_ambiguous else exact_valid,
            lr_limit_px=lr_limit)
        metrics.update({
            "raft_selected_orientation": selected_name,
            "raft_alternate_orientation": alternate_name,
            "raft_orientation_support_margin_pct": float(
                selected_preview.get("raft_supported_texture_pct", 0.0)
                - alternate_preview.get("raft_supported_texture_pct", 0.0)),
            "raft_alternate_supported_texture_pct": float(
                alternate_preview.get("raft_supported_texture_pct", 0.0)),
            "raft_lr_consistency_limit_px": lr_limit,
        })
        if orientation_ambiguous:
            metrics["status"] = "abstained"
            metrics["reason"] = "left_to_right_orientation_ambiguous"
            metrics["exact_comparison_status"] = "abstained_orientation_ambiguous"
        frames.append({"path": str(path), "metrics": metrics})
        if save_fields:
            identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
            np.save(save_fields / f"raft_disparity_{identity}.npy", forward)
            np.save(save_fields / f"raft_reverse_disparity_{identity}.npy", reverse)
    return {
        "schema": SCHEMA,
        "oracle": "RAFT-Stereo",
        "role": "diagnostic_experimental",
        "direction": "signed_left_to_right_displacement_px",
        "validity": "finite + in-bounds + flipped/swapped left-right consistency + texture",
        "license": "MIT; model implementation and checkpoint remain external",
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint": str(checkpoint),
        "repo": str(repo),
        "repo_revision": _git_revision(repo),
        "device": target,
        "torch_version": torch.__version__,
        "valid_iters": int(valid_iters),
        "max_eye_width": int(max_eye_width),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optional RAFT-Stereo correspondence diagnostics on SBS PNG files")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="SBS PNGs or directories containing sbs_*.png")
    parser.add_argument("--repo", required=True, type=Path,
                        help="external princeton-vl/RAFT-Stereo checkout")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="official raftstereo-middlebury.pth checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--max-eye-width", type=int, default=0,
                        help="optional inference downscale; 0 preserves native eye resolution")
    parser.add_argument("--save-fields", type=Path,
                        help="optional directory for diagnostic disparity arrays")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        paths = _iter_inputs(args.inputs)
        result = evaluate_paths(
            paths, args.repo, args.checkpoint, args.device, args.valid_iters,
            args.max_eye_width, args.save_fields)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

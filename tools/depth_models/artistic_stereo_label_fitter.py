#!/usr/bin/env python3
"""Deterministic stereo label fitter for Apollo artistic-policy supervision.

The deployable policy sees one RGB image. During dataset construction this
offline algorithm also uses the corresponding authored right eye to recover
sub-pixel binocular disparity, rejects unreliable/occluded pixels, preserves
the source disparity distribution inside Apollo's comfort envelope, and fits
that target to Apollo's exact current full-binocular disparity field.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LabelFitterConfig:
    analysis_width: int = 512
    search_fraction: float = 0.125
    comfort_limit: float = 0.03
    lr_threshold_px: float = 1.5
    photo_threshold: float = 24.0 / 255.0
    texture_threshold: float = 0.012
    min_support: float = 0.03
    min_fit_pixels: int = 512
    max_vertical_disparity_px: float = 1.0


def resize_to_width(image, width, interpolation=cv2.INTER_AREA):
    if image.shape[1] == width:
        return image
    height = max(32, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _as_u8(gray):
    array = np.asarray(gray)
    if array.dtype == np.uint8:
        return array
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _stereo_matcher(width, config):
    extent = max(16, int(np.ceil(width * config.search_fraction / 16.0)) * 16)
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=-extent,
        numDisparities=extent * 2,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher, extent


def estimate_disparity(left, right, config=LabelFitterConfig()):
    """Return left-referenced disparity, confidence, validity and diagnostics.

    Positive disparity means high-near in Apollo's convention: a point at x in
    the left eye is found at x-disparity in the right eye.  A reverse match and
    photometric remap reject occlusions and false correspondences.
    """
    left_u8 = resize_to_width(_as_u8(left), config.analysis_width)
    right_u8 = resize_to_width(_as_u8(right), config.analysis_width)
    if left_u8.shape != right_u8.shape:
        raise ValueError(
            f"stereo eyes have different shapes: {left_u8.shape}/{right_u8.shape}"
        )
    height, width = left_u8.shape
    matcher, extent = _stereo_matcher(width, config)
    disparity = matcher.compute(left_u8, right_u8).astype(np.float32) / 16.0
    reverse = matcher.compute(right_u8, left_u8).astype(np.float32) / 16.0

    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    right_x = xx - disparity
    reverse_at_match = cv2.remap(
        reverse, right_x, yy, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
    )
    right_at_match = cv2.remap(
        right_u8.astype(np.float32) / 255.0, right_x, yy, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
    )
    left_f = left_u8.astype(np.float32) / 255.0
    photo_error = cv2.boxFilter(
        np.abs(left_f - right_at_match), -1, (5, 5),
        borderType=cv2.BORDER_REPLICATE,
    )
    mean = cv2.boxFilter(left_f, -1, (5, 5), borderType=cv2.BORDER_REPLICATE)
    square_mean = cv2.boxFilter(
        left_f * left_f, -1, (5, 5), borderType=cv2.BORDER_REPLICATE
    )
    texture = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    lr_error = np.abs(disparity + reverse_at_match)

    finite = (
        np.isfinite(disparity) & np.isfinite(reverse_at_match)
        & np.isfinite(photo_error)
    )
    inside = (right_x >= 2.0) & (right_x <= width - 3.0)
    in_range = (disparity > -extent + 1.0) & (disparity < extent - 1.0)
    valid = (
        finite & inside & in_range
        & (lr_error <= config.lr_threshold_px)
        & (photo_error <= config.photo_threshold)
        & (texture >= config.texture_threshold)
    )
    photo_conf = np.clip(1.0 - photo_error / config.photo_threshold, 0.0, 1.0)
    lr_conf = np.clip(1.0 - lr_error / config.lr_threshold_px, 0.0, 1.0)
    texture_conf = np.clip(
        (texture - config.texture_threshold) / (config.texture_threshold * 4.0),
        0.0, 1.0,
    )
    confidence = np.where(
        valid, photo_conf * np.sqrt(texture_conf) * lr_conf, 0.0
    ).astype(np.float32)
    diagnostics = {
        "support_pct": float(valid.mean() * 100.0),
        "lr_consistency_pct": float(
            np.mean(lr_error[finite & inside] <= config.lr_threshold_px) * 100.0
        ) if np.any(finite & inside) else 0.0,
        "photometric_p95": float(np.percentile(photo_error[valid], 95) * 255.0)
        if np.any(valid) else None,
        "search_extent_px": extent,
    }
    return disparity, confidence, valid, diagnostics


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, np.float64).ravel()
    weights = np.asarray(weights, np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return None
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    position = np.clip(quantile, 0.0, 1.0) * cumulative[-1]
    return float(values[np.searchsorted(cumulative, position, side="left")])


def robust_positive_scale(depth, disparity, valid, weights, min_pixels=512):
    """Fit disparity=scale*depth without permitting polarity reversal."""
    mask = (
        valid & np.isfinite(depth) & np.isfinite(disparity)
        & np.isfinite(weights) & (weights > 0.0)
    )
    if int(mask.sum()) < min_pixels:
        return None

    def fit(use):
        z = depth[use].astype(np.float64)
        d = disparity[use].astype(np.float64)
        w = weights[use].astype(np.float64)
        denominator = float(np.sum(w * z * z))
        if denominator <= 1e-9:
            return None
        scale = float(np.sum(w * z * d) / denominator)
        if scale <= 1e-7:
            return None
        return scale

    model = fit(mask)
    if model is None:
        return None
    scale = model
    for percentile in (90, 85):
        residual = np.abs(disparity - scale * depth)
        cutoff = weighted_quantile(residual[mask], weights[mask], percentile / 100.0)
        if cutoff is None:
            break
        refined = fit(mask & (residual <= max(cutoff, 1e-5)))
        if refined is not None:
            scale = refined
    residual = np.abs(disparity - scale * depth)
    cutoff = weighted_quantile(residual[mask], weights[mask], 0.90)
    inlier = mask & (residual <= max(cutoff or 0.0, 1e-5))
    return {
        "scale": scale,
        "inlier_pct": float(inlier.sum() / max(mask.sum(), 1) * 100.0),
        "residual_p90": float(cutoff or 0.0),
    }


def frame_analysis(left, right, baseline_disparity, config=LabelFitterConfig(),
                   reference_disparity=None, vertical_disparity=None):
    disparity_px, confidence, valid, diagnostics = estimate_disparity(
        left, right, config
    )
    if reference_disparity is not None:
        reference = np.asarray(reference_disparity, np.float32)
        if reference.ndim != 2:
            raise ValueError(
                f"reference disparity must be HxW, got {reference.shape}"
            )
        source_width = reference.shape[1]
        reference = cv2.resize(
            reference, (disparity_px.shape[1], disparity_px.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ) * (disparity_px.shape[1] / source_width)
        reference_valid = np.isfinite(reference)
        agreement_error = np.abs(disparity_px - reference)
        agreement = reference_valid & (
            agreement_error <= config.lr_threshold_px
        )
        valid &= agreement
        confidence *= np.where(
            agreement,
            np.exp(-agreement_error / max(config.lr_threshold_px, 1e-6)),
            0.0,
        ).astype(np.float32)
        disparity_px = np.where(reference_valid, reference, disparity_px)
        diagnostics.update({
            "reference_disparity_used": True,
            "reference_agreement_pct": float(agreement.mean() * 100.0),
        })
    else:
        diagnostics["reference_disparity_used"] = False
    if vertical_disparity is not None:
        vertical = np.asarray(vertical_disparity, np.float32)
        vertical = cv2.resize(
            vertical, (disparity_px.shape[1], disparity_px.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        # A horizontal-only SBS warp cannot reproduce vertical stereo parallax.
        vertical_valid = np.isfinite(vertical) & (
            np.abs(vertical) <= config.max_vertical_disparity_px
        )
        valid &= vertical_valid
        confidence *= vertical_valid.astype(np.float32)
        diagnostics["vertical_rejection_pct"] = float(
            (1.0 - vertical_valid.mean()) * 100.0
        )
    baseline_disparity = resize_to_width(
        np.asarray(baseline_disparity, np.float32), config.analysis_width,
        interpolation=cv2.INTER_LINEAR,
    )
    if baseline_disparity.shape != disparity_px.shape:
        baseline_disparity = cv2.resize(
            baseline_disparity,
            (disparity_px.shape[1], disparity_px.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    normalized = disparity_px / disparity_px.shape[1]
    low = weighted_quantile(normalized[valid], confidence[valid], 0.01)
    high = weighted_quantile(normalized[valid], confidence[valid], 0.99)
    if low is None or high is None:
        return None
    positive = max(0.0, high)
    negative = max(0.0, -low)
    diagnostics.update({
        "raw_positive_tail_pct": positive * 100.0,
        "raw_negative_tail_pct": negative * 100.0,
    })
    return {
        "baseline_disparity": baseline_disparity,
        "disparity": normalized,
        "confidence_map": confidence,
        "valid": valid,
        "diagnostics": diagnostics,
        "positive_tail": positive,
        "negative_tail": negative,
    }


def _weighted_median(values, weights):
    result = weighted_quantile(values, weights, 0.5)
    if result is None:
        raise RuntimeError("cannot take weighted median of empty values")
    return result


def finalize_shot(analyses, config=LabelFitterConfig(), scale_delta_max=0.50):
    """Apply one comfort scale and latch one global policy over a complete shot."""
    usable = [analysis for analysis in analyses if analysis is not None]
    if not usable:
        return None
    worst_tail = max(
        max(item["positive_tail"], item["negative_tail"]) for item in usable
    )
    comfort_scale = min(
        1.0, config.comfort_limit / max(worst_tail, 1e-9)
    )
    fitted = []
    for item in usable:
        target = item["disparity"] * comfort_scale
        fit = robust_positive_scale(
            item["baseline_disparity"], target, item["valid"],
            item["confidence_map"],
            config.min_fit_pixels,
        )
        if fit is None:
            continue
        support = float(item["valid"].mean())
        if support < config.min_support:
            continue
        fit["frame_weight"] = max(
            1e-6, support * float(item["confidence_map"][item["valid"]].mean())
        )
        fit["analysis"] = item
        fitted.append(fit)
    if not fitted:
        return None

    weights = np.asarray([item["frame_weight"] for item in fitted])
    scales = np.asarray([item["scale"] for item in fitted])
    shot_scale_raw = _weighted_median(scales, weights)
    shot_scale = float(np.clip(
        shot_scale_raw, 1.0 - scale_delta_max, 1.0 + scale_delta_max
    ))
    scale_mad = _weighted_median(np.abs(scales - shot_scale_raw), weights)
    temporal_confidence = float(np.exp(-scale_mad / 0.10))

    outputs = []
    fitted_by_id = {id(item["analysis"]): item for item in fitted}
    for analysis in analyses:
        fit = fitted_by_id.get(id(analysis)) if analysis is not None else None
        if fit is None:
            outputs.append(None)
            continue
        support = float(analysis["valid"].mean())
        mean_match = float(
            analysis["confidence_map"][analysis["valid"]].mean()
        )
        global_confidence = float(np.clip(
            min(1.0, support / 0.20) * np.sqrt(max(mean_match, 0.0))
            * min(1.0, fit["inlier_pct"] / 80.0) * temporal_confidence,
            0.0, 1.0,
        ))
        diagnostics = dict(analysis["diagnostics"])
        diagnostics.update({
            "comfort_scale": comfort_scale,
            "global_inlier_pct": fit["inlier_pct"],
            "global_residual_p90_pct": fit["residual_p90"] * 100.0,
            "shot_scale_raw": float(shot_scale_raw),
            "shot_scale_mad_pct": scale_mad * 100.0,
            "scale_clamped": bool(shot_scale != shot_scale_raw),
            "global_clamped": bool(shot_scale != shot_scale_raw),
        })
        outputs.append({
            "baseline_multiplier": shot_scale,
            "confidence": global_confidence,
            "diagnostics": diagnostics,
        })
    return outputs

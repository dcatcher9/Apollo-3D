#!/usr/bin/env python3
"""Calibrate the diagnostic depth-confidence map against independent visual evidence.

This tool consumes an existing real-pipeline eval run.  It does not rescore the run, alter the
metric schema, or participate in feature decisions.  The candidate confidence map is validated
against final-eye source-relative corruption and, where available, GT depth-boundary placement.
"""
import argparse
import html
import json
import os
import shutil

import numpy as np
from PIL import Image

import sbsbench


MIN_CLASS_PIXELS = 32
MAX_EXAMPLES = 8


def horizontal_max(values, radius):
    """Horizontal maximum filter without wraparound."""
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="constant")
    height, width = values.shape
    return np.maximum.reduce([
        padded[:, radius + offset:radius + offset + width]
        for offset in range(-radius, radius + 1)
    ]).reshape(height, width)


def depth_confidence_map(depth, src_gray, previous_depth=None, previous_src=None,
                         reference_flow=None, reference_valid=None):
    """Diagnostic depth reliability and warp safety near object silhouettes.

    This is deliberately not an eval vote and does not alter rendering. ``model_risk`` describes
    prediction reliability. ``warp_risk`` describes how visible a disocclusion failure would be,
    and ``confidence`` is one minus that warp risk. Every component is returned for visual audit.
    """
    depth = np.asarray(depth, np.float32)
    depth_height, depth_width = depth.shape
    source = sbsbench.resize_to(src_gray, depth_width, depth_height)
    gx_depth = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    depth_threshold = max(float(np.percentile(gx_depth, 99.3)), sbsbench.MIN_DEPTH_STEP)
    edge = gx_depth >= depth_threshold
    zeros = np.zeros_like(depth, np.float32)
    if not edge.any():
        return {
            "confidence": np.ones_like(depth, np.float32), "risk": zeros,
            "band": edge.copy(), "edge": edge, "alignment": zeros,
            "softness": zeros, "edge_strength": zeros, "texture": zeros,
            "temporal": zeros, "temporal_valid": np.zeros_like(edge),
            "model_risk": zeros, "warp_risk": zeros,
        }

    gx_source = np.abs(np.diff(source, axis=1, prepend=source[:, :1]))
    source_threshold = max(float(np.percentile(gx_source, 97.0)), 4.0 / 255.0)
    source_edge = gx_source >= source_threshold
    edge_distance = sbsbench.hdist_x(source_edge, 12)
    hazard_radius = 9
    band = sbsbench.hdilate(edge, hazard_radius)
    alignment_seed = np.where(edge, np.clip(edge_distance / 4.0, 0.0, 1.0), 0.0)
    alignment = horizontal_max(alignment_seed, hazard_radius) * band

    local_gradient = np.zeros_like(gx_depth)
    for offset in range(-2, 3):
        local_gradient += sbsbench._shift_x_edge(gx_depth, offset)
    concentration = np.divide(
        gx_depth, local_gradient + 1e-6, out=np.zeros_like(gx_depth), where=edge)
    softness_seed = np.where(edge, np.clip((0.72 - concentration) / 0.52, 0.0, 1.0), 0.0)
    softness = horizontal_max(softness_seed, hazard_radius) * band
    strength_seed = np.where(edge, np.clip(gx_depth / 0.25, 0.0, 1.0), 0.0)
    edge_strength = horizontal_max(strength_seed, hazard_radius) * band
    gy_source = np.abs(np.diff(source, axis=0, prepend=source[:1, :]))
    texture_seed = np.clip(np.hypot(gx_source, gy_source) / 0.12, 0.0, 1.0)
    texture = horizontal_max(np.where(band, texture_seed, 0.0), hazard_radius) * band

    temporal = zeros.copy()
    temporal_valid = np.zeros_like(edge)
    have_temporal = previous_depth is not None and previous_src is not None
    if have_temporal:
        validation_width = min(256, depth_width)
        validation_height = max(24, round(depth_height * validation_width / depth_width))
        now_source = sbsbench.resize_to(src_gray, validation_width, validation_height)
        old_source = sbsbench.resize_to(previous_src, validation_width, validation_height)
        if reference_flow is not None:
            flow_u, flow_v, flow_valid = sbsbench.resize_forward_flow_to_current(
                reference_flow, reference_valid, validation_width, validation_height)
        else:
            flow_u, flow_v, flow_valid = sbsbench.dense_source_flow(
                previous_src, src_gray, validation_width, validation_height)
        warped_source, source_valid = sbsbench.warp_previous_with_flow(
            old_source, flow_u, flow_v)
        reliable = (flow_valid & source_valid &
                    (np.abs(now_source - warped_source) <= 10.0 / 255.0))
        old_depth = sbsbench.resize_depth(
            previous_depth, validation_width, validation_height)
        now_depth = sbsbench.resize_depth(depth, validation_width, validation_height)
        warped_depth, depth_valid = sbsbench.warp_previous_nearest_with_flow(
            old_depth, flow_u, flow_v)
        valid_small = reliable & depth_valid
        residual = np.where(
            valid_small, np.clip(np.abs(now_depth - warped_depth) / 0.08, 0.0, 1.0), 0.0)
        temporal = sbsbench.resize_depth(residual, depth_width, depth_height).copy()
        temporal_valid = nearest_bool(valid_small, depth_width, depth_height)
        temporal = horizontal_max(temporal * temporal_valid, hazard_radius) * band

    if have_temporal:
        model_risk = 0.50 * alignment + 0.25 * softness + 0.25 * temporal
        warp_risk = (0.55 * texture + 0.25 * softness + 0.15 * edge_strength
                     + 0.05 * temporal)
    else:
        model_risk = 0.67 * alignment + 0.33 * softness
        warp_risk = 0.58 * texture + 0.26 * softness + 0.16 * edge_strength
    model_risk = np.where(
        band, np.clip(model_risk, 0.0, 1.0), 0.0).astype(np.float32)
    warp_risk = np.where(
        band, np.clip(warp_risk, 0.0, 1.0), 0.0).astype(np.float32)
    return {
        "confidence": 1.0 - warp_risk, "risk": warp_risk, "band": band, "edge": edge,
        "alignment": alignment.astype(np.float32), "softness": softness.astype(np.float32),
        "edge_strength": edge_strength.astype(np.float32), "texture": texture.astype(np.float32),
        "temporal": temporal.astype(np.float32), "temporal_valid": temporal_valid,
        "model_risk": model_risk, "warp_risk": warp_risk,
    }


def rank_auc(scores, labels):
    """Tie-aware ROC AUC without scipy; None means only one label class is present."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, bool).ravel()
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, np.float64)
    begin = 0
    while begin < scores.size:
        end = begin + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1) + 1.0
        begin = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def nearest_bool(mask, width, height):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST)) > 127


def output_artifact_map(sbs_path, source, depth):
    """Independent final-eye artifact severity after intended horizontal parallax is free."""
    sbs = sbsbench.load_gray(sbs_path)
    left, right = sbsbench.split_eyes(sbs)
    scale = min(1.0, 256.0 / left.shape[1])
    width = max(32, round(left.shape[1] * scale))
    height = max(24, round(left.shape[0] * scale))
    eyes = [sbsbench.resize_to(eye, width, height) for eye in (left, right)]
    edge = sbsbench.silhouette_edges(depth, width, height, 99.0)
    band = sbsbench.hdilate(edge, 4)
    severity = np.zeros((height, width), np.float32)
    for eye in eyes:
        best, aligned, radius = sbsbench.source_align_map(eye, source)
        valid = np.ones_like(band)
        if width > 2 * radius:
            valid[:, :radius] = False
            valid[:, width - radius:] = False
        eye_ridge = np.clip(eye - sbsbench._hopen(eye, 1), 0.0, None)
        src_ridge = np.clip(aligned - sbsbench._hopen(aligned, 1), 0.0, None)
        halo = np.clip(eye_ridge - src_ridge, 0.0, None) * 255.0
        # Broad source-relative corruption and excess thin ridges are independent of the
        # confidence inputs.  One normalized severity unit is a visibly meaningful violation.
        eye_severity = np.maximum(best * 255.0 / 12.0, halo / 3.0)
        severity = np.maximum(severity, np.where(valid, eye_severity, 0.0))
    return severity, band


def gt_edge_labels(depth, ground_truth, kind):
    """Good/bad predicted boundaries using the evaluator's polarity-safe GT contract."""
    prediction = np.asarray(depth, np.float32)
    if kind in ("metric", "depth"):
        gt, valid = sbsbench.resize_metric_depth(
            ground_truth, prediction.shape[1], prediction.shape[0])
        target = np.zeros_like(gt)
        target[valid] = 1.0 / gt[valid]
    else:
        gt = sbsbench.resize_depth(
            ground_truth, prediction.shape[1], prediction.shape[0])
        valid = np.isfinite(gt) & (gt >= 0.0)
        target = gt
    if int(valid.sum()) < 64:
        return None, None, False, 0, 0
    aligned, _ = sbsbench.align_relative_depth(prediction, target, valid)
    target_range = float(np.percentile(target[valid], 95) - np.percentile(target[valid], 5))
    threshold = max(0.02, target_range * 0.08)
    gx_t = np.abs(np.diff(target, axis=1, prepend=target[:, :1]))
    gy_t = np.abs(np.diff(target, axis=0, prepend=target[:1, :]))
    gx_p = np.abs(np.diff(aligned, axis=1, prepend=aligned[:, :1]))
    gy_p = np.abs(np.diff(aligned, axis=0, prepend=aligned[:1, :]))
    valid_x = valid & np.concatenate((valid[:, :1], valid[:, :-1]), axis=1)
    valid_y = valid & np.concatenate((valid[:1, :], valid[:-1, :]), axis=0)
    gt_edge = (valid_x & (gx_t >= threshold)) | (valid_y & (gy_t >= threshold))
    pred_edge = (valid_x & (gx_p >= threshold)) | (valid_y & (gy_p >= threshold))
    good = pred_edge & sbsbench.dilate2d(gt_edge, 1)
    bad = pred_edge & ~sbsbench.dilate2d(gt_edge, 1)
    return (good, bad, bool(gt_edge.any() or pred_edge.any()),
            int(gt_edge.sum()), int(pred_edge.sum()))


def validation_row(confidence, severity, artifact_band):
    height, width = severity.shape
    risk = sbsbench.resize_depth(confidence["risk"], width, height)
    conf_band = nearest_bool(confidence["band"], width, height)
    support = artifact_band & conf_band
    artifact = severity >= 1.0
    positives = support & artifact
    negatives = support & ~artifact
    enough_classes = (int(positives.sum()) >= MIN_CLASS_PIXELS and
                      int(negatives.sum()) >= MIN_CLASS_PIXELS)
    auc = rank_auc(risk[support], artifact[support]) if enough_classes else None
    low_confidence = risk >= 0.35
    capture = (float(np.mean(low_confidence[positives])) * 100.0
               if positives.any() else None)
    risky = support & low_confidence
    safe = support & (risk <= 0.15)
    risky_rate = float(np.mean(artifact[risky]) * 100.0) if risky.any() else None
    safe_rate = float(np.mean(artifact[safe]) * 100.0) if safe.any() else None
    row = {
        "artifact_auc": auc,
        "artifact_capture_pct": capture,
        "artifact_rate_low_conf_pct": risky_rate,
        "artifact_rate_high_conf_pct": safe_rate,
        "artifact_support_px": int(support.sum()),
        "artifact_positive_px": int(positives.sum()),
        "artifact_negative_px": int(negatives.sum()),
        "risk_p95": float(np.percentile(risk[support], 95)) if support.any() else 0.0,
        "artifact_severity_p95": (
            float(np.percentile(severity[support], 95)) if support.any() else 0.0),
    }
    return row, risk, support, artifact


def gt_validation(depth, risk, ground_truth, kind):
    good, bad, eligible, gt_edge_count, pred_edge_count = gt_edge_labels(
        depth, ground_truth, kind)
    if good is None:
        return None, 0, 0, False, 0, 0
    support = good | bad
    good_count, bad_count = int(good.sum()), int(bad.sum())
    auc = (rank_auc(risk[support], bad[support])
           if good_count >= MIN_CLASS_PIXELS and bad_count >= MIN_CLASS_PIXELS else None)
    return auc, good_count, bad_count, eligible, gt_edge_count, pred_edge_count


def heatmap(values):
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(values * 2.0, 0.0, 1.0)
    blue = np.clip((1.0 - values) * 1.6, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(values - 0.5) * 2.0, 0.0, 1.0) * 0.55
    return np.stack((red, green, blue), axis=2)


def save_visual(path, source_path, depth, risk, severity, support):
    source = sbsbench.load_rgb(source_path)
    width = min(640, source.shape[1])
    height = max(1, round(source.shape[0] * width / source.shape[1]))
    source_image = Image.fromarray((source * 255.0).astype(np.uint8)).resize(
        (width, height), Image.Resampling.LANCZOS)
    source_rgb = np.asarray(source_image, np.float32) / 255.0
    depth_view = sbsbench.resize_depth(depth, width, height)
    risk_view = sbsbench.resize_depth(risk, width, height)
    severity_view = sbsbench.resize_depth(np.clip(severity / 2.0, 0.0, 1.0), width, height)
    support_view = nearest_bool(support, width, height)
    confidence_overlay = source_rgb * 0.45 + heatmap(risk_view) * 0.55
    artifact_overlay = source_rgb.copy()
    artifact_overlay[support_view] = (
        source_rgb[support_view] * 0.35 + heatmap(severity_view)[support_view] * 0.65)
    strips = [source_rgb, np.repeat(depth_view[..., None], 3, axis=2),
              confidence_overlay, artifact_overlay]
    canvas = np.concatenate(strips, axis=1)
    Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def fmt(value, suffix=""):
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def aggregate(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def calibration_decision(rows, gt_frames_eligible, gt_frames_available=0):
    """Independently validate warp screening and predicted-boundary diagnostics."""
    total_frames = len(rows)
    artifact_auc = aggregate(rows, "artifact_auc")
    artifact_capture = aggregate(rows, "artifact_capture_pct")
    gt_auc = aggregate(rows, "gt_bad_edge_auc")
    artifact_auc_frames = sum(row.get("artifact_auc") is not None for row in rows)
    gt_auc_frames = sum(row.get("gt_bad_edge_auc") is not None for row in rows)
    artifact_evidence_ok = artifact_auc_frames >= max(1, (total_frames + 3) // 4)
    model_evidence_ok = (gt_frames_eligible > 0 and
                         gt_auc_frames >= max(1, (gt_frames_eligible + 1) // 2))
    warp_screening_validated = (
        artifact_evidence_ok and artifact_auc is not None and artifact_auc >= 0.65 and
        artifact_capture is not None and artifact_capture >= 50.0)
    model_boundary_validated = None
    if gt_frames_eligible > 0:
        model_boundary_validated = (
            model_evidence_ok and gt_auc is not None and gt_auc >= 0.55)
    return {
        "warp_screening_validated": warp_screening_validated,
        "model_boundary_validated": model_boundary_validated,
        "artifact_auc": artifact_auc,
        "artifact_capture_pct": artifact_capture,
        "gt_bad_edge_auc": gt_auc,
        "artifact_auc_frames": artifact_auc_frames,
        "gt_auc_frames": gt_auc_frames,
        "gt_frames_eligible": gt_frames_eligible,
        "gt_frames_available": gt_frames_available,
        "gt_missing_prediction_frames": sum(
            row.get("gt_edge_px", 0) >= MIN_CLASS_PIXELS and
            row.get("pred_edge_px", 0) < MIN_CLASS_PIXELS for row in rows),
        "total_frames": total_frames,
    }


def require_frame_ids(label, expected, actual):
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"{label} frame-id mismatch: missing={missing}, extra={extra}")


def write_report(out_dir, summary, examples):
    verdict = summary["conclusion"]
    model_verdict = summary["model_conclusion"]
    clip_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(clip["clip"]), fmt(clip.get("artifact_auc")),
            fmt(clip.get("artifact_capture_pct"), "%"), fmt(clip.get("gt_bad_edge_auc")),
            clip["frames"])
        for clip in summary["clips"])
    cards = "".join(
        f'<figure><img src="assets/{html.escape(example["asset"])}">'
        f'<figcaption><b>{html.escape(example["clip"])} frame {example["frame"]}</b> · '
        f'artifact AUC {fmt(example.get("artifact_auc"))} · '
        f'capture {fmt(example.get("artifact_capture_pct"), "%")}<br>'
        'Source · depth · warp risk · independently measured output artifact</figcaption>'
        '</figure>' for example in examples)
    page = f"""<!doctype html><meta charset="utf-8"><title>Depth confidence audit</title>
<style>
body{{font:15px system-ui;background:#101318;color:#e8edf5;margin:0}}main{{max-width:1200px;
margin:auto;padding:28px}}h1{{margin-bottom:6px}}.hero{{background:#19212c;border:1px solid #334155;
border-radius:14px;padding:20px;margin:18px 0}}.go{{color:#7ee787}}.hold{{color:#ffcc66}}
table{{width:100%;border-collapse:collapse;background:#161c25}}th,td{{padding:9px 12px;
border-bottom:1px solid #303947;text-align:left}}details{{margin:18px 0;background:#161c25;
padding:12px;border-radius:10px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,
minmax(460px,1fr));gap:16px}}figure{{margin:0;background:#161c25;padding:10px;border-radius:10px}}
img{{width:100%;height:auto}}figcaption{{line-height:1.45;padding:8px 2px;color:#c7d0dd}}
code{{color:#9ecbff}}</style><main>
<h1>Depth confidence calibration</h1><div>Diagnostic only · existing real-pipeline artifacts ·
no gate or renderer change</div>
<section class="hero"><h2 class="{'go' if summary['warp_screening_validated'] else 'hold'}">
{html.escape(verdict)}</h2>
<p>Median warp-risk versus final-output artifact AUC:
<b>{fmt(summary.get('artifact_auc'))}</b> (0.5 = chance). ·
Artifact capture at risk ≥ 0.35: <b>{fmt(summary.get('artifact_capture_pct'), '%')}</b>. ·
Model-risk versus GT misplaced-predicted-boundary AUC:
<b>{fmt(summary.get('gt_bad_edge_auc'))}</b>.</p>
<p>Evidence-bearing frames: artifact AUC <b>{summary['artifact_auc_frames']}/
{summary['total_frames']}</b> · GT AUC <b>{summary['gt_auc_frames']}/
{summary['gt_frames_eligible']}</b> boundary-eligible
({summary['gt_frames_available']} GT frames available).</p>
<p><b>{html.escape(model_verdict)}</b> · GT frames with a boundary but no predicted boundary:
<b>{summary['gt_missing_prediction_frames']}</b>.</p>
<p>{html.escape(summary['explanation'])}</p></section>
<details><summary><b>Definition and validation rules</b></summary><p>Model risk combines source-edge
misalignment, depth-edge softness, and flow-compensated depth disagreement. Warp risk combines
visible source texture (55%), softness (25%), depth-step strength (15%), and temporal disagreement
(5%) over a nine-native-pixel hazard strip. Outside that strip confidence is 1.0. Warp screening
is validated independently when it ranks final-eye artifacts above chance with sufficient capture
and evidence volume. Model-boundary validation is separate and only measures whether model risk
ranks misplaced versus aligned <em>predicted</em> boundaries; it cannot detect a GT boundary that
the model omitted. Each AUC frame needs at least {MIN_CLASS_PIXELS} pixels in both classes; at
least 25% of all frames must carry artifact AUC evidence, and at least 50% of boundary-eligible GT
frames must carry model AUC evidence.</p></details>
<h2>Validation by clip</h2><table><thead><tr><th>Clip</th><th>Output artifact AUC</th>
<th>Artifact capture</th><th>GT predicted-boundary AUC</th><th>Frames</th></tr></thead>
<tbody>{clip_rows}</tbody></table>
<h2>Visual evidence</h2><div class="grid">{cards}</div></main>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as report:
        report.write(page)


def audit(run_dir, out_dir):
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as results_file:
        results = json.load(results_file)
    clips_root = results["meta"]["clips_root"]
    if not os.path.isabs(clips_root):
        clips_root = os.path.abspath(clips_root)
    assets_dir = os.path.join(out_dir, "assets")
    shutil.rmtree(assets_dir, ignore_errors=True)
    os.makedirs(assets_dir, exist_ok=True)
    rows = []
    candidates = []
    clip_summaries = []
    gt_frames_available = 0
    gt_frames_eligible = 0
    for clip in sorted(results["clips"]):
        sequence = os.path.join(run_dir, clip)
        frames = os.path.join(clips_root, clip)
        sbs_files = sbsbench.indexed_files(os.path.join(sequence, "sbs_*.png"), "sbs_")
        depth_files = sbsbench.indexed_files(os.path.join(sequence, "depth_*.png"), "depth_")
        source_files = sbsbench.indexed_files(os.path.join(frames, "frame_*.*"), "frame_")
        gt_files = sbsbench.indexed_files(
            os.path.join(frames, "gt_depth", "frame_*.*"), "frame_")
        flow_files = sbsbench.indexed_files(
            os.path.join(frames, "gt_flow", "frame_*.npz"), "frame_")
        with open(os.path.join(frames, "meta.json"), encoding="utf-8") as meta_file:
            clip_meta = json.load(meta_file)
        gt_kind = clip_meta.get("gt_depth_kind", "disparity")
        frame_ids = sorted(sbs_files)
        if not frame_ids:
            raise ValueError(f"{clip}: no sbs_*.png frames found")
        require_frame_ids(f"{clip} depth", frame_ids, depth_files)
        require_frame_ids(f"{clip} source", frame_ids, source_files)
        if gt_files:
            require_frame_ids(f"{clip} GT depth", frame_ids, gt_files)
        expected_flow_ids = frame_ids[1:]
        if flow_files:
            require_frame_ids(f"{clip} GT flow", expected_flow_ids, flow_files)
        require_gt_depth = bool(clip_meta.get(
            "required_gt_depth", clip_meta.get("dataset")))
        require_gt_flow = bool(clip_meta.get(
            "required_gt_flow", clip_meta.get("dataset") == "TartanAir V2"))
        if require_gt_depth and not gt_files:
            raise ValueError(f"{clip}: metadata requires GT depth, but none was found")
        if require_gt_flow and not flow_files:
            raise ValueError(f"{clip}: metadata requires GT flow, but none was found")
        gt_frames_available += len(gt_files)
        previous_depth = previous_source = None
        clip_rows = []
        for frame_id in frame_ids:
            depth = sbsbench.load_depth(depth_files[frame_id])
            source = sbsbench.load_gray(source_files[frame_id])
            reference_flow = reference_valid = None
            if frame_id in flow_files:
                with np.load(flow_files[frame_id], allow_pickle=False) as flow_data:
                    reference_flow = np.asarray(flow_data["flow"], np.float32)
                    if "valid" in flow_data:
                        reference_valid = np.asarray(flow_data["valid"], bool)
            confidence = depth_confidence_map(
                depth, source, previous_depth, previous_source,
                reference_flow=reference_flow, reference_valid=reference_valid)
            severity, artifact_band = output_artifact_map(
                sbs_files[frame_id], source, depth)
            row, risk_small, support, _ = validation_row(
                confidence, severity, artifact_band)
            row.update({"clip": clip, "frame": frame_id})
            if frame_id in gt_files:
                ground_truth = sbsbench.load_depth(gt_files[frame_id])
                (gt_auc, good_count, bad_count, gt_eligible,
                 gt_edge_count, pred_edge_count) = gt_validation(
                     depth, confidence["model_risk"], ground_truth, gt_kind)
                row.update({"gt_bad_edge_auc": gt_auc, "gt_good_edges": good_count,
                            "gt_bad_edges": bad_count,
                            "gt_boundary_eligible": gt_eligible,
                            "gt_edge_px": gt_edge_count,
                            "pred_edge_px": pred_edge_count})
                gt_frames_eligible += int(gt_eligible)
            rows.append(row)
            clip_rows.append(row)
            candidates.append({
                "score": row["artifact_severity_p95"], "clip": clip, "frame": frame_id,
                "depth": depth_files[frame_id], "source": source_files[frame_id],
                "row": row, "severity": severity, "support": support,
                "risk_small": risk_small,
            })
            candidates.sort(key=lambda item: item["score"], reverse=True)
            del candidates[MAX_EXAMPLES:]
            previous_depth, previous_source = depth, source
        clip_summaries.append({
            "clip": clip, "frames": len(clip_rows),
            "artifact_auc": aggregate(clip_rows, "artifact_auc"),
            "artifact_capture_pct": aggregate(clip_rows, "artifact_capture_pct"),
            "gt_bad_edge_auc": aggregate(clip_rows, "gt_bad_edge_auc"),
        })

    decision_stats = calibration_decision(
        rows, gt_frames_eligible, gt_frames_available)
    if decision_stats["warp_screening_validated"]:
        conclusion = "GO: warp-risk screening is calibrated"
        explanation = ("Warp risk predicts independently measured output artifacts above chance "
                       "and captures enough affected pixels. It may screen a controlled warp "
                       "experiment, which must still pass the normal core and extended gates.")
    else:
        conclusion = "HOLD: warp-risk screening is not calibrated"
        explanation = ("The independent warp-risk evidence missed at least one calibration bar. "
                       "Do not use it to screen or alter rendering; inspect the examples first.")
    if decision_stats["model_boundary_validated"] is True:
        model_conclusion = "GO: predicted-boundary model risk is calibrated"
    elif decision_stats["model_boundary_validated"] is False:
        model_conclusion = "HOLD: predicted-boundary model risk lacks sufficient evidence"
    else:
        model_conclusion = "N/A: no eligible GT boundary evidence"
    summary = {
        "schema": 3, "source_run": run_dir,
        "conclusion": conclusion, "model_conclusion": model_conclusion,
        "explanation": explanation,
        **decision_stats, "clips": clip_summaries, "frames": rows,
    }
    selected = candidates
    examples = []
    for item in selected:
        asset = f'{item["clip"]}-{item["frame"]:05d}.jpg'
        save_visual(os.path.join(assets_dir, asset), item["source"],
                    sbsbench.load_depth(item["depth"]), item["risk_small"],
                    item["severity"], item["support"])
        examples.append({"asset": asset, "clip": item["clip"], "frame": item["frame"],
                         **item["row"]})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    write_report(out_dir, summary, examples)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="existing run containing results.json and frame artifacts")
    parser.add_argument("--out", help="report output directory")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out or os.path.join(args.run, "depth-confidence-audit"))
    result = audit(args.run, out_dir)
    print(result["conclusion"])
    print(result["model_conclusion"])
    print(f"artifact AUC={fmt(result['artifact_auc'])}, "
          f"capture={fmt(result['artifact_capture_pct'], '%')}, "
          f"GT AUC={fmt(result['gt_bad_edge_auc'])}")
    print(os.path.join(out_dir, "report.html"))


if __name__ == "__main__":
    main()

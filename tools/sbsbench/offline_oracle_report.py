"""Render optional offline-oracle results as a non-decisive report appendix."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import statistics
from typing import Callable


ORACLE_FILES = {
    "raft-stereo": "raft_stereo.json",
    "sea-raft": "sea_raft_temporal.json",
    "nvidia-flip": "nvidia_flip_appearance.json",
    "apple-isqoe": "apple_isqoe.json",
}


def _load_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as error:
        return None, f"invalid: {error}"
    if not isinstance(payload, dict):
        return None, "invalid: JSON root is not an object"
    return payload, None


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _median(values) -> float | None:
    finite = [number for value in values if (number := _number(value)) is not None]
    return statistics.median(finite) if finite else None


def _fmt(value, unit="", digits=2) -> str:
    number = _number(value)
    if number is None:
        return "&mdash;"
    return f"{number:.{digits}f}{html.escape(unit)}"


def _status(value) -> str:
    text = str(value or "unknown")
    css = "p-info" if text in ("ok", "complete") else "p-warn"
    if text == "failed":
        css = "p-crit"
    return f'<span class="pill {css}">{html.escape(text)}</span>'


def _coverage(measured, total, abstained=0, cut=0) -> str:
    measured_value = int(_number(measured) or 0)
    total_value = int(_number(total) or 0)
    text = f"{measured_value}/{total_value}"
    details = []
    if int(_number(abstained) or 0):
        details.append(f"{int(_number(abstained) or 0)} abstain")
    if int(_number(cut) or 0):
        details.append(f"{int(_number(cut) or 0)} cut")
    return html.escape(text + (" · " + ", ".join(details) if details else ""))


def _raft_row(clip: str, display_name: str, payload: dict | None, error: str | None) -> str:
    if payload is None:
        return (f"<tr><td>{html.escape(display_name)}</td><td>{_status(error)}</td>"
                '<td colspan="5">No valid per-clip payload</td></tr>')
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    metrics = [frame.get("metrics", {}) for frame in frames if isinstance(frame, dict)]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    texture = _median(metric.get("raft_supported_texture_pct") for metric in metrics)
    residual = _median(metric.get("raft_correspondence_residual_p95") for metric in metrics)
    exact = _median(metric.get("raft_exact_p95_px") for metric in metrics)
    bad_one = _median(metric.get("raft_exact_bad_1px_pct") for metric in metrics)
    vertical = _median(metric.get("raft_vertical_abs_p95_px") for metric in metrics)
    exact_text = _fmt(exact, " px")
    if bad_one is not None:
        exact_text += f" / {_fmt(bad_one, '%')} bad&gt;1"
    measured = summary.get("frames_measured", sum(
        metric.get("status") == "ok" for metric in metrics))
    abstained = summary.get("frames_abstained", sum(
        metric.get("status") == "abstained" for metric in metrics))
    total = summary.get("frames_total", len(frames))
    return (f"<tr data-clip=\"{html.escape(clip, quote=True)}\">"
            f"<td>{html.escape(display_name)}</td><td>{_status(payload.get('status'))}</td>"
            f"<td>{_coverage(measured, total, abstained)}</td>"
            f"<td>{_fmt(texture, '%')}</td><td>{_fmt(residual, '/255')}</td>"
            f"<td>{exact_text}</td><td>{_fmt(vertical, ' px')}</td></tr>")


def _aggregate(payload: dict, metric: str) -> float | None:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    return (_number(aggregate.get(f"{metric}_p95"))
            if _number(aggregate.get(f"{metric}_p95")) is not None
            else _number(aggregate.get(f"{metric}_p50")))


def _sea_row(clip: str, display_name: str, payload: dict | None, error: str | None) -> str:
    if payload is None:
        return (f"<tr><td>{html.escape(display_name)}</td><td>{_status(error)}</td>"
                '<td colspan="5">No valid per-clip payload</td></tr>')
    edge_ghost = _aggregate(payload, "sea_flow_edge_ghost_p95")
    flicker = _aggregate(payload, "sea_flow_flicker_p95")
    static = _aggregate(payload, "sea_static_jitter_p95")
    left_motion = _aggregate(payload, "sea_left_motion_mismatch_p95_px")
    right_motion = _aggregate(payload, "sea_right_motion_mismatch_p95_px")
    motion_values = [value for value in (left_motion, right_motion) if value is not None]
    motion = max(motion_values) if motion_values else None
    coverage = _coverage(
        payload.get("pairs_measured"), payload.get("pairs_total"),
        payload.get("pairs_abstained"), payload.get("pairs_cut"))
    return (f"<tr data-clip=\"{html.escape(clip, quote=True)}\">"
            f"<td>{html.escape(display_name)}</td><td>{_status(payload.get('status'))}</td>"
            f"<td>{coverage}</td>"
            f"<td>{_fmt(edge_ghost, '/255')}</td><td>{_fmt(flicker, '/255')}</td>"
            f"<td>{_fmt(static, '/255')}</td><td>{_fmt(motion, ' px')}</td></tr>")


def _flip_row(clip: str, display_name: str,
              payload: dict | None, error: str | None) -> str:
    if payload is None:
        return (f"<tr><td>{html.escape(display_name)}</td><td>{_status(error)}</td>"
                '<td colspan="6">No valid per-clip payload</td></tr>')
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    measured = [frame for frame in frames
                if isinstance(frame, dict) and frame.get("status") == "ok"]
    metrics = [value if isinstance(value := frame.get("metrics"), dict) else {}
               for frame in measured]
    support = [value if isinstance(value := frame.get("support"), dict) else {}
               for frame in measured]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    area_threshold = _number(payload.get("area_threshold"))
    if area_threshold is None:
        area_threshold = 0.05
    area_suffix = f"{int(round(area_threshold * 1000)):03d}"
    support_pct = _median(item.get("pct") for item in support)
    worst_eye = _median(item.get("flip_worst_eye_p99") for item in metrics)
    worst_area = _median(
        item.get(f"flip_worst_eye_area_gt_{area_suffix}_pct") for item in metrics)
    eye_imbalance = _median(
        item.get("flip_interocular_error_imbalance_p99") for item in metrics)
    area_imbalance = _median(
        item.get(f"flip_interocular_area_imbalance_gt_{area_suffix}_pct")
        for item in metrics)
    coverage = _coverage(
        summary.get("frames_measured"), summary.get("frames_total"),
        summary.get("frames_abstained"))
    unavailable = int(_number(summary.get("frames_unavailable")) or 0)
    failed = int(_number(summary.get("frames_failed")) or 0)
    if unavailable or failed:
        details = []
        if unavailable:
            details.append(f"{unavailable} unavailable")
        if failed:
            details.append(f"{failed} failed")
        coverage += ("<br><span class=\"sub\">" +
                     html.escape(", ".join(details)) + "</span>")
    return (f"<tr data-clip=\"{html.escape(clip, quote=True)}\">"
            f"<td>{html.escape(display_name)}</td><td>{_status(payload.get('status'))}</td>"
            f"<td>{coverage}</td>"
            f"<td>{_fmt(support_pct, '%')}</td><td>{_fmt(worst_eye)}</td>"
            f"<td>{_fmt(worst_area, '%')}</td><td>{_fmt(eye_imbalance)}</td>"
            f"<td>{_fmt(area_imbalance, '%')}</td></tr>")


def _isqoe_row(clip: str, display_name: str,
               payload: dict | None, error: str | None) -> str:
    if payload is None:
        return (f"<tr><td>{html.escape(display_name)}</td><td>{_status(error)}</td>"
                '<td colspan="5">No valid per-clip payload</td></tr>')
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    measured = [frame for frame in frames
                if isinstance(frame, dict) and frame.get("status") == "ok"]
    metrics = [value if isinstance(value := frame.get("metrics"), dict) else {}
               for frame in measured]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    mean_score = _median(item.get("isqoe_mean_score") for item in metrics)
    worst_score = _median(item.get("isqoe_worst_score") for item in metrics)
    order_delta = _median(item.get("isqoe_eye_order_delta") for item in metrics)
    frame_worst = max(
        (value for item in metrics
         if (value := _number(item.get("isqoe_worst_score"))) is not None),
        default=None,
    )
    coverage = _coverage(
        summary.get("frames_measured"), summary.get("frames_total"),
        summary.get("frames_abstained"))
    unavailable = int(_number(summary.get("frames_unavailable")) or 0)
    failed = int(_number(summary.get("frames_failed")) or 0)
    if unavailable or failed:
        details = []
        if unavailable:
            details.append(f"{unavailable} unavailable")
        if failed:
            details.append(f"{failed} failed")
        coverage += ("<br><span class=\"sub\">" +
                     html.escape(", ".join(details)) + "</span>")
    return (f'<tr data-clip="{html.escape(clip, quote=True)}">'
            f"<td>{html.escape(display_name)}</td><td>{_status(payload.get('status'))}</td>"
            f"<td>{coverage}</td><td>{_fmt(mean_score, digits=3)}</td>"
            f"<td>{_fmt(worst_score, digits=3)}</td><td>{_fmt(frame_worst, digits=3)}</td>"
            f"<td>{_fmt(order_delta, digits=4)}</td></tr>")


def _isqoe_provenance(manifest: dict | None) -> tuple[str, list[str]]:
    """Render auditable official-checkpoint identity and return contract issues."""
    if not manifest:
        return "", []
    oracle_roots = manifest.get("oracles")
    oracle_root = (oracle_roots.get("apple-isqoe")
                   if isinstance(oracle_roots, dict) else None)
    provenance = (oracle_root.get("provenance")
                  if isinstance(oracle_root, dict) else None)
    if not isinstance(provenance, dict):
        return "", ["Apple iSQoE root result lacks model/checkpoint provenance."]
    checkpoint_id = provenance.get("official_checkpoint_id")
    checkpoint_sha = provenance.get("checkpoint_sha256")
    repository_revision = provenance.get("repository_revision")
    source_url = provenance.get("official_checkpoint_url")
    summary = oracle_root.get("summary")
    measured = (_number(summary.get("frames_measured"))
                if isinstance(summary, dict) else None)
    issues = []
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        issues.append("Apple iSQoE provenance lacks official_checkpoint_id.")
    if measured and (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64):
        issues.append("Apple iSQoE provenance lacks a full checkpoint SHA-256.")
    if provenance.get("checkpoint_matches_known_official_sha256") is False:
        issues.append("Apple iSQoE checkpoint does not match the recorded official release.")
    parts = []
    if checkpoint_id:
        parts.append(f"checkpoint {html.escape(str(checkpoint_id))}")
    if checkpoint_sha:
        parts.append(f"SHA-256 {html.escape(str(checkpoint_sha))}")
    if repository_revision:
        parts.append(f"repo {html.escape(str(repository_revision))}")
    if source_url:
        safe_url = html.escape(str(source_url), quote=True)
        parts.append(f'<a href="{safe_url}">official download</a>')
    if not parts:
        return "", issues
    return '<p class="sub">Model provenance: ' + "; ".join(parts) + ".</p>", issues


def build_section(treatment_dir: str | Path, clips: list[str],
                  display_name: Callable[[str], str]) -> str:
    """Return collapsed offline-oracle HTML, or empty text when no manifest exists."""
    treatment = Path(treatment_dir)
    manifest_path = treatment / "offline_oracles.json"
    manifest, manifest_error = _load_json(manifest_path)
    if manifest is None and manifest_error == "missing":
        return ""

    selected = manifest.get("selected_oracles", []) if manifest else []
    selected = [oracle for oracle in selected if oracle in ORACLE_FILES]
    if not selected and manifest:
        selected = [oracle for oracle in manifest.get("oracles", {}) if oracle in ORACLE_FILES]
    contract_issues = []
    if manifest is None:
        contract_issues.append(f"Root manifest {manifest_error}.")
    elif manifest.get("training_label_eligible") is not False:
        contract_issues.append("Root manifest lacks training_label_eligible=false.")

    payloads = {oracle: {} for oracle in selected}
    missing_payloads = 0
    for oracle in selected:
        for clip in clips:
            path = treatment / clip / "offline_oracles" / ORACLE_FILES[oracle]
            payload, error = _load_json(path)
            payloads[oracle][clip] = (payload, error)
            if payload is None:
                missing_payloads += 1
            elif payload.get("training_label_eligible") is not False:
                contract_issues.append(
                    f"{clip}/{ORACLE_FILES[oracle]} lacks training_label_eligible=false.")
            elif oracle == "apple-isqoe":
                frames = payload.get("frames")
                if isinstance(frames, list) and any(
                        not isinstance(frame, dict) or
                        frame.get("training_label_eligible") is not False
                        for frame in frames):
                    contract_issues.append(
                        f"{clip}/{ORACLE_FILES[oracle]} has a label-eligible frame.")

    isqoe_provenance, provenance_issues = _isqoe_provenance(
        manifest if "apple-isqoe" in selected else None)
    contract_issues.extend(provenance_issues)

    manifest_status = manifest.get("status") if manifest else "invalid"
    summary = (f"Offline-oracle diagnostics &mdash; {html.escape(str(manifest_status))}; "
               f"{len(selected)} selected, {missing_payloads} missing payloads")
    warning = ""
    if contract_issues:
        items = "".join(f"<li>{html.escape(issue)}</li>" for issue in contract_issues)
        warning = (f'<div class="evidence-card evidence-cost"><div class="ic-head">'
                   f'<span class="pill">contract warning</span></div><ul>{items}</ul></div>')

    sections = []
    if "raft-stereo" in selected:
        rows = "".join(_raft_row(clip, display_name(clip), *payloads["raft-stereo"][clip])
                       for clip in clips)
        sections.append(
            '<h3>RAFT-Stereo correspondence</h3>'
            '<p class="sub">Independent learned correspondence. Values are per-clip medians; '
            'exact-map error is shown only where orientation and correspondence support were valid.</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th>Clip</th><th>Status</th>'
            '<th>Frames</th><th>Texture support</th><th>Photo residual p95</th>'
            f'<th>Exact-map p95 / bad&gt;1</th><th>Vertical p95</th></tr></thead><tbody>{rows}'
            '</tbody></table></div>')
    if "sea-raft" in selected:
        rows = "".join(_sea_row(clip, display_name(clip), *payloads["sea-raft"][clip])
                       for clip in clips)
        sections.append(
            '<h3>SEA-RAFT temporal diagnostics</h3>'
            '<p class="sub">Flow-compensated, source-subtracted temporal evidence. Values prefer '
            'the across-pair p95 aggregate; scene cuts and low-confidence flow abstain.</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th>Clip</th><th>Status</th>'
            '<th>Pairs</th><th>Edge ghost p95</th><th>Flicker p95</th>'
            f'<th>Static jitter p95</th><th>Motion mismatch p95</th></tr></thead><tbody>{rows}'
            '</tbody></table></div>')
    if "nvidia-flip" in selected:
        rows = "".join(_flip_row(clip, display_name(clip), *payloads["nvidia-flip"][clip])
                       for clip in clips)
        sections.append(
            '<h3>NVIDIA FLIP registered appearance</h3>'
            '<p class="sub">Official LDR FLIP after exact production-map registration. Values '
            'are per-clip frame medians over uniquely visible mutual source support; HDR preview '
            'PNGs abstain because they are not valid FLIP-HDR inputs.</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th>Clip</th><th>Status</th>'
            '<th>Frames</th><th>Support</th><th>Worst-eye p99</th>'
            '<th>Area above FLIP threshold</th><th>Eye imbalance p99</th>'
            f'<th>Area imbalance</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if "apple-isqoe" in selected:
        rows = "".join(_isqoe_row(
            clip, display_name(clip), *payloads["apple-isqoe"][clip]) for clip in clips)
        sections.append(
            '<h3>Apple iSQoE headset-preference cross-check</h3>'
            '<p class="sub">Official VR-preference model; lower is preferred by its training '
            'population. The score mixes fidelity, comfort, and depth taste, so it is not an '
            'Apollo style target or a training label. Both eye orders are measured to expose '
            'order sensitivity. Compare candidates for the same source; absolute cross-clip '
            'ranking is unsupported. HDR debug previews abstain.</p>'
            f'{isqoe_provenance}'
            '<div style="overflow-x:auto"><table><thead><tr><th>Clip</th><th>Status</th>'
            '<th>Frames</th><th>Median mean</th><th>Median worse-order</th>'
            f'<th>Worst frame</th><th>Median order delta</th></tr></thead><tbody>{rows}'
            '</tbody></table></div>')
    if not sections:
        sections.append('<p class="sub">No recognized offline oracle is selected.</p>')

    return (f'<details class="fold learned-oracles"><summary>{summary}</summary>'
            '<div class="fold-body"><p class="sub"><b>Diagnostic only:</b> this appendix is not '
            'used by gates, the A/B verdict, the conclusion, or training labels. Oracle outputs '
            'remain unqualified until controlled-corruption, benign-invariance, and headset '
            'viewing-condition validation.</p>'
            f'{warning}{"".join(sections)}</div></details>')

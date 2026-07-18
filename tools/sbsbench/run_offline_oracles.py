#!/usr/bin/env python3
"""Run optional expensive SBS diagnostics over an existing evaluator run.

This is deliberately separate from ``run_eval.py``: missing third-party checkouts or model
weights must never make the deterministic evaluator unusable.  Once explicitly invoked, however,
the runner fails closed unless ``--allow-unavailable`` was requested.  Offline-oracle results are
diagnostic evidence only and are stamped ``training_label_eligible=false`` at every manifest level.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

try:
    import flip_appearance_oracle
    import isqoe_oracle
    import raft_stereo_oracle
    import sea_raft_temporal_oracle
except ImportError:  # pragma: no cover - package import used by unittest discovery
    from . import flip_appearance_oracle
    from . import isqoe_oracle
    from . import raft_stereo_oracle
    from . import sea_raft_temporal_oracle


SCHEMA = 1
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ORACLE_NAMES = ("raft-stereo", "sea-raft", "nvidia-flip", "apple-isqoe")
OUTPUT_NAMES = {
    "raft-stereo": "raft_stereo.json",
    "sea-raft": "sea_raft_temporal.json",
    "nvidia-flip": "nvidia_flip_appearance.json",
    "apple-isqoe": "apple_isqoe.json",
}
ENVIRONMENT_PATHS = {
    "raft_repo": "APOLLO_RAFT_STEREO_REPO",
    "raft_checkpoint": "APOLLO_RAFT_STEREO_CHECKPOINT",
    "sea_repo": "APOLLO_SEA_RAFT_REPO",
    "sea_checkpoint": "APOLLO_SEA_RAFT_CHECKPOINT",
    "sea_config": "APOLLO_SEA_RAFT_CONFIG",
    "isqoe_repo": "APOLLO_ISQOE_REPO",
    "isqoe_checkpoint": "APOLLO_ISQOE_CHECKPOINT",
}


class OracleRunnerError(RuntimeError):
    """Input/run-contract error that prevents trustworthy orchestration."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON result so interrupted runs leave no partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configured_path(value: Path | None, environment_name: str) -> Path | None:
    if value is not None:
        return value.expanduser().resolve()
    environment_value = os.environ.get(environment_name)
    return Path(environment_value).expanduser().resolve() if environment_value else None


def _safe_clip_name(name: str) -> str:
    path = Path(name)
    if not name or path.is_absolute() or len(path.parts) != 1 or path.name != name:
        raise OracleRunnerError(f"unsafe clip identity in results.json: {name!r}")
    return name


def _load_run(run_dir: Path, clips_root_override: Path | None) -> tuple[dict, Path, list[str]]:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise OracleRunnerError(f"existing evaluator results are missing: {results_path}")
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OracleRunnerError(f"cannot read evaluator results: {error}") from error
    clips_value = results.get("clips")
    if not isinstance(clips_value, dict) or not clips_value:
        raise OracleRunnerError("results.json has no non-empty clips object")
    clips = [_safe_clip_name(str(name)) for name in clips_value]

    if clips_root_override is not None:
        clips_root = clips_root_override.expanduser().resolve()
    else:
        configured = results.get("meta", {}).get("clips_root")
        if not isinstance(configured, str) or not configured:
            raise OracleRunnerError(
                "results.json has no clips_root; pass --clips-root explicitly")
        clips_root = Path(configured).expanduser()
        if not clips_root.is_absolute():
            clips_root = REPO_ROOT / clips_root
        clips_root = clips_root.resolve()
    if not clips_root.is_dir():
        raise OracleRunnerError(f"source clips root does not exist: {clips_root}")
    return results, clips_root, clips


def _dependency_issue(paths: dict[str, Path | None], required: tuple[str, ...]) -> str | None:
    missing_configuration = [name for name in required if paths.get(name) is None]
    if missing_configuration:
        variables = [ENVIRONMENT_PATHS[name] for name in missing_configuration]
        return "missing path option/environment variable: " + ", ".join(variables)
    missing_files = [f"{name}={paths[name]}" for name in required
                     if paths[name] is not None and not paths[name].exists()]
    if missing_files:
        return "configured dependency does not exist: " + ", ".join(missing_files)
    return None


def _status_payload(oracle: str, status: str, reason: str, clip: str | None = None) -> dict:
    payload = {
        "schema": SCHEMA,
        "oracle": oracle,
        "status": status,
        "role": "optional_eval_only_oracle",
        "training_label_eligible": False,
        "reason": reason,
    }
    if clip is not None:
        payload["clip"] = clip
    return payload


def _frame_summary(frames: list[dict]) -> dict[str, int]:
    counts = {"frames_total": len(frames), "frames_measured": 0,
              "frames_abstained": 0, "frames_failed": 0}
    for frame in frames:
        status = frame.get("status") or frame.get("metrics", {}).get("status")
        if status == "ok":
            counts["frames_measured"] += 1
        elif status == "abstained":
            counts["frames_abstained"] += 1
        else:
            counts["frames_failed"] += 1
    return counts


def _clip_output(run_dir: Path, clip: str, oracle: str) -> Path:
    return run_dir / clip / "offline_oracles" / OUTPUT_NAMES[oracle]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _indexed_paths(directory: Path, prefix: str,
                   suffixes: tuple[str, ...]) -> dict[int, Path]:
    """Index exact numeric frame IDs and reject ambiguous duplicate source inputs."""
    indexed: dict[int, Path] = {}
    suffix_set = {suffix.lower() for suffix in suffixes}
    if not directory.is_dir():
        return indexed
    for path in directory.iterdir():
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        if path.suffix.lower() not in suffix_set:
            continue
        frame_token = path.stem[len(prefix):]
        if not frame_token.isdigit():
            continue
        frame_id = int(frame_token)
        if frame_id in indexed:
            raise OracleRunnerError(
                f"ambiguous frame {frame_id}: {indexed[frame_id]} and {path}")
        indexed[frame_id] = path
    return indexed


def _load_json_object(path: Path, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OracleRunnerError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise OracleRunnerError(f"{description} is not a JSON object: {path}")
    return payload


def _flip_frame_failure(frame_id: int, reason: str, inputs: dict[str, str]) -> dict:
    return {
        "schema": flip_appearance_oracle.SCHEMA,
        "oracle": flip_appearance_oracle.ORACLE,
        "status": "failed",
        "role": flip_appearance_oracle.ROLE,
        "qualification": "experimental_diagnostic_only",
        "training_label_eligible": False,
        "frame_id": frame_id,
        "inputs": inputs,
        "reason": reason,
        "metrics": {},
    }


def _flip_summary_status(summary: dict[str, int]) -> str:
    if summary["frames_failed"]:
        return "failed"
    if summary["frames_unavailable"]:
        return "unavailable"
    if summary["frames_measured"]:
        return "ok"
    return "abstained"


def _run_flip(run_dir: Path, clips_root: Path, clips: list[str], ppd: float,
              area_threshold: float, save_evidence: bool
              ) -> tuple[dict, dict[str, dict]]:
    """Run exact-map-registered LDR FLIP, preserving explicit HDR abstention."""
    records: dict[str, dict] = {}
    totals = {
        "frames_total": 0,
        "frames_measured": 0,
        "frames_abstained": 0,
        "frames_unavailable": 0,
        "frames_failed": 0,
    }
    for clip in clips:
        source_dir = clips_root / clip
        output_dir = run_dir / clip
        sources = _indexed_paths(
            source_dir, "frame_", (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        sbs_frames = _indexed_paths(output_dir, "sbs_", (".png",))
        warp_maps = _indexed_paths(output_dir, "warp_map_", (".f32",))
        shape_path = output_dir / "warp_map_shape.json"
        hdr_path = output_dir / "hdr_output_stats.json"
        shape = (_load_json_object(shape_path, "warp-map shape contract")
                 if shape_path.is_file() else None)
        hdr_stats = (_load_json_object(hdr_path, "HDR preview contract")
                     if hdr_path.is_file() else None)
        frames: list[dict] = []
        clip_summary = {key: 0 for key in totals}

        if not sbs_frames:
            clip_level_reason = f"no sbs_*.png artifacts in {output_dir}"
        elif shape is None:
            clip_level_reason = f"exact warp-map shape contract is missing: {shape_path}"
        else:
            clip_level_reason = ""

        extra_maps = sorted(set(warp_maps) - set(sbs_frames))
        if extra_maps:
            clip_level_reason = (
                f"warp maps without matching SBS frames: {extra_maps}")

        for frame_id, sbs_path in sorted(sbs_frames.items()):
            source_path = sources.get(frame_id)
            map_path = warp_maps.get(frame_id)
            inputs = {
                "source": str(source_path) if source_path else "",
                "sbs": _relative(sbs_path, run_dir),
                "warp_map": _relative(map_path, run_dir) if map_path else "",
                "warp_shape": _relative(shape_path, run_dir) if shape is not None else "",
            }
            if hdr_path.is_file():
                inputs["hdr_stats"] = _relative(hdr_path, run_dir)
            failure = clip_level_reason
            if source_path is None:
                failure = f"source frame_{frame_id:05d} is missing from {source_dir}"
            elif map_path is None:
                failure = f"exact warp_map_{frame_id:05d}.f32 is missing"
            if failure:
                frame = _flip_frame_failure(frame_id, failure, inputs)
            else:
                try:
                    with Image.open(source_path) as image:
                        source = np.asarray(image.convert("RGB"))
                    with Image.open(sbs_path) as image:
                        sbs = np.asarray(image.convert("RGB"))
                    if sbs.shape[1] % 2:
                        raise ValueError(f"SBS width must be even, got {sbs.shape[1]}")
                    eye_width = sbs.shape[1] // 2
                    packed = np.fromfile(map_path, dtype=np.float32)
                    expected_count = sbs.shape[0] * sbs.shape[1]
                    if packed.size != expected_count:
                        raise ValueError(
                            f"warp map has {packed.size} floats, expected {expected_count}")
                    packed = packed.reshape(sbs.shape[:2])
                    measured = flip_appearance_oracle.measure_flip_appearance(
                        source, sbs[:, :eye_width], sbs[:, eye_width:],
                        packed[:, :eye_width], packed[:, eye_width:], shape,
                        hdr_output_stats=hdr_stats, ppd=ppd,
                        area_threshold=area_threshold,
                        return_maps=save_evidence,
                    )
                    if save_evidence:
                        payload, maps = measured
                        if maps:
                            evidence_path = (output_dir / "offline_oracles" /
                                             "nvidia_flip_evidence" /
                                             f"flip_{frame_id:05d}.npz")
                            evidence_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(evidence_path, **maps)
                            payload["evidence_npz"] = _relative(evidence_path, run_dir)
                    else:
                        payload = measured
                    frame = copy.deepcopy(payload)
                    frame.update({"frame_id": frame_id, "inputs": inputs})
                except Exception as error:
                    frame = _flip_frame_failure(
                        frame_id, f"{type(error).__name__}: {error}", inputs)
            frames.append(frame)
            clip_summary["frames_total"] += 1
            status = frame.get("status")
            summary_key = {
                "ok": "frames_measured",
                "abstained": "frames_abstained",
                "unavailable": "frames_unavailable",
            }.get(status, "frames_failed")
            clip_summary[summary_key] += 1

        if not frames and clip_level_reason:
            clip_summary["frames_failed"] = 1
        status = _flip_summary_status(clip_summary)
        clip_payload = {
            "schema": SCHEMA,
            "oracle": flip_appearance_oracle.ORACLE,
            "status": status,
            "role": "optional_eval_only_oracle",
            "qualification": "experimental_diagnostic_only",
            "training_label_eligible": False,
            "clip": clip,
            "pixels_per_degree": ppd,
            "area_threshold": area_threshold,
            "exact_map_required": True,
            "summary": clip_summary,
            "frames": frames,
        }
        if not frames and clip_level_reason:
            clip_payload["reason"] = clip_level_reason
        output = _clip_output(run_dir, clip, "nvidia-flip")
        _write_json(output, clip_payload)
        for key, value in clip_summary.items():
            totals[key] += value
        records[clip] = {
            "status": status,
            "result": _relative(output, run_dir),
            "summary": clip_summary,
            "training_label_eligible": False,
        }

    root = {
        "status": _flip_summary_status(totals),
        "summary": totals,
        "oracle": flip_appearance_oracle.ORACLE,
        "qualification": "experimental_diagnostic_only",
        "pixels_per_degree": ppd,
        "area_threshold": area_threshold,
        "registration": "production exact source-U maps; unique mutual source support",
        "hdr_contract": "HDR preview PNGs abstain; raw calibrated HDR is required",
        "training_label_eligible": False,
    }
    return root, records


def _write_status_for_clips(run_dir: Path, clips: list[str], oracle: str,
                            status: str, reason: str) -> dict[str, dict]:
    records = {}
    for clip in clips:
        output = _clip_output(run_dir, clip, oracle)
        _write_json(output, _status_payload(oracle, status, reason, clip))
        records[clip] = {
            "status": status,
            "result": _relative(output, run_dir),
            "training_label_eligible": False,
        }
    return records


def _run_raft(run_dir: Path, clips: list[str], paths: dict[str, Path | None],
              device: str | None, valid_iters: int, max_eye_width: int,
              save_fields: bool) -> tuple[dict, dict[str, dict]]:
    inputs: list[Path] = []
    owners: dict[Path, str] = {}
    fields_root = run_dir / "offline_oracle_fields" / "raft_stereo" if save_fields else None
    for clip in clips:
        clip_dir = run_dir / clip
        frame_paths = sorted(clip_dir.glob("sbs_*.png"))
        if not frame_paths:
            raise OracleRunnerError(f"{clip}: no sbs_*.png artifacts in {clip_dir}")
        for frame_path in frame_paths:
            resolved = frame_path.resolve()
            inputs.append(resolved)
            owners[resolved] = clip

    result = raft_stereo_oracle.evaluate_paths(
        inputs, paths["raft_repo"], paths["raft_checkpoint"], device,
        valid_iters, max_eye_width, fields_root)
    common = {key: copy.deepcopy(value) for key, value in result.items() if key != "frames"}
    common.update({
        "role": "optional_eval_only_oracle",
        "training_label_eligible": False,
    })
    grouped = {clip: [] for clip in clips}
    for frame in result.get("frames", []):
        raw_path = frame.get("path")
        if not isinstance(raw_path, str):
            raise OracleRunnerError("RAFT-Stereo returned a frame without a path")
        frame_path = Path(raw_path).resolve()
        clip = owners.get(frame_path)
        if clip is None:
            raise OracleRunnerError(f"RAFT-Stereo returned an unknown frame: {frame_path}")
        grouped[clip].append(frame)
    if sum(map(len, grouped.values())) != len(inputs):
        raise OracleRunnerError(
            "RAFT-Stereo did not return exactly one result for every input frame")

    records = {}
    total_summary = {"frames_total": 0, "frames_measured": 0,
                     "frames_abstained": 0, "frames_failed": 0}
    for clip in clips:
        summary = _frame_summary(grouped[clip])
        status = "ok" if summary["frames_measured"] else "abstained"
        clip_payload = {
            **copy.deepcopy(common),
            "status": status,
            "clip": clip,
            "summary": summary,
            "frames": grouped[clip],
        }
        output = _clip_output(run_dir, clip, "raft-stereo")
        _write_json(output, clip_payload)
        for key, value in summary.items():
            total_summary[key] += value
        records[clip] = {
            "status": status,
            "result": _relative(output, run_dir),
            "summary": summary,
            "training_label_eligible": False,
        }
    root = {
        "status": "ok" if total_summary["frames_measured"] else "abstained",
        "summary": total_summary,
        "provenance": common,
        "training_label_eligible": False,
    }
    return root, records


def _run_sea(run_dir: Path, clips_root: Path, clips: list[str],
             paths: dict[str, Path | None], device: str | None,
             source_only_flow: bool, evidence: bool) -> tuple[dict, dict[str, dict]]:
    config = paths["sea_config"]
    if config is None:
        config = paths["sea_repo"] / "config" / "eval" / "spring-M.json"
    model = sea_raft_temporal_oracle.SeaRaftModel(
        paths["sea_repo"], paths["sea_checkpoint"], config, device)
    records = {}
    totals = {"pairs_total": 0, "pairs_measured": 0,
              "pairs_cut": 0, "pairs_abstained": 0}
    for clip in clips:
        source_dir = clips_root / clip
        output_dir = run_dir / clip
        if not source_dir.is_dir():
            raise OracleRunnerError(f"{clip}: source directory is missing: {source_dir}")
        evidence_dir = (output_dir / "offline_oracles" / "sea_raft_evidence"
                        if evidence else None)
        result = sea_raft_temporal_oracle.evaluate_sequence(
            source_dir, output_dir, model, "frame_*.*", "sbs_*.png",
            evidence_dir, not source_only_flow)
        result.update({
            "clip": clip,
            "status": "ok" if result.get("pairs_measured", 0) else "abstained",
            "role": "optional_eval_only_oracle",
            "training_label_eligible": False,
        })
        output = _clip_output(run_dir, clip, "sea-raft")
        _write_json(output, result)
        summary = {key: int(result.get(key, 0)) for key in totals}
        for key, value in summary.items():
            totals[key] += value
        records[clip] = {
            "status": result["status"],
            "result": _relative(output, run_dir),
            "summary": summary,
            "training_label_eligible": False,
        }
    root = {
        "status": "ok" if totals["pairs_measured"] else "abstained",
        "summary": totals,
        "model": {
            "repo": str(model.repo),
            "checkpoint": str(model.checkpoint),
            "config": str(model.config),
            "device": model.device,
        },
        "training_label_eligible": False,
    }
    return root, records


def _isqoe_status_frame(frame_id: int, path: Path, status: str, reason: str) -> dict:
    return {
        "schema": isqoe_oracle.SCHEMA,
        "oracle": isqoe_oracle.ORACLE,
        "status": status,
        "role": isqoe_oracle.ROLE,
        "qualification": "experimental_diagnostic_only",
        "training_label_eligible": False,
        "frame_id": frame_id,
        "path": str(path.resolve()),
        "reason": reason,
        "metrics": {},
    }


def _isqoe_summary(frames: list[dict]) -> dict[str, int]:
    counts = {
        "frames_total": len(frames),
        "frames_measured": 0,
        "frames_abstained": 0,
        "frames_unavailable": 0,
        "frames_failed": 0,
    }
    destinations = {
        "ok": "frames_measured",
        "abstained": "frames_abstained",
        "unavailable": "frames_unavailable",
    }
    for frame in frames:
        counts[destinations.get(frame.get("status"), "frames_failed")] += 1
    return counts


def _isqoe_summary_status(summary: dict[str, int]) -> str:
    if summary["frames_failed"]:
        return "failed"
    if summary["frames_unavailable"]:
        return "unavailable"
    if summary["frames_measured"]:
        return "ok"
    return "abstained"


def _run_isqoe(
    run_dir: Path,
    clips: list[str],
    paths: dict[str, Path | None],
    device: str | None,
) -> tuple[dict, dict[str, dict]]:
    """Run Apple's headset-preference model once per LDR SBS frame.

    iSQoE was trained on display-ready stereo images, not Apollo's tone-mapped debug preview of a
    linear HDR surface.  HDR clips therefore abstain before model or dependency loading.
    """
    inputs: dict[str, list[tuple[int, Path]]] = {}
    hdr_reasons: dict[str, str] = {}
    for clip in clips:
        clip_dir = run_dir / clip
        frames = _indexed_paths(clip_dir, "sbs_", (".png",))
        if not frames:
            raise OracleRunnerError(f"{clip}: no sbs_*.png artifacts in {clip_dir}")
        hdr_path = clip_dir / "hdr_output_stats.json"
        hdr_stats = (_load_json_object(hdr_path, "HDR preview contract")
                     if hdr_path.is_file() else None)
        reason = flip_appearance_oracle._hdr_preview_reason(hdr_stats)
        if reason:
            hdr_reasons[clip] = reason
        inputs[clip] = sorted(frames.items())

    ldr_clips = [clip for clip in clips if clip not in hdr_reasons]
    model = None
    dependency_issue = None
    model_failure = None
    if ldr_clips:
        dependency_issue = _dependency_issue(
            paths, ("isqoe_repo", "isqoe_checkpoint"))
        if dependency_issue is None:
            try:
                # One model instance is shared across every LDR frame and clip in this run.
                model = isqoe_oracle.IsqoeModel(
                    paths["isqoe_repo"], paths["isqoe_checkpoint"], device)
            except Exception as error:
                model_failure = f"{type(error).__name__}: {error}"

    totals = {
        "frames_total": 0,
        "frames_measured": 0,
        "frames_abstained": 0,
        "frames_unavailable": 0,
        "frames_failed": 0,
    }
    records: dict[str, dict] = {}
    for clip in clips:
        frames = []
        hdr_reason = hdr_reasons.get(clip)
        for frame_id, path in inputs[clip]:
            if hdr_reason:
                frame = _isqoe_status_frame(
                    frame_id, path, "abstained", hdr_reason)
            elif dependency_issue:
                frame = _isqoe_status_frame(
                    frame_id, path, "unavailable", dependency_issue)
            elif model_failure:
                frame = _isqoe_status_frame(
                    frame_id, path, "failed", model_failure)
            else:
                try:
                    frame = model.evaluate_path(path)
                    frame["frame_id"] = frame_id
                except Exception as error:
                    frame = _isqoe_status_frame(
                        frame_id, path, "failed", f"{type(error).__name__}: {error}")
            frames.append(frame)

        summary = _isqoe_summary(frames)
        status = _isqoe_summary_status(summary)
        provenance = model.provenance if model is not None else {
            "official_repository_url": isqoe_oracle.OFFICIAL_REPOSITORY_URL,
            "official_checkpoint_id": isqoe_oracle.OFFICIAL_CHECKPOINT_ID,
            "official_checkpoint_url": isqoe_oracle.OFFICIAL_CHECKPOINT_URL,
            "input_contract": "HDR debug previews are unsupported",
        }
        clip_payload = {
            "schema": SCHEMA,
            "oracle": isqoe_oracle.ORACLE,
            "status": status,
            "role": isqoe_oracle.ROLE,
            "qualification": "experimental_diagnostic_only",
            "training_label_eligible": False,
            "clip": clip,
            "summary": summary,
            "provenance": provenance,
            "frames": frames,
        }
        output = _clip_output(run_dir, clip, "apple-isqoe")
        _write_json(output, clip_payload)
        for key, value in summary.items():
            totals[key] += value
        records[clip] = {
            "status": status,
            "result": _relative(output, run_dir),
            "summary": summary,
            "training_label_eligible": False,
        }

    status = _isqoe_summary_status(totals)
    provenance = model.provenance if model is not None else {
        "official_repository_url": isqoe_oracle.OFFICIAL_REPOSITORY_URL,
        "official_checkpoint_id": isqoe_oracle.OFFICIAL_CHECKPOINT_ID,
        "official_checkpoint_url": isqoe_oracle.OFFICIAL_CHECKPOINT_URL,
        "input_contract": "HDR debug previews are unsupported",
    }
    root = {
        "status": status,
        "summary": totals,
        "role": isqoe_oracle.ROLE,
        "qualification": "experimental_diagnostic_only",
        "provenance": provenance,
        "interpretation": (
            "holistic VR preference cross-check; higher is worse; never a standalone label"),
        "training_label_eligible": False,
    }
    if dependency_issue:
        root["reason"] = dependency_issue
    elif model_failure:
        root["reason"] = model_failure
    return root, records


def run(args: argparse.Namespace) -> tuple[dict, int]:
    run_dir = args.run_dir.expanduser().resolve()
    results, clips_root, clips = _load_run(run_dir, args.clips_root)
    selected = list(dict.fromkeys(args.oracles))
    paths = {
        name: _configured_path(getattr(args, name), environment)
        for name, environment in ENVIRONMENT_PATHS.items()
    }
    root = {
        "schema": SCHEMA,
        "role": "optional_eval_only_oracle_manifest",
        "training_label_eligible": False,
        "run_dir": str(run_dir),
        "run_name": results.get("meta", {}).get("run_name", run_dir.name),
        "eval_schema": results.get("meta", {}).get("eval_schema"),
        "clips_root": str(clips_root),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_oracles": selected,
        "oracles": {},
        "clips": {clip: {} for clip in clips},
    }
    must_fail = False
    ran = 0
    for oracle in selected:
        required = (("raft_repo", "raft_checkpoint") if oracle == "raft-stereo"
                    else (("sea_repo", "sea_checkpoint")
                          if oracle == "sea-raft" else ()))
        issue = _dependency_issue(paths, required) if required else None
        if oracle == "sea-raft" and issue is None:
            if paths["sea_config"] is None:
                paths["sea_config"] = paths["sea_repo"] / "config" / "eval" / "spring-M.json"
            issue = _dependency_issue(paths, ("sea_config",))
        if issue is not None:
            records = _write_status_for_clips(
                run_dir, clips, oracle, "unavailable", issue)
            root["oracles"][oracle] = _status_payload(oracle, "unavailable", issue)
            for clip, record in records.items():
                root["clips"][clip][oracle] = record
            must_fail |= not args.allow_unavailable
            continue
        try:
            if oracle == "raft-stereo":
                oracle_root, records = _run_raft(
                    run_dir, clips, paths, args.device, args.raft_valid_iters,
                    args.raft_max_eye_width, args.save_raft_fields)
            elif oracle == "sea-raft":
                oracle_root, records = _run_sea(
                    run_dir, clips_root, clips, paths, args.device,
                    args.sea_source_only_flow, args.save_sea_evidence)
            elif oracle == "nvidia-flip":
                oracle_root, records = _run_flip(
                    run_dir, clips_root, clips, args.flip_ppd,
                    args.flip_area_threshold, args.save_flip_evidence)
            else:
                oracle_root, records = _run_isqoe(
                    run_dir, clips, paths, args.device)
            root["oracles"][oracle] = oracle_root
            for clip, record in records.items():
                root["clips"][clip][oracle] = record
            oracle_status = oracle_root.get("status")
            if oracle_status == "unavailable":
                must_fail |= not args.allow_unavailable
            elif oracle_status == "failed":
                must_fail = True
                ran += 1
            else:
                ran += 1
        except Exception as error:  # model/checkout failures must survive in the manifest
            reason = f"{type(error).__name__}: {error}"
            records = _write_status_for_clips(
                run_dir, clips, oracle, "failed", reason)
            root["oracles"][oracle] = _status_payload(oracle, "failed", reason)
            for clip, record in records.items():
                root["clips"][clip][oracle] = record
            must_fail = True

    unavailable = any(value["status"] == "unavailable"
                      for value in root["oracles"].values())
    failed = any(value["status"] == "failed" for value in root["oracles"].values())
    if failed:
        root["status"] = "failed"
    elif unavailable:
        root["status"] = "partial" if ran else "unavailable"
    else:
        root["status"] = "complete"
    _write_json(run_dir / "offline_oracles.json", root)
    return root, 2 if must_fail else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="existing run_eval output containing results.json")
    parser.add_argument("--clips-root", type=Path,
                        help="source clips override (default: results.json meta.clips_root)")
    parser.add_argument("--oracles", nargs="+", choices=ORACLE_NAMES,
                        default=list(ORACLE_NAMES))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--allow-unavailable", action="store_true",
                        help="write unavailable records and succeed when dependencies are absent")
    parser.add_argument("--raft-repo", type=Path)
    parser.add_argument("--raft-checkpoint", type=Path)
    parser.add_argument("--raft-valid-iters", type=int, default=32)
    parser.add_argument("--raft-max-eye-width", type=int, default=0)
    parser.add_argument("--save-raft-fields", action="store_true")
    parser.add_argument("--sea-repo", type=Path)
    parser.add_argument("--sea-checkpoint", type=Path)
    parser.add_argument("--sea-config", type=Path)
    parser.add_argument("--sea-source-only-flow", action="store_true")
    parser.add_argument("--save-sea-evidence", action="store_true")
    parser.add_argument("--isqoe-repo", type=Path)
    parser.add_argument("--isqoe-checkpoint", type=Path)
    parser.add_argument("--flip-ppd", type=float,
                        default=flip_appearance_oracle.DEFAULT_PPD,
                        help="FLIP viewing density in pixels per degree")
    parser.add_argument("--flip-area-threshold", type=float,
                        default=flip_appearance_oracle.DEFAULT_AREA_THRESHOLD)
    parser.add_argument("--save-flip-evidence", action="store_true",
                        help="save registered references, support, and FLIP maps as NPZ")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root, exit_code = run(args)
    except OracleRunnerError as error:
        parser.error(str(error))
    for name, result in root["oracles"].items():
        print(f"{name}: {result['status']}")
        if result.get("reason"):
            print(f"  {result['reason']}")
    print(f"manifest: {args.run_dir.resolve() / 'offline_oracles.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

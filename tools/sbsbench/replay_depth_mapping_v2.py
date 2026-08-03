#!/usr/bin/env python3
"""Raw-recomputed mapping-v2 experiment through Apollo's D3D SBS warp.

This tool is intentionally an experiment gate, not a production implementation.  It derives a
final one-eye parallax field from the dump's exact raw-depth bytes, encodes that field using the
``SBS_DIRECT_PARALLAX`` harness contract, then runs Apollo's real reprojection, forward-coverage,
inverse-map, and standard sbsbench scorer.  ``source.png`` is the dump's diagnostic color preview;
the recomputed geometry is deterministic for those bytes, while their originating ONNX model is
authoritative only when the capture manifest binds both hashes.  This is **not** an exact
captured-state replay: shot-latched center and temporal state can differ from the values used by
the live frame. For an exact captured-state check, use the dump's authenticated
``shadow_final_parallax`` field as direct-final input; candidate and canonical fields are
diagnostic evidence only. HDR color fidelity is not claimed.

The calibrated spatial projection is composed in a fixed order: first the column-wise vertical
near-preserving majorant, then the row-wise horizontal near-preserving majorant. Vertical shear
is expressed as horizontal disparity pixels per source-image vertical pixel and assumes the
calibrated DAV2 grid preserves the source aspect ratio.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict

import numpy as np

try:
    from .depth_mapping_v2 import (
        MappingV2Config,
        encode_direct_parallax,
        generate_depth_mapping_v2,
    )
except ImportError:  # Direct execution from tools/sbsbench.
    from depth_mapping_v2 import (  # type: ignore
        MappingV2Config,
        encode_direct_parallax,
        generate_depth_mapping_v2,
    )

try:
    from . import direct_geometry_contract as direct_geometry
    from . import depth_coordinate_v2_dump_contract as v2_dump_contract
    from . import raw_model_provenance
except ImportError:  # Direct execution from tools/sbsbench.
    import direct_geometry_contract as direct_geometry  # type: ignore
    import depth_coordinate_v2_dump_contract as v2_dump_contract  # type: ignore
    import raw_model_provenance  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _resolve_mapping_calibration(
        provenance: raw_model_provenance.RawModelProvenance,
        experimental_raw_coordinate_scale: float | None) -> tuple[float, Dict[str, Any]]:
    if provenance.authoritative:
        if experimental_raw_coordinate_scale is not None:
            raise ValueError(
                "--experimental-raw-coordinate-scale is only valid for unverified legacy dumps")
        if (provenance.calibrated_raw_coordinate_scale is None or
                provenance.calibration_id is None):
            raise ValueError("authoritative provenance lacks its fixed raw-coordinate scale")
        scale = provenance.calibrated_raw_coordinate_scale
        return scale, {
            "status": "authoritative",
            "authoritative": True,
            "calibration_id": provenance.calibration_id,
            "onnx_sha256": provenance.onnx_sha256,
            "preprocess_profile": provenance.preprocess_profile,
            "preprocess_source_closure_sha256":
                provenance.preprocess_source_closure_sha256,
            "raw_coordinate_scale": scale,
            "source": "depth-coordinate-v2 canonical model calibration manifest",
        }
    if experimental_raw_coordinate_scale is None:
        raise ValueError(
            "unverified legacy replay requires --experimental-raw-coordinate-scale")
    return experimental_raw_coordinate_scale, {
        "status": "experiment-unverified",
        "authoritative": False,
        "calibration_id": None,
        "onnx_sha256": None,
        "preprocess_profile": None,
        "preprocess_source_closure_sha256": None,
        "raw_coordinate_scale": experimental_raw_coordinate_scale,
        "source": "explicit experiment-only CLI override; not a model calibration claim",
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _inspect_optional_shadow_state(dump: Path) -> Dict[str, Any]:
    state_path = dump / "shadow_state.json"
    stats_path = dump / "shadow_frame_stats.json"
    if not state_path.exists() and not stats_path.exists():
        return {"status": "not-captured"}
    if not state_path.is_file() or not stats_path.is_file():
        raise ValueError("dump has an incomplete depth-coordinate-v2 shadow-state pair")
    state_document = _read_json(state_path)
    state = v2_dump_contract.validate_shadow_state_document(state_document)
    stats = v2_dump_contract.validate_shadow_frame_stats_document(_read_json(stats_path))
    collapse_epsilon = float(state_document["constants"]["collapse_abs_epsilon"])
    expected_frame_valid = bool(
        stats["valid"] > 0.5 and stats["population_std"] > collapse_epsilon)
    if bool(state["frame_valid"] > 0.5) != expected_frame_valid:
        raise ValueError("dump shadow state disagrees with its current-frame statistics")
    return {
        "status": "validated",
        "state_schema": v2_dump_contract.SHADOW_STATE_DUMP_SCHEMA,
        "frame_stats_schema": v2_dump_contract.SHADOW_FRAME_STATS_DUMP_SCHEMA,
        "frame_valid": bool(state["frame_valid"] > 0.5),
        "camera_valid": bool(
            state["inverse_scale"] > 0.0 and state["calibration_revision"] > 0),
        "calibration_revision": state["calibration_revision"],
        "confirmed_cut_count": state["confirmed_cut_count"],
        "frame_stats_valid": bool(stats["valid"] > 0.5),
    }


def _inspect_optional_v2_dump_manifest(dump: Path) -> Dict[str, Any]:
    manifest_path = dump / "dump_manifest.json"
    if not manifest_path.exists():
        return {"status": "not-captured"}
    if not manifest_path.is_file():
        raise ValueError("dump_manifest.json is not a regular file")
    decoded = v2_dump_contract.validate_v2_dump_manifest_document(
        _read_json(manifest_path))
    return {
        "status": "validated",
        "manifest_schema": v2_dump_contract.DUMP_MANIFEST_SCHEMA,
        **decoded,
    }


def _validate_direct_replay_contract(
        harness: Path, contract: Dict[str, Any], field_path: Path, order_path: Path,
        expected_width: int, expected_height: int) -> None:
    try:
        validated = direct_geometry.validate_artifacts(str(harness), contract, {1})
    except ValueError as exc:
        raise RuntimeError(
            f"harness did not attest the exact direct-geometry path: {exc}") from exc

    field = validated["manifest"]["fields"][0]
    if (field.get("frame_id") != "00001" or
            field.get("width") != expected_width or field.get("height") != expected_height or
            field.get("parallax_sha256") != direct_geometry.file_sha256(str(field_path)) or
            field.get("order_sha256") != direct_geometry.file_sha256(str(order_path))):
        raise RuntimeError("harness manifest does not authenticate the replayed geometry fields")

    expected_depth = harness / "depth_00001.f32"
    expected_parallax = harness / "parallax_00001.f32"
    if (validated["depth_files"] != {1: str(expected_depth)} or
            validated["parallax_files"] != {1: str(expected_parallax)} or
            validated["shapes"] != {1: (expected_height, expected_width)}):
        raise RuntimeError("harness published an unexpected direct-geometry artifact set")


def _new_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise ValueError(f"output directory must be new or empty: {path}")
    path.mkdir(parents=True)


def _load_raw_depth(dump: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    shape = _read_json(dump / "raw_shape.json")
    try:
        width = int(shape["width"])
        height = int(shape["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw_shape.json lacks positive integer width/height") from exc
    if width <= 0 or height <= 0 or shape.get("dtype") != "float32-le" or \
            shape.get("layout") != "row-major":
        raise ValueError("raw_shape.json is not finite float32-le row-major depth")
    raw_path = dump / "raw_depth.f32"
    try:
        raw = np.fromfile(raw_path, dtype="<f4")
    except OSError as exc:
        raise ValueError(f"cannot read {raw_path}: {exc}") from exc
    expected = width * height
    if raw.size != expected:
        raise ValueError(f"raw_depth.f32 has {raw.size} values, expected {expected}")
    raw = raw.reshape(height, width)
    if not np.isfinite(raw).all():
        raise ValueError("raw_depth.f32 contains non-finite values")
    return raw, shape


def _mapping_metrics(result) -> Dict[str, Any]:
    width = result.parallax.shape[1]
    pre = result.pre_limiter_parallax.astype(np.float64)
    vertical = result.post_vertical_parallax.astype(np.float64)
    post = result.parallax.astype(np.float64)
    if width > 1:
        pre_slope = np.abs(np.diff(pre, axis=1)) * width
        post_slope = np.abs(np.diff(post, axis=1)) * width
    else:
        pre_slope = np.zeros((post.shape[0], 0), dtype=np.float64)
        post_slope = pre_slope
    if post.shape[0] > 1:
        pre_vertical_shear = np.abs(np.diff(pre, axis=0)) * width
        post_vertical_stage_shear = np.abs(np.diff(vertical, axis=0)) * width
        post_vertical_shear = np.abs(np.diff(post, axis=0)) * width
    else:
        pre_vertical_shear = np.zeros((0, post.shape[1]), dtype=np.float64)
        post_vertical_stage_shear = pre_vertical_shear
        post_vertical_shear = pre_vertical_shear

    def slope_summary(values: np.ndarray) -> Dict[str, float]:
        if not values.size:
            return {"p99": 0.0, "p99_9": 0.0, "maximum": 0.0}
        return {
            "p99": float(np.quantile(values, 0.99)),
            "p99_9": float(np.quantile(values, 0.999)),
            "maximum": float(np.max(values)),
        }

    near = result.canonical.astype(np.float64) >= 0.5
    changed = np.abs(post - pre) > 1.0e-8
    metrics: Dict[str, Any] = {
        "schema": 2,
        "slope_units": "one-eye parallax derivative per source-U",
        "vertical_shear_units": (
            "horizontal source-image disparity pixels per vertical source-image pixel; "
            "aspect-matched depth/source grid"),
        "pre_limiter_horizontal_slope": slope_summary(pre_slope),
        "post_limiter_horizontal_slope": slope_summary(post_slope),
        "post_limiter_slope_violation_fraction": float(
            np.mean(post_slope > result.diagnostics.max_horizontal_slope + 1.0e-6)
        ) if post_slope.size else 0.0,
        "pre_limiter_vertical_shear": slope_summary(pre_vertical_shear),
        "post_vertical_stage_vertical_shear": slope_summary(post_vertical_stage_shear),
        "post_limiter_vertical_shear": slope_summary(post_vertical_shear),
        "post_limiter_vertical_shear_violation_fraction": float(
            np.mean(post_vertical_shear > result.diagnostics.max_vertical_shear + 1.0e-6)
        ) if post_vertical_shear.size else 0.0,
        "changed_fraction": float(np.mean(changed)),
        "near_changed_fraction": float(np.mean(changed[near])) if np.any(near) else 0.0,
    }
    if np.count_nonzero(near) >= 2:
        pre_near = pre[near]
        post_near = post[near]
        pre_span = float(np.quantile(pre_near, 0.95) - np.quantile(pre_near, 0.05))
        post_span = float(np.quantile(post_near, 0.95) - np.quantile(post_near, 0.05))
        metrics["near_p05_p95_span_pre"] = pre_span
        metrics["near_p05_p95_span_post"] = post_span
        metrics["near_relief_retention"] = post_span / pre_span if pre_span > 1.0e-12 else None
    else:
        metrics["near_p05_p95_span_pre"] = None
        metrics["near_p05_p95_span_post"] = None
        metrics["near_relief_retention"] = None
    return metrics


def _run_checked(command: list[str], cwd: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("SUNSHINE_SBS_BENCH", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot run {' '.join(command)}: {exc}") from exc
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n{tail}")


def _build_parser() -> argparse.ArgumentParser:
    defaults = MappingV2Config()
    parser = argparse.ArgumentParser(
        description=("Recompute DAV2 mapping from raw depth and render it through Apollo's D3D "
                     "SBS warp (not an exact captured-state replay)"))
    parser.add_argument("--dump", required=True, type=Path, help="Dump 3D package")
    parser.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--build-dir", type=Path,
                        default=REPO / "cmake-build-relwithdebinfo")
    parser.add_argument("--conf", type=Path, default=SCRIPT_DIR / "bench.conf")
    parser.add_argument(
        "--harness-model", default="depth_anything_v2_fp16",
        help=("model used by the harness' mandatory legacy estimator; it does not attest or "
              "regenerate the dump's raw-depth geometry"))
    parser.add_argument(
        "--allow-unverified-model-provenance", action="store_true",
        help=("permit exact replay of raw_depth.f32 when the capture did not record a bound "
              "ONNX SHA-256; the report remains explicitly unverified"))
    parser.add_argument(
        "--experimental-raw-coordinate-scale", type=_positive_float,
        help=("experiment-only fixed raw-coordinate scale for an unverified legacy dump; "
              "legacy dumps never inherit the calibrated DAV2 Small value"))
    parser.add_argument("--pop-strength", type=float, default=defaults.pop_strength,
                        help=("requested artistic maximum; mapped once through the calibrated "
                              f"{defaults.gain_per_pop:g} source-U gain per pop unit"))
    parser.add_argument("--far-tau", type=float, default=defaults.far_tau)
    parser.add_argument("--near-log-tau", type=float, default=defaults.near_log_tau)
    parser.add_argument("--max-horizontal-slope", type=float,
                        default=defaults.max_horizontal_slope)
    parser.add_argument(
        "--max-vertical-shear", type=float, default=defaults.max_vertical_shear,
        help=("maximum horizontal disparity-pixel change per vertical source pixel on the "
              "calibrated aspect-matched grid"))
    parser.add_argument("--skip-score", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dump = args.dump.resolve()
    output = args.out.resolve()
    if not dump.is_dir():
        parser.error(f"dump is not a directory: {dump}")
    source = dump / "source.png"
    if not source.is_file():
        parser.error(f"dump is missing source.png: {dump}")
    try:
        model_provenance = raw_model_provenance.inspect_dump(dump)
        shadow_state_capture = _inspect_optional_shadow_state(dump)
        raw_model_provenance.require_authoritative(
            model_provenance, args.allow_unverified_model_provenance)
        dump_manifest_capture = (
            _inspect_optional_v2_dump_manifest(dump)
            if model_provenance.authoritative or
               shadow_state_capture.get("status") == "validated"
            else {"status": "not-applicable-unverified-legacy"}
        )
        manifest_active = bool(
            dump_manifest_capture.get("status") == "validated" and
            dump_manifest_capture.get("active"))
        state_captured = shadow_state_capture.get("status") == "validated"
        if manifest_active != state_captured:
            raise ValueError(
                "dump V2 manifest availability disagrees with its shadow-state artifacts")
        raw_coordinate_scale, mapping_calibration = _resolve_mapping_calibration(
            model_provenance, args.experimental_raw_coordinate_scale)
        _new_output_directory(output)
        raw, raw_shape = _load_raw_depth(dump)
        if raw_model_provenance.file_sha256(dump / "raw_depth.f32") != \
                model_provenance.raw_depth_sha256:
            raise ValueError("raw_depth.f32 changed after provenance validation")
        config = MappingV2Config(
            raw_coordinate_scale=raw_coordinate_scale,
            far_tau=args.far_tau,
            near_log_tau=args.near_log_tau,
            pop_strength=args.pop_strength,
            max_horizontal_slope=args.max_horizontal_slope,
            max_vertical_shear=args.max_vertical_shear,
        )
        result = generate_depth_mapping_v2(raw, config)
        encoded = encode_direct_parallax(result.parallax)
        canonical = result.canonical.astype("<f4")
        if not np.isfinite(canonical).all():
            raise ValueError("canonical ordering field is not finite float32")

        frames = output / "frames"
        direct_root = output / "direct_parallax"
        direct_clip = direct_root / frames.name
        harness = output / "harness"
        frames.mkdir()
        direct_clip.mkdir(parents=True)
        harness.mkdir()
        shutil.copyfile(source, frames / "frame_00001.png")
        (frames / "meta.json").write_text(
            json.dumps({
                "name": "mapping-v2-dump-replay",
                "description": "Single Dump 3D witness replayed through direct final parallax.",
                "content_type": "unclassified",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        encoded.tofile(direct_clip / "parallax_00001.f32")
        canonical.tofile(direct_clip / "order_00001.f32")
        canonical.tofile(output / "canonical.f32")
        result.desired_parallax.astype("<f4").tofile(output / "desired_parallax.f32")
        result.pre_limiter_parallax.astype("<f4").tofile(
            output / "pre_limiter_parallax.f32")
        result.post_vertical_parallax.astype("<f4").tofile(
            output / "post_vertical_parallax.f32")
        result.parallax.astype("<f4").tofile(output / "conditioned_parallax.f32")

        report = {
            "schema": 7,
            "experiment": "depth-mapping-v2-raw-recomputed-direct-parallax-experiment",
            "captured_state_replay": False,
            "captured_state_limitation": (
                "shot-latched center and temporal state are recomputed, not replayed; exact "
                "capture replay must use the authenticated shadow_final_parallax field"),
            "dump": str(dump),
            "raw_shape": raw_shape,
            "source_geometry": {
                "authority": "exact-raw-depth-input-to-non-captured-state-recomputation",
                "raw_depth_file": "raw_depth.f32",
                "raw_depth_sha256": model_provenance.raw_depth_sha256,
                "raw_shape_file": "raw_shape.json",
                "raw_shape_sha256": raw_model_provenance.file_sha256(
                    dump / "raw_shape.json"),
                "model_provenance_status": model_provenance.status,
            },
            "raw_model_provenance": model_provenance.to_dict(),
            "shadow_state_capture": shadow_state_capture,
            "dump_manifest_capture": dump_manifest_capture,
            "mapping_calibration": mapping_calibration,
            "harness_legacy_estimator": {
                "model": args.harness_model,
                "geometry_role": (
                    "none; direct canonical order and conditioned parallax replace its geometry"
                ),
                "provenance_role": "none",
            },
            "color_replay": (
                "dump source.png diagnostic preview; deterministic raw-depth recomputation and "
                "real D3D warp, but not captured-state or HDR color-fidelity replay"
            ),
            "config": vars(config),
            "diagnostics": result.diagnostics.to_dict(),
            "mapping_metrics": _mapping_metrics(result),
        }
        (output / "mapping_v2_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        executable = args.build_dir.resolve() / "sunshine.exe"
        if not executable.is_file() or not args.conf.resolve().is_file():
            raise ValueError("build-dir must contain sunshine.exe and --conf must be readable")
        harness_command = [
            str(executable), str(args.conf.resolve()), "--sbs-bench",
            "--frames", str(frames), "--out", str(harness),
            "--model", args.harness_model,
            "--direct-parallax-root", str(direct_root),
        ]
        _run_checked(harness_command, args.build_dir.resolve(), output / "harness.log")

        contract = _read_json(harness / "contract.json")
        _validate_direct_replay_contract(
            harness, contract, direct_clip / "parallax_00001.f32",
            direct_clip / "order_00001.f32",
            raw.shape[1], raw.shape[0])

        if not args.skip_score:
            score_command = [
                sys.executable, str(SCRIPT_DIR / "sbsbench.py"),
                "--seq", str(harness), "--frames", str(frames),
                "--json", str(output / "scorecard.json"),
            ]
            _run_checked(score_command, REPO, output / "score.log")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))

    print(
        f"wrote raw-recomputed mapping-v2 experiment to {output} "
        f"(raw model provenance: {model_provenance.status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

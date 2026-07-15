#!/usr/bin/env python3
"""Generate provenance-checked Apollo depth maps for a stereo training suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_semantic_file_hash(path: Path):
    """Match run_eval.py's configuration identity exactly."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest = hashlib.sha256()
    digest.update(path.name.encode())
    digest.update(data)
    return digest.hexdigest()[:16]


def frame_ids(root: Path, prefix: str):
    result = set()
    for path in root.glob(f"{prefix}*"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            result.add(int(suffix))
    return result


def source_fingerprint(source: Path):
    frames = []
    for path in sorted(source.glob("frame_*.png")):
        frames.append({"name": path.name, "sha256": sha256(path)})
    return hashlib.sha256(
        json.dumps(frames, sort_keys=True).encode("utf-8")
    ).hexdigest()


def valid_completed_clip(source: Path, output: Path, model: str,
                         executable_sha256=None, conf_sha256=None):
    contract_path = output / "contract.json"
    identity_path = output / "generation_identity.json"
    if not contract_path.is_file() or not identity_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    source_ids = frame_ids(source, "frame_")
    output_ids = frame_ids(output, "depth_")
    disparity_ids = frame_ids(output, "baseline_disparity_")
    unclamped_disparity_ids = frame_ids(
        output, "baseline_unclamped_disparity_"
    )
    return (
        contract.get("model") == model
        and contract.get("schema") == 24
        and contract.get("artifact_mode") == "depth+baseline-disparity"
        and contract.get("depth_step") == "current-once"
        and contract.get("artistic_policy") is False
        and contract.get("artistic_policy_consumed") is False
        and contract.get("artistic_policy_authorization") == "none"
        and contract.get("model_onnx_sha256") == ""
        and contract.get("policy_metadata_sha256") == ""
        and contract.get("deployment_geometry_allowlist_sha256") == ""
        and float(contract.get("artistic_scale_override", 0.0)) == 0.0
        and contract.get("warp_disparity") ==
        "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"
        and contract.get("warp_unclamped_disparity") ==
        "unclamped_full_binocular_normalized_at_artistic_scale_1_"
        "output_eye_raster_zero_bars"
        and contract.get("artistic_disparity_contract") ==
        "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
        "times_aspect_scale_times_content_scale_x)"
        and int(contract.get("source_width", 0)) > 0
        and int(contract.get("source_height", 0)) > 0
        and int(contract.get("eye_width", 0)) > 0
        and int(contract.get("eye_height", 0)) > 0
        and int(contract.get("disparity_raster_width", 0)) > 0
        and int(contract.get("disparity_raster_height", 0)) > 0
        and int(contract["disparity_raster_width"]) == int(contract["eye_width"])
        and int(contract["disparity_raster_height"]) == int(contract["eye_height"])
        and isinstance(contract.get("policy_warp_source_sha256"), str)
        and len(contract["policy_warp_source_sha256"]) == 64
        and float(contract.get("artistic_full_clamp_abs", 0.0)) > 0.0
        and source_ids == output_ids == disparity_ids == unclamped_disparity_ids
        and identity.get("schema") == 2
        and identity.get("source_sha256") == source_fingerprint(source)
        and (executable_sha256 is None or
             identity.get("executable_sha256") == executable_sha256)
        and (conf_sha256 is None or identity.get("conf_sha256") == conf_sha256)
    )


def generate(suite: Path, output: Path, executable: Path, conf: Path,
             model: str, timeout: int, resume: bool):
    suite_manifest_path = suite / "dataset_manifest.json"
    if not suite_manifest_path.is_file():
        raise RuntimeError(f"missing dataset manifest: {suite_manifest_path}")
    suite_manifest = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    sequences = suite_manifest.get("sequences", [])
    if not sequences and suite_manifest.get("shots"):
        # Schema-1 movie manifests written before sequences became part of the
        # common depth-run contract can be consumed without re-extracting video.
        domain = suite_manifest.get("domain")
        if not domain:
            raise RuntimeError("movie manifest has shots but no domain")
        sequences = [
            {
                "clip": f"{domain}_shot_{int(row['shot']):04d}",
                "frames": row.get("samples"),
                "split": row.get("split"),
            }
            for row in suite_manifest["shots"]
        ]
    if not sequences:
        raise RuntimeError("dataset manifest contains no sequences or movie shots")

    results = []
    policy_hashes = set()
    metric_hashes = set()
    executable_sha = sha256(executable)
    conf_sha = eval_semantic_file_hash(conf)
    for row in sequences:
        clip_name = row["clip"]
        source = suite / clip_name
        destination = output / clip_name
        if resume and valid_completed_clip(
                source, destination, model, executable_sha, conf_sha):
            print(f"[{clip_name}] reuse", flush=True)
            status = "reused"
        else:
            shutil.rmtree(destination, ignore_errors=True)
            command = [
                str(executable.resolve()), str(conf.resolve()), "--sbs-bench",
                "--frames", str(source.resolve()), "--out", str(destination.resolve()),
                "--model", model, "--depth-only",
                "--no-artistic-policy",
            ]
            print(f"[{clip_name}] depth", flush=True)
            try:
                process = subprocess.run(
                    command, cwd=executable.parent, capture_output=True, text=True,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"{clip_name}: harness timed out") from error
            if process.returncode != 0:
                tail = (process.stdout + process.stderr)[-4000:]
                raise RuntimeError(
                    f"{clip_name}: harness failed ({process.returncode})\n{tail}"
                )
            identity = {
                "schema": 2,
                "source_sha256": source_fingerprint(source),
                "executable_sha256": executable_sha,
                "conf_sha256": conf_sha,
                "model": model,
            }
            (destination / "generation_identity.json").write_text(
                json.dumps(identity, indent=2) + "\n", encoding="utf-8"
            )
            if not valid_completed_clip(
                    source, destination, model, executable_sha, conf_sha):
                raise RuntimeError(f"{clip_name}: incomplete or mismatched depth output")
            status = "generated"
        clip_contract = json.loads(
            (destination / "contract.json").read_text(encoding="utf-8")
        )
        policy_hashes.add(clip_contract["policy_warp_source_sha256"])
        metric_hashes.add(clip_contract["metric_sha256"])
        results.append({
            "clip": clip_name,
            "frames": len(frame_ids(destination, "depth_")),
            "status": status,
            "contract_sha256": sha256(destination / "contract.json"),
        })

    if len(policy_hashes) != 1 or len(metric_hashes) != 1:
        raise RuntimeError("depth run mixed policy-warp or metric contracts")

    manifest = {
        "schema": 2,
        "purpose": "artistic-policy depth supervision",
        "artifact_contract": {
            "exact": "baseline_disparity_<frame>.f32",
            "unclamped_scale_1": (
                "baseline_unclamped_disparity_<frame>.f32"
            ),
            "scaled": (
                "clamp(unclamped_scale_1 * artistic_scale, comfort_limit)"
            ),
            "clamp_and_source_geometry": "per-clip contract.json",
        },
        "suite": str(suite.resolve()),
        "suite_manifest_sha256": sha256(suite_manifest_path),
        "executable": str(executable.resolve()),
        "executable_sha256": executable_sha,
        "conf": str(conf.resolve()),
        "conf_sha256": conf_sha,
        "model": model,
        "policy_warp_source_sha256": next(iter(policy_hashes)),
        "metric_sha256": next(iter(metric_hashes)),
        "clips": results,
        "clip_count": len(results),
        "frame_count": sum(row["frames"] for row in results),
    }
    (output / "depth_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--build-dir", type=Path, default=repo / "cmake-build-relwithdebinfo"
    )
    parser.add_argument("--conf", type=Path,
                        default=repo / "tools" / "sbsbench" / "bench.conf")
    parser.add_argument("--model", default="depth_anything_v2_fp16")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    executable = args.build_dir / "sunshine.exe"
    for path, description in (
        (executable, "benchmark executable"), (args.conf, "benchmark config")
    ):
        if not path.is_file():
            raise RuntimeError(f"missing {description}: {path}")
    manifest = generate(
        args.suite, args.output, executable, args.conf, args.model,
        args.timeout, not args.no_resume
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

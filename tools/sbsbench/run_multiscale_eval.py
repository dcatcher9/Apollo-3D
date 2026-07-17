#!/usr/bin/env python3
"""Render and score one clip's ordinal scale grid with one DA-V2 sequence.

This offline-only driver keeps peak storage bounded to one clip.  It publishes
the C++ batch byte manifest, asks isolated ``run_eval`` children to score the
authenticated scales with bounded concurrency, then retains sparse visual
evidence at 1.00/1.30/1.50 and deletes the batch only after every scorer has
succeeded.  ``--stage render`` and ``--stage score`` expose that same exact
transaction as two resumable phases so the corpus orchestrator can overlap the
CPU scorer with the next serialized GPU render.  Live streaming never calls
this entry point.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import multiscale_batch  # noqa: E402
import build_clip_hash_manifest as clip_hashes  # noqa: E402
import depth_state_cache  # noqa: E402
import run_eval  # noqa: E402


SCHEMA = 7
CONTRACT = "apollo-artistic-multiscale-eval-driver-v7"
RENDER_IDENTITY_SCHEMA = 2
RENDER_IDENTITY_CONTRACT = "apollo-artistic-render-identity-v2"
RENDER_IDENTITY_FILENAME = "render_identity.json"
ARTIFACT_SCALES = frozenset((1.0, 1.3, 1.5))
CHILD_SCORE_WORKERS = 1
DEFAULT_SCALE_SCORE_JOBS = 8
WINDOWS_LEGACY_PATH_LIMIT = 259
LONGEST_BATCH_ARTIFACT = "warp_unclamped_disparity_4294967295.f32"


def canonical_bytes(value):
    return (json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _path_independent_clip_content_rows(clip_dir, clip, provenance):
    """Return authenticated path-independent file rows and their source mode."""
    manifest_path = provenance.get("clip_hash_manifest")
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            files = manifest["clips"][clip]["files"]
        except (KeyError, OSError, TypeError, UnicodeError,
                json.JSONDecodeError) as error:
            raise RuntimeError("cannot resolve clip content identity") from error
        mode = "frozen-clip-manifest-sha256-list"
    else:
        files = [{
            "path": path.relative_to(clip_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": multiscale_batch.sha256_file(path),
        } for path in sorted(Path(clip_dir).rglob("*")) if path.is_file()]
        mode = "direct-content-sha256-list"
    normalized = []
    for row in files:
        if not isinstance(row, dict):
            raise RuntimeError("clip content identity row is invalid")
        relative = str(row.get("path", "")).replace("\\", "/")
        sha256 = row.get("sha256")
        size = row.get("size")
        if (not relative or relative.startswith("/") or
                any(part in {"", ".", ".."} for part in relative.split("/")) or
                not isinstance(size, int) or isinstance(size, bool) or size < 0 or
                not isinstance(sha256, str) or
                not re.fullmatch(r"[0-9a-f]{64}", sha256)):
            raise RuntimeError("clip content identity row is invalid")
        normalized.append({
            "path": relative,
            "size": size,
            "sha256": sha256,
        })
    if not normalized:
        raise RuntimeError("clip content identity is empty")
    normalized.sort(key=lambda row: row["path"])
    if len({row["path"] for row in normalized}) != len(normalized):
        raise RuntimeError("clip content identity repeats a path")
    return mode, normalized


def _clip_content_identity(mode, normalized):
    return {
        "contract": "apollo-path-independent-clip-content-v1",
        "source": mode,
        "file_count": len(normalized),
        "content_sha256": canonical_sha256(normalized),
    }


def _path_independent_clip_content_identity(clip_dir, clip, provenance):
    """Bind source bytes without embedding dataset/workspace absolute paths."""
    return _clip_content_identity(
        *_path_independent_clip_content_rows(clip_dir, clip, provenance)
    )


def _render_identity(*, clip_content, clip_sha1, executable_sha256,
                     conf_sha256, metric_sha256, model, extra,
                     output_selection, scales,
                     depth_state_identity_sha256):
    if (not isinstance(depth_state_identity_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", depth_state_identity_sha256) is None):
        raise RuntimeError("render identity lacks an authenticated depth state")
    inputs = {
        "driver": {"schema": SCHEMA, "contract": CONTRACT},
        "harness": {
            "schema": multiscale_batch.HARNESS_SCHEMA,
            "contract": multiscale_batch.HARNESS_CONTRACT,
            "artifact_writer_contract":
                multiscale_batch.ARTIFACT_WRITER_CONTRACT,
        },
        "clip_content": clip_content,
        "clip_sha1": clip_sha1,
        "executable_sha256": executable_sha256,
        "conf_sha256": conf_sha256,
        "metric_sha256": metric_sha256,
        "model": model,
        "depth_state_identity_sha256": depth_state_identity_sha256,
        "extra": list(extra),
        "source_frame_ids": output_selection["source_frame_ids"],
        "label_frame_ids": output_selection["label_frame_ids"],
        "output_selected_frame_ids": output_selection["output_frame_ids"],
        "output_selection_mode": output_selection["mode"],
        "output_label_frames_sha256":
            output_selection["label_frames_sha256"],
        "scale_rows": [{
            "scale": scale,
            "float32_bits": multiscale_batch.scale_float32_bits(scale),
        } for scale in scales],
    }
    return {
        "schema": RENDER_IDENTITY_SCHEMA,
        "contract": RENDER_IDENTITY_CONTRACT,
        "render_identity_sha256": canonical_sha256(inputs),
        "inputs": inputs,
    }


def depth_state_identity_context(*, repo, build_dir, conf_sha256,
                                 executable_sha256, model, clip_dir,
                                 source_content_rows, source_ids,
                                 selected_frame_ids, extra):
    """Build the exact production depth/runtime identity used by rendering."""
    identity_args = {
        "repo": Path(repo).resolve(),
        "build_dir": Path(build_dir).resolve(),
        "conf_sha256": conf_sha256,
        "executable_sha256": executable_sha256,
        "model": model,
        "clip_dir": Path(clip_dir).resolve(),
        "source_content_rows": source_content_rows,
        "source_ids": list(source_ids),
        "selected_frame_ids": list(selected_frame_ids),
        "extra": list(extra),
    }
    runtime_snapshot = depth_state_cache.runtime_snapshot(build_dir)
    identity = depth_state_cache.identity(
        **identity_args, runtime=runtime_snapshot["identity"]
    )
    return {
        "identity": identity,
        "identity_sha256": depth_state_cache.identity_sha256(identity),
        "identity_args": identity_args,
        "runtime_snapshot": runtime_snapshot,
    }


def _write_render_receipt(batch_clip_root, identity):
    manifest_path = Path(batch_clip_root) / multiscale_batch.MANIFEST
    receipt = {
        **identity,
        "batch_manifest_sha256": multiscale_batch.sha256_file(manifest_path),
    }
    path = Path(batch_clip_root) / RENDER_IDENTITY_FILENAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(receipt))
    os.replace(temporary, path)
    return receipt


def _validate_render_receipt(batch_clip_root, identity):
    path = Path(batch_clip_root) / RENDER_IDENTITY_FILENAME
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read multiscale render identity") from error
    expected = {
        **identity,
        "batch_manifest_sha256": multiscale_batch.sha256_file(
            Path(batch_clip_root) / multiscale_batch.MANIFEST
        ),
    }
    if path.read_bytes() != canonical_bytes(receipt) or receipt != expected:
        raise RuntimeError("multiscale render identity differs")
    return receipt


def _parse_scales(value):
    try:
        scales = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from error
    if not scales or list(scales) != sorted(set(scales)):
        raise argparse.ArgumentTypeError("scales must be unique and strictly increasing")
    try:
        for scale in scales:
            multiscale_batch.scale_slug(scale)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return scales


def _validated_extra(extra):
    """Reject options whose value/ordering is owned by this authenticated driver."""
    forbidden = {
        "--frames", "--out", "--model", "--limit",
        "--artistic-scale-grid", "--artistic-scale-override",
        "--artistic-policy", "--no-artistic-policy",
        "--depth-every", "--output-every", "--output-label-frames",
        "--output-gt-right-only", "--depth-only", "--depth-override-root",
        "--depth-override-all", "--runtime-scene-evidence", "--literal-bestv2",
        "--depth-state-export-root", "--depth-state-replay-root",
        "--depth-state-cache-key", "--depth-state-manifest-sha256",
    }
    for value in extra:
        option = value.split("=", 1)[0]
        if option in forbidden:
            raise RuntimeError(
                f"driver extra args contain driver-owned option {option}"
            )
    return list(extra)


def reverify_depth_state_inputs(expected, identity_args, runtime_snapshot,
                                clips_root, clip, clip_identities,
                                clip_hash_provenance):
    """Revalidate every cache-key input after the harness consumed sources."""
    manifest_path = clip_hash_provenance.get("clip_hash_manifest")
    if manifest_path:
        manifest_sha256 = clip_hash_provenance.get(
            "clip_hash_manifest_sha256"
        )
        clip_hashes.verify_selected_clips(
            manifest_path, clips_root, [clip], full=True
        )
        if multiscale_batch.sha256_file(manifest_path) != manifest_sha256:
            raise RuntimeError(
                "clip hash manifest changed during full source verification"
            )
    run_eval.revalidate_clip_hashes(
        str(clips_root), [clip], clip_identities, clip_hash_provenance, False,
    )
    _mode, current_rows = _path_independent_clip_content_rows(
        identity_args["clip_dir"], clip, clip_hash_provenance,
    )
    if "--native-hdr-scrgb" in identity_args.get("extra", ()):
        # The RGB clip manifest does not authenticate the linear scRGB model
        # sources, so re-read every sidecar and preview before any render or
        # shared depth object is published.
        depth_state_cache.native_hdr_capture.validate_clip(
            identity_args["clip_dir"], full=True,
        )
    current_args = {**identity_args, "source_content_rows": current_rows}
    runtime = depth_state_cache.verify_runtime_snapshot(runtime_snapshot)
    observed = depth_state_cache.identity(
        **current_args,
        runtime=runtime,
    )
    if observed != expected:
        raise RuntimeError(
            "depth-state cache inputs changed while the harness was running"
        )
    return observed


def _invalidate_summary(path):
    """Remove stale success/temporary summaries before mutating run outputs."""
    path = Path(path).resolve()
    temporary = path.with_name(path.name + ".tmp")
    for candidate in (path, temporary):
        if candidate.exists():
            if not candidate.is_file():
                raise RuntimeError(
                    f"multiscale summary path is not a file: {candidate}"
                )
            candidate.unlink()
    return path


def _publish_summary_then_cleanup(summary_path, summary, batch_group_root,
                                  clip):
    """Commit success before treating the authenticated batch as scratch."""
    summary_path = Path(summary_path)
    batch_group_root = Path(batch_group_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(summary))
    os.replace(temporary, summary_path)
    try:
        shutil.rmtree(batch_group_root)
    except OSError as error:
        print(
            f"[multiscale] {clip}: warning: rendered batch cleanup deferred: "
            f"{error}",
            flush=True,
        )


def _resolve_output_selection(clip_dir, selected_label_frames):
    source_by_id = run_eval.sbsbench.indexed_files(
        str(Path(clip_dir) / "frame_*.*"), "frame_"
    )
    source_ids = sorted(source_by_id)
    if not source_ids:
        raise RuntimeError(f"clip has no source frames: {clip_dir}")
    try:
        selection = run_eval.resolve_output_selection(
            str(clip_dir), source_ids, 1, False, selected_label_frames
        )
        return {**selection, "source_frame_ids": source_ids}
    except ValueError as error:
        raise RuntimeError(f"invalid selected-label-frame contract: {clip_dir}") from error


def _harness_selection_args(selected_label_frames):
    return ["--output-label-frames"] if selected_label_frames else []


def _frame_gate_args(selected_label_frames):
    return (["--publish-selected-frame-gates"] if selected_label_frames else
            ["--publish-frame-gates"])


def _copy_visual_evidence(source, destination, clip_dir, scale):
    try:
        frame_ids = run_eval.sbsbench.load_label_frame_ids(str(clip_dir))
    except ValueError as error:
        raise RuntimeError(f"invalid label frame contract: {clip_dir}") from error
    if frame_ids is None:
        raise RuntimeError(f"artifact scale requires label_frames.json: {clip_dir}")
    indexed = {
        (prefix, extension): run_eval.sbsbench.indexed_files(
            str(source / f"{prefix}*{extension}"), prefix
        )
        for prefix, extension in (
            ("sbs_", ".png"), ("warp_mask_", ".png"),
            ("warp_disparity_", ".f32"),
        )
    }
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for frame_id in frame_ids:
        for prefix, extension in (
                ("sbs_", ".png"), ("warp_mask_", ".png"),
                ("warp_disparity_", ".f32")):
            src_value = indexed[(prefix, extension)].get(frame_id)
            if src_value is None:
                raise RuntimeError(
                    f"missing visual evidence for frame {frame_id}: {prefix}*{extension}"
                )
            src = Path(src_value)
            dst = destination / src.name
            shutil.copyfile(src, dst)
            files.append({
                "name": dst.name,
                "sha256": multiscale_batch.sha256_file(dst),
                "size": dst.stat().st_size,
            })
    manifest = {
        "schema": 1,
        "contract": "apollo-ordinal-sparse-visual-evidence-v1",
        "scale": scale,
        "frame_ids": frame_ids,
        "files": sorted(files, key=lambda item: item["name"]),
    }
    (destination / "visual_evidence.json").write_bytes(canonical_bytes(manifest))


def _copy_provenance(batch_clip_root, scale_root, result_root, clip):
    destination = result_root / "multiscale_provenance" / clip
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in (
            batch_clip_root / multiscale_batch.MANIFEST,
            batch_clip_root / multiscale_batch.HARNESS_MANIFEST,
            batch_clip_root / RENDER_IDENTITY_FILENAME,
            scale_root / "contract.json"):
        if not source.is_file():
            raise RuntimeError(f"missing multiscale provenance: {source}")
        shutil.copyfile(source, destination / source.name)
    return {
        path.name: multiscale_batch.sha256_file(path)
        for path in destination.iterdir() if path.is_file()
    }


def _scale_score_command(*, build_dir, conf, clips_root, clip, label,
                         batch_group_root, executable_sha256, extra,
                         selected_label_frames, scale):
    """Build one isolated scale scorer with exactly one metric worker."""
    return [
        sys.executable, str(SCRIPT_DIR / "run_eval.py"),
        "--build-dir", str(build_dir), "--conf", str(conf),
        "--clips-root", str(clips_root), "--clips", clip,
        "--label", label, "--score-workers", str(CHILD_SCORE_WORKERS),
        "--comparison-only", *_frame_gate_args(selected_label_frames),
        "--precomputed-multiscale-root", str(batch_group_root),
        "--prevalidated-current-build-sha256", executable_sha256,
        "--extra", *extra,
        "--depth-every", "1", "--output-every", "1",
        "--no-artistic-policy", "--runtime-scene-evidence",
        *_harness_selection_args(selected_label_frames),
        "--artistic-scale-override", f"{scale:.2f}",
    ]


def _score_scale_child(job, cwd):
    scored = subprocess.run(
        job["command"], cwd=cwd, capture_output=True, text=True,
        timeout=1800,
    )
    if scored.returncode:
        raise RuntimeError(
            f"multiscale scoring failed at {job['scale']:.2f}: " +
            (scored.stdout + scored.stderr)[-4000:]
        )
    return job


def _score_scales(jobs, max_workers, cwd):
    """Score independent scale trees concurrently and return input order.

    Publication is intentionally outside this function.  On failure, queued
    children are cancelled and running children are awaited before the error is
    exposed, so callers can retain the complete batch tree for diagnosis.
    """
    if max_workers < 1:
        raise ValueError("scale score workers must be positive")
    jobs = list(jobs)
    completed = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    try:
        for job in jobs:
            future = executor.submit(_score_scale_child, job, cwd)
            futures[future] = job
        for future in concurrent.futures.as_completed(futures):
            job = future.result()
            completed[job["scale_index"]] = job
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return [completed[job["scale_index"]] for job in jobs]


def _validate_published_batch(root, *, clip, clip_sha1,
                              executable_sha256, conf_sha256,
                              metric_sha256, scales):
    """Validate an existing rendered transaction without re-authenticating it.

    This deliberately checks every exact scale against the already-published
    byte manifest.  A resumed render stage may adopt only an unchanged complete
    transaction; a partial or modified scratch tree is regenerated instead.
    """
    validated = None
    for scale in scales:
        current = multiscale_batch.validate(
            root,
            clip=clip,
            clip_sha1=clip_sha1,
            executable_sha256=executable_sha256,
            conf_sha256=conf_sha256,
            metric_sha256=metric_sha256,
            scale=scale,
        )
        if validated is None:
            validated = current["manifest"]
        elif current["manifest"] != validated:
            raise RuntimeError("multiscale batch manifest changed during validation")
    if validated is None:
        raise RuntimeError("multiscale scale grid is empty")
    return validated


def _selection_matches(manifest, output_selection):
    return (
        manifest["source_frame_ids"] == output_selection["source_frame_ids"] and
        manifest["label_frame_ids"] == output_selection["label_frame_ids"] and
        manifest["output_selected_frame_ids"] ==
        output_selection["output_frame_ids"] and
        manifest["output_selection_mode"] == output_selection["mode"] and
        manifest["output_label_frames_sha256"] ==
        output_selection["label_frames_sha256"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--scales", type=_parse_scales, required=True)
    parser.add_argument(
        "--selected-label-frames", action="store_true",
        help="render only authenticated label_frames.json targets",
    )
    parser.add_argument(
        "--score-workers", type=int, default=DEFAULT_SCALE_SCORE_JOBS,
        help=(
            "maximum concurrent exact-scale scoring jobs; every isolated "
            "run_eval child uses one metric worker"
        ),
    )
    parser.add_argument(
        "--stage", choices=("all", "render", "score"), default="all",
        help=(
            "run the complete transaction, publish only the authenticated "
            "render batch, or score an existing authenticated render batch"
        ),
    )
    parser.add_argument(
        "--depth-state-cache-root", type=Path,
        help=(
            "authenticated immutable cross-workspace cache for the production "
            "pre-warp depth state; requires --depth-state-cache-split"
        ),
    )
    parser.add_argument(
        "--depth-state-cache-split", choices=("training", "development"),
        help="working split authorization for depth-state cache source access",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    conf = args.conf.resolve()
    clips_root = args.clips_root.resolve()
    clip = run_eval.validate_path_component(args.clip, "clip name")
    label_prefix = run_eval.validate_path_component(args.label_prefix, "run label prefix")
    clip_dir = Path(run_eval.contained_component(clips_root, clip, "clip name"))
    if not clip_dir.is_dir():
        raise RuntimeError(f"missing clip: {clip_dir}")
    if args.score_workers < 1:
        raise RuntimeError("--score-workers must be positive")
    if bool(args.depth_state_cache_root) != bool(args.depth_state_cache_split):
        raise RuntimeError(
            "depth-state cache root and working split must be supplied together"
        )
    if args.depth_state_cache_split:
        depth_state_cache.require_working_split(args.depth_state_cache_split)
    extra = _validated_extra(args.extra)
    output_selection = _resolve_output_selection(
        clip_dir, args.selected_label_frames
    )

    executable = build_dir / "sunshine.exe"
    if not executable.is_file() or not conf.is_file():
        raise RuntimeError("missing evaluator executable/config")
    run_eval.require_current_build(str(build_dir))
    executable_sha256 = multiscale_batch.sha256_file(executable)
    conf_sha256 = run_eval.sha256_files([str(conf)])
    metric_sha256 = run_eval.metric_contract_sha()
    profile = run_eval.expected_profile(str(conf), extra)
    model = run_eval.expected_depth_model(str(conf), profile, extra)
    clip_set_sha1, clip_hash_provenance = run_eval.resolve_clip_hashes(
        str(clips_root), [clip], False
    )
    clip_content_mode, clip_content_rows = _path_independent_clip_content_rows(
        clip_dir, clip, clip_hash_provenance,
    )
    clip_content = _clip_content_identity(clip_content_mode, clip_content_rows)
    depth_context = depth_state_identity_context(
        repo=SCRIPT_DIR.parents[1],
        build_dir=build_dir,
        conf_sha256=conf_sha256,
        executable_sha256=executable_sha256,
        model=model,
        clip_dir=clip_dir,
        source_content_rows=clip_content_rows,
        source_ids=output_selection["source_frame_ids"],
        selected_frame_ids=output_selection["output_frame_ids"],
        extra=extra,
    )
    reverify_depth_state_inputs(
        depth_context["identity"], depth_context["identity_args"],
        depth_context["runtime_snapshot"], clips_root, clip,
        clip_set_sha1, clip_hash_provenance,
    )
    render_identity = _render_identity(
        clip_content=clip_content,
        clip_sha1=clip_set_sha1[clip],
        executable_sha256=executable_sha256,
        conf_sha256=conf_sha256,
        metric_sha256=metric_sha256,
        model=model,
        extra=extra,
        output_selection=output_selection,
        scales=args.scales,
        depth_state_identity_sha256=depth_context["identity_sha256"],
    )

    # A render stage starts a new transaction.  An earlier success summary must
    # not survive output replacement.  Score-only is permitted only when the
    # orchestrator has already established that no completed summary exists.
    summary_path = args.summary.resolve()
    if args.stage in {"all", "render"}:
        summary_path = _invalidate_summary(summary_path)
    elif summary_path.exists():
        raise RuntimeError(
            "score stage refuses an already-published multiscale summary"
        )

    eval_root = build_dir / "sbs_eval"
    batch_group_root = Path(run_eval.contained_component(
        eval_root, f"{label_prefix}-batch", "multiscale batch label"
    ))
    batch_clip_root = batch_group_root / clip
    longest_output = (
        batch_clip_root / "scales" /
        multiscale_batch.scale_slug(max(args.scales)) /
        LONGEST_BATCH_ARTIFACT
    )
    if os.name == "nt" and len(str(longest_output)) > WINDOWS_LEGACY_PATH_LIMIT:
        raise RuntimeError(
            "multiscale output exceeds the legacy Windows path budget: "
            f"{longest_output}"
        )
    manifest = None
    if args.stage in {"all", "render"}:
        adopted = False
        if args.stage == "render" and batch_group_root.exists():
            try:
                manifest = _validate_published_batch(
                    batch_clip_root,
                    clip=clip,
                    clip_sha1=clip_set_sha1[clip],
                    executable_sha256=executable_sha256,
                    conf_sha256=conf_sha256,
                    metric_sha256=metric_sha256,
                    scales=args.scales,
                )
                _validate_render_receipt(batch_clip_root, render_identity)
                adopted = True
                print(
                    f"[multiscale] {clip}: adopting authenticated rendered batch",
                    flush=True,
                )
            except (OSError, RuntimeError, ValueError) as error:
                print(
                    f"[multiscale] {clip}: replacing incomplete rendered batch: "
                    f"{error}",
                    flush=True,
                )
                shutil.rmtree(batch_group_root)
        elif batch_group_root.exists():
            shutil.rmtree(batch_group_root)

        if not adopted:
            batch_clip_root.mkdir(parents=True)
            cache_store = None
            cache_identity = depth_context["identity"]
            cache_identity_args = depth_context["identity_args"]
            cache_runtime_snapshot = depth_context["runtime_snapshot"]
            cache_export = None
            cache_args = []
            if args.depth_state_cache_root:
                cache_store = depth_state_cache.cache(
                    args.depth_state_cache_root.resolve()
                )
                cache_key = depth_context["identity_sha256"]
                cache_hit = depth_state_cache.validated_sequence(
                    cache_store, cache_identity
                )
                if cache_hit is None:
                    cache_export = batch_clip_root / ".depth-state-export"
                    cache_args = [
                        "--depth-state-export-root", str(cache_export),
                        "--depth-state-cache-key", cache_key,
                    ]
                    print(
                        f"[multiscale] {clip}: depth-state cache miss "
                        f"{cache_key[:12]}; exporting selected state",
                        flush=True,
                    )
                else:
                    cache_args = [
                        "--depth-state-replay-root", str(cache_hit["payload"]),
                        "--depth-state-cache-key", cache_key,
                        "--depth-state-manifest-sha256",
                        cache_hit["inner_manifest_sha256"],
                    ]
                    print(
                        f"[multiscale] {clip}: authenticated depth-state cache hit "
                        f"{cache_key[:12]}",
                        flush=True,
                    )
            command = [
                str(executable), str(conf), "--sbs-bench",
                *extra,
                "--frames", str(clip_dir), "--out", str(batch_clip_root),
                "--model", model, "--depth-every", "1", "--output-every", "1",
                "--no-artistic-policy", "--runtime-scene-evidence",
                *_harness_selection_args(args.selected_label_frames),
                "--artistic-scale-grid", ",".join(
                    f"{scale:.2f}" for scale in args.scales
                ),
                *cache_args,
            ]
            print(
                f"[multiscale] {clip}: advancing "
                f"{len(output_selection['source_frame_ids'])} source frames; "
                f"rendering {len(output_selection['output_frame_ids'])} selected "
                f"frames at {len(args.scales)} exact scales",
                flush=True,
            )
            result = subprocess.run(
                command, cwd=build_dir, capture_output=True, text=True,
                timeout=1800,
            )
            if result.returncode:
                raise RuntimeError(
                    "multiscale harness failed: " +
                    (result.stdout + result.stderr)[-4000:]
                )
            try:
                reverify_depth_state_inputs(
                    cache_identity, cache_identity_args,
                    cache_runtime_snapshot, clips_root, clip,
                    clip_set_sha1, clip_hash_provenance,
                )
            except (OSError, RuntimeError, ValueError) as error:
                # This label-specific batch root was deleted/adopted above,
                # then created by this transaction. Never touch a shared cache
                # object or any neighboring evaluator output here.
                if batch_group_root.exists():
                    shutil.rmtree(batch_group_root)
                raise RuntimeError(
                    "discarded owned multiscale render transaction because "
                    "depth-state inputs changed"
                ) from error
            if cache_export is not None:
                assert cache_store is not None and cache_identity is not None
                cache_store.publish(cache_identity, cache_export)
                if depth_state_cache.validated_sequence(
                        cache_store, cache_identity) is None:
                    raise RuntimeError("depth-state cache publication disappeared")
                shutil.rmtree(cache_export)
            print(
                f"[multiscale] {clip}: inference/render batch complete",
                flush=True,
            )
            manifest = multiscale_batch.publish(
                batch_clip_root,
                clip=clip,
                clip_sha1=clip_set_sha1[clip],
                executable_sha256=executable_sha256,
                conf_sha256=conf_sha256,
                metric_sha256=metric_sha256,
                scales=args.scales,
            )
            _write_render_receipt(batch_clip_root, render_identity)
    else:
        if not batch_group_root.is_dir():
            raise RuntimeError(
                f"score stage requires an authenticated rendered batch: "
                f"{batch_group_root}"
            )
        # The isolated scale scorers validate every referenced artifact against
        # this manifest before measurement.  Validate one scale here as an
        # early identity/selection check; the complete grid must succeed before
        # any publication or deletion below.
        manifest = multiscale_batch.validate(
            batch_clip_root,
            clip=clip,
            clip_sha1=clip_set_sha1[clip],
            executable_sha256=executable_sha256,
            conf_sha256=conf_sha256,
            metric_sha256=metric_sha256,
            scale=args.scales[0],
        )["manifest"]
        _validate_render_receipt(batch_clip_root, render_identity)

    assert manifest is not None
    if not _selection_matches(manifest, output_selection):
        raise RuntimeError("multiscale harness selected-frame identity differs")
    manifest_sha256 = multiscale_batch.sha256_file(
        batch_clip_root / multiscale_batch.MANIFEST
    )
    render_receipt_sha256 = multiscale_batch.sha256_file(
        batch_clip_root / RENDER_IDENTITY_FILENAME
    )
    if args.stage == "render":
        print(
            f"[multiscale] {clip}: authenticated render batch ready",
            flush=True,
        )
        return

    scale_jobs = []
    for scale_index, scale in enumerate(args.scales, 1):
        slug = multiscale_batch.scale_slug(scale)
        label = f"{label_prefix}-{slug}"
        scale_jobs.append({
            "scale_index": scale_index,
            "scale": scale,
            "slug": slug,
            "label": label,
            "command": _scale_score_command(
                build_dir=build_dir,
                conf=conf,
                clips_root=clips_root,
                clip=clip,
                label=label,
                batch_group_root=batch_group_root,
                executable_sha256=executable_sha256,
                extra=extra,
                selected_label_frames=args.selected_label_frames,
                scale=scale,
            ),
        })
    print(
        f"[multiscale] {clip}: scoring {len(scale_jobs)} exact scales "
        f"with up to {args.score_workers} concurrent jobs",
        flush=True,
    )
    scored_jobs = _score_scales(
        scale_jobs, args.score_workers, SCRIPT_DIR.parents[1]
    )
    print(
        f"[multiscale] {clip}: every scale scorer completed; "
        "authenticating publications",
        flush=True,
    )

    # Phase two starts only after every child scorer succeeds.  First validate
    # the complete result grid without mutating the shared batch tree.
    publications = []
    for job in scored_jobs:
        scale = job["scale"]
        slug = job["slug"]
        label = job["label"]
        result_root = eval_root / label
        results_path = result_root / "results.json"
        gates_path = result_root / run_eval.FRAME_GATE_EVIDENCE_FILENAME
        if not results_path.is_file() or not gates_path.is_file():
            raise RuntimeError(f"multiscale scale publication is incomplete: {label}")
        scale_root = batch_clip_root / "scales" / slug
        publications.append({
            **job,
            "result_root": result_root,
            "results_path": results_path,
            "gates_path": gates_path,
            "scale_root": scale_root,
        })

    scale_results = []
    for publication in publications:
        scale = publication["scale"]
        slug = publication["slug"]
        result_root = publication["result_root"]
        results_path = publication["results_path"]
        gates_path = publication["gates_path"]
        scale_root = publication["scale_root"]
        provenance = _copy_provenance(
            batch_clip_root, scale_root, result_root, clip
        )
        if scale in ARTIFACT_SCALES:
            _copy_visual_evidence(
                scale_root,
                result_root / "artifact_evidence" / clip,
                clip_dir,
                scale,
            )
        scale_results.append({
            "scale": scale,
            "scale_slug": slug,
            "run": str(result_root.resolve()),
            "results_sha256": multiscale_batch.sha256_file(results_path),
            "frame_gate_evidence_sha256":
                multiscale_batch.sha256_file(gates_path),
            "provenance_sha256": provenance,
        })
    print(f"[multiscale] {clip}: all scales authenticated", flush=True)

    # Every durable scale result carries a copy of the manifest, harness
    # contract, exact scale contract, runtime scene evidence, and optional
    # visual evidence.  Keep the authenticated render transaction intact until
    # the success summary itself is atomically durable.  A crash anywhere
    # before that point must remain resumable without another inference pass.
    summary = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "clip": clip,
        "clip_sha1": clip_set_sha1[clip],
        "executable_sha256": executable_sha256,
        "conf_sha256": conf_sha256,
        "metric_sha256": metric_sha256,
        "batch_manifest_sha256": manifest_sha256,
        "render_identity_sha256":
            render_identity["render_identity_sha256"],
        "render_identity_receipt_sha256": render_receipt_sha256,
        "source_frame_ids": manifest["source_frame_ids"],
        "label_frame_ids": manifest["label_frame_ids"],
        "output_selected_frame_ids": manifest["output_selected_frame_ids"],
        "output_selection_mode": manifest["output_selection_mode"],
        "output_label_frames_sha256":
            manifest["output_label_frames_sha256"],
        "scale_score_jobs": args.score_workers,
        "child_score_workers": CHILD_SCORE_WORKERS,
        "scale_results": scale_results,
        "rendered_batch_retained_until_success_summary": True,
        "batch_cleanup_policy": "best-effort-after-success-summary",
    }
    _publish_summary_then_cleanup(
        summary_path, summary, batch_group_root, clip
    )
    print(summary_path)


if __name__ == "__main__":
    main()

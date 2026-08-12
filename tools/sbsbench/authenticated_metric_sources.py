"""Shared fail-closed discovery for authenticated-source metric falsification suites.

This module intentionally contains no quality metric.  Validators reuse one deterministic clip,
provenance, frame-selection, resize, and source-hash contract instead of importing a rejected
detector suite merely for its I/O helpers.
"""

import glob
import hashlib
import json
import os
import re

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLIPS_ROOT = os.path.join(SCRIPT_DIR, "clips")
DATASET_MANIFEST = os.path.join(SCRIPT_DIR, "datasets", "manifest.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _frame_files(clip_dir):
    return sorted(
        path for path in (
            os.path.join(clip_dir, name) for name in os.listdir(clip_dir)
        )
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        and os.path.basename(path).lower().startswith("frame_")
    )


def _frame_ids(paths, label):
    """Return numeric frame IDs while rejecting malformed or duplicate identities."""
    ids = set()
    pattern = re.compile(r"^frame_(\d+)\.[^.]+$", re.IGNORECASE)
    for path in paths:
        match = pattern.match(os.path.basename(path))
        if match is None:
            raise ValueError(f"malformed {label} frame name: {path}")
        frame_id = int(match.group(1))
        if frame_id in ids:
            raise ValueError(f"duplicate {label} frame id {frame_id}: {path}")
        ids.add(frame_id)
    return ids


def _load_binary_mask(path, shape):
    """Load an authenticated source-sized 8-bit L PNG containing only 0/255."""
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(
                f"mask must be an 8-bit single-channel PNG, got "
                f"format={image.format!r}, mode={image.mode!r}: {path}")
        values = np.asarray(image, dtype=np.uint8)
    if values.shape != shape:
        raise ValueError(
            f"mask dimensions {values.shape} do not match source {shape}: {path}")
    if not np.all((values == 0) | (values == 255)):
        raise ValueError(f"mask contains values other than 0/255: {path}")
    return values == 255


def discover_clips(roots):
    """Find direct child clip directories with authenticated metadata and source frames."""
    clips = []
    seen_ids = set()
    for requested_root in roots:
        root = os.path.abspath(requested_root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"clip root does not exist: {root}")
        root_clip_count = 0
        for name in sorted(os.listdir(root)):
            clip_dir = os.path.join(root, name)
            meta_path = os.path.join(clip_dir, "meta.json")
            if not os.path.isdir(clip_dir) or not os.path.isfile(meta_path):
                continue
            frames = _frame_files(clip_dir)
            if not frames:
                continue
            with open(meta_path, encoding="utf-8") as stream:
                meta = json.load(stream)
            if not meta.get("name"):
                raise ValueError(f"unauthenticated clip {clip_dir}: missing name metadata")
            if "required_gt_subtitle_region" in meta and not isinstance(
                    meta["required_gt_subtitle_region"], bool):
                raise ValueError(
                    f"unauthenticated clip {clip_dir}: "
                    "required_gt_subtitle_region must be boolean")
            if ("required_gt_subtitle_tight_mask" in meta and
                    not isinstance(meta["required_gt_subtitle_tight_mask"], bool)):
                raise ValueError(
                    f"unauthenticated clip {clip_dir}: "
                    "required_gt_subtitle_tight_mask must be boolean")
            if meta.get("required_gt_subtitle_tight_mask") is True:
                if meta.get("required_gt_subtitle_region") is not True:
                    raise ValueError(
                        f"unauthenticated clip {clip_dir}: "
                        "required_gt_subtitle_tight_mask requires authored subtitle region")
                source_ids = _frame_ids(frames, "source")
                sidecars_by_name = {}
                for directory in ("gt_subtitle_region", "gt_subtitle_overlay_mask"):
                    sidecars = glob.glob(os.path.join(clip_dir, directory, "frame_*.png"))
                    sidecar_ids = _frame_ids(sidecars, directory)
                    if sidecar_ids != source_ids:
                        missing = sorted(source_ids - sidecar_ids)
                        extra = sorted(sidecar_ids - source_ids)
                        raise ValueError(
                            f"unauthenticated clip {clip_dir}: subtitle tight-mask "
                            f"{directory}/source frame-id mismatch: missing={missing}, "
                            f"extra={extra}")
                    sidecars_by_name[directory] = {
                        int(re.search(r"frame_(\d+)", os.path.basename(path)).group(1)): path
                        for path in sidecars
                    }
                source_by_id = {
                    int(re.search(r"frame_(\d+)", os.path.basename(path)).group(1)): path
                    for path in frames
                }
                for frame_id in sorted(source_ids):
                    with Image.open(source_by_id[frame_id]) as source:
                        source_shape = (source.height, source.width)
                    loose = _load_binary_mask(
                        sidecars_by_name["gt_subtitle_region"][frame_id], source_shape)
                    tight = _load_binary_mask(
                        sidecars_by_name["gt_subtitle_overlay_mask"][frame_id], source_shape)
                    if np.any(tight & ~loose):
                        raise ValueError(
                            f"unauthenticated clip {clip_dir}: tight subtitle overlay mask "
                            f"escapes loose region at frame {frame_id}")
                    if bool(np.any(tight)) != bool(np.any(loose)):
                        raise ValueError(
                            f"unauthenticated clip {clip_dir}: tight/loose subtitle masks "
                            f"disagree on empty state at frame {frame_id}")
            if ("required_gt_subtitle_sanitizer_oracle" in meta and
                    not isinstance(meta["required_gt_subtitle_sanitizer_oracle"], bool)):
                raise ValueError(
                    f"unauthenticated clip {clip_dir}: "
                    "required_gt_subtitle_sanitizer_oracle must be boolean")
            if meta.get("required_gt_subtitle_sanitizer_oracle") is True:
                if (meta.get("required_gt_subtitle_region") is not True or
                        "subtitle_transition_contract" not in meta):
                    raise ValueError(
                        f"unauthenticated clip {clip_dir}: "
                        "required_gt_subtitle_sanitizer_oracle requires authored subtitle "
                        "region and transition contracts")
                source_ids = _frame_ids(frames, "source")
                for directory in ("gt_subtitle_overlay_mask", "gt_subtitle_free"):
                    sidecars = glob.glob(os.path.join(clip_dir, directory, "frame_*.png"))
                    sidecar_ids = _frame_ids(sidecars, directory)
                    if sidecar_ids != source_ids:
                        missing = sorted(source_ids - sidecar_ids)
                        extra = sorted(sidecar_ids - source_ids)
                        raise ValueError(
                            f"unauthenticated clip {clip_dir}: subtitle sanitizer "
                            f"{directory}/source frame-id mismatch: missing={missing}, "
                            f"extra={extra}")
            if meta.get("suite"):
                if "required_gt_stereo" in meta:
                    retired = meta.pop("required_gt_stereo")
                    if not isinstance(retired, bool):
                        raise ValueError(
                            f"unauthenticated extended clip {clip_dir}: retired "
                            "required_gt_stereo must be boolean")
                    if ("reference_stereo_available" in meta and
                            meta["reference_stereo_available"] != retired):
                        raise ValueError(
                            f"unauthenticated extended clip {clip_dir}: conflicting "
                            "retired/current stereo reference declarations")
                    meta["reference_stereo_available"] = retired
                required = ("dataset", "citation", "license_note")
                missing = [key for key in required if not meta.get(key)]
                evidence_keys = (
                    "required_gt_depth", "required_gt_flow",
                    "required_gt_subtitle_region", "required_gt_subtitle_tight_mask",
                )
                has_consumed_gt = any(meta.get(key) is True for key in evidence_keys)
                if (not has_consumed_gt and meta.get("reference_stereo_available") is True and
                        "evaluation_role" not in meta):
                    meta["evaluation_role"] = "reference-only"
                is_reference_only = meta.get("evaluation_role") == "reference-only"
                has_diagnostic_pair = meta.get("reference_stereo_available") is True
                if missing or not (has_consumed_gt or
                                   (is_reference_only and has_diagnostic_pair)):
                    raise ValueError(
                        f"unauthenticated extended clip {clip_dir}: missing provenance "
                        f"{missing}, consumed depth/flow GT or subtitle-region GT, or an "
                        "explicit reference-only pair")
                if is_reference_only and has_consumed_gt:
                    raise ValueError(
                        f"unauthenticated extended clip {clip_dir}: reference-only clips "
                        "cannot declare consumed depth/flow GT or subtitle-region GT")
                reference_patterns = {
                    "required_gt_depth": os.path.join(clip_dir, "gt_depth", "frame_*.*"),
                    "required_gt_flow": os.path.join(clip_dir, "gt_flow", "frame_*.npz"),
                    "required_gt_subtitle_region": os.path.join(
                        clip_dir, "gt_subtitle_region", "frame_*.png"),
                    "required_gt_subtitle_tight_mask": os.path.join(
                        clip_dir, "gt_subtitle_overlay_mask", "frame_*.png"),
                    "reference_stereo_available": os.path.join(
                        clip_dir, "gt_right", "frame_*.*"),
                }
                absent = [key for key, pattern in reference_patterns.items()
                          if meta.get(key) is True and not glob.glob(pattern)]
                if absent:
                    raise ValueError(
                        f"unauthenticated extended clip {clip_dir}: declared reference "
                        f"sidecars are absent for {absent}")
            if name in seen_ids:
                raise ValueError(f"duplicate clip id across roots: {name}")
            seen_ids.add(name)
            clips.append({
                "id": name,
                "directory": clip_dir,
                "frames": frames,
                "meta": meta,
            })
            root_clip_count += 1
        if root_clip_count == 0:
            raise ValueError(
                f"no authenticated frame clips were readable under requested root {root}")
    if not clips:
        raise ValueError(f"no authenticated frame clips found under {roots}")
    return clips


def suite_roots(suite):
    roots = []
    if suite in ("core", "both"):
        roots.append(DEFAULT_CLIPS_ROOT)
    if suite in ("extended", "both"):
        with open(DATASET_MANIFEST, encoding="utf-8") as stream:
            manifest = json.load(stream)
        cache = os.environ.get("APOLLO_SBS_DATASETS") or manifest["default_cache"]
        roots.append(os.path.join(
            os.path.abspath(cache), "prepared", manifest["prepared_suite"]))
    return roots


def deterministic_frame_sample(frames, count):
    """Evenly sample interior frames without filesystem-order dependence."""
    if count < 1:
        raise ValueError("frames_per_clip must be at least 1")
    if count >= len(frames):
        return list(frames)
    indices = []
    for index in range(count):
        position = int(np.floor((index + 1) * len(frames) / (count + 1)))
        position = min(max(position, 0), len(frames) - 1)
        if position not in indices:
            indices.append(position)
    return [frames[index] for index in indices]


def load_frame(path, max_width):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_width and image.width > max_width:
            height = max(32, int(round(image.height * max_width / image.width)))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        value = np.asarray(image, dtype=np.float32) / 255.0
    if value.shape[0] < 32 or value.shape[1] < 64:
        raise ValueError(f"frame is too small after resize: {path} -> {value.shape}")
    return value


def sample_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Private alias retained for existing validator result-schema compatibility.
_sample_sha256 = sample_sha256


__all__ = [
    "DEFAULT_CLIPS_ROOT", "deterministic_frame_sample", "discover_clips", "load_frame",
    "sample_sha256", "suite_roots",
]

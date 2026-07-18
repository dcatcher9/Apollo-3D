"""Shared fail-closed discovery for real-source metric falsification suites.

This module intentionally contains no quality metric.  Validators reuse one deterministic clip,
provenance, frame-selection, resize, and source-hash contract instead of importing a rejected
detector suite merely for its I/O helpers.
"""

import glob
import hashlib
import json
import os

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
                evidence_keys = ("required_gt_depth", "required_gt_flow")
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
                        f"{missing}, consumed depth/flow GT, or an explicit reference-only pair")
                if is_reference_only and has_consumed_gt:
                    raise ValueError(
                        f"unauthenticated extended clip {clip_dir}: reference-only clips "
                        "cannot declare consumed depth/flow GT")
                reference_patterns = {
                    "required_gt_depth": os.path.join(clip_dir, "gt_depth", "frame_*.*"),
                    "required_gt_flow": os.path.join(clip_dir, "gt_flow", "frame_*.npz"),
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

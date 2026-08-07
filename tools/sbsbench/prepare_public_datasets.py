#!/usr/bin/env python3
"""Download and deterministically prepare the external public SBS evaluation suite.

Media is intentionally kept outside Git.  The committed manifest fixes source URLs and frame
windows; this tool turns those archives into the exact ``frame_*``/reference layout consumed by
run_eval.py. Full downloads are resumable and verified when their manifest has a SHA-256 digest.
Range-backed ZIP selections validate every HTTP byte range and ZIP-member CRC, then require a
manifest-pinned SHA-256 of the exact decoded frame/depth/mask evidence consumed by evaluation.
This authenticates selective extraction without downloading unrelated multi-gigabyte members.
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import math
import ntpath
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tarfile
import urllib.request
import zipfile
import struct
import zlib

import numpy as np
from PIL import Image

from depth_coordinate_v2_contract import MODEL_CALIBRATIONS


HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "datasets", "manifest.json")
CONTENT_TYPES = {
    "ai-generated", "anime", "unclassified", "synthetic",
    "real-capture", "animation", "simulation",
}

# Keep the external decision suite on the exact live/offline Host SBS V2 shape contract. Dataset
# pixels may be centered on a larger black canvas, but are never resized or cropped. These values
# mirror host_sbs_v2_live_calibration and the dynamic TensorRT profile; focused tests bind them to
# the native declarations so a production fitter change cannot silently leave preparation stale.
PRODUCTION_DEPTH_SHORT_SIDE = 432
PRODUCTION_DEPTH_MAX_ASPECT = 4.0
PRODUCTION_V2_CALIBRATION = MODEL_CALIBRATIONS[0]
PRODUCTION_DEPTH_PATCH = PRODUCTION_V2_CALIBRATION.preprocess.patch_multiple
PRODUCTION_DEPTH_MAX_DIM = PRODUCTION_V2_CALIBRATION.preprocess.maximum_dimension
PRODUCTION_V2_TENSOR_SHAPES = frozenset(
    PRODUCTION_V2_CALIBRATION.calibrated_input_shapes)
PREPARATION_GEOMETRY_SCHEMA = 1
ADAPTER_SELECTION_CONTRACTS = {
    "tum_rgbd_zip": {
        "source_fields": ("rgb_timestamp", "depth_timestamp"),
        "identity_fields": (),
    },
    "tartanair_v2_zip": {
        "source_fields": ("dataset_frame",),
        "identity_fields": ("trajectory", "camera"),
    },
    "sintel_stereo_zip": {
        "source_fields": ("dataset_frame",),
        "identity_fields": ("sequence", "pass"),
    },
    "spring_http_range_zip": {
        "source_fields": ("dataset_frame",),
        "identity_fields": ("sequence", "split"),
    },
    "vkitti2_tar": {
        "source_fields": ("dataset_frame",),
        "identity_fields": ("scene", "variant", "camera"),
    },
}
ADAPTER_EVIDENCE_EXTENSIONS = {
    "tum_rgbd_zip": {"source": ".png", "depth": ".png"},
    "tartanair_v2_zip": {
        "source": ".png", "depth": ".npy", "flow": ".npz",
    },
    "sintel_stereo_zip": {
        "source": ".png", "depth": ".npy", "stereo": ".png",
    },
    "spring_http_range_zip": {"source": ".png", "stereo": ".png"},
    "vkitti2_tar": {"source": ".png", "depth": ".png"},
}


def fail(message):
    raise RuntimeError(message)


def _positive_manifest_dimension(value, description):
    if type(value) is not int or value <= 0:
        fail(f"{description} must be a positive integer")
    return value


def _round_to_patch(value, patch=PRODUCTION_DEPTH_PATCH):
    """Match C++ std::round for the fitter's positive dimensions."""
    return max(patch, int(math.floor(float(value) / patch + 0.5)) * patch)


def fit_host_sbs_v2_depth_tensor_shape(source_width, source_height):
    """Python mirror of ``fit_host_sbs_v2_depth_tensor_shape`` for dataset preflight."""
    if type(source_width) is not int or type(source_height) is not int:
        return (0, 0)
    if source_width <= 0 or source_height <= 0:
        return (0, 0)

    patch = PRODUCTION_DEPTH_PATCH
    aspect = source_width / source_height
    if aspect >= 1.0:
        fitted_aspect = min(aspect, PRODUCTION_DEPTH_MAX_ASPECT)
    else:
        fitted_aspect = 1.0 / min(1.0 / aspect, PRODUCTION_DEPTH_MAX_ASPECT)
    bounded_short = min(max(PRODUCTION_DEPTH_SHORT_SIDE, patch),
                        PRODUCTION_DEPTH_MAX_DIM)
    max_width = max(patch, (min(source_width, PRODUCTION_DEPTH_MAX_DIM) // patch) * patch)
    max_height = max(patch, (min(source_height, PRODUCTION_DEPTH_MAX_DIM) // patch) * patch)
    requested_short = _round_to_patch(bounded_short, patch)

    if fitted_aspect >= 1.0:
        for height in range(min(requested_short, max_height), patch - 1, -patch):
            width = _round_to_patch(height * fitted_aspect, patch)
            if width <= max_width:
                return (width, height)
    else:
        for width in range(min(requested_short, max_width), patch - 1, -patch):
            height = _round_to_patch(width / fitted_aspect, patch)
            if height <= max_height:
                return (width, height)
    return (patch, patch)


def preparation_geometry_contract(clip_id, clip):
    """Validate and resolve one manifest-owned identity or center-padding transform."""
    source = clip.get("source_shape")
    if source is None:
        return None  # Backward-compatible unit fixtures using manifest schema 1/2.
    if not isinstance(source, dict) or set(source) != {"width", "height"}:
        fail(f"{clip_id}: source_shape must contain exactly width and height")
    source_width = _positive_manifest_dimension(
        source.get("width"), f"{clip_id}.source_shape.width")
    source_height = _positive_manifest_dimension(
        source.get("height"), f"{clip_id}.source_shape.height")

    canvas = clip.get("evaluation_canvas")
    if canvas is None:
        canvas_width, canvas_height = source_width, source_height
        method = "identity"
    else:
        if not isinstance(canvas, dict) or set(canvas) != {"width", "height"}:
            fail(f"{clip_id}: evaluation_canvas must contain exactly width and height")
        canvas_width = _positive_manifest_dimension(
            canvas.get("width"), f"{clip_id}.evaluation_canvas.width")
        canvas_height = _positive_manifest_dimension(
            canvas.get("height"), f"{clip_id}.evaluation_canvas.height")
        if canvas_width < source_width or canvas_height < source_height:
            fail(f"{clip_id}: evaluation_canvas cannot crop the declared source_shape")
        if (canvas_width, canvas_height) == (source_width, source_height):
            fail(f"{clip_id}: redundant evaluation_canvas must be omitted")
        method = "center-pad-black-no-resize"

    left = (canvas_width - source_width) // 2
    top = (canvas_height - source_height) // 2
    right = canvas_width - source_width - left
    bottom = canvas_height - source_height - top
    tensor_width, tensor_height = fit_host_sbs_v2_depth_tensor_shape(
        canvas_width, canvas_height)
    if (tensor_width, tensor_height) not in PRODUCTION_V2_TENSOR_SHAPES:
        fail(
            f"{clip_id}: prepared canvas {canvas_width}x{canvas_height} fits unsupported "
            f"V2 tensor {tensor_width}x{tensor_height}")

    return {
        "schema": PREPARATION_GEOMETRY_SCHEMA,
        "method": method,
        "source_shape": {"width": source_width, "height": source_height},
        "canvas_shape": {"width": canvas_width, "height": canvas_height},
        "content_offset": {"x": left, "y": top},
        "padding": {"left": left, "top": top, "right": right, "bottom": bottom},
        "canvas_fill_rgb": [0, 0, 0],
        "source_pixels": "bit-exact-no-resize-no-crop",
        "depth_tensor_shape": {"width": tensor_width, "height": tensor_height},
    }


def safe_path_component(value, description):
    """Validate a manifest-owned name before joining it to a dataset cache directory."""
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError(f"{description} must be a non-empty basename")
    # Manifests move between Windows and POSIX machines. Reject either platform's absolute,
    # drive-qualified, or separator-bearing spelling regardless of the current host.
    if (os.path.isabs(value) or ntpath.isabs(value)
            or os.path.basename(value) != value
            or ntpath.basename(value) != value):
        raise ValueError(
            f"{description} must be a basename without path separators: {value!r}")
    return value


def confined_child(root, component, description):
    """Resolve one child and prove it remains below ``root`` before replacing/deleting it."""
    component = safe_path_component(component, description)
    resolved_root = os.path.realpath(os.path.abspath(root))
    resolved_child = os.path.realpath(os.path.join(resolved_root, component))
    try:
        contained = os.path.commonpath((resolved_root, resolved_child)) == resolved_root
    except ValueError:
        contained = False
    if not contained or resolved_child == resolved_root:
        raise ValueError(f"{description} escapes dataset cache root: {component!r}")
    return resolved_child


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") not in (1, 2, 3):
        fail(f"unsupported dataset manifest schema: {data.get('schema')}")
    datasets, clips = data.get("datasets", {}), data.get("clips", {})
    safe_path_component(data.get("prepared_suite"), "prepared suite")
    for dataset_id, dataset in datasets.items():
        for archive_id, spec in (dataset.get("archives") or {}).items():
            if "filename" in spec:
                safe_path_component(
                    spec["filename"], f"{dataset_id}.{archive_id} archive filename")
    for clip_id, clip in clips.items():
        safe_path_component(clip_id, "clip ID")
        if data.get("schema") >= 3 and "source_shape" not in clip:
            fail(f"{clip_id}: manifest schema 3 requires source_shape")
        preparation_geometry_contract(clip_id, clip)
        if clip.get("content_type") not in CONTENT_TYPES:
            fail(f"{clip_id}: content_type must be one of {sorted(CONTENT_TYPES)}")
        try:
            archive_specs = [datasets[clip["dataset"]]["archives"][name]
                             for name in clip["archives"]]
        except (KeyError, TypeError) as exc:
            fail(f"invalid archive references for clip {clip_id}: {exc}")
        if any(spec.get("access") == "http_range_zip" for spec in archive_specs):
            digest = clip.get("prepared_evidence_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                fail(f"{clip_id}: range-backed clip requires prepared_evidence_sha256")
    return data


def archive_spec(manifest, clip, archive_name):
    dataset = manifest["datasets"][clip["dataset"]]
    try:
        return dataset["archives"][archive_name]
    except KeyError:
        fail(f"unknown archive {clip['dataset']}.{archive_name}")


def download_archive(spec, downloads_dir):
    os.makedirs(downloads_dir, exist_ok=True)
    path = confined_child(downloads_dir, spec["filename"], "archive filename")
    expected = spec.get("sha256")
    if os.path.exists(path) and (not expected or sha256(path) == expected.lower()):
        return path
    print(f"download: {spec['url']}\n      -> {path}", flush=True)
    cmd = ["curl.exe" if os.name == "nt" else "curl", "-L", "--fail", "--retry", "5",
           "--retry-delay", "3", "-C", "-", "-o", path, spec["url"]]
    result = subprocess.run(cmd)
    if result.returncode:
        fail(f"download failed ({result.returncode}): {spec['url']}")
    if expected:
        actual = sha256(path)
        if actual != expected.lower():
            fail(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return path


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only HTTP file backed by deterministic Range requests.

    Large public ZIPs can be sampled without downloading unrelated multi-gigabyte members. The
    manifest still pins the immutable datafile URL, byte size, and upstream digest.
    """

    def __init__(self, url, expected_size=None):
        self.url = url
        self.pos = 0
        _, content_range = self._range(0, 0)
        match = re.search(r"/(\d+)$", content_range or "")
        if not match:
            fail(f"remote archive does not advertise a byte-range size: {url}")
        self.size = int(match.group(1))
        if expected_size is not None and self.size != int(expected_size):
            fail(f"remote archive size mismatch for {url}: {self.size} != {expected_size}")

    def _range(self, start, end):
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                     "User-Agent": "Apollo-sbsbench/1"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
            content_range = response.headers.get("Content-Range")
        expected = end - start + 1
        if len(data) != expected:
            fail(f"short HTTP range read for {self.url}: {len(data)} != {expected}")
        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range or "")
        if not match or (int(match.group(1)), int(match.group(2))) != (start, end):
            fail(f"invalid Content-Range for {self.url}: {content_range!r}")
        return data, content_range

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.pos + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if target < 0:
            raise OSError("negative seek position")
        self.pos = target
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.size - 1, self.pos + size - 1)
        data, _ = self._range(self.pos, end)
        self.pos += len(data)
        return data


def _member_ending(zf, suffix):
    matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        fail(f"expected exactly one ZIP member ending {suffix!r}, found {len(matches)}")
    return matches[0]


def _parse_tum_list(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), fields[1].replace("\\", "/")))
    return rows


def associate_timestamps(rgb, depth, max_delta):
    """Associate each RGB sample to its nearest unused depth timestamp."""
    pairs, used = [], set()
    for rgb_ts, rgb_path in rgb:
        candidates = [(abs(rgb_ts - dts), i, dts, path)
                      for i, (dts, path) in enumerate(depth) if i not in used]
        if not candidates:
            break
        delta, i, depth_ts, depth_path = min(candidates)
        if delta <= max_delta:
            used.add(i)
            pairs.append((rgb_ts, rgb_path, depth_ts, depth_path))
    return pairs


def expected_source_indices(clip):
    """Return the exact source-window indexes shared by extraction and metadata refresh."""
    start, stride, count = (int(clip[k]) for k in ("start", "stride", "count"))
    if start < 0 or stride <= 0 or count <= 0:
        fail(f"invalid selection window {start}:{stride}:{count}")
    return [start + i * stride for i in range(count)]


def selected(items, clip):
    indexes = expected_source_indices(clip)
    if indexes[-1] >= len(items):
        start, stride, count = (int(clip[k]) for k in ("start", "stride", "count"))
        fail(f"selection {start}:{stride}:{count} exceeds {len(items)} available samples")
    return [(i, items[i]) for i in indexes]


def adapter_selection_identity(clip):
    """Return the manifest-owned identity fields emitted by this adapter's selection rows."""
    adapter = clip.get("adapter")
    try:
        fields = ADAPTER_SELECTION_CONTRACTS[adapter]["identity_fields"]
    except KeyError:
        fail(f"unsupported adapter selection contract: {adapter!r}")
    values = {**clip, "split": "test"}
    try:
        return {field: values[field] for field in fields}
    except KeyError as exc:
        fail(f"{adapter}: selection identity is missing {exc.args[0]!r}")


def make_selection_entry(clip, source_index, **source_values):
    """Build one selection row from the same adapter contract refresh validation consumes."""
    adapter = clip.get("adapter")
    try:
        expected_fields = set(ADAPTER_SELECTION_CONTRACTS[adapter]["source_fields"])
    except KeyError:
        fail(f"unsupported adapter selection contract: {adapter!r}")
    actual_fields = set(source_values)
    if actual_fields != expected_fields:
        fail(
            f"{adapter}: invalid selection source fields "
            f"(missing={sorted(expected_fields - actual_fields)}, "
            f"unexpected={sorted(actual_fields - expected_fields)})")
    return {
        "source_index": source_index,
        **source_values,
        **adapter_selection_identity(clip),
    }


def _zip_relative_member(zf, list_member, relative):
    base = list_member.rsplit("/", 1)[0] if "/" in list_member else ""
    candidate = f"{base}/{relative}" if base else relative
    if candidate not in zf.namelist():
        fail(f"ZIP member referenced by association file is missing: {candidate}")
    return candidate


def _write_image_bytes(data, path, rgb=False):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB" if rgb else image.mode)
        image.save(path, compress_level=3)


_PADDED_IMAGE_DIRECTORIES = frozenset({
    ".", "gt_depth", "gt_right", "gt_depth_valid", "gt_depth_valid_all",
    "gt_depth_valid_nonocc", "gt_occlusion", "gt_outofframe",
})


def _prepared_frame_evidence(directory):
    return sorted(
        path for path in glob.glob(
            os.path.join(directory, "**", "frame_*.*"), recursive=True)
        if os.path.isfile(path))


def _evidence_directory(directory, path):
    relative_parent = os.path.relpath(os.path.dirname(path), directory).replace("\\", "/")
    return relative_parent


def _validate_spatial_shape(clip_id, path, actual, expected):
    if tuple(actual) != tuple(expected):
        relative = os.path.relpath(path).replace("\\", "/")
        fail(
            f"{clip_id}: prepared evidence {relative} has spatial shape "
            f"{tuple(actual)}, expected {tuple(expected)}")


def validate_prepared_evidence_geometry(clip_id, directory, width, height):
    """Require every prepared source/reference sidecar to share one canvas geometry."""
    paths = _prepared_frame_evidence(directory)
    if not paths:
        fail(f"{clip_id}: prepared clip contains no frame evidence")
    expected = (height, width)
    for path in paths:
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".png":
            with Image.open(path) as image:
                actual = (image.height, image.width)
        elif suffix == ".npy":
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            if value.ndim != 2:
                fail(f"{clip_id}: padded NPY evidence must be one spatial plane: {path}")
            actual = value.shape
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"flow", "valid"}:
                    fail(f"{clip_id}: flow evidence must contain exactly flow and valid: {path}")
                flow, valid = archive["flow"], archive["valid"]
            if flow.ndim != 3 or flow.shape[2] != 2 or valid.ndim != 2:
                fail(f"{clip_id}: flow evidence has invalid array ranks: {path}")
            _validate_spatial_shape(clip_id, path, flow.shape[:2], expected)
            actual = valid.shape
        else:
            fail(f"{clip_id}: unsupported prepared evidence format: {path}")
        _validate_spatial_shape(clip_id, path, actual, expected)


def _center_pad_array(value, geometry, fill=0):
    source = geometry["source_shape"]
    canvas = geometry["canvas_shape"]
    offset = geometry["content_offset"]
    expected = (source["height"], source["width"])
    if value.shape[:2] != expected:
        raise ValueError(f"array shape {value.shape[:2]} does not match source {expected}")
    shape = (canvas["height"], canvas["width"], *value.shape[2:])
    padded = np.full(shape, fill, dtype=value.dtype)
    y, x = offset["y"], offset["x"]
    padded[y:y + source["height"], x:x + source["width"], ...] = value
    return padded


def apply_center_padding(clip_id, directory, geometry):
    """Center-pad every source/reference artifact without resampling any original pixel."""
    if geometry["method"] != "center-pad-black-no-resize":
        return
    source = geometry["source_shape"]
    canvas = geometry["canvas_shape"]
    x, y = geometry["content_offset"]["x"], geometry["content_offset"]["y"]
    for path in _prepared_frame_evidence(directory):
        suffix = os.path.splitext(path)[1].lower()
        evidence_directory = _evidence_directory(directory, path)
        if suffix == ".png":
            if evidence_directory not in _PADDED_IMAGE_DIRECTORIES:
                fail(
                    f"{clip_id}: no padding semantics declared for "
                    f"{evidence_directory}/{os.path.basename(path)}")
            with Image.open(path) as image:
                image.load()
                _validate_spatial_shape(
                    clip_id, path, (image.height, image.width),
                    (source["height"], source["width"]))
                original = image.copy()
                mode = image.mode
            # Out-of-frame is the sole inverted mask: padded pixels have no source-camera GT.
            fill = 255 if evidence_directory == "gt_outofframe" else 0
            padded = Image.new(mode, (canvas["width"], canvas["height"]), fill)
            padded.paste(original, (x, y))
            padded.save(path, compress_level=3)
        elif suffix == ".npy":
            if evidence_directory != "gt_depth":
                fail(f"{clip_id}: no padding semantics declared for NPY evidence {path}")
            value = np.load(path, allow_pickle=False)
            if value.ndim != 2:
                fail(f"{clip_id}: depth NPY must be one spatial plane: {path}")
            np.save(path, _center_pad_array(value, geometry, fill=0))
        elif suffix == ".npz":
            if evidence_directory != "gt_flow":
                fail(f"{clip_id}: no padding semantics declared for NPZ evidence {path}")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"flow", "valid"}:
                    fail(f"{clip_id}: flow evidence must contain exactly flow and valid: {path}")
                flow = archive["flow"]
                valid = archive["valid"]
            if flow.ndim != 3 or flow.shape[2] != 2 or valid.ndim != 2:
                fail(f"{clip_id}: flow evidence has invalid array ranks: {path}")
            np.savez_compressed(
                path,
                flow=_center_pad_array(flow, geometry, fill=0),
                valid=_center_pad_array(valid.astype(bool), geometry, fill=False),
            )
        else:
            fail(f"{clip_id}: unsupported prepared evidence format: {path}")


def finalize_prepared_geometry(clip_id, clip, directory):
    """Authenticate extracted native geometry, apply its canvas, and bind final dimensions."""
    geometry = preparation_geometry_contract(clip_id, clip)
    if geometry is None:
        return None
    source = geometry["source_shape"]
    validate_prepared_evidence_geometry(
        clip_id, directory, source["width"], source["height"])
    apply_center_padding(clip_id, directory, geometry)
    canvas = geometry["canvas_shape"]
    validate_prepared_evidence_geometry(
        clip_id, directory, canvas["width"], canvas["height"])
    return geometry


def prepared_evidence_sha256(directory):
    """Hash decoded prepared evidence independently of PNG/NPY container encoding."""
    paths = sorted(
        path for path in glob.glob(os.path.join(directory, "**", "frame_*.*"), recursive=True)
        if os.path.isfile(path))
    if not paths:
        fail(f"no prepared frame evidence under {directory}")
    digest = hashlib.sha256()
    for path in paths:
        relative = os.path.relpath(path, directory).replace("\\", "/").encode("utf-8")
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix in (".png", ".jpg", ".jpeg"):
            with Image.open(path) as image:
                value = np.asarray(image).copy()
        else:
            fail(f"unsupported prepared evidence format: {path}")
        value = np.ascontiguousarray(value)
        shape = json.dumps(value.shape, separators=(",", ":")).encode("ascii")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(shape)
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def authenticate_prepared_range_evidence(clip_id, clip, directory):
    """Reject range-selected data unless its consumed decoded evidence matches the manifest."""
    expected = clip.get("prepared_evidence_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        fail(f"{clip_id}: range-backed clip requires prepared_evidence_sha256")
    actual = prepared_evidence_sha256(directory)
    if actual != expected:
        fail(
            f"{clip_id}: prepared evidence SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def prepare_tum(clip_id, clip, dataset, archives, out_dir, suite):
    archive = archives[clip["archives"][0]]
    with zipfile.ZipFile(archive) as zf:
        rgb_list = _member_ending(zf, "/rgb.txt")
        depth_list = _member_ending(zf, "/depth.txt")
        rgb = _parse_tum_list(zf.read(rgb_list).decode("utf-8"))
        depth = _parse_tum_list(zf.read(depth_list).decode("utf-8"))
        pairs = associate_timestamps(rgb, depth, float(clip["max_timestamp_delta"]))
        chosen = selected(pairs, clip)
        os.makedirs(os.path.join(out_dir, "gt_depth"))
        selection = []
        for output_id, (source_i, pair) in enumerate(chosen):
            rgb_ts, rgb_path, depth_ts, depth_path = pair
            _write_image_bytes(zf.read(_zip_relative_member(zf, rgb_list, rgb_path)),
                               os.path.join(out_dir, f"frame_{output_id:05d}.png"), rgb=True)
            _write_image_bytes(zf.read(_zip_relative_member(zf, depth_list, depth_path)),
                               os.path.join(out_dir, "gt_depth", f"frame_{output_id:05d}.png"))
            selection.append(make_selection_entry(
                clip, source_i, rgb_timestamp=rgb_ts, depth_timestamp=depth_ts))
    return selection


def _tartan_members(zf, trajectory, folder, ending):
    marker = f"/{trajectory}/{folder}/"
    rx = re.compile(r"/(\d+).*" + re.escape(ending) + r"$")
    result = {}
    for name in zf.namelist():
        if marker not in name or not name.endswith(ending):
            continue
        match = rx.search(name)
        if match:
            result[int(match.group(1))] = name
    return result


def _load_npy_member(zf, member):
    return np.load(io.BytesIO(zf.read(member)), allow_pickle=False)


def _decode_tartan_depth(data):
    """Decode TartanAir's lossless float32-in-RGBA PNG without requiring OpenCV.

    The official writer/reader uses OpenCV BGRA arrays. Pillow exposes the encoded RGBA channel
    order, so swap R/B back before viewing each four bytes as a little-endian float.
    """
    with Image.open(io.BytesIO(data)) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    depth = bgra.view("<f4").reshape(bgra.shape[:2])
    return np.asarray(depth, dtype=np.float32)


def _decode_tartan_flow(data):
    """Decode TartanAir's uint16 BGR PNG into float flow and validity mask."""
    rgb = _decode_png_rgb16(data)
    bgr = rgb[..., [2, 1, 0]]
    flow = (bgr[..., :2].astype(np.float32) - 32768.0) / 64.0
    return flow, bgr[..., 2].astype(np.uint8)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def _decode_png_rgb16(data):
    """Minimal non-interlaced 16-bit RGB PNG decoder.

    Pillow truncates multi-channel 16-bit PNGs to uint8. TartanAir encodes optical flow in the
    low bits, so truncation turns ordinary motion into roughly -510 px. This small decoder keeps
    all bits and avoids adding OpenCV as a 50+ MB evaluator dependency.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        fail("invalid PNG signature")
    offset, payloads, header = 8, [], None
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            payloads.append(payload)
        elif kind == b"IEND":
            break
    if not header:
        fail("PNG has no IHDR")
    width, height, bits, color, compression, filtering, interlace = header
    if (bits, color, compression, filtering, interlace) != (16, 2, 0, 0, 0):
        fail(f"expected non-interlaced 16-bit RGB PNG, got IHDR {header}")
    bpp, row_bytes = 6, width * 6
    raw = zlib.decompress(b"".join(payloads))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        fail(f"PNG scanline length mismatch: expected {expected}, got {len(raw)}")
    rows = np.zeros((height, row_bytes), dtype=np.uint8)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        encoded = raw[pos:pos + row_bytes]
        pos += row_bytes
        for x, byte in enumerate(encoded):
            left = int(rows[y, x - bpp]) if x >= bpp else 0
            up = int(rows[y - 1, x]) if y else 0
            upper_left = int(rows[y - 1, x - bpp]) if y and x >= bpp else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = _paeth(left, up, upper_left)
            else:
                fail(f"unsupported PNG filter {filter_type}")
            rows[y, x] = (int(byte) + predictor) & 255
    return rows.reshape(height, width, 3, 2).astype(np.uint16).dot(
        np.array([256, 1], dtype=np.uint16))


def _normalize_flow(array):
    flow = np.asarray(array, dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow, 0, -1)
    if flow.ndim != 3 or flow.shape[-1] < 2:
        fail(f"unexpected TartanAir flow shape {flow.shape}")
    return flow[..., :2]


def prepare_tartanair(clip_id, clip, dataset, archives, out_dir, suite):
    image_zip = zipfile.ZipFile(archives["archviz_image"])
    depth_zip = zipfile.ZipFile(archives["archviz_depth"])
    flow_zip = zipfile.ZipFile(archives["archviz_flow"])
    try:
        trajectory, camera = clip["trajectory"], clip["camera"]
        images = _tartan_members(image_zip, trajectory, f"image_{camera}", ".png")
        depths = _tartan_members(depth_zip, trajectory, f"depth_{camera}", "_depth.png")
        # TartanAir names forward flow by its first frame (000000_000001_flow.npy).
        flows = _tartan_members(flow_zip, trajectory, f"flow_{camera}", "_flow.png")
        chosen = selected(sorted(set(images) & set(depths)), clip)
        source_ids = [frame_id for _, frame_id in chosen]
        if int(clip["stride"]) != 1:
            fail("TartanAir GT flow currently requires stride=1 (flow composition is not implicit)")
        os.makedirs(os.path.join(out_dir, "gt_depth"))
        os.makedirs(os.path.join(out_dir, "gt_flow"))
        selection = []
        for output_id, (source_i, frame_id) in enumerate(chosen):
            _write_image_bytes(image_zip.read(images[frame_id]),
                               os.path.join(out_dir, f"frame_{output_id:05d}.png"), rgb=True)
            depth = _decode_tartan_depth(depth_zip.read(depths[frame_id]))
            np.save(os.path.join(out_dir, "gt_depth", f"frame_{output_id:05d}.npy"), depth)
            # The sidecar stored on current output N maps previous N-1 -> current N.
            if output_id:
                previous_source = source_ids[output_id - 1]
                if previous_source not in flows:
                    fail(f"missing forward flow for TartanAir frame {previous_source}")
                flow, mask = _decode_tartan_flow(flow_zip.read(flows[previous_source]))
                flow = _normalize_flow(flow)
                np.savez_compressed(os.path.join(out_dir, "gt_flow", f"frame_{output_id:05d}.npz"),
                                    flow=flow, valid=(mask == 0) & np.isfinite(flow).all(axis=2))
            selection.append(make_selection_entry(
                clip, source_i, dataset_frame=frame_id))
        return selection
    finally:
        image_zip.close()
        depth_zip.close()
        flow_zip.close()


def _indexed_archive_members(names, marker, pattern):
    result = {}
    rx = re.compile(pattern)
    for name in names:
        normalized = name.replace("\\", "/")
        searchable = "/" + normalized.lstrip("/")
        if marker not in searchable:
            continue
        match = rx.search(searchable)
        if match:
            result[int(match.group(1))] = name
    return result


def prepare_sintel(clip_id, clip, dataset, archives, out_dir, suite):
    archive = archives[clip["archives"][0]]
    range_backed = isinstance(archive, dict)
    sequence, render_pass = clip["sequence"], clip["pass"]
    with contextlib.ExitStack() as stack:
        if range_backed:
            if archive.get("access") != "http_range_zip":
                fail(f"unsupported remote Sintel archive contract: {archive.get('access')!r}")
            archive = stack.enter_context(
                HTTPRangeReader(archive["url"], archive.get("size")))
        zf = stack.enter_context(zipfile.ZipFile(archive))
        names = zf.namelist()
        left_marker = f"/{render_pass}_left/{sequence}/"
        right_marker = f"/{render_pass}_right/{sequence}/"
        left = _indexed_archive_members(names, left_marker, r"frame_(\d+)\.png$")
        right = _indexed_archive_members(names, right_marker, r"frame_(\d+)\.png$")
        disparities = _indexed_archive_members(
            names, f"/disparities/{sequence}/", r"frame_(\d+)\.png$")
        occlusions = _indexed_archive_members(
            names, f"/occlusions/{sequence}/", r"frame_(\d+)\.png$")
        outofframe = _indexed_archive_members(
            names, f"/outofframe/{sequence}/", r"frame_(\d+)\.png$")
        if not left:
            fail(f"no Sintel {render_pass}_left frames found for {sequence}")
        chosen = selected(sorted(left), clip)
        os.makedirs(os.path.join(out_dir, "gt_right"))
        os.makedirs(os.path.join(out_dir, "gt_depth"))
        for directory in (
                "gt_depth_valid", "gt_depth_valid_all", "gt_depth_valid_nonocc",
                "gt_occlusion", "gt_outofframe"):
            os.makedirs(os.path.join(out_dir, directory))
        selection = []
        for output_id, (source_i, frame_id) in enumerate(chosen):
            if frame_id not in right:
                fail(f"missing Sintel right-eye frame {sequence}/{frame_id}")
            if frame_id not in disparities:
                fail(f"missing Sintel disparity frame {sequence}/{frame_id}")
            if frame_id not in occlusions:
                fail(f"missing Sintel occlusion mask {sequence}/{frame_id}")
            if frame_id not in outofframe:
                fail(f"missing Sintel out-of-frame mask {sequence}/{frame_id}")
            left_bytes, right_bytes = zf.read(left[frame_id]), zf.read(right[frame_id])
            with Image.open(io.BytesIO(left_bytes)) as image:
                left_image = image.convert("RGB")
            with Image.open(io.BytesIO(right_bytes)) as image:
                right_image = image.convert("RGB")
            with Image.open(io.BytesIO(zf.read(disparities[frame_id]))) as disparity_png:
                encoded = np.asarray(disparity_png.convert("RGB"), dtype=np.float32)
            disparity = encoded[..., 0] * 4.0 + encoded[..., 1] / 64.0 + encoded[..., 2] / 16384.0
            with Image.open(io.BytesIO(zf.read(occlusions[frame_id]))) as image:
                occlusion = np.asarray(image.convert("L"), dtype=np.uint8)
            with Image.open(io.BytesIO(zf.read(outofframe[frame_id]))) as image:
                outside = np.asarray(image.convert("L"), dtype=np.uint8)
            expected_size = (left_image.height, left_image.width)
            shapes = {
                "right": (right_image.height, right_image.width),
                "disparity": disparity.shape,
                "occlusion": occlusion.shape,
                "outofframe": outside.shape,
            }
            mismatched = {name: shape for name, shape in shapes.items()
                          if tuple(shape) != expected_size}
            if mismatched:
                fail(
                    f"Sintel frame geometry mismatch {sequence}/{frame_id}: "
                    f"left={expected_size}, sidecars={mismatched}")
            for name, mask in (("occlusion", occlusion), ("outofframe", outside)):
                levels = set(np.unique(mask).tolist())
                if not levels.issubset({0, 255}):
                    fail(
                        f"Sintel {name} mask is not binary {sequence}/{frame_id}: "
                        f"values={sorted(levels)[:8]}")
            occluded, outside = occlusion != 0, outside != 0
            finite = np.isfinite(disparity)
            valid_all = finite & ~outside
            valid_nonocc = valid_all & ~occluded
            frame_name = f"frame_{output_id:05d}.png"
            left_image.save(os.path.join(out_dir, frame_name), compress_level=3)
            right_image.save(os.path.join(out_dir, "gt_right", frame_name), compress_level=3)
            np.save(os.path.join(out_dir, "gt_depth", f"frame_{output_id:05d}.npy"),
                    disparity.astype(np.float32))
            masks = {
                "gt_depth_valid": valid_nonocc,
                "gt_depth_valid_all": valid_all,
                "gt_depth_valid_nonocc": valid_nonocc,
                "gt_occlusion": occluded,
                "gt_outofframe": outside,
            }
            for directory, mask in masks.items():
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    os.path.join(out_dir, directory, frame_name), compress_level=3)
            selection.append(make_selection_entry(
                clip, source_i, dataset_frame=frame_id))
        if range_backed:
            authenticate_prepared_range_evidence(clip_id, clip, out_dir)
        return selection


def prepare_spring(clip_id, clip, dataset, archives, out_dir, suite):
    sequence = clip["sequence"]
    left_spec, right_spec = archives["test_left"], archives["test_right"]
    left_reader = HTTPRangeReader(left_spec["url"], left_spec.get("size"))
    right_reader = HTTPRangeReader(right_spec["url"], right_spec.get("size"))
    with zipfile.ZipFile(left_reader) as left_zip, zipfile.ZipFile(right_reader) as right_zip:
        left = _indexed_archive_members(
            left_zip.namelist(), f"/test/{sequence}/frame_left/", r"frame_left_(\d+)\.png$")
        right = _indexed_archive_members(
            right_zip.namelist(), f"/test/{sequence}/frame_right/", r"frame_right_(\d+)\.png$")
        available = sorted(set(left) & set(right))
        if not available:
            fail(f"no Spring stereo frames found for sequence {sequence}")
        chosen = selected(available, clip)
        os.makedirs(os.path.join(out_dir, "gt_right"))
        selection = []
        for output_id, (source_i, frame_id) in enumerate(chosen):
            if output_id == 0 or (output_id + 1) % 6 == 0 or output_id + 1 == len(chosen):
                print(f"extract: {clip_id} {output_id + 1}/{len(chosen)}", flush=True)
            _write_image_bytes(left_zip.read(left[frame_id]),
                               os.path.join(out_dir, f"frame_{output_id:05d}.png"), rgb=True)
            _write_image_bytes(right_zip.read(right[frame_id]),
                               os.path.join(out_dir, "gt_right", f"frame_{output_id:05d}.png"),
                               rgb=True)
            selection.append(make_selection_entry(
                clip, source_i, dataset_frame=frame_id))
        authenticate_prepared_range_evidence(clip_id, clip, out_dir)
        return selection


def _tar_member_bytes(tf, member_name):
    member = tf.getmember(member_name)
    fh = tf.extractfile(member)
    if fh is None:
        fail(f"cannot read TAR member: {member_name}")
    return fh.read()


def prepare_vkitti2(clip_id, clip, dataset, archives, out_dir, suite):
    scene, variant, camera = clip["scene"], clip["variant"], clip["camera"]
    rgb_tar = tarfile.open(archives["rgb"], "r:")
    depth_tar = tarfile.open(archives["depth"], "r:")
    try:
        rgb_marker = f"/{scene}/{variant}/frames/rgb/{camera}/"
        depth_marker = f"/{scene}/{variant}/frames/depth/{camera}/"
        rgbs = _indexed_archive_members(
            (m.name for m in rgb_tar.getmembers() if m.isfile()), rgb_marker, r"rgb_(\d+)\.jpg$")
        depths = _indexed_archive_members(
            (m.name for m in depth_tar.getmembers() if m.isfile()), depth_marker,
            r"depth_(\d+)\.png$")
        available = sorted(set(rgbs) & set(depths))
        if not available:
            fail(f"no matching VKITTI RGB/depth frames for {scene}/{variant}/{camera}")
        chosen = selected(available, clip)
        os.makedirs(os.path.join(out_dir, "gt_depth"))
        selection = []
        for output_id, (source_i, frame_id) in enumerate(chosen):
            _write_image_bytes(_tar_member_bytes(rgb_tar, rgbs[frame_id]),
                               os.path.join(out_dir, f"frame_{output_id:05d}.png"), rgb=True)
            _write_image_bytes(_tar_member_bytes(depth_tar, depths[frame_id]),
                               os.path.join(out_dir, "gt_depth", f"frame_{output_id:05d}.png"))
            selection.append(make_selection_entry(
                clip, source_i, dataset_frame=frame_id))
        return selection
    finally:
        rgb_tar.close()
        depth_tar.close()


def prepared_clip_metadata(manifest, clip, selection, preparation_geometry=None):
    """Build the source-semantic metadata shared by extraction and metadata-only refreshes."""
    dataset = manifest["datasets"][clip["dataset"]]
    has_gt_depth = clip["adapter"] != "spring_http_range_zip"
    meta = {
        "name": clip["name"], "description": clip["description"],
        "content_type": clip["content_type"],
        "dataset": dataset["title"], "homepage": dataset["homepage"],
        "citation": dataset["citation"], "license_note": dataset["license_note"],
        "suite": manifest["prepared_suite"],
        "required_gt_depth": has_gt_depth,
        "required_gt_flow": clip["adapter"] == "tartanair_v2_zip",
        "evaluation_role": ("reference-only" if not has_gt_depth else "ground-truth"),
        "selection": selection,
    }
    if has_gt_depth:
        meta["gt_depth_kind"] = ("disparity" if clip["adapter"] == "sintel_stereo_zip"
                                 else "metric")
    if clip["adapter"] in ("sintel_stereo_zip", "spring_http_range_zip"):
        meta["reference_stereo_available"] = True
    for key in ("source_artifacts", "source_artifact_frame"):
        if key in clip:
            meta[key] = clip[key]
    if "prepared_evidence_sha256" in clip:
        meta["prepared_evidence_sha256"] = clip["prepared_evidence_sha256"]
    if preparation_geometry is not None:
        meta["preparation_geometry"] = preparation_geometry
    return meta


def validate_prepared_selection(clip_id, clip, selection):
    """Authenticate a cached selection against the adapter's exact manifest window."""
    expected_indices = expected_source_indices(clip)
    if not isinstance(selection, list) or len(selection) != len(expected_indices):
        fail(
            f"{clip_id}: prepared selection does not match manifest count "
            f"{len(expected_indices)}")
    adapter = clip.get("adapter")
    try:
        source_fields = set(ADAPTER_SELECTION_CONTRACTS[adapter]["source_fields"])
    except KeyError:
        fail(f"{clip_id}: unsupported adapter selection contract: {adapter!r}")
    identity = adapter_selection_identity(clip)
    expected_keys = {"source_index", *source_fields, *identity}
    for output_id, (entry, expected_index) in enumerate(zip(selection, expected_indices)):
        if not isinstance(entry, dict):
            fail(f"{clip_id}: prepared selection row {output_id} must be an object")
        actual_keys = set(entry)
        if actual_keys != expected_keys:
            fail(
                f"{clip_id}: prepared selection row {output_id} fields do not match "
                f"{adapter} contract (missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)})")
        source_index = entry["source_index"]
        if type(source_index) is not int or source_index != expected_index:
            fail(
                f"{clip_id}: prepared selection row {output_id} source_index="
                f"{source_index!r}, expected {expected_index}")
        for key, expected in identity.items():
            if entry[key] != expected:
                fail(
                    f"{clip_id}: prepared selection row {output_id} {key}="
                    f"{entry[key]!r}, expected {expected!r}")
        if "dataset_frame" in source_fields and type(entry["dataset_frame"]) is not int:
            fail(
                f"{clip_id}: prepared selection row {output_id} dataset_frame "
                "must be an integer")
        for timestamp in ("rgb_timestamp", "depth_timestamp"):
            if timestamp in source_fields:
                value = entry[timestamp]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not np.isfinite(value)):
                    fail(
                        f"{clip_id}: prepared selection row {output_id} {timestamp} "
                        "must be finite numeric evidence")
    return selection


def require_prepared_frame_ids(
        clip_id, directory, expected_ids, evidence_name, expected_extension):
    """Require exact numeric IDs, regular files, and the adapter's emitted evidence format."""
    if not os.path.isdir(directory):
        fail(f"{clip_id}: prepared {evidence_name} directory is missing")
    if not re.fullmatch(r"\.[a-z0-9]+", expected_extension):
        raise ValueError(f"invalid expected evidence extension: {expected_extension!r}")
    paths = glob.glob(os.path.join(directory, "frame_*.*"))
    identities = {}
    malformed = []
    non_regular = []
    wrong_extensions = []
    for path in paths:
        match = re.fullmatch(r"frame_(\d+)\.[^.]+", os.path.basename(path))
        if match is None:
            malformed.append(os.path.basename(path))
            continue
        frame_id = int(match.group(1))
        identities.setdefault(frame_id, []).append(path)
        try:
            regular = stat.S_ISREG(os.lstat(path).st_mode)
        except OSError:
            regular = False
        if not regular:
            non_regular.append(os.path.basename(path))
        if os.path.splitext(path)[1].lower() != expected_extension:
            wrong_extensions.append(os.path.basename(path))
    duplicates = sorted(frame_id for frame_id, matches in identities.items()
                        if len(matches) != 1)
    actual_ids = set(identities)
    expected_ids = set(expected_ids)
    if (malformed or non_regular or wrong_extensions or duplicates
            or actual_ids != expected_ids):
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        fail(
            f"{clip_id}: prepared {evidence_name} identities are invalid "
            f"(missing={missing}, unexpected={unexpected}, duplicates={duplicates}, "
            f"malformed={sorted(malformed)}, non_regular={sorted(non_regular)}, "
            f"wrong_extensions={sorted(wrong_extensions)}, "
            f"expected_extension={expected_extension})")


def refresh_prepared_clip_metadata(manifest, clip_id, clip, prepared_root):
    """Refresh manifest-owned metadata without downloading or rewriting authenticated pixels."""
    final = confined_child(prepared_root, clip_id, "clip ID")
    meta_path = os.path.join(final, "meta.json")
    try:
        with open(meta_path, encoding="utf-8") as stream:
            existing = json.load(stream)
    except (OSError, ValueError) as exc:
        fail(f"{clip_id}: cannot refresh invalid prepared metadata: {exc}")
    if not isinstance(existing, dict):
        fail(f"{clip_id}: prepared metadata root must be an object")
    selection = validate_prepared_selection(clip_id, clip, existing.get("selection"))
    preparation_geometry = preparation_geometry_contract(clip_id, clip)
    if preparation_geometry is not None:
        if existing.get("preparation_geometry") != preparation_geometry:
            fail(f"{clip_id}: prepared metadata has stale preparation_geometry")
        canvas = preparation_geometry["canvas_shape"]
        validate_prepared_evidence_geometry(
            clip_id, final, canvas["width"], canvas["height"])

    adapter = clip["adapter"]
    try:
        evidence_extensions = ADAPTER_EVIDENCE_EXTENSIONS[adapter]
    except KeyError:
        fail(f"{clip_id}: unsupported adapter evidence contract: {adapter!r}")
    expected_frame_ids = range(len(selection))
    require_prepared_frame_ids(
        clip_id, final, expected_frame_ids, "source frame",
        evidence_extensions["source"])
    if "depth" in evidence_extensions:
        require_prepared_frame_ids(
            clip_id, os.path.join(final, "gt_depth"), expected_frame_ids,
            "depth evidence", evidence_extensions["depth"])
    if "stereo" in evidence_extensions:
        require_prepared_frame_ids(
            clip_id, os.path.join(final, "gt_right"), expected_frame_ids,
            "stereo evidence", evidence_extensions["stereo"])
    if "flow" in evidence_extensions:
        require_prepared_frame_ids(
            clip_id, os.path.join(final, "gt_flow"), range(1, len(selection)),
            "flow evidence", evidence_extensions["flow"])

    dataset = manifest["datasets"][clip["dataset"]]
    for key, expected in (
            ("name", clip["name"]),
            ("dataset", dataset["title"]),
            ("suite", manifest["prepared_suite"])):
        if existing.get(key) != expected:
            fail(f"{clip_id}: prepared metadata {key}={existing.get(key)!r}, "
                 f"expected {expected!r}")
    archive_specs = [dataset["archives"][name] for name in clip["archives"]]
    if any(spec.get("access") == "http_range_zip" for spec in archive_specs):
        authenticate_prepared_range_evidence(clip_id, clip, final)

    refreshed = {
        **existing,
        **prepared_clip_metadata(
            manifest, clip, selection, preparation_geometry=preparation_geometry),
    }
    if preparation_geometry is None:
        refreshed.pop("preparation_geometry", None)
    refreshed.pop("required_gt_stereo", None)
    descriptor, temporary = tempfile.mkstemp(
        prefix="meta.", suffix=".json.tmp", dir=final, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(refreshed, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, meta_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print(f"metadata refreshed: {clip_id} -> {meta_path}", flush=True)


def prepare_clip(manifest, clip_id, clip, downloads_dir, prepared_root):
    dataset = manifest["datasets"][clip["dataset"]]
    archives = {}
    for name in clip["archives"]:
        spec = archive_spec(manifest, clip, name)
        if spec.get("access") == "http_range_zip":
            archives[name] = spec
            continue
        path = confined_child(downloads_dir, spec["filename"], "archive filename")
        if not os.path.exists(path):
            fail(f"archive missing; run without --no-download first: {path}")
        archives[name] = path
    os.makedirs(prepared_root, exist_ok=True)
    final = confined_child(prepared_root, clip_id, "clip ID")
    temp = tempfile.mkdtemp(prefix=clip_id + ".", dir=prepared_root)
    temp = confined_child(
        prepared_root, os.path.basename(temp), "temporary prepared clip directory")
    try:
        if clip["adapter"] == "tum_rgbd_zip":
            selection = prepare_tum(clip_id, clip, dataset, archives, temp,
                                    manifest["prepared_suite"])
        elif clip["adapter"] == "tartanair_v2_zip":
            selection = prepare_tartanair(clip_id, clip, dataset, archives, temp,
                                          manifest["prepared_suite"])
        elif clip["adapter"] == "sintel_stereo_zip":
            selection = prepare_sintel(clip_id, clip, dataset, archives, temp,
                                       manifest["prepared_suite"])
        elif clip["adapter"] == "spring_http_range_zip":
            selection = prepare_spring(clip_id, clip, dataset, archives, temp,
                                       manifest["prepared_suite"])
        elif clip["adapter"] == "vkitti2_tar":
            selection = prepare_vkitti2(clip_id, clip, dataset, archives, temp,
                                        manifest["prepared_suite"])
        else:
            fail(f"unsupported adapter: {clip['adapter']}")
        preparation_geometry = finalize_prepared_geometry(
            clip_id, clip, temp)
        meta = prepared_clip_metadata(
            manifest, clip, selection,
            preparation_geometry=preparation_geometry)
        with open(os.path.join(temp, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        if os.path.isdir(final):
            shutil.rmtree(final)
        os.replace(temp, final)
        print(f"prepared: {clip_id} -> {final}", flush=True)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--cache", help="dataset cache root (default: APOLLO_SBS_DATASETS or manifest)")
    ap.add_argument("--clips", nargs="*", help="clip IDs to prepare (default: all)")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument(
        "--refresh-metadata-only", action="store_true",
        help="authenticate existing prepared evidence and atomically refresh manifest-owned "
             "metadata without downloading or rewriting frames")
    args = ap.parse_args()
    if args.refresh_metadata_only and args.download_only:
        fail("--refresh-metadata-only cannot be combined with --download-only")
    manifest = load_manifest(args.manifest)
    cache = os.path.abspath(args.cache or os.environ.get("APOLLO_SBS_DATASETS")
                            or manifest["default_cache"])
    downloads = os.path.join(cache, "downloads")
    prepared = confined_child(
        os.path.join(cache, "prepared"), manifest["prepared_suite"], "prepared suite")
    clip_ids = args.clips or list(manifest["clips"])
    unknown = sorted(set(clip_ids) - set(manifest["clips"]))
    if unknown:
        fail(f"unknown clip IDs: {unknown}")
    if args.refresh_metadata_only:
        for clip_id in clip_ids:
            refresh_prepared_clip_metadata(
                manifest, clip_id, manifest["clips"][clip_id], prepared)
        print(f"suite root: {prepared}")
        return
    if not args.no_download:
        seen = set()
        for clip_id in clip_ids:
            clip = manifest["clips"][clip_id]
            for archive_name in clip["archives"]:
                spec = archive_spec(manifest, clip, archive_name)
                if spec.get("access") == "http_range_zip":
                    continue
                if spec["filename"] not in seen:
                    download_archive(spec, downloads)
                    seen.add(spec["filename"])
    else:
        seen = set()
        for clip_id in clip_ids:
            clip = manifest["clips"][clip_id]
            for archive_name in clip["archives"]:
                spec = archive_spec(manifest, clip, archive_name)
                if spec.get("access") == "http_range_zip":
                    continue
                if spec["filename"] in seen:
                    continue
                path = confined_child(downloads, spec["filename"], "archive filename")
                if not os.path.exists(path):
                    fail(f"archive missing: {path}")
                if spec.get("sha256") and sha256(path) != spec["sha256"].lower():
                    fail(f"SHA-256 mismatch for {path}")
                seen.add(spec["filename"])
    if not args.download_only:
        for clip_id in clip_ids:
            prepare_clip(manifest, clip_id, manifest["clips"][clip_id], downloads, prepared)
    print(f"suite root: {prepared}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"prepare_public_datasets: {exc}", file=sys.stderr)
        raise SystemExit(2)

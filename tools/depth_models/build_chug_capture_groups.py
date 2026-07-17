#!/usr/bin/env python3
"""Build source-bound CHUG capture groups before dataset selection or splitting.

The visual identity transform is for duplicate detection only.  It explicitly
maps native PQ/BT.2020 into BT.709 for a frozen DINOv2 feature extractor; it is
not Apollo's production color path and is never a training image source.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, __version__ as PILLOW_VERSION

import fetch_chug_hdr_references as chug
from artistic_policy_model import load_depth_anything_small


VISUAL_CACHE_SCHEMA = 1
GROUP_MANIFEST_SCHEMA = 1
FEATURE_DIMENSION = 3072
ANCHOR_COUNT = 5
LANDSCAPE_SIZE = (518, 291)
PORTRAIT_SIZE = (291, 518)
MODEL_SIZE = 518
MAX_VISUAL_JOBS = 8
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
PHASH_COSINE = np.cos(
    (np.arange(32, dtype=np.float64)[None, :] + 0.5)
    * np.arange(8, dtype=np.float64)[:, None]
    * math.pi
    / 32.0
)
IMG_PATTERN = re.compile(r"^Files_([^_]+)_IMG_([0-9]{4})(?:\D|$)", re.IGNORECASE)
PXL_PATTERN = re.compile(
    r"^Files_([^_]+)_PXL_([0-9]{8})_([0-9]{6})[0-9]{3}(?:\D|$)",
    re.IGNORECASE,
)
IOS_PATTERN = re.compile(
    r"^Files_([^_]+)_([0-9]{8})_([0-9]{6})[0-9]*_iOS(?:\D|$)",
    re.IGNORECASE,
)
BATCH_PATTERN = re.compile(r"^Files_([^_]+)_", re.IGNORECASE)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return True


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command, timeout=180):
    process = subprocess.run(command, capture_output=True, timeout=timeout)
    if process.returncode:
        detail = process.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"command failed ({process.returncode}): {detail.strip()}")
    return process.stdout


def executable_version(path: Path):
    output = run_command([str(path), "-version"], timeout=30)
    return output.decode("utf-8", errors="replace").splitlines()[0]


def validate_review_manifest(path: Path, source_manifest):
    review = load_json(path)
    if review.get("schema") != 1 or review.get("dataset") != source_manifest["id"]:
        raise RuntimeError("CHUG capture review manifest is stale")
    visual = review.get("visual_identity", {})
    anchors = visual.get("anchor_fractions")
    if anchors != sorted(anchors or []) or len(anchors or []) != ANCHOR_COUNT:
        raise RuntimeError("CHUG review needs five increasing anchor fractions")
    if any(not 0.0 < float(value) < 1.0 for value in anchors):
        raise RuntimeError("CHUG review anchor fraction is outside (0, 1)")
    if not chug.SHA256_PATTERN.fullmatch(visual.get("depth_weights_sha256", "")):
        raise RuntimeError("CHUG review has no pinned DA-V2 weights")
    thresholds = review.get("perceptual_thresholds", {})
    value = float(thresholds.get("bidirectional_nearest_median_min", 0.0))
    if not 0.0 < value <= 1.0:
        raise RuntimeError("invalid bidirectional perceptual threshold")
    margin = float(thresholds.get("manual_review_margin", -1.0))
    if not 0.0 <= margin < 0.1:
        raise RuntimeError("invalid perceptual manual-review margin")
    capture = review.get("capture_session")
    capture_keys = (
        "img_max_consecutive_gap",
        "img_max_group_span",
        "timestamp_max_consecutive_seconds",
        "timestamp_max_group_span_seconds",
        "perceptual_img_max_delta",
        "perceptual_timestamp_max_seconds",
    )
    if not isinstance(capture, dict) or any(
            not isinstance(capture.get(key), int) or capture[key] <= 0
            for key in capture_keys):
        raise RuntimeError("invalid CHUG capture-session thresholds")
    positives = review.get("confirmed_positive_groups")
    if not isinstance(positives, list) or not positives:
        raise RuntimeError("CHUG review needs confirmed positive groups")
    negatives = review.get("hard_negative_pairs")
    if not isinstance(negatives, list) or not negatives:
        raise RuntimeError("CHUG review needs explicit hard-negative pairs")
    sessions = review.get("manual_capture_session_groups")
    if not isinstance(sessions, list):
        raise RuntimeError("CHUG review manual capture sessions are invalid")
    overrides = review.get("perceptual_boundary_overrides")
    if not isinstance(overrides, list):
        raise RuntimeError("CHUG perceptual boundary overrides are invalid")
    for override in overrides:
        if override.get("decision") not in {"accept", "reject"}:
            raise RuntimeError("CHUG perceptual boundary override has invalid decision")
    return review


def validate_candidate_manifest(path: Path, source_manifest):
    payload = load_json(path)
    required = {
        "schema": 1,
        "dataset": source_manifest["id"],
        "repository_commit": source_manifest["repository_commit"],
        "probe_query_schema": chug.CANDIDATE_PROBE_QUERY_SCHEMA,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"candidate probe manifest has stale {key}")
    semantic = {
        key: payload[key]
        for key in (
            "schema",
            "dataset",
            "repository_commit",
            "probe_query_schema",
            "producer_ffprobe_version",
            "catalog_reference_rows",
            "valid_candidates",
            "contract_rejections",
        )
    }
    semantic["summary"] = {
        key: payload["summary"][key]
        for key in ("probed", "valid", "contract_rejected", "probe_errors")
    }
    if chug.canonical_sha256(semantic) != payload.get("semantic_sha256"):
        raise RuntimeError("candidate probe semantic identity is invalid")
    if payload["summary"]["probe_errors"] or not payload["valid_candidates"]:
        raise RuntimeError("candidate probe manifest is incomplete")
    return payload


def visual_source_identity(dataset_root: Path, row, source_manifest):
    local = dataset_root / source_manifest["videos"]["relative_directory"] / f"{row['video_id']}.mp4"
    reported_bytes = row["candidate_audit"].get("reported_bytes")
    if local.is_file():
        if reported_bytes is not None and local.stat().st_size != reported_bytes:
            raise RuntimeError(f"{row['video_id']}: local master byte count changed")
        source = local
    else:
        source = row["candidate_url"]
    return source, {
        "url": row["candidate_url"],
        "reported_bytes": reported_bytes,
    }


def visual_binding(
        row,
        source_manifest,
        candidate_manifest,
        review,
        ffmpeg_version,
        weights_sha256,
        source_identity):
    audit = row["candidate_audit"]
    return {
        "video_id": row["video_id"],
        "source_url": row["candidate_url"],
        "repository_commit": source_manifest["repository_commit"],
        "candidate_probe_semantic_sha256": candidate_manifest["semantic_sha256"],
        "candidate_audit_sha256": chug.canonical_sha256(audit),
        "ffmpeg_version": ffmpeg_version,
        "fingerprint_transform": review["visual_identity"]["fingerprint_transform"],
        "anchor_fractions": review["visual_identity"]["anchor_fractions"],
        "feature_contract": review["visual_identity"]["feature_contract"],
        "depth_weights_sha256": weights_sha256,
        "source_object": source_identity,
        "thumbnail_encoder": f"Pillow {PILLOW_VERSION}",
    }


def decode_embedding(identity):
    shape = identity.get("embedding_shape")
    if shape != [ANCHOR_COUNT, FEATURE_DIMENSION]:
        raise RuntimeError("visual identity embedding shape is stale")
    try:
        raw = base64.b64decode(identity["embedding_q15"], validate=True)
    except Exception as error:
        raise RuntimeError("visual identity embedding is not valid base64") from error
    expected = ANCHOR_COUNT * FEATURE_DIMENSION * 2
    if len(raw) != expected:
        raise RuntimeError("visual identity embedding byte count is stale")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(shape)
    values /= 32767.0
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms < 0.9):
        raise RuntimeError("visual identity embedding is invalid")
    return values / norms


def load_visual_cache(path: Path, binding, dataset_root: Path):
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
        if payload.get("schema") != VISUAL_CACHE_SCHEMA or payload.get("binding") != binding:
            return None
        identity = payload["identity"]
        decode_embedding(identity)
        decode_canonical_gray(identity)
        thumbnail = chug.safe_output_path(dataset_root, identity["thumbnail"]["path"])
        if chug.file_identity(thumbnail) != {
                "bytes": identity["thumbnail"]["bytes"],
                "sha256": identity["thumbnail"]["sha256"]}:
            return None
        return payload
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None


def extract_anchor_tile(ffmpeg: Path, source, row, review):
    duration = float(row["candidate_audit"]["duration_seconds"])
    fractions = review["visual_identity"]["anchor_fractions"]
    width, height = LANDSCAPE_SIZE if row["orientation"] == "Landscape" else PORTRAIT_SIZE
    anchor_seconds = [duration * float(fraction) for fraction in fractions]
    interval = anchor_seconds[1] - anchor_seconds[0]
    rate = 1.0 / interval
    filters = (
        f"fps=fps={rate:.9f}:start_time={anchor_seconds[0]:.6f},"
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=full,"
        f"scale={width}:{height}:flags=lanczos,format=rgb24,"
        f"tile={ANCHOR_COUNT}x1"
    )
    command = [str(ffmpeg), "-v", "error", "-i", str(source)]
    command.extend((
        "-vf", filters,
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ))
    raw = run_command(command, timeout=240)
    expected = ANCHOR_COUNT * width * height * 3
    if len(raw) != expected:
        raise RuntimeError(f"{row['video_id']}: anchor tile has {len(raw)} bytes, expected {expected}")
    tile = np.frombuffer(raw, dtype=np.uint8).reshape(height, ANCHOR_COUNT * width, 3).copy()
    frames = np.stack(
        [tile[:, index * width:(index + 1) * width] for index in range(ANCHOR_COUNT)]
    )
    return tile, frames, anchor_seconds


def preprocess_frames(frames):
    letterboxed = np.zeros(
        (frames.shape[0], MODEL_SIZE, MODEL_SIZE, 3),
        dtype=np.uint8,
    )
    height, width = frames.shape[1:3]
    top = (MODEL_SIZE - height) // 2
    left = (MODEL_SIZE - width) // 2
    letterboxed[:, top:top + height, left:left + width] = frames
    image = letterboxed.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return torch.from_numpy(image.transpose(0, 3, 1, 2).copy())


def load_feature_model(depth_root: Path, weights: Path, device):
    model = load_depth_anything_small(depth_root, weights)
    model.requires_grad_(False)
    model.eval().to(device)
    return model


def embed_frame_batch(model, images, device):
    tensor = torch.cat([preprocess_frames(frames) for frames in images]).to(device)
    indices = model.intermediate_layer_idx[model.encoder]
    with torch.inference_mode():
        with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"):
            features = model.pretrained.get_intermediate_layers(
                tensor,
                indices,
                return_class_token=True,
            )
            pooled = []
            for tokens, class_token in features:
                pooled.extend((class_token.float(), tokens.float().mean(dim=1)))
            embedding = torch.cat(pooled, dim=1)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
    values = embedding.cpu().numpy()
    if values.shape[1] != FEATURE_DIMENSION or not np.isfinite(values).all():
        raise RuntimeError(f"unexpected DINO visual feature shape: {values.shape}")
    return values.reshape(len(images), ANCHOR_COUNT, FEATURE_DIMENSION)


def encode_embedding(values):
    quantized = np.clip(np.rint(values * 32767.0), -32767, 32767).astype("<i2")
    return base64.b64encode(quantized.tobytes(order="C")).decode("ascii")


def canonical_gray_frames(frames):
    grayscale = []
    for frame in frames:
        luma = np.clip(
            np.rint(
                frame[:, :, 0].astype(np.float32) * 0.2126
                + frame[:, :, 1].astype(np.float32) * 0.7152
                + frame[:, :, 2].astype(np.float32) * 0.0722
            ),
            0,
            255,
        ).astype(np.uint8)
        image = Image.fromarray(luma, mode="L").resize((64, 64), Image.Resampling.LANCZOS)
        grayscale.append(np.asarray(image, dtype=np.uint8))
    return np.stack(grayscale)


def perceptual_hash(gray):
    reduced = gray.reshape(32, 2, 32, 2).mean(axis=(1, 3), dtype=np.float64)
    coefficients = PHASH_COSINE @ reduced @ PHASH_COSINE.T
    threshold = float(np.median(coefficients.reshape(-1)[1:]))
    bits = coefficients.reshape(-1) > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def decode_canonical_gray(identity):
    if identity.get("canonical_gray_shape") != [ANCHOR_COUNT, 64, 64]:
        raise RuntimeError("visual identity canonical-gray shape is stale")
    try:
        raw = base64.b64decode(identity["canonical_gray_u8"], validate=True)
    except Exception as error:
        raise RuntimeError("canonical-gray payload is not valid base64") from error
    if len(raw) != ANCHOR_COUNT * 64 * 64:
        raise RuntimeError("canonical-gray byte count is stale")
    values = np.frombuffer(raw, dtype=np.uint8).reshape(ANCHOR_COUNT, 64, 64)
    expected_hashes = [perceptual_hash(frame) for frame in values]
    if identity.get("phash64") != expected_hashes:
        raise RuntimeError("canonical-gray perceptual hashes are stale")
    return values


def hamming_hex(first, second):
    return (int(first, 16) ^ int(second, 16)).bit_count()


def global_ssim(first, second):
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first_mean = float(first.mean())
    second_mean = float(second.mean())
    first_variance = float(first.var())
    second_variance = float(second.var())
    covariance = float(((first - first_mean) * (second - second_mean)).mean())
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * first_mean * second_mean + c1) * (2.0 * covariance + c2)
    denominator = (
        (first_mean ** 2 + second_mean ** 2 + c1)
        * (first_variance + second_variance + c2)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def exact_recut_score(first_identity, second_identity, max_hamming=6, minimum_ssim=0.92):
    first_gray = decode_canonical_gray(first_identity)
    second_gray = decode_canonical_gray(second_identity)
    first_hashes = first_identity["phash64"]
    second_hashes = second_identity["phash64"]
    candidates = []
    for first_index in range(ANCHOR_COUNT):
        for second_index in range(ANCHOR_COUNT):
            distance = hamming_hex(first_hashes[first_index], second_hashes[second_index])
            if distance > max_hamming:
                continue
            similarity = global_ssim(first_gray[first_index], second_gray[second_index])
            if similarity >= minimum_ssim:
                candidates.append((first_index, second_index, distance, similarity))
    lengths = {}
    previous = {}
    best = None
    for index, candidate in enumerate(candidates):
        first_index, second_index = candidate[:2]
        options = [
            other
            for other in range(index)
            if candidates[other][0] < first_index and candidates[other][1] < second_index
        ]
        parent = max(options, key=lambda item: lengths[item], default=None)
        lengths[index] = 1 + (lengths[parent] if parent is not None else 0)
        previous[index] = parent
        if best is None or lengths[index] > lengths[best]:
            best = index
    matches = []
    while best is not None:
        matches.append(candidates[best])
        best = previous[best]
    matches.reverse()
    return {
        "monotonic_match_count": len(matches),
        "matches": [
            {
                "first_anchor": item[0],
                "second_anchor": item[1],
                "phash_hamming": item[2],
                "ssim": item[3],
            }
            for item in matches
        ],
        "max_phash_hamming": max_hamming,
        "minimum_ssim": minimum_ssim,
    }


def publish_visual_identity(
        dataset_root: Path,
        cache_path: Path,
        binding,
        row,
        frames,
        tile,
        anchor_seconds,
        embedding):
    thumbnail = dataset_root / "visual_thumbnails" / f"{row['video_id']}.png"
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    temporary = thumbnail.with_name(thumbnail.stem + ".writing.png")
    Image.fromarray(tile, mode="RGB").save(temporary, format="PNG")
    temporary.replace(thumbnail)
    canonical_gray = canonical_gray_frames(frames)
    identity = {
        "anchor_seconds": anchor_seconds,
        "embedding_shape": [ANCHOR_COUNT, FEATURE_DIMENSION],
        "embedding_q15": encode_embedding(embedding),
        "canonical_gray_shape": [ANCHOR_COUNT, 64, 64],
        "canonical_gray_u8": base64.b64encode(canonical_gray.tobytes(order="C")).decode("ascii"),
        "phash64": [perceptual_hash(frame) for frame in canonical_gray],
        "thumbnail": {
            "path": thumbnail.relative_to(dataset_root).as_posix(),
            **chug.file_identity(thumbnail),
        },
        "training_eligible": False,
        "purpose": "capture-group perceptual identity only",
    }
    payload = {
        "schema": VISUAL_CACHE_SCHEMA,
        "binding": binding,
        "identity": identity,
    }
    chug.atomic_write_json(cache_path, payload)
    return payload


def build_visual_identities(
        rows,
        dataset_root: Path,
        source_manifest,
        candidate_manifest,
        review,
        ffmpeg: Path,
        depth_root: Path,
        weights: Path,
        jobs,
        clip_batch_size,
        device_name,
        refresh=False,
        cache_only=False):
    if not 1 <= jobs <= MAX_VISUAL_JOBS:
        raise RuntimeError(f"--jobs must be between 1 and {MAX_VISUAL_JOBS}")
    ffmpeg_version = executable_version(ffmpeg)
    weights_sha256 = chug.sha256(weights)
    if weights_sha256 != review["visual_identity"]["depth_weights_sha256"]:
        raise RuntimeError("DA-V2 Small weights do not match the reviewed visual identity")
    identities = {}
    missing = []
    for row in rows:
        source, source_identity = visual_source_identity(dataset_root, row, source_manifest)
        binding = visual_binding(
            row,
            source_manifest,
            candidate_manifest,
            review,
            ffmpeg_version,
            weights_sha256,
            source_identity,
        )
        cache_path = dataset_root / "visual_identity" / f"{row['video_id']}.json"
        cached = None if refresh else load_visual_cache(cache_path, binding, dataset_root)
        if cached is None:
            missing.append((row, binding, cache_path, source))
        else:
            identities[row["video_id"]] = cached
    if not missing:
        return identities, {
            "cache_hits": len(rows),
            "generated": 0,
            "ffmpeg_version": ffmpeg_version,
            "weights_sha256": weights_sha256,
        }
    if cache_only:
        raise RuntimeError(
            f"--cache-only requested but {len(missing)} visual identity cache(s) are missing or stale"
        )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_feature_model(depth_root, weights, device)
    pending = []
    generated = 0

    def flush_pending():
        nonlocal generated
        if not pending:
            return
        embeddings = embed_frame_batch(model, [item[3] for item in pending], device)
        for item, embedding in zip(pending, embeddings):
            row, binding, cache_path, _frames, tile, anchor_seconds = item
            payload = publish_visual_identity(
                dataset_root,
                cache_path,
                binding,
                row,
                _frames,
                tile,
                anchor_seconds,
                embedding,
            )
            identities[row["video_id"]] = payload
            generated += 1
            print(f"visual identities {len(identities)}/{len(rows)}: {row['video_id']}", flush=True)
        pending.clear()

    def extract(item):
        row, binding, cache_path, source = item
        tile, frames, anchor_seconds = extract_anchor_tile(ffmpeg, source, row, review)
        return row, binding, cache_path, frames, tile, anchor_seconds

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(extract, item): item[0] for item in missing}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                pending.append(future.result())
            except Exception as error:
                raise RuntimeError(f"{row['video_id']}: visual extraction failed: {error}") from error
            if len(pending) >= clip_batch_size:
                flush_pending()
    flush_pending()
    return identities, {
        "cache_hits": len(rows) - len(missing),
        "generated": generated,
        "ffmpeg_version": ffmpeg_version,
        "weights_sha256": weights_sha256,
    }


def capture_coordinate(row):
    name = row["content_name"]
    match = IMG_PATTERN.match(name)
    if match:
        return ("IMG", match.group(1).lower()), int(match.group(2))
    match = PXL_PATTERN.match(name)
    if match:
        date = match.group(2)
        time = match.group(3)
        seconds = int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:])
        return ("PXL", match.group(1).lower(), date), seconds
    match = IOS_PATTERN.match(name)
    if match:
        date = match.group(2)
        time = match.group(3)
        seconds = int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:])
        return ("iOS", match.group(1).lower(), date), seconds
    return None, None


def capture_batch(row):
    match = BATCH_PATTERN.match(row["content_name"])
    if not match:
        raise RuntimeError(f"CHUG content name has no capture batch: {row['content_name']}")
    return match.group(1).lower()


def plausible_perceptual_session(first, second, review):
    """Require semantic matches to come from one plausible capture neighborhood."""
    first_key, first_coordinate = capture_coordinate(first)
    second_key, second_coordinate = capture_coordinate(second)
    if first_key is None or first_key != second_key:
        return False
    config = review["capture_session"]
    if first_key[0] == "IMG":
        maximum = int(config["perceptual_img_max_delta"])
    else:
        maximum = int(config["perceptual_timestamp_max_seconds"])
    return abs(first_coordinate - second_coordinate) <= maximum


def _bounded_coordinate_clusters(items, maximum_gap, maximum_span):
    ordered = sorted(items, key=lambda item: (item[0], item[1]))
    clusters = []
    current = []
    start = None
    previous = None
    for coordinate, video_id in ordered:
        if not current or (
                coordinate - previous <= maximum_gap
                and coordinate - start <= maximum_span):
            current.append((coordinate, video_id))
        else:
            clusters.append(current)
            current = [(coordinate, video_id)]
        if len(current) == 1:
            start = coordinate
        previous = coordinate
    if current:
        clusters.append(current)
    return clusters


def capture_session_edges(rows, review):
    grouped = {}
    for row in rows:
        key, coordinate = capture_coordinate(row)
        if key is not None:
            grouped.setdefault(key, []).append((coordinate, row["video_id"]))
    config = review["capture_session"]
    edges = []
    for key in sorted(grouped):
        if key[0] == "IMG":
            maximum_gap = int(config["img_max_consecutive_gap"])
            maximum_span = int(config["img_max_group_span"])
        else:
            maximum_gap = int(config["timestamp_max_consecutive_seconds"])
            maximum_span = int(config["timestamp_max_group_span_seconds"])
        for cluster in _bounded_coordinate_clusters(grouped[key], maximum_gap, maximum_span):
            ids = [item[1] for item in cluster]
            for first, second in zip(ids, ids[1:]):
                edges.append({
                    "first": first,
                    "second": second,
                    "kind": "capture_filename_session",
                    "capture_key": list(key),
                })
    return edges


def pair_score(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first_aggregate = first.mean(axis=0)
    second_aggregate = second.mean(axis=0)
    first_aggregate /= max(float(np.linalg.norm(first_aggregate)), 1e-8)
    second_aggregate /= max(float(np.linalg.norm(second_aggregate)), 1e-8)
    aggregate = float(np.dot(first_aggregate, second_aggregate))
    matrix = first @ second.T
    nearest = np.concatenate((matrix.max(axis=1), matrix.max(axis=0)))
    remaining_first = set(range(ANCHOR_COUNT))
    remaining_second = set(range(ANCHOR_COUNT))
    matches = []
    while remaining_first and remaining_second:
        best = max(
            (
                (float(matrix[i, j]), i, j)
                for i in remaining_first
                for j in remaining_second
            ),
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        matches.append(best[0])
        remaining_first.remove(best[1])
        remaining_second.remove(best[2])
    return {
        "aggregate_cosine": aggregate,
        "best_anchor_cosine": matches[0],
        "top3_anchor_cosine_mean": float(sum(matches[:3]) / 3.0),
        "bidirectional_nearest_median": float(np.median(nearest)),
        "matched_anchor_cosines": matches,
    }


def is_perceptual_match(score, thresholds):
    return score["bidirectional_nearest_median"] >= float(
        thresholds["bidirectional_nearest_median_min"]
    )


def build_groups(rows, identities, review):
    video_ids = sorted(row["video_id"] for row in rows)
    rows_by_id = {row["video_id"]: row for row in rows}
    union = UnionFind(video_ids)
    edge_by_pair = {}
    manual_decisions = {}

    def register_manual_pairs(group, decision):
        ids = group["video_ids"]
        for first_index in range(len(ids)):
            for second_index in range(first_index + 1, len(ids)):
                pair = tuple(sorted((ids[first_index], ids[second_index])))
                previous = manual_decisions.setdefault(pair, decision)
                if previous != decision:
                    raise RuntimeError(f"conflicting manual pair decision: {pair}")

    for group in review["confirmed_positive_groups"]:
        register_manual_pairs(group, "accept")
    for group in review["manual_capture_session_groups"]:
        register_manual_pairs(group, "accept")
    for pair in review["hard_negative_pairs"]:
        register_manual_pairs(pair, "reject")
    for pair in review["perceptual_boundary_overrides"]:
        register_manual_pairs(pair, pair["decision"])
    for pair in manual_decisions:
        missing = set(pair) - set(union.parent)
        if missing:
            raise RuntimeError(f"manual CHUG review references invalid video IDs: {sorted(missing)}")

    def add_edge(first, second, evidence):
        if first == second:
            return
        pair = tuple(sorted((first, second)))
        edge_by_pair.setdefault(pair, []).append(evidence)
        union.union(*pair)

    for edge in capture_session_edges(rows, review):
        add_edge(edge["first"], edge["second"], edge)
    for group in review["confirmed_positive_groups"]:
        ids = group["video_ids"]
        for video_id in ids:
            if video_id not in union.parent:
                raise RuntimeError(f"reviewed positive is absent from valid refs: {video_id}")
        for first, second in zip(ids, ids[1:]):
            add_edge(first, second, {
                "first": first,
                "second": second,
                "kind": "human_confirmed_positive",
                "review_name": group["name"],
            })
    for group in review["manual_capture_session_groups"]:
        ids = group["video_ids"]
        for video_id in ids:
            if video_id not in union.parent:
                raise RuntimeError(f"reviewed capture session is absent from valid refs: {video_id}")
        for first, second in zip(ids, ids[1:]):
            add_edge(first, second, {
                "first": first,
                "second": second,
                "kind": "human_confirmed_capture_session",
                "review_name": group["name"],
            })
    embeddings = {
        video_id: decode_embedding(payload["identity"])
        for video_id, payload in identities.items()
    }
    aggregate = []
    for video_id in video_ids:
        value = embeddings[video_id].mean(axis=0)
        aggregate.append(value / max(float(np.linalg.norm(value)), 1e-8))
    aggregate = np.stack(aggregate)
    coarse = aggregate @ aggregate.T
    batches = {row["video_id"]: capture_batch(row) for row in rows}
    thresholds = review["perceptual_thresholds"]
    coarse_min = min(
        float(thresholds["bidirectional_nearest_median_min"]),
        0.95,
    ) - 0.05
    pair_scores = []
    review_required_pairs = []
    for first_index in range(len(video_ids)):
        for second_index in range(first_index + 1, len(video_ids)):
            if batches[video_ids[first_index]] != batches[video_ids[second_index]]:
                continue
            first = video_ids[first_index]
            second = video_ids[second_index]
            pair = tuple(sorted((first, second)))
            decision = manual_decisions.get(pair)
            if decision == "accept":
                add_edge(first, second, {
                    "first": first,
                    "second": second,
                    "kind": "human_confirmed_pair",
                })
                continue
            if decision == "reject":
                continue
            if not plausible_perceptual_session(rows_by_id[first], rows_by_id[second], review):
                continue
            if float(coarse[first_index, second_index]) < coarse_min:
                continue
            score = pair_score(embeddings[first], embeddings[second])
            threshold = float(thresholds["bidirectional_nearest_median_min"])
            margin = float(thresholds["manual_review_margin"])
            if abs(score["bidirectional_nearest_median"] - threshold) <= margin:
                if decision is None:
                    review_required_pairs.append({
                        "first": first,
                        "second": second,
                        "score": score,
                        "threshold": threshold,
                        "margin": margin,
                    })
                    continue
            if is_perceptual_match(score, thresholds):
                evidence = {
                    "first": first,
                    "second": second,
                    "kind": "dav2_multiframe_perceptual",
                    "score": score,
                }
                add_edge(first, second, evidence)
                pair_scores.append(evidence)
    exact_recut_edges = []
    phashes = {
        video_id: identities[video_id]["identity"]["phash64"]
        for video_id in video_ids
    }
    for first_index in range(len(video_ids)):
        for second_index in range(first_index + 1, len(video_ids)):
            first = video_ids[first_index]
            second = video_ids[second_index]
            if batches[first] == batches[second]:
                continue
            coarse_matches = sum(
                hamming_hex(first_hash, second_hash) <= 6
                for first_hash in phashes[first]
                for second_hash in phashes[second]
            )
            if coarse_matches < 3:
                continue
            score = exact_recut_score(
                identities[first]["identity"],
                identities[second]["identity"],
            )
            if score["monotonic_match_count"] < 3:
                continue
            evidence = {
                "first": first,
                "second": second,
                "kind": "cross_batch_exact_recut",
                "score": score,
            }
            add_edge(first, second, evidence)
            exact_recut_edges.append(evidence)
    components = {}
    for video_id in video_ids:
        components.setdefault(union.find(video_id), []).append(video_id)
    groups = []
    membership = {}
    grouping_contract_sha256 = chug.canonical_sha256({
        "visual_identity": review["visual_identity"],
        "capture_session": review["capture_session"],
        "perceptual_thresholds": review["perceptual_thresholds"],
        "confirmed_positive_groups": review["confirmed_positive_groups"],
        "manual_capture_session_groups": review["manual_capture_session_groups"],
        "hard_negative_pairs": review["hard_negative_pairs"],
        "perceptual_boundary_overrides": review["perceptual_boundary_overrides"],
    })
    for members in sorted((sorted(value) for value in components.values()), key=lambda value: value[0]):
        group_id = chug.stable_hash(
            "chug-capture-group-v2",
            grouping_contract_sha256,
            *members,
        )
        for video_id in members:
            membership[video_id] = group_id
        groups.append({
            "capture_group_id": group_id,
            "members": members,
            "member_count": len(members),
        })
    calibration = []
    for group in review["confirmed_positive_groups"]:
        ids = group["video_ids"]
        for first_index in range(len(ids)):
            for second_index in range(first_index + 1, len(ids)):
                score = pair_score(embeddings[ids[first_index]], embeddings[ids[second_index]])
                calibration.append({
                    "label": "positive",
                    "review_name": group["name"],
                    "first": ids[first_index],
                    "second": ids[second_index],
                    "score": score,
                    "same_group": membership[ids[first_index]] == membership[ids[second_index]],
                })
    for pair in review["hard_negative_pairs"]:
        first, second = pair["video_ids"]
        if first not in embeddings or second not in embeddings:
            raise RuntimeError(f"reviewed hard negative is absent: {pair['video_ids']}")
        score = pair_score(embeddings[first], embeddings[second])
        same_group = membership[first] == membership[second]
        calibration.append({
            "label": "negative",
            "review_name": pair["name"],
            "first": first,
            "second": second,
            "score": score,
            "same_group": same_group,
        })
        if same_group:
            raise RuntimeError(f"hard-negative pair joined one capture group: {pair['name']}")
    if any(not item["same_group"] for item in calibration if item["label"] == "positive"):
        raise RuntimeError("a confirmed positive capture pair was split")
    return groups, membership, {
        "edges": [
            {"pair": list(pair), "evidence": evidence}
            for pair, evidence in sorted(edge_by_pair.items())
        ],
        "perceptual_edges": pair_scores,
        "exact_recut_edges": exact_recut_edges,
        "calibration_pairs": calibration,
        "grouping_contract_sha256": grouping_contract_sha256,
        "review_required_pairs": review_required_pairs,
    }


def build_manifest(
        source_manifest,
        candidate_manifest,
        review_path: Path,
        review,
        identities,
        execution,
        groups,
        membership,
        evidence):
    visual_identities = []
    for video_id in sorted(identities):
        payload = identities[video_id]
        visual_identities.append({
            "video_id": video_id,
            "binding_sha256": chug.canonical_sha256(payload["binding"]),
            "identity_sha256": chug.canonical_sha256(payload["identity"]),
            "capture_group_id": membership[video_id],
        })
    semantic = {
        "schema": GROUP_MANIFEST_SCHEMA,
        "dataset": source_manifest["id"],
        "repository_commit": source_manifest["repository_commit"],
        "candidate_probe_semantic_sha256": candidate_manifest["semantic_sha256"],
        "review_manifest": {
            "path": review_path.name,
            **chug.file_identity(review_path),
        },
        "visual_contract": review["visual_identity"],
        "capture_session_contract": review["capture_session"],
        "perceptual_thresholds": review["perceptual_thresholds"],
        "valid_reference_count": len(identities),
        "visual_identities": visual_identities,
        "capture_groups": groups,
        "evidence": evidence,
        "status": "review_required" if evidence["review_required_pairs"] else "decision_ready",
    }
    return {
        **semantic,
        "semantic_sha256": chug.canonical_sha256(semantic),
        "execution": {
            **execution,
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=Path(__file__).with_name("chug_capture_group_review.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-visuals", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="fail instead of extracting features when any visual identity cache is missing or stale",
    )
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    source_path = dataset / "source_metadata" / "apollo_chug_source_manifest.json"
    candidate_path = dataset / "candidate_probe_manifest.json"
    source_manifest = chug.load_source_manifest(source_path)
    candidate_manifest = validate_candidate_manifest(candidate_path, source_manifest)
    review_path = args.review_manifest.resolve()
    review = validate_review_manifest(review_path, source_manifest)
    if not args.ffmpeg.is_file() or not args.depth_weights.is_file():
        raise RuntimeError("ffmpeg and DA-V2 weights are required")
    if not args.depth_anything_root.is_dir():
        raise RuntimeError("Depth Anything V2 source root is required")
    rows = candidate_manifest["valid_candidates"]
    identities, execution = build_visual_identities(
        rows,
        dataset,
        source_manifest,
        candidate_manifest,
        review,
        args.ffmpeg.resolve(),
        args.depth_anything_root.resolve(),
        args.depth_weights.resolve(),
        args.jobs,
        args.clip_batch_size,
        args.device,
        refresh=args.refresh_visuals,
        cache_only=args.cache_only,
    )
    if set(identities) != {row["video_id"] for row in rows}:
        raise RuntimeError("visual identity coverage is incomplete")
    groups, membership, evidence = build_groups(rows, identities, review)
    manifest = build_manifest(
        source_manifest,
        candidate_manifest,
        review_path,
        review,
        identities,
        execution,
        groups,
        membership,
        evidence,
    )
    output = args.output.resolve() if args.output else dataset / "capture_group_manifest.json"
    chug.atomic_write_json(output, manifest)
    print(json.dumps({
        "valid_references": len(rows),
        "capture_groups": len(groups),
        "multi_member_groups": sum(group["member_count"] > 1 for group in groups),
        "largest_group": max(group["member_count"] for group in groups),
        "cache_hits": execution["cache_hits"],
        "generated": execution["generated"],
        "semantic_sha256": manifest["semantic_sha256"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    if manifest["status"] != "decision_ready":
        raise RuntimeError(
            f"{len(evidence['review_required_pairs'])} perceptual pair(s) need manual review; "
            f"see {output}"
        )


if __name__ == "__main__":
    main()

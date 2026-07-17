#!/usr/bin/env python3
"""Fetch and authenticate pristine native-PQ references from the CHUG dataset.

The tool deliberately stops at native video.  It never decodes PQ through an
implicit 8-bit RGB path; a future training derivative must use a separately
versioned HDR-to-SDR transform and retain the native clip receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


MAX_DOWNLOAD_JOBS = 8
CHUNK_BYTES = 1024 * 1024
CANDIDATE_PROBE_CACHE_SCHEMA = 1
CANDIDATE_PROBE_QUERY_SCHEMA = "native-pq-stream-metadata-v1"
VIDEO_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
YUV420_DEPTH_PATTERN = re.compile(r"^yuv420p([0-9]+)(?:le|be)$")
REQUIRED_COLUMNS = {
    "Video",
    "ref",
    "name",
    "bitladder",
    "resolution",
    "bitrate",
    "orientation",
    "framerate",
    "content_name",
    "height",
    "width",
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_output_path(root: Path, relative_path):
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"unsafe relative output path: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"output path escapes root: {relative_path!r}") from error
    return resolved


def _positive_int(value, label):
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be a positive integer") from error
    if parsed <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return parsed


def load_source_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise RuntimeError(f"unsupported CHUG source manifest schema: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", payload.get("repository_commit", "")):
        raise RuntimeError("CHUG source manifest has no pinned repository commit")
    if payload.get("license") != "CC BY-NC-SA 4.0":
        raise RuntimeError("CHUG must use the pinned CC BY-NC-SA 4.0 license")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"csv", "license"}:
        raise RuntimeError("CHUG source manifest needs pinned CSV and license metadata")
    for name, item in metadata.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid CHUG metadata entry: {name}")
        safe_output_path(Path.cwd(), item.get("relative_path", ""))
        if not str(item.get("url", "")).startswith("https://"):
            raise RuntimeError(f"{name}: metadata URL must use HTTPS")
        _positive_int(item.get("bytes"), f"{name} bytes")
        if not SHA256_PATTERN.fullmatch(item.get("sha256", "")):
            raise RuntimeError(f"{name}: invalid SHA-256")
    contract = payload.get("catalog_contract", {})
    total = _positive_int(contract.get("total_rows"), "catalog total_rows")
    references = _positive_int(contract.get("reference_rows"), "catalog reference_rows")
    degraded = _positive_int(contract.get("degraded_rows"), "catalog degraded_rows")
    if references + degraded != total or contract.get("reference_value") != "1":
        raise RuntimeError("CHUG catalog row contract is inconsistent")
    videos = payload.get("videos", {})
    if "{video_id}" not in videos.get("url_template", ""):
        raise RuntimeError("CHUG video URL template has no {video_id} token")
    if not videos["url_template"].startswith("https://"):
        raise RuntimeError("CHUG video URL must use HTTPS")
    safe_output_path(Path.cwd(), videos.get("relative_directory", ""))
    selection = payload.get("selection", {})
    default_limit = _positive_int(selection.get("default_limit"), "selection default_limit")
    if default_limit > references:
        raise RuntimeError("selection default exceeds the CHUG reference count")
    buckets = selection.get("framerate_buckets")
    if not isinstance(buckets, list) or len(buckets) < 2:
        raise RuntimeError("selection needs frame-rate buckets")
    previous = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket.get("name"):
            raise RuntimeError("frame-rate bucket has no name")
        maximum = bucket.get("max_exclusive")
        if index == len(buckets) - 1:
            if maximum is not None:
                raise RuntimeError("last frame-rate bucket must be open-ended")
        else:
            try:
                maximum = float(maximum)
            except (TypeError, ValueError) as error:
                raise RuntimeError("frame-rate bucket maximum must be numeric") from error
            if not math.isfinite(maximum) or maximum <= previous:
                raise RuntimeError("frame-rate bucket maxima must be increasing")
            previous = maximum
    split = payload.get("split", {})
    weights = split.get("weights")
    if not isinstance(weights, dict) or set(weights) != {
            "training", "development", "test"}:
        raise RuntimeError("CHUG content split needs training/development/test weights")
    if any(_positive_int(value, f"split {name} weight") <= 0 for name, value in weights.items()):
        raise RuntimeError("CHUG content split weights are invalid")
    color = payload.get("native_color_contract", {})
    required_color = {
        "codec": "hevc",
        "color_range": "tv",
        "color_primaries": "bt2020",
        "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
        "frame_decode": "forbidden",
    }
    if any(color.get(key) != value for key, value in required_color.items()):
        raise RuntimeError("CHUG native PQ color contract is incomplete")
    if float(color.get("minimum_duration_seconds", 0.0)) <= 0.0:
        raise RuntimeError("CHUG minimum duration is invalid")
    if float(color.get("maximum_duration_seconds", 0.0)) <= float(
            color["minimum_duration_seconds"]):
        raise RuntimeError("CHUG maximum duration is invalid")
    return payload


def file_identity(path: Path):
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _header(headers, name):
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, candidate in headers.items():
        if str(key).lower() == lowered:
            return candidate
    return None


def _request(url, offset):
    headers = {"User-Agent": "Apollo-3D-CHUG-HDR/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def _remove_partial(partial: Path, marker: Path):
    partial.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)


def download_http(
        url,
        destination: Path,
        expected_bytes=None,
        expected_sha256=None,
        timeout=90):
    """Download one URL atomically, resuming a matching .part file when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        identity = file_identity(destination)
        size_ok = expected_bytes is None or identity["bytes"] == expected_bytes
        hash_ok = expected_sha256 is None or identity["sha256"] == expected_sha256
        if size_ok and hash_ok:
            return {"status": "existing_verified", **identity}
    partial = destination.with_name(destination.name + ".part")
    marker = destination.with_name(destination.name + ".part.url")
    previous_url = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if partial.is_file() and previous_url != url:
        _remove_partial(partial, marker)
    marker.write_text(url + "\n", encoding="utf-8")
    restarted = False
    for attempt in range(2):
        offset = partial.stat().st_size if partial.is_file() else 0
        try:
            response = urllib.request.urlopen(_request(url, offset), timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code == 416 and offset and attempt == 0:
                _remove_partial(partial, marker)
                marker.write_text(url + "\n", encoding="utf-8")
                restarted = True
                continue
            raise RuntimeError(f"download failed for {url}: HTTP {error.code}") from error
        with response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            content_length = _header(response.headers, "Content-Length")
            expected_total = None
            if offset and status == 206:
                content_range = _header(response.headers, "Content-Range") or ""
                match = CONTENT_RANGE_PATTERN.fullmatch(str(content_range))
                if not match or int(match.group(1)) != offset:
                    raise RuntimeError(
                        f"invalid resume response for {url}: {content_range!r}"
                    )
                expected_total = int(match.group(3))
                if int(match.group(2)) + 1 != expected_total:
                    raise RuntimeError(f"incomplete Content-Range for {url}: {content_range}")
                mode = "ab"
            elif status == 200:
                if offset:
                    restarted = True
                offset = 0
                mode = "wb"
                if content_length is not None:
                    expected_total = int(content_length)
            else:
                raise RuntimeError(f"unexpected HTTP status {status} for {url}")
            transferred = offset
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    stream.write(chunk)
                    transferred += len(chunk)
            if expected_total is not None and transferred != expected_total:
                raise RuntimeError(
                    f"truncated download for {url}: got {transferred}, expected {expected_total}"
                )
        identity = file_identity(partial)
        if expected_bytes is not None and identity["bytes"] != expected_bytes:
            _remove_partial(partial, marker)
            raise RuntimeError(
                f"download size mismatch for {url}: {identity['bytes']} != {expected_bytes}"
            )
        if expected_sha256 is not None and identity["sha256"] != expected_sha256:
            _remove_partial(partial, marker)
            raise RuntimeError(f"download SHA-256 mismatch for {url}")
        partial.replace(destination)
        marker.unlink(missing_ok=True)
        return {
            "status": (
                "downloaded_restarted" if restarted else
                "downloaded_resumed" if offset else "downloaded"
            ),
            **identity,
        }
    raise RuntimeError(f"cannot restart download for {url}")


def fetch_locked_metadata(manifest, output: Path, timeout=90):
    receipts = {}
    for name in sorted(manifest["metadata"]):
        item = manifest["metadata"][name]
        destination = safe_output_path(output, item["relative_path"])
        result = download_http(
            item["url"],
            destination,
            expected_bytes=int(item["bytes"]),
            expected_sha256=item["sha256"],
            timeout=timeout,
        )
        receipts[name] = {
            "path": str(Path(item["relative_path"]).as_posix()),
            "url": item["url"],
            "canonical_url": item.get("canonical_url"),
            **result,
        }
    return receipts


def parse_reference_row(row, row_number):
    if row.get("ref") != "1":
        raise RuntimeError(f"CSV row {row_number} is a degraded/non-reference rendition")
    video_id = row.get("Video", "").strip().lower()
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise RuntimeError(f"CSV row {row_number} has invalid Video id")
    if row.get("resolution") != "ref" or row.get("bitrate") != "ref":
        raise RuntimeError(f"CSV row {row_number} is not marked as the pristine reference")
    if not row.get("bitladder", "").endswith("_ref_"):
        raise RuntimeError(f"CSV row {row_number} has a degraded bit-ladder label")
    orientation = row.get("orientation", "").strip()
    if orientation not in {"Landscape", "Portrait"}:
        raise RuntimeError(f"CSV row {row_number} has invalid orientation")
    content_name = row.get("content_name", "").strip()
    if not content_name:
        raise RuntimeError(f"CSV row {row_number} has no content identity")
    try:
        frame_rate = float(row["framerate"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"CSV row {row_number} has invalid geometry/frame rate") from error
    if not math.isfinite(frame_rate) or frame_rate <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(f"CSV row {row_number} has invalid geometry/frame rate")
    if (orientation == "Landscape") != (width > height):
        raise RuntimeError(f"CSV row {row_number} orientation disagrees with geometry")
    return {
        "video_id": video_id,
        "content_name": content_name,
        "orientation": orientation,
        "catalog_frame_rate": frame_rate,
        "width": width,
        "height": height,
        "csv_row": row_number,
    }


def load_chug_references(path: Path, contract=None):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CHUG CSV is missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    invalid_refs = sorted({row.get("ref") for row in raw_rows} - {"0", "1"})
    if invalid_refs:
        raise RuntimeError(f"CHUG CSV has invalid ref values: {invalid_refs}")
    references = [
        parse_reference_row(row, index + 2)
        for index, row in enumerate(raw_rows)
        if row.get("ref") == "1"
    ]
    degraded_count = sum(row.get("ref") == "0" for row in raw_rows)
    if contract:
        expected = {
            "total": int(contract["total_rows"]),
            "references": int(contract["reference_rows"]),
            "degraded": int(contract["degraded_rows"]),
        }
        actual = {
            "total": len(raw_rows),
            "references": len(references),
            "degraded": degraded_count,
        }
        if actual != expected:
            raise RuntimeError(f"CHUG CSV row identity changed: {actual} != {expected}")
    video_ids = [row["video_id"] for row in references]
    content_names = [row["content_name"] for row in references]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("CHUG CSV has duplicate reference Video ids")
    if len(set(content_names)) != len(content_names):
        raise RuntimeError("CHUG CSV has duplicate reference content identities")
    return references, {
        "total_rows": len(raw_rows),
        "reference_rows": len(references),
        "degraded_rows_excluded": degraded_count,
    }


def frame_rate_bucket(frame_rate, buckets):
    for bucket in buckets:
        maximum = bucket.get("max_exclusive")
        if maximum is None or frame_rate < float(maximum):
            return bucket["name"]
    raise RuntimeError(f"no frame-rate bucket covers {frame_rate}")


def _allocate_quotas(groups, limit, salt):
    keys = sorted(groups)
    quotas = {key: 0 for key in keys}
    if limit >= len(keys):
        quotas = {key: 1 for key in keys}
    else:
        ranked = sorted(
            keys,
            key=lambda key: (-len(groups[key]), stable_hash(salt, "stratum", *key)),
        )
        for key in ranked[:limit]:
            quotas[key] = 1
        return quotas
    remaining = limit - sum(quotas.values())
    capacities = {key: len(groups[key]) - quotas[key] for key in keys}
    total_capacity = sum(capacities.values())
    if remaining <= 0:
        return quotas
    if remaining > total_capacity:
        raise RuntimeError("selection limit exceeds available reference rows")
    ideals = {
        key: remaining * capacities[key] / total_capacity
        for key in keys
    }
    floors = {key: min(capacities[key], int(math.floor(ideals[key]))) for key in keys}
    for key in keys:
        quotas[key] += floors[key]
    leftover = remaining - sum(floors.values())
    ranked = sorted(
        keys,
        key=lambda key: (
            -(ideals[key] - floors[key]),
            stable_hash(salt, "remainder", *key),
        ),
    )
    for key in ranked:
        if leftover <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            leftover -= 1
    if leftover:
        raise RuntimeError("cannot allocate deterministic CHUG selection quota")
    return quotas


def select_hash_stratified(rows, limit, selection):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise RuntimeError("--limit must be a non-negative integer")
    if limit == 0:
        limit = len(rows)
    if limit > len(rows):
        raise RuntimeError(f"--limit {limit} exceeds {len(rows)} CHUG references")
    salt = selection["hash_salt"]
    groups = {}
    enriched = []
    for source in rows:
        row = dict(source)
        if "probed_frame_rate" not in row:
            raise RuntimeError(f"{row['video_id']}: selection requires probed frame rate")
        rate_bucket = frame_rate_bucket(
            row["probed_frame_rate"], selection["framerate_buckets"]
        )
        row["frame_rate_bucket"] = rate_bucket
        row["selection_stratum"] = f"{row['orientation']}:{rate_bucket}"
        row["selection_hash"] = stable_hash(
            salt,
            row["orientation"],
            rate_bucket,
            row["content_name"],
            row["video_id"],
        )
        key = (row["orientation"], rate_bucket)
        groups.setdefault(key, []).append(row)
        enriched.append(row)
    quotas = _allocate_quotas(groups, limit, salt)
    selected = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: (row["selection_hash"], row["video_id"]))
        selected.extend(ordered[:quotas[key]])
    if len(selected) != limit:
        raise RuntimeError(f"selected {len(selected)} CHUG rows, expected {limit}")
    return sorted(selected, key=lambda row: (row["selection_hash"], row["video_id"]))


def exact_split_counts(total, split):
    weights = split["weights"]
    weight_total = sum(int(value) for value in weights.values())
    ideals = {
        name: total * int(weight) / weight_total
        for name, weight in weights.items()
    }
    counts = {name: int(math.floor(value)) for name, value in ideals.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        weights,
        key=lambda name: (
            -(ideals[name] - counts[name]),
            stable_hash(split["hash_salt"], "split-remainder", name),
        ),
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def assign_exact_content_splits(rows, split):
    prepared = []
    content_ids = set()
    for source in rows:
        item = dict(source)
        item["content_id"] = stable_hash("chug-content-id-v1", item["content_name"])
        item["split_hash"] = stable_hash(
            split["hash_salt"], item["content_name"]
        )
        if item["content_id"] in content_ids:
            raise RuntimeError(f"duplicate selected content identity: {item['content_name']}")
        content_ids.add(item["content_id"])
        prepared.append(item)
    counts = exact_split_counts(len(prepared), split)
    ordered = sorted(prepared, key=lambda item: (item["split_hash"], item["content_id"]))
    offset = 0
    for name in ("training", "development", "test"):
        end = offset + counts[name]
        for rank, item in enumerate(ordered[offset:end], offset):
            item["split"] = name
            item["split_rank"] = rank
        offset = end
    if offset != len(ordered):
        raise RuntimeError("exact CHUG content split did not consume every clip")
    return sorted(ordered, key=lambda item: (item["selection_hash"], item["video_id"]))


def assign_exact_capture_group_splits(rows, split):
    """Assign complete capture groups while preserving exact clip-count split quotas."""
    prepared = []
    groups = {}
    content_ids = set()
    for source in rows:
        item = dict(source)
        group_id = item.get("capture_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("capture-group split requires every row to have a group identity")
        item["content_id"] = stable_hash("chug-content-id-v1", item["content_name"])
        if item["content_id"] in content_ids:
            raise RuntimeError(f"duplicate selected content identity: {item['content_name']}")
        content_ids.add(item["content_id"])
        prepared.append(item)
        groups.setdefault(group_id, []).append(item)

    counts = exact_split_counts(len(prepared), split)
    ordered_groups = sorted(
        groups.items(),
        key=lambda pair: (
            stable_hash(split["hash_salt"], "capture-group-split-order-v1", pair[0]),
            pair[0],
        ),
    )
    target = (counts["development"], counts["test"])
    states = {(0, 0): ()}
    for group_id, members in ordered_groups:
        size = len(members)
        choices = sorted(
            ("training", "development", "test"),
            key=lambda name: stable_hash(
                split["hash_salt"], "capture-group-split-choice-v1", group_id, name
            ),
        )
        next_states = {}
        for (development, test), assignment in states.items():
            for name in choices:
                next_development = development + (size if name == "development" else 0)
                next_test = test + (size if name == "test" else 0)
                if next_development > target[0] or next_test > target[1]:
                    continue
                next_states.setdefault(
                    (next_development, next_test),
                    assignment + (name,),
                )
        states = next_states
    if target not in states:
        raise RuntimeError("capture groups cannot satisfy the exact CHUG split quotas")

    group_split = {
        group_id: name
        for (group_id, _members), name in zip(ordered_groups, states[target])
    }
    split_ranks = {name: 0 for name in counts}
    for item in sorted(
            prepared,
            key=lambda row: (
                stable_hash(split["hash_salt"], row["capture_group_id"]),
                row["content_id"],
            )):
        name = group_split[item["capture_group_id"]]
        item["split_hash"] = stable_hash(
            split["hash_salt"], "capture-group-split-v1", item["capture_group_id"]
        )
        item["split"] = name
        item["split_rank"] = split_ranks[name]
        split_ranks[name] += 1
    if count_by(prepared, "split") != dict(sorted(counts.items())):
        raise RuntimeError("capture-group split did not produce the exact CHUG quotas")
    if any(len({item["split"] for item in members}) != 1 for members in groups.values()):
        raise RuntimeError("capture-group split leaked a group across dataset splits")
    return sorted(prepared, key=lambda item: (item["selection_hash"], item["video_id"]))


def prepare_selection(rows, manifest, limit):
    selected = select_hash_stratified(rows, limit, manifest["selection"])
    return assign_exact_content_splits(selected, manifest["split"])


def prepare_existing_capture_group_selection(
        candidates,
        group_manifest,
        manifest,
        video_ids,
        limit):
    if len(video_ids) != limit or len(video_ids) != len(set(video_ids)):
        raise RuntimeError("existing CHUG selection does not match the requested unique clip count")
    by_id = {row["video_id"]: row for row in candidates}
    if not set(video_ids) <= set(by_id):
        raise RuntimeError("existing CHUG selection contains a stale candidate")
    group_by_member = {}
    for group in group_manifest["capture_groups"]:
        for video_id in group["members"]:
            group_by_member[video_id] = group
    selected = select_hash_stratified(
        [by_id[video_id] for video_id in video_ids],
        limit,
        manifest["selection"],
    )
    for item in selected:
        group = group_by_member[item["video_id"]]
        item.update({
            "capture_group_id": group["capture_group_id"],
            "capture_group_member_count": len(group["members"]),
            "capture_group_members": group["members"],
        })
    return assign_exact_capture_group_splits(selected, manifest["split"])


def load_capture_group_manifest(path: Path, source_manifest, candidate_semantic_sha256, candidates):
    if not path.is_file():
        raise RuntimeError(
            "capture-group manifest is required before CHUG selection; "
            "run build_chug_capture_groups.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise RuntimeError("unsupported CHUG capture-group manifest schema")
    if payload.get("dataset") != source_manifest["id"]:
        raise RuntimeError("capture-group dataset identity is stale")
    if payload.get("repository_commit") != source_manifest["repository_commit"]:
        raise RuntimeError("capture-group source commit is stale")
    if payload.get("candidate_probe_semantic_sha256") != candidate_semantic_sha256:
        raise RuntimeError("capture-group candidate-probe identity is stale")
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in {"semantic_sha256", "execution"}
    }
    if canonical_sha256(semantic) != payload.get("semantic_sha256"):
        raise RuntimeError("capture-group semantic identity is invalid")
    if payload.get("status") != "decision_ready":
        raise RuntimeError("capture-group manifest is not decision-ready")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("capture-group evidence is invalid")
    review_required = evidence.get("review_required_pairs")
    if not isinstance(review_required, list) or review_required:
        raise RuntimeError("capture-group manifest has unresolved perceptual review pairs")
    candidate_ids = {row["video_id"] for row in candidates}
    visual_ids = {item["video_id"] for item in payload.get("visual_identities", [])}
    if visual_ids != candidate_ids or payload.get("valid_reference_count") != len(candidate_ids):
        raise RuntimeError("capture-group visual identity coverage is incomplete")
    grouped_ids = []
    for group in payload.get("capture_groups", []):
        members = group.get("members")
        if not isinstance(members, list) or members != sorted(set(members)):
            raise RuntimeError("capture-group members are invalid")
        if group.get("member_count") != len(members):
            raise RuntimeError("capture-group member count is invalid")
        grouped_ids.extend(members)
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != candidate_ids:
        raise RuntimeError("capture groups do not partition every valid reference exactly once")
    calibration = evidence.get("calibration_pairs", [])
    if any(item.get("label") == "positive" and not item.get("same_group") for item in calibration):
        raise RuntimeError("capture-group manifest split a confirmed positive pair")
    if any(item.get("label") == "negative" and item.get("same_group") for item in calibration):
        raise RuntimeError("capture-group manifest merged a hard-negative pair")
    return payload


def collapse_capture_group_representatives(candidates, group_manifest, selection):
    by_id = {row["video_id"]: row for row in candidates}
    representatives = []
    for group in group_manifest["capture_groups"]:
        group_id = group["capture_group_id"]
        ranked = sorted(
            group["members"],
            key=lambda video_id: (
                stable_hash(
                    selection["hash_salt"],
                    "capture-group-representative-v1",
                    group_id,
                    video_id,
                ),
                video_id,
            ),
        )
        video_id = ranked[0]
        representative = dict(by_id[video_id])
        representative.update({
            "capture_group_id": group_id,
            "capture_group_member_count": len(group["members"]),
            "capture_group_members": group["members"],
            "capture_group_representative_hash": stable_hash(
                selection["hash_salt"],
                "capture-group-representative-v1",
                group_id,
                video_id,
            ),
        })
        representatives.append(representative)
    if len({item["capture_group_id"] for item in representatives}) != len(representatives):
        raise RuntimeError("capture-group representative collapse is not one-to-one")
    return sorted(representatives, key=lambda item: item["video_id"])


def _valid_jobs(jobs):
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= MAX_DOWNLOAD_JOBS:
        raise RuntimeError(f"--jobs must be between 1 and {MAX_DOWNLOAD_JOBS}")
    return jobs


def download_video(row, output: Path, manifest, timeout, audit_only=False):
    video_dir = safe_output_path(output, manifest["videos"]["relative_directory"])
    video = video_dir / f"{row['video_id']}.mp4"
    url = manifest["videos"]["url_template"].format(video_id=row["video_id"])
    if audit_only:
        if not video.is_file():
            raise RuntimeError(f"audit-only clip is missing: {video}")
        result = {"status": "audit_only_existing", **file_identity(video)}
    else:
        result = download_http(url, video, timeout=timeout)
    return {
        "video_id": row["video_id"],
        "path": str(video.resolve()),
        "relative_path": video.relative_to(output.resolve()).as_posix(),
        "url": url,
        **result,
    }


def download_selected(rows, output: Path, manifest, jobs, timeout=90, audit_only=False):
    jobs = _valid_jobs(jobs)
    completed = []
    rejected = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                download_video,
                row,
                output,
                manifest,
                timeout,
                audit_only,
            ): row
            for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = futures[future]
            try:
                completed.append(future.result())
                action = "verified local" if audit_only else "downloaded"
                print(f"[{index}/{len(rows)}] {action} {row['video_id']}", flush=True)
            except Exception as error:
                rejected.append({
                    "video_id": row["video_id"],
                    "content_name": row["content_name"],
                    "stage": "download",
                    "reason": str(error),
                })
                print(f"[{index}/{len(rows)}] rejected {row['video_id']}: {error}", flush=True)
    return sorted(completed, key=lambda item: item["video_id"]), sorted(
        rejected, key=lambda item: item["video_id"]
    )


def resolve_ffprobe(path=None):
    candidate = Path(path) if path else None
    if candidate is None:
        located = shutil.which("ffprobe")
        candidate = Path(located) if located else None
    if candidate is None or not candidate.is_file():
        raise RuntimeError("ffprobe is required; pass --ffprobe")
    return candidate.resolve()


def run_command(command, timeout):
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if process.returncode:
        detail = (process.stdout + process.stderr).strip()[-2000:]
        raise RuntimeError(f"command failed ({process.returncode}): {detail}")
    return process.stdout


def ffprobe_version(ffprobe: Path):
    output = run_command([str(ffprobe), "-version"], 30)
    return output.splitlines()[0] if output else ""


def probe_video(ffprobe: Path, video: Path):
    output = run_command([
        str(ffprobe),
        "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,bits_per_raw_sample,color_range,"
        "color_space,color_transfer,color_primaries,width,height,r_frame_rate,"
        "avg_frame_rate,duration:format=duration,size,format_name",
        "-of", "json",
        str(video),
    ], 60)
    try:
        return json.loads(output)
    except ValueError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video}") from error


def _finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"invalid {label}: {value!r}")
    return result


def _frame_rate(value):
    if not value or value == "0/0":
        return 0.0
    if "/" in str(value):
        numerator, denominator = str(value).split("/", 1)
        denominator = _finite_float(denominator, "frame-rate denominator")
        if denominator == 0.0:
            return 0.0
        return _finite_float(numerator, "frame-rate numerator") / denominator
    return _finite_float(value, "frame rate")


def validate_native_pq_probe(payload, row, contract):
    streams = [
        stream for stream in payload.get("streams", [])
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if len(streams) != 1:
        raise RuntimeError("ffprobe must identify exactly one video stream")
    stream = streams[0]
    if str(stream.get("codec_name", "")).lower() != contract["codec"]:
        raise RuntimeError(f"codec is not HEVC: {stream.get('codec_name')!r}")
    pixel_format = str(stream.get("pix_fmt", "")).lower()
    match = YUV420_DEPTH_PATTERN.fullmatch(pixel_format)
    raw_depth = stream.get("bits_per_raw_sample")
    raw_depth = int(raw_depth) if str(raw_depth).isdigit() else 0
    pixel_depth = int(match.group(1)) if match else 0
    bit_depth = max(raw_depth, pixel_depth)
    if not match or bit_depth < int(contract["minimum_bit_depth"]):
        raise RuntimeError(f"pixel format is not yuv420p10+: {pixel_format!r}")
    expected_fields = {
        "color_range": contract["color_range"],
        "color_primaries": contract["color_primaries"],
        "color_space": contract["color_space"],
        "color_transfer": contract["color_transfer"],
    }
    for field, expected in expected_fields.items():
        actual = str(stream.get(field, "")).lower()
        if actual != expected:
            raise RuntimeError(f"{field} is {actual!r}, expected {expected!r}")
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if (width, height) != (row["width"], row["height"]):
        raise RuntimeError(
            f"decoded geometry {(width, height)} != CSV {(row['width'], row['height'])}"
        )
    if (row["orientation"] == "Landscape") != (width > height):
        raise RuntimeError("decoded geometry disagrees with CSV orientation")
    video_duration = stream.get("duration")
    if video_duration in {None, "", "N/A"}:
        video_duration = payload.get("format", {}).get("duration")
    duration = _finite_float(video_duration, "duration")
    minimum = float(contract["minimum_duration_seconds"])
    maximum = float(contract["maximum_duration_seconds"])
    if not minimum <= duration <= maximum:
        raise RuntimeError(f"duration {duration:.3f}s is outside [{minimum}, {maximum}]")
    rate = _frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    if rate <= 0.0:
        raise RuntimeError("ffprobe reports no valid frame rate")
    reported_size = payload.get("format", {}).get("size")
    reported_size = int(reported_size) if str(reported_size).isdigit() else None
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "bit_depth": bit_depth,
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "width": width,
        "height": height,
        "frame_rate": rate,
        "duration_seconds": duration,
        "reported_bytes": reported_size,
        "container": payload.get("format", {}).get("format_name"),
        "native_color_contract": "pq-bt2020-yuv420p10+-preserved",
        "frame_decode": "not_performed",
    }


def candidate_probe_binding(row, manifest, ffprobe_version_text):
    url = manifest["videos"]["url_template"].format(video_id=row["video_id"])
    return {
        "video_id": row["video_id"],
        "source_url": url,
        "repository_commit": manifest["repository_commit"],
        "probe_query_schema": CANDIDATE_PROBE_QUERY_SCHEMA,
        "producer_ffprobe_version": ffprobe_version_text,
    }


def candidate_probe_cache_payload(binding, probe):
    return {
        "schema": CANDIDATE_PROBE_CACHE_SCHEMA,
        "binding": binding,
        "probe": probe,
    }


def _load_candidate_probe_cache(path: Path, binding):
    if not path.is_file():
        return None, "missing"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "corrupt"
    if not isinstance(cached, dict):
        return None, "corrupt"
    if cached.get("schema") != CANDIDATE_PROBE_CACHE_SCHEMA:
        return None, "unbound_or_incompatible"
    if cached.get("binding") != binding:
        return None, "binding_mismatch"
    payload = cached.get("probe")
    if not isinstance(payload, dict):
        return None, "corrupt"
    if not isinstance(payload.get("streams"), list):
        return None, "corrupt"
    if not isinstance(payload.get("format"), dict):
        return None, "corrupt"
    return payload, "cache_hit"


def probe_reference_candidate(
        row,
        output: Path,
        manifest,
        ffprobe: Path,
        ffprobe_version_text,
        attempts=3,
        refresh_probe=False):
    binding = candidate_probe_binding(row, manifest, ffprobe_version_text)
    url = binding["source_url"]
    probe_path = output / "candidate_ffprobe" / f"{row['video_id']}.json"
    if refresh_probe:
        payload, cache_status = None, "refresh_requested"
    else:
        payload, cache_status = _load_candidate_probe_cache(probe_path, binding)
    probe_source = "cache" if payload is not None else "remote"
    if payload is None:
        errors = []
        for _ in range(attempts):
            try:
                payload = probe_video(ffprobe, url)
                break
            except Exception as error:
                errors.append(str(error))
        if payload is None:
            raise RuntimeError("; ".join(errors))
        atomic_write_json(
            probe_path,
            candidate_probe_cache_payload(binding, payload),
        )
    try:
        audit = validate_native_pq_probe(
            payload,
            row,
            manifest["native_color_contract"],
        )
    except Exception as error:
        return None, {
            "video_id": row["video_id"],
            "content_name": row["content_name"],
            "stage": "candidate_contract",
            "reason": str(error),
            "url": url,
            "probe": probe_path.relative_to(output.resolve()).as_posix(),
            "probe_source": probe_source,
            "probe_cache_status": cache_status,
        }
    candidate = {
        **row,
        "probed_frame_rate": audit["frame_rate"],
        "probed_duration_seconds": audit["duration_seconds"],
        "probed_orientation": "Landscape" if audit["width"] > audit["height"] else "Portrait",
        "candidate_url": url,
        "candidate_probe": probe_path.relative_to(output.resolve()).as_posix(),
        "candidate_audit": audit,
        "probe_source": probe_source,
        "probe_cache_status": cache_status,
    }
    return candidate, None


def probe_reference_candidates(
        rows,
        output: Path,
        manifest,
        ffprobe: Path,
        ffprobe_version_text,
        jobs,
        refresh_probes=False):
    jobs = _valid_jobs(jobs)
    valid = []
    rejected = []
    probe_errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                probe_reference_candidate,
                row,
                output,
                manifest,
                ffprobe,
                ffprobe_version_text,
                refresh_probe=refresh_probes,
            ): row
            for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = futures[future]
            try:
                candidate, rejection = future.result()
                if candidate is not None:
                    valid.append(candidate)
                else:
                    rejected.append(rejection)
            except Exception as error:
                probe_errors.append({
                    "video_id": row["video_id"],
                    "content_name": row["content_name"],
                    "stage": "candidate_probe_error",
                    "reason": str(error),
                    "url": manifest["videos"]["url_template"].format(
                        video_id=row["video_id"]
                    ),
                    "probe_source": "remote",
                    "attempts": 3,
                })
            if index % 32 == 0 or index == len(rows):
                print(
                    f"candidate probes {index}/{len(rows)}: "
                    f"{len(valid)} valid, {len(rejected)} contract rejects, "
                    f"{len(probe_errors)} errors",
                    flush=True,
                )
    return (
        sorted(valid, key=lambda item: item["video_id"]),
        sorted(rejected, key=lambda item: item["video_id"]),
        sorted(probe_errors, key=lambda item: item["video_id"]),
    )


def validate_candidate_consistency(row, download, audit):
    expected = row["candidate_audit"]
    exact_fields = (
        "codec",
        "pixel_format",
        "bit_depth",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "width",
        "height",
    )
    for field in exact_fields:
        if audit.get(field) != expected.get(field):
            raise RuntimeError(
                f"downloaded {field} changed after candidate probe: "
                f"{expected.get(field)!r} -> {audit.get(field)!r}"
            )
    if abs(audit["duration_seconds"] - expected["duration_seconds"]) > 0.001:
        raise RuntimeError("downloaded duration changed after candidate probe")
    if abs(audit["frame_rate"] - expected["frame_rate"]) > 0.001:
        raise RuntimeError("downloaded frame rate changed after candidate probe")
    reported_bytes = expected.get("reported_bytes")
    if reported_bytes is not None and download["bytes"] != reported_bytes:
        raise RuntimeError(
            f"downloaded bytes {download['bytes']} != remote probe {reported_bytes}"
        )


def audit_downloads(downloads, rows, output: Path, manifest, ffprobe: Path):
    by_id = {row["video_id"]: row for row in rows}
    probe_dir = output / "ffprobe"
    accepted = []
    rejected = []
    for index, item in enumerate(downloads, 1):
        row = by_id[item["video_id"]]
        video = Path(item["path"])
        try:
            payload = probe_video(ffprobe, video)
            probe_path = probe_dir / f"{row['video_id']}.json"
            atomic_write_json(probe_path, payload)
            audit = validate_native_pq_probe(
                payload,
                row,
                manifest["native_color_contract"],
            )
            validate_candidate_consistency(row, item, audit)
            accepted.append({
                **row,
                "download": item,
                "probe": probe_path.relative_to(output.resolve()).as_posix(),
                "audit": audit,
            })
            print(f"[{index}/{len(downloads)}] audited {row['video_id']}", flush=True)
        except Exception as error:
            rejected.append({
                "video_id": row["video_id"],
                "content_name": row["content_name"],
                "stage": "ffprobe_audit",
                "reason": str(error),
                "download": item,
            })
            print(f"[{index}/{len(downloads)}] audit rejected {row['video_id']}: {error}", flush=True)
    return sorted(accepted, key=lambda item: item["video_id"]), sorted(
        rejected, key=lambda item: item["video_id"]
    )


def count_by(items, key):
    counts = {}
    for item in items:
        value = item[key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def without_probe_execution(item):
    return {
        key: value
        for key, value in item.items()
        if key not in {"probe_source", "probe_cache_status"}
    }


def stable_metadata_receipts(metadata):
    return {
        name: {
            key: value
            for key, value in item.items()
            if key != "status"
        }
        for name, item in metadata.items()
    }


def build_candidate_semantic_payload(
        source_manifest,
        ffprobe_version_text,
        reference_count,
        candidates,
        rejections,
        probe_error_count=0):
    stable_candidates = [without_probe_execution(item) for item in candidates]
    stable_rejections = [without_probe_execution(item) for item in rejections]
    stable_summary = {
        "probed": reference_count,
        "valid": len(stable_candidates),
        "contract_rejected": len(stable_rejections),
        "probe_errors": probe_error_count,
    }
    return {
        "schema": 1,
        "dataset": source_manifest["id"],
        "repository_commit": source_manifest["repository_commit"],
        "probe_query_schema": CANDIDATE_PROBE_QUERY_SCHEMA,
        "producer_ffprobe_version": ffprobe_version_text,
        "catalog_reference_rows": reference_count,
        "valid_candidates": stable_candidates,
        "contract_rejections": stable_rejections,
        "summary": stable_summary,
    }


def build_selection_manifest(
        source_manifest,
        source_manifest_hash,
        metadata,
        stats,
        candidate_manifest,
        capture_group_manifest,
        selected,
        limit,
        selection_method="capture-group-representative-sha256-stratified-v2",
        representative_only=True):
    payload = {
        "schema": 1,
        "dataset": source_manifest["id"],
        "repository_commit": source_manifest["repository_commit"],
        "source_manifest_sha256": source_manifest_hash,
        "metadata": metadata,
        "catalog": stats,
        "candidate_probe_manifest": candidate_manifest,
        "capture_group_manifest": capture_group_manifest,
        "selection": {
            "requested_limit": limit,
            "selected_references": len(selected),
            "method": selection_method,
            "capture_group_representative_only": representative_only,
            "hash_salt": source_manifest["selection"]["hash_salt"],
            "strata": count_by(selected, "selection_stratum"),
        },
        "split": {
            "unit": source_manifest["catalog_contract"]["content_id_column"],
            "method": "exact-6-1-1-sha256-content-group-before-frame-extraction-v1",
            "hash_salt": source_manifest["split"]["hash_salt"],
            "weights": source_manifest["split"]["weights"],
            "counts": count_by(selected, "split"),
            "capture_group_exclusive": True,
        },
        "clips": selected,
        "native_color_contract": source_manifest["native_color_contract"],
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def load_existing_accepted_ids(path: Path, dataset, expected_count):
    if not path.is_file():
        raise RuntimeError("existing CHUG download receipt is required for local-only migration")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get("dataset") != dataset:
        raise RuntimeError("existing CHUG download receipt is stale")
    if payload.get("rejected"):
        raise RuntimeError("existing CHUG download receipt contains rejected clips")
    accepted = payload.get("accepted")
    if not isinstance(accepted, list) or len(accepted) != expected_count:
        raise RuntimeError("existing CHUG receipt does not contain the requested clip count")
    video_ids = [item.get("video_id") for item in accepted]
    if any(not isinstance(video_id, str) for video_id in video_ids):
        raise RuntimeError("existing CHUG receipt has an invalid video identity")
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("existing CHUG receipt repeats a video identity")
    return video_ids


def copy_source_manifest(path: Path, output: Path):
    destination = output / "source_metadata" / "apollo_chug_source_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".writing")
    temporary.write_bytes(path.read_bytes())
    temporary.replace(destination)
    return {
        "path": destination.relative_to(output.resolve()).as_posix(),
        **file_identity(destination),
    }


def run_fetch(args):
    source_path = args.source_manifest.resolve()
    source_manifest = load_source_manifest(source_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_copy = copy_source_manifest(source_path, output)
    metadata = fetch_locked_metadata(source_manifest, output, timeout=args.timeout)
    csv_path = safe_output_path(output, source_manifest["metadata"]["csv"]["relative_path"])
    references, stats = load_chug_references(csv_path, source_manifest["catalog_contract"])
    limit = source_manifest["selection"]["default_limit"] if args.limit is None else args.limit
    if limit < 0:
        raise RuntimeError("--limit must be a non-negative integer")
    ffprobe = resolve_ffprobe(args.ffprobe)
    ffprobe_version_text = ffprobe_version(ffprobe)
    candidates, candidate_rejected, probe_errors = probe_reference_candidates(
        references,
        output,
        source_manifest,
        ffprobe,
        ffprobe_version_text,
        args.jobs,
        refresh_probes=args.refresh_probes,
    )
    cache_hits = sum(item["probe_source"] == "cache" for item in candidates)
    cache_hits += sum(item["probe_source"] == "cache" for item in candidate_rejected)
    remote_probes = len(candidates) + len(candidate_rejected) + len(probe_errors) - cache_hits
    candidate_semantic_payload = build_candidate_semantic_payload(
        source_manifest,
        ffprobe_version_text,
        len(references),
        candidates,
        candidate_rejected,
        probe_error_count=len(probe_errors),
    )
    stable_candidates = candidate_semantic_payload["valid_candidates"]
    stable_summary = candidate_semantic_payload["summary"]
    candidate_semantic_sha256 = canonical_sha256(candidate_semantic_payload)
    probe_execution = sorted(
        [
            {
                "video_id": item["video_id"],
                "probe_source": item["probe_source"],
                "probe_cache_status": item["probe_cache_status"],
            }
            for item in candidates + candidate_rejected
        ],
        key=lambda item: item["video_id"],
    )
    candidate_payload = {
        **candidate_semantic_payload,
        "semantic_sha256": candidate_semantic_sha256,
        "ffprobe": {
            "path": str(ffprobe),
            "version": ffprobe_version_text,
        },
        "probe_errors": probe_errors,
        "probe_execution": probe_execution,
        "summary": {
            **stable_summary,
            "cache_hits": cache_hits,
            "remote_probes": remote_probes,
            "refresh_probes": args.refresh_probes,
        },
    }
    candidate_path = output / "candidate_probe_manifest.json"
    atomic_write_json(candidate_path, candidate_payload)
    candidate_selection_identity = {
        "path": candidate_path.relative_to(output).as_posix(),
        "semantic_sha256": candidate_semantic_sha256,
        "summary": stable_summary,
    }
    candidate_receipt_identity = {
        "path": candidate_path.relative_to(output).as_posix(),
        **file_identity(candidate_path),
        "semantic_sha256": candidate_semantic_sha256,
        "summary": candidate_payload["summary"],
    }
    if probe_errors:
        raise RuntimeError(
            f"{len(probe_errors)} CHUG candidate probe(s) failed; selection would not be reproducible; "
            f"see {candidate_path}"
        )
    if args.probe_only:
        print(json.dumps(candidate_payload["summary"], indent=2, sort_keys=True))
        return candidate_payload
    capture_group_path = (
        args.capture_groups.resolve()
        if args.capture_groups
        else output / "capture_group_manifest.json"
    )
    try:
        capture_group_relative = capture_group_path.relative_to(output).as_posix()
    except ValueError as error:
        raise RuntimeError("capture-group manifest must be inside the CHUG dataset root") from error
    capture_groups = load_capture_group_manifest(
        capture_group_path,
        source_manifest,
        candidate_semantic_sha256,
        stable_candidates,
    )
    capture_group_selection_identity = {
        "path": capture_group_relative,
        "semantic_sha256": capture_groups["semantic_sha256"],
        "valid_reference_count": capture_groups["valid_reference_count"],
        "capture_group_count": len(capture_groups["capture_groups"]),
    }
    capture_group_receipt_identity = {
        **capture_group_selection_identity,
        **file_identity(capture_group_path),
    }
    if args.reuse_existing_selection:
        if not args.audit_only:
            raise RuntimeError("--reuse-existing-selection requires --audit-only to prohibit downloads")
        existing_ids = load_existing_accepted_ids(
            output / "download_receipt.json",
            source_manifest["id"],
            limit,
        )
        selected = prepare_existing_capture_group_selection(
            stable_candidates,
            capture_groups,
            source_manifest,
            existing_ids,
            limit,
        )
        selection_method = "capture-group-preserving-existing-local-migration-v1"
        representative_only = False
    else:
        representatives = collapse_capture_group_representatives(
            stable_candidates,
            capture_groups,
            source_manifest["selection"],
        )
        selected = prepare_selection(representatives, source_manifest, limit)
        if len({item["capture_group_id"] for item in selected}) != len(selected):
            raise RuntimeError("selected CHUG references are not capture-group exclusive")
        selection_method = "capture-group-representative-sha256-stratified-v2"
        representative_only = True
    group_splits = {}
    for item in selected:
        group_splits.setdefault(item["capture_group_id"], set()).add(item["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("selected CHUG references leak a capture group across splits")
    selection = build_selection_manifest(
        source_manifest,
        manifest_copy["sha256"],
        stable_metadata_receipts(metadata),
        stats,
        candidate_selection_identity,
        capture_group_selection_identity,
        selected,
        limit,
        selection_method=selection_method,
        representative_only=representative_only,
    )
    selection_path = output / "selection_manifest.json"
    atomic_write_json(selection_path, selection)
    downloads, rejected = download_selected(
        selected,
        output,
        source_manifest,
        args.jobs,
        timeout=args.timeout,
        audit_only=args.audit_only,
    )
    accepted, audit_rejected = audit_downloads(
        downloads,
        selected,
        output,
        source_manifest,
        ffprobe,
    )
    rejected.extend(audit_rejected)
    receipt = {
        "schema": 1,
        "dataset": source_manifest["id"],
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": source_manifest["license"],
        "license_url": source_manifest["license_url"],
        "usage": "non-commercial research only; ShareAlike applies to shared adaptations",
        "source_manifest": manifest_copy,
        "metadata": metadata,
        "candidate_probe_manifest": candidate_receipt_identity,
        "capture_group_manifest": capture_group_receipt_identity,
        "selection_manifest": {
            "path": selection_path.relative_to(output).as_posix(),
            **file_identity(selection_path),
        },
        "ffprobe": {
            "path": str(ffprobe),
            "version": ffprobe_version_text,
        },
        "native_color_contract": source_manifest["native_color_contract"],
        "accepted": accepted,
        "rejected": sorted(rejected, key=lambda item: (item["video_id"], item["stage"])),
        "summary": {
            "selected": len(selected),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accepted_bytes": sum(item["download"]["bytes"] for item in accepted),
            "accepted_splits": count_by(accepted, "split"),
            "accepted_strata": count_by(accepted, "selection_stratum"),
        },
    }
    receipt_path = output / "download_receipt.json"
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    if rejected:
        raise RuntimeError(
            f"{len(rejected)} selected CHUG clip(s) were rejected; see {receipt_path}"
        )
    (output / "QUARANTINED.json").unlink(missing_ok=True)
    return receipt


def main():
    default_manifest = Path(__file__).with_name("chug_hdr_reference_sources.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="deterministic eligible-reference count; default 96, 0 selects every contract-valid reference",
    )
    parser.add_argument("--jobs", type=int, default=4, help=f"parallel downloads (1-{MAX_DOWNLOAD_JOBS})")
    parser.add_argument("--timeout", type=int, default=90, help="per-request timeout in seconds")
    parser.add_argument("--ffprobe", type=Path, help="ffprobe executable; defaults to PATH")
    parser.add_argument(
        "--capture-groups",
        type=Path,
        help="authenticated capture_group_manifest.json; defaults inside --output",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="publish authenticated candidate probes, but do not select or download before grouping",
    )
    parser.add_argument(
        "--refresh-probes",
        action="store_true",
        help="ignore reusable candidate_ffprobe JSON and remotely re-probe every pristine reference",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="do not download video; re-hash and re-probe the selected existing clips",
    )
    parser.add_argument(
        "--reuse-existing-selection",
        action="store_true",
        help="migrate the existing receipt as whole capture groups; requires --audit-only",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        raise RuntimeError("--timeout must be positive")
    if args.reuse_existing_selection and not args.audit_only:
        raise RuntimeError("--reuse-existing-selection requires --audit-only")
    if args.reuse_existing_selection and args.probe_only:
        raise RuntimeError("--reuse-existing-selection cannot be combined with --probe-only")
    run_fetch(args)


if __name__ == "__main__":
    main()

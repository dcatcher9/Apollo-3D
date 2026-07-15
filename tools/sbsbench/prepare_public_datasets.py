#!/usr/bin/env python3
"""Download and deterministically prepare the external public SBS evaluation suite.

Media is intentionally kept outside Git.  The committed manifest fixes source URLs and frame
windows; this tool turns those archives into the exact ``frame_*``/reference layout consumed by
run_eval.py. Full downloads are resumable and verified when their manifest has a SHA-256 digest.
Range-backed ZIP selections validate every HTTP byte range and ZIP-member CRC while retaining the
upstream whole-archive checksum as provenance.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
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


HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "datasets", "manifest.json")


def fail(message):
    raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema") not in (1, 2):
        fail(f"unsupported dataset manifest schema: {data.get('schema')}")
    return data


def archive_spec(manifest, clip, archive_name):
    dataset = manifest["datasets"][clip["dataset"]]
    try:
        return dataset["archives"][archive_name]
    except KeyError:
        fail(f"unknown archive {clip['dataset']}.{archive_name}")


def download_archive(spec, downloads_dir):
    os.makedirs(downloads_dir, exist_ok=True)
    path = os.path.join(downloads_dir, spec["filename"])
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


def selected(items, clip):
    start, stride, count = (int(clip[k]) for k in ("start", "stride", "count"))
    indexes = [start + i * stride for i in range(count)]
    if not indexes or indexes[-1] >= len(items):
        fail(f"selection {start}:{stride}:{count} exceeds {len(items)} available samples")
    return [(i, items[i]) for i in indexes]


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
            selection.append({"source_index": source_i, "rgb_timestamp": rgb_ts,
                              "depth_timestamp": depth_ts})
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
            selection.append({"source_index": source_i, "dataset_frame": frame_id})
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
    sequence, render_pass = clip["sequence"], clip["pass"]
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        left_marker = f"/{render_pass}_left/{sequence}/"
        right_marker = f"/{render_pass}_right/{sequence}/"
        left = _indexed_archive_members(names, left_marker, r"frame_(\d+)\.png$")
        right = _indexed_archive_members(names, right_marker, r"frame_(\d+)\.png$")
        disparities = _indexed_archive_members(
            names, f"/disparities/{sequence}/", r"frame_(\d+)\.png$")
        if not left:
            fail(f"no Sintel {render_pass}_left frames found for {sequence}")
        chosen = selected(sorted(left), clip)
        os.makedirs(os.path.join(out_dir, "gt_right"))
        os.makedirs(os.path.join(out_dir, "gt_depth"))
        selection = []
        for output_id, (source_i, frame_id) in enumerate(chosen):
            if frame_id not in right:
                fail(f"missing Sintel right-eye frame {sequence}/{frame_id}")
            if frame_id not in disparities:
                fail(f"missing Sintel disparity frame {sequence}/{frame_id}")
            _write_image_bytes(zf.read(left[frame_id]),
                               os.path.join(out_dir, f"frame_{output_id:05d}.png"), rgb=True)
            _write_image_bytes(zf.read(right[frame_id]),
                               os.path.join(out_dir, "gt_right", f"frame_{output_id:05d}.png"),
                               rgb=True)
            with Image.open(io.BytesIO(zf.read(disparities[frame_id]))) as disparity_png:
                encoded = np.asarray(disparity_png.convert("RGB"), dtype=np.float32)
            disparity = encoded[..., 0] * 4.0 + encoded[..., 1] / 64.0 + encoded[..., 2] / 16384.0
            np.save(os.path.join(out_dir, "gt_depth", f"frame_{output_id:05d}.npy"),
                    disparity.astype(np.float32))
            selection.append({"source_index": source_i, "dataset_frame": frame_id,
                              "sequence": sequence, "pass": render_pass})
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
            selection.append({"source_index": source_i, "dataset_frame": frame_id,
                              "sequence": sequence, "split": "test"})
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
            selection.append({"source_index": source_i, "dataset_frame": frame_id,
                              "scene": scene, "variant": variant, "camera": camera})
        return selection
    finally:
        rgb_tar.close()
        depth_tar.close()


def prepare_clip(manifest, clip_id, clip, downloads_dir, prepared_root):
    dataset = manifest["datasets"][clip["dataset"]]
    archives = {}
    for name in clip["archives"]:
        spec = archive_spec(manifest, clip, name)
        if spec.get("access") == "http_range_zip":
            archives[name] = spec
            continue
        path = os.path.join(downloads_dir, spec["filename"])
        if not os.path.exists(path):
            fail(f"archive missing; run without --no-download first: {path}")
        archives[name] = path
    final = os.path.join(prepared_root, clip_id)
    os.makedirs(prepared_root, exist_ok=True)
    temp = tempfile.mkdtemp(prefix=clip_id + ".", dir=prepared_root)
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
        has_gt_depth = clip["adapter"] != "spring_http_range_zip"
        meta = {
            "name": clip["name"], "description": clip["description"],
            "dataset": dataset["title"], "homepage": dataset["homepage"],
            "citation": dataset["citation"], "license_note": dataset["license_note"],
            "suite": manifest["prepared_suite"],
            "required_gt_depth": has_gt_depth,
            "required_gt_flow": clip["adapter"] == "tartanair_v2_zip",
            "selection": selection,
        }
        if has_gt_depth:
            meta["gt_depth_kind"] = ("disparity" if clip["adapter"] == "sintel_stereo_zip"
                                     else "metric")
        if clip["adapter"] in ("sintel_stereo_zip", "spring_http_range_zip"):
            meta["required_gt_stereo"] = True
        for key in ("source_artifacts", "source_artifact_frame"):
            if key in clip:
                meta[key] = clip[key]
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
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    cache = os.path.abspath(args.cache or os.environ.get("APOLLO_SBS_DATASETS")
                            or manifest["default_cache"])
    downloads = os.path.join(cache, "downloads")
    prepared = os.path.join(cache, "prepared", manifest["prepared_suite"])
    clip_ids = args.clips or list(manifest["clips"])
    unknown = sorted(set(clip_ids) - set(manifest["clips"]))
    if unknown:
        fail(f"unknown clip IDs: {unknown}")
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
                path = os.path.join(downloads, spec["filename"])
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

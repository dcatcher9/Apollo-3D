"""Inspect lossless native Game 3D artifacts; optional PNGs are display previews only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from game3d_ui_discovery import artifact_facts, discover_ui, format_report, review_template

MAX_ARTIFACTS = 40  # game3d_debug_protocol.h v3; disk artifact schema stays v1.
PRIMARY_ARTIFACTS = {"source_color", "raw_depth", "candidate", "vertical_majorant",
                     "vertical_field", "final_field", "sbs", "linear_color", "ui_source_color"}
ALPHA_FORMATS = {2, 10, 24, 28, 29, 87, 91}  # Decoded formats with a real A component.

# DXGI typed formats admitted by the diagnostic receiver.
FORMATS = {
    2: ("<f4", 4), 10: ("<f2", 4), 16: ("<f4", 2),
    28: ("u1", 4), 29: ("u1", 4), 34: ("<f2", 2), 41: ("<f4", 1),
    54: ("<f2", 1), 56: ("<u2", 1), 61: ("u1", 1), 62: ("u1", 1),
    87: ("u1", 4), 91: ("u1", 4), 24: ("<u4", 1),
}


def read_artifact(root, descriptor, *, include_digest=False):
    root = Path(root).resolve()
    path = (root / descriptor["file"]).resolve()
    if path.parent != root:
        raise ValueError("Artifact must be a direct child of the dump directory")
    fmt = descriptor["dxgi_format"]
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported DXGI format {fmt}")
    dtype, channels = FORMATS[fmt]
    width, height = descriptor["width"], descriptor["height"]
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError("Invalid texture dimensions")
    row = width * channels * np.dtype(dtype).itemsize
    size = row * height
    if size > 768 * 1024 * 1024 or descriptor["row_bytes"] != row:
        raise ValueError("Invalid packed row size")
    if descriptor["byte_count"] != size or path.stat().st_size != size:
        raise ValueError("Artifact byte count differs from its descriptor")
    value = np.fromfile(path, dtype=dtype).reshape(height, width, channels)
    # Bind review to the native bytes, before channel swizzles or normalization.
    digest = hashlib.sha256(value).hexdigest() if include_digest else None
    if fmt == 24:
        packed = value[:, :, 0]
        channels = [(packed >> shift) & mask for shift, mask in
                    ((0, 1023), (10, 1023), (20, 1023), (30, 3))]
        value = np.stack(channels, axis=-1).astype(np.float32) / [1023, 1023, 1023, 3]
    else:
        if fmt in (87, 91):
            value = value[:, :, [2, 1, 0, 3]]
        if np.issubdtype(value.dtype, np.integer) and fmt != 62:
            value = value.astype(np.float32) / np.iinfo(value.dtype).max
        else:
            value = value.astype(np.float32)
    return (value, digest) if include_digest else value


def statistics(value):
    finite = value[np.isfinite(value)]
    return {
        "finite_count": int(finite.size), "nonfinite_count": int(value.size - finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean(dtype=np.float64)) if finite.size else None,
    }


def color_preview(value, transfer):
    invalid = ~np.all(np.isfinite(value[:, :, :3]), axis=-1)
    rgb = np.nan_to_num(value[:, :, :3], nan=0, posinf=0, neginf=0)
    if transfer == 3:  # HDR10: ST2084 to linear Rec.2020, then Rec.709.
        power = np.clip(rgb, 0, 1) ** (1 / 78.84375)
        nits = 10000 * (np.maximum(power - 0.8359375, 0) /
                        np.maximum(18.8515625 - 18.6875 * power, 1e-8)) ** (1 / 0.1593017578125)
        rgb = nits @ np.array([[1.660491, -0.124550, -0.018151],
                              [-0.587641, 1.132900, -0.100579],
                              [-0.072850, -0.008349, 1.118730]]) / 80
    if transfer in (2, 3):
        # Simple diagnostic tone mapping, not the host's display/encoding transform.
        linear = np.maximum(rgb, 0)
        linear = linear / (1 + linear)
        rgb = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
    pixels = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    pixels[invalid] = (255, 0, 255)
    return pixels


def scalar_preview(value, low=0.0, high=1.0, signed=False):
    invalid = ~np.isfinite(value)
    scalar = np.nan_to_num(value, nan=0, posinf=0, neginf=0)
    if signed:
        extent = max(abs(low), abs(high))
        amount = np.clip(abs(scalar) / extent, 0, 1) if extent else np.zeros_like(scalar)
        endpoint = np.where((scalar < 0)[..., None], [45, 110, 245], [240, 95, 35])
        pixels = np.rint(225 + amount[..., None] * (endpoint - 225)).astype(np.uint8)
    else:
        shade = (np.rint(np.clip((scalar - low) / (high - low), 0, 1) * 255).astype(np.uint8)
                 if high > low else np.full(scalar.shape, 128, dtype=np.uint8))
        pixels = np.repeat(shade[..., None], 3, axis=-1)
    pixels[invalid] = (255, 0, 255)
    return pixels


def resource_metadata(metadata, key, artifact):
    rows = resource_rows(metadata, key)
    if not isinstance(rows, list):
        return {}
    return next((row for row in rows if isinstance(row, dict) and
                 (row.get("file_stem") == artifact["kind"] or
                  (artifact.get("artifact_id") is not None and
                  row.get("artifact_id") == artifact["artifact_id"]))), {})


def resource_rows(metadata, key):
    # Production nests independent API observations; capture decisions belong to
    # the outer producer. Prefer the canonical location even when it is empty.
    latest = metadata.get("latest_observations", {})
    if key == "ui_resources" and isinstance(latest, dict) and key in latest:
        return latest[key]
    return metadata.get(key, [])


def save_preview(pixels, path, optional, result):
    from PIL import Image
    try:
        Image.fromarray(pixels).save(path)
    except OSError as error:
        if not optional:
            raise
        result.setdefault("preview_errors", []).append(str(error))


def inspect(root, previews=False, ui_review=None):
    root = Path(root)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "sunshine.game3d.dump.v1":
        raise ValueError("This is not a native Game 3D dump")
    report = {"status": manifest.get("status"), "reason": manifest.get("reason"), "artifacts": []}
    metadata = manifest.get("producer_metadata", {})
    report["consumed_depth"] = metadata.get("consumed_depth", {})
    report["render_parameters"] = metadata.get("render_parameters", {})
    report["pairing_evidence"] = metadata.get("pairing_evidence", {})
    report["render_scene_policy"] = metadata.get("render_scene_policy", {})
    report["ui_resources"] = resource_rows(metadata, "ui_resources")
    report["optional_captures"] = metadata.get("optional_captures", [])
    report["optional_capture_errors"] = manifest.get("optional_capture_errors", [])
    report["ui_source"] = metadata.get("ui_source", {})
    report["ui_alpha_source"] = metadata.get("replay", {}).get("ui_alpha_source", "legacy")
    report["ui_mask_channel"] = metadata.get("replay", {}).get("ui_constant_binding", {}).get("mask_channel", "alpha")
    if report["ui_mask_channel"] not in ("alpha", "red"):
        raise ValueError("Unknown consumed UI mask channel")
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) > MAX_ARTIFACTS:
        raise ValueError("Invalid artifact list")
    if report["ui_alpha_source"] == "ui_source_color" and not any(a.get("kind") == "ui_source_color" for a in artifacts):
        raise ValueError("Required consumed UI-alpha texture is missing; optional SL snapshots cannot substitute")
    total_bytes = 0
    facts = []
    for artifact in sorted(artifacts, key=lambda item: item.get("kind") not in PRIMARY_ARTIFACTS):
        optional = artifact["kind"] not in PRIMARY_ARTIFACTS
        try:
            if not isinstance(artifact["byte_count"], int) or not 0 <= artifact["byte_count"] <= 768 * 1024 * 1024 - total_bytes:
                raise ValueError("Capture byte budget exceeded")
            value, digest = read_artifact(root, artifact, include_digest=True)
            total_bytes += artifact["byte_count"]
        except (ValueError, OSError, KeyError, TypeError) as error:
            if not optional:
                raise
            report["artifacts"].append({"kind": artifact["kind"], "status": "unavailable", "reason": str(error)})
            facts.append(artifact_facts(artifact, error=error))
            continue
        facts.append(artifact_facts(artifact, value, artifact_sha256=digest))
        observation = resource_metadata(metadata, "ui_resources", artifact) if optional else {}
        captured = resource_metadata(metadata, "optional_captures", artifact) if optional else {}
        role = observation.get("role", "unknown")
        result = {"kind": artifact["kind"], **statistics(value)}
        if optional:
            result.update(role=role, capture=captured,
                          allocation="full; no depth crop or jitter applied")
        if previews:
            output = root / "previews"
            output.mkdir(exist_ok=True)
            stem = Path(artifact["file"]).stem
            if value.shape[-1] >= 3 and (not optional or role not in ("mask", "alpha", "motion")):
                transfer = metadata.get("color_space", 0)
                if optional:
                    transfer = (captured.get("color_space", 0)
                                if captured.get("transfer_status") in ("declared", "assumed") else 0)
                    if transfer not in (1, 2, 3):
                        transfer = 0
                elif artifact["kind"] == "ui_source_color":
                    transfer = 0  # Mask channel only; do not invent RGB transfer metadata.
                elif artifact["kind"] == "linear_color" or (artifact["kind"] == "sbs" and transfer == 3):
                    transfer = 2
                pixels = color_preview(value, transfer)
                result["preview_mapping"] = ("Unknown transfer; RGB code values displayed, no final-color transfer or premultiplication assumed"
                                             if not transfer else
                                             f"Display-only color preview; transfer {captured.get('transfer_status', 'captured')}; HDR uses diagnostic tone mapping")
                if optional and value.shape[-1] == 4:
                    save_preview(scalar_preview(value[:, :, 3]), output / (stem + "_alpha.png"), optional, result)
                    result["alpha_preview"] = dict(channel="A", black=0, white=1,
                                                   **statistics(value[:, :, 3]))
            elif optional:
                result["component_previews"] = []
                for channel, name in enumerate("RGBA"[:value.shape[-1]]):
                    scalar = value[:, :, channel]
                    stats = statistics(scalar)
                    motion = role == "motion"
                    lo, hi = (stats["min"] or 0, stats["max"] or 0) if motion else (0, 1)
                    component = scalar_preview(scalar, lo, hi, signed=motion)
                    file = stem + ("_" + name if channel else "") + ".png"
                    save_preview(component, output / file, optional, result)
                    result["component_previews"].append(dict(channel=name, black=lo, white=hi,
                                                              mapping="signed full finite range" if motion else "fixed [0,1]",
                                                              **stats))
                report["artifacts"].append(result)
                continue
            else:
                scalar = value[:, :, 0]
                finite = scalar[np.isfinite(scalar)]
                lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0, 0)
                pixels = scalar_preview(scalar, lo, hi)
                result["preview_mapping"] = {
                    "black": lo, "white": hi, "note": "full finite range; no depth orientation inferred"}
            # Use the validated basename; provider metadata cannot choose another directory.
            save_preview(pixels, output / (stem + ".png"), optional, result)
            if artifact["kind"] == "source_color" and artifact["dxgi_format"] in ALPHA_FORMATS:
                save_preview(scalar_preview(value[:, :, 3]), output / "source_alpha.png", False, result)
                result["source_alpha_preview"] = dict(
                    file="source_alpha.png", source_artifact="source_color", channel="A", black=0, white=1,
                    semantic="uninterpreted_source_alpha",
                    note="Raw source color alpha; UI meaning is not guaranteed. Full allocation, no transfer, crop, jitter or threshold. Never SBS alpha. Generating this preview does not enable source-alpha UI protection.",
                    **statistics(value[:, :, 3]))
            if artifact["kind"] == "ui_source_color":
                red = report["ui_mask_channel"] == "red"
                if not red and artifact["dxgi_format"] not in ALPHA_FORMATS:
                    raise ValueError("Consumed alpha channel requires a texture with alpha")
                mask = value[:, :, 0 if red else 3]
                name = "ui_source_mask" if red else "ui_source_alpha"
                save_preview(scalar_preview(mask), output / (name + ".png"), False, result)
                result[name + "_preview"] = dict(
                    file=name + ".png", source_artifact="ui_source_color", channel="R" if red else "A", black=0, white=1,
                    semantic="consumed_ui_mask" if red else "consumed_ui_alpha",
                    note="Exact consumed mask, not a latest optional SL snapshot. No color transfer, crop, jitter or threshold; same-game-frame pairing is not proven.",
                    **statistics(mask))
        report["artifacts"].append(result)
    report["ui_discovery"] = discover_ui(
        manifest, facts, ui_review, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--previews", action="store_true", help="write diagnostic PNGs; never modify lossless data")
    parser.add_argument("--ui-review", type=Path,
                        help="apply an explicit review bound to this dump and its artifact bytes")
    parser.add_argument("--write-ui-review", type=Path,
                        help="create an unreviewed UI qualification template; never overwrite a file")
    parser.add_argument("--ui-report", type=Path,
                        help="create a readable Markdown UI discovery report; never overwrite a file")
    args = parser.parse_args()
    review = json.loads(args.ui_review.read_text(encoding="utf-8")) if args.ui_review else None
    report = inspect(args.dump, args.previews, review)
    if args.write_ui_review:
        with args.write_ui_review.open("x", encoding="utf-8") as output:
            json.dump(review_template(report["ui_discovery"]), output, indent=2, allow_nan=False)
            output.write("\n")
    if args.ui_report:
        with args.ui_report.open("x", encoding="utf-8") as output:
            output.write(format_report(report["ui_discovery"]))
    print(json.dumps(report, indent=2, allow_nan=False))

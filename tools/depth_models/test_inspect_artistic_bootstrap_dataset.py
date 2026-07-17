import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import inspect_artistic_bootstrap_dataset as audit  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def png_bytes(color, size=(2, 2)):
    import io
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_f32(path, value, width=2, height=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.asarray((width, height), dtype="<u4").tobytes()
    values = np.full((height, width), value, dtype="<f4").tobytes()
    path.write_bytes(header + values)


def geometry(color_mode, eye_width, eye_height):
    scale_x, scale_y = audit.geometry_contract.source_content_scales(
        2, 2, eye_width, eye_height
    )
    return {
        "source_width": 2,
        "source_height": 2,
        "model_input_width": 14,
        "model_input_height": 14,
        "depth_short_side": 196,
        "depth_max_aspect": 8.0,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": eye_width,
        "disparity_raster_height": eye_height,
        "color_mode": color_mode,
    }


def policy_manifests():
    geometries = audit.geometry_contract.build_allowlist([
        geometry(color_mode, size, size)
        for color_mode in (
            audit.input_color.COLOR_MODE_SDR,
            audit.input_color.COLOR_MODE_HDR,
        )
        for size in (2, 3)
    ])
    variants = audit.label_merge.build_input_variant_manifest(
        audit.label_merge.policy_input_variants()
    )
    return geometries, variants


def target_for(variant, scale, evidence):
    variant_hash = audit.input_color.input_variant_sha256(variant)
    action = abs(scale - 1.0) >= audit.ACTION_EPSILON
    reliability = 0.8 if action else 0.0
    styles = {
        "clean": 1.0,
        "balanced": 1.0 + 0.5 * (scale - 1.0),
        "immersive": scale,
    }
    rendered = [item["safe_ceiling_render_target"] for item in evidence]
    conservative = audit.label_merge.conservative_target(rendered, scale)
    return {
        "schema": 1,
        "contract": audit.CONDITION_TARGET_CONTRACT,
        "input_variant": variant,
        "input_variant_sha256": variant_hash,
        "deployment_geometry_variant_count": 2,
        "safe_scale_min": 1.0,
        "safe_scale_max": scale,
        "safe_scale_ceiling": scale,
        "ceiling_confidence": float(action),
        "safety_margin_reliability": reliability,
        "identity_feasible": True,
        "identity_infeasible_variants": [],
        "style_targets": styles,
        "safe_ceiling_render_target": conservative,
        "safe_ceiling_exact_pop_spread_pct": conservative[
            "exact_pop_spread_pct"
        ],
        "style_render_targets": {
            name: audit.label_merge.conservative_target([
                item["style_render_targets"][name] for item in evidence
            ], style_scale)
            for name, style_scale in styles.items()
        },
    }


def evidence_for(workspace, stem, variant, scale, geometry_index):
    color_mode = variant["color_mode"]
    geometry_value = geometry(
        color_mode, 2 + geometry_index, 2 + geometry_index
    )
    path = workspace / "e" / f"{stem}-{geometry_index}.f32"
    write_f32(
        path, 0.01 * (geometry_index + 1),
        width=geometry_value["disparity_raster_width"],
        height=geometry_value["disparity_raster_height"],
    )
    raw = audit.load_float_texture(path)
    reliability = 0.8 if abs(scale - 1.0) >= audit.ACTION_EPSILON else 0.0
    styles = {
        "clean": 1.0,
        "balanced": 1.0 + 0.5 * (scale - 1.0),
        "immersive": scale,
    }
    return {
        "geometry": geometry_value,
        "input_variant": variant,
        "input_variant_sha256": audit.input_color.input_variant_sha256(
            variant
        ),
        "baseline_unclamped_disparity": str(path.resolve()),
        "baseline_unclamped_disparity_sha256": audit.sha256(path),
        "artistic_full_clamp_abs": 0.04,
        "safe_scale_min": 1.0,
        "safe_scale_max": scale,
        "safety_margin_reliability": reliability,
        "identity_feasible": True,
        "identity_violations": [],
        "safe_ceiling_render_target": audit.label_merge.render_target(
            raw, geometry_value, scale, 0.04
        ),
        "style_render_targets": {
            name: audit.label_merge.render_target(
                raw, geometry_value, style_scale, 0.04
            ) for name, style_scale in styles.items()
        },
    }


def write_merged_bundle(workspace, dataset_key, dataset_root, split,
                        production, clip, frame_count, variants,
                        native_frames=None):
    output = workspace / "merged" / dataset_key
    output.mkdir(parents=True)
    rows = []
    geometry_manifest, input_manifest = policy_manifests()
    for frame_id in range(frame_count):
        source = dataset_root / clip / f"frame_{frame_id:05d}.png"
        targets = []
        evidence = []
        for condition_index, variant in enumerate(variants):
            scale = 1.2 if frame_id % 3 else 1.0
            stem = f"{dataset_key[:2]}-{frame_id}-{condition_index}"
            condition_evidence = [
                evidence_for(workspace, stem, variant, scale, 0),
                evidence_for(workspace, stem, variant, scale, 1),
            ]
            target = target_for(variant, scale, condition_evidence)
            targets.append(target)
            evidence.extend(condition_evidence)
            condition = (
                "sdr" if variant["kind"] == audit.input_color.INPUT_KIND_SDR
                else (
                    f"w{variant['windows_sdr_white_level_raw']}"
                    if variant["kind"] ==
                    audit.input_color.INPUT_KIND_WINDOWS_HDR else None
                )
            )
            depth_root = (
                workspace / "depth" / dataset_key
                if condition is None else
                workspace / "depth" / f"{dataset_key}-{condition}"
            )
            depth_path = depth_root / clip / f"depth_{frame_id:05d}.png"
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (2, 2), frame_id % 255).save(depth_path)
        first_target = targets[0]
        row = {
            "label_schema": audit.LABEL_SCHEMA,
            "policy_contract": audit.POLICY_CONTRACT,
            "condition_target_contract": audit.CONDITION_TARGET_CONTRACT,
            "deployment_geometry_allowlist_sha256": (
                audit.geometry_contract.allowlist_sha256(geometry_manifest)
            ),
            "input_variant_manifest": input_manifest,
            "input_variant_manifest_sha256": (
                audit.label_merge.input_variant_manifest_sha256(input_manifest)
            ),
            "depth_input_color_contract_sha256": (
                audit.input_color.color_contract_sha256()
            ),
            "source": str(source.resolve()),
            "source_sha256": audit.sha256(source),
            "clip": clip,
            "frame": frame_id,
            "split": split,
            "film_id": production,
            "input_condition_targets": targets,
            "deployment_geometry_variants": evidence,
            "safe_scale_ceiling": first_target["safe_scale_ceiling"],
            "ceiling_confidence": first_target["ceiling_confidence"],
            "safety_margin_reliability": first_target[
                "safety_margin_reliability"
            ],
            "identity_feasible": True,
        }
        if native_frames is not None:
            native = native_frames[frame_id]
            row.update({
                "model_source": str(native["model_path"].resolve()),
                "model_source_sha256": native["sha256"],
                "model_source_encoding": audit.native_hdr_capture.CAPTURE_ENCODING,
            })
        rows.append(row)
    labels = output / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    code_path = output / "frozen_label_code.py"
    code_path.write_text("# frozen test label code\n", encoding="utf-8")
    code_hash = audit.sha256(code_path)
    contract = {
        "schema": audit.LABEL_SCHEMA,
        "policy_contract": audit.POLICY_CONTRACT,
        "condition_target_contract": audit.CONDITION_TARGET_CONTRACT,
        "deployment_geometry_allowlist": geometry_manifest,
        "deployment_geometry_allowlist_sha256": (
            audit.geometry_contract.allowlist_sha256(geometry_manifest)
        ),
        "input_variant_manifest": input_manifest,
        "input_variant_manifest_sha256": (
            audit.label_merge.input_variant_manifest_sha256(input_manifest)
        ),
        "depth_input_color_contract_sha256": (
            audit.input_color.color_contract_sha256()
        ),
        "code": {
            role: {"path": str(code_path), "sha256": code_hash}
            for role in sorted(
                audit.label_merge.MERGED_LABEL_FITTER_CODE_ROLES
            )
        },
    }
    contract_path = output / "label_fitter_contract.json"
    write_json(contract_path, contract)
    write_json(output / "summary.json", {
        "schema": audit.LABEL_SCHEMA,
        "accepted": len(rows),
        "labels_sha256": audit.sha256(labels),
        "label_fitter_contract_sha256": audit.sha256(contract_path),
        "condition_target_contract": audit.CONDITION_TARGET_CONTRACT,
    })
    return output


def write_depth_publications(workspace, dataset_key, dataset_root, clip,
                             frame_count, variants):
    steps = []
    dataset_manifest = dataset_root / "dataset_manifest.json"
    clip_manifest = dataset_root / audit.clip_hashes.MANIFEST_NAME
    for variant in variants:
        condition = audit.condition_name(variant)
        if condition == "native-pq":
            output = workspace / "depth" / dataset_key
        else:
            suffix = "sdr" if condition == "sdr" else condition.removeprefix(
                "hdr-"
            )
            output = workspace / "depth" / f"{dataset_key}-{suffix}"
        clip_root = output / clip
        clip_root.mkdir(parents=True, exist_ok=True)
        write_json(clip_root / "contract.json", {
            "schema": 1,
            "input_variant": variant,
            "input_variant_sha256": audit.input_variant_key(variant),
        })
        write_json(clip_root / "generation_identity.json", {
            "schema": 1, "clip": clip,
        })
        for frame_id in range(frame_count):
            write_f32(
                clip_root / f"baseline_disparity_{frame_id:05d}.f32",
                0.01,
            )
            write_f32(
                clip_root / f"baseline_unclamped_disparity_{frame_id:05d}.f32",
                0.01,
            )
        identity = audit.depth_run.depth_artifact_identity(clip_root)
        manifest = {
            "schema": audit.depth_run.DEPTH_RUN_MANIFEST_SCHEMA,
            "purpose": "artistic-policy depth supervision",
            "suite": str(dataset_root.resolve()),
            "suite_manifest_sha256": audit.sha256(dataset_manifest),
            "input_variant": variant,
            "input_variant_sha256": audit.input_variant_key(variant),
            "depth_input_color_contract_sha256": (
                audit.input_color.color_contract_sha256()
            ),
            "clips": [{
                "clip": clip,
                "frames": frame_count,
                "contract_sha256": audit.sha256(clip_root / "contract.json"),
                **identity,
            }],
            "clip_count": 1,
            "frame_count": frame_count,
        }
        write_json(output / "depth_run_manifest.json", manifest)
        steps.append({
            "key": f"depth-{dataset_key}-{condition}",
            "phase": "depth",
            "kind": "depth",
            "output": str(output.resolve()),
            "argv": [],
            "metadata": {
                "dataset": dataset_key,
                "condition": condition,
                "dataset_root": str(dataset_root.resolve()),
                "dataset_manifest_sha256": audit.sha256(dataset_manifest),
                "clip_hash_manifest_sha256": audit.sha256(clip_manifest),
                "input_variant": variant,
            },
        })
    return steps


def write_sdr_dataset(workspace, source, split, count, color_offset,
                      leak_bytes=None):
    root = workspace / "datasets" / source / split
    clip = f"{source}_{split}"
    clip_root = root / clip
    clip_root.mkdir(parents=True)
    frames = []
    for frame_id in range(count):
        value = (color_offset + frame_id) % 250 + 1
        data = (
            leak_bytes if frame_id == 0 and leak_bytes is not None
            else png_bytes((value, value // 2, 255 - value))
        )
        path = clip_root / f"frame_{frame_id:05d}.png"
        path.write_bytes(data)
        frames.append({
            "output": path.name,
            "sha256": audit.sha256(path),
            "source_sha256": audit.sha256(path),
        })
    write_json(clip_root / "label_frames.json", {
        "schema": 1, "frame_ids": list(range(count)),
    })
    production = f"{source}_mono_hdr_bootstrap_v1_{split}"
    write_json(clip_root / "meta.json", {
        "split": split, "production_id": production,
        "source_kind": "mono-video",
    })
    source_sequence = root / "source_sequence_manifest.json"
    write_json(source_sequence, {
        "schema": 1, "production_id": production,
        "sequences": [{"clip": clip, "frames": frames}],
    })
    dataset_manifest = root / "dataset_manifest.json"
    write_json(dataset_manifest, {
        "schema": 2,
        "dataset": source,
        "production_id": production,
        "split": split,
        "source_kind": "mono-video",
        "source_sequence_manifest": source_sequence.name,
        "video_sha256": audit.sha256(source_sequence),
        "sequences": [{"clip": clip, "label_frames": count, "split": split}],
    })
    clip_manifest, clip_manifest_path = audit.clip_hashes.build_and_write(
        root, clips=[clip], workers=1
    )
    dataset_key = f"{source}-{split}"
    variants = [
        audit.input_color.sdr_input_variant(),
        *(audit.input_color.windows_hdr_input_variant(value)
          for value in (1000, 2500, 6000)),
    ]
    write_merged_bundle(
        workspace, dataset_key, root, split, production, clip, count, variants
    )
    depth_steps = write_depth_publications(
        workspace, dataset_key, root, clip, count, variants
    )
    return {
        "source": source,
        "split": split,
        "production_id": production,
        "root": str(root.resolve()),
        "output_root": str(root.resolve()),
        "clips": [clip],
        "label_frames": count,
        "label_frame_count": count,
        "context_frame_count": count,
        "dataset_manifest_sha256": audit.sha256(dataset_manifest),
        "clip_hash_manifest_sha256": audit.sha256(clip_manifest_path),
        "clip_hash_semantic_content_sha256": clip_manifest[
            audit.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ],
        "source_sequence_manifest_sha256": audit.sha256(source_sequence),
        "depth_steps": depth_steps,
        "first_frame_bytes": (clip_root / "frame_00000.png").read_bytes(),
    }


def build_sdr_workspace(root, leak=False):
    workspace = root / "sdr"
    specifications = [
        ("reds", "training", 40, 10),
        ("spring", "training", 20, 70),
        ("reds", "development", 10, 130),
        ("spring", "development", 10, 170),
    ]
    rows = []
    leaked = None
    for source, split, count, offset in specifications:
        use_leak = (
            leaked
            if leak and split == "development" and source == "reds"
            else None
        )
        row = write_sdr_dataset(
            workspace, source, split, count, offset, use_leak
        )
        if source == "reds" and split == "training":
            leaked = row["first_frame_bytes"]
        row.pop("first_frame_bytes")
        rows.append(row)
    bootstrap = {
        "schema": 1,
        "preparation_contract": audit.SDR_BOOTSTRAP_CONTRACT,
        "datasets": [{
            key: value for key, value in row.items()
            if key not in {"root", "label_frames", "depth_steps"}
        } for row in rows],
    }
    bootstrap_path = workspace / "datasets" / "bootstrap_manifest.json"
    write_json(bootstrap_path, bootstrap)
    plan = {
        "schema": audit.SDR_PLAN_SCHEMA,
        "contract": audit.SDR_PLAN_CONTRACT,
        "workspace": str(workspace.resolve()),
        "bootstrap_manifest": str(bootstrap_path.resolve()),
        "bootstrap_manifest_sha256": audit.sha256(bootstrap_path),
        "deployment_geometry_manifest": policy_manifests()[0],
        "deployment_geometry_manifest_identity": (
            audit.geometry_contract.allowlist_sha256(policy_manifests()[0])
        ),
        "input_variant_manifest": policy_manifests()[1],
        "input_variant_manifest_identity": (
            audit.label_merge.input_variant_manifest_sha256(
                policy_manifests()[1]
            )
        ),
        "condition_target_contract": audit.CONDITION_TARGET_CONTRACT,
        "datasets": [{
            key: value for key, value in row.items()
            if key not in {
                "output_root", "production_id", "label_frame_count",
                "context_frame_count", "clip_hash_semantic_content_sha256",
                "source_sequence_manifest_sha256",
                "depth_steps",
            }
        } for row in rows],
        "steps": [step for row in rows for step in row["depth_steps"]],
    }
    write_json(workspace / audit.SDR_PLAN, plan)
    return workspace


def flow_selection():
    return {
        "contract": audit.chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT,
        "flow_support_contract": audit.chug_prepare.FLOW_SUPPORT_CONTRACT,
        "flow_support_metric_sha256": (
            audit.chug_prepare.flow_support_metric_sha256()
        ),
        "preferred_pair": "previous-source-frame-to-label-frame",
        "minimum_support": audit.chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT,
        "search_radius_frames": (
            audit.chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES
        ),
        "search_order": "nominal-then-negative-positive-by-distance",
        "nominal_source_label_frame_id": 1,
        "selected_source_label_frame_id": 1,
        "selected_offset_frames": 0,
        "selected_previous_source_frame_id": 0,
        "selected_pair_flow_support": 0.5,
    }


def native_manifest(clip_root, frame_count, conversion_hash,
                    value_offset=0, nonfinite=False, selection=None):
    model_root = clip_root / audit.native_hdr_capture.MODEL_SOURCE_DIRECTORY
    model_root.mkdir()
    frame_rows = []
    semantic_rows = []
    for frame_id in range(frame_count):
        model_path = model_root / f"frame_{frame_id:05d}.scrgb16"
        rgba = np.full(
            (2, 2, 4), 2.0 + (value_offset + frame_id) / 1000.0,
            dtype="<f2",
        )
        rgba[..., 3] = 1.0
        if nonfinite and frame_id == 0:
            rgba[0, 0, 0] = np.float16(np.nan)
        model_path.write_bytes(rgba.tobytes())
        preview = clip_root / f"frame_{frame_id:05d}.png"
        preview.write_bytes(png_bytes((
            80 + (value_offset + frame_id) % 150,
            20 + value_offset // 100,
            10,
        )))
        stat = model_path.stat()
        row = {
            "frame": frame_id,
            "path": model_path.relative_to(clip_root).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": audit.sha256(model_path),
            "preview": preview.name,
            "preview_sha256": audit.sha256(preview),
            "timestamp_seconds": frame_id / 24.0,
            "stats": {},
        }
        frame_rows.append(row)
        semantic_rows.append({
            key: row[key] for key in (
                "frame", "path", "size", "sha256", "preview",
                "preview_sha256", "timestamp_seconds",
            )
        })
    source_video = {
        "dataset": "CHUG",
        "license": "CC BY-NC-SA 4.0",
        "sha256": audit.canonical_sha256(str(clip_root)),
        "frame_selection_contract": audit.chug_prepare.FRAME_SELECTION_CONTRACT,
        "source_label_frame_id": selection["selected_source_label_frame_id"],
        "temporal_evidence_selection": selection,
    }
    conversion = {"contract_sha256": conversion_hash}
    semantic = {
        "contract": audit.native_hdr_capture.MANIFEST_CONTRACT,
        "capture_encoding": audit.native_hdr_capture.CAPTURE_ENCODING,
        "preview_encoding": audit.native_hdr_capture.PREVIEW_ENCODING,
        "width": 2,
        "height": 2,
        "row_pitch_bytes": 16,
        "source_video": source_video,
        "conversion": conversion,
        "frames": semantic_rows,
    }
    payload = {
        "schema": audit.native_hdr_capture.MANIFEST_SCHEMA,
        **semantic,
        "frames": frame_rows,
        "frame_count": frame_count,
        "content_sha256": audit.native_hdr_capture.canonical_sha256(semantic),
    }
    write_json(clip_root / audit.native_hdr_capture.MANIFEST_NAME, payload)
    return audit.native_hdr_capture.validate_clip(clip_root, full=True)


def write_native_dataset(workspace, split, count, conversion_hash,
                         nonfinite=False):
    root = workspace / "datasets" / split
    clip = f"native_{split}"
    clip_root = root / clip
    clip_root.mkdir(parents=True)
    selection = flow_selection()
    authentication = native_manifest(
        clip_root, count, conversion_hash,
        0 if split == "training" else 100, nonfinite, selection,
    )
    write_json(clip_root / "label_frames.json", {
        "schema": 1, "frame_ids": list(range(count)),
    })
    production = f"chug_native_pq_v1_{split}"
    capture_group = f"capture-{split}"
    write_json(clip_root / "meta.json", {
        "split": split,
        "production_id": production,
        "source_kind": "native-hdr-video",
        "conversion_contract_sha256": conversion_hash,
        "capture_group_id": capture_group,
        "frame_selection": {
            "contract": audit.chug_prepare.FRAME_SELECTION_CONTRACT,
            "source_label_frame_id": selection[
                "selected_source_label_frame_id"
            ],
            "temporal_evidence_selection": selection,
        },
    })
    dataset_manifest = root / "dataset_manifest.json"
    write_json(dataset_manifest, {
        "schema": 2,
        "dataset": "chug-native-pq-v1",
        "production_id": production,
        "split": split,
        "source_kind": "native-hdr-video",
        "preparation_contract": audit.NATIVE_BOOTSTRAP_CONTRACT,
        "temporal_evidence_selection_contract": (
            audit.chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT
        ),
        "source_flow_support_contract": audit.chug_prepare.FLOW_SUPPORT_CONTRACT,
        "source_flow_metric_sha256": (
            audit.chug_prepare.flow_support_metric_sha256()
        ),
        "source_flow_support_minimum": (
            audit.chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT
        ),
        "conversion_contract_sha256": conversion_hash,
        "label_frame_count": count,
        "sequences": [{
            "clip": clip, "label_frames": count, "split": split,
            "capture_group_id": capture_group,
            "nominal_source_label_frame_id": selection[
                "nominal_source_label_frame_id"
            ],
            "source_label_frame_id": selection[
                "selected_source_label_frame_id"
            ],
            "selected_pair_flow_support": selection[
                "selected_pair_flow_support"
            ],
            "temporal_evidence_selection": selection,
        }],
    })
    clip_manifest, clip_manifest_path = audit.clip_hashes.build_and_write(
        root, clips=[clip], workers=1
    )
    dataset_key = f"chug-native-pq-{split}"
    write_merged_bundle(
        workspace, dataset_key, root, split, production, clip, count,
        [audit.input_color.native_pq_input_variant()],
        native_frames=authentication["frames"],
    )
    depth_steps = write_depth_publications(
        workspace, dataset_key, root, clip, count,
        [audit.input_color.native_pq_input_variant()],
    )
    return {
        "split": split,
        "root": str(root.resolve()),
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": audit.sha256(dataset_manifest),
        "clip_hash_manifest": {
            "path": str(clip_manifest_path.resolve()),
            "sha256": audit.sha256(clip_manifest_path),
            "semantic_content_sha256": clip_manifest[
                audit.clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
            ],
        },
        "clip_hash_manifest_sha256": audit.sha256(clip_manifest_path),
        "clips": [clip],
        "label_frames": count,
        "label_frame_count": count,
        "capture_group_ids": [capture_group],
        "depth_steps": depth_steps,
    }


def build_native_workspace(root, nonfinite=False):
    workspace = root / "native"
    conversion = {
        "schema": 1,
        "contract": audit.NATIVE_CONVERSION_CONTRACT,
        "source": {
            "codec": "hevc",
            "pixel_format": "yuv420p10+",
            "range": "limited",
            "primaries": "bt2020",
            "matrix": "bt2020nc",
            "transfer": "smpte2084",
        },
        "scrgb_reference_white_nits": 80.0,
        "depth_input_color_contract_sha256": (
            audit.input_color.color_contract_sha256()
        ),
    }
    conversion_path = workspace / "datasets" / "conversion_contract.json"
    write_json(conversion_path, conversion)
    conversion_hash = audit.sha256(conversion_path)
    rows = [
        write_native_dataset(
            workspace, "training", 60, conversion_hash, nonfinite
        ),
        write_native_dataset(
            workspace, "development", 20, conversion_hash, False
        ),
    ]
    bootstrap = {
        "schema": audit.NATIVE_BOOTSTRAP_SCHEMA,
        "contract": audit.NATIVE_BOOTSTRAP_CONTRACT,
        "output_root": str((workspace / "datasets").resolve()),
        "sealed_test_policy": "CHUG test masters were not decoded or opened",
        "conversion_contract": str(conversion_path.resolve()),
        "conversion_contract_sha256": audit.sha256(conversion_path),
        "temporal_evidence_selection_contract": (
            audit.chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT
        ),
        "source_flow_support_contract": audit.chug_prepare.FLOW_SUPPORT_CONTRACT,
        "source_flow_metric_sha256": (
            audit.chug_prepare.flow_support_metric_sha256()
        ),
        "source_flow_support_minimum": (
            audit.chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT
        ),
        "datasets": {row["split"]: {
            key: value for key, value in row.items()
            if key not in {
                "clip_hash_manifest_sha256", "label_frames", "depth_steps"
            }
        } for row in rows},
    }
    bootstrap_path = workspace / "datasets" / "native_hdr_bootstrap_manifest.json"
    write_json(bootstrap_path, bootstrap)
    plan = {
        "schema": audit.NATIVE_PLAN_SCHEMA,
        "contract": audit.NATIVE_PLAN_CONTRACT,
        "training_command_present": False,
        "bootstrap_manifest": str(bootstrap_path.resolve()),
        "bootstrap_manifest_sha256": audit.sha256(bootstrap_path),
        "deployment_geometry_manifest": policy_manifests()[0],
        "deployment_geometry_manifest_identity": (
            audit.geometry_contract.allowlist_sha256(policy_manifests()[0])
        ),
        "input_variant_manifest": policy_manifests()[1],
        "input_variant_manifest_identity": (
            audit.label_merge.input_variant_manifest_sha256(
                policy_manifests()[1]
            )
        ),
        "condition_target_contract": audit.CONDITION_TARGET_CONTRACT,
        "datasets": [{
            key: value for key, value in row.items()
            if key not in {
                "dataset_manifest", "clip_hash_manifest",
                "capture_group_ids", "label_frame_count",
                "depth_steps",
            }
        } for row in rows],
        "steps": [step for row in rows for step in row["depth_steps"]],
    }
    write_json(workspace / audit.NATIVE_PLAN, plan)
    return workspace


class InspectArtisticBootstrapTests(unittest.TestCase):
    def test_16bit_depth_contact_uses_percentile_display_stretch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            values = np.linspace(
                1000, 60000, num=64, dtype=np.uint16
            ).reshape(8, 8)
            Image.fromarray(values).save(path)

            uri = audit.depth_image_data_uri(path)
            encoded = uri.split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                rendered = np.asarray(image.convert("L"))

            self.assertGreater(int(rendered.max()) - int(rendered.min()), 220)
            self.assertGreater(np.unique(rendered).size, 16)

    def test_complete_two_branch_inspection_passes_and_embeds_contact_sheets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            output = root / "report"

            result = audit.inspect(
                sdr, native, output, contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "pass")
            self.assertEqual(
                result["actual_combined_policy_cardinality"],
                {"training": 300, "development": 100},
            )
            self.assertFalse(result["training_started"])
            self.assertFalse(result["sealed_test_accessed"])
            self.assertEqual(
                set(result["runtime_regimes"]),
                {
                    "sdr", "hdr", "hdr-w1000", "hdr-w2500",
                    "hdr-w6000", "native-pq",
                },
            )
            self.assertEqual(
                result["runtime_regimes"]["hdr"]["splits"]["training"][
                    "sample_count"
                ],
                180,
            )
            contact_conditions = [
                row["condition"] for row in result["contact_sheet_samples"]
            ]
            self.assertEqual(
                set(contact_conditions),
                {"sdr", "hdr-w1000", "hdr-w2500", "hdr-w6000", "native-pq"},
            )
            self.assertEqual(len(contact_conditions), 10)
            self.assertTrue((output / "inspection.json").is_file())
            report = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("data:image/jpeg;base64,", report)
            self.assertIn("Native HDR signal audit", report)
            self.assertIn("Runtime-regime gate", report)
            self.assertIn("p1-p99 display stretch", report)
            self.assertIn("baseline → selected", report)

    def test_missing_branch_requires_allow_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            missing = root / "missing-native"

            rejected = audit.inspect(
                sdr, missing, root / "rejected", contact_samples_per_split=1
            )
            admitted = audit.inspect(
                sdr, missing, root / "partial", allow_partial=True,
                contact_samples_per_split=1,
            )

            self.assertEqual(rejected["verdict"], "fail")
            self.assertEqual(admitted["verdict"], "pass_partial")
            self.assertEqual(
                admitted["actual_combined_policy_cardinality"],
                {"training": 240, "development": 80},
            )

    def test_native_nonfinite_fp16_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root, nonfinite=True)

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("non-finite", " ".join(result["errors"]))

    def test_train_development_source_sha_leakage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root, leak=True)
            native = build_native_workspace(root)

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("leakage", " ".join(result["errors"]))

    def test_condition_with_one_geometry_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            labels = (
                native / "merged" / "chug-native-pq-training" /
                "labels.jsonl"
            )
            rows = [json.loads(line) for line in labels.read_text(
                encoding="utf-8"
            ).splitlines()]
            rows[0]["deployment_geometry_variants"].pop()
            labels.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path = labels.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["labels_sha256"] = audit.sha256(labels)
            write_json(summary_path, summary)

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("geometry evidence cardinality", " ".join(
                result["errors"]
            ))

    def test_stale_displayed_depth_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            depth = next((
                sdr / "depth" / "reds-training-sdr" /
                "reds_training"
            ).glob("depth_*.png"))
            depth.write_bytes(png_bytes((1, 2, 3)))

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("depth artifact identity differs", " ".join(
                result["errors"]
            ))

    def test_changed_label_fitter_code_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            contract_path = (
                sdr / "merged" / "reds-training" /
                "label_fitter_contract.json"
            )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            code_path = Path(contract["code"]["image_loader"]["path"])
            code_path.write_text("# changed after labels\n", encoding="utf-8")

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("code identity is missing or changed", " ".join(
                result["errors"]
            ))

    def test_stale_geometry_manifest_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            plan_path = sdr / audit.SDR_PLAN
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["deployment_geometry_manifest_identity"] = "0" * 64
            write_json(plan_path, plan)

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("geometry-manifest identity differs", " ".join(
                result["errors"]
            ))

    def test_stale_safe_render_target_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdr = build_sdr_workspace(root)
            native = build_native_workspace(root)
            labels = sdr / "merged" / "reds-training" / "labels.jsonl"
            rows = [json.loads(line) for line in labels.read_text(
                encoding="utf-8"
            ).splitlines()]
            rows[0]["input_condition_targets"][0][
                "safe_ceiling_render_target"
            ]["exact_pop_spread_pct"] += 1.0
            labels.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path = labels.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["labels_sha256"] = audit.sha256(labels)
            write_json(summary_path, summary)

            result = audit.inspect(
                sdr, native, root / "report", contact_samples_per_split=1
            )

            self.assertEqual(result["verdict"], "fail")
            self.assertIn("safe_ceiling_render_target", " ".join(
                result["errors"]
            ))


if __name__ == "__main__":
    unittest.main()

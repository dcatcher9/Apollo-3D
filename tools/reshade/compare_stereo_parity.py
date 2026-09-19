"""Validate full-effect capture matching and report synthetic edge diagnostics.

This reads actual ReShade artifacts. It neither implements a stereo renderer nor
accepts a production calibration policy. Exit 0 means an eligible comparison,
not that visual quality has passed. Final outputs are never masked or rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import PIL
import onnx


class Ineligible(ValueError):
    """The retained evidence cannot support a matched comparison."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_runtime(control_path: Path, treatment_path: Path) -> dict:
    current = (f"{sys.executable}\n{sys.version}\nnumpy {np.__version__}\n"
               f"Pillow {PIL.__version__}\nonnx {onnx.__version__}\n")
    control, treatment = (p.read_text(encoding="utf-8-sig") for p in (control_path, treatment_path))
    if control != treatment or control != current:
        raise Ineligible("Control, treatment and reporting numeric-runtime fingerprints differ")
    return {"fingerprint": current, "control_record": str(control_path.resolve()),
            "treatment_record": str(treatment_path.resolve()),
            "control_record_sha256": sha256(control_path), "treatment_record_sha256": sha256(treatment_path)}


def landmarks(pixels: np.ndarray) -> np.ndarray:
    """Measure the fixture's three independent interior luminance markers."""
    h, packed, _ = pixels.shape
    w = packed // 2
    top = h * 3 // 4
    result = np.empty((3, 2), dtype=np.float64)
    for plane in range(3):
        y = top + (2 * plane + 1) * (h - top) // 6
        for eye in range(2):
            row = pixels[y, eye * w:(eye + 1) * w, 0].astype(np.float64)
            background = .5 * (row[w // 4] + row[w * 3 // 4])
            weight = np.maximum(0, row[w // 4:w * 3 // 4] - background)
            mass = weight.sum()
            if not 3 < mass < 20:
                raise Ineligible("Missing, clipped, or distorted interior landmark")
            x = np.arange(w // 4, w * 3 // 4, dtype=np.float64) + .5
            result[plane, eye] = (x * weight).sum() / mass - w / 2
    return result


def match_landmarks(control: np.ndarray, treatment: np.ndarray) -> dict:
    """Require same signed per-eye positions as well as binocular strength."""
    delta = treatment - control
    disparities = lambda x: x[:, 0] - x[:, 1]
    binocular = disparities(treatment) - disparities(control)
    eye_error = float(np.abs(delta[:2]).max())
    binocular_error = float(np.abs(binocular[:2]).max())
    if eye_error > .25 or binocular_error > .25:
        raise Ineligible(f"Unmatched endpoint geometry: eye={eye_error:.6f}px, binocular={binocular_error:.6f}px")
    if abs(float(disparities(control)[0] - disparities(control)[1])) <= 1:
        raise Ineligible("Reference relief is too weak for a matched edge-quality comparison")
    return {"control_eye_shifts_px": control.tolist(), "treatment_eye_shifts_px": treatment.tolist(),
            "max_endpoint_eye_error_px": eye_error, "max_endpoint_binocular_error_px": binocular_error,
            "holdout_eye_delta_px": delta[2].tolist(), "holdout_binocular_delta_px": float(binocular[2])}


def edge_diagnostic(pixels: np.ndarray, source: np.ndarray, depth: np.ndarray,
                    measured: np.ndarray) -> dict:
    """Chroma leakage into a known constant-background synthetic step.

    Each row is two known, constant-color planes. Behind the foreground the
    fixture's background has the same color. We measure beyond a four-pixel
    margin from the analytically translated foreground silhouette: this also
    excludes the deliberately mixed two-pixel source fringe. This is a limited
    synthetic halo diagnostic, not an oracle for unknown game disocclusions.
    """
    h, w = depth.shape
    top = h * 3 // 4
    foreground = np.isclose(depth, .02, rtol=0, atol=1e-7)
    values, foreground_values = [], []
    bands = {(0, 2): [], (2, 4): [], (4, 8): [], (8, 32): []}
    x = np.arange(w, dtype=np.float64) + .5
    for y in range(16, top - 16):
        if abs(y - top // 2) < 16:
            continue
        changes = np.flatnonzero(foreground[y, 1:] != foreground[y, :-1]) + 1
        if len(changes) != 1:
            raise Ineligible("Step diagnostic source must have exactly one depth silhouette per row")
        boundary = int(changes[0])
        left_foreground = bool(foreground[y, 0])
        fg = source[y, w // 4 if left_foreground else 3 * w // 4, :3].astype(np.float64)
        bg = source[y, 3 * w // 4 if left_foreground else w // 4, :3].astype(np.float64)
        direction = fg - bg
        norm = float(direction @ direction)
        if norm < .1:
            raise Ineligible("Step diagnostic source has insufficient foreground/background contrast")
        for eye in range(2):
            # Native pixel boundaries, shifted by the independently measured
            # foreground interior. No shader's boundary depth is consulted.
            distance = x - (boundary + measured[0, eye])
            if not left_foreground:
                distance = -distance
            rgb = pixels[y, eye * w:(eye + 1) * w, :3].astype(np.float64)
            chroma = ((rgb - bg) @ direction) / norm
            values.extend(np.maximum(0, chroma[(distance >= 4) & (distance < 32)]).tolist())
            foreground_values.extend(chroma[(distance <= -4) & (distance > -32)].tolist())
            for (low, high), samples in bands.items():
                samples.extend(np.maximum(0, chroma[(distance >= low) & (distance < high)]).tolist())
    if not values or not foreground_values:
        raise Ineligible("No supported silhouette samples")
    leakage = np.array(values)
    retained = np.array(foreground_values)
    return {"known_background_chroma_leakage_mean": float(leakage.mean()),
            "known_background_chroma_leakage_p99": float(np.quantile(leakage, .99)),
            "known_background_chroma_leakage_max": float(leakage.max()),
            "known_background_sample_count": int(leakage.size),
            "foreground_retained_chroma_mean": float(retained.mean()),
            "foreground_retained_chroma_p01": float(np.quantile(retained, .01)),
            "margin_px": 4, "ring_outer_px": 32,
            "unexcluded_background_bands_descriptive": {
                f"{low}_to_{high}_px": {"mean": float(np.mean(samples)), "p99": float(np.quantile(samples, .99)),
                                       "max": float(np.max(samples)), "samples": len(samples)}
                for (low, high), samples in bands.items()},
            "inner_band_caveat": "0-4px includes ordinary AA and deliberately mixed source fringe; report, do not discard or call all of it a defect",
            "scope": "Known constant-color step/fringe only; not whole-image quality or unknown hidden game texture"}


def metadata(directory: Path) -> tuple[int, int, int]:
    text = (directory / "measurements.txt").read_text()
    found = re.search(r"renderer=\S+ variant=.*? width=(\d+) height=(\d+) source_color=(\d+)", text)
    if not found:
        raise Ineligible("Missing parity dimensions/color declaration")
    if "dump=float32_little_endian_RGBA" not in text or "rgb=linear_Rec709" not in text:
        raise Ineligible("Unsupported final pixel representation")
    return tuple(map(int, found.groups()))


def read_pixels(path: Path, w: int, h: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != h * 2 * w * 4 or not np.isfinite(values).all():
        raise Ineligible(f"Invalid full final output: {path.name}")
    return values.reshape(h, 2 * w, 4)


def final_aa_cases(control: Path, treatment: Path) -> tuple[int, ...]:
    """An explicit paired contract is required to narrow legacy AA0/AA1 evidence."""
    paths = [directory / "final-aa-contract.txt" for directory in (control, treatment)]
    present = [path.exists() for path in paths]
    if not any(present):
        return (0, 1)
    if not all(present):
        raise Ineligible("Final-AA contract is present on only one side")
    contracts = [path.read_bytes() for path in paths]
    if contracts[0] != contracts[1]:
        raise Ineligible("Control and treatment final-AA contracts differ")
    if contracts[0] != b"schema=final-aa-parity-1\nfinal_aa=off\ncase_branches=1\n":
        raise Ineligible("Unsupported or malformed final-AA contract")
    return (0,)


def compare(control: Path, treatment: Path) -> dict:
    shape = metadata(control)
    if metadata(treatment) != shape:
        raise Ineligible("Dimensions or source color differ")
    w, h, color = shape
    if color not in (1, 2) or w < 256 or h < 128:
        raise Ineligible("Supported scope is SDR/scRGB fixtures at least 256x128")
    # The first two entries identify the actual test executable and official
    # runtime. Different source trees are allowed; different capture machinery
    # needs a separately justified comparison and is not accepted here.
    provenance = [(p / "provenance.txt").read_text().splitlines() for p in (control, treatment)]
    if any(len(lines) < 2 for lines in provenance) or provenance[0][:2] != provenance[1][:2]:
        raise Ineligible("Fixture executable or official runtime provenance differs")
    aa_cases = final_aa_cases(control, treatment)
    inputs = []
    motion_names = [f"motion-step-aa{aa}-f{frame}" for aa in aa_cases for frame in range(7)]
    motion_present = [sorted(p.name for p in directory.glob("motion-*.rgba.f32")) for directory in (control, treatment)]
    expected_motion = sorted(f"{name}.rgba.f32" for name in motion_names)
    if motion_present[0] != motion_present[1] or (motion_present[0] and motion_present[0] != expected_motion):
        raise Ineligible("Moving sequence is absent, incomplete or different between runs")
    if not motion_present[0]:
        motion_names = []
    for scene in ("calibration", "step", "detail", "fringe", *motion_names):
        for suffix in ("source.bin", "depth.f32"):
            name = f"{scene}.{suffix}"
            if (control / name).read_bytes() != (treatment / name).read_bytes():
                raise Ineligible(f"Source/depth bytes differ: {name}")
            inputs.append({"name": name, "sha256": sha256(control / name)})
    cases = []
    captures = [(scene, f"{scene}-aa{aa}", scene != "detail")
                for scene in ("step", "detail", "fringe") for aa in aa_cases]
    captures += [(name, name, True) for name in motion_names]
    for scene, label, known_step in captures:
        dtype = np.dtype("<f2") if color == 2 else np.dtype("u1")
        source = np.fromfile(control / f"{scene}.source.bin", dtype=dtype).reshape(h, w, 4).astype(np.float32)
        if color == 1:
            source /= 255
            source[..., :3] = np.where(source[..., :3] <= .04045, source[..., :3] / 12.92,
                                       ((source[..., :3] + .055) / 1.055) ** 2.4)
        depth = np.fromfile(control / f"{scene}.depth.f32", dtype="<f4").reshape(h, w)
        name = f"{label}.rgba.f32"
        a, b = (read_pixels(p / name, w, h) for p in (control, treatment))
        ma, mb = landmarks(a), landmarks(b)
        match = match_landmarks(ma, mb)
        delta = b.astype(np.float64) - a
        row = {"name": name, "control_sha256": sha256(control / name),
               "treatment_sha256": sha256(treatment / name), "geometry": match,
               "identical_full_rgba": bool(np.array_equal(a, b)),
               "full_linear_rgb_rms_difference_descriptive": float(np.sqrt(np.mean(delta[..., :3] ** 2))),
               "full_linear_rgb_max_difference_descriptive": float(np.abs(delta[..., :3]).max()),
               "full_alpha_max_difference_descriptive": float(np.abs(delta[..., 3]).max())}
        if known_step:
            row["control_edge"] = edge_diagnostic(a, source, depth, ma)
            row["treatment_edge"] = edge_diagnostic(b, source, depth, mb)
        cases.append(row)
    return {"schema": "sunshine-matched-parity-diagnostics-1", "eligible_matched_comparison": True,
            "quality_pass": None, "production_calibration_validated": False,
            "dimensions": [w, h], "source_color": color, "identical_inputs": inputs, "cases": cases,
            "final_aa_case_branches": list(aa_cases),
            "moving_sequence_frames_per_aa": 7 if motion_names else 0,
            "limitations": ["Synthetic fixtures only; no moving-game/headset acceptance",
                            "Moving sequence advances one effect frame per input, not controlled real-time cadence",
                            "Endpoint fit holds out middle depth; its transfer difference remains visible",
                            "Full outputs retained; image differences are descriptive, not error against ground truth",
                            "Matched automatic camera values are test injection, not evidence of automatic calibration"],
            "numeric_runtime": {"executable": sys.executable, "version": sys.version, "numpy": np.__version__}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-runtime", required=True, type=Path)
    parser.add_argument("--treatment-runtime", required=True, type=Path)
    args = parser.parse_args()
    try:
        runtime = numeric_runtime(args.control_runtime, args.treatment_runtime)
        result = compare(args.control, args.treatment)
        result["numeric_runtime"] = runtime
    except (Ineligible, OSError, ValueError) as error:
        print(f"INELIGIBLE: {error}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Eligible matched comparison; quality verdict intentionally unassigned: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

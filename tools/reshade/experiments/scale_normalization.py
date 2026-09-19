# SPDX-License-Identifier: GPL-3.0-only
"""Reproducible distribution experiment; no game, shader or installed DLL edits.

Use the repository's designated evaluation Python for generate and analyze.
The companion C++ executable evaluates the actual scene_gain.h controller.
Units below are normalized ideal depth differences, NOT rendered pixel shifts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import PIL
import onnx


ROOT = Path(__file__).resolve().parents[3]
METHODS = ["center", "variance", "bounded_1.5", "bounded_2", "bounded_3", "effective_span", "scene_mean"]
SHAPE = (18, 32)
N = 576


def fingerprint():
    return {"executable": sys.executable, "python": sys.version,
            "numpy": np.__version__, "Pillow": PIL.__version__, "onnx": onnx.__version__}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ramp(low, high):
    return np.tile(np.linspace(low, high, 32, dtype=np.float32), (18, 1))


def planes(count=288, low=.125, high=.375):
    # Vertical-first ordering preserves a meaningful left-to-right silhouette.
    a = np.full((32, 18), low, dtype=np.float32)
    a.flat[:count] = high
    return a.T.copy()


def targets(a):
    x = a.astype(np.float64)
    center = x[7:11, 14:18].mean()
    sigma = x.std()
    span = x.max() - x.min()
    mad = np.mean(np.abs(x - x.mean()))
    ratio = 2 * sigma**2 / mad if mad else 0
    variance = .5 / sigma if sigma else 0
    return [1 / center if center else 0, variance,
            min(variance, 1.5 / span) if span else 0,
            min(variance, 2 / span) if span else 0,
            min(variance, 3 / span) if span else 0, 1 / ratio if ratio else 0,
            1 / x.mean() if x.mean() else 0]


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Use a fresh experiment output directory")
    cases, grids, provenance = [], [], []

    def add(sequence, time, group, phase, a, core=(.125, .375)):
        a = np.asarray(a, dtype=np.float32).reshape(SHAPE)
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError("Fixture is outside hardware raw depth domain")
        cases.append([len(grids), sequence, time, group, phase, *core])
        grids.append(a.copy())

    for count in range(1, N):
        add(f"coverage-{count}", 0, "coverage", str(count), planes(count))
    for start in range(29):
        a = np.full(SHAPE, .125, dtype=np.float32)
        a[:, start:start+4] = .375
        add(f"translation-{start}", 0, "translation", str(start), a)

    base = planes(low=.02, high=.08)
    for count in [0, 1, 2, 4, 6, 12, 24, 48]:
        a = base.copy()
        a.flat[:count] = 1
        add(f"outlier-{count}", 0, "outlier", str(count), a, (.02, .08))
    for amplitude in [.25, .05, .01, .001, .0001, .00001, 0]:
        a = ramp(.25-amplitude/2, .25+amplitude/2)
        add(f"slope-{amplitude}", 0, "slope", str(amplitude), a,
            (.25-amplitude/2, .25+amplitude/2))
    rng = np.random.default_rng(20260917)
    for amplitude in [.05, .001, .0001, .00001]:
        a = (.25 + rng.uniform(-amplitude, amplitude, SHAPE)).astype(np.float32)
        add(f"noise-{amplitude}", 0, "noise", str(amplitude), a)
    for count in [18, 72, 144, 288, 432, 558]:
        add(f"sky-{count}", 0, "sky", str(count), planes(count, 0, .25), (0, .25))
    # A foreground occupying the center over a nearly infinite background.
    # Unlike the .125/.375 fixture, no arbitrary far-depth floor hides mean gain.
    for half_w, half_h in [(2, 2), (3, 3), (4, 4), (7, 4), (8, 6)]:
        a = np.full(SHAPE, .0001, dtype=np.float32)
        a[9-half_h:9+half_h, 16-half_w:16+half_w] = .125
        count = 4*half_w*half_h
        add(f"far-background-{count}", 0, "far_background", str(count), a, (.0001, .125))
    add("flat", 0, "flat", "flat", np.full(SHAPE, .25))

    # Finite-far offset, unit scale, and precision stress; orientation is
    # already toward the viewer. Exact FP32 transformations are retained.
    for a, b in [(1, 0), (1, .4), (.5, .1), (.001, 0), (.000001, 0),
                 (.000001, .9), (.00000001, .9)]:
        transformed = (planes() * np.float32(a) + np.float32(b)).astype(np.float32)
        add(f"affine-{a}-{b}", 0, "affine", f"{a}:{b}", transformed,
            (float(np.float32(.125)*np.float32(a)+np.float32(b)),
             float(np.float32(.375)*np.float32(a)+np.float32(b))))

    # Controlled dynamic fixtures share exactly the same inputs, captures,
    # zero-plane-independent pair differences, and production temporal policy.
    for name in ["coverage-ramp", "center-crossing", "narrow-to-room", "room-wall-room",
                 "mild-narrow-to-room", "mild-room-wall-room",
                 "outlier-brief", "outlier-sustained", "room-step", "flat-hold",
                 "noise-hold", "sparse-object", "seed-low", "seed-high"]:
        for index, tick in enumerate(range(0, 40001, 250)):
            sec = tick / 1000
            a, phase, core = planes(), "reference", (.125, .375)
            if name == "coverage-ramp":
                count = round(288 - 282 * min(1, max(0, (sec-3)/12)))
                a, phase = planes(count), "shrinking" if sec >= 3 else "reference"
            elif name == "center-crossing":
                pos = min(28, max(0, int((sec-3)*2)))
                a = np.full(SHAPE, .125, dtype=np.float32)
                a[:, pos:pos+4] = .375
                phase = "moving-object"
            elif name.endswith("narrow-to-room") or name.endswith("room-wall-room"):
                narrow = sec < 8 if name.endswith("narrow-to-room") else 8 <= sec < 20
                if narrow:
                    half_span = .0005 if name.startswith("mild") else .00005
                    core = (.25-half_span, .25+half_span)
                    a, phase = ramp(*core), "narrow-wall"
                else:
                    a, phase = planes(), "room"
            elif name.startswith("outlier"):
                a, core = planes(low=.02, high=.08), (.02, .08)
                end = 8.5 if name == "outlier-brief" else 20
                if 8 <= sec < end:
                    a.flat[0] = 1
                    phase = "contaminant-or-thin-object"
                else:
                    phase = "clean"
            elif name == "room-step":
                if sec >= 8:
                    a, phase, core = planes(low=.0625, high=.1875), "new-scale", (.0625, .1875)
            elif name == "flat-hold" and 8 <= sec < 20:
                a, phase = np.full(SHAPE, .25, dtype=np.float32), "flat"
            elif name == "noise-hold" and 8 <= sec < 20:
                a = (.25 + rng.uniform(-.0001, .0001, SHAPE)).astype(np.float32)
                phase = "noise"
            elif name == "sparse-object":
                a, core = ramp(.02, .08), (.02, .08)
                if 8 <= sec < 20:
                    a.flat[:6] = .98
                    phase = "sparse-foreground"
            elif name.startswith("seed") and sec < 8:
                a = planes(low=.01, high=.03) if name == "seed-high" else planes(low=.25, high=.75)
                phase = "different-startup"
            add(name, tick, "dynamic", phase, a, core)

    # Replay earlier independent synthetic geometry traces, not old gains or
    # source/readiness state. No recorded real-game depth grids exist here.
    replay_dir = ROOT / "cmake-build-relwithdebinfo/game3d-online-gain-20260914/distribution-review-20260914"
    replay_names = ["walking-toward-fixed-surfaces", "wall-to-common", "vista-to-common",
                    "flat-interior", "sparse-sustained", "sparse-transient-aligned",
                    "affine-encoding-unreported"]
    for name in replay_names:
        table, binary = replay_dir / f"{name}.csv", replay_dir / f"{name}.grid.f32"
        if not table.exists() or not binary.exists():
            raise RuntimeError(f"Required historical fixture missing: {name}")
        data = binary.read_bytes()
        provenance.extend({"path": str(p), "sha256": digest(p)} for p in [table, binary])
        for row in csv.DictReader(table.open(newline="")):
            a = np.frombuffer(data, dtype="<f4", count=N, offset=int(row["grid_byte_offset"])).reshape(SHAPE)
            if int(row["normal"]):
                a = np.float32(1) - a
            add(f"replay-{name}", int(row["time_ms"]), "historical_synthetic",
                row["event"].replace(" ", "_") or "scene", a, (float(a.min()), float(a.max())))

    with (out / "cases.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "sequence", "time_ms", "group", "phase", "core_far", "core_near"])
        writer.writerows(cases)
    np.stack(grids).astype("<f4").tofile(out / "grids.f32")
    sources = [Path(__file__), Path(__file__).with_suffix(".cpp"),
               ROOT / "tools/reshade/scene_gain.h", ROOT / "tools/reshade/raw_reference_statistics.h",
               ROOT / "tools/reshade/depth_selection_policy.h"]
    snapshot = out / "source-snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    manifest = {"schema": 1, "runtime": fingerprint(), "grids": len(grids),
                "reference": "raw .125/.375, equal area, gain 4, normalized separation 1",
                "units": "ideal normalized depth-difference field; not actual pixel disparity",
                "methods": METHODS, "seed": 20260917, "historical_fixture_kind": "synthetic",
                "inputs": [{"path": str(p), "sha256": digest(p)} for p in sources],
                "replays": provenance,
                "cases_sha256": digest(out / "cases.tsv"), "grids_sha256": digest(out / "grids.f32")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(grids)} immutable grids; runtime identity and SHA256 recorded")


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["runtime"] != fingerprint():
        raise RuntimeError("Numeric runtime identity changed; evaluation invalid")
    for filename in ["cases", "grids"]:
        path = out / ("cases.tsv" if filename == "cases" else "grids.f32")
        if digest(path) != manifest[f"{filename}_sha256"]:
            raise RuntimeError("Input provenance mismatch")
    rows = list(csv.DictReader((out / "results.csv").open(newline="")))
    cases = list(csv.DictReader((out / "cases.tsv").open(newline=""), delimiter="\t"))
    grids = np.fromfile(out / "grids.f32", dtype="<f4").reshape(-1, *SHAPE)
    if len(rows) != len(grids)*len(METHODS)*2:
        raise RuntimeError("Missing experiment results")
    # Independent two-pass NumPy check against C++ Welford implementation.
    expected = [targets(a) for a in grids]
    worst_error = 0
    for row in rows:
        actual = float(row["target"])
        want = expected[int(row["id"])][METHODS.index(row["method"])]
        error = abs(actual-want) / max(1, abs(want))
        worst_error = max(worst_error, error)
        if error > 2e-7:
            raise AssertionError(f"Independent formula mismatch: {row['sequence']}: {actual} vs {want}")
    index = {}
    for row in rows:
        index.setdefault((row["sequence"], row["method"], row["gate"]), []).append(row)

    def series(sequence, method, gate="existing_content"):
        return index[(sequence, method, gate)]

    def value(sequence, method, key="target"):
        return float(series(sequence, method)[0][key])

    summary = {"runtime": fingerprint(), "input_grids": len(grids), "result_rows": len(rows),
               "independent_formula_relative_error": worst_error, "methods": {}}
    for method in METHODS:
        reference = value("coverage-288", method)
        if not math.isclose(reference, 4, abs_tol=1e-12):
            raise AssertionError("Candidates were not matched at reference strength")
        cover = [value(f"coverage-{i}", method)*.25 for i in range(1, N)]
        translation = [value(f"translation-{i}", method) for i in range(29)]
        clean = value("outlier-0", method)
        dirty = value("outlier-1", method)
        affine_ref = value("affine-1-0", method)
        offset = value("affine-1-0.4", method)
        narrow = series("narrow-to-room", method)
        first_room = next(r for r in narrow if r["phase"] == "room")
        mild_room = next(r for r in series("mild-narrow-to-room", method) if r["phase"] == "room")
        room_wall = series("mild-room-wall-room", method)
        room_return = next(r for r in room_wall if int(r["time_ms"]) >= 20000)
        room_step = series("room-step", method)
        final_target = float(room_step[-1]["target"])
        settling = next((int(r["time_ms"])/1000-8 for r in room_step
                         if int(r["time_ms"]) >= 8000 and
                         abs(float(r["current"])/final_target-1) <= .1), None)
        # Check current-to-current rate, excluding initialization and rearming.
        maximum_log_speed = 0
        for key, sequence in index.items():
            if key[1] != method:
                continue
            for old, new in zip(sequence, sequence[1:]):
                a, b = float(old["current"]), float(new["current"])
                dt = (int(new["time_ms"])-int(old["time_ms"]))/1000
                if a > 0 and b > 0 and dt > 0:
                    maximum_log_speed = max(maximum_log_speed, abs(math.log(b/a))/dt)
        if maximum_log_speed > math.log(2)+1e-6:
            raise AssertionError("Actual controller rate bound exceeded")
        summary["methods"][method] = {
            "coverage_max_over_min": max(cover)/min(cover),
            "coverage_6_cells_pair_separation": value("coverage-6", method)*.25,
            "coverage_1_cell_pair_separation": value("coverage-1", method)*.25,
            "translation_max_over_min_gain": max(translation)/min(translation),
            "one_outlier_core_strength_remaining": dirty/clean,
            "one_outlier_total_span": dirty*.98,
            "offset_same_scene_strength_ratio": offset/affine_ref,
            "narrow_slope_gain": value("slope-0.0001", method),
            "narrow_content_kind": int(series("slope-0.0001", method)[0]["content_kind"]),
            "narrow_to_room_first_pair_separation": float(first_room["core_separation"]),
            "narrow_to_room_first_gain": float(first_room["current"]),
            "mild_narrow_to_room_first_pair_separation": float(mild_room["core_separation"]),
            "mild_room_wall_room_first_return_pair_separation": float(room_return["core_separation"]),
            "far_background_112_pair_separation": value("far-background-112", method)*(.125-.0001),
            "far_background_112_content_kind": int(series("far-background-112", method)[0]["content_kind"]),
            "far_background_16_pair_separation": value("far-background-16", method)*(.125-.0001),
            "room_step_time_to_10_percent_seconds": settling,
            "maximum_observed_log_speed": maximum_log_speed,
        }
    # Affirm mathematical invariants without treating them as quality verdicts.
    assert summary["methods"]["effective_span"]["coverage_max_over_min"] < 1+1e-9
    assert summary["methods"]["effective_span"]["translation_max_over_min_gain"] < 1+1e-9
    for method in METHODS[1:6]:
        assert abs(summary["methods"][method]["offset_same_scene_strength_ratio"]-1) < 1e-6
    summary["source_hashes_unchanged"] = all(digest(Path(p["path"])) == p["sha256"] for p in manifest["inputs"])
    if not summary["source_hashes_unchanged"]:
        raise RuntimeError("Experiment source changed during evaluation")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "analyze"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (generate if args.action == "generate" else analyze)(args.output.resolve())

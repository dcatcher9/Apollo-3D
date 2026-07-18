#!/usr/bin/env python3
"""Profile metric computation from preserved SBS benchmark artifacts.

This intentionally skips TensorRT/harness execution and report rendering so metric-code
optimizations can be measured without regenerating a run.  It is a development tool, not part of
the evaluator contract or metric hash.
"""

import argparse
import cProfile
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import sbsbench  # noqa: E402
import run_eval  # noqa: E402


def source_revision():
    """Best-effort repository revision for non-authoritative timing provenance."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR.parent.parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="preserved sbs_eval run")
    parser.add_argument("--clips-root", type=Path, default=SCRIPT_DIR / "clips")
    parser.add_argument("--clips", nargs="*", help="clip names (default: every run directory)")
    parser.add_argument("--workers", type=int, default=1,
                        help="spatial worker count; use 1 for complete cProfile attribution")
    parser.add_argument("--stats", type=Path, help="optional cProfile output")
    parser.add_argument("--top", type=int, default=40, help="number of profile rows to print")
    parser.add_argument("--json", type=Path, help="optional machine-readable timing summary")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.workers <= sbsbench.SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS:
        raise SystemExit(
            "--workers must be from 1 to "
            f"{sbsbench.SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS}")
    os.environ[sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV] = str(args.workers)
    clip_names = args.clips or sorted(
        entry.name for entry in args.run_dir.iterdir()
        if entry.is_dir() and (entry / "contract.json").exists())
    if not clip_names:
        raise SystemExit(f"no clip artifacts found under {args.run_dir}")

    sbsbench.enable_reusable_spatial_executor()
    profiler = cProfile.Profile()
    timings = []
    profiler.enable()
    started = time.perf_counter()
    for clip_name in clip_names:
        clip_start = time.perf_counter()
        rows, _aggregate = sbsbench.measure_sequence(
            str(args.run_dir / clip_name), str(args.clips_root / clip_name))
        elapsed = time.perf_counter() - clip_start
        timings.append({"clip": clip_name, "frames": len(rows), "seconds": elapsed})
        print(f"{clip_name}: {len(rows)} frames in {elapsed:.3f}s", flush=True)
    total = time.perf_counter() - started
    profiler.disable()

    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.stats)
    pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(args.top)
    summary = {
        "schema": 1,
        "workers": args.workers,
        "backend": os.environ.get(sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV, "process"),
        "pixel_budget_mpx": float(os.environ.get(
            sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV,
            sbsbench.SEQUENCE_SPATIAL_DEFAULT_PIXEL_BUDGET_MPX)),
        "metric_sha256": run_eval.metric_contract_sha(),
        "metric_runtime": run_eval.metric_runtime_provenance(),
        "source_revision": source_revision(),
        "platform": platform.platform(),
        "logical_processors": os.cpu_count(),
        "clips": timings,
        "frames": sum(item["frames"] for item in timings),
        "seconds": total,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

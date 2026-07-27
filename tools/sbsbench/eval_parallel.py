"""Deterministic clip-level orchestration for SBS metric evaluation.

This module is deliberately outside the metric-contract hash: it changes scheduling, not metric
math. The production GPU harness remains serial in ``run_eval.py``. These outer threads only
coordinate CPU scoring, while ``sbsbench`` retains its bounded process pool for frame-level work.
"""

import concurrent.futures
import contextlib
import json
import math
import os
import threading

import sbsbench
from PIL import Image


CLIP_SCORING_MAX_WORKERS = 16


def _configured_pixel_budget_pixels():
    """Return the run-global packed-image allowance in pixels."""
    configured = os.environ.get(
        sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV,
        str(sbsbench.SEQUENCE_SPATIAL_DEFAULT_PIXEL_BUDGET_MPX))
    try:
        budget_mpx = float(configured)
    except ValueError as exc:
        raise ValueError(
            f"{sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV} must be positive") from exc
    if not math.isfinite(budget_mpx) or budget_mpx <= 0.0:
        raise ValueError(
            f"{sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV} must be positive")
    return max(1, int(budget_mpx * 1_000_000))


def _clip_spatial_reservation(job, budget_pixels):
    """Return this clip's peak packed-image demand under the inner worker policy."""
    clip, seq_dir, _ = job
    sbs_files = sbsbench.indexed_files(
        os.path.join(seq_dir, "sbs_*.png"), "sbs_")
    if not sbs_files:
        # Measurement will provide the clip-local missing-artifact error. Reserve one token so a
        # malformed clip still follows the ordinary fail-closed path without blocking peers.
        return 1
    max_pixels = 0
    for path in sbs_files.values():
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"clip scoring failed for {clip!r}: cannot inspect {path}: {exc}") from exc
        max_pixels = max(max_pixels, width * height)
    # Mirror the worker-count input used by ``_measure_sequence_spatial_rows``. Most harness
    # artifacts publish packed dimensions in warp_map_shape.json; sequences without a map pass
    # ``None`` and therefore use the inner pool's ordinary CPU cap. Assuming the image size here
    # when the worker later assumes ``None`` would under-reserve exactly the legacy/no-map case.
    packed_pixels = None
    shape_path = os.path.join(seq_dir, "warp_map_shape.json")
    if os.path.exists(shape_path):
        try:
            with open(shape_path, encoding="utf-8") as shape_file:
                shape = json.load(shape_file)
            width, height = shape.get("width"), shape.get("height")
            if (isinstance(width, bool) or not isinstance(width, int) or width < 1 or
                    isinstance(height, bool) or not isinstance(height, int) or height < 1):
                raise ValueError("width and height must be positive integers")
            packed_pixels = width * height
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"clip scoring failed for {clip!r}: cannot inspect {shape_path}: {exc}") from exc
    workers = sbsbench._sequence_spatial_worker_count(
        len(sbs_files), packed_pixels)
    per_worker_pixels = max(max_pixels, packed_pixels or 0)
    demand = per_worker_pixels * max(1, workers)
    if per_worker_pixels > budget_pixels:
        raise ValueError(
            f"{clip!r} has a {per_worker_pixels / 1_000_000:.2f} MP packed frame above the "
            f"run-global {budget_pixels / 1_000_000:.2f} MP allowance; increase "
            f"{sbsbench.SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV}")
    if demand > budget_pixels:
        worker_source = (
            f"{sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV} requests"
            if os.environ.get(sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV) is not None
            else "the inner scorer selects"
        )
        raise ValueError(
            f"{worker_source} {workers} workers for "
            f"{clip!r}, requiring {demand / 1_000_000:.2f} MP above the run-global "
            f"{budget_pixels / 1_000_000:.2f} MP allowance")
    return demand


class _WeightedPixelBudget:
    """Thread-safe weighted semaphore for packed-image scoring work."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("pixel-budget capacity must be a positive integer")
        self.capacity = capacity
        self.in_use = 0
        self.condition = threading.Condition()

    @contextlib.contextmanager
    def reserve(self, weight):
        if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= self.capacity:
            raise ValueError("pixel-budget reservation must fit the configured capacity")
        with self.condition:
            while self.in_use + weight > self.capacity:
                self.condition.wait()
            self.in_use += weight
        try:
            yield
        finally:
            with self.condition:
                self.in_use -= weight
                self.condition.notify_all()


def _measure_reserved_clip(job, reservation, limiter):
    with limiter.reserve(reservation):
        return _measure_clip_sequence_job(job)


def clip_scoring_worker_count(requested, clip_count):
    """Validate and bound the number of ordered clip-scoring threads."""
    if (isinstance(requested, bool) or not isinstance(requested, int)
            or not 1 <= requested <= CLIP_SCORING_MAX_WORKERS):
        raise ValueError(
            "clip scoring jobs must be an integer from 1 to "
            f"{CLIP_SCORING_MAX_WORKERS}")
    if (isinstance(clip_count, bool) or not isinstance(clip_count, int) or clip_count < 0):
        raise ValueError("clip count must be a non-negative integer")
    return min(requested, clip_count) if clip_count else 0


def _measure_clip_sequence_job(job):
    """Measure one named clip and attach its identity to every worker-side failure."""
    clip, seq_dir, frames_dir = job
    try:
        measured = sbsbench.measure_sequence(seq_dir, frames_dir, compact=True)
    except Exception as exc:
        raise RuntimeError(f"clip scoring failed for {clip!r}: {exc}") from exc
    if measured is None:
        raise RuntimeError(f"clip scoring failed for {clip!r}: no measurable SBS artifacts")
    return clip, measured


def measure_clip_sequences(sequence_jobs, jobs=1):
    """Measure independent clips concurrently and return results in input order.

    ``sequence_jobs`` contains ``(clip_id, artifact_dir, source_frames_dir)`` tuples. The shared
    frame-level process executor remains the only pool running metric kernels. A run-global
    weighted pixel semaphore bounds all concurrently submitted clip batches together; the
    per-sequence worker calculation alone is not a global memory bound. Results retain input order,
    so failures and returned rows remain deterministic when workers finish out of order.
    """
    prepared = []
    clip_ids = set()
    for index, job in enumerate(sequence_jobs):
        if not isinstance(job, (tuple, list)) or len(job) != 3:
            raise ValueError(
                f"clip scoring job {index} must be (clip_id, artifact_dir, source_frames_dir)")
        clip, seq_dir, frames_dir = job
        if not isinstance(clip, str) or not clip:
            raise ValueError(f"clip scoring job {index} has an invalid clip identity")
        if clip in clip_ids:
            raise ValueError(f"duplicate clip scoring job: {clip!r}")
        clip_ids.add(clip)
        prepared.append((clip, seq_dir, frames_dir))

    worker_count = clip_scoring_worker_count(jobs, len(prepared))
    if not prepared:
        return []
    budget_pixels = _configured_pixel_budget_pixels()
    reservations = [
        _clip_spatial_reservation(job, budget_pixels) for job in prepared
    ]
    limiter = _WeightedPixelBudget(budget_pixels)
    if worker_count == 1:
        return [
            _measure_reserved_clip(job, reservation, limiter)
            for job, reservation in zip(prepared, reservations)
        ]

    # Create the process pool before worker threads race to submit their independent frame maps.
    sbsbench.enable_reusable_spatial_executor()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="sbsbench-clip") as executor:
        futures = [
            executor.submit(_measure_reserved_clip, job, reservation, limiter)
            for job, reservation in zip(prepared, reservations)
        ]
        # Consume in source order. A later job can execute first, but it cannot change result or
        # failure association.
        return [future.result() for future in futures]

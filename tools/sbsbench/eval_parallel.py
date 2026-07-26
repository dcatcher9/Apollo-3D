"""Deterministic clip-level orchestration for SBS metric evaluation.

This module is deliberately outside the metric-contract hash: it changes scheduling, not metric
math. The production GPU harness remains serial in ``run_eval.py``. These outer threads only
coordinate CPU scoring, while ``sbsbench`` retains its bounded process pool for frame-level work.
"""

import concurrent.futures

import sbsbench


CLIP_SCORING_MAX_WORKERS = 16


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
    frame-level process executor remains the only pool running metric kernels, globally bounding
    active CPU work even when several clip threads submit frames. ``Executor.map`` preserves input
    order, so failures and returned results remain deterministic when workers finish out of order.
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
    if worker_count == 1:
        return [_measure_clip_sequence_job(job) for job in prepared]

    # Create the process pool before worker threads race to submit their independent frame maps.
    sbsbench.enable_reusable_spatial_executor()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="sbsbench-clip") as executor:
        return list(executor.map(_measure_clip_sequence_job, prepared))

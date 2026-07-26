#!/usr/bin/env python3
"""Windows-safe parallel entry point for the authenticated SBS HTML report.

``build_report.py`` intentionally contains the report implementation at module scope so its pure
helpers can be inspected by focused tests. Import it only below this guarded entry point: Windows
process-pool children re-run this file as ``__mp_main__``, skip the guard, and can then execute
sbsbench workers without recursively constructing the report.
"""
import argparse
import os
import sys


REPORT_SCORING_JOBS_ENV = "SBSBENCH_REPORT_SCORING_JOBS"
REPORT_SCORING_MAX_JOBS = 16  # Keep aligned with eval_parallel.CLIP_SCORING_MAX_WORKERS.


def _positive_job_count(value):
    try:
        jobs = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if not 1 <= jobs <= REPORT_SCORING_MAX_JOBS:
        raise argparse.ArgumentTypeError(
            f"must be from 1 to {REPORT_SCORING_MAX_JOBS}")
    return jobs


def report_scoring_options(argv):
    """Return verification concurrency and report arguments with that option removed."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scoring-jobs", type=_positive_job_count)
    options, remaining = parser.parse_known_args(argv)
    jobs = (options.scoring_jobs if options.scoring_jobs is not None else
            min(8, os.cpu_count() or 1, REPORT_SCORING_MAX_JOBS))
    return jobs, remaining


def report_scoring_jobs(argv):
    """Read verification concurrency without exposing parser details to callers/tests."""
    return report_scoring_options(argv)[0]


def main():
    scoring_jobs, report_arguments = report_scoring_options(sys.argv[1:])
    # build_report executes when imported below. Use a report-only environment handoff so the
    # guarded Windows launcher resolves and removes the option before the top-level report module
    # binds its three positional paths.
    os.environ[REPORT_SCORING_JOBS_ENV] = str(scoring_jobs)
    os.environ["SBSBENCH_SPATIAL_BACKEND"] = "process"
    sys.argv = [sys.argv[0], *report_arguments]
    # Import executes the implementation after argv has already been supplied to this launcher.
    import build_report  # noqa: F401,E402


if __name__ == "__main__":
    main()

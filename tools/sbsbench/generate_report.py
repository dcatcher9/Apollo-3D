#!/usr/bin/env python3
"""Windows-safe parallel entry point for the authenticated SBS HTML report.

``build_report.py`` intentionally contains the report implementation at module scope so its pure
helpers can be inspected by focused tests. Import it only below this guarded entry point: Windows
process-pool children re-run this file as ``__mp_main__``, skip the guard, and can then execute
sbsbench workers without recursively constructing the report.
"""
import os


def main():
    os.environ["SBSBENCH_SPATIAL_BACKEND"] = "process"
    # Import executes the implementation after argv has already been supplied to this launcher.
    import build_report  # noqa: F401,E402


if __name__ == "__main__":
    main()

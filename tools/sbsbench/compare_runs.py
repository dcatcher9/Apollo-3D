"""Compare two eval runs metric-by-metric, split by clip content type.

Why this exists. Reading a run pair by hand -- loading both results.json and averaging a metric
across the suite -- produced two separate wrong conclusions during the cardboarding work:

  * A metric was averaged over whichever clips happened to carry it IN EACH RUN INDEPENDENTLY, so
    the two means covered different clip subsets and the delta was meaningless.
  * A suite mean was quoted as a result when it was produced almost entirely by synthetic probe
    clips, which exist to expose a specific failure mode and are not representative content.
    `static_jitter_p95` reads +47% across core and +0.10% on the five c*-named clips in the legacy
    grouping; the suite mean is the number that is misleading, not the split. Those clips are now
    classified as AI-generated or unclassified, never retroactively asserted to be real captures.

This tool removes both footguns: it intersects on clips where BOTH runs have a valid, applicable
value, and it never prints a single suite mean without the per-content-type breakdown beside it.

Usage:
    python tools/sbsbench/compare_runs.py <control-run-dir> <treatment-run-dir> [--metrics m1,m2]
"""
import argparse
import importlib.util
import json
import pathlib
import re
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sbsbench = _load_module("sbsbench")
run_eval = _load_module("run_eval")

# Content that cannot support the classified-content aggregate. Constructed probes are legitimate
# diagnostics, while ``unclassified`` is deliberately fail-closed until a reviewer assigns a
# provenance-backed content type.
NON_DECISIVE = {
    "synthetic": "constructed probe",
    "unclassified": "unclassified content",
}


def load(run_dir):
    path = pathlib.Path(run_dir)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def content_of(clip_entry):
    return (clip_entry.get("meta") or {}).get("content_type") or "unclassified"


def usable(metric, spec, entry):
    """Applicable evidence AND a finite value -- the pair a delta may be computed from."""
    agg = entry["aggregate"]
    meta = entry.get("meta") or {}
    if sbsbench.metric_evidence_state(metric, spec, agg, meta) != "applicable":
        return False
    return sbsbench.metric_value_valid(agg, metric)


def compatibility_error(control, treatment):
    """Return why two runs cannot be compared under the current evaluator contract."""
    runs = (("control", control), ("treatment", treatment))
    valid_suites = {"core", "extended"}
    valid_run_kinds = {"baseline-gated", "baseline-update", "comparison-only"}
    current = {
        "eval_schema": run_eval.EVAL_SCHEMA,
        "metric_sha256": run_eval.metric_contract_sha(),
        "label_contract_sha256": run_eval.label_contract_sha(),
        "metric_runtime": run_eval.metric_runtime_provenance(),
    }
    for side, run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("meta"), dict):
            return f"{side} run has no metadata object"
        if not isinstance(run.get("clips"), dict):
            return f"{side} run has no clips object"
        if run["meta"].get("mode") != "canonical-v2":
            return f"{side} mode must be 'canonical-v2', got {run['meta'].get('mode')!r}"
        if run["meta"].get("suite") not in valid_suites:
            return (f"{side} suite must be one of {sorted(valid_suites)!r}, "
                    f"got {run['meta'].get('suite')!r}")
        if run["meta"].get("run_kind") not in valid_run_kinds:
            return (f"{side} run_kind must be one of {sorted(valid_run_kinds)!r}, "
                    f"got {run['meta'].get('run_kind')!r}")
        for key, expected in current.items():
            observed = run["meta"].get(key)
            if observed != expected:
                return (f"{side} {key} is stale or incompatible "
                        f"({observed!r}; current {expected!r})")

    # These are evidence-contract fields, not treatment levers. Model/config/executable/depth-step
    # differences remain valid A/B dimensions for this textual comparator.
    for key in ("clip_set_sha1", "eval_schema", "suite", "mode", "run_kind",
                "metric_sha256", "label_contract_sha256", "metric_runtime"):
        left = control["meta"].get(key)
        right = treatment["meta"].get(key)
        if left != right:
            return f"{key} differs between runs ({left!r} vs {right!r})"

    for side, run in runs:
        clip_hashes = run["meta"].get("clip_set_sha1")
        if not isinstance(clip_hashes, dict):
            return f"{side} clip_set_sha1 is not an object"
        invalid_hashes = sorted(
            clip for clip, digest in clip_hashes.items()
            if not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{12}", digest) is None
        )
        if invalid_hashes:
            return (f"{side} clip_set_sha1 has invalid lowercase 12-hex digest(s) "
                    f"for {invalid_hashes!r}")
        if set(run["clips"]) != set(clip_hashes):
            return (f"{side} clips do not match clip_set_sha1 "
                    f"({sorted(run['clips'])!r} vs {sorted(clip_hashes)!r})")
        for clip, entry in run["clips"].items():
            if not isinstance(entry, dict):
                return f"{side} clip {clip!r} is not an object"
            if not isinstance(entry.get("meta"), dict):
                return f"{side} clip {clip!r} has no metadata object"
            if not isinstance(entry.get("aggregate"), dict):
                return f"{side} clip {clip!r} has no aggregate object"

    for clip in sorted(control["clips"]):
        left = control["clips"][clip]["meta"].get("content_type")
        right = treatment["clips"][clip]["meta"].get("content_type")
        if not isinstance(left, str) or not left.strip():
            return f"control clip {clip!r} has no explicit content_type classification"
        if not isinstance(right, str) or not right.strip():
            return f"treatment clip {clip!r} has no explicit content_type classification"
        if left not in run_eval.CONTENT_TYPES:
            return (f"control clip {clip!r} content_type {left!r} is not one of "
                    f"{sorted(run_eval.CONTENT_TYPES)!r}")
        if right not in run_eval.CONTENT_TYPES:
            return (f"treatment clip {clip!r} content_type {right!r} is not one of "
                    f"{sorted(run_eval.CONTENT_TYPES)!r}")
        if left != right:
            return (f"content_type differs for clip {clip!r} "
                    f"({left!r} vs {right!r})")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--metrics", help="comma-separated subset (default: every gated/diagnostic metric)")
    ap.add_argument("--all-roles", action="store_true",
                    help="include diagnostic metrics too (default: primary and hard only)")
    args = ap.parse_args()

    ctrl, treat = load(args.control), load(args.treatment)
    thresholds = json.loads((_HERE / "thresholds.json").read_text(encoding="utf-8"))
    specs = {k: v for k, v in thresholds["metrics"].items() if isinstance(v, dict)}

    incompatible = compatibility_error(ctrl, treat)
    if incompatible:
        print(f"REFUSING: {incompatible}", file=sys.stderr)
        return 2

    cc, tc = ctrl["clips"], treat["clips"]
    clips = sorted(cc)
    if not clips:
        print("no clips in common", file=sys.stderr)
        return 2

    groups = {}
    for c in clips:
        groups.setdefault(content_of(cc[c]), []).append(c)

    wanted = ([m.strip() for m in args.metrics.split(",")] if args.metrics else
              [m for m, s in specs.items()
               if args.all_roles or s.get("role") in ("primary", "hard")])
    if args.metrics:
        unknown = sorted({metric for metric in wanted if not metric or metric not in specs})
        if unknown:
            print("REFUSING: unknown requested metric(s): " + ", ".join(
                repr(metric) for metric in unknown), file=sys.stderr)
            return 2

    print(f"control  : {args.control}")
    print(f"treatment: {args.treatment}")
    print(f"suite    : {ctrl['meta'].get('suite')}  schema {ctrl['meta'].get('eval_schema')}")
    print("groups   : " + ", ".join(f"{g}({len(v)})" for g, v in sorted(groups.items())))
    print()

    for metric in wanted:
        spec = specs.get(metric)
        if spec is None:
            continue
        pairs = [(c, cc[c]["aggregate"][metric], tc[c]["aggregate"][metric])
                 for c in clips
                 if usable(metric, spec, cc[c]) and usable(metric, spec, tc[c])]
        if not pairs:
            continue
        better = spec.get("better")
        skipped = len(clips) - len(pairs)
        head = f"{metric}  [{spec.get('role')}, better={better}]"
        if skipped:
            head += f"   ({skipped} clip(s) without applicable evidence in both runs)"
        print(head)

        decisive_rows = []
        for group in sorted(groups):
            rows = [(c, a, b) for c, a, b in pairs if content_of(cc[c]) == group]
            if not rows:
                continue
            ma = statistics.mean(r[1] for r in rows)
            mb = statistics.mean(r[2] for r in rows)
            delta = (mb - ma) / abs(ma) * 100.0 if ma else float("nan")
            reason = NON_DECISIVE.get(group)
            tag = f"  <- NOT DECISIVE ({reason})" if reason else ""
            print(f"    {group:<16} n={len(rows):<3} {ma:11.4f} -> {mb:11.4f}  {delta:+8.2f}%{tag}")
            if group not in NON_DECISIVE:
                decisive_rows += rows

        if decisive_rows:
            ma = statistics.mean(r[1] for r in decisive_rows)
            mb = statistics.mean(r[2] for r in decisive_rows)
            delta = (mb - ma) / abs(ma) * 100.0 if ma else float("nan")
            print(f"    {'ALL CLASSIFIED NON-PROBE':<16} n={len(decisive_rows):<3} "
                  f"{ma:11.4f} -> {mb:11.4f}  {delta:+8.2f}%")
            worst = max(decisive_rows,
                        key=lambda r: ((r[2] - r[1]) if better == "lower" else (r[1] - r[2]))
                        / max(abs(r[1]), 1e-9))
            wd = (worst[2] - worst[1]) / abs(worst[1]) * 100.0 if worst[1] else float("nan")
            print(f"    worst clip: {worst[0]}  {worst[1]:.4f} -> {worst[2]:.4f}  {wd:+.2f}%")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

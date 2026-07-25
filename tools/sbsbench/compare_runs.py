"""Compare two eval runs metric-by-metric, split by clip content type.

Why this exists. Reading a run pair by hand -- loading both results.json and averaging a metric
across the suite -- produced two separate wrong conclusions during the cardboarding work:

  * A metric was averaged over whichever clips happened to carry it IN EACH RUN INDEPENDENTLY, so
    the two means covered different clip subsets and the delta was meaningless.
  * A suite mean was quoted as a result when it was produced almost entirely by synthetic probe
    clips, which exist to expose a specific failure mode and are not representative content.
    `static_jitter_p95` reads +47% across core and +0.10% on real captures; the suite mean is the
    number that is misleading, not the split.

This tool removes both footguns: it intersects on clips where BOTH runs have a valid, applicable
value, and it never prints a single suite mean without the per-content-type breakdown beside it.

Usage:
    python tools/sbsbench/compare_runs.py <control-run-dir> <treatment-run-dir> [--metrics m1,m2]
"""
import argparse
import importlib.util
import json
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sbsbench = _load_module("sbsbench")

# Clips whose content is constructed to expose one failure mode. They are legitimate probes, but a
# mean that mixes them with real content is not a statement about real content.
NON_DECISIVE = {"synthetic"}


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

    for key in ("clip_set_sha1", "eval_schema", "suite"):
        a, b = ctrl["meta"].get(key), treat["meta"].get(key)
        if a != b:
            print(f"REFUSING: {key} differs between runs ({a!r} vs {b!r})", file=sys.stderr)
            return 2

    cc, tc = ctrl["clips"], treat["clips"]
    clips = sorted(set(cc) & set(tc))
    if not clips:
        print("no clips in common", file=sys.stderr)
        return 2

    groups = {}
    for c in clips:
        groups.setdefault(content_of(cc[c]), []).append(c)

    wanted = ([m.strip() for m in args.metrics.split(",")] if args.metrics else
              [m for m, s in specs.items()
               if args.all_roles or s.get("role") in ("primary", "hard")])

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
            tag = "  <- NOT DECISIVE (constructed probe)" if group in NON_DECISIVE else ""
            print(f"    {group:<16} n={len(rows):<3} {ma:11.4f} -> {mb:11.4f}  {delta:+8.2f}%{tag}")
            if group not in NON_DECISIVE:
                decisive_rows += rows

        if decisive_rows:
            ma = statistics.mean(r[1] for r in decisive_rows)
            mb = statistics.mean(r[2] for r in decisive_rows)
            delta = (mb - ma) / abs(ma) * 100.0 if ma else float("nan")
            print(f"    {'ALL NON-SYNTHETIC':<16} n={len(decisive_rows):<3} "
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

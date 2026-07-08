#!/usr/bin/env python3
"""Assemble the SBS A/B report directly from two run_eval.py runs (control + treatment):
a control-vs-treatment scorecard (row per clip, auto-discovered), the gate's verdict, and one
section per triggered issue with control/treatment crops at each issue's WORST frame.

Usage: build_report.py <control_run_dir> <treat_run_dir> <out.html>
       (run dirs = <build-dir>/sbs_eval/<label>/ containing results.json + <clip>/sbs_*.png)
"""
import base64
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

ctrl_dir, treat_dir, out_html = sys.argv[1], sys.argv[2], sys.argv[3]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CTRL = json.load(open(os.path.join(ctrl_dir, "results.json")))
TREAT = json.load(open(os.path.join(treat_dir, "results.json")))
THR = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))["metrics"]
CLIPS = sorted(CTRL["clips"])

# metric, header, worse-is-higher, always-show, notable-threshold
COLS = [
    ("pop_px_p50", "pop", False, True, 0), ("edge_acc_p50", "edge_acc", True, False, 2.0),
    ("stretch_area", "stretch", True, False, 2.0), ("rim_over_p95", "rim", True, False, 1.0),
    ("swim_p50", "swim", True, False, 1.0), ("flicker_p50", "flick", True, True, 0),
    ("flicker_disocc_p50", "flick_dis", True, True, 0), ("vmisalign_px", "vmis", True, False, 0.5),
    ("disocc_smear", "smear", True, False, 0.02),
]
SHORT = {k: h for k, h, *_ in COLS}
ISSUE_DEFS = {  # metric -> (title, temporal?, description)
    "stretch_area": ("Disocclusion stretch band", False,
                     "Background rubber-banded horizontally to fill the gap the foreground "
                     "uncovered — eye-asymmetric (left eye smears left, right eye right)."),
    "rim_over_p95": ("Silhouette white line", False,
                     "A thin bright fringe hugging the silhouette — the residual sliver where "
                     "the fill doesn't reach the foreground edge."),
    "edge_acc_p50": ("Soft / floating silhouettes", False,
                     "The depth silhouette sits off the true object edge, so the cut-out is "
                     "loosely placed."),
    "disocc_smear": ("Disocclusion fill blur", False,
                     "Horizontal-detail deficit in the band beside silhouettes — on flat/synthetic "
                     "content this also fingerprints hallucinated depth edges."),
    "flicker_disocc_p50": ("Disocclusion shimmer", True,
                           "The fill re-hallucinates frame to frame in the disocclusion bands — "
                           "boiling along edges. Temporal: numbers, not stills."),
    "vmisalign_px": ("Vertical misalignment", False,
                     "Geometry fault: parallax must be horizontal-only."),
}


def durl(im, w=None, jpg=False):
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    if jpg:
        im.convert("RGB").save(b, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def load_depth(p):
    im = Image.open(p)
    a = np.asarray(im).astype(np.float32)
    if im.mode in ("I;16", "I;16B", "I"):
        return a / 65535.0
    if a.ndim == 3:
        a = a[..., 0]
    return a / 255.0


def frame_path(run, clip, i):
    return os.path.join(run, clip, f"sbs_{i:05d}.png")


def mid_frame(run, clip):
    n = len(glob.glob(os.path.join(run, clip, "sbs_*.png")))
    return max(0, n // 2)


def thumb(clip):
    im = Image.open(frame_path(ctrl_dir, clip, mid_frame(ctrl_dir, clip)))
    left = im.crop((0, 0, im.width // 2, im.height))
    left.thumbnail((132, 264), Image.LANCZOS)
    return durl(left)


def crop_at_silhouette(clip, idx):
    """Control/treatment left-eye crops at the strongest depth silhouette of frame idx (falls
    back to center if the depth is flat). Returns (ctrl_durl, treat_durl) or None."""
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    dp = os.path.join(ctrl_dir, clip, f"depth_{idx:05d}.png")
    if not (os.path.exists(cp) and os.path.exists(tp) and os.path.exists(dp)):
        return None
    depth = load_depth(dp)
    sbs_c, sbs_t = Image.open(cp), Image.open(tp)
    ew, eh = sbs_c.width // 2, sbs_c.height
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    dh, dw = depth.shape
    band = gx[int(dh * 0.15):int(dh * 0.85)]
    colscore = band.sum(0)
    lo, hi = int(dw * 0.1), int(dw * 0.9)
    cx_d = int(np.argmax(colscore[lo:hi]) + lo) if colscore[lo:hi].max() > 0.1 else dw // 2
    rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
    cy_d = int(np.argmax(rowscore)) if rowscore.max() > 0 else dh // 2
    cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, cx - cw // 2))
    y0 = max(0, min(eh - ch, cy - ch // 2))
    out = []
    for img in (sbs_c, sbs_t):
        # Native-resolution crop as JPEG; the page CSS scales it up (browsers interpolate fine,
        # and a 2x-nearest PNG here balloons the report size).
        out.append(durl(img.crop((x0, y0, x0 + cw, y0 + ch)), jpg=True))
    return out


def run_label(run, default):
    """Human name for a run: its extra harness args, else 'mode (model)', else a default."""
    ex = run["meta"].get("extra_args") or []
    if ex:
        return " ".join(ex).replace("--", "")
    mode = run["meta"].get("mode", "")
    models = sorted({e["meta"].get("model", "") for e in run["clips"].values()})
    if mode:
        return f"{mode}" + (f" ({models[0]})" if len(models) == 1 and models[0] else "")
    return default


CTRL_MODE = CTRL["meta"].get("mode")
TREAT_MODE = TREAT["meta"].get("mode")
IS_MODE_CMP = bool(CTRL_MODE and TREAT_MODE and CTRL_MODE != TREAT_MODE)
CTRL_NAME = run_label(CTRL, "control")
TREAT_NAME = run_label(TREAT, "treatment")
# Short tags for inline value labels and image captions (arrow is always CTRL -> TREAT).
CTRL_TAG = CTRL_MODE if IS_MODE_CMP else "control"
TREAT_TAG = TREAT_MODE if IS_MODE_CMP else "treatment"


def treatment_name():
    return TREAT_NAME


def delta_chip(a, b, worse_high):
    if a == 0 and b == 0:
        return '<span class="d d-flat">—</span>'
    d = b - a
    pct = d / a * 100 if a else (100.0 if b else 0.0)
    if abs(pct) < 5:
        return f'<span class="d d-flat">{b:.2f}</span>'
    better = (d < 0) if worse_high else (d > 0)
    cls = "d-good" if better else "d-bad"
    return f'<span class="d {cls}">{b:.2f} {"▼" if d < 0 else "▲"}{abs(pct):.0f}%</span>'


ctrl_agg = {c: CTRL["clips"][c]["aggregate"] for c in CLIPS}
treat_agg = {c: TREAT["clips"][c]["aggregate"] for c in CLIPS}
colmax = {k: max(max(ctrl_agg[c].get(k, 0), treat_agg[c].get(k, 0)) for c in CLIPS) for k, *_ in COLS}
ACTIVE = [col for col in COLS if col[3] or colmax[col[0]] > col[4]]
CLEAN = [col for col in COLS if col not in ACTIVE and col[2]]

# Per-clip signature: the control issue with the highest value/trigger ratio.
sig = {}
for c in CLIPS:
    best = None
    for i in CTRL["issues"]:
        if i["clip"] == c:
            ratio = i["value"] / i["trigger"]
            if not best or ratio > best[1]:
                best = (i["metric"], ratio)
    sig[c] = (f"{SHORT.get(best[0], best[0])} ×{best[1]:.1f}", "crit" if best[1] > 2 else "warn") \
        if best else ("clean", "good")


def scorecard_rows():
    out = []
    for c in CLIPS:
        s, cls = sig[c]
        ident = (f'<td class="idcell"><img class="thumb" src="{thumb(c)}" alt="{c}">'
                 f'<div class="idmeta"><span class="clipname">{c}</span>'
                 f'<span class="pill p-{cls}">{s}</span></div></td>')
        cells = [ident]
        for k, _, worse, _, _ in ACTIVE:
            a, b = ctrl_agg[c].get(k, 0), treat_agg[c].get(k, 0)
            cells.append(f'<td><div class="cv">{a:.2f}</div>{delta_chip(a, b, worse)}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


# metric -> (short header, what it measures, direction). Only the ones that appear render.
METRIC_DEFS = [
    ("pop_px_p50", "pop", "L↔R horizontal disparity (sub-pixel tile phase-correlation) — the amount of stereo depth.", "higher = more 3D"),
    ("depth_spread", "dspread", "p95−p5 of the normalized depth = pop available at the source.", "higher = more depth to work with"),
    ("edge_acc_p50", "edge_acc", "Distance (depth-px) from each depth silhouette to the nearest true SOURCE color edge.", "lower = silhouette sits on the real edge"),
    ("swim_p50", "swim", "Frame-to-frame depth change where the SOURCE is static — depth instability, separated from real motion.", "lower = steadier depth"),
    ("stretch_area", "stretch", "Area (‰ of the eye) of the large horizontal disocclusion smear beside silhouettes (bg rubber-banded to fill the gap).", "lower = less smear"),
    ("rim_over_p95", "rim", "Brightness of the thin white line hugging a silhouette (the residual fill fringe), luma ×255.", "lower = fainter fringe"),
    ("disocc_smear", "smear", "Horizontal-detail deficit in the narrow band beside silhouettes; on flat content also fingerprints hallucinated depth edges.", "lower = crisper fill"),
    ("flicker_p50", "flick", "Whole-frame temporal change of the SBS luma (×255).", "lower = steadier"),
    ("flicker_disocc_p50", "flick_dis", "Flicker restricted to the disocclusion bands — inpaint/stretch re-hallucination shimmer.", "lower = less boiling along edges"),
    ("vmisalign_px", "vmis", "Median vertical L↔R offset — parallax must be horizontal-only, so this is a geometry correctness check.", "must be ≈ 0"),
]


DEF_BY_KEY = {k: (what, d) for k, h, what, d in METRIC_DEFS}


def tip_text(metric):
    d = DEF_BY_KEY.get(metric)
    return f"{d[0]} ({d[1]})".replace('"', "'") if d else ""


def mtip(metric, label):
    """Metric label wrapped with a native-title tooltip (reliable inside the scroll container)."""
    t = tip_text(metric)
    return f'<span class="mtip" title="{t}">{label}</span>' if t else label


def metrics_section():
    present = {k for k, *_ in COLS if k in colmax} | {i["metric"] for i in CTRL["issues"]}
    rows = "".join(
        f'<tr><td class="mname">{h}</td><td class="mwhat">{what}</td><td class="mdir">{d}</td></tr>'
        for k, h, what, d in METRIC_DEFS if k in present)
    return (f'<section><h2>What the metrics mean</h2>'
            f'<p class="sub">Definitions for the metrics used below. All are computed on the real '
            f'SBS frames the headset would receive (no CPU replica). Absolute values are '
            f'resolution-dependent, so compare within a run, not across clip sets.</p>'
            f'<div class="tablewrap"><table class="mtab"><thead><tr><th>metric</th>'
            f'<th>what it measures</th><th>direction</th></tr></thead><tbody>{rows}</tbody></table></div></section>')


def conclusion_section():
    """Auto-derived verdict: per-metric mean across clips, classified into wins/costs, plus a
    shippability call from the gate. Regenerated with every report, so it always reflects the run."""
    wins, costs = [], []
    for k, h, worse, _, _ in COLS:
        a = np.mean([ctrl_agg[c].get(k, 0) for c in CLIPS])
        b = np.mean([treat_agg[c].get(k, 0) for c in CLIPS])
        if a < 1e-6 and b < 1e-6:
            continue
        pct = (b - a) / a * 100 if a else 100.0
        # Significant = both a relative move AND an absolute one (half the gate's abs_floor),
        # so sub-pixel noise on near-zero metrics doesn't read as a headline.
        floor = THR.get(k, {}).get("abs_floor", 0.0) / 2.0
        if abs(pct) < 5 or abs(b - a) < floor:
            continue
        # In a mode comparison neither direction is "better/worse" globally (it's a tradeoff);
        # split by which run each metric favors instead.
        favors_treat = (pct < 0) if worse else (pct > 0)
        txt = f"{mtip(k, '<b>' + h + '</b>')} {CTRL_TAG} {a:.2f} → {TREAT_TAG} {b:.2f} ({pct:+.0f}%)"
        (wins if favors_treat else costs).append(txt)
    li = ""
    if IS_MODE_CMP:
        if wins:
            li += f'<li class="c-win">{TREAT_NAME} is better on: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">{CTRL_NAME} is better on: {" · ".join(costs)}</li>'
        verdict = (f"<b>Tradeoff, not a regression:</b> these are two different pipeline "
                   f"configs, so the gate below (which compares {TREAT_NAME} to the committed "
                   f"{CTRL_MODE} baselines) reads as differences, not regressions. Pick per the "
                   f"tradeoff above and the per-clip evidence.")
    else:
        regs = TREAT.get("regressions", [])
        if wins:
            li += f'<li class="c-win">Improved: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">Worsened: {" · ".join(costs)}</li>'
        verdict = (f"<b>Not shippable as-is:</b> the gate fails with {len(regs)} regression(s) "
                   f"past threshold — the costs outweigh the wins above." if regs
                   else "<b>Shippable:</b> measurable wins with no regressions past threshold."
                   if wins else "<b>No meaningful effect:</b> all metrics within baseline noise.")
    head = (f"{CTRL_NAME} → {TREAT_NAME}" if IS_MODE_CMP else f"Treatment: <b>{treatment_name()}</b>")
    return (f'<section><h2>Conclusion</h2>'
            f'<p class="sub" style="margin-bottom:12px">{head} — mean movement across all '
            f'{len(CLIPS)} clips.</p>'
            f'<ul class="concl">{li}<li>{verdict}</li></ul>{gate_strip()}</section>')


def gate_strip():
    regs = TREAT.get("regressions", [])
    noun = "difference(s) vs " + CTRL_MODE + " baseline" if IS_MODE_CMP else "regression(s)"
    if not regs:
        return ('<div class="gate gate-pass"><b>Gate: PASS</b> — no '
                + noun + ' past threshold (run_eval exit 0).</div>')
    arrow = "→"
    items = "".join(f'<li><code>{r["clip"]}.{r["metric"]}</code> {r["baseline"]} {arrow} {r["value"]}'
                    + (f' <span class="wf">worst frame {r["frame"]}</span>' if "frame" in r else "")
                    + "</li>" for r in regs)
    cls = "gate-fail" if not IS_MODE_CMP else "gate-info"
    label = (f"{len(regs)} {noun}" if IS_MODE_CMP else f"{len(regs)} REGRESSION(S) — run_eval exit 1")
    return f'<div class="gate {cls}"><b>Gate: {label}</b><ul>{items}</ul></div>'


def issue_sections():
    # Per metric, gather two kinds of clips: ABSOLUTE issues (value over the trigger, in either
    # run) and REGRESSIONS (the treatment worsened it past tolerance, even if still under the
    # trigger). The second kind is why the biggest MOVER — e.g. c525 stretch 1.5->3.5, a
    # regression that stays below the 4.0 trigger — still gets its crop shown.
    metrics = [m for m in ISSUE_DEFS]
    reg_by = {}
    for r in TREAT.get("regressions", []):
        reg_by.setdefault(r["metric"], {})[r["clip"]] = r.get("frame")
    html = []
    for metric in metrics:
        title, temporal, desc = ISSUE_DEFS[metric]
        trig = THR.get(metric, {}).get("trigger", 1e9)
        entries = {}  # clip -> (kind, frame, sort_severity)
        for i in CTRL["issues"]:
            if i["metric"] == metric:
                entries[i["clip"]] = ("issue", i.get("frame"), i["value"] / i["trigger"])
        for c, frame in reg_by.get(metric, {}).items():
            a, b = ctrl_agg[c].get(metric, 0), treat_agg[c].get(metric, 0)
            if c in entries:  # already an absolute issue; note it also regressed
                entries[c] = ("issue+regressed", entries[c][1], entries[c][2])
            else:
                entries[c] = ("regressed", frame, max(b, a) / trig)
        if not entries:
            continue
        # Separate quotas so a mover is never crowded out by absolute issues: top-3 absolute
        # issues + up to 2 pure regressions (the clips the treatment pushed the worse way).
        abs_e = sorted((e for e in entries.items() if not e[1][0] == "regressed"),
                       key=lambda kv: -kv[1][2])[:3]
        reg_e = sorted((e for e in entries.items() if e[1][0] == "regressed"),
                       key=lambda kv: -kv[1][2])[:2]
        order = abs_e + reg_e
        cards = []
        for c, (kind, frame, _) in order:
            a, b = ctrl_agg[c].get(metric, 0), treat_agg[c].get(metric, 0)
            pct = (b - a) / a * 100 if a else (100 if b else 0)
            badge = ('<span class="pill p-crit">regressed</span>' if kind == "regressed"
                     else '<span class="pill p-crit">also regressed</span>' if "regressed" in kind
                     else '<span class="pill p-warn">issue</span>')
            imgs = ""
            pair = crop_at_silhouette(c, frame if frame is not None else mid_frame(ctrl_dir, c))
            if pair:
                if temporal:
                    imgs = (f'<div class="pair single"><figure><span class="tag">{CTRL_TAG} · worst '
                            f'frame {frame}</span><img src="{pair[0]}"></figure></div>')
                else:
                    imgs = (f'<div class="pair"><figure><span class="tag">{CTRL_TAG}</span>'
                            f'<img src="{pair[0]}"></figure><figure><span class="tag t-treat">'
                            f'{TREAT_TAG}</span><img src="{pair[1]}"></figure></div>')
            cards.append(f'<div class="issue-clip"><div class="ic-head"><span class="clipname">{c}'
                         f'</span> {badge} <span class="metricval">{mtip(metric, SHORT.get(metric, metric))}: '
                         f'<b>{a:.2f}</b> &rarr; {b:.2f} ({pct:+.0f}%) &middot; worst frame {frame}'
                         f'</span></div>{imgs}</div>')
        note = ' <span class="pill p-info">temporal</span>' if temporal else ""
        html.append(f'<section><h2>{title}{note}</h2><p class="sub">{desc}</p>{"".join(cards)}</section>')
    return "\n".join(html)


def clean_footer():
    if not CLEAN:
        return ""
    items = ", ".join(f"{h} {colmax[k]:.2f}" for k, h, *_ in CLEAN)
    return (f'<p class="foot"><b>Clean this run (max ≈ 0, collapsed):</b> {items}. Still measured '
            f'every run — any one auto-appears as a column when it crosses threshold.</p>')


meta = CTRL["meta"]
hdr_cells = "".join(f'<th title="{tip_text(k)}">{h}</th>' for k, h, *_ in ACTIVE)

HTML = """<style>
:root{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;
  --accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;
  --line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;--accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;}
:root[data-theme="dark"]{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;--line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}
*{box-sizing:border-box}
.wrap{max-width:1060px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);font-family:var(--sans);line-height:1.6;background:var(--bg);-webkit-font-smoothing:antialiased}
h1,h2{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-size:36px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:16.5px;max-width:68ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;display:flex;gap:18px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
section{margin-top:52px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.sub{color:var(--muted);font-size:14px;margin:0 0 20px;max-width:72ch}
.foot{margin-top:14px;color:var(--muted);font-size:13px}.foot b{color:var(--ink)}
.gate{border-radius:10px;padding:14px 18px;font-size:14px;margin-top:26px;border:1px solid var(--line)}
.gate-pass{background:color-mix(in srgb,var(--good) 9%,transparent);border-color:color-mix(in srgb,var(--good) 40%,var(--line))}
.gate-fail{background:color-mix(in srgb,var(--crit) 8%,transparent);border-color:color-mix(in srgb,var(--crit) 40%,var(--line))}
.gate-info{background:color-mix(in srgb,var(--accent) 8%,transparent);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.gate ul{margin:8px 0 0;padding-left:20px}.gate li{margin:2px 0}
.gate .wf{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.concl{margin:0 0 18px;padding-left:20px;font-size:14.5px}
.concl li{margin:7px 0;max-width:78ch}
.concl b{color:var(--ink)}
.c-win{color:var(--good)}.c-win b{color:var(--good)}
.c-cost{color:var(--crit)}.c-cost b{color:var(--crit)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left;min-width:225px}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
thead th[title]:not([title=""]),.mtip{cursor:help;text-decoration:underline dotted;text-underline-offset:3px;text-decoration-color:color-mix(in srgb,var(--muted) 60%,transparent)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
.cv{font-size:14px;color:var(--ink)}
.mtab td,.mtab th{text-align:left;white-space:normal}
.mtab .mname{font-family:var(--mono);font-size:12.5px;color:var(--accent);font-weight:600;white-space:nowrap;vertical-align:top}
.mtab .mwhat{font-family:var(--sans);font-size:13.5px;color:var(--ink);max-width:60ch}
.mtab .mdir{font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:nowrap;vertical-align:top}
.d{font-size:11px;font-family:var(--mono);display:inline-block;margin-top:3px;padding:1px 6px;border-radius:20px;font-weight:600}
.d-flat{color:var(--muted);background:color-mix(in srgb,var(--muted) 12%,transparent)}
.d-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.d-bad{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.idcell{display:flex;align-items:center;gap:11px}
.thumb{width:64px;height:auto;border-radius:5px;border:1px solid var(--line);display:block;flex:0 0 auto}
.idmeta{display:flex;flex-direction:column;gap:5px;align-items:flex-start}
.clipname{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.p-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
.issue-clip{margin-top:20px}
.ic-head{display:flex;align-items:baseline;gap:14px;margin-bottom:10px;flex-wrap:wrap}
.metricval{font-family:var(--mono);font-size:13px;color:var(--muted)}.metricval b{color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pair.single{grid-template-columns:1fr;max-width:540px}
.pair figure{margin:0;position:relative}
.pair img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.tag{position:absolute;top:8px;left:8px;font-family:var(--mono);font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;background:color-mix(in srgb,var(--bg) 82%,transparent);border:1px solid var(--line);color:var(--ink)}
.tag.t-treat{color:var(--warn)}
pre{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:5px}
@media (max-width:640px){.pair{grid-template-columns:1fr}h1{font-size:28px}}
</style>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; run_eval A/B report</div>
  <h1>__H1__</h1>
  <p class="lede">Generated from two <code>run_eval.py</code> runs over the committed clip set —
  the real pipeline and gated metrics. __LEDE__</p>
  <div class="meta"><span>__DATE__</span><span>git __SHA____DIRTY__</span>
  <span>__NCLIPS__ clips</span><span>__MODELS__</span></div>

  __METRICS__

  __CONCLUSION__

  <section>
    <h2>Scorecard — __CTRL_NAME__ → __TREAT_NAME__</h2>
    <p class="sub">One row per clip (auto-discovered; signature = its strongest triggered issue,
    as value×trigger ratio). Each cell shows the <b>__CTRL_TAG__</b> value on top and a chip for
    <b>__TREAT_TAG__</b> below (its value + %Δ; green = __TREAT_TAG__ better on this metric, red =
    worse, grey &lt; 5%). pop is higher-is-better; the rest higher-is-worse (see the definitions
    above). Flat metrics collapse to the footer and auto-return when non-zero.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>clip</th>__HDR__</tr></thead>
      <tbody>__ROWS__</tbody>
    </table></div>
    __FOOTER__
  </section>

  __ISSUES__

  <section>
    <h2>Reproduce</h2>
    <pre>python tools/sbsbench/run_eval.py --label ctrl                              # control (gates vs baselines)
python tools/sbsbench/run_eval.py --label treat --extra __TREAT_ARGS__     # treatment
python tools/sbsbench/build_report.py &lt;build&gt;/sbs_eval/ctrl &lt;build&gt;/sbs_eval/treat report.html</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:12px">Metrics: <code>tools/sbsbench/sbsbench.py</code>
    &middot; gate: <code>thresholds.json</code> &middot; plan: <code>docs/sbs-benchmark-plan.md</code></p>
  </section>
</div>
"""

models = ", ".join(sorted({m for r in (CTRL, TREAT)
                           for m in {e["meta"].get("model", "?") for e in r["clips"].values()}}))
if IS_MODE_CMP:
    h1 = f"{CTRL_NAME} vs. {TREAT_NAME}"
    lede = (f"Comparing two pipeline modes on identical clips: <b>{CTRL_NAME}</b> against "
            f"<b>{TREAT_NAME}</b>. Neither is a regression of the other — it is a tradeoff, "
            f"read from the per-metric split and the per-clip evidence below.")
else:
    h1 = "Control vs. treatment, by issue"
    lede = f"Treatment under test: <b>{TREAT_NAME}</b>, gated against the committed baselines."
HTML = (HTML.replace("__H1__", h1).replace("__LEDE__", lede)
        .replace("__CTRL_NAME__", CTRL_NAME).replace("__TREAT_NAME__", TREAT_NAME)
        .replace("__DATE__", meta["timestamp"][:10]).replace("__SHA__", meta["git_sha"])
        .replace("__DIRTY__", "+dirty" if meta["git_dirty"] else "")
        .replace("__NCLIPS__", str(len(CLIPS)))
        .replace("__MODELS__", models).replace("__CONCLUSION__", conclusion_section())
        .replace("__CTRL_TAG__", CTRL_TAG).replace("__TREAT_TAG__", TREAT_TAG)
        .replace("__HDR__", hdr_cells).replace("__ROWS__", scorecard_rows())
        .replace("__METRICS__", metrics_section())
        .replace("__FOOTER__", clean_footer()).replace("__ISSUES__", issue_sections())
        .replace("__TREAT_ARGS__", " ".join(TREAT["meta"].get("extra_args") or ["--mode game"])))
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_html, f"({len(HTML) // 1024} KB)")

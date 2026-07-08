#!/usr/bin/env python3
"""Assemble the SBS visual-benchmark A/B report: a control-vs-treatment scorecard (one row per
clip) and, per triggered issue, a section showing the affected clips' control/treatment crops.
Session deliverable. Usage: build_report.py <control_dir> <treat_dir> <assets_dir> <out.html>
  <control_dir>  <clip>.json + <clip>/ harness output (baseline)
  <treat_dir>    <clip>.json + <clip>/ harness output (the change under test)
  <assets_dir>   <clip>/crop_{control,treat,depth}.png (from make_report_crops.py)
"""
import base64
import io
import json
import os
import sys

from PIL import Image

ctrl_dir, treat_dir, assets, out_html = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TREATMENT = "2× divergence"  # the change under test in this run

CLIPS = ["c525", "c747", "c339", "c647", "c841"]
SIG = {"c525": ("strong clean 3D", "good"), "c747": ("white-line + soft silhouettes", "crit"),
       "c339": ("temporal shimmer", "warn"), "c647": ("stretch + unstable", "crit"),
       "c841": ("calm / stable", "good")}
# metric, header, worse-is-higher, always-show, notable-threshold
COLS = [
    ("pop_px_p50", "pop", False, True, 0), ("edge_acc_p50", "edge_acc", True, False, 2.0),
    ("stretch_area", "stretch", True, False, 2.0), ("rim_over_p95", "rim", True, False, 1.0),
    ("swim_p50", "swim", True, False, 1.0), ("flicker_p50", "flick", True, True, 0),
    ("flicker_disocc_p50", "flick_dis", True, True, 0), ("vmisalign_px", "vmis", True, False, 0.5),
    ("disocc_smear", "smear", True, False, 0.02),
]
# title, metric, control-threshold to "trigger", is-temporal, description
ISSUES = [
    ("Disocclusion stretch band", "stretch_area", 4.0, False,
     "Background rubber-banded horizontally to fill the gap the foreground uncovered — "
     "eye-asymmetric (left eye smears left, right eye right)."),
    ("Silhouette white line", "rim_over_p95", 3.0, False,
     "A thin bright fringe hugging the silhouette — the residual sliver where the fill "
     "doesn't reach the foreground edge."),
    ("Soft / floating silhouettes", "edge_acc_p50", 5.0, False,
     "The depth silhouette sits off the true object edge, so the cut-out is loosely placed."),
    ("Disocclusion shimmer", "flicker_disocc_p50", 8.0, True,
     "The ¼-res fill re-hallucinates frame to frame in the disocclusion bands — boiling "
     "along edges. Temporal, so the still crops can't show it, but the numbers do."),
]

ctrl = {c: json.load(open(os.path.join(ctrl_dir, c + ".json")))["aggregate"] for c in CLIPS}
treat = {c: json.load(open(os.path.join(treat_dir, c + ".json")))["aggregate"] for c in CLIPS}


def durl(path, w=None):
    im = Image.open(path).convert("RGB")
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def thumb(clip, w=132):
    im = Image.open(os.path.join(ctrl_dir, clip, "sbs_00016.png")).convert("RGB")
    left = im.crop((0, 0, im.width // 2, im.height))
    left.thumbnail((w, w * 2), Image.LANCZOS)
    b = io.BytesIO()
    left.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


thumbs = {c: thumb(c) for c in CLIPS}
crop = {c: {w: durl(os.path.join(assets, c, f"crop_{w}.png"), 560) for w in ("control", "treat")}
        for c in CLIPS if os.path.isdir(os.path.join(assets, c))}

# adaptive columns: shown if always or max (over control+treatment) crosses notable
colmax = {k: max(max(ctrl[c].get(k, 0), treat[c].get(k, 0)) for c in CLIPS) for k, *_ in COLS}
ACTIVE = [col for col in COLS if col[3] or colmax[col[0]] > col[4]]
CLEAN = [col for col in COLS if col not in ACTIVE and col[2]]


def delta_chip(a, b, worse_high):
    if a == 0 and b == 0:
        return '<span class="d d-flat">—</span>'
    d = b - a
    pct = d / a * 100 if a else (100.0 if b else 0.0)
    if abs(pct) < 5:
        return f'<span class="d d-flat">{b:.2f}</span>'
    better = (d < 0) if worse_high else (d > 0)
    arrow = ("▼" if d < 0 else "▲")
    cls = "d-good" if better else "d-bad"
    return f'<span class="d {cls}">{b:.2f} {arrow}{abs(pct):.0f}%</span>'


def scorecard_rows():
    out = []
    for c in CLIPS:
        sig, cls = SIG[c]
        ident = (f'<td class="idcell"><img class="thumb" src="{thumbs[c]}" alt="{c}">'
                 f'<div class="idmeta"><span class="clipname">{c}</span>'
                 f'<span class="pill p-{cls}">{sig}</span></div></td>')
        cells = [ident]
        for k, _, worse, _, _ in ACTIVE:
            a, b = ctrl[c].get(k, 0), treat[c].get(k, 0)
            cells.append(f'<td><div class="cv">{a:.2f}</div>{delta_chip(a, b, worse)}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


def clean_footer():
    if not CLEAN:
        return ""
    items = ", ".join(f"{h} {colmax[k]:.2f}" for k, h, *_ in CLEAN)
    return (f'<p class="foot"><b>Clean this run (max ≈ 0, collapsed):</b> {items}. Still measured '
            f'every run &mdash; any one auto-appears as a column the moment it crosses threshold.</p>')


def issue_sections():
    html = []
    for title, metric, thr, temporal, desc in ISSUES:
        hits = sorted([c for c in CLIPS if ctrl[c].get(metric, 0) > thr],
                      key=lambda c: -ctrl[c].get(metric, 0))
        if not hits:
            continue
        cards = []
        for c in hits:
            a, b = ctrl[c].get(metric, 0), treat[c].get(metric, 0)
            imgs = ""
            if c in crop and not temporal:
                imgs = (f'<div class="pair">'
                        f'<figure><span class="tag">control</span><img src="{crop[c]["control"]}" alt="control"></figure>'
                        f'<figure><span class="tag t-treat">{TREATMENT}</span><img src="{crop[c]["treat"]}" alt="treat"></figure>'
                        f'</div>')
            elif temporal and c in crop:
                imgs = (f'<div class="pair single"><figure><span class="tag">control frame</span>'
                        f'<img src="{crop[c]["control"]}" alt="frame"></figure></div>')
            cards.append(
                f'<div class="issue-clip"><div class="ic-head"><span class="clipname">{c}</span>'
                f'<span class="metricval">{metric.replace("_p50","").replace("_p95","")}: '
                f'<b>{a:.2f}</b> &rarr; {b:.2f}</span></div>{imgs}</div>')
        note = (' <span class="pill p-info">temporal &mdash; numbers, not stills</span>' if temporal else "")
        html.append(
            f'<section><h2>{title}{note}</h2><p class="sub">{desc}</p>{"".join(cards)}</section>')
    return "\n".join(html)


hdr_cells = "".join(f"<th>{h}</th>" for _, h, *_ in ACTIVE)

HTML = """<style>
:root{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;
  --accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;
  --line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;--accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;}
:root[data-theme="dark"]{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;--line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}
*{box-sizing:border-box}
.wrap{max-width:1040px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);font-family:var(--sans);line-height:1.6;background:var(--bg);-webkit-font-smoothing:antialiased}
h1,h2,h3{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-size:37px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:17px;max-width:66ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;margin-top:14px}
.legend span{display:flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
section{margin-top:52px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.sub{color:var(--muted);font-size:14px;margin:0 0 22px;max-width:72ch}
.foot{margin-top:14px;color:var(--muted);font-size:13px}.foot b{color:var(--ink)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left;min-width:230px}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
.cv{font-size:14px;color:var(--ink)}
.d{font-size:11px;font-family:var(--mono);display:inline-block;margin-top:3px;padding:1px 6px;border-radius:20px;font-weight:600}
.d-flat{color:var(--muted);background:color-mix(in srgb,var(--muted) 12%,transparent)}
.d-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.d-bad{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.idcell{display:flex;align-items:center;gap:11px}
.thumb{width:66px;height:auto;border-radius:5px;border:1px solid var(--line);display:block;flex:0 0 auto}
.idmeta{display:flex;flex-direction:column;gap:5px;align-items:flex-start}
.clipname{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.p-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
.issue-clip{margin-top:20px}
.ic-head{display:flex;align-items:baseline;gap:14px;margin-bottom:10px}
.metricval{font-family:var(--mono);font-size:13px;color:var(--muted)}.metricval b{color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pair.single{grid-template-columns:1fr;max-width:520px}
.pair figure{margin:0;position:relative}
.pair img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.tag{position:absolute;top:8px;left:8px;font-family:var(--mono);font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;background:color-mix(in srgb,var(--bg) 82%,transparent);border:1px solid var(--line);color:var(--ink)}
.tag.t-treat{color:var(--warn)}
pre{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:5px}
@media (max-width:640px){.pair{grid-template-columns:1fr}h1{font-size:29px}}
</style>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; host visual A/B</div>
  <h1>Control vs. treatment, by issue</h1>
  <p class="lede">Metrics on the real side-by-side frames the headset receives &mdash; five clips
  run through the shipped pipeline (<b>control</b>) and the change under test (<b>treatment:
  __TREATMENT__</b>). The scorecard compares them per clip; each issue below drills into the clips
  that show it, control beside treatment.</p>
  <div class="meta"><span>2026-07-07</span><span>movie &middot; da3mono-large + MLBW l4</span><span>5 clips &middot; 32 frames</span></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--good)"></span>treatment better</span>
    <span><span class="sw" style="background:var(--crit)"></span>treatment worse</span>
    <span><span class="sw" style="background:color-mix(in srgb,var(--muted) 40%,transparent)"></span>&lt;5% change</span>
  </div>

  <section>
    <h2>Scorecard &mdash; control &rarr; treatment</h2>
    <p class="sub">One row per clip. Each cell shows the control value with a chip for the treatment
    result and its change (red = the treatment made this metric worse, green = better). pop is
    higher-is-better; the rest are higher-is-worse. Columns appear adaptively &mdash; a flat metric
    is collapsed to the footer but returns as a column if any run makes it non-zero.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>clip</th>__HDR__</tr></thead>
      <tbody>__ROWS__</tbody>
    </table></div>
    __FOOTER__
  </section>

  __ISSUES__

  <section>
    <h2>Reproduce</h2>
    <pre>cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench --frames clips/c525 --out out/control --movie
./sunshine.exe ...                                --frames clips/c525 --out out/treat   --movie --divergence 0.027
python tools/sbsbench/sbsbench.py --seq out/control --frames clips/c525 --json control.json
python tools/sbsbench/sbsbench.py --seq out/treat   --frames clips/c525 --baseline control.json</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:12px">Harness: <code>src/sbs_bench_harness.cpp</code>.
    Metrics: <code>tools/sbsbench/sbsbench.py</code>. Plan: <code>docs/sbs-benchmark-plan.md</code>.</p>
  </section>
</div>
"""

HTML = (HTML.replace("__TREATMENT__", TREATMENT).replace("__HDR__", hdr_cells)
        .replace("__ROWS__", scorecard_rows()).replace("__FOOTER__", clean_footer())
        .replace("__ISSUES__", issue_sections()))
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_html, f"({len(HTML)//1024} KB)")

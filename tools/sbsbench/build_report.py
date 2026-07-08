#!/usr/bin/env python3
"""Assemble the SBS visual-benchmark HTML report: all metric definitions, a real-data scorecard
read from the per-clip JSONs, and a control/treatment A/B with image crops. Session deliverable.
Usage: build_report.py <all_dir> <ab_dir> <assets_dir> <out.html>
  <all_dir>   dir of <clip>.json scorecards (from `sbsbench --seq --json`)
  <ab_dir>    dir with c525_control/ c525_treat/ (for the A/B crops' full frame)
  <assets_dir> crop_control.png / crop_treat.png / crop_depth.png
"""
import base64
import io
import json
import os
import sys

from PIL import Image

alld, ab, assets, out_html = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

CLIPS = ["c525", "c747", "c339", "c647", "c841"]
SIG = {  # per-clip plain-language read + severity class
    "c525": ("strong clean 3D", "good"),
    "c747": ("white-line + soft silhouettes", "crit"),
    "c339": ("temporal shimmer", "warn"),
    "c647": ("stretch band + unstable depth", "crit"),
    "c841": ("calm / stable", "good"),
}
# scorecard columns: key, header, higher-is-worse?
COLS = [
    ("pop_px_p50", "pop", False), ("vmisalign_px", "vmis", True),
    ("depth_spread", "dspread", False), ("edge_acc_p50", "edge_acc", True),
    ("disocc_smear", "smear", True), ("stretch_area", "stretch", True),
    ("rim_over_p95", "rim", True), ("flicker_p50", "flick", True),
    ("flicker_disocc_p50", "flick_dis", True), ("swim_p50", "swim", True),
]


def durl(path, w=None):
    im = Image.open(path).convert("RGB")
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


data = {c: json.load(open(os.path.join(alld, c + ".json")))["aggregate"] for c in CLIPS}
# per-column max (of the worse-is-higher ones) for a light heat tint
colmax = {k: max(data[c].get(k, 0) for c in CLIPS) or 1 for k, _, _ in COLS}


def scorecard_rows():
    out = []
    for c in CLIPS:
        a = data[c]
        sig, cls = SIG[c]
        tds = [f'<td class="lab">{c}</td>']
        for k, _, worse in COLS:
            v = a.get(k, 0)
            tint = ""
            if worse and colmax[k] > 0:
                f = min(v / colmax[k], 1.0)
                if f > 0.15:
                    tint = f' style="background:color-mix(in srgb,var(--crit) {int(f*22)}%,transparent)"'
            tds.append(f"<td{tint}>{v:.2f}</td>")
        tds.append(f'<td class="lab"><span class="pill p-{cls}">{sig}</span></td>')
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "\n".join(out)


full = durl(os.path.join(ab, "c525_control", "sbs_00016.png"), 1200)
ctrl = durl(os.path.join(assets, "crop_control.png"), 720)
treat = durl(os.path.join(assets, "crop_treat.png"), 720)
depth = durl(os.path.join(assets, "crop_depth.png"), 720)

hdr_cells = "".join(f"<th>{h}</th>" for _, h, _ in COLS)

HTML = """<style>
:root {
  --bg:#f5f6f7; --panel:#ffffff; --ink:#12181d; --muted:#5c6a74; --line:#dbe1e6;
  --accent:#0e8f9c; --accent-soft:#d7eef0; --good:#1f9d63; --warn:#c98a1a; --crit:#c4483a;
  --mono:ui-monospace,"SF Mono","Cascadia Mono","JetBrains Mono",Consolas,monospace;
  --sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0c1013; --panel:#141a1f; --ink:#e8edf0; --muted:#8b9aa4; --line:#252e35;
  --accent:#38c0cd; --accent-soft:#123037; --good:#3fca86; --warn:#e0a94a; --crit:#e56a5c;
}}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;--accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;}
:root[data-theme="dark"]{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;--line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}
*{box-sizing:border-box}
.wrap{max-width:1040px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);
  font-family:var(--sans);line-height:1.6;background:var(--bg);-webkit-font-smoothing:antialiased}
h1,h2,h3{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-size:38px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:17px;max-width:66ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
section{margin-top:56px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.04em;text-transform:uppercase;color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:8px}
.sub{color:var(--muted);font-size:14px;margin:0 0 22px;max-width:70ch}
p{max-width:70ch}
.group{font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:24px 0 12px}
.grid{display:grid;gap:14px}
.defs{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px 17px}
.card h3{font-size:14px;font-weight:600;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.card .name{font-family:var(--mono);font-size:12.5px;color:var(--accent);background:var(--accent-soft);padding:2px 7px;border-radius:5px}
.card p{font-size:13px;color:var(--muted);margin:9px 0 0}
.card .dir{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:10px 13px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
td.lab{font-family:var(--sans)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.p-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
figure{margin:0}
.full img{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
figcaption{font-size:12.5px;color:var(--muted);margin-top:10px;font-family:var(--mono)}
.ab{grid-template-columns:1fr 1fr;margin-top:8px}
.ab figure img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.ab .tag{font-family:var(--mono);font-size:12px;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.depthrow{grid-template-columns:2fr 3fr;align-items:center;margin-top:16px;gap:22px}
.depthrow img{width:100%;border-radius:9px;border:1px solid var(--line)}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:15px 17px;font-size:13.5px;color:var(--muted)}
.note b{color:var(--ink)}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:5px;color:var(--ink)}
pre{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
@media (max-width:640px){.ab,.depthrow{grid-template-columns:1fr}h1{font-size:30px}}
</style>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; host visual benchmark</div>
  <h1>The metric suite, on real footage</h1>
  <p class="lede">Ten no-reference metrics computed on the actual side-by-side frames the headset
  receives &mdash; produced by the real depth estimator and warp shaders over fixed clips, not a
  CPU replica. Together they localize a change to the right subsystem and give each catalogued
  artifact its own number.</p>
  <div class="meta"><span>2026-07-07</span><span>movie mode &middot; da3mono-large + MLBW l4</span>
  <span>5 clips &middot; 32 frames each</span><span>tools/sbsbench</span></div>

  <section>
    <h2>Metrics</h2>
    <p class="sub">Grouped by what they watch. A change to the warp moves the disocclusion + pop
    metrics; a change to depth moves edge accuracy + swim &mdash; so a delta says <em>where</em> it landed.</p>

    <div class="group">Stereo &amp; geometry</div>
    <div class="grid defs">
      <div class="card"><h3><span class="name">pop_px</span> p50 / p95</h3><p>L&harr;R horizontal
        disparity (tile phase-correlation) &mdash; the amount of stereo depth.</p><div class="dir">higher &rarr; more 3D</div></div>
      <div class="card"><h3><span class="name">vmisalign_px</span></h3><p>Median vertical L&harr;R
        offset. Parallax must be horizontal-only, so this is a correctness check.</p><div class="dir">must be &asymp; 0</div></div>
    </div>

    <div class="group">Depth quality</div>
    <div class="grid defs">
      <div class="card"><h3><span class="name">depth_spread</span></h3><p>p95&minus;p5 of the
        normalized depth = pop available at the source. Separates a flat model from a flat warp.</p><div class="dir">higher &rarr; more depth</div></div>
      <div class="card"><h3><span class="name">edge_acc</span> p50 / p95</h3><p>Depth-px distance
        from each depth silhouette to the nearest <b>source</b> color edge. Catches soft / bent /
        floating silhouettes. (needs <code>--frames</code>)</p><div class="dir">small = on the real edge</div></div>
      <div class="card"><h3><span class="name">swim</span> p50</h3><p>Frame-to-frame depth change
        where the <b>source</b> is static &mdash; scene-cut / flat-content depth instability,
        separated from real motion. (needs <code>--frames</code>)</p><div class="dir">lower &rarr; steadier depth</div></div>
    </div>

    <div class="group">Disocclusion &amp; silhouette artifacts</div>
    <div class="grid defs">
      <div class="card"><h3><span class="name">disocc_smear</span></h3><p>Horizontal-detail deficit
        in the <em>narrow</em> band beside a silhouette &mdash; small-scale blur of the fill.</p><div class="dir">0 = clean &middot; &rarr;1 smeared</div></div>
      <div class="card"><h3><span class="name">stretch_area</span></h3><p>The <b>large</b> horizontal
        stretch band (background rubber-banded to fill the gap; eye-asymmetric). Area of wide
        low-h-gradient / vertically-streaked runs by a silhouette, per-mille of the eye.</p><div class="dir">higher &rarr; bigger smear patches</div></div>
      <div class="card"><h3><span class="name">rim_over</span> p95</h3><p>The <b>white line</b>
        hugging a silhouette &mdash; a thin bright ridge (horizontal white top-hat). The residual
        bright sliver where the fill doesn't reach the fg edge.</p><div class="dir">~0 = none &middot; higher = brighter line</div></div>
    </div>

    <div class="group">Temporal (needs a clip)</div>
    <div class="grid defs">
      <div class="card"><h3><span class="name">flicker</span> p50 / p95</h3><p>Frame-to-frame
        mean&#124;&Delta;&#124; of the whole SBS luma (&times;255). On the same clip, motion cancels
        in a baseline diff.</p><div class="dir">lower &rarr; steadier</div></div>
      <div class="card"><h3><span class="name">flicker_disocc</span></h3><p>Flicker restricted to
        the disocclusion bands &mdash; isolates the &frac14;-res inpaint / stretch re-hallucination
        shimmer from ordinary motion (runs ~2&ndash;3&times; frame flicker).</p><div class="dir">lower &rarr; less shimmer where it matters</div></div>
    </div>
  </section>

  <section>
    <h2>Real-data scorecard &mdash; 5 clips</h2>
    <p class="sub">Same movie pipeline, five clips. Each lights up a different artifact, so the row
    reads as a fingerprint. Red tint = worse (per column). Note <code>disocc_smear</code> is 0
    everywhere while <code>stretch_area</code> ranges 1.3&ndash;9.8 &mdash; the l4 warp keeps the
    <em>narrow</em> band clean, but the <em>large</em> stretch is real; that gap is why both exist.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>clip</th>__HDR__<th>signature</th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table></div>
    <p style="margin-top:14px;color:var(--muted);font-size:13px"><b style="color:var(--ink)">vmisalign
    = 0 on all 160 frames.</b> pop / depth_spread: higher is better (not tinted). Everything else:
    higher is worse.</p>
  </section>

  <section>
    <h2>A/B example &mdash; control vs. treatment</h2>
    <p class="sub">The per-change workflow: fix the clip, change one thing, diff. Lever =
    <b>divergence</b> (parallax) on c525, control vs 2&times;. pop +90% (measured); depth metrics
    stay flat because divergence is a warp-only lever &mdash; the suite attributes the change correctly.</p>
    <figure class="full"><img src="__FULL__" alt="Full SBS frame">
      <figcaption>Full SBS frame (control) &mdash; left &amp; right eye, 6132&times;1728.</figcaption></figure>
    <div class="grid ab">
      <figure><div class="tag"><span class="pill p-info">control</span> divergence 0.0135</div><img src="__CTRL__" alt="control"></figure>
      <figure><div class="tag"><span class="pill p-warn">treatment</span> divergence 0.027 (2&times;)</div><img src="__TREAT__" alt="treatment"></figure>
    </div>
    <div class="grid depthrow">
      <figure><img src="__DEPTH__" alt="depth"><figcaption>depth &middot; red = detected silhouette</figcaption></figure>
      <div class="note"><b>Reading it.</b> The hand shifts visibly further left at 2&times; divergence
      (pop +90%). The disocclusion fills cleanly here because the revealed background is low-texture
      mist &mdash; so <code>stretch_area</code> and <code>disocc_smear</code> stay near zero, correctly.
      On a textured background (e.g. a crowd) the same lever would light <code>stretch_area</code> up.</div>
    </div>
  </section>

  <section>
    <h2>Artifact metrics &mdash; injection-validated</h2>
    <p class="sub">Each artifact metric was checked by injecting the artifact into a real frame and
    confirming it fires (and stays quiet on look-alikes).</p>
    <div class="tablewrap"><table>
      <thead><tr><th>metric</th><th>clean</th><th>artifact injected</th><th>look-alike (must stay low)</th></tr></thead>
      <tbody>
        <tr><td class="lab">rim_over (white line)</td><td>2.0</td><td>206 &nbsp;<span class="pill p-crit">1px white line</span></td><td>2.0 &nbsp;<span class="pill p-good">broad bright patch</span></td></tr>
        <tr><td class="lab">stretch_area</td><td>1.7</td><td>6.5 &nbsp;<span class="pill p-crit">textured stretch</span></td><td>1.6 &nbsp;<span class="pill p-good">smooth fill</span></td></tr>
        <tr><td class="lab">flicker</td><td>&asymp;0</td><td colspan="2">converges to 0 on a static clip as the depth EMA settles</td></tr>
      </tbody>
    </table></div>
  </section>

  <section>
    <h2>Reproduce</h2>
    <pre>python tools/sbsbench/split_video.py clip.mp4 -o clips/c525 --max 32
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench --frames clips/c525 --out out/c525 --movie
python tools/sbsbench/sbsbench.py --seq out/c525 --frames clips/c525 --json c525.json
python tools/sbsbench/sbsbench.py --seq out/NEW --frames clips/c525 --baseline c525.json   # A/B diff</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:12px">Harness: <code>src/sbs_bench_harness.cpp</code>.
    Metrics: <code>tools/sbsbench/sbsbench.py</code>. Per-stage perf: <code>sbs_3d_perf_stats</code>.
    Plan: <code>docs/sbs-benchmark-plan.md</code>.</p>
  </section>
</div>
"""

HTML = (HTML.replace("__HDR__", hdr_cells).replace("__ROWS__", scorecard_rows())
        .replace("__FULL__", full).replace("__CTRL__", ctrl)
        .replace("__TREAT__", treat).replace("__DEPTH__", depth))
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_html, f"({len(HTML)//1024} KB)")

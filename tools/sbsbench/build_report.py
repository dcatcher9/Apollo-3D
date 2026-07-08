#!/usr/bin/env python3
"""Assemble the SBS visual-benchmark HTML report (embeds crops as data URIs). Internal to this
session's deliverable, not a general tool. Usage: build_report.py <ab_dir> <assets_dir> <out.html>"""
import base64
import io
import os
import sys

from PIL import Image

ab, assets, out_html = sys.argv[1], sys.argv[2], sys.argv[3]


def durl(path, w=None):
    im = Image.open(path).convert("RGB")
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# Full control SBS frame (both eyes) for context, downscaled.
full = durl(os.path.join(ab, "c525_control", "sbs_00016.png"), 1200)
ctrl = durl(os.path.join(assets, "crop_control.png"), 720)
treat = durl(os.path.join(assets, "crop_treat.png"), 720)
depth = durl(os.path.join(assets, "crop_depth.png"), 720)

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
.wrap{max-width:1000px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);
  font-family:var(--sans);line-height:1.6;background:var(--bg)}
.wrap{-webkit-font-smoothing:antialiased}
h1,h2,h3{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin-bottom:14px}
h1{font-size:38px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:17px;max-width:65ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;
  display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
section{margin-top:56px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.04em;text-transform:uppercase;
  color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:24px}
p{max-width:68ch}
.grid{display:grid;gap:16px}
.defs{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.card h3{font-size:15px;font-weight:640;display:flex;align-items:center;gap:9px}
.card .name{font-family:var(--mono);font-size:13px;color:var(--accent);background:var(--accent-soft);
  padding:2px 7px;border-radius:5px}
.card p{font-size:13.5px;color:var(--muted);margin:10px 0 0}
.card .f{font-family:var(--mono);font-size:12.5px;color:var(--ink);background:var(--bg);
  border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin-top:12px;overflow-x:auto}
.dir{font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;text-transform:uppercase;margin-top:10px;color:var(--muted)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:right;padding:11px 16px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);font-weight:600;background:var(--panel)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
td.lab{font-family:var(--sans)}
tr.flag td{background:color-mix(in srgb,var(--crit) 8%,transparent)}
.pill{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.p-good{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
figure{margin:0}
.full img{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
figcaption{font-size:12.5px;color:var(--muted);margin-top:10px;font-family:var(--mono)}
.ab{grid-template-columns:1fr 1fr;margin-top:8px}
.ab figure img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.ab .tag{font-family:var(--mono);font-size:12px;font-weight:600;margin-bottom:8px;display:flex;
  align-items:center;gap:8px}
.depthrow{grid-template-columns:2fr 3fr;align-items:center;margin-top:16px;gap:22px}
.depthrow img{width:100%;border-radius:9px;border:1px solid var(--line)}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:8px;padding:16px 18px;font-size:14px;color:var(--muted)}
.note b{color:var(--ink)}
code{font-family:var(--mono);font-size:12.5px;background:var(--panel);border:1px solid var(--line);
  padding:1px 6px;border-radius:5px;color:var(--ink)}
pre{font-family:var(--mono);font-size:12.5px;background:var(--panel);border:1px solid var(--line);
  border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
@media (max-width:640px){.ab,.depthrow{grid-template-columns:1fr}h1{font-size:30px}}
</style>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; host visual benchmark</div>
  <h1>Measuring 2D&rarr;3D quality on the real pipeline</h1>
  <p class="lede">Metrics computed on the actual side-by-side frames the headset receives &mdash;
  produced by the real depth estimator and warp shaders over fixed video clips, not a CPU replica.
  Every number is meant to move with a real change and be diffed against a baseline.</p>
  <div class="meta">
    <span>2026-07-07</span><span>movie mode &middot; da3mono-large + MLBW l4</span>
    <span>5 clips &middot; 30&nbsp;fps</span><span>tools/sbsbench</span>
  </div>

  <section>
    <h2>What each metric means</h2>
    <div class="grid defs">
      <div class="card"><h3><span class="name">pop</span></h3>
        <p>Horizontal L&harr;R disparity from tile phase-correlation between the eyes &mdash; the
        amount of stereo depth. Reported p50 / p95 in pixels and as % of eye width.</p>
        <div class="f">pop = median &#124;dx&#124; over textured tiles</div>
        <div class="dir">higher &rarr; more 3D depth</div></div>
      <div class="card"><h3><span class="name">vmisalign</span></h3>
        <p>Median vertical L&harr;R offset. Parallax must be purely horizontal, so this is a
        correctness check &mdash; it should stay at zero.</p>
        <div class="f">vmisalign = median &#124;dy&#124;</div>
        <div class="dir">must be &asymp; 0 (else geometry fault)</div></div>
      <div class="card"><h3><span class="name">disocc_frac</span></h3>
        <p>Fraction of the eye that sits in a narrow band beside a real depth silhouette &mdash;
        the region the warp had to invent because the foreground moved.</p>
        <div class="f">frac = area(bands beside top-0.7% depth edges)</div>
        <div class="dir">context for smear, not good/bad</div></div>
      <div class="card"><h3><span class="name">disocc_smear</span></h3>
        <p>Horizontal-detail deficit inside those bands &mdash; stretched or smeared fill loses
        horizontal texture. The disocclusion artifact severity.</p>
        <div class="f">smear = 1 &minus; &#124;dI/dx&#124;<sub>band</sub> / &#124;dI/dx&#124;<sub>clean</sub></div>
        <div class="dir">0 = fill as crisp as clean regions &middot; &rarr;1 worse</div></div>
      <div class="card"><h3><span class="name">flicker</span></h3>
        <p>Frame-to-frame change of the SBS luma &mdash; the temporal metric the single-frame
        offline sim cannot produce. On the same clip, shared motion cancels in a baseline diff,
        isolating added shimmer (e.g. depth instability, &frac14;-res inpaint re-hallucination).</p>
        <div class="f">flicker&#8345; = mean&#124;luma(F&#8345;) &minus; luma(F&#8345;&#8331;&#8321;)&#124; &times; 255</div>
        <div class="dir">lower &rarr; steadier &middot; validated &rarr;0 on a static clip</div></div>
      <div class="card"><h3><span class="name">depth_spread</span></h3>
        <p>p95&minus;p5 of the normalized depth map &mdash; how much depth range the model +
        normalization produced. Separates a flat model from a flat warp.</p>
        <div class="f">spread = depth.p95 &minus; depth.p5</div>
        <div class="dir">higher &rarr; more depth to work with</div></div>
    </div>
  </section>

  <section>
    <h2>Clip scorecard &mdash; 5 real clips, movie mode</h2>
    <p>Same pipeline, five different 1.6&nbsp;s clips. The benchmark discriminates content: it
    flags <b>c647</b> as unstable &mdash; near-flat depth (pop p50&nbsp;0) yet the highest flicker,
    the classic flat-content depth-hallucination failure, now a number.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>clip</th><th>pop p50</th><th>pop p95</th><th>pop&nbsp;%</th>
        <th>vmisalign</th><th>flicker p50</th><th>flicker p95</th><th>read</th></tr></thead>
      <tbody>
        <tr><td class="lab">c525</td><td>21.9</td><td>36.0</td><td>0.71</td><td>0.0</td><td>1.6</td><td>5.6</td><td><span class="pill p-good">strong 3D</span></td></tr>
        <tr><td class="lab">c747</td><td>13.7</td><td>18.0</td><td>0.45</td><td>0.0</td><td>1.6</td><td>4.0</td><td><span class="pill p-good">solid</span></td></tr>
        <tr><td class="lab">c339</td><td>7.4</td><td>28.7</td><td>0.24</td><td>0.0</td><td>3.6</td><td>11.0</td><td><span class="pill p-warn">some shimmer</span></td></tr>
        <tr><td class="lab">c841</td><td>0.0</td><td>23.8</td><td>0.00</td><td>0.0</td><td>0.5</td><td>1.0</td><td><span class="pill p-good">flat, stable</span></td></tr>
        <tr class="flag"><td class="lab">c647</td><td>0.0</td><td>32.4</td><td>0.00</td><td>0.0</td><td>8.6</td><td>14.7</td><td><span class="pill p-crit">unstable depth</span></td></tr>
      </tbody>
    </table></div>
    <p style="margin-top:16px;color:var(--muted);font-size:13.5px"><b style="color:var(--ink)">vmisalign
    = 0.0 across all 240 frames</b> &mdash; the geometry correctness check holds on every clip.</p>
  </section>

  <section>
    <h2>A/B example &mdash; control vs. treatment</h2>
    <p>The workflow you run per change: fix the clip, change one thing, diff. Here the lever is
    <b>divergence</b> (parallax strength) on clip <b>c525</b> &mdash; control at the shipped value,
    treatment at 2&times;. Same 32 input frames, so any delta is the change alone.</p>

    <div class="tablewrap" style="margin-bottom:22px"><table>
      <thead><tr><th>metric</th><th>control</th><th>treatment (2&times; div)</th><th>&Delta;</th><th></th></tr></thead>
      <tbody>
        <tr><td class="lab">pop p50 (px)</td><td>20.7</td><td>39.4</td><td>+90.0%</td><td><span class="pill p-info">lever working</span></td></tr>
        <tr><td class="lab">pop p95 (px)</td><td>34.7</td><td>64.9</td><td>+87.2%</td><td><span class="pill p-info">&mdash;</span></td></tr>
        <tr><td class="lab">vmisalign (px)</td><td>0.00</td><td>0.00</td><td>0.0%</td><td><span class="pill p-good">geometry clean</span></td></tr>
        <tr><td class="lab">disocc_frac</td><td>0.036</td><td>0.036</td><td>0.0%</td><td><span class="pill p-info">same silhouettes</span></td></tr>
        <tr><td class="lab">disocc_smear</td><td>0.00</td><td>0.00</td><td>0.0%</td><td><span class="pill p-good">no stretch</span></td></tr>
        <tr><td class="lab">flicker p50</td><td>1.22</td><td>1.46</td><td>+19.2%</td><td><span class="pill p-warn">more shimmer</span></td></tr>
      </tbody>
    </table></div>

    <figure class="full"><img src="__FULL__" alt="Full SBS frame, both eyes">
      <figcaption>Full SBS frame (control) &mdash; left &amp; right eye, 6132&times;1728. Crop site: the
      near hand against the misty pagoda background.</figcaption></figure>

    <div class="grid ab">
      <figure><div class="tag"><span class="pill p-info">control</span> divergence 0.0135</div>
        <img src="__CTRL__" alt="Control crop"></figure>
      <figure><div class="tag"><span class="pill p-warn">treatment</span> divergence 0.027 (2&times;)</div>
        <img src="__TREAT__" alt="Treatment crop"></figure>
    </div>

    <div class="grid depthrow">
      <figure><img src="__DEPTH__" alt="Depth at crop, silhouette marked">
        <figcaption>depth &middot; red = detected silhouette</figcaption></figure>
      <div class="note"><b>What the images show.</b> The hand shifts visibly further left under 2&times;
      divergence &mdash; the +90% pop is real, not a number artifact. Yet the disocclusion beside the
      hand fills <b>cleanly in both</b>: the revealed background is low-texture mist, so there is
      nothing to smear &mdash; and <code>disocc_smear</code> correctly reads&nbsp;0. The metric is
      content-aware; it flags stretch only where texture is actually lost.</div>
    </div>
  </section>

  <section>
    <h2>Reproduce</h2>
    <pre># 1. video -> fixed frame clip
python tools/sbsbench/split_video.py clip.mp4 -o clips/c525 --max 32

# 2. frames -> real SBS frames (run from the build dir; conf supplies warp/divergence)
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench \
    --frames clips/c525 --out out/control --movie
./sunshine.exe ... --frames clips/c525 --out out/treat --movie --divergence 0.027

# 3. score + diff
python tools/sbsbench/sbsbench.py --seq out/control --json base.json
python tools/sbsbench/sbsbench.py --seq out/treat --baseline base.json</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:14px">Harness: <code>src/sbs_bench_harness.cpp</code>
    (<code>--sbs-bench</code>). Metrics: <code>tools/sbsbench/sbsbench.py</code>. Per-stage perf is the
    separate in-app <code>sbs_3d_perf_stats</code> counter. Plan: <code>docs/sbs-benchmark-plan.md</code>.</p>
  </section>
</div>
"""

HTML = (HTML.replace("__FULL__", full).replace("__CTRL__", ctrl)
        .replace("__TREAT__", treat).replace("__DEPTH__", depth))
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_html, f"({len(HTML)//1024} KB)")

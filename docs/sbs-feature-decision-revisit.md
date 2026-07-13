# SBS feature-decision revisit after evaluator and pipeline fixes (historical)

> Archived experiment ledger. The VD3D production warp/profile and its reproduction tools were
> removed after the final headset decision. Paths below document old evidence and are not current
> commands; use `tools/sbsbench/README.md` for the active workflow.

Reviewed 2026-07-11 against eval schema 7 after the stereo-volume correction, source-relative
warp metrics, comfort/integrity gates, GT/flow validation, resolution normalization, expanded
public data, anime/AI clips, and color/HDR fixes.

Schema 7 excludes invalid metric-depth boundaries from GT edge scoring, regularizes the
source-relative correspondence field so per-pixel shift hopping cannot hide artifacts, and records
the named production profile in run provenance. A fresh full-core comparison preserves the prior
warp verdict: VD3D has one halo win, one halo cost, and one stability cost versus Apollo.

Schema 8 freezes the accepted processor stack: range→pixel ordering and Bestv2/P2–P98/P5–P95
constants are permanent, while rejected and ineffective processors were physically removed.
The cleanup preserves the comparison verdict: core remains a 1-win/2-cost tradeoff and extended
remains neutral. Reports: `sbs_eval/evalv8-cleanup-vd3d/report.html` and
`sbs_eval/evalv8-cleanup-extended-vd3d/report.html`.

## Decision rule

- Hard comfort or image-integrity failure rejects a treatment.
- Warp artifacts and stereo volume have equal priority. Stability and available GT depth evidence
  remain independent primary axes; a scalar score cannot cancel movement between them.
- Core candidates are checked on `extended-v2`. Mixed results are tuned or retained conservatively.
- Phase-A fidelity and Phase-B perceived quality are different decisions. A VD3D-matching feature
  can remain in `bestv2-phase-a.conf` without being enabled in either quality profile.

Controls:

- Core: `sbs_eval/revisit-control2-{apollo,vd3d}` (11 clips, including `flat_transition`).
- Public: `sbs_eval/revisit-extended-control-{apollo,vd3d}` (8 prepared public clips).
- Final saved profiles: `sbs_eval/revisit-final-profile-{apollo,vd3d}`.

## Revised active-stack ledger

| Feature / processor | Apollo quality | VD3D quality | Revised evidence |
|---|---|---|---|
| Bestv2 shift field | **keep** | **keep** | Switching to Apollo shift loses stereo volume on all 10 ordinary core clips and produces 17/19 primary costs. |
| Depth short side 432 | **keep** | **keep** | 336 causes 6 costs/1 win on Apollo and 4 costs/0 wins on VD3D across stability, warp and stereo. |
| P2/P98 normalization | **keep for fidelity; quality-neutral** | same | Raw min/max is neutral on core and public schema-6 decisions. P2/P98 expands pre-warp p95-p5 spread by ~12-13% rather than crushing it, but final stereo is unchanged and raw has slightly lower GT RMSE on 7/8 public clips. Do not call P2/P98 a quality improvement. |
| Range-bounds EMA 0.18 | **keep** | **keep** | Removing it causes a warp cost on Apollo and warp+stability costs on VD3D, with no win. |
| Per-pixel EMA 0.5 | **keep (tradeoff)** | **keep (tradeoff)** | On the final profiles, removal has mixed stability and adds 1/2 warp costs. |
| EMA ordering | **range → pixel, permanent** | **range → pixel, permanent** | Core gives 4/3 stability wins and no costs; extended is neutral. The rejected ordering and its configuration were removed. |
| Subject tracking / band field | **keep** | **keep** | Disabling the whole processor loses stereo volume on all 10 ordinary core clips for both warps, despite some warp wins. |
| Subject stretch | **keep** | **keep pending stronger contrary evidence** | Removal causes 3 stability costs/0 wins on Apollo. VD3D removal is mixed (4 wins/5 costs). |
| Subject lock | **0.5** | **0.5** | Old 0.95 is no longer accepted. Zero lock gives large warp wins but one extended Sintel stereo cost. At 0.5, Apollo extended has 3 warp wins/0 costs; VD3D core has 10 wins/0 costs and extended 4/0. |
| Subject recenter 0.35 | **keep** | **keep** | Apollo removal is favorable on core but only 1 halo win/1 stereo cost on extended. VD3D removal has more costs than wins. |
| Guided upsample | **removed** | **removed** | Rejected/tradeoff evidence showed no net benefit; implementation and configuration removed. |
| Exact subject plane lock 0.28 | **off / rejected** | **off / rejected** | Schema-6 tradeoff is cost-dominated: Apollo 4 wins/12 costs; VD3D 2/13, including stereo and warp costs. |
| Foreground curvature 0.07 | **removed** | **removed** | Apollo was below sensitivity and VD3D regressed stability; implementation removed. |
| Bestv2 sharpen 0.2 | **off / hard reject** | **off / hard reject** | Fails ≥90% source coverage on seven core clips for both geometries; several fall to about 74–80%. |
| Scene-cut min/max snap 1.6 | **removed** | **removed** | No validated movement, including the hard-cut sequence. |
| Range floor 0.5 | **removed** | **removed** | Ineffective because DA-V2 retains a large hallucinated raw range on flat content. |
| Depth floor 0.25 | **removed** | **removed** | It was unreachable under the permanent Bestv2 field. |
| Border fade 0.02 | **removed** | **removed** | No validated primary movement on either geometry. |
| Bestv2 smear suppression + Fast repair | **off / rejected** | **off / rejected** | Schema-6 core: Apollo gains 0.36 px source-relative halo but loses 0.28 px rim overshoot and adds disocclusion flicker; VD3D gains only 0.11 px halo while losing 0.41 px rim. The Fast repair component is byte-identical on the strongest changed clip; changes come from the 60% blend-to-flat smear stage. Literal implementation costs ~17.2 ms/frame. All 264 raw and pre-warp depth artifacts per geometry remain byte-identical. |
| Bestv2 cinematic-window sculpt | **off / headset candidate** | **off / rejected** | Apollo core has 6 warp wins/0 costs and extended 3/0, but normal-scale visual inspection is inconclusive, median pop falls ~10% core/~3% public, and rim overshoot rises. VD3D public has one stretch win/one halo cost, with lower stereo spread and more rim. Do not enable Apollo without a headset A/B confirming the subtle artifact redistribution is preferable. |
| Exact Bestv2 DOF 0.3 | **off / rejected** | **off / rejected** | Schema-6 core is neutral on both profiles. Maximum per-frame mean pixel change is only ~0.0075/255 (peak 2/255), with no visible or validated benefit; GPU cost is ~0.017 ms. All 264 raw and pre-warp depth artifacts remain byte-identical per geometry. |
| VD3D forward blend | n/a | **0.35** | 1.0 is cost-dominated; 0.0 is mixed. At 0.35 with the revised stack, core has 10 wins/0 costs and extended 4/0. |

## Final quality profiles

Both profiles permanently use Bestv2 shift, 432 depth, P2/P98, range→pixel EMA, range EMA 0.18,
per-pixel EMA 0.5, and subject tracking/stretch/recenter. Rejected processors listed as removed
above no longer have dormant production branches.

Differences and revised values:

| setting | Apollo | VD3D |
|---|---:|---:|
| warp | `apollo` | `vd3d` |
| EMA ordering | range → pixel | range → pixel |
| subject lock | `0.5` | `0.5` |
| subject recenter | `0.35` | `0.35` |
| VD3D forward blend | n/a | `0.35` |

Saved-profile core results versus the former profiles:

- Apollo: 10 wins / 2 warp costs; five stability wins; no stereo cost or hard failure.
- VD3D: 10 wins / 0 costs across warp, stereo and stability; no hard failure.

Public midpoint evidence:

- Apollo: 3 warp wins / 0 costs.
- VD3D: 4 warp wins / 0 costs.
- Visual inspection of the report-selected TartanAir motion frames found the reported halo/
  residual reduction without a new obvious hole, line, clipping, or eye-asymmetric corruption.

Reports:

- `sbs_eval/revisit-final-profile-apollo/report.html`
- `sbs_eval/revisit-final-profile-vd3d/report.html`
- `sbs_eval/revisit-ext-lock50-stack-apollo/report.html`
- `sbs_eval/revisit-ext-lock50-stack-vd3d/report.html`

Every newly generated report also writes `decision.json`. The HTML and JSON reuse the same single
`evaluate_ab_decision(ctrl_agg, treat_agg, ...)` result; consumers must not reconstruct a verdict
from the outer per-clip result wrappers. `audit_depth_transform.py` writes a separate native-depth
spread/saturation audit for processors that can reshape depth.

## Removed processors: schema-6 revalidation history

These implementations were deleted before the corrected evaluator existed. Their old decisions
must not be presented as current schema-6 evidence:

| Removed processor | Old status | Current status |
|---|---|---|
| Bestv2 cinematic-window sculpt | rejected and reverted | **schema-6 Apollo headset candidate; VD3D rejected; remains removed** |
| VD3D conceal/smear suppression + Fast repair | rejected and removed | **schema-6 rejected on both quality profiles; removed again** |
| Exact Bestv2 DOF | rejected and removed | **schema-6 neutral/rejected on both profiles; removed again** |
| MLBW learned field / old alternate warp | removed architecturally | **obsolete unless explicitly reconsidering a third warp** |

The three still-relevant removed processors have now been re-evaluated. Only Apollo window sculpt
reaches automated-candidate status, and it remains off pending a headset preference A/B.

Concealment reports:

- `sbs_eval/revisit-conceal-v3-fast-apollo/report.html`
- `sbs_eval/revisit-conceal-v3-fast-vd3d/report.html`

Window-sculpt reports:

- `sbs_eval/revisit-window-v3-apollo/report.html`
- `sbs_eval/revisit-window-v3-vd3d/report.html`
- `sbs_eval/revisit-window-v3-ext-apollo/report.html`
- `sbs_eval/revisit-window-v3-ext-vd3d/report.html`

DOF reports:

- `sbs_eval/revisit-dof-v3-apollo/report.html`
- `sbs_eval/revisit-dof-v3-vd3d/report.html`

## Final warp comparison after the revisit

The two saved quality profiles are not measurably separated on overall quality. On core, VD3D has
one source-halo win (`c525`), one source-halo cost (`c747`), and one stability cost (`c647`). On
the eight-clip public suite there are no validated primary differences. Aggregate clean score is
only +0.05 for VD3D on both suites. Stereo spread is +1.4% on core and -0.05% on public, so the
revised profiles are volume-matched within normal noise/tolerance.

VD3D is the performance winner: its geometry measures about 0.019 ms versus 0.141 ms for Apollo
on both suites (~7.4x faster). This is not enough evidence to delete Apollo because quality remains
a tradeoff on core and neutral on public. Retain both paths until headset A/B testing resolves the
different halo/rim/stability character; use VD3D when geometry cost is the deciding constraint.

Schema-7 profile-contract report:

- `sbs_eval/evalv7-profile-vd3d/report.html` (control: `evalv7-profile-apollo`)

Earlier schema-6 final reports:

- `sbs_eval/final-warp-comparison-core/report.html`
- `sbs_eval/final-warp-comparison-extended/report.html`

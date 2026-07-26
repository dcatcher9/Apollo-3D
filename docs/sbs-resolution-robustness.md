# SBS resolution and aspect-ratio review

Updated after removal of the historical VD3D geometry; current validation applies to Apollo only.

This review follows each coordinate space from captured mono pixels to the encoded SBS image. The
invariants are:

- Depth inference preserves the captured image aspect as closely as the model's 14-pixel patch grid
  permits.
- Depth-only filters are calibrated in model-grid or normalized coordinates, never encoder pixels.
- Bestv2 pixel parameters are interpreted in mono-source pixels and converted once at output.
- Each SBS eye receives the same centered aspect fit; packing cannot create false disparity.
- Host SBS accepts identity-oriented capture only. Portrait uses an explicit `W < H` display mode;
  non-identity Windows display rotation is rejected before shader setup.
- Evaluation gates use eye-relative disparity/alignment values. Pixel values remain diagnostic.

## Processor audit

| Processor | Coordinate space | Resolution/aspect handling |
|---|---|---|
| TensorRT input | model patches | The short side is budgeted, then the long side is derived from source aspect. Both are selected together under native and engine limits. A 5120x2160 source becomes 994x420, not the distorted 1008x420 grid. |
| Raw-depth normalization and EMA | model grid / depth values + 434-reference texels | Histogram, percentile, min/max and temporal state depend on sample count or depth values, not encoder size. The moving-edge mask's spatial-gradient gate uses the same actual-short-side / 434 transfer as the subject/adaptive-pop pass; its depth-change gate remains resolution-independent. |
| Subject histogram and recenter/stretch | normalized image/depth + 434-reference texels | Center weighting uses normalized x/y. Spatial depth gradients are multiplied by the actual resolved short side / 434 before both silhouette suppression and adaptive-pop weighting, so changing `depth_short_side` does not silently move the calibrated 0.02/0.025 thresholds. The adaptive-pop saturation cap receives the same scale. Depth bands and convergence otherwise operate in normalized depth or Bestv2 source-pixel units. |
| Apollo inverse warp | source UV | Bestv2 shifts and search bounds use the validated 854px calibration width for wider sources, preserving eye-relative stereo volume on 5120px desktops; smaller sources retain literal preset pixels. Depth tap size comes from the depth texture. |
| Encoder conversion/downscale | encoded raster | The SBS intermediate is already the final encoder size. There is no second post-warp rescale; YUV conversion sees the final packed dimensions. |
| Eval metrics/report | eye-relative/common raster | Stereo volume and vertical alignment gates use percentages. A/B evidence normalizes mismatched run sizes to a common per-eye raster before crops or heatmaps. |

The 854-pixel calibration is an internal resolution correction, not a user tuning knob.
`sbs_3d_pop_strength` is the final shared parallax multiplier (`0.25`-`2`, default `1.20`) and
changes apparent depth consistently after resolution normalization. The default scene latch may
select up to `sbs_3d_adaptive_pop_max = 2.00`; the selected value and median zero plane are held
between detected geometry cuts, so resolution changes and exposure changes do not alter them.
Appearance evidence combines broad RGB replacement with exposure-invariant ordinal structure and
must still be corroborated by depth before it can pulse shot state. Geometry and appearance rearm
independently; a depth-motion EMA admits a later geometry edge without allowing steady persistent
evidence to periodically reset convergence. Galaxy XR validation is specified in the
[persistent-motion test plan](sbs-persistent-motion-test-plan.md).

Adaptive-pop risk is calibrated at the resolved 434-texel short side (`depth_short_side = 434`, or
nearby requests rounded to the 14-pixel model patch). The producer uses
`min(target_w, target_h) / 434`, where `target_w/target_h` are the actual post-cap tensor
dimensions. It applies that factor to the finite-difference gradient and to the 8x saturation cap.
For a one-texel step, the per-edge weight then changes inversely with the edge-pixel fraction; the
aggregate risk stays stable. The deterministic CPU reference test covers the supported 16:9
`700x392`, `742x420`, and `770x434` grids in both the linear and saturated regimes; patch-aligned
long-side rounding leaves less than 0.7% residual.

## Validation

- Production `sunshine` RelWithDebInfo build succeeds.
- Run the canonical SBS evaluator suite with
  `python -m unittest discover -s tools/sbsbench -p "test_*.py"`; optional learned-oracle tests
  skip when their external runtime is unavailable.
- Real D3D11 harness compilation/execution succeeds for Apollo at 80% output scale.
- Letterbox smoke: both eyes had identical 30-row top/bottom bars.
- Pillarbox smoke: both eyes had identical 49-column left/right bars.
- 5120x2160 synthetic target smoke, native versus the 8192 packed-width cap:
  - Apollo normalized stereo spread changed by about +1.3%.
  - Pixel disparity fell by about 19%, as expected for a 20% linear downscale.
- At 80% scale Apollo changes normalized stereo spread by about -2.3% on the small core clips.
  Do not compensate for raster scaling by increasing divergence; that would overdrive the real
  high-resolution stream.

Generated comparison reports:

- `cmake-build-relwithdebinfo/sbs_eval/resrobust-apollo-scale80/report.html`

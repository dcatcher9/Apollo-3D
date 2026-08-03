# SBS resolution and aspect-ratio review

Updated for the Host SBS Depth Coordinate V2 cutover. Historical Bestv2 normalization and subject
controls remain available to offline conversion and the evaluator, but are not live geometry.

This review follows each coordinate space from captured mono pixels to the encoded SBS image. The
invariants are:

- Depth inference preserves the captured image aspect as closely as the model's 14-pixel patch grid
  permits.
- The authenticated live tensor family preserves source aspect on its patch-aligned grid.
- Raw DAV2 coordinates use the authenticated model/shape scale, never frame endpoint normalization.
- Each SBS eye receives the same centered aspect fit; packing cannot create false disparity.
- Host SBS accepts identity-oriented capture only. Portrait uses an explicit `W < H` display mode;
  non-identity Windows display rotation is rejected before shader setup.
- Evaluation gates use eye-relative disparity/alignment values. Pixel values remain diagnostic.

## Processor audit

| Processor | Coordinate space | Resolution/aspect handling |
|---|---|---|
| TensorRT input | authenticated model patches | Standard landscape, ultrawide, and portrait streams select one of six authenticated grids: `770x434`, `1022x434`, `1036x434`, and their transposes. Configuration cannot authorize another live shape or model. |
| V2 raw coordinate | DAV2 raw units | A fixed authenticated scale maps raw DAV2 Small output around the scene-latched center. Frame min/max, percentiles, subject recentering, and adaptive-pop values do not rescale live geometry. |
| Private cut evidence | normalized model grid + 434-reference texels | A separate cut-only path uses fixed EMA/min-max calibration. Only the moving-edge EMA gradient gate is scaled to the actual short side; depth-change and appearance evidence are valid-texel fractions. The private normalized depth never reaches the V2 geometry mapper or renderer. |
| V2 candidate and conditioner | normalized source-U displacement | The literal requested pop feeds a frame-local 4% source-U container. Exact vertical shear-2 upper/lower envelopes are shared 75/25 before a pure slope-0.5 row majorant; both bounds are independent of encoder resolution. |
| V2 inverse renderer | source UV | Both eyes solve the same bounded final field with opposite signs and a unique 12-step contractive inverse, then take one native linear color sample. There is no collar filter, V1 probe spacing, forward owner, or synthetic fill. |
| Encoder conversion/downscale | encoded raster | The SBS intermediate is already the final encoder size. There is no second post-warp rescale; YUV conversion sees the final packed dimensions. |
| Eval metrics/report | eye-relative/common raster | Stereo volume and vertical alignment gates use percentages. A/B evidence normalizes mismatched run sizes to a common per-eye raster before crops or heatmaps. |

`sbs_3d_pop_strength` is the literal live parallax request (`0.25`-`2`, default `1.20`). Live V2
does not add an adaptive-pop multiplier or use the configured legacy zero plane. It acquires its
scene center and near-tail shoulder on the first usable field and again on a confirmed cut, then
holds those values through the shot. Configured Base/custom models, tensor-size tuning,
adaptive-pop, subject, normalization, and zero-plane controls remain offline/evaluator options.
Appearance evidence combines broad RGB replacement with exposure-invariant ordinal structure and
must still be corroborated by depth before it can pulse shot state. Geometry and appearance rearm
independently; a depth-motion EMA admits a later geometry edge without allowing steady persistent
evidence to periodically reset convergence. Galaxy XR validation is specified in the
[persistent-motion test plan](sbs-persistent-motion-test-plan.md).

Private normalized-depth history uses `min(target_w, target_h) / 434` only for the moving-edge
EMA gradient gate. Depth-change and appearance fractions are normalized by valid texel count. The
old weighted edge-risk accumulator and its saturation cap remain in the legacy offline/evaluator
analysis path and are not dispatched by live Host SBS V2.

## Validation

The measurements below are retained historical evaluator and packing evidence. Current live V2
release evidence must additionally authenticate the DAV2 Small model, one of the six live tensor
shapes, the V2 producer/renderer closures, and the cut-only analysis path.

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

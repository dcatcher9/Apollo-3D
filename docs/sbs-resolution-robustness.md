# SBS resolution and aspect-ratio review

Updated after removal of the historical VD3D geometry; current validation applies to Apollo only.

This review follows each coordinate space from captured mono pixels to the encoded SBS image. The
invariants are:

- Depth inference preserves the captured image aspect as closely as the model's 14-pixel patch grid
  permits.
- Depth-only filters are calibrated in model-grid or normalized coordinates, never encoder pixels.
- Bestv2 pixel parameters are interpreted in mono-source pixels and converted once at output.
- Each SBS eye receives the same centered aspect fit; packing cannot create false disparity.
- Evaluation gates use eye-relative disparity/alignment values. Pixel values remain diagnostic.

## Processor audit

| Processor | Coordinate space | Resolution/aspect handling |
|---|---|---|
| TensorRT input | model patches | The short side is budgeted, then the long side is derived from source aspect. Both are selected together under native and engine limits. A 5120x2160 source becomes 994x420, not the distorted 1008x420 grid. |
| Raw-depth normalization and EMA | model grid / depth values | Histogram, percentile, min/max and temporal state depend on sample count or depth values, not encoder size. No output-resolution adjustment is needed. |
| Subject histogram and recenter/stretch | normalized image/depth | Center weighting uses normalized x/y. Depth bands and convergence operate in normalized depth or Bestv2 source-pixel units. |
| Apollo inverse warp | source UV | Bestv2 shifts and search bounds use the validated 854px calibration width for wider sources, preserving eye-relative stereo volume on 5120px desktops; smaller sources retain literal preset pixels. Depth tap size comes from the depth texture. |
| Encoder conversion/downscale | encoded raster | The SBS intermediate is already the final encoder size. There is no second post-warp rescale; YUV conversion sees the final packed dimensions. |

The 854-pixel calibration is an internal resolution correction, not a user tuning knob.
`sbs_3d_pop_strength` is the final shared parallax multiplier (`0.25`-`2`, default `1.25`) and
changes apparent depth consistently after resolution normalization. The default scene latch may
select up to `sbs_3d_adaptive_pop_max = 1.30`; the selected value is held until a hard cut, so
resolution changes do not alter it and ordinary motion cannot make it breathe.
| Eval metrics/report | eye-relative/common raster | Stereo volume and vertical alignment gates use percentages. A/B evidence normalizes mismatched run sizes to a common per-eye raster before crops or heatmaps. |

## Validation

- Production `sunshine` RelWithDebInfo build succeeds.
- All 94 SBS eval tests pass.
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

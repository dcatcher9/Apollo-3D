# SBS resolution and aspect-ratio review

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
| Exact plane lock | model grid | Bestv2's morphology radii now scale from its 336-pixel-short-side reference, keeping the same image-relative support on other inference grids. |
| Guided upsample | model grid to source grid | Output is at most 2x the model grid and never larger than the source. This avoids changing edge behavior by supersampling small inputs. |
| Foreground curvature | normalized image/depth | Radial position uses both output dimensions independently. It follows the actual guided/raw depth texture size. |
| Apollo inverse warp | source UV | Bestv2 shifts and search bounds are normalized by mono-source width. Depth tap size comes from the depth texture. Per-eye aspect fitting is applied before reprojection. |
| VD3D forward warp | output-eye raster + source UV | Parallax uses mono-source width; the output shift includes fitted content width. Hole-fill radius converts source pixels to output-content pixels. Integer splat placement still has unavoidable subpixel quantization on very small rasters. |
| VD3D backward blend | source UV | Uses the same shared disparity field and per-eye aspect transform as Apollo. |
| Bestv2 sharpen | source-pixel footprint | Kernel taps now use fractional output sampling at one mono-source-pixel spacing, instead of always one output pixel. Eyes and aspect-fit bars remain isolated. |
| Encoder conversion/downscale | encoded raster | The SBS intermediate is already the final encoder size. There is no second post-warp rescale; YUV conversion sees the final packed dimensions. |
| Eval metrics/report | eye-relative/common raster | Stereo volume and vertical alignment gates use percentages. A/B evidence normalizes mismatched run sizes to a common per-eye raster before crops or heatmaps. |

## Validation

- Production `sunshine` RelWithDebInfo build succeeds.
- All 38 SBS eval tests pass.
- Real D3D11 harness compilation/execution succeeds for VD3D at 80% output scale.
- Letterbox smoke: both eyes had identical 30-row top/bottom bars.
- Pillarbox smoke: both eyes had identical 49-column left/right bars.
- 5120x2160 synthetic target smoke, native versus the 8192 packed-width cap:
  - Apollo normalized stereo spread changed by about +1.3%.
  - VD3D normalized stereo spread changed by about +1.2%.
  - Pixel disparity fell by about 19%, as expected for a 20% linear downscale.
- The committed small core clips exaggerate VD3D integer-splat quantization: at 80% scale Apollo
  changes normalized stereo spread by -2.3% on average, while VD3D changes by -13.2%. This is a
  raster-resolution limitation of the forward component, not the behavior at the 5120-to-4096
  deployment scale. It should not be compensated by increasing divergence, which would overdrive
  the real high-resolution stream.

Generated comparison reports:

- `cmake-build-relwithdebinfo/sbs_eval/resrobust-apollo-scale80/report.html`
- `cmake-build-relwithdebinfo/sbs_eval/resrobust-vd3d-scale80/report.html`

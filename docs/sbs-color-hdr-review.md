# SBS color and HDR review

Reviewed 2026-07-11. Scope: Windows D3D11 capture, depth preprocessing and guidance, both SBS
warps, debug/eval output, YUV conversion, and encoder signaling.

## Result

The SBS geometry stages now preserve the source's color encoding and precision. The important
distinction is not merely "HDR on/off": Windows can provide linear FP16 for 10-bit SDR as well as
for HDR. The pipeline therefore carries three explicit depth-input modes:

| Capture | Texture | Depth-model input | SBS intermediate | Encode conversion |
|---|---|---|---|---|
| 8-bit SDR | BGRA UNORM, display-referred sRGB | clamp only | BGRA UNORM | RGB to requested SDR YUV |
| 10-bit SDR | RGBA FP16, linear Rec.709 | sRGB OETF | RGBA FP16 | linear to sRGB, then SDR YUV |
| HDR | RGBA FP16, linear scRGB/Rec.709 | luminance tone map, then sRGB OETF | RGBA FP16 | Rec.709 to Rec.2020, 80 nits per scRGB unit, ST 2084 PQ, then YUV |

Apollo and VD3D use the same color handling. Their only intended difference is geometry/hole
resolution; neither warp applies a color transfer function or clamps FP16 HDR samples.

## Corrections made

1. Depth inference no longer hard-clamps linear HDR or feeds it to an SDR-trained model as if it
   were gamma encoded. It now uses a luminance-preserving Reinhard mapping and the exact sRGB OETF.
2. Guided depth uses that same transform at both guide resolutions. Its range sigma now has a
   consistent display-referred domain across SDR and HDR.
3. 10-bit SDR FP16 is no longer selected from the gamma-input conversion shaders merely because
   the display itself is SDR. Conversion is selected from the actual capture texture encoding.
4. The SBS intermediate is FP16 for all linear capture, and Bestv2 sharpen is disabled for linear
   input because that shader is an SDR display-referred operation and clamps its output.
5. HDR debug PNGs no longer use independent R/G/B Reinhard curves, which could show false hue
   shifts. They use the same luminance-preserving diagnostic map. Linear SDR dumps receive only
   the sRGB OETF.
6. The offline D3D11 harness can simulate FP16 scRGB input and records output min/max and non-finite
   component counts. This is a pre-encode color-path smoke, not a colorimetric HDR evaluation.

## Audited stages

- **Capture:** WGC requests FP16 when the client asks for 10-bit. Desktop Duplication prefers
  FP16 for Advanced Color and falls back to 8-bit formats for SDR.
- **Depth:** preprocessing and guided filtering share `depth_color.hlsl`. Negative wide-gamut
  scRGB components are mapped into the model gamut; source pixels used for rendering remain
  untouched.
- **Warp:** source RGB is sampled and copied without a transfer/gamut conversion. Empty regions are
  black in the same linear or encoded domain. FP16 remains unclamped through Apollo and VD3D.
- **Post-warp sharpen:** enabled only for BGRA SDR. It remains deliberately absent for linear SDR
  and HDR pending a properly linear-light or display-referred implementation.
- **HDR conversion:** the existing shader converts linear Rec.709/scRGB to Rec.2020, uses the
  Windows scRGB definition of 1.0 = 80 nits, applies ST 2084 PQ, then the selected range/matrix.
- **YUV range:** 8-bit paths use normalized matrix/range constants. 10-bit packed/planar paths use
  integer-domain H.273-style coefficients and rounding. Limited/full range follows the client CSC
  mode.
- **Signaling:** HDR selects 10-bit BT.2020 non-constant-luminance with SMPTE 2084 in FFmpeg/NVENC
  VUI. Apollo also sends display mastering metadata through its HDR control event; MaxCLL/MaxFALL
  remain zero because DXGI does not expose content-specific values.

## Validation

- RelWithDebInfo production build: pass.
- Python evaluator/unit suite: 41/41 pass.
- Normal SDR c525 baseline gate: pass.
- Simulated HDR, Apollo warp: output range 0..4 scRGB, zero NaN/Inf components.
- Simulated HDR, VD3D warp: output range 0..4 scRGB, zero NaN/Inf components.
- Visual inspection of both tone-mapped SBS outputs: expected color and highlight structure; no
  new hue discontinuity or clipping was visible.

## Remaining limitations and live checks

1. The harness ends before Rec.2020/PQ YUV conversion and NVENC. Final HDR accuracy must be checked
   on the Galaxy XR with an HDR display active. Use a clip containing neutral gray, saturated
   primaries, diffuse white, and specular highlights; compare SBS off/on without changing the
   client HDR mode.
2. Debug PNGs are intentionally tone-mapped SDR previews. They can reveal clipping, NaN/Inf, hue
   discontinuities, and warp artifacts, but cannot prove PQ luminance accuracy.
3. `encoderCscMode` can request Rec.601 or BT.2020 SDR for a Rec.709 desktop. The inherited Apollo/
   Sunshine SDR shaders change the YUV matrix and metadata but do not perform a full RGB-primary
   gamut conversion; this is protocol-compatible legacy behavior, not unique to SBS. Rec.709 is
   the colorimetrically matched SDR mode.
4. HDR mastering primaries are deliberately signaled as Rec.2020 after conversion, while display
   min/max luminance comes from DXGI. Content light levels are unavailable and are not guessed.
5. The cursor uses a fixed 300-nit HDR white target. Windows now exposes an SDR-white-level query on
   supported systems, so this can be calibrated in a separate cursor change; it does not affect
   source video, depth, or either warp.

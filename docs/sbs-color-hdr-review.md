# SBS color and HDR review

Reviewed 2026-07-11 and updated 2026-07-22 after the host color/HDR correctness pass. Scope:
Windows D3D11 capture, depth preprocessing, Apollo SBS warp, local AR presentation, debug/eval
output, YUV conversion, HDR negotiation, and native NVENC signaling.

## Result

The SBS geometry stages now preserve the source's color encoding and precision. The important
distinction is not merely "HDR on/off": Windows can provide linear FP16 for 10-bit SDR as well as
for HDR. The pipeline therefore carries three explicit depth-input modes:

| Capture | Texture | Depth-model input | SBS intermediate | Encode conversion |
|---|---|---|---|---|
| 8-bit SDR | BGRA UNORM, display-referred sRGB | existing sRGB code values | BGRA UNORM | sRGB EOTF, target Rec.709/Rec.2020 primaries and BT.709/BT.2020 OETF, then requested SDR YUV |
| 10-bit SDR | RGBA FP16, linear Rec.709 | sRGB OETF | RGBA FP16 | target Rec.709/Rec.2020 primaries and BT.709/BT.2020 OETF, then requested SDR YUV |
| HDR | RGBA FP16, linear scRGB/Rec.709 | luminance tone map, then sRGB OETF | RGBA FP16 | Rec.709 to Rec.2020, 80 nits per scRGB unit, ST 2084 PQ, then YUV |

Apollo's warp does not apply a color transfer function or clamp FP16 HDR samples. Transfer and
gamut conversion happen at the final encode boundary, after the warp.

## Corrections made

1. Depth inference no longer hard-clamps linear HDR or feeds it to an SDR-trained model as if it
   were gamma encoded. It now uses a luminance-preserving Reinhard mapping and the exact sRGB OETF.
2. Guided depth uses that same transform at both guide resolutions. Its range sigma now has a
   consistent display-referred domain across SDR and HDR.
3. 10-bit SDR FP16 is no longer selected from the gamma-input conversion shaders merely because
   the display itself is SDR. Conversion is selected from the actual capture texture encoding.
4. The SBS intermediate is FP16 for all linear capture. The rejected post-warp Bestv2 sharpen was
   removed, eliminating its SDR-only clamp and its separate intermediate.
5. HDR debug PNGs no longer use independent R/G/B Reinhard curves, which could show false hue
   shifts. They use the same luminance-preserving diagnostic map. Linear SDR dumps receive only
   the sRGB OETF.
6. The offline D3D11 harness can simulate FP16 scRGB input and records output min/max and non-finite
   component counts. This is a pre-encode color-path smoke, not a colorimetric HDR evaluation.
7. BGRA8 SDR is no longer treated as if its sRGB code values were BT.709 code values. Rec.601/
   Rec.709 output uses a no-`pow` bounded polynomial for the composed sRGB-to-BT.709 transfer
   (maximum code error below `3.7e-5`); Rec.2020 SDR performs the full linear-light gamut and
   BT.2020 OETF conversion.
8. Chroma filtering now converts every RGB tap to the declared target transfer function and gamut
   before averaging. This prevents transfer-after-subsampling errors at high-contrast color and
   luminance edges for BGRA Rec.601/709/2020, linear FP16 SDR, and HDR/PQ alike.
9. NV12/P010 padding bars now use the selected matrix/range offsets, so limited-range black and
   neutral chroma are encoded at their proper code values instead of full-range zero.
10. HDR-to-SDR fallback conversion is eligible only in the linear FP16 shader selected from the
    actual frame format. BGRA remains display-referred SDR even when its physical display has HDR
    enabled, preventing WGC SDR capture from being tone-mapped a second time. Eligible FP16 input
    uses the source display's SDR reference-white setting and a luminance-preserving shoulder
    instead of hard clipping highlights.
11. HDR cursor bitmaps are decoded from sRGB before entering linear scRGB. Their white level comes
    from `DISPLAYCONFIG_GET_SDR_WHITE_LEVEL`, with a 203-nit fallback only when Windows cannot
    provide it.
12. DXGI output descriptions are zero-initialized and checked before their color state or mastering
    values are consumed. The same validated metadata snapshot drives Apollo's HDR control event and
    native NVENC HEVC/AV1 mastering-display payloads.
13. HTTP launch HDR mode and RTSP dynamic range must agree exactly for a real capture session.
    Either mismatch direction fails closed before capture/encode setup.
14. The local AR presenter selects its RGB shader from the swap-chain transfer contract: G10 linear
    scRGB for HDR and G22 Rec.709 for SDR. A physical HDR output can legitimately host the SDR
    swap-chain; DWM performs that presentation mapping.

## Audited stages

- **Capture:** WGC requests FP16 when the client asks for 10-bit. Desktop Duplication prefers
  FP16 for Advanced Color and falls back to 8-bit formats for SDR.
- **Depth:** preprocessing and guided filtering share `depth_color.hlsl`. Negative wide-gamut
  scRGB components are mapped into the model gamut; source pixels used for rendering remain
  untouched.
- **Warp:** source RGB is sampled and copied without a transfer/gamut conversion. Empty regions are
  black in the same linear or encoded domain. FP16 remains unclamped through Apollo.
- **Post-warp:** no additional color operation; the rejected Bestv2 sharpen was removed.
- **SDR conversion:** BGRA8 sRGB and linear FP16 are converted to the transfer and primaries declared
  by the output VUI. Rec.601 and Rec.709 share the BT.709 OETF here; Rec.2020 SDR includes the
  Rec.709-to-Rec.2020 linear-light gamut matrix and BT.2020 10-bit OETF.
- **HDR-to-SDR:** if an HDR source reaches an SDR conversion boundary during a mode transition, the
  shader normalizes by the queried SDR reference white, applies a smooth luminance shoulder, and
  uniformly fits residual out-of-gamut peaks without changing hue.
- **HDR conversion:** the existing shader converts linear Rec.709/scRGB to Rec.2020, uses the
  Windows scRGB definition of 1.0 = 80 nits, applies ST 2084 PQ, then the selected range/matrix.
- **YUV range:** 8-bit paths use normalized matrix/range constants. 10-bit packed/planar paths use
  integer-domain H.273-style coefficients and rounding. Limited/full range follows the client CSC
  mode. Padding clears use those same constants.
- **Chroma:** every source tap is converted into the target transfer function and gamut before
  target-encoded RGB is averaged for 4:2:0. This ordering applies to both chroma sitings and all
  BGRA, linear FP16 SDR, and HDR/PQ shader variants.
- **Signaling:** HDR selects 10-bit BT.2020 non-constant-luminance with SMPTE 2084 in FFmpeg/NVENC
  VUI. Apollo sends the validated display metadata through its HDR control event and embeds
  codec-unit-correct mastering-display metadata in native NVENC HEVC/AV1 bitstreams. Content-light
  metadata is emitted only when DXGI supplies nonzero content values; unavailable MaxCLL/MaxFALL
  values are not guessed.
- **Local AR:** the presentation swap-chain declares the target transfer explicitly. Exact-format
  frames use the copy path; shader fallback converts linear input to G22 only for an SDR target,
  independent of the physical output's Advanced Color state.

## Validation

- RelWithDebInfo production build: pass.
- DirectX compilation test covering BGRA8, FP16 SDR, PQ, planar luma, both chroma sitings, and HDR
  cursor variants: pass.
- Dense 16-bit-code transfer approximation, HDR negotiation, native HEVC/AV1 metadata-unit, and
  full/limited-range vector tests: pass.
- Python evaluator/unit suite: 71/71 pass.
- Normal SDR c525 baseline gate: pass.
- Simulated HDR, Apollo warp: output range 0..4 scRGB, zero NaN/Inf components.
- Visual inspection of the tone-mapped Apollo SBS output: expected color and highlight structure; no
  new hue discontinuity or clipping was visible.

## Remaining limitations and live checks

1. The harness ends before Rec.2020/PQ YUV conversion and NVENC. Final HDR accuracy must be checked
   on the Galaxy XR with an HDR display active. Use a clip containing neutral gray, saturated
   primaries, diffuse white, and specular highlights; compare SBS off/on without changing the
   client HDR mode.
2. Debug PNGs are intentionally tone-mapped SDR previews. They can reveal clipping, NaN/Inf, hue
   discontinuities, and warp artifacts, but cannot prove PQ luminance accuracy.
3. HDR mastering primaries are deliberately signaled as Rec.2020 after conversion, while display
   min/max luminance comes from DXGI. When DXGI does not supply content light levels, they are not
   guessed.
4. The bounded SDR transfer polynomial is materially cheaper than a per-tap `pow` and has less than
   0.01 of one 8-bit code step of error. The nonlinear FP16/PQ and BGRA Rec.2020 paths now perform
   more ALU work for correct chroma, but only in the quarter-resolution UV pass: the 2/4/6-tap
   layouts evaluate the conversion at the equivalent of 0.5/1/1.5 full-resolution samples, not
   2/4/6 full-frame passes. End-to-end 7680x2160 GPU timing still belongs in the live Galaxy XR
   validation loop.

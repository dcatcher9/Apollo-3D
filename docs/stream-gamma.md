# Stream gamma

Moonlight 3D selects **Windows default**, **Gamma 2.2**, or **Gamma 2.4** for an HDR
stream. Sunshine 3D applies the selection once in its final GPU color conversion.
The correction uses this fork's Windows D3D11/NVENC HDR encoder path.
Windows default bypasses the correction and retains the existing transfer operations.
The setting belongs to the remote session; it does not change the PC's display gamma ramp.
In-stream choices apply automatically after a short debounce. Gamma and live quality requests
are serialized; settings that require a reconnect use the established session's guarded resume
request automatically, without quitting or replacing the host application.
The requested mode is retained separately from encoder proof. A temporary SDR source uses
the default conversion without forgetting the request, so an HDR encoder rebuild can restore it.

The curve follows the uncalibrated gamma-only HDR path in
[Gloam's source](https://github.com/halideworks/gloam/blob/79d4c471bc5e8802fd4288712014776af023b4d2/src/Gloam.Core/LutGenerator.cs)
and its exact IEC sRGB transfer functions. The implementation evaluates the analytical
curve instead of reproducing Gloam's 1024-entry LUT or Windows' 256-entry hardware ramp
interpolation. Night mode, measured calibration, brightness, shadow lift, and white-point
adjustments are outside this setting.

## Windows SDR and HDR review

Windows Advanced Color composes SDR and HDR applications in linear FP16 scRGB. Ordinary
SDR content is decoded with the piecewise sRGB transfer and scaled to the display's selected
SDR reference white. Native HDR content can extend above that white. In an HDR scRGB desktop,
`1.0` represents 80 nits; the desktop does not carry an SDR/HDR origin label per composed pixel.
See Microsoft's [Advanced Color architecture](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range).

Advanced Color SDR FP16 instead uses `1.0` for normalized display reference white. The host
requires an HDR request **and** an active HDR source display before selecting PQ output, and
the linear SDR branch uses normalized SDR directly. Ten-bit P010 alone is not proof of HDR.

sRGB's linear toe produces more light in shadows than a pure 2.2 or 2.4 power law. For example,
a neutral 5% SDR signal at a 200-nit reference white produces about 0.787 nits with sRGB,
0.275 with gamma 2.2, or 0.151 with gamma 2.4. Gloam reconstructs that signal from linear light
and applies the chosen power law. Both sRGB and power-law presentation have valid uses; this is
an explicit viewing choice, not a claim that every Windows application has an incorrect curve.

The live Gloam setting uses the output's hardware gamma ramp. That display-specific stage can
change the LG while leaving desktop capture unchanged. Its measured MHC2 calibration is a
separate mechanism and must not be conflated with the live gamma setting. The stream correction
therefore runs inside the host's final conversion, before encoding.

The reviewed presentation routes are:

| Route | Current color handling |
| --- | --- |
| HDR stream | WGC requests FP16; DDup prioritizes FP16. The final conversion maps linear Rec.709 scRGB into Rec.2020 PQ, with matching encoder color declarations. |
| Direct 2D / Host SBS client | MediaCodec buffers retain their actual color metadata through direct presentation. No client gamma or intentional SDR tone map is added. |
| Client SBS | HDR-capable full-resolution color remains PQ after actual GL channel/intermediate precision checks. The depth-model input has its own SDR tone map; inadequate presentation precision selects an explicit SDR fallback. |
| SDR stream from linear HDR capture | The existing host HDR-to-SDR tone mapper runs before SDR encoding. |
| WGC SDR stream from an HDR desktop | WGC currently requests BGRA8. Highlight headroom can be discarded before the FP16-only HDR-to-SDR tone mapper can run. This existing limitation is outside the gamma change. |

Microsoft explicitly [warns against BGRA8 capture of an HDR desktop](https://learn.microsoft.com/en-us/windows/apps/develop/media-authoring-processing/screen-capture).
This limitation does not establish that the requested FP16 HDR stream is being flattened.
Full/limited range controls code scaling, independently of the color transfer. Display peak
metadata comes from the captured output's advertised capabilities; it does not measure headset
brightness. A screenshot's format or viewer can also flatten HDR, so resemblance to an SDR
screenshot is not proof that the HDR stream lost its HDR values.

## Color contract

For each linear Rec.2020 channel with absolute light `L`, reference white `W`, and
selected exponent `g`, the correction below `W` is
`W * pow(sRGB_OETF(clamp(L / W, 0, 1)), g)`. Each channel at or above `W` passes through.
The conversion order is native scRGB -> linear Rec.2020 -> gamma correction -> PQ -> YUV.
BGRA SDR sources feeding HDR first decode sRGB and restore the captured display's SDR white.
The ordinary SDR stream conversion is unchanged; nondefault requests for SDR streams are
acknowledged as unsupported with the actual applied mode.

`W` comes from the captured Windows display's `DISPLAYCONFIG_SDR_WHITE_LEVEL`, not EDID peak
luminance or headset brightness. Active correction polls it at most once a second, including
on a static desktop. Finite positive readings are bounded to 40..1000 nits, following Gloam's
runtime range; unavailable readings use the host's existing 203-nit fallback.

Correction runs after capture and after Host AI depth preprocessing and stereo rendering.
It covers 2D, Host AI, Game 3D, and Movie 3D through the common encoder conversion. While active,
the optional Host AI warp/luma MRT gives way to the canonical separate Y/UV passes so both
planes use the same curve. The authenticated native warp and preprocessing sources are unchanged.
A gamma or white change invalidates cached encoder planes while preserving reusable native color
and geometry. Switching to Windows default restores eligibility for the existing MRT.

Client SBS receives the corrected encoded stream, so its decoded model input can change.
Its preprocessing and rendering do not apply another gamma correction. A composed capture has
no reliable per-pixel SDR/HDR origin label: dark native HDR channels are affected too, and a
colored highlight may have channels on both sides of `W`. Neutral highlights above `W` retain
their encoded brightness; this feature cannot increase the headset's physical brightness.

## Negotiation and proof

The authenticated server-info field `StreamGammaV1Supported=1` advertises support. A supported
client includes `streamGamma=0|1|2` in launch/resume. DESCRIBE advertises
`a=x-ss-video.streamGammaVersion:1`; ANNOUNCE confirms
`a=x-ml-video.streamGammaVersion:1` before the host applies the launch selection or accepts
live requests. Enum values are 0 Windows default, 1 Gamma 2.2, and 2 Gamma 2.4. Older hosts
receive no gamma request and retain their existing output.

The control request is `0x300C` with an 8-byte body: version `u8=1`, mode `u8`, reserved
`u16LE=0`, request ID `u32LE`. Client request IDs are nonzero. The `0x300D` acknowledgement
has a 16-byte body: version `u8=1`, status `u8`, requested mode `u8`, applied mode `u8`,
request ID `u32LE`, generation `u32LE`, and reference white `f32LE` in nits.
Statuses are 0 applied, 1 invalid, 2 unsupported, 3 failed. Request ID zero announces initial
or rebuilt encoder state or a reference-white update. A nonzero generation identifies a
converted and successfully encoded frame, not merely receipt of a request. The client shows
the applied state after validating the acknowledgement and keeps desired settings separate
from host proof. Late replies from a retired connection cannot change the current session.
An ID-zero state update retains the session's requested mode even when the applied mode is
temporarily default because the source is SDR. Recovery to HDR can then acknowledge the requested
gamma without another user selection.

## Verification

`StreamGammaShader` tests compile the production HLSL and execute it using D3D11 WARP.
Reference values come from the pinned original Gloam core. Coverage includes exact default
parity, shadows, white, monotonicity, Rec.2020 channel order, and neutral HDR highlights.
Protocol and client state tests exercise packet sizes, enum/version/reserved validation,
request correlation, generation, and unsupported-host behavior. Physical acceptance also
checks retained-source application and unsupported SDR requests.

For physical acceptance, hold game HDR, Windows SDR white, headset brightness, resolution,
and codec fixed. Change the session setting on a static desktop and in a dark HDR game scene.
Verify shadows change and neutral highlights above SDR white retain their distinction. Check
2D and each used stereo producer, reconnect, and an older host. Physical acceptance must be
reported separately from automated shader/protocol evidence.

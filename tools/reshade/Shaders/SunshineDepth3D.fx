// SPDX-License-Identifier: GPL-3.0-only
// Original native-depth stereo for Sunshine's ReShade integration.
// No upstream effect, game profile, AI renderer, or external include is required.

#if __RESHADE__ < 60800 || (__RENDERER__ != 0xb000 && __RENDERER__ != 0xc000)
    #error "Sunshine native-depth stereo requires ReShade 6.8 with Direct3D 11 or 12."
#endif
#if BUFFER_COLOR_SPACE != 1 && BUFFER_COLOR_SPACE != 2 && BUFFER_COLOR_SPACE != 3
    #error "Sunshine native-depth stereo requires SDR, scRGB, or HDR10 PQ source color."
#endif
#if BUFFER_WIDTH <= 0 || BUFFER_HEIGHT <= 0 || BUFFER_WIDTH > 8192 || BUFFER_HEIGHT > 8192 || (BUFFER_WIDTH % 2) || (BUFFER_HEIGHT % 2)
    #error "Sunshine native-depth stereo requires even source dimensions up to 8192 x 8192."
#endif

namespace SunshineDepth3D
{
uniform float Strength <
    ui_type = "slider"; ui_min = 0.0; ui_max = 2.0; ui_step = 0.01;
    ui_label = "3D strength";
    ui_tooltip = "Zero gives identical eyes. Increase gradually for more separation.";
> = 1.0;
uniform float ScreenPlane <
    ui_type = "slider"; ui_min = -1.0; ui_max = 1.0; ui_step = 0.01;
    ui_label = "Screen plane";
    ui_tooltip = "Objects at this calibrated depth remain on the screen. Positive moves the screen plane toward nearer objects.";
> = 0.0;
uniform int DepthDirection <
    ui_type = "combo"; ui_items = "Automatic\0Normal depth\0Reversed depth\0";
    ui_label = "Depth direction";
    ui_tooltip = "The Sunshine add-on applies this choice when calibrating the selected game depth buffer.";
>;
uniform bool DepthView <
    ui_label = "Show depth";
    ui_tooltip = "Calibrated raw depth: nearer is white, farther is black. This view is independent of 3D strength and screen plane.";
> = false;
uniform bool EdgeAntialias <
    ui_label = "Smooth stereo edges";
    ui_tooltip = "Use subpixel coverage at stereo boundaries. Unchanged surfaces retain the game's original anti-aliasing.";
> = true;

// The add-on updates this complete calibration together. Omit initializers so
// ReShade performance mode keeps these runtime-fed values mutable, including
// the direction choice read by the add-on. ReShade's zeroed constant buffer
// keeps the effect flat before the first complete valid calibration update.
uniform bool Sunshine_DepthReady < source = "bufready_depth"; hidden = true; >;
uniform bool Sunshine_Calibrated < hidden = true; >;
uniform float Sunshine_RawAnchor < hidden = true; >;
uniform float Sunshine_RawGain < hidden = true; >;
uniform float4 Sunshine_DepthRect < hidden = true; >;

texture2D GameColor : COLOR;
texture2D GameDepth : DEPTH;
sampler2D NativePoint {
    Texture = GameColor; AddressU = CLAMP; AddressV = CLAMP;
    MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;
    SRGBTexture = false;
};
sampler2D<float> RawDepth {
    Texture = GameDepth; AddressU = CLAMP; AddressV = CLAMP;
    MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;
};

texture2D DoubleTex <
    sunshine_sbs_export = 1;
    sunshine_sbs_layout = "sbs_lr";
    sunshine_sbs_source_color_space = BUFFER_COLOR_SPACE;
    #if BUFFER_COLOR_SPACE == 1
        sunshine_sbs_color_space = "srgb";
    #else
        sunshine_sbs_color_space = "scrgb";
    #endif
    sunshine_sbs_source_width = BUFFER_WIDTH;
    sunshine_sbs_source_height = BUFFER_HEIGHT;
> {
    Width = 2 * BUFFER_WIDTH; Height = BUFFER_HEIGHT;
    #if BUFFER_COLOR_SPACE == 1
        Format = RGB10A2;
    #else
        Format = RGBA16F;
    #endif
};

bool Finite(float value)
{
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool CalibrationReady()
{
    return Sunshine_DepthReady && Sunshine_Calibrated &&
        Finite(Sunshine_RawAnchor) && Finite(Sunshine_RawGain) && Sunshine_RawGain != 0.0 &&
        Finite(Sunshine_DepthRect.x) && Finite(Sunshine_DepthRect.y) &&
        Finite(Sunshine_DepthRect.z) && Finite(Sunshine_DepthRect.w) &&
        all(Sunshine_DepthRect.xy > 0.0) && all(Sunshine_DepthRect.zw >= 0.0) &&
        all(Sunshine_DepthRect.xy + Sunshine_DepthRect.zw <= 1.000001);
}

// Keep raw hardware depth and centering in float32. In particular, do not first
// store reversed-Z depth in a half-float intermediate or subtract it from one.
float2 Nearness(float2 source_uv)
{
    float2 depth_uv = source_uv * Sunshine_DepthRect.xy + Sunshine_DepthRect.zw;
    float raw = tex2Dlod(RawDepth, float4(depth_uv, 0.0, 0.0));
    float near_value = (raw - Sunshine_RawAnchor) * Sunshine_RawGain;
    bool valid = all(source_uv >= 0.0) && all(source_uv <= 1.0) &&
        Finite(raw) && raw >= 0.0 && raw <= 1.0 && Finite(near_value);
    return float2(valid ? near_value : 0.0, valid ? 1.0 : 0.0);
}

float3 DecodePQ(float3 code)
{
    // ST.2084 absolute luminance, followed by Rec.2020 -> Rec.709.
    // scRGB uses 80 nits per unit. The matrix's negative values are intentional.
    float3 p = pow(saturate(code), 32.0 / 2523.0);
    float3 light = 125.0 * pow(max(p - 3424.0 / 4096.0, 0.0) /
        max(2413.0 / 128.0 - (2392.0 / 128.0) * p, 0.000001), 16384.0 / 2610.0);
    return float3(
        dot(light, float3(1.6604910021, -0.5876411388, -0.0728498633)),
        dot(light, float3(-0.1245504745, 1.1328998971, -0.0083494226)),
        dot(light, float3(-0.0181507634, -0.1005788980, 1.1187296614)));
}

float3 DecodeNative(float3 code)
{
    #if BUFFER_COLOR_SPACE == 1
        return code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4);
    #elif BUFFER_COLOR_SPACE == 3
        return DecodePQ(code);
    #else
        return code;
    #endif
}

float3 LinearColor(float2 uv, float maximum_shift, float screen_plane)
{
    // The warp changes only X; Y stays on its original native texel center.
    // Decode both neighboring texels before interpolating. Explicit SDR decode
    // also supports 10-bit UNORM swapchains, which have no hardware sRGB view.
    float pixel = saturate(uv.x) * float(BUFFER_WIDTH) - 0.5;
    float base = floor(pixel), blend = frac(pixel);
    float2 da = Nearness(float2((base + 0.5) / BUFFER_WIDTH, uv.y));
    float2 db = Nearness(float2((base + 1.5) / BUFFER_WIDTH, uv.y));
    float disparity_a = clamp(da.x - screen_plane, -1.0, 1.0) * maximum_shift;
    float disparity_b = clamp(db.x - screen_plane, -1.0, 1.0) * maximum_shift;
    // RayDepth keeps separate native pixel footprints at a depth cliff. Color
    // reconstruction must respect that same ownership: linear mixing here would
    // pull background into a foreground hit (or vice versa), creating a halo.
    // Stereo coverage AA already combines independently resolved output rays.
    if (da.y > 0.0 && db.y > 0.0 && abs(disparity_b - disparity_a) > 1.0)
        blend = blend < 0.5 ? 0.0 : 1.0;
    float3 a = DecodeNative(tex2Dlod(NativePoint, float4((base + 0.5) / BUFFER_WIDTH, uv.y, 0, 0)).rgb);
    float3 b = DecodeNative(tex2Dlod(NativePoint, float4((base + 1.5) / BUFFER_WIDTH, uv.y, 0, 0)).rgb);
    return lerp(a, b, blend);
}

float3 EncodeOutput(float3 linear_color)
{
    #if BUFFER_COLOR_SPACE == 1
        float3 c = max(linear_color, 0.0);
        return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    #else
        return linear_color;
    #endif
}

float4 FlatColor(float2 uv)
{
    float3 color = tex2Dlod(NativePoint, float4(uv, 0.0, 0.0)).rgb;
    #if BUFFER_COLOR_SPACE == 3
        color = DecodePQ(color);
    #endif
    return float4(color, 1.0);
}

float BackgroundInterior(float boundary, float direction, float background_near,
    float y, float maximum_shift, int margin)
{
    // TAA/upscaling can mix foreground color into the first background texel.
    // Repeating that texel across an uncovered gap turns the mixture into a
    // horizontal spike. Move only the fill sample into the same background
    // surface; observed geometry/color and thin foreground objects stay intact.
    float source = boundary;
    [loop]
    for (int step = 1; step <= margin; ++step) {
        float candidate = boundary + direction * float(step);
        float2 depth = Nearness(float2(candidate / BUFFER_WIDTH, y));
        // Even a slightly closer thin object must not become a donor that is
        // stretched across the hole. A nearer-sloping background conservatively
        // keeps the last supported sample instead of relaxing this guard.
        if (depth.y == 0.0 || depth.x > background_near ||
            (background_near - depth.x) * maximum_shift > 1.0)
            break;
        source = candidate;
    }
    return source;
}

// The inverse ray samples a continuous depth field on smooth surfaces. Across
// a depth cliff use the native pixel footprint instead of inventing a surface
// joining foreground and background. Raw calibration and diagnostics stay FP32.
// Result: bounded disparity in source pixels, unbounded nearness, validity.
float3 RayDepth(float source_x, float y, float maximum_shift, float screen_plane)
{
    float base = floor(source_x - 0.5), blend = source_x - 0.5 - base;
    float2 a = Nearness(float2((base + 0.5) / BUFFER_WIDTH, y));
    float2 b = Nearness(float2((base + 1.5) / BUFFER_WIDTH, y));
    float da = clamp(a.x - screen_plane, -1.0, 1.0) * maximum_shift;
    float db = clamp(b.x - screen_plane, -1.0, 1.0) * maximum_shift;
    float2 depth = blend < 0.5 ? a : b;
    float disparity = blend < 0.5 ? da : db;
    if (a.y > 0.0 && b.y > 0.0 && abs(db - da) <= 1.0) {
        depth = float2(lerp(a.x, b.x, blend), 1.0);
        // Clamp the samples before interpolation. Clamping an interpolated
        // extreme raw value would introduce an unvisited saturation knot and
        // could hide a ray intersection inside this supposedly smooth interval.
        disparity = lerp(da, db, blend);
    }
    if (source_x < 0.0 || source_x >= BUFFER_WIDTH) depth.y = 0.0;
    return float3(disparity, depth);
}

void RefineRay(float destination, float y, float eye_sign, float maximum_shift, float screen_plane,
    float hi, float lo, float rhi, float rlo, inout float source, inout float found)
{
    if (found > 0.0 || hi <= lo || rhi < -0.0001 || rlo > 0.0001)
        return;
    // The affine root normally needs one landing. Retain bounded refinement
    // and a fresh field sample for rounding and exact footprint boundaries.
    [loop]
    for (int refine = 0; refine < 6; ++refine) {
        float range = rhi - rlo;
        float q = range > 0.000001 ? lerp(lo, hi, saturate(-rlo / range)) : hi;
        float candidate = destination - eye_sign * q;
        float3 depth = RayDepth(candidate, y, maximum_shift, screen_plane);
        float residual = q - depth.x;
        if (depth.z > 0.0 && abs(residual) <= 0.002) {
            source = candidate;
            found = 1.0;
            break;
        }
        if (depth.z == 0.0) break;
        if (residual > 0.0) { hi = q; rhi = residual; }
        else { lo = q; rlo = residual; }
    }
}

// Solve q - depth(destination - eye*q) = 0 on one affine interval of
// RayDepth. Its samples are shared by all three destination rays. q bounds are
// clipped per ray; no source surface is projected into a destination interval.
void TraceInterval(float near_x, float length, float near_disparity, float far_disparity,
    float3 destination, float y, float eye_sign, float maximum_shift, float screen_plane,
    inout float3 source, inout float3 found)
{
    float3 grid_near = eye_sign * (destination - near_x);
    float3 hi = min(maximum_shift, grid_near);
    float3 lo = max(-maximum_shift, grid_near - length);
    float3 rhi = hi - lerp(near_disparity, far_disparity, (grid_near - hi) / length);
    float3 rlo = lo - lerp(near_disparity, far_disparity, (grid_near - lo) / length);
    // Scalar refinement avoids FXC's dynamic vector-lvalue lowering inside a
    // loop. Depth sampling and interval arithmetic remain shared across lanes.
    RefineRay(destination.x, y, eye_sign, maximum_shift, screen_plane,
        hi.x, lo.x, rhi.x, rlo.x, source.x, found.x);
    RefineRay(destination.y, y, eye_sign, maximum_shift, screen_plane,
        hi.y, lo.y, rhi.y, rlo.y, source.y, found.y);
    RefineRay(destination.z, y, eye_sign, maximum_shift, screen_plane,
        hi.z, lo.z, rhi.z, rlo.z, source.z, found.z);
}

// Independently authored inverse depth ray search. Each destination follows
// source(q) = destination - eye*q, from near (+M) to far (-M). The three rays
// traverse the same native source grid, so each new pixel depth is fetched once
// rather than repeatedly probing it in three independent marches.
void TraceRays(float center, float aa, float y, float eye_sign,
    float maximum_shift, float screen_plane, int fill_margin,
    out float3 source, out float3 found)
{
    float3 destination = center + float3(-aa, 0.0, aa);
    source = destination;
    // Only the middle lane is consumed with AA disabled.
    found = aa > 0.0 ? 0.0 : float3(1.0, 0.0, 1.0);
    float3 gap_source = destination, gap_near = 0.0, gap_found = 0.0;
    float start_x = center - eye_sign * (maximum_shift + aa);
    float first = (eye_sign > 0.0 ? floor(start_x - 0.5) : ceil(start_x - 0.5)) + 0.5;
    int steps = min(168, (int)ceil(2.0 * (maximum_shift + aa)) + 2);
    float near_x = first;
    float2 a = Nearness(float2(near_x / BUFFER_WIDTH, y));
    float da = clamp(a.x - screen_plane, -1.0, 1.0) * maximum_shift;
    [loop]
    for (int step = 0; step < steps; ++step) {
        float far_x = first + eye_sign * float(step + 1);
        float2 b = Nearness(float2(far_x / BUFFER_WIDTH, y));
        float db = clamp(b.x - screen_plane, -1.0, 1.0) * maximum_shift;
        if (a.y > 0.0 && b.y > 0.0 && abs(db - da) <= 1.0) {
            // The entire center-to-center interval is affine. There is no
            // extra knot at its midpoint when the neighboring depths agree.
            TraceInterval(near_x, 1.0, da, db, destination, y, eye_sign,
                maximum_shift, screen_plane, source, found);
        } else {
            // Across a cliff, intersect each constant half-pixel footprint
            // separately. A thin native pixel retains both of its halves;
            // there is no invented ramp through an unobserved disocclusion.
            float middle_x = near_x + 0.5 * eye_sign;
            if (a.y > 0.0)
                TraceInterval(near_x, 0.5, da, da, destination, y, eye_sign,
                    maximum_shift, screen_plane, source, found);
            if (a.y > 0.0 && b.y > 0.0 && db - da > 1.0) {
                float3 q = eye_sign * (destination - middle_x);
                [unroll]
                for (int ray = 0; ray < 3; ++ray) {
                    if (found[ray] == 0.0 && gap_found[ray] == 0.0 && da < q[ray] && db > q[ray]) {
                        // The preceding, farther sample is already a native
                        // color-texel center, not a fractional march probe.
                        gap_source[ray] = near_x;
                        gap_near[ray] = a.x;
                        gap_found[ray] = 1.0;
                    }
                }
            }
            if (b.y > 0.0)
                TraceInterval(middle_x, 0.5, db, db, destination, y, eye_sign,
                    maximum_shift, screen_plane, source, found);
        }
        if (all(found > 0.0)) break;
        near_x = far_x;
        a = b;
        da = db;
    }
    [unroll]
    for (int ray = 0; ray < 3; ++ray) {
        if (found[ray] == 0.0) {
            source[ray] = gap_found[ray] > 0.0 ?
                BackgroundInterior(gap_source[ray], -eye_sign, gap_near[ray], y, maximum_shift, fill_margin) :
                gap_source[ray];
        }
    }
}

float4 RenderEye(float2 uv, float eye_sign)
{
    if (!CalibrationReady()) return FlatColor(uv);
    if (DepthView) {
        float2 depth = Nearness(uv);
        float gray = depth.y > 0.0 ? saturate(0.5 + 0.5 * depth.x) : 0.0;
        return float4(gray.xxx, 1.0);
    }
    if (!Finite(Strength) || Strength <= 0.0) return FlatColor(uv);

    float maximum_shift = 0.005 * float(BUFFER_WIDTH) * clamp(Strength, 0.0, 2.0);
    float screen_plane = Finite(ScreenPlane) ? clamp(ScreenPlane, -1.0, 1.0) : 0.0;
    float center = uv.x * float(BUFFER_WIDTH);
    // One depth-texel footprint plus the native color-filter support, in source
    // pixels. This follows the selected depth viewport, including DLSS padding,
    // rather than assuming depth and displayed color have the same resolution.
    float active_depth_width = float(tex2Dsize(RawDepth).x) * Sunshine_DepthRect.x;
    int fill_margin = min(8, (int)ceil(float(BUFFER_WIDTH) / max(active_depth_width, 1.0)) + 1);
    float3 source, found;
    TraceRays(center, EdgeAntialias ? 0.25 : 0.0, uv.y, eye_sign,
        maximum_shift, screen_plane, fill_margin, source, found);
    float3 color = LinearColor(float2(source.y / BUFFER_WIDTH, uv.y), maximum_shift, screen_plane);
    float confidence = found.y;
    if (EdgeAntialias) {
        bool edge = found.x != found.z || found.y != found.x || abs((source.z - source.x) - 0.5) > 0.75;
        if (edge) {
            color = 0.5 * (LinearColor(float2(source.x / BUFFER_WIDTH, uv.y), maximum_shift, screen_plane) +
                LinearColor(float2(source.z / BUFFER_WIDTH, uv.y), maximum_shift, screen_plane));
            confidence = min(found.x, found.z);
        }
    }
    return float4(EncodeOutput(color), confidence);
}

void StereoVS(uint vertex : SV_VertexID, out float4 position : SV_Position)
{
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 StereoPS(float4 position : SV_Position) : SV_Target
{
    bool right_eye = position.x >= float(BUFFER_WIDTH);
    float2 uv = float2(position.x - (right_eye ? BUFFER_WIDTH : 0), position.y) /
        float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    return RenderEye(uv, right_eye ? -1.0 : 1.0);
}

// Every pass has an explicit offscreen render target. ReShade's native game
// backbuffer is never replaced, including while depth is missing or disabled.
technique SunshineDepth3D < ui_label = "Sunshine 3D"; >
{
    pass Stereo {
        VertexShader = StereoVS;
        PixelShader = StereoPS;
        RenderTarget = DoubleTex;
        SRGBWriteEnable = false;
    }
}
}

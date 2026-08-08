// Production Host SBS renderer for the authenticated Depth Coordinate V2 final field.
//
// The producer supplies a signed, one-eye displacement in source-U units whose horizontal slope
// is strictly below one. Each eye map therefore has one inverse; a short fixed-point solve is the
// complete visibility rule. Invalid or unauthenticated state renders the current color frame
// through identity so a producer fault can never expose stale pixels or legacy geometry.

Texture2D<float4> SourceColor : register(t0);
Texture2D<float> FinalParallax : register(t1);
StructuredBuffer<float4> ParallaxState : register(t2);
SamplerState LinearSampler : register(s0);

cbuffer HostSbsV2Geometry : register(b2) {
    float content_scale_x;
    float content_scale_y;
    float video_roi_active;
    float geometry_reserved;
    float4 video_roi_source_uv;
};

#include "include/depth_coordinate_v2_contract.generated.hlsl"

// Renderer-local mirror of the producer's authenticated row-slope contract. Keeping the collar
// bound here avoids making a renderer-only ROI feature part of the generated producer ABI. The
// executable GPU test cross-checks this literal against depth_coordinate_v2::max_horizontal_slope.
static const float VIDEO_ROI_MAX_HORIZONTAL_SLOPE = 0.5f;

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

bool WarpAvailable() {
    float4 mapping_state = ParallaxState[V2_STATE_VECTOR_CAMERA_CENTER_INTEGRITY_BITS];
    bool roi_mode_valid = video_roi_active == 0.0f || video_roi_active == 1.0f;
    bool roi_reserved_valid = geometry_reserved == 0.0f;
    bool roi_rect_valid = video_roi_active == 0.0f ||
        (all(!isnan(video_roi_source_uv)) && all(!isinf(video_roi_source_uv)) &&
         video_roi_source_uv.x >= 0.0f && video_roi_source_uv.y >= 0.0f &&
         video_roi_source_uv.z <= 1.0f && video_roi_source_uv.w <= 1.0f &&
         video_roi_source_uv.z > video_roi_source_uv.x &&
         video_roi_source_uv.w > video_roi_source_uv.y);
    return asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) == V2_CONTRACT_TAG &&
        roi_mode_valid && roi_reserved_valid && roi_rect_valid;
}

bool ContentToSourceUV(float2 output_uv, out float2 source_uv) {
    float2 scale = max(float2(content_scale_x, content_scale_y), float2(1e-6f, 1e-6f));
    float2 lo = 0.5f * (float2(1.0f, 1.0f) - scale);
    float2 hi = lo + scale;
    if (any(output_uv < lo) || any(output_uv > hi)) {
        source_uv = float2(0.0f, 0.0f);
        return false;
    }
    source_uv = saturate((output_uv - lo) / scale);
    return true;
}

float SampleVideoRoiParallax(
    float source_x,
    float source_y,
    float roi_width,
    float vertical_budget_per_source_v
) {
    float effective_parallax = 0.0f;
    // The ROI-local field uses the entire DAV2 grid and retains the ordinary V2 limiter slopes.
    // Convert its local-U displacement to full-source U before constructing the smallest signed
    // exterior collar allowed by those same horizontal/vertical slope contracts. The projection
    // leaves every point inside the ROI untouched. Outside, soft-thresholding reaches exact zero
    // as quickly as the authenticated slopes permit; the summed anisotropic distance gives the
    // corresponding minimum diamond at a corner.
    float2 roi_min = video_roi_source_uv.xy;
    float2 roi_max = video_roi_source_uv.zw;
    float2 source_uv = float2(source_x, source_y);
    float2 projected_uv = clamp(source_uv, roi_min, roi_max);
    float2 local_uv = (projected_uv - roi_min) / (roi_max - roi_min);
    float roi_local_parallax = FinalParallax.SampleLevel(
        LinearSampler,
        local_uv,
        0
    );
    float full_source_parallax = roi_width * roi_local_parallax;
    float2 outside_distance = abs(source_uv - projected_uv);
    float collar_budget =
        VIDEO_ROI_MAX_HORIZONTAL_SLOPE * outside_distance.x +
        vertical_budget_per_source_v * outside_distance.y;
    float magnitude = abs(full_source_parallax);
    if (collar_budget < magnitude) {
        float retained_magnitude = magnitude - collar_budget;
        effective_parallax = full_source_parallax < 0.0f ?
            -retained_magnitude : retained_magnitude;
    }
    return effective_parallax;
}

float2 Reproject(float2 destination_uv, float eye_sign) {
    // out(x) = x - eye_sign * parallax(x). The authenticated field is contractive, so this
    // iteration converges to its unique inverse without ownership search or hole filling.
    float source_x = destination_uv.x;
    if (video_roi_active == 0.0f) {
        // Keep the ordinary full-frame path exactly as before and pay no ROI dimension work.
        [unroll]
        for (int iteration = 0; iteration < 11; ++iteration) {
            source_x = destination_uv.x + eye_sign * FinalParallax.SampleLevel(
                LinearSampler,
                float2(source_x, destination_uv.y),
                0
            );
        }
    } else {
        float roi_width = video_roi_source_uv.z - video_roi_source_uv.x;
        float roi_height = video_roi_source_uv.w - video_roi_source_uv.y;
        uint source_width;
        uint source_height;
        SourceColor.GetDimensions(source_width, source_height);
        uint depth_width;
        uint depth_height;
        FinalParallax.GetDimensions(depth_width, depth_height);
        float source_height_in_source_u =
            (float)source_height / max((float)source_width, 1.0f);
        // The producer's vertical step remains V2_MAX_VERTICAL_SHEAR / depth_width per depth
        // row. Convert that local-grid derivative exactly into full-source coordinates. Keeping
        // the actual ROI and tensor aspects retains their patch-rounding difference.
        float roi_pixel_aspect =
            (roi_width * (float)source_width) /
            max(roi_height * (float)source_height, 1.0e-6f);
        float vertical_slope = V2_MAX_VERTICAL_SHEAR * roi_pixel_aspect *
            ((float)depth_height / max((float)depth_width, 1.0f));
        float vertical_budget_per_source_v =
            vertical_slope * source_height_in_source_u;
        [unroll]
        for (int iteration = 0; iteration < 11; ++iteration) {
            source_x = destination_uv.x + eye_sign * SampleVideoRoiParallax(
                source_x,
                destination_uv.y,
                roi_width,
                vertical_budget_per_source_v
            );
        }
    }
    return float2(source_x, destination_uv.y);
}

bool PackedToSource(float2 packed_uv, out float2 source_uv, out float eye_sign) {
    bool right_eye = packed_uv.x > 0.5f;
    eye_sign = right_eye ? 1.0f : -1.0f;
    float2 eye_uv = packed_uv;
    eye_uv.x = right_eye ? (packed_uv.x - 0.5f) * 2.0f : packed_uv.x * 2.0f;
    return ContentToSourceUV(eye_uv, source_uv);
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Null/uninitialized buffers read as an invalid contract. Keep the exact current color frame
    // live while the producer starts or after it fails terminally.
    float2 sample_uv = WarpAvailable() ? Reproject(source_uv, eye_sign) : source_uv;
    sample_uv.x = saturate(sample_uv.x);
    return SourceColor.Sample(LinearSampler, sample_uv);
}

// Production Host SBS renderer for the authenticated Depth Coordinate V2 final field.
//
// The producer supplies a signed, one-eye displacement in source-U units whose horizontal slope
// is strictly below one. Each eye map therefore has one inverse; an at-most-11-step fixed-point
// solve is the complete visibility rule. The solve may stop only when the next source coordinate
// has exactly the same IEEE-754 bits. Invalid or unauthenticated state renders the current color
// frame through identity so a producer fault can never expose stale pixels or legacy geometry.

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
    uint4 video_roi_tensor_content_rect;
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

bool WarpAuthorized() {
    float4 mapping_state = ParallaxState[V2_STATE_VECTOR_CAMERA_CENTER_INTEGRITY_BITS];
    bool roi_mode_valid = video_roi_active == 0.0f || video_roi_active == 1.0f;
    bool roi_reserved_valid = geometry_reserved == 0.0f;
    return asuint(V2_STATE_RENDERER_AUTHORIZATION_BITS(mapping_state)) == V2_CONTRACT_TAG &&
        roi_mode_valid && roi_reserved_valid;
}

bool VideoRoiRectValid() {
    return all(!isnan(video_roi_source_uv)) && all(!isinf(video_roi_source_uv)) &&
        video_roi_source_uv.x >= 0.0f && video_roi_source_uv.y >= 0.0f &&
        video_roi_source_uv.z <= 1.0f && video_roi_source_uv.w <= 1.0f &&
        video_roi_source_uv.z > video_roi_source_uv.x &&
        video_roi_source_uv.w > video_roi_source_uv.y;
}

bool VideoRoiContentValid(uint2 depth_size) {
    return video_roi_tensor_content_rect.x < video_roi_tensor_content_rect.z &&
        video_roi_tensor_content_rect.y < video_roi_tensor_content_rect.w &&
        video_roi_tensor_content_rect.z <= depth_size.x &&
        video_roi_tensor_content_rect.w <= depth_size.y;
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
    float roi_min_x,
    float roi_max_x,
    float roi_width,
    float tensor_u_scale,
    float tensor_u_bias,
    float tensor_v,
    float vertical_collar_budget
) {
    // The ROI-local field occupies its authenticated integer content rectangle within the fixed
    // DAV2 grid. Convert its local-U displacement to full-source U before constructing the
    // smallest signed exterior collar allowed by the same content-local slope contracts. The
    // projection leaves every point inside the ROI untouched. Outside, soft-thresholding reaches
    // exact zero as quickly as those slopes permit; the summed anisotropic distance gives the
    // corresponding minimum diamond at a corner.
    float projected_x = clamp(source_x, roi_min_x, roi_max_x);
    float collar_budget =
        VIDEO_ROI_MAX_HORIZONTAL_SLOPE * abs(source_x - projected_x) +
        vertical_collar_budget;
    float tensor_u = projected_x * tensor_u_scale + tensor_u_bias;
    float roi_local_parallax = FinalParallax.SampleLevel(
        LinearSampler,
        float2(tensor_u, tensor_v),
        0
    );
    float full_source_parallax = roi_width * roi_local_parallax;
    float magnitude = abs(full_source_parallax);
    if (collar_budget < magnitude) {
        float retained_magnitude = magnitude - collar_budget;
        return full_source_parallax < 0.0f ?
            -retained_magnitude : retained_magnitude;
    }
    return 0.0f;
}

float2 ReprojectFull(float2 destination_uv, float eye_sign) {
    // out(x) = x - eye_sign * parallax(x). The authenticated field is contractive, so this
    // iteration converges to its unique inverse without ownership search or hole filling.
    float source_x = destination_uv.x;
    [unroll]
    for (int iteration = 0; iteration < 11; ++iteration) {
        float next_source_x = destination_uv.x + eye_sign * FinalParallax.SampleLevel(
            LinearSampler,
            float2(source_x, destination_uv.y),
            0
        );
        bool exactly_settled = asuint(next_source_x) == asuint(source_x);
        source_x = next_source_x;
#if !defined(HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE)
        if (exactly_settled) {
            break;
        }
#endif
    }
    return float2(source_x, destination_uv.y);
}

float2 ReprojectVideoRoi(float2 destination_uv, float eye_sign, uint2 depth_size) {
    float source_x = destination_uv.x;
    float2 roi_min = video_roi_source_uv.xy;
    float2 roi_max = video_roi_source_uv.zw;
    float2 roi_extent = roi_max - roi_min;
    uint2 content_extent =
        video_roi_tensor_content_rect.zw - video_roi_tensor_content_rect.xy;
    float2 depth_size_f = float2(depth_size);
    float2 content_min = float2(video_roi_tensor_content_rect.xy) / depth_size_f;
    float2 content_max = float2(video_roi_tensor_content_rect.zw) / depth_size_f;
    // Hoist the ROI-to-tensor affine transform out of the unrolled inverse. Destination V is
    // invariant across all fixed-point steps, so its projection, tensor coordinate, and vertical
    // collar contribution are also one-time work.
    float2 tensor_uv_scale = (content_max - content_min) / roi_extent;
    float2 tensor_uv_bias = content_min - roi_min * tensor_uv_scale;
    float projected_y = clamp(destination_uv.y, roi_min.y, roi_max.y);
    float tensor_v = projected_y * tensor_uv_scale.y + tensor_uv_bias.y;
    // The producer's vertical step is V2_MAX_VERTICAL_SHEAR / content_width per real-source
    // row. The source-height/source-width factors in the local pixel aspect and source-V
    // conversion cancel. Rect/content validation proves every divisor here is positive.
    float vertical_budget_per_source_v =
        V2_MAX_VERTICAL_SHEAR * (roi_extent.x / roi_extent.y) *
        ((float)content_extent.y / (float)content_extent.x);
    float vertical_collar_budget =
        vertical_budget_per_source_v * abs(destination_uv.y - projected_y);
    float maximum_full_source_parallax =
        roi_extent.x * V2_DIRECT_CONTAINER_LIMIT;
    // The authenticated final field is bounded by V2_DIRECT_CONTAINER_LIMIT. If even that largest
    // possible displacement cannot survive the destination's collar, the first inverse step is
    // exactly the destination again and every later step is identical. Skip all eleven fetches.
    // Equality also reduced to zero in the original soft-threshold rule.
    float projected_destination_x = clamp(destination_uv.x, roi_min.x, roi_max.x);
    float destination_collar_budget =
        VIDEO_ROI_MAX_HORIZONTAL_SLOPE *
            abs(destination_uv.x - projected_destination_x) +
        vertical_collar_budget;
    if (destination_collar_budget >= maximum_full_source_parallax) {
        return destination_uv;
    }
    [unroll]
    for (int iteration = 0; iteration < 11; ++iteration) {
        float next_source_x = destination_uv.x + eye_sign * SampleVideoRoiParallax(
            source_x,
            roi_min.x,
            roi_max.x,
            roi_extent.x,
            tensor_uv_scale.x,
            tensor_uv_bias.x,
            tensor_v,
            vertical_collar_budget
        );
        bool exactly_settled = asuint(next_source_x) == asuint(source_x);
        source_x = next_source_x;
#if !defined(HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE)
        if (exactly_settled) {
            break;
        }
#endif
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

float2 ReprojectIfAuthorized(float2 source_uv, float eye_sign) {
    float2 sample_uv = source_uv;
    if (WarpAuthorized()) {
        if (video_roi_active == 0.0f) {
            sample_uv = ReprojectFull(source_uv, eye_sign);
        } else if (VideoRoiRectValid()) {
            uint depth_width;
            uint depth_height;
            FinalParallax.GetDimensions(depth_width, depth_height);
            uint2 depth_size = uint2(depth_width, depth_height);
            if (VideoRoiContentValid(depth_size)) {
                sample_uv = ReprojectVideoRoi(source_uv, eye_sign, depth_size);
            }
        }
    }
    return sample_uv;
}

float4 main_ps(PS_INPUT input) : SV_TARGET {
    float2 source_uv;
    float eye_sign;
    if (!PackedToSource(input.TexCoord, source_uv, eye_sign)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Null/uninitialized buffers read as an invalid contract. Keep the exact current color frame
    // live while the producer starts or after it fails terminally.
    float2 sample_uv = ReprojectIfAuthorized(source_uv, eye_sign);
    sample_uv.x = saturate(sample_uv.x);
    return SourceColor.Sample(LinearSampler, sample_uv);
}

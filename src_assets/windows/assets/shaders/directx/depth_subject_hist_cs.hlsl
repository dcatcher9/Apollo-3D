// Weighted depth histogram for Bestv2-derived subject tracking:
// every texel of the NORMALIZED depth map votes for its depth bin with weight
//   center Gaussian (favor the frame center, where the subject usually is)
//   x smoothness (downweight depth edges, so silhouette ramps don't skew the estimate).
// depth_subject_resolve_cs then takes the weighted 35th-percentile-from-NEAR as the
// subject depth. Runs at depth cadence on the low-res depth grid -- negligible cost.

Texture2D<float>         DepthTexture : register(t0);  // normalized depth, high = near
RWStructuredBuffer<uint> SubjectHist  : register(u0);  // 256 bins, weight in 1/1024 units
RWStructuredBuffer<uint> PlainHist    : register(u1);  // bins + cut-evidence counts
Texture2D<float>         PreviousDepth : register(t1);  // last structurally reliable endpoint
StructuredBuffer<float>  CurrentModelInput : register(t2);  // completed frame, NCHW ImageNet
StructuredBuffer<float>  PreviousModelInput : register(t3);
StructuredBuffer<float4> MinMaxEma : register(t4);  // w = current-frame validity
StructuredBuffer<float>  CurrentAppearanceOrdinal : register(t5);  // pre-tone-map point maxRGB
StructuredBuffer<float>  PreviousAppearanceOrdinal : register(t6);
StructuredBuffer<uint4>  FrameRoiTransform : register(t7);
StructuredBuffer<uint4>  PreviousFrameRoiTransform : register(t8);
StructuredBuffer<float>  CurrentRawDepth : register(t9);
Texture2D<uint>           PreviousReliableValidity : register(t10);

// Shared depth-pass cbuffer (only target_w/target_h are used here).
#include "include/depth_constants.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

#define NUM_BINS 256
#define STRUCTURE_TILE_WIDTH 18
#define STRUCTURE_TILE_TEXELS (STRUCTURE_TILE_WIDTH * STRUCTURE_TILE_WIDTH)
static const int2 STRUCTURE_ORDINAL_OFFSETS[5] = {
    int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
};

groupshared uint g_hist[NUM_BINS];
groupshared uint g_plain[NUM_BINS];
groupshared uint g_edge_count;
groupshared uint g_change_count;
groupshared uint g_structural_change_count;
groupshared uint g_raw_rgb_change_count;
groupshared uint g_current_structural_support_count;
groupshared uint g_previous_structural_support_count;
groupshared uint g_common_structural_support_count;
// A 16x16 group plus a one-pixel halo of the pre-tone-map point-sampled ordinal signal.
groupshared float g_current_appearance_ordinal[STRUCTURE_TILE_TEXELS];
groupshared float g_previous_appearance_ordinal[STRUCTURE_TILE_TEXELS];

float3 CurrentModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        CurrentModelInput[idx] * 0.229f + 0.485f,
        CurrentModelInput[idx + plane] * 0.224f + 0.456f,
        CurrentModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

float3 PreviousModelColor(uint2 p) {
    uint plane = target_w * target_h;
    uint idx = p.y * target_w + p.x;
    return float3(
        PreviousModelInput[idx] * 0.229f + 0.485f,
        PreviousModelInput[idx + plane] * 0.224f + 0.456f,
        PreviousModelInput[idx + 2u * plane] * 0.225f + 0.406f);
}

SbsFrameRoiTransformData LoadPreviousFrameRoiTransform() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(
        PreviousFrameRoiTransform);
}

bool TryTargetPlane(out uint plane) {
    plane = 0u;
    if (target_w == 0u ||
        target_h == 0u ||
        target_w > 0xffffffffu / target_h) {
        return false;
    }
    plane = target_w * target_h;
    return true;
}

bool FrameRoiAcceptedBoundsContains(
    uint4 bounds,
    uint2 pixel)
{
    return all(pixel >= bounds.xy) &&
           all(pixel < bounds.zw);
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    SbsFrameRoiTransformData current_transform =
        FrameRoiTransformLoad();
    SbsFrameRoiTransformData previous_transform =
        LoadPreviousFrameRoiTransform();

    uint depth_width;
    uint depth_height;
    DepthTexture.GetDimensions(depth_width, depth_height);
    uint subject_hist_count;
    uint subject_hist_stride;
    SubjectHist.GetDimensions(
        subject_hist_count,
        subject_hist_stride);
    uint plain_hist_count;
    uint plain_hist_stride;
    PlainHist.GetDimensions(
        plain_hist_count,
        plain_hist_stride);
    uint previous_depth_width;
    uint previous_depth_height;
    PreviousDepth.GetDimensions(
        previous_depth_width,
        previous_depth_height);
    uint current_model_count;
    uint current_model_stride;
    CurrentModelInput.GetDimensions(
        current_model_count,
        current_model_stride);
    uint previous_model_count;
    uint previous_model_stride;
    PreviousModelInput.GetDimensions(
        previous_model_count,
        previous_model_stride);
    uint minmax_count;
    uint minmax_stride;
    MinMaxEma.GetDimensions(
        minmax_count,
        minmax_stride);
    uint current_ordinal_count;
    uint current_ordinal_stride;
    CurrentAppearanceOrdinal.GetDimensions(
        current_ordinal_count,
        current_ordinal_stride);
    uint previous_ordinal_count;
    uint previous_ordinal_stride;
    PreviousAppearanceOrdinal.GetDimensions(
        previous_ordinal_count,
        previous_ordinal_stride);
    uint current_transform_vectors;
    uint current_transform_stride;
    FrameRoiTransform.GetDimensions(
        current_transform_vectors,
        current_transform_stride);
    uint previous_transform_vectors;
    uint previous_transform_stride;
    PreviousFrameRoiTransform.GetDimensions(
        previous_transform_vectors,
        previous_transform_stride);
    uint current_raw_count;
    uint current_raw_stride;
    CurrentRawDepth.GetDimensions(
        current_raw_count,
        current_raw_stride);
    uint previous_validity_width;
    uint previous_validity_height;
    PreviousReliableValidity.GetDimensions(
        previous_validity_width,
        previous_validity_height);

    uint plane;
    bool target_plane_safe =
        TryTargetPlane(plane);
    bool current_unbound =
        FrameRoiDataUnboundZero(current_transform);
    bool previous_unbound =
        FrameRoiDataUnboundZero(previous_transform);
    bool legacy_unbound_pair =
        current_unbound &&
        previous_unbound;
    bool current_transform_valid =
        current_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        current_transform_stride == 16u &&
        FrameRoiDataValid(current_transform);
    bool previous_transform_valid =
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        FrameRoiDataValid(previous_transform);
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(current_transform) ==
            uint2(target_w, target_h));
    bool current_transform_usable =
        legacy_unbound_pair ||
        transform_dimensions_match;
    bool current_canonical_full_frame =
        legacy_unbound_pair ||
        (transform_dimensions_match &&
         FrameRoiDataCanonicalFullFrame(current_transform));
    bool current_active_roi =
        transform_dimensions_match &&
        FrameRoiDataActive(current_transform);
    bool same_transform_geometry =
        legacy_unbound_pair ||
        (transform_dimensions_match &&
         previous_transform_valid &&
         !FrameRoiDataGeometryReseedRequired(
             current_transform,
             previous_transform));
    bool previous_canonical_full_frame =
        legacy_unbound_pair ||
        (previous_transform_valid &&
         FrameRoiDataCanonicalFullFrame(previous_transform));
    uint4 current_accepted_bounds =
        current_transform_valid ?
        FrameRoiDataAcceptedModelBounds(current_transform) :
        0u.xxxx;
    uint4 previous_accepted_bounds =
        previous_transform_valid ?
        FrameRoiDataAcceptedModelBounds(previous_transform) :
        0u.xxxx;
    bool histogram_outputs_safe =
        subject_hist_count >= NUM_BINS &&
        subject_hist_stride == 4u &&
        plain_hist_count >= NUM_BINS + 7u &&
        plain_hist_stride == 4u;
    bool current_resources_safe =
        target_plane_safe &&
        current_transform_usable &&
        histogram_outputs_safe &&
        depth_width >= target_w &&
        depth_height >= target_h &&
        current_model_stride == 4u &&
        plane <= current_model_count / 3u &&
        current_ordinal_stride == 4u &&
        current_ordinal_count >= plane &&
        minmax_count >= 1u &&
        minmax_stride == 16u &&
        (!current_active_roi ||
         (current_raw_stride == 4u &&
          current_raw_count >= plane));
    bool previous_tuple_resources_safe =
        target_plane_safe &&
        previous_depth_width >= target_w &&
        previous_depth_height >= target_h &&
        previous_model_stride == 4u &&
        plane <= previous_model_count / 3u &&
        previous_ordinal_stride == 4u &&
        previous_ordinal_count >= plane &&
        (legacy_unbound_pair ||
         (previous_validity_width >= target_w &&
          previous_validity_height >= target_h));
    bool previous_tuple_readable =
        same_transform_geometry &&
        previous_tuple_resources_safe;

    uint lin = tid.y * 16 + tid.x;  // 256 threads/group: one shared bin each
    g_hist[lin] = 0u;
    g_plain[lin] = 0u;
    if (lin == 0u) {
        g_edge_count = 0u;
        g_change_count = 0u;
        g_structural_change_count = 0u;
        g_raw_rgb_change_count = 0u;
        g_current_structural_support_count = 0u;
        g_previous_structural_support_count = 0u;
        g_common_structural_support_count = 0u;
    }

    // Cooperatively load the 18x18 ordinal tile. The final 68 halo samples are handled by the
    // first 68 threads' second loop iteration; clamping exactly matches the depth-gradient rule.
    for (uint tile_idx = lin; tile_idx < STRUCTURE_TILE_TEXELS; tile_idx += 256u) {
        float current_ordinal = 0.0f;
        float previous_ordinal = 0.0f;
        if (target_plane_safe) {
            uint tile_x =
                tile_idx % STRUCTURE_TILE_WIDTH;
            uint tile_y =
                tile_idx / STRUCTURE_TILE_WIDTH;
            int source_x = clamp(
                (int)(gid.x * 16u + tile_x) - 1,
                0,
                (int)target_w - 1);
            int source_y = clamp(
                (int)(gid.y * 16u + tile_y) - 1,
                0,
                (int)target_h - 1);
            uint source_index =
                (uint)source_y * target_w +
                (uint)source_x;
            if (current_resources_safe) {
                current_ordinal =
                    CurrentAppearanceOrdinal[source_index];
            }
            if (previous_tuple_readable) {
                previous_ordinal =
                    PreviousAppearanceOrdinal[source_index];
            }
        }
        g_current_appearance_ordinal[tile_idx] =
            current_ordinal;
        g_previous_appearance_ordinal[tile_idx] =
            previous_ordinal;
    }
    GroupMemoryBarrierWithGroupSync();

    bool current_pixel_accepted =
        current_resources_safe &&
        dtid.x < target_w &&
        dtid.y < target_h &&
        (current_canonical_full_frame ||
         FrameRoiAcceptedBoundsContains(
             current_accepted_bounds,
             dtid.xy));
    uint current_index =
        dtid.y * target_w + dtid.x;
    float current_raw = -1.0f;
    float current_normalized = 0.0f;
    if (current_pixel_accepted) {
        if (current_active_roi) {
            current_raw =
                CurrentRawDepth[current_index];
        }
        current_normalized = DepthTexture[dtid.xy];
    }
    bool current_pixel_valid =
        current_pixel_accepted &&
        (current_canonical_full_frame ||
         (!isnan(current_raw) &&
          !isinf(current_raw) &&
          current_raw >= 0.0f &&
          !isnan(current_normalized) &&
          !isinf(current_normalized)));

    if (current_pixel_valid && MinMaxEma[0].w > 0.5f) {
        float d = current_normalized;

        // Forward-difference gradient (clamped at the far edges). The focus boundary is a
        // controller boundary, not a scene silhouette, so a neighbor outside exact focus reuses
        // the center value and contributes zero gradient.
        uint xn = min(dtid.x + 1, target_w - 1);
        uint yn = min(dtid.y + 1, target_h - 1);
        uint2 x_neighbor = uint2(xn, dtid.y);
        uint2 y_neighbor = uint2(dtid.x, yn);
        bool x_neighbor_accepted =
            current_canonical_full_frame ||
            FrameRoiAcceptedBoundsContains(
                current_accepted_bounds,
                x_neighbor);
        bool y_neighbor_accepted =
            current_canonical_full_frame ||
            FrameRoiAcceptedBoundsContains(
                current_accepted_bounds,
                y_neighbor);
        uint x_neighbor_index =
            x_neighbor.y * target_w + x_neighbor.x;
        uint y_neighbor_index =
            y_neighbor.y * target_w + y_neighbor.x;
        float x_neighbor_raw = -1.0f;
        float y_neighbor_raw = -1.0f;
        if (current_active_roi &&
            x_neighbor_accepted) {
            x_neighbor_raw =
                CurrentRawDepth[x_neighbor_index];
        }
        if (current_active_roi &&
            y_neighbor_accepted) {
            y_neighbor_raw =
                CurrentRawDepth[y_neighbor_index];
        }
        float x_neighbor_depth =
            x_neighbor_accepted ?
            DepthTexture[x_neighbor] :
            d;
        float y_neighbor_depth =
            y_neighbor_accepted ?
            DepthTexture[y_neighbor] :
            d;
        if (!current_canonical_full_frame) {
            x_neighbor_accepted =
                x_neighbor_accepted &&
                !isnan(x_neighbor_raw) &&
                !isinf(x_neighbor_raw) &&
                x_neighbor_raw >= 0.0f &&
                !isnan(x_neighbor_depth) &&
                !isinf(x_neighbor_depth);
            y_neighbor_accepted =
                y_neighbor_accepted &&
                !isnan(y_neighbor_raw) &&
                !isinf(y_neighbor_raw) &&
                y_neighbor_raw >= 0.0f &&
                !isnan(y_neighbor_depth) &&
                !isinf(y_neighbor_depth);
        }
        float gx =
            (x_neighbor_accepted ? x_neighbor_depth : d) - d;
        float gy =
            (y_neighbor_accepted ? y_neighbor_depth : d) - d;
        float grad = sqrt(gx * gx + gy * gy);
        // Express every spatial-gradient threshold on the 434-short-side calibration grid.
        // The scale comes from target_w/target_h, which are the resolved TensorRT dimensions
        // after patch alignment and profile/native caps, rather than the requested config value.
        float reference_texel_scale = DepthReferenceTexelScale();
        float reference_grad = grad * reference_texel_scale;
        // Fixed controller thresholds: changing the independent EMA ablation knobs must not
        // silently alter scene classification.
        if (reference_grad >= EDGE_GRADIENT_THRESHOLD) {
            // Weight each edge texel by how far past the threshold it is, instead of counting it
            // once. Warp stress scales with the disparity STEP a silhouette produces, so a few
            // violent discontinuities outrank many gentle ones -- a distinction a threshold count
            // cannot make, and the reason edge-dense-but-soft scenes were classified alongside
            // sharp ones. Capped so a handful of extreme texels cannot dominate the frame. Scale
            // BOTH the linear weight and the cap: a one-texel discontinuity occupies a fraction
            // inverse to grid resolution, including after the linear term saturates.
            float weight = min(
                reference_grad * (1.0f / EDGE_GRADIENT_THRESHOLD),
                EDGE_WEIGHT_MAX * reference_texel_scale);
            InterlockedAdd(g_edge_count, (uint)(weight * EDGE_WEIGHT_SCALE + 0.5f));
        }
        bool previous_center_accepted =
            previous_tuple_readable &&
            (legacy_unbound_pair ||
             PreviousReliableValidity[dtid.xy] != 0u) &&
            (previous_canonical_full_frame ||
             FrameRoiAcceptedBoundsContains(
                 previous_accepted_bounds,
                 dtid.xy));
        if (previous_center_accepted &&
            abs(d - PreviousDepth[dtid.xy]) >= 0.05f) {
            InterlockedAdd(g_change_count, 1u);
        }

        // Broad replacement is measured on the exact display-referred model inputs that produced
        // the matched depths. Structural order comes from the separately preserved capture-domain
        // point maxRGB signal: it precedes HDR Reinhard and avoids nonlinear spatial mixing.
        // Under an identical monotone curve on every channel, pair order can only stay ordered or
        // collapse to a rejected tie. Codec noise, color matrices, and local tone mapping remain
        // outside that exposure model. Depth geometry is still the shot-reset authority.
        uint tile_center = (tid.y + 1u) * STRUCTURE_TILE_WIDTH + tid.x + 1u;
        float3 current_color = CurrentModelColor(dtid.xy);
        float3 previous_color =
            previous_center_accepted ?
            PreviousModelColor(dtid.xy) :
            current_color;
        float3 raw_rgb_delta = abs(current_color - previous_color);
        if (previous_center_accepted &&
            max(raw_rgb_delta.r, max(raw_rgb_delta.g, raw_rgb_delta.b)) >=
            RAW_RGB_PIXEL_DELTA) {
            InterlockedAdd(g_raw_rgb_change_count, 1u);
        }

        float current_samples[5];
        float previous_samples[5];
        bool current_stencil_valid = true;
        bool previous_stencil_valid =
            previous_tuple_readable;
        [unroll]
        for (int sample_index = 0; sample_index < 5; ++sample_index) {
            int2 offset = STRUCTURE_ORDINAL_OFFSETS[sample_index];
            int tile_index =
                (int)tile_center + offset.y * STRUCTURE_TILE_WIDTH + offset.x;
            current_samples[sample_index] =
                g_current_appearance_ordinal[tile_index];
            previous_samples[sample_index] =
                g_previous_appearance_ordinal[tile_index];
            int2 sample_pixel = clamp(
                int2(dtid.xy) + offset,
                int2(0, 0),
                int2((int)target_w - 1, (int)target_h - 1));
            if (!current_canonical_full_frame &&
                !FrameRoiAcceptedBoundsContains(
                    current_accepted_bounds,
                    (uint2)sample_pixel)) {
                current_stencil_valid = false;
            }
            uint sample_index_linear =
                (uint)sample_pixel.y * target_w +
                (uint)sample_pixel.x;
            float sample_raw = -1.0f;
            if (current_active_roi) {
                sample_raw =
                    CurrentRawDepth[sample_index_linear];
            }
            if (current_active_roi &&
                (isnan(sample_raw) ||
                 isinf(sample_raw) ||
                 sample_raw < 0.0f)) {
                current_stencil_valid = false;
            }
            if (!previous_tuple_readable) {
                previous_stencil_valid = false;
            } else {
                if (!legacy_unbound_pair &&
                    PreviousReliableValidity[
                        (uint2)sample_pixel] == 0u) {
                    previous_stencil_valid = false;
                }
                if (!previous_canonical_full_frame &&
                    !FrameRoiAcceptedBoundsContains(
                        previous_accepted_bounds,
                        (uint2)sample_pixel)) {
                    previous_stencil_valid = false;
                }
            }
        }
        uint current_comparisons = 0u;
        uint previous_comparisons = 0u;
        uint common_comparisons = 0u;
        uint ordering_flips = 0u;
        [unroll]
        for (int first = 0; first < 4; ++first) {
            [unroll]
            for (int second = first + 1; second < 5; ++second) {
                float current_delta = current_samples[first] - current_samples[second];
                float previous_delta = previous_samples[first] - previous_samples[second];
                bool current_reliable =
                    current_stencil_valid &&
                    abs(current_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR;
                bool previous_reliable =
                    previous_stencil_valid &&
                    abs(previous_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR;
                current_comparisons += current_reliable ? 1u : 0u;
                previous_comparisons += previous_reliable ? 1u : 0u;
                bool common_reliable = current_reliable && previous_reliable;
                if (common_reliable) {
                    ++common_comparisons;
                    if ((current_delta < 0.0f) != (previous_delta < 0.0f)) {
                        ++ordering_flips;
                    }
                }
            }
        }
        bool ordinal_changed =
            common_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON &&
            ordering_flips >= STRUCTURAL_ORDINAL_MIN_FLIPS &&
            ordering_flips * 2u >= common_comparisons;
        if (current_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_current_structural_support_count, 1u);
        }
        if (previous_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_previous_structural_support_count, 1u);
        }
        if (common_comparisons >= STRUCTURAL_ORDINAL_MIN_COMMON) {
            InterlockedAdd(g_common_structural_support_count, 1u);
        }
        if (ordinal_changed) {
            InterlockedAdd(g_structural_change_count, 1u);
        }

        // Subject smoothness uses the same reference-texel gradient. Otherwise changing the depth
        // grid would silently change which silhouette ramps can vote for recenter/stretch even if
        // adaptive-pop risk itself were normalized.
        // smooth_w = 1 - sigmoid(10 * (reference_grad - 0.025)).
        float smooth_w =
            1.0f - 1.0f / (1.0f + exp(-10.0f * (reference_grad - 0.025f)));

        // Center Gaussian in [-1,1] focus coordinates (Bestv2 sigmas: y 0.55, x 0.70).
        // Keep the exact historical arithmetic for canonical full-frame operation.
        float nx;
        float ny;
        if (current_canonical_full_frame) {
            nx = (float)dtid.x / (float)max(target_w - 1, 1u) * 2.0f - 1.0f;
            ny = (float)dtid.y / (float)max(target_h - 1, 1u) * 2.0f - 1.0f;
        } else {
            float2 model_uv =
                (float2(dtid.xy) + 0.5f) /
                float2(target_w, target_h);
            float2 source_uv;
            FrameRoiDataModelToSourceUv(
                current_transform,
                model_uv,
                source_uv);
            float4 focus =
                FrameRoiDataFocus(current_transform);
            float2 focus_uv =
                saturate(
                    (source_uv - focus.xy) /
                    max(focus.zw - focus.xy, 1e-8f.xx));
            nx = focus_uv.x * 2.0f - 1.0f;
            ny = focus_uv.y * 2.0f - 1.0f;
        }
        float center_w = exp(-0.5f * ((ny / 0.55f) * (ny / 0.55f) + (nx / 0.70f) * (nx / 0.70f)));

        float w = center_w * smooth_w;
        uint bin = min((uint)(saturate(d) * (float)NUM_BINS), NUM_BINS - 1u);
        InterlockedAdd(g_hist[bin], (uint)(w * 1024.0f + 0.5f));
        InterlockedAdd(g_plain[bin], 1u);  // unweighted, for the stretch 5/95 percentiles
    }

    GroupMemoryBarrierWithGroupSync();
    if (histogram_outputs_safe &&
        g_hist[lin] > 0u) {
        InterlockedAdd(SubjectHist[lin], g_hist[lin]);
    }
    if (histogram_outputs_safe &&
        g_plain[lin] > 0u) {
        InterlockedAdd(PlainHist[lin], g_plain[lin]);
    }
    if (histogram_outputs_safe &&
        lin == 0u) {
        InterlockedAdd(PlainHist[NUM_BINS], g_edge_count);
        InterlockedAdd(PlainHist[NUM_BINS + 1], g_change_count);
        InterlockedAdd(PlainHist[NUM_BINS + 2], g_structural_change_count);
        InterlockedAdd(PlainHist[NUM_BINS + 3], g_raw_rgb_change_count);
        InterlockedAdd(PlainHist[NUM_BINS + 4], g_current_structural_support_count);
        InterlockedAdd(PlainHist[NUM_BINS + 5], g_previous_structural_support_count);
        InterlockedAdd(PlainHist[NUM_BINS + 6], g_common_structural_support_count);
    }
}

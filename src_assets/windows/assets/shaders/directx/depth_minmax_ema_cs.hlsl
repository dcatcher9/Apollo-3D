// 1-thread pass: fold this frame's raw min/max into an EMA'd min/max (temporal scale
// stability, a la iw3's IncrementalEMAScaler), then reset the raw accumulator for the
// next frame's reduction. Runs on the D3D timeline right after depth_minmax_cs.
//
// w is this-frame validity: 0 = invalid/hold, 1 = valid with history, 2 = first valid frame.
RWStructuredBuffer<float4> MinMaxEma : register(u0);  // [0]={min,max,initialized,frame_state}
RWByteAddressBuffer        MinMaxRaw : register(u1);  // min bits, max bits, valid count
RWStructuredBuffer<uint>   Histogram : register(u2);  // permanent P2/P98 histogram from depth_hist_cs
// Append-only diagnostics. SubjectState[0..2] remain the production warp contract:
//   [3].zw = valid-depth fraction, effective EMA range width
//   [4].zw = empty-raw count, collapsed-raw count (stored as uint bits)
//   [5].xy = current range-collapsed flag, depth-ready flag
RWStructuredBuffer<float4> DiagnosticState : register(u3);
StructuredBuffer<uint4>    FrameRoiTransform : register(t0);
StructuredBuffer<uint4>    PreviousFrameRoiTransform : register(t1);

#include "include/depth_constants.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"
#include "include/sbs_frame_roi_transform.hlsl"

#define NUM_BINS 256

SbsFrameRoiTransformData LoadPreviousFrameRoiTransform() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(
        PreviousFrameRoiTransform);
}

bool TryTargetPixelCount(out uint count) {
    count = 0u;
    if (target_w == 0u ||
        target_h == 0u ||
        target_w > 0xffffffffu / target_h) {
        return false;
    }
    count = target_w * target_h;
    return true;
}

[numthreads(1, 1, 1)]
void main() {
    uint minmax_ema_count;
    uint minmax_ema_stride;
    MinMaxEma.GetDimensions(
        minmax_ema_count,
        minmax_ema_stride);
    uint accumulator_bytes;
    MinMaxRaw.GetDimensions(accumulator_bytes);
    uint histogram_count;
    uint histogram_stride;
    Histogram.GetDimensions(
        histogram_count,
        histogram_stride);
    uint diagnostic_count;
    uint diagnostic_stride;
    DiagnosticState.GetDimensions(
        diagnostic_count,
        diagnostic_stride);
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

    float new_min = asfloat(MinMaxRaw.Load(0));
    float new_max = asfloat(MinMaxRaw.Load(4));
    uint valid_count = MinMaxRaw.Load(8);
    SbsFrameRoiTransformData current_transform =
        FrameRoiTransformLoad();
    SbsFrameRoiTransformData previous_transform =
        LoadPreviousFrameRoiTransform();
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
    bool transform_dimensions_match =
        current_transform_valid &&
        all(FrameRoiDataModelDimensions(current_transform) ==
            uint2(target_w, target_h));
    bool current_transform_usable =
        legacy_unbound_pair ||
        transform_dimensions_match;
    bool previous_transform_resource_valid =
        previous_transform_vectors >= SBS_FRAME_ROI_VECTOR_COUNT &&
        previous_transform_stride == 16u &&
        FrameRoiDataValid(previous_transform);
    uint target_pixel_count;
    bool target_count_safe =
        TryTargetPixelCount(target_pixel_count);
    bool accumulator_safe =
        accumulator_bytes >=
            (current_transform_valid ? 16u : 12u);
    bool histogram_safe =
        histogram_count >= NUM_BINS &&
        histogram_stride == 4u;
    bool state_resources_safe =
        minmax_ema_count >= 1u &&
        minmax_ema_stride == 16u &&
        diagnostic_count >= SBS_ADAPTIVE_STATE_VECTOR_COUNT &&
        diagnostic_stride == 16u;
    uint reduced_accepted_focus_count = 0u;
    if (current_transform_valid &&
        accumulator_bytes >= 16u) {
        reduced_accepted_focus_count =
            MinMaxRaw.Load(12);
    }
    uint declared_accepted_focus_count =
        current_transform_valid ?
        FrameRoiDataAcceptedFocusPixelCount(current_transform) :
        target_pixel_count;
    uint accepted_focus_count =
        current_transform_valid ?
        reduced_accepted_focus_count :
        target_pixel_count;
    bool accepted_count_matches =
        accepted_focus_count ==
        declared_accepted_focus_count;
    bool geometry_reseed =
        current_transform_valid &&
        transform_dimensions_match &&
        (!previous_transform_resource_valid ||
        FrameRoiDataGeometryReseedRequired(
            current_transform,
            previous_transform));
    bool valid_bounds =
        target_count_safe &&
        accumulator_safe &&
        histogram_safe &&
        state_resources_safe &&
        current_transform_usable &&
        valid_count > 0u &&
        valid_count <= accepted_focus_count &&
        !isnan(new_min) &&
        !isinf(new_min) &&
        !isnan(new_max) &&
        !isinf(new_max) &&
        new_max >= new_min &&
        accepted_focus_count > 0u &&
        accepted_count_matches;
    float4 telemetry =
        state_resources_safe ?
        DiagnosticState[SBS_STATE_VECTOR_LATCHED_EDGE_FRACTION] :
        0.0f.xxxx;
    uint4 health_counters =
        state_resources_safe ?
        asuint(DiagnosticState[SBS_STATE_VECTOR_HARD_CUT_COUNT]) :
        0u.xxxx;
    float4 telemetry_flags =
        state_resources_safe ?
        DiagnosticState[SBS_STATE_VECTOR_RANGE_COLLAPSED] :
        0.0f.xxxx;

    // Robust percentile bounds: replace the raw min/max with the permanent P2/P98 percentiles
    // scanned from the 256-bin histogram (depth_hist_cs, binned over the raw range). Outlier
    // pixels -- DA-V2's near-spike tail is the classic case -- land in the extreme bins and
    // are excluded from the normalization range instead of squeezing the whole scene's
    // parallax. Bin centers quantize the bound to ~0.4% of the frame range, far below the
    // temporal EMA's smoothing. Values outside [lo,hi] saturate in buffer_to_tex_cs.
    {
        float bin_w = valid_bounds ? max(new_max - new_min, 1e-12f) / (float)NUM_BINS : 0.0f;
        float total = (float)valid_count;
        float lo_count = 0.02f * total;
        float hi_count = 0.98f * total;
        float pct_min = new_min;
        float pct_max = new_max;
        bool found_lo = false;
        bool found_hi = false;
        uint histogram_total = 0u;
        float cum = 0.0f;
        [loop]
        for (uint b = 0; histogram_safe && b < NUM_BINS; b++) {
            uint bin_count = Histogram[b];
            histogram_total += bin_count;
            cum += (float)bin_count;
            // Take the crossing bin's OUTER EDGE, not its center. A percentile only excludes its
            // nominal 2% when the distribution is smooth across the crossing bin; when a large
            // atom sits there -- sky, a far wall, DA-V2's flat background -- a bin-center bound
            // cuts through the atom and clips a large share of the frame to a single depth. c525
            // puts 66.1% of its pixels in bin 0, and the centered bound saturated 49.9% of the
            // frame to exactly 0. An outer edge can only widen the range, so it clips at most the
            // nominal fraction, and costs at most one bin (~0.4%) of range precision.
            if (!found_lo && cum >= lo_count) {
                pct_min = new_min + (float)b * bin_w;
                found_lo = true;
            }
            if (!found_hi && cum >= hi_count) {
                pct_max = new_min + ((float)b + 1.0f) * bin_w;
                found_hi = true;
            }
            Histogram[b] = 0u;  // reset for the next frame's accumulation
        }
        valid_bounds =
            valid_bounds &&
            histogram_total == valid_count;
        if (valid_bounds &&
            pct_max - pct_min > 1e-9f) {
            new_min = pct_min;
            new_max = pct_max;
        }
    }

    float collapse_scale = max(1.0f, max(abs(new_min), abs(new_max)));
    bool range_collapsed = valid_bounds &&
                           new_max - new_min <= collapse_scale * 1.0e-5f;
    if (!valid_bounds) {
        SBS_STATE_EMPTY_RAW_COUNT(health_counters) =
            min(SBS_STATE_EMPTY_RAW_COUNT(health_counters) + 1u, 0xfffffffeu);
    } else if (range_collapsed) {
        SBS_STATE_COLLAPSED_RAW_COUNT(health_counters) =
            min(SBS_STATE_COLLAPSED_RAW_COUNT(health_counters) + 1u, 0xfffffffeu);
    }
    SBS_STATE_VALID_DEPTH_FRACTION(telemetry) =
        (float)valid_count / (float)max(accepted_focus_count, 1u);

    float4 s =
        state_resources_safe ?
        MinMaxEma[0] :
        0.0f.xxxx;
    if (!valid_bounds) {
        // Preserve the last real normalization state and mark this frame as a hold. In particular,
        // do not manufacture an initialized [0,1] range when the first TensorRT result is invalid.
        s.w = 0.0f;
    } else if (geometry_reseed ||
               s.z < 0.5f || any(isnan(s.xy)) || any(isinf(s.xy))) {
        // First valid frame, including the first valid frame of a new ROI geometry: seed the
        // scale directly. An invalid first result never reaches this branch, so the old reliable
        // transform remains authoritative and reset debt carries into the next completion.
        s.x = new_min;
        s.y = new_max;
        s.z = 1.0f;
        s.w = 2.0f;
    } else {
        // Attack fast, release slow. A symmetric EMA lags the live percentiles, and whenever the
        // smoothed range is NARROWER than the frame's, buffer_to_tex_cs saturates the difference
        // away -- lag becomes clipped depth. Expanding immediately to cover [new_min, new_max]
        // makes that impossible, while contraction still decays at minmax_alpha. Expansion is also
        // the safe direction for stability: the range is a multiplicative gain, so growing it
        // LOWERS the gain, and it is fast SHRINKING that makes the depth scale breathe.
        s.x = min(lerp(s.x, new_min, minmax_alpha), new_min);
        s.y = max(lerp(s.y, new_max, minmax_alpha), new_max);
        s.w = 1.0f;
    }
    if (state_resources_safe) {
        MinMaxEma[0] = s;
    }
    SBS_STATE_EFFECTIVE_RAW_RANGE_WIDTH(telemetry) =
        s.z > 0.5f ? max(s.y - s.x, 0.0f) : 0.0f;
    SBS_STATE_RANGE_COLLAPSED(telemetry_flags) =
        range_collapsed ? 1.0f : 0.0f;
    SBS_STATE_DEPTH_READY(telemetry_flags) = s.z > 0.5f ? 1.0f : 0.0f;
    if (state_resources_safe) {
        DiagnosticState[SBS_STATE_VECTOR_LATCHED_EDGE_FRACTION] =
            telemetry;
        DiagnosticState[SBS_STATE_VECTOR_HARD_CUT_COUNT] =
            asfloat(health_counters);
        DiagnosticState[SBS_STATE_VECTOR_RANGE_COLLAPSED] =
            telemetry_flags;
    }

    // Reset accumulator so next frame's InterlockedMin/Max start from the identity.
    if (accumulator_bytes >= 12u) {
        MinMaxRaw.Store(0, 0xFFFFFFFFu);
        MinMaxRaw.Store(4, 0u);
        MinMaxRaw.Store(8, 0u);
    }
    if (accumulator_bytes >= 16u) {
        MinMaxRaw.Store(12, 0u);
    }
}

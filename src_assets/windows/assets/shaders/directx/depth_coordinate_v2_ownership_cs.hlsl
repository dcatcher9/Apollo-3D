// Conservatively realign an existing foreground depth cliff to the matching full-resolution
// source-color contour before the V2 safety envelopes run.
//
// This pass never invents a depth edge and never lowers parallax. A far-side boundary texel is
// eligible only when the candidate field supplies one unique cardinal near direction plus stable
// near/far plateaus. The corresponding source-color cell must contain one strong contour along
// that same normal. Its foreground coverage then raises only that mixed boundary texel toward the
// existing near plateau. Non-monotone texture, detected competing contours, and unsupported
// source geometry are exact no-ops. The later vertical share and row majorant remain the sole
// slope authorities.

Texture2D<float> Candidate : register(t0);
Texture2D<float4> SourceColor : register(t1);
Texture2D<uint> TensorExclusion : register(t2);
RWTexture2D<float> OwnershipRefined : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_color.hlsl"
#include "include/depth_coordinate_v2.hlsl"

bool LoadCandidate(int2 position, out float value) {
    value = 0.0f;
    if (position.x < 0 || position.y < 0 ||
        position.x >= (int)target_w || position.y >= (int)target_h) {
        return false;
    }
    if (TensorExclusion[position] != 0u) {
        return false;
    }
    value = Candidate.Load(int3(position, 0));
    return V2Finite(value);
}

#define OWNERSHIP_PROFILE_SAMPLES 10
#define OWNERSHIP_COVERAGE_MIN 0.10f
#define OWNERSHIP_COVERAGE_MAX 0.99f
#define OWNERSHIP_COVERAGE_ERROR 0.0002f

float3 OwnershipSrgbToLinear(float3 color) {
    color = saturate(color);
    float3 low = color / 12.92f;
    float3 high = pow((color + 0.055f) / 1.055f, 2.4f);
    return color <= 0.04045f ? low : high;
}

float3 SourceLinearColor(int2 position) {
    // Decode all capture modes through the same display-referred signal used by preprocessing,
    // then compare in the linear-RGB units on which the ownership oracle was calibrated.
    return OwnershipSrgbToLinear(DepthColorToSrgb(
        SourceColor.Load(int3(position, 0)).rgb,
        color_mode
    ));
}

float OwnershipProfileCoordinate(int index) {
    // Ten fixed quarter-cell samples span [-1.625, +0.625]. The source must be at least as large
    // as the DAV2 field, so these endpoints stay outside the one-source-pixel bilinear ramp for
    // every admitted crossing in [-0.99, +0.10], including a 99%-owned far cell at 720p.
    return (float)index * 0.25f - 1.625f;
}

float3 SampleSourceLinear(float2 source_position, uint2 source_size) {
    // Source positions use pixel-edge coordinates (texel centers are n + 0.5). Decode each texel
    // to linear light before interpolation; bilinear filtering in sRGB code space biases the
    // ownership crossing.
    float2 texel_position = source_position - 0.5f;
    float2 lower_position = floor(texel_position);
    int2 base = int2(lower_position);
    float2 fraction = texel_position - lower_position;
    int2 maximum = int2(source_size) - 1;
    int2 p00 = clamp(base, int2(0, 0), maximum);
    int2 p10 = clamp(base + int2(1, 0), int2(0, 0), maximum);
    int2 p01 = clamp(base + int2(0, 1), int2(0, 0), maximum);
    int2 p11 = clamp(base + int2(1, 1), int2(0, 0), maximum);
    float3 c00 = SourceLinearColor(p00);
    float3 c10 = SourceLinearColor(p10);
    float3 c01 = SourceLinearColor(p01);
    float3 c11 = SourceLinearColor(p11);
    float3 top = c00 + fraction.x * (c10 - c00);
    float3 bottom = c01 + fraction.x * (c11 - c01);
    return top + fraction.y * (bottom - top);
}

float3 OwnershipProfileColor(
    float profile_coordinate,
    float boundary,
    float tangent_center,
    float normal_step,
    float tangent_step,
    int normal_sign,
    bool horizontal_normal,
    uint2 source_size
) {
    float normal_position =
        boundary + profile_coordinate * (float)normal_sign * normal_step;
    float3 color = 0.0f;
    [unroll]
    for (int tangent_index = 0; tangent_index < 2; ++tangent_index) {
        float tangent_coordinate = tangent_index == 0 ? -0.25f : 0.25f;
        float tangent_position = tangent_center + tangent_coordinate * tangent_step;
        float2 source_position = horizontal_normal ?
            float2(normal_position, tangent_position) :
            float2(tangent_position, normal_position);
        color += 0.5f * SampleSourceLinear(source_position, source_size);
    }
    return color;
}

bool FullResolutionForegroundCoverage(
    uint2 target_position,
    int2 near_direction,
    out float coverage
) {
    coverage = 0.0f;
    uint source_w = 0u;
    uint source_h = 0u;
    SourceColor.GetDimensions(source_w, source_h);
    if (source_w < target_w || source_h < target_h ||
        target_w == 0u || target_h == 0u) {
        return false;
    }

    uint2 source_size = uint2(source_w, source_h);
    float2 source_scale = float2(source_size) / float2(target_w, target_h);
    float2 source_center = (float2(target_position) + 0.5f) * source_scale;
    bool horizontal_normal = near_direction.x != 0;
    int normal_sign = horizontal_normal ? near_direction.x : near_direction.y;
    float normal_step = horizontal_normal ? source_scale.x : source_scale.y;
    float tangent_step = horizontal_normal ? source_scale.y : source_scale.x;
    float normal_center = horizontal_normal ? source_center.x : source_center.y;
    float tangent_center = horizontal_normal ? source_center.y : source_center.x;
    float boundary = normal_center + 0.5f * (float)normal_sign * normal_step;
    float3 colors[OWNERSHIP_PROFILE_SAMPLES];
    float projected[OWNERSHIP_PROFILE_SAMPLES];
    float filtered[OWNERSHIP_PROFILE_SAMPLES];
    float derivatives[OWNERSHIP_PROFILE_SAMPLES];

    [unroll]
    for (int sample_index = 0; sample_index < OWNERSHIP_PROFILE_SAMPLES; ++sample_index) {
        colors[sample_index] = OwnershipProfileColor(
            OwnershipProfileCoordinate(sample_index),
            boundary,
            tangent_center,
            normal_step,
            tangent_step,
            normal_sign,
            horizontal_normal,
            source_size
        );
    }

    // With source dimensions no smaller than the DAV2 field, the -1.625/+0.625 endpoints stay
    // beyond the bilinear ramp for every admitted crossing in [-0.99,+0.10]. Nearer "plateau"
    // averages contaminate the far reference at high foreground coverage and make the result
    // depend on source raster phase.
    float3 far_color = colors[0];
    float3 near_color = colors[OWNERSHIP_PROFILE_SAMPLES - 1];
    float3 color_axis = near_color - far_color;
    float contrast_squared = dot(color_axis, color_axis);
    if (contrast_squared < 0.01f) {
        return false;
    }

    float previous_projection = 0.0f;
    [loop]
    for (int project_index = 0; project_index < OWNERSHIP_PROFILE_SAMPLES; ++project_index) {
        projected[project_index] =
            dot(colors[project_index] - far_color, color_axis) / contrast_squared;
        // A single foreground-ownership contour is monotone along its declared far-to-near
        // normal. Check the raw profile before smoothing: otherwise two source-pixel-scale
        // opposite transitions can merge into one filtered derivative cluster at 4K.
        if (project_index > 0 && projected[project_index] + 1e-5f < previous_projection) {
            return false;
        }
        previous_projection = projected[project_index];
    }
    [loop]
    for (int filter_index = 0; filter_index < OWNERSHIP_PROFILE_SAMPLES; ++filter_index) {
        float previous = 0.0f;
        float next = 0.0f;
        if (filter_index > 0) {
            previous = projected[filter_index - 1];
        }
        if (filter_index + 1 < OWNERSHIP_PROFILE_SAMPLES) {
            next = projected[filter_index + 1];
        }
        filtered[filter_index] =
            0.25f * previous + 0.5f * projected[filter_index] + 0.25f * next;
    }

    int crossing_count = 0;
    float edge_coordinate = 0.0f;
    int edge_interval = -1;
    [loop]
    for (int crossing_index = 0;
         crossing_index + 1 < OWNERSHIP_PROFILE_SAMPLES;
         ++crossing_index) {
        float coordinate = OwnershipProfileCoordinate(crossing_index);
        float before = filtered[crossing_index];
        float after = filtered[crossing_index + 1];
        if (before < 0.5f && after >= 0.5f) {
            float fraction = (0.5f - before) / max(after - before, 1e-8f);
            edge_coordinate = lerp(
                coordinate,
                OwnershipProfileCoordinate(crossing_index + 1),
                fraction
            );
            edge_interval = crossing_index;
            ++crossing_count;
        }
    }
    if (crossing_count != 1) {
        return false;
    }

    // The filtered profile is deliberately used to decide whether there is one trustworthy
    // contour. Its symmetric smoothing shifts a valid sub-cell crossing by roughly 0.005 model
    // cells at some 720p pixel phases, however, which must not change ownership solely because
    // capture resolution changed. Measure coverage from the unfiltered samples in the already
    // authenticated interval; if that interval does not itself bracket the crossing, abstain.
    float raw_before = projected[edge_interval];
    float raw_after = projected[edge_interval + 1];
    if (!(raw_before <= 0.5f && raw_after >= 0.5f)) {
        return false;
    }

    float maximum_derivative = -3.402823466e+38f;
    [loop]
    for (int derivative_index = 0;
         derivative_index < OWNERSHIP_PROFILE_SAMPLES;
         ++derivative_index) {
        float derivative;
        if (derivative_index == 0) {
            derivative = 4.0f * (filtered[1] - filtered[0]);
        } else if (derivative_index + 1 == OWNERSHIP_PROFILE_SAMPLES) {
            derivative = 4.0f *
                (filtered[derivative_index] - filtered[derivative_index - 1]);
        } else {
            derivative = 2.0f *
                (filtered[derivative_index + 1] - filtered[derivative_index - 1]);
        }
        derivatives[derivative_index] = derivative;
        maximum_derivative = max(maximum_derivative, derivative);
    }
    float derivative_threshold = max(0.25f, 0.20f * maximum_derivative);
    float primary_mass = 0.0f;
    float secondary_mass = 0.0f;
    int cluster_index = 0;
    [loop]
    while (cluster_index < OWNERSHIP_PROFILE_SAMPLES) {
        bool active = derivatives[cluster_index] > derivative_threshold;
        if (!active) {
            ++cluster_index;
            continue;
        }
        int cluster_start = cluster_index;
        int cluster_end = cluster_index;
        [loop]
        while (cluster_end + 1 < OWNERSHIP_PROFILE_SAMPLES &&
               derivatives[cluster_end + 1] > derivative_threshold) {
            ++cluster_end;
        }
        float mass = 0.0f;
        if (cluster_end > cluster_start) {
            [loop]
            for (int mass_index = cluster_start; mass_index < cluster_end; ++mass_index) {
                mass += 0.5f *
                    (derivatives[mass_index] + derivatives[mass_index + 1]) *
                    0.25f;
            }
        } else {
            mass = derivatives[cluster_start] * 0.25f;
        }
        if (mass > primary_mass) {
            secondary_mass = primary_mass;
            primary_mass = mass;
        } else {
            secondary_mass = max(secondary_mass, mass);
        }
        cluster_index = cluster_end + 1;
    }
    if (primary_mass <= 0.0f || secondary_mass >= 0.5f * primary_mass) {
        return false;
    }

    // Refine only the authenticated raw bracket. A single secant across the 0.25-cell coarse
    // interval is biased once a 4K one-pixel color ramp becomes narrower than that interval.
    // Five bounded bisections reduce the remaining interval below 0.008 model cells without
    // increasing work for rejected depth cliffs or ambiguous color profiles.
    float low_coordinate = OwnershipProfileCoordinate(edge_interval);
    float high_coordinate = OwnershipProfileCoordinate(edge_interval + 1);
    float low_projection = raw_before;
    float high_projection = raw_after;
    [unroll]
    for (int refinement = 0; refinement < 5; ++refinement) {
        float middle_coordinate = 0.5f * (low_coordinate + high_coordinate);
        float3 middle_color = OwnershipProfileColor(
            middle_coordinate,
            boundary,
            tangent_center,
            normal_step,
            tangent_step,
            normal_sign,
            horizontal_normal,
            source_size
        );
        float middle_projection =
            dot(middle_color - far_color, color_axis) / contrast_squared;
        if (!V2Finite(middle_projection)) {
            return false;
        }
        // Preserve the same monotone-contour contract inside the refined bracket. This reuses
        // the samples already required for sub-cell coverage and catches source-pixel-scale
        // rise/fall/rise structure that can alias between the coarse quarter-cell samples.
        if (middle_projection + 1e-5f < low_projection ||
            middle_projection > high_projection + 1e-5f) {
            return false;
        }
        if (middle_projection < 0.5f) {
            low_coordinate = middle_coordinate;
            low_projection = middle_projection;
        } else {
            high_coordinate = middle_coordinate;
            high_projection = middle_projection;
        }
    }
    float refined_fraction =
        (0.5f - low_projection) / max(high_projection - low_projection, 1e-8f);
    edge_coordinate = lerp(low_coordinate, high_coordinate, refined_fraction);
    float measured_coverage = saturate(-edge_coordinate);
    if (measured_coverage < OWNERSHIP_COVERAGE_MIN - OWNERSHIP_COVERAGE_ERROR ||
        measured_coverage > OWNERSHIP_COVERAGE_MAX + OWNERSHIP_COVERAGE_ERROR) {
        return false;
    }

    // The fixed refinement's measured worst-case phase error is below 1.1e-4 model cells across
    // every authenticated long side. Admit a 2e-4 error band, then publish the exact contract
    // interval so an almost-threshold estimate cannot weaken or exceed the requested bounds.
    coverage = clamp(
        measured_coverage,
        OWNERSHIP_COVERAGE_MIN,
        OWNERSHIP_COVERAGE_MAX
    );
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= target_w || id.y >= target_h) {
        return;
    }

    int2 position = int2(id.xy);
    float center = Candidate.Load(int3(position, 0));
    if (!V2Finite(center)) {
        OwnershipRefined[id.xy] = 0.0f;
        return;
    }
    // Exclusion is an exact no-op for the center cell. The tensor cell is outside every
    // ownership decision, but its candidate value still has to pass unchanged through this
    // intermediate field so the later zero-plane conditioner remains the sole overlay-depth
    // authority. LoadCandidate intentionally remains exclusion-aware for every neighbor/stencil.
    if (TensorExclusion[position] != 0u) {
        OwnershipRefined[id.xy] = center;
        return;
    }

    static const int2 directions[4] = {
        int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)
    };
    float best_jump = 0.0f;
    float second_jump = 0.0f;
    float near_value = center;
    int best_direction = -1;
    [unroll]
    for (int direction = 0; direction < 4; ++direction) {
        float neighbor;
        if (!LoadCandidate(position + directions[direction], neighbor)) {
            continue;
        }
        float jump = neighbor - center;
        if (jump > best_jump) {
            second_jump = best_jump;
            best_jump = jump;
            near_value = neighbor;
            best_direction = direction;
        } else {
            second_jump = max(second_jump, jump);
        }
    }

    float cliff_floor = 8.0f * v2_max_horizontal_slope / max((float)target_w, 1.0f);
    if (best_direction < 0 || best_jump < cliff_floor ||
        second_jump * 2.0f >= best_jump) {
        OwnershipRefined[id.xy] = center;
        return;
    }

    int2 normal = directions[best_direction];
    float near_two;
    float far_one;
    if (!LoadCandidate(position + normal * 2, near_two) ||
        !LoadCandidate(position - normal, far_one) ||
        4.0f * abs(near_two - near_value) > best_jump ||
        4.0f * abs(center - far_one) > best_jump) {
        OwnershipRefined[id.xy] = center;
        return;
    }

    float foreground_coverage;
    if (!FullResolutionForegroundCoverage(id.xy, normal, foreground_coverage)) {
        OwnershipRefined[id.xy] = center;
        return;
    }

    // Fractional pull represents a mixed source cell without the full-texel dilation that the
    // offline crown trial showed enlarges the later safety collar. This remains in [far, near].
    OwnershipRefined[id.xy] = max(
        center,
        lerp(center, near_value, foreground_coverage)
    );
}

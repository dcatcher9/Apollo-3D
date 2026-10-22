// Five-center nearest-source-pixel Euclidean transform. The separable lower-envelope
// algorithm uses FP32 distances at real field-node positions; native tests bound rounding error.
// Centers are EVERY current admitted native pixel plus its up/down/left/right offsets of 2.
// Virtual centers outside the native crop are retained. No projected pins or L1 distances feed
// this transform. Dispatch both entries through the receipt-gated grid16 slot; inactive groups
// return uniformly. The resulting q is independent of sigma: the proposal writer may
// evaluate exp(-q/(2*sigma*sigma)) and the user's literal UI/Base blend.
//
// horizontal_main: one 64-thread group per extended source row; writes R32_FLOAT
//   scratch(field_width, crop_height+4).
// vertical_main: one 64-thread group per field column; reads that distinct scratch and writes
//   R32_FLOAT q(field_width, field_height). All padding/empty/invalid cells are positive infinity.
// Unbind each UAV before using its SRV. No input aliases u0. Both inputs remain immutable.

#include "include/depth_coordinate_v2_contract.generated.hlsl"

Texture2D<uint> NativeSupport : register(t0);
Texture2D<float> HorizontalSquaredDistance : register(t1);
RWTexture2D<float> SquaredDistanceOutput : register(u0);

cbuffer GaussianDistanceConstants : register(b0) {
    uint4 gaussian_source;  // native source W,H,crop_top,crop_height
    uint4 gaussian_field;   // field W,H,reserved0,reserved1
    uint4 gaussian_content; // half-open actual content rectangle
};

#define GAUSSIAN_LANES 64u
#define GAUSSIAN_CENTER_OFFSET ((int)V2_SUBTITLE_CONTOUR_GAUSSIAN_CENTER_OFFSET_SOURCE_PIXELS)
#define GAUSSIAN_MAX_EXTENDED_WIDTH 5124u
#define GAUSSIAN_MAX_EXTENDED_HEIGHT 858u
#define GAUSSIAN_MAX_FIELD_SHORT_SIDE 868u
#define GAUSSIAN_MAX_FIELD_LONG_SIDE 3080u

// Encoded source indices are index+1, with 0 meaning no seed. The completed prefix is monotone,
// so a binary search finds the first successor without a second full-width shared array.
groupshared uint GaussianPrefix[GAUSSIAN_MAX_EXTENDED_WIDTH];
groupshared uint GaussianChunkLast[GAUSSIAN_LANES];
groupshared uint GaussianChunkCarry[GAUSSIAN_LANES];

// Both entry points combined declare 31,316 bytes of groupshared storage, below SM5's 32 KiB.
// Only the vertical entry consumes this sorted parabola hull.
groupshared float GaussianRowCost[GAUSSIAN_MAX_EXTENDED_HEIGHT];
groupshared uint GaussianHullRow[GAUSSIAN_MAX_EXTENDED_HEIGHT];
groupshared float GaussianHullStart[GAUSSIAN_MAX_EXTENDED_HEIGHT + 1u];
groupshared uint GaussianHullCount;
groupshared uint GaussianColumnValid;

float GaussianInfinity() {
    return asfloat(V2_SUBTITLE_CONTOUR_DISTANCE_EMPTY_BITS);
}

bool GaussianDomainValid() {
    uint expected_top = 0u, expected_height = 0u;
    return all(gaussian_source.xy != 0u) &&
        all(gaussian_source.xy <= V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) &&
        V2SubtitleOcrComputeCrop(gaussian_source.x, gaussian_source.y, expected_top, expected_height) &&
        gaussian_source.z == expected_top && gaussian_source.w == expected_height &&
        gaussian_source.x + 4u <= GAUSSIAN_MAX_EXTENDED_WIDTH &&
        gaussian_source.w + 4u <= GAUSSIAN_MAX_EXTENDED_HEIGHT &&
        all(gaussian_field.xy != 0u) && all(gaussian_field.zw == 0u) &&
        min(gaussian_field.x, gaussian_field.y) <= GAUSSIAN_MAX_FIELD_SHORT_SIDE &&
        max(gaussian_field.x, gaussian_field.y) <= GAUSSIAN_MAX_FIELD_LONG_SIDE &&
        all(gaussian_content.xy < gaussian_content.zw) &&
        all(gaussian_content.zw <= gaussian_field.xy);
}

bool GaussianNativeSeed(int2 p) {
    if (any(p < int2(0, 0)) || p.x >= (int)gaussian_source.x || p.y >= (int)gaussian_source.w) return false;
    // NativeSupport is the admitted producer mask. Current authority/SCT validation remains
    // downstream; this transform cannot authorize stale, missing or malformed current tuples.
    return NativeSupport.Load(int3(p, 0)) != 0u;
}

bool GaussianExpandedSeed(int2 center) {
    return GaussianNativeSeed(center) ||
        GaussianNativeSeed(center + int2(GAUSSIAN_CENTER_OFFSET, 0)) ||
        GaussianNativeSeed(center - int2(GAUSSIAN_CENTER_OFFSET, 0)) ||
        GaussianNativeSeed(center + int2(0, GAUSSIAN_CENTER_OFFSET)) ||
        GaussianNativeSeed(center - int2(0, GAUSSIAN_CENTER_OFFSET));
}

float GaussianSourceX(uint field_x) {
    precise float node = (float)(field_x - gaussian_content.x) + 0.5f;
    precise float source = node * (float)gaussian_source.x /
        (float)(gaussian_content.z - gaussian_content.x) - 0.5f;
    return source;
}

float GaussianSourceExtendedY(uint field_y) {
    precise float node = (float)(field_y - gaussian_content.y) + 0.5f;
    precise float source = node * (float)gaussian_source.y /
        (float)(gaussian_content.w - gaussian_content.y) - 0.5f;
    precise float extended = source - (float)gaussian_source.z + (float)GAUSSIAN_CENTER_OFFSET;
    return extended;
}

[numthreads(64, 1, 1)]
void horizontal_main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    uint output_w, output_h, input_w, input_h;
    SquaredDistanceOutput.GetDimensions(output_w, output_h);
    NativeSupport.GetDimensions(input_w, input_h);
    uint grid_width = (gaussian_field.x + 15u) / 16u;
    uint row = group_id.x + group_id.y * grid_width;
    // This branch is uniform for every lane; no lane returns ahead of a group barrier.
    if (row >= output_h) return;
    bool valid = GaussianDomainValid() && input_w == gaussian_source.x && input_h == gaussian_source.w &&
        output_w == gaussian_field.x && output_h == gaussian_source.w + 4u;
    if (!valid) {
        [loop]
        for (uint x = lane; x < output_w; x += GAUSSIAN_LANES) {
            SquaredDistanceOutput[uint2(x, row)] = GaussianInfinity();
        }
        return;
    }

    uint extended_w = gaussian_source.x + 4u;
    uint begin = lane * extended_w / GAUSSIAN_LANES;
    uint end = (lane + 1u) * extended_w / GAUSSIAN_LANES;
    uint nearest = 0u;
    [loop]
    for (uint x = begin; x < end; ++x) {
        if (GaussianExpandedSeed(int2((int)x - GAUSSIAN_CENTER_OFFSET, (int)row - GAUSSIAN_CENTER_OFFSET))) {
            nearest = x + 1u;
        }
        GaussianPrefix[x] = nearest;
    }
    GaussianChunkLast[lane] = nearest;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        uint carry = 0u;
        [loop]
        for (uint chunk = 0u; chunk < GAUSSIAN_LANES; ++chunk) {
            GaussianChunkCarry[chunk] = carry;
            if (GaussianChunkLast[chunk] != 0u) carry = GaussianChunkLast[chunk];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (uint x = begin; x < end; ++x) {
        if (GaussianPrefix[x] == 0u) GaussianPrefix[x] = GaussianChunkCarry[lane];
    }
    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint x = lane; x < output_w; x += GAUSSIAN_LANES) {
        float best = GaussianInfinity();
        if (x >= gaussian_content.x && x < gaussian_content.z) {
            precise float source_x = GaussianSourceX(x);
            precise float extended_x = source_x + (float)GAUSSIAN_CENTER_OFFSET;
            uint floor_x = (uint)clamp(floor(extended_x), 0.0f, (float)(extended_w - 1u));
            uint left = GaussianPrefix[floor_x];
            if (left != 0u) {
                precise float dx = source_x - (float)((int)left - 1 - GAUSSIAN_CENTER_OFFSET);
                best = dx * dx;
            }
            if (GaussianPrefix[extended_w - 1u] > floor_x + 1u) {
                uint lo = floor_x + 1u, hi = extended_w - 1u;
                [loop]
                while (lo < hi) {
                    uint mid = lo + (hi - lo) / 2u;
                    if (GaussianPrefix[mid] > floor_x + 1u) hi = mid;
                    else lo = mid + 1u;
                }
                uint right = GaussianPrefix[lo];
                precise float dx = source_x - (float)((int)right - 1 - GAUSSIAN_CENTER_OFFSET);
                precise float cost = dx * dx;
                // Equal computed costs retain the lower source x. No output argmin list is required.
                if (cost < best) best = cost;
            }
        }
        SquaredDistanceOutput[uint2(x, row)] = best;
    }
}

float GaussianParabolaCost(uint row, float extended_y) {
    precise float dy = extended_y - (float)row;
    precise float squared = dy * dy;
    precise float cost = GaussianRowCost[row] + squared;
    return cost;
}

[numthreads(64, 1, 1)]
void vertical_main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    uint output_w, output_h, input_w, input_h;
    SquaredDistanceOutput.GetDimensions(output_w, output_h);
    HorizontalSquaredDistance.GetDimensions(input_w, input_h);
    uint grid_width = (gaussian_field.x + 15u) / 16u;
    uint column = group_id.x + group_id.y * grid_width;
    if (column >= output_w) return;
    bool valid = GaussianDomainValid() && input_w == gaussian_field.x && input_h == gaussian_source.w + 4u &&
        output_w == gaussian_field.x && output_h == gaussian_field.y;
    if (!valid || column < gaussian_content.x || column >= gaussian_content.z) {
        [loop]
        for (uint y = lane; y < output_h; y += GAUSSIAN_LANES) {
            SquaredDistanceOutput[uint2(column, y)] = GaussianInfinity();
        }
        return;
    }

    uint rows = gaussian_source.w + 4u;
    [loop]
    for (uint row = lane; row < rows; row += GAUSSIAN_LANES) {
        GaussianRowCost[row] = HorizontalSquaredDistance.Load(int3(column, row, 0));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        uint count = 0u;
        GaussianColumnValid = 1u;
        precise float maximum_dx = (float)(gaussian_source.x + 4u);
        precise float maximum_cost = maximum_dx * maximum_dx;
        [loop]
        for (uint row = 0u; row < rows; ++row) {
            float cost = GaussianRowCost[row];
            if (cost == GaussianInfinity()) continue;
            if (!isfinite(cost) || cost < 0.0f || cost > maximum_cost) {
                GaussianColumnValid = 0u;
                break;
            }
            float start = -GaussianInfinity();
            [loop]
            while (count != 0u) {
                uint prior = GaussianHullRow[count - 1u];
                // Difference-first form avoids subtracting two large y-squared intercepts.
                precise float delta_cost = cost - GaussianRowCost[prior];
                precise float delta_row = (float)(row - prior);
                precise float midpoint = 0.5f * (float)(row + prior);
                start = midpoint + delta_cost / (2.0f * delta_row);
                if (start > GaussianHullStart[count - 1u]) break;
                --count;
            }
            GaussianHullRow[count] = row;
            GaussianHullStart[count] = count == 0u ? -GaussianInfinity() : start;
            ++count;
        }
        GaussianHullStart[count] = GaussianInfinity();
        GaussianHullCount = count;
    }
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (uint y = lane; y < output_h; y += GAUSSIAN_LANES) {
        float best = GaussianInfinity();
        if (GaussianColumnValid != 0u && GaussianHullCount != 0u &&
            y >= gaussian_content.y && y < gaussian_content.w) {
            precise float extended_y = GaussianSourceExtendedY(y);
            uint lo = 0u, hi = GaussianHullCount;
            [loop]
            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2u;
                if (GaussianHullStart[mid] < extended_y) lo = mid + 1u;
                else hi = mid;
            }
            uint selected = lo == 0u ? 0u : lo - 1u;
            best = GaussianParabolaCost(GaussianHullRow[selected], extended_y);
            // Recheck adjacent retained hull parabolas at the actual query coordinate to
            // reduce sensitivity to a rounded breakpoint. Native tests bound FP32 error.
            if (selected != 0u) {
                float prior_cost = GaussianParabolaCost(GaussianHullRow[selected - 1u], extended_y);
                if (prior_cost <= best) best = prior_cost;
            }
            if (selected + 1u < GaussianHullCount) {
                float next_cost = GaussianParabolaCost(GaussianHullRow[selected + 1u], extended_y);
                if (next_cost < best) best = next_cost;
            }
        }
        SquaredDistanceOutput[uint2(column, y)] = best;
    }
}

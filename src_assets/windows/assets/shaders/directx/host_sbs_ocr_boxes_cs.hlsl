#include "include/depth_coordinate_v2_contract.generated.hlsl"

// OCR6 is an authenticated generated ABI.  A stale header must fail compilation rather than
// publishing a locally reconstructed same-sized record.
#if !defined(V2_OCR_OUTPUT_N) || !defined(V2_OCR_OUTPUT_C) || \
    !defined(V2_OCR_OUTPUT_WIDTH) || !defined(V2_OCR_OUTPUT_HEIGHT) || \
    !defined(V2_OCR_RECORD_SCHEMA) || !defined(V2_OCR_RECORD_TAG) || \
    !defined(V2_OCR_RECORD_WORD_COUNT) || !defined(V2_OCR_RECORD_HEADER_WORD_COUNT) || \
    !defined(V2_OCR_BOX_WORD_COUNT) || !defined(V2_OCR_RAW_BOX_OFFSET) || \
    !defined(V2_OCR_RAW_BOX_CAPACITY) || !defined(V2_OCR_FINAL_BOX_OFFSET) || \
    !defined(V2_OCR_FINAL_BOX_CAPACITY) || !defined(V2_OCR_FIELD_WIDTH) || \
    !defined(V2_OCR_FIELD_HEIGHT) || !defined(V2_OCR_ROI_TOP) || \
    !defined(V2_OCR_ROI_BOTTOM)
#error "Generated V2 OCR6 record contract macros are required"
#endif

#define OCR_WIDTH V2_OCR_OUTPUT_WIDTH
#define OCR_HEIGHT V2_OCR_OUTPUT_HEIGHT
#define OCR_CELL_WIDTH 8u
#define OCR_GRID_WIDTH (OCR_WIDTH / OCR_CELL_WIDTH)
#define OCR_CELL_WORDS 8u
#define OCR6_WORDS V2_OCR_RECORD_WORD_COUNT
#define OCR6_RAW_OFFSET V2_OCR_RAW_BOX_OFFSET
#define OCR6_RAW_CAPACITY V2_OCR_RAW_BOX_CAPACITY
#define OCR6_FINAL_CAPACITY V2_OCR_FINAL_BOX_CAPACITY
#define OCR6_BOX_WORDS V2_OCR_BOX_WORD_COUNT
#define OCR6_FINAL_OFFSET V2_OCR_FINAL_BOX_OFFSET
#define OCR6_SCHEMA V2_OCR_RECORD_SCHEMA
#define OCR6_TAG V2_OCR_RECORD_TAG
#define OCR6_VALID 1u
#define OCR6_OVERFLOW 2u
#define OCR6_NONFINITE 4u
#define FIELD_ROI_TOP V2_OCR_ROI_TOP
#define FIELD_ROI_BOTTOM V2_OCR_ROI_BOTTOM

StructuredBuffer<float> ProbabilityMap : register(t0);
RWStructuredBuffer<uint> CellStatsWrite : register(u0);

groupshared uint active_count;
groupshared uint probability_sum_q12;
groupshared uint min_x;
groupshared uint max_x;
groupshared uint min_y;
groupshared uint max_y;
groupshared uint nonfinite_count;

bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

// One 8x1 group emits a compact exact-pixel bound and confidence summary. Keeping model rows
// separate prevents two close bilingual lines from being fused by a coarse vertical cell.
[numthreads(OCR_CELL_WIDTH, 1, 1)]
void cells_main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    if (group_thread_id.x == 0u) {
        active_count = 0u;
        probability_sum_q12 = 0u;
        min_x = 0xffffffffu;
        max_x = 0u;
        min_y = 0xffffffffu;
        max_y = 0u;
        nonfinite_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint2 pixel = uint2(group_id.x * OCR_CELL_WIDTH + group_thread_id.x, group_id.y);
    if (pixel.x < OCR_WIDTH && pixel.y < OCR_HEIGHT) {
        float probability = ProbabilityMap[pixel.y * OCR_WIDTH + pixel.x];
        if (!FiniteFloat(probability)) {
            InterlockedAdd(nonfinite_count, 1u);
        } else if (probability > 0.2f) {
            InterlockedAdd(active_count, 1u);
            InterlockedAdd(
                probability_sum_q12,
                (uint)round(saturate(probability) * 4095.0f));
            InterlockedMin(min_x, pixel.x);
            InterlockedMax(max_x, pixel.x + 1u);
            InterlockedMin(min_y, pixel.y);
            InterlockedMax(max_y, pixel.y + 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (group_thread_id.x == 0u) {
        uint base = (group_id.y * OCR_GRID_WIDTH + group_id.x) * OCR_CELL_WORDS;
        CellStatsWrite[base + 0u] = active_count;
        CellStatsWrite[base + 1u] = probability_sum_q12;
        CellStatsWrite[base + 2u] = min_x;
        CellStatsWrite[base + 3u] = max_x;
        CellStatsWrite[base + 4u] = min_y;
        CellStatsWrite[base + 5u] = max_y;
        CellStatsWrite[base + 6u] = nonfinite_count;
        CellStatsWrite[base + 7u] = 0u;
    }
}

// Keep resolve bindings distinct from cells_main. D3DCompiler validates the complete source file
// before entrypoint dead-code elimination and rejects two differently typed globals on one slot.
StructuredBuffer<uint> CellStats : register(t1);
RWStructuredBuffer<uint> OcrBoxRecord : register(u1);

cbuffer OcrResolveConstants : register(b0) {
    uint frame_lo;
    uint frame_hi;
    uint analysis_generation_lo;
    uint analysis_generation_hi;
    uint source_width;
    uint source_height;
    uint field_width;
    uint field_height;
    uint crop_top_pixels;
    uint crop_height_pixels;
    uint2 resolve_reserved;
};

uint CellBase(uint x, uint y) {
    return (y * OCR_GRID_WIDTH + x) * OCR_CELL_WORDS;
}

bool RowActive(uint y) {
    [loop]
    for (uint x = 0u; x < OCR_GRID_WIDTH; ++x) {
        if (CellStats[CellBase(x, y)] != 0u) return true;
    }
    return false;
}

bool ColumnActive(uint x, uint y0, uint y1) {
    [loop]
    for (uint y = y0; y < y1; ++y) {
        if (CellStats[CellBase(x, y)] != 0u) return true;
    }
    return false;
}

uint MapXFloor(uint x) {
    return min((uint)floor((float)x * (float)field_width / (float)OCR_WIDTH), field_width);
}

uint MapXCeil(uint x) {
    return min((uint)ceil((float)x * (float)field_width / (float)OCR_WIDTH), field_width);
}

uint MapYFloor(uint y) {
    float source_y = (float)crop_top_pixels +
        (float)y * (float)crop_height_pixels / (float)OCR_HEIGHT;
    uint mapped = min(
        (uint)floor(source_y * (float)field_height / (float)source_height),
        field_height);
    return clamp(mapped, FIELD_ROI_TOP, FIELD_ROI_BOTTOM);
}

uint MapYCeil(uint y) {
    float source_y = (float)crop_top_pixels +
        (float)y * (float)crop_height_pixels / (float)OCR_HEIGHT;
    uint mapped = min(
        (uint)ceil(source_y * (float)field_height / (float)source_height),
        field_height);
    return clamp(mapped, FIELD_ROI_TOP, FIELD_ROI_BOTTOM);
}

void StoreBox(uint base, uint index, uint4 box, float score) {
    uint offset = base + index * OCR6_BOX_WORDS;
    OcrBoxRecord[offset + 0u] = box.x;
    OcrBoxRecord[offset + 1u] = box.y;
    OcrBoxRecord[offset + 2u] = box.z;
    OcrBoxRecord[offset + 3u] = box.w;
    OcrBoxRecord[offset + 4u] = asuint(score);
    OcrBoxRecord[offset + 5u] = 0u;
    OcrBoxRecord[offset + 6u] = 0u;
    OcrBoxRecord[offset + 7u] = 0u;
}

uint4 MapBox(uint left, uint top, uint right, uint bottom) {
    return uint4(MapXFloor(left), MapYFloor(top), MapXCeil(right), MapYCeil(bottom));
}

// Deterministic subtitle-oriented DB postprocess. It preserves exact model-pixel bounds, joins
// character groups only across modest horizontal gaps, and therefore keeps the far-right logo as
// a separate square candidate for SLR6's generic horizontal geometry filter to reject.
[numthreads(1, 1, 1)]
void resolve_main(uint3 id : SV_DispatchThreadID) {
    [loop]
    for (uint word = 0u; word < OCR6_WORDS; ++word) OcrBoxRecord[word] = 0u;

    uint status = OCR6_VALID;
    [loop]
    for (uint cell = 0u; cell < OCR_GRID_WIDTH * OCR_HEIGHT; ++cell) {
        if (CellStats[cell * OCR_CELL_WORDS + 6u] != 0u) status |= OCR6_NONFINITE;
    }

    uint raw_total = 0u;
    uint final_total = 0u;
    uint y = 0u;
    [loop]
    while (y < OCR_HEIGHT) {
        while (y < OCR_HEIGHT && !RowActive(y)) ++y;
        if (y >= OCR_HEIGHT) break;
        uint band_top = y;
        while (y < OCR_HEIGHT && RowActive(y)) ++y;
        uint band_bottom = y;

        uint x = 0u;
        [loop]
        while (x < OCR_GRID_WIDTH) {
            while (x < OCR_GRID_WIDTH && !ColumnActive(x, band_top, band_bottom)) ++x;
            if (x >= OCR_GRID_WIDTH) break;
            uint run_start = x;
            uint last_active = x;
            uint gap = 0u;
            ++x;
            [loop]
            while (x < OCR_GRID_WIDTH) {
                if (ColumnActive(x, band_top, band_bottom)) {
                    last_active = x;
                    gap = 0u;
                } else {
                    ++gap;
                    if (gap > 12u) break;
                }
                ++x;
            }
            uint run_end = last_active + 1u;

            uint tight_left = 0xffffffffu;
            uint tight_top = 0xffffffffu;
            uint tight_right = 0u;
            uint tight_bottom = 0u;
            uint active_pixels = 0u;
            uint sum_q12 = 0u;
            [loop]
            for (uint row = band_top; row < band_bottom; ++row) {
                [loop]
                for (uint column = run_start; column < run_end; ++column) {
                    uint base = CellBase(column, row);
                    uint count = CellStats[base + 0u];
                    if (count == 0u) continue;
                    active_pixels += count;
                    sum_q12 += CellStats[base + 1u];
                    tight_left = min(tight_left, CellStats[base + 2u]);
                    tight_right = max(tight_right, CellStats[base + 3u]);
                    tight_top = min(tight_top, CellStats[base + 4u]);
                    tight_bottom = max(tight_bottom, CellStats[base + 5u]);
                }
            }
            if (active_pixels == 0u || tight_left == 0xffffffffu) continue;
            uint tight_width = tight_right - tight_left;
            uint tight_height = tight_bottom - tight_top;
            if (min(tight_width, tight_height) < 3u) continue;

            float score = (float)sum_q12 / ((float)active_pixels * 4095.0f);
            if (!FiniteFloat(score) || score < 0.4f) continue;
            uint4 tight_box = MapBox(tight_left, tight_top, tight_right, tight_bottom);
            if (tight_box.x >= tight_box.z || tight_box.y >= tight_box.w) continue;
            uint raw_index = raw_total++;
            if (raw_index < OCR6_RAW_CAPACITY) {
                StoreBox(OCR6_RAW_OFFSET, raw_index, tight_box, score);
            } else {
                status |= OCR6_OVERFLOW;
            }

            float perimeter = 2.0f * (float)(tight_width + tight_height);
            float expand = ceil((float)(tight_width * tight_height) * 1.4f / max(perimeter, 1.0f));
            uint pad = (uint)max(expand, 1.0f);
            uint expanded_left = tight_left > pad ? tight_left - pad : 0u;
            uint expanded_top = tight_top > pad ? tight_top - pad : 0u;
            uint expanded_right = min(tight_right + pad, OCR_WIDTH);
            uint expanded_bottom = min(tight_bottom + pad, OCR_HEIGHT);
            if (min(expanded_right - expanded_left, expanded_bottom - expanded_top) < 5u) continue;
            uint4 expanded_box = MapBox(
                expanded_left, expanded_top, expanded_right, expanded_bottom);
            if (expanded_box.x >= expanded_box.z || expanded_box.y >= expanded_box.w) continue;

            uint final_index = final_total++;
            if (final_index < OCR6_FINAL_CAPACITY) {
                StoreBox(OCR6_FINAL_OFFSET, final_index, expanded_box, score);
            } else {
                status |= OCR6_OVERFLOW;
            }
        }
    }

    const bool authoritative = status == OCR6_VALID;
    if (!authoritative) {
        [loop]
        for (uint word = 0u; word < OCR6_WORDS; ++word) OcrBoxRecord[word] = 0u;
    }
    OcrBoxRecord[0] = OCR6_SCHEMA;
    OcrBoxRecord[1] = OCR6_TAG;
    OcrBoxRecord[2] = authoritative ? OCR6_VALID : 0u;
    OcrBoxRecord[3] = authoritative ? raw_total : 0u;
    OcrBoxRecord[4] = authoritative ? final_total : 0u;
    OcrBoxRecord[5] = frame_lo;
    OcrBoxRecord[6] = frame_hi;
    OcrBoxRecord[7] = analysis_generation_lo;
    OcrBoxRecord[8] = analysis_generation_hi;
    OcrBoxRecord[9] = source_width;
    OcrBoxRecord[10] = source_height;
    OcrBoxRecord[11] = field_width;
    OcrBoxRecord[12] = field_height;
    OcrBoxRecord[13] = FIELD_ROI_TOP;
    OcrBoxRecord[14] = FIELD_ROI_BOTTOM;
    OcrBoxRecord[15] = 0u;
}

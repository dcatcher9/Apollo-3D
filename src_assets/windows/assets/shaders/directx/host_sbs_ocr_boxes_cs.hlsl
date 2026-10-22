#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"

#define OCR_WIDTH V2_OCR_OUTPUT_WIDTH
#define OCR_HEIGHT V2_OCR_OUTPUT_HEIGHT
#define OCR_CELL_WIDTH 8u
#define OCR_GRID_WIDTH (OCR_WIDTH / OCR_CELL_WIDTH)
#define OCR_CELLS_PER_GROUP 32u
#define OCR_CELL_ROWS_PER_GROUP 4u
#define OCR_CELL_WORDS 1u
#define OCR_CELL_ACTIVE_MASK 0xffu
#define OCR_CELL_NONFINITE_BIT 0x100u
#define OCR_CELL_SUM_SHIFT 9u
#define OCR_CELL_SUM_MASK 0x7fffu
#define OCR8_WORDS V2_OCR_RECORD_WORD_COUNT
#define OCR8_RAW_OFFSET V2_OCR_RAW_BOX_OFFSET
#define OCR8_RAW_CAPACITY V2_OCR_RAW_BOX_CAPACITY
#define OCR8_FINAL_CAPACITY V2_OCR_FINAL_BOX_CAPACITY
#define OCR8_BOX_WORDS V2_OCR_BOX_WORD_COUNT
#define OCR8_FINAL_OFFSET V2_OCR_FINAL_BOX_OFFSET
#define OCR8_SCHEMA V2_OCR_RECORD_SCHEMA
#define OCR8_TAG V2_OCR_RECORD_TAG
#define OCR8_VALID 1u
StructuredBuffer<float> ProbabilityMap : register(t0);
RWStructuredBuffer<uint> CellStatsWrite : register(u0);

bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

// One lane summarizes one 8x1 cell entirely in registers. Each of the four group rows is a full
// warp of 32 adjacent cells instead of one quarter-full warp per cell, removing the old shared
// atomics and two barriers. The packed word retains the exact active-pixel mask, any-nonfinite
// state, and Q12 probability sum; bounds and count are reconstructed losslessly.
[numthreads(OCR_CELLS_PER_GROUP, OCR_CELL_ROWS_PER_GROUP, 1)]
void cells_main(
    uint3 group_id : SV_GroupID,
    uint3 group_thread_id : SV_GroupThreadID)
{
    uint cell_x = group_id.x * OCR_CELLS_PER_GROUP + group_thread_id.x;
    uint cell_y = group_id.y * OCR_CELL_ROWS_PER_GROUP + group_thread_id.y;
    if (cell_x >= OCR_GRID_WIDTH || cell_y >= OCR_HEIGHT) return;

    uint active_mask = 0u;
    uint probability_sum_q12 = 0u;
    uint nonfinite = 0u;
    uint pixel_base = cell_y * OCR_WIDTH + cell_x * OCR_CELL_WIDTH;
    [unroll]
    for (uint pixel_in_cell = 0u; pixel_in_cell < OCR_CELL_WIDTH; ++pixel_in_cell) {
        float probability = ProbabilityMap[pixel_base + pixel_in_cell];
        if (!FiniteFloat(probability)) {
            nonfinite = OCR_CELL_NONFINITE_BIT;
        } else if (probability > V2_OCR_ACTIVE_PROBABILITY_THRESHOLD) {
            active_mask |= 1u << pixel_in_cell;
            probability_sum_q12 += (uint)round(saturate(probability) * 4095.0f);
        }
    }

    CellStatsWrite[cell_y * OCR_GRID_WIDTH + cell_x] =
        active_mask | nonfinite |
        ((probability_sum_q12 & OCR_CELL_SUM_MASK) << OCR_CELL_SUM_SHIFT);
}

// Keep resolve bindings distinct from cells_main. D3DCompiler validates the complete source file
// before entrypoint dead-code elimination and rejects two differently typed globals on one slot.
StructuredBuffer<uint> CellStats : register(t1);
RWStructuredBuffer<uint> OcrBoxRecord : register(u1);

#define OCR_RESOLVE_THREADS 256u
#define OCR_ACTIVE_MASK_BITS 32u
#define OCR_ACTIVE_MASK_WORDS_PER_ROW \
    ((OCR_GRID_WIDTH + OCR_ACTIVE_MASK_BITS - 1u) / OCR_ACTIVE_MASK_BITS)
#define OCR_ACTIVE_MASK_WORD_COUNT (OCR_HEIGHT * OCR_ACTIVE_MASK_WORDS_PER_ROW)

// resolve_main builds this compact occupancy index cooperatively before its deterministic
// topology walk. The old single-lane RowActive/ColumnActive helpers repeatedly fetched the same
// 19,200-cell UAV output through a strided SRV. Four row-local words preserve every cell decision
// exactly while moving those repeated queries to groupshared memory.
groupshared uint ResolveActiveMask[OCR_ACTIVE_MASK_WORD_COUNT];
groupshared uint ResolveNonfiniteByThread[OCR_RESOLVE_THREADS];

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
    uint roi_top;
    uint roi_bottom;
    uint4 tensor_content;
    int line_center_twice_source;
    uint line_reference_width;
    int line_search_top_source;
    uint line_reserved;
};

uint CellBase(uint x, uint y) {
    return (y * OCR_GRID_WIDTH + x) * OCR_CELL_WORDS;
}

uint CellActiveMask(uint packed) {
    return packed & OCR_CELL_ACTIVE_MASK;
}

uint CellProbabilitySumQ12(uint packed) {
    return (packed >> OCR_CELL_SUM_SHIFT) & OCR_CELL_SUM_MASK;
}

bool RowActive(uint y) {
    uint row_base = y * OCR_ACTIVE_MASK_WORDS_PER_ROW;
    [unroll]
    for (uint word = 0u; word < OCR_ACTIVE_MASK_WORDS_PER_ROW; ++word) {
        if (ResolveActiveMask[row_base + word] != 0u) return true;
    }
    return false;
}

uint4 BuildBandActiveMask(uint y0, uint y1) {
    uint4 active = uint4(0u, 0u, 0u, 0u);
    [loop]
    for (uint y = y0; y < y1; ++y) {
        uint row_base = y * OCR_ACTIVE_MASK_WORDS_PER_ROW;
        [unroll]
        for (uint word = 0u; word < OCR_ACTIVE_MASK_WORDS_PER_ROW; ++word) {
            active[word] |= ResolveActiveMask[row_base + word];
        }
    }
    return active;
}

bool BandColumnActive(uint x, uint4 active) {
    uint word = x / OCR_ACTIVE_MASK_BITS;
    uint bit = 1u << (x % OCR_ACTIVE_MASK_BITS);
    return (active[word] & bit) != 0u;
}

// Exact rational column mapping, matching MapYRational. FXC compiles a float division by the
// constant OCR_WIDTH as a multiply by its rounded-up float reciprocal, which can lift an exact
// integer quotient above itself and make the ceil one field column too wide. ResolveGeometryValid
// bounds x <= OCR_WIDTH and content_width <= field_width <= 0xffff, so x * content_width fits uint32.
uint MapXFloor(uint x) {
    uint content_width = tensor_content.z - tensor_content.x;
    return tensor_content.x + min(x * content_width / OCR_WIDTH, content_width);
}

uint MapXCeil(uint x) {
    uint content_width = tensor_content.z - tensor_content.x;
    return tensor_content.x + min(
        (x * content_width + OCR_WIDTH - 1u) / OCR_WIDTH, content_width);
}

uint MapYRational(uint y, bool round_up) {
    // ResolveGeometryValid proves both products fit uint32. Factor the division before scaling so
    // the exact rational mapping remains valid without Shader Model 6 uint64 support.
    uint denominator = OCR_HEIGHT * source_height;
    uint source_numerator = crop_top_pixels * OCR_HEIGHT + y * crop_height_pixels;
    uint whole = source_numerator / denominator;
    uint remainder = source_numerator % denominator;
    uint content_height = tensor_content.w - tensor_content.y;
    uint scaled_remainder = remainder * content_height;
    uint mapped = tensor_content.y + whole * content_height +
        scaled_remainder / denominator;
    if (round_up && scaled_remainder % denominator != 0u) ++mapped;
    return clamp(min(mapped, field_height), roi_top, roi_bottom);
}

uint MapYFloor(uint y) {
    return MapYRational(y, false);
}

uint MapYCeil(uint y) {
    return MapYRational(y, true);
}

// The host authenticates the DAV2 field shape before enabling OCR.  This shader still validates
// every runtime dimension and proves that its ROI is a non-empty subset of the exact bottom-6:1
// crop. OCR8 then carries those dimensions and bounds to the current locator, which cross-checks them against the
// actual BaseField dispatch. The 80-byte cbuffer also binds the integer tensor-content rectangle,
// avoiding a shape-specific OCR record or a second ABI.
bool ResolveGeometryValid() {
    if (line_reference_width == 0u ||
        line_reference_width > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT ||
        line_center_twice_source < -(int)line_reference_width ||
        line_center_twice_source > (int)line_reference_width ||
        line_search_top_source < -(int)V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT ||
        line_search_top_source > (int)V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT ||
        line_reserved != 0u ||
        source_width > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT ||
        source_height > V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT ||
        source_width == 0u || source_height == 0u || field_width == 0u || field_height == 0u ||
        field_width > 0xffffu || field_height > 0xffffu ||
        roi_top >= roi_bottom || roi_bottom > field_height || crop_height_pixels == 0u ||
        crop_top_pixels > source_height ||
        crop_height_pixels > source_height - crop_top_pixels ||
        tensor_content.x >= tensor_content.z || tensor_content.y >= tensor_content.w ||
        tensor_content.z > field_width || tensor_content.w > field_height ||
        roi_top < tensor_content.y || roi_bottom > tensor_content.w) {
        return false;
    }

    if (!V2SubtitleOcrFieldIsCalibrated(field_width, field_height)) return false;

    uint expected_crop_top;
    uint expected_crop_height;
    if (!V2SubtitleOcrComputeCrop(
            source_width, source_height, expected_crop_top, expected_crop_height)) {
        return false;
    }
    if (crop_height_pixels != expected_crop_height ||
        crop_top_pixels != expected_crop_top) {
        return false;
    }
    uint expected_roi_top;
    uint expected_roi_bottom;
    if (!V2SubtitleOcrProjectContentRowCeil(
            source_width, source_height, field_width, field_height, tensor_content,
            V2_OCR_SAFE_ROW_TOP, expected_roi_top) ||
        !V2SubtitleOcrProjectContentRowCeil(
            source_width, source_height, field_width, field_height, tensor_content,
            V2_OCR_SAFE_ROW_BOTTOM, expected_roi_bottom)) {
        return false;
    }
    return roi_top == expected_roi_top && roi_bottom == expected_roi_bottom;
}

void PublishHeader(bool authoritative, uint raw_total, uint final_total) {
    OcrBoxRecord[0] = OCR8_SCHEMA;
    OcrBoxRecord[1] = OCR8_TAG;
    OcrBoxRecord[2] = authoritative ? OCR8_VALID : 0u;
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
    OcrBoxRecord[13] = roi_top;
    OcrBoxRecord[14] = roi_bottom;
    OcrBoxRecord[15] = 0u;
}

void ClearAndPublishInvalidRecord() {
    [loop]
    for (uint word = 0u; word < OCR8_WORDS; ++word) OcrBoxRecord[word] = 0u;
    PublishHeader(false, 0u, 0u);
}

void StoreBox(
    uint base,
    uint index,
    uint4 box,
    float score,
    uint kind_flags,
    uint island_count,
    uint structural_gap_count) {
    uint offset = base + index * OCR8_BOX_WORDS;
    OcrBoxRecord[offset + 0u] = box.x;
    OcrBoxRecord[offset + 1u] = box.y;
    OcrBoxRecord[offset + 2u] = box.z;
    OcrBoxRecord[offset + 3u] = box.w;
    OcrBoxRecord[offset + 4u] = asuint(score);
    OcrBoxRecord[offset + 5u] = kind_flags;
    OcrBoxRecord[offset + 6u] = island_count;
    OcrBoxRecord[offset + 7u] = structural_gap_count;
}

uint4 MapBox(uint left, uint top, uint right, uint bottom) {
    return uint4(MapXFloor(left), MapYFloor(top), MapXCeil(right), MapYCeil(bottom));
}

struct CandidateStats {
    uint tight_left;
    uint tight_top;
    uint tight_right;
    uint tight_bottom;
    uint active_pixels;
    uint sum_q12;
    uint island_count;
    uint structural_gap_count;
};

// Lane zero queues the independent cores before selection. A line union is eligibility only;
// original constituent covers and the downstream pixel/Gaussian mask remain unchanged.
groupshared CandidateStats LineCandidates[OCR_GRID_WIDTH];
groupshared float LineScores[OCR_GRID_WIDTH];
groupshared uint LineKinds[OCR_GRID_WIDTH];
groupshared uint LineMembership[OCR_GRID_WIDTH];
groupshared uint4 LineBounds[OCR_GRID_WIDTH];
groupshared uint2 LineAnchorSpans[OCR_GRID_WIDTH];

void ResetCandidateStats(out CandidateStats stats) {
    stats.tight_left = 0xffffffffu;
    stats.tight_top = 0xffffffffu;
    stats.tight_right = 0u;
    stats.tight_bottom = 0u;
    stats.active_pixels = 0u;
    stats.sum_q12 = 0u;
    stats.island_count = 0u;
    stats.structural_gap_count = 0u;
}

void GatherCandidateStats(
    uint band_top,
    uint band_bottom,
    uint column_start,
    uint column_end,
    out CandidateStats stats) {
    ResetCandidateStats(stats);
    [loop]
    for (uint row = band_top; row < band_bottom; ++row) {
        [loop]
        for (uint word = column_start / OCR_ACTIVE_MASK_BITS;
             word <= (column_end - 1u) / OCR_ACTIVE_MASK_BITS;
             ++word) {
            uint word_column = word * OCR_ACTIVE_MASK_BITS;
            uint first_column = max(column_start, word_column);
            uint end_column = min(column_end, word_column + OCR_ACTIVE_MASK_BITS);
            uint active = ResolveActiveMask[row * OCR_ACTIVE_MASK_WORDS_PER_ROW + word];
            uint first_bit = first_column - word_column;
            uint end_bit = end_column - word_column;
            if (first_bit != 0u) active &= 0xffffffffu << first_bit;
            if (end_bit != OCR_ACTIVE_MASK_BITS) active &= (1u << end_bit) - 1u;

            [loop]
            while (active != 0u) {
                uint active_bit = (uint)firstbitlow(active);
                uint column = word_column + active_bit;
                uint packed = CellStats[CellBase(column, row)];
                uint mask = CellActiveMask(packed);
                uint count = countbits(mask);
                uint first = (uint)firstbitlow(mask);
                uint last = (uint)firstbithigh(mask);
                uint cell_left = column * OCR_CELL_WIDTH;
                stats.active_pixels += count;
                stats.sum_q12 += CellProbabilitySumQ12(packed);
                stats.tight_left = min(stats.tight_left, cell_left + first);
                stats.tight_right = max(stats.tight_right, cell_left + last + 1u);
                stats.tight_top = min(stats.tight_top, row);
                stats.tight_bottom = max(stats.tight_bottom, row + 1u);
                active &= active - 1u;
            }
        }
    }
}

bool CandidateEvidenceValid(CandidateStats stats, out float score) {
    score = 0.0f;
    if (stats.active_pixels == 0u || stats.tight_left == 0xffffffffu ||
        stats.tight_right <= stats.tight_left || stats.tight_bottom <= stats.tight_top ||
        min(stats.tight_right - stats.tight_left, stats.tight_bottom - stats.tight_top) < 3u) {
        return false;
    }
    score = (float)stats.sum_q12 / ((float)stats.active_pixels * 4095.0f);
    return FiniteFloat(score) && score >= V2_OCR_MIN_MEAN_SCORE;
}

void MergeRetainedIsland(
    CandidateStats island,
    uint island_start,
    uint island_end,
    inout CandidateStats candidate,
    inout uint retained_count,
    inout uint previous_island_end) {
    if (retained_count == 0u) {
        candidate.tight_left = island.tight_left;
        candidate.tight_top = island.tight_top;
        candidate.tight_right = island.tight_right;
        candidate.tight_bottom = island.tight_bottom;
    } else {
        candidate.tight_left = min(candidate.tight_left, island.tight_left);
        candidate.tight_top = min(candidate.tight_top, island.tight_top);
        candidate.tight_right = max(candidate.tight_right, island.tight_right);
        candidate.tight_bottom = max(candidate.tight_bottom, island.tight_bottom);
        uint retained_gap = island_start - previous_island_end;
        if (retained_gap >= V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS) {
            ++candidate.structural_gap_count;
        }
    }
    candidate.active_pixels += island.active_pixels;
    candidate.sum_q12 += island.sum_q12;
    ++retained_count;
    candidate.island_count = retained_count;
    previous_island_end = island_end;
}

bool PrepareCandidateBoxes(
    CandidateStats stats, uint kind_flags, out uint4 tight_box, out uint4 expanded_box) {
    uint tight_width = stats.tight_right - stats.tight_left;
    uint tight_height = stats.tight_bottom - stats.tight_top;
    tight_box = MapBox(
        stats.tight_left, stats.tight_top, stats.tight_right, stats.tight_bottom);
    if (tight_box.x >= tight_box.z || tight_box.y >= tight_box.w) return false;

    float perimeter = 2.0f * (float)(tight_width + tight_height);
    float expand = ceil((float)(tight_width * tight_height) * 1.4f / max(perimeter, 1.0f));
    // DB's area/perimeter unclip becomes height-sized after many text islands are merged into one
    // long line. Bound only an identified ribbon in the fixed detector domain so it cannot absorb
    // unrelated content above it. Ordinary subtitle expansion retains the unbounded DB pad.
    uint db_pad = (uint)max(expand, 1.0f);
    uint pad = kind_flags == V2_OCR_BOX_FLAG_RIBBON ?
        min(db_pad, V2_OCR_RIBBON_COVER_PAD_LIMIT) : db_pad;
    // A detector can omit a short token at either end of an otherwise valid text line. Keep a
    // one-cell horizontal search guard; its pixels still need the ordinary RGB confirmations.
    // Vertical DB padding and the ribbon's bounded full-width cover keep their existing rules.
    uint horizontal_pad = pad + (kind_flags == V2_OCR_BOX_FLAG_RIBBON ? 0u :
        V2_OCR_ORDINARY_COVER_HORIZONTAL_GUARD_CELLS * OCR_CELL_WIDTH);
    uint expanded_left = stats.tight_left > horizontal_pad ?
        stats.tight_left - horizontal_pad : 0u;
    uint expanded_top = stats.tight_top > pad ? stats.tight_top - pad : 0u;
    uint expanded_right = min(stats.tight_right + horizontal_pad, OCR_WIDTH);
    uint expanded_bottom = min(stats.tight_bottom + pad, OCR_HEIGHT);
    if (min(expanded_right - expanded_left, expanded_bottom - expanded_top) < 5u) return false;

    // A ribbon conditions the complete bottom strip. Ordinary candidates preserve their own
    // expanded cover, including gaps only among independently valid retained islands.
    expanded_box = kind_flags == V2_OCR_BOX_FLAG_RIBBON ?
        uint4(tensor_content.x, MapYFloor(expanded_top),
              tensor_content.z, tensor_content.w) :
        MapBox(expanded_left, expanded_top, expanded_right, expanded_bottom);
    if (expanded_box.x >= expanded_box.z || expanded_box.y >= expanded_box.w) return false;

    return true;
}

void PublishCandidate(
    CandidateStats stats, float score, uint kind_flags,
    inout uint raw_total, inout uint final_total) {
    uint4 tight_box, expanded_box;
    if (!PrepareCandidateBoxes(stats, kind_flags, tight_box, expanded_box)) return;
    uint pair_capacity = min(OCR8_RAW_CAPACITY, OCR8_FINAL_CAPACITY);
    uint pair_index = final_total;
    if (final_total < pair_capacity) {
        ++raw_total;
        ++final_total;
    } else {
        // Keep the largest tight cores without giving crowded text a global veto. Equal areas
        // keep their earlier raster discovery. Evict the last equal minimum, then shift whole
        // core/cover pairs so publication order remains the original discovery order.
        uint smallest_area = 0xffffffffu;
        uint remove_index = 0u;
        [loop]
        for (uint slot = 0u; slot < pair_capacity; ++slot) {
            uint offset = OCR8_RAW_OFFSET + slot * OCR8_BOX_WORDS;
            uint area = (OcrBoxRecord[offset + 2u] - OcrBoxRecord[offset + 0u]) *
                (OcrBoxRecord[offset + 3u] - OcrBoxRecord[offset + 1u]);
            if (area <= smallest_area) {
                smallest_area = area;
                remove_index = slot;
            }
        }
        uint candidate_area = (tight_box.z - tight_box.x) * (tight_box.w - tight_box.y);
        if (candidate_area <= smallest_area) return;
        [loop]
        for (uint shift = remove_index; shift + 1u < pair_capacity; ++shift) {
            [unroll]
            for (uint word = 0u; word < OCR8_BOX_WORDS; ++word) {
                OcrBoxRecord[OCR8_RAW_OFFSET + shift * OCR8_BOX_WORDS + word] =
                    OcrBoxRecord[OCR8_RAW_OFFSET + (shift + 1u) * OCR8_BOX_WORDS + word];
                OcrBoxRecord[OCR8_FINAL_OFFSET + shift * OCR8_BOX_WORDS + word] =
                    OcrBoxRecord[OCR8_FINAL_OFFSET + (shift + 1u) * OCR8_BOX_WORDS + word];
            }
        }
        pair_index = pair_capacity - 1u;
    }
    StoreBox(
        OCR8_RAW_OFFSET, pair_index, tight_box, score,
        kind_flags, stats.island_count, stats.structural_gap_count);
    StoreBox(
        OCR8_FINAL_OFFSET, pair_index, expanded_box, score,
        kind_flags, stats.island_count, stats.structural_gap_count);
}

bool QueueLineCandidate(
    CandidateStats stats, float score, uint kind_flags, inout uint candidate_count) {
    uint4 tight_box, expanded_box;
    if (!PrepareCandidateBoxes(stats, kind_flags, tight_box, expanded_box)) return true;
    if (candidate_count >= OCR_GRID_WIDTH) return false;
    LineCandidates[candidate_count] = stats;
    LineScores[candidate_count] = score;
    LineKinds[candidate_count] = kind_flags;
    ++candidate_count;
    return true;
}

bool SameTextLine(CandidateStats candidate, uint2 anchor, uint4 bounds) {
    uint intersection_top = max(candidate.tight_top, anchor.x);
    uint intersection_bottom = min(candidate.tight_bottom, anchor.y);
    if (intersection_top >= intersection_bottom) return false;
    uint union_height = max(candidate.tight_bottom, anchor.y) -
        min(candidate.tight_top, anchor.x);
    if ((intersection_bottom - intersection_top) *
        V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR <
        union_height * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR) return false;
    // Compare native-source lengths, including rounded or height-clipped OCR crops. A row
    // at the same height in a distant column cannot lend this line a false centered union.
    uint gap_begin = max(candidate.tight_left, bounds.x);
    uint gap_end = min(candidate.tight_right, bounds.z);
    uint gap = gap_begin > gap_end ? gap_begin - gap_end : 0u;
    uint height = max(candidate.tight_bottom - candidate.tight_top, anchor.y - anchor.x);
    // Validated source extents, exact crop and the fixed 4/1 policy bound both uint32 products.
    return gap * source_width * OCR_HEIGHT *
        V2_OCR_LINE_GROUP_MAX_HORIZONTAL_GAP_HEIGHT_DENOMINATOR <=
        height * crop_height_pixels * OCR_WIDTH *
        V2_OCR_LINE_GROUP_MAX_HORIZONTAL_GAP_HEIGHT_NUMERATOR;
}

bool LineIsCentered(uint4 bounds) {
    // Native extents/reference are bounded to 5120. Even the largest signed difference times
    // the tolerance denominator is <=147,456,000: exact int32 comparison keeps both edges equal.
    int delta = (int)(bounds.x + bounds.z) * (int)source_width -
        line_center_twice_source * (int)OCR_WIDTH;
    uint error = (uint)abs(delta);
    return error * V2_OCR_LINE_CENTER_TOLERANCE_DENOMINATOR <=
        2u * line_reference_width * OCR_WIDTH * V2_OCR_LINE_CENTER_TOLERANCE_NUMERATOR;
}

void PublishCenteredLines(uint candidate_count, inout uint raw_total, inout uint final_total) {
    uint group_count = 0u;
    [loop]
    for (uint candidate = 0u; candidate < candidate_count; ++candidate) {
        CandidateStats stats = LineCandidates[candidate];
        uint group = 0u;
        [loop]
        while (group < group_count &&
            !SameTextLine(stats, LineAnchorSpans[group], LineBounds[group])) ++group;
        uint4 bounds = uint4(stats.tight_left, stats.tight_top, stats.tight_right, stats.tight_bottom);
        if (group == group_count) {
            // Keep the first member's vertical span fixed: growing anchors can bridge other rows.
            LineAnchorSpans[group] = uint2(stats.tight_top, stats.tight_bottom);
            LineBounds[group] = bounds;
            ++group_count;
        } else {
            // Horizontal adjacency follows the current union so nearby words can form a long
            // line. The separate first-member vertical anchor above never grows.
            LineBounds[group] = uint4(min(LineBounds[group].xy, bounds.xy),
                                     max(LineBounds[group].zw, bounds.zw));
        }
        LineMembership[candidate] = group;
    }
    [loop]
    for (uint candidate = 0u; candidate < candidate_count; ++candidate) {
        if (LineIsCentered(LineBounds[LineMembership[candidate]])) {
            PublishCandidate(LineCandidates[candidate], LineScores[candidate], LineKinds[candidate],
                             raw_total, final_total);
        }
    }
}

uint LineSearchBegin() {
    // Intersect the matched reference's lower quarter with the existing OCR crop and safe rows.
    // Validation bounds the signed subtraction and this exact ceiling's numerator below uint32.
    int delta = line_search_top_source - (int)crop_top_pixels;
    uint row = delta <= 0 ? 0u : min(OCR_HEIGHT,
        ((uint)delta * OCR_HEIGHT + crop_height_pixels - 1u) / crop_height_pixels);
    return max(row, V2_OCR_SAFE_ROW_TOP);
}

// Deterministic lower-text DB postprocess. All lanes first build a compact, exact cell-occupancy
// index and clear the fixed OCR8 record. Lane zero then preserves the existing serial topology and
// publication order, but its repeated row/column tests no longer rescan global cell records.
// A broad run retains the larger index gap only long enough to classify a bottom ribbon.
// Non-ribbon text is rescanned with its smaller join gap, and every retained island must carry
// valid text evidence on its own so a distant weak pixel cannot borrow a subtitle's confidence or
// enlarge its conditioning cover.
[numthreads(OCR_RESOLVE_THREADS, 1, 1)]
void resolve_main(uint3 group_thread_id : SV_GroupThreadID) {
    uint thread_index = group_thread_id.x;
    if (thread_index < OCR8_WORDS) {
        OcrBoxRecord[thread_index] = 0u;
    }

    uint local_nonfinite = 0u;
    [loop]
    for (uint mask_word = thread_index;
         mask_word < OCR_ACTIVE_MASK_WORD_COUNT;
         mask_word += OCR_RESOLVE_THREADS) {
        uint row = mask_word / OCR_ACTIVE_MASK_WORDS_PER_ROW;
        uint word_in_row = mask_word % OCR_ACTIVE_MASK_WORDS_PER_ROW;
        uint first_x = word_in_row * OCR_ACTIVE_MASK_BITS;
        uint active_mask = 0u;
        [unroll]
        for (uint bit_index = 0u; bit_index < OCR_ACTIVE_MASK_BITS; ++bit_index) {
            uint x = first_x + bit_index;
            if (x < OCR_GRID_WIDTH) {
                uint packed = CellStats[CellBase(x, row)];
                if (CellActiveMask(packed) != 0u) {
                    active_mask |= 1u << bit_index;
                }
                local_nonfinite |= (packed & OCR_CELL_NONFINITE_BIT) != 0u ? 1u : 0u;
            }
        }
        ResolveActiveMask[mask_word] = active_mask;
    }
    ResolveNonfiniteByThread[thread_index] = local_nonfinite;

    // This orders both the groupshared index and the cooperative OCR8 UAV clear before lane zero
    // begins overwriting authoritative header/box words. Every lane reaches this sole barrier.
    AllMemoryBarrierWithGroupSync();
    if (thread_index != 0u) {
        return;
    }

    if (!ResolveGeometryValid()) {
        PublishHeader(false, 0u, 0u);
        return;
    }

    [loop]
    for (uint thread = 0u; thread < OCR_RESOLVE_THREADS; ++thread) {
        if (ResolveNonfiniteByThread[thread] != 0u) {
            ClearAndPublishInvalidRecord();
            return;
        }
    }

    uint raw_total = 0u;
    uint final_total = 0u;
    uint y = LineSearchBegin();
    uint search_end = V2_OCR_SAFE_ROW_BOTTOM;
    [loop]
    while (y < search_end) {
        while (y < search_end && !RowActive(y)) ++y;
        if (y >= search_end) break;
        uint band_top = y;
        while (y < search_end && RowActive(y)) ++y;
        uint band_bottom = y;
        uint4 band_active = BuildBandActiveMask(band_top, band_bottom);
        uint band_candidate_count = 0u;

        uint x = 0u;
        [loop]
        while (x < OCR_GRID_WIDTH) {
            while (x < OCR_GRID_WIDTH && !BandColumnActive(x, band_active)) ++x;
            if (x >= OCR_GRID_WIDTH) break;
            uint broad_start = x;
            uint broad_last_active = x;
            uint broad_gap = 0u;
            uint broad_island_count = 1u;
            uint broad_structural_gap_count = 0u;
            ++x;
            [loop]
            while (x < OCR_GRID_WIDTH) {
                if (BandColumnActive(x, band_active)) {
                    if (broad_gap != 0u) {
                        ++broad_island_count;
                        if (broad_gap >= V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS) {
                            ++broad_structural_gap_count;
                        }
                    }
                    broad_last_active = x;
                    broad_gap = 0u;
                } else {
                    ++broad_gap;
                    if (broad_gap > V2_OCR_RIBBON_JOIN_GAP_CELLS) break;
                }
                ++x;
            }
            uint broad_end = broad_last_active + 1u;

            // Preserve every broad-run island for the ribbon conjunction. A micro island is allowed
            // to describe index topology, but never ordinary subtitle geometry unless it passes the
            // independent evidence gate below.
            CandidateStats broad;
            GatherCandidateStats(band_top, band_bottom, broad_start, broad_end, broad);
            broad.island_count = broad_island_count;
            broad.structural_gap_count = broad_structural_gap_count;
            float broad_score;
            bool broad_is_ribbon = false;
            if (CandidateEvidenceValid(broad, broad_score)) {
                uint broad_width = broad.tight_right - broad.tight_left;
                bool bottom_attached =
                    broad.tight_bottom + V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS >=
                    V2_OCR_SAFE_ROW_BOTTOM;
                bool wide = broad_width * V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR >=
                    OCR_WIDTH * V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR;
                broad_is_ribbon = bottom_attached && wide &&
                    broad.structural_gap_count >= V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS;
            }
            if (broad_is_ribbon) {
                if (!QueueLineCandidate(broad, broad_score, V2_OCR_BOX_FLAG_RIBBON, band_candidate_count)) {
                    ClearAndPublishInvalidRecord();
                    return;
                }
                continue;
            }

            // A non-ribbon broad run may contain several ordinary lines separated by an index-sized
            // gap. Re-form compact text runs, then retain only islands that independently meet the
            // same minimum geometry and confidence evidence as a complete candidate.
            uint text_x = broad_start;
            [loop]
            while (text_x < broad_end) {
                while (text_x < broad_end &&
                       !BandColumnActive(text_x, band_active)) ++text_x;
                if (text_x >= broad_end) break;
                uint text_start = text_x;
                uint text_last_active = text_x;
                uint text_gap = 0u;
                ++text_x;
                [loop]
                while (text_x < broad_end) {
                    if (BandColumnActive(text_x, band_active)) {
                        text_last_active = text_x;
                        text_gap = 0u;
                    } else {
                        ++text_gap;
                        if (text_gap > V2_OCR_TEXT_JOIN_GAP_CELLS) break;
                    }
                    ++text_x;
                }
                uint text_end = text_last_active + 1u;

                CandidateStats retained;
                ResetCandidateStats(retained);
                uint retained_count = 0u;
                uint previous_island_end = 0u;
                uint island_x = text_start;
                [loop]
                while (island_x < text_end) {
                    while (island_x < text_end &&
                           !BandColumnActive(island_x, band_active)) ++island_x;
                    if (island_x >= text_end) break;
                    uint island_start = island_x;
                    while (island_x < text_end &&
                           BandColumnActive(island_x, band_active)) ++island_x;
                    uint island_end = island_x;

                    CandidateStats island;
                    GatherCandidateStats(
                        band_top, band_bottom, island_start, island_end, island);
                    float island_score;
                    if (CandidateEvidenceValid(island, island_score)) {
                        // Invalid micro-islands must not reset the effective join distance. If the
                        // next independently valid island is farther than the ordinary limit from
                        // the previous retained island, finish the current candidate before
                        // starting another. This closes A-gap-speck-gap-B bridge chains.
                        if (retained_count != 0u &&
                            island_start - previous_island_end > V2_OCR_TEXT_JOIN_GAP_CELLS) {
                            float retained_score;
                            if (CandidateEvidenceValid(retained, retained_score)) {
                                if (!QueueLineCandidate(retained, retained_score, 0u, band_candidate_count)) {
                                    ClearAndPublishInvalidRecord();
                                    return;
                                }
                            }
                            ResetCandidateStats(retained);
                            retained_count = 0u;
                            previous_island_end = 0u;
                        }
                        MergeRetainedIsland(
                            island, island_start, island_end, retained,
                            retained_count, previous_island_end);
                    }
                }

                float text_score;
                if (CandidateEvidenceValid(retained, text_score)) {
                    if (!QueueLineCandidate(retained, text_score, 0u, band_candidate_count)) {
                        ClearAndPublishInvalidRecord();
                        return;
                    }
                }
            }
        }
        PublishCenteredLines(band_candidate_count, raw_total, final_total);
    }

    PublishHeader(true, raw_total, final_total);
}

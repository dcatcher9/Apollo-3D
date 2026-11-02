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
// Fixed GPU-only scratch. Six uint4 records per candidate preserve readable full-width stats;
// the local-row bound below avoids a shape-dependent allocation or a CPU readback.
RWStructuredBuffer<uint4> OcrCandidateScratch : register(u2);

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
groupshared uint ClassifierHistory[V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT];
groupshared uint ClassifierClockValid;
groupshared uint ClassifierCanLearn;
groupshared uint TrustedCurrentBins;
groupshared uint TrustedCurrentPixels;
groupshared uint2 TrustedCurrentSpan;
groupshared uint4 RetainedTrusted[OCR8_FINAL_CAPACITY];

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
    uint2 source_clock_us;
    uint input_domain_reset;
    uint classifier_reserved;
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

bool RestrictedRowActive(uint y, uint column_start, uint column_end) {
    [loop]
    for (uint word = column_start / OCR_ACTIVE_MASK_BITS;
         word <= (column_end - 1u) / OCR_ACTIVE_MASK_BITS; ++word) {
        uint first = max(column_start, word * OCR_ACTIVE_MASK_BITS) -
            word * OCR_ACTIVE_MASK_BITS;
        uint end = min(column_end, (word + 1u) * OCR_ACTIVE_MASK_BITS) -
            word * OCR_ACTIVE_MASK_BITS;
        uint active = ResolveActiveMask[y * OCR_ACTIVE_MASK_WORDS_PER_ROW + word];
        if (first != 0u) active &= 0xffffffffu << first;
        if (end != OCR_ACTIVE_MASK_BITS) active &= (1u << end) - 1u;
        if (active != 0u) return true;
    }
    return false;
}

// Exact rational column mapping, matching MapYRational. FXC compiles a float division by the
// constant OCR_WIDTH as a multiply by its rounded-up float reciprocal, which can lift an exact
// integer quotient above itself and make the ceil one field column too wide. x is clamped to
// OCR_WIDTH here and ResolveGeometryValid bounds content_width <= field_width <= 0xffff, so
// x * content_width fits uint32 and the mapped column never passes the content's right edge.
uint MapXFloor(uint x) {
    uint content_width = tensor_content.z - tensor_content.x;
    return tensor_content.x + min(x, OCR_WIDTH) * content_width / OCR_WIDTH;
}

uint MapXCeil(uint x) {
    uint content_width = tensor_content.z - tensor_content.x;
    return tensor_content.x + (min(x, OCR_WIDTH) * content_width + OCR_WIDTH - 1u) / OCR_WIDTH;
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
        line_reserved != 0u || classifier_reserved != 0u || input_domain_reset > 1u ||
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

static const uint CANDIDATE_ENABLED = 1u;
static const uint CANDIDATE_PRIOR_GEOMETRY_ALLOWED = 2u;
static const uint CLASS_UNKNOWN = 0u;
static const uint CLASS_HORIZONTAL = 1u;
static const uint CLASS_AMBIGUOUS = 2u;
static const uint CLASS_SCENE = 3u;
static const uint CLASS_STACK_ASSISTED = 4u;

uint ScratchBase(uint candidate) {
    return candidate * (V2_OCR_LOCAL_CANDIDATE_WORD_COUNT / 4u);
}

CandidateStats LoadCandidate(uint candidate) {
    uint base = ScratchBase(candidate);
    uint4 bounds = OcrCandidateScratch[base];
    uint4 evidence = OcrCandidateScratch[base + 1u];
    CandidateStats stats;
    stats.tight_left = bounds.x;
    stats.tight_top = bounds.y;
    stats.tight_right = bounds.z;
    stats.tight_bottom = bounds.w;
    stats.active_pixels = evidence.x;
    stats.sum_q12 = evidence.y;
    stats.island_count = evidence.z;
    stats.structural_gap_count = evidence.w;
    return stats;
}

void StoreCandidateStats(uint candidate, CandidateStats stats) {
    uint base = ScratchBase(candidate);
    OcrCandidateScratch[base] = uint4(
        stats.tight_left, stats.tight_top, stats.tight_right, stats.tight_bottom);
    OcrCandidateScratch[base + 1u] = uint4(
        stats.active_pixels, stats.sum_q12, stats.island_count, stats.structural_gap_count);
}

bool CandidateEnabled(uint candidate) {
    return (OcrCandidateScratch[ScratchBase(candidate) + 3u].x & CANDIDATE_ENABLED) != 0u;
}

void DisableCandidate(uint candidate) {
    uint base = ScratchBase(candidate) + 3u;
    uint4 meta = OcrCandidateScratch[base];
    meta.x &= ~CANDIDATE_ENABLED;
    OcrCandidateScratch[base] = meta;
}

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
    // The detector can miss a line-end token or the lower strokes darkened by player controls.
    // Ordinary search adds the existing four-cell word-gap reach horizontally and half the
    // current tight height vertically. These bounds only authorize current native pixel search;
    // they do not protect the whole cover or change tight cores, classification, or prior votes.
    // Ribbons retain their independently bounded full-width bottom-strip policy.
    uint horizontal_pad = pad + (kind_flags == V2_OCR_BOX_FLAG_RIBBON ? 0u :
        V2_OCR_ORDINARY_COVER_HORIZONTAL_GUARD_CELLS * OCR_CELL_WIDTH);
    uint vertical_guard = (tight_height * V2_OCR_ORDINARY_COVER_VERTICAL_GUARD_HEIGHT_NUMERATOR +
        V2_OCR_ORDINARY_COVER_VERTICAL_GUARD_HEIGHT_DENOMINATOR - 1u) /
        V2_OCR_ORDINARY_COVER_VERTICAL_GUARD_HEIGHT_DENOMINATOR;
    uint vertical_pad = pad + (kind_flags == V2_OCR_BOX_FLAG_RIBBON ? 0u : vertical_guard);
    uint expanded_left = stats.tight_left > horizontal_pad ?
        stats.tight_left - horizontal_pad : 0u;
    uint expanded_top = stats.tight_top > vertical_pad ? stats.tight_top - vertical_pad : 0u;
    uint expanded_right = min(stats.tight_right + horizontal_pad, OCR_WIDTH);
    uint expanded_bottom = min(stats.tight_bottom + vertical_pad, OCR_HEIGHT);
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
    CandidateStats stats, float score, uint kind_flags, bool independently_strong,
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
            RetainedTrusted[shift] = RetainedTrusted[shift + 1u];
        }
        pair_index = pair_capacity - 1u;
    }
    StoreBox(
        OCR8_RAW_OFFSET, pair_index, tight_box, score,
        kind_flags, stats.island_count, stats.structural_gap_count);
    StoreBox(
        OCR8_FINAL_OFFSET, pair_index, expanded_box, score,
        kind_flags, stats.island_count, stats.structural_gap_count);
    uint bin = min(V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT - 1u,
        (stats.tight_bottom - 1u) * V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT / OCR_HEIGHT);
    RetainedTrusted[pair_index] = independently_strong ?
        uint4(1u << bin, stats.active_pixels, stats.tight_top, stats.tight_bottom) :
        uint4(0u, 0u, 0u, 0u);
}

bool QueueLineCandidate(
    CandidateStats stats, float score, uint kind_flags, inout uint candidate_count) {
    // A band of height h has <=60 column islands and <=floor((h+1)/4) valid local row spans
    // per island (height>=3 and at least one blank row between). For B disjoint global bands,
    // sum(h)+(B-1)<=131, so sum(floor((h+1)/4))<=33. Thus 60*33 bounds the WHOLE observation,
    // not just one band. A ribbon replaces its broad run rather than adding ordinary pieces.
    // This guard diagnoses an internal contract violation, not eight-pair publication overflow.
    if (candidate_count >= V2_OCR_LOCAL_CANDIDATE_CAPACITY) return false;
    uint base = ScratchBase(candidate_count);
    StoreCandidateStats(candidate_count, stats);
    uint column_start = stats.tight_left / OCR_CELL_WIDTH;
    uint column_end = (stats.tight_right + OCR_CELL_WIDTH - 1u) / OCR_CELL_WIDTH;
    OcrCandidateScratch[base + 2u] = uint4(
        stats.tight_top, stats.tight_bottom, column_end, column_start);
    OcrCandidateScratch[base + 3u] = uint4(
        CANDIDATE_ENABLED, kind_flags, 0u, column_start * OCR_HEIGHT + stats.tight_top);
    OcrCandidateScratch[base + 4u] = uint4(0u, 0u, 0u, 0u);
    OcrCandidateScratch[base + 5u] = uint4(0u, 0u, CLASS_UNKNOWN, 0u);
    ++candidate_count;
    return true;
}

void SortCandidates(uint candidate_begin, uint candidate_end) {
    // A local upper line can begin much farther right than another piece in its parent column
    // island. Sort by the actual occupied columns before joins; parent projection edges must
    // never make that distant line borrow the nearer piece's horizontal gap.
    [loop]
    for (uint candidate = candidate_begin + 1u; candidate < candidate_end; ++candidate) {
        uint4 saved[6];
        [unroll]
        for (uint word = 0u; word < 6u; ++word) {
            saved[word] = OcrCandidateScratch[ScratchBase(candidate) + word];
        }
        uint position = candidate;
        [loop]
        while (position > candidate_begin) {
            // HLSL logical operators do not promise short-circuit evaluation. Test the bound
            // separately so the insertion cannot read before this band's range or underflow.
            if (OcrCandidateScratch[ScratchBase(position - 1u) + 3u].w <= saved[3].w) break;
            [unroll]
            for (uint word = 0u; word < 6u; ++word) {
                OcrCandidateScratch[ScratchBase(position) + word] =
                    OcrCandidateScratch[ScratchBase(position - 1u) + word];
            }
            --position;
        }
        [unroll]
        for (uint word = 0u; word < 6u; ++word) {
            OcrCandidateScratch[ScratchBase(position) + word] = saved[word];
        }
    }
}

bool SpanHasHalfIoU(uint top, uint bottom, uint2 anchor) {
    uint intersection_top = max(top, anchor.x);
    uint intersection_bottom = min(bottom, anchor.y);
    if (intersection_top >= intersection_bottom) return false;
    uint union_height = max(bottom, anchor.y) - min(top, anchor.x);
    return (intersection_bottom - intersection_top) *
        V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR >=
        union_height * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR;
}

void MergeScratchCandidate(uint receiver, uint member, uint retained_gap) {
    CandidateStats accumulated = LoadCandidate(receiver);
    CandidateStats piece = LoadCandidate(member);
    accumulated.tight_left = min(accumulated.tight_left, piece.tight_left);
    accumulated.tight_top = min(accumulated.tight_top, piece.tight_top);
    accumulated.tight_right = max(accumulated.tight_right, piece.tight_right);
    accumulated.tight_bottom = max(accumulated.tight_bottom, piece.tight_bottom);
    accumulated.active_pixels += piece.active_pixels;
    accumulated.sum_q12 += piece.sum_q12;
    accumulated.island_count += piece.island_count;
    accumulated.structural_gap_count += piece.structural_gap_count;
    if (retained_gap >= V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS) {
        ++accumulated.structural_gap_count;
    }
    StoreCandidateStats(receiver, accumulated);
    uint4 anchor = OcrCandidateScratch[ScratchBase(receiver) + 2u];
    anchor.z = max(anchor.z, OcrCandidateScratch[ScratchBase(member) + 2u].z);
    anchor.w = min(anchor.w, OcrCandidateScratch[ScratchBase(member) + 2u].w);
    OcrCandidateScratch[ScratchBase(receiver) + 2u] = anchor;
    uint4 meta = OcrCandidateScratch[ScratchBase(receiver) + 3u];
    meta.w = min(meta.w, OcrCandidateScratch[ScratchBase(member) + 3u].w);
    OcrCandidateScratch[ScratchBase(receiver) + 3u] = meta;
    DisableCandidate(member);
}

void MergeOrdinaryCandidates(uint candidate_begin, uint candidate_end) {
    [loop]
    for (uint member = candidate_begin; member < candidate_end; ++member) {
        if (!CandidateEnabled(member) ||
            OcrCandidateScratch[ScratchBase(member) + 3u].y != 0u) continue;
        uint member_class = OcrCandidateScratch[ScratchBase(member) + 5u].z;
        // Preserve each atomic piece's explicit orientation decision. A neighboring flat line
        // cannot erase tilted evidence by dominating a later aggregate envelope fit.
        if (member_class != CLASS_HORIZONTAL && member_class != CLASS_UNKNOWN) continue;
        CandidateStats piece = LoadCandidate(member);
        uint piece_start = piece.tight_left / OCR_CELL_WIDTH;
        [loop]
        for (uint receiver = candidate_begin; receiver < member; ++receiver) {
            if (!CandidateEnabled(receiver) ||
                OcrCandidateScratch[ScratchBase(receiver) + 3u].y != 0u) continue;
            uint receiver_class = OcrCandidateScratch[ScratchBase(receiver) + 5u].z;
            if (receiver_class != CLASS_HORIZONTAL && receiver_class != CLASS_UNKNOWN) continue;
            uint4 anchor = OcrCandidateScratch[ScratchBase(receiver) + 2u];
            uint gap = piece_start > anchor.z ? piece_start - anchor.z : 0u;
            if (gap <= V2_OCR_TEXT_JOIN_GAP_CELLS &&
                SpanHasHalfIoU(piece.tight_top, piece.tight_bottom, anchor.xy)) {
                MergeScratchCandidate(receiver, member, gap);
                break;
            }
        }
    }
}

uint ActivePixelsInRow(uint row, uint left, uint right) {
    uint total = 0u;
    [loop]
    for (uint column = left / OCR_CELL_WIDTH;
         column <= (right - 1u) / OCR_CELL_WIDTH; ++column) {
        uint cell_left = column * OCR_CELL_WIDTH;
        uint first = max(left, cell_left) - cell_left;
        uint end = min(right, cell_left + OCR_CELL_WIDTH) - cell_left;
        uint mask = CellActiveMask(CellStats[CellBase(column, row)]);
        if (first != 0u) mask &= 0xffu << first;
        if (end != OCR_CELL_WIDTH) mask &= (1u << end) - 1u;
        total += countbits(mask);
    }
    return total;
}

uint ClassifyCurrentGeometry(CandidateStats stats, out float low_slope, out float high_slope) {
    low_slope = 0.0f;
    high_slope = 0.0f;
    uint width = stats.tight_right - stats.tight_left;
    uint populated = 0u;
    float sum_x = 0.0f;
    float sum_xx = 0.0f;
    float sum_low = 0.0f;
    float sum_high = 0.0f;
    float sum_xlow = 0.0f;
    float sum_xhigh = 0.0f;
    [loop]
    for (uint bin = 0u; bin < V2_OCR_CLASSIFIER_X_BINS; ++bin) {
        uint left = stats.tight_left + bin * width / V2_OCR_CLASSIFIER_X_BINS;
        uint right = stats.tight_left + (bin + 1u) * width / V2_OCR_CLASSIFIER_X_BINS;
        if (left >= right) continue;
        uint total = 0u;
        [loop]
        for (uint row = stats.tight_top; row < stats.tight_bottom; ++row) {
            total += ActivePixelsInRow(row, left, right);
        }
        if (total == 0u) continue;
        uint cumulative = 0u;
        uint low = 0xffffffffu;
        uint high = 0xffffffffu;
        [loop]
        for (uint row = stats.tight_top; row < stats.tight_bottom; ++row) {
            cumulative += ActivePixelsInRow(row, left, right);
            if (low == 0xffffffffu &&
                cumulative * V2_OCR_CLASSIFIER_ENVELOPE_QUANTILE_DENOMINATOR >=
                total * V2_OCR_CLASSIFIER_ENVELOPE_LOW_QUANTILE_NUMERATOR) low = row;
            if (cumulative * V2_OCR_CLASSIFIER_ENVELOPE_QUANTILE_DENOMINATOR >=
                total * V2_OCR_CLASSIFIER_ENVELOPE_HIGH_QUANTILE_NUMERATOR) {
                high = row;
                break;
            }
        }
        // Relative coordinates condition the fit for narrow glyphs far from detector x=0.
        // Scale both axes into native-source pixels; portrait/rounded crops are not square by fiat.
        float x = 0.5f * (float)(left + right - 2u * stats.tight_left) *
            (float)source_width / (float)OCR_WIDTH;
        float low_y = ((float)(low - stats.tight_top) + 0.5f) *
            (float)crop_height_pixels / (float)OCR_HEIGHT;
        float high_y = ((float)(high - stats.tight_top) + 0.5f) *
            (float)crop_height_pixels / (float)OCR_HEIGHT;
        ++populated;
        sum_x += x;
        sum_xx += x * x;
        sum_low += low_y;
        sum_high += high_y;
        sum_xlow += x * low_y;
        sum_xhigh += x * high_y;
    }
    if (populated < V2_OCR_CLASSIFIER_MIN_POPULATED_BINS) return CLASS_UNKNOWN;
    float count = (float)populated;
    float denominator = count * sum_xx - sum_x * sum_x;
    if (!FiniteFloat(denominator) || denominator <= 0.0f) return CLASS_UNKNOWN;
    low_slope = (count * sum_xlow - sum_x * sum_low) / denominator;
    high_slope = (count * sum_xhigh - sum_x * sum_high) / denominator;
    if (!FiniteFloat(low_slope) || !FiniteFloat(high_slope)) return CLASS_UNKNOWN;
    if (abs(low_slope) <= V2_OCR_CLASSIFIER_HORIZONTAL_TAN &&
        abs(high_slope) <= V2_OCR_CLASSIFIER_HORIZONTAL_TAN) return CLASS_HORIZONTAL;
    if ((low_slope > V2_OCR_CLASSIFIER_SCENE_TAN &&
         high_slope > V2_OCR_CLASSIFIER_SCENE_TAN) ||
        (low_slope < -V2_OCR_CLASSIFIER_SCENE_TAN &&
         high_slope < -V2_OCR_CLASSIFIER_SCENE_TAN)) return CLASS_SCENE;
    return CLASS_AMBIGUOUS;
}

bool PriorGeometryAllowed(uint classification, float low_slope, float high_slope) {
    // Too few occupied bins have no reliable orientation. Preserve their separate current
    // line/prior path rather than interpreting a narrow glyph as an oblique object.
    if (classification == CLASS_UNKNOWN) return true;
    if (classification != CLASS_AMBIGUOUS ||
        !FiniteFloat(low_slope) || !FiniteFloat(high_slope)) return false;
    float2 magnitudes = abs(float2(low_slope, high_slope));
    if (max(magnitudes.x, magnitudes.y) > V2_OCR_CLASSIFIER_SCENE_TAN) return false;
    // A learned position may assist moderate line tilt or asymmetric glyph ink around one
    // horizontal envelope. It cannot turn two diverging, non-horizontal object edges into text.
    return min(magnitudes.x, magnitudes.y) <= V2_OCR_CLASSIFIER_HORIZONTAL_TAN ||
        (low_slope > 0.0f && high_slope > 0.0f) ||
        (low_slope < 0.0f && high_slope < 0.0f);
}

void ClassifyCandidates(uint candidate_begin, uint candidate_end) {
    [loop]
    for (uint candidate = candidate_begin; candidate < candidate_end; ++candidate) {
        if (!CandidateEnabled(candidate)) continue;
        uint4 classification = OcrCandidateScratch[ScratchBase(candidate) + 5u];
        float low_slope, high_slope;
        classification.z = ClassifyCurrentGeometry(LoadCandidate(candidate), low_slope, high_slope);
        // Transient current evidence, consumed by stack assistance before publication reuses
        // xy for its fixed line anchor. No slopes or inferred stack mask become persistent state.
        classification.xy = asuint(float2(low_slope, high_slope));
        OcrCandidateScratch[ScratchBase(candidate) + 5u] = classification;
        // Cache the current geometry verdict before publication reuses the slope words. Both
        // eligibility passes must apply the same veto, even after a group anchor is written.
        uint4 meta = OcrCandidateScratch[ScratchBase(candidate) + 3u];
        meta.x &= ~CANDIDATE_PRIOR_GEOMETRY_ALLOWED;
        if (PriorGeometryAllowed(classification.z, low_slope, high_slope)) {
            meta.x |= CANDIDATE_PRIOR_GEOMETRY_ALLOWED;
        }
        OcrCandidateScratch[ScratchBase(candidate) + 3u] = meta;
    }
}

void AttachContainedPunctuation(uint candidate_begin, uint candidate_end) {
    // A short punctuation span may precede its word in raster order. Attach after independent
    // current-line classification, without promoting a short first anchor into tall scene texture.
    // The receiver's fixed span and classification do not grow; only paired core evidence does.
    [loop]
    for (uint member = candidate_begin; member < candidate_end; ++member) {
        if (!CandidateEnabled(member) ||
            OcrCandidateScratch[ScratchBase(member) + 3u].y != 0u) continue;
        uint member_class = OcrCandidateScratch[ScratchBase(member) + 5u].z;
        // Containment cannot override an explicit classification. Only an uncertain short
        // piece can borrow a current line; moderate oblique text still needs a qualified prior.
        if (member_class != CLASS_UNKNOWN) continue;
        CandidateStats piece = LoadCandidate(member);
        [loop]
        for (uint receiver = candidate_begin; receiver < candidate_end; ++receiver) {
            if (receiver == member || !CandidateEnabled(receiver) ||
                OcrCandidateScratch[ScratchBase(receiver) + 3u].y != 0u ||
                OcrCandidateScratch[ScratchBase(receiver) + 5u].z != CLASS_HORIZONTAL) continue;
            uint4 anchor = OcrCandidateScratch[ScratchBase(receiver) + 2u];
            if (piece.tight_top < anchor.x || piece.tight_bottom > anchor.y ||
                piece.tight_bottom - piece.tight_top >= anchor.y - anchor.x) continue;
            CandidateStats target = LoadCandidate(receiver);
            uint left_begin = max(piece.tight_left / OCR_CELL_WIDTH,
                                  target.tight_left / OCR_CELL_WIDTH);
            uint right_end = min((piece.tight_right + OCR_CELL_WIDTH - 1u) / OCR_CELL_WIDTH,
                                 (target.tight_right + OCR_CELL_WIDTH - 1u) / OCR_CELL_WIDTH);
            uint gap = left_begin > right_end ? left_begin - right_end : 0u;
            if (gap > V2_OCR_TEXT_JOIN_GAP_CELLS) continue;
            MergeScratchCandidate(receiver, member, gap);
            break;
        }
    }
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
    // the tolerance denominator is <=737,280,000: exact int32 comparison keeps both edges equal.
    int delta = (int)(bounds.x + bounds.z) * (int)source_width -
        line_center_twice_source * (int)OCR_WIDTH;
    uint error = (uint)abs(delta);
    return error * V2_OCR_LINE_CENTER_TOLERANCE_DENOMINATOR <=
        2u * line_reference_width * OCR_WIDTH * V2_OCR_LINE_CENTER_TOLERANCE_NUMERATOR;
}

bool LineSpansReferenceCenter(uint4 bounds) {
    // Position/height memory cannot independently authorize a small fragment merely because
    // its midpoint lies in the broad center tolerance. Require the current bounded line group
    // to contain the signed reference center in its half-open horizontal span. All products
    // fit int32 under ResolveGeometryValid, including a reference center outside this crop.
    int center = line_center_twice_source * (int)OCR_WIDTH;
    int left = 2 * (int)bounds.x * (int)source_width;
    int right = 2 * (int)bounds.z * (int)source_width;
    return left <= center && center < right;
}

void AssistCurrentBracketedLines(uint candidate_count) {
    // A glyph's capline and baseline need not have identical ink quantiles. Keep the strict
    // independent-horizontal test unchanged; a shape-ambiguous line can instead use two current,
    // independently horizontal neighbours as a bounded three-line layout witness. Assisted
    // candidates cannot anchor another candidate, enter ordinary word merges, or teach history.
    [loop]
    for (uint candidate = 0u; candidate < candidate_count; ++candidate) {
        if (!CandidateEnabled(candidate)) continue;
        uint base = ScratchBase(candidate);
        uint4 classification = OcrCandidateScratch[base + 5u];
        if (classification.z != CLASS_AMBIGUOUS) continue;
        float2 slopes = abs(asfloat(classification.xy));
        if (!FiniteFloat(slopes.x) || !FiniteFloat(slopes.y) ||
            min(slopes.x, slopes.y) > V2_OCR_CLASSIFIER_HORIZONTAL_TAN ||
            max(slopes.x, slopes.y) > V2_OCR_CLASSIFIER_SCENE_TAN) continue;
        CandidateStats stats = LoadCandidate(candidate);
        uint4 bounds = uint4(stats.tight_left, stats.tight_top,
                             stats.tight_right, stats.tight_bottom);
        if (!LineIsCentered(bounds)) continue;
        uint height = stats.tight_bottom - stats.tight_top;
        bool above = false;
        bool below = false;
        [loop]
        for (uint anchor = 0u; anchor < candidate_count; ++anchor) {
            if (!CandidateEnabled(anchor) ||
                OcrCandidateScratch[ScratchBase(anchor) + 5u].z != CLASS_HORIZONTAL) continue;
            CandidateStats strong = LoadCandidate(anchor);
            uint4 strong_bounds = uint4(strong.tight_left, strong.tight_top,
                                        strong.tight_right, strong.tight_bottom);
            if (!LineIsCentered(strong_bounds) ||
                max(stats.tight_left, strong.tight_left) >=
                    min(stats.tight_right, strong.tight_right)) continue;
            uint strong_height = strong.tight_bottom - strong.tight_top;
            if ((float)height < (float)strong_height * V2_OCR_CLASSIFIER_MIN_HEIGHT_RATIO ||
                (float)height > (float)strong_height * V2_OCR_CLASSIFIER_MAX_HEIGHT_RATIO) continue;
            uint reference_height = max(height, strong_height);
            if (strong.tight_bottom <= stats.tight_top) {
                uint gap = stats.tight_top - strong.tight_bottom;
                if (gap * V2_OCR_CLASSIFIER_STACK_MAX_GAP_HEIGHT_DENOMINATOR <=
                    reference_height * V2_OCR_CLASSIFIER_STACK_MAX_GAP_HEIGHT_NUMERATOR) above = true;
            } else if (stats.tight_bottom <= strong.tight_top) {
                uint gap = strong.tight_top - stats.tight_bottom;
                if (gap * V2_OCR_CLASSIFIER_STACK_MAX_GAP_HEIGHT_DENOMINATOR <=
                    reference_height * V2_OCR_CLASSIFIER_STACK_MAX_GAP_HEIGHT_NUMERATOR) below = true;
            }
            if (above && below) break;
        }
        if (above && below) {
            classification.z = CLASS_STACK_ASSISTED;
            OcrCandidateScratch[base + 5u] = classification;
        }
    }
}

bool ClockLess(uint2 left, uint2 right) {
    return left.y < right.y || (left.y == right.y && left.x < right.x);
}

bool LearningGapElapsed(uint2 now, uint2 before) {
    if (ClockLess(now, before)) return false;
    uint borrow = now.x < before.x ? 1u : 0u;
    uint2 elapsed = uint2(now.x - before.x, now.y - before.y - borrow);
    return elapsed.y != 0u || elapsed.x >= V2_OCR_CLASSIFIER_LEARNING_GAP_US;
}

void ClearClassifierHistory() {
    [loop]
    for (uint word = 0u; word < V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT; ++word) {
        ClassifierHistory[word] = 0u;
    }
}

bool ClassifierHistoryMatchesContext() {
    if (ClassifierHistory[0u] != V2_OCR_CLASSIFIER_HISTORY_TAG ||
        ClassifierHistory[1u] != V2_OCR_CLASSIFIER_HISTORY_SCHEMA ||
        ClassifierHistory[2u] != V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS ||
        ClassifierHistory[3u] > V2_OCR_CLASSIFIER_PRIOR_CAP ||
        !any(uint2(ClassifierHistory[4u], ClassifierHistory[5u]) != 0u) ||
        ClassifierHistory[8u] != analysis_generation_lo ||
        ClassifierHistory[9u] != analysis_generation_hi ||
        ClassifierHistory[10u] != source_width || ClassifierHistory[11u] != source_height ||
        ClassifierHistory[12u] != crop_top_pixels || ClassifierHistory[13u] != crop_height_pixels ||
        ClassifierHistory[14u] != field_width || ClassifierHistory[15u] != field_height ||
        any(uint4(ClassifierHistory[16u], ClassifierHistory[17u],
                  ClassifierHistory[18u], ClassifierHistory[19u]) != tensor_content) ||
        ClassifierHistory[20u] != asuint(line_center_twice_source) ||
        ClassifierHistory[21u] != line_reference_width ||
        ClassifierHistory[22u] != asuint(line_search_top_source) ||
        ClassifierHistory[23u] > 1u || ClassifierHistory[24u] > 1u ||
        ClassifierHistory[30u] != 0u || ClassifierHistory[31u] != 0u) return false;
    float height = asfloat(ClassifierHistory[27u]);
    if (!FiniteFloat(height) || height < 0.0f ||
        height > (float)(V2_OCR_SAFE_ROW_BOTTOM - V2_OCR_SAFE_ROW_TOP)) return false;
    if (height == 0.0f) {
        if (ClassifierHistory[28u] != 0u || ClassifierHistory[29u] != 0u) return false;
    } else if (ClassifierHistory[28u] < V2_OCR_SAFE_ROW_TOP ||
               ClassifierHistory[28u] >= ClassifierHistory[29u] ||
               ClassifierHistory[29u] > V2_OCR_SAFE_ROW_BOTTOM) return false;
    uint2 previous_clock = uint2(ClassifierHistory[4u], ClassifierHistory[5u]);
    uint2 gap_clock = uint2(ClassifierHistory[25u], ClassifierHistory[26u]);
    if (ClassifierHistory[24u] == 0u) {
        if (any(gap_clock != 0u)) return false;
    } else if (!any(gap_clock != 0u) || ClockLess(previous_clock, gap_clock)) return false;
    [unroll]
    for (uint bin = 0u; bin < V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT; ++bin) {
        uint vote = ClassifierHistory[32u + bin];
        if (vote > V2_OCR_CLASSIFIER_PRIOR_CAP ||
            (vote != 0u && (height == 0.0f || ClassifierHistory[3u] == 0u))) return false;
    }
    return true;
}

void StampClassifierIdentity() {
    ClassifierHistory[0u] = V2_OCR_CLASSIFIER_HISTORY_TAG;
    ClassifierHistory[1u] = V2_OCR_CLASSIFIER_HISTORY_SCHEMA;
    ClassifierHistory[2u] = V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS;
    ClassifierHistory[4u] = source_clock_us.x;
    ClassifierHistory[5u] = source_clock_us.y;
    ClassifierHistory[6u] = frame_lo;
    ClassifierHistory[7u] = frame_hi;
    ClassifierHistory[8u] = analysis_generation_lo;
    ClassifierHistory[9u] = analysis_generation_hi;
    ClassifierHistory[10u] = source_width;
    ClassifierHistory[11u] = source_height;
    ClassifierHistory[12u] = crop_top_pixels;
    ClassifierHistory[13u] = crop_height_pixels;
    ClassifierHistory[14u] = field_width;
    ClassifierHistory[15u] = field_height;
    ClassifierHistory[16u] = tensor_content.x;
    ClassifierHistory[17u] = tensor_content.y;
    ClassifierHistory[18u] = tensor_content.z;
    ClassifierHistory[19u] = tensor_content.w;
    ClassifierHistory[20u] = asuint(line_center_twice_source);
    ClassifierHistory[21u] = line_reference_width;
    ClassifierHistory[22u] = asuint(line_search_top_source);
}

void BeginClassifierHistory() {
    TrustedCurrentBins = 0u;
    TrustedCurrentPixels = 0u;
    TrustedCurrentSpan = uint2(0u, 0u);
    [unroll]
    for (uint slot = 0u; slot < OCR8_FINAL_CAPACITY; ++slot) {
        RetainedTrusted[slot] = uint4(0u, 0u, 0u, 0u);
    }
    ClassifierClockValid = any(source_clock_us != 0u) ? 1u : 0u;
    ClassifierCanLearn = ClassifierClockValid;
    if (ClassifierClockValid == 0u) {
        ClearClassifierHistory();
        return;
    }
    bool compatible = input_domain_reset == 0u && ClassifierHistoryMatchesContext();
    if (compatible) {
        uint2 previous_clock = uint2(ClassifierHistory[4u], ClassifierHistory[5u]);
        uint2 previous_frame = uint2(ClassifierHistory[6u], ClassifierHistory[7u]);
        uint2 current_frame = uint2(frame_lo, frame_hi);
        if (ClockLess(source_clock_us, previous_clock) || ClockLess(current_frame, previous_frame) ||
            (all(current_frame == previous_frame) && any(source_clock_us != previous_clock))) {
            // A backwards source clock cannot qualify a historical location. Current strong
            // spatial evidence may still classify; do not manufacture learning on this sample.
            ClearClassifierHistory();
            ClassifierClockValid = 0u;
            ClassifierCanLearn = 0u;
            return;
        }
        if (all(source_clock_us == previous_clock)) ClassifierCanLearn = 0u;
    } else {
        ClearClassifierHistory();
    }
    // Repeated source observations carry the exact current frame identity but never add votes.
    StampClassifierIdentity();
}

uint CandidatePriorBin(CandidateStats stats) {
    return min(V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT - 1u,
        (stats.tight_bottom - 1u) * V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT / OCR_HEIGHT);
}

bool CandidateHasQualifiedPrior(uint candidate) {
    if ((OcrCandidateScratch[ScratchBase(candidate) + 3u].x &
         CANDIDATE_PRIOR_GEOMETRY_ALLOWED) == 0u || ClassifierClockValid == 0u ||
        ClassifierHistory[2u] != V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS) return false;
    CandidateStats stats = LoadCandidate(candidate);
    float trusted_height = asfloat(ClassifierHistory[27u]);
    float height = (float)(stats.tight_bottom - stats.tight_top);
    if (trusted_height <= 0.0f ||
        height < trusted_height * V2_OCR_CLASSIFIER_MIN_HEIGHT_RATIO ||
        height > trusted_height * V2_OCR_CLASSIFIER_MAX_HEIGHT_RATIO) return false;
    uint bin = CandidatePriorBin(stats);
    uint begin = bin > 0u ? bin - 1u : 0u;
    uint end = min(bin + 1u, V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT - 1u);
    [loop]
    for (uint current = begin; current <= end; ++current) {
        if (ClassifierHistory[32u + current] >= V2_OCR_CLASSIFIER_PRIOR_ASSIST) return true;
    }
    return false;
}

bool CandidateCanJoinClassifiedLine(uint candidate) {
    uint classification = OcrCandidateScratch[ScratchBase(candidate) + 5u].z;
    if (classification == CLASS_SCENE) return false;
    return classification == CLASS_HORIZONTAL || classification == CLASS_UNKNOWN ||
        classification == CLASS_STACK_ASSISTED ||
        CandidateHasQualifiedPrior(candidate);
}

void FinishClassifierHistory(uint final_total) {
    // Capacity eviction moves this metadata with its paired core/cover. Rejected or evicted
    // candidates cannot teach a location that the current authoritative record did not retain.
    [loop]
    for (uint slot = 0u; slot < final_total; ++slot) {
        uint4 trusted = RetainedTrusted[slot];
        TrustedCurrentBins |= trusted.x;
        if (trusted.y > TrustedCurrentPixels) {
            TrustedCurrentPixels = trusted.y;
            TrustedCurrentSpan = trusted.zw;
        }
    }
    if (ClassifierClockValid != 0u && ClassifierCanLearn != 0u) {
        if (final_total == 0u) {
            ClassifierHistory[23u] = 0u;
            if (ClassifierHistory[24u] == 0u) {
                ClassifierHistory[24u] = 1u;
                ClassifierHistory[25u] = source_clock_us.x;
                ClassifierHistory[26u] = source_clock_us.y;
            }
        } else {
            bool first_episode = ClassifierHistory[3u] == 0u;
            bool separated_episode = ClassifierHistory[23u] == 0u &&
                ClassifierHistory[24u] != 0u && LearningGapElapsed(source_clock_us,
                    uint2(ClassifierHistory[25u], ClassifierHistory[26u]));
            if (TrustedCurrentBins != 0u && (first_episode || separated_episode)) {
                [unroll]
                for (uint bin = 0u; bin < V2_OCR_CLASSIFIER_PRIOR_BIN_COUNT; ++bin) {
                    uint vote = ClassifierHistory[32u + bin];
                    vote = vote > V2_OCR_CLASSIFIER_PRIOR_DECAY ?
                        vote - V2_OCR_CLASSIFIER_PRIOR_DECAY : 0u;
                    if ((TrustedCurrentBins & (1u << bin)) != 0u) {
                        vote = min(vote + V2_OCR_CLASSIFIER_PRIOR_INCREMENT,
                                   V2_OCR_CLASSIFIER_PRIOR_CAP);
                    }
                    ClassifierHistory[32u + bin] = vote;
                }
                ClassifierHistory[3u] = min(ClassifierHistory[3u] + 1u,
                                            V2_OCR_CLASSIFIER_PRIOR_CAP);
                ClassifierHistory[27u] = asuint((float)(TrustedCurrentSpan.y - TrustedCurrentSpan.x));
                ClassifierHistory[28u] = TrustedCurrentSpan.x;
                ClassifierHistory[29u] = TrustedCurrentSpan.y;
            }
            ClassifierHistory[23u] = 1u;
            ClassifierHistory[24u] = 0u;
            ClassifierHistory[25u] = 0u;
            ClassifierHistory[26u] = 0u;
        }
    }
    [loop]
    for (uint word = 0u; word < V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT; ++word) {
        OcrBoxRecord[V2_OCR_CLASSIFIER_HISTORY_OFFSET + word] = ClassifierHistory[word];
    }
}

void PublishCenteredLines(uint candidate_count, inout uint raw_total, inout uint final_total) {
    [loop]
    for (uint candidate = 0u; candidate < candidate_count; ++candidate) {
        if (!CandidateEnabled(candidate) || !CandidateCanJoinClassifiedLine(candidate)) continue;
        CandidateStats stats = LoadCandidate(candidate);
        uint group = candidate;
        [loop]
        for (uint prior = 0u; prior < candidate; ++prior) {
            if (!CandidateEnabled(prior) || !CandidateCanJoinClassifiedLine(prior) ||
                OcrCandidateScratch[ScratchBase(prior) + 3u].z != prior) continue;
            uint4 anchor = OcrCandidateScratch[ScratchBase(prior) + 5u];
            if (SameTextLine(stats, anchor.xy,
                             OcrCandidateScratch[ScratchBase(prior) + 4u])) {
                group = prior;
                break;
            }
        }
        uint4 meta = OcrCandidateScratch[ScratchBase(candidate) + 3u];
        meta.z = group;
        OcrCandidateScratch[ScratchBase(candidate) + 3u] = meta;
        uint4 bounds = uint4(stats.tight_left, stats.tight_top, stats.tight_right, stats.tight_bottom);
        uint4 anchor = OcrCandidateScratch[ScratchBase(group) + 5u];
        if (group == candidate) {
            anchor.xy = uint2(stats.tight_top, stats.tight_bottom);
            anchor.w = anchor.z == CLASS_HORIZONTAL ? 1u : 0u;
            OcrCandidateScratch[ScratchBase(group) + 4u] = bounds;
        } else {
            uint4 previous_bounds = OcrCandidateScratch[ScratchBase(group) + 4u];
            OcrCandidateScratch[ScratchBase(group) + 4u] = uint4(
                min(previous_bounds.xy, bounds.xy), max(previous_bounds.zw, bounds.zw));
            if (OcrCandidateScratch[ScratchBase(candidate) + 5u].z == CLASS_HORIZONTAL) {
                anchor.w = 1u;
            }
        }
        OcrCandidateScratch[ScratchBase(group) + 5u] = anchor;
    }
    [loop]
    for (uint candidate = 0u; candidate < candidate_count; ++candidate) {
        if (!CandidateEnabled(candidate) || !CandidateCanJoinClassifiedLine(candidate)) continue;
        uint base = ScratchBase(candidate);
        uint group = OcrCandidateScratch[base + 3u].z;
        if (!LineIsCentered(OcrCandidateScratch[ScratchBase(group) + 4u])) continue;
        // Ordinary subtitle groups must straddle the actual reference center,
        // including strong/current-line paths. Expanded covers are never evidence.
        // Keep the existing ribbon classification/publication path separate.
        uint kind_flags = OcrCandidateScratch[base + 3u].y;
        if ((kind_flags & V2_OCR_BOX_FLAG_RIBBON) == 0u &&
            !LineSpansReferenceCenter(OcrCandidateScratch[ScratchBase(group) + 4u])) continue;
        CandidateStats stats = LoadCandidate(candidate);
        uint classification = OcrCandidateScratch[base + 5u].z;
        bool strong = classification == CLASS_HORIZONTAL;
        bool current_line = classification == CLASS_STACK_ASSISTED ||
            (classification == CLASS_UNKNOWN &&
             OcrCandidateScratch[ScratchBase(group) + 5u].w != 0u);
        if (!strong && !current_line) {
            if (!CandidateHasQualifiedPrior(candidate) ||
                !LineSpansReferenceCenter(OcrCandidateScratch[ScratchBase(group) + 4u])) continue;
        }
        float score;
        if (!CandidateEvidenceValid(stats, score)) continue;
        PublishCandidate(stats, score, OcrCandidateScratch[base + 3u].y, strong,
                         raw_total, final_total);
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

// Deterministic lower-text DB postprocess. All lanes build a compact exact occupancy index and
// clear the current OCR prefix while retaining the separately authenticated classification tail.
// Lane zero refines local rows, aggregates compatible words, and classifies current lines before
// centered eligibility, paired-cover expansion, and bounded largest-core publication.
// A broad run retains the larger index gap only long enough to classify a bottom ribbon.
// Non-ribbon text is rescanned with its smaller join gap, and every retained island must carry
// valid text evidence on its own so a distant weak pixel cannot borrow a subtitle's confidence or
// enlarge its conditioning cover.
[numthreads(OCR_RESOLVE_THREADS, 1, 1)]
void resolve_main(uint3 group_thread_id : SV_GroupThreadID) {
    uint thread_index = group_thread_id.x;
    uint record_words;
    uint record_stride;
    OcrBoxRecord.GetDimensions(record_words, record_stride);
    if (thread_index < V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT) {
        ClassifierHistory[thread_index] = 0u;
        if (record_words == OCR8_WORDS && record_stride == 4u) {
            ClassifierHistory[thread_index] =
                OcrBoxRecord[V2_OCR_CLASSIFIER_HISTORY_OFFSET + thread_index];
        }
    }
    // The tail belongs to classification history. Preserve it until lane zero has validated
    // source-clock/context identity; only the current authoritative OCR prefix is cleared here.
    if (thread_index < min(V2_OCR_CLASSIFIER_HISTORY_OFFSET, record_words)) {
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

    uint scratch_elements;
    uint scratch_stride;
    OcrCandidateScratch.GetDimensions(scratch_elements, scratch_stride);
    if (record_words != OCR8_WORDS || record_stride != 4u) return;
    if (!ResolveGeometryValid() || scratch_stride != 16u ||
        scratch_elements != V2_OCR_LOCAL_CANDIDATE_CAPACITY *
            (V2_OCR_LOCAL_CANDIDATE_WORD_COUNT / 4u)) {
        ClearAndPublishInvalidRecord();
        return;
    }

    [loop]
    for (uint thread = 0u; thread < OCR_RESOLVE_THREADS; ++thread) {
        if (ResolveNonfiniteByThread[thread] != 0u) {
            ClearAndPublishInvalidRecord();
            return;
        }
    }

    BeginClassifierHistory();

    uint raw_total = 0u;
    uint final_total = 0u;
    uint candidate_count = 0u;
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
        uint band_candidate_begin = candidate_count;

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
                if (!QueueLineCandidate(broad, broad_score, V2_OCR_BOX_FLAG_RIBBON, candidate_count)) {
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

                    // A remote column can fill this global band's blank rows. Find the local
                    // row bands inside this column island before collecting any core statistics.
                    uint local_y = band_top;
                    [loop]
                    while (local_y < band_bottom) {
                        while (local_y < band_bottom &&
                               !RestrictedRowActive(local_y, island_start, island_end)) ++local_y;
                        if (local_y >= band_bottom) break;
                        uint local_top = local_y;
                        while (local_y < band_bottom &&
                               RestrictedRowActive(local_y, island_start, island_end)) ++local_y;
                        CandidateStats piece;
                        GatherCandidateStats(local_top, local_y, island_start, island_end, piece);
                        piece.island_count = 1u;
                        float piece_score;
                        // Queue mutates scratch/count. Keep the evidence gate as control flow;
                        // HLSL may evaluate both sides of a logical && expression.
                        if (CandidateEvidenceValid(piece, piece_score)) {
                            if (!QueueLineCandidate(piece, piece_score, 0u, candidate_count)) {
                                ClearAndPublishInvalidRecord();
                                return;
                            }
                        }
                    }
                }
            }
        }
        SortCandidates(band_candidate_begin, candidate_count);
        ClassifyCandidates(band_candidate_begin, candidate_count);
        MergeOrdinaryCandidates(band_candidate_begin, candidate_count);
        // H/unknown word pieces may form a credible line together. Explicit scene/ambiguous
        // pieces stayed independent above, so recomputation cannot launder their orientation.
        ClassifyCandidates(band_candidate_begin, candidate_count);
        AttachContainedPunctuation(band_candidate_begin, candidate_count);
        SortCandidates(band_candidate_begin, candidate_count);
    }

    // Only the stack witness and eligibility/publication need all current bands. Per-band x
    // sorting and assembly above preserve row-band raster order, tie breaks, and atomic vetoes.
    AssistCurrentBracketedLines(candidate_count);
    PublishCenteredLines(candidate_count, raw_total, final_total);
    PublishHeader(true, raw_total, final_total);
    FinishClassifierHistory(final_total);
}

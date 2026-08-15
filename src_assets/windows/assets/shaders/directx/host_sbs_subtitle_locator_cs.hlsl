// Compact SLR12 lower-text authority.
//
// OCR8 is the sole geometry source.  A coherent subtitle stack plus any bottom ribbon candidates
// need two distinct, exact-frame observations before becoming one shared-plane owner. Only tight
// cores copied from the current OCR8 record may select or track geometry; only their paired
// same-frame covers may condition the BaseField. Cached owner, pending, target, and death-grace
// state never manufacture current geometry. Ordinary covers retain the established four-sided
// analytic collar; a canonical full-width/bottom ribbon cover exposes only its top collar.

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

StructuredBuffer<float4> CutBridge : register(t1);
Texture2D<float> BaseField : register(t2);
StructuredBuffer<uint> LocatorStateRead : register(t3);
StructuredBuffer<uint> ConditionParams : register(t4);
StructuredBuffer<uint> OcrRecord : register(t7);

RWStructuredBuffer<uint> LocatorState : register(u2);
RWTexture2D<float> ConditionedField : register(u3);
RWStructuredBuffer<uint> ConditionParamsOut : register(u4);
RWBuffer<uint> ConditionDispatchArgs : register(u5);

cbuffer SubtitleLocatorConstants : register(b2) {
    uint4 locator_field;   // width, height, ROI top, ROI bottom
    uint4 locator_source;  // analysis source width, height, enabled, input-domain reset
    uint4 locator_frame;   // matched frame lo/hi, analysis generation lo/hi
    uint4 locator_content; // integer half-open real-source rectangle in locator_field
};

static const uint FLAG_OWNER = 1u;
static const uint FLAG_PENDING = 2u;
static const uint FLAG_TARGET_VALID = 4u;
static const uint FLAG_TARGET_RESET = 8u;
static const uint KNOWN_FLAGS = 15u;

static const uint EVENT_NONE = 0u;
static const uint EVENT_BIRTH = 1u;
static const uint EVENT_DEATH = 2u;
static const uint EVENT_HANDOFF = 3u;

static const uint MAX_LINES = V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY;
static const uint STACK_BASE = 0u;
static const uint OLD_OWNER_BASE = 4u;
static const uint OLD_PENDING_BASE = 8u;
static const uint NEW_OWNER_BASE = 12u;
static const uint NEW_PENDING_BASE = 16u;
static const uint NEW_CURRENT_BASE = 20u;
static const uint MATCHED_BASE = 24u;
static const uint DEATH_GRACE_OBSERVATIONS =
    V2_SUBTITLE_LOCATOR_DEATH_GRACE_OBSERVATIONS;

groupshared uint PreviousState[V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT];
groupshared uint4 QualifiedCores[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint4 QualifiedCovers[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint QualifiedKinds[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint QualifiedMasks[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint4 WorkRects[28];
groupshared uint WorkKinds[28];
groupshared uint4 StackCovers[MAX_LINES];
groupshared uint4 MatchedCovers[MAX_LINES];
groupshared float TargetSamples[32];
groupshared float LineCenters[MAX_LINES];

static const uint CONDITION_PARAM_SCHEMA_WORD = 0u;
static const uint CONDITION_PARAM_TAG_WORD = 1u;
static const uint CONDITION_PARAM_CURRENT_COUNT_WORD = 2u;
static const uint CONDITION_PARAM_CURRENT_KINDS_WORD = 3u;
static const uint CONDITION_PARAM_FADE_STEP_WORD = 4u;
static const uint CONDITION_PARAM_TARGET_WORD = 5u;
static const uint CONDITION_PARAM_ORIGIN_X_WORD = 6u;
static const uint CONDITION_PARAM_ORIGIN_Y_WORD = 7u;
static const uint CONDITION_PARAM_WORD_COUNT = V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT;

bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool ZeroRect(uint4 rectangle) {
    return all(rectangle == uint4(0u, 0u, 0u, 0u));
}

uint LocatorContentWidth() {
    return locator_content.z - locator_content.x;
}

bool SubtitleTargetIsValid(float target) {
    return FiniteFloat(target) && FiniteFloat(v2_direct_container_limit) &&
        v2_direct_container_limit > 0.0f && abs(target) <= v2_direct_container_limit;
}

// The active field/ROI is carried by the existing runtime cbuffer and repeated in OCR8.  The host
// enables this path only after authenticating the DAV2 tensor shape; SLR12 independently checks
// finite ABI-sized geometry, the actual BaseField dispatch dimensions, and that the ROI is a
// non-empty subset of the exact bottom-6:1 detector crop. Any disagreement publishes no current
// authority; the complete writer publishes BaseField exactly, while full-content live production
// emits zero indirect groups and leaves the final UAV untouched.
bool LocatorDomainGeometryValid() {
    if (locator_source.x == 0u || locator_source.y == 0u || locator_source.z > 1u ||
        locator_field.x == 0u || locator_field.y == 0u ||
        locator_field.x > 0xffffu || locator_field.y > 0xffffu ||
        locator_field.z >= locator_field.w || locator_field.w > locator_field.y ||
        target_w != locator_field.x || target_h != locator_field.y ||
        locator_content.x >= locator_content.z || locator_content.y >= locator_content.w ||
        locator_content.z > locator_field.x || locator_content.w > locator_field.y ||
        locator_field.z < locator_content.y || locator_field.w > locator_content.w ||
        any(locator_content != DepthAnalysisContentCells())) {
        return false;
    }

    if (!V2SubtitleOcrFieldIsCalibrated(locator_field.x, locator_field.y)) return false;

    uint expected_roi_top;
    uint expected_roi_bottom;
    if (!V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content, V2_OCR_SAFE_ROW_TOP, expected_roi_top) ||
        !V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content, V2_OCR_SAFE_ROW_BOTTOM, expected_roi_bottom)) return false;
    return locator_field.z == expected_roi_top && locator_field.w == expected_roi_bottom;
}

bool LocatorGeometryValid() {
    return locator_source.z == 1u && LocatorDomainGeometryValid();
}

bool ValidRoiRect(uint4 rectangle) {
    return rectangle.x < rectangle.z && rectangle.y < rectangle.w &&
        rectangle.x >= locator_content.x && rectangle.z <= locator_content.z &&
        rectangle.y >= locator_field.z && rectangle.w <= locator_field.w;
}

bool ValidFieldRect(uint4 rectangle) {
    return rectangle.x < rectangle.z && rectangle.y < rectangle.w &&
        rectangle.x >= locator_content.x && rectangle.y >= locator_content.y &&
        rectangle.z <= locator_content.z && rectangle.w <= locator_content.w;
}

uint RectArea(uint4 rectangle) {
    return (rectangle.z - rectangle.x) * (rectangle.w - rectangle.y);
}

float RectIou(uint4 a, uint4 b) {
    uint left = max(a.x, b.x);
    uint top = max(a.y, b.y);
    uint right = min(a.z, b.z);
    uint bottom = min(a.w, b.w);
    uint intersection = right > left && bottom > top ? (right - left) * (bottom - top) : 0u;
    uint union_area = RectArea(a) + RectArea(b) - intersection;
    return union_area == 0u ? 0.0f : (float)intersection / (float)union_area;
}

uint4 RectSummary(uint base, uint count) {
    if (count == 0u) return uint4(0u, 0u, 0u, 0u);
    uint4 summary = WorkRects[base];
    [unroll]
    for (uint index = 1u; index < MAX_LINES; ++index) {
        if (index < count) {
            uint4 rectangle = WorkRects[base + index];
            summary.x = min(summary.x, rectangle.x);
            summary.y = min(summary.y, rectangle.y);
            summary.z = max(summary.z, rectangle.z);
            summary.w = max(summary.w, rectangle.w);
        }
    }
    return summary;
}

uint RectAreaSum(uint base, uint count) {
    uint area = 0u;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < count) area += RectArea(WorkRects[base + index]);
    }
    return area;
}

void CopyRects(uint destination, uint source, uint count) {
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[destination + index] = index < count ? WorkRects[source + index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[destination + index] = index < count ? WorkKinds[source + index] : 0u;
    }
}

bool CoherentLines(uint4 a, uint4 b) {
    uint width_a = a.z - a.x;
    uint width_b = b.z - b.x;
    uint height_a = a.w - a.y;
    uint height_b = b.w - b.y;
    uint overlap = min(a.z, b.z) > max(a.x, b.x) ?
        min(a.z, b.z) - max(a.x, b.x) : 0u;
    uint center_x_delta_twice = (a.x + a.z) > (b.x + b.z) ?
        (a.x + a.z) - (b.x + b.z) : (b.x + b.z) - (a.x + a.z);
    uint left_delta = a.x > b.x ? a.x - b.x : b.x - a.x;
    uint right_delta = a.z > b.z ? a.z - b.z : b.z - a.z;
    uint center_y_delta_twice = (a.y + a.w) > (b.y + b.w) ?
        (a.y + a.w) - (b.y + b.w) : (b.y + b.w) - (a.y + a.w);
    uint gap = 0u;
    if (a.w <= b.y) gap = b.y - a.w;
    else if (b.w <= a.y) gap = a.y - b.w;
    return overlap * 2u >= min(width_a, width_b) &&
        (center_x_delta_twice <= max(height_a, height_b) ||
         left_delta <= max(height_a, height_b) || right_delta <= max(height_a, height_b)) &&
        max(height_a, height_b) <= 2u * min(height_a, height_b) &&
        center_y_delta_twice >= min(height_a, height_b) &&
        gap * 2u <= max(height_a, height_b);
}

bool SameBaselineSegments(uint4 a, uint4 b) {
    // OCR deliberately keeps ordinary covers independent when a horizontal gap exceeds the text
    // join limit. Re-associate only strongly aligned, nearby core segments for owner selection; do
    // not union their geometry. The strict vertical relation prevents a nearby upper/lower scene
    // text line from bridging otherwise separate components.
    bool a_before_b = a.z <= b.x;
    bool b_before_a = b.z <= a.x;
    if (!a_before_b && !b_before_a) return false;
    uint height_a = a.w - a.y;
    uint height_b = b.w - b.y;
    uint shorter_height = min(height_a, height_b);
    uint taller_height = max(height_a, height_b);
    uint vertical_overlap = min(a.w, b.w) > max(a.y, b.y) ?
        min(a.w, b.w) - max(a.y, b.y) : 0u;
    uint center_y_delta_twice = (a.y + a.w) > (b.y + b.w) ?
        (a.y + a.w) - (b.y + b.w) : (b.y + b.w) - (a.y + a.w);
    uint horizontal_gap = a_before_b ? b.x - a.z : a.x - b.z;
    uint combined_span = max(a.z, b.z) - min(a.x, b.x);
    uint maximum_width = LocatorContentWidth() * V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
        V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
    return vertical_overlap * 4u >= shorter_height * 3u &&
        taller_height <= 2u * shorter_height &&
        center_y_delta_twice <= shorter_height &&
        horizontal_gap <= 8u * taller_height &&
        combined_span <= maximum_width;
}

bool ValidOcrBoxPayload(uint offset) {
    uint4 rectangle = uint4(
        OcrRecord[offset + 0u], OcrRecord[offset + 1u],
        OcrRecord[offset + 2u], OcrRecord[offset + 3u]);
    float score = asfloat(OcrRecord[offset + 4u]);
    return ValidFieldRect(rectangle) && FiniteFloat(score) &&
        score >= V2_OCR_MIN_MEAN_SCORE && score <= 1.0f &&
        (OcrRecord[offset + 5u] & ~V2_OCR_BOX_KNOWN_FLAGS) == 0u &&
        OcrRecord[offset + 6u] != 0u && OcrRecord[offset + 6u] <= V2_OCR_OUTPUT_WIDTH &&
        OcrRecord[offset + 7u] < OcrRecord[offset + 6u];
}

bool ValidOcrPair(uint slot) {
    uint raw_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    uint final_offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
    if (!ValidOcrBoxPayload(raw_offset) || !ValidOcrBoxPayload(final_offset)) return false;
    uint4 core = uint4(
        OcrRecord[raw_offset + 0u], OcrRecord[raw_offset + 1u],
        OcrRecord[raw_offset + 2u], OcrRecord[raw_offset + 3u]);
    uint4 cover = uint4(
        OcrRecord[final_offset + 0u], OcrRecord[final_offset + 1u],
        OcrRecord[final_offset + 2u], OcrRecord[final_offset + 3u]);
    if (!ValidRoiRect(core) || cover.x > core.x || cover.y > core.y ||
        cover.z < core.z || cover.w < core.w ||
        OcrRecord[raw_offset + 4u] != OcrRecord[final_offset + 4u] ||
        OcrRecord[raw_offset + 5u] != OcrRecord[final_offset + 5u] ||
        OcrRecord[raw_offset + 6u] != OcrRecord[final_offset + 6u] ||
        OcrRecord[raw_offset + 7u] != OcrRecord[final_offset + 7u]) {
        return false;
    }
    bool ribbon = (OcrRecord[raw_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u;
    if (ribbon) {
        uint width = core.z - core.x;
        uint minimum_bottom = 0u;
        bool projected = V2SubtitleOcrProjectContentRowCeil(
            locator_source.x, locator_source.y, locator_field.x, locator_field.y,
            locator_content,
            V2_OCR_SAFE_ROW_BOTTOM - V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS,
            minimum_bottom);
        if (width * V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR <
                LocatorContentWidth() * V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR ||
            !projected || core.w < minimum_bottom || core.w > locator_field.w ||
            OcrRecord[raw_offset + 6u] <= OcrRecord[raw_offset + 7u] ||
            OcrRecord[raw_offset + 7u] < V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS ||
            cover.x != locator_content.x || cover.z != locator_content.z ||
            cover.w != locator_content.w ||
            cover.y < locator_field.z) {
            return false;
        }
    } else if (!ValidRoiRect(cover)) return false;
    return true;
}

bool ZeroOcrBox(uint offset) {
    [unroll]
    for (uint word = 0u; word < V2_OCR_BOX_WORD_COUNT; ++word) {
        if (OcrRecord[offset + word] != 0u) return false;
    }
    return true;
}

bool ValidateOcrRecord(out uint final_count) {
    final_count = 0u;
    if (!LocatorGeometryValid()) {
        return false;
    }
    uint raw_count = OcrRecord[3u];
    final_count = OcrRecord[4u];
    if (OcrRecord[0u] != V2_OCR_RECORD_SCHEMA || OcrRecord[1u] != V2_OCR_RECORD_TAG ||
        OcrRecord[2u] != 1u || raw_count > V2_OCR_RAW_BOX_CAPACITY ||
        final_count > V2_OCR_FINAL_BOX_CAPACITY || raw_count != final_count ||
        OcrRecord[5u] != locator_frame.x || OcrRecord[6u] != locator_frame.y ||
        OcrRecord[7u] != locator_frame.z || OcrRecord[8u] != locator_frame.w ||
        OcrRecord[9u] != locator_source.x || OcrRecord[10u] != locator_source.y ||
        OcrRecord[11u] != locator_field.x || OcrRecord[12u] != locator_field.y ||
        OcrRecord[13u] != locator_field.z || OcrRecord[14u] != locator_field.w ||
        OcrRecord[15u] != 0u) {
        return false;
    }
    [loop]
    for (uint slot = 0u; slot < V2_OCR_RAW_BOX_CAPACITY; ++slot) {
        uint offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
        if (slot < raw_count ? !ValidOcrPair(slot) : !ZeroOcrBox(offset)) return false;
    }
    [loop]
    for (uint final_slot = 0u; final_slot < V2_OCR_FINAL_BOX_CAPACITY; ++final_slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + final_slot * V2_OCR_BOX_WORD_COUNT;
        if (final_slot >= final_count && !ZeroOcrBox(offset)) return false;
    }
    return true;
}

uint BuildCurrentStack(uint final_count) {
    uint qualified_count = 0u;
    uint ribbon_mask = 0u;
    [loop]
    for (uint slot = 0u; slot < V2_OCR_FINAL_BOX_CAPACITY; ++slot) {
        if (slot < final_count) {
            uint raw_offset = V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            uint final_offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            uint4 core = uint4(
                OcrRecord[raw_offset + 0u], OcrRecord[raw_offset + 1u],
                OcrRecord[raw_offset + 2u], OcrRecord[raw_offset + 3u]);
            uint4 cover = uint4(
                OcrRecord[final_offset + 0u], OcrRecord[final_offset + 1u],
                OcrRecord[final_offset + 2u], OcrRecord[final_offset + 3u]);
            uint kind = (OcrRecord[raw_offset + 5u] & V2_OCR_BOX_FLAG_RIBBON) != 0u ? 1u : 0u;
            uint width = core.z - core.x;
            uint height = core.w - core.y;
            // Generic subtitle-line geometry.  In particular, square badges/logos fail the
            // aspect gate without a position- or brand-specific exclusion.
            // Every calibrated DAV2 tensor has square analysis cells and the same 434-cell short
            // side, so font/target-sampling distances remain physical cell counts.  Only the
            // maximum ordinary subtitle span is a fraction of the active field width.  A ribbon
            // has independent detector topology evidence and may legitimately span the field.
            uint maximum_width = LocatorContentWidth() * V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
                V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
            if (width >= V2_SUBTITLE_LOCATOR_MIN_WIDTH_CELLS &&
                (kind != 0u || width <= maximum_width) &&
                height >= V2_SUBTITLE_LOCATOR_MIN_HEIGHT_CELLS &&
                width * V2_SUBTITLE_LOCATOR_MIN_ASPECT_DENOMINATOR >=
                    V2_SUBTITLE_LOCATOR_MIN_ASPECT_NUMERATOR * height) {
                QualifiedCores[qualified_count] = core;
                QualifiedCovers[qualified_count] = cover;
                QualifiedKinds[qualified_count] = kind;
                if (kind != 0u) ribbon_mask |= 1u << qualified_count;
                ++qualified_count;
            }
        }
    }
    if (qualified_count == 0u) return 0u;

    [loop]
    for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
        QualifiedMasks[index] = index < qualified_count && QualifiedKinds[index] == 0u ?
            (1u << index) : 0u;
    }
    [loop]
    for (uint closure_pass = 0u; closure_pass < V2_OCR_FINAL_BOX_CAPACITY; ++closure_pass) {
        [loop]
        for (uint a = 0u; a < V2_OCR_FINAL_BOX_CAPACITY; ++a) {
            if (a < qualified_count) {
                uint expanded = QualifiedMasks[a];
                [loop]
                for (uint b = 0u; b < V2_OCR_FINAL_BOX_CAPACITY; ++b) {
                    if (b < qualified_count && QualifiedKinds[b] == 0u &&
                        (expanded & (1u << b)) != 0u) {
                        [loop]
                        for (uint c = 0u; c < V2_OCR_FINAL_BOX_CAPACITY; ++c) {
                            if (c < qualified_count && QualifiedKinds[c] == 0u &&
                                (CoherentLines(QualifiedCores[b], QualifiedCores[c]) ||
                                 SameBaselineSegments(QualifiedCores[b], QualifiedCores[c]))) {
                                expanded |= 1u << c;
                            }
                        }
                    }
                }
                QualifiedMasks[a] = expanded;
            }
        }
    }

    uint best_mask = 0u;
    uint best_area = 0u;
    uint4 best_bbox = uint4(0u, 0u, 0u, 0u);
    [loop]
    for (uint root = 0u; root < V2_OCR_FINAL_BOX_CAPACITY; ++root) {
        if (root < qualified_count && QualifiedKinds[root] == 0u) {
            uint mask = QualifiedMasks[root];
            // Evaluate each connected component once and reject unsupported >4-line blocks.
            if ((mask & ((1u << root) - 1u)) == 0u) {
                uint count = 0u;
                uint area = 0u;
                uint4 bbox = uint4(0xffffffffu, 0xffffffffu, 0u, 0u);
                [loop]
                for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
                    if (index < qualified_count && (mask & (1u << index)) != 0u) {
                        uint4 rectangle = QualifiedCores[index];
                        ++count;
                        area += RectArea(rectangle);
                        bbox.x = min(bbox.x, rectangle.x);
                        bbox.y = min(bbox.y, rectangle.y);
                        bbox.z = max(bbox.z, rectangle.z);
                        bbox.w = max(bbox.w, rectangle.w);
                    }
                }
                uint maximum_width = LocatorContentWidth() *
                    V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR /
                    V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR;
                bool better = count <= MAX_LINES && bbox.z - bbox.x <= maximum_width &&
                    (best_mask == 0u || area > best_area ||
                    (area == best_area && (bbox.w > best_bbox.w ||
                    (bbox.w == best_bbox.w && (bbox.y > best_bbox.y ||
                    (bbox.y == best_bbox.y && bbox.x < best_bbox.x))))));
                if (better) {
                    best_mask = mask;
                    best_area = area;
                    best_bbox = bbox;
                }
            }
        }
    }
    uint selected_mask = best_mask | ribbon_mask;
    if (selected_mask == 0u) return 0u;

    uint selected_count = 0u;
    [loop]
    for (uint selected_index = 0u; selected_index < V2_OCR_FINAL_BOX_CAPACITY; ++selected_index) {
        if (selected_index < qualified_count && (selected_mask & (1u << selected_index)) != 0u) {
            ++selected_count;
        }
    }
    // The compact authenticated state has four rectangles.  Never silently discard a detected
    // ribbon or an ordinary line to make a mixed owner fit: an over-capacity observation abstains.
    if (selected_count > MAX_LINES) return 0u;

    uint stack_count = 0u;
    [loop]
    for (uint collect_index = 0u; collect_index < V2_OCR_FINAL_BOX_CAPACITY; ++collect_index) {
        if (collect_index < qualified_count && (selected_mask & (1u << collect_index)) != 0u) {
            WorkRects[STACK_BASE + stack_count] = QualifiedCores[collect_index];
            WorkKinds[STACK_BASE + stack_count] = QualifiedKinds[collect_index];
            StackCovers[stack_count] = QualifiedCovers[collect_index];
            ++stack_count;
        }
    }
    // Canonical top/left order makes temporal comparison and dumps deterministic.
    [unroll]
    for (uint sort_a = 0u; sort_a < MAX_LINES; ++sort_a) {
        [unroll]
        for (uint sort_b = sort_a + 1u; sort_b < MAX_LINES; ++sort_b) {
            if (sort_b < stack_count) {
                uint4 left = WorkRects[STACK_BASE + sort_a];
                uint4 right = WorkRects[STACK_BASE + sort_b];
                if (right.y < left.y || (right.y == left.y && right.x < left.x)) {
                    WorkRects[STACK_BASE + sort_a] = right;
                    WorkRects[STACK_BASE + sort_b] = left;
                    uint4 cover_swap = StackCovers[sort_a];
                    StackCovers[sort_a] = StackCovers[sort_b];
                    StackCovers[sort_b] = cover_swap;
                    uint kind_swap = WorkKinds[STACK_BASE + sort_a];
                    WorkKinds[STACK_BASE + sort_a] = WorkKinds[STACK_BASE + sort_b];
                    WorkKinds[STACK_BASE + sort_b] = kind_swap;
                }
            }
        }
    }
    return stack_count;
}

bool ValidateAndAccumulateRoiRectangle(
    uint4 rectangle, uint slot, uint count, inout uint4 bbox, inout uint area) {
    if (slot >= count) return ZeroRect(rectangle);
    if (!ValidRoiRect(rectangle)) return false;
    if (slot == 0u) bbox = rectangle;
    else {
        bbox.x = min(bbox.x, rectangle.x);
        bbox.y = min(bbox.y, rectangle.y);
        bbox.z = max(bbox.z, rectangle.z);
        bbox.w = max(bbox.w, rectangle.w);
    }
    area += RectArea(rectangle);
    return true;
}

bool ValidateCurrentRectangle(uint4 rectangle, uint slot, uint count, uint kinds) {
    if (slot >= count) return ZeroRect(rectangle);
    bool ribbon = ((kinds >> slot) & 1u) != 0u;
    return ribbon ?
        (ValidFieldRect(rectangle) && rectangle.x == locator_content.x &&
         rectangle.z == locator_content.z && rectangle.w == locator_content.w &&
        rectangle.y >= locator_field.z) : ValidRoiRect(rectangle);
}

bool CanonicalCoreOrder(uint4 previous, uint4 current) {
    return current.y > previous.y || (current.y == previous.y && current.x >= previous.x);
}

bool ValidatePreviousRectBlock(uint offset, uint count, uint4 expected_bbox, uint expected_area) {
    uint4 bbox = uint4(0u, 0u, 0u, 0u);
    uint area = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = offset + slot * 4u;
        uint4 rectangle = uint4(
            PreviousState[base + 0u], PreviousState[base + 1u],
            PreviousState[base + 2u], PreviousState[base + 3u]);
        if (!ValidateAndAccumulateRoiRectangle(
                rectangle, slot, count, bbox, area)) return false;
        if (slot > 0u && slot < count) {
            uint previous_base = base - 4u;
            uint4 previous = uint4(
                PreviousState[previous_base + 0u], PreviousState[previous_base + 1u],
                PreviousState[previous_base + 2u], PreviousState[previous_base + 3u]);
            if (!CanonicalCoreOrder(previous, rectangle)) return false;
        }
    }
    return all(bbox == expected_bbox) && area == expected_area;
}

bool ValidatePreviousCurrent(uint count) {
    uint kinds = (PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
        uint4 rectangle = uint4(
            PreviousState[base + 0u], PreviousState[base + 1u],
            PreviousState[base + 2u], PreviousState[base + 3u]);
        if (!ValidateCurrentRectangle(rectangle, slot, count, kinds)) return false;
    }
    return true;
}

uint KindMaskForCount(uint count) {
    return count == 0u ? 0u : (1u << count) - 1u;
}

bool PackedKindsValid(uint packed, uint shift, uint count) {
    uint kinds = (packed >> shift) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    return (kinds & ~KindMaskForCount(count)) == 0u;
}

bool ValidatePreviousState() {
    uint flags = PreviousState[2u];
    uint owner_count = PreviousState[4u];
    uint pending_count = PreviousState[12u];
    uint current_count = PreviousState[20u];
    bool owner = (flags & FLAG_OWNER) != 0u;
    bool pending = (flags & FLAG_PENDING) != 0u;
    bool target_valid = (flags & FLAG_TARGET_VALID) != 0u;
    bool target_reset = (flags & FLAG_TARGET_RESET) != 0u;
    uint packed_kinds = PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD];
    uint known_kind_bits =
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    if (PreviousState[0u] != V2_SUBTITLE_LOCATOR_STATE_SCHEMA ||
        PreviousState[1u] != V2_SUBTITLE_LOCATOR_STATE_TAG || (flags & ~KNOWN_FLAGS) != 0u ||
        owner_count > MAX_LINES || pending_count > MAX_LINES || current_count > MAX_LINES ||
        PreviousState[21u] > EVENT_HANDOFF || PreviousState[24u] > 2u ||
        PreviousState[27u] != locator_field.x || PreviousState[28u] != locator_field.y ||
        (packed_kinds & ~known_kind_bits) != 0u ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT, owner_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT, pending_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT, current_count) ||
        owner != (owner_count != 0u) ||
        pending != (pending_count != 0u) || owner != (PreviousState[3u] != 0u) ||
        (current_count != 0u && (!owner || !target_valid || current_count > owner_count))) {
        return false;
    }
    if (!ValidatePreviousRectBlock(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, owner_count,
            uint4(PreviousState[5u], PreviousState[6u], PreviousState[7u], PreviousState[8u]),
            PreviousState[9u]) ||
        !ValidatePreviousRectBlock(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, pending_count,
            uint4(PreviousState[13u], PreviousState[14u], PreviousState[15u], PreviousState[16u]),
            PreviousState[17u]) || !ValidatePreviousCurrent(current_count)) {
        return false;
    }
    float target = asfloat(PreviousState[18u]);
    uint lifetime_count = PreviousState[25u];
    if (owner) {
        if (lifetime_count > V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS ||
            PreviousState[29u] != 0u || PreviousState[30u] != 0u ||
            (lifetime_count != 0u && PreviousState[21u] != EVENT_NONE)) return false;
        if (target_valid) {
            if (target_reset || PreviousState[19u] != PreviousState[3u] ||
                !SubtitleTargetIsValid(target) ||
                PreviousState[24u] == 0u) return false;
        } else if (target_reset) {
            if (lifetime_count != 0u || PreviousState[18u] != 0u ||
                PreviousState[19u] != 0u || current_count != 0u ||
                PreviousState[24u] != 0u) return false;
        } else if (lifetime_count != 0u || PreviousState[18u] != 0u ||
                   PreviousState[19u] != 0u || current_count != 0u ||
                   PreviousState[24u] != 0u) return false;
    } else if (lifetime_count == 0u) {
        if (target_valid || target_reset || PreviousState[18u] != 0u ||
            PreviousState[19u] != 0u || PreviousState[29u] != 0u ||
            PreviousState[30u] != 0u || current_count != 0u || PreviousState[24u] != 0u) {
            return false;
        }
    } else {
        if (lifetime_count > DEATH_GRACE_OBSERVATIONS) return false;
        uint4 bounds = uint4(
            PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
            PreviousState[29u] >> 16u, PreviousState[30u] >> 16u);
        if (target_valid || target_reset || PreviousState[19u] != 0u ||
            !SubtitleTargetIsValid(target) ||
            !ValidRoiRect(bounds) || current_count != 0u || PreviousState[24u] != 0u) return false;
    }
    return true;
}

void LoadPreviousRects(uint state_offset, uint kind_shift, uint work_base, uint count) {
    uint kinds = (PreviousState[V2_SUBTITLE_LOCATOR_KIND_WORD] >> kind_shift) &
        V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint offset = state_offset + slot * 4u;
        WorkRects[work_base + slot] = slot < count ? uint4(
            PreviousState[offset + 0u], PreviousState[offset + 1u],
            PreviousState[offset + 2u], PreviousState[offset + 3u]) :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[work_base + slot] = slot < count ? ((kinds >> slot) & 1u) : 0u;
    }
}

bool StackCompatible(uint first_base, uint first_count, uint second_base, uint second_count) {
    if (first_count == 0u || first_count != second_count) return false;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < first_count) {
            if (WorkKinds[first_base + index] != WorkKinds[second_base + index]) return false;
            // A stack transaction confirms every member, not merely enough aggregate overlap.
            // Otherwise two unchanged lines can lend their IoU to a newly disjoint third line and
            // give that one-observation geometry immediate owner authority.
            if (RectIou(WorkRects[first_base + index], WorkRects[second_base + index]) <
                V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD) {
                return false;
            }
        }
    }
    return true;
}

uint MatchCurrentToOwner(uint current_count, uint owner_count) {
    uint used = 0u;
    uint matched = 0u;
    [unroll]
    for (uint current_index = 0u; current_index < MAX_LINES; ++current_index) {
        if (current_index < current_count) {
            float best_iou = V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD;
            uint best_owner = MAX_LINES;
            [unroll]
            for (uint owner_index = 0u; owner_index < MAX_LINES; ++owner_index) {
                if (owner_index < owner_count && (used & (1u << owner_index)) == 0u) {
                    if (WorkKinds[STACK_BASE + current_index] !=
                        WorkKinds[OLD_OWNER_BASE + owner_index]) continue;
                    float iou = RectIou(
                        WorkRects[STACK_BASE + current_index], WorkRects[OLD_OWNER_BASE + owner_index]);
                    if (iou >= best_iou) {
                        best_iou = iou;
                        best_owner = owner_index;
                    }
                }
            }
            if (best_owner < MAX_LINES) {
                used |= 1u << best_owner;
                WorkRects[MATCHED_BASE + matched] = WorkRects[STACK_BASE + current_index];
                WorkKinds[MATCHED_BASE + matched] = WorkKinds[STACK_BASE + current_index];
                MatchedCovers[matched] = StackCovers[current_index];
                ++matched;
            }
        }
    }
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index >= matched) {
            WorkRects[MATCHED_BASE + index] = uint4(0u, 0u, 0u, 0u);
            WorkKinds[MATCHED_BASE + index] = 0u;
            MatchedCovers[index] = uint4(0u, 0u, 0u, 0u);
        }
    }
    return matched;
}

void CopyStackCoversToCurrent(uint count) {
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[NEW_CURRENT_BASE + index] = index < count ? StackCovers[index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[NEW_CURRENT_BASE + index] = index < count ?
            WorkKinds[STACK_BASE + index] : 0u;
    }
}

void CopyMatchedCoversToCurrent(uint count) {
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        WorkRects[NEW_CURRENT_BASE + index] = index < count ? MatchedCovers[index] :
            uint4(0u, 0u, 0u, 0u);
        WorkKinds[NEW_CURRENT_BASE + index] = index < count ?
            WorkKinds[MATCHED_BASE + index] : 0u;
    }
}

uint NextGeneration(uint value) {
    return value == 0u || value >= 0xfffffffdu ? 1u : value + 1u;
}

bool SampleOwnerTarget(uint owner_base, uint owner_count, out float target) {
    target = 0.0f;
    if ((owner_base != NEW_OWNER_BASE && owner_base != MATCHED_BASE) ||
        owner_count == 0u || owner_count > MAX_LINES || locator_source.x == 0u ||
        !FiniteFloat(v2_direct_container_limit) || v2_direct_container_limit <= 0.0f) return false;
    [unroll]
    for (uint center_index = 0u; center_index < MAX_LINES; ++center_index) {
        LineCenters[center_index] = center_index < owner_count ?
            0.5f * (float)(WorkRects[owner_base + center_index].x +
                           WorkRects[owner_base + center_index].z - 1u) : 0.0f;
    }
    [unroll]
    for (uint center_a = 0u; center_a < MAX_LINES; ++center_a) {
        [unroll]
        for (uint center_b = center_a + 1u; center_b < MAX_LINES; ++center_b) {
            if (center_b < owner_count && LineCenters[center_b] < LineCenters[center_a]) {
                float swap_value = LineCenters[center_a];
                LineCenters[center_a] = LineCenters[center_b];
                LineCenters[center_b] = swap_value;
            }
        }
    }
    float center = (owner_count & 1u) != 0u ? LineCenters[owner_count / 2u] :
        0.5f * (LineCenters[owner_count / 2u - 1u] + LineCenters[owner_count / 2u]);
    uint owner_top = RectSummary(owner_base, owner_count).y;
    uint outer_y_offset = 10u;
    uint inner_y_offset = 4u;
    uint sample_y0 = clamp(
        owner_top >= outer_y_offset ? owner_top - outer_y_offset : 0u,
        locator_content.y,
        locator_content.w - 1u);
    uint sample_y1 = clamp(
        owner_top >= inner_y_offset ? owner_top - inner_y_offset : 0u,
        locator_content.y,
        locator_content.w - 1u);
    bool first_row_valid = true;
    bool second_row_valid = true;
    [loop]
    for (uint sample_index = 0u; sample_index < 16u; ++sample_index) {
        float sample_x_float = center - 30.0f + 4.0f * (float)sample_index;
        uint sample_x = (uint)clamp(
            floor(sample_x_float + 0.5f),
            (float)locator_content.x,
            (float)(locator_content.z - 1u));
        float first = BaseField.Load(int3(sample_x, sample_y0, 0));
        float second = BaseField.Load(int3(sample_x, sample_y1, 0));
        bool first_valid = FiniteFloat(first) && abs(first) <= v2_direct_container_limit;
        bool second_valid = FiniteFloat(second) && abs(second) <= v2_direct_container_limit;
        first_row_valid = first_row_valid && first_valid;
        second_row_valid = second_row_valid && second_valid;
        TargetSamples[sample_index] = first_valid ? first : 0.0f;
        TargetSamples[16u + sample_index] = second_valid ? second : 0.0f;
    }
    // Each row must independently describe the same local supporting plane. Sort the rows first
    // so their medians reject a depth edge that happens to divide the fixed sampling strip.
    [loop]
    for (uint row = 0u; row < 2u; ++row) {
        uint row_base = row * 16u;
        bool row_valid = row == 0u ? first_row_valid : second_row_valid;
        [loop]
        for (uint row_sort_outer = 0u; row_sort_outer < 16u; ++row_sort_outer) {
            uint outer_index = row_base + row_sort_outer;
            uint minimum_index = outer_index;
            [loop]
            for (uint row_sort_scan = row_sort_outer + 1u;
                 row_sort_scan < 16u; ++row_sort_scan) {
                uint scan_index = row_base + row_sort_scan;
                if (row_valid && TargetSamples[scan_index] < TargetSamples[minimum_index]) {
                    minimum_index = scan_index;
                }
            }
            float swap_value = TargetSamples[outer_index];
            TargetSamples[outer_index] = TargetSamples[minimum_index];
            TargetSamples[minimum_index] = swap_value;
        }
    }
    precise float binocular_scale = 2.0f * (float)locator_source.x;
    precise float first_row_median = 0.5f * (TargetSamples[7u] + TargetSamples[8u]);
    precise float second_row_median = 0.5f * (TargetSamples[23u] + TargetSamples[24u]);
    if (!FiniteFloat(binocular_scale) || binocular_scale <= 0.0f ||
        !FiniteFloat(V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS) ||
        V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS < 0.0f ||
        !FiniteFloat(V2_SUBTITLE_TARGET_MAX_ROW_MEDIAN_DELTA_BINOCULAR_SOURCE_PIXELS) ||
        V2_SUBTITLE_TARGET_MAX_ROW_MEDIAN_DELTA_BINOCULAR_SOURCE_PIXELS < 0.0f) {
        return false;
    }
    precise float first_row_q1 = 0.5f * (TargetSamples[3u] + TargetSamples[4u]);
    precise float first_row_q3 = 0.5f * (TargetSamples[11u] + TargetSamples[12u]);
    precise float first_row_iqr = first_row_q3 - first_row_q1;
    precise float second_row_q1 = 0.5f * (TargetSamples[19u] + TargetSamples[20u]);
    precise float second_row_q3 = 0.5f * (TargetSamples[27u] + TargetSamples[28u]);
    precise float second_row_iqr = second_row_q3 - second_row_q1;
    bool first_row_coherent = first_row_valid &&
        first_row_iqr * binocular_scale <=
            V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS;
    bool second_row_coherent = second_row_valid &&
        second_row_iqr * binocular_scale <=
            V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS;
    if (!first_row_coherent && !second_row_coherent) return false;
    if (first_row_valid && second_row_valid) {
        precise float median_delta_pixels =
            abs(second_row_median - first_row_median) * binocular_scale;
        target = median_delta_pixels <=
                V2_SUBTITLE_TARGET_MAX_ROW_MEDIAN_DELTA_BINOCULAR_SOURCE_PIXELS ?
            0.5f * (first_row_median + second_row_median) :
            max(first_row_median, second_row_median);
    } else if (first_row_coherent) {
        target = first_row_median;
    } else {
        target = second_row_median;
    }
    return SubtitleTargetIsValid(target);
}

bool UpdateOwnerTarget(float previous, float desired, out float updated) {
    updated = previous;
    if (!SubtitleTargetIsValid(previous) || !SubtitleTargetIsValid(desired) ||
        !FiniteFloat(V2_SUBTITLE_TARGET_DEADBAND_BINOCULAR_SOURCE_PIXELS) ||
        !FiniteFloat(V2_SUBTITLE_TARGET_EMA_ALPHA) ||
        !FiniteFloat(V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS) ||
        V2_SUBTITLE_TARGET_DEADBAND_BINOCULAR_SOURCE_PIXELS < 0.0f ||
        V2_SUBTITLE_TARGET_EMA_ALPHA <= 0.0f || V2_SUBTITLE_TARGET_EMA_ALPHA > 1.0f ||
        V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS <= 0.0f) {
        return false;
    }

    precise float binocular_scale = 2.0f * (float)locator_source.x;
    precise float delta_u = desired - previous;
    precise float delta_pixels = delta_u * binocular_scale;
    if (!FiniteFloat(V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS) ||
        V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS < 0.0f ||
        abs(delta_pixels) > V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS) {
        return false;
    }
    // This branch is the temporal-stability contract: sub-deadband observations retain the exact
    // previous R32_FLOAT bits rather than reconstructing an algebraically equal value.
    if (abs(delta_pixels) <=
        V2_SUBTITLE_TARGET_DEADBAND_BINOCULAR_SOURCE_PIXELS) return true;

    precise float ema_pixels = delta_pixels * V2_SUBTITLE_TARGET_EMA_ALPHA;
    precise float step_pixels = clamp(
        ema_pixels,
        -V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS,
        V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS);
    precise float step_u = step_pixels / binocular_scale;
    precise float candidate = previous + step_u;
    updated = clamp(candidate, -v2_direct_container_limit, v2_direct_container_limit);
    return SubtitleTargetIsValid(updated);
}

void StoreRectBlock(uint offset, uint base, uint count) {
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint4 rectangle = slot < count ? WorkRects[base + slot] : uint4(0u, 0u, 0u, 0u);
        LocatorState[offset + slot * 4u + 0u] = rectangle.x;
        LocatorState[offset + slot * 4u + 1u] = rectangle.y;
        LocatorState[offset + slot * 4u + 2u] = rectangle.z;
        LocatorState[offset + slot * 4u + 3u] = rectangle.w;
    }
}

uint PackKinds(uint base, uint count, uint shift) {
    uint kinds = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < count) kinds |= (WorkKinds[base + slot] & 1u) << slot;
    }
    return kinds << shift;
}

void PublishState(
    uint owner_generation, uint owner_count, uint pending_count, uint current_count,
    float target, bool target_valid, bool target_reset, uint fade_step, uint lifetime_count,
    uint4 grace_bounds, uint event, uint scene_epoch
) {
    uint flags = (owner_count != 0u ? FLAG_OWNER : 0u) |
        (pending_count != 0u ? FLAG_PENDING : 0u) |
        (target_valid ? FLAG_TARGET_VALID : 0u) | (target_reset ? FLAG_TARGET_RESET : 0u);
    uint4 owner_bbox = RectSummary(NEW_OWNER_BASE, owner_count);
    uint4 pending_bbox = RectSummary(NEW_PENDING_BASE, pending_count);
    LocatorState[0u] = V2_SUBTITLE_LOCATOR_STATE_SCHEMA;
    LocatorState[1u] = V2_SUBTITLE_LOCATOR_STATE_TAG;
    LocatorState[2u] = flags;
    LocatorState[3u] = owner_count != 0u ? owner_generation : 0u;
    LocatorState[4u] = owner_count;
    LocatorState[5u] = owner_bbox.x;
    LocatorState[6u] = owner_bbox.y;
    LocatorState[7u] = owner_bbox.z;
    LocatorState[8u] = owner_bbox.w;
    LocatorState[9u] = RectAreaSum(NEW_OWNER_BASE, owner_count);
    LocatorState[10u] = locator_frame.z;
    LocatorState[11u] = locator_frame.w;
    LocatorState[12u] = pending_count;
    LocatorState[13u] = pending_bbox.x;
    LocatorState[14u] = pending_bbox.y;
    LocatorState[15u] = pending_bbox.z;
    LocatorState[16u] = pending_bbox.w;
    LocatorState[17u] = RectAreaSum(NEW_PENDING_BASE, pending_count);
    LocatorState[18u] = asuint(target);
    LocatorState[19u] = target_valid ? owner_generation : 0u;
    LocatorState[20u] = target_valid ? current_count : 0u;
    LocatorState[21u] = event;
    LocatorState[22u] = locator_frame.x;
    LocatorState[23u] = locator_frame.y;
    LocatorState[24u] = target_valid ? fade_step : 0u;
    LocatorState[25u] = lifetime_count;
    LocatorState[26u] = scene_epoch;
    LocatorState[27u] = locator_field.x;
    LocatorState[28u] = locator_field.y;
    LocatorState[29u] = owner_count == 0u && lifetime_count != 0u ?
        (grace_bounds.z << 16u) | grace_bounds.x : 0u;
    LocatorState[30u] = owner_count == 0u && lifetime_count != 0u ?
        (grace_bounds.w << 16u) | grace_bounds.y : 0u;
    LocatorState[V2_SUBTITLE_LOCATOR_KIND_WORD] =
        PackKinds(NEW_OWNER_BASE, owner_count, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        PackKinds(NEW_PENDING_BASE, pending_count, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        PackKinds(NEW_CURRENT_BASE, target_valid ? current_count : 0u,
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    StoreRectBlock(V2_SUBTITLE_LOCATOR_OWNER_OFFSET, NEW_OWNER_BASE, owner_count);
    StoreRectBlock(V2_SUBTITLE_LOCATOR_PENDING_OFFSET, NEW_PENDING_BASE, pending_count);
    StoreRectBlock(
        V2_SUBTITLE_LOCATOR_CURRENT_OFFSET, NEW_CURRENT_BASE,
        target_valid ? current_count : 0u);
}

[numthreads(1, 1, 1)]
void resolve_main(uint3 dispatch_id : SV_DispatchThreadID) {
    [loop]
    for (uint index = 0u; index < V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT; ++index) {
        PreviousState[index] = LocatorState[index];
        LocatorState[index] = 0u;
    }
    [loop]
    for (uint work_index = 0u; work_index < 28u; ++work_index) {
        WorkRects[work_index] = uint4(0u, 0u, 0u, 0u);
        WorkKinds[work_index] = 0u;
    }
    [unroll]
    for (uint cover_index = 0u; cover_index < MAX_LINES; ++cover_index) {
        StackCovers[cover_index] = uint4(0u, 0u, 0u, 0u);
        MatchedCovers[cover_index] = uint4(0u, 0u, 0u, 0u);
    }

    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint scene_epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    // A no-submit/abstaining observation still has a valid source/field domain in which the
    // previous owner lifetime can age. Only current OCR authority requires locator_source.z == 1.
    bool locator_domain_valid = LocatorDomainGeometryValid();
    bool old_valid = locator_domain_valid && ValidatePreviousState();
    uint old_owner_count = old_valid ? PreviousState[4u] : 0u;
    uint old_pending_count = old_valid ? PreviousState[12u] : 0u;
    if (old_valid) {
        LoadPreviousRects(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT,
            OLD_OWNER_BASE, old_owner_count);
        LoadPreviousRects(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT,
            OLD_PENDING_BASE, old_pending_count);
    }
    bool distinct_observation = !old_valid || PreviousState[22u] != locator_frame.x ||
        PreviousState[23u] != locator_frame.y || PreviousState[10u] != locator_frame.z ||
        PreviousState[11u] != locator_frame.w;
    bool hard_cut = cut_valid && ((distinct_observation && SBS_STATE_HARD_CUT_PULSE(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_PULSE]) > 0.5f) ||
        (old_valid && PreviousState[26u] != scene_epoch));

    bool old_target_valid = old_valid && (PreviousState[2u] & FLAG_TARGET_VALID) != 0u;
    float old_target = old_target_valid ? asfloat(PreviousState[18u]) : 0.0f;
    uint old_unreliable_holds = old_valid && old_owner_count != 0u ? PreviousState[25u] : 0u;
    uint old_grace = old_valid && old_owner_count == 0u ? PreviousState[25u] : 0u;
    uint4 old_grace_bounds = old_grace != 0u ? uint4(
        PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
        PreviousState[29u] >> 16u, PreviousState[30u] >> 16u) :
        uint4(0u, 0u, 0u, 0u);
    float old_cached_target = old_grace != 0u ? asfloat(PreviousState[18u]) : 0.0f;

    uint final_count = 0u;
    bool ocr_valid = ValidateOcrRecord(final_count);
    if (!cut_valid || !locator_domain_valid ||
        ((locator_source.w != 0u || hard_cut) && !ocr_valid)) {
        PublishState(0u, 0u, 0u, 0u, 0.0f, false, false, 0u, 0u,
                     uint4(0u, 0u, 0u, 0u), EVENT_NONE, scene_epoch);
        return;
    }
    if (!ocr_valid) {
        // Missing, stale, abstaining, and malformed OCR have no current geometry and can never
        // confirm pending state. They are nevertheless a missed observation of an otherwise valid
        // owner lifetime: clear pending, cache the bounded target, and age its ordinary six-step
        // death grace. A hard cut above remains an explicit scene boundary, so invalid OCR on that
        // boundary cannot carry the old plane into the new scene.
        uint grace = 0u;
        uint4 grace_bounds = uint4(0u, 0u, 0u, 0u);
        float cached_target = 0.0f;
        uint event = EVENT_NONE;
        if (old_owner_count != 0u && old_target_valid) {
            grace = DEATH_GRACE_OBSERVATIONS;
            grace_bounds = RectSummary(OLD_OWNER_BASE, old_owner_count);
            cached_target = old_target;
            event = EVENT_DEATH;
        } else if (old_grace != 0u) {
            grace = distinct_observation ? old_grace - 1u : old_grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                cached_target = old_cached_target;
            }
        }
        PublishState(0u, 0u, 0u, 0u, cached_target, false, false, 0u, grace,
                     grace_bounds, event, scene_epoch);
        return;
    }
    uint stack_count = BuildCurrentStack(final_count);

    uint new_owner_count = 0u;
    uint new_pending_count = 0u;
    uint authority_count = 0u;
    uint owner_generation = 0u;
    uint event = EVENT_NONE;
    bool continuing_owner = false;
    bool new_owner = false;
    bool cut_survivor = false;
    bool inherit_target = false;
    float inherited_target = 0.0f;
    uint grace = 0u;
    uint4 grace_bounds = uint4(0u, 0u, 0u, 0u);

    if (locator_source.w != 0u) {
        if (stack_count != 0u) {
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            new_pending_count = stack_count;
        }
    } else if (hard_cut) {
        if (old_owner_count != 0u && stack_count != 0u) {
            uint matched = MatchCurrentToOwner(stack_count, old_owner_count);
            if (matched != 0u) {
                CopyRects(NEW_OWNER_BASE, MATCHED_BASE, matched);
                CopyMatchedCoversToCurrent(matched);
                new_owner_count = matched;
                authority_count = matched;
                // A cut starts a new owner generation even when geometry survives. The old scene
                // target is discarded below; reliable current evidence restarts at half strength.
                owner_generation = NextGeneration(PreviousState[3u]);
                cut_survivor = true;
                if (matched < stack_count) {
                    CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                    new_pending_count = stack_count;
                }
            } else {
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                new_pending_count = stack_count;
                event = EVENT_DEATH;
            }
        } else {
            if (stack_count != 0u) {
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                new_pending_count = stack_count;
            }
            if (old_owner_count != 0u) event = EVENT_DEATH;
        }
    } else if (stack_count == 0u) {
        if (old_owner_count != 0u) {
            event = EVENT_DEATH;
            if (old_target_valid) {
                grace = DEATH_GRACE_OBSERVATIONS;
                grace_bounds = RectSummary(OLD_OWNER_BASE, old_owner_count);
                inherited_target = old_target;
            }
        } else if (old_grace != 0u) {
            grace = distinct_observation ? old_grace - 1u : old_grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                inherited_target = old_cached_target;
            }
        }
    } else if (old_owner_count != 0u) {
        if (old_pending_count != 0u && distinct_observation &&
            StackCompatible(OLD_PENDING_BASE, old_pending_count, STACK_BASE, stack_count)) {
            CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(PreviousState[3u]);
            new_owner = true;
            event = EVENT_HANDOFF;
            if (old_target_valid) {
                inherit_target = true;
                inherited_target = old_target;
            }
        } else {
            uint matched = MatchCurrentToOwner(stack_count, old_owner_count);
            if (matched == stack_count && stack_count <= old_owner_count) {
                CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
                CopyStackCoversToCurrent(stack_count);
                new_owner_count = stack_count;
                authority_count = stack_count;
                owner_generation = PreviousState[3u];
                continuing_owner = true;
            } else {
                CopyRects(NEW_OWNER_BASE, OLD_OWNER_BASE, old_owner_count);
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                CopyMatchedCoversToCurrent(matched);
                new_owner_count = old_owner_count;
                new_pending_count = stack_count;
                authority_count = matched;
                owner_generation = PreviousState[3u];
                continuing_owner = true;
            }
        }
    } else {
        bool confirmed = old_pending_count != 0u && distinct_observation &&
            StackCompatible(OLD_PENDING_BASE, old_pending_count, STACK_BASE, stack_count);
        if (confirmed) {
            CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
            CopyStackCoversToCurrent(stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(0u);
            new_owner = true;
            event = EVENT_BIRTH;
            if (old_grace != 0u) {
                inherit_target = true;
                inherited_target = old_cached_target;
            }
        } else {
            CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
            new_pending_count = stack_count;
            grace = old_grace;
            if (grace != 0u && distinct_observation) --grace;
            if (grace != 0u) {
                grace_bounds = old_grace_bounds;
                inherited_target = old_cached_target;
            }
        }
    }

    bool target_valid = false;
    bool target_reset = false;
    float target = 0.0f;
    uint fade_step = 0u;
    uint unreliable_holds = 0u;
    if (new_owner_count != 0u) {
        // Every distinct authoritative owner observation samples the same local supporting plane.
        // Reliable samples are not moved toward an absolute screen plane; an established target
        // moves only through deadbanded EMA plus a binocular source-pixel slew bound. Exact
        // redispatches do not sample and retain the previous target/fade bits.
        bool retain_duplicate = old_target_valid && !distinct_observation && continuing_owner;
        bool retain_failed_duplicate = old_valid &&
            (PreviousState[2u] & FLAG_TARGET_RESET) != 0u && !distinct_observation &&
            continuing_owner;
        bool retain_without_current_authority = old_target_valid && continuing_owner &&
            authority_count == 0u;
        if (retain_failed_duplicate) {
            // The same observation that failed local-plane qualification cannot acquire authority
            // by being redispatched; only a distinct reliable observation may restart at fade 1.
        } else if (retain_duplicate || retain_without_current_authority) {
            target = old_target;
            target_valid = SubtitleTargetIsValid(target);
            fade_step = PreviousState[24u];
            unreliable_holds = old_unreliable_holds;
        } else {
            float desired = 0.0f;
            uint sample_base = continuing_owner ? MATCHED_BASE : NEW_OWNER_BASE;
            uint sample_count = continuing_owner ? authority_count : new_owner_count;
            bool sampled = SampleOwnerTarget(sample_base, sample_count, desired);
            if (sampled && cut_survivor) {
                // A scene boundary invalidates the meaning of the old supporting plane. Restart
                // from same-frame reliable evidence at half strength and never inherit the old
                // scene target.
                target = desired;
                target_valid = true;
                fade_step = 1u;
            } else if (sampled && new_owner && inherit_target) {
                precise float residual_pixels =
                    abs(desired - inherited_target) * (2.0f * (float)locator_source.x);
                if (residual_pixels > V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS) {
                    target = desired;
                    target_valid = true;
                } else {
                    target_valid = UpdateOwnerTarget(inherited_target, desired, target);
                }
                fade_step = 1u;
            } else if (sampled && new_owner) {
                target = desired;
                target_valid = true;
                fade_step = 1u;
            } else if (sampled && continuing_owner && old_target_valid) {
                precise float residual_pixels =
                    abs(desired - old_target) * (2.0f * (float)locator_source.x);
                if (residual_pixels > V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS) {
                    target = desired;
                    target_valid = true;
                    fade_step = 1u;
                } else {
                    target_valid = UpdateOwnerTarget(old_target, desired, target);
                    fade_step = distinct_observation ? min(PreviousState[24u] + 1u, 2u) :
                        PreviousState[24u];
                }
            } else if (sampled) {
                // Fresh birth, or recovery after a target reset, begins at half strength on the
                // reliable current local-plane observation.
                target = desired;
                target_valid = true;
                fade_step = 1u;
            } else if (continuing_owner && old_target_valid && authority_count != 0u &&
                       distinct_observation &&
                       old_unreliable_holds < V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS) {
                // A short same-scene measurement failure must not expose unconditioned glyph
                // geometry immediately. Hold the exact prior target for two distinct current
                // observations; a third failure resets below. Duplicates were retained above.
                target = old_target;
                target_valid = true;
                fade_step = PreviousState[24u];
                unreliable_holds = old_unreliable_holds + 1u;
            }
        }
        if (!target_valid) {
            target = 0.0f;
            target_reset = true;
            authority_count = 0u;
            fade_step = 0u;
        }
    } else if (grace != 0u) {
        target = inherited_target;
    }

    PublishState(
        owner_generation, new_owner_count, new_pending_count, authority_count,
        target, target_valid, target_reset, fade_step,
        new_owner_count != 0u ? unreliable_holds : grace,
        grace_bounds, event, scene_epoch);
}

bool ValidateConditionRectBlock(uint offset, uint count, out uint4 bbox, out uint area) {
    bbox = uint4(0u, 0u, 0u, 0u);
    area = 0u;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = offset + slot * 4u;
        uint4 rectangle = uint4(
            LocatorStateRead[base + 0u], LocatorStateRead[base + 1u],
            LocatorStateRead[base + 2u], LocatorStateRead[base + 3u]);
        if (!ValidateAndAccumulateRoiRectangle(
                rectangle, slot, count, bbox, area)) return false;
        if (slot > 0u && slot < count) {
            uint previous_base = base - 4u;
            uint4 previous = uint4(
                LocatorStateRead[previous_base + 0u], LocatorStateRead[previous_base + 1u],
                LocatorStateRead[previous_base + 2u], LocatorStateRead[previous_base + 3u]);
            if (!CanonicalCoreOrder(previous, rectangle)) return false;
        }
    }
    return true;
}

bool ValidateConditionCurrentBlock(uint count) {
    uint kinds = (LocatorStateRead[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
                  V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
        uint4 rectangle = uint4(
            LocatorStateRead[base + 0u], LocatorStateRead[base + 1u],
            LocatorStateRead[base + 2u], LocatorStateRead[base + 3u]);
        if (!ValidateCurrentRectangle(rectangle, slot, count, kinds)) return false;
    }
    return true;
}

bool ConditionStateValid(out uint current_count, out float target, out uint fade_step) {
    current_count = LocatorStateRead[20u];
    target = asfloat(LocatorStateRead[18u]);
    fade_step = LocatorStateRead[24u];
    uint flags = LocatorStateRead[2u];
    uint owner_count = LocatorStateRead[4u];
    uint pending_count = LocatorStateRead[12u];
    uint packed_kinds = LocatorStateRead[V2_SUBTITLE_LOCATOR_KIND_WORD];
    uint known_kind_bits =
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT) |
        (V2_SUBTITLE_LOCATOR_KIND_MASK << V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT);
    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint scene_epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    if (!LocatorGeometryValid() ||
        !cut_valid || LocatorStateRead[26u] != scene_epoch ||
        LocatorStateRead[0u] != V2_SUBTITLE_LOCATOR_STATE_SCHEMA ||
        LocatorStateRead[1u] != V2_SUBTITLE_LOCATOR_STATE_TAG ||
        (flags & ~KNOWN_FLAGS) != 0u || (flags & FLAG_OWNER) == 0u ||
        (flags & FLAG_TARGET_VALID) == 0u || (flags & FLAG_TARGET_RESET) != 0u ||
        ((flags & FLAG_PENDING) != 0u) != (pending_count != 0u) ||
        owner_count == 0u || owner_count > MAX_LINES || current_count == 0u ||
        current_count > owner_count || pending_count > MAX_LINES ||
        LocatorStateRead[3u] == 0u || LocatorStateRead[19u] != LocatorStateRead[3u] ||
        LocatorStateRead[10u] != locator_frame.z || LocatorStateRead[11u] != locator_frame.w ||
        LocatorStateRead[22u] != locator_frame.x || LocatorStateRead[23u] != locator_frame.y ||
        LocatorStateRead[21u] > EVENT_HANDOFF ||
        (fade_step != 1u && fade_step != 2u) ||
        LocatorStateRead[25u] > V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS ||
        (LocatorStateRead[25u] != 0u && LocatorStateRead[21u] != EVENT_NONE) ||
        LocatorStateRead[27u] != locator_field.x || LocatorStateRead[28u] != locator_field.y ||
        LocatorStateRead[29u] != 0u || LocatorStateRead[30u] != 0u ||
        (packed_kinds & ~known_kind_bits) != 0u ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT, owner_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT, pending_count) ||
        !PackedKindsValid(packed_kinds, V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT, current_count) ||
        !SubtitleTargetIsValid(target) ||
        !FiniteFloat(v2_max_horizontal_slope) || v2_max_horizontal_slope < 0.0f ||
        !FiniteFloat(v2_max_vertical_shear) || v2_max_vertical_shear < 0.0f) {
        return false;
    }
    uint4 owner_bbox;
    uint owner_area;
    uint4 pending_bbox;
    uint pending_area;
    if (!ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, owner_count, owner_bbox, owner_area) ||
        !ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, pending_count, pending_bbox, pending_area) ||
        !ValidateConditionCurrentBlock(current_count) ||
        any(owner_bbox != uint4(
            LocatorStateRead[5u], LocatorStateRead[6u],
            LocatorStateRead[7u], LocatorStateRead[8u])) || owner_area != LocatorStateRead[9u] ||
        any(pending_bbox != uint4(
            LocatorStateRead[13u], LocatorStateRead[14u],
            LocatorStateRead[15u], LocatorStateRead[16u])) || pending_area != LocatorStateRead[17u]) {
        return false;
    }
    return true;
}

bool ConditionContentValid() {
    return target_w != 0u && target_h != 0u &&
        target_w == locator_field.x && target_h == locator_field.y &&
        locator_content.x < locator_content.z && locator_content.y < locator_content.w &&
        locator_content.z <= target_w && locator_content.w <= target_h &&
        all(locator_content == DepthAnalysisContentCells());
}

uint ConservativeConditionPad(
    float max_delta,
    float core_range,
    float slope,
    uint content_width,
    uint axis_extent) {
    if (axis_extent == 0u || max_delta <= core_range) return 0u;
    precise float step = slope / (float)content_width;
    if (!FiniteFloat(step) || step <= 0.0f) return axis_extent;
    precise float raw_pad = (max_delta - core_range) / step;
    // Precheck against the bounded field extent before converting float to uint. One additional
    // cell plus the excluded-cell proof below makes float rounding conservative, never narrow.
    if (!FiniteFloat(raw_pad) || raw_pad >= (float)axis_extent) return axis_extent;
    uint pad = min(axis_extent, (uint)ceil(raw_pad) + 1u);
    precise float first_excluded_budget =
        core_range + (float)(pad + 1u) * step;
    return FiniteFloat(first_excluded_budget) && first_excluded_budget >= max_delta ?
        pad : axis_extent;
}

bool PrepareFullContentActiveRegion(
    uint current_count,
    uint current_kinds,
    float condition_target,
    out uint2 dispatch_origin,
    out uint3 dispatch_groups) {
    dispatch_origin = uint2(0u, 0u);
    dispatch_groups = uint3(0u, 0u, 0u);
    precise float max_delta = v2_direct_container_limit + abs(condition_target);
    precise float core_range = 0.5f / (float)locator_source.x;
    if (!FiniteFloat(max_delta) || !FiniteFloat(core_range)) {
        dispatch_groups = uint3((target_w + 15u) / 16u, (target_h + 15u) / 16u, 1u);
        return true;
    }
    if (max_delta <= core_range) {
        return false;
    }
    uint content_width = LocatorContentWidth();
    uint content_height = locator_content.w - locator_content.y;
    uint horizontal_pad = ConservativeConditionPad(
        max_delta, core_range, v2_max_horizontal_slope,
        content_width, content_width);
    uint vertical_pad = ConservativeConditionPad(
        max_delta, core_range, v2_max_vertical_shear,
        content_width, content_height);
    uint4 bounds = uint4(
        locator_content.z, locator_content.w, locator_content.x, locator_content.y);
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < current_count) {
            uint offset = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
            uint4 rectangle = uint4(
                LocatorStateRead[offset + 0u], LocatorStateRead[offset + 1u],
                LocatorStateRead[offset + 2u], LocatorStateRead[offset + 3u]);
            bool ribbon = ((current_kinds >> slot) & 1u) != 0u;
            uint4 expanded;
            if (ribbon) {
                expanded = uint4(
                    locator_content.x,
                    rectangle.y - min(vertical_pad, rectangle.y - locator_content.y),
                    locator_content.z,
                    locator_content.w);
            } else {
                expanded = uint4(
                    rectangle.x - min(horizontal_pad, rectangle.x - locator_content.x),
                    rectangle.y - min(vertical_pad, rectangle.y - locator_content.y),
                    rectangle.z + min(horizontal_pad, locator_content.z - rectangle.z),
                    rectangle.w + min(vertical_pad, locator_content.w - rectangle.w));
            }
            bounds.xy = min(bounds.xy, expanded.xy);
            bounds.zw = max(bounds.zw, expanded.zw);
        }
    }
    if (bounds.x >= bounds.z || bounds.y >= bounds.w) return false;
    dispatch_origin = (bounds.xy / 16u) * 16u;
    uint2 dispatch_extent = bounds.zw - dispatch_origin;
    dispatch_groups = uint3(
        (dispatch_extent.x + 15u) / 16u,
        (dispatch_extent.y + 15u) / 16u,
        1u);
    return dispatch_groups.x != 0u && dispatch_groups.y != 0u;
}

// Authenticate the state once per field, then publish the tiny immutable parameter closure used
// by both condition writers. Full-content indirect arguments are zero without current authority;
// otherwise they cover only the conservatively expanded union that could change. Padded ROI fields
// still publish full-field arguments so synthetic cells can extend the nearest real content cell.
[numthreads(1, 1, 1)]
void condition_prepare_main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint current_count;
    float target;
    uint fade_step;
    bool state_valid = ConditionStateValid(current_count, target, fade_step);
    uint current_kinds = state_valid ?
        ((LocatorStateRead[V2_SUBTITLE_LOCATOR_KIND_WORD] >>
          V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT) & V2_SUBTITLE_LOCATOR_KIND_MASK) : 0u;

    bool content_valid = ConditionContentValid();
    bool has_padding = content_valid && any(locator_content != uint4(0u, 0u, target_w, target_h));
    uint2 dispatch_origin = uint2(0u, 0u);
    uint3 dispatch_groups = uint3(0u, 0u, 0u);
    if (has_padding) {
        dispatch_groups = uint3((target_w + 15u) / 16u, (target_h + 15u) / 16u, 1u);
    } else if (content_valid && state_valid) {
        PrepareFullContentActiveRegion(
            current_count, current_kinds, target, dispatch_origin, dispatch_groups);
    }

    ConditionParamsOut[CONDITION_PARAM_SCHEMA_WORD] =
        state_valid ? V2_SUBTITLE_CONDITION_PARAM_SCHEMA : 0u;
    ConditionParamsOut[CONDITION_PARAM_TAG_WORD] =
        state_valid ? V2_SUBTITLE_CONDITION_PARAM_TAG : 0u;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_COUNT_WORD] =
        state_valid ? current_count : 0u;
    ConditionParamsOut[CONDITION_PARAM_CURRENT_KINDS_WORD] = current_kinds;
    ConditionParamsOut[CONDITION_PARAM_FADE_STEP_WORD] = state_valid ? fade_step : 0u;
    ConditionParamsOut[CONDITION_PARAM_TARGET_WORD] = state_valid ? asuint(target) : 0u;
    ConditionParamsOut[CONDITION_PARAM_ORIGIN_X_WORD] = dispatch_origin.x;
    ConditionParamsOut[CONDITION_PARAM_ORIGIN_Y_WORD] = dispatch_origin.y;
    ConditionDispatchArgs[0u] = dispatch_groups.x;
    ConditionDispatchArgs[1u] = dispatch_groups.y;
    ConditionDispatchArgs[2u] = dispatch_groups.z;
}

bool ConditionParamsValid(
    out uint current_count,
    out uint current_kinds,
    out uint fade_step,
    out float target,
    out uint2 dispatch_origin) {
    current_count = ConditionParams[CONDITION_PARAM_CURRENT_COUNT_WORD];
    current_kinds = ConditionParams[CONDITION_PARAM_CURRENT_KINDS_WORD];
    fade_step = ConditionParams[CONDITION_PARAM_FADE_STEP_WORD];
    target = asfloat(ConditionParams[CONDITION_PARAM_TARGET_WORD]);
    dispatch_origin = uint2(
        ConditionParams[CONDITION_PARAM_ORIGIN_X_WORD],
        ConditionParams[CONDITION_PARAM_ORIGIN_Y_WORD]);
    return ConditionParams[CONDITION_PARAM_SCHEMA_WORD] == V2_SUBTITLE_CONDITION_PARAM_SCHEMA &&
        ConditionParams[CONDITION_PARAM_TAG_WORD] == V2_SUBTITLE_CONDITION_PARAM_TAG &&
        current_count != 0u && current_count <= MAX_LINES &&
        (current_kinds & ~V2_SUBTITLE_LOCATOR_KIND_MASK) == 0u &&
        PackedKindsValid(current_kinds, 0u, current_count) &&
        (fade_step == 1u || fade_step == 2u) && SubtitleTargetIsValid(target) &&
        dispatch_origin.x < target_w && dispatch_origin.y < target_h &&
        (dispatch_origin.x & 15u) == 0u && (dispatch_origin.y & 15u) == 0u;
}

float EvaluateConditionedBase(
    uint2 condition_position,
    float base,
    bool state_valid,
    uint current_count,
    uint current_kinds,
    uint fade_step,
    float condition_target,
    out bool changed) {
    changed = false;
    if (!state_valid || !FiniteFloat(base) || abs(base) > v2_direct_container_limit) {
        return base;
    }
    float best_distance = 3.402823466e+38f;
    precise float horizontal_step = v2_max_horizontal_slope /
        (float)LocatorContentWidth();
    precise float vertical_step = v2_max_vertical_shear /
        (float)LocatorContentWidth();
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < current_count) {
            uint offset = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
            uint4 rectangle = uint4(
                LocatorStateRead[offset + 0u], LocatorStateRead[offset + 1u],
                LocatorStateRead[offset + 2u], LocatorStateRead[offset + 3u]);
            bool ribbon = ((current_kinds >> slot) & 1u) != 0u;
            // A canonical ribbon cover spans the full field from its corrected top to the bottom.
            // Make the one-edge policy explicit: the strip and everything below its top use core
            // budget, while only rows above it receive the vertical analytic collar. Ordinary
            // subtitle covers retain the established four-sided Manhattan collar.
            uint dx = ribbon ? 0u :
                (condition_position.x < rectangle.x ? rectangle.x - condition_position.x :
                 (condition_position.x >= rectangle.z ?
                    condition_position.x - (rectangle.z - 1u) : 0u));
            uint dy = ribbon ?
                (condition_position.y < rectangle.y ? rectangle.y - condition_position.y : 0u) :
                (condition_position.y < rectangle.y ? rectangle.y - condition_position.y :
                 (condition_position.y >= rectangle.w ?
                    condition_position.y - (rectangle.w - 1u) : 0u));
            precise float horizontal_distance = (float)dx * horizontal_step;
            precise float vertical_distance = (float)dy * vertical_step;
            precise float distance = horizontal_distance + vertical_distance;
            if (distance < best_distance) {
                best_distance = distance;
            }
        }
    }
    precise float core_range = 0.5f / (float)locator_source.x;
    precise float budget = core_range + best_distance;
    precise float delta = base - condition_target;
    // Exact Base is a semantic branch, not an algebraic coincidence: bypassing reconstruction
    // avoids changing an already-safe R32_FLOAT bit pattern through target + (base - target).
    if (abs(delta) <= budget) {
        return base;
    }
    precise float full = condition_target + (delta < 0.0f ? -budget : budget);
    precise float faded = base + 0.5f * (full - base);
    changed = true;
    return fade_step == 1u ? faded : full;
}

// Complete out-of-place writer used when Dump 3D must preserve BaseField or tensor padding must
// extend the nearest real content cell. Every output cell is written, so no preparatory texture
// copy is required and an invalid condition verdict still publishes Base exactly.
[numthreads(16, 16, 1)]
void condition_main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x >= target_w || dispatch_id.y >= target_h) return;
    bool content_valid = ConditionContentValid();
    uint2 condition_position = content_valid ?
        DepthAnalysisClampCell(dispatch_id.xy) : dispatch_id.xy;
    uint current_count;
    uint current_kinds;
    uint fade_step;
    float condition_target;
    uint2 dispatch_origin;
    bool state_valid = ConditionParamsValid(
        current_count, current_kinds, fade_step, condition_target, dispatch_origin);
    float base = BaseField.Load(int3(condition_position, 0));
    bool changed;
    float conditioned = EvaluateConditionedBase(
        condition_position, base, state_valid, current_count, current_kinds,
        fade_step, condition_target, changed);
    ConditionedField[dispatch_id.xy] = conditioned;
}

// Ordinary full-content live production aliases Base and output intentionally. There is no t2 SRV
// binding in this entrypoint: each bounded invocation adds the authenticated group-aligned origin,
// then reads and conditionally replaces its u3 cell. This avoids a D3D11 SRV/UAV alias and all
// texture work when preparation emits zero groups.
[numthreads(16, 16, 1)]
void condition_in_place_main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint current_count;
    uint current_kinds;
    uint fade_step;
    float condition_target;
    uint2 dispatch_origin;
    bool state_valid = ConditionParamsValid(
        current_count, current_kinds, fade_step, condition_target, dispatch_origin);
    if (!state_valid || !ConditionContentValid() ||
        any(locator_content != uint4(0u, 0u, target_w, target_h)) ||
        dispatch_id.x >= target_w - dispatch_origin.x ||
        dispatch_id.y >= target_h - dispatch_origin.y) return;
    uint2 condition_position = dispatch_id.xy + dispatch_origin;
    float base = ConditionedField[condition_position];
    bool changed;
    float conditioned = EvaluateConditionedBase(
        condition_position, base, true, current_count, current_kinds,
        fade_step, condition_target, changed);
    if (changed) ConditionedField[condition_position] = conditioned;
}

// Compact SLR6 subtitle authority.
//
// OCR6 is the sole geometry source.  A coherent line stack needs two distinct, exact-frame
// observations before it becomes an owner.  Only rectangles copied from the current OCR6 record
// may condition the current BaseField; cached owner, pending, target, and death-grace state never
// manufacture current geometry.  The conditioner preserves each line rectangle and its gaps and
// evaluates the V2-safe collar analytically, so no row/history/distance resources exist.

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"
#include "include/sbs_adaptive_state_contract.generated.hlsl"

// OCR6/SLR6 is an authenticated generated ABI.  Never reconstruct it locally when a stale
// generated header is present: missing identity/layout macros are a compile-time failure.
#if !defined(V2_OCR_RECORD_SCHEMA) || !defined(V2_OCR_RECORD_TAG) || \
    !defined(V2_OCR_RECORD_WORD_COUNT) || !defined(V2_OCR_RECORD_HEADER_WORD_COUNT) || \
    !defined(V2_OCR_BOX_WORD_COUNT) || !defined(V2_OCR_RAW_BOX_OFFSET) || \
    !defined(V2_OCR_RAW_BOX_CAPACITY) || !defined(V2_OCR_FINAL_BOX_OFFSET) || \
    !defined(V2_OCR_FINAL_BOX_CAPACITY) || !defined(V2_OCR_FIELD_WIDTH) || \
    !defined(V2_OCR_FIELD_HEIGHT) || !defined(V2_OCR_ROI_TOP) || \
    !defined(V2_OCR_ROI_BOTTOM) || !defined(V2_SUBTITLE_LOCATOR_STATE_SCHEMA) || \
    !defined(V2_SUBTITLE_LOCATOR_STATE_TAG) || \
    !defined(V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT) || \
    !defined(V2_SUBTITLE_LOCATOR_HEADER_WORD_COUNT) || \
    !defined(V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY) || \
    !defined(V2_SUBTITLE_LOCATOR_OWNER_OFFSET) || \
    !defined(V2_SUBTITLE_LOCATOR_PENDING_OFFSET) || \
    !defined(V2_SUBTITLE_LOCATOR_CURRENT_OFFSET)
#error "Generated V2 OCR6/SLR6 contract macros are required"
#endif

StructuredBuffer<float4> CutBridge : register(t1);
Texture2D<float> BaseField : register(t2);
StructuredBuffer<uint> LocatorStateRead : register(t3);
StructuredBuffer<uint> OcrRecord : register(t7);

RWStructuredBuffer<uint> LocatorState : register(u2);
RWTexture2D<float> ConditionedField : register(u3);

cbuffer SubtitleLocatorConstants : register(b2) {
    uint4 locator_field;   // width, height, ROI top, ROI bottom
    uint4 locator_source;  // analysis source width, height, enabled, input-domain reset
    uint4 locator_frame;   // matched frame lo/hi, analysis generation lo/hi
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
static const uint DEATH_GRACE_OBSERVATIONS = 6u;

groupshared uint PreviousState[V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT];
groupshared uint4 QualifiedBoxes[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint QualifiedMasks[V2_OCR_FINAL_BOX_CAPACITY];
groupshared uint4 WorkRects[28];
groupshared float TargetSamples[32];
groupshared float LineCenters[MAX_LINES];

bool FiniteFloat(float value) {
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

bool ZeroRect(uint4 rectangle) {
    return all(rectangle == uint4(0u, 0u, 0u, 0u));
}

bool ValidRoiRect(uint4 rectangle) {
    return rectangle.x < rectangle.z && rectangle.y < rectangle.w &&
        rectangle.z <= locator_field.x && rectangle.w <= locator_field.y &&
        rectangle.y >= locator_field.z && rectangle.w <= locator_field.w;
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

bool RectanglesOverlap(uint4 a, uint4 b) {
    return min(a.z, b.z) > max(a.x, b.x) && min(a.w, b.w) > max(a.y, b.y);
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

bool ValidOcrBox(uint offset) {
    uint4 rectangle = uint4(
        OcrRecord[offset + 0u], OcrRecord[offset + 1u],
        OcrRecord[offset + 2u], OcrRecord[offset + 3u]);
    float score = asfloat(OcrRecord[offset + 4u]);
    return ValidRoiRect(rectangle) && FiniteFloat(score) && score >= 0.4f && score <= 1.0f &&
        OcrRecord[offset + 5u] == 0u && OcrRecord[offset + 6u] == 0u &&
        OcrRecord[offset + 7u] == 0u;
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
    if (locator_field.x != V2_OCR_FIELD_WIDTH || locator_field.y != V2_OCR_FIELD_HEIGHT ||
        locator_field.z != V2_OCR_ROI_TOP || locator_field.w != V2_OCR_ROI_BOTTOM ||
        target_w != locator_field.x || target_h != locator_field.y ||
        locator_source.x == 0u || locator_source.y == 0u || locator_source.z != 1u) {
        return false;
    }
    uint raw_count = OcrRecord[3u];
    final_count = OcrRecord[4u];
    if (OcrRecord[0u] != V2_OCR_RECORD_SCHEMA || OcrRecord[1u] != V2_OCR_RECORD_TAG ||
        OcrRecord[2u] != 1u || raw_count > V2_OCR_RAW_BOX_CAPACITY ||
        final_count > V2_OCR_FINAL_BOX_CAPACITY || final_count > raw_count ||
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
        if (slot < raw_count ? !ValidOcrBox(offset) : !ZeroOcrBox(offset)) return false;
    }
    [loop]
    for (uint final_slot = 0u; final_slot < V2_OCR_FINAL_BOX_CAPACITY; ++final_slot) {
        uint offset = V2_OCR_FINAL_BOX_OFFSET + final_slot * V2_OCR_BOX_WORD_COUNT;
        if (final_slot < final_count ? !ValidOcrBox(offset) : !ZeroOcrBox(offset)) return false;
    }
    return true;
}

uint BuildCurrentStack(uint final_count) {
    uint qualified_count = 0u;
    [loop]
    for (uint slot = 0u; slot < V2_OCR_FINAL_BOX_CAPACITY; ++slot) {
        if (slot < final_count) {
            uint offset = V2_OCR_FINAL_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT;
            uint4 rectangle = uint4(
                OcrRecord[offset + 0u], OcrRecord[offset + 1u],
                OcrRecord[offset + 2u], OcrRecord[offset + 3u]);
            uint width = rectangle.z - rectangle.x;
            uint height = rectangle.w - rectangle.y;
            // Generic subtitle-line geometry.  In particular, square badges/logos fail the
            // aspect gate without a position- or brand-specific exclusion.
            if (width >= 48u && width <= 693u && height >= 6u && width >= 2u * height) {
                QualifiedBoxes[qualified_count++] = rectangle;
            }
        }
    }
    if (qualified_count == 0u) return 0u;

    [loop]
    for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
        QualifiedMasks[index] = index < qualified_count ? (1u << index) : 0u;
    }
    [loop]
    for (uint closure_pass = 0u; closure_pass < V2_OCR_FINAL_BOX_CAPACITY; ++closure_pass) {
        [loop]
        for (uint a = 0u; a < V2_OCR_FINAL_BOX_CAPACITY; ++a) {
            if (a < qualified_count) {
                uint expanded = QualifiedMasks[a];
                [loop]
                for (uint b = 0u; b < V2_OCR_FINAL_BOX_CAPACITY; ++b) {
                    if (b < qualified_count && (expanded & (1u << b)) != 0u) {
                        [loop]
                        for (uint c = 0u; c < V2_OCR_FINAL_BOX_CAPACITY; ++c) {
                            if (c < qualified_count && CoherentLines(QualifiedBoxes[b], QualifiedBoxes[c])) {
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
        if (root < qualified_count) {
            uint mask = QualifiedMasks[root];
            // Evaluate each connected component once and reject unsupported >4-line blocks.
            if ((mask & ((1u << root) - 1u)) == 0u) {
                uint count = 0u;
                uint area = 0u;
                uint4 bbox = uint4(0xffffffffu, 0xffffffffu, 0u, 0u);
                [loop]
                for (uint index = 0u; index < V2_OCR_FINAL_BOX_CAPACITY; ++index) {
                    if (index < qualified_count && (mask & (1u << index)) != 0u) {
                        uint4 rectangle = QualifiedBoxes[index];
                        ++count;
                        area += RectArea(rectangle);
                        bbox.x = min(bbox.x, rectangle.x);
                        bbox.y = min(bbox.y, rectangle.y);
                        bbox.z = max(bbox.z, rectangle.z);
                        bbox.w = max(bbox.w, rectangle.w);
                    }
                }
                bool better = count <= MAX_LINES && (best_mask == 0u || area > best_area ||
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
    if (best_mask == 0u) return 0u;

    uint stack_count = 0u;
    [loop]
    for (uint collect_index = 0u; collect_index < V2_OCR_FINAL_BOX_CAPACITY; ++collect_index) {
        if (collect_index < qualified_count && (best_mask & (1u << collect_index)) != 0u) {
            WorkRects[STACK_BASE + stack_count++] = QualifiedBoxes[collect_index];
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
                }
            }
        }
    }
    return stack_count;
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
        if (slot < count) {
            if (!ValidRoiRect(rectangle)) return false;
            if (slot == 0u) bbox = rectangle;
            else {
                bbox.x = min(bbox.x, rectangle.x);
                bbox.y = min(bbox.y, rectangle.y);
                bbox.z = max(bbox.z, rectangle.z);
                bbox.w = max(bbox.w, rectangle.w);
            }
            area += RectArea(rectangle);
        } else if (!ZeroRect(rectangle)) {
            return false;
        }
    }
    return all(bbox == expected_bbox) && area == expected_area;
}

bool ValidatePreviousCurrent(uint count) {
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint base = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
        uint4 rectangle = uint4(
            PreviousState[base + 0u], PreviousState[base + 1u],
            PreviousState[base + 2u], PreviousState[base + 3u]);
        if (slot < count ? !ValidRoiRect(rectangle) : !ZeroRect(rectangle)) return false;
    }
    return true;
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
    if (PreviousState[0u] != V2_SUBTITLE_LOCATOR_STATE_SCHEMA ||
        PreviousState[1u] != V2_SUBTITLE_LOCATOR_STATE_TAG || (flags & ~KNOWN_FLAGS) != 0u ||
        owner_count > MAX_LINES || pending_count > MAX_LINES || current_count > MAX_LINES ||
        PreviousState[21u] > EVENT_HANDOFF || PreviousState[24u] > 2u ||
        PreviousState[27u] != locator_field.x || PreviousState[28u] != locator_field.y ||
        PreviousState[31u] != 0u || owner != (owner_count != 0u) ||
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
    uint grace = PreviousState[25u];
    if (owner) {
        if (grace != 0u || PreviousState[29u] != 0u || PreviousState[30u] != 0u) return false;
        if (target_valid) {
            if (target_reset || PreviousState[19u] != PreviousState[3u] ||
                !FiniteFloat(target) || abs(target) > v2_direct_container_limit ||
                PreviousState[24u] == 0u) return false;
        } else if (target_reset) {
            if (PreviousState[18u] != 0u || PreviousState[19u] != 0u || current_count != 0u ||
                PreviousState[24u] != 0u) return false;
        } else if (PreviousState[18u] != 0u || PreviousState[19u] != 0u || current_count != 0u ||
                   PreviousState[24u] != 0u) return false;
    } else if (grace == 0u) {
        if (target_valid || target_reset || PreviousState[18u] != 0u ||
            PreviousState[19u] != 0u || PreviousState[29u] != 0u ||
            PreviousState[30u] != 0u || current_count != 0u || PreviousState[24u] != 0u) {
            return false;
        }
    } else {
        uint4 bounds = uint4(
            PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
            PreviousState[29u] >> 16u, PreviousState[30u] >> 16u);
        if (target_valid || target_reset || PreviousState[19u] != 0u ||
            !FiniteFloat(target) || abs(target) > v2_direct_container_limit ||
            !ValidRoiRect(bounds) || current_count != 0u || PreviousState[24u] != 0u) return false;
    }
    return true;
}

void LoadPreviousRects(uint state_offset, uint work_base, uint count) {
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        uint offset = state_offset + slot * 4u;
        WorkRects[work_base + slot] = slot < count ? uint4(
            PreviousState[offset + 0u], PreviousState[offset + 1u],
            PreviousState[offset + 2u], PreviousState[offset + 3u]) :
            uint4(0u, 0u, 0u, 0u);
    }
}

bool StackCompatible(uint first_base, uint first_count, uint second_base, uint second_count) {
    if (first_count == 0u || first_count != second_count) return false;
    float sum = 0.0f;
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index < first_count) sum += RectIou(WorkRects[first_base + index], WorkRects[second_base + index]);
    }
    return sum >= 0.6f * (float)first_count;
}

uint MatchCurrentToOwner(uint current_count, uint owner_count) {
    uint used = 0u;
    uint matched = 0u;
    [unroll]
    for (uint current_index = 0u; current_index < MAX_LINES; ++current_index) {
        if (current_index < current_count) {
            float best_iou = 0.6f;
            uint best_owner = MAX_LINES;
            [unroll]
            for (uint owner_index = 0u; owner_index < MAX_LINES; ++owner_index) {
                if (owner_index < owner_count && (used & (1u << owner_index)) == 0u) {
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
                WorkRects[MATCHED_BASE + matched++] = WorkRects[STACK_BASE + current_index];
            }
        }
    }
    [unroll]
    for (uint index = 0u; index < MAX_LINES; ++index) {
        if (index >= matched) WorkRects[MATCHED_BASE + index] = uint4(0u, 0u, 0u, 0u);
    }
    return matched;
}

uint NextGeneration(uint value) {
    return value == 0u || value >= 0xfffffffdu ? 1u : value + 1u;
}

bool SampleOwnerTarget(uint owner_count, out float target) {
    target = 0.0f;
    if (owner_count == 0u || locator_source.x == 0u ||
        !FiniteFloat(v2_direct_container_limit) || v2_direct_container_limit <= 0.0f) return false;
    [unroll]
    for (uint center_index = 0u; center_index < MAX_LINES; ++center_index) {
        LineCenters[center_index] = center_index < owner_count ?
            0.5f * (float)(WorkRects[NEW_OWNER_BASE + center_index].x +
                           WorkRects[NEW_OWNER_BASE + center_index].z - 1u) : 0.0f;
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
    uint owner_top = RectSummary(NEW_OWNER_BASE, owner_count).y;
    uint sample_y0 = owner_top >= 10u ? owner_top - 10u : 0u;
    uint sample_y1 = owner_top >= 4u ? owner_top - 4u : 0u;
    [loop]
    for (uint sample_index = 0u; sample_index < 16u; ++sample_index) {
        float sample_x_float = center - 30.0f + 4.0f * (float)sample_index;
        uint sample_x = (uint)clamp(floor(sample_x_float + 0.5f), 0.0f, (float)(locator_field.x - 1u));
        float first = BaseField.Load(int3(sample_x, sample_y0, 0));
        float second = BaseField.Load(int3(sample_x, sample_y1, 0));
        if (!FiniteFloat(first) || !FiniteFloat(second) ||
            abs(first) > v2_direct_container_limit || abs(second) > v2_direct_container_limit) {
            return false;
        }
        TargetSamples[sample_index] = first;
        TargetSamples[16u + sample_index] = second;
    }
    [loop]
    for (uint sample_sort_outer = 0u; sample_sort_outer < 32u; ++sample_sort_outer) {
        uint minimum_index = sample_sort_outer;
        [loop]
        for (uint sample_sort_scan = 0u; sample_sort_scan < 32u; ++sample_sort_scan) {
            if (sample_sort_scan > sample_sort_outer &&
                TargetSamples[sample_sort_scan] < TargetSamples[minimum_index]) {
                minimum_index = sample_sort_scan;
            }
        }
        float swap_value = TargetSamples[sample_sort_outer];
        TargetSamples[sample_sort_outer] = TargetSamples[minimum_index];
        TargetSamples[minimum_index] = swap_value;
    }
    target = 0.5f * (TargetSamples[15u] + TargetSamples[16u]);
    return FiniteFloat(target) && abs(target) <= v2_direct_container_limit;
}

float UpdateTarget(float previous, float observation) {
    float requested = 0.15f * (observation - previous);
    float max_step = 0.25f / (float)locator_source.x;
    return previous + clamp(requested, -max_step, max_step);
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

void PublishState(
    uint owner_generation, uint owner_count, uint pending_count, uint current_count,
    float target, bool target_valid, bool target_reset, uint fade_step, uint grace,
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
    LocatorState[25u] = owner_count == 0u ? grace : 0u;
    LocatorState[26u] = scene_epoch;
    LocatorState[27u] = locator_field.x;
    LocatorState[28u] = locator_field.y;
    LocatorState[29u] = owner_count == 0u && grace != 0u ?
        (grace_bounds.z << 16u) | grace_bounds.x : 0u;
    LocatorState[30u] = owner_count == 0u && grace != 0u ?
        (grace_bounds.w << 16u) | grace_bounds.y : 0u;
    LocatorState[31u] = 0u;
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
    }

    bool cut_valid = asuint(SBS_STATE_CUT_CONTRACT_TAG_BITS(
        CutBridge[SBS_STATE_VECTOR_CUT_CONTRACT_TAG_BITS])) == SBS_CUT_CONTRACT_TAG;
    uint scene_epoch = cut_valid ? asuint(SBS_STATE_HARD_CUT_COUNT(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_COUNT])) : 0u;
    uint final_count = 0u;
    bool ocr_valid = ValidateOcrRecord(final_count);
    if (!cut_valid || !ocr_valid) {
        PublishState(0u, 0u, 0u, 0u, 0.0f, false, false, 0u, 0u,
                     uint4(0u, 0u, 0u, 0u), EVENT_NONE, scene_epoch);
        return;
    }
    uint stack_count = BuildCurrentStack(final_count);
    bool old_valid = ValidatePreviousState();
    uint old_owner_count = old_valid ? PreviousState[4u] : 0u;
    uint old_pending_count = old_valid ? PreviousState[12u] : 0u;
    if (old_valid) {
        LoadPreviousRects(V2_SUBTITLE_LOCATOR_OWNER_OFFSET, OLD_OWNER_BASE, old_owner_count);
        LoadPreviousRects(V2_SUBTITLE_LOCATOR_PENDING_OFFSET, OLD_PENDING_BASE, old_pending_count);
    }
    bool distinct_observation = !old_valid || PreviousState[22u] != locator_frame.x ||
        PreviousState[23u] != locator_frame.y || PreviousState[10u] != locator_frame.z ||
        PreviousState[11u] != locator_frame.w;
    bool hard_cut = cut_valid && (SBS_STATE_HARD_CUT_PULSE(
        CutBridge[SBS_STATE_VECTOR_HARD_CUT_PULSE]) > 0.5f ||
        (old_valid && PreviousState[26u] != scene_epoch));

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

    bool old_target_valid = old_valid && (PreviousState[2u] & FLAG_TARGET_VALID) != 0u;
    float old_target = old_target_valid ? asfloat(PreviousState[18u]) : 0.0f;
    uint old_grace = old_valid && old_owner_count == 0u ? PreviousState[25u] : 0u;
    uint4 old_grace_bounds = old_grace != 0u ? uint4(
        PreviousState[29u] & 0xffffu, PreviousState[30u] & 0xffffu,
        PreviousState[29u] >> 16u, PreviousState[30u] >> 16u) :
        uint4(0u, 0u, 0u, 0u);
    float old_cached_target = old_grace != 0u ? asfloat(PreviousState[18u]) : 0.0f;

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
                CopyRects(NEW_CURRENT_BASE, MATCHED_BASE, matched);
                new_owner_count = matched;
                authority_count = matched;
                owner_generation = PreviousState[3u];
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
            CopyRects(NEW_CURRENT_BASE, STACK_BASE, stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(PreviousState[3u]);
            new_owner = true;
            event = EVENT_HANDOFF;
            if (old_target_valid && RectanglesOverlap(
                    RectSummary(OLD_OWNER_BASE, old_owner_count),
                    RectSummary(STACK_BASE, stack_count))) {
                inherit_target = true;
                inherited_target = old_target;
            }
        } else {
            uint matched = MatchCurrentToOwner(stack_count, old_owner_count);
            if (matched == stack_count && stack_count <= old_owner_count) {
                CopyRects(NEW_OWNER_BASE, STACK_BASE, stack_count);
                CopyRects(NEW_CURRENT_BASE, STACK_BASE, stack_count);
                new_owner_count = stack_count;
                authority_count = stack_count;
                owner_generation = PreviousState[3u];
                continuing_owner = true;
            } else {
                CopyRects(NEW_OWNER_BASE, OLD_OWNER_BASE, old_owner_count);
                CopyRects(NEW_PENDING_BASE, STACK_BASE, stack_count);
                CopyRects(NEW_CURRENT_BASE, MATCHED_BASE, matched);
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
            CopyRects(NEW_CURRENT_BASE, STACK_BASE, stack_count);
            new_owner_count = stack_count;
            authority_count = stack_count;
            owner_generation = NextGeneration(0u);
            new_owner = true;
            event = EVENT_BIRTH;
            if (old_grace != 0u && RectanglesOverlap(
                    old_grace_bounds, RectSummary(STACK_BASE, stack_count))) {
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
    if (new_owner_count != 0u) {
        float observation = 0.0f;
        bool retain_duplicate = continuing_owner && !distinct_observation && old_target_valid;
        bool sampled = retain_duplicate || SampleOwnerTarget(new_owner_count, observation);
        if (sampled) {
            if (retain_duplicate) {
                target = old_target;
                fade_step = PreviousState[24u];
            } else if (cut_survivor) {
                target = observation;
                fade_step = 2u;
            } else if (continuing_owner && old_target_valid) {
                target = UpdateTarget(old_target, observation);
                fade_step = distinct_observation ? min(PreviousState[24u] + 1u, 2u) :
                    PreviousState[24u];
            } else if (new_owner && inherit_target) {
                target = UpdateTarget(inherited_target, observation);
                fade_step = 1u;
            } else {
                target = observation;
                fade_step = 1u;
            }
            target_valid = FiniteFloat(target) && abs(target) <= v2_direct_container_limit;
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
        target, target_valid, target_reset, fade_step, grace, grace_bounds, event, scene_epoch);
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
        if (slot < count) {
            if (!ValidRoiRect(rectangle)) return false;
            if (slot == 0u) bbox = rectangle;
            else {
                bbox.x = min(bbox.x, rectangle.x);
                bbox.y = min(bbox.y, rectangle.y);
                bbox.z = max(bbox.z, rectangle.z);
                bbox.w = max(bbox.w, rectangle.w);
            }
            area += RectArea(rectangle);
        } else if (!ZeroRect(rectangle)) return false;
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
    if (locator_source.z != 1u || locator_source.x == 0u ||
        locator_field.x != V2_OCR_FIELD_WIDTH || locator_field.y != V2_OCR_FIELD_HEIGHT ||
        locator_field.z != V2_OCR_ROI_TOP || locator_field.w != V2_OCR_ROI_BOTTOM ||
        target_w != locator_field.x || target_h != locator_field.y ||
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
        (fade_step != 1u && fade_step != 2u) || LocatorStateRead[25u] != 0u ||
        LocatorStateRead[27u] != locator_field.x || LocatorStateRead[28u] != locator_field.y ||
        LocatorStateRead[29u] != 0u || LocatorStateRead[30u] != 0u ||
        LocatorStateRead[31u] != 0u || !FiniteFloat(target) ||
        abs(target) > v2_direct_container_limit ||
        !FiniteFloat(v2_max_horizontal_slope) || v2_max_horizontal_slope < 0.0f ||
        !FiniteFloat(v2_max_vertical_shear) || v2_max_vertical_shear < 0.0f) {
        return false;
    }
    uint4 owner_bbox;
    uint owner_area;
    uint4 pending_bbox;
    uint pending_area;
    uint4 current_bbox;
    uint current_area;
    if (!ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_OWNER_OFFSET, owner_count, owner_bbox, owner_area) ||
        !ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_PENDING_OFFSET, pending_count, pending_bbox, pending_area) ||
        !ValidateConditionRectBlock(
            V2_SUBTITLE_LOCATOR_CURRENT_OFFSET, current_count, current_bbox, current_area) ||
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

[numthreads(16, 16, 1)]
void condition_main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x >= target_w || dispatch_id.y >= target_h) return;
    float base = BaseField.Load(int3(dispatch_id.xy, 0));
    uint current_count;
    float target;
    uint fade_step;
    if (!ConditionStateValid(current_count, target, fade_step) || !FiniteFloat(base) ||
        abs(base) > v2_direct_container_limit) {
        ConditionedField[dispatch_id.xy] = base;
        return;
    }

    uint best_dx = 0xffffffffu;
    uint best_dy = 0xffffffffu;
    float best_distance = 3.402823466e+38f;
    precise float horizontal_step = v2_max_horizontal_slope / (float)target_w;
    precise float vertical_step = v2_max_vertical_shear / (float)target_w;
    [unroll]
    for (uint slot = 0u; slot < MAX_LINES; ++slot) {
        if (slot < current_count) {
            uint offset = V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + slot * 4u;
            uint4 rectangle = uint4(
                LocatorStateRead[offset + 0u], LocatorStateRead[offset + 1u],
                LocatorStateRead[offset + 2u], LocatorStateRead[offset + 3u]);
            uint dx = dispatch_id.x < rectangle.x ? rectangle.x - dispatch_id.x :
                (dispatch_id.x >= rectangle.z ? dispatch_id.x - (rectangle.z - 1u) : 0u);
            uint dy = dispatch_id.y < rectangle.y ? rectangle.y - dispatch_id.y :
                (dispatch_id.y >= rectangle.w ? dispatch_id.y - (rectangle.w - 1u) : 0u);
            precise float horizontal_distance = (float)dx * horizontal_step;
            precise float vertical_distance = (float)dy * vertical_step;
            precise float distance = horizontal_distance + vertical_distance;
            if (distance < best_distance) {
                best_distance = distance;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    precise float core_range = 0.5f / (float)locator_source.x;
    precise float budget = core_range + best_distance;
    precise float delta = base - target;
    // Exact Base is a semantic branch, not an algebraic coincidence: bypassing reconstruction
    // avoids changing an already-safe R32_FLOAT bit pattern through target + (base - target).
    if (abs(delta) <= budget) {
        ConditionedField[dispatch_id.xy] = base;
        return;
    }
    precise float full = target + (delta < 0.0f ? -budget : budget);
    precise float faded = base + 0.5f * (full - base);
    ConditionedField[dispatch_id.xy] = fade_step == 1u ? faded : full;
}

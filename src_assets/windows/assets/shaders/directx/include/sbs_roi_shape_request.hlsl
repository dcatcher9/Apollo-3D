#ifndef SBS_ROI_SHAPE_REQUEST_HLSL
#define SBS_ROI_SHAPE_REQUEST_HLSL

// Exact four-vector / 64-byte ABI shared with src/sbs_roi_shape_request.h.
//
// [0] header:   request schema, flags, reason, deterministic request id
// [1] identity: backend generation, ROI generation, rule update count, rule-schema float bits
// [2] shape:    source width, source height, model width, model height
// [3] ROI:      exact committed source-UV x0, y0, x1, y1 float bits

#define SBS_ROI_SHAPE_REQUEST_SCHEMA_VERSION 1u
#define SBS_ROI_SHAPE_REQUEST_VECTOR_COUNT 4u
#define SBS_ROI_SHAPE_REQUEST_PATCH_SIZE 14u
#define SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION 1036u
#define SBS_ROI_SHAPE_REQUEST_MAX_ASPECT_LIMIT 8.0f
#define SBS_ROI_SHAPE_CONTROLLER_SCHEMA_FLOAT_BITS 0x3F800000u

#define SBS_ROI_SHAPE_REQUEST_VECTOR_HEADER 0u
#define SBS_ROI_SHAPE_REQUEST_VECTOR_IDENTITY 1u
#define SBS_ROI_SHAPE_REQUEST_VECTOR_SHAPE 2u
#define SBS_ROI_SHAPE_REQUEST_VECTOR_COMMITTED_ROI 3u

#define SBS_ROI_SHAPE_REQUEST_FLAG_VALID (1u << 0u)
#define SBS_ROI_SHAPE_REQUEST_FLAG_FULL_FRAME (1u << 1u)
#define SBS_ROI_SHAPE_REQUEST_FLAG_ACTIVE_ROI (1u << 2u)
#define SBS_ROI_SHAPE_REQUEST_FLAG_FALLBACK (1u << 3u)
#define SBS_ROI_SHAPE_REQUEST_FLAG_ASPECT_CLAMPED (1u << 4u)
#define SBS_ROI_SHAPE_REQUEST_FLAG_PROFILE_CLAMPED (1u << 5u)
#define SBS_ROI_SHAPE_REQUEST_KNOWN_FLAGS \
    (SBS_ROI_SHAPE_REQUEST_FLAG_VALID | \
     SBS_ROI_SHAPE_REQUEST_FLAG_FULL_FRAME | \
     SBS_ROI_SHAPE_REQUEST_FLAG_ACTIVE_ROI | \
     SBS_ROI_SHAPE_REQUEST_FLAG_FALLBACK | \
     SBS_ROI_SHAPE_REQUEST_FLAG_ASPECT_CLAMPED | \
     SBS_ROI_SHAPE_REQUEST_FLAG_PROFILE_CLAMPED)

#define SBS_ROI_SHAPE_REQUEST_REASON_NONE 0u
#define SBS_ROI_SHAPE_REQUEST_REASON_INACTIVE 1u
#define SBS_ROI_SHAPE_REQUEST_REASON_INVALID_SOURCE_OR_PROFILE 2u
#define SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_SCHEMA_MISMATCH 3u
#define SBS_ROI_SHAPE_REQUEST_REASON_BACKEND_GENERATION_MISMATCH 4u
#define SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_OUTPUT_INVALID 5u
#define SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_UNINITIALIZED 6u
#define SBS_ROI_SHAPE_REQUEST_REASON_ROI_NOT_LOCKED 7u
#define SBS_ROI_SHAPE_REQUEST_REASON_CONTROLLER_FALLBACK_ACTIVE 8u
#define SBS_ROI_SHAPE_REQUEST_REASON_INVALID_RULE_IDENTITY 9u
#define SBS_ROI_SHAPE_REQUEST_REASON_MALFORMED_COMMITTED_ROI 10u

struct SbsRoiShapeRequestData {
    uint4 header;
    uint4 identity;
    uint4 shape;
    uint4 committed_roi_bits;
};

SbsRoiShapeRequestData SbsRoiShapeRequestDecode(
    uint4 header,
    uint4 identity,
    uint4 shape,
    uint4 committed_roi_bits)
{
    SbsRoiShapeRequestData value;
    value.header = header;
    value.identity = identity;
    value.shape = shape;
    value.committed_roi_bits = committed_roi_bits;
    return value;
}

#define SBS_ROI_SHAPE_REQUEST_DECODE_RESOURCE(resource) \
    SbsRoiShapeRequestDecode( \
        resource[SBS_ROI_SHAPE_REQUEST_VECTOR_HEADER], \
        resource[SBS_ROI_SHAPE_REQUEST_VECTOR_IDENTITY], \
        resource[SBS_ROI_SHAPE_REQUEST_VECTOR_SHAPE], \
        resource[SBS_ROI_SHAPE_REQUEST_VECTOR_COMMITTED_ROI])

uint SbsRoiShapeRequestHashWord(uint hash, uint word) {
    return (hash ^ word) * 16777619u;
}

uint SbsRoiShapeRequestIdFromFields(
    uint request_schema,
    uint backend_generation,
    uint roi_generation,
    uint sampled_rule_update_count,
    uint2 source_dimensions,
    uint2 model_dimensions,
    uint4 committed_roi_bits)
{
    uint hash = 2166136261u;
    hash = SbsRoiShapeRequestHashWord(hash, request_schema);
    hash = SbsRoiShapeRequestHashWord(hash, backend_generation);
    hash = SbsRoiShapeRequestHashWord(hash, roi_generation);
    hash = SbsRoiShapeRequestHashWord(hash, sampled_rule_update_count);
    hash = SbsRoiShapeRequestHashWord(hash, source_dimensions.x);
    hash = SbsRoiShapeRequestHashWord(hash, source_dimensions.y);
    hash = SbsRoiShapeRequestHashWord(hash, model_dimensions.x);
    hash = SbsRoiShapeRequestHashWord(hash, model_dimensions.y);
    hash = SbsRoiShapeRequestHashWord(hash, committed_roi_bits.x);
    hash = SbsRoiShapeRequestHashWord(hash, committed_roi_bits.y);
    hash = SbsRoiShapeRequestHashWord(hash, committed_roi_bits.z);
    hash = SbsRoiShapeRequestHashWord(hash, committed_roi_bits.w);
    return hash == 0u ? 0xA511E9B3u : hash;
}

uint SbsRoiShapeRequestId(SbsRoiShapeRequestData value) {
    return SbsRoiShapeRequestIdFromFields(
        value.header.x,
        value.identity.x,
        value.identity.y,
        value.identity.z,
        value.shape.xy,
        value.shape.zw,
        value.committed_roi_bits);
}

bool SbsRoiShapeRequestRectValid(uint4 bits) {
    float4 rect = asfloat(bits);
    return all(isfinite(rect)) &&
           all(rect.xy >= 0.0f.xx) &&
           all(rect.zw <= 1.0f.xx) &&
           all(rect.zw > rect.xy);
}

bool SbsRoiShapeRequestPatchAligned(uint value) {
    return value >= SBS_ROI_SHAPE_REQUEST_PATCH_SIZE &&
           value % SBS_ROI_SHAPE_REQUEST_PATCH_SIZE == 0u;
}

bool SbsRoiShapeRequestValid(SbsRoiShapeRequestData value) {
    uint flags = value.header.y;
    bool full_frame =
        (flags & SBS_ROI_SHAPE_REQUEST_FLAG_FULL_FRAME) != 0u;
    bool active_roi =
        (flags & SBS_ROI_SHAPE_REQUEST_FLAG_ACTIVE_ROI) != 0u;
    bool fallback =
        (flags & SBS_ROI_SHAPE_REQUEST_FLAG_FALLBACK) != 0u;
    bool common =
        value.header.x == SBS_ROI_SHAPE_REQUEST_SCHEMA_VERSION &&
        (flags & SBS_ROI_SHAPE_REQUEST_FLAG_VALID) != 0u &&
        (flags & ~SBS_ROI_SHAPE_REQUEST_KNOWN_FLAGS) == 0u &&
        full_frame != active_roi &&
        value.header.z <=
            SBS_ROI_SHAPE_REQUEST_REASON_MALFORMED_COMMITTED_ROI &&
        fallback == (value.header.z != 0u) &&
        all(value.shape.xy > 0u.xx) &&
        SbsRoiShapeRequestPatchAligned(value.shape.z) &&
        SbsRoiShapeRequestPatchAligned(value.shape.w) &&
        value.shape.z <= value.shape.x &&
        value.shape.w <= value.shape.y &&
        value.shape.z <= SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION &&
        value.shape.w <= SBS_ROI_SHAPE_REQUEST_ENGINE_MAX_DIMENSION &&
        value.header.w != 0u &&
        value.header.w == SbsRoiShapeRequestId(value);
    if (!common) {
        return false;
    }
    if (active_roi) {
        return !fallback &&
               value.identity.x != 0u &&
               value.identity.y != 0u &&
               value.identity.z != 0u &&
               value.identity.w ==
                   SBS_ROI_SHAPE_CONTROLLER_SCHEMA_FLOAT_BITS &&
               SbsRoiShapeRequestRectValid(value.committed_roi_bits);
    }
    return fallback;
}

bool SbsRoiShapeRequestMatchesExact(
    SbsRoiShapeRequestData value,
    uint expected_backend_generation,
    uint expected_roi_generation,
    uint expected_rule_update_count,
    uint expected_request_id,
    uint2 expected_source_dimensions,
    uint2 expected_model_dimensions,
    uint4 expected_committed_roi_bits)
{
    return SbsRoiShapeRequestValid(value) &&
           value.identity.x == expected_backend_generation &&
           value.identity.y == expected_roi_generation &&
           (expected_rule_update_count == 0u ||
            value.identity.z == expected_rule_update_count) &&
           value.header.w == expected_request_id &&
           all(value.shape.xy == expected_source_dimensions) &&
           all(value.shape.zw == expected_model_dimensions) &&
           all(value.committed_roi_bits == expected_committed_roi_bits);
}

bool SbsRoiShapeRequestCurrentRuleMatches(
    SbsRoiShapeRequestData value,
    uint current_backend_generation,
    uint current_roi_generation,
    uint current_rule_update_count,
    uint2 current_source_dimensions,
    uint4 current_committed_roi_bits)
{
    return value.identity.x == current_backend_generation &&
           value.identity.y == current_roi_generation &&
           current_rule_update_count >= value.identity.z &&
           all(value.shape.xy == current_source_dimensions) &&
           all(value.committed_roi_bits == current_committed_roi_bits);
}

#endif

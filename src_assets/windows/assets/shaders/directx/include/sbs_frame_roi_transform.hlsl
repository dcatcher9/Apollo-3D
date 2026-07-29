#ifndef SBS_FRAME_ROI_TRANSFORM_HLSL
#define SBS_FRAME_ROI_TRANSFORM_HLSL

// GPU ABI for the exact source/model transform owned by one accepted DA-V2 frame.
//
// Every resource is StructuredBuffer<uint4> / RWStructuredBuffer<uint4> with exactly eight
// elements. Integer identity fields remain exact integers; rectangle/scalar floats are stored
// with asuint()/asfloat(). Consumers declare the current resource with the exact global name
// `FrameRoiTransform` at any free t-register before including this file.
//
//   [0] header:       schema, flags, source_frame_id low, source_frame_id high
//   [1] identity:     roi_generation, backend_generation, source_width, source_height
//   [2] model:        model_width, model_height, accepted_pixel_count, shape_request_id
//   [3] focus:        exact source-UV x0, y0, x1, y1
//   [4] crop:         enclosing source-UV x0, y0, x1, y1
//   [5] accepted:     exact model-pixel bounds x0, y0, x1, y1 (half-open)
//   [6] feather:      source-UV left, top, right, bottom inward feather widths
//   [7] lifecycle:    transform_version low/high, GPU bank identity, fallback reason
//
// ABI v1 has no model viewport, letterbox, pillarbox, or synthetic padding. Every model texel
// maps to real source content inside `crop`. The accepted-focus mask is exact integer model-pixel
// bounds so every shader classifies boundary pixels identically.

#define SBS_FRAME_ROI_SCHEMA_VERSION 1u
#define SBS_FRAME_ROI_VECTOR_COUNT 8u
#define SBS_FRAME_ROI_MODEL_PATCH_SIZE 14u
#define SBS_FRAME_ROI_BANK_COUNT 2u
#define SBS_FRAME_ROI_ACTIVE_ASPECT_REL_TOLERANCE 0.0001f
#define SBS_FRAME_ROI_LEGACY_ASPECT_REL_TOLERANCE 0.02f

#define SBS_FRAME_ROI_VECTOR_HEADER 0u
#define SBS_FRAME_ROI_VECTOR_IDENTITY 1u
#define SBS_FRAME_ROI_VECTOR_MODEL 2u
#define SBS_FRAME_ROI_VECTOR_FOCUS 3u
#define SBS_FRAME_ROI_VECTOR_CROP 4u
#define SBS_FRAME_ROI_VECTOR_ACCEPTED_BOUNDS 5u
#define SBS_FRAME_ROI_VECTOR_FEATHER 6u
#define SBS_FRAME_ROI_VECTOR_DIAGNOSTICS 7u

#define SBS_FRAME_ROI_FLAG_VALID (1u << 0u)
#define SBS_FRAME_ROI_FLAG_FULL_FRAME (1u << 1u)
#define SBS_FRAME_ROI_FLAG_ACTIVE_ROI (1u << 2u)
#define SBS_FRAME_ROI_FLAG_RESET_DEBT (1u << 3u)
#define SBS_FRAME_ROI_KNOWN_FLAGS \
    (SBS_FRAME_ROI_FLAG_VALID | SBS_FRAME_ROI_FLAG_FULL_FRAME | \
     SBS_FRAME_ROI_FLAG_ACTIVE_ROI | SBS_FRAME_ROI_FLAG_RESET_DEBT)

#define SBS_FRAME_ROI_FALLBACK_NONE 0u
#define SBS_FRAME_ROI_FALLBACK_INACTIVE 1u
#define SBS_FRAME_ROI_FALLBACK_CONTROLLER_INVALID 2u
#define SBS_FRAME_ROI_FALLBACK_NO_LOCKED_ROI 3u
#define SBS_FRAME_ROI_FALLBACK_MALFORMED_ROI 4u
#define SBS_FRAME_ROI_FALLBACK_ROI_TOO_SMALL 5u
#define SBS_FRAME_ROI_FALLBACK_ASPECT_ENVELOPE_IMPOSSIBLE 6u
#define SBS_FRAME_ROI_FALLBACK_EMPTY_FOCUS 7u
#define SBS_FRAME_ROI_FALLBACK_SHAPE_REQUEST_MISMATCH 8u
#define SBS_FRAME_ROI_FALLBACK_FULL_FRAME_SHAPE_MISMATCH 9u
#define SBS_FRAME_ROI_FALLBACK_INVALID_DIMENSIONS 10u

#define SBS_FRAME_ROI_DECODE_RESOURCE(resource) \
    FrameRoiDecode( \
        resource[SBS_FRAME_ROI_VECTOR_HEADER], \
        resource[SBS_FRAME_ROI_VECTOR_IDENTITY], \
        resource[SBS_FRAME_ROI_VECTOR_MODEL], \
        resource[SBS_FRAME_ROI_VECTOR_FOCUS], \
        resource[SBS_FRAME_ROI_VECTOR_CROP], \
        resource[SBS_FRAME_ROI_VECTOR_ACCEPTED_BOUNDS], \
        resource[SBS_FRAME_ROI_VECTOR_FEATHER], \
        resource[SBS_FRAME_ROI_VECTOR_DIAGNOSTICS])

struct SbsFrameRoiTransformData {
    uint4 header;
    uint4 identity;
    uint4 model;
    uint4 focus_bits;
    uint4 crop_bits;
    uint4 accepted_bounds;
    uint4 feather_bits;
    uint4 diagnostics;
};

SbsFrameRoiTransformData FrameRoiDecode(
    uint4 header,
    uint4 identity,
    uint4 model,
    uint4 focus_bits,
    uint4 crop_bits,
    uint4 accepted_bounds,
    uint4 feather_bits,
    uint4 diagnostics)
{
    SbsFrameRoiTransformData value;
    value.header = header;
    value.identity = identity;
    value.model = model;
    value.focus_bits = focus_bits;
    value.crop_bits = crop_bits;
    value.accepted_bounds = accepted_bounds;
    value.feather_bits = feather_bits;
    value.diagnostics = diagnostics;
    return value;
}

SbsFrameRoiTransformData FrameRoiTransformLoad() {
    return SBS_FRAME_ROI_DECODE_RESOURCE(FrameRoiTransform);
}

uint FrameRoiDataFlags(SbsFrameRoiTransformData value) {
    return value.header.y;
}

uint2 FrameRoiDataSourceFrameId(SbsFrameRoiTransformData value) {
    return value.header.zw;
}

uint FrameRoiDataRoiGeneration(SbsFrameRoiTransformData value) {
    return value.identity.x;
}

uint FrameRoiDataBackendGeneration(SbsFrameRoiTransformData value) {
    return value.identity.y;
}

uint2 FrameRoiDataSourceDimensions(SbsFrameRoiTransformData value) {
    return value.identity.zw;
}

uint2 FrameRoiDataModelDimensions(SbsFrameRoiTransformData value) {
    return value.model.xy;
}

uint FrameRoiDataAcceptedFocusPixelCount(SbsFrameRoiTransformData value) {
    return value.model.z;
}

uint FrameRoiDataShapeRequestId(SbsFrameRoiTransformData value) {
    return value.model.w;
}

uint4 FrameRoiDataAcceptedModelBounds(SbsFrameRoiTransformData value) {
    return value.accepted_bounds;
}

uint2 FrameRoiDataTransformVersion(SbsFrameRoiTransformData value) {
    return value.diagnostics.xy;
}

uint FrameRoiDataBankIdentity(SbsFrameRoiTransformData value) {
    return value.diagnostics.z;
}

uint FrameRoiDataFallbackReason(SbsFrameRoiTransformData value) {
    return value.diagnostics.w;
}

bool FrameRoiDataUnboundZero(SbsFrameRoiTransformData value) {
    return all(value.header == 0u.xxxx) &&
           all(value.identity == 0u.xxxx) &&
           all(value.model == 0u.xxxx) &&
           all(value.focus_bits == 0u.xxxx) &&
           all(value.crop_bits == 0u.xxxx) &&
           all(value.accepted_bounds == 0u.xxxx) &&
           all(value.feather_bits == 0u.xxxx) &&
           all(value.diagnostics == 0u.xxxx);
}

float4 FrameRoiDataFocus(SbsFrameRoiTransformData value) {
    return asfloat(value.focus_bits);
}

float4 FrameRoiDataCrop(SbsFrameRoiTransformData value) {
    return asfloat(value.crop_bits);
}

float4 FrameRoiDataFeather(SbsFrameRoiTransformData value) {
    return asfloat(value.feather_bits);
}

bool FrameRoiRectValid(float4 rect) {
    return all(isfinite(rect)) &&
           all(rect.xy >= 0.0f.xx) &&
           all(rect.zw <= 1.0f.xx) &&
           all(rect.zw > rect.xy);
}

bool FrameRoiRectContains(float4 outer_rect, float4 inner_rect) {
    return all(inner_rect.xy >= outer_rect.xy) &&
           all(inner_rect.zw <= outer_rect.zw);
}

bool FrameRoiRectIsCanonicalFull(float4 rect) {
    return all(asuint(rect) == asuint(float4(0.0f, 0.0f, 1.0f, 1.0f)));
}

void FrameRoiAxisAcceptedBounds(
    float lo,
    float hi,
    uint extent,
    out uint first,
    out uint end)
{
    first = (uint)clamp(
        (int)ceil(lo * (float)extent - 0.5f),
        0,
        (int)extent);
    end = (uint)clamp(
        (int)ceil(hi * (float)extent - 0.5f),
        0,
        (int)extent);
}

uint4 FrameRoiExpectedAcceptedModelBounds(
    float4 focus,
    float4 crop,
    uint2 model_size)
{
    float2 crop_size = crop.zw - crop.xy;
    float2 focus_model_lo = (focus.xy - crop.xy) / crop_size;
    float2 focus_model_hi = (focus.zw - crop.xy) / crop_size;
    uint x0;
    uint x1;
    uint y0;
    uint y1;
    FrameRoiAxisAcceptedBounds(
        focus_model_lo.x,
        focus_model_hi.x,
        model_size.x,
        x0,
        x1);
    FrameRoiAxisAcceptedBounds(
        focus_model_lo.y,
        focus_model_hi.y,
        model_size.y,
        y0,
        y1);
    return uint4(x0, y0, x1, y1);
}

bool FrameRoiAcceptedBoundsValid(uint4 bounds, uint2 model_size) {
    return bounds.z > bounds.x &&
           bounds.w > bounds.y &&
           bounds.z <= model_size.x &&
           bounds.w <= model_size.y;
}

uint FrameRoiAcceptedBoundsPixelCount(uint4 bounds) {
    return (bounds.z - bounds.x) * (bounds.w - bounds.y);
}

bool FrameRoiPhysicalAspectMatches(
    float4 crop,
    uint2 source_size,
    uint2 model_size,
    float relative_tolerance)
{
    float2 crop_pixels =
        (crop.zw - crop.xy) * float2(source_size);
    float left = crop_pixels.x * (float)model_size.y;
    float right = crop_pixels.y * (float)model_size.x;
    float scale = max(max(abs(left), abs(right)), 1.0f);
    return isfinite(left) &&
           isfinite(right) &&
           abs(left - right) <= scale * relative_tolerance;
}

bool FrameRoiDataValid(SbsFrameRoiTransformData value) {
    uint flags = FrameRoiDataFlags(value);
    bool full_frame = (flags & SBS_FRAME_ROI_FLAG_FULL_FRAME) != 0u;
    bool active_roi = (flags & SBS_FRAME_ROI_FLAG_ACTIVE_ROI) != 0u;
    uint2 source_size = FrameRoiDataSourceDimensions(value);
    uint2 model_size = FrameRoiDataModelDimensions(value);
    float4 focus = FrameRoiDataFocus(value);
    float4 crop = FrameRoiDataCrop(value);
    float4 feather = FrameRoiDataFeather(value);
    uint4 accepted_bounds = FrameRoiDataAcceptedModelBounds(value);
    uint accepted = FrameRoiDataAcceptedFocusPixelCount(value);
    bool model_patch_aligned =
        all((model_size % SBS_FRAME_ROI_MODEL_PATCH_SIZE) == 0u.xx);
    bool model_pixel_count_safe =
        model_size.x <= 0xffffffffu / max(model_size.y, 1u);
    bool accepted_valid =
        FrameRoiAcceptedBoundsValid(accepted_bounds, model_size) &&
        accepted == FrameRoiAcceptedBoundsPixelCount(accepted_bounds);
    bool common =
        value.header.x == SBS_FRAME_ROI_SCHEMA_VERSION &&
        (flags & SBS_FRAME_ROI_FLAG_VALID) != 0u &&
        (flags & ~SBS_FRAME_ROI_KNOWN_FLAGS) == 0u &&
        full_frame != active_roi &&
        all(source_size > 0u.xx) &&
        all(model_size > 0u.xx) &&
        model_patch_aligned &&
        model_pixel_count_safe &&
        any(FrameRoiDataTransformVersion(value) != 0u.xx) &&
        FrameRoiDataBankIdentity(value) < SBS_FRAME_ROI_BANK_COUNT &&
        FrameRoiRectValid(focus) &&
        FrameRoiRectValid(crop) &&
        FrameRoiRectContains(crop, focus) &&
        accepted_valid &&
        all(isfinite(feather)) &&
        all(feather >= 0.0f.xxxx);
    if (!common) {
        return false;
    }
    if (full_frame) {
        return FrameRoiRectIsCanonicalFull(focus) &&
               FrameRoiRectIsCanonicalFull(crop) &&
               all(accepted_bounds == uint4(0u, 0u, model_size)) &&
               accepted == model_size.x * model_size.y &&
               all(feather == 0.0f.xxxx) &&
               FrameRoiPhysicalAspectMatches(
                   crop,
                   source_size,
                   model_size,
                   SBS_FRAME_ROI_LEGACY_ASPECT_REL_TOLERANCE);
    }

    float2 focus_size = focus.zw - focus.xy;
    uint4 expected_bounds =
        FrameRoiExpectedAcceptedModelBounds(focus, crop, model_size);
    bool feather_valid =
        feather.x <= 0.25f * focus_size.x &&
        feather.z <= 0.25f * focus_size.x &&
        feather.y <= 0.25f * focus_size.y &&
        feather.w <= 0.25f * focus_size.y;
    return FrameRoiDataRoiGeneration(value) != 0u &&
           FrameRoiDataBackendGeneration(value) != 0u &&
           FrameRoiDataShapeRequestId(value) != 0u &&
           all(accepted_bounds == expected_bounds) &&
           feather_valid &&
           FrameRoiPhysicalAspectMatches(
               crop,
               source_size,
               model_size,
               SBS_FRAME_ROI_ACTIVE_ASPECT_REL_TOLERANCE);
}

bool FrameRoiDataCanonicalFullFrame(SbsFrameRoiTransformData value) {
    return FrameRoiDataValid(value) &&
           (FrameRoiDataFlags(value) & SBS_FRAME_ROI_FLAG_FULL_FRAME) != 0u;
}

bool FrameRoiDataActive(SbsFrameRoiTransformData value) {
    return FrameRoiDataValid(value) &&
           (FrameRoiDataFlags(value) & SBS_FRAME_ROI_FLAG_ACTIVE_ROI) != 0u;
}

bool FrameRoiDataResetDebt(SbsFrameRoiTransformData value) {
    return FrameRoiDataValid(value) &&
           (FrameRoiDataFlags(value) & SBS_FRAME_ROI_FLAG_RESET_DEBT) != 0u;
}

bool FrameRoiDataSameGeometry(
    SbsFrameRoiTransformData first,
    SbsFrameRoiTransformData second)
{
    if (!FrameRoiDataValid(first) || !FrameRoiDataValid(second)) {
        return false;
    }
    uint geometry_flags =
        SBS_FRAME_ROI_FLAG_FULL_FRAME |
        SBS_FRAME_ROI_FLAG_ACTIVE_ROI;
    return
        (FrameRoiDataFlags(first) & geometry_flags) ==
            (FrameRoiDataFlags(second) & geometry_flags) &&
        all(first.identity == second.identity) &&
        all(first.model.xyz == second.model.xyz) &&
        all(first.focus_bits == second.focus_bits) &&
        all(first.crop_bits == second.crop_bits) &&
        all(first.accepted_bounds == second.accepted_bounds) &&
        all(first.feather_bits == second.feather_bits);
}

bool FrameRoiDataGeometryReseedRequired(
    SbsFrameRoiTransformData current,
    SbsFrameRoiTransformData previous)
{
    return !FrameRoiDataSameGeometry(current, previous) ||
           (FrameRoiDataResetDebt(current) &&
            !FrameRoiDataResetDebt(previous));
}

bool FrameRoiPointInsideRect(float2 coordinate, float4 rect) {
    return all(coordinate >= rect.xy) && all(coordinate < rect.zw);
}

bool FrameRoiDataSourceInsideFocus(
    SbsFrameRoiTransformData value,
    float2 source_uv)
{
    return FrameRoiDataValid(value) &&
           FrameRoiPointInsideRect(source_uv, FrameRoiDataFocus(value));
}

bool FrameRoiDataSourceToModelUv(
    SbsFrameRoiTransformData value,
    float2 source_uv,
    out float2 model_uv)
{
    if (!FrameRoiDataValid(value)) {
        model_uv = 0.0f.xx;
        return false;
    }
    float4 crop = FrameRoiDataCrop(value);
    model_uv = (source_uv - crop.xy) / (crop.zw - crop.xy);
    return FrameRoiPointInsideRect(source_uv, crop);
}

bool FrameRoiDataModelToSourceUv(
    SbsFrameRoiTransformData value,
    float2 model_uv,
    out float2 source_uv)
{
    if (!FrameRoiDataValid(value)) {
        source_uv = 0.0f.xx;
        return false;
    }
    float4 crop = FrameRoiDataCrop(value);
    source_uv = crop.xy + model_uv * (crop.zw - crop.xy);
    return all(model_uv >= 0.0f.xx) && all(model_uv < 1.0f.xx);
}

bool FrameRoiDataAcceptedModelPixel(
    SbsFrameRoiTransformData value,
    uint2 model_pixel)
{
    if (!FrameRoiDataValid(value)) {
        return false;
    }
    uint4 bounds = FrameRoiDataAcceptedModelBounds(value);
    return all(model_pixel >= bounds.xy) &&
           all(model_pixel < bounds.zw);
}

float2 FrameRoiDataClampModelUv(
    SbsFrameRoiTransformData value,
    float2 model_uv,
    uint2 texture_dimensions)
{
    if (!FrameRoiDataValid(value)) {
        return saturate(model_uv);
    }
    float2 half_texel =
        0.5f / max(float2(texture_dimensions), 1.0f.xx);
    return clamp(model_uv, half_texel, 1.0f.xx - half_texel);
}

float FrameRoiFeatherRamp(float distance_inside, float width) {
    return width > 1e-8f ?
        smoothstep(0.0f, width, max(distance_inside, 0.0f)) :
        (distance_inside >= 0.0f ? 1.0f : 0.0f);
}

float FrameRoiDataFeatherWeight(
    SbsFrameRoiTransformData value,
    float2 source_uv)
{
    if (!FrameRoiDataSourceInsideFocus(value, source_uv)) {
        return 0.0f;
    }
    if (FrameRoiDataCanonicalFullFrame(value)) {
        return 1.0f;
    }
    float4 focus = FrameRoiDataFocus(value);
    float4 feather = FrameRoiDataFeather(value);
    float left = FrameRoiFeatherRamp(source_uv.x - focus.x, feather.x);
    float top = FrameRoiFeatherRamp(source_uv.y - focus.y, feather.y);
    float right = FrameRoiFeatherRamp(focus.z - source_uv.x, feather.z);
    float bottom = FrameRoiFeatherRamp(focus.w - source_uv.y, feather.w);
    return min(min(left, right), min(top, bottom));
}

// Convenience wrappers for the shader's current `FrameRoiTransform` resource.
bool FrameRoiTransformValid() {
    return FrameRoiDataValid(FrameRoiTransformLoad());
}

bool FrameRoiTransformCanonicalFullFrame() {
    SbsFrameRoiTransformData value = FrameRoiTransformLoad();
    return FrameRoiDataUnboundZero(value) ||
           FrameRoiDataCanonicalFullFrame(value);
}

bool FrameRoiTransformActive() {
    return FrameRoiDataActive(FrameRoiTransformLoad());
}

bool FrameRoiTransformResetDebt() {
    return FrameRoiDataResetDebt(FrameRoiTransformLoad());
}

uint FrameRoiTransformAcceptedFocusPixelCount() {
    return FrameRoiDataAcceptedFocusPixelCount(FrameRoiTransformLoad());
}

bool FrameRoiSourceInsideFocus(float2 source_uv) {
    return FrameRoiDataSourceInsideFocus(
        FrameRoiTransformLoad(),
        source_uv);
}

bool FrameRoiSourceToModelUv(float2 source_uv, out float2 model_uv) {
    return FrameRoiDataSourceToModelUv(
        FrameRoiTransformLoad(),
        source_uv,
        model_uv);
}

bool FrameRoiModelToSourceUv(float2 model_uv, out float2 source_uv) {
    return FrameRoiDataModelToSourceUv(
        FrameRoiTransformLoad(),
        model_uv,
        source_uv);
}

bool FrameRoiAcceptedModelPixel(uint2 model_pixel) {
    return FrameRoiDataAcceptedModelPixel(
        FrameRoiTransformLoad(),
        model_pixel);
}

float2 FrameRoiClampModelUv(
    float2 model_uv,
    uint2 texture_dimensions)
{
    return FrameRoiDataClampModelUv(
        FrameRoiTransformLoad(),
        model_uv,
        texture_dimensions);
}

float FrameRoiFeatherWeight(float2 source_uv) {
    return FrameRoiDataFeatherWeight(
        FrameRoiTransformLoad(),
        source_uv);
}

#endif

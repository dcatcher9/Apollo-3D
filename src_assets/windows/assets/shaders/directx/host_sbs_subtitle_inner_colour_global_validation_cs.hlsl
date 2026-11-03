// PRIVATE FROZEN-V5 INNER-COLOUR EXTRACTION. Not a producer/draw-authority manifest member.
// Source-only port: authentic current RGB/P/S/OCR -> per-unit palette -> per-mode E/Seeds/search/cover.
// Parent/CC boundary rejection and one BOX3 dilation live in the separately frozen five-pass operator.
// Fixed command loops use GPU ModeDispatchArgs; no CPU palette verdict or history/age borrowing.
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"

cbuffer InnerColourConstants : register(b0) {
    uint4 contour_source;
    uint4 contour_field;
    uint4 contour_content;
    uint4 contour_frame;
    uint4 contour_observation;
    uint4 contour_options;
};
cbuffer InnerColourRequest : register(b1) { uint4 inner_request; }; // slot, mode, 0, 0
Texture2D<float4> CurrentDisplayBand : register(t0);
StructuredBuffer<float> CurrentProbability : register(t1);
Texture2D<uint> CurrentSupport : register(t2);
StructuredBuffer<uint> CurrentOcrRecord : register(t3);
StructuredBuffer<uint> PaletteRead : register(t4);
Texture2D<uint> ModeBody : register(t5);
Texture2D<uint> ModeFinal : register(t6);
StructuredBuffer<uint> ComponentParent : register(t7);
RWStructuredBuffer<uint> PaletteWrite : register(u0);
RWTexture2D<uint> EligibleWrite : register(u1);
RWTexture2D<uint> SeedsWrite : register(u2);
RWTexture2D<uint> DomainWrite : register(u3);
RWTexture2D<uint> CoverWrite : register(u4);
RWTexture2D<uint> AccumulatedBody : register(u5);
RWTexture2D<uint> AccumulatedFinal : register(u6);
RWByteAddressBuffer ModeDispatchArgs : register(u7); //96B/8 uint3, unbound before indirect consumers

static const uint INNER_MAGIC = 0x35434c49u;
static const uint INNER_SCHEMA = 1u;
static const uint INNER_WORDS = 128u;
static const uint INNER_MODE_OFFSET = 64u;
static const uint INNER_MODE_WORDS = 8u;
static const uint INNER_INVALID = 0u;
static const uint INNER_EMPTY = 1u;
static const uint INNER_READY = 2u;
static const uint INNER_SLOT_OUTSIDE = 3u;

groupshared uint CurrentCount;
groupshared int CurrentReferenceCenterTwice;
groupshared uint4 CurrentCores[8];
groupshared uint4 CurrentCovers[8];
groupshared uint GTyped;
groupshared uint GFault;
groupshared uint GAnchorCount;
groupshared uint GAnchorPeak;
groupshared uint GBinCount[64];
groupshared uint GBinRepresentative[64];
groupshared uint GModeCount;
groupshared uint GModeReady;
groupshared uint GPlaneShape;
groupshared uint4 GSearch;
groupshared uint3 GRepresentative;

bool ContourFinite(float v) { return !isnan(v) && !isinf(v); }
bool ContourFinite3(float3 v) { return all(!isnan(v)) && all(!isinf(v)); }
bool BandPositionValid(int2 p) {
    return all(p >= int2(0,0)) && p.x < (int)contour_source.x && p.y < (int)contour_source.w;
}
bool LoadDisplayColor(int2 p, out float3 color) {
    color=0.0f;
    if (!BandPositionValid(p)) return false;
    float4 v=CurrentDisplayBand.Load(int3(p,0));
    if (v.a!=1.0f || !ContourFinite3(v.rgb) || any(v.rgb<0.0f) || any(v.rgb>1.0f)) return false;
    color=v.rgb; return true;
}
uint ReadConstant(uint index) {
    if(index<4u)return contour_source[index];
    if(index<8u)return contour_field[index-4u];
    if(index<12u)return contour_content[index-8u];
    if(index<16u)return contour_frame[index-12u];
    if(index<20u)return contour_observation[index-16u];
    return contour_options[index-20u];
}
uint4 ReadRect(uint offset,uint slot) {
    uint start=offset+slot*8u;
    return uint4(CurrentOcrRecord[start],CurrentOcrRecord[start+1u],CurrentOcrRecord[start+2u],CurrentOcrRecord[start+3u]);
}
bool RectValid(uint4 r) {
    return all(r.xy<r.zw) && all(r.xy>=contour_content.xy) && all(r.zw<=contour_content.zw);
}
bool TypedCurrentValid(out uint count,out int reference) {
    count=0u;reference=(int)contour_source.x;
    uint2 rgbSize,sSize;uint pWords,pStride,oWords,oStride;
    CurrentDisplayBand.GetDimensions(rgbSize.x,rgbSize.y);
    CurrentSupport.GetDimensions(sSize.x,sSize.y);
    CurrentProbability.GetDimensions(pWords,pStride);
    CurrentOcrRecord.GetDimensions(oWords,oStride);
    if (any(contour_source.xy==0u) || any(contour_source.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        contour_source.w==0u || contour_source.z>contour_source.y ||
        contour_source.w!=contour_source.y-contour_source.z ||
        any(rgbSize!=contour_source.xw) || any(sSize!=rgbSize) ||
        pWords!=V2_OCR_OUTPUT_WIDTH*V2_OCR_OUTPUT_HEIGHT || pStride!=4u || oWords!=256u || oStride!=4u ||
        any(contour_field.xy==0u) || any(contour_field.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        contour_field.z>=contour_field.w || contour_field.w>contour_field.y ||
        any(contour_content.xy>=contour_content.zw) || any(contour_content.zw>contour_field.xy) ||
        contour_observation.w!=0u || any(contour_options.zw!=0u) || any(inner_request.zw!=0u)) return false;
    if (CurrentOcrRecord[0]!=4u || CurrentOcrRecord[1]!=0x3852434fu || CurrentOcrRecord[2]!=1u ||
        CurrentOcrRecord[3]!=CurrentOcrRecord[4] || CurrentOcrRecord[4]>8u ||
        any(uint4(CurrentOcrRecord[5],CurrentOcrRecord[6],CurrentOcrRecord[7],CurrentOcrRecord[8])!=contour_frame) ||
        any(uint2(CurrentOcrRecord[9],CurrentOcrRecord[10])!=contour_source.xy) ||
        any(uint4(CurrentOcrRecord[11],CurrentOcrRecord[12],CurrentOcrRecord[13],CurrentOcrRecord[14])!=contour_field)) return false;
    count=CurrentOcrRecord[4];
    [loop] for(uint pairSlot=0u;pairSlot<count;++pairSlot) {
        uint4 core=ReadRect(16u,pairSlot),cover=ReadRect(144u,pairSlot);
        if(!RectValid(core)||!RectValid(cover)||any(cover.xy>core.xy)||any(cover.zw<core.zw)) return false;
    }
    uint tail=0u;
    [loop] for(uint tailWord=208u;tailWord<256u;++tailWord) tail|=CurrentOcrRecord[tailWord];
    if(tail!=0u) reference=asint(CurrentOcrRecord[228u]);
    if(reference<-(int)V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT || reference>(int)V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) return false;
    return true;
}
void LoadCurrentGeometry() {
    [unroll] for(uint geometrySlot=0u;geometrySlot<8u;++geometrySlot) {
        CurrentCores[geometrySlot]=GTyped!=0u && geometrySlot<CurrentCount?ReadRect(16u,geometrySlot):0u;
        CurrentCovers[geometrySlot]=GTyped!=0u && geometrySlot<CurrentCount?ReadRect(144u,geometrySlot):0u;
    }
}
bool SourcePixelInRect(int2 p,uint4 r) {
    if(!BandPositionValid(p))return false;
    uint2 scaled=uint2(2u*contour_content.x*contour_source.x+(2u*(uint)p.x+1u)*(contour_content.z-contour_content.x),
        2u*contour_content.y*contour_source.y+(2u*((uint)p.y+contour_source.z)+1u)*(contour_content.w-contour_content.y));
    return scaled.x>=2u*r.x*contour_source.x && scaled.x<2u*r.z*contour_source.x &&
        scaled.y>=2u*r.y*contour_source.y && scaled.y<2u*r.w*contour_source.y;
}
bool WideGreater(uint a_low, uint a_high, uint b_low, uint b_high) {
    return a_high > b_high || (a_high == b_high && a_low > b_low);
}

void CurrentWideProduct(uint a, uint b, out uint low, out uint high) {
    // FXC/SM5 has no umulExtended intrinsic. Split both operands into 16-bit limbs;
    // every intermediate fits uint32, including carries, and the result is exact.
    uint p0 = (a & 65535u) * (b & 65535u);
    uint t = (a >> 16u) * (b & 65535u) + (p0 >> 16u);
    uint cross = (a & 65535u) * (b >> 16u) + (t & 65535u);
    low = (cross << 16u) | (p0 & 65535u);
    high = (a >> 16u) * (b >> 16u) + (t >> 16u) + (cross >> 16u);
}

bool CurrentSlotRibbon(uint slot) {
    return (CurrentOcrRecord[V2_OCR_RAW_BOX_OFFSET + slot * V2_OCR_BOX_WORD_COUNT + 5u] &
        V2_OCR_BOX_FLAG_RIBBON) != 0u;
}

uint4 CurrentOrdinarySupportSearch(uint slot) {
    uint4 cover = CurrentCovers[slot], core = CurrentCores[slot];
    // Ribbon covers have their own producer meaning; none of the ordinary expansion is
    // applied to them. Group only against this fixed core, never a growing/transitive union.
    if (CurrentSlotRibbon(slot)) return cover;
    uint left = core.x, right = core.z;
    [loop]
    for (uint member = 0u; member < CurrentCount; ++member) {
        if (CurrentSlotRibbon(member)) continue;
        uint4 other = CurrentCores[member];
        uint overlap_top = max(core.y, other.y), overlap_bottom = min(core.w, other.w);
        if (overlap_bottom <= overlap_top) continue;
        uint intersection = overlap_bottom - overlap_top;
        uint union_height = max(core.w, other.w) - min(core.y, other.y);
        if (intersection * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR <
            union_height * V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR) continue;
        left = min(left, other.x);
        right = max(right, other.z);
    }
    uint content_width = contour_content.z - contour_content.x;
    uint center_field = left + right - 2u * contour_content.x;
    uint line_low, line_high;
    CurrentWideProduct(center_field * contour_source.x, V2_OCR_OUTPUT_WIDTH, line_low, line_high);
    int reference_scaled = CurrentReferenceCenterTwice * (int)V2_OCR_OUTPUT_WIDTH;
    int margin = (int)(2u * V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_CENTER_BIAS_DETECTOR_PIXELS * contour_source.x);
    int upper = reference_scaled + margin, lower = reference_scaled - margin;
    bool right_biased = upper < 0;
    if (upper >= 0) {
        uint upper_low, upper_high;
        CurrentWideProduct((uint)upper, content_width, upper_low, upper_high);
        right_biased = WideGreater(line_low, line_high, upper_low, upper_high);
    }
    bool left_biased = false;
    if (lower > 0) {
        uint lower_low, lower_high;
        CurrentWideProduct((uint)lower, content_width, lower_low, lower_high);
        left_biased = WideGreater(lower_low, lower_high, line_low, line_high);
    }
    uint inset = V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_INSET_DETECTOR_PIXELS * content_width / V2_OCR_OUTPUT_WIDTH;
    // Preserve the current core and any producer side already clamped to actual content.
    if (cover.x != contour_content.x && !right_biased) cover.x = min(core.x, cover.x + inset);
    if (cover.z != contour_content.z && !left_biased) cover.z = max(core.z, cover.z - min(inset, cover.z));
    uint pad = ((core.w - core.y) * V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_NUMERATOR +
        V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR - 1u) /
        V2_SUBTITLE_CONTOUR_ORDINARY_SEARCH_PAD_HEIGHT_DENOMINATOR;
    cover.y = max(cover.y, core.y - min(pad, core.y));
    return cover;
}
bool CurrentProbabilityEvidence(int2 p, out float probability) {
    probability = 0.0f;
    // This strict rendering threshold is also authenticated by saved-state readers. Keep
    // each FP32 operation and its rounding point explicit; fused lerp/MAD can cross it.
    precise float2 ratio = float2((float)V2_OCR_OUTPUT_WIDTH / (float)contour_source.x,
        (float)V2_OCR_OUTPUT_HEIGHT / (float)contour_source.w);
    precise float2 centered = float2(p) + 0.5f;
    precise float2 scaled = centered * ratio;
    precise float2 detector = scaled - 0.5f;
    int2 lo = int2(floor(detector));
    precise float2 blend = detector - floor(detector);
    int2 a = clamp(lo, int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    int2 b = clamp(lo + int2(1, 1), int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH - 1u, V2_OCR_OUTPUT_HEIGHT - 1u));
    float tl = CurrentProbability[(uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x];
    float tr = CurrentProbability[(uint)a.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x];
    float bl = CurrentProbability[(uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)a.x];
    float br = CurrentProbability[(uint)b.y * V2_OCR_OUTPUT_WIDTH + (uint)b.x];
    // Global map validity already participates in OCR8's resolver verdict. A locally invalid
    // probability is still never interpolated into a positive contour candidate.
    if (!ContourFinite(tl) || !ContourFinite(tr) || !ContourFinite(bl) || !ContourFinite(br) ||
        tl < 0.0f || tl > 1.0f || tr < 0.0f || tr > 1.0f ||
        bl < 0.0f || bl > 1.0f || br < 0.0f || br > 1.0f) return false;
    precise float upper_delta = tr - tl;
    precise float upper_product = upper_delta * blend.x;
    precise float upper = tl + upper_product;
    precise float lower_delta = br - bl;
    precise float lower_product = lower_delta * blend.x;
    precise float lower = bl + lower_product;
    precise float vertical_delta = lower - upper;
    precise float vertical_product = vertical_delta * blend.y;
    precise float sampled = upper + vertical_product;
    probability = sampled;
    return true;
}



float ColorDistance(float3 a, float3 b) {
    float3 d = abs(a - b);
    return max(d.x, max(d.y, d.z));
}

bool CurrentContrastAt(int2 p, float3 color) {
    float3 neighbor;
    if (!LoadDisplayColor(p, neighbor)) return false;
    return ColorDistance(color, neighbor) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE;
}

bool2 CurrentSpatialEdges(int2 p, float3 color) {
    const int radius = (int)V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RADIUS_SOURCE_PIXELS;
    bool x = CurrentContrastAt(p + int2(radius, 0), color) || CurrentContrastAt(p - int2(radius, 0), color);
    bool y = CurrentContrastAt(p + int2(0, radius), color) || CurrentContrastAt(p - int2(0, radius), color);
    return bool2(x, y);
}

bool CurrentSpatialEdge(int2 p, float3 color) {
    return any(CurrentSpatialEdges(p, color));
}

uint MaxChannel(uint3 value) { return max(value.x, max(value.y, value.z)); }

uint3 PaletteQuantize(float3 color) {
    precise float3 scaled = color * (float)V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX;
    precise float3 rounded = scaled + 0.5f;
    return (uint3)floor(rounded);
}

int2 CurrentPaletteSamplePosition(uint4 core, uint sample_index) {
    uint gx = sample_index % V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS;
    uint gy = sample_index / V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS;
    uint content_width = contour_content.z - contour_content.x;
    uint content_height = contour_content.w - contour_content.y;
    // Integer field-core grid centers, then floor to a native source pixel. At the
    // bounded 5120 extents even the 64-column numerator fits an unsigned 32-bit word.
    uint x = (((core.x - contour_content.x) * (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS) +
        (2u * gx + 1u) * (core.z - core.x)) * contour_source.x) /
        (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_COLUMNS * content_width);
    uint absolute_y = (((core.y - contour_content.y) * (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS) +
        (2u * gy + 1u) * (core.w - core.y)) * contour_source.y) /
        (2u * V2_SUBTITLE_CONTOUR_PALETTE_GRID_ROWS * content_height);
    return int2(min(x, contour_source.x - 1u),
        clamp((int)absolute_y - (int)contour_source.z, 0, (int)contour_source.w - 1));
}

uint3 PaletteHue(uint3 color) {
    uint peak = max(MaxChannel(color), 1u);
    return (color * V2_SUBTITLE_CONTOUR_PALETTE_HUE_CHANNEL_MAX + peak / 2u) / peak;
}
bool CurrentShortTangent(int2 p, float3 color, int2 direction, uint maximum) {
    uint length = 1u;
    if (maximum == 0u) return false;
    [unroll]
    for (int sign = -1; sign <= 1; sign += 2) {
        [loop]
        for (uint distance = 1u; distance <= maximum; ++distance) {
            float3 neighbor;
            bool neighbor_valid = LoadDisplayColor(p + direction * sign * (int)distance, neighbor);
            if (!neighbor_valid) break;
            float contrast = ColorDistance(color, neighbor);
            if (contrast > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) break;
            ++length;
            if (length > maximum) return false;
        }
    }
    return true;
}

bool BlackNeighborAt(int2 neighbor,float3 own,uint representativePeak) {
    float3 rgb;
    if(!LoadDisplayColor(neighbor,rgb))return false;
    uint3 q=PaletteQuantize(rgb);uint peak=MaxChannel(q),low=min(q.x,min(q.y,q.z));
    return V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_DENOMINATOR*peak<=
        V2_SUBTITLE_CONTOUR_OUTLINE_VALUE_NUMERATOR*representativePeak &&
        V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_DENOMINATOR*(peak-low)<=
        V2_SUBTITLE_CONTOUR_OUTLINE_CHROMA_NUMERATOR*V2_SUBTITLE_CONTOUR_PALETTE_CHANNEL_MAX &&
        ColorDistance(own,rgb)>V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE;
}
bool BlackNeighborWitness(int2 p,float3 own,uint representativePeak) {
    int radius=(int)V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RADIUS_SOURCE_PIXELS;
    return BlackNeighborAt(p+int2(-radius,0),own,representativePeak) ||
        BlackNeighborAt(p+int2(radius,0),own,representativePeak) ||
        BlackNeighborAt(p+int2(0,-radius),own,representativePeak) ||
        BlackNeighborAt(p+int2(0,radius),own,representativePeak);
}

// Private per-call numeric proof. u0/PaletteWrite is stage-local: it binds the
// dedicated32-word receipt here and the original128-word palette in palette_main.
// No parent/root flags, SCT word, or production producer contract is reused.
StructuredBuffer<uint> CurrentValidationRead : register(t8);
static const uint INNER_VALIDATION_MAGIC = 0x31564c49u; // ILV1
static const uint INNER_VALIDATION_SCHEMA = 1u;
static const uint INNER_VALIDATION_WORDS = 32u;

bool CurrentValidationPacketMatches() {
    uint validationWords, validationStride;
    CurrentValidationRead.GetDimensions(validationWords,validationStride);
    if(validationWords!=INNER_VALIDATION_WORDS || validationStride!=4u)return false;
    if(CurrentValidationRead[0u]!=INNER_VALIDATION_MAGIC ||
       CurrentValidationRead[1u]!=INNER_VALIDATION_SCHEMA ||
       CurrentValidationRead[2u]!=1u || CurrentValidationRead[3u]!=0u)return false;
    [loop] for(uint validationConstantIndex=0u;validationConstantIndex<24u;++validationConstantIndex)
        if(CurrentValidationRead[4u+validationConstantIndex]!=ReadConstant(validationConstantIndex))return false;
    [unroll] for(uint validationReservedIndex=28u;validationReservedIndex<32u;++validationReservedIndex)
        if(CurrentValidationRead[validationReservedIndex]!=0u)return false;
    return true;
}

[numthreads(16,16,1)]
void validate_main(uint3 group:SV_GroupID,uint3 thread:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    uint validationOutputWords,validationOutputStride;
    PaletteWrite.GetDimensions(validationOutputWords,validationOutputStride);
    if(validationOutputWords!=INNER_VALIDATION_WORDS || validationOutputStride!=4u)return;
    // Canonical native grid hasZ1. Malformed extraZ groups cannot duplicate
    // non-atomic header stores or silently grant a valid proof.
    if(group.z!=0u) {
        if(lane==0u)InterlockedOr(PaletteWrite[3u],8u);
        return;
    }
    if(lane==0u)GTyped=(TypedCurrentValid(CurrentCount,CurrentReferenceCenterTwice) &&
        all(contour_source.xw==inner_request.xy))?1u:0u;
    GroupMemoryBarrierWithGroupSync();
    if(all(group.xy==0u) && lane==0u) {
        PaletteWrite[0u]=INNER_VALIDATION_MAGIC;
        PaletteWrite[1u]=INNER_VALIDATION_SCHEMA;
        PaletteWrite[2u]=GTyped!=0u?1u:0u;
        // Fault is only atomicOR after the dedicated pre-dispatch ClearUAV.
        // It is NEVER reset here, including by the header-stamping thread.
        if(GTyped==0u)InterlockedOr(PaletteWrite[3u],1u);
        [unroll] for(uint validationStampIndex=0u;validationStampIndex<24u;++validationStampIndex)
            PaletteWrite[4u+validationStampIndex]=ReadConstant(validationStampIndex);
    }
    if(GTyped==0u)return;
    if(thread.x>=contour_source.x || thread.y>=contour_source.w)return;
    uint validationNativeCount=contour_source.x*contour_source.w;
    uint validationLinearId=thread.y*contour_source.x+thread.x;
    float3 validationColor;
    if(!LoadDisplayColor(int2(thread.xy),validationColor) || CurrentSupport.Load(int3(thread.xy,0))>1u)
        InterlockedOr(PaletteWrite[3u],2u);
    // Every canonical native linearID is unique. This also covers all153600P
    // when the source crop is smaller than the probability plane.
    [loop] for(uint validationProbabilityIndex=validationLinearId;
        validationProbabilityIndex<V2_OCR_OUTPUT_WIDTH*V2_OCR_OUTPUT_HEIGHT;
        validationProbabilityIndex+=validationNativeCount) {
        float validationProbability=CurrentProbability[validationProbabilityIndex];
        if(!ContourFinite(validationProbability)||validationProbability<0.0f||validationProbability>1.0f)
            InterlockedOr(PaletteWrite[3u],4u);
    }
}

[numthreads(256,1,1)]
void palette_main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    if(any(group!=0u))return;
    uint outputWords,outputStride,argBytes;
    PaletteWrite.GetDimensions(outputWords,outputStride);ModeDispatchArgs.GetDimensions(argBytes);
    if(lane==0u) {
        GTyped=(TypedCurrentValid(CurrentCount,CurrentReferenceCenterTwice) && CurrentValidationPacketMatches())?1u:0u;
        if(outputWords!=INNER_WORDS || outputStride!=4u || argBytes!=96u)GTyped=0u;
        GFault=0u;GAnchorCount=0u;GAnchorPeak=0u;GModeCount=0u;
        LoadCurrentGeometry();
        GSearch=0u;
        if(GTyped!=0u && inner_request.x<CurrentCount)GSearch=CurrentOrdinarySupportSearch(inner_request.x);
    }
    if(lane<64u){GBinCount[lane]=0u;GBinRepresentative[lane]=0u;}
    if(lane<outputWords && lane<INNER_WORDS)PaletteWrite[lane]=0u;
    if(lane<8u && argBytes==96u)ModeDispatchArgs.Store3(lane*12u,uint3(0u,0u,0u));
    AllMemoryBarrierWithGroupSync();
    // Completed preceding native-grid validator owns all-map numeric proof.
    // Every unit checks dedicated receipt32+all24CB words; no per-unit rescan.
    GroupMemoryBarrierWithGroupSync();
    if(GTyped!=0u && GFault==0u && inner_request.x<CurrentCount) {
        uint4 core=CurrentCores[inner_request.x];
        [loop] for(uint sampleIndex=lane;sampleIndex<512u;sampleIndex+=256u) {
            int2 p=CurrentPaletteSamplePosition(core,sampleIndex);float3 color;float probability;
            bool colorValid=LoadDisplayColor(p,color),pValid=CurrentProbabilityEvidence(p,probability);
            if(colorValid && pValid && probability>V2_OCR_ACTIVE_PROBABILITY_THRESHOLD && CurrentSpatialEdge(p,color)) {
                uint3 q=PaletteQuantize(color),bin=min((uint3)(color*4.0f),uint3(3u,3u,3u));
                uint code=bin.x*16u+bin.y*4u+bin.z,peak=MaxChannel(q);
                uint key=(peak<<10u)|(1023u-sampleIndex);
                InterlockedAdd(GBinCount[code],1u);InterlockedMax(GBinRepresentative[code],key);
                InterlockedMax(GAnchorPeak,peak);InterlockedAdd(GAnchorCount,1u);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if(lane==0u && outputWords==INNER_WORDS && outputStride==4u) {
        bool valid=GTyped!=0u && GFault==0u;
        PaletteWrite[0]=INNER_MAGIC;PaletteWrite[1]=INNER_SCHEMA;PaletteWrite[2]=valid?1u:0u;
        PaletteWrite[3]=inner_request.x;PaletteWrite[7]=valid?CurrentCount:0u;
        PaletteWrite[21]=valid?asuint(CurrentReferenceCenterTwice):0u;
        [unroll] for(uint constantWord=0u;constantWord<24u;++constantWord)PaletteWrite[24u+constantWord]=ReadConstant(constantWord);
        uint status=!valid?INNER_INVALID:CurrentCount==0u?INNER_EMPTY:inner_request.x>=CurrentCount?INNER_SLOT_OUTSIDE:INNER_READY;
        PaletteWrite[22]=status;
        if(status==INNER_READY) {
            uint4 core=CurrentCores[inner_request.x],cover=CurrentCovers[inner_request.x];
            [unroll] for(uint rectLane=0u;rectLane<4u;++rectLane){PaletteWrite[8u+rectLane]=GSearch[rectLane];PaletteWrite[12u+rectLane]=cover[rectLane];PaletteWrite[16u+rectLane]=core[rectLane];}
            PaletteWrite[20]=CurrentSlotRibbon(inner_request.x)?1u:0u;
            PaletteWrite[5]=GAnchorCount;PaletteWrite[6]=GAnchorPeak;
            uint visitedLow=0u,visitedHigh=0u;
            [loop] for(uint attemptIndex=0u;attemptIndex<64u;++attemptIndex) {
                uint best=64u,bestPeak=0u,bestCount=1u;
                [loop] for(uint binIndex=0u;binIndex<64u;++binIndex) {
                    uint votes=GBinCount[binIndex],peak=GBinRepresentative[binIndex]>>10u;
                    uint visitedWord=binIndex<32u?visitedLow:visitedHigh;
                    if(votes==0u || (visitedWord&(1u<<(binIndex%32u)))!=0u || 2u*peak<GAnchorPeak)continue;
                    if(best==64u || peak>bestPeak || (peak==bestPeak && (votes>bestCount || (votes==bestCount && binIndex<best)))) {best=binIndex;bestPeak=peak;bestCount=votes;}
                }
                if(best==64u)break;
                if(best<32u)visitedLow|=1u<<best;else visitedHigh|=1u<<(best-32u);
                uint gridIndex=1023u-(GBinRepresentative[best]&1023u);float3 rgb;
                if(!LoadDisplayColor(CurrentPaletteSamplePosition(core,gridIndex),rgb))continue;
                uint3 representative=PaletteQuantize(rgb),hue=PaletteHue(representative);bool represented=false;
                [loop] for(uint priorIndex=0u;priorIndex<GModeCount;++priorIndex) {
                    uint priorStart=INNER_MODE_OFFSET+priorIndex*INNER_MODE_WORDS;
                    uint3 prior=uint3(PaletteWrite[priorStart],PaletteWrite[priorStart+1u],PaletteWrite[priorStart+2u]);
                    uint distance=MaxChannel((uint3)abs(int3(hue)-int3(PaletteHue(prior))));
                    if(20u*distance<=3u*4095u)represented=true;
                }
                if(represented)continue;
                uint start=INNER_MODE_OFFSET+GModeCount*INNER_MODE_WORDS;
                PaletteWrite[start]=representative.x;PaletteWrite[start+1u]=representative.y;PaletteWrite[start+2u]=representative.z;
                PaletteWrite[start+3u]=MaxChannel(representative);PaletteWrite[start+4u]=best;PaletteWrite[start+5u]=bestCount;PaletteWrite[start+6u]=gridIndex;
                if(argBytes==96u && MaxChannel(representative)!=0u)ModeDispatchArgs.Store3(GModeCount*12u,uint3((contour_source.x+15u)/16u,(contour_source.w+15u)/16u,1u));
                ++GModeCount;if(GModeCount==8u)break;
            }
            PaletteWrite[4]=GModeCount;
        }
    }
}

bool PaletteContextValid() {
    uint words,stride;PaletteRead.GetDimensions(words,stride);
    if(words!=INNER_WORDS || stride!=4u || GTyped==0u || inner_request.x>=CurrentCount)return false;
    if(PaletteRead[0]!=INNER_MAGIC || PaletteRead[1]!=INNER_SCHEMA || PaletteRead[2]!=1u || PaletteRead[3]!=inner_request.x ||
       PaletteRead[4]>8u || PaletteRead[5]>512u || PaletteRead[6]>65535u || PaletteRead[7]!=CurrentCount ||
       PaletteRead[20]!=(CurrentSlotRibbon(inner_request.x)?1u:0u) || PaletteRead[21]!=asuint(CurrentReferenceCenterTwice) || PaletteRead[22]!=INNER_READY || PaletteRead[23]!=0u)return false;
    uint4 core=CurrentCores[inner_request.x],cover=CurrentCovers[inner_request.x],search=CurrentOrdinarySupportSearch(inner_request.x);
    [unroll] for(uint contextLane=0u;contextLane<4u;++contextLane)if(PaletteRead[8u+contextLane]!=search[contextLane] || PaletteRead[12u+contextLane]!=cover[contextLane] || PaletteRead[16u+contextLane]!=core[contextLane])return false;
    [loop] for(uint checkConstant=0u;checkConstant<24u;++checkConstant)if(PaletteRead[24u+checkConstant]!=ReadConstant(checkConstant))return false;
    [unroll] for(uint reservedWord=48u;reservedWord<64u;++reservedWord)if(PaletteRead[reservedWord]!=0u)return false;
    if(inner_request.y>=PaletteRead[4])return false;
    uint modeStart=INNER_MODE_OFFSET+inner_request.y*INNER_MODE_WORDS;
    GRepresentative=uint3(PaletteRead[modeStart],PaletteRead[modeStart+1u],PaletteRead[modeStart+2u]);
    uint peak=MaxChannel(GRepresentative);
    if(peak==0u || any(GRepresentative>65535u) || PaletteRead[modeStart+3u]!=peak || PaletteRead[modeStart+4u]>=64u || PaletteRead[modeStart+5u]==0u || PaletteRead[modeStart+5u]>512u || PaletteRead[modeStart+6u]>=512u || PaletteRead[modeStart+7u]!=0u)return false;
    GSearch=search;return true;
}
void PrepareEmit(uint lane) {
    if(lane==0u) {
        GTyped=TypedCurrentValid(CurrentCount,CurrentReferenceCenterTwice)?1u:0u;LoadCurrentGeometry();
        GModeReady=PaletteContextValid()?1u:0u;
        uint2 e,s,d,c;EligibleWrite.GetDimensions(e.x,e.y);SeedsWrite.GetDimensions(s.x,s.y);DomainWrite.GetDimensions(d.x,d.y);CoverWrite.GetDimensions(c.x,c.y);
        GPlaneShape=all(e==contour_source.xw)&&all(s==e)&&all(d==e)&&all(c==e)?1u:0u;
    }
    GroupMemoryBarrierWithGroupSync();
}
void ClearEmitAt(uint2 p) {
    uint2 e,s,d,c;EligibleWrite.GetDimensions(e.x,e.y);SeedsWrite.GetDimensions(s.x,s.y);DomainWrite.GetDimensions(d.x,d.y);CoverWrite.GetDimensions(c.x,c.y);
    if(all(p<e))EligibleWrite[p]=0u;if(all(p<s))SeedsWrite[p]=0u;if(all(p<d))DomainWrite[p]=0u;if(all(p<c))CoverWrite[p]=0u;
}
[numthreads(16,16,1)]
void emit_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    PrepareEmit(lane);
    if(GPlaneShape==0u) {
        if(any(group!=0u))return;
        uint2 e,s,d,c;EligibleWrite.GetDimensions(e.x,e.y);SeedsWrite.GetDimensions(s.x,s.y);DomainWrite.GetDimensions(d.x,d.y);CoverWrite.GetDimensions(c.x,c.y);
        uint2 extent=max(max(e,s),max(d,c));
        [loop]for(uint clearY=dispatch.y;clearY<extent.y;clearY+=16u)[loop]for(uint clearX=dispatch.x;clearX<extent.x;clearX+=16u)ClearEmitAt(uint2(clearX,clearY));
        return;
    }
    if(any(dispatch.xy>=contour_source.xw))return;
    int2 p=int2(dispatch.xy);uint eligible=0u,seed=0u,domain=0u,cover=0u;
    if(GModeReady!=0u) {
        bool inSearch=SourcePixelInRect(p,GSearch);domain=inSearch?inner_request.x+1u:0u;
        cover=SourcePixelInRect(p,CurrentCovers[inner_request.x])?1u:0u;
        float3 color;bool valid=LoadDisplayColor(p,color);
        if(inSearch && valid) {
            uint3 q=PaletteQuantize(color),hue=PaletteHue(q),modeHue=PaletteHue(GRepresentative);
            uint hueDistance=MaxChannel((uint3)abs(int3(hue)-int3(modeHue))),peak=MaxChannel(GRepresentative);
            eligible=2u*MaxChannel(q)>=peak && 20u*hueDistance<=3u*4095u?1u:0u;
            if(eligible!=0u) {
                float probability;bool pValid=CurrentProbabilityEvidence(p,probability);
                if(pValid && (CurrentSupport.Load(int3(p,0))==1u || probability>V2_OCR_ACTIVE_PROBABILITY_THRESHOLD)) {
                    bool2 edges=CurrentSpatialEdges(p,color);uint4 core=CurrentCores[inner_request.x];
                    uint limit=3u*(core.w-core.y)*contour_source.y/(contour_content.w-contour_content.y);
                    bool tangent=(edges.x&&CurrentShortTangent(p,color,int2(0,1),limit)) || (edges.y&&CurrentShortTangent(p,color,int2(1,0),limit));
                    if(tangent && BlackNeighborWitness(p,color,peak))seed=1u;
                }
            }
        }
    }
    EligibleWrite[dispatch.xy]=eligible;SeedsWrite[dispatch.xy]=seed;DomainWrite[dispatch.xy]=domain;CoverWrite[dispatch.xy]=cover;
}

[numthreads(16,16,1)]
void clear_main(uint3 dispatch:SV_DispatchThreadID,uint3 group:SV_GroupID) {
    uint2 bodySize,finalSize;AccumulatedBody.GetDimensions(bodySize.x,bodySize.y);AccumulatedFinal.GetDimensions(finalSize.x,finalSize.y);
    bool canonical=all(contour_source.xw>0u)&&all(contour_source.xw<=V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT)&&all(bodySize==contour_source.xw)&&all(finalSize==bodySize);
    if(canonical){if(all(dispatch.xy<bodySize)){AccumulatedBody[dispatch.xy]=0u;AccumulatedFinal[dispatch.xy]=0u;}return;}
    if(any(group!=0u))return;
    uint2 extent=max(bodySize,finalSize);
    [loop]for(uint clearBodyY=dispatch.y;clearBodyY<extent.y;clearBodyY+=16u)[loop]for(uint clearBodyX=dispatch.x;clearBodyX<extent.x;clearBodyX+=16u){uint2 p=uint2(clearBodyX,clearBodyY);if(all(p<bodySize))AccumulatedBody[p]=0u;if(all(p<finalSize))AccumulatedFinal[p]=0u;}
}
[numthreads(16,16,1)]
void accumulate_main(uint3 dispatch:SV_DispatchThreadID,uint lane:SV_GroupIndex) {
    if(lane==0u) {
        GTyped=TypedCurrentValid(CurrentCount,CurrentReferenceCenterTwice)?1u:0u;LoadCurrentGeometry();GModeReady=PaletteContextValid()?1u:0u;
        uint2 b,f,ab,af;ModeBody.GetDimensions(b.x,b.y);ModeFinal.GetDimensions(f.x,f.y);AccumulatedBody.GetDimensions(ab.x,ab.y);AccumulatedFinal.GetDimensions(af.x,af.y);
        uint parentWords,parentStride;ComponentParent.GetDimensions(parentWords,parentStride);
        if(any(b!=contour_source.xw)||any(f!=b)||any(ab!=b)||any(af!=b)||parentWords!=contour_source.x*contour_source.w+1u||parentStride!=4u)GModeReady=0u;
        else if(ComponentParent[0]!=0u)GModeReady=0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if(GModeReady==0u || any(dispatch.xy>=contour_source.xw))return;
    uint body=ModeBody.Load(int3(dispatch.xy,0)),final=ModeFinal.Load(int3(dispatch.xy,0));
    if(body<=1u && final<=1u) {AccumulatedBody[dispatch.xy]|=body;AccumulatedFinal[dispatch.xy]|=final;}
}

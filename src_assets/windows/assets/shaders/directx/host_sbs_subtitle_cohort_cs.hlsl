// PRIVATE EXPERIMENT: withdrawal from an independently authenticated current baseline S.
// Not part of the published producer manifest. These resources never grant draw authority.
//
// prepare -> members -> filter are separate ordered dispatches. Filter reads NEXT SRVs only
// and publishes persistent state after every old-state reader is finished. Reuse dispatches
// zero groups for all three entries. No CPU flip, verdict, readback or unconditional copy.
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/host_sbs_subtitle_history_compatibility.hlsl"
#include "include/host_sbs_subtitle_cohort_contract.h"

cbuffer SubtitleCohortConstants : register(b0) {
    uint4 cohort_source;
    uint4 cohort_field;
    uint4 cohort_content;
    uint4 cohort_frame;
    uint4 cohort_observation;
    uint4 cohort_control;
};

Texture2D<float4> CurrentSource : register(t0);
StructuredBuffer<float> CurrentP : register(t1);
StructuredBuffer<uint> CurrentOCR : register(t2);
StructuredBuffer<uint> CurrentUIP : register(t3);
Texture2D<float4> PreviousSource : register(t4);
StructuredBuffer<float> PreviousP : register(t5);
StructuredBuffer<uint> PreviousSCT : register(t6);
StructuredBuffer<uint> PreviousState : register(t7);
Texture2D<uint> PreviousOwner : register(t8);
Texture2D<uint> PreviousBirthRGB : register(t9);
Texture2D<uint> BaselineSupport : register(t10);
StructuredBuffer<uint> NextStateRead : register(t11);
Texture2D<uint> NextOwnerRead : register(t12);
Texture2D<uint> NextBirthRGBRead : register(t13);
// Exact previous unfiltered baseline S, paired with PreviousSource/P/SCT.
Texture2D<uint> PreviousBaselineSupport : register(t14);

RWStructuredBuffer<uint> NextState : register(u0);
RWTexture2D<uint> NextOwner : register(u1);
RWTexture2D<uint> NextBirthRGB : register(u2);
RWTexture2D<uint> Eligibility : register(u3);
RWTexture2D<uint> FilteredSupport : register(u4);
RWStructuredBuffer<uint> PersistentStateWrite : register(u5);
RWTexture2D<uint> PersistentOwnerWrite : register(u6);
RWTexture2D<uint> PersistentBirthRGBWrite : register(u7);

groupshared uint GCurrent;
groupshared uint GCount;
groupshared uint GPreviousSnapshot;
groupshared uint GPreviousDuplicate;
groupshared uint GCutEpoch;
groupshared uint GOld;
groupshared uint GDuplicate;
groupshared int GReference;
groupshared uint GReferenceWidth;
groupshared uint GInvalid[256];
groupshared uint GQuiet[256];
groupshared uint GTested[256];
groupshared uint GStable[256];
groupshared uint GLost[256];
groupshared uint GMatches[SBC_CAPACITY];
groupshared uint GPredecessor[SBC_CAPACITY];
groupshared uint GSuccessors[SBC_CAPACITY];
groupshared uint GPhase[SBC_CAPACITY];
groupshared uint GFlags[SBC_CAPACITY];
groupshared uint GId[SBC_CAPACITY];
groupshared uint GSlot;
groupshared uint GNextReadable;

bool Finite(float x) { return (asuint(x) & 0x7f800000u) != 0x7f800000u; }
bool Unit(float x) { return Finite(x) && x >= 0.0f && x <= 1.0f; }
bool ClockLess(uint2 a, uint2 b) { return V2SubtitleHistoryClockLess(a, b); }
uint Entry(uint slot) { return SBC_HEADER_WORDS + slot * SBC_ENTRY_WORDS; }
uint2 Stride() { return max(((cohort_field.xy + 15u) / 16u) * 16u, uint2(16u, 16u)); }
bool Position(int2 p) { return all(p >= 0) && p.x < (int)cohort_source.x && p.y < (int)cohort_source.w; }

bool LoadColor(bool previous, int2 p, out float3 rgb) {
    rgb = 0.0f;
    if (!Position(p)) return false;
    float4 x = previous ? PreviousSource.Load(int3(p, 0)) : CurrentSource.Load(int3(p, 0));
    if (x.a != 1.0f || !Unit(x.r) || !Unit(x.g) || !Unit(x.b)) return false;
    rgb = x.rgb;
    return true;
}

uint PackRGB(float3 x) {
    uint3 q = (uint3)floor(x * 255.0f + 0.5f);
    return q.x | (q.y << 8u) | (q.z << 16u);
}
uint3 UnpackRGB(uint p) { return uint3(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u); }
uint RGBError(uint a, uint b) {
    uint3 d = (uint3)abs(int3(UnpackRGB(a)) - int3(UnpackRGB(b)));
    return max(d.x, max(d.y, d.z));
}
uint OwnerContext(uint owner) { return owner & SBC_OWNER_ID_MASK; }
bool OwnerUnknown(uint owner) { return (owner & SBC_OWNER_UNKNOWN_BIT) != 0u; }
bool BirthAnchorLost(float3 current, uint born) {
    // Retirement tolerates the existing history RGB distance from the immutable
    // birth color. This never qualifies acquisition or changes lifetime membership.
    float3 birth = float3(UnpackRGB(born)) / 255.0f;
    float3 delta = abs(current - birth);
    return max(delta.x, max(delta.y, delta.z)) > V2_SUBTITLE_CONTOUR_HISTORY_RGB_DISTANCE;
}
bool Edge(int2 p, float3 color) {
    const int r = (int)V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RADIUS_SOURCE_PIXELS;
    float3 q;
    if (LoadColor(false, p + int2(r, 0), q) && max(abs(color.r-q.r), max(abs(color.g-q.g), abs(color.b-q.b))) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) return true;
    if (LoadColor(false, p - int2(r, 0), q) && max(abs(color.r-q.r), max(abs(color.g-q.g), abs(color.b-q.b))) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) return true;
    if (LoadColor(false, p + int2(0, r), q) && max(abs(color.r-q.r), max(abs(color.g-q.g), abs(color.b-q.b))) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE) return true;
    return LoadColor(false, p - int2(0, r), q) && max(abs(color.r-q.r), max(abs(color.g-q.g), abs(color.b-q.b))) > V2_SUBTITLE_CONTOUR_SPATIAL_EDGE_RGB_DISTANCE;
}

// Same explicit FP32 operation boundaries as current native contour probability sampling.
bool ProbabilityAt(int2 p, out float value) {
    value = 0.0f;
    precise float2 ratio = float2((float)V2_OCR_OUTPUT_WIDTH / (float)cohort_source.x,
                                  (float)V2_OCR_OUTPUT_HEIGHT / (float)cohort_source.w);
    precise float2 centered = float2(p) + 0.5f;
    precise float2 scaled = centered * ratio;
    precise float2 xy = scaled - 0.5f;
    int2 lo = int2(floor(xy));
    precise float2 f = xy - floor(xy);
    int2 a = clamp(lo, int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH-1u, V2_OCR_OUTPUT_HEIGHT-1u));
    int2 b = clamp(lo + 1, int2(0, 0), int2(V2_OCR_OUTPUT_WIDTH-1u, V2_OCR_OUTPUT_HEIGHT-1u));
    float tl = CurrentP[a.y * V2_OCR_OUTPUT_WIDTH + a.x];
    float tr = CurrentP[a.y * V2_OCR_OUTPUT_WIDTH + b.x];
    float bl = CurrentP[b.y * V2_OCR_OUTPUT_WIDTH + a.x];
    float br = CurrentP[b.y * V2_OCR_OUTPUT_WIDTH + b.x];
    if (!Unit(tl) || !Unit(tr) || !Unit(bl) || !Unit(br)) return false;
    precise float ud = tr-tl;
    precise float up = ud*f.x;
    precise float upper = tl+up;
    precise float ld = br-bl;
    precise float lp = ld*f.x;
    precise float lower = bl+lp;
    precise float vd = lower-upper;
    precise float vp = vd*f.y;
    value = upper+vp;
    return true;
}

uint4 OCRRect(uint slot, bool cover) {
    uint b = (cover ? V2_OCR_FINAL_BOX_OFFSET : V2_OCR_RAW_BOX_OFFSET) + slot * V2_OCR_BOX_WORD_COUNT;
    return uint4(CurrentOCR[b], CurrentOCR[b+1u], CurrentOCR[b+2u], CurrentOCR[b+3u]);
}
uint4 OldRect(uint slot, uint offset) {
    uint b = Entry(slot)+offset;
    return uint4(PreviousState[b], PreviousState[b+1u], PreviousState[b+2u], PreviousState[b+3u]);
}
bool RectValid(uint4 r) {
    return r.x < r.z && r.y < r.w && all(r.xy >= cohort_content.xy) && all(r.zw <= cohort_content.zw);
}
bool Contains(int2 p, uint4 r) {
    uint2 a = uint2(2u*cohort_content.x*cohort_source.x + (2u*(uint)p.x+1u)*(cohort_content.z-cohort_content.x),
                   2u*cohort_content.y*cohort_source.y + (2u*((uint)p.y+cohort_source.z)+1u)*(cohort_content.w-cohort_content.y));
    return a.x >= 2u*r.x*cohort_source.x && a.x < 2u*r.z*cohort_source.x &&
           a.y >= 2u*r.y*cohort_source.y && a.y < 2u*r.w*cohort_source.y;
}
uint Kind(uint slot) { return CurrentOCR[V2_OCR_RAW_BOX_OFFSET+slot*V2_OCR_BOX_WORD_COUNT+5u] & V2_OCR_BOX_FLAG_RIBBON; }

bool ClassifierReference() {
    GReference = (int)cohort_source.x;
    GReferenceWidth = cohort_source.x;
    uint b = V2_OCR_CLASSIFIER_HISTORY_OFFSET;
    bool empty = true;
    [loop] for (uint cohortLoop0_i=0u; cohortLoop0_i<V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT; ++cohortLoop0_i) empty = empty && CurrentOCR[b+cohortLoop0_i]==0u;
    if (empty) return true;
    if (CurrentOCR[b] != V2_OCR_CLASSIFIER_HISTORY_TAG || CurrentOCR[b+1u] != V2_OCR_CLASSIFIER_HISTORY_SCHEMA ||
        CurrentOCR[b+2u] != V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS ||
        any(uint2(CurrentOCR[b+4u],CurrentOCR[b+5u]) != cohort_observation.xy)) return false;
    [unroll] for (uint cohortLoop1_i=0u; cohortLoop1_i<4u; ++cohortLoop1_i) {
        if (CurrentOCR[b+6u+cohortLoop1_i] != cohort_frame[cohortLoop1_i] || CurrentOCR[b+10u+cohortLoop1_i] != cohort_source[cohortLoop1_i] ||
            CurrentOCR[b+16u+cohortLoop1_i] != cohort_content[cohortLoop1_i]) return false;
    }
    if (CurrentOCR[b+14u]!=cohort_field.x || CurrentOCR[b+15u]!=cohort_field.y) return false;
    int center = asint(CurrentOCR[b+20u]);
    uint width = CurrentOCR[b+21u];
    if (width==0u || width>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT || center<-(int)width || center>(int)width) return false;
    GReference=center; GReferenceWidth=width;
    return true;
}

bool CurrentValid() {
    uint n,s,w,h,ow,oh,bw,bh;
    CurrentSource.GetDimensions(w,h);
    BaselineSupport.GetDimensions(bw,bh);
    NextOwner.GetDimensions(ow,oh);
    if (cohort_control.x!=1u || cohort_control.y!=SBC_SCHEMA || any(cohort_control.zw!=0u) ||
        cohort_observation.z>1u || cohort_observation.w!=0u ||
        !any(cohort_frame.xy!=0u) || !any(cohort_observation.xy!=0u) ||
        any(cohort_source.xy==0u) || any(cohort_source.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        any(cohort_field.xy==0u) || any(cohort_field.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        !V2SubtitleOcrFieldIsCalibrated(cohort_field.x,cohort_field.y) ||
        w!=cohort_source.x || h!=cohort_source.w || bw!=w || bh!=h || ow!=w || oh!=h ||
        cohort_content.x>=cohort_content.z || cohort_content.y>=cohort_content.w ||
        any(cohort_content.zw>cohort_field.xy)) return false;
    NextBirthRGB.GetDimensions(ow,oh);
    if (ow!=w || oh!=h) return false;
    NextState.GetDimensions(n,s);
    if (n!=SBC_STATE_WORDS || s!=4u) return false;
    CurrentP.GetDimensions(n,s);
    if (n!=V2_OCR_OUTPUT_WIDTH*V2_OCR_OUTPUT_HEIGHT || s!=4u) return false;
    CurrentOCR.GetDimensions(n,s);
    if (n!=V2_OCR_RECORD_WORD_COUNT || s!=4u) return false;
    CurrentUIP.GetDimensions(n,s);
    if (n!=V2_SUBTITLE_PLANE_STATE_WORD_COUNT || s!=4u) return false;
    uint top,height,rt,rb;
    if (!V2SubtitleOcrComputeCrop(cohort_source.x,cohort_source.y,top,height) ||
        top!=cohort_source.z || height!=cohort_source.w ||
        !V2SubtitleOcrProjectContentRowCeil(cohort_source.x,cohort_source.y,cohort_field.x,cohort_field.y,cohort_content,V2_OCR_SAFE_ROW_TOP,rt) ||
        !V2SubtitleOcrProjectContentRowCeil(cohort_source.x,cohort_source.y,cohort_field.x,cohort_field.y,cohort_content,V2_OCR_SAFE_ROW_BOTTOM,rb) ||
        cohort_field.z!=rt || cohort_field.w!=rb || rt>=rb) return false;
    uint count=CurrentOCR[4u];
    if (CurrentOCR[0u]!=V2_OCR_RECORD_SCHEMA || CurrentOCR[1u]!=V2_OCR_RECORD_TAG || CurrentOCR[2u]!=1u ||
        CurrentOCR[3u]!=count || count>SBC_CAPACITY) return false;
    if (CurrentUIP[0u]!=V2_SUBTITLE_PLANE_STATE_SCHEMA || CurrentUIP[1u]!=V2_SUBTITLE_PLANE_STATE_TAG ||
        (CurrentUIP[V2_SUBTITLE_PLANE_FLAGS_WORD]&~V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG)!=0u ||
        CurrentUIP[V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD]!=(1u<<count)-1u ||
        (count!=0u && CurrentUIP[V2_SUBTITLE_PLANE_FLAGS_WORD]!=V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG) ||
        (CurrentUIP[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD]&~((1u<<count)-1u))!=0u ||
        CurrentUIP[V2_SUBTITLE_PLANE_WARMUP_WORD]!=0u || CurrentUIP[V2_SUBTITLE_PLANE_RESERVED_OFFSET]!=0u) return false;
    [unroll] for (uint cohortLoop2_k=0u; cohortLoop2_k<2u; ++cohortLoop2_k) {
        if (CurrentOCR[5u+cohortLoop2_k]!=cohort_frame[cohortLoop2_k] || CurrentOCR[7u+cohortLoop2_k]!=cohort_frame[2u+cohortLoop2_k] ||
            CurrentOCR[9u+cohortLoop2_k]!=cohort_source[cohortLoop2_k] || CurrentOCR[11u+cohortLoop2_k]!=cohort_field[cohortLoop2_k] || CurrentOCR[13u+cohortLoop2_k]!=cohort_field[2u+cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_FRAME_OFFSET+cohortLoop2_k]!=cohort_frame[cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET+cohortLoop2_k]!=cohort_frame[2u+cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET+cohortLoop2_k]!=cohort_observation[cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET+cohortLoop2_k]!=cohort_source[cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET+cohortLoop2_k]!=cohort_field[cohortLoop2_k] ||
            CurrentUIP[V2_SUBTITLE_PLANE_ROI_OFFSET+cohortLoop2_k]!=cohort_field[2u+cohortLoop2_k]) return false;
    }
    if (CurrentOCR[15u]!=0u) return false;
    [unroll] for (uint cohortLoop3_k=0u; cohortLoop3_k<4u; ++cohortLoop3_k) if (CurrentUIP[V2_SUBTITLE_PLANE_CONTENT_OFFSET+cohortLoop3_k]!=cohort_content[cohortLoop3_k]) return false;
    [loop] for (uint cohortLoop4_slot=0u; cohortLoop4_slot<SBC_CAPACITY; ++cohortLoop4_slot) {
        uint a=V2_OCR_RAW_BOX_OFFSET+cohortLoop4_slot*V2_OCR_BOX_WORD_COUNT;
        uint b=V2_OCR_FINAL_BOX_OFFSET+cohortLoop4_slot*V2_OCR_BOX_WORD_COUNT;
        if (cohortLoop4_slot>=count) {
            [unroll] for (uint cohortLoop5_k=0u; cohortLoop5_k<V2_OCR_BOX_WORD_COUNT; ++cohortLoop5_k) if (CurrentOCR[a+cohortLoop5_k]!=0u || CurrentOCR[b+cohortLoop5_k]!=0u) return false;
            continue;
        }
        uint4 core=OCRRect(cohortLoop4_slot,false),cover=OCRRect(cohortLoop4_slot,true);
        float score=asfloat(CurrentOCR[a+4u]);
        if (!RectValid(core) || !RectValid(cover) || any(cover.xy>core.xy) || any(cover.zw<core.zw) ||
            core.y<rt || core.w>rb || !Unit(score) || score<V2_OCR_MIN_MEAN_SCORE ||
            (CurrentOCR[a+5u]&~V2_OCR_BOX_KNOWN_FLAGS)!=0u || CurrentOCR[a+6u]==0u ||
            CurrentOCR[a+6u]>V2_OCR_OUTPUT_WIDTH || CurrentOCR[a+7u]>=CurrentOCR[a+6u]) return false;
        [unroll] for (uint cohortLoop6_k=4u; cohortLoop6_k<V2_OCR_BOX_WORD_COUNT; ++cohortLoop6_k) if (CurrentOCR[a+cohortLoop6_k]!=CurrentOCR[b+cohortLoop6_k]) return false;
        if (Kind(cohortLoop4_slot)==0u) { if (cover.y<rt || cover.w>rb) return false; }
        else if (cover.x!=cohort_content.x || cover.z!=cohort_content.z || cover.w!=cohort_content.w || cover.y<rt) return false;
    }
    if (!ClassifierReference()) return false;
    // Read the GPU-owned cut only after this UIP's complete current observation
    // and geometry joins succeed. No CPU cut mirror participates in authority.
    GCutEpoch=CurrentUIP[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD];
    GCount=count;
    return true;
}

bool PreviousSnapshotValid() {
    uint w,h,n,s;
    PreviousSource.GetDimensions(w,h);
    PreviousP.GetDimensions(n,s);
    if (w!=cohort_source.x || h!=cohort_source.w || n!=V2_OCR_OUTPUT_WIDTH*V2_OCR_OUTPUT_HEIGHT || s!=4u) return false;
    PreviousBaselineSupport.GetDimensions(w,h);
    if (w!=cohort_source.x || h!=cohort_source.w) return false;
    bool duplicate;
    bool result=V2SubtitleHistoryCompatible(PreviousSCT,cohort_source,cohort_field,cohort_content,
        cohort_frame,cohort_observation.xy,cohort_observation.z,GCutEpoch,duplicate);
    GPreviousDuplicate=result && duplicate ? 1u:0u;
    return result;
}

bool PreviousStateValid() {
    uint n,s,w,h;
    PreviousState.GetDimensions(n,s);
    PreviousOwner.GetDimensions(w,h);
    if (n!=SBC_STATE_WORDS || s!=4u || w!=cohort_source.x || h!=cohort_source.w || cohort_observation.z!=0u) return false;
    PreviousBirthRGB.GetDimensions(w,h);
    if (w!=cohort_source.x || h!=cohort_source.w || PreviousState[SBC_H_TAG]!=SBC_TAG || PreviousState[SBC_H_SCHEMA]!=SBC_SCHEMA ||
        PreviousState[SBC_H_VALID]!=1u || PreviousState[SBC_H_CUT]!=GCutEpoch ||
        PreviousState[SBC_H_COUNT]>SBC_CAPACITY || PreviousState[SBC_H_NEXT_ID]==0u ||
        PreviousState[SBC_H_NEXT_ID]>SBC_OWNER_ID_LIMIT || PreviousState[SBC_H_EPOCH]==0u ||
        (PreviousState[SBC_H_FLAGS]&~SBC_HEADER_KNOWN_FLAGS)!=0u ||
        PreviousState[SBC_H_REFERENCE]!=asuint(GReference) || PreviousState[SBC_H_REFERENCE+1u]!=GReferenceWidth) return false;
    [unroll] for (uint cohortLoop7_k=0u;cohortLoop7_k<2u;++cohortLoop7_k) {
        if (PreviousState[SBC_H_GENERATION+cohortLoop7_k]!=cohort_frame[2u+cohortLoop7_k] || PreviousState[SBC_H_SOURCE+cohortLoop7_k]!=cohort_source[cohortLoop7_k] ||
            PreviousState[SBC_H_CROP+cohortLoop7_k]!=cohort_source[2u+cohortLoop7_k] || PreviousState[SBC_H_FIELD+cohortLoop7_k]!=cohort_field[cohortLoop7_k]) return false;
    }
    [unroll] for (uint cohortLoop8_k=0u;cohortLoop8_k<4u;++cohortLoop8_k) if (PreviousState[SBC_H_CONTENT+cohortLoop8_k]!=cohort_content[cohortLoop8_k]) return false;
    [unroll] for (uint cohortLoop9_k=SBC_H_RESERVED;cohortLoop9_k<SBC_HEADER_WORDS;++cohortLoop9_k) if (PreviousState[cohortLoop9_k]!=0u) return false;
    uint2 oldFrame=uint2(PreviousState[SBC_H_FRAME],PreviousState[SBC_H_FRAME+1u]);
    uint2 oldTime=uint2(PreviousState[SBC_H_CLOCK],PreviousState[SBC_H_CLOCK+1u]);
    if (!any(oldFrame!=0u) || !any(oldTime!=0u)) return false;
    bool duplicate=all(oldFrame==cohort_frame.xy) && all(oldTime==cohort_observation.xy);
    if (!duplicate && !(ClockLess(oldFrame,cohort_frame.xy) && ClockLess(oldTime,cohort_observation.xy))) return false;
    // Old pixels, source and P must name exactly the same completed observation.
    if (GPreviousSnapshot==0u || any(oldFrame!=uint2(PreviousSCT[4u],PreviousSCT[5u])) ||
        any(oldTime!=uint2(PreviousSCT[20u],PreviousSCT[21u]))) return false;
    uint count=PreviousState[SBC_H_COUNT];
    [loop] for (uint cohortLoop10_slot=0u;cohortLoop10_slot<SBC_CAPACITY;++cohortLoop10_slot) {
        uint b=Entry(cohortLoop10_slot);
        if (cohortLoop10_slot>=count) { [unroll] for (uint cohortLoop11_k=0u;cohortLoop11_k<SBC_ENTRY_WORDS;++cohortLoop11_k) if (PreviousState[b+cohortLoop11_k]!=0u) return false; continue; }
        uint phase=PreviousState[b+SBC_E_PHASE],id=PreviousState[b+SBC_E_ID];
        if (phase>SBC_PHASE_SUSPENDED || (PreviousState[b+SBC_E_FLAGS]&~SBC_ENTRY_KNOWN_FLAGS)!=0u || PreviousState[b+SBC_E_KIND]>1u ||
            !RectValid(OldRect(cohortLoop10_slot,SBC_E_LAST_CORE)) || !RectValid(OldRect(cohortLoop10_slot,SBC_E_LAST_COVER)) ||
            PreviousState[b+SBC_E_RESERVED]!=0u || PreviousState[b+SBC_E_RESERVED+1u]!=0u) return false;
        if (phase==SBC_PHASE_UNKNOWN) { if (id!=0u) return false; }
        else {
            if (id==0u || id>=SBC_OWNER_ID_LIMIT || id>=PreviousState[SBC_H_NEXT_ID] || !RectValid(OldRect(cohortLoop10_slot,SBC_E_BIRTH_CORE)) ||
                !RectValid(OldRect(cohortLoop10_slot,SBC_E_BIRTH_COVER)) || PreviousState[b+SBC_E_REFERENCE_FLAGS]!=SBC_REFERENCE_KNOWN_FLAGS) return false;
            if (phase==SBC_PHASE_SUSPENDED &&
                ((PreviousState[b+SBC_E_FLAGS]&SBC_ENTRY_BLOCKING_FLAGS)==0u ||
                 (PreviousState[b+SBC_E_FLAGS]&SBC_ENTRY_NEW_BIRTH)!=0u)) return false;
            [unroll] for (uint cohortLoop12_prior=0u;cohortLoop12_prior<SBC_CAPACITY;++cohortLoop12_prior) if (cohortLoop12_prior<cohortLoop10_slot && PreviousState[Entry(cohortLoop12_prior)+SBC_E_ID]==id) return false;
        }
    }
    GDuplicate=duplicate?1u:0u;
    return true;
}

bool Association(uint4 current, uint oldSlot, uint kind) {
    uint b=Entry(oldSlot);
    uint phase=PreviousState[b+SBC_E_PHASE],flags=PreviousState[b+SBC_E_FLAGS];
    // Suspended provenance can associate geometrically without regaining qualification.
    // Invalid-reference/ambiguous-pixel suspensions cannot carry a trusted old owner.
    bool suspended=phase==SBC_PHASE_SUSPENDED && flags==SBC_ENTRY_UNCERTAIN_REPLACEMENT;
    if (PreviousState[b+SBC_E_PHASE]==SBC_PHASE_UNKNOWN || PreviousState[b+SBC_E_ID]==0u ||
        (!suspended && (flags&SBC_ENTRY_BLOCKING_FLAGS)!=0u) ||
        (phase==SBC_PHASE_SUSPENDED && !suspended) || PreviousState[b+SBC_E_KIND]!=kind) return false;
    uint4 fixed=OldRect(oldSlot,SBC_E_BIRTH_CORE);
    uint top=max(current.y,fixed.y),bottom=min(current.w,fixed.w);
    return top<bottom && (bottom-top)*V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_DENOMINATOR>=
        (max(current.w,fixed.w)-min(current.y,fixed.y))*V2_OCR_LINE_GROUP_MIN_VERTICAL_IOU_NUMERATOR &&
        max(current.x,fixed.x)<min(current.z,fixed.z);
}

int FloorDiv(int n,int d) { return n>=0?n/d:-((-n+d-1)/d); }
int CeilDiv(int n,int d) { return n>=0?(n+d-1)/d:-((-n)/d); }
uint4 Query(uint4 cover) {
    // field<=5120 and canonical crop<=854 bound the reduced Y numerator*320 < int32.
    int cw=(int)(cohort_content.z-cohort_content.x),ch=(int)(cohort_content.w-cohort_content.y);
    int den=ch*(int)cohort_source.w;
    int l=((int)cover.x-(int)cohort_content.x)*(int)V2_OCR_OUTPUT_WIDTH;
    int r=((int)cover.z-(int)cohort_content.x)*(int)V2_OCR_OUTPUT_WIDTH;
    int t=(((int)cover.y-(int)cohort_content.y)*(int)cohort_source.y-(int)cohort_source.z*ch)*(int)V2_OCR_OUTPUT_HEIGHT;
    int b=(((int)cover.w-(int)cohort_content.y)*(int)cohort_source.y-(int)cohort_source.z*ch)*(int)V2_OCR_OUTPUT_HEIGHT;
    return uint4(clamp(FloorDiv(2*l-cw,2*cw),0,(int)V2_OCR_OUTPUT_WIDTH),
                 clamp(FloorDiv(2*t-den,2*den),0,(int)V2_OCR_OUTPUT_HEIGHT),
                 clamp(CeilDiv(2*r+cw,2*cw),0,(int)V2_OCR_OUTPUT_WIDTH),
                 clamp(CeilDiv(2*b+den,2*den),0,(int)V2_OCR_OUTPUT_HEIGHT));
}
int2 Sample(uint4 core,uint sampleIndex) {
    uint gx=sampleIndex%64u,gy=sampleIndex/64u;
    uint x=(((core.x-cohort_content.x)*128u+(2u*gx+1u)*(core.z-core.x))*cohort_source.x)/
        (128u*(cohort_content.z-cohort_content.x));
    uint y=(((core.y-cohort_content.y)*16u+(2u*gy+1u)*(core.w-core.y))*cohort_source.y)/
        (16u*(cohort_content.w-cohort_content.y));
    return int2(min(x,cohort_source.x-1u),clamp((int)y-(int)cohort_source.z,0,(int)cohort_source.w-1));
}
void StoreRect(uint b,uint offset,uint4 r) { [unroll] for (uint cohortLoop13_k=0u;cohortLoop13_k<4u;++cohortLoop13_k) NextState[b+offset+cohortLoop13_k]=r[cohortLoop13_k]; }
void Header(uint flags) {
    NextState[SBC_H_TAG]=SBC_TAG; NextState[SBC_H_SCHEMA]=SBC_SCHEMA; NextState[SBC_H_VALID]=GCurrent;
    NextState[SBC_H_CUT]=GCutEpoch;
    [unroll] for (uint cohortLoop14_k=0u;cohortLoop14_k<2u;++cohortLoop14_k) {
        NextState[SBC_H_FRAME+cohortLoop14_k]=cohort_frame[cohortLoop14_k]; NextState[SBC_H_GENERATION+cohortLoop14_k]=cohort_frame[2u+cohortLoop14_k];
        NextState[SBC_H_CLOCK+cohortLoop14_k]=cohort_observation[cohortLoop14_k]; NextState[SBC_H_SOURCE+cohortLoop14_k]=cohort_source[cohortLoop14_k];
        NextState[SBC_H_CROP+cohortLoop14_k]=cohort_source[2u+cohortLoop14_k]; NextState[SBC_H_FIELD+cohortLoop14_k]=cohort_field[cohortLoop14_k];
    }
    [unroll] for (uint cohortLoop15_k=0u;cohortLoop15_k<4u;++cohortLoop15_k) NextState[SBC_H_CONTENT+cohortLoop15_k]=cohort_content[cohortLoop15_k];
    NextState[SBC_H_REFERENCE]=asuint(GReference); NextState[SBC_H_REFERENCE+1u]=GReferenceWidth;
    NextState[SBC_H_NEXT_ID]=GOld!=0u?PreviousState[SBC_H_NEXT_ID]:1u;
    uint epoch=GOld!=0u?PreviousState[SBC_H_EPOCH]:1u;
    if (GOld==0u) {
        uint words,stride;PreviousState.GetDimensions(words,stride);
        if (words==SBC_STATE_WORDS && stride==4u && PreviousState[SBC_H_TAG]==SBC_TAG && PreviousState[SBC_H_SCHEMA]==SBC_SCHEMA) {
            uint oldEpoch=PreviousState[SBC_H_EPOCH];
            epoch=oldEpoch<0xffffffffu?oldEpoch+1u:1u;
        }
    }
    NextState[SBC_H_EPOCH]=epoch; NextState[SBC_H_COUNT]=GCurrent!=0u?GCount:0u;
    NextState[SBC_H_FLAGS]=flags;
}

void BeginBirth(uint outputEntry,uint4 core,uint4 cover) {
    uint id=NextState[SBC_H_NEXT_ID];
    if (id==0u || id>=SBC_OWNER_ID_LIMIT) {
        NextState[SBC_H_FLAGS]|=SBC_HEADER_ID_EXHAUSTED;
        NextState[outputEntry+SBC_E_FLAGS]=SBC_ENTRY_INVALID_REFERENCE;
        return;
    }
    NextState[SBC_H_NEXT_ID]=id+1u;
    NextState[outputEntry+SBC_E_ID]=id;
    NextState[outputEntry+SBC_E_PHASE]=SBC_PHASE_PENDING;
    NextState[outputEntry+SBC_E_FLAGS]=SBC_ENTRY_NEW_BIRTH;
    NextState[outputEntry+SBC_E_PREVIOUS_SLOT]=SBC_INVALID_SLOT;
    NextState[outputEntry+SBC_E_TESTED_ANCHORS]=0u;
    NextState[outputEntry+SBC_E_STABLE_ANCHORS]=0u;
    StoreRect(outputEntry,SBC_E_BIRTH_CORE,core);
    StoreRect(outputEntry,SBC_E_BIRTH_COVER,cover);
    [unroll] for (uint birthStampComponent=0u;birthStampComponent<2u;++birthStampComponent) {
        NextState[outputEntry+SBC_E_BIRTH_FRAME+birthStampComponent]=cohort_frame[birthStampComponent];
        NextState[outputEntry+SBC_E_BIRTH_CLOCK+birthStampComponent]=cohort_observation[birthStampComponent];
        NextState[outputEntry+SBC_E_REFERENCE_FRAME+birthStampComponent]=PreviousSCT[4u+birthStampComponent];
    }
    NextState[outputEntry+SBC_E_REFERENCE_FLAGS]=SBC_REFERENCE_KNOWN_FLAGS;
}

void RetainBirth(uint outputEntry,uint oldSlot,uint phase) {
    uint oldEntry=Entry(oldSlot);
    NextState[outputEntry+SBC_E_ID]=PreviousState[oldEntry+SBC_E_ID];
    NextState[outputEntry+SBC_E_PHASE]=phase;
    NextState[outputEntry+SBC_E_FLAGS]=phase==SBC_PHASE_SUSPENDED?SBC_ENTRY_UNCERTAIN_REPLACEMENT:0u;
    NextState[outputEntry+SBC_E_PREVIOUS_SLOT]=oldSlot;
    StoreRect(outputEntry,SBC_E_BIRTH_CORE,OldRect(oldSlot,SBC_E_BIRTH_CORE));
    StoreRect(outputEntry,SBC_E_BIRTH_COVER,OldRect(oldSlot,SBC_E_BIRTH_COVER));
    [unroll] for (uint retainedBirthComponent=0u;retainedBirthComponent<4u;++retainedBirthComponent)
        NextState[outputEntry+SBC_E_BIRTH_FRAME+retainedBirthComponent]=PreviousState[oldEntry+SBC_E_BIRTH_FRAME+retainedBirthComponent];
    [unroll] for (uint retainedReferenceComponent=0u;retainedReferenceComponent<3u;++retainedReferenceComponent)
        NextState[outputEntry+SBC_E_REFERENCE_FRAME+retainedReferenceComponent]=PreviousState[oldEntry+SBC_E_REFERENCE_FRAME+retainedReferenceComponent];
    if (phase==SBC_PHASE_SUSPENDED) NextState[outputEntry+SBC_E_STABLE_ANCHORS]=0u;
}

[numthreads(256,1,1)]
void prepare_main(uint3 thread:SV_GroupThreadID) {
    uint lane=thread.x,n,s; NextState.GetDimensions(n,s);
    if (n!=SBC_STATE_WORDS || s!=4u) return;
    [loop] for (uint cohortLoop16_word=lane;cohortLoop16_word<SBC_STATE_WORDS;cohortLoop16_word+=256u) NextState[cohortLoop16_word]=0u;
    AllMemoryBarrierWithGroupSync();
    if (lane==0u) {
        GCount=0u;GPreviousSnapshot=0u;GPreviousDuplicate=0u;GCutEpoch=0u;GOld=0u;GDuplicate=0u;
        GReference=(int)cohort_source.x;GReferenceWidth=cohort_source.x;
        GCurrent=CurrentValid()?1u:0u;
        if (GCurrent!=0u) GPreviousSnapshot=PreviousSnapshotValid()?1u:0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint invalid=0u;
    if (GCurrent!=0u) {
        [loop] for (uint cohortLoop17_i=lane;cohortLoop17_i<V2_OCR_OUTPUT_WIDTH*V2_OCR_OUTPUT_HEIGHT;cohortLoop17_i+=256u) {
            if (!Unit(CurrentP[cohortLoop17_i])) invalid|=1u;
            if (GPreviousSnapshot!=0u && !Unit(PreviousP[cohortLoop17_i])) invalid|=2u;
            if (GPreviousDuplicate!=0u && asuint(CurrentP[cohortLoop17_i])!=asuint(PreviousP[cohortLoop17_i])) invalid|=1u;
        }
        // Previous baseline S is a raw snapshot input, never this qualifier's output.
        // Validate every pixel before it can distinguish a new support seed.
        [loop] for (uint previousSupportIndex=lane;previousSupportIndex<cohort_source.x*cohort_source.w;previousSupportIndex+=256u) {
            int2 supportPosition=int2(previousSupportIndex%cohort_source.x,previousSupportIndex/cohort_source.x);
            uint currentSupport=BaselineSupport.Load(int3(supportPosition,0));
            if (currentSupport>1u) invalid|=1u;
            if (GPreviousSnapshot!=0u) {
                uint previousSupport=PreviousBaselineSupport.Load(int3(supportPosition,0));
                if (previousSupport>1u) invalid|=3u;
                if (GPreviousDuplicate!=0u && currentSupport!=previousSupport) invalid|=1u;
            }
        }
    }
    GInvalid[lane]=invalid;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint cohortLoop18_step=128u;cohortLoop18_step>0u;cohortLoop18_step>>=1u) {
        if (lane<cohortLoop18_step) GInvalid[lane]|=GInvalid[lane+cohortLoop18_step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0u) {
        if ((GInvalid[0]&1u)!=0u) GCurrent=0u;
        if ((GInvalid[0]&2u)!=0u) GPreviousSnapshot=0u;
        if (GCurrent!=0u) GOld=PreviousStateValid()?1u:0u;
        if (GOld==0u) GDuplicate=0u;
        uint flags=GCurrent!=0u?SBC_HEADER_P_MAP_VALID:SBC_HEADER_CURRENT_INVALID;
        if (GOld!=0u) flags|=SBC_HEADER_OLD_VALID;
        if (GPreviousSnapshot!=0u) flags|=SBC_HEADER_PREVIOUS_SNAPSHOT_VALID;
        if (GDuplicate!=0u) flags|=SBC_HEADER_DUPLICATE;
        Header(flags);
        [unroll] for (uint cohortLoop19_i=0u;cohortLoop19_i<SBC_CAPACITY;++cohortLoop19_i) { GMatches[cohortLoop19_i]=0u;GPredecessor[cohortLoop19_i]=SBC_INVALID_SLOT;GSuccessors[cohortLoop19_i]=0u; }
        if (GCurrent!=0u && GOld!=0u) {
            [loop] for (uint cohortLoop20_now=0u;cohortLoop20_now<GCount;++cohortLoop20_now) {
                [loop] for (uint cohortLoop21_old=0u;cohortLoop21_old<PreviousState[SBC_H_COUNT];++cohortLoop21_old) {
                    if (Association(OCRRect(cohortLoop20_now,false),cohortLoop21_old,Kind(cohortLoop20_now))) { ++GMatches[cohortLoop20_now];GPredecessor[cohortLoop20_now]=cohortLoop21_old;++GSuccessors[cohortLoop21_old]; }
                }
            }
        }
    }
    AllMemoryBarrierWithGroupSync();
    if (GCurrent==0u) return;
    [loop] for (uint cohortLoop22_slot=0u;cohortLoop22_slot<GCount;++cohortLoop22_slot) {
        if (lane==0u) {
            GSlot=GMatches[cohortLoop22_slot]==1u?GPredecessor[cohortLoop22_slot]:SBC_INVALID_SLOT;
            if (GSlot!=SBC_INVALID_SLOT && GSuccessors[GSlot]!=1u) GSlot=SBC_INVALID_SLOT;
        }
        GroupMemoryBarrierWithGroupSync();
        uint quiet=0u,tested=0u,stable=0u,lost=0u;
        uint4 query=Query(OCRRect(cohortLoop22_slot,true));
        uint width=query.z-query.x,area=width*(query.w-query.y);
        if (GPreviousSnapshot!=0u && GPreviousDuplicate==0u && width!=0u) {
            [loop] for (uint cohortLoop23_k=lane;cohortLoop23_k<area;cohortLoop23_k+=256u) {
                uint2 p=query.xy+uint2(cohortLoop23_k%width,cohortLoop23_k/width);
                if (PreviousP[p.y*V2_OCR_OUTPUT_WIDTH+p.x]>V2_OCR_ACTIVE_PROBABILITY_THRESHOLD) quiet=1u;
            }
        } else quiet=1u;
        if (GSlot!=SBC_INVALID_SLOT) {
            uint id=PreviousState[Entry(GSlot)+SBC_E_ID]; uint4 fixed=OldRect(GSlot,SBC_E_BIRTH_CORE);
            [unroll] for (uint cohortLoop24_k=0u;cohortLoop24_k<2u;++cohortLoop24_k) {
                int2 p=Sample(fixed,lane+cohortLoop24_k*256u);
                uint born=PreviousBirthRGB.Load(int3(p,0));
                if (PreviousOwner.Load(int3(p,0))!=id || (born&~SBC_BIRTH_KNOWN_MASK)!=0u || (born&SBC_BIRTH_ANCHOR_BIT)==0u) continue;
                ++tested;
                float3 color,previous; float probability;
                bool cv=LoadColor(false,p,color),pv=LoadColor(true,p,previous);
                if (!cv || BirthAnchorLost(color,born)) ++lost;
                // Qualification remains strict: a tolerant retirement match cannot
                // acquire an anchor, add age or admit a new member.
                if (cv && pv && RGBError(PackRGB(color),born)<=SBC_ACQUISITION_RGB8_MAX_DELTA &&
                    RGBError(PackRGB(color),PackRGB(previous))<=SBC_ACQUISITION_RGB8_MAX_DELTA &&
                    ProbabilityAt(p,probability) && probability>V2_OCR_ACTIVE_PROBABILITY_THRESHOLD && Edge(p,color)) ++stable;
            }
        }
        GQuiet[lane]=quiet;GTested[lane]=tested;GStable[lane]=stable;GLost[lane]=lost;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint cohortLoop25_step=128u;cohortLoop25_step>0u;cohortLoop25_step>>=1u) {
            if (lane<cohortLoop25_step) { GQuiet[lane]|=GQuiet[lane+cohortLoop25_step];GTested[lane]+=GTested[lane+cohortLoop25_step];GStable[lane]+=GStable[lane+cohortLoop25_step];GLost[lane]+=GLost[lane+cohortLoop25_step]; }
            GroupMemoryBarrierWithGroupSync();
        }
        // Local new support is handled by typed UNKNOWN seeds over the full native
        // grid in MEMBERS. It does not suspend a proven old line or gain its age.
        if (lane==0u) {
            uint b=Entry(cohortLoop22_slot);uint4 core=OCRRect(cohortLoop22_slot,false),cover=OCRRect(cohortLoop22_slot,true);
            StoreRect(b,SBC_E_LAST_CORE,core);StoreRect(b,SBC_E_LAST_COVER,cover);
            NextState[b+SBC_E_KIND]=Kind(cohortLoop22_slot);NextState[b+SBC_E_PREVIOUS_SLOT]=SBC_INVALID_SLOT;
            NextState[b+SBC_E_TESTED_ANCHORS]=GTested[0];NextState[b+SBC_E_STABLE_ANCHORS]=GStable[0];
            bool ambiguous=GMatches[cohortLoop22_slot]>1u || (GMatches[cohortLoop22_slot]==1u && GSuccessors[GPredecessor[cohortLoop22_slot]]!=1u);
            if (ambiguous) { NextState[b+SBC_E_FLAGS]=SBC_ENTRY_AMBIGUOUS_PIXELS;NextState[SBC_H_FLAGS]|=SBC_HEADER_AMBIGUOUS_ASSOCIATION; }
            else if (GSlot!=SBC_INVALID_SLOT) {
                uint old=Entry(GSlot);
                bool suspended=PreviousState[old+SBC_E_PHASE]==SBC_PHASE_SUSPENDED;
                bool veto=(CurrentUIP[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD]&(1u<<cohortLoop22_slot))!=0u ||
                    (GTested[0]!=0u && GLost[0]*2u>GTested[0]);
                bool duplicateGeometry=all(core==OldRect(GSlot,SBC_E_LAST_CORE)) && all(cover==OldRect(GSlot,SBC_E_LAST_COVER));
                if (GDuplicate!=0u && !duplicateGeometry) NextState[b+SBC_E_FLAGS]=SBC_ENTRY_UNCERTAIN_REPLACEMENT;
                else if (GDuplicate!=0u) RetainBirth(b,GSlot,PreviousState[old+SBC_E_PHASE]);
                else if (suspended && GQuiet[0]==0u) BeginBirth(b,core,cover);
                else if (suspended || veto) {
                    // A stable patch is still ambiguous on later observations. Retain provenance,
                    // not qualification; never add pixels or transfer the old age to a new glyph.
                    RetainBirth(b,GSlot,SBC_PHASE_SUSPENDED);
                } else if (GStable[0]!=0u) RetainBirth(b,GSlot,SBC_PHASE_ACTIVE);
            } else if (!ambiguous && GDuplicate==0u && GQuiet[0]==0u) {
                BeginBirth(b,core,cover);
            } else NextState[b+SBC_E_FLAGS]=SBC_ENTRY_INVALID_REFERENCE;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

void MemberFallback(uint slot,uint reason) {
    uint b=Entry(slot);
    InterlockedOr(NextState[b+SBC_E_FLAGS],reason);
    // Do not mutate phase/ID here: later groups must see the same prepared ownership decision.
    // Filter consumes the final OR-reduced flags and publishes invalid owners as SUSPENDED.
}

// MEMBERS reads old persistent maps and immutable prepared NEXT state. Local
// permission is stored only at its own source-transition seed, never at neighbors.
[numthreads(16,16,1)]
void members_main(uint3 dispatch:SV_DispatchThreadID,uint groupIndex:SV_GroupIndex) {
    uint width,height;NextOwner.GetDimensions(width,height);
    if (groupIndex==0u) {
        GCurrent=NextState[SBC_H_TAG]==SBC_TAG && NextState[SBC_H_SCHEMA]==SBC_SCHEMA && NextState[SBC_H_VALID]==1u?1u:0u;
        GCount=GCurrent!=0u?min(NextState[SBC_H_COUNT],SBC_CAPACITY):0u;
        GDuplicate=GCurrent!=0u && (NextState[SBC_H_FLAGS]&SBC_HEADER_DUPLICATE)!=0u?1u:0u;
        GOld=GCurrent!=0u && (NextState[SBC_H_FLAGS]&SBC_HEADER_OLD_VALID)!=0u?1u:0u;
        GPreviousSnapshot=GCurrent!=0u && (NextState[SBC_H_FLAGS]&SBC_HEADER_PREVIOUS_SNAPSHOT_VALID)!=0u?1u:0u;
        [unroll] for (uint memberInitSlot=0u;memberInitSlot<SBC_CAPACITY;++memberInitSlot) {
            uint initEntry=Entry(memberInitSlot);
            GPhase[memberInitSlot]=NextState[initEntry+SBC_E_PHASE];
            // These immutable bits exclude the flags ORed by other member groups.
            GFlags[memberInitSlot]=NextState[initEntry+SBC_E_FLAGS]&(SBC_ENTRY_UNCERTAIN_REPLACEMENT|SBC_ENTRY_NEW_BIRTH);
            GId[memberInitSlot]=NextState[initEntry+SBC_E_ID];
        }
    }
    GroupMemoryBarrierWithGroupSync();uint2 stride=Stride();
    [loop] for (uint memberY=dispatch.y;memberY<height;memberY+=stride.y) {
      [loop] for (uint memberX=dispatch.x;memberX<width;memberX+=stride.x) {
        int2 p=int2(memberX,memberY);uint owner=0u,birth=0u;float3 color;
        bool colorValid=LoadColor(false,p,color);
        if (GCurrent!=0u && Position(p)) {
            uint currentSupport=BaselineSupport.Load(int3(p,0));
            uint oldOwner=GOld!=0u?PreviousOwner.Load(int3(p,0)):0u;
            uint oldRGB=GOld!=0u?PreviousBirthRGB.Load(int3(p,0)):0u;
            uint oldContext=OwnerContext(oldOwner);
            bool oldUnknown=OwnerUnknown(oldOwner);
            bool oldRoleValid=oldOwner==0u;
            if (GOld!=0u && oldOwner!=0u) {
                [loop] for (uint priorRoleSlot=0u;priorRoleSlot<PreviousState[SBC_H_COUNT];++priorRoleSlot) {
                    uint priorEntry=Entry(priorRoleSlot);
                    if (PreviousState[priorEntry+SBC_E_ID]==oldContext && oldContext!=0u &&
                        (!oldUnknown || PreviousState[priorEntry+SBC_E_PHASE]==SBC_PHASE_ACTIVE ||
                         PreviousState[priorEntry+SBC_E_PHASE]==SBC_PHASE_PENDING)) oldRoleValid=true;
                }
                if (!oldRoleValid || (oldUnknown && (oldRGB&~SBC_UNKNOWN_RGB_MASK)!=0u))
                    InterlockedOr(NextState[SBC_H_FLAGS],SBC_HEADER_CURRENT_INVALID);
            }
            uint hits=0u,hit=0u;
            [loop] for (uint memberCoverSlot=0u;memberCoverSlot<GCount;++memberCoverSlot)
                if (Contains(p,OCRRect(memberCoverSlot,true))) { ++hits;hit=memberCoverSlot; }
            if (hits>1u && GDuplicate==0u) {
                [loop] for (uint overlapSlot=0u;overlapSlot<GCount;++overlapSlot)
                    if (Contains(p,OCRRect(overlapSlot,true))) MemberFallback(overlapSlot,SBC_ENTRY_AMBIGUOUS_PIXELS);
            } else {
                uint ownerSlot=SBC_INVALID_SLOT;
                [loop] for (uint carrySlot=0u;carrySlot<GCount;++carrySlot) {
                    uint carryEntry=Entry(carrySlot);
                    bool suspended=GPhase[carrySlot]==SBC_PHASE_SUSPENDED && GFlags[carrySlot]==SBC_ENTRY_UNCERTAIN_REPLACEMENT;
                    bool normal=(GPhase[carrySlot]==SBC_PHASE_PENDING || GPhase[carrySlot]==SBC_PHASE_ACTIVE) &&
                        (GFlags[carrySlot]&SBC_ENTRY_BLOCKING_FLAGS)==0u;
                    if (GOld==0u || !oldRoleValid || GId[carrySlot]==0u || GId[carrySlot]!=oldContext ||
                        NextState[carryEntry+SBC_E_PREVIOUS_SLOT]==SBC_INVALID_SLOT) continue;
                    if (!oldUnknown && (normal || suspended)) {
                        // Known lifetime membership never changes with current pixel color/S.
                        if ((oldRGB&~SBC_BIRTH_KNOWN_MASK)!=0u) MemberFallback(carrySlot,SBC_ENTRY_INVALID_REFERENCE);
                        else { owner=oldOwner;birth=oldRGB;ownerSlot=carrySlot; }
                    } else if (oldUnknown && normal) {
                        if ((oldRGB&~SBC_UNKNOWN_RGB_MASK)!=0u || !colorValid) MemberFallback(carrySlot,SBC_ENTRY_INVALID_REFERENCE);
                        else if (!BirthAnchorLost(color,oldRGB)) { owner=oldOwner;birth=oldRGB;ownerSlot=carrySlot; }
                        else if (GDuplicate==0u && currentSupport==1u) {
                            // Refresh only this local UNKNOWN reference, never known age.
                            owner=oldOwner;birth=PackRGB(color);ownerSlot=carrySlot;
                        } else if (GDuplicate!=0u) MemberFallback(carrySlot,SBC_ENTRY_INVALID_REFERENCE);
                        // A genuine mismatch with S0 clears UNKNOWN. A fixed match with
                        // S0 above retains its reference without drawing absent support.
                    }
                    break;
                }
                if (owner!=0u && hits==1u && GId[hit]!=0u && GId[hit]!=OwnerContext(owner)) {
                    MemberFallback(hit,SBC_ENTRY_AMBIGUOUS_PIXELS);
                    MemberFallback(ownerSlot,SBC_ENTRY_AMBIGUOUS_PIXELS);
                    owner=0u;birth=0u;
                }
                if (owner==0u && hits==1u && GDuplicate==0u &&
                    (GFlags[hit]&SBC_ENTRY_BLOCKING_FLAGS)==0u) {
                    bool newBirth=GPhase[hit]==SBC_PHASE_PENDING && (GFlags[hit]&SBC_ENTRY_NEW_BIRTH)!=0u;
                    bool active=GPhase[hit]==SBC_PHASE_ACTIVE;
                    if ((newBirth || (active && currentSupport==1u)) && GPreviousSnapshot!=0u) {
                        float3 previous;
                        bool previousValid=LoadColor(true,p,previous);
                        if (!colorValid || !previousValid) {
                            if (currentSupport==1u) MemberFallback(hit,SBC_ENTRY_INVALID_REFERENCE);
                        } else {
                            uint rgb=PackRGB(color),previousRGB=PackRGB(previous);
                            bool rawChanged=any(color!=previous);
                            if (newBirth && rgb!=previousRGB) {
                                owner=GId[hit];birth=rgb;float probability;
                                if (ProbabilityAt(p,probability) && probability>V2_OCR_ACTIVE_PROBABILITY_THRESHOLD && Edge(p,color)) birth|=SBC_BIRTH_ANCHOR_BIT;
                            } else if (currentSupport==1u &&
                                ((newBirth && rawChanged) || (active &&
                                 (PreviousBaselineSupport.Load(int3(p,0))==0u || rawChanged)))) {
                                owner=SBC_OWNER_UNKNOWN_BIT|GId[hit];birth=rgb;
                            }
                        }
                    }
                }
            }
        }
        NextOwner[uint2(memberX,memberY)]=owner;NextBirthRGB[uint2(memberX,memberY)]=birth;
      }
    }
}

// Exact M1 dependency K: Chebyshev body steps plus the integer collar footprint.
// The bounded loops are not unrolled, and only immutable NEXT seed coordinates
// are read. Permission never becomes a seed and cannot recursively grow K.
bool SeedNear(int2 p,uint id,uint4 cover,bool newKnownBirth) {
    const int reach=(int)(V2_SUBTITLE_GLYPH_MATTE_STEPS+V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SOURCE_PIXELS);
    [loop] for (int seedDy=-reach;seedDy<=reach;++seedDy) {
        [loop] for (int seedDx=-reach;seedDx<=reach;++seedDx) {
            int2 residual=max(abs(int2(seedDx,seedDy))-(int)V2_SUBTITLE_GLYPH_MATTE_STEPS,int2(0,0));
            if (residual.x*residual.x+residual.y*residual.y>(int)V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SQUARED) continue;
            int2 q=p+int2(seedDx,seedDy);
            if (!Position(q) || !Contains(q,cover)) continue;
            uint seed=NextOwnerRead.Load(int3(q,0));
            uint seedRGB=NextBirthRGBRead.Load(int3(q,0));
            if (newKnownBirth) {
                if (seed==id && (seedRGB&~SBC_BIRTH_KNOWN_MASK)==0u && BaselineSupport.Load(int3(q,0))==1u) return true;
            } else if (seed==(SBC_OWNER_UNKNOWN_BIT|id) && (seedRGB&~SBC_UNKNOWN_RGB_MASK)==0u) return true;
        }
    }
    return false;
}

// FILTER has no old state/owner/RGB SRVs bound. Raw previous RGB/S remain read-only;
// NEXT is immutable, so persistent private UAV writes cannot race another group.
// Only current baseline S may survive, with known ownership or local UNKNOWN permission.
[numthreads(16,16,1)]
void filter_main(uint3 dispatch:SV_DispatchThreadID,uint groupIndex:SV_GroupIndex) {
    uint width,height,fw,fh,pw,ph,bw,bh;
    Eligibility.GetDimensions(width,height);FilteredSupport.GetDimensions(fw,fh);
    PersistentOwnerWrite.GetDimensions(pw,ph);PersistentBirthRGBWrite.GetDimensions(bw,bh);
    if (groupIndex==0u) {
        uint n,s;NextStateRead.GetDimensions(n,s);
        GNextReadable=n==SBC_STATE_WORDS && s==4u && NextStateRead[SBC_H_TAG]==SBC_TAG && NextStateRead[SBC_H_SCHEMA]==SBC_SCHEMA &&
            NextStateRead[SBC_H_FRAME]==cohort_frame.x && NextStateRead[SBC_H_FRAME+1u]==cohort_frame.y &&
            NextStateRead[SBC_H_CLOCK]==cohort_observation.x && NextStateRead[SBC_H_CLOCK+1u]==cohort_observation.y?1u:0u;
        GCurrent=GNextReadable!=0u && NextStateRead[SBC_H_VALID]==1u &&
            (NextStateRead[SBC_H_FLAGS]&SBC_HEADER_CURRENT_INVALID)==0u?1u:0u;
        GCount=GCurrent!=0u?min(NextStateRead[SBC_H_COUNT],SBC_CAPACITY):0u;
        GDuplicate=GCurrent!=0u && (NextStateRead[SBC_H_FLAGS]&SBC_HEADER_DUPLICATE)!=0u?1u:0u;
        GPreviousSnapshot=GCurrent!=0u && (NextStateRead[SBC_H_FLAGS]&SBC_HEADER_PREVIOUS_SNAPSHOT_VALID)!=0u?1u:0u;
        [unroll] for (uint cohortLoop37_i=0u;cohortLoop37_i<SBC_CAPACITY;++cohortLoop37_i) {
            uint b=Entry(cohortLoop37_i);GPhase[cohortLoop37_i]=GNextReadable!=0u?NextStateRead[b+SBC_E_PHASE]:SBC_PHASE_UNKNOWN;
            GFlags[cohortLoop37_i]=GNextReadable!=0u?NextStateRead[b+SBC_E_FLAGS]:SBC_ENTRY_INVALID_REFERENCE;GId[cohortLoop37_i]=GNextReadable!=0u?NextStateRead[b+SBC_E_ID]:0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();uint2 stride=Stride();
    [loop] for (uint cohortLoop38_y=dispatch.y;cohortLoop38_y<height;cohortLoop38_y+=stride.y) {
      [loop] for (uint cohortLoop39_x=dispatch.x;cohortLoop39_x<width;cohortLoop39_x+=stride.x) {
        int2 p=int2(cohortLoop39_x,cohortLoop38_y);uint eligibility=1u;
        uint owner=GCurrent!=0u?NextOwnerRead.Load(int3(p,0)):0u;
        uint birth=GCurrent!=0u?NextBirthRGBRead.Load(int3(p,0)):0u;
        uint currentSupport=BaselineSupport.Load(int3(p,0));
        bool unknownContext=false;
        bool covered=false,fallback=false,owned=false;
        [loop] for (uint cohortLoop40_i=0u;cohortLoop40_i<GCount;++cohortLoop40_i) {
            uint b=Entry(cohortLoop40_i);
            uint4 r=uint4(NextStateRead[b+SBC_E_LAST_COVER],NextStateRead[b+SBC_E_LAST_COVER+1u],NextStateRead[b+SBC_E_LAST_COVER+2u],NextStateRead[b+SBC_E_LAST_COVER+3u]);
            bool validContext=GId[cohortLoop40_i]!=0u && GId[cohortLoop40_i]<SBC_OWNER_ID_LIMIT &&
                (GFlags[cohortLoop40_i]&SBC_ENTRY_BLOCKING_FLAGS)==0u;
            bool active=GPhase[cohortLoop40_i]==SBC_PHASE_ACTIVE && validContext;
            bool ownCover=Contains(p,r);
            if (OwnerUnknown(owner) && OwnerContext(owner)==GId[cohortLoop40_i] && validContext &&
                (GPhase[cohortLoop40_i]==SBC_PHASE_ACTIVE || GPhase[cohortLoop40_i]==SBC_PHASE_PENDING)) unknownContext=true;
            if (ownCover) { covered=true;if (!active) fallback=true; }
            if (active && ownCover) {
                if (owner==GId[cohortLoop40_i]) owned=true;
                // S0 never incurs a neighbor lookup or draws remembered membership.
                else if (currentSupport==1u && SeedNear(p,GId[cohortLoop40_i],r,false)) owned=true;
            }
            // During a NEW_BIRTH pending observation, unchanged RGB white body can
            // acquire local UNKNOWN provenance near independently new KNOWN ink.
            // Read NEXT only: newly published permission cannot recursively seed.
            if (owner==0u && currentSupport==1u && ownCover && validContext &&
                GPhase[cohortLoop40_i]==SBC_PHASE_PENDING &&
                (GFlags[cohortLoop40_i]&SBC_ENTRY_NEW_BIRTH)!=0u && GDuplicate==0u && GPreviousSnapshot!=0u &&
                PreviousBaselineSupport.Load(int3(p,0))==0u) {
                float3 color,previous;
                if (LoadColor(false,p,color) && LoadColor(true,p,previous) && all(color==previous) &&
                    SeedNear(p,GId[cohortLoop40_i],r,true)) {
                    owner=SBC_OWNER_UNKNOWN_BIT|GId[cohortLoop40_i];birth=PackRGB(color);unknownContext=true;
                }
            }
        }
        if (OwnerUnknown(owner) && (!unknownContext || (birth&~SBC_UNKNOWN_RGB_MASK)!=0u)) { owner=0u;birth=0u; }
        if (covered && !fallback) eligibility=owned?1u:0u;
        Eligibility[uint2(cohortLoop39_x,cohortLoop38_y)]=eligibility;
        if (cohortLoop39_x<fw && cohortLoop38_y<fh) FilteredSupport[uint2(cohortLoop39_x,cohortLoop38_y)]=BaselineSupport.Load(int3(p,0))==1u && eligibility==1u?1u:0u;
        if (cohortLoop39_x<pw && cohortLoop38_y<ph) PersistentOwnerWrite[uint2(cohortLoop39_x,cohortLoop38_y)]=owner;
        if (cohortLoop39_x<bw && cohortLoop38_y<bh) PersistentBirthRGBWrite[uint2(cohortLoop39_x,cohortLoop38_y)]=owner!=0u?birth:0u;
      }
    }
    uint n,s;PersistentStateWrite.GetDimensions(n,s);
    uint flat=dispatch.y*stride.x+dispatch.x,total=stride.x*stride.y;
    if (n==SBC_STATE_WORDS && s==4u) {
        [loop] for (uint cohortLoop41_word=flat;cohortLoop41_word<SBC_STATE_WORDS;cohortLoop41_word+=total) {
            uint value=GNextReadable!=0u?NextStateRead[cohortLoop41_word]:0u;
            if (GNextReadable!=0u && GCurrent==0u) {
                // A malformed raw support snapshot invalidates publication globally.
                // Retain diagnostic source context, never the prepared private owners.
                if (cohortLoop41_word==SBC_H_VALID || cohortLoop41_word==SBC_H_COUNT ||
                    cohortLoop41_word>=SBC_HEADER_WORDS) value=0u;
            } else if (GNextReadable!=0u && cohortLoop41_word>=SBC_HEADER_WORDS) {
                uint relative=cohortLoop41_word-SBC_HEADER_WORDS;
                uint b=Entry(relative/SBC_ENTRY_WORDS),member=relative%SBC_ENTRY_WORDS;
                uint reasons=NextStateRead[b+SBC_E_FLAGS];
                if (NextStateRead[b+SBC_E_ID]!=0u && (reasons&(SBC_ENTRY_INVALID_REFERENCE|SBC_ENTRY_AMBIGUOUS_PIXELS))!=0u) {
                    if (member==SBC_E_PHASE) value=SBC_PHASE_SUSPENDED;
                    if (member==SBC_E_STABLE_ANCHORS) value=0u;
                }
            }
            PersistentStateWrite[cohortLoop41_word]=value;
        }
    }
}

// PRIVATE RESEARCH ONLY. Final subset guard for an already completed native M1.
// Independent of the frozen cohort/snapshot closures; never fills a rectangle,
// creates support, draws remembered S, or changes the production matte.
#include "include/depth_coordinate_v2_ocr_assert.generated.hlsl"
#include "include/host_sbs_subtitle_cohort_contract.h"

// With source and field extents <=5120, all center/edge products below fit uint32.
// The conservative bound 4*max_extent^2 must continue to fit if the contract changes.
#if V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT > 32767u
#error "Matte guard integer source-center arithmetic requires a wider product"
#endif

cbuffer SubtitleMatteGuardConstants : register(b0) {
    uint4 guard_source;
    uint4 guard_field;
    uint4 guard_content;
    uint4 guard_frame;
    uint4 guard_observation;
    uint4 guard_control;
};

Texture2D<uint> RawMatte : register(t0);
Texture2D<uint> FilteredSupport : register(t1);
Texture2D<uint> CurrentOwner : register(t2);
StructuredBuffer<uint> CurrentState : register(t3);
StructuredBuffer<uint> CurrentOCR : register(t4);
StructuredBuffer<uint> CurrentUIP : register(t5);
StructuredBuffer<uint> CurrentSCT : register(t6);
RWTexture2D<uint> GuardedMatte : register(u0);

// 0: raw identity; 1: frozen guard; 2: authenticated current empty; 3: unusable target.
groupshared uint GuardMode;
groupshared uint GuardCount;
groupshared uint ActiveCount;
groupshared uint4 Covers[SBC_CAPACITY];
groupshared uint Lines[SBC_CAPACITY];
groupshared uint Active[SBC_CAPACITY];
groupshared int ReferenceCenter;
groupshared uint ReferenceWidth;

bool Finite(float value) { return (asuint(value)&0x7f800000u)!=0x7f800000u; }
bool Unit(float value) { return Finite(value) && value>=0.0f && value<=1.0f; }
uint Entry(uint slot) { return SBC_HEADER_WORDS+slot*SBC_ENTRY_WORDS; }
bool Position(int2 p) {
    return all(p>=int2(0,0)) && p.x<(int)guard_source.x && p.y<(int)guard_source.w;
}
bool RectValid(uint4 rect) {
    return rect.x<rect.z && rect.y<rect.w &&
        all(rect.xy>=guard_content.xy) && all(rect.zw<=guard_content.zw);
}
uint4 OCRRect(uint slot,bool cover) {
    uint base=(cover?V2_OCR_FINAL_BOX_OFFSET:V2_OCR_RAW_BOX_OFFSET)+slot*V2_OCR_BOX_WORD_COUNT;
    return uint4(CurrentOCR[base],CurrentOCR[base+1u],CurrentOCR[base+2u],CurrentOCR[base+3u]);
}
bool Contains(int2 p,uint4 rect) {
    // Same integer rational, half-open source-center membership as the CPU guard.
    if (!Position(p)) return false;
    uint2 scaled=uint2(
        2u*guard_content.x*guard_source.x+(2u*(uint)p.x+1u)*(guard_content.z-guard_content.x),
        2u*guard_content.y*guard_source.y+(2u*((uint)p.y+guard_source.z)+1u)*(guard_content.w-guard_content.y));
    return scaled.x>=2u*rect.x*guard_source.x && scaled.x<2u*rect.z*guard_source.x &&
        scaled.y>=2u*rect.y*guard_source.y && scaled.y<2u*rect.w*guard_source.y;
}

bool ClassifierContextValid() {
    ReferenceCenter=(int)guard_source.x;ReferenceWidth=guard_source.x;
    uint base=V2_OCR_CLASSIFIER_HISTORY_OFFSET;
    bool empty=true;
    [loop] for (uint historyWord=0u;historyWord<V2_OCR_CLASSIFIER_HISTORY_WORD_COUNT;++historyWord)
        empty=empty && CurrentOCR[base+historyWord]==0u;
    if (empty) return true;
    if (CurrentOCR[base]!=V2_OCR_CLASSIFIER_HISTORY_TAG ||
        CurrentOCR[base+1u]!=V2_OCR_CLASSIFIER_HISTORY_SCHEMA ||
        CurrentOCR[base+2u]!=V2_OCR_CLASSIFIER_HISTORY_KNOWN_FLAGS ||
        any(uint2(CurrentOCR[base+4u],CurrentOCR[base+5u])!=guard_observation.xy)) return false;
    [unroll] for (uint historyComponent=0u;historyComponent<4u;++historyComponent) {
        if (CurrentOCR[base+6u+historyComponent]!=guard_frame[historyComponent] ||
            CurrentOCR[base+10u+historyComponent]!=guard_source[historyComponent] ||
            CurrentOCR[base+16u+historyComponent]!=guard_content[historyComponent]) return false;
    }
    if (CurrentOCR[base+14u]!=guard_field.x || CurrentOCR[base+15u]!=guard_field.y) return false;
    int center=asint(CurrentOCR[base+20u]);uint width=CurrentOCR[base+21u];
    if (width==0u || width>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT || center<-(int)width || center>(int)width) return false;
    ReferenceCenter=center;ReferenceWidth=width;
    return true;
}

bool CurrentPacketsValid(uint width,uint height) {
    uint words,stride;
    if (guard_control.x!=1u || guard_control.y!=SBC_SCHEMA || any(guard_control.zw!=0u) ||
        guard_observation.z>1u || guard_observation.w!=0u ||
        !any(guard_frame.xy!=0u) || !any(guard_observation.xy!=0u) ||
        any(guard_source.xy==0u) || any(guard_source.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        any(guard_field.xy==0u) || any(guard_field.xy>V2_SUBTITLE_CONTOUR_SOURCE_MAX_EXTENT) ||
        !V2SubtitleOcrFieldIsCalibrated(guard_field.x,guard_field.y) ||
        width!=guard_source.x || height!=guard_source.w ||
        guard_content.x>=guard_content.z || guard_content.y>=guard_content.w || any(guard_content.zw>guard_field.xy)) return false;
    CurrentOCR.GetDimensions(words,stride);
    if (words!=V2_OCR_RECORD_WORD_COUNT || stride!=4u) return false;
    CurrentUIP.GetDimensions(words,stride);
    if (words!=V2_SUBTITLE_PLANE_STATE_WORD_COUNT || stride!=4u) return false;
    CurrentSCT.GetDimensions(words,stride);
    if (words!=V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT || stride!=4u) return false;
    uint cropTop,cropHeight,roiTop,roiBottom;
    if (!V2SubtitleOcrComputeCrop(guard_source.x,guard_source.y,cropTop,cropHeight) ||
        cropTop!=guard_source.z || cropHeight!=guard_source.w ||
        !V2SubtitleOcrProjectContentRowCeil(guard_source.x,guard_source.y,guard_field.x,guard_field.y,
            guard_content,V2_OCR_SAFE_ROW_TOP,roiTop) ||
        !V2SubtitleOcrProjectContentRowCeil(guard_source.x,guard_source.y,guard_field.x,guard_field.y,
            guard_content,V2_OCR_SAFE_ROW_BOTTOM,roiBottom) ||
        roiTop!=guard_field.z || roiBottom!=guard_field.w || roiTop>=roiBottom) return false;
    uint count=CurrentOCR[4u];
    if (CurrentOCR[0u]!=V2_OCR_RECORD_SCHEMA || CurrentOCR[1u]!=V2_OCR_RECORD_TAG || CurrentOCR[2u]!=1u ||
        CurrentOCR[3u]!=count || count>SBC_CAPACITY || CurrentOCR[15u]!=0u) return false;
    if (CurrentUIP[0u]!=V2_SUBTITLE_PLANE_STATE_SCHEMA || CurrentUIP[1u]!=V2_SUBTITLE_PLANE_STATE_TAG ||
        (CurrentUIP[V2_SUBTITLE_PLANE_FLAGS_WORD]&~V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG)!=0u ||
        CurrentUIP[V2_SUBTITLE_PLANE_CURRENT_SLOT_MASK_WORD]!=(1u<<count)-1u ||
        (count!=0u && CurrentUIP[V2_SUBTITLE_PLANE_FLAGS_WORD]!=V2_SUBTITLE_PLANE_CURRENT_VALID_FLAG) ||
        (CurrentUIP[V2_SUBTITLE_PLANE_HISTORY_RESET_MASK_WORD]&~((1u<<count)-1u))!=0u ||
        CurrentUIP[V2_SUBTITLE_PLANE_WARMUP_WORD]!=0u || CurrentUIP[V2_SUBTITLE_PLANE_RESERVED_OFFSET]!=0u ||
        CurrentSCT[0u]!=V2_SUBTITLE_CONTOUR_SCHEMA || CurrentSCT[1u]!=V2_SUBTITLE_CONTOUR_HISTORY_TAG ||
        CurrentSCT[2u]>1u || CurrentSCT[22u]!=V2_SUBTITLE_CONTOUR_RECIPE ||
        CurrentSCT[V2_SUBTITLE_CONTOUR_HISTORY_SNAPSHOT_VALID_WORD]!=1u ||
        CurrentUIP[V2_SUBTITLE_PLANE_CUT_EPOCH_WORD]!=CurrentSCT[3u]) return false;
    [unroll] for (uint packetComponent=0u;packetComponent<2u;++packetComponent) {
        if (CurrentOCR[5u+packetComponent]!=guard_frame[packetComponent] ||
            CurrentOCR[7u+packetComponent]!=guard_frame[2u+packetComponent] ||
            CurrentOCR[9u+packetComponent]!=guard_source[packetComponent] ||
            CurrentOCR[11u+packetComponent]!=guard_field[packetComponent] ||
            CurrentOCR[13u+packetComponent]!=guard_field[2u+packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_FRAME_OFFSET+packetComponent]!=guard_frame[packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_ANALYSIS_GENERATION_OFFSET+packetComponent]!=guard_frame[2u+packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_OBSERVATION_TIMESTAMP_OFFSET+packetComponent]!=guard_observation[packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_SOURCE_EXTENT_OFFSET+packetComponent]!=guard_source[packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_FIELD_EXTENT_OFFSET+packetComponent]!=guard_field[packetComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_ROI_OFFSET+packetComponent]!=guard_field[2u+packetComponent] ||
            CurrentSCT[20u+packetComponent]!=guard_observation[packetComponent]) return false;
    }
    [unroll] for (uint tupleComponent=0u;tupleComponent<4u;++tupleComponent) {
        if (CurrentSCT[4u+tupleComponent]!=guard_frame[tupleComponent] ||
            CurrentSCT[8u+tupleComponent]!=guard_source[tupleComponent] ||
            CurrentSCT[12u+tupleComponent]!=guard_field[tupleComponent] ||
            CurrentSCT[16u+tupleComponent]!=guard_content[tupleComponent] ||
            CurrentUIP[V2_SUBTITLE_PLANE_CONTENT_OFFSET+tupleComponent]!=guard_content[tupleComponent]) return false;
    }
    [unroll] for (uint sctReserved=V2_SUBTITLE_CONTOUR_HISTORY_RESERVED_OFFSET;
                 sctReserved<V2_SUBTITLE_CONTOUR_HISTORY_WORD_COUNT;++sctReserved)
        if (CurrentSCT[sctReserved]!=0u) return false;
    [loop] for (uint boxSlot=0u;boxSlot<SBC_CAPACITY;++boxSlot) {
        uint rawBase=V2_OCR_RAW_BOX_OFFSET+boxSlot*V2_OCR_BOX_WORD_COUNT;
        uint coverBase=V2_OCR_FINAL_BOX_OFFSET+boxSlot*V2_OCR_BOX_WORD_COUNT;
        if (boxSlot>=count) {
            [unroll] for (uint unusedWord=0u;unusedWord<V2_OCR_BOX_WORD_COUNT;++unusedWord)
                if (CurrentOCR[rawBase+unusedWord]!=0u || CurrentOCR[coverBase+unusedWord]!=0u) return false;
        } else {
            uint4 core=OCRRect(boxSlot,false),cover=OCRRect(boxSlot,true);
            float score=asfloat(CurrentOCR[rawBase+4u]);
            uint kind=CurrentOCR[rawBase+5u];
            if (!RectValid(core) || !RectValid(cover) || any(cover.xy>core.xy) || any(cover.zw<core.zw) ||
                core.y<roiTop || core.w>roiBottom || !Unit(score) || score<V2_OCR_MIN_MEAN_SCORE ||
                (kind&~V2_OCR_BOX_KNOWN_FLAGS)!=0u || CurrentOCR[rawBase+6u]==0u ||
                CurrentOCR[rawBase+6u]>V2_OCR_OUTPUT_WIDTH || CurrentOCR[rawBase+7u]>=CurrentOCR[rawBase+6u]) return false;
            [unroll] for (uint pairedWord=4u;pairedWord<V2_OCR_BOX_WORD_COUNT;++pairedWord)
                if (CurrentOCR[rawBase+pairedWord]!=CurrentOCR[coverBase+pairedWord]) return false;
            if (kind==0u) { if (cover.y<roiTop || cover.w>roiBottom) return false; }
            else if (cover.x!=guard_content.x || cover.z!=guard_content.z || cover.w!=guard_content.w || cover.y<roiTop) return false;
            Covers[boxSlot]=cover;
        }
    }
    if (!ClassifierContextValid()) return false;
    GuardCount=count;
    return true;
}

bool PrivateContextValid() {
    uint words,stride;CurrentState.GetDimensions(words,stride);
    if (words!=SBC_STATE_WORDS || stride!=4u || CurrentState[SBC_H_TAG]!=SBC_TAG ||
        CurrentState[SBC_H_SCHEMA]!=SBC_SCHEMA || CurrentState[SBC_H_VALID]!=1u ||
        CurrentState[SBC_H_CUT]!=CurrentSCT[3u] || CurrentState[SBC_H_COUNT]!=GuardCount ||
        CurrentState[SBC_H_NEXT_ID]==0u || CurrentState[SBC_H_NEXT_ID]>SBC_OWNER_ID_LIMIT ||
        CurrentState[SBC_H_EPOCH]==0u || (CurrentState[SBC_H_FLAGS]&~SBC_HEADER_KNOWN_FLAGS)!=0u ||
        CurrentState[SBC_H_REFERENCE]!=asuint(ReferenceCenter) || CurrentState[SBC_H_REFERENCE+1u]!=ReferenceWidth) return false;
    [unroll] for (uint privateComponent=0u;privateComponent<2u;++privateComponent) {
        if (CurrentState[SBC_H_FRAME+privateComponent]!=guard_frame[privateComponent] ||
            CurrentState[SBC_H_GENERATION+privateComponent]!=guard_frame[2u+privateComponent] ||
            CurrentState[SBC_H_CLOCK+privateComponent]!=guard_observation[privateComponent] ||
            CurrentState[SBC_H_SOURCE+privateComponent]!=guard_source[privateComponent] ||
            CurrentState[SBC_H_CROP+privateComponent]!=guard_source[2u+privateComponent] ||
            CurrentState[SBC_H_FIELD+privateComponent]!=guard_field[privateComponent]) return false;
    }
    [unroll] for (uint privateContent=0u;privateContent<4u;++privateContent)
        if (CurrentState[SBC_H_CONTENT+privateContent]!=guard_content[privateContent]) return false;
    [unroll] for (uint headerReserved=SBC_H_RESERVED;headerReserved<SBC_HEADER_WORDS;++headerReserved)
        if (CurrentState[headerReserved]!=0u) return false;
    [loop] for (uint stateSlot=0u;stateSlot<SBC_CAPACITY;++stateSlot) {
        uint entry=Entry(stateSlot);
        if (stateSlot>=GuardCount) {
            [unroll] for (uint emptyEntryWord=0u;emptyEntryWord<SBC_ENTRY_WORDS;++emptyEntryWord)
                if (CurrentState[entry+emptyEntryWord]!=0u) return false;
        } else {
            uint id=CurrentState[entry+SBC_E_ID],phase=CurrentState[entry+SBC_E_PHASE],flags=CurrentState[entry+SBC_E_FLAGS];
            if (phase>SBC_PHASE_SUSPENDED || (flags&~SBC_ENTRY_KNOWN_FLAGS)!=0u ||
                CurrentState[entry+SBC_E_RESERVED]!=0u || CurrentState[entry+SBC_E_RESERVED+1u]!=0u) return false;
            if (phase==SBC_PHASE_UNKNOWN) { if (id!=0u) return false; }
            else if (id==0u || id>=SBC_OWNER_ID_LIMIT || id>=CurrentState[SBC_H_NEXT_ID]) return false;
            [loop] for (uint priorSlot=0u;priorSlot<stateSlot;++priorSlot)
                if (id!=0u && CurrentState[Entry(priorSlot)+SBC_E_ID]==id) return false;
        }
    }
    return true;
}

bool Admitted(int2 p) {
    bool covered=false;
    [loop] for (uint unionSlot=0u;unionSlot<GuardCount;++unionSlot) {
        if (Contains(p,Covers[unionSlot])) {
            covered=true;
            if (ActiveCount==0u || Active[unionSlot]==0u) return true;
        }
    }
    if (!covered) return false;
    // D1 has exactly9 offsets; known lifetime seeds do not require current S.
    [loop] for (int knownDy=-1;knownDy<=1;++knownDy) {
        [loop] for (int knownDx=-1;knownDx<=1;++knownDx) {
            int2 q=p+int2(knownDx,knownDy);
            if (!Position(q)) continue;
            uint owner=CurrentOwner.Load(int3(q,0));
            [loop] for (uint knownSlot=0u;knownSlot<GuardCount;++knownSlot)
                if (Active[knownSlot]!=0u && owner==Lines[knownSlot] && Contains(q,Covers[knownSlot])) return true;
        }
    }
    const int reach=(int)(V2_SUBTITLE_GLYPH_MATTE_STEPS+V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SOURCE_PIXELS);
    [loop] for (int supportDy=-reach;supportDy<=reach;++supportDy) {
        [loop] for (int supportDx=-reach;supportDx<=reach;++supportDx) {
            int2 residual=max(abs(int2(supportDx,supportDy))-(int)V2_SUBTITLE_GLYPH_MATTE_STEPS,int2(0,0));
            if (residual.x*residual.x+residual.y*residual.y>(int)V2_SUBTITLE_GLYPH_MATTE_COLLAR_RADIUS_SQUARED) continue;
            int2 q=p+int2(supportDx,supportDy);
            if (!Position(q)) continue;
            uint support=FilteredSupport.Load(int3(q,0));
            // Native inputs must be binary. A local malformed value cannot justify
            // removing this raw-M pixel; the harness rejects such a plane globally.
            if (support>1u) return true;
            uint owner=CurrentOwner.Load(int3(q,0));
            [loop] for (uint supportSlot=0u;supportSlot<GuardCount;++supportSlot) {
                if (!Contains(q,Covers[supportSlot])) continue;
                if (Active[supportSlot]==0u || (support==1u && owner!=Lines[supportSlot])) return true;
            }
        }
    }
    return false;
}

[numthreads(16,16,1)]
void guard_main(uint3 dispatch_id:SV_DispatchThreadID,uint3 group_id:SV_GroupID,uint group_index:SV_GroupIndex) {
    uint width,height,rawWidth,rawHeight;GuardedMatte.GetDimensions(width,height);RawMatte.GetDimensions(rawWidth,rawHeight);
    if (group_index==0u) {
        GuardCount=0u;ActiveCount=0u;
        GuardMode=(rawWidth==width && rawHeight==height)?0u:3u;
        if (GuardMode==0u && CurrentPacketsValid(width,height)) {
            if (GuardCount==0u) GuardMode=2u;
            else {
                uint supportWidth,supportHeight,ownerWidth,ownerHeight;
                FilteredSupport.GetDimensions(supportWidth,supportHeight);CurrentOwner.GetDimensions(ownerWidth,ownerHeight);
                if (supportWidth==width && supportHeight==height && ownerWidth==width && ownerHeight==height && PrivateContextValid()) {
                    GuardMode=1u;
                    [loop] for (uint activeSlot=0u;activeSlot<GuardCount;++activeSlot) {
                        uint entry=Entry(activeSlot);Lines[activeSlot]=CurrentState[entry+SBC_E_ID];
                        uint4 lastCover=uint4(CurrentState[entry+SBC_E_LAST_COVER],CurrentState[entry+SBC_E_LAST_COVER+1u],
                            CurrentState[entry+SBC_E_LAST_COVER+2u],CurrentState[entry+SBC_E_LAST_COVER+3u]);
                        bool qualified=all(lastCover==Covers[activeSlot]) &&
                            (CurrentState[SBC_H_FLAGS]&(SBC_HEADER_CURRENT_INVALID|SBC_HEADER_AMBIGUOUS_ASSOCIATION))==0u &&
                            Lines[activeSlot]!=0u && Lines[activeSlot]<SBC_OWNER_ID_LIMIT &&
                            CurrentState[entry+SBC_E_PHASE]==SBC_PHASE_ACTIVE &&
                            (CurrentState[entry+SBC_E_FLAGS]&SBC_ENTRY_BLOCKING_FLAGS)==0u;
                        Active[activeSlot]=qualified?1u:0u;ActiveCount+=Active[activeSlot];
                    }
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Only trusted current context may define the dispatch stride. On malformed
    // context exactly group0 covers the destination with a fixed16x16 stride.
    bool trustedDispatch=GuardMode==1u || GuardMode==2u;
    if (!trustedDispatch && any(group_id!=0u)) return;
    uint2 stride=trustedDispatch?max(((guard_field.xy+15u)/16u)*16u,uint2(16u,16u)):uint2(16u,16u);
    [loop] for (uint outputY=dispatch_id.y;outputY<height;outputY+=stride.y) {
        [loop] for (uint outputX=dispatch_id.x;outputX<width;outputX+=stride.x) {
            int2 p=int2(outputX,outputY);uint output=0u;
            if (GuardMode!=2u && GuardMode!=3u) {
                uint raw=RawMatte.Load(int3(p,0));
                if (raw==1u) output=(GuardMode==0u || Admitted(p))?1u:0u;
            }
            GuardedMatte[uint2(outputX,outputY)]=output;
        }
    }
}

// Work and Caps have already joined this observation's OCR record and display snapshot.
// History keeps raw predictions for cue confirmation and the last blended per-line opacity for display continuity.
#include "include/host_sbs_direct_glyph_line.hlsl"

cbuffer Schedule : register(b0) {
    uint2 Extent;
    uint2 Token;
    uint SourceFlags; // Display snapshot (bit 0) and current video-caption OCR8 (bit 1).
    uint HasPrevious;
    uint Slot;
    uint LineCenterTwice;
};
ByteAddressBuffer Caps : register(t0);
StructuredBuffer<GlyphLine> PreviousLines : register(t1);
Texture2D<float4> PreviousSource : register(t2);
Texture2D<float4> CurrentSource : register(t3);
struct GlyphOpacity {
    float measured;
    float published;
};
StructuredBuffer<GlyphOpacity> PreviousAlpha : register(t4);
StructuredBuffer<float> LineAlpha : register(t5);
StructuredBuffer<GlyphLine> Planned : register(t6);
StructuredBuffer<uint> PaletteWinners : register(t7);
StructuredBuffer<uint> VideoCaptionOcr : register(t8);

RWByteAddressBuffer Work : register(u0);
RWStructuredBuffer<GlyphLine> Lines : register(u1);
RWStructuredBuffer<uint2> Sizes : register(u2);
RWStructuredBuffer<GlyphOpacity> NextAlpha : register(u3);
RWTexture2D<uint> Protection : register(u4);

bool rectangle(uint4 v) {
    return all(v.xy < v.zw) && all(v.zw <= Extent);
}
uint4 intersect_rect(uint4 a, uint4 b) {
    return uint4(max(a.xy, b.xy), min(a.zw, b.zw));
}
uint area(uint4 r) {
    return all(r.xy < r.zw) ? (r.z-r.x)*(r.w-r.y) : 0u;
}
bool overlaps(uint4 a, uint4 b) {
    uint common = area(intersect_rect(a,b));
    return 3u * common > area(a) + area(b);
}
uint4 expanded_work(uint index, uint4 cap) {
    uint4 original = Work.Load4(16u+16u*index);
    uint margin = (original.w-original.y+15u)/16u;
    uint4 box = uint4(max(cap.xy, max(original.xy, margin)-margin), min(cap.zw, original.zw+margin));
    // A detector can trim an isolated stroke at one end of a centered caption.
    if (LineCenterTwice > 0u && 2u * box.x <= LineCenterTwice && LineCenterTwice < 2u * box.z) {
        int offset = int(box.x + box.z) - int(LineCenterTwice);
        uint extension = min(box.w - box.y, uint(abs(offset)));
        if (offset > 0)
            box.x -= min(extension, box.x - cap.x);
        else
            box.z += min(extension, cap.z - box.z);
    }
    return box;
}

[numthreads(1,1,1)]
void begin_history() {
    [unroll] for (uint s=0; s<8; ++s) {
        Lines[s] = (GlyphLine)0;
        Sizes[s] = 0;
    }
    uint4 header = Work.Load4(0);
    if (header.x != 2u) return;
    if ((SourceFlags & 1u) == 0u || header.y == 0 || header.y > 8 || any(header.zw != 0) ||
        any(Caps.Load4(0) != uint4(1u,header.y,Token))) {
        Work.Store4(0,uint4(0u,0u,4u,0u));
        return;
    }
    uint4 bases[8], covers[8];
    [loop] for (uint i=0; i<header.y; ++i) {
        uint4 cover=Caps.Load4(16+16*i), original=Work.Load4(16+16*i);
        if (!rectangle(cover) || !rectangle(original) || any(cover.xy>original.xy) || any(cover.zw<original.zw)) {
            Work.Store4(0,uint4(0u,0u,4u,0u));
            return;
        }
        covers[i]=cover;
        bases[i]=expanded_work(i,cover);
    }
    [loop] for (uint slot=0; slot<header.y; ++slot) {
        GlyphLine record = (GlyphLine)0;
        record.base=bases[slot]; record.box=record.base; record.cover=covers[slot];
        if ((SourceFlags & 2u) != 0u) {
            // The work planner authenticated these OCR8 cores and their matching detector caps.
            uint core = 16u + 8u * slot;
            record.state.w = uint(VideoCaptionOcr[core + 5u] == 0u && asfloat(VideoCaptionOcr[core + 4u]) < 0.90f);
        }
        if (HasPrevious != 0) {
            uint n=0;
            [loop] for (uint prior=0; prior<8; ++prior) {
                GlyphLine previous=PreviousLines[prior];
                if (previous.state.w==NoGlyphInk) continue;
                uint4 p=previous.base;
                if (!overlaps(p,record.base)) continue;
                uint owners=0;
                [loop] for (uint other=0; other<header.y; ++other)
                    owners+=overlaps(p,bases[other]) ? 1u : 0u;
                if (owners==1) { record.state.x=prior+1u; ++n; }
            }
            if (n!=1) record.state.x=0;
        }
        Lines[slot]=record;
    }
}

int3 rgb8(Texture2D<float4> source, int2 p) {
    return int3(floor(saturate(source.Load(int3(p,0)).rgb)*255.0f+0.5f));
}
int luma(Texture2D<float4> source, int2 p) {
    int3 rgb=rgb8(source,p);
    return (rgb.x+2*rgb.y+rgb.z)/4;
}
int2 edge(Texture2D<float4> source, int2 p) {
    return int2(luma(source,p+int2(2,0))-luma(source,p-int2(2,0)),
                luma(source,p+int2(0,2))-luma(source,p-int2(0,2)));
}
uint raw_index(GlyphLine record,uint2 p) {
    return record.state.y+(p.y-record.box.y)*(record.box.z-record.box.x)+p.x-record.box.x;
}
// Compare 10*dot^2 >= 7*oldPower*newPower at integer precision.
uint2 product(uint a,uint b) {
    uint lo=(a&65535u)*(b&65535u);
    uint mid=(a>>16)*(b&65535u)+(a&65535u)*(b>>16)+(lo>>16);
    return uint2((a>>16)*(b>>16)+(mid>>16),(mid<<16)|(lo&65535u));
}
bool direction_agrees(int2 a,int2 b,uint oldPower) {
    int similarity=a.x*b.x+a.y*b.y;
    if (similarity<=0) return false;
    uint2 lhs=product(10u*uint(similarity),uint(similarity));
    uint2 rhs=product(7u*oldPower,uint(b.x*b.x+b.y*b.y));
    return lhs.x>rhs.x || (lhs.x==rhs.x && lhs.y>=rhs.y);
}
groupshared uint2 Counts[128];

[numthreads(128,1,1)]
void confirm_history(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint slot=group.x;
    GlyphLine record=Lines[slot];
    GlyphLine old=(GlyphLine)0;
    if (record.state.x>0) old=PreviousLines[record.state.x-1];
    uint4 carried=intersect_rect(uint4(min(record.base.xy,old.base.xy),max(record.base.zw,old.base.zw)),record.cover);
    // Geometry must be decided before inference. Only search when it can enlarge the crop.
    uint4 r=intersect_rect(old.box,record.base);
    uint2 size=max(r.zw,r.xy)-r.xy;
    uint stride=max(2u,size.x/250u);
    uint2 sites=(max(size,4u)-4u+stride-1u)/stride;
    uint2 votes=0;
    uint count=record.state.x!=0 && any(carried!=record.base) ? sites.x*sites.y : 0u;
    for (uint i=lane; i<count; i+=128u) {
        uint2 p=r.xy+2u+stride*uint2(i%sites.x,i/sites.x);
        if (PreviousAlpha[raw_index(old,p)].measured < 38.0f/255.0f) continue;
        int2 a=edge(PreviousSource,p);
        uint power=uint(a.x*a.x+a.y*a.y);
        if (power<24u*24u) continue;
        votes.x++;
        int3 delta=abs(rgb8(PreviousSource,p)-rgb8(CurrentSource,p));
        if (all(delta<=32) && direction_agrees(a,edge(CurrentSource,p),power)) votes.y++;
    }
    Counts[lane]=votes;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step=64; step>0; step/=2) {
        if (lane<step) Counts[lane]+=Counts[lane+step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0 && Counts[0].x>=24 && 10u*Counts[0].y>=7u*Counts[0].x) {
        record.box=carried;
        Lines[slot]=record;
    }
}

[numthreads(1,1,1)]
void finish_windows() {
    // The ninth pair accompanies the extents in the same nonblocking readback.
    // A held root preserves the last inferred lines, raw alpha and source RGB.
    bool held = all(Work.Load4(0) == uint4(3u, 0u, 0u, 0u));
    Sizes[8] = uint2(held ? 1u : 0u, 0u);
    uint offset=0;
    [unroll] for (uint slot=0; slot<8; ++slot) {
        GlyphLine record=Lines[slot];
        // Keep the OCR work height as the sampling scale. The real detector cover
        // defines the first and only vertical inference interval.
        uint reference_height = record.box.w - record.box.y;
        record.box.y = record.cover.y;
        record.box.w = record.cover.w;
        uint2 size=record.box.zw-record.box.xy;
        Sizes[slot]=size;
        Sizes[9u + slot] = uint2(reference_height, 0u);
        if (all(size>0)) {
            record.state.y=offset;
            Lines[slot]=record;
            Work.Store4(16+16*slot,record.box);
            offset+=size.x*size.y;
        }
    }
}

groupshared uint4 ChangeCounts[128];
// With stride max(2, width/250), the widest sample grid is 373 columns (width 749).
groupshared uint4 ColumnCounts[375];
groupshared uint WindowRejected;

bool matches_ink(int3 rgb, int3 fill, int3 outline) {
    return all(abs(rgb-fill)<=16) || all(abs(rgb-outline)<=16);
}

bool ink_changed(uint4 counts, uint allowance, uint limit) {
    float background=max(1u,counts.z);
    return 100u*counts.y>limit*counts.x ||
           100.0f*counts.y*background>counts.x*(allowance*background+50.0f*counts.w);
}

[numthreads(128,1,1)]
void confirm_mask_history(uint lane : SV_GroupIndex) {
    GlyphLine record=Lines[Slot];
    uint fill=PaletteWinners[0], outline=PaletteWinners[1];
    if (fill==NoGlyphInk) {
        if (lane==0) {
            record.state.w=NoGlyphInk;
            record.state.z=0u;
            Lines[Slot]=record;
        }
        return;
    }
    record.state.w=fill|(outline<<13);
    if (HasPrevious==0) {
        if (lane==0) Lines[Slot]=record;
        return;
    }
    GlyphLine old=(GlyphLine)0;
    if (record.state.x>0) old=PreviousLines[record.state.x-1];
    int3 fillNow=int3(PalColor(fill)), outlineNow=int3(PalColor(outline));
    int3 fillThen=int3(PalColor(old.state.w&8191u)), outlineThen=int3(PalColor(old.state.w>>13));
    uint4 overlap=intersect_rect(old.box,record.box);
    uint2 size=max(overlap.zw,overlap.xy)-overlap.xy;
    uint stride=max(2u,size.x/250u);
    uint2 sites=(max(size,4u)-4u+stride-1u)/stride;
    uint2 extent=record.box.zw-record.box.xy;
    uint4 votes=0;
    for (uint x=lane; x<sites.x; x+=128u) {
        uint4 column=0;
        for (uint y=0; y<sites.y; ++y) {
            uint2 p=overlap.xy+2u+stride*uint2(x,y);
            uint2 local=p-record.box.xy;
            uint now=uint(round(saturate(LineAlpha[local.y*extent.x+local.x])*255.0f));
            uint then=uint(round(saturate(PreviousAlpha[raw_index(old,p)].measured)*255.0f));
            int3 rgbNow=rgb8(CurrentSource,p), rgbThen=rgb8(PreviousSource,p);
            uint changed=any(abs(rgbNow-rgbThen)>32) ? 1u : 0u;
            if (max(then,now)<38u) column+=uint4(0,0,1,changed);
            else if ((now>=128u && matches_ink(rgbNow,fillNow,outlineNow)) ||
                     (then>=128u && matches_ink(rgbThen,fillThen,outlineThen))) column+=uint4(1,changed,0,0);
        }
        ColumnCounts[x]=column;
        votes+=column;
    }
    ChangeCounts[lane]=votes;
    if (lane==0) WindowRejected=0;
    GroupMemoryBarrierWithGroupSync();
    uint height=max(1u,extent.y);
    uint firstX=overlap.x+2u;
    // Evaluate overlapping two-height windows so a local replacement among many unchanged letters can veto reuse.
    for (uint cell=lane; cell*height<extent.x; cell+=128u) {
        uint left=record.box.x+cell*height, right=left+2u*height;
        uint begin=min(sites.x,(max(left,firstX)-firstX+stride-1u)/stride);
        uint end=min(sites.x,(max(right,firstX)-firstX+stride-1u)/stride);
        uint4 localCounts=0;
        for (uint x=begin; x<end; ++x) localCounts+=ColumnCounts[x];
        if (localCounts.x>=8u && ink_changed(localCounts,14u,50u)) InterlockedOr(WindowRejected,1u);
    }
    [unroll] for (uint step=64; step>0; step/=2) {
        if (lane<step) ChangeCounts[lane]+=ChangeCounts[lane+step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0) {
        uint4 total=ChangeCounts[0];
        record.state.z=total.x>=24u && WindowRejected==0u && !ink_changed(total,4u,40u);
        Lines[Slot]=record;
    }
}

// Backdrop motion may change a translucent edge while nearby solid ink stays in place.
float stroke_motion(GlyphLine current, GlyphLine old, int2 p) {
    int step = max(1, int((current.base.w - current.base.y + 16u) / 32u));
    float motion = 100.0f;
    bool changed_stroke = false;
    uint4 overlap = intersect_rect(current.box, old.box);
    int3 fill = int3(PalColor(current.state.w & 8191u));
    int3 outline = int3(PalColor(current.state.w >> 13));
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            int2 q = p + step * int2(x, y);
            if (any(q < int2(overlap.xy)) || any(q >= int2(overlap.zw))) continue;
            uint2 local = uint2(q) - current.box.xy;
            float now = LineAlpha[local.y * (current.box.z - current.box.x) + local.x];
            float then = PreviousAlpha[raw_index(old, uint2(q))].measured;
            changed_stroke = changed_stroke || (min(now, then) < 38.0f / 255.0f && max(now, then) >= 0.85f);
            if (min(now, then) < 0.85f) continue;
            int3 rgb = rgb8(CurrentSource, q);
            if (!matches_ink(rgb, fill, outline)) continue;
            int3 delta = abs(rgb - rgb8(PreviousSource, q));
            motion = min(motion, max(delta.r, max(delta.g, delta.b)) / 24.0f);
        }
    }
    return changed_stroke ? 100.0f : motion;
}

[numthreads(16,8,1)]
void publish_line(uint3 id: SV_DispatchThreadID) {
    GlyphLine record=Planned[Slot];
    uint2 size=record.box.zw-record.box.xy;
    if (any(id.xy>=size)) return;
    uint2 p=record.box.xy+id.xy;
    float a=LineAlpha[id.y*size.x+id.x];
    if (!isfinite(a) || a<0 || a>1) {
        uint ignored;
        Work.InterlockedOr(8u, 8u, ignored);
        return;
    }
    GlyphOpacity opacity;
    opacity.measured = a;
    if (record.state.z!=0) {
        GlyphLine old=PreviousLines[record.state.x-1];
        if (all(p>=old.box.xy) && all(p<old.box.zw)) {
            int3 delta=abs(rgb8(CurrentSource,p)-rgb8(PreviousSource,p));
            float d=max(delta.x,max(delta.y,delta.z))/24.0f;
            GlyphOpacity previous = PreviousAlpha[raw_index(old,p)];
            float prior = previous.measured;
            bool moved = any(record.box.yw != old.box.yw) &&
                         (any(record.box.yw != record.base.yw) || any(old.box.yw != old.base.yw));
            float current_code = round(255.0f * a);
            bool disputed = (current_code >= 38.0f) != (round(255.0f * prior) >= 38.0f);
            // Strong new ink uses the normal current-frame path even near an unchanged stroke.
            bool new_ink = current_code >= 76.0f && round(255.0f * previous.published) < 38.0f;
            [branch] if (moved && disputed && !new_ink && min(a, prior) >= 2.0f / 255.0f &&
                         max(a, prior) < 0.85f && d > 0.0f) {
                float nearby = stroke_motion(record, old, int2(p));
                if (nearby < d) {
                    prior = previous.published;
                    d = nearby;
                }
            }
            a += 0.95f * exp(-d*d) * (prior-a);
        }
    }
    opacity.published = a;
    NextAlpha[raw_index(record,p)] = opacity;
    InterlockedMax(Protection[p],uint(round(saturate(a)*255)));
}

[numthreads(1,1,1)]
void finish_history() {
    uint4 header=Work.Load4(0);
    if (header.x==2u && header.y>0u && header.y<=8u && header.z==0) {
        Work.Store(12u,1u);
    } else {
        [unroll] for (uint i=0;i<8;++i) Lines[i]=(GlyphLine)0;
    }
}

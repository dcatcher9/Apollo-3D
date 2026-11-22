// SPDX-License-Identifier: GPL-3.0-only
// PRIVATE authored comparison only. Stage beside an exact canonical shader tree.
// Production source/receipt/current-RGB/queued gates and native colour sampling
// are included unchanged. This file grants no live or current-producer authority.

// Keep the original row entry available as an independent control. Rename only
// its local declarations so the exact same owner can be instantiated below on
// a distance workspace, without editing or copying the canonical algorithm.
#define SunshineApplyUICS PrivateOriginalRowPinCS
#define SunshineUIPinAnchors PrivateOriginalPinAnchors
#define SunshineUIPinSlack PrivateOriginalPinSlack
#define SunshineUIPinCarryAnchors PrivateOriginalPinCarryAnchors
#define SunshineUIPinCarrySlack PrivateOriginalPinCarrySlack
#define SunshineUIPinRamp PrivateOriginalPinRamp
#define SunshineUIPinBound PrivateOriginalPinBound
#define SunshineUIPinMinSlack PrivateOriginalPinMinSlack
#define SunshineUIPinClamp PrivateOriginalPinClamp
#define SunshineUIPinSlackOf PrivateOriginalPinSlackOf
#include "host_sbs_shared_ui_warp.hlsl"
#undef SunshineApplyUICS
#undef SunshineUIPinAnchors
#undef SunshineUIPinSlack
#undef SunshineUIPinCarryAnchors
#undef SunshineUIPinCarrySlack
#undef SunshineUIPinRamp
#undef SunshineUIPinBound
#undef SunshineUIPinMinSlack
#undef SunshineUIPinClamp
#undef SunshineUIPinSlackOf
#undef SUNSHINE_UI_PIN_BLOCK

// All planes are distinct. Original native Base and authorized weights remain
// intact for diagnostics. The horizontal distance becomes immutable input to
// the column pass; neither the column pass nor its output is a new mask.
Texture2D<float> PrivateOriginalNativeBase : register(t15);
Texture2D<float> PrivateHorizontalDistance : register(t16);
RWTexture2D<float> PrivateHorizontalDistanceStore : register(u3);
RWTexture2D<float> PrivateFinalFieldStore : register(u4);
RWTexture2D<float> PrivateForwardDistanceStore : register(u5);
RWTexture2D<float> PrivateFinalDistanceStore : register(u6);

float PrivateDistancePlane() { return 0.0f; }

// For actual binary weights, applying the unchanged row owner to finite R0=1
// at plane zero gives min(R0, 0.5*max(horizontal_distance-1,0)/W).
// R0 exceeds every permitted |Base-UI| gap (<=0.08 U). A capped far distance
// therefore cannot alter geometry; an empty mask leaves Base exactly intact.
#define SunshineHostFinalStore PrivateHorizontalDistanceStore
#define SunshineUIPlaneParallax PrivateDistancePlane
#define SunshineApplyUICS private_horizontal_distance_main
#include "include/ui_pin_band_shared.hlsl"
#undef SunshineHostFinalStore
#undef SunshineUIPlaneParallax
#undef SunshineApplyUICS

float PrivateColumnSeed(uint x, uint y) {
    uint last = PinSourceShape.w - 1u;
    float here = PrivateHorizontalDistance.Load(int3(x,y,0));
    float before = PrivateHorizontalDistance.Load(int3(x,y > 0u ? y-1u : 0u,0));
    float after = PrivateHorizontalDistance.Load(int3(x,min(y+1u,last),0));
    return min(here,min(before,after));
}

float PrivateColumnBound(float slack, uint distance) {
    // Parallax is source-U in BOTH axes. Never divide this vertical step by H.
    return slack + 0.5f * float(distance) / float(PinSourceShape.x);
}

// A deterministic column scan is sufficient for this private comparison. One
// thread owns one entire column. It reads immutable horizontal distances in
// both directions and its own distinct forward scratch; no cross-group UAV
// dependence, parallel prefix scheduling assumption, or CPU renderer is used.
[numthreads(64,1,1)]
void private_column_distance_and_clamp_main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    if (x >= PinSourceShape.x || PinSourceShape.w == 0u) return;
    float best = 1.0f;
    uint anchor = 0u;
    [loop] for (uint y=0u; y<PinSourceShape.w; ++y) {
        float seed = PrivateColumnSeed(x,y);
        if (seed <= PrivateColumnBound(best,y-anchor)) {
            best=seed; anchor=y;
        }
        PrivateForwardDistanceStore[uint2(x,y)] = PrivateColumnBound(best,y-anchor);
    }
    // Each subsequent load is by this same thread from its own completed
    // forward writes. This explicit device barrier also documents their order.
    DeviceMemoryBarrier();
    best=1.0f; anchor=PinSourceShape.w-1u;
    uint2 authorized_size; float plane=0.0f;
    bool can_pin = SubtitleRenderAuthorized(authorized_size,plane,true) && SelectedMatteAuthorized() &&
                   all(authorized_size == PinSourceShape.xy);
    [loop] for (int yi=int(PinSourceShape.w)-1; yi>=0; --yi) {
        uint y=uint(yi);
        float seed = PrivateColumnSeed(x,y);
        if (seed <= PrivateColumnBound(best,anchor-y)) {
            best=seed; anchor=y;
        }
        float distance = min(PrivateForwardDistanceStore[uint2(x,y)],
                             PrivateColumnBound(best,anchor-y));
        float original = PrivateOriginalNativeBase.Load(int3(x,y,0));
        // Explicit unchanged endpoint preserves native Base bits. A zero cone
        // bound pins exactly to the unchanged numeric subtitle target.
        float value = !can_pin ? original : distance == 0.0f ? plane : abs(original-plane) <= distance ? original :
                      clamp(original,plane-distance,plane+distance);
        PrivateFinalDistanceStore[uint2(x,y)] = distance;
        PrivateFinalFieldStore[uint2(x,y)] = value;
        if (asuint(value) != asuint(original))
            InterlockedOr(SharedPinRowStore[uint2(0u,y)],1u);
    }
}

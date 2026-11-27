// Project authenticated OCR8 geometry and publish direct RGB glyph scores.
#include "include/host_sbs_direct_glyph_plan.hlsl"

static const uint DirectUnavailable = 0u;
static const uint DirectEmpty = 1u;
static const uint DirectActive = 2u;
static const uint DirectHeld = 3u;
static const uint DirectCutoffAlpha8 = 38u;
static const uint DirectCoreMargin = 12u;
static const uint DirectNativeScores = 1u;

[numthreads(1, 1, 1)]
void plan_direct_main() {
    // The palette decoder marks its current native scores in the fourth header word.
    WorkStore.Store4(0u, uint4(DirectUnavailable, 0u, 0u, 0u));
    [unroll] for (uint clear_slot = 0u; clear_slot < 8u; ++clear_slot)
        WorkStore.Store4(16u + 16u * clear_slot, 0u);

    Evidence.Store4(0u, 0u);
    [unroll] for (uint i = 0u; i < 8u; ++i) Evidence.Store4(16u + 16u * i, 0u);
    bool held = any(ActualRoot.Load3(240u) == 0u);
    uint2 groups = (Shape.xy + uint2(TileX - 1u, TileY - 1u)) / uint2(TileX, TileY);
    PublishArgs.Store4(0u, uint4(groups, held ? 0u : 1u, 0u));
    if (held) {
        WorkStore.Store(0u, DirectHeld);
        return;
    }

    uint mode = (Shape.w >> 8u) & 255u;
    uint expected = (Shape.w >> 16u) & 255u;
    if (Shape.z != DirectCutoffAlpha8 || expected != 1u ||
        (mode != 1u && mode != 2u && mode != 4u) ||
        !EarlyContextCurrent() || !EarlyRootCurrent(mode, expected)) {
        WorkStore.Store(8u, 1u);
        return;
    }
    if (EmptyWitnessCurrent(mode, expected)) {
        WorkStore.Store(0u, DirectEmpty);
        return;
    }

    uint count = OCR[4u];
    // DSA1 certifies the original FP16 snapshot without authoring either CGS5 texture.
    if (!OcrCurrent(count) || count == 0u ||
        !SourceCurrent(0x31415344u, 64u, uint4(10u, 0u, 0u, 1u)) ||
        SourceEnvelope[3u] != ((1u << count) - 1u)) {
        WorkStore.Store(8u, 2u);
        return;
    }
    [loop] for (uint slot = 0u; slot < count; ++slot) {
        uint4 core = Rect(16u + 8u * slot);
        uint4 cover = Rect(144u + 8u * slot);
        if (any(core.xy >= core.zw) || any(cover.xy >= cover.zw) ||
            any(cover.xy < Content.xy) || any(cover.zw > Content.zw) ||
            any(core.xy < cover.xy) || any(core.zw > cover.zw)) {
            WorkStore.Store(8u, 3u);
            return;
        }
        uint4 wide = NativeCover(cover);
        if (any(wide.xy >= wide.zw)) {
            WorkStore.Store(8u, 3u);
            return;
        }
        // Tighten the published support only. The palette transport adds its read-only RGB context.
        uint4 native_core = NativeCover(core);
        uint core_height = native_core.w - native_core.y;
        uint margin = clamp((3u * core_height + 7u) / 8u, DirectCoreMargin, 64u);
        uint4 outset = PaddedCover(native_core, margin);
        uint4 tight = uint4(max(wide.xy, outset.xy), min(wide.zw, outset.zw));
        uint4 output_cover = all(native_core.xy < native_core.zw) && all(tight.xy < tight.zw) ? tight : wide;
        WorkStore.Store4(16u + 16u * slot, output_cover);
        // Keep the original band-clipped detector cap for the palette decoder.
        Evidence.Store4(16u + 16u * slot, wide);
    }
    Evidence.Store4(0u, uint4(1u, count, ClockToken.zw));
    WorkStore.Store(4u, count);
    WorkStore.Store(0u, DirectActive);
}

[numthreads(TileX, TileY, 1)]
void publish_direct_main(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= Shape.xy)) return;

    uint4 header = Work.Load4(0u);
    // A failed transport may leave an Active plan without the decoder's native-score proof.
    bool active = header.x == DirectActive && header.y > 0u && header.y <= 8u &&
                  header.z == 0u && header.w == DirectNativeScores;
    bool empty = header.x == DirectEmpty && all(header.yzw == 0u);
    uint alpha = 0u;
    [branch] if (active) {
        [loop] for (uint slot = 0u; slot < header.y; ++slot) {
            uint4 rectangle = Work.Load4(16u + 16u * slot);
            if (all(id.xy >= rectangle.xy) && all(id.xy < rectangle.zw)) {
                alpha = Candidate.Load(int3(id.xy, 0));
                break;
            }
        }
    }
    OutputAlpha8[id.xy] = alpha;
    OutputMask[id.xy] = alpha >= DirectCutoffAlpha8 ? 1u : 0u;
    if (all(id.xy == 0u)) {
        bool current = active || empty;
        PublishEarlyReceipt(current ? 4u : 1u, 4u, current);
    }
}

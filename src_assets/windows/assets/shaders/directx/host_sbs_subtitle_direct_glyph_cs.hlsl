// Project authenticated OCR8 geometry and publish direct RGB glyph scores.
#include "host_sbs_raw_matte_early_cs.hlsl"
#include "include/host_sbs_direct_glyph_matte.hlsl"
#include "include/host_sbs_direct_glyph_resolved_fringe.hlsl"

static const uint DirectUnavailable = 0u;
static const uint DirectEmpty = 1u;
static const uint DirectActive = 2u;
static const uint DirectHeld = 3u;
static const uint DirectCutoffAlpha8 = 38u;
static const uint DirectCoreMargin = 12u;
static const uint DirectAdaptiveContext = 16u;
static const uint DirectDetectorCaps = 64u;

[numthreads(1, 1, 1)]
void plan_direct_main() {
    if ((Shape.w & HasLazyLegacy) != 0u) LegacyArgs.Store4(0u, 0u);
    // finish_score writes the current score lattice to the fourth header word.
    WorkStore.Store4(0u, uint4(DirectUnavailable, 0u, 0u, 0u));
    [unroll] for (uint clear_slot = 0u; clear_slot < 8u; ++clear_slot)
        WorkStore.Store4(16u + 16u * clear_slot, 0u);

    bool requestCaps = (Shape.w & DirectDetectorCaps) != 0u;
    bool requestFringe = (Shape.w & DirectResolvedFringeFlag) != 0u || requestCaps;
    if (requestFringe) {
        Evidence.Store4(0u, 0u);
        [unroll] for (uint i = 0u; i < 8u; ++i) Evidence.Store4(16u + 16u * i, 0u);
    }
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
        // Tighten the published support only. Transport adds its read-only RGB context.
        uint4 native_core = NativeCover(core);
        uint margin = DirectCoreMargin;
        if ((Shape.w & DirectAdaptiveContext) != 0u) {
            uint core_height = native_core.w - native_core.y;
            margin = clamp((3u * core_height + 7u) / 8u, DirectCoreMargin, 64u);
        }
        uint4 outset = PaddedCover(native_core, margin);
        uint4 tight = uint4(max(wide.xy, outset.xy), min(wide.zw, outset.zw));
        uint4 output_cover = all(native_core.xy < native_core.zw) && all(tight.xy < tight.zw) ? tight : wide;
        WorkStore.Store4(16u + 16u * slot, output_cover);
        if (requestFringe) {
            // Keep the original band-clipped detector cap and old Work unmodified.
            uint insetHeight = all(native_core.xy < native_core.zw) ? native_core.w - native_core.y : 0u;
            uint4 requested = PaddedCover(output_cover, (insetHeight + 7u) / 8u);
            Evidence.Store4(16u + 16u * slot, requestCaps ? wide :
                            uint4(max(requested.xy, wide.xy), min(requested.zw, wide.zw)));
        }
    }
    if (requestFringe) Evidence.Store4(0u, uint4(1u, count, ClockToken.zw));
    WorkStore.Store(4u, count);
    WorkStore.Store(0u, DirectActive);
}

[numthreads(TileX, TileY, 1)]
void publish_direct_main(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= Shape.xy)) return;

    uint4 header = Work.Load4(0u);
    bool active = header.x == DirectActive && header.y > 0u && header.y <= 8u &&
                  header.z == 0u && DirectScoreReady(header.w);
    bool empty = header.x == DirectEmpty && all(header.yzw == 0u);
    uint alpha = 0u;
    bool inBaseWork = false;
    [branch] if (active) {
        [loop] for (uint slot = 0u; slot < header.y; ++slot) {
            uint4 rectangle = Work.Load4(16u + 16u * slot);
            if (all(id.xy >= rectangle.xy) && all(id.xy < rectangle.zw)) {
                inBaseWork = true;
                alpha = DirectNativeAlpha(Candidate, id.xy, Shape.xy, header.w);
                break;
            }
        }
    }
    [branch] if (active && !inBaseWork && (Shape.w & DirectResolvedFringeFlag) != 0u &&
                 DirectCurrentResolvedFringeContains(id.xy, header.y, ClockToken.zw)) {
        uint completedCount;
        if (DirectOriginalScoreReady(header.w, completedCount) &&
            DirectOriginalTapsCovered(id.xy, Shape.xy, header.w, completedCount)) {
            uint added = DirectNativeAlpha(Candidate, id.xy, Shape.xy, header.w);
            if (added >= DirectResolvedFringeCutoff) alpha = added;
        }
    }
    OutputAlpha8[id.xy] = alpha;
    OutputMask[id.xy] = alpha >= DirectCutoffAlpha8 ? 1u : 0u;
    if (all(id.xy == 0u)) {
        // A failed transport may leave an Active plan without finish_score's proof.
        bool needs_legacy = !active && !empty && (header.x == DirectUnavailable || header.x == DirectActive);
        if (needs_legacy && (Shape.w & HasLazyLegacy) != 0u) {
            uint mode = (Shape.w >> 8u) & 255u;
            uint expected = (Shape.w >> 16u) & 255u;
            uint4 grid = LegacyFieldGrid();
            if (expected == 1u && (mode == 1u || mode == 2u || mode == 4u) &&
                EarlyContextCurrent() && EarlyRootCurrent(mode, expected) &&
                all(ActualRoot.Load4(176u) == grid)) {
                LegacyArgs.Store4(0u, grid);
                PublishEarlyReceipt(1u, 1u, true);
                return;
            }
        }
        bool current = active || empty;
        PublishEarlyReceipt(current ? 4u : 1u, 4u, current);
    }
}

// Production Host adapter. The pin band itself is the exact shared Game 3D pass.
// Original source/Base/SCP/SCT/matte/receipt operands remain unchanged. No RGB fill.
#define HOST_SBS_BASE_ONLY 1
#include "sbs_reprojection_v2_live_ps.hlsl"
#undef HOST_SBS_BASE_ONLY
#include "include/host_sbs_shared_current_glyph.hlsl"

cbuffer SharedPinGeometry : register(b6) {
    uint4 PinSourceShape; // full width, full height, native band top, native band height
};
Texture2D<float> SharedPinWeights : register(t12);
Texture2D<float> SharedPinField : register(t13);
Texture2D<uint> SharedPinRows : register(t14);
RWTexture2D<float> SunshineHostFinalStore : register(u0);
RWTexture2D<float> SharedPinWeightStore : register(u1);
RWTexture2D<uint> SharedPinRowStore : register(u2);

#if !defined(HOST_SBS_TEXT_GAME_PIN)
// Per-draw preparation: initialize args at byte 0, row-pin args at byte 16, and
// {selected receipt, subtitle authorized, Base warp authorized, reserved zero} at byte 32.
// The owner unbinds u3 before indirect execution, then binds the same allocation at t8.
ByteAddressBuffer SharedPinPreparation : register(t8);
RWByteAddressBuffer SharedPinPreparationStore : register(u3);
groupshared uint SharedPinGroupWarpAuthorized;

[numthreads(1,1,1)]
void prepare_main(uint3 id : SV_DispatchThreadID) {
    if (any(id != 0u)) return;
    uint bytes; SharedPinPreparationStore.GetDimensions(bytes);
    if (bytes != 48u) return;
    uint warp_authorized = WarpAuthorized() ? 1u : 0u;
    uint selected_receipt = SelectedMatteAuthorized() ? 1u : 0u;
    uint2 source_size; float plane;
    uint subtitle_authorized = SubtitleRenderAuthorized(source_size, plane) &&
        selected_receipt != 0u && all(source_size == PinSourceShape.xy) ? 1u : 0u;
    // The initializer builds both eyes in one thread. The pin grid covers the stacked rows.
    SharedPinPreparationStore.Store4(0u,
        uint4((PinSourceShape.x + 15u) / 16u, (PinSourceShape.w + 15u) / 16u,
              subtitle_authorized, 0u));
    SharedPinPreparationStore.Store4(16u,
        uint4(2u * PinSourceShape.w, 1u, subtitle_authorized, 0u));
    SharedPinPreparationStore.Store4(32u,
        uint4(selected_receipt, subtitle_authorized, warp_authorized, 0u));
    // The owner has cleared every row. Keep this literal receipt header even when no
    // subtitle is authorized: empty rows still select the exact ordinary Base inverse.
    SharedPinRowStore[uint2(0u,PinSourceShape.w)] = selected_receipt;
}
#endif

#if defined(HOST_SBS_TEXT_GAME_PIN)
// A Game 3D text game supplies its own text: the helper loaded into it publishes the coverage of
// every piece of text the game draws. That picture is the pin weight, so no part of the Host
// subtitle detection, its matte or its receipts takes part in this variant.
// The glyphs themselves are pinned, as a video subtitle's matte and a game's UI alpha are. Beside
// a pinned texel the pin pass lets the scene field leave the plane by its distance rule,
// 0.5 * max(d - 1, 0) / W at d pixels, along the row. This variant applies the same rule down
// each column first, so a depth edge also leaves the text gradually above and below it instead
// of tearing between a row with a glyph and the next row without one.
cbuffer TextGamePin : register(b7) {
    float4 TextGameRect;   // the game's picture in source pixels: left, top, 1/width, 1/height
    uint4 TextGameControl; // enabled, UI plane (float bits, full-source U), first and end source row of the current text
    float4 TextGameScene;  // the scene's positive display bound (full-source U), SDR white of an HDR capture (scRGB), reserved
};
Texture2D<float4> TextGameCoverage : register(t11);
float3 DepthColorToSrgb(float3 c, uint color_mode);
float3 DepthLinearToSrgb(float3 c);
// The captured picture as the display code values the game drew. An HDR capture holds SDR
// content scaled by the desktop's SDR white, which is undone exactly; the tone curve the depth
// input uses there would move a colour by more than the comparison allows.
float3 TextGameShownColour(float3 c) {
    uint mode = subtitle_render_control.y;
    float white = TextGameScene.y;
    if (mode == 2u && isfinite(white) && white > 0.0f) return DepthLinearToSrgb(max(c, 0.0f) / white);
    return DepthColorToSrgb(c, mode);
}
// The helper knows when a game draws text, not when the game shows it: an engine draws a whole
// line at once and reveals it glyph by glyph, and hides its dialogue without drawing anything. So
// the text is trusted only while the picture shows it. The coverage picture holds the text in its
// own colours; the verdict counts the texels at the core of the glyphs, light ones (the fill) and
// dark ones (an outline) apart, and how many of each the picture shows in that colour. Both kinds
// must be there: a bright scene alone looks like the fill, a dark one like the outline. And they
// must be there together: a scene that is bright in one place and dark in another looks like some
// of each. So a core with a core of the other kind beside it in its row counts as shown only when
// the picture shows that one too: the fill next to its outline, as only the text itself has it.
StructuredBuffer<uint> TextGameVerdict : register(t10);
RWStructuredBuffer<uint> TextGameVerdictStore : register(u3);
// The observation also feeds the Game 3D UI plane policy (tools/reshade/game3d_ui_adaptive.h,
// docs/reshade-sbs.md). After the four verdict words come six words for each tile of a 16-by-16
// grid over the game's picture: the glyph texels pinned exactly (coverage of 1/8 or more), and
// for each of the policy's five plane levels how many of those have scene in front of it. The
// scene is the field before any pinning, so the plane never measures its own correction.
static const uint TextGameVerdictWords = 4u + 256u * 6u;
// The game hides one line and leaves another: a name stays while the words under it go. Judged as
// one, the text still shown would vouch for the line that is gone, and its rows would stay pinned
// over the bare scene. So each line also answers for itself. The observation keeps one record for
// each row of the band: its glyph texels (coverage of a half or more), its cores and shown cores
// of each kind as the verdict counts them for the whole text, and the row's answer.
static const uint TextGameRowWords = 6u;
static const uint TextGameRowLimit = 4096u;
RWStructuredBuffer<uint> TextGameRowStore : register(u4);
static const float TextGamePlaneLevels[5] = {0.0f, 0.001f, 0.002f, 0.003f, 0.0035f};
float SharedOriginalParallax(float2 uv, uint eye);
bool TextGameLight(float3 drawn) {
    return dot(drawn, float3(0.2126f, 0.7152f, 0.0722f)) >= 0.5f;
}
// Whether the picture shows the drawn colour at a source pixel, both as display code values.
bool TextGameShownAt(uint2 pixel, float3 drawn) {
    float3 shown = TextGameShownColour(SourceColor.Load(int3(pixel, 0)).rgb);
    return all(isfinite(shown)) && all(abs(shown - drawn) <= 0.2f);
}
// How far along the row a core looks for a core of the other kind, in texels of the coverage picture.
static const uint TextGameNeighbourReach = 4u;
[numthreads(16,16,1)]
void text_game_verify_main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= PinSourceShape.x || id.y >= PinSourceShape.w) return;
    uint2 pixel = uint2(id.x, id.y + PinSourceShape.z);
    // Only the current text counts: glyphs of earlier lines left in other rows of the coverage
    // picture match a reference without text and would vouch for text that is not there.
    if (pixel.y < TextGameControl.z || pixel.y >= TextGameControl.w) return;
    float2 uv = (float2(pixel) + 0.5f - TextGameRect.xy) * TextGameRect.zw;
    if (any(uv < 0.0f) || any(uv >= 1.0f)) return;
    float4 text = TextGameCoverage.SampleLevel(LinearSampler, uv, 0);
    float coverage = text.a;
    bool row_counted = id.y < TextGameRowLimit;
    uint row_base = id.y * TextGameRowWords;
    if (row_counted && coverage >= 0.5f) InterlockedAdd(TextGameRowStore[row_base], 1u);
    float cap = TextGameScene.x;
    if (coverage >= 0.125f && isfinite(cap) && cap >= 0.0f) {
        uint2 tile = min(uint2(uv * 16.0f), uint2(15u, 15u));
        uint base = 4u + (tile.y * 16u + tile.x) * 6u;
        // The plane policy measures Base, the one field that is the same for both eyes.
        float scene = SharedOriginalParallax((float2(pixel) + 0.5f) / float2(PinSourceShape.xy), 2u);
        InterlockedAdd(TextGameVerdictStore[base], 1u);
        [unroll] for (uint level = 0u; level < 5u; ++level)
            if (scene + 0.05f * cap > min(TextGamePlaneLevels[level], 0.5f * cap))
                InterlockedAdd(TextGameVerdictStore[base + 1u + level], 1u);
    }
    if (coverage < 0.95f) return;
    // Both sides as display code values, whatever the capture's transfer.
    float3 drawn = saturate(text.rgb / coverage);
    bool light = TextGameLight(drawn);
    uint kind = light ? 0u : 2u;
    InterlockedAdd(TextGameVerdictStore[kind], 1u);
    if (row_counted) InterlockedAdd(TextGameRowStore[row_base + 1u + kind], 1u);
    bool shown = TextGameShownAt(pixel, drawn);
    if (shown) {
        // The nearest core of the other kind in the row, left before right at each distance. A
        // core without one in reach, as in text without an outline, answers for itself.
        uint coverage_width, coverage_height;
        TextGameCoverage.GetDimensions(coverage_width, coverage_height);
        float texel = max(1.0f, 1.0f / (TextGameRect.z * float(max(coverage_width, 1u))));
        [loop] for (uint probe = 0u; probe < 2u * TextGameNeighbourReach; ++probe) {
            float offset = round(float(probe / 2u + 1u) * texel) * ((probe & 1u) == 0u ? -1.0f : 1.0f);
            float column = float(pixel.x) + offset;
            if (column < 0.0f || column >= float(PinSourceShape.x)) continue;
            uint2 beside = uint2(uint(column), pixel.y);
            float2 beside_uv = (float2(beside) + 0.5f - TextGameRect.xy) * TextGameRect.zw;
            if (beside_uv.x < 0.0f || beside_uv.x >= 1.0f) continue;
            float4 other = TextGameCoverage.SampleLevel(LinearSampler, beside_uv, 0);
            if (other.a < 0.95f) continue;
            float3 other_drawn = saturate(other.rgb / other.a);
            if (TextGameLight(other_drawn) == light) continue;
            shown = TextGameShownAt(beside, other_drawn);
            break;
        }
    }
    if (shown) {
        InterlockedAdd(TextGameVerdictStore[kind + 1u], 1u);
        if (row_counted) InterlockedAdd(TextGameRowStore[row_base + 2u + kind], 1u);
    }
}
// A kind too rare to judge does not count against the text. Otherwise a sixteenth of it must be
// on screen: a line being revealed starts with its first glyphs, and all of it is pinned then.
bool TextGameKindShown(uint cores, uint shown) {
    return cores < 16u || shown * 16u >= cores;
}
bool TextGameTextVisible() {
    uint count, stride; TextGameVerdict.GetDimensions(count, stride);
    if (count != TextGameVerdictWords || stride != 4u) return false;
    // Without enough glyph cores of either kind there is nothing to judge, and nothing is pinned.
    return TextGameVerdict[0u] + TextGameVerdict[2u] >= 16u &&
        TextGameKindShown(TextGameVerdict[0u], TextGameVerdict[1u]) && TextGameKindShown(TextGameVerdict[2u], TextGameVerdict[3u]);
}
bool TextGamePinAuthorized() {
    float plane = asfloat(TextGameControl.y);
    return TextGameControl.x == 1u && TextGameControl.z < TextGameControl.w && WarpAuthorized() &&
        isfinite(plane) && plane >= 0.0f && plane <= V2_DIRECT_CONTAINER_LIMIT &&
        all(isfinite(TextGameRect)) && all(TextGameRect.zw > 0.0f);
}
// Text coverage at a source pixel. The coverage picture is current only in the rows of the current
// text; its other rows still hold earlier lines and are never read.
float TextGameCoverageAt(uint2 pixel) {
    if (pixel.y < TextGameControl.z || pixel.y >= TextGameControl.w) return 0.0f;
    float2 uv = (float2(pixel) + 0.5f - TextGameRect.xy) * TextGameRect.zw;
    if (any(uv < 0.0f) || any(uv > 1.0f)) return 0.0f;
    return saturate(TextGameCoverage.SampleLevel(LinearSampler, uv, 0).a);
}
static const float TextGameNoGlyph = 1.0e6f;
// A line is a run of band rows that hold glyph texels. It is judged by the rule the whole text is
// judged by. A run with too few cores to judge is left to the whole-text verdict, which every
// pinned texel needs as well.
groupshared uint TextGameLineBits[TextGameRowLimit / 32u];
bool TextGameRowVisible(uint row) {
    return row >= TextGameRowLimit || (TextGameLineBits[row >> 5u] & (1u << (row & 31u))) != 0u;
}
void TextGameJudgeLines(bool publish) {
    uint rows = min(PinSourceShape.w, TextGameRowLimit);
    [loop] for (uint word = 0u; word < TextGameRowLimit / 32u; ++word) TextGameLineBits[word] = 0u;
    uint light = 0u, light_shown = 0u, dark = 0u, dark_shown = 0u;
    [loop] for (uint row = 0u; row < rows; ++row) {
        uint base = row * TextGameRowWords;
        if (TextGameRowStore[base] != 0u) {
            light += TextGameRowStore[base + 1u];
            light_shown += TextGameRowStore[base + 2u];
            dark += TextGameRowStore[base + 3u];
            dark_shown += TextGameRowStore[base + 4u];
            // The last row of the line carries its answer.
            if (row + 1u >= rows || TextGameRowStore[base + TextGameRowWords] == 0u) {
                if (light + dark < 16u || (TextGameKindShown(light, light_shown) && TextGameKindShown(dark, dark_shown)))
                    TextGameLineBits[row >> 5u] |= 1u << (row & 31u);
                light = 0u; light_shown = 0u; dark = 0u; dark_shown = 0u;
            }
        }
    }
    // Every other row of the line takes the answer of the row below it.
    [loop] for (uint back = rows; back > 0u; --back) {
        uint at = back - 1u;
        uint base = at * TextGameRowWords;
        bool glyphs = TextGameRowStore[base] != 0u;
        if (glyphs && back < rows && TextGameRowStore[base + TextGameRowWords] != 0u && TextGameRowVisible(back))
            TextGameLineBits[at >> 5u] |= 1u << (at & 31u);
        if (publish) TextGameRowStore[base + 5u] = glyphs && TextGameRowVisible(at) ? 1u : 0u;
    }
}
// One thread walks one column of the pinned rows down and back up, and leaves in the field plane
// the number of rows from each texel to the nearest glyph texel of its column, once in each eye's
// rows. The initializer reads its own texel's distance and replaces it with that eye's field.
// Each group judges the lines for its own columns; the first also leaves the rows' answers in the
// record for the initializer.
[numthreads(64,1,1)]
void text_game_columns_main(uint3 id : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID, uint3 group : SV_GroupID) {
    if (lane.x == 0u) TextGameJudgeLines(group.x == 0u);
    GroupMemoryBarrierWithGroupSync();
    if (id.x >= PinSourceShape.x) return;
    float distance = TextGameNoGlyph;
    [loop] for (uint row = 0u; row < PinSourceShape.w; ++row) {
        bool glyph = TextGameRowVisible(row) && TextGameCoverageAt(uint2(id.x, row + PinSourceShape.z)) >= 0.5f;
        distance = glyph ? 0.0f : min(distance + 1.0f, TextGameNoGlyph);
        SunshineHostFinalStore[uint2(id.x, row)] = distance;
        SunshineHostFinalStore[uint2(id.x, row + PinSourceShape.w)] = distance;
    }
    distance = TextGameNoGlyph;
    [loop] for (uint back = PinSourceShape.w; back > 0u; --back) {
        uint2 at = uint2(id.x, back - 1u);
        float down = SunshineHostFinalStore[at];
        distance = down == 0.0f ? 0.0f : min(distance + 1.0f, TextGameNoGlyph);
        if (distance < down) {
            SunshineHostFinalStore[at] = distance;
            SunshineHostFinalStore[uint2(at.x, at.y + PinSourceShape.w)] = distance;
        }
    }
}
#endif

// The scene field of one eye (0 left, 1 right) from the eye-specific limiter's store, whose rows
// are the left eye's and then the right eye's. Eye 2, or a draw without that store, reads Base.
float SharedSceneFieldLoad(int2 texel, int row_offset, bool eye_field) {
    if (eye_field) return EyeParallax.Load(int3(texel.x, texel.y + row_offset, 0));
    return BaseParallax.Load(int3(texel, 0));
}
float SharedNativeBaseSample(float2 uv, uint eye) {
    // Building a second sampled field needs FP32 coefficients: hardware bilinear's
    // quantized fractions can make a source 0.5-pixel slope become >0.5 on this grid.
    uint2 size;BaseParallax.GetDimensions(size.x,size.y);
    uint2 eye_size;EyeParallax.GetDimensions(eye_size.x,eye_size.y);
    bool eye_field=eye<2u&&eye_size.y>=2u;
    if(eye_field)size=uint2(eye_size.x,eye_size.y/2u);
    int row_offset=eye_field?(int)(eye*size.y):0;
    float2 position=saturate(uv)*float2(size)-0.5f;
    int2 low=int2(floor(position)),last=int2(size)-1;float2 fraction=position-float2(low);
    float a=SharedSceneFieldLoad(clamp(low,int2(0,0),last),row_offset,eye_field);
    float b=SharedSceneFieldLoad(clamp(low+int2(1,0),int2(0,0),last),row_offset,eye_field);
    float c=SharedSceneFieldLoad(clamp(low+int2(0,1),int2(0,0),last),row_offset,eye_field);
    float d=SharedSceneFieldLoad(clamp(low+int2(1,1),int2(0,0),last),row_offset,eye_field);
    return lerp(lerp(a,b,fraction.x),lerp(c,d,fraction.x),fraction.y);
}
float SharedOriginalParallax(float2 uv, uint eye) {
#if defined(HOST_SBS_TEXT_GAME_PIN)
    if (!WarpAuthorized()) return 0.0f;
#else
    // Preparation and initialization share immutable draw operands. Pixel shaders still
    // evaluate their independent live authorization.
    if (SharedPinGroupWarpAuthorized == 0u) return 0.0f;
#endif
    if (video_roi_active == 0.0f) return SharedNativeBaseSample(uv, eye);
    uint2 field_size; BaseParallax.GetDimensions(field_size.x, field_size.y);
    if (!VideoRoiRectValid() || !VideoRoiContentValid(field_size)) return 0.0f;
    float2 extent = video_roi_source_uv.zw - video_roi_source_uv.xy;
    uint2 content_extent = video_roi_tensor_content_rect.zw - video_roi_tensor_content_rect.xy;
    float2 content_min = float2(video_roi_tensor_content_rect.xy) / float2(field_size);
    float2 content_max = float2(video_roi_tensor_content_rect.zw) / float2(field_size);
    float2 scale = (content_max - content_min) / extent;
    float2 bias = content_min - video_roi_source_uv.xy * scale;
    float projected_y = clamp(uv.y, video_roi_source_uv.y, video_roi_source_uv.w);
    float budget_per_v = V2_MAX_VERTICAL_SHEAR * (extent.x / extent.y) *
        ((float)content_extent.y / (float)content_extent.x);
    float projected_x=clamp(uv.x,video_roi_source_uv.x,video_roi_source_uv.z);
    float collar=VIDEO_ROI_MAX_HORIZONTAL_SLOPE*abs(uv.x-projected_x)+budget_per_v*abs(uv.y-projected_y);
    float p=extent.x*SharedNativeBaseSample(float2(projected_x*scale.x+bias.x,projected_y*scale.y+bias.y),eye);
    float magnitude=abs(p);
    if(collar<magnitude){float retained=magnitude-collar;return p<0.0f?-retained:retained;}
    return 0.0f;
}

groupshared uint SharedPinGroupAuthorized;
groupshared uint SharedPinGroupSelectedReceipt;
[numthreads(16,16,1)]
void initialize_main(uint3 id : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID) {
    if (all(lane.xy == 0u)) {
#if defined(HOST_SBS_TEXT_GAME_PIN)
        uint2 source_size; SourceColor.GetDimensions(source_size.x, source_size.y);
        SharedPinGroupSelectedReceipt = TextGamePinAuthorized() && TextGameTextVisible() && all(source_size == PinSourceShape.xy) ? 1u : 0u;
        SharedPinGroupAuthorized = SharedPinGroupSelectedReceipt;
#else
        uint4 prepared = SharedPinPreparation.Load4(32u);
        bool valid = all(prepared.xyz <= 1u) && prepared.w == 0u;
        SharedPinGroupSelectedReceipt = valid && prepared.x == 1u ? 1u : 0u;
        SharedPinGroupWarpAuthorized = valid && prepared.z == 1u ? 1u : 0u;
        SharedPinGroupAuthorized = valid && prepared.y == 1u &&
            SharedPinGroupSelectedReceipt != 0u && SharedPinGroupWarpAuthorized != 0u ? 1u : 0u;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    if (id.x >= PinSourceShape.x) return;
#if defined(HOST_SBS_TEXT_GAME_PIN)
    if (id.y >= 2u * PinSourceShape.w) return;
    // The scene-dependent column clamp can produce a different weight for each eye.
    uint eye = id.y >= PinSourceShape.w ? 1u : 0u;
    uint band_row = id.y - eye * PinSourceShape.w;
#else
    if (id.y >= PinSourceShape.w) return;
    uint band_row = id.y;
#endif
    // One owned per-draw header avoids repeating the long immutable receipt check in the PS.
    if (all(id.xy == 0u)) SharedPinRowStore[uint2(0u,PinSourceShape.w)] = SharedPinGroupSelectedReceipt;
    uint2 pixel = uint2(id.x, band_row + PinSourceShape.z);
    float2 uv = (float2(pixel) + 0.5f) / float2(PinSourceShape.xy);
#if defined(HOST_SBS_TEXT_GAME_PIN)
    float rows_to_glyph = SunshineHostFinalStore[id.xy];
    float original = SharedOriginalParallax(uv, eye);
    float bounded = original, weight = 0.0f;
    if (SharedPinGroupAuthorized != 0u) {
        float text_plane = asfloat(TextGameControl.y);
        float bound = 0.5f * max(rows_to_glyph - 1.0f, 0.0f) / (float)PinSourceShape.x;
        bounded = clamp(original, text_plane - bound, text_plane + bound);
        // A glyph texel carries its coverage, as a UI texel carries its alpha. A texel the column
        // rule moved lies exactly on its bound; the faintest weight makes that bound its slack, so
        // the pin pass carries it along the row and no texel beside it can jump past it.
        // Only a line the picture shows is pinned; see TextGameJudgeLines.
        bool line_shown = band_row >= TextGameRowLimit || TextGameRowStore[band_row * TextGameRowWords + 5u] != 0u;
        weight = line_shown ? TextGameCoverageAt(pixel) : 0.0f;
        if (weight <= 0.0f && asuint(bounded) != asuint(original)) weight = 1.0f / 1024.0f;
    }
    SunshineHostFinalStore[id.xy] = bounded;
#else
    // Both eyes use the same current glyph witness. Sample their scene fields separately.
    uint2 right = id.xy + uint2(0u, PinSourceShape.w);
    SunshineHostFinalStore[id.xy] = SharedOriginalParallax(uv, 0u);
    SunshineHostFinalStore[right] = SharedOriginalParallax(uv, 1u);
    float weight = SharedPinGroupAuthorized != 0u && CurrentGlyphContributor(int2(pixel)) ? 1.0f : 0.0f;
    SharedPinWeightStore[right] = weight;
#endif
    SharedPinWeightStore[id.xy] = weight;
    if (weight > 0.0f) InterlockedOr(SharedPinRowStore[uint2(0u,band_row)], 1u);
}

#define BUFFER_WIDTH PinSourceShape.x
#define BUFFER_HEIGHT PinSourceShape.w
// Both eyes' rows are pinned; each store row has its own weight row.
#define SUNSHINE_UI_PIN_FIELD_ROWS (2u * PinSourceShape.w)
static const uint Sunshine_SourceAlphaUI = 1u;
float SunshineUIPinWeight(uint x, uint y) { return SharedPinWeights.Load(int3(x,y,0)); }
float SunshineUIPlaneParallax() {
#if defined(HOST_SBS_TEXT_GAME_PIN)
    return asfloat(TextGameControl.y);
#else
    return asfloat(SubtitleCondition[5u]) * ((float)subtitle_render_source.z / (float)PinSourceShape.x);
#endif
}
#if defined(HOST_SBS_TEXT_GAME_PIN)
// The column pass deliberately produces fractional weights.
#define SUNSHINE_UI_PIN_SOFT_ONLY
#endif
#include "include/ui_pin_band_shared.hlsl"

#if !defined(HOST_SBS_TEXT_GAME_PIN)
// One group owns a complete row of one eye. Neighboring lanes load and update
// neighboring texels. Packed occupancy selects the nearest binary anchor.
#define HOST_PIN_LANES 128u
// One bit per texel at the D3D11 texture width limit.
#define HOST_PIN_WORDS 512u
groupshared uint HostPinPacked[HOST_PIN_WORDS];
groupshared uint HostPinPresence[HOST_PIN_WORDS / 32u];
groupshared int2 HostPinNeighbors[HOST_PIN_WORDS];

int HostPinLeftWord(uint block, uint bit) {
    uint before = HostPinPresence[block] & ((1u << bit) - 1u);
    if (before != 0u) return int(block * 32u) + firstbithigh(before);
    [loop] for (int prior = int(block) - 1; prior >= 0; --prior) {
        before = HostPinPresence[uint(prior)];
        if (before != 0u) return prior * 32 + firstbithigh(before);
    }
    return -1;
}

int HostPinRightWord(uint block, uint bit, uint blocks) {
    uint after = HostPinPresence[block] & (~1u << bit);
    if (after != 0u) return int(block * 32u) + firstbitlow(after);
    [loop] for (uint next = block + 1u; next < blocks; ++next) {
        after = HostPinPresence[next];
        if (after != 0u) return int(next * 32u) + firstbitlow(after);
    }
    return -1;
}

[numthreads(HOST_PIN_LANES, 1, 1)]
void HostBinaryPinCS(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint row = group.x;
    uint width = BUFFER_WIDTH;
    if (row >= SUNSHINE_UI_PIN_FIELD_ROWS || width == 0u || width > HOST_PIN_WORDS * 32u) return;
    if (SharedPinRows.Load(int3(0, row % BUFFER_HEIGHT, 0)) == 0u) return;

    uint words = (width + 31u) / 32u;
    uint blocks = (words + 31u) / 32u;
    for (uint clear_word = lane; clear_word < words; clear_word += HOST_PIN_LANES) HostPinPacked[clear_word] = 0u;
    if (lane < blocks) HostPinPresence[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint sample = lane; sample < width; sample += HOST_PIN_LANES) {
        if (SunshineUIPinWeight(sample, row) >= 1.0f) InterlockedOr(HostPinPacked[sample / 32u], 1u << (sample % 32u));
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint mask_word = lane; mask_word < words; mask_word += HOST_PIN_LANES) {
        if (HostPinPacked[mask_word] != 0u) InterlockedOr(HostPinPresence[mask_word / 32u], 1u << (mask_word % 32u));
    }
    GroupMemoryBarrierWithGroupSync();
    uint occupied = 0u;
    [loop] for (uint block = 0u; block < blocks; ++block) occupied |= HostPinPresence[block];
    if (occupied == 0u) return;

    for (uint neighbor_word = lane; neighbor_word < words; neighbor_word += HOST_PIN_LANES) {
        uint block = neighbor_word / 32u, bit = neighbor_word % 32u;
        int left = HostPinLeftWord(block, bit), right = HostPinRightWord(block, bit, blocks);
        uint left_bits = HostPinPacked[uint(max(left, 0))], right_bits = HostPinPacked[uint(max(right, 0))];
        HostPinNeighbors[neighbor_word] = int2(left < 0 ? -int(width) : left * 32 + firstbithigh(left_bits),
            right < 0 ? 2 * int(width) : right * 32 + firstbitlow(right_bits));
    }
    GroupMemoryBarrierWithGroupSync();

    float plane = SunshineUIPlaneParallax();
    [loop] for (uint x = lane; x < width; x += HOST_PIN_LANES) {
        uint word = x / 32u, bit = x % 32u;
        uint packed = HostPinPacked[word];
        uint at_or_left = packed & (~0u >> (31u - bit)), at_or_right = packed >> bit;
        int2 remote = HostPinNeighbors[word];
        int left = at_or_left != 0u ? int(word * 32u) + firstbithigh(at_or_left) : remote.x;
        int right = at_or_right != 0u ? int(x) + firstbitlow(at_or_right) : remote.y;
        uint distance = uint(max(min(int(x) - left, right - int(x)) - 1, 0));
        uint2 at = uint2(x, row);
        float original = SunshineHostFinalStore[at];
        float pinned = SunshineUIPinClamp(original, plane, 0.0f, distance);
        if (asuint(pinned) != asuint(original)) SunshineHostFinalStore[at] = pinned;
    }
}
#endif

bool SharedPinRowContributes(float source_v) {
    float y = saturate(source_v) * (float)PinSourceShape.y - 0.5f;
    int low = (int)floor(y); float fraction = y - (float)low;
    int first = (int)PinSourceShape.z, end = first + (int)PinSourceShape.w;
    int a = clamp(low, 0, (int)PinSourceShape.y - 1);
    int b = clamp(low + 1, 0, (int)PinSourceShape.y - 1);
    if (1.0f-fraction > 0.0f && a >= first && a < end && SharedPinRows.Load(int3(0,a-first,0)) != 0u) return true;
    return fraction > 0.0f && b >= first && b < end && SharedPinRows.Load(int3(0,b-first,0)) != 0u;
}
float2 SharedPinSourceUV(float2 destination_uv, float eye_sign) {
    // The authorized state remains a draw-time gate as well as a preparation gate.
    if (!WarpAuthorized()) return destination_uv;
    if (SharedPinRows.Load(int3(0,PinSourceShape.w,0)) == 0u) return ReprojectIfAuthorized(destination_uv,eye_sign);
    // Empty rows and missing/stale/revoked masks retain the exact ordinary inverse.
    if (!SharedPinRowContributes(destination_uv.y)) return ReprojectIfAuthorized(destination_uv,eye_sign);
    float x = destination_uv.x;
    // This eye's rows of the field plane. The clamp is the sampler's edge clamp on one eye's band.
    float band_rows = (float)PinSourceShape.w;
    float band_row = clamp(destination_uv.y * (float)PinSourceShape.y - (float)PinSourceShape.z, 0.5f, band_rows - 0.5f);
    float v = (band_row + (eye_sign > 0.0f ? band_rows : 0.0f)) / (2.0f * band_rows);
    [loop] for (int iteration=0; iteration<11; ++iteration) {
        float next = destination_uv.x + eye_sign * SharedPinField.SampleLevel(LinearSampler,float2(x,v),0);
        bool settled = asuint(next) == asuint(x); x = next; if (settled) break;
    }
    return float2(x,destination_uv.y);
}
float4 shared_pin_main_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return 0.0f;
    float native_y=uv.y*(float)PinSourceShape.y-0.5f;
    if (native_y <= (float)PinSourceShape.z-1.0f || native_y >= (float)(PinSourceShape.z+PinSourceShape.w))
        return BackgroundOriginalSample(ReprojectIfAuthorized(uv,eye_sign));
    // Missing selected authority retains only the canonical ordinary inverse.
    if (SharedPinRows.Load(int3(0,PinSourceShape.w,0)) == 0u)
        return BackgroundOriginalSample(ReprojectIfAuthorized(uv,eye_sign));
    return BackgroundOriginalSample(SharedPinSourceUV(uv,eye_sign));
}
float4 shared_pin_mapping_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return float4(-1,-1,0,0);
    float2 source_uv = SharedPinSourceUV(uv,eye_sign);
    return float4(source_uv,SharedPinRowContributes(uv.y)?1.0f:0.0f,1.0f);
}
float4 shared_pin_boundary_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return float4(0.0f,0.0f,0.0f,1.0f);
    float2 source_uv = SharedPinSourceUV(uv,eye_sign);
    return float4(source_uv.x < 0.0f || source_uv.x > 1.0f ? 1.0f : 0.0f,0.0f,0.0f,1.0f);
}
float4 shared_pin_coverage_ps(PS_INPUT input) : SV_TARGET {
    float2 uv; float eye_sign;
    if (!PackedToSource(input.TexCoord,uv,eye_sign)) return 0.0f;
    uint2 source_size; float plane;
    if (!SubtitleRenderAuthorized(source_size,plane) || !SelectedMatteAuthorized()) return 0.0f;
    float2 ui_uv=float2(uv.x+eye_sign*plane,uv.y);
    float2 position=saturate(ui_uv)*float2(source_size)-0.5f;
    int2 low=int2(floor(position));float2 fraction=position-float2(low);
    float coverage=0.0f;bool any_marked=false,all_marked=true;
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
        float weight=(x==0?1.0f-fraction.x:fraction.x)*(y==0?1.0f-fraction.y:fraction.y);
        if(weight<=0.0f)continue;
        if(CurrentGlyphContributor(clamp(low+int2(x,y),int2(0,0),int2(source_size)-1))) {coverage+=weight;any_marked=true;}
        else all_marked=false;
    }
    return float4(coverage,any_marked&&all_marked?1.0f:0.0f,0.0f,1.0f);
}

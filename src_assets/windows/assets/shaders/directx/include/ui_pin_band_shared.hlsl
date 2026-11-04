// SPDX-License-Identifier: GPL-3.0-only
// Shared Game 3D / Host native-mask UI pin band.
// Adapter contract: BUFFER_WIDTH/HEIGHT are immutable uniform dimensions;
// SunshineHostFinalStore is the conditioned RWTexture2D<float> field;
// Sunshine_SourceAlphaUI enables the pass; SunshineUIPinWeight(x,y) and
// SunshineUIPlaneParallax() supply current authorized mask weight and plane.
// Dispatch ceil(BUFFER_HEIGHT / 8) groups in X, one in Y and Z.
// Keep arithmetic and scan/carry order exact: binary pinning preserves the
// existing Game 3D float32 distance rule, including its non-precise bounds.

// UI pinning runs as its own pass after the scene field h. Each texel u with
// pin weight w(u) > 0 has slack r(u) = (1 - w(u))|h(u) - p| around the UI plane
// p, and the field becomes clamp(h, p - b, p + b) with
// b(x) = min_u r(u) + 0.5*max(|x - u| - 1, 0)/W (docs/reshade-sbs.md, UI pin
// band). Bilinear color reaches one texel beyond a positive-alpha texel
// center, so its neighbors (the collar) share its slack, and a neighboring
// output cannot pull a faint copy of an antialiased glyph into otherwise
// unmasked background. A bound is kept as an anchor texel a and a slack r and
// evaluated in closed form, r + 0.5*|x - a|/W, so a binary mask reproduces the
// distance rule 0.5*max(d - 1, 0)/W exactly.
// Each group owns eight adjacent rows split into eight chunks; no row is held
// in group memory. Per chunk: the anchor and slack reaching furthest right and
// left, then the incoming {left, right} anchors and slacks; slack -1 is none.
groupshared int2 SunshineUIPinAnchors[8u][8u];
groupshared float2 SunshineUIPinSlack[8u][8u];
groupshared int2 SunshineUIPinCarryAnchors[8u][8u];
groupshared float2 SunshineUIPinCarrySlack[8u][8u];
// Each pinning scan loads this many texels of its chunk (weights and field
// values) before it walks them in order: the loads of a block are independent
// and overlap, while the walk does exactly the arithmetic, in exactly the
// order, of a texel-by-texel scan. A block's field values are the ones the
// texel-by-texel scan would read: each texel is read before the scan writes it,
// and the scan writes only texels it has passed. Finer chunks would shorten
// the scans but move the carries' float comparisons, so the field would not
// stay bit-identical.
#define SUNSHINE_UI_PIN_BLOCK 16u

float SunshineUIPinRamp(uint distance) {
    return 0.5 * float(distance) / float(BUFFER_WIDTH);
}

// The bound an anchor with this slack puts on a texel this far away.
float SunshineUIPinBound(float slack, uint distance) {
    return slack + SunshineUIPinRamp(distance);
}

// The smaller of two slacks, either of which may be none.
float SunshineUIPinMinSlack(float a, float b) {
    return a < 0.0 ? b : b < 0.0 ? a : min(a, b);
}

// Zero slack is the binary distance rule in its own arithmetic, so a binary
// mask pins bit for bit as before; a zero bound is the plane itself.
float SunshineUIPinClamp(float value, float plane, float slack, uint distance) {
    float ramp = SunshineUIPinRamp(distance);
    if (slack <= 0.0) return distance == 0u ? plane : clamp(value, plane - ramp, plane + ramp);
    return clamp(value, (plane - slack) - ramp, (plane + slack) + ramp);
}

// The slack of one texel of the original field, or -1 without UI.
float SunshineUIPinSlackOf(float value, float weight, float plane) {
    return weight > 0.0 ? (1.0 - weight) * abs(value - plane) : -1.0;
}

// UI pinning runs after the scene field is complete. A probe frame observes
// the unpinned field first, so it never measures its own pinned field.
[numthreads(8, 8, 1)]
void SunshineApplyUICS(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    if (Sunshine_SourceAlphaUI == 0u) return;
    uint row = thread_id.x, lane = thread_id.y;
    uint y = group_id.x * 8u + row;
    // A partial last group keeps its idle threads until the final barrier.
    bool active = y < BUFFER_HEIGHT;
    uint chunk_start = lane * BUFFER_WIDTH / 8u;
    uint chunk_end = (lane + 1u) * BUFFER_WIDTH / 8u;
    float plane = SunshineUIPlaneParallax();
    // Beyond its chunk a texel u bounds the field from its collar: anchor u + 1
    // to the right and u - 1 to the left. Keep the one with the least bound
    // where the next chunk starts and where the previous one ends.
    int2 anchors = int2(0, 0);
    float2 slack = float2(-1.0, -1.0), reach = float2(0.0, 0.0);
    if (active) {
        [loop]
        for (uint block = chunk_start; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK) {
            float weights[SUNSHINE_UI_PIN_BLOCK], values[SUNSHINE_UI_PIN_BLOCK];
            [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
                const uint at = min(block + i, chunk_end - 1u);
                weights[i] = SunshineUIPinWeight(at, y);
                values[i] = SunshineHostFinalStore[int2(uint2(at, y))];
            }
            [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
                const uint x = block + j;
                float weight = weights[j];
                if (x >= chunk_end || weight <= 0.0) continue;
                // Fully weighted UI has no slack whatever the field.
                float r = weight >= 1.0 ? 0.0 : SunshineUIPinSlackOf(values[j], weight, plane);
                float right = SunshineUIPinBound(r, chunk_end - 1u - x);
                if (slack.x < 0.0 || right <= reach.x) { anchors.x = (int)x + 1; slack.x = r; reach.x = right; }
                float left = SunshineUIPinBound(r, x - chunk_start);
                if (slack.y < 0.0 || left < reach.y) { anchors.y = (int)x - 1; slack.y = r; reach.y = left; }
            }
        }
    }
    SunshineUIPinAnchors[lane][row] = anchors;
    SunshineUIPinSlack[lane][row] = slack;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        // Each chunk's left carry is the best rightward anchor of the chunks
        // before it, compared where it starts; ties keep the nearer anchor.
        int anchor = 0;
        float best = -1.0;
        [unroll]
        for (uint chunk = 0u; chunk < 8u; ++chunk) {
            SunshineUIPinCarryAnchors[chunk][row].x = anchor;
            SunshineUIPinCarrySlack[chunk][row].x = best;
            int next = (int)((chunk + 1u) * BUFFER_WIDTH / 8u);
            int offered_anchor = SunshineUIPinAnchors[chunk][row].x;
            float offered = SunshineUIPinSlack[chunk][row].x;
            if (offered >= 0.0 && (best < 0.0 ||
                    SunshineUIPinBound(offered, (uint)(next - offered_anchor)) <= SunshineUIPinBound(best, (uint)(next - anchor)))) {
                anchor = offered_anchor; best = offered;
            }
        }
        // The right carry: the best leftward anchor of the chunks after it,
        // compared at its last texel.
        anchor = 0; best = -1.0;
        [unroll]
        for (int chunk = 7; chunk >= 0; --chunk) {
            SunshineUIPinCarryAnchors[(uint)chunk][row].y = anchor;
            SunshineUIPinCarrySlack[(uint)chunk][row].y = best;
            int previous = (int)((uint)chunk * BUFFER_WIDTH / 8u) - 1;
            int offered_anchor = SunshineUIPinAnchors[(uint)chunk][row].y;
            float offered = SunshineUIPinSlack[(uint)chunk][row].y;
            if (offered >= 0.0 && (best < 0.0 ||
                    SunshineUIPinBound(offered, (uint)(offered_anchor - previous)) <= SunshineUIPinBound(best, (uint)(anchor - previous)))) {
                anchor = offered_anchor; best = offered;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 carry_anchor = SunshineUIPinCarryAnchors[lane][row];
    float2 carry_slack = SunshineUIPinCarrySlack[lane][row];
    // A row without UI is left bit-for-bit intact.
    if (!active || (carry_slack.x < 0.0 && carry_slack.y < 0.0 && slack.x < 0.0)) return;
    // The same plane again: FXC marks everything evaluated before this precise
    // arithmetic precise, and the clamps below must stay the multiply-adds of
    // the binary distance rule.
    plane = SunshineUIPlaneParallax();

    // Backward: bound each texel by the best anchor at or right of it. In the
    // chunk a texel's collar slack is the least slack of itself and its
    // neighbors; neighbors in other chunks arrive through the carries.
    int anchor = carry_anchor.y;
    float best = carry_slack.y;
    float value = 0.0, r_here = -1.0, r_next = -1.0;
    bool flat = true;
    if (chunk_start < chunk_end) {
        value = SunshineHostFinalStore[int2(uint2(chunk_end - 1u, y))];
        r_here = SunshineUIPinSlackOf(value, SunshineUIPinWeight(chunk_end - 1u, y), plane);
    }
    // Blocks from the chunk's end: texel top - j reads its left neighbor
    // top - j - 1, which the scan writes only after it.
    [loop]
    for (int top = (int)chunk_end - 1; top >= (int)chunk_start; top -= (int)SUNSHINE_UI_PIN_BLOCK) {
        float previous_weights[SUNSHINE_UI_PIN_BLOCK], previous_values[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            const uint at = (uint)max(top - (int)i - 1, (int)chunk_start);
            previous_weights[i] = SunshineUIPinWeight(at, y);
            previous_values[i] = SunshineHostFinalStore[int2(uint2(at, y))];
        }
        [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
            const int scan_x = top - (int)j;
            if (scan_x < (int)chunk_start) break;
            uint x = (uint)scan_x;
            float previous = 0.0, r_previous = -1.0;
            if (x > chunk_start) {
                previous = previous_values[j];
                r_previous = SunshineUIPinSlackOf(previous, previous_weights[j], plane);
            }
            float collar = SunshineUIPinMinSlack(SunshineUIPinMinSlack(r_previous, r_here), r_next);
            if (collar >= 0.0 && (best < 0.0 || collar <= SunshineUIPinBound(best, (uint)(anchor - scan_x)))) {
                anchor = scan_x; best = collar;
            }
            flat = flat && best == 0.0 && anchor == scan_x;
            if (best >= 0.0) {
                float pinned = SunshineUIPinClamp(value, plane, best, (uint)(anchor - scan_x));
                if (asuint(pinned) != asuint(value)) SunshineHostFinalStore[int2(uint2(x, y))] = pinned;
            }
            r_next = r_here; r_here = r_previous; value = previous;
        }
    }
    // A chunk that is the plane throughout has nothing left to bound.
    if (flat) return;

    // Forward: bound each texel by the best anchor at or left of it. The left
    // carry was taken from the original field. The backward pass replaced the
    // field in this chunk, so a collar texel offers |pinned - p|, never more
    // than its collar slack; whatever it adds beyond that is no tighter than
    // the backward bound or than h itself (h changes by at most 0.5/W a texel).
    anchor = carry_anchor.x;
    best = carry_slack.x;
    float w_previous = 0.0, w_here = chunk_start < chunk_end ? SunshineUIPinWeight(chunk_start, y) : 0.0;
    // Blocks from the chunk's start: texel x is read (as the backward pass
    // left it) before the scan writes it, and the scan writes only behind.
    [loop]
    for (uint block = chunk_start; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK) {
        float next_weights[SUNSHINE_UI_PIN_BLOCK], currents[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            next_weights[i] = SunshineUIPinWeight(min(block + i + 1u, chunk_end - 1u), y);
            currents[i] = SunshineHostFinalStore[int2(uint2(min(block + i, chunk_end - 1u), y))];
        }
        [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
            const uint x = block + j;
            if (x >= chunk_end) break;
            float w_next = x + 1u < chunk_end ? next_weights[j] : 0.0;
            bool collar = w_previous > 0.0 || w_here > 0.0 || w_next > 0.0;
            // Next to fully weighted UI the backward pass left the plane itself.
            if (max(max(w_previous, w_here), w_next) >= 1.0) {
                anchor = (int)x; best = 0.0;
            } else if (collar || best >= 0.0) {
                float current = currents[j];
                if (collar) {
                    float offered = abs(current - plane);
                    if (best < 0.0 || offered <= SunshineUIPinBound(best, (uint)((int)x - anchor))) { anchor = (int)x; best = offered; }
                }
                float pinned = SunshineUIPinClamp(current, plane, best, (uint)((int)x - anchor));
                if (asuint(pinned) != asuint(current)) SunshineHostFinalStore[int2(uint2(x, y))] = pinned;
            }
            w_previous = w_here; w_here = w_next;
        }
    }
}

// SPDX-License-Identifier: GPL-3.0-only
// Shared Game 3D / Host native-mask UI pin band.
// Adapter contract: BUFFER_WIDTH/HEIGHT are immutable uniform dimensions;
// SunshineHostFinalStore is the conditioned RWTexture2D<float> field;
// Sunshine_SourceAlphaUI enables the pass; SunshineUIPinWeight(x,y) and
// SunshineUIPlaneParallax() supply current authorized mask weight and plane.
// Dispatch ceil(BUFFER_HEIGHT / 8) groups in X, one in Y and Z.
// A store may hold the field more than once (one copy per eye): an adapter then
// defines SUNSHINE_UI_PIN_FIELD_ROWS, the rows of the store to pin, and
// SUNSHINE_UI_PIN_MASK_ROW(y), the mask row of store row y, and dispatches
// ceil(SUNSHINE_UI_PIN_FIELD_ROWS / 8) groups. Rows are independent, so every
// copy is pinned exactly as a single field is.
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
// Each group owns eight adjacent rows split into eight chunks. Per chunk:
// the anchor and slack reaching furthest right and
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
#define SUNSHINE_UI_PIN_WORDS 32u
// Each word holds one block of hard pixels. A word-presence mask fits in a
// single uint per chunk for widths up to 4096, including ragged chunks.
groupshared uint SunshineUIPinHardWords[SUNSHINE_UI_PIN_WORDS][8u][8u];
groupshared uint SunshineUIPinSoftChunks[8u][8u];
groupshared uint SunshineUIPinSoftRows[8u];
#ifndef SUNSHINE_UI_PIN_FIELD_ROWS
#define SUNSHINE_UI_PIN_FIELD_ROWS BUFFER_HEIGHT
#endif
#ifndef SUNSHINE_UI_PIN_MASK_ROW
#define SUNSHINE_UI_PIN_MASK_ROW(y) (y)
#endif

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

void SunshineUIPinOffer(uint first, uint last, float r, uint start, uint end,
        inout int2 anchors, inout float2 slack, inout float2 reach) {
    float right = SunshineUIPinBound(r, end - 1u - last);
    if (slack.x < 0.0 || right <= reach.x) { anchors.x = (int)last + 1; slack.x = r; reach.x = right; }
    float left = SunshineUIPinBound(r, first - start);
    if (slack.y < 0.0 || left < reach.y) { anchors.y = (int)first - 1; slack.y = r; reach.y = left; }
}

// All anchors on this row have zero slack. The nearest covered texel sets
// both bounds, so fold the two sweeps into one clamp with the original ramp.
void SunshineUIPinHardChunk(uint start, uint end, uint y, uint lane, uint row,
        uint present, int2 carry, float2 carry_slack, float plane) {
    int previous = carry_slack.x >= 0.0 ? carry.x - 1 : -int(BUFFER_WIDTH);
    int right_carry = carry_slack.y >= 0.0 ? carry.y + 1 : 2 * int(BUFFER_WIDTH);
    [loop]
    for (uint base = start, word = 0u; base < end; base += SUNSHINE_UI_PIN_BLOCK, ++word) {
        uint bits = SunshineUIPinHardWords[word][lane][row];
        uint later = present & (~1u << word);
        int following = right_carry;
        if (later != 0u) {
            uint next_word = (uint)firstbitlow(later);
            following = int(start + next_word * SUNSHINE_UI_PIN_BLOCK) + firstbitlow(SunshineUIPinHardWords[next_word][lane][row]);
        }
        uint count = min(SUNSHINE_UI_PIN_BLOCK, end - base);
        [unroll]
        for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            if (i >= count) break;
            int x = int(base + i);
            uint at_or_left = bits & ((2u << i) - 1u);
            uint at_or_right = bits >> i;
            int left = at_or_left != 0u ? int(base) + firstbithigh(at_or_left) : previous;
            int right = at_or_right != 0u ? x + firstbitlow(at_or_right) : following;
            uint distance = (uint)max(min(x - left, right - x) - 1, 0);
            int2 position = int2(x, y);
            float value = SunshineHostFinalStore[position];
            float pinned = SunshineUIPinClamp(value, plane, 0.0, distance);
            if (asuint(pinned) != asuint(value)) SunshineHostFinalStore[position] = pinned;
        }
        if (bits != 0u) previous = int(base) + firstbithigh(bits);
    }
}

// UI pinning runs after the scene field is complete. A probe frame observes
// the unpinned field first, so it never measures its own pinned field.
[numthreads(8, 8, 1)]
void SunshineApplyUICS(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    if (Sunshine_SourceAlphaUI == 0u) return;
    uint row = thread_id.x, lane = thread_id.y;
    uint y = group_id.x * 8u + row;
    // A partial last group keeps its idle threads until the final barrier.
    bool active = y < SUNSHINE_UI_PIN_FIELD_ROWS;
    // The mask row this store row is pinned by.
    uint mask_y = SUNSHINE_UI_PIN_MASK_ROW(y);
    uint chunk_start = lane * BUFFER_WIDTH / 8u;
    uint chunk_end = (lane + 1u) * BUFFER_WIDTH / 8u;
    float plane = SunshineUIPlaneParallax();
    // Beyond its chunk a texel u bounds the field from its collar: anchor u + 1
    // to the right and u - 1 to the left. Keep the one with the least bound
    // where the next chunk starts and where the previous one ends.
    int2 anchors = int2(0, 0);
    float2 slack = float2(-1.0, -1.0), reach = float2(0.0, 0.0);
#ifdef SUNSHINE_UI_PIN_SOFT_ONLY
    const bool hard_words_fit = false;
#else
    const bool hard_words_fit = BUFFER_WIDTH <= 8u * SUNSHINE_UI_PIN_WORDS * SUNSHINE_UI_PIN_BLOCK;
#endif
    uint present = 0u;
    bool soft_chunk = !hard_words_fit;
    if (active) {
        [loop]
        for (uint block = chunk_start, word = 0u; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK, ++word) {
            float weights[SUNSHINE_UI_PIN_BLOCK], values[SUNSHINE_UI_PIN_BLOCK];
            uint bits = 0u;
            bool soft_block = false;
            [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
                const uint x = block + i;
                const uint at = min(x, chunk_end - 1u);
                float weight = SunshineUIPinWeight(at, mask_y);
                weights[i] = weight;
                if (!hard_words_fit) values[i] = SunshineHostFinalStore[int2(uint2(at, y))];
                if (!soft_chunk && x < chunk_end) {
                    bits |= weight >= 1.0 ? 1u << i : 0u;
                    soft_block = soft_block || (weight > 0.0 && weight < 1.0);
                }
            }
            if (!soft_chunk && !soft_block) {
                SunshineUIPinHardWords[word][lane][row] = bits;
                if (bits != 0u) {
                    present |= 1u << word;
                    SunshineUIPinOffer(block + firstbitlow(bits), block + firstbithigh(bits), 0.0,
                        chunk_start, chunk_end, anchors, slack, reach);
                }
                continue;
            }
            soft_chunk = true;
            // Where packing is enabled, delay field reads until a soft weight
            // needs them. Otherwise retain the interleaved setup loads.
            if (hard_words_fit) {
                [unroll] for (uint sample_index = 0u; sample_index < SUNSHINE_UI_PIN_BLOCK; ++sample_index) {
                    const uint at = min(block + sample_index, chunk_end - 1u);
                    values[sample_index] = SunshineHostFinalStore[int2(uint2(at, y))];
                }
            }
            [unroll] for (uint j = 0u; j < SUNSHINE_UI_PIN_BLOCK; ++j) {
                const uint x = block + j;
                float weight = weights[j];
                if (x >= chunk_end || weight <= 0.0) continue;
                // Fully weighted UI has no slack whatever the field.
                float r = weight >= 1.0 ? 0.0 : SunshineUIPinSlackOf(values[j], weight, plane);
                SunshineUIPinOffer(x, x, r, chunk_start, chunk_end, anchors, slack, reach);
            }
        }
    }
    SunshineUIPinAnchors[lane][row] = anchors;
    SunshineUIPinSlack[lane][row] = slack;
    if (hard_words_fit) SunshineUIPinSoftChunks[lane][row] = soft_chunk ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        // Each chunk's left carry is the best rightward anchor of the chunks
        // before it, compared where it starts; ties keep the nearer anchor.
        int anchor = 0;
        float best = -1.0;
        uint soft_row = 0u;
        [unroll]
        for (uint chunk = 0u; chunk < 8u; ++chunk) {
            if (hard_words_fit) soft_row |= SunshineUIPinSoftChunks[chunk][row];
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
        if (hard_words_fit) SunshineUIPinSoftRows[row] = soft_row;
        // The right carry: the best leftward anchor of the chunks after it,
        // compared at its last texel.
        anchor = 0; best = -1.0;
        [unroll]
        for (int reverse_chunk = 7; reverse_chunk >= 0; --reverse_chunk) {
            SunshineUIPinCarryAnchors[(uint)reverse_chunk][row].y = anchor;
            SunshineUIPinCarrySlack[(uint)reverse_chunk][row].y = best;
            int previous = (int)((uint)reverse_chunk * BUFFER_WIDTH / 8u) - 1;
            int offered_anchor = SunshineUIPinAnchors[(uint)reverse_chunk][row].y;
            float offered = SunshineUIPinSlack[(uint)reverse_chunk][row].y;
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
    // Keep the sweeps and the hard clamp from diverging within a group.
    if (hard_words_fit) {
        uint soft_group = 0u;
        [unroll] for (uint group_row = 0u; group_row < 8u; ++group_row)
            soft_group |= SunshineUIPinSoftRows[group_row];
        if (soft_group == 0u) {
            SunshineUIPinHardChunk(chunk_start, chunk_end, y, lane, row, present, carry_anchor, carry_slack, plane);
            return;
        }
    }

    // Backward: bound each texel by the best anchor at or right of it. In the
    // chunk a texel's collar slack is the least slack of itself and its
    // neighbors; neighbors in other chunks arrive through the carries.
    int anchor = carry_anchor.y;
    float best = carry_slack.y;
    float value = 0.0, r_here = -1.0, r_next = -1.0;
    bool flat = true;
    if (chunk_start < chunk_end) {
        value = SunshineHostFinalStore[int2(uint2(chunk_end - 1u, y))];
        r_here = SunshineUIPinSlackOf(value, SunshineUIPinWeight(chunk_end - 1u, mask_y), plane);
    }
    // Blocks from the chunk's end: texel top - j reads its left neighbor
    // top - j - 1, which the scan writes only after it.
    [loop]
    for (int top = (int)chunk_end - 1; top >= (int)chunk_start; top -= (int)SUNSHINE_UI_PIN_BLOCK) {
        float previous_weights[SUNSHINE_UI_PIN_BLOCK], previous_values[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            const uint at = (uint)max(top - (int)i - 1, (int)chunk_start);
            previous_weights[i] = SunshineUIPinWeight(at, mask_y);
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
    float w_previous = 0.0, w_here = chunk_start < chunk_end ? SunshineUIPinWeight(chunk_start, mask_y) : 0.0;
    // Blocks from the chunk's start: texel x is read (as the backward pass
    // left it) before the scan writes it, and the scan writes only behind.
    [loop]
    for (uint block = chunk_start; block < chunk_end; block += SUNSHINE_UI_PIN_BLOCK) {
        float next_weights[SUNSHINE_UI_PIN_BLOCK], currents[SUNSHINE_UI_PIN_BLOCK];
        [unroll] for (uint i = 0u; i < SUNSHINE_UI_PIN_BLOCK; ++i) {
            next_weights[i] = SunshineUIPinWeight(min(block + i + 1u, chunk_end - 1u), mask_y);
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

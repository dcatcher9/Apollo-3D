// Guided depth upsampling, pass 2 of 2: joint-bilateral upsample of the low-res model
// depth into a 2x-res depth texture whose edges follow the full-res COLOR edges.
//
// Why: the depth model's output has soft, patch-scale silhouettes -- measured ~8-9 low-res
// texels (~55 source px at 798x336 depth under a 5120-wide frame) of ramp at a face/window
// edge. Sampling that ramp bilinearly in the SBS reprojection smears and bends contours
// (bent sword handles, mushy nose profiles). For each output texel this pass gathers a
// (2r+1)^2 neighborhood of low-res depth, weighting each tap by spatial distance AND by how
// well the tap's guide color matches the full-res color at the output position. Taps on the
// wrong side of a color edge are suppressed, so the written depth steps where the COLOR
// steps -- silhouettes snap to the image, cutting the ramp to roughly one output texel.
//
// It cannot invent depth detail that the model never produced (e.g. a nose bump with no
// depth signal); it only relocates existing depth transitions onto color edges.
//
// Runs every frame (cheap: the taps read two low-res textures), so between depth-inference
// frames the STALE depth is re-snapped to the CURRENT frame's color edges -- silhouettes
// track slow motion even while the depth itself is a few frames old.

Texture2D<float>  DepthIn      : register(t0);  // in_w x in_h, un-dilated normalized depth
Texture2D<float4> GuideLow     : register(t1);  // in_w x in_h, tonemapped color guide (pass 1)
Texture2D<float4> ColorTexture : register(t2);  // full-res captured frame
SamplerState      LinearSampler : register(s0);
RWTexture2D<float> DepthOut    : register(u0);  // out_w x out_h (2x depth res), R32F

cbuffer Params : register(b0) {
    uint in_w;           // low-res depth map dims
    uint in_h;
    uint out_w;          // guided output dims
    uint out_h;
    float inv2sig_sp2;   // 1 / (2 * sigma_spatial^2), sigma in low-res texels
    float inv2sig_r2;    // 1 / (2 * sigma_range^2), range on tonemapped RGB distance
    uint is_hdr;
    float radius;        // kernel radius in low-res texels (spans the model's edge ramp)
};

float3 Tonemap(float3 c) {
    if (is_hdr) {
        c = c / (1.0f + c);  // Reinhard: linear scRGB -> [0,1)
    }
    return saturate(c);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= out_w || id.y >= out_h) {
        return;
    }
    float2 uv = (float2(id.xy) + 0.5f) / float2(out_w, out_h);

    // Reference color at this output position, from the FULL-RES frame (this is what makes
    // the result sharper than the low-res depth: the reference knows the true edge position).
    float3 centerColor = Tonemap(ColorTexture.SampleLevel(LinearSampler, uv, 0).rgb);

    // Position in low-res texel space (texel centers at integer coordinates).
    float2 lr = uv * float2(in_w, in_h) - 0.5f;
    int2 base = int2(round(lr));

    int r = (int) radius;
    float wsum = 0.0f;
    float dsum = 0.0f;
    float dmin = 1.0f;
    float dmax = 0.0f;

    // Foreground bias: scale each tap's weight up with its depth so that where the color
    // guide is AMBIGUOUS (e.g. an object's dark anti-aliased outline against a dark sky --
    // its color matches the background better than the object), the tie resolves toward the
    // NEARER surface. Without this, thin foreground objects get eroded: their outline strip
    // takes background depth, so one eye compresses it and the other stretches it (the
    // "handle looks thin in one eye" artifact). Slightly-fat foreground is the eye-friendly
    // failure direction (same rationale as iw3's edge dilation, but color-aware).
    const float FG_BIAS = 2.0f;

    [loop]
    for (int dy = -r; dy <= r; dy++) {
        [loop]
        for (int dx = -r; dx <= r; dx++) {
            int2 q = clamp(base + int2(dx, dy), int2(0, 0), int2((int) in_w - 1, (int) in_h - 1));
            float2 sp = float2(q) - lr;
            float3 dc = GuideLow.Load(int3(q, 0)).rgb - centerColor;
            float d = DepthIn.Load(int3(q, 0));
            float w = exp(-dot(sp, sp) * inv2sig_sp2 - dot(dc, dc) * inv2sig_r2) * (1.0f + FG_BIAS * d);
            wsum += w;
            dsum += d * w;
            dmin = min(dmin, d);
            dmax = max(dmax, d);
        }
    }

    // Degenerate window (no tap matches the reference color): fall back to plain bilinear
    // rather than dividing by ~0.
    float dOut = (wsum > 1e-4f) ? (dsum / wsum) : DepthIn.SampleLevel(LinearSampler, uv, 0);

    // Silhouette bimodal snap: across a depth CLIFF (local tap range > threshold; sloped
    // surfaces stay untouched), push the result toward whichever side it leans. The joint-
    // bilateral average otherwise leaves an "aura" of intermediate depth around silhouettes
    // whose value wobbles with the background texture; in the reprojection those noisy
    // in-between depths compete in near-ties at occlusion edges and the per-pixel winner
    // flips (the dotted fringe). Collapsing the aura makes the cliff clean and binary; the
    // reprojection's own smoothed sampling then widens it into a monotone, noise-free ramp.
    if (dmax - dmin > 0.2f) {
        float mid = 0.5f * (dmin + dmax);
        dOut = clamp(mid + (dOut - mid) * 3.0f, dmin, dmax);
    }
    DepthOut[id.xy] = dOut;
}

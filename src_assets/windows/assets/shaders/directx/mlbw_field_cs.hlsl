// Learned warp (iw3 MLBW), pass 3 of 3 (per eye): pack the model's raw output buffers
// (delta [L,FH,FW] = per-layer horizontal offsets in grid pixels, layer_weight [L,FH,FW]
// = softmax blend weights, L <= 4) into two RGBA32F field textures: DeltaTex = (d0..d3),
// WeightTex = (w0..w3), unused channels zero. A single bilinear tap of each texture in the
// composite PS then interpolates all layers at once. The right eye's model ran on flipped
// input, so its fields are read mirrored and the deltas NEGATED to express them in
// unflipped source coordinates.
//
// BOTH planes get a 3x3 tent pre-blur:
// - WEIGHTS: iw3 upsamples layer_weight with antialias=True (backward_warp.py) and plain
//   HW-bilinear instead produces a per-field-texel sawtooth on high-parallax silhouettes
//   (fullscreen close-ups; dump_20260706_085745_01). Tent + bilinear reproduces iw3's own
//   composite to mean|d| 0.0007.
// - DELTAS: goes BEYOND iw3 -- the disocclusion band (dark zipper beside foreground
//   silhouettes, present even in iw3's own render) is per-field-texel delta quantization;
//   the tent smooths it to near-2x-grid quality for free (a real 2x grid measured 4x cost,
//   ~31 ms/eye fp16 -- non-viable). Validated offline on the arm scene (both eyes) + head
//   scene: zipper strongly reduced, wispy hair/thin structures unchanged, no geometry bend
//   visible (the tent's reach is <=1 field texel).

StructuredBuffer<float> DeltaBuf  : register(t0);  // L*FH*FW
StructuredBuffer<float> WeightBuf : register(t1);  // L*FH*FW
RWTexture2D<float4> DeltaTex      : register(u0);  // FW x FH RGBA32F
RWTexture2D<float4> WeightTex     : register(u1);  // FW x FH RGBA32F

cbuffer Params : register(b0) {
    uint fw;
    uint fh;
    uint is_right;  // 1 = mirror + negate (input was flipped)
    uint layers;    // model layer count L (<= 4)
};

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= fw || id.y >= fh) {
        return;
    }
    uint xs = is_right ? (fw - 1 - id.x) : id.x;
    float sgn = is_right ? -1.0f : 1.0f;
    uint plane = fw * fh;
    float4 d = 0;
    float4 w = 0;
    [unroll]
    for (uint i = 0; i < 4; i++) {
        if (i < layers) {
            // 3x3 tent ([1 2 1]^2 / 16, clamped) over both planes, in SOURCE (possibly
            // mirrored) coordinates -- the kernel is symmetric so mirroring commutes, and
            // the delta negation is linear so it commutes with the blur too.
            float acc_d = 0.0f;
            float acc_w = 0.0f;
            [unroll]
            for (int dy = -1; dy <= 1; dy++) {
                uint yy = (uint) clamp((int) id.y + dy, 0, (int) fh - 1);
                [unroll]
                for (int dx = -1; dx <= 1; dx++) {
                    uint xx = (uint) clamp((int) xs + dx, 0, (int) fw - 1);
                    float k = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f);
                    acc_d += k * DeltaBuf[i * plane + yy * fw + xx];
                    acc_w += k * WeightBuf[i * plane + yy * fw + xx];
                }
            }
            d[i] = sgn * acc_d / 16.0f;
            w[i] = acc_w / 16.0f;
        }
    }
    DeltaTex[id.xy] = d;
    WeightTex[id.xy] = w;
}

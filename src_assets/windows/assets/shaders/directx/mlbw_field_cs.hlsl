// Learned warp (iw3 MLBW), pass 3 of 3 (per eye): pack the model's raw output buffers
// (delta [L,FH,FW] = per-layer horizontal offsets in grid pixels, layer_weight [L,FH,FW]
// = softmax blend weights, L <= 4) into two RGBA32F field textures: DeltaTex = (d0..d3),
// WeightTex = (w0..w3), unused channels zero. A single bilinear tap of each texture in the
// composite PS then interpolates all layers at once. The right eye's model ran on flipped
// input, so its fields are read mirrored and the deltas NEGATED to express them in
// unflipped source coordinates.

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
    uint idx = id.y * fw + xs;
    float4 d = 0;
    float4 w = 0;
    [unroll]
    for (uint i = 0; i < 4; i++) {
        if (i < layers) {
            d[i] = sgn * DeltaBuf[i * plane + idx];
            w[i] = WeightBuf[i * plane + idx];
        }
    }
    DeltaTex[id.xy] = d;
    WeightTex[id.xy] = w;
}

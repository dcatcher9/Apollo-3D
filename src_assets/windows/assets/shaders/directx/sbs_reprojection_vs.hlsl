struct VS_OUTPUT {
    float4 Pos : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VS_OUTPUT main_vs(uint vertex_id : SV_VertexID) {
    VS_OUTPUT output;
    
    // Generate a full screen triangle
    if (vertex_id == 0) {
        output.Pos = float4(-1.0f, -1.0f, 0.0f, 1.0f);
        output.TexCoord = float2(0.0f, 1.0f);
    } else if (vertex_id == 1) {
        output.Pos = float4(-1.0f, 3.0f, 0.0f, 1.0f);
        output.TexCoord = float2(0.0f, -1.0f);
    } else {
        output.Pos = float4(3.0f, -1.0f, 0.0f, 1.0f);
        output.TexCoord = float2(2.0f, 1.0f);
    }
    
    return output;
}

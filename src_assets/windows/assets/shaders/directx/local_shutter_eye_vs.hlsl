// One triangle that covers the whole back buffer. The pixel shader works from pixel positions.
float4 main_vs(uint vertex_id : SV_VertexID) : SV_Position
{
    return float4(vertex_id == 2 ? 3.0 : -1.0, vertex_id == 1 ? 3.0 : -1.0, 0.0, 1.0);
}

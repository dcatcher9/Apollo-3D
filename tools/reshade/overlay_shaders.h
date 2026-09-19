// SPDX-License-Identifier: GPL-3.0-only
// The native GUI vertex/target-0 shader and color conversions are adapted from
// ReShade 6.8.0, commit 18deaa52de0c425a78b329e9cb3c497281cd00ec,
// res/shaders/imgui_{vs_4_0,ps_4_0,hdr}.hlsl. Upstream license follows.
/*
Copyright 2014 Patrick Mours. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

namespace sunshine::overlay {
  inline constexpr char gui_shader[] = R"hlsl(
cbuffer PushConstants : register(b0) {
  float4x4 projection;
  uint color_space;
  float hdr_overlay_brightness;
};
Texture2D image : register(t0);
SamplerState linear_sampler : register(s0);
struct vertex { float2 pos : POSITION; float4 col : COLOR0; float2 uv : TEXCOORD0; };
struct fragment { float4 pos : SV_POSITION; float4 col : COLOR0; float2 uv : TEXCOORD0; };
fragment gui_vs(vertex v) {
  fragment f;
  f.pos = mul(projection, float4(v.pos, 0, 1));
  f.col = v.col;
  f.uv = v.uv;
  return f;
}
float3 linear_srgb(float3 c) {
  return float3(c.r <= .04045 ? c.r / 12.92 : pow((c.r + .055) / 1.055, 2.4),
                c.g <= .04045 ? c.g / 12.92 : pow((c.g + .055) / 1.055, 2.4),
                c.b <= .04045 ? c.b / 12.92 : pow((c.b + .055) / 1.055, 2.4));
}
float3 to2020(float3 c) {
  return mul(float3x3(.6274039149284363,.3292830288410187,.04331306740641594,
                     .06909728795289993,.9195404052734375,.01136231515556574,
                     .016391439363360405,.08801330626010895,.8955952525138855), c);
}
float3 pq_encode(float3 c) {
  c = pow(c, .1593017578125);
  return pow((.8359375 + 18.8515625 * c) / (1 + 18.6875 * c), 78.84375);
}
float3 pq_to_scrgb(float3 c) {
  c = pow(max(c, 0), 1.0 / 78.84375);
  c = pow(max(c - .8359375, 0) / max(18.8515625 - 18.6875 * c, 1e-8), 1.0 / .1593017578125);
  return mul(float3x3(1.660491002108435,-.58764113878855,-.072849863319885,
                     -.12455047452159,1.13289989712596,-.00834942260437,
                     -.01815076335491,-.10057889800801,1.11872966136292), c) * 125;
}
float hlg(float c) { return c <= 1.0/12.0 ? sqrt(3*c) : .17883277*log(12*c-.28466892)+.5599107146263123; }
float hlg_decode(float c) { return c <= .5 ? c*c/3 : (exp((c-.5599107146263123)/.17883277)+.28466892)/12; }
float3 hlg_to_scrgb(float3 c) {
  c = float3(hlg_decode(c.r), hlg_decode(c.g), hlg_decode(c.b));
  return mul(float3x3(1.660491002108435,-.58764113878855,-.072849863319885,
                     -.12455047452159,1.13289989712596,-.00834942260437,
                     -.01815076335491,-.10057889800801,1.11872966136292), c) * 12.5;
}
struct outputs { float4 native_color : SV_TARGET0; float4 layer : SV_TARGET1; };
outputs gui_ps(fragment f) {
  float4 v = f.col;
  if (color_space == 2) v.rgb = pow(v.rgb, 2.2) * hdr_overlay_brightness / 80;
  else if (color_space == 3) v.rgb = pq_encode(to2020(pow(v.rgb, 2.2)) * hdr_overlay_brightness / 10000);
  else if (color_space == 4) {
    float3 c = to2020(pow(v.rgb, 2.2)) * hdr_overlay_brightness / 1000;
    v.rgb = float3(hlg(c.r), hlg(c.g), hlg(c.b));
  }
  outputs o;
  o.native_color = image.Sample(linear_sampler, f.uv) * v;
  o.layer = o.native_color;
  // Capture straight linear color. Independent fixed-function blending stores
  // premultiplied color + coverage in FP16, including overlapping UI windows.
  if (color_space == 3) o.layer.rgb = pq_to_scrgb(o.layer.rgb);
  else if (color_space == 4) o.layer.rgb = hlg_to_scrgb(o.layer.rgb);
  else if (color_space != 2) o.layer.rgb = linear_srgb(o.layer.rgb);
  return o;
}
)hlsl";

  inline constexpr char composite_shader[] = R"hlsl(
Texture2D<float4> stereo : register(t0);
Texture2D<float4> layer : register(t1);
float4 fullscreen_vs(uint id : SV_VertexID) : SV_POSITION {
  return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0, 1);
}
float3 linear_srgb(float3 c) {
  return float3(c.r <= .04045 ? c.r / 12.92 : pow((c.r + .055) / 1.055, 2.4),
                c.g <= .04045 ? c.g / 12.92 : pow((c.g + .055) / 1.055, 2.4),
                c.b <= .04045 ? c.b / 12.92 : pow((c.b + .055) / 1.055, 2.4));
}
float3 encoded_srgb(float3 c) {
  return float3(c.r <= .0031308 ? 12.92*c.r : 1.055*pow(c.r, 1.0/2.4)-.055,
                c.g <= .0031308 ? 12.92*c.g : 1.055*pow(c.g, 1.0/2.4)-.055,
                c.b <= .0031308 ? 12.92*c.b : 1.055*pow(c.b, 1.0/2.4)-.055);
}
float4 composite_ps(float4 pos : SV_POSITION) : SV_TARGET {
  uint w, h;
  layer.GetDimensions(w, h);
  uint2 p = uint2(pos.xy);
  float4 source = stereo.Load(int3(p, 0));
  float4 ui = layer.Load(int3(uint2(p.x % w, p.y), 0));
  if (ui.a == 0) return source;
#if EXPORT_HDR
  source.rgb = ui.rgb + source.rgb * (1 - ui.a);
#else
  source.rgb = encoded_srgb(ui.rgb + linear_srgb(source.rgb) * (1 - ui.a));
#endif
  return source;
}
)hlsl";
}  // namespace sunshine::overlay

#pragma once

#ifdef _WIN32
#include <array>
#include <d3d11.h>

namespace models::host_sbs_subtitle_colour_commands {
// These layouts retain each caller's original binding widths and unbind points.
// They do not change the original palette -> emit -> 4CC/BOX3 -> accumulate recipe.
template<UINT ColourSrvs, UINT PaletteSrvs, UINT ColourBuffers, UINT GraphBuffers,
         bool UnbindBeforePalette>
struct layout_t {
  static constexpr UINT colour_srvs = ColourSrvs, palette_srvs = PaletteSrvs;
  static constexpr UINT colour_buffers = ColourBuffers, graph_buffers = GraphBuffers;
  static constexpr bool unbind_before_palette = UnbindBeforePalette;
};
using recorded_layout_t = layout_t<8u, 4u, 2u, 1u, true>;
using validated_layout_t = layout_t<9u, 9u, 2u, 1u, true>;
using enclosure_layout_t = layout_t<9u, 9u, 3u, 3u, false>;

struct bindings_t {
  ID3D11Buffer *request_constants = nullptr, *mode_arguments = nullptr, *root = nullptr;
  // Optional immutable {unit,mode,0,0} records. Other callers retain the
  // original mutable request path and its public diagnostic behaviour.
  std::array<ID3D11Buffer *, 128u> immutable_requests{};
  UINT root_offset = 0u;
  std::array<ID3D11Buffer *, 3u> colour_constants{}, graph_constants{};
  std::array<ID3D11ShaderResourceView *, 9u> colour_inputs{};
  std::array<ID3D11ShaderResourceView *, 4u> component_inputs{};
  // Accumulation appends current mode BODY, mode final and parent at t5..7.
  std::array<ID3D11ShaderResourceView *, 3u> accumulation_views{};
  std::array<ID3D11UnorderedAccessView *, 8u> palette_outputs{}, emission_outputs{},
      component_outputs{}, accumulation_outputs{};
  // Optional independent inner route. The original palette/arguments/accumulation stay untouched.
  ID3D11ShaderResourceView *trusted_palette = nullptr;
  ID3D11UnorderedAccessView *trusted_palette_output = nullptr, *trusted_mode_output = nullptr;
  ID3D11UnorderedAccessView *trusted_inner_output = nullptr, *trusted_publication_output = nullptr;
  ID3D11Buffer *trusted_mode_arguments = nullptr;
};

struct no_palette_hook_t { void operator()(UINT) const noexcept {} };
struct no_mode_hook_t { void operator()(UINT, UINT, UINT) const noexcept {} };

// The same command body is used on immediate and deferred contexts. No allocation,
// Get*, Map, readback, CPU colour decision or resource ownership belongs here.
// Hooks run after palette and accumulation have unbound their working views, so CBC
// can collect the current mode before the next mode reuses its scratch.
template<class Layout, std::size_t N, class Unbind,
         class AfterPalette = no_palette_hook_t, class AfterMode = no_mode_hook_t>
inline void emit_modes(ID3D11DeviceContext *context,
                       const std::array<ID3D11ComputeShader *, N> &colour,
                       const std::array<ID3D11ComputeShader *, 6u> &components,
                       const bindings_t &b, Unbind unbind,
                       AfterPalette after_palette = {}, AfterMode after_mode = {}) {
  static_assert(N >= 4u);
  static_assert(Layout::colour_srvs <= 9u && Layout::palette_srvs <= 9u);
  static_assert(Layout::colour_buffers <= 3u && Layout::graph_buffers <= 3u);
  constexpr std::array<UINT, 4u> zero{};
  for (UINT slot = 0u; slot < 8u; ++slot) {
    if constexpr (Layout::unbind_before_palette) unbind(context);
    // A root-zero palette dispatch does not write arguments; always clear them.
    context->ClearUnorderedAccessViewUint(b.palette_outputs[7], zero.data());
    auto colour_constants = b.colour_constants;
    auto graph_constants = b.graph_constants;
    const auto select_request = [&](UINT mode, UINT route = 0u) {
      if (auto *immutable = b.immutable_requests[route * 64u + slot * 8u + mode]) {
        colour_constants[1u] = immutable;
        if (b.graph_constants[1u] == b.request_constants) graph_constants[1u] = immutable;
      } else {
        colour_constants = b.colour_constants;
        graph_constants = b.graph_constants;
        const std::array<UINT, 4u> request{slot, mode, route, 0u};
        context->UpdateSubresource(b.request_constants, 0u, nullptr, request.data(), 0u, 0u);
      }
    };
    select_request(0u);
    context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
    auto palette_inputs = b.colour_inputs;
    palette_inputs[4] = nullptr; // Never bind the palette as both t4 and u0.
    context->CSSetShaderResources(0u, Layout::palette_srvs, palette_inputs.data());
    context->CSSetUnorderedAccessViews(0u, 8u, b.palette_outputs.data(), nullptr);
    context->CSSetShader(colour[0], nullptr, 0u);
    context->DispatchIndirect(b.root, b.root_offset);
    unbind(context);
    after_palette(slot);
    for (UINT mode = 0u; mode < 8u; ++mode) {
      select_request(mode);
      const UINT argument_offset = mode * 12u;
      context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
      context->CSSetShaderResources(0u, Layout::colour_srvs, b.colour_inputs.data());
      context->CSSetUnorderedAccessViews(0u, 8u, b.emission_outputs.data(), nullptr);
      context->CSSetShader(colour[1], nullptr, 0u);
      context->DispatchIndirect(b.mode_arguments, argument_offset);
      unbind(context);

      context->CSSetConstantBuffers(0u, Layout::graph_buffers, graph_constants.data());
      context->CSSetShaderResources(0u, 4u, b.component_inputs.data());
      context->CSSetUnorderedAccessViews(0u, 8u, b.component_outputs.data(), nullptr);
      for (UINT entry : {0u, 1u, 2u, 3u, 5u}) {
        context->CSSetShader(components[entry], nullptr, 0u);
        context->DispatchIndirect(b.mode_arguments, argument_offset);
      }
      unbind(context);

      context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
      auto accumulation_inputs = b.colour_inputs;
      for (UINT i = 0u; i < 3u; ++i) accumulation_inputs[5u + i] = b.accumulation_views[i];
      context->CSSetShaderResources(0u, Layout::colour_srvs, accumulation_inputs.data());
      context->CSSetUnorderedAccessViews(0u, 8u, b.accumulation_outputs.data(), nullptr);
      context->CSSetShader(colour[3], nullptr, 0u);
      context->DispatchIndirect(b.mode_arguments, argument_offset);
      unbind(context);
      after_mode(slot, mode, argument_offset);
    }
    if (b.trusted_inner_output) {
      // All original collectors have consumed this unit's scratch and palette. Capture a
      // separate modal palette, then reuse only the graph scratch; never call CBC hooks here.
      context->ClearUnorderedAccessViewUint(b.trusted_mode_output, zero.data());
      select_request(0u, 1u);
      auto trusted_palette_outputs = b.palette_outputs;
      trusted_palette_outputs[0u] = b.trusted_palette_output;
      trusted_palette_outputs[7u] = b.trusted_mode_output;
      context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
      context->CSSetShaderResources(0u, Layout::palette_srvs, palette_inputs.data());
      context->CSSetUnorderedAccessViews(0u, 8u, trusted_palette_outputs.data(), nullptr);
      context->CSSetShader(colour[0u], nullptr, 0u);
      context->DispatchIndirect(b.root, b.root_offset);
      unbind(context);
      auto trusted_inputs = b.colour_inputs;
      trusted_inputs[4u] = b.trusted_palette;
      for (UINT mode = 0u; mode < 8u; ++mode) {
        select_request(mode, 1u);
        const UINT offset = mode * 12u;
        context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
        context->CSSetShaderResources(0u, Layout::colour_srvs, trusted_inputs.data());
        context->CSSetUnorderedAccessViews(0u, 8u, b.emission_outputs.data(), nullptr);
        context->CSSetShader(colour[1u], nullptr, 0u);
        context->DispatchIndirect(b.trusted_mode_arguments, offset);
        unbind(context);
        context->CSSetConstantBuffers(0u, Layout::graph_buffers, graph_constants.data());
        context->CSSetShaderResources(0u, 4u, b.component_inputs.data());
        context->CSSetUnorderedAccessViews(0u, 8u, b.component_outputs.data(), nullptr);
        for (UINT entry : {0u, 1u, 2u, 3u}) {
          context->CSSetShader(components[entry], nullptr, 0u);
          context->DispatchIndirect(b.trusted_mode_arguments, offset);
        }
        unbind(context);
        auto accumulation_inputs = trusted_inputs;
        for (UINT i = 0u; i < 3u; ++i) accumulation_inputs[5u + i] = b.accumulation_views[i];
        std::array<ID3D11UnorderedAccessView *, 8u> trusted_outputs{};
        trusted_outputs[6u] = b.trusted_inner_output;
        trusted_outputs[7u] = b.trusted_publication_output;
        context->CSSetConstantBuffers(0u, Layout::colour_buffers, colour_constants.data());
        context->CSSetShaderResources(0u, Layout::colour_srvs, accumulation_inputs.data());
        context->CSSetUnorderedAccessViews(0u, 8u, trusted_outputs.data(), nullptr);
        context->CSSetShader(colour[3u], nullptr, 0u);
        context->DispatchIndirect(b.trusted_mode_arguments, offset);
        unbind(context);
      }
    }
  }
}
} // namespace models::host_sbs_subtitle_colour_commands
#endif

/**
 * @file src/platform/windows/host_sbs_v2_renderer.cpp
 * @brief Stateless D3D11 command recorder for one Host-SBS V2 fullscreen draw.
 */
#include "host_sbs_v2_renderer.h"

#ifdef _WIN32

namespace platf::dxgi {

  bool record_host_sbs_v2_draw(
    ID3D11DeviceContext *context,
    const host_sbs_v2_draw_command_t &command
  ) noexcept {
    if (!context) {
      return false;
    }
    if (command.render_target_count == 0u || command.render_target_count > command.render_targets.size()) {
      return false;
    }
    if (!command.vertex_shader || !command.pixel_shader || !command.sampler) {
      return false;
    }
    if (!command.shader_resources[0] || !command.geometry_constants) {
      return false;
    }
    for (UINT index = 0u; index < command.render_target_count; ++index) {
      if (!command.render_targets[index]) {
        return false;
      }
    }

    context->OMSetRenderTargets(
      command.render_target_count,
      command.render_targets.data(),
      nullptr
    );
    context->IASetInputLayout(nullptr);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(command.vertex_shader, nullptr, 0u);
    context->PSSetShader(command.pixel_shader, nullptr, 0u);
    context->RSSetViewports(1u, &command.viewport);
    auto *sampler = command.sampler;
    context->PSSetSamplers(0u, 1u, &sampler);
    context->PSSetShaderResources(
      0u,
      static_cast<UINT>(command.shader_resources.size()),
      command.shader_resources.data()
    );
    auto *geometry_constants = command.geometry_constants;
    context->PSSetConstantBuffers(2u, 1u, &geometry_constants);
    context->Draw(3u, 0u);

    context->OMSetRenderTargets(0u, nullptr, nullptr);
    std::array<ID3D11ShaderResourceView *, 6u> null_resources {};
    context->PSSetShaderResources(
      0u,
      static_cast<UINT>(null_resources.size()),
      null_resources.data()
    );
    return true;
  }

}  // namespace platf::dxgi

#endif  // _WIN32

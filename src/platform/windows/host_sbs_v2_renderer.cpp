/**
 * @file src/platform/windows/host_sbs_v2_renderer.cpp
 * @brief Stateless D3D11 command recorder for one Host-SBS V2 fullscreen draw.
 */
#include "host_sbs_v2_renderer.h"

#ifdef _WIN32
  #include <wrl/client.h>

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

    // Ordinary override=false preserves all legacy slots and predication. Queued override=true
    // requires both operands and no predication BEFORE binding/emitting anything. The caller can
    // then render current flat identity. Strong references preserve the original b5/t10 values.
    struct guard_scope_t {
      ID3D11DeviceContext *context = nullptr;
      Microsoft::WRL::ComPtr<ID3D11Buffer> cb;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;

      ~guard_scope_t() {
        if (context) {
          auto *old_cb = cb.Get();
          auto *old_srv = srv.Get();
          context->PSSetShaderResources(10u, 1u, &old_srv);
          context->PSSetConstantBuffers(5u, 1u, &old_cb);
        }
      }
    } guard_scope;

    if (command.queued_guard_override) {
      if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !command.queued_guard_constants || !command.queued_guard_transaction) {
        // Missing b5 must never read as a disabled/legacy gate for a queued draw.
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      context->GetPredication(&predicate, &value);
      if (predicate) {
        return false;
      }
      context->PSGetConstantBuffers(5u, 1u, guard_scope.cb.GetAddressOf());
      context->PSGetShaderResources(10u, 1u, guard_scope.srv.GetAddressOf());
      guard_scope.context = context;
      auto *cb = command.queued_guard_constants;
      auto *srv = command.queued_guard_transaction;
      context->PSSetConstantBuffers(5u, 1u, &cb);
      context->PSSetShaderResources(10u, 1u, &srv);
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
    auto *subtitle_constants = command.subtitle_constants;
    context->PSSetConstantBuffers(3u, 1u, &subtitle_constants);
    if (command.observe_bound_draw) {
      command.observe_bound_draw(command.bound_draw_observer_state, context);
    }
    context->Draw(3u, 0u);

    context->OMSetRenderTargets(0u, nullptr, nullptr);
    std::array<ID3D11ShaderResourceView *, 7u> null_resources {};
    context->PSSetShaderResources(
      0u,
      static_cast<UINT>(null_resources.size()),
      null_resources.data()
    );
    ID3D11Buffer *null_constants = nullptr;
    context->PSSetConstantBuffers(3u, 1u, &null_constants);
    return true;
  }

}  // namespace platf::dxgi

#endif  // _WIN32

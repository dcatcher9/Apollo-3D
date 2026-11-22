/**
 * @file src/platform/windows/host_sbs_v2_renderer.cpp
 * @brief Stateless D3D11 command recorder for one Host-SBS V2 fullscreen draw.
 */
#include "host_sbs_v2_renderer.h"

#include "src/host_sbs_eye_field.h"

#ifdef _WIN32
  #include <algorithm>
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

    if (command.shared_pin) {
      if (!command.shared_pin->geometry_constants) {
        return false;
      }
      for (auto *plane : command.shared_pin->planes) {
        if (!plane) {
          return false;
        }
      }
    }
    if (command.queued_guard_override && (!command.queued_guard_constants || !command.queued_guard_transaction)) {
      // Missing b5 must never read as a disabled/legacy gate for a queued draw.
      return false;
    }
    if (command.shared_pin || command.queued_guard_override) {
      if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
      BOOL value = FALSE;
      context->GetPredication(&predicate, &value);
      if (predicate) {
        return false;
      }
    }

    // Generic calls preserve caller-owned b5/t10 bindings. Shared-pin calls own these slots,
    // including their cleanup, and need no snapshot of the previous context state.
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

    if (command.queued_guard_override && !command.shared_pin) {
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
    const auto *resources = command.shader_resources.data();
    auto resource_count = static_cast<UINT>(command.shader_resources.size());
    UINT constant_count = 2u;
    std::array<ID3D11ShaderResourceView *, 15u> shared_resources {};
    std::array<ID3D11Buffer *, 5u> constants {command.geometry_constants, command.subtitle_constants};
    if (const auto &pin = command.shared_pin) {
      std::copy(command.shader_resources.begin(), command.shader_resources.end(), shared_resources.begin());
      shared_resources[7u] = pin->selected_receipt;
      shared_resources[10u] = command.queued_guard_override ? command.queued_guard_transaction : nullptr;
      std::copy(pin->planes.begin(), pin->planes.end(), shared_resources.begin() + 12u);
      constants[3u] = command.queued_guard_override ? command.queued_guard_constants : nullptr;
      constants[4u] = pin->geometry_constants;
      resources = shared_resources.data();
      resource_count = static_cast<UINT>(shared_resources.size());
      constant_count = static_cast<UINT>(constants.size());
    }
    context->PSSetShaderResources(0u, resource_count, resources);
    // A Base view carries its eye-specific limiter field. It rides in its own slot, outside
    // t0..t14, and is absent for views the producer attached none to (flat, replay, tests).
    const auto eye_field = models::host_sbs_eye_field::companion(command.shader_resources[1]);
    if (eye_field) {
      auto *eye_view = eye_field.Get();
      context->PSSetShaderResources(models::host_sbs_eye_field::pixel_slot, 1u, &eye_view);
    }
    context->PSSetConstantBuffers(2u, constant_count, constants.data());
    if (command.observe_bound_draw) {
      command.observe_bound_draw(command.bound_draw_observer_state, context);
    }
    context->Draw(3u, 0u);

    context->OMSetRenderTargets(0u, nullptr, nullptr);
    constexpr std::array<ID3D11ShaderResourceView *, 15u> null_resources {};
    constexpr std::array<ID3D11Buffer *, 4u> null_constants {};
    context->PSSetShaderResources(0u, resource_count, null_resources.data());
    if (eye_field) {
      ID3D11ShaderResourceView *null_view = nullptr;
      context->PSSetShaderResources(models::host_sbs_eye_field::pixel_slot, 1u, &null_view);
    }
    context->PSSetConstantBuffers(3u, constant_count - 1u, null_constants.data());
    return true;
  }

}  // namespace platf::dxgi

#endif  // _WIN32

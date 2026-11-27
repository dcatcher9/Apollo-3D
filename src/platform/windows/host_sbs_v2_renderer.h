/**
 * @file src/platform/windows/host_sbs_v2_renderer.h
 * @brief Stateless D3D11 command recorder for one Host-SBS V2 fullscreen draw.
 */
#pragma once

// standard includes
#include <array>
#include <optional>

// platform includes
#include <d3d11.h>

namespace platf::dxgi {

  /** Non-owning prepared pin planes and bindings owned by one production draw. */
  struct host_sbs_v2_shared_pin_bindings_t {
    /** Weights, conditioned field and current row/header evidence at t12..t14. */
    std::array<ID3D11ShaderResourceView *, 3u> planes {};
    /** Exact SharedPinGeometry constant buffer at b6. */
    ID3D11Buffer *geometry_constants = nullptr;
    /** Current selected-matte receipt at t7; null for the text-game route. */
    ID3D11ShaderResourceView *selected_receipt = nullptr;
  };

  /**
   * Non-owning operands for one already-authorized Host-SBS draw.
   *
   * The caller retains every lifecycle decision: live V2 versus fail-flat, optional P010 MRT,
   * packed-presentation reuse, direct replay, and offline timing. The recorder only emits the
   * common D3D11 binding/draw/unbind sequence. Without shared_pin, pixel constant buffers outside
   * b2/b3 (for example the live P010 matrix at b0 or direct-replay constants at b4) remain caller-owned.
   */
  struct host_sbs_v2_draw_command_t {
    std::array<ID3D11RenderTargetView *, 2u> render_targets {};
    UINT render_target_count = 1u;
    ID3D11VertexShader *vertex_shader = nullptr;
    ID3D11PixelShader *pixel_shader = nullptr;
    D3D11_VIEWPORT viewport {};
    ID3D11SamplerState *sampler = nullptr;
    /** Source, Base, state, native glyph matte, joined RGB snapshot, SCP, SCT. */
    std::array<ID3D11ShaderResourceView *, 7u> shader_resources {};
    /** Exact HostSbsV2Geometry/HostSbsFlatGeometry constant buffer at b2. */
    ID3D11Buffer *geometry_constants = nullptr;
    /** Optional HostSbsSubtitleRender constants at b3; null disables glyph selection. */
    ID3D11Buffer *subtitle_constants = nullptr;
    /** Own t0..t14 and b3..b6 for this draw; require an unpredicated immediate context.
     * The caller has already validated the prepared resources and their device ownership.
     */
    std::optional<host_sbs_v2_shared_pin_bindings_t> shared_pin;
    /** Requested diagnostics only: observe actual bindings before Draw and cleanup. */
    void *bound_draw_observer_state = nullptr;
    void (*observe_bound_draw)(void *, ID3D11DeviceContext *) noexcept = nullptr;
  };

  /** Record one fullscreen-triangle draw and leave all RTVs, t0..t6 and b3 unbound.
   * A shared-pin draw also clears t7..t14 and b4..b6; b2 remains the recorded geometry.
   */
  [[nodiscard]] bool record_host_sbs_v2_draw(
    ID3D11DeviceContext *context,
    const host_sbs_v2_draw_command_t &command
  ) noexcept;

}  // namespace platf::dxgi

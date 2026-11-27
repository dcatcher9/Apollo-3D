/**
 * @file src/platform/windows/host_sbs_pipeline.h
 * @brief The production Host SBS V2 pass shared by live Host 3D and the offline native pass.
 *
 * Live Host 3D (the converter in display_vram.cpp) and the offline native pass
 * (sbs_bench_harness.cpp, used by the evaluator and by Convert Video's worker) render one
 * production frame: the estimator's authenticated V2 Base/state field, the Direct Glyph selected
 * matte and the shared UI pin warp, recorded by record_host_sbs_v2_draw. This owner holds that
 * pass's setup (renderer authentication, subtitle programs, shared warp storage and the b2/b3
 * constant buffers) and its per-frame dispatch order (constants, draw operands, Direct Glyph pin
 * preparation and its ordinary-Base fallback). Capture, pacing, authority, NVENC, Game 3D routes,
 * file I/O, timelines and diagnostics stay with the callers.
 */
#pragma once

// standard includes
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

// platform includes
#include <d3d11.h>
#include <wrl/client.h>

// local includes
#include "host_sbs_v2_renderer.h"
#include "src/host_sbs_shader_cache.h"
#include "src/host_sbs_shared_ui_warp_runtime.h"
#include "src/host_sbs_subtitle_render.h"
#include "src/host_sbs_v2_geometry.h"
#include "src/video_depth_estimator.h"

namespace platf::dxgi::host_sbs_pipeline {

  // ---- Setup ----

  /** The installed `parallax_v2_live_renderer` closure group: fullscreen vertex and V2 pixel. */
  struct renderer_bytecode_t {
    /** Closure of the installed sources; compared with the pinned production identity. */
    std::string observed_closure;
    /** True only when the sources are readable and their closure is the pinned one. */
    bool authenticated = false;
    /** Both empty unless authenticated; either may still be empty if compilation failed. */
    models::host_sbs_shader_cache::bytecode_t vertex, pixel;
  };

  /** Authenticate the production renderer sources under `shader_root` and resolve its bytecode. */
  [[nodiscard]] renderer_bytecode_t load_renderer(const std::filesystem::path &shader_root);

  /** The s0 sampler every Host SBS pass binds: trilinear, clamped in U and V. */
  [[nodiscard]] D3D11_SAMPLER_DESC sampler_desc() noexcept;

  /** Direct Glyph and shared UI warp programs, built before the estimator's first input. */
  struct subtitle_programs_t {
    models::host_sbs_shared_ui_warp::programs_t shared_warp;
    models::subtitle_direct_glyph_configuration_t configuration;
    std::string error;
    bool ready = false;
  };

  /**
   * Device-object preparation only; it may run on a pipeline-build worker. Authenticates the
   * completed shared warp and its Direct Glyph prepare stage, then configures the estimator's
   * Direct Glyph route. The shared subtitle warp is the only renderer of that matte.
   */
  [[nodiscard]] subtitle_programs_t prepare_subtitle_programs(
    models::video_depth_estimator &estimator,
    ID3D11Device *device,
    const models::subtitle_direct_glyph_configuration_t &configuration
  );

  // ---- Per-frame values ----

  /**
   * b2 for one draw: full-source geometry, plus the exact video ROI and its tensor content when
   * the completed region is a valid video ROI. Empty when that ROI has no valid tensor content.
   */
  [[nodiscard]] std::optional<models::host_sbs_v2_geometry_t> geometry_constants(
    float content_scale_x,
    float content_scale_y,
    const models::depth_input_region_t *input_region = nullptr,
    const models::depth_tensor_content_rect_t *field_content = nullptr
  ) noexcept;

  /** b3 for one draw over the completed region's source rectangle. */
  [[nodiscard]] models::host_sbs_subtitle_render_constants_t subtitle_constants(
    bool enabled,
    models::input_color_space color_space,
    const models::depth_input_region_t &region
  ) noexcept;

  /** The completed subtitle operands one observation publishes for the draw. */
  struct subtitle_inputs_t {
    ID3D11ShaderResourceView *selected_matte = nullptr;  ///< t3 when the pin is selected
    ID3D11ShaderResourceView *display_band = nullptr;  ///< t4: joined source snapshot (contour source)
    ID3D11ShaderResourceView *condition_params = nullptr;  ///< t5: SCP
    ID3D11ShaderResourceView *contour_identity = nullptr;  ///< t6: SCT
    ID3D11ShaderResourceView *selected_receipt = nullptr;  ///< t7 when the pin is selected

    /** The band, condition and identity a glyph-enabled draw requires. */
    [[nodiscard]] bool complete() const noexcept {
      return display_band && condition_params && contour_identity;
    }
  };

  [[nodiscard]] subtitle_inputs_t subtitle_inputs(const models::depth_observation_views_t &completed) noexcept;

  /** The colour and geometry one draw samples. */
  struct frame_t {
    ID3D11ShaderResourceView *source = nullptr;  ///< t0: the colour actually drawn
    ID3D11ShaderResourceView *base = nullptr;  ///< t1: Base (or a replay's encoded field)
    ID3D11ShaderResourceView *state = nullptr;  ///< t2: ParallaxState (or the replay's health state)
  };

  /** Where and with which programs one draw renders. */
  struct output_t {
    ID3D11RenderTargetView *target = nullptr;
    ID3D11VertexShader *vertex_shader = nullptr;
    ID3D11PixelShader *pixel_shader = nullptr;
    D3D11_VIEWPORT viewport {};
    ID3D11SamplerState *sampler = nullptr;
  };

  /** The Direct Glyph pin a draw selected; empty views when it renders ordinary Base. */
  struct subtitle_pin_t {
    models::host_sbs_shared_ui_warp::views_t views;
    bool selected = false;
  };

  /** One converter's or one offline run's owner of the production pass. */
  class pass_t {
  public:
    pass_t() = default;
    pass_t(const pass_t &) = delete;
    pass_t &operator=(const pass_t &) = delete;

    // ---- Setup ----

    /** Adopt the programs a build prepared, before the estimator's first input. */
    void adopt(subtitle_programs_t programs);

    /** The stream stays flat: no later draw selects the subtitle pin. */
    void disable_subtitles() noexcept {
      subtitles_ready_ = false;
    }

    /** Drop the device-owned pin storage, for example when a converter resets its device. */
    void reset_warp() noexcept {
      warp_ = {};
    }

    /** The adopted subtitle programs authenticated and configured the estimator. */
    [[nodiscard]] bool subtitles_ready() const noexcept {
      return subtitles_ready_;
    }

    /** Programs for routes outside this pass that share them (text-game pin, diagnostics). */
    [[nodiscard]] models::host_sbs_shared_ui_warp::programs_t &programs() noexcept {
      return programs_;
    }

    [[nodiscard]] const models::host_sbs_shared_ui_warp::programs_t &programs() const noexcept {
      return programs_;
    }

    [[nodiscard]] const std::filesystem::path &shader_root() const noexcept {
      return shader_root_;
    }

    // ---- Per-frame constants ----

    /** Create b2 on first use and update it only when the value changes. */
    [[nodiscard]] HRESULT upload_geometry(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const models::host_sbs_v2_geometry_t &geometry
    );

    /** Create b3 on first use and update it only when the value changes. */
    [[nodiscard]] HRESULT upload_subtitle_constants(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const models::host_sbs_subtitle_render_constants_t &constants
    );

    [[nodiscard]] ID3D11Buffer *geometry_buffer() const noexcept {
      return geometry_buffer_.Get();
    }

    [[nodiscard]] ID3D11Buffer *subtitle_buffer() const noexcept {
      return subtitle_buffer_.Get();
    }

    // ---- Per-frame dispatch ----

    /**
     * The production draw operands: t0..t2 from the frame, t4..t6 and b3 when the draw carries
     * the completed subtitle operands, and this pass's b2. t3 stays empty until the pin is selected.
     */
    [[nodiscard]] host_sbs_v2_draw_command_t command(
      const output_t &output,
      const frame_t &frame,
      const subtitle_inputs_t *subtitles
    ) const noexcept;

    /**
     * Select the Direct Glyph shared UI pin for this draw when `receipt` grants it and b3 enables
     * subtitles. Selected, the draw samples through the completed shared-warp program and binds
     * t0..t14 and b2..b6. Otherwise it is the ordinary Base draw: no matte at t3 and subtitles
     * disabled in b3. Returns false only when b3 cannot be rewritten; the reason the pin was not
     * selected is in `error`.
     */
    [[nodiscard]] bool select_subtitle_pin(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      host_sbs_v2_draw_command_t &command,
      ID3D11ShaderResourceView *selected_matte,
      ID3D11ShaderResourceView *receipt,
      models::host_sbs_subtitle_render_constants_t &constants,
      bool materialize_inactive,
      subtitle_pin_t &pin,
      std::string &error
    );

  private:
    models::host_sbs_shared_ui_warp::programs_t programs_;
    models::host_sbs_shared_ui_warp::resources_t warp_;
    std::filesystem::path shader_root_;
    bool subtitles_ready_ = false;
    Microsoft::WRL::ComPtr<ID3D11Buffer> geometry_buffer_, subtitle_buffer_;
    std::optional<models::host_sbs_v2_geometry_t> geometry_value_;
    std::optional<models::host_sbs_subtitle_render_constants_t> subtitle_value_;
  };

}  // namespace platf::dxgi::host_sbs_pipeline

/**
 * @file src/platform/windows/host_sbs_pipeline.cpp
 * @brief The production Host SBS V2 pass shared by live Host 3D and the offline native pass.
 */
#include "host_sbs_pipeline.h"

#include "src/host_sbs_completed_subtitle_programs.h"

#include <algorithm>
#include <utility>

namespace platf::dxgi::host_sbs_pipeline {

  renderer_bytecode_t load_renderer(const std::filesystem::path &shader_root) {
    namespace cache = models::host_sbs_shader_cache;
    renderer_bytecode_t result;
    const auto sources = cache::snapshot_sources(shader_root, cache::parallax_v2_live_renderer_specs);
    result.observed_closure = cache::source_closure_sha256(sources);
    result.authenticated = sources && result.observed_closure == cache::parallax_v2_live_renderer_source_closure_sha256;
    if (result.authenticated) {
      result.pixel = cache::get(sources, cache::parallax_v2_live_renderer);
      result.vertex = cache::get(sources, cache::sbs_reprojection_vertex);
    }
    return result;
  }

  D3D11_SAMPLER_DESC sampler_desc() noexcept {
    D3D11_SAMPLER_DESC desc {};
    desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    desc.MinLOD = 0;
    desc.MaxLOD = D3D11_FLOAT32_MAX;
    return desc;
  }

  subtitle_programs_t prepare_subtitle_programs(
    models::video_depth_estimator &estimator,
    ID3D11Device *device,
    const models::subtitle_direct_glyph_configuration_t &configuration
  ) {
    subtitle_programs_t result;
    result.configuration = configuration;
    result.ready = models::host_sbs_completed_subtitle_programs::prepare(
      device, estimator, configuration, result.shared_warp, result.error
    );
    return result;
  }

  std::optional<models::host_sbs_v2_geometry_t> geometry_constants(
    const float content_scale_x,
    const float content_scale_y,
    const models::depth_input_region_t *input_region,
    const models::depth_tensor_content_rect_t *field_content
  ) noexcept {
    auto geometry = models::make_host_sbs_v2_full_frame_geometry(content_scale_x, content_scale_y);
    if (input_region && input_region->is_video_region() && input_region->valid()) {
      const auto &renderer_content = field_content ? *field_content : input_region->tensor_content;
      if (!renderer_content.valid()) {
        return std::nullopt;
      }
      const auto source_width = static_cast<float>(input_region->source_width);
      const auto source_height = static_cast<float>(input_region->source_height);
      geometry.video_roi_active = 1.0f;
      geometry.video_roi_left = static_cast<float>(input_region->left) / source_width;
      geometry.video_roi_top = static_cast<float>(input_region->top) / source_height;
      geometry.video_roi_right = static_cast<float>(input_region->right) / source_width;
      geometry.video_roi_bottom = static_cast<float>(input_region->bottom) / source_height;
      geometry.tensor_content_left = renderer_content.left;
      geometry.tensor_content_top = renderer_content.top;
      geometry.tensor_content_right = renderer_content.right;
      geometry.tensor_content_bottom = renderer_content.bottom;
    }
    return geometry;
  }

  models::host_sbs_subtitle_render_constants_t subtitle_constants(
    const bool enabled,
    const models::input_color_space color_space,
    const models::depth_input_region_t &region
  ) noexcept {
    models::host_sbs_subtitle_render_constants_t constants;
    constants.enabled = enabled ? 1u : 0u;
    constants.color_mode = static_cast<std::uint32_t>(color_space);
    constants.analysis_generation_low = static_cast<std::uint32_t>(region.analysis_generation);
    constants.analysis_generation_high = static_cast<std::uint32_t>(region.analysis_generation >> 32u);
    constants.source_left = region.left;
    constants.source_top = region.top;
    constants.source_width = region.width();
    constants.source_height = region.height();
    return constants;
  }

  subtitle_inputs_t subtitle_inputs(const models::depth_observation_views_t &completed) noexcept {
    return {
      .selected_matte = completed.subtitle_selected_matte.Get(),
      .display_band = completed.subtitle_contour_source.Get(),
      .condition_params = completed.subtitle_condition_params.Get(),
      .contour_identity = completed.subtitle_contour_identity.Get(),
      .selected_receipt = completed.subtitle_selected_matte_receipt.Get(),
    };
  }

  void pass_t::adopt(subtitle_programs_t programs) {
    programs_ = std::move(programs.shared_warp);
    shader_root_ = std::filesystem::path(programs.configuration.shader_root);
    subtitles_ready_ = programs.ready;
  }

  namespace {
    // A value-cached DEFAULT constant buffer: created with its first value, then rewritten only
    // when the value changes.
    template<class T>
    HRESULT upload_constants(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
      std::optional<T> &uploaded,
      const T &value
    ) {
      static_assert(sizeof(T) % 16u == 0u, "cbuffer must be 16-byte aligned");
      if (!buffer) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = sizeof(T);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial {};
        initial.pSysMem = &value;
        Microsoft::WRL::ComPtr<ID3D11Buffer> created;
        const auto status = device ? device->CreateBuffer(&desc, &initial, &created) : E_INVALIDARG;
        if (FAILED(status)) {
          return status;
        }
        buffer = std::move(created);
        uploaded = value;
      } else if (!uploaded || !(*uploaded == value)) {
        context->UpdateSubresource(buffer.Get(), 0u, nullptr, &value, 0u, 0u);
        uploaded = value;
      }
      return S_OK;
    }
  }  // namespace

  HRESULT pass_t::upload_geometry(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const models::host_sbs_v2_geometry_t &geometry
  ) {
    return upload_constants(device, context, geometry_buffer_, geometry_value_, geometry);
  }

  HRESULT pass_t::upload_subtitle_constants(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const models::host_sbs_subtitle_render_constants_t &constants
  ) {
    return upload_constants(device, context, subtitle_buffer_, subtitle_value_, constants);
  }

  host_sbs_v2_draw_command_t pass_t::command(
    const output_t &output,
    const frame_t &frame,
    const subtitle_inputs_t *subtitles
  ) const noexcept {
    return {
      .render_targets = {output.target, nullptr},
      .render_target_count = 1u,
      .vertex_shader = output.vertex_shader,
      .pixel_shader = output.pixel_shader,
      .viewport = output.viewport,
      .sampler = output.sampler,
      .shader_resources = {
        frame.source,
        frame.base,
        frame.state,
        nullptr,
        subtitles ? subtitles->display_band : nullptr,
        subtitles ? subtitles->condition_params : nullptr,
        subtitles ? subtitles->contour_identity : nullptr,
      },
      .geometry_constants = geometry_buffer_.Get(),
      .subtitle_constants = subtitles ? subtitle_buffer_.Get() : nullptr,
    };
  }

  bool pass_t::select_subtitle_pin(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    host_sbs_v2_draw_command_t &command,
    ID3D11ShaderResourceView *selected_matte,
    ID3D11ShaderResourceView *receipt,
    models::host_sbs_subtitle_render_constants_t &constants,
    const bool materialize_inactive,
    subtitle_pin_t &pin,
    std::string &error
  ) {
    namespace shared_warp = models::host_sbs_shared_ui_warp;
    pin = {};
    error.clear();
    command.shader_resources[3u] = receipt ? selected_matte : nullptr;
    // One fresh pin field uses the current actual draw source. The immutable draw keeps the
    // selected receipt/current RGB checks and performs no RGB reconstruction. Preparation and
    // the draw derive the same literal request from the recorder command.
    if (receipt && constants.enabled == 1u) {
      shared_warp::request_t request;
      std::copy(command.shader_resources.begin(), command.shader_resources.end(), request.draw_srvs.begin());
      request.draw_srvs[7u] = receipt;
      request.geometry_constants = command.geometry_constants;
      request.subtitle_constants = command.subtitle_constants;
      request.sampler = command.sampler;
      request.materialize_inactive = materialize_inactive;
      const auto program = programs_.direct_glyph_completed_program();
      if (warp_.prepare_subtitle(
            device, context, program, request,
            {constants.source_width, constants.source_height, constants.source_top},
            error
          ) &&
          warp_.matches(context, request)) {
        pin.views = warp_.views();
        pin.selected = true;
        command.pixel_shader = program.pixel;
        command.shared_pin = host_sbs_v2_shared_pin_bindings_t {
          {pin.views.weights.Get(), pin.views.field.Get(), pin.views.rows.Get()},
          pin.views.geometry.Get(),
          receipt,
        };
        return true;
      }
    }
    // Shared programs being ready does not guarantee THIS frame's preparation. This draw uses
    // ordinary authenticated Base geometry and current RGB, with no native render mask or
    // subtitle permission.
    command.shader_resources[3u] = nullptr;
    if (constants.enabled != 0u) {
      constants.enabled = 0u;
      if (FAILED(upload_subtitle_constants(device, context, constants))) {
        return false;
      }
    }
    return true;
  }

}  // namespace platf::dxgi::host_sbs_pipeline

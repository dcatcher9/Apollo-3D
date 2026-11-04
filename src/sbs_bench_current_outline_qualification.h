/** @file File-fed qualification of the actual completed current-outline draw. */
#pragma once

#ifdef _WIN32
  #include "host_sbs_current_outline_runtime_programs.h"
  #include "host_sbs_resolution.h"
  #include "host_sbs_selected_glyph_push_pull_runtime.h"
  #include "platform/windows/host_sbs_v2_renderer.h"
  #include "video_depth_estimator.h"

  #include <cstring>

namespace sbs_bench::current_outline_qualification {
  using Microsoft::WRL::ComPtr;
  namespace runtime = models::current_outline_runtime;
  namespace bg = models::selected_glyph_push_pull_bg_gpu_v1;
  namespace bg_runtime = models::selected_glyph_push_pull_runtime_v1;

  struct producer_stage_spec_t {
    std::size_t slot;
    const char *name;
    const char *role;
    UINT words;
    bool raw;
  };

  inline constexpr std::array producer_stage_specs {
    producer_stage_spec_t {2u, "cbc_meta", "current_outline_stages[2]:CBCMeta512", 512u, false},
    producer_stage_spec_t {7u, "raw_meta", "current_outline_stages[7]:RawMeta512", 512u, false},
    producer_stage_spec_t {8u, "f2_data", "current_outline_stages[8]:F2data68608", 68608u, true}
  };

  /** Diagnostic storage validation only; these aliases never grant draw or model authority. */
  inline bool producer_stage_buffer(ID3D11Device *device, ID3D11ShaderResourceView *view, const producer_stage_spec_t &spec, ComPtr<ID3D11Buffer> &buffer) {
    buffer.Reset();
    if (!device || !view) {
      return false;
    }
    ComPtr<ID3D11Device> actual_device;
    view->GetDevice(&actual_device);
    if (actual_device.Get() != device) {
      return false;
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
    view->GetDesc(&srv);
    const bool full_view = spec.raw ?
                             srv.ViewDimension == D3D11_SRV_DIMENSION_BUFFEREX && srv.Format == DXGI_FORMAT_R32_TYPELESS &&
                               srv.BufferEx.FirstElement == 0u && srv.BufferEx.NumElements == spec.words &&
                               srv.BufferEx.Flags == D3D11_BUFFEREX_SRV_FLAG_RAW :
                             srv.ViewDimension == D3D11_SRV_DIMENSION_BUFFER && srv.Format == DXGI_FORMAT_UNKNOWN &&
                               srv.Buffer.FirstElement == 0u && srv.Buffer.NumElements == spec.words;
    if (!full_view) {
      return false;
    }
    ComPtr<ID3D11Resource> resource;
    view->GetResource(&resource);
    if (FAILED(resource.As(&buffer))) {
      return false;
    }
    D3D11_BUFFER_DESC desc {};
    buffer->GetDesc(&desc);
    const UINT misc = spec.raw ? D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS : D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    if (desc.ByteWidth != spec.words * sizeof(std::uint32_t) || desc.StructureByteStride != (spec.raw ? 0u : 4u) || desc.MiscFlags != misc || (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0u || desc.Usage != D3D11_USAGE_DEFAULT || desc.CPUAccessFlags != 0u) {
      buffer.Reset();
      return false;
    }
    return true;
  }

  struct evidence_t {
    models::selected_matte_receipt_v1::words_t selected {};
    models::selected_matte_receipt_v1::contour_identity_t contour {};
    std::array<std::uint32_t, 6u> condition {};
    models::depth_coordinate_v2::state_words_t state {};
    std::array<std::uint32_t, 96u> background {};
    std::array<std::uint32_t, 16u> background_expected {};
    std::array<std::uint32_t, 8u> subtitle_constants {};
    std::uint64_t background_serial = 0u;
    bool bindings_observed = false;
  };

  /** Keeps actual production owners alive through the exact serialized file-fed draw. */
  class draw_t {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11PixelShader> completed_;
    bg_runtime::programs_t background_programs_;
    models::selected_glyph_receipt_c24_transport_v1::resources_t contour_transport_;
    bg::resources_t background_;
    models::push_pull_current_draw_lease_v1::lease_t background_lease_;

    template<std::size_t N>
    bool read_words(ID3D11DeviceContext *context, ID3D11Buffer *source, std::array<std::uint32_t, N> &words) const {
      if (!source) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      source->GetDesc(&desc);
      if (desc.ByteWidth != sizeof(words)) {
        return false;
      }
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device_->CreateBuffer(&desc, nullptr, &staging))) {
        return false;
      }
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        return false;
      }
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      context->Unmap(staging.Get(), 0u);
      return true;
    }

    template<std::size_t N>
    bool read_words(ID3D11DeviceContext *context, ID3D11ShaderResourceView *view, std::array<std::uint32_t, N> &words) const {
      if (!view) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Buffer> buffer;
      view->GetResource(&resource);
      return SUCCEEDED(resource.As(&buffer)) && read_words(context, buffer.Get(), words);
    }

    struct observer_t {
      ID3D11PixelShader *pixel = nullptr;
      std::array<ID3D11ShaderResourceView *, 12u> resources {};
      ID3D11Buffer *geometry = nullptr, *subtitle = nullptr, *background = nullptr;
      bool matched = false;

      static void observe(void *opaque, ID3D11DeviceContext *context) noexcept {
        auto &expected = *static_cast<observer_t *>(opaque);
        ComPtr<ID3D11PixelShader> pixel;
        context->PSGetShader(&pixel, nullptr, nullptr);
        expected.matched = pixel.Get() == expected.pixel;
        ID3D11ShaderResourceView *actual[12u] {};
        context->PSGetShaderResources(0u, 12u, actual);
        for (std::size_t i = 0u; i < expected.resources.size(); ++i) {
          expected.matched = expected.matched && actual[i] == expected.resources[i];
          if (actual[i]) {
            actual[i]->Release();
          }
        }
        ID3D11Buffer *constants[3u] {};
        context->PSGetConstantBuffers(2u, 3u, constants);
        expected.matched = expected.matched && constants[0u] == expected.geometry &&
                           constants[1u] == expected.subtitle && constants[2u] == expected.background;
        for (auto *constant : constants) {
          if (constant) {
            constant->Release();
          }
        }
      }
    };

  public:
    bool prepare(ID3D11Device *device, models::video_depth_estimator &estimator, const std::filesystem::path &shader_root, std::string &error) {
      device_ = device;
      const auto root = shader_root / "current_outline";
      runtime::pixel_programs_t pixels;
      if (!pixels.load(device, runtime::completed_root(root), runtime::completed_closure, {}, {}, error) || !background_programs_.load(device, root / "background", error)) {
        return false;
      }
      completed_.Attach(pixels.take_completed());
      models::subtitle_selected_matte_research_configuration_t config;
      config.current_outline = true;
      config.current_outline_shader_root = root.string();
      config.selector_shader_root = runtime::selector_root(root).string();
      config.selector_source_closure_sha256 = runtime::selector_closure;
      config.completed_shader_root = runtime::completed_root(root).string();
      config.completed_source_closure_sha256 = runtime::completed_closure;
      return estimator.configure_subtitle_selected_matte_research(config, error);
    }

    bool record(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary, ID3D11RenderTargetView *output, const models::estimate_result &result, evidence_t &evidence, std::string &error) {
      error.clear();
      if (!device_ || !context || !output || !result.research_selected_matte || !result.research_selected_matte_receipt || !result.completed_frame_valid || !models::parallax_v2_result_is_authenticated(result) || result.input_region.is_video_region() || result.subtitle_work_suppressed) {
        error = "current-outline qualification lacks its current completed full-source tuple";
        return false;
      }
      auto draw = ordinary;
      draw.render_targets = {output, nullptr};
      draw.render_target_count = 1u;
      draw.pixel_shader = completed_.Get();
      auto *legacy = ordinary.shader_resources[3u];
      auto *receipt = result.research_selected_matte_receipt.Get();
      draw.shader_resources[3u] = result.research_selected_matte.Get();
      if (!contour_transport_.ensure(device_.Get(), error) || !contour_transport_.queue(context, receipt, error) || !background_.ensure(device_.Get(), result.input_region.width(), models::subtitle_ocr_source_crop_height(result.input_region.width(), result.input_region.height()), error)) {
        return false;
      }
      bg::request_t request;
      std::copy(draw.shader_resources.begin(), draw.shader_resources.end(), request.draw_srvs.begin());
      request.draw_srvs[7u] = receipt;
      request.actual_contour_constants96 = contour_transport_.constants();
      request.actual_subtitle_render_constants32 = draw.subtitle_constants;
      const auto band = models::subtitle_ocr_source_crop_height(result.input_region.width(), result.input_region.height());
      request.analysis_shape = {result.input_region.width(), result.input_region.height(), result.input_region.height() - band, band};
      if (!background_.queue(context, background_programs_.program(), request, error)) {
        return false;
      }
      const auto views = background_.views();
      const models::push_pull_current_draw_lease_v1::lease_t::key_t color_key {
        result.completed_frame_id,
        result.publication.observation_timestamp_us,
        result.publication.publication_sequence,
        static_cast<std::uint64_t>(result.color_space)
      };
      background_lease_.capture(context, request, color_key, views.draw_serial);
      if (!background_lease_.matches(context, request, color_key, views.draw_serial)) {
        error = "current-outline qualification lost its exact draw-role lease";
        return false;
      }
      observer_t observer;
      observer.pixel = draw.pixel_shader;
      std::copy(draw.shader_resources.begin(), draw.shader_resources.end(), observer.resources.begin());
      observer.resources[7u] = receipt;
      observer.resources[8u] = legacy;
      observer.resources[9u] = views.background.Get();
      observer.resources[11u] = views.proof.Get();
      observer.geometry = draw.geometry_constants;
      observer.subtitle = draw.subtitle_constants;
      observer.background = views.expected_constants.Get();
      draw.bound_draw_observer_state = &observer;
      draw.observe_bound_draw = &observer_t::observe;
      {
        models::research_current_outline_selected_matte_v2::scoped_receipt_t selected;
        bg_runtime::scoped_background_t background;
        if (!selected.bind(context, receipt, legacy) || !background.bind(context, &views) || !platf::dxgi::record_host_sbs_v2_draw(context, draw) || !observer.matched) {
          error = "current-outline qualification did not observe exact selected/BG draw bindings";
          return false;
        }
      }
      evidence.bindings_observed = observer.matched;
      evidence.background_serial = views.draw_serial;
      if (!read_words(context, receipt, evidence.selected) || !read_words(context, result.subtitle_contour_identity.Get(), evidence.contour) || !read_words(context, result.subtitle_condition_params.Get(), evidence.condition) || !read_words(context, result.shadow_state.Get(), evidence.state) || !read_words(context, views.proof.Get(), evidence.background) || !read_words(context, views.expected_constants.Get(), evidence.background_expected) || !read_words(context, draw.subtitle_constants, evidence.subtitle_constants)) {
        error = "current-outline qualification cannot read its exact post-draw evidence";
        return false;
      }
      return true;
    }
  };
}  // namespace sbs_bench::current_outline_qualification
#endif

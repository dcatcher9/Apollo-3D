/** @file File-fed qualification of the actual completed current-outline draw. */
#pragma once

#ifdef _WIN32
  #include "host_sbs_completed_subtitle_programs.h"
  #include "host_sbs_resolution.h"
  #include "host_sbs_shared_ui_warp_runtime.h"
  #include "platform/windows/host_sbs_v2_renderer.h"
  #include "video_depth_estimator.h"

  #include <algorithm>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <vector>

namespace sbs_bench::current_outline_qualification {
  using Microsoft::WRL::ComPtr;
  namespace shared_warp = models::host_sbs_shared_ui_warp;
  using shared_warp::raw_model_prepare_closure;

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

  struct native_feature_spec_t {
    std::size_t slot;
    const char *name;
    const char *role;
  };

  inline constexpr std::array native_feature_specs {
    native_feature_spec_t {1u, "cbc_body", "current_outline_stages[1]:CBCBody"},
    native_feature_spec_t {4u, "cbc_selected_bits", "current_outline_stages[4]:CBCSelectedBits"},
    native_feature_spec_t {5u, "cbc_cover_bits", "current_outline_stages[5]:CBCCoverBits"},
    native_feature_spec_t {6u, "raw_m", "current_outline_stages[6]:RawM"}
  };

  /** Match the whole native feature texture from the completed composition lease. */
  inline bool native_feature_texture_matches(ID3D11Device *device, ID3D11ShaderResourceView *view, UINT width, UINT height) {
    if (!device || !view) {
      return false;
    }
    ComPtr<ID3D11Device> actual_device;
    view->GetDevice(&actual_device);
    D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
    view->GetDesc(&srv);
    if (actual_device.Get() != device || srv.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
        srv.Format != DXGI_FORMAT_R32_UINT || srv.Texture2D.MostDetailedMip != 0u || srv.Texture2D.MipLevels != 1u) {
      return false;
    }
    ComPtr<ID3D11Resource> resource;
    ComPtr<ID3D11Texture2D> texture;
    view->GetResource(&resource);
    if (!resource || FAILED(resource.As(&texture))) {
      return false;
    }
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    return desc.Width == width && desc.Height == height && desc.Format == DXGI_FORMAT_R32_UINT &&
           desc.MipLevels == 1u && desc.ArraySize == 1u && desc.SampleDesc.Count == 1u;
  }

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
    std::array<std::uint32_t, 8u> subtitle_constants {};
    std::array<UINT, 4u> shared_warp_shape {};
    std::vector<std::uint32_t> shared_warp_rows;
    std::uint32_t shared_warp_selected_receipt_header = 0u;
    std::uint64_t shared_warp_serial = 0u;
    bool bindings_observed = false;
    // A retained view of mutable output. Copy before the next draw changes it.
    ComPtr<ID3D11ShaderResourceView> shared_warp_field;
  };

  /** Keeps actual production owners alive through the exact serialized file-fed draw. */
  class draw_t {
    ComPtr<ID3D11Device> device_;
    shared_warp::programs_t programs_;
    shared_warp::resources_t shared_warp_;
    shared_warp::resources_t raw_model_warp_;
    bool reduced_ = false;

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

    bool read_rows(ID3D11DeviceContext *context, const shared_warp::views_t &views, evidence_t &evidence) const {
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Texture2D> texture, staging;
      if (!views.rows) return false;
      views.rows->GetResource(&resource);
      if (FAILED(resource.As(&texture))) return false;
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Width != 1u || desc.Height != views.shape[3u] + 1u || desc.Format != DXGI_FORMAT_R32_UINT) return false;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device_->CreateTexture2D(&desc, nullptr, &staging))) return false;
      context->CopyResource(staging.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return false;
      evidence.shared_warp_rows.resize(views.shape[3u]);
      for (UINT row = 0u; row <= views.shape[3u]; ++row) {
        std::uint32_t value;
        std::memcpy(&value, static_cast<const std::byte *>(mapped.pData) + row * mapped.RowPitch, sizeof(value));
        if (row == views.shape[3u]) evidence.shared_warp_selected_receipt_header = value;
        else evidence.shared_warp_rows[row] = value;
      }
      context->Unmap(staging.Get(), 0u);
      evidence.shared_warp_shape = views.shape;
      evidence.shared_warp_serial = views.serial;
      evidence.shared_warp_field.Reset();
      if (std::any_of(evidence.shared_warp_rows.begin(), evidence.shared_warp_rows.end(), [](auto row) { return row != 0u; })) {
        evidence.shared_warp_field = views.field;
      }
      return true;
    }

    struct observer_t {
      ID3D11PixelShader *pixel = nullptr;
      std::array<ID3D11ShaderResourceView *, 15u> resources {};
      ID3D11Buffer *geometry = nullptr, *subtitle = nullptr, *shared_geometry = nullptr;
      bool matched = false;

      static void observe(void *opaque, ID3D11DeviceContext *context) noexcept {
        auto &expected = *static_cast<observer_t *>(opaque);
        ComPtr<ID3D11PixelShader> pixel;
        context->PSGetShader(&pixel, nullptr, nullptr);
        expected.matched = pixel.Get() == expected.pixel;
        ID3D11ShaderResourceView *actual[15u] {};
        context->PSGetShaderResources(0u, 15u, actual);
        for (std::size_t i = 0u; i < expected.resources.size(); ++i) {
          // t8 is unconsumed by the shared renderer and carries no draw authority.
          if (i != 8u) expected.matched = expected.matched && actual[i] == expected.resources[i];
          if (actual[i]) {
            actual[i]->Release();
          }
        }
        ID3D11Buffer *constants[5u] {};
        context->PSGetConstantBuffers(2u, 5u, constants);
        expected.matched = expected.matched && constants[0u] == expected.geometry &&
                           constants[1u] == expected.subtitle && constants[2u] == nullptr &&
                           constants[4u] == expected.shared_geometry;
        for (auto *constant : constants) {
          if (constant) {
            constant->Release();
          }
        }
      }
    };

  public:
    bool prepare(ID3D11Device *device, models::video_depth_estimator &estimator, const std::filesystem::path &shader_root,
                 bool omit_foreground_edge, bool omit_stroke_coverage, bool weak_aa_oracle, std::string &error,
                 const std::string &reduced_shader_root = {}, bool direct_glyph = false,
                 bool capture_direct_plan = false, bool adaptive_direct_context = false,
                 bool direct_resolved_fringe = false, bool line_palette = false) {
      namespace completed = models::host_sbs_completed_subtitle_programs;
      auto config = completed::canonical_configuration(shader_root, direct_glyph, adaptive_direct_context);
      config.capture_direct_glyph_plan = capture_direct_plan;
      config.direct_glyph_line_palette = line_palette;
      config.direct_glyph_resolved_fringe = direct_resolved_fringe;
      config.current_outline_omit_foreground_edge = omit_foreground_edge;
      config.current_outline_omit_stroke_coverage = omit_stroke_coverage;
      config.current_outline_weak_aa_oracle = weak_aa_oracle;
      config.current_outline_raw_matte_reduced_shader_root = reduced_shader_root;
      if (!completed::prepare(device, estimator, config, programs_, error)) {
        return false;
      }
      device_ = device;
      reduced_ = direct_glyph || !reduced_shader_root.empty();
      return true;
    }

    bool prepare_raw_model(const std::filesystem::path &shader_root, std::string &error) {
      return programs_.load_raw_model_completed(device_.Get(), shader_root, error);
    }

    bool record(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary, ID3D11RenderTargetView *output, const models::depth_observation_views_t &result, ID3D11Buffer *subtitle_constants, evidence_t *evidence, std::string &error) {
      const auto program = reduced_ ? programs_.raw_model_completed_program() : programs_.program();
      return record_draw(context, ordinary, output, result, subtitle_constants, result.research_selected_matte.Get(),
                         result.research_selected_matte_receipt.Get(), shared_warp_, program, evidence, error);
    }

    bool record_raw_model(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary,
                          ID3D11RenderTargetView *output, const models::depth_observation_views_t &result,
                          ID3D11Buffer *subtitle_constants, ID3D11ShaderResourceView *candidate_matte,
                          ID3D11ShaderResourceView *candidate_receipt, evidence_t *evidence, std::string &error) {
      namespace receipt_abi = models::selected_matte_receipt_v1;
      const UINT width = result.input_region.width(), height = result.input_region.height();
      const UINT band = models::subtitle_ocr_source_crop_height(width, height);
      constexpr producer_stage_spec_t receipt_spec {0u, "selected_receipt", "raw-model-draw", receipt_abi::word_count, false};
      ComPtr<ID3D11Device> context_device;
      if (context) context->GetDevice(&context_device);
      ComPtr<ID3D11Buffer> receipt_buffer;
      if (!context || !evidence || !programs_.raw_model_completed_ready() || !device_ || context_device.Get() != device_.Get() ||
          !native_feature_texture_matches(device_.Get(), candidate_matte, width, band) ||
          !producer_stage_buffer(device_.Get(), candidate_receipt, receipt_spec, receipt_buffer)) {
        error = "Raw model shared-warp draw has incomplete native inputs";
        return false;
      }
      auto *rule_matte = result.research_selected_matte.Get();
      auto *rule_receipt = result.research_selected_matte_receipt.Get();
      const bool fallback = candidate_matte == rule_matte && candidate_receipt == rule_receipt;
      if (!fallback && (candidate_matte == rule_matte || candidate_receipt == rule_receipt)) {
        error = "Raw model shared-warp draw has mismatched Rule and model bindings";
        return false;
      }
      receipt_abi::words_t rule_words, candidate_words;
      const auto token = result.publication.transaction_token;
      ComPtr<ID3D11Buffer> rule_receipt_buffer;
      if (!producer_stage_buffer(device_.Get(), rule_receipt, receipt_spec, rule_receipt_buffer) ||
          !read_words(context, rule_receipt_buffer.Get(), rule_words) || rule_words[0u] != receipt_abi::tag ||
          rule_words[1u] != receipt_abi::schema || rule_words[2u] != 3u || rule_words[3u] != 0u ||
          rule_words[60u] != static_cast<std::uint32_t>(token) ||
          rule_words[61u] != static_cast<std::uint32_t>(token >> 32u) || token == 0u ||
          rule_words[62u] != receipt_abi::current_outline_algorithm || rule_words[63u] != 0u) {
        error = "Raw model shared-warp draw lacks its exact current Rule authority";
        return false;
      }
      if (fallback) {
        candidate_words = rule_words;
      } else {
        auto expected = rule_words;
        expected[2u] = 4u;
        expected[62u] = 3u;
        if (!read_words(context, receipt_buffer.Get(), candidate_words) || candidate_words != expected) {
          error = "Raw model shared-warp receipt differs from its current Rule tuple";
          return false;
        }
      }
      const auto program = programs_.raw_model_completed_program();
      if (!record_draw(context, ordinary, output, result, subtitle_constants, candidate_matte, candidate_receipt,
                       raw_model_warp_, program, evidence, error)) {
        return false;
      }
      if (evidence->selected != candidate_words || evidence->shared_warp_selected_receipt_header != 1u) {
        error = "Raw model shared-warp prepare did not authenticate the selected draw pair";
        return false;
      }
      return true;
    }

    bool record_reduced_raw_model(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary,
                                  ID3D11RenderTargetView *output, const models::depth_observation_views_t &result,
                                  ID3D11Buffer *subtitle_constants, evidence_t &evidence, std::string &error) {
      if (!programs_.raw_model_completed_ready()) {
        error = "Reduced RawMatte draw has no typed shared-warp prepare";
        return false;
      }
      const auto program = programs_.raw_model_completed_program();
      if (!record_draw(context, ordinary, output, result, subtitle_constants, result.research_selected_matte.Get(),
            result.research_selected_matte_receipt.Get(), raw_model_warp_, program, &evidence, error)) {
        return false;
      }
      const auto &selected = evidence.selected;
      const bool model = selected[2u] == 4u && selected[62u] == 3u;
      const bool legacy = selected[2u] == 1u && selected[62u] == 1u;
      if (selected[3u] != 0u || (!model && !legacy) || evidence.shared_warp_selected_receipt_header != 1u) {
        error = "Reduced RawMatte draw did not bind a current model or current legacy pair";
        return false;
      }
      return true;
    }

  private:
    bool record_draw(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary,
                     ID3D11RenderTargetView *output, const models::depth_observation_views_t &result,
                     ID3D11Buffer *subtitle_constants, ID3D11ShaderResourceView *matte, ID3D11ShaderResourceView *receipt,
                     shared_warp::resources_t &warp, const shared_warp::program_t &program,
                     evidence_t *evidence, std::string &error) {
      error.clear();
      if (!device_ || !context || !output || !subtitle_constants || !matte || !receipt || !result.completed_frame_valid || !models::parallax_v2_result_is_authenticated(result) || result.input_region.is_video_region() || result.subtitle_work_suppressed) {
        error = "shared subtitle draw lacks its current completed full-source tuple";
        return false;
      }
      auto draw = ordinary;
      draw.render_targets = {output, nullptr};
      draw.render_target_count = 1u;
      draw.pixel_shader = program.pixel;
      draw.subtitle_constants = subtitle_constants;
      draw.bound_draw_observer_state = nullptr;
      draw.observe_bound_draw = nullptr;
      draw.shader_resources[3u] = matte;
      draw.shader_resources[4u] = result.subtitle_contour_source.Get();
      draw.shader_resources[5u] = result.subtitle_condition_params.Get();
      draw.shader_resources[6u] = result.subtitle_contour_identity.Get();
      const UINT width = result.input_region.width(), height = result.input_region.height();
      const UINT band = models::subtitle_ocr_source_crop_height(width, height);
      const UINT band_top = height - band;
      const UINT padded_top = band_top > 0u ? band_top - 1u : 0u;
      const std::array<UINT, 4u> shape {width, height, padded_top, height - padded_top};
      if (!warp.ensure(device_.Get(), shape, error)) {
        return false;
      }
      shared_warp::request_t request;
      std::copy(draw.shader_resources.begin(), draw.shader_resources.end(), request.draw_srvs.begin());
      request.draw_srvs[7u] = receipt;
      request.geometry_constants = draw.geometry_constants;
      request.subtitle_constants = draw.subtitle_constants;
      request.sampler = draw.sampler;
      request.materialize_inactive = false; // Qualification reads rows and rendered pixels.
      if (!warp.queue(context, program, request, error) || !warp.matches(context, request)) {
        return false;
      }
      const auto views = warp.views();
      observer_t observer;
      if (evidence) {
        observer.pixel = draw.pixel_shader;
        std::copy(draw.shader_resources.begin(), draw.shader_resources.end(), observer.resources.begin());
        observer.resources[7u] = receipt;
        observer.resources[12u] = views.weights.Get();
        observer.resources[13u] = views.field.Get();
        observer.resources[14u] = views.rows.Get();
        observer.geometry = draw.geometry_constants;
        observer.subtitle = draw.subtitle_constants;
        observer.shared_geometry = views.geometry.Get();
        draw.bound_draw_observer_state = &observer;
        draw.observe_bound_draw = &observer_t::observe;
      }
      {
        models::research_current_outline_selected_matte_v2::scoped_receipt_t selected;
        shared_warp::scoped_binding_t shared;
        if (!selected.bind(context, receipt) || !shared.bind(context, &views) || !platf::dxgi::record_host_sbs_v2_draw(context, draw) || (evidence && !observer.matched)) {
          error = "current-outline qualification did not observe exact selected/shared-warp draw bindings";
          return false;
        }
      }
      if (!evidence) return true; // Follow conversion adds no observer or diagnostic readback.
      evidence->bindings_observed = observer.matched;
      if (!read_words(context, receipt, evidence->selected) || !read_words(context, result.subtitle_contour_identity.Get(), evidence->contour) || !read_words(context, result.subtitle_condition_params.Get(), evidence->condition) || !read_words(context, result.shadow_state.Get(), evidence->state) || !read_words(context, draw.subtitle_constants, evidence->subtitle_constants) || !read_rows(context, views, *evidence)) {
        error = "current-outline qualification cannot read its exact post-draw evidence";
        return false;
      }
      return true;
    }
  };
}  // namespace sbs_bench::current_outline_qualification
#endif

/** @file File-fed qualification of the actual completed Direct Glyph subtitle draw. */
#pragma once

#include "host_sbs_completed_subtitle_programs.h"
#include "host_sbs_resolution.h"
#include "host_sbs_selected_matte_receipt.h"
#include "host_sbs_shared_ui_warp_runtime.h"
#include "platform/windows/host_sbs_v2_renderer.h"
#include "video_depth_estimator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sbs_bench::subtitle_qualification {
  using Microsoft::WRL::ComPtr;
  namespace shared_warp = models::host_sbs_shared_ui_warp;

  struct producer_stage_spec_t {
    std::size_t slot;
    const char *name;
    const char *role;
    UINT words;
    bool raw;
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
    bool prepare(ID3D11Device *device, models::video_depth_estimator &estimator,
                 const models::subtitle_direct_glyph_configuration_t &configuration, std::string &error) {
      if (!models::host_sbs_completed_subtitle_programs::prepare(device, estimator, configuration, programs_, error)) {
        return false;
      }
      device_ = device;
      return true;
    }

    bool record(ID3D11DeviceContext *context, const platf::dxgi::host_sbs_v2_draw_command_t &ordinary, ID3D11RenderTargetView *output, const models::depth_observation_views_t &result, ID3D11Buffer *subtitle_constants, evidence_t *evidence, std::string &error) {
      return record_draw(context, ordinary, output, result, subtitle_constants, result.subtitle_selected_matte.Get(),
                         result.subtitle_selected_matte_receipt.Get(), shared_warp_, programs_.direct_glyph_completed_program(),
                         evidence, error);
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
        shared_warp::scoped_receipt_t selected;
        shared_warp::scoped_binding_t shared;
        if (!selected.bind(context, receipt) || !shared.bind(context, &views) || !platf::dxgi::record_host_sbs_v2_draw(context, draw) || (evidence && !observer.matched)) {
          error = "subtitle qualification did not observe exact selected/shared-warp draw bindings";
          return false;
        }
      }
      if (!evidence) return true; // Follow conversion adds no observer or diagnostic readback.
      evidence->bindings_observed = observer.matched;
      if (!read_words(context, receipt, evidence->selected) || !read_words(context, result.subtitle_contour_identity.Get(), evidence->contour) || !read_words(context, result.subtitle_condition_params.Get(), evidence->condition) || !read_words(context, result.shadow_state.Get(), evidence->state) || !read_words(context, draw.subtitle_constants, evidence->subtitle_constants) || !read_rows(context, views, *evidence)) {
        error = "subtitle qualification cannot read its exact post-draw evidence";
        return false;
      }
      return true;
    }
  };
}  // namespace sbs_bench::subtitle_qualification

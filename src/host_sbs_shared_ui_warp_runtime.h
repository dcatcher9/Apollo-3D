/** @file Shared Game 3D pinning and one-source-sample Host subtitle renderer. */
#pragma once

#ifdef _WIN32
  #include "host_sbs_compute_program_lease.h"
  #include "host_sbs_shader_cache.h"

  #include <algorithm>
  #include <array>
  #include <cstdint>
  #include <exception>
  #include <filesystem>
  #include <limits>
  #include <string>
  #include <string_view>
  #include <utility>
  #include <d3d11.h>
  #include <wrl/client.h>

namespace models::host_sbs_shared_ui_warp {
  namespace cache = host_sbs_shader_cache;
  using Microsoft::WRL::ComPtr;

  inline constexpr std::array<cache::shader_spec, 6u> specs {{
    {"host_sbs_shared_ui_warp.hlsl", "initialize_main", "cs_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "SunshineApplyUICS", "cs_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_main_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "initialize_main", "cs_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "SunshineApplyUICS", "cs_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_main_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view closure = "21c15ac603ca0a61b415b0a5889a57b1827b818dfafcedbfc5d6305c5989dd01";
  inline constexpr std::array<cache::shader_spec, 6u> diagnostic_specs {{
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_mapping_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_coverage_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_boundary_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_mapping_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_coverage_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_boundary_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view diagnostic_closure = "b42715f138eb69ab87a8af5c2314600f1bb274200a0ad31a3cce0a966a5c3916";

  struct program_t {
    ID3D11ComputeShader *initialize = nullptr, *pin = nullptr;
    ID3D11PixelShader *pixel = nullptr;
    bool queued = false;
  };

  // Preparation only creates device objects from one immutable authenticated source snapshot.
  // It never uses the immediate context and can run on the tracked pipeline-build worker.
  class programs_t {
    struct route_t {
      std::array<ComPtr<ID3D11ComputeShader>, 2u> compute;
      ComPtr<ID3D11PixelShader> pixel;
    };
    std::array<route_t, 2u> routes_;
    std::array<ComPtr<ID3D11PixelShader>, 6u> diagnostics_;

  public:
    bool load(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        const auto source = cache::snapshot_sources(root, specs);
        if (!device || !source || cache::source_closure_sha256(source) != closure) {
          error = "Shared UI warp shader source mismatch";
          return false;
        }
        programs_t next;
        for (std::size_t route = 0u; route < 2u; ++route) {
          for (std::size_t stage = 0u; stage < 3u; ++stage) {
            const auto code = cache::get(source, specs[route * 3u + stage]);
            if (!code || code->empty() || (stage < 2u ?
                FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &next.routes_[route].compute[stage])) :
                FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &next.routes_[route].pixel)))) {
              error = "Shared UI warp shader creation failed";
              return false;
            }
          }
        }
        *this = std::move(next);
        return true;
      } catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }

    bool load_diagnostics(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      // Diagnostics belong to the same device as the retained production programs.
      // Their immutable authenticated objects can then be reused without re-reading source.
      ComPtr<ID3D11Device> program_device;
      if (routes_[0].pixel) routes_[0].pixel->GetDevice(&program_device);
      if (!device || program_device.Get() != device) {
        error = "Shared UI warp diagnostic device does not match its production programs";
        return false;
      }
      if (std::all_of(diagnostics_.begin(), diagnostics_.end(), [](const auto &shader) { return bool(shader); })) return true;
      try {
        const auto source = cache::snapshot_sources(root, diagnostic_specs);
        if (!source || cache::source_closure_sha256(source) != diagnostic_closure) {
          error = "Shared UI warp diagnostic source mismatch";
          return false;
        }
        std::array<ComPtr<ID3D11PixelShader>, 6u> next;
        for (std::size_t i = 0u; i < next.size(); ++i) {
          const auto code = cache::get(source, diagnostic_specs[i]);
          if (!code || code->empty() || FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &next[i]))) {
            error = "Shared UI warp diagnostic shader creation failed";
            return false;
          }
        }
        diagnostics_ = std::move(next);
        return true;
      } catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }

    [[nodiscard]] bool ready() const noexcept { return routes_[0].pixel && routes_[1].pixel; }
    [[nodiscard]] program_t program(bool queued = false) const noexcept {
      const auto route = queued ? 1u : 0u;
      const auto &shaders = routes_[route];
      return {shaders.compute[0].Get(), shaders.compute[1].Get(), shaders.pixel.Get(), queued};
    }
    [[nodiscard]] ID3D11PixelShader *mapping(bool queued = false) const noexcept { return diagnostics_[queued ? 3u : 0u].Get(); }
    [[nodiscard]] ID3D11PixelShader *coverage(bool queued = false) const noexcept { return diagnostics_[queued ? 4u : 1u].Get(); }
    [[nodiscard]] ID3D11PixelShader *boundary(bool queued = false) const noexcept { return diagnostics_[queued ? 5u : 2u].Get(); }
  };

  struct request_t {
    std::array<ID3D11ShaderResourceView *, 8u> draw_srvs {}; // literal t0..7, including the selected mask receipt
    ID3D11Buffer *geometry_constants = nullptr, *subtitle_constants = nullptr;
    ID3D11SamplerState *sampler = nullptr;
    ID3D11Buffer *queued_guard_constants = nullptr;
    ID3D11ShaderResourceView *queued_guard_transaction = nullptr;
    bool queued = false;
  };

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> weights, field, rows;
    ComPtr<ID3D11Buffer> geometry;
    std::array<UINT, 4u> shape {}; // full source W/H, native full-source band top/height
    std::uint64_t serial = 0u;
  };

  class resources_t {
    struct plane_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };
    ComPtr<ID3D11Device> device_;
    plane_t field_, weights_, rows_;
    ComPtr<ID3D11Buffer> geometry_;
    std::array<UINT, 4u> shape_ {};
    host_sbs_compute_program_lease::group_t<2u> programs_;
    std::array<ComPtr<ID3D11ShaderResourceView>, 8u> inputs_;
    ComPtr<ID3D11Buffer> draw_geometry_, draw_subtitle_, guard_;
    ComPtr<ID3D11ShaderResourceView> transaction_;
    ComPtr<ID3D11SamplerState> sampler_;
    ID3D11DeviceContext *context_ = nullptr;
    std::uint64_t serial_ = 0u;
    bool issued_ = false, queued_ = false;

    template<class T> bool same_device(T *object) const {
      if (!object) return false;
      ComPtr<ID3D11Device> actual;
      object->GetDevice(&actual);
      return actual.Get() == device_.Get();
    }

    bool constant(ID3D11Buffer *buffer, UINT bytes) const {
      if (!same_device(buffer)) return false;
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      return desc.ByteWidth == bytes && desc.BindFlags == D3D11_BIND_CONSTANT_BUFFER;
    }

    // Restore every touched slot, sampler, shader and dynamic class instance. The pin planes
    // are reserved PS12..14 outputs; unbind them while writing to avoid SRV/UAV hazards.
    struct saved_t {
      ID3D11DeviceContext *context;
      std::array<ComPtr<ID3D11ShaderResourceView>, 15u> srvs;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 3u> uavs;
      std::array<ComPtr<ID3D11Buffer>, 7u> constants;
      std::array<ComPtr<ID3D11ShaderResourceView>, 3u> pixel;
      ComPtr<ID3D11SamplerState> sampler;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT count = 256u;

      explicit saved_t(ID3D11DeviceContext *c): context(c) {
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[3] {};
        ID3D11Buffer *b[7] {};
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0u, 15u, s); c->CSGetUnorderedAccessViews(0u, 3u, u);
        c->CSGetConstantBuffers(0u, 7u, b); c->CSGetSamplers(0u, 1u, &sampler);
        c->CSGetShader(&shader, ci, &count); c->PSGetShaderResources(12u, 3u, p);
        for (std::size_t i = 0u; i < srvs.size(); ++i) srvs[i].Attach(s[i]);
        for (std::size_t i = 0u; i < uavs.size(); ++i) uavs[i].Attach(u[i]);
        for (std::size_t i = 0u; i < constants.size(); ++i) constants[i].Attach(b[i]);
        for (std::size_t i = 0u; i < pixel.size(); ++i) pixel[i].Attach(p[i]);
        for (UINT i = 0u; i < count; ++i) instances[i].Attach(ci[i]);
      }
      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[3] {};
        c->CSSetShaderResources(0u, 15u, s); c->CSSetUnorderedAccessViews(0u, 3u, u, nullptr);
        c->PSSetShaderResources(12u, 3u, p);
      }
      ~saved_t() {
        unbind(context);
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[3] {};
        ID3D11Buffer *b[7] {};
        ID3D11ClassInstance *ci[256] {};
        for (std::size_t i = 0u; i < srvs.size(); ++i) s[i] = srvs[i].Get();
        for (std::size_t i = 0u; i < uavs.size(); ++i) u[i] = uavs[i].Get();
        for (std::size_t i = 0u; i < constants.size(); ++i) b[i] = constants[i].Get();
        for (std::size_t i = 0u; i < pixel.size(); ++i) p[i] = pixel[i].Get();
        for (UINT i = 0u; i < count; ++i) ci[i] = instances[i].Get();
        context->CSSetShaderResources(0u, 15u, s); context->CSSetUnorderedAccessViews(0u, 3u, u, nullptr);
        context->CSSetConstantBuffers(0u, 7u, b);
        auto *old_sampler = sampler.Get(); context->CSSetSamplers(0u, 1u, &old_sampler);
        context->CSSetShader(shader.Get(), ci, count); context->PSSetShaderResources(12u, 3u, p);
      }
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) noexcept = default;
    resources_t &operator=(resources_t &&) noexcept = default;

    bool ensure(ID3D11Device *device, const std::array<UINT, 4u> &shape, std::string &error) {
      issued_ = false;
      error.clear();
      // The rows plane needs one extra texel for the selected-receipt header.
      if (!device || shape[0] < 8u || shape[0] > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
          shape[1] == 0u || shape[1] > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
          shape[2] >= shape[1] || shape[3] == 0u || shape[3] > shape[1] - shape[2] ||
          shape[3] >= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        error = "Shared UI warp native source domain unavailable"; return false;
      }
      if (device_.Get() == device && shape_ == shape && geometry_) return true;
      resources_t next;
      next.device_ = device; next.shape_ = shape; next.serial_ = serial_;
      const auto plane = [&](UINT width, UINT height, DXGI_FORMAT format, plane_t &output) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1u;
        desc.Format = format; desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &output.texture)) &&
               SUCCEEDED(device->CreateShaderResourceView(output.texture.Get(), nullptr, &output.srv)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(output.texture.Get(), nullptr, &output.uav));
      };
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(shape); desc.Usage = D3D11_USAGE_IMMUTABLE; desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA initial {shape.data(), 0u, 0u};
      if (!plane(shape[0], shape[3], DXGI_FORMAT_R32_FLOAT, next.field_) ||
          !plane(shape[0], shape[3], DXGI_FORMAT_R32_FLOAT, next.weights_) ||
          !plane(1u, shape[3] + 1u, DXGI_FORMAT_R32_UINT, next.rows_) ||
          FAILED(device->CreateBuffer(&desc, &initial, &next.geometry_))) {
        error = "Shared UI warp native allocation failed"; return false;
      }
      *this = std::move(next); return true;
    }

    // Serial immediate-context submission only. CPU shape/resource checks grant no mask
    // authority; current SCT/SCP/state/receipt/RGB are validated by the initializer itself.
    // Keep all operands immutable until the matching draw, before any subsequent queue.
    bool queue(ID3D11DeviceContext *context, const program_t &program, const request_t &request, std::string &error) {
      issued_ = false; error.clear();
      if (!geometry_ || request.queued != program.queued || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context) ||
          !programs_.acquire(device_.Get(), {program.initialize, program.pin}) || !same_device(program.pixel) ||
          !constant(request.geometry_constants, 48u) || !constant(request.subtitle_constants, 32u) || !same_device(request.sampler) ||
          (request.queued && (!constant(request.queued_guard_constants, 48u) || !same_device(request.queued_guard_transaction)))) {
        error = "Shared UI warp program/context/draw operands unavailable"; return false;
      }
      ComPtr<ID3D11Predicate> predicate; BOOL predicate_value = FALSE;
      context->GetPredication(&predicate, &predicate_value);
      if (predicate) { error = "Shared UI warp requires an unpredicated draw"; return false; }
      for (auto *view : request.draw_srvs) {
        if (!same_device(view)) { error = "Shared UI warp draw resource missing or foreign"; return false; }
        ComPtr<ID3D11Resource> resource; view->GetResource(&resource);
        if (resource.Get() == field_.texture.Get() || resource.Get() == weights_.texture.Get() || resource.Get() == rows_.texture.Get()) {
          error = "Shared UI warp writable input alias"; return false;
        }
      }
      ComPtr<ID3D11Resource> source; ComPtr<ID3D11Texture2D> source_texture;
      request.draw_srvs[0]->GetResource(&source);
      D3D11_TEXTURE2D_DESC source_desc {}; D3D11_SHADER_RESOURCE_VIEW_DESC view_desc {};
      request.draw_srvs[0]->GetDesc(&view_desc);
      if (FAILED(source.As(&source_texture))) { error = "Shared UI warp source texture unavailable"; return false; }
      source_texture->GetDesc(&source_desc);
      if (source_desc.Width != shape_[0] || source_desc.Height != shape_[1] || source_desc.ArraySize != 1u ||
          source_desc.MipLevels != 1u || source_desc.SampleDesc.Count != 1u || view_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
          view_desc.Texture2D.MostDetailedMip != 0u || view_desc.Texture2D.MipLevels != 1u || serial_ == std::numeric_limits<std::uint64_t>::max()) {
        error = "Shared UI warp source view/serial unavailable"; return false;
      }
      saved_t saved(context); saved_t::unbind(context);
      const UINT clear[4] {}; context->ClearUnorderedAccessViewUint(rows_.uav.Get(), clear);
      ID3D11ShaderResourceView *srvs[15] {};
      std::copy(request.draw_srvs.begin(), request.draw_srvs.end(), srvs);
      if (request.queued) srvs[10u] = request.queued_guard_transaction;
      auto *geometry = request.geometry_constants, *subtitle = request.subtitle_constants;
      auto *pin_geometry = geometry_.Get(), *guard = request.queued ? request.queued_guard_constants : nullptr;
      auto *sampler = request.sampler;
      context->CSSetConstantBuffers(2u, 1u, &geometry); context->CSSetConstantBuffers(3u, 1u, &subtitle);
      context->CSSetConstantBuffers(5u, 1u, &guard); context->CSSetConstantBuffers(6u, 1u, &pin_geometry);
      context->CSSetSamplers(0u, 1u, &sampler); context->CSSetShaderResources(0u, 15u, srvs);
      ID3D11UnorderedAccessView *uavs[3] {field_.uav.Get(), weights_.uav.Get(), rows_.uav.Get()};
      context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      context->CSSetShader(program.initialize, nullptr, 0u);
      context->Dispatch((shape_[0] + 15u) / 16u, (shape_[3] + 15u) / 16u, 1u);
      ID3D11UnorderedAccessView *null_uavs[3] {}; context->CSSetUnorderedAccessViews(0u, 3u, null_uavs, nullptr);
      srvs[12u] = weights_.srv.Get(); context->CSSetShaderResources(0u, 15u, srvs);
      uavs[1u] = uavs[2u] = nullptr; context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      context->CSSetShader(program.pin, nullptr, 0u); context->Dispatch((shape_[3] + 7u) / 8u, 1u, 1u);
      for (std::size_t i = 0u; i < inputs_.size(); ++i) inputs_[i] = request.draw_srvs[i];
      draw_geometry_ = geometry; draw_subtitle_ = subtitle; sampler_ = sampler;
      guard_ = request.queued_guard_constants; transaction_ = request.queued_guard_transaction;
      context_ = context; queued_ = request.queued; ++serial_; issued_ = true; return true;
    }

    [[nodiscard]] bool matches(ID3D11DeviceContext *context, const request_t &request) const noexcept {
      if (!issued_ || context != context_ || request.queued != queued_ || request.geometry_constants != draw_geometry_.Get() ||
          request.subtitle_constants != draw_subtitle_.Get() || request.sampler != sampler_.Get() ||
          request.queued_guard_constants != guard_.Get() || request.queued_guard_transaction != transaction_.Get()) return false;
      for (std::size_t i = 0u; i < inputs_.size(); ++i) if (inputs_[i].Get() != request.draw_srvs[i]) return false;
      return true;
    }
    [[nodiscard]] views_t views() const { return issued_ ? views_t {weights_.srv, field_.srv, rows_.srv, geometry_, shape_, serial_} : views_t {}; }
  };

  class scoped_binding_t {
    ID3D11DeviceContext *context_ = nullptr;
    std::array<ComPtr<ID3D11ShaderResourceView>, 3u> prior_;
    ComPtr<ID3D11Buffer> geometry_;
    ComPtr<ID3D11ShaderResourceView> background_, background_proof_;
    ComPtr<ID3D11Buffer> background_constants_;

  public:
    scoped_binding_t() = default;
    scoped_binding_t(const scoped_binding_t &) = delete;
    scoped_binding_t &operator=(const scoped_binding_t &) = delete;

    bool bind(ID3D11DeviceContext *context, const views_t *views) {
      if (!context || context_) return false;
      ID3D11ShaderResourceView *prior[3] {}; context->PSGetShaderResources(12u, 3u, prior);
      for (std::size_t i = 0u; i < prior_.size(); ++i) prior_[i].Attach(prior[i]);
      context->PSGetConstantBuffers(6u, 1u, &geometry_); context_ = context;
      context->PSGetShaderResources(9u, 1u, &background_); context->PSGetShaderResources(11u, 1u, &background_proof_);
      context->PSGetConstantBuffers(4u, 1u, &background_constants_);
      ID3D11ShaderResourceView *null_view = nullptr; ID3D11Buffer *null_constants = nullptr;
      context->PSSetShaderResources(9u, 1u, &null_view); context->PSSetShaderResources(11u, 1u, &null_view);
      context->PSSetConstantBuffers(4u, 1u, &null_constants);
      ID3D11ShaderResourceView *current[3] {views ? views->weights.Get() : nullptr, views ? views->field.Get() : nullptr, views ? views->rows.Get() : nullptr};
      auto *geometry = views ? views->geometry.Get() : nullptr;
      context->PSSetShaderResources(12u, 3u, current); context->PSSetConstantBuffers(6u, 1u, &geometry); return true;
    }
    ~scoped_binding_t() {
      if (!context_) return;
      ID3D11ShaderResourceView *prior[3] {prior_[0].Get(), prior_[1].Get(), prior_[2].Get()}; auto *geometry = geometry_.Get();
      context_->PSSetShaderResources(12u, 3u, prior); context_->PSSetConstantBuffers(6u, 1u, &geometry);
      auto *background = background_.Get(), *proof = background_proof_.Get(); auto *constants = background_constants_.Get();
      context_->PSSetShaderResources(9u, 1u, &background); context_->PSSetShaderResources(11u, 1u, &proof);
      context_->PSSetConstantBuffers(4u, 1u, &constants);
    }
  };
} // namespace models::host_sbs_shared_ui_warp
#endif

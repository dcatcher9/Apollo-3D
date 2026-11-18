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

  inline constexpr std::array<cache::shader_spec, 3u> completed_specs {{
    {"host_sbs_shared_ui_warp.hlsl", "initialize_main", "cs_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "SunshineApplyUICS", "cs_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_main_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view completed_closure = "5fd9b51806ea68a9584b45b259b47e0012096c3c75db2391736d9d8c8f477df9";
  inline constexpr std::array<cache::shader_spec, 3u> queued_specs {{
    {"host_sbs_shared_ui_warp_queued.hlsl", "initialize_main", "cs_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "SunshineApplyUICS", "cs_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_main_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view queued_closure = "40efadc228b15d4107bbfb5c4f6e4267a2f0d05e6a845b63271e88687d2995f7";
  inline constexpr std::array<cache::shader_spec, 3u> completed_diagnostic_specs {{
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_mapping_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_coverage_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp.hlsl", "shared_pin_boundary_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view completed_diagnostic_closure = "67aa5f04f4a0878cdc7246128e642d0d9b54f569a045464db4771950c57e55a0";
  inline constexpr std::array<cache::shader_spec, 3u> queued_diagnostic_specs {{
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_mapping_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_coverage_ps", "ps_5_0"},
    {"host_sbs_shared_ui_warp_queued.hlsl", "shared_pin_boundary_ps", "ps_5_0"},
  }};
  inline constexpr std::string_view queued_diagnostic_closure = "204523d91460879e5c41ecad18ea9e6f06c14fd83211464947ab83d121162c51";

  inline constexpr std::string_view source_closure(bool queued) noexcept {
    return queued ? queued_closure : completed_closure;
  }

  inline constexpr std::string_view diagnostic_source_closure(bool queued) noexcept {
    return queued ? queued_diagnostic_closure : completed_diagnostic_closure;
  }

  inline constexpr std::string_view renderer_kind(bool queued) noexcept {
    return queued ? "shared-ui-pin-queued-v2" : "shared-ui-pin-completed-v2";
  }

  // The Game 3D text-game route. Its pin weight is the text coverage a game's own helper reports,
  // so it reads no subtitle matte, receipt or condition; everything after the weight is shared.
  inline constexpr std::array<cache::shader_spec, 5u> text_game_specs {{
    {"host_sbs_text_game_ui_warp.hlsl", "initialize_main", "cs_5_0"},
    {"host_sbs_text_game_ui_warp.hlsl", "SunshineApplyUICS", "cs_5_0"},
    {"host_sbs_text_game_ui_warp.hlsl", "shared_pin_main_ps", "ps_5_0"},
    {"host_sbs_text_game_ui_warp.hlsl", "text_game_verify_main", "cs_5_0"},
    {"host_sbs_text_game_ui_warp.hlsl", "text_game_columns_main", "cs_5_0"},
  }};
  inline constexpr std::string_view text_game_closure = "7747eb879047342ee2214e34a193c60abb4c935fd4ddcdb9e4271e32255261ca";

  struct program_t {
    ID3D11ComputeShader *initialize = nullptr, *pin = nullptr;
    ID3D11PixelShader *pixel = nullptr;
    bool queued = false;
    bool text_game = false;
    ID3D11ComputeShader *text_game_verify = nullptr, *text_game_columns = nullptr;
  };

  /** b7 of the text-game route. */
  struct text_game_constants_t {
    float left = 0.0f, top = 0.0f, inverse_width = 0.0f, inverse_height = 0.0f;  ///< The game's picture in source pixels.
    std::uint32_t enabled = 0u;
    float plane = 0.0f;  ///< UI plane as a full-source U displacement.
    std::uint32_t text_top = 0u, text_bottom = 0u;  ///< Source rows the coverage picture is current for.
    float front_cap = 0.0f;  ///< The scene's positive display bound, as a full-source U displacement.
    std::uint32_t reserved[3] {};
  };
  /// Verdict words: light glyph cores and those shown, dark ones and those shown, then
  /// {covered, conflicts at five levels} a tile.
  inline constexpr std::size_t text_game_verdict_words = 4u + 256u * 6u;
  static_assert(sizeof(text_game_constants_t) == 48u);

  // Preparation only creates device objects from one immutable authenticated source snapshot.
  // It never uses the immediate context and can run on the tracked pipeline-build worker.
  class programs_t {
    struct route_t {
      std::array<ComPtr<ID3D11ComputeShader>, 2u> compute;
      ComPtr<ID3D11PixelShader> pixel;
    };
    std::array<route_t, 2u> routes_;
    route_t text_game_;
    ComPtr<ID3D11ComputeShader> text_game_verify_, text_game_columns_;
    using diagnostics_t = std::array<ComPtr<ID3D11PixelShader>, 3u>;
    std::array<diagnostics_t, 2u> diagnostics_;
    std::array<std::string, 2u> route_errors_;

    static bool create_route(ID3D11Device *device, const cache::source_snapshot_t &source, bool queued, route_t &output, std::string &error) {
      const auto &specs = queued ? queued_specs : completed_specs;
      for (std::size_t stage = 0u; stage < specs.size(); ++stage) {
        const auto code = cache::get(source, specs[stage]);
        if (!code || code->empty() || (stage < 2u ?
            FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &output.compute[stage])) :
            FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &output.pixel)))) {
          error = queued ? "Queued shared UI warp shader creation failed" : "Completed shared UI warp shader creation failed";
          return false;
        }
      }
      return true;
    }

  public:
    bool load(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        const auto source = cache::snapshot_sources(root, completed_specs);
        if (!device || !source || cache::source_closure_sha256(source) != completed_closure) {
          error = route_errors_[0] = "Shared UI warp shader source mismatch";
          return false;
        }
        programs_t next;
        if (!create_route(device, source, false, next.routes_[0], error)) {
          route_errors_[0] = error;
          return false;
        }
        *this = std::move(next);
        return true;
      } catch (const std::exception &exception) {
        error = route_errors_[0] = exception.what();
        return false;
      }
    }

    // Explicit optional preparation. Keep the already authenticated completed objects
    // on any queued source/compilation/device failure, including a wrong-device request.
    bool load_queued(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        ComPtr<ID3D11Device> completed_device;
        if (routes_[0].pixel) routes_[0].pixel->GetDevice(&completed_device);
        if (!device || !ready(false) || completed_device.Get() != device) {
          error = route_errors_[1] = "Queued shared UI warp device does not match its completed programs";
          return false;
        }
        const auto source = cache::snapshot_sources(root, queued_specs);
        if (!source || cache::source_closure_sha256(source) != queued_closure) {
          error = route_errors_[1] = "Shared UI warp shader source mismatch";
          return false;
        }
        if (!ready(true)) {
          route_t next;
          if (!create_route(device, source, true, next, error)) {
            route_errors_[1] = error;
            return false;
          }
          routes_[1] = std::move(next);
        }
        route_errors_[1].clear();
        return true;
      } catch (const std::exception &exception) {
        error = route_errors_[1] = exception.what();
        return false;
      }
    }

    // Optional and independent: a failure here leaves the subtitle routes exactly as they were.
    bool load_text_game(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        const auto source = cache::snapshot_sources(root, text_game_specs);
        if (!device || !source || cache::source_closure_sha256(source) != text_game_closure) {
          error = "Text-game UI warp shader source mismatch";
          return false;
        }
        route_t next;
        for (std::size_t stage = 0u; stage < 3u; ++stage) {
          const auto code = cache::get(source, text_game_specs[stage]);
          if (!code || code->empty() || (stage < 2u ?
              FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &next.compute[stage])) :
              FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &next.pixel)))) {
            error = "Text-game UI warp shader creation failed";
            return false;
          }
        }
        ComPtr<ID3D11ComputeShader> verify, columns;
        const auto verify_code = cache::get(source, text_game_specs[3u]);
        const auto columns_code = cache::get(source, text_game_specs[4u]);
        if (!verify_code || verify_code->empty() || FAILED(device->CreateComputeShader(verify_code->data(), verify_code->size(), nullptr, &verify)) ||
            !columns_code || columns_code->empty() || FAILED(device->CreateComputeShader(columns_code->data(), columns_code->size(), nullptr, &columns))) {
          error = "Text-game UI warp shader creation failed";
          return false;
        }
        text_game_ = std::move(next);
        text_game_verify_ = std::move(verify);
        text_game_columns_ = std::move(columns);
        return true;
      } catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }
    [[nodiscard]] bool text_game_ready() const noexcept { return text_game_.compute[0] && text_game_.compute[1] && text_game_.pixel && text_game_verify_ && text_game_columns_; }
    [[nodiscard]] program_t text_game_program() const noexcept {
      return {text_game_.compute[0].Get(), text_game_.compute[1].Get(), text_game_.pixel.Get(), false, true, text_game_verify_.Get(), text_game_columns_.Get()};
    }

    bool load_diagnostics(ID3D11Device *device, const std::filesystem::path &root, std::string &error, bool queued = false) {
      error.clear();
      // The selected route owns both the production and diagnostic shaders on one device.
      const auto route = queued ? 1u : 0u;
      ComPtr<ID3D11Device> program_device;
      if (routes_[route].pixel) {
        routes_[route].pixel->GetDevice(&program_device);
      }
      if (!device || program_device.Get() != device) {
        error = "Shared UI warp diagnostic device does not match its production programs";
        return false;
      }
      auto &diagnostics = diagnostics_[route];
      if (std::all_of(diagnostics.begin(), diagnostics.end(), [](const auto &shader) {
            return bool(shader);
          })) {
        return true;
      }
      try {
        const auto &specs = queued ? queued_diagnostic_specs : completed_diagnostic_specs;
        const auto source = cache::snapshot_sources(root, specs);
        if (!source || cache::source_closure_sha256(source) != diagnostic_source_closure(queued)) {
          error = "Shared UI warp diagnostic source mismatch";
          return false;
        }
        diagnostics_t next;
        for (std::size_t i = 0u; i < next.size(); ++i) {
          const auto code = cache::get(source, specs[i]);
          if (!code || code->empty() || FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &next[i]))) {
            error = "Shared UI warp diagnostic shader creation failed";
            return false;
          }
        }
        diagnostics = std::move(next);
        return true;
      } catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }

    [[nodiscard]] bool ready(bool queued = false) const noexcept {
      const auto &route = routes_[queued ? 1u : 0u];
      return route.compute[0] && route.compute[1] && route.pixel;
    }
    [[nodiscard]] const std::string &route_error(bool queued = false) const noexcept { return route_errors_[queued ? 1u : 0u]; }
    [[nodiscard]] program_t program(bool queued = false) const noexcept {
      const auto route = queued ? 1u : 0u;
      const auto &shaders = routes_[route];
      return {shaders.compute[0].Get(), shaders.compute[1].Get(), shaders.pixel.Get(), queued};
    }

    [[nodiscard]] ID3D11PixelShader *mapping(bool queued = false) const noexcept {
      return diagnostics_[queued ? 1u : 0u][0u].Get();
    }

    [[nodiscard]] ID3D11PixelShader *coverage(bool queued = false) const noexcept {
      return diagnostics_[queued ? 1u : 0u][1u].Get();
    }

    [[nodiscard]] ID3D11PixelShader *boundary(bool queued = false) const noexcept {
      return diagnostics_[queued ? 1u : 0u][2u].Get();
    }
  };

  struct request_t {
    std::array<ID3D11ShaderResourceView *, 8u> draw_srvs {}; // literal t0..7, including the selected mask receipt
    ID3D11Buffer *geometry_constants = nullptr, *subtitle_constants = nullptr;
    ID3D11SamplerState *sampler = nullptr;
    ID3D11Buffer *queued_guard_constants = nullptr;
    ID3D11ShaderResourceView *queued_guard_transaction = nullptr;
    bool queued = false;
    // Text-game route: t3..7 stay unbound, and the weight comes from this coverage and rectangle.
    bool text_game = false;
    ID3D11ShaderResourceView *text_game_coverage = nullptr;
    ID3D11Buffer *text_game_constants = nullptr;
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
    // Text-game route only: {glyph-core pixels, those still showing their reference colour}.
    ComPtr<ID3D11Buffer> verdict_;
    ComPtr<ID3D11ShaderResourceView> verdict_srv_;
    ComPtr<ID3D11UnorderedAccessView> verdict_uav_;
    ComPtr<ID3D11Buffer> geometry_;
    std::array<UINT, 4u> shape_ {};
    host_sbs_compute_program_lease::group_t<2u> programs_;
    std::array<ComPtr<ID3D11ShaderResourceView>, 8u> inputs_;
    ComPtr<ID3D11Buffer> draw_geometry_, draw_subtitle_, guard_, text_game_constants_;
    ComPtr<ID3D11ShaderResourceView> transaction_, text_game_coverage_;
    ComPtr<ID3D11SamplerState> sampler_;
    ID3D11DeviceContext *context_ = nullptr;
    std::uint64_t serial_ = 0u;
    bool issued_ = false, queued_ = false, text_game_ = false;

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
      std::array<ComPtr<ID3D11UnorderedAccessView>, 4u> uavs;
      std::array<ComPtr<ID3D11Buffer>, 8u> constants;
      std::array<ComPtr<ID3D11ShaderResourceView>, 3u> pixel;
      ComPtr<ID3D11SamplerState> sampler;
      ComPtr<ID3D11ComputeShader> shader;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances;
      UINT count = 256u;

      explicit saved_t(ID3D11DeviceContext *c): context(c) {
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[4] {};
        ID3D11Buffer *b[8] {};
        ID3D11ClassInstance *ci[256] {};
        c->CSGetShaderResources(0u, 15u, s); c->CSGetUnorderedAccessViews(0u, 4u, u);
        c->CSGetConstantBuffers(0u, 8u, b); c->CSGetSamplers(0u, 1u, &sampler);
        c->CSGetShader(&shader, ci, &count); c->PSGetShaderResources(12u, 3u, p);
        for (std::size_t i = 0u; i < srvs.size(); ++i) srvs[i].Attach(s[i]);
        for (std::size_t i = 0u; i < uavs.size(); ++i) uavs[i].Attach(u[i]);
        for (std::size_t i = 0u; i < constants.size(); ++i) constants[i].Attach(b[i]);
        for (std::size_t i = 0u; i < pixel.size(); ++i) pixel[i].Attach(p[i]);
        for (UINT i = 0u; i < count; ++i) instances[i].Attach(ci[i]);
      }
      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[4] {};
        c->CSSetShaderResources(0u, 15u, s); c->CSSetUnorderedAccessViews(0u, 4u, u, nullptr);
        c->PSSetShaderResources(12u, 3u, p);
      }
      ~saved_t() {
        unbind(context);
        ID3D11ShaderResourceView *s[15] {}, *p[3] {};
        ID3D11UnorderedAccessView *u[4] {};
        ID3D11Buffer *b[8] {};
        ID3D11ClassInstance *ci[256] {};
        for (std::size_t i = 0u; i < srvs.size(); ++i) s[i] = srvs[i].Get();
        for (std::size_t i = 0u; i < uavs.size(); ++i) u[i] = uavs[i].Get();
        for (std::size_t i = 0u; i < constants.size(); ++i) b[i] = constants[i].Get();
        for (std::size_t i = 0u; i < pixel.size(); ++i) p[i] = pixel[i].Get();
        for (UINT i = 0u; i < count; ++i) ci[i] = instances[i].Get();
        context->CSSetShaderResources(0u, 15u, s); context->CSSetUnorderedAccessViews(0u, 4u, u, nullptr);
        context->CSSetConstantBuffers(0u, 8u, b);
        auto *old_sampler = sampler.Get(); context->CSSetSamplers(0u, 1u, &old_sampler);
        context->CSSetShader(shader.Get(), ci, count); context->PSSetShaderResources(12u, 3u, p);
      }
    };

    bool create_verdict() {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(text_game_verdict_words * 4u); desc.Usage = D3D11_USAGE_DEFAULT; desc.StructureByteStride = 4u;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      return SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &verdict_)) &&
             SUCCEEDED(device_->CreateShaderResourceView(verdict_.Get(), nullptr, &verdict_srv_)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(verdict_.Get(), nullptr, &verdict_uav_));
    }

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
          !plane(1u, shape[3] + 1u, DXGI_FORMAT_R32_UINT, next.rows_) || !next.create_verdict() ||
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
      if (!geometry_ || request.queued != program.queued || request.text_game != program.text_game || (request.text_game && request.queued) || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context) ||
          !programs_.acquire(device_.Get(), {program.initialize, program.pin}) || !same_device(program.pixel) ||
          !constant(request.geometry_constants, 48u) || !constant(request.subtitle_constants, 32u) || !same_device(request.sampler) ||
          (request.queued && (!constant(request.queued_guard_constants, 48u) || !same_device(request.queued_guard_transaction))) ||
          (request.text_game && (!constant(request.text_game_constants, 48u) || !same_device(request.text_game_coverage) || !program.text_game_verify || !program.text_game_columns))) {
        error = "Shared UI warp program/context/draw operands unavailable"; return false;
      }
      ComPtr<ID3D11Predicate> predicate; BOOL predicate_value = FALSE;
      context->GetPredication(&predicate, &predicate_value);
      if (predicate) { error = "Shared UI warp requires an unpredicated draw"; return false; }
      for (std::size_t slot = 0u; slot < request.draw_srvs.size(); ++slot) {
        auto *view = request.draw_srvs[slot];
        // The text-game route has no subtitle operands: its t3..7 must be absent, not optional.
        if (request.text_game && slot >= 3u) {
          if (view) { error = "Text-game UI warp takes no subtitle operand"; return false; }
          continue;
        }
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
      if (request.text_game) {
        srvs[11u] = request.text_game_coverage;
        auto *text_game = request.text_game_constants;
        context->CSSetConstantBuffers(7u, 1u, &text_game);
      }
      auto *geometry = request.geometry_constants, *subtitle = request.subtitle_constants;
      auto *pin_geometry = geometry_.Get(), *guard = request.queued ? request.queued_guard_constants : nullptr;
      auto *sampler = request.sampler;
      context->CSSetConstantBuffers(2u, 1u, &geometry); context->CSSetConstantBuffers(3u, 1u, &subtitle);
      context->CSSetConstantBuffers(5u, 1u, &guard); context->CSSetConstantBuffers(6u, 1u, &pin_geometry);
      context->CSSetSamplers(0u, 1u, &sampler);
      if (request.text_game) {
        // A fresh observation for this draw: whether the picture shows the text, and the scene
        // under it for the plane.
        context->ClearUnorderedAccessViewUint(verdict_uav_.Get(), clear);
        context->CSSetShaderResources(0u, 15u, srvs);
        auto *verdict = verdict_uav_.Get();
        context->CSSetUnorderedAccessViews(3u, 1u, &verdict, nullptr);
        context->CSSetShader(program.text_game_verify, nullptr, 0u);
        context->Dispatch((shape_[0] + 15u) / 16u, (shape_[3] + 15u) / 16u, 1u);
        verdict = nullptr;
        context->CSSetUnorderedAccessViews(3u, 1u, &verdict, nullptr);
        srvs[10u] = verdict_srv_.Get();
      }
      context->CSSetShaderResources(0u, 15u, srvs);
      ID3D11UnorderedAccessView *uavs[3] {field_.uav.Get(), weights_.uav.Get(), rows_.uav.Get()};
      context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      if (request.text_game) {
        // Rows to the nearest glyph of each column, left in the field plane for the initializer.
        context->CSSetShader(program.text_game_columns, nullptr, 0u);
        context->Dispatch((shape_[0] + 63u) / 64u, 1u, 1u);
      }
      context->CSSetShader(program.initialize, nullptr, 0u);
      context->Dispatch((shape_[0] + 15u) / 16u, (shape_[3] + 15u) / 16u, 1u);
      ID3D11UnorderedAccessView *null_uavs[3] {}; context->CSSetUnorderedAccessViews(0u, 3u, null_uavs, nullptr);
      srvs[12u] = weights_.srv.Get(); context->CSSetShaderResources(0u, 15u, srvs);
      uavs[1u] = uavs[2u] = nullptr; context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      context->CSSetShader(program.pin, nullptr, 0u); context->Dispatch((shape_[3] + 7u) / 8u, 1u, 1u);
      for (std::size_t i = 0u; i < inputs_.size(); ++i) inputs_[i] = request.draw_srvs[i];
      draw_geometry_ = geometry; draw_subtitle_ = subtitle; sampler_ = sampler;
      guard_ = request.queued_guard_constants; transaction_ = request.queued_guard_transaction;
      text_game_constants_ = request.text_game_constants; text_game_coverage_ = request.text_game_coverage; text_game_ = request.text_game;
      context_ = context; queued_ = request.queued; ++serial_; issued_ = true; return true;
    }

    [[nodiscard]] bool matches(ID3D11DeviceContext *context, const request_t &request) const noexcept {
      if (!issued_ || context != context_ || request.queued != queued_ || request.geometry_constants != draw_geometry_.Get() ||
          request.subtitle_constants != draw_subtitle_.Get() || request.sampler != sampler_.Get() ||
          request.queued_guard_constants != guard_.Get() || request.queued_guard_transaction != transaction_.Get() ||
          request.text_game != text_game_ || request.text_game_constants != text_game_constants_.Get() ||
          request.text_game_coverage != text_game_coverage_.Get()) return false;
      for (std::size_t i = 0u; i < inputs_.size(); ++i) if (inputs_[i].Get() != request.draw_srvs[i]) return false;
      return true;
    }
    /** Text-game route: the verdict of the last queued draw, for diagnostics only. */
    [[nodiscard]] ID3D11Buffer *text_game_verdict() const noexcept { return verdict_.Get(); }
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

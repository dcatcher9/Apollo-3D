/** @file Shared Game 3D pinning and one-source-sample Host subtitle renderer. */
#pragma once

#include "host_sbs_compute_program_lease.h"
#include "host_sbs_eye_field.h"
#include "host_sbs_shader_cache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <d3d11.h>
#include <wrl/client.h>

namespace models::host_sbs_shared_ui_warp {
  namespace cache = host_sbs_shader_cache;
  using Microsoft::WRL::ComPtr;

  // The completed, diagnostic and text-game route programs are the `shared_ui_warp_completed`,
  // `shared_ui_warp_completed_diagnostic` and `text_game_ui_warp` registry closure groups.
  // The text-game route's pin weight is the text coverage a game's own helper reports, so it
  // reads no subtitle matte, receipt or condition; everything after the weight is shared.
  inline constexpr std::string_view renderer_kind = "shared-ui-pin-completed-v2";

  struct program_t {
    ID3D11ComputeShader *initialize = nullptr, *pin = nullptr, *prepare = nullptr;
    ID3D11PixelShader *pixel = nullptr;
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
    float sdr_white = 0.0f;  ///< SDR white of an HDR capture in scRGB units; unused otherwise.
    std::uint32_t reserved[2] {};
  };
  /// Verdict words: light glyph cores and those shown, dark ones and those shown, then
  /// {covered, conflicts at five levels} a tile.
  inline constexpr std::size_t text_game_verdict_words = 4u + 256u * 6u;
  // The shader's TextGameRowWords and TextGameRowLimit: one record for each row of the band.
  inline constexpr std::size_t text_game_row_words = 6u, text_game_row_limit = 4096u;
  static_assert(sizeof(text_game_constants_t) == 48u);

  // Preparation only creates device objects from one immutable authenticated source snapshot.
  // It never uses the immediate context and can run on the tracked pipeline-build worker.
  class programs_t {
    struct route_t {
      std::array<ComPtr<ID3D11ComputeShader>, 3u> compute;
      ComPtr<ID3D11PixelShader> pixel;
    };
    route_t route_;
    ComPtr<ID3D11ComputeShader> direct_glyph_prepare_;
    std::string_view model_prepare_closure_;
    route_t text_game_;
    ComPtr<ID3D11ComputeShader> text_game_verify_, text_game_columns_;
    std::array<ComPtr<ID3D11PixelShader>, 3u> diagnostics_;
    std::string route_error_;

    static bool create_route(ID3D11Device *device, const cache::source_snapshot_t &source, route_t &output, std::string &error) {
      for (std::size_t stage = 0u; stage < cache::shared_ui_warp_completed_specs.size(); ++stage) {
        const auto code = cache::get(source, cache::shared_ui_warp_completed_specs[stage]);
        if (!code || code->empty() || (stage < output.compute.size() ?
            FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &output.compute[stage])) :
            FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &output.pixel)))) {
          error = "Completed shared UI warp shader creation failed";
          return false;
        }
      }
      return true;
    }

  public:
    bool load(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        const auto source = cache::snapshot_sources(root, cache::shared_ui_warp_completed_specs);
        if (!device || !source || cache::source_closure_sha256(source) != cache::shared_ui_warp_completed_source_closure_sha256) {
          error = route_error_ = "Shared UI warp shader source mismatch";
          return false;
        }
        programs_t next;
        if (!create_route(device, source, next.route_, error)) {
          route_error_ = error;
          return false;
        }
        *this = std::move(next);
        return true;
      } catch (const std::exception &exception) {
        error = route_error_ = exception.what();
        return false;
      }
    }

    // The Direct Glyph route replaces only the prepare stage. Initialize, pin and draw
    // still use the already authenticated completed programs on the same device.
    bool load_direct_glyph_completed(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        ComPtr<ID3D11Device> completed_device;
        if (route_.pixel) route_.pixel->GetDevice(&completed_device);
        if (!device || !ready() || completed_device.Get() != device) {
          error = "Direct glyph shared-warp device does not match its completed programs";
          return false;
        }
        const auto &specs = cache::direct_glyph_prepare_specs;
        const auto closure = cache::direct_glyph_prepare_source_closure_sha256;
        const auto source = cache::snapshot_sources(root, specs);
        if (!source || cache::source_closure_sha256(source) != closure) {
          error = "Direct glyph shared-warp prepare source mismatch";
          return false;
        }
        const auto code = cache::get(source, specs[0]);
        ComPtr<ID3D11ComputeShader> next;
        if (!code || code->empty() || FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &next))) {
          error = "Direct glyph shared-warp prepare creation failed";
          return false;
        }
        direct_glyph_prepare_ = std::move(next);
        model_prepare_closure_ = closure;
        return true;
      } catch (const std::exception &exception) {
        error = exception.what();
        return false;
      }
    }

    // Optional and independent: a failure here leaves the subtitle routes exactly as they were.
    bool load_text_game(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      try {
        const auto source = cache::snapshot_sources(root, cache::text_game_ui_warp_specs);
        if (!device || !source || cache::source_closure_sha256(source) != cache::text_game_ui_warp_source_closure_sha256) {
          error = "Text-game UI warp shader source mismatch";
          return false;
        }
        route_t next;
        for (std::size_t stage = 0u; stage < 3u; ++stage) {
          const auto code = cache::get(source, cache::text_game_ui_warp_specs[stage]);
          if (!code || code->empty() || (stage < 2u ?
              FAILED(device->CreateComputeShader(code->data(), code->size(), nullptr, &next.compute[stage])) :
              FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &next.pixel)))) {
            error = "Text-game UI warp shader creation failed";
            return false;
          }
        }
        ComPtr<ID3D11ComputeShader> verify, columns;
        const auto verify_code = cache::get(source, cache::text_game_ui_warp_specs[3u]);
        const auto columns_code = cache::get(source, cache::text_game_ui_warp_specs[4u]);
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
      return {text_game_.compute[0].Get(), text_game_.compute[1].Get(), nullptr, text_game_.pixel.Get(), true, text_game_verify_.Get(), text_game_columns_.Get()};
    }

    bool load_diagnostics(ID3D11Device *device, const std::filesystem::path &root, std::string &error) {
      error.clear();
      // The completed route owns both the production and diagnostic shaders on one device.
      ComPtr<ID3D11Device> program_device;
      if (route_.pixel) {
        route_.pixel->GetDevice(&program_device);
      }
      if (!device || program_device.Get() != device) {
        error = "Shared UI warp diagnostic device does not match its production programs";
        return false;
      }
      if (std::all_of(diagnostics_.begin(), diagnostics_.end(), [](const auto &shader) {
            return bool(shader);
          })) {
        return true;
      }
      try {
        const auto source = cache::snapshot_sources(root, cache::shared_ui_warp_completed_diagnostic_specs);
        if (!source || cache::source_closure_sha256(source) != cache::shared_ui_warp_completed_diagnostic_source_closure_sha256) {
          error = "Shared UI warp diagnostic source mismatch";
          return false;
        }
        std::array<ComPtr<ID3D11PixelShader>, 3u> next;
        for (std::size_t i = 0u; i < next.size(); ++i) {
          const auto code = cache::get(source, cache::shared_ui_warp_completed_diagnostic_specs[i]);
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

    [[nodiscard]] bool ready() const noexcept {
      return route_.compute[0] && route_.compute[1] && route_.compute[2] && route_.pixel;
    }
    [[nodiscard]] const std::string &route_error() const noexcept { return route_error_; }
    [[nodiscard]] program_t program() const noexcept {
      return {route_.compute[0].Get(), route_.compute[1].Get(), route_.compute[2].Get(), route_.pixel.Get()};
    }

    [[nodiscard]] bool direct_glyph_completed_ready() const noexcept { return ready() && direct_glyph_prepare_; }
    [[nodiscard]] std::string_view model_prepare_closure() const noexcept { return model_prepare_closure_; }
    [[nodiscard]] program_t direct_glyph_completed_program() const noexcept {
      auto result = program();
      result.prepare = direct_glyph_prepare_.Get();
      return result;
    }

    [[nodiscard]] ID3D11PixelShader *mapping() const noexcept {
      return diagnostics_[0u].Get();
    }

    [[nodiscard]] ID3D11PixelShader *coverage() const noexcept {
      return diagnostics_[1u].Get();
    }

    [[nodiscard]] ID3D11PixelShader *boundary() const noexcept {
      return diagnostics_[2u].Get();
    }
  };

  struct request_t {
    std::array<ID3D11ShaderResourceView *, 8u> draw_srvs {}; // literal t0..7, including the selected mask receipt
    ID3D11Buffer *geometry_constants = nullptr, *subtitle_constants = nullptr;
    ID3D11SamplerState *sampler = nullptr;
    // Text-game route: t3..7 stay unbound, and the weight comes from this coverage and rectangle.
    bool text_game = false;
    ID3D11ShaderResourceView *text_game_coverage = nullptr;
    ID3D11Buffer *text_game_constants = nullptr;
    // Text-game route: only observe whether the picture shows the text and what lies under it.
    // Nothing is issued for a draw; the verdict is the whole result.
    bool text_game_observe_only = false;
    // Consumers of the raw field or weights require dense output on inactive draws.
    bool materialize_inactive = true;
  };

  struct views_t {
    ComPtr<ID3D11ShaderResourceView> weights, field, rows;
    ComPtr<ID3D11Buffer> geometry;
    std::array<UINT, 4u> shape {}; // full source W/H, native full-source band top/height
    std::uint64_t serial = 0u;
  };

  struct subtitle_domain_t {
    UINT width, height, top;
  };

  class resources_t {
    struct plane_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };
    ComPtr<ID3D11Device> device_;
    plane_t field_, weights_, rows_;
    // Text-game route only: the observation of the last queued draw (text_game_verdict_words).
    ComPtr<ID3D11Buffer> verdict_;
    ComPtr<ID3D11ShaderResourceView> verdict_srv_;
    ComPtr<ID3D11UnorderedAccessView> verdict_uav_;
    // Text-game route only: the same observation row by row, from which each line is judged.
    ComPtr<ID3D11Buffer> line_rows_;
    ComPtr<ID3D11UnorderedAccessView> line_rows_uav_;
    ComPtr<ID3D11Buffer> preparation_;
    ComPtr<ID3D11ShaderResourceView> preparation_srv_;
    ComPtr<ID3D11UnorderedAccessView> preparation_uav_;
    ComPtr<ID3D11Buffer> geometry_;
    std::array<UINT, 4u> shape_ {};
    host_sbs_compute_program_lease::group_t<2u> programs_;
    host_sbs_compute_program_lease::group_t<1u> prepare_program_;
    std::array<ComPtr<ID3D11ShaderResourceView>, 8u> inputs_;
    ComPtr<ID3D11Buffer> draw_geometry_, draw_subtitle_, text_game_constants_;
    ComPtr<ID3D11ShaderResourceView> text_game_coverage_;
    ComPtr<ID3D11SamplerState> sampler_;
    ID3D11DeviceContext *context_ = nullptr;
    std::uint64_t serial_ = 0u;
    bool issued_ = false, text_game_ = false, materialize_inactive_ = true;

    struct subtitle_inputs_t {
      ComPtr<ID3D11Resource> source, matte;
      D3D11_TEXTURE2D_DESC source_desc {};
    };

    static bool describe_texture(const ComPtr<ID3D11Resource> &resource, D3D11_TEXTURE2D_DESC &desc) {
      ComPtr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource.As(&texture))) return false;
      texture->GetDesc(&desc);
      return true;
    }

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

    static constexpr UINT compute_srv_count = host_sbs_eye_field::pixel_slot + 1u;

    static void unbind_compute_resources(ID3D11DeviceContext *context) {
      ID3D11ShaderResourceView *srvs[compute_srv_count] {};
      ID3D11UnorderedAccessView *uavs[5] {};
      context->CSSetShaderResources(0u, compute_srv_count, srvs);
      context->CSSetUnorderedAccessViews(0u, 5u, uavs, nullptr);
    }

    // The pin planes are reserved PS12..14 outputs. Preparation unbinds them before writes.
    static void unbind_preparation_resources(ID3D11DeviceContext *context) {
      unbind_compute_resources(context);
      ID3D11ShaderResourceView *pixel[3] {};
      context->PSSetShaderResources(12u, 3u, pixel);
    }

    // The converter owns its compute bindings: preparation leaves every slot it touched clear
    // for the converter's next pass, and the draw command binds the planes itself.
    static void clear_preparation_bindings(ID3D11DeviceContext *context, const bool text_game) {
      unbind_compute_resources(context);
      ID3D11Buffer *constants[3] {};
      context->CSSetConstantBuffers(2u, 2u, constants);
      context->CSSetConstantBuffers(5u, text_game ? 3u : 2u, constants);
      ID3D11SamplerState *sampler = nullptr;
      context->CSSetSamplers(0u, 1u, &sampler);
      context->CSSetShader(nullptr, nullptr, 0u);
    }

    bool create_verdict() {
      if (verdict_uav_ && line_rows_uav_) return true;
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(text_game_verdict_words * 4u); desc.Usage = D3D11_USAGE_DEFAULT; desc.StructureByteStride = 4u;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      if (!verdict_uav_ &&
          (FAILED(device_->CreateBuffer(&desc, nullptr, verdict_.ReleaseAndGetAddressOf())) ||
           FAILED(device_->CreateShaderResourceView(verdict_.Get(), nullptr, verdict_srv_.ReleaseAndGetAddressOf())) ||
           FAILED(device_->CreateUnorderedAccessView(verdict_.Get(), nullptr, verdict_uav_.ReleaseAndGetAddressOf())))) {
        verdict_uav_.Reset();
        return false;
      }
      desc.ByteWidth = static_cast<UINT>(text_game_row_words * text_game_row_limit * 4u);
      desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      return SUCCEEDED(device_->CreateBuffer(&desc, nullptr, line_rows_.ReleaseAndGetAddressOf())) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(line_rows_.Get(), nullptr, line_rows_uav_.ReleaseAndGetAddressOf()));
    }

    bool create_preparation() {
      if (preparation_uav_) return true;
      // Two padded DispatchIndirect grids followed by the per-draw verdict.
      constexpr UINT words = 12u;
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = words * sizeof(UINT);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv {};
      srv.Format = DXGI_FORMAT_R32_TYPELESS;
      srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      srv.BufferEx.NumElements = words;
      srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
      uav.Format = DXGI_FORMAT_R32_TYPELESS;
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = words;
      uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      return SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &preparation_)) &&
             SUCCEEDED(device_->CreateShaderResourceView(preparation_.Get(), &srv, &preparation_srv_)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(preparation_.Get(), &uav, &preparation_uav_));
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
          shape[3] > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2u) {
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
      // Field and weights hold the band once per eye: the left eye's rows, then the right eye's.
      if (!plane(shape[0], shape[3] * 2u, DXGI_FORMAT_R32_FLOAT, next.field_) ||
          !plane(shape[0], shape[3] * 2u, DXGI_FORMAT_R32_FLOAT, next.weights_) ||
          !plane(1u, shape[3] + 1u, DXGI_FORMAT_R32_UINT, next.rows_) ||
          FAILED(device->CreateBuffer(&desc, &initial, &next.geometry_))) {
        error = "Shared UI warp native allocation failed"; return false;
      }
      *this = std::move(next); return true;
    }

    // The supplied domain is the one uploaded to the subtitle constants for this draw.
    // Inspect the literal color and matte once to place the full-width pin band.
    bool prepare_subtitle(ID3D11Device *device, ID3D11DeviceContext *context, const program_t &program,
                          const request_t &request, const subtitle_domain_t domain, std::string &error) {
      issued_ = false;
      if (request.text_game || !request.draw_srvs[0] || !request.draw_srvs[3]) {
        error = "Shared UI warp source or subtitle matte unavailable";
        return false;
      }
      subtitle_inputs_t inputs;
      request.draw_srvs[0]->GetResource(&inputs.source);
      request.draw_srvs[3]->GetResource(&inputs.matte);
      D3D11_TEXTURE2D_DESC matte_desc {};
      if (!describe_texture(inputs.source, inputs.source_desc) || !describe_texture(inputs.matte, matte_desc)) {
        error = "Shared UI warp source or subtitle matte texture unavailable";
        return false;
      }
      // A plane of this owner is never an input. Refuse it before its size can place a new band:
      // the field plane is twice the band, so as a source it would describe another shape.
      for (ID3D11Resource *input : {inputs.source.Get(), inputs.matte.Get()}) {
        if (input == field_.texture.Get() || input == weights_.texture.Get() || input == rows_.texture.Get()) {
          error = "Shared UI warp writable input alias";
          return false;
        }
      }
      const auto &source = inputs.source_desc;
      if (matte_desc.Width != domain.width || matte_desc.Height == 0u || matte_desc.Height > domain.height ||
          domain.top > source.Height || domain.height > source.Height - domain.top) {
        error = "Shared UI warp subtitle band outside source domain";
        return false;
      }
      const UINT band_end = domain.top + domain.height;
      const UINT band_top = band_end - matte_desc.Height;
      const UINT padded_top = band_top > 0u ? band_top - 1u : 0u;
      const std::array<UINT, 4u> shape {
        source.Width, source.Height, padded_top, std::min(source.Height, band_end + 1u) - padded_top,
      };
      return ensure(device, shape, error) && queue_draw(context, program, request, error, &inputs);
    }

    bool queue(ID3D11DeviceContext *context, const program_t &program, const request_t &request, std::string &error) {
      return queue_draw(context, program, request, error, nullptr);
    }

  private:
    // Serial immediate-context submission only. CPU shape/resource checks grant no mask
    // authority; the GPU validates the current SCT/SCP/state/receipt/RGB for this draw.
    // Keep all operands immutable until the matching draw, before any subsequent queue.
    // Preparation leaves its compute bindings clear; the draw command binds the planes.
    bool queue_draw(ID3D11DeviceContext *context, const program_t &program, const request_t &request, std::string &error,
                    const subtitle_inputs_t *subtitle_inputs) {
      issued_ = false; error.clear();
      if (!geometry_ || request.text_game != program.text_game || !context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || !same_device(context) ||
          !programs_.acquire(device_.Get(), {program.initialize, program.pin}) || !same_device(program.pixel) ||
          (!request.text_game && !prepare_program_.acquire(device_.Get(), {program.prepare})) ||
          !constant(request.geometry_constants, 48u) || !constant(request.subtitle_constants, 32u) || !same_device(request.sampler) ||
          (request.text_game && (!constant(request.text_game_constants, 48u) || !same_device(request.text_game_coverage) || !program.text_game_verify || !program.text_game_columns))) {
        error = "Shared UI warp program/context/draw operands unavailable"; return false;
      }
      ComPtr<ID3D11Predicate> predicate; BOOL predicate_value = FALSE;
      context->GetPredication(&predicate, &predicate_value);
      if (predicate) { error = "Shared UI warp requires an unpredicated draw"; return false; }
      ComPtr<ID3D11Resource> standalone_source;
      for (std::size_t slot = 0u; slot < request.draw_srvs.size(); ++slot) {
        auto *view = request.draw_srvs[slot];
        // The text-game route has no subtitle operands: its t3..7 must be absent, not optional.
        if (request.text_game && slot >= 3u) {
          if (view) { error = "Text-game UI warp takes no subtitle operand"; return false; }
          continue;
        }
        if (!same_device(view)) { error = "Shared UI warp draw resource missing or foreign"; return false; }
        auto *identity = subtitle_inputs && slot == 0u ? subtitle_inputs->source.Get() :
                         subtitle_inputs && slot == 3u ? subtitle_inputs->matte.Get() : nullptr;
        ComPtr<ID3D11Resource> resource;
        if (!identity) {
          view->GetResource(&resource);
          identity = resource.Get();
        }
        if (identity == field_.texture.Get() || identity == weights_.texture.Get() || identity == rows_.texture.Get()) {
          error = "Shared UI warp writable input alias"; return false;
        }
        if (!subtitle_inputs && slot == 0u) standalone_source = std::move(resource);
      }
      D3D11_TEXTURE2D_DESC standalone_source_desc {};
      if (!subtitle_inputs && !describe_texture(standalone_source, standalone_source_desc)) {
        error = "Shared UI warp source texture unavailable"; return false;
      }
      const auto &source_desc = subtitle_inputs ? subtitle_inputs->source_desc : standalone_source_desc;
      D3D11_SHADER_RESOURCE_VIEW_DESC view_desc {};
      request.draw_srvs[0]->GetDesc(&view_desc);
      if (source_desc.Width != shape_[0] || source_desc.Height != shape_[1] || source_desc.ArraySize != 1u ||
          source_desc.MipLevels != 1u || source_desc.SampleDesc.Count != 1u || view_desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D ||
          view_desc.Texture2D.MostDetailedMip != 0u || view_desc.Texture2D.MipLevels != 1u || serial_ == std::numeric_limits<std::uint64_t>::max()) {
        error = "Shared UI warp source view/serial unavailable"; return false;
      }
      if (request.text_game ? !create_verdict() : !create_preparation()) {
        error = "Shared UI warp preparation allocation failed"; return false;
      }
      unbind_preparation_resources(context);
      const UINT clear[4] {}; context->ClearUnorderedAccessViewUint(rows_.uav.Get(), clear);
      ID3D11ShaderResourceView *srvs[compute_srv_count] {};
      std::copy(request.draw_srvs.begin(), request.draw_srvs.end(), srvs);
      if (request.text_game) srvs[11u] = request.text_game_coverage;
      auto *geometry = request.geometry_constants, *subtitle = request.subtitle_constants;
      auto *pin_geometry = geometry_.Get();
      auto *sampler = request.sampler;
      ID3D11Buffer *draw_constants[]{geometry, subtitle};
      // b5 is unused by every route; it stays clear while b6/b7 are bound.
      ID3D11Buffer *pin_constants[]{nullptr, pin_geometry, request.text_game_constants};
      context->CSSetConstantBuffers(2u, 2u, draw_constants);
      context->CSSetConstantBuffers(5u, request.text_game ? 3u : 2u, pin_constants);
      context->CSSetSamplers(0u, 1u, &sampler);
      // Each eye's band starts from that eye's scene field, when the producer attached one to Base.
      const auto eye_field = host_sbs_eye_field::companion(request.draw_srvs[1]);
      srvs[host_sbs_eye_field::pixel_slot] = eye_field.Get();
      context->CSSetShaderResources(0u, compute_srv_count, srvs);
      ID3D11UnorderedAccessView *null_view = nullptr;
      if (request.text_game) {
        // A fresh observation for this draw: whether the picture shows the text, and the scene
        // under it for the plane.
        context->ClearUnorderedAccessViewUint(verdict_uav_.Get(), clear);
        context->ClearUnorderedAccessViewUint(line_rows_uav_.Get(), clear);
        // The rows' record stays bound for the column pass, which judges each line from it, and
        // for the initializer, which reads the rows' answers.
        ID3D11UnorderedAccessView *observation[2] {verdict_uav_.Get(), line_rows_uav_.Get()};
        context->CSSetUnorderedAccessViews(3u, 2u, observation, nullptr);
        context->CSSetShader(program.text_game_verify, nullptr, 0u);
        context->Dispatch((shape_[0] + 15u) / 16u, (shape_[3] + 15u) / 16u, 1u);
        context->CSSetUnorderedAccessViews(3u, 1u, &null_view, nullptr);
        if (request.text_game_observe_only) {
          clear_preparation_bindings(context, true);
          return true;
        }
        auto *verdict_input = verdict_srv_.Get();
        context->CSSetShaderResources(10u, 1u, &verdict_input);
      } else {
        ID3D11UnorderedAccessView *outputs[2] {rows_.uav.Get(), preparation_uav_.Get()};
        context->CSSetUnorderedAccessViews(2u, 2u, outputs, nullptr);
        context->CSSetShader(program.prepare, nullptr, 0u);
        context->Dispatch(1u, 1u, 1u);
        context->CSSetUnorderedAccessViews(3u, 1u, &null_view, nullptr);
        auto *preparation = preparation_srv_.Get();
        context->CSSetShaderResources(8u, 1u, &preparation);
      }
      ID3D11UnorderedAccessView *uavs[3] {field_.uav.Get(), weights_.uav.Get(), rows_.uav.Get()};
      context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      if (request.text_game) {
        // Rows to the nearest glyph of each column, left in the field plane for the initializer.
        context->CSSetShader(program.text_game_columns, nullptr, 0u);
        context->Dispatch((shape_[0] + 63u) / 64u, 1u, 1u);
      }
      const bool dense = request.text_game || request.materialize_inactive;
      context->CSSetShader(program.initialize, nullptr, 0u);
      if (dense) {
        const auto initialize_rows = request.text_game ? 2u * shape_[3] : shape_[3];
        context->Dispatch((shape_[0] + 15u) / 16u, (initialize_rows + 15u) / 16u, 1u);
      } else {
        context->DispatchIndirect(preparation_.Get(), 0u);
      }
      ID3D11UnorderedAccessView *null_uavs[3] {}; context->CSSetUnorderedAccessViews(0u, 3u, null_uavs, nullptr);
      auto *weights = weights_.srv.Get(); context->CSSetShaderResources(12u, 1u, &weights);
      if (!request.text_game) {
        auto *rows = rows_.srv.Get(); context->CSSetShaderResources(14u, 1u, &rows);
      }
      uavs[1u] = uavs[2u] = nullptr; context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
      context->CSSetShader(program.pin, nullptr, 0u);
      if (dense) {
        const auto stacked_rows = shape_[3] * 2u;
        context->Dispatch(request.text_game ? (stacked_rows + 7u) / 8u : stacked_rows, 1u, 1u);
      } else {
        context->DispatchIndirect(preparation_.Get(), 16u);
      }
      for (std::size_t i = 0u; i < inputs_.size(); ++i) inputs_[i] = request.draw_srvs[i];
      draw_geometry_ = geometry; draw_subtitle_ = subtitle; sampler_ = sampler;
      text_game_constants_ = request.text_game_constants; text_game_coverage_ = request.text_game_coverage; text_game_ = request.text_game;
      materialize_inactive_ = request.materialize_inactive;
      context_ = context; ++serial_; issued_ = true;
      clear_preparation_bindings(context, request.text_game);
      return true;
    }

  public:
    [[nodiscard]] bool matches(ID3D11DeviceContext *context, const request_t &request) const noexcept {
      if (!issued_ || context != context_ || request.geometry_constants != draw_geometry_.Get() ||
          request.subtitle_constants != draw_subtitle_.Get() || request.sampler != sampler_.Get() ||
          request.materialize_inactive != materialize_inactive_ ||
          request.text_game != text_game_ || request.text_game_constants != text_game_constants_.Get() ||
          request.text_game_coverage != text_game_coverage_.Get()) return false;
      for (std::size_t i = 0u; i < inputs_.size(); ++i) if (inputs_[i].Get() != request.draw_srvs[i]) return false;
      return true;
    }
    /** Text-game route: the observation of the last queued draw, for a late CPU readback. */
    [[nodiscard]] ID3D11Buffer *text_game_verdict() const noexcept { return verdict_.Get(); }
    [[nodiscard]] views_t views() const { return issued_ ? views_t {weights_.srv, field_.srv, rows_.srv, geometry_, shape_, serial_} : views_t {}; }
  };

} // namespace models::host_sbs_shared_ui_warp

#pragma once

// Private research orchestration only. Compiled shaders and source provenance belong to the caller.
// No drawing authority, CPU palette verdict, readback, frame flip, or production configuration.
#ifdef _WIN32

#include "depth_coordinate_v2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <d3d11.h>
#include <wrl/client.h>

namespace models::host_sbs_subtitle_inner_colour_gpu {
  using Microsoft::WRL::ComPtr;

  // AddRef preserves storage, not contents. These are MUTABLE working views and are not an
  // immutable source/frame publication. Copy on the D3D owner before the next dispatch/ensure.
  struct mutable_views_t {
    UINT width = 0u;
    UINT crop_height = 0u;
    ComPtr<ID3D11ShaderResourceView> body;
    ComPtr<ID3D11ShaderResourceView> matte;
    ComPtr<ID3D11ShaderResourceView> palette;
    ComPtr<ID3D11ShaderResourceView> mode_args;
    ComPtr<ID3D11ShaderResourceView> parent;
    ComPtr<ID3D11ShaderResourceView> root_flags;
    ComPtr<ID3D11ShaderResourceView> eligible;
    ComPtr<ID3D11ShaderResourceView> seeds;
    ComPtr<ID3D11ShaderResourceView> domain_tag;
    ComPtr<ID3D11ShaderResourceView> cover_clip;
    ComPtr<ID3D11ShaderResourceView> mode_body;
    ComPtr<ID3D11ShaderResourceView> mode_final;
    ComPtr<ID3D11Buffer> current_mode_args;
    ComPtr<ID3D11Buffer> native_clear_args;
  };

  class resources_t {
    struct texture_t {
      ComPtr<ID3D11Texture2D> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };
    struct buffer_t {
      ComPtr<ID3D11Buffer> resource;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    // Only the touched CS slots are captured. Hidden append/consume counters are preserved by
    // nullptr initial-count arguments. Existing shaders use no interfaces, but preserve them too.
    class saved_cs_t {
    public:
      explicit saved_cs_t(ID3D11DeviceContext *context): context_(context) {
        std::array<ID3D11ShaderResourceView *, 8u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        std::array<ID3D11Buffer *, 2u> constants {};
        context_->CSGetShaderResources(0u, static_cast<UINT>(srvs.size()), srvs.data());
        context_->CSGetUnorderedAccessViews(0u, static_cast<UINT>(uavs.size()), uavs.data());
        context_->CSGetConstantBuffers(0u, static_cast<UINT>(constants.size()), constants.data());
        for (std::size_t slot = 0u; slot < srvs.size(); ++slot) srvs_[slot].Attach(srvs[slot]);
        for (std::size_t slot = 0u; slot < uavs.size(); ++slot) uavs_[slot].Attach(uavs[slot]);
        for (std::size_t slot = 0u; slot < constants.size(); ++slot) constants_[slot].Attach(constants[slot]);
        ID3D11ComputeShader *shader = nullptr;
        std::array<ID3D11ClassInstance *, 256u> instances {};
        instance_count_ = static_cast<UINT>(instances.size());
        context_->CSGetShader(&shader, instances.data(), &instance_count_);
        shader_.Attach(shader);
        for (UINT index = 0u; index < instance_count_; ++index) instances_[index].Attach(instances[index]);
      }
      saved_cs_t(const saved_cs_t &) = delete;
      saved_cs_t &operator=(const saved_cs_t &) = delete;
      ~saved_cs_t() {
        unbind(context_);
        std::array<ID3D11ShaderResourceView *, 8u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        std::array<ID3D11Buffer *, 2u> constants {};
        std::array<ID3D11ClassInstance *, 256u> instances {};
        for (std::size_t slot = 0u; slot < srvs.size(); ++slot) srvs[slot] = srvs_[slot].Get();
        for (std::size_t slot = 0u; slot < uavs.size(); ++slot) uavs[slot] = uavs_[slot].Get();
        for (std::size_t slot = 0u; slot < constants.size(); ++slot) constants[slot] = constants_[slot].Get();
        for (UINT index = 0u; index < instance_count_; ++index) instances[index] = instances_[index].Get();
        context_->CSSetUnorderedAccessViews(0u, static_cast<UINT>(uavs.size()), uavs.data(), nullptr);
        context_->CSSetShaderResources(0u, static_cast<UINT>(srvs.size()), srvs.data());
        context_->CSSetConstantBuffers(0u, static_cast<UINT>(constants.size()), constants.data());
        context_->CSSetShader(shader_.Get(), instance_count_ ? instances.data() : nullptr, instance_count_);
      }
      static void unbind(ID3D11DeviceContext *context) noexcept {
        std::array<ID3D11ShaderResourceView *, 8u> srvs {};
        std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
        context->CSSetShaderResources(0u, static_cast<UINT>(srvs.size()), srvs.data());
        context->CSSetUnorderedAccessViews(0u, static_cast<UINT>(uavs.size()), uavs.data(), nullptr);
      }
    private:
      ID3D11DeviceContext *context_;
      std::array<ComPtr<ID3D11ShaderResourceView>, 8u> srvs_;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 8u> uavs_;
      std::array<ComPtr<ID3D11Buffer>, 2u> constants_;
      ComPtr<ID3D11ComputeShader> shader_;
      std::array<ComPtr<ID3D11ClassInstance>, 256u> instances_;
      UINT instance_count_ = 0u;
    };

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

    bool ensure(ID3D11Device *device, UINT width, UINT crop_height, std::string &error) {
      error.clear();
      constexpr UINT maximum = depth_coordinate_v2::subtitle_contour_source_max_extent;
      const auto pixels = std::uint64_t {width} * crop_height;
      if (!device || width == 0u || crop_height == 0u || width > maximum || crop_height > maximum ||
          pixels + 1u > std::numeric_limits<UINT>::max() / sizeof(std::uint32_t)) {
        error = "Invalid private inner-colour native extent/device.";
        return false;
      }
      if (device_.Get() == device && width_ == width && crop_height_ == crop_height) return true;
      resources_t next;
      next.device_ = device;
      next.width_ = width;
      next.crop_height_ = crop_height;
      next.pixels_ = static_cast<UINT>(pixels);
      try {
        if (!next.allocate(error)) return false;
      } catch (const std::exception &exception) {
        error = std::string {"Could not allocate private inner-colour working storage: "} + exception.what();
        return false;
      }
      *this = std::move(next);
      return true;
    }

    // Shader order is frozen: colour={palette,emit,clear,accumulate}, components=
    // {init,union,reduce,resolve,circle3,box3}. Only box3 is selected. The caller keeps every
    // source/CB/argument resource immutable through this ordered command interval.
    // All operands/shaders belong to device_; working outputs must be unbound outside the touched CS slots.
    // conditional_args[root_one_offset] must be the canonical GPU observation group count0/1.
    // A GPU four-byte copy places that gate in native clearArgs.Z; XY are static native geometry.
    // Root reuse0 therefore holds accumulated masks while clearing mode args prevents stale work.
    // True reports command recording only, never GPU completion, current evidence or mask quality.
    bool dispatch(
      ID3D11DeviceContext *context,
      const std::array<ID3D11ComputeShader *, 4u> &colour,
      const std::array<ID3D11ComputeShader *, 6u> &components,
      ID3D11Buffer *current_constants96,
      ID3D11ShaderResourceView *current_rgb,
      ID3D11ShaderResourceView *current_probability,
      ID3D11ShaderResourceView *current_baseline_support,
      ID3D11ShaderResourceView *current_ocr,
      ID3D11Buffer *conditional_args,
      UINT root_one_offset,
      std::string &error
    ) {
      error.clear();
      if (!context || !device_ || !current_constants96 || !current_rgb || !current_probability ||
          !current_baseline_support || !current_ocr || !conditional_args ||
          context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        error = "Private inner-colour dispatch lacks immediate-context/current inputs.";
        return false;
      }
      for (auto *shader : colour) if (!shader) {error = "Private colour shader array is incomplete.";return false;}
      for (auto *shader : components) if (!shader) {error = "Private component shader array is incomplete.";return false;}
      ComPtr<ID3D11Device> actual_device;
      context->GetDevice(&actual_device);
      if (actual_device.Get() != device_.Get()) {error = "Private inner-colour device/context mismatch.";return false;}
      D3D11_BUFFER_DESC constants_desc {}, args_desc {};
      current_constants96->GetDesc(&constants_desc);
      conditional_args->GetDesc(&args_desc);
      if (constants_desc.ByteWidth != 96u || (constants_desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0u ||
          (args_desc.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) == 0u ||
          root_one_offset % 4u != 0u || root_one_offset > args_desc.ByteWidth ||
          args_desc.ByteWidth - root_one_offset < 12u) {
        error = "Private inner-colour CB96/root-one indirect descriptor is invalid.";
        return false;
      }
      const saved_cs_t saved {context};
      saved_cs_t::unbind(context);
      const D3D11_BOX gate_word {root_one_offset, 0u, 0u, root_one_offset + 4u, 1u, 1u};
      context->CopySubresourceRegion(clear_args_.Get(), 0u, 8u, 0u, 0u,
                                    conditional_args, 0u, &gate_word);
      ID3D11Buffer *constants[2] {current_constants96, request_constants_.Get()};
      context->CSSetConstantBuffers(0u, 2u, constants);
      std::array<ID3D11UnorderedAccessView *, 8u> uavs {};
      uavs[5] = accumulated_body_.uav.Get();
      uavs[6] = accumulated_final_.uav.Get();
      context->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
      context->CSSetShader(colour[2], nullptr, 0u);
      context->DispatchIndirect(clear_args_.Get(), 0u);
      saved_cs_t::unbind(context);

      const std::array<ID3D11ShaderResourceView *, 8u> colour_inputs {
        current_rgb, current_probability, current_baseline_support, current_ocr,
        palette_.srv.Get(), nullptr, nullptr, nullptr
      };
      constexpr std::array<UINT, 4u> zero {};
      for (UINT slot = 0u; slot < 8u; ++slot) {
        saved_cs_t::unbind(context);
        // This unconditional argument clear is essential: a zero root dispatch does not write it.
        context->ClearUnorderedAccessViewUint(mode_args_.uav.Get(), zero.data());
        const std::array<UINT, 4u> palette_request {slot, 0u, 0u, 0u};
        context->UpdateSubresource(request_constants_.Get(), 0u, nullptr, palette_request.data(), 0u, 0u);
        constants[0] = current_constants96;
        context->CSSetConstantBuffers(0u, 2u, constants);
        // The palette output is not bound as t4 while u0 writes it.
        context->CSSetShaderResources(0u, 4u, colour_inputs.data());
        uavs.fill(nullptr);
        uavs[0] = palette_.uav.Get();
        uavs[7] = mode_args_.uav.Get();
        context->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
        context->CSSetShader(colour[0], nullptr, 0u);
        context->DispatchIndirect(conditional_args, root_one_offset);
        saved_cs_t::unbind(context);
        for (UINT mode = 0u; mode < 8u; ++mode) {
          const std::array<UINT, 4u> request {slot, mode, 0u, 0u};
          context->UpdateSubresource(request_constants_.Get(), 0u, nullptr, request.data(), 0u, 0u);
          const UINT argument_offset = mode * 12u;
          constants[0] = current_constants96;
          context->CSSetConstantBuffers(0u, 2u, constants);
          context->CSSetShaderResources(0u, 8u, colour_inputs.data());
          uavs.fill(nullptr);
          uavs[1] = eligible_.uav.Get();
          uavs[2] = seeds_.uav.Get();
          uavs[3] = domain_.uav.Get();
          uavs[4] = cover_.uav.Get();
          context->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
          context->CSSetShader(colour[1], nullptr, 0u);
          context->DispatchIndirect(mode_args_.resource.Get(), argument_offset);
          saved_cs_t::unbind(context);

          // The exact graph operator reads only this unit/mode's four emitted planes.
          ID3D11Buffer *graph_constants = graph_constants_.Get();
          context->CSSetConstantBuffers(0u, 1u, &graph_constants);
          const std::array<ID3D11ShaderResourceView *, 4u> component_inputs {
            eligible_.srv.Get(), seeds_.srv.Get(), domain_.srv.Get(), cover_.srv.Get()
          };
          context->CSSetShaderResources(0u, 4u, component_inputs.data());
          uavs.fill(nullptr);
          uavs[0] = parent_.uav.Get();
          uavs[1] = flags_.uav.Get();
          uavs[2] = mode_body_.uav.Get();
          uavs[3] = mode_final_.uav.Get();
          context->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
          for (const auto entry : {0u, 1u, 2u, 3u, 5u}) {
            context->CSSetShader(components[entry], nullptr, 0u);
            context->DispatchIndirect(mode_args_.resource.Get(), argument_offset);
          }
          saved_cs_t::unbind(context);

          constants[0] = current_constants96;
          context->CSSetConstantBuffers(0u, 2u, constants);
          auto accumulation_inputs = colour_inputs;
          accumulation_inputs[5] = mode_body_.srv.Get();
          accumulation_inputs[6] = mode_final_.srv.Get();
          accumulation_inputs[7] = parent_.srv.Get();
          context->CSSetShaderResources(0u, 8u, accumulation_inputs.data());
          uavs.fill(nullptr);
          uavs[5] = accumulated_body_.uav.Get();
          uavs[6] = accumulated_final_.uav.Get();
          context->CSSetUnorderedAccessViews(0u, 8u, uavs.data(), nullptr);
          context->CSSetShader(colour[3], nullptr, 0u);
          context->DispatchIndirect(mode_args_.resource.Get(), argument_offset);
          saved_cs_t::unbind(context);
        }
      }
      return true;
    }

    [[nodiscard]] mutable_views_t mutable_views() const {
      return {width_, crop_height_, accumulated_body_.srv, accumulated_final_.srv, palette_.srv,
        mode_args_.srv, parent_.srv, flags_.srv, eligible_.srv, seeds_.srv, domain_.srv,
        cover_.srv, mode_body_.srv, mode_final_.srv, mode_args_.resource, clear_args_};
    }

  private:
    bool allocate(std::string &error) {
      // Allocation-only initialization makes an initial zero-root observation hold defined zero.
      // This vector is not used per frame and is never a segmentation oracle.
      const auto zero_words = pixels_ >= 127u ? static_cast<std::size_t>(pixels_) + 1u : std::size_t {128u};
      const std::vector<std::uint32_t> zero(zero_words, 0u);
      const auto failure = [&](const char *what) {error = what;return false;};
      const auto make_texture = [&](texture_t &out) {
        D3D11_TEXTURE2D_DESC d {};
        d.Width = width_;d.Height = crop_height_;d.MipLevels = 1u;d.ArraySize = 1u;
        d.Format = DXGI_FORMAT_R32_UINT;d.SampleDesc.Count = 1u;d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        const D3D11_SUBRESOURCE_DATA initial {zero.data(), static_cast<UINT>(width_ * sizeof(std::uint32_t)), 0u};
        return SUCCEEDED(device_->CreateTexture2D(&d, &initial, &out.resource)) &&
          SUCCEEDED(device_->CreateShaderResourceView(out.resource.Get(), nullptr, &out.srv)) &&
          SUCCEEDED(device_->CreateUnorderedAccessView(out.resource.Get(), nullptr, &out.uav));
      };
      for (auto *plane : {&eligible_, &seeds_, &domain_, &cover_, &mode_body_, &mode_final_, &accumulated_body_, &accumulated_final_}) {
        if (!make_texture(*plane)) return failure("Could not allocate private inner-colour native plane.");
      }
      const auto make_structured = [&](UINT words, buffer_t &out) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = words * sizeof(std::uint32_t);d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride = sizeof(std::uint32_t);
        const D3D11_SUBRESOURCE_DATA initial {zero.data(), 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&d, &initial, &out.resource)) &&
          SUCCEEDED(device_->CreateShaderResourceView(out.resource.Get(), nullptr, &out.srv)) &&
          SUCCEEDED(device_->CreateUnorderedAccessView(out.resource.Get(), nullptr, &out.uav));
      };
      if (!make_structured(128u, palette_) || !make_structured(pixels_ + 1u, parent_) ||
          !make_structured(pixels_ + 1u, flags_)) return failure("Could not allocate private inner-colour structured buffer.");
      const std::array<UINT, 24u> zero_args {};
      D3D11_BUFFER_DESC args {};
      args.ByteWidth = sizeof(zero_args);args.Usage = D3D11_USAGE_DEFAULT;
      args.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      args.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      const D3D11_SUBRESOURCE_DATA args_initial {zero_args.data(), 0u, 0u};
      D3D11_SHADER_RESOURCE_VIEW_DESC arg_srv {};
      arg_srv.Format = DXGI_FORMAT_R32_TYPELESS;arg_srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      arg_srv.BufferEx.NumElements = 24u;arg_srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      D3D11_UNORDERED_ACCESS_VIEW_DESC arg_uav {};
      arg_uav.Format = DXGI_FORMAT_R32_TYPELESS;arg_uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      arg_uav.Buffer.NumElements = 24u;arg_uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device_->CreateBuffer(&args, &args_initial, &mode_args_.resource)) ||
          FAILED(device_->CreateShaderResourceView(mode_args_.resource.Get(), &arg_srv, &mode_args_.srv)) ||
          FAILED(device_->CreateUnorderedAccessView(mode_args_.resource.Get(), &arg_uav, &mode_args_.uav)))
        return failure("Could not allocate private inner-colour mode indirect arguments.");
      const std::array<UINT, 3u> native_args {(width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 0u};
      D3D11_BUFFER_DESC native_desc {};
      native_desc.ByteWidth = sizeof(native_args);native_desc.Usage = D3D11_USAGE_DEFAULT;
      native_desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      const D3D11_SUBRESOURCE_DATA native_initial {native_args.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&native_desc, &native_initial, &clear_args_)))
        return failure("Could not allocate private inner-colour native clear arguments.");
      const auto make_constants = [&](const std::array<UINT, 4u> &values, ComPtr<ID3D11Buffer> &out) {
        D3D11_BUFFER_DESC d {};
        d.ByteWidth = sizeof(values);d.Usage = D3D11_USAGE_DEFAULT;d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const D3D11_SUBRESOURCE_DATA initial {values.data(), 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&d, &initial, &out));
      };
      if (!make_constants({0u, 0u, 0u, 0u}, request_constants_) ||
          !make_constants({width_, crop_height_, pixels_, 0u}, graph_constants_))
        return failure("Could not allocate private inner-colour graph/request constants.");
      return true;
    }

    ComPtr<ID3D11Device> device_;
    UINT width_ = 0u, crop_height_ = 0u, pixels_ = 0u;
    texture_t eligible_, seeds_, domain_, cover_, mode_body_, mode_final_;
    texture_t accumulated_body_, accumulated_final_;
    buffer_t palette_, parent_, flags_, mode_args_;
    ComPtr<ID3D11Buffer> clear_args_, request_constants_, graph_constants_;
  };
}  // namespace models::host_sbs_subtitle_inner_colour_gpu
#endif  // _WIN32

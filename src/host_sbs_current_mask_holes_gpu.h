#pragma once

// Optional current-frame mask completion. Original proof and the original final
// mask are immutable inputs; newly completed pixels never supply another donor.
#ifdef _WIN32
  #include <d3d11.h>
  #include <wrl/client.h>
  #include <array>
  #include <cstdint>
  #include <limits>
  #include <string>
  #include <utility>
  #include <vector>

namespace models::host_sbs_current_mask_holes_gpu {
  using Microsoft::WRL::ComPtr;
  inline constexpr std::size_t shader_count = 9u;
  using programs_t = std::array<ID3D11ComputeShader *, shader_count>;
  using coverage_programs_t = std::array<ID3D11ComputeShader *, 3u>;

  struct inputs_t {
    ID3D11ShaderResourceView *rgb = nullptr, *ocr = nullptr, *selected = nullptr, *cover = nullptr;
    ID3D11ShaderResourceView *cbc = nullptr, *raw_meta = nullptr, *raw_parent = nullptr, *raw_flags = nullptr;
    ID3D11ShaderResourceView *raw_passable = nullptr, *inner = nullptr, *raw_witness = nullptr;
    ID3D11ShaderResourceView *old_final = nullptr, *raw_m = nullptr, *code1 = nullptr, *models = nullptr;
    ID3D11UnorderedAccessView *final_m = nullptr, *data = nullptr;
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
    texture_t proof_;
    buffer_t parent_, flags_, args_;
    texture_t coverage_offers_;
    buffer_t coverage_args_;
    // Immutable banks belong to this owner and are replaced atomically with it.
    // Slot9 is the invalid-unit hook; it preserves fail-closed public API calls.
    std::array<ComPtr<ID3D11Buffer>, 10u> unit_hooks_;
    std::array<ComPtr<ID3D11Buffer>, 8u> width_hooks_;
    ComPtr<ID3D11Buffer> hook_; // Bank sentinel retained by recorded signatures.
    ID3D11Buffer *active_hook_ = nullptr;
    std::array<ComPtr<ID3D11Buffer>, 2u> coverage_diagnostic_modes_;
    bool coverage_diagnostics_enabled_ = false;
    ComPtr<ID3D11Device> device_;
    UINT width_ = 0u, height_ = 0u;

    static void unbind(ID3D11DeviceContext *context) {
      const std::array<ID3D11ShaderResourceView *, 16u> in {};
      const std::array<ID3D11UnorderedAccessView *, 8u> out {};
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    }

    void stage(ID3D11DeviceContext *context, UINT entry, const programs_t &programs,
               ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
               ID3D11Buffer *arguments, UINT argument_offset) {
      unbind(context);
      std::array<ID3D11ShaderResourceView *, 16u> in {
        views.rgb, views.ocr, views.selected, views.cover, views.cbc,
        views.raw_meta, views.raw_parent, views.raw_flags, views.raw_passable,
        views.inner, views.raw_witness, nullptr, views.raw_m, nullptr,
        views.code1, views.models
      };
      std::array<ID3D11UnorderedAccessView *, 8u> out {};
      out[1] = parent_.uav.Get();
      out[2] = flags_.uav.Get();
      out[5] = views.data;
      if (entry == 0u || entry == 1u || entry == 3u) {
        out[0] = proof_.uav.Get();
      } else {
        in[13] = proof_.srv.Get();
      }
      if (entry == 0u || entry == 2u) out[3] = args_.uav.Get();
      if (entry == 3u) in[11] = views.old_final;
      if (entry == 7u) out[4] = views.final_m;
      ID3D11Buffer *buffers[3] {constants, active_hook_, shape};
      context->CSSetConstantBuffers(0u, 3u, buffers);
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
      context->CSSetShader(programs[entry], nullptr, 0u);
      context->DispatchIndirect(arguments, argument_offset);
      unbind(context);
    }

    void set_unit(ID3D11DeviceContext *, UINT unit, bool width_phase = false) {
      active_hook_ = width_phase && unit < width_hooks_.size() ? width_hooks_[unit].Get() :
                     !width_phase && unit < 9u ? unit_hooks_[unit].Get() : unit_hooks_[9u].Get();
    }

    void coverage_stage(ID3D11DeviceContext *context, UINT entry,
                        const coverage_programs_t &programs,
                        ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
                        ID3D11Buffer *arguments, UINT offset) {
      unbind(context);
      const std::array<ID3D11ShaderResourceView *, 16u> in {
        views.rgb, views.ocr, views.selected, views.cover, views.cbc,
        views.raw_meta, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        views.raw_m, proof_.srv.Get(), views.code1, views.models
      };
      std::array<ID3D11UnorderedAccessView *, 8u> out {};
      out[1] = parent_.uav.Get();
      out[2] = flags_.uav.Get();
      if (entry == 1u) out[3] = coverage_args_.uav.Get();
      out[5] = views.data;
      out[6] = coverage_diagnostics_enabled_ ? coverage_offers_.uav.Get() : nullptr;
      ID3D11Buffer *buffers[4] {constants, active_hook_, shape,
        coverage_diagnostic_modes_[coverage_diagnostics_enabled_ ? 1u : 0u].Get()};
      context->CSSetConstantBuffers(0u, 4u, buffers);
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
      context->CSSetShader(programs[entry], nullptr, 0u);
      context->DispatchIndirect(arguments, offset);
      unbind(context);
    }

  public:
    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      error.clear();
      if (!device || !width || !height || width > 5120u || height > 5120u ||
          uint64_t(width) * height + 1u > std::numeric_limits<UINT>::max() / 4u) {
        error = "Invalid optional current-mask-completion shape.";
        return false;
      }
      if (device_.Get() == device && width_ == width && height_ == height &&
          proof_.uav && parent_.uav && flags_.uav && args_.uav && hook_ &&
          unit_hooks_[9u] && width_hooks_[7u]) return true;
      resources_t next;
      next.device_ = device;
      next.width_ = width;
      next.height_ = height;
      D3D11_TEXTURE2D_DESC texture {};
      texture.Width = width;
      texture.Height = height;
      texture.MipLevels = texture.ArraySize = texture.SampleDesc.Count = 1u;
      texture.Format = DXGI_FORMAT_R32_UINT;
      texture.Usage = D3D11_USAGE_DEFAULT;
      texture.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      // Diagnostic views can read these planes before the first authenticated clear, so they
      // start as literal zero instead of undefined driver memory.
      const std::vector<UINT> zero(static_cast<std::size_t>(width) * height + 32u, 0u);
      const D3D11_SUBRESOURCE_DATA zero_texture {zero.data(), width * 4u, 0u};
      const D3D11_SUBRESOURCE_DATA zero_buffer {zero.data(), 0u, 0u};
      if (FAILED(device->CreateTexture2D(&texture, &zero_texture, &next.proof_.resource)) ||
          FAILED(device->CreateShaderResourceView(next.proof_.resource.Get(), nullptr, &next.proof_.srv)) ||
          FAILED(device->CreateUnorderedAccessView(next.proof_.resource.Get(), nullptr, &next.proof_.uav))) {
        error = "Could not allocate optional current-mask proof.";
        return false;
      }
      auto structured = [&](buffer_t &target) {
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = (width * height + 1u) * 4u;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = 4u;
        return SUCCEEDED(device->CreateBuffer(&desc, &zero_buffer, &target.resource)) &&
               SUCCEEDED(device->CreateShaderResourceView(target.resource.Get(), nullptr, &target.srv)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(target.resource.Get(), nullptr, &target.uav));
      };
      if (!structured(next.parent_) || !structured(next.flags_)) {
        error = "Could not allocate optional current-mask graph.";
        return false;
      }
      D3D11_BUFFER_DESC args {};
      args.ByteWidth = 128u;
      args.Usage = D3D11_USAGE_DEFAULT;
      args.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      args.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
      uav.Format = DXGI_FORMAT_R32_TYPELESS;
      uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav.Buffer.NumElements = 32u;
      uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device->CreateBuffer(&args, &zero_buffer, &next.args_.resource)) ||
          FAILED(device->CreateUnorderedAccessView(next.args_.resource.Get(), &uav, &next.args_.uav))) {
        error = "Could not allocate optional current-mask indirect arguments.";
        return false;
      }
      D3D11_BUFFER_DESC hook {};
      hook.ByteWidth = 16u;
      hook.Usage = D3D11_USAGE_IMMUTABLE;
      hook.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const auto make_hook = [&](ComPtr<ID3D11Buffer> &target, UINT unit, UINT phase) {
        const std::array<UINT, 4u> words {unit, phase, 0u, 0u};
        const D3D11_SUBRESOURCE_DATA initial {words.data(), 0u, 0u};
        return SUCCEEDED(device->CreateBuffer(&hook, &initial, &target));
      };
      for (UINT unit = 0u; unit < next.unit_hooks_.size(); ++unit) {
        if (!make_hook(next.unit_hooks_[unit], unit < 9u ? unit : UINT(-1), 0u)) {
          error = "Could not allocate immutable current-mask unit constants.";return false;
        }
      }
      for (UINT unit = 0u; unit < next.width_hooks_.size(); ++unit) {
        if (!make_hook(next.width_hooks_[unit], unit, 1u)) {
          error = "Could not allocate immutable current-mask width-phase constants.";return false;
        }
      }
      next.hook_ = next.unit_hooks_[0u];
      next.active_hook_ = next.hook_.Get();
      *this = std::move(next);
      return true;
    }

    template<class Add>
    void signature_objects(Add &&add) const {
      const std::array<IUnknown *, 19u> objects {
        proof_.resource.Get(), proof_.srv.Get(), proof_.uav.Get(),
        parent_.resource.Get(), parent_.srv.Get(), parent_.uav.Get(),
        flags_.resource.Get(), flags_.srv.Get(), flags_.uav.Get(),
        args_.resource.Get(), args_.uav.Get(), hook_.Get(), device_.Get(),
        coverage_args_.resource.Get(), coverage_args_.uav.Get(),
        coverage_offers_.resource.Get(), coverage_offers_.srv.Get(), coverage_offers_.uav.Get(),
        coverage_diagnostic_modes_[coverage_diagnostics_enabled_ ? 1u : 0u].Get()
      };
      for (IUnknown *value : objects) add(value);
    }

    // Coverage selection needs only arguments and its immutable diagnostic-OFF mode.
    // The optional offer plane never controls selection or the shared pending commit.
    bool ensure_coverage(std::string &error, bool diagnostics = false) {
      error.clear();coverage_diagnostics_enabled_ = false;
      if (!device_ || !proof_.uav || !width_ || !height_) {
        error = "Current-mask proof is unavailable for stroke coverage.";return false;
      }
      D3D11_BUFFER_DESC mode {};
      mode.ByteWidth = 16u;mode.Usage = D3D11_USAGE_IMMUTABLE;
      mode.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const auto make_mode = [&](ComPtr<ID3D11Buffer> &target, UINT enabled) {
        const std::array<UINT, 4u> words {enabled, 0u, 0u, 0u};
        const D3D11_SUBRESOURCE_DATA initial {words.data(), 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&mode, &initial, &target));
      };
      if (!coverage_args_.uav || !coverage_diagnostic_modes_[0u]) {
        buffer_t next_args;ComPtr<ID3D11Buffer> next_mode;
        D3D11_BUFFER_DESC args {};
        args.ByteWidth = 128u;args.Usage = D3D11_USAGE_DEFAULT;
        args.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        args.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = 32u;uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device_->CreateBuffer(&args, nullptr, &next_args.resource)) ||
            FAILED(device_->CreateUnorderedAccessView(next_args.resource.Get(), &uav, &next_args.uav)) ||
            !make_mode(next_mode, 0u)) {
          error = "Could not allocate optional stroke-coverage arguments.";return false;
        }
        coverage_args_ = std::move(next_args);coverage_diagnostic_modes_[0u] = std::move(next_mode);
      }
      if (!diagnostics) return true;
      if (!coverage_offers_.uav || !coverage_diagnostic_modes_[1u]) {
        texture_t next_offers;ComPtr<ID3D11Buffer> next_mode;
        D3D11_TEXTURE2D_DESC texture {};
        texture.Width = width_;texture.Height = height_;
        texture.MipLevels = texture.ArraySize = texture.SampleDesc.Count = 1u;
        texture.Format = DXGI_FORMAT_R32_UINT;texture.Usage = D3D11_USAGE_DEFAULT;
        texture.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(device_->CreateTexture2D(&texture, nullptr, &next_offers.resource)) ||
            FAILED(device_->CreateShaderResourceView(next_offers.resource.Get(), nullptr, &next_offers.srv)) ||
            FAILED(device_->CreateUnorderedAccessView(next_offers.resource.Get(), nullptr, &next_offers.uav)) ||
            !make_mode(next_mode, 1u)) {
          // An unavailable diagnostic cannot revoke an otherwise usable model.
          error = "Stroke-coverage offer diagnostics are unavailable.";return true;
        }
        coverage_offers_ = std::move(next_offers);coverage_diagnostic_modes_[1u] = std::move(next_mode);
      }
      coverage_diagnostics_enabled_ = true;
      return true;
    }

    void begin(ID3D11DeviceContext *context, const programs_t &programs,
               ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
               ID3D11Buffer *native_arguments) {
      set_unit(context, 0u);
      stage(context, 0u, programs, constants, shape, views, native_arguments, 0u);
    }

    void begin_coverage(ID3D11DeviceContext *context, const coverage_programs_t &programs,
                        ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
                        ID3D11Buffer *native_arguments) {
      if (!coverage_diagnostics_enabled_) return;
      set_unit(context, 0u);
      coverage_stage(context, 0u, programs, constants, shape, views, native_arguments, 0u);
    }

    void capture_unit(ID3D11DeviceContext *context, UINT unit, const programs_t &programs,
                      ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
                      ID3D11Buffer *unit_arguments) {
      set_unit(context, unit);
      stage(context, 1u, programs, constants, shape, views, unit_arguments, 0u);
    }

    void finish(ID3D11DeviceContext *context, const programs_t &programs,
                ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
                ID3D11Buffer *root_arguments, UINT root_offset, ID3D11Buffer *native_arguments,
                const coverage_programs_t &coverage_programs = {}, bool coverage_enabled = false) {
      // Root0 must not execute any graph from an older nonzero indirect vector.
      unbind(context);
      const std::array<UINT, 4u> zero {};
      context->ClearUnorderedAccessViewUint(args_.uav.Get(), zero.data());
      set_unit(context, 0u);
      stage(context, 3u, programs, constants, shape, views, native_arguments, 0u);
      stage(context, 2u, programs, constants, shape, views, root_arguments, root_offset);
      if (coverage_enabled) {
        context->ClearUnorderedAccessViewUint(coverage_args_.uav.Get(), zero.data());
        coverage_stage(context, 1u, coverage_programs, constants, shape, views, root_arguments, root_offset);
        for (UINT unit = 0u; unit < 8u; ++unit) {
          set_unit(context, unit);
          coverage_stage(context, 2u, coverage_programs, constants, shape, views,
                         coverage_args_.resource.Get(), 16u * unit);
        }
      }
      for (UINT unit = 0u; unit < 8u; ++unit) {
        set_unit(context, unit);
        for (UINT entry : {4u, 5u, 6u})
          stage(context, entry, programs, constants, shape, views, args_.resource.Get(), 16u * unit);
        // Complete cheap component vetoes before any width work. Reuse entry6
        // with a separate immutable hook; no new shader entry/manifest ABI.
        set_unit(context, unit, true);
        stage(context, 6u, programs, constants, shape, views, args_.resource.Get(), 16u * unit);
        set_unit(context, unit);
        for (UINT entry : {8u, 7u})
          stage(context, entry, programs, constants, shape, views, args_.resource.Get(), 16u * unit);
      }
      // Per-unit publication records private pending bits. Commit only after
      // every graph completed and the whole-phase fault word remained clear.
      set_unit(context, 8u);
      stage(context, 7u, programs, constants, shape, views, native_arguments, 0u);
    }

    ID3D11ShaderResourceView *proof() const noexcept { return proof_.srv.Get(); }
    ID3D11ShaderResourceView *parents() const noexcept { return parent_.srv.Get(); }
    ID3D11ShaderResourceView *flags() const noexcept { return flags_.srv.Get(); }
    bool coverage_diagnostics_enabled() const noexcept { return coverage_diagnostics_enabled_; }
    ID3D11ShaderResourceView *coverage_offers() const noexcept {
      return coverage_diagnostics_enabled_ ? coverage_offers_.srv.Get() : nullptr;
    }
  };
}
#endif

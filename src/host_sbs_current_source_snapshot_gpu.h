#pragma once
// Default-off research source transport. No detection, Base or current-EMPTY permission.
#ifdef _WIN32
  #include <array>
  #include <cstdint>
  #include <d3d11.h>
  #include <string>
  #include <utility>
  #include <wrl/client.h>

namespace models::research_current_source_snapshot_v1 {
  using Microsoft::WRL::ComPtr;
  inline constexpr std::uint32_t tag = 0x31505343u;

  struct alignas(16) constants_t {
    std::array<std::uint32_t, 4> control {}, source_rect {}, band_field {}, content {}, current {}, scope {}, domain {}, reserved {};
  };

  static_assert(sizeof(constants_t) == 128u);
  using receipt_t = std::array<std::uint32_t, 64>;

  struct mutable_views_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> source;
    ComPtr<ID3D11Buffer> receipt_buffer;
    ComPtr<ID3D11ShaderResourceView> receipt;
  };

  class resources_t {
    class saved_cs_t {
      ID3D11DeviceContext *context_;
      ComPtr<ID3D11ShaderResourceView> srv_;
      std::array<ComPtr<ID3D11UnorderedAccessView>, 2> uavs_;
      ComPtr<ID3D11Buffer> cb_;
      ComPtr<ID3D11ComputeShader> shader_;
      std::array<ComPtr<ID3D11ClassInstance>, 256> instances_;
      UINT count_ = 256u;

    public:
      explicit saved_cs_t(ID3D11DeviceContext *c):
          context_(c) {
        c->CSGetShaderResources(0, 1, srv_.GetAddressOf());
        ID3D11UnorderedAccessView *u[2] {};
        c->CSGetUnorderedAccessViews(0, 2, u);
        for (UINT i = 0; i < 2u; ++i) {
          uavs_[i].Attach(u[i]);
        }
        c->CSGetConstantBuffers(0, 1, cb_.GetAddressOf());
        ID3D11ClassInstance *instances[256] {};
        c->CSGetShader(shader_.GetAddressOf(), instances, &count_);
        for (UINT i = 0; i < count_; ++i) {
          instances_[i].Attach(instances[i]);
        }
      }

      ~saved_cs_t() {
        unbind(context_);
        ID3D11UnorderedAccessView *u[2] {uavs_[0].Get(), uavs_[1].Get()};
        context_->CSSetUnorderedAccessViews(0, 2, u, nullptr);
        auto *s = srv_.Get();
        context_->CSSetShaderResources(0, 1, &s);
        auto *b = cb_.Get();
        context_->CSSetConstantBuffers(0, 1, &b);
        ID3D11ClassInstance *instances[256] {};
        for (UINT i = 0; i < count_; ++i) {
          instances[i] = instances_[i].Get();
        }
        context_->CSSetShader(shader_.Get(), count_ ? instances : nullptr, count_);
      }

      static void unbind(ID3D11DeviceContext *c) {
        ID3D11ShaderResourceView *s = nullptr;
        ID3D11UnorderedAccessView *u[2] {};
        c->CSSetShaderResources(0, 1, &s);
        c->CSSetUnorderedAccessViews(0, 2, u, nullptr);
      }

      saved_cs_t(const saved_cs_t &) = delete;
      saved_cs_t &operator=(const saved_cs_t &) = delete;
    };

    ComPtr<ID3D11Device> device_;
    UINT width_ = 0, height_ = 0;
    mutable_views_t views_;
    ComPtr<ID3D11UnorderedAccessView> source_uav_, receipt_uav_;
    ComPtr<ID3D11Buffer> cb_;

  public:
    resources_t() = default;
    resources_t(const resources_t &) = delete;
    resources_t &operator=(const resources_t &) = delete;
    resources_t(resources_t &&) = default;
    resources_t &operator=(resources_t &&) = default;

    bool ensure(ID3D11Device *device, UINT width, UINT height, std::string &error) {
      error.clear();
      if (!device || !width || !height || width > 5120u || height > 5120u) {
        error = "Invalid current-source extent/device";
        return false;
      }
      if (device_.Get() == device && width_ == width && height_ == height) {
        return true;
      }
      resources_t next;
      next.device_ = device;
      next.width_ = width;
      next.height_ = height;
      D3D11_TEXTURE2D_DESC td {};
      td.Width = width;
      td.Height = height;
      td.MipLevels = td.ArraySize = td.SampleDesc.Count = 1;
      td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      D3D11_BUFFER_DESC bd {};
      bd.ByteWidth = sizeof(receipt_t);
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      bd.StructureByteStride = 4;
      if (FAILED(device->CreateTexture2D(&td, nullptr, &next.views_.texture)) || FAILED(device->CreateShaderResourceView(next.views_.texture.Get(), nullptr, &next.views_.source)) || FAILED(device->CreateUnorderedAccessView(next.views_.texture.Get(), nullptr, &next.source_uav_)) || FAILED(device->CreateBuffer(&bd, nullptr, &next.views_.receipt_buffer)) || FAILED(device->CreateShaderResourceView(next.views_.receipt_buffer.Get(), nullptr, &next.views_.receipt)) || FAILED(device->CreateUnorderedAccessView(next.views_.receipt_buffer.Get(), nullptr, &next.receipt_uav_))) {
        error = "Current-source allocation failed";
        return false;
      }
      bd = {};
      bd.ByteWidth = sizeof(constants_t);
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device->CreateBuffer(&bd, nullptr, &next.cb_))) {
        error = "Current-source constants allocation failed";
        return false;
      }
      *this = std::move(next);
      return true;
    }

    // Caller owns the actual current capture permission, source/route identity and texture
    // lifetime. Echoed serial/clock/domain tags cannot establish that provenance themselves.
    // All operands must be same-device, serialized, and unbound outside touched CS slots.
    // true means commands recorded, not completion/readiness. Inspect the own receipt after
    // completion. Returned views are mutable and must be copied before a later dispatch/ensure.
    bool dispatch(ID3D11DeviceContext *context, const std::array<ID3D11ComputeShader *, 3> &shaders, ID3D11ShaderResourceView *current_capture, const constants_t &constants, std::string &error) {
      error.clear();
      if (!context || !device_ || !current_capture || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
        error = "Invalid current-source dispatch";
        return false;
      }
      ComPtr<ID3D11Predicate> predicate;
      BOOL predicate_value = FALSE;
      context->GetPredication(&predicate, &predicate_value);
      if (predicate) {
        error = "Current-source dispatch cannot run under predication";
        return false;
      }
      ComPtr<ID3D11Device> owner;
      context->GetDevice(&owner);
      if (owner.Get() != device_.Get()) {
        error = "Current-source context device mismatch";
        return false;
      }
      current_capture->GetDevice(owner.ReleaseAndGetAddressOf());
      if (owner.Get() != device_.Get()) {
        error = "Current-source capture device mismatch";
        return false;
      }
      for (auto *shader : shaders) {
        if (!shader) {
          error = "Missing current-source shader";
          return false;
        }
        shader->GetDevice(owner.ReleaseAndGetAddressOf());
        if (owner.Get() != device_.Get()) {
          error = "Current-source shader device mismatch";
          return false;
        }
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC desc {};
      current_capture->GetDesc(&desc);
      if (desc.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || desc.Texture2D.MipLevels != 1u) {
        error = "Current-source requires one Texture2D mip";
        return false;
      }
      ComPtr<ID3D11Resource> capture_resource;
      current_capture->GetResource(&capture_resource);
      ComPtr<IUnknown> capture_identity, output_identity;
      if (!capture_resource || FAILED(capture_resource.As(&capture_identity)) || FAILED(views_.texture.As(&output_identity)) || capture_identity.Get() == output_identity.Get()) {
        error = "Current-source input must not alias its mutable output";
        return false;
      }
      if (desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT && desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT) {
        error = "Unsupported current-source color view";
        return false;
      }
      const saved_cs_t saved {context};
      saved_cs_t::unbind(context);
      context->UpdateSubresource(cb_.Get(), 0, nullptr, &constants, 0, 0);
      auto *cb = cb_.Get();
      context->CSSetConstantBuffers(0, 1, &cb);
      context->CSSetShaderResources(0, 1, &current_capture);
      ID3D11UnorderedAccessView *outputs[2] {source_uav_.Get(), receipt_uav_.Get()};
      context->CSSetUnorderedAccessViews(0, 2, outputs, nullptr);
      context->CSSetShader(shaders[0], nullptr, 0);
      context->Dispatch(1, 1, 1);
      context->CSSetShader(shaders[1], nullptr, 0);
      context->Dispatch((width_ + 15) / 16, (height_ + 15) / 16, 1);
      context->CSSetShader(shaders[2], nullptr, 0);
      context->Dispatch(1, 1, 1);
      return true;
    }

    // Mutable private CB alias for benchmark-owned literal GPU readback only.
    // It grants no producer/current-UI authority and must be copied before dispatch/ensure.
    ComPtr<ID3D11Buffer> working_constants_buffer() const { return cb_; }
    mutable_views_t working_views() const {
      return views_;
    }
  };
}  // namespace models::research_current_source_snapshot_v1
#endif

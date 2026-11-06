#pragma once

// Optional source-colour FG edge refinement AFTER Inside and C7/G1 publication.
// The source mask is copied once; no emitted pixel ever becomes another donor.
#ifdef _WIN32
  #include <d3d11.h>
  #include <wrl/client.h>
  #include <array>
  #include <string>
  #include <utility>

namespace models::host_sbs_current_foreground_edge_gpu {
  using Microsoft::WRL::ComPtr;
  using programs_t = std::array<ID3D11ComputeShader *, 1u>;

  struct inputs_t {
    ID3D11ShaderResourceView *rgb = nullptr, *ocr = nullptr, *selected = nullptr, *cover = nullptr;
    ID3D11ShaderResourceView *cbc = nullptr, *models = nullptr, *sct = nullptr;
    ID3D11ShaderResourceView *inside = nullptr, *body = nullptr, *raw_m = nullptr, *code1 = nullptr;
    ID3D11ShaderResourceView *trusted_inner = nullptr;
    ID3D11Resource *original_final = nullptr;
    ID3D11UnorderedAccessView *final_m = nullptr, *data = nullptr;
  };

  class resources_t {
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Texture2D> snapshot_;
    ComPtr<ID3D11ShaderResourceView> snapshot_view_;
    ComPtr<ID3D11Resource> source_final_;
    ComPtr<ID3D11UnorderedAccessView> final_view_;
    UINT width_ = 0u, height_ = 0u;

    static void unbind(ID3D11DeviceContext *context) {
      const std::array<ID3D11ShaderResourceView *, 16u> in {};
      const std::array<ID3D11UnorderedAccessView *, 8u> out {};
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    }

  public:
    bool ensure(ID3D11Device *device, UINT width, UINT height,
                ID3D11Resource *original_final, ID3D11UnorderedAccessView *final_view,
                std::string &error) {
      error.clear();
      if (!device || !original_final || !final_view || width == 0u || height == 0u || width > 5120u || height > 5120u) {
        error = "Invalid optional foreground-edge snapshot shape.";return false;
      }
      if (device_.Get() == device && width_ == width && height_ == height && snapshot_view_ &&
          source_final_.Get() == original_final && final_view_.Get() == final_view) return true;
      ComPtr<ID3D11Resource> output;
      final_view->GetResource(&output);
      ComPtr<ID3D11Texture2D> source;
      ComPtr<ID3D11Device> source_device, view_device;
      original_final->GetDevice(&source_device);final_view->GetDevice(&view_device);
      D3D11_UNORDERED_ACCESS_VIEW_DESC view_desc {};final_view->GetDesc(&view_desc);
      if (output.Get() != original_final || source_device.Get() != device || view_device.Get() != device ||
          FAILED(original_final->QueryInterface(IID_PPV_ARGS(&source))) ||
          view_desc.ViewDimension != D3D11_UAV_DIMENSION_TEXTURE2D || view_desc.Texture2D.MipSlice != 0u ||
          view_desc.Format != DXGI_FORMAT_R32_UINT) {
        error = "Optional foreground-edge output is not the owned native mask.";return false;
      }
      D3D11_TEXTURE2D_DESC source_desc {};source->GetDesc(&source_desc);
      if (source_desc.Width != width || source_desc.Height != height || source_desc.Format != DXGI_FORMAT_R32_UINT ||
          source_desc.MipLevels != 1u || source_desc.ArraySize != 1u || source_desc.SampleDesc.Count != 1u || source_desc.SampleDesc.Quality != 0u) {
        error = "Optional foreground-edge source mask shape/format mismatch.";return false;
      }
      resources_t next;
      next.device_ = device;next.width_ = width;next.height_ = height;
      next.source_final_ = original_final;next.final_view_ = final_view;
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1u;
      desc.Format = DXGI_FORMAT_R32_UINT;desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &next.snapshot_)) ||
          FAILED(device->CreateShaderResourceView(next.snapshot_.Get(), nullptr, &next.snapshot_view_))) {
        error = "Could not allocate optional post-completion mask snapshot.";return false;
      }
      *this = std::move(next);return true;
    }

    template<class Add>
    void signature_objects(Add &&add) const {
      add(snapshot_.Get());add(snapshot_view_.Get());add(source_final_.Get());add(final_view_.Get());
    }

    // The parent owns/validates all inputs and keeps its source/command lease
    // through selector consumption. native_arguments is the existing Root1
    // dispatch, never a retained per-unit graph vector. Copying a held Root0
    // image grants no freshness: its zero dispatch still issues no shader work.
    void finish(ID3D11DeviceContext *context, const programs_t &programs,
                ID3D11Buffer *constants, ID3D11Buffer *shape, const inputs_t &views,
                ID3D11Buffer *native_arguments) {
      if (!context || !programs[0u] || !snapshot_ || !snapshot_view_ ||
          !constants || !shape || !native_arguments || !views.original_final ||
          !views.final_m || !views.data) return;
      // Descriptor/device checks are cached by ensure's exact immutable tuple;
      // recording below remains pure GPU commands (no Get*, query or readback).
      if (views.original_final != source_final_.Get() || views.final_m != final_view_.Get()) return;
      unbind(context);
      context->CopyResource(snapshot_.Get(), views.original_final);
      const std::array<ID3D11ShaderResourceView *, 16u> in {
        views.rgb, views.ocr, views.selected, views.cover, views.cbc, nullptr,
        views.models, views.sct, nullptr, views.inside, views.body,
        snapshot_view_.Get(), views.raw_m, views.code1, nullptr, nullptr
      };
      std::array<ID3D11UnorderedAccessView *, 8u> out {};
      out[4u] = views.final_m;out[5u] = views.data;
      // F2 is read via the UAV at u5; t5 is deliberately NULL to avoid a
      // simultaneous SRV/UAV alias while unique-add counters are adjusted.
      ID3D11Buffer *buffers[2u] {constants, shape};
      context->CSSetConstantBuffers(0u, 2u, buffers);
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
      context->CSSetShader(programs[0u], nullptr, 0u);
      context->DispatchIndirect(native_arguments, 0u);
      unbind(context);
    }

    // Separate optional program and zero-work indirect gate. It does not depend on the
    // foreground-edge snapshot allocation or enablement and cannot suppress the old pass.
    static void finish_trusted(ID3D11DeviceContext *context, ID3D11ComputeShader *program,
                               ID3D11Buffer *constants, ID3D11Buffer *shape,
                               const inputs_t &views, ID3D11Buffer *publication_arguments) {
      if (!context || !program || !constants || !shape || !views.trusted_inner ||
          !views.final_m || !views.data || !publication_arguments) return;
      unbind(context);
      const std::array<ID3D11ShaderResourceView *, 16u> in {
        views.rgb, views.ocr, views.selected, views.cover, views.cbc, nullptr,
        views.models, views.sct, views.trusted_inner, nullptr, nullptr,
        nullptr, views.raw_m, views.code1, nullptr, nullptr
      };
      std::array<ID3D11UnorderedAccessView *, 8u> out{};
      out[4u] = views.final_m;out[5u] = views.data;
      ID3D11Buffer *buffers[2u] {constants, shape};
      context->CSSetConstantBuffers(0u, 2u, buffers);
      context->CSSetShaderResources(0u, UINT(in.size()), in.data());
      context->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
      context->CSSetShader(program, nullptr, 0u);
      context->DispatchIndirect(publication_arguments, 0u);
      unbind(context);
    }

    // Observation only. A held/disabled invocation is not a new snapshot claim.
    ID3D11ShaderResourceView *original_mask() const noexcept { return snapshot_view_.Get(); }
  };
}
#endif

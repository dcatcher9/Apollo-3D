#pragma once

#ifdef _WIN32
#include "host_sbs_eye_field.h"
#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <d3d11.h>
#include <wrl/client.h>

namespace models::subtitle_publication {
  using Microsoft::WRL::ComPtr;

  // These are the seven GPU operands of one completed subtitle draw. Current source RGB
  // still comes from the converter on every draw and is compared with this display band.
  enum operand : std::size_t { base, state, matte, display_band, condition, identity, receipt, count };
  using inputs_t = std::array<ID3D11ShaderResourceView *, count>;
  struct views_t {
    std::array<ComPtr<ID3D11ShaderResourceView>, count> srv;
  };

  class pool_t {
    struct resource_copy_t {
      ComPtr<ID3D11ShaderResourceView> source, view;
      ComPtr<ID3D11Resource> original, copy;
    };
    struct slot_t {
      std::shared_ptr<views_t> views;
      std::array<resource_copy_t, count> operands;
      resource_copy_t eye_field;
    };
    // On the converter thread one publication may still be retained while its successor is
    // completed. A reader holding its lease keeps that slot out of the writer's rotation.
    std::array<slot_t, 2> slots_;

    static bool prepare(ID3D11Device *device, resource_copy_t &snapshot, ID3D11ShaderResourceView *source) {
      if (!source) return false;
      if (snapshot.source.Get() == source) return true;
      ComPtr<ID3D11Device> owner;
      source->GetDevice(&owner);
      if (owner.Get() != device) return false;
      ComPtr<ID3D11Resource> original, copy;
      source->GetResource(&original);
      D3D11_RESOURCE_DIMENSION dimension;
      original->GetType(&dimension);
      if (dimension == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> buffer, target;
        if (FAILED(original.As(&buffer))) return false;
        D3D11_BUFFER_DESC desc;
        buffer->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags &= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if (FAILED(device->CreateBuffer(&desc, nullptr, &target))) return false;
        copy = target;
      } else if (dimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> texture, target;
        if (FAILED(original.As(&texture))) return false;
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &target))) return false;
        copy = target;
      } else {
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC view;
      source->GetDesc(&view);
      ComPtr<ID3D11ShaderResourceView> srv;
      if (FAILED(device->CreateShaderResourceView(copy.Get(), &view, &srv))) return false;
      snapshot.original = std::move(original);
      snapshot.copy = std::move(copy);
      snapshot.view = std::move(srv);
      snapshot.source = source;
      return true;
    }

  public:
    std::shared_ptr<const views_t> publish(ID3D11Device *device, ID3D11DeviceContext *context, const inputs_t &sources) {
      if (!device || !context) return {};
      for (auto &slot : slots_) {
        if (slot.views && slot.views.use_count() != 1) continue;
        if (!slot.views) slot.views = std::make_shared<views_t>();
        for (std::size_t i = 0; i < count; ++i) {
          if (!prepare(device, slot.operands[i], sources[i])) return {};
          slot.views->srv[i] = slot.operands[i].view;
        }
        // The companion is private data on the Base SRV. CopyResource only copies texels.
        // Snapshot both fields before the producer can advance to another frame.
        const auto eye = host_sbs_eye_field::companion(sources[base]);
        if (eye) {
          if (!prepare(device, slot.eye_field, eye.Get()) ||
              !host_sbs_eye_field::attach(slot.views->srv[base].Get(), slot.eye_field.view.Get())) return {};
          context->CopyResource(slot.eye_field.copy.Get(), slot.eye_field.original.Get());
        } else if (slot.eye_field.view) {
          if (!host_sbs_eye_field::attach(slot.views->srv[base].Get(), nullptr)) return {};
          slot.eye_field = {};
        }
        // Ordered on the same immediate context as the producer's writes. No CPU wait,
        // query, flush, or transfer across the PCIe bus is required to close a publication.
        for (const auto &operand : slot.operands) context->CopyResource(operand.copy.Get(), operand.original.Get());
        return slot.views;
      }
      return {};
    }
  };
}
#endif

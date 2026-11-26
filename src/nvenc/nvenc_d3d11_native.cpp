/**
 * @file src/nvenc/nvenc_d3d11_native.cpp
 * @brief Definitions for native Direct3D11 NVENC encoder.
 */
// this include
#include "nvenc_d3d11_native.h"

// local includes
#include "nvenc_utils.h"

#include <algorithm>
#include <cassert>

namespace nvenc {

  nvenc_d3d11_native::nvenc_d3d11_native(ID3D11Device *d3d_device, bool allow_p010_chroma_store):
      nvenc_d3d11(),
      d3d_device(d3d_device),
      allow_p010_chroma_store(allow_p010_chroma_store) {
    device = d3d_device;
  }

  nvenc_d3d11_native::~nvenc_d3d11_native() {
    if (encoder) {
      destroy_encoder();
    }
  }

  ID3D11Texture2D *
    nvenc_d3d11_native::get_input_texture() {
    return get_input_texture(0);
  }

  unsigned nvenc_d3d11_native::input_surface_count() const noexcept {
    const auto depth = pipeline_depth();
    return depth > 1 ? depth + 1 : 1;
  }

  ID3D11Texture2D *nvenc_d3d11_native::get_input_texture(unsigned surface) {
    assert(surface < input_surface_count());
    return input_surfaces[surface].GetInterfacePtr();
  }

  std::optional<unsigned> nvenc_d3d11_native::begin_input_conversion() noexcept {
    if (!input_registration_is_mapped(registered_input_buffers[converted_input])) {
      return converted_input;
    }
    for (unsigned surface = 0; surface < input_surface_count(); ++surface) {
      if (!input_registration_is_mapped(registered_input_buffers[surface])) {
        return surface;
      }
    }
    return std::nullopt;
  }

  void nvenc_d3d11_native::set_converted_input(unsigned surface) noexcept {
    assert(surface < input_surface_count());
    assert(!input_registration_is_mapped(registered_input_buffers[surface]));
    converted_input = surface;
  }

  bool nvenc_d3d11_native::input_waits_for_retirement() const noexcept {
    return input_registration_in_flight(registered_input_buffers[converted_input]);
  }

  bool nvenc_d3d11_native::create_and_register_input_buffer() {
    converted_input = 0;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = encoder_params.width;
    desc.Height = encoder_params.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = dxgi_format_from_nvenc_format(encoder_params.buffer_format);
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (allow_p010_chroma_store && desc.Format == DXGI_FORMAT_P010) {
      desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    }
    const auto count = input_surface_count();
    for (unsigned surface = count; surface < input_surfaces.size(); ++surface) {
      input_surfaces[surface] = nullptr;
    }
    const auto register_surface = [&](unsigned surface) {
      auto &input = input_surfaces[surface];
      auto &registration = registered_input_buffers[surface];
      if (registration) {
        return true;
      }
      if (!input && d3d_device->CreateTexture2D(&desc, nullptr, &input) != S_OK) {
        BOOST_LOG(debug) << "NvEnc: couldn't create input texture " << surface << " with bind flags " << desc.BindFlags;
        return false;
      }
      NV_ENC_REGISTER_RESOURCE register_resource = {NV_ENC_REGISTER_RESOURCE_VER};
      register_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
      register_resource.width = encoder_params.width;
      register_resource.height = encoder_params.height;
      register_resource.resourceToRegister = input.GetInterfacePtr();
      register_resource.bufferFormat = encoder_params.buffer_format;
      register_resource.bufferUsage = NV_ENC_INPUT_IMAGE;

      if (nvenc_failed(nvenc->nvEncRegisterResource(encoder, &register_resource))) {
        BOOST_LOG(debug) << "NvEnc: NvEncRegisterResource() failed: " << last_nvenc_error_string;
        return false;
      }

      registration = register_resource.registeredResource;
      return true;
    };
    for (unsigned surface = 0; surface < count; ++surface) {
      if (register_surface(surface)) {
        continue;
      }
      auto failed_desc = desc;
      if (input_surfaces[surface]) {
        input_surfaces[surface]->GetDesc(&failed_desc);
      }
      if (failed_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) {
        // The failed surface has no registration. Leave previously registered inputs owned
        // by NVENC and use raster-compatible allocations for this and subsequent surfaces.
        input_surfaces[surface] = nullptr;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        BOOST_LOG(info) << "NvEnc: using regular P010 render targets after unordered input setup failed";
        if (register_surface(surface)) {
          continue;
        }
      }
      BOOST_LOG(error) << "NvEnc: couldn't allocate or register input texture " << surface;
      return false;
    }

    return true;
  }

  NV_ENC_REGISTERED_PTR nvenc_d3d11_native::prepare_input(unsigned) {
    const auto input = registered_input_buffers[converted_input];
    // A repeat can reuse the previous surface once its picture has retired and unmapped.
    return release_input_registration(input) ? input : nullptr;
  }

#ifdef SUNSHINE_TESTS
  std::optional<unsigned> nvenc_d3d11_native::input_surface_for_picture_for_test(unsigned slot) const noexcept {
    const auto input = mapped_input_registration_for_test(slot);
    if (input) {
      const auto found = std::find(registered_input_buffers.begin(), registered_input_buffers.end(), input);
      if (found != registered_input_buffers.end()) {
        return static_cast<unsigned>(found - registered_input_buffers.begin());
      }
    }
    return std::nullopt;
  }
#endif

}  // namespace nvenc

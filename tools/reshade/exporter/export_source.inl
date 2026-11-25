// SPDX-License-Identifier: GPL-3.0-only
// Function-only part at the end of exporter.cpp; original anonymous namespace.
// The export source of a Present: the native renderer's packed SBS output.
namespace SUNSHINE_EXPORTER_INTERNAL {
  inline bool publisher_t::describe(api::effect_runtime *runtime, runtime_t &proof, source_t &source) {
    std::uint32_t width = 0, height = 0;
    runtime->get_screenshot_width_and_height(&width, &height);
    if (width != proof.width || height != proof.height) {
      return false;
    }
    source.resource = proof.native_output;
    if (!source.resource.handle) {
      return false;
    }
    source.width = width;
    source.height = height;
    source.color = proof.color;
    source.window = static_cast<HWND>(runtime->get_hwnd());
    // ReShade's aggregate-return getters are not MSVC/MinGW ABI compatible. The build
    // adapts only get_resource_from_view; native COM supplies the other descriptions,
    // including MinGW's explicit aggregate-return wrapper for D3D12 GetDesc.
    if (runtime->get_device()->get_api() == api::device_api::d3d11) {
      D3D11_TEXTURE2D_DESC desc {};
      reinterpret_cast<ID3D11Texture2D *>(source.resource.handle)->GetDesc(&desc);
      if (desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1) {
        return false;
      }
      source.format = typed_format(desc.Format);
      if (!proof.adapter_known) {
        com_ptr<IDXGIDevice> device;
        com_ptr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC adapter_desc {};
        auto *native = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
        if (FAILED(native->QueryInterface(IID_PPV_ARGS(device.put()))) || FAILED(device->GetAdapter(adapter.put())) || FAILED(adapter->GetDesc(&adapter_desc))) {
          return false;
        }
        std::memcpy(&proof.adapter_luid, &adapter_desc.AdapterLuid, sizeof(proof.adapter_luid));
        proof.adapter_known = true;
      }
    } else {
      const auto desc = reinterpret_cast<ID3D12Resource *>(source.resource.handle)->GetDesc();
      if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) {
        return false;
      }
      source.format = typed_format(desc.Format);
      if (!proof.adapter_known) {
        auto *native = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native());
        const LUID luid = native->GetAdapterLuid();
        std::memcpy(&proof.adapter_luid, &luid, sizeof(proof.adapter_luid));
        proof.adapter_known = true;
      }
    }
    source.adapter = proof.adapter_luid;
    return wire::supported_format(static_cast<std::uint32_t>(source.format), source.color.output);
  }

} // namespace

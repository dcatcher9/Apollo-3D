// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_v1_resource_state.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
  using namespace sunshine_streamline::v1_resource_state;
  using proof = sunshine_scene_depth::state_proof;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

  struct resource_stub final : ID3D12Resource {
    ULONG references{1};
    unsigned descriptions{}, properties{};
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
      *out = nullptr;
      if (iid != __uuidof(ID3D12Resource) && iid != __uuidof(IUnknown)) return E_NOINTERFACE;
      *out = static_cast<ID3D12Resource *>(this); AddRef(); return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
    ULONG STDMETHODCALLTYPE Release() override { return --references; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID key, UINT *bytes, void *data) override {
      ++properties;
      if (key != property_id || *bytes != sizeof(std::uint32_t)) return E_INVALIDARG;
      const std::uint32_t value = present | texture_read | storage_read;
      std::memcpy(data, &value, sizeof(value)); *bytes = sizeof(value); return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void **) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE Map(UINT, const D3D12_RANGE *, void **) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE *) override {}
    D3D12_RESOURCE_DESC description() {
      ++descriptions;
      D3D12_RESOURCE_DESC value{};
      value.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; value.Width = 2228; value.Height = 1256;
      value.MipLevels = value.DepthOrArraySize = value.SampleDesc.Count = 1; value.Format = DXGI_FORMAT_R32_FLOAT;
      return value;
    }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
    D3D12_RESOURCE_DESC *STDMETHODCALLTYPE GetDesc(D3D12_RESOURCE_DESC *out) override { *out = description(); return out; }
#else
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override { return description(); }
#endif
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return 0; }
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT, const D3D12_BOX *, const void *, UINT, UINT) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void *, UINT, UINT, UINT, const D3D12_BOX *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS *) override { return E_NOTIMPL; }
  };
  // This object has only the three IUnknown slots. Any GetDesc/GetPrivateData
  // on the original pointer instead of the returned resource is invalid.
  struct unknown_stub final : IUnknown {
    resource_stub backing;
    unsigned queries{};
    bool refuse{};
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
      ++queries; *out = nullptr;
      if (refuse || iid != __uuidof(ID3D12Resource)) return E_NOINTERFACE;
      *out = static_cast<ID3D12Resource *>(&backing); backing.AddRef(); return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
  };

  void mapping() {
    struct pair { std::uint32_t chi, native; };
    constexpr std::array<pair, 11> examples{{
      {present, D3D12_RESOURCE_STATE_COMMON}, {general, D3D12_RESOURCE_STATE_COMMON},
      {texture_read, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE},
      {storage_read, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
      {texture_read | storage_read, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
      {storage_read | storage_write, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
      {texture_read | storage_read | storage_write, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
      {color_write, D3D12_RESOURCE_STATE_RENDER_TARGET}, {depth_write, D3D12_RESOURCE_STATE_DEPTH_WRITE},
      {depth_read | copy_source, D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_COPY_SOURCE},
      {copy_destination, D3D12_RESOURCE_STATE_COPY_DEST}
    }};
    for (const auto &value : examples) for (const auto marker : {0u, std::uint32_t(present)}) {
      std::uint32_t native{};
      require(decode(value.chi | marker, native) && native == value.native,
        "Pinned chi state did not match the upstream native conversion");
    }
    for (const std::uint32_t invalid : {0u, std::uint32_t(storage_write), 1u << 1, 1u << 8,
        1u << 14, 1u << 16, 1u << 19, 1u << 31, UINT32_MAX,
        std::uint32_t(depth_write | depth_read), std::uint32_t(copy_destination | copy_source),
        std::uint32_t(color_write | texture_read), std::uint32_t(storage_read | storage_write | copy_source)}) {
      std::uint32_t native{};
      require(!decode(invalid, native) && native == UINT32_MAX, "Unsupported/contradictory chi state was accepted");
    }
  }

  void property_contract() {
    std::uint32_t native{};
    require(decode_property(S_OK, 4, present, native) && native == 0, "Known COMMON was confused with missing state");
    require(!decode_property(E_FAIL, 4, present, native) && native == UINT32_MAX, "Failed property read was accepted");
    require(!decode_property(S_FALSE, 4, present, native), "A non-success property result was accepted");
    for (const UINT bytes : {0u, 1u, 3u, 5u, 8u})
      require(!decode_property(S_OK, bytes, present, native), "Wrong property ABI size was accepted");
    require(!decode_property(S_OK, 4, 0, native), "Provider eUnknown became known COMMON");
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = 2228; desc.Height = 1256;
    desc.MipLevels = desc.DepthOrArraySize = desc.SampleDesc.Count = 1;
    for (const auto format : {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS}) {
      desc.Format = format;
      require(supported_resource(desc), "A supported depth-plane texture was rejected");
    }
    for (const auto format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN}) {
      desc.Format = format;
      require(!supported_resource(desc), "A color or unknown format used depth state");
    }
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    auto changed = desc; changed.MipLevels = 2;
    require(!supported_resource(changed), "Multiple mips used whole-resource state");
    changed = desc; changed.DepthOrArraySize = 2;
    require(!supported_resource(changed), "Texture array used whole-resource state");
    changed = desc; changed.SampleDesc.Count = 4;
    require(!supported_resource(changed), "MSAA used unsupported provider state");
    changed = desc; changed.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    require(!supported_resource(changed), "Buffer was admitted as a depth texture");
  }

  void precedence_and_fallback() {
    unsigned queries{};
    const auto query = [&](std::uint32_t &native) {
      ++queries;
      return decode_property(S_OK, 4, present | texture_read | storage_read, native);
    };
    auto value = resolve_with(D3D12_RESOURCE_STATE_COPY_SOURCE, true, query);
    require(!queries && value.native == D3D12_RESOURCE_STATE_COPY_SOURCE && value.proof == proof::declared,
      "Explicit V1 native state did not take precedence");
    value = resolve_with(0, false, query);
    require(!queries && value.native == 0 && value.proof == proof::declared,
      "Modern/unsupported ABI queried the private V1 property");
    value = resolve_with(UINT32_MAX, true, query);
    require(!queries && value.proof == proof::unavailable, "Missing metadata became a provider state query");
    value = resolve_with(0, true, query);
    require(queries == 1 && value.native == 0xc0 && value.proof == proof::declared,
      "Omitted V1 state did not use the provider shader-read state");
    value = resolve_with(0, true, [](std::uint32_t &native) { return decode_property(S_OK, 4, present, native); });
    require(value.native == 0 && value.proof == proof::declared, "Provider COMMON lost its explicit proof");
    for (const auto invalid : {0u, UINT32_MAX}) {
      value = resolve_with(0, true, [invalid](std::uint32_t &native) { return decode_property(S_OK, 4, invalid, native); });
      require(value.native == 0 && value.proof == proof::observed_nonzero,
        "Unavailable provider state did not retain the observed nonzero fallback");
    }
    value = resolve(0, true, nullptr);
    require(value.native == 0 && value.proof == proof::observed_nonzero,
      "Unavailable/unsupported common-module gate changed omitted V1 state to declared COMMON");
    // A modern or explicitly tagged state must never dereference this pointer.
    auto *invalid_resource = reinterpret_cast<ID3D12Resource *>(std::uintptr_t(1));
    require(resolve(0, false, invalid_resource).proof == proof::declared &&
      resolve(D3D12_RESOURCE_STATE_COPY_SOURCE, true, invalid_resource).native == D3D12_RESOURCE_STATE_COPY_SOURCE,
      "Unnecessary COM lookup changed declared state handling");
  }

  void exact_interface() {
    unknown_stub source;
    read_details details;
    auto value = resolve(0, true, &source, &details);
    require(value.native == 0xc0 && value.proof == proof::declared && source.queries == 1 &&
      source.backing.descriptions == 1 && source.backing.properties == 1 && source.backing.references == 1,
      "Provider lookup did not use/release the exact QI-returned resource interface");
    require(details.result == S_OK && details.bytes == 4 && details.encoded == (present | texture_read | storage_read) &&
      std::strcmp(details.stage, "ready") == 0, "Successful state diagnostics lost the actual property result");
    source.refuse = true;
    details = {};
    value = resolve(0, true, &source, &details);
    require(value.proof == proof::observed_nonzero && source.queries == 2 &&
      source.backing.descriptions == 1 && source.backing.properties == 1,
      "Failed Resource QI touched a larger interface or created state evidence");
    require(details.result == E_NOINTERFACE && std::strcmp(details.stage, "resource_interface") == 0,
      "Failed state diagnostics did not preserve the exact QI rejection");
    require(resolve(0, false, &source).proof == proof::declared && source.queries == 2 &&
      resolve(D3D12_RESOURCE_STATE_COPY_SOURCE, true, &source).native == D3D12_RESOURCE_STATE_COPY_SOURCE && source.queries == 2,
      "Modern/explicit state unnecessarily queried the resource interface");
  }
}

int main() {
  try {
    mapping(); property_contract(); precedence_and_fallback(); exact_interface();
    std::puts("PASS Streamline 1.1.1 resource state: exact chi mapping, property ABI, packed depth scope, explicit-state precedence and observed fallback");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Streamline 1.1.1 resource state: %s\n", error.what());
    return 1;
  }
}

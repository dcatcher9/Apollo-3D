// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <d3d12.h>
#include <atomic>
#include <cstdint>
#include <mutex>

// One object identity shared by API source adapters and ReShade preservation.
// These cookies identify live objects, never their current GPU contents.
namespace sunshine_native_identity {
  inline constexpr GUID resource_guid{0x527b8421, 0x71a4, 0x4d16, {0xb2, 0x75, 0xa6, 0x46, 0x83, 0x91, 0x40, 0x33}};
  inline constexpr GUID device_guid{0xa95f795a, 0x32bc, 0x4e7b, {0xa9, 0xb6, 0x83, 0xc6, 0xc8, 0xa4, 0x21, 0xfe}};
  // Resource/device cookies and ReShade backup/fallback IDs share one domain.
  inline std::uint64_t allocate_identity();
  namespace detail {
    inline std::mutex creation_mutex;
    inline std::atomic<std::uint64_t> serial{1};
    struct restore_error { DWORD value{GetLastError()}; ~restore_error() { SetLastError(value); } };
    inline std::uint64_t read(ID3D12Object *object, REFGUID key) {
      std::uint64_t value{}; UINT size = sizeof(value);
      return object && SUCCEEDED(object->GetPrivateData(key, &size, &value)) && size == sizeof(value) ? value : 0;
    }
    // The caller holds a validated, live interface. Creation is serialized
    // across all users; proxy/native interfaces sharing private data share IDs.
    inline std::uint64_t cookie(ID3D12Object *object, REFGUID key, bool create) {
      const restore_error error;
      if (!object) return 0;
      if (!create) return read(object, key);
      std::lock_guard lock(creation_mutex);
      if (const auto existing = read(object, key)) return existing;
      const auto value = allocate_identity();
      return SUCCEEDED(object->SetPrivateData(key, sizeof(value), &value)) ? value : 0;
    }
  }
  inline std::uint64_t allocate_identity() {
    return detail::serial.fetch_add(1, std::memory_order_relaxed) + 1;
  }
  inline std::uint64_t resource_cookie(ID3D12Object *object, bool create = false) {
    return detail::cookie(object, resource_guid, create);
  }
  inline std::uint64_t device_cookie(ID3D12Device *device, bool create = false) {
    return detail::cookie(device, device_guid, create);
  }
  struct resource_identity {
    std::uint64_t native{}, resource{}, device{};
    explicit operator bool() const { return native && resource && device; }
  };
  // Incoming value is a live COM object, possibly a base interface/proxy.
  // Never read resource/device methods before exact QueryInterface succeeds.
  // Result is value-only; callers retaining it must separately retain lifetime.
  inline resource_identity identify_resource(std::uint64_t native, bool create = true) {
    const detail::restore_error error;
    resource_identity out;
    if (!native) return out;
    ID3D12Resource *resource{};
    if (FAILED(reinterpret_cast<IUnknown *>(native)->QueryInterface(IID_ID3D12Resource,
        reinterpret_cast<void **>(&resource))) || !resource) return out;
    ID3D12Device *device{};
    if (SUCCEEDED(resource->GetDevice(IID_ID3D12Device, reinterpret_cast<void **>(&device))) && device) {
      out.native = reinterpret_cast<std::uint64_t>(resource);
      out.resource = resource_cookie(resource, create);
      out.device = device_cookie(device, create);
      device->Release();
    }
    resource->Release();
    return out ? out : resource_identity{};
  }
}

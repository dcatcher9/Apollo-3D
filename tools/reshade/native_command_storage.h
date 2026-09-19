// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <d3d12.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace sunshine_native_command {
  struct recording_lifetime {
    // Reset publishes a new recording identity. Native object destruction
    // publishes zero; capture-slot retirement still waits for its GPU fences.
    std::atomic<std::uint64_t> cookie{};
  };

  // One State instantiation per add-on. The two private GUIDs belong to this
  // storage contract; callers must not reuse them for another payload type.
  // The caller guarantees a live ID3D12Object and serializes creation, payload
  // access and Reset. Reference counting alone does not synchronize State.
  template<class State>
  class storage {
    inline static constexpr GUID property_id{0x952b50a7, 0x6c20, 0x43cb,
      {0x9b, 0xa7, 0x12, 0xe3, 0x9e, 0xbb, 0x02, 0x73}};
    inline static constexpr GUID interface_id{0x7c9f1d16, 0xa445, 0x4307,
      {0x8c, 0x22, 0x6a, 0xf0, 0x8d, 0x20, 0xf5, 0xc9}};
    inline static std::atomic<std::uint32_t> live_{};

    struct owner final : IUnknown {
      State payload{};
      const std::shared_ptr<recording_lifetime> lifetime{std::make_shared<recording_lifetime>()};
      std::atomic<ULONG> references{1};

      owner() { live_.fetch_add(1, std::memory_order_relaxed); }
      ~owner() {
        // No native-object reference, callback, mutex or GPU work here. This
        // may run during the driver's private-data teardown on any thread.
        lifetime->cookie.store(0, std::memory_order_release);
        live_.fetch_sub(1, std::memory_order_relaxed);
      }
      HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!IsEqualIID(iid, __uuidof(IUnknown)) && !IsEqualIID(iid, interface_id)) return E_NOINTERFACE;
        *out = static_cast<IUnknown *>(this);
        AddRef();
        return S_OK;
      }
      ULONG STDMETHODCALLTYPE AddRef() override { return references.fetch_add(1, std::memory_order_relaxed) + 1; }
      ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = references.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (!remaining) delete this;
        return remaining;
      }
    };

    static bool pin_module() noexcept {
      HMODULE module{};
      return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(&pin_module), &module) != FALSE;
    }

  public:
    class handle {
      friend class storage;
      owner *value_{};
      explicit handle(owner *value) noexcept : value_(value) {}

    public:
      handle() = default;
      handle(const handle &) = delete;
      handle &operator=(const handle &) = delete;
      handle(handle &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
      handle &operator=(handle &&other) noexcept {
        if (this != &other) {
          if (value_) value_->Release();
          value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
      }
      ~handle() { if (value_) value_->Release(); }
      explicit operator bool() const noexcept { return value_ != nullptr; }
      State *operator->() const noexcept { return &value_->payload; }
      State &operator*() const noexcept { return value_->payload; }
      std::shared_ptr<recording_lifetime> life() const noexcept {
        return value_ ? value_->lifetime : std::shared_ptr<recording_lifetime>{};
      }
    };

    static handle acquire(ID3D12Object *object, bool create) noexcept {
      if (!object) return {};
      IUnknown *stored{};
      UINT bytes = sizeof(stored);
      const HRESULT result = object->GetPrivateData(property_id, &bytes, &stored);
      if (SUCCEEDED(result)) {
        if (bytes != sizeof(stored) || !stored) return {};
        IUnknown *verified{};
        const HRESULT queried = stored->QueryInterface(interface_id, reinterpret_cast<void **>(&verified));
        // GetPrivateData adds a reference for SetPrivateDataInterface values;
        // the handle adopts the separate reference returned by successful QI.
        stored->Release();
        if (FAILED(queried) || !verified) return {};
        return handle(static_cast<owner *>(verified));
      }
      // Only absence allows installation. Never overwrite malformed data or
      // reinterpret a driver error as proof that no owner exists.
      if (!create || result != DXGI_ERROR_NOT_FOUND) return {};
      static const bool pinned = pin_module();
      if (!pinned) return {};
      try {
        handle value(new owner);
        if (FAILED(object->SetPrivateDataInterface(property_id, value.value_))) return {};
        return value; // The native object owns its own reference after Set.
      } catch (...) {
        return {};
      }
    }

    static std::uint32_t live_count() noexcept { return live_.load(std::memory_order_relaxed); }
  };
}

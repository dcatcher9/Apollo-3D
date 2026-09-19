// SPDX-License-Identifier: GPL-3.0-only
#include "reshade_bridge.h"

#include "src/logging.h"
#include "src/reshade_bridge_protocol.h"

#include <array>
#include <atomic>
#include <cstring>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <string>
#include <utility>
#include <wrl/client.h>

namespace platf::reshade_bridge {
  namespace {
    namespace wire = ::reshade_bridge;
    using Microsoft::WRL::ComPtr;

    class handle_t {
    public:
      explicit handle_t(HANDLE value = nullptr):
          value_(value) {}

      ~handle_t() {
        reset();
      }

      handle_t(const handle_t &) = delete;
      handle_t &operator=(const handle_t &) = delete;

      HANDLE get() const {
        return value_;
      }

      void reset(HANDLE value = nullptr) {
        if (value_) {
          CloseHandle(value_);
        }
        value_ = value;
      }

    private:
      HANDLE value_;
    };

    LONG read32(std::uint32_t &value) {
      return InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&value), 0, 0);
    }

    std::uint64_t read64(std::uint64_t &value) {
      return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&value), 0, 0));
    }

    bool claim(wire::slot_t &slot, std::uint64_t generation, wire::slot_state from, wire::slot_state to) {
      const auto expected = wire::slot_control(generation, from);
      return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, to)), static_cast<LONG64>(expected))) == expected;
    }

    bool snapshot(wire::shared_state_t &shared, wire::metadata_t &metadata) {
      const auto before = read32(shared.metadata_sequence);
      if (before & 1) {
        return false;
      }
      std::memcpy(&metadata, &shared.metadata, sizeof(metadata));
      MemoryBarrier();
      return before == read32(shared.metadata_sequence);
    }

    std::uint64_t creation_time(HANDLE process) {
      FILETIME created {}, exited {}, kernel {}, user {};
      if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
        return 0;
      }
      return (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
    }

    std::uint64_t new_nonce() {
      static std::atomic<std::uint64_t> counter {[] {
        LARGE_INTEGER now {};
        QueryPerformanceCounter(&now);
        return static_cast<std::uint64_t>(now.QuadPart) ^ (static_cast<std::uint64_t>(GetCurrentProcessId()) << 32);
      }()};
      const auto value = counter.fetch_add(1);
      return value ? value : counter.fetch_add(1);
    }

    std::chrono::steady_clock::time_point frame_time(std::uint64_t qpc) {
      LARGE_INTEGER now {}, frequency {};
      const auto steady_now = std::chrono::steady_clock::now();
      if (!QueryPerformanceCounter(&now) || !QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0 || qpc == 0 || qpc > static_cast<std::uint64_t>(now.QuadPart)) {
        return steady_now;
      }
      const auto elapsed = std::chrono::duration<double>(
        static_cast<double>(static_cast<std::uint64_t>(now.QuadPart) - qpc) / static_cast<double>(frequency.QuadPart)
      );
      return steady_now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed);
    }
  }  // namespace

  class receiver_t::impl_t {
  public:
    impl_t(ID3D11Device *device, ID3D11DeviceContext *context, observer_t observe):
        device_(device),
        context_(context),
        observe_(std::move(observe)) {
      ComPtr<IDXGIDevice> dxgi_device;
      ComPtr<IDXGIAdapter> adapter;
      DXGI_ADAPTER_DESC desc {};
      if (device && context && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5_))) && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgi_device))) && SUCCEEDED(dxgi_device->GetAdapter(&adapter)) && SUCCEEDED(adapter->GetDesc(&desc))) {
        std::memcpy(&adapter_luid_, &desc.AdapterLuid, sizeof(adapter_luid_));
        supported_ = true;
      } else {
        BOOST_LOG(warning) << "ReShade SBS requires D3D11 shared NT textures and shared fences on the capture adapter.";
      }
    }

    ~impl_t() {
      detach();
    }

    std::optional<frame_t> poll(RECT source, int width, int height) {
      if (!supported_ || width <= 0 || height <= 0 || source.right <= source.left || source.bottom <= source.top) {
        return std::nullopt;
      }
      if (!complete_copy()) {
        return std::nullopt;
      }
      const auto observation = observe_();
      const foreground_window::rect_t expected {source.left, source.top, source.right, source.bottom};
      if (observation.status != foreground_window::status_e::ok || observation.client_screen_rect != expected || observation.process_id == 0 || observation.window == 0) {
        detach();
        return std::nullopt;
      }
      if (pid_ != observation.process_id || window_ != observation.window) {
        detach();
        pid_ = observation.process_id;
        window_ = observation.window;
      }
      if (!shared_) {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_attach_) {
          return std::nullopt;
        }
        next_attach_ = now + std::chrono::milliseconds(250);
        if (!attach()) {
          return std::nullopt;
        }
      }
      if (WaitForSingleObject(process_.get(), 0) != WAIT_TIMEOUT) {
        detach();
        return std::nullopt;
      }

      wire::metadata_t metadata;
      if (!snapshot(*shared_, metadata)) {
        // A resize/disable is in progress. Never present a retained frame across a generation.
        cached_.reset();
        return std::nullopt;
      }
      if (!identity_matches(metadata)) {
        cached_.reset();
        return std::nullopt;
      }
      if (!nonce_) {
        nonce_ = new_nonce();
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared_->consumer_nonce), static_cast<LONG64>(nonce_));
        return std::nullopt;
      }
      if (read64(shared_->consumer_nonce) != nonce_) {
        // A replacement converter owns the connection now. Do not fight it by writing again.
        cached_.reset();
        return std::nullopt;
      }
      // Fullscreen may scale the game's raster to a different desktop extent. Window coverage
      // proves ownership above; the requested output independently proves exact authored eyes.
      // Local AR requests twice the display width, retaining its display-size requirement.
      if (metadata.accepted_consumer_nonce != nonce_ || !wire::matches_output(metadata, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)) || metadata.adapter_luid != adapter_luid_) {
        cached_.reset();
        return std::nullopt;
      }
      if (metadata.generation != metadata_.generation) {
        reset_resources();
        if (!open_resources(metadata)) {
          return std::nullopt;
        }
        metadata_ = metadata;
        BOOST_LOG(info) << "ReShade SBS connected: process " << pid_ << ", " << width << 'x' << height
                        << ", " << (metadata.color_transfer == wire::transfer::scrgb ? "linear scRGB HDR" : "sRGB SDR")
                        << ", generation " << metadata.generation << '.';
      } else if (std::memcmp(&metadata, &metadata_, sizeof(metadata)) != 0) {
        // Resource identity, color and dimensions are immutable within an acknowledged generation.
        cached_.reset();
        return std::nullopt;
      }
      if (pending_slot_ >= 0) {
        return cached_;
      }

      const auto completed = ready_fence_->GetCompletedValue();
      if (completed == UINT64_MAX) {
        reset_resources();
        return std::nullopt;
      }
      int selected = -1;
      std::uint64_t newest = cached_ ? cached_->sequence : 0;
      for (std::uint32_t i = 0; i < wire::slot_count; ++i) {
        auto &slot = shared_->slots[i];
        if (read64(slot.control) == wire::slot_control(metadata_.generation, wire::slot_state::ready)) {
          const auto sequence = read64(slot.sequence);
          if (sequence > newest && sequence <= completed) {
            selected = static_cast<int>(i);
            newest = sequence;
          }
        }
      }
      if (selected < 0) {
        return cached_;
      }
      auto &slot = shared_->slots[selected];
      if (!claim(slot, metadata_.generation, wire::slot_state::ready, wire::slot_state::reading)) {
        return cached_;
      }
      const auto sequence = read64(slot.sequence);
      const auto qpc = read64(slot.qpc);
      wire::metadata_t current;
      if (!snapshot(*shared_, current) || current.generation != metadata_.generation || current.accepted_consumer_nonce != nonce_ || read64(shared_->consumer_nonce) != nonce_) {
        cached_.reset();
        // Retiring generations own their old state; a new nonce creates fresh resources.
        return std::nullopt;
      }
      if (sequence == 0 || sequence > ready_fence_->GetCompletedValue() || (cached_ && sequence <= cached_->sequence)) {
        claim(slot, metadata_.generation, wire::slot_state::reading, wire::slot_state::ready);
        return cached_;
      }
      context_->CopyResource(private_texture_.Get(), textures_[selected].Get());
      context_->End(copy_complete_.Get());
      // Flush submits bounded work; neither producer nor consumer waits for the GPU.
      context_->Flush();
      pending_slot_ = selected;
      pending_sequence_ = sequence;
      pending_generation_ = metadata_.generation;
      cached_ = frame_t {
        .texture = private_texture_.Get(),
        .view = private_view_.Get(),
        .linear = metadata_.color_transfer == wire::transfer::scrgb,
        .timestamp = frame_time(qpc),
        .sequence = sequence,
        .producer_process_id = metadata_.producer_pid,
        .producer_creation_time = metadata_.producer_creation_time,
        .resource_generation = metadata_.generation,
      };
      return cached_;
    }

  private:
    bool identity_matches(const wire::metadata_t &metadata) const {
      return metadata.signature == wire::magic && metadata.protocol_version == wire::version &&
             metadata.metadata_bytes == sizeof(wire::metadata_t) && metadata.producer_pid == pid_ &&
             metadata.producer_creation_time == process_creation_ && metadata.window == window_;
    }

    bool attach() {
      process_.reset(OpenProcess(PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, pid_));
      if (!process_.get() || !(process_creation_ = creation_time(process_.get()))) {
        return false;
      }
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(pid_);
      mapping_.reset(OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name.c_str()));
      if (!mapping_.get()) {
        return false;
      }
      shared_ = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping_.get(), FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(wire::shared_state_t)));
      if (!shared_) {
        return false;
      }
      if (shared_->shared_bytes != sizeof(wire::shared_state_t)) {
        UnmapViewOfFile(shared_);
        shared_ = nullptr;
        return false;
      }
      return true;
    }

    bool open_resources(const wire::metadata_t &metadata) {
      const auto duplicate = [this](std::uint64_t source) -> HANDLE {
        HANDLE copy = nullptr;
        if (!DuplicateHandle(process_.get(), reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(source)), GetCurrentProcess(), &copy, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
          return nullptr;
        }
        return copy;
      };
      handle_t fence(duplicate(metadata.ready_fence_handle));
      if (!fence.get() || FAILED(device5_->OpenSharedFence(fence.get(), IID_PPV_ARGS(&ready_fence_)))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      for (std::uint32_t i = 0; i < wire::slot_count; ++i) {
        handle_t texture(duplicate(metadata.texture_handles[i]));
        if (!texture.get() || FAILED(device5_->OpenSharedResource1(texture.get(), IID_PPV_ARGS(&textures_[i])))) {
          reset_resources();
          return false;
        }
        textures_[i]->GetDesc(&desc);
        if (desc.Width != metadata.packed_width || desc.Height != metadata.packed_height || desc.Format != static_cast<DXGI_FORMAT>(metadata.dxgi_format) || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0) {
          reset_resources();
          return false;
        }
      }
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.CPUAccessFlags = desc.MiscFlags = 0;
      const D3D11_QUERY_DESC query {D3D11_QUERY_EVENT, 0};
      if (FAILED(device_->CreateTexture2D(&desc, nullptr, &private_texture_)) || FAILED(device_->CreateShaderResourceView(private_texture_.Get(), nullptr, &private_view_)) || FAILED(device_->CreateQuery(&query, &copy_complete_))) {
        reset_resources();
        return false;
      }
      return true;
    }

    bool complete_copy() {
      if (pending_slot_ < 0 || !shared_ || !copy_complete_) {
        return true;
      }
      const auto result = context_->GetData(copy_complete_.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
      if (FAILED(result)) {
        BOOST_LOG(warning) << "ReShade SBS receiver GPU copy failed; this device must be recreated (HRESULT " << result << ").";
        reset_resources();
        supported_ = false;
        return false;
      }
      if (result != S_OK) {
        return true;
      }
      wire::metadata_t current;
      auto &slot = shared_->slots[pending_slot_];
      if (snapshot(*shared_, current) && current.generation == pending_generation_ && current.accepted_consumer_nonce == nonce_ && read64(slot.sequence) == pending_sequence_) {
        claim(slot, pending_generation_, wire::slot_state::reading, wire::slot_state::free);
      }
      pending_slot_ = -1;
      return true;
    }

    void reset_resources() {
      cached_.reset();
      // An in-flight read is abandoned, never unlocked prematurely. A new generation has new
      // resources; the D3D runtime retains submitted resource references through completion.
      pending_slot_ = -1;
      copy_complete_.Reset();
      private_view_.Reset();
      private_texture_.Reset();
      for (auto &texture : textures_) {
        texture.Reset();
      }
      ready_fence_.Reset();
      metadata_ = {};
    }

    void detach() {
      reset_resources();
      if (shared_) {
        UnmapViewOfFile(shared_);
        shared_ = nullptr;
      }
      mapping_.reset();
      process_.reset();
      pid_ = 0;
      window_ = nonce_ = process_creation_ = 0;
      next_attach_ = {};
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11Device5> device5_;
    ComPtr<ID3D11DeviceContext> context_;
    observer_t observe_;
    bool supported_ = false;
    std::uint64_t adapter_luid_ = 0;
    DWORD pid_ = 0;
    std::uint64_t window_ = 0, process_creation_ = 0, nonce_ = 0;
    handle_t process_, mapping_;
    wire::shared_state_t *shared_ = nullptr;
    wire::metadata_t metadata_;
    std::array<ComPtr<ID3D11Texture2D>, wire::slot_count> textures_;
    ComPtr<ID3D11Fence> ready_fence_;
    ComPtr<ID3D11Texture2D> private_texture_;
    ComPtr<ID3D11ShaderResourceView> private_view_;
    ComPtr<ID3D11Query> copy_complete_;
    std::optional<frame_t> cached_;
    int pending_slot_ = -1;
    std::uint64_t pending_sequence_ = 0, pending_generation_ = 0;
    std::chrono::steady_clock::time_point next_attach_ {};
  };

  receiver_t::receiver_t(ID3D11Device *device, ID3D11DeviceContext *context, observer_t observe):
      impl_(std::make_unique<impl_t>(device, context, std::move(observe))) {}

  receiver_t::~receiver_t() = default;

  std::optional<frame_t> receiver_t::poll(RECT source_rect, int output_width, int output_height) {
    return impl_->poll(source_rect, output_width, output_height);
  }
}  // namespace platf::reshade_bridge

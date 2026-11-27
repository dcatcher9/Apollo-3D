/**
 * @file src/platform/windows/local_shutter_frame_ring.h
 * @brief The shutter monitor's two shared stereo packets between capture and the shutter thread,
 *        and the hand-off from one producer's packets to its replacement's.
 */
#pragma once

// standard includes
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

// platform includes
#include <d3d11_4.h>
#include <wrl/client.h>

// local includes
#include "display.h"
#include "local_rgb_converter.h"

namespace platf::dxgi {
  /** Internals of run_local_presenter(), for local_presenter.cpp only. */
  namespace local_presenter {
    // Two GPU-owned packets bound publication: capture never overwrites the stereo pair whose
    // left/right eyes are being scanned out. Each device has its own immediate context; keyed
    // mutexes carry GPU completion between them without a CPU readback or context-thread race.
    //
    // A keyed mutex changes hands as soon as the producer has submitted its commands, not when
    // the GPU has executed them. A consumer that then reads the packet waits on the GPU for the
    // whole conversion behind it, and so does the eye it is about to present. The producer
    // therefore signals a fence after each packet, and the consumer takes a packet only once
    // that fence has completed; until then it keeps scanning out the pair it already owns.
    // One capture thread produces packets and one shutter thread consumes them. Slot state
    // passes CPU ownership without putting the shutter clock behind producer driver calls.
    class local_shutter_frame_ring_t {
    public:
      struct slot_t {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> producer_texture;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> producer_target;
        Microsoft::WRL::ComPtr<IDXGIKeyedMutex> producer_mutex;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> consumer_texture;
        Microsoft::WRL::ComPtr<IDXGIKeyedMutex> consumer_mutex;
        enum class state_e { free, producing, pending, current };
        std::atomic<state_e> state {state_e::free};
        // Published with pending. The producer cannot replace it until the slot is free.
        std::uint64_t completion = 0;
      };

      bool init(ID3D11Device *producer, ID3D11Device *consumer, UINT width, UINT height, DXGI_FORMAT format) {
        Microsoft::WRL::ComPtr<ID3D11Device1> consumer1;
        if (FAILED(consumer->QueryInterface(IID_PPV_ARGS(&consumer1)))) {
          return false;
        }
        Microsoft::WRL::ComPtr<ID3D11Device5> producer5;
        if (FAILED(producer->QueryInterface(IID_PPV_ARGS(&producer5))) ||
            FAILED(producer5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion_fence_)))) {
          return false;
        }
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = format;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        for (auto &slot : slots_) {
          if (FAILED(producer->CreateTexture2D(&desc, nullptr, &slot.producer_texture)) ||
              FAILED(producer->CreateRenderTargetView(slot.producer_texture.Get(), nullptr, &slot.producer_target)) ||
              FAILED(slot.producer_texture.As(&slot.producer_mutex))) {
            return false;
          }
          Microsoft::WRL::ComPtr<IDXGIResource1> resource;
          HANDLE shared_handle = nullptr;
          if (FAILED(slot.producer_texture.As(&resource)) ||
              FAILED(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &shared_handle))) {
            return false;
          }
          const auto opened = consumer1->OpenSharedResource1(shared_handle, IID_PPV_ARGS(&slot.consumer_texture));
          CloseHandle(shared_handle);
          if (FAILED(opened) || FAILED(slot.consumer_texture.As(&slot.consumer_mutex))) {
            return false;
          }
        }
        return true;
      }

      slot_t *acquire_producer() {
        if (failed()) {
          return nullptr;
        }
        for (auto &slot : slots_) {
          if (slot.state.load(std::memory_order_acquire) != slot_t::state_e::free) {
            continue;
          }
          const auto status = slot.producer_mutex->AcquireSync(0, 0);
          if (status == WAIT_TIMEOUT) {
            continue;
          }
          if (status != S_OK) {
            failed_ = true;
            return nullptr;
          }
          slot.state.store(slot_t::state_e::producing, std::memory_order_relaxed);
          return &slot;
        }
        return nullptr;
      }

      [[nodiscard]] bool has_published_pair() const {
        return completion_ != 0;
      }

      [[nodiscard]] local_rgb_converter_t::destination_e rgb_destination() const {
        using destination_e = local_rgb_converter_t::destination_e;
        return has_published_pair() ? destination_e::shutter_with_pair : destination_e::shutter;
      }

      bool finish_producer(slot_t &slot, bool publish_pair, ID3D11DeviceContext *producer_context) {
        if (publish_pair) {
          Microsoft::WRL::ComPtr<ID3D11DeviceContext4> producer_context4;
          if (FAILED(producer_context->QueryInterface(IID_PPV_ARGS(&producer_context4))) ||
              FAILED(producer_context4->Signal(completion_fence_.Get(), completion_ + 1))) {
            failed_ = true;
            return false;
          }
#ifdef SUNSHINE_TESTS
          if (before_flush_for_test) {
            before_flush_for_test();
          }
#endif
        }
        // The conversion can enqueue upstream work even when it reuses the published RGB.
        producer_context->Flush();
        if (FAILED(slot.producer_mutex->ReleaseSync(publish_pair ? 1 : 0))) {
          failed_ = true;
          return false;
        }
        if (publish_pair) {
          slot.completion = ++completion_;
        }
        slot.state.store(publish_pair ? slot_t::state_e::pending : slot_t::state_e::free, std::memory_order_release);
        return true;
      }

      ID3D11Texture2D *acquire_pair(bool advance = true) {
        // A removed device reports UINT64_MAX. It cannot authorize any packet.
        const auto completed = completion_fence_->GetCompletedValue();
        if (completed == std::numeric_limits<std::uint64_t>::max()) {
          failed_ = true;
        }
        if (failed()) {
          return nullptr;
        }
        if (!advance) {
          return current_ ? current_->consumer_texture.Get() : nullptr;
        }
        slot_t *newest = nullptr;
        for (auto &slot : slots_) {
          if (slot.state.load(std::memory_order_acquire) == slot_t::state_e::pending &&
              slot.completion <= completed && (!newest || slot.completion > newest->completion)) {
            newest = &slot;
          }
        }
        if (!newest) {
          return current_ ? current_->consumer_texture.Get() : nullptr;
        }
        const auto acquired = newest->consumer_mutex->AcquireSync(1, 0);
        if (acquired == WAIT_TIMEOUT) {
          return current_ ? current_->consumer_texture.Get() : nullptr;
        }
        if (acquired != S_OK) {
          failed_ = true;
          return nullptr;
        }
        if (current_) {
          release_consumer_slot(*current_);
        }
        current_ = newest;
        current_->state.store(slot_t::state_e::current, std::memory_order_relaxed);
        if (failed()) {
          return nullptr;
        }
        // Only startup can have two pending packets. Retire the older one before the producer
        // runs again; subsequent cycles always have one current and at most one pending packet.
        for (auto &slot : slots_) {
          if (slot.state.load(std::memory_order_acquire) == slot_t::state_e::pending &&
              slot.completion < current_->completion &&
              slot.consumer_mutex->AcquireSync(1, 0) == S_OK) {
            release_consumer_slot(slot);
          }
        }
        return failed() ? nullptr : current_->consumer_texture.Get();
      }

      [[nodiscard]] bool failed() const {
        return failed_.load(std::memory_order_acquire);
      }

#ifdef SUNSHINE_TESTS
      std::function<void()> before_flush_for_test;
#endif

      ~local_shutter_frame_ring_t() {
        if (current_) {
          current_->consumer_mutex->ReleaseSync(0);
        }
      }

    private:
      void release_consumer_slot(slot_t &slot) {
        if (FAILED(slot.consumer_mutex->ReleaseSync(0))) {
          failed_ = true;
          return;
        }
        slot.state.store(slot_t::state_e::free, std::memory_order_release);
      }

      std::array<slot_t, 2> slots_;
      Microsoft::WRL::ComPtr<ID3D11Fence> completion_fence_;
      std::uint64_t completion_ = 0;
      slot_t *current_ = nullptr;
      std::atomic<bool> failed_ = false;
    };

    struct local_shutter_source_t {
      local_shutter_frame_ring_t ring;
      std::shared_ptr<local_monitor_source_switch_t::request_t> request;
    };

    // Capture publishes the fully initialized producer. The shutter thread alone adopts its
    // first completed packet, at a whole-pair boundary. Capture retains source owners so the
    // shutter thread never has to retire a converter or destroy its last ring reference.
    class local_shutter_pair_handoff_t {
    public:
      explicit local_shutter_pair_handoff_t(std::shared_ptr<local_shutter_source_t> initial):
          current_(std::move(initial)) {}

      void publish(std::shared_ptr<local_shutter_source_t> candidate) {
        candidate_.store(std::move(candidate), std::memory_order_release);
      }

      template<class TryAdopt>
      ID3D11Texture2D *acquire_pair(TryAdopt try_adopt) {
        if (failed_) {
          return nullptr;
        }
        auto candidate = candidate_.load(std::memory_order_acquire);
        const bool changing_source = candidate && candidate != current_;
        auto *current_pair = current_->ring.acquire_pair(!changing_source);
        if (current_->ring.failed()) {
          failed_ = true;
          return nullptr;
        }
        using state_e = local_monitor_source_switch_t::state_e;
        if (!changing_source || !candidate->request || candidate->request->state.load(std::memory_order_acquire) != state_e::requested) {
          return current_pair;
        }

        auto *new_pair = candidate->ring.acquire_pair();
        if (candidate->ring.failed()) {
          auto expected = state_e::requested;
          candidate->request->state.compare_exchange_strong(expected, state_e::failed, std::memory_order_acq_rel);
          return current_pair;
        }
        if (new_pair && try_adopt(candidate->request)) {
          current_ = std::move(candidate);
          return new_pair;
        }
        return current_pair;
      }

      [[nodiscard]] const std::shared_ptr<local_monitor_source_switch_t::request_t> &current_request() const noexcept {
        return current_->request;
      }

      [[nodiscard]] bool failed() const noexcept {
        return failed_.load(std::memory_order_acquire);
      }

    private:
      std::shared_ptr<local_shutter_source_t> current_;
      std::atomic<std::shared_ptr<local_shutter_source_t>> candidate_;
      std::atomic<bool> failed_ = false;
    };
  }  // namespace local_presenter
}  // namespace platf::dxgi

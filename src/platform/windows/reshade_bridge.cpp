// SPDX-License-Identifier: GPL-3.0-only
#include "reshade_bridge.h"

#include "src/logging.h"
#include "src/reshade_bridge_protocol.h"

#include <array>
#include <atomic>
#include <cstring>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <mutex>
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

    bool snapshot(wire::shared_state_t &shared, wire::metadata_t &metadata, std::uint32_t *sequence = nullptr) {
      const auto before = read32(shared.metadata_sequence);
      if (before & 1) {
        return false;
      }
      std::memcpy(&metadata, &shared.metadata, sizeof(metadata));
      MemoryBarrier();
      if (sequence) {
        *sequence = static_cast<std::uint32_t>(before);
      }
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
      if (wait_) {
        disarm_wake();
        // A callback already running finishes here; none starts after the wait is cleared.
        WaitForThreadpoolWaitCallbacks(wait_, TRUE);
        CloseThreadpoolWait(wait_);
      }
    }

    std::optional<frame_t> poll(RECT source, int width, int height) {
      auto frame = poll_frame(source, width, height);
      if (frame) {
        // The caller may read the returned frame again: reads_recorded() must follow.
        held_reads_ended_ = false;
        arm_wake();
      }
      return frame;
    }

    std::optional<source_status_t> status(RECT source, int width, int height) {
      if (nonce_) {
        // A status observer is never the producer's consumer.
        detach();
      }
      if (!connect(source, width, height)) {
        return std::nullopt;
      }
      wire::metadata_t metadata;
      // Without a consumer the producer publishes its source (generation zero) while it renders
      // for the foreground window, and its bare identity once it stops.
      if (!snapshot(*shared_, metadata) || !identity_matches(metadata) || !wire::valid_source_metadata(metadata) || metadata.adapter_luid != adapter_luid_) {
        return std::nullopt;
      }
      const auto fit = wire::fit_output(metadata, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
      report_fit(metadata, fit, width, height);
      if (fit == wire::output_fit::aspect_mismatch) {
        return std::nullopt;
      }
      return source_status_t {metadata.producer_pid, metadata.producer_creation_time, metadata.generation};
    }

    void set_stream_pq(bool stream_pq) {
      const auto capabilities = wire::consumer_accepts_pq | (stream_pq ? wire::consumer_stream_pq : 0u);
      if (capabilities == capabilities_) {
        return;
      }
      capabilities_ = capabilities;
      if (nonce_) {
        // The producer honours capabilities only for the nonce they were written with.
        detach();
      }
    }

    void set_frame_wake(std::function<void()> wake) {
      if (wait_) {
        disarm_wake();
        // The callback reads wake_; replace it only once no callback can still be running.
        WaitForThreadpoolWaitCallbacks(wait_, TRUE);
      }
      wake_ = std::move(wake);
      if (!wake_ || wait_) {
        return;
      }
      wake_event_.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
      wait_ = wake_event_.get() ? CreateThreadpoolWait(&impl_t::on_fence_advanced, this, nullptr) : nullptr;
      if (!wait_) {
        BOOST_LOG(warning) << "ReShade SBS: cannot wait on the export fence (error " << GetLastError() << "); polling at stream cadence instead.";
        wake_ = {};
      }
    }

    bool frame_wake_active() const {
      return wait_ && wake_ && cached_ && wake_generation_ != 0 && wake_generation_ == metadata_.generation && wake_armed_.load(std::memory_order_acquire);
    }

    bool frame_held() const {
      return cached_.has_value();
    }

    bool frame_pending() const {
      // Unattached, nothing here can change: captures and keepalives call poll(), and connect()
      // attaches from there on its own schedule.
      if (!shared_) {
        return false;
      }
      if (static_cast<std::uint32_t>(read32(shared_->metadata_sequence)) != observed_metadata_sequence_ || read64(shared_->consumer_nonce) != nonce_ || WaitForSingleObject(process_.get(), 0) != WAIT_TIMEOUT) {
        return true;
      }
      // No acknowledged generation is open: its publication is the next change.
      if (!ready_fence_) {
        return false;
      }
      const auto completed = ready_fence_->GetCompletedValue();
      if (completed == UINT64_MAX) {
        return true;
      }
      const auto newest = cached_ ? cached_->sequence : 0;
      for (std::uint32_t i = 0; i < wire::slot_count; ++i) {
        auto &slot = shared_->slots[i];
        if (static_cast<int>(i) != held_slot_ && read64(slot.control) == wire::slot_control(metadata_.generation, wire::slot_state::ready)) {
          const auto sequence = read64(slot.sequence);
          if (sequence > newest && sequence <= completed) {
            return true;
          }
        }
      }
      return false;
    }

    void retire() {
      retire_slots();
    }

    // Ends the held slot's event query after every read recorded so far, so the claim that
    // replaces it can prove those reads complete without issuing (and waiting for) a new one.
    void reads_recorded() {
      if (held_slot_ < 0 || !retire_queries_[held_slot_]) {
        return;
      }
      context_->End(retire_queries_[held_slot_].Get());
      // Submits the conversion just recorded; the encoder would submit it next anyway.
      context_->Flush();
      held_reads_ended_ = true;
    }

  private:
    // Foreground ownership, the shared mapping and a live producer process: the prerequisites of
    // both a consumer connection and a status observation.
    bool connect(RECT source, int width, int height) {
      if (!supported_ || width <= 0 || height <= 0 || source.right <= source.left || source.bottom <= source.top) {
        return false;
      }
      const auto observation = observe_();
      const foreground_window::rect_t expected {source.left, source.top, source.right, source.bottom};
      if (observation.status != foreground_window::status_e::ok || observation.client_screen_rect != expected || observation.process_id == 0 || observation.window == 0) {
        detach();
        return false;
      }
      if (pid_ != observation.process_id || window_ != observation.window) {
        detach();
        pid_ = observation.process_id;
        window_ = observation.window;
      }
      if (!shared_) {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_attach_) {
          return false;
        }
        next_attach_ = now + std::chrono::milliseconds(250);
        if (!attach()) {
          return false;
        }
      }
      if (WaitForSingleObject(process_.get(), 0) != WAIT_TIMEOUT) {
        detach();
        return false;
      }
      return true;
    }

    std::optional<frame_t> poll_frame(RECT source, int width, int height) {
      if (!supported_ || !retire_slots() || !connect(source, width, height)) {
        return std::nullopt;
      }

      wire::metadata_t metadata;
      if (!snapshot(*shared_, metadata, &observed_metadata_sequence_)) {
        // A resize/disable is in progress. Never present a retained frame across a generation.
        cached_.reset();
        return std::nullopt;
      }
      if (!identity_matches(metadata)) {
        cached_.reset();
        return std::nullopt;
      }
      // Fullscreen may scale the game's raster to a different desktop extent. Window coverage
      // proves ownership above. Eyes authored at another size but the same aspect are scaled
      // by the consumer's linear sampling; a different aspect stays unavailable. Whenever the
      // snapshot describes a source, with a consumer or not, that is decided before any ring is
      // requested: no nonce is written for it, and a live connection that stops fitting is
      // withdrawn, so the producer neither allocates a ring nor packs stereo for a stream that
      // stays in 2D.
      if (wire::valid_source_metadata(metadata) && metadata.adapter_luid == adapter_luid_) {
        const auto fit = wire::fit_output(metadata, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
        report_fit(metadata, fit, width, height);
        if (fit == wire::output_fit::aspect_mismatch) {
          withdraw();
          return std::nullopt;
        }
      }
      if (!nonce_) {
        nonce_ = new_nonce();
        // Capabilities first, then their nonce, then the request itself (full barriers). A
        // producer trusts the bits only for the nonce they name, so it never applies them
        // to another consumer's request.
        InterlockedExchange(reinterpret_cast<volatile LONG *>(&shared_->consumer_capabilities), static_cast<LONG>(capabilities_));
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared_->capability_nonce), static_cast<LONG64>(nonce_));
        InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&shared_->consumer_nonce), static_cast<LONG64>(nonce_));
        return std::nullopt;
      }
      if (read64(shared_->consumer_nonce) != nonce_) {
        // A replacement converter owns the connection now. Do not fight it by writing again.
        cached_.reset();
        return std::nullopt;
      }
      if (metadata.accepted_consumer_nonce != nonce_ || !wire::valid_metadata(metadata) || metadata.adapter_luid != adapter_luid_) {
        cached_.reset();
        return std::nullopt;
      }
      if (metadata.generation != metadata_.generation) {
        reset_resources();
        if (!open_resources(metadata)) {
          return std::nullopt;
        }
        metadata_ = metadata;
        const char *transfer_name = "sRGB SDR";
        if (metadata.color_transfer == wire::transfer::scrgb) {
          transfer_name = "linear scRGB HDR";
        } else if (metadata.color_transfer == wire::transfer::pq) {
          transfer_name = "PQ HDR10";
        }
        BOOST_LOG(info) << "ReShade SBS connected: process " << pid_ << ", " << width << 'x' << height
                        << ", " << transfer_name << ", protocol " << metadata.protocol_version
                        << ", generation " << metadata.generation << '.';
      } else if (std::memcmp(&metadata, &metadata_, sizeof(metadata)) != 0) {
        // Resource identity, color and dimensions are immutable within an acknowledged generation.
        cached_.reset();
        return std::nullopt;
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
        if (static_cast<int>(i) == held_slot_) {
          continue;
        }
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
      float ui_parallax_uv = 0.0f;
      const bool cursor_plane_valid = wire::read_ui_parallax(metadata_.protocol_version, slot, ui_parallax_uv);
      // Capture all slot fields before validating their generation. Replacement resets
      // the mapped slots, so reading a field after this check could mix generations.
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
      if (!cursor_plane_valid) {
        // Retain the previous texture and its plane together. A malformed new frame must
        // neither move its cursor nor keep an unusable slot at the head of the ready ring.
        claim(slot, metadata_.generation, wire::slot_state::reading, wire::slot_state::free);
        return cached_;
      }
      // Conversion reads the shared slot directly; it stays `reading` while it is the newest
      // frame, so repeat conversions keep their exact pixels. Every earlier read of the
      // previously held slot is already recorded on this context, so an event query ended
      // after them covers all of them. The slot returns to the producer only once that query
      // completes: right here when reads_recorded() ended it and it already completed (the
      // reads normally finished during the previous encode), otherwise from retire_slots().
      if (held_slot_ >= 0) {
        auto &held_slot = shared_->slots[held_slot_];
        if (held_reads_ended_ && context_->GetData(retire_queries_[held_slot_].Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK &&
            read64(held_slot.sequence) == held_sequence_) {
          claim(held_slot, metadata_.generation, wire::slot_state::reading, wire::slot_state::free);
        } else {
          if (!held_reads_ended_) {
            context_->End(retire_queries_[held_slot_].Get());
          }
          retiring_[held_slot_] = {true, held_sequence_, metadata_.generation, 0};
          // Flush submits bounded work; neither producer nor consumer waits for the GPU.
          context_->Flush();
        }
      }
      held_reads_ended_ = false;
      held_slot_ = selected;
      held_sequence_ = sequence;
      const auto transfer = metadata_.color_transfer == wire::transfer::scrgb ? transfer_e::scrgb :
                            metadata_.color_transfer == wire::transfer::pq    ? transfer_e::pq :
                                                                                transfer_e::srgb;
      cached_ = frame_t {
        .texture = textures_[selected].Get(),
        .view = views_[selected].Get(),
        .linear = transfer == transfer_e::scrgb,
        .transfer = transfer,
        .timestamp = frame_time(qpc),
        .sequence = sequence,
        .producer_process_id = metadata_.producer_pid,
        .producer_creation_time = metadata_.producer_creation_time,
        .resource_generation = metadata_.generation,
        .ui_parallax_uv = ui_parallax_uv,
      };
      return cached_;
    }

  private:
    // Names a size mismatch once per generation and requested output. Without it an
    // incompatible game resolution silently leaves the stream waiting in 2D.
    void report_fit(const wire::metadata_t &metadata, wire::output_fit fit, int width, int height) {
      if (fit == wire::output_fit::exact || (metadata.generation == reported_generation_ &&
            width == reported_width_ && height == reported_height_)) {
        return;
      }
      reported_generation_ = metadata.generation;
      reported_width_ = width;
      reported_height_ = height;
      const auto game_width = metadata.packed_width / 2, stream_width = static_cast<std::uint32_t>(width) / 2;
      if (fit == wire::output_fit::scaled) {
        BOOST_LOG(info) << "ReShade SBS: game eyes " << game_width << 'x' << metadata.packed_height
                        << " are scaled to the stream's " << stream_width << 'x' << height
                        << " eyes. Run the game at " << stream_width << 'x' << height << " for native sharpness.";
      } else {
        BOOST_LOG(warning) << "ReShade SBS: game eyes " << game_width << 'x' << metadata.packed_height
                           << " cannot fill the stream's " << stream_width << 'x' << height
                           << " eyes without distortion (different aspect ratio); staying in 2D. Run the game at "
                           << stream_width << 'x' << height << " or set the stream resolution to "
                           << game_width << 'x' << metadata.packed_height << '.';
      }
    }

    bool identity_matches(const wire::metadata_t &metadata) const {
      return metadata.signature == wire::magic && wire::supported_version(metadata.protocol_version) &&
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
        if (desc.Width != metadata.packed_width || desc.Height != metadata.packed_height || desc.Format != static_cast<DXGI_FORMAT>(metadata.dxgi_format) || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0 || FAILED(device_->CreateShaderResourceView(textures_[i].Get(), nullptr, &views_[i]))) {
          reset_resources();
          return false;
        }
      }
      const D3D11_QUERY_DESC query {D3D11_QUERY_EVENT, 0};
      for (auto &retire_query : retire_queries_) {
        if (FAILED(device_->CreateQuery(&query, &retire_query))) {
          reset_resources();
          return false;
        }
      }
      return true;
    }

    // Returns a replaced slot to the producer once every read recorded before its event query
    // has completed. Never blocks. Later conversion work normally submits the query; a driver
    // may otherwise hold a lone event query back, so a query still pending after two polls is
    // checked once more with a flush.
    bool retire_slots() {
      if (!shared_) {
        return true;
      }
      for (std::uint32_t i = 0; i < wire::slot_count; ++i) {
        auto &retiring = retiring_[i];
        if (!retiring.active || !retire_queries_[i]) {
          continue;
        }
        auto result = context_->GetData(retire_queries_[i].Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (result == S_FALSE && ++retiring.pending_polls > 2) {
          result = context_->GetData(retire_queries_[i].Get(), nullptr, 0, 0);
        }
        if (FAILED(result)) {
          BOOST_LOG(warning) << "ReShade SBS receiver GPU read failed; this device must be recreated (HRESULT " << result << ").";
          reset_resources();
          supported_ = false;
          return false;
        }
        if (result != S_OK) {
          continue;
        }
        wire::metadata_t current;
        auto &slot = shared_->slots[i];
        if (snapshot(*shared_, current) && current.generation == retiring.generation && current.accepted_consumer_nonce == nonce_ && read64(slot.sequence) == retiring.sequence) {
          claim(slot, retiring.generation, wire::slot_state::reading, wire::slot_state::free);
        }
        retiring = {};
      }
      return true;
    }

    // Arms the wake on the held generation's fence once. Each completion re-arms it for the next
    // value on the thread pool, so every producer frame wakes the owner exactly once.
    void arm_wake() {
      if (!wait_ || !wake_ || !ready_fence_ || (wake_generation_ == metadata_.generation && wake_armed_.load(std::memory_order_acquire))) {
        return;
      }
      std::lock_guard lock(wake_lock_);
      // Releasing a replaced fence may signal registrations it still held; start clean.
      ResetEvent(wake_event_.get());
      wake_fence_ = ready_fence_;
      if (!rearm_locked()) {
        wake_fence_.Reset();
        wake_armed_.store(false, std::memory_order_release);
        wake_generation_ = 0;
        return;
      }
      wake_armed_.store(true, std::memory_order_release);
      wake_generation_ = metadata_.generation;
    }

    // Caller holds wake_lock_. A removed device completes every value: stop instead of spinning.
    bool rearm_locked() {
      const auto completed = wake_fence_->GetCompletedValue();
      if (completed == UINT64_MAX || FAILED(wake_fence_->SetEventOnCompletion(completed + 1, wake_event_.get()))) {
        return false;
      }
      SetThreadpoolWait(wait_, wake_event_.get(), nullptr);
      return true;
    }

    void disarm_wake() {
      wake_generation_ = 0;
      if (!wait_) {
        return;
      }
      std::lock_guard lock(wake_lock_);
      wake_armed_.store(false, std::memory_order_release);
      wake_fence_.Reset();
      SetThreadpoolWait(wait_, nullptr, nullptr);
    }

    static void CALLBACK on_fence_advanced(PTP_CALLBACK_INSTANCE, PVOID context, PTP_WAIT, TP_WAIT_RESULT) {
      auto *self = static_cast<impl_t *>(context);
      {
        std::lock_guard lock(self->wake_lock_);
        if (self->wake_fence_ && !self->rearm_locked()) {
          // The owner falls back to polling at stream cadence and re-arms on its next frame.
          self->wake_fence_.Reset();
          self->wake_armed_.store(false, std::memory_order_release);
        }
      }
      // A completion left over from a replaced fence only causes one spurious wake.
      if (self->wake_) {
        self->wake_();
      }
    }

    void reset_resources() {
      disarm_wake();
      cached_.reset();
      // Held and retiring reads are abandoned, never unlocked prematurely. A new generation has
      // new resources; the D3D runtime retains submitted resource references through completion.
      held_slot_ = -1;
      held_sequence_ = 0;
      held_reads_ended_ = false;
      retiring_ = {};
      for (auto &retire_query : retire_queries_) {
        retire_query.Reset();
      }
      for (auto &view : views_) {
        view.Reset();
      }
      for (auto &texture : textures_) {
        texture.Reset();
      }
      ready_fence_.Reset();
      metadata_ = {};
    }

    // Withdraws this receiver's request but keeps the mapping: its ring resources go, and its nonce
    // is cleared only while still current (a replacement converter's stays). The next poll that
    // fits writes a fresh nonce.
    void withdraw() {
      reset_resources();
      if (shared_ && nonce_) {
        InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&shared_->consumer_nonce), 0, static_cast<LONG64>(nonce_));
      }
      nonce_ = 0;
    }

    void detach() {
      reset_resources();
      if (shared_) {
        // Withdraw this connection so the producer does not build a generation
        // for a receiver that is gone (for example after a focus change). A
        // replacement receiver's nonce is left untouched.
        if (nonce_) {
          InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&shared_->consumer_nonce), 0, static_cast<LONG64>(nonce_));
        }
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
    // Advertised with each new nonce; see set_stream_pq().
    std::uint32_t capabilities_ = wire::consumer_accepts_pq;
    handle_t process_, mapping_;
    wire::shared_state_t *shared_ = nullptr;
    wire::metadata_t metadata_;
    struct retiring_t {
      bool active = false;
      std::uint64_t sequence = 0, generation = 0;
      std::uint32_t pending_polls = 0;
    };

    std::array<ComPtr<ID3D11Texture2D>, wire::slot_count> textures_;
    std::array<ComPtr<ID3D11ShaderResourceView>, wire::slot_count> views_;
    std::array<ComPtr<ID3D11Query>, wire::slot_count> retire_queries_;
    std::array<retiring_t, wire::slot_count> retiring_ {};
    ComPtr<ID3D11Fence> ready_fence_;
    std::optional<frame_t> cached_;
    // The slot behind cached_ (or the last frame before cached_ was dropped), still `reading`.
    int held_slot_ = -1;
    std::uint64_t held_sequence_ = 0;
    // The held slot's query was ended after every read since poll() last returned its frame.
    bool held_reads_ended_ = false;
    std::chrono::steady_clock::time_point next_attach_ {};
    std::uint64_t reported_generation_ = 0;
    int reported_width_ = 0, reported_height_ = 0;
    // Metadata sequence of the last poll that read the metadata; a change means it was replaced.
    std::uint32_t observed_metadata_sequence_ = 0;
    // Owner-thread state of the fence wake. The thread-pool callback shares only wake_fence_.
    std::function<void()> wake_;
    handle_t wake_event_;
    PTP_WAIT wait_ = nullptr;
    std::uint64_t wake_generation_ = 0;
    std::atomic<bool> wake_armed_ {false};
    std::mutex wake_lock_;
    ComPtr<ID3D11Fence> wake_fence_;
  };

  receiver_t::receiver_t(ID3D11Device *device, ID3D11DeviceContext *context, observer_t observe):
      impl_(std::make_unique<impl_t>(device, context, std::move(observe))) {}

  receiver_t::~receiver_t() = default;

  std::optional<frame_t> receiver_t::poll(RECT source_rect, int output_width, int output_height) {
    return impl_->poll(source_rect, output_width, output_height);
  }

  std::optional<source_status_t> receiver_t::status(RECT source_rect, int output_width, int output_height) {
    return impl_->status(source_rect, output_width, output_height);
  }

  void receiver_t::set_stream_pq(bool stream_pq) {
    impl_->set_stream_pq(stream_pq);
  }

  void receiver_t::set_frame_wake(std::function<void()> wake) {
    impl_->set_frame_wake(std::move(wake));
  }

  bool receiver_t::frame_wake_active() const {
    return impl_->frame_wake_active();
  }

  bool receiver_t::frame_held() const {
    return impl_->frame_held();
  }

  bool receiver_t::frame_pending() const {
    return impl_->frame_pending();
  }

  void receiver_t::retire() {
    impl_->retire();
  }

  void receiver_t::reads_recorded() {
    impl_->reads_recorded();
  }
}  // namespace platf::reshade_bridge

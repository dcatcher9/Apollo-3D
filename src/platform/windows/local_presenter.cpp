/**
 * @file src/platform/windows/local_presenter.cpp
 * @brief Local AR and shutter-monitor presentation: capture a virtual source display, convert it
 *        with the production RGB converter, and present it on a physical output.
 */
// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// platform includes
#include <d3d11_4.h>
#include <dwmapi.h>
#include <wrl/client.h>

// local includes
#include "display.h"
#include "display_config.h"
#include "local_monitor_pointer_geometry.h"
#include "local_presenter_window.h"
#include "local_rgb_converter.h"
#include "local_shutter_frame_ring.h"
#include "local_shutter_session.h"
#include "misc.h"
#include "shutter_eye_pass.h"
#include "src/capture_image_pool.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/video.h"
#include "utils.h"

#if !defined(SUNSHINE_SHADERS_DIR)  // for testing this needs to be defined in cmake as we don't do an install
  #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
#endif

namespace platf::dxgi {
  using namespace std::literals;
  using namespace local_presenter;

#ifdef SUNSHINE_TESTS
  namespace {
    std::vector<std::uint32_t> read_shutter_probe_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      ID3D11Texture2D *texture,
      UINT left,
      UINT width
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      desc.Width = width;
      desc.MiscFlags = desc.BindFlags = 0;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
        return {};
      }
      const D3D11_BOX box {left, 0, 0, left + width, desc.Height, 1};
      context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, texture, 0, &box);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return {};
      }
      std::vector<std::uint32_t> pixels(width * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(pixels.data() + y * width, static_cast<const std::uint8_t *>(mapped.pData) + y * mapped.RowPitch, width * sizeof(std::uint32_t));
      }
      context->Unmap(staging.Get(), 0);
      return pixels;
    }
  }  // namespace

  local_shutter_frame_probe_t probe_local_shutter_frame_ring() {
    local_shutter_frame_probe_t result;
    Microsoft::WRL::ComPtr<ID3D11Device> producer;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> producer_context;
    Microsoft::WRL::ComPtr<ID3D11Device> consumer;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> consumer_context;
    const D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
          levels, std::size(levels), D3D11_SDK_VERSION, &producer, nullptr, &producer_context))) {
      return result;
    }
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (FAILED(producer.As(&dxgi_device)) || FAILED(dxgi_device->GetAdapter(&adapter)) ||
        FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
          levels, std::size(levels), D3D11_SDK_VERSION, &consumer, nullptr, &consumer_context))) {
      return result;
    }
    local_shutter_frame_ring_t frames;
    result.available = true;
    if (!frames.init(producer.Get(), consumer.Get(), 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM)) {
      return result;
    }
    const auto publish = [&](const std::array<std::uint32_t, 8> &pixels) {
      auto *slot = frames.acquire_producer();
      if (!slot) {
        return false;
      }
      producer_context->UpdateSubresource(slot->producer_texture.Get(), 0, nullptr, pixels.data(), 4 * sizeof(std::uint32_t), 0);
      return frames.finish_producer(*slot, true, producer_context.Get());
    };
    const auto wait_pair = [&]() {
      ID3D11Texture2D *pair = nullptr;
      for (int attempt = 0; attempt < 1000 && !pair; ++attempt) {
        pair = frames.acquire_pair();
        if (!pair) {
          std::this_thread::sleep_for(1ms);
        }
      }
      return pair;
    };
    constexpr std::array<std::uint32_t, 8> first {0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00,
                                               0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00};
    constexpr std::array<std::uint32_t, 8> second {0xffff0000, 0xffff0000, 0xffffffff, 0xffffffff,
                                                0xffff0000, 0xffff0000, 0xffffffff, 0xffffffff};
    if (!publish(first)) {
      return result;
    }
    auto *pair = wait_pair();
    if (!pair) {
      return result;
    }
    bool signalled = false;
    frames.before_flush_for_test = [&] { signalled = true; };
    auto *unused_slot = frames.acquire_producer();
    if (!unused_slot) {
      return result;
    }
    const auto unused_completion = unused_slot->completion;
    result.retained_pair_not_republished =
      frames.finish_producer(*unused_slot, false, producer_context.Get()) &&
      !signalled && unused_slot->completion == unused_completion && frames.acquire_pair() == pair;

    // Hold the real producer inside publication. The consumer must return the first pair
    // before the producer is allowed to flush and release the second slot.
    std::promise<void> publisher_entered, allow_flush;
    auto publication_entered = publisher_entered.get_future();
    auto flush_allowed = allow_flush.get_future();
    frames.before_flush_for_test = [&] {
      publisher_entered.set_value();
      flush_allowed.wait();
    };
    std::future<ID3D11Texture2D *> acquiring;
    auto publishing = std::async(std::launch::async, [&] { return publish(second); });
    auto unblock = util::fail_guard([&] { allow_flush.set_value(); });
    if (publication_entered.wait_for(2s) != std::future_status::ready) {
      return result;
    }
    acquiring = std::async(std::launch::async, [&] { return frames.acquire_pair(); });
    result.consumer_progress_during_publication =
      acquiring.wait_for(2s) == std::future_status::ready && acquiring.get() == pair;
    allow_flush.set_value();
    unblock.disable();
    if (!publishing.get() || !result.consumer_progress_during_publication) {
      return result;
    }
    frames.before_flush_for_test = {};
    result.publication_bounded = frames.acquire_producer() == nullptr;
    const auto read = [&](ID3D11Texture2D *texture, UINT left, UINT width) {
      return read_shutter_probe_texture(consumer.Get(), consumer_context.Get(), texture, left, width);
    };
    result.current_pair_immutable = read(pair, 0, 4) == std::vector<std::uint32_t>(first.begin(), first.end());
    result.eye_crop_exact = read(pair, 0, 2) == std::vector<std::uint32_t>(4, 0xff0000ff) &&
                            read(pair, 2, 2) == std::vector<std::uint32_t>(4, 0xff00ff00);
    ID3D11Texture2D *next = nullptr;
    for (int attempt = 0; attempt < 1000; ++attempt) {
      next = frames.acquire_pair();
      if (next != pair) {
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    result.next_pair_published = next && next != pair && read(next, 0, 4) == std::vector<std::uint32_t>(second.begin(), second.end());
    return result;
  }

  local_shutter_handoff_probe_t probe_local_shutter_pair_handoff() {
    const auto probe = [](D3D_DRIVER_TYPE driver) {
      using Microsoft::WRL::ComPtr;
      using state_e = local_monitor_source_switch_t::state_e;
      local_shutter_handoff_probe_t result;
      ComPtr<ID3D11Device> initial_device, candidate_device, consumer_device;
      ComPtr<ID3D11DeviceContext> initial_context, candidate_context, consumer_context;
      const auto make_device = [driver](auto &device, auto &context) {
        constexpr D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
        return SUCCEEDED(D3D11CreateDevice(nullptr, driver, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &device, nullptr, &context));
      };
      if (!make_device(initial_device, initial_context) || !make_device(candidate_device, candidate_context) || !make_device(consumer_device, consumer_context)) {
        return result;
      }

      auto initial = std::make_shared<local_shutter_source_t>();
      auto candidate = std::make_shared<local_shutter_source_t>();
      if (
        !initial->ring.init(initial_device.Get(), consumer_device.Get(), 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM) ||
        !candidate->ring.init(candidate_device.Get(), consumer_device.Get(), 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM)
      ) {
        return result;
      }

      // An independent producer context holds the candidate's real ring completion fence below
      // its signalled value without relying on execution time or reading back the candidate.
      ComPtr<ID3D11Device5> initial_device5, candidate_device5;
      ComPtr<ID3D11DeviceContext4> initial_context4, candidate_context4;
      ComPtr<ID3D11Fence> signal_gate, wait_gate;
      HANDLE gate_handle = nullptr;
      if (
        FAILED(initial_device.As(&initial_device5)) || FAILED(candidate_device.As(&candidate_device5)) ||
        FAILED(initial_context.As(&initial_context4)) || FAILED(candidate_context.As(&candidate_context4)) ||
        FAILED(initial_device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&signal_gate))) ||
        FAILED(signal_gate->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &gate_handle))
      ) {
        return result;
      }
      const auto opened_gate = candidate_device5->OpenSharedFence(gate_handle, IID_PPV_ARGS(&wait_gate));
      CloseHandle(gate_handle);
      if (FAILED(opened_gate)) {
        return result;
      }
      result.available = true;
      result.used_warp = driver == D3D_DRIVER_TYPE_WARP;

      candidate->request = std::make_shared<local_monitor_source_switch_t::request_t>(
        ::video::SBS_GAME_SBS,
        1u,
        std::chrono::steady_clock::now() + 5s
      );
      local_shutter_pair_handoff_t handoff(initial);
      unsigned admission_attempts = 0;
      bool authorized = false;
      const auto try_adopt = [&](const auto &request) {
        ++admission_attempts;
        auto expected = state_e::requested;
        return authorized && request == candidate->request &&
               request->state.compare_exchange_strong(expected, state_e::adopted);
      };
      const auto publish = [](local_shutter_source_t &source, ID3D11DeviceContext *context, const auto &pixels) {
        auto *slot = source.ring.acquire_producer();
        if (!slot) {
          return false;
        }
        context->UpdateSubresource(slot->producer_texture.Get(), 0, nullptr, pixels.data(), 4 * sizeof(std::uint32_t), 0);
        return source.ring.finish_producer(*slot, true, context);
      };
      constexpr std::array<std::uint32_t, 8> old_pixels {0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00, 0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00};
      constexpr std::array<std::uint32_t, 8> new_pixels {0xffff0000, 0xffff0000, 0xffffffff, 0xffffffff, 0xffff0000, 0xffff0000, 0xffffffff, 0xffffffff};
      if (!publish(*initial, initial_context.Get(), old_pixels)) {
        return result;
      }
      ID3D11Texture2D *old_pair = nullptr;
      for (int attempt = 0; attempt < 1000 && !old_pair; ++attempt) {
        old_pair = handoff.acquire_pair(try_adopt);
        if (!old_pair) {
          std::this_thread::sleep_for(1ms);
        }
      }
      if (!old_pair || !publish(*initial, initial_context.Get(), new_pixels)) {
        return result;
      }
      const auto read = [&](ID3D11Texture2D *texture) {
        return read_shutter_probe_texture(consumer_device.Get(), consumer_context.Get(), texture, 0, 4);
      };
      handoff.publish(candidate);
      result.unpublished_retained = handoff.acquire_pair(try_adopt) == old_pair && !handoff.current_request() &&
                                    admission_attempts == 0u && read(old_pair) == std::vector<std::uint32_t>(old_pixels.begin(), old_pixels.end());

      if (FAILED(candidate_context4->Wait(wait_gate.Get(), 1))) {
        return result;
      }
      const auto release_gate = [&] {
        const auto status = initial_context4->Signal(signal_gate.Get(), 1);
        initial_context->Flush();
        return SUCCEEDED(status);
      };
      auto unblock = util::fail_guard([&] {
        (void) release_gate();
      });
      if (!publish(*candidate, candidate_context.Get(), new_pixels)) {
        return result;
      }
      result.unfenced_retained = handoff.acquire_pair(try_adopt) == old_pair && !handoff.current_request() && admission_attempts == 0u;
      if (!release_gate()) {
        return result;
      }
      unblock.disable();

      ID3D11Texture2D *denied_pair = nullptr;
      for (int attempt = 0; attempt < 1000 && admission_attempts == 0u; ++attempt) {
        denied_pair = handoff.acquire_pair(try_adopt);
        if (admission_attempts == 0u) {
          std::this_thread::sleep_for(1ms);
        }
      }
      result.denied_retained = admission_attempts == 1u && denied_pair == old_pair && !handoff.current_request() &&
                               candidate->request->state.load() == state_e::requested &&
                               read(old_pair) == std::vector<std::uint32_t>(old_pixels.begin(), old_pixels.end());
      authorized = true;
      auto *new_pair = handoff.acquire_pair(try_adopt);
      result.adopted_exact_pair = new_pair && new_pair != old_pair && !handoff.failed() &&
                                  handoff.current_request() == candidate->request &&
                                  candidate->request->state.load() == state_e::adopted &&
                                  read(new_pair) == std::vector<std::uint32_t>(new_pixels.begin(), new_pixels.end());
      return result;
    };
    auto result = probe(D3D_DRIVER_TYPE_WARP);
    return result.available ? result : probe(D3D_DRIVER_TYPE_HARDWARE);
  }
#endif

  local_presenter_result_e run_local_presenter(const local_presenter_config_t &config, std::stop_token stop_token) {
    using switch_state_e = local_monitor_source_switch_t::state_e;
    using switch_request_t = local_monitor_source_switch_t::request_t;
    const auto source_switch = config.frame_sequential ? config.source_switch : nullptr;
    const auto mark_switch_failed = [](const std::shared_ptr<switch_request_t> &request) {
      if (request) {
        auto state = request->state.load();
        while ((state == switch_state_e::requested || state == switch_state_e::adopted) &&
               !request->state.compare_exchange_weak(state, switch_state_e::failed)) {}
      }
    };
    // A capture attempt cannot hand an unfinished producer to its successor.
    auto fail_unfinished_switch = util::fail_guard([&]() {
      if (source_switch) {
        mark_switch_failed(source_switch->request.load());
      }
    });
    // Internal optical-calibration aid, sampled once for this presenter initialization.
    // It replaces only the physical shutter image; packet ownership and all timing gates stay active.
    std::array<wchar_t, 2> test_pattern_flag {};
    const bool shutter_test_pattern = config.frame_sequential &&
                                      GetEnvironmentVariableW(L"SUNSHINE_LOCAL_MONITOR_TEST_PATTERN", test_pattern_flag.data(), static_cast<DWORD>(test_pattern_flag.size())) == 1 &&
                                      test_pattern_flag[0] == L'1';
    std::array<wchar_t, 2> slot_timing_flag {};
    const bool shutter_slot_timing = config.frame_sequential &&
                                     GetEnvironmentVariableW(L"SUNSHINE_LOCAL_MONITOR_SLOT_TIMING", slot_timing_flag.data(), static_cast<DWORD>(slot_timing_flag.size())) == 1 &&
                                     slot_timing_flag[0] == L'1';
    // Internal glasses diagnostic, sampled once for this presenter initialization. One logical
    // slot keeps every presentation and timing gate but sends no USB eye command, so the lens it
    // would have driven shows only what the glasses do without their own signal.
    std::optional<unsigned> shutter_muted_slot;
    if (config.frame_sequential) {
      std::array<wchar_t, 3> mute_value {};
      SetLastError(ERROR_SUCCESS);
      const auto mute_length = GetEnvironmentVariableW(L"SUNSHINE_LOCAL_MONITOR_MUTE_SLOT", mute_value.data(), static_cast<DWORD>(mute_value.size()));
      if (mute_length || GetLastError() != ERROR_ENVVAR_NOT_FOUND) {
        if (mute_length != 1 || (mute_value[0] != L'L' && mute_value[0] != L'R')) {
          BOOST_LOG(error) << "Local shutter single-slot diagnostic requires SUNSHINE_LOCAL_MONITOR_MUTE_SLOT to be exactly L or R."sv;
          return local_presenter_result_e::error;
        }
        shutter_muted_slot = mute_value[0] == L'L' ? 0u : 1u;
        const bool muted_physical_left = (*shutter_muted_slot == 0u) != config.monitor_config.flip_eyes;
        BOOST_LOG(info) << "Local shutter single-slot diagnostic: logical "sv << (*shutter_muted_slot == 0u ? 'L' : 'R')
                        << " sends no USB eye command (nominal physical "sv << (muted_physical_left ? 'L' : 'R')
                        << " lens); presentation, timing gates and the other slot are unchanged."sv;
      }
    }
    auto read_target_rect = [&]() {
      std::lock_guard lock(config.live_target->mutex);
      return config.live_target->rect;
    };
    auto read_target_display_name = [&]() {
      std::lock_guard lock(config.live_target->mutex);
      return config.live_target->display_name;
    };
    auto target_rect = read_target_rect();
    auto target_display_name = read_target_display_name();
    const auto source_name_wide = platf::from_utf8(config.source_display_name);
    const auto target_name_wide = platf::from_utf8(target_display_name);
    const auto source_identity_at_start = query_local_presenter_display_identity(source_name_wide);
    const auto target_identity_at_start = query_local_presenter_display_identity(target_name_wide);
    if (!source_identity_at_start || !target_identity_at_start || !local_presenter_identity_is_coherent(config.live_target, *source_identity_at_start, *target_identity_at_start, config.target_adapter_id)) {
      BOOST_LOG(warning) << "Local AR presenter rejected a stale source/sink display-name generation."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_rect = query_local_presenter_display_rect(source_name_wide).value_or(RECT {});
    local_presenter_pointer_isolation_t pointer_isolation {
      source_name_wide,
      source_rect,
      target_rect,
      config.live_target,
      config.cursor_clip,
      config.frame_sequential,
    };
    // A cursor clip belongs to the complete local session, not one DXGI attempt. Reconcile it
    // before any early return so repeated pre-window reinitializations cannot leave stale bounds.
    pointer_isolation.refresh_clip();
    const int output_width = target_rect.right - target_rect.left;
    const int output_height = target_rect.bottom - target_rect.top;
    if (output_width <= 0 || output_height <= 0 || config.source_display_name.empty()) {
      BOOST_LOG(error) << "Local AR presenter received invalid source or target geometry."sv;
      return local_presenter_result_e::error;
    }
    const int eye_repeats = config.frame_sequential ? config.monitor_config.eye_repeats : 1;
    if (config.frame_sequential &&
        (eye_repeats < 1 || eye_repeats > 4 || config.target_refresh_millihz <= 0 ||
         !::video::is_packed_mode(config.sbs_mode) ||
         output_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / 2)) {
      BOOST_LOG(error) << "Local shutter monitor received an unsupported stereo raster or eye cadence."sv;
      return local_presenter_result_e::error;
    }
    const int packed_width = config.frame_sequential ? output_width * 2 : output_width;
    const int capture_refresh_millihz = config.frame_sequential ?
                                          std::max(1, config.target_refresh_millihz / (2 * eye_repeats)) :
                                          config.target_refresh_millihz;

    ::video::config_t capture_config {};
    capture_config.width = ::video::is_packed_mode(config.sbs_mode) ? packed_width / 2 : output_width;
    capture_config.height = output_height;
    capture_config.framerate = std::max(1, (capture_refresh_millihz + 500) / 1000);
    capture_config.dynamicRange = config.hdr ? 1 : 0;
    capture_config.encodingFramerate = capture_refresh_millihz;

    const auto preferred_capture_backend = config.capture_failover->preferred_backend();
    // A display-mode change after this point invalidates the duplication opened below.
    const auto display_epoch_at_open = platf::display_change_epoch();
    const local_monitor_switch_is_current_t switch_is_current {source_switch, stop_token, display_epoch_at_open};
    auto display = platf::display(
      config.source_display_name,
      capture_config,
      preferred_capture_backend
    );
    if (display) {
      config.capture_failover->note_backend_opened(display->capture_backend());
    }
    auto dxgi_display = std::dynamic_pointer_cast<display_base_t>(display);
    if (!dxgi_display) {
      BOOST_LOG(warning) << "Local AR presenter could not yet open virtual source display "sv
                         << config.source_display_name << "; refreshing display topology."sv;
      return local_presenter_result_e::reinit;
    }
    // Follow the sink's exact milliHz capture cadence. Remote capture's whole-rate fallback only
    // adapts downward and would turn, for example, 60.120 Hz into 60 fps.
    dxgi_display->set_local_frame_rate_millihz(capture_refresh_millihz);
    if (dxgi_display->is_hdr() != config.hdr) {
      BOOST_LOG(info) << "Local AR virtual source has not settled to "sv
                      << (config.hdr ? "HDR"sv : "SDR"sv) << "; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_identity_after_open = query_local_presenter_display_identity(source_name_wide);
    DXGI_ADAPTER_DESC1 source_adapter_desc {};
    // DXGI attributes the SudoVDA output to its render GPU, while CCD reports the IddCx adapter.
    // The zero-copy contract is that capture opened on the same GPU that presents to the sink.
    if (!source_identity_after_open || !display_config::equal_device_paths(source_identity_after_open->device_path, source_identity_at_start->device_path) || !dxgi_display->adapter || FAILED(dxgi_display->adapter->GetDesc1(&source_adapter_desc)) || !display_config::same_luid(source_adapter_desc.AdapterLuid, config.target_adapter_id)) {
      BOOST_LOG(info) << "Local AR virtual source identity changed while DXGI capture was opening; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }

    const auto open_converter = [&](int mode, const std::shared_ptr<safe::event_t<bool>> &ready) -> local_rgb_converter_t {
      auto instance = local_rgb_converter_t::create();
      if (instance.init(display, dxgi_display->adapter.get(), platf::pix_fmt_e::nv12) ||
          instance.init_rgb_output(packed_width, output_height, mode, config.sbs_config, ready)) {
        BOOST_LOG(error) << "Local AR presenter could not initialize the production RGB converter."sv;
        return {};
      }
      return instance;
    };
    int producer_mode = config.sbs_mode;
    auto depth_pipeline_ready_event = std::make_shared<safe::event_t<bool>>();
    auto converter = open_converter(producer_mode, depth_pipeline_ready_event);
    if (!converter) {
      return local_presenter_result_e::error;
    }

    DXGI_OUTPUT_DESC source_output_desc {};
    if (!dxgi_display->output || FAILED(dxgi_display->output->GetDesc(&source_output_desc))) {
      BOOST_LOG(error) << "Local AR presenter could not query the virtual source geometry."sv;
      return local_presenter_result_e::error;
    }
    pointer_isolation.source_fallback_rect = source_output_desc.DesktopCoordinates;
    pointer_isolation.refresh_clip();

    if (!register_local_presenter_window_class()) {
      BOOST_LOG(error) << "Local AR presenter could not register its window class: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }

    std::promise<std::pair<HWND, DWORD>> window_ready;
    auto window_future = window_ready.get_future();
    std::jthread window_thread([&pointer_isolation,
                                target_rect,
                                output_width,
                                output_height,
                                ready = std::move(window_ready)](std::stop_token window_stop_token) mutable {
      HWND created_window = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        local_presenter_window_class,
        L"Sunshine 3D Local SBS AI",
        WS_POPUP,
        target_rect.left,
        target_rect.top,
        output_width,
        output_height,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        &pointer_isolation
      );
      const DWORD create_error = created_window ? ERROR_SUCCESS : GetLastError();
      if (created_window) {
        // Hovering a taskbar thumbnail makes Windows hide every other window to preview that
        // one. The presenter is the 3D picture itself, not a window of the desktop: hidden, it
        // would leave the bare physical output with an empty frame where the picture was.
        BOOL excluded = TRUE;
        DwmSetWindowAttribute(created_window, DWMWA_EXCLUDED_FROM_PEEK, &excluded, sizeof(excluded));
        pointer_isolation.acquire_clip();
      }
      ready.set_value({created_window, create_error});
      if (!created_window) {
        return;
      }

      // Capture may block for 200 ms on a static desktop. Keep Win32 input on this dedicated
      // message loop so pointer entry is handled immediately regardless of frame
      // cadence. Stopping the owner posts WM_CLOSE and wakes GetMessageW().
      std::stop_callback stop_window(window_stop_token, [created_window]() {
        PostMessageW(created_window, WM_CLOSE, 0, 0);
      });
      // A low-level mouse hook is called on the thread that installed it, through its message
      // loop. This thread does nothing else, so it answers at once.
      HHOOK mouse_hook = nullptr;
      if (pointer_isolation.park_only) {
        pointer_isolation.refresh_hook_sink();
        hooked_pointer_isolation.store(&pointer_isolation);
        mouse_hook = SetWindowsHookExW(WH_MOUSE_LL, local_presenter_mouse_hook, GetModuleHandleW(nullptr), 0);
        if (!mouse_hook) {
          hooked_pointer_isolation.store(nullptr);
          BOOST_LOG(warning) << "Local shutter monitor could not watch pointer movement ("sv << GetLastError()
                             << "); the pointer may jump to the desktop corner at an outer edge."sv;
        }
      }
      MSG message {};
      while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      if (mouse_hook) {
        UnhookWindowsHookEx(mouse_hook);
        hooked_pointer_isolation.store(nullptr);
      }
      if (IsWindow(created_window)) {
        DestroyWindow(created_window);
      }
    });
    const auto [window, window_create_error] = window_future.get();
    if (!window) {
      BOOST_LOG(error) << "Local AR presenter could not create its output window: "sv
                       << window_create_error;
      return local_presenter_result_e::error;
    }
    const auto close_window = [&]() {
      const auto close_started = std::chrono::steady_clock::now();
      window_thread.request_stop();
      if (IsWindow(window)) {
        PostMessageW(window, WM_CLOSE, 0, 0);
      }
      if (window_thread.joinable()) {
        window_thread.join();
      }
      SetThreadExecutionState(ES_CONTINUOUS);
      const auto close_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - close_started
      );
      if (close_ms > 250ms) {
        BOOST_LOG(info) << "Local AR window-input shutdown took "sv << close_ms.count() << " ms."sv;
      }
    };
    auto destroy_window = util::fail_guard(close_window);

    if (!SetWindowPos(
          window,
          HWND_TOPMOST,
          target_rect.left,
          target_rect.top,
          output_width,
          output_height,
          SWP_NOACTIVATE
        )) {
      BOOST_LOG(warning) << "Local AR presenter could not place its hidden output window: "sv
                         << GetLastError();
      return local_presenter_result_e::reinit;
    }
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapchain;
    Microsoft::WRL::ComPtr<ID3D11Device> shutter_device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> shutter_context;
    auto status = converter.device()->QueryInterface(IID_PPV_ARGS(&dxgi_device));
    if (SUCCEEDED(status)) {
      status = dxgi_device->GetAdapter(&adapter);
    }
    dxgi_device.Reset();
    if (SUCCEEDED(status)) {
      status = adapter->GetParent(IID_PPV_ARGS(&factory));
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not obtain its DXGI factory: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    DXGI_ADAPTER_DESC presenter_adapter_desc {};
    status = adapter->GetDesc(&presenter_adapter_desc);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not identify its graphics adapter: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    if (!display_config::same_luid(presenter_adapter_desc.AdapterLuid, config.target_adapter_id)) {
      BOOST_LOG(info) << "Local AR physical output migrated to another graphics adapter; "sv
                         "requesting a complete presentation-session rebuild."sv;
      return local_presenter_result_e::error;
    }
    // The driver reads its variable-refresh setting for this program when a graphics device is
    // created. A shutter presenter on a Windows desktop refresh rate allows variable refresh
    // before it creates its device and asks for a fixed refresh again when it ends. Every other
    // presenter leaves the profile as start-up wrote it, with a fixed refresh.
    const bool variable_refresh = config.frame_sequential && config.target_desktop_refresh_rate;
    const auto fixed_refresh_again = util::fail_guard([variable_refresh]() {
      if (variable_refresh) {
        platf::set_presentation_variable_refresh(false);
      }
    });
    if (variable_refresh) {
      const bool written = platf::set_presentation_variable_refresh(true);
      BOOST_LOG(info) << "Local shutter monitor refresh: Windows desktop rate; the driver keeps its variable refresh for this presenter"sv
                      << (written ? "."sv : " (the driver profile could not be written)."sv);
    }
    if (config.frame_sequential) {
      const D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      status = D3D11CreateDevice(
        adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        levels, std::size(levels), D3D11_SDK_VERSION, &shutter_device, nullptr, &shutter_context
      );
      if (FAILED(status)) {
        BOOST_LOG(error) << "Local shutter monitor could not create its independent presentation device: "sv << util::log_hex(status);
        return local_presenter_result_e::error;
      }
      // Give the independent eye-copy/present device the same relative GPU priority as
      // capture/conversion, preserving the existing process scheduling policy.
      Microsoft::WRL::ComPtr<IDXGIDevice> shutter_dxgi;
      auto priority_result = shutter_device.As(&shutter_dxgi);
      if (SUCCEEDED(priority_result)) {
        priority_result = shutter_dxgi->SetGPUThreadPriority(7);
      }
      if (FAILED(priority_result)) {
        BOOST_LOG(warning) << "Local shutter monitor could not request relative GPU priority 7: "sv
                           << util::log_hex(priority_result) << "; scanout validation remains required."sv;
      } else {
        INT actual_priority = 0;
        const auto readback_result = shutter_dxgi->GetGPUThreadPriority(&actual_priority);
        if (FAILED(readback_result)) {
          BOOST_LOG(warning) << "Local shutter monitor could not verify its relative GPU priority: "sv
                             << util::log_hex(readback_result) << '.';
        } else if (actual_priority != 7) {
          BOOST_LOG(warning) << "Local shutter monitor GPU priority readback: requested=7 actual="sv
                             << actual_priority << " result="sv << util::log_hex(readback_result) << '.';
        } else {
          BOOST_LOG(info) << "Local shutter presentation GPU context uses relative priority 7; process scheduling policy retained."sv;
        }
      }
    }
    ID3D11Device *presentation_device = config.frame_sequential ? shutter_device.Get() : converter.device();

    Microsoft::WRL::ComPtr<IDXGIOutput> target_output;
    DXGI_OUTPUT_DESC target_output_desc {};
    const auto target_display_name_wide = target_name_wide;
    for (UINT output_index = 0;; ++output_index) {
      Microsoft::WRL::ComPtr<IDXGIOutput> candidate;
      const auto enum_status = adapter->EnumOutputs(output_index, &candidate);
      if (enum_status == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (FAILED(enum_status) || FAILED(candidate->GetDesc(&target_output_desc))) {
        continue;
      }
      if (display_config::equal_device_paths(target_output_desc.DeviceName, target_display_name_wide)) {
        target_output = std::move(candidate);
        break;
      }
    }
    if (!target_output) {
      BOOST_LOG(warning) << "Local AR physical output "sv << target_display_name
                         << " is not yet available through DXGI; refreshing topology."sv;
      return local_presenter_result_e::reinit;
    }
    const auto source_identity_before_present = query_local_presenter_display_identity(source_name_wide);
    const auto target_identity_before_present = query_local_presenter_display_identity(target_display_name_wide);
    if (!source_identity_before_present || !target_identity_before_present || !display_config::equal_device_paths(source_identity_before_present->device_path, source_identity_at_start->device_path) || !display_config::equal_device_paths(target_identity_before_present->device_path, target_identity_at_start->device_path) || !local_presenter_identity_is_coherent(config.live_target, *source_identity_before_present, *target_identity_before_present, config.target_adapter_id)) {
      BOOST_LOG(info) << "Local AR source/sink identity changed while presentation resources were opening; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }

    const auto actual_target_rect = target_output_desc.DesktopCoordinates;
    const auto actual_target_width = actual_target_rect.right - actual_target_rect.left;
    const auto actual_target_height = actual_target_rect.bottom - actual_target_rect.top;
    if (actual_target_width != output_width || actual_target_height != output_height) {
      BOOST_LOG(info) << "Local AR physical output mode changed from "sv << output_width << 'x'
                      << output_height << " to "sv << actual_target_width << 'x'
                      << actual_target_height << "; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    Microsoft::WRL::ComPtr<IDXGIOutput6> target_output6;
    DXGI_OUTPUT_DESC1 target_desc1 {};
    const bool target_color_volume_available =
      SUCCEEDED(target_output.As(&target_output6)) && SUCCEEDED(target_output6->GetDesc1(&target_desc1));
    if (target_color_volume_available) {
      BOOST_LOG(info) << "Local AR physical color volume: colorspace="sv
                      << dxgi_display->colorspace_to_string(target_desc1.ColorSpace)
                      << " bits="sv << target_desc1.BitsPerColor
                      << " luminance="sv << target_desc1.MinLuminance << '-'
                      << target_desc1.MaxLuminance << " nits full_frame="sv
                      << target_desc1.MaxFullFrameLuminance << " nits."sv;
    } else {
      BOOST_LOG(warning) << "Could not query the local AR physical output color volume."sv;
    }
    if (config.hdr && (!target_color_volume_available || target_desc1.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)) {
      BOOST_LOG(info) << "Local AR physical output has not settled to HDR10/PQ; reinitializing."sv;
      return local_presenter_result_e::reinit;
    }
    target_rect = actual_target_rect;
    if (!SetWindowPos(
          window,
          HWND_TOPMOST,
          target_rect.left,
          target_rect.top,
          output_width,
          output_height,
          SWP_NOACTIVATE | SWP_SHOWWINDOW
        )) {
      BOOST_LOG(warning) << "Local AR presenter could not place its output window on the verified sink: "sv
                         << GetLastError();
      return local_presenter_result_e::reinit;
    }
    ShowWindow(window, SW_SHOWNOACTIVATE);

    DXGI_SWAP_CHAIN_DESC1 swapchain_desc {};
    swapchain_desc.Width = output_width;
    swapchain_desc.Height = output_height;
    // HDR capture is linear scRGB. Preserve it in an FP16 scRGB swapchain and let DWM perform the
    // final conversion to the restricted physical output's HDR10/PQ color volume.
    swapchain_desc.Format = config.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT :
                                         DXGI_FORMAT_B8G8R8A8_UNORM;
    swapchain_desc.SampleDesc.Count = 1;
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    // The shutter presenter shows an eye by presenting it again on every refresh instead of
    // asking Windows to hold one present for several refreshes: Windows was measured to hold a
    // finished image one refresh too long about every ten seconds. The remaining presents of the
    // displayed eye and all presents of the next eye are queued together, so the queue is two
    // eyes deep: one buffer per queued present, one on screen and one being drawn.
    const UINT shutter_queue_depth = config.frame_sequential ? static_cast<UINT>(2 * eye_repeats) : 1u;
    swapchain_desc.BufferCount = config.frame_sequential ? shutter_queue_depth + 2u : 2u;
    swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    swapchain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    status = factory->CreateSwapChainForHwnd(
      presentation_device,
      window,
      &swapchain_desc,
      nullptr,
      target_output.Get(),
      &swapchain
    );
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not create its swapchain: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);

    Microsoft::WRL::ComPtr<IDXGISwapChain2> swapchain2;
    status = swapchain.As(&swapchain2);
    if (SUCCEEDED(status)) {
      status = swapchain2->SetMaximumFrameLatency(shutter_queue_depth);
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not configure bounded frame latency: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    const auto frame_latency_waitable = swapchain2->GetFrameLatencyWaitableObject();
    if (!frame_latency_waitable) {
      BOOST_LOG(error) << "Local AR presenter could not obtain its frame-latency waitable object: "sv
                       << GetLastError();
      return local_presenter_result_e::error;
    }
    auto close_frame_latency_waitable = util::fail_guard([&]() {
      CloseHandle(frame_latency_waitable);
    });

    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapchain3;
    status = swapchain.As(&swapchain3);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not query IDXGISwapChain3: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    const auto presentation_color_space = config.hdr ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 :
                                                       DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    UINT color_space_support = 0;
    status = swapchain3->CheckColorSpaceSupport(presentation_color_space, &color_space_support);
    if (FAILED(status) || !(color_space_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
      BOOST_LOG(error) << "Local AR presenter does not support the required "sv
                       << (config.hdr ? "linear scRGB"sv : "Rec.709"sv)
                       << " swapchain color space: "sv << util::log_hex(status);
      return local_presenter_result_e::error;
    }
    status = swapchain3->SetColorSpace1(presentation_color_space);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not set its swapchain color space: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    // D3D11 flip-model swapchains expose the current writable buffer as buffer 0.
    // Unlike D3D12, querying every physical buffer by index is not supported.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> backbuffer;
    status = swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> backbuffer_rtv;
    if (SUCCEEDED(status)) {
      status = presentation_device->CreateRenderTargetView(backbuffer.Get(), nullptr, &backbuffer_rtv);
    }
    if (FAILED(status)) {
      BOOST_LOG(error) << "Local AR presenter could not create the swapchain render target: "sv
                       << util::log_hex(status);
      return local_presenter_result_e::error;
    }

    local_shutter_test_pattern_t test_pattern;
    if (shutter_test_pattern) {
      D3D11_FEATURE_DATA_D3D11_OPTIONS options {};
      status = shutter_device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof(options));
      if (SUCCEEDED(status) && options.ClearView) {
        status = shutter_context.As(&test_pattern.context);
      }
      if (FAILED(status) || !options.ClearView || !test_pattern.context) {
        BOOST_LOG(error) << "Local shutter calibration pattern requires ClearView support: "sv << util::log_hex(status);
        return local_presenter_result_e::error;
      }
      // Five columns by seven rows. Precompute the large centered L/R glyphs once;
      // each eye needs one background clear and one bounded batch of white rectangles.
      constexpr std::array<std::array<std::uint8_t, 7>, 2> glyph_rows {{
        {{0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111}},
        {{0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001}},
      }};
      const LONG cell = std::min(output_width / 9, output_height / 11);
      if (cell < 1) {
        BOOST_LOG(error) << "Local shutter calibration pattern received a raster too small for its eye labels."sv;
        return local_presenter_result_e::error;
      }
      const LONG glyph_left = (output_width - 5 * cell) / 2;
      const LONG glyph_top = (output_height - 7 * cell) / 2;
      for (std::size_t eye = 0; eye < glyph_rows.size(); ++eye) {
        auto &glyph = test_pattern.glyphs[eye];
        for (unsigned row = 0; row < 7; ++row) {
          for (unsigned column = 0; column < 5; ++column) {
            if (glyph_rows[eye][row] & (1u << (4u - column))) {
              const LONG left = glyph_left + static_cast<LONG>(column) * cell;
              const LONG top = glyph_top + static_cast<LONG>(row) * cell;
              glyph.rectangles[glyph.count++] = {left, top, left + cell, top + cell};
            }
          }
        }
      }
      BOOST_LOG(info) << "Local shutter calibration pattern enabled: left=red L, right=blue R; RGB values are 0 and 1 (0 and 6, 480 nits, in HDR), "sv
                         "production packet ownership, clock and USB emitter remain active; emitter_flip_eyes="sv
                      << config.monitor_config.flip_eyes << '.';
    }

    // Experiment: a lens window longer than the time the whole screen shows one picture, with
    // the other eye's picture taken out of the lines that are redrawn meanwhile. It is off
    // unless this names a settings file, which is read again whenever it is saved. Both objects
    // outlive the shutter thread declared below.
    detail::shutter_eye_pass_t ghost_pass;
    std::optional<detail::shutter_ghost_settings_file_t> ghost_settings_file;
    if (config.frame_sequential) {
      std::array<wchar_t, 512> ghost_settings_path {};
      const auto length = GetEnvironmentVariableW(L"SUNSHINE_LOCAL_MONITOR_GHOST_SETTINGS", ghost_settings_path.data(), static_cast<DWORD>(ghost_settings_path.size()));
      if (length > 0 && length < ghost_settings_path.size()) {
        auto vertex_code = compile_vertex_shader(SUNSHINE_SHADERS_DIR "/local_shutter_eye_vs.hlsl");
        auto pixel_code = compile_pixel_shader(SUNSHINE_SHADERS_DIR "/local_shutter_eye_ps.hlsl");
        if (!vertex_code || !pixel_code ||
            !ghost_pass.init(shutter_device.Get(), vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), pixel_code->GetBufferPointer(), pixel_code->GetBufferSize())) {
          BOOST_LOG(error) << "Local shutter ghost correction could not prepare its shaders."sv;
          return local_presenter_result_e::error;
        }
        ghost_settings_file.emplace(
          std::filesystem::path {ghost_settings_path.data()},
          [](const std::filesystem::path &path, const detail::shutter_ghost_settings_t &settings, const std::vector<std::string> &problems) {
            BOOST_LOG(info) << "Local shutter ghost correction settings from "sv << path.string()
                            << ": top "sv << settings.top_peak << " over "sv << settings.top_extent
                            << " of the height, bottom "sv << settings.bottom_peak << " over "sv << settings.bottom_extent
                            << ", black_raise "sv << settings.black_raise << ", white_nits "sv << settings.white_nits
                            << ", pattern "sv << settings.pattern << ", view "sv << (settings.view_left ? "left"sv : "right"sv) << '.';
            for (const auto &problem : problems) {
              BOOST_LOG(warning) << "Local shutter ghost correction settings: "sv << problem;
            }
          }
        );
        BOOST_LOG(info) << "Local shutter ghost correction is available and follows its settings file."sv;
      }
    }

    const auto open_shutter_source = [&](ID3D11Device *device, std::shared_ptr<switch_request_t> request = {}) -> std::shared_ptr<local_shutter_source_t> {
      auto source = std::make_shared<local_shutter_source_t>();
      source->request = std::move(request);
      if (!source->ring.init(device, shutter_device.Get(), packed_width, output_height, swapchain_desc.Format)) {
        BOOST_LOG(error) << "Local shutter monitor could not create its bounded shared stereo packets."sv;
        return nullptr;
      }
      return source;
    };
    auto shutter_source = config.frame_sequential ? open_shutter_source(converter.device()) : nullptr;
    std::atomic<bool> shutter_failed = false;
    std::atomic<bool> shutter_pair_presented = false;
    if (config.frame_sequential && !shutter_source) {
      return local_presenter_result_e::error;
    }
    local_shutter_pair_handoff_t shutter_pairs {shutter_source};
    // Retire the previous ring on the producer thread after the new pair has been confirmed.
    std::shared_ptr<local_shutter_source_t> retired_shutter_source;
    // What the shutter thread borrows. Everything it names is declared above and the thread
    // below, so the thread is joined before any of it is destroyed.
    const local_shutter_presenter_t shutter_presenter {
      .config = config,
      .session_stop = stop_token,
      .window = window,
      .eye_repeats = eye_repeats,
      .pairs = shutter_pairs,
      .failed = shutter_failed,
      .pair_presented = shutter_pair_presented,
      .source_switch = source_switch,
      .switch_is_current = switch_is_current,
      .swapchain = swapchain,
      .swapchain_desc = swapchain_desc,
      .frame_latency_waitable = frame_latency_waitable,
      .device = shutter_device,
      .context = shutter_context,
      .backbuffer = backbuffer,
      .backbuffer_rtv = backbuffer_rtv,
      .output_width = output_width,
      .output_height = output_height,
      .test_pattern = shutter_test_pattern,
      .test_pattern_labels = test_pattern,
      .ghost_pass = ghost_pass,
      .ghost_settings_file = ghost_settings_file,
      .slot_timing = shutter_slot_timing,
      .muted_slot = shutter_muted_slot,
      .display_epoch_at_open = display_epoch_at_open,
      .source_name = source_name_wide,
      .target_name = target_display_name_wide,
      .target_display_name = target_display_name,
      .source_identity_at_start = source_identity_at_start,
      .target_identity_at_start = target_identity_at_start,
      .target_output = target_output,
      .target_output_desc = target_output_desc,
      .target_desc1 = target_desc1,
    };
    std::jthread shutter_thread;
    if (config.frame_sequential) {
      shutter_thread = std::jthread([&shutter_presenter](std::stop_token shutter_stop) {
        run_local_shutter_presentation(shutter_presenter, std::move(shutter_stop));
      });
    }

    BOOST_LOG(info) << "Local AR presentation started: source="sv << config.source_display_name
                    << " target="sv << target_display_name << " ["sv << target_rect.left << ','
                    << target_rect.top << "] "sv << output_width << 'x' << output_height << '@'
                    << capture_config.framerate << " mode="sv
                    << (config.sbs_mode == ::video::SBS_AI       ? "SBS AI"sv :
                        config.sbs_mode == ::video::SBS_GAME_SBS ? "Game 3D"sv :
                        config.sbs_mode == ::video::SBS_GAME_MONO ? "Game 3D mono"sv :
                                                                    "normal"sv)
                    << " color="sv << (config.hdr ? "HDR linear scRGB FP16"sv : "SDR Rec.709 BGRA8"sv)
                    << "; physical output is presentation-only."sv;

    ::video::detail::capture_image_pool_t<3> image_pool;
    const bool diagnostics_enabled = ::config::sunshine.diagnostics_enabled;
    bool capture_cursor = true;
    bool window_closed = false;
    auto next_pointer_fallback = std::chrono::steady_clock::now();
    auto poll_presenter_state = [&]() {
      window_closed = !IsWindow(window);
      // ClipCursor normally prevents entry without any polling. Keep a throttled global fallback
      // for applications that replace it while owning Win32 mouse capture: in that case movement
      // is routed to the captured source window and the presenter HWND receives no mouse message.
      const auto now = std::chrono::steady_clock::now();
      if (!window_closed && now >= next_pointer_fallback) {
        pointer_isolation.clamp_pointer_to_source_edge();
        next_pointer_fallback = now + 100ms;
      }
    };

    auto pull_image = [&](std::shared_ptr<platf::img_t> &image) {
      image.reset();
      if (stop_token.stop_requested() || window_closed) {
        return false;
      }
      image = image_pool.try_acquire([&] {
        return display->alloc_img();
      });
      return static_cast<bool>(image);
    };

    bool presentation_reinit_requested = false;
    std::uint64_t capture_attempt_frames = 0;
    std::uint64_t captured_frames = 0;
    std::uint64_t presented_frames = 0;
    std::uint64_t busy_present_retries = 0;
    std::uint64_t latency_wait_retries = 0;
    std::uint64_t present_call_retries = 0;
    std::shared_ptr<platf::img_t> retained_presenter_source;
    detail::local_presenter_retry_state_t presenter_retry;
    detail::local_presenter_schedule_t presenter_schedule {capture_refresh_millihz};
    bool replacement_published = false;
    const auto prepare_requested_source = [&]() {
      // A request that raced a capture reinitialization must wait for this attempt's own
      // confirmed pair. An empty initial ring cannot anchor a continuing shutter clock.
      if (!source_switch || !shutter_pair_presented.load(std::memory_order_acquire)) {
        return true;
      }
      if (shutter_source->request && shutter_source->request->state.load() == switch_state_e::presented) {
        retired_shutter_source.reset();
      }
      const auto request = source_switch->request.load();
      if (!request || request == shutter_source->request || request->state.load() != switch_state_e::requested) {
        return true;
      }
      if (!switch_is_current(request) || request->sbs_mode == producer_mode ||
          (request->sbs_mode != ::video::SBS_AI && request->sbs_mode != ::video::SBS_GAME_SBS)) {
        mark_switch_failed(request);
        return false;
      }

      // The old converter can retire its estimator on a background thread. The replacement
      // owns a distinct immediate context and starts with fresh input and depth authority.
      // Completion of the old ring is independent of either converter's destruction.
      converter.reset();
      auto ready = std::make_shared<safe::event_t<bool>>();
      auto replacement = open_converter(request->sbs_mode, ready);
      auto replacement_source = replacement ? open_shutter_source(replacement.device(), request) : nullptr;
      if (!replacement_source || !switch_is_current(request) || request->state.load() != switch_state_e::requested) {
        mark_switch_failed(request);
        return false;
      }
      retired_shutter_source = std::exchange(shutter_source, std::move(replacement_source));
      converter = std::move(replacement);
      depth_pipeline_ready_event = std::move(ready);
      producer_mode = request->sbs_mode;
      replacement_published = false;
      presenter_retry = {};
      presenter_schedule = detail::local_presenter_schedule_t {capture_refresh_millihz};
      if (retained_presenter_source) {
        presenter_retry.observe_source();
      }
      return true;
    };
    auto present_stats_started = diagnostics_enabled ? std::chrono::steady_clock::now() :
                                                       std::chrono::steady_clock::time_point {};
    auto log_present_stats = [&]() {
      if (!diagnostics_enabled) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = now - present_stats_started;
      if (elapsed < std::chrono::seconds(5)) {
        return;
      }
      const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
      BOOST_LOG(info) << "Local AR presenter stats: captured="sv << captured_frames
                      << " presented="sv << presented_frames
                      << " busy_retries="sv << busy_present_retries
                      << " latency_wait_retries="sv << latency_wait_retries
                      << " present_call_retries="sv << present_call_retries
                      << " output_fps="sv << (presented_frames / elapsed_seconds);
      captured_frames = 0;
      presented_frames = 0;
      busy_present_retries = 0;
      latency_wait_retries = 0;
      present_call_retries = 0;
      present_stats_started = now;
    };
    auto push_image = [&](std::shared_ptr<platf::img_t> &&image, bool frame_captured) {
      poll_presenter_state();
      if (stop_token.stop_requested() || window_closed) {
        return false;
      }
      if (config.frame_sequential && (shutter_failed.load(std::memory_order_acquire) || shutter_pairs.failed())) {
        return false;
      }
      if (frame_captured) {
        if (!image) {
          BOOST_LOG(error) << "Local AR capture reported a frame without an image."sv;
          return false;
        }
        retained_presenter_source = std::move(image);
        presenter_retry.observe_source();
        ++capture_attempt_frames;
        if (diagnostics_enabled) {
          ++captured_frames;
        }
      }

      if (!prepare_requested_source() || !converter) {
        return true;
      }
      if (shutter_source && shutter_source->request) {
        const auto &request = shutter_source->request;
        const auto state = request->state.load();
        if (state != switch_state_e::presented &&
            ((state != switch_state_e::requested && state != switch_state_e::adopted) || !switch_is_current(request))) {
          mark_switch_failed(request);
          return true;
        }
      }

      const bool depth_pipeline_ready =
        depth_pipeline_ready_event && depth_pipeline_ready_event->peek();
      const bool conversion_poll_pending = converter.needs_conversion_poll();
      if (!presenter_retry.should_process(
            static_cast<bool>(retained_presenter_source),
            depth_pipeline_ready,
            conversion_poll_pending
          )) {
        return true;
      }

      const auto latest_target_rect = read_target_rect();
      if (latest_target_rect.left != target_rect.left || latest_target_rect.top != target_rect.top || latest_target_rect.right != target_rect.right || latest_target_rect.bottom != target_rect.bottom) {
        const auto latest_width = latest_target_rect.right - latest_target_rect.left;
        const auto latest_height = latest_target_rect.bottom - latest_target_rect.top;
        if (latest_width != output_width || latest_height != output_height) {
          BOOST_LOG(info) << "Local AR target size changed; reinitializing presentation resources."sv;
          presentation_reinit_requested = true;
          return false;
        }
        if (!SetWindowPos(
              window,
              HWND_TOPMOST,
              latest_target_rect.left,
              latest_target_rect.top,
              output_width,
              output_height,
              SWP_NOACTIVATE | SWP_SHOWWINDOW
            )) {
          BOOST_LOG(warning) << "Local AR presenter could not follow the physical sink position: "sv
                             << GetLastError();
          presentation_reinit_requested = true;
          return false;
        }
        target_rect = latest_target_rect;
      }

      if (config.frame_sequential) {
        // Capture/conversion is a producer, never the physical eye clock. If both packets are
        // owned, retain the newest source and service it later rather than queueing GPU work or
        // blocking the independent scanout thread.
        auto &frames = shutter_source->ring;
        const auto source_failure = [&]() {
          const auto &request = shutter_source->request;
          if (request && request->state.load() != switch_state_e::presented) {
            mark_switch_failed(request);
            return true;
          }
          return false;
        };
        auto *slot = frames.acquire_producer();
        if (!slot) {
          return !frames.failed() || source_failure();
        }
        if (depth_pipeline_ready && depth_pipeline_ready_event) {
          depth_pipeline_ready_event->pop(0ms);
        }
        const auto deadline = producer_mode == ::video::SBS_AI ?
                                std::optional {presenter_schedule.begin_frame(std::chrono::steady_clock::now())} :
                                std::nullopt;
        using rgb_conversion_e = local_rgb_converter_t::conversion_e;
        const auto converted = converter.convert_rgb(
          *retained_presenter_source,
          slot->producer_texture.Get(), slot->producer_target.Get(),
          config.hdr, deadline, frames.rgb_destination()
        );
        const bool publish_pair = converted == rgb_conversion_e::converted;
        if (!frames.finish_producer(*slot, publish_pair, converter.device_ctx()) ||
            converted == rgb_conversion_e::failed) {
          BOOST_LOG(error) << "Local shutter monitor failed to publish a complete stereo pair."sv;
          return source_failure();
        }
        if (publish_pair && shutter_source->request && !replacement_published) {
          if (!switch_is_current(shutter_source->request)) {
            mark_switch_failed(shutter_source->request);
            return true;
          }
          shutter_pairs.publish(shutter_source);
          replacement_published = true;
        }
        presenter_retry.record_slot_acquired();
        presenter_retry.record_converted();
        presenter_retry.record_presented();
        presenter_schedule.record_presented();
        return true;
      }

      // Never let a slower physical output back-pressure capture, asynchronous depth processing,
      // or the desktop being interacted with. Retain only the newest source and retry it on the
      // next capture timeout if DWM has not retired the previous flip yet.
      // Acquire one slot for the complete draw/Present transaction. A nonblocking Present retry
      // must not wait for another slot before submitting the frame that owns this one.
      if (presenter_retry.needs_presentation_slot()) {
        const auto frame_latency_status = WaitForSingleObject(frame_latency_waitable, 0);
        if (frame_latency_status == WAIT_FAILED) {
          BOOST_LOG(error) << "Local AR presenter frame-latency wait failed: "sv << GetLastError();
          return false;
        }
        if (frame_latency_status != WAIT_OBJECT_0) {
          if (diagnostics_enabled) {
            ++busy_present_retries;
            ++latency_wait_retries;
            log_present_stats();
          }
          return true;
        }
        presenter_retry.record_slot_acquired();
      }

      // Start the render budget only after DXGI admits this draw. Calls that still own a busy
      // Present retain their original target, including when a newer captured source replaces it.
      const auto presentation_target = producer_mode == ::video::SBS_AI ?
                                         std::optional {presenter_schedule.begin_frame(std::chrono::steady_clock::now())} :
                                         std::nullopt;

      const bool conversion_required = presenter_retry.should_convert(
        static_cast<bool>(retained_presenter_source),
        depth_pipeline_ready,
        conversion_poll_pending
      );
      if (conversion_required) {
        // The swapchain transfer contract controls RGB presentation. In SDR, DWM may composite
        // the G22 swapchain onto an HDR physical output, so the output's live HDR state is not
        // relevant.
        if (depth_pipeline_ready && depth_pipeline_ready_event) {
          // Consume only the notification sampled before should_process(). A build that becomes
          // ready during conversion must remain armed for the next static-source timeout.
          depth_pipeline_ready_event->pop(0ms);
        }
        if (converter.convert_rgb(*retained_presenter_source, backbuffer.Get(), backbuffer_rtv.Get(), config.hdr, presentation_target) !=
            local_rgb_converter_t::conversion_e::converted) {
          BOOST_LOG(error) << "Local AR presenter failed to convert a captured frame."sv;
          return false;
        }
        // A ready/poll-triggered conversion may update a source whose earlier flat output was
        // already presented. Keep this newly rendered backbuffer retryable if Present is busy;
        // only a successful Present clears the owner.
        presenter_retry.record_converted();
      }
      DXGI_PRESENT_PARAMETERS present_parameters {};
      status = swapchain->Present1(
        0,
        DXGI_PRESENT_RESTRICT_TO_OUTPUT | DXGI_PRESENT_DO_NOT_WAIT,
        &present_parameters
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        if (diagnostics_enabled) {
          ++busy_present_retries;
          ++present_call_retries;
          log_present_stats();
        }
        return true;
      }
      if (status == DXGI_ERROR_RESTRICT_TO_OUTPUT_STALE) {
        BOOST_LOG(info) << "Local AR restricted output changed; reinitializing presentation resources."sv;
        presentation_reinit_requested = true;
        return false;
      }
      if (FAILED(status)) {
        BOOST_LOG(error) << "Local AR presenter Present() failed: "sv << util::log_hex(status);
        return false;
      }
      if (diagnostics_enabled) {
        ++presented_frames;
        log_present_stats();
      }
      config.presented_frames->fetch_add(1, std::memory_order_relaxed);
      presenter_retry.record_presented();
      presenter_schedule.record_presented();
      return true;
    };

    const auto capture_backend = display->capture_backend();
    const auto capture_started = std::chrono::steady_clock::now();
    const auto has_pending_work = [&]() {
      const auto request = source_switch ? source_switch->request.load() : nullptr;
      return (config.frame_sequential && (shutter_failed.load(std::memory_order_acquire) || shutter_pairs.failed())) ||
             (request && request != shutter_source->request && request->state.load() == switch_state_e::requested) ||
             (converter && (presenter_retry.should_process(
               static_cast<bool>(retained_presenter_source),
               depth_pipeline_ready_event && depth_pipeline_ready_event->peek(),
               converter.needs_conversion_poll()
             ) || (retained_presenter_source && converter.has_pending_depth_pipeline_build())));
    };
    const auto capture_status = dxgi_display->capture_with_pending_work(
      push_image,
      pull_image,
      &capture_cursor,
      has_pending_work
    );
    shutter_thread.request_stop();
    if (shutter_thread.joinable()) {
      shutter_thread.join();
    }
    // A mode-loss reinit can race an off-head request during graphics destruction. Publish the
    // same stopped proof for every completed capture attempt, before those destructors run.
    close_window();
    destroy_window.disable();
    config.on_quiesced();
    if (config.frame_sequential && (shutter_failed.load(std::memory_order_acquire) || shutter_pairs.failed())) {
      return local_presenter_result_e::error;
    }
    if (config.capture_failover->note_capture_result({
                                     capture_backend,
                                     capture_status,
                                     capture_attempt_frames,
                                     std::chrono::steady_clock::now() - capture_started,
                                     display_epoch_at_open,
                                     platf::display_change_epoch(),
                                   })) {
      BOOST_LOG(warning) << "Local AR Desktop Duplication failed repeatedly before stable capture without a "sv
                            "display-mode change; using Windows.Graphics.Capture for the rest of this session."sv;
    }
    if (presentation_reinit_requested && !stop_token.stop_requested()) {
      return local_presenter_result_e::reinit;
    }
    if ((capture_status == capture_e::reinit || capture_status == capture_e::error) && !stop_token.stop_requested()) {
      BOOST_LOG(info) << "Local AR capture backend stopped; reinitializing the presenter."sv;
      return local_presenter_result_e::reinit;
    }
    if (!stop_token.stop_requested() && !window_closed) {
      BOOST_LOG(warning) << "Local AR capture stopped unexpectedly with status "sv
                         << (int) capture_status << '.';
      return local_presenter_result_e::error;
    }

    BOOST_LOG(info) << "Local AR capture and window input stopped; releasing presentation resources."sv;
    return local_presenter_result_e::stopped;
  }
}  // namespace platf::dxgi

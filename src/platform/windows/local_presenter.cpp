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
#include "local_shutter_slot_timing.h"
#include "misc.h"
#include "open3d_emitter.h"
#include "shutter_eye_pass.h"
#include "shutter_presentation_clock.h"
#include "shutter_slot_timing.h"
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
    const auto switch_is_current = [&](const std::shared_ptr<switch_request_t> &request) {
      return source_switch && request && !stop_token.stop_requested() &&
             request->display_epoch == display_epoch_at_open &&
             request->display_epoch == platf::display_change_epoch() &&
             std::chrono::steady_clock::now() < request->expires &&
             source_switch->mode_is_requested && source_switch->mode_is_requested(request->sbs_mode) &&
             source_switch->request.load() == request;
    };
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

    Microsoft::WRL::ComPtr<ID3D11DeviceContext1> test_pattern_context;

    struct test_pattern_glyph_t {
      std::array<D3D11_RECT, 35> rectangles {};
      UINT count = 0;
    };

    std::array<test_pattern_glyph_t, 2> test_pattern_glyphs {};
    if (shutter_test_pattern) {
      D3D11_FEATURE_DATA_D3D11_OPTIONS options {};
      status = shutter_device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof(options));
      if (SUCCEEDED(status) && options.ClearView) {
        status = shutter_context.As(&test_pattern_context);
      }
      if (FAILED(status) || !options.ClearView || !test_pattern_context) {
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
        auto &glyph = test_pattern_glyphs[eye];
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
    std::jthread shutter_thread;
    if (config.frame_sequential) {
      shutter_thread = std::jthread([&](std::stop_token shutter_stop) {
        auto presentation_scheduling = platf::acquire_presentation_scheduling();
        platf::adjust_thread_priority(platf::thread_priority_e::high);
        const auto stopped = [&]() {
          return shutter_stop.stop_requested() || stop_token.stop_requested() || !IsWindow(window);
        };
        auto fail = [&](std::string_view reason) {
          BOOST_LOG(error) << "Local shutter monitor stopped: "sv << reason;
          if (config.on_failure) {
            config.on_failure(std::string(reason));
          }
          shutter_failed.store(true, std::memory_order_release);
        };
        auto feedback_waiter = platf::create_deadline_waiter();
        if (!feedback_waiter) {
          fail("high-resolution presentation feedback waits are unavailable on this system");
          return;
        }
        std::stop_source emitter_setup_stop;
        const auto cancel_waits = [&]() {
          emitter_setup_stop.request_stop();
          feedback_waiter->notify();
        };
        std::stop_callback cancel_session_setup(stop_token, cancel_waits);
        std::stop_callback cancel_capture_setup(shutter_stop, cancel_waits);
        platf::open3d::emitter_t emitter;
        const auto emitter_eye_period = platf::open3d::eye_period_us(config.target_refresh_millihz, eye_repeats);
        if (!emitter_eye_period) {
          fail("the selected display refresh cannot represent a safe shutter-eye interval");
          return;
        }
        const auto emitter_limits = platf::open3d::timing_limits(config.monitor_config.ir_protocol, config.monitor_config.signal_spacing_us);
        // The configured delay runs from the eye's refresh boundary to the lens opening. The
        // host holds the command for part of it and the emitter counts the rest.
        const auto command_plan = emitter_limits ?
                                    detail::plan_shutter_command(
                                      config.monitor_config.frame_delay_us,
                                      config.monitor_config.frame_duration_us,
                                      config.monitor_config.signal_spacing_us + emitter_limits->min_period_tail_base_us,
                                      emitter_limits->min_delay_us,
                                      *emitter_eye_period
                                    ) :
                                    std::nullopt;
        if (!command_plan) {
          fail("the shutter delay and duration do not fit the selected display refresh");
          return;
        }
        const int command_hold_us = command_plan->hold_us;
        platf::open3d::settings_t emitter_settings {
          .port = config.monitor_config.emitter_port,
          .protocol = config.monitor_config.ir_protocol,
          .frame_delay_us = command_plan->emitter_delay_us,
          .frame_duration_us = config.monitor_config.frame_duration_us,
          .signal_spacing_us = config.monitor_config.signal_spacing_us,
          .flip_eyes = config.monitor_config.flip_eyes,
          .target_frametime_us = *emitter_eye_period,
          .averaging = config.monitor_config.emitter_averaging,
        };
        const auto feedback_budget = platf::open3d::detail::host_feedback_budget_us(emitter_settings);
        if (!feedback_budget) {
          fail("the shutter profile leaves no time for current-eye presentation feedback");
          return;
        }
        // The budget is how late an eye command may leave the host and still open its own
        // window in time. An averaging emitter opens on its own rhythm and ignores a late
        // command, so a long window does not shorten the time the host has to confirm an eye.
        // Half the eye interval stays well short of the lateness the firmware would mistake
        // for a dropped frame, which is a whole interval less a millisecond.
        // A window that reaches into the next eye interval is held back by the host first, so
        // the eye is older when its command leaves; the plan's budget includes that hold.
        const auto host_age_budget_us = emitter_settings.averaging ? std::max(command_plan->host_age_budget_us, *emitter_eye_period / 2) :
                                                                     command_plan->host_age_budget_us;
        BOOST_LOG(info) << "Local shutter command timing: lens opens "sv << config.monitor_config.frame_delay_us
                        << " us after the eye's refresh boundary for "sv << config.monitor_config.frame_duration_us
                        << " us; host holds the command "sv << command_hold_us << " us, emitter delay "sv
                        << emitter_settings.frame_delay_us << " us, eye interval "sv << *emitter_eye_period
                        << " us, window "sv << (command_plan->late_window ? "reaches into the next eye interval"sv : "inside the eye's own interval"sv)
                        << '.';
        if (!emitter.connect(emitter_settings, emitter_setup_stop.get_token())) {
          if (!stopped()) {
            fail("could not configure and verify the selected USB emitter");
          }
          return;
        }
        detail::shutter_presentation_clock_t eye_clock {static_cast<unsigned>(eye_repeats)};
        const auto qpc_now = []() {
          LARGE_INTEGER counter {};
          QueryPerformanceCounter(&counter);
          return counter.QuadPart;
        };
        LARGE_INTEGER qpc_frequency {};
        if (!QueryPerformanceFrequency(&qpc_frequency) || qpc_frequency.QuadPart <= 0) {
          fail("the Windows presentation clock is unavailable");
          return;
        }
        std::unique_ptr<local_shutter_slot_timing_t> slot_timing;
        if (shutter_slot_timing) {
          slot_timing = std::make_unique<local_shutter_slot_timing_t>(qpc_frequency.QuadPart, static_cast<unsigned>(eye_repeats));
        }
        // Cold DWM/MPO transitions can publish stale IDs or switch refresh-counter domains
        // without a disjoint HRESULT. Establish twelve consecutive intervals before any IR.
        constexpr unsigned acquisition_samples = 13;
        constexpr std::int64_t cadence_tolerance_us = 100;
        // Armed images are identified by their exact refresh-count difference. Their timestamp
        // only has to rule out a neighbouring refresh, so allow a quarter of one refresh period
        // of timestamp error, at most 750 us. Measured errors at 330 Hz reach about 620 us.
        const std::int64_t armed_cadence_tolerance_us = std::clamp<std::int64_t>(
          250000000LL / config.target_refresh_millihz,
          cadence_tolerance_us,
          750
        );
        // Measured refresh timestamps lead the query clock by up to about 140 us.
        const std::int64_t armed_future_tolerance_us = std::min<std::int64_t>(armed_cadence_tolerance_us, 250);
        const auto make_feedback_tracker = [&]() {
          return detail::shutter_feedback_tracker_t {
            static_cast<unsigned>(eye_repeats), config.target_refresh_millihz, qpc_frequency.QuadPart,
            host_age_budget_us, cadence_tolerance_us, acquisition_samples, armed_cadence_tolerance_us,
            armed_future_tolerance_us, static_cast<unsigned>(eye_repeats),
          };
        };
        auto feedback_tracker = make_feedback_tracker();
        const auto acquisition_wait = std::chrono::milliseconds {
          std::clamp<std::int64_t>((static_cast<std::int64_t>(*emitter_eye_period) * 3 + 999) / 1000, 25, 250)
        };
        std::optional<std::chrono::steady_clock::time_point> acquisition_deadline;
        unsigned acquisition_presents = 0;
        struct acquisition_sample_t {
          UINT requested = 0;
          detail::shutter_feedback_wait_status_e status {};
          detail::shutter_feedback_t feedback {};
          std::int64_t present_begin = 0, present_end = 0, query = 0, flush_begin = 0, flush_end = 0;
          unsigned streak = 0;
        };
        std::array<acquisition_sample_t, 8> acquisition_history {};
        std::array<unsigned, 6> acquisition_wait_counts {};
        unsigned acquisition_maximum_streak = 0;
        std::uint64_t confirmed_eye_commands = 0, held_eye_images = 0, extra_held_refreshes = 0;
        // An averaging emitter keeps its own rhythm, so a late image leaves the picture whole
        // refreshes behind it. The presenter repays them by showing following eyes for fewer
        // refreshes, which needs no pause. These count the refreshes still owed, the presents
        // of the eye now being shown, and those of the eye before it.
        std::uint64_t absorbed_late_images = 0;
        unsigned owed_refreshes = 0;
        unsigned eye_presents = static_cast<unsigned>(eye_repeats);
        unsigned previous_eye_presents = static_cast<unsigned>(eye_repeats);
        std::uint64_t clock_recoveries = 0;
        std::uint64_t muted_eye_commands = 0;
        std::uint32_t maximum_confirmed_refresh_delta = 0;
        std::optional<detail::shutter_feedback_t> first_feedback;
        bool feedback_disjoint = false;
        HRESULT flush_result = S_FALSE;
        std::int64_t flush_start_qpc = 0;
        std::int64_t flush_end_qpc = 0;
        const detail::shutter_feedback_wait_io_t feedback_io {
          .now = []() {
            return std::chrono::steady_clock::now();
          },
          .cancelled = stopped,
          .query = [&]() {
            DXGI_FRAME_STATISTICS statistics {};
            const auto queried = swapchain->GetFrameStatistics(&statistics);
            const detail::shutter_feedback_t feedback {
              queried == S_OK                                 ? detail::shutter_feedback_status_e::available :
              queried == DXGI_ERROR_FRAME_STATISTICS_DISJOINT ? detail::shutter_feedback_status_e::disjoint :
                                                                detail::shutter_feedback_status_e::unsupported,
              statistics.PresentCount,
              statistics.PresentRefreshCount,
              statistics.SyncRefreshCount,
              statistics.SyncQPCTime.QuadPart,
              static_cast<std::int32_t>(queried),
            };
            feedback_disjoint |= feedback.status == detail::shutter_feedback_status_e::disjoint;
            if (!first_feedback) {
              first_feedback = feedback;
            }
            return feedback;
          },
          // Hardware Flip Queue may require a flush to publish statistics. The wait policy
          // permits this synchronous call only while acquiring with IR quiet; armed frames
          // must never spend their current shutter window blocked inside DwmFlush.
          .flush = [&]() {
            flush_start_qpc = qpc_now();
            flush_result = DwmFlush();
            flush_end_qpc = qpc_now();
            return SUCCEEDED(flush_result);
          },
          .wait_until = [&](auto deadline) {
            return feedback_waiter->wait_until(deadline);
          },
        };
        ID3D11Texture2D *pair = nullptr;
        struct submission_t {
          // The eye's first present: its identity and when it was handed to DXGI.
          std::uint32_t present_id = 0;
          std::int64_t present_begin_qpc = 0, present_end_qpc = 0;
          HRESULT result = S_FALSE, count_result = S_FALSE;
        };
        detail::shutter_ghost_settings_t ghost_settings;
        const auto draw_eye = [&](bool left_eye) {
          if (ghost_settings_file) {
            ghost_settings_file->take(ghost_settings);
          }
          // The settings file's own patterns replace the red and blue one. A picture is
          // corrected only when there is a pair to take the other eye from; if the pass cannot
          // draw, the plain copy below still shows the eye.
          if ((ghost_settings.pattern != 0 || (!shutter_test_pattern && ghost_settings.corrects() && pair)) &&
              ghost_pass.draw(
                shutter_context.Get(),
                backbuffer_rtv.Get(),
                pair,
                left_eye,
                static_cast<std::uint32_t>(output_width),
                static_cast<std::uint32_t>(output_height),
                config.hdr,
                ghost_settings
              )) {
            return;
          }
          if (shutter_test_pattern) {
            // In an HDR swapchain 1.0 is only 80 nits, too dim to judge through a lens that is
            // open for a small part of the time; 6.0 is 480 nits.
            const FLOAT level = config.hdr ? 6.0f : 1.0f;
            const FLOAT red[4] {level, 0.0f, 0.0f, 1.0f};
            const FLOAT blue[4] {0.0f, 0.0f, level, 1.0f};
            const FLOAT white[4] {level, level, level, 1.0f};
            const auto &glyph = test_pattern_glyphs[left_eye ? 0 : 1];
            test_pattern_context->ClearView(backbuffer_rtv.Get(), left_eye ? red : blue, nullptr, 0);
            test_pattern_context->ClearView(backbuffer_rtv.Get(), white, glyph.rectangles.data(), glyph.count);
          } else {
            const UINT left = left_eye ? 0 : static_cast<UINT>(output_width);
            const D3D11_BOX eye_box {left, 0, 0, left + static_cast<UINT>(output_width), static_cast<UINT>(output_height), 1};
            shutter_context->CopySubresourceRegion(backbuffer.Get(), 0, 0, 0, 0, pair, 0, &eye_box);
          }
        };
        // One eye is eye_repeats presents of the same image, each for exactly one refresh. The
        // first identifies the eye and marks its boundary; the others keep it on screen. Every
        // present after a flip-discard needs the image drawn again into the new back buffer.
        // An eye that repays a late image submits fewer presents.
        const auto submit_eye = [&](bool left_eye, unsigned presents) {
          submission_t submission;
          draw_eye(left_eye);
          DXGI_PRESENT_PARAMETERS parameters {};
          submission.present_begin_qpc = qpc_now();
          submission.result = swapchain->Present1(1, DXGI_PRESENT_RESTRICT_TO_OUTPUT, &parameters);
          submission.present_end_qpc = qpc_now();
          if (submission.result == S_OK) {
            submission.count_result = swapchain->GetLastPresentCount(&submission.present_id);
            for (unsigned repeat = 1; repeat < presents && submission.result == S_OK; ++repeat) {
              // The queue is two eyes deep, so a slot is normally free at once. A slot that
              // does not appear means the output stopped retiring images.
              if (WaitForSingleObject(frame_latency_waitable, 100) != WAIT_OBJECT_0) {
                submission.result = HRESULT_FROM_WIN32(ERROR_TIMEOUT);
                break;
              }
              draw_eye(left_eye);
              const auto repeated = swapchain->Present1(1, DXGI_PRESENT_RESTRICT_TO_OUTPUT, &parameters);
              if (repeated != S_OK) {
                // Occlusion is a success code; an eye that is only partly queued is a failure.
                submission.result = FAILED(repeated) ? repeated : E_FAIL;
              }
            }
            if (presents > 1 && submission.result == S_OK && SUCCEEDED(submission.count_result)) {
              UINT last_present = 0;
              const auto last_result = swapchain->GetLastPresentCount(&last_present);
              if (FAILED(last_result) || last_present != static_cast<UINT>(submission.present_id + static_cast<UINT>(presents) - 1u)) {
                // Another present entered the sequence, so the eye's identity stride is unknown.
                submission.count_result = FAILED(last_result) ? last_result : E_UNEXPECTED;
              }
            }
          }
          return submission;
        };
        const auto first_pair_deadline = std::chrono::steady_clock::now() + 5s;
        bool has_displayed = false;
        std::shared_ptr<switch_request_t> confirmed_left_source;
        bool recovering_clock = false;
        std::optional<std::uint32_t> previous_displayed_refresh;
        const auto log_acquisition_history = [&]() {
          log_local_shutter_window_snapshot(window, read_target_rect(), target_output_desc.Monitor);
          BOOST_LOG(error) << "Local shutter acquisition outcomes: presents="sv << acquisition_presents
                           << " maximum_streak="sv << acquisition_maximum_streak
                           << " displayed="sv << acquisition_wait_counts[0] << " cancelled="sv << acquisition_wait_counts[1]
                           << " timeout="sv << acquisition_wait_counts[2] << " unsupported="sv << acquisition_wait_counts[3]
                           << " flush_failed="sv << acquisition_wait_counts[4] << " wait_failed="sv << acquisition_wait_counts[5]
                           << " qpc_frequency="sv << qpc_frequency.QuadPart << " host_age_budget_us="sv << host_age_budget_us << '.';
          const auto count = std::min<std::size_t>(acquisition_presents, acquisition_history.size());
          for (std::size_t i = acquisition_presents - count; i < acquisition_presents; ++i) {
            const auto &sample = acquisition_history[i % acquisition_history.size()];
            BOOST_LOG(error) << "Local shutter acquisition sample: requested="sv << sample.requested
                             << " wait_status="sv << static_cast<unsigned>(sample.status)
                             << " raw_status="sv << static_cast<unsigned>(sample.feedback.status)
                             << " raw_hr="sv << util::log_hex(sample.feedback.result_code)
                             << " present="sv << sample.feedback.present_count << " refresh="sv << sample.feedback.refresh_count
                             << " sync_refresh="sv << sample.feedback.sync_refresh_count << " sync_qpc="sv << sample.feedback.sync_qpc
                             << " query_qpc="sv << sample.query << " present_begin_qpc="sv << sample.present_begin
                             << " present_end_qpc="sv << sample.present_end << " flush_begin_qpc="sv << sample.flush_begin
                             << " flush_end_qpc="sv << sample.flush_end << " streak="sv << sample.streak << '.';
          }
        };
        const auto recovery_context_valid = [&]() {
          if (stopped() || shutter_pairs.failed() || !config.shutter_recovery_allowed ||
              !config.shutter_recovery_allowed() || platf::display_change_epoch() != display_epoch_at_open) {
            return false;
          }
          const auto source = query_local_presenter_display_identity(source_name_wide);
          const auto target = query_local_presenter_display_identity(target_display_name_wide);
          DXGI_OUTPUT_DESC output {};
          Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
          DXGI_OUTPUT_DESC1 color {};
          if (!source || !target || !display_config::equal_device_paths(source->device_path, source_identity_at_start->device_path) ||
              !display_config::equal_device_paths(target->device_path, target_identity_at_start->device_path) ||
              !local_presenter_identity_is_coherent(config.live_target, *source, *target, config.target_adapter_id) ||
              FAILED(target_output->GetDesc(&output)) || output.Monitor != target_output_desc.Monitor ||
              !EqualRect(&output.DesktopCoordinates, &target_output_desc.DesktopCoordinates) ||
              FAILED(target_output.As(&output6)) || FAILED(output6->GetDesc1(&color)) ||
              color.ColorSpace != target_desc1.ColorSpace || color.BitsPerColor != target_desc1.BitsPerColor ||
              FAILED(shutter_device->GetDeviceRemovedReason())) {
            return false;
          }
          return !stopped();
        };
        const auto recover_clock = [&](std::string_view reason) {
          if (!recovery_context_valid()) {
            BOOST_LOG(warning) << "Local shutter clock recovery rejected a changed display, device or desktop context."sv;
            return false;
          }
          ++clock_recoveries;
          BOOST_LOG(warning) << "Local shutter clock lost; holding presentation while the emitter drains: "sv << reason;
          if (slot_timing) {
            slot_timing->recovering();
          }
          // Present1 may have displayed the new eye even when its metadata was rejected. Freeze
          // NEW image submissions here, rather than repainting while an old USB write or finite
          // OPEN/CLOSE timer may still be active. The presents already queued for the current
          // eye still appear during this drain.
          if (!emitter.quiesce(emitter_setup_stop.get_token())) {
            BOOST_LOG(warning) << "Local shutter clock recovery could not verify a quiet USB emitter."sv;
            return false;
          }
          if (!recovery_context_valid()) {
            BOOST_LOG(warning) << "Local shutter clock recovery context changed while the emitter drained."sv;
            return false;
          }
          // A failed armed tracker is sticky. Only a successful same-handle FIFO barrier and
          // firmware drain authorize a wholly new clock; never repair IDs or infer eye parity.
          feedback_tracker = make_feedback_tracker();
          eye_clock = detail::shutter_presentation_clock_t {static_cast<unsigned>(eye_repeats)};
          owed_refreshes = 0;
          eye_presents = previous_eye_presents = static_cast<unsigned>(eye_repeats);
          has_displayed = false;
          confirmed_left_source.reset();
          recovering_clock = true;
          previous_displayed_refresh.reset();
          acquisition_presents = 0;
          acquisition_history = {};
          acquisition_wait_counts = {};
          acquisition_maximum_streak = 0;
          acquisition_deadline = std::chrono::steady_clock::now() + detail::shutter_acquisition_timeout;
          BOOST_LOG(info) << "Local shutter emitter quiet; reacquiring an unsignalled left-eye clock without changing the desktop."sv;
          return true;
        };
        while (!stopped()) {
          if (!emitter.healthy() || shutter_pairs.failed()) {
            fail("USB emitter or shared stereo packet became unavailable");
            break;
          }
          const bool acquiring = !feedback_tracker.armed();
          const bool left_eye = eye_clock.left_eye();
          // The current packet stays keyed-locked through both eyes, including all repeated
          // refreshes. Publishing another capture can never change the second eye of this pair.
          if (!pair || (has_displayed && eye_clock.starts_pair())) {
            pair = shutter_pairs.acquire_pair([&](const std::shared_ptr<switch_request_t> &request) {
              if (acquiring || !left_eye || !switch_is_current(request)) {
                return false;
              }
              auto expected = switch_state_e::requested;
              return request->state.compare_exchange_strong(expected, switch_state_e::adopted);
            });
          }
          if (!pair) {
            if (has_displayed) {
              fail("the shared stereo frame became unavailable during presentation");
              break;
            }
            if (std::chrono::steady_clock::now() >= first_pair_deadline) {
              fail("no stereo frame became available for monitor startup");
              break;
            }
            std::this_thread::sleep_for(1ms);
            continue;
          }
          if (acquiring) {
            if (!acquisition_deadline) {
              acquisition_deadline = std::chrono::steady_clock::now() + detail::shutter_acquisition_timeout;
            }
            if (std::chrono::steady_clock::now() >= *acquisition_deadline) {
              log_acquisition_history();
              BOOST_LOG(error) << "Local shutter clock acquisition expired: presents="sv << acquisition_presents
                               << " stable_samples="sv << feedback_tracker.streak() << '.';
              fail("DXGI could not establish a stable monitor clock before starting the glasses; returning to 2D");
              break;
            }
          }
          const auto slot_deadline = std::min(
            std::chrono::steady_clock::now() + std::max(50ms, acquisition_wait),
            acquiring ? *acquisition_deadline : std::chrono::steady_clock::time_point::max()
          );
          DWORD latency = WAIT_TIMEOUT;
          while (!stopped() && std::chrono::steady_clock::now() < slot_deadline) {
            const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(slot_deadline - std::chrono::steady_clock::now());
            latency = WaitForSingleObject(frame_latency_waitable, static_cast<DWORD>(std::clamp<std::int64_t>(remaining.count(), 1, 50)));
            if (latency != WAIT_TIMEOUT) {
              break;
            }
          }
          if (latency != WAIT_OBJECT_0) {
            if (!stopped()) {
              if (acquiring) {
                // The acquisition deadline also bounds this wait, so an expired acquisition
                // can end here. Record what the output reported before giving up.
                log_acquisition_history();
                BOOST_LOG(error) << "Local shutter clock acquisition ended waiting for a presentation slot: presents="sv
                                 << acquisition_presents << " stable_samples="sv << feedback_tracker.streak()
                                 << " wait_result="sv << latency << '.';
              }
              fail("the physical output did not admit a bounded presentation slot");
            }
            break;
          }
          const auto logical_slot = left_eye ? 0u : 1u;
          std::optional<local_shutter_slot_timing_t::sample_t> timing_sample;
          if (slot_timing) {
            timing_sample.emplace();
            timing_sample->logical_slot = logical_slot;
            timing_sample->acquiring = acquiring;
            timing_sample->segment = slot_timing->segment;
            timing_sample->previous = slot_timing->previous;
          }
          const auto record_slot_timing = util::fail_guard([&]() {
            if (timing_sample) {
              slot_timing->finish(*timing_sample);
            }
          });
          const auto set_timing_outcome = [&](local_shutter_slot_timing_t::outcome_e outcome) {
            if (timing_sample) {
              timing_sample->outcome = outcome;
            }
          };
          first_feedback.reset();
          feedback_disjoint = false;
          flush_result = S_FALSE;
          flush_start_qpc = flush_end_qpc = 0;
          if (stopped()) {
            break;
          }
          eye_presents = static_cast<unsigned>(eye_repeats);
          if (owed_refreshes != 0 && has_displayed && eye_presents > 1) {
            // Keep at least one present, which identifies the eye.
            const unsigned repaid = std::min(owed_refreshes, eye_presents - 1);
            eye_presents -= repaid;
            owed_refreshes -= repaid;
          }
          const auto submission = submit_eye(left_eye, eye_presents);
          const auto present_start_qpc = submission.present_begin_qpc;
          const auto present_end_qpc = submission.present_end_qpc;
          const auto presented = submission.result;
          if (slot_timing) {
            timing_sample->present_begin = present_start_qpc;
            timing_sample->present_end = present_end_qpc;
          }
          if (presented != S_OK) {
            // Occlusion is not a successful scanout. Never continue emitting an alternating eye
            // sequence while Windows is showing some other image on the selected monitor.
            fail("the restricted swapchain was occluded, stale, or could not present");
            break;
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::present_id_failed);
          const auto requested_present = submission.present_id;
          const auto present_count_result = submission.count_result;
          if (slot_timing) {
            timing_sample->requested = requested_present;
          }
          if (FAILED(present_count_result)) {
            BOOST_LOG(error) << "Local shutter present identity failure: Present1="sv << util::log_hex(presented)
                             << " GetLastPresentCount="sv << util::log_hex(present_count_result) << '.';
            fail("DXGI could not identify the submitted eye frame");
            break;
          }
          const auto feedback_timeout = acquiring ?
                                          std::min(acquisition_wait, std::max(0ms,
                                            std::chrono::duration_cast<std::chrono::milliseconds>(*acquisition_deadline - std::chrono::steady_clock::now()))) :
                                          250ms;
          const auto confirmation = detail::await_shutter_present(feedback_io, requested_present, acquiring, feedback_timeout);
          const auto confirmation_qpc = qpc_now();
          if (slot_timing) {
            timing_sample->receipt = confirmation_qpc;
            timing_sample->feedback = confirmation.feedback;
            timing_sample->outcome = local_shutter_slot_timing_t::outcome_e::feedback_unconfirmed;
          }
          const auto log_feedback_failure = [&]() {
            // Capture all HWND/session evidence before notifying the controller to restore.
            log_local_shutter_window_snapshot(window, read_target_rect(), target_output_desc.Monitor);
            const auto first = first_feedback.value_or(detail::shutter_feedback_t {});
            const auto &last = confirmation.feedback;
            UINT latest_submitted = 0;
            const auto latest_count_result = swapchain->GetLastPresentCount(&latest_submitted);
            constexpr std::array status_names {"displayed", "cancelled", "timeout", "unsupported", "flush_failed", "wait_failed"};
            BOOST_LOG(error) << "Local shutter scanout feedback: result="sv
                             << status_names[static_cast<std::size_t>(confirmation.status)]
                             << " requested="sv << requested_present << " latest_submitted="sv << latest_submitted
                             << " first_eye="sv << !has_displayed << " left="sv << left_eye << " repeats="sv << eye_repeats
                             << " Present1="sv << util::log_hex(presented)
                             << " count_hr="sv << util::log_hex(present_count_result)
                             << " latest_count_hr="sv << util::log_hex(latest_count_result)
                             << " queries="sv << confirmation.queries << " flushes="sv << confirmation.flushes << " waits="sv << confirmation.waits
                             << " first_hr="sv << util::log_hex(first.result_code)
                             << " first_present="sv << first.present_count << " first_refresh="sv << first.refresh_count
                             << " first_sync_refresh="sv << first.sync_refresh_count << " first_sync_qpc="sv << first.sync_qpc
                             << " last_hr="sv << util::log_hex(last.result_code)
                             << " last_present="sv << last.present_count << " last_refresh="sv << last.refresh_count
                             << " last_sync_refresh="sv << last.sync_refresh_count << " last_sync_qpc="sv << last.sync_qpc
                             << " previous_displayed_refresh="sv << previous_displayed_refresh.value_or(0)
                             << " qpc_frequency="sv << qpc_frequency.QuadPart
                             << " present_start_qpc="sv << present_start_qpc << " present_end_qpc="sv << present_end_qpc
                             << " confirmation_qpc="sv << confirmation_qpc
                             << " acquiring="sv << acquiring << " stable_samples="sv << feedback_tracker.streak()
                             << " host_age_budget_us="sv << host_age_budget_us
                             << " flush_hr="sv << util::log_hex(flush_result)
                             << " flush_start_qpc="sv << flush_start_qpc << " flush_end_qpc="sv << flush_end_qpc << '.';

            // Read-only, failure-only probes distinguish composition feedback from a stale or
            // wrong output. They never authorize an eye or replace the confirmed-present gate.
            BOOL fullscreen = FALSE;
            const auto fullscreen_result = swapchain->GetFullscreenState(&fullscreen, nullptr);
            Microsoft::WRL::ComPtr<IDXGIOutput> containing_output;
            DXGI_OUTPUT_DESC containing_desc {};
            auto containing_result = swapchain->GetContainingOutput(&containing_output);
            if (SUCCEEDED(containing_result)) {
              containing_result = containing_output->GetDesc(&containing_desc);
            }
            Microsoft::WRL::ComPtr<IDXGISwapChainMedia> media;
            DXGI_FRAME_STATISTICS_MEDIA media_stats {};
            auto media_result = swapchain.As(&media);
            if (SUCCEEDED(media_result)) {
              media_result = media->GetFrameStatisticsMedia(&media_stats);
            }
            BOOST_LOG(error) << "Local shutter scanout context: target="sv << target_display_name
                             << " containing="sv << platf::to_utf8(containing_desc.DeviceName)
                             << " containing_hr="sv << util::log_hex(containing_result)
                             << " fullscreen="sv << fullscreen << " fullscreen_hr="sv << util::log_hex(fullscreen_result)
                             << " format="sv << swapchain_desc.Format << " swap_effect="sv << swapchain_desc.SwapEffect
                             << " flags="sv << swapchain_desc.Flags << " present_flags="sv << DXGI_PRESENT_RESTRICT_TO_OUTPUT
                             << " hdr="sv << config.hdr << " media_hr="sv << util::log_hex(media_result)
                             << " media_present="sv << media_stats.PresentCount << " media_refresh="sv << media_stats.PresentRefreshCount
                             << " media_sync_refresh="sv << media_stats.SyncRefreshCount << " media_sync_qpc="sv << media_stats.SyncQPCTime.QuadPart
                             << " composition_mode="sv << media_stats.CompositionMode << '.';
          };
          if (confirmation.status == detail::shutter_feedback_wait_status_e::cancelled) {
            set_timing_outcome(local_shutter_slot_timing_t::outcome_e::feedback_cancelled);
            break;
          }
          if (acquiring) {
            set_timing_outcome(local_shutter_slot_timing_t::outcome_e::cold_unsignalled);
            ++acquisition_presents;
            if (recovering_clock && confirmation.status != detail::shutter_feedback_wait_status_e::displayed &&
                !detail::shutter_feedback_is_timing_interruption(confirmation) &&
                !detail::shutter_feedback_superseded_within_eye(confirmation, requested_present, eye_presents)) {
              log_feedback_failure();
              fail("presentation feedback failed while reacquiring the shutter clock");
              break;
            }
            if (confirmation.status != detail::shutter_feedback_wait_status_e::displayed || feedback_disjoint) {
              // A timeout's last raw sample is diagnostic data, never display authorization.
              feedback_tracker.observe(requested_present, detail::shutter_feedback_t {}, confirmation_qpc);
            }
            const auto acquisition_status = confirmation.status == detail::shutter_feedback_wait_status_e::displayed ?
                                              feedback_tracker.observe(requested_present, confirmation.feedback, confirmation_qpc) :
                                              detail::shutter_feedback_tracker_status_e::acquiring;
            ++acquisition_wait_counts[static_cast<std::size_t>(confirmation.status)];
            acquisition_maximum_streak = std::max(acquisition_maximum_streak, feedback_tracker.streak());
            acquisition_history[(acquisition_presents - 1) % acquisition_history.size()] = {
              requested_present, confirmation.status, confirmation.feedback,
              present_start_qpc, present_end_qpc, confirmation_qpc, flush_start_qpc, flush_end_qpc, feedback_tracker.streak(),
            };
            if (acquisition_status == detail::shutter_feedback_tracker_status_e::ready) {
              if (recovering_clock && !recovery_context_valid()) {
                if (!stopped()) {
                  fail("the display or Windows desktop changed while reacquiring the shutter clock");
                }
                break;
              }
              if (!feedback_tracker.arm()) {
                fail("the acquired shutter clock could not be armed");
                break;
              }
              recovering_clock = false;
              BOOST_LOG(info) << "Local shutter monitor clock acquired: unsignalled_presents="sv << acquisition_presents
                              << " stable_intervals="sv << acquisition_samples - 1
                              << " host_age_budget_us="sv << host_age_budget_us
                              << " cadence_tolerance_us="sv << cadence_tolerance_us
                              << " armed_cadence_tolerance_us="sv << armed_cadence_tolerance_us
                              << " armed_future_tolerance_us="sv << armed_future_tolerance_us << '.';
            }
            // Hold the same left image and immutable pair while acquiring. The first eye must
            // then continue this exact clock; startup never sends an emitter eye command.
            continue;
          }
          if (confirmation.status != detail::shutter_feedback_wait_status_e::displayed) {
            log_feedback_failure();
            if (detail::shutter_feedback_is_timing_interruption(confirmation) &&
                recover_clock("presentation feedback interrupted"sv)) {
              continue;
            }
            if (detail::shutter_feedback_superseded_within_eye(confirmation, requested_present, eye_presents) &&
                recover_clock("the eye's first present was superseded before its feedback was read"sv)) {
              continue;
            }
            if (stopped()) {
              break;
            }
            fail("DXGI scanout feedback is unavailable, late, or invalid on this display configuration; returning to 2D");
            break;
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::tracker_rejected);
          if (feedback_tracker.observe(requested_present, confirmation.feedback, confirmation_qpc, previous_eye_presents) != detail::shutter_feedback_tracker_status_e::accepted) {
            log_feedback_failure();
            if (recover_clock("displayed-eye identity, cadence or freshness changed"sv)) {
              continue;
            }
            if (stopped()) {
              break;
            }
            fail("the displayed eye lost its acquired refresh clock or arrived too late for the shutter window");
            break;
          }
          if (slot_timing) {
            timing_sample->boundary = feedback_tracker.last_present_qpc().value_or(0);
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::eye_clock_rejected);
          if (!eye_clock.observe_displayed(confirmation.feedback.refresh_count, previous_eye_presents)) {
            log_feedback_failure();
            if (recover_clock("eye scanout cadence interrupted"sv)) {
              continue;
            }
            if (stopped()) {
              break;
            }
            fail("DXGI could not prove an uninterrupted eye scanout cadence; restart after checking display timing");
            break;
          }
          const auto confirmed_refresh_delta = previous_displayed_refresh ?
                                                 static_cast<std::uint32_t>(confirmation.feedback.refresh_count - *previous_displayed_refresh) :
                                                 static_cast<std::uint32_t>(eye_repeats);
          if (slot_timing) {
            timing_sample->delta_known = previous_displayed_refresh.has_value();
            timing_sample->refresh_delta = confirmed_refresh_delta;
          }
          previous_displayed_refresh = confirmation.feedback.refresh_count;
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::expired_before_emit);
          if (const auto boundary = feedback_tracker.last_present_qpc(); boundary && command_hold_us > 0) {
            const auto send_at = *boundary + command_hold_us * qpc_frequency.QuadPart / 1000000;
            // The usual hold is too short to sleep: the scheduler's granularity is longer than
            // the whole of it. A hold of milliseconds, for a window that reaches into the next
            // eye interval, sleeps on the high-resolution timer and spins only at the end.
            const auto spin_from = send_at - 1000 * qpc_frequency.QuadPart / 1000000;
            while (!stopped()) {
              const auto remaining = std::chrono::microseconds {(spin_from - qpc_now()) * 1000000 / qpc_frequency.QuadPart};
              if (remaining < 300us || !feedback_waiter->wait_until(std::chrono::steady_clock::now() + remaining)) {
                break;
              }
            }
            while (qpc_now() < send_at) {
              YieldProcessor();
            }
          }
          if (!feedback_tracker.fresh_for_emit(qpc_now())) {
            log_feedback_failure();
            if (recover_clock("displayed eye expired before the USB command"sv)) {
              continue;
            }
            if (stopped()) {
              break;
            }
            fail("the displayed eye became too old before its USB shutter command");
            break;
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::cancelled_before_emit);
          if (stopped()) {
            break;
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::signal_failed);
          if (slot_timing) {
            timing_sample->signal_begin = qpc_now();
          }
          auto *write_timing = timing_sample ? &timing_sample->write : nullptr;
          const bool slot_muted = shutter_muted_slot && *shutter_muted_slot == logical_slot;
          if (slot_muted) {
            ++muted_eye_commands;
            if (timing_sample) {
              // Nothing is written for a muted slot, so there is no write to time.
              timing_sample->signal_begin = 0;
            }
          }
          const bool signalled = slot_muted || emitter.signal(left_eye, write_timing);
          if (slot_timing) {
            timing_sample->signal_end = qpc_now();
          }
          if (!signalled) {
            fail("the USB emitter missed an eye boundary");
            break;
          }
          set_timing_outcome(local_shutter_slot_timing_t::outcome_e::signalled);
          if (!left_eye && has_displayed && !shutter_pair_presented.load(std::memory_order_relaxed)) {
            shutter_pair_presented.store(true, std::memory_order_release);
          }
          if (const auto &request = shutter_pairs.current_request(); request && request->state.load() == switch_state_e::adopted) {
            if (left_eye) {
              confirmed_left_source = request;
            } else if (request == confirmed_left_source && switch_is_current(request)) {
              auto expected = switch_state_e::adopted;
              const auto previous_mode = source_switch->presented_mode.exchange(request->sbs_mode);
              if (!request->state.compare_exchange_strong(expected, switch_state_e::presented)) {
                // Cancellation won the acknowledgement race. A later attempt starts from
                // the last confirmed mode after this thread has joined.
                source_switch->presented_mode.store(previous_mode);
              }
              confirmed_left_source.reset();
            }
          }
          ++confirmed_eye_commands;
          maximum_confirmed_refresh_delta = std::max(maximum_confirmed_refresh_delta, confirmed_refresh_delta);
          // The eye before this one was meant to last previous_eye_presents refreshes. Elapsed
          // time decides how long it did last; Windows' refresh counter can trail it by one.
          const auto elapsed_refresh_delta = std::max(confirmed_refresh_delta, feedback_tracker.interval_refreshes());
          const bool late_image = has_displayed && elapsed_refresh_delta > previous_eye_presents;
          const unsigned late_refreshes = late_image ? elapsed_refresh_delta - previous_eye_presents : 0;
          if (late_image) {
            ++held_eye_images;
            extra_held_refreshes += late_refreshes;
          }
          has_displayed = true;
          previous_eye_presents = eye_presents;
          config.presented_frames->fetch_add(eye_presents, std::memory_order_relaxed);
          // A late image moves every later eye by whole refreshes. An averaging emitter keeps
          // its earlier rhythm and its firmware ignores so large a correction, so the lenses
          // would stay out of step with the picture. Repay the refreshes by shortening the
          // following eyes; if that is not possible or too many are owed, start the rhythm again.
          if (late_image && emitter_settings.averaging && eye_repeats > 1 &&
              owed_refreshes + late_refreshes <= 2u * static_cast<unsigned>(eye_repeats)) {
            owed_refreshes += late_refreshes;
            ++absorbed_late_images;
          } else if (late_image && emitter_settings.averaging) {
            if (recover_clock("a late image moved the eye rhythm away from the averaging emitter"sv)) {
              continue;
            }
            if (stopped()) {
              break;
            }
            fail("the averaging emitter could not be restarted after a late image");
            break;
          }
        }
        BOOST_LOG(info) << "Local shutter presentation summary: confirmed_eye_commands="sv << confirmed_eye_commands
                        << " held_eye_images="sv << held_eye_images << " extra_held_refreshes="sv << extra_held_refreshes
                        << " absorbed_late_images="sv << absorbed_late_images
                        << " clock_recoveries="sv << clock_recoveries
                        << " maximum_refresh_delta="sv << maximum_confirmed_refresh_delta << " repeats="sv << eye_repeats << '.';
        if (shutter_muted_slot) {
          BOOST_LOG(info) << "Local shutter single-slot diagnostic summary: muted_logical_slot="sv << (*shutter_muted_slot == 0u ? 'L' : 'R')
                          << " skipped_usb_commands="sv << muted_eye_commands
                          << " sent_usb_commands="sv << confirmed_eye_commands - muted_eye_commands << '.';
        }
        // PC serial mode uses finite pulses with averaging disabled. Closing the transport
        // leaves no autonomous eye-toggle source running after the last pulse expires.
        emitter.close();
        shutter_context->Flush();
        if (slot_timing) {
          slot_timing->log_summary(config.monitor_config.flip_eyes, host_age_budget_us);
        }
      });
    }

    BOOST_LOG(info) << "Local AR presentation started: source="sv << config.source_display_name
                    << " target="sv << target_display_name << " ["sv << target_rect.left << ','
                    << target_rect.top << "] "sv << output_width << 'x' << output_height << '@'
                    << capture_config.framerate << " mode="sv
                    << (config.sbs_mode == ::video::SBS_AI ? "SBS AI"sv : "normal"sv)
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

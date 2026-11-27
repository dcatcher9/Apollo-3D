/**
 * @file src/platform/windows/local_shutter_session.h
 * @brief The shutter monitor's presentation thread: its emitter connection, its eye clock and
 *        the timed USB command for every eye it presents.
 */
#pragma once

// standard includes
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>

// platform includes
#include <d3d11_4.h>
#include <wrl/client.h>

// local includes
#include "display.h"
#include "local_presenter_window.h"
#include "local_shutter_frame_ring.h"
#include "shutter_eye_pass.h"
#include "src/platform/common.h"

namespace platf::dxgi {
  /** Internals of run_local_presenter(), for local_presenter.cpp only. */
  namespace local_presenter {
    /**
     * Whether a source change is still one this presenter attempt may show: the attempt has not
     * been stopped, the request has not expired, it was made in the display epoch the attempt
     * opened and no display change has followed, its mode is still wanted, and it is still the
     * request being made. Capture and the shutter thread ask the same question.
     */
    struct local_monitor_switch_is_current_t {
      const std::shared_ptr<local_monitor_source_switch_t> &source_switch;
      const std::stop_token &stop_token;
      std::uint64_t display_epoch_at_open;

      bool operator()(const std::shared_ptr<local_monitor_source_switch_t::request_t> &request) const {
        return source_switch && request && !stop_token.stop_requested() &&
               request->display_epoch == display_epoch_at_open &&
               request->display_epoch == platf::display_change_epoch() &&
               std::chrono::steady_clock::now() < request->expires &&
               source_switch->mode_is_requested && source_switch->mode_is_requested(request->sbs_mode) &&
               source_switch->request.load() == request;
      }
    };

    /** The calibration pattern's large L and R labels: five by seven cells, centred on the eye. */
    struct local_shutter_test_pattern_t {
      struct glyph_t {
        std::array<D3D11_RECT, 35> rectangles {};
        UINT count = 0;
      };

      Microsoft::WRL::ComPtr<ID3D11DeviceContext1> context;
      std::array<glyph_t, 2> glyphs {};
    };

    /**
     * What the shutter thread uses of run_local_presenter(), named here instead of captured.
     *
     * Everything is borrowed. run_local_presenter() builds this before it starts the shutter
     * thread and joins the thread before it destroys any of the objects referred to, so every
     * reference outlives the thread. None of these values changes while the thread runs. The
     * thread shares live state only through what is made for it: the stereo-pair hand-off, the
     * two flags and the source-switch request with capture, and the config's live target (read
     * under its lock), presented-frame counter and failure callback with the session owner.
     */
    struct local_shutter_presenter_t {
      const local_presenter_config_t &config;
      const std::stop_token &session_stop;  ///< run_local_presenter()'s own stop token.
      HWND window;
      int eye_repeats;

      // Shared with capture.
      local_shutter_pair_handoff_t &pairs;
      std::atomic<bool> &failed;  ///< Set when the thread stops presentation on a fault.
      std::atomic<bool> &pair_presented;  ///< Set once the thread has shown a whole pair.
      const std::shared_ptr<local_monitor_source_switch_t> &source_switch;
      const local_monitor_switch_is_current_t &switch_is_current;

      // The physical output. Only the shutter thread presents to it.
      const Microsoft::WRL::ComPtr<IDXGISwapChain1> &swapchain;
      const DXGI_SWAP_CHAIN_DESC1 &swapchain_desc;
      HANDLE frame_latency_waitable;
      const Microsoft::WRL::ComPtr<ID3D11Device> &device;
      const Microsoft::WRL::ComPtr<ID3D11DeviceContext> &context;
      const Microsoft::WRL::ComPtr<ID3D11Texture2D> &backbuffer;
      const Microsoft::WRL::ComPtr<ID3D11RenderTargetView> &backbuffer_rtv;
      int output_width;
      int output_height;

      // What the eyes show instead of, or besides, the plain pair.
      bool test_pattern;
      const local_shutter_test_pattern_t &test_pattern_labels;
      detail::shutter_eye_pass_t &ghost_pass;
      std::optional<detail::shutter_ghost_settings_file_t> &ghost_settings_file;

      // Diagnostics, sampled once for this presenter initialization.
      bool slot_timing;
      std::optional<unsigned> muted_slot;

      // What a clock recovery must find unchanged, and what the failure logs name.
      std::uint64_t display_epoch_at_open;
      const std::wstring &source_name;
      const std::wstring &target_name;
      const std::string &target_display_name;
      const std::optional<local_presenter_display_identity_t> &source_identity_at_start;
      const std::optional<local_presenter_display_identity_t> &target_identity_at_start;
      const Microsoft::WRL::ComPtr<IDXGIOutput> &target_output;
      const DXGI_OUTPUT_DESC &target_output_desc;
      const DXGI_OUTPUT_DESC1 &target_desc1;
    };

    /**
     * The shutter thread's body. It connects the emitter, holds one left-eye image while it
     * acquires the output's clock, then presents eyes and sends each eye's command at its time
     * until presentation stops, recovering the clock quietly where it can. On a fault it sets
     * presenter.failed and reports through config.on_failure.
     */
    void run_local_shutter_presentation(const local_shutter_presenter_t &presenter, std::stop_token shutter_stop);
  }  // namespace local_presenter
}  // namespace platf::dxgi

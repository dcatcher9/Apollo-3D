/**
 * @file src/platform/windows/local_rgb_converter.h
 * @brief The production RGB converter, as a local presenter owns and drives it.
 */
#pragma once

// standard includes
#include <chrono>
#include <memory>
#include <optional>

// platform includes
#include <d3d11.h>

// local includes
#include "display.h"
#include "src/config.h"
#include "src/thread_safe.h"

namespace platf::dxgi {
  class d3d_base_encode_device;

  /**
   * One production RGB converter of a local presenter.
   *
   * The converter is the one a streaming encoder uses, private to display_vram.cpp. Local AR and
   * the shutter monitor open it on their capture display and convert each captured image into an
   * RGB target of their own. Every call goes straight to the converter.
   */
  class local_rgb_converter_t {
  public:
    enum class conversion_e {
      failed,
      converted,
      retained
    };
    enum class destination_e {
      backbuffer,
      shutter,
      shutter_with_pair
    };

    /** No converter. */
    local_rgb_converter_t() noexcept;
    local_rgb_converter_t(local_rgb_converter_t &&) noexcept;
    local_rgb_converter_t &operator=(local_rgb_converter_t &&) noexcept;
    ~local_rgb_converter_t();

    /** A new converter, not yet initialized. */
    static local_rgb_converter_t create();

    explicit operator bool() const noexcept {
      return static_cast<bool>(converter_);
    }

    /** Destroy the converter. */
    void reset() noexcept;

    /** Open the converter's device on the capture display's adapter. Zero on success. */
    int init(std::shared_ptr<platf::display_t> display, adapter_t::pointer adapter, pix_fmt_e pix_fmt);

    /** Set the converter up for RGB output of this packed size and mode. Zero on success. */
    int init_rgb_output(
      int width,
      int height,
      int sbs_mode,
      const config::video_t::sbs_t &settings,
      std::shared_ptr<safe::event_t<bool>> depth_pipeline_ready_event
    );

    [[nodiscard]] ID3D11Device *device() const noexcept;
    [[nodiscard]] ID3D11DeviceContext *device_ctx() const noexcept;

    /** Whether the converter has work to show without a new capture. */
    [[nodiscard]] bool needs_conversion_poll() const;

    [[nodiscard]] bool has_pending_depth_pipeline_build() const;

    /** Only a shutter sink with a published pair from this converter can retain RGB output. */
    conversion_e convert_rgb(
      platf::img_t &img,
      ID3D11Texture2D *target_texture,
      ID3D11RenderTargetView *target,
      bool target_is_linear,
      std::optional<std::chrono::steady_clock::time_point> presentation_target,
      destination_e destination = destination_e::backbuffer
    );

  private:
    std::unique_ptr<d3d_base_encode_device> converter_;
  };
}  // namespace platf::dxgi

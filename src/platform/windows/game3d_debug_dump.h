// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "foreground_window_region.h"
#include "src/game3d_debug_protocol.h"

#include <atomic>
#include <d3d11.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace platf::game3d_debug {
  namespace detail {
    std::uint32_t bytes_per_pixel(std::uint32_t format) noexcept;
    // Validates all lengths before any producer handle or payload is consumed.
    bool validate_response(const ::game3d_debug::response_t &response, std::uint64_t nonce, std::string &reason);
  }  // namespace detail

  class dumper {
  public:
    using observer_t = std::function<foreground_window::observation_t()>;
    explicit dumper(observer_t observe = foreground_window::sample, std::filesystem::path output_directory = {});
    ~dumper();
    dumper(const dumper &) = delete;
    dumper &operator=(const dumper &) = delete;

    void set_button_request(std::shared_ptr<std::atomic<bool>> request);
    // All GPU work stays on the caller's immediate-context thread. Publication owns only bytes.
    void poll(ID3D11Device *device, ID3D11DeviceContext *context, RECT expected_fullscreen, int source_width, int source_height, int wire_mode);
    void cancel() noexcept;
    bool needs_conversion_poll() const noexcept;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::game3d_debug

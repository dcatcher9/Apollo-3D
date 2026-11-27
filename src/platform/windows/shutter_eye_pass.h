/**
 * @file src/platform/windows/shutter_eye_pass.h
 * @brief Draws one eye of a packed pair for the local shutter monitor and takes out the other
 *        eye's picture where the lens is open while the screen is being redrawn.
 *
 * This is an experiment that is off unless a settings file is named (see display_vram.cpp).
 * The arithmetic and its reason are in local_shutter_eye_ps.hlsl.
 */
#pragma once

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

// platform includes
#include <d3d11.h>
#include <wrl/client.h>

namespace platf::dxgi::detail {
  /**
   * How much of the other eye's picture each screen line carries, and what to do about it.
   * A peak is the share of the lens window at the outermost line; it falls to nothing over the
   * extent, which is a part of the screen height.
   */
  struct shutter_ghost_settings_t {
    float top_peak = 0.0f;
    float top_extent = 0.0f;
    float bottom_peak = 0.0f;
    float bottom_extent = 0.0f;
    /// 0 leaves black alone; 1 raises it enough to cancel white beside black.
    float black_raise = 0.0f;
    /// Brightness of white in an HDR picture. An SDR picture has its own white.
    float white_nits = 200.0f;
    /// 0 draws the pair, 1 the balance pattern, 2 the bars pattern.
    int pattern = 0;
    /// The balance pattern is judged through the left eye.
    bool view_left = false;

    [[nodiscard]] constexpr bool corrects() const noexcept {
      return (top_peak > 0.0f && top_extent > 0.0f) || (bottom_peak > 0.0f && bottom_extent > 0.0f);
    }

    /// False when a plain copy of the eye gives the same picture.
    [[nodiscard]] constexpr bool needs_pass() const noexcept {
      return pattern != 0 || corrects();
    }

    bool operator==(const shutter_ghost_settings_t &) const = default;
  };

  inline constexpr float shutter_ghost_largest_peak = 0.45f;
  inline constexpr float shutter_ghost_largest_extent = 0.5f;

  /**
   * Reads "name = value" lines. '#' starts a comment. Values outside their range are brought
   * inside it. Lines that cannot be used are described in `problems` and otherwise ignored.
   */
  [[nodiscard]] inline shutter_ghost_settings_t parse_shutter_ghost_settings(std::string_view text, std::vector<std::string> *problems = nullptr) {
    shutter_ghost_settings_t settings;
    const auto trim = [](std::string_view part) {
      const auto first = part.find_first_not_of(" \t\r");
      if (first == std::string_view::npos) {
        return std::string_view {};
      }
      return part.substr(first, part.find_last_not_of(" \t\r") - first + 1);
    };
    const auto report = [&](std::string_view line, std::string_view reason) {
      if (problems) {
        problems->emplace_back(std::string {reason} + ": " + std::string {line});
      }
    };
    while (!text.empty()) {
      const auto end = text.find('\n');
      auto line = text.substr(0, end);
      text = end == std::string_view::npos ? std::string_view {} : text.substr(end + 1);
      const auto whole_line = trim(line);
      line = trim(line.substr(0, line.find('#')));
      if (line.empty()) {
        continue;
      }
      const auto equals = line.find('=');
      if (equals == std::string_view::npos) {
        report(whole_line, "no '='");
        continue;
      }
      const auto name = trim(line.substr(0, equals));
      const auto value = trim(line.substr(equals + 1));
      if (name == "view") {
        if (value != "left" && value != "right") {
          report(whole_line, "view is left or right");
          continue;
        }
        settings.view_left = value == "left";
        continue;
      }
      float number = 0.0f;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
      if (value.empty() || parsed.ec != std::errc {} || parsed.ptr != value.data() + value.size() || !(number == number)) {
        report(whole_line, "not a number");
        continue;
      }
      if (name == "top_peak") {
        settings.top_peak = std::clamp(number, 0.0f, shutter_ghost_largest_peak);
      } else if (name == "top_extent") {
        settings.top_extent = std::clamp(number, 0.0f, shutter_ghost_largest_extent);
      } else if (name == "bottom_peak") {
        settings.bottom_peak = std::clamp(number, 0.0f, shutter_ghost_largest_peak);
      } else if (name == "bottom_extent") {
        settings.bottom_extent = std::clamp(number, 0.0f, shutter_ghost_largest_extent);
      } else if (name == "black_raise") {
        settings.black_raise = std::clamp(number, 0.0f, 1.0f);
      } else if (name == "white_nits") {
        settings.white_nits = std::clamp(number, 40.0f, 1000.0f);
      } else if (name == "pattern") {
        settings.pattern = std::clamp(static_cast<int>(number), 0, 2);
      } else {
        report(whole_line, "unknown name");
      }
    }
    return settings;
  }

  /**
   * Follows a settings file so the numbers can be changed while a picture is on screen. A
   * thread of its own reads the file; the presenting thread only ever copies a finished value.
   */
  class shutter_ghost_settings_file_t {
  public:
    using report_t = void (*)(const std::filesystem::path &, const shutter_ghost_settings_t &, const std::vector<std::string> &);

    shutter_ghost_settings_file_t(std::filesystem::path path, report_t report):
        reader_ {[this, path = std::move(path), report](std::stop_token stop) {
          std::filesystem::file_time_type seen {};
          bool seen_any = false;
          std::mutex wait_mutex;
          std::condition_variable_any wake;
          while (!stop.stop_requested()) {
            std::error_code error;
            const auto written = std::filesystem::last_write_time(path, error);
            if (!error && (!seen_any || written != seen)) {
              std::ifstream file {path, std::ios::binary};
              std::ostringstream text;
              text << file.rdbuf();
              // A file caught half-written is read again when its time changes once more.
              if (file) {
                seen = written;
                seen_any = true;
                std::vector<std::string> problems;
                const auto settings = parse_shutter_ghost_settings(text.str(), &problems);
                {
                  std::lock_guard lock {mutex_};
                  latest_ = settings;
                }
                version_.fetch_add(1, std::memory_order_release);
                if (report) {
                  report(path, settings, problems);
                }
              }
            }
            std::unique_lock lock {wait_mutex};
            wake.wait_for(lock, stop, std::chrono::milliseconds {200}, [] {
              return false;
            });
          }
        }} {
    }

    /// Copies newer settings into `settings` and says whether it did.
    bool take(shutter_ghost_settings_t &settings) {
      const auto version = version_.load(std::memory_order_acquire);
      if (version == taken_) {
        return false;
      }
      std::lock_guard lock {mutex_};
      settings = latest_;
      taken_ = version;
      return true;
    }

  private:
    std::mutex mutex_;
    shutter_ghost_settings_t latest_;
    std::atomic<std::uint32_t> version_ {0};
    std::uint32_t taken_ = 0;
    std::jthread reader_;
  };

  /// Matches local_shutter_eye_cbuffer in local_shutter_eye_ps.hlsl.
  struct shutter_eye_constants_t {
    std::int32_t own_left;
    std::int32_t other_left;
    std::int32_t eye_height;
    std::int32_t srgb_coded;
    float top_peak;
    float top_extent;
    float bottom_peak;
    float bottom_extent;
    float black_raise;
    float white_level;
    std::int32_t pattern;
    std::int32_t drawing_left;
    std::int32_t view_left;
    std::int32_t eye_width;
    float unused[2];
  };
  static_assert(sizeof(shutter_eye_constants_t) == 64);

  [[nodiscard]] constexpr shutter_eye_constants_t shutter_eye_constants(
    const shutter_ghost_settings_t &settings,
    bool left_eye,
    std::uint32_t eye_width,
    std::uint32_t eye_height,
    bool hdr
  ) noexcept {
    return {
      .own_left = left_eye ? 0 : static_cast<std::int32_t>(eye_width),
      .other_left = left_eye ? static_cast<std::int32_t>(eye_width) : 0,
      .eye_height = static_cast<std::int32_t>(eye_height),
      .srgb_coded = hdr ? 0 : 1,
      .top_peak = settings.top_peak,
      .top_extent = settings.top_extent,
      .bottom_peak = settings.bottom_peak,
      .bottom_extent = settings.bottom_extent,
      .black_raise = settings.black_raise,
      // Linear scRGB counts 80 nits as 1.
      .white_level = hdr ? settings.white_nits / 80.0f : 1.0f,
      .pattern = settings.pattern,
      .drawing_left = left_eye ? 1 : 0,
      .view_left = settings.view_left ? 1 : 0,
      .eye_width = static_cast<std::int32_t>(eye_width),
      .unused = {},
    };
  }

  /// The draw itself, on the device that presents to the shutter monitor.
  class shutter_eye_pass_t {
  public:
    bool init(ID3D11Device *device, const void *vertex_code, std::size_t vertex_size, const void *pixel_code, std::size_t pixel_size) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(shutter_eye_constants_t);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      device_ = device;
      return device && vertex_code && pixel_code &&
             SUCCEEDED(device->CreateVertexShader(vertex_code, vertex_size, nullptr, &vertex_)) &&
             SUCCEEDED(device->CreatePixelShader(pixel_code, pixel_size, nullptr, &pixel_)) &&
             SUCCEEDED(device->CreateBuffer(&desc, nullptr, &constants_));
    }

    [[nodiscard]] bool ready() const noexcept {
      return vertex_ && pixel_ && constants_;
    }

    /**
     * Draws the eye into `target`. `pair` holds the left eye in its left half and the right eye
     * in its right half; a pattern needs no pair.
     */
    bool draw(
      ID3D11DeviceContext *context,
      ID3D11RenderTargetView *target,
      ID3D11Texture2D *pair,
      bool left_eye,
      std::uint32_t eye_width,
      std::uint32_t eye_height,
      bool hdr,
      const shutter_ghost_settings_t &settings
    ) {
      if (!ready() || !context || !target) {
        return false;
      }
      ID3D11ShaderResourceView *view = nullptr;
      if (settings.pattern == 0) {
        view = pair ? pair_view(pair) : nullptr;
        if (!view) {
          return false;
        }
      }
      const auto constants = shutter_eye_constants(settings, left_eye, eye_width, eye_height, hdr);
      context->UpdateSubresource(constants_.Get(), 0, nullptr, &constants, 0, 0);
      const D3D11_VIEWPORT viewport {0.0f, 0.0f, static_cast<FLOAT>(eye_width), static_cast<FLOAT>(eye_height), 0.0f, 1.0f};
      context->IASetInputLayout(nullptr);
      context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      context->VSSetShader(vertex_.Get(), nullptr, 0);
      context->PSSetShader(pixel_.Get(), nullptr, 0);
      context->PSSetConstantBuffers(0, 1, constants_.GetAddressOf());
      context->PSSetShaderResources(0, 1, &view);
      context->RSSetViewports(1, &viewport);
      context->OMSetRenderTargets(1, &target, nullptr);
      context->Draw(3, 0);
      // The pair goes back to its producer between eyes; leave nothing of it bound.
      ID3D11ShaderResourceView *no_view = nullptr;
      ID3D11RenderTargetView *no_target = nullptr;
      context->PSSetShaderResources(0, 1, &no_view);
      context->OMSetRenderTargets(1, &no_target, nullptr);
      return true;
    }

  private:
    // The presenter alternates between the two textures of one ring, so two views are enough.
    // A view keeps its texture alive, which also keeps its address from being used again.
    ID3D11ShaderResourceView *pair_view(ID3D11Texture2D *pair) {
      for (auto &[texture, view] : views_) {
        if (texture == pair) {
          return view.Get();
        }
      }
      auto &[texture, view] = views_[next_view_];
      next_view_ = (next_view_ + 1) % views_.size();
      texture = nullptr;
      view.Reset();
      if (FAILED(device_->CreateShaderResourceView(pair, nullptr, &view))) {
        return nullptr;
      }
      texture = pair;
      return view.Get();
    }

    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> constants_;
    std::array<std::pair<ID3D11Texture2D *, Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>>, 2> views_ {};
    std::size_t next_view_ = 0;
  };
}  // namespace platf::dxgi::detail

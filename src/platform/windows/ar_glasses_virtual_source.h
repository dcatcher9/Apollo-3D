/**
 * @file src/platform/windows/ar_glasses_virtual_source.h
 * @brief Virtual source setup: finding the owned SudoVDA source by its stable identity and bringing
 * its mode, refresh and color to the physical output's contract.
 */
#pragma once

#include "ar_glasses_ccd.h"
#include "ar_glasses_devices.h"
#include "virtual_display.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    using namespace std::chrono_literals;

    bool same_virtual_display_identity(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &left,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &right
    );

    template<class Query, class Apply, class Wait>
    std::optional<int> configure_source_refresh_with_io(
      int requested_millihz,
      Query query,
      Apply apply,
      Wait wait,
      const std::function<bool()> &cancelled
    ) {
      for (int attempt = 0; attempt < 3 && !cancelled(); ++attempt) {
        const auto source = query();
        if (source && source_refresh_matches(source->refresh_millihz, requested_millihz)) {
          return cancelled() ? std::nullopt : std::make_optional(source->refresh_millihz);
        }
        if (source) {
          apply(*source);
        }
        for (int poll = 0; poll < 4 && !cancelled(); ++poll) {
          wait(50ms);
          const auto observed = query();
          if (observed && source_refresh_matches(observed->refresh_millihz, requested_millihz)) {
            return cancelled() ? std::nullopt : std::make_optional(observed->refresh_millihz);
          }
        }
      }
      if (cancelled()) {
        return std::nullopt;
      }
      const auto source = query();
      return !cancelled() && source && source->refresh_millihz > 0 ?
               std::make_optional(source->refresh_millihz) :
               std::nullopt;
    }

    template<class Query, class Apply, class Wait, class Now>
    bool establish_source_mode_with_io(
      int width,
      int height,
      Query query,
      Apply apply,
      Wait wait,
      Now now,
      const std::function<bool()> &cancelled
    ) {
      // Add publishes an exact identity before Windows necessarily applies the requested mode.
      // A remembered AR 1080p mode must not make a new native-monitor source unresolvable before
      // its first setter. Query is identity-only here; callers retain strict geometry afterward.
      // Allow five seconds for publication, applying at most once per repair attempt. Windows
      // setters are synchronous; their time counts against the deadline but cannot be interrupted.
      const auto deadline = now() + 5s;
      for (int attempt = 0; attempt < 5 && !cancelled(); ++attempt) {
        bool applied = false;
        for (int poll = 0; poll < 20 && !cancelled() && now() < deadline; ++poll) {
          const auto source = query();
          if (cancelled() || now() >= deadline) {
            return false;
          }
          if (source && source->rect.right - source->rect.left == width && source->rect.bottom - source->rect.top == height && source->refresh_millihz > 0) {
            return !cancelled();
          }
          if (source && !applied) {
            apply(*source);
            applied = true;
          }
          if (cancelled() || now() >= deadline) {
            return false;
          }
          wait(50ms);
        }
      }
      return false;
    }

    struct resolved_virtual_display_t {
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {};
      std::wstring gdi_name;
      std::wstring device_path;
      int refresh_millihz = 0;
      RECT rect {};
    };

    struct source_color_state_t {
      resolved_virtual_display_t display;
      std::optional<bool> hdr;
    };

    template<class Query, class SetMode, class SetColor, class Wait, class Now>
    bool settle_source_color_with_io(
      int width,
      int height,
      int refresh_millihz,
      bool expected_hdr,
      std::chrono::milliseconds timeout,
      Query query,
      SetMode set_mode,
      SetColor set_color,
      Wait wait,
      Now now,
      const std::function<bool()> &cancelled
    ) {
      const auto deadline = now() + timeout;
      auto next_mode_set = now();
      auto next_color_set = now();
      auto next_stable_observation = now();
      int mode_attempts = 0;
      int color_attempts = 0;
      int stable_observations = 0;
      // In addition to the clock deadline, retain a finite query bound for injected I/O.
      const auto max_polls = timeout / 50ms + 1;
      for (std::int64_t poll = 0; poll < max_polls && !cancelled() && now() < deadline; ++poll) {
        const auto source = query();
        if (cancelled() || now() >= deadline) {
          return false;
        }
        const bool native_mode = source &&
                                 source->display.rect.right - source->display.rect.left == width &&
                                 source->display.rect.bottom - source->display.rect.top == height &&
                                 source_refresh_matches(source->display.refresh_millihz, refresh_millihz);
        if (!native_mode) {
          stable_observations = 0;
          next_stable_observation = now();
          // HDR may restore Windows' remembered raster/rate. Only an exact owned identity may
          // receive a mode setter, and color is retried only after its native cadence is proven.
          if (source && mode_attempts < 3 && now() >= next_mode_set) {
            set_mode(source->display);
            ++mode_attempts;
            next_mode_set = now() + 1s;
          }
        } else if (source->hdr && *source->hdr == expected_hdr) {
          if (now() >= next_stable_observation) {
            next_stable_observation = now() + 200ms;
            if (++stable_observations >= 3) {
              return !cancelled();
            }
          }
        } else {
          stable_observations = 0;
          next_stable_observation = now();
          if (source->hdr && color_attempts < 3 && now() >= next_color_set) {
            set_color(source->display, expected_hdr);
            ++color_attempts;
            next_color_set = now() + 1s;
          }
        }
        if (cancelled() || now() >= deadline) {
          return false;
        }
        wait(50ms);
      }
      return false;
    }

    enum class virtual_display_presence_e {
      indeterminate,
      absent,
      present,
    };

    struct virtual_display_probe_t {
      virtual_display_presence_e presence {virtual_display_presence_e::indeterminate};
      std::optional<resolved_virtual_display_t> resolved;
    };

    virtual_display_probe_t probe_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name,
      int expected_width = source_width,
      int expected_height = source_height,
      bool require_geometry = true
    );

    std::optional<resolved_virtual_display_t> resolve_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name,
      bool *topology_query_succeeded = nullptr,
      int expected_width = source_width,
      int expected_height = source_height,
      bool require_geometry = true
    );

    bool configure_virtual_display_hdr(
      std::wstring &display_name,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view physical_device_path,
      bool enable_hdr,
      bool &configured_hdr,
      const std::function<bool()> &cancelled
    );
  }  // namespace internal
}  // namespace ar_glasses

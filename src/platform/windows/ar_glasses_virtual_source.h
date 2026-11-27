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

    /** Whether an owned source shows the required raster at the required refresh. */
    inline bool source_mode_ready(const resolved_virtual_display_t &source, int width, int height, int refresh_millihz) {
      return source.rect.right - source.rect.left == width && source.rect.bottom - source.rect.top == height &&
             source_refresh_matches(source.refresh_millihz, refresh_millihz);
    }

    /** What one look at a display shows while its color settles. */
    enum class color_presence_e {
      unknown,  ///< Windows could not say this time. The look neither counts nor ends the wait.
      gone,  ///< The display, or the display generation the request was made for, has left.
      present,
    };

    struct color_observation_t {
      color_presence_e presence = color_presence_e::unknown;
      bool mode_ready = false;  ///< The display shows the mode the caller requires.
      std::optional<bool> hdr;  ///< Windows' Advanced Color state; empty when it could not be read.
    };

    enum class color_settle_e {
      settled,  ///< Three looks at least 200 ms apart showed the requested color at the required mode.
      rejected,  ///< Windows refused every color request.
      timed_out,  ///< The display did not show the requested color and mode within the budget.
      disconnected,  ///< A look found the display gone.
      cancelled,  ///< The caller's request ended first.
    };

    struct color_settle_request_t {
      bool hdr = false;
      std::chrono::milliseconds budget {};
      bool repair_mode = false;  ///< Whether a display at the wrong mode may be set to it again.
    };

    struct color_settle_result_t {
      color_settle_e outcome = color_settle_e::timed_out;
      int color_requests = 0;
      int refused_requests = 0;
      int mode_repairs = 0;
      std::chrono::milliseconds elapsed {};
    };

    /** The time an owned source has to show a requested color. Windows takes longer to enter HDR. */
    constexpr std::chrono::milliseconds source_color_budget(bool hdr) {
      return hdr ? 15s : 5s;
    }

    /**
     * The time a source that was just created has to show SDR. Windows publishes a new indirect
     * display's mode late; the set-up before the one settle rule waited for it twice, 5 s each,
     * before it gave up.
     */
    inline constexpr std::chrono::milliseconds new_source_sdr_budget = 10s;

    /**
     * The one color settle rule of local presentation, for the owned source and the physical
     * output alike. The display is looked at at once and then every 50 ms until the budget ends:
     * - a look that cannot see the display, or sees the wrong mode or color, restarts the count of
     *   good looks;
     * - a look that finds the display gone ends the wait as disconnected;
     * - a display at the wrong mode is set to it again when the caller allows a mode repair, at most
     *   three times and at least a second apart;
     * - a display at the required mode in the wrong, known color receives the color request, at most
     *   three times and at least a second apart. A refused request counts too, and when Windows
     *   has refused all three the wait ends as rejected;
     * - three good looks at least 200 ms apart settle it.
     * Windows' report is the authority: an accepted request proves nothing until it is seen, and a
     * refused one does not stop a display that shows the color anyway. The caller says what a look
     * sees (which display, what mode it needs, what gone means) and which display the setters
     * address; set_color returns whether Windows accepted the request. Nothing is logged here.
     */
    template<class Observe, class SetMode, class SetColor, class Wait, class Now>
    color_settle_result_t settle_color_with_io(
      const color_settle_request_t &request,
      Observe observe,
      SetMode set_mode,
      SetColor set_color,
      Wait wait,
      Now now,
      const std::function<bool()> &cancelled
    ) {
      constexpr auto poll = 50ms;
      constexpr auto good_look_spacing = 200ms;
      constexpr int good_looks_needed = 3;
      constexpr auto request_spacing = 1s;
      constexpr int request_limit = 3;
      const auto started = now();
      const auto deadline = started + request.budget;
      auto next_mode_repair = started;
      auto next_color_request = started;
      auto next_good_look = started;
      int good_looks = 0;
      color_settle_result_t result;
      const auto finish = [&](color_settle_e outcome) {
        result.outcome = cancelled() ? color_settle_e::cancelled : outcome;
        result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now() - started);
        return result;
      };
      // In addition to the clock deadline, retain a finite look bound for injected I/O.
      const auto max_looks = request.budget / poll + 1;
      for (std::int64_t look = 0; look < max_looks && !cancelled() && now() < deadline; ++look) {
        const auto seen = observe();
        if (cancelled()) {
          break;
        }
        if (seen.presence == color_presence_e::gone) {
          return finish(color_settle_e::disconnected);
        }
        if (now() >= deadline) {
          break;
        }
        const bool good = seen.presence == color_presence_e::present && seen.mode_ready && seen.hdr && *seen.hdr == request.hdr;
        if (!good) {
          good_looks = 0;
          next_good_look = now();
        }
        if (seen.presence != color_presence_e::present) {
          // Neither a setter nor a good look without an authoritative observation.
        } else if (!seen.mode_ready) {
          // HDR may restore Windows' remembered raster or rate. Color is requested again only
          // after the required mode is seen.
          if (request.repair_mode && result.mode_repairs < request_limit && now() >= next_mode_repair) {
            set_mode();
            ++result.mode_repairs;
            next_mode_repair = now() + request_spacing;
          }
        } else if (good) {
          if (now() >= next_good_look) {
            next_good_look = now() + good_look_spacing;
            if (++good_looks >= good_looks_needed) {
              return finish(color_settle_e::settled);
            }
          }
        } else if (seen.hdr && result.color_requests < request_limit && now() >= next_color_request) {
          ++result.color_requests;
          if (!set_color(request.hdr) && ++result.refused_requests >= request_limit) {
            return finish(color_settle_e::rejected);
          }
          next_color_request = now() + request_spacing;
        }
        if (cancelled() || now() >= deadline) {
          break;
        }
        wait(poll);
      }
      return finish(color_settle_e::timed_out);
    }

    /**
     * Bring an owned source to the requested color with the rule above. Right after the source was
     * created Windows often ignores an HDR request, so a new source first settles in SDR. A source
     * that may present color-managed SDR on an HDR output (glasses, not a shutter monitor) settles
     * back in SDR when it does not reach HDR, unless its output is gone or the request ended.
     * `settle(hdr, budget)` runs the rule for one color within that budget. Returns the color the
     * source reached, or nothing when it reached neither the request nor its fallback.
     */
    template<class Settle>
    std::optional<bool> establish_source_color_with_io(bool expected_hdr, bool after_creation, bool sdr_fallback, Settle settle) {
      if (after_creation) {
        if (settle(false, new_source_sdr_budget).outcome != color_settle_e::settled) {
          return std::nullopt;
        }
        if (!expected_hdr) {
          return false;
        }
      }
      const auto requested = settle(expected_hdr, source_color_budget(expected_hdr)).outcome;
      if (requested == color_settle_e::settled) {
        return expected_hdr;
      }
      if (!expected_hdr || !sdr_fallback || requested == color_settle_e::disconnected || requested == color_settle_e::cancelled) {
        return std::nullopt;
      }
      return settle(false, source_color_budget(false)).outcome == color_settle_e::settled ? std::make_optional(false) : std::nullopt;
    }

    /** Log a settled, rejected or timed-out outcome. The caller logs what ended a disconnected
     * or cancelled one, since only it knows.
     */
    void log_color_settle(std::string_view display, const color_settle_request_t &request, const color_settle_result_t &result);

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
  }  // namespace internal
}  // namespace ar_glasses

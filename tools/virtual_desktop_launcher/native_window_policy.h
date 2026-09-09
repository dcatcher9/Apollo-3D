/** Causal admission for one native-shell launch gesture from a virtual display. */
#pragma once

#include <cstdint>
#include <optional>

namespace desktop_launcher {
  enum class native_input_action_e {
    motion,
    navigate,
    launch,
  };

  struct native_input_t {
    std::uint64_t tick = 0;
    std::uintptr_t source_root = 0;
    native_input_action_e action = native_input_action_e::motion;
    // Physical input and this session's tagged remote injection are trusted. Other injected
    // input must cancel routing because its launch origin cannot be attributed here.
    bool trusted_input = false;
    bool target_available = false;
    // Pointer location for mouse input; a qualified foreground source for keyboard input.
    bool inside_target = false;
    // Qualified native launch surface, or an explicit native-shell keyboard shortcut.
    bool shell_surface = false;
    // Only the native adapter may qualify a keyboard shell hop after prior virtual navigation.
    // Never set this for pointer input: clicking another monitor must cancel routing.
    bool inherits_virtual_shell_origin = false;
    // Optional ordering evidence captured by the low-level hook before native dispatch.
    std::uint64_t observation_serial = 0;
    std::uintptr_t foreground_root_before_input = 0;
  };

  struct native_foreground_t {
    std::uint64_t event_tick = 0;
    std::uint64_t now_tick = 0;
    std::uint64_t expected_generation = 0;
    std::uintptr_t root_window = 0;
    bool eligible_application = false;
    // Infrastructure only: taskbars, desktop, Start/Search and menus. Explorer app windows
    // may be eligible destinations even though Explorer is also a native launch surface.
    bool shell_surface = false;
    bool is_current_foreground = false;
    bool target_available = false;
    bool input_idle_unchanged = false;
    // Captured at foreground-callback entry from the same serial counter as input records.
    std::uint64_t observation_serial = 0;
  };

  class native_window_intent_t {
  public:
    static constexpr std::uint64_t timeout_ms = 5000;

    void cancel() {
      ++generation_;
      navigation_.reset();
      pending_.reset();
    }

    void observe_input(const native_input_t &input) {
      if (last_input_tick_ && input.tick < *last_input_tick_) {
        cancel();
        return;
      }
      last_input_tick_ = input.tick;
      if (!input.trusted_input || !input.target_available) {
        cancel();
        return;
      }
      const bool inherited_origin = input.action != native_input_action_e::motion &&
                                    input.shell_surface && input.inherits_virtual_shell_origin &&
                                    navigation_ && fresh(input.tick, navigation_->tick);
      if (!input.inside_target && !inherited_origin) {
        cancel();
        return;
      }
      // Motion neither arms nor extends an intent. It cannot create keyboard provenance.
      if (input.action == native_input_action_e::motion) {
        return;
      }
      if (!input.shell_surface || !input.source_root) {
        cancel();
        return;
      }
      ++generation_;
      navigation_ = source_t {input.source_root, input.tick, input.observation_serial, input.foreground_root_before_input};
      pending_ = input.action == native_input_action_e::launch ? navigation_ : std::nullopt;
    }

    [[nodiscard]] std::uint64_t generation() const {
      return generation_;
    }

    [[nodiscard]] bool armed() const {
      return pending_.has_value();
    }

    [[nodiscard]] std::optional<std::uintptr_t> source_root() const {
      return pending_ ? std::optional {pending_->root} : std::nullopt;
    }

    /** Called only for a foreground event, after the adapter rechecks live HWND/process identity. */
    [[nodiscard]] bool claim(const native_foreground_t &event) {
      if (!pending_ || event.expected_generation != generation_) {
        return false;
      }
      if (!event.target_available || !event.input_idle_unchanged || !fresh(event.now_tick, pending_->tick)) {
        cancel();
        return false;
      }
      // Windows timestamps have millisecond granularity. Serial order alone cannot prove an
      // asynchronously queued event is new: also require a transition away from the foreground
      // root observed before the input. An already-active app's stale equal-tick event fails.
      const bool same_tick_transition = event.event_tick == pending_->tick &&
                                        pending_->observation_serial != 0 &&
                                        event.observation_serial > pending_->observation_serial &&
                                        pending_->foreground_before != 0 &&
                                        event.root_window != pending_->foreground_before;
      if ((event.event_tick <= pending_->tick && !same_tick_transition) || event.event_tick > event.now_tick || !event.root_window || event.root_window == pending_->root || !event.eligible_application || event.shell_surface || !event.is_current_foreground) {
        return false;
      }
      // Reused application windows intentionally qualify. Only this one foreground root is
      // admitted; future activations require another qualified native launch gesture.
      cancel();
      return true;
    }

  private:
    struct source_t {
      std::uintptr_t root;
      std::uint64_t tick;
      std::uint64_t observation_serial;
      std::uintptr_t foreground_before;
    };

    static bool fresh(std::uint64_t now, std::uint64_t started) {
      return now >= started && now - started <= timeout_ms;
    }

    std::uint64_t generation_ = 0;
    std::optional<std::uint64_t> last_input_tick_;
    std::optional<source_t> navigation_;
    std::optional<source_t> pending_;
  };
}  // namespace desktop_launcher

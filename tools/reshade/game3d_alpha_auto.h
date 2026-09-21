// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace sunshine_game3d {
  // Generic coverage heuristic, not a claim that alpha has UI semantics.
  // The owning contract is docs/reshade-sbs.md, UI protection under Setup.
  inline constexpr std::uint64_t alpha_startup_window_ms = 300000;
  inline constexpr std::uint64_t alpha_initial_window_ms = 60000;
  inline constexpr std::uint64_t alpha_fast_probe_interval_ms = 100;
  inline constexpr std::uint64_t alpha_slow_probe_interval_ms = 1000;
  inline constexpr std::uint64_t alpha_selective_stability_ms = 500;
  inline constexpr std::uint64_t alpha_observation_max_age_ms = 500;

  class alpha_auto_policy;
  struct alpha_auto_source {
    std::uint64_t now_ms{}, epoch{}, revision{}, sequence{}, tick_ms{};
    std::uint32_t viewport{};
    bool retained{};
    alpha_auto_policy *session{}; // Caller-owned game session; never owned by a renderer.
  };

  struct alpha_coverage_sample {
    std::uint64_t sequence{}, tick_ms{};
    std::uint32_t covered{}, pixels{}, invalid{};
  };

  enum class alpha_auto_state { waiting_for_source, collecting, automatic_on, automatic_off, manual_on, manual_off };

  inline const char *name(alpha_auto_state value) {
    switch (value) {
      case alpha_auto_state::waiting_for_source: return "startup_waiting";
      case alpha_auto_state::automatic_on: return "startup_on";
      case alpha_auto_state::automatic_off: return "startup_off";
      case alpha_auto_state::manual_on: return "manual_on";
      case alpha_auto_state::manual_off: return "manual_off";
      default: return "startup_observing";
    }
  }

  struct alpha_auto_decision {
    bool enabled{};
    alpha_auto_state state = alpha_auto_state::waiting_for_source;
    std::uint32_t covered{}, pixels{};
    std::uint64_t sample_sequence{}, sample_tick_ms{};
    bool monitoring = false;
    bool window_started = false;
    std::uint64_t window_start_ms{}, accepted_samples{};
    std::uint64_t probe_interval_ms{};
  };

  class alpha_auto_policy {
  public:
    alpha_auto_policy() {
      result_.monitoring = true;
      result_.probe_interval_ms = alpha_fast_probe_interval_ms;
    }

    explicit alpha_auto_policy(std::uint64_t start_ms) : minimum_sample_tick_(start_ms), last_now_ms_(start_ms) {
      result_.monitoring = result_.window_started = true;
      result_.window_start_ms = start_ms;
      result_.state = alpha_auto_state::collecting;
      result_.probe_interval_ms = alpha_fast_probe_interval_ms;
    }

    void begin_observation(std::uint64_t now_ms) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!result_.monitoring || result_.window_started) return;
      result_.window_started = true;
      result_.window_start_ms = now_ms;
      result_.state = alpha_auto_state::collecting;
      result_.probe_interval_ms = alpha_fast_probe_interval_ms;
      last_now_ms_ = now_ms;
      // A retained input can have been captured just before this first eligible
      // dispatch. Its freshness is checked separately from the window clock.
    }

    void set_manual(bool enabled) {
      std::lock_guard<std::mutex> lock(mutex_);
      histories_.clear();
      result_.enabled = enabled;
      result_.monitoring = false;
      result_.probe_interval_ms = 0;
      result_.state = enabled ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
    }

    void set_automatic(std::uint64_t now_ms) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (result_.monitoring || result_.state == alpha_auto_state::automatic_on || result_.state == alpha_auto_state::automatic_off) return;
      histories_.clear();
      // A copy recorded before manual mode ended is not fresh evidence for
      // resumed Auto, even if another renderer completes it after this edit.
      if (!qualified_) minimum_sample_tick_ = now_ms;
      result_.enabled = false;
      result_.monitoring = true;
      result_.state = result_.window_started ? alpha_auto_state::collecting : alpha_auto_state::waiting_for_source;
      finish(now_ms); // Returning to Auto never opens a new startup window.
      refresh_provisional(now_ms);
    }

    void reset_continuity(std::uint64_t observation_source = 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!result_.monitoring) return;
      histories_.erase(observation_source);
      if (last_observation_source_ == observation_source) {
        result_.covered = result_.pixels = 0;
        result_.sample_sequence = result_.sample_tick_ms = 0;
      }
      refresh_provisional(last_now_ms_);
    }

    void observe(const alpha_coverage_sample &sample, std::uint64_t now_ms, std::uint64_t observation_source = 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      finish(now_ms);
      if (!result_.monitoring || !result_.window_started) return;
      refresh_provisional(now_ms);
      // Only completed, fresh startup observations qualify. Re-presenting one
      // immutable FG input cannot build a selective-coverage interval. Different
      // runtimes have independent sequences and must not borrow each other's
      // selective streak or reject each other's lower-numbered observations.
      const auto previous = histories_.find(observation_source);
      if (!sample.sequence || (previous != histories_.end() && sample.sequence <= previous->second.sequence)) return;
      if (sample.tick_ms < minimum_sample_tick_ || sample.tick_ms > now_ms ||
          now_ms - sample.tick_ms > alpha_observation_max_age_ms || !sample.pixels ||
          sample.covered > sample.pixels || sample.invalid) {
        if (previous != histories_.end()) previous->second.selective = false;
        refresh_provisional(now_ms);
        return;
      }
      if (previous != histories_.end() && sample.tick_ms < previous->second.tick_ms) return;
      auto &history = histories_[observation_source];
      if (history.sequence && sample.tick_ms - history.tick_ms > alpha_observation_max_age_ms) history.selective = false;
      history.sequence = sample.sequence;
      history.tick_ms = sample.tick_ms;
      ++result_.accepted_samples;
      last_observation_source_ = observation_source;
      result_.sample_sequence = sample.sequence;
      result_.sample_tick_ms = sample.tick_ms;
      result_.covered = sample.covered;
      result_.pixels = sample.pixels;
      const bool selective = sample.covered && std::uint64_t(sample.covered) * 1000 < std::uint64_t(sample.pixels) * 999;
      if (selective && !history.selective) history.streak_start = sample.tick_ms;
      history.selective = selective;
      if (selective && sample.tick_ms - history.streak_start >= alpha_selective_stability_ms) qualified_ = true;
      finish(now_ms);
      refresh_provisional(now_ms);
    }

    alpha_auto_decision decision(std::uint64_t now_ms) {
      std::lock_guard<std::mutex> lock(mutex_);
      finish(now_ms);
      refresh_provisional(now_ms);
      return result_;
    }

  private:
    struct observation_history {
      std::uint64_t sequence{}, tick_ms{}, streak_start{};
      bool selective{};
    };

    void finish(std::uint64_t now_ms) {
      last_now_ms_ = now_ms;
      if (!result_.monitoring || !result_.window_started || (!qualified_ &&
          (now_ms < result_.window_start_ms || now_ms - result_.window_start_ms < alpha_startup_window_ms))) return;
      histories_.clear();
      result_.monitoring = false;
      result_.probe_interval_ms = 0;
      result_.enabled = qualified_;
      result_.state = qualified_ ? alpha_auto_state::automatic_on : alpha_auto_state::automatic_off;
    }
    void refresh_provisional(std::uint64_t now_ms) {
      if (!result_.monitoring) return;
      result_.enabled = false;
      for (const auto &entry : histories_) {
        const auto &history = entry.second;
        if (history.selective && now_ms >= history.tick_ms && now_ms - history.tick_ms <= alpha_observation_max_age_ms) {
          result_.enabled = true;
          break;
        }
      }
      // A sparse selective probe requests a short frequent-sampling burst. It
      // cannot accumulate confirmation from isolated once-per-second samples.
      result_.probe_interval_ms = result_.enabled || !result_.window_started ||
        now_ms < result_.window_start_ms || now_ms - result_.window_start_ms < alpha_initial_window_ms ?
        alpha_fast_probe_interval_ms : alpha_slow_probe_interval_ms;
    }
    std::mutex mutex_;
    alpha_auto_decision result_;
    std::uint64_t minimum_sample_tick_{};
    std::uint64_t last_now_ms_{};
    std::unordered_map<std::uint64_t, observation_history> histories_;
    std::uint64_t last_observation_source_{};
    bool qualified_{};
  };
}

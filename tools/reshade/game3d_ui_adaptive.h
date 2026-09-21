// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace sunshine_game3d::ui_adaptive {
  inline constexpr std::array<float, 5> levels_uv {0.f, .001f, .002f, .003f, .0035f};
  inline constexpr std::uint32_t max_target_index = 4;
  inline constexpr float max_live_fraction = .5f;
  inline constexpr std::uint64_t max_age_ms = 250;
  inline constexpr std::uint64_t probe_interval_ms = 100;
  inline constexpr std::uint64_t release_dwell_ms = 1500;
  inline constexpr std::uint64_t approach_dwell_ms = 100;
  inline constexpr std::uint32_t entry_conflict_percent = 20;
  inline constexpr std::uint32_t release_conflict_percent = 15;
  inline constexpr std::uint32_t entry_area_per_mille = 20;
  inline constexpr std::uint32_t release_area_per_mille = 15;
  inline constexpr float comfort_cap_uv = .0035f;
  inline constexpr double approach_rate_uv_per_second = .03;
  inline constexpr double retreat_rate_uv_per_second = .005;

  [[nodiscard]] constexpr float level_uv(unsigned index, float front_cap_uv) {
    return std::min(levels_uv[index], max_live_fraction * front_cap_uv);
  }

  // Existing 16-by-16 probe tiles exactly delimit the central 75% of each
  // dimension: [floor(size/8), floor(7*size/8)). Edge UI is not placement evidence.
  [[nodiscard]] constexpr bool central_tile(unsigned x, unsigned y) {
    return x >= 2 && x < 14 && y >= 2 && y < 14;
  }

  struct source {
    std::uint64_t now_ms{}, tick_ms{}, epoch{}, revision{}, source_id{}, sequence{}, mask_sequence{};
    std::uint32_t viewport{};
    bool eligible{};
    float front_cap_uv = .01f;
    // Generic depth rotates physical source_id/layout revisions within one
    // admitted routing group. Provider source_id already identifies a logical
    // feature and leaves these zero. Probe provenance keeps both identities.
    std::uint64_t generic_basis_epoch{}, generic_routing_epoch{};
  };

  [[nodiscard]] inline bool valid_scope(const source &value) {
    return bool(value.generic_basis_epoch) == bool(value.generic_routing_epoch);
  }
  [[nodiscard]] inline bool same_scope(const source &a, const source &b) {
    return valid_scope(a) && valid_scope(b) && a.epoch == b.epoch && a.viewport == b.viewport &&
      a.generic_basis_epoch == b.generic_basis_epoch && a.generic_routing_epoch == b.generic_routing_epoch &&
      (a.generic_basis_epoch || (a.revision == b.revision && a.source_id == b.source_id));
  }
  [[nodiscard]] inline bool fresh(const source &value) {
    return value.eligible && valid_scope(value) && value.sequence && value.tick_ms &&
      value.tick_ms <= value.now_ms && value.now_ms - value.tick_ms <= max_age_ms;
  }

  // Two RGBA32_UINT texels per tile. Each bad count uses the corresponding
  // absolute level clipped by the current scene's half-cap.
  struct tile {
    std::uint32_t covered{}, invalid{};
    std::array<std::uint32_t, 5> bad {};
    std::uint32_t pixels{};
  };
  static_assert(sizeof(tile) == 8 * sizeof(std::uint32_t));

  struct sample {
    source provenance;
    std::uint32_t width{}, height{};
    std::array<tile, 256> tiles {};
  };

  struct decision {
    float applied_fraction{};
    float applied_uv{}, target_uv{}, required_uv{}, limit_uv{}, observed_front_cap_uv{};
    std::uint32_t target_index{}, required_index{};
    bool capped_conflict{};
    std::uint64_t accepted_sequence{}, accepted_tick{}, accepted_mask_sequence{};
    // Coverage and conflicts refer only to UI pixels in the central region.
    std::uint64_t probe_age_ms{}, center_pixels{}, covered_pixels{}, conflict_pixels{}, invalid_pixels{};
    std::array<std::uint64_t, 5> conflict_counts {};
    const char *status {"no_observation"};
  };

  class policy {
  public:
    void reset() { *this = {}; }
    const decision &current() const { return output_; }

    // Called for every presentation, including unavailable inputs. Only an
    // eligible new scope resets placement; temporary loss holds its last value.
    decision prepare(const source &live) {
      const bool backwards = have_clock_ && live.now_ms < clock_;
      const auto elapsed = have_clock_ && !backwards ? std::min(live.now_ms - clock_, max_age_ms) : 0;
      clock_ = live.now_ms;
      have_clock_ = true;
      live_ = live;
      have_live_ = true;
      if (!std::isfinite(live.front_cap_uv) || live.front_cap_uv < 0.f || live.front_cap_uv > .04f) {
        suspend("invalid_cap");
        update_age(live.now_ms);
        return output_;
      }
      const bool cap_changed = !have_cap_ || cap_ != live.front_cap_uv;
      if (cap_changed) {
        cap_ = live.front_cap_uv;
        have_cap_ = true;
        cap_since_ = live.now_ms;
        output_.limit_uv = std::min(comfort_cap_uv, max_live_fraction * cap_);
        applied_ = std::min(applied_, double(output_.limit_uv));
        output_.target_uv = static_cast<float>(applied_);
        suspend("cap_changed");
        publish_placement();
      }
      if (!ui_adaptive::fresh(live)) {
        suspend("ineligible");
        update_age(live.now_ms);
        return output_;
      }
      if (!have_scope_ || !same_scope(scope_, live)) {
        reset();
        scope_ = live_ = live;
        have_scope_ = have_live_ = have_clock_ = true;
        clock_ = live.now_ms;
        cap_ = live.front_cap_uv;
        cap_since_ = live.now_ms;
        have_cap_ = true;
        output_.limit_uv = std::min(comfort_cap_uv, max_live_fraction * cap_);
        output_.status = "scope_changed";
        return output_;
      }
      update_age(live.now_ms);
      if (backwards) {
        suspend("clock_regressed");
      } else if (cap_changed) {
        suspend("cap_changed");
      } else if (!have_accepted_ || !fresh(output_.accepted_tick, live.now_ms)) {
        suspend(have_accepted_ ? "observation_expired" : "no_observation");
      } else if (ramp_armed_) {
        const double target = output_.target_uv;
        const double distance = double(elapsed) * (target > applied_ ?
          approach_rate_uv_per_second : retreat_rate_uv_per_second) / 1000.;
        if (target > applied_) applied_ = std::min(target, applied_ + distance);
        else applied_ = std::max(target, applied_ - distance);
        publish_placement();
      }
      return output_;
    }

    // Accepted samples must be new real source observations. Re-presenting a
    // retained depth or retained mask never extends release evidence or age.
    bool observe(const sample &value, std::uint64_t now) {
      const auto &origin = value.provenance;
      if (!have_live_ || !live_.eligible || !have_scope_ || !origin.eligible ||
          !same_scope(scope_, origin) || !same_scope(live_, origin)) return reject("sample_scope");
      if (!fresh(origin.tick_ms, now) || !fresh(live_.tick_ms, now) || now < clock_)
        return reject("sample_expired");
      if (!have_cap_ || origin.front_cap_uv != cap_ || live_.front_cap_uv != cap_ ||
          origin.now_ms < cap_since_) return reject("sample_cap");
      if (!origin.sequence || origin.sequence > live_.sequence ||
          (origin.mask_sequence == 0) != (live_.mask_sequence == 0) ||
          (origin.mask_sequence && origin.mask_sequence > live_.mask_sequence)) return reject("sample_identity");
      if (have_accepted_ && (origin.sequence <= output_.accepted_sequence || origin.tick_ms < output_.accepted_tick ||
          (origin.mask_sequence && origin.mask_sequence <= last_retained_mask_))) return reject("sample_duplicate");

      std::array<std::uint64_t, 5> totals {};
      std::uint64_t center_pixels{}, covered{}, invalid{};
      if (!valid_counts(value, totals, center_pixels, covered, invalid)) return reject("malformed_counts");
      if (invalid) return reject("invalid_coverage");
      if (have_accepted_ && (origin.tick_ms - output_.accepted_tick > max_age_ms ||
          now < accepted_at_ || now - accepted_at_ > max_age_ms)) clear_evidence();

      const auto conflicts = [&](unsigned index, bool exiting) {
        if (!covered) return false;
        return exiting ?
          totals[index] * 100 >= covered * release_conflict_percent ||
            totals[index] * 1000 >= center_pixels * release_area_per_mille :
          totals[index] * 100 > covered * entry_conflict_percent ||
            totals[index] * 1000 > center_pixels * entry_area_per_mille;
      };
      unsigned required{};
      while (required < max_target_index && conflicts(required, false)) ++required;
      const float target = level_uv(required, cap_);
      output_.required_index = required;
      output_.required_uv = target;
      output_.capped_conflict = conflicts(max_target_index, false);
      output_.accepted_sequence = origin.sequence;
      output_.observed_front_cap_uv = origin.front_cap_uv;
      output_.accepted_tick = origin.tick_ms;
      output_.accepted_mask_sequence = origin.mask_sequence;
      output_.covered_pixels = covered;
      output_.center_pixels = center_pixels;
      output_.invalid_pixels = invalid;
      output_.conflict_counts = totals;
      if (origin.mask_sequence) last_retained_mask_ = origin.mask_sequence;
      have_accepted_ = true;
      accepted_at_ = now;
      clock_ = now;
      update_age(now);
      ramp_armed_ = true;

      if (target > output_.target_uv) {
        clear_release();
        // Only the nearest requirement shared by all confirming samples is
        // sustained evidence. A final one-frame spike cannot choose the target.
        if (!approaching_) {
          approaching_ = true;
          approach_since_ = origin.tick_ms;
          approach_candidate_ = target;
        } else approach_candidate_ = std::min(approach_candidate_, target);
        if (origin.tick_ms - approach_since_ >= approach_dwell_ms) {
          output_.target_uv = approach_candidate_;
          clear_approach();
          output_.status = "approaching";
        } else {
          ramp_armed_ = false;
          output_.status = "approach_dwell";
        }
      } else if (applied_ != double(output_.target_uv)) {
        clear_approach();
        clear_release();
        output_.status = "transitioning";
      } else {
        clear_approach();
        unsigned release{};
        while (release < max_target_index && conflicts(release, true)) ++release;
        const float candidate = level_uv(release, cap_);
        if (candidate < output_.target_uv) {
          if (!releasing_) {
            releasing_ = true;
            release_since_ = origin.tick_ms;
            release_candidate_ = candidate;
          } else release_candidate_ = std::max(release_candidate_, candidate);
          if (origin.tick_ms - release_since_ >= release_dwell_ms) {
            output_.target_uv = release_candidate_;
            clear_release();
            output_.status = "retreating";
          } else output_.status = "release_dwell";
        } else {
          clear_release();
          output_.status = covered ? "stable" : "no_coverage";
        }
      }
      publish_placement();
      return true;
    }

  private:
    static bool fresh(std::uint64_t tick, std::uint64_t now) {
      return tick && tick <= now && now - tick <= max_age_ms;
    }
    static bool valid_counts(const sample &value, std::array<std::uint64_t, 5> &bad,
        std::uint64_t &center_pixels, std::uint64_t &covered, std::uint64_t &invalid) {
      if (!value.width || !value.height || value.width > 8192 || value.height > 8192) return false;
      std::uint64_t pixels{};
      for (unsigned index = 0; index != value.tiles.size(); ++index) {
        const auto &cell = value.tiles[index];
        const unsigned x = index % 16, y = index / 16;
        const std::uint64_t expected = ((x + 1) * value.width / 16 - x * value.width / 16) *
          ((y + 1) * value.height / 16 - y * value.height / 16);
        if (cell.pixels != expected || cell.covered > cell.pixels || cell.invalid > cell.covered) return false;
        std::uint32_t previous = cell.covered - cell.invalid;
        for (unsigned i = 0; i != cell.bad.size(); ++i) {
          if (cell.bad[i] > previous) return false;
          previous = cell.bad[i];
          if (central_tile(x, y)) bad[i] += cell.bad[i];
        }
        pixels += cell.pixels;
        if (central_tile(x, y)) {
          center_pixels += cell.pixels;
          covered += cell.covered;
          invalid += cell.invalid;
        }
      }
      return pixels == std::uint64_t(value.width) * value.height;
    }
    static unsigned plane_index(float uv, float cap) {
      unsigned index{};
      while (index < max_target_index && uv > level_uv(index, cap)) ++index;
      return index;
    }
    void publish_placement() {
      const float desired = std::min(static_cast<float>(applied_), output_.limit_uv);
      output_.applied_fraction = cap_ > 0.f ? std::min(desired / cap_, max_live_fraction) : 0.f;
      // The shader receives fraction, so bound the actual rounded product as
      // well as the controller's UV state. Report that exact rendered value.
      while (output_.applied_fraction > 0.f && output_.applied_fraction * cap_ > desired)
        output_.applied_fraction = std::nextafter(output_.applied_fraction, 0.f);
      output_.applied_uv = output_.applied_fraction * cap_;
      output_.target_index = plane_index(output_.target_uv, cap_);
      // A cap change may hold an intermediate ramp position. Its lower sampled
      // level gives a conservative count until a fresh target is chosen.
      unsigned lower{};
      while (lower < max_target_index && level_uv(lower + 1, output_.observed_front_cap_uv) <= output_.target_uv) ++lower;
      output_.conflict_pixels = output_.conflict_counts[lower];
    }
    void update_age(std::uint64_t now) {
      output_.probe_age_ms = have_accepted_ && now >= output_.accepted_tick ? now - output_.accepted_tick : 0;
    }
    void clear_release() { releasing_ = false; release_since_ = 0; }
    void clear_approach() { approaching_ = false; approach_since_ = 0; }
    void clear_evidence() { clear_release(); clear_approach(); }
    void suspend(const char *reason) { clear_evidence(); ramp_armed_ = false; output_.status = reason; }
    bool reject(const char *reason) { suspend(reason); return false; }

    decision output_;
    source scope_, live_;
    double applied_{};
    float cap_{}, approach_candidate_{}, release_candidate_{};
    std::uint64_t clock_{}, accepted_at_{}, last_retained_mask_{}, release_since_{}, approach_since_{}, cap_since_{};
    bool have_scope_{}, have_live_{}, have_clock_{}, have_accepted_{}, have_cap_{}, ramp_armed_{}, releasing_{}, approaching_{};
  };
}

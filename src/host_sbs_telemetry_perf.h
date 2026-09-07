#pragma once

#include "host_sbs_gpu_outcomes.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

// Diagnostic data has no rendering or scheduling authority. Each collector belongs to one
// encoder generation; queued old packets retain only their old collector during a rebuild.
namespace host_sbs_telemetry {
  using clock_t = std::chrono::steady_clock;

  inline std::uint32_t host_ms(clock_t::time_point now) noexcept {
    return static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
  }
  enum class stage : std::size_t {
    conversion,
    encode,
    new_content_age,
    warp,
    preprocess,
    conditional_transaction,
    postprocess,
    output,
    count,
  };
  constexpr std::uint32_t outcomes_valid = 1u << 11;
  constexpr std::uint32_t outputs_valid = 1u << 12;

  constexpr std::uint32_t stage_valid(std::size_t index) {
    return 1u << (13 + index);
  }

  struct stage_sample {
    float mean_ms = 0;
    std::uint32_t count = 0;
    std::uint32_t latest_host_ms = 0;
  };

  struct sample {
    std::uint32_t valid_fields = 0;
    std::uint32_t outcome_epoch = 0;
    std::uint32_t outcome_sequence = 0;
    std::uint32_t outcome_host_ms = 0;
    models::host_sbs_gpu_outcomes::counts_t outcomes;
    std::uint32_t output_host_ms = 0;
    std::uint32_t warped = 0, repeated = 0, flat = 0;
    std::array<stage_sample, static_cast<std::size_t>(stage::count)> stages {};
  };

  class collector {
  public:
    void record(stage which, double elapsed_ms, clock_t::time_point measured_at) {
      const auto index = static_cast<std::size_t>(which);
      if (index >= sums.size() || !std::isfinite(elapsed_ms) || elapsed_ms < 0 || elapsed_ms > std::numeric_limits<float>::max()) {
        return;
      }
      std::lock_guard lock(mutex);
      auto &entry = current.stages[index];
      // Saturation is bounded and avoids inventing a fresh window after integer overflow.
      if (entry.count == std::numeric_limits<std::uint32_t>::max()) {
        return;
      }
      sums[index] += elapsed_ms;
      entry.mean_ms = static_cast<float>(sums[index] / ++entry.count);
      entry.latest_host_ms = host_ms(measured_at);
      current.valid_fields |= stage_valid(index);
    }

    void record_output(bool packed_repeat, bool warped, clock_t::time_point measured_at) {
      std::lock_guard lock(mutex);
      auto &count = packed_repeat ? current.repeated : warped ? current.warped :
                                                                current.flat;
      if (count != std::numeric_limits<std::uint32_t>::max()) {
        ++count;
      }
      current.output_host_ms = host_ms(measured_at);
      current.valid_fields |= outputs_valid;
    }

    void record_outcomes(models::host_sbs_gpu_outcomes::counts_t counts, clock_t::time_point copied_at) {
      std::lock_guard lock(mutex);
      if (outcome_restart_pending || current.outcome_epoch == 0 || counts.infer < current.outcomes.infer || counts.reuse < current.outcomes.reuse || counts.invalid < current.outcomes.invalid) {
        if (++current.outcome_epoch == 0) {
          ++current.outcome_epoch;
        }
      }
      outcome_restart_pending = false;
      current.outcomes = counts;
      if (++current.outcome_sequence == 0) {
        ++current.outcome_sequence;
      }
      current.outcome_host_ms = host_ms(copied_at);
      current.valid_fields |= outcomes_valid;
    }

    void invalidate_outcomes() {
      std::lock_guard lock(mutex);
      current.valid_fields &= ~outcomes_valid;
      outcome_restart_pending = true;
    }

    [[nodiscard]] sample snapshot() const {
      std::lock_guard lock(mutex);
      return current;
    }

  private:
    mutable std::mutex mutex;
    sample current;
    bool outcome_restart_pending = true;
    std::array<double, static_cast<std::size_t>(stage::count)> sums {};
  };
}  // namespace host_sbs_telemetry

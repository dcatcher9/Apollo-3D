/**
 * @file src/host_sbs_gpu_outcomes.h
 * @brief Diagnostic-only cumulative GPU outcomes, independent of the Dump ring ABI.
 */
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

namespace models::host_sbs_gpu_outcomes {
  inline constexpr std::uint32_t schema = 1u;
  inline constexpr std::uint32_t tag = 0x314F5447u;  // GTO1.
  inline constexpr std::uint32_t word_count = 8u;
  inline constexpr std::uint32_t byte_count = word_count * sizeof(std::uint32_t);
  using words_t = std::array<std::uint32_t, word_count>;
  inline constexpr words_t initial_words {schema, tag, 0u, 0u, 0u, 0u, 0u, 0u};

  struct counts_t {
    std::uint64_t infer = 0u;
    std::uint64_t reuse = 0u;
    std::uint64_t invalid = 0u;

    [[nodiscard]] double reuse_percent() const noexcept {
      const auto valid = static_cast<double>(infer) + static_cast<double>(reuse);
      return valid > 0.0 ? 100.0 * static_cast<double>(reuse) / valid : 0.0;
    }
  };

  [[nodiscard]] inline std::optional<counts_t> decode(const words_t &words) noexcept {
    if (words[0u] != schema || words[1u] != tag) {
      return std::nullopt;
    }
    const auto counter = [&words](const std::uint32_t low) {
      return static_cast<std::uint64_t>(words[low]) |
             (static_cast<std::uint64_t>(words[low + 1u]) << 32u);
    };
    return counts_t {counter(2u), counter(4u), counter(6u)};
  }

  // Missing or late snapshots do not lose events: every observation is cumulative. GPU
  // low-word carry is decoded before differencing. Resource/session resets start a new epoch;
  // a regression also resets the baseline defensively, including a full uint64 overflow.
  class delta_tracker_t {
  public:
    void reset() noexcept {
      previous = {};
      pending = {};
    }

    [[nodiscard]] bool observe(const counts_t current) noexcept {
      const bool restarted = current.infer < previous.infer ||
                             current.reuse < previous.reuse ||
                             current.invalid < previous.invalid;
      if (restarted) {
        reset();
      }
      pending.infer += current.infer - previous.infer;
      pending.reuse += current.reuse - previous.reuse;
      pending.invalid += current.invalid - previous.invalid;
      previous = current;
      return restarted;
    }

    [[nodiscard]] counts_t totals() const noexcept {
      return previous;
    }

    [[nodiscard]] counts_t take_delta() noexcept {
      return std::exchange(pending, {});
    }

  private:
    counts_t previous;
    counts_t pending;
  };
}  // namespace models::host_sbs_gpu_outcomes

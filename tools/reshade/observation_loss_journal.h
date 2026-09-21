// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sunshine_streamline::loss_diagnostics {
  enum class reason {
    records_busy, commands_busy, resource_table_full, invalid_input, invalid_camera,
    camera_reset, tag_replaced, sdk_failure, token_replaced, lifecycle, test_injected, tokens_busy,
    metadata_storage_busy
  };
  inline const char *name(reason value) noexcept {
    switch (value) {
#define SUNSHINE_LOSS_NAME(value) case reason::value: return #value
      SUNSHINE_LOSS_NAME(records_busy);
      SUNSHINE_LOSS_NAME(tokens_busy);
      SUNSHINE_LOSS_NAME(metadata_storage_busy);
      SUNSHINE_LOSS_NAME(commands_busy);
      SUNSHINE_LOSS_NAME(resource_table_full);
      SUNSHINE_LOSS_NAME(invalid_input);
      SUNSHINE_LOSS_NAME(invalid_camera);
      SUNSHINE_LOSS_NAME(camera_reset);
      SUNSHINE_LOSS_NAME(tag_replaced);
      SUNSHINE_LOSS_NAME(sdk_failure);
      SUNSHINE_LOSS_NAME(token_replaced);
      SUNSHINE_LOSS_NAME(lifecycle);
      SUNSHINE_LOSS_NAME(test_injected);
#undef SUNSHINE_LOSS_NAME
    }
    return "unknown";
  }
  struct context {
    std::uint64_t sequence{};
    std::uint32_t viewport{UINT32_MAX}, feature{UINT32_MAX};
    std::int32_t sdk_result{};
    bool has_sdk_result{};
    std::uint32_t reset{UINT32_MAX};
  };
  struct event {
    std::uint64_t revision{}, tick{};
    std::uint32_t thread_id{}, line{};
    reason cause{};
    // Call sites supply static storage, normally __func__ or a string literal.
    // The journal copies this pointer; it does not own or allocate text.
    const char *site{};
    context details;
  };

  // Diagnostic evidence only. Revisions come from the observation owner; this
  // journal neither generates revisions nor changes camera/source readiness.
  // Revision zero is unavailable. A contended slot drops the operation rather
  // than waiting, so an absent entry must not be interpreted as a loss cause.
  class journal {
  public:
    static constexpr std::size_t capacity = 64;

    bool publish(const event &value) noexcept {
      if (!value.revision) return false;
      auto &entry = slots_[value.revision % capacity];
      try_lock held(entry.lock);
      if (!held || entry.value.revision > value.revision) return false;
      entry.value = value;
      return true;
    }
    bool query(std::uint64_t revision, event &out) const noexcept {
      out = {};
      if (!revision) return false;
      const auto &entry = slots_[revision % capacity];
      try_lock held(entry.lock);
      if (!held || entry.value.revision != revision) return false;
      out = entry.value;
      return true;
    }

  private:
    struct slot {
      mutable std::atomic_flag lock = ATOMIC_FLAG_INIT;
      event value;
    };
    class try_lock {
    public:
      explicit try_lock(std::atomic_flag &value) noexcept:
        flag_(value), acquired_(!value.test_and_set(std::memory_order_acquire)) {}
      ~try_lock() { if (acquired_) flag_.clear(std::memory_order_release); }
      try_lock(const try_lock &) = delete;
      try_lock &operator=(const try_lock &) = delete;
      explicit operator bool() const noexcept { return acquired_; }
    private:
      std::atomic_flag &flag_;
      bool acquired_;
    };
    std::array<slot, capacity> slots_{};
#ifdef SUNSHINE_OBSERVATION_LOSS_JOURNAL_TEST
    friend struct journal_test_access;
#endif
  };
}

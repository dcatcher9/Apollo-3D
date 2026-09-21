// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace sunshine_streamline::observation {
  enum class write_failure { none, owner_busy, storage_busy };

  // Private observer metadata publication. Readers pin an immutable version;
  // they never own the writer flag. Storage, acquisition attempts and copies
  // are bounded. The owner must outlive every pin and transaction.
  //
  // Reclamation destroys T only after releasing the writer flag. A transaction
  // pins its base until commit/abort releases that flag, keeping inherited
  // leases alive across mutation. Callers must likewise retain newly supplied
  // leases until the transaction ends, and check their observation generation
  // before committing. Default T must contain no external leases. This class
  // does not define observation validity.
  template<class T, std::size_t Slots = 8>
  class snapshot_owner {
    static_assert(Slots >= 2 && Slots <= 256);
    static_assert(std::is_nothrow_default_constructible_v<T>);
    static_assert(std::is_nothrow_copy_assignable_v<T>);
    static_assert(std::is_nothrow_destructible_v<T>);
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
    static constexpr std::uint32_t writing = 0x80000000u;
    static constexpr std::uint32_t retired = 0x40000000u;
    static constexpr std::uint64_t index_mask = 255;
    static constexpr std::uint64_t ticket_step = 256;
    static constexpr std::size_t no_slot = Slots;

    struct slot {
      std::atomic<std::uint32_t> state{}; // Ownership flags and a reader count.
      T value;
    };

  public:
    class read_pin {
    public:
      read_pin() noexcept = default;
      read_pin(const read_pin &) = delete;
      read_pin &operator=(const read_pin &) = delete;
      read_pin(read_pin &&other) noexcept { take(other); }
      read_pin &operator=(read_pin &&other) noexcept {
        if (this != &other) { reset(); take(other); }
        return *this;
      }
      ~read_pin() { reset(); }
      explicit operator bool() const noexcept { return owner_ != nullptr; }
      const T *operator->() const noexcept { return &owner_->slots_[index_].value; }
      const T &operator*() const noexcept { return owner_->slots_[index_].value; }
      std::uint64_t ticket() const noexcept { return ticket_; }
      void reset() noexcept {
        auto *owner = owner_;
        const auto index = index_;
        owner_ = nullptr; index_ = no_slot; ticket_ = 0;
        if (owner) owner->unpin(index);
      }
    private:
      friend class snapshot_owner;
      read_pin(snapshot_owner *owner, std::size_t index, std::uint64_t ticket) noexcept:
        owner_(owner), index_(index), ticket_(ticket) {}
      void take(read_pin &other) noexcept {
        owner_ = std::exchange(other.owner_, nullptr);
        index_ = std::exchange(other.index_, no_slot);
        ticket_ = std::exchange(other.ticket_, 0);
      }
      snapshot_owner *owner_{};
      std::size_t index_{no_slot};
      std::uint64_t ticket_{};
    };

    class transaction {
    public:
      transaction() noexcept = default;
      transaction(const transaction &) = delete;
      transaction &operator=(const transaction &) = delete;
      transaction(transaction &&other) noexcept { take(other); }
      transaction &operator=(transaction &&other) noexcept {
        if (this != &other) { reset(); take(other); }
        return *this;
      }
      ~transaction() { reset(); }
      explicit operator bool() const noexcept { return owner_ != nullptr; }
      T *operator->() noexcept { return &owner_->slots_[index_].value; }
      T &operator*() noexcept { return owner_->slots_[index_].value; }
      write_failure failure() const noexcept { return failure_; }
      bool commit() noexcept {
        if (!owner_) return false;
        auto *owner = owner_;
        // Never wrap a publication ticket and make an ancient pin look current.
        if (owner->generation_ == std::numeric_limits<std::uint64_t>::max() / ticket_step) {
          failure_ = write_failure::storage_busy;
          reset();
          return false;
        }
        const auto ticket = ++owner->generation_ * ticket_step + index_;
        // Finish candidate readiness before redirecting readers. The writer
        // owner excludes reserve/clear until publication. A delayed reader of
        // this slot's old generation cannot pass the full-ticket recheck.
        // If this thread is preempted here, readers still see the complete base.
        owner->slots_[index_].state.store(0, std::memory_order_release);
#ifdef SUNSHINE_OBSERVER_SNAPSHOT_TEST
        if (owner->commit_interlock_) owner->commit_interlock_(owner->commit_context_);
#endif
        owner->published_.store(ticket, std::memory_order_release);
        // The base pin prevents last-reader retirement until the publication
        // has moved away and the writer has been released.
        if (base_) owner->slots_[static_cast<std::size_t>(base_.ticket() & index_mask)].state.fetch_or(retired, std::memory_order_acq_rel);
        owner_ = nullptr; index_ = no_slot;
        owner->writer_.clear(std::memory_order_release);
        base_.reset();
        return true;
      }
      void reset() noexcept {
        auto *owner = owner_;
        const auto index = index_;
        owner_ = nullptr; index_ = no_slot;
        if (!owner) return;
        owner->writer_.clear(std::memory_order_release);
        base_.reset();
        owner->rebuild(index); // Candidate is still write-owned and unpublished.
        owner->slots_[index].state.store(0, std::memory_order_release);
      }
      void abort() noexcept { reset(); }
    private:
      friend class snapshot_owner;
      explicit transaction(write_failure failure) noexcept: failure_(failure) {}
      transaction(snapshot_owner *owner, std::size_t index, read_pin base) noexcept:
        owner_(owner), index_(index), base_(std::move(base)) {}
      void take(transaction &other) noexcept {
        owner_ = std::exchange(other.owner_, nullptr);
        index_ = std::exchange(other.index_, no_slot);
        base_ = std::move(other.base_);
        failure_ = other.failure_;
      }
      snapshot_owner *owner_{};
      std::size_t index_{no_slot};
      read_pin base_;
      write_failure failure_{write_failure::none};
    };

    snapshot_owner() noexcept = default;
    snapshot_owner(const snapshot_owner &) = delete;
    snapshot_owner &operator=(const snapshot_owner &) = delete;

    bool has_publication() const noexcept {
      return published_.load(std::memory_order_acquire) != 0;
    }

    read_pin read() noexcept {
      for (std::size_t attempt = 0; attempt != Slots; ++attempt) {
        const auto ticket = published_.load(std::memory_order_acquire);
        if (!ticket) return {};
#ifdef SUNSHINE_OBSERVER_SNAPSHOT_TEST
        if (read_interlock_) read_interlock_(read_context_, ticket);
#endif
        const auto index = static_cast<std::size_t>(ticket & index_mask);
        auto &entry = slots_[index];
        auto state = entry.state.load(std::memory_order_relaxed);
        if (state >= retired - 1 || !entry.state.compare_exchange_strong(state, state + 1,
              std::memory_order_acq_rel, std::memory_order_relaxed)) continue;
        // Full generation+index comparison prevents reuse/ABA from accepting
        // a different version after the first publication load.
        if (published_.load(std::memory_order_acquire) == ticket)
          return read_pin(this, index, ticket);
        unpin(index);
      }
      return {};
    }

    transaction try_write() noexcept {
      if (writer_.test_and_set(std::memory_order_acquire))
        return transaction(write_failure::owner_busy);
      const auto index = reserve();
      if (index == no_slot) {
        writer_.clear(std::memory_order_release);
        return transaction(write_failure::storage_busy);
      }
      // Every reusable slot is default-empty: abort, lifecycle clear and last
      // retired pin rebuild it before releasing ownership. Copy assignment
      // therefore cannot release an old candidate's external lease here.
      read_pin base;
      const auto ticket = published_.load(std::memory_order_acquire);
      if (ticket) {
        const auto current = static_cast<std::size_t>(ticket & index_mask);
        // The writer owner excludes publication/reclamation, so the current
        // version cannot be write-owned. Reader-count saturation fails closed.
        const auto prior = slots_[current].state.fetch_add(1, std::memory_order_acq_rel);
        if (prior >= retired - 1) {
          slots_[current].state.fetch_sub(1, std::memory_order_release);
          writer_.clear(std::memory_order_release);
          slots_[index].state.store(0, std::memory_order_release);
          return transaction(write_failure::storage_busy);
        }
        base = read_pin(this, current, ticket);
        slots_[index].value = slots_[current].value;
      }
      return transaction(this, index, std::move(base));
    }

    // Lifecycle callers close observation admission first. This operation may
    // fail only because a writer is active; it never waits for reader pins.
    // Retired readers remain valid values, not valid observation generations.
    bool clear() noexcept {
      if (writer_.test_and_set(std::memory_order_acquire)) return false;
      published_.store(0, std::memory_order_release);
      std::array<bool, Slots> claimed{};
      for (std::size_t i = 0; i != Slots; ++i) {
        // The fixed RMW closes reader admission without waiting for their count
        // to stop changing. An existing reclaimer may temporarily see both
        // ownership flags; it always finishes by storing zero after rebuild.
        const auto prior = slots_[i].state.fetch_or(retired, std::memory_order_acq_rel);
        if ((prior & writing) || (prior & ~(writing | retired)) != 0) continue;
        auto expected = retired;
        claimed[i] = slots_[i].state.compare_exchange_strong(expected, writing, std::memory_order_acq_rel);
        // A failed claim belongs to the final reader's reclaimer, or a delayed
        // pre-clear reader that has already failed its full-ticket recheck.
      }
      writer_.clear(std::memory_order_release);
      for (std::size_t i = 0; i != Slots; ++i) if (claimed[i]) {
        rebuild(i);
        slots_[i].state.store(0, std::memory_order_release);
      }
      return true;
    }

  private:
    // All reservation decisions are made under the writer owner, preventing a
    // racing publication from turning a reusable slot into the current slot.
    std::size_t reserve() noexcept {
      const auto ticket = published_.load(std::memory_order_acquire);
      const auto current = ticket ? static_cast<std::size_t>(ticket & index_mask) : no_slot;
      for (std::size_t i = 0; i != Slots; ++i) {
        if (i == current) continue;
        std::uint32_t expected = 0;
        if (slots_[i].state.compare_exchange_strong(expected, writing, std::memory_order_acq_rel)) return i;
      }
      return no_slot;
    }
    void unpin(std::size_t index) noexcept {
      if (slots_[index].state.fetch_sub(1, std::memory_order_acq_rel) != retired + 1) return;
      auto expected = retired;
      if (!slots_[index].state.compare_exchange_strong(expected, writing, std::memory_order_acq_rel)) return;
      // A retired version cannot be published. Its last reader owns cleanup
      // without ever taking the writer flag, including on lifecycle shutdown.
      rebuild(index);
      slots_[index].state.store(0, std::memory_order_release);
    }
    void rebuild(std::size_t index) noexcept {
      // Placement construction avoids a large temporary T on callback stacks.
      slots_[index].value.~T();
      ::new (static_cast<void *>(&slots_[index].value)) T();
    }
    std::array<slot, Slots> slots_{};
    std::atomic_flag writer_ = ATOMIC_FLAG_INIT;
    std::atomic<std::uint64_t> published_{};
    std::uint64_t generation_{}; // Writer-owned; never reset by clear().
#ifdef SUNSHINE_OBSERVER_SNAPSHOT_TEST
    friend struct snapshot_test_access;
    void (*read_interlock_)(void *, std::uint64_t) noexcept{};
    void *read_context_{};
    void (*commit_interlock_)(void *) noexcept{};
    void *commit_context_{};
#endif
  };
}

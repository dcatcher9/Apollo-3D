// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "raw_scene_policy.h"

namespace sunshine_raw_scene {
  inline constexpr std::size_t maximum_sources = 4;
  inline constexpr std::size_t maximum_history_sources = 16;

  // Membership is authoritative resource metadata, never a fabricated current
  // frame. Matching dimensions or content statistics do not identify a basis.
  struct source_basis {
    source_key source;
    std::uint64_t layout_epoch{}, convention_epoch{};
    orientation direction{orientation::automatic};
  };
  struct source_roster {
    std::uint64_t routing_epoch{}, basis_epoch{};
    frame_key observed_frame{}; // Authoritative completed device-present watermark.
    std::uint32_t count{};
    std::array<source_basis, maximum_sources> members{};
  };

  class pool {
  public:
    void configure(sunshine_scene_gain::limits budget) noexcept {
      budget_ = budget;
      for (auto &entry : entries_) if (entry.occupied) entry.controller.configure(budget);
    }
    bool reset(std::uint64_t basis_epoch, std::uint64_t now_ms) noexcept {
      if (!basis_epoch || basis_epoch <= epoch_) return false;
      *this = pool{};
      epoch_ = basis_epoch;
      wall_ms_ = now_ms;
      have_wall_ = true;
      return true;
    }

    bool contains(const selected_frame &current) const noexcept {
      // Historical membership is sufficient for log classification, but never
      // for sample admission or rendering (which additionally require active).
      for (const auto &entry : entries_)
        if (entry.occupied && matches(entry.basis, current)) return true;
      return false;
    }

    // The caller validates this bounded immutable identity snapshot against the
    // authoritative live-resource inventory, then applies its same-slot mask
    // immediately before synchronize. No borrowed GPU handles or numeric state.
    std::array<source_basis, maximum_history_sources> history_sources() const noexcept {
      std::array<source_basis, maximum_history_sources> result{};
      for (std::size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].occupied) result[i] = entries_[i].basis;
      return result;
    }
    void retain_history(std::uint32_t live_mask) noexcept {
      for (std::size_t i = 0; i < entries_.size(); ++i)
        if (!(live_mask & (std::uint32_t{1} << i))) entries_[i] = {};
      // Validation and roster synchronization form one admission boundary.
      sync_status_ = status::uninitialized;
    }

    // Membership and the capture watermark are authoritative even on presents
    // without usable depth. A failed synchronization closes all later admission.
    bool synchronize(const source_roster &roster, std::uint64_t now_ms) noexcept {
      sync_status_ = status::uninitialized;
      if (!epoch_) return false;
      if (have_wall_ && now_ms < wall_ms_) return reject_sync(status::clock_went_backwards);
      wall_ms_ = now_ms;
      have_wall_ = true;
      if (roster.basis_epoch != epoch_) return reject_sync(status::basis_epoch_mismatch);
      if (!valid(roster)) return reject_sync(status::invalid_source);
      if (!roster.observed_frame.frame || !roster.observed_frame.token_generation || (have_observed_ &&
          (roster.observed_frame.frame < observed_frame_.frame ||
           roster.observed_frame.token_generation != observed_frame_.token_generation)))
        return reject_sync(status::frame_mismatch);
      observed_frame_ = roster.observed_frame;
      have_observed_ = true;
      synchronize_members(roster, now_ms);
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.advance(now_ms);
      sync_status_ = status::ready;
      return true;
    }

    // Completed samples update only their exact roster member, including an
    // off-turn member. Unrelated/future packets cannot poison its replay state.
    void observe(const sample &packet, std::uint64_t now_ms) noexcept {
      if (sync_status_ != status::ready || now_ms != wall_ms_ ||
          packet.metadata.basis_epoch != epoch_ ||
          packet.metadata.frame.frame > observed_frame_.frame ||
          packet.metadata.frame.token_generation != observed_frame_.token_generation) return;
      for (auto &entry : entries_)
        if (entry.occupied && entry.active && matches(entry.basis, packet.metadata)) {
          entry.controller.configure(budget_);
          entry.controller.observe(packet, observed_frame_, now_ms);
          return;
        }
    }

    // Presentation decides only whether this current pair can use its model.
    // No previous depth pixels are returned or made current by numeric history.
    output evaluate(const selected_frame &current, std::uint64_t now_ms) noexcept {
      if (sync_status_ != status::ready) return failure(sync_status_);
      if (now_ms != wall_ms_) return failure(status::clock_went_backwards);
      if (current.basis_epoch != epoch_) {
        invalidate_all();
        return failure(status::basis_epoch_mismatch);
      }
      if (!current.frame.frame && !current.depth_ready) {
        suspend_all();
        return failure(status::depth_unavailable);
      }
      if (!current.frame.frame || current.frame.frame != observed_frame_.frame ||
          current.frame.token_generation != observed_frame_.token_generation || (have_current_ &&
          (current.frame.frame < current_frame_.frame ||
           (current.frame.frame == current_frame_.frame && !(current.frame == current_frame_))))) {
        suspend_all();
        return failure(status::frame_mismatch);
      }
      current_frame_ = current.frame;
      have_current_ = true;
      if (!current.depth_ready || current.copy_ambiguous || !current.aligned_viewport_assumed) {
        suspend_all();
        return failure(!current.depth_ready ? status::depth_unavailable :
          current.copy_ambiguous ? status::ambiguous_copy : status::alignment_unavailable);
      }
      for (auto &entry : entries_)
        if (entry.occupied && entry.active && matches(entry.basis, current))
          return entry.controller.evaluate(current, now_ms);
      suspend_all();
      return failure(status::source_mismatch);
    }

    output update(const source_roster &roster, const selected_frame &current,
                  const sample *packet, std::uint64_t now_ms) noexcept {
      synchronize(roster, now_ms);
      if (packet) observe(*packet, now_ms);
      return evaluate(current, now_ms);
    }

    void scene_cut(std::uint64_t now_ms) noexcept {
      if (!epoch_ || (have_wall_ && now_ms < wall_ms_)) return;
      wall_ms_ = cut_ms_ = now_ms;
      have_wall_ = have_cut_ = true;
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.scene_cut(now_ms);
    }

  private:
    struct entry_t {
      source_basis basis;
      policy controller;
      std::uint64_t last_admitted_ms{};
      bool occupied{}, active{};
    };
    static bool same(const source_basis &a, const source_basis &b) noexcept {
      return a.source == b.source && a.layout_epoch == b.layout_epoch &&
        a.convention_epoch == b.convention_epoch && a.direction == b.direction;
    }
    static bool matches(const source_basis &a, const selected_frame &b) noexcept {
      return same(a, {b.source, b.layout_epoch, b.convention_epoch, b.direction});
    }
    static bool valid(const source_roster &roster) noexcept {
      if (!roster.routing_epoch || roster.count > maximum_sources) return false;
      for (std::size_t i = 0; i < roster.count; ++i) {
        const auto &member = roster.members[i];
        const auto &source = member.source;
        if (!member.layout_epoch || !source.native || !source.lifetime || !source.width || !source.height ||
            !source.extent_width || !source.extent_height || source.left >= source.width || source.top >= source.height ||
            source.extent_width > source.width - source.left || source.extent_height > source.height - source.top ||
            (member.direction != orientation::automatic && member.direction != orientation::normal &&
             member.direction != orientation::reversed)) return false;
        for (std::size_t j = 0; j < i; ++j)
          if (member.source.native == roster.members[j].source.native) return false;
      }
      return true;
    }
    void synchronize_members(const source_roster &roster, std::uint64_t now_ms) noexcept {
      // A capture slot is temporary. Keep an exact still-live model outside the
      // active roster, while its targets continue to expire normally. Protect
      // every returning member before choosing an inactive LRU slot for a new
      // source, so roster order cannot evict another admitted source's history.
      for (auto &entry : entries_) {
        bool active = false;
        for (std::size_t i = 0; entry.occupied && i < roster.count; ++i)
          active = active || same(entry.basis, roster.members[i]);
        if (entry.active != active) entry.controller.suspend();
        entry.active = active;
      }
      for (std::size_t i = 0; i < roster.count; ++i) {
        auto *entry = static_cast<entry_t *>(nullptr);
        for (auto &candidate : entries_) {
          if (!candidate.occupied) continue;
          if (same(candidate.basis, roster.members[i])) entry = &candidate;
          else if (candidate.basis.source.native == roster.members[i].source.native)
            candidate = {}; // Changed allocation/basis can never resurrect an older key.
        }
        if (entry) { entry->last_admitted_ms = now_ms; continue; }
        for (auto &candidate : entries_)
          if (!candidate.occupied) { entry = &candidate; break; }
        if (!entry)
          for (auto &candidate : entries_)
            if (!candidate.active && (!entry || candidate.last_admitted_ms < entry->last_admitted_ms)) entry = &candidate;
        // There are at most four protected members and sixteen history slots.
        if (!entry) continue;
        *entry = {};
        entry->occupied = entry->active = true;
        entry->last_admitted_ms = now_ms;
        entry->basis = roster.members[i];
        // A genuinely new or evicted basis still starts after current admission.
        entry->controller.reset(epoch_, now_ms);
        entry->controller.configure(budget_);
        selected_frame basis;
        basis.basis_epoch = epoch_;
        basis.source = entry->basis.source;
        basis.layout_epoch = entry->basis.layout_epoch;
        basis.convention_epoch = entry->basis.convention_epoch;
        basis.direction = entry->basis.direction;
        entry->controller.bind(basis, now_ms);
        if (have_cut_ && now_ms == cut_ms_) entry->controller.scene_cut(now_ms);
      }
    }
    bool reject_sync(status reason) noexcept {
      sync_status_ = reason;
      invalidate_all();
      return false;
    }
    void invalidate_all() noexcept {
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.invalidate(wall_ms_);
    }
    void suspend_all() noexcept {
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.suspend();
    }
    static output failure(status reason) noexcept {
      output value;
      value.reason = reason;
      return value;
    }

    std::array<entry_t, maximum_history_sources> entries_{};
    sunshine_scene_gain::limits budget_;
    frame_key current_frame_{}, observed_frame_{};
    status sync_status_{status::uninitialized};
    std::uint64_t epoch_{}, wall_ms_{}, cut_ms_{};
    bool have_wall_{}, have_current_{}, have_cut_{}, have_observed_{};
  };
}

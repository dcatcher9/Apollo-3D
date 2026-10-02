// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"
#include "raw_scene_policy.h"
#include "raw_scene_pool.h"

#include <algorithm>

// The API adapter owns a logical depth encoding across rotating GPU resources.
// Feed that identity to the reference controller; texture addresses,
// allocation padding and dynamic render size do not define a new encoding.
namespace sunshine_provided_raw {
  inline sunshine_raw_scene::selected_frame selected(const sunshine_scene_depth::frame &value,
      std::uint64_t basis_epoch, bool ready) noexcept {
    sunshine_raw_scene::selected_frame out;
    out.basis_epoch = basis_epoch;
    out.layout_epoch = value.epoch;
    // This is a logical normalized domain, never a native GPU resource handle.
    const auto generation = value.source_id ? value.source_id :
      value.provider == sunshine_scene_depth::provider_kind::streamline ? value.epoch : 0;
    out.source = {static_cast<std::uint64_t>(value.provider) + 1, generation,
      value.viewport, 1, 1, 0, 0, 1, 1};
    out.frame = {value.sequence, value.epoch};
    if (value.projection.supplied || value.projection.direction_supplied)
      out.direction = value.projection.reversed ? sunshine_depth::depth_orientation::reversed :
        sunshine_depth::depth_orientation::normal;
    out.depth_ready = ready;
    out.aligned_viewport_assumed = true; // Shader and readback both use the adapter's explicit active rect.
    out.feedback = value.feedback;
    return out;
  }

  inline sunshine_raw_scene::sample measured(const sunshine_scene_depth::frame &value,
      std::uint64_t id, std::uint64_t basis_epoch,
      const std::array<float, sunshine_raw_scene::grid_width * sunshine_raw_scene::grid_height> &raw) noexcept {
    sunshine_raw_scene::sample out;
    out.id = id;
    out.capture_ms = value.tick;
    out.metadata = selected(value, basis_epoch, true);
    out.readback_frame = out.metadata.frame;
    out.readback_source = out.metadata.source;
    out.readback_layout_epoch = out.metadata.layout_epoch;
    out.raw = raw;
    return out;
  }

  // Frame generation moves presentation between providers and between the raw
  // and projection paths. Keep one exact-key reference per logical encoding,
  // each with its own sequence and sample watermarks, so a returning source does
  // not repeat its startup window. Inactive entries receive no packets, so a
  // returning entry keeps gain and zero but renders only after a target captured
  // after its return. A reset or revision revokes that evidence, not the
  // reference. Identity change, LRU eviction, basis epoch and runtime reset
  // discard it; there is no age cap and no production Recenter.
  class retained_policy {
    using policy = sunshine_raw_scene::policy;
    using output = sunshine_raw_scene::output;
    using status = sunshine_raw_scene::status;
    using selected_frame = sunshine_raw_scene::selected_frame;

  public:
    bool reset(std::uint64_t basis_epoch, std::uint64_t now_ms) noexcept {
      if (!basis_epoch || basis_epoch <= epoch_) return false;
      restart(basis_epoch, now_ms);
      return true;
    }
    // Fixture Recenter on the projection path: forget every encoding without
    // a new basis epoch, and require captures strictly after this action.
    void discard(std::uint64_t now_ms) noexcept {
      if (!epoch_) return;
      const auto budget = budget_;
      const auto at = std::max(wall_ms_, now_ms);
      restart(epoch_, at);
      configure(budget);
      scene_cut(at);
    }
    void configure(sunshine_scene_gain::limits budget) noexcept {
      budget_ = budget;
      fallback_.configure(budget);
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.configure(budget);
    }
    // Every retained encoding, including an absent one, needs a target captured
    // strictly after the cut. Encodings admitted later inherit it.
    void scene_cut(std::uint64_t now_ms) noexcept {
      if (!epoch_ || now_ms < wall_ms_) return;
      wall_ms_ = cut_ms_ = now_ms;
      have_cut_ = true;
      fallback_.scene_cut(now_ms);
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.scene_cut(now_ms);
    }
    // The projection path took this present. Generic interludes do not leave,
    // so selection flicker cannot starve the active encoding of fresh targets.
    void leave() noexcept {
      if (auto *entry = active()) {
        entry->controller.suspend();
        entry->departed = true;
      }
    }

    void bind(const selected_frame &basis, std::uint64_t now_ms) noexcept {
      if (!advance(now_ms)) return;
      // An unknown identity is not another encoding: as with one policy, the
      // active encoding is only suspended and continues when it is seen again.
      if (basis.basis_epoch != epoch_ || !policy::valid_source(basis)) {
        if (auto *entry = active()) entry->controller.suspend();
        return;
      }
      auto *entry = find(basis);
      const bool returning = entry && (entry != active() || entry->departed);
      if (!entry) entry = admit(basis, now_ms);
      entry->controller.bind(basis, now_ms);
      if (returning) {
        entry->controller.invalidate(now_ms);
        entry->reentry_pending = true;
      }
      if (auto *previous = active(); previous && previous != entry) previous->controller.suspend();
      active_ = static_cast<std::size_t>(entry - entries_.data());
      entry->departed = false;
      entry->last_used_ms = now_ms;
    }
    void observe(const sunshine_raw_scene::sample &value, const sunshine_raw_scene::frame_key &observed_frame,
        std::uint64_t now_ms) noexcept {
      if (!advance(now_ms)) return;
      if (auto *entry = active(); entry && !entry->departed) entry->controller.observe(value, observed_frame, now_ms);
    }
    output evaluate(const selected_frame &current, std::uint64_t now_ms) noexcept {
      if (!epoch_) return fallback_.evaluate(current, now_ms);
      if (!advance(now_ms)) {
        output out;
        out.reason = status::clock_went_backwards;
        return out;
      }
      auto *entry = active();
      if (current.basis_epoch != epoch_) invalidate_all(now_ms);
      else if (entry && !entry->departed && policy::same_basis(current, entry->basis)) {
        auto out = entry->controller.evaluate(current, now_ms);
        // A retained reference is not evidence of the current scene.
        if (entry->reentry_pending && out.reason == status::holding_reference) {
          out.ready = false;
          out.reason = status::no_target;
        } else if (out.reason == status::ready) {
          entry->reentry_pending = false;
        }
        return out;
      }
      if (entry) entry->controller.suspend();
      return fallback_.evaluate(current, now_ms);
    }
    // Fixture convenience in the production call order.
    output update(const selected_frame &current, const sunshine_raw_scene::sample *packet, std::uint64_t now_ms) noexcept {
      bind(current, now_ms);
      if (packet) observe(*packet, current.frame, now_ms);
      return evaluate(current, now_ms);
    }

  private:
    struct entry_t {
      selected_frame basis;
      policy controller;
      std::uint64_t last_used_ms{};
      bool occupied{}, departed{}, reentry_pending{};
    };
    static constexpr std::size_t none = sunshine_raw_scene::maximum_sources;

    void restart(std::uint64_t basis_epoch, std::uint64_t now_ms) noexcept {
      *this = retained_policy{};
      epoch_ = basis_epoch;
      wall_ms_ = reset_ms_ = now_ms;
      fallback_.reset(basis_epoch, now_ms);
    }
    bool advance(std::uint64_t now_ms) noexcept {
      if (!epoch_) return false;
      if (now_ms < wall_ms_) {
        invalidate_all(wall_ms_);
        return false;
      }
      wall_ms_ = now_ms;
      return true;
    }
    void invalidate_all(std::uint64_t now_ms) noexcept {
      for (auto &entry : entries_)
        if (entry.occupied) entry.controller.invalidate(now_ms);
    }
    entry_t *active() noexcept { return active_ < entries_.size() ? &entries_[active_] : nullptr; }
    entry_t *find(const selected_frame &basis) noexcept {
      for (auto &entry : entries_)
        if (entry.occupied && policy::same_basis(entry.basis, basis)) return &entry;
      return nullptr;
    }
    entry_t *admit(const selected_frame &basis, std::uint64_t now_ms) noexcept {
      entry_t *slot = nullptr;
      for (auto &entry : entries_)
        if (!entry.occupied) {
          slot = &entry;
          break;
        }
      if (!slot)
        for (auto &entry : entries_)
          if (&entry != active() && (!slot || entry.last_used_ms < slot->last_used_ms)) slot = &entry;
      // policy::reset refuses an equal epoch: value-initialize first, so a
      // reused slot keeps none of the evicted encoding's clock, cut or gain.
      *slot = {};
      slot->occupied = true;
      slot->basis = basis;
      // As with one policy, the first encoding after a reset may adopt a packet
      // captured since that reset; later admissions start at their own bind.
      slot->controller.reset(epoch_, admitted_ ? now_ms : reset_ms_);
      if (have_cut_) slot->controller.scene_cut(cut_ms_);
      slot->controller.configure(budget_);
      admitted_ = true;
      return slot;
    }

    std::array<entry_t, sunshine_raw_scene::maximum_sources> entries_{};
    policy fallback_; // Never bound: reports the single policy's rejection statuses.
    sunshine_scene_gain::limits budget_;
    std::uint64_t epoch_{}, wall_ms_{}, reset_ms_{}, cut_ms_{};
    std::size_t active_{none};
    bool have_cut_{}, admitted_{};
  };
}

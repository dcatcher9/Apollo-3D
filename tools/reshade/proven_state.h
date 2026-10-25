// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

// The per-resource last-proven state: the state a texture is proven to be in
// after the work executed on its device, for a copy whose own recording
// names nothing about it (the contract matrix's D10). The capture owner
// (streamline_depth_capture.cpp) feeds it under its lock: the texture's
// creation state (ReShade's init_resource), then, in submission order, the
// state the observed barriers of each executed list left it in. Whatever may
// have changed it unseen (an unobserved list, observer loss, an aliasing
// barrier, a split or blocked barrier) leaves it unproven until a later
// barrier proves it again.
//
// A proof belongs to the queue whose executed list made it: ordering across
// queues is not observed, so a copy on another queue finds it unproven. A
// creation state holds on every queue until a list uses the texture, except
// once enhanced barriers were seen (creation_trusted): ReShade reports a
// texture created with an initial layout in the same form as one created in a
// legacy state. Bounded: the least recently used texture is forgotten first,
// and a texture that is not tracked is unproven. Pure; the caller locks.
//
// The enhanced-layout latch (the contract matrix's D6 and N8): once an
// executed list's enhanced barrier named a texture, its layout belongs to
// enhanced barriers, and a legacy barrier from a claim (a declared state, an
// SDK contract) is illegal on it (Enhanced Barriers, legacy interop). The
// latch never clears while the texture is held: a later legacy barrier,
// unseen work or another queue leave it set. A texture is held tracked (its
// proofs serve the queries above) or only watched: an adapter's source,
// followed from its retention for the latch and for a copy the latch decides,
// never for the queries that admit a copy from a proof alone. Forgetting a
// latched texture (a full table) loses its latch, so from then on a texture
// that is not held counts as latched (latch_lost).
namespace sunshine_proven_state {
  struct state {
    std::uint64_t version {};  // Nonzero while proven; every update takes a new one.
    std::uint64_t queue {};  // The queue whose executed list proved it; 0: its creation.
    std::uint32_t value {};  // A legacy D3D12 state, when not enhanced.
    bool enhanced {};
    std::uint32_t layout {}, access {};  // The enhanced layout and access.

    explicit operator bool() const {
      return version != 0;
    }
  };

  // A texture's enhanced-layout latch, and its proof on the queue that made
  // it (never a creation state; empty while unproven).
  struct latch {
    bool latched {};
    state proven;
  };

  template<unsigned Capacity>
  class table {
  public:
    // A texture tracked from its creation, in initial (a legacy state) when
    // known. Tracking it again restarts it; a watched texture becomes
    // tracked, unproven until a later barrier proves it. The latch stays.
    void track(std::uint64_t resource, bool known, std::uint32_t initial) {
      if (!resource) {
        return;
      }
      auto *found = find(resource);
      if (!found) {
        found = insert(resource);
      }
      found->tracked = true;
      found->value = {};
      if (known) {
        found->value.version = ++serial;
        found->value.value = initial;
      }
      found->used = ++serial;
    }

    // A texture followed from now on, unproven, for its latch only; a texture
    // already held is unchanged.
    void watch(std::uint64_t resource) {
      if (resource && !find(resource)) {
        insert(resource)->used = ++serial;
      }
    }

    bool tracked(std::uint64_t resource) const {
      const auto *found = find(resource);
      return found && found->tracked;
    }

    // Tracked or watched: executed lists update it.
    bool held(std::uint64_t resource) const {
      return find(resource) != nullptr;
    }

    // An executed enhanced barrier named the texture.
    void set_latch(std::uint64_t resource) {
      if (auto *found = find(resource)) {
        found->latched = true;
      }
    }

    // An executed list on queue left the texture in a legacy state, or in an
    // enhanced layout with access.
    void prove(std::uint64_t resource, std::uint64_t queue, std::uint32_t value) {
      if (auto *found = find(resource)) {
        found->value = {++serial, queue, value};
        found->used = serial;
      }
    }

    void prove_layout(std::uint64_t resource, std::uint64_t queue, std::uint32_t layout, std::uint32_t access) {
      if (auto *found = find(resource)) {
        found->value = {++serial, queue, 0, true, layout, access};
        found->used = serial;
      }
    }

    // Something may have changed it unseen.
    void unprove(std::uint64_t resource) {
      if (auto *found = find(resource)) {
        found->value = {};
      }
    }

    void unprove_all() {
      for (unsigned i = 0; i != count; ++i) {
        entries[i].value = {};
      }
    }

    // The tracked texture's proven state for a copy executed on queue.
    state query(std::uint64_t resource, std::uint64_t queue, bool creation_trusted) {
      auto *found = find(resource);
      if (!found || !found->tracked || !found->value || (found->value.queue ? found->value.queue != queue : !creation_trusted)) {
        return {};
      }
      found->used = ++serial;
      return found->value;
    }

    // The held texture's latch and, when latched, its proof on whatever
    // queue made it. A texture not held is latched once a latch was lost.
    latch latched(std::uint64_t resource) {
      auto *found = find(resource);
      if (!found) {
        return {lost, {}};
      }
      latch out {found->latched, {}};
      if (found->latched && found->value && found->value.queue) {
        found->used = ++serial;
        out.proven = found->value;
      }
      return out;
    }

    // The version a proof still has (0: unproven or not held).
    std::uint64_t version(std::uint64_t resource) const {
      const auto *found = find(resource);
      return found ? found->value.version : 0;
    }

    // Held textures, tracked or watched.
    unsigned size() const {
      return count;
    }

    unsigned latched_count() const {
      return unsigned(std::count_if(entries.begin(), entries.begin() + count, [](const entry &value) {
        return value.latched;
      }));
    }

    bool latch_lost() const {
      return lost;
    }

  private:
    struct entry {
      std::uint64_t resource {}, used {};
      state value;
      bool tracked {}, latched {};
    };

    std::array<entry, Capacity> entries {};  // Sorted by resource.
    unsigned count {};
    std::uint64_t serial {};
    bool lost {};

    const entry *find(std::uint64_t resource) const {
      const auto end = entries.begin() + count;
      const auto found = std::lower_bound(entries.begin(), end, resource, [](const entry &value, std::uint64_t key) {
        return value.resource < key;
      });
      return found != end && found->resource == resource ? &*found : nullptr;
    }

    entry *find(std::uint64_t resource) {
      return const_cast<entry *>(static_cast<const table *>(this)->find(resource));
    }

    // A full table forgets its least recently used texture first.
    entry *insert(std::uint64_t resource) {
      if (count == Capacity) {
        const auto oldest = std::min_element(entries.begin(), entries.end(), [](const entry &a, const entry &b) {
          return a.used < b.used;
        });
        lost = lost || oldest->latched;
        std::move(oldest + 1, entries.end(), oldest);
        --count;
      }
      const auto end = entries.begin() + count;
      const auto at = std::lower_bound(entries.begin(), end, resource, [](const entry &value, std::uint64_t key) {
        return value.resource < key;
      });
      std::move_backward(at, end, end + 1);
      *at = entry {};
      at->resource = resource;
      ++count;
      return &*at;
    }
  };
}  // namespace sunshine_proven_state

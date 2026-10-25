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
// legacy state. Bounded: a destroyed texture is forgotten, then the least
// recently used one when the table is full, and a texture that is not
// tracked is unproven. Pure; the caller locks.
//
// The enhanced-layout latch (the contract matrix's D6 and N8): once an
// executed list's enhanced barrier named a texture, its layout belongs to
// enhanced barriers, and a legacy barrier from a claim (a declared state, an
// SDK contract) is illegal on it (Enhanced Barriers, legacy interop). The
// latch never clears while the texture is held: a later legacy barrier,
// unseen work or another queue leave it set. A texture is held tracked (its
// proofs serve the queries above) or only watched: an adapter's source,
// followed from its retention for the latch and for a copy the latch decides,
// never for the queries that admit a copy from a proof alone. A full table
// that forgets a latched texture keeps its identity in a bounded record:
// queried while not held, or held again (watched or adopted), it is latched.
// Once that record drops one, every texture that is not held counts as
// latched (latch_lost); a texture held again takes its latch from the record
// alone, since one first held then cannot be told from one forgotten long
// ago, and latching every such texture would refuse every new source's claim.
// A texture tracked from its creation is new and starts unlatched.
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

  // The barriers a texture's layout belongs to, for a proof that names no
  // barrier of its own (a depth clear): legacy states or enhanced layouts,
  // or unknown.
  enum class regime {
    legacy,
    enhanced,
    unknown
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
        found = insert(resource, false);
      }
      found->tracked = true;
      found->value = {};
      if (known) {
        found->value.version = ++serial;
        found->value.value = initial;
      }
      found->used = ++serial;
    }

    // A live texture tracked from now on, not from its creation: unproven
    // until a later barrier proves it. A watched texture becomes tracked and
    // keeps its latch; one not held is latched when the record holds it as
    // forgotten latched. A tracked texture is unchanged.
    void adopt(std::uint64_t resource) {
      if (!resource) {
        return;
      }
      auto *found = find(resource);
      if (found && found->tracked) {
        return;
      }
      if (!found) {
        found = insert(resource, true);
      }
      found->tracked = true;
      found->value = {};
      found->used = ++serial;
    }

    // A live texture followed from now on, unproven, for its latch only (as
    // adopt latches it); a texture already held is unchanged.
    void watch(std::uint64_t resource) {
      if (resource && !find(resource)) {
        insert(resource, true)->used = ++serial;
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

    // A destroyed texture is forgotten with its latch (never recorded), so
    // it neither takes a place nor loses a latch when the table is full.
    void forget(std::uint64_t resource) {
      if (auto *found = find(resource)) {
        std::move(found + 1, entries.begin() + count, found);
        --count;
      }
      std::replace(forgotten.begin(), forgotten.end(), resource, std::uint64_t {});
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
    // queue made it. A texture not held is latched when it was forgotten
    // latched, or once a latch was lost.
    latch latched(std::uint64_t resource) {
      auto *found = find(resource);
      if (!found) {
        return {forgotten_latched(resource), {}};
      }
      latch out {found->latched, {}};
      if (found->latched && found->value && found->value.queue) {
        found->used = ++serial;
        out.proven = found->value;
      }
      return out;
    }

    // The regime of a held texture that an executed list cleared after
    // naming it in no barrier: its proof's, made by an executed list; else
    // legacy, unless an enhanced barrier latched it (then unknown, so the
    // clear proves nothing).
    regime clear_regime(std::uint64_t resource) const {
      const auto *found = find(resource);
      if (found && found->value && found->value.queue) {
        return found->value.enhanced ? regime::enhanced : regime::legacy;
      }
      return (found ? found->latched : forgotten_latched(resource)) ? regime::unknown : regime::legacy;
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
    // Latched textures a full table forgot (0: a free slot); when none is
    // free, one is overwritten in turn and the latch is lost.
    std::array<std::uint64_t, Capacity> forgotten {};
    unsigned forgotten_next {};
    bool lost {};

    bool forgotten_latched(std::uint64_t resource) const {
      return lost || std::find(forgotten.begin(), forgotten.end(), resource) != forgotten.end();
    }

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

    // A full table forgets its least recently used texture first, and
    // records it when latched. A live texture held again (alive) takes its
    // latch from that record; a created one starts unlatched.
    entry *insert(std::uint64_t resource, bool alive) {
      bool latched = false;
      if (alive) {
        // The entry carries a remembered latch again, freeing its record
        // before an eviction below needs one.
        const auto remembered = std::find(forgotten.begin(), forgotten.end(), resource);
        latched = remembered != forgotten.end();
        if (latched) {
          *remembered = 0;
        }
      }
      if (count == Capacity) {
        const auto oldest = std::min_element(entries.begin(), entries.end(), [](const entry &a, const entry &b) {
          return a.used < b.used;
        });
        if (oldest->latched) {
          auto slot = std::find(forgotten.begin(), forgotten.end(), std::uint64_t {});
          if (slot == forgotten.end()) {
            slot = forgotten.begin() + forgotten_next++ % Capacity;
            lost = true;
          }
          *slot = oldest->resource;
        }
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
      at->latched = latched;
      ++count;
      return &*at;
    }
  };
}  // namespace sunshine_proven_state

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

  template<unsigned Capacity>
  class table {
  public:
    // A texture tracked from its creation, in initial (a legacy state) when
    // known. Tracking it again restarts it.
    void track(std::uint64_t resource, bool known, std::uint32_t initial) {
      if (!resource) {
        return;
      }
      auto *found = find(resource);
      if (!found) {
        found = insert(resource);
      }
      found->value = {};
      if (known) {
        found->value.version = ++serial;
        found->value.value = initial;
      }
      found->used = ++serial;
    }

    bool tracked(std::uint64_t resource) const {
      return find(resource) != nullptr;
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

    // The texture's proven state for a copy executed on queue.
    state query(std::uint64_t resource, std::uint64_t queue, bool creation_trusted) {
      auto *found = find(resource);
      if (!found || !found->value || (found->value.queue ? found->value.queue != queue : !creation_trusted)) {
        return {};
      }
      found->used = ++serial;
      return found->value;
    }

    // The version a proof still has (0: unproven or untracked).
    std::uint64_t version(std::uint64_t resource) const {
      const auto *found = find(resource);
      return found ? found->value.version : 0;
    }

    unsigned size() const {
      return count;
    }

  private:
    struct entry {
      std::uint64_t resource {}, used {};
      state value;
    };

    std::array<entry, Capacity> entries {};  // Sorted by resource.
    unsigned count {};
    std::uint64_t serial {};

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

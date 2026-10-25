// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

// Which D3D12 depth-stencil views are read-only, by CPU descriptor handle,
// for the Generic path's boundaries of a bound view (its unbind and a
// clear-like draw; the contract matrix's D11c). A read-only view
// (D3D12_DSV_FLAG_READ_ONLY_DEPTH or _STENCIL) leaves its texture in
// DEPTH_READ with read bits the API does not name, so no state is declared
// there. ReShade keeps no DSV flags with a view: it reports them only as the
// usage of the view's creation (depth_stencil_read when any flag is set), so
// depth_addon.cpp records each depth-stencil view it is told of here.
//
// Lock-free: views are created on any thread while lists record on others.
// Bounded: a read-only view takes a slot for its handle, which keeps it; a
// writable view created over that handle marks the slot writable, and a
// handle never read-only takes no slot. A read-only view that finds no slot
// within its probes sets overflow: from then on a handle the table does not
// hold may be read-only, so read_only answers true for it, which only refuses
// copies. A slot whose flag is not stored yet reads as read-only too. A
// descriptor copied into a DSV heap (CopyDescriptors) is not reported by
// ReShade; its handle keeps what was last created there.
namespace sunshine_read_only_dsv {
  template<unsigned Capacity, unsigned Probes = 16>
  class table {
    static_assert(Capacity && Probes && Probes <= Capacity);

  public:
    // A depth-stencil view was created at handle view.
    void created(std::uint64_t view, bool read_only) {
      if (!view) {
        return;
      }
      for (unsigned n = 0, i = start(view); n != Probes; ++n, i = (i + 1) % Capacity) {
        auto key = keys[i].load(std::memory_order_acquire);
        if (!key) {
          // Slots are never emptied, so the handle is in no later slot.
          if (!read_only) {
            return;
          }
          if (!keys[i].compare_exchange_strong(key, view, std::memory_order_acq_rel) && key != view) {
            continue;
          }
          key = view;
        }
        if (key == view) {
          flags[i].store(read_only ? read_only_flag : writable_flag, std::memory_order_release);
          return;
        }
      }
      if (read_only) {
        overflow.store(true, std::memory_order_release);
      }
    }

    // Whether the view at handle view may be read-only.
    bool read_only(std::uint64_t view) const {
      for (unsigned n = 0, i = start(view); n != Probes; ++n, i = (i + 1) % Capacity) {
        const auto key = keys[i].load(std::memory_order_acquire);
        if (key == view) {
          return flags[i].load(std::memory_order_acquire) != writable_flag;
        }
        if (!key) {
          break;
        }
      }
      return overflow.load(std::memory_order_acquire);
    }

    bool overflowed() const {
      return overflow.load(std::memory_order_acquire);
    }

  private:
    static constexpr std::uint8_t writable_flag = 1, read_only_flag = 2;

    static unsigned start(std::uint64_t view) {
      // Descriptor handles are addresses a descriptor size apart.
      return unsigned(((view >> 3) * 0x9E3779B97F4A7C15ull) >> 32) % Capacity;
    }

    std::array<std::atomic<std::uint64_t>, Capacity> keys {};
    std::array<std::atomic<std::uint8_t>, Capacity> flags {};
    std::atomic<bool> overflow {};
  };
}  // namespace sunshine_read_only_dsv

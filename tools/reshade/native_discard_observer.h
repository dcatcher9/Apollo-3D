// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <Windows.h>
#include <cstdint>

namespace sunshine_streamline::native_discard {
  using invalidate_callback = void (*)(std::uint64_t command, std::uint64_t resource);
  struct counters {
    std::uint64_t calls{}, observed{}, unreadable{}, dropped{}, targets{}, installed{}, rejected{};
  };
  // Owner serializes install_pending with its other hook management. Vtable
  // storage, original code and this module remain pinned after installation.
  // Callback (0,0) requests a new content epoch when coverage changes.
  void initialize(invalidate_callback callback);
  void set_active(bool active);
  void shutdown();
  // Call only with a live native COM object from the public SDK's get_native().
  // Requires exact GraphicsCommandList QI and uses its returned pointer. Discovery
  // retains a module-owned slot; installation later patches only that slot, never
  // globally shared forwarding code. Temporary/heap vtable storage is rejected.
  void observe_command(std::uint64_t native_command);
  bool install_pending();
  counters counts();
  unsigned discard_vtable_slot();
#ifdef SUNSHINE_NATIVE_DISCARD_TEST
  // Fixture only; caller guarantees quiescence. Restores only slots still owned
  // by this observer and preserves any third-party replacement.
  void reset_after_hooks_removed();
#endif
}

// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The S3 frame clocks (docs/reshade-sbs.md, UI decision framework, S3
// snapshot ticket; the label semantics are game3d_ui_ticket.h's). Per ReShade
// device, two 256-byte default-heap buffers hold one 4-byte word each at
// offset 0, monotonic, 0 meaning never written:
// - C_P, the present clock: advance() writes the next Present label on the
//   presenting queue at every foreground Present, before the UI layer's
//   Present observation and the Game 3D render;
// - C_T, the token clock: the Streamline capture owner writes the low 32 bits
//   of a Backbuffer tag's token generation into it in the game's list, after
//   that tag's snapshot copy (native_token_clock).
// stamp() records, in the same list right after a snapshot copy, two 4-byte
// copies of C_P and C_T into the snapshot's 16-byte stamp entry. No barrier
// touches a game resource: on D3D12 the clocks and stamp entries are buffers
// created in COMMON that implicit promotion and decay carry between copy
// destination and copy source; D3D11 has no states. The D3D12 present clock
// is written by a copy from an 8-entry CPU-mapped upload ring indexed by
// label (so no update_buffer_region is recorded), on the queue's immediate
// list, which ReShade flushes before the Present; D3D11 writes it through the
// immediate context, which is the queue. Nothing here decides anything.
#include "game3d_frame_clock_native.h"

#include <windows.h>

#include <cstdint>
#include <reshade_api.hpp>

namespace sunshine_game3d::frame_clock {
  namespace api = reshade::api;

  inline constexpr std::uint64_t clock_bytes = 256, ring_entries = 8, ring_entry_bytes = 16;

  // A foreground Present on this queue: the device's present label advances
  // by one and is written into C_P on the queue (creating the clocks on
  // first use). Called for the foreground swapchain on every Present,
  // including Presents without a Game 3D render.
  void advance(api::command_queue *queue, api::swapchain *swapchain);

  // The device's current present label: the last advance()'s, 0 before the
  // first.
  std::uint32_t label(api::device *device);

  // The queue of the device's last advance() (its presenting queue), or null.
  api::command_queue *presenting_queue(api::device *device);

  // Records C_P to destination + offset and C_T to destination + offset + 4
  // (4 bytes each) in this list, right after the snapshot copy it stamps.
  // False, recording nothing, before the device's clocks exist: the entry then
  // keeps 0, unstamped.
  bool stamp(api::command_list *commands, api::resource destination, std::uint64_t offset);

  // The native clock buffers for a capture owner that records natively:
  // native_present_clock and native_token_clock (game3d_frame_clock_native.h,
  // ReShade-free).

  // The ui_ticket::interposer bits of the frame-generation modules loaded in
  // this process (ui_ticket::interposer_modules). GetModuleHandleW changes no
  // reference count; callers probe at most once a second.
  std::uint32_t loaded_interposers();

  // destroy_device releases a device's clocks.
  void register_events();
  void unregister_events();
}  // namespace sunshine_game3d::frame_clock

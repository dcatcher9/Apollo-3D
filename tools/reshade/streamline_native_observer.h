// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <Windows.h>
#include <cstdint>

struct D3D12_RESOURCE_BARRIER;
struct ID3D12Resource;

namespace sunshine_streamline::native_observer {
  struct command_identity {
    std::uint64_t native_command{}, recording_cookie{};
  };
  // One texture barrier of an enhanced Barrier call: the texture, its layout
  // afterwards (D3D12_BARRIER_LAYOUT) and whether the barrier's subresource
  // range includes subresource 0 (mip 0, array slice 0, plane 0).
  struct enhanced_texture {
    ID3D12Resource *resource{};
    std::uint32_t layout_after{};
    bool first_subresource{};
  };
  struct callbacks {
    // All callbacks are synchronous. Neither arrays nor native objects are
    // retained. Barrier fields are the exact native type/flags/state/subresource.
    void (*barriers)(std::uint64_t command, std::uint64_t cookie, unsigned count, const D3D12_RESOURCE_BARRIER *values){};
    // Identities are frozen BEFORE ExecuteCommandLists; notification occurs only
    // AFTER its native original returns. No GPU completion is implied.
    void (*submitted)(std::uint64_t queue, unsigned count, const command_identity *commands){};
    // Hook coverage changed, an input could not be snapshotted, or discovery failed.
    // Owner must invalidate evidence without freeing in-flight GPU resources.
    void (*invalidated)(){};
    // Texture barriers of an enhanced (CommandList7) Barrier call, after the
    // native call; global and buffer barriers change no texture state and are
    // not reported.
    void (*enhanced_textures)(std::uint64_t command, std::uint64_t cookie, unsigned count, const enhanced_texture *textures){};
    // Called at native operation entry, before forwarding, only for an object
    // without a recording cookie. The owner may authenticate a new recording;
    // post-call evidence freezes its returned identity across any later Reset.
    void (*associate_recording)(std::uint64_t command, std::uint64_t *cookie){};
  };
  struct counters {
    std::uint64_t calls{}, observed{}, unreadable{}, dropped{}, targets{}, installed{}, rejected{}, nested{}, suppressed{};
    // Overflow now means snapshot allocation/size failure, never exceeding
    // the inline fast path. Valid large batches are observed in full.
    std::uint64_t barrier_overflow{}, submission_overflow{}, discovery_contention{};
  };

  // Owner serializes install_pending calls. Initialize only while owner
  // callbacks are quiescent; callback
  // code must remain loaded. Shutdown is nonblocking and leaves retained hooks
  // and callback pointers intact: an already entered callback may still finish.
  void initialize(callbacks value);
  void set_active(bool value);
  void shutdown();
  // Live IUnknown objects only. Discovery first requires QueryInterface for the
  // exact graphics-command-list/queue interface and uses the returned pointer;
  // unrelated interfaces are rejected before reading any D3D12 method offsets.
  // Discovery retains module-owned vtable slots and their original code. It
  // never installs hooks from a lifecycle callback. Each method has at most
  // eight native vtable slots. Deferred installation replaces only the exact
  // authenticated slot; shared code used by other interfaces is untouched. Optional CL4/CL7
  // methods are discovered only through successful QueryInterface; readiness
  // requires coverage of every interface supported by the live object. Only
  // ResourceBarrier and (CL7) enhanced Barrier are hooked on command lists:
  // their lifecycle comes from ReShade's events.
  void observe_command(std::uint64_t command);
  void observe_queue(std::uint64_t queue);
  bool install_pending();
  // The list's resource states are observed: its barrier (and, if supported,
  // enhanced Barrier) hooks are installed.
  bool states_ready(std::uint64_t command);
  bool queue_ready(std::uint64_t queue);
  counters counts();

  // Live ID3D12Object-compatible COM objects only. The private-data GUID is local
  // to this addon. Proxies that forward Get/SetPrivateData share the native cookie.
  // Zero means unassociated; owner supplies a unique nonzero recording cookie.
  bool set_recording_cookie(std::uint64_t command, std::uint64_t cookie);
  std::uint64_t get_recording_cookie(std::uint64_t command);
  // Exact absence only. Malformed private data and driver errors are not a new
  // recording and must never permit lazy identity creation.
  bool recording_cookie_absent(std::uint64_t command);

  // Callbacks automatically suppress nested observer notifications. Use this
  // scope around capture-issued native work outside a callback as well.
  class suppression_scope {
  public:
    suppression_scope();
    ~suppression_scope();
    suppression_scope(const suppression_scope &) = delete;
    suppression_scope &operator=(const suppression_scope &) = delete;
  };

#ifdef SUNSHINE_STREAMLINE_NATIVE_OBSERVER_TEST
  // Fixture only: callbacks are quiescent. Restore only still-owned vtable
  // slots, preserving any third-party replacements, then release the banks.
  void reset_after_hooks_removed();
#endif
}

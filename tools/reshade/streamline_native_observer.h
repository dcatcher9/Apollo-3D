// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "queue_order.h"

#include <Windows.h>
#include <cstdint>

struct D3D12_RESOURCE_BARRIER;
struct ID3D12CommandQueue;
struct ID3D12Fence;
struct ID3D12Resource;

namespace sunshine_streamline::native_observer {
  struct command_identity {
    std::uint64_t native_command{}, recording_cookie{};
  };
  // One texture barrier of an enhanced Barrier call: the texture, its layout
  // afterwards (D3D12_BARRIER_LAYOUT) and whether the barrier's subresource
  // range includes subresource 0 (mip 0, array slice 0, plane 0), with its
  // access afterwards and both sync scopes (raw D3D12_BARRIER_ACCESS/SYNC).
  // A SyncAfter of SPLIT begins a split barrier; a SyncBefore of SPLIT ends it.
  struct enhanced_texture {
    ID3D12Resource *resource{};
    std::uint32_t layout_after{};
    bool first_subresource{};
    std::uint32_t access_after{}, sync_before{}, sync_after{};
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
  // A second owner's view of submissions (the offscreen UI layer's cross-queue
  // fence, game3d_ui_layer.h): called like callbacks::submitted, after the
  // native ExecuteCommandLists returns and under the same suppression, while
  // the capture owner keeps the observer active. Its failures are its own: a
  // listener never invalidates the capture owner's evidence. Null removes it;
  // an already entered call may still finish.
  using submission_listener = void (*)(std::uint64_t queue, unsigned count, const command_identity *commands) noexcept;
  void set_submission_listener(submission_listener value);
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

  // Queue order (Step 4; docs/reshade-sbs.md, Queue order). The observer
  // also hooks ID3D12CommandQueue::Signal and ::Wait (discovered with
  // ExecuteCommandLists by observe_queue) and ID3D12Fence::Signal (discovered
  // at a fence's first sight in one of those), and gives each hooked
  // Execute/Signal/Wait call on a tracked queue a ticket. Signals are logged
  // before they are forwarded, waits after they returned success; the
  // queue_order.h model judges, the host never waits for a queue and never
  // adds a GPU Wait. Its slots are a separate invalidation family: their
  // coverage changes reset the order model only, never the capture owner's
  // evidence.
  using position = sunshine_queue_order::position;
  // QueueFenceOrder (default on). Off: no order slot is discovered,
  // ExecuteCommandLists takes no ticket and every verdict is disabled.
  void set_queue_order(bool enabled);
  bool queue_order_enabled();
  // The position of the latest positioned ExecuteCommandLists this thread
  // made: the game's inside its submitted callback, the add-on's own right
  // after its suppressed call returned. Empty without a ticket.
  position last_execute_position();
  // Whether every call submitted on reader (a live native queue the caller
  // holds a reference to) from now on is ordered after producer.
  sunshine_queue_order::verdict queue_ordered(std::uint64_t reader, position producer);
  // A read of tag (a capture id) that relied on an ordered verdict
  // (late_conflict).
  void queue_order_rely(const sunshine_queue_order::verdict &value, std::uint64_t tag);
  // O1: whether a read of tag relied on a fence that a late conflict
  // poisoned since; its presentation copy is then not held any further.
  bool queue_order_conflicted(std::uint64_t tag);
  // A released device (its native ID3D12Device pointer): its records go.
  void queue_order_forget_device(std::uint64_t device);
  struct queue_order_status {
    bool enabled{};
    unsigned signal_hooks{}, wait_hooks{}, fence_signal_hooks{}, pending{};
    sunshine_queue_order::counts model{};
    std::uint64_t ordered{}, resets{}, cpu_unhooked{};
    // A call that may not wait (an Execute, or one the observer does not
    // log) found the order lock held where its queue needed a new record:
    // it took no ticket.
    std::uint64_t contention{};
    std::uint64_t refused[static_cast<unsigned>(sunshine_queue_order::refusal::count)]{};
  };
  queue_order_status queue_order_counts();
  // The add-on's own Signal of a fence of its own that no game queue waits
  // on (design finding 8): suppressed, so the order hooks never log it as
  // the game's and it never waits on the order lock. Every such add-on
  // Signal goes through here.
  HRESULT signal_own(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 value);
  // The native methods the observer hooks, as pinned by the hook-set test.
  struct hooked_method {
    const char *interface_name, *method;
    unsigned slot;
  };
  const hooked_method *hooked_methods(unsigned &count);

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
  // Vtable slot addresses read so far (a checked read each).
  std::uint64_t slot_reads_for_testing();
#endif
}

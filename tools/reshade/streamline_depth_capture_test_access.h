// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// streamline_depth_capture.cpp's internal types, constants and the templates
// its white-box regressions (test_streamline_depth_capture.cpp) instantiate,
// and in a test build the declarations of the other internals they reach.
// Only the capture file, those regressions and the seam characterization
// (test_game3d_seam_characterization.cpp) include this header.
//
// The shipping build keeps all of them internal, in anonymous namespaces. A
// SUNSHINE_STREAMLINE_PROBE_TEST build gives the same definitions external
// linkage in an inline namespace, so that the regressions can run in their own
// translation unit; lookup inside the capture file is unchanged.
#include "streamline_depth_capture.h"
#include "capture_state_policy.h"
#include "freshness.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "streamline_native_observer.h"
#include <d3d12.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
#define SUNSHINE_CAPTURE_INTERNAL inline namespace test_access
#define SUNSHINE_CAPTURE_STATIC
#else
#define SUNSHINE_CAPTURE_INTERNAL namespace
#define SUNSHINE_CAPTURE_STATIC static
#endif

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    template<class T> struct com_ptr {
      T *p{};
      ~com_ptr() { reset(); }
      com_ptr() = default;
      com_ptr(const com_ptr &) = delete;
      com_ptr &operator=(const com_ptr &) = delete;
      T *operator->() const { return p; }
      void reset() { if (auto *owned = std::exchange(p, nullptr)) owned->Release(); }
      T **put() { reset(); return &p; }
    };
    // One lazily allocated pool serves eight rotating source members plus
    // several recordings in flight and metadata nominations. No second pool
    // or per-source synchronization implementation belongs to Generic depth.
    constexpr unsigned slot_limit = 32, source_limit = 64;
    // Queue places (register_queue): one for each DIRECT queue ReShade reports
    // and for each queue that submits a copy awaiting its producer. Games
    // create several (Assassin's Creed Shadows, 10-25: five DIRECT queues
    // before its presenting queue, then DLSS-G's COMPUTE queue), so this is a
    // safety bound far above any game's live queues, not a budget: a full
    // table reclaims an idle place (free_place), and a place no queue can
    // take is reported once (capacity_report).
    constexpr unsigned queue_limit = 64;
    // The queues that read one snapshot (slot::retirements): the add-on's own
    // reads on its presenting queue, not the game's queues.
    constexpr unsigned consumer_queue_limit = 4;
    constexpr unsigned pixel_slot_limit = slot_limit - 4; // Source metadata must survive a full GPU pool.
    constexpr unsigned diagnostic_slot_limit = 32;
    constexpr std::uint64_t diagnostic_byte_limit = 256ull * 1024 * 1024;
    constexpr D3D12_RESOURCE_STATES sampled_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    using copy_state::write_states;
    constexpr GUID source_guid = sunshine_native_identity::resource_guid;
    constexpr GUID queue_guid{0x091eab97, 0xfcc2, 0x47fb, {0x97, 0x3b, 0x02, 0x7c, 0x69, 0xb0, 0xd5, 0x26}};
    constexpr unsigned provider_count = 2;
    using provider_kind = sunshine_scene_depth::provider_kind;
    struct copy_region {
      std::uint32_t width{}, height{};
      sunshine_scene_depth::extent area;
    };
    // enhanced: the latest barrier was an enhanced Barrier into layout with
    // access; value is then only the layout's legacy equivalent (diagnostic,
    // UINT32_MAX without one).
    // pending_split: a split barrier's state is unknown until its end or a
    // later plain barrier defines it. blocked is sticky until Reset and no
    // later barrier clears it.
    // order: the recording's barrier order (command_state::barrier_order) of
    // the latest barrier that set this entry.
    // enhanced_named: any enhanced barrier on the recording named its
    // subresource 0, whatever followed (the enhanced-layout latch,
    // proven_state.h).
    struct source_state {
      std::uint64_t source{}; std::uint32_t value{}; bool known{}, blocked{}, enhanced{}, pending_split{};
      std::uint32_t layout{}, access{};
      std::uint32_t order{};
      bool enhanced_named{};
    };
    // A texture named at a barrier order by an aliasing barrier (before or
    // after) or by a depth clear of its subresource 0. Neither changes a state
    // entry, so captures at an API call ignore them; only the per-resource
    // proven state (proven_state.h) reads them: an aliasing barrier revokes
    // it, a depth clear proves what the D3D12 clear contract requires.
    struct resource_mark {
      std::uint64_t source{}; std::uint32_t order{};
    };
    struct command_state {
      std::uint64_t cookie{};
      std::uint64_t observation_generation{};
      std::uint64_t native_barrier_calls{}, native_transition_count{}, last_barrier_command{};
      // The number of alias/split resources in a game recording is not bounded
      // by the number of depth sources we capture. Reuse this storage on Reset.
      std::vector<source_state> states;
      std::vector<resource_mark> aliases, clears;
      // Each state change and aliasing barrier takes the next order; the
      // latest aliasing barrier with neither resource (any placed or reserved
      // resource) is at wildcard_alias (0 none).
      std::uint32_t barrier_order{}, wildcard_alias{};
      // aliases is sorted by source, then order (an executed recording,
      // which nothing appends to until its Reset).
      bool aliases_sorted{};
      bool closed{}, invalid{}, render_pass{};
      // ReShade's lifecycle (or a registered runtime list's observed
      // submissions) drives this recording.
      bool reported{};
      // A barrier arrived after Close: the list was Reset without ReShade's
      // lifecycle, so these states are not what its next execution does.
      bool missed_reset{};
      recording_loss invalidation{recording_loss::none};
      // The queue place (queue_state::cookie) that last executed this list
      // (0 never): the list's, not one recording's, so restart keeps it.
      // A copy recorded on it takes copy-out storage unless it last ran on
      // the queue that reads depth (U-C1b, record_impl).
      std::uint64_t executed_queue{};
      void restart(std::uint64_t next_cookie, std::uint64_t generation) noexcept {
        states.clear();
        aliases.clear();
        aliases_sorted = false;
        clears.clear();
        cookie = next_cookie; observation_generation = generation;
        native_barrier_calls = native_transition_count = last_barrier_command = 0;
        barrier_order = wildcard_alias = 0;
        closed = invalid = render_pass = reported = missed_reset = false;
        invalidation = recording_loss::none;
      }
    };
    using command_storage = sunshine_native_command::storage<command_state>;
    using recording_ref = std::shared_ptr<sunshine_native_command::recording_lifetime>;
    // A queue's own command lists for copy-outs (copy_out), of the queue's
    // type: created at first use, each reused once the fence value signalled
    // after its execution completed. A safety bound, not a budget: with every
    // list still in flight a copy-out is not made and that copy keeps the
    // Reset rule (counted as ring_busy).
    constexpr unsigned copy_out_lists = 8;
    struct copy_out_list {
      com_ptr<ID3D12CommandAllocator> allocator;
      com_ptr<ID3D12GraphicsCommandList> list;
      std::uint64_t fence{};
    };
    struct queue_state {
      com_ptr<ID3D12CommandQueue> queue;
      com_ptr<ID3D12Device> device;
      com_ptr<ID3D12Fence> fence;
      std::uint64_t cookie{}, next_fence{}, last_epoch{}, last_sequence{}, nominated_epoch{}, present{}, present_capture{};
      std::uint64_t admission_after{}, last_source_tick{};
      std::uint64_t device_identity{};
      // When a lookup last found this place (known_queue, queue): a reclaimed
      // idle place is the longest unused (free_place).
      std::uint64_t last_use{};
      // compute_producer: a COMPUTE queue (register_compute_producer).
      bool retiring{}, provider_established{}, compute_producer{};
      provider_kind provider{provider_kind::streamline};
      std::uint64_t source_id{};
      std::uint32_t viewport{};
      // The newest acquisition's choice (depth_choice), when chosen. Whether
      // depth is read elsewhere is the newest acquisition's (the capture's
      // reader, read_elsewhere), not this flag, which stays once set.
      depth_choice choice;
      bool chosen{};
      std::array<copy_out_list, copy_out_lists> copy_out;
    };
  }

  // Declared in streamline_depth_capture.h. Their com_ptr members are internal
  // in the shipping build, where only the capture file defines these types, so
  // GCC's warning about a header type used by several files does not apply.
#if defined(__GNUC__) && !defined(__clang__) && !defined(SUNSHINE_STREAMLINE_PROBE_TEST)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
  // device in both: the device as the resource names it (ReShade's device
  // proxy under ReShade), not owned: a reference to the proxy kept the
  // game's device from ever being destroyed (the lifecycle test's former
  // L1). When ReShade destroys the device, release_device releases the
  // objects of every source and snapshot of it, under the capture lock,
  // whoever still holds them (released, then resource and views are null),
  // so that none outlives it.
  struct source_reference {
    mutable com_ptr<ID3D12Resource> resource;
    mutable ID3D12Device *device{};
    D3D12_RESOURCE_DESC desc{};
    std::uint64_t cookie{}, device_identity{};
    mutable std::atomic<std::uint64_t> present_generation{1};
    mutable std::atomic<bool> released{};
    // When a copy of this source was last submitted on a queue other than
    // the one that now reads depth (copy_out; 0 never): its copies recorded
    // within the source age of it take copy-out storage (slot::published,
    // read_elsewhere_recently). Never before a queue first read depth, so a
    // game that copies on its presenting queue allocates none. It lapses
    // once the source's copies are submitted only where they are read (FG
    // off again, the render queue reading), however many queues copied it;
    // a source copied on both queues keeps it.
    mutable std::atomic<std::uint64_t> foreign_tick{};
  };
  // Whether a copy of value recorded at now takes copy-out storage: a copy of
  // it was submitted on a queue other than the reader within the source age,
  // the age past which no copy is read. A stamp after now (another thread's)
  // is recent.
  inline bool read_elsewhere_recently(const source_reference &value, std::uint64_t now) {
    const auto tick = value.foreign_tick.load(std::memory_order_relaxed);
    return tick && sunshine_freshness::ahead_tolerant_lt(tick, now, sunshine_freshness::source_age_ms);
  }
  struct texture_reference {
    com_ptr<ID3D12Resource> resource;
    ID3D12Device *device{};
    com_ptr<ID3D12DescriptorHeap> views;
    std::atomic<bool> released{};
    std::uint64_t shader_resource{}, device_identity{}, identity{};
    HANDLE shared_handle{};
    std::uint64_t allocation_bytes{};
    // Local auxiliary snapshots: the format of the SRV at shader_resource,
    // created on the first lease (lease_local_view); protected by mutex.
    DXGI_FORMAT view_format{DXGI_FORMAT_UNKNOWN};
    // The state the storage rests in between copies, fixed at allocation:
    // sampled_state for a depth snapshot, COMMON for auxiliary storage and
    // for a depth snapshot first recorded on a COMPUTE list (which cannot
    // name the pixel-shader state). Every copy into or out of it starts and
    // ends there.
    D3D12_RESOURCE_STATES rest{D3D12_RESOURCE_STATE_COMMON};
    ~texture_reference() { if (shared_handle) CloseHandle(shared_handle); }
  };
#if defined(__GNUC__) && !defined(__clang__) && !defined(SUNSHINE_STREAMLINE_PROBE_TEST)
#pragma GCC diagnostic pop
#endif
  SUNSHINE_CAPTURE_INTERNAL {
    struct fence_point { std::uint64_t queue{}, value{}; };
    struct slot {
      std::shared_ptr<texture_reference> texture;
      // Copy-out (docs/reshade-sbs.md, Copy-out on the producer queue): the
      // add-on's private copy of texture, of the same shape, format and view
      // (its own rest: texture_reference::rest). The submission hook copies texture into it on the producer
      // queue with the add-on's own list, right after the game's
      // ExecuteCommandLists that carried the copy into texture and before the
      // producer fence (copied_out). A later execution of the game's list
      // writes texture again, never this copy, so once the producer fence
      // completed every consumer reads it (read_storage) without the
      // recording's Reset. It lives with the slot as texture does: reused
      // with it, kept as an idle slot's cached storage (release_idle_slots)
      // and released with its device.
      std::shared_ptr<texture_reference> published;
      input metadata;
      // One auxiliary copy destination per snapshot. Keep the actual resource
      // alive through consumer Reset AND GPU completion, including legal replay.
      source_ref auxiliary_destination;
      std::uint64_t id{}, command{}, queue{}, producer_fence{}, consumer_queue{};
      // The queue-order position another queue's read must follow before
      // producer_fence completed (U-C1, readable_elsewhere): the copy-out list
      // once copied out, else the game's submission (read only once its
      // recording retired); empty without a ticket.
      native_observer::position producer_position;
      // Values are comparable only within their own private queue timeline.
      // Reset may discard recording identities, never these GPU obligations.
      std::array<fence_point, consumer_queue_limit> retirements{};
      recording_ref producer_recording;
      std::array<std::uint64_t, 4> consumers{};
      std::array<recording_ref, 4> consumer_recordings;
      // A runtime immediate list that recorded a read and has not yet been
      // observed in a submission on consumer_queue. See consume_owned.
      std::uint64_t pending_immediate{};
      bool finished{}, success{}, acquired{}, invalid{}, producer_submitted{}, retirement_unknown{};
      bool preservation_only{};
      bool diagnostic_only{}, diagnostic_released{}, diagnostic_reusable{};
      bool source_nominated{}, nomination_only{}, nomination_invalid{};
      // Recorded on a COMPUTE list (record_impl; S22, N7).
      bool compute_list{};
      // published holds the copy-out of this slot's first submission, ordered
      // before producer_fence (copy_out); cleared when the slot is reused.
      bool copied_out{};
      // published was taken only because the recording's list was not known
      // to execute on the reader's queue (U-C1b, record_impl): released by
      // the idle sweep once the slot's first submission made no copy-out
      // (release_idle_slots).
      bool predicted_storage{};
      status pixel_failure{status::unavailable};
      capture_failure failure{capture_failure::none};
      // The pre-copy state's provenance of an auxiliary snapshot (diagnostic;
      // nothing here admits, retires or reuses a slot).
      state_basis basis{state_basis::unknown};
      // An empty slot reserved by a record attempt that allocates its storage
      // without the lock (record_impl). Expires after reservation_timeout_ms;
      // publication never trusts it. reserved_bytes counts toward the
      // diagnostic budget while live.
      std::uint64_t reservation{}, reservation_tick{}, reserved_bytes{};
      bool reserved_live{}; // The reservation is for a live snapshot (its budget).
      // When a colour/mask slot's capture retired and it kept only its cached
      // live allocation (collect_diagnostics). poll() releases old idle storage.
      std::uint64_t idle_tick{};
      // Direct binding (lease_local_view): the runtime list and recording on
      // which this snapshot sits in the shader-resource state until
      // end_local_views returns it to COMMON.
      std::uint64_t lease_list{}, lease_cookie{};
      // A copy from its source's last proven state (record_preserved at the
      // end of the frame): that proof's source, version and queue, checked
      // when its recording is executed (0 none).
      std::uint64_t proven_source{}, proven_version{}, proven_queue{};
      // present_ordinal() when recorded (frame_presenting), and when its
      // producer list was first seen submitted on a presenting queue, else
      // zero (presented_by).
      std::uint64_t present{}, submitted_present{};
    };
    struct evaluation_head {
      std::uint64_t epoch{}, sequence{}, tick{}, source_id{};
      std::uint32_t viewport{};
      // Since when every finished evaluation of this view was rejected (zero
      // after an accepted one); result_sequence orders late finishes.
      std::uint64_t rejected_since{}, result_sequence{};
      // Capture tick of this view's newest recorded copy the API accepted
      // (frame_generation_scope_live).
      std::uint64_t accepted_tick{};
      // The FG scope's newest frame outcomes by observation sequence: the
      // newest accepted copy's, and the newest frame without one
      // (frame_generation_missed). The newer of the two decides the scope.
      std::uint64_t accepted_sequence{}, missed_sequence{};
      // The newest recorded copy's sequence, and whether the viewport's
      // newest frame has recorded no copy yet (frame_generation_frame).
      std::uint64_t recorded_sequence{};
      bool awaiting{};
      // present_ordinal() at the newest evaluation's call (frame_presenting).
      std::uint64_t present{};
    };
    struct evaluation_namespace {
      std::array<evaluation_head, 4> views{};
      std::uint64_t epoch{}, sequence{}, source_id{};
      evaluation_head pending_nomination;
    };
    // FG Off retired the FG role of a viewport's display lineage in epoch
    // (retire_source): the options call's observation serial, and whether
    // FG stays off (no FG On since, frame_generation_restored).
    struct fg_retirement {
      std::uint64_t epoch{}, serial{};
      std::uint32_t viewport{};
      bool active{};
    };
    // Constant-initialized Present/FG state; the sole object stays at the
    // original presents position in streamline_depth_capture.cpp.
    struct scope_state {
      std::atomic<std::uint64_t> presents{};
      // A stamp is the frame the Present being decided presents when no other
      // Present came since its call (frame_presenting's first rule, the only
      // one without frame generation). Such a view is active and such a
      // snapshot current however long the frame took after its call (a hitch
      // between an SR evaluation and its Present), as they are the depth of
      // the frame that Present shows: the source age bounds holds over later
      // Presents, never a frame's own depth at its own Present (10-27; until
      // then that Present was mono, inactive_views, after a hitch of the source
      // age). A Present of any swapchain, a generated one too, ends it, so a
      // generated Present between an evaluation and its frame's Present leaves
      // the wall-clock rule alone.
      // Require mutex. The newest present_ordinal() an evaluation began with
      // (begin_evaluation_locked), the newest one a frame-generation input's
      // evaluation began with or its copy was stamped with, and the Presents
      // DLSS-G adds per real frame that the selection being decided grants
      // (select_provider, selection_policy::generated_presents).
      std::uint64_t newest_evaluation_present{}, frame_generation_present{};
      std::uint32_t presenting_generated{};
      // FG moves Present to DLSS-G's queue at its first Present, after the
      // render queue's copies of that frame were recorded and submitted while
      // that queue still read depth, so for the source age after FG turns on
      // the reader's own copies are copied out too (U-C1b, reader_may_move).
      // Two windows, each closed once Present moved to another queue since it
      // opened (reader_moves): from FG options turning on
      // (frame_generation_enabled; 0 never), and from the first FG activity
      // of a run (an FG frame's begin, DLSS-G's NGX evaluation:
      // frame_generation_activity) after FG options turned on or after
      // fg_silent_ms without any. The second covers a game that turns FG on
      // away from gameplay (a menu, a loading screen, saved settings at boot),
      // whose first FG Present comes long after the options call, and DLSS-G
      // seen only through NGX, with no options call at all.
      std::atomic<std::uint64_t> frame_generation_on_tick{}, frame_generation_on_moves{};
      std::atomic<std::uint64_t> frame_generation_run_tick{}, frame_generation_run_moves{}, frame_generation_activity_tick{};
      // Protected by mutex: the viewports whose FG role FG Off retired, per
      // epoch (retire_source; the oldest gives way).
      std::array<fg_retirement, 4> fg_retirements{};
    };
    static_assert(std::is_trivially_destructible_v<scope_state>);
    struct queue_progress {
      std::uint64_t device_identity{}, completed{};
      bool queried{};
      bool valid() const { return queried && completed != UINT64_MAX; }
    };
    struct capture_pick {
      slot *value{};
      const slot *latest{};
      const evaluation_head *pending_nomination{};
      // The namespace's newest nomination (its epoch, sequence and source)
      // as its view names it, with that view's viewport, tick and present:
      // set only when that view is the namespace's one active view
      // (select_latest_capture). Policy picks carry none.
      const evaluation_head *head{};
      // The pick's provider namespace and its newest nomination's view,
      // whatever its views' activity: the source an acquisition names when
      // no capture or nomination in progress does (classify_acquisition), so
      // the display never holds another viewport's or feature generation's
      // copy under it. Policy picks carry none.
      provider_kind provider{provider_kind::streamline};
      const evaluation_head *named{};
      status result{status::unavailable};
      selection_reason selection{selection_reason::not_attempted};
      std::uint64_t consumed_completed_capture{};
      queue_progress progress;
      // The fresh head of the one active view produced no readable pixels:
      // it failed, was refused (with or without a capture slot), or is a
      // nomination without readable pixels. That is a missing frame of its
      // display lineage, as a pending one is (select_capture): it moves no
      // admission watermark (interrupts_admission).
      bool missing_frame{};
    };
    template<std::size_t N> bool identify_submissions(std::array<slot, N> &captures, std::uint32_t count,
        const native_observer::command_identity *values, std::array<bool, N> &producers,
        std::array<bool, N> &consumers) {
      bool used = false;
      for (unsigned n = 0; n != count; ++n) {
        const auto cookie = values[n].recording_cookie;
        for (unsigned i = 0; i != captures.size(); ++i) {
          auto &value = captures[i];
          if (!value.id) continue;
          if (value.pending_immediate && value.pending_immediate == values[n].native_command) { consumers[i] = true; used = true; }
          // Zero is an unobserved recording, not a producer whose recording was
          // retired by Reset. It must not change capture state or create a fence.
          if (!cookie) continue;
          if (value.command == cookie) {
            if (value.producer_submitted) invalidate(value, capture_failure::replay); // No replayed old frame.
            producers[i] = true; used = true;
          }
          for (const auto consumer : value.consumers) if (consumer && consumer == cookie) { consumers[i] = true; used = true; }
        }
      }
      return used;
    }
    template<class Complete> bool gpu_retired(const slot &value, Complete complete) {
      if (value.retirement_unknown || !value.producer_fence || !complete(fence_point{value.queue, value.producer_fence})) return false;
      for (const auto &point : value.retirements) if (point.queue && (!point.value || !complete(point))) return false;
      return true;
    }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    // Defined in streamline_depth_capture.cpp. Default arguments stay with the
    // definitions there; the regressions pass every argument.
    extern std::atomic<std::uint64_t> serial;
    extern std::atomic<status> reason;
    extern std::array<std::atomic<status>, provider_count> attempt_status;
    extern std::atomic<bool> requested;
    extern std::mutex mutex;
    extern scope_state scope;
    std::uint64_t present_ordinal();
    bool frame_presenting(std::uint64_t ordinal);
    std::uint64_t presented_by(const slot &value);
    bool view_presenting(const evaluation_head &view, provider_kind provider);
    unsigned provider_index(provider_kind value);
    bool valid_projection(const sunshine_scene_depth::projection_basis &value);
    std::uint64_t source_cookie(ID3D12Object *value);
    std::uint64_t retain_source_cookie(ID3D12Object *value);
    bool diagnostic_description(const D3D12_RESOURCE_DESC &desc, std::uint32_t declared);
    status source_region(const D3D12_RESOURCE_DESC &desc,
      const sunshine_scene_depth::resource_description &resource, copy_region &out);
    status capture_region(const D3D12_RESOURCE_DESC &desc,
      const sunshine_scene_depth::resource_description &resource, copy_region &out);
    extern std::atomic<std::uint64_t> command_generation;
    extern std::array<slot, slot_limit> slots;
    extern std::array<slot, diagnostic_slot_limit> diagnostic_slots;
    extern std::array<std::unique_ptr<queue_state>, queue_limit> queues;
    extern std::array<evaluation_namespace, provider_count> evaluations;
    // A copy was ever recorded on a COMPUTE list (record_impl); liveness
    // queries look for readable compute-list copies only then.
    extern bool compute_copies_recorded, compute_unplaced;
    // Queue places (register_queue): a free or idle one, and whether an idle
    // one was ever reclaimed.
    std::unique_ptr<queue_state> *free_place();
    extern bool places_reclaimed;
    queue_state *known_queue(std::uint64_t native);
    // Requires mutex. The retired slots' storage moves to discard (released
    // after the lock, on a pool thread).
    void collect_retired(std::vector<std::shared_ptr<texture_reference>> &discard);
    bool matching_evaluation(const evaluation_head &head, const input &value);
    void end_nomination(const input &value);
    void invalidate(slot &value, capture_failure why);
    command_storage::handle command(std::uint64_t native, std::uint64_t cookie, bool create);
    void invalidate(command_state &owner, recording_loss why);
    void associate_recording(std::uint64_t native, std::uint64_t *out_cookie);
    extern bool fail_state_growth;
    source_state *state(command_state &owner, std::uint64_t identity, bool create);
    void invalidate_all();
    void barriers(std::uint64_t native, std::uint64_t cookie, std::uint32_t count, const D3D12_RESOURCE_BARRIER *values);
    bool aliased_after(const command_state &owner, std::uint64_t source, std::uint32_t order);
    void sort_aliases(command_state &owner);
    void retire_recording(slot &value, std::uint64_t old);
    void retire_dead_recordings(slot &value);
    void reset(std::uint64_t native, std::uint64_t old, HRESULT result, bool reported);
    void close(std::uint64_t native, std::uint64_t cookie, HRESULT result);
    void enhanced_textures(std::uint64_t native, std::uint64_t cookie, unsigned count,
      const native_observer::enhanced_texture *textures);
    copy_state::observation observation_of(const source_state *observed);
    state_basis copy_basis(const source_state *observed, const copy_state::decision &chosen);
    void render_pass(std::uint64_t native, std::uint64_t cookie, bool inside);
    void producer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled);
    void accept_compute_copy(const slot &value);
    bool submitted_success(const slot &value, std::uint64_t native);
    status diagnostic_admission(const slot &value, const queue_progress &progress);
    status submission_status(const slot &value, std::uint64_t native,
      std::uint64_t consumer_device, const queue_progress &progress);
    status nomination_status(const slot &value, std::uint64_t consumer_device);
    void bind_pixel_consumer(slot &value, std::uint64_t consumer, status pixels);
    bool pending_frame(const capture_pick &chosen, const queue_state &owner);
    acquisition_decision classify_acquisition(const capture_pick &chosen, const queue_state &owner,
      selection_policy policy);
    depth_choice describe_choice(const capture_pick &chosen, const acquisition_decision &decision,
      const queue_state &owner, selection_policy policy, std::uint64_t present, std::uint64_t tick, bool pixel_ready);
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
      const input &head, const slot &completed, bool compare_storage);
    slot *lineage_snapshot(const queue_state &owner, std::uint64_t native, std::uint64_t present, const input &head,
      std::uint64_t now);
    input head_identity(const evaluation_head &head, provider_kind provider);
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
      std::uint64_t present, provider_kind provider, std::uint64_t now);
    bool interrupts_admission(const queue_state &owner, const capture_pick &chosen);
    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
      std::uint64_t present, std::uint64_t now, selection_policy policy);
    void establish_provider(queue_state &owner, const slot &value, std::uint64_t present);
    void consumer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled);
    bool uses_queue(const slot &value, std::uint64_t native);
    bool needs_submission_fence(const std::array<bool, slot_limit> &producers,
      const std::array<bool, slot_limit> &consumers);
    bool reclaimable(slot &value);
    bool reserve_diagnostic_bytes(std::array<slot, diagnostic_slot_limit> &pool,
      const slot *replacement, std::uint64_t bytes, bool live, std::uint64_t own,
      std::vector<std::shared_ptr<texture_reference>> *discard);
    std::uint64_t record_nomination(const input &value, std::uint64_t normalized_source,
      std::uint64_t cookie, const recording_ref &recording);
    bool reusable_preserved(const slot &entry, const input &value, std::uint64_t recording);
    void describe_packet(const slot &value, std::uint64_t consumer, packet &out);
    void expire_association(queue_state &owner, std::uint64_t now);
    // Nonzero: frame_generation_scope_live's fixed clock (the seam characterization).
    extern std::atomic<std::uint64_t> scope_clock;
    void record_frame_generation_copy(const input &value);
    consumer_status preserved_admission(const slot &value, const preservation_ticket &ticket,
      std::uint64_t consumer, std::uint64_t consumer_device, std::uint64_t recording);
    // The white-box fixtures' resets (test_streamline_depth_capture_fixture.h),
    // one per owner of the state it clears, so that regrouping an owner's
    // globals changes its reset here instead of every regression.
    // Snapshot pool: no capture slots.
    inline void reset_pool() { slots = {}; }
    // API evaluation bookkeeping: no evaluations.
    inline void reset_evaluations() { evaluations = {}; }
    // Presentation scope: no FG role retirement, no evaluation or frame
    // generation Present, and no generated Presents.
    inline void reset_scope() {
      scope.fg_retirements = {};
      scope.newest_evaluation_present = scope.frame_generation_present = 0;
      scope.presenting_generated = 0;
    }
    // Record admission: every provider's last attempt unavailable.
    inline void reset_attempts() {
      for (auto &value : attempt_status) {
        value = status::unavailable;
      }
    }
    // Queue places: no queue owners.
    inline void reset_queues() {
      for (auto &entry : queues) {
        entry.reset();
      }
    }
#endif
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  bool begin_nomination(const input &value, std::uint64_t superseded_source_id);
  bool source_lifetime_current(const input &value, std::uint64_t current_generation);
  // The texture's proven state for a copy that queue executes; false (and an
  // empty out) when it is unproven there. The diagnostic-texture regressions
  // read it; no shipping code does.
  bool last_proven_state(std::uint64_t resource, std::uint64_t queue, proven_state &out);
#endif
}

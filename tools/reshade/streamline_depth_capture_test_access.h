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
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "streamline_native_observer.h"
#include <d3d12.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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
      // The newest acquisition's choice (depth_choice), when chosen.
      depth_choice choice;
      bool chosen{};
    };
  }

  // Declared in streamline_depth_capture.h. Their com_ptr members are internal
  // in the shipping build, where only the capture file defines these types, so
  // GCC's warning about a header type used by several files does not apply.
#if defined(__GNUC__) && !defined(__clang__) && !defined(SUNSHINE_STREAMLINE_PROBE_TEST)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
  struct source_reference {
    com_ptr<ID3D12Resource> resource;
    com_ptr<ID3D12Device> device;
    D3D12_RESOURCE_DESC desc{};
    std::uint64_t cookie{}, device_identity{};
    mutable std::atomic<std::uint64_t> present_generation{1};
  };
  struct texture_reference {
    com_ptr<ID3D12Resource> resource;
    com_ptr<ID3D12Device> device;
    com_ptr<ID3D12DescriptorHeap> views;
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
      input metadata;
      // One auxiliary copy destination per snapshot. Keep the actual resource
      // alive through consumer Reset AND GPU completion, including legal replay.
      source_ref auxiliary_destination;
      std::uint64_t id{}, command{}, queue{}, producer_fence{}, consumer_queue{};
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
      // The newest recorded copy's sequence, the viewport's newest frame
      // (its key), and whether that frame has recorded no copy yet
      // (frame_generation_frame).
      std::uint64_t recorded_sequence{}, frame{};
      bool awaiting{};
    };
    struct evaluation_namespace {
      std::array<evaluation_head, 4> views{};
      std::uint64_t epoch{}, sequence{}, source_id{};
      evaluation_head pending_nomination;
    };
    struct queue_progress {
      std::uint64_t device_identity{}, completed{};
      bool queried{};
      bool valid() const { return queried && completed != UINT64_MAX; }
    };
    struct capture_pick {
      slot *value{};
      const slot *latest{};
      const evaluation_head *pending_nomination{};
      status result{status::unavailable};
      selection_reason selection{selection_reason::not_attempted};
      std::uint64_t consumed_completed_capture{};
      queue_progress progress;
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
    void collect_retired();
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
      const slot &latest, const slot &completed);
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
      std::uint64_t present, provider_kind provider, std::uint64_t now);
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
#endif
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  bool begin_nomination(const input &value, std::uint64_t superseded_source_id);
  bool source_lifetime_current(const input &value, std::uint64_t current_generation);
#endif
}

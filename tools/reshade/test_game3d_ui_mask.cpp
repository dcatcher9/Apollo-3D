// SPDX-License-Identifier: GPL-3.0-only
// Exercises the live UI-alpha owner with deterministic native snapshot states.
// Real D3D resource/command/fence retirement is tested by the shared capture suite.
#include "game3d_ui_mask.h"

#include <cstdio>
#include <atomic>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>

namespace capture = sunshine_streamline::depth_capture;
namespace mask = sunshine_game3d::ui_mask;
namespace scene = sunshine_scene_depth;

namespace {
  struct native_snapshot {
    capture::input input;
    capture::texture_state_policy state_policy{};
    capture::diagnostic_texture texture;
    capture::status status{capture::status::recorded};
    bool finished{}, success{}, released{};
  };
  std::map<std::uint64_t, native_snapshot> snapshots;
  std::uint64_t next_ticket{};
  std::uint64_t present_generation = 4;
  std::uint64_t local_consumer_queue{};
  unsigned records{}, releases{};
  bool fail_record{};
  std::uint32_t record_format = 24;
  std::function<void()> during_record;
  std::function<void()> during_acquire;
  constexpr std::uint64_t device = 0x100, runtime = 0x200;

  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  mask::request request() { return {runtime, device, 7, 3, 11, 3840, 2160, true}; }
  void reset() {
    mask::invalidate_all();
    snapshots.clear(); next_ticket = 0; records = releases = 0;
    fail_record = false; record_format = 24; present_generation = 4; local_consumer_queue = 0; during_record = {}; during_acquire = {};
    mask::set_request(request());
  }
  void gate_diagnostics_preserve_scope_mismatch_without_capture() {
    reset();
    auto wanted = request();
    std::uint32_t matches{};
    require(mask::interested(wanted.epoch, wanted.revision, wanted.viewport, &matches) && matches == 1,
      "Gate diagnostics lost the unique request match");
    mask::capture_gate_observation gate{mask::capture_gate::admitted, wanted.epoch, wanted.revision, 9, 1000,
      wanted.viewport, mask::source_mask(mask::source_kind::hudless), matches};
    mask::observe_gate(gate);
    mask::diagnostic_snapshot observed;
    require(mask::query_diagnostic(runtime, observed) && observed.hook_gate.sequence == 9 &&
        !observed.latest_boundary.source.sequence && !records,
      "Hook gate metadata fabricated a capture boundary or GPU work");
    const auto generation = observed.request_generation;
    ++wanted.revision; mask::set_request(wanted);
    require(!mask::interested(gate.epoch, gate.revision, gate.viewport, &matches) && !matches,
      "Gate diagnostics bypassed a mismatching observation scope");
    gate.state = mask::capture_gate::no_matching_request; gate.matching_requests = matches; ++gate.sequence;
    mask::observe_gate(gate);
    auto older = gate; --older.sequence; older.state = mask::capture_gate::admitted;
    mask::observe_gate(older);
    require(mask::query_diagnostic(runtime, observed) && observed.request_generation > generation &&
        observed.hook_gate.state == mask::capture_gate::no_matching_request &&
        observed.hook_gate.revision != observed.wanted.revision && !observed.latest_boundary.source.sequence,
      "Request replacement hid its rejected gate or an older callback rewound diagnostics");
    auto other = wanted; ++other.runtime; mask::set_request(other);
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport, &matches) && matches == 2,
      "Gate diagnostics hid an ambiguous consumer request");
  }
  capture::input input(std::uint64_t sequence, std::uint64_t tick = 1000) {
    capture::input value;
    value.epoch = 7; value.observation_revision = 3; value.sequence = sequence; value.tick = tick; value.viewport = 11;
    value.resource = {0x400 + sequence, 3840, 2160, {0, 0, 3840, 2160}};
    value.source = capture::source_ref(reinterpret_cast<const capture::source_reference *>(0x400 + sequence), [](auto *) {});
    value.proof = scene::state_proof::declared; value.native_state = 8;
    value.valid_until = scene::lifetime::at_call; value.force_snapshot = true;
    value.frame_generation_input = true; value.source_frame_token = 123;
    value.source_frame_generation = 6; value.source_frame_numeric = 456; value.source_frame_has_numeric = true;
    value.source_present_generation = present_generation;
    return value;
  }
  mask::attempt begin(const capture::input &value, mask::source_kind kind = mask::source_kind::backbuffer) {
    return mask::begin({value, 0x300, 1, kind}, value);
  }
  mask::attempt capture_one(std::uint64_t sequence, std::uint64_t tick = 1000) {
    auto value = begin(input(sequence, tick)); mask::finish(value, true); return value;
  }
  void complete(const mask::attempt &value) { snapshots.at(value.ticket.id).status = capture::status::ready; }
  mask::selection selected(std::uint64_t sequence, std::uint64_t now = 1001) {
    mask::selection result;
    require(mask::acquire(runtime, result, now), "completed alpha was unavailable");
    require(result.origin.source.sequence == sequence, "wrong real-alpha sequence selected");
    return result;
  }
  void absent(std::uint64_t now = 1001) {
    mask::selection result;
    require(!mask::acquire(runtime, result, now) && !result.ticket && !result.texture.texture,
      "unavailable alpha exposed pixels");
  }

  void completed_and_pending() {
    reset(); auto first = capture_one(1); absent();
    snapshots.at(first.ticket.id).status = capture::status::submitted; absent();
    complete(first);
    auto chosen = selected(1);
    require(chosen.origin.command == 0x300 && chosen.origin.tag_scope == 1 &&
      chosen.origin.source.resource.native == 0x401 && chosen.origin.source.source_frame_token == 123 &&
      chosen.origin.source.source_frame_numeric == 456 && chosen.origin.source.source_frame_has_numeric,
      "consumed alpha lost its exact tag provenance");
    mask::set_request(request()); selected(1);
    auto second = capture_one(2, 1010); selected(1, 1011);
    complete(second); selected(2, 1011);
    require(snapshots.at(first.ticket.id).released && !snapshots.at(second.ticket.id).released,
      "newest completion did not retire only the older completed lease");
    selected(2, 1012); // Reuse does not consume or refresh source age.
    selected(2, 1259); absent(1260);
    require(snapshots.at(second.ticket.id).released, "expired alpha lease was retained");
  }

  void bounded_pending() {
    reset(); auto first = capture_one(1); complete(first); selected(1);
    auto second = capture_one(2, 1010);
    auto skipped = capture_one(3, 1030);
    require(!skipped.ticket && skipped.runtime == runtime && records == 2 && releases == 0,
      "full owner evicted a pending snapshot or allocated unbounded work");
    selected(1, 1031);
    complete(second); selected(2, 1031);
    auto third = capture_one(4, 1040);
    require(third.ticket && records == 3 && !snapshots.at(second.ticket.id).released,
      "completed replacement did not free bounded capacity without evicting pending work");
  }

  void revoke_scopes() {
    for (unsigned change = 0; change != 8; ++change) {
      reset(); auto first = capture_one(1); complete(first); selected(1);
      auto next = request();
      if (change == 0) next.epoch++;
      if (change == 1) next.revision++;
      if (change == 2) next.device_identity++;
      if (change == 3) next.viewport++;
      if (change == 4) next.width /= 2;
      if (change == 5) next.height /= 2;
      if (change == 6) next.enabled = false;
      if (change == 7) { mask::invalidate(runtime); }
      else mask::set_request(next);
      absent(); require(snapshots.at(first.ticket.id).released, "scope change retained stale alpha");
      mask::finish(first, true); absent();
    }
    reset(); auto pending = begin(input(1));
    mask::invalidate_scope(7, 11); mask::finish(pending, true);
    absent(); require(!mask::interested(7, 3, 11), "FG Off left capture requested");
    reset();
    during_record = [] { mask::invalidate_all(); };
    auto canceled = begin(input(1)); mask::finish(canceled, true);
    require(records == 1 && releases == 1 && !snapshots.begin()->second.success,
      "record returning across invalidation revived its native lease");
    absent();
  }

  void reservation_race() {
    reset(); auto first = capture_one(1); complete(first); selected(1);
    std::atomic<bool> recorded{}, publish{};
    during_record = [&] {
      recorded.store(true, std::memory_order_release);
      while (!publish.load(std::memory_order_acquire)) std::this_thread::yield();
    };
    mask::attempt second;
    std::thread producer([&] { second = capture_one(2, 1010); });
    while (!recorded.load(std::memory_order_acquire)) std::this_thread::yield();
    // collect snapshots the still-reserved second slot, then queries first.
    // Complete record publication before collect commits its earlier view.
    during_acquire = [&] { publish.store(true, std::memory_order_release); producer.join(); };
    selected(1, 1011);
    complete(second); selected(2, 1011);
    require(!snapshots.at(second.ticket.id).released, "collector retired a concurrently published reservation");
  }

  void invalid_tags_and_sdk_failure() {
    for (unsigned invalid = 0; invalid != 6; ++invalid) {
      reset(); auto first = capture_one(1); complete(first); selected(1);
      auto bad = input(2, 1010);
      if (invalid == 0) { bad.resource.native = 0; bad.source = {}; }
      if (invalid == 1) bad.resource.width = 1920;
      if (invalid == 2) bad.resource.area.left = 1;
      if (invalid == 3) bad.resource.area.width = 1920;
      if (invalid == 4) bad.valid_until = scene::lifetime::unsupported;
      auto next = begin(bad); mask::finish(next, invalid != 5);
      absent(1011); require(snapshots.at(first.ticket.id).released, "null/shape/lifetime/SDK rejection reused old alpha");
    }
    reset(); auto first = capture_one(1); complete(first); selected(1);
    fail_record = true; auto failed_capture = capture_one(2, 1010);
    require(!failed_capture.ticket, "failed native record returned a snapshot");
    selected(1, 1011); // Temporary native capacity/state failure may reuse bounded last real pixels.
    mask::finish(failed_capture, false); absent(1011); // SDK failure cannot.
    reset(); auto older = begin(input(1)); auto newer = capture_one(2, 1010); complete(newer);
    mask::finish(older, false); selected(2, 1011);
    require(snapshots.at(older.ticket.id).released, "failed older attempt was not revoked");
  }

  void admission_and_isolation() {
    for (unsigned invalid = 0; invalid != 5; ++invalid) {
      reset(); auto first = capture_one(1); complete(first);
      auto &texture = snapshots.at(first.ticket.id).texture;
      if (invalid == 0) texture.device_identity++;
      if (invalid == 1) texture.width /= 2;
      if (invalid == 2) texture.area.width /= 2;
      if (invalid == 3) texture.area.left++;
      if (invalid == 4) texture.format = 88; // BGRX is not alpha.
      absent(); require(snapshots.at(first.ticket.id).released, "foreign/partial/non-alpha snapshot was accepted");
    }
    for (auto format : {2u, 10u, 24u, 28u, 29u, 87u, 91u}) {
      reset(); auto first = capture_one(1); complete(first);
      snapshots.at(first.ticket.id).texture.format = format; selected(1);
    }
    reset(); auto foreign = input(1); foreign.viewport++;
    require(!begin(foreign).runtime && !records, "other viewport captured into requested UI scope");
    foreign = input(1); foreign.epoch++;
    require(!begin(foreign).runtime && !records, "other epoch captured into requested UI scope");
    foreign = input(1); foreign.observation_revision++;
    require(!begin(foreign).runtime && !records, "other observation revision captured into requested UI scope");
    auto other = request(); other.runtime++;
    mask::set_request(other);
    require(!mask::interested(7, 3, 11) && !begin(input(1)).runtime && !records,
      "ambiguous consumer scope guessed one runtime");
    mask::invalidate(other.runtime); require(mask::interested(7, 3, 11), "removing other consumer did not restore scope");
    auto first = capture_one(2, 1100); complete(first); selected(2, 1101);
    require(!begin(input(1, 1000)).runtime && records == 1, "late older tag displaced the newest source");
    selected(2, 1101);
    absent(999); // A backwards tick cannot extend a source's lifetime.
    selected(2, 1101); // It also cannot retire a future snapshot.
  }

  void diagnostic_rejections_and_no_attempt() {
    reset();
    mask::diagnostic_snapshot report;
    require(mask::query_diagnostic(runtime, report) && report.wanted.runtime == runtime &&
      report.wanted.width == 3840 && !report.latest_boundary.source.sequence &&
      !report.record_attempted && !report.record_completed && !report.sdk_result_known,
      "requested scope without a boundary was reported as a capture attempt");
    fail_record = true;
    auto rejected = begin(input(1));
    require(!rejected.ticket && mask::query_diagnostic(runtime, report) && report.record_attempted && report.record_completed &&
      !report.sdk_result_known && report.latest_boundary.source.sequence == 1 &&
      report.latest_boundary.source.native_state == 8 && report.record.native_state == 8 &&
      report.record.observed_state == 128 && report.record.observed && !report.record.blocked &&
      report.record.recording_cookie == 0xc01 && report.record.command == 0x300 &&
      report.record.resource == 0x401 && report.record.result == capture::status::conflicting_state &&
      report.record.stage == capture::record_stage::conflicting_state,
      "native rejection lost its exact requested/observed state or recording identity");
    mask::finish(rejected, true);
    require(mask::query_diagnostic(runtime, report) && report.sdk_result_known && report.sdk_successful &&
      report.record.result == capture::status::conflicting_state,
      "successful SDK call rewrote the native capture rejection");
    auto null_tag = input(2, 1010); null_tag.resource.native = 0; null_tag.source = {};
    auto skipped = begin(null_tag); mask::finish(skipped, true);
    require(mask::query_diagnostic(runtime, report) && report.latest_boundary.source.sequence == 2 &&
      !report.record_attempted && !report.record_completed && report.record.stage == capture::record_stage::not_attempted &&
      report.sdk_result_known && report.sdk_successful && records == 1,
      "invalid tag retained an older rejection or fabricated a native attempt");

    reset(); capture_one(1); capture_one(2, 1010); capture_one(3, 1020);
    capture_one(4, 1030);
    require(mask::query_diagnostic(runtime, report) && report.latest_boundary.source.sequence == 4 &&
      !report.record_attempted && report.sdk_result_known && records == 2,
      "owner capacity skip was mislabeled as a native capture rejection");
  }

  void diagnostic_newest_scope_wins() {
    reset();
    mask::attempt newer;
    during_record = [&] { newer = capture_one(2, 1010); };
    auto older = begin(input(1));
    mask::diagnostic_snapshot report;
    require(mask::query_diagnostic(runtime, report) && report.latest_boundary.source.sequence == 2 &&
      report.record.recording_cookie == 0xc02 && report.record.resource == 0x402 &&
      report.sdk_result_known && report.sdk_successful,
      "older native record completion replaced a newer diagnostic");
    mask::finish(older, false);
    require(mask::query_diagnostic(runtime, report) && report.latest_boundary.source.sequence == 2 &&
      report.sdk_result_known && report.sdk_successful,
      "older failed SDK callback replaced the newest diagnostic");
    mask::set_request(request());
    require(mask::query_diagnostic(runtime, report) && report.latest_boundary.source.sequence == 2,
      "identical request discarded its newest diagnostic");
    auto changed = request(); changed.revision++;
    mask::set_request(changed); mask::finish(newer, false);
    require(mask::query_diagnostic(runtime, report) && report.wanted.revision == changed.revision &&
      !report.latest_boundary.source.sequence && !report.record_attempted && !report.sdk_result_known,
      "scope change retained or resurrected an older diagnostic");

    reset();
    during_record = [] { mask::invalidate(runtime); mask::set_request(request()); };
    auto revoked = begin(input(1)); mask::finish(revoked, true);
    require(mask::query_diagnostic(runtime, report) && !report.latest_boundary.source.sequence &&
      !report.record_attempted && !report.sdk_result_known,
      "record return across invalidation revived diagnostics in a replacement owner");
    mask::invalidate(runtime);
    require(!mask::query_diagnostic(runtime, report) && !report.wanted.runtime &&
      !report.latest_boundary.source.sequence,
      "invalidated request exposed stale diagnostic metadata");

    reset();
    auto source = input(1);
    std::weak_ptr<const capture::source_reference> source_lifetime = source.source;
    auto attempt = begin(source); mask::finish(attempt, true); source.source.reset();
    require(!source_lifetime.expired() && mask::query_diagnostic(runtime, report),
      "live candidate did not retain source present-generation identity");
    mask::invalidate(runtime);
    require(source_lifetime.expired() && report.latest_boundary.source.sequence,
      "plain diagnostic metadata retained a revoked application source COM lease");
  }

  void diagnostic_in_flight_is_not_a_rejection() {
    reset();
    bool checked_pending = false;
    during_record = [&] {
      mask::diagnostic_snapshot pending;
      require(mask::query_diagnostic(runtime, pending) && pending.record_attempted &&
        !pending.record_completed && !pending.sdk_result_known &&
        pending.latest_boundary.source.sequence == 1,
        "in-flight native call was not distinguished from a completed rejection");
      checked_pending = true;
    };
    auto recorded = begin(input(1));
    mask::diagnostic_snapshot report;
    require(checked_pending && mask::query_diagnostic(runtime, report) && report.record_attempted &&
      report.record_completed && report.record.result == capture::status::recorded && !report.sdk_result_known &&
      report.record.copy_state_known && report.record.used_observed_state && report.record.copy_state == 128 &&
      report.record.native_state == 8,
      "returned native call did not complete its diagnostic independently of the SDK call");
    mask::finish(recorded, true);
    require(mask::query_diagnostic(runtime, report) && report.record_completed &&
      report.sdk_result_known && report.sdk_successful,
      "SDK completion lost the completed native diagnostic");
  }

  void dedicated_kind_format_and_provenance() {
    for (auto format : {41u, 54u, 56u, 61u}) {
      reset(); record_format = format;
      auto alpha = begin(input(1), mask::source_kind::alpha); mask::finish(alpha, true); complete(alpha);
      const auto value = selected(1);
      require(value.origin.kind == mask::source_kind::alpha && value.texture.format == format,
        "Single-channel UIAlpha lost its explicit tag semantics or exact format");
    }
    for (auto format : {2u, 10u, 24u, 28u, 29u, 87u, 91u}) {
      reset(); record_format = format;
      auto color = begin(input(1), mask::source_kind::color_and_alpha); mask::finish(color, true); complete(color);
      require(selected(1).origin.kind == mask::source_kind::color_and_alpha,
        "Dedicated UI color lost its alpha-channel semantics");
    }
    for (auto kind : {mask::source_kind::alpha, mask::source_kind::color_and_alpha}) {
      reset(); record_format = kind == mask::source_kind::alpha ? 24 : 61;
      auto wrong = begin(input(1), kind); mask::finish(wrong, true); complete(wrong); absent();
      require(snapshots.at(wrong.ticket.id).released, "UI kind guessed a mask channel from an incompatible format");
    }
    reset();
    require(!begin(input(1), static_cast<mask::source_kind>(54)).runtime && !records,
      "A non-UI Streamline resource entered the live mask owner");
  }

  void dedicated_preference_and_same_batch_fallback() {
    reset(); auto back = capture_one(1); complete(back);
    auto color = begin(input(2, 1010), mask::source_kind::color_and_alpha); mask::finish(color, true);
    selected(1, 1011); // Ready lower-priority pixels survive a preferred pending copy.
    auto second_back = capture_one(3, 1020);
    require(second_back.ticket && records == 3, "Preferred candidate blocked independent backbuffer acquisition");
    complete(color); selected(2, 1021);
    record_format = 61;
    auto alpha = begin(input(4, 1030), mask::source_kind::alpha); mask::finish(alpha, true);
    record_format = 24;
    auto second_color = begin(input(4, 1030), mask::source_kind::color_and_alpha); mask::finish(second_color, true);
    require(second_color.ticket && records == 5, "Same-batch preferred candidate starved an independently validated UI source");
    complete(second_color);
    complete(alpha); selected(4, 1031);
    mask::selection independent;
    require(mask::acquire_kind(runtime, mask::source_kind::color_and_alpha, independent, 1031) &&
        independent.ticket.id == second_color.ticket.id &&
        mask::acquire_kind(runtime, mask::source_kind::backbuffer, independent, 1031) && independent.ticket.id == back.ticket.id,
      "Default priority selection retired a valid lower candidate");
    auto null_back = input(5, 1040); null_back.resource.native = 0; null_back.source = {};
    auto unrelated = begin(null_back); mask::finish(unrelated, true); selected(4, 1041);
    auto null_alpha = input(6, 1050); null_alpha.resource.native = 0; null_alpha.source = {};
    auto revoked = begin(null_alpha, mask::source_kind::alpha);
    record_format = 24;
    auto fallback = begin(input(6, 1050), mask::source_kind::color_and_alpha);
    mask::finish(revoked, false); mask::finish(fallback, true); complete(fallback);
    require(selected(6, 1051).origin.kind == mask::source_kind::color_and_alpha,
      "Null preferred tag or its delayed completion erased same-batch valid fallback");
    auto low = capture_one(7, 1300); complete(low);
    require(selected(7, 1301).origin.kind == mask::source_kind::backbuffer,
      "Expired dedicated input permanently suppressed the ordinary real backbuffer");

    reset(); fail_record = true;
    auto rejected = begin(input(1), mask::source_kind::alpha);
    fail_record = false;
    auto usable = begin(input(1), mask::source_kind::color_and_alpha);
    mask::finish(rejected, true); mask::finish(usable, true); complete(usable);
    selected(1);
    require(records == 2 && !rejected.ticket && usable.ticket,
      "Native rejection of preferred UI blocked a valid same-call alternative");
  }

  void hudless_independent_capacity_and_pairing_provenance() {
    reset(); record_format = 61;
    auto alpha = begin(input(1), mask::source_kind::alpha); mask::finish(alpha, true); complete(alpha);
    auto pending_alpha = begin(input(2, 1010), mask::source_kind::alpha); mask::finish(pending_alpha, true);
    require(!begin(input(3, 1020), mask::source_kind::alpha).ticket, "Alpha exceeded its independent two-slot limit");
    record_format = 24;
    auto source = input(3, 1020); source.frame_generation_input = false;
    auto hudless = begin(source, mask::source_kind::hudless); mask::finish(hudless, true); complete(hudless);
    mask::selection chosen;
    require(mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1021) && chosen.ticket.id == hudless.ticket.id &&
        chosen.origin.kind == mask::source_kind::hudless && !chosen.origin.source.frame_generation_input &&
        chosen.origin.source.source_frame_token == source.source_frame_token && chosen.origin.tag_scope == 1 &&
        chosen.origin.source_present_generation == 4 && chosen.current_source_present_generation == 4,
      "Full alpha queue starved FG-off HUD-less capture or erased its exact pairing provenance");
    ++present_generation;
    require(mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1022) &&
        chosen.origin.source_present_generation == 4 && chosen.current_source_present_generation == 5,
      "Current Present mutation rewrote frozen HUD-less tag provenance");
    selected(1, 1022); // Availability ranking cannot consume or suppress HUD-less.
    auto next = begin(input(4, 1030), mask::source_kind::hudless); mask::finish(next, true);
    auto full = begin(input(5, 1040), mask::source_kind::hudless); mask::finish(full, true);
    require(next.ticket && !full.ticket && records == 4 &&
        mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1041) && chosen.ticket.id == hudless.ticket.id,
      "HUD-less latest-ready plus pending capacity was unbounded or lost its ready snapshot");
    mask::finish(next, false);
    require(!mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1042) &&
        mask::acquire_kind(runtime, mask::source_kind::alpha, chosen, 1042),
      "Failed HUD-less source retained earlier comparison pixels or revoked unrelated alpha");
  }

  // S3: the newest HUD-less and Backbuffer pair of one game frame, by token
  // generation (numbered or not), read-only.
  void token_batch_pairs_the_newest_common_frame() {
    reset();
    const auto tagged = [](std::uint64_t sequence, std::uint64_t generation, std::uint64_t tick) {
      auto value = input(sequence, tick);
      value.source_frame_generation = generation; value.source_frame_has_numeric = false; value.source_frame_numeric = 0;
      return value;
    };
    const auto capture = [](const capture::input &value, mask::source_kind kind, bool ready) {
      auto attempt = begin(value, kind); mask::finish(attempt, true);
      if (ready) complete(attempt);
      return attempt;
    };
    // Frame F's HUD-less and Backbuffer are recorded before F's HUD-less
    // completes (one game list), so F-1 stays retained beside it.
    const auto hudless_previous = capture(tagged(1, 20, 1000), mask::source_kind::hudless, true);
    const auto backbuffer_previous = capture(tagged(1, 20, 1000), mask::source_kind::backbuffer, true);
    const auto hudless_current = capture(tagged(2, 21, 1010), mask::source_kind::hudless, false);
    const auto backbuffer_current = capture(tagged(2, 21, 1010), mask::source_kind::backbuffer, false);
    complete(hudless_current);
    mask::selection hudless, backbuffer;
    // HUD-less F and F-1 ready, Backbuffer F-1 only: the pair is F-1.
    require(mask::acquire_batch(runtime, 1011, 0x900, hudless, backbuffer) && hudless.token_generation == 20 &&
        backbuffer.token_generation == 20 && hudless.ticket.id == hudless_previous.ticket.id &&
        backbuffer.ticket.id == backbuffer_previous.ticket.id && hudless.origin.kind == mask::source_kind::hudless &&
        backbuffer.origin.kind == mask::source_kind::backbuffer,
      "HUD-less F and F-1 with only Backbuffer F-1 ready did not pair F-1");
    require(!snapshots.at(hudless_previous.ticket.id).released && !snapshots.at(hudless_current.ticket.id).released,
      "The token batch retired a snapshot");
    complete(backbuffer_current);
    require(mask::acquire_batch(runtime, 1012, 0x900, hudless, backbuffer) && hudless.token_generation == 21 &&
        hudless.ticket.id == hudless_current.ticket.id && backbuffer.ticket.id == backbuffer_current.ticket.id,
      "The newest common token was not chosen once its Backbuffer was ready");
    require(!mask::acquire_batch(runtime, 1012, 0, hudless, backbuffer) && !hudless.ticket && !backbuffer.ticket,
      "A token batch without a consumer queue exposed pixels");
    // Different generations, or none, never pair.
    reset();
    capture(tagged(1, 30, 1000), mask::source_kind::hudless, true);
    capture(tagged(1, 31, 1000), mask::source_kind::backbuffer, true);
    require(!mask::acquire_batch(runtime, 1001, 0x900, hudless, backbuffer), "Different token generations paired");
    reset();
    capture(tagged(1, 0, 1000), mask::source_kind::hudless, true);
    capture(tagged(1, 0, 1000), mask::source_kind::backbuffer, true);
    require(!mask::acquire_batch(runtime, 1001, 0x900, hudless, backbuffer), "Snapshots without a token generation paired");
    // A scope change (epoch or viewport) revokes both: nothing pairs across it.
    reset();
    capture(tagged(1, 40, 1000), mask::source_kind::hudless, true);
    capture(tagged(1, 40, 1000), mask::source_kind::backbuffer, true);
    require(mask::acquire_batch(runtime, 1001, 0x900, hudless, backbuffer), "One frame's tags did not pair");
    auto moved = request(); ++moved.viewport; mask::set_request(moved);
    require(!mask::acquire_batch(runtime, 1002, 0x900, hudless, backbuffer), "A token batch crossed a viewport change");
    mask::set_request(request());
    // The token clock write is asked for after Backbuffer snapshots only.
    reset();
    require(begin(tagged(1, 50, 1000), mask::source_kind::backbuffer).ticket &&
        snapshots.rbegin()->second.input.token_clock, "A Backbuffer snapshot did not ask for the token clock");
    require(begin(tagged(1, 50, 1000), mask::source_kind::hudless).ticket &&
        !snapshots.rbegin()->second.input.token_clock, "A HUD-less snapshot asked for the token clock");
  }

  // S3: every begin() that makes no attempt names its stage.
  void begin_refusals_name_their_stage() {
    reset();
    using stage = sunshine_game3d::ui_ticket::begin_stage;
    const auto counts = [] {
      mask::diagnostic_snapshot value;
      require(mask::query_diagnostic(runtime, value), "No diagnostic for the request");
      return value;
    };
    const auto refused = [&](const capture::input &value, mask::source_kind kind, stage expected, bool named) {
      const auto before = counts();
      const auto attempt = begin(value, kind);
      const auto after = counts();
      require(!attempt.reservation && after.begin_refusals[std::size_t(expected)] == before.begin_refusals[std::size_t(expected)] + 1 &&
          (!named || after.begin_refusal == expected), "A begin() without an attempt did not name its stage");
    };
    auto other_epoch = input(1); other_epoch.epoch = 99;
    refused(other_epoch, mask::source_kind::backbuffer, stage::no_request, false);
    auto bad_shape = input(1); bad_shape.resource.width = 1920;
    refused(bad_shape, mask::source_kind::backbuffer, stage::shape, true);
    auto unsupported = input(2); unsupported.valid_until = scene::lifetime::unsupported;
    refused(unsupported, mask::source_kind::backbuffer, stage::unsupported_lifetime, true);
    auto first = begin(input(3), mask::source_kind::backbuffer); mask::finish(first, true);
    auto second = begin(input(4), mask::source_kind::backbuffer); mask::finish(second, true);
    refused(input(5), mask::source_kind::backbuffer, stage::no_reservation, true);
    refused(input(5), mask::source_kind::backbuffer, stage::not_newer, true);
    auto filtered = request(); filtered.allowed_kinds = mask::source_mask(mask::source_kind::hudless);
    mask::set_request(filtered);
    refused(input(6), mask::source_kind::alpha, stage::kind_filtered, true);
    auto ambiguous = request(); ambiguous.runtime = runtime + 1; mask::set_request(request()); mask::set_request(ambiguous);
    refused(input(7), mask::source_kind::backbuffer, stage::ambiguous_request, false);
    mask::invalidate(runtime + 1);
    // An attempt clears the request's latest refusal.
    reset();
    require(begin(input(1), mask::source_kind::backbuffer).reservation && counts().begin_refusal == stage::none,
      "An attempt kept a refusal stage");
  }

  void local_queue_acquisition_preserves_current_candidate() {
    reset(); auto source = input(1); source.frame_generation_input = false;
    const auto captured = begin(source, mask::source_kind::hudless); mask::finish(captured, true);
    snapshots.at(captured.ticket.id).status = capture::status::submitted;
    mask::selection chosen;
    require(!mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1001),
      "Strict diagnostic acquisition exposed an unfinished GPU snapshot");
    require(mask::acquire_kind(runtime, mask::source_kind::hudless, chosen, 1001, 0x900) &&
        local_consumer_queue == 0x900 && chosen.ticket.id == captured.ticket.id &&
        !snapshots.at(captured.ticket.id).released,
      "UI owner failed to use queue-ordered local acquisition for its current candidate");
  }

  void explicit_source_filter_prevents_higher_priority_starvation() {
    reset();
    auto wanted = request(); wanted.allowed_kinds = mask::source_mask(mask::source_kind::backbuffer);
    mask::set_request(wanted);
    require(mask::interested(7, 3, 11), "Explicit backbuffer filter disabled capture");
    require(!begin(input(99), mask::source_kind::alpha).runtime &&
      !begin(input(99), mask::source_kind::color_and_alpha).runtime && records == 0,
      "Unselected higher-priority mask allocated or reserved capture work");
    // An ignored high sequence must not prevent the selected source's next call.
    auto back = capture_one(1); complete(back);
    require(selected(1).origin.kind == mask::source_kind::backbuffer,
      "Selected backbuffer was starved by an unselected dedicated source");
    require(!begin(input(2, 1010), mask::source_kind::alpha).runtime && records == 1,
      "Higher-priority tag displaced an explicitly selected backbuffer");
    mask::set_request(wanted); selected(1);

    for (const auto kind : {mask::source_kind::alpha, mask::source_kind::color_and_alpha}) {
      reset(); wanted = request(); wanted.allowed_kinds = mask::source_mask(kind); mask::set_request(wanted);
      require(!begin(input(1)).runtime && records == 0, "Dedicated source filter admitted backbuffer fallback");
      record_format = kind == mask::source_kind::alpha ? 61 : 28;
      auto captured = begin(input(1), kind); mask::finish(captured, true); complete(captured);
      require(selected(1).origin.kind == kind, "Exact dedicated source filter did not admit its own input");
    }
    reset(); wanted = request(); wanted.allowed_kinds = 0; mask::set_request(wanted);
    require(!mask::interested(7, 3, 11) && !begin(input(1)).runtime && records == 0,
      "Empty source filter admitted capture work");
    wanted.allowed_kinds = 0x80000000u; mask::set_request(wanted);
    require(!mask::interested(7, 3, 11), "Unsupported source-mask bits enabled capture");
  }

  void source_filter_change_revokes_ready_pending_and_inflight_attempts() {
    reset();
    auto ready = capture_one(1); complete(ready); selected(1);
    auto pending = begin(input(2, 1010));
    auto wanted = request(); wanted.allowed_kinds = mask::source_mask(mask::source_kind::backbuffer);
    mask::set_request(wanted);
    require(snapshots.at(ready.ticket.id).released && snapshots.at(pending.ticket.id).released,
      "Source filter change retained completed or pending captures");
    mask::finish(pending, true); complete(pending); absent(1011);
    require(!snapshots.at(pending.ticket.id).success, "Stale completion after filter change revived old pixels");
    auto fresh = capture_one(3, 1020); complete(fresh); selected(3, 1021);
    mask::finish(ready, false); selected(3, 1021);

    reset(); record_format = 61;
    during_record = [] {
      auto changed = request(); changed.allowed_kinds = mask::source_mask(mask::source_kind::backbuffer);
      mask::set_request(changed);
    };
    auto stale = begin(input(1), mask::source_kind::alpha); mask::finish(stale, true);
    require(records == 1 && releases == 1 && !snapshots.begin()->second.success,
      "Native recording across a filter change published its old reservation");
    absent(); record_format = 28;
    auto back = capture_one(2, 1010); complete(back); selected(2, 1011);
    require(records == 2, "Canceled higher-priority work starved selected backbuffer after filter change");
  }

  void declared_lifetimes_preserve_state_policy() {
    for (const auto kind : {mask::source_kind::backbuffer, mask::source_kind::color_and_alpha, mask::source_kind::alpha, mask::source_kind::hudless}) {
      for (const auto lifetime : {scene::lifetime::at_call, scene::lifetime::until_present, scene::lifetime::until_evaluation}) {
        reset(); record_format = kind == mask::source_kind::alpha ? 61 : 28;
        auto source = input(1); source.valid_until = lifetime; source.native_state = 128;
        auto captured = begin(source, kind);
        require(bool(captured.ticket), "Supported declared UI lifetime did not reach native capture");
        const auto &recorded = snapshots.at(captured.ticket.id);
        require(recorded.input.valid_until == lifetime && recorded.input.native_state == source.native_state &&
            recorded.input.proof == source.proof && recorded.input.force_snapshot &&
            recorded.state_policy == (lifetime == scene::lifetime::at_call ?
              capture::texture_state_policy::prefer_observed_recording : capture::texture_state_policy::source_contract),
          "Live UI snapshot rewrote lifetime/state provenance or selected the wrong state policy");
        mask::finish(captured, true); complete(captured);
        const auto chosen = selected(1);
        require(chosen.origin.kind == kind && chosen.origin.source.valid_until == lifetime,
          "Selected UI snapshot lost its original declared lifetime");
        source.sequence = 2; source.valid_until = scene::lifetime::unsupported;
        const auto invalid = begin(source, kind); mask::finish(invalid, true);
        require(!invalid.ticket && records == 1, "Unsupported UI lifetime reached native capture");
        absent();
      }
    }
  }
  void hudless_pairs_with_the_real_frame_under_frame_generation() {
    using present = mask::hudless_present;
    constexpr auto kind = [](std::uint64_t tagged, std::uint64_t current, bool fg, std::uint32_t generated) {
      return mask::pair_hudless_present(tagged, current, fg, generated).kind;
    };
    constexpr auto ago = [](std::uint64_t tagged, std::uint64_t current, bool fg, std::uint32_t generated) {
      return mask::pair_hudless_present(tagged, current, fg, generated).presents_ago;
    };
    // Without FG the next present is the tag's own frame.
    static_assert(kind(10, 11, false, 0) == present::real_frame && ago(10, 11, false, 0) == 0);
    static_assert(kind(10, 10, false, 0) == present::unpaired);
    // A capture that completes after its frame pairs with that frame's retained
    // color, at most max_late_presents ago.
    static_assert(kind(10, 12, false, 0) == present::earlier_real_frame && ago(10, 12, false, 0) == 1);
    static_assert(kind(10, 13, false, 0) == present::earlier_real_frame && ago(10, 13, false, 0) == 2);
    static_assert(kind(10, 14, false, 0) == present::unpaired);
    // Hogwarts Legacy, SL 2.6.10 DLSS-G: the generated frame is presented
    // first, then the real frame that matches HUDLessColor two presents on.
    static_assert(kind(3725, 3726, true, 1) == present::generated_frame);
    static_assert(kind(3725, 3727, true, 1) == present::real_frame);
    static_assert(kind(3725, 3728, true, 1) == present::earlier_real_frame && ago(3725, 3728, true, 1) == 1);
    // Enabled FG with an unreported count means one generated frame.
    static_assert(kind(5, 7, true, 0) == present::real_frame);
    // Multi-frame generation follows the reported count without a multiplier
    // constant (T1's tag bound keeps a wrong count fail-safe).
    static_assert(kind(5, 8, true, 3) == present::generated_frame);
    static_assert(kind(5, 9, true, 3) == present::real_frame);
    static_assert(kind(5, 9, true, 7) == present::generated_frame);
    static_assert(kind(5, 13, true, 7) == present::real_frame);
    static_assert(kind(5, 14, true, 7) == present::earlier_real_frame && ago(5, 14, true, 7) == 1);
    // 6x: five generated Presents, then the real frame, then late pairs.
    static_assert(kind(5, 6, true, 5) == present::generated_frame && kind(5, 10, true, 5) == present::generated_frame);
    static_assert(kind(5, 11, true, 5) == present::real_frame);
    static_assert(kind(5, 12, true, 5) == present::earlier_real_frame && ago(5, 12, true, 5) == 1);
    static_assert(kind(5, 13, true, 5) == present::earlier_real_frame && ago(5, 13, true, 5) == 2);
    static_assert(kind(5, 14, true, 5) == present::unpaired);
    // FG status only moves the real frame when it is known enabled.
    static_assert(kind(5, 6, false, 1) == present::real_frame && kind(5, 7, false, 1) == present::earlier_real_frame);
    // Unknown, sentinel and reversed generations never pair.
    static_assert(kind(0, 1, false, 0) == present::unpaired);
    static_assert(kind(UINT64_MAX, 0, true, 1) == present::unpaired);
    static_assert(kind(9, 3, true, 1) == present::unpaired);
    // Hogwarts tags HUDLessColor and Backbuffer in one batch; their counters
    // started at different values but advanced by the same Presents.
    static_assert(mask::same_tag_interval(619, 621, 1, 3));
    static_assert(mask::same_tag_interval(10, 10, 4, 4));
    static_assert(!mask::same_tag_interval(619, 621, 1, 4));
    static_assert(!mask::same_tag_interval(0, 2, 1, 3) && !mask::same_tag_interval(5, 4, 1, 0));
    static_assert(!mask::same_tag_interval(UINT64_MAX, 2, 1, 3));
  }
}

namespace sunshine_streamline::depth_capture {
  std::uint64_t source_present_generation(const source_ref &source) { return source ? present_generation : 0; }
  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      texture_state_policy state_policy) {
    require(value.force_snapshot &&
      state_policy == (value.valid_until == scene::lifetime::at_call ?
        texture_state_policy::prefer_observed_recording : texture_state_policy::source_contract),
      "Live UI owner changed source lifetime or state-policy scope");
    ++records;
    if (diagnostic) {
      *diagnostic = {};
      diagnostic->command = command; diagnostic->resource = value.resource.native;
      diagnostic->format = record_format;
      diagnostic->native_state = value.native_state; diagnostic->observed_state = 128;
      diagnostic->observed = true; diagnostic->recording_cookie = 0xc00 + value.sequence;
      diagnostic->result = fail_record ? status::conflicting_state : status::recorded;
      diagnostic->stage = fail_record ? record_stage::conflicting_state : record_stage::recorded;
      if (!fail_record) {
        diagnostic->copy_state = 128; diagnostic->copy_state_known = true; diagnostic->used_observed_state = true;
      }
    }
    if (fail_record) return {};
    const auto id = ++next_ticket;
    auto lease = std::shared_ptr<const texture_reference>(reinterpret_cast<const texture_reference *>(id), [](auto *) {});
    auto &snapshot = snapshots[id];
    snapshot.input = value; snapshot.input.source.reset(); snapshot.state_policy = state_policy;
    snapshot.texture.ownership = lease; snapshot.texture.texture = 0x500 + id;
    snapshot.texture.device = device; snapshot.texture.device_identity = device;
    snapshot.texture.width = value.resource.width; snapshot.texture.height = value.resource.height;
    snapshot.texture.area = value.resource.area; snapshot.texture.format = record_format; snapshot.texture.capture_id = id;
    if (during_record) { auto callback = std::move(during_record); callback(); }
    return {id, std::move(lease)};
  }
  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful) {
    auto &snapshot = snapshots.at(ticket.id);
    snapshot.finished = true; snapshot.success = successful && !snapshot.released;
  }
  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out) {
    if (during_acquire) { auto callback = std::move(during_acquire); callback(); }
    const auto &snapshot = snapshots.at(ticket.id);
    if (snapshot.released || (snapshot.finished && !snapshot.success)) return status::failed;
    if (!snapshot.finished) return status::recorded;
    if (snapshot.status == status::ready) out = snapshot.texture;
    return snapshot.status;
  }
  status acquire_local_texture(const diagnostic_ticket &ticket, std::uint64_t consumer_queue, diagnostic_texture &out) {
    local_consumer_queue = consumer_queue;
    const auto state = acquire_diagnostic_texture(ticket, out);
    if (state == status::submitted && consumer_queue == 0x900) { out = snapshots.at(ticket.id).texture; return status::ready; }
    return state;
  }
  void release_diagnostic_texture(const diagnostic_ticket &ticket) {
    auto &snapshot = snapshots.at(ticket.id);
    require(!snapshot.released, "owner released one ticket twice");
    snapshot.released = true; ++releases;
  }
}

int main() {
  try {
    completed_and_pending(); std::puts("PASS completed real alpha, exact provenance, pending hold, unchanged request and fixed age");
    bounded_pending(); std::puts("PASS per-source two-slot bound preserves pending GPU work and replaces only completed old snapshots");
    gate_diagnostics_preserve_scope_mismatch_without_capture(); std::puts("PASS bounded hook gate metadata preserves exact request mismatches without authorizing capture");
    revoke_scopes(); std::puts("PASS epoch/revision/device/viewport/size/off/runtime and in-flight invalidation");
    reservation_race(); std::puts("PASS collect versus in-flight native-record publication preserves the pending reservation");
    invalid_tags_and_sdk_failure(); std::puts("PASS null/partial/unsupported tags, SDK failure, native transient failure and out-of-order calls");
    admission_and_isolation(); std::puts("PASS actual shape/device/alpha format, unique runtime scope and monotonic source admission");
    diagnostic_rejections_and_no_attempt(); std::puts("PASS exact live rejection diagnostics and distinct no-attempt states");
    diagnostic_newest_scope_wins(); std::puts("PASS newest diagnostic ownership, stale completion/invalidation guards and metadata-only lifetime");
    diagnostic_in_flight_is_not_a_rejection(); std::puts("PASS pending native diagnostic is distinct from a completed result and SDK completion");
    dedicated_kind_format_and_provenance(); std::puts("PASS typed dedicated UI alpha/color formats and exact tag semantics");
    dedicated_preference_and_same_batch_fallback(); std::puts("PASS independent UI candidates retain lower sources through priority selection and same-call null/native fallback");
    hudless_independent_capacity_and_pairing_provenance(); std::puts("PASS bounded FG-off HUD-less capture and immutable tag/current presentation-generation provenance");
    local_queue_acquisition_preserves_current_candidate(); std::puts("PASS explicit local consumer queue selects current submitted pixels without changing strict diagnostic acquisition");
    explicit_source_filter_prevents_higher_priority_starvation(); std::puts("PASS exact source filtering prevents unreviewed priority starvation and fallback");
    source_filter_change_revokes_ready_pending_and_inflight_attempts(); std::puts("PASS source-filter changes revoke completed, pending and in-flight old reservations");
    hudless_pairs_with_the_real_frame_under_frame_generation(); std::puts("PASS HUD-less pairs with its real frame after generated presents, late captures within retained history, never stale or reversed generations");
    declared_lifetimes_preserve_state_policy(); std::puts("PASS UI tag lifetimes preserve provenance and choose observed-at-call or strict longer-lived state policy");
    token_batch_pairs_the_newest_common_frame(); std::puts("PASS S3 token batch: the newest HUD-less and Backbuffer pair of one token generation, numbered or not, read-only; different or missing tokens and a scope change never pair; only Backbuffer snapshots ask for the token clock");
    begin_refusals_name_their_stage(); std::puts("PASS S3 begin refusals: no or ambiguous request, filtered kind, not newer, shape, unsupported lifetime and no reservation each name their stage");
    mask::invalidate_all();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}

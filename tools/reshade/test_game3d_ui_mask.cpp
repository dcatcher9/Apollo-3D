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
    capture::diagnostic_texture texture;
    capture::status status{capture::status::recorded};
    bool finished{}, success{}, released{};
  };
  std::map<std::uint64_t, native_snapshot> snapshots;
  std::uint64_t next_ticket{};
  unsigned records{}, releases{};
  bool fail_record{};
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
    fail_record = false; during_record = {}; during_acquire = {};
    mask::set_request(request());
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
    return value;
  }
  mask::attempt begin(const capture::input &value) { return mask::begin({value, 0x300, 1}, value); }
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
    auto second = capture_one(2, 1010); auto third = capture_one(3, 1020);
    auto skipped = capture_one(4, 1030);
    require(!skipped.ticket && skipped.runtime == runtime && records == 3 && releases == 0,
      "full owner evicted a pending snapshot or allocated unbounded work");
    selected(1, 1031);
    complete(second); selected(2, 1031);
    auto fourth = capture_one(5, 1040);
    require(fourth.ticket && records == 4 && !snapshots.at(third.ticket.id).released,
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
      !report.record_attempted && report.sdk_result_known && records == 3,
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
    require(source_lifetime.expired() && mask::query_diagnostic(runtime, report),
      "diagnostic metadata retained an application source COM lease");
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
}

namespace sunshine_streamline::depth_capture {
  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      local_texture_state_policy state_policy) {
    require(state_policy == local_texture_state_policy::prefer_observed_recording && value.frame_generation_input &&
      value.valid_until == scene::lifetime::at_call && value.force_snapshot,
      "Live UI owner did not prefer observed-recording state for its synchronous FG input");
    ++records;
    if (diagnostic) {
      *diagnostic = {};
      diagnostic->command = command; diagnostic->resource = value.resource.native;
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
    snapshot.texture.ownership = lease; snapshot.texture.texture = 0x500 + id;
    snapshot.texture.device = device; snapshot.texture.device_identity = device;
    snapshot.texture.width = value.resource.width; snapshot.texture.height = value.resource.height;
    snapshot.texture.area = value.resource.area; snapshot.texture.format = 24; snapshot.texture.capture_id = id;
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
  void release_diagnostic_texture(const diagnostic_ticket &ticket) {
    auto &snapshot = snapshots.at(ticket.id);
    require(!snapshot.released, "owner released one ticket twice");
    snapshot.released = true; ++releases;
  }
}

int main() {
  try {
    completed_and_pending(); std::puts("PASS completed real alpha, exact provenance, pending hold, unchanged request and fixed age");
    bounded_pending(); std::puts("PASS three-slot bound preserves pending GPU work and replaces only completed old snapshots");
    revoke_scopes(); std::puts("PASS epoch/revision/device/viewport/size/off/runtime and in-flight invalidation");
    reservation_race(); std::puts("PASS collect versus in-flight native-record publication preserves the pending reservation");
    invalid_tags_and_sdk_failure(); std::puts("PASS null/partial/unsupported tags, SDK failure, native transient failure and out-of-order calls");
    admission_and_isolation(); std::puts("PASS actual shape/device/alpha format, unique runtime scope and monotonic source admission");
    diagnostic_rejections_and_no_attempt(); std::puts("PASS exact live rejection diagnostics and distinct no-attempt states");
    diagnostic_newest_scope_wins(); std::puts("PASS newest diagnostic ownership, stale completion/invalidation guards and metadata-only lifetime");
    diagnostic_in_flight_is_not_a_rejection(); std::puts("PASS pending native diagnostic is distinct from a completed result and SDK completion");
    mask::invalidate_all();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}

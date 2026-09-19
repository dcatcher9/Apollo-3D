// SPDX-License-Identifier: GPL-3.0-only
// Pure CPU recording/submission identity model. No graphics runtime or native API.
#include "streamline_command_association.h"
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_streamline::commands;
  constexpr std::uint64_t device = 1, queue = 10, copy_command = 20, evaluation_command = 21, loss = 7;
  void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
  struct fixture {
    tracker model;
    fixture() {
      model.queue_initialized(queue, device);
      model.command_initialized(copy_command, device);
      model.command_initialized(evaluation_command, device);
    }
    void submit(std::uint64_t command, std::uint64_t target = queue) {
      model.command_closed(command);
      model.command_executed(target, command);
    }
  };
  void test_separate_recordings() {
    fixture f;
    const auto copy = f.model.mark(copy_command, loss);
    const auto evaluation = f.model.mark(evaluation_command, loss);
    require(f.model.same_open_recording(evaluation, loss), "evaluation entry not open");
    require(f.model.associate(evaluation, copy, queue, loss).state == status::open_recording, "unsubmitted open list accepted");
    f.submit(copy_command);
    f.submit(evaluation_command);
    auto result = f.model.associate(evaluation, copy, queue, loss);
    require(!result.associated() && result.state == status::same_queue_callbacks_order_unverified &&
        !result.same_recording && result.ordering == order::unknown &&
        result.copy_submission < result.evaluation_submission && result.queue == queue && result.queue_generation,
      "separate-list callback order was claimed as native queue order");
    require(!result.content_registered && !result.final_color_registered, "command order fabricated content or color proof");
    require(!f.model.same_open_recording(evaluation, loss), "submitted evaluation still marked open");
    // CPU marker capture order is deliberately opposite queue submission order.
    f.model.command_reset(copy_command);
    f.model.command_reset(evaluation_command);
    const auto earlier_cpu_copy = f.model.mark(copy_command, loss);
    const auto later_cpu_eval = f.model.mark(evaluation_command, loss);
    f.submit(evaluation_command);
    f.submit(copy_command);
    result = f.model.associate(later_cpu_eval, earlier_cpu_copy, queue, loss);
    require(!result.associated() && result.state == status::same_queue_callbacks_order_unverified &&
        result.ordering == order::unknown && result.evaluation_submission < result.copy_submission,
      "reversed callback order became a cross-recording execution proof");
  }
  void test_same_recording_and_reset() {
    fixture f;
    const auto first = f.model.mark(copy_command, loss);
    const auto second = f.model.mark(copy_command, loss);
    require(first.event != second.event, "markers do not identify distinct positions");
    f.submit(copy_command);
    auto result = f.model.associate(second, first, queue, loss);
    require(result.associated() && result.same_recording && result.ordering == order::copy_before_evaluation,
      "same recording marker order lost");
    require(f.model.associate(first, first, queue, loss).state == status::ambiguous_marker,
      "one event was claimed as two different recording observations");
    require(!f.model.mark(copy_command, loss), "closed/submitted list accepted new capture marker");
    f.model.command_reset(copy_command);
    const auto next = f.model.mark(copy_command, loss);
    require(next.recording_generation != first.recording_generation && next.object_generation == first.object_generation,
      "reset failed to distinguish recording from object lifetime");
    require(!f.model.same_open_recording(first, loss), "reset during evaluation retained old recording proof");
    // An earlier uniquely submitted recording remains historical evidence after
    // reset; it is not mistaken for the newly recorded native command object.
    require(f.model.associate(second, first, queue, loss).associated(), "reset erased unambiguous submitted history");
    f.model.command_destroyed(copy_command);
    require(f.model.associate(second, first, queue, loss).state == status::unknown_command, "destroy left recording applicable");
    f.model.command_initialized(copy_command, device);
    require(f.model.associate(second, first, queue, loss).state == status::command_recreated, "native address reuse inherited old object lifetime");
  }
  void test_submission_rejections() {
    fixture f;
    const auto copy = f.model.mark(copy_command, loss), evaluation = f.model.mark(evaluation_command, loss);
    f.model.command_closed(copy_command);
    f.model.command_closed(evaluation_command);
    require(f.model.associate(evaluation, copy, queue, loss).state == status::missing_submission,
      "close observation alone fabricated a queue submission");
    f.model.command_executed(queue, copy_command);
    f.model.command_executed(queue, evaluation_command);
    require(f.model.associate(evaluation, copy, 999, loss).state == status::unknown_queue, "unknown target queue accepted");
    f.model.command_executed(queue, copy_command);
    require(f.model.associate(evaluation, copy, queue, loss).state == status::repeated_submission,
      "repeated recording submission accepted");
    fixture unknown;
    const auto c = unknown.model.mark(copy_command, loss), e = unknown.model.mark(evaluation_command, loss);
    unknown.submit(copy_command);
    unknown.submit(evaluation_command, 999);
    require(unknown.model.associate(e, c, queue, loss).state == status::unknown_queue, "submission to unobserved queue accepted");
    unknown.model.queue_initialized(999, device);
    require(unknown.model.associate(e, c, 999, loss).state == status::unknown_queue,
      "later queue initialization retroactively validated unknown submission");
    fixture unclosed;
    const auto u = unclosed.model.mark(copy_command, loss), v = unclosed.model.mark(evaluation_command, loss);
    unclosed.model.command_executed(queue, copy_command);
    unclosed.submit(evaluation_command);
    require(unclosed.model.associate(v, u, queue, loss).state == status::missing_close, "missing close observation accepted");
  }
  void test_queue_identity_and_nesting() {
    fixture f;
    f.model.queue_initialized(11, device);
    const auto copy = f.model.mark(copy_command, loss), evaluation = f.model.mark(evaluation_command, loss);
    f.submit(copy_command);
    f.submit(evaluation_command, 11);
    require(f.model.associate(evaluation, copy, queue, loss).state == status::different_queue,
      "different queues were implicitly GPU-ordered");
    f.model.queue_destroyed(11);
    require(f.model.associate(evaluation, copy, queue, loss).state == status::unknown_queue, "destroyed queue accepted");
    f.model.queue_initialized(11, device);
    require(f.model.associate(evaluation, copy, 11, loss).state == status::queue_recreated, "queue address reuse inherited submission");
    fixture devices;
    devices.model.queue_initialized(12, 2);
    const auto c = devices.model.mark(copy_command, loss), e = devices.model.mark(evaluation_command, loss);
    devices.submit(copy_command);
    devices.submit(evaluation_command, 12);
    require(devices.model.associate(e, c, queue, loss).state == status::different_device, "cross-device submission accepted");
    fixture nested;
    const auto n = nested.model.mark(copy_command, loss), m = nested.model.mark(evaluation_command, loss);
    nested.model.command_secondary_executed(copy_command, evaluation_command);
    nested.submit(copy_command);
    nested.submit(evaluation_command);
    require(nested.model.associate(m, n, queue, loss).state == status::nested_execution,
      "secondary execution was silently flattened into primary ordering");
  }
  void test_loss_and_eviction() {
    fixture f;
    require(!f.model.mark(999, loss), "unobserved command identity accepted");
    const auto copy = f.model.mark(copy_command, loss), evaluation = f.model.mark(evaluation_command, loss);
    f.submit(copy_command);
    f.submit(evaluation_command);
    require(f.model.associate(evaluation, copy, queue, loss + 1).state == status::lost_observation,
      "stale observation epoch accepted");
    for (unsigned i = 0; i != 260; ++i) f.model.command_reset(copy_command);
    require(f.model.associate(evaluation, copy, queue, loss).state == status::recording_evicted,
      "evicted recording inherited a reused bounded slot");
    f.model.clear();
    require(f.model.associate(evaluation, copy, queue, loss).state == status::lost_observation,
      "tracker reset inherited old epoch");
  }
  void test_cookie_recovery() {
    tracker model;
    constexpr std::uint64_t command_cookie = 401, queue_cookie = 501;
    model.queue_initialized(queue, device, queue_cookie);
    model.command_initialized(copy_command, device, command_cookie);
    const auto copy = model.mark(copy_command, loss);
    model.command_initialized(copy_command, device, command_cookie);
    const auto evaluation = model.mark(copy_command, loss);
    require(copy.object_generation == evaluation.object_generation && copy.recording_generation == evaluation.recording_generation,
      "idempotent live-object ensure changed recording identity");
    model.command_closed(copy_command);
    model.queue_initialized(queue, device, queue_cookie);
    model.command_executed(queue, copy_command);
    require(model.associate(evaluation, copy, queue, loss).associated(), "idempotent queue ensure invalidated proof");
    model.clear(); // Missed lifecycle invalidates all old markers and cached identities.
    model.command_reset(copy_command);
    require(!model.mark(copy_command, loss + 1), "native-only reset reseeded an unknown lifetime");
    model.command_reset(copy_command, device, command_cookie);
    const auto fresh_copy = model.mark(copy_command, loss + 1), fresh_eval = model.mark(copy_command, loss + 1);
    model.command_closed(copy_command);
    model.queue_initialized(queue, device, queue_cookie);
    model.command_executed(queue, copy_command);
    require(model.associate(fresh_eval, fresh_copy, queue, loss + 1).associated(),
      "ordinary reset/execute with private lifetime cookies failed recovery");
    require(model.associate(evaluation, copy, queue, loss + 1).state == status::lost_observation,
      "cookie recovery revived stale pre-loss markers");
    model.command_initialized(copy_command, device, command_cookie + 1);
    require(model.associate(fresh_eval, fresh_copy, queue, loss + 1).state == status::command_recreated,
      "different private cookie inherited native address lifetime");
  }
}
int main() {
  try {
    test_separate_recordings();
    test_same_recording_and_reset();
    test_submission_rejections();
    test_queue_identity_and_nesting();
    test_loss_and_eviction();
    test_cookie_recovery();
    std::puts("PASS bounded D3D12 recording identity, callback provenance, uncertain-order rejection and cookie recovery");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL command association: %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: GPL-3.0-only
#include "depth_probe_policy.h"
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_depth_probe;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

  void unrelated_completion_preserves_pending_visit() {
    // The real failing interval: A is sampled every 250 ms while B waits for
    // the old challenger's 50-frame retirement. No backup belongs to B yet.
    std::uint64_t pending = 79;
    for (unsigned frame = 0; frame != 50; ++frame) {
      if (frame % 15 == 0 && !keep_unrelated_probe(true, 80, pending)) pending = 0;
      require(pending == 79, "Anchor completion cancelled a probe before backup allocation");
    }
    require(keep_unrelated_probe(true, 80, pending), "Allocated visit lost the same ownership protection");
    require(!keep_unrelated_probe(true, 79, pending), "A visit's own completion was exempted from normal completion policy");
    require(!keep_unrelated_probe(false, 80, pending), "Non-raw completion changed the legacy lifecycle");
    require(!keep_unrelated_probe(true, 80, 0), "An unrelated completion invented a pending visit");
  }

  void alternating_present_discovery() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1);
    require(!hint.take(10), "One observation prematurely prioritized a peer");
    hint.observe(80, 10, 79, 1);
    require(!hint.take(10), "Repeated effects counted twice");
    hint.observe(80, 11, 0, 0); // A has its own valid copy on this present.
    require(!hint.take(11), "Off-turn hint was treated as current rendering");
    hint.observe(80, 12, 79, 1);
    require(hint.take(12) == 79, "Repeated uncovered B present waited for the full candidate round robin");
  }

  void sampler_free_on_another_members_present() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1); hint.observe(80, 11, 0, 0);
    hint.observe(80, 12, 0, 0); hint.observe(80, 13, 79, 1);
    hint.observe(80, 14, 0, 0); // Readback completes on A/C, not on B's turn.
    require(hint.take(14) == 79, "Probe ownership phase-locked to uncovered B presents");
    hint = {};
    hint.observe(80, 10, 79, 1); hint.observe(80, 11, 81, 1);
    hint.observe(80, 13, 79, 1); hint.observe(80, 14, 81, 1);
    hint.observe(80, 15, 0, 0);
    require(hint.take(15) == 81, "Recent hint did not prefer the latest observed candidate");
  }

  void capture_ownership_and_retirement() {
    std::array<capture_owner, 4> owners {};
    require(!capture_keeps_backup(79, owners), "Sampler-only reference skipped retirement");
    owners[0] = {79, 1234, 0};
    require(capture_keeps_backup(79, owners), "Adopted capture allocation paid an unrelated retirement delay");
    owners[0].retire_after = 100;
    require(!capture_keeps_backup(79, owners), "Retiring capture slot was treated as a permanent owner");
    owners[1] = {79, 0, 0};
    require(!capture_keeps_backup(79, owners), "Unallocated capture slot claimed ownership");
    owners[2] = {80, 1234, 0};
    require(!capture_keeps_backup(79, owners), "A different source's view granted ownership");
    owners[3] = {79, 5678, 0};
    require(capture_keeps_backup(79, owners), "A live matching owner was hidden by retiring entries");
    require(!capture_keeps_backup(0, owners), "Missing source granted ownership");
  }

  void retirement_wait_does_not_consume_measurement_time() {
    std::uint64_t phase_started = 1000;
    require(!phase_expired(phase_started, 2562), "50-frame retirement at32FPS exhausted queue allowance");
    require(phase_expired(phase_started, 3000), "A probe unable to allocate could wait forever");
    require(phase_expired(phase_started, 999), "Clock rollback extended a phase");
    // Allocation changes ownership. At32FPS a three-way rotating source has
    // ~94ms turns, so125ms sampling can need188ms between usable captures.
    phase_started = 2562;
    for (std::uint64_t completed : {2687u, 2875u, 3063u})
      require(!phase_expired(phase_started, completed), "Queued retirement time starved three actual capture completions");
    require(phase_expired(1000, 3063), "Regression schedule no longer distinguishes a shared queue/capture deadline");
    require(!phase_expired(phase_started, 4561) && phase_expired(phase_started, 4562),
      "Allocated capture timeout was removed or extended by progress");
  }

  void three_member_rotation() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1);
    hint.observe(80, 11, 81, 1);
    hint.observe(80, 12, 0, 0);
    hint.observe(80, 13, 79, 1);
    require(hint.take(13) == 79, "Three-frame rotation lost repeated active evidence");
    hint.round_robin_started();
    hint.observe(80, 14, 81, 1);
    require(hint.take(14) == 81, "Uncovered B/C/B/C reset each other's discovery history");
  }

  void four_member_rotation_and_crowded_hints() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1); hint.observe(80, 11, 81, 1);
    hint.observe(80, 12, 82, 1); hint.observe(80, 13, 0, 0);
    hint.observe(80, 14, 79, 1);
    require(hint.take(14) == 79, "Four-way rotation was never discoverable");
    hint.round_robin_started();
    hint.observe(80, 15, 81, 1);
    require(hint.take(15) == 81, "Four-way peer lost independent discovery progress");
    hint = {};
    for (std::uint64_t frame = 1; frame <= 5; ++frame) hint.observe(80, frame, 100 + frame, 1);
    hint.observe(80, 6, 101, 1);
    require(!hint.take(6), "Evicted/expired candidate kept its old observation");
    hint.observe(80, 7, 105, 1);
    require(hint.take(7) == 105, "Bounded hints lost a recent candidate under churn");
  }

  void bounded_priority() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1); hint.observe(80, 12, 79, 1);
    require(hint.take(12) == 79, "Initial priority visit missing");
    for (std::uint64_t frame = 14; frame < 100; frame += 2) {
      hint.observe(80, frame, 79, 1);
      require(!hint.take(frame), "Uncapturable recurring peer starved round-robin discovery");
    }
    hint.round_robin_started();
    hint.observe(80, 100, 79, 1);
    require(hint.take(100) == 79, "Priority never recovered after yielding an ordinary visit");
  }

  void stale_or_changed_identity_is_not_reused() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1); hint.observe(80, 12, 79, 1);
    require(!hint.take(11), "A future observation was used on an older present");
    require(!hint.take(17), "Expired queued activity was reused");
    hint.observe(80, 17, 79, 1);
    require(!hint.take(17), "Activity beyond the normal grace retained old observations");
    hint.observe(80, 18, 79, 2);
    require(!hint.take(18), "Changed capture layout reused old activity evidence");
    hint.observe(80, 19, 81, 2);
    require(!hint.take(19), "Replacement resource inherited another candidate's evidence");
    hint.observe(80, 20, 81, 2);
    require(hint.take(20) == 81, "Live replacement did not establish its own hint");
  }

  void anchor_and_clock_changes_reset() {
    discovery_hint hint;
    hint.observe(80, 10, 79, 1); hint.observe(80, 12, 79, 1);
    hint.observe(82, 13, 79, 1);
    require(!hint.take(13), "New anchor inherited the old discovery observation");
    hint.observe(82, 14, 79, 1);
    require(hint.take(14) == 79, "New anchor could not rediscover a current peer");
    hint.observe(82, 2, 79, 1);
    require(!hint.take(2), "Clock rollback reused a prior hint");
    hint.observe(0, 3, 79, 1); hint.observe(82, 4, 79, 1);
    require(!hint.take(4), "Missing anchor retained a prior hint");
  }
}

int main() {
  try {
    unrelated_completion_preserves_pending_visit(); alternating_present_discovery(); three_member_rotation();
    four_member_rotation_and_crowded_hints(); bounded_priority();
    stale_or_changed_identity_is_not_reused(); anchor_and_clock_changes_reset(); capture_ownership_and_retirement();
    sampler_free_on_another_members_present(); retirement_wait_does_not_consume_measurement_time();
    std::puts("depth_probe_policy: 10 cases passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

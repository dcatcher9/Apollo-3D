// SPDX-License-Identifier: GPL-3.0-only
// The per-resource last-proven state (proven_state.h), which admits the
// Generic end-of-frame copy (the contract matrix's D10): a creation state
// holds on every queue until a list uses the texture, and not once enhanced
// barriers were seen; an executed list's proof holds only on its own queue;
// every update takes a new version, so a copy recorded from a proof can tell
// that it changed before its list ran; whatever may have changed a texture
// unseen leaves it unproven; and a full table forgets its least recently used
// texture. The enhanced-layout latch (D6, N8) of a tracked or watched texture
// survives unseen work and later proofs, a watched texture's proof serves the
// latch alone, and forgetting a latched texture latches every texture not
// held. The capture owner's feeding of it is proven on real lists by the
// contract matrix's device layer (end_of_frame and earlier enhanced layout
// cells).
#include "proven_state.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
  using sunshine_proven_state::state;
  using table = sunshine_proven_state::table<4>;

  constexpr std::uint32_t depth_write = 0x10, depth_read = 0x20, shader_read = 0xc0;
  constexpr std::uint64_t direct = 0x1000, other = 0x2000;

  unsigned failures {};

  void check(bool value, const char *what) {
    if (!value) {
      std::fprintf(stderr, "FAIL %s\n", what);
      ++failures;
    }
  }

  bool proven(const state &value, std::uint32_t expected) {
    return value && !value.enhanced && value.value == expected;
  }
}  // namespace

int main() {
  {
    table owner;
    owner.track(1, true, depth_write);
    check(proven(owner.query(1, direct, true), depth_write) && proven(owner.query(1, other, true), depth_write), "A creation state holds on every queue until a list uses the texture");
    check(!owner.query(1, direct, false), "A creation state proves nothing once enhanced barriers were seen");
    owner.track(2, false, 0);
    check(owner.tracked(2) && !owner.query(2, direct, true), "A texture created in no legacy state is tracked but unproven");
    check(!owner.query(3, direct, true) && !owner.version(3), "An untracked texture is unproven");
  }
  {
    table owner;
    owner.track(1, true, depth_write);
    const auto created = owner.version(1);
    owner.prove(1, direct, depth_read);
    const auto read = owner.query(1, direct, false);
    check(proven(read, depth_read) && read.queue == direct, "An executed list's proof holds on its queue, also when creation states are not trusted");
    check(!owner.query(1, other, true), "Another queue's copy is not ordered after that list: unproven there");
    check(read.version && read.version != created, "Each proof takes a new version");
    owner.prove(1, direct, depth_read);
    check(owner.version(1) != read.version, "Proving the same state again still takes a new version");
    owner.prove(1, other, shader_read);
    check(!owner.query(1, direct, true) && proven(owner.query(1, other, true), shader_read), "A later list on another queue moves the proof to that queue");
    owner.prove_layout(1, direct, 3, 0x40);
    const auto layout = owner.query(1, direct, true);
    check(layout && layout.enhanced && layout.layout == 3 && layout.access == 0x40, "An enhanced barrier proves its layout and access");
    owner.unprove(1);
    check(owner.tracked(1) && !owner.query(1, direct, true) && !owner.version(1), "Unseen changes leave a tracked texture unproven");
    owner.prove(1, direct, depth_write);
    owner.track(2, true, depth_write);
    owner.unprove_all();
    check(!owner.query(1, direct, true) && !owner.query(2, direct, true) && owner.size() == 2, "An unobserved execution unproves every texture and keeps tracking them");
    owner.prove(2, direct, depth_read);
    check(proven(owner.query(2, direct, true), depth_read), "A later barrier proves it again");
    owner.track(2, true, depth_write);
    check(proven(owner.query(2, other, true), depth_write) && !owner.query(2, other, true).queue, "Tracking again restarts from the creation state");
  }
  {
    table owner;
    for (std::uint64_t resource = 1; resource <= 4; ++resource) {
      owner.track(resource * 10, true, depth_write);
    }
    owner.query(10, direct, true);
    owner.prove(20, direct, depth_read);
    owner.track(50, true, depth_write);
    check(owner.size() == 4 && !owner.tracked(30) && owner.tracked(10) && owner.tracked(20) && owner.tracked(40) && owner.tracked(50), "A full table forgets its least recently used texture first");
    owner.track(5, true, depth_read);
    check(!owner.tracked(40) && proven(owner.query(5, direct, true), depth_read) && proven(owner.query(50, direct, true), depth_write) && proven(owner.query(20, direct, true), depth_read), "Lookups stay exact after insertions before and after the others");
  }
  {
    table owner;
    owner.track(1, true, depth_write);
    owner.watch(2);
    check(owner.held(2) && !owner.tracked(2) && !owner.latched(2).latched, "A watched texture is held, not tracked, and starts unlatched");
    owner.prove(2, direct, shader_read);
    check(!owner.query(2, direct, true), "A watched texture's proof never serves a query");
    owner.prove_layout(1, direct, 3, 0x40);
    check(!owner.latched(1).latched, "A proof alone does not latch");
    owner.set_latch(1);
    const auto latched = owner.latched(1);
    check(latched.latched && latched.proven.enhanced && latched.proven.layout == 3 && latched.proven.queue == direct, "A latched texture reports its proof on the queue that made it");
    owner.prove(1, other, depth_read);
    owner.unprove_all();
    check(owner.latched(1).latched && !owner.latched(1).proven, "The latch survives a later legacy proof and unseen work; the proof does not");
    owner.track(1, true, depth_write);
    check(owner.latched(1).latched && !owner.latched(1).proven, "Tracking again keeps the latch; a creation state is no latched proof");
    owner.set_latch(2);
    owner.prove_layout(2, other, 3, 0x40);
    check(owner.latched(2).proven.queue == other && !owner.query(2, other, true), "A watched texture latches and keeps its proof for the latch alone");
    owner.track(2, false, 0);
    check(owner.tracked(2) && owner.latched(2).latched && !owner.query(2, other, true), "A watched texture becomes tracked, unproven, latch kept");
    owner.watch(1);
    check(owner.tracked(1), "Watching a tracked texture leaves it tracked");
    check(!owner.latch_lost() && !owner.latched(9).latched && owner.latched_count() == 2, "A texture never held is unlatched until a latch is lost");
    owner.watch(3);
    owner.watch(4);
    owner.watch(5);  // Forgets 1, the least recently used.
    check(!owner.held(1) && owner.latch_lost() && owner.latched(1).latched && owner.latched(9).latched && !owner.latched(3).latched, "Forgetting a latched texture latches every texture not held");
  }
  if (failures) {
    std::fprintf(stderr, "%u proven-state check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::puts("PASS per-resource last-proven state");
  return EXIT_SUCCESS;
}

// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_content.h"
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_streamline;
  constexpr content::resource_key source{101, 11}, backup{102, 12};
  constexpr std::uint64_t device = 44, command = 201, other_command = 202;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  struct fixture {
    commands::tracker commands;
    content::ledger ledger;
    fixture() {
      commands.command_initialized(command, device, 1);
      commands.command_initialized(other_command, device, 2);
      ledger.resource_initialized(source, device, true);
      ledger.resource_initialized(backup, device, true);
    }
    commands::recording_marker mark(std::uint64_t native = command) { return commands.mark(native, 0); }
    content::use_snapshot use(std::uint64_t native = command) { return ledger.use(mark(native), source.native, device); }
    content::copy_snapshot copy(std::uint64_t native = command) { return ledger.copy(mark(native), source, backup, device); }
    void write(content::resource_key key = source, std::uint64_t native = command) { ledger.write(mark(native), key); }
    content::association associate(const content::use_snapshot &use, const content::copy_snapshot &copy) {
      return ledger.associate(use, copy, source, backup);
    }
  };
  void accepted_orders() {
    fixture f;
    auto copy = f.copy(); auto use = f.use();
    auto match = f.associate(use, copy);
    require(match.matched() && !match.coverage_complete && !match.final_color_registered, "copy/use unchanged did not retain narrow observed match");
    f.commands.command_reset(command); f.ledger.command_reset(command);
    use = f.use(); copy = f.copy();
    f.write(); // Clear the source after both observations: the backup remains intact.
    require(f.associate(use, copy).matched(), "source clear after use+copy rejected preserved backup");
    require(f.ledger.unchanged(use) == content::status::different_source_content, "within-evaluate clear was not detected");
  }
  void source_mutations() {
    fixture f;
    auto copy = f.copy(); f.write(); auto use = f.use();
    require(f.associate(use, copy).state == content::status::different_source_content, "copy/clear/evaluate accepted");
    use = f.use(); f.write(); copy = f.copy();
    require(f.associate(use, copy).state == content::status::different_source_content, "evaluate/write/copy accepted");
    use = f.use(); f.write(source, other_command); copy = f.copy();
    require(!f.associate(use, copy).matched(), "foreign recording write between use/copy accepted");
    use = f.use(); copy = f.copy(); f.write(source, other_command); f.write();
    require(!f.associate(use, copy).matched(), "foreign recording interference hidden by a later same-recording write");
  }
  void backup_mutations() {
    fixture f;
    auto use = f.use(); auto copy = f.copy();
    f.write({backup.native, 0});
    require(f.associate(use, copy).state == content::status::backup_overwritten, "native-only backup mutation retained copy");
    copy = f.copy(); auto replacement = f.copy();
    require(f.associate(use, copy).state == content::status::backup_overwritten, "new copy did not supersede old copy");
    require(f.associate(use, replacement).matched(), "latest identical source copy rejected");
    f.ledger.copy(f.mark(), {999, 9}, backup, device);
    require(!f.associate(use, replacement).matched(), "copy from untracked source preserved an older backup binding");
  }
  void lifetime_and_binding() {
    fixture f;
    auto use = f.use(); auto copy = f.copy();
    require(f.ledger.associate(use, copy, source, {backup.native, 0}).state == content::status::backup_binding_mismatch, "query accepted zero backup lifetime");
    f.ledger.resource_initialized(backup, device, true);
    require(f.associate(use, copy).matched(), "idempotent live registration invalidated copy");
    f.ledger.resource_destroyed(backup);
    f.ledger.resource_initialized(backup, device, true);
    require(!f.associate(use, copy).matched(), "destroy/reinitialize same caller ID resurrected copy");
    copy = f.copy();
    f.ledger.resource_initialized({backup.native, backup.lifetime + 1}, device, true);
    require(!f.associate(use, copy).matched(), "pooled backup reassignment retained copy");
    f.ledger.resource_destroyed(backup); // Stale destroy must not remove new generation.
    require(f.ledger.use(f.mark(), backup.native, device).resource.lifetime == backup.lifetime + 1, "stale destroy removed newer resource");
    f.ledger.resource_initialized(backup, device, true);
    copy = f.copy();
    f.ledger.resource_initialized({source.native, source.lifetime + 1}, device, true);
    require(!f.associate(use, copy).matched(), "source native reuse retained old snapshot");
  }
  void invalidation_and_reset() {
    fixture f;
    auto use = f.use(); auto copy = f.copy();
    f.ledger.invalidate(f.mark());
    require(f.associate(use, copy).state == content::status::invalidated_resource, "unknown alias did not invalidate prior content evidence");
    require(!f.use().valid(), "new use escaped recording ban");
    f.commands.command_reset(command); f.ledger.command_reset(command);
    require(!f.associate(use, copy).matched(), "reset resurrected old poisoned snapshots");
    use = f.use(); copy = f.copy();
    require(f.associate(use, copy).matched(), "fresh recording did not recover from prior recording ban");
    f.ledger.invalidate(f.mark(), source);
    require(!f.associate(use, copy).matched(), "unsupported source mutation did not revoke snapshots");
    f.ledger.clear();
    f.ledger.resource_initialized(source, device, true); f.ledger.resource_initialized(backup, device, true);
    require(!f.associate(use, copy).matched(), "tracking loss resurrected old snapshot after resource re-ensure");
  }
  void bounded_history_and_guards() {
    fixture f;
    f.write(); f.write(); auto use = f.use(); auto copy = f.copy();
    require(f.associate(use, copy).matched(), "coalesced draws did not retain content");
    auto delayed = f.mark(); f.write();
    require(f.ledger.use(delayed, source.native, device).state == content::status::superseded_marker, "delayed entry snapshot used later content");
    require(f.ledger.use({}, source.native, device).state == content::status::unknown_recording, "untracked command accepted");
    require(f.ledger.use(f.mark(), source.native, device + 1).state == content::status::different_device, "cross-device use accepted");
    require(f.ledger.copy(f.mark(), source, backup, device + 1).state == content::status::different_device, "cross-device copy accepted");
    use = f.use(); copy = f.copy(other_command);
    require(f.associate(use, copy).state == content::status::different_recording, "cross-recording content accepted");
    use = f.use(); copy = f.copy();
    for (unsigned i = 0; i != 1030; ++i) {
      f.commands.command_reset(other_command);
      f.ledger.use(f.mark(other_command), source.native, device);
    }
    require(!f.associate(use, copy).matched(), "state eviction resurrected snapshot");
    use = f.use(); copy = f.copy();
    for (unsigned i = 0; i != 260; ++i) f.ledger.resource_initialized({1000 + i, 5000 + i}, device, true);
    f.ledger.resource_initialized(source, device, true); f.ledger.resource_initialized(backup, device, true);
    require(!f.associate(use, copy).matched(), "resource registry eviction resurrected snapshot");
    f.ledger.resource_initialized(source, device, false);
    require(!f.use().valid(), "unsupported flat shape accepted");
  }
  void destroyed_recording_churn() {
    fixture f;
    auto old_use = f.use(); auto old_copy = f.copy();
    f.ledger.invalidate(f.mark());
    f.commands.command_destroyed(command); f.ledger.command_destroyed(command);
    require(!f.associate(old_use, old_copy).matched(), "destroy cleanup resurrected poisoned content");
    for (unsigned i = 0; i != 600; ++i) {
      const std::uint64_t native = 10000 + i;
      f.commands.command_initialized(native, device, native);
      f.ledger.invalidate(f.mark(native));
      f.commands.command_destroyed(native); f.ledger.command_destroyed(native);
    }
    f.commands.command_initialized(command, device, 999);
    require(f.use().valid(), "retired recording bans exhausted capacity after create/destroy churn");
  }
  void unknown_cross_recording_mutation() {
    fixture f;
    auto use = f.use(); auto copy = f.copy();
    f.ledger.invalidate(f.mark(other_command));
    require(!f.associate(use, copy).matched(), "unknown alias in another recording retained content evidence");
    use = f.use(); copy = f.copy();
    require(f.associate(use, copy).matched(), "fresh same-recording snapshots could not recover after earlier unknown operation");
    f.ledger.invalidate(f.mark(other_command)); // Repeat while B is already banned.
    require(!f.associate(use, copy).matched(), "repeat unknown operation in a banned recording retained other-recording evidence");
  }
  void observation_restart() {
    fixture f;
    auto use = f.use(); auto copy = f.copy();
    f.ledger.restart_observations();
    require(!f.associate(use, copy).matched(), "content evidence crossed an observation gap");
    use = f.use(); copy = f.copy();
    require(f.associate(use, copy).matched(), "observation restart lost continuously tracked resource lifetimes");
  }
  void preserved_presentation_chain() {
    constexpr content::resource_key presentation{103, 13};
    fixture f;
    f.ledger.resource_initialized(presentation, device, true);
    const auto use = f.use();
    const auto preserved = f.copy();
    f.write(); // The game clears the source after the preservation boundary.
    const auto presented = f.ledger.forward(f.mark(), preserved, presentation, device);
    require(presented.valid() && content::same(presented.source.resource, source) &&
      content::same(presented.backup, presentation) &&
      f.ledger.associate(use, presented, source, presentation).matched(),
      "Same-recording source/snapshot/presentation lineage was lost or relabeled");
    require(f.ledger.associate(use, preserved, source, presentation).state == content::status::backup_binding_mismatch,
      "A source-to-snapshot marker incorrectly claimed the presentation allocation");
    f.write(backup); // Once forwarded, later snapshot reuse cannot change presentation pixels.
    require(f.ledger.associate(use, presented, source, presentation).matched(),
      "Reusing the intermediate snapshot invalidated already-copied presentation pixels");
    const auto stale = f.ledger.forward(f.mark(), preserved, presentation, device);
    require(stale.state == content::status::backup_overwritten &&
      !f.ledger.associate(use, presented, source, presentation).matched(),
      "Forwarding an overwritten snapshot inherited old lineage or failed to revoke the destination");

    fixture other;
    other.ledger.resource_initialized(presentation, device, true);
    const auto source_use = other.use();
    const auto snapshot = other.copy();
    const auto cross = other.ledger.forward(other.mark(other_command), snapshot, presentation, device);
    require(cross.state == content::status::different_recording &&
      !other.ledger.associate(source_use, cross, source, presentation).matched(),
      "Cross-recording CPU observation fabricated a content ordering proof");
    const auto wrong_order = other.ledger.forward(snapshot.recording, snapshot, presentation, device);
    require(wrong_order.state == content::status::superseded_marker,
      "Forwarded copy accepted a non-increasing recording marker");
    other.ledger.resource_destroyed(backup);
    other.ledger.resource_initialized({backup.native, backup.lifetime+1}, device, true);
    require(!other.ledger.forward(other.mark(), snapshot, presentation, device).valid(),
      "Reused snapshot address inherited prior source lineage");
  }
}
int main() {
  try {
    accepted_orders(); source_mutations(); backup_mutations(); lifetime_and_binding();
    invalidation_and_reset(); bounded_history_and_guards(); destroyed_recording_churn(); unknown_cross_recording_mutation(); observation_restart();
    preserved_presentation_chain();
    std::puts("PASS bounded observed depth content: copy/use order, mutations, overwrite, lifetimes, loss and eviction");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL %s\n", error.what()); return 1; }
}

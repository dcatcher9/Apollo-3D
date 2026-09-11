/**
 * @file tests/unit/test_primary_display.cpp
 * @brief Primary-display transactions against an in-memory CCD and journal adapter.
 */
#include "../tests_common.h"
#include "src/platform/windows/primary_display.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>

namespace {
  using namespace platf::primary_display::detail;

  snapshot_t make_snapshot(const layout_t &positions, bool virtual_aware = true) {
    snapshot_t result;
    for (size_t i = 0; i < positions.size(); ++i) {
      DISPLAYCONFIG_PATH_INFO path {};
      path.flags = DISPLAYCONFIG_PATH_ACTIVE;
      path.sourceInfo.adapterId = {1, 0};
      path.sourceInfo.id = static_cast<UINT32>(i);
      path.targetInfo.adapterId = {1, 0};
      path.targetInfo.id = static_cast<UINT32>(10 + i);
      path.targetInfo.targetAvailable = TRUE;
      path.targetInfo.rotation = DISPLAYCONFIG_ROTATION_IDENTITY;
      path.targetInfo.scaling = DISPLAYCONFIG_SCALING_IDENTITY;
      path.targetInfo.refreshRate = {60000, 1000};
      path.targetInfo.outputTechnology = DISPLAYCONFIG_OUTPUT_TECHNOLOGY_DISPLAYPORT_EXTERNAL;
      path.sourceInfo.modeInfoIdx = static_cast<UINT32>(result.modes.size());
      path.targetInfo.modeInfoIdx = static_cast<UINT32>(result.modes.size() + 1);
      if (virtual_aware) {
        path.flags |= DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE;
        path.sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
        path.sourceInfo.sourceModeInfoIdx = static_cast<UINT32>(result.modes.size());
        path.targetInfo.desktopModeInfoIdx = DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
        path.targetInfo.targetModeInfoIdx = static_cast<UINT32>(result.modes.size() + 1);
      }
      DISPLAYCONFIG_MODE_INFO source {};
      source.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE;
      source.adapterId = path.sourceInfo.adapterId;
      source.id = path.sourceInfo.id;
      source.sourceMode.position = {positions[i].x, positions[i].y};
      source.sourceMode.width = 1920;
      source.sourceMode.height = 1080;
      source.sourceMode.pixelFormat = DISPLAYCONFIG_PIXELFORMAT_32BPP;
      DISPLAYCONFIG_MODE_INFO target {};
      target.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_TARGET;
      target.adapterId = path.targetInfo.adapterId;
      target.id = path.targetInfo.id;
      result.paths.push_back(path);
      result.modes.push_back(source);
      result.modes.push_back(target);
      result.device_paths.push_back(positions[i].device_path);
      result.colors.push_back(platf::display_config::advanced_color_state_t {platf::display_config::advanced_color_api_e::modern, false, true, false, false, false, false, 8, DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR});
    }
    return result;
  }

  const layout_t baseline {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}, {L"virtual", 1920, -200}};

  struct fake_io_t {
    snapshot_t current = make_snapshot(baseline);
    std::optional<journal_t> journal;
    snapshot_t catalog = make_snapshot(baseline);
    std::optional<snapshot_t> available_override;
    bool load_ok = true;
    bool save_ok = true;
    bool clear_ok = true;
    bool apply_ok = true;
    bool ignore_apply = false;
    bool reset_colors_on_apply = false;
    bool color_set_ok = true;
    int query_count = 0;
    int mutate_on_query = 0;
    std::set<int> failed_queries;
    std::function<void(snapshot_t &)> mutation;
    std::function<void(snapshot_t &)> applied_mutation;
    std::set<std::wstring> preserved_exclusive;
    std::vector<std::string> events;

    io_t io() {
      return {
        [this]() -> std::optional<snapshot_t> {
          events.push_back("query");
          const int current_query = ++query_count;
          if (current_query == mutate_on_query && mutation) {
            mutation(current);
          }
          if (failed_queries.contains(current_query)) {
            return std::nullopt;
          }
          return current;
        },
        [this](snapshot_t next) {
          events.push_back("apply");
          EXPECT_TRUE(journal.has_value()) << "A durable recovery record must precede every CCD mutation";
          if (!ignore_apply) {
            current = std::move(next);
            if (reset_colors_on_apply) {
              for (auto &color : current.colors) {
                color = platf::display_config::advanced_color_state_t {platf::display_config::advanced_color_api_e::modern, false, true, false, false, false, false, 8, DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR};
              }
            }
            if (applied_mutation) {
              applied_mutation(current);
            }
          }
          return apply_ok;
        },
        [this]() {
          events.push_back("load");
          return load_result_t {load_ok, journal};
        },
        [this](const journal_t &value) {
          events.push_back("save");
          if (save_ok) {
            journal = value;
          }
          return save_ok;
        },
        [this]() {
          events.push_back("clear");
          if (clear_ok) {
            journal.reset();
          }
          return clear_ok;
        },
        [this]() -> std::optional<snapshot_t> {
          events.push_back("query_all");
          if (available_override) {
            return available_override;
          }
          auto result = current;
          for (size_t i = 0; i < catalog.paths.size(); ++i) {
            if (std::ranges::find(current.device_paths, catalog.device_paths[i]) != current.device_paths.end()) {
              continue;
            }
            auto path = catalog.paths[i];
            path.flags &= ~DISPLAYCONFIG_PATH_ACTIVE;
            path.sourceInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
            path.targetInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
            path.sourceInfo.id += 100;
            path.targetInfo.id += 1000;
            result.paths.push_back(path);
            result.device_paths.push_back(catalog.device_paths[i]);
          }
          result.colors.resize(result.paths.size());
          return result;
        },
        [this](const DISPLAYCONFIG_PATH_INFO &path, const platf::display_config::advanced_color_state_t &color) {
          events.push_back("set_color");
          if (!color_set_ok) {
            return false;
          }
          for (size_t i = 0; i < current.paths.size(); ++i) {
            const auto &candidate = current.paths[i];
            if (candidate.targetInfo.id == path.targetInfo.id && candidate.targetInfo.adapterId.HighPart == path.targetInfo.adapterId.HighPart && candidate.targetInfo.adapterId.LowPart == path.targetInfo.adapterId.LowPart) {
              current.colors[i] = color;
              return true;
            }
          }
          return false;
        },
        [this](const DISPLAYCONFIG_PATH_INFO &, std::wstring_view identity) {
          return preserved_exclusive.contains(std::wstring(identity));
        },
      };
    }
  };

  void expect_layout(const snapshot_t &snapshot, const layout_t &expected) {
    const auto observed = inspect(snapshot);
    ASSERT_TRUE(observed);
    ASSERT_EQ(observed->size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(observed->at(i).device_path, expected[i].device_path);
      EXPECT_EQ(observed->at(i).x, expected[i].x);
      EXPECT_EQ(observed->at(i).y, expected[i].y);
    }
  }

  size_t named_index(const snapshot_t &snapshot, std::wstring_view identity) {
    const auto found = std::ranges::find(snapshot.device_paths, identity);
    EXPECT_NE(found, snapshot.device_paths.end());
    return static_cast<size_t>(found - snapshot.device_paths.begin());
  }

  void expect_positions(const snapshot_t &snapshot, const layout_t &expected) {
    const auto actual = inspect(snapshot);
    ASSERT_TRUE(actual);
    ASSERT_EQ(actual->size(), expected.size());
    for (const auto &position : expected) {
      const auto index = named_index(snapshot, position.device_path);
      ASSERT_LT(index, actual->size());
      EXPECT_EQ(actual->at(index).x, position.x);
      EXPECT_EQ(actual->at(index).y, position.y);
    }
  }

  void start_exclusive(fake_io_t &fake) {
    manager_t manager(fake.io());
    ASSERT_TRUE(manager.prepare(true));
    ASSERT_TRUE(manager.bind_pending(L"virtual"));
    ASSERT_TRUE(manager.promote(L"virtual", true));
    ASSERT_EQ(fake.current.paths.size(), 1u);
    ASSERT_EQ(fake.current.device_paths[0], L"virtual");
  }
}  // namespace

TEST(PrimaryDisplayExclusiveState, FailedPromotionRemainsPendingUntilExplicitRestore) {
  exclusive_session_state_t state;
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::recover);
  const auto idle_generation = state.generation();

  state.prepared();
  EXPECT_TRUE(state.expected());
  EXPECT_GT(state.generation(), idle_generation);
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::defer);

  ASSERT_TRUE(state.can_use_identity(L"virtual"));
  state.bound(L"virtual");
  ASSERT_TRUE(state.pending_identity());
  EXPECT_EQ(*state.pending_identity(), L"virtual");
  state.bound(L"other");
  EXPECT_EQ(*state.pending_identity(), L"virtual");
  const auto pending_generation = state.generation();

  // begin_promotion() reserves ownership before platform I/O. A failed attempt deliberately has
  // no completion transition, so display notifications cannot interpret its journal as inactive.
  EXPECT_TRUE(state.begin_promotion(L"virtual"));
  EXPECT_EQ(state.generation(), pending_generation);
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::defer);
  EXPECT_FALSE(state.begin_promotion(L"other"));
  EXPECT_EQ(state.restore_action(L"other"), exclusive_restore_action_e::reject);
  EXPECT_EQ(state.generation(), pending_generation);

  ASSERT_TRUE(state.promotion_succeeded(L"virtual"));
  ASSERT_TRUE(state.active_identity());
  EXPECT_FALSE(state.pending_identity());
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::reconcile);
  EXPECT_GT(state.generation(), pending_generation);

  EXPECT_EQ(state.restore_action(L"virtual"), exclusive_restore_action_e::disarm_before);
  const auto active_generation = state.generation();
  state.disarm();
  EXPECT_FALSE(state.expected());
  EXPECT_FALSE(state.active_identity());
  EXPECT_GT(state.generation(), active_generation);
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::recover);
}

TEST(PrimaryDisplayExclusiveState, UnboundSessionDisarmsOnlyAfterAcceptedRestore) {
  exclusive_session_state_t state;
  state.prepared();
  const auto prepared_generation = state.generation();

  EXPECT_EQ(state.restore_action(L"virtual"), exclusive_restore_action_e::disarm_after_success);
  EXPECT_TRUE(state.expected());
  EXPECT_EQ(state.generation(), prepared_generation);

  const auto action = state.restore_action(L"virtual");
  state.restore_finished(action, false);
  EXPECT_TRUE(state.expected());
  EXPECT_EQ(state.generation(), prepared_generation);

  state.restore_finished(action, true);
  EXPECT_FALSE(state.expected());
  EXPECT_GT(state.generation(), prepared_generation);
}

TEST(PrimaryDisplayExclusiveState, FailedPostApplyReadbackCannotTriggerInactiveRecoveryBetweenRetries) {
  fake_io_t fake;
  manager_t manager(fake.io());
  exclusive_session_state_t state;

  ASSERT_TRUE(manager.prepare(true));
  state.prepared();
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  state.bound(L"virtual");
  ASSERT_TRUE(state.begin_promotion(L"virtual"));

  // prepare and bind consume queries 1-2; promotion queries the baseline and latest state at 3-4,
  // applies the exclusive topology, then receives a transient failure for readback 5.
  fake.failed_queries.insert(5);
  EXPECT_FALSE(manager.promote(L"virtual", true));
  EXPECT_EQ(fake.current.paths.size(), 1u);
  EXPECT_EQ(fake.current.device_paths.front(), L"virtual");
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::defer);

  int inactive_recoveries = 0;
  if (state.reconcile_action() == exclusive_reconcile_action_e::recover) {
    ++inactive_recoveries;
    manager.recover_inactive_exclusive();
  }
  EXPECT_EQ(inactive_recoveries, 0);
  EXPECT_EQ(fake.current.paths.size(), 1u);

  EXPECT_TRUE(state.begin_promotion(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_TRUE(state.promotion_succeeded(L"virtual"));
  EXPECT_EQ(state.reconcile_action(), exclusive_reconcile_action_e::reconcile);
}

TEST(PrimaryDisplay, PromotionTranslatesAllPositionsAndJournalsBeforeApply) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"VIRTUAL"));
  expect_layout(fake.current, {{L"physical-left", -3840, 320}, {L"physical-primary", -1920, 200}, {L"virtual", 0, 0}});
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(fake.journal->original_primary, L"physical-primary");
  EXPECT_EQ(fake.events, (std::vector<std::string> {"load", "query", "save", "query", "apply", "query"}));
  EXPECT_EQ(fake.current.modes[0].sourceMode.width, 1920u);
  EXPECT_EQ(fake.current.paths[0].targetInfo.refreshRate.Numerator, 60000u);
}

TEST(PrimaryDisplay, RestorePreservesCurrentResolutionAndRefresh) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.current.modes[4].sourceMode.width = 3840;
  fake.current.paths[2].targetInfo.refreshRate = {90, 1};
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_layout(fake.current, baseline);
  EXPECT_EQ(fake.current.modes[4].sourceMode.width, 3840u);
  EXPECT_EQ(fake.current.paths[2].targetInfo.refreshRate.Numerator, 90u);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, RecoveryUsesStablePathsAfterSourceAndTargetRenumbering) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  // Rebuilding in another path order also changes every path's source/target IDs and
  // mode indices, as a driver restart can do. Persisted device paths remain the key.
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"physical-primary", -1920, 200}, {L"physical-left", -3840, 320}});
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.restore());
  expect_layout(fake.current, {{L"virtual", 1920, -200}, {L"physical-primary", 0, 0}, {L"physical-left", -1920, 120}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, RepeatedPromotionKeepsFirstBaselineAcrossResize) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  const auto first_record = serialize(*fake.journal);
  fake.current.modes[4].sourceMode.width = 3840;
  fake.events.clear();
  ASSERT_TRUE(manager.promote(L"virtual"));
  EXPECT_EQ(serialize(*fake.journal), first_record);
  EXPECT_EQ(fake.events, (std::vector<std::string> {"load", "query"}));
}

TEST(PrimaryDisplay, FailedJournalWriteNeverAppliesDisplayChange) {
  fake_io_t fake;
  fake.save_ok = false;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  expect_layout(fake.current, baseline);
}

TEST(PrimaryDisplay, RaceAfterSavingJournalDoesNotOverwriteUserLayout) {
  fake_io_t fake;
  fake.mutate_on_query = 2;
  fake.mutation = [](snapshot_t &snapshot) {
    snapshot.modes[0].sourceMode.position.y += 50;
  };
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  EXPECT_TRUE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, AcceptedButUnappliedChangeFailsReadback) {
  fake_io_t fake;
  fake.ignore_apply = true;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  ASSERT_TRUE(fake.journal);
  EXPECT_TRUE(manager.restore());
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, FailedApiWithPartialChangeCanBeRecoveredByNewManager) {
  fake_io_t fake;
  fake.apply_ok = false;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  ASSERT_TRUE(fake.journal);
  fake.apply_ok = true;
  manager_t restarted(fake.io());
  EXPECT_TRUE(restarted.restore());
  expect_layout(fake.current, baseline);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, FailedRestoreReadbackKeepsRecoveryRecord) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.ignore_apply = true;
  EXPECT_FALSE(manager.restore());
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplay, RestoredStateCanRetryJournalCleanupWithoutAnotherMutation) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.clear_ok = false;
  EXPECT_FALSE(manager.restore());
  expect_layout(fake.current, baseline);
  ASSERT_TRUE(fake.journal);
  fake.clear_ok = true;
  fake.events.clear();
  EXPECT_TRUE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, RestoreRefusesDifferentSessionIdentity) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.events.clear();
  EXPECT_FALSE(manager.restore(L"different-virtual"));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplay, UserPositionEditIsPreservedAndRecoveryRetained) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.current.modes[0].sourceMode.position.y += 200;
  fake.events.clear();
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplay, MissingPhysicalMonitorDefersRecovery) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.events.clear();
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplay, MissingVirtualMonitorClearsOnlyVerifiedOriginalLayout) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.current = make_snapshot({{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  fake.events.clear();
  EXPECT_TRUE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, MissingVirtualMonitorRestoresKnownPhysicalPrimaryAfterAutomaticRebase) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  fake.current = make_snapshot({{L"physical-left", 0, 0}, {L"physical-primary", 1920, -120}});
  EXPECT_TRUE(manager.restore());
  expect_layout(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, NewlyAttachedMonitorIsNotOverwrittenByOldTransaction) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  auto layout = *inspect(fake.current);
  layout.push_back({L"new-monitor", 3840, 0});
  fake.current = make_snapshot(layout);
  EXPECT_FALSE(manager.restore());
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplay, MalformedJournalBlocksMutation) {
  fake_io_t fake;
  fake.load_ok = false;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, OriginalPrimaryAlreadyRequestedDoesNotCreateTransaction) {
  fake_io_t fake;
  manager_t manager(fake.io());
  EXPECT_TRUE(manager.promote(L"physical-primary"));
  EXPECT_FALSE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, MissingTargetDoesNotChangeDisplays) {
  fake_io_t fake;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"missing"));
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, LegacyAndVirtualAwareModeIndicesAreBothHandled) {
  for (const bool aware : {false, true}) {
    fake_io_t fake;
    fake.current = make_snapshot(baseline, aware);
    manager_t manager(fake.io());
    EXPECT_TRUE(manager.promote(L"virtual"));
    EXPECT_TRUE(manager.restore());
    expect_layout(fake.current, baseline);
  }
}

TEST(PrimaryDisplay, ClonedSourcesAndDuplicateDevicePathsAreRejected) {
  auto snapshot = make_snapshot(baseline);
  snapshot.paths[1].sourceInfo = snapshot.paths[0].sourceInfo;
  EXPECT_FALSE(inspect(snapshot));
  snapshot = make_snapshot(baseline);
  snapshot.device_paths[1] = L"PHYSICAL-LEFT";
  EXPECT_FALSE(inspect(snapshot));
}

TEST(PrimaryDisplay, InvalidModeIndicesTypesAndAdaptersAreRejected) {
  auto snapshot = make_snapshot(baseline);
  snapshot.paths[1].sourceInfo.sourceModeInfoIdx = DISPLAYCONFIG_PATH_SOURCE_MODE_IDX_INVALID;
  EXPECT_FALSE(inspect(snapshot));
  snapshot = make_snapshot(baseline);
  snapshot.modes[0].infoType = DISPLAYCONFIG_MODE_INFO_TYPE_TARGET;
  EXPECT_FALSE(inspect(snapshot));
  snapshot = make_snapshot(baseline);
  snapshot.modes[0].adapterId.LowPart = 9;
  EXPECT_FALSE(inspect(snapshot));
  snapshot = make_snapshot(baseline);
  snapshot.paths[1].targetInfo.targetModeInfoIdx = DISPLAYCONFIG_PATH_TARGET_MODE_IDX_INVALID;
  EXPECT_FALSE(inspect(snapshot));
}

TEST(PrimaryDisplay, CoordinateTranslationOverflowDoesNotApply) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"far-left", std::numeric_limits<LONG>::min(), 0}, {L"virtual", 100, 1080}});
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.promote(L"virtual"));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, JournalRoundTripAndCorruptCoordinates) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  const auto encoded = serialize(*fake.journal);
  ASSERT_TRUE(deserialize(encoded));
  EXPECT_EQ(serialize(*deserialize(encoded)), encoded);
  auto value = nlohmann::json::parse(encoded);
  value["promoted"][0]["x"] = 999;
  EXPECT_FALSE(deserialize(value.dump()));
  value = nlohmann::json::parse(encoded);
  value["original"][0]["x"] = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(deserialize(value.dump()));
  value = nlohmann::json::parse(encoded);
  value["version"] = 2;
  EXPECT_FALSE(deserialize(value.dump()));
  EXPECT_FALSE(deserialize("{"));
}

TEST(PrimaryDisplay, PrepareCapturesPhysicalPrimaryBeforeRememberedVirtualPromotion) {
  fake_io_t fake;
  const layout_t physical {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}};
  fake.current = make_snapshot(physical);
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  ASSERT_TRUE(fake.journal);
  ASSERT_TRUE(fake.journal->prepared);
  ASSERT_TRUE(deserialize(serialize(*fake.journal)));
  // AddVirtualDisplay itself can restore this old primary arrangement before bind.
  fake.current = make_snapshot({{L"physical-left", -3840, 320}, {L"physical-primary", -1920, 200}, {L"virtual", 0, 0}});
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(fake.journal);
  EXPECT_FALSE(fake.journal->prepared);
  EXPECT_EQ(fake.journal->original_primary, L"physical-primary");
  ASSERT_TRUE(manager.promote(L"virtual"));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_layout(fake.current, baseline);
}

TEST(PrimaryDisplay, BoundBaselineSurvivesModeChangeBeforePromotion) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  const auto first_record = serialize(*fake.journal);
  fake.current.modes[2].sourceMode.width = 3840;
  ASSERT_TRUE(manager.promote(L"virtual"));
  EXPECT_EQ(serialize(*fake.journal), first_record);
  ASSERT_TRUE(manager.restore());
  EXPECT_EQ(fake.current.modes[2].sourceMode.width, 3840u);
  expect_layout(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
}

TEST(PrimaryDisplay, PreparedRecoveryWaitsForUnknownTargetToDisappear) {
  fake_io_t fake;
  const layout_t physical {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}};
  fake.current = make_snapshot(physical);
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  fake.current = make_snapshot({{L"physical-left", -3840, 320}, {L"physical-primary", -1920, 200}, {L"unknown", 0, 0}});
  manager_t restarted(fake.io());
  fake.events.clear();
  EXPECT_FALSE(restarted.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  ASSERT_TRUE(fake.journal);
  // Driver cleanup removed the orphan; Windows chose another known physical primary.
  fake.current = make_snapshot({{L"physical-left", 0, 0}, {L"physical-primary", 1920, -120}});
  EXPECT_TRUE(restarted.restore());
  expect_layout(fake.current, physical);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, PreparedRecoveryWithExactIdentityCanRestoreImmediately) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  fake.current = make_snapshot({{L"physical-primary", -1920, 0}, {L"virtual", 0, 0}});
  EXPECT_TRUE(manager.restore(L"virtual"));
  expect_layout(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, PreparedRetirementCanRecoverAfterExactTargetAlreadyDisappeared) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  EXPECT_TRUE(manager.restore(L"virtual-already-gone"));
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplay, BindDoesNotAdoptUnrelatedPhysicalArrangement) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  fake.current = make_snapshot({{L"physical-left", -1920, 400}, {L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  EXPECT_FALSE(manager.bind_pending(L"virtual"));
  EXPECT_TRUE(fake.journal->prepared);
}

TEST(PrimaryDisplay, BindIsIdempotentForRetainedTargetAndCompletedJournal) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  const auto record = serialize(*fake.journal);
  EXPECT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_EQ(serialize(*fake.journal), record);
  EXPECT_FALSE(manager.bind_pending(L"unrelated"));
  EXPECT_TRUE(manager.restore(L"virtual"));
  EXPECT_TRUE(manager.bind_pending(L"virtual"));
}

TEST(PrimaryDisplay, HeadlessPreparationNeedsNoOriginalPrimaryRecord) {
  fake_io_t fake;
  fake.current = {};
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare());
  EXPECT_FALSE(fake.journal);
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  EXPECT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_TRUE(manager.promote(L"virtual"));
  EXPECT_TRUE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplay, IndependentOwnersCannotRecoverTheSameLiveJournal) {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() / ("apollo-primary-owner-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(nonce) + ".lock");
  {
    auto first = std::make_unique<ownership_t>();
    ownership_t second;
    ASSERT_TRUE(first->acquire(path));
    EXPECT_TRUE(first->acquire(path));
    EXPECT_FALSE(first->acquire(path.wstring() + L".other-config"));
    EXPECT_FALSE(second.acquire(path));
    first.reset();
    EXPECT_TRUE(second.acquire(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(PrimaryDisplayExclusive, ReactivatesInactivePhysicalOutputsAndRetainsVirtualForReconnect) {
  fake_io_t fake;
  start_exclusive(fake);
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_started);
  EXPECT_EQ(nlohmann::json::parse(serialize(*fake.journal))["version"], 3);
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(fake.journal);
  // The retained virtual output stays active; reconnect captures a new complete baseline.
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_EQ(fake.current.paths.size(), 1u);
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
}

TEST(PrimaryDisplayExclusive, PauseDetachesVirtualTargetAndReconnectRestoresItsExactMode) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current.modes[0].sourceMode.width = 2560;
  fake.current.modes[0].sourceMode.height = 1440;
  fake.current.paths[0].targetInfo.refreshRate = {59941, 1000};
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;

  ASSERT_TRUE(manager.pause(L"virtual", retained));
  ASSERT_TRUE(retained);
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
  const auto paused = retained;
  const auto applies = std::ranges::count(fake.events, "apply");
  EXPECT_TRUE(manager.pause(L"virtual", retained));
  EXPECT_EQ(retained, paused);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);

  ASSERT_TRUE(manager.reactivate(retained, true));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive && fake.journal->exclusive_started);
  EXPECT_EQ(fake.journal->original_topology->paths.size(), 2u);
  const auto index = named_index(fake.current, L"virtual");
  const auto &mode = fake.current.modes[fake.current.paths[index].sourceInfo.sourceModeInfoIdx].sourceMode;
  EXPECT_EQ(mode.width, 2560u);
  EXPECT_EQ(mode.height, 1440u);
  EXPECT_EQ(fake.current.paths[index].targetInfo.refreshRate.Numerator, 59941u);
  EXPECT_EQ(fake.current.paths[index].targetInfo.refreshRate.Denominator, 1000u);
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  retained.reset();
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
}

TEST(PrimaryDisplayExclusive, PausedReconnectCanSwitchToPrimaryOnlyAndRollback) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  ASSERT_TRUE(manager.reactivate(retained, false));
  EXPECT_FALSE(fake.journal);
  ASSERT_TRUE(manager.promote(L"virtual", false));
  ASSERT_TRUE(fake.journal && !fake.journal->exclusive);

  // A failed policy-toggle resume can leave its ordinary primary-only journal pending.
  // Pause first completes that exact recovery before detaching with full color protection.
  retained.reset();
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
}

TEST(PrimaryDisplayExclusive, PausedReconnectPreservesPhysicalEditsMadeWhileDisconnected) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  const auto physical = named_index(fake.current, L"physical-left");
  fake.current.modes[fake.current.paths[physical].sourceInfo.sourceModeInfoIdx].sourceMode.position.y = 240;
  fake.current.colors[physical]->hdr_user_enabled = true;
  fake.current.colors[physical]->advanced_color_active = true;
  fake.current.colors[physical]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  const auto expected = fake.current;
  fake.reset_colors_on_apply = true;

  ASSERT_TRUE(manager.reactivate(retained, true));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  retained.reset();
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, *inspect(expected));
  const auto restored = named_index(fake.current, L"physical-left");
  EXPECT_TRUE(fake.current.colors[restored]->hdr_user_enabled);
  EXPECT_EQ(fake.current.colors[restored]->active_mode, DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR);
}

TEST(PrimaryDisplayExclusive, PausePreservesApprovedArOutputWhileDetachingVirtual) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  ASSERT_TRUE(manager.reactivate(retained, true));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
}

TEST(PrimaryDisplayExclusive, PauseNeverDetachesTheOnlyAvailableOutput) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.available_override = fake.current;
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  EXPECT_FALSE(manager.pause(L"virtual", retained));
  EXPECT_TRUE(retained);
  EXPECT_TRUE(fake.journal);
  expect_positions(fake.current, {{L"virtual", 0, 0}});

  fake.journal.reset();
  EXPECT_TRUE(manager.pause(L"virtual", retained));
  EXPECT_TRUE(manager.reactivate(retained, true));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
}

TEST(PrimaryDisplayExclusive, PauseRestoresAvailableSurvivorsAndKeepsMissingOutputJournal) {
  fake_io_t fake;
  start_exclusive(fake);
  auto available = fake.io().query_all();
  ASSERT_TRUE(available);
  available->paths[named_index(*available, L"physical-left")].targetInfo.targetAvailable = FALSE;
  fake.available_override = *available;
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  EXPECT_FALSE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  EXPECT_EQ(fake.journal->pending_restore->paths.size(), 1u);

  fake.available_override.reset();
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, FailedPauseColorRestoreKeepsSnapshotAndRecoveryUntilRetry) {
  fake_io_t fake;
  fake.current.colors[0]->hdr_user_enabled = true;
  fake.current.colors[0]->advanced_color_active = true;
  fake.current.colors[0]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  start_exclusive(fake);
  fake.reset_colors_on_apply = true;
  fake.color_set_ok = false;
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  EXPECT_FALSE(manager.pause(L"virtual", retained));
  ASSERT_TRUE(retained);
  const auto saved = retained;
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.current.device_paths, L"virtual"), 0);
  fake.color_set_ok = true;
  EXPECT_TRUE(manager.pause(L"virtual", retained));
  EXPECT_EQ(retained, saved);
  EXPECT_FALSE(fake.journal);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-left")]->hdr_user_enabled);
}

TEST(PrimaryDisplayExclusive, ReactivationRefusesMissingAmbiguousOrUnjournaledTarget) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  const auto applies = std::ranges::count(fake.events, "apply");
  fake.available_override = fake.current;
  EXPECT_FALSE(manager.reactivate(retained, true));
  EXPECT_FALSE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);

  fake.available_override.reset();
  auto available = fake.io().query_all();
  ASSERT_TRUE(available);
  const auto index = named_index(*available, L"virtual");
  auto duplicate = available->paths[index];
  ++duplicate.targetInfo.id;
  available->paths.push_back(duplicate);
  available->device_paths.push_back(L"virtual");
  fake.available_override = *available;
  EXPECT_FALSE(manager.reactivate(retained, true));
  EXPECT_FALSE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);

  fake.available_override.reset();
  fake.save_ok = false;
  EXPECT_FALSE(manager.reactivate(retained, true));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);
}

TEST(PrimaryDisplayExclusive, FailedReactivationRetainsColorRecoveryAndCanPauseAgain) {
  fake_io_t fake;
  fake.current.colors[1]->hdr_user_enabled = true;
  fake.current.colors[1]->advanced_color_active = true;
  fake.current.colors[1]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  fake.reset_colors_on_apply = true;
  fake.color_set_ok = false;
  EXPECT_FALSE(manager.reactivate(retained, true));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_started);
  EXPECT_EQ(std::ranges::count(fake.current.device_paths, L"virtual"), 1);
  fake.color_set_ok = true;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  EXPECT_EQ(std::ranges::count(fake.current.device_paths, L"virtual"), 0);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-primary")]->hdr_user_enabled);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, ReactivationRetriesColorVerificationAfterAnAcceptedApply) {
  fake_io_t fake;
  fake.current.colors[1]->hdr_user_enabled = true;
  fake.current.colors[1]->advanced_color_active = true;
  fake.current.colors[1]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  fake.reset_colors_on_apply = true;
  fake.color_set_ok = false;
  EXPECT_FALSE(manager.reactivate(retained, true));
  const auto applies = std::ranges::count(fake.events, "apply");
  fake.color_set_ok = true;
  EXPECT_TRUE(manager.reactivate(retained, true));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-primary")]->hdr_user_enabled);
  EXPECT_TRUE(manager.promote(L"virtual", true));
}

TEST(PrimaryDisplayExclusive, PauseRefusesDifferentRetainedIdentity) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  const auto applies = std::ranges::count(fake.events, "apply");
  EXPECT_FALSE(manager.pause(L"physical-primary", retained));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);
}

TEST(PrimaryDisplayExclusive, PausedTargetCanResumeAfterAllPhysicalOutputsDisappear) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  fake.current = {};
  for (auto &path : fake.catalog.paths) {
    path.targetInfo.targetAvailable = FALSE;
  }
  fake.catalog.paths[named_index(fake.catalog, L"virtual")].targetInfo.targetAvailable = TRUE;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  ASSERT_TRUE(manager.reactivate(retained, true));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  ASSERT_TRUE(fake.journal);
  EXPECT_TRUE(fake.journal->original_topology->paths.empty());
  EXPECT_TRUE(fake.journal->before_exclusive->paths.empty());
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal) << "An empty physical baseline must survive crash recovery serialization";
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, HeadlessResumeJournalRestoresPhysicalOutputThatReturns) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  fake.current = {};
  ASSERT_TRUE(manager.reactivate(retained, true));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"physical-primary", 1920, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", -1920, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, FailedReactivationModeReadbackCanStillDetachItsExactTarget) {
  fake_io_t fake;
  start_exclusive(fake);
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  fake.applied_mutation = [](snapshot_t &snapshot) {
    const auto index = named_index(snapshot, L"virtual");
    snapshot.modes[snapshot.paths[index].sourceInfo.sourceModeInfoIdx].sourceMode.width = 1280;
  };
  EXPECT_FALSE(manager.reactivate(retained, true));
  ASSERT_TRUE(fake.journal);
  fake.applied_mutation = {};
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  ASSERT_TRUE(manager.reactivate(retained, true));
  const auto index = named_index(fake.current, L"virtual");
  EXPECT_EQ(fake.current.modes[fake.current.paths[index].sourceInfo.sourceModeInfoIdx].sourceMode.width, 1920u);
}

TEST(PrimaryDisplayLocalExclusive, PauseRetainsSelectedSinkAndItsCurrentHardwareMode) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  const auto sink = named_index(fake.current, L"physical-left");
  fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode.width = 2560;
  fake.current.colors[sink]->hdr_user_enabled = true;
  fake.current.colors[sink]->advanced_color_active = true;
  fake.current.colors[sink]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-left", -2560, 120}, {L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
  ASSERT_TRUE(manager.reactivate(retained, true));
  ASSERT_TRUE(manager.is_local_exclusive(L"virtual"));
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(fake.journal->local_sink, L"physical-left");
  EXPECT_EQ(fake.journal->exclusive_preserved, (std::vector<std::wstring> {L"physical-left"}));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -2560, 0}});
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-left")]->hdr_user_enabled);
}

TEST(PrimaryDisplayLocalExclusive, ExtendedLocalPauseUsesCurrentPhysicalLayoutAndExplicitSink) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained, L"physical-left"));
  EXPECT_FALSE(fake.journal);
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  // The separate extended-local rectangle owner can finish its restoration after detach.
  const auto sink = named_index(fake.current, L"physical-left");
  fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode.position.y = 300;
  ASSERT_TRUE(manager.reactivate(retained, false));
  EXPECT_FALSE(fake.journal);
  expect_positions(fake.current, {{L"physical-left", -1920, 300}, {L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  EXPECT_FALSE(manager.pause(L"virtual", retained, L"physical-primary"));
}

TEST(PrimaryDisplayLocalExclusive, MissingSinkPauseRestoresOrdinaryOutputsAndDefersLocalResume) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.catalog.paths[named_index(fake.catalog, L"physical-left")].targetInfo.targetAvailable = FALSE;
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
  EXPECT_FALSE(manager.reactivate(retained, true));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
}

TEST(PrimaryDisplayLocalExclusive, ExtendedPauseDetachesSourceAfterItsPhysicalSinkDisappears) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  fake.catalog.paths[named_index(fake.catalog, L"physical-left")].targetInfo.targetAvailable = FALSE;
  manager_t manager(fake.io());
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained, L"physical-left"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
  EXPECT_FALSE(manager.reactivate(retained, false));
}

TEST(PrimaryDisplayLocalExclusive, SoleVirtualOutputDoesNotResumeWithoutItsRetainedSink) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  fake.current = make_snapshot({{L"physical-left", 0, 0}, {L"virtual", 1920, 0}});
  fake.catalog = fake.current;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.catalog.paths[0].targetInfo.targetAvailable = FALSE;
  platf::primary_display::retained_display_ptr retained;
  ASSERT_TRUE(manager.pause(L"virtual", retained));
  EXPECT_FALSE(fake.journal);
  EXPECT_FALSE(manager.reactivate(retained, true));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"physical-left", 1920, 0}});
  fake.catalog.paths[0].targetInfo.targetAvailable = TRUE;
  ASSERT_TRUE(manager.reactivate(retained, true));
  EXPECT_TRUE(manager.is_local_exclusive(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
}

TEST(PrimaryDisplayExclusive, KeepsApprovedArOutputActiveAndDisablesOrdinaryMonitors) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_EQ(fake.current.paths.size(), 2u);
  expect_positions(fake.current, {
    {L"virtual", 0, 0},
    {L"physical-left", -1920, 0},
  });
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_topology);
  expect_positions(*fake.journal->exclusive_topology, {
    {L"virtual", 0, 0},
    {L"physical-left", -1920, 0},
  });

  const auto virtual_index = named_index(fake.current, L"virtual");
  fake.current.modes[fake.current.paths[virtual_index].sourceInfo.sourceModeInfoIdx].sourceMode.width = 3840;
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_EQ(fake.current.paths.size(), 2u);

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(fake.journal);

  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
}

TEST(PrimaryDisplayExclusive, PacksOnlyAvailablePreservedArOutputsBesideVirtualDisplay) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  fake.preserved_exclusive.insert(L"physical-primary");
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  fake.available_override = fake.current;
  fake.available_override->paths[named_index(*fake.available_override, L"physical-primary")].targetInfo.targetAvailable = FALSE;

  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {
    {L"virtual", 0, 0},
    {L"physical-left", -1920, 0},
  });
}

TEST(PrimaryDisplayExclusive, RememberedVirtualOnlyTopologyReactivatesApprovedArOutput) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.prepare(true));
  // Windows may restore an old virtual-only arrangement as part of adding the virtual target.
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_started);
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
  EXPECT_EQ(std::ranges::count(fake.current.device_paths, L"physical-primary"), 0);

  // Ownership is the durable launch-time allowlist, not a later mutable classification.
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal);
  fake.preserved_exclusive.clear();
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, 0},
  });
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, RetriesPreservedArColorWithoutReapplyingTopology) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  auto &ar_color = *fake.current.colors[named_index(fake.current, L"physical-left")];
  ar_color.hdr_user_enabled = true;
  ar_color.advanced_color_active = true;
  ar_color.active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  fake.reset_colors_on_apply = true;
  fake.color_set_ok = false;
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_FALSE(manager.promote(L"virtual", true));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_topology);
  EXPECT_FALSE(fake.current.colors[named_index(fake.current, L"physical-left")]->hdr_user_enabled);

  const auto applies = std::ranges::count(fake.events, "apply");
  fake.color_set_ok = true;
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), applies);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-left")]->hdr_user_enabled);
  ASSERT_TRUE(manager.restore(L"virtual"));
}

TEST(PrimaryDisplayExclusive, KeepsApprovedArOutputThatConnectsDuringStreamActiveOnRestore) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-ar", 1920, 0},
    {L"virtual", 3840, 0},
  });
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, HotpluggedArRestoreIntentSurvivesColdRetry) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  for (size_t i = 0; i < fake.catalog.paths.size(); ++i) {
    if (fake.catalog.device_paths[i] != L"virtual") {
      fake.catalog.paths[i].targetInfo.targetAvailable = FALSE;
    }
  }
  manager_t manager(fake.io());
  fake.events.clear();

  EXPECT_FALSE(manager.restore(L"virtual"));
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  EXPECT_NE(
    std::ranges::find(fake.journal->pending_restore->device_paths, L"hotplugged-ar"),
    fake.journal->pending_restore->device_paths.end()
  );
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal);

  fake.preserved_exclusive.clear();
  for (auto &path : fake.catalog.paths) {
    path.targetInfo.targetAvailable = TRUE;
  }
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-ar", 1920, 0},
    {L"virtual", 3840, 0},
  });
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, KeepsApprovedHotpluggedArWhenVirtualDisappearedBeforeRestore) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = make_snapshot({{L"hotplugged-ar", 0, 0}});
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-ar", 1920, 0},
  });
  EXPECT_EQ(std::ranges::count(fake.current.device_paths, L"virtual"), 0);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, WarmResumeKeepsDurablyAdoptedHotpluggedArActive) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  for (size_t i = 0; i < fake.catalog.paths.size(); ++i) {
    if (fake.catalog.device_paths[i] != L"virtual") {
      fake.catalog.paths[i].targetInfo.targetAvailable = FALSE;
    }
  }
  manager_t manager(fake.io());

  ASSERT_FALSE(manager.restore(L"virtual"));
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);

  // Resume owns the journaled hotplug even if the live policy cannot be consulted anymore.
  fake.preserved_exclusive.clear();
  fake.events.clear();
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 1);
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_topology);
  EXPECT_TRUE(deserialize(serialize(*fake.journal)));

  for (auto &path : fake.catalog.paths) {
    path.targetInfo.targetAvailable = TRUE;
  }
  ASSERT_TRUE(manager.restore(L"virtual"));
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, RepeatedPromotionAndLiveResizeKeepOriginalTopology) {
  fake_io_t fake;
  start_exclusive(fake);
  ASSERT_TRUE(fake.journal && fake.journal->original_topology);
  const auto original_virtual = named_index(*fake.journal->original_topology, L"virtual");
  const auto original_source = fake.journal->original_topology->paths[original_virtual].sourceInfo.sourceModeInfoIdx;
  const auto original_width = fake.journal->original_topology->modes[original_source].sourceMode.width;
  fake.current.modes[0].sourceMode.width = 3840;
  fake.current.paths[0].targetInfo.refreshRate = {90, 1};
  manager_t manager(fake.io());
  EXPECT_TRUE(manager.promote(L"virtual", true));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_topology);
  EXPECT_EQ(fake.journal->original_topology->modes[original_source].sourceMode.width, original_width);
  const auto exclusive_virtual = named_index(*fake.journal->exclusive_topology, L"virtual");
  const auto exclusive_source = fake.journal->exclusive_topology->paths[exclusive_virtual].sourceInfo.sourceModeInfoIdx;
  EXPECT_EQ(fake.journal->exclusive_topology->modes[exclusive_source].sourceMode.width, 3840u);
  EXPECT_EQ(fake.journal->exclusive_topology->paths[exclusive_virtual].targetInfo.refreshRate.Numerator, 90u);
  EXPECT_TRUE(manager.restore());
  const auto index = named_index(fake.current, L"virtual");
  const auto source = fake.current.paths[index].sourceInfo.sourceModeInfoIdx;
  EXPECT_EQ(fake.current.modes[source].sourceMode.width, 3840u);
  EXPECT_EQ(fake.current.paths[index].targetInfo.refreshRate.Numerator, 90u);
}

TEST(PrimaryDisplayExclusive, ResizedLeftVirtualMovesBesideRestoredPhysicalInsteadOfOverlapping) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"virtual", -1920, 0}});
  start_exclusive(fake);
  fake.current.modes[0].sourceMode.width = 3840;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.restore());
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
}

TEST(PrimaryDisplayExclusive, CrashWithNoActiveOutputsRestoresAvailablePhysicalDisplays) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = {};
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.restore());
  expect_positions(fake.current, {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, UnavailablePrimaryStillLightsOtherPhysicalAndKeepsVirtualAndJournal) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.catalog.paths[1].targetInfo.targetAvailable = FALSE;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  ASSERT_EQ(fake.current.paths.size(), 2u);
  expect_positions(fake.current, {{L"physical-left", 0, 0}, {L"virtual", 1920, 0}});
  fake.catalog.paths[1].targetInfo.targetAvailable = TRUE;
  // Windows may reactivate the returned original primary before the next recovery attempt.
  fake.current = make_snapshot(baseline);
  fake.current.modes[4].sourceMode.width = 3840;
  ASSERT_TRUE(manager.restore());
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, MissingMiddleOutputPacksSurvivorsUntilExactRestoreCanFinish) {
  const layout_t saved {
    {L"physical-left", -1920, 0},
    {L"physical-middle", 0, 0},
    {L"physical-right", 1920, 0},
    {L"virtual", 3840, 0},
  };
  fake_io_t fake;
  fake.current = make_snapshot(saved);
  fake.catalog = fake.current;
  start_exclusive(fake);
  fake.catalog.paths[named_index(fake.catalog, L"physical-middle")].targetInfo.targetAvailable = FALSE;
  manager_t manager(fake.io());

  EXPECT_FALSE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", 0, 0},
    {L"physical-right", 1920, 0},
    {L"virtual", 3840, 0},
  });
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  expect_positions(*fake.journal->pending_restore, {
    {L"physical-left", 0, 0},
    {L"physical-right", 1920, 0},
    {L"virtual", 3840, 0},
  });
  expect_positions(*fake.journal->original_topology, saved);
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal);

  fake.catalog.paths[named_index(fake.catalog, L"physical-middle")].targetInfo.targetAvailable = TRUE;
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.recover_inactive_exclusive());
  expect_positions(fake.current, saved);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, PreparedRecoveryPacksAroundUnavailableMiddleOutput) {
  const layout_t saved {
    {L"physical-left", -1920, 0},
    {L"physical-middle", 0, 0},
    {L"physical-right", 1920, 0},
  };
  fake_io_t fake;
  fake.current = make_snapshot(saved);
  fake.catalog = fake.current;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  fake.current = {};
  fake.catalog.paths[named_index(fake.catalog, L"physical-middle")].targetInfo.targetAvailable = FALSE;

  EXPECT_FALSE(manager.restore());
  expect_positions(fake.current, {
    {L"physical-left", 0, 0},
    {L"physical-right", 1920, 0},
  });
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  expect_positions(*fake.journal->pending_restore, {
    {L"physical-left", 0, 0},
    {L"physical-right", 1920, 0},
  });
  expect_positions(*fake.journal->original_topology, saved);
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal);

  fake.catalog.paths[named_index(fake.catalog, L"physical-middle")].targetInfo.targetAvailable = TRUE;
  manager_t restarted(fake.io());
  ASSERT_TRUE(restarted.restore());
  expect_positions(fake.current, saved);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, UnavailableAllPhysicalOutputsNeverDisablesRemainingVirtual) {
  fake_io_t fake;
  start_exclusive(fake);
  for (auto &path : fake.catalog.paths) {
    path.targetInfo.targetAvailable = FALSE;
  }
  fake.events.clear();
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(fake.current.device_paths, (std::vector<std::wstring> {L"virtual"}));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayExclusive, FailedRestoreApplyOrReadbackRetainsDurableRecovery) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.ignore_apply = true;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  fake.ignore_apply = false;
  fake.apply_ok = false;
  EXPECT_FALSE(manager.restore());
  EXPECT_TRUE(fake.journal);
  fake.apply_ok = true;
  EXPECT_TRUE(manager.restore());
  expect_positions(fake.current, baseline);
}

TEST(PrimaryDisplayExclusive, FailedIntentOrRecoveryJournalWritePreventsTopologyMutation) {
  fake_io_t fake;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  fake.save_ok = false;
  fake.events.clear();
  EXPECT_FALSE(manager.promote(L"virtual", true));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  fake.save_ok = true;
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.save_ok = false;
  fake.events.clear();
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayExclusive, CapturesAndRestoresHdrAfterReactivationAndRetriesColorFailure) {
  fake_io_t fake;
  auto &hdr = *fake.current.colors[1];
  hdr.hdr_user_enabled = true;
  hdr.advanced_color_active = true;
  hdr.active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  fake.reset_colors_on_apply = true;
  start_exclusive(fake);
  fake.color_set_ok = false;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(fake.current.paths.size(), 3u);
  EXPECT_TRUE(fake.journal);
  fake.color_set_ok = true;
  ASSERT_TRUE(manager.restore());
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-primary")]->hdr_user_enabled);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, UnknownOriginalColorStateFailsBeforeDisablingPhysicalDisplays) {
  fake_io_t fake;
  fake.current.colors[0] = std::nullopt;
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.prepare(true));
  EXPECT_FALSE(fake.journal);
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplayExclusive, TopologyTransitionsPreserveCurrentVirtualHdrState) {
  fake_io_t fake;
  auto &hdr = *fake.current.colors[2];
  hdr.hdr_user_enabled = true;
  hdr.advanced_color_active = true;
  hdr.active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  fake.reset_colors_on_apply = true;
  start_exclusive(fake);
  EXPECT_TRUE(fake.current.colors[0]->hdr_user_enabled);
  manager_t manager(fake.io());
  fake.color_set_ok = false;
  EXPECT_FALSE(manager.restore());
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  const auto virtual_index = named_index(fake.current, L"virtual");
  EXPECT_FALSE(fake.current.colors[virtual_index]->hdr_user_enabled);
  const auto pending_index = named_index(*fake.journal->pending_restore, L"virtual");
  EXPECT_TRUE(fake.journal->pending_restore->colors[pending_index]->hdr_user_enabled);
  // A concurrent legitimate VD resize is retained while the saved HDR intent is retried.
  fake.current.modes[fake.current.paths[virtual_index].sourceInfo.sourceModeInfoIdx].sourceMode.width = 3840;
  fake.color_set_ok = true;
  ASSERT_TRUE(manager.restore());
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"virtual")]->hdr_user_enabled);
  EXPECT_EQ(fake.current.modes[fake.current.paths[named_index(fake.current, L"virtual")].sourceInfo.sourceModeInfoIdx].sourceMode.width, 3840u);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, ExactAddBindingRecoversWindowsRememberedVirtualOnlyTopology) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(fake.journal && fake.journal->exclusive_started);
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
}

TEST(PrimaryDisplayExclusive, PreparedCrashLightsPhysicalWithoutClaimingOrMovingUnknownOutput) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  fake.current = make_snapshot({{L"unknown", 0, 0}});
  manager_t restarted(fake.io());
  EXPECT_FALSE(restarted.restore());
  expect_positions(fake.current, {{L"unknown", 0, 0}, {L"physical-primary", 1920, 0}});
  ASSERT_TRUE(fake.journal && fake.journal->prepared && fake.journal->pending_restore);
  EXPECT_TRUE(fake.journal->promoted_primary.empty());
  EXPECT_TRUE(deserialize(serialize(*fake.journal)));
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  EXPECT_TRUE(restarted.restore());
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, PreparedCrashAfterOrphanDisappearsCanReactivateAllInactivePhysicals) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  fake.current = {};
  EXPECT_TRUE(manager.restore());
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
}

TEST(PrimaryDisplayExclusive, ExactLateBindingCompletesPreparedPhysicalRecovery) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  fake.current = make_snapshot({{L"unknown", 0, 0}});
  EXPECT_FALSE(manager.restore());
  ASSERT_TRUE(fake.journal && fake.journal->prepared && fake.journal->pending_restore);
  ASSERT_TRUE(manager.bind_pending(L"unknown"));
  EXPECT_TRUE(fake.journal->exclusive_started);
  EXPECT_FALSE(fake.journal->prepared);
  ASSERT_TRUE(manager.restore(L"unknown"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"unknown", -1920, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, UserTopologyEditsAndUnknownActiveOutputsArePreserved) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"virtual", 2000, 300}, {L"new-monitor", 3920, 0}});
  fake.events.clear();
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayExclusive, InactiveRouteMatchingHandlesSourceConflictsAndRenumberedIds) {
  fake_io_t fake;
  start_exclusive(fake);
  auto routes = make_snapshot(baseline);
  for (size_t i = 0; i < routes.paths.size(); ++i) {
    auto &path = routes.paths[i];
    path.flags &= ~DISPLAYCONFIG_PATH_ACTIVE;
    path.sourceInfo.adapterId = {50, 1};
    path.targetInfo.adapterId = {60, 2};
    path.targetInfo.id += 200;
    path.sourceInfo.id = i == 2 ? 2 : 0;
    path.sourceInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
    path.targetInfo.modeInfoIdx = DISPLAYCONFIG_PATH_MODE_IDX_INVALID;
  }
  auto alternate = routes.paths[1];
  alternate.sourceInfo.id = 1;
  routes.paths.push_back(alternate);
  routes.device_paths.push_back(L"physical-primary");
  fake.available_override = routes;
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.restore());
  expect_positions(fake.current, baseline);
  const auto primary = named_index(fake.current, L"physical-primary");
  EXPECT_EQ(fake.current.paths[primary].sourceInfo.id, 1u);
  EXPECT_EQ(fake.current.paths[primary].targetInfo.id, 211u);
  EXPECT_EQ(fake.current.paths[primary].targetInfo.adapterId.LowPart, 60u);
}

TEST(PrimaryDisplayExclusive, DesktopImageModeRestoresUsingCurrentTargetIdentity) {
  fake_io_t fake;
  DISPLAYCONFIG_MODE_INFO desktop {};
  desktop.infoType = static_cast<DISPLAYCONFIG_MODE_INFO_TYPE>(3);
  desktop.id = fake.current.paths[1].targetInfo.id;
  desktop.adapterId = fake.current.paths[1].targetInfo.adapterId;
  std::array<LONG, 10> fields {1920, 1080, 0, 0, 1920, 1080, 0, 0, 1920, 1080};
  std::memcpy(&desktop.targetMode, fields.data(), sizeof(fields));
  fake.current.paths[1].targetInfo.desktopModeInfoIdx = static_cast<UINT32>(fake.current.modes.size());
  fake.current.modes.push_back(desktop);
  start_exclusive(fake);
  const auto encoded = serialize(*fake.journal);
  ASSERT_TRUE(deserialize(encoded));
  EXPECT_EQ(serialize(*deserialize(encoded)), encoded);
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.restore());
  const auto index = named_index(fake.current, L"physical-primary");
  const auto &path = fake.current.paths[index];
  const auto &restored = fake.current.modes[path.targetInfo.desktopModeInfoIdx];
  EXPECT_EQ(restored.id, path.targetInfo.id);
  EXPECT_NE(restored.id, path.sourceInfo.id);
  EXPECT_EQ(restored.adapterId.LowPart, path.targetInfo.adapterId.LowPart);
  std::array<LONG, 10> restored_fields {};
  std::memcpy(restored_fields.data(), &restored.targetMode, sizeof(restored_fields));
  EXPECT_EQ(restored_fields, fields);
}

TEST(PrimaryDisplayExclusive, ExactTimingReadbackMismatchRetainsJournal) {
  fake_io_t fake;
  start_exclusive(fake);
  fake.applied_mutation = [](snapshot_t &snapshot) {
    auto &path = snapshot.paths[0];
    snapshot.modes[path.targetInfo.targetModeInfoIdx].targetMode.targetVideoSignalInfo.videoStandard ^= 1;
  };
  manager_t manager(fake.io());
  EXPECT_FALSE(manager.restore());
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayExclusive, CorruptVersionThreeCcdPayloadIsRejectedAndOlderVersionsStillLoad) {
  fake_io_t fake;
  start_exclusive(fake);
  auto value = nlohmann::json::parse(serialize(*fake.journal));
  value["original_topology"]["modes"][0] = "00";
  EXPECT_FALSE(deserialize(value.dump()));
  value = nlohmann::json::parse(serialize(*fake.journal));
  value["version"] = 2;
  value.erase("exclusive_preserved");
  value.erase("exclusive_topology");
  EXPECT_TRUE(deserialize(value.dump()));
  fake_io_t primary_only;
  manager_t manager(primary_only.io());
  ASSERT_TRUE(manager.promote(L"virtual"));
  const auto legacy = serialize(*primary_only.journal);
  EXPECT_EQ(nlohmann::json::parse(legacy)["version"], 1);
  EXPECT_TRUE(deserialize(legacy));
}

TEST(PrimaryDisplayExclusiveCursor, ClipsOnlyVerifiedTopologyAndRefreshesForResizeAndReconnect) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  std::vector<std::optional<cursor_bounds_t>> clips;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view identity, std::optional<cursor_bounds_t> bounds) {
    EXPECT_EQ(identity, L"virtual");
    fake.events.push_back(bounds ? "cursor_acquire" : "cursor_release");
    clips.push_back(bounds);
    if (bounds) {
      EXPECT_EQ(fake.current.paths.size(), 2u);
      EXPECT_EQ(fake.current.device_paths[0], L"virtual");
      EXPECT_EQ(bounds->x, 0);
      EXPECT_EQ(bounds->y, 0);
    }
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_EQ(clips.back(), (cursor_bounds_t {0, 0, 1920, 1080}));
  EXPECT_LT(std::ranges::find(fake.events, "apply"), std::ranges::find(fake.events, "cursor_acquire"));

  const auto index = named_index(fake.current, L"virtual");
  fake.current.modes[fake.current.paths[index].sourceInfo.sourceModeInfoIdx].sourceMode.width = 3840;
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_EQ(clips.back(), (cursor_bounds_t {0, 0, 3840, 1080}));
  fake.events.clear();
  ASSERT_TRUE(manager.restore(L"virtual"));
  ASSERT_FALSE(fake.events.empty());
  EXPECT_EQ(fake.events.front(), "cursor_release");
  EXPECT_FALSE(clips.back());
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_TRUE(clips.back());
}

TEST(PrimaryDisplayExclusiveCursor, SoleVirtualOutputDoesNotAcquireGlobalClip) {
  fake_io_t fake;
  std::vector<std::optional<cursor_bounds_t>> clips;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    clips.push_back(bounds);
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_EQ(fake.current.paths.size(), 1u);
  ASSERT_EQ(clips.size(), 1u);
  EXPECT_FALSE(clips.back());
  ASSERT_TRUE(manager.restore(L"virtual"));
  EXPECT_FALSE(clips.back());
}

TEST(PrimaryDisplayLocalExclusive, SavesExactSinkPolicyAndRejectsUnapprovedTargets) {
  fake_io_t fake;
  fake.preserved_exclusive = {L"physical-left", L"physical-primary"};
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(fake.journal->local_sink, L"physical-left");
  EXPECT_EQ(fake.journal->exclusive_preserved, (std::vector<std::wstring> {L"physical-left"}));
  const auto decoded = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->local_sink, L"physical-left");
  auto malformed = nlohmann::json::parse(serialize(*fake.journal));
  malformed["local_sink"] = "other-output";
  EXPECT_FALSE(deserialize(malformed.dump()));
  fake.journal.reset();
  fake.preserved_exclusive.clear();
  EXPECT_FALSE(manager.prepare(true, L"physical-left"));
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, PromotesSourceAndConfinesCursorWithOnlySelectedSinkActive) {
  fake_io_t fake;
  fake.preserved_exclusive = {L"physical-left", L"physical-primary"};
  std::optional<cursor_bounds_t> clip;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    clip = bounds;
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_TRUE(manager.is_local_exclusive(L"virtual"));
  EXPECT_FALSE(manager.is_local_exclusive(L"other-source"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
  EXPECT_EQ(clip, (cursor_bounds_t {0, 0, 1920, 1080}));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(clip);
}

TEST(PrimaryDisplayLocalExclusive, ReconcileAdoptsHardwareModeColorAndRestoresSinkPosition) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  const auto sink = named_index(fake.current, L"physical-left");
  auto &source = fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode;
  source.width = 3840;
  source.position.x = -3840;
  fake.current.paths[sink].targetInfo.refreshRate = {90000, 1000};
  fake.current.colors[sink]->hdr_user_enabled = true;
  fake.current.colors[sink]->advanced_color_active = true;
  fake.current.colors[sink]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  const auto expected_color = *fake.current.colors[sink];
  const auto virtual_index = named_index(fake.current, L"virtual");
  fake.current.colors[virtual_index] = expected_color;
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -3840, 0}});
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"virtual")]->hdr_user_enabled);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"physical-left")]->hdr_user_enabled);
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(fake.journal->original_topology->modes[0].sourceMode.width, 1920u);
  EXPECT_FALSE(fake.journal->original_topology->colors[0]->hdr_user_enabled);
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-left", -3840, 120}, {L"physical-primary", 0, 0}, {L"virtual", 1920, -200}});
  const auto restored_sink = named_index(fake.current, L"physical-left");
  EXPECT_EQ(fake.current.paths[restored_sink].targetInfo.refreshRate.Numerator, 90000u);
  EXPECT_TRUE(fake.current.colors[restored_sink]->hdr_user_enabled);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, UnpluggedSinkDoesNotBlockVerifiedOrdinaryDisplayRestore) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.catalog.paths[named_index(fake.catalog, L"physical-left")].targetInfo.targetAvailable = FALSE;
  fake.ignore_apply = true;
  EXPECT_FALSE(manager.restore(L"virtual"));
  EXPECT_TRUE(fake.journal);
  fake.ignore_apply = false;
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, MissingOrdinaryOutputStillRetainsRecoveryBarrier) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.catalog.paths[named_index(fake.catalog, L"physical-primary")].targetInfo.targetAvailable = FALSE;
  EXPECT_FALSE(manager.restore(L"virtual"));
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, SolePrimaryGlassesCanStartAndUnplugToHeadless) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"glasses", 0, 0}});
  fake.catalog = make_snapshot({{L"glasses", 0, 0}, {L"virtual", 1920, 0}});
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  fake.current = fake.catalog;
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"glasses", -1920, 0}});
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.catalog.paths[0].targetInfo.targetAvailable = FALSE;
  ASSERT_TRUE(manager.restore(L"virtual"));
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, NewApprovedGlassesRemainDisabledUnlessTheyAreSelectedSink) {
  fake_io_t fake;
  fake.preserved_exclusive = {L"physical-left", L"new-glasses"};
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.catalog = make_snapshot({{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}, {L"virtual", 1920, -200}, {L"new-glasses", 3840, 0}});
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"physical-left", -1920, 0}, {L"new-glasses", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
  ASSERT_TRUE(fake.journal);
  EXPECT_EQ(fake.journal->exclusive_preserved, (std::vector<std::wstring> {L"physical-left"}));
  ASSERT_TRUE(manager.restore(L"virtual"));
  EXPECT_NE(std::ranges::find(fake.current.device_paths, L"new-glasses"), fake.current.device_paths.end());
}

TEST(PrimaryDisplayLocalExclusive, FreshAddNormalizationKeepsPreAddPhysicalBaseline) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}});
  fake.catalog = make_snapshot({{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"virtual", 3840, 0}});
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"virtual", 1920, 0}, {L"glasses", 3840, 0}});
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"virtual", 3840, 0}});
}

TEST(PrimaryDisplayLocalExclusive, GrowingMiddleSinkRestoresContiguousTopologyWithoutOverlap) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"physical-right", 3840, 0}, {L"virtual", 5760, 0}});
  fake.catalog = fake.current;
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  const auto sink = named_index(fake.current, L"glasses");
  auto &source = fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode;
  source.width = 3840;
  source.position.x = -3840;
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"physical-right", 5760, 0}, {L"virtual", 7680, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, ShrinkingMiddleSinkClosesTheGapBeforeRestore) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"physical-right", 5760, 0}, {L"virtual", 7680, 0}});
  fake.current.modes[fake.current.paths[1].sourceInfo.sourceModeInfoIdx].sourceMode.width = 3840;
  fake.catalog = fake.current;
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  const auto sink = named_index(fake.current, L"glasses");
  auto &source = fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode;
  source.width = 1920;
  source.position.x = -1920;
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}, {L"physical-right", 3840, 0}, {L"virtual", 5760, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, HardwareSbsSwitchRecoversRememberedExtendedLayout) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}});
  fake.current.modes[0].sourceMode.width = 5120;
  fake.current.modes[0].sourceMode.height = 2160;
  fake.current.colors[0]->hdr_user_enabled = true;
  fake.current.colors[0]->advanced_color_active = true;
  fake.current.colors[0]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  const auto monitor_color = *fake.current.colors[0];
  fake.preserved_exclusive.insert(L"glasses");
  std::optional<cursor_bounds_t> clip;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    clip = bounds;
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  fake.current = make_snapshot({{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}, {L"virtual", 7040, 0}});
  fake.current.modes[0].sourceMode.width = 5120;
  fake.current.modes[0].sourceMode.height = 2160;
  fake.current.colors[0] = monitor_color;
  fake.catalog = fake.current;
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"glasses", -1920, 0}});

  // Captured Windows state from the physical glasses' 2D -> SBS switch: the LG
  // becomes primary, the source moves to x=5120, and the wider sink moves to 7040.
  fake.current = make_snapshot({{L"lg-monitor", 0, 0}, {L"virtual", 5120, 0}, {L"glasses", 7040, 0}});
  fake.current.modes[0].sourceMode.width = 5120;
  fake.current.modes[0].sourceMode.height = 2160;
  fake.current.colors[0] = monitor_color;
  fake.current.modes[4].sourceMode.width = 3840;
  fake.current.paths[2].targetInfo.refreshRate = {120013, 1000};
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"glasses", -3840, 0}});
  const auto sink = named_index(fake.current, L"glasses");
  EXPECT_EQ(fake.current.modes[fake.current.paths[sink].sourceInfo.sourceModeInfoIdx].sourceMode.width, 3840u);
  EXPECT_EQ(fake.current.paths[sink].targetInfo.refreshRate.Numerator, 120013u);
  EXPECT_FALSE(fake.current.colors[sink]->hdr_user_enabled);
  EXPECT_EQ(clip, (cursor_bounds_t {0, 0, 1920, 1080}));
  ASSERT_TRUE(fake.journal);
  expect_positions(*fake.journal->original_topology, {{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}, {L"virtual", 8960, 0}});
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"lg-monitor")]->hdr_user_enabled);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, ReassertsSourcePrimaryWhenGlassesHardwareMakesSinkPrimary) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"physical-left", 0, 0}, {L"virtual", 3840, 0}});
  fake.current.modes[0].sourceMode.width = 3840;
  ASSERT_TRUE(manager.promote(L"virtual", true));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -3840, 0}});
}

TEST(PrimaryDisplayLocalExclusive, DirectPromotionDoesNotAdoptAnUnrecordedOutput) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"unrecorded-monitor", 0, 0}, {L"physical-left", 1920, 0}, {L"virtual", 3840, 0}});
  fake.events.clear();
  EXPECT_FALSE(manager.promote(L"virtual", true));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplayExclusive, RemoteModeDoesNotAdoptRepositionedPhysicalModeGeneration) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"physical-left", 1920, 0}, {L"virtual", 5760, 0}});
  fake.current.modes[2].sourceMode.width = 3840;
  fake.events.clear();
  EXPECT_FALSE(manager.reconcile_active_exclusive(L"virtual"));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
}

TEST(PrimaryDisplayLocalExclusive, StartupRecoversHardwareModeSwitchAfterDriverRetiresSource) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}, {L"virtual", 7040, 0}});
  fake.current.modes[0].sourceMode.width = 5120;
  fake.current.modes[0].sourceMode.height = 2160;
  fake.current.colors[0]->hdr_user_enabled = true;
  fake.current.colors[0]->advanced_color_active = true;
  fake.current.colors[0]->active_mode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
  fake.catalog = fake.current;
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.journal = deserialize(serialize(*fake.journal));
  ASSERT_TRUE(fake.journal);

  fake.current = make_snapshot({{L"lg-monitor", 0, 0}, {L"glasses", 7040, 0}});
  fake.current.modes[0].sourceMode.width = 5120;
  fake.current.modes[0].sourceMode.height = 2160;
  fake.current.modes[2].sourceMode.width = 3840;
  fake.current.paths[1].targetInfo.refreshRate = {120013, 1000};
  manager_t restarted(fake.io());
  fake.ignore_apply = true;
  EXPECT_FALSE(restarted.restore());
  EXPECT_TRUE(fake.journal);
  fake.ignore_apply = false;
  ASSERT_TRUE(restarted.restore());
  expect_positions(fake.current, {{L"lg-monitor", 0, 0}, {L"glasses", 5120, 0}});
  EXPECT_EQ(fake.current.paths[named_index(fake.current, L"glasses")].targetInfo.refreshRate.Numerator, 120013u);
  EXPECT_TRUE(fake.current.colors[named_index(fake.current, L"lg-monitor")]->hdr_user_enabled);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, SourceAbsentRecoveryDoesNotAdoptUnknownModeGeneration) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"physical-left"));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"physical-left", 1920, 0}, {L"unrecorded-output", 5760, 0}});
  fake.current.modes[2].sourceMode.width = 3840;
  fake.events.clear();
  EXPECT_FALSE(manager.restore());
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayLocalExclusive, GlassesUnpluggedBeforeAddDoesNotLeavePreparedRecoveryBlocked) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"physical-primary", 0, 0}, {L"glasses", 1920, 0}});
  fake.catalog = fake.current;
  fake.preserved_exclusive.insert(L"glasses");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true, L"glasses"));
  fake.current = make_snapshot({{L"physical-primary", 0, 0}});
  fake.catalog.paths[1].targetInfo.targetAvailable = FALSE;
  ASSERT_TRUE(manager.restore());
  expect_positions(fake.current, {{L"physical-primary", 0, 0}});
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, DisplayChangeReDisablesAnOrdinaryBaselineMonitor) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({
    {L"virtual", 0, 0},
    {L"physical-left", -1920, 0},
    {L"physical-primary", 1920, 0},
  });

  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"physical-left", -1920, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
}

TEST(PrimaryDisplayExclusive, DisconnectRestoresWhenAnOrdinaryMonitorReactivatesFirst) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({
    {L"virtual", 0, 0},
    {L"physical-left", -1920, 0},
    {L"physical-primary", 1920, 0},
  });

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, DisplayChangeCreatesRecoveryBaselineForAHeadlessSession) {
  fake_io_t fake;
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  fake.catalog = make_snapshot({{L"virtual", 0, 0}, {L"new-monitor", 1920, 0}});
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.promote(L"virtual", true));
  EXPECT_FALSE(fake.journal);

  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"new-monitor", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  ASSERT_TRUE(fake.journal);

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"new-monitor", 1920, 0}});
}

TEST(PrimaryDisplayExclusive, DisplayChangeDisablesAndLaterRestoresANewOrdinaryOutput) {
  fake_io_t fake;
  fake.catalog = make_snapshot({
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, -200},
    {L"hotplugged-monitor", 3840, 0},
  });
  start_exclusive(fake);
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-monitor", 1920, 0}});
  manager_t manager(fake.io());

  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore);
  EXPECT_NE(
    std::ranges::find(fake.journal->pending_restore->device_paths, L"hotplugged-monitor"),
    fake.journal->pending_restore->device_paths.end()
  );

  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-monitor", 1920, 0},
    {L"virtual", 3840, 0},
  });
}

TEST(PrimaryDisplayExclusive, DisplayChangeKeepsANewApprovedArOutputAndExcludesItFromCursorBounds) {
  fake_io_t fake;
  fake.catalog = make_snapshot({
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, -200},
    {L"hotplugged-ar", 3840, 0},
  });
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  std::vector<std::optional<cursor_bounds_t>> clips;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    clips.push_back(bounds);
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  ASSERT_FALSE(clips.empty());
  EXPECT_FALSE(clips.back());
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});

  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});
  ASSERT_FALSE(clips.empty());
  EXPECT_EQ(clips.back(), (cursor_bounds_t {0, 0, 1920, 1080}));
  ASSERT_TRUE(fake.journal);
  EXPECT_NE(
    std::ranges::find(fake.journal->exclusive_preserved, L"hotplugged-ar"),
    fake.journal->exclusive_preserved.end()
  );

  // Losing the only preserved output releases ClipCursor, while the exclusive session remains
  // alive so a later display notification can still be reconciled.
  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = FALSE;
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  EXPECT_FALSE(clips.back());

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = TRUE;
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});
  EXPECT_EQ(clips.back(), (cursor_bounds_t {0, 0, 1920, 1080}));
  ASSERT_TRUE(manager.restore(L"virtual"));
}

TEST(PrimaryDisplayExclusive, DisplayChangeReclassifiesAFormerlyDisconnectedHotplug) {
  fake_io_t fake;
  fake.catalog = make_snapshot({
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, -200},
    {L"hotplugged-ar", 3840, 0},
    {L"hotplugged-monitor", 5760, 0},
  });
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));

  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = FALSE;
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-monitor", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}});
  ASSERT_TRUE(fake.journal && fake.journal->pending_restore && fake.journal->exclusive_topology);
  EXPECT_EQ(std::ranges::count(fake.journal->exclusive_preserved, L"hotplugged-ar"), 0);
  EXPECT_EQ(std::ranges::count(fake.journal->exclusive_topology->device_paths, L"hotplugged-ar"), 0);
  EXPECT_EQ(std::ranges::count(fake.journal->pending_restore->device_paths, L"hotplugged-ar"), 0);
  EXPECT_EQ(std::ranges::count(fake.journal->pending_restore->device_paths, L"hotplugged-monitor"), 1);

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = TRUE;
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  expect_positions(fake.current, {{L"virtual", 0, 0}, {L"hotplugged-ar", -1920, 0}});

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = FALSE;
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-monitor", 1920, 0},
    {L"virtual", 3840, 0},
  });
}

TEST(PrimaryDisplayExclusive, DisconnectMergesANewOutputAfterApprovedArUnplugs) {
  fake_io_t fake;
  fake.catalog = make_snapshot({
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, -200},
    {L"hotplugged-ar", 3840, 0},
    {L"hotplugged-monitor", 5760, 0},
  });
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = FALSE;
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-monitor", 1920, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, {
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"hotplugged-monitor", 1920, 0},
    {L"virtual", 3840, 0},
  });
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusive, DisconnectRetiresAnUnavailableApprovedHotplugWithoutNotification) {
  fake_io_t fake;
  fake.catalog = make_snapshot({
    {L"physical-left", -1920, 120},
    {L"physical-primary", 0, 0},
    {L"virtual", 1920, -200},
    {L"hotplugged-ar", 3840, 0},
  });
  fake.preserved_exclusive.insert(L"hotplugged-ar");
  manager_t manager(fake.io());
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.current = make_snapshot({{L"virtual", 0, 0}, {L"hotplugged-ar", 1920, 0}});
  ASSERT_TRUE(manager.reconcile_active_exclusive(L"virtual"));

  fake.catalog.paths[named_index(fake.catalog, L"hotplugged-ar")].targetInfo.targetAvailable = FALSE;
  fake.current = make_snapshot({{L"virtual", 0, 0}});
  ASSERT_TRUE(manager.restore(L"virtual"));
  expect_positions(fake.current, baseline);
  EXPECT_FALSE(fake.journal);
}

TEST(PrimaryDisplayExclusiveCursor, PrimaryOnlyModeDoesNotAcquireAGlobalClip) {
  fake_io_t fake;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    EXPECT_FALSE(bounds);
    return true;
  };
  manager_t manager(io);
  EXPECT_TRUE(manager.promote(L"virtual", false));
  EXPECT_TRUE(manager.restore(L"virtual"));
}

TEST(PrimaryDisplayExclusiveCursor, FailedTopologyVerificationCannotAcquireClip) {
  fake_io_t fake;
  fake.ignore_apply = true;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    EXPECT_FALSE(bounds);
    return true;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_FALSE(manager.promote(L"virtual", true));
}

TEST(PrimaryDisplayExclusiveCursor, FailedClipAcquisitionRejectsPromotionAndReleasesTentativeOwnership) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  std::vector<bool> requested;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    requested.push_back(bounds.has_value());
    return !bounds;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  EXPECT_FALSE(manager.promote(L"virtual", true));
  EXPECT_EQ(requested, (std::vector<bool> {true, false}));
  EXPECT_TRUE(fake.journal);
}

TEST(PrimaryDisplayExclusiveCursor, FailedClipReleasePreventsTopologyRestoreUntilRetry) {
  fake_io_t fake;
  fake.preserved_exclusive.insert(L"physical-left");
  bool release_ok = false;
  auto io = fake.io();
  io.cursor_clip = [&](std::wstring_view, std::optional<cursor_bounds_t> bounds) {
    return bounds.has_value() || release_ok;
  };
  manager_t manager(io);
  ASSERT_TRUE(manager.prepare(true));
  ASSERT_TRUE(manager.bind_pending(L"virtual"));
  ASSERT_TRUE(manager.promote(L"virtual", true));
  fake.events.clear();
  EXPECT_FALSE(manager.restore(L"virtual"));
  EXPECT_EQ(std::ranges::count(fake.events, "apply"), 0);
  EXPECT_TRUE(fake.journal);
  release_ok = true;
  EXPECT_TRUE(manager.restore(L"virtual"));
}

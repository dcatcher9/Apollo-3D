/**
 * @file tests/unit/test_primary_display.cpp
 * @brief Primary-display transactions against an in-memory CCD and journal adapter.
 */
#include "../tests_common.h"
#include "src/platform/windows/primary_display.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>

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
    }
    return result;
  }

  const layout_t baseline {{L"physical-left", -1920, 120}, {L"physical-primary", 0, 0}, {L"virtual", 1920, -200}};

  struct fake_io_t {
    snapshot_t current = make_snapshot(baseline);
    std::optional<journal_t> journal;
    bool load_ok = true;
    bool save_ok = true;
    bool clear_ok = true;
    bool apply_ok = true;
    bool ignore_apply = false;
    int query_count = 0;
    int mutate_on_query = 0;
    std::function<void(snapshot_t &)> mutation;
    std::vector<std::string> events;

    io_t io() {
      return {
        [this]() -> std::optional<snapshot_t> {
          events.push_back("query");
          if (++query_count == mutate_on_query && mutation) {
            mutation(current);
          }
          return current;
        },
        [this](snapshot_t next) {
          events.push_back("apply");
          EXPECT_TRUE(journal.has_value()) << "A durable recovery record must precede every CCD mutation";
          if (!ignore_apply) {
            current = std::move(next);
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
}  // namespace

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

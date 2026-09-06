#ifdef _WIN32

  #include "src/platform/windows/virtual_display.h"

  #include <cstring>
  #include <limits>

  #include <gtest/gtest.h>

namespace {
  SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity(uint32_t lowPart, LONG highPart, UINT targetId) {
    return {{lowPart, highPart}, targetId};
  }

  DISPLAYCONFIG_PATH_INFO activePath(
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &target,
    UINT sourceId,
    UINT sourceModeIndex,
    UINT targetModeIndex,
    UINT32 extraFlags = 0
  ) {
    DISPLAYCONFIG_PATH_INFO path {};
    path.sourceInfo.adapterId = target.AdapterLuid;
    path.sourceInfo.id = sourceId;
    path.sourceInfo.modeInfoIdx = sourceModeIndex;
    path.targetInfo.adapterId = target.AdapterLuid;
    path.targetInfo.id = target.TargetId;
    path.targetInfo.modeInfoIdx = targetModeIndex;
    path.targetInfo.targetAvailable = TRUE;
    path.flags = DISPLAYCONFIG_PATH_ACTIVE | extraFlags;
    return path;
  }

  DISPLAYCONFIG_MODE_INFO sourceMode(
    const LUID &adapter,
    UINT sourceId,
    LONG x,
    LONG y,
    UINT width = 1920,
    UINT height = 1080
  ) {
    DISPLAYCONFIG_MODE_INFO mode {};
    mode.infoType = DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE;
    mode.id = sourceId;
    mode.adapterId = adapter;
    mode.sourceMode.width = width;
    mode.sourceMode.height = height;
    mode.sourceMode.pixelFormat = DISPLAYCONFIG_PIXELFORMAT_32BPP;
    mode.sourceMode.position = {x, y};
    return mode;
  }

  class VirtualDisplayPublication: public testing::Test {
  protected:
    using clock_t = std::chrono::steady_clock;
    using milliseconds = std::chrono::milliseconds;
    const clock_t::time_point started_at {std::chrono::seconds(42)};
    clock_t::time_point current_time = started_at;
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT added = identity(42, 7, 3);
    const VDISPLAY::display_identity_query_t published {
      VDISPLAY::display_identity_state_e::present,
      LR"(\\.\DISPLAY12)",
      LR"(\\?\DISPLAY#SMKD1CE#added)",
      L"Virtual Display",
    };
    std::vector<milliseconds> query_times;
    std::vector<milliseconds> wait_targets;

    VDISPLAY::display_identity_query_t poll(
      const std::function<VDISPLAY::display_identity_query_t()> &query
    ) {
      return VDISPLAY::waitForDisplayIdentityForTest(
        added.AdapterLuid,
        added.TargetId,
        [&](const LUID &adapter, uint32_t target) {
          EXPECT_EQ(adapter.LowPart, added.AdapterLuid.LowPart);
          EXPECT_EQ(adapter.HighPart, added.AdapterLuid.HighPart);
          EXPECT_EQ(target, added.TargetId);
          query_times.push_back(std::chrono::duration_cast<milliseconds>(current_time - started_at));
          return query();
        },
        [&]() {
          return current_time;
        },
        [&](clock_t::time_point deadline) {
          EXPECT_GT(deadline, current_time);
          wait_targets.push_back(std::chrono::duration_cast<milliseconds>(deadline - started_at));
          current_time = deadline;
        }
      );
    }
  };
}

TEST_F(VirtualDisplayPublication, ReturnsImmediatePublicationWithoutWaiting) {
  const auto result = poll([&]() {
    return published;
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::present);
  EXPECT_EQ(result.display_name, published.display_name);
  EXPECT_EQ(result.device_path, published.device_path);
  EXPECT_EQ(result.friendly_name, published.friendly_name);
  EXPECT_EQ(query_times, (std::vector<milliseconds> {milliseconds(0)}));
  EXPECT_TRUE(wait_targets.empty());
}

TEST_F(VirtualDisplayPublication, ObservesPublicationDuringTheFinalWait) {
  const auto result = poll([&]() {
    return current_time - started_at >= milliseconds(1000) ?
             published :
             VDISPLAY::display_identity_query_t {VDISPLAY::display_identity_state_e::absent};
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::present);
  EXPECT_EQ(result.display_name, published.display_name);
  ASSERT_EQ(query_times.size(), 7u);
  EXPECT_EQ(query_times.back(), milliseconds(1260));
}

TEST_F(VirtualDisplayPublication, AcceptsPublicationOnTheDeadline) {
  const auto result = poll([&]() {
    return current_time - started_at >= milliseconds(1260) ?
             published :
             VDISPLAY::display_identity_query_t {VDISPLAY::display_identity_state_e::absent};
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::present);
  EXPECT_EQ(current_time - started_at, milliseconds(1260));
}

TEST_F(VirtualDisplayPublication, TimesOutAfterCheckingTheEntireWaitBudget) {
  const auto result = poll([]() {
    return VDISPLAY::display_identity_query_t {VDISPLAY::display_identity_state_e::absent};
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::absent);
  EXPECT_TRUE(result.display_name.empty());
  EXPECT_EQ(query_times, (std::vector<milliseconds> {
                          milliseconds(0), milliseconds(20), milliseconds(60), milliseconds(140),
                          milliseconds(300), milliseconds(620), milliseconds(1260),
                        }));
  ASSERT_EQ(wait_targets.size(), 6u);
  EXPECT_EQ(wait_targets.back(), milliseconds(1260));
  EXPECT_EQ(current_time - started_at, milliseconds(1260));
}

TEST_F(VirtualDisplayPublication, RetriesTransientIdentityQueryFailures) {
  const auto result = poll([&]() {
    switch (query_times.size()) {
      case 1:
        return VDISPLAY::display_identity_query_t {};
      case 2:
        return VDISPLAY::display_identity_query_t {VDISPLAY::display_identity_state_e::absent};
      case 3:
        return VDISPLAY::display_identity_query_t {};
      default:
        return published;
    }
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::present);
  EXPECT_EQ(result.display_name, published.display_name);
  EXPECT_EQ(result.device_path, published.device_path);
  EXPECT_EQ(query_times.size(), 4u);
  EXPECT_EQ(current_time - started_at, milliseconds(140));
}

TEST_F(VirtualDisplayPublication, QueryTimeConsumesTheWaitBudget) {
  const auto result = poll([&]() {
    current_time += milliseconds(400);
    return VDISPLAY::display_identity_query_t {};
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::indeterminate);
  EXPECT_EQ(query_times, (std::vector<milliseconds> {
                          milliseconds(0), milliseconds(420), milliseconds(860),
                        }));
  EXPECT_EQ(wait_targets, (std::vector<milliseconds> {milliseconds(420), milliseconds(860)}));
  EXPECT_EQ(current_time - started_at, milliseconds(1260));
}

TEST_F(VirtualDisplayPublication, CapsTheFinalWaitToTheRemainingBudget) {
  const auto result = poll([&]() {
    current_time += milliseconds(100);
    return VDISPLAY::display_identity_query_t {};
  });

  EXPECT_EQ(result.state, VDISPLAY::display_identity_state_e::indeterminate);
  EXPECT_EQ(wait_targets, (std::vector<milliseconds> {
                           milliseconds(120), milliseconds(260), milliseconds(440),
                           milliseconds(700), milliseconds(1120), milliseconds(1260),
                         }));
  ASSERT_EQ(query_times.size(), 7u);
  EXPECT_EQ(query_times.back(), milliseconds(1260));
  // The final Windows query may itself take time after the wait deadline.
  EXPECT_EQ(current_time - started_at, milliseconds(1360));
}

TEST(VirtualDisplayIdentity, RejectsAnUnrelatedSudoOutput) {
  const auto retiring = identity(42, 7, 3);
  const auto unrelated = identity(42, 7, 4);

  EXPECT_FALSE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
    unrelated,
    LR"(\\?\DISPLAY#SMKD1CE#unrelated)"
  ));
}

TEST(VirtualDisplayWatchdog, KeepsFrequentHeartbeatsWithALongerDriverLease) {
  EXPECT_EQ(VDISPLAY::watchdogPingIntervalMsForTest(1), 333u);
  EXPECT_EQ(VDISPLAY::watchdogPingIntervalMsForTest(3), 1000u);
  EXPECT_EQ(VDISPLAY::watchdogPingIntervalMsForTest(30), 1000u);
}

TEST(VirtualDisplayIdentity, MatchesTheExactUnpublishedDriverIdentity) {
  const auto retiring = identity(42, 7, 3);

  EXPECT_TRUE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    {},
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#candidate)"
  ));
}

TEST(VirtualDisplayIdentity, DoesNotFollowAnExactIdReusedByAPhysicalOutput) {
  const auto retiring = identity(42, 7, 3);

  EXPECT_FALSE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    {},
    retiring,
    LR"(\\?\DISPLAY#TCL03D4#physical)"
  ));
}

TEST(VirtualDisplayIdentity, FollowsTheLearnedPathAcrossTargetRenumbering) {
  const auto retiring = identity(42, 7, 3);
  const auto renumbered = identity(43, 8, 9);
  constexpr std::wstring_view learnedPath = LR"(\\?\DISPLAY#SMKD1CE#retiring)";

  EXPECT_TRUE(VDISPLAY::virtualDisplayIdentityMatchesForTest(
    retiring,
    learnedPath,
    renumbered,
    learnedPath
  ));
}

TEST(VirtualDisplayDetach, ClearsOnlyTheExactVirtualTarget) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  constexpr UINT32 preserved_flag =
    DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE | DISPLAYCONFIG_PATH_BOOST_REFRESH_RATE;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 0, 0, 1, preserved_flag),
    activePath(retiring, 1, 2, 3, preserved_flag),
  };
  paths[0].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[0].sourceInfo.sourceModeInfoIdx = 0;
  paths[0].targetInfo.desktopModeInfoIdx = DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
  paths[0].targetInfo.targetModeInfoIdx = 1;
  paths[1].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[1].sourceInfo.sourceModeInfoIdx = 2;
  paths[1].targetInfo.desktopModeInfoIdx = DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID;
  paths[1].targetInfo.targetModeInfoIdx = 3;
  const auto original_physical = paths[0];
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[1],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ready
  );
  EXPECT_EQ(std::memcmp(&paths[0], &original_physical, sizeof(original_physical)), 0);
  EXPECT_EQ(paths[1].flags, preserved_flag);
  EXPECT_EQ(paths[1].sourceInfo.modeInfoIdx, DISPLAYCONFIG_PATH_MODE_IDX_INVALID);
  EXPECT_EQ(paths[1].sourceInfo.cloneGroupId, DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID);
  EXPECT_EQ(paths[1].sourceInfo.sourceModeInfoIdx, DISPLAYCONFIG_PATH_SOURCE_MODE_IDX_INVALID);
  EXPECT_EQ(paths[1].targetInfo.modeInfoIdx, DISPLAYCONFIG_PATH_MODE_IDX_INVALID);
  EXPECT_EQ(paths[1].targetInfo.desktopModeInfoIdx, DISPLAYCONFIG_PATH_DESKTOP_IMAGE_IDX_INVALID);
  EXPECT_EQ(paths[1].targetInfo.targetModeInfoIdx, DISPLAYCONFIG_PATH_TARGET_MODE_IDX_INVALID);
}

TEST(VirtualDisplayDetach, PreservesAPhysicalCloneSharingTheSourceMode) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 5, 0, 10, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
    activePath(retiring, 5, 0, 11, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
  };
  // A clone shares the source adapter/id and source mode. Only the retiring target path may change.
  paths[1].sourceInfo.adapterId = paths[0].sourceInfo.adapterId;
  paths[0].sourceInfo.cloneGroupId = 4;
  paths[0].sourceInfo.sourceModeInfoIdx = 0;
  paths[1].sourceInfo.cloneGroupId = 4;
  paths[1].sourceInfo.sourceModeInfoIdx = 0;
  const auto original_physical = paths[0];
  std::vector<DISPLAYCONFIG_MODE_INFO> modes {
    sourceMode(paths[0].sourceInfo.adapterId, paths[0].sourceInfo.id, 0, 0),
  };
  const auto original_modes = modes;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[1],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ready
  );
  EXPECT_EQ(std::memcmp(&paths[0], &original_physical, sizeof(original_physical)), 0);
  EXPECT_NE(paths[0].flags & DISPLAYCONFIG_PATH_ACTIVE, 0u);
  EXPECT_EQ(paths[0].sourceInfo.cloneGroupId, 4u);
  EXPECT_EQ(paths[0].sourceInfo.sourceModeInfoIdx, 0u);
  EXPECT_EQ(paths[1].flags & DISPLAYCONFIG_PATH_ACTIVE, 0u);
  EXPECT_EQ(paths[1].sourceInfo.modeInfoIdx, DISPLAYCONFIG_PATH_MODE_IDX_INVALID);
  EXPECT_EQ(paths[1].targetInfo.modeInfoIdx, DISPLAYCONFIG_PATH_MODE_IDX_INVALID);
  EXPECT_TRUE(VDISPLAY::rebaseVirtualDisplaySurvivorsForTest(paths, modes));
  EXPECT_EQ(
    std::memcmp(modes.data(), original_modes.data(), sizeof(modes[0]) * modes.size()),
    0
  );
}

TEST(VirtualDisplayDetach, AcceptsIdenticalCrossAdapterCloneGeometry) {
  const auto physical_a = identity(11, 1, 1);
  const auto physical_b = identity(12, 1, 2);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical_a, 0, 0, 0),
    activePath(physical_b, 1, 0, 0, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
    activePath(retiring, 2, 0, 0, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
  };
  paths[1].sourceInfo.cloneGroupId = 8;
  paths[1].sourceInfo.sourceModeInfoIdx = 1;
  paths[2].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[2].sourceInfo.sourceModeInfoIdx = 2;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes {
    sourceMode(physical_a.AdapterLuid, 0, 0, 0),
    sourceMode(physical_b.AdapterLuid, 1, 0, 0),
    sourceMode(retiring.AdapterLuid, 2, 1920, 0),
  };
  const auto original_modes = modes;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#clone-a)",
    LR"(\\?\DISPLAY#TCL03D4#clone-b)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  ASSERT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[2],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ready
  );
  EXPECT_TRUE(VDISPLAY::rebaseVirtualDisplaySurvivorsForTest(paths, modes));
  EXPECT_EQ(
    std::memcmp(modes.data(), original_modes.data(), sizeof(modes[0]) * modes.size()),
    0
  );
}

TEST(VirtualDisplayDetach, RefusesToDeactivateTheOnlyDesktopOutput) {
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {activePath(retiring, 0, 0, 1)};
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[0],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::skipped_only_active
  );
  EXPECT_EQ(std::memcmp(paths.data(), original.data(), sizeof(paths[0])), 0);
}

TEST(VirtualDisplayDetach, RefusesAnUnavailableStalePathAsTheOnlySurvivor) {
  const auto stale_physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(stale_physical, 0, 0, 1),
    activePath(retiring, 1, 2, 3),
  };
  paths[0].targetInfo.targetAvailable = FALSE;
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#stale)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[1],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::skipped_only_active
  );
  EXPECT_EQ(
    std::memcmp(paths.data(), original.data(), sizeof(paths[0]) * paths.size()),
    0
  );
}

TEST(VirtualDisplayDetach, RejectsAGapLeftByARetiringPrimarySource) {
  const auto physical_left = identity(11, 1, 1);
  const auto physical_right = identity(12, 1, 2);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical_left, 0, 0, 0),
    activePath(physical_right, 1, 0, 0, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
    activePath(retiring, 2, 0, 0, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
  };
  paths[1].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[1].sourceInfo.sourceModeInfoIdx = 1;
  paths[2].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[2].sourceInfo.sourceModeInfoIdx = 2;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes {
    sourceMode(physical_left.AdapterLuid, 0, -1920, 0),
    sourceMode(physical_right.AdapterLuid, 1, 1920, 0, 2560, 1440),
    sourceMode(retiring.AdapterLuid, 2, 0, 0),
  };
  const auto original_modes = modes;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#left)",
    LR"(\\?\DISPLAY#TCL03D4#right)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  ASSERT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[2],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ready
  );
  EXPECT_FALSE(VDISPLAY::rebaseVirtualDisplaySurvivorsForTest(paths, modes));
  EXPECT_EQ(
    std::memcmp(modes.data(), original_modes.data(), sizeof(modes[0]) * modes.size()),
    0
  );
}

TEST(VirtualDisplayDetach, RebasesALegacyIndexedSurvivorAfterPrimaryRemoval) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 0, 1, 0),
    activePath(retiring, 1, 0, 0, DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE),
  };
  paths[1].sourceInfo.cloneGroupId = DISPLAYCONFIG_PATH_CLONE_GROUP_INVALID;
  paths[1].sourceInfo.sourceModeInfoIdx = 0;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes {
    sourceMode(retiring.AdapterLuid, 1, 0, 0),
    sourceMode(physical.AdapterLuid, 0, 1920, 0, 2560, 1440),
  };
  const auto retiring_mode = modes[0];
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
  };

  ASSERT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      device_paths[1],
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ready
  );
  ASSERT_TRUE(VDISPLAY::rebaseVirtualDisplaySurvivorsForTest(paths, modes));
  EXPECT_EQ(modes[1].sourceMode.position.x, 0);
  EXPECT_EQ(modes[1].sourceMode.position.y, 0);
  EXPECT_EQ(modes[1].sourceMode.width, 2560u);
  EXPECT_EQ(modes[1].sourceMode.height, 1440u);
  EXPECT_EQ(std::memcmp(&modes[0], &retiring_mode, sizeof(modes[0])), 0);
}

TEST(VirtualDisplayDetach, TreatsAnInactiveTargetAsIdempotentlyDetached) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {activePath(physical, 0, 0, 1)};
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      LR"(\\?\DISPLAY#SMKD1CE#retiring)",
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::already_inactive
  );
  EXPECT_EQ(std::memcmp(paths.data(), original.data(), sizeof(paths[0])), 0);
}

TEST(VirtualDisplayDetach, RejectsAmbiguousLearnedPathMatchesWithoutMutation) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 0, 0, 1),
    activePath(retiring, 1, 2, 3),
  };
  const auto original = paths;
  constexpr std::wstring_view learned_path = LR"(\\?\DISPLAY#SMKD1CE#duplicate)";
  const std::vector<std::wstring> device_paths {
    std::wstring(learned_path),
    std::wstring(learned_path),
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      learned_path,
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ambiguous_identity
  );
  EXPECT_EQ(
    std::memcmp(paths.data(), original.data(), sizeof(paths[0]) * paths.size()),
    0
  );
}

TEST(VirtualDisplayDetach, DoesNotDetachAPhysicalPathWithAReusedDriverIdentity) {
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {activePath(retiring, 0, 0, 1)};
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      {},
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::already_inactive
  );
  EXPECT_EQ(std::memcmp(paths.data(), original.data(), sizeof(paths[0])), 0);
}

TEST(VirtualDisplayDetach, DoesNotMatchTheSameTargetIdOnAnotherAdapter) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  const auto other_adapter = identity(43, 8, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 0, 0, 1),
    activePath(other_adapter, 1, 2, 3),
  };
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    LR"(\\?\DISPLAY#SMKD1CE#other-adapter)",
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      {},
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::already_inactive
  );
  EXPECT_EQ(
    std::memcmp(paths.data(), original.data(), sizeof(paths[0]) * paths.size()),
    0
  );
}

TEST(VirtualDisplayDetach, RejectsAnEmptyExactDevicePathAsAmbiguous) {
  const auto physical = identity(11, 1, 1);
  const auto retiring = identity(42, 7, 3);
  std::vector<DISPLAYCONFIG_PATH_INFO> paths {
    activePath(physical, 0, 0, 1),
    activePath(retiring, 1, 2, 3),
  };
  const auto original = paths;
  const std::vector<std::wstring> device_paths {
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    {},
  };

  EXPECT_EQ(
    VDISPLAY::prepareVirtualDisplayDetachPathsForTest(
      retiring,
      {},
      device_paths,
      paths
    ),
    VDISPLAY::desktop_detach_plan_e::ambiguous_identity
  );
  EXPECT_EQ(
    std::memcmp(paths.data(), original.data(), sizeof(paths[0]) * paths.size()),
    0
  );
}

TEST(VirtualDisplayRetirement, KeepsAnAvailableExactVirtualTargetPresent) {
  const auto retiring = identity(42, 7, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
    true,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::present
  );
}

TEST(VirtualDisplayRetirement, FollowsALearnedPathAfterTargetRenumbering) {
  const auto retiring = identity(42, 7, 3);
  const auto renumbered = identity(43, 8, 9);
  constexpr std::wstring_view learned_path = LR"(\\?\DISPLAY#SMKD1CE#retiring)";
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    renumbered,
    std::wstring(learned_path),
    true,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, learned_path, candidates),
    VDISPLAY::display_identity_state_e::present
  );
}

TEST(VirtualDisplayRetirement, TreatsAnExactTargetNameFailureAsIndeterminate) {
  const auto retiring = identity(42, 7, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    retiring,
    {},
    true,
    false,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::indeterminate
  );
}

TEST(VirtualDisplayRetirement, TreatsAnEmptyAvailableDevicePathAsIndeterminate) {
  const auto retiring = identity(42, 7, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    retiring,
    {},
    true,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::indeterminate
  );
}

TEST(VirtualDisplayRetirement, IgnoresAnUnavailableStaleTarget) {
  const auto retiring = identity(42, 7, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    retiring,
    LR"(\\?\DISPLAY#SMKD1CE#retiring)",
    false,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::absent
  );
}

TEST(VirtualDisplayRetirement, IgnoresAnUnrelatedAvailableTarget) {
  const auto retiring = identity(42, 7, 3);
  const auto unrelated = identity(43, 8, 9);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    unrelated,
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    true,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::absent
  );
}

TEST(VirtualDisplayRetirement, IgnoresAnExactIdentityReusedByAPhysicalTarget) {
  const auto retiring = identity(42, 7, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    retiring,
    LR"(\\?\DISPLAY#TCL03D4#physical)",
    true,
    true,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, {}, candidates),
    VDISPLAY::display_identity_state_e::absent
  );
}

TEST(VirtualDisplayRetirement, ANameFailureBlocksLearnedPathAbsenceProof) {
  const auto retiring = identity(42, 7, 3);
  const auto unrelated = identity(43, 8, 9);
  constexpr std::wstring_view learned_path = LR"(\\?\DISPLAY#SMKD1CE#retiring)";
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {{
    unrelated,
    {},
    true,
    false,
  }};

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, learned_path, candidates),
    VDISPLAY::display_identity_state_e::indeterminate
  );
}

TEST(VirtualDisplayRetirement, ADefiniteLearnedMatchOverridesAnEarlierNameFailure) {
  const auto retiring = identity(42, 7, 3);
  const auto failed = identity(43, 8, 9);
  const auto renumbered = identity(44, 8, 10);
  constexpr std::wstring_view learned_path = LR"(\\?\DISPLAY#SMKD1CE#retiring)";
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {
    {
      failed,
      {},
      true,
      false,
    },
    {
      renumbered,
      std::wstring(learned_path),
      true,
      true,
    },
  };

  EXPECT_EQ(
    VDISPLAY::virtualDisplayRetirementStateForTest(retiring, learned_path, candidates),
    VDISPLAY::display_identity_state_e::present
  );
}

TEST(VirtualDisplayDetach, AReappearingActivePathInvalidatesOlderInactiveEvidence) {
  const auto latest_active = VDISPLAY::desktopDetachEvidenceForTest(
    false,
    3,
    std::chrono::milliseconds(500)
  );
  EXPECT_FALSE(latest_active.path_confirmed_inactive);
  EXPECT_FALSE(latest_active.shell_settled);
}

TEST(VirtualDisplayDetach, AnUnknownLatestObservationCannotAuthorizeRemoval) {
  // Both an active result and a failed/null observation map to latestPathInactive=false.
  // Even arbitrarily old counters from prior absence observations must not survive it.
  const auto latest_unknown = VDISPLAY::desktopDetachEvidenceForTest(
    false,
    std::numeric_limits<unsigned int>::max(),
    std::chrono::hours(24)
  );
  EXPECT_FALSE(latest_unknown.path_confirmed_inactive);
  EXPECT_FALSE(latest_unknown.shell_settled);
}

TEST(VirtualDisplayDetach, RequiresRepeatedAbsenceAndShellSettleTime) {
  const auto too_few_observations = VDISPLAY::desktopDetachEvidenceForTest(
    true,
    2,
    std::chrono::seconds(1)
  );
  EXPECT_FALSE(too_few_observations.path_confirmed_inactive);
  EXPECT_FALSE(too_few_observations.shell_settled);

  const auto not_shell_settled = VDISPLAY::desktopDetachEvidenceForTest(
    true,
    3,
    std::chrono::milliseconds(499)
  );
  EXPECT_TRUE(not_shell_settled.path_confirmed_inactive);
  EXPECT_FALSE(not_shell_settled.shell_settled);

  const auto settled = VDISPLAY::desktopDetachEvidenceForTest(
    true,
    3,
    std::chrono::milliseconds(500)
  );
  EXPECT_TRUE(settled.path_confirmed_inactive);
  EXPECT_TRUE(settled.shell_settled);
}

TEST(VirtualDisplayColorRestore, DoesNotReassertMatchingUserStateForActiveModeLag) {
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionForTest(
      false,
      false,
      false,
      true,
      true,
      false,
      false,
      false,
      true,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR
    ),
    VDISPLAY::color_reconcile_action_e::wait_for_active_mode
  );

  EXPECT_EQ(
    VDISPLAY::colorReconcileActionForTest(
      false,
      false,
      false,
      true,
      true,
      false,
      false,
      true,
      true,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR
    ),
    VDISPLAY::color_reconcile_action_e::wait_for_active_mode
  );
}

TEST(VirtualDisplayColorRestore, ActiveModeOnlyLagStopsBlockingAfterBoundedEvidence) {
  EXPECT_FALSE(VDISPLAY::activeColorModeObservationSettledForTest(
    false,
    true,
    3,
    std::chrono::milliseconds(500)
  ));
  EXPECT_FALSE(VDISPLAY::activeColorModeObservationSettledForTest(
    true,
    false,
    3,
    std::chrono::milliseconds(500)
  ));
  EXPECT_FALSE(VDISPLAY::activeColorModeObservationSettledForTest(
    true,
    true,
    2,
    std::chrono::seconds(1)
  ));
  EXPECT_FALSE(VDISPLAY::activeColorModeObservationSettledForTest(
    true,
    true,
    3,
    std::chrono::milliseconds(499)
  ));
  EXPECT_TRUE(VDISPLAY::activeColorModeObservationSettledForTest(
    true,
    true,
    3,
    std::chrono::milliseconds(500)
  ));
}

TEST(VirtualDisplayColorRestore, SelectsOnlyMismatchedUserStateSetters) {
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionForTest(
      true,
      false,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR
    ),
    VDISPLAY::color_reconcile_action_e::set_legacy_advanced_color
  );
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionForTest(
      false,
      false,
      false,
      false,
      true,
      false,
      false,
      false,
      false,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR
    ),
    VDISPLAY::color_reconcile_action_e::set_hdr_user_state
  );
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionForTest(
      false,
      false,
      false,
      true,
      true,
      false,
      true,
      false,
      false,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR,
      DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR
    ),
    VDISPLAY::color_reconcile_action_e::set_wcg_user_state
  );
}

TEST(VirtualDisplayColorRestore, TriesTheOtherUserBitAfterOneSetterWasAccepted) {
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionAfterAcceptedSettersForTest(
      false,
      true,
      false,
      true,
      true,
      false
    ),
    VDISPLAY::color_reconcile_action_e::set_wcg_user_state
  );
  EXPECT_EQ(
    VDISPLAY::colorReconcileActionAfterAcceptedSettersForTest(
      false,
      true,
      false,
      true,
      true,
      true
    ),
    VDISPLAY::color_reconcile_action_e::wait_for_active_mode
  );
}

TEST(VirtualDisplayColorRestore, AcceptedSetterIsNeverRepeatedDuringOneDetach) {
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_legacy_advanced_color,
      20
    ),
    1u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_hdr_user_state,
      20
    ),
    1u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_wcg_user_state,
      20
    ),
    1u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::wait_for_active_mode,
      20
    ),
    0u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::settled,
      20
    ),
    0u
  );
}

TEST(VirtualDisplayColorRestore, NoSetterIsConsumedWithoutAPollIteration) {
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_legacy_advanced_color,
      0
    ),
    0u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_hdr_user_state,
      0
    ),
    0u
  );
  EXPECT_EQ(
    VDISPLAY::repeatedColorSetterAttemptCountForTest(
      VDISPLAY::color_reconcile_action_e::set_wcg_user_state,
      0
    ),
    0u
  );
}

TEST(VirtualDisplayColorRestore, RejectedSetterRetriesUseAOneSecondBackoff) {
  EXPECT_FALSE(VDISPLAY::rejectedColorSetterRetryAllowedForTest(
    std::chrono::milliseconds(0)
  ));
  EXPECT_FALSE(VDISPLAY::rejectedColorSetterRetryAllowedForTest(
    std::chrono::milliseconds(999)
  ));
  EXPECT_TRUE(VDISPLAY::rejectedColorSetterRetryAllowedForTest(
    std::chrono::milliseconds(1000)
  ));
  EXPECT_TRUE(VDISPLAY::rejectedColorSetterRetryAllowedForTest(
    std::chrono::seconds(10)
  ));
}

TEST(VirtualDisplayColorRestore, BackedOffHdrSetterDoesNotBlockIndependentWcgRestore) {
  EXPECT_TRUE(VDISPLAY::wcgSetterSelectedDuringHdrBackoffForTest());
}

TEST(VirtualDisplayColorRestore, RetiresOnlyAStablyMissingSurvivorAndRearmsOnReturn) {
  EXPECT_FALSE(VDISPLAY::missingColorSurvivorRetiredForTest(
    false,
    30,
    std::chrono::seconds(5)
  ));
  EXPECT_FALSE(VDISPLAY::missingColorSurvivorRetiredForTest(
    true,
    2,
    std::chrono::seconds(1)
  ));
  EXPECT_FALSE(VDISPLAY::missingColorSurvivorRetiredForTest(
    true,
    3,
    std::chrono::milliseconds(499)
  ));
  EXPECT_TRUE(VDISPLAY::missingColorSurvivorRetiredForTest(
    true,
    3,
    std::chrono::milliseconds(500)
  ));
  EXPECT_TRUE(VDISPLAY::retiredColorSurvivorReactivatesForTest());
}

TEST(VirtualDisplayColorRestore, RetryCapturesOnlyWhenADetachNeedsAnUncapturedContract) {
  EXPECT_FALSE(VDISPLAY::detachRetryCapturesColorContractForTest(false, false, false));
  EXPECT_FALSE(VDISPLAY::detachRetryCapturesColorContractForTest(false, true, false));
  EXPECT_FALSE(VDISPLAY::detachRetryCapturesColorContractForTest(true, false, false));
  EXPECT_FALSE(VDISPLAY::detachRetryCapturesColorContractForTest(true, true, false));

  EXPECT_TRUE(VDISPLAY::detachRetryCapturesColorContractForTest(false, false, true));
  EXPECT_TRUE(VDISPLAY::detachRetryCapturesColorContractForTest(false, true, true));
  EXPECT_TRUE(VDISPLAY::detachRetryCapturesColorContractForTest(true, false, true));
  EXPECT_FALSE(VDISPLAY::detachRetryCapturesColorContractForTest(true, true, true));
}

TEST(VirtualDisplayColorRestore, RetryAddsNewSurvivorsWithoutOverwritingOriginalContracts) {
  EXPECT_TRUE(VDISPLAY::colorSurvivorContractMergePreservesExistingForTest());
}

TEST(VirtualDisplayColorRestore, NewDetachApplyResetsPerTransitionObservationState) {
  EXPECT_TRUE(VDISPLAY::colorSurvivorTransitionStateResetsForReapplyForTest());
}

TEST(VirtualDisplayRetirement, CoalescesDuplicateAllPathTargetsBeforeIdentityQueries) {
  const auto duplicate = identity(42, 7, 3);
  const auto unrelated = identity(43, 8, 9);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {
    {duplicate, {}, true, true},
    {duplicate, {}, false, true},
    {unrelated, {}, true, true},
  };

  EXPECT_EQ(
    VDISPLAY::coalescedRetirementCandidateCountForTest(candidates),
    2u
  );
}

TEST(VirtualDisplayRetirement, CoalescingKeepsTheAdapterAsPartOfTheTargetIdentity) {
  const auto first_adapter = identity(42, 7, 3);
  const auto second_adapter = identity(43, 8, 3);
  const std::vector<VDISPLAY::retirement_path_candidate_t> candidates {
    {first_adapter, {}, true, true},
    {second_adapter, {}, true, true},
  };

  EXPECT_EQ(
    VDISPLAY::coalescedRetirementCandidateCountForTest(candidates),
    2u
  );
}

TEST(VirtualDisplayRetirement, DriverRemovalRequiresAuthoritativeActivePathAbsence) {
  EXPECT_TRUE(VDISPLAY::isDriverRemovalSafeAfterDesktopDetach(
    true,
    VDISPLAY::display_identity_state_e::absent
  ));
  EXPECT_FALSE(VDISPLAY::isDriverRemovalSafeAfterDesktopDetach(
    true,
    VDISPLAY::display_identity_state_e::present
  ));
  EXPECT_FALSE(VDISPLAY::isDriverRemovalSafeAfterDesktopDetach(
    true,
    VDISPLAY::display_identity_state_e::indeterminate
  ));

  // The sole-output path cannot be detached while retaining a desktop and is the only explicit
  // exception. There is no surviving multi-monitor Explorer state to protect in that case.
  EXPECT_TRUE(VDISPLAY::isDriverRemovalSafeAfterDesktopDetach(
    false,
    VDISPLAY::display_identity_state_e::present
  ));
}

TEST(VirtualDisplayRetirement, StableAbsenceEvidenceResetsOnInvalidationOrNonAbsence) {
  EXPECT_EQ(
    VDISPLAY::advanceStableAbsenceEvidence(
      2,
      false,
      VDISPLAY::display_identity_state_e::absent
    ),
    3u
  );
  EXPECT_EQ(
    VDISPLAY::advanceStableAbsenceEvidence(
      2,
      false,
      VDISPLAY::display_identity_state_e::present
    ),
    0u
  );
  EXPECT_EQ(
    VDISPLAY::advanceStableAbsenceEvidence(
      2,
      false,
      VDISPLAY::display_identity_state_e::indeterminate
    ),
    0u
  );
  EXPECT_EQ(
    VDISPLAY::advanceStableAbsenceEvidence(
      2,
      true,
      VDISPLAY::display_identity_state_e::absent
    ),
    0u
  );
}

#endif

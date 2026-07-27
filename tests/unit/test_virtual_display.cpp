#ifdef _WIN32

  #include "src/platform/windows/virtual_display.h"

  #include <cstring>

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

#endif

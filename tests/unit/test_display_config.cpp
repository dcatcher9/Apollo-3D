/**
 * @file tests/unit/test_display_config.cpp
 * @brief Tests for shared Windows CCD and Advanced Color adapters.
 */
#include "../tests_common.h"
#include "src/platform/common.h"
#include "src/platform/windows/display_config.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cwchar>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <vector>

namespace {
  struct target_name_t {
    UINT32 id;
    std::wstring device_path;
    std::wstring display_name;
  };

  std::vector<target_name_t> target_names;

  LONG WINAPI get_target_name(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    for (const auto &target : target_names) {
      if (target.id != header->id) {
        continue;
      }
      if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME) {
        auto *name = reinterpret_cast<DISPLAYCONFIG_TARGET_DEVICE_NAME *>(header);
        std::wcsncpy(name->monitorDevicePath, target.device_path.c_str(), std::size(name->monitorDevicePath) - 1);
        return ERROR_SUCCESS;
      }
      if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME) {
        auto *name = reinterpret_cast<DISPLAYCONFIG_SOURCE_DEVICE_NAME *>(header);
        std::wcsncpy(name->viewGdiDeviceName, target.display_name.c_str(), std::size(name->viewGdiDeviceName) - 1);
        return ERROR_SUCCESS;
      }
    }
    return ERROR_NOT_FOUND;
  }

  std::vector<DISPLAYCONFIG_PATH_INFO> target_paths() {
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    for (const auto &target : target_names) {
      DISPLAYCONFIG_PATH_INFO path {};
      path.flags = DISPLAYCONFIG_PATH_ACTIVE;
      path.sourceInfo.id = path.targetInfo.id = target.id;
      paths.push_back(path);
    }
    return paths;
  }

  LONG WINAPI named_buffer_sizes(UINT32, UINT32 *path_count, UINT32 *mode_count) {
    *path_count = static_cast<UINT32>(target_names.size());
    *mode_count = 0;
    return ERROR_SUCCESS;
  }

  LONG WINAPI named_query(
    UINT32,
    UINT32 *path_count,
    DISPLAYCONFIG_PATH_INFO *paths,
    UINT32 *mode_count,
    DISPLAYCONFIG_MODE_INFO *,
    DISPLAYCONFIG_TOPOLOGY_ID *
  ) {
    const auto active = target_paths();
    std::copy(active.begin(), active.end(), paths);
    *path_count = static_cast<UINT32>(active.size());
    *mode_count = 0;
    return ERROR_SUCCESS;
  }

  void WINAPI no_sleep(DWORD) {
  }

  LONG WINAPI denied_device_info(DISPLAYCONFIG_DEVICE_INFO_HEADER *) {
    return ERROR_ACCESS_DENIED;
  }

  LONG WINAPI unterminated_target_name(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    auto *name = reinterpret_cast<DISPLAYCONFIG_TARGET_DEVICE_NAME *>(header);
    std::fill(std::begin(name->monitorDevicePath), std::end(name->monitorDevicePath), L'x');
    return ERROR_SUCCESS;
  }

  struct fake_set_config_state_t {
    std::vector<UINT32> flags;
    std::vector<UINT32> path_counts;
    LONG validate_status = ERROR_SUCCESS;
    LONG apply_status = ERROR_SUCCESS;
  } fake_set_config_state;

  LONG WINAPI fake_set_display_config(UINT32 path_count, DISPLAYCONFIG_PATH_INFO *, UINT32, DISPLAYCONFIG_MODE_INFO *, UINT32 flags) {
    fake_set_config_state.flags.push_back(flags);
    fake_set_config_state.path_counts.push_back(path_count);
    return flags & SDC_VALIDATE ? fake_set_config_state.validate_status : fake_set_config_state.apply_status;
  }

  struct fake_query_state_t {
    unsigned int size_calls = 0;
    unsigned int query_calls = 0;
    std::vector<DWORD> sleeps;
    bool never_settles = false;
  } fake_query_state;

  LONG WINAPI fake_get_buffer_sizes(
    UINT32,
    UINT32 *path_count,
    UINT32 *mode_count
  ) {
    ++fake_query_state.size_calls;
    if (fake_query_state.size_calls == 1) {
      return ERROR_INSUFFICIENT_BUFFER;
    }
    *path_count = 2;
    *mode_count = 3;
    return ERROR_SUCCESS;
  }

  LONG WINAPI fake_query_display_config(
    UINT32,
    UINT32 *path_count,
    DISPLAYCONFIG_PATH_INFO *,
    UINT32 *mode_count,
    DISPLAYCONFIG_MODE_INFO *,
    DISPLAYCONFIG_TOPOLOGY_ID *
  ) {
    ++fake_query_state.query_calls;
    if (fake_query_state.never_settles || fake_query_state.query_calls == 1) {
      return ERROR_INSUFFICIENT_BUFFER;
    }
    *path_count = 1;
    *mode_count = 2;
    return ERROR_SUCCESS;
  }

  void WINAPI fake_sleep(const DWORD milliseconds) {
    fake_query_state.sleeps.push_back(milliseconds);
  }

  struct fake_color_state_t {
    LONG get_status = ERROR_SUCCESS;
    LONG set_status = ERROR_SUCCESS;
    std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> get_types;
    std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> set_types;
    std::vector<std::uint64_t> set_epochs;
  } fake_color_state;

  LONG WINAPI fake_get_device_info(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    fake_color_state.get_types.push_back(header->type);
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2) {
      if (fake_color_state.get_status != ERROR_SUCCESS) {
        return fake_color_state.get_status;
      }
      auto *info = reinterpret_cast<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 *>(header);
      info->advancedColorSupported = 1;
      info->advancedColorActive = 1;
      info->highDynamicRangeSupported = 1;
      info->highDynamicRangeUserEnabled = 1;
      info->wideColorSupported = 1;
      info->wideColorUserEnabled = 0;
      info->bitsPerColorChannel = 10;
      info->activeColorMode = DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR;
      return ERROR_SUCCESS;
    }
    return ERROR_INVALID_PARAMETER;
  }

  LONG WINAPI fake_set_device_info(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    fake_color_state.set_types.push_back(header->type);
    fake_color_state.set_epochs.push_back(platf::display_change_epoch());
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE || header->type == DISPLAYCONFIG_DEVICE_INFO_SET_WCG_STATE) {
      return fake_color_state.set_status;
    }
    return ERROR_INVALID_PARAMETER;
  }

  platf::display_config::query_api_t fake_query_api() {
    return {fake_get_buffer_sizes, fake_query_display_config, fake_sleep};
  }

  platf::display_config::device_info_api_t fake_device_info_api() {
    return {fake_get_device_info, fake_set_device_info};
  }
}  // namespace

TEST(DisplayConfigDevicePath, NamesCompareWithoutCase) {
  using namespace platf::display_config;
  // CCD and SetupAPI spell the same monitor interface path in different cases.
  EXPECT_TRUE(equal_device_paths(LR"(\\?\DISPLAY#GSM5B7F#5&1a2b&0&UID4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7})", LR"(\\?\display#gsm5b7f#5&1a2b&0&uid4352#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7})"));
  EXPECT_TRUE(same_device_path(LR"(\\.\DISPLAY1)", LR"(\\.\display1)"));
  EXPECT_FALSE(equal_device_paths(LR"(\\.\DISPLAY1)", LR"(\\.\DISPLAY12)"));
  EXPECT_FALSE(same_device_path(LR"(\\.\DISPLAY1)", LR"(\\.\DISPLAY2)"));
}

TEST(DisplayConfigDevicePath, AnEmptyNameIsEqualToAnotherButNamesNoDevice) {
  using namespace platf::display_config;
  EXPECT_TRUE(equal_device_paths({}, {}));
  EXPECT_FALSE(equal_device_paths({}, LR"(\\.\DISPLAY1)"));
  EXPECT_FALSE(same_device_path({}, {}));
  EXPECT_FALSE(same_device_path({}, LR"(\\.\DISPLAY1)"));
  EXPECT_FALSE(same_device_path(LR"(\\.\DISPLAY1)", {}));
}

TEST(DisplayConfigRect, ARectangleNeedsAPositiveWidthAndHeightThatFitAnInt) {
  using namespace platf::display_config;
  EXPECT_TRUE(valid_rect({-1920, -200, 0, 880}));
  EXPECT_FALSE(valid_rect({0, 0, 0, 1080}));
  EXPECT_FALSE(valid_rect({0, 0, 1920, 0}));
  EXPECT_FALSE(valid_rect({1920, 0, 0, 1080}));
  const LONG maximum = std::numeric_limits<LONG>::max();
  EXPECT_TRUE(valid_rect({maximum - 1000, maximum - 1000, maximum, maximum}));
  EXPECT_FALSE(valid_rect({std::numeric_limits<LONG>::min(), 0, maximum, 100}));
  EXPECT_FALSE(valid_rect({0, std::numeric_limits<LONG>::min(), 100, maximum}));
}

TEST(DisplayConfigRect, RectanglesAreTheSameOnlyWithEveryEdgeEqual) {
  using namespace platf::display_config;
  EXPECT_TRUE(same_rect({-1920, 0, 0, 1080}, {-1920, 0, 0, 1080}));
  EXPECT_FALSE(same_rect({-1920, 0, 0, 1080}, {-1920, 0, 0, 1081}));
  EXPECT_FALSE(same_rect({-1920, 0, 0, 1080}, {-1919, 0, 0, 1080}));
}

TEST(DisplayConfigTarget, GdiNameMatchesWithoutCase) {
  using namespace platf::display_config;
  target_names = {{1, L"monitor-a", LR"(\\.\DISPLAY1)"}, {2, L"monitor-b", LR"(\\.\DISPLAY2)"}};
  const auto found = find_display_target(target_paths(), {}, LR"(\\.\display2)", {get_target_name, nullptr});
  ASSERT_TRUE(found);
  EXPECT_EQ(found->target_id, 2u);
}

TEST(DisplayConfigTarget, LearnedIdentitySurvivesGdiNameAndTargetRenumbering) {
  using namespace platf::display_config;
  const device_info_api_t api {get_target_name, nullptr};
  target_names = {{1, L"monitor-a", L"DISPLAY1"}, {2, L"monitor-b", L"DISPLAY2"}};
  const auto original = find_display_target(target_paths(), {}, L"DISPLAY1", api);
  ASSERT_TRUE(original);
  ASSERT_EQ(original->device_path, L"monitor-a");

  target_names = {{3, L"MONITOR-A", L"DISPLAY2"}, {4, L"monitor-b", L"DISPLAY1"}};
  const auto restored = find_display_target(target_paths(), original->device_path, L"DISPLAY1", api);
  ASSERT_TRUE(restored);
  EXPECT_EQ(restored->target_id, 3u);
  EXPECT_EQ(restored->display_name, L"DISPLAY2");
}

TEST(DisplayConfigTarget, AbsentOriginalNeverFallsBackToARecycledName) {
  using namespace platf::display_config;
  target_names = {{4, L"replacement", L"DISPLAY1"}};
  EXPECT_FALSE(find_display_target(target_paths(), L"original", L"DISPLAY1", {get_target_name, nullptr}));
}

TEST(DisplayConfigTarget, AmbiguousInitialNameOrLearnedIdentityIsRejected) {
  using namespace platf::display_config;
  target_names = {{1, L"monitor-a", L"DISPLAY1"}, {2, L"monitor-b", L"DISPLAY1"}};
  EXPECT_FALSE(find_display_target(target_paths(), {}, L"DISPLAY1", {get_target_name, nullptr}));
  target_names[1].device_path = L"monitor-a";
  EXPECT_FALSE(find_display_target(target_paths(), L"monitor-a", {}, {get_target_name, nullptr}));
}

TEST(DisplayConfigNamedTopology, ListsEachTargetsMonitorPathInPathOrder) {
  using namespace platf::display_config;
  target_names = {{1, L"monitor-a", L"DISPLAY1"}, {2, L"monitor-b", L"DISPLAY2"}};
  named_topology_t topology;
  const auto result = query_named_topology(QDC_ONLY_ACTIVE_PATHS, topology, unnamed_target_e::rejected, {}, {named_buffer_sizes, named_query, no_sleep}, {get_target_name, nullptr});
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.unnamed_target);
  ASSERT_EQ(topology.paths.size(), 2u);
  EXPECT_EQ(topology.device_paths, (std::vector<std::wstring> {L"monitor-a", L"monitor-b"}));
}

TEST(DisplayConfigNamedTopology, ATargetWithoutAMonitorPathFailsTheQueryUnlessAllowed) {
  using namespace platf::display_config;
  target_names = {{1, L"monitor-a", L"DISPLAY1"}, {2, L"", L"DISPLAY2"}};
  const query_api_t api {named_buffer_sizes, named_query, no_sleep};
  named_topology_t topology;
  const auto rejected = query_named_topology(QDC_ONLY_ACTIVE_PATHS, topology, unnamed_target_e::rejected, {}, api, {get_target_name, nullptr});
  EXPECT_FALSE(rejected);
  EXPECT_EQ(rejected.status, ERROR_NOT_FOUND);
  ASSERT_TRUE(rejected.unnamed_target);
  EXPECT_EQ(rejected.unnamed_target->id, 2u);
  EXPECT_TRUE(topology.paths.empty());
  EXPECT_TRUE(topology.device_paths.empty());

  const auto allowed = query_named_topology(QDC_ALL_PATHS, topology, unnamed_target_e::allowed, {}, api, {get_target_name, nullptr});
  ASSERT_TRUE(allowed);
  EXPECT_EQ(topology.device_paths, (std::vector<std::wstring> {L"monitor-a", L""}));
}

TEST(DisplayConfigNamedTopology, AFailedLookupReportsItsStatusOrLeavesAnAllowedTargetUnnamed) {
  using namespace platf::display_config;
  target_names = {{1, L"monitor-a", L"DISPLAY1"}};
  const query_api_t api {named_buffer_sizes, named_query, no_sleep};
  named_topology_t topology;
  const auto rejected = query_named_topology(QDC_ONLY_ACTIVE_PATHS, topology, unnamed_target_e::rejected, {}, api, {denied_device_info, nullptr});
  EXPECT_EQ(rejected.status, ERROR_ACCESS_DENIED);
  ASSERT_TRUE(rejected.unnamed_target);
  EXPECT_EQ(rejected.unnamed_target->id, 1u);

  const auto allowed = query_named_topology(QDC_ALL_PATHS, topology, unnamed_target_e::allowed, {}, api, {denied_device_info, nullptr});
  ASSERT_TRUE(allowed);
  EXPECT_EQ(topology.device_paths, (std::vector<std::wstring> {L""}));
}

TEST(DisplayConfigNamedTopology, ANameWithoutItsTerminatorIsAFailedLookup) {
  using namespace platf::display_config;
  const auto name = query_target_name({}, 1, {unterminated_target_name, nullptr});
  EXPECT_FALSE(name);
  EXPECT_EQ(name.status, ERROR_INVALID_DATA);
  EXPECT_TRUE(name.device_path.empty());
}

TEST(DisplayConfigSuppliedApply, ValidatesThenAppliesWithTheCallersFlags) {
  using namespace platf::display_config;
  fake_set_config_state = {};
  topology_t topology;
  topology.paths.resize(2);
  const auto result = validate_and_apply(topology, SDC_VIRTUAL_MODE_AWARE, SDC_NO_OPTIMIZATION, {}, fake_set_display_config);
  EXPECT_EQ(result.state, supplied_apply_e::applied);
  EXPECT_EQ(result.status, ERROR_SUCCESS);
  const std::vector<UINT32> expected_flags {
    SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_MODE_AWARE,
    SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_MODE_AWARE | SDC_NO_OPTIMIZATION,
  };
  EXPECT_EQ(fake_set_config_state.flags, expected_flags);
  EXPECT_EQ(fake_set_config_state.path_counts, (std::vector<UINT32> {2, 2}));
}

TEST(DisplayConfigSuppliedApply, AnInvalidTopologyIsNeitherRequeriedNorApplied) {
  using namespace platf::display_config;
  fake_set_config_state = {};
  fake_set_config_state.validate_status = ERROR_INVALID_PARAMETER;
  topology_t topology;
  bool requeried = false;
  const auto requery = [&](topology_t &) {
    requeried = true;
    return true;
  };
  const auto result = validate_and_apply(topology, 0, 0, requery, fake_set_display_config);
  EXPECT_EQ(result.state, supplied_apply_e::invalid);
  EXPECT_EQ(result.status, ERROR_INVALID_PARAMETER);
  EXPECT_FALSE(requeried);
  EXPECT_EQ(fake_set_config_state.flags.size(), 1u);
}

TEST(DisplayConfigSuppliedApply, TheRequeryReplacesTheTopologyOrStopsTheApply) {
  using namespace platf::display_config;
  fake_set_config_state = {};
  topology_t topology;
  topology.paths.resize(1);
  const auto changed_desktop = [](topology_t &) {
    return false;
  };
  const auto stale = validate_and_apply(topology, 0, 0, changed_desktop, fake_set_display_config);
  EXPECT_EQ(stale.state, supplied_apply_e::stale);
  EXPECT_EQ(fake_set_config_state.flags.size(), 1u);

  fake_set_config_state = {};
  fake_set_config_state.apply_status = ERROR_GEN_FAILURE;
  const auto fresh_observation = [](topology_t &fresh) {
    fresh.paths.resize(3);
    return true;
  };
  const auto failed = validate_and_apply(topology, 0, 0, fresh_observation, fake_set_display_config);
  EXPECT_EQ(failed.state, supplied_apply_e::failed);
  EXPECT_EQ(failed.status, ERROR_GEN_FAILURE);
  EXPECT_EQ(fake_set_config_state.path_counts, (std::vector<UINT32> {1, 3}));
}

TEST(DisplayConfigQueryTest, RetriesSizeAndDataRacesWithBoundedBackoff) {
  fake_query_state = {};
  platf::display_config::topology_t topology;

  const auto result = platf::display_config::query_topology(
    QDC_ONLY_ACTIVE_PATHS,
    topology,
    {},
    fake_query_api()
  );

  ASSERT_TRUE(result);
  EXPECT_EQ(fake_query_state.size_calls, 3u);
  EXPECT_EQ(fake_query_state.query_calls, 2u);
  EXPECT_EQ(topology.paths.size(), 1u);
  EXPECT_EQ(topology.modes.size(), 2u);
  EXPECT_EQ(fake_query_state.sleeps, (std::vector<DWORD> {1, 2}));
}

TEST(DisplayConfigQueryTest, StopsAfterTheConfiguredAttemptBudget) {
  fake_query_state = {};
  fake_query_state.never_settles = true;
  platf::display_config::topology_t topology;
  topology.paths.resize(1);
  topology.modes.resize(1);

  const auto result = platf::display_config::query_topology(
    QDC_ONLY_ACTIVE_PATHS,
    topology,
    {3, platf::display_config::retry_delay_e::exponential_milliseconds},
    fake_query_api()
  );

  EXPECT_FALSE(result);
  EXPECT_EQ(result.status, ERROR_INSUFFICIENT_BUFFER);
  EXPECT_EQ(fake_query_state.size_calls, 3u);
  EXPECT_TRUE(topology.paths.empty());
  EXPECT_TRUE(topology.modes.empty());
}

TEST(DisplayConfigQueryTest, ImmediatePolicyPreservesCallerRetryCadence) {
  fake_query_state = {};
  fake_query_state.never_settles = true;
  platf::display_config::topology_t topology;

  const auto result = platf::display_config::query_topology(
    QDC_ONLY_ACTIVE_PATHS,
    topology,
    {4, platf::display_config::retry_delay_e::none},
    fake_query_api()
  );

  EXPECT_FALSE(result);
  EXPECT_EQ(fake_query_state.size_calls, 4u);
  EXPECT_TRUE(fake_query_state.sleeps.empty());
}

TEST(DisplayConfigColorTest, FailedQueryReportsNoStateAndTriesNoOtherApi) {
  // Windows 11 24H2 is the minimum: a failed query has no older API to fall back to. Errors that
  // once selected the pre-24H2 query report no state, as every other failure does.
  for (const LONG status : std::initializer_list<LONG> {ERROR_GEN_FAILURE, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED}) {
    SCOPED_TRACE(status);
    fake_color_state = {};
    fake_color_state.get_status = status;
    EXPECT_FALSE(platf::display_config::query_advanced_color(LUID {}, 7, fake_device_info_api()));
    EXPECT_EQ(fake_color_state.get_types, (std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2}));
  }
}

TEST(DisplayConfigColorTest, ReportsHdrAndWcgFields) {
  fake_color_state = {};
  const auto state = platf::display_config::query_advanced_color(LUID {}, 9, fake_device_info_api());

  ASSERT_TRUE(state);
  EXPECT_TRUE(state->hdr_supported);
  EXPECT_TRUE(state->hdr_user_enabled);
  EXPECT_FALSE(state->wcg_user_enabled);
  EXPECT_TRUE(state->advanced_color_active);
  EXPECT_EQ(state->bits_per_color_channel, 10u);
  EXPECT_EQ(state->active_mode, DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR);
  EXPECT_EQ(fake_color_state.get_types, (std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2}));
}

TEST(DisplayConfigColorTest, RejectedHdrRequestFailsWithoutAnotherRequest) {
  fake_color_state = {};
  fake_color_state.set_status = ERROR_NOT_SUPPORTED;

  EXPECT_FALSE(platf::display_config::set_hdr_state(LUID {}, 11, true, fake_device_info_api()));
  EXPECT_EQ(fake_color_state.set_types, (std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> {DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE}));
}

TEST(DisplayConfigColorTest, ColorRequestsRecordADisplayChangeBeforeAndAfterTheCall) {
  // Advanced Color changes restart Desktop Duplication; capture failover reads this epoch.
  fake_color_state = {};
  const auto before = platf::display_change_epoch();
  EXPECT_TRUE(platf::display_config::set_hdr_state(LUID {}, 11, true, fake_device_info_api()));
  ASSERT_EQ(fake_color_state.set_epochs.size(), 1u);
  // Each request, accepted or rejected, is bracketed by its own changes.
  EXPECT_GT(fake_color_state.set_epochs[0], before);
  EXPECT_GT(platf::display_change_epoch(), fake_color_state.set_epochs[0]);

  // A rejected WCG request is bracketed too: Windows may have applied part of it.
  fake_color_state.set_status = ERROR_NOT_SUPPORTED;
  const auto before_wcg = platf::display_change_epoch();
  EXPECT_FALSE(platf::display_config::set_wcg_state(LUID {}, 11, false, fake_device_info_api()));
  ASSERT_EQ(fake_color_state.set_epochs.size(), 2u);
  EXPECT_GT(fake_color_state.set_epochs[1], before_wcg);
  EXPECT_GT(platf::display_change_epoch(), fake_color_state.set_epochs[1]);

  // Without a setter no request is made, so no change is recorded.
  const auto unchanged = platf::display_change_epoch();
  EXPECT_FALSE(platf::display_config::set_hdr_state(LUID {}, 11, true, {fake_get_device_info, nullptr}));
  EXPECT_EQ(platf::display_change_epoch(), unchanged);
}

namespace {
  using namespace std::chrono_literals;

  struct color_settle_probe_t {
    int looks = 0;
    int pending_looks = 0;
    std::vector<std::chrono::milliseconds> waits;
    bool wait_ok = true;

    platf::display_config::color_change_result_e run(platf::display_config::color_settle_policy_t policy, bool accepted = true, bool fails = false) {
      using platf::display_config::color_observation_e;
      return platf::display_config::change_color_and_settle(
        [accepted]() {
          return accepted;
        },
        [this, fails]() {
          ++looks;
          if (fails) {
            return color_observation_e::failed;
          }
          return looks > pending_looks ? color_observation_e::matched : color_observation_e::pending;
        },
        policy,
        [this](std::chrono::milliseconds delay) {
          waits.push_back(delay);
          return wait_ok;
        }
      );
    }
  };
}  // namespace

TEST(DisplayConfigColorSettle, ARejectedRequestIsNotObserved) {
  color_settle_probe_t probe;
  EXPECT_EQ(probe.run({500ms, 50ms}, false), platf::display_config::color_change_result_e::rejected);
  EXPECT_EQ(probe.looks, 0);
  EXPECT_TRUE(probe.waits.empty());
}

TEST(DisplayConfigColorSettle, LooksAtOnceAndAfterEachPollUntilWindowsReportsTheChange) {
  color_settle_probe_t probe;
  probe.pending_looks = 2;
  EXPECT_EQ(probe.run({500ms, 50ms}), platf::display_config::color_change_result_e::verified);
  EXPECT_EQ(probe.looks, 3);
  EXPECT_EQ(probe.waits, (std::vector<std::chrono::milliseconds> {50ms, 50ms}));

  color_settle_probe_t immediate;
  EXPECT_EQ(immediate.run({500ms, 50ms}), platf::display_config::color_change_result_e::verified);
  EXPECT_EQ(immediate.looks, 1);
  EXPECT_TRUE(immediate.waits.empty());
}

TEST(DisplayConfigColorSettle, TheBoundLimitsTheLooksAndAFailedLookEndsThem) {
  color_settle_probe_t never;
  never.pending_looks = 1000;
  EXPECT_EQ(never.run({500ms, 50ms}), platf::display_config::color_change_result_e::unverified);
  EXPECT_EQ(never.looks, 11);
  EXPECT_EQ(never.waits.size(), 10u);

  // The HDR worker's single 100 ms wait before it makes its request again.
  color_settle_probe_t worker;
  worker.pending_looks = 1000;
  EXPECT_EQ(worker.run({100ms, 100ms}), platf::display_config::color_change_result_e::unverified);
  EXPECT_EQ(worker.looks, 2);
  EXPECT_EQ(worker.waits, (std::vector<std::chrono::milliseconds> {100ms}));

  color_settle_probe_t failed;
  EXPECT_EQ(failed.run({500ms, 50ms}, true, true), platf::display_config::color_change_result_e::unverified);
  EXPECT_EQ(failed.looks, 1);
  EXPECT_TRUE(failed.waits.empty());
}

TEST(DisplayConfigColorSettle, AnInterruptedWaitEndsTheSettle) {
  color_settle_probe_t probe;
  probe.pending_looks = 1000;
  probe.wait_ok = false;
  EXPECT_EQ(probe.run({500ms, 50ms}), platf::display_config::color_change_result_e::interrupted);
  EXPECT_EQ(probe.looks, 1);
  EXPECT_EQ(probe.waits.size(), 1u);
}

/**
 * @file tests/unit/test_display_config.cpp
 * @brief Tests for shared Windows CCD and Advanced Color adapters.
 */
#include "../tests_common.h"

#include "src/platform/windows/display_config.h"

#include <vector>

namespace {
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
    LONG modern_get_status = ERROR_SUCCESS;
    LONG legacy_get_status = ERROR_SUCCESS;
    LONG modern_set_status = ERROR_SUCCESS;
    LONG legacy_set_status = ERROR_SUCCESS;
    unsigned int modern_get_calls = 0;
    unsigned int legacy_get_calls = 0;
    std::vector<DISPLAYCONFIG_DEVICE_INFO_TYPE> set_types;
  } fake_color_state;

  LONG WINAPI fake_get_device_info(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2) {
      ++fake_color_state.modern_get_calls;
      if (fake_color_state.modern_get_status != ERROR_SUCCESS) {
        return fake_color_state.modern_get_status;
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
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO) {
      ++fake_color_state.legacy_get_calls;
      if (fake_color_state.legacy_get_status != ERROR_SUCCESS) {
        return fake_color_state.legacy_get_status;
      }
      auto *info = reinterpret_cast<DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO *>(header);
      info->advancedColorSupported = 1;
      info->advancedColorEnabled = 1;
      info->advancedColorForceDisabled = 0;
      info->bitsPerColorChannel = 8;
      return ERROR_SUCCESS;
    }
    return ERROR_INVALID_PARAMETER;
  }

  LONG WINAPI fake_set_device_info(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
    fake_color_state.set_types.push_back(header->type);
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE ||
        header->type == DISPLAYCONFIG_DEVICE_INFO_SET_WCG_STATE) {
      return fake_color_state.modern_set_status;
    }
    if (header->type == DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE) {
      return fake_color_state.legacy_set_status;
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

TEST(DisplayConfigQueryTest, RetriesSizeAndDataRacesWithBoundedBackoff) {
  fake_query_state = {};
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;

  const auto result = platf::display_config::query_display_config(
    QDC_ONLY_ACTIVE_PATHS,
    paths,
    modes,
    fake_query_api()
  );

  ASSERT_TRUE(result);
  EXPECT_EQ(fake_query_state.size_calls, 3u);
  EXPECT_EQ(fake_query_state.query_calls, 2u);
  EXPECT_EQ(paths.size(), 1u);
  EXPECT_EQ(modes.size(), 2u);
  EXPECT_EQ(fake_query_state.sleeps, (std::vector<DWORD> {1, 2}));
}

TEST(DisplayConfigQueryTest, StopsAfterTheConfiguredAttemptBudget) {
  fake_query_state = {};
  fake_query_state.never_settles = true;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(1);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(1);

  const auto result = platf::display_config::query_display_config(
    QDC_ONLY_ACTIVE_PATHS,
    paths,
    modes,
    fake_query_api(),
    {3, platf::display_config::retry_delay_e::exponential_milliseconds}
  );

  EXPECT_FALSE(result);
  EXPECT_EQ(result.status, ERROR_INSUFFICIENT_BUFFER);
  EXPECT_EQ(fake_query_state.size_calls, 3u);
  EXPECT_TRUE(paths.empty());
  EXPECT_TRUE(modes.empty());
}

TEST(DisplayConfigQueryTest, ImmediatePolicyPreservesCallerRetryCadence) {
  fake_query_state = {};
  fake_query_state.never_settles = true;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths;
  std::vector<DISPLAYCONFIG_MODE_INFO> modes;

  const auto result = platf::display_config::query_display_config(
    QDC_ONLY_ACTIVE_PATHS,
    paths,
    modes,
    fake_query_api(),
    {4, platf::display_config::retry_delay_e::none}
  );

  EXPECT_FALSE(result);
  EXPECT_EQ(fake_query_state.size_calls, 4u);
  EXPECT_TRUE(fake_query_state.sleeps.empty());
}

TEST(DisplayConfigColorTest, FallsBackToLegacyOnlyForTheSelectedFailurePolicy) {
  fake_color_state = {};
  fake_color_state.modern_get_status = ERROR_GEN_FAILURE;
  const LUID adapter {};

  EXPECT_FALSE(platf::display_config::query_advanced_color(
    adapter,
    7,
    platf::display_config::legacy_fallback_e::unsupported_modern_api,
    fake_device_info_api()
  ));
  EXPECT_EQ(fake_color_state.legacy_get_calls, 0u);

  const auto state = platf::display_config::query_advanced_color(
    adapter,
    7,
    platf::display_config::legacy_fallback_e::any_modern_failure,
    fake_device_info_api()
  );
  ASSERT_TRUE(state);
  EXPECT_EQ(state->api, platf::display_config::advanced_color_api_e::legacy);
  EXPECT_TRUE(state->advanced_color_enabled);
  EXPECT_TRUE(state->hdr_user_enabled);
  EXPECT_EQ(fake_color_state.legacy_get_calls, 1u);
}

TEST(DisplayConfigColorTest, ReportsModernHdrAndWcgFieldsWithoutLegacyQuery) {
  fake_color_state = {};
  const auto state = platf::display_config::query_advanced_color(
    LUID {},
    9,
    platf::display_config::legacy_fallback_e::unsupported_modern_api,
    fake_device_info_api()
  );

  ASSERT_TRUE(state);
  EXPECT_EQ(state->api, platf::display_config::advanced_color_api_e::modern);
  EXPECT_TRUE(state->hdr_supported);
  EXPECT_TRUE(state->hdr_user_enabled);
  EXPECT_FALSE(state->wcg_user_enabled);
  EXPECT_EQ(state->active_mode, DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR);
  EXPECT_EQ(fake_color_state.legacy_get_calls, 0u);
}

TEST(DisplayConfigColorTest, HdrSetterFallsBackToLegacyAfterModernFailure) {
  fake_color_state = {};
  fake_color_state.modern_set_status = ERROR_NOT_SUPPORTED;

  EXPECT_TRUE(platf::display_config::set_hdr_state_with_legacy_fallback(
    LUID {},
    11,
    true,
    fake_device_info_api()
  ));
  ASSERT_EQ(fake_color_state.set_types.size(), 2u);
  EXPECT_EQ(fake_color_state.set_types[0], DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE);
  EXPECT_EQ(
    fake_color_state.set_types[1],
    DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE
  );
}

/**
 * @file tests/unit/platform/test_wgc_cursor.cpp
 * @brief Exercise WGC cursor configuration against an in-process projected WinRT session.
 */

#include <gtest/gtest.h>

#include "src/platform/windows/display.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>

namespace platf::dxgi {
  struct wgc_capture_test_access {
    static void assign_session(wgc_capture_t &capture, winrt::Windows::Graphics::Capture::GraphicsCaptureSession session) {
      capture.assign_capture_session(std::move(session));
    }

    static bool has_frame_pool(const wgc_capture_t &capture) {
      return capture.frame_pool != nullptr;
    }
  };
}  // namespace platf::dxgi

namespace {
  using platf::dxgi::wgc_capture_t;
  using access_t = platf::dxgi::wgc_capture_test_access;
  using namespace winrt::Windows::Graphics::Capture;

  struct session_stats_t {
    bool cursor_enabled = true;
    bool change_before_setter_failure = false;
    HRESULT getter_result = S_OK;
    HRESULT setter_result = S_OK;
    HRESULT close_result = S_OK;
    unsigned cursor_queries = 0;
    unsigned getters = 0;
    unsigned setters = 0;
    unsigned starts = 0;
    unsigned closes = 0;
    unsigned destructions = 0;
  };

  struct capture_session_t: winrt::implements<capture_session_t, IGraphicsCaptureSession, IGraphicsCaptureSession2, winrt::Windows::Foundation::IClosable> {
    using base_t = winrt::implements<capture_session_t, IGraphicsCaptureSession, IGraphicsCaptureSession2, winrt::Windows::Foundation::IClosable>;

    explicit capture_session_t(std::shared_ptr<session_stats_t> stats):
        stats_ {std::move(stats)} {
    }

    ~capture_session_t() {
      ++stats_->destructions;
    }

    void *find_interface(const winrt::guid &id) const noexcept override {
      if (id == winrt::guid_of<IGraphicsCaptureSession2>()) {
        ++stats_->cursor_queries;
      }
      return base_t::find_interface(id);
    }

    bool IsCursorCaptureEnabled() {
      ++stats_->getters;
      winrt::check_hresult(stats_->getter_result);
      return stats_->cursor_enabled;
    }

    void IsCursorCaptureEnabled(bool enabled) {
      ++stats_->setters;
      if (!stats_->change_before_setter_failure) {
        winrt::check_hresult(stats_->setter_result);
      }
      stats_->cursor_enabled = enabled;
      winrt::check_hresult(stats_->setter_result);
    }

    void StartCapture() {
      ++stats_->starts;
      ADD_FAILURE() << "The cursor fixture must never start real capture";
    }

    void Close() {
      ++stats_->closes;
      winrt::check_hresult(stats_->close_result);
    }

  private:
    std::shared_ptr<session_stats_t> stats_;
  };

  GraphicsCaptureSession make_session(const std::shared_ptr<session_stats_t> &stats) {
    return winrt::make_self<capture_session_t>(stats).as<GraphicsCaptureSession>();
  }

  class WgcCursorInitialStateTest: public testing::TestWithParam<std::pair<bool, bool>> {};

  TEST_P(WgcCursorInitialStateTest, ReadsInitialStateThenSkipsOneThousandStableRequests) {
    const auto [initial, desired] = GetParam();
    auto stats = std::make_shared<session_stats_t>();
    stats->cursor_enabled = initial;
    {
      wgc_capture_t capture;
      access_t::assign_session(capture, make_session(stats));
      EXPECT_FALSE(access_t::has_frame_pool(capture));

      ASSERT_EQ(capture.set_cursor_visible(desired), 0);
      EXPECT_EQ(stats->cursor_enabled, desired);
      EXPECT_EQ(stats->getters, 1u);
      EXPECT_EQ(stats->setters, initial == desired ? 0u : 1u);
      ASSERT_GT(stats->cursor_queries, 0u);
      const auto queries = stats->cursor_queries;
      const auto getters = stats->getters;
      const auto setters = stats->setters;
      for (int request = 0; request < 1000; ++request) {
        ASSERT_EQ(capture.set_cursor_visible(desired), 0);
      }
      RecordProperty("stable_requests", 1000);
      RecordProperty("stable_getter_calls", std::to_string(stats->getters - getters));
      RecordProperty("stable_setter_calls", std::to_string(stats->setters - setters));
      RecordProperty("stable_cursor_interface_queries", std::to_string(stats->cursor_queries - queries));
      RecordProperty("total_getter_calls", std::to_string(stats->getters));
      RecordProperty("total_setter_calls", std::to_string(stats->setters));
      EXPECT_EQ(stats->cursor_queries, queries);
      EXPECT_EQ(stats->getters, 1u);
      EXPECT_EQ(stats->setters, initial == desired ? 0u : 1u);
      EXPECT_EQ(stats->starts, 0u);
      EXPECT_FALSE(access_t::has_frame_pool(capture));
    }
    EXPECT_EQ(stats->closes, 1u);
    EXPECT_EQ(stats->destructions, 1u);
  }

  INSTANTIATE_TEST_SUITE_P(
    InitialAndRequestedState,
    WgcCursorInitialStateTest,
    testing::Values(std::pair {false, false}, std::pair {false, true}, std::pair {true, false}, std::pair {true, true}),
    [](const auto &info) {
      return std::string {info.param.first ? "InitiallyOn" : "InitiallyOff"} + (info.param.second ? "RequestOn" : "RequestOff");
    }
  );

  TEST(WgcCursorTest, TogglesQueryActualStateAndCacheOnlyTheLastSuccessfulRequest) {
    auto stats = std::make_shared<session_stats_t>();
    {
      wgc_capture_t capture;
      access_t::assign_session(capture, make_session(stats));
      EXPECT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_EQ(capture.set_cursor_visible(false), 0);
      EXPECT_FALSE(stats->cursor_enabled);
      EXPECT_EQ(stats->getters, 2u);
      EXPECT_EQ(stats->setters, 1u);
      const auto queries = stats->cursor_queries;
      for (int request = 0; request < 1000; ++request) {
        ASSERT_EQ(capture.set_cursor_visible(false), 0);
      }
      EXPECT_EQ(stats->cursor_queries, queries);
      EXPECT_EQ(stats->getters, 2u);
      EXPECT_EQ(stats->setters, 1u);
      EXPECT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_TRUE(stats->cursor_enabled);
      EXPECT_EQ(stats->getters, 3u);
      EXPECT_EQ(stats->setters, 2u);
      EXPECT_EQ(stats->starts, 0u);
    }
    EXPECT_EQ(stats->closes, 1u);
  }

  TEST(WgcCursorTest, UnknownGetterFailureRetriesWithoutPublishingTheRequestedState) {
    auto stats = std::make_shared<session_stats_t>();
    stats->getter_result = E_ACCESSDENIED;
    wgc_capture_t capture;
    access_t::assign_session(capture, make_session(stats));

    EXPECT_EQ(capture.set_cursor_visible(false), -1);
    EXPECT_EQ(stats->getters, 1u);
    EXPECT_EQ(stats->setters, 0u);
    EXPECT_TRUE(stats->cursor_enabled);
    stats->getter_result = S_OK;
    EXPECT_EQ(capture.set_cursor_visible(false), 0);
    EXPECT_EQ(stats->getters, 2u);
    EXPECT_EQ(stats->setters, 1u);
    EXPECT_FALSE(stats->cursor_enabled);
  }

  TEST(WgcCursorTest, GetterFailureOnAChangedRequestInvalidatesThePreviouslyKnownValue) {
    auto stats = std::make_shared<session_stats_t>();
    wgc_capture_t capture;
    access_t::assign_session(capture, make_session(stats));
    ASSERT_EQ(capture.set_cursor_visible(true), 0);

    stats->getter_result = E_FAIL;
    EXPECT_EQ(capture.set_cursor_visible(false), -1);
    EXPECT_EQ(stats->getters, 2u);
    EXPECT_EQ(stats->setters, 0u);
    stats->getter_result = S_OK;
    // Even the previously successful value must query again after the intervening failure.
    EXPECT_EQ(capture.set_cursor_visible(true), 0);
    EXPECT_EQ(stats->getters, 3u);
    EXPECT_EQ(stats->setters, 0u);
  }

  class WgcCursorSetterFailureTest: public testing::TestWithParam<bool> {};

  TEST_P(WgcCursorSetterFailureTest, InvalidatesPriorStateAndRequeriesBeforeRetryingAPossiblyAppliedWrite) {
    auto stats = std::make_shared<session_stats_t>();
    stats->change_before_setter_failure = GetParam();
    wgc_capture_t capture;
    access_t::assign_session(capture, make_session(stats));
    ASSERT_EQ(capture.set_cursor_visible(true), 0);

    stats->setter_result = E_ACCESSDENIED;
    EXPECT_EQ(capture.set_cursor_visible(false), -1);
    EXPECT_EQ(stats->getters, 2u);
    EXPECT_EQ(stats->setters, 1u);
    EXPECT_EQ(stats->cursor_enabled, !GetParam());
    stats->setter_result = S_OK;
    EXPECT_EQ(capture.set_cursor_visible(true), 0);
    EXPECT_EQ(stats->getters, 3u);
    EXPECT_EQ(stats->setters, GetParam() ? 2u : 1u);
    EXPECT_TRUE(stats->cursor_enabled);

    stats->setter_result = E_ACCESSDENIED;
    EXPECT_EQ(capture.set_cursor_visible(false), -1);
    stats->setter_result = S_OK;
    EXPECT_EQ(capture.set_cursor_visible(false), 0);
    EXPECT_EQ(stats->getters, 5u);
    // A failed write that applied the value needs no duplicate setter on its retry.
    EXPECT_EQ(stats->setters, 3u);
    EXPECT_FALSE(stats->cursor_enabled);
    const auto queries = stats->cursor_queries;
    EXPECT_EQ(capture.set_cursor_visible(false), 0);
    EXPECT_EQ(stats->cursor_queries, queries);
    EXPECT_EQ(stats->getters, 5u);
    EXPECT_EQ(stats->setters, 3u);
  }

  INSTANTIATE_TEST_SUITE_P(
    FailureTiming,
    WgcCursorSetterFailureTest,
    testing::Bool(),
    [](const auto &info) {
      return info.param ? "AfterStateChange" : "BeforeStateChange";
    }
  );

  TEST(WgcCursorTest, SessionReplacementNullResetAndSameSessionReassignmentInvalidateConfiguration) {
    auto first_stats = std::make_shared<session_stats_t>();
    auto second_stats = std::make_shared<session_stats_t>();
    second_stats->cursor_enabled = false;
    const auto first = make_session(first_stats);
    const auto second = make_session(second_stats);
    {
      wgc_capture_t capture;
      EXPECT_EQ(capture.set_cursor_visible(true), -1);
      access_t::assign_session(capture, first);
      ASSERT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_EQ(first_stats->getters, 1u);

      access_t::assign_session(capture, second);
      EXPECT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_EQ(second_stats->getters, 1u);
      EXPECT_EQ(second_stats->setters, 1u);
      EXPECT_TRUE(second_stats->cursor_enabled);
      EXPECT_EQ(first_stats->getters, 1u);

      const auto queries = second_stats->cursor_queries;
      access_t::assign_session(capture, nullptr);
      EXPECT_EQ(capture.set_cursor_visible(true), -1);
      EXPECT_EQ(second_stats->cursor_queries, queries);
      EXPECT_EQ(second_stats->getters, 1u);
      access_t::assign_session(capture, second);
      EXPECT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_EQ(second_stats->getters, 2u);
      EXPECT_EQ(second_stats->setters, 1u);

      access_t::assign_session(capture, second);
      EXPECT_EQ(capture.set_cursor_visible(true), 0);
      EXPECT_EQ(second_stats->getters, 3u);
      EXPECT_EQ(second_stats->setters, 1u);
      EXPECT_FALSE(access_t::has_frame_pool(capture));
      EXPECT_EQ(first_stats->starts, 0u);
      EXPECT_EQ(second_stats->starts, 0u);
    }
    EXPECT_EQ(second_stats->closes, 1u);
  }

  TEST(WgcCursorTest, DestructorClosesTheFakeSessionEvenWhenCloseReportsFailure) {
    auto stats = std::make_shared<session_stats_t>();
    stats->close_result = E_FAIL;
    {
      wgc_capture_t capture;
      access_t::assign_session(capture, make_session(stats));
      ASSERT_EQ(capture.set_cursor_visible(false), 0);
      EXPECT_FALSE(access_t::has_frame_pool(capture));
    }
    EXPECT_EQ(stats->closes, 1u);
    EXPECT_EQ(stats->destructions, 1u);
    EXPECT_EQ(stats->starts, 0u);
  }
}  // namespace

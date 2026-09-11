#include "src/session_resume_lifecycle.h"

#include <gtest/gtest.h>

using namespace std::chrono_literals;

namespace {
  using state_t = session_lifecycle::resume_state_t;
  constexpr state_t::time_point origin {};
}  // namespace

TEST(SessionResumeLifecycle, DisconnectRestorationRetriesKeepTheOriginalSixtySecondDeadline) {
  state_t state;
  state.activate();
  state.disconnect(origin, 60s);
  ASSERT_TRUE(state.retained());
  state.disconnect(origin + 20s, 60s);
  state.disconnect(origin + 59s, 60s);
  EXPECT_EQ(state.deadline(), origin + 60s);
  EXPECT_FALSE(state.expired(origin + 59999ms));
  EXPECT_TRUE(state.expired(origin + 60s));
}

TEST(SessionResumeLifecycle, SuccessfulResumeInvalidatesOldExpiryAndNextDisconnectGetsANewWindow) {
  state_t state;
  state.activate();
  state.disconnect(origin, 60s);
  state.activate();
  EXPECT_FALSE(state.retained());
  EXPECT_FALSE(state.deadline());
  EXPECT_FALSE(state.expired(origin + 2min));
  state.disconnect(origin + 2min, 60s);
  EXPECT_EQ(state.deadline(), origin + 3min);
}

TEST(SessionResumeLifecycle, FreshAcceptedConnectionReplacesRetainedGraceWithBoundedHandshake) {
  state_t state;
  state.activate();
  state.disconnect(origin, 60s);
  state.clear();
  EXPECT_EQ(state.phase(), session_lifecycle::phase_e::idle);
  EXPECT_FALSE(state.deadline());
  state.await_connection(origin + 10s, 15s);
  EXPECT_EQ(state.phase(), session_lifecycle::phase_e::connecting);
  EXPECT_FALSE(state.retained());
  EXPECT_EQ(state.deadline(), origin + 25s);
  EXPECT_TRUE(state.expired(origin + 25s));
}

TEST(SessionResumeLifecycle, DisabledGraceAndInvalidNegativeTimeoutExpireImmediately) {
  state_t state;
  state.activate();
  state.disconnect(origin, 0ms);
  EXPECT_TRUE(state.expired(origin));
  state.await_connection(origin, -1s);
  EXPECT_EQ(state.deadline(), origin);
  EXPECT_TRUE(state.expired(origin));
}

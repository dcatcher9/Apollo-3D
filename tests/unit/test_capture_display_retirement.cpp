/**
 * @file tests/unit/test_capture_display_retirement.cpp
 * @brief Exercise original-controlblock capture-display retirement without platform capture.
 */
#include "src/capture_display_retirement.h"
#include "src/platform/common.h"

#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <latch>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace {
  using namespace std::chrono_literals;
  using retirement_t = platf::capture_display_retirement_t;

  struct destruction_state_t {
    std::atomic<unsigned> destroyed {0};
    std::weak_ptr<int> resource;
    std::mutex mutex;
    std::thread::id thread;

    std::thread::id destruction_thread() {
      std::lock_guard lock {mutex};
      return thread;
    }
  };

  // Like display_vram_t, shared_from_this is inherited by the concrete allocated class.
  class shared_display_base_t: public platf::display_t, public std::enable_shared_from_this<shared_display_base_t> {
  public:
    platf::capture_e capture(const push_captured_image_cb_t &, const pull_free_image_cb_t &, bool *) override {
      ADD_FAILURE() << "Retirement tests must not enter platform capture";
      return platf::capture_e::error;
    }

    std::shared_ptr<platf::img_t> alloc_img() override {
      ADD_FAILURE() << "Retirement tests must not allocate capture images";
      return {};
    }

    int dummy_img(platf::img_t *) override {
      ADD_FAILURE() << "Retirement tests must not fill capture images";
      return -1;
    }
  };

  class concrete_display_t final: public shared_display_base_t {
  public:
    explicit concrete_display_t(std::shared_ptr<destruction_state_t> state):
        state_(std::move(state)),
        resource_(std::make_shared<int>(42)) {
      state_->resource = resource_;
    }

    ~concrete_display_t() override {
      {
        std::lock_guard lock {state_->mutex};
        state_->thread = std::this_thread::get_id();
      }
      ++state_->destroyed;
    }

  private:
    std::shared_ptr<destruction_state_t> state_;
    std::shared_ptr<int> resource_;
  };
}  // namespace

TEST(CaptureDisplayRetirementTest, PreservesInheritedSharedFromThisAndOriginalOwnerIdentity) {
  const auto state = std::make_shared<destruction_state_t>();
  auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
  std::shared_ptr<platf::display_t> erased = display;
  auto inherited = display->shared_from_this();
  EXPECT_EQ(inherited.get(), static_cast<shared_display_base_t *>(display.get()));
  EXPECT_FALSE(display.owner_before(inherited));
  EXPECT_FALSE(inherited.owner_before(display));
  EXPECT_FALSE(erased.owner_before(inherited));
  EXPECT_FALSE(inherited.owner_before(erased));
  EXPECT_EQ(std::get_deleter<platf::capture_display_deleter_t>(display), std::get_deleter<platf::capture_display_deleter_t>(erased));

  const auto retirement = std::make_shared<retirement_t>();
  ASSERT_TRUE(platf::arm_capture_display_retirement(erased, retirement));
  display.reset();
  erased.reset();
  EXPECT_FALSE(retirement->wait_for(0ms));
  EXPECT_EQ(state->destroyed.load(), 0u);
  inherited.reset();
  auto returned = retirement->wait_for(2s);
  ASSERT_TRUE(returned);
  EXPECT_EQ(state->destroyed.load(), 0u);
  returned.reset();
  EXPECT_EQ(state->destroyed.load(), 1u);
  EXPECT_TRUE(state->resource.expired());
}

TEST(CaptureDisplayRetirementTest, LastForeignOwnerReturnsRawBeforeCaptureThreadDestruction) {
  const auto state = std::make_shared<destruction_state_t>();
  const auto retirement = std::make_shared<retirement_t>();
  auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
  auto *const expected = static_cast<platf::display_t *>(display.get());
  std::weak_ptr<platf::display_t> weak_display = display;
  ASSERT_TRUE(platf::arm_capture_display_retirement(display, retirement));
  auto foreign_owner = display->shared_from_this();
  display.reset();
  EXPECT_FALSE(retirement->wait_for(0ms));

  std::jthread releaser([foreign_owner = std::move(foreign_owner)]() mutable {
    foreign_owner.reset();
  });
  const auto releasing_thread = releaser.get_id();
  auto returned = retirement->wait_for(2s);
  releaser.join();
  ASSERT_TRUE(returned);
  EXPECT_EQ(returned.get(), expected);
  EXPECT_TRUE(weak_display.expired());
  EXPECT_EQ(state->destroyed.load(), 0u);
  EXPECT_FALSE(state->resource.expired());

  const auto capture_thread = std::this_thread::get_id();
  returned.reset();
  EXPECT_EQ(state->destroyed.load(), 1u);
  EXPECT_EQ(state->destruction_thread(), capture_thread);
  EXPECT_NE(state->destruction_thread(), releasing_thread);
  EXPECT_TRUE(state->resource.expired());
}

TEST(CaptureDisplayRetirementTest, DelayedUnarmedFailureCannotAcknowledgeSelectedFallback) {
  const auto failed_state = std::make_shared<destruction_state_t>();
  const auto selected_state = std::make_shared<destruction_state_t>();
  const auto retirement = std::make_shared<retirement_t>();
  auto failed = platf::make_unarmed_capture_display<concrete_display_t>(failed_state);
  auto failed_borrower = failed->shared_from_this();
  failed.reset();
  auto selected = platf::make_unarmed_capture_display<concrete_display_t>(selected_state);
  auto *const expected = static_cast<platf::display_t *>(selected.get());
  ASSERT_TRUE(platf::arm_capture_display_retirement(selected, retirement));

  // Initialization failure leaves a candidate unarmed, even if a temporary owner releases late.
  failed_borrower.reset();
  EXPECT_EQ(failed_state->destroyed.load(), 1u);
  EXPECT_FALSE(retirement->wait_for(0ms));
  EXPECT_EQ(selected_state->destroyed.load(), 0u);

  selected.reset();
  auto returned = retirement->wait_for(2s);
  ASSERT_TRUE(returned);
  EXPECT_EQ(returned.get(), expected);
  returned.reset();
  EXPECT_EQ(selected_state->destroyed.load(), 1u);
}

TEST(CaptureDisplayRetirementTest, RejectsRearmingOneControlBlockAndSharingOneSlot) {
  const auto first_slot = std::make_shared<retirement_t>();
  const auto second_slot = std::make_shared<retirement_t>();
  const auto first_state = std::make_shared<destruction_state_t>();
  const auto second_state = std::make_shared<destruction_state_t>();
  auto first = platf::make_unarmed_capture_display<concrete_display_t>(first_state);
  auto second = platf::make_unarmed_capture_display<concrete_display_t>(second_state);
  auto *const first_identity = static_cast<platf::display_t *>(first.get());
  auto *const second_identity = static_cast<platf::display_t *>(second.get());
  ASSERT_TRUE(platf::arm_capture_display_retirement(first, first_slot));
  EXPECT_FALSE(platf::arm_capture_display_retirement(first, first_slot));
  EXPECT_FALSE(platf::arm_capture_display_retirement(first, second_slot));
  EXPECT_FALSE(platf::arm_capture_display_retirement(second, first_slot));
  ASSERT_TRUE(platf::arm_capture_display_retirement(second, second_slot));

  first.reset();
  second.reset();
  auto first_returned = first_slot->wait_for(2s);
  auto second_returned = second_slot->wait_for(2s);
  ASSERT_TRUE(first_returned);
  ASSERT_TRUE(second_returned);
  EXPECT_EQ(first_returned.get(), first_identity);
  EXPECT_EQ(second_returned.get(), second_identity);
  first_returned.reset();
  second_returned.reset();
  EXPECT_EQ(first_state->destroyed.load(), 1u);
  EXPECT_EQ(second_state->destroyed.load(), 1u);
}

TEST(CaptureDisplayRetirementTest, RejectsOrdinaryExpiredAndCancelledArmingWithoutTakingOwnership) {
  const auto ordinary_state = std::make_shared<destruction_state_t>();
  auto ordinary = std::make_shared<concrete_display_t>(ordinary_state);
  const auto retirement = std::make_shared<retirement_t>();
  EXPECT_FALSE(platf::arm_capture_display_retirement(ordinary, retirement));
  EXPECT_FALSE(platf::arm_capture_display_retirement({}, retirement));

  const auto unarmed_state = std::make_shared<destruction_state_t>();
  auto unarmed = platf::make_unarmed_capture_display<concrete_display_t>(unarmed_state);
  EXPECT_FALSE(platf::arm_capture_display_retirement(unarmed, {}));
  retirement->cancel();
  EXPECT_FALSE(platf::arm_capture_display_retirement(unarmed, retirement));
  ordinary.reset();
  unarmed.reset();
  EXPECT_EQ(ordinary_state->destroyed.load(), 1u);
  EXPECT_EQ(unarmed_state->destroyed.load(), 1u);
  EXPECT_FALSE(retirement->detach());
}

TEST(CaptureDisplayRetirementTest, CancellationRetainsOwnershipForDetachBeforeOrAfterReturn) {
  for (const bool cancel_before_return : {true, false}) {
    SCOPED_TRACE(cancel_before_return);
    const auto state = std::make_shared<destruction_state_t>();
    const auto retirement = std::make_shared<retirement_t>();
    auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
    auto *const expected = static_cast<platf::display_t *>(display.get());
    ASSERT_TRUE(platf::arm_capture_display_retirement(display, retirement));
    if (cancel_before_return) {
      retirement->cancel();
    }
    display.reset();
    retirement->cancel();
    EXPECT_FALSE(retirement->wait_for(0ms));
    EXPECT_EQ(state->destroyed.load(), 0u);
    auto returned = retirement->detach();
    ASSERT_TRUE(returned);
    EXPECT_EQ(returned.get(), expected);
    EXPECT_FALSE(retirement->detach());
    returned.reset();
    EXPECT_EQ(state->destroyed.load(), 1u);
    EXPECT_TRUE(state->resource.expired());
  }
}

TEST(CaptureDisplayRetirementTest, CancellationWakesAnActualWaitingConsumer) {
  const auto retirement = std::make_shared<retirement_t>();
  std::promise<void> entered;
  auto entered_signal = entered.get_future();
  std::promise<bool> completed;
  auto completion = completed.get_future();
  std::jthread consumer([&]() {
    entered.set_value();
    // A bounded backstop keeps assertion failure cleanup independent of notification correctness.
    completed.set_value(static_cast<bool>(retirement->wait_for(5s)));
  });
  ASSERT_EQ(entered_signal.wait_for(2s), std::future_status::ready);
  retirement->cancel();
  EXPECT_EQ(completion.wait_for(2s), std::future_status::ready);
  consumer.join();
  EXPECT_FALSE(completion.get());
}

TEST(CaptureDisplayRetirementTest, DetachedAndExpiredSlotsDeleteLateOwnersOnTheReleasingThread) {
  for (const bool expire_slot : {false, true}) {
    SCOPED_TRACE(expire_slot);
    const auto state = std::make_shared<destruction_state_t>();
    auto retirement = std::make_shared<retirement_t>();
    const auto weak_slot = std::weak_ptr<retirement_t> {retirement};
    auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
    ASSERT_TRUE(platf::arm_capture_display_retirement(display, retirement));
    EXPECT_FALSE(retirement->detach());
    if (expire_slot) {
      retirement.reset();
      EXPECT_TRUE(weak_slot.expired());
    }
    std::jthread releaser([display = std::move(display)]() mutable {
      display.reset();
    });
    const auto releasing_thread = releaser.get_id();
    releaser.join();
    EXPECT_EQ(state->destroyed.load(), 1u);
    EXPECT_EQ(state->destruction_thread(), releasing_thread);
    EXPECT_TRUE(state->resource.expired());
    if (retirement) {
      EXPECT_FALSE(retirement->wait_for(0ms));
      EXPECT_FALSE(retirement->detach());
    }
  }
}

TEST(CaptureDisplayRetirementTest, SlotDestructionCleansReturnedDisplayWithoutControlBlockCycle) {
  const auto state = std::make_shared<destruction_state_t>();
  auto retirement = std::make_shared<retirement_t>();
  const auto weak_slot = std::weak_ptr<retirement_t> {retirement};
  auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
  const auto weak_display = std::weak_ptr<platf::display_t> {display};
  ASSERT_TRUE(platf::arm_capture_display_retirement(display, retirement));
  display.reset();
  EXPECT_TRUE(weak_display.expired());
  EXPECT_EQ(state->destroyed.load(), 0u);
  retirement.reset();
  EXPECT_TRUE(weak_slot.expired());
  EXPECT_EQ(state->destroyed.load(), 1u);
  EXPECT_TRUE(state->resource.expired());
}

TEST(CaptureDisplayRetirementTest, ConcurrentCancellationDetachAndLastReleaseDestroyExactlyOnce) {
  for (unsigned iteration = 0; iteration < 32; ++iteration) {
    SCOPED_TRACE(iteration);
    const auto state = std::make_shared<destruction_state_t>();
    const auto retirement = std::make_shared<retirement_t>();
    auto display = platf::make_unarmed_capture_display<concrete_display_t>(state);
    ASSERT_TRUE(platf::arm_capture_display_retirement(display, retirement));
    std::latch start {1};
    std::jthread releaser([display = std::move(display), &start]() mutable {
      start.wait();
      display.reset();
    });
    const auto releasing_thread = releaser.get_id();
    retirement->cancel();
    start.count_down();
    auto returned = retirement->detach();
    returned.reset();
    releaser.join();
    EXPECT_EQ(state->destroyed.load(), 1u);
    EXPECT_TRUE(state->resource.expired());
    const auto destruction_thread = state->destruction_thread();
    EXPECT_TRUE(destruction_thread == std::this_thread::get_id() || destruction_thread == releasing_thread);
    EXPECT_FALSE(retirement->wait_for(0ms));
    EXPECT_FALSE(retirement->detach());
  }
}

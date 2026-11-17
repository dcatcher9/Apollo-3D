/** Exercise the actual private capture-generation authority without a capture worker or GPU. */
#include "src/capture_display_retirement.h"
#include "src/platform/common.h"
#include "src/utility.h"
#include "src/video_capture_generation.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {
  using namespace std::chrono_literals;
  using handoff_t = video::detail::capture_generation_handoff_t;

  class bounded_gate_t {
  public:
    void open() {
      std::lock_guard lock {mutex_};
      open_ = true;
      changed_.notify_all();
    }

    bool wait() {
      std::unique_lock lock {mutex_};
      return changed_.wait_for(lock, 3s, [&] {
        return open_;
      });
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool open_ = false;
  };

  struct lifetime_t {
    std::atomic<unsigned> destroyed {0};
    std::function<void()> on_destroy;
  };

  class inert_display_t final: public platf::display_t, public std::enable_shared_from_this<inert_display_t> {
  public:
    explicit inert_display_t(std::shared_ptr<lifetime_t> lifetime):
        lifetime_(std::move(lifetime)) {
    }

    ~inert_display_t() override {
      if (lifetime_->on_destroy) {
        lifetime_->on_destroy();
      }
      ++lifetime_->destroyed;
    }

    platf::capture_e capture(const push_captured_image_cb_t &, const pull_free_image_cb_t &, bool *) override {
      ADD_FAILURE() << "The generation authority must not capture";
      return platf::capture_e::error;
    }

    std::shared_ptr<platf::img_t> alloc_img() override {
      ADD_FAILURE() << "The generation authority must not allocate images";
      return {};
    }

    int dummy_img(platf::img_t *) override {
      ADD_FAILURE() << "The generation authority must not initialize an encoder";
      return -1;
    }

  private:
    std::shared_ptr<lifetime_t> lifetime_;
  };

  std::shared_ptr<inert_display_t> make_display(
    const std::shared_ptr<handoff_t::generation_t> &generation,
    const std::shared_ptr<lifetime_t> &lifetime
  ) {
    if (!generation) {
      throw std::logic_error("The actual handoff failed to begin a generation");
    }
    auto display = platf::make_unarmed_capture_display<inert_display_t>(lifetime);
    if (!platf::arm_capture_display_retirement(display, generation->retirement())) {
      throw std::logic_error("The actual display factory helper failed to arm");
    }
    return display;
  }
}  // namespace

TEST(VideoCaptureGenerationTest, AcquireAndRevokeShareOneConcurrentAdmissionBoundary) {
  handoff_t handoff;
  const auto generation = handoff.begin();
  ASSERT_TRUE(generation);
  auto display = make_display(generation, std::make_shared<lifetime_t>());
  ASSERT_TRUE(handoff.publish(generation, display));

  constexpr unsigned count = 4;
  std::array<handoff_t::display_lease_t, count> raced;
  std::atomic<bool> incorrect {false};
  std::atomic<unsigned> entered_count {0};
  bounded_gate_t entered, start, revoked;
  std::array<std::jthread, count> borrowers;
  auto release_borrowers = util::fail_guard([&] {
    start.open();
    revoked.open();
  });
  for (unsigned index = 0; index < count; ++index) {
    borrowers[index] = std::jthread([&, index] {
      if (entered_count.fetch_add(1) + 1 == count) {
        entered.open();
      }
      if (!start.wait()) {
        incorrect = true;
        return;
      }
      raced[index] = handoff.acquire();
      if (!revoked.wait()) {
        incorrect = true;
        return;
      }
      if (raced[index] && (!raced[index].revoked() || raced[index].display().get() != display.get())) {
        incorrect = true;
      }
      for (unsigned attempt = 0; attempt < 100; ++attempt) {
        if (handoff.acquire()) {
          incorrect = true;
        }
      }
    });
  }
  EXPECT_TRUE(entered.wait());
  start.open();
  const bool accepted = handoff.revoke(generation);
  revoked.open();
  for (auto &borrower : borrowers) {
    borrower.join();
  }
  EXPECT_TRUE(accepted);
  EXPECT_FALSE(incorrect.load());
  EXPECT_FALSE(handoff.acquire());
}

TEST(VideoCaptureGenerationTest, LeaseKeepsOriginalOwnershipAndStickyRevocationAcrossGenerations) {
  handoff_t handoff;
  const auto first = handoff.begin();
  auto lifetime = std::make_shared<lifetime_t>();
  auto display = make_display(first, lifetime);
  const auto weak_display = std::weak_ptr<inert_display_t> {display};
  ASSERT_TRUE(handoff.publish(first, display));
  auto lease = handoff.acquire();
  ASSERT_TRUE(lease);
  EXPECT_FALSE(lease.revoked());
  EXPECT_FALSE(lease.display().owner_before(display));
  EXPECT_FALSE(display.owner_before(lease.display()));
  display.reset();
  EXPECT_FALSE(weak_display.expired());
  ASSERT_TRUE(handoff.revoke(first));
  const auto next = handoff.begin();
  ASSERT_TRUE(next);
  EXPECT_TRUE(lease.revoked());
  EXPECT_TRUE(lease);
  EXPECT_FALSE(first->retirement()->wait_for(0ms));
  lease = {};
  EXPECT_TRUE(weak_display.expired());
  auto returned = first->retirement()->wait_for(0ms);
  ASSERT_TRUE(returned);
  EXPECT_EQ(lifetime->destroyed.load(), 0u);
  returned.reset();
  EXPECT_EQ(lifetime->destroyed.load(), 1u);
  EXPECT_TRUE(handoff.reinit_pending());
}

TEST(VideoCaptureGenerationTest, StopRejectsLatePublicationWithoutTurningCancellationIntoRelease) {
  handoff_t handoff;
  const auto opening = handoff.begin();
  auto lifetime = std::make_shared<lifetime_t>();
  auto display = make_display(opening, lifetime);
  handoff.stop();
  EXPECT_FALSE(handoff.publish(opening, display));
  EXPECT_FALSE(handoff.acquire());
  EXPECT_FALSE(handoff.begin());
  display.reset();
  EXPECT_FALSE(opening->retirement()->wait_for(0ms));
  EXPECT_EQ(lifetime->destroyed.load(), 0u);
  auto returned = opening->retirement()->detach();
  ASSERT_TRUE(returned);
  returned.reset();
  EXPECT_EQ(lifetime->destroyed.load(), 1u);
  handoff.stop();
  EXPECT_FALSE(handoff.begin());
}

TEST(VideoCaptureGenerationTest, OldRetirementCannotCompleteOrRevokeTheNextGeneration) {
  handoff_t handoff;
  const auto first = handoff.begin();
  auto first_display = make_display(first, std::make_shared<lifetime_t>());
  ASSERT_TRUE(handoff.publish(first, first_display));
  auto old_borrower = handoff.acquire();
  ASSERT_TRUE(handoff.revoke(first));

  // This authority-only test prepares B without opening a physical backend. The real worker
  // additionally waits for and destroys A before beginning B, covered by the full-run fixture.
  const auto next = handoff.begin();
  auto next_lifetime = std::make_shared<lifetime_t>();
  auto next_display = make_display(next, next_lifetime);
  ASSERT_TRUE(handoff.publish(next, next_display));
  auto next_borrower = handoff.acquire();
  ASSERT_TRUE(next_borrower);
  first_display.reset();
  old_borrower = {};
  auto old_returned = first->retirement()->wait_for(0ms);
  ASSERT_TRUE(old_returned);
  old_returned.reset();
  EXPECT_FALSE(next->retirement()->wait_for(0ms));
  EXPECT_FALSE(handoff.revoke(first));
  EXPECT_FALSE(handoff.publish(first, next_display));
  EXPECT_FALSE(next_borrower.revoked());
  EXPECT_EQ(handoff.acquire().display().get(), next_display.get());
  EXPECT_EQ(next_lifetime->destroyed.load(), 0u);
  ASSERT_TRUE(handoff.revoke(next));
  next_display.reset();
  EXPECT_FALSE(next->retirement()->wait_for(0ms));
  next_borrower = {};
  auto next_returned = next->retirement()->wait_for(0ms);
  ASSERT_TRUE(next_returned);
  next_returned.reset();
  EXPECT_EQ(next_lifetime->destroyed.load(), 1u);
}

TEST(VideoCaptureGenerationTest, RejectsNullForeignStaleAndRepeatedPublication) {
  handoff_t handoff, foreign;
  EXPECT_FALSE(handoff.acquire());
  EXPECT_TRUE(handoff_t::display_lease_t {}.revoked());
  const auto generation = handoff.begin();
  const auto foreign_generation = foreign.begin();
  auto display = make_display(generation, std::make_shared<lifetime_t>());
  EXPECT_FALSE(handoff.begin());
  EXPECT_FALSE(handoff.publish({}, display));
  EXPECT_FALSE(handoff.publish(foreign_generation, display));
  EXPECT_FALSE(handoff.revoke({}));
  EXPECT_FALSE(handoff.revoke(foreign_generation));
  EXPECT_FALSE(handoff.publish(generation, {}));
  ASSERT_TRUE(handoff.publish(generation, display));
  EXPECT_FALSE(handoff.publish(generation, display));
  EXPECT_FALSE(handoff.begin());
  ASSERT_TRUE(handoff.revoke(generation));
  EXPECT_TRUE(handoff.revoke(generation));
  const auto next = handoff.begin();
  ASSERT_TRUE(next);
  EXPECT_FALSE(handoff.publish(generation, display));
  EXPECT_FALSE(handoff.revoke(generation));
  EXPECT_TRUE(handoff.reinit_pending());
}

TEST(VideoCaptureGenerationTest, ReplacingAnOldStateDestroysReturnedStorageOutsideAdmissionLock) {
  handoff_t handoff;
  auto generation = handoff.begin();
  auto lifetime = std::make_shared<lifetime_t>();
  bool reentered = false;
  auto display = make_display(generation, lifetime);
  ASSERT_TRUE(handoff.publish(generation, display));
  ASSERT_TRUE(handoff.revoke(generation));
  lifetime->on_destroy = [&] {
    EXPECT_FALSE(handoff.acquire());
    reentered = true;
  };
  auto clear_callback = util::fail_guard([&] {
    lifetime->on_destroy = {};
  });
  display.reset();
  generation.reset();
  EXPECT_EQ(lifetime->destroyed.load(), 0u);
  const auto next = handoff.begin();
  EXPECT_TRUE(next);
  EXPECT_TRUE(reentered);
  EXPECT_EQ(lifetime->destroyed.load(), 1u);
}

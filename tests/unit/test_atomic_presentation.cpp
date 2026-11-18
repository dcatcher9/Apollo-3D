/**
 * @file tests/unit/test_atomic_presentation.cpp
 * @brief The actual per-session presentation admission and encoder-receipt owner.
 */
#include "../tests_common.h"
#include "src/atomic_presentation.h"

#include <memory>
#include <string>
#include <utility>

namespace {
  using owner_t = stream::detail::atomic_presentation_t;

  owner_t::request_t request(std::uint32_t client_id, int width) {
    owner_t::request_t result {};
    result.mail = std::make_shared<safe::mail_raw_t>();
    result.effective_mode = std::make_shared<video::effective_video_mode_publisher_t>(video::effective_video_mode_t {});
    result.change.request_id = client_id;
    result.change.width = width;
    result.launch_session_id = 7;
    result.client_name = std::string(160, 'x') + std::to_string(width);
    return result;
  }

  video::video_mode_applied_t receipt(const owner_t::request_ptr &active, bool applied = true) {
    video::video_mode_applied_t result {};
    result.requests.push_back({active->change.request_id, active->change.transaction_id});
    result.applied = applied;
    result.mode.width = active->change.width;
    return result;
  }
}  // namespace

TEST(AtomicPresentationOwnerTest, CoalescesOnlyPendingOwnedRequests) {
  owner_t owner;
  auto original = request(13, 1920);
  const auto original_name = original.client_name;
  const auto first = owner.submit(original);
  EXPECT_FALSE(first.rejected);
  EXPECT_FALSE(first.superseded);
  original.client_name = "changed after submission";
  original.change.width = 1;

  const auto second = owner.submit(request(13, 2560));
  ASSERT_TRUE(second.superseded);
  EXPECT_EQ(second.superseded->client_name, original_name);
  EXPECT_EQ(second.superseded->change.width, 1920);
  EXPECT_NE(second.superseded->change.transaction_id, 0u);
  EXPECT_FALSE(second.rejected);

  const auto active = owner.wait_next();
  ASSERT_TRUE(active);
  EXPECT_EQ(active->change.width, 2560);
  EXPECT_NE(active->change.transaction_id, second.superseded->change.transaction_id);
  EXPECT_TRUE(owner.finish(active->change.transaction_id));
}

TEST(AtomicPresentationOwnerTest, ActiveRequestRetainsAdmissionThroughReceiptUntilFinish) {
  owner_t owner;
  owner.submit(request(44, 1920));
  const auto active = owner.wait_next();
  ASSERT_TRUE(active);

  const auto first_pending = owner.submit(request(44, 2560));
  EXPECT_FALSE(first_pending.superseded);
  const auto second_pending = owner.submit(request(44, 3840));
  ASSERT_TRUE(second_pending.superseded);
  EXPECT_EQ(second_pending.superseded->change.width, 2560);
  EXPECT_FALSE(owner.finish(second_pending.superseded->change.transaction_id));

  auto completed = receipt(active, false);
  // Matching remains host-token based: opaque client IDs can repeat, and a client-ID
  // mismatch is left for the worker's existing diagnostic rather than dropping the receipt.
  completed.requests.insert(completed.requests.begin(), {44, 0});
  completed.requests.back().request_id = 999;
  ASSERT_TRUE(owner.complete(std::move(completed)));
  const auto proof = owner.wait_completion(active->change.transaction_id);
  ASSERT_TRUE(proof);
  EXPECT_FALSE(proof->applied);
  EXPECT_EQ(proof->requests.back().request_id, 999u);
  EXPECT_FALSE(owner.finish(0));
  ASSERT_TRUE(owner.finish(active->change.transaction_id));

  const auto next = owner.wait_next();
  ASSERT_TRUE(next);
  EXPECT_EQ(next->change.request_id, active->change.request_id);
  EXPECT_EQ(next->change.width, 3840);
  EXPECT_NE(next->change.transaction_id, active->change.transaction_id);
  EXPECT_TRUE(owner.finish(next->change.transaction_id));
}

TEST(AtomicPresentationOwnerTest, FirstReceiptStaysLatchedAfterConsumptionUntilFinish) {
  owner_t owner;
  owner.submit(request(1, 1920));
  const auto active = owner.wait_next();
  ASSERT_TRUE(active);

  auto stale = receipt(active);
  stale.requests.front().transaction_id = 0;
  EXPECT_FALSE(owner.complete(std::move(stale)));
  ASSERT_TRUE(owner.complete(receipt(active, false)));
  EXPECT_FALSE(owner.complete(receipt(active, true)));
  const auto completed = owner.wait_completion(active->change.transaction_id);
  ASSERT_TRUE(completed);
  EXPECT_FALSE(completed->applied);

  // This deliberately strengthens the old empty-slot behavior during worker rollback/ACK.
  EXPECT_FALSE(owner.complete(receipt(active, true)));
  owner.submit(request(1, 2560));
  ASSERT_TRUE(owner.finish(active->change.transaction_id));
  const auto next = owner.wait_next();
  ASSERT_TRUE(next);
  EXPECT_FALSE(owner.complete(receipt(active)));
  ASSERT_TRUE(owner.complete(receipt(next)));
  const auto next_completed = owner.wait_completion(next->change.transaction_id);
  ASSERT_TRUE(next_completed);
  EXPECT_TRUE(next_completed->applied);
  EXPECT_EQ(next_completed->mode.width, 2560);
  EXPECT_TRUE(owner.finish(next->change.transaction_id));
}

TEST(AtomicPresentationOwnerTest, StopDropsPendingAndReceiptButKeepsActiveUntilJoinedCleanup) {
  auto owner = std::make_shared<owner_t>();
  auto active_request = request(5, 1920);
  std::weak_ptr<safe::mail_raw_t> active_mail = active_request.mail;
  owner->submit(std::move(active_request));
  auto active = owner->wait_next();
  ASSERT_TRUE(active);
  const auto token = active->change.transaction_id;
  EXPECT_TRUE(owner->complete(receipt(active)));

  auto pending_request = request(6, 2560);
  std::weak_ptr<safe::mail_raw_t> pending_mail = pending_request.mail;
  owner->submit(std::move(pending_request));
  active.reset();
  owner->stop();
  EXPECT_TRUE(owner->stopping());
  EXPECT_TRUE(pending_mail.expired());
  EXPECT_FALSE(active_mail.expired());
  EXPECT_FALSE(owner->wait_next());
  EXPECT_FALSE(owner->wait_completion(token));

  owner->clear_stopped();
  EXPECT_TRUE(active_mail.expired());
  // A retained config endpoint keeps no former session dependencies after the worker joins.
  EXPECT_TRUE(owner->stopping());
  owner->stop();
  owner->clear_stopped();
}

TEST(AtomicPresentationOwnerTest, ClosedOwnerReturnsRejectedPayloadWithoutReopening) {
  owner_t owner;
  owner.stop();
  auto input = request(8, 3840);
  const auto mail = input.mail;
  const auto submitted = owner.submit(std::move(input));
  EXPECT_FALSE(submitted.superseded);
  ASSERT_TRUE(submitted.rejected);
  EXPECT_EQ(submitted.rejected->mail, mail);
  EXPECT_EQ(submitted.rejected->change.width, 3840);
  EXPECT_EQ(submitted.rejected->change.transaction_id, 0u);
  EXPECT_FALSE(owner.wait_next());
  EXPECT_FALSE(owner.wait_completion(1));
  EXPECT_FALSE(owner.finish(1));
  EXPECT_FALSE(owner.complete(video::video_mode_applied_t {}));
}

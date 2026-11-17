/**
 * @file tests/unit/test_atomic_presentation_lifecycle.cpp
 * @brief Actual live-presentation worker ordering with inert process/display edges.
 */
#include "../tests_common.h"
#include "src/atomic_presentation_test.h"
#include "src/host_sbs_resolution.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using result_e = proc::live_video_mode_result_e;
  using fixture_t = stream::testing::atomic_presentation_fixture_t;

  class gate_t {
  public:
    bool enter() {
      std::unique_lock lock(mutex_);
      entered_ = true;
      changed_.notify_all();
      if (!changed_.wait_for(lock, 10s, [&] {
            return released_;
          })) {
        timed_out_ = true;
        return false;
      }
      return true;
    }

    bool wait_entered() {
      std::unique_lock lock(mutex_);
      return changed_.wait_for(lock, 3s, [&] {
        return entered_;
      });
    }

    void release() {
      std::lock_guard lock(mutex_);
      released_ = true;
      changed_.notify_all();
    }

    bool timed_out() const {
      std::lock_guard lock(mutex_);
      return timed_out_;
    }

  private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false;
    bool released_ = false;
    bool timed_out_ = false;
  };

  struct process_call_t {
    int width;
    int height;
    int millihz;
    std::uint32_t launch_id;
  };

  class process_script_t: public std::enable_shared_from_this<process_script_t> {
  public:
    struct action_t {
      result_e result;
      std::shared_ptr<gate_t> gate;
    };

    void needs_change(bool value) {
      std::lock_guard lock(mutex_);
      needs_change_ = value;
    }

    void add(result_e result, std::shared_ptr<gate_t> gate = {}) {
      std::lock_guard lock(mutex_);
      actions_.push_back({result, gate});
      if (gate) {
        gates_.push_back(std::move(gate));
      }
    }

    stream::testing::atomic_presentation_process_hooks_t hooks() {
      const auto self = shared_from_this();
      return {
        [self](int width, int height, int millihz) {
          std::lock_guard lock(self->mutex_);
          self->queries_.push_back({width, height, millihz, 0});
          return self->needs_change_;
        },
        [self](int width, int height, int millihz, std::uint32_t launch_id) {
          action_t action;
          {
            std::lock_guard lock(self->mutex_);
            self->calls_.push_back({width, height, millihz, launch_id});
            if (self->actions_.empty()) {
              ++self->unexpected_calls_;
              return result_e::failed;
            }
            action = std::move(self->actions_.front());
            self->actions_.pop_front();
          }
          if (action.gate && !action.gate->enter()) {
            return result_e::failed;
          }
          return action.result;
        },
      };
    }

    std::vector<process_call_t> calls() const {
      std::lock_guard lock(mutex_);
      return calls_;
    }

    std::vector<process_call_t> queries() const {
      std::lock_guard lock(mutex_);
      return queries_;
    }

    void release_all() {
      std::lock_guard lock(mutex_);
      for (const auto &gate : gates_) {
        gate->release();
      }
    }

    void verify() const {
      std::lock_guard lock(mutex_);
      EXPECT_EQ(unexpected_calls_, 0U);
      for (const auto &gate : gates_) {
        EXPECT_FALSE(gate->timed_out());
      }
    }

  private:
    mutable std::mutex mutex_;
    bool needs_change_ = true;
    unsigned unexpected_calls_ = 0;
    std::deque<action_t> actions_;
    std::vector<std::shared_ptr<gate_t>> gates_;
    std::vector<process_call_t> calls_;
    std::vector<process_call_t> queries_;
  };

  class AtomicPresentationLifecycleTest: public ::testing::Test {
  protected:
    void SetUp() override {
      script_ = std::make_shared<process_script_t>();
      fixture_ = std::make_unique<fixture_t>(initial_, script_->hooks());
    }

    void TearDown() override {
      // Release fixture-owned barriers before the real worker join, including ASSERT unwinds.
      if (script_) {
        script_->release_all();
      }
      if (fixture_) {
        fixture_->release_encode_wake();
        fixture_->stop();
        fixture_->join();
        EXPECT_FALSE(fixture_->gate_timed_out());
      }
      if (script_) {
        script_->verify();
      }
      fixture_.reset();
    }

    static video::video_mode_change_t change(std::uint32_t id, int width = 1920, int height = 1080) {
      return {width, height, 60, 6000, 60000, 22000, id, 0, video::SBS_OFF};
    }

    static video::effective_video_mode_t proven(const video::video_mode_change_t &request, std::uint32_t generation) {
      return {request.width, request.height, request.framerateX100, request.bitrate, request.width, request.height, request.width, request.height, request.requested_sbs_mode, generation};
    }

    static video::video_mode_applied_t completion(const video::video_mode_change_t &request, bool applied, const video::effective_video_mode_t &mode, video::desktop_video_mode_t desktop = {1280, 720, 60000}) {
      return {{{request.request_id, request.transaction_id}}, applied, mode, desktop};
    }

    static void expect_ack(const stream::live_video_mode_ack_t &ack, std::uint32_t id, stream::live_video_mode_ack_e status, const video::effective_video_mode_t &mode) {
      EXPECT_EQ(ack.request_id, id);
      EXPECT_EQ(ack.status, status);
      EXPECT_EQ(ack.applied_generation, mode.generation);
      EXPECT_EQ(ack.applied_source_width, mode.source_width);
      EXPECT_EQ(ack.applied_source_height, mode.source_height);
      EXPECT_EQ(ack.applied_encoded_width, mode.encoded_width);
      EXPECT_EQ(ack.applied_encoded_height, mode.encoded_height);
      EXPECT_EQ(ack.applied_framerate_x100, mode.framerateX100);
      EXPECT_EQ(ack.applied_bitrate_kbps, mode.bitrate);
      EXPECT_EQ(ack.applied_sbs_mode, mode.sbs_mode);
    }

    void expect_call(const process_call_t &call, int width, int height, int millihz) const {
      EXPECT_EQ(call.width, width);
      EXPECT_EQ(call.height, height);
      EXPECT_EQ(call.millihz, millihz);
      EXPECT_EQ(call.launch_id, fixture_->launch_session_id());
    }

    video::effective_video_mode_t initial_ {1280, 720, 6000, 10000, 1280, 720, 1280, 720, video::SBS_OFF, 1};
    std::shared_ptr<process_script_t> script_;
    std::unique_ptr<fixture_t> fixture_;
  };
}  // namespace

TEST_F(AtomicPresentationLifecycleTest, CompletionBeforeWaitPreservesTheExactAppliedAckAndPacingProof) {
  script_->add(result_e::applied);
  fixture_->gate_next_encode_wake();
  fixture_->submit(change(17));
  ASSERT_TRUE(fixture_->wait_encode_wake(3s));
  const auto request = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(request);
  ASSERT_NE(request->transaction_id, 0U);
  EXPECT_FALSE(fixture_->pacing_diagnostics_rearmed());
  const video::effective_video_mode_t actual {1600, 900, 5994, 18001, 1920, 1080, 1600, 900, video::SBS_OFF, 42};
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*request, true, actual));
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  fixture_->release_encode_wake();
  const auto ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(ack);
  expect_ack(*ack, 17, stream::live_video_mode_ack_e::applied, actual);
  EXPECT_TRUE(fixture_->pacing_diagnostics_rearmed());
  const auto calls = script_->calls();
  ASSERT_EQ(calls.size(), 1U);
  expect_call(calls[0], 1920, 1080, 60000);
}

TEST_F(AtomicPresentationLifecycleTest, PendingRequestsCoalesceWhileRollbackKeepsTheActiveTransaction) {
  const auto rollback_gate = std::make_shared<gate_t>();
  script_->add(result_e::applied);
  script_->add(result_e::unchanged, rollback_gate);
  script_->add(result_e::applied);
  fixture_->submit(change(1));
  const auto first = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(first);
  fixture_->submit(change(2, 2560, 1440));
  fixture_->submit(change(3, 3840, 2160));
  const auto superseded = fixture_->pop_ack(3s);
  ASSERT_TRUE(superseded);
  expect_ack(*superseded, 2, stream::live_video_mode_ack_e::failed, initial_);
  fixture_->deliver_completion(completion(*first, false, initial_, {1360, 768, 59940}));
  ASSERT_TRUE(rollback_gate->wait_entered());
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  rollback_gate->release();
  const auto failed = fixture_->pop_ack(3s);
  ASSERT_TRUE(failed);
  expect_ack(*failed, 1, stream::live_video_mode_ack_e::failed, initial_);
  const auto latest = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest->request_id, 3U);
  EXPECT_NE(latest->transaction_id, first->transaction_id);
  const auto actual = proven(*latest, 2);
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*latest, true, actual));
  const auto applied = fixture_->pop_ack(3s);
  ASSERT_TRUE(applied);
  expect_ack(*applied, 3, stream::live_video_mode_ack_e::applied, actual);
  const auto calls = script_->calls();
  ASSERT_EQ(calls.size(), 3U);
  expect_call(calls[0], 1920, 1080, 60000);
  expect_call(calls[1], 1360, 768, 59940);
  expect_call(calls[2], 3840, 2160, 60000);
}

TEST_F(AtomicPresentationLifecycleTest, RepeatedClientIdsUseHostTokensAndStaleCompletionsCannotReleaseSuccessor) {
  script_->needs_change(false);
  fixture_->submit(change(77));
  const auto first = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(first);
  const auto first_mode = proven(*first, 2);
  fixture_->publish_effective_mode(first_mode);
  fixture_->deliver_completion(completion(*first, true, first_mode));
  ASSERT_TRUE(fixture_->pop_ack(3s));
  fixture_->submit(change(77, 2560, 1440));
  const auto second = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(second);
  ASSERT_NE(first->transaction_id, second->transaction_id);
  fixture_->submit(change(99, 3840, 2160));
  fixture_->deliver_completion(completion(*first, true, first_mode));
  auto unknown = completion(*second, true, first_mode);
  unknown.requests[0].transaction_id = 0;
  fixture_->deliver_completion(std::move(unknown));
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  const auto second_mode = proven(*second, 3);
  fixture_->publish_effective_mode(second_mode);
  auto matching = completion(*second, true, second_mode);
  // The old worker logs this malformed internal correlation but answers the active client ID.
  matching.requests[0].request_id = 123456;
  fixture_->deliver_completion(std::move(matching));
  const auto second_ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(second_ack);
  expect_ack(*second_ack, 77, stream::live_video_mode_ack_e::applied, second_mode);
  const auto third = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(third);
  EXPECT_EQ(third->request_id, 99U);
  EXPECT_NE(third->transaction_id, second->transaction_id);
  const auto third_mode = proven(*third, 4);
  fixture_->publish_effective_mode(third_mode);
  fixture_->deliver_completion(completion(*third, true, third_mode));
  const auto third_ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(third_ack);
  expect_ack(*third_ack, 99, stream::live_video_mode_ack_e::applied, third_mode);
  EXPECT_TRUE(script_->calls().empty());
}

TEST_F(AtomicPresentationLifecycleTest, DisplayFailuresReportLastProvenModeWithoutEncoderCommit) {
  for (const auto result : {result_e::failed, result_e::needs_reconnect}) {
    script_->add(result);
    fixture_->submit(change(result == result_e::failed ? 21 : 22));
    const auto ack = fixture_->pop_ack(3s);
    ASSERT_TRUE(ack);
    expect_ack(*ack, result == result_e::failed ? 21 : 22, result == result_e::failed ? stream::live_video_mode_ack_e::failed : stream::live_video_mode_ack_e::rejected_needs_reconnect, initial_);
    EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  }
  EXPECT_EQ(script_->calls().size(), 2U);
}

TEST_F(AtomicPresentationLifecycleTest, BitrateOnlyRequestsSkipDisplayWorkAndStillRequireEncoderProof) {
  script_->needs_change(false);
  fixture_->submit(change(31, initial_.source_width, initial_.source_height));
  const auto request = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(request);
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  const auto actual = proven(*request, 2);
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*request, true, actual));
  const auto ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(ack);
  expect_ack(*ack, 31, stream::live_video_mode_ack_e::applied, actual);
  fixture_->submit(change(32, initial_.source_width, initial_.source_height));
  const auto refused = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(refused);
  fixture_->deliver_completion(completion(*refused, false, actual));
  const auto failed = fixture_->pop_ack(3s);
  ASSERT_TRUE(failed);
  expect_ack(*failed, 32, stream::live_video_mode_ack_e::failed, actual);
  EXPECT_TRUE(script_->calls().empty());
  const auto queries = script_->queries();
  ASSERT_EQ(queries.size(), 2U);
  EXPECT_EQ(queries[0].width, initial_.source_width);
  EXPECT_EQ(queries[0].height, initial_.source_height);
  EXPECT_EQ(queries[0].millihz, 60000);
}

TEST_F(AtomicPresentationLifecycleTest, FailedEncoderUsesExactDesktopReceiptAndFailedRollbackRequiresReconnect) {
  for (const auto rollback_result : {result_e::unchanged, result_e::failed}) {
    script_->add(result_e::applied);
    script_->add(rollback_result);
    const auto id = rollback_result == result_e::unchanged ? 41U : 42U;
    fixture_->submit(change(id));
    const auto request = fixture_->pop_encode_request(3s);
    ASSERT_TRUE(request);
    fixture_->deliver_completion(completion(*request, false, initial_, {2560, 1440, 59940}));
    const auto ack = fixture_->pop_ack(3s);
    ASSERT_TRUE(ack);
    expect_ack(*ack, id, rollback_result == result_e::unchanged ? stream::live_video_mode_ack_e::failed : stream::live_video_mode_ack_e::rejected_needs_reconnect, initial_);
  }
  const auto calls = script_->calls();
  ASSERT_EQ(calls.size(), 4U);
  expect_call(calls[1], 2560, 1440, 59940);
  expect_call(calls[3], 2560, 1440, 59940);
}

TEST_F(AtomicPresentationLifecycleTest, StopDuringDisplayApplyWaitsForTheOwnedWorkerAndSuppressesLateCommit) {
  const auto apply_gate = std::make_shared<gate_t>();
  script_->add(result_e::applied, apply_gate);
  fixture_->submit(change(51));
  ASSERT_TRUE(apply_gate->wait_entered());
  fixture_->submit(change(52, 2560, 1440));
  std::promise<void> joining;
  auto join_started = joining.get_future();
  std::future<void> joined;
  auto release_before_future = util::fail_guard([&] {
    script_->release_all();
  });
  fixture_->stop();
  EXPECT_TRUE(fixture_->stopping());
  joined = std::async(std::launch::async, [&] {
    joining.set_value();
    fixture_->join();
  });
  ASSERT_EQ(join_started.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(joined.wait_for(20ms), std::future_status::timeout);
  apply_gate->release();
  ASSERT_EQ(joined.wait_for(3s), std::future_status::ready);
  joined.get();
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  EXPECT_EQ(script_->calls().size(), 1U);
}

TEST_F(AtomicPresentationLifecycleTest, StopDuringEncoderWaitDropsPendingAndRejectsLateCompletion) {
  script_->needs_change(false);
  fixture_->submit(change(61));
  const auto request = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(request);
  fixture_->submit(change(62, 2560, 1440));
  fixture_->stop();
  fixture_->join();
  EXPECT_TRUE(fixture_->stopping());
  const auto actual = proven(*request, 2);
  fixture_->deliver_completion(completion(*request, true, actual));
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  EXPECT_FALSE(fixture_->pop_ack(0ms));
  EXPECT_TRUE(script_->calls().empty());
}

TEST_F(AtomicPresentationLifecycleTest, UnsupportedAiGeometryIsRejectedBeforeEitherProcessEdge) {
  script_->needs_change(false);
  auto invalid = change(71, 7680, 4320);
  invalid.requested_sbs_mode = video::SBS_AI;
  ASSERT_FALSE(models::host_sbs_v2_source_resolution_is_supported(invalid.width, invalid.height));
  fixture_->submit(invalid);
  const auto rejected = fixture_->pop_ack(3s);
  ASSERT_TRUE(rejected);
  expect_ack(*rejected, 71, stream::live_video_mode_ack_e::rejected_invalid, initial_);
  EXPECT_TRUE(script_->queries().empty());
  EXPECT_TRUE(script_->calls().empty());
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
  // The worker's guard belongs to Host AI; ordinary video may reach the encoder's own clamp.
  invalid.request_id = 72;
  invalid.requested_sbs_mode = video::SBS_OFF;
  fixture_->submit(invalid);
  const auto request = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(request);
  const video::effective_video_mode_t actual {3840, 2160, 6000, 22000, 7680, 4320, 3840, 2160, video::SBS_OFF, 2};
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*request, true, actual));
  const auto applied = fixture_->pop_ack(3s);
  ASSERT_TRUE(applied);
  expect_ack(*applied, 72, stream::live_video_mode_ack_e::applied, actual);
  EXPECT_EQ(script_->queries().size(), 1U);
}

TEST_F(AtomicPresentationLifecycleTest, StopBeforeFirstSubmissionRejectsWithoutStartingWorker) {
  fixture_->stop();
  EXPECT_TRUE(fixture_->stopping());
  fixture_->submit(change(91));
  const auto rejected = fixture_->pop_ack(3s);
  ASSERT_TRUE(rejected);
  expect_ack(*rejected, 91, stream::live_video_mode_ack_e::failed, initial_);
  fixture_->join();
  fixture_->submit(change(92));
  const auto after_join = fixture_->pop_ack(3s);
  ASSERT_TRUE(after_join);
  expect_ack(*after_join, 92, stream::live_video_mode_ack_e::failed, initial_);
  EXPECT_TRUE(script_->queries().empty());
  EXPECT_TRUE(script_->calls().empty());
  EXPECT_FALSE(fixture_->pop_encode_request(0ms));
}

TEST_F(AtomicPresentationLifecycleTest, DirectCompletionDoesNotDependOnControlRelay) {
  script_->needs_change(false);
  fixture_->submit(change(81));
  const auto request = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(request);
  const auto actual = proven(*request, 2);
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*request, true, actual));
  const auto ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(ack);
  expect_ack(*ack, 81, stream::live_video_mode_ack_e::applied, actual);
}

TEST_F(AtomicPresentationLifecycleTest, ConsumedCompletionCannotBeReplacedWhileRollbackHoldsTheTransaction) {
  const auto rollback_gate = std::make_shared<gate_t>();
  script_->add(result_e::applied);
  script_->add(result_e::unchanged, rollback_gate);
  script_->add(result_e::applied);
  fixture_->submit(change(101));
  const auto first = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(first);
  fixture_->submit(change(102, 2560, 1440));
  fixture_->deliver_completion(completion(*first, false, initial_));
  ASSERT_TRUE(rollback_gate->wait_entered());
  // The first receipt has been consumed, but rollback still owns this active token. A duplicate
  // must not become a second receipt or remain queued to impersonate the successor's result.
  const video::effective_video_mode_t bogus {640, 480, 3000, 7777, 640, 480, 640, 480, video::SBS_OFF, 99};
  fixture_->deliver_completion(completion(*first, true, bogus));
  rollback_gate->release();
  const auto failed = fixture_->pop_ack(3s);
  ASSERT_TRUE(failed);
  expect_ack(*failed, 101, stream::live_video_mode_ack_e::failed, initial_);
  const auto second = fixture_->pop_encode_request(3s);
  ASSERT_TRUE(second);
  ASSERT_EQ(second->request_id, 102U);
  ASSERT_NE(second->transaction_id, first->transaction_id);
  const auto actual = proven(*second, 2);
  fixture_->publish_effective_mode(actual);
  fixture_->deliver_completion(completion(*second, true, actual));
  const auto ack = fixture_->pop_ack(3s);
  ASSERT_TRUE(ack);
  expect_ack(*ack, 102, stream::live_video_mode_ack_e::applied, actual);
  const auto calls = script_->calls();
  ASSERT_EQ(calls.size(), 3U);
  expect_call(calls[0], 1920, 1080, 60000);
  expect_call(calls[1], 1280, 720, 60000);
  expect_call(calls[2], 2560, 1440, 60000);
}

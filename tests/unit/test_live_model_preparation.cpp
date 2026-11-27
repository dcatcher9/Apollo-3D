/** Actual eager preparation-owner orchestration, with only the two GPU operations replaced. */
#include <gtest/gtest.h>

#include "src/config.h"
#include "src/live_model_preparation.h"
#include "src/live_model_preparation_test.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using access_t = models::testing::live_model_preparation_test_access;
  using hooks_t = models::testing::preparation_hooks_t;
  using stage_t = models::testing::preparation_stage_e;

  struct operation_gate_t {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    bool timed_out = false;

    void hold() {
      std::unique_lock lock(mutex);
      entered = true;
      changed.notify_all();
      timed_out = !changed.wait_for(lock, 10s, [&] {
        return released;
      });
    }

    bool wait_entered() {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, 3s, [&] {
        return entered;
      });
    }

    void release() {
      std::lock_guard lock(mutex);
      released = true;
      changed.notify_all();
    }
  };

  struct release_gate_t {
    operation_gate_t &gate;

    ~release_gate_t() {
      gate.release();
    }
  };

  struct joined_operation_t {
    operation_gate_t &gate;
    std::thread thread;

    ~joined_operation_t() {
      gate.release();
      if (thread.joinable()) {
        thread.join();
      }
    }
  };

  config::depth_model_info model() {
    return {"owned-test-model-" + std::string(96, 'm'), "https://invalid.example/" + std::string(96, 'u')};
  }

  constexpr auto expected_stages = std::to_array<stage_t>({
    stage_t::worker_enter,
    stage_t::depth_finished,
    stage_t::ocr_finished,
    stage_t::worker_exit,
  });
}  // namespace

TEST(LiveModelPreparationTest, StartedWorkerOwnsItsArgumentsAndRunsTheOriginalOperationOrder) {
  operation_gate_t gate;
  std::mutex mutex;
  std::vector<stage_t> stages;
  std::vector<std::string> calls;
  std::filesystem::path depth_assets, ocr_assets;
  config::depth_model_info observed_model;
  std::string depth_adapter, ocr_adapter;
  std::thread::id depth_thread, ocr_thread;
  auto input_model = model();
  const auto expected_model = input_model;
  std::string adapter = "owned-adapter-" + std::string(96, 'a');
  const auto expected_adapter = adapter;
  hooks_t hooks;
  hooks.observe = [&](stage_t stage) {
    std::lock_guard lock(mutex);
    stages.push_back(stage);
  };
  hooks.prepare_depth = [&](const auto &assets, const auto &selected_model, const auto &selected_adapter) {
    gate.hold();
    {
      std::lock_guard lock(mutex);
      calls.emplace_back("depth");
      depth_assets = assets;
      observed_model = selected_model;
      depth_adapter = selected_adapter;
      depth_thread = std::this_thread::get_id();
    }
    return true;
  };
  hooks.prepare_ocr = [&](const auto &assets, const auto &selected_adapter) {
    std::lock_guard lock(mutex);
    calls.emplace_back("ocr");
    ocr_assets = assets;
    ocr_adapter = selected_adapter;
    ocr_thread = std::this_thread::get_id();
    return true;
  };
  auto owner = access_t::start(input_model, adapter, std::move(hooks));
  release_gate_t release {gate};
  ASSERT_TRUE(gate.wait_entered());
  input_model = {"replacement", "replacement"};
  adapter = "replacement";
  gate.release();
  access_t::join(*owner);
  access_t::join(*owner);

  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(calls, (std::vector<std::string> {"depth", "ocr"}));
  EXPECT_EQ(stages, (std::vector<stage_t>(expected_stages.begin(), expected_stages.end())));
  EXPECT_EQ(observed_model.name, expected_model.name);
  EXPECT_EQ(observed_model.url, expected_model.url);
  EXPECT_EQ(depth_adapter, expected_adapter);
  EXPECT_EQ(ocr_adapter, expected_adapter);
  EXPECT_FALSE(depth_assets.empty());
  EXPECT_EQ(depth_assets, ocr_assets);
  EXPECT_EQ(depth_thread, ocr_thread);
  EXPECT_NE(depth_thread, std::this_thread::get_id());
}

TEST(LiveModelPreparationTest, FailedDepthAndOptionalOcrStillFollowTheOriginalWorkerSequence) {
  for (const auto results : {std::pair {false, true}, std::pair {true, false}, std::pair {false, false}}) {
    SCOPED_TRACE(::testing::Message() << "depth=" << results.first << ", OCR=" << results.second);
    std::vector<stage_t> stages;
    std::vector<std::string> calls;
    hooks_t hooks;
    hooks.observe = [&](stage_t stage) {
      stages.push_back(stage);
    };
    hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
      calls.emplace_back("depth");
      return results.first;
    };
    hooks.prepare_ocr = [&](const auto &, const auto &) {
      calls.emplace_back("ocr");
      return results.second;
    };
    auto owner = access_t::start(model(), "adapter", std::move(hooks));
    access_t::join(*owner);
    EXPECT_EQ(calls, (std::vector<std::string> {"depth", "ocr"}));
    EXPECT_EQ(stages, (std::vector<stage_t>(expected_stages.begin(), expected_stages.end())));
  }
}

TEST(LiveModelPreparationTest, DestructorJoinsAnAlreadyStartedDepthOperation) {
  operation_gate_t gate;
  std::atomic<unsigned> ocr_calls {0};
  hooks_t hooks;
  hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    gate.hold();
    return true;
  };
  hooks.prepare_ocr = [&](const auto &, const auto &) {
    ++ocr_calls;
    return true;
  };
  auto owner = access_t::start(model(), "adapter", std::move(hooks));
  release_gate_t release {gate};
  ASSERT_TRUE(gate.wait_entered());
  std::promise<void> entered, completed;
  auto entered_future = entered.get_future();
  auto completed_future = completed.get_future();
  joined_operation_t operation {gate};
  operation.thread = std::thread([&] {
    entered.set_value();
    owner.reset();
    completed.set_value();
  });
  ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(completed_future.wait_for(0ms), std::future_status::timeout);
  gate.release();
  ASSERT_EQ(completed_future.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(ocr_calls.load(), 1u);
  EXPECT_FALSE(gate.timed_out);
}

TEST(LiveModelPreparationTest, ExplicitJoinWaitsForOptionalOcrAndCanBeRepeated) {
  operation_gate_t gate;
  hooks_t hooks;
  hooks.prepare_depth = [](const auto &, const auto &, const auto &) {
    return true;
  };
  hooks.prepare_ocr = [&](const auto &, const auto &) {
    gate.hold();
    return false;
  };
  auto owner = access_t::start(model(), "adapter", std::move(hooks));
  release_gate_t release {gate};
  ASSERT_TRUE(gate.wait_entered());
  std::promise<void> entered, completed;
  auto entered_future = entered.get_future();
  auto completed_future = completed.get_future();
  joined_operation_t operation {gate};
  operation.thread = std::thread([&] {
    entered.set_value();
    access_t::join(*owner);
    completed.set_value();
  });
  ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(completed_future.wait_for(0ms), std::future_status::timeout);
  gate.release();
  ASSERT_EQ(completed_future.wait_for(3s), std::future_status::ready);
  operation.thread.join();
  access_t::join(*owner);
  EXPECT_FALSE(gate.timed_out);
}

TEST(LiveModelPreparationTest, SequentialOwnersDoNotRetainPriorOperationsOrObservers) {
  operation_gate_t first_gate;
  std::atomic<unsigned> first_calls {0}, second_calls {0};
  std::vector<stage_t> first_stages, second_stages;
  hooks_t first_hooks;
  first_hooks.observe = [&](stage_t stage) {
    first_stages.push_back(stage);
  };
  first_hooks.prepare_depth = [&](const auto &, const auto &, const auto &) {
    ++first_calls;
    first_gate.hold();
    return true;
  };
  first_hooks.prepare_ocr = [&](const auto &, const auto &) {
    ++first_calls;
    return true;
  };
  auto first_owner = access_t::start(model(), "first-adapter", std::move(first_hooks));
  release_gate_t release {first_gate};
  ASSERT_TRUE(first_gate.wait_entered());
  first_gate.release();
  access_t::join(*first_owner);
  first_owner.reset();
  ASSERT_EQ(first_calls.load(), 2u);
  ASSERT_EQ(first_stages, (std::vector<stage_t>(expected_stages.begin(), expected_stages.end())));
  std::string second_adapter;
  hooks_t second_hooks;
  second_hooks.observe = [&](stage_t stage) {
    second_stages.push_back(stage);
  };
  second_hooks.prepare_depth = [&](const auto &, const auto &, const auto &adapter) {
    ++second_calls;
    second_adapter = adapter;
    return false;
  };
  second_hooks.prepare_ocr = [&](const auto &, const auto &) {
    ++second_calls;
    return true;
  };
  auto second_owner = access_t::start(model(), "second-adapter", std::move(second_hooks));
  access_t::join(*second_owner);
  EXPECT_EQ(second_calls.load(), 2u);
  EXPECT_EQ(first_calls.load(), 2u);
  EXPECT_EQ(second_adapter, "second-adapter");
  EXPECT_EQ(second_stages, (std::vector<stage_t>(expected_stages.begin(), expected_stages.end())));
  EXPECT_EQ(first_stages, (std::vector<stage_t>(expected_stages.begin(), expected_stages.end())));
  EXPECT_FALSE(first_gate.timed_out);
}

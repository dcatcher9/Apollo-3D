#include "src/native_encode_owner.h"
#include "src/nvenc/nvenc_base.h"
#include "src/platform/common.h"
#include "src/tracked_async_worker.h"
#include "src/video_session_recovery.h"

#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace {
  using namespace std::chrono_literals;
  using owner_t = platf::native_encode_owner_t;

  struct destruction_state_t {
    explicit destruction_state_t(bool held):
        entered(entered_promise.get_future()),
        release_signal(release_promise.get_future().share()) {
      if (!held) {
        release();
      }
    }

    void release() {
      if (!released.exchange(true)) {
        release_promise.set_value();
      }
    }

    std::promise<void> entered_promise, release_promise;
    std::future<void> entered;
    std::shared_future<void> release_signal;
    std::atomic<bool> released {false}, gate_timed_out {false};
    std::atomic<unsigned> destruction_count {0};
    std::thread::id destructor_thread;
    std::weak_ptr<int> resources;
  };

  class inert_encoder_t final: public nvenc::nvenc_base {
  public:
    inert_encoder_t():
        nvenc_base(NV_ENC_DEVICE_TYPE_DIRECTX) {
    }

  private:
    bool init_library() override {
      ADD_FAILURE() << "Owner tests must not initialize an encoder driver";
      return false;
    }

    bool create_and_register_input_buffer() override {
      ADD_FAILURE() << "Owner tests must not register GPU inputs";
      return false;
    }

    void release_async_event() override {
      ADD_FAILURE() << "Owner tests must not own native completion events";
    }
  };

  class inert_backend_t final: public platf::nvenc_encode_device_t {
  public:
    explicit inert_backend_t(std::shared_ptr<destruction_state_t> state):
        state_(std::move(state)),
        completion_ {state_},
        resources_(std::make_shared<int>(42)) {
      state_->resources = resources_;
    }

    ~inert_backend_t() override {
      state_->destructor_thread = std::this_thread::get_id();
      state_->entered_promise.set_value();
      // A bounded backstop also prevents a broken fixture from hanging process shutdown.
      state_->gate_timed_out = state_->release_signal.wait_for(5s) != std::future_status::ready;
    }

    int convert(platf::img_t &) override {
      ADD_FAILURE() << "Owner tests do not convert images";
      return -1;
    }

    nvenc::nvenc_base &encoder() override {
      return encoder_;
    }

  private:
    struct completion_marker_t {
      std::shared_ptr<destruction_state_t> state;

      ~completion_marker_t() {
        ++state->destruction_count;
      }
    };

    std::shared_ptr<destruction_state_t> state_;
    // Reverse member destruction releases the encoder and resources before this observation.
    completion_marker_t completion_;
    std::shared_ptr<int> resources_;
    inert_encoder_t encoder_;
  };

  struct local_workers_t {
    ~local_workers_t() {
      for (const auto &state : states) {
        state->release();
      }
      workers.seal_and_drain();
    }

    std::shared_ptr<destruction_state_t> state(bool held = false) {
      auto result = std::make_shared<destruction_state_t>(held);
      states.push_back(result);
      return result;
    }

    owner_t owner(const std::shared_ptr<destruction_state_t> &state) {
      return platf::native_encode_owner_test_access::adopt(std::make_unique<inert_backend_t>(state), workers);
    }

    video::detail::tracked_async_worker_pool_t workers;
    std::vector<std::shared_ptr<destruction_state_t>> states;
  };

  struct release_guard_t {
    std::shared_ptr<destruction_state_t> state;

    ~release_guard_t() {
      state->release();
    }
  };

  // Invoke potentially regressed synchronous retirement away from the test thread. On any
  // assertion failure, release the destructor before joining this caller or draining its pool.
  class scoped_operation_t {
  public:
    template<class Operation>
    scoped_operation_t(std::shared_ptr<destruction_state_t> state, Operation operation):
        state_(std::move(state)) {
      std::packaged_task<void()> task(std::move(operation));
      completion_ = task.get_future();
      caller_ = std::thread(std::move(task));
    }

    ~scoped_operation_t() {
      state_->release();
      join();
    }

    bool returned() {
      return completion_.wait_for(3s) == std::future_status::ready;
    }

    void finish() {
      join();
      completion_.get();
    }

  private:
    void join() {
      if (caller_.joinable()) {
        caller_.join();
      }
    }

    std::shared_ptr<destruction_state_t> state_;
    std::future<void> completion_;
    std::thread caller_;
  };

  void expect_destroyed_once(const std::shared_ptr<destruction_state_t> &state) {
    EXPECT_EQ(state->destruction_count.load(), 1u);
    EXPECT_TRUE(state->resources.expired());
    EXPECT_FALSE(state->gate_timed_out.load());
  }
}  // namespace

TEST(NativeEncodeOwnerTest, EmptyOwnerHasNoBackendOrRetirementReceipt) {
  local_workers_t local;
  owner_t empty;
  EXPECT_FALSE(empty);
  EXPECT_EQ(empty.get(), nullptr);
  EXPECT_EQ(empty.operator->(), nullptr);
  EXPECT_FALSE(std::move(empty).retire().valid());
  owner_t moved(std::move(empty));
  EXPECT_FALSE(empty);
  EXPECT_FALSE(moved);
  auto bound_empty = platf::native_encode_owner_test_access::adopt({}, local.workers);
  EXPECT_FALSE(bound_empty);
  EXPECT_FALSE(std::move(bound_empty).retire().valid());
}

TEST(NativeEncodeOwnerTest, MoveAndSelfMovePreserveTheOnlyBackend) {
  local_workers_t local;
  const auto state = local.state();
  auto original = local.owner(state);
  auto *backend = original.get();
  auto *encoder = &original->encoder();
  owner_t moved(std::move(original));
  EXPECT_FALSE(original);
  EXPECT_EQ(moved.get(), backend);
  EXPECT_EQ(&moved->encoder(), encoder);
  auto &alias = moved;
  moved = std::move(alias);
  EXPECT_EQ(moved.get(), backend);
  EXPECT_EQ(state->destruction_count.load(), 0u);
  auto completion = std::move(moved).retire();
  ASSERT_TRUE(completion.valid());
  ASSERT_EQ(completion.wait_for(3s), std::future_status::ready);
  completion.get();
  EXPECT_FALSE(moved);
  EXPECT_FALSE(std::move(moved).retire().valid());
  expect_destroyed_once(state);
}

TEST(NativeEncodeOwnerTest, MoveAssignmentRetiresTheOldBackendAndAdoptsTheNewPool) {
  local_workers_t previous_workers, replacement_workers;
  const auto previous = previous_workers.state(true);
  const auto replacement = replacement_workers.state();
  auto destination = previous_workers.owner(previous);
  auto source = replacement_workers.owner(replacement);
  auto *replacement_backend = source.get();
  release_guard_t release {previous};
  std::thread::id assigning_thread;
  scoped_operation_t assignment(previous, [&]() {
    assigning_thread = std::this_thread::get_id();
    destination = std::move(source);
  });
  ASSERT_EQ(previous->entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(assignment.returned()) << "Move assignment waited for backend destruction";
  EXPECT_NE(previous->destructor_thread, assigning_thread);
  EXPECT_FALSE(previous->resources.expired());
  EXPECT_EQ(previous->destruction_count.load(), 0u);
  previous->release();
  assignment.finish();
  EXPECT_FALSE(source);
  ASSERT_EQ(destination.get(), replacement_backend);
  previous_workers.workers.seal_and_drain();
  expect_destroyed_once(previous);

  // The old pool is sealed: incorrectly retaining its binding would destroy inline here.
  const auto caller = std::this_thread::get_id();
  auto completion = std::move(destination).retire();
  ASSERT_TRUE(completion.valid());
  ASSERT_EQ(completion.wait_for(3s), std::future_status::ready);
  completion.get();
  EXPECT_NE(replacement->destructor_thread, caller);
  expect_destroyed_once(replacement);
}

TEST(NativeEncodeOwnerTest, DestructorRetiresWithoutWaitingForTheBackend) {
  local_workers_t local;
  const auto state = local.state(true);
  release_guard_t release {state};
  std::thread::id disposing_thread;
  scoped_operation_t disposal(state, [&]() {
    disposing_thread = std::this_thread::get_id();
    auto owner = local.owner(state);
  });
  ASSERT_EQ(state->entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(disposal.returned()) << "Owner destruction waited for backend destruction";
  EXPECT_NE(state->destructor_thread, disposing_thread);
  EXPECT_FALSE(state->resources.expired());
  EXPECT_EQ(state->destruction_count.load(), 0u);
  state->release();
  disposal.finish();
  local.workers.seal_and_drain();
  expect_destroyed_once(state);
}

TEST(NativeEncodeOwnerTest, RetirementReceiptWaitsUntilBackendResourcesAreDestroyed) {
  local_workers_t local;
  const auto state = local.state(true);
  auto owner = local.owner(state);
  release_guard_t release {state};
  std::future<void> receipt;
  scoped_operation_t retirement(state, [&]() {
    receipt = std::move(owner).retire();
  });
  ASSERT_EQ(state->entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(retirement.returned()) << "Explicit retirement waited for backend destruction";
  retirement.finish();
  EXPECT_FALSE(owner);
  ASSERT_TRUE(receipt.valid());
  EXPECT_EQ(receipt.wait_for(0ms), std::future_status::timeout);
  EXPECT_FALSE(state->resources.expired());
  EXPECT_EQ(state->destruction_count.load(), 0u);
  state->release();
  ASSERT_EQ(receipt.wait_for(3s), std::future_status::ready);
  receipt.get();
  expect_destroyed_once(state);
  EXPECT_FALSE(std::move(owner).retire().valid());
}

TEST(NativeEncodeOwnerTest, RecoveryRemainsPendingUntilTheConsumedOwnerIsDestroyed) {
  local_workers_t local;
  const auto state = local.state(true);
  auto owner = local.owner(state);
  release_guard_t release {state};
  std::future<void> receipt;
  scoped_operation_t retirement(state, [&]() {
    receipt = std::move(owner).retire();
  });
  ASSERT_EQ(state->entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(retirement.returned());
  retirement.finish();
  ASSERT_TRUE(receipt.valid());
  video::detail::encoder_recovery_t recovery;
  const video::detail::encoder_recovery_t::clock::time_point start {};
  recovery.start(std::move(receipt), start);
  EXPECT_EQ(recovery.poll(start), video::detail::encoder_recovery_t::state_t::pending);
  EXPECT_EQ(recovery.poll(start + 9s), video::detail::encoder_recovery_t::state_t::pending);
  EXPECT_FALSE(state->resources.expired());
  state->release();
  local.workers.seal_and_drain();
  EXPECT_EQ(recovery.poll(start + 9s), video::detail::encoder_recovery_t::state_t::ready);
  expect_destroyed_once(state);
}

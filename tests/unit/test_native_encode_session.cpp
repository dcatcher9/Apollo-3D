#include "src/native_encode_owner.h"
#include "src/nvenc/nvenc_base.h"
#include "src/platform/common.h"
#include "src/process.h"
#include "src/tracked_async_worker.h"
#include "src/video.h"
#include "src/video_encode_session.h"

#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

namespace {
  using namespace std::chrono_literals;

  struct backend_state_t {
    explicit backend_state_t(bool held):
        destructor_entered(entered_promise.get_future()),
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
    std::future<void> destructor_entered;
    std::shared_future<void> release_signal;
    std::atomic<bool> released {false}, gate_timed_out {false};
    std::atomic<unsigned> destroyed {0}, encoder_accesses {0};
    std::thread::id destructor_thread;
    std::weak_ptr<int> resources;
    platf::img_t *converted_image = nullptr;
    std::optional<std::chrono::steady_clock::time_point> encode_target;
    unsigned conversions = 0;
  };

  class inert_encoder_t final: public nvenc::nvenc_base {
  public:
    inert_encoder_t():
        nvenc_base(NV_ENC_DEVICE_TYPE_DIRECTX) {}

  private:
    bool init_library() override {
      ADD_FAILURE() << "Session fixtures must not load an encoder driver";
      return false;
    }

    bool create_and_register_input_buffer() override {
      ADD_FAILURE() << "Session fixtures must not register GPU inputs";
      return false;
    }

    void release_async_event() override {
      ADD_FAILURE() << "Session fixtures must not own native completion events";
    }
  };

  class ready_backend_t final: public platf::nvenc_encode_device_t {
  public:
    ready_backend_t(std::shared_ptr<backend_state_t> state, const video::sunshine_colorspace_t &selected_colorspace, const std::optional<SS_HDR_METADATA> &metadata):
        state_(std::move(state)),
        completion_ {state_},
        resources_(std::make_shared<int>(42)) {
      colorspace = selected_colorspace;
      hdr_metadata = metadata;
      state_->resources = resources_;
    }

    ~ready_backend_t() override {
      state_->destructor_thread = std::this_thread::get_id();
      state_->entered_promise.set_value();
      state_->gate_timed_out = state_->release_signal.wait_for(5s) != std::future_status::ready;
    }

    nvenc::nvenc_base &encoder() override {
      ++state_->encoder_accesses;
      return encoder_;
    }

    int convert(platf::img_t &image) override {
      state_->converted_image = &image;
      ++state_->conversions;
      return 17;
    }

    int convert_with_encode_target(platf::img_t &image, std::chrono::steady_clock::time_point target) override {
      state_->converted_image = &image;
      state_->encode_target = target;
      ++state_->conversions;
      return 23;
    }

    bool needs_conversion_poll() const override {
      return true;
    }

    std::optional<std::chrono::steady_clock::time_point> rendered_content_timestamp() const override {
      return std::chrono::steady_clock::time_point {123ms};
    }

  private:
    struct completion_marker_t {
      std::shared_ptr<backend_state_t> state;

      ~completion_marker_t() {
        ++state->destroyed;
      }
    };

    std::shared_ptr<backend_state_t> state_;
    completion_marker_t completion_;
    std::shared_ptr<int> resources_;
    inert_encoder_t encoder_;
  };

  class ready_display_t final: public platf::display_t {
  public:
    ready_display_t(std::shared_ptr<backend_state_t> state, video::detail::tracked_async_worker_pool_t &workers):
        state_(std::move(state)),
        workers_(workers) {}

    platf::native_encode_owner_t make_nvenc_encode_device(const video::config_t &config, const video::sunshine_colorspace_t &colorspace) override {
      ++factory_calls;
      offered_config = config;
      offered_colorspace = colorspace;
      auto candidate = platf::native_encode_owner_test_access::adopt(
        std::make_unique<ready_backend_t>(state_, colorspace, metadata),
        workers_
      );
      // Model a failed virtual ready-backend offer, not Windows create-stage internals.
      if (reject_offer) {
        return {};
      }
      return candidate;
    }

    bool is_hdr() override {
      return hdr_display;
    }

    platf::capture_e capture(const push_captured_image_cb_t &, const pull_free_image_cb_t &, bool *) override {
      ADD_FAILURE() << "Session fixtures must not start capture";
      return platf::capture_e::error;
    }

    std::shared_ptr<platf::img_t> alloc_img() override {
      ADD_FAILURE() << "Session fixtures must not allocate capture images";
      return {};
    }

    int dummy_img(platf::img_t *) override {
      ADD_FAILURE() << "Session fixtures must not create dummy GPU images";
      return -1;
    }

    bool hdr_display = false, reject_offer = false;
    unsigned factory_calls = 0;
    video::config_t offered_config {};
    video::sunshine_colorspace_t offered_colorspace {};
    std::optional<SS_HDR_METADATA> metadata;

  private:
    std::shared_ptr<backend_state_t> state_;
    video::detail::tracked_async_worker_pool_t &workers_;
  };

  struct fixture_t {
    explicit fixture_t(bool held = false):
        state(std::make_shared<backend_state_t>(held)),
        display(state, workers) {}

    ~fixture_t() {
      state->release();
      workers.seal_and_drain();
    }

    video::detail::tracked_async_worker_pool_t workers;
    std::shared_ptr<backend_state_t> state;
    ready_display_t display;
  };

  class scoped_operation_t {
  public:
    template<class Operation>
    scoped_operation_t(std::shared_ptr<backend_state_t> state, Operation operation):
        state_(std::move(state)) {
      std::packaged_task<void()> task(std::move(operation));
      returned_ = task.get_future();
      caller_ = std::thread(std::move(task));
    }

    ~scoped_operation_t() {
      state_->release();
      if (caller_.joinable()) {
        caller_.join();
      }
    }

    bool returned() {
      return returned_.wait_for(3s) == std::future_status::ready;
    }

    void finish() {
      caller_.join();
      returned_.get();
    }

  private:
    std::shared_ptr<backend_state_t> state_;
    std::future<void> returned_;
    std::thread caller_;
  };

  video::config_t stream_config() {
    video::config_t config {};
    config.width = 1920;
    config.height = 1080;
    config.framerate = 60;
    config.encodingFramerate = 60;
    config.bitrate = 50'000;
    config.videoFormat = 1;
    config.encoderCscMode = 3;  // Full-range Rec. 709.
    return config;
  }

  SS_HDR_METADATA metadata() {
    SS_HDR_METADATA result {};
    for (unsigned i = 0; i < 3; ++i) {
      result.displayPrimaries[i].x = 100 + i;
      result.displayPrimaries[i].y = 200 + i;
    }
    result.whitePoint.x = 300;
    result.whitePoint.y = 400;
    result.maxDisplayLuminance = 1500;
    result.minDisplayLuminance = 7;
    result.maxContentLightLevel = 1400;
    result.maxFrameAverageLightLevel = 500;
    result.maxFullFrameLuminance = 450;
    return result;
  }

  void expect_metadata(const SS_HDR_METADATA &actual, const SS_HDR_METADATA &expected) {
    for (unsigned i = 0; i < 3; ++i) {
      EXPECT_EQ(actual.displayPrimaries[i].x, expected.displayPrimaries[i].x);
      EXPECT_EQ(actual.displayPrimaries[i].y, expected.displayPrimaries[i].y);
    }
    EXPECT_EQ(actual.whitePoint.x, expected.whitePoint.x);
    EXPECT_EQ(actual.whitePoint.y, expected.whitePoint.y);
    EXPECT_EQ(actual.maxDisplayLuminance, expected.maxDisplayLuminance);
    EXPECT_EQ(actual.minDisplayLuminance, expected.minDisplayLuminance);
    EXPECT_EQ(actual.maxContentLightLevel, expected.maxContentLightLevel);
    EXPECT_EQ(actual.maxFrameAverageLightLevel, expected.maxFrameAverageLightLevel);
    EXPECT_EQ(actual.maxFullFrameLuminance, expected.maxFullFrameLuminance);
  }

  void expect_destroyed_once(const fixture_t &fixture) {
    EXPECT_EQ(fixture.state->destroyed.load(), 1u);
    EXPECT_TRUE(fixture.state->resources.expired());
    EXPECT_FALSE(fixture.state->gate_timed_out.load());
  }
}  // namespace

TEST(NativeEncodeSessionTest, FactoryReturnsTheActualForwardingWrapperAndDestructorRetiresIt) {
  fixture_t fixture;
  const auto config = stream_config();
  auto session = video::detail::make_native_encode_session_for_test(fixture.display, config);
  ASSERT_TRUE(session);
  EXPECT_EQ(fixture.display.factory_calls, 1u);
  EXPECT_EQ(fixture.display.offered_config.width, config.width);
  EXPECT_EQ(fixture.display.offered_config.height, config.height);
  EXPECT_EQ(fixture.display.offered_config.videoFormat, config.videoFormat);
  EXPECT_EQ(fixture.display.offered_config.bitrate, config.bitrate);
  EXPECT_EQ(fixture.display.offered_colorspace.colorspace, video::colorspace_e::rec709);
  EXPECT_EQ(fixture.display.offered_colorspace.bit_depth, 8u);
  EXPECT_TRUE(fixture.display.offered_colorspace.full_range);
  EXPECT_EQ(fixture.state->encoder_accesses.load(), 1u);
  platf::img_t first, second;
  EXPECT_EQ(session->convert(first), 17);
  EXPECT_EQ(fixture.state->converted_image, &first);
  const std::chrono::steady_clock::time_point target {456ms};
  EXPECT_EQ(session->convert_with_encode_target(second, target), 23);
  EXPECT_EQ(fixture.state->converted_image, &second);
  EXPECT_EQ(fixture.state->encode_target, target);
  EXPECT_EQ(fixture.state->conversions, 2u);
  EXPECT_TRUE(session->needs_conversion_poll());
  EXPECT_EQ(session->rendered_content_timestamp(), std::chrono::steady_clock::time_point {123ms});
  session.reset();
  fixture.workers.seal_and_drain();
  expect_destroyed_once(fixture);
}

TEST(NativeEncodeSessionTest, FailedReadyBackendOfferRetiresCandidateWithoutReturningAWrapper) {
  fixture_t fixture(true);
  fixture.display.reject_offer = true;
  std::unique_ptr<video::nvenc_encode_session_t> session;
  std::thread::id factory_thread;
  scoped_operation_t creation(fixture.state, [&]() {
    factory_thread = std::this_thread::get_id();
    session = video::detail::make_native_encode_session_for_test(fixture.display, stream_config());
  });
  ASSERT_EQ(fixture.state->destructor_entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(creation.returned()) << "Failed factory waited for candidate destruction";
  creation.finish();
  EXPECT_FALSE(session);
  EXPECT_EQ(fixture.display.factory_calls, 1u);
  EXPECT_EQ(fixture.state->encoder_accesses.load(), 0u);
  EXPECT_NE(fixture.state->destructor_thread, factory_thread);
  EXPECT_FALSE(fixture.state->resources.expired());
  fixture.state->release();
  fixture.workers.seal_and_drain();
  expect_destroyed_once(fixture);
}

TEST(NativeEncodeSessionTest, HdrSnapshotIsOwnedAndSurvivesConsumedSessionRetirement) {
  fixture_t fixture;
  fixture.display.hdr_display = true;
  fixture.display.metadata = metadata();
  auto config = stream_config();
  config.dynamicRange = 1;
  auto session = video::detail::make_native_encode_session_for_test(fixture.display, config);
  ASSERT_TRUE(session);
  EXPECT_EQ(fixture.display.offered_colorspace.colorspace, video::colorspace_e::bt2020);
  EXPECT_EQ(fixture.display.offered_colorspace.bit_depth, 10u);
  auto snapshot = video::detail::native_encode_session_hdr_info_for_test(*session);
  auto independent = video::detail::native_encode_session_hdr_info_for_test(*session);
  ASSERT_TRUE(snapshot);
  ASSERT_TRUE(independent);
  EXPECT_TRUE(snapshot->enabled);
  EXPECT_TRUE(independent->enabled);
  EXPECT_NE(snapshot.get(), independent.get());
  snapshot->metadata.maxDisplayLuminance = 1;
  expect_metadata(independent->metadata, metadata());
  auto completion = video::detail::retire_native_encode_session_for_test(std::move(session));
  EXPECT_FALSE(session);
  ASSERT_TRUE(completion.valid());
  ASSERT_EQ(completion.wait_for(3s), std::future_status::ready);
  completion.get();
  expect_destroyed_once(fixture);
  fixture.display.metadata.reset();
  expect_metadata(independent->metadata, metadata());
  EXPECT_EQ(snapshot->metadata.maxDisplayLuminance, 1u);
}

TEST(NativeEncodeSessionTest, MissingHdrMetadataIsNonfatalAndSdrDoesNotAdvertiseHdr) {
  for (const bool request_hdr : {true, false}) {
    SCOPED_TRACE(request_hdr ? "HDR without metadata" : "SDR with unused metadata");
    fixture_t fixture;
    fixture.display.hdr_display = true;
    if (!request_hdr) {
      fixture.display.metadata = metadata();
    }
    auto config = stream_config();
    config.dynamicRange = request_hdr ? 1 : 0;
    auto session = video::detail::make_native_encode_session_for_test(fixture.display, config);
    ASSERT_TRUE(session);
    auto snapshot = video::detail::native_encode_session_hdr_info_for_test(*session);
    ASSERT_TRUE(snapshot);
    EXPECT_FALSE(snapshot->enabled);
    expect_metadata(snapshot->metadata, SS_HDR_METADATA {});
    auto completion = video::detail::retire_native_encode_session_for_test(std::move(session));
    ASSERT_TRUE(completion.valid());
    ASSERT_EQ(completion.wait_for(3s), std::future_status::ready);
    completion.get();
    expect_destroyed_once(fixture);
  }
}

TEST(NativeEncodeSessionTest, LiveHdrAdmissionRejectsIncoherenceBeforeTheProbeFactory) {
  fixture_t fixture;
  auto config = stream_config();
  config.dynamicRange = proc::proc.get_status().enable_hdr ? 0 : 1;
  auto rejected = video::detail::make_native_encode_session_for_test(fixture.display, config, true);
  ASSERT_FALSE(rejected);
  ASSERT_EQ(fixture.display.factory_calls, 0u);
  EXPECT_EQ(fixture.state->encoder_accesses.load(), 0u);
  auto probe = video::detail::make_native_encode_session_for_test(fixture.display, config, false);
  ASSERT_TRUE(probe);
  EXPECT_EQ(fixture.display.factory_calls, 1u);
  auto completion = video::detail::retire_native_encode_session_for_test(std::move(probe));
  ASSERT_TRUE(completion.valid());
  ASSERT_EQ(completion.wait_for(3s), std::future_status::ready);
  completion.get();
  expect_destroyed_once(fixture);
}

TEST(NativeEncodeSessionTest, ConsumingWrapperRetirementReceiptWaitsForTheBackend) {
  fixture_t fixture(true);
  auto session = video::detail::make_native_encode_session_for_test(fixture.display, stream_config());
  ASSERT_TRUE(session);
  std::future<void> receipt;
  scoped_operation_t retirement(fixture.state, [&]() {
    receipt = video::detail::retire_native_encode_session_for_test(std::move(session));
  });
  ASSERT_EQ(fixture.state->destructor_entered.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(retirement.returned()) << "Wrapper retirement waited for backend destruction";
  retirement.finish();
  EXPECT_FALSE(session);
  ASSERT_TRUE(receipt.valid());
  EXPECT_EQ(receipt.wait_for(0ms), std::future_status::timeout);
  EXPECT_FALSE(fixture.state->resources.expired());
  fixture.state->release();
  ASSERT_EQ(receipt.wait_for(3s), std::future_status::ready);
  receipt.get();
  fixture.workers.seal_and_drain();
  expect_destroyed_once(fixture);
  EXPECT_FALSE(video::detail::retire_native_encode_session_for_test({}).valid());
}

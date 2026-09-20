#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <src/nvenc/nvenc_base.h>
#include <src/video_session_recovery.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

  // Exercise the production create/destroy path without a driver or GPU. Each callback checks
  // ownership acquired by an earlier successful call, including partially initialized sessions.
  class nvenc_lifecycle_probe final: public nvenc::nvenc_base {
  public:
    enum class failure_t {
      none,
      codec_query,
      initialize,
      event_registration,
      bitstream_creation,
      input_registration,
    };

    explicit nvenc_lifecycle_probe(failure_t failure = failure_t::none):
        nvenc_base(NV_ENC_DEVICE_TYPE_DIRECTX),
        failure(failure) {
      device = this;
      async_event_handle = this;
      nvenc = std::make_shared<NV_ENCODE_API_FUNCTION_LIST>();
      nvenc->nvEncOpenEncodeSessionEx = open_session;
      nvenc->nvEncGetEncodeGUIDCount = codec_count;
      nvenc->nvEncGetEncodeGUIDs = codecs;
      nvenc->nvEncGetEncodeCaps = capability;
      nvenc->nvEncGetEncodePresetConfigEx = preset;
      nvenc->nvEncInitializeEncoder = initialize;
      nvenc->nvEncRegisterAsyncEvent = register_event;
      nvenc->nvEncUnregisterAsyncEvent = unregister_event;
      nvenc->nvEncCreateBitstreamBuffer = create_bitstream;
      nvenc->nvEncDestroyBitstreamBuffer = destroy_bitstream;
      nvenc->nvEncUnregisterResource = unregister_input;
      nvenc->nvEncDestroyEncoder = destroy_session;
      nvenc->nvEncMapInputResource = map_input;
      nvenc->nvEncEncodePicture = submit_picture;
      nvenc->nvEncLockBitstream = lock_bitstream;
      nvenc->nvEncUnlockBitstream = unlock_bitstream;
      nvenc->nvEncUnmapInputResource = unmap_input;
    }

    ~nvenc_lifecycle_probe() override {
      // Fatal assertions before a scoped teardown worker exists must also release fake waits.
      release_all_waits();
      destroy_encoder();
    }

    bool create() {
      video::config_t stream {};
      stream.width = 1920;
      stream.height = 1080;
      stream.framerate = 60;
      stream.bitrate = 50'000;
      stream.slicesPerFrame = 1;
      stream.videoFormat = 1;
      nvenc::nvenc_config settings;
      settings.hevc_unidirectional_b = false;
      const nvenc::nvenc_colorspace_t colorspace {
        NV_ENC_VUI_COLOR_PRIMARIES_BT709,
        NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709,
        NV_ENC_VUI_MATRIX_COEFFS_BT709,
        false,
      };
      return create_encoder(settings, stream, colorspace, NV_ENC_BUFFER_FORMAT_NV12, std::nullopt);
    }

    failure_t failure;
    int registration_calls = 0;
    int unregistration_calls = 0;
    int sessions_destroyed = 0;
    bool event_registered = false;
    bool bitstream_owned = false;
    bool input_owned = false;
    std::vector<std::string> cleanup;

    struct snapshot_t {
      bool input_mapped;
      bool picture_pending;
      bool bitstream_locked;
      bool frame_event_registered;
      bool flush_event_registered;
      bool bitstream_owned;
      bool input_owned;
      int open_calls;
      int map_calls;
      int submit_calls;
      int frame_wait_calls;
      int flush_wait_calls;
      int lock_calls;
      int unlock_calls;
      int unmap_calls;
      int eos_calls;
      int sessions_destroyed;
      std::vector<std::uint32_t> frame_wait_timeouts;
      std::vector<std::string> operations;
    };

    snapshot_t snapshot() {
      std::lock_guard lock(mutex);
      return {
        input_mapped,
        picture_pending,
        bitstream_locked,
        event_registered,
        flush_registered,
        bitstream_owned,
        input_owned,
        open_calls,
        map_calls,
        submit_calls,
        frame_wait_calls,
        flush_wait_calls,
        lock_calls,
        unlock_calls,
        unmap_calls,
        eos_calls,
        sessions_destroyed,
        frame_wait_timeouts,
        operations,
      };
    }

    void defer_frame_completion() {
      defer_frame = true;
    }

    void defer_flush_completion() {
      defer_flush = true;
    }

    void defer_session_destruction() {
      defer_destroy = true;
    }

    void complete_frame() {
      std::lock_guard lock(mutex);
      frame_released = true;
      condition.notify_all();
    }

    void release_all_waits() {
      std::lock_guard lock(mutex);
      frame_released = true;
      flush_released = true;
      destroy_released = true;
      condition.notify_all();
    }

    enum class boundary_t {
      frame_wait,
      flush_wait,
      destroy_session
    };

    bool await_boundary(boundary_t boundary) {
      std::unique_lock lock(mutex);
      return condition.wait_for(lock, std::chrono::seconds(3), [&] {
        return boundary == boundary_t::frame_wait ? teardown_frame_wait_entered :
               boundary == boundary_t::flush_wait ? flush_wait_entered :
                                                    destroy_entered;
      });
    }

    NVENCSTATUS submission_status = NV_ENC_SUCCESS;
    int lock_failures = 0;
    int unlock_failures = 0;
    int unmap_failures = 0;
    int frame_timeouts_remaining = 0;
    int flush_timeouts_remaining = 0;
    std::optional<nvenc::nvenc_event_wait_result> next_frame_wait_result;
    std::chrono::milliseconds extra_frame_timeout_elapsed {0};

  protected:
    bool init_library() override {
      ADD_FAILURE() << "The fixture must not load a real encoder driver";
      return false;
    }

    bool create_and_register_input_buffer() override {
      std::lock_guard lock(mutex);
      if (failure == failure_t::input_registration) {
        return false;
      }
      input_owned = true;
      registered_input_buffer = this;
      return true;
    }

    void release_async_event() override {
      async_event_handle = nullptr;
    }

    void *create_flush_event() override {
      return &flush_token;
    }

    std::chrono::steady_clock::time_point async_wait_clock_now() const override {
      std::lock_guard lock(mutex);
      return fake_now;
    }

    nvenc::nvenc_event_wait_result wait_for_async_event(std::uint32_t timeout_ms) override {
      std::unique_lock lock(mutex);
      ++frame_wait_calls;
      frame_wait_timeouts.push_back(timeout_ms);
      EXPECT_TRUE(event_registered);
      EXPECT_TRUE(input_mapped);
      EXPECT_TRUE(picture_pending);
      if (frame_event_consumed) {
        ADD_FAILURE() << "An auto-reset picture event cannot be consumed twice";
        return {nvenc::nvenc_event_wait_status::ready};  // Report the ownership error without hanging teardown.
      }
      if (next_frame_wait_result) {
        const auto result = *std::exchange(next_frame_wait_result, std::nullopt);
        if (result.status == nvenc::nvenc_event_wait_status::timeout) {
          fake_now += std::chrono::milliseconds(timeout_ms) + extra_frame_timeout_elapsed;
        }
        if (result.status != nvenc::nvenc_event_wait_status::ready) {
          operations.emplace_back(result.status == nvenc::nvenc_event_wait_status::timeout ? "frame-timeout" : "frame-wait-failed");
          return result;
        }
      }
      if (frame_timeouts_remaining > 0) {
        --frame_timeouts_remaining;
        fake_now += std::chrono::milliseconds(timeout_ms) + extra_frame_timeout_elapsed;
        operations.emplace_back("frame-timeout");
        return {nvenc::nvenc_event_wait_status::timeout, 0x102};
      }
      if (!eos_submitted && ((defer_frame && !frame_released) || submission_status == NV_ENC_ERR_NEED_MORE_INPUT)) {
        fake_now += std::chrono::milliseconds(timeout_ms) + extra_frame_timeout_elapsed;
        operations.emplace_back("frame-timeout");
        return {nvenc::nvenc_event_wait_status::timeout, 0x102};
      }
      if (defer_frame && !frame_released) {
        teardown_frame_wait_entered = true;
        condition.notify_all();
        if (!condition.wait_for(lock, std::chrono::seconds(10), [&] {
              return frame_released;
            })) {
          ADD_FAILURE() << "Test failed to release delayed picture completion";
        }
      }
      EXPECT_TRUE(submission_status != NV_ENC_ERR_NEED_MORE_INPUT || eos_submitted)
        << "Buffered input must be flushed before waiting for its output";
      frame_event_consumed = true;
      frame_completed = true;
      operations.emplace_back("frame-ready");
      return {nvenc::nvenc_event_wait_status::ready};
    }

    nvenc::nvenc_event_wait_result wait_for_flush_event(std::uint32_t timeout_ms) override {
      std::unique_lock lock(mutex);
      ++flush_wait_calls;
      EXPECT_TRUE(flush_registered);
      EXPECT_TRUE(eos_submitted);
      EXPECT_FALSE(picture_pending);
      EXPECT_FALSE(bitstream_locked);
      EXPECT_FALSE(input_mapped);
      EXPECT_TRUE(event_registered);
      EXPECT_TRUE(bitstream_owned);
      EXPECT_TRUE(input_owned);
      if (flush_timeouts_remaining > 0) {
        --flush_timeouts_remaining;
        EXPECT_EQ(timeout_ms, 100u);
        fake_now += std::chrono::milliseconds(timeout_ms);
        operations.emplace_back("flush-timeout");
        return {nvenc::nvenc_event_wait_status::timeout, 0x102};
      }
      if (defer_flush && !flush_released) {
        flush_wait_entered = true;
        condition.notify_all();
        if (!condition.wait_for(lock, std::chrono::seconds(10), [&] {
              return flush_released;
            })) {
          ADD_FAILURE() << "Test failed to release delayed EOS completion";
        }
      }
      flush_completed = true;
      operations.emplace_back("flush-ready");
      return {nvenc::nvenc_event_wait_status::ready};
    }

  private:
    mutable std::mutex mutex;
    std::chrono::steady_clock::time_point fake_now {};
    std::condition_variable condition;
    int flush_token = 0;
    bool flush_registered = false;
    bool input_mapped = false;
    bool picture_pending = false;
    bool bitstream_locked = false;
    bool frame_completed = false;
    bool frame_event_consumed = false;
    bool eos_submitted = false;
    bool flush_completed = false;
    bool accepted_picture = false;
    bool defer_frame = false;
    bool defer_flush = false;
    bool defer_destroy = false;
    bool frame_released = false;
    bool flush_released = false;
    bool destroy_released = false;
    bool teardown_frame_wait_entered = false;
    bool flush_wait_entered = false;
    bool destroy_entered = false;
    int open_calls = 0;
    int map_calls = 0;
    int submit_calls = 0;
    int frame_wait_calls = 0;
    int flush_wait_calls = 0;
    int lock_calls = 0;
    int unlock_calls = 0;
    int unmap_calls = 0;
    int eos_calls = 0;
    std::uint64_t picture_index = 0;
    std::uint8_t encoded_bytes[4] {0, 0, 1, 0x26};
    std::vector<std::uint32_t> frame_wait_timeouts;
    std::vector<std::string> operations;

    static nvenc_lifecycle_probe &self(void *encoder) {
      return *static_cast<nvenc_lifecycle_probe *>(encoder);
    }

    static NVENCSTATUS NVENCAPI open_session(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *params, void **encoder) {
      auto &probe = self(params->device);
      std::lock_guard lock(probe.mutex);
      ++probe.open_calls;
      probe.accepted_picture = false;
      probe.eos_submitted = false;
      probe.flush_completed = false;
      *encoder = params->device;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI codec_count(void *encoder, std::uint32_t *count) {
      if (self(encoder).failure == failure_t::codec_query) {
        return NV_ENC_ERR_GENERIC;
      }
      *count = 1;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI codecs(void *, GUID *guids, std::uint32_t, std::uint32_t *count) {
      guids[0] = NV_ENC_CODEC_HEVC_GUID;
      *count = 1;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI capability(void *, GUID, NV_ENC_CAPS_PARAM *params, int *value) {
      switch (params->capsToQuery) {
        case NV_ENC_CAPS_WIDTH_MAX:
        case NV_ENC_CAPS_HEIGHT_MAX:
          *value = 8192;
          break;
        case NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT:
        case NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES:
          *value = 1;
          break;
        default:
          *value = 0;
      }
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI preset(void *, GUID, GUID, NV_ENC_TUNING_INFO, NV_ENC_PRESET_CONFIG *) {
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI initialize(void *encoder, NV_ENC_INITIALIZE_PARAMS *) {
      return self(encoder).failure == failure_t::initialize ? NV_ENC_ERR_GENERIC : NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI register_event(void *encoder, NV_ENC_EVENT_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      if (params->completionEvent == &probe.flush_token) {
        EXPECT_FALSE(probe.flush_registered);
        probe.flush_registered = true;
        probe.operations.emplace_back("register-flush");
        return NV_ENC_SUCCESS;
      }
      EXPECT_EQ(params->completionEvent, encoder);
      ++probe.registration_calls;
      if (probe.failure == failure_t::event_registration) {
        return NV_ENC_ERR_GENERIC;
      }
      EXPECT_FALSE(probe.event_registered);
      probe.event_registered = true;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unregister_event(void *encoder, NV_ENC_EVENT_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_FALSE(probe.picture_pending);
      EXPECT_TRUE(!probe.accepted_picture || probe.flush_completed);
      if (params->completionEvent == &probe.flush_token) {
        EXPECT_TRUE(probe.flush_registered);
        probe.flush_registered = false;
        probe.operations.emplace_back("unregister-flush");
        return NV_ENC_SUCCESS;
      }
      ++probe.unregistration_calls;
      EXPECT_TRUE(probe.event_registered) << "Unregister requires successful registration";
      probe.event_registered = false;
      probe.cleanup.emplace_back("event");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI create_bitstream(void *encoder, NV_ENC_CREATE_BITSTREAM_BUFFER *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      if (probe.failure == failure_t::bitstream_creation) {
        return NV_ENC_ERR_GENERIC;
      }
      EXPECT_FALSE(probe.bitstream_owned);
      probe.bitstream_owned = true;
      params->bitstreamBuffer = encoder;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI destroy_bitstream(void *encoder, NV_ENC_OUTPUT_PTR) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_FALSE(probe.picture_pending);
      EXPECT_FALSE(probe.bitstream_locked);
      EXPECT_TRUE(!probe.accepted_picture || probe.flush_completed);
      EXPECT_TRUE(probe.bitstream_owned);
      probe.bitstream_owned = false;
      probe.cleanup.emplace_back("bitstream");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unregister_input(void *encoder, NV_ENC_REGISTERED_PTR) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_FALSE(probe.input_mapped);
      EXPECT_FALSE(probe.picture_pending);
      EXPECT_TRUE(probe.input_owned);
      probe.input_owned = false;
      probe.cleanup.emplace_back("input");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI destroy_session(void *encoder) {
      auto &probe = self(encoder);
      std::unique_lock lock(probe.mutex);
      EXPECT_FALSE(probe.event_registered);
      EXPECT_FALSE(probe.flush_registered);
      EXPECT_FALSE(probe.bitstream_owned);
      EXPECT_FALSE(probe.input_owned);
      if (probe.defer_destroy && !probe.destroy_released) {
        probe.destroy_entered = true;
        probe.condition.notify_all();
        if (!probe.condition.wait_for(lock, std::chrono::seconds(10), [&] {
              return probe.destroy_released;
            })) {
          ADD_FAILURE() << "Test failed to release NvEncDestroyEncoder completion";
        }
      }
      ++probe.sessions_destroyed;
      probe.cleanup.emplace_back("session");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI map_input(void *encoder, NV_ENC_MAP_INPUT_RESOURCE *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      ++probe.map_calls;
      EXPECT_TRUE(probe.input_owned);
      EXPECT_FALSE(probe.input_mapped);
      EXPECT_FALSE(probe.picture_pending);
      probe.input_mapped = true;
      params->mappedResource = encoder;
      params->mappedBufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      probe.operations.emplace_back("map");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI submit_picture(void *encoder, NV_ENC_PIC_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      if (params->encodePicFlags & NV_ENC_PIC_FLAG_EOS) {
        EXPECT_TRUE(probe.accepted_picture);
        EXPECT_TRUE(probe.flush_registered);
        EXPECT_EQ(params->completionEvent, &probe.flush_token);
        EXPECT_NE(params->completionEvent, probe.async_event_handle);
        EXPECT_FALSE(probe.eos_submitted);
        probe.eos_submitted = true;
        ++probe.eos_calls;
        probe.operations.emplace_back("eos");
        return NV_ENC_SUCCESS;
      }
      EXPECT_TRUE(probe.input_mapped);
      EXPECT_FALSE(probe.picture_pending);
      EXPECT_EQ(params->inputBuffer, encoder);
      EXPECT_EQ(params->completionEvent, probe.async_event_handle);
      ++probe.submit_calls;
      probe.operations.emplace_back("submit");
      if (probe.submission_status == NV_ENC_SUCCESS || probe.submission_status == NV_ENC_ERR_NEED_MORE_INPUT) {
        probe.accepted_picture = true;
        probe.picture_pending = true;
        probe.frame_completed = false;
        probe.frame_event_consumed = false;
        probe.picture_index = params->inputTimeStamp;
      }
      return probe.submission_status;
    }

    static NVENCSTATUS NVENCAPI lock_bitstream(void *encoder, NV_ENC_LOCK_BITSTREAM *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      ++probe.lock_calls;
      EXPECT_TRUE(probe.picture_pending);
      EXPECT_TRUE(probe.frame_completed);
      EXPECT_TRUE(probe.input_mapped);
      EXPECT_FALSE(probe.bitstream_locked);
      if (probe.lock_failures > 0) {
        --probe.lock_failures;
        probe.operations.emplace_back("lock-failed");
        return NV_ENC_ERR_LOCK_BUSY;
      }
      probe.bitstream_locked = true;
      params->bitstreamBufferPtr = probe.encoded_bytes;
      params->bitstreamSizeInBytes = sizeof(probe.encoded_bytes);
      params->outputTimeStamp = probe.picture_index;
      params->pictureType = NV_ENC_PIC_TYPE_IDR;
      probe.operations.emplace_back("lock");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unlock_bitstream(void *encoder, NV_ENC_OUTPUT_PTR) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      ++probe.unlock_calls;
      EXPECT_TRUE(probe.bitstream_locked);
      EXPECT_TRUE(probe.input_mapped);
      if (probe.unlock_failures > 0) {
        --probe.unlock_failures;
        probe.operations.emplace_back("unlock-failed");
        return NV_ENC_ERR_GENERIC;
      }
      probe.bitstream_locked = false;
      probe.picture_pending = false;
      probe.operations.emplace_back("unlock");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unmap_input(void *encoder, NV_ENC_INPUT_PTR) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      ++probe.unmap_calls;
      EXPECT_TRUE(probe.input_mapped);
      EXPECT_FALSE(probe.bitstream_locked);
      EXPECT_FALSE(probe.picture_pending);
      if (probe.unmap_failures > 0) {
        --probe.unmap_failures;
        probe.operations.emplace_back("unmap-failed");
        return NV_ENC_ERR_GENERIC;
      }
      probe.input_mapped = false;
      probe.operations.emplace_back("unmap");
      return NV_ENC_SUCCESS;
    }
  };

  // Every assertion may return early. Release both fake device gates before joining the owner.
  class scoped_encoder_teardown {
  public:
    explicit scoped_encoder_teardown(nvenc_lifecycle_probe &probe):
        probe(probe) {
      std::packaged_task<void()> teardown([&probe] {
        probe.destroy_encoder();
      });
      completion = teardown.get_future();
      worker = std::thread(std::move(teardown));
    }

    std::future<void> take_completion() {
      return std::move(completion);
    }

    ~scoped_encoder_teardown() {
      probe.release_all_waits();
      worker.join();
    }

  private:
    nvenc_lifecycle_probe &probe;
    std::future<void> completion;
    std::thread worker;
  };

}  // namespace

TEST(NvencLifecycleTest, FailureBeforeEventRegistrationDoesNotUnregisterUnownedEvent) {
  for (const auto failure : {nvenc_lifecycle_probe::failure_t::codec_query, nvenc_lifecycle_probe::failure_t::initialize, nvenc_lifecycle_probe::failure_t::event_registration}) {
    nvenc_lifecycle_probe probe(failure);
    EXPECT_FALSE(probe.create());
    EXPECT_EQ(probe.unregistration_calls, 0);
    EXPECT_EQ(probe.sessions_destroyed, 1);
    EXPECT_EQ(probe.cleanup, (std::vector<std::string> {"session"}));
    probe.destroy_encoder();
    EXPECT_EQ(probe.sessions_destroyed, 1);
  }
}

TEST(NvencLifecycleTest, FailureAfterEventRegistrationReleasesOnlyAcquiredResources) {
  nvenc_lifecycle_probe bitstream_failure(nvenc_lifecycle_probe::failure_t::bitstream_creation);
  EXPECT_FALSE(bitstream_failure.create());
  EXPECT_EQ(bitstream_failure.registration_calls, 1);
  EXPECT_EQ(bitstream_failure.unregistration_calls, 1);
  EXPECT_EQ(bitstream_failure.cleanup, (std::vector<std::string> {"event", "session"}));

  nvenc_lifecycle_probe input_failure(nvenc_lifecycle_probe::failure_t::input_registration);
  EXPECT_FALSE(input_failure.create());
  EXPECT_EQ(input_failure.unregistration_calls, 1);
  EXPECT_EQ(input_failure.cleanup, (std::vector<std::string> {"bitstream", "event", "session"}));
}

TEST(NvencLifecycleTest, RepeatedCreateAndDestroyReleaseEachRegistrationOnce) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_TRUE(probe.create());
  EXPECT_EQ(probe.registration_calls, 2);
  EXPECT_EQ(probe.unregistration_calls, 1);
  EXPECT_EQ(probe.sessions_destroyed, 1);

  probe.destroy_encoder();
  probe.destroy_encoder();
  EXPECT_EQ(probe.unregistration_calls, 2);
  EXPECT_EQ(probe.sessions_destroyed, 2);
  EXPECT_EQ(probe.cleanup, (std::vector<std::string> {
                             "bitstream",
                             "event",
                             "input",
                             "session",
                             "bitstream",
                             "event",
                             "input",
                             "session",
                           }));
}

TEST(NvencLifecycleTest, TimedOutPictureKeepsWholeOwnerUntilCompletionAndDestroyReturns) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.defer_frame_completion();
  probe.defer_session_destruction();

  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto timed_out = probe.snapshot();
  ASSERT_EQ(timed_out.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25, 25, 25, 25, 25, 25}));
  EXPECT_TRUE(timed_out.input_mapped);
  EXPECT_TRUE(timed_out.picture_pending);
  EXPECT_EQ(timed_out.lock_calls, 0);
  EXPECT_EQ(timed_out.unmap_calls, 0);
  // A second submission must not overwrite the retained surface or open another session.
  EXPECT_TRUE(probe.encode_frame(2, false).data.empty());
  EXPECT_EQ(probe.snapshot().map_calls, 1);
  EXPECT_EQ(probe.snapshot().submit_calls, 1);
  EXPECT_EQ(probe.snapshot().frame_wait_calls, 7);
  nvenc_lifecycle_probe replacement;
  EXPECT_FALSE(replacement.create());
  EXPECT_EQ(replacement.snapshot().open_calls, 0);

  video::detail::encoder_recovery_t recovery;
  const auto recovery_started = std::chrono::steady_clock::now();
  {
    scoped_encoder_teardown teardown(probe);
    recovery.start(teardown.take_completion(), recovery_started);
    ASSERT_TRUE(probe.await_boundary(nvenc_lifecycle_probe::boundary_t::frame_wait));
    EXPECT_EQ(recovery.poll(recovery_started), video::detail::encoder_recovery_t::state_t::pending);
    const auto waiting = probe.snapshot();
    EXPECT_TRUE(waiting.input_mapped);
    EXPECT_TRUE(waiting.picture_pending);
    EXPECT_TRUE(waiting.frame_event_registered);
    EXPECT_TRUE(waiting.flush_event_registered);
    EXPECT_TRUE(waiting.bitstream_owned);
    EXPECT_TRUE(waiting.input_owned);
    EXPECT_EQ(waiting.unmap_calls, 0);
    EXPECT_EQ(waiting.sessions_destroyed, 0);
    EXPECT_EQ(waiting.eos_calls, 1);
    EXPECT_FALSE(replacement.create());

    probe.complete_frame();
    ASSERT_TRUE(probe.await_boundary(nvenc_lifecycle_probe::boundary_t::destroy_session));
    const auto destroying = probe.snapshot();
    EXPECT_FALSE(destroying.input_mapped);
    EXPECT_EQ(destroying.lock_calls, 1);
    EXPECT_EQ(destroying.unlock_calls, 1);
    EXPECT_EQ(destroying.unmap_calls, 1);
    EXPECT_EQ(destroying.flush_wait_calls, 1);
    EXPECT_EQ(destroying.sessions_destroyed, 0);
    // Driver destruction is still running even though its child resources have been released.
    EXPECT_EQ(recovery.poll(recovery_started), video::detail::encoder_recovery_t::state_t::pending);
    EXPECT_FALSE(replacement.create());
    EXPECT_EQ(replacement.snapshot().open_calls, 0);
  }

  EXPECT_EQ(probe.snapshot().sessions_destroyed, 1);
  EXPECT_EQ(recovery.poll(recovery_started), video::detail::encoder_recovery_t::state_t::ready);
  EXPECT_TRUE(replacement.create());
  EXPECT_EQ(replacement.snapshot().open_calls, 1);
}

TEST(NvencLifecycleTest, RepeatedPictureAndEosWaitTimeoutsReopenAdmissionAfterCleanup) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  // Seven waits exhaust the 250 ms streaming budget; two further timeouts exercise teardown's
  // retry loop before picture completion. Every fake timeout advances the monotonic clock.
  probe.frame_timeouts_remaining = 9;
  probe.flush_timeouts_remaining = 2;
  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());

  nvenc_lifecycle_probe replacement;
  EXPECT_FALSE(replacement.create());
  EXPECT_EQ(replacement.snapshot().open_calls, 0);

  probe.destroy_encoder();
  const auto cleaned = probe.snapshot();
  EXPECT_EQ(cleaned.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25, 25, 25, 25, 25, 25, 100, 100, 100}));
  EXPECT_EQ(cleaned.flush_wait_calls, 3);
  EXPECT_EQ(cleaned.eos_calls, 1);
  EXPECT_EQ(cleaned.lock_calls, 1);
  EXPECT_EQ(cleaned.unlock_calls, 1);
  EXPECT_EQ(cleaned.unmap_calls, 1);
  EXPECT_EQ(cleaned.sessions_destroyed, 1);
  EXPECT_FALSE(cleaned.input_mapped);
  EXPECT_FALSE(cleaned.frame_event_registered);
  EXPECT_FALSE(cleaned.flush_event_registered);
  EXPECT_TRUE(replacement.create());
  EXPECT_EQ(replacement.snapshot().open_calls, 1);
}

TEST(NvencLifecycleTest, SoftTimeoutCompletesTheSamePictureOnceAndAllowsTheNextFrame) {
  for (const auto device_reason : {std::optional<std::int32_t> {}, std::optional<std::int32_t> {0}, std::optional<std::int32_t> {1}}) {
    SCOPED_TRACE(device_reason ? std::to_string(*device_reason) : "device status unavailable");
    nvenc_lifecycle_probe probe;
    ASSERT_TRUE(probe.create());
    probe.next_frame_wait_result = nvenc::nvenc_event_wait_result {nvenc::nvenc_event_wait_status::timeout, 0x102, 0, device_reason};

    const auto recovered = probe.encode_frame(7, true);
    ASSERT_FALSE(recovered.data.empty());
    EXPECT_EQ(recovered.frame_index, 7u);
    const auto completed = probe.snapshot();
    EXPECT_EQ(completed.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25}));
    EXPECT_EQ(completed.map_calls, 1);
    EXPECT_EQ(completed.submit_calls, 1);
    EXPECT_EQ(completed.lock_calls, 1);
    EXPECT_EQ(completed.unlock_calls, 1);
    EXPECT_EQ(completed.unmap_calls, 1);
    EXPECT_EQ(std::count(completed.operations.begin(), completed.operations.end(), "frame-ready"), 1);
    EXPECT_FALSE(completed.input_mapped);
    EXPECT_FALSE(completed.picture_pending);
    EXPECT_EQ(completed.eos_calls, 0);

    const auto next = probe.encode_frame(8, false);
    ASSERT_FALSE(next.data.empty());
    EXPECT_EQ(next.frame_index, 8u);
    const auto continued = probe.snapshot();
    EXPECT_EQ(continued.map_calls, 2);
    EXPECT_EQ(continued.submit_calls, 2);
    EXPECT_EQ(continued.lock_calls, 2);
    EXPECT_EQ(continued.unlock_calls, 2);
    EXPECT_EQ(continued.unmap_calls, 2);
    EXPECT_EQ(continued.open_calls, 1);
    EXPECT_EQ(continued.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25, 100}));
  }
}

TEST(NvencLifecycleTest, CompletionCanArriveAfterSeveralSoftTimeoutSlicesWithoutResubmission) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.frame_timeouts_remaining = 6;

  EXPECT_FALSE(probe.encode_frame(1, true).data.empty());
  const auto completed = probe.snapshot();
  EXPECT_EQ(completed.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25, 25, 25, 25, 25, 25}));
  EXPECT_EQ(completed.map_calls, 1);
  EXPECT_EQ(completed.submit_calls, 1);
  EXPECT_EQ(completed.lock_calls, 1);
  EXPECT_EQ(completed.unmap_calls, 1);
  EXPECT_FALSE(completed.input_mapped);
  EXPECT_FALSE(completed.picture_pending);
}

TEST(NvencLifecycleTest, NativeWaitFailureDoesNotEnterSoftTimeoutRecovery) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  // WAIT_FAILED with ERROR_INVALID_HANDLE must stay a native wait error, not become a timeout.
  probe.next_frame_wait_result = nvenc::nvenc_event_wait_result {nvenc::nvenc_event_wait_status::failed, 0xFFFFFFFF, 6, 0};

  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto failed = probe.snapshot();
  EXPECT_EQ(failed.frame_wait_timeouts, (std::vector<std::uint32_t> {100}));
  EXPECT_EQ(failed.map_calls, 1);
  EXPECT_EQ(failed.submit_calls, 1);
  EXPECT_EQ(failed.lock_calls, 0);
  EXPECT_EQ(failed.unmap_calls, 0);
  EXPECT_TRUE(failed.input_mapped);
  EXPECT_TRUE(failed.picture_pending);

  probe.destroy_encoder();
  EXPECT_EQ(probe.snapshot().sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, RemovedDeviceDoesNotEnterSoftTimeoutRecovery) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  // DXGI_ERROR_DEVICE_REMOVED remains terminal even when the event itself reports WAIT_TIMEOUT.
  probe.next_frame_wait_result = nvenc::nvenc_event_wait_result {
    nvenc::nvenc_event_wait_status::timeout,
    0x102,
    0,
    static_cast<std::int32_t>(0x887A0005u)
  };

  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto removed = probe.snapshot();
  EXPECT_EQ(removed.frame_wait_timeouts, (std::vector<std::uint32_t> {100}));
  EXPECT_EQ(removed.map_calls, 1);
  EXPECT_EQ(removed.submit_calls, 1);
  EXPECT_EQ(removed.lock_calls, 0);
  EXPECT_EQ(removed.unmap_calls, 0);
  EXPECT_TRUE(removed.input_mapped);
  EXPECT_TRUE(removed.picture_pending);

  probe.destroy_encoder();
  EXPECT_EQ(probe.snapshot().sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, InitialWaitSchedulingOverrunDoesNotStartAnotherWait) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.frame_timeouts_remaining = 1;
  probe.extra_frame_timeout_elapsed = std::chrono::milliseconds(151);

  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto expired = probe.snapshot();
  EXPECT_EQ(expired.frame_wait_timeouts, (std::vector<std::uint32_t> {100}));
  EXPECT_EQ(expired.map_calls, 1);
  EXPECT_EQ(expired.submit_calls, 1);
  EXPECT_EQ(expired.lock_calls, 0);
  EXPECT_EQ(expired.unmap_calls, 0);
  EXPECT_TRUE(expired.input_mapped);
  EXPECT_TRUE(expired.picture_pending);
}

TEST(NvencLifecycleTest, FinalSoftTimeoutSliceUsesOnlyTheRemainingBudget) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.frame_timeouts_remaining = 7;
  probe.extra_frame_timeout_elapsed = std::chrono::milliseconds(1);

  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto expired = probe.snapshot();
  EXPECT_EQ(expired.frame_wait_timeouts, (std::vector<std::uint32_t> {100, 25, 25, 25, 25, 25, 19}));
  EXPECT_EQ(expired.map_calls, 1);
  EXPECT_EQ(expired.submit_calls, 1);
  EXPECT_EQ(expired.lock_calls, 0);
  EXPECT_EQ(expired.unmap_calls, 0);
  EXPECT_TRUE(expired.input_mapped);
  EXPECT_TRUE(expired.picture_pending);
}

TEST(NvencLifecycleTest, CompletedPictureWithFailedLockDoesNotWaitItsAutoResetEventTwice) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.lock_failures = 1;
  EXPECT_TRUE(probe.encode_frame(7, true).data.empty());
  const auto failed = probe.snapshot();
  EXPECT_EQ(failed.frame_wait_calls, 1);
  EXPECT_EQ(failed.lock_calls, 1);
  EXPECT_TRUE(failed.input_mapped);
  EXPECT_TRUE(failed.picture_pending);
  EXPECT_EQ(failed.unmap_calls, 0);

  probe.destroy_encoder();
  const auto cleaned = probe.snapshot();
  EXPECT_EQ(cleaned.frame_wait_calls, 1);
  EXPECT_EQ(cleaned.lock_calls, 2);
  EXPECT_EQ(cleaned.unlock_calls, 1);
  EXPECT_EQ(cleaned.unmap_calls, 1);
  EXPECT_EQ(cleaned.eos_calls, 1);
  EXPECT_EQ(cleaned.flush_wait_calls, 1);
  EXPECT_EQ(cleaned.sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, RejectedSubmissionUnmapsWithoutWaitingOrFlushingUnacceptedWork) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.submission_status = NV_ENC_ERR_INVALID_PARAM;
  EXPECT_TRUE(probe.encode_frame(1, true).data.empty());
  const auto rejected = probe.snapshot();
  EXPECT_FALSE(rejected.input_mapped);
  EXPECT_FALSE(rejected.picture_pending);
  EXPECT_EQ(rejected.frame_wait_calls, 0);
  EXPECT_EQ(rejected.lock_calls, 0);
  EXPECT_EQ(rejected.unmap_calls, 1);

  probe.destroy_encoder();
  EXPECT_EQ(probe.snapshot().eos_calls, 0);
  EXPECT_EQ(probe.snapshot().sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, NeedMoreInputRetainsAcceptedInputAndEosReleasesItsOutput) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.submission_status = NV_ENC_ERR_NEED_MORE_INPUT;
  EXPECT_TRUE(probe.encode_frame(3, true).data.empty());
  const auto buffered = probe.snapshot();
  EXPECT_TRUE(buffered.input_mapped);
  EXPECT_TRUE(buffered.picture_pending);
  EXPECT_EQ(buffered.unmap_calls, 0);
  EXPECT_EQ(buffered.eos_calls, 0);

  probe.destroy_encoder();
  const auto cleaned = probe.snapshot();
  EXPECT_EQ(cleaned.eos_calls, 1);
  EXPECT_EQ(cleaned.lock_calls, 1);
  EXPECT_EQ(cleaned.unlock_calls, 1);
  EXPECT_EQ(cleaned.unmap_calls, 1);
  EXPECT_EQ(cleaned.flush_wait_calls, 1);
  EXPECT_FALSE(cleaned.input_mapped);
  EXPECT_EQ(cleaned.sessions_destroyed, 1);
  const auto eos = std::find(cleaned.operations.begin(), cleaned.operations.end(), "eos");
  const auto output = std::find(cleaned.operations.begin(), cleaned.operations.end(), "frame-ready");
  ASSERT_NE(eos, cleaned.operations.end());
  ASSERT_NE(output, cleaned.operations.end());
  EXPECT_LT(eos, output);
}

TEST(NvencLifecycleTest, FailedUnlockKeepsBitstreamAndMappedInputUntilCleanupRetries) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.unlock_failures = 1;
  probe.encode_frame(1, true);
  const auto failed = probe.snapshot();
  EXPECT_TRUE(failed.bitstream_locked);
  EXPECT_TRUE(failed.input_mapped);
  EXPECT_TRUE(failed.picture_pending);
  EXPECT_EQ(failed.unlock_calls, 1);
  EXPECT_EQ(failed.unmap_calls, 0);
  EXPECT_TRUE(probe.encode_frame(2, false).data.empty());
  EXPECT_EQ(probe.snapshot().map_calls, 1);

  probe.destroy_encoder();
  const auto cleaned = probe.snapshot();
  EXPECT_EQ(cleaned.frame_wait_calls, 1);
  EXPECT_EQ(cleaned.lock_calls, 1);
  EXPECT_EQ(cleaned.unlock_calls, 2);
  EXPECT_EQ(cleaned.unmap_calls, 1);
  EXPECT_EQ(cleaned.sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, FailedUnmapKeepsInputUntilCleanupRetriesWithoutRelocking) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  probe.unmap_failures = 1;
  probe.encode_frame(1, true);
  const auto failed = probe.snapshot();
  EXPECT_FALSE(failed.bitstream_locked);
  EXPECT_FALSE(failed.picture_pending);
  EXPECT_TRUE(failed.input_mapped);
  EXPECT_EQ(failed.unmap_calls, 1);
  EXPECT_TRUE(probe.encode_frame(2, false).data.empty());
  EXPECT_EQ(probe.snapshot().map_calls, 1);

  probe.destroy_encoder();
  const auto cleaned = probe.snapshot();
  EXPECT_EQ(cleaned.frame_wait_calls, 1);
  EXPECT_EQ(cleaned.lock_calls, 1);
  EXPECT_EQ(cleaned.unlock_calls, 1);
  EXPECT_EQ(cleaned.unmap_calls, 2);
  EXPECT_EQ(cleaned.sessions_destroyed, 1);
}

TEST(NvencLifecycleTest, EosCompletionRetainsEventsBuffersAndSessionAfterFrameIsDrained) {
  nvenc_lifecycle_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_FALSE(probe.encode_frame(1, true).data.empty());
  probe.defer_flush_completion();
  {
    scoped_encoder_teardown teardown(probe);
    ASSERT_TRUE(probe.await_boundary(nvenc_lifecycle_probe::boundary_t::flush_wait));
    const auto waiting = probe.snapshot();
    EXPECT_FALSE(waiting.input_mapped);
    EXPECT_FALSE(waiting.picture_pending);
    EXPECT_TRUE(waiting.frame_event_registered);
    EXPECT_TRUE(waiting.flush_event_registered);
    EXPECT_TRUE(waiting.bitstream_owned);
    EXPECT_TRUE(waiting.input_owned);
    EXPECT_EQ(waiting.eos_calls, 1);
    EXPECT_EQ(waiting.sessions_destroyed, 0);
  }
  EXPECT_EQ(probe.snapshot().sessions_destroyed, 1);
}

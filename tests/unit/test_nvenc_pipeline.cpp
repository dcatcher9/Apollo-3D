#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <src/nvenc/nvenc_base.h>
#include <src/video_encode_pipeline.h>
#include <string>
#include <thread>
#include <vector>

namespace {
  using namespace std::chrono_literals;

  // A driver-free NVENC session with a picture slot per input surface, bitstream and completion
  // event. It checks NVENC's asynchronous-mode contract: each picture in flight uses its own
  // input, bitstream and event; bitstreams are locked in submission order, after their event; an
  // input is neither copied into nor unmapped while its picture is in flight.
  class nvenc_pipeline_probe final: public nvenc::nvenc_base {
  public:
    explicit nvenc_pipeline_probe(bool extra_events = true):
        nvenc_base(NV_ENC_DEVICE_TYPE_DIRECTX),
        extra_events(extra_events) {
      device = this;
      async_event_handle = &events[0];
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
      nvenc->nvEncInvalidateRefFrames = invalidate;
      nvenc->nvEncReconfigureEncoder = reconfigure;
    }

    ~nvenc_pipeline_probe() override {
      complete_every_picture();
      destroy_encoder();
    }

    bool create(int sbs_mode = video::SBS_GAME_SBS) {
      video::config_t stream {};
      stream.width = 3840;
      stream.height = 1080;
      stream.framerate = 90;
      stream.bitrate = 50'000;
      stream.slicesPerFrame = 1;
      stream.videoFormat = 1;
      stream.sbs_mode = sbs_mode;
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

    /** The picture in `slot` completes: its event is signaled after `waits_first` more timed-out
     *  waits on it (each advances the fake clock by its timeout). */
    void complete(unsigned slot, int waits_first = 0) {
      std::lock_guard lock(mutex);
      signaled[slot] = true;
      timeouts_first[slot] = waits_first;
    }

    /** Every picture completes as soon as it is waited for. */
    void complete_every_picture() {
      std::lock_guard lock(mutex);
      auto_complete = true;
      timeouts_first = {};
    }

    std::vector<std::string> operations() {
      std::lock_guard lock(mutex);
      return log;
    }

    void clear_operations() {
      std::lock_guard lock(mutex);
      log.clear();
    }

    std::array<nvenc::input_producer_state, 2> producer {nvenc::input_producer_state::unknown, nvenc::input_producer_state::unknown};

  protected:
    bool init_library() override {
      ADD_FAILURE() << "The fixture must not load a real encoder driver";
      return false;
    }

    bool create_and_register_input_buffer() override {
      std::lock_guard lock(mutex);
      for (unsigned slot = 0; slot < pipeline_depth(); ++slot) {
        registered_input_buffers[slot] = &inputs[slot];
        log.push_back("register-input-" + std::to_string(slot));
      }
      return true;
    }

    void prepare_input(unsigned slot) override {
      std::lock_guard lock(mutex);
      EXPECT_FALSE(state[slot].mapped) << "A mapped input must not be written";
      EXPECT_FALSE(state[slot].pending);
      log.push_back("copy-" + std::to_string(slot));
    }

    void *create_completion_event(unsigned slot) override {
      return extra_events ? &events[slot] : nullptr;
    }

    void release_async_event() override {
      async_event_handle = nullptr;
    }

    void *create_flush_event() override {
      return &flush_token;
    }

    nvenc::nvenc_event_wait_result wait_for_flush_event(std::uint32_t) override {
      std::lock_guard lock(mutex);
      log.emplace_back("flush-ready");
      return {nvenc::nvenc_event_wait_status::ready};
    }

    std::chrono::steady_clock::time_point async_wait_clock_now() const override {
      std::lock_guard lock(mutex);
      return fake_now;
    }

    void mark_input_producer_end(unsigned slot) override {
      std::lock_guard lock(mutex);
      EXPECT_TRUE(state[slot].mapped);
      log.push_back("mark-" + std::to_string(slot));
    }

    nvenc::input_producer_state poll_input_producer(unsigned slot) override {
      std::lock_guard lock(mutex);
      log.push_back("poll-" + std::to_string(slot));
      return producer[slot];
    }

    nvenc::nvenc_event_wait_result wait_for_async_event(void *event, std::uint32_t timeout_ms) override {
      std::lock_guard lock(mutex);
      const auto slot = static_cast<unsigned>(static_cast<int *>(event) - events.data());
      EXPECT_LT(slot, 2u);
      EXPECT_TRUE(state[slot].pending) << "Only a submitted picture's event is waited for";
      if ((auto_complete || signaled[slot]) && timeouts_first[slot] == 0) {
        signaled[slot] = false;
        state[slot].completed = true;
        log.push_back("ready-" + std::to_string(slot));
        return {nvenc::nvenc_event_wait_status::ready};
      }
      if (timeouts_first[slot] > 0) {
        --timeouts_first[slot];
      }
      fake_now += std::chrono::milliseconds(timeout_ms);
      log.push_back("timeout-" + std::to_string(slot));
      return {nvenc::nvenc_event_wait_status::timeout, 0x102};
    }

  private:
    struct slot_state_t {
      bool mapped = false, pending = false, completed = false, locked = false;
      std::uint64_t index = 0;
    };

    bool extra_events;
    mutable std::mutex mutex;
    std::chrono::steady_clock::time_point fake_now {};
    std::array<int, 2> events {}, inputs {}, mapped {}, bitstreams {};
    std::array<slot_state_t, 2> state {};
    std::array<bool, 2> signaled {};
    std::array<int, 2> timeouts_first {};
    std::deque<unsigned> submitted;
    unsigned bitstreams_created = 0;
    bool auto_complete = false;
    int flush_token = 0;
    std::vector<std::string> log;

    static nvenc_pipeline_probe &self(void *encoder) {
      return *static_cast<nvenc_pipeline_probe *>(encoder);
    }

    template<class Array>
    static unsigned index_of(const Array &array, const void *pointer) {
      const auto index = static_cast<unsigned>(static_cast<const int *>(pointer) - array.data());
      EXPECT_LT(index, array.size());
      return index;
    }

    static NVENCSTATUS NVENCAPI open_session(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *params, void **encoder) {
      *encoder = params->device;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI codec_count(void *, std::uint32_t *count) {
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
        case NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION:
        case NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE:
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

    static NVENCSTATUS NVENCAPI initialize(void *, NV_ENC_INITIALIZE_PARAMS *) {
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI register_event(void *encoder, NV_ENC_EVENT_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      probe.log.push_back(params->completionEvent == &probe.flush_token ? "register-flush" : "register-event-" + std::to_string(index_of(probe.events, params->completionEvent)));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unregister_event(void *encoder, NV_ENC_EVENT_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_TRUE(probe.submitted.empty());
      probe.log.push_back(params->completionEvent == &probe.flush_token ? "unregister-flush" : "unregister-event-" + std::to_string(index_of(probe.events, params->completionEvent)));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI create_bitstream(void *encoder, NV_ENC_CREATE_BITSTREAM_BUFFER *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_LT(probe.bitstreams_created, 2u);
      params->bitstreamBuffer = &probe.bitstreams[probe.bitstreams_created++ % 2];
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI destroy_bitstream(void *encoder, NV_ENC_OUTPUT_PTR bitstream) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_TRUE(probe.submitted.empty());
      probe.log.push_back("destroy-bitstream-" + std::to_string(index_of(probe.bitstreams, bitstream)));
      --probe.bitstreams_created;
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unregister_input(void *encoder, NV_ENC_REGISTERED_PTR input) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      const auto slot = index_of(probe.inputs, input);
      EXPECT_FALSE(probe.state[slot].mapped);
      probe.log.push_back("unregister-input-" + std::to_string(slot));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI destroy_session(void *encoder) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      probe.log.emplace_back("destroy-session");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI map_input(void *encoder, NV_ENC_MAP_INPUT_RESOURCE *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      const auto slot = index_of(probe.inputs, params->registeredResource);
      EXPECT_FALSE(probe.state[slot].mapped);
      probe.state[slot].mapped = true;
      params->mappedResource = &probe.mapped[slot];
      params->mappedBufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      probe.log.push_back("map-" + std::to_string(slot));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI submit_picture(void *encoder, NV_ENC_PIC_PARAMS *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      if (params->encodePicFlags & NV_ENC_PIC_FLAG_EOS) {
        EXPECT_EQ(params->completionEvent, &probe.flush_token);
        probe.log.emplace_back("eos");
        return NV_ENC_SUCCESS;
      }
      const auto slot = index_of(probe.mapped, params->inputBuffer);
      EXPECT_TRUE(probe.state[slot].mapped);
      EXPECT_FALSE(probe.state[slot].pending);
      EXPECT_EQ(params->outputBitstream, &probe.bitstreams[slot]);
      EXPECT_EQ(params->completionEvent, &probe.events[slot]);
      probe.state[slot].pending = true;
      probe.state[slot].completed = false;
      probe.state[slot].index = params->inputTimeStamp;
      probe.submitted.push_back(slot);
      probe.log.push_back("submit-" + std::to_string(slot) + ":" + std::to_string(params->inputTimeStamp) + (params->encodePicFlags & NV_ENC_PIC_FLAG_FORCEIDR ? ":idr" : ""));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI lock_bitstream(void *encoder, NV_ENC_LOCK_BITSTREAM *params) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      const auto slot = index_of(probe.bitstreams, params->outputBitstream);
      EXPECT_TRUE(probe.state[slot].pending);
      EXPECT_TRUE(probe.state[slot].completed) << "Locked before its event";
      EXPECT_FALSE(probe.state[slot].locked);
      EXPECT_FALSE(probe.submitted.empty());
      if (!probe.submitted.empty()) {
        EXPECT_EQ(probe.submitted.front(), slot) << "Bitstreams are locked in submission order";
      }
      probe.state[slot].locked = true;
      static std::uint8_t bytes[4] {0, 0, 1, 0x26};
      params->bitstreamBufferPtr = bytes;
      params->bitstreamSizeInBytes = sizeof(bytes);
      params->outputTimeStamp = probe.state[slot].index;
      params->pictureType = NV_ENC_PIC_TYPE_P;
      probe.log.push_back("lock-" + std::to_string(slot));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unlock_bitstream(void *encoder, NV_ENC_OUTPUT_PTR bitstream) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      const auto slot = index_of(probe.bitstreams, bitstream);
      EXPECT_TRUE(probe.state[slot].locked);
      probe.state[slot].locked = false;
      probe.state[slot].pending = false;
      if (!probe.submitted.empty()) {
        probe.submitted.pop_front();
      }
      probe.log.push_back("unlock-" + std::to_string(slot));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI unmap_input(void *encoder, NV_ENC_INPUT_PTR input) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      const auto slot = index_of(probe.mapped, input);
      EXPECT_TRUE(probe.state[slot].mapped);
      EXPECT_FALSE(probe.state[slot].pending) << "An input is unmapped only after its picture";
      EXPECT_FALSE(probe.state[slot].locked);
      probe.state[slot].mapped = false;
      probe.log.push_back("unmap-" + std::to_string(slot));
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI reconfigure(void *encoder, NV_ENC_RECONFIGURE_PARAMS *) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_TRUE(probe.submitted.empty()) << "Reconfiguration runs with no picture in flight";
      probe.log.emplace_back("reconfigure");
      return NV_ENC_SUCCESS;
    }

    static NVENCSTATUS NVENCAPI invalidate(void *encoder, std::uint64_t frame) {
      auto &probe = self(encoder);
      std::lock_guard lock(probe.mutex);
      EXPECT_TRUE(probe.submitted.empty()) << "Invalidation runs with no picture in flight";
      probe.log.push_back("invalidate-" + std::to_string(frame));
      return NV_ENC_SUCCESS;
    }
  };

  std::vector<std::string> only(const std::vector<std::string> &operations, const std::vector<std::string> &prefixes) {
    std::vector<std::string> filtered;
    for (const auto &operation : operations) {
      if (std::any_of(prefixes.begin(), prefixes.end(), [&](const std::string &prefix) {
            return operation.starts_with(prefix);
          })) {
        filtered.push_back(operation);
      }
    }
    return filtered;
  }
}  // namespace

TEST(NvencPipelineTest, IndependentProviderGetsTwoPicturesInFlightAndOthersOne) {
  nvenc_pipeline_probe game;
  ASSERT_TRUE(game.create(video::SBS_GAME_SBS));
  EXPECT_EQ(game.pipeline_depth(), 2u);
  EXPECT_EQ(only(game.operations(), {"register-"}), (std::vector<std::string> {"register-event-0", "register-event-1", "register-input-0", "register-input-1"}));

  nvenc_pipeline_probe desktop;
  ASSERT_TRUE(desktop.create(video::SBS_OFF));
  EXPECT_EQ(desktop.pipeline_depth(), 1u);
  EXPECT_EQ(only(desktop.operations(), {"register-"}), (std::vector<std::string> {"register-event-0", "register-input-0"}));
  desktop.complete_every_picture();
  ASSERT_FALSE(desktop.encode_frame(1, true).data.empty());
  // One picture at a time encodes the converted surface itself: no copy, unmapped at once.
  EXPECT_EQ(only(desktop.operations(), {"copy", "map", "submit", "lock", "unlock", "unmap"}), (std::vector<std::string> {"map-0", "submit-0:1:idr", "lock-0", "unlock-0", "unmap-0"}));

  nvenc_pipeline_probe host_sbs;
  ASSERT_TRUE(host_sbs.create(video::SBS_AI));
  EXPECT_EQ(host_sbs.pipeline_depth(), 1u);

  nvenc_pipeline_probe without_events(false);
  ASSERT_TRUE(without_events.create(video::SBS_GAME_SBS));
  EXPECT_EQ(without_events.pipeline_depth(), 1u);  // Fails safe to one picture at a time.
}

TEST(NvencPipelineTest, TwoPicturesInFlightUseTheirOwnInputBitstreamAndEvent) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.clear_operations();

  ASSERT_TRUE(probe.submit_frame(1, true));
  ASSERT_TRUE(probe.submit_frame(2, false));
  EXPECT_EQ(probe.frames_in_flight(), 2u);
  // A third picture has no free input until the oldest is retrieved; nothing is copied or mapped.
  EXPECT_FALSE(probe.submit_frame(3, false));
  EXPECT_EQ(probe.operations(), (std::vector<std::string> {"copy-0", "map-0", "mark-0", "submit-0:1:idr", "copy-1", "map-1", "mark-1", "submit-1:2"}));

  probe.complete(0);
  probe.complete(1);
  const auto first = probe.retrieve_frame();
  ASSERT_FALSE(first.data.empty());
  EXPECT_EQ(first.frame_index, 1u);
  EXPECT_EQ(probe.frames_in_flight(), 1u);
  probe.clear_operations();

  // Slot 0 is free again: the submitting thread unmaps its input before copying the next frame in.
  ASSERT_TRUE(probe.submit_frame(3, false));
  EXPECT_EQ(probe.operations(), (std::vector<std::string> {"unmap-0", "copy-0", "map-0", "mark-0", "submit-0:3"}));
  probe.complete(0);
  EXPECT_EQ(probe.retrieve_frame().frame_index, 2u);
  EXPECT_EQ(probe.retrieve_frame().frame_index, 3u);
  EXPECT_EQ(probe.frames_in_flight(), 0u);
  EXPECT_TRUE(probe.retrieve_frame().data.empty());  // Nothing in flight.
}

TEST(NvencPipelineTest, RetrievalWaitsForTheOlderPictureEvenWhenTheNewerCompletedFirst) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_TRUE(probe.submit_frame(1, false));
  ASSERT_TRUE(probe.submit_frame(2, false));
  probe.complete(1);  // The newer picture completed first.
  probe.complete(0, 3);  // The older one after three more 100/25/25 ms waits.
  probe.clear_operations();

  const auto first = probe.retrieve_frame();
  ASSERT_FALSE(first.data.empty());
  EXPECT_EQ(first.frame_index, 1u);
  const auto second = probe.retrieve_frame();
  EXPECT_EQ(second.frame_index, 2u);
  EXPECT_EQ(only(probe.operations(), {"timeout", "ready", "lock"}), (std::vector<std::string> {"timeout-0", "timeout-0", "timeout-0", "ready-0", "lock-0", "ready-1", "lock-1"}));
}

TEST(NvencPipelineTest, PicturesRetrievedOnASecondThreadWhileTheNextIsSubmitted) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.complete_every_picture();
  std::vector<std::uint64_t> retrieved;
  {
    video::detail::encode_pipeline_t<std::uint64_t> pipeline(probe.pipeline_depth(), [&](std::uint64_t &frame) {
      const auto encoded = probe.retrieve_frame();
      if (encoded.data.empty() || encoded.frame_index != frame) {
        return false;
      }
      retrieved.push_back(encoded.frame_index);
      return true;
    });
    for (std::uint64_t frame = 1; frame <= 500; ++frame) {
      ASSERT_TRUE(pipeline.acquire());
      ASSERT_TRUE(probe.submit_frame(frame, frame == 1));
      pipeline.push(frame);
    }
    EXPECT_TRUE(pipeline.drain());
  }
  ASSERT_EQ(retrieved.size(), 500u);
  for (std::uint64_t i = 0; i < retrieved.size(); ++i) {
    EXPECT_EQ(retrieved[i], i + 1);
  }
}

TEST(NvencPipelineTest, TeardownWithTwoPicturesInFlightDrainsThemInSubmissionOrder) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_TRUE(probe.submit_frame(1, true));
  ASSERT_TRUE(probe.submit_frame(2, false));
  probe.clear_operations();
  probe.complete_every_picture();

  probe.destroy_encoder();
  EXPECT_EQ(probe.operations(), (std::vector<std::string> {
                                  "register-flush",
                                  "eos",
                                  "ready-0",
                                  "lock-0",
                                  "unlock-0",
                                  "unmap-0",
                                  "ready-1",
                                  "lock-1",
                                  "unlock-1",
                                  "unmap-1",
                                  "flush-ready",
                                  "destroy-bitstream-0",
                                  "destroy-bitstream-1",
                                  "unregister-flush",
                                  "unregister-event-0",
                                  "unregister-event-1",
                                  "unregister-input-0",
                                  "unregister-input-1",
                                  "destroy-session",
                                }));
  // The probe is reusable afterwards, as after any teardown.
  probe.clear_operations();
  ASSERT_TRUE(probe.create());
  EXPECT_EQ(probe.pipeline_depth(), 2u);
}

TEST(NvencPipelineTest, RetrievedButUnmappedInputIsReleasedByTeardown) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.complete_every_picture();
  ASSERT_TRUE(probe.submit_frame(1, true));
  ASSERT_FALSE(probe.retrieve_frame().data.empty());
  probe.clear_operations();
  probe.destroy_encoder();
  const auto ops = probe.operations();
  EXPECT_EQ(only(ops, {"unmap", "lock", "ready"}), (std::vector<std::string> {"unmap-0"}));
}

TEST(NvencPipelineTest, OldestPictureTimeoutBlocksSubmissionAndAdmissionUntilTeardown) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_TRUE(probe.submit_frame(1, true));
  ASSERT_TRUE(probe.submit_frame(2, false));
  probe.complete(1);
  // Picture 1 never completes within the 250 ms budget (100 ms + 6 x 25 ms of fake time).
  EXPECT_TRUE(probe.retrieve_frame().data.empty());
  EXPECT_EQ(only(probe.operations(), {"timeout-", "ready-"}), (std::vector<std::string> {"timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0"}));
  EXPECT_EQ(probe.frames_in_flight(), 2u);
  EXPECT_FALSE(probe.submit_frame(3, false));
  EXPECT_TRUE(probe.retrieve_frame().data.empty());
  nvenc_pipeline_probe replacement;
  EXPECT_FALSE(replacement.create());

  probe.complete_every_picture();
  probe.destroy_encoder();
  EXPECT_EQ(only(probe.operations(), {"lock", "unmap"}), (std::vector<std::string> {"lock-0", "unmap-0", "lock-1", "unmap-1"}));
  EXPECT_TRUE(replacement.create());
}

TEST(NvencPipelineTest, LateInputExtendsOnlyItsOwnPicturesBudget) {
  // 61f29a89 with pictures in flight: the encoder's 250 ms start once the picture's own input
  // completed. The second picture's wait starts after the first completed.
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  ASSERT_TRUE(probe.submit_frame(1, false));
  ASSERT_TRUE(probe.submit_frame(2, false));
  probe.producer = {nvenc::input_producer_state::pending, nvenc::input_producer_state::complete};
  probe.complete(0, 12);  // 100 ms + 11 x 25 ms = 375 ms, its input still being produced.
  probe.complete(1);
  probe.clear_operations();
  const auto first = probe.retrieve_frame();
  ASSERT_FALSE(first.data.empty());
  EXPECT_EQ(first.frame_index, 1u);
  const auto ops = probe.operations();
  EXPECT_EQ(std::count(ops.begin(), ops.end(), "timeout-0"), 12);
  EXPECT_GE(std::count(ops.begin(), ops.end(), "poll-0"), 1);
  EXPECT_EQ(std::count(ops.begin(), ops.end(), "poll-1"), 0);
  EXPECT_EQ(probe.retrieve_frame().frame_index, 2u);

  // The same late picture with a completed input fails at the encoder's budget.
  nvenc_pipeline_probe silent;
  ASSERT_TRUE(silent.create());
  ASSERT_TRUE(silent.submit_frame(1, false));
  silent.producer = {nvenc::input_producer_state::complete, nvenc::input_producer_state::complete};
  silent.complete(0, 12);
  EXPECT_TRUE(silent.retrieve_frame().data.empty());
  const auto silent_ops = silent.operations();
  EXPECT_EQ(std::count(silent_ops.begin(), silent_ops.end(), "timeout-0"), 7);
}

TEST(NvencPipelineTest, InvalidationWaitsForNoPictureInFlightAndMarksTheNextSubmitted) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.complete_every_picture();
  ASSERT_TRUE(probe.submit_frame(1, true));
  ASSERT_FALSE(probe.retrieve_frame().data.empty());
  ASSERT_TRUE(probe.submit_frame(2, false));
  // The invalidation itself must not run with a picture in flight: the caller waits for it first.
  EXPECT_TRUE(probe.would_invalidate_ref_frames(2, 2));
  EXPECT_FALSE(probe.reconfigure_bitrate(40'000));
  EXPECT_TRUE(only(probe.operations(), {"reconfigure"}).empty());
  const auto second = probe.retrieve_frame();
  EXPECT_FALSE(second.after_ref_frame_invalidation);
  EXPECT_TRUE(probe.reconfigure_bitrate(40'000));
  EXPECT_EQ(only(probe.operations(), {"reconfigure"}), (std::vector<std::string> {"reconfigure"}));

  // Drained: the range extends to the last submitted picture and the next one confirms it.
  EXPECT_TRUE(probe.would_invalidate_ref_frames(2, 2));
  EXPECT_TRUE(probe.invalidate_ref_frames(2, 2));
  EXPECT_EQ(only(probe.operations(), {"invalidate"}), (std::vector<std::string> {"invalidate-2"}));
  EXPECT_FALSE(probe.would_invalidate_ref_frames(2, 2));  // Done.
  ASSERT_TRUE(probe.submit_frame(3, false));
  ASSERT_TRUE(probe.submit_frame(4, false));
  const auto confirmed = probe.retrieve_frame();
  EXPECT_EQ(confirmed.frame_index, 3u);
  EXPECT_TRUE(confirmed.after_ref_frame_invalidation);
  EXPECT_FALSE(probe.retrieve_frame().after_ref_frame_invalidation);
}

TEST(NvencPipelineTest, InvalidationThatNeedsNoDriverCallIsDecidedWithPicturesInFlight) {
  // A range already handled, malformed or too large for the DPB (5 pictures for HEVC) calls no
  // NvEncInvalidateRefFrames(): it is decided while both pictures still encode, so the recovery
  // frame converts at once instead of after both completed.
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.complete_every_picture();
  for (std::uint64_t frame = 1; frame <= 4; ++frame) {
    ASSERT_TRUE(probe.submit_frame(frame, frame == 1));
    ASSERT_FALSE(probe.retrieve_frame().data.empty());
  }
  ASSERT_TRUE(probe.submit_frame(5, false));
  ASSERT_TRUE(probe.submit_frame(6, false));
  ASSERT_EQ(probe.frames_in_flight(), 2u);

  // One that would invalidate is refused while a picture is in flight (an IDR), never run: the
  // caller waits for the pictures first.
  EXPECT_TRUE(probe.would_invalidate_ref_frames(5, 5));
  EXPECT_FALSE(probe.invalidate_ref_frames(5, 5));
  EXPECT_TRUE(only(probe.operations(), {"invalidate"}).empty());

  // 2-6 (extended to the last submitted picture) is five pictures: an IDR, decided now.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(2, 3));
  EXPECT_FALSE(probe.invalidate_ref_frames(2, 3));
  // Handled by that IDR: done.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(3, 4));
  EXPECT_TRUE(probe.invalidate_ref_frames(3, 4));
  // Malformed: an IDR.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(9, 8));
  EXPECT_FALSE(probe.invalidate_ref_frames(9, 8));
  EXPECT_TRUE(only(probe.operations(), {"invalidate"}).empty());
  EXPECT_EQ(probe.frames_in_flight(), 2u);

  // In flight pictures keep the confirmation they were submitted with; the next one carries it.
  EXPECT_FALSE(probe.retrieve_frame().after_ref_frame_invalidation);
  EXPECT_FALSE(probe.retrieve_frame().after_ref_frame_invalidation);
  ASSERT_TRUE(probe.submit_frame(7, true));
  EXPECT_TRUE(probe.retrieve_frame().after_ref_frame_invalidation);
}

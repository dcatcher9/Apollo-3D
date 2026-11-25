#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <src/nvenc/nvenc_base.h>
#include <src/platform/common.h>
#include <src/tracked_async_worker.h>
#include <src/video_encode_session.h>
#include <stdexcept>
#include <src/video_encode_pipeline.h>
#include <string>
#include <thread>
#include <utility>
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

    /** A conversion draws `content` into the conversion target (the encoder's input surface). */
    void convert(int content) {
      std::lock_guard lock(mutex);
      target = content;
    }

    /** Each locked picture's frame index and the content NVENC encoded for it: its own input's
     *  copy of the target with pictures in flight, else the target itself. */
    std::vector<std::pair<std::uint64_t, int>> encoded_contents() {
      std::lock_guard lock(mutex);
      return encoded;
    }

    unsigned preparation_call_count() {
      std::lock_guard lock(mutex);
      return preparation_calls;
    }

    // Pool positions are not picture associations: this registration can move while mapped.
    void move_first_registration_to_spare() {
      std::lock_guard lock(mutex);
      registered_input_buffers[max_pipeline_depth] = std::exchange(registered_input_buffers[0], nullptr);
    }

    bool registration_is_mapped(unsigned input) noexcept {
      return input_registration_is_mapped(&inputs[input]);
    }

    bool registration_in_flight(unsigned input) noexcept {
      return input_registration_in_flight(&inputs[input]);
    }

    bool release_registration(unsigned input) {
      return release_input_registration(&inputs[input]);
    }

    bool slot_maps_input(unsigned slot, unsigned input) const noexcept {
      return mapped_input_registration_for_test(slot) == &inputs[input];
    }

    void clear_operations() {
      std::lock_guard lock(mutex);
      log.clear();
    }

    std::array<nvenc::input_producer_state, 2> producer {nvenc::input_producer_state::unknown, nvenc::input_producer_state::unknown};
    /** Video memory runs short for a second picture's input (E_OUTOFMEMORY in the native class). */
    bool second_input_out_of_memory = false;
    bool refuse_prepared_input = false;
    NVENCSTATUS next_picture_status = NV_ENC_SUCCESS;  ///< Inject one accepted status.

  protected:
    bool init_library() override {
      ADD_FAILURE() << "The fixture must not load a real encoder driver";
      return false;
    }

    bool create_and_register_input_buffer() override {
      std::lock_guard lock(mutex);
      for (unsigned slot = 0; slot < pipeline_depth(); ++slot) {
        if (slot > 0 && second_input_out_of_memory) {
          log.push_back("input-" + std::to_string(slot) + "-out-of-memory");
          return false;
        }
        registered_input_buffers[slot] = &inputs[slot];
        log.push_back("register-input-" + std::to_string(slot));
      }
      return true;
    }

    NV_ENC_REGISTERED_PTR prepare_input(unsigned slot) override {
      std::lock_guard lock(mutex);
      ++preparation_calls;
      if (refuse_prepared_input) {
        return nullptr;
      }
      if (pipeline_depth() == 1) {
        return registered_input_buffers[slot];
      }
      EXPECT_FALSE(input_registration_is_mapped(&inputs[slot]));
      EXPECT_FALSE(state[slot].mapped) << "A mapped input must not be written";
      EXPECT_FALSE(state[slot].pending);
      input_content[slot] = target;
      log.push_back("copy-" + std::to_string(slot));
      return &inputs[slot];
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
    int target = 0;  ///< The conversion target's content.
    std::array<int, 2> input_content {};  ///< Each picture input's copy of the target (depth 2).
    std::vector<std::pair<std::uint64_t, int>> encoded;
    std::array<slot_state_t, 2> state {};
    std::array<bool, 2> signaled {};
    std::array<int, 2> timeouts_first {};
    std::deque<unsigned> submitted;
    unsigned bitstreams_created = 0;
    unsigned preparation_calls = 0;
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
      return std::exchange(probe.next_picture_status, NV_ENC_SUCCESS);
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
      probe.encoded.emplace_back(probe.state[slot].index, probe.pipeline_depth() > 1 ? probe.input_content[slot] : probe.target);
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

TEST(NvencPipelineTest, IndependentProviderAndHostSbsGetTwoPicturesInFlightAndPlainDesktopOne) {
  nvenc_pipeline_probe game;
  ASSERT_TRUE(game.create(video::SBS_GAME_SBS));
  EXPECT_EQ(game.pipeline_depth(), 2u);
  EXPECT_EQ(only(game.operations(), {"register-"}), (std::vector<std::string> {"register-event-0", "register-event-1", "register-input-0", "register-input-1"}));

  nvenc_pipeline_probe game_mono;
  ASSERT_TRUE(game_mono.create(video::SBS_GAME_MONO));
  EXPECT_EQ(game_mono.pipeline_depth(), 2u);

  nvenc_pipeline_probe desktop;
  ASSERT_TRUE(desktop.create(video::SBS_OFF));
  EXPECT_EQ(desktop.pipeline_depth(), 1u);
  EXPECT_EQ(only(desktop.operations(), {"register-"}), (std::vector<std::string> {"register-event-0", "register-input-0"}));
  desktop.complete_every_picture();
  ASSERT_FALSE(desktop.encode_frame(1, true).data.empty());
  EXPECT_EQ(desktop.preparation_call_count(), 1u);
  // One picture at a time encodes the converted surface itself: no copy, unmapped at once.
  EXPECT_EQ(only(desktop.operations(), {"copy", "map", "submit", "lock", "unlock", "unmap"}), (std::vector<std::string> {"map-0", "submit-0:1:idr", "lock-0", "unlock-0", "unmap-0"}));

  // Desktop Host SBS converts at capture cadence, but its AI conversion and 2W x H picture overlap
  // as an independent provider's do.
  nvenc_pipeline_probe host_sbs;
  ASSERT_TRUE(host_sbs.create(video::SBS_AI));
  EXPECT_EQ(host_sbs.pipeline_depth(), 2u);
  EXPECT_EQ(only(host_sbs.operations(), {"register-"}), (std::vector<std::string> {"register-event-0", "register-event-1", "register-input-0", "register-input-1"}));

  nvenc_pipeline_probe without_events(false);
  ASSERT_TRUE(without_events.create(video::SBS_GAME_SBS));
  EXPECT_EQ(without_events.pipeline_depth(), 1u);  // Fails safe to one picture at a time.
  nvenc_pipeline_probe host_sbs_without_events(false);
  ASSERT_TRUE(host_sbs_without_events.create(video::SBS_AI));
  EXPECT_EQ(host_sbs_without_events.pipeline_depth(), 1u);
}

TEST(NvencPipelineTest, HostSbsEncodesOnePictureAtATimeWhenASecondInputDoesNotFit) {
  // Host SBS beside a GPU-bound game near its video-memory budget: the second picture's input
  // cannot be created. The stream keeps its 3D encoder, one picture at a time from the conversion
  // target, as without a completion event for every picture, instead of failing (which would revert
  // a live toggle into Host SBS or end its launch).
  nvenc_pipeline_probe probe;
  probe.second_input_out_of_memory = true;
  ASSERT_TRUE(probe.create(video::SBS_AI));
  EXPECT_EQ(probe.pipeline_depth(), 1u);
  EXPECT_EQ(only(probe.operations(), {"register-input", "unregister-input", "input-"}), (std::vector<std::string> {"register-input-0", "input-1-out-of-memory", "unregister-input-0", "register-input-0"}));

  probe.clear_operations();
  probe.convert(7);
  probe.complete_every_picture();
  ASSERT_FALSE(probe.encode_frame(1, true).data.empty());
  // The conversion target itself, as for plain desktop: no copy, unmapped at once.
  EXPECT_EQ(only(probe.operations(), {"copy", "map", "submit", "lock", "unlock", "unmap"}), (std::vector<std::string> {"map-0", "submit-0:1:idr", "lock-0", "unlock-0", "unmap-0"}));
  EXPECT_EQ(probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{1, 7}}));

  // The second slot's event and bitstream, unused, are released with the encoder.
  probe.clear_operations();
  probe.destroy_encoder();
  const auto released = only(probe.operations(), {"destroy-bitstream", "unregister-event", "unregister-input"});
  EXPECT_EQ(released, (std::vector<std::string> {"destroy-bitstream-0", "destroy-bitstream-1", "unregister-event-0", "unregister-event-1", "unregister-input-0"}));

  // Plain desktop never asks for a second input.
  nvenc_pipeline_probe desktop;
  desktop.second_input_out_of_memory = true;
  ASSERT_TRUE(desktop.create(video::SBS_OFF));
  EXPECT_EQ(desktop.pipeline_depth(), 1u);
  EXPECT_TRUE(only(desktop.operations(), {"input-", "unregister-input"}).empty());
}

TEST(NvencPipelineTest, DesktopHostSbsKeepsItsCaptureCadenceWithTwoPicturesInFlight) {
  // The depth is encoder scheduling only: desktop Host SBS still converts each capture at capture
  // cadence, not as an independent provider's frames as they finish (encode_run()).
  video::config_t config {};
  config.sbs_mode = video::SBS_OFF;
  EXPECT_EQ(video::nvenc_pipeline_depth(config), 1u);
  EXPECT_FALSE(video::converts_independent_provider(config));
  config.sbs_mode = video::SBS_AI;
  EXPECT_EQ(video::nvenc_pipeline_depth(config), 2u);
  EXPECT_FALSE(video::converts_independent_provider(config));
  // The saved glasses source switch selects no route by itself: only a game mode is an
  // independent provider.
  config.sbs_config.reshade = true;
  EXPECT_EQ(video::nvenc_pipeline_depth(config), 2u);
  EXPECT_FALSE(video::converts_independent_provider(config));
  config.sbs_config.reshade = false;
  for (const int mode : {video::SBS_GAME_MONO, video::SBS_GAME_SBS}) {
    config.sbs_mode = mode;
    EXPECT_EQ(video::nvenc_pipeline_depth(config), 2u);
    EXPECT_TRUE(video::converts_independent_provider(config));
  }
}

TEST(NvencPipelineTest, HostSbsPicturesKeepTheConversionTheyWereSubmittedWith) {
  // Desktop Host SBS with two pictures in flight: encode_run() converts while earlier pictures still
  // encode. A retained-source poll of pending inference keeps the conversion target when the
  // inference is busy (a packed repeat without a draw) and redraws it with the frame's own depth
  // once complete; the next capture, a depth-ready install or a stream-gamma reconversion redraws it
  // too. Each picture encodes the target as it was when submitted, in submission order, also when
  // the encoder tears down with pictures in flight.
  enum content_e : int {
    a_with_previous_depth = 1,  ///< Capture A; its inference missed the same-frame budget.
    a_with_own_depth,
    b,  ///< The next capture.
    b_new_gamma,
  };

  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create(video::SBS_AI));
  ASSERT_EQ(probe.pipeline_depth(), 2u);
  probe.clear_operations();

  probe.convert(a_with_previous_depth);
  ASSERT_TRUE(probe.submit_frame(1, true));
  // The retained-source poll finds the inference busy: the target keeps its packed output.
  ASSERT_TRUE(probe.submit_frame(2, false));
  // The next poll finds it complete and redraws A with its own depth while both pictures encode.
  probe.convert(a_with_own_depth);
  probe.complete(0);
  EXPECT_EQ(probe.retrieve_frame().frame_index, 1u);
  ASSERT_TRUE(probe.submit_frame(3, false));
  // Capture B converts while pictures 2 and 3 encode.
  probe.convert(b);
  probe.complete(1);
  EXPECT_EQ(probe.retrieve_frame().frame_index, 2u);
  ASSERT_TRUE(probe.submit_frame(4, false));
  // A stream-gamma request reconverts B, then the encoder tears down with pictures 3 and 4 in flight.
  probe.convert(b_new_gamma);
  probe.complete_every_picture();
  probe.destroy_encoder();

  EXPECT_EQ(probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{1, a_with_previous_depth}, {2, a_with_previous_depth}, {3, a_with_own_depth}, {4, b}}));
  EXPECT_EQ(only(probe.operations(), {"copy", "submit", "lock"}), (std::vector<std::string> {
                                                                    "copy-0",
                                                                    "submit-0:1:idr",
                                                                    "copy-1",
                                                                    "submit-1:2",
                                                                    "lock-0",
                                                                    "copy-0",
                                                                    "submit-0:3",
                                                                    "lock-1",
                                                                    "copy-1",
                                                                    "submit-1:4",
                                                                    "lock-0",
                                                                    "lock-1",
                                                                  }));
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

TEST(NvencPipelineTest, MappedRegistrationSurvivesPoolMovementAndReleasesOnlyAfterRetirement) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create(video::SBS_AI));
  probe.convert(11);
  ASSERT_TRUE(probe.submit_frame(1, true));
  probe.move_first_registration_to_spare();
  EXPECT_TRUE(probe.slot_maps_input(0, 0));
  EXPECT_TRUE(probe.registration_is_mapped(0));
  EXPECT_TRUE(probe.registration_in_flight(0));
  EXPECT_FALSE(probe.release_registration(0));

  probe.convert(22);
  ASSERT_TRUE(probe.submit_frame(2, false));
  probe.complete(0);
  ASSERT_EQ(probe.retrieve_frame().frame_index, 1u);
  // Retrieval copied the bitstream and unlocked it, but only the submitting thread unmaps.
  EXPECT_TRUE(probe.registration_is_mapped(0));
  EXPECT_FALSE(probe.registration_in_flight(0));
  probe.clear_operations();
  ASSERT_TRUE(probe.release_registration(0));
  EXPECT_EQ(probe.operations(), (std::vector<std::string> {"unmap-0"}));
  EXPECT_FALSE(probe.registration_is_mapped(0));
  EXPECT_FALSE(probe.slot_maps_input(0, 0));
  EXPECT_TRUE(probe.registration_in_flight(1));
  EXPECT_FALSE(probe.release_registration(1));

  // The pool's fixed slot-zero entry is now null. prepare_input still selects the real spare
  // registration, and its successful map becomes this picture's new exact association.
  probe.convert(33);
  ASSERT_TRUE(probe.submit_frame(3, false));
  EXPECT_TRUE(probe.slot_maps_input(0, 0));
  EXPECT_TRUE(probe.registration_in_flight(0));
  probe.complete_every_picture();
  ASSERT_EQ(probe.retrieve_frame().frame_index, 2u);
  ASSERT_EQ(probe.retrieve_frame().frame_index, 3u);
  EXPECT_EQ(probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{1, 11}, {2, 22}, {3, 33}}));

  // The next picture would use slot one; release can still find slot zero's real registration.
  ASSERT_TRUE(probe.release_registration(0));
  EXPECT_FALSE(probe.registration_is_mapped(0));
  EXPECT_TRUE(probe.registration_is_mapped(1));
  EXPECT_FALSE(probe.registration_in_flight(1));
  probe.clear_operations();
  probe.destroy_encoder();
  EXPECT_EQ(only(probe.operations(), {"unmap", "unregister-input"}),
    (std::vector<std::string> {"unmap-1", "unregister-input-1", "unregister-input-0"}));
  EXPECT_FALSE(probe.registration_is_mapped(0));
  EXPECT_FALSE(probe.registration_is_mapped(1));
  EXPECT_FALSE(probe.registration_in_flight(0));
  EXPECT_FALSE(probe.registration_in_flight(1));
}

TEST(NvencPipelineTest, RefusedPreparedRegistrationDoesNotMapOrAdvanceThePicture) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.clear_operations();
  probe.refuse_prepared_input = true;
  EXPECT_FALSE(probe.submit_frame(1, true));
  EXPECT_EQ(probe.frames_in_flight(), 0u);
  EXPECT_TRUE(probe.operations().empty());
  EXPECT_FALSE(probe.registration_is_mapped(0));

  probe.refuse_prepared_input = false;
  probe.complete_every_picture();
  probe.convert(11);
  ASSERT_TRUE(probe.submit_frame(1, true));
  EXPECT_TRUE(probe.slot_maps_input(0, 0));
  EXPECT_EQ(probe.retrieve_frame().frame_index, 1u);
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
    },
                                                             false);
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
  EXPECT_EQ(only(probe.operations(), {"timeout-", "ready-"}), (std::vector<std::string> {"timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0", "timeout-0"}));
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
  EXPECT_EQ(std::count(silent_ops.begin(), silent_ops.end(), "timeout-0"), 8);  // Includes the zero-time deadline poll.
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
  EXPECT_EQ(probe.invalidate_ref_frames(2, 2), nvenc::nvenc_base::ref_frame_invalidation_e::applied);
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
  EXPECT_EQ(probe.invalidate_ref_frames(5, 5), nvenc::nvenc_base::ref_frame_invalidation_e::needs_idr);
  EXPECT_TRUE(only(probe.operations(), {"invalidate"}).empty());

  // 2-6 (extended to the last submitted picture) is five pictures: an IDR, decided now.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(2, 3));
  EXPECT_EQ(probe.invalidate_ref_frames(2, 3), nvenc::nvenc_base::ref_frame_invalidation_e::needs_idr);
  // Handled by that IDR: done.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(3, 4));
  EXPECT_EQ(probe.invalidate_ref_frames(3, 4), nvenc::nvenc_base::ref_frame_invalidation_e::unchanged);
  // Malformed: an IDR.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(9, 8));
  EXPECT_EQ(probe.invalidate_ref_frames(9, 8), nvenc::nvenc_base::ref_frame_invalidation_e::needs_idr);
  // After the last submitted picture (6), the very next index included: an IDR, never an
  // empty invalidation.
  EXPECT_FALSE(probe.would_invalidate_ref_frames(7, 7));
  EXPECT_EQ(probe.invalidate_ref_frames(7, 7), nvenc::nvenc_base::ref_frame_invalidation_e::needs_idr);
  EXPECT_FALSE(probe.would_invalidate_ref_frames(8, 9));
  EXPECT_EQ(probe.invalidate_ref_frames(8, 9), nvenc::nvenc_base::ref_frame_invalidation_e::needs_idr);
  EXPECT_TRUE(only(probe.operations(), {"invalidate"}).empty());
  EXPECT_EQ(probe.frames_in_flight(), 2u);

  // In flight pictures keep the confirmation they were submitted with; the next one carries it.
  EXPECT_FALSE(probe.retrieve_frame().after_ref_frame_invalidation);
  EXPECT_FALSE(probe.retrieve_frame().after_ref_frame_invalidation);
  ASSERT_TRUE(probe.submit_frame(7, true));
  EXPECT_TRUE(probe.retrieve_frame().after_ref_frame_invalidation);
}

namespace {
  // Exercise the real conversion/session/base forwarding path. The fake prepares independent
  // picture inputs so tests can observe retained content and submission ownership.
  class session_backend_t final: public platf::nvenc_encode_device_t {
  public:
    nvenc::nvenc_base &encoder() override {
      return probe;
    }

    int convert(platf::img_t &) override {
      ++conversions;
      if (!kept_input && !packed_repeat) {
        probe.convert(content);
      }
      if (throw_after_conversion) {
        throw std::runtime_error("conversion failed after possibly writing input");
      }
      return result;
    }

    int convert_with_encode_target(platf::img_t &image, std::chrono::steady_clock::time_point target) override {
      encode_target = target;
      return convert(image);
    }

    bool conversion_kept_input() const override {
      return kept_input;
    }

    void encoder_consumed_input() override {
      ++consumed;
    }

    nvenc_pipeline_probe probe;
    bool kept_input = false;
    bool packed_repeat = false;
    bool throw_after_conversion = false;
    int content = 0;
    int result = 0;
    unsigned conversions = 0, consumed = 0;
    std::optional<std::chrono::steady_clock::time_point> encode_target;
  };

  class NativeEncodeSessionInputTest: public testing::Test {
  protected:
    void SetUp() override {
      auto candidate = std::make_unique<session_backend_t>();
      backend = candidate.get();
      ASSERT_TRUE(backend->probe.create());
      backend->probe.complete_every_picture();
      session = std::make_unique<video::nvenc_encode_session_t>(
        platf::native_encode_owner_test_access::adopt(std::move(candidate), workers)
      );
    }

    void TearDown() override {
      session.reset();
      workers.seal_and_drain();
    }

    video::detail::tracked_async_worker_pool_t workers;
    session_backend_t *backend = nullptr;
    std::unique_ptr<video::nvenc_encode_session_t> session;
    platf::img_t image;
  };
}  // namespace

TEST_F(NativeEncodeSessionInputTest, SynchronousAndPipelinedEncodeBothRetireSourceReads) {
  ASSERT_FALSE(session->encode_frame(10, {}).data.empty());
  backend->kept_input = true;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_TRUE(session->submit_frame(11));
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  EXPECT_EQ(backend->consumed, 2u);
}

TEST_F(NativeEncodeSessionInputTest, WritingAndRetainedConversionsPreservePreparedContent) {
  backend->content = 11;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->encode_frame(10, {}).data.empty());
  backend->kept_input = true;
  const std::chrono::steady_clock::time_point target {123ms};
  ASSERT_EQ(session->convert_with_encode_target(image, target), 0);
  ASSERT_TRUE(session->submit_frame(11));
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  backend->kept_input = false;
  backend->content = 22;
  ASSERT_EQ(session->convert_with_encode_target(image, target + 1ms), 0);
  ASSERT_FALSE(session->encode_frame(12, {}).data.empty());
  EXPECT_EQ(backend->encode_target, target + 1ms);
  EXPECT_EQ(backend->probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}, {11, 11}, {12, 22}}));
}

TEST_F(NativeEncodeSessionInputTest, PackedRepeatStillPresentsAndHonorsIdr) {
  backend->content = 11;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->encode_frame(10, {}).data.empty());

  backend->packed_repeat = true;
  ASSERT_EQ(session->convert(image), 0);
  EXPECT_FALSE(session->conversion_kept_input());
  session->request_idr_frame();
  ASSERT_TRUE(session->submit_frame(11));
  ASSERT_FALSE(session->retrieve_frame().data.empty());

  const std::chrono::steady_clock::time_point target {123ms};
  ASSERT_EQ(session->convert_with_encode_target(image, target), 0);
  EXPECT_FALSE(session->conversion_kept_input());
  EXPECT_EQ(backend->encode_target, target);
  ASSERT_FALSE(session->encode_frame(12, {}).data.empty());

  // A redraw of the same source still writes the target, as after a gamma change.
  backend->packed_repeat = false;
  ASSERT_EQ(session->convert_with_encode_target(image, target + 1ms), 0);
  ASSERT_TRUE(session->submit_frame(13));
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  EXPECT_EQ(backend->probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}, {11, 11}, {12, 11}, {13, 11}}));
  EXPECT_EQ(only(backend->probe.operations(), {"submit"}), (std::vector<std::string> {"submit-0:10", "submit-1:11:idr", "submit-0:12", "submit-1:13"}));
}

TEST_F(NativeEncodeSessionInputTest, RetainedRetryAfterConversionFailureUsesPreparedContent) {
  backend->content = 11;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->encode_frame(10, {}).data.empty());

  backend->content = 22;
  backend->result = -1;
  EXPECT_EQ(session->convert(image), -1);
  backend->result = 0;
  backend->packed_repeat = true;
  ASSERT_EQ(session->convert(image), 0);
  EXPECT_FALSE(session->conversion_kept_input());
  ASSERT_FALSE(session->encode_frame(11, {}).data.empty());

  backend->packed_repeat = false;
  backend->content = 33;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->encode_frame(12, {}).data.empty());

  const std::chrono::steady_clock::time_point target {123ms};
  backend->content = 44;
  backend->throw_after_conversion = true;
  EXPECT_THROW(session->convert_with_encode_target(image, target), std::runtime_error);
  backend->throw_after_conversion = false;
  backend->packed_repeat = true;
  ASSERT_EQ(session->convert_with_encode_target(image, target + 1ms), 0);
  EXPECT_FALSE(session->conversion_kept_input());
  ASSERT_TRUE(session->submit_frame(13));
  ASSERT_FALSE(session->retrieve_frame().data.empty());

  backend->packed_repeat = false;
  backend->content = 55;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->encode_frame(14, {}).data.empty());
  EXPECT_EQ(backend->probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}, {11, 22}, {12, 33}, {13, 44}, {14, 55}}));
}

TEST_F(NativeEncodeSessionInputTest, BusyRefusalPreservesNewContentForRetainedRetry) {
  backend->content = 11;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_TRUE(session->submit_frame(10));
  backend->kept_input = true;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_TRUE(session->submit_frame(11));
  ASSERT_EQ(backend->probe.frames_in_flight(), 2u);
  backend->kept_input = false;
  backend->content = 22;
  ASSERT_EQ(session->convert(image), 0);  // The actual target changes before admission can refuse.
  EXPECT_FALSE(session->submit_frame(12));
  EXPECT_EQ(backend->probe.frames_in_flight(), 2u);
  backend->kept_input = true;
  ASSERT_EQ(session->convert(image), 0);
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  ASSERT_TRUE(session->submit_frame(12));
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  ASSERT_FALSE(session->retrieve_frame().data.empty());
  EXPECT_EQ(backend->probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}, {11, 11}, {12, 22}}));
  EXPECT_EQ(backend->consumed, 4u);  // Existing input-retirement callback also runs on refusal.
}

TEST_F(NativeEncodeSessionInputTest, NeedMoreInputStillReturnsFalseAndRetainsAcceptedInputForTeardown) {
  backend->content = 11;
  ASSERT_EQ(session->convert(image), 0);
  backend->probe.next_picture_status = NV_ENC_ERR_NEED_MORE_INPUT;
  EXPECT_FALSE(session->submit_frame(10));
  EXPECT_EQ(backend->probe.frames_in_flight(), 1u);
  EXPECT_TRUE(session->retrieve_frame().data.empty());
  EXPECT_TRUE(only(backend->probe.operations(), {"unmap", "unlock"}).empty());
  backend->probe.destroy_encoder();
  EXPECT_EQ(backend->probe.frames_in_flight(), 0u);
  EXPECT_EQ(backend->probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}}));
  EXPECT_EQ(only(backend->probe.operations(), {"eos", "lock", "unlock", "unmap"}), (std::vector<std::string> {"eos", "lock-0", "unlock-0", "unmap-0"}));
  EXPECT_EQ(backend->consumed, 1u);
}

TEST(NvencPipelineTest, DirectSubmitAndSynchronousEncodePreservePreparedContent) {
  nvenc_pipeline_probe probe;
  ASSERT_TRUE(probe.create());
  probe.complete_every_picture();
  probe.convert(11);
  ASSERT_TRUE(probe.submit_frame(10, false));
  ASSERT_FALSE(probe.retrieve_frame().data.empty());
  probe.convert(22);
  ASSERT_FALSE(probe.encode_frame(11, false).data.empty());
  EXPECT_EQ(probe.encoded_contents(), (std::vector<std::pair<std::uint64_t, int>> {{10, 11}, {11, 22}}));
}

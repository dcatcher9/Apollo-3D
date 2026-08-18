/**
 * @file src/nvenc/nvenc_base.cpp
 * @brief Definitions for abstract platform-agnostic base of standalone NVENC encoder.
 */
// this include
#include "nvenc_base.h"

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <format>
#include <string_view>

// local includes
#include "src/logging.h"
#include "src/utility.h"

#define MAKE_NVENC_VER(major, minor) ((major) | ((minor) << 24))

// Make sure we audit the encoder integration when bumping the Video Codec SDK version.
// Things to look out for:
// - NV_ENC_*_VER definitions where the value inside NVENCAPI_STRUCT_VERSION() was increased
// - Incompatible struct changes in nvEncodeAPI.h (fields removed, semantics changed, etc.)
// - Test every supported codec and bit depth on the minimum supported driver
#if NVENCAPI_VERSION != MAKE_NVENC_VER(13U, 1U)
  #error Check and update NVENC code for the new API contract!
#endif

namespace {

  std::array<std::atomic<int>, 3> observed_codec_max_widths {};
  std::array<std::atomic<int>, 3> observed_codec_max_heights {};

  struct quality_preset_t {
    const GUID *guid;
    std::string_view name;
  };

  const std::array quality_presets {
    quality_preset_t {&NV_ENC_PRESET_P1_GUID, "P1"},
    quality_preset_t {&NV_ENC_PRESET_P2_GUID, "P2"},
    quality_preset_t {&NV_ENC_PRESET_P3_GUID, "P3"},
    quality_preset_t {&NV_ENC_PRESET_P4_GUID, "P4"},
    quality_preset_t {&NV_ENC_PRESET_P5_GUID, "P5"},
    quality_preset_t {&NV_ENC_PRESET_P6_GUID, "P6"},
    quality_preset_t {&NV_ENC_PRESET_P7_GUID, "P7"},
  };

  GUID quality_preset_guid_from_number(unsigned number) {
    number = std::clamp(number, 1u, static_cast<unsigned>(quality_presets.size()));
    return *quality_presets[number - 1].guid;
  }

  bool equal_guids(const GUID &guid1, const GUID &guid2) {
    return std::memcmp(&guid1, &guid2, sizeof(GUID)) == 0;
  }

  std::string_view quality_preset_string_from_guid(const GUID &guid) {
    for (const auto &preset : quality_presets) {
      if (equal_guids(guid, *preset.guid)) {
        return preset.name;
      }
    }
    return "Unknown";
  }

}  // namespace

namespace nvenc {

  std::optional<int> max_encode_width_for_codec(int video_format) {
    if (video_format < 0 || video_format >= static_cast<int>(observed_codec_max_widths.size())) {
      return std::nullopt;
    }
    const int width = observed_codec_max_widths[video_format].load(std::memory_order_relaxed);
    return width > 0 ? std::optional<int> {width} : std::nullopt;
  }

  std::optional<int> max_encode_height_for_codec(int video_format) {
    if (video_format < 0 || video_format >= static_cast<int>(observed_codec_max_heights.size())) {
      return std::nullopt;
    }
    const int height = observed_codec_max_heights[video_format].load(std::memory_order_relaxed);
    return height > 0 ? std::optional<int> {height} : std::nullopt;
  }

  nvenc_hdr_metadata_t hdr_metadata_from_sunshine(
    const std::optional<SS_HDR_METADATA> &metadata,
    int video_format
  ) {
    nvenc_hdr_metadata_t result;
    if (!metadata || (video_format != 1 && video_format != 2)) {
      return result;
    }

    const auto rescale = [](std::uint32_t value, std::uint32_t numerator, std::uint32_t denominator) {
      return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(value) * numerator + denominator / 2) / denominator
      );
    };

    const std::uint32_t chroma_denominator = video_format == 2 ? 65536u : 50000u;
    const auto chroma = [&](const auto &point) {
      return CHROMA_POINTS {
        static_cast<std::uint16_t>(std::min(rescale(point.x, chroma_denominator, 50000u), 65535u)),
        static_cast<std::uint16_t>(std::min(rescale(point.y, chroma_denominator, 50000u), 65535u)),
      };
    };

    MASTERING_DISPLAY_INFO mastering_display {};
    // Sunshine stores primaries in RGB order; NVENC's structure is ordered G, B, R.
    mastering_display.r = chroma(metadata->displayPrimaries[0]);
    mastering_display.g = chroma(metadata->displayPrimaries[1]);
    mastering_display.b = chroma(metadata->displayPrimaries[2]);
    mastering_display.whitePoint = chroma(metadata->whitePoint);
    if (video_format == 2) {
      // AV1 mastering-display metadata uses 1/256 nit maximum and 1/16384 nit minimum.
      mastering_display.maxLuma =
        static_cast<std::uint32_t>(metadata->maxDisplayLuminance) * 256u;
      mastering_display.minLuma =
        rescale(metadata->minDisplayLuminance, 16384u, 10000u);
    } else {
      // HEVC mastering-display colour volume SEI uses 1/10000 nit for both luminances.
      mastering_display.maxLuma =
        static_cast<std::uint32_t>(metadata->maxDisplayLuminance) * 10000u;
      mastering_display.minLuma = metadata->minDisplayLuminance;
    }
    result.mastering_display = mastering_display;

    if (metadata->maxContentLightLevel || metadata->maxFrameAverageLightLevel) {
      result.content_light_level = CONTENT_LIGHT_LEVEL {
        metadata->maxContentLightLevel,
        metadata->maxFrameAverageLightLevel,
      };
    }
    return result;
  }

  nvenc_base::nvenc_base(NV_ENC_DEVICE_TYPE device_type):
      device_type(device_type) {
  }

  nvenc_base::~nvenc_base() {
    // Use destroy_encoder() instead
  }

  void nvenc_base::apply_async_encode_capability(int capability) {
    if (!async_event_handle || capability != 0) {
      return;
    }

    BOOST_LOG(warning) << "NvEnc: gpu doesn't support async encode";
    release_async_event();
  }

  bool nvenc_base::create_encoder(
    const nvenc_config &config,
    const video::config_t &client_config,
    const nvenc_colorspace_t &colorspace,
    NV_ENC_BUFFER_FORMAT buffer_format,
    const std::optional<SS_HDR_METADATA> &hdr_metadata
  ) {
    if (!nvenc && !init_library()) {
      return false;
    }

    if (encoder) {
      destroy_encoder();
    }
    auto fail_guard = util::fail_guard([this] {
      destroy_encoder();
    });

    encoder_params.width = client_config.width;
    encoder_params.height = client_config.height;
    encoder_params.buffer_format = buffer_format;
    encoder_params.rfi = true;
    encoder_params.video_format = client_config.videoFormat;
    this->hdr_metadata = hdr_metadata_from_sunshine(hdr_metadata, client_config.videoFormat);

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
    session_params.device = device;
    session_params.deviceType = device_type;
    session_params.apiVersion = NVENCAPI_VERSION;
    if (nvenc_failed(nvenc->nvEncOpenEncodeSessionEx(&session_params, &encoder))) {
      BOOST_LOG(error) << "NvEnc: NvEncOpenEncodeSessionEx() failed: " << last_nvenc_error_string;
      return false;
    }

    uint32_t encode_guid_count = 0;
    if (nvenc_failed(nvenc->nvEncGetEncodeGUIDCount(encoder, &encode_guid_count))) {
      BOOST_LOG(error) << "NvEnc: NvEncGetEncodeGUIDCount() failed: " << last_nvenc_error_string;
      return false;
    };

    std::vector<GUID> encode_guids(encode_guid_count);
    if (nvenc_failed(nvenc->nvEncGetEncodeGUIDs(encoder, encode_guids.data(), encode_guids.size(), &encode_guid_count))) {
      BOOST_LOG(error) << "NvEnc: NvEncGetEncodeGUIDs() failed: " << last_nvenc_error_string;
      return false;
    }

    NV_ENC_INITIALIZE_PARAMS init_params = {NV_ENC_INITIALIZE_PARAMS_VER};

    switch (client_config.videoFormat) {
      case 0:
        // H.264
        init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
        break;

      case 1:
        // HEVC
        init_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
        break;

      case 2:
        // AV1
        init_params.encodeGUID = NV_ENC_CODEC_AV1_GUID;
        break;

      default:
        BOOST_LOG(error) << "NvEnc: unknown video format " << client_config.videoFormat;
        return false;
    }

    {
      auto search_predicate = [&](const GUID &guid) {
        return equal_guids(init_params.encodeGUID, guid);
      };
      if (std::find_if(encode_guids.begin(), encode_guids.end(), search_predicate) == encode_guids.end()) {
        BOOST_LOG(error) << "NvEnc: encoding format is not supported by the gpu";
        return false;
      }
    }

    auto get_encoder_cap = [&](NV_ENC_CAPS cap) {
      NV_ENC_CAPS_PARAM param = {NV_ENC_CAPS_PARAM_VER};
      param.capsToQuery = cap;
      int value = 0;
      if (nvenc->nvEncGetEncodeCaps(encoder, init_params.encodeGUID, &param, &value) == NV_ENC_SUCCESS) {
        return value;
      }
      return 0;
    };

    auto buffer_is_10bit = [&]() {
      return buffer_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    };

    {
      auto supported_width = get_encoder_cap(NV_ENC_CAPS_WIDTH_MAX);
      auto supported_height = get_encoder_cap(NV_ENC_CAPS_HEIGHT_MAX);
      if (supported_width > 0 && supported_height > 0 && client_config.videoFormat >= 0 && client_config.videoFormat < static_cast<int>(observed_codec_max_widths.size())) {
        const int previous_width = observed_codec_max_widths[client_config.videoFormat].exchange(
          supported_width,
          std::memory_order_relaxed
        );
        const int previous_height = observed_codec_max_heights[client_config.videoFormat].exchange(
          supported_height,
          std::memory_order_relaxed
        );
        if (previous_width != supported_width || previous_height != supported_height) {
          BOOST_LOG(info) << "NvEnc: codec " << client_config.videoFormat
                          << " maximum encode dimensions " << supported_width << 'x'
                          << supported_height;
        }
      }
      if (encoder_params.width > supported_width || encoder_params.height > supported_height) {
        BOOST_LOG(error) << "NvEnc: gpu max encode resolution " << supported_width << "x" << supported_height << ", requested " << encoder_params.width << "x" << encoder_params.height;
        return false;
      }
    }

    if (buffer_is_10bit() && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE)) {
      BOOST_LOG(error) << "NvEnc: gpu doesn't support 10-bit encode";
      return false;
    }

    if (async_event_handle) {
      apply_async_encode_capability(
        get_encoder_cap(NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT)
      );
    }

    encoder_params.rfi = get_encoder_cap(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION);

    init_params.presetGUID = quality_preset_guid_from_number(config.quality_preset);
    init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    init_params.enablePTD = 1;
    init_params.enableEncodeAsync = async_event_handle ? 1 : 0;
    const bool hevc_unidirectional_b_supported =
      client_config.videoFormat == 1 && get_encoder_cap(NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B);
    if (should_enable_hevc_unidirectional_b(config, client_config.videoFormat, hevc_unidirectional_b_supported)) {
      init_params.enableUniDirectionalB = 1;
    } else if (config.hevc_unidirectional_b && client_config.videoFormat == 1) {
      if (!hevc_unidirectional_b_supported) {
        BOOST_LOG(warning) << "NvEnc: HEVC unidirectional B-frames requested but not supported by the GPU";
      }
    }

    init_params.encodeWidth = encoder_params.width;
    init_params.darWidth = encoder_params.width;
    init_params.encodeHeight = encoder_params.height;
    init_params.darHeight = encoder_params.height;
    init_params.frameRateNum = client_config.framerate;
    init_params.frameRateDen = 1;
    if (client_config.framerateX100 > 0) {
      const auto framerate = video::framerate_x100_to_rational(client_config.framerateX100);
      init_params.frameRateNum = framerate.num;
      init_params.frameRateDen = framerate.den;
    }

    const int encoder_engine_count = client_config.videoFormat > 0 ?
                                       get_encoder_cap(NV_ENC_CAPS_NUM_ENCODER_ENGINES) :
                                       0;
    const bool split_frame_encoding = should_force_split_frame_encoding(
      client_config.sbs_mode == video::SBS_AI,
      client_config.videoFormat,
      encoder_params.width,
      encoder_engine_count
    );

    // Packed Host SBS is wider than 4K and can saturate one NVENC engine before reaching the
    // requested cadence. Let the driver split that workload across its available engines,
    // while preserving the single-engine bitstream for ordinary HEVC/AV1 streams.
    if (client_config.videoFormat > 0) {
      init_params.splitEncodeMode = split_frame_encoding ?
                                      NV_ENC_SPLIT_AUTO_FORCED_MODE :
                                      NV_ENC_SPLIT_DISABLE_MODE;
    }

    NV_ENC_PRESET_CONFIG preset_config = {};
    preset_config.version = NV_ENC_PRESET_CONFIG_VER;
    preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
    if (nvenc_failed(nvenc->nvEncGetEncodePresetConfigEx(encoder, init_params.encodeGUID, init_params.presetGUID, init_params.tuningInfo, &preset_config))) {
      BOOST_LOG(error) << "NvEnc: NvEncGetEncodePresetConfigEx() failed: " << last_nvenc_error_string;
      return false;
    }

    NV_ENC_CONFIG enc_config = preset_config.presetCfg;
    enc_config.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
    enc_config.gopLength = NVENC_INFINITE_GOPLENGTH;
    enc_config.frameIntervalP = 1;
    enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    enc_config.rcParams.zeroReorderDelay = 1;
    enc_config.rcParams.enableLookahead = 0;
    enc_config.rcParams.lowDelayKeyFrameScale = 1;
    enc_config.rcParams.multiPass = config.two_pass == nvenc_two_pass::quarter_resolution ? NV_ENC_TWO_PASS_QUARTER_RESOLUTION :
                                    config.two_pass == nvenc_two_pass::full_resolution    ? NV_ENC_TWO_PASS_FULL_RESOLUTION :
                                                                                            NV_ENC_MULTI_PASS_DISABLED;

    enc_config.rcParams.enableAQ = config.adaptive_quantization;
    const bool custom_vbv_supported =
      get_encoder_cap(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE) != 0;
    const auto rate_control = nvenc_rate_control_values(
      client_config.bitrate,
      client_config.framerate,
      config.vbv_percentage_increase,
      custom_vbv_supported
    );
    if (!rate_control) {
      BOOST_LOG(error) << "NvEnc: invalid bitrate/cadence for rate control";
      return false;
    }
    enc_config.rcParams.averageBitRate = rate_control->average_bitrate;
    if (rate_control->vbv_buffer_size) {
      enc_config.rcParams.vbvBufferSize = *rate_control->vbv_buffer_size;
    }

    auto set_h264_hevc_common_format_config = [&](auto &format_config) {
      format_config.repeatSPSPPS = 1;
      format_config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
      format_config.sliceMode = 3;
      format_config.sliceModeData = client_config.slicesPerFrame;
    };

    auto set_ref_frames = [&](uint32_t &ref_frames_option, NV_ENC_NUM_REF_FRAMES &L0_option, uint32_t ref_frames_default) {
      if (client_config.numRefFrames > 0) {
        ref_frames_option = client_config.numRefFrames;
      } else {
        ref_frames_option = ref_frames_default;
      }
      if (ref_frames_option > 0 && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)) {
        ref_frames_option = 1;
        encoder_params.rfi = false;
      }
      encoder_params.ref_frames_in_dpb = ref_frames_option;
      // This limits ref frames any frame can use to 1, but allows larger buffer size for fallback if some frames are invalidated through rfi
      L0_option = NV_ENC_NUM_REF_FRAMES_1;
    };

    auto fill_h264_hevc_vui = [&](auto &vui_config) {
      vui_config.videoSignalTypePresentFlag = 1;
      vui_config.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
      vui_config.videoFullRangeFlag = colorspace.full_range;
      vui_config.colourDescriptionPresentFlag = 1;
      vui_config.colourPrimaries = colorspace.primaries;
      vui_config.transferCharacteristics = colorspace.transfer_function;
      vui_config.colourMatrix = colorspace.matrix;
      vui_config.chromaSampleLocationFlag = 1;
      vui_config.chromaSampleLocationTop = 0;
      vui_config.chromaSampleLocationBot = 0;

      // This is critical for low decoding latency on certain devices.
      vui_config.bitstreamRestrictionFlag = 1;
    };

    switch (client_config.videoFormat) {
      case 0:
        {
          // H.264
          enc_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
          auto &format_config = enc_config.encodeCodecConfig.h264Config;
          set_h264_hevc_common_format_config(format_config);
          if (!get_encoder_cap(NV_ENC_CAPS_SUPPORT_CABAC)) {
            format_config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
          } else {
            format_config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
          }
          set_ref_frames(format_config.maxNumRefFrames, format_config.numRefL0, 5);
          fill_h264_hevc_vui(format_config.h264VUIParameters);
          break;
        }

      case 1:
        {
          // HEVC
          auto &format_config = enc_config.encodeCodecConfig.hevcConfig;
          set_h264_hevc_common_format_config(format_config);
          if (buffer_is_10bit()) {
            format_config.inputBitDepth = NV_ENC_BIT_DEPTH_10;
            format_config.outputBitDepth = NV_ENC_BIT_DEPTH_10;
          }
          set_ref_frames(format_config.maxNumRefFramesInDPB, format_config.numRefL0, 5);
          if (init_params.enableUniDirectionalB) {
            format_config.numRefL1 = NV_ENC_NUM_REF_FRAMES_1;
          }
          fill_h264_hevc_vui(format_config.hevcVUIParameters);
          format_config.outputMasteringDisplay = this->hdr_metadata.mastering_display.has_value();
          format_config.outputMaxCll = this->hdr_metadata.content_light_level.has_value();
          break;
        }

      case 2:
        {
          // AV1
          auto &format_config = enc_config.encodeCodecConfig.av1Config;
          format_config.repeatSeqHdr = 1;
          format_config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
          if (buffer_is_10bit()) {
            format_config.inputBitDepth = NV_ENC_BIT_DEPTH_10;
            format_config.outputBitDepth = NV_ENC_BIT_DEPTH_10;
          }
          format_config.colorPrimaries = colorspace.primaries;
          format_config.transferCharacteristics = colorspace.transfer_function;
          format_config.matrixCoefficients = colorspace.matrix;
          format_config.colorRange = colorspace.full_range;
          format_config.chromaSamplePosition = 1;
          format_config.outputMasteringDisplay = this->hdr_metadata.mastering_display.has_value();
          format_config.outputMaxCll = this->hdr_metadata.content_light_level.has_value();
          set_ref_frames(format_config.maxNumRefFramesInDPB, format_config.numFwdRefs, 8);
          if (client_config.slicesPerFrame > 1) {
            // NVENC only supports slice counts that are powers of two, so we'll pick powers of two
            // with bias to rows due to hopefully more similar macroblocks with a row vs a column.
            format_config.numTileRows = std::pow(2, std::ceil(std::log2(client_config.slicesPerFrame) / 2));
            format_config.numTileColumns = std::pow(2, std::floor(std::log2(client_config.slicesPerFrame) / 2));
          }
          break;
        }
    }

    init_params.encodeConfig = &enc_config;

    if (nvenc_failed(nvenc->nvEncInitializeEncoder(encoder, &init_params))) {
      BOOST_LOG(error) << "NvEnc: NvEncInitializeEncoder() failed: " << last_nvenc_error_string;
      return false;
    }

    if (async_event_handle) {
      NV_ENC_EVENT_PARAMS event_params = {NV_ENC_EVENT_PARAMS_VER};
      event_params.completionEvent = async_event_handle;
      if (nvenc_failed(nvenc->nvEncRegisterAsyncEvent(encoder, &event_params))) {
        BOOST_LOG(error) << "NvEnc: NvEncRegisterAsyncEvent() failed: " << last_nvenc_error_string;
        return false;
      }
    }

    NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream_buffer = {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
    if (nvenc_failed(nvenc->nvEncCreateBitstreamBuffer(encoder, &create_bitstream_buffer))) {
      BOOST_LOG(error) << "NvEnc: NvEncCreateBitstreamBuffer() failed: " << last_nvenc_error_string;
      return false;
    }
    output_bitstream = create_bitstream_buffer.bitstreamBuffer;

    if (!create_and_register_input_buffer()) {
      return false;
    }

    {
      auto f = stat_trackers::two_digits_after_decimal();
      BOOST_LOG(debug) << "NvEnc: requested encoded frame size " << f % (client_config.bitrate / 8. / client_config.framerate) << " kB";
    }

    {
      auto video_format_string = client_config.videoFormat == 0 ? "H.264 " :
                                 client_config.videoFormat == 1 ? "HEVC " :
                                 client_config.videoFormat == 2 ? "AV1 " :
                                                                  " ";
      std::string extra;
      if (init_params.enableEncodeAsync) {
        extra += " async";
      }
      if (buffer_is_10bit()) {
        extra += " 10-bit";
      }
      if (enc_config.rcParams.multiPass != NV_ENC_MULTI_PASS_DISABLED) {
        extra += " two-pass";
      }
      if (config.vbv_percentage_increase > 0 && custom_vbv_supported) {
        extra += std::format(" vbv+{}", config.vbv_percentage_increase);
      }
      if (encoder_params.rfi) {
        extra += " rfi";
      }
      if (init_params.enableUniDirectionalB) {
        extra += " hevc-unidirectional-b";
      }
      if (split_frame_encoding) {
        extra += std::format(" split-encode auto-forced ({} engines available)", encoder_engine_count);
      }
      if (enc_config.rcParams.enableAQ) {
        extra += " spatial-aq";
      }

      BOOST_LOG(info) << "NvEnc: created encoder " << video_format_string << quality_preset_string_from_guid(init_params.presetGUID) << extra;
    }

    encoder_state = {};
    encoder_state.bitrate_reconfiguration_supported =
      get_encoder_cap(NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE) != 0;
    encoder_state.custom_vbv_supported = custom_vbv_supported;
    encoder_state.framerate = client_config.framerate;
    encoder_state.vbv_percentage_increase = config.vbv_percentage_increase;
    encoder_state.initialize_params = init_params;
    // Never retain a pointer to create_encoder()'s stack-local configuration. Reconfiguration
    // installs a pointer to its own candidate copy for the duration of the driver call.
    encoder_state.initialize_params.encodeConfig = nullptr;
    encoder_state.encode_config = enc_config;
    fail_guard.disable();
    return true;
  }

  void nvenc_base::destroy_encoder() {
    if (output_bitstream) {
      if (nvenc_failed(nvenc->nvEncDestroyBitstreamBuffer(encoder, output_bitstream))) {
        BOOST_LOG(error) << "NvEnc: NvEncDestroyBitstreamBuffer() failed: " << last_nvenc_error_string;
      }
      output_bitstream = nullptr;
    }
    if (encoder && async_event_handle) {
      NV_ENC_EVENT_PARAMS event_params = {NV_ENC_EVENT_PARAMS_VER};
      event_params.completionEvent = async_event_handle;
      if (nvenc_failed(nvenc->nvEncUnregisterAsyncEvent(encoder, &event_params))) {
        BOOST_LOG(error) << "NvEnc: NvEncUnregisterAsyncEvent() failed: " << last_nvenc_error_string;
      }
    }
    if (registered_input_buffer) {
      if (nvenc_failed(nvenc->nvEncUnregisterResource(encoder, registered_input_buffer))) {
        BOOST_LOG(error) << "NvEnc: NvEncUnregisterResource() failed: " << last_nvenc_error_string;
      }
      registered_input_buffer = nullptr;
    }
    if (encoder) {
      if (nvenc_failed(nvenc->nvEncDestroyEncoder(encoder))) {
        BOOST_LOG(error) << "NvEnc: NvEncDestroyEncoder() failed: " << last_nvenc_error_string;
      }
      encoder = nullptr;
    }

    encoder_state = {};
    encoder_params = {};
    hdr_metadata = {};
  }

  nvenc_encoded_frame nvenc_base::encode_frame(
    uint64_t frame_index,
    bool force_idr,
    std::vector<std::uint8_t> frame_buffer
  ) {
    if (!encoder) {
      return {};
    }

    assert(registered_input_buffer);
    assert(output_bitstream);

    NV_ENC_MAP_INPUT_RESOURCE mapped_input_buffer = {NV_ENC_MAP_INPUT_RESOURCE_VER};
    mapped_input_buffer.registeredResource = registered_input_buffer;

    if (nvenc_failed(nvenc->nvEncMapInputResource(encoder, &mapped_input_buffer))) {
      BOOST_LOG(error) << "NvEnc: NvEncMapInputResource() failed: " << last_nvenc_error_string;
      return {};
    }
    auto unmap_guard = util::fail_guard([&] {
      if (nvenc_failed(nvenc->nvEncUnmapInputResource(encoder, mapped_input_buffer.mappedResource))) {
        BOOST_LOG(error) << "NvEnc: NvEncUnmapInputResource() failed: " << last_nvenc_error_string;
      }
    });

    NV_ENC_PIC_PARAMS pic_params = {NV_ENC_PIC_PARAMS_VER};
    pic_params.inputWidth = encoder_params.width;
    pic_params.inputHeight = encoder_params.height;
    pic_params.encodePicFlags = force_idr ? NV_ENC_PIC_FLAG_FORCEIDR : 0;
    pic_params.inputTimeStamp = frame_index;
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic_params.inputBuffer = mapped_input_buffer.mappedResource;
    pic_params.bufferFmt = mapped_input_buffer.mappedBufferFmt;
    pic_params.outputBitstream = output_bitstream;
    pic_params.completionEvent = async_event_handle;

    if (encoder_params.video_format == 1) {
      pic_params.codecPicParams.hevcPicParams.pMasteringDisplay =
        hdr_metadata.mastering_display ? &*hdr_metadata.mastering_display : nullptr;
      pic_params.codecPicParams.hevcPicParams.pMaxCll =
        hdr_metadata.content_light_level ? &*hdr_metadata.content_light_level : nullptr;
    } else if (encoder_params.video_format == 2) {
      pic_params.codecPicParams.av1PicParams.pMasteringDisplay =
        hdr_metadata.mastering_display ? &*hdr_metadata.mastering_display : nullptr;
      pic_params.codecPicParams.av1PicParams.pMaxCll =
        hdr_metadata.content_light_level ? &*hdr_metadata.content_light_level : nullptr;
    }

    if (nvenc_failed(nvenc->nvEncEncodePicture(encoder, &pic_params))) {
      BOOST_LOG(error) << "NvEnc: NvEncEncodePicture() failed: " << last_nvenc_error_string;
      return {};
    }

    NV_ENC_LOCK_BITSTREAM lock_bitstream = {NV_ENC_LOCK_BITSTREAM_VER};
    lock_bitstream.outputBitstream = output_bitstream;
    lock_bitstream.doNotWait = async_event_handle ? 1 : 0;

    if (async_event_handle && !wait_for_async_event(100)) {
      BOOST_LOG(error) << "NvEnc: frame " << frame_index << " encode wait timeout";
      return {};
    }

    if (nvenc_failed(nvenc->nvEncLockBitstream(encoder, &lock_bitstream))) {
      BOOST_LOG(error) << "NvEnc: NvEncLockBitstream() failed: " << last_nvenc_error_string;
      return {};
    }

    const auto frame_average_qp = lock_bitstream.frameAvgQP;
    const auto frame_satd = lock_bitstream.frameSatd;
    auto data_pointer = (uint8_t *) lock_bitstream.bitstreamBufferPtr;
    // The ENCODED_PACKET_QUEUE_LIMIT bounds how many of these buffers can be in flight. Reusing
    // their capacity removes a large allocation from the encode path while still copying before
    // NvEncUnlockBitstream(), as required by the NVENC ownership contract.
    frame_buffer.assign(
      data_pointer,
      data_pointer + lock_bitstream.bitstreamSizeInBytes
    );
    nvenc_encoded_frame encoded_frame {
      std::move(frame_buffer),
      lock_bitstream.outputTimeStamp,
      lock_bitstream.pictureType == NV_ENC_PIC_TYPE_IDR,
      encoder_state.rfi_needs_confirmation,
    };

    if (encoder_state.rfi_needs_confirmation) {
      // Invalidation request has been fulfilled, and video network packet will be marked as such
      encoder_state.rfi_needs_confirmation = false;
    }

    encoder_state.last_encoded_frame_index = frame_index;

    if (encoded_frame.idr) {
      BOOST_LOG(debug) << "NvEnc: idr frame " << encoded_frame.frame_index;
    }

    if (nvenc_failed(nvenc->nvEncUnlockBitstream(encoder, lock_bitstream.outputBitstream))) {
      BOOST_LOG(error) << "NvEnc: NvEncUnlockBitstream() failed: " << last_nvenc_error_string;
    }

    encoder_state.frame_size_logger.collect_and_log(encoded_frame.data.size() / 1000.);
    // Both values are returned by NV_ENC_LOCK_BITSTREAM without enabling the substantially richer
    // getRCStats path. When diagnostics are disabled, each logger is only one predictable branch.
    encoder_state.frame_qp_logger.collect_and_log(frame_average_qp);
    encoder_state.frame_satd_logger.collect_and_log(frame_satd);

    return encoded_frame;
  }

  bool nvenc_base::reconfigure_bitrate(int bitrate_kbps) {
    if (!encoder || !encoder_state.bitrate_reconfiguration_supported) {
      return false;
    }

    const auto rate_control = nvenc_rate_control_values(
      bitrate_kbps,
      encoder_state.framerate,
      encoder_state.vbv_percentage_increase,
      encoder_state.custom_vbv_supported
    );
    if (!rate_control) {
      BOOST_LOG(error) << "NvEnc: refusing invalid bitrate-only reconfiguration";
      return false;
    }

    NV_ENC_CONFIG candidate_config = encoder_state.encode_config;
    candidate_config.rcParams.averageBitRate = rate_control->average_bitrate;
    if (rate_control->vbv_buffer_size) {
      candidate_config.rcParams.vbvBufferSize = *rate_control->vbv_buffer_size;
    }

    // Treat an identical request as a successful no-op. This preserves the live session without
    // asking the driver to reset any internal rate-control history.
    if (candidate_config.rcParams.averageBitRate == encoder_state.encode_config.rcParams.averageBitRate && candidate_config.rcParams.vbvBufferSize == encoder_state.encode_config.rcParams.vbvBufferSize) {
      return true;
    }

    NV_ENC_RECONFIGURE_PARAMS reconfigure_params = {NV_ENC_RECONFIGURE_PARAMS_VER};
    reconfigure_params.reInitEncodeParams = encoder_state.initialize_params;
    reconfigure_params.reInitEncodeParams.encodeConfig = &candidate_config;
    reconfigure_params.resetEncoder = 0;
    reconfigure_params.forceIDR = 0;

    if (nvenc_failed(nvenc->nvEncReconfigureEncoder(encoder, &reconfigure_params))) {
      BOOST_LOG(warning) << "NvEnc: bitrate-only NvEncReconfigureEncoder() failed: "
                         << last_nvenc_error_string
                         << "; rebuilding the encode session";
      return false;
    }

    // Commit only after the driver accepted the complete candidate. A failed call leaves these
    // snapshots untouched, allowing capture_async's existing proven-mode rebuild to remain the
    // sole fallback authority.
    encoder_state.encode_config = candidate_config;
    BOOST_LOG(info) << "NvEnc: reconfigured bitrate to " << bitrate_kbps
                    << "kbps without resetting the session or forcing an IDR";
    return true;
  }

  bool nvenc_base::invalidate_ref_frames(uint64_t first_frame, uint64_t last_frame) {
    if (!encoder || !encoder_params.rfi) {
      return false;
    }

    if (first_frame >= encoder_state.last_rfi_range.first && last_frame <= encoder_state.last_rfi_range.second) {
      BOOST_LOG(debug) << "NvEnc: rfi request " << first_frame << "-" << last_frame << " already done";
      return true;
    }

    encoder_state.rfi_needs_confirmation = true;

    if (last_frame < first_frame) {
      BOOST_LOG(error) << "NvEnc: invaid rfi request " << first_frame << "-" << last_frame << ", generating IDR";
      return false;
    }

    BOOST_LOG(debug) << "NvEnc: rfi request " << first_frame << "-" << last_frame << " expanding to last encoded frame " << encoder_state.last_encoded_frame_index;
    last_frame = encoder_state.last_encoded_frame_index;

    encoder_state.last_rfi_range = {first_frame, last_frame};

    if (last_frame - first_frame + 1 >= encoder_params.ref_frames_in_dpb) {
      BOOST_LOG(debug) << "NvEnc: rfi request too large, generating IDR";
      return false;
    }

    for (auto i = first_frame; i <= last_frame; i++) {
      if (nvenc_failed(nvenc->nvEncInvalidateRefFrames(encoder, i))) {
        BOOST_LOG(error) << "NvEnc: NvEncInvalidateRefFrames() " << i << " failed: " << last_nvenc_error_string;
        return false;
      }
    }

    return true;
  }

  bool nvenc_base::nvenc_failed(NVENCSTATUS status) {
    auto status_string = [](NVENCSTATUS status) -> std::string {
      switch (status) {
#define nvenc_status_case(x) \
  case x: \
    return #x;
        nvenc_status_case(NV_ENC_SUCCESS);
        nvenc_status_case(NV_ENC_ERR_NO_ENCODE_DEVICE);
        nvenc_status_case(NV_ENC_ERR_UNSUPPORTED_DEVICE);
        nvenc_status_case(NV_ENC_ERR_INVALID_ENCODERDEVICE);
        nvenc_status_case(NV_ENC_ERR_INVALID_DEVICE);
        nvenc_status_case(NV_ENC_ERR_DEVICE_NOT_EXIST);
        nvenc_status_case(NV_ENC_ERR_INVALID_PTR);
        nvenc_status_case(NV_ENC_ERR_INVALID_EVENT);
        nvenc_status_case(NV_ENC_ERR_INVALID_PARAM);
        nvenc_status_case(NV_ENC_ERR_INVALID_CALL);
        nvenc_status_case(NV_ENC_ERR_OUT_OF_MEMORY);
        nvenc_status_case(NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
        nvenc_status_case(NV_ENC_ERR_UNSUPPORTED_PARAM);
        nvenc_status_case(NV_ENC_ERR_LOCK_BUSY);
        nvenc_status_case(NV_ENC_ERR_NOT_ENOUGH_BUFFER);
        nvenc_status_case(NV_ENC_ERR_INVALID_VERSION);
        nvenc_status_case(NV_ENC_ERR_MAP_FAILED);
        nvenc_status_case(NV_ENC_ERR_NEED_MORE_INPUT);
        nvenc_status_case(NV_ENC_ERR_ENCODER_BUSY);
        nvenc_status_case(NV_ENC_ERR_EVENT_NOT_REGISTERD);
        nvenc_status_case(NV_ENC_ERR_GENERIC);
        nvenc_status_case(NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY);
        nvenc_status_case(NV_ENC_ERR_UNIMPLEMENTED);
        nvenc_status_case(NV_ENC_ERR_RESOURCE_REGISTER_FAILED);
        nvenc_status_case(NV_ENC_ERR_RESOURCE_NOT_REGISTERED);
        nvenc_status_case(NV_ENC_ERR_RESOURCE_NOT_MAPPED);
        nvenc_status_case(NV_ENC_ERR_NEED_MORE_OUTPUT);
        // Newer versions of sdk may add more constants, look for them at the end of NVENCSTATUS enum
#undef nvenc_status_case
        default:
          return std::to_string(status);
      }
    };

    last_nvenc_error_string.clear();
    if (status != NV_ENC_SUCCESS) {
      last_nvenc_error_string += status_string(status);
      return true;
    }

    return false;
  }

}  // namespace nvenc

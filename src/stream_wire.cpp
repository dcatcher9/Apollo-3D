/**
 * @file src/stream_wire.cpp
 * @brief Definitions for the fixed-layout stream control-message codecs.
 */

// standard includes
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

// local includes
#include "process.h"
#include "stream.h"
#include "video.h"

namespace stream {
  bool sbs_debug_dump_request_allowed(
    int requested_sbs_mode,
    bool has_session_request_latch
  ) noexcept {
    return has_session_request_latch &&
           (requested_sbs_mode == ::video::SBS_AI || ::video::is_game_mode(requested_sbs_mode));
  }

  stream_gamma_request_decode_e decode_stream_gamma_request_payload(
    std::string_view payload,
    video::stream_gamma_request_t &request
  ) noexcept {
    request = {};
    if (payload.size() != STREAM_GAMMA_REQUEST_PAYLOAD_SIZE) {
      return stream_gamma_request_decode_e::invalid;
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    if (!video::valid_stream_gamma_mode(bytes[1])) {
      return stream_gamma_request_decode_e::invalid;
    }
    video::stream_gamma_request_t decoded;
    decoded.mode = static_cast<video::stream_gamma_mode_e>(bytes[1]);
    for (unsigned i = 0; i < 4; ++i) {
      decoded.request_id |= static_cast<std::uint32_t>(bytes[4 + i]) << (i * 8);
    }
    if (decoded.request_id == 0) {
      return stream_gamma_request_decode_e::invalid;
    }
    request = decoded;
    if (bytes[0] != STREAM_GAMMA_VERSION) {
      return stream_gamma_request_decode_e::unsupported_version;
    }
    return bytes[2] == 0 && bytes[3] == 0 ?
             stream_gamma_request_decode_e::ok :
             stream_gamma_request_decode_e::invalid;
  }

  bool encode_stream_gamma_ack_payload(
    const video::stream_gamma_ack_t &ack,
    std::uint8_t (&out)[STREAM_GAMMA_ACK_PAYLOAD_SIZE]
  ) noexcept {
    if (static_cast<std::uint8_t>(ack.status) > 3 ||
        !video::valid_stream_gamma_mode(static_cast<std::uint8_t>(ack.requested_mode)) ||
        !video::valid_stream_gamma_mode(static_cast<std::uint8_t>(ack.applied.mode)) ||
        ack.applied.generation == 0 || !std::isfinite(ack.applied.white_nits) ||
        ack.applied.white_nits < 40.0f || ack.applied.white_nits > 1000.0f ||
        (ack.status == video::stream_gamma_status_e::applied && ack.requested_mode != ack.applied.mode)) {
      return false;
    }
    std::uint8_t encoded[STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
    encoded[0] = STREAM_GAMMA_VERSION;
    encoded[1] = static_cast<std::uint8_t>(ack.status);
    encoded[2] = static_cast<std::uint8_t>(ack.requested_mode);
    encoded[3] = static_cast<std::uint8_t>(ack.applied.mode);
    const auto put_u32 = [&](std::size_t offset, std::uint32_t value) {
      for (unsigned i = 0; i < 4; ++i) {
        encoded[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
      }
    };
    put_u32(4, ack.request_id);
    put_u32(8, ack.applied.generation);
    put_u32(12, std::bit_cast<std::uint32_t>(ack.applied.white_nits));
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  live_video_mode_ack_e live_video_mode_ack_status(proc::live_video_mode_result_e result) {
    switch (result) {
      case proc::live_video_mode_result_e::applied:
      // "Unchanged" means the request repeats the session's requested mode, which the last proven
      // display transaction applied; an application may since have chosen another display mode,
      // and the host leaves that in place. The requested mode is live once the encoder rebuilds.
      // From the client's point of view that is indistinguishable from a real transition, and it
      // must not be reported as a failure.
      case proc::live_video_mode_result_e::unchanged:
        return live_video_mode_ack_e::applied;
      case proc::live_video_mode_result_e::needs_reconnect:
        return live_video_mode_ack_e::rejected_needs_reconnect;
      case proc::live_video_mode_result_e::failed:
        return live_video_mode_ack_e::failed;
    }

    // An unmapped outcome is a host bug, not a client one. Report the retryable status so a
    // client is never told to give up on a mode the host may still be able to deliver.
    return live_video_mode_ack_e::failed;
  }

  live_video_mode_request_decode_e decode_live_video_mode_request_payload(
    std::string_view payload,
    live_video_mode_wire_request_t &request
  ) noexcept {
    request = {};
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    const auto read_u16 = [&](std::size_t offset) {
      return static_cast<std::uint16_t>(bytes[offset]) |
             static_cast<std::uint16_t>(bytes[offset + 1] << 8);
    };
    const auto read_u32 = [&](std::size_t offset) {
      return static_cast<std::uint32_t>(bytes[offset]) |
             (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
             (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
             (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    };

    if (payload.size() != ATOMIC_PRESENTATION_V2_REQUEST_PAYLOAD_SIZE) {
      return live_video_mode_request_decode_e::invalid;
    }

    request.protocol_version = bytes[0];
    request.desired_sbs_mode = bytes[1];
    request.flags = read_u16(2);
    request.request_id = read_u32(4);
    request.source_width = read_u16(8);
    request.source_height = read_u16(10);
    request.framerate_x100 = read_u32(12);
    request.bitrate_kbps = read_u32(16);
    return request.protocol_version == ATOMIC_PRESENTATION_VERSION ?
             live_video_mode_request_decode_e::v2 :
             live_video_mode_request_decode_e::unsupported_version;
  }

  live_video_mode_ack_t make_live_video_mode_ack(
    std::uint32_t request_id,
    live_video_mode_ack_e status,
    const video::effective_video_mode_t &mode
  ) noexcept {
    return {
      request_id,
      status,
      mode.framerateX100,
      mode.bitrate,
      mode.sbs_mode,
      mode.source_width,
      mode.source_height,
      mode.encoded_width,
      mode.encoded_height,
      mode.generation,
    };
  }

  bool encode_atomic_presentation_ack_payload(
    const live_video_mode_ack_t &ack,
    std::uint8_t (&out)[ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE]
  ) {
    constexpr std::int64_t u16_max = std::numeric_limits<std::uint16_t>::max();
    constexpr std::int64_t u32_max = std::numeric_limits<std::uint32_t>::max();
    if (ack.status < live_video_mode_ack_e::applied || ack.status > live_video_mode_ack_e::failed || ack.request_id < 0 || ack.request_id > u32_max || (ack.applied_sbs_mode < video::SBS_OFF || ack.applied_sbs_mode > video::SBS_GAME_SBS) || ack.applied_source_width < 0 || ack.applied_source_width > u16_max || ack.applied_source_height < 0 || ack.applied_source_height > u16_max || ack.applied_encoded_width < 0 || ack.applied_encoded_width > u16_max || ack.applied_encoded_height < 0 || ack.applied_encoded_height > u16_max || ack.applied_framerate_x100 < 0 || ack.applied_framerate_x100 > u32_max || ack.applied_bitrate_kbps < 0 || ack.applied_bitrate_kbps > u32_max || ack.applied_generation == 0) {
      return false;
    }

    std::uint8_t encoded[ATOMIC_PRESENTATION_V2_ACK_PAYLOAD_SIZE] {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xFF);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xFF);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
      encoded[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
      encoded[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
    };

    encoded[0] = ATOMIC_PRESENTATION_VERSION;
    encoded[1] = static_cast<std::uint8_t>(ack.status);
    encoded[2] = static_cast<std::uint8_t>(ack.applied_sbs_mode);
    encoded[3] = 0;  // response flags, reserved
    write_u32(4, static_cast<std::uint32_t>(ack.request_id));
    write_u32(8, ack.applied_generation);
    write_u16(12, static_cast<std::uint16_t>(ack.applied_source_width));
    write_u16(14, static_cast<std::uint16_t>(ack.applied_source_height));
    write_u16(16, static_cast<std::uint16_t>(ack.applied_encoded_width));
    write_u16(18, static_cast<std::uint16_t>(ack.applied_encoded_height));
    write_u32(20, static_cast<std::uint32_t>(ack.applied_framerate_x100));
    write_u32(24, static_cast<std::uint32_t>(ack.applied_bitrate_kbps));
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  bool game_source_status_matches_mode(
    const video::game_source_state_t &state,
    const video::effective_video_mode_t &mode
  ) noexcept {
    return video::is_game_mode(mode.sbs_mode) && mode.generation != 0 &&
           state.presentation_generation == mode.generation &&
           state.source_width == mode.source_width && state.source_height == mode.source_height &&
           ((mode.sbs_mode == video::SBS_GAME_MONO && state.state != video::GAME_SOURCE_READY) ||
            (mode.encoded_width == (mode.sbs_mode == video::SBS_GAME_SBS ? mode.source_width * 2 : mode.source_width) &&
             mode.encoded_height == mode.source_height));
  }

  bool encode_game_source_status_payload(
    const video::game_source_state_t &state,
    std::uint8_t (&out)[GAME_SOURCE_STATUS_PAYLOAD_SIZE]
  ) noexcept {
    if (state.state > video::GAME_SOURCE_UNSUPPORTED || state.provider > video::GAME_PROVIDER_RESHADE || (state.state == video::GAME_SOURCE_READY && state.provider == video::GAME_PROVIDER_NONE) || state.presentation_generation == 0 || state.source_revision == 0 || !is_valid_live_video_mode_dimension(state.source_width) || !is_valid_live_video_mode_dimension(state.source_height) || state.packed_width != state.source_width * 2 || state.packed_height != state.source_height) {
      return false;
    }
    std::uint8_t encoded[GAME_SOURCE_STATUS_PAYLOAD_SIZE] {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value);
      encoded[offset + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      write_u16(offset, static_cast<std::uint16_t>(value));
      write_u16(offset + 2, static_cast<std::uint16_t>(value >> 16));
    };
    encoded[0] = GAME_SOURCE_STATUS_VERSION;
    encoded[1] = state.state;
    encoded[2] = state.provider;
    write_u32(4, state.presentation_generation);
    write_u32(8, state.source_revision);
    write_u16(12, state.source_width);
    write_u16(14, state.source_height);
    write_u16(16, state.packed_width);
    write_u16(18, state.packed_height);
    std::copy(std::begin(encoded), std::end(encoded), std::begin(out));
    return true;
  }

  sbs_telemetry_subscription_decode_e decode_sbs_telemetry_subscription_payload(
    std::string_view payload,
    sbs_telemetry_subscription_request_t &request
  ) {
    request = {};
    if (payload.size() != SBS_TELEMETRY_SUBSCRIPTION_PAYLOAD_SIZE) {
      return sbs_telemetry_subscription_decode_e::invalid;
    }

    const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
    request.flags = bytes[1];
    request.request_id =
      static_cast<std::uint16_t>(bytes[2]) |
      static_cast<std::uint16_t>(bytes[3] << 8);
    request.interval_ms =
      static_cast<std::uint16_t>(bytes[4]) |
      static_cast<std::uint16_t>(bytes[5] << 8);
    if (bytes[0] != SBS_TELEMETRY_VERSION) {
      return sbs_telemetry_subscription_decode_e::unsupported_version;
    }

    const std::uint16_t reserved =
      static_cast<std::uint16_t>(bytes[6]) |
      static_cast<std::uint16_t>(bytes[7] << 8);
    constexpr std::uint8_t known_flags =
      SBS_TELEMETRY_SUBSCRIBE_ENABLED | SBS_TELEMETRY_SUBSCRIBE_FOCUSED;
    if (reserved != 0 || (request.flags & ~known_flags) != 0) {
      return sbs_telemetry_subscription_decode_e::invalid;
    }

    return sbs_telemetry_subscription_decode_e::ok;
  }

  std::uint16_t effective_sbs_telemetry_interval_ms(
    const sbs_telemetry_subscription_request_t &request
  ) noexcept {
    if (request.interval_ms == 0) {
      return request.focused() ?
               SBS_TELEMETRY_INTERVAL_FOCUSED_MS :
               SBS_TELEMETRY_INTERVAL_BACKGROUND_MS;
    }
    return std::clamp(
      request.interval_ms,
      SBS_TELEMETRY_INTERVAL_MIN_MS,
      SBS_TELEMETRY_INTERVAL_MAX_MS
    );
  }

  sbs_telemetry_status_e sbs_telemetry_wire_status(
    video::sbs_telemetry_sample_status_e status
  ) noexcept {
    switch (status) {
      case video::sbs_telemetry_sample_status_e::ready:
        return sbs_telemetry_status_e::ok;
      case video::sbs_telemetry_sample_status_e::unavailable:
        return sbs_telemetry_status_e::unavailable;
      case video::sbs_telemetry_sample_status_e::failed:
        return sbs_telemetry_status_e::failed;
    }
    return sbs_telemetry_status_e::failed;
  }

  bool encode_sbs_telemetry_state_payload(
    const sbs_telemetry_state_t &state,
    std::uint8_t (&out)[SBS_TELEMETRY_STATE_PAYLOAD_SIZE]
  ) {
    const auto status = static_cast<std::uint8_t>(state.status);
    if (status > static_cast<std::uint8_t>(sbs_telemetry_status_e::failed)) {
      return false;
    }

    const auto &snapshot = state.snapshot;
    const auto performance = snapshot.performance ? snapshot.performance->snapshot() : host_sbs_telemetry::sample {};
    const std::array<float, 8> float_fields {
      snapshot.pop_floor,
      snapshot.pop_ceiling,
      snapshot.effective_pop,
      snapshot.edge_fraction,
      snapshot.zero_anchor_shift_px,
      snapshot.subject_depth,
      snapshot.valid_depth_fraction,
      snapshot.effective_range_width,
    };
    if (std::any_of(float_fields.begin(), float_fields.end(), [](float value) {
          return !std::isfinite(value);
        })) {
      return false;
    }

    std::array<std::uint8_t, SBS_TELEMETRY_STATE_PAYLOAD_SIZE> encoded {};
    const auto write_u16 = [&](std::size_t offset, std::uint16_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xffu);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    };
    const auto write_u32 = [&](std::size_t offset, std::uint32_t value) {
      encoded[offset] = static_cast<std::uint8_t>(value & 0xffu);
      encoded[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
      encoded[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
      encoded[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
    };
    const auto write_f32 = [&](std::size_t offset, float value) {
      write_u32(offset, std::bit_cast<std::uint32_t>(value));
    };
    const auto write_u64 = [&](std::size_t offset, std::uint64_t value) {
      write_u32(offset, static_cast<std::uint32_t>(value));
      write_u32(offset + 4, static_cast<std::uint32_t>(value >> 32));
    };

    encoded[0] = SBS_TELEMETRY_VERSION;
    encoded[1] = status;
    write_u16(2, state.request_id);
    write_u32(4, snapshot.generation);
    write_u32(8, snapshot.sequence);
    write_u32(12, snapshot.valid_fields | performance.valid_fields);
    write_u32(16, snapshot.runtime_flags);
    write_u16(20, snapshot.depth_width);
    write_u16(22, snapshot.depth_height);
    encoded[24] = snapshot.zero_plane_mode;
    // encoded[25..27] remain reserved zero.
    write_f32(28, snapshot.pop_floor);
    write_f32(32, snapshot.pop_ceiling);
    write_f32(36, snapshot.effective_pop);
    write_f32(40, snapshot.edge_fraction);
    // encoded[44..47] stays zero with valid bit 3 clear; protocol 13 clients parse a float there.
    write_f32(48, snapshot.zero_anchor_shift_px);
    write_f32(52, snapshot.subject_depth);
    write_f32(56, snapshot.valid_depth_fraction);
    write_f32(60, snapshot.effective_range_width);
    // encoded[64..75] stay zero with valid bits 8 and 9 clear; protocol 13 clients parse three
    // uint32 words there.
    write_u32(76, snapshot.empty_raw_count);
    write_u32(80, snapshot.collapsed_raw_count);
    write_u32(84, snapshot.sampled_frame_id);
    write_u32(88, performance.outcome_epoch);
    write_u32(92, performance.outcome_sequence);
    write_u32(96, performance.outcome_host_ms);
    write_u64(104, performance.outcomes.infer);
    write_u64(112, performance.outcomes.reuse);
    write_u64(120, performance.outcomes.invalid);
    write_u32(128, performance.output_host_ms);
    write_u32(132, performance.warped);
    write_u32(136, performance.repeated);
    write_u32(140, performance.flat);
    for (std::size_t i = 0; i < performance.stages.size(); ++i) {
      write_f32(144 + i * 12, performance.stages[i].mean_ms);
      write_u32(148 + i * 12, performance.stages[i].count);
      write_u32(152 + i * 12, performance.stages[i].latest_host_ms);
    }
    std::copy(encoded.begin(), encoded.end(), out);
    return true;
  }

  video::sbs_telemetry_snapshot_t unavailable_sbs_telemetry_snapshot() noexcept {
    video::sbs_telemetry_snapshot_t snapshot;
    snapshot.status = video::sbs_telemetry_sample_status_e::unavailable;
    return snapshot;
  }
}  // namespace stream

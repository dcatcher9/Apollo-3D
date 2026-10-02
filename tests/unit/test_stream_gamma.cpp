#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <winsock2.h>

// Keep Boost.Asio/Winsock2 ahead of rtsp.h's Windows headers.
// clang-format off
#include <src/stream.h>
#include <src/nvhttp.h>
// clang-format on

#include "../tests_common.h"

namespace {
  std::string gamma_request(std::uint8_t mode = 2, std::uint8_t version = 1) {
    const std::array<std::uint8_t, 8> bytes {version, mode, 0, 0, 0x78, 0x56, 0x34, 0x12};
    return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
  }

  video::stream_gamma_ack_t gamma_ack() {
    return {
      video::stream_gamma_status_e::applied,
      video::stream_gamma_mode_e::gamma_2_4,
      0x12345678,
      {video::stream_gamma_mode_e::gamma_2_4, 0x11223344, 203.0f},
    };
  }
}  // namespace

TEST(StreamGammaProtocolTest, RequestHasAnExactVersionedLittleEndianContract) {
  for (std::uint8_t mode = 0; mode <= 2; ++mode) {
    video::stream_gamma_request_t request;
    EXPECT_EQ(stream::decode_stream_gamma_request_payload(gamma_request(mode), request), stream::stream_gamma_request_decode_e::ok);
    EXPECT_EQ(static_cast<std::uint8_t>(request.mode), mode);
    EXPECT_EQ(request.request_id, 0x12345678u);
  }
  video::stream_gamma_request_t request;
  EXPECT_EQ(stream::decode_stream_gamma_request_payload(gamma_request(2, 2), request), stream::stream_gamma_request_decode_e::unsupported_version);
  EXPECT_EQ(request.request_id, 0x12345678u);
}

TEST(StreamGammaProtocolTest, MalformedRequestsDoNotInventCorrelation) {
  for (std::size_t size : {0u, 7u, 9u, 64u}) {
    video::stream_gamma_request_t request;
    EXPECT_EQ(stream::decode_stream_gamma_request_payload(std::string(size, '\0'), request), stream::stream_gamma_request_decode_e::invalid);
    EXPECT_EQ(request.request_id, 0u);
  }
  video::stream_gamma_request_t request;
  EXPECT_EQ(stream::decode_stream_gamma_request_payload(gamma_request(3), request), stream::stream_gamma_request_decode_e::invalid);
  EXPECT_EQ(request.request_id, 0u);
  auto payload = gamma_request();
  std::fill(payload.begin() + 4, payload.end(), '\0');
  EXPECT_EQ(stream::decode_stream_gamma_request_payload(payload, request), stream::stream_gamma_request_decode_e::invalid);
  EXPECT_EQ(request.request_id, 0u);
  payload = gamma_request();
  payload[3] = 1;
  EXPECT_EQ(stream::decode_stream_gamma_request_payload(payload, request), stream::stream_gamma_request_decode_e::invalid);
  EXPECT_EQ(request.request_id, 0x12345678u);  // Exact body permits a correlated invalid-flags refusal.
}

TEST(StreamGammaProtocolTest, AppliedAckHasAnExactLayoutAndProvenState) {
  std::uint8_t body[stream::STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_stream_gamma_ack_payload(gamma_ack(), body));
  const std::array<std::uint8_t, 16> expected {
    1,
    0,
    2,
    2,
    0x78,
    0x56,
    0x34,
    0x12,
    0x44,
    0x33,
    0x22,
    0x11,
    0,
    0,
    0x4b,
    0x43,
  };
  EXPECT_TRUE(std::equal(std::begin(body), std::end(body), expected.begin()));
  auto startup = gamma_ack();
  startup.request_id = 0;
  EXPECT_TRUE(stream::encode_stream_gamma_ack_payload(startup, body));
}

TEST(StreamGammaProtocolTest, RefusedRequestsEchoTheActualAppliedMode) {
  auto ack = gamma_ack();
  ack.status = video::stream_gamma_status_e::rejected_unsupported;
  ack.applied.mode = video::stream_gamma_mode_e::windows_default;
  std::uint8_t body[stream::STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_stream_gamma_ack_payload(ack, body));
  EXPECT_EQ(body[1], 2);
  EXPECT_EQ(body[2], 2);
  EXPECT_EQ(body[3], 0);
  ack.status = video::stream_gamma_status_e::applied;
  EXPECT_FALSE(stream::encode_stream_gamma_ack_payload(ack, body));
}

TEST(StreamGammaProtocolTest, UnprovenAndNonFiniteAcksLeaveTheDestinationUntouched) {
  std::uint8_t body[stream::STREAM_GAMMA_ACK_PAYLOAD_SIZE];
  std::fill(std::begin(body), std::end(body), 0x5a);
  auto ack = gamma_ack();
  ack.applied.generation = 0;
  EXPECT_FALSE(stream::encode_stream_gamma_ack_payload(ack, body));
  for (float white : {0.0f, 39.0f, 1001.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    ack = gamma_ack();
    ack.applied.white_nits = white;
    EXPECT_FALSE(stream::encode_stream_gamma_ack_payload(ack, body));
  }
  EXPECT_TRUE(std::all_of(std::begin(body), std::end(body), [](auto byte) {
    return byte == 0x5a;
  }));
}

TEST(StreamGammaProtocolTest, StateIsUnprovenUntilEncoderPublicationAndIsSessionLocal) {
  video::stream_gamma_publisher_t first;
  video::stream_gamma_publisher_t second;
  EXPECT_EQ(first.current().generation, 0u);
  const auto applied = first.publish(video::stream_gamma_mode_e::gamma_2_4, 240.0f);
  EXPECT_EQ(applied.generation, 1u);
  EXPECT_EQ(first.current().mode, video::stream_gamma_mode_e::gamma_2_4);
  EXPECT_EQ(first.current().white_nits, 240.0f);
  EXPECT_EQ(second.current().mode, video::stream_gamma_mode_e::windows_default);
  EXPECT_EQ(second.current().generation, 0u);
  EXPECT_EQ(first.publish(video::stream_gamma_mode_e::windows_default, 203.0f).generation, 2u);
}

TEST(StreamGammaProtocolTest, RequestedGammaSurvivesSdrBypassAndRestoresOnHdr) {
  using mode = video::stream_gamma_mode_e;
  video::stream_gamma_publisher_t state {mode::gamma_2_4};
  EXPECT_EQ(state.requested_mode(), mode::gamma_2_4);
  EXPECT_EQ(state.current().generation, 0u);  // A negotiated launch choice is not encoder proof.
  const auto first_hdr = state.publish(state.requested_mode(), 240.0f);
  EXPECT_EQ(first_hdr.mode, mode::gamma_2_4);

  // A same-connection display rebuild into SDR bypasses correction but retains the choice.
  const auto sdr = state.publish(mode::windows_default, 203.0f);
  EXPECT_EQ(sdr.generation, first_hdr.generation + 1);
  video::stream_gamma_ack_t ack {
    video::stream_gamma_status_e::rejected_unsupported, state.requested_mode(), 0, sdr,
  };
  std::uint8_t body[stream::STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_stream_gamma_ack_payload(ack, body));
  EXPECT_EQ(body[1], 2);
  EXPECT_EQ(body[2], 2);
  EXPECT_EQ(body[3], 0);

  // encode_run selects requested_mode() for its replacement encoder, not current().mode.
  const auto recovered_hdr = state.publish(state.requested_mode(), 240.0f);
  EXPECT_EQ(recovered_hdr.mode, mode::gamma_2_4);
  EXPECT_EQ(recovered_hdr.generation, sdr.generation + 1);
  ack = {video::stream_gamma_status_e::applied, state.requested_mode(), 0, recovered_hdr};
  ASSERT_TRUE(stream::encode_stream_gamma_ack_payload(ack, body));
  EXPECT_EQ(body[1], 0);
  EXPECT_EQ(body[2], 2);
  EXPECT_EQ(body[3], 2);
}

TEST(StreamGammaProtocolTest, ExplicitWindowsDefaultSupersedesDesiredGammaDuringSdrBypass) {
  using mode = video::stream_gamma_mode_e;
  video::stream_gamma_publisher_t state {mode::gamma_2_4};
  const auto sdr = state.publish(mode::windows_default, 203.0f);
  ASSERT_TRUE(state.set_requested_mode(mode::windows_default));
  EXPECT_EQ(state.current().generation, sdr.generation);
  EXPECT_EQ(state.requested_mode(), mode::windows_default);

  // Returning to HDR must not resurrect the older gamma choice after the explicit reset.
  const auto hdr = state.publish(state.requested_mode(), 240.0f);
  EXPECT_EQ(hdr.mode, mode::windows_default);
  EXPECT_EQ(hdr.generation, sdr.generation + 1);
  video::stream_gamma_publisher_t separate_session {mode::gamma_2_2};
  EXPECT_EQ(separate_session.requested_mode(), mode::gamma_2_2);
  EXPECT_EQ(separate_session.current().generation, 0u);
}

TEST(StreamGammaProtocolTest, LiveDesiredChoiceDoesNotInventAppliedProof) {
  using mode = video::stream_gamma_mode_e;
  video::stream_gamma_publisher_t state {mode::gamma_2_4};
  const auto proven = state.publish(mode::gamma_2_4, 240.0f);
  ASSERT_TRUE(state.set_requested_mode(mode::gamma_2_2));
  EXPECT_EQ(state.requested_mode(), mode::gamma_2_2);
  EXPECT_EQ(state.current().mode, proven.mode);
  EXPECT_EQ(state.current().generation, proven.generation);

  // A failed live setter reports the existing proof but the valid choice can be retried.
  video::stream_gamma_ack_t ack {
    video::stream_gamma_status_e::failed, state.requested_mode(), 7, state.current(),
  };
  std::uint8_t body[stream::STREAM_GAMMA_ACK_PAYLOAD_SIZE] {};
  ASSERT_TRUE(stream::encode_stream_gamma_ack_payload(ack, body));
  EXPECT_EQ(body[1], 3);
  EXPECT_EQ(body[2], 1);
  EXPECT_EQ(body[3], 2);
  EXPECT_FALSE(state.set_requested_mode(static_cast<mode>(3)));
  EXPECT_EQ(state.requested_mode(), mode::gamma_2_2);
}

TEST(StreamGammaProtocolTest, LaunchSelectionRejectsUnknownAndMalformedValues) {
  for (int mode = 0; mode <= 2; ++mode) {
    EXPECT_EQ(nvhttp::parse_launch_int(nvhttp::launch_int_field::stream_gamma, std::to_string(mode)), mode);
  }
  for (const auto *invalid : {"-1", "3", "2.4", "", "2147483648", "2garbage"}) {
    EXPECT_FALSE(nvhttp::parse_launch_int(nvhttp::launch_int_field::stream_gamma, invalid));
  }
  EXPECT_EQ(rtsp_stream::launch_session_t {}.stream_gamma, video::stream_gamma_mode_e::windows_default);
  EXPECT_EQ(video::config_t {}.stream_gamma, video::stream_gamma_mode_e::windows_default);
}

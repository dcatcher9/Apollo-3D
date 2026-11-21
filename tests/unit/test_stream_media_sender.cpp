#include "src/globals.h"
#include "src/stream_media_test.h"

#include <algorithm>
#include <array>
#include <boost/endian/conversion.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <openssl/evp.h>
#include <optional>
#include <rs.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <moonlight-common-c/src/Limelight-internal.h>
}

namespace {
  using namespace std::chrono_literals;
  namespace media = stream::testing;
  using lane_e = media::media_lane_e;
  using bytes_t = std::vector<std::uint8_t>;
  using cipher_context_t = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
  constexpr std::size_t video_header_size = sizeof(RTP_PACKET) + 4 + sizeof(NV_VIDEO_PACKET);
  constexpr std::size_t short_frame_header_size = 8;

  bytes_t authored_bytes(std::size_t count, unsigned seed) {
    bytes_t result(count);
    for (std::size_t index = 0; index < count; ++index) {
      result[index] = static_cast<std::uint8_t>(seed + index * 13);
    }
    return result;
  }

  std::string hex(std::span<const std::uint8_t> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (auto byte : bytes) {
      result.push_back(digits[byte >> 4]);
      result.push_back(digits[byte & 15]);
    }
    return result;
  }

  std::uint16_t little_u16(const std::uint8_t *bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(bytes[1]) << 8;
  }

  struct decoded_video_t {
    ENC_VIDEO_HEADER encrypted {};
    RTP_PACKET rtp {};
    NV_VIDEO_PACKET video {};
    bytes_t plaintext;
  };

  // Decode the actual client wire format with OpenSSL, independently of the host cipher wrapper.
  std::optional<decoded_video_t> decrypt_video(const bytes_t &datagram, const media::media_sender_config_t &config) {
    if (datagram.size() < sizeof(ENC_VIDEO_HEADER) + video_header_size) {
      return std::nullopt;
    }
    decoded_video_t result;
    std::memcpy(&result.encrypted, datagram.data(), sizeof(result.encrypted));
    cipher_context_t cipher {EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free};
    if (!cipher || EVP_DecryptInit_ex(cipher.get(), EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1 || EVP_CIPHER_CTX_ctrl(cipher.get(), EVP_CTRL_GCM_SET_IVLEN, sizeof(result.encrypted.iv), nullptr) != 1 || EVP_DecryptInit_ex(cipher.get(), nullptr, nullptr, config.key.data(), result.encrypted.iv) != 1) {
      return std::nullopt;
    }
    const auto ciphertext_size = datagram.size() - sizeof(ENC_VIDEO_HEADER);
    result.plaintext.resize(ciphertext_size + EVP_MAX_BLOCK_LENGTH);
    int written = 0, tail = 0;
    if (EVP_DecryptUpdate(cipher.get(), result.plaintext.data(), &written, datagram.data() + sizeof(ENC_VIDEO_HEADER), static_cast<int>(ciphertext_size)) != 1 || EVP_CIPHER_CTX_ctrl(cipher.get(), EVP_CTRL_GCM_SET_TAG, sizeof(result.encrypted.tag), result.encrypted.tag) != 1 || EVP_DecryptFinal_ex(cipher.get(), result.plaintext.data() + written, &tail) != 1) {
      return std::nullopt;
    }
    result.plaintext.resize(written + tail);
    if (result.plaintext.size() < video_header_size) {
      return std::nullopt;
    }
    std::memcpy(&result.rtp, result.plaintext.data(), sizeof(result.rtp));
    std::memcpy(&result.video, result.plaintext.data() + sizeof(RTP_PACKET) + 4, sizeof(result.video));
    return result;
  }

  std::optional<bytes_t> decrypt_audio(
    std::span<const std::uint8_t> ciphertext,
    const media::media_sender_config_t &config,
    std::uint16_t sequence
  ) {
    std::array<std::uint8_t, 16> iv {};
    const auto identifier = config.audio_key_id + sequence;
    for (std::size_t index = 0; index < 4; ++index) {
      iv[index] = static_cast<std::uint8_t>(identifier >> (24 - index * 8));
    }
    cipher_context_t cipher {EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free};
    if (!cipher || EVP_DecryptInit_ex(cipher.get(), EVP_aes_128_cbc(), nullptr, config.key.data(), iv.data()) != 1) {
      return std::nullopt;
    }
    bytes_t plaintext(ciphertext.size() + EVP_MAX_BLOCK_LENGTH);
    int written = 0, tail = 0;
    if (EVP_DecryptUpdate(cipher.get(), plaintext.data(), &written, ciphertext.data(), static_cast<int>(ciphertext.size())) != 1 || EVP_DecryptFinal_ex(cipher.get(), plaintext.data() + written, &tail) != 1) {
      return std::nullopt;
    }
    plaintext.resize(written + tail);
    return plaintext;
  }

  void expect_no_worker_error(const media::media_sender_state_t &state) {
    EXPECT_TRUE(state.video_error.empty()) << state.video_error;
    EXPECT_TRUE(state.audio_error.empty()) << state.audio_error;
  }

  void expect_video_nonce(const decoded_video_t &packet, std::uint64_t counter) {
    for (std::size_t index = 0; index < sizeof(counter); ++index) {
      EXPECT_EQ(packet.encrypted.iv[index], static_cast<std::uint8_t>(counter >> (index * 8)));
    }
    EXPECT_EQ(packet.encrypted.iv[8], 0);
    EXPECT_EQ(packet.encrypted.iv[9], 0);
    EXPECT_EQ(packet.encrypted.iv[10], 0);
    EXPECT_EQ(packet.encrypted.iv[11], 'V');
  }
}  // namespace

TEST(StreamMediaSenderTest, VideoAuthenticatesPayloadSourceIdentitySequenceNonceAndRtpEpoch) {
  media::media_sender_config_t config;
  config.video_min_fec_packets = 0;
  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.wait_ready(lane_e::audio, 3s));

  const std::array<bool, 5> idr {true, false, false, false, true};
  const std::array<bool, 5> retained {false, true, false, true, true};
  const std::array<bool, 5> invalidated {false, false, false, true, false};
  const std::array<std::uint16_t, 5> source_tokens {1, 1, 2, 3, 4};
  const std::array<std::uint8_t, 5> frame_types {2, 1, 1, 5, 2};
  for (std::size_t index = 0; index < idr.size(); ++index) {
    SCOPED_TRACE(index);
    const auto payload = authored_bytes(91, 37 + static_cast<unsigned>(index));
    const auto frame_index = static_cast<std::int64_t>(7 + index);
    ASSERT_TRUE(sender.enqueue_video({payload, frame_index, idr[index], retained[index], invalidated[index], 1s + index * 10ms}));
    const auto wire = sender.receive(lane_e::video, 1, 3s);
    ASSERT_EQ(wire.size(), 1u);
    const auto decoded = decrypt_video(wire.front(), config);
    ASSERT_TRUE(decoded.has_value());
    ASSERT_EQ(decoded->plaintext.size(), static_cast<std::size_t>(config.video_packet_size + MAX_RTP_HEADER_SIZE));
    EXPECT_EQ(decoded->encrypted.frameNumber, frame_index);
    expect_video_nonce(*decoded, config.video_iv_counter + index);
    EXPECT_EQ(decoded->rtp.header, 0x80 | FLAG_EXTENSION);
    EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.sequenceNumber), config.video_sequence + index);
    EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.timestamp), 90'000u + index * 900u);
    EXPECT_EQ(decoded->video.frameIndex, frame_index);
    EXPECT_EQ(decoded->video.streamPacketIndex, (config.video_sequence + index) << 8);
    EXPECT_EQ(decoded->video.flags, FLAG_CONTAINS_PIC_DATA | FLAG_SOF | FLAG_EOF);
    EXPECT_EQ(decoded->video.fecInfo >> 22, 1u);
    const auto *header = decoded->plaintext.data() + video_header_size;
    EXPECT_EQ(header[0], 1);
    EXPECT_EQ(little_u16(header + 1), 0);
    EXPECT_EQ(header[3], frame_types[index]);
    EXPECT_EQ(little_u16(header + 4), payload.size() + short_frame_header_size);
    EXPECT_EQ(little_u16(header + 6), source_tokens[index]);
    EXPECT_TRUE(std::equal(payload.begin(), payload.end(), header + short_frame_header_size));
    auto corrupted = wire.front();
    corrupted[offsetof(ENC_VIDEO_HEADER, tag)] ^= 1;
    EXPECT_FALSE(decrypt_video(corrupted, config).has_value());
    ASSERT_TRUE(sender.wait_processed(lane_e::video, index + 1, 3s));
  }
  sender.stop();
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence + idr.size());
  EXPECT_EQ(state.video_iv_counter, config.video_iv_counter + idr.size());
  expect_no_worker_error(state);
}

TEST(StreamMediaSenderTest, VideoFecPacketsAuthenticateAndRecoverOneMissingPayloadShard) {
  media::media_sender_config_t config;
  config.video_min_fec_packets = 0;
  config.video_fec_percentage = 20;
  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  const auto payload = authored_bytes(2500, 61);
  ASSERT_TRUE(sender.enqueue_video({payload, 42}));
  const auto wire = sender.receive(lane_e::video, 4, 3s);
  ASSERT_EQ(wire.size(), 4u);
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 1, 3s));
  sender.stop();

  std::array<bytes_t, 4> shards;
  for (const auto &datagram : wire) {
    const auto decoded = decrypt_video(datagram, config);
    ASSERT_TRUE(decoded.has_value());
    const auto shard_index = (decoded->video.fecInfo >> 12) & 0x3ff;
    ASSERT_LT(shard_index, shards.size());
    ASSERT_TRUE(shards[shard_index].empty());
    EXPECT_EQ(decoded->video.fecInfo >> 22, 3u);
    EXPECT_EQ((decoded->video.fecInfo >> 4) & 0xff, 20u);
    EXPECT_EQ(decoded->video.frameIndex, 42u);
    EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.sequenceNumber), config.video_sequence + shard_index);
    expect_video_nonce(*decoded, config.video_iv_counter + shard_index);
    shards[shard_index] = decoded->plaintext;
  }
  bytes_t joined;
  for (std::size_t index = 0; index < 3; ++index) {
    ASSERT_GT(shards[index].size(), video_header_size);
    joined.insert(joined.end(), shards[index].begin() + video_header_size, shards[index].end());
  }
  ASSERT_GE(joined.size(), payload.size() + short_frame_header_size);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), joined.begin() + short_frame_header_size));
  const auto missing = shards[1];
  std::fill(shards[1].begin(), shards[1].end(), 0);
  std::array<std::uint8_t *, 4> pointers;
  for (std::size_t index = 0; index < shards.size(); ++index) {
    ASSERT_EQ(shards[index].size(), missing.size());
    pointers[index] = shards[index].data();
  }
  std::array<std::uint8_t, 4> missing_marks {0, 1, 0, 0};
  std::unique_ptr<reed_solomon, decltype(&reed_solomon_release)> decoder {reed_solomon_new(3, 1), reed_solomon_release};
  ASSERT_NE(decoder, nullptr);
  ASSERT_EQ(reed_solomon_decode(decoder.get(), pointers.data(), missing_marks.data(), pointers.size(), missing.size()), 0);
  // RTP/NV headers are filled after parity generation. Client recovery authenticates picture
  // payload bytes; those mutable transport headers are not a parity oracle.
  EXPECT_TRUE(std::equal(missing.begin() + video_header_size, missing.end(), shards[1].begin() + video_header_size));
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence + shards.size());
  EXPECT_EQ(state.video_iv_counter, config.video_iv_counter + shards.size());
  expect_no_worker_error(state);
}

TEST(StreamMediaSenderTest, VideoSplitsFecBlocksAndClearsTheTailOfReusedDataShards) {
  media::media_sender_config_t config;
  config.video_packet_size = 201;
  config.video_fec_percentage = 1;
  config.video_min_fec_packets = 2;
  config.video_sequence = 65'520;
  const std::size_t picture_bytes = config.video_packet_size - sizeof(NV_VIDEO_PACKET);
  // At 1% FEC, 252 data shards fit in one block. The next shard forces a split.
  constexpr std::array<std::size_t, 2> data_per_block {127, 126};
  constexpr std::size_t total_data = data_per_block[0] + data_per_block[1];
  constexpr std::size_t total_wire = total_data + 4;
  const auto payload = authored_bytes(total_data * picture_bytes - short_frame_header_size, 61);

  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.enqueue_video({payload, 7, true, false, false, 1s}));
  const auto wire = sender.receive(lane_e::video, total_wire, 3s);
  ASSERT_EQ(wire.size(), total_wire);
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 1, 3s));

  bytes_t joined;
  std::size_t wire_index = 0;
  for (std::size_t block = 0; block < data_per_block.size(); ++block) {
    const auto data = data_per_block[block];
    for (std::size_t shard = 0; shard < data + 2; ++shard, ++wire_index) {
      const auto decoded = decrypt_video(wire[wire_index], config);
      ASSERT_TRUE(decoded.has_value());
      EXPECT_EQ(decoded->video.multiFecBlocks, (block << 4) | 0x40);
      EXPECT_EQ(decoded->video.fecInfo, (data << 22) | (shard << 12) | 0x10);
      EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.sequenceNumber), static_cast<std::uint16_t>(config.video_sequence + wire_index));
      EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.timestamp), 90'000u);
      expect_video_nonce(*decoded, config.video_iv_counter + wire_index);
      if (shard < data) {
        const auto flags = FLAG_CONTAINS_PIC_DATA | (shard == 0 ? FLAG_SOF : 0) | (shard == data - 1 ? FLAG_EOF : 0);
        EXPECT_EQ(decoded->video.flags, flags);
        EXPECT_EQ(decoded->video.multiFecFlags, 0x10);
        EXPECT_EQ(decoded->video.streamPacketIndex, (config.video_sequence + wire_index) << 8);
        joined.insert(joined.end(), decoded->plaintext.begin() + video_header_size, decoded->plaintext.end());
      }
    }
  }
  ASSERT_EQ(joined.size(), total_data * picture_bytes);
  EXPECT_EQ(little_u16(joined.data() + 4), picture_bytes);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), joined.begin() + short_frame_header_size));

  // This short frame reuses the previous workspace and raises parity to the negotiated minimum.
  const auto short_payload = authored_bytes(17, 93);
  ASSERT_TRUE(sender.enqueue_video({short_payload, 8, false}));
  const auto short_wire = sender.receive(lane_e::video, 3, 3s);
  ASSERT_EQ(short_wire.size(), 3u);
  const auto decoded = decrypt_video(short_wire.front(), config);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->video.multiFecBlocks, 0);
  EXPECT_EQ(decoded->video.fecInfo, (1u << 22) | (200u << 4));
  const auto *header = decoded->plaintext.data() + video_header_size;
  const auto used = short_frame_header_size + short_payload.size();
  EXPECT_EQ(little_u16(header + 4), used);
  EXPECT_TRUE(std::equal(short_payload.begin(), short_payload.end(), header + short_frame_header_size));
  EXPECT_TRUE(std::all_of(header + used, header + picture_bytes, [](auto byte) {
    return byte == 0;
  }));
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 2, 3s));
  sender.stop();
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence + total_wire + 3);
  EXPECT_EQ(state.video_iv_counter, config.video_iv_counter + total_wire + 3);
  expect_no_worker_error(state);
}

TEST(StreamMediaSenderTest, VideoSendsDataOnlyBlocksAboveReedSolomonCapacity) {
  media::media_sender_config_t config;
  config.video_packet_size = 200;
  config.video_fec_percentage = 20;
  constexpr std::array<std::size_t, 4> block_sizes {257, 257, 257, 254};
  constexpr std::size_t packet_count = 1025;
  const std::size_t picture_bytes = config.video_packet_size - sizeof(NV_VIDEO_PACKET);
  const auto payload = authored_bytes(packet_count * picture_bytes - short_frame_header_size, 41);

  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.enqueue_video({payload, 11}));
  const auto wire = sender.receive(lane_e::video, packet_count, 3s);
  ASSERT_EQ(wire.size(), packet_count);
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 1, 3s));
  sender.stop();

  bytes_t joined;
  std::size_t wire_index = 0;
  // The planned parity would require five blocks. The sender uses four data-only blocks,
  // each of which may have more shards than the Reed–Solomon pointer table can hold.
  for (std::size_t block = 0; block < block_sizes.size(); ++block) {
    for (std::size_t shard = 0; shard < block_sizes[block]; ++shard, ++wire_index) {
      const auto decoded = decrypt_video(wire[wire_index], config);
      ASSERT_TRUE(decoded.has_value());
      EXPECT_EQ(decoded->video.multiFecBlocks, (block << 4) | 0xc0);
      EXPECT_EQ(decoded->video.fecInfo, (block_sizes[block] << 22) | (shard << 12));
      EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.sequenceNumber), config.video_sequence + wire_index);
      expect_video_nonce(*decoded, config.video_iv_counter + wire_index);
      joined.insert(joined.end(), decoded->plaintext.begin() + video_header_size, decoded->plaintext.end());
    }
  }
  ASSERT_EQ(joined.size(), packet_count * picture_bytes);
  EXPECT_TRUE(std::equal(payload.begin(), payload.end(), joined.begin() + short_frame_header_size));
  expect_no_worker_error(sender.state_after_stop());
}

TEST(StreamMediaSenderTest, OversizedFrameRequiresRecoveryIdrWithoutConsumingSequenceOrNonce) {
  media::media_sender_config_t config;
  config.video_packet_size = 200;
  const std::size_t picture_bytes = config.video_packet_size - sizeof(NV_VIDEO_PACKET);
  const auto oversized = authored_bytes(4 * 1023 * picture_bytes - short_frame_header_size + 1, 23);
  const auto payload = authored_bytes(17, 93);

  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.enqueue_video({oversized, 7}));
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 1, 3s));
  ASSERT_TRUE(sender.enqueue_video({payload, 8, false}));
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 2, 3s));
  ASSERT_TRUE(sender.enqueue_video({payload, 9}));
  const auto wire = sender.receive(lane_e::video, 1, 3s);
  ASSERT_EQ(wire.size(), 1u);
  const auto decoded = decrypt_video(wire.front(), config);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->video.frameIndex, 9u);
  EXPECT_EQ(boost::endian::big_to_native(decoded->rtp.sequenceNumber), config.video_sequence);
  expect_video_nonce(*decoded, config.video_iv_counter);
  ASSERT_TRUE(sender.wait_processed(lane_e::video, 3, 3s));
  sender.stop();
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence + 1);
  EXPECT_EQ(state.video_iv_counter, config.video_iv_counter + 1);
  expect_no_worker_error(state);
}

TEST(StreamMediaSenderTest, VideoEncryptionFailureDropsTheBatchAndWaitsForRecoveryIdr) {
  // The first size fits all three shards in one batch. The second hits the 64 KiB
  // batch limit after one shard, so an earlier authenticated datagram can arrive.
  for (const auto &[packet_size, sent_before_failure] : {std::pair {1024, 0u}, std::pair {60000, 1u}}) {
    SCOPED_TRACE(packet_size);
    media::media_sender_config_t config;
    config.video_packet_size = packet_size;
    config.video_min_fec_packets = 0;
    config.video_sequence = 65534;
    config.video_fail_encryption_at_iv = config.video_iv_counter + 2;
    const std::size_t picture_bytes = packet_size - sizeof(NV_VIDEO_PACKET);
    const auto payload = authored_bytes(17, 93);

    media::media_sender_fixture_t sender(config);
    sender.start();
    ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
    ASSERT_TRUE(sender.enqueue_video({payload, 7}));
    const auto warm = sender.receive(lane_e::video, 1, 3s);
    ASSERT_EQ(warm.size(), 1u);
    ASSERT_TRUE(decrypt_video(warm.front(), config).has_value());

    ASSERT_TRUE(sender.enqueue_video({authored_bytes(3 * picture_bytes - short_frame_header_size, 23), 8, false}));
    ASSERT_TRUE(sender.wait_processed(lane_e::video, 2, 3s));
    ASSERT_TRUE(sender.enqueue_video({payload, 9, false}));
    ASSERT_TRUE(sender.wait_processed(lane_e::video, 3, 3s));
    ASSERT_TRUE(sender.enqueue_video({payload, 10}));
    ASSERT_TRUE(sender.wait_processed(lane_e::video, 4, 3s));

    const auto wire = sender.receive(lane_e::video, sent_before_failure + 1, 3s);
    ASSERT_EQ(wire.size(), sent_before_failure + 1);
    if (sent_before_failure) {
      const auto prior = decrypt_video(wire.front(), config);
      ASSERT_TRUE(prior.has_value());
      EXPECT_EQ(prior->video.frameIndex, 8u);
      EXPECT_EQ(boost::endian::big_to_native(prior->rtp.sequenceNumber), config.video_sequence + 1);
      expect_video_nonce(*prior, config.video_iv_counter + 1);
    }
    const auto recovery = decrypt_video(wire.back(), config);
    ASSERT_TRUE(recovery.has_value());
    EXPECT_EQ(recovery->video.frameIndex, 10u);
    EXPECT_EQ(boost::endian::big_to_native(recovery->rtp.sequenceNumber), static_cast<std::uint16_t>(config.video_sequence + 4));
    expect_video_nonce(*recovery, config.video_iv_counter + 3);
    EXPECT_TRUE(sender.receive(lane_e::video, 1, 0ms).empty());

    sender.stop();
    const auto state = sender.state_after_stop();
    EXPECT_EQ(state.video_sequence, config.video_sequence + 5);
    EXPECT_EQ(state.video_iv_counter, config.video_iv_counter + 4);
    expect_no_worker_error(state);
  }
}

TEST(StreamMediaSenderTest, AudioDecryptsFourPayloadsAndEmitsTwoFecShardsWithExactHeaders) {
  media::media_sender_config_t config;
  media::media_sender_fixture_t sender(config);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::audio, 3s));
  std::array<bytes_t, 4> expected;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = authored_bytes(24, 17 + static_cast<unsigned>(index) * 41);
    ASSERT_TRUE(sender.enqueue_audio(expected[index]));
  }
  const auto wire = sender.receive(lane_e::audio, 6, 3s);
  ASSERT_EQ(wire.size(), 6u);
  ASSERT_TRUE(sender.wait_processed(lane_e::audio, expected.size(), 3s));
  sender.stop();
  std::array<bool, 4> seen_data {};
  std::array<bytes_t, 2> parity;
  for (const auto &datagram : wire) {
    ASSERT_GE(datagram.size(), sizeof(RTP_PACKET));
    RTP_PACKET rtp;
    std::memcpy(&rtp, datagram.data(), sizeof(rtp));
    EXPECT_EQ(rtp.header, 0x80);
    const auto sequence = boost::endian::big_to_native(rtp.sequenceNumber);
    if (rtp.packetType == 97) {
      ASSERT_GE(sequence, config.audio_sequence);
      const auto index = sequence - config.audio_sequence;
      ASSERT_LT(index, expected.size());
      ASSERT_FALSE(seen_data[index]);
      seen_data[index] = true;
      EXPECT_EQ(boost::endian::big_to_native(rtp.timestamp), config.audio_timestamp + index * config.audio_packet_duration);
      const auto plaintext = decrypt_audio(std::span(datagram).subspan(sizeof(rtp)), config, sequence);
      ASSERT_TRUE(plaintext.has_value());
      EXPECT_EQ(*plaintext, expected[index]);
    } else {
      ASSERT_EQ(rtp.packetType, 127);
      ASSERT_EQ(datagram.size(), sizeof(RTP_PACKET) + sizeof(AUDIO_FEC_HEADER) + 32u);
      AUDIO_FEC_HEADER fec;
      std::memcpy(&fec, datagram.data() + sizeof(rtp), sizeof(fec));
      ASSERT_LT(fec.fecShardIndex, parity.size());
      ASSERT_TRUE(parity[fec.fecShardIndex].empty());
      EXPECT_EQ(sequence, config.audio_sequence + 4 + fec.fecShardIndex);
      EXPECT_EQ(fec.payloadType, 97);
      EXPECT_EQ(boost::endian::big_to_native(fec.baseSequenceNumber), config.audio_sequence);
      EXPECT_EQ(boost::endian::big_to_native(fec.baseTimestamp), config.audio_timestamp);
      EXPECT_EQ(fec.ssrc, 0u);
      parity[fec.fecShardIndex].assign(datagram.begin() + sizeof(rtp) + sizeof(fec), datagram.end());
    }
  }
  EXPECT_TRUE(std::all_of(seen_data.begin(), seen_data.end(), [](bool seen) {
    return seen;
  }));
  ASSERT_EQ(parity[0].size(), 32u);
  ASSERT_EQ(parity[1].size(), 32u);
  EXPECT_NE(parity[0], parity[1]);
  // Fixed ciphertext parity recorded by the qualified actual-sender control.
  EXPECT_EQ(hex(parity[0]), "a81e1329157662af7c3d95e8300e09b83cb59adb34c850e0fc811653f75f13d3");
  EXPECT_EQ(hex(parity[1]), "744b9e2f73226936b027a0f5d1f89fd6c12d5c6197b14bc54f06c75fdb2a5618");
  RecordProperty("audio_fec_0_hex", hex(parity[0]));
  RecordProperty("audio_fec_1_hex", hex(parity[1]));
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.audio_sequence, config.audio_sequence + expected.size());
  EXPECT_EQ(state.audio_timestamp, config.audio_timestamp + expected.size() * config.audio_packet_duration);
  expect_no_worker_error(state);
}

namespace {
  class StreamMediaSenderAdmissionTest: public testing::TestWithParam<std::pair<lane_e, bool>> {};
}  // namespace

TEST_P(StreamMediaSenderAdmissionTest, RejectsPoppedPacketWhenStopOrDeactivationWinsAdmission) {
  const auto [lane, shutdown] = GetParam();
  media::media_sender_config_t config;
  config.video_min_fec_packets = 0;
  media::media_sender_fixture_t sender(config);
  sender.gate_next_admission(lane);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane, 3s));
  if (lane == lane_e::video) {
    ASSERT_TRUE(sender.enqueue_video({authored_bytes(24, 31), 1}));
  } else {
    ASSERT_TRUE(sender.enqueue_audio(authored_bytes(24, 31)));
  }
  ASSERT_TRUE(sender.wait_admission(lane, 3s));
  if (shutdown) {
    sender.request_shutdown(lane);
  } else {
    sender.deactivate(lane);
  }
  sender.release_admission(lane);
  // Prove the popped packet completed its admission check BEFORE stop() can set shutdown.
  ASSERT_TRUE(sender.wait_processed(lane, 1, 3s));
  sender.stop();
  EXPECT_TRUE(sender.receive(lane, 1, 30ms).empty());
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence);
  EXPECT_EQ(state.video_iv_counter, config.video_iv_counter);
  EXPECT_EQ(state.audio_sequence, config.audio_sequence);
  EXPECT_EQ(state.audio_timestamp, config.audio_timestamp);
  expect_no_worker_error(state);
}

INSTANTIATE_TEST_SUITE_P(
  BothLanes,
  StreamMediaSenderAdmissionTest,
  testing::Values(
    std::pair {lane_e::video, false},
    std::pair {lane_e::video, true},
    std::pair {lane_e::audio, false},
    std::pair {lane_e::audio, true}
  ),
  [](const auto &info) {
    return std::string(info.param.first == lane_e::video ? "Video" : "Audio") +
           (info.param.second ? "Shutdown" : "Deactivated");
  }
);

TEST(StreamMediaSenderTest, EmptyQueuesWakeForStopAndReleaseOwnedChannelsAndPoolOnClose) {
  media::media_sender_config_t config;
  media::media_sender_fixture_t sender(config);
  const auto video_channel = sender.weak_channel(lane_e::video);
  const auto audio_channel = sender.weak_channel(lane_e::audio);
  const auto pool = sender.weak_frame_pool();
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.wait_ready(lane_e::audio, 3s));
  sender.stop();
  sender.stop();
  const auto state = sender.state_after_stop();
  EXPECT_EQ(state.video_sequence, config.video_sequence);
  EXPECT_EQ(state.audio_sequence, config.audio_sequence);
  EXPECT_TRUE(state.video_shutdown);
  EXPECT_TRUE(state.audio_shutdown);
  expect_no_worker_error(state);
  EXPECT_FALSE(video_channel.expired());
  EXPECT_FALSE(audio_channel.expired());
  EXPECT_FALSE(pool.expired());
  sender.close();
  sender.close();
  EXPECT_TRUE(video_channel.expired());
  EXPECT_TRUE(audio_channel.expired());
  EXPECT_TRUE(pool.expired());
}

TEST(StreamMediaSenderTest, CloseReleasesQueuedPacketsAndTheirOriginatingFramePool) {
  media::media_sender_fixture_t sender;
  const auto video_channel = sender.weak_channel(lane_e::video);
  const auto audio_channel = sender.weak_channel(lane_e::audio);
  const auto pool = sender.weak_frame_pool();
  sender.gate_next_admission(lane_e::video);
  sender.gate_next_admission(lane_e::audio);
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.wait_ready(lane_e::audio, 3s));
  ASSERT_TRUE(sender.enqueue_video({authored_bytes(40, 1), 1}));
  ASSERT_TRUE(sender.enqueue_audio(authored_bytes(24, 1)));
  ASSERT_TRUE(sender.wait_admission(lane_e::video, 3s));
  ASSERT_TRUE(sender.wait_admission(lane_e::audio, 3s));
  ASSERT_TRUE(sender.enqueue_video({authored_bytes(40, 2), 2}));
  ASSERT_TRUE(sender.enqueue_audio(authored_bytes(24, 2)));
  sender.stop();  // Opens both gates, rejects admitted work and retains the two unconsumed packets.
  EXPECT_FALSE(pool.expired());
  EXPECT_FALSE(video_channel.expired());
  EXPECT_FALSE(audio_channel.expired());
  EXPECT_TRUE(sender.receive(lane_e::video, 1, 30ms).empty());
  EXPECT_TRUE(sender.receive(lane_e::audio, 1, 30ms).empty());
  expect_no_worker_error(sender.state_after_stop());
  sender.close();
  EXPECT_TRUE(pool.expired());
  EXPECT_TRUE(video_channel.expired());
  EXPECT_TRUE(audio_channel.expired());
}

TEST(SessionSenderOwnerTest, QueueEofJoinsAndReleasesQueuesWithoutRaisingShutdown) {
  ASSERT_TRUE(mail::man);
  const auto broadcast_shutdown = mail::man->event<bool>(mail::broadcast_shutdown);
  ASSERT_FALSE(broadcast_shutdown->peek());
  media::media_sender_fixture_t sender;
  const auto video_queue = sender.weak_queue(lane_e::video);
  const auto audio_queue = sender.weak_queue(lane_e::audio);
  const auto video_channel = sender.weak_channel(lane_e::video);
  const auto audio_channel = sender.weak_channel(lane_e::audio);
  ASSERT_FALSE(video_queue.expired());
  ASSERT_FALSE(audio_queue.expired());
  sender.start();
  ASSERT_TRUE(sender.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(sender.wait_ready(lane_e::audio, 3s));
  sender.finish_senders_without_shutdown();
  const auto state = sender.state_after_stop();
  EXPECT_FALSE(state.video_shutdown);
  EXPECT_FALSE(state.audio_shutdown);
  EXPECT_FALSE(broadcast_shutdown->peek());
  EXPECT_TRUE(video_queue.expired());
  EXPECT_TRUE(audio_queue.expired());
  // The facade and channel owners remain alive: queue retirement is the actual owner's helper.
  EXPECT_FALSE(video_channel.expired());
  EXPECT_FALSE(audio_channel.expired());
  expect_no_worker_error(state);
  sender.close();
  EXPECT_TRUE(video_channel.expired());
  EXPECT_TRUE(audio_channel.expired());
  EXPECT_FALSE(broadcast_shutdown->peek());
}

TEST(SessionSenderOwnerTest, SuccessorQueuesCannotConsumeInactivePredecessorPackets) {
  media::media_sender_fixture_t predecessor;
  predecessor.gate_next_admission(lane_e::video);
  predecessor.gate_next_admission(lane_e::audio);
  predecessor.start();
  ASSERT_TRUE(predecessor.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(predecessor.wait_ready(lane_e::audio, 3s));
  ASSERT_TRUE(predecessor.enqueue_video({authored_bytes(31, 11), 1}));
  ASSERT_TRUE(predecessor.enqueue_audio(authored_bytes(24, 11)));
  ASSERT_TRUE(predecessor.wait_admission(lane_e::video, 3s));
  ASSERT_TRUE(predecessor.wait_admission(lane_e::audio, 3s));
  ASSERT_TRUE(predecessor.enqueue_video({authored_bytes(31, 12), 2}));
  ASSERT_TRUE(predecessor.enqueue_audio(authored_bytes(24, 12)));
  predecessor.deactivate(lane_e::video);
  predecessor.deactivate(lane_e::audio);

  media::media_sender_config_t config;
  config.video_min_fec_packets = 0;
  config.video_sequence = 1700;
  config.video_iv_counter = 71;
  config.audio_sequence = 2400;
  config.audio_timestamp = 41'000;
  config.audio_key_id = 0x40506070;
  for (std::size_t index = 0; index < config.key.size(); ++index) {
    config.key[index] = static_cast<std::uint8_t>(0xA0 + index);
  }
  media::media_sender_fixture_t successor(config);
  for (const auto lane : {lane_e::video, lane_e::audio}) {
    const auto old_queue = predecessor.weak_queue(lane);
    const auto new_queue = successor.weak_queue(lane);
    EXPECT_TRUE(old_queue.owner_before(new_queue) || new_queue.owner_before(old_queue));
  }
  successor.start();
  ASSERT_TRUE(successor.wait_ready(lane_e::video, 3s));
  ASSERT_TRUE(successor.wait_ready(lane_e::audio, 3s));
  const auto video_payload = authored_bytes(53, 91);
  const auto audio_payload = authored_bytes(24, 92);
  ASSERT_TRUE(successor.enqueue_video({video_payload, 51, true, false, false, 2s}));
  ASSERT_TRUE(successor.enqueue_audio(audio_payload));
  const auto video_wire = successor.receive(lane_e::video, 1, 3s);
  const auto audio_wire = successor.receive(lane_e::audio, 1, 3s);
  ASSERT_EQ(video_wire.size(), 1u);
  ASSERT_EQ(audio_wire.size(), 1u);
  const auto decoded_video = decrypt_video(video_wire.front(), config);
  ASSERT_TRUE(decoded_video.has_value());
  EXPECT_EQ(decoded_video->encrypted.frameNumber, 51u);
  EXPECT_EQ(boost::endian::big_to_native(decoded_video->rtp.sequenceNumber), config.video_sequence);
  expect_video_nonce(*decoded_video, config.video_iv_counter);
  ASSERT_GE(decoded_video->plaintext.size(), video_header_size + short_frame_header_size + video_payload.size());
  EXPECT_TRUE(std::equal(video_payload.begin(), video_payload.end(), decoded_video->plaintext.begin() + video_header_size + short_frame_header_size));
  ASSERT_GE(audio_wire.front().size(), sizeof(RTP_PACKET));
  RTP_PACKET rtp;
  std::memcpy(&rtp, audio_wire.front().data(), sizeof(rtp));
  EXPECT_EQ(rtp.packetType, 97);
  EXPECT_EQ(boost::endian::big_to_native(rtp.sequenceNumber), config.audio_sequence);
  const auto decoded_audio = decrypt_audio(std::span(audio_wire.front()).subspan(sizeof(rtp)), config, config.audio_sequence);
  ASSERT_TRUE(decoded_audio.has_value());
  EXPECT_EQ(*decoded_audio, audio_payload);
  ASSERT_TRUE(successor.wait_processed(lane_e::video, 1, 3s));
  ASSERT_TRUE(successor.wait_processed(lane_e::audio, 1, 3s));

  predecessor.release_admission(lane_e::video);
  predecessor.release_admission(lane_e::audio);
  ASSERT_TRUE(predecessor.wait_processed(lane_e::video, 2, 3s));
  ASSERT_TRUE(predecessor.wait_processed(lane_e::audio, 2, 3s));
  predecessor.stop();
  successor.stop();
  EXPECT_TRUE(predecessor.receive(lane_e::video, 1, 30ms).empty());
  EXPECT_TRUE(predecessor.receive(lane_e::audio, 1, 30ms).empty());
  EXPECT_TRUE(successor.receive(lane_e::video, 1, 30ms).empty());
  EXPECT_TRUE(successor.receive(lane_e::audio, 1, 30ms).empty());
  const auto old_state = predecessor.state_after_stop();
  const auto new_state = successor.state_after_stop();
  EXPECT_EQ(old_state.video_sequence, media::media_sender_config_t {}.video_sequence);
  EXPECT_EQ(old_state.audio_sequence, media::media_sender_config_t {}.audio_sequence);
  EXPECT_EQ(new_state.video_sequence, config.video_sequence + 1);
  EXPECT_EQ(new_state.audio_sequence, config.audio_sequence + 1);
  expect_no_worker_error(old_state);
  expect_no_worker_error(new_state);
}

/** Host-side golden packets for the shared common-C authored haptics parser. */
#include "src/client_features.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>

TEST(ClientFeatures, AuthoredPcmHasExactSharedWireLayout) {
  const std::array<std::uint8_t, 8> samples {0x34, 0x12, 0xcc, 0xed, 0xff, 0x7f, 0x00, 0x80};
  const auto packet = client_features::encode_pcm({
    15,
    5,
    2,
    0x89abcdef,
    0x1020304050607080,
    samples,
  });
  ASSERT_TRUE(packet);
  const std::array<std::uint8_t, 40> expected {
    0x0a,
    0x55,
    0x24,
    0x00,  // control type, payload length
    0x01,
    0x05,
    0x1c,
    0x00,
    0x0f,
    0x00,
    0x02,
    0x00,  // version, flags, header, pad, frames
    0xef,
    0xcd,
    0xab,
    0x89,  // sequence
    0x80,
    0x70,
    0x60,
    0x50,
    0x40,
    0x30,
    0x20,
    0x10,  // presentation time
    0x80,
    0xbb,
    0x00,
    0x00,
    0x02,
    0x10,
    0x00,
    0x00,  // 48 kHz, stereo, S16, reserved
    0x34,
    0x12,
    0xcc,
    0xed,
    0xff,
    0x7f,
    0x00,
    0x80,
  };
  ASSERT_EQ(packet->size, expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), packet->bytes.begin()));
}

TEST(ClientFeatures, AuthoredPcmPreservesEmptyStreamBoundary) {
  const auto packet = client_features::encode_pcm({0, 2, 0, 0xffffffff, 0xffffffffffffffff, {}});
  ASSERT_TRUE(packet);
  EXPECT_EQ(packet->size, 32);
  EXPECT_EQ(packet->bytes[2], 28);
  EXPECT_EQ(packet->bytes[5], 2);
  EXPECT_EQ(packet->bytes[10], 0);
  EXPECT_EQ(packet->bytes[11], 0);
  for (unsigned index = 12; index < 24; ++index) {
    EXPECT_EQ(packet->bytes[index], 0xff);
  }
}

TEST(ClientFeatures, AuthoredPcmMaximumPacketCopiesEverySample) {
  std::array<std::uint8_t, 960> samples;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    samples[i] = static_cast<std::uint8_t>(i);
  }
  const auto packet = client_features::encode_pcm({1, 0, 240, 3, 4, samples});
  ASSERT_TRUE(packet);
  EXPECT_EQ(packet->size, 992);
  EXPECT_EQ(packet->bytes[2], 0xdc);
  EXPECT_EQ(packet->bytes[3], 3);
  EXPECT_TRUE(std::equal(samples.begin(), samples.end(), packet->bytes.begin() + 32));
}

TEST(ClientFeatures, AuthoredPcmRejectsInvalidControllerFlagsAndLengths) {
  const std::array<std::uint8_t, 4> sample {};
  EXPECT_FALSE(client_features::encode_pcm({16, 0, 1, 0, 0, sample}));
  EXPECT_FALSE(client_features::encode_pcm({0, 8, 1, 0, 0, sample}));
  EXPECT_FALSE(client_features::encode_pcm({0, 0x80, 1, 0, 0, sample}));
  EXPECT_FALSE(client_features::encode_pcm({0, 0, 241, 0, 0, sample}));
  EXPECT_FALSE(client_features::encode_pcm({0, 0, 0, 0, 0, sample}));
  EXPECT_FALSE(client_features::encode_pcm({0, 0, 1, 0, 0, {}}));
}

TEST(ClientFeatures, AuthoredFlagsDoNotOverlapSharedSbsCapabilities) {
  constexpr std::uint32_t existing_client_features = 0x1f;
  EXPECT_EQ(client_features::client_authored_pcm, 0x20);
  EXPECT_EQ(client_features::client_authored_ir_v2, 0x40);
  EXPECT_EQ(existing_client_features & client_features::client_authored_pcm, 0);
  EXPECT_EQ(existing_client_features & client_features::client_authored_ir_v2, 0);
  EXPECT_EQ(client_features::client_authored_pcm & client_features::client_authored_ir_v2, 0);
  EXPECT_EQ(client_features::encryption_microphone, 0x08);
}

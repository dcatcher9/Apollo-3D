/**
 * @file tests/unit/platform/test_windows_audio_pcm_staging.cpp
 * @brief FIFO and relocation regressions for the actual WASAPI PCM staging helper.
 */

#include "src/platform/windows/audio_pcm_staging.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
  using platf::audio::detail::pcm_staging_t;

  struct fifo_reference_t {
    std::size_t capacity;
    std::deque<float> samples;

    std::size_t append(const float *input, std::size_t count, bool silent, float gain) {
      const auto accepted = std::min(count, capacity - samples.size());
      for (std::size_t index = 0; index < accepted; ++index) {
        samples.push_back(silent ? 0.0f : gain == 1.0f ? input[index] :
                                                         input[index] * gain);
      }
      return accepted;
    }
  };

  std::vector<float> consecutive_samples(std::size_t count, std::size_t first = 1) {
    std::vector<float> samples(count);
    for (std::size_t index = 0; index < count; ++index) {
      samples[index] = static_cast<float>(first + index);
    }
    return samples;
  }

  std::size_t append_and_compare(pcm_staging_t &staging, fifo_reference_t &reference, const float *input, std::size_t count, bool silent = false, float gain = 1.0f) {
    const auto accepted = staging.append_packet(input, count, silent, gain);
    EXPECT_EQ(accepted, reference.append(input, count, silent, gain));
    EXPECT_EQ(staging.readable_count(), reference.samples.size());
    return accepted;
  }

  void read_and_compare(pcm_staging_t &staging, fifo_reference_t &reference, std::size_t count) {
    constexpr float untouched = -987.5f;
    std::vector<float> output(count, untouched);
    const bool complete = count <= reference.samples.size();
    EXPECT_EQ(staging.read_into(output), complete);
    if (complete) {
      for (std::size_t index = 0; index < count; ++index) {
        const auto expected = reference.samples.front();
        reference.samples.pop_front();
        EXPECT_EQ(std::bit_cast<std::uint32_t>(output[index]), std::bit_cast<std::uint32_t>(expected)) << "sample " << index;
      }
    } else {
      for (const auto value : output) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(value), std::bit_cast<std::uint32_t>(untouched));
      }
    }
    EXPECT_EQ(staging.readable_count(), reference.samples.size());
  }

  class WindowsAudioPcmStagingFramesTest: public testing::TestWithParam<std::pair<int, int>> {};

  TEST_P(WindowsAudioPcmStagingFramesTest, FourFrameBurstMatchesFifoAndCountsRelocatedSamples) {
    const auto [duration_ms, channels] = GetParam();
    const auto frame_samples = static_cast<std::size_t>(48 * duration_ms * channels);
    pcm_staging_t staging;
    staging.reset(4 * frame_samples);
    fifo_reference_t reference {4 * frame_samples, {}};

    // A native buffer of two requested frames gives the existing four-frame staging capacity.
    // One successful event wait can drain multiple native packets before capture reads frames.
    const auto first_packet = consecutive_samples(2 * frame_samples);
    const auto second_packet = consecutive_samples(2 * frame_samples, 2 * frame_samples + 1);
    ASSERT_EQ(append_and_compare(staging, reference, first_packet.data(), first_packet.size()), first_packet.size());
    ASSERT_EQ(append_and_compare(staging, reference, second_packet.data(), second_packet.size()), second_packet.size());
    for (int frame = 0; frame < 4; ++frame) {
      SCOPED_TRACE(frame);
      read_and_compare(staging, reference, frame_samples);
    }

    EXPECT_TRUE(reference.samples.empty());
    const auto relocated = staging.relocated_samples_for_test();
    EXPECT_EQ(relocated, 0u);
    RecordProperty("samples_per_frame", std::to_string(frame_samples));
    RecordProperty("relocated_samples", std::to_string(relocated));
  }

  TEST_P(WindowsAudioPcmStagingFramesTest, PartialDrainAndRefillPreserveEveryPendingSample) {
    const auto [duration_ms, channels] = GetParam();
    const auto frame_samples = static_cast<std::size_t>(48 * duration_ms * channels);
    pcm_staging_t staging;
    staging.reset(4 * frame_samples);
    fifo_reference_t reference {4 * frame_samples, {}};
    const auto first_packet = consecutive_samples(2 * frame_samples);
    const auto second_packet = consecutive_samples(frame_samples / 2, 2 * frame_samples + 1);
    ASSERT_EQ(append_and_compare(staging, reference, first_packet.data(), first_packet.size()), first_packet.size());
    ASSERT_EQ(append_and_compare(staging, reference, second_packet.data(), second_packet.size()), second_packet.size());
    read_and_compare(staging, reference, frame_samples);
    read_and_compare(staging, reference, frame_samples);
    ASSERT_EQ(staging.readable_count(), frame_samples / 2);

    // Less than one requested frame remains, so the real capture loop must refill now.
    const auto third_packet = consecutive_samples(frame_samples, 5 * frame_samples / 2 + 1);
    const auto fourth_packet = consecutive_samples(frame_samples / 2, 7 * frame_samples / 2 + 1);
    ASSERT_EQ(append_and_compare(staging, reference, third_packet.data(), third_packet.size()), third_packet.size());
    ASSERT_EQ(append_and_compare(staging, reference, fourth_packet.data(), fourth_packet.size()), fourth_packet.size());
    const auto refill_relocated = staging.relocated_samples_for_test();
    EXPECT_EQ(refill_relocated, frame_samples / 2);
    for (int frame = 0; frame < 2; ++frame) {
      SCOPED_TRACE(frame);
      read_and_compare(staging, reference, frame_samples);
    }

    EXPECT_TRUE(reference.samples.empty());
    EXPECT_EQ(staging.relocated_samples_for_test(), frame_samples / 2);
    RecordProperty("samples_per_frame", std::to_string(frame_samples));
    RecordProperty("refill_window_relocated_samples", std::to_string(refill_relocated));
    RecordProperty("total_relocated_samples", std::to_string(staging.relocated_samples_for_test()));
  }

  INSTANTIATE_TEST_SUITE_P(
    NegotiatedFrames,
    WindowsAudioPcmStagingFramesTest,
    testing::Values(std::pair {5, 2}, std::pair {5, 6}, std::pair {5, 8}, std::pair {10, 2}, std::pair {10, 6}, std::pair {10, 8}),
    [](const auto &info) {
      return "Ms" + std::to_string(info.param.first) + "Channels" + std::to_string(info.param.second);
    }
  );

  TEST(WindowsAudioPcmStagingTest, RecoveredHeadSpaceAcceptsPacketPrefixAndDropsOnlyNewestOverflow) {
    pcm_staging_t staging;
    staging.reset(12);
    fifo_reference_t reference {12, {}};
    const auto first = consecutive_samples(10);
    ASSERT_EQ(append_and_compare(staging, reference, first.data(), first.size(), false, 0.5f), 10u);
    read_and_compare(staging, reference, 3);
    read_and_compare(staging, reference, 3);

    const auto before_empty_append = staging.relocated_samples_for_test();
    EXPECT_EQ(append_and_compare(staging, reference, nullptr, 0), 0u);
    EXPECT_EQ(staging.relocated_samples_for_test(), before_empty_append);
    // Four old samples remain. The next append must reclaim six slots before packet 11..19.
    const auto incoming = consecutive_samples(9, 11);
    EXPECT_EQ(append_and_compare(staging, reference, incoming.data(), incoming.size(), false, 2.0f), 8u);
    const std::array more {91.0f, 92.0f};
    EXPECT_EQ(append_and_compare(staging, reference, more.data(), more.size()), 0u);
    read_and_compare(staging, reference, 12);
    EXPECT_TRUE(reference.samples.empty());
  }

  TEST(WindowsAudioPcmStagingTest, IncompleteReadLeavesDestinationAndPendingSamplesUntouched) {
    pcm_staging_t staging;
    staging.reset(8);
    fifo_reference_t reference {8, {}};
    const auto prefix = consecutive_samples(3);
    ASSERT_EQ(append_and_compare(staging, reference, prefix.data(), prefix.size()), 3u);
    read_and_compare(staging, reference, 5);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
    read_and_compare(staging, reference, 5);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);

    // A later successful refill completes the same pending prefix; no failed read consumed it.
    const auto suffix = consecutive_samples(2, 4);
    ASSERT_EQ(append_and_compare(staging, reference, suffix.data(), suffix.size()), 2u);
    read_and_compare(staging, reference, 5);
    EXPECT_TRUE(reference.samples.empty());
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
  }

  TEST(WindowsAudioPcmStagingTest, DefaultEmptyAndFullDrainOperationsDoNotRelocateSamples) {
    pcm_staging_t staging;
    fifo_reference_t reference {0, {}};
    read_and_compare(staging, reference, 0);
    EXPECT_EQ(append_and_compare(staging, reference, nullptr, 7, true, 0.5f), 0u);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);

    staging.reset(9);
    reference.capacity = 9;
    const auto packet = consecutive_samples(9);
    ASSERT_EQ(append_and_compare(staging, reference, packet.data(), packet.size()), 9u);
    read_and_compare(staging, reference, 0);
    EXPECT_EQ(append_and_compare(staging, reference, nullptr, 0), 0u);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
    read_and_compare(staging, reference, 9);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
    EXPECT_TRUE(reference.samples.empty());
  }

  TEST(WindowsAudioPcmStagingTest, ResetDropsOldDeviceSamplesChangesCapacityAndClearsRelocationCount) {
    pcm_staging_t staging;
    staging.reset(8);
    fifo_reference_t reference {8, {}};
    const auto old_packet = consecutive_samples(7);
    ASSERT_EQ(append_and_compare(staging, reference, old_packet.data(), old_packet.size()), 7u);
    read_and_compare(staging, reference, 3);

    staging.reset(5);
    reference = {5, {}};
    EXPECT_EQ(staging.readable_count(), 0u);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
    const auto new_packet = consecutive_samples(8, 101);
    EXPECT_EQ(append_and_compare(staging, reference, new_packet.data(), new_packet.size()), 5u);
    read_and_compare(staging, reference, 5);

    staging.reset(12);
    reference = {12, {}};
    const auto larger_packet = consecutive_samples(12, 201);
    EXPECT_EQ(append_and_compare(staging, reference, larger_packet.data(), larger_packet.size()), 12u);
    read_and_compare(staging, reference, 12);
    EXPECT_EQ(staging.relocated_samples_for_test(), 0u);
  }

  TEST(WindowsAudioPcmStagingTest, SilentNullPacketsAndGainApplicationPreserveExactPcmBits) {
    pcm_staging_t staging;
    staging.reset(32);
    fifo_reference_t reference {32, {}};
    constexpr std::array<std::uint32_t, 7> bits {
      0x00000000,
      0x80000000,
      0x3f800000,
      0xbf000000,
      0x00000001,
      0x7fc01234,
      0xffc04567,
    };
    std::array<float, bits.size()> unity;
    std::transform(bits.begin(), bits.end(), unity.begin(), [](std::uint32_t value) {
      return std::bit_cast<float>(value);
    });
    EXPECT_EQ(append_and_compare(staging, reference, unity.data(), unity.size()), unity.size());
    read_and_compare(staging, reference, unity.size());

    const std::array finite {-1.0f, -0.5f, -0.0f, 0.0f, 0.5f, 1.0f};
    for (const auto gain : {0.5f, 0.0f}) {
      SCOPED_TRACE(gain);
      EXPECT_EQ(append_and_compare(staging, reference, finite.data(), finite.size(), false, gain), finite.size());
      read_and_compare(staging, reference, finite.size());
    }

    // WASAPI may supply no source pointer for silence; it writes +0 without applying gain.
    for (const auto gain : {1.0f, 0.5f, std::numeric_limits<float>::quiet_NaN()}) {
      EXPECT_EQ(append_and_compare(staging, reference, nullptr, 4, true, gain), 4u);
      read_and_compare(staging, reference, 4);
    }
    EXPECT_TRUE(reference.samples.empty());
  }
}  // namespace

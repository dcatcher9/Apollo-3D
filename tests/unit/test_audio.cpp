/**
 * @file tests/unit/test_audio.cpp
 * @brief Regression tests for host audio level, timing, and Opus bitrate contracts.
 */

#include "../tests_common.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <opus/opus_multistream.h>
#include <src/audio.h>
#include <src/platform/windows/audio_helpers.h>
#include <vector>

using namespace audio;

struct AudioTest: PlatformTestSuite, testing::WithParamInterface<std::tuple<std::basic_string_view<char>, config_t>> {
  void SetUp() override {
    m_config = std::get<1>(GetParam());
    m_mail = std::make_shared<safe::mail_raw_t>();
  }

  config_t m_config;
  safe::mail_t m_mail;
};

constexpr std::bitset<config_t::MAX_FLAGS> config_flags(const int flag = -1) {
  std::bitset<config_t::MAX_FLAGS> result;
  if (flag >= 0) {
    result.set(flag);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
  Configurations,
  AudioTest,
  testing::Values(
    std::make_tuple("HIGH_STEREO", config_t {5, 2, 0x3, config_flags(config_t::HIGH_QUALITY)}),
    std::make_tuple("SURROUND51", config_t {5, 6, 0x3F, config_flags()}),
    std::make_tuple("SURROUND71", config_t {5, 8, 0x63F, config_flags()})
  ),
  [](const auto &info) {
    return std::string(std::get<0>(info.param));
  }
);

TEST_P(AudioTest, TestEncode) {
  std::thread timer([&] {
    // Terminate the audio capture after 100 ms
    std::this_thread::sleep_for(100ms);
    const auto shutdown_event = m_mail->event<bool>(mail::shutdown);
    const auto audio_packets = m_mail->queue<packet_t>(mail::audio_packets);
    shutdown_event->raise(true);
    audio_packets->stop();
  });
  std::thread capture([&] {
    const auto packets = m_mail->queue<packet_t>(mail::audio_packets);
    const auto shutdown_event = m_mail->event<bool>(mail::shutdown);
    while (const auto packet = packets->pop()) {
      if (shutdown_event->peek()) {
        break;
      }
      if (auto packet_data = packet->second; packet_data.size() == 0) {
        FAIL() << "Empty packet data";
      }
    }
  });
  audio::capture(m_mail, m_config, nullptr);

  timer.join();
  capture.join();
}

TEST(AudioPacketLifetimeTest, RetainsBroadcastStateUntilPacketIsConsumed) {
  auto channel = std::make_shared<int>(42);
  std::weak_ptr<int> weak_channel = channel;
  audio::packet_t packet {channel, util::buffer_t<std::uint8_t> {1}};

  channel.reset();
  EXPECT_FALSE(weak_channel.expired());

  packet.first.reset();
  EXPECT_TRUE(weak_channel.expired());
}

TEST(AudioStreamConfigTest, SelectsTheExactTotalBitrateForAllNegotiatedProfiles) {
  struct expected_t {
    int channels;
    bool high_quality;
    audio::stream_config_e index;
    int bitrate_kbps;
  };

  constexpr std::array expected {
    expected_t {2, false, audio::STEREO, 96},
    expected_t {2, true, audio::HIGH_STEREO, 512},
    expected_t {6, false, audio::SURROUND51, 256},
    expected_t {6, true, audio::HIGH_SURROUND51, 1536},
    expected_t {8, false, audio::SURROUND71, 450},
    expected_t {8, true, audio::HIGH_SURROUND71, 2048},
  };

  for (const auto &profile : expected) {
    const auto &selected = audio::select_stream_config(profile.channels, profile.high_quality);
    EXPECT_EQ(&selected, &audio::stream_configs[profile.index]);
    EXPECT_EQ(selected.channelCount, profile.channels);
    EXPECT_EQ(selected.bitrate, profile.bitrate_kbps * 1000);
    EXPECT_EQ(
      audio::selected_bitrate_kbps(profile.channels, profile.high_quality),
      profile.bitrate_kbps
    );
  }
}

TEST(WindowsAudioTimingTest, ConvertsReferenceTimeToACeilingMillisecondWait) {
  using platf::audio::detail::reference_time_to_wait_ms;

  EXPECT_EQ(reference_time_to_wait_ms(-1), 1u);
  EXPECT_EQ(reference_time_to_wait_ms(0), 1u);
  EXPECT_EQ(reference_time_to_wait_ms(1), 1u);
  EXPECT_EQ(reference_time_to_wait_ms(9'999), 1u);
  EXPECT_EQ(reference_time_to_wait_ms(10'000), 1u);
  EXPECT_EQ(reference_time_to_wait_ms(10'001), 2u);
  EXPECT_EQ(reference_time_to_wait_ms(50'000), 5u);
  EXPECT_EQ(reference_time_to_wait_ms(100'000), 10u);
  EXPECT_EQ(
    reference_time_to_wait_ms(std::numeric_limits<std::int64_t>::max()),
    std::numeric_limits<std::uint32_t>::max()
  );
}

TEST(WindowsAudioVolumeTest, AppliesMuteAndDecibelGainExactlyOnce) {
  using platf::audio::detail::apply_endpoint_volume;
  using platf::audio::detail::loopback_software_gain;

  constexpr float minus_six_db = -6.020599913f;
  const auto half_gain = loopback_software_gain(false, minus_six_db, false);
  EXPECT_NEAR(half_gain, 0.5f, 1e-6f);

  std::array samples {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  apply_endpoint_volume(samples, half_gain);
  EXPECT_NEAR(samples[0], -0.5f, 1e-6f);
  EXPECT_NEAR(samples[1], -0.25f, 1e-6f);
  EXPECT_FLOAT_EQ(samples[2], 0.0f);
  EXPECT_NEAR(samples[3], 0.25f, 1e-6f);
  EXPECT_NEAR(samples[4], 0.5f, 1e-6f);

  apply_endpoint_volume(samples, loopback_software_gain(false, 0.0f, true));
  for (const auto sample : samples) {
    EXPECT_FLOAT_EQ(sample, 0.0f);
  }

  // Native post-volume samples already contain Windows' attenuation. Software must stay at
  // unity even when the endpoint snapshot says muted or attenuated, or gain is applied twice.
  EXPECT_FLOAT_EQ(loopback_software_gain(true, minus_six_db, false), 1.0f);
  EXPECT_FLOAT_EQ(loopback_software_gain(true, 0.0f, true), 1.0f);
}

TEST(WindowsAudioVolumeTest, RetriesTransientSnapshotFailuresWithoutBusyLoopingOrLogStorms) {
  using refresh_state_t = platf::audio::detail::endpoint_volume_refresh_state_t;
  using namespace std::chrono_literals;

  refresh_state_t state;
  const refresh_state_t::clock_t::time_point start {10s};

  // The initial endpoint snapshot is pending immediately.
  EXPECT_TRUE(state.begin_if_due(start));
  EXPECT_FALSE(state.begin_if_due(start));

  // A failed COM snapshot re-arms the request, but it cannot spin on every capture packet.
  state.failed(start);
  EXPECT_TRUE(state.should_log_failure(start));
  EXPECT_FALSE(state.begin_if_due(start + refresh_state_t::retry_interval - 1ms));
  EXPECT_TRUE(state.begin_if_due(start + refresh_state_t::retry_interval));

  // Repeated failures remain pending while warnings are independently rate-limited.
  const auto retry_time = start + refresh_state_t::retry_interval;
  state.failed(retry_time);
  EXPECT_FALSE(state.should_log_failure(retry_time));
  const auto second_retry_time = retry_time + refresh_state_t::retry_interval;
  EXPECT_TRUE(state.begin_if_due(second_retry_time));
  EXPECT_FALSE(state.should_log_failure(start + refresh_state_t::warning_interval - 1ms));
  const auto warning_time = start + refresh_state_t::warning_interval;
  state.failed(warning_time);
  EXPECT_TRUE(state.should_log_failure(warning_time));
  EXPECT_FALSE(state.begin_if_due(warning_time + refresh_state_t::retry_interval - 1ms));
  const auto final_retry_time = warning_time + refresh_state_t::retry_interval;
  EXPECT_TRUE(state.begin_if_due(final_retry_time));

  // Success consumes the retry; a later callback schedules a fresh immediate snapshot.
  state.succeeded();
  EXPECT_FALSE(state.begin_if_due(final_retry_time));
  state.notify();
  EXPECT_TRUE(state.begin_if_due(final_retry_time));
}

TEST(AudioOpusLevelTest, StereoRoundTripDoesNotIntroduceSustainedAttenuation) {
  const auto &stream = audio::select_stream_config(2, false);
  int error = OPUS_OK;
  using encoder_ptr = std::unique_ptr<OpusMSEncoder, decltype(&opus_multistream_encoder_destroy)>;
  using decoder_ptr = std::unique_ptr<OpusMSDecoder, decltype(&opus_multistream_decoder_destroy)>;

  encoder_ptr encoder {
    opus_multistream_encoder_create(
      stream.sampleRate,
      stream.channelCount,
      stream.streams,
      stream.coupledStreams,
      stream.mapping,
      OPUS_APPLICATION_RESTRICTED_LOWDELAY,
      &error
    ),
    opus_multistream_encoder_destroy,
  };
  ASSERT_EQ(error, OPUS_OK);
  ASSERT_TRUE(encoder);
  ASSERT_EQ(opus_multistream_encoder_ctl(encoder.get(), OPUS_SET_BITRATE(stream.bitrate)), OPUS_OK);
  ASSERT_EQ(opus_multistream_encoder_ctl(encoder.get(), OPUS_SET_VBR(0)), OPUS_OK);

  decoder_ptr decoder {
    opus_multistream_decoder_create(
      stream.sampleRate,
      stream.channelCount,
      stream.streams,
      stream.coupledStreams,
      stream.mapping,
      &error
    ),
    opus_multistream_decoder_destroy,
  };
  ASSERT_EQ(error, OPUS_OK);
  ASSERT_TRUE(decoder);

  constexpr int packet_duration_ms = 5;
  constexpr int packet_count = 120;
  constexpr int warmup_packets = 20;
  const int frames_per_packet = stream.sampleRate * packet_duration_ms / 1000;
  std::vector<float> input(frames_per_packet * stream.channelCount);
  std::vector<float> decoded(input.size());
  std::array<unsigned char, 1400> packet {};

  double input_energy = 0.0;
  double decoded_energy = 0.0;
  std::uint64_t compared_samples = 0;
  std::uint64_t generated_frames = 0;
  constexpr float amplitude = 0.25f;
  constexpr float frequency = 1000.0f;

  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    for (int frame = 0; frame < frames_per_packet; ++frame, ++generated_frames) {
      const auto phase = 2.0f * std::numbers::pi_v<float> * frequency *
                         static_cast<float>(generated_frames) /
                         static_cast<float>(stream.sampleRate);
      const auto value = amplitude * std::sin(phase);
      input[frame * 2] = value;
      input[frame * 2 + 1] = value;
    }

    const auto bytes = opus_multistream_encode_float(
      encoder.get(),
      input.data(),
      frames_per_packet,
      packet.data(),
      static_cast<opus_int32>(packet.size())
    );
    ASSERT_GT(bytes, 0);
    const auto decoded_frames = opus_multistream_decode_float(
      decoder.get(),
      packet.data(),
      bytes,
      decoded.data(),
      frames_per_packet,
      0
    );
    ASSERT_EQ(decoded_frames, frames_per_packet);

    if (packet_index >= warmup_packets) {
      for (std::size_t index = 0; index < input.size(); ++index) {
        input_energy += static_cast<double>(input[index]) * input[index];
        decoded_energy += static_cast<double>(decoded[index]) * decoded[index];
      }
      compared_samples += input.size();
    }
  }

  ASSERT_GT(compared_samples, 0u);
  const auto input_rms = std::sqrt(input_energy / compared_samples);
  const auto decoded_rms = std::sqrt(decoded_energy / compared_samples);
  const auto level_delta_db = 20.0 * std::log10(decoded_rms / input_rms);
  EXPECT_NEAR(level_delta_db, 0.0, 1.0);
}

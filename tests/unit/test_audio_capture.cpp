/**
 * @file tests/unit/test_audio_capture.cpp
 * @brief Ownership and failure regressions for the PCM capture producer.
 */

#include "src/audio.h"
#include "src/audio_capture.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <numbers>
#include <opus/opus_multistream.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
  using audio::detail::capture_pcm_frame;
  using audio::detail::pcm_frame_t;
  using audio::detail::pcm_queue_t;
  using platf::capture_e;
  using namespace std::chrono_literals;

  struct sample_step_t {
    capture_e status;
    std::vector<float> written;
  };

  class scripted_mic_t final: public platf::mic_t {
  public:
    explicit scripted_mic_t(std::initializer_list<sample_step_t> steps):
        steps_ {steps} {
    }

    capture_e sample(std::span<float> frame) override {
      buffers.push_back(frame.data());
      sizes.push_back(frame.size());
      if (next_ >= steps_.size()) {
        ADD_FAILURE() << "Unexpected microphone sample";
        return capture_e::error;
      }
      const auto &step = steps_[next_++];
      EXPECT_LE(step.written.size(), frame.size());
      if (step.status == capture_e::ok) {
        EXPECT_EQ(step.written.size(), frame.size());
      }
      std::copy_n(step.written.begin(), std::min(step.written.size(), frame.size()), frame.begin());
      return step.status;
    }

    // Record addresses and extents, without retaining or reading borrowed PCM storage.
    std::vector<float *> buffers;
    std::vector<std::size_t> sizes;

  private:
    std::vector<sample_step_t> steps_;
    std::size_t next_ {0};
  };

  TEST(AudioCaptureProducerTest, RetainsPendingStorageAcrossTimeoutAndMicrophoneReplacement) {
    const std::vector<float> expected {0.25f, -0.5f, 0.75f, -1.0f, 0.0f, 0.5f, -0.25f, 1.0f};
    pcm_queue_t samples {30};
    pcm_frame_t pending;
    auto mic = std::make_unique<scripted_mic_t>(std::initializer_list<sample_step_t> {
      {capture_e::timeout, {91.0f, 92.0f}},
      {capture_e::reinit, {-93.0f}},
    });

    EXPECT_EQ(capture_pcm_frame(*mic, pending, expected.size(), samples), capture_e::timeout);
    ASSERT_TRUE(pending);
    const auto *address = pending.get();
    EXPECT_FLOAT_EQ(pending[0], 91.0f);
    EXPECT_FLOAT_EQ(pending[1], 92.0f);
    EXPECT_FALSE(samples.pop(0ms));

    EXPECT_EQ(capture_pcm_frame(*mic, pending, expected.size(), samples), capture_e::reinit);
    ASSERT_TRUE(pending);
    EXPECT_EQ(pending.get(), address);
    EXPECT_FLOAT_EQ(pending[0], -93.0f);
    EXPECT_FALSE(samples.pop(0ms));
    EXPECT_EQ(mic->buffers, (std::vector<float *> {pending.get(), pending.get()}));
    EXPECT_EQ(mic->sizes, (std::vector<std::size_t> {expected.size(), expected.size()}));

    // Production replaces the microphone on reinit, while the capture owner keeps its storage.
    mic = std::make_unique<scripted_mic_t>(std::initializer_list<sample_step_t> {{capture_e::ok, expected}});
    EXPECT_EQ(capture_pcm_frame(*mic, pending, expected.size(), samples), capture_e::ok);
    EXPECT_FALSE(pending);
    ASSERT_EQ(mic->buffers.size(), 1u);
    EXPECT_EQ(mic->buffers.front(), address);
    auto queued = samples.pop(0ms);
    ASSERT_TRUE(queued);
    EXPECT_EQ(queued.get(), address);
    EXPECT_EQ(std::vector<float>(queued.get(), queued.get() + expected.size()), expected);
    EXPECT_FALSE(samples.pop(0ms));
  }

  TEST(AudioCaptureProducerTest, ErrorNeverPublishesAPartiallyWrittenFrame) {
    pcm_queue_t samples {30};
    pcm_frame_t pending;
    scripted_mic_t mic {{capture_e::error, {81.0f, -82.0f}}};

    EXPECT_EQ(capture_pcm_frame(mic, pending, 16, samples), capture_e::error);
    ASSERT_TRUE(pending);
    ASSERT_EQ(mic.buffers.size(), 1u);
    EXPECT_EQ(pending.get(), mic.buffers.front());
    EXPECT_FLOAT_EQ(pending[0], 81.0f);
    EXPECT_FLOAT_EQ(pending[1], -82.0f);
    EXPECT_FALSE(samples.pop(0ms));
    // The terminal-error return destroys this still-owned pending frame with the capture owner.
  }

  TEST(AudioCaptureProducerTest, QueuedFramesKeepTheirContentsThroughLaterCaptureAndMicrophoneDestruction) {
    const std::vector<float> first {0.25f, -0.25f, 0.5f, -0.5f};
    const std::vector<float> second {0.75f, -0.75f, 1.0f, -1.0f};
    pcm_queue_t samples {30};
    pcm_frame_t pending;
    auto mic = std::make_unique<scripted_mic_t>(std::initializer_list<sample_step_t> {
      {capture_e::ok, first},
      {capture_e::timeout, {99.0f}},
      {capture_e::ok, second},
    });

    EXPECT_EQ(capture_pcm_frame(*mic, pending, first.size(), samples), capture_e::ok);
    EXPECT_FALSE(pending);
    EXPECT_EQ(capture_pcm_frame(*mic, pending, first.size(), samples), capture_e::timeout);
    ASSERT_TRUE(pending);
    ASSERT_EQ(samples.unsafe().size(), 1u);
    EXPECT_NE(pending.get(), samples.unsafe().front().get());
    EXPECT_EQ(capture_pcm_frame(*mic, pending, first.size(), samples), capture_e::ok);
    EXPECT_FALSE(pending);
    mic.reset();

    auto queued_first = samples.pop(0ms);
    auto queued_second = samples.pop(0ms);
    ASSERT_TRUE(queued_first);
    ASSERT_TRUE(queued_second);
    EXPECT_NE(queued_first.get(), queued_second.get());
    EXPECT_EQ(std::vector<float>(queued_first.get(), queued_first.get() + first.size()), first);
    EXPECT_EQ(std::vector<float>(queued_second.get(), queued_second.get() + second.size()), second);
    EXPECT_FALSE(samples.pop(0ms));
  }

  TEST(AudioCaptureProducerTest, OverflowDestroysStaleQueueOwnershipAndKeepsTheNewestFullFrame) {
    pcm_queue_t samples {2};
    pcm_frame_t pending;
    scripted_mic_t mic {
      {capture_e::ok, {1.0f, 2.0f}},
      {capture_e::ok, {3.0f, 4.0f}},
      {capture_e::ok, {5.0f, 6.0f}},
    };

    EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::ok);
    EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::ok);
    ASSERT_EQ(samples.unsafe().size(), 2u);
    EXPECT_FALSE(pending);
    EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::ok);
    EXPECT_FALSE(pending);
    ASSERT_EQ(samples.unsafe().size(), 1u);
    auto newest = samples.pop(0ms);
    ASSERT_TRUE(newest);
    EXPECT_FLOAT_EQ(newest[0], 5.0f);
    EXPECT_FLOAT_EQ(newest[1], 6.0f);
    EXPECT_FALSE(samples.pop(0ms));
  }

  TEST(AudioCaptureProducerTest, StoppedQueueKeepsPendingOwnershipUntilCaptureCleanup) {
    pcm_frame_t pending;
    {
      pcm_queue_t samples {30};
      scripted_mic_t mic {
        {capture_e::ok, {1.0f, 2.0f}},
        {capture_e::timeout, {-7.0f}},
        {capture_e::ok, {3.0f, 4.0f}},
      };
      EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::ok);
      EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::timeout);
      ASSERT_TRUE(pending);
      const auto *address = pending.get();
      samples.stop();

      // The encoder can stop its queue while a successful microphone call is in progress.
      EXPECT_EQ(capture_pcm_frame(mic, pending, 2, samples), capture_e::ok);
      ASSERT_TRUE(pending);
      EXPECT_EQ(pending.get(), address);
      EXPECT_FLOAT_EQ(pending[0], 3.0f);
      EXPECT_FLOAT_EQ(pending[1], 4.0f);
      EXPECT_FALSE(samples.pop(0ms));
      ASSERT_EQ(samples.unsafe().size(), 1u);
      EXPECT_NE(samples.unsafe().front().get(), pending.get());
      EXPECT_FLOAT_EQ(samples.unsafe().front()[0], 1.0f);
      EXPECT_FLOAT_EQ(samples.unsafe().front()[1], 2.0f);
    }
    // Destroying the stopped queue releases queued frames without taking pending ownership.
    ASSERT_TRUE(pending);
    EXPECT_FLOAT_EQ(pending[0], 3.0f);
    EXPECT_FLOAT_EQ(pending[1], 4.0f);
  }

  class AudioCaptureShapeTest: public testing::TestWithParam<std::pair<int, int>> {};

  TEST_P(AudioCaptureShapeTest, OwnedNegotiatedFramesRoundTripThroughOpus) {
    const auto [duration_ms, channels] = GetParam();
    const auto &stream = audio::select_stream_config(channels, false);
    const int frame_size = duration_ms * stream.sampleRate / 1000;
    const auto sample_count = static_cast<std::size_t>(frame_size * stream.channelCount);
    int error = OPUS_OK;
    using encoder_ptr = std::unique_ptr<OpusMSEncoder, decltype(&opus_multistream_encoder_destroy)>;
    using decoder_ptr = std::unique_ptr<OpusMSDecoder, decltype(&opus_multistream_decoder_destroy)>;
    encoder_ptr encoder {
      opus_multistream_encoder_create(stream.sampleRate, stream.channelCount, stream.streams, stream.coupledStreams, stream.mapping, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error),
      opus_multistream_encoder_destroy,
    };
    ASSERT_EQ(error, OPUS_OK);
    ASSERT_TRUE(encoder);
    encoder_ptr control_encoder {
      opus_multistream_encoder_create(stream.sampleRate, stream.channelCount, stream.streams, stream.coupledStreams, stream.mapping, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error),
      opus_multistream_encoder_destroy,
    };
    ASSERT_EQ(error, OPUS_OK);
    ASSERT_TRUE(control_encoder);
    for (auto *codec : {encoder.get(), control_encoder.get()}) {
      ASSERT_EQ(opus_multistream_encoder_ctl(codec, OPUS_SET_BITRATE(stream.bitrate)), OPUS_OK);
      ASSERT_EQ(opus_multistream_encoder_ctl(codec, OPUS_SET_VBR(0)), OPUS_OK);
    }
    decoder_ptr decoder {
      opus_multistream_decoder_create(stream.sampleRate, stream.channelCount, stream.streams, stream.coupledStreams, stream.mapping, &error),
      opus_multistream_decoder_destroy,
    };
    ASSERT_EQ(error, OPUS_OK);
    ASSERT_TRUE(decoder);

    pcm_queue_t samples {30};
    pcm_frame_t pending;
    std::vector<float> input(sample_count);
    std::vector<float> decoded(sample_count);
    std::array<unsigned char, 1400> packet;
    std::array<unsigned char, 1400> control_packet;
    double decoded_energy = 0.0;
    for (int packet_index = 0; packet_index < 12; ++packet_index) {
      for (int frame = 0; frame < frame_size; ++frame) {
        const auto phase = 2.0f * std::numbers::pi_v<float> * 1000.0f *
                           static_cast<float>(packet_index * frame_size + frame) / stream.sampleRate;
        for (int channel = 0; channel < channels; ++channel) {
          input[frame * channels + channel] = 0.25f * std::sin(phase);
        }
      }
      scripted_mic_t mic {{capture_e::ok, input}};
      EXPECT_EQ(capture_pcm_frame(mic, pending, sample_count, samples), capture_e::ok);
      EXPECT_FALSE(pending);
      ASSERT_EQ(mic.sizes.size(), 1u);
      EXPECT_EQ(mic.sizes.front(), sample_count);
      auto captured = samples.pop(0ms);
      ASSERT_TRUE(captured);
      EXPECT_EQ(std::vector<float>(captured.get(), captured.get() + sample_count), input);
      const auto bytes = opus_multistream_encode_float(encoder.get(), captured.get(), frame_size, packet.data(), static_cast<opus_int32>(packet.size()));
      ASSERT_GT(bytes, 0);
      const auto control_bytes = opus_multistream_encode_float(control_encoder.get(), input.data(), frame_size, control_packet.data(), static_cast<opus_int32>(control_packet.size()));
      ASSERT_EQ(bytes, control_bytes);
      EXPECT_EQ(std::vector<unsigned char>(packet.data(), packet.data() + bytes), std::vector<unsigned char>(control_packet.data(), control_packet.data() + control_bytes));
      ASSERT_EQ(opus_multistream_decode_float(decoder.get(), packet.data(), bytes, decoded.data(), frame_size, 0), frame_size);
      for (const auto value : decoded) {
        EXPECT_TRUE(std::isfinite(value));
        decoded_energy += static_cast<double>(value) * value;
      }
    }
    EXPECT_GT(decoded_energy, 0.001);
    EXPECT_FALSE(samples.pop(0ms));
  }

  INSTANTIATE_TEST_SUITE_P(
    NegotiatedFrames,
    AudioCaptureShapeTest,
    testing::Values(std::pair {5, 2}, std::pair {5, 6}, std::pair {5, 8}, std::pair {10, 2}, std::pair {10, 6}, std::pair {10, 8}),
    [](const auto &info) {
      return "Ms" + std::to_string(info.param.first) + "Channels" + std::to_string(info.param.second);
    }
  );
}  // namespace

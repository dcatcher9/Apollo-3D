/**
 * @file src/audio.cpp
 * @brief Definitions for audio capture and encoding.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <thread>

// lib includes
#include <opus/opus_multistream.h>

// local includes
#include "audio.h"
#include "config.h"
#include "globals.h"
#include "logging.h"
#include "platform/common.h"
#include "thread_safe.h"
#include "utility.h"

namespace audio {
  using namespace std::literals;
  using opus_t = util::safe_ptr<OpusMSEncoder, opus_multistream_encoder_destroy>;
  using sample_queue_t = std::shared_ptr<safe::queue_t<std::vector<float>>>;

  static int start_audio_control(audio_ctx_t &ctx);
  static void stop_audio_control(audio_ctx_t &);

  constexpr auto SAMPLE_RATE = 48000;
  constexpr auto LEVEL_TELEMETRY_INTERVAL = 5s;

  const opus_stream_config_t stream_configs[MAX_STREAM_CONFIG] {
    {
      SAMPLE_RATE,
      2,
      1,
      1,
      platf::speaker::map_stereo,
      96000,
    },
    {
      SAMPLE_RATE,
      2,
      1,
      1,
      platf::speaker::map_stereo,
      512000,
    },
    {
      SAMPLE_RATE,
      6,
      4,
      2,
      platf::speaker::map_surround51,
      256000,
    },
    {
      SAMPLE_RATE,
      6,
      6,
      0,
      platf::speaker::map_surround51,
      1536000,
    },
    {
      SAMPLE_RATE,
      8,
      5,
      3,
      platf::speaker::map_surround71,
      450000,
    },
    {
      SAMPLE_RATE,
      8,
      8,
      0,
      platf::speaker::map_surround71,
      2048000,
    },
  };

  namespace {
    class rolling_level_meter_t {
    public:
      void observe(const std::vector<float> &samples) {
        for (const auto sample : samples) {
          if (!std::isfinite(sample)) {
            continue;
          }

          const auto magnitude = std::abs(sample);
          peak_ = std::max(peak_, magnitude);
          sum_squares_ += static_cast<double>(sample) * sample;
          ++sample_count_;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - interval_start_ < LEVEL_TELEMETRY_INTERVAL) {
          return;
        }

        const auto rms = sample_count_ ?
                           std::sqrt(sum_squares_ / static_cast<double>(sample_count_)) :
                           0.0;
        const auto to_dbfs = [](double amplitude) {
          return amplitude > 0.0 ? 20.0 * std::log10(amplitude) :
                                   -std::numeric_limits<double>::infinity();
        };

        BOOST_LOG(info) << std::format(
          "Pre-Opus audio level (rolling 5 s): peak {:.1f} dBFS, RMS {:.1f} dBFS",
          to_dbfs(peak_),
          to_dbfs(rms)
        );

        interval_start_ = now;
        sample_count_ = 0;
        sum_squares_ = 0.0;
        peak_ = 0.0;
      }

    private:
      std::chrono::steady_clock::time_point interval_start_ {std::chrono::steady_clock::now()};
      std::uint64_t sample_count_ {};
      double sum_squares_ {};
      float peak_ {};
    };

    int map_stream(int channels, bool quality) {
      const int shift = quality ? 1 : 0;
      switch (channels) {
        case 2:
          return STEREO + shift;
        case 6:
          return SURROUND51 + shift;
        case 8:
          return SURROUND71 + shift;
      }
      return STEREO;
    }
  }  // namespace

  const opus_stream_config_t &select_stream_config(int channels, bool high_quality) {
    return stream_configs[map_stream(channels, high_quality)];
  }

  int selected_bitrate_kbps(int channels, bool high_quality) {
    return select_stream_config(channels, high_quality).bitrate / 1000;
  }

  void encodeThread(sample_queue_t samples, config_t config, std::shared_ptr<void> channel_data) {
    auto packets = mail::man->queue<packet_t>(mail::audio_packets);
    const auto &stream = select_stream_config(config.channels, config.flags[config_t::HIGH_QUALITY]);

    // Encoding takes place on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    opus_t opus {opus_multistream_encoder_create(
      stream.sampleRate,
      stream.channelCount,
      stream.streams,
      stream.coupledStreams,
      stream.mapping,
      OPUS_APPLICATION_RESTRICTED_LOWDELAY,
      nullptr
    )};

    opus_multistream_encoder_ctl(opus.get(), OPUS_SET_BITRATE(stream.bitrate));
    opus_multistream_encoder_ctl(opus.get(), OPUS_SET_VBR(0));

    BOOST_LOG(info) << "Opus initialized: "sv << stream.sampleRate / 1000 << " kHz, "sv
                    << stream.channelCount << " channels, "sv
                    << stream.bitrate / 1000 << " kbps (total), LOWDELAY"sv;

    auto frame_size = config.packetDuration * stream.sampleRate / 1000;
    rolling_level_meter_t level_meter;
    while (auto sample = samples->pop()) {
      level_meter.observe(*sample);
      buffer_t packet {1400};

      int bytes = opus_multistream_encode_float(opus.get(), sample->data(), frame_size, std::begin(packet), packet.size());
      if (bytes < 0) {
        BOOST_LOG(error) << "Couldn't encode audio: "sv << opus_strerror(bytes);
        packets->stop();

        return;
      }

      packet.fake_resize(bytes);
      packets->raise(channel_data, std::move(packet));
    }
  }

  void capture(safe::mail_t mail, config_t config, std::shared_ptr<void> channel_data) {
    auto shutdown_event = mail->event<bool>(mail::shutdown);
    const auto &stream = select_stream_config(config.channels, config.flags[config_t::HIGH_QUALITY]);

    auto ref = get_audio_ctx_ref();
    if (!ref) {
      return;
    }

    auto init_failure_fg = util::fail_guard([&shutdown_event]() {
      BOOST_LOG(error) << "Unable to initialize audio capture. The stream will not have audio."sv;

      // Wait for shutdown to be signalled if we fail init.
      // This allows streaming to continue without audio.
      shutdown_event->view();
      return;
    });

    auto &control = ref->control;
    if (!control) {
      return;
    }

    // Order of priority:
    // 1. Virtual sink
    // 2. Audio sink
    // 3. Host
    std::string *sink = &ref->sink.host;
    if (!config::audio.sink.empty()) {
      sink = &config::audio.sink;
    }

    // Prefer the virtual sink if host playback is disabled or there's no other sink
    if (ref->sink.null && (!config.flags[config_t::HOST_AUDIO] || sink->empty())) {
      auto &null = *ref->sink.null;
      switch (stream.channelCount) {
        case 2:
          sink = &null.stereo;
          break;
        case 6:
          sink = &null.surround51;
          break;
        case 8:
          sink = &null.surround71;
          break;
      }
    }

    BOOST_LOG(info) << "Selected audio sink: "sv << *sink;

    // Every capture explicitly applies its requested sink. Keep restoration intent monotonic so
    // releasing this context always returns to the host device after any virtual-sink capture.
    const bool differs_from_host = ref->sink.host != *sink;
    if (ref->restore_sink || differs_from_host) {
      if (control->set_sink(*sink)) {
        return;
      }
    }
    ref->restore_sink = ref->restore_sink || differs_from_host;

    auto frame_size = config.packetDuration * stream.sampleRate / 1000;
    auto mic = control->microphone(stream.mapping, stream.channelCount, stream.sampleRate, frame_size);
    if (!mic) {
      return;
    }

    // Audio is initialized, so we don't want to print the failure message
    init_failure_fg.disable();

    // Capture takes place on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::critical);

    auto samples = std::make_shared<sample_queue_t::element_type>(30);
    std::thread thread {encodeThread, samples, config, channel_data};

    auto fg = util::fail_guard([&]() {
      samples->stop();
      thread.join();

      shutdown_event->view();
    });

    int samples_per_frame = frame_size * stream.channelCount;

    while (!shutdown_event->peek()) {
      std::vector<float> sample_buffer;
      sample_buffer.resize(samples_per_frame);

      auto status = mic->sample(sample_buffer);
      switch (status) {
        case platf::capture_e::ok:
          break;
        case platf::capture_e::timeout:
          continue;
        case platf::capture_e::reinit:
          BOOST_LOG(info) << "Reinitializing audio capture"sv;
          mic.reset();
          do {
            mic = control->microphone(stream.mapping, stream.channelCount, stream.sampleRate, frame_size);
            if (!mic) {
              BOOST_LOG(warning) << "Couldn't re-initialize audio input"sv;
            }
          } while (!mic && !shutdown_event->view(5s));

          continue;
        default:
          return;
      }

      samples->raise(std::move(sample_buffer));
    }
  }

  audio_ctx_ref_t get_audio_ctx_ref() {
    static auto control_shared {safe::make_shared<audio_ctx_t>(start_audio_control, stop_audio_control)};
    return control_shared.ref();
  }

  int start_audio_control(audio_ctx_t &ctx) {
    auto fg = util::fail_guard([]() {
      BOOST_LOG(warning) << "There will be no audio"sv;
    });

    // The default sink has not been replaced yet.
    ctx.restore_sink = false;

    if (!(ctx.control = platf::audio_control())) {
      return 0;
    }

    auto sink = ctx.control->sink_info();
    if (!sink) {
      // Let the calling code know it failed
      ctx.control.reset();
      return 0;
    }

    ctx.sink = std::move(*sink);

    fg.disable();
    return 0;
  }

  void stop_audio_control(audio_ctx_t &ctx) {
    // restore audio-sink if applicable
    if (!ctx.restore_sink) {
      return;
    }

    // Change back to the host sink, unless there was none
    const std::string &sink = ctx.sink.host.empty() ? config::audio.sink : ctx.sink.host;
    if (!sink.empty()) {
      // Best effort, it's allowed to fail
      ctx.control->set_sink(sink);
    }
  }

}  // namespace audio

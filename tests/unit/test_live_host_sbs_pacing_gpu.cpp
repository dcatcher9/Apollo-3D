#include <gtest/gtest.h>

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
  #include "src/globals.h"
  #include "src/live_model_preparation.h"
  #include "src/platform/windows/display.h"
  #include "src/platform/windows/host_sbs_actual_converter_probe.h"
  #include "src/platform/windows/model_preparation_converter_test.h"
  #include "src/video_capture_test.h"

  #include <algorithm>
  #include <atomic>
  #include <chrono>
  #include <cstdint>
  #include <cstdlib>
  #include <exception>
  #include <iostream>
  #include <mutex>
  #include <optional>
  #include <string_view>
  #include <thread>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using namespace std::chrono_literals;
  using clock_t = std::chrono::steady_clock;
  namespace fixture = platf::dxgi::testing;

  std::optional<fixture::model_preparation_adapter_t> nvidia_adapter() {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      return std::nullopt;
    }
    for (UINT index = 0;; ++index) {
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
      if (FAILED(factory->EnumAdapters1(index, &adapter))) {
        return std::nullopt;
      }
      DXGI_ADAPTER_DESC1 description {};
      if (SUCCEEDED(adapter->GetDesc1(&description)) && description.VendorId == 0x10deu &&
          !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
        return fixture::model_preparation_adapter_t {
          description.AdapterLuid.LowPart, description.AdapterLuid.HighPart,
          description.VendorId, description.DeviceId
        };
      }
    }
  }

  TEST(LiveHostSbsPacingGpuTest, RetainedSourceReachesNativePacketsAfterModelReadiness) {
    const auto *enabled = std::getenv("APOLLO_RUN_TENSORRT_TESTS");
    if (!enabled || std::string_view {enabled} != "1") {
      GTEST_SKIP() << "Set APOLLO_RUN_TENSORRT_TESTS=1 for the local NVIDIA/TensorRT integration check.";
    }

    const auto adapter = nvidia_adapter();
    ASSERT_TRUE(adapter);
    constexpr unsigned width = 1280, height = 720;
    auto display = fixture::make_native_pacing_display(*adapter, width, height);
    auto mail = std::make_shared<safe::mail_raw_t>();
    auto images = std::make_shared<video::img_event_t::element_type>();
    auto packets = std::make_shared<video::packet_queue_t::element_type>();
    auto shutdown = mail->event<bool>(mail::shutdown);
    auto depth_status = mail->event<int>(mail::sbs_depth_status);
    auto telemetry = mail->event<video::sbs_telemetry_snapshot_t>(mail::sbs_telemetry);
    auto subscription = std::make_shared<video::sbs_telemetry_subscription_t>();
    subscription->update(true, true, 50);

    video::config_t config {};
    config.width = 2 * width;
    config.height = height;
    config.framerate = 90;
    config.framerateX100 = 9000;
    config.encodingFramerate = 90000;
    config.bitrate = 20000;
    config.slicesPerFrame = 1;
    config.numRefFrames = 1;
    config.encoderCscMode = 2;
    config.videoFormat = 1;  // HEVC
    config.sbs_mode = video::SBS_AI;
    config.sbs_depth_status_event = depth_status;
    config.sbs_depth_pipeline_ready_event = std::make_shared<safe::event_t<bool>>();
    config.sbs_telemetry_event = telemetry;
    config.sbs_telemetry_subscription = subscription;
    config.sbs_telemetry_generation = 1;

    // Two visibly distinct shared images stay alive until their native encode ownership ends.
    std::vector<std::uint32_t> pixels(std::size_t(width) * height);
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        pixels[std::size_t(y) * width + x] =
          (x / 64 + y / 48) % 2 ? 0xff3853a0u : 0xffa48b45u;
      }
    }
    auto first = fixture::upload_native_pacing_image(*display, pixels);
    for (auto &pixel : pixels) {
      pixel ^= 0x00569662u;
    }
    auto second = fixture::upload_native_pacing_image(*display, pixels);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);

    models::live_model_preparation_t preparation(video::host_sbs_v2_depth_model(), {});
    video::detail::native_encode_loop_test_result_t result;
    std::exception_ptr loop_error;
    {
      std::jthread encoder([&](std::stop_token stop) {
        std::stop_callback stop_loop(stop, [&]() {
          shutdown->raise(true);
          images->wake();
        });
        try {
          result = video::detail::run_native_encode_loop_for_test(display, mail, images, config, packets);
        } catch (...) {
          loop_error = std::current_exception();
        }
        packets->stop();
      });
      auto captured = clock_t::now();
      first->frame_timestamp = first->content_timestamp = captured;
      images->raise(first);

      std::int64_t last_index = 0;
      const auto next_packet = [&]() -> video::packet_t {
        auto next = packets->pop(100ms);
        if (!next) {
          return {};
        }
        EXPECT_GT(next->data_size(), 0u);
        EXPECT_GT(next->frame_index(), last_index);
        last_index = next->frame_index();
        return next;
      };
      bool retained_source_published_depth = false;
      const auto preparation_limit = clock_t::now() + 30s;
      while (packets->running() && clock_t::now() < preparation_limit) {
        const auto packet = next_packet();
        const auto sample = telemetry->view(0ms);
        const auto pipeline_status = depth_status->view(0ms);
        if (packet && packet->content_timestamp == captured && packet->processing_started &&
            pipeline_status && *pipeline_status == 2 &&
            sample && sample->status == video::sbs_telemetry_sample_status_e::ready &&
            sample->sampled_frame_id > 0 && sample->depth_width > 0) {
          retained_source_published_depth = true;
          break;
        }
      }
      EXPECT_TRUE(retained_source_published_depth);

      captured = clock_t::now();
      second->frame_timestamp = second->content_timestamp = captured;
      images->raise(second);
      video::packet_t matched;
      const auto packet_limit = captured + 2s;
      while (packets->running() && clock_t::now() < packet_limit) {
        auto packet = next_packet();
        if (packet && packet->content_timestamp == captured) {
          matched = std::move(packet);
          break;
        }
      }
      EXPECT_TRUE(matched);
      if (matched) {
        EXPECT_TRUE(matched->processing_started);
        EXPECT_GE(matched->encoded_timestamp, captured);
        std::cout << "Live Host SBS native packet frame=" << matched->frame_index()
                  << " bytes=" << matched->data_size()
                  << " capture_to_packet_us="
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                       matched->encoded_timestamp - captured
                     ).count() << '\n';
      }
      encoder.request_stop();
    }
    if (loop_error) {
      std::rethrow_exception(loop_error);
    }
    EXPECT_TRUE(result.session_created);
    EXPECT_EQ(result.pipeline_depth, 2u);
    EXPECT_GT(result.next_frame_nr, 2);
  }

  namespace probe = platf::dxgi::actual_converter_probe_v1;

  // Copy values only: diagnostic resource aliases must not outlive their observed conversion.
  struct pacing_event_t {
    probe::event_e kind {};
    std::uint64_t conversion = 0, model = 0, target = 0, before = 0, after = 0;
    std::uint64_t actual_pixels = 0;
    std::int64_t source = 0, current = 0, packed_source = 0;
    bool flat = false, repeat = false, current_publication = false;
    bool submitted = false;
    bool held = false, queried_busy = false, pending_target = false;
    bool packed_valid = false, route_matches = false;
  };

  struct pacing_delivery_t {
    fixture::native_pacing_delivery_t call;
    clock_t::time_point previous_finished {};
    pacing_event_t output;

    bool owns_packet(const video::encoded_packet_t &packet) const {
      return call.result == 0 && packet.processing_started &&
             *packet.processing_started > previous_finished &&
             *packet.processing_started <= call.started &&
             packet.encoded_timestamp >= call.finished;
    }
  };

  struct held_pacing_t {
    std::mutex mutex;
    std::atomic<std::int64_t> held_source {0};
    video::img_event_t images;
    safe::mail_t mail;
    std::shared_ptr<platf::img_t> next_capture;
    std::weak_ptr<platf::img_t> held_image;
    std::vector<pacing_event_t> events;
    std::vector<fixture::native_pacing_poll_t> polls;
    std::vector<pacing_delivery_t> deliveries;
    pacing_event_t last_output, held_output;
    clock_t::time_point previous_finished {}, stop_requested {};
    std::int64_t next_stamp = 0;
    bool stop_case = false, next_published = false, next_observed_held = false;
    bool pending_image_alive = false, callback_failed = false;

    static void record(void *opaque, const probe::event_t &event) noexcept {
      auto &self = *static_cast<held_pacing_t *>(opaque);
      try {
        std::lock_guard lock {self.mutex};
        pacing_event_t copy;
        copy.kind = event.kind;
        copy.conversion = event.conversion;
        copy.model = event.model_frame;
        copy.target = event.literal_query_target_frame;
        copy.before = event.watermark_before;
        copy.after = event.watermark_after;
        copy.actual_pixels = event.actual_draw_pixel_source;
        copy.source = event.source_stamp_ticks;
        copy.current = event.current_source_stamp_ticks;
        copy.packed_source = event.pixel_source_stamp_ticks;
        copy.flat = event.flat;
        copy.repeat = event.repeat;
        copy.current_publication = event.publication_current;
        copy.submitted = event.force_infer || event.gpu_undecided;
        copy.held = event.gate_entered && !event.gate_released;
        copy.queried_busy = event.query_observed && event.literal_last_query_result == CUDA_ERROR_NOT_READY;
        copy.packed_valid = event.packed_valid;
        copy.route_matches = event.route_matches;
        for (std::size_t slot = 0; slot < event.slot_frames.size(); ++slot) {
          copy.pending_target |= event.slot_pending[slot] && event.slot_frames[slot] == copy.target;
        }
        self.events.push_back(copy);
        if (event.kind != probe::event_e::convert_end && event.kind != probe::event_e::packed_repeat) {
          return;
        }
        self.last_output = copy;
        if (!self.next_published && copy.current == self.held_source.load() &&
            copy.held && copy.queried_busy && copy.pending_target) {
          self.held_output = copy;
          const auto captured = clock_t::now();
          self.next_stamp = captured.time_since_epoch().count();
          self.next_capture->frame_timestamp = self.next_capture->content_timestamp = captured;
          self.next_published = true;
          self.images->raise(std::move(self.next_capture));
        } else if (self.next_published && copy.current == self.next_stamp &&
                   copy.held && copy.queried_busy && copy.pending_target) {
          self.next_observed_held = true;
          self.pending_image_alive = !self.held_image.expired();
          if (self.stop_case && self.stop_requested == clock_t::time_point {}) {
            // The owner has actually delivered C with B still held. Normal lifecycle checks
            // must win before encoding C; no test-thread sleep or scheduling race is required.
            self.stop_requested = clock_t::now();
            self.mail->event<bool>(mail::shutdown)->raise(true);
            self.images->wake();
          }
        }
      } catch (...) {
        std::lock_guard lock {self.mutex};
        self.callback_failed = true;
      }
    }

    static void record_poll(void *opaque, const fixture::native_pacing_poll_t &poll) noexcept {
      auto &self = *static_cast<held_pacing_t *>(opaque);
      try {
        std::lock_guard lock {self.mutex};
        self.polls.push_back(poll);
      } catch (...) {
        std::lock_guard lock {self.mutex};
        self.callback_failed = true;
      }
    }

    static void record_delivery(void *opaque, const fixture::native_pacing_delivery_t &call) noexcept {
      auto &self = *static_cast<held_pacing_t *>(opaque);
      try {
        std::lock_guard lock {self.mutex};
        self.deliveries.push_back({call, self.previous_finished, self.last_output});
        self.previous_finished = call.finished;
      } catch (...) {
        std::lock_guard lock {self.mutex};
        self.callback_failed = true;
      }
    }

    // processing_started is sampled immediately before the production convert call. Together
    // with consecutive real call boundaries it identifies a packet's exact conversion, even
    // when an older flat packet with the same content stamp retires after a later depth draw.
    bool is_matched_packet(const video::packet_t &packet, std::int64_t stamp) const {
      if (!packet->processing_started || !packet->content_timestamp ||
          packet->content_timestamp->time_since_epoch().count() != stamp) {
        return false;
      }
      return std::any_of(deliveries.begin(), deliveries.end(), [&](const auto &delivery) {
        const auto &output = delivery.output;
        return delivery.owns_packet(*packet) && output.source == stamp && output.model != 0 &&
               !output.flat && !output.repeat && output.current_publication;
      });
    }
  };

  class LiveHostSbsHeldDepthGpuTest: public ::testing::TestWithParam<bool> {};

  TEST_P(LiveHostSbsHeldDepthGpuTest, ExpiredDeliveryNewCaptureAndStopUseRealNativeOwnership) {
    const auto *enabled = std::getenv("APOLLO_RUN_TENSORRT_TESTS");
    if (!enabled || std::string_view {enabled} != "1") {
      GTEST_SKIP() << "Set APOLLO_RUN_TENSORRT_TESTS=1 for the local NVIDIA/TensorRT integration check.";
    }
    const auto adapter = nvidia_adapter();
    ASSERT_TRUE(adapter);
    constexpr unsigned width = 1280, height = 720;
    held_pacing_t observed;
    observed.stop_case = GetParam();
    observed.mail = std::make_shared<safe::mail_raw_t>();
    observed.images = std::make_shared<video::img_event_t::element_type>();
    fixture::native_pacing_probe_t binding;
    binding.held_source_stamp_ticks = &observed.held_source;
    binding.hold_ms = 120;
    binding.converter_observer = {&observed, held_pacing_t::record};
    binding.record_poll = held_pacing_t::record_poll;
    binding.record_delivery = held_pacing_t::record_delivery;
    auto display = fixture::make_native_pacing_display(*adapter, width, height, binding);
    auto packets = std::make_shared<video::packet_queue_t::element_type>();
    auto shutdown = observed.mail->event<bool>(mail::shutdown);
    video::config_t config {};
    config.width = 2 * width;
    config.height = height;
    config.framerate = 90;
    config.framerateX100 = 9000;
    config.encodingFramerate = 90000;
    config.bitrate = 20000;
    config.slicesPerFrame = 1;
    config.numRefFrames = 1;
    config.encoderCscMode = 2;
    config.videoFormat = 1;
    config.sbs_mode = video::SBS_AI;
    config.sbs_depth_status_event = observed.mail->event<int>(mail::sbs_depth_status);
    config.sbs_depth_pipeline_ready_event = std::make_shared<safe::event_t<bool>>();

    std::vector<std::uint32_t> pixels(std::size_t(width) * height);
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        pixels[std::size_t(y) * width + x] = (x / 64 + y / 48) % 2 ? 0xff3853a0u : 0xffa48b45u;
      }
    }
    auto first = fixture::upload_native_pacing_image(*display, pixels);
    for (auto &pixel : pixels) pixel ^= 0x00569662u;
    auto second = fixture::upload_native_pacing_image(*display, pixels);
    for (auto &pixel : pixels) pixel ^= 0x00a8531du;
    observed.next_capture = fixture::upload_native_pacing_image(*display, pixels);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(observed.next_capture);
    models::live_model_preparation_t preparation(video::host_sbs_v2_depth_model(), {});
    video::detail::native_encode_loop_test_result_t result;
    std::exception_ptr loop_error;
    std::atomic<bool> loop_finished {false};
    std::vector<video::packet_t> received;
    std::int64_t last_index = 0;
    const auto collect = [&](video::packet_t packet) {
      EXPECT_GT(packet->data_size(), 0u);
      EXPECT_GT(packet->frame_index(), last_index);
      last_index = packet->frame_index();
      received.push_back(std::move(packet));
    };
    const auto has_matched_packet = [&](std::int64_t stamp) {
      std::lock_guard lock {observed.mutex};
      return std::any_of(received.begin(), received.end(), [&](const auto &packet) {
        return observed.is_matched_packet(packet, stamp);
      });
    };
    const auto await = [&](auto complete, auto limit) {
      while (!complete() && clock_t::now() < limit && !loop_finished.load()) {
        if (auto packet = packets->pop(20ms)) collect(std::move(packet));
      }
      return complete();
    };
    const auto first_stamp = clock_t::now();
    first->frame_timestamp = first->content_timestamp = first_stamp;
    const auto first_ticks = first_stamp.time_since_epoch().count();
    {
      std::jthread encoder([&](std::stop_token stop) {
        std::stop_callback stop_loop(stop, [&]() { shutdown->raise(true); observed.images->wake(); });
        try {
          result = video::detail::run_native_encode_loop_for_test(display, observed.mail, observed.images, config, packets);
        } catch (...) {
          loop_error = std::current_exception();
        }
        loop_finished.store(true);
      });
      observed.images->raise(std::move(first));
      const bool warm = await([&]() { return has_matched_packet(first_ticks); }, clock_t::now() + 30s);
      EXPECT_TRUE(warm) << "A must reach an authenticated draw and its exact native packet before arming B.";
      if (warm) {
        const auto captured = clock_t::now();
        second->frame_timestamp = second->content_timestamp = captured;
        {
          std::lock_guard lock {observed.mutex};
          observed.held_image = second;
          observed.held_source.store(captured.time_since_epoch().count());
        }
        observed.images->raise(std::move(second));
        const bool completed = await([&]() {
          if (observed.stop_case) return loop_finished.load();
          std::int64_t stamp;
          { std::lock_guard lock {observed.mutex}; stamp = observed.next_stamp; }
          return stamp != 0 && has_matched_packet(stamp);
        }, clock_t::now() + 5s);
        EXPECT_TRUE(completed);
      }
      encoder.request_stop();
    }
    // The real loop has drained and joined retrieval. Keep the queue running until its final
    // submitted pictures have been collected; queue_t::stop() deliberately hides queued values.
    while (auto packet = packets->pop(0ms)) collect(std::move(packet));
    packets->stop();
    if (loop_error) std::rethrow_exception(loop_error);
    EXPECT_TRUE(result.session_created);
    EXPECT_EQ(result.pipeline_depth, 2u);

    std::lock_guard lock {observed.mutex};
    EXPECT_FALSE(observed.callback_failed);
    EXPECT_TRUE(observed.next_published);
    EXPECT_TRUE(observed.next_observed_held);
    EXPECT_TRUE(observed.pending_image_alive) << "B must remain owned after the source slot advances to C.";
    EXPECT_TRUE(observed.held_image.expired()) << "The joined loop teardown must retire B's final owner.";
    const auto &held = observed.held_output;
    EXPECT_TRUE(held.held && held.queried_busy && held.pending_target);
    EXPECT_NE(held.target, 0u);
    const auto expired = std::find_if(observed.polls.begin(), observed.polls.end(), [&](const auto &poll) {
      return poll.conversion == held.conversion && !poll.ready &&
             (poll.timed_out || !poll.eligible) && poll.query_count > 0;
    });
    EXPECT_NE(expired, observed.polls.end()) << "The held source must survive a real expired/ineligible joined query.";
    if (held.repeat) {
      EXPECT_EQ(held.before, held.after);
      EXPECT_TRUE(held.packed_valid && held.route_matches);
      EXPECT_EQ(held.packed_source, first_ticks);
    } else if (!held.flat) {
      // A can finish at the prior poll before B is admitted. The encoded delivery
      // may then legitimately draw that exact older source from its completed slot.
      EXPECT_EQ(held.source, first_ticks);
      EXPECT_GT(held.model, 0u);
      EXPECT_LT(held.model, held.target);
      EXPECT_TRUE(held.current_publication);
    }
    const auto fallback_delivery = std::find_if(
      observed.deliveries.begin(), observed.deliveries.end(), [&](const auto &delivery) {
        return delivery.output.conversion == held.conversion && delivery.call.source_timestamp &&
               delivery.call.source_timestamp->time_since_epoch().count() == observed.held_source.load();
      }
    );
    ASSERT_NE(fallback_delivery, observed.deliveries.end());
    EXPECT_TRUE(std::any_of(received.begin(), received.end(), [&](const auto &packet) {
      const auto expected = held.flat ? observed.held_source.load() : first_ticks;
      return fallback_delivery->owns_packet(*packet) && packet->content_timestamp &&
             packet->content_timestamp->time_since_epoch().count() == expected;
    })) << "The held B delivery must publish the selected fallback as an actual native packet.";
    std::uint64_t watermark = 0;
    for (const auto &event : observed.events) {
      if (event.kind == probe::event_e::convert_end || event.kind == probe::event_e::packed_repeat) {
        EXPECT_GE(event.after, event.before);
        EXPECT_GE(event.after, watermark);
        watermark = event.after;
      }
    }
    if (observed.stop_case) {
      EXPECT_NE(observed.stop_requested, clock_t::time_point {});
      for (const auto &packet : received) {
        EXPECT_NE(packet->content_timestamp, std::optional {clock_t::time_point {clock_t::duration {observed.next_stamp}}});
      }
      for (const auto &delivery : observed.deliveries) EXPECT_LE(delivery.call.started, observed.stop_requested);
      EXPECT_FALSE(std::any_of(observed.events.begin(), observed.events.end(), [&](const auto &event) {
        return event.kind == probe::event_e::ack && event.model == held.target;
      }));
    } else {
      const auto ack = std::find_if(observed.events.begin(), observed.events.end(), [&](const auto &event) {
        return event.kind == probe::event_e::ack && event.model == held.target &&
               event.source == observed.held_source.load() && event.current_publication;
      });
      EXPECT_NE(ack, observed.events.end()) << "Only an authentic joined completion may retire B.";
      const bool b_packet = std::any_of(received.begin(), received.end(), [&](const auto &packet) {
        return observed.is_matched_packet(packet, observed.held_source.load());
      });
      const bool b_drawn = std::any_of(observed.deliveries.begin(), observed.deliveries.end(), [&](const auto &delivery) {
        const auto &output = delivery.output;
        return output.model == held.target && output.source == observed.held_source.load() &&
               !output.flat && !output.repeat && output.current_publication;
      });
      EXPECT_EQ(b_packet, b_drawn);
      if (!b_drawn && ack != observed.events.end()) {
        EXPECT_TRUE(std::any_of(observed.events.begin(), ack, [&](const auto &event) {
          return event.current == observed.next_stamp && event.flat && event.actual_pixels > held.target;
        })) << "A newer C draw, rather than a stale B delivery, must explain skipping B.";
      }
      EXPECT_TRUE(std::any_of(received.begin(), received.end(), [&](const auto &packet) {
        return observed.is_matched_packet(packet, observed.next_stamp);
      }));
      bool b_retired = false;
      for (const auto &event : observed.events) {
        b_retired |= event.kind == probe::event_e::ack && event.model == held.target;
        if (event.kind == probe::event_e::admission && event.current == observed.next_stamp && event.submitted) {
          EXPECT_TRUE(b_retired) << "C cannot enqueue while the exact B root is still pending.";
        }
      }
    }
    std::cout << "Live Host SBS held-depth arm=" << (observed.stop_case ? "stop" : "complete")
              << " packets=" << received.size() << " B_model=" << held.target
              << " C_held=" << observed.next_observed_held << " pending_owner=" << observed.pending_image_alive
              << " fallback=" << (held.repeat ? "packed-A" : held.flat ? "flat-B" : "matched-A") << '\n';
    for (const auto &poll : observed.polls) {
      if (poll.conversion < held.conversion) continue;
      std::cout << "Live Host SBS held-depth poll conversion=" << poll.conversion
                << " prior=" << poll.prior << " eligible=" << poll.eligible
                << " ready=" << poll.ready << " expired=" << poll.timed_out
                << " queries=" << poll.query_count << " wait_us="
                << std::chrono::duration_cast<std::chrono::microseconds>(poll.wait_duration).count() << '\n';
    }
  }

  INSTANTIATE_TEST_SUITE_P(
    RealNativeLoop, LiveHostSbsHeldDepthGpuTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool> &info) { return info.param ? "StopWhileHeld" : "Complete"; }
  );
}  // namespace
#endif

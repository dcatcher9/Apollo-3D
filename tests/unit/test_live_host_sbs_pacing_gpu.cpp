#include <gtest/gtest.h>

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
  #include "src/globals.h"
  #include "src/live_model_preparation.h"
  #include "src/platform/windows/display.h"
  #include "src/platform/windows/model_preparation_converter_test.h"
  #include "src/video_capture_test.h"

  #include <chrono>
  #include <cstdint>
  #include <cstdlib>
  #include <exception>
  #include <iostream>
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
}  // namespace
#endif

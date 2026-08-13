/**
 * @file tests/unit/platform/test_sbs_debug_dump_async.cpp
 * @brief Behavioral tests for Dump 3D publication lifetime and request ordering.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <atomic>
  #include <array>
  #include <bit>
  #include <chrono>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <future>
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

  #include <src/generated/depth_coordinate_v2_contract.h>
  #include <src/platform/windows/sbs_debug_dump.h>
  #include <src/platform/windows/sbs_debug_dump_async.h>

namespace {
  using namespace std::chrono_literals;
  namespace dump_detail = platf::sbs_debug::detail;
  namespace v2 = models::depth_coordinate_v2;

  template<std::size_t WordCount>
  std::vector<std::uint8_t> word_bytes(const std::array<std::uint32_t, WordCount> &words) {
    std::vector<std::uint8_t> bytes(sizeof(words));
    std::memcpy(bytes.data(), words.data(), sizeof(words));
    return bytes;
  }

  platf::sbs_debug::frame subtitle_frame(
    const models::depth_tensor_content_rect_t content,
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    platf::sbs_debug::frame frame;
    frame.model_width = 770;
    frame.model_height = 434;
    frame.matched_frame_id = 41u;
    frame.depth_input_region = {
      .source_width = 3840u,
      .source_height = 2160u,
      .left = 0u,
      .top = 0u,
      .right = source_width,
      .bottom = source_height,
      .tensor_content = content,
      .analysis_generation = 17u,
      .video_region = true,
      .authority = models::depth_analysis_authority_e::foreground_client,
    };
    return frame;
  }

  std::array<std::uint32_t, v2::subtitle_ocr_record_word_count> empty_ocr(
    const platf::sbs_debug::frame &frame,
    const models::subtitle_analysis_geometry_t geometry
  ) {
    std::array<std::uint32_t, v2::subtitle_ocr_record_word_count> words {};
    words[0] = v2::subtitle_ocr_record_schema;
    words[1] = v2::subtitle_ocr_record_tag;
    words[5] = static_cast<std::uint32_t>(frame.matched_frame_id);
    words[6] = static_cast<std::uint32_t>(frame.matched_frame_id >> 32u);
    words[7] = static_cast<std::uint32_t>(frame.depth_input_region.analysis_generation);
    words[8] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation >> 32u
    );
    words[9] = frame.depth_input_region.width();
    words[10] = frame.depth_input_region.height();
    words[11] = static_cast<std::uint32_t>(frame.model_width);
    words[12] = static_cast<std::uint32_t>(frame.model_height);
    words[13] = geometry.roi_top;
    words[14] = geometry.roi_bottom;
    return words;
  }

  std::array<std::uint32_t, v2::subtitle_locator_state_word_count> cached_grace_state(
    const platf::sbs_debug::frame &frame,
    const models::subtitle_analysis_geometry_t geometry,
    const std::uint32_t grace
  ) {
    std::array<std::uint32_t, v2::subtitle_locator_state_word_count> words {};
    words[0] = v2::subtitle_locator_state_schema;
    words[1] = v2::subtitle_locator_state_tag;
    words[10] = static_cast<std::uint32_t>(frame.depth_input_region.analysis_generation);
    words[11] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation >> 32u
    );
    words[18] = std::bit_cast<std::uint32_t>(0.006f);
    words[22] = static_cast<std::uint32_t>(frame.matched_frame_id);
    words[23] = static_cast<std::uint32_t>(frame.matched_frame_id >> 32u);
    words[25] = grace;
    words[26] = 3u;
    words[27] = static_cast<std::uint32_t>(frame.model_width);
    words[28] = static_cast<std::uint32_t>(frame.model_height);
    constexpr std::uint32_t left = 120u;
    constexpr std::uint32_t right = 650u;
    const std::uint32_t top = geometry.roi_top + 1u;
    const std::uint32_t bottom = geometry.roi_bottom - 1u;
    words[29] = left | (right << 16u);
    words[30] = top | (bottom << 16u);
    return words;
  }

  class scoped_dump_trigger_environment {
  public:
    scoped_dump_trigger_environment():
        diagnostics_enabled_(config::sunshine.diagnostics_enabled) {
      if (const char *value = std::getenv("APOLLO_SBS_DUMP")) {
        previous_directory_ = value;
      }
      root_ = std::filesystem::temp_directory_path() /
              ("apollo_dump_trigger_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()
              ));
      std::error_code error;
      ready_ = std::filesystem::create_directory(root_, error) && !error;
      if (ready_) {
        std::ofstream trigger(root_ / "dump.trigger", std::ios::binary);
        trigger << "dump\n";
        ready_ = trigger.good() && _putenv_s("APOLLO_SBS_DUMP", root_.string().c_str()) == 0;
      }
      config::sunshine.diagnostics_enabled = true;
    }

    ~scoped_dump_trigger_environment() {
      config::sunshine.diagnostics_enabled = diagnostics_enabled_;
      _putenv_s(
        "APOLLO_SBS_DUMP",
        previous_directory_ ? previous_directory_->c_str() : ""
      );
      std::error_code error;
      std::filesystem::remove_all(root_, error);
    }

    bool ready() const noexcept {
      return ready_;
    }

  private:
    bool diagnostics_enabled_ = false;
    std::optional<std::string> previous_directory_;
    std::filesystem::path root_;
    bool ready_ = false;
  };

  TEST(SbsDebugDumpAsyncTest, ReleasingSessionDoesNotWaitForPublication) {
    auto state = dump_detail::publication_state::create();
    std::weak_ptr<dump_detail::publication_state> weak_state = state;

    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();
    std::promise<void> finished_promise;
    auto finished = finished_promise.get_future();

    ASSERT_TRUE(state->enqueue([&started_promise, release, &finished_promise] {
      started_promise.set_value();
      release.wait();
      finished_promise.set_value();
    }));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);

    const auto begin = std::chrono::steady_clock::now();
    state.reset();
    const auto release_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - begin
    ).count();
    EXPECT_LT(release_duration_ms, 250);
    EXPECT_FALSE(weak_state.expired());

    release_promise.set_value();
    ASSERT_EQ(finished.wait_for(2s), std::future_status::ready);
  }

  TEST(SbsDebugDumpAsyncTest, SingleFlightRecoversAfterCompletionAndException) {
    auto state = dump_detail::publication_state::create();
    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();

    ASSERT_TRUE(state->enqueue([&started_promise, release] {
      started_promise.set_value();
      release.wait();
    }));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);
    EXPECT_TRUE(state->busy());
    EXPECT_FALSE(state->enqueue([] {}));

    release_promise.set_value();
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(state->busy());

    ASSERT_TRUE(state->enqueue([] { throw 7; }));
    EXPECT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(state->busy());
  }

  TEST(SbsDebugDumpAsyncTest, CancellationEpochRejectsAnOlderFailure) {
    auto state = dump_detail::publication_state::create();
    auto button = std::make_shared<std::atomic<bool>>(false);
    const auto old_token = state->allow_retries_and_token();

    std::promise<void> started_promise;
    auto started = started_promise.get_future();
    std::promise<void> release_promise;
    auto release = release_promise.get_future().share();
    std::promise<bool> rearmed_promise;
    auto rearmed = rearmed_promise.get_future();
    ASSERT_TRUE(state->enqueue(
      [state, button, old_token, &started_promise, release, &rearmed_promise] {
        started_promise.set_value();
        release.wait();
        rearmed_promise.set_value(
          state->record_publication_failure(old_token, true, true, button)
        );
      }
    ));
    ASSERT_EQ(started.wait_for(2s), std::future_status::ready);

    state->cancel_retries(button);
    const auto new_token = state->allow_retries_and_token();
    EXPECT_NE(new_token, old_token);
    release_promise.set_value();

    ASSERT_EQ(rearmed.wait_for(2s), std::future_status::ready);
    EXPECT_FALSE(rearmed.get());
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(button->load());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_FALSE(state->take_file_retry_pending());
  }

  TEST(SbsDebugDumpAsyncTest, CurrentFailureRearmsExactlyOnceAndSuccessDoesNot) {
    auto state = dump_detail::publication_state::create();
    auto button = std::make_shared<std::atomic<bool>>(false);
    const auto token = state->allow_retries_and_token();

    ASSERT_TRUE(state->record_publication_failure(token, true, true, button));
    EXPECT_TRUE(button->exchange(false));
    EXPECT_TRUE(state->take_publication_failed());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_TRUE(state->take_file_retry_pending());
    EXPECT_FALSE(state->take_file_retry_pending());

    std::promise<void> success_promise;
    auto success = success_promise.get_future();
    ASSERT_TRUE(state->enqueue([&success_promise] { success_promise.set_value(); }));
    ASSERT_EQ(success.wait_for(2s), std::future_status::ready);
    ASSERT_TRUE(state->wait_idle_for(2s));
    EXPECT_FALSE(button->load());
    EXPECT_FALSE(state->take_publication_failed());
    EXPECT_FALSE(state->take_file_retry_pending());
  }

  TEST(SbsDebugDumpAsyncTest, ButtonGuardPreservesALaterClick) {
    auto button = std::make_shared<std::atomic<bool>>(true);
    {
      dump_detail::button_request_guard request(button);
      ASSERT_TRUE(request.consumed());
      EXPECT_FALSE(button->load());
      button->store(true);
      request.commit();
    }
    EXPECT_TRUE(button->load());

    button->store(true);
    {
      dump_detail::button_request_guard request(button);
      ASSERT_TRUE(request.consumed());
      EXPECT_FALSE(button->load());
      // An uncommitted validation/capture/enqueue failure restores the consumed click.
    }
    EXPECT_TRUE(button->load());
  }

  TEST(SbsDebugDumpAsyncTest, LatchedButtonWakesRetainedSource) {
    platf::sbs_debug::dumper dumper;
    auto button = std::make_shared<std::atomic<bool>>(false);
    dumper.set_button_request(button);

    EXPECT_FALSE(dumper.needs_conversion_poll());
    button->store(true, std::memory_order_release);
    EXPECT_TRUE(dumper.needs_conversion_poll());

    dumper.cancel_pending_request();
    EXPECT_FALSE(dumper.needs_conversion_poll());
  }

  TEST(SbsDebugDumpAsyncTest, ExistingFileTriggerWakesRetainedSource) {
    scoped_dump_trigger_environment environment;
    ASSERT_TRUE(environment.ready());

    platf::sbs_debug::dumper dumper;
    EXPECT_TRUE(dumper.needs_conversion_poll());
  }

  TEST(SbsDebugDumpAsyncTest, CollectionChunksAreBoundedAlignedAndComplete) {
    constexpr std::size_t alignment = sizeof(float);
    static_assert(dump_detail::cpu_collection_byte_budget == 64u * 1024u * 1024u);
    constexpr std::size_t representative_4k_dump_bytes = 393u * 1024u * 1024u;
    std::size_t remaining = representative_4k_dump_bytes;
    std::size_t copied = 0;
    unsigned polls = 0;
    while (remaining != 0) {
      const std::size_t chunk = dump_detail::bounded_collection_chunk_bytes(
        remaining,
        alignment,
        dump_detail::cpu_collection_byte_budget,
        true
      );
      ASSERT_NE(chunk, 0u);
      EXPECT_EQ(chunk % alignment, 0u);
      EXPECT_LE(chunk, dump_detail::cpu_collection_byte_budget);
      copied += chunk;
      remaining -= chunk;
      ++polls;
    }
    EXPECT_EQ(copied, representative_4k_dump_bytes);
    EXPECT_EQ(polls, 7u);
    EXPECT_LT(polls, 99u);  // The former 4 MiB budget needed about 99 polls.
  }

  TEST(SbsDebugDumpAsyncTest, WideTextureRowGetsOneEmptyPollOvershoot) {
    constexpr std::size_t wide_row =
      dump_detail::cpu_collection_byte_budget + 256u;
    EXPECT_EQ(
      dump_detail::bounded_collection_chunk_bytes(
        wide_row * 2u,
        wide_row,
        dump_detail::cpu_collection_byte_budget,
        true
      ),
      wide_row
    );
    EXPECT_EQ(
      dump_detail::bounded_collection_chunk_bytes(
        wide_row * 2u,
        wide_row,
        dump_detail::cpu_collection_byte_budget,
        false
      ),
      0u
    );
    EXPECT_EQ(
      dump_detail::bounded_collection_chunk_bytes(7u, 4u, 4u, true),
      0u
    );
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationUsesVerticallyPaddedContent) {
    const auto frame = subtitle_frame({0u, 0u, 770u, 433u}, 2536u, 1427u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, frame.depth_input_region.tensor_content
    );
    ASSERT_TRUE(geometry.valid());
    ASSERT_EQ(geometry.roi_top, 324u);
    ASSERT_EQ(geometry.ribbon_min_bottom, 428u);
    ASSERT_EQ(geometry.roi_bottom, 429u);

    auto ocr = empty_ocr(frame, geometry);
    EXPECT_TRUE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(ocr), frame
    ));

    // The pre-Dump29 whole-field validator expected these rows and rejected the real record.
    ocr[13] = 325u;
    ocr[14] = 430u;
    EXPECT_FALSE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(ocr), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationConfinesRibbonToContent) {
    constexpr models::depth_tensor_content_rect_t content {111u, 0u, 659u, 434u};
    const auto frame = subtitle_frame(content, 1862u, 1473u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());

    auto ocr = empty_ocr(frame, geometry);
    ocr[2] = 1u;
    ocr[3] = 1u;
    ocr[4] = 1u;
    const std::uint32_t core_left = content.left + 20u;
    const std::uint32_t core_right = content.right - 20u;
    const std::uint32_t core_top = geometry.ribbon_min_bottom - 10u;
    const std::uint32_t core_bottom = geometry.ribbon_min_bottom + 1u;
    for (const auto offset : {v2::subtitle_ocr_raw_box_offset,
                              v2::subtitle_ocr_final_box_offset}) {
      ocr[offset] = core_left;
      ocr[offset + 1u] = core_top;
      ocr[offset + 2u] = core_right;
      ocr[offset + 3u] = core_bottom;
      ocr[offset + 4u] = std::bit_cast<std::uint32_t>(0.9f);
      ocr[offset + 5u] = v2::subtitle_ocr_box_flag_ribbon;
      ocr[offset + 6u] = v2::subtitle_ocr_ribbon_min_structural_gaps + 1u;
      ocr[offset + 7u] = v2::subtitle_ocr_ribbon_min_structural_gaps;
    }
    ocr[v2::subtitle_ocr_final_box_offset] = content.left;
    ocr[v2::subtitle_ocr_final_box_offset + 2u] = content.right;
    ocr[v2::subtitle_ocr_final_box_offset + 3u] = content.bottom;

    EXPECT_TRUE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(ocr), frame
    ));

    // A whole-field ribbon cover is outside the authenticated pillarbox content.
    ocr[v2::subtitle_ocr_final_box_offset] = 0u;
    ocr[v2::subtitle_ocr_final_box_offset + 2u] = 770u;
    EXPECT_FALSE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(ocr), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationAuthenticatesDeathGraceLimit) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());
    const auto ocr = empty_ocr(frame, geometry);
    auto locator = cached_grace_state(
      frame, geometry, v2::subtitle_locator_death_grace_observations
    );
    EXPECT_TRUE(dump_detail::subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));

    locator[25] = v2::subtitle_locator_death_grace_observations + 1u;
    EXPECT_FALSE(dump_detail::subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));
  }
}  // namespace
#endif

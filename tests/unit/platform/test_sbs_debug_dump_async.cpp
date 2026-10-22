/**
 * @file tests/unit/platform/test_sbs_debug_dump_async.cpp
 * @brief Behavioral tests for Dump 3D publication lifetime and request ordering.
 */
#ifdef _WIN32
  // Boost.Asio must precede Windows headers included by the test and D3D helpers.
  #include <src/stream.h>
#endif

#include "../../tests_common.h"

#ifdef _WIN32
  #include <algorithm>
  #include <atomic>
  #include <array>
  #include <bit>
  #include <chrono>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <future>
  #include <iterator>
  #include <limits>
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

  #include <nlohmann/json.hpp>

  #include <src/generated/depth_coordinate_v2_contract.h>
  #include <src/depth_coordinate_v2.h>
  #include <src/host_sbs_gpu_trace.h>
  #include <src/host_sbs_subtitle_plane.h>
  #include <src/host_sbs_shader_cache.h>
  #include <src/platform/windows/sbs_debug_dump.h>
  #include <src/platform/windows/sbs_debug_dump_async.h>
  #include <src/prod_zipdepth_convex2x.h>

namespace {
  using namespace std::chrono_literals;
  namespace dump_detail = platf::sbs_debug::detail;
  namespace v2 = models::depth_coordinate_v2;
  constexpr std::uint32_t subtitle_scene_epoch = 3u;
  namespace gpu_trace = models::host_sbs_gpu_trace;

  template<std::size_t WordCount>
  std::vector<std::uint8_t> word_bytes(const std::array<std::uint32_t, WordCount> &words) {
    std::vector<std::uint8_t> bytes(sizeof(words));
    std::memcpy(bytes.data(), words.data(), sizeof(words));
    return bytes;
  }

  std::vector<std::uint8_t> repeated_word_bytes(const std::uint32_t value, const std::size_t count) {
    const std::vector<std::uint32_t> words(count, value);
    std::vector<std::uint8_t> bytes(count * sizeof(value));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
  }

  bool subtitle_records_match_frame(
    const std::vector<std::uint8_t> &ocr,
    const std::vector<std::uint8_t> &locator,
    const platf::sbs_debug::frame &frame
  ) {
    return dump_detail::subtitle_records_match_frame(
      ocr, locator, frame, subtitle_scene_epoch
    );
  }

  platf::sbs_debug::frame subtitle_frame(
    const models::depth_tensor_content_rect_t content,
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    platf::sbs_debug::frame frame;
    frame.model_width = 770;
    frame.model_height = 434;
    frame.raw_width = 770;
    frame.raw_height = 434;
    frame.field_width = 770;
    frame.field_height = 434;
    frame.field_content = content;
    frame.matched_frame_id = 41u;
    frame.matched_observation_timestamp_us = 1'000'000u + frame.matched_frame_id;
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.0f);
    frame.depth_input_region = {
      .source_width = 3840u,
      .source_height = 2160u,
      .left = 0u,
      .top = 0u,
      .right = source_width,
      .bottom = source_height,
      .tensor_content = content,
      .analysis_generation = 17u,
      .authority = models::depth_analysis_authority_e::foreground_client,
    };
    return frame;
  }

  platf::sbs_debug::frame gpu_trace_frame() {
    platf::sbs_debug::frame frame;
    frame.model_width = 770;
    frame.model_height = 434;
    frame.raw_width = 770;
    frame.raw_height = 434;
    frame.field_width = 770;
    frame.field_height = 434;
    frame.field_content = {0u, 0u, 770u, 434u};
    frame.matched_frame_id = 41u;
    frame.matched_observation_timestamp_us = 1'000'000u + frame.matched_frame_id;
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.0f);
    frame.color_space = models::input_color_space::srgb;
    frame.depth_input_region = {
      .source_width = 1920u,
      .source_height = 1080u,
      .left = 0u,
      .top = 0u,
      .right = 1920u,
      .bottom = 1080u,
      .tensor_content = {0u, 0u, 770u, 434u},
      .analysis_generation = 0u,
      .authority = models::depth_analysis_authority_e::full_source,
    };
    return frame;
  }

  models::host_sbs_subtitle_plane::words_t current_subtitle_plane(
    const platf::sbs_debug::frame &frame, const std::uint32_t count,
    const bool current_valid = true, const float target = 0.001f
  ) {
    models::host_sbs_subtitle_plane::words_t words {};
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.field_width, frame.field_height}, frame.field_content);
    words[0u] = v2::subtitle_plane_state_schema;
    words[1u] = v2::subtitle_plane_state_tag;
    words[v2::subtitle_plane_flags_word] = current_valid ? v2::subtitle_plane_current_valid_flag : 0u;
    words[v2::subtitle_plane_current_slot_mask_word] = current_valid ? (1u << count) - 1u : 0u;
    const auto put64 = [&words](const std::size_t index, const std::uint64_t value) {
      words[index] = static_cast<std::uint32_t>(value);
      words[index + 1u] = static_cast<std::uint32_t>(value >> 32u);
    };
    put64(v2::subtitle_plane_frame_offset, frame.matched_frame_id);
    put64(v2::subtitle_plane_analysis_generation_offset, frame.depth_input_region.analysis_generation);
    put64(v2::subtitle_plane_observation_timestamp_offset, frame.matched_observation_timestamp_us);
    words[v2::subtitle_plane_source_extent_offset] = frame.depth_input_region.width();
    words[v2::subtitle_plane_source_extent_offset + 1u] = frame.depth_input_region.height();
    words[v2::subtitle_plane_field_extent_offset] = static_cast<std::uint32_t>(frame.field_width);
    words[v2::subtitle_plane_field_extent_offset + 1u] = static_cast<std::uint32_t>(frame.field_height);
    words[v2::subtitle_plane_roi_offset] = geometry.roi_top;
    words[v2::subtitle_plane_roi_offset + 1u] = geometry.roi_bottom;
    words[v2::subtitle_plane_content_offset] = frame.field_content.left;
    words[v2::subtitle_plane_content_offset + 1u] = frame.field_content.top;
    words[v2::subtitle_plane_content_offset + 2u] = frame.field_content.right;
    words[v2::subtitle_plane_content_offset + 3u] = frame.field_content.bottom;
    words[v2::subtitle_plane_cut_epoch_word] = subtitle_scene_epoch;
    const auto a = v2::subtitle_plane_adaptive_offset;
    words[a] = 1u; words[a + 1u] = 8u;
    words[a + 3u] = std::bit_cast<std::uint32_t>(target);
    put64(a + 4u, frame.matched_observation_timestamp_us);
    words[a + 12u] = std::bit_cast<std::uint32_t>(v2::display_budget());
    return words;
  }

  std::vector<std::uint8_t> canonical_gpu_trace_ring(
    const platf::sbs_debug::frame &frame,
    const gpu_trace::submission_class_e submission_class =
      gpu_trace::submission_class_e::force_infer,
    const cuda_conditional_graph::branch_e branch =
      cuda_conditional_graph::branch_e::infer,
    const bool label_authenticated_depth = true,
    const bool optional_child_present = true
  ) {
    std::vector<std::uint32_t> words(gpu_trace::ring_word_count, 0u);
    words[gpu_trace::word_index(gpu_trace::header_word_e::schema)] =
      gpu_trace::ring_schema;
    words[gpu_trace::word_index(gpu_trace::header_word_e::tag)] =
      gpu_trace::ring_tag;
    words[gpu_trace::word_index(gpu_trace::header_word_e::capacity)] =
      gpu_trace::capacity;
    words[gpu_trace::word_index(gpu_trace::header_word_e::record_words)] =
      gpu_trace::record_word_count;
    words[gpu_trace::word_index(gpu_trace::header_word_e::next_sequence_low)] = 2u;
    words[gpu_trace::word_index(gpu_trace::header_word_e::next_slot)] = 1u;
    words[gpu_trace::word_index(gpu_trace::header_word_e::committed_count)] = 1u;

    const auto base = gpu_trace::record_base(0u);
    const auto set = [&](const gpu_trace::record_word_e field, const std::uint32_t value) {
      words[base + gpu_trace::word_index(field)] = value;
    };
    constexpr std::uint64_t token = 0x1020304050607080ull;
    const auto request = cuda_conditional_graph::make_request(
      token, cuda_conditional_graph::work_flag_e::optional_ocr
    );
    const auto proposal = cuda_conditional_graph::make_proposal(branch, token);
    const auto receipt = cuda_conditional_graph::resolve_proposal(
      proposal, request, optional_child_present
    );
    set(gpu_trace::record_word_e::schema, gpu_trace::ring_schema);
    set(gpu_trace::record_word_e::commit_tag, gpu_trace::record_tag);
    set(gpu_trace::record_word_e::sequence_low, 1u);
    set(gpu_trace::record_word_e::frame_low,
        static_cast<std::uint32_t>(frame.matched_frame_id));
    set(gpu_trace::record_word_e::frame_high,
        static_cast<std::uint32_t>(frame.matched_frame_id >> 32u));
    set(gpu_trace::record_word_e::analysis_generation_low,
        static_cast<std::uint32_t>(frame.depth_input_region.analysis_generation));
    set(gpu_trace::record_word_e::analysis_generation_high,
        static_cast<std::uint32_t>(frame.depth_input_region.analysis_generation >> 32u));
    const auto domain = models::near_identical_input_domain_tag(
      frame.depth_input_region,
      frame.color_space,
      static_cast<std::uint32_t>(frame.model_width),
      static_cast<std::uint32_t>(frame.model_height)
    );
    set(gpu_trace::record_word_e::domain_tag_low, static_cast<std::uint32_t>(domain));
    set(gpu_trace::record_word_e::domain_tag_high, static_cast<std::uint32_t>(domain >> 32u));
    set(gpu_trace::record_word_e::transaction_token_low,
        static_cast<std::uint32_t>(token));
    set(gpu_trace::record_word_e::transaction_token_high,
        static_cast<std::uint32_t>(token >> 32u));
    set(gpu_trace::record_word_e::submission_class,
        static_cast<std::uint32_t>(submission_class));
    const auto expected_depth =
      submission_class == gpu_trace::submission_class_e::force_infer &&
        branch == cuda_conditional_graph::branch_e::reuse ?
        gpu_trace::depth_disposition_e::invalid :
      branch == cuda_conditional_graph::branch_e::reuse ?
        gpu_trace::depth_disposition_e::reuse : gpu_trace::depth_disposition_e::infer;
    set(gpu_trace::record_word_e::depth_disposition,
        label_authenticated_depth ? static_cast<std::uint32_t>(expected_depth) :
                                    static_cast<std::uint32_t>(gpu_trace::depth_disposition_e::reuse));
    set(gpu_trace::record_word_e::expected_work,
        cuda_conditional_graph::work_flags_value(
          cuda_conditional_graph::work_flag_e::optional_ocr));
    const bool branch_gated =
      submission_class == gpu_trace::submission_class_e::gpu_undecided;
    const bool subtitle_held =
      branch_gated && branch == cuda_conditional_graph::branch_e::reuse;
    const bool subtitle_current =
      branch == cuda_conditional_graph::branch_e::infer && optional_child_present;
    set(
      gpu_trace::record_word_e::subtitle_disposition,
      static_cast<std::uint32_t>(
        subtitle_held ? gpu_trace::subtitle_disposition_e::held_with_depth :
        subtitle_current ? gpu_trace::subtitle_disposition_e::optional_ocr :
                           gpu_trace::subtitle_disposition_e::abstention
      )
    );
    const auto flags =
      (branch_gated ? gpu_trace::subtitle_branch_gated :
                      gpu_trace::ocr_record_submitted | gpu_trace::condition_executed) |
      (frame.input_domain_reset ? gpu_trace::input_domain_reset : 0u);
    set(gpu_trace::record_word_e::flags, flags);
    set(gpu_trace::record_word_e::host_subtitle_outcome,
        static_cast<std::uint32_t>(gpu_trace::host_subtitle_outcome_e::ordinary_record));
    set(gpu_trace::record_word_e::source_width, frame.depth_input_region.width());
    set(gpu_trace::record_word_e::source_height, frame.depth_input_region.height());
    set(gpu_trace::record_word_e::field_width,
        static_cast<std::uint32_t>(frame.field_width));
    set(gpu_trace::record_word_e::field_height,
        static_cast<std::uint32_t>(frame.field_height));
    set(gpu_trace::record_word_e::transaction_words,
        gpu_trace::transaction_word_count);
    std::memcpy(
      words.data() + base + gpu_trace::word_index(
        gpu_trace::record_word_e::transaction_begin
      ),
      &receipt,
      sizeof(receipt)
    );
    std::memcpy(
      words.data() + base + gpu_trace::word_index(
        gpu_trace::record_word_e::transaction_begin
      ) + sizeof(receipt) / sizeof(std::uint32_t),
      &request,
      sizeof(request)
    );
    const auto locator =
      base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin);
    auto publication = frame;
    if (subtitle_held) {
      --publication.matched_frame_id;
      --publication.matched_observation_timestamp_us;
    }
    const auto target = std::bit_cast<std::uint32_t>(0.01f);
    const auto plane = current_subtitle_plane(
      publication, subtitle_current || subtitle_held ? 1u : 0u,
      subtitle_current || subtitle_held, 0.01f);
    std::copy(plane.begin(), plane.end(), words.begin() + locator);
    const auto condition = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_condition_begin);
    words[condition] = v2::subtitle_condition_param_schema;
    words[condition + 1u] = v2::subtitle_condition_param_tag;
    if (subtitle_current || subtitle_held) {
      words[condition + 2u] = 1u;
      words[condition + 4u] = v2::subtitle_condition_param_fully_active_value;
      words[condition + 5u] = target;
    }
    set(gpu_trace::record_word_e::observation_timestamp_low, static_cast<std::uint32_t>(frame.matched_observation_timestamp_us));
    set(gpu_trace::record_word_e::observation_timestamp_high, static_cast<std::uint32_t>(frame.matched_observation_timestamp_us >> 32u));

    std::vector<std::uint8_t> bytes(words.size() * sizeof(std::uint32_t));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
  }

  std::uint32_t trace_word(
    const std::vector<std::uint8_t> &bytes,
    const std::size_t index
  ) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
    return value;
  }

  void set_trace_word(
    std::vector<std::uint8_t> &bytes,
    const std::size_t index,
    const std::uint32_t value
  ) {
    std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
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
    words[11] = static_cast<std::uint32_t>(frame.field_width);
    words[12] = static_cast<std::uint32_t>(frame.field_height);
    words[13] = geometry.roi_top;
    words[14] = geometry.roi_bottom;
    return words;
  }

  using subtitle_rect_t = std::array<std::uint32_t, 4>;

  void store_ocr_pair(
    std::array<std::uint32_t, v2::subtitle_ocr_record_word_count> &words,
    const std::uint32_t slot,
    const subtitle_rect_t core,
    const subtitle_rect_t cover
  ) {
    const auto raw = v2::subtitle_ocr_raw_box_offset +
      static_cast<std::size_t>(slot) * v2::subtitle_ocr_box_word_count;
    const auto final = v2::subtitle_ocr_final_box_offset +
      static_cast<std::size_t>(slot) * v2::subtitle_ocr_box_word_count;
    for (std::size_t index = 0u; index < core.size(); ++index) {
      words[raw + index] = core[index];
      words[final + index] = cover[index];
    }
    words[raw + 4u] = std::bit_cast<std::uint32_t>(0.9f);
    words[raw + 5u] = 0u;
    words[raw + 6u] = 1u;
    words[raw + 7u] = 0u;
    std::copy_n(words.begin() + raw + 4u, 4u, words.begin() + final + 4u);
  }

  models::host_sbs_subtitle_plane::words_t two_line_subtitle_state(
    const platf::sbs_debug::frame &frame,
    const subtitle_rect_t first, const subtitle_rect_t second
  ) {
    auto words = current_subtitle_plane(frame, 2u);
    const auto overlap_width = std::max(0, static_cast<int>(std::min(first[2u], second[2u])) - static_cast<int>(std::max(first[0u], second[0u])));
    const auto overlap_height = std::max(0, static_cast<int>(std::min(first[3u], second[3u])) - static_cast<int>(std::max(first[1u], second[1u])));
    words[v2::subtitle_plane_adaptive_offset + 13u] =
      (first[2u] - first[0u]) * (first[3u] - first[1u]) +
      (second[2u] - second[0u]) * (second[3u] - second[1u]) -
      static_cast<std::uint32_t>(overlap_width * overlap_height);
    return words;
  }

  struct contour_dump_fixture {
    platf::sbs_debug::frame completed;
    std::array<std::uint32_t, v2::subtitle_ocr_record_word_count> ocr;
    std::array<std::uint32_t, v2::subtitle_plane_state_word_count> locator;
    std::array<std::uint32_t, v2::subtitle_contour_history_word_count> identity {};
    std::vector<float> state;
  };

  contour_dump_fixture current_contour_dump_fixture(
    const std::uint32_t first_width = 240u,
    const std::uint32_t second_width = 240u
  ) {
    contour_dump_fixture fixture;
    auto &completed = fixture.completed;
    completed = subtitle_frame({24u, 0u, 746u, 434u}, 1280u, 720u);
    completed.matched_frame_id = 0x200000041ull;
    completed.matched_observation_timestamp_us = 0x300000001ull;
    completed.depth_input_region.analysis_generation = 0x400000011ull;
    const auto geometry = models::fit_subtitle_analysis_geometry(
      completed.depth_input_region.width(), completed.depth_input_region.height(),
      {completed.field_width, completed.field_height}, completed.field_content);
    const subtitle_rect_t first {120u, geometry.roi_top + 2u, 120u + first_width, geometry.roi_bottom - 4u};
    const subtitle_rect_t second {450u, geometry.roi_top + 3u, 450u + second_width, geometry.roi_bottom - 3u};
    fixture.ocr = empty_ocr(completed, geometry);
    fixture.ocr[2u] = 1u;
    fixture.ocr[3u] = fixture.ocr[4u] = 2u;
    store_ocr_pair(fixture.ocr, 0u, first, first);
    store_ocr_pair(fixture.ocr, 1u, second, second);
    fixture.locator = two_line_subtitle_state(completed, first, second);
    constexpr auto adaptive = v2::subtitle_plane_adaptive_offset;
    fixture.locator[adaptive + 4u] = static_cast<std::uint32_t>(completed.matched_observation_timestamp_us);
    fixture.locator[adaptive + 5u] = static_cast<std::uint32_t>(completed.matched_observation_timestamp_us >> 32u);

    auto &identity = fixture.identity;
    identity[0u] = v2::subtitle_contour_schema;
    identity[1u] = v2::subtitle_contour_history_tag;
    identity[2u] = 1u;
    identity[3u] = subtitle_scene_epoch;
    identity[4u] = static_cast<std::uint32_t>(completed.matched_frame_id);
    identity[5u] = static_cast<std::uint32_t>(completed.matched_frame_id >> 32u);
    identity[6u] = static_cast<std::uint32_t>(completed.depth_input_region.analysis_generation);
    identity[7u] = static_cast<std::uint32_t>(completed.depth_input_region.analysis_generation >> 32u);
    identity[8u] = completed.depth_input_region.width();
    identity[9u] = completed.depth_input_region.height();
    identity[11u] = models::subtitle_ocr_source_crop_height(identity[8u], identity[9u]);
    identity[10u] = identity[9u] - identity[11u];
    identity[12u] = static_cast<std::uint32_t>(completed.field_width);
    identity[13u] = static_cast<std::uint32_t>(completed.field_height);
    identity[14u] = geometry.roi_top;
    identity[15u] = geometry.roi_bottom;
    identity[16u] = completed.field_content.left;
    identity[17u] = completed.field_content.top;
    identity[18u] = completed.field_content.right;
    identity[19u] = completed.field_content.bottom;
    identity[20u] = fixture.locator[adaptive + 4u];
    identity[21u] = fixture.locator[adaptive + 5u];
    identity[22u] = v2::subtitle_contour_recipe;
    identity[v2::subtitle_contour_history_snapshot_valid_word] = 1u;

    // The existing camera-authentication tests validate its seal separately; these tests join
    // contour evidence to the exact full-sized completed state without emulating a renderer.
    fixture.state.resize(v2::state_float_count, 0.0f);
    fixture.state[v2::frame_valid] = 1.0f;
    fixture.state[v2::renderer_authorization_bits] = std::bit_cast<float>(v2::contract_tag);
    fixture.state[v2::joint_plane_mode_bits] = std::bit_cast<float>(v2::adaptive_policy_id);
    fixture.state[v2::confirmed_cut_count] = std::bit_cast<float>(subtitle_scene_epoch);
    fixture.state[v2::gain_last_observation_low] = std::bit_cast<float>(identity[20u]);
    fixture.state[v2::gain_last_observation_high] = std::bit_cast<float>(identity[21u]);
    return fixture;
  }

  bool contour_identity_matches(const contour_dump_fixture &fixture) {
    return dump_detail::subtitle_contour_identity_matches_completion(
      word_bytes(fixture.identity), word_bytes(fixture.ocr), word_bytes(fixture.locator),
      fixture.state, fixture.completed);
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

  TEST(SbsDebugDumpGpuTraceTest, NativeValidatorAuthenticatesMatchedNormalPendingRoot) {
    const auto frame = gpu_trace_frame();
    const auto ring = canonical_gpu_trace_ring(frame);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, frame));
    const auto base = gpu_trace::record_base(0u);
    const auto flags = trace_word(
      ring, base + gpu_trace::word_index(gpu_trace::record_word_e::flags)
    );
    EXPECT_EQ(flags & gpu_trace::dump_forced, 0u)
      << "a dump may harvest a root that was already pending before the request";
  }

  TEST(SbsDebugDumpGpuTraceTest, MatchedPublicationClockMustEqualSourceTransaction) {
    auto frame = gpu_trace_frame();
    const auto ring = canonical_gpu_trace_ring(frame);
    frame.matched_observation_timestamp_us = 1'000'000u + frame.matched_frame_id;
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, frame));
    ++frame.matched_observation_timestamp_us;
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(ring, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, SingleHighFieldExtentOwnsTheHighDomainIdentity) {
    auto frame = gpu_trace_frame();
    frame.model_width = 1540;
    frame.model_height = 868;
    frame.raw_width = 1540;
    frame.raw_height = 868;
    frame.field_width = 1540;
    frame.field_height = 868;
    frame.field_content = {0u, 0u, 1540u, 868u};
    frame.depth_input_region.tensor_content = frame.field_content;
    frame.refined_live_geometry_active = true;
    const auto ring = canonical_gpu_trace_ring(frame);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, frame));

    const auto base = gpu_trace::record_base(0u);
    EXPECT_EQ(
      trace_word(ring, base + gpu_trace::word_index(
        gpu_trace::record_word_e::field_width)),
      1540u
    );
    EXPECT_EQ(
      trace_word(ring, base + gpu_trace::word_index(
        gpu_trace::record_word_e::field_height)),
      868u
    );
    const auto high_domain = models::near_identical_input_domain_tag(
      frame.depth_input_region,
      frame.color_space,
      static_cast<std::uint32_t>(frame.model_width),
      static_cast<std::uint32_t>(frame.model_height)
    );
    EXPECT_EQ(
      gpu_trace::join_u64(
        trace_word(ring, base + gpu_trace::word_index(
          gpu_trace::record_word_e::domain_tag_low)),
        trace_word(ring, base + gpu_trace::word_index(
          gpu_trace::record_word_e::domain_tag_high))
      ),
      high_domain
    );

    auto wrong_field_claim = frame;
    --wrong_field_claim.field_width;
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(ring, wrong_field_claim));
  }

  TEST(SbsDebugDumpTensorGridTest, AuthenticatesLegacyAndSingleHighButRejectsSplitGrid) {
    auto legacy = gpu_trace_frame();
    EXPECT_TRUE(dump_detail::capture_tensor_grid_is_authenticated(legacy));

    auto single_high = legacy;
    single_high.model_width = 1540;
    single_high.model_height = 868;
    single_high.raw_width = 1540;
    single_high.raw_height = 868;
    single_high.field_width = 1540;
    single_high.field_height = 868;
    single_high.field_content = {0u, 0u, 1540u, 868u};
    single_high.depth_input_region.tensor_content = single_high.field_content;
    single_high.refined_live_geometry_active = true;
    EXPECT_TRUE(dump_detail::capture_tensor_grid_is_authenticated(single_high));

    struct fused_shape_case_t {
      std::uint32_t source_width;
      std::uint32_t source_height;
      int coarse_width;
      int coarse_height;
    };
    constexpr std::array fused_shapes {
      fused_shape_case_t {1920u, 1080u, 770, 434},
      fused_shape_case_t {2560u, 1080u, 1022, 434},
      fused_shape_case_t {3440u, 1440u, 1036, 434},
      fused_shape_case_t {1080u, 1920u, 434, 770},
      fused_shape_case_t {1080u, 2560u, 434, 1022},
      fused_shape_case_t {1440u, 3440u, 434, 1036},
    };
    for (const auto &test_case : fused_shapes) {
      auto supported = legacy;
      supported.model_width = 2 * test_case.coarse_width;
      supported.model_height = 2 * test_case.coarse_height;
      supported.raw_width = supported.model_width;
      supported.raw_height = supported.model_height;
      supported.field_width = supported.model_width;
      supported.field_height = supported.model_height;
      supported.field_content = {
        0u,
        0u,
        static_cast<std::uint32_t>(supported.model_width),
        static_cast<std::uint32_t>(supported.model_height),
      };
      supported.depth_input_region.source_width = test_case.source_width;
      supported.depth_input_region.source_height = test_case.source_height;
      supported.depth_input_region.right = test_case.source_width;
      supported.depth_input_region.bottom = test_case.source_height;
      supported.depth_input_region.tensor_content = supported.field_content;
      supported.refined_live_geometry_active = true;
      SCOPED_TRACE(
        std::to_string(test_case.source_width) + "x" +
        std::to_string(test_case.source_height)
      );
      EXPECT_TRUE(dump_detail::capture_tensor_grid_is_authenticated(supported));
    }

    auto stale_split_grid = single_high;
    stale_split_grid.model_width = 770;
    stale_split_grid.model_height = 434;
    stale_split_grid.raw_width = 770;
    stale_split_grid.raw_height = 434;
    EXPECT_FALSE(dump_detail::capture_tensor_grid_is_authenticated(stale_split_grid));

    auto high_without_fused_authority = single_high;
    high_without_fused_authority.refined_live_geometry_active = false;
    EXPECT_FALSE(dump_detail::capture_tensor_grid_is_authenticated(
      high_without_fused_authority
    ));

    auto unsupported_high_claim = single_high;
    --unsupported_high_claim.model_width;
    unsupported_high_claim.raw_width = unsupported_high_claim.model_width;
    unsupported_high_claim.field_width = unsupported_high_claim.model_width;
    unsupported_high_claim.field_content.right =
      static_cast<std::uint32_t>(unsupported_high_claim.model_width);
    unsupported_high_claim.depth_input_region.tensor_content =
      unsupported_high_claim.field_content;
    EXPECT_FALSE(dump_detail::capture_tensor_grid_is_authenticated(
      unsupported_high_claim
    ));
  }

  TEST(SbsDebugDumpTensorGridTest, SingleHighRequiresExactCompositeRuntimeProvenance) {
    auto legacy = gpu_trace_frame();
    const auto &calibration = v2::model_calibrations.front();
    legacy.raw_model_provenance = std::make_shared<models::raw_model_provenance_t>(
      models::raw_model_provenance_t {
        .depth_model = std::string {calibration.depth_model},
        .depth_model_url = std::string {calibration.depth_model_url},
        .onnx_sha256 = std::string {calibration.onnx_sha256},
        .preprocess_profile = std::string {calibration.preprocess.profile},
        .preprocess_source_closure_sha256 =
          std::string {calibration.preprocess.source_closure_sha256},
      }
    );
    EXPECT_TRUE(dump_detail::composite_runtime_provenance_is_authenticated(legacy));

    auto high = legacy;
    high.refined_live_geometry_active = true;
    EXPECT_FALSE(dump_detail::composite_runtime_provenance_is_authenticated(high));
    const std::string engine_artifact =
      std::string {models::prod_zipdepth_convex2x::logical_model} + "." +
      std::string {models::prod_zipdepth_convex2x::engine_recipe} +
      ".cache-" + std::string(64u, 'a') + ".engine";
    high.composite_depth_runtime_provenance =
      std::make_shared<models::composite_depth_runtime_provenance_t>(
        models::composite_depth_runtime_provenance_t {
          .model = std::string {models::prod_zipdepth_convex2x::logical_model},
          .onnx_sha256 =
            std::string {models::prod_zipdepth_convex2x::fused_onnx_sha256},
          .embedded_dav2_onnx_sha256 =
            std::string {models::prod_zipdepth_convex2x::dav2_onnx_sha256},
          .zipdepth_checkpoint_sha256 =
            std::string {models::prod_zipdepth_convex2x::zipdepth_checkpoint_sha256},
          .guidance_preprocess_source_closure_sha256 =
            std::string {calibration.preprocess.source_closure_sha256},
          .engine_recipe =
            std::string {models::prod_zipdepth_convex2x::engine_recipe},
          .engine_artifact = engine_artifact,
          .active_engine_manifest =
            std::string {models::prod_zipdepth_convex2x::logical_model} +
            ".active-engine.json",
        }
      );
    EXPECT_TRUE(dump_detail::composite_runtime_provenance_is_authenticated(high));

    auto stale = high;
    auto stale_runtime =
      std::make_shared<models::composite_depth_runtime_provenance_t>(
        *high.composite_depth_runtime_provenance
      );
    stale_runtime->engine_recipe = "stale";
    stale.composite_depth_runtime_provenance = std::move(stale_runtime);
    EXPECT_FALSE(dump_detail::composite_runtime_provenance_is_authenticated(stale));

    legacy.composite_depth_runtime_provenance =
      high.composite_depth_runtime_provenance;
    EXPECT_FALSE(dump_detail::composite_runtime_provenance_is_authenticated(legacy));
  }

  TEST(SbsDebugDumpTensorGridTest, PortraitSingleHighPassesSubtitleCaptureGate) {
    int resource_sentinel = 0;
    auto *const resource =
      reinterpret_cast<ID3D11ShaderResourceView *>(&resource_sentinel);
    struct portrait_case_t {
      std::uint32_t source_width;
      std::uint32_t source_height;
      int field_width;
      int field_height;
    };
    constexpr std::array portrait_shapes {
      portrait_case_t {1080u, 1920u, 868, 1540},
      portrait_case_t {1080u, 2560u, 868, 2044},
      portrait_case_t {1440u, 3440u, 868, 2072},
    };
    for (const auto &test_case : portrait_shapes) {
      auto completed = gpu_trace_frame();
      completed.model_width = test_case.field_width;
      completed.model_height = test_case.field_height;
      completed.raw_width = test_case.field_width;
      completed.raw_height = test_case.field_height;
      completed.field_width = test_case.field_width;
      completed.field_height = test_case.field_height;
      completed.field_content = {
        0u,
        0u,
        static_cast<std::uint32_t>(test_case.field_width),
        static_cast<std::uint32_t>(test_case.field_height),
      };
      completed.depth_input_region.source_width = test_case.source_width;
      completed.depth_input_region.source_height = test_case.source_height;
      completed.depth_input_region.right = test_case.source_width;
      completed.depth_input_region.bottom = test_case.source_height;
      completed.depth_input_region.tensor_content = completed.field_content;
      completed.refined_live_geometry_active = true;
      completed.ocr_box_record = resource;
      completed.subtitle_locator_state = resource;
      completed.shadow_base_final_parallax = resource;
      completed.subtitle_contour_support = resource;
      completed.subtitle_contour_field_pin = resource;
      completed.subtitle_contour_distance = resource;
      completed.subtitle_contour_identity = resource;
      completed.subtitle_contour_history = resource;
      completed.subtitle_contour_source = resource;
      completed.subtitle_ocr_probability = resource;
      completed.subtitle_gaussian_weight = resource;
      completed.subtitle_gaussian_blend = resource;
      completed.subtitle_gaussian_vertical_conditioned = resource;
      completed.subtitle_gaussian_control = resource;
      SCOPED_TRACE(
        std::to_string(test_case.field_width) + "x" +
        std::to_string(test_case.field_height)
      );
      ASSERT_TRUE(dump_detail::capture_tensor_grid_is_authenticated(completed));
      EXPECT_TRUE(dump_detail::subtitle_capture_resources_are_authenticated(completed));
      for (auto member : {
             &platf::sbs_debug::frame::subtitle_contour_support,
             &platf::sbs_debug::frame::subtitle_contour_field_pin,
             &platf::sbs_debug::frame::subtitle_contour_distance,
             &platf::sbs_debug::frame::subtitle_contour_identity,
             &platf::sbs_debug::frame::subtitle_contour_history,
             &platf::sbs_debug::frame::subtitle_contour_source,
             &platf::sbs_debug::frame::subtitle_ocr_probability,
             &platf::sbs_debug::frame::subtitle_gaussian_weight,
             &platf::sbs_debug::frame::subtitle_gaussian_blend,
             &platf::sbs_debug::frame::subtitle_gaussian_vertical_conditioned,
             &platf::sbs_debug::frame::subtitle_gaussian_control,
           }) {
        auto partial_contour = completed;
        partial_contour.*member = nullptr;
        EXPECT_FALSE(dump_detail::subtitle_capture_resources_are_authenticated(partial_contour));
      }
      completed.subtitle_work_suppressed = true;
      completed.subtitle_contour_source = nullptr;
      EXPECT_TRUE(dump_detail::subtitle_capture_resources_are_authenticated(completed))
        << "suppressed subtitle work keeps the existing none descriptor";
    }

    auto partial = gpu_trace_frame();
    partial.ocr_box_record = resource;
    EXPECT_FALSE(dump_detail::subtitle_capture_resources_are_authenticated(partial));

    auto missing_base = gpu_trace_frame();
    missing_base.ocr_box_record = resource;
    missing_base.subtitle_locator_state = resource;
    EXPECT_FALSE(dump_detail::subtitle_capture_resources_are_authenticated(missing_base));
  }

  TEST(SbsDebugDumpContourTest, JoinsCurrentSct5ToCropDomainCutAndSourceClock) {
    const auto too_tall = current_contour_dump_fixture(180u, 200u);
    ASSERT_TRUE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(too_tall.ocr), too_tall.completed));
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(too_tall.ocr), word_bytes(too_tall.locator), too_tall.completed))
      << "all producer-qualified current OCR pairs are retained, including tall cores";
    const auto fixture = current_contour_dump_fixture();
    ASSERT_TRUE(subtitle_records_match_frame(
      word_bytes(fixture.ocr), word_bytes(fixture.locator), fixture.completed));
    ASSERT_TRUE(contour_identity_matches(fixture));
    for (const std::size_t index : {
           0u, 1u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u,
           14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 24u, 25u, 26u, 31u,
         }) {
      auto forged = fixture;
      forged.identity[index] ^= 1u;
      EXPECT_FALSE(contour_identity_matches(forged)) << "SCT5 word " << index;
    }
    auto forged = fixture;
    forged.identity[2u] = 2u;
    EXPECT_FALSE(contour_identity_matches(forged));
    forged = fixture;
    forged.state[v2::gain_last_observation_low] = std::bit_cast<float>(fixture.identity[20u] + 1u);
    EXPECT_FALSE(contour_identity_matches(forged));
    forged = fixture;
    forged.state[v2::confirmed_cut_count] = std::bit_cast<float>(subtitle_scene_epoch + 1u);
    EXPECT_FALSE(contour_identity_matches(forged));
    forged = fixture;
    forged.state[v2::frame_valid] = 0.0f;
    EXPECT_FALSE(contour_identity_matches(forged));
    forged.identity[2u] = forged.identity[3u] = 0u;
    forged.identity[v2::subtitle_contour_history_snapshot_valid_word] = 0u;
    forged.state[v2::renderer_authorization_bits] = 0.0f;
    forged.locator[v2::subtitle_plane_flags_word] = 0u;
    forged.locator[v2::subtitle_plane_current_slot_mask_word] = 0u;
    EXPECT_TRUE(contour_identity_matches(forged)) << "invalid geometry can publish canonical empty contour evidence";
  }

  TEST(SbsDebugDumpContourTest, CurrentCoversCannotForgeInactivePixelAuthority) {
    auto fixture = current_contour_dump_fixture();
    ASSERT_TRUE(contour_identity_matches(fixture));
    fixture.identity[2u] = 0u;
    EXPECT_FALSE(contour_identity_matches(fixture));
  }

  TEST(SbsDebugDumpContourTest, ReadyActualOcrCannotForgeCurrentPlaneAbstention) {
    auto fixture = current_contour_dump_fixture();
    ASSERT_TRUE(contour_identity_matches(fixture));
    fixture.locator[v2::subtitle_plane_flags_word] = 0u;
    fixture.locator[v2::subtitle_plane_current_slot_mask_word] = 0u;
    fixture.identity[2u] = 0u;
    EXPECT_FALSE(contour_identity_matches(fixture))
      << "a ready actual nonempty OCR record grants all current slots even if support is cleared";

    auto empty = current_contour_dump_fixture();
    empty.ocr[3u] = empty.ocr[4u] = 0u;
    std::fill(empty.ocr.begin() + v2::subtitle_ocr_raw_box_offset, empty.ocr.end(), 0u);
    empty.locator[v2::subtitle_plane_current_slot_mask_word] = 0u;
    empty.identity[2u] = 0u;
    ASSERT_TRUE(contour_identity_matches(empty))
      << "ready authoritative empty OCR retains a valid current publication without pins";
    empty.locator[v2::subtitle_plane_flags_word] = 0u;
    EXPECT_FALSE(contour_identity_matches(empty));

    auto invalid_probe = current_contour_dump_fixture();
    invalid_probe.locator[v2::subtitle_plane_adaptive_offset + 15u] = 1u;
    invalid_probe.identity[2u] = 0u;
    EXPECT_TRUE(contour_identity_matches(invalid_probe))
      << "invalid Base probe suppresses pins while retaining exact current OCR slots";
  }

  TEST(SbsDebugDumpContourTest, ControllerClockIsIndependentButInvalidProbeAbstains) {
    auto fixture = current_contour_dump_fixture();
    const auto a = v2::subtitle_plane_adaptive_offset;
    fixture.locator[a + 4u] = 1234u;
    fixture.locator[a + 5u] = 0u;
    ASSERT_TRUE(contour_identity_matches(fixture));
    fixture.locator[a + 1u] = 0u;
    fixture.locator[a + 4u] = 0u;
    ASSERT_TRUE(contour_identity_matches(fixture));
    fixture.locator[a + 15u] = 1u;
    EXPECT_FALSE(contour_identity_matches(fixture));
    fixture.identity[2u] = 0u;
    EXPECT_TRUE(contour_identity_matches(fixture));
    EXPECT_EQ(fixture.identity[v2::subtitle_contour_history_snapshot_valid_word], 1u);
  }

  TEST(SbsDebugDumpContourTest, HeldTupleUsesRetainedSourceClockAndCannotFreshStampIdentity) {
    auto held = current_contour_dump_fixture();
    ++held.completed.matched_frame_id;
    held.completed.matched_observation_timestamp_us += 33333u;
    held.completed.gpu_undecided_completion = true;
    // subtitle_records_match_completion additionally proves the older OCR8/UIP1 tuple from the
    // committed completion ring. SCT5 must independently join that same older frame and clock.
    ASSERT_TRUE(contour_identity_matches(held));
    auto forged = held;
    forged.identity[4u] = static_cast<std::uint32_t>(held.completed.matched_frame_id);
    forged.identity[5u] = static_cast<std::uint32_t>(held.completed.matched_frame_id >> 32u);
    EXPECT_FALSE(contour_identity_matches(forged));
    forged = held;
    forged.identity[20u] = static_cast<std::uint32_t>(held.completed.matched_observation_timestamp_us);
    forged.identity[21u] = static_cast<std::uint32_t>(held.completed.matched_observation_timestamp_us >> 32u);
    EXPECT_FALSE(contour_identity_matches(forged));
    held.completed.gpu_undecided_completion = false;
    EXPECT_FALSE(contour_identity_matches(held));
  }

  TEST(SbsDebugDumpContourTest, HeldOlderFrameMayRetainTheCurrentSourceTimestamp) {
    auto held = current_contour_dump_fixture();
    ++held.completed.matched_frame_id;
    held.completed.gpu_undecided_completion = true;
    // The source timeline is nondecreasing. Equal PTS belongs to the frozen older tuple;
    // it does not create another RGB confirmation or spend adaptive controller time.
    ASSERT_TRUE(contour_identity_matches(held));
    auto future = held;
    --future.completed.matched_observation_timestamp_us;
    EXPECT_FALSE(contour_identity_matches(future));
    held.identity[4u] = static_cast<std::uint32_t>(held.completed.matched_frame_id);
    held.identity[5u] = static_cast<std::uint32_t>(held.completed.matched_frame_id >> 32u);
    EXPECT_FALSE(contour_identity_matches(held));
  }

  TEST(SbsDebugDumpContourTest, TextureAbiRequiresExactTypedBytesAndUnsupportedPlaceholder) {
    using role_e = dump_detail::subtitle_contour_texture_e;
    auto completed = current_contour_dump_fixture().completed;
    const auto verify = [&](role_e role, DXGI_FORMAT format, std::uint32_t width,
                            std::uint32_t height, std::size_t pixel_bytes) {
      const auto row_bytes = static_cast<std::size_t>(width) * pixel_bytes;
      const auto byte_count = row_bytes * height;
      EXPECT_TRUE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
        role, format, width, height, row_bytes, byte_count, completed));
      EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
        role, format == DXGI_FORMAT_R32_FLOAT ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R32_FLOAT,
        width, height, row_bytes, byte_count, completed));
      EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
        role, format, width, height, row_bytes + 4u, byte_count, completed));
      EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
        role, format, width, height, row_bytes, byte_count - 1u, completed));
      EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
        role, format, width + 1u, height, row_bytes, byte_count, completed));
    };
    const auto crop_height = models::subtitle_ocr_source_crop_height(1280u, 720u);
    verify(role_e::support, DXGI_FORMAT_R32_UINT, 1280u, crop_height, 4u);
    verify(role_e::history, DXGI_FORMAT_R32G32B32A32_FLOAT, 1280u, crop_height, 16u);
    verify(role_e::source, DXGI_FORMAT_R16G16B16A16_FLOAT, 1280u, crop_height, 8u);
    verify(role_e::field_pin, DXGI_FORMAT_R32_UINT, 770u, 434u, 4u);
    verify(role_e::distance, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_weight, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_blend, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_vertical_conditioned, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    completed.depth_input_region.source_width = completed.depth_input_region.right =
      v2::subtitle_contour_source_max_extent + 1u;
    verify(role_e::support, DXGI_FORMAT_R32_UINT, 1u, 1u, 4u);
    verify(role_e::history, DXGI_FORMAT_R32G32B32A32_FLOAT, 1u, 1u, 16u);
    verify(role_e::source, DXGI_FORMAT_R16G16B16A16_FLOAT, 1u, 1u, 8u);
    verify(role_e::field_pin, DXGI_FORMAT_R32_UINT, 770u, 434u, 4u);
    verify(role_e::distance, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_weight, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_blend, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    verify(role_e::gaussian_vertical_conditioned, DXGI_FORMAT_R32_FLOAT, 770u, 434u, 4u);
    EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
      role_e::source, DXGI_FORMAT_R16G16B16A16_FLOAT, 1u, 2u, 8u, 16u, completed));
    EXPECT_FALSE(dump_detail::subtitle_contour_texture_layout_is_authenticated(
      role_e::support, DXGI_FORMAT_R32_UINT, 1280u, crop_height, 5120u,
      5120u * crop_height, completed));
  }

  TEST(SbsDebugDumpContourTest, EmptyContourGatePreservesIndependentLegalEmptyDetectorSnapshot) {
    auto fixture = current_contour_dump_fixture();
    fixture.identity[2u] = 0u;
    fixture.ocr[3u] = fixture.ocr[4u] = 0u;
    std::fill(fixture.ocr.begin() + v2::subtitle_ocr_raw_box_offset, fixture.ocr.end(), 0u);
    const auto native_pixels = static_cast<std::size_t>(fixture.identity[8u]) * fixture.identity[11u];
    const auto field_pixels = static_cast<std::size_t>(fixture.identity[12u]) * fixture.identity[13u];
    std::vector<std::uint8_t> support(native_pixels * 4u), pin(field_pixels * 4u);
    std::vector<std::uint8_t> history(native_pixels * 16u), source(native_pixels * 8u);
    std::vector<std::uint8_t> probability(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u);
    auto distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, field_pixels);
    source[1u] = 0x38u;  // RGBA16F red=0.5, alpha=1, matching the finite display-crop ABI.
    source[7u] = 0x3cu;
    history[3u] = 0x3fu;  // RGB red=0.5 from the same source pixel.
    history[14u] = 0x00u; history[15u] = 0x40u;  // Valid RGB, empty UI mask: code2.
    probability[3u] = 0x3fu;  // Opaque float32 detector probability=0.5.
    const auto valid = [&]() {
      return dump_detail::subtitle_contour_clear_policy_is_authenticated(
        word_bytes(fixture.identity), word_bytes(fixture.ocr), fixture.state,
        support, pin, distance, history, source, probability);
    };
    ASSERT_TRUE(valid()) << "legal empty OCR is valid evidence before any current cover is admitted";
    for (auto *payload : {&support, &pin, &history}) {
      (*payload)[0u] = 1u;
      EXPECT_FALSE(valid());
      (*payload)[0u] = 0u;
    }
    distance[0u] = 1u;
    EXPECT_FALSE(valid());
    distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, field_pixels);
    fixture.ocr[2u] = 0u;
    EXPECT_FALSE(valid()) << "globally invalid detector output cannot authorize raw snapshots";
    std::fill(source.begin(), source.end(), 0u);
    std::fill(probability.begin(), probability.end(), 0u);
    std::fill(history.begin(), history.end(), 0u);
    fixture.identity[v2::subtitle_contour_history_snapshot_valid_word] = 0u;
    fixture.identity[3u] = 0u;
    EXPECT_TRUE(valid());
    fixture.ocr[2u] = 1u;
    fixture.state[v2::gain_last_observation_low] = std::bit_cast<float>(fixture.identity[20u] + 1u);
    probability[0u] = 1u;
    EXPECT_FALSE(valid()) << "a stale geometry clock cannot authorize the next Pmap";
    probability[0u] = 0u;
    EXPECT_TRUE(valid());
  }

  TEST(SbsDebugDumpContourTest, UnsupportedSourceAcceptsOnlyCanonicalEmptySeedPayloads) {
    auto fixture = current_contour_dump_fixture();
    fixture.identity[2u] = fixture.identity[3u] = 0u;
    fixture.identity[v2::subtitle_contour_history_snapshot_valid_word] = 0u;
    fixture.identity[8u] = v2::subtitle_contour_source_max_extent + 1u;
    const auto field_pixels = static_cast<std::size_t>(fixture.identity[12u]) * fixture.identity[13u];
    std::vector<std::uint8_t> support(4u), pin(field_pixels * 4u), history(16u), source(8u);
    std::vector<std::uint8_t> probability(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u);
    const auto distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, field_pixels);
    const auto valid = [&]() {
      return dump_detail::subtitle_contour_clear_policy_is_authenticated(
        word_bytes(fixture.identity), word_bytes(fixture.ocr), fixture.state,
        support, pin, distance, history, source, probability);
    };
    ASSERT_TRUE(valid());
    for (auto *payload : {&support, &pin, &history, &source, &probability}) {
      (*payload)[0u] = 1u;
      EXPECT_FALSE(valid());
      (*payload)[0u] = 0u;
    }
    source.push_back(0u);
    EXPECT_FALSE(valid()) << "even cleared unsupported snapshots require the exact placeholder byte count";
    source.pop_back();
    fixture.identity[2u] = 1u;
    EXPECT_FALSE(valid());
  }

  TEST(SbsDebugDumpContourTest, GaussianFieldsRequireJoinedControlEndpointsAndExactEmptyBase) {
    // This isolates saved-field admission; real q/plane production is covered by native shaders.
    auto fixture = current_contour_dump_fixture();
    fixture.identity[12u] = 3u; fixture.identity[13u] = 2u;
    fixture.identity[16u] = 1u; fixture.identity[17u] = 0u;
    fixture.identity[18u] = 2u; fixture.identity[19u] = 2u;
    auto locator = fixture.locator;
    locator[v2::subtitle_plane_current_slot_mask_word] = 1u;
    locator[v2::subtitle_plane_field_extent_offset] = 3u;
    locator[v2::subtitle_plane_field_extent_offset + 1u] = 2u;
    locator[v2::subtitle_plane_content_offset] = 1u;
    locator[v2::subtitle_plane_content_offset + 1u] = 0u;
    locator[v2::subtitle_plane_content_offset + 2u] = 2u;
    locator[v2::subtitle_plane_content_offset + 3u] = 2u;
    locator[v2::subtitle_plane_adaptive_offset + 13u] = 2u;
    constexpr auto target_bits = 0x80000000u;
    locator[v2::subtitle_plane_adaptive_offset + 3u] = target_bits;
    auto distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, 6u);
    auto weight = repeated_word_bytes(0u, 6u);
    auto base = repeated_word_bytes(std::bit_cast<std::uint32_t>(0.01234567f), 6u);
    const auto store_word = [](auto &bytes, std::size_t index, std::uint32_t value) {
      std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
    };
    // Real content has signed zero in row0 and a non-Q30 constant in row1. Padding is poison.
    store_word(base, 1u, target_bits);
    auto expected_base = base;
    for (std::size_t row = 0u; row < 2u; ++row) {
      const auto i = row * 3u + 1u;
      std::uint32_t value;
      std::memcpy(&value, base.data() + i * 4u, 4u);
      store_word(expected_base, row * 3u, value);
      store_word(expected_base, row * 3u + 2u, value);
    }
    auto blend = expected_base, vertical = expected_base, final = expected_base;
    auto control = repeated_word_bytes(0u, 1u);
    const auto valid = [&]() {
      return dump_detail::subtitle_gaussian_fields_are_authenticated(
        word_bytes(fixture.identity), word_bytes(locator), word_bytes(fixture.ocr), distance, weight, blend,
        vertical, control, base, final);
    };
    ASSERT_TRUE(valid());
    for (auto *payload : {&blend, &vertical, &final}) {
      (*payload)[0u] ^= 1u;
      EXPECT_FALSE(valid());
      (*payload)[0u] ^= 1u;
    }
    weight[0u] = 1u;
    EXPECT_FALSE(valid());
    weight[0u] = 0u;
    control[0u] = 1u;
    EXPECT_FALSE(valid()) << "empty q cannot run shared conditioning";
    control[0u] = 0u;
    control.push_back(0u);
    EXPECT_FALSE(valid()) << "control is exactly one uint32";
    control.pop_back();

    for (std::size_t i : {1u, 4u}) store_word(distance, i, 0u);
    weight = repeated_word_bytes(std::bit_cast<std::uint32_t>(1.0f), 6u);
    blend = repeated_word_bytes(target_bits, 6u);
    vertical = blend; final = blend; control[0u] = 1u;
    ASSERT_TRUE(valid()) << "q0/M1 retains signed target, including nearest-content padding";
    for (auto *payload : {&weight, &blend}) {
      (*payload)[0u] ^= 1u;
      EXPECT_FALSE(valid());
      (*payload)[0u] ^= 1u;
    }
    store_word(distance, 1u, target_bits);
    EXPECT_FALSE(valid()) << "q zero is positive, not signed depth zero";
    store_word(distance, 1u, 0u);
    locator[v2::subtitle_plane_flags_word] = 0u;
    EXPECT_FALSE(valid()) << "current full-strength plane authority is required";
  }

  TEST(SbsDebugDumpContourTest, WarmupGaussianRequiresExactCurrentCoversAndExpiresWithEmptyNativeQ) {
    auto fixture = current_contour_dump_fixture();
    auto locator = fixture.locator;
    locator[v2::subtitle_plane_warmup_word] = v2::subtitle_plane_warmup_observations;
    locator[v2::subtitle_plane_adaptive_offset + 3u] = std::bit_cast<std::uint32_t>(0.0f);
    const auto width = fixture.identity[12u], height = fixture.identity[13u];
    const auto left = fixture.identity[16u], top = fixture.identity[17u];
    const auto right = fixture.identity[18u], bottom = fixture.identity[19u];
    const auto pixels = static_cast<std::size_t>(width) * height;
    const auto distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, pixels);
    auto weight = repeated_word_bytes(0u, pixels);
    const auto zero = repeated_word_bytes(0u, pixels);
    auto control = repeated_word_bytes(1u, 1u);
    const double sigma = std::min(fixture.identity[8u], fixture.identity[9u]) / 60.0;
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const double fx = std::clamp(x, left, right - 1u) + .5;
        const double fy = std::clamp(y, top, bottom - 1u) + .5;
        double q = std::numeric_limits<double>::infinity();
        for (std::uint32_t slot = 0u; slot < fixture.ocr[4u]; ++slot) {
          const auto offset = v2::subtitle_ocr_final_box_offset + slot * v2::subtitle_ocr_box_word_count;
          const double dx = std::max({fixture.ocr[offset] - fx, 0.0, fx - fixture.ocr[offset + 2u]}) *
            fixture.identity[8u] / (right - left);
          const double dy = std::max({fixture.ocr[offset + 1u] - fy, 0.0, fy - fixture.ocr[offset + 3u]}) *
            fixture.identity[9u] / (bottom - top);
          q = std::min(q, dx * dx + dy * dy);
        }
        const float m = static_cast<float>(std::exp(-q / (2.0 * sigma * sigma)));
        std::memcpy(weight.data() + (static_cast<std::size_t>(y) * width + x) * 4u, &m, 4u);
      }
    }
    const auto valid = [&]() {
      return dump_detail::subtitle_gaussian_fields_are_authenticated(
        word_bytes(fixture.identity), word_bytes(locator), word_bytes(fixture.ocr),
        distance, weight, zero, zero, control, zero, zero);
    };
    ASSERT_TRUE(valid()) << "presentation warmup can be finite while raw pixel q remains empty";
    for (const auto packed : {1u, 2u, 3u, 4u, 9u, 10u, 11u, 12u}) {
      locator[v2::subtitle_plane_warmup_word] = packed;
      EXPECT_TRUE(valid());
    }
    for (const auto packed : {0u, 8u, 5u, 6u, 7u, 13u, 14u, 15u, 16u, 0xffffffffu}) {
      locator[v2::subtitle_plane_warmup_word] = packed;
      EXPECT_FALSE(valid()) << "expired or malformed countdown cannot admit rectangle weight";
    }
    locator[v2::subtitle_plane_warmup_word] = 4u;
    fixture.ocr[5u] ^= 1u;
    EXPECT_FALSE(valid()) << "a stale current OCR frame cannot authorize warmup";
    fixture.ocr[5u] ^= 1u;
    fixture.ocr[v2::subtitle_ocr_final_box_offset + 4u] ^= 1u;
    EXPECT_FALSE(valid()) << "raw/final cover payload correspondence remains authenticated";
  }

  TEST(SbsDebugDumpContourTest, ConfirmationCodesBindBinarySupportAndBoundedCurrentAnchors) {
    auto fixture = current_contour_dump_fixture();
    const auto native_pixels = static_cast<std::size_t>(fixture.identity[8u]) * fixture.identity[11u];
    const auto field_pixels = static_cast<std::size_t>(fixture.identity[12u]) * fixture.identity[13u];
    const auto pixel = 64u * static_cast<std::size_t>(fixture.identity[8u]) + 200u;
    std::vector<std::uint8_t> support(native_pixels * 4u), pin(field_pixels * 4u);
    std::vector<std::uint8_t> history(native_pixels * 16u), source(native_pixels * 8u);
    std::vector<std::uint8_t> probability(v2::subtitle_ocr_output_width * v2::subtitle_ocr_output_height * 4u);
    const auto distance = repeated_word_bytes(v2::subtitle_contour_distance_empty_bits, field_pixels);
    const std::array<std::uint16_t, 4u> color {0x3800u, 0x3800u, 0x3800u, 0x3c00u};
    std::memcpy(source.data() + pixel * sizeof(color), color.data(), sizeof(color));
    const auto set_history = [&](const std::array<float, 4u> &feature) {
      std::memcpy(history.data() + pixel * sizeof(feature), feature.data(), sizeof(feature));
    };
    const auto set_mask = [&](const std::uint32_t value) {
      std::memcpy(support.data() + pixel * sizeof(value), &value, sizeof(value));
    };
    const auto valid = [&]() {
      return dump_detail::subtitle_contour_clear_policy_is_authenticated(
        word_bytes(fixture.identity), word_bytes(fixture.ocr), fixture.state,
        support, pin, distance, history, source, probability);
    };
    // This checks typed saved-state admission, not a fabricated four-observation chain.
    for (std::uint32_t count = 0u; count <= v2::subtitle_contour_history_confirmation_observations; ++count) {
      const auto mask = static_cast<std::uint32_t>(count == v2::subtitle_contour_history_confirmation_observations);
      const auto code = v2::subtitle_contour_history_feature_valid_bit |
        (count << v2::subtitle_contour_history_feature_count_shift) |
        (mask ? v2::subtitle_contour_history_feature_mask_bit : 0u);
      set_mask(mask);
      set_history({0.5f, 0.5f, 0.5f, static_cast<float>(code)});
      ASSERT_TRUE(valid()) << "legal count " << count;
      set_mask(1u - mask);
      EXPECT_FALSE(valid()) << "support disagrees with count " << count;
      set_mask(mask);
      set_history({0.5625f, 0.5f, 0.5f, static_cast<float>(code)});
      EXPECT_EQ(valid(), count >= 2u) << "fixed anchor is permitted only at count2+";
    }
    set_mask(0u);
    for (const auto code : {0.0f, 1.0f, 3.0f, 7.0f, 11.0f, 15.0f, 18.0f, 19.0f, 22.0f,
                           2.5f, -0.0f, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN()}) {
      set_history({0.5f, 0.5f, 0.5f, code});
      EXPECT_FALSE(valid()) << "unknown/mismatched code " << code;
    }
    for (const auto anchor : {0.625f, -0.001f, 1.001f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
      set_history({anchor, 0.5f, 0.5f, 10.0f});
      EXPECT_FALSE(valid()) << "invalid/distant count2 anchor " << anchor;
    }
    set_history({0.5625f, 0.5f, 0.5f, 10.0f});
    ASSERT_TRUE(valid());
    fixture.identity[2u] = 0u;
    EXPECT_FALSE(valid()) << "no current authority may carry a confirmation streak";
    set_history({0.5f, 0.5f, 0.5f, 2.0f});
    EXPECT_TRUE(valid()) << "empty current covers still retains current RGB at count0";
    source[pixel * sizeof(color) + 7u] = 0u;
    EXPECT_FALSE(valid()) << "alpha-zero source with nonzero RGB is not canonical";
    std::fill(source.begin() + pixel * sizeof(color), source.begin() + (pixel + 1u) * sizeof(color), 0u);
    set_history({0.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_TRUE(valid());
    set_history({0.0f, 0.0f, 0.0f, -0.0f});
    EXPECT_FALSE(valid()) << "invalid source must have bitwise zero history";
  }

  TEST(SbsDebugDumpTensorGridTest, SingleHighRoiRetainsCoarsePlannerAndDoublesContent) {
    constexpr models::depth_source_rect_t semantic_rect {100u, 80u, 1700u, 980u};
    const auto coarse_plan = models::plan_host_sbs_v2_video_region(
      semantic_rect,
      1920u,
      1080u,
      {770, 434}
    );
    ASSERT_TRUE(coarse_plan);
    const auto high_content = models::host_sbs_convex2x_content_rect(
      coarse_plan->tensor_content,
      {770, 434}
    );
    ASSERT_TRUE(high_content);

    auto frame = gpu_trace_frame();
    frame.model_width = 1540;
    frame.model_height = 868;
    frame.raw_width = 1540;
    frame.raw_height = 868;
    frame.field_width = 1540;
    frame.field_height = 868;
    frame.field_content = *high_content;
    frame.refined_live_geometry_active = true;
    frame.depth_input_region.left = coarse_plan->source_rect.left;
    frame.depth_input_region.top = coarse_plan->source_rect.top;
    frame.depth_input_region.right = coarse_plan->source_rect.right;
    frame.depth_input_region.bottom = coarse_plan->source_rect.bottom;
    frame.depth_input_region.tensor_content = *high_content;
    frame.depth_input_region.analysis_generation = 7u;
    frame.depth_input_region.authority =
      models::depth_analysis_authority_e::foreground_client;
    frame.depth_video_plan = *coarse_plan;
    frame.window_region = platf::sbs_debug::window_region_snapshot {
      .authority_kind =
        platf::sbs_debug::window_region_authority_kind_e::foreground_client,
      .matched_frame_id = frame.matched_frame_id,
      .source_width = 1920u,
      .source_height = 1080u,
      .left = static_cast<std::int32_t>(semantic_rect.left),
      .top = static_cast<std::int32_t>(semantic_rect.top),
      .right = static_cast<std::int32_t>(semantic_rect.right),
      .bottom = static_cast<std::int32_t>(semantic_rect.bottom),
      .hwnd = 1u,
      .process_id = 2u,
      .generation = 3u,
      .latest_observation_age_ms_at_capture = 4u,
      .maximum_observation_age_ms = 100u,
      .geometry_continuity_ms_at_capture = 10u,
      .source_content_age_ms_at_capture = 5u,
    };
    frame.window_region_observer_status = "ok";
    frame.window_region_mapping_status = "ok";
    EXPECT_TRUE(dump_detail::capture_tensor_grid_is_authenticated(frame));

    auto stale_coarse_content = frame;
    stale_coarse_content.depth_input_region.tensor_content =
      coarse_plan->tensor_content;
    stale_coarse_content.field_content = coarse_plan->tensor_content;
    EXPECT_FALSE(dump_detail::capture_tensor_grid_is_authenticated(
      stale_coarse_content
    ));

    auto high_grid_planner = frame;
    high_grid_planner.depth_video_plan = models::plan_host_sbs_v2_video_region(
      semantic_rect,
      1920u,
      1080u,
      {1540, 868}
    );
    EXPECT_FALSE(dump_detail::capture_tensor_grid_is_authenticated(high_grid_planner));
  }

  TEST(SbsDebugDumpGpuTraceTest, HeldOrdinaryReuseAuthenticatesTheOlderPublishedSubtitleTuple) {
    auto completed = gpu_trace_frame();
    completed.gpu_undecided_completion = true;
    auto evidence = completed;
    --evidence.matched_frame_id;
    --evidence.matched_observation_timestamp_us;
    const auto geometry = models::fit_subtitle_analysis_geometry(
      evidence.depth_input_region.width(),
      evidence.depth_input_region.height(),
      {evidence.model_width, evidence.model_height},
      evidence.depth_input_region.tensor_content
    );
    ASSERT_TRUE(geometry.valid());
    auto ocr = empty_ocr(evidence, geometry);
    const subtitle_rect_t first {120u, 350u, 300u, 400u};
    const subtitle_rect_t second {320u, 350u, 650u, 400u};
    store_ocr_pair(ocr, 0u, first, first);
    store_ocr_pair(ocr, 1u, second, second);
    ocr[2u] = 1u;
    ocr[3u] = 2u;
    ocr[4u] = 2u;
    const auto locator = two_line_subtitle_state(evidence, first, second);

    auto ring = canonical_gpu_trace_ring(
      completed,
      gpu_trace::submission_class_e::gpu_undecided,
      cuda_conditional_graph::branch_e::reuse
    );
    auto falsely_current_ocr = empty_ocr(completed, geometry);
    store_ocr_pair(falsely_current_ocr, 0u, first, first);
    store_ocr_pair(falsely_current_ocr, 1u, second, second);
    falsely_current_ocr[2u] = 1u;
    falsely_current_ocr[3u] = 2u;
    falsely_current_ocr[4u] = 2u;
    const auto falsely_current_locator = two_line_subtitle_state(
      completed, first, second
    );
    EXPECT_FALSE(dump_detail::subtitle_records_match_completion(
      word_bytes(falsely_current_ocr), word_bytes(falsely_current_locator), ring,
      completed, subtitle_scene_epoch
    )) << "an authenticated ordinary reuse may not fresh-stamp its frozen subtitle tuple";

    const auto base = gpu_trace::record_base(0u);
    const auto locator_base = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    for (std::size_t index = 0u; index < locator.size(); ++index) {
      set_trace_word(ring, locator_base + index, locator[index]);
    }
    const auto condition_base = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );
    set_trace_word(ring, condition_base + 0u, v2::subtitle_condition_param_schema);
    set_trace_word(ring, condition_base + 1u, v2::subtitle_condition_param_tag);
    set_trace_word(ring, condition_base + 2u, 2u);
    set_trace_word(ring, condition_base + 3u, 0u);
    set_trace_word(ring, condition_base + 4u, 2u);
    set_trace_word(ring, condition_base + 5u, locator[v2::subtitle_plane_adaptive_offset + 3u]);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, completed));
    EXPECT_TRUE(dump_detail::subtitle_records_match_completion(
      word_bytes(ocr), word_bytes(locator), ring, completed, subtitle_scene_epoch
    ));

    set_trace_word(
      ring,
      locator_base + v2::subtitle_plane_current_slot_mask_word,
      locator[v2::subtitle_plane_current_slot_mask_word] + 1u
    );
    EXPECT_FALSE(dump_detail::subtitle_records_match_completion(
      word_bytes(ocr), word_bytes(locator), ring, completed, subtitle_scene_epoch
    ));
  }

  TEST(SbsDebugDumpGpuTraceTest, NativeValidatorRejectsTornAuthDerivedAndDomainDrift) {
    const auto frame = gpu_trace_frame();
    const auto canonical = canonical_gpu_trace_ring(frame);
    const auto base = gpu_trace::record_base(0u);
    const auto transaction = base + gpu_trace::word_index(
      gpu_trace::record_word_e::transaction_begin
    );
    const auto locator = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    const auto condition = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );
    const auto reject = [&](const std::size_t index, const std::uint32_t value) {
      auto changed = canonical;
      set_trace_word(changed, index, value);
      EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(changed, frame));
    };
    reject(gpu_trace::word_index(gpu_trace::header_word_e::tag), 0u);
    reject(base + gpu_trace::word_index(gpu_trace::record_word_e::commit_tag), 0u);
    reject(base + gpu_trace::word_index(gpu_trace::record_word_e::sequence_low), 9u);
    reject(transaction + 1u, trace_word(canonical, transaction + 1u) ^ 1u);
    reject(
      base + gpu_trace::word_index(gpu_trace::record_word_e::depth_disposition),
      static_cast<std::uint32_t>(gpu_trace::depth_disposition_e::reuse)
    );
    reject(
      base + gpu_trace::word_index(gpu_trace::record_word_e::domain_tag_low),
      trace_word(
        canonical,
        base + gpu_trace::word_index(gpu_trace::record_word_e::domain_tag_low)
      ) ^ 1u
    );
    reject(condition + 5u, trace_word(canonical, condition + 5u) ^ 1u);
    reject(locator + 2u, trace_word(canonical, locator + 2u) | 0x10u);
    reject(locator + v2::subtitle_plane_current_slot_mask_word, 1u << v2::subtitle_plane_cover_capacity);
  }

  TEST(SbsDebugDumpGpuTraceTest, AbstentionCannotClaimTheAuthorityOfAnExecutedEmptyOcrChild) {
    const auto frame = gpu_trace_frame();
    const auto abstention = canonical_gpu_trace_ring(
      frame, gpu_trace::submission_class_e::force_infer,
      cuda_conditional_graph::branch_e::infer, true, false);
    const auto base = gpu_trace::record_base(0u);
    const auto plane = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin);
    const auto condition = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_condition_begin);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(abstention, frame));
    EXPECT_EQ(trace_word(abstention, plane + v2::subtitle_plane_flags_word), 0u);
    EXPECT_EQ(trace_word(abstention, plane + v2::subtitle_plane_current_slot_mask_word), 0u);

    auto forged_empty_authority = abstention;
    set_trace_word(forged_empty_authority, plane + v2::subtitle_plane_flags_word, v2::subtitle_plane_current_valid_flag);
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(forged_empty_authority, frame))
      << "no optional OCR child cannot claim even valid-empty current OCR authority";
    auto forged_active = canonical_gpu_trace_ring(frame);
    std::memcpy(forged_active.data() + base * sizeof(std::uint32_t),
                abstention.data() + base * sizeof(std::uint32_t),
                gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin) * sizeof(std::uint32_t));
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(forged_active, frame));

    auto valid_empty = canonical_gpu_trace_ring(frame);
    set_trace_word(valid_empty, plane + v2::subtitle_plane_current_slot_mask_word, 0u);
    for (std::size_t index = 2u; index < v2::subtitle_condition_param_word_count; ++index)
      set_trace_word(valid_empty, condition + index, 0u);
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(valid_empty, frame))
      << "an executed OCR child may publish a valid current empty record";
  }

  TEST(SbsDebugDumpGpuTraceTest, HeldOlderTupleAllowsEqualTimestampWithoutItsOriginInTheRing) {
    auto completed = gpu_trace_frame();
    completed.gpu_undecided_completion = true;
    auto evidence = completed;
    --evidence.matched_frame_id;
    const auto geometry = models::fit_subtitle_analysis_geometry(
      evidence.depth_input_region.width(), evidence.depth_input_region.height(),
      {evidence.model_width, evidence.model_height}, evidence.depth_input_region.tensor_content);
    ASSERT_TRUE(geometry.valid());
    auto ocr = empty_ocr(evidence, geometry);
    const subtitle_rect_t cover {120u, 350u, 650u, 400u};
    store_ocr_pair(ocr, 0u, cover, cover);
    ocr[2u] = 1u;
    ocr[3u] = ocr[4u] = 1u;
    const auto plane = current_subtitle_plane(evidence, 1u, true, 0.01f);
    auto ring = canonical_gpu_trace_ring(
      completed, gpu_trace::submission_class_e::gpu_undecided,
      cuda_conditional_graph::branch_e::reuse);
    const auto base = gpu_trace::record_base(0u);
    const auto plane_base = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin);
    for (std::size_t index = 0u; index < plane.size(); ++index)
      set_trace_word(ring, plane_base + index, plane[index]);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, completed));
    EXPECT_TRUE(dump_detail::subtitle_records_match_completion(
      word_bytes(ocr), word_bytes(plane), ring, completed, subtitle_scene_epoch));
    auto future = ring;
    set_trace_word(future, plane_base + v2::subtitle_plane_observation_timestamp_offset,
                   static_cast<std::uint32_t>(completed.matched_observation_timestamp_us + 1u));
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(future, completed));
  }

  TEST(SbsDebugDumpGpuTraceTest, ForceReuseIsInvalidButSubtitleFallbackRemainsExact) {
    const auto frame = gpu_trace_frame();
    const auto invalid = canonical_gpu_trace_ring(
      frame,
      gpu_trace::submission_class_e::force_infer,
      cuda_conditional_graph::branch_e::reuse,
      true
    );
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(invalid, frame));
    const auto base = gpu_trace::record_base(0u);
    EXPECT_EQ(
      trace_word(invalid, base + gpu_trace::word_index(
        gpu_trace::record_word_e::depth_disposition)),
      static_cast<std::uint32_t>(gpu_trace::depth_disposition_e::invalid)
    );
    EXPECT_EQ(
      trace_word(invalid, base + gpu_trace::word_index(
        gpu_trace::record_word_e::subtitle_disposition)),
      static_cast<std::uint32_t>(gpu_trace::subtitle_disposition_e::abstention)
    );
    const auto mislabeled = canonical_gpu_trace_ring(
      frame,
      gpu_trace::submission_class_e::force_infer,
      cuda_conditional_graph::branch_e::reuse,
      false
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(mislabeled, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, AuthenticatedOrdinaryReuseHoldsCoherentSubtitleTuple) {
    const auto frame = gpu_trace_frame();
    auto reused = canonical_gpu_trace_ring(
      frame,
      gpu_trace::submission_class_e::gpu_undecided,
      cuda_conditional_graph::branch_e::reuse
    );
    const auto base = gpu_trace::record_base(0u);
    const auto flags = base + gpu_trace::word_index(gpu_trace::record_word_e::flags);
    const auto disposition =
      base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_disposition);
    EXPECT_EQ(trace_word(reused, disposition), static_cast<std::uint32_t>(
      gpu_trace::subtitle_disposition_e::held_with_depth
    ));
    EXPECT_EQ(
      trace_word(reused, flags) &
        (gpu_trace::ocr_record_submitted | gpu_trace::condition_executed |
         gpu_trace::subtitle_branch_gated),
      gpu_trace::subtitle_branch_gated
    );
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(reused, frame));

    const auto locator =
      base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin);
    EXPECT_EQ(trace_word(reused, locator + v2::subtitle_plane_frame_offset), frame.matched_frame_id - 1u);
    auto falsely_current_locator = reused;
    set_trace_word(
      falsely_current_locator,
      locator + v2::subtitle_plane_frame_offset,
      static_cast<std::uint32_t>(frame.matched_frame_id)
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(
      falsely_current_locator, frame
    ));

    for (const auto forbidden_host_claim : {
           gpu_trace::ocr_record_submitted, gpu_trace::condition_executed,
         }) {
      auto false_execution_proof = reused;
      set_trace_word(
        false_execution_proof, flags,
        trace_word(reused, flags) | forbidden_host_claim
      );
      EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(
        false_execution_proof, frame
      ));
    }

    auto relabeled = reused;
    set_trace_word(
      relabeled, disposition,
      static_cast<std::uint32_t>(gpu_trace::subtitle_disposition_e::invalid)
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(relabeled, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, HeldSubtitleTupleEqualsImmediatelyPriorRecord) {
    const auto frame = gpu_trace_frame();
    auto held = canonical_gpu_trace_ring(
      frame,
      gpu_trace::submission_class_e::gpu_undecided,
      cuda_conditional_graph::branch_e::reuse
    );
    auto prior_frame = frame;
    --prior_frame.matched_frame_id;
    --prior_frame.matched_observation_timestamp_us;
    const auto prior = canonical_gpu_trace_ring(prior_frame);
    const auto current_base = gpu_trace::record_base(0u);
    const auto oldest_base = gpu_trace::record_base(gpu_trace::capacity - 1u);
    std::memcpy(
      held.data() + oldest_base * sizeof(std::uint32_t),
      prior.data() + current_base * sizeof(std::uint32_t),
      gpu_trace::record_word_count * sizeof(std::uint32_t)
    );
    set_trace_word(
      held,
      oldest_base + gpu_trace::word_index(gpu_trace::record_word_e::sequence_low),
      1u
    );
    set_trace_word(
      held,
      current_base + gpu_trace::word_index(gpu_trace::record_word_e::sequence_low),
      2u
    );
    set_trace_word(
      held, gpu_trace::word_index(gpu_trace::header_word_e::next_sequence_low), 3u
    );
    set_trace_word(
      held, gpu_trace::word_index(gpu_trace::header_word_e::next_slot), 1u
    );
    set_trace_word(
      held, gpu_trace::word_index(gpu_trace::header_word_e::committed_count), 2u
    );
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(held, frame));

    auto changed = held;
    const auto current_locator = current_base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    const auto current_condition = current_base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );
    const auto changed_target = trace_word(changed, current_locator + v2::subtitle_plane_adaptive_offset + 3u) ^ 1u;
    set_trace_word(changed, current_locator + v2::subtitle_plane_adaptive_offset + 3u, changed_target);
    set_trace_word(changed, current_condition + 5u, changed_target);
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(changed, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, JsonSerializerPublishesBranchGatedHoldSemantics) {
    const models::host_sbs_gpu_trace_provenance_t provenance {
      .source_closure_schema = models::host_sbs_shader_cache::source_closure_schema,
      .source_compile_flags = models::host_sbs_shader_cache::shader_compile_flags,
      .source_macro_count = 0u,
      .source_closure_sha256 = std::string {
        models::host_sbs_shader_cache::gpu_trace_source_closure_sha256
      },
    };
    const auto contract = nlohmann::json::parse(
      dump_detail::gpu_trace_contract_json(provenance)
    );
    EXPECT_EQ(
      contract.at("enums").at("subtitle_disposition").at("held_with_depth"),
      5u
    );
    EXPECT_EQ(
      contract.at("enums").at("subtitle_disposition").at("invalid"),
      6u
    );
    EXPECT_EQ(
      contract.at("enums").at("flags").at("subtitle_branch_gated"),
      gpu_trace::subtitle_branch_gated
    );
    EXPECT_NE(
      contract.at("record_sections").at("subtitle_locator").at("validity")
        .get<std::string>().find("held_with_depth"),
      std::string::npos
    );

    const auto frame = gpu_trace_frame();
    const auto reused = nlohmann::json::parse(dump_detail::gpu_trace_decoded_json(
      canonical_gpu_trace_ring(
        frame,
        gpu_trace::submission_class_e::gpu_undecided,
        cuda_conditional_graph::branch_e::reuse
      ),
      frame
    ));
    const auto &held_record = reused.at("records").at(0u);
    EXPECT_EQ(held_record.at("subtitle_disposition").at("name"), "held-with-depth");
    EXPECT_TRUE(held_record.at("flags").at("subtitle_branch_gated"));
    EXPECT_FALSE(held_record.at("flags").at("condition_executed"));
    EXPECT_FALSE(held_record.at("flags").at("condition_executed_host_proven"));
    EXPECT_TRUE(held_record.at("subtitle_condition").at("held_with_depth"));
    EXPECT_TRUE(held_record.at("subtitle_condition").at("active"));

    const auto inferred = nlohmann::json::parse(dump_detail::gpu_trace_decoded_json(
      canonical_gpu_trace_ring(
        frame,
        gpu_trace::submission_class_e::gpu_undecided,
        cuda_conditional_graph::branch_e::infer
      ),
      frame
    ));
    const auto &infer_record = inferred.at("records").at(0u);
    EXPECT_TRUE(infer_record.at("flags").at("subtitle_branch_gated"));
    EXPECT_TRUE(infer_record.at("flags").at("condition_executed"));
    EXPECT_FALSE(infer_record.at("flags").at("condition_executed_host_proven"));
    EXPECT_FALSE(infer_record.at("subtitle_condition").at("held_with_depth"));
    EXPECT_TRUE(infer_record.at("subtitle_condition").at("executed"));
    EXPECT_TRUE(infer_record.at("subtitle_condition").at("active"));
  }

  TEST(SbsDebugDumpGpuTraceTest, EmptyAndInvalidProbeRequireFreshInactiveScp4) {
    const auto frame = gpu_trace_frame();
    const auto canonical = canonical_gpu_trace_ring(frame);
    const auto base = gpu_trace::record_base(0u);
    const auto plane = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_locator_begin);
    const auto condition = base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_condition_begin);
    auto inactive = canonical;
    set_trace_word(inactive, plane + v2::subtitle_plane_flags_word, 0u);
    set_trace_word(inactive, plane + v2::subtitle_plane_current_slot_mask_word, 0u);
    for (std::size_t index = 2u; index < v2::subtitle_condition_param_word_count; ++index)
      set_trace_word(inactive, condition + index, 0u);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(inactive, frame));
    auto missing_schema = inactive;
    set_trace_word(missing_schema, condition, 0u);
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(missing_schema, frame));
    auto stale_payload = inactive;
    set_trace_word(stale_payload, condition + 2u, 1u);
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(stale_payload, frame));
    auto invalid_probe = canonical;
    set_trace_word(invalid_probe, plane + v2::subtitle_plane_adaptive_offset + 13u, 100u);
    set_trace_word(invalid_probe, plane + v2::subtitle_plane_adaptive_offset + 15u, 1u);
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(invalid_probe, frame))
      << "canonical invalid Base sample cannot retain active SCP4";
    for (std::size_t index = 2u; index < v2::subtitle_condition_param_word_count; ++index)
      set_trace_word(invalid_probe, condition + index, 0u);
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(invalid_probe, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, WrappedChronologyAndSuppressedFrozenSectionsValidate) {
    const auto frame = gpu_trace_frame();
    auto wrapped = canonical_gpu_trace_ring(frame);
    const auto template_base = gpu_trace::record_base(0u);
    const auto oldest_base = gpu_trace::record_base(gpu_trace::capacity - 1u);
    std::vector<std::uint8_t> record(
      gpu_trace::record_word_count * sizeof(std::uint32_t)
    );
    std::memcpy(
      record.data(),
      wrapped.data() + template_base * sizeof(std::uint32_t),
      record.size()
    );
    std::memcpy(
      wrapped.data() + oldest_base * sizeof(std::uint32_t),
      record.data(),
      record.size()
    );
    set_trace_word(
      wrapped,
      oldest_base + gpu_trace::word_index(gpu_trace::record_word_e::sequence_low),
      500u
    );
    set_trace_word(
      wrapped,
      template_base + gpu_trace::word_index(gpu_trace::record_word_e::sequence_low),
      501u
    );
    set_trace_word(
      wrapped, gpu_trace::word_index(gpu_trace::header_word_e::next_sequence_low), 502u
    );
    set_trace_word(
      wrapped, gpu_trace::word_index(gpu_trace::header_word_e::next_slot), 1u
    );
    set_trace_word(
      wrapped, gpu_trace::word_index(gpu_trace::header_word_e::committed_count), 2u
    );
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(wrapped, frame));

    auto reset_frame = frame;
    reset_frame.input_domain_reset = true;
    auto suppressed = canonical_gpu_trace_ring(
      reset_frame,
      gpu_trace::submission_class_e::gpu_undecided,
      cuda_conditional_graph::branch_e::reuse
    );
    const auto base = gpu_trace::record_base(0u);
    constexpr std::uint64_t token = 0x1020304050607080ull;
    const auto request = cuda_conditional_graph::make_request(
      token, cuda_conditional_graph::work_flag_e::none
    );
    const auto receipt = cuda_conditional_graph::resolve_proposal(
      cuda_conditional_graph::make_proposal(
        cuda_conditional_graph::branch_e::reuse, token
      ),
      request,
      false
    );
    const auto transaction = base + gpu_trace::word_index(
      gpu_trace::record_word_e::transaction_begin
    );
    std::memcpy(
      suppressed.data() + transaction * sizeof(std::uint32_t),
      &receipt,
      sizeof(receipt)
    );
    std::memcpy(
      suppressed.data() + (transaction + 8u) * sizeof(std::uint32_t),
      &request,
      sizeof(request)
    );
    set_trace_word(
      suppressed,
      base + gpu_trace::word_index(gpu_trace::record_word_e::expected_work),
      0u
    );
    set_trace_word(
      suppressed,
      base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_disposition),
      static_cast<std::uint32_t>(gpu_trace::subtitle_disposition_e::suppressed)
    );
    set_trace_word(
      suppressed,
      base + gpu_trace::word_index(gpu_trace::record_word_e::flags),
      gpu_trace::input_domain_reset | gpu_trace::subtitle_suppressed
    );
    set_trace_word(
      suppressed,
      base + gpu_trace::word_index(gpu_trace::record_word_e::host_subtitle_outcome),
      static_cast<std::uint32_t>(gpu_trace::host_subtitle_outcome_e::suppressed)
    );
    const auto locator = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    const auto condition = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );
    set_trace_word(suppressed, locator, 0xffffffffu);
    set_trace_word(suppressed, locator + 2u, 0xffffffffu);
    set_trace_word(suppressed, condition, 0xffffffffu);
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(suppressed, reset_frame))
      << "explicit suppression remains distinct even when depth authentically reuses";
  }

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

  TEST(SbsDebugDumpAsyncTest, DiagnosticsDisabledAllowsManualDumpWithoutFilePolling) {
    scoped_dump_trigger_environment environment;
    ASSERT_TRUE(environment.ready());
    config::sunshine.diagnostics_enabled = false;

    platf::sbs_debug::dumper dumper;
    auto button = std::make_shared<std::atomic<bool>>(false);
    dumper.set_button_request(button);

    // The fixture creates a file trigger, but diagnostics-off must leave it dormant.
    EXPECT_FALSE(dumper.needs_conversion_poll());
    EXPECT_FALSE(dumper.snapshot_requested());

    ASSERT_TRUE(stream::sbs_debug_dump_request_allowed(video::SBS_AI, (bool) button));
    button->store(true, std::memory_order_release);
    EXPECT_TRUE(dumper.needs_conversion_poll());
    EXPECT_TRUE(dumper.snapshot_requested());

    dumper.cancel_pending_request();
    EXPECT_FALSE(button->load());
    EXPECT_FALSE(dumper.needs_conversion_poll());
    EXPECT_FALSE(dumper.snapshot_requested());
  }

  TEST(SbsDebugDumpAsyncTest, Schema49PackagesCompleteContourTupleAndNoScalarPreviews) {
    std::ifstream stream(
      std::filesystem::path(SUNSHINE_SOURCE_DIR) /
        "src/platform/windows/sbs_debug_dump.cpp",
      std::ios::binary
    );
    ASSERT_TRUE(stream);
    const std::string source {
      std::istreambuf_iterator<char> {stream},
      std::istreambuf_iterator<char> {},
    };
    EXPECT_NE(
      source.find("artifacts[\"shadow_base_final_parallax.f32\"]"),
      std::string::npos
    );
    EXPECT_NE(
      source.find("artifacts[\"shadow_coordinate.f32\"]"),
      std::string::npos
    );
    EXPECT_NE(
      source.find("dimensions[\"shadow_coordinate\"]"),
      std::string::npos
    );
    for (const std::string_view retired : {
           "warp_depth.f32",
           "shadow_base_final_parallax_shape.json",
           "shadow_base_final_parallax.png",
           "shadow_base_final_parallax_heat.png",
         }) {
      EXPECT_EQ(source.find(retired), std::string::npos) << retired;
    }
    EXPECT_NE(source.find("{\"schema\", 49}"), std::string::npos);
    EXPECT_NE(source.find("{\"mode\", \"subtitle-contours-v1\"}"), std::string::npos);
    EXPECT_NE(source.find("nlohmann::json::parse(subtitle_contour_policy_json)"), std::string::npos);
    for (const std::string_view artifact : {
           "contour_support.u32", "contour_field_pin.u32", "contour_distance.f32",
           "contour_identity.u32", "contour_history.rgba32f", "contour_source.rgba16f",
           "ocr_probability.f32", "gaussian_weight.f32", "gaussian_blend.f32",
           "gaussian_vertical_conditioned.f32", "gaussian_control.u32",
         }) {
      EXPECT_NE(source.find(artifact), std::string::npos) << artifact;
    }
    const auto terminal_query = source.find("ctx->End(pending->completion.Get())");
    ASSERT_NE(terminal_query, std::string::npos);
    for (const std::string_view resource : {
           "subtitle_contour_support", "subtitle_contour_field_pin", "subtitle_contour_distance",
           "subtitle_contour_identity", "subtitle_contour_history", "subtitle_contour_source",
           "subtitle_ocr_probability", "subtitle_gaussian_weight", "subtitle_gaussian_blend",
           "subtitle_gaussian_vertical_conditioned", "subtitle_gaussian_control",
         }) {
      const auto staging = source.find("device, ctx, completed." + std::string(resource));
      EXPECT_NE(staging, std::string::npos) << resource;
      EXPECT_LT(staging, terminal_query) << resource;
      EXPECT_NE(source.find("job.completed." + std::string(resource) + " = nullptr;"), std::string::npos)
        << resource;
    }
    EXPECT_NE(source.find("{\"matched_observation_timestamp_us\", completed.matched_observation_timestamp_us}"), std::string::npos);
    EXPECT_NE(
      source.find("{\"warp_input_artifact\", \"shadow_final_parallax.f32\"}"),
      std::string::npos
    );
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

  TEST(SbsDebugDumpAsyncTest, DirectCurrentPublicationRetainsEveryPairedOcrSlot) {
    const auto frame = subtitle_frame({0u, 0u, 770u, 434u}, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.field_width, frame.field_height}, frame.field_content);
    ASSERT_TRUE(geometry.valid());
    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u; ocr[3u] = ocr[4u] = v2::subtitle_plane_cover_capacity;
    auto plane = current_subtitle_plane(frame, v2::subtitle_plane_cover_capacity);
    for (std::uint32_t slot = 0u; slot < v2::subtitle_plane_cover_capacity; ++slot) {
      const subtitle_rect_t core {4u + slot * 80u, geometry.roi_top + 2u,
        14u + slot * 80u, geometry.roi_top + 22u};
      const subtitle_rect_t cover {core[0u] - 2u, core[1u] - 1u, core[2u] + 2u, core[3u] + 1u};
      store_ocr_pair(ocr, slot, core, cover);
    }
    ASSERT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(plane), frame));
    // Current producer order may change; no winner geometry is duplicated in UIP1.
    for (const auto offset : {v2::subtitle_ocr_raw_box_offset, v2::subtitle_ocr_final_box_offset}) {
      for (std::size_t word = 0u; word < v2::subtitle_ocr_box_word_count; ++word)
        std::swap(ocr[offset + word], ocr[offset + 7u * v2::subtitle_ocr_box_word_count + word]);
    }
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(plane), frame));
    auto missing_slot = plane;
    missing_slot[v2::subtitle_plane_current_slot_mask_word] = 127u;
    EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(missing_slot), frame));
    auto mismatched_pair = ocr;
    mismatched_pair[v2::subtitle_ocr_final_box_offset + 4u] ^= 1u;
    EXPECT_FALSE(subtitle_records_match_frame(word_bytes(mismatched_pair), word_bytes(plane), frame));
    for (const auto index : {v2::subtitle_plane_frame_offset, v2::subtitle_plane_analysis_generation_offset,
        v2::subtitle_plane_observation_timestamp_offset, v2::subtitle_plane_source_extent_offset,
        v2::subtitle_plane_field_extent_offset, v2::subtitle_plane_roi_offset,
        v2::subtitle_plane_content_offset, v2::subtitle_plane_cut_epoch_word, v2::subtitle_plane_reserved_offset}) {
      auto corrupt = plane; corrupt[index] ^= 1u;
      EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(corrupt), frame)) << index;
    }
    // Legal empty has exact current identity; invalid OCR can publish only no-current authority.
    ocr = empty_ocr(frame, geometry); ocr[2u] = 1u;
    plane = current_subtitle_plane(frame, 0u);
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(plane), frame));
    ocr[2u] = 0u;
    EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(plane), frame));
    plane[v2::subtitle_plane_flags_word] = 0u;
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(plane), frame));
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


  TEST(SbsDebugDumpAsyncTest, NativeAdaptiveUiAuthenticatesHostCapAndRejectsOtherModes) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    auto frame = subtitle_frame(content, 3840u, 2160u);
    frame.parallax_v2_joint_plane_mode = 3u;
    frame.parallax_v2_requested_gain = 0.0065625f;
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.field_width, frame.field_height}, content);
    ASSERT_TRUE(geometry.valid());
    const auto ocr = empty_ocr(frame, geometry);
    auto locator = current_subtitle_plane(frame, 0u, false);
    constexpr auto a = v2::subtitle_plane_adaptive_offset;
    locator[a] = 1u;
    locator[a + 1u] = 8u;
    locator[a + 4u] = 1234u;
    const auto cap = std::bit_cast<std::uint32_t>(v2::direct_container_limit);
    locator[a + 12u] = cap;
    ASSERT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    for (const auto wrong_cap : {cap - 1u, cap + 1u, 0x7fc00000u, 0x7f800000u}) {
      locator[a + 12u] = wrong_cap;
      EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    }
    locator[a + 12u] = cap;
    const auto maximum_level = (frame.depth_input_region.width() + 99u) / 100u;
    locator[a + 2u] = maximum_level;
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    ++locator[a + 2u];
    EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    locator[a + 2u] = 0u;
    for (const auto mode : {0u, 1u, 2u, 4u}) {
      auto wrong_mode = frame;
      wrong_mode.parallax_v2_joint_plane_mode = mode;
      EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), wrong_mode));
    }
    // Depth strength is independent of the full Host UI representation guard.
    frame.parallax_v2_requested_gain = 0.03f;
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    for (const auto gain : {0.0f, -0.001f, std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
      auto wrong_gain = frame;
      wrong_gain.parallax_v2_requested_gain = gain;
      EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), wrong_gain));
    }
    // The capped-conflict bit must agree with the observed top-candidate counters.
    locator[a + 13u] = 100u;
    locator[a + 14u] = 21u;
    EXPECT_FALSE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
    locator[a + 1u] |= 4u;
    EXPECT_TRUE(subtitle_records_match_frame(word_bytes(ocr), word_bytes(locator), frame));
  }

  TEST(SbsDebugDumpAsyncTest, NativeAdaptiveStateSerializationAdmitsStartupAndDisarmedFlatTuples) {
    auto frame = gpu_trace_frame();
    frame.parallax_v2_joint_plane_mode = 3u;
    frame.parallax_v2_raw_coordinate_scale = 2.25f;
    frame.parallax_v2_requested_pop_strength = 1.75f;
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.75f);
    std::vector<float> stats {2.0f, 0.5f, 1.0f, 4.0f, 16.0f, 16.0f, 1.0f, 0.0f};
    stats.resize(v2::frame_stats_float_count, 0.0f);
    stats[v2::frame_stat_percentile_low] = 1.0f;
    stats[v2::frame_stat_percentile_high] = 4.0f;
    stats[v2::frame_stat_percentile_valid] = 1.0f;
    stats[v2::frame_stat_percentile_bin_width] = 3.0f / 256.0f;
    v2::state_words_t words {};
    words[v2::container_scale] = std::bit_cast<std::uint32_t>(1.0f);
    words[v2::contract_tag_bits] = v2::contract_tag;
    words[v2::joint_plane_mode_bits] = 3u;
    const auto matches = [&] {
      words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
      std::vector<float> state(v2::state_float_count);
      std::memcpy(state.data(), words.data(), sizeof(words));
      return dump_detail::parallax_v2_state_matches_frame(state, stats, frame);
    };
    // An empty controller is an authenticated flat diagnostic before its first valid clock.
    ASSERT_TRUE(matches());
    words[v2::gain_last_observation_low] = 1000001u;
    words[v2::gain_clock_armed] = 1u;
    words[v2::gain_seed_count] = 1u;
    words[v2::gain_target_zero] = std::bit_cast<std::uint32_t>(2.0f);
    words[v2::gain_target_inverse_scale] = std::bit_cast<std::uint32_t>(0.25f);
    words[v2::gain_target_nearest] = std::bit_cast<std::uint32_t>(4.0f);
    words[v2::gain_display_limit] = std::bit_cast<std::uint32_t>(v2::direct_container_limit);
    words[v2::gain_seed_first_low] = words[v2::gain_seed_last_low] = 1000001u;
    words[v2::gain_seed_mean_nearest] = std::bit_cast<std::uint32_t>(4.0f);
    words[v2::gain_seed_mean_zero] = std::bit_cast<std::uint32_t>(2.0f);
    words[v2::center] = std::bit_cast<std::uint32_t>(2.0f);
    words[v2::inverse_scale] = std::bit_cast<std::uint32_t>(0.25f);
    words[v2::calibration_revision] = 1u;
    words[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
    words[v2::renderer_authorization_bits] = v2::contract_tag;
    ASSERT_TRUE(matches());
    words[v2::frame_valid] = 0u;
    words[v2::renderer_authorization_bits] = 0u;
    words[v2::gain_clock_armed] = 0u;
    words[v2::gain_target_zero] = words[v2::gain_target_inverse_scale] = words[v2::gain_target_nearest] = 0u;
    ASSERT_TRUE(matches());
    for (const auto mode : {0u, 1u, 2u}) {
      frame.parallax_v2_joint_plane_mode = mode;
      EXPECT_FALSE(matches());
    }
    frame.parallax_v2_joint_plane_mode = 3u;
    frame.parallax_v2_requested_pop_strength = 1.0f;
    // A different valid requested gain remains independent of the retained scale and guard.
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.0f);
    EXPECT_TRUE(matches());
  }

  TEST(SbsDebugDumpAsyncTest, NativeAdaptiveSerializationRequiresQuantilesForValidOutput) {
    auto frame = gpu_trace_frame();
    frame.parallax_v2_joint_plane_mode = 3u;
    frame.parallax_v2_raw_coordinate_scale = 2.25f;
    frame.parallax_v2_requested_pop_strength = 1.0f;
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.0f);
    v2::state_words_t words {};
    words[v2::center] = std::bit_cast<std::uint32_t>(-2.0f);
    words[v2::inverse_scale] = std::bit_cast<std::uint32_t>(0.25f);
    words[v2::container_scale] = std::bit_cast<std::uint32_t>(1.0f);
    words[v2::calibration_revision] = 1u;
    words[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
    words[v2::contract_tag_bits] = words[v2::renderer_authorization_bits] = v2::contract_tag;
    words[v2::joint_plane_mode_bits] = 3u;
    words[v2::gain_last_observation_low] = words[v2::gain_seed_first_low] = words[v2::gain_seed_last_low] = 1000001u;
    words[v2::gain_clock_armed] = words[v2::gain_seed_count] = 1u;
    words[v2::gain_target_zero] = words[v2::gain_seed_mean_zero] = words[v2::center];
    words[v2::gain_target_inverse_scale] = words[v2::inverse_scale];
    words[v2::gain_target_nearest] = words[v2::gain_seed_mean_nearest] = std::bit_cast<std::uint32_t>(4.0f);
    words[v2::gain_display_limit] = std::bit_cast<std::uint32_t>(v2::direct_container_limit);
    words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
    std::vector<float> state(v2::state_float_count);
    std::memcpy(state.data(), words.data(), sizeof(words));
    // Signed raw depth is valid Host evidence; no positive Game3D nearest-depth assumption.
    std::vector<float> stats {-2.0f, 0.5f, -5.0f, 0.0f, 16.0f, 16.0f, 1.0f, 0.0f};
    stats.resize(v2::frame_stats_float_count, 0.0f);
    stats[v2::frame_stat_percentile_low] = -4.0f;
    stats[v2::frame_stat_percentile_high] = 0.0f;
    stats[v2::frame_stat_percentile_valid] = 1.0f;
    stats[v2::frame_stat_percentile_bin_width] = 5.0f / 256.0f;
    ASSERT_TRUE(dump_detail::parallax_v2_state_matches_frame(state, stats, frame));
    stats[v2::frame_stat_percentile_valid] = 0.0f;
    EXPECT_FALSE(dump_detail::parallax_v2_state_matches_frame(state, stats, frame));
    stats[v2::frame_stat_percentile_valid] = 1.0f;
    stats[v2::frame_stat_percentile_high] = 1.0f;
    EXPECT_FALSE(dump_detail::parallax_v2_state_matches_frame(state, stats, frame));
    stats[v2::frame_stat_percentile_high] = 0.0f;
    stats[v2::frame_stat_population_std] = 0.0f;
    EXPECT_FALSE(dump_detail::parallax_v2_state_matches_frame(state, stats, frame));
  }

  TEST(SbsDebugDumpAsyncTest, NativeFlatSerializationRejectsMalformedQuantileTails) {
    auto frame = gpu_trace_frame();
    frame.parallax_v2_raw_coordinate_scale = 2.25f;
    frame.parallax_v2_requested_pop_strength = 1.0f;
    frame.parallax_v2_requested_gain = v2::requested_gain_for_config(1.0f);
    for (const auto mode : {v2::adaptive_policy_id}) {
      SCOPED_TRACE(mode);
      frame.parallax_v2_joint_plane_mode = mode;
      v2::state_words_t words {};
      words[v2::container_scale] = std::bit_cast<std::uint32_t>(1.0f);
      words[v2::contract_tag_bits] = v2::contract_tag;
      words[v2::joint_plane_mode_bits] = mode;
      words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
      std::vector<float> state(v2::state_float_count);
      std::memcpy(state.data(), words.data(), sizeof(words));
      std::vector<float> unavailable(v2::frame_stats_float_count, 0.0f);
      unavailable[v2::frame_stat_valid_count] = unavailable[v2::frame_stat_texel_count] = 16.0f;
      const auto matches = [&](const std::vector<float> &stats) {
        return dump_detail::parallax_v2_state_matches_frame(state, stats, frame);
      };
      ASSERT_TRUE(matches(unavailable));
      for (const auto invalid_flag : {-1.0f, 0.5f, 2.0f}) {
        auto corrupt = unavailable;
        corrupt[v2::frame_stat_percentile_valid] = invalid_flag;
        EXPECT_FALSE(matches(corrupt));
      }
      for (const auto field : {v2::frame_stat_percentile_low, v2::frame_stat_percentile_high,
             v2::frame_stat_percentile_bin_width}) {
        auto corrupt = unavailable;
        corrupt[field] = 0.001f;
        EXPECT_FALSE(matches(corrupt));
      }
      auto declared = unavailable;
      declared[v2::frame_stat_percentile_valid] = 1.0f;
      EXPECT_FALSE(matches(declared)); // Quantiles cannot authorize invalid moments.

      // Collapsed current statistics keep output flat, but their independent tail must still
      // obey the same flag, range and width contract as the Python diagnostic reader.
      declared[v2::frame_stat_valid] = 1.0f;
      declared[v2::frame_stat_minimum] = -1.0f;
      declared[v2::frame_stat_maximum] = 1.0f;
      declared[v2::frame_stat_percentile_low] = -0.5f;
      declared[v2::frame_stat_percentile_high] = 0.5f;
      declared[v2::frame_stat_percentile_bin_width] = 2.0f / 256.0f;
      ASSERT_TRUE(matches(declared));
      for (const auto field : {v2::frame_stat_percentile_low, v2::frame_stat_percentile_high,
             v2::frame_stat_percentile_bin_width}) {
        auto corrupt = declared;
        corrupt[field] = field == v2::frame_stat_percentile_high ? 2.0f : -2.0f;
        EXPECT_FALSE(matches(corrupt));
      }
      auto reversed = declared;
      reversed[v2::frame_stat_percentile_low] = 0.75f;
      EXPECT_FALSE(matches(reversed));
    }
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationAcceptsOnlyNonnegativeUiPlaneWithinContainer) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 3440u, 1440u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());
    const auto ocr = empty_ocr(frame, geometry);
    auto locator = current_subtitle_plane(frame, 0u, false);

    const float local_plane = 4.0f / 3440.0f;
    for (const float candidate : {
           0.0f,
           local_plane,
           v2::direct_container_limit,
         }) {
      locator[v2::subtitle_plane_adaptive_offset + 3u] = std::bit_cast<std::uint32_t>(candidate);
      EXPECT_TRUE(subtitle_records_match_frame(
        word_bytes(ocr), word_bytes(locator), frame
      ));
    }
    for (const float candidate : {
           -local_plane,
           -v2::direct_container_limit,
           std::nextafter(
             -v2::direct_container_limit,
             -std::numeric_limits<float>::infinity()
           ),
           std::nextafter(
             v2::direct_container_limit,
             std::numeric_limits<float>::infinity()
           ),
           std::numeric_limits<float>::quiet_NaN(),
         }) {
      locator[v2::subtitle_plane_adaptive_offset + 3u] = std::bit_cast<std::uint32_t>(candidate);
      EXPECT_FALSE(subtitle_records_match_frame(
        word_bytes(ocr), word_bytes(locator), frame
      ));
    }
  }

}  // namespace
#endif

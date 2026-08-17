/**
 * @file tests/unit/platform/test_sbs_debug_dump_async.cpp
 * @brief Behavioral tests for Dump 3D publication lifetime and request ordering.
 */
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
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

  #include <nlohmann/json.hpp>

  #include <src/generated/depth_coordinate_v2_contract.h>
  #include <src/host_sbs_gpu_trace.h>
  #include <src/host_sbs_shader_cache.h>
  #include <src/platform/windows/sbs_debug_dump.h>
  #include <src/platform/windows/sbs_debug_dump_async.h>

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

  platf::sbs_debug::frame gpu_trace_frame() {
    platf::sbs_debug::frame frame;
    frame.model_width = 770;
    frame.model_height = 434;
    frame.matched_frame_id = 41u;
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
      .video_region = false,
      .authority = models::depth_analysis_authority_e::full_source,
    };
    return frame;
  }

  std::vector<std::uint8_t> canonical_gpu_trace_ring(
    const platf::sbs_debug::frame &frame,
    const gpu_trace::submission_class_e submission_class =
      gpu_trace::submission_class_e::force_infer,
    const cuda_conditional_graph::branch_e branch =
      cuda_conditional_graph::branch_e::infer,
    const bool label_authenticated_depth = true
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
      token, cuda_conditional_graph::work_flag_e::subtitle_observation
    );
    const auto proposal = cuda_conditional_graph::make_proposal(branch, token);
    const auto receipt = cuda_conditional_graph::resolve_proposal(
      proposal, request, false
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
          cuda_conditional_graph::work_flag_e::subtitle_observation));
    const bool branch_gated =
      submission_class == gpu_trace::submission_class_e::gpu_undecided;
    const bool subtitle_held =
      branch_gated && branch == cuda_conditional_graph::branch_e::reuse;
    set(
      gpu_trace::record_word_e::subtitle_disposition,
      static_cast<std::uint32_t>(
        subtitle_held ? gpu_trace::subtitle_disposition_e::held_with_depth :
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
        static_cast<std::uint32_t>(frame.model_width));
    set(gpu_trace::record_word_e::field_height,
        static_cast<std::uint32_t>(frame.model_height));
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
    constexpr std::uint32_t owner_generation = 1u;
    const auto target = std::bit_cast<std::uint32_t>(0.01f);
    words[locator + 0u] = v2::subtitle_locator_state_schema;
    words[locator + 1u] = v2::subtitle_locator_state_tag;
    words[locator + 2u] = 1u | 4u;
    words[locator + 3u] = owner_generation;
    words[locator + 4u] = 1u;
    words[locator + 10u] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation
    );
    words[locator + 11u] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation >> 32u
    );
    words[locator + 18u] = target;
    words[locator + 19u] = owner_generation;
    words[locator + 20u] = 1u;
    words[locator + 21u] = 1u;
    const auto locator_frame_id =
      subtitle_held ? frame.matched_frame_id - 1u : frame.matched_frame_id;
    words[locator + 22u] = static_cast<std::uint32_t>(locator_frame_id);
    words[locator + 23u] = static_cast<std::uint32_t>(locator_frame_id >> 32u);
    words[locator + 24u] = 1u;
    words[locator + 26u] = subtitle_scene_epoch;
    words[locator + 27u] = static_cast<std::uint32_t>(frame.model_width);
    words[locator + 28u] = static_cast<std::uint32_t>(frame.model_height);
    const auto condition =
      base + gpu_trace::word_index(gpu_trace::record_word_e::subtitle_condition_begin);
    words[condition + 0u] = v2::subtitle_condition_param_schema;
    words[condition + 1u] = v2::subtitle_condition_param_tag;
    words[condition + 2u] = 1u;
    words[condition + 4u] = 1u;
    words[condition + 5u] = target;
    const auto observation_timestamp_us = 1'000'000u + frame.matched_frame_id;
    set(gpu_trace::record_word_e::observation_timestamp_low,
        static_cast<std::uint32_t>(observation_timestamp_us));
    set(gpu_trace::record_word_e::observation_timestamp_high,
        static_cast<std::uint32_t>(observation_timestamp_us >> 32u));

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
    words[11] = static_cast<std::uint32_t>(frame.model_width);
    words[12] = static_cast<std::uint32_t>(frame.model_height);
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

  std::array<std::uint32_t, v2::subtitle_locator_state_word_count>
  two_line_subtitle_state(
    const platf::sbs_debug::frame &frame,
    const subtitle_rect_t first,
    const subtitle_rect_t second
  ) {
    std::array<std::uint32_t, v2::subtitle_locator_state_word_count> words {};
    constexpr std::uint32_t owner_generation = 9u;
    words[0u] = v2::subtitle_locator_state_schema;
    words[1u] = v2::subtitle_locator_state_tag;
    words[2u] = 1u | 4u;
    words[3u] = owner_generation;
    words[4u] = 2u;
    words[5u] = std::min(first[0], second[0]);
    words[6u] = std::min(first[1], second[1]);
    words[7u] = std::max(first[2], second[2]);
    words[8u] = std::max(first[3], second[3]);
    words[9u] = (first[2] - first[0]) * (first[3] - first[1]) +
                (second[2] - second[0]) * (second[3] - second[1]);
    words[10u] = static_cast<std::uint32_t>(frame.depth_input_region.analysis_generation);
    words[11u] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation >> 32u
    );
    words[18u] = std::bit_cast<std::uint32_t>(
      4.0f / (2.0f * static_cast<float>(frame.depth_input_region.width()))
    );
    words[19u] = owner_generation;
    words[20u] = 2u;
    words[21u] = 1u;
    words[22u] = static_cast<std::uint32_t>(frame.matched_frame_id);
    words[23u] = static_cast<std::uint32_t>(frame.matched_frame_id >> 32u);
    words[24u] = 2u;
    words[26u] = subtitle_scene_epoch;
    words[27u] = static_cast<std::uint32_t>(frame.model_width);
    words[28u] = static_cast<std::uint32_t>(frame.model_height);
    for (std::size_t index = 0u; index < first.size(); ++index) {
      words[v2::subtitle_locator_owner_offset + index] = first[index];
      words[v2::subtitle_locator_owner_offset + 4u + index] = second[index];
      words[v2::subtitle_locator_current_offset + index] = first[index];
      words[v2::subtitle_locator_current_offset + 4u + index] = second[index];
    }
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
    words[18] = std::bit_cast<std::uint32_t>(
      4.0f / (2.0f * static_cast<float>(frame.depth_input_region.width()))
    );
    words[22] = static_cast<std::uint32_t>(frame.matched_frame_id);
    words[23] = static_cast<std::uint32_t>(frame.matched_frame_id >> 32u);
    words[25] = grace;
    words[26] = subtitle_scene_epoch;
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

  TEST(SbsDebugDumpGpuTraceTest, HeldOrdinaryReuseAuthenticatesTheOlderPublishedSubtitleTuple) {
    auto completed = gpu_trace_frame();
    completed.gpu_undecided_completion = true;
    auto evidence = completed;
    --evidence.matched_frame_id;
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
    set_trace_word(ring, condition_base + 5u, locator[18u]);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, completed));
    EXPECT_TRUE(dump_detail::subtitle_records_match_completion(
      word_bytes(ocr), word_bytes(locator), ring, completed, subtitle_scene_epoch
    ));

    set_trace_word(
      ring,
      locator_base + v2::subtitle_locator_owner_offset,
      locator[v2::subtitle_locator_owner_offset] + 1u
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
    reject(locator + 4u, v2::subtitle_locator_rectangle_capacity + 1u);
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
    EXPECT_EQ(trace_word(reused, locator + 22u), frame.matched_frame_id - 1u);
    auto falsely_current_locator = reused;
    set_trace_word(
      falsely_current_locator,
      locator + 22u,
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
    const auto changed_target = trace_word(changed, current_locator + 18u) ^ 1u;
    set_trace_word(changed, current_locator + 18u, changed_target);
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

  TEST(SbsDebugDumpGpuTraceTest, InactiveLocatorRequiresCanonicalZeroConditionVerdict) {
    const auto frame = gpu_trace_frame();
    const auto canonical = canonical_gpu_trace_ring(frame);
    const auto base = gpu_trace::record_base(0u);
    const auto locator = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    const auto condition = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );

    auto inactive = canonical;
    set_trace_word(inactive, locator + 2u, 0u);
    set_trace_word(inactive, locator + 3u, 0u);
    set_trace_word(inactive, locator + 4u, 0u);
    set_trace_word(inactive, locator + 12u, 0u);
    set_trace_word(inactive, locator + 18u, 0u);
    set_trace_word(inactive, locator + 19u, 0u);
    set_trace_word(inactive, locator + 20u, 0u);
    set_trace_word(inactive, locator + 21u, 0u);
    set_trace_word(inactive, locator + 24u, 0u);
    set_trace_word(inactive, locator + 25u, 0u);
    set_trace_word(inactive, locator + v2::subtitle_locator_kind_word, 0u);
    for (std::size_t index = 0u; index < v2::subtitle_condition_param_word_count; ++index) {
      set_trace_word(inactive, condition + index, 0u);
    }
    EXPECT_TRUE(dump_detail::gpu_trace_ring_is_canonical(inactive, frame));

    auto stale_condition = inactive;
    set_trace_word(
      stale_condition, condition, v2::subtitle_condition_param_schema
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(stale_condition, frame));

    auto missing_active_condition = canonical;
    for (std::size_t index = 0u; index < v2::subtitle_condition_param_word_count; ++index) {
      set_trace_word(missing_active_condition, condition + index, 0u);
    }
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(missing_active_condition, frame));
  }

  TEST(SbsDebugDumpGpuTraceTest, ProvisionalConditionUsesEphemeralTupleAndStructuralBounds) {
    const auto frame = gpu_trace_frame();
    auto ring = canonical_gpu_trace_ring(frame);
    const auto base = gpu_trace::record_base(0u);
    const auto locator = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_locator_begin
    );
    const auto condition = base + gpu_trace::word_index(
      gpu_trace::record_word_e::subtitle_condition_begin
    );
    constexpr subtitle_rect_t owner {203u, 369u, 566u, 384u};
    constexpr subtitle_rect_t pending {342u, 368u, 415u, 386u};
    constexpr subtitle_rect_t cover {337u, 364u, 420u, 390u};
    const auto provisional_target = std::bit_cast<std::uint32_t>(
      6.0f / (2.0f * 1920.0f)
    );
    set_trace_word(
      ring, locator + 2u,
      1u | 2u | 4u | v2::subtitle_locator_provisional_current_flag
    );
    set_trace_word(ring, locator + 12u, 1u);
    set_trace_word(ring, locator + 20u, 1u);
    set_trace_word(ring, locator + 21u, 0u);
    set_trace_word(ring, locator + 24u, 2u);
    set_trace_word(
      ring, locator + v2::subtitle_locator_provisional_target_word, provisional_target
    );
    set_trace_word(ring, locator + v2::subtitle_locator_provisional_fade_word, 2u);
    for (std::size_t coordinate = 0u; coordinate < 4u; ++coordinate) {
      set_trace_word(
        ring, locator + v2::subtitle_locator_owner_offset + coordinate, owner[coordinate]
      );
      set_trace_word(
        ring, locator + v2::subtitle_locator_pending_offset + coordinate, pending[coordinate]
      );
      set_trace_word(
        ring, locator + v2::subtitle_locator_current_offset + coordinate, cover[coordinate]
      );
    }
    set_trace_word(ring, condition + 4u, 2u);
    set_trace_word(ring, condition + 5u, provisional_target);
    ASSERT_TRUE(dump_detail::gpu_trace_ring_is_canonical(ring, frame));

    auto durable_tuple = ring;
    set_trace_word(
      durable_tuple, condition + 5u, trace_word(ring, locator + 18u)
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(durable_tuple, frame));

    auto widened_cover = ring;
    set_trace_word(
      widened_cover, locator + v2::subtitle_locator_current_offset + 0u,
      pending[0] + 1u
    );
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(widened_cover, frame));

    // Match equality belongs to established owner continuity, never the provisional bridge.
    auto iou_equality = ring;
    constexpr subtitle_rect_t equal_owner {100u, 360u, 200u, 370u};
    constexpr subtitle_rect_t equal_pending {125u, 360u, 225u, 370u};
    constexpr subtitle_rect_t equal_cover {120u, 356u, 230u, 374u};
    for (std::size_t coordinate = 0u; coordinate < 4u; ++coordinate) {
      set_trace_word(
        iou_equality, locator + v2::subtitle_locator_owner_offset + coordinate,
        equal_owner[coordinate]
      );
      set_trace_word(
        iou_equality, locator + v2::subtitle_locator_pending_offset + coordinate,
        equal_pending[coordinate]
      );
      set_trace_word(
        iou_equality, locator + v2::subtitle_locator_current_offset + coordinate,
        equal_cover[coordinate]
      );
    }
    EXPECT_FALSE(dump_detail::gpu_trace_ring_is_canonical(iou_equality, frame));
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

  TEST(SbsDebugDumpAsyncTest, ActiveBasePreviewsAreDeclaredInTheDumpManifest) {
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
    for (const std::string_view name : {
           "shadow_base_final_parallax.f32",
           "shadow_base_final_parallax_shape.json",
           "shadow_base_final_parallax.png",
           "shadow_base_final_parallax_heat.png",
         }) {
      EXPECT_NE(
        source.find("artifacts[\"" + std::string(name) + "\"]"),
        std::string::npos
      ) << name;
    }
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
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));

    locator[18] = std::bit_cast<std::uint32_t>(
      std::nextafter(
        v2::direct_container_limit,
        std::numeric_limits<float>::infinity()
      )
    );
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));

    locator = cached_grace_state(
      frame, geometry, v2::subtitle_locator_death_grace_observations
    );

    locator[25] = v2::subtitle_locator_death_grace_observations + 1u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationAcceptsSignedLocalPlaneWithinContainer) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 3440u, 1440u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());
    const auto ocr = empty_ocr(frame, geometry);
    auto locator = cached_grace_state(frame, geometry, 1u);

    const float local_plane = 4.0f / 3440.0f;
    for (const float candidate : {
           -v2::direct_container_limit,
           -local_plane,
           0.0f,
           local_plane,
           v2::direct_container_limit,
         }) {
      locator[18] = std::bit_cast<std::uint32_t>(candidate);
      EXPECT_TRUE(subtitle_records_match_frame(
        word_bytes(ocr), word_bytes(locator), frame
      ));
    }
    for (const float candidate : {
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
      locator[18] = std::bit_cast<std::uint32_t>(candidate);
      EXPECT_FALSE(subtitle_records_match_frame(
        word_bytes(ocr), word_bytes(locator), frame
      ));
    }
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationKeepsResetStateOutOfGrace) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());

    constexpr subtitle_rect_t first {180u, 350u, 590u, 360u};
    constexpr subtitle_rect_t second {180u, 364u, 500u, 374u};
    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u;
    ocr[3u] = 2u;
    ocr[4u] = 2u;
    store_ocr_pair(ocr, 0u, first, first);
    store_ocr_pair(ocr, 1u, second, second);

    auto reset = two_line_subtitle_state(frame, first, second);
    reset[2u] = 1u | 8u;  // owner + target-reset
    reset[18u] = 0u;
    reset[19u] = 0u;
    reset[20u] = 0u;
    reset[24u] = 0u;
    std::fill_n(
      reset.begin() + v2::subtitle_locator_current_offset,
      v2::subtitle_locator_rectangle_capacity * 4u,
      0u
    );
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(reset), frame
    ));

    // An unreliable or excessive-residual observation retains only the owner transaction. Its
    // zero reset marker is not a cached plane and therefore cannot enter death grace.
    reset[25u] = 1u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(reset), frame
    ));
    reset[25u] = 0u;
    reset[18u] = std::bit_cast<std::uint32_t>(-0.0f);
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(reset), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationBoundsUnreliableOwnerHold) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());

    constexpr subtitle_rect_t first {180u, 350u, 590u, 360u};
    constexpr subtitle_rect_t second {180u, 364u, 500u, 374u};
    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u;
    ocr[3u] = 2u;
    ocr[4u] = 2u;
    store_ocr_pair(ocr, 0u, first, first);
    store_ocr_pair(ocr, 1u, second, second);

    auto held = two_line_subtitle_state(frame, first, second);
    held[21u] = 0u;  // A hold is an ordinary continuing-owner observation.
    for (std::uint32_t count = 1u;
         count <= v2::subtitle_target_max_unreliable_holds;
         ++count) {
      held[25u] = count;
      EXPECT_TRUE(subtitle_records_match_frame(
        word_bytes(ocr), word_bytes(held), frame
      ));
    }

    held[25u] = v2::subtitle_target_max_unreliable_holds + 1u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(held), frame
    ));

    // Word 25 is an unreliable-measurement hold only while the owner retains a valid target. It
    // cannot be attached to a birth/handoff/death event. An observation with no matched current
    // authority preserves the counter without aging it and conditions exact Base for that frame.
    held[25u] = 1u;
    held[21u] = 3u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(held), frame
    ));
    held[21u] = 0u;
    held[20u] = 0u;
    std::fill_n(
      held.begin() + v2::subtitle_locator_current_offset,
      v2::subtitle_locator_rectangle_capacity * 4u,
      0u
    );
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(held), frame
    ));
    held[2u] = 1u | 8u;
    held[18u] = 0u;
    held[19u] = 0u;
    held[24u] = 0u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(held), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationBindsCurrentToSelectedOcrOrder) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());

    constexpr subtitle_rect_t first {180u, 350u, 590u, 360u};
    constexpr subtitle_rect_t second {180u, 364u, 500u, 374u};
    // This is valid OCR8 evidence but deliberately fails SLR13's 48-cell/two-to-one core gate.
    constexpr subtitle_rect_t unselected {40u, 380u, 60u, 400u};
    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u;
    ocr[3u] = 3u;
    ocr[4u] = 3u;
    store_ocr_pair(ocr, 0u, first, first);
    store_ocr_pair(ocr, 1u, second, second);
    store_ocr_pair(ocr, 2u, unselected, unselected);
    ASSERT_TRUE(dump_detail::subtitle_ocr_record_is_canonical_for_frame(
      word_bytes(ocr), frame
    ));

    const auto valid = two_line_subtitle_state(frame, first, second);
    ASSERT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(valid), frame
    ));

    auto foreign_scene_epoch = valid;
    foreign_scene_epoch[26u] = subtitle_scene_epoch + 1u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(foreign_scene_epoch), frame
    ));

    const auto store_current = [](
      auto &words,
      const std::size_t slot,
      const subtitle_rect_t rectangle
    ) {
      std::copy(
        rectangle.begin(), rectangle.end(),
        words.begin() + v2::subtitle_locator_current_offset + slot * 4u
      );
    };

    auto selected_subsequence = valid;
    selected_subsequence[20u] = 1u;
    store_current(selected_subsequence, 0u, second);
    store_current(selected_subsequence, 1u, {});
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(selected_subsequence), frame
    ));

    auto reversed = valid;
    store_current(reversed, 0u, second);
    store_current(reversed, 1u, first);
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(reversed), frame
    ));

    auto duplicate = valid;
    store_current(duplicate, 1u, first);
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(duplicate), frame
    ));

    auto outside_selection = valid;
    outside_selection[20u] = 1u;
    store_current(outside_selection, 0u, unselected);
    store_current(outside_selection, 1u, {});
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(outside_selection), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationAuthenticatesProvisionalExactPair) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());
    constexpr subtitle_rect_t owner {203u, 369u, 566u, 384u};
    constexpr subtitle_rect_t pending {342u, 368u, 415u, 386u};
    constexpr subtitle_rect_t cover {337u, 364u, 420u, 390u};

    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u;
    ocr[3u] = 1u;
    ocr[4u] = 1u;
    store_ocr_pair(ocr, 0u, pending, cover);

    std::array<std::uint32_t, v2::subtitle_locator_state_word_count> locator {};
    constexpr std::uint32_t generation = 9u;
    locator[0u] = v2::subtitle_locator_state_schema;
    locator[1u] = v2::subtitle_locator_state_tag;
    locator[2u] = 1u | 2u | 4u | v2::subtitle_locator_provisional_current_flag;
    locator[3u] = generation;
    locator[4u] = 1u;
    std::copy(owner.begin(), owner.end(), locator.begin() + 5u);
    locator[9u] = (owner[2] - owner[0]) * (owner[3] - owner[1]);
    locator[10u] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation
    );
    locator[11u] = static_cast<std::uint32_t>(
      frame.depth_input_region.analysis_generation >> 32u
    );
    locator[12u] = 1u;
    std::copy(pending.begin(), pending.end(), locator.begin() + 13u);
    locator[17u] = (pending[2] - pending[0]) * (pending[3] - pending[1]);
    locator[18u] = std::bit_cast<std::uint32_t>(2.0f / (2.0f * 1920.0f));
    locator[19u] = generation;
    locator[20u] = 1u;
    locator[22u] = static_cast<std::uint32_t>(frame.matched_frame_id);
    locator[23u] = static_cast<std::uint32_t>(frame.matched_frame_id >> 32u);
    locator[24u] = 2u;
    locator[26u] = subtitle_scene_epoch;
    locator[27u] = static_cast<std::uint32_t>(frame.model_width);
    locator[28u] = static_cast<std::uint32_t>(frame.model_height);
    locator[v2::subtitle_locator_provisional_target_word] =
      std::bit_cast<std::uint32_t>(6.0f / (2.0f * 1920.0f));
    locator[v2::subtitle_locator_provisional_fade_word] = 2u;
    std::copy(
      owner.begin(), owner.end(), locator.begin() + v2::subtitle_locator_owner_offset
    );
    std::copy(
      pending.begin(), pending.end(), locator.begin() + v2::subtitle_locator_pending_offset
    );
    std::copy(
      cover.begin(), cover.end(), locator.begin() + v2::subtitle_locator_current_offset
    );
    ASSERT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));

    auto wrong_cover = locator;
    wrong_cover[v2::subtitle_locator_current_offset] = cover[0] - 1u;
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(wrong_cover), frame
    ));

    auto wrong_geometry = locator;
    constexpr subtitle_rect_t unrelated {100u, 330u, 500u, 340u};
    std::copy(unrelated.begin(), unrelated.end(), wrong_geometry.begin() + 5u);
    wrong_geometry[9u] =
      (unrelated[2] - unrelated[0]) * (unrelated[3] - unrelated[1]);
    std::copy(
      unrelated.begin(), unrelated.end(),
      wrong_geometry.begin() + v2::subtitle_locator_owner_offset
    );
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(wrong_geometry), frame
    ));
  }

  TEST(SbsDebugDumpAsyncTest, NativeSubtitleValidationMirrorsStrictCornerCoreFilter) {
    constexpr models::depth_tensor_content_rect_t content {0u, 0u, 770u, 434u};
    const auto frame = subtitle_frame(content, 1920u, 1080u);
    const auto geometry = models::fit_subtitle_analysis_geometry(
      frame.depth_input_region.width(), frame.depth_input_region.height(),
      {frame.model_width, frame.model_height}, content
    );
    ASSERT_TRUE(geometry.valid());
    const auto edge_threshold = content.width() / v2::subtitle_locator_corner_edge_divisor;
    const auto bottom_threshold =
      geometry.roi_bottom - v2::subtitle_locator_corner_bottom_rows;
    ASSERT_EQ(edge_threshold, 24u);

    const subtitle_rect_t center {
      200u, bottom_threshold - 10u, 400u, bottom_threshold,
    };
    const subtitle_rect_t edge_equal {
      edge_threshold, bottom_threshold - 10u,
      edge_threshold + 116u, bottom_threshold,
    };
    auto ocr = empty_ocr(frame, geometry);
    ocr[2u] = 1u;
    ocr[3u] = 2u;
    ocr[4u] = 2u;
    store_ocr_pair(ocr, 0u, edge_equal, edge_equal);
    store_ocr_pair(ocr, 1u, center, center);
    const auto locator = two_line_subtitle_state(frame, edge_equal, center);
    EXPECT_TRUE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));

    // Moving the same core one cell strictly inside the corner threshold removes it from the
    // selected ordinary component, so an SLR state claiming both lines must fail authentication.
    const subtitle_rect_t strict_corner {
      edge_threshold - 1u, bottom_threshold - 10u,
      edge_threshold + 115u, bottom_threshold,
    };
    store_ocr_pair(ocr, 0u, strict_corner, strict_corner);
    EXPECT_FALSE(subtitle_records_match_frame(
      word_bytes(ocr), word_bytes(locator), frame
    ));
  }
}  // namespace
#endif

/**
 * @file tests/unit/test_video.cpp
 * @brief Test src/video.*.
 */
#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <src/depth_coordinate_v2.h>
#include <src/generated/sbs_adaptive_state_contract.h>
#include <src/host_sbs_shader_cache.h>
#include <src/nvenc/nvenc_base.h>
#include <src/nvenc/nvenc_config.h>
#include <src/platform/windows/video_dom_client.h>
#include <src/video.h>
#include <src/video_colorspace.h>
#include <src/video_depth_estimator.h>
#include <thread>
#include <tuple>
#include <vector>

#ifdef _WIN32
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <wrl/client.h>

namespace platf::dxgi {
  int init();
}
#endif

namespace {
  TEST(AdaptiveMotionProbeTest, ValidatesExactIdentityAndShadowVerdict) {
    std::array<std::uint32_t, models::adaptive_motion_probe_word_count> words {};
    constexpr std::uint64_t current_id = 0x0000000200000003ull;
    constexpr std::uint64_t baseline_id = 0x0000000100000002ull;
    words[0] = models::adaptive_motion_probe_contract_tag;
    words[1] = models::adaptive_motion_probe_settled_flags;
    words[2] = static_cast<std::uint32_t>(current_id);
    words[3] = static_cast<std::uint32_t>(current_id >> 32u);
    words[4] = static_cast<std::uint32_t>(baseline_id);
    words[5] = static_cast<std::uint32_t>(baseline_id >> 32u);
    words[6] = 7u;
    words[7] = 9u;
    words[8] = 3u;
    words[10] = 1u;
    words[11] = 770u * 434u;
    words[21] = 770u * 129u;

    models::adaptive_motion_probe_sample sample;
    ASSERT_TRUE(models::decode_adaptive_motion_probe_words(
      words, current_id, baseline_id, 770, 434, sample
    ));
    EXPECT_EQ(
      models::adaptive_motion_probe_exact_verdict(sample),
      models::adaptive_motion_probe_exact_verdict_e::quiet_evidence
    );

    words[13] = 1u;
    words[18] = 1u;
    ASSERT_TRUE(models::decode_adaptive_motion_probe_words(
      words, current_id, baseline_id, 770, 434, sample
    ));
    EXPECT_EQ(
      models::adaptive_motion_probe_exact_verdict(sample),
      models::adaptive_motion_probe_exact_verdict_e::motion_veto
    );
    EXPECT_FALSE(models::decode_adaptive_motion_probe_words(
      words, current_id, baseline_id + 1u, 770, 434, sample
    ));
    EXPECT_FALSE(models::decode_adaptive_motion_probe_words(
      words, baseline_id - 1u, baseline_id, 770, 434, sample
    ));

    words[1] &= ~models::adaptive_motion_probe_flag_hard_cut_count_valid;
    EXPECT_TRUE(models::decode_adaptive_motion_probe_words(
      words, current_id, baseline_id, 770, 434, sample
    ));
    EXPECT_EQ(
      models::adaptive_motion_probe_exact_verdict(sample),
      models::adaptive_motion_probe_exact_verdict_e::invalid
    );
    words[1] = models::adaptive_motion_probe_settled_flags;
    words[6] = 0xFFFFFFFFu;  // Reserved overflow sentinel from the packed uint counter contract.
    EXPECT_FALSE(models::decode_adaptive_motion_probe_words(
      words, current_id, baseline_id, 770, 434, sample
    ));
  }

  TEST(AdaptiveMotionProbeTest, RejectsMalformedCountersMaximaAndState) {
    constexpr std::uint64_t current_id = 11u;
    constexpr std::uint64_t baseline_id = 10u;
    constexpr std::uint32_t area = 16u * 16u;
    const auto valid_words = [] {
      std::array<std::uint32_t, models::adaptive_motion_probe_word_count> words {};
      words[0] = models::adaptive_motion_probe_contract_tag;
      words[1] = models::adaptive_motion_probe_settled_flags;
      words[2] = static_cast<std::uint32_t>(current_id);
      words[4] = static_cast<std::uint32_t>(baseline_id);
      words[6] = 1u;
      words[7] = 8u;
      words[8] = 3u;
      words[10] = 1u;
      words[11] = area;
      words[21] = 16u * 6u;
      return words;
    };
    const auto rejects = [&](const std::size_t word, const std::uint32_t value) {
      auto words = valid_words();
      words[word] = value;
      models::adaptive_motion_probe_sample sample;
      return !models::decode_adaptive_motion_probe_words(
        words, current_id, baseline_id, 16, 16, sample
      );
    };

    EXPECT_TRUE(rejects(0u, 0u));
    EXPECT_TRUE(rejects(1u, models::adaptive_motion_probe_settled_flags | (1u << 31u)));
    EXPECT_TRUE(rejects(7u, models::adaptive_motion_probe_max_exact_numeric_counter + 1u));
    EXPECT_TRUE(rejects(8u, sbs_adaptive_state::known_cut_flag_mask + 1u));
    EXPECT_TRUE(rejects(9u, sbs_adaptive_state::known_analysis_flag_mask + 1u));
    EXPECT_TRUE(rejects(10u, 5u));
    EXPECT_TRUE(rejects(11u, area + 1u));
    EXPECT_TRUE(rejects(13u, 1u));  // exact changes cannot exceed the zero default tile maximum.
    EXPECT_TRUE(rejects(17u, std::bit_cast<std::uint32_t>(
      std::numeric_limits<float>::quiet_NaN()
    )));
    EXPECT_TRUE(rejects(21u, area + 1u));
    EXPECT_TRUE(rejects(24u, std::bit_cast<std::uint32_t>(1.0f)));
  }

  TEST(RenderedContentTimestampTest, MatchedT0IsPreservedWhileCurrentCadenceIsT1) {
    const auto t0 = std::chrono::steady_clock::time_point {10ms};
    const auto t1 = std::chrono::steady_clock::time_point {20ms};

    const auto rendered = video::detail::select_rendered_content_timestamp(
      false,
      std::nullopt,
      true,
      t0,
      t0,
      t1,
      t1
    );

    EXPECT_EQ(rendered, t0);
    EXPECT_EQ(t1, std::chrono::steady_clock::time_point {20ms});
  }

  TEST(RenderedContentTimestampTest, RepeatedOutputKeepsPriorRenderedContentTime) {
    const auto t0 = std::chrono::steady_clock::time_point {10ms};
    const auto t1 = std::chrono::steady_clock::time_point {20ms};

    EXPECT_EQ(
      video::detail::select_rendered_content_timestamp(
        true,
        t0,
        false,
        std::nullopt,
        std::nullopt,
        t1,
        t1
      ),
      t0
    );
  }

  TEST(RenderedContentTimestampTest, CursorOnlyPresentationKeepsRetainedContentTime) {
    const auto t0 = std::chrono::steady_clock::time_point {10ms};
    const auto t1 = std::chrono::steady_clock::time_point {20ms};

    EXPECT_EQ(
      video::detail::select_rendered_content_timestamp(
        false,
        std::nullopt,
        false,
        std::nullopt,
        std::nullopt,
        t0,
        t1
      ),
      t0
    );
  }

  TEST(RenderedContentTimestampTest, ProcessingTelemetryPrefersContentButCadenceStaysSeparate) {
    const auto content_t0 = std::chrono::steady_clock::time_point {10ms};
    const auto presentation_t1 = std::chrono::steady_clock::time_point {20ms};

    EXPECT_EQ(
      video::detail::select_processing_timestamp(content_t0, presentation_t1),
      content_t0
    );
    EXPECT_EQ(presentation_t1, std::chrono::steady_clock::time_point {20ms});
  }

  std::string read_source_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
      return {};
    }
    return {
      std::istreambuf_iterator<char> {input},
      std::istreambuf_iterator<char> {}
    };
  }

  float apply_color_vector(const float (&color_vector)[4], float red, float green, float blue) {
    return color_vector[0] * red + color_vector[1] * green + color_vector[2] * blue + color_vector[3];
  }

  float reference_srgb_code_to_bt709_code(float value) {
    const auto x = std::clamp(value, 0.0f, 1.0f);
    const auto linear = x <= 0.04045f ?
                          x / 12.92f :
                          std::pow((x + 0.055f) / 1.055f, 2.4f);
    return linear < 0.018f ?
             4.5f * linear :
             1.099f * std::pow(linear, 0.45f) - 0.099f;
  }

  struct structural_evidence_t {
    float change_fraction;
    float current_support_fraction;
    float previous_support_fraction;
    float common_support_fraction;
  };

  structural_evidence_t structural_evidence(
    const std::vector<float> &current,
    const std::vector<float> &previous,
    int width,
    int height
  ) {
    constexpr std::array<std::array<int, 2>, 5> offsets {{{{0, 0}}, {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}}}};
    const auto sample = [&](const std::vector<float> &values, int sx, int sy) {
      sx = std::clamp(sx, 0, width - 1);
      sy = std::clamp(sy, 0, height - 1);
      return values[static_cast<std::size_t>(sy * width + sx)];
    };
    std::size_t changed = 0;
    std::size_t current_supported = 0;
    std::size_t previous_supported = 0;
    std::size_t common_supported = 0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        std::array<float, 5> current_samples {};
        std::array<float, 5> previous_samples {};
        for (std::size_t index = 0; index < offsets.size(); ++index) {
          current_samples[index] =
            sample(current, x + offsets[index][0], y + offsets[index][1]);
          previous_samples[index] =
            sample(previous, x + offsets[index][0], y + offsets[index][1]);
        }
        unsigned current_comparisons = 0;
        unsigned previous_comparisons = 0;
        unsigned common_comparisons = 0;
        unsigned ordering_flips = 0;
        for (std::size_t first = 0; first < 4; ++first) {
          for (std::size_t second = first + 1; second < 5; ++second) {
            const float current_delta =
              current_samples[first] - current_samples[second];
            const float previous_delta =
              previous_samples[first] - previous_samples[second];
            const float current_threshold = std::max(
              0.0001f,
              0.04f * std::max(
                        std::abs(current_samples[first]),
                        std::abs(current_samples[second])
                      )
            );
            const float previous_threshold = std::max(
              0.0001f,
              0.04f * std::max(
                        std::abs(previous_samples[first]),
                        std::abs(previous_samples[second])
                      )
            );
            const bool current_reliable =
              std::abs(current_delta) >= current_threshold;
            const bool previous_reliable =
              std::abs(previous_delta) >= previous_threshold;
            current_comparisons += current_reliable ? 1u : 0u;
            previous_comparisons += previous_reliable ? 1u : 0u;
            if (current_reliable && previous_reliable) {
              ++common_comparisons;
              if ((current_delta < 0.0f) != (previous_delta < 0.0f)) {
                ++ordering_flips;
              }
            }
          }
        }
        current_supported += current_comparisons >= 4 ? 1u : 0u;
        previous_supported += previous_comparisons >= 4 ? 1u : 0u;
        common_supported += common_comparisons >= 4 ? 1u : 0u;
        if (common_comparisons >= 4 && ordering_flips >= 2 && ordering_flips * 2 >= common_comparisons) {
          ++changed;
        }
      }
    }
    const auto denominator = static_cast<float>(width * height);
    return {
      static_cast<float>(changed) / denominator,
      static_cast<float>(current_supported) / denominator,
      static_cast<float>(previous_supported) / denominator,
      static_cast<float>(common_supported) / denominator,
    };
  }

  float structural_change_fraction(
    const std::vector<float> &current,
    const std::vector<float> &previous,
    int width,
    int height
  ) {
    return structural_evidence(current, previous, width, height).change_fraction;
  }

  constexpr unsigned cut_flag_geometry_armed = 1u;
  constexpr unsigned cut_flag_appearance_armed = 2u;
  constexpr unsigned cut_flag_geometry_low_once = 4u;
  constexpr unsigned cut_flag_appearance_quiet_once = 8u;
  constexpr unsigned cut_flag_latched = 16u;
  constexpr unsigned cut_flag_appearance_recovery = 32u;
  constexpr unsigned cut_flag_geometry_confirmation_pending = 64u;
  constexpr unsigned cut_flags_ready =
    cut_flag_geometry_armed | cut_flag_appearance_armed;

  struct shot_cut_state_t {
    unsigned cut_flags = cut_flags_ready;
    float scene_age = 8.0f;
    float depth_change_baseline = 0.0f;
    float appearance_change_baseline = 0.0f;
    float model_input_history_state = 1.0f;
    bool initialized = true;
  };

  bool advance_shot_cut(
    shot_cut_state_t &state,
    float depth_change_fraction,
    float raw_rgb_change_fraction,
    float structural_change_fraction,
    float current_structural_support_fraction = 1.0f,
    float previous_structural_support_fraction = 1.0f,
    float common_structural_support_fraction = 1.0f,
    float brightness_rise_fraction = 1.0f,
    float brightness_fall_fraction = 0.0f,
    unsigned stream_frame_delta = 1u
  ) {
    state.scene_age = state.initialized ?
                        std::min(
                          state.scene_age + static_cast<float>(std::clamp(
                                              stream_frame_delta,
                                              1u,
                                              65535u
                                            )),
                          65535.0f
                        ) :
                        0.0f;
    const bool model_input_history_valid =
      state.model_input_history_state > 0.5f;
    const bool model_input_history_gap =
      state.model_input_history_state > 1.5f &&
      state.model_input_history_state < 2.5f;
    const bool low_structure_scene =
      state.model_input_history_state > 2.5f &&
      state.model_input_history_state < 3.5f;
    const bool current_structure_reliable =
      current_structural_support_fraction >= 0.01f;
    const bool previous_structure_reliable =
      previous_structural_support_fraction >= 0.01f;
    const bool common_structure_reliable =
      common_structural_support_fraction >= 0.01f;
    const bool common_structure_representative =
      common_structure_reliable &&
      common_structural_support_fraction >=
        0.50f * std::min(
                  current_structural_support_fraction,
                  previous_structural_support_fraction
                );
    state.appearance_change_baseline =
      std::clamp(state.appearance_change_baseline, 0.0f, 1.0f);
    const bool broad_appearance_proposal =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f &&
      structural_change_fraction >= 0.03f;
    const bool localized_appearance_proposal =
      model_input_history_valid &&
      current_structure_reliable &&
      previous_structure_reliable &&
      raw_rgb_change_fraction >= 0.18f &&
      structural_change_fraction >= 0.03f &&
      raw_rgb_change_fraction >= state.appearance_change_baseline + 0.12f &&
      raw_rgb_change_fraction >= std::max(
                                   0.18f,
                                   state.appearance_change_baseline * 3.0f
                                 );
    const bool appearance_proposal =
      broad_appearance_proposal || localized_appearance_proposal;
    const bool broad_rgb_transition =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f;
    const bool brightness_direction_consistent =
      std::max(brightness_rise_fraction, brightness_fall_fraction) >=
      0.80f * raw_rgb_change_fraction;
    const bool exposure_structure_preserved =
      structural_change_fraction < 0.01f ||
      (structural_change_fraction <=
         0.05f * raw_rgb_change_fraction &&
       brightness_direction_consistent);
    const bool exposure_like_transition =
      broad_rgb_transition &&
      current_structure_reliable &&
      previous_structure_reliable &&
      common_structure_representative &&
      exposure_structure_preserved;
    const bool structureless_candidate =
      model_input_history_valid &&
      previous_structure_reliable &&
      !current_structure_reliable;
    const bool structureless_transition =
      structureless_candidate &&
      !model_input_history_gap;
    const bool persistent_structureless_transition =
      structureless_candidate &&
      model_input_history_gap;
    const bool same_scene_gap_return =
      model_input_history_gap &&
      current_structure_reliable &&
      raw_rgb_change_fraction < 0.01f &&
      structural_change_fraction < 0.01f;
    const bool base_appearance_veto =
      exposure_like_transition ||
      structureless_transition ||
      same_scene_gap_return;
    const bool photometric_recovery_veto =
      exposure_like_transition || same_scene_gap_return;
    const bool geometry_armed =
      (state.cut_flags & cut_flag_geometry_armed) != 0u;
    const bool appearance_armed =
      (state.cut_flags & cut_flag_appearance_armed) != 0u;
    const bool cut_latched = (state.cut_flags & cut_flag_latched) != 0u;
    const bool appearance_recovery =
      (state.cut_flags & cut_flag_appearance_recovery) != 0u;
    const bool geometry_confirmation_pending =
      (state.cut_flags & cut_flag_geometry_confirmation_pending) != 0u;
    const bool appearance_recovery_tail =
      appearance_recovery && !appearance_proposal;
    const bool appearance_veto =
      base_appearance_veto || appearance_recovery_tail;
    state.depth_change_baseline =
      std::clamp(state.depth_change_baseline, 0.0f, 1.0f);
    const bool relative_geometry_spike =
      cut_latched && !geometry_armed &&
      !appearance_veto &&
      state.scene_age >= 8.0f &&
      depth_change_fraction >= 0.30f &&
      (depth_change_fraction >= state.depth_change_baseline + 0.20f ||
       depth_change_fraction >= state.depth_change_baseline * 2.0f);
    const bool immediate_appearance_cut =
      appearance_armed &&
      appearance_proposal &&
      !appearance_veto &&
      depth_change_fraction >= 0.25f;
    // Mirrors the shader's structureless-reference waiver: a uniform slate reference produces
    // exactly zero structural change, so the bridge-return arm must not require it.
    const bool reference_structureless =
      model_input_history_valid && !previous_structure_reliable;
    const bool geometry_structure_corroborated =
      persistent_structureless_transition ||
      reference_structureless ||
      structural_change_fraction >= 0.005f;
    const bool geometry_confirmation_candidate =
      !appearance_veto &&
      geometry_structure_corroborated &&
      ((geometry_armed && depth_change_fraction >= 0.60f) ||
       (low_structure_scene && current_structure_reliable &&
        depth_change_fraction >= 0.60f) ||
       (geometry_confirmation_pending && current_structure_reliable &&
        depth_change_fraction >= 0.60f) ||
       relative_geometry_spike);
    const bool structureless_candidate_already_confirmed =
      persistent_structureless_transition &&
      geometry_confirmation_candidate;
    const bool confirmed_geometry_cut =
      geometry_confirmation_pending &&
      geometry_confirmation_candidate;
    const bool shot_cut =
      state.initialized &&
      (immediate_appearance_cut ||
       structureless_candidate_already_confirmed ||
       confirmed_geometry_cut);
    const bool start_geometry_confirmation =
      state.initialized &&
      !shot_cut &&
      !geometry_confirmation_pending &&
      geometry_confirmation_candidate &&
      !persistent_structureless_transition;

    if (!state.initialized) {
      state.scene_age = 0.0f;
      state.cut_flags = 0u;
      state.depth_change_baseline = depth_change_fraction;
      state.appearance_change_baseline = 0.0f;
    } else if (shot_cut) {
      state.scene_age = 0.0f;
      state.cut_flags = cut_flag_latched;
      state.depth_change_baseline = depth_change_fraction;
      state.appearance_change_baseline = 0.0f;
    } else {
      if (
        !structureless_transition &&
        !appearance_recovery_tail &&
        !start_geometry_confirmation
      ) {
        state.depth_change_baseline +=
          (depth_change_fraction - state.depth_change_baseline) * 0.125f;
      }
      if (model_input_history_valid && !start_geometry_confirmation) {
        state.appearance_change_baseline +=
          (raw_rgb_change_fraction - state.appearance_change_baseline) * 0.25f;
      }
      if (!cut_latched) {
        if (state.scene_age >= 8.0f) {
          state.cut_flags = cut_flags_ready;
        }
      } else {
        if (!geometry_armed) {
          if (depth_change_fraction < 0.10f) {
            if ((state.cut_flags & cut_flag_geometry_low_once) != 0u) {
              state.cut_flags |= cut_flag_geometry_armed;
              state.cut_flags &= ~cut_flag_geometry_low_once;
            } else {
              state.cut_flags |= cut_flag_geometry_low_once;
            }
          } else {
            state.cut_flags &= ~cut_flag_geometry_low_once;
          }
        }
        if (!appearance_armed) {
          if (!appearance_proposal) {
            if ((state.cut_flags & cut_flag_appearance_quiet_once) != 0u) {
              state.cut_flags |= cut_flag_appearance_armed;
              state.cut_flags &= ~cut_flag_appearance_quiet_once;
            } else {
              state.cut_flags |= cut_flag_appearance_quiet_once;
            }
          } else {
            state.cut_flags &= ~cut_flag_appearance_quiet_once;
          }
        }
      }
      if (photometric_recovery_veto) {
        state.cut_flags |= cut_flag_appearance_recovery;
      } else {
        state.cut_flags &= ~cut_flag_appearance_recovery;
      }
      state.cut_flags &= ~cut_flag_geometry_confirmation_pending;
      if (start_geometry_confirmation) {
        state.cut_flags |= cut_flag_geometry_confirmation_pending;
      }
    }
    state.model_input_history_state =
      start_geometry_confirmation ? 4.0f :
                                    (structureless_transition ? 2.0f :
                                     (persistent_structureless_transition ||
                                      (low_structure_scene && !current_structure_reliable)) ?
                                                                3.0f :
                                                                1.0f);
    state.initialized = true;
    return shot_cut;
  }

}  // namespace

#ifdef _WIN32
TEST(DirectxShaderTest, CompilesAllColorShaderVariants) {
  // D3DCompileFromFile does not require a D3D device. This covers BGRA8, FP16 SDR, PQ,
  // planar luma, both chroma sitings, and the HDR cursor shader in one focused check.
  EXPECT_EQ(platf::dxgi::init(), 0);
}

TEST(DirectxShaderTest, BgraYuvConvertersCompileWithoutDiagnostics) {
  using Microsoft::WRL::ComPtr;

  // These are the three entry points that include convert_base.hlsl. FXC previously emitted an
  // X4000 "potentially uninitialized CONVERT_FUNCTION" warning for each one at host startup.
  constexpr std::array filenames {
    "convert_yuv420_packed_uv_type0_ps.hlsl",
    "convert_yuv420_packed_uv_type0s_ps.hlsl",
    "convert_yuv420_planar_y_ps.hlsl",
  };
  for (const auto *filename : filenames) {
    const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main_ps",
      "ps_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &shader,
      &diagnostics
    );
    const std::string_view diagnostic_text = diagnostics ?
                                               std::string_view {
                                                 static_cast<const char *>(
                                                   diagnostics->GetBufferPointer()
                                                 ),
                                                 diagnostics->GetBufferSize()
                                               } :
                                               std::string_view {};
    ASSERT_TRUE(SUCCEEDED(status)) << filename << ": " << diagnostic_text;
    EXPECT_TRUE(diagnostic_text.empty()) << filename << ": " << diagnostic_text;
  }
}

TEST(DirectxShaderTest, ProductionV2ShadersArePermanentPrewarmSet) {
  namespace cache = models::host_sbs_shader_cache;
  const std::filesystem::path shader_root = SUNSHINE_SHADERS_DIR;
  const auto assets_dir = shader_root.parent_path().parent_path();
  ASSERT_TRUE(cache::prewarm(assets_dir));
  const auto producer_sources = cache::snapshot_sources(
    shader_root,
    cache::parallax_v2_producer_specs
  );
  ASSERT_TRUE(producer_sources);

  // The preprocess-only main+content+padding closure remains available only to compute the
  // calibrated model-input identity. Each production entry point is owned exactly once by the
  // full producer closure; this test deliberately never requests bytecode from the smaller
  // snapshot.
  const auto preprocess_sources = cache::snapshot_sources(
    shader_root,
    cache::preprocess_specs
  );
  ASSERT_TRUE(preprocess_sources);
  EXPECT_FALSE(cache::source_closure_sha256(preprocess_sources).empty());
  for (const auto &preprocess : cache::preprocess_specs) {
    const auto producer_matches = std::count_if(
      cache::parallax_v2_producer_specs.begin(),
      cache::parallax_v2_producer_specs.end(),
      [&preprocess](const cache::shader_spec &producer) {
        return producer.filename == preprocess.filename &&
               producer.entrypoint == preprocess.entrypoint &&
               producer.target == preprocess.target;
      }
    );
    EXPECT_EQ(producer_matches, 1) << preprocess.filename;
  }

  for (const auto &producer : cache::parallax_v2_producer_specs) {
    const auto first = cache::get(producer_sources, producer);
    const auto second = cache::get(producer_sources, producer);
    ASSERT_TRUE(first) << producer.filename;
    ASSERT_TRUE(second) << producer.filename;
    EXPECT_EQ(first.get(), second.get()) << producer.filename;
    const auto exact_matches = std::count_if(
      cache::parallax_v2_producer_specs.begin(),
      cache::parallax_v2_producer_specs.end(),
      [&producer](const cache::shader_spec &candidate) {
        return candidate.filename == producer.filename &&
               candidate.entrypoint == producer.entrypoint &&
               candidate.target == producer.target;
      }
    );
    EXPECT_EQ(exact_matches, 1) << producer.filename;
    EXPECT_EQ(producer.filename.find("roi"), std::string_view::npos);
  }

  const auto has_producer_shader = [](const cache::shader_spec &wanted) {
    return std::ranges::any_of(
      cache::parallax_v2_producer_specs,
      [&](const cache::shader_spec &candidate) {
        return candidate.filename == wanted.filename &&
               candidate.entrypoint == wanted.entrypoint &&
               candidate.target == wanted.target;
      }
    );
  };
  EXPECT_TRUE(has_producer_shader(cache::depth_scene_cut_evidence));
  EXPECT_TRUE(has_producer_shader(cache::depth_scene_cut_resolve));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_preprocess));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_cells));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_resolve));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_subtitle_locator_resolve));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_subtitle_condition_prepare));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_subtitle_condition_in_place));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_subtitle_condition));

  const auto diagnostic_sources = cache::snapshot_sources(
    shader_root,
    cache::parallax_v2_diagnostic_specs
  );
  ASSERT_TRUE(diagnostic_sources);
  const auto coordinate_diagnostic = cache::get(
    diagnostic_sources,
    cache::depth_coordinate_v2_coordinate_diagnostic
  );
  ASSERT_TRUE(coordinate_diagnostic);

  EXPECT_FALSE(has_producer_shader(cache::host_sbs_current_frame_motion_probe));
  const auto motion_probe_sources = cache::snapshot_sources(
    shader_root,
    cache::adaptive_motion_probe_specs
  );
  ASSERT_TRUE(motion_probe_sources);
  ASSERT_TRUE(cache::get(
    motion_probe_sources,
    cache::host_sbs_current_frame_motion_probe
  ));

  const auto live_sources = cache::snapshot_sources(
    shader_root,
    cache::parallax_v2_live_renderer_specs
  );
  ASSERT_TRUE(live_sources);
  for (const auto &live : cache::parallax_v2_live_renderer_specs) {
    const auto first = cache::get(live_sources, live);
    const auto second = cache::get(live_sources, live);
    ASSERT_TRUE(first) << live.filename << ':' << live.entrypoint;
    ASSERT_TRUE(second) << live.filename << ':' << live.entrypoint;
    EXPECT_EQ(first.get(), second.get()) << live.filename << ':' << live.entrypoint;
  }
  const auto flat_sources = cache::snapshot_sources(
    shader_root,
    cache::sbs_flat_fallback_specs
  );
  ASSERT_TRUE(flat_sources);
  for (const auto &flat : cache::sbs_flat_fallback_specs) {
    const auto first = cache::get(flat_sources, flat);
    const auto second = cache::get(flat_sources, flat);
    ASSERT_TRUE(first) << flat.filename << ':' << flat.entrypoint;
    EXPECT_EQ(first.get(), second.get()) << flat.filename << ':' << flat.entrypoint;
  }
  EXPECT_EQ(cache::parallax_v2_live_renderer_specs.size(), 2u);
  EXPECT_EQ(cache::sbs_flat_fallback_specs.size(), 2u);
  EXPECT_EQ(cache::parallax_v2_live_diagnostic_specs.size(), 2u);
}

TEST(DirectxShaderTest, PersistentHostSbsCacheSurvivesProcessEquivalentRoots) {
  namespace cache = models::host_sbs_shader_cache;
  const auto unique = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()
  );
  const auto temporary_root =
    std::filesystem::temp_directory_path() /
    ("apollo-host-sbs-shader-cache-test-" + unique);
  const auto cache_directory = temporary_root / "cache";

  struct cleanup_t {
    std::filesystem::path root;

    ~cleanup_t() {
      cache::configure_persistent_cache({});
      std::error_code ignored;
      std::filesystem::remove_all(root, ignored);
    }
  } cleanup {temporary_root};

  constexpr cache::shader_spec spec {
    "persistent_cache_test_cs.hlsl",
    "main",
    "cs_5_0"
  };
  constexpr std::array specs {spec};
  constexpr std::string_view source =
    "RWStructuredBuffer<uint> Output : register(u0);\n"
    "[numthreads(1, 1, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) { Output[id.x] = 7u; }\n";
  const auto make_snapshot = [&](const std::string &name) {
    const auto root = temporary_root / name;
    std::error_code error;
    std::filesystem::create_directories(root, error);
    EXPECT_FALSE(error);
    std::ofstream output(root / spec.filename, std::ios::binary | std::ios::trunc);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
    output.close();
    EXPECT_TRUE(output.good());
    return cache::snapshot_sources(root, specs);
  };

  cache::configure_persistent_cache(cache_directory);
  const auto initial = cache::cache_statistics();
  const auto first_sources = make_snapshot("root-a");
  ASSERT_TRUE(first_sources);
  const auto first = cache::get(first_sources, spec);
  ASSERT_TRUE(first);
  const auto after_first = cache::cache_statistics();
  EXPECT_EQ(after_first.compiled - initial.compiled, 1u);
  EXPECT_EQ(after_first.persistent_writes - initial.persistent_writes, 1u);

  // A different install root with the same authenticated closure models a fresh process: the
  // process-local key misses because it includes the root, while the persistent key intentionally
  // does not. No FXC invocation is needed.
  const auto second_sources = make_snapshot("root-b");
  ASSERT_TRUE(second_sources);
  const auto second = cache::get(second_sources, spec);
  ASSERT_TRUE(second);
  EXPECT_EQ(*first, *second);
  const auto after_second = cache::cache_statistics();
  EXPECT_EQ(after_second.persistent_hits - after_first.persistent_hits, 1u);
  EXPECT_EQ(after_second.compiled, after_first.compiled);

  std::vector<std::filesystem::path> artifacts;
  for (const auto &entry : std::filesystem::directory_iterator(cache_directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".dxbc") {
      artifacts.push_back(entry.path());
    }
  }
  ASSERT_EQ(artifacts.size(), 1u);
  {
    std::ofstream corrupt(artifacts.front(), std::ios::binary | std::ios::trunc);
    corrupt << "corrupt";
  }
  const auto third_sources = make_snapshot("root-c");
  ASSERT_TRUE(third_sources);
  const auto third = cache::get(third_sources, spec);
  ASSERT_TRUE(third);
  EXPECT_EQ(*first, *third);
  const auto after_third = cache::cache_statistics();
  EXPECT_EQ(
    after_third.rejected_artifacts - after_second.rejected_artifacts,
    1u
  );
  EXPECT_EQ(after_third.compiled - after_second.compiled, 1u);
  EXPECT_EQ(
    after_third.persistent_writes - after_second.persistent_writes,
    1u
  );

  const auto fourth_sources = make_snapshot("root-d");
  ASSERT_TRUE(fourth_sources);
  const auto fourth = cache::get(fourth_sources, spec);
  ASSERT_TRUE(fourth);
  EXPECT_EQ(*first, *fourth);
  const auto after_fourth = cache::cache_statistics();
  EXPECT_EQ(after_fourth.persistent_hits - after_third.persistent_hits, 1u);
  EXPECT_EQ(after_fourth.compiled, after_third.compiled);
}

TEST(ParallaxV2ContractTest, ProductionContractCarriesAttributableState) {
  namespace v2 = models::depth_coordinate_v2;
  EXPECT_FALSE(models::input_color_space_is_linear(models::input_color_space::srgb));
  EXPECT_TRUE(models::input_color_space_is_linear(models::input_color_space::linear_sdr));
  EXPECT_TRUE(models::input_color_space_is_linear(models::input_color_space::scrgb_hdr));
  EXPECT_FLOAT_EQ(v2::gain_per_pop, 0.00375f);
  EXPECT_FLOAT_EQ(v2::parallax_gain, 0.00375f);
  EXPECT_FLOAT_EQ(v2::requested_pop_strength(1.2f), 1.2f);
  EXPECT_FLOAT_EQ(v2::requested_gain_for_config(1.2f), 0.0045f);
  EXPECT_FLOAT_EQ(v2::requested_pop_strength(-1.0f), 0.0f);
  EXPECT_FLOAT_EQ(v2::requested_gain_for_config(-1.0f), 0.0f);
  EXPECT_FLOAT_EQ(v2::direct_container_limit, 0.04f);
  EXPECT_FLOAT_EQ(v2::pointwise_container(0.0f), 0.0f);
  EXPECT_NEAR(
    v2::pointwise_container(-0.01f),
    -v2::pointwise_container(0.01f),
    1.0e-8f
  );
  EXPECT_NEAR(v2::pointwise_container(1.0e-5f), 1.0e-5f, 1.0e-10f);
  float previous_contained = -v2::direct_container_limit;
  for (const float requested : std::array {
         -1.0e30f,
         -1.0f,
         -0.04f,
         -0.01f,
         0.0f,
         0.01f,
         0.04f,
         1.0f,
         1.0e30f
       }) {
    const float contained = v2::pointwise_container(requested);
    EXPECT_TRUE(std::isfinite(contained));
    EXPECT_GE(contained, previous_contained);
    EXPECT_LE(std::abs(contained), v2::direct_container_limit);
    previous_contained = contained;
  }
  EXPECT_FLOAT_EQ(v2::max_horizontal_slope, 0.5f);
  EXPECT_FLOAT_EQ(v2::convergence_curve_default, 0.0f);
  EXPECT_TRUE(v2::convergence_curve_is_valid(0.0f));
  EXPECT_FALSE(v2::convergence_curve_is_valid(-0.1f));
  EXPECT_FLOAT_EQ(v2::far_tau, 0.75f);
  EXPECT_FLOAT_EQ(v2::near_log_tau, 0.5f);
  EXPECT_GT(v2::max_horizontal_slope, 0.0f);
  EXPECT_LT(v2::max_horizontal_slope, 1.0f);
  EXPECT_FLOAT_EQ(v2::vertical_majorant_share, 0.75f);
  EXPECT_EQ(v2::contract_schema, 54u);
  EXPECT_EQ(v2::capture_provenance_schema, 3u);
  EXPECT_EQ(v2::shadow_state_dump_schema, 16u);
  EXPECT_EQ(v2::shadow_frame_stats_dump_schema, 2u);
  EXPECT_EQ(v2::capture_provenance_manifest_key, "raw_model_provenance");
  EXPECT_EQ(
    v2::capture_provenance_binding,
    "raw-depth-model-input-and-preprocess-source-produced-by-calibrated-identity-v3"
  );
  ASSERT_EQ(v2::model_calibrations.size(), 1u);
  const auto &small_calibration = v2::model_calibrations.front();
  const auto preprocess_sources = models::host_sbs_shader_cache::snapshot_sources(
    SUNSHINE_SHADERS_DIR,
    models::host_sbs_shader_cache::preprocess_specs
  );
  ASSERT_TRUE(preprocess_sources);
  EXPECT_EQ(
    models::host_sbs_shader_cache::source_closure_sha256(preprocess_sources),
    small_calibration.preprocess.source_closure_sha256
  );
  EXPECT_EQ(models::host_sbs_shader_cache::shader_compile_flags, 0x00008800u);
  const auto producer_sources = models::host_sbs_shader_cache::snapshot_sources(
    SUNSHINE_SHADERS_DIR,
    models::host_sbs_shader_cache::parallax_v2_producer_specs
  );
  ASSERT_TRUE(producer_sources);
  EXPECT_EQ(
    models::host_sbs_shader_cache::source_closure_sha256(producer_sources),
    v2::shader_source_closure_sha256
  );
  EXPECT_EQ(
    v2::shader_source_closure_schema,
    models::host_sbs_shader_cache::source_closure_schema
  );
  EXPECT_EQ(
    v2::shader_source_compile_flags,
    models::host_sbs_shader_cache::shader_compile_flags
  );
  EXPECT_EQ(v2::shader_source_macro_count, 0u);
  const auto live_renderer_sources =
    models::host_sbs_shader_cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR,
      models::host_sbs_shader_cache::parallax_v2_live_renderer_specs
    );
  ASSERT_TRUE(live_renderer_sources);
  EXPECT_EQ(
    models::host_sbs_shader_cache::source_closure_sha256(live_renderer_sources),
    models::host_sbs_shader_cache::
      parallax_v2_live_renderer_source_closure_sha256
  );
  const auto live_diagnostic_sources =
    models::host_sbs_shader_cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR,
      models::host_sbs_shader_cache::parallax_v2_live_diagnostic_specs
    );
  ASSERT_TRUE(live_diagnostic_sources);
  EXPECT_EQ(
    models::host_sbs_shader_cache::source_closure_sha256(live_diagnostic_sources),
    models::host_sbs_shader_cache::parallax_v2_diagnostic_source_closure_sha256
  );
  const auto flat_fallback_sources =
    models::host_sbs_shader_cache::snapshot_sources(
      SUNSHINE_SHADERS_DIR,
      models::host_sbs_shader_cache::sbs_flat_fallback_specs
    );
  ASSERT_TRUE(flat_fallback_sources);
  EXPECT_EQ(
    models::host_sbs_shader_cache::source_closure_sha256(flat_fallback_sources),
    models::host_sbs_shader_cache::sbs_flat_fallback_source_closure_sha256
  );
  ASSERT_EQ(
    v2::shader_source_specs.size(),
    models::host_sbs_shader_cache::parallax_v2_producer_specs.size()
  );
  for (std::size_t index = 0; index < v2::shader_source_specs.size(); ++index) {
    const auto &contract_spec = v2::shader_source_specs[index];
    const auto &runtime_spec =
      models::host_sbs_shader_cache::parallax_v2_producer_specs[index];
    EXPECT_EQ(contract_spec.source_file, runtime_spec.filename);
    EXPECT_EQ(contract_spec.source_entrypoint, runtime_spec.entrypoint);
    EXPECT_EQ(contract_spec.source_target, runtime_spec.target);
  }
  EXPECT_EQ(
    v2::find_model_calibration(
      small_calibration.depth_model,
      small_calibration.depth_model_url,
      small_calibration.onnx_sha256
    ),
    &small_calibration
  );
  EXPECT_EQ(
    v2::find_model_calibration(
      small_calibration.depth_model,
      small_calibration.depth_model_url,
      std::string(64u, '0')
    ),
    nullptr
  );
  EXPECT_EQ(
    v2::find_capture_calibration(
      small_calibration.depth_model,
      small_calibration.depth_model_url,
      small_calibration.onnx_sha256,
      small_calibration.preprocess.profile,
      small_calibration.preprocess.source_closure_sha256,
      770u,
      434u
    ),
    &small_calibration
  );
  EXPECT_EQ(
    v2::find_capture_calibration(
      small_calibration.depth_model,
      small_calibration.depth_model_url,
      small_calibration.onnx_sha256,
      "different-preprocess",
      small_calibration.preprocess.source_closure_sha256,
      770u,
      434u
    ),
    nullptr
  );
  const std::array duplicate_calibrations {
    small_calibration,
    small_calibration,
  };
  EXPECT_EQ(
    v2::find_model_calibration_in(
      duplicate_calibrations,
      small_calibration.depth_model,
      small_calibration.depth_model_url,
      small_calibration.onnx_sha256
    ),
    nullptr
  );
  EXPECT_EQ(small_calibration.depth_model, "depth_anything_v2_fp16");
  EXPECT_FLOAT_EQ(small_calibration.raw_coordinate_scale, 2.25f);
  EXPECT_EQ(
    small_calibration.preprocess.profile,
    "apollo-dav2-centered-integer-contain-edge-pad-area-hdr-srgb-imagenet-v2"
  );
  EXPECT_EQ(
    small_calibration.preprocess.stage,
    "exact model input after centered integer contain-fit area resize, HDR tone mapping, sRGB "
    "conversion, ImageNet normalization, and edge-replicated tensor padding excluded from the "
    "analysis domain"
  );
  EXPECT_EQ(
    small_calibration.preprocess.source_closure_schema,
    models::host_sbs_shader_cache::source_closure_schema
  );
  EXPECT_EQ(
    small_calibration.preprocess.source_file,
    models::host_sbs_shader_cache::rgb_to_nchw.filename
  );
  EXPECT_EQ(
    small_calibration.preprocess.source_entrypoint,
    models::host_sbs_shader_cache::rgb_to_nchw.entrypoint
  );
  EXPECT_EQ(
    small_calibration.preprocess.source_target,
    models::host_sbs_shader_cache::rgb_to_nchw.target
  );
  EXPECT_EQ(
    small_calibration.preprocess.source_compile_flags,
    models::host_sbs_shader_cache::shader_compile_flags
  );
  EXPECT_EQ(small_calibration.preprocess.source_macro_count, 0u);
  const std::array supported_shapes {
    std::pair {770u, 434u},
    std::pair {1022u, 434u},
    std::pair {1036u, 434u},
    std::pair {434u, 770u},
    std::pair {434u, 1022u},
    std::pair {434u, 1036u},
  };
  for (const auto &[width, height] : supported_shapes) {
    EXPECT_TRUE(v2::model_calibration_supports_shape(small_calibration, width, height)) << width << 'x' << height;
    EXPECT_TRUE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, small_calibration.onnx_sha256, small_calibration.preprocess.profile, small_calibration.preprocess.source_closure_sha256, width, height)) << width << 'x' << height;
  }
  EXPECT_EQ(v2::constant_float_count, 8u);
  EXPECT_EQ(v2::constant_vector_count, 2u);
  EXPECT_EQ(sizeof(v2::constants_t), 32u);
  EXPECT_EQ(alignof(v2::constants_t), 16u);
  EXPECT_EQ(v2::frame_stats_float_count, 8u);
  EXPECT_EQ(v2::frame_stats_vector_count, 2u);
  EXPECT_EQ(v2::state_float_count, 12u);
  EXPECT_EQ(v2::state_vector_count, 3u);
  EXPECT_EQ(v2::state_float_count, v2::state_field_names.size());
  EXPECT_EQ(v2::constant_float_count, v2::constant_field_names.size());
  EXPECT_EQ(v2::frame_stats_float_count, v2::frame_stat_names.size());
  EXPECT_STREQ(v2::constant_field_names[0], "raw_coordinate_scale");
  EXPECT_STREQ(v2::constant_field_names[4], "requested_gain");
  EXPECT_STREQ(v2::constant_field_names[7], "convergence_curve_default");
  EXPECT_STREQ(v2::frame_stat_names[v2::frame_stat_mean], "mean");
  EXPECT_STREQ(v2::frame_stat_names[v2::frame_stat_valid], "valid");
  EXPECT_STREQ(v2::state_field_names[v2::calibration_revision], "calibration_revision");
  EXPECT_STREQ(v2::state_field_names[v2::convergence_curve], "convergence_curve");
  EXPECT_STREQ(v2::state_field_names[v2::container_scale], "container_scale");
  EXPECT_STREQ(v2::state_field_names[v2::camera_center_integrity_bits], "camera_center_integrity_bits");
  EXPECT_TRUE(v2::state_word_is_uint_bits(v2::state_word_e::calibration_revision));
  EXPECT_TRUE(v2::state_word_is_uint_bits(v2::state_word_e::confirmed_cut_count));
  EXPECT_TRUE(v2::state_word_is_uint_bits(v2::state_word_e::contract_tag_bits));
  EXPECT_TRUE(v2::state_word_is_uint_bits(v2::state_word_e::camera_center_integrity_bits));
  EXPECT_TRUE(v2::state_word_is_uint_bits(v2::state_word_e::renderer_authorization_bits));
  EXPECT_FALSE(v2::state_word_is_uint_bits(v2::state_word_e::center));
  EXPECT_EQ(v2::state_initial_words[v2::calibration_revision], 0u);
  EXPECT_EQ(v2::state_initial_words[v2::confirmed_cut_count], 0u);
  EXPECT_EQ(v2::state_initial_words[v2::frame_valid], std::bit_cast<std::uint32_t>(0.0f));
  EXPECT_EQ(v2::state_initial_words[v2::convergence_curve], std::bit_cast<std::uint32_t>(0.0f));
  EXPECT_EQ(v2::state_initial_words[v2::container_scale], std::bit_cast<std::uint32_t>(1.0f));
  EXPECT_EQ(v2::state_initial_words[v2::contract_tag_bits], v2::contract_tag);
  EXPECT_EQ(v2::state_initial_words[v2::camera_center_integrity_bits], 0u);
  EXPECT_EQ(v2::state_initial_words[v2::renderer_authorization_bits], 0u);
  EXPECT_EQ(v2::state_initial_words[v2::mapping_state_reserved_1], 0u);
  EXPECT_EQ(v2::state_initial_words[v2::mapping_state_reserved_2], 0u);
  EXPECT_TRUE(v2::camera_center_integrity_is_valid(0u, 0u, 0u, 0u, 0u));
  EXPECT_FALSE(v2::camera_center_integrity_is_valid(0u, 0u, std::bit_cast<std::uint32_t>(-0.1f), 0u, 0u));
  EXPECT_FALSE(v2::camera_center_integrity_is_valid(std::bit_cast<std::uint32_t>(1.0f), 0u, 0u, 0u, 0u));
  EXPECT_FALSE(v2::cut_generation_changed(7u, 7u, false));
  EXPECT_TRUE(v2::cut_generation_changed(7u, 8u, false));
  EXPECT_TRUE(v2::cut_generation_changed(7u, 7u, true));

  EXPECT_TRUE(v2::model_calibration_supports_shape(small_calibration, 1036u, 434u));
  EXPECT_TRUE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, small_calibration.onnx_sha256, small_calibration.preprocess.profile, small_calibration.preprocess.source_closure_sha256, 770u, 434u));
  EXPECT_TRUE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, small_calibration.onnx_sha256, small_calibration.preprocess.profile, small_calibration.preprocess.source_closure_sha256, 1036u, 434u));
  EXPECT_FALSE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, small_calibration.onnx_sha256, "different-preprocess", small_calibration.preprocess.source_closure_sha256, 770u, 434u));
  EXPECT_FALSE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, std::string(64u, '0'), small_calibration.preprocess.profile, small_calibration.preprocess.source_closure_sha256, 770u, 434u));
  EXPECT_FALSE(v2::capture_identity_is_calibrated(small_calibration.depth_model, small_calibration.depth_model_url, small_calibration.onnx_sha256, small_calibration.preprocess.profile, std::string(64u, '0'), 770u, 434u));
}

TEST(ParallaxV2ContractTest, LiveModelResolverPinsAuthenticatedProductionIdentity) {
  namespace v2 = models::depth_coordinate_v2;
  ASSERT_EQ(v2::model_calibrations.size(), 1u);
  const auto &production = v2::model_calibrations.front();

  const auto live = video::host_sbs_v2_depth_model();
  EXPECT_EQ(live.name, production.depth_model);
  EXPECT_EQ(live.url, production.depth_model_url);
}

TEST(ParallaxV2ContractTest, SerializedRendererStateFailsClosedOnAnyBrokenSeal) {
  namespace v2 = models::depth_coordinate_v2;
  const float raw_scale = v2::model_calibrations.front().raw_coordinate_scale;

  auto empty = v2::state_initial_words;
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(empty, raw_scale));

  auto initialized = empty;
  initialized[v2::center] = std::bit_cast<std::uint32_t>(0.25f);
  initialized[v2::inverse_scale] =
    std::bit_cast<std::uint32_t>(1.0f / raw_scale);
  initialized[v2::calibration_revision] = 1u;
  initialized[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
  initialized[v2::renderer_authorization_bits] = v2::contract_tag;
  initialized[v2::camera_center_integrity_bits] =
    v2::camera_center_integrity_for_words(
      initialized[v2::center],
      initialized[v2::inverse_scale],
      initialized[v2::convergence_curve],
      initialized[v2::calibration_revision]
    );
  ASSERT_TRUE(v2::parallax_state_words_are_authenticated(initialized, raw_scale));

  for (const auto index : {
         v2::contract_tag_bits,
         v2::camera_center_integrity_bits,
         v2::renderer_authorization_bits,
         v2::mapping_state_reserved_1,
       }) {
    auto corrupt = initialized;
    corrupt[index] ^= 1u;
    EXPECT_FALSE(v2::parallax_state_words_are_authenticated(corrupt, raw_scale))
      << "state word " << index << " was not authenticated";
  }
  EXPECT_FALSE(v2::parallax_state_words_are_authenticated(initialized, raw_scale * 1.01f));
}

TEST(ParallaxV2ContractTest, RuntimeConstantsAndShaderProvenanceFailClosed) {
  namespace v2 = models::depth_coordinate_v2;
  const float raw_scale = v2::model_calibrations.front().raw_coordinate_scale;
  constexpr float pop = 1.75f;
  const float gain = v2::requested_gain_for_config(pop);

  EXPECT_TRUE(v2::parallax_runtime_constants_are_valid(raw_scale, pop, gain));
  EXPECT_FALSE(v2::parallax_runtime_constants_are_valid(0.0f, pop, gain));
  EXPECT_FALSE(v2::parallax_runtime_constants_are_valid(raw_scale, 0.0f, 0.0f));
  EXPECT_FALSE(v2::parallax_runtime_constants_are_valid(raw_scale, -pop, gain));
  EXPECT_FALSE(v2::parallax_runtime_constants_are_valid(raw_scale, pop, gain * 2.0f));
  EXPECT_FALSE(v2::parallax_runtime_constants_are_valid(
    raw_scale,
    std::numeric_limits<float>::quiet_NaN(),
    gain
  ));

  models::parallax_v2_shader_provenance_t identity {
    .source_closure_schema = v2::shader_source_closure_schema,
    .source_compile_flags = v2::shader_source_compile_flags,
    .source_macro_count = v2::shader_source_macro_count,
    .source_closure_sha256 = std::string {v2::shader_source_closure_sha256},
  };
  EXPECT_TRUE(models::parallax_v2_shader_provenance_matches_current_contract(identity));
  ++identity.source_closure_schema;
  EXPECT_FALSE(models::parallax_v2_shader_provenance_matches_current_contract(identity));
  identity.source_closure_schema = v2::shader_source_closure_schema;
  ++identity.source_compile_flags;
  EXPECT_FALSE(models::parallax_v2_shader_provenance_matches_current_contract(identity));
  identity.source_compile_flags = v2::shader_source_compile_flags;
  ++identity.source_macro_count;
  EXPECT_FALSE(models::parallax_v2_shader_provenance_matches_current_contract(identity));
  identity.source_macro_count = v2::shader_source_macro_count;
  identity.source_closure_sha256.assign(64u, '0');
  EXPECT_FALSE(models::parallax_v2_shader_provenance_matches_current_contract(identity));
}

TEST(ParallaxV2RendererTest, AuthenticationLatchesV2OrLiveFlat) {
  using models::host_sbs_renderer_e;
  using models::latch_host_sbs_renderer;

  EXPECT_EQ(
    latch_host_sbs_renderer(host_sbs_renderer_e::awaiting_v2, true),
    host_sbs_renderer_e::parallax_v2
  );
  EXPECT_EQ(
    latch_host_sbs_renderer(host_sbs_renderer_e::awaiting_v2, false),
    host_sbs_renderer_e::failed_flat
  );
  // Once chosen, no later frame can change geometry authority.
  EXPECT_EQ(
    latch_host_sbs_renderer(host_sbs_renderer_e::parallax_v2, false),
    host_sbs_renderer_e::parallax_v2
  );
  EXPECT_EQ(
    latch_host_sbs_renderer(host_sbs_renderer_e::failed_flat, true),
    host_sbs_renderer_e::failed_flat
  );

  EXPECT_TRUE(models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer_e::awaiting_v2));
  EXPECT_TRUE(models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer_e::parallax_v2));
  EXPECT_FALSE(models::host_sbs_renderer_uses_depth_pipeline(host_sbs_renderer_e::failed_flat));
  EXPECT_TRUE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::parallax_v2, false, true));
  EXPECT_FALSE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::failed_flat, false, true));
  EXPECT_FALSE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::parallax_v2, true, true));
  EXPECT_TRUE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::parallax_v2, false, true, models::host_sbs_v2_max_matched_repeat_age));
  EXPECT_FALSE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::parallax_v2, false, true, models::host_sbs_v2_max_matched_repeat_age + std::chrono::milliseconds(1)));
  EXPECT_TRUE(models::host_sbs_matched_completion_is_current(true, models::host_sbs_v2_max_matched_repeat_age + std::chrono::hours(1)));
  EXPECT_FALSE(models::host_sbs_matched_completion_is_current(false, models::host_sbs_v2_max_matched_repeat_age + std::chrono::milliseconds(1)));
  EXPECT_TRUE(models::host_sbs_should_repeat_matched_output(host_sbs_renderer_e::parallax_v2, false, true, models::host_sbs_v2_max_matched_repeat_age + std::chrono::hours(1), true));
  EXPECT_EQ(
    models::fail_host_sbs_renderer_flat(host_sbs_renderer_e::parallax_v2),
    host_sbs_renderer_e::failed_flat
  );
  EXPECT_EQ(
    models::fail_host_sbs_renderer_flat(host_sbs_renderer_e::awaiting_v2),
    host_sbs_renderer_e::failed_flat
  );
  EXPECT_EQ(
    models::fail_host_sbs_renderer_flat(host_sbs_renderer_e::failed_flat),
    host_sbs_renderer_e::failed_flat
  );
}

TEST(ParallaxV2RendererTest, InteractiveMoveSuppressesOnlyOptionalSubtitleWork) {
  using models::depth_optional_work_mode_e;
  using models::select_depth_optional_work_mode;

  EXPECT_EQ(
    select_depth_optional_work_mode(false, false),
    depth_optional_work_mode_e::ordinary
  );
  EXPECT_EQ(
    select_depth_optional_work_mode(true, false),
    depth_optional_work_mode_e::suppress_subtitle
  );
  // An explicit diagnostic snapshot remains a complete exact-frame observation.
  EXPECT_EQ(
    select_depth_optional_work_mode(true, true),
    depth_optional_work_mode_e::ordinary
  );
  EXPECT_EQ(
    select_depth_optional_work_mode(false, false, true),
    depth_optional_work_mode_e::redispatch_subtitle
  );
  // Complete diagnostics and native move/size suppression outrank the optimization hint.
  EXPECT_EQ(
    select_depth_optional_work_mode(false, true, true),
    depth_optional_work_mode_e::ordinary
  );
  EXPECT_EQ(
    select_depth_optional_work_mode(true, false, true),
    depth_optional_work_mode_e::suppress_subtitle
  );
}

TEST(ParallaxV2RendererTest, SubtitleOcrDamageBandUsesExactAnalysisCropPlacement) {
  const auto full = models::subtitle_ocr_source_crop_rect(
    models::depth_source_rect_t {0u, 0u, 1920u, 1080u}
  );
  ASSERT_TRUE(full);
  EXPECT_EQ(*full, (models::depth_source_rect_t {0u, 760u, 1920u, 1080u}));

  const auto roi = models::subtitle_ocr_source_crop_rect(
    models::depth_source_rect_t {100u, 200u, 1060u, 740u}
  );
  ASSERT_TRUE(roi);
  EXPECT_EQ(*roi, (models::depth_source_rect_t {100u, 580u, 1060u, 740u}));

  const auto ultrawide = models::subtitle_ocr_source_crop_rect(
    models::depth_source_rect_t {50u, 10u, 2610u, 1090u}
  );
  ASSERT_TRUE(ultrawide);
  EXPECT_EQ(*ultrawide, (models::depth_source_rect_t {50u, 663u, 2610u, 1090u}));

  EXPECT_FALSE(models::subtitle_ocr_source_crop_rect({}));
}

TEST(TensorRtContextLifecycleTest, WarmedContextQuarantineStartsOnlyAfterAsyncExecution) {
  using failure_e = models::detail::warmed_execution_context_failure_e;
  using health_t = models::detail::warmed_execution_context_health_t;

  std::size_t quarantined_contexts = 0u;
  for (std::size_t session = 0u; session < 4u; ++session) {
    health_t session_health;
    session_health.observe(failure_e::pre_enqueue_interop_or_binding);
    quarantined_contexts += session_health.poisoned() ? 1u : 0u;
  }
  EXPECT_EQ(quarantined_contexts, 0u);

  health_t execution_health;
  execution_health.observe(failure_e::asynchronous_execution_or_query);
  EXPECT_TRUE(execution_health.poisoned());

  health_t teardown_health;
  teardown_health.observe(failure_e::unsafe_teardown);
  EXPECT_TRUE(teardown_health.poisoned());
}

TEST(TensorRtContextLifecycleTest, ExactFrameJoinWaitsForBothStreamsAndFailsConservatively) {
  using stream_e = models::detail::async_stream_readiness_e;
  using joined_e = models::detail::joined_stream_readiness_e;
  const auto joined = models::detail::joined_stream_readiness;

  EXPECT_EQ(joined(stream_e::ready, false, stream_e::failed), joined_e::ready);
  EXPECT_EQ(joined(stream_e::busy, false, stream_e::ready), joined_e::busy);
  EXPECT_EQ(joined(stream_e::failed, false, stream_e::ready), joined_e::failed);

  EXPECT_EQ(joined(stream_e::ready, true, stream_e::ready), joined_e::ready);
  EXPECT_EQ(joined(stream_e::ready, true, stream_e::busy), joined_e::busy);
  EXPECT_EQ(joined(stream_e::busy, true, stream_e::failed), joined_e::failed);
  EXPECT_EQ(joined(stream_e::failed, true, stream_e::ready), joined_e::failed);
  EXPECT_EQ(joined(stream_e::ready, true, stream_e::failed), joined_e::failed);
}

#ifdef _WIN32
TEST(TensorRtSameFramePollGpuTest, HotProductionObservationPollsWithinBudgetAndConsumesExactFrameOnce) {
  const auto *enabled = std::getenv("APOLLO_RUN_TENSORRT_TESTS");
  if (!enabled || std::string_view {enabled} != "1") {
    GTEST_SKIP() << "Set APOLLO_RUN_TENSORRT_TESTS=1 for the local NVIDIA/TensorRT integration check.";
  }

  using Microsoft::WRL::ComPtr;
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL actual {};
  constexpr D3D_FEATURE_LEVEL requested[] = {
    D3D_FEATURE_LEVEL_11_1,
    D3D_FEATURE_LEVEL_11_0,
  };
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_HARDWARE,
    nullptr,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    requested,
    static_cast<UINT>(std::size(requested)),
    D3D11_SDK_VERSION,
    &device,
    &actual,
    &context
  )));

  constexpr UINT width = 1280u;
  constexpr UINT height = 720u;
  const std::vector<std::uint32_t> pixels(
    static_cast<std::size_t>(width) * height,
    0xff304050u
  );
  D3D11_TEXTURE2D_DESC desc {};
  desc.Width = width;
  desc.Height = height;
  desc.MipLevels = 1u;
  desc.ArraySize = 1u;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1u;
  desc.Usage = D3D11_USAGE_IMMUTABLE;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  const D3D11_SUBRESOURCE_DATA initial {
    pixels.data(), width * sizeof(std::uint32_t), 0u
  };
  ComPtr<ID3D11Texture2D> texture;
  ComPtr<ID3D11ShaderResourceView> source;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, &initial, &texture)));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
    texture.Get(), nullptr, &source
  )));

  models::video_depth_estimator estimator {
    device,
    context,
    std::filesystem::path {SUNSHINE_ASSETS_DIR},
    config::video_t::sbs_t {},
    video::host_sbs_v2_depth_model(),
  };
  ASSERT_TRUE(estimator.is_valid());

  // The first exact signature runs an ordinary enqueue; the second captures, instantiates, and
  // launches its CUDA graph. A few additional replays bring an otherwise idle adapter out of its
  // low-power state before measuring the same two-millisecond budget used by production.
  constexpr std::uint64_t warm_frame_base = 0xabc000u;
  constexpr std::uint64_t warm_observations = 8u;
  for (std::uint64_t i = 1u; i <= warm_observations; ++i) {
    const auto warm_frame_id = warm_frame_base + i;
    const auto warm = estimator.estimate_depth(
      source.Get(),
      models::input_color_space::srgb,
      warm_frame_id
    );
    ASSERT_TRUE(warm.inference_enqueued);
    const auto warmed = estimator.finish_pending_depth_for_evaluation();
    ASSERT_TRUE(warmed.completed_frame_valid);
    ASSERT_EQ(warmed.completed_frame_id, warm_frame_id);
  }

  constexpr std::uint64_t observed_frame_base = 0xabd000u;
  constexpr std::uint64_t measured_observations = 12u;
  std::uint64_t immediate_hits = 0u;
  std::uint64_t bounded_hits = 0u;
  std::uint64_t timeouts = 0u;
  std::uint64_t total_bounded_queries = 0u;
  for (std::uint64_t i = 1u; i <= measured_observations; ++i) {
    // An event-ready finish may expose the output before the independently queued interop-unmap
    // tail is reusable. Production naturally has a frame interval and warp work here; this tight
    // fixture waits only for that full-stream admission proof before submitting the next unit.
    const auto reuse_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds {50};
    bool reusable = false;
    do {
      reusable = estimator.can_accept_frame();
      if (!reusable) {
        std::this_thread::yield();
      }
    } while (!reusable && std::chrono::steady_clock::now() < reuse_deadline);
    ASSERT_TRUE(reusable);

    const auto observed_frame_id = observed_frame_base + i;
    const auto submitted = estimator.estimate_depth(
      source.Get(),
      models::input_color_space::srgb,
      observed_frame_id
    );
    ASSERT_TRUE(submitted.inference_enqueued);

    // An expired budget still performs exactly one nonblocking joined query. If it is busy, the
    // second call exercises the production two-millisecond cap. A timeout is valid under external
    // GPU load and must preserve the exact pending owner for the synchronous test fallback.
    auto completed = estimator.try_finish_pending_depth_until(
      models::input_color_space::srgb,
      std::chrono::steady_clock::now(),
      1u
    );
    ASSERT_EQ(completed.query_count, 1u);
    const bool immediate_ready = completed.ready;
    if (!immediate_ready) {
      ASSERT_TRUE(completed.timed_out);
      completed = estimator.try_finish_pending_depth_until(
        models::input_color_space::srgb,
        std::chrono::steady_clock::now() + std::chrono::milliseconds {2},
        4096u
      );
      total_bounded_queries += completed.query_count;
    }

    if (completed.ready) {
      immediate_hits += immediate_ready ? 1u : 0u;
      bounded_hits += immediate_ready ? 0u : 1u;
      ASSERT_FALSE(completed.timed_out);
      ASSERT_TRUE(completed.result.completed_frame_valid);
      ASSERT_EQ(completed.result.completed_frame_id, observed_frame_id);
    } else {
      ++timeouts;
      ASSERT_TRUE(completed.timed_out);
      const auto fallback = estimator.finish_pending_depth_for_evaluation();
      ASSERT_TRUE(fallback.completed_frame_valid);
      ASSERT_EQ(fallback.completed_frame_id, observed_frame_id);
    }

    const auto consumed_once = estimator.try_finish_pending_depth_nonblocking(
      models::input_color_space::srgb,
      false
    );
    EXPECT_TRUE(consumed_once.ready);
    EXPECT_FALSE(consumed_once.result.completed_frame_valid);
    EXPECT_EQ(consumed_once.query_count, 0u);
  }

  BOOST_LOG(info) << "TensorRT same-frame poll hardware: immediate hits " << immediate_hits
                  << ", bounded hits " << bounded_hits << ", timeouts " << timeouts
                  << ", bounded queries " << total_bounded_queries << '.';
  EXPECT_GT(immediate_hits + bounded_hits, 0u)
    << "No hot observation completed inside the production two-millisecond poll budget.";
}
#endif

TEST(TensorRtContextLifecycleTest, AccountingBoundsQuarantinedAndReusableContextsTogether) {
  models::detail::execution_context_accounting_t accounting;
  EXPECT_TRUE(accounting.reserve(2u));
  accounting.mark_warmed();
  EXPECT_TRUE(accounting.reserve(2u));
  EXPECT_FALSE(accounting.reserve(2u));
  EXPECT_EQ(accounting.allocated(), 2u);
  EXPECT_EQ(accounting.warmed(), 1u);

  accounting.quarantine(true);
  EXPECT_EQ(accounting.usable(), 1u);
  EXPECT_EQ(accounting.warmed(), 0u);
  EXPECT_EQ(accounting.quarantined(), 1u);
  EXPECT_FALSE(accounting.reserve(2u));

  accounting.release_reservation();
  EXPECT_TRUE(accounting.reserve(2u));
  accounting.mark_warmed();
  accounting.detach_pooled(1u);
  EXPECT_EQ(accounting.usable(), 0u);
  EXPECT_EQ(accounting.warmed(), 0u);
  EXPECT_EQ(accounting.quarantined(), 2u);
}

TEST(TensorRtCudaGraphPolicyTest, SignatureChangesWarmBeforeCaptureAndFailuresStayFallback) {
  using action_e = models::detail::cuda_graph_enqueue_action_e;
  using policy_t = models::detail::cuda_graph_replay_policy_t;
  using signature_t = models::detail::cuda_graph_signature_t;

  policy_t policy;
  const signature_t depth {1u, 2u, 770, 434};
  const signature_t resized {3u, 4u, 518, 294};

  EXPECT_TRUE(models::detail::select_cuda_graph_signature(policy, depth));
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, true, false),
    action_e::ordinary
  );
  EXPECT_TRUE(policy.signature_warmed);
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, true, false),
    action_e::capture
  );
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, true, true),
    action_e::replay
  );

  EXPECT_TRUE(models::detail::select_cuda_graph_signature(policy, resized));
  EXPECT_FALSE(policy.signature_warmed);
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, true, false),
    action_e::ordinary
  );
  EXPECT_FALSE(models::detail::select_cuda_graph_signature(policy, resized));

  policy.capture_failed = true;
  policy.signature_warmed = false;
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, true, false),
    action_e::ordinary
  );
  EXPECT_FALSE(policy.signature_warmed);

  policy.capture_failed = false;
  EXPECT_EQ(
    models::detail::next_cuda_graph_enqueue_action(policy, false, false),
    action_e::ordinary
  );
  EXPECT_FALSE(policy.signature_warmed);
}

TEST(DepthInputRegionTest, RequiresCanonicalFullSourceOrAuthenticatedRoiAuthority) {
  const models::depth_input_region_t full_source {
    .source_width = 1920u,
    .source_height = 1080u,
    .left = 0u,
    .top = 0u,
    .right = 1920u,
    .bottom = 1080u,
    .tensor_content = {0u, 0u, 770u, 434u},
  };
  EXPECT_TRUE(full_source.valid());

  auto partial_without_video_identity = full_source;
  partial_without_video_identity.left = 100u;
  EXPECT_FALSE(partial_without_video_identity.valid());

  auto full_with_generation = full_source;
  full_with_generation.analysis_generation = 1u;
  EXPECT_FALSE(full_with_generation.valid());

  const models::depth_input_region_t video_region {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 820u,
    .top = 510u,
    .right = 2471u,
    .bottom = 1439u,
    .tensor_content = {0u, 0u, 770u, 433u},
    .analysis_generation = 7u,
    .video_region = true,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  EXPECT_TRUE(video_region.valid());

  auto video_without_identity = video_region;
  video_without_identity.analysis_generation = 0u;
  EXPECT_FALSE(video_without_identity.valid());

  auto roi_without_authority = video_region;
  roi_without_authority.authority = models::depth_analysis_authority_e::full_source;
  EXPECT_FALSE(roi_without_authority.valid());

  auto roi_with_unknown_authority = video_region;
  roi_with_unknown_authority.authority =
    static_cast<models::depth_analysis_authority_e>(0xFFu);
  EXPECT_FALSE(roi_with_unknown_authority.valid());

  auto foreground_region = video_region;
  foreground_region.authority = models::depth_analysis_authority_e::foreground_client;
  EXPECT_TRUE(foreground_region.valid());

  auto full_with_roi_authority = full_source;
  full_with_roi_authority.authority = models::depth_analysis_authority_e::foreground_client;
  EXPECT_FALSE(full_with_roi_authority.valid());

  auto out_of_bounds = video_region;
  out_of_bounds.right = out_of_bounds.source_width + 1u;
  EXPECT_FALSE(out_of_bounds.valid());
}

TEST(DepthInputRegionTest, AnalysisDomainIgnoresOnlyPurePositionMoves) {
  const models::depth_input_region_t original {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 820u,
    .top = 510u,
    .right = 2471u,
    .bottom = 1439u,
    .tensor_content = {0u, 0u, 770u, 433u},
    .analysis_generation = 7u,
    .video_region = true,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  auto moved = original;
  moved.left += 100u;
  moved.right += 100u;
  EXPECT_TRUE(original.same_analysis_domain(moved));
  EXPECT_NE(original, moved);

  auto resized = original;
  resized.right += 1u;
  EXPECT_FALSE(original.same_analysis_domain(resized));

  auto new_content_mapping = original;
  --new_content_mapping.tensor_content.bottom;
  EXPECT_FALSE(original.same_analysis_domain(new_content_mapping));

  auto new_video = original;
  new_video.analysis_generation += 1u;
  EXPECT_FALSE(original.same_analysis_domain(new_video));

  auto new_authority = original;
  new_authority.authority = models::depth_analysis_authority_e::foreground_client;
  EXPECT_FALSE(original.same_analysis_domain(new_authority));

  auto new_source = original;
  new_source.source_width += 1u;
  EXPECT_FALSE(original.same_analysis_domain(new_source));

  auto full_source = original;
  full_source.video_region = false;
  full_source.left = 0u;
  full_source.top = 0u;
  full_source.right = full_source.source_width;
  full_source.bottom = full_source.source_height;
  full_source.analysis_generation = 0u;
  full_source.authority = models::depth_analysis_authority_e::full_source;
  EXPECT_FALSE(original.same_analysis_domain(full_source));
}

TEST(DepthInputRegionTest, GenerationIdentityIncludesAuthorityKind) {
  models::depth_analysis_generation_tracker_t generation_tracker;
  const models::depth_analysis_domain_key_t chromium {
    .source_width = 1920u,
    .source_height = 1080u,
    .semantic_width = 1280u,
    .semantic_height = 720u,
    .crop_width = 1280u,
    .crop_height = 720u,
    .hwnd = 0x1234u,
    .process_id = 52u,
    .document_id = 7,
    .video_id = 11,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  const auto chromium_generation = generation_tracker.select(chromium);
  ASSERT_NE(chromium_generation, 0u);
  EXPECT_EQ(generation_tracker.select(chromium), chromium_generation);

  auto foreground = chromium;
  foreground.authority = models::depth_analysis_authority_e::foreground_client;
  const auto foreground_generation = generation_tracker.select(foreground);
  EXPECT_NE(foreground_generation, chromium_generation);
  EXPECT_EQ(generation_tracker.select(foreground), foreground_generation);
}

TEST(DepthInputRegionTest, DomainTransitionDecisionResetsExactlyOncePerStableChange) {
  models::depth_analysis_generation_tracker_t generation_tracker;
  models::depth_input_domain_tracker_t domain_tracker;
  const models::depth_input_region_t full_source {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 0u,
    .top = 0u,
    .right = 3840u,
    .bottom = 2160u,
    .tensor_content = {0u, 0u, 770u, 434u},
  };
  EXPECT_TRUE(domain_tracker.update(full_source, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(full_source, models::input_color_space::srgb));

  models::depth_analysis_domain_key_t key {
    .source_width = 3840u,
    .source_height = 2160u,
    .semantic_width = 2682u,
    .semantic_height = 1508u,
    .crop_width = 2674u,
    .crop_height = 1504u,
    .hwnd = 0x1234u,
    .process_id = 52u,
    .document_id = 7,
    .video_id = 11,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  const auto first_generation = generation_tracker.select(key);
  ASSERT_NE(first_generation, 0u);
  models::depth_input_region_t roi {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 580u,
    .top = 326u,
    .right = 3254u,
    .bottom = 1830u,
    .tensor_content = {0u, 0u, 770u, 433u},
    .analysis_generation = first_generation,
    .video_region = true,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  ASSERT_TRUE(roi.valid());
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::srgb));

  // Observer generation is matched-frame provenance outside the stable key. An otherwise
  // identical heartbeat therefore preserves both generation and estimator analysis state.
  std::uint64_t observer_generation = 100u;
  ++observer_generation;
  EXPECT_EQ(generation_tracker.select(key), first_generation);
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::srgb));
  EXPECT_EQ(observer_generation, 101u);

  auto moved = roi;
  moved.left += 40u;
  moved.right += 40u;
  ASSERT_TRUE(moved.valid());
  EXPECT_FALSE(domain_tracker.update(moved, models::input_color_space::srgb));

  auto semantic_resize = key;
  ++semantic_resize.semantic_width;
  roi.analysis_generation = generation_tracker.select(semantic_resize);
  EXPECT_NE(roi.analysis_generation, first_generation);
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::srgb));

  auto new_identity = semantic_resize;
  ++new_identity.video_id;
  roi.analysis_generation = generation_tracker.select(new_identity);
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::srgb));

  auto foreground_key = new_identity;
  foreground_key.document_id = 0;
  foreground_key.video_id = 0;
  foreground_key.authority = models::depth_analysis_authority_e::foreground_client;
  const auto foreground_generation = generation_tracker.select(foreground_key);
  EXPECT_NE(foreground_generation, roi.analysis_generation);
  roi.analysis_generation = foreground_generation;
  roi.authority = models::depth_analysis_authority_e::foreground_client;
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::srgb));

  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::linear_sdr));

  generation_tracker.select_full_source();
  EXPECT_TRUE(domain_tracker.update(full_source, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(full_source, models::input_color_space::linear_sdr));
  new_identity.authority = models::depth_analysis_authority_e::chromium_video;
  roi.analysis_generation = generation_tracker.select(new_identity);
  roi.authority = models::depth_analysis_authority_e::chromium_video;
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
}

  #ifdef _WIN32
TEST(ParallaxV2RendererTest, AuthenticationRejectsMissingOrTamperedIdentity) {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  constexpr D3D_FEATURE_LEVEL requested_levels[] {D3D_FEATURE_LEVEL_11_0};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested_levels,
    static_cast<UINT>(std::size(requested_levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  )));

  D3D11_TEXTURE2D_DESC desc {};
  desc.Width = 1;
  desc.Height = 1;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_R32_FLOAT;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> texture;
  ComPtr<ID3D11ShaderResourceView> view;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &texture)));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &view)));

  const auto &calibration = v2::model_calibrations.front();
  models::estimate_result result;
  result.shadow_candidate_parallax = view;
  result.shadow_ownership_refined_parallax = view;
  result.shadow_vertical_majorant = view;
  result.shadow_vertical_conditioned = view;
  result.shadow_base_final_parallax = view;
  result.shadow_final_parallax = view;
  result.shadow_state = view;
  result.shadow_frame_stats = view;
  result.ocr_box_record = view;
  result.subtitle_locator_state = view;
  result.raw_model_provenance =
    std::make_shared<const models::raw_model_provenance_t>(
      models::raw_model_provenance_t {
        .depth_model = std::string {calibration.depth_model},
        .depth_model_url = std::string {calibration.depth_model_url},
        .onnx_sha256 = std::string {calibration.onnx_sha256},
        .preprocess_profile = std::string {calibration.preprocess.profile},
        .preprocess_source_closure_sha256 =
          std::string {calibration.preprocess.source_closure_sha256},
      }
    );
  result.parallax_v2_shader_provenance =
    std::make_shared<const models::parallax_v2_shader_provenance_t>(
      models::parallax_v2_shader_provenance_t {
        .source_closure_schema = v2::shader_source_closure_schema,
        .source_compile_flags = v2::shader_source_compile_flags,
        .source_macro_count = v2::shader_source_macro_count,
        .source_closure_sha256 = std::string {v2::shader_source_closure_sha256},
      }
    );
  result.raw_width = 770;
  result.raw_height = 434;
  result.completed_frame_valid = true;
  result.completed_frame_id = 1;
  result.parallax_v2_producer_active = true;
  result.parallax_v2_raw_coordinate_scale = calibration.raw_coordinate_scale;
  result.parallax_v2_requested_pop_strength = 2.0f;
  result.parallax_v2_requested_gain = v2::requested_gain_for_config(2.0f);
  result.input_region = {
    .source_width = 1920u,
    .source_height = 1080u,
    .left = 0u,
    .top = 0u,
    .right = 1920u,
    .bottom = 1080u,
    .tensor_content = {0u, 0u, 770u, 434u},
  };
  EXPECT_FALSE(result.shadow_coordinate);
  EXPECT_TRUE(models::parallax_v2_result_is_authenticated(result));

  auto subtitle_suppressed = result;
  subtitle_suppressed.subtitle_work_suppressed = true;
  EXPECT_TRUE(models::parallax_v2_result_is_authenticated(subtitle_suppressed));

  auto missing_input_region = result;
  missing_input_region.input_region = {};
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_input_region));

  auto dump_augmented = result;
  dump_augmented.shadow_coordinate = view;
  EXPECT_TRUE(models::parallax_v2_result_is_authenticated(dump_augmented));

  auto missing_resource = result;
  missing_resource.shadow_state.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_resource));

  auto missing_vertical_majorant = result;
  missing_vertical_majorant.shadow_vertical_majorant.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_vertical_majorant));

  auto missing_ownership_refined = result;
  missing_ownership_refined.shadow_ownership_refined_parallax.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_ownership_refined));

  auto missing_vertical_conditioned = result;
  missing_vertical_conditioned.shadow_vertical_conditioned.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_vertical_conditioned));

  auto missing_ocr_record = result;
  missing_ocr_record.ocr_box_record.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_ocr_record));

  auto missing_subtitle_state = result;
  missing_subtitle_state.subtitle_locator_state.Reset();
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_subtitle_state));

  struct supported_shape_t {
    std::uint32_t source_width;
    std::uint32_t source_height;
    int depth_width;
    int depth_height;
  };

  const std::array supported_shapes {
    supported_shape_t {1920u, 1080u, 770, 434},
    supported_shape_t {2560u, 1080u, 1022, 434},
    supported_shape_t {3440u, 1440u, 1036, 434},
    supported_shape_t {1080u, 1920u, 434, 770},
    supported_shape_t {1080u, 2560u, 434, 1022},
    supported_shape_t {1440u, 3440u, 434, 1036},
  };
  for (const auto &[source_width, source_height, width, height] : supported_shapes) {
    auto supported_shape = result;
    supported_shape.raw_width = width;
    supported_shape.raw_height = height;
    supported_shape.input_region = {
      .source_width = source_width,
      .source_height = source_height,
      .left = 0u,
      .top = 0u,
      .right = source_width,
      .bottom = source_height,
      .tensor_content = {
        0u, 0u,
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
      },
    };
    EXPECT_TRUE(models::parallax_v2_result_is_authenticated(supported_shape))
      << width << 'x' << height;
  }

  auto video_region = result;
  video_region.input_region = {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 820u,
    .top = 510u,
    .right = 2471u,
    .bottom = 1439u,
    .tensor_content = {0u, 0u, 770u, 433u},
    .analysis_generation = 7u,
    .video_region = true,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };
  EXPECT_TRUE(models::parallax_v2_result_is_authenticated(video_region));

  auto missing_video_identity = video_region;
  missing_video_identity.input_region.analysis_generation = 0u;
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(missing_video_identity));

  auto analysis_shape_mismatch = video_region;
  analysis_shape_mismatch.input_region.right -= 300u;
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(analysis_shape_mismatch));

  auto partial_full_source = result;
  partial_full_source.input_region.left = 1u;
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(partial_full_source));

  auto wrong_shape = result;
  wrong_shape.raw_width = 1008;
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(wrong_shape));

  auto wrong_shader = result;
  auto shader = *result.parallax_v2_shader_provenance;
  shader.source_closure_sha256 = std::string(64u, '0');
  wrong_shader.parallax_v2_shader_provenance =
    std::make_shared<const models::parallax_v2_shader_provenance_t>(shader);
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(wrong_shader));

  auto wrong_gain = result;
  wrong_gain.parallax_v2_requested_gain *= 2.0f;
  EXPECT_FALSE(models::parallax_v2_result_is_authenticated(wrong_gain));
}
  #endif

TEST(ParallaxV2ContractTest, DebugDumpUsesNonblockingGpuStagingBeforeCpuPublication) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp"
  );
  const auto async_source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump_async.h"
  );
  ASSERT_FALSE(source.empty());
  ASSERT_FALSE(async_source.empty());

  const auto publisher = source.rfind(
    "dump_publish_result publish_captured_dump(const captured_dump_job &job)"
  );
  const auto poll = source.find("void dumper::poll_pending_readback(");
  const auto capture = source.find("bool dumper::maybe_dump(");
  ASSERT_NE(publisher, std::string::npos);
  ASSERT_NE(poll, std::string::npos);
  ASSERT_NE(capture, std::string::npos);
  ASSERT_LT(publisher, poll);
  ASSERT_LT(poll, capture);

  const auto publisher_body = source.substr(publisher, poll - publisher);
  const auto poll_body = source.substr(poll, capture - poll);
  const auto capture_body = source.substr(capture);
  EXPECT_NE(publisher_body.find("write_color_preview("), std::string::npos);
  EXPECT_NE(publisher_body.find("models::file_sha256_hex("), std::string::npos);
  EXPECT_NE(
    publisher_body.find("std::filesystem::create_directories("),
    std::string::npos
  );
  EXPECT_EQ(publisher_body.find("read_texture(device"), std::string::npos);
  EXPECT_EQ(publisher_body.find("ID3D11DeviceContext"), std::string::npos);

  const auto enqueue = poll_body.find("worker_state->enqueue(");
  ASSERT_NE(enqueue, std::string::npos);
  EXPECT_NE(source.find("D3D11_ASYNC_GETDATA_DONOTFLUSH"), std::string::npos);
  EXPECT_NE(source.find("D3D11_MAP_FLAG_DO_NOT_WAIT"), std::string::npos);
  EXPECT_NE(capture_body.find("stage_texture("), std::string::npos);
  EXPECT_NE(capture_body.find("stage_buffer("), std::string::npos);
  EXPECT_NE(capture_body.find("ctx->End(pending->completion.Get())"), std::string::npos);
  EXPECT_EQ(capture_body.find("ctx->Map("), std::string::npos);
  EXPECT_EQ(capture_body.find("GetData("), std::string::npos);
  EXPECT_EQ(poll_body.find("ctx->CopyResource("), std::string::npos);
  EXPECT_EQ(poll_body.find("write_png("), std::string::npos);
  EXPECT_EQ(
    poll_body.find("models::file_sha256_hex("),
    std::string::npos
  );
  EXPECT_EQ(
    poll_body.find("std::filesystem::create_directories("),
    std::string::npos
  );

  EXPECT_NE(async_source.find("std::jthread worker_;"), std::string::npos);
  EXPECT_NE(
    async_source.find("process_publication_queue()"),
    std::string::npos
  );
  EXPECT_NE(source.find("dumper::~dumper()"), std::string::npos);
  EXPECT_NE(source.find("async_.reset();"), std::string::npos);
  EXPECT_NE(
    capture_body.find("job.completed.source = nullptr;"),
    std::string::npos
  );
  EXPECT_NE(
    capture_body.find("detail::button_request_guard button_request"),
    std::string::npos
  );
  EXPECT_NE(
    poll_body.find("worker_state->record_publication_failure("),
    std::string::npos
  );
  EXPECT_NE(source.find("async_->cancel_retries(button_request_);"), std::string::npos);

  const auto header = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.h"
  );
  ASSERT_FALSE(header.empty());
  EXPECT_NE(
    header.find("std::unique_ptr<detail::pending_gpu_capture> pending_gpu_capture_"),
    std::string::npos
  );
  EXPECT_NE(capture_body.find("pending_gpu_capture_ = std::move(pending)"), std::string::npos);
  EXPECT_NE(
    source.find("if (pending_gpu_capture_ || (async_ && async_->busy()))"),
    std::string::npos
  );
}

TEST(ParallaxV2ContractTest, DebugDumpSubtitleResolverProvenanceMatchesSchema32Contract) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp"
  );
  ASSERT_FALSE(source.empty());

  const auto resolver_begin = source.find(
    "nlohmann::json subtitle_locator_resolver_contract_json("
  );
  const auto resolver_end = source.find(
    "bool subtitle_target_is_representable(", resolver_begin
  );
  ASSERT_NE(resolver_begin, std::string::npos);
  ASSERT_NE(resolver_end, std::string::npos);
  ASSERT_LT(resolver_begin, resolver_end);
  const auto resolver = source.substr(resolver_begin, resolver_end - resolver_begin);

  const auto resolve = resolver.find("{\"entrypoint\", \"resolve_main\"}");
  const auto prepare = resolver.find("{\"entrypoint\", \"condition_prepare_main\"}");
  const auto condition = resolver.find("{\"entrypoint\", \"condition_main\"}");
  ASSERT_NE(resolve, std::string::npos);
  ASSERT_NE(prepare, std::string::npos);
  ASSERT_NE(condition, std::string::npos);
  EXPECT_LT(resolve, prepare);
  EXPECT_LT(prepare, condition);

  EXPECT_NE(source.find("{\"schema\", 32}"), std::string::npos);
  for (const auto *placement_key : {
         "{\"placement\", {",
         "{\"fallback_step_denominator\"",
         "{\"fallback_max_radius_steps\"",
         "{\"fallback_requires_unclamped_sample_strip\"",
         "{\"fallback_minimum_coherent_rows\", 2u}",
         "{\"fallback_pair_conflict\", \"unreliable-stop-search\"}",
         "{\"ribbon_places_fallback_with_ordinary\", false}",
       }) {
    EXPECT_NE(resolver.find(placement_key), std::string::npos) << placement_key;
  }
}

TEST(ParallaxV2ContractTest, DebugDumpStagingPreservesEventAndRequestOrdering) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp"
  );
  const auto display = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(source.empty());
  ASSERT_FALSE(display.empty());

  const auto prepare = source.find("bool dumper::prepare_requested_v2_frame(");
  const auto poll = source.find("void dumper::poll_pending_readback(");
  const auto submit = source.find("bool dumper::maybe_dump(");
  ASSERT_NE(prepare, std::string::npos);
  ASSERT_NE(poll, std::string::npos);
  ASSERT_NE(submit, std::string::npos);
  ASSERT_LT(prepare, poll);
  ASSERT_LT(poll, submit);

  const auto prepare_body = source.substr(prepare, poll - prepare);
  const auto poll_body = source.substr(poll, submit - poll);
  const auto submit_body = source.substr(submit);
  EXPECT_EQ(prepare_body.find("Map("), std::string::npos);
  EXPECT_EQ(prepare_body.find("GetData("), std::string::npos);
  EXPECT_EQ(submit_body.find("Map("), std::string::npos);
  EXPECT_EQ(submit_body.find("GetData("), std::string::npos);

  const auto get_data = poll_body.find("ctx->GetData(");
  const auto gpu_ready = poll_body.find("pending.gpu_ready = true");
  const auto collect = poll_body.find("collection_budget budget(poll_started)");
  ASSERT_NE(get_data, std::string::npos);
  ASSERT_NE(gpu_ready, std::string::npos);
  ASSERT_NE(collect, std::string::npos);
  EXPECT_LT(get_data, gpu_ready);
  EXPECT_LT(gpu_ready, collect);
  EXPECT_LT(get_data, collect);
  EXPECT_NE(
    poll_body.find("D3D11_ASYNC_GETDATA_DONOTFLUSH", get_data),
    std::string::npos
  );
  EXPECT_NE(
    poll_body.find("while (pending_gpu_capture_)"),
    std::string::npos
  );
  EXPECT_NE(
    poll_body.find("if (budget.exhausted())"),
    std::string::npos
  );
  const auto first_stage = submit_body.find("stage_texture(");
  const auto event_end = submit_body.find("ctx->End(pending->completion.Get())");
  const auto publish_pending = submit_body.find(
    "pending_gpu_capture_ = std::move(pending)"
  );
  ASSERT_NE(first_stage, std::string::npos);
  ASSERT_NE(event_end, std::string::npos);
  ASSERT_NE(publish_pending, std::string::npos);
  EXPECT_LT(first_stage, event_end);
  EXPECT_LT(event_end, publish_pending);

  const auto readback_poll = display.find("sbs_dumper.poll_pending_readback(");
  const auto request_poll = display.find("sbs_dumper.snapshot_requested()");
  ASSERT_NE(readback_poll, std::string::npos);
  ASSERT_NE(request_poll, std::string::npos);
  EXPECT_LT(readback_poll, request_poll);
  EXPECT_NE(display.find("sbs_dumper.needs_conversion_poll()"), std::string::npos);
  EXPECT_NE(source.find("next_file_trigger_poll_"), std::string::npos);
  EXPECT_NE(source.find("std::filesystem::exists(dir_ / \"dump.trigger\""), std::string::npos);
  EXPECT_EQ(display.find("preflight_requested_v2_frame("), std::string::npos);
}

TEST(ParallaxV2ContractTest, GpuTimerKeepsEveryNotReadyTimestampPending) {
  const auto display = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(display.empty());

  const auto resolve = display.find("void resolve_sbs_gpu_timers()");
  const auto begin = display.find("sbs_gpu_timer_slot_t *begin_sbs_gpu_timer()", resolve);
  ASSERT_NE(resolve, std::string::npos);
  ASSERT_NE(begin, std::string::npos);
  const auto body = display.substr(resolve, begin - resolve);

  EXPECT_NE(body.find("if (ready == S_FALSE)"), std::string::npos);
  EXPECT_NE(body.find("if (ready != S_OK)"), std::string::npos);
  EXPECT_NE(body.find("start_status == S_FALSE"), std::string::npos);
  EXPECT_NE(body.find("convert_status == S_FALSE"), std::string::npos);
  EXPECT_NE(body.find("start_status != S_OK"), std::string::npos);
  EXPECT_NE(body.find("convert_status != S_OK"), std::string::npos);
  EXPECT_EQ(body.find("SUCCEEDED(start_status)"), std::string::npos);
  const auto not_ready = body.find("start_status == S_FALSE");
  const auto retire_failure = body.find("start_status != S_OK");
  const auto retire_slot = body.find("slot.pending = false", retire_failure);
  ASSERT_NE(not_ready, std::string::npos);
  ASSERT_NE(retire_failure, std::string::npos);
  ASSERT_NE(retire_slot, std::string::npos);
  EXPECT_LT(not_ready, retire_failure);
  EXPECT_LT(retire_failure, retire_slot);

  std::size_t nonflushing_queries = 0;
  for (std::size_t offset = 0;
       (offset = body.find("D3D11_ASYNC_GETDATA_DONOTFLUSH", offset)) !=
       std::string::npos;
       offset += 1) {
    ++nonflushing_queries;
  }
  EXPECT_EQ(nonflushing_queries, 6u);
}

TEST(ParallaxV2ContractTest, DebugDumpAcceptsAdvertisedWindowsCaptureColorFormats) {
  const auto dump = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp"
  );
  const auto display = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(dump.empty());
  ASSERT_FALSE(display.empty());

  for (const auto *format : {
         "DXGI_FORMAT_B8G8R8A8_UNORM",
         "DXGI_FORMAT_B8G8R8X8_UNORM",
         "DXGI_FORMAT_R8G8B8A8_UNORM",
         "DXGI_FORMAT_R16G16B16A16_FLOAT",
       }) {
    EXPECT_NE(display.find(format), std::string::npos) << format;
    EXPECT_NE(dump.find(format), std::string::npos) << format;
  }
  EXPECT_NE(
    dump.find("case DXGI_FORMAT_B8G8R8X8_UNORM:\n        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:"),
    std::string::npos
  );
  EXPECT_NE(
    dump.find(
      "case DXGI_FORMAT_B8G8R8X8_UNORM:\n          return \"DXGI_FORMAT_B8G8R8X8_UNORM\";"
    ),
    std::string::npos
  );
  EXPECT_NE(
    dump.find("case DXGI_FORMAT_B8G8R8X8_UNORM:\n            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:"),
    std::string::npos
  );
}

TEST(ParallaxV2ContractTest, CalibrationRevisionRejectsReservedSentinel) {
  namespace v2 = models::depth_coordinate_v2;
  EXPECT_TRUE(v2::calibration_revision_word_is_valid(0u));
  EXPECT_TRUE(v2::calibration_revision_word_is_valid(1u));
  EXPECT_FALSE(v2::calibration_revision_word_is_valid(v2::reserved_calibration_revision));
  EXPECT_FALSE(v2::acquired_calibration_revision_is_valid(0u));
  EXPECT_TRUE(v2::acquired_calibration_revision_is_valid(1u));
  EXPECT_FALSE(v2::acquired_calibration_revision_is_valid(
    v2::reserved_calibration_revision
  ));
}

TEST(DirectxShaderTest, FixedShapeHostSbsCacheOwnsAuthenticatedSourceSnapshots) {
  namespace cache = models::host_sbs_shader_cache;
  const auto unique = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()
  );
  const auto shader_root =
    std::filesystem::temp_directory_path() / ("apollo-sbs-shader-cache-" + unique);
  ASSERT_TRUE(std::filesystem::create_directories(shader_root));

  struct cleanup_t {
    std::filesystem::path path;

    ~cleanup_t() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup {shader_root};

  const auto root_path = shader_root / "root.hlsl";
  const auto include_path = shader_root / "shared.hlsl";
  const std::string root_source =
    "#include \"shared.hlsl\"\n"
    "RWStructuredBuffer<float> Out : register(u0);\n"
    "[numthreads(1, 1, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) { Out[0] = SHARED_VALUE; }\n";
  const auto write_source = [](const std::filesystem::path &path, const std::string_view source) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
    return output.good();
  };
  ASSERT_TRUE(write_source(root_path, root_source));
  ASSERT_TRUE(write_source(include_path, "#define SHARED_VALUE 1.0f\n"));

  constexpr std::array specs {cache::shader_spec {"root.hlsl"}};
  const auto initial = cache::snapshot_sources(shader_root, specs);
  ASSERT_TRUE(initial);
  const auto initial_digest = cache::source_closure_sha256(initial);
  ASSERT_EQ(initial_digest.size(), 64u);

  // The snapshot owns A's bytes. A disk mutation before compilation cannot create bytecode B
  // carrying identity A; a new snapshot receives a new transitive-closure identity.
  ASSERT_TRUE(write_source(include_path, "#define SHARED_VALUE 2.0f\n"));
  const auto initial_bytecode = cache::get(initial, specs.front());
  ASSERT_TRUE(initial_bytecode);
  const auto changed = cache::snapshot_sources(shader_root, specs);
  ASSERT_TRUE(changed);
  EXPECT_NE(cache::source_closure_sha256(changed), initial_digest);
  const auto changed_bytecode = cache::get(changed, specs.front());
  ASSERT_TRUE(changed_bytecode);
  EXPECT_NE(*initial_bytecode, *changed_bytecode);

  ASSERT_TRUE(write_source(include_path, "#define SHARED_VALUE 1.0f\n"));
  const auto restored = cache::snapshot_sources(shader_root, specs);
  ASSERT_TRUE(restored);
  EXPECT_EQ(cache::source_closure_sha256(restored), initial_digest);
  EXPECT_EQ(cache::get(restored, specs.front()), initial_bytecode);

  // Root/spec changes are bound; an unrelated file is not. Line-ending conversion is normalized
  // so a Git CRLF checkout does not invalidate the reviewed source contract.
  ASSERT_TRUE(write_source(shader_root / "unrelated.hlsl", "// unrelated\n"));
  EXPECT_EQ(
    cache::source_closure_sha256(cache::snapshot_sources(shader_root, specs)),
    initial_digest
  );
  ASSERT_TRUE(write_source(root_path, root_source + "// changed generation\n"));
  const auto current = cache::snapshot_sources(shader_root, specs);
  ASSERT_TRUE(current);
  EXPECT_NE(cache::source_closure_sha256(current), initial_digest);
  constexpr std::array different_entry {
    cache::shader_spec {"root.hlsl", "different", "cs_5_0"}
  };
  EXPECT_NE(
    cache::source_closure_sha256(cache::snapshot_sources(shader_root, different_entry)),
    cache::source_closure_sha256(current)
  );
  constexpr std::array different_target {
    cache::shader_spec {"root.hlsl", "main", "cs_4_0"}
  };
  EXPECT_NE(
    cache::source_closure_sha256(cache::snapshot_sources(shader_root, different_target)),
    cache::source_closure_sha256(current)
  );
  ASSERT_TRUE(write_source(root_path,
                           "#include \"shared.hlsl\"\r\n"
                           "RWStructuredBuffer<float> Out : register(u0);\r\n"
                           "[numthreads(1, 1, 1)]\r\n"
                           "void main(uint3 id : SV_DispatchThreadID) { Out[0] = SHARED_VALUE; }\r\n"));
  ASSERT_TRUE(write_source(include_path, "#define SHARED_VALUE 1.0f\r\n"));
  EXPECT_EQ(
    cache::source_closure_sha256(cache::snapshot_sources(shader_root, specs)),
    initial_digest
  );
  ASSERT_TRUE(write_source(root_path,
                           "#include \"shared.hlsl\"\r"
                           "RWStructuredBuffer<float> Out : register(u0);\r"
                           "[numthreads(1, 1, 1)]\r"
                           "void main(uint3 id : SV_DispatchThreadID) { Out[0] = SHARED_VALUE; }\r"));
  ASSERT_TRUE(write_source(include_path, "#define SHARED_VALUE 1.0f\r"));
  EXPECT_EQ(
    cache::source_closure_sha256(cache::snapshot_sources(shader_root, specs)),
    initial_digest
  );
}

TEST(DirectxShaderTest, FixedShapeHostSbsCacheCompilesAuthenticatedNestedIncludeEdges) {
  namespace cache = models::host_sbs_shader_cache;
  const auto unique = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()
  );
  const auto root = std::filesystem::temp_directory_path() /
                    ("apollo-sbs-shader-nested-" + unique);
  ASSERT_TRUE(std::filesystem::create_directories(root / "nested"));

  struct cleanup_t {
    std::filesystem::path path;

    ~cleanup_t() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup {root};

  const auto write_source = [](const std::filesystem::path &path, const std::string_view source) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
    return output.good();
  };
  ASSERT_TRUE(write_source(root / "root.hlsl",
                           "#include \"nested/parent.hlsl\"\n"
                           "RWStructuredBuffer<float> Out : register(u0);\n"
                           "[numthreads(1, 1, 1)]\n"
                           "void main(uint3 id : SV_DispatchThreadID) { Out[0] = NESTED_VALUE; }\n"));
  ASSERT_TRUE(write_source(root / "nested" / "parent.hlsl", "#include \"shared.hlsl\"\n"));
  ASSERT_TRUE(write_source(root / "nested" / "shared.hlsl", "#define NESTED_VALUE 2.0f\n"));
  ASSERT_TRUE(write_source(root / "shared.hlsl", "#define NESTED_VALUE 1.0f\n"));
  constexpr std::array specs {cache::shader_spec {"root.hlsl"}};
  const auto initial = cache::snapshot_sources(root, specs);
  ASSERT_TRUE(initial);
  const auto initial_digest = cache::source_closure_sha256(initial);
  ASSERT_TRUE(cache::get(initial, specs.front()));

  // The same token names a different root file, but that unreachable file is neither hashed nor
  // compiled. The authenticated parent/token edge selects nested/shared.hlsl exactly.
  ASSERT_TRUE(write_source(root / "shared.hlsl", "#define NESTED_VALUE 9.0f\n"));
  const auto unrelated = cache::snapshot_sources(root, specs);
  ASSERT_TRUE(unrelated);
  EXPECT_EQ(cache::source_closure_sha256(unrelated), initial_digest);
  ASSERT_TRUE(write_source(root / "nested" / "shared.hlsl", "#define NESTED_VALUE 3.0f\n"));
  const auto changed = cache::snapshot_sources(root, specs);
  ASSERT_TRUE(changed);
  EXPECT_NE(cache::source_closure_sha256(changed), initial_digest);
  ASSERT_TRUE(cache::get(changed, specs.front()));
}

TEST(DirectxShaderTest, FixedShapeHostSbsCacheRejectsUnauthenticatedIncludes) {
  namespace cache = models::host_sbs_shader_cache;
  const auto unique = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()
  );
  const auto parent = std::filesystem::temp_directory_path() /
                      ("apollo-sbs-shader-include-" + unique);
  const auto root = parent / "root";
  ASSERT_TRUE(std::filesystem::create_directories(root));

  struct cleanup_t {
    std::filesystem::path path;

    ~cleanup_t() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup {parent};

  const auto write_source = [](const std::filesystem::path &path, const std::string_view source) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
    return output.good();
  };
  constexpr std::array specs {cache::shader_spec {"root.hlsl"}};

  ASSERT_TRUE(write_source(root / "root.hlsl", "#include \"missing.hlsl\"\n"));
  EXPECT_FALSE(cache::snapshot_sources(root, specs));
  ASSERT_TRUE(write_source(parent / "outside.hlsl", "#define VALUE 1\n"));
  ASSERT_TRUE(write_source(root / "root.hlsl", "#include \"../outside.hlsl\"\n"));
  EXPECT_FALSE(cache::snapshot_sources(root, specs));
  ASSERT_TRUE(write_source(root / "root.hlsl", "#include <outside.hlsl>\n"));
  EXPECT_FALSE(cache::snapshot_sources(root, specs));
  ASSERT_TRUE(write_source(root / "root.hlsl", "#define WHICH \"outside.hlsl\"\n#include WHICH\n"));
  EXPECT_FALSE(cache::snapshot_sources(root, specs));

  std::error_code symlink_error;
  std::filesystem::create_symlink(parent / "outside.hlsl", root / "linked.hlsl", symlink_error);
  if (!symlink_error) {
    ASSERT_TRUE(write_source(root / "root.hlsl", "#include \"linked.hlsl\"\n"));
    EXPECT_FALSE(cache::snapshot_sources(root, specs));
  }
}

TEST(DirectxShaderTest, CompilesGeneratedAdaptiveStateConsumers) {
  using Microsoft::WRL::ComPtr;

  constexpr std::array shaders {
    std::tuple {"rgb_to_nchw_cs.hlsl", "content_main", "cs_5_0"},
    std::tuple {"rgb_to_nchw_cs.hlsl", "pad_main", "cs_5_0"},
    std::tuple {"buffer_to_tex_cs.hlsl", "pad_main", "cs_5_0"},
    std::tuple {"depth_ema_motion_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_minmax_ema_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_hist_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_scene_cut_evidence_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_scene_cut_resolve_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_valid_history_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_coordinate_v2_ownership_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_flat_identity_ps.hlsl", "main_ps", "ps_5_0"},
    std::tuple {"sbs_reprojection_v2_live_ps.hlsl", "main_ps", "ps_5_0"},
    std::tuple {"sbs_reprojection_v2_diagnostics_ps.hlsl", "mapping_ps", "ps_5_0"},
    std::tuple {"sbs_reprojection_v2_diagnostics_ps.hlsl", "mask_ps", "ps_5_0"},
  };
  for (const auto &[filename, entrypoint, target] : shaders) {
    const std::filesystem::path shader_path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> shader_errors;
    const auto status = D3DCompileFromFile(
      shader_path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entrypoint,
      target,
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &shader_blob,
      &shader_errors
    );
    EXPECT_TRUE(SUCCEEDED(status))
      << filename << ": "
      << (shader_errors ?
            static_cast<const char *>(shader_errors->GetBufferPointer()) :
            "no compiler diagnostics");
  }
}

TEST(DirectxShaderTest, FusedV2MomentsPreserveNormalizationBitSemantics) {
  using Microsoft::WRL::ComPtr;

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  constexpr D3D_FEATURE_LEVEL requested_levels[] {D3D_FEATURE_LEVEL_11_0};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested_levels,
    static_cast<UINT>(std::size(requested_levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  )));
  ASSERT_GE(feature_level, D3D_FEATURE_LEVEL_11_0);

  const auto compile_shader = [&](const char *filename) {
    const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &bytecode,
      &diagnostics
    );
    EXPECT_TRUE(SUCCEEDED(status))
      << filename << ": "
      << (diagnostics ?
            static_cast<const char *>(diagnostics->GetBufferPointer()) :
            "no compiler diagnostics");
    return bytecode;
  };
  const auto moments_bytecode = compile_shader(
    "depth_coordinate_v2_moments_cs.hlsl"
  );
  const auto frame_bytecode = compile_shader(
    "depth_coordinate_v2_frame_resolve_cs.hlsl"
  );
  ASSERT_TRUE(moments_bytecode);
  ASSERT_TRUE(frame_bytecode);
  ComPtr<ID3D11ComputeShader> moments_shader;
  ComPtr<ID3D11ComputeShader> frame_shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    moments_bytecode->GetBufferPointer(),
    moments_bytecode->GetBufferSize(),
    nullptr,
    &moments_shader
  )));
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    frame_bytecode->GetBufferPointer(),
    frame_bytecode->GetBufferSize(),
    nullptr,
    &frame_shader
  )));

  constexpr UINT width = 4u;
  constexpr UINT height = 1u;
  constexpr UINT texel_count = width * height;
  std::array<float, texel_count> initial_raw {};
  D3D11_BUFFER_DESC raw_desc {};
  raw_desc.ByteWidth = sizeof(initial_raw);
  raw_desc.Usage = D3D11_USAGE_DEFAULT;
  raw_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  raw_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  raw_desc.StructureByteStride = sizeof(float);
  D3D11_SUBRESOURCE_DATA raw_data {initial_raw.data(), 0, 0};
  ComPtr<ID3D11Buffer> raw_buffer;
  ComPtr<ID3D11ShaderResourceView> raw_srv;
  ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&raw_desc, &raw_data, &raw_buffer)));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
    raw_buffer.Get(),
    nullptr,
    &raw_srv
  )));

  const std::array<std::uint32_t, texel_count> exclusion {};
  D3D11_TEXTURE2D_DESC exclusion_desc {};
  exclusion_desc.Width = width;
  exclusion_desc.Height = height;
  exclusion_desc.MipLevels = 1u;
  exclusion_desc.ArraySize = 1u;
  exclusion_desc.Format = DXGI_FORMAT_R32_UINT;
  exclusion_desc.SampleDesc.Count = 1u;
  exclusion_desc.Usage = D3D11_USAGE_DEFAULT;
  exclusion_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA exclusion_data {};
  exclusion_data.pSysMem = exclusion.data();
  exclusion_data.SysMemPitch = width * sizeof(std::uint32_t);
  ComPtr<ID3D11Texture2D> exclusion_texture;
  ComPtr<ID3D11ShaderResourceView> exclusion_srv;
  ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(
    &exclusion_desc,
    &exclusion_data,
    &exclusion_texture
  )));
  ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(
    exclusion_texture.Get(),
    nullptr,
    &exclusion_srv
  )));

  const auto create_structured_output = [&](const UINT vector_count, ComPtr<ID3D11Buffer> &buffer, ComPtr<ID3D11ShaderResourceView> *srv, ComPtr<ID3D11UnorderedAccessView> &uav) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = vector_count * sizeof(std::uint32_t) * 4u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS |
                     (srv ? D3D11_BIND_SHADER_RESOURCE : 0u);
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(std::uint32_t) * 4u;
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
           (!srv || SUCCEEDED(device->CreateShaderResourceView(
                      buffer.Get(),
                      nullptr,
                      srv->ReleaseAndGetAddressOf()
                    ))) &&
           SUCCEEDED(device->CreateUnorderedAccessView(
             buffer.Get(),
             nullptr,
             &uav
           ));
  };
  ComPtr<ID3D11Buffer> partial_buffer;
  ComPtr<ID3D11ShaderResourceView> partial_srv;
  ComPtr<ID3D11UnorderedAccessView> partial_uav;
  ComPtr<ID3D11Buffer> frame_buffer;
  ComPtr<ID3D11UnorderedAccessView> frame_uav;
  ASSERT_TRUE(create_structured_output(
    6u,
    partial_buffer,
    &partial_srv,
    partial_uav
  ));
  ASSERT_TRUE(create_structured_output(
    2u,
    frame_buffer,
    nullptr,
    frame_uav
  ));

  const std::array<std::uint32_t, 4> normalization_identity {{
    0xFFFFFFFFu,
    0u,
    0u,
    0u,
  }};
  D3D11_BUFFER_DESC normalization_desc {};
  normalization_desc.ByteWidth = sizeof(normalization_identity);
  normalization_desc.Usage = D3D11_USAGE_DEFAULT;
  normalization_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  normalization_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  D3D11_SUBRESOURCE_DATA normalization_data {
    normalization_identity.data(),
    0,
    0
  };
  ComPtr<ID3D11Buffer> normalization_buffer;
  ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
    &normalization_desc,
    &normalization_data,
    &normalization_buffer
  )));
  D3D11_UNORDERED_ACCESS_VIEW_DESC normalization_view {};
  normalization_view.Format = DXGI_FORMAT_R32_TYPELESS;
  normalization_view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  normalization_view.Buffer.NumElements = 4u;
  normalization_view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
  ComPtr<ID3D11UnorderedAccessView> normalization_uav;
  ASSERT_TRUE(SUCCEEDED(device->CreateUnorderedAccessView(
    normalization_buffer.Get(),
    &normalization_view,
    &normalization_uav
  )));

  std::array<std::uint32_t, 16> constants {};
  constants[0] = width;
  constants[1] = height;
  constants[5] = 512u;
  constants[11] = width;
  constants[12] = height;
  D3D11_BUFFER_DESC constant_desc {};
  constant_desc.ByteWidth = sizeof(constants);
  constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
  constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  D3D11_SUBRESOURCE_DATA constant_data {constants.data(), 0, 0};
  ComPtr<ID3D11Buffer> constant_buffer;
  ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
    &constant_desc,
    &constant_data,
    &constant_buffer
  )));

  const auto read_words = [&](ID3D11Buffer *source, const UINT word_count) {
    D3D11_BUFFER_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0u;
    desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
      return std::vector<std::uint32_t> {};
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return std::vector<std::uint32_t> {};
    }
    const auto *words = static_cast<const std::uint32_t *>(mapped.pData);
    std::vector<std::uint32_t> result(words, words + word_count);
    context->Unmap(staging.Get(), 0u);
    return result;
  };

  const auto run = [&](const std::array<float, texel_count> &raw, const std::array<std::uint32_t, texel_count> &excluded) {
    context->UpdateSubresource(raw_buffer.Get(), 0u, nullptr, raw.data(), 0u, 0u);
    context->UpdateSubresource(
      exclusion_texture.Get(),
      0u,
      nullptr,
      excluded.data(),
      width * sizeof(std::uint32_t),
      0u
    );
    ID3D11Buffer *constant_buffers[] = {constant_buffer.Get()};
    context->CSSetConstantBuffers(0u, 1u, constant_buffers);
    ID3D11ShaderResourceView *moments_srvs[] = {raw_srv.Get(), exclusion_srv.Get()};
    ID3D11UnorderedAccessView *moments_uavs[] = {partial_uav.Get()};
    context->CSSetShader(moments_shader.Get(), nullptr, 0u);
    context->CSSetShaderResources(0u, 2u, moments_srvs);
    context->CSSetUnorderedAccessViews(0u, 1u, moments_uavs, nullptr);
    context->Dispatch(2u, 1u, 1u);

    ID3D11ShaderResourceView *null_moments_srvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
    context->CSSetShaderResources(0u, 2u, null_moments_srvs);
    context->CSSetUnorderedAccessViews(0u, 1u, null_uavs, nullptr);

    ID3D11ShaderResourceView *frame_srvs[] = {partial_srv.Get()};
    ID3D11UnorderedAccessView *frame_uavs[] = {
      frame_uav.Get(),
      normalization_uav.Get(),
    };
    context->CSSetShader(frame_shader.Get(), nullptr, 0u);
    context->CSSetShaderResources(0u, 1u, frame_srvs);
    context->CSSetUnorderedAccessViews(0u, 2u, frame_uavs, nullptr);
    context->Dispatch(1u, 1u, 1u);
    ID3D11ShaderResourceView *null_frame_srvs[] = {nullptr};
    context->CSSetShaderResources(0u, 1u, null_frame_srvs);
    context->CSSetUnorderedAccessViews(0u, 2u, null_uavs, nullptr);
    return std::pair {
      read_words(frame_buffer.Get(), 8u),
      read_words(normalization_buffer.Get(), 4u),
    };
  };

  const auto [finite_frame, finite_normalization] = run(
    {{-1.0f, 0.0f, -0.0f, 1.0f}},
    exclusion
  );
  ASSERT_EQ(finite_frame.size(), 8u);
  ASSERT_EQ(finite_normalization.size(), 4u);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[0]), 0.0f);
  EXPECT_NEAR(std::bit_cast<float>(finite_frame[1]), std::sqrt(0.5f), 1.0e-6f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[2]), -1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[3]), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[4]), 4.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[5]), 4.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(finite_frame[6]), 1.0f);
  EXPECT_EQ(finite_normalization[0], 0x00000000u);
  EXPECT_EQ(finite_normalization[1], 0x80000000u);
  EXPECT_EQ(finite_normalization[2], 3u);
  EXPECT_EQ(finite_normalization[3], 4u);

  // With no negative value, signed zero alone retains the legacy unsigned ordering: -0 is the
  // maximum bit pattern and the smallest positive value is the minimum. The resulting numeric
  // bounds are intentionally invalid even though valid_count equals eligible_count.
  const auto [zero_order_frame, zero_order_normalization] = run(
    {{-0.0f, 1.0f, 2.0f, 3.0f}},
    exclusion
  );
  ASSERT_EQ(zero_order_frame.size(), 8u);
  ASSERT_EQ(zero_order_normalization.size(), 4u);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(zero_order_frame[6]), 1.0f);
  EXPECT_EQ(zero_order_normalization[0], std::bit_cast<std::uint32_t>(1.0f));
  EXPECT_EQ(zero_order_normalization[1], 0x80000000u);
  EXPECT_EQ(zero_order_normalization[2], 4u);
  EXPECT_EQ(zero_order_normalization[3], 4u);

  const auto [invalid_frame, invalid_normalization] = run(
    {{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -2.0f,
      2.0f,
    }},
    exclusion
  );
  ASSERT_EQ(invalid_frame.size(), 8u);
  ASSERT_EQ(invalid_normalization.size(), 4u);
  for (std::size_t index = 0u; index < 4u; ++index) {
    EXPECT_FLOAT_EQ(std::bit_cast<float>(invalid_frame[index]), 0.0f);
  }
  EXPECT_FLOAT_EQ(std::bit_cast<float>(invalid_frame[4]), 2.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(invalid_frame[5]), 4.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(invalid_frame[6]), 0.0f);
  EXPECT_EQ(invalid_normalization[0], std::bit_cast<std::uint32_t>(2.0f));
  EXPECT_EQ(invalid_normalization[1], std::bit_cast<std::uint32_t>(2.0f));
  EXPECT_EQ(invalid_normalization[2], 1u);
  EXPECT_EQ(invalid_normalization[3], 4u);

  const std::array<std::uint32_t, texel_count> excluded_tail {{0u, 1u, 1u, 1u}};
  const auto [excluded_frame, excluded_normalization] = run(
    {{
      1.0f,
      -1.0f,
      std::numeric_limits<float>::quiet_NaN(),
      -0.0f,
    }},
    excluded_tail
  );
  ASSERT_EQ(excluded_frame.size(), 8u);
  ASSERT_EQ(excluded_normalization.size(), 4u);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(excluded_frame[0]), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(excluded_frame[4]), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(excluded_frame[5]), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(excluded_frame[6]), 1.0f);
  EXPECT_EQ(excluded_normalization[0], std::bit_cast<std::uint32_t>(1.0f));
  EXPECT_EQ(excluded_normalization[1], std::bit_cast<std::uint32_t>(1.0f));
  EXPECT_EQ(excluded_normalization[2], 1u);
  EXPECT_EQ(excluded_normalization[3], 1u);
}

TEST(DirectxShaderTest, BufferToTexMainThenPadReplicatesBoundaryDepthAndClearsPaddingMotion) {
  using Microsoft::WRL::ComPtr;

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  constexpr D3D_FEATURE_LEVEL requested_levels[] {D3D_FEATURE_LEVEL_11_0};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested_levels,
    static_cast<UINT>(std::size(requested_levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  )));
  ASSERT_GE(feature_level, D3D_FEATURE_LEVEL_11_0);

  const auto shader_path =
    std::filesystem::path(SUNSHINE_SHADERS_DIR) / "buffer_to_tex_cs.hlsl";
  const auto compile_shader = [&](const char *entrypoint) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      shader_path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entrypoint,
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &bytecode,
      &diagnostics
    );
    EXPECT_TRUE(SUCCEEDED(status))
      << entrypoint << ": "
      << (diagnostics ?
            static_cast<const char *>(diagnostics->GetBufferPointer()) :
            "no compiler diagnostics");
    return bytecode;
  };

  const auto main_bytecode = compile_shader("main");
  const auto pad_bytecode = compile_shader("pad_main");
  ASSERT_TRUE(main_bytecode);
  ASSERT_TRUE(pad_bytecode);
  ComPtr<ID3D11ComputeShader> main_shader;
  ComPtr<ID3D11ComputeShader> pad_shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    main_bytecode->GetBufferPointer(),
    main_bytecode->GetBufferSize(),
    nullptr,
    &main_shader
  )));
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    pad_bytecode->GetBufferPointer(),
    pad_bytecode->GetBufferSize(),
    nullptr,
    &pad_shader
  )));

  constexpr UINT width = 7u;
  constexpr UINT height = 6u;
  constexpr UINT texel_count = width * height;
  constexpr models::depth_tensor_content_rect_t content {2u, 1u, 6u, 5u};

  std::array<float, texel_count> raw_depth {};
  std::array<float, texel_count> previous_depth {};
  std::array<std::uint32_t, texel_count> exclusion {};
  std::array<float, texel_count> initial_output {};
  std::array<std::uint32_t, texel_count> initial_motion {};
  for (UINT y = 0u; y < height; ++y) {
    for (UINT x = 0u; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y) * width + x;
      raw_depth[index] = static_cast<float>(index + 1u);
      previous_depth[index] = 0.75f;
      exclusion[index] =
        x >= content.left && x < content.right &&
            y >= content.top && y < content.bottom ?
          0u :
          1u;
      initial_output[index] = -123.0f;
      initial_motion[index] = 0xdeadbeefu;
    }
  }

  const auto create_structured_srv = [&]<typename T>(
                                       const T *values,
                                       UINT count,
                                       ComPtr<ID3D11Buffer> &buffer,
                                       ComPtr<ID3D11ShaderResourceView> &srv
                                     ) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = count * sizeof(T);
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(T);
    D3D11_SUBRESOURCE_DATA data {};
    data.pSysMem = values;
    return SUCCEEDED(device->CreateBuffer(&desc, &data, &buffer)) &&
           SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
  };

  ComPtr<ID3D11Buffer> raw_buffer;
  ComPtr<ID3D11ShaderResourceView> raw_srv;
  ASSERT_TRUE(create_structured_srv(
    raw_depth.data(), texel_count, raw_buffer, raw_srv
  ));
  const std::array<float, 4> minmax_ema {0.0f, 64.0f, 1.0f, 2.0f};
  ComPtr<ID3D11Buffer> minmax_buffer;
  ComPtr<ID3D11ShaderResourceView> minmax_srv;
  ASSERT_TRUE(create_structured_srv(
    &minmax_ema, 1u, minmax_buffer, minmax_srv
  ));

  const auto create_input_texture = [&](DXGI_FORMAT format,
                                        const void *values,
                                        ComPtr<ID3D11Texture2D> &texture,
                                        ComPtr<ID3D11ShaderResourceView> &srv) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1u;
    desc.ArraySize = 1u;
    desc.Format = format;
    desc.SampleDesc.Count = 1u;
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA data {};
    data.pSysMem = values;
    data.SysMemPitch = width * sizeof(std::uint32_t);
    return SUCCEEDED(device->CreateTexture2D(&desc, &data, &texture)) &&
           SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &srv));
  };

  ComPtr<ID3D11Texture2D> previous_texture;
  ComPtr<ID3D11ShaderResourceView> previous_srv;
  ASSERT_TRUE(create_input_texture(
    DXGI_FORMAT_R32_FLOAT,
    previous_depth.data(),
    previous_texture,
    previous_srv
  ));
  ComPtr<ID3D11Texture2D> exclusion_texture;
  ComPtr<ID3D11ShaderResourceView> exclusion_srv;
  ASSERT_TRUE(create_input_texture(
    DXGI_FORMAT_R32_UINT,
    exclusion.data(),
    exclusion_texture,
    exclusion_srv
  ));

  const auto create_output_texture = [&](DXGI_FORMAT format,
                                         const void *values,
                                         ComPtr<ID3D11Texture2D> &texture,
                                         ComPtr<ID3D11UnorderedAccessView> &uav) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1u;
    desc.ArraySize = 1u;
    desc.Format = format;
    desc.SampleDesc.Count = 1u;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_SUBRESOURCE_DATA data {};
    data.pSysMem = values;
    data.SysMemPitch = width * sizeof(std::uint32_t);
    return SUCCEEDED(device->CreateTexture2D(&desc, &data, &texture)) &&
           SUCCEEDED(device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav));
  };

  ComPtr<ID3D11Texture2D> output_texture;
  ComPtr<ID3D11UnorderedAccessView> output_uav;
  ASSERT_TRUE(create_output_texture(
    DXGI_FORMAT_R32_FLOAT,
    initial_output.data(),
    output_texture,
    output_uav
  ));
  ComPtr<ID3D11Texture2D> motion_texture;
  ComPtr<ID3D11UnorderedAccessView> motion_uav;
  ASSERT_TRUE(create_output_texture(
    DXGI_FORMAT_R32_UINT,
    initial_motion.data(),
    motion_texture,
    motion_uav
  ));

  std::array<std::uint32_t, 16> constants {};
  constants[0] = width;
  constants[1] = height;
  constants[3] = std::bit_cast<std::uint32_t>(1.0f);
  constants[9] = content.left;
  constants[10] = content.top;
  constants[11] = content.right;
  constants[12] = content.bottom;
  D3D11_BUFFER_DESC constant_desc {};
  constant_desc.ByteWidth = sizeof(constants);
  constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
  constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  D3D11_SUBRESOURCE_DATA constant_data {};
  constant_data.pSysMem = constants.data();
  ComPtr<ID3D11Buffer> constant_buffer;
  ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(
    &constant_desc, &constant_data, &constant_buffer
  )));

  ID3D11ShaderResourceView *srvs[] = {
    raw_srv.Get(),
    minmax_srv.Get(),
    previous_srv.Get(),
    exclusion_srv.Get(),
  };
  ID3D11UnorderedAccessView *uavs[] = {output_uav.Get(), motion_uav.Get()};
  ID3D11Buffer *constant_buffers[] = {constant_buffer.Get()};
  context->CSSetShaderResources(0u, static_cast<UINT>(std::size(srvs)), srvs);
  context->CSSetUnorderedAccessViews(0u, static_cast<UINT>(std::size(uavs)), uavs, nullptr);
  context->CSSetConstantBuffers(0u, 1u, constant_buffers);
  context->CSSetShader(main_shader.Get(), nullptr, 0u);
  context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
  context->CSSetShader(pad_shader.Get(), nullptr, 0u);
  context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);

  ID3D11ShaderResourceView *null_srvs[] = {nullptr, nullptr, nullptr, nullptr};
  ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
  ID3D11Buffer *null_constant_buffers[] = {nullptr};
  context->CSSetShader(nullptr, nullptr, 0u);
  context->CSSetShaderResources(
    0u, static_cast<UINT>(std::size(null_srvs)), null_srvs
  );
  context->CSSetUnorderedAccessViews(
    0u, static_cast<UINT>(std::size(null_uavs)), null_uavs, nullptr
  );
  context->CSSetConstantBuffers(0u, 1u, null_constant_buffers);

  const auto read_texture_bits = [&](ID3D11Texture2D *source,
                                     std::array<std::uint32_t, texel_count> &bits) {
    D3D11_TEXTURE2D_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    for (UINT y = 0u; y < height; ++y) {
      std::memcpy(
        bits.data() + static_cast<std::size_t>(y) * width,
        static_cast<const std::byte *>(mapped.pData) +
          static_cast<std::size_t>(y) * mapped.RowPitch,
        width * sizeof(std::uint32_t)
      );
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  };

  std::array<std::uint32_t, texel_count> depth_bits {};
  std::array<std::uint32_t, texel_count> motion_bits {};
  ASSERT_TRUE(read_texture_bits(output_texture.Get(), depth_bits));
  ASSERT_TRUE(read_texture_bits(motion_texture.Get(), motion_bits));

  const auto initial_depth_bits = std::bit_cast<std::uint32_t>(-123.0f);
  const auto held_history_bits = std::bit_cast<std::uint32_t>(0.75f);
  for (UINT y = 0u; y < height; ++y) {
    for (UINT x = 0u; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y) * width + x;
      if (exclusion[index] == 0u) {
        EXPECT_NE(depth_bits[index], initial_depth_bits) << x << ',' << y;
        EXPECT_NE(depth_bits[index], held_history_bits) << x << ',' << y;
        continue;
      }

      const auto clamped_x = std::clamp(x, content.left, content.right - 1u);
      const auto clamped_y = std::clamp(y, content.top, content.bottom - 1u);
      const auto clamped_index =
        static_cast<std::size_t>(clamped_y) * width + clamped_x;
      EXPECT_EQ(depth_bits[index], depth_bits[clamped_index]) << x << ',' << y;
      EXPECT_EQ(motion_bits[index], 0u) << x << ',' << y;
    }
  }
}

TEST(DirectxShaderTest, RgbToNchwAreaSamplingMatchesExactFootprints) {
  using Microsoft::WRL::ComPtr;

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  D3D_FEATURE_LEVEL feature_level {};
  constexpr D3D_FEATURE_LEVEL requested_levels[] = {
    D3D_FEATURE_LEVEL_11_0
  };
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
    nullptr,
    D3D_DRIVER_TYPE_WARP,
    nullptr,
    0,
    requested_levels,
    static_cast<UINT>(std::size(requested_levels)),
    D3D11_SDK_VERSION,
    &device,
    &feature_level,
    &context
  )));
  ASSERT_GE(feature_level, D3D_FEATURE_LEVEL_11_0);

  const std::filesystem::path shader_path =
    SUNSHINE_SHADERS_DIR "/rgb_to_nchw_cs.hlsl";
  const auto compile_shader = [&](const char *entrypoint) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto status = D3DCompileFromFile(
      shader_path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entrypoint,
      "cs_5_0",
      D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &bytecode,
      &diagnostics
    );
    EXPECT_TRUE(SUCCEEDED(status))
      << entrypoint << ": "
      << (diagnostics ?
            static_cast<const char *>(diagnostics->GetBufferPointer()) :
            "no compiler diagnostics");
    return bytecode;
  };
  const auto main_bytecode = compile_shader("main");
  const auto content_bytecode = compile_shader("content_main");
  const auto pad_bytecode = compile_shader("pad_main");
  ASSERT_TRUE(main_bytecode);
  ASSERT_TRUE(content_bytecode);
  ASSERT_TRUE(pad_bytecode);

  ComPtr<ID3D11ComputeShader> shader;
  ComPtr<ID3D11ComputeShader> content_shader;
  ComPtr<ID3D11ComputeShader> pad_shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    main_bytecode->GetBufferPointer(),
    main_bytecode->GetBufferSize(),
    nullptr,
    &shader
  )));
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    content_bytecode->GetBufferPointer(),
    content_bytecode->GetBufferSize(),
    nullptr,
    &content_shader
  )));
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    pad_bytecode->GetBufferPointer(),
    pad_bytecode->GetBufferSize(),
    nullptr,
    &pad_shader
  )));

  using rgba_pixel_t = std::array<float, 4>;
  const auto run_case = [&](
                          UINT source_width,
                          UINT source_height,
                          UINT target_width,
                          UINT target_height,
                          const std::vector<rgba_pixel_t> &source_pixels,
                          std::vector<float> &model_output,
                          std::vector<float> &appearance_ordinal,
                          std::uint32_t color_mode = 0u,
                          models::depth_tensor_content_rect_t tensor_content = {},
                          std::vector<std::uint32_t> *tensor_exclusion = nullptr
                        ) {
    if (source_pixels.size() != static_cast<std::size_t>(source_width) * source_height) {
      return false;
    }

    D3D11_TEXTURE2D_DESC texture_desc {};
    texture_desc.Width = source_width;
    texture_desc.Height = source_height;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
    texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA texture_data {};
    texture_data.pSysMem = source_pixels.data();
    texture_data.SysMemPitch = source_width * sizeof(rgba_pixel_t);

    ComPtr<ID3D11Texture2D> input_texture;
    ComPtr<ID3D11ShaderResourceView> input_srv;
    if (FAILED(device->CreateTexture2D(&texture_desc, &texture_data, &input_texture)) || FAILED(device->CreateShaderResourceView(input_texture.Get(), nullptr, &input_srv))) {
      return false;
    }

    const auto create_output = [&](
                                 UINT float_count,
                                 ComPtr<ID3D11Buffer> &buffer,
                                 ComPtr<ID3D11UnorderedAccessView> &uav
                               ) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = float_count * sizeof(float);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = sizeof(float);
      return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
             SUCCEEDED(device->CreateUnorderedAccessView(
               buffer.Get(),
               nullptr,
               &uav
             ));
    };

    const UINT target_texels = target_width * target_height;
    ComPtr<ID3D11Buffer> model_buffer;
    ComPtr<ID3D11UnorderedAccessView> model_uav;
    ComPtr<ID3D11Buffer> appearance_buffer;
    ComPtr<ID3D11UnorderedAccessView> appearance_uav;
    if (!create_output(target_texels * 3u, model_buffer, model_uav) || !create_output(target_texels, appearance_buffer, appearance_uav)) {
      return false;
    }

    D3D11_TEXTURE2D_DESC exclusion_desc {};
    exclusion_desc.Width = target_width;
    exclusion_desc.Height = target_height;
    exclusion_desc.MipLevels = 1u;
    exclusion_desc.ArraySize = 1u;
    exclusion_desc.Format = DXGI_FORMAT_R32_UINT;
    exclusion_desc.SampleDesc.Count = 1u;
    exclusion_desc.Usage = D3D11_USAGE_DEFAULT;
    exclusion_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> exclusion_texture;
    ComPtr<ID3D11UnorderedAccessView> exclusion_uav;
    if (FAILED(device->CreateTexture2D(
          &exclusion_desc, nullptr, &exclusion_texture)) ||
        FAILED(device->CreateUnorderedAccessView(
          exclusion_texture.Get(), nullptr, &exclusion_uav))) {
      return false;
    }

    const models::depth_tensor_shape_t target_shape {
      static_cast<int>(target_width),
      static_cast<int>(target_height),
    };
    if (!tensor_content.valid(target_shape)) {
      tensor_content = {0u, 0u, target_width, target_height};
    }
    std::array<std::uint32_t, 16> constants {};
    constants[0] = target_width;
    constants[1] = target_height;
    constants[2] = color_mode;
    constants[9] = tensor_content.left;
    constants[10] = tensor_content.top;
    constants[11] = tensor_content.right;
    constants[12] = tensor_content.bottom;
    D3D11_BUFFER_DESC constant_desc {};
    constant_desc.ByteWidth = sizeof(constants);
    constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
    constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constant_data {};
    constant_data.pSysMem = constants.data();
    ComPtr<ID3D11Buffer> constant_buffer;
    if (FAILED(device->CreateBuffer(&constant_desc, &constant_data, &constant_buffer))) {
      return false;
    }

    ID3D11ShaderResourceView *input_srvs[] = {input_srv.Get()};
    ID3D11UnorderedAccessView *output_uavs[] = {
      model_uav.Get(),
      appearance_uav.Get(),
      exclusion_uav.Get(),
    };
    ID3D11Buffer *constant_buffers[] = {constant_buffer.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, input_srvs);
    context->CSSetUnorderedAccessViews(0, 3, output_uavs, nullptr);
    context->CSSetConstantBuffers(0, 1, constant_buffers);
    if (!tensor_content.full(target_shape)) {
      context->CSSetShader(content_shader.Get(), nullptr, 0);
      context->Dispatch(
        (tensor_content.width() + 15u) / 16u,
        (tensor_content.height() + 15u) / 16u,
        1u
      );
      context->CSSetShader(pad_shader.Get(), nullptr, 0);
      context->Dispatch(
        (target_width + 15u) / 16u,
        (target_height + 15u) / 16u,
        1u
      );
    } else {
      context->Dispatch(
        (target_width + 15u) / 16u,
        (target_height + 15u) / 16u,
        1u
      );
    }

    ID3D11ShaderResourceView *null_srvs[] = {nullptr};
    ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr, nullptr};
    context->CSSetShaderResources(0, 1, null_srvs);
    context->CSSetUnorderedAccessViews(0, 3, null_uavs, nullptr);

    const auto read_buffer = [&](
                               ID3D11Buffer *source,
                               UINT float_count,
                               std::vector<float> &values
                             ) {
      D3D11_BUFFER_DESC staging_desc {};
      source->GetDesc(&staging_desc);
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      staging_desc.StructureByteStride = 0;

      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) {
        return false;
      }
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      const auto *begin = static_cast<const float *>(mapped.pData);
      values.assign(begin, begin + float_count);
      context->Unmap(staging.Get(), 0);
      return true;
    };

    const auto read_exclusion = [&]() {
      if (!tensor_exclusion) {
        return true;
      }
      auto staging_desc = exclusion_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        return false;
      }
      context->CopyResource(staging.Get(), exclusion_texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        return false;
      }
      tensor_exclusion->resize(target_texels);
      for (UINT y = 0u; y < target_height; ++y) {
        std::memcpy(
          tensor_exclusion->data() + static_cast<std::size_t>(y) * target_width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          static_cast<std::size_t>(target_width) * sizeof(std::uint32_t)
        );
      }
      context->Unmap(staging.Get(), 0u);
      return true;
    };

    return read_buffer(
             model_buffer.Get(),
             target_texels * 3u,
             model_output
           ) &&
           read_buffer(
             appearance_buffer.Get(),
             target_texels,
             appearance_ordinal
           ) && read_exclusion();
  };

  constexpr std::array imagenet_mean {0.485f, 0.456f, 0.406f};
  constexpr std::array imagenet_stddev {0.229f, 0.224f, 0.225f};
  const auto reconstructed_channel = [&](
                                       const std::vector<float> &model_output,
                                       UINT target_texels,
                                       UINT target_index,
                                       UINT channel
                                     ) {
    return model_output[channel * target_texels + target_index] *
             imagenet_stddev[channel] +
           imagenet_mean[channel];
  };
  const auto linear_to_srgb = [](float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.0031308f ?
             value * 12.92f :
             1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
  };
  const auto srgb_to_linear = [](float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.04045f ?
             value / 12.92f :
             std::pow((value + 0.055f) / 1.055f, 2.4f);
  };
  const auto depth_color_reference = [&](rgba_pixel_t pixel, std::uint32_t color_mode) {
    std::array<float, 3> rgb {pixel[0], pixel[1], pixel[2]};
    if (color_mode == 2u) {
      for (auto &channel : rgb) {
        channel = std::max(channel, 0.0f);
      }
      const float luminance = std::max(
        0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2],
        0.0f
      );
      for (auto &channel : rgb) {
        channel /= 1.0f + luminance;
      }
      const float peak = std::max({rgb[0], rgb[1], rgb[2], 1.0f});
      for (auto &channel : rgb) {
        channel /= peak;
      }
    }
    if (color_mode != 0u) {
      for (auto &channel : rgb) {
        channel = linear_to_srgb(channel);
      }
    } else {
      for (auto &channel : rgb) {
        channel = std::clamp(channel, 0.0f, 1.0f);
      }
    }
    return rgb;
  };

  {
    std::vector<rgba_pixel_t> source_pixels(
      25,
      rgba_pixel_t {0.0f, 0.0f, 0.0f, 1.0f}
    );
    source_pixels[12] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    ASSERT_TRUE(run_case(
      5,
      5,
      1,
      1,
      source_pixels,
      model_output,
      appearance_ordinal
    ));
    ASSERT_EQ(model_output.size(), 3u);
    ASSERT_EQ(appearance_ordinal.size(), 1u);
    for (UINT channel = 0; channel < 3u; ++channel) {
      EXPECT_NEAR(
        reconstructed_channel(model_output, 1, 0, channel),
        1.0f / 25.0f,
        2e-6f
      );
    }
    EXPECT_FLOAT_EQ(appearance_ordinal[0], 1.0f);
  }

  {
    std::vector<rgba_pixel_t> source_pixels {
      rgba_pixel_t {0.0f, 0.0f, 0.0f, 1.0f},
      rgba_pixel_t {0.3f, 0.3f, 0.3f, 1.0f},
      rgba_pixel_t {0.6f, 0.6f, 0.6f, 1.0f}
    };
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    ASSERT_TRUE(run_case(
      3,
      1,
      2,
      1,
      source_pixels,
      model_output,
      appearance_ordinal
    ));
    ASSERT_EQ(model_output.size(), 6u);
    ASSERT_EQ(appearance_ordinal.size(), 2u);
    constexpr std::array expected_model_values {0.1f, 0.5f};
    for (UINT target_index = 0; target_index < 2u; ++target_index) {
      for (UINT channel = 0; channel < 3u; ++channel) {
        EXPECT_NEAR(
          reconstructed_channel(model_output, 2, target_index, channel),
          expected_model_values[target_index],
          2e-6f
        );
      }
    }
    EXPECT_FLOAT_EQ(appearance_ordinal[0], 0.0f);
    EXPECT_NEAR(appearance_ordinal[1], srgb_to_linear(0.6f), 2e-6f);
  }

  // FP16 Advanced-Color SDR is linear, not already encoded sRGB. Execute the real shader path
  // and verify each source sample enters the common display-referred model domain exactly once.
  {
    const rgba_pixel_t source {0.18f, 0.0031308f, 2.0f, 1.0f};
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    ASSERT_TRUE(run_case(
      1,
      1,
      1,
      1,
      {source},
      model_output,
      appearance_ordinal,
      static_cast<std::uint32_t>(models::input_color_space::linear_sdr)
    ));
    const auto expected = depth_color_reference(source, 1u);
    for (UINT channel = 0; channel < 3u; ++channel) {
      const float actual = reconstructed_channel(model_output, 1u, 0u, channel);
      EXPECT_TRUE(std::isfinite(actual));
      EXPECT_NEAR(actual, expected[channel], 3e-6f);
    }
    ASSERT_EQ(appearance_ordinal.size(), 1u);
    EXPECT_FLOAT_EQ(appearance_ordinal[0], 2.0f);
  }

  // Transfer-equivalent SDR capture must produce the same model tensor and the same ordinal
  // evidence. This specifically protects the per-source conversion-before-area-resize order.
  {
    const std::vector<rgba_pixel_t> encoded {
      {0.10f, 0.25f, 0.50f, 1.0f},
      {0.20f, 0.30f, 0.60f, 1.0f},
      {0.30f, 0.40f, 0.70f, 1.0f},
      {0.40f, 0.50f, 0.80f, 1.0f},
      {0.15f, 0.35f, 0.55f, 1.0f},
      {0.25f, 0.45f, 0.65f, 1.0f},
      {0.35f, 0.55f, 0.75f, 1.0f},
      {0.45f, 0.65f, 0.85f, 1.0f},
      {0.12f, 0.22f, 0.42f, 1.0f},
      {0.22f, 0.32f, 0.52f, 1.0f},
      {0.32f, 0.42f, 0.62f, 1.0f},
      {0.42f, 0.52f, 0.72f, 1.0f},
      {0.18f, 0.28f, 0.48f, 1.0f},
      {0.28f, 0.38f, 0.58f, 1.0f},
      {0.38f, 0.48f, 0.68f, 1.0f},
      {0.48f, 0.58f, 0.78f, 1.0f},
    };
    auto linear = encoded;
    for (auto &pixel : linear) {
      for (UINT channel = 0; channel < 3u; ++channel) {
        pixel[channel] = srgb_to_linear(pixel[channel]);
      }
    }
    std::vector<float> encoded_model, encoded_ordinal;
    std::vector<float> linear_model, linear_ordinal;
    ASSERT_TRUE(run_case(4, 4, 2, 2, encoded, encoded_model, encoded_ordinal, 0u));
    ASSERT_TRUE(run_case(4, 4, 2, 2, linear, linear_model, linear_ordinal, 1u));
    ASSERT_EQ(encoded_model.size(), linear_model.size());
    ASSERT_EQ(encoded_ordinal.size(), linear_ordinal.size());
    for (std::size_t index = 0; index < encoded_model.size(); ++index) {
      EXPECT_NEAR(encoded_model[index], linear_model[index], 6e-6f);
    }
    for (std::size_t index = 0; index < encoded_ordinal.size(); ++index) {
      EXPECT_NEAR(encoded_ordinal[index], linear_ordinal[index], 4e-6f);
    }
  }

  // A single HDR highlight must be tone-mapped before it contributes to the area average. If the
  // order is reversed, the 5x5 footprint becomes almost white instead of retaining 24 black cells.
  {
    std::vector<rgba_pixel_t> source_pixels(
      25,
      rgba_pixel_t {0.0f, 0.0f, 0.0f, 1.0f}
    );
    source_pixels[12] = {100.0f, 100.0f, 100.0f, 1.0f};
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    ASSERT_TRUE(run_case(5, 5, 1, 1, source_pixels, model_output, appearance_ordinal, 2u));
    const auto mapped_highlight = depth_color_reference(source_pixels[12], 2u);
    for (UINT channel = 0; channel < 3u; ++channel) {
      const float actual = reconstructed_channel(model_output, 1u, 0u, channel);
      EXPECT_NEAR(actual, mapped_highlight[channel] / 25.0f, 6e-6f);
      EXPECT_LT(actual, 0.05f);
    }
    ASSERT_EQ(appearance_ordinal.size(), 1u);
    EXPECT_FLOAT_EQ(appearance_ordinal[0], 100.0f);
  }

  // HDR capture is linear scRGB. The model input must use luminance-preserving Reinhard,
  // optional uniform peak normalization, and then the same sRGB OETF. HDR appearance evidence
  // stays in scene-linear scRGB and therefore deliberately retains values above one.
  {
    const std::vector<rgba_pixel_t> source_pixels {
      rgba_pixel_t {0.6f, 0.3f, 0.15f, 1.0f},
      rgba_pixel_t {-0.5f, 4.0f, 1.0f, 1.0f},
    };
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    ASSERT_TRUE(run_case(
      2,
      1,
      2,
      1,
      source_pixels,
      model_output,
      appearance_ordinal,
      static_cast<std::uint32_t>(models::input_color_space::scrgb_hdr)
    ));
    for (UINT pixel = 0; pixel < 2u; ++pixel) {
      const auto expected = depth_color_reference(source_pixels[pixel], 2u);
      for (UINT channel = 0; channel < 3u; ++channel) {
        const float actual = reconstructed_channel(model_output, 2u, pixel, channel);
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected[channel], 4e-6f);
      }
    }
    ASSERT_EQ(appearance_ordinal.size(), 2u);
    EXPECT_FLOAT_EQ(appearance_ordinal[0], 0.6f);
    EXPECT_FLOAT_EQ(appearance_ordinal[1], 4.0f);
  }

  // Downsampling must replicate the resized boundary cell, not sample the outermost source pixel.
  // The first source column deliberately differs from the point/area sample of the first content
  // cell, so this catches source-edge padding that the upsampling case above cannot distinguish.
  {
    std::vector<rgba_pixel_t> source_pixels;
    source_pixels.reserve(32u);
    for (UINT y = 0u; y < 4u; ++y) {
      for (UINT x = 0u; x < 8u; ++x) {
        const float value = static_cast<float>(y * 8u + x) / 32.0f;
        source_pixels.push_back({value, 1.0f - value, value * 0.5f, 1.0f});
      }
    }
    std::vector<float> model_output;
    std::vector<float> appearance_ordinal;
    std::vector<std::uint32_t> exclusion;
    constexpr UINT target_width = 6u;
    constexpr UINT target_height = 4u;
    constexpr models::depth_tensor_content_rect_t content {1u, 1u, 5u, 3u};
    ASSERT_TRUE(run_case(
      8u, 4u, target_width, target_height, source_pixels,
      model_output, appearance_ordinal, 0u,
      content, &exclusion
    ));
    constexpr UINT target_texels = target_width * target_height;
    ASSERT_EQ(model_output.size(), 3u * target_texels);
    ASSERT_EQ(appearance_ordinal.size(), target_texels);
    ASSERT_EQ(exclusion.size(), target_texels);
    for (UINT y = 0u; y < target_height; ++y) {
      for (UINT x = 0u; x < target_width; ++x) {
        const UINT index = y * target_width + x;
        const bool admitted = x >= content.left && x < content.right &&
                              y >= content.top && y < content.bottom;
        EXPECT_EQ(exclusion[index], admitted ? 0u : 1u) << x << ',' << y;
        if (admitted) {
          continue;
        }
        const UINT source_x = std::clamp(x, content.left, content.right - 1u);
        const UINT source_y = std::clamp(y, content.top, content.bottom - 1u);
        const UINT source_index = source_y * target_width + source_x;
        for (UINT channel = 0u; channel < 3u; ++channel) {
          const UINT channel_offset = channel * target_texels;
          EXPECT_EQ(
            std::bit_cast<std::uint32_t>(model_output[channel_offset + index]),
            std::bit_cast<std::uint32_t>(model_output[channel_offset + source_index])
          ) << x << ',' << y << " channel " << channel;
        }
        EXPECT_EQ(
          std::bit_cast<std::uint32_t>(appearance_ordinal[index]),
          std::bit_cast<std::uint32_t>(appearance_ordinal[source_index])
        ) << x << ',' << y;
      }
    }
  }
}
#endif

TEST(DirectxShaderSourceTest, ConvertsEveryChromaTapBeforeAveraging) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";

  struct shader_family_t {
    const char *entrypoint;
    const char *converter;
  };

  constexpr std::array families {
    shader_family_t {"convert_yuv420_packed_uv_type0_ps.hlsl", "convert_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0s_ps.hlsl", "convert_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0_ps_linear.hlsl", "convert_linear_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0s_ps_linear.hlsl", "convert_linear_base.hlsl"},
    shader_family_t {
      "convert_yuv420_packed_uv_type0_ps_perceptual_quantizer.hlsl",
      "convert_perceptual_quantizer_base.hlsl"
    },
    shader_family_t {
      "convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer.hlsl",
      "convert_perceptual_quantizer_base.hlsl"
    },
  };

  for (const auto &family : families) {
    const auto path = shader_dir + family.entrypoint;
    std::ifstream entry_input(path, std::ios::binary);
    ASSERT_TRUE(entry_input.is_open()) << path;
    const std::string entry {
      std::istreambuf_iterator<char> {entry_input},
      std::istreambuf_iterator<char> {}
    };

    const auto converter =
      entry.find(std::string {"#include \"include/"} + family.converter + '"');
    const auto packed_uv =
      entry.find("#include \"include/convert_yuv420_packed_uv_ps_base.hlsl\"");
    ASSERT_NE(converter, std::string::npos) << family.entrypoint;
    ASSERT_NE(packed_uv, std::string::npos) << family.entrypoint;
    EXPECT_LT(converter, packed_uv) << family.entrypoint;
  }

  const auto path = shader_dir + "include/convert_yuv420_packed_uv_ps_base.hlsl";
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input.is_open()) << path;

  const std::string shader {
    std::istreambuf_iterator<char> {input},
    std::istreambuf_iterator<char> {}
  };
  const auto sample_begin = shader.find("float3 SampleChromaInput");
  const auto main_begin = shader.find("float2 main_ps", sample_begin);
  ASSERT_NE(sample_begin, std::string::npos);
  ASSERT_NE(main_begin, std::string::npos);

  const auto sample_helper = shader.substr(sample_begin, main_begin - sample_begin);
  EXPECT_NE(
    sample_helper.find(
      "return CONVERT_FUNCTION(image.Sample(def_sampler, tex_coord).rgb);"
    ),
    std::string::npos
  );
  EXPECT_EQ(sample_helper.find("#if"), std::string::npos);

  // All 2/4/6-tap layouts in the shared body must go through the converted sampler. A raw
  // texture fetch or conversion in main_ps would reintroduce post-average conversion.
  const auto filter_body = shader.substr(main_begin);
  EXPECT_EQ(filter_body.find("image.Sample"), std::string::npos);
  EXPECT_EQ(filter_body.find("CONVERT_FUNCTION"), std::string::npos);
  EXPECT_EQ(shader.find("CONVERT_CHROMA_PER_TAP"), std::string::npos);
}

TEST(DirectxShaderSourceTest, HostSbsLatestV2LineageIsNotCurrentRenderAuthorization) {
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  ASSERT_FALSE(display.empty());

  const auto retain_begin = display.find("[[nodiscard]] bool retain_latest_v2_lineage(");
  const auto retain_end = display.find("bool ensure_sbs_intermediate_storage", retain_begin);
  ASSERT_NE(retain_begin, std::string::npos);
  ASSERT_NE(retain_end, std::string::npos);
  const auto retain = display.substr(retain_begin, retain_end - retain_begin);
  EXPECT_NE(
    retain.find("host_sbs_latest_v2_completion_retention_allowed"),
    std::string::npos
  );
  EXPECT_EQ(retain.find("matched_input_reuse_kind"), std::string::npos);
  EXPECT_EQ(retain.find("matched_low_motion_candidate"), std::string::npos);

  const auto frame_begin = display.find("const auto frame_id = ++sbs_frame_sequence");
  const auto post_completion = display.find(
    "// Cached fields alias the estimator's singleton V2 textures",
    frame_begin
  );
  ASSERT_NE(frame_begin, std::string::npos);
  ASSERT_NE(post_completion, std::string::npos);
  const auto selection = display.substr(frame_begin, post_completion - frame_begin);
  EXPECT_NE(selection.find("host_sbs_latest_v2_lineage_reset_required"), std::string::npos);
  EXPECT_NE(selection.find("const bool cached_geometry_matches"), std::string::npos);
  EXPECT_NE(selection.find("host_sbs_cached_geometry_render_allowed"), std::string::npos);

  // Route/dump/reprocess/terminal revocation plus every completion owner keeps the singleton-alias
  // reset matrix in the pre-render selection path.
  std::size_t resets = 0u;
  for (std::size_t offset = 0u;
       (offset = selection.find("latest_v2_lineage.reset();", offset)) != std::string::npos;
       ++offset) {
    ++resets;
  }
  EXPECT_EQ(resets, 7u);
  EXPECT_EQ(display.find("reusable_v2"), std::string::npos);

  const auto fail_flat = display.find("void fail_depth_pipeline_flat()");
  const auto init_output = display.find("int init_output(");
  ASSERT_NE(fail_flat, std::string::npos);
  ASSERT_NE(init_output, std::string::npos);
  EXPECT_NE(
    display.find("latest_v2_lineage.reset();", fail_flat),
    std::string::npos
  );
  EXPECT_NE(
    display.find("latest_v2_lineage.reset();", init_output),
    std::string::npos
  );
}

TEST(DirectxShaderSourceTest, AdaptiveMotionProbeAuthenticatesCutStateEncodings) {
  const auto shader = read_source_file(
    SUNSHINE_SOURCE_DIR
    "/src_assets/windows/assets/shaders/directx/host_sbs_current_frame_motion_probe_cs.hlsl"
  );
  ASSERT_FALSE(shader.empty());
  EXPECT_NE(shader.find("ProbeCanonicalBoolean"), std::string::npos);
  EXPECT_NE(shader.find("ProbeFiniteWholeInRange"), std::string::npos);
  EXPECT_NE(
    shader.find("uint hard_cut_count_value = asuint(SBS_STATE_HARD_CUT_COUNT"),
    std::string::npos
  );
  EXPECT_NE(
    shader.find("prior_flags |= state_fields_valid ? 1u << 9u : 0u;"),
    std::string::npos
  );
}

TEST(DirectxShaderSourceTest, CurrentFrameProbeIsCandidateOwnedAndTelemetryOnly) {
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  ASSERT_FALSE(display.empty());

  const auto build_flag = display.find(
    "const bool adaptive_probe_shadow = adaptive_motion_shadow_enabled();"
  );
  const auto build_capture = display.find("adaptive_probe_shadow]() mutable", build_flag);
  const auto build_argument = display.find("active,\n          adaptive_probe_shadow", build_capture);
  ASSERT_NE(build_flag, std::string::npos);
  ASSERT_NE(build_capture, std::string::npos);
  ASSERT_NE(build_argument, std::string::npos);

  const auto damage = display.find("const auto cached_motion_damage =");
  const auto adaptive_use = display.find("const auto &damage = cached_motion_damage;", damage);
  const auto candidate_record = display.find(
    "const auto audit_record = adaptive_motion_audit.record(",
    adaptive_use
  );
  const auto recorded_gate = display.find("if (audit_record.recorded)", candidate_record);
  const auto baseline_copy = display.find(
    "adaptive_probe_baseline_frame_id =",
    recorded_gate
  );
  const auto probe_plan = display.find(
    "host_sbs_current_frame_probe_plan(",
    baseline_copy
  );
  const auto estimate = display.find("est = depth_estimator->estimate_depth(", probe_plan);
  const auto observer = display.find("record_current_frame_motion_probe(", estimate);
  const auto completion_poll = display.find("host_sbs_same_frame_poll_plan(", observer);
  ASSERT_NE(damage, std::string::npos);
  ASSERT_NE(adaptive_use, std::string::npos);
  ASSERT_NE(candidate_record, std::string::npos);
  ASSERT_NE(recorded_gate, std::string::npos);
  ASSERT_NE(baseline_copy, std::string::npos);
  ASSERT_NE(probe_plan, std::string::npos);
  ASSERT_NE(estimate, std::string::npos);
  ASSERT_NE(observer, std::string::npos);
  ASSERT_NE(completion_poll, std::string::npos);
  EXPECT_LT(candidate_record, recorded_gate);
  EXPECT_LT(recorded_gate, baseline_copy);
  EXPECT_LT(probe_plan, estimate);
  EXPECT_LT(estimate, observer);
  EXPECT_LT(observer, completion_poll);

  const auto lineage_reset = display.find(
    "if (detail::host_sbs_latest_v2_lineage_reset_required(",
    adaptive_use
  );
  ASSERT_NE(lineage_reset, std::string::npos);
  const auto shared_damage_block = display.substr(damage, lineage_reset - damage);
  std::size_t damage_walks = 0u;
  std::size_t damage_walk = 0u;
  while ((damage_walk = shared_damage_block.find(
            "matched_motion_damage(",
            damage_walk
          )) != std::string::npos) {
    ++damage_walks;
    ++damage_walk;
  }
  EXPECT_EQ(damage_walks, 1u);
  EXPECT_NE(
    shared_damage_block.find(
      "current_ddup_damage,\n                adaptive_route_observable"
    ),
    std::string::npos
  );

  const auto observer_definition = display.find(
    "void record_current_frame_motion_probe("
  );
  const auto reset_definition = display.find(
    "void reset_adaptive_motion_runtime(",
    observer_definition
  );
  ASSERT_NE(observer_definition, std::string::npos);
  ASSERT_NE(reset_definition, std::string::npos);
  const auto observer_body = display.substr(
    observer_definition,
    reset_definition - observer_definition
  );
  EXPECT_NE(
    observer_body.find("host_sbs_current_frame_probe_identity_matches"),
    std::string::npos
  );
  EXPECT_EQ(observer_body.find("adaptive_shadow_decision ="), std::string::npos);
  EXPECT_EQ(observer_body.find("adaptive_shadow_cadence"), std::string::npos);
  EXPECT_EQ(observer_body.find("latest_v2_lineage"), std::string::npos);
  EXPECT_EQ(observer_body.find("matched_candidate_slot"), std::string::npos);
}

TEST(DirectxShaderSourceTest, HostSbsRejectsRotationAndUsesMatchedV2Frames) {
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  const auto live_shader = read_source_file(
    SUNSHINE_SOURCE_DIR
    "/src_assets/windows/assets/shaders/directx/sbs_reprojection_v2_live_ps.hlsl"
  );
  ASSERT_FALSE(display.empty());
  ASSERT_FALSE(live_shader.empty());
  EXPECT_NE(
    display.find("display->display_rotation != DXGI_MODE_ROTATION_IDENTITY"),
    std::string::npos
  );
  EXPECT_NE(display.find("find_pending_matched_slot(est.completed_frame_id)"), std::string::npos);
  EXPECT_NE(display.find("parallax_v2_result_is_authenticated(est)"), std::string::npos);
  EXPECT_NE(display.find("v2_live_resources_complete"), std::string::npos);
  EXPECT_EQ(
    live_shader.find("Texture2D<float> CandidateParallax"),
    std::string::npos
  );
  EXPECT_NE(
    live_shader.find("return SourceColor.Sample(LinearSampler, sample_uv);"),
    std::string::npos
  );
}

TEST(DirectxShaderSourceTest, LocalRgbPresentationPreservesTheTransferContract) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  const auto srgb_to_linear =
    read_source_file(shader_dir + "rgb_present_srgb_to_linear_ps.hlsl");

  ASSERT_FALSE(display.empty());
  ASSERT_FALSE(srgb_to_linear.empty());

  // CopyResource is valid only when both the storage layout and the transfer contract match.
  // A BGRA/sRGB source can otherwise look layout-compatible with an FP16 intermediate while the
  // local HDR swapchain interprets its code values as linear light.
  EXPECT_NE(
    display.find("input_is_linear == rgb_present_target_is_linear"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("rgb_present_srgb_to_linear_ps.get()"),
    std::string::npos
  );
  EXPECT_NE(
    srgb_to_linear.find("RemoveSRGBCurve(source.rgb) * source_sdr_white_scrgb"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("sdr_white_nits / 80.0f"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("PSSetConstantBuffers(1, 1, &sdr_white)"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("copy_rgb(final_sbs_texture, final_sbs_is_linear)"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("copy_rgb(img_ctx.encoder_texture.get(), input_is_linear)"),
    std::string::npos
  );
}

TEST(HostSbsSceneCutTest, OrdinalEvidenceRejectsMonotoneExposureButDetectsContent) {
  constexpr int width = 32;
  constexpr int height = 24;
  std::vector<float> original(width * height);
  std::vector<float> exposed(width * height);
  std::vector<float> different(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y * width + x);
      original[index] =
        0.18f + 0.28f * static_cast<float>((x * 3 + y * 5) % 11) / 10.0f;
      exposed[index] = original[index] * 1.35f + 0.05f;
      different[index] =
        0.18f + 0.28f * static_cast<float>((x * 7 + y * 2 + 3) % 11) / 10.0f;
    }
  }

  const auto exposure_evidence =
    structural_evidence(exposed, original, width, height);
  EXPECT_FLOAT_EQ(exposure_evidence.change_fraction, 0.0f);
  EXPECT_GT(exposure_evidence.current_support_fraction, 0.01f);
  EXPECT_GT(exposure_evidence.previous_support_fraction, 0.01f);
  EXPECT_GT(exposure_evidence.common_support_fraction, 0.01f);
  EXPECT_GT(
    structural_change_fraction(different, original, width, height),
    0.02f
  );

  // Opposing sensor noise below the confidence floor must not become two unit directions.
  std::vector<float> low_gradient_a(width * height, 0.50f);
  std::vector<float> low_gradient_b(width * height, 0.50f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y * width + x);
      low_gradient_a[index] += 0.009f * static_cast<float>(x) / width;
      low_gradient_b[index] -= 0.009f * static_cast<float>(x) / width;
    }
  }
  EXPECT_FLOAT_EQ(
    structural_change_fraction(low_gradient_b, low_gradient_a, width, height),
    0.0f
  );

  // A pure gain can move a relation across the absolute reliability floor. Requiring the SAME
  // pair to be reliable in both frames keeps that threshold crossing from becoming structure.
  std::vector<float> low_contrast_ramp(width * height);
  std::vector<float> gained_ramp(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y * width + x);
      low_contrast_ramp[index] = 0.05f + 0.012f * static_cast<float>(x);
      gained_ramp[index] = 0.05f + 2.0f * (low_contrast_ramp[index] - 0.05f);
    }
  }
  EXPECT_FLOAT_EQ(
    structural_change_fraction(gained_ramp, low_contrast_ramp, width, height),
    0.0f
  );

  // All ten cross-five pairings are needed: center-vs-neighbor comparisons alone have no common
  // reliable relation when a horizontal ramp becomes a vertical ramp.
  std::vector<float> horizontal_ramp(width * height);
  std::vector<float> vertical_ramp(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y * width + x);
      horizontal_ramp[index] = 0.10f + 0.021f * static_cast<float>(x);
      vertical_ramp[index] = 0.10f + 0.021f * static_cast<float>(y);
    }
  }
  const auto rotated_ramp_evidence =
    structural_evidence(vertical_ramp, horizontal_ramp, width, height);
  // The 4%-relative reliability floor deliberately abstains on weak pairings, so this is no
  // longer expected to approach 1.0. It must still retain broad common support and remain far
  // above the production structural-cut threshold (0.03).
  EXPECT_GT(rotated_ramp_evidence.change_fraction, 0.40f);
  EXPECT_GT(rotated_ramp_evidence.common_support_fraction, 0.40f);

  // This nonlinear two-dimensional clipped-gain pattern rotates normalized gradient direction
  // even though it is a pure 2x exposure. Monotone ordinal relations can only stay ordered or
  // collapse into an abstaining tie.
  std::vector<float> clipped_pattern(width * height);
  std::vector<float> clipped_gain(width * height);
  constexpr float pi = 3.14159265358979323846f;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y * width + x);
      clipped_pattern[index] = std::clamp(
        0.47f + 0.235f * std::sin(pi * static_cast<float>(x) / 3.0f) +
          0.235f * std::sin(2.0f * pi * static_cast<float>(y) / 5.0f),
        0.0f,
        1.0f
      );
      clipped_gain[index] = std::clamp(2.0f * clipped_pattern[index], 0.0f, 1.0f);
    }
  }
  EXPECT_FLOAT_EQ(
    structural_change_fraction(clipped_gain, clipped_pattern, width, height),
    0.0f
  );

  // A flat endpoint is not evidence that ordinal structure was preserved. It has no reliable
  // current or common comparison sites, even though the structured history remains well supported.
  const std::vector<float> black(width * height, 0.0f);
  const auto black_evidence =
    structural_evidence(black, original, width, height);
  EXPECT_FLOAT_EQ(black_evidence.change_fraction, 0.0f);
  EXPECT_FLOAT_EQ(black_evidence.current_support_fraction, 0.0f);
  EXPECT_GT(black_evidence.previous_support_fraction, 0.01f);
  EXPECT_FLOAT_EQ(black_evidence.common_support_fraction, 0.0f);
}

TEST(DirectxShaderTest, CompilesContractiveDirectParallaxRendererAndDiagnostics) {
  using Microsoft::WRL::ComPtr;

  constexpr std::array shaders {
    std::tuple {"sbs_direct_replay_ps.hlsl", "main_ps", "ps_5_0"},
    std::tuple {"sbs_direct_replay_ps.hlsl", "mapping_ps", "ps_5_0"},
    std::tuple {"sbs_direct_replay_ps.hlsl", "mask_ps", "ps_5_0"},
  };
  for (const auto &[filename, entrypoint, target] : shaders) {
    const std::filesystem::path shader_path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> shader_errors;
    const auto status = D3DCompileFromFile(
      shader_path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entrypoint,
      target,
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0,
      &shader_blob,
      &shader_errors
    );
    EXPECT_TRUE(SUCCEEDED(status))
      << filename << ": "
      << (shader_errors ?
            static_cast<const char *>(shader_errors->GetBufferPointer()) :
            "no compiler diagnostics");
  }
}

TEST(EncodeWaitPolicyTests, ReadyDepthCannotSpinBeforeFirstRealFrame) {
  EXPECT_FALSE(video::detail::should_poll_ready_depth_without_wait(false, false));
  EXPECT_FALSE(video::detail::should_poll_ready_depth_without_wait(false, true));
  EXPECT_FALSE(video::detail::should_poll_ready_depth_without_wait(true, false));
  EXPECT_TRUE(video::detail::should_poll_ready_depth_without_wait(true, true));
}

TEST(EncodeWaitPolicyTests, CadenceTargetIsResolvedBeforeConversionWithoutChangingTimestamp) {
  using namespace std::chrono_literals;
  const auto target = std::chrono::steady_clock::time_point {100ms};

  const auto paced = video::detail::select_encode_frame_schedule(
    target + 1ms,
    target,
    16ms,
    4ms
  );
  EXPECT_EQ(paced.presentation_timestamp, target);
  EXPECT_EQ(paced.next_encode_target, target + 16ms);

  const auto discontinuity = video::detail::select_encode_frame_schedule(
    target + 10ms,
    target,
    16ms,
    4ms
  );
  EXPECT_EQ(discontinuity.presentation_timestamp, target + 10ms);
  EXPECT_EQ(discontinuity.next_encode_target, target + 26ms);

  const auto first_source = std::chrono::steady_clock::time_point {3s};
  const auto first = video::detail::select_encode_frame_schedule(
    first_source,
    {},
    16ms,
    4ms
  );
  EXPECT_EQ(first.presentation_timestamp, first_source);
  EXPECT_EQ(first.next_encode_target, first_source + 16ms);
}

TEST(EncodedFrameBufferPoolTests, RecyclesBoundedReasonableBuffers) {
  video::encoded_frame_buffer_pool_t pool;
  std::vector<std::uint8_t> buffer(1024, 0x5A);
  const auto capacity = buffer.capacity();
  pool.recycle(std::move(buffer));

  auto reused = pool.acquire();
  EXPECT_TRUE(reused.empty());
  EXPECT_GE(reused.capacity(), capacity);
  EXPECT_TRUE(pool.acquire().empty());
}

TEST(EncodedFrameBufferPoolTests, PacketDestructionReturnsStorageToPool) {
  auto pool = std::make_shared<video::encoded_frame_buffer_pool_t>();
  std::vector<std::uint8_t> storage(4096, 0x3C);
  const auto capacity = storage.capacity();
  {
    video::packet_raw_generic packet {std::move(storage), 7, false, pool};
    EXPECT_EQ(packet.data_size(), 4096u);
  }

  auto reused = pool->acquire();
  EXPECT_TRUE(reused.empty());
  EXPECT_GE(reused.capacity(), capacity);
}

TEST(EncodedFrameBufferPoolTests, PoolIsBoundedAndRejectsOversizedCapacity) {
  video::encoded_frame_buffer_pool_t pool;
  for (std::size_t i = 0; i < video::ENCODED_FRAME_BUFFER_POOL_LIMIT + 2; ++i) {
    std::vector<std::uint8_t> storage(256 + i);
    pool.recycle(std::move(storage));
  }

  std::size_t retained = 0;
  for (std::size_t i = 0; i < video::ENCODED_FRAME_BUFFER_POOL_LIMIT + 2; ++i) {
    retained += pool.acquire().capacity() != 0 ? 1u : 0u;
  }
  EXPECT_EQ(retained, video::ENCODED_FRAME_BUFFER_POOL_LIMIT);

  std::vector<std::uint8_t> oversized(
    video::ENCODED_FRAME_BUFFER_RETAIN_LIMIT + 1
  );
  pool.recycle(std::move(oversized));
  EXPECT_EQ(pool.acquire().capacity(), 0u);
}

TEST(EffectiveVideoModePublisherTests, PublishesOnlyCoherentProvenPacingPairs) {
  video::effective_video_mode_publisher_t publisher {{1920, 1080, 6000, 20000}};

  auto pacing = publisher.pacing();
  EXPECT_EQ(pacing.bitrate, 20000);
  EXPECT_EQ(pacing.framerate_millihz, 60000);

  // Constructing a speculative request cannot affect packet pacing. Production performs the same
  // separation by publishing only after make_encode_device() returns a real encoder.
  const video::effective_video_mode_t unproven {3840, 2160, 9000, 50000};
  pacing = publisher.pacing();
  EXPECT_EQ(pacing.bitrate, 20000);
  EXPECT_EQ(pacing.framerate_millihz, 60000);

  publisher.publish(unproven);
  pacing = publisher.pacing();
  EXPECT_EQ(pacing.bitrate, 50000);
  EXPECT_EQ(pacing.framerate_millihz, 90000);
  const auto effective = publisher.current();
  EXPECT_EQ(effective.width, 3840);
  EXPECT_EQ(effective.height, 2160);
  EXPECT_EQ(effective.framerateX100, 9000);
  EXPECT_EQ(effective.bitrate, pacing.bitrate);

  // Fractional cadence is retained exactly as millihertz in the sender snapshot.
  publisher.publish({1920, 1080, 2997, 17000});
  pacing = publisher.pacing();
  EXPECT_EQ(pacing.bitrate, 17000);
  EXPECT_EQ(pacing.framerate_millihz, 29970);
}

TEST(EffectiveVideoModePublisherTests, ProductionPublishesAfterEncoderInitialization) {
  const auto source = read_source_file(SUNSHINE_SOURCE_DIR "/src/video.cpp");
  ASSERT_FALSE(source.empty());

  const auto capture_begin = source.find("void capture_async(");
  const auto hdr_snapshot = source.find(
    "hdr_info_t hdr_info = std::make_unique<hdr_info_raw_t>(false);",
    capture_begin
  );
  const auto hdr_device_read = source.find(
    "colorspace_is_hdr(encode_device->colorspace)",
    hdr_snapshot
  );
  const auto encoder_init = source.find(
    "auto encode_session = make_encode_session(",
    capture_begin
  );
  const auto encode_device_transfer = source.find(
    "std::move(encode_device)",
    encoder_init
  );
  const auto capture_end = source.find(
    "\n  void capture(",
    encode_device_transfer
  );
  const auto post_transfer_device_read = source.find(
    "encode_device->",
    encode_device_transfer
  );
  const auto effective_publish = source.find(
    "config.effective_mode->publish(effective_mode);",
    encoder_init
  );
  const auto completion_publish = source.find(
    "video_mode_applied_queue->raise(video_mode_applied_t",
    effective_publish
  );
  const auto hdr_publish = source.find(
    "hdr_event->raise(std::move(hdr_info));",
    encoder_init
  );
  ASSERT_NE(capture_begin, std::string::npos);
  ASSERT_NE(hdr_snapshot, std::string::npos);
  ASSERT_NE(hdr_device_read, std::string::npos);
  ASSERT_NE(encoder_init, std::string::npos);
  ASSERT_NE(encode_device_transfer, std::string::npos);
  ASSERT_NE(capture_end, std::string::npos);
  ASSERT_NE(effective_publish, std::string::npos);
  ASSERT_NE(completion_publish, std::string::npos);
  ASSERT_NE(hdr_publish, std::string::npos);
  EXPECT_LT(hdr_snapshot, encoder_init);
  EXPECT_LT(hdr_device_read, encoder_init);
  EXPECT_TRUE(
    post_transfer_device_read == std::string::npos ||
    post_transfer_device_read >= capture_end
  );
  EXPECT_LT(encoder_init, effective_publish);
  EXPECT_LT(effective_publish, completion_publish);
  EXPECT_LT(encoder_init, hdr_publish);
}

TEST(HostSbsSceneCutTest, AppearanceFusionRejectsExposureAndHonorsExactBounds) {
  // The exposure fixture clears the broad RGB gate but not the ordinal gate. The
  // exposure-like relation also vetoes a coincident neural-depth jump on the geometry arm.
  shot_cut_state_t exposure;
  exposure.cut_flags = cut_flags_ready;
  exposure.depth_change_baseline = 0.60f;
  EXPECT_FALSE(advance_shot_cut(exposure, 0.60f, 0.80f, 0.009f));

  // Detailed motion can clear the ordinal cue, but not frame-wide RGB replacement.
  shot_cut_state_t detailed_motion;
  detailed_motion.cut_flags = cut_flag_latched | cut_flag_appearance_armed;
  detailed_motion.depth_change_baseline = 0.30f;
  EXPECT_FALSE(advance_shot_cut(detailed_motion, 0.30f, 0.13f, 0.08f));

  shot_cut_state_t below_raw;
  below_raw.appearance_change_baseline = 0.58f;
  EXPECT_FALSE(advance_shot_cut(below_raw, 0.25f, 0.699f, 0.03f));
  shot_cut_state_t below_ordinal;
  EXPECT_FALSE(advance_shot_cut(below_ordinal, 0.25f, 0.70f, 0.029f));
  shot_cut_state_t below_corroboration;
  EXPECT_FALSE(advance_shot_cut(below_corroboration, 0.249f, 0.70f, 0.03f));

  // Exact appearance bounds remain authoritative for a semantic transition whose comparison
  // support does not describe one preserved exposure relation.
  shot_cut_state_t exact_appearance;
  EXPECT_TRUE(advance_shot_cut(
    exact_appearance,
    0.25f,
    0.70f,
    0.03f,
    1.0f,
    1.0f,
    0.40f
  ));
  shot_cut_state_t exact_geometry;
  EXPECT_FALSE(advance_shot_cut(exact_geometry, 0.60f, 0.0f, 0.005f));
  EXPECT_NE(
    exact_geometry.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_FLOAT_EQ(exact_geometry.model_input_history_state, 4.0f);
  EXPECT_TRUE(advance_shot_cut(exact_geometry, 0.60f, 0.0f, 0.005f));
}

TEST(HostSbsSceneCutTest, LocalizedAppearanceSurpriseRemainsImmediate) {
  shot_cut_state_t localized_cut;
  localized_cut.appearance_change_baseline = 0.02f;

  EXPECT_TRUE(advance_shot_cut(
    localized_cut,
    0.3864f,
    0.2019f,
    0.0396f,
    0.50f,
    0.50f,
    0.50f
  ));
  EXPECT_EQ(localized_cut.cut_flags, cut_flag_latched);

  shot_cut_state_t ordinary_local_motion;
  ordinary_local_motion.appearance_change_baseline = 0.09f;
  EXPECT_FALSE(advance_shot_cut(
    ordinary_local_motion,
    0.3864f,
    0.2019f,
    0.0396f,
    0.50f,
    0.50f,
    0.50f
  ));
  EXPECT_EQ(
    ordinary_local_motion.cut_flags &
      cut_flag_geometry_confirmation_pending,
    0u
  );
}

TEST(HostSbsSceneCutTest, GeometryOnlyCutRequiresHeldEndpointConfirmation) {
  shot_cut_state_t geometry_cut;
  EXPECT_FALSE(advance_shot_cut(geometry_cut, 0.6864f, 0.02f, 0.029f));
  EXPECT_NE(
    geometry_cut.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_FLOAT_EQ(geometry_cut.model_input_history_state, 4.0f);
  EXPECT_FLOAT_EQ(geometry_cut.depth_change_baseline, 0.0f);

  EXPECT_TRUE(advance_shot_cut(geometry_cut, 0.6875f, 0.02f, 0.028f));
  EXPECT_EQ(geometry_cut.cut_flags, cut_flag_latched);
  EXPECT_FLOAT_EQ(geometry_cut.scene_age, 0.0f);
}

TEST(HostSbsSceneCutTest, TwoFrameDepthSpikeWithQuietStructureDoesNotCut) {
  shot_cut_state_t normalization_motion;

  // Measured video_sidebar_ad f35/f36 evidence: normalization/motion changes most depth texels
  // for two updates while exposure-invariant ordinal structure remains quiet. Neither update may
  // start the held-endpoint confirmation, much less pulse a cut.
  EXPECT_FALSE(advance_shot_cut(
    normalization_motion,
    0.771f,
    0.0147f,
    0.002f
  ));
  EXPECT_EQ(
    normalization_motion.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_FLOAT_EQ(normalization_motion.model_input_history_state, 1.0f);

  EXPECT_FALSE(advance_shot_cut(
    normalization_motion,
    0.674f,
    0.0147f,
    0.002f
  ));
  EXPECT_EQ(
    normalization_motion.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_FLOAT_EQ(normalization_motion.model_input_history_state, 1.0f);
  EXPECT_NE(
    normalization_motion.cut_flags & cut_flag_latched,
    cut_flag_latched
  );
}

TEST(HostSbsSceneCutTest, ExposureLikeTransitionVetoesAbsoluteAndRelativeDepthAuthority) {
  shot_cut_state_t absolute;
  absolute.cut_flags = cut_flags_ready;
  EXPECT_FALSE(advance_shot_cut(absolute, 0.95f, 0.70f, 0.009f));
  EXPECT_EQ(
    absolute.cut_flags,
    cut_flags_ready | cut_flag_appearance_recovery
  );

  shot_cut_state_t relative;
  relative.cut_flags = cut_flag_latched;
  relative.scene_age = 100.0f;
  relative.depth_change_baseline = 0.20f;
  EXPECT_FALSE(advance_shot_cut(relative, 0.95f, 0.70f, 0.0f));
  EXPECT_EQ(relative.cut_flags & cut_flag_latched, cut_flag_latched);

  // The veto describes broad appearance replacement, not a blanket reduction in geometry
  // sensitivity. A depth-authoritative change with no frame-wide RGB transition still cuts
  // after it survives the one-update held-endpoint confirmation.
  shot_cut_state_t geometry_only;
  EXPECT_FALSE(advance_shot_cut(geometry_only, 0.60f, 0.699f, 0.005f));
  EXPECT_TRUE(advance_shot_cut(geometry_only, 0.60f, 0.699f, 0.005f));

  // At the exact quiet boundary, a structural change that is not small relative to the raw
  // replacement remains ambiguous and preserves standalone geometry authority.
  shot_cut_state_t ambiguous_structure;
  EXPECT_FALSE(advance_shot_cut(ambiguous_structure, 0.60f, 0.19f, 0.01f));
  EXPECT_TRUE(advance_shot_cut(ambiguous_structure, 0.60f, 0.19f, 0.01f));

  // A semantic cut with little common structural support is never exposure-like, even if its
  // aggregate flip fraction is small relative to a broad RGB replacement.
  shot_cut_state_t structural_cut;
  EXPECT_TRUE(advance_shot_cut(
    structural_cut,
    0.60f,
    0.70f,
    0.03f,
    0.80f,
    0.80f,
    0.20f
  ));

  // A frame-wide gain over a moving video is still exposure: raw replacement dominates the
  // unrelated ordinal flips, and reliable comparison support remains present in both frames.
  shot_cut_state_t moving_video_exposure;
  EXPECT_FALSE(advance_shot_cut(
    moving_video_exposure,
    0.90f,
    0.789f,
    0.036f,
    0.566f,
    0.495f,
    0.459f,
    0.750f,
    0.039f
  ));
  EXPECT_NE(
    moving_video_exposure.cut_flags & cut_flag_appearance_recovery,
    0u
  );

  // The inverse exposure direction is equally valid: a global return/dimming event has one
  // dominant sign even while unrelated playing-video motion produces ordinal flips.
  shot_cut_state_t moving_video_exposure_return;
  EXPECT_FALSE(advance_shot_cut(
    moving_video_exposure_return,
    0.90f,
    0.789f,
    0.036f,
    0.495f,
    0.566f,
    0.459f,
    0.039f,
    0.750f
  ));

  // The same 4% ordinal/raw ratio is not exposure when a semantic replacement contains balanced
  // brightening and darkening. The depth-authoritative cut must remain visible.
  shot_cut_state_t mixed_sign_semantic_cut;
  EXPECT_TRUE(advance_shot_cut(
    mixed_sign_semantic_cut,
    0.60f,
    0.80f,
    0.032f,
    0.80f,
    0.80f,
    0.75f,
    0.40f,
    0.40f
  ));

  // Ordinary motion never enters the ratio path without the upstream frame-wide RGB gate.
  shot_cut_state_t ordinary_motion;
  EXPECT_FALSE(advance_shot_cut(
    ordinary_motion,
    0.60f,
    0.17f,
    0.03f,
    0.80f,
    0.80f,
    0.75f
  ));
  EXPECT_TRUE(advance_shot_cut(
    ordinary_motion,
    0.60f,
    0.17f,
    0.03f,
    0.80f,
    0.80f,
    0.75f
  ));

  // Even with broad RGB replacement, structural change above the conservative 5% ratio remains
  // eligible for depth-authoritative cut detection.
  shot_cut_state_t broad_content_cut;
  EXPECT_TRUE(advance_shot_cut(
    broad_content_cut,
    0.60f,
    0.80f,
    0.041f,
    0.80f,
    0.80f,
    0.75f
  ));
}

TEST(HostSbsSceneCutTest, ExposureRecoveryBlocksOnlyTheDelayedGeometryUpdate) {
  shot_cut_state_t state;

  // These are the retained SDR smoke measurements. A->flash and flash->A
  // preserve more than 92% of the smaller endpoint's structural support, so
  // both are exposure-like and refresh the bounded recovery guard.
  EXPECT_FALSE(advance_shot_cut(
    state,
    0.810272932f,
    1.0f,
    0.000182216f,
    0.123987697f,
    0.169116467f,
    0.115342572f
  ));
  EXPECT_NE(state.cut_flags & cut_flag_appearance_recovery, 0u);
  EXPECT_FALSE(advance_shot_cut(
    state,
    0.999979794f,
    1.0f,
    0.000182216f,
    0.169116467f,
    0.123987697f,
    0.115342572f
  ));
  EXPECT_NE(state.cut_flags & cut_flag_appearance_recovery, 0u);

  // Neural-depth normalization can still jump on the first visually quiet
  // update. Consume exactly that delayed geometry-only spike.
  const auto baseline_before_recovery = state.depth_change_baseline;
  EXPECT_FALSE(advance_shot_cut(
    state,
    0.776198566f,
    0.0f,
    0.0f,
    0.169116467f,
    0.169116467f,
    0.169116467f
  ));
  EXPECT_EQ(state.cut_flags & cut_flag_appearance_recovery, 0u);
  EXPECT_FLOAT_EQ(state.depth_change_baseline, baseline_before_recovery);

  // The real A->B cut has individually structured endpoints but only 13%
  // representative overlap. It is not an exposure relation even though its
  // aggregate ordinal-flip fraction is below the quiet threshold.
  EXPECT_FALSE(advance_shot_cut(
    state,
    0.827988386f,
    0.998319566f,
    0.008665371f,
    0.160147399f,
    0.169116467f,
    0.020610625f
  ));
  EXPECT_TRUE(advance_shot_cut(
    state,
    0.827988386f,
    0.998319566f,
    0.008665371f,
    0.160147399f,
    0.169116467f,
    0.020610625f
  ));

  // A genuinely new broad appearance proposal is never delayed by the recovery guard.
  shot_cut_state_t immediate_cut;
  EXPECT_FALSE(advance_shot_cut(immediate_cut, 0.83f, 0.99f, 0.005f));
  EXPECT_TRUE(advance_shot_cut(
    immediate_cut,
    0.60f,
    0.99f,
    0.03f,
    0.16f,
    0.17f,
    0.02f,
    0.50f,
    0.49f
  ));

  // Recovery is one following valid update, not one quiet update. Normal playback motion can
  // resume before neural depth settles; consume that delayed spike, clear the guard, and leave
  // the next independently authoritative geometry update visible.
  shot_cut_state_t moving_recovery;
  EXPECT_FALSE(advance_shot_cut(
    moving_recovery,
    0.83f,
    0.99f,
    0.005f
  ));
  const auto moving_baseline_before_recovery =
    moving_recovery.depth_change_baseline;
  EXPECT_FALSE(advance_shot_cut(
    moving_recovery,
    0.75f,
    0.13f,
    0.034f,
    0.60f,
    0.60f,
    0.55f
  ));
  EXPECT_EQ(
    moving_recovery.cut_flags & cut_flag_appearance_recovery,
    0u
  );
  EXPECT_FLOAT_EQ(
    moving_recovery.depth_change_baseline,
    moving_baseline_before_recovery
  );
  EXPECT_FALSE(advance_shot_cut(
    moving_recovery,
    0.60f,
    0.13f,
    0.034f,
    0.60f,
    0.60f,
    0.55f
  ));
  EXPECT_TRUE(advance_shot_cut(
    moving_recovery,
    0.60f,
    0.13f,
    0.034f,
    0.60f,
    0.60f,
    0.55f
  ));
}

TEST(HostSbsSceneCutTest, StructurelessGapBridgesSaturatedFlashAndFindsDifferentReturn) {
  // Entering a fully clipped frame has no usable current/common ordinal support. Suppress its
  // neural-depth jump and hold the last reliable appearance/depth history instead of treating
  // abstention as proof of preserved exposure structure.
  shot_cut_state_t flash;
  EXPECT_FALSE(advance_shot_cut(
    flash,
    0.95f,
    1.0f,
    0.0f,
    0.0f,
    0.90f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(flash.model_input_history_state, 2.0f);
  EXPECT_FLOAT_EQ(flash.depth_change_baseline, 0.0f);

  // Appearance and depth history remain paired to A. The exact appearance endpoint proves this
  // is the same shot, so consume the return without a cut and resume normal history.
  EXPECT_FALSE(advance_shot_cut(
    flash,
    0.05f,
    0.0f,
    0.0f,
    0.90f,
    0.90f,
    0.90f
  ));
  EXPECT_FLOAT_EQ(flash.model_input_history_state, 1.0f);

  // Raw RGB cannot qualify support loss: a flat value can match the dominant color of the
  // preceding scene. Reliable-to-structureless transitions are therefore held even at negligible
  // raw delta; the next supported frame is the authoritative endpoint comparison.
  shot_cut_state_t white_flash;
  EXPECT_FALSE(advance_shot_cut(
    white_flash,
    0.95f,
    0.085f,
    0.0f,
    0.0f,
    0.90f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(white_flash.model_input_history_state, 2.0f);

  // A second consecutive low-structure update is persistent content, not a one-frame flash.
  // Release the veto, compare against the held A depth, and advance history after the decision.
  shot_cut_state_t persistent_flat;
  EXPECT_FALSE(advance_shot_cut(
    persistent_flat,
    0.95f,
    1.0f,
    0.0f,
    0.0f,
    0.90f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(persistent_flat.model_input_history_state, 2.0f);
  EXPECT_TRUE(advance_shot_cut(
    persistent_flat,
    0.60f,
    1.0f,
    0.0f,
    0.0f,
    0.90f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(persistent_flat.model_input_history_state, 3.0f);
  EXPECT_EQ(persistent_flat.cut_flags, cut_flag_latched);
  // The accepted flat endpoint is now the reference, so persistence cannot pulse periodically.
  EXPECT_FALSE(advance_shot_cut(
    persistent_flat,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(persistent_flat.model_input_history_state, 3.0f);
  EXPECT_NE(persistent_flat.cut_flags & cut_flag_latched, 0u);
  // The first supported return starts a held comparison even though the ordinary post-cut
  // geometry arm is still refractory. A second update at the same structured endpoint confirms
  // that this is not another one-frame neural transient.
  EXPECT_FALSE(advance_shot_cut(
    persistent_flat,
    0.60f,
    0.01f,
    // A uniform slate reference yields exactly zero structural change (every ordinal pair
    // has zero contrast); the return cut is reachable via the structureless-reference waiver.
    0.0f,
    0.90f,
    0.0f,
    0.0f
  ));
  EXPECT_NE(
    persistent_flat.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_TRUE(advance_shot_cut(
    persistent_flat,
    0.60f,
    0.01f,
    // A uniform slate reference yields exactly zero structural change (every ordinal pair
    // has zero contrast); the return cut is reachable via the structureless-reference waiver.
    0.0f,
    0.90f,
    0.0f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(persistent_flat.model_input_history_state, 1.0f);
  EXPECT_EQ(persistent_flat.cut_flags, cut_flag_latched);

  // A changed structured return must retain standalone A-vs-B geometry authority even when both
  // appearance signals are quiet. The old "< broad RGB" return veto suppressed this exact case.
  shot_cut_state_t changed_return;
  changed_return.model_input_history_state = 2.0f;
  EXPECT_FALSE(advance_shot_cut(
    changed_return,
    0.60f,
    0.01f,
    0.005f,
    0.90f,
    0.90f,
    0.90f
  ));
  EXPECT_TRUE(advance_shot_cut(
    changed_return,
    0.60f,
    0.01f,
    0.005f,
    0.90f,
    0.90f,
    0.90f
  ));

  shot_cut_state_t slate;
  EXPECT_FALSE(advance_shot_cut(
    slate,
    0.95f,
    1.0f,
    0.0f,
    0.0f,
    0.90f,
    0.0f
  ));
  EXPECT_FLOAT_EQ(slate.model_input_history_state, 2.0f);

  // A different structured endpoint is compared with the preserved pre-slate shot. Broad RGB,
  // changed ordinal structure, and depth corroboration accept exactly that visible return.
  EXPECT_TRUE(advance_shot_cut(
    slate,
    0.60f,
    0.80f,
    0.10f,
    0.90f,
    0.90f,
    0.40f
  ));
  EXPECT_FLOAT_EQ(slate.model_input_history_state, 1.0f);
}

TEST(HostSbsSceneCutTest, ExposureClassificationRequiresExactStructuralSupportFloor) {
  shot_cut_state_t below_support;
  EXPECT_FALSE(advance_shot_cut(
    below_support,
    0.60f,
    0.70f,
    0.005f,
    0.90f,
    0.90f,
    0.009f
  ));
  EXPECT_TRUE(advance_shot_cut(
    below_support,
    0.60f,
    0.70f,
    0.005f,
    0.90f,
    0.90f,
    0.009f
  ));

  shot_cut_state_t exact_support;
  EXPECT_FALSE(advance_shot_cut(
    exact_support,
    0.95f,
    0.70f,
    0.005f,
    0.01f,
    0.01f,
    0.01f
  ));
}

TEST(HostSbsSceneCutTest, ExposureClassificationRequiresRepresentativeCommonSupport) {
  shot_cut_state_t below_ratio;
  EXPECT_FALSE(advance_shot_cut(
    below_ratio,
    0.60f,
    0.70f,
    0.005f,
    0.90f,
    0.90f,
    0.449f
  ));
  EXPECT_TRUE(advance_shot_cut(
    below_ratio,
    0.60f,
    0.70f,
    0.005f,
    0.90f,
    0.90f,
    0.449f
  ));

  shot_cut_state_t exact_ratio;
  EXPECT_FALSE(advance_shot_cut(
    exact_ratio,
    0.95f,
    0.70f,
    0.005f,
    0.90f,
    0.90f,
    0.45f
  ));
}

TEST(HostSbsSceneCutTest, StartupArmsOnSettledUpdateAndFiresOnlyAfterward) {
  shot_cut_state_t startup;
  startup.cut_flags = 0u;
  startup.scene_age = 7.0f;
  startup.depth_change_baseline = 0.60f;

  EXPECT_FALSE(advance_shot_cut(startup, 0.60f, 0.95f, 0.10f));
  EXPECT_EQ(startup.cut_flags, cut_flags_ready);
  EXPECT_TRUE(advance_shot_cut(startup, 0.60f, 0.95f, 0.10f));
}

TEST(HostSbsSceneCutTest, SettleAgeUsesSkippedSourceStreamFrames) {
  shot_cut_state_t startup;
  startup.cut_flags = 0u;
  startup.scene_age = 0.0f;
  startup.depth_change_baseline = 0.0f;

  EXPECT_FALSE(advance_shot_cut(
    startup,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    8u
  ));
  EXPECT_FLOAT_EQ(startup.scene_age, 8.0f);
  EXPECT_EQ(startup.cut_flags, cut_flags_ready);
}

TEST(HostSbsSceneCutTest, PersistentEvidencePulsesOnceAndIndependentArmsRecover) {
  shot_cut_state_t state;
  EXPECT_TRUE(advance_shot_cut(state, 0.60f, 0.80f, 0.05f));
  EXPECT_EQ(state.cut_flags, cut_flag_latched);

  // Steady shot-level evidence never rearms or periodically resets scene state.
  for (int update = 0; update < 12; ++update) {
    EXPECT_FALSE(advance_shot_cut(state, 0.60f, 0.80f, 0.05f)) << update;
    EXPECT_EQ(state.cut_flags, cut_flag_latched) << update;
  }

  // The measured fast_motion-like evidence is quiet for the qualified appearance proposal and
  // low for geometry. Each branch independently rearms after its second qualifying update.
  EXPECT_FALSE(advance_shot_cut(state, 0.07f, 0.05f, 0.04f));
  EXPECT_EQ(
    state.cut_flags,
    cut_flag_latched | cut_flag_geometry_low_once | cut_flag_appearance_quiet_once
  );
  EXPECT_FALSE(advance_shot_cut(state, 0.07f, 0.05f, 0.04f));
  EXPECT_EQ(
    state.cut_flags,
    cut_flag_latched | cut_flag_geometry_armed | cut_flag_appearance_armed
  );
  EXPECT_NE(state.cut_flags & cut_flag_latched, 0u)
    << "independent proposal rearm must retain the permanent post-cut phase marker";

  // Arming affects the following update, so the second quiet update did not pulse.
  EXPECT_TRUE(advance_shot_cut(state, 0.60f, 0.80f, 0.05f));
}

TEST(HostSbsSceneCutTest, RelativeGeometrySpikeEscapesWithoutPeriodicCooldown) {
  shot_cut_state_t state;
  state.cut_flags = cut_flag_latched;
  state.depth_change_baseline = 0.35f;
  state.scene_age = 100.0f;

  EXPECT_FALSE(advance_shot_cut(state, 0.62f, 0.05f, 0.04f));
  EXPECT_NE(
    state.cut_flags & cut_flag_geometry_confirmation_pending,
    0u
  );
  EXPECT_FLOAT_EQ(state.depth_change_baseline, 0.35f);
  EXPECT_TRUE(advance_shot_cut(state, 0.62f, 0.05f, 0.04f));
  EXPECT_FLOAT_EQ(state.depth_change_baseline, 0.62f);

  // Resetting the baseline to the accepted spike prevents the new steady level from retriggering.
  for (int update = 0; update < 20; ++update) {
    EXPECT_FALSE(advance_shot_cut(state, 0.62f, 0.05f, 0.04f)) << update;
  }
  EXPECT_NE(state.cut_flags & cut_flag_appearance_armed, 0u);
  EXPECT_EQ(state.cut_flags & cut_flag_geometry_armed, 0u);

  shot_cut_state_t below_margin;
  below_margin.cut_flags = cut_flag_latched;
  below_margin.depth_change_baseline = 0.35f;
  EXPECT_FALSE(advance_shot_cut(below_margin, 0.549f, 0.0f, 0.0f));

  shot_cut_state_t exact_margin;
  exact_margin.cut_flags = cut_flag_latched;
  exact_margin.depth_change_baseline = 0.35f;
  EXPECT_FALSE(advance_shot_cut(exact_margin, 0.55f, 0.0f, 0.005f));
  EXPECT_TRUE(advance_shot_cut(exact_margin, 0.55f, 0.0f, 0.005f));

  shot_cut_state_t exact_multiplier;
  exact_multiplier.cut_flags = cut_flag_latched;
  exact_multiplier.depth_change_baseline = 0.16f;
  EXPECT_FALSE(advance_shot_cut(exact_multiplier, 0.32f, 0.0f, 0.005f));
  EXPECT_TRUE(advance_shot_cut(exact_multiplier, 0.32f, 0.0f, 0.005f));
}

TEST(HostSbsSceneCutTest, RelativeGeometryIgnoresPostCutNormalizationSettling) {
  shot_cut_state_t state;

  // An appearance-authoritative semantic cut can arrive at the 0.25 corroboration floor and
  // exact ordinal threshold when common support does not qualify it as preserved exposure.
  EXPECT_TRUE(advance_shot_cut(
    state,
    0.25f,
    0.70f,
    0.03f,
    1.0f,
    1.0f,
    0.40f
  ));
  EXPECT_FLOAT_EQ(state.scene_age, 0.0f);
  EXPECT_FLOAT_EQ(state.depth_change_baseline, 0.25f);

  // The next normalized field may jump by more than both relative thresholds. The settling
  // refractory blocks it, and the EMA catches up so steady settling evidence cannot pulse later.
  for (int update = 1; update <= 12; ++update) {
    EXPECT_FALSE(advance_shot_cut(state, 0.55f, 0.0f, 0.0f)) << update;
  }
}

TEST(HostSbsSceneCutTest, GeometryLowRearmIsStrictAndConsecutive) {
  shot_cut_state_t state;
  state.cut_flags = cut_flag_latched | cut_flag_appearance_armed;
  state.depth_change_baseline = 0.10f;

  EXPECT_FALSE(advance_shot_cut(state, 0.10f, 0.0f, 0.0f));
  EXPECT_EQ(state.cut_flags & cut_flag_geometry_low_once, 0u);
  EXPECT_FALSE(advance_shot_cut(state, 0.099f, 0.0f, 0.0f));
  EXPECT_NE(state.cut_flags & cut_flag_geometry_low_once, 0u);
  EXPECT_FALSE(advance_shot_cut(state, 0.10f, 0.0f, 0.0f));
  EXPECT_EQ(state.cut_flags & cut_flag_geometry_low_once, 0u);
  EXPECT_FALSE(advance_shot_cut(state, 0.099f, 0.0f, 0.0f));
  EXPECT_FALSE(advance_shot_cut(state, 0.099f, 0.0f, 0.0f));
  EXPECT_NE(state.cut_flags & cut_flag_geometry_armed, 0u);
}

TEST(DirectxShaderSourceTest, FusedDepthMapperUsesResolvedReferenceGridGradients) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto mapper = read_source_file(shader_dir + "buffer_to_tex_cs.hlsl");
  const auto constants = read_source_file(shader_dir + "include/depth_constants.hlsl");

  ASSERT_FALSE(mapper.empty());
  ASSERT_FALSE(constants.empty());
  EXPECT_NE(
    constants.find("#define DEPTH_GRADIENT_REFERENCE_SHORT_SIDE 434.0f"),
    std::string::npos
  );
  EXPECT_NE(
    constants.find(
      "return (float)min(content.z - content.x, content.w - content.y) /"
    ),
    std::string::npos
  );
  EXPECT_NE(
    mapper.find("reference_gradient = gradient * DepthReferenceTexelScale()"),
    std::string::npos
  );
  EXPECT_NE(
    mapper.find("reference_gradient >= ema_edge_gradient"),
    std::string::npos
  );
  EXPECT_NE(
    mapper.find("RWTexture2D<uint>         MotionMask : register(u1)"),
    std::string::npos
  );
  EXPECT_NE(
    mapper.find("OutputTexture[DTid.xy] = moving ?"),
    std::string::npos
  );
}

TEST(DirectxShaderSourceTest, FusedDepthMapperSpecializesSyntheticPadding) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto mapper = read_source_file(shader_dir + "buffer_to_tex_cs.hlsl");

  ASSERT_FALSE(mapper.empty());
  EXPECT_NE(mapper.find("void pad_main"), std::string::npos);
  EXPECT_NE(
    mapper.find("OutputTexture[DepthAnalysisClampCell(DTid.xy)]"),
    std::string::npos
  );
  EXPECT_NE(mapper.find("MotionMask[DTid.xy] = 0u"), std::string::npos);
}

TEST(DirectxShaderSourceTest, HostTelemetryReadbackIsNonblockingAndCutBridgeContractIsHonest) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto resolve = read_source_file(shader_dir + "depth_scene_cut_resolve_cs.hlsl");
  const auto estimator =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp");
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");

  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(display.empty());

  // The transport stays a fixed 32-word CutBridgeState; renderer geometry is carried by the
  // separate V2 state contract. Compact evaluator projection is covered by its generated-contract
  // tests and should not be pinned here to one harness implementation spelling.
  EXPECT_EQ(sbs_adaptive_state::word_count, 32u);
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::word_count"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::initial_words"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::known_cut_flag_mask"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::known_analysis_flag_mask"),
    std::string::npos
  );
  for (const auto *honest_vector : {
         "CutBridgeState[SBS_STATE_VECTOR_SCENE_AGE]",
         "CutBridgeState[SBS_STATE_VECTOR_DEPTH_CHANGE_BASELINE_EMA]",
         "CutBridgeState[SBS_STATE_VECTOR_CUT_FLAGS]",
         "CutBridgeState[SBS_STATE_VECTOR_CURRENT_DEPTH_CHANGE_FRACTION]",
         "CutBridgeState[SBS_STATE_VECTOR_STRUCTURAL_CHANGE_FRACTION]",
       }) {
    EXPECT_NE(resolve.find(honest_vector), std::string::npos);
  }

  // Scope negative assertions to the telemetry implementation. The same source also contains a
  // deliberately bounded teardown drain for GPU timers, which is allowed to flush/sleep after the
  // live encode path has ended.
  const auto readback_begin = estimator.find("bool ensure_telemetry_readback()");
  const auto readback_end = estimator.find("void perf_try_resolve", readback_begin);
  ASSERT_NE(readback_begin, std::string::npos);
  ASSERT_NE(readback_end, std::string::npos);
  const auto readback = estimator.substr(readback_begin, readback_end - readback_begin);

  EXPECT_NE(
    estimator.find("std::array<telemetry_readback_slot, 3> telemetry_readback_slots"),
    std::string::npos
  );
  EXPECT_NE(readback.find("D3D11_USAGE_STAGING"), std::string::npos);
  EXPECT_NE(readback.find("D3D11_CPU_ACCESS_READ"), std::string::npos);
  EXPECT_NE(readback.find("D3D11_QUERY_EVENT"), std::string::npos);
  const auto readiness_query =
    readback.find("D3D11_ASYNC_GETDATA_DONOTFLUSH");
  const auto nonblocking_map =
    readback.find("D3D11_MAP_FLAG_DO_NOT_WAIT");
  ASSERT_NE(readiness_query, std::string::npos);
  ASSERT_NE(nonblocking_map, std::string::npos);
  EXPECT_LT(readiness_query, nonblocking_map);
  EXPECT_NE(readback.find("DXGI_ERROR_WAS_STILL_DRAWING"), std::string::npos);
  EXPECT_NE(readback.find("if (slot.pending)"), std::string::npos);
  EXPECT_NE(readback.find("result.copy_scheduled = true"), std::string::npos);
  EXPECT_EQ(readback.find("Flush("), std::string::npos);
  EXPECT_EQ(readback.find("sleep_for"), std::string::npos);
  EXPECT_EQ(readback.find("while ("), std::string::npos);

  // External copies remain subscription-gated; the separate adaptive shadow may request its own
  // exact-frame evidence. Either demand remains ordered after production output.
  const auto external_due_declaration = display.find("const bool external_due =");
  const auto enabled_gate = display.find("enabled &&", external_due_declaration);
  const auto adaptive_due_declaration = display.find("const bool adaptive_due =", enabled_gate);
  const auto adaptive_requested_gate = display.find(
    "adaptive_requested &&",
    adaptive_due_declaration
  );
  const auto due_declaration = display.find(
    "const bool due = external_due || adaptive_due",
    adaptive_due_declaration
  );
  const auto poll_call =
    display.find("depth_estimator->poll_depth_telemetry(", due_declaration);
  const auto due_argument = display.find("due && producer_active", poll_call);
  ASSERT_NE(external_due_declaration, std::string::npos);
  ASSERT_NE(enabled_gate, std::string::npos);
  ASSERT_NE(adaptive_due_declaration, std::string::npos);
  ASSERT_NE(adaptive_requested_gate, std::string::npos);
  ASSERT_NE(due_declaration, std::string::npos);
  ASSERT_NE(poll_call, std::string::npos);
  ASSERT_NE(due_argument, std::string::npos);
  EXPECT_LT(external_due_declaration, enabled_gate);
  EXPECT_LT(enabled_gate, adaptive_due_declaration);
  EXPECT_LT(adaptive_due_declaration, adaptive_requested_gate);
  EXPECT_LT(adaptive_requested_gate, due_declaration);
  EXPECT_LT(due_declaration, poll_call);
  EXPECT_LT(poll_call, due_argument);
  const auto producer_failure_gate = display.find(
    "if (!producer_active)",
    poll_call
  );
  const auto producer_failure_publish = display.find(
    "publish_sbs_telemetry_failure();",
    producer_failure_gate
  );
  const auto producer_failure_return = display.find("return;", producer_failure_publish);
  ASSERT_NE(producer_failure_gate, std::string::npos);
  ASSERT_NE(producer_failure_publish, std::string::npos);
  ASSERT_NE(producer_failure_return, std::string::npos);
  EXPECT_LT(producer_failure_gate, producer_failure_publish);
  EXPECT_LT(producer_failure_publish, producer_failure_return);
  const auto output_end = display.find("end_sbs_gpu_timer(gpu_timer);");
  const auto telemetry_poll = display.find("poll_sbs_telemetry_after_output();", output_end);
  ASSERT_NE(output_end, std::string::npos);
  ASSERT_NE(telemetry_poll, std::string::npos);
  EXPECT_LT(output_end, telemetry_poll);

  // The shipped V2 renderer has no legacy subject, edge-risk, range, or anchor authority. Keep
  // those payload slots invalid instead of publishing stale bridge diagnostics as rendered state.
  EXPECT_EQ(
    display.find("sbs_telemetry_valid_field::subject"),
    std::string::npos
  );
  EXPECT_EQ(
    display.find("sbs_telemetry_valid_field::edge"),
    std::string::npos
  );
  EXPECT_EQ(
    display.find("sbs_telemetry_valid_field::anchor"),
    std::string::npos
  );
  EXPECT_EQ(
    display.find("sbs_telemetry_valid_field::range"),
    std::string::npos
  );
  const auto cut_flags = display.find(
    "sbs_adaptive_state::cut_flag_geometry_armed"
  );
  ASSERT_NE(cut_flags, std::string::npos);
  EXPECT_NE(
    display.find("sbs_adaptive_state::cut_flag_appearance_armed", cut_flags),
    std::string::npos
  );
  EXPECT_EQ(
    display.find("constexpr std::uint32_t cut_flag_geometry_armed"),
    std::string::npos
  );
}

TEST(ColorTransferTest, CompositeSrgbToBt709MatchesReferencePipeline) {
  // Exercise every 16-bit input code. This catches both transfer knees and validates the bounded
  // no-pow shader approximation against an explicit sRGB decode plus BT.709 encode. Its maximum
  // error is less than 0.01 of one 8-bit code step.
  for (unsigned code = 0; code <= 65535; ++code) {
    const auto input = static_cast<float>(code) / 65535.0f;
    EXPECT_NEAR(
      video::srgb_code_to_bt709_code(input),
      reference_srgb_code_to_bt709_code(input),
      3.8e-5f
    ) << "code="
      << code;
  }
}

TEST(ColorTransferTest, CompositeSrgbToBt709ClampsToCodeRange) {
  EXPECT_FLOAT_EQ(video::srgb_code_to_bt709_code(-1.0f), 0.0f);
  EXPECT_FLOAT_EQ(video::srgb_code_to_bt709_code(2.0f), 1.0f);
}

TEST(ColorTransferTest, HdrToSdrToneMapRequiresLinearHdrInputAndSdrTarget) {
  EXPECT_TRUE(video::hdr_to_sdr_tonemap_required(false, true, true));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(false, true, false));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(false, false, true));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(true, true, true));
}

TEST(HdrNegotiationTest, RequiresHttpAndRtspToSelectTheSameDynamicRange) {
  EXPECT_FALSE(video::hdr_stream_negotiation_is_coherent(true, 0));
  EXPECT_TRUE(video::hdr_stream_negotiation_is_coherent(true, 1));
  EXPECT_TRUE(video::hdr_stream_negotiation_is_coherent(false, 0));
  EXPECT_FALSE(video::hdr_stream_negotiation_is_coherent(false, 1));
}

TEST(NvencHdrMetadataTest, MapsHevcMasteringDisplayAndContentLightUnits) {
  SS_HDR_METADATA source {};
  source.displayPrimaries[0] = {34000, 16000};
  source.displayPrimaries[1] = {13250, 34500};
  source.displayPrimaries[2] = {7500, 3000};
  source.whitePoint = {15635, 16450};
  source.maxDisplayLuminance = 1000;
  source.minDisplayLuminance = 500;
  source.maxContentLightLevel = 1200;
  source.maxFrameAverageLightLevel = 400;

  const auto mapped = nvenc::hdr_metadata_from_sunshine(source, 1);
  ASSERT_TRUE(mapped.mastering_display);
  EXPECT_EQ(mapped.mastering_display->r.x, 34000);
  EXPECT_EQ(mapped.mastering_display->g.x, 13250);
  EXPECT_EQ(mapped.mastering_display->b.x, 7500);
  EXPECT_EQ(mapped.mastering_display->whitePoint.y, 16450);
  EXPECT_EQ(mapped.mastering_display->maxLuma, 10000000u);
  EXPECT_EQ(mapped.mastering_display->minLuma, 500u);
  ASSERT_TRUE(mapped.content_light_level);
  EXPECT_EQ(mapped.content_light_level->maxContentLightLevel, 1200);
  EXPECT_EQ(mapped.content_light_level->maxPicAverageLightLevel, 400);
}

TEST(NvencHdrMetadataTest, MapsAv1MasteringDisplayDenominators) {
  SS_HDR_METADATA source {};
  source.displayPrimaries[0] = {34000, 16000};
  source.displayPrimaries[1] = {13250, 34500};
  source.displayPrimaries[2] = {7500, 3000};
  source.whitePoint = {15635, 16450};
  source.maxDisplayLuminance = 1000;
  source.minDisplayLuminance = 500;

  const auto mapped = nvenc::hdr_metadata_from_sunshine(source, 2);
  ASSERT_TRUE(mapped.mastering_display);
  EXPECT_EQ(mapped.mastering_display->r.x, 44564);
  EXPECT_EQ(mapped.mastering_display->g.y, 45220);
  EXPECT_EQ(mapped.mastering_display->b.x, 9830);
  EXPECT_EQ(mapped.mastering_display->whitePoint.x, 20493);
  EXPECT_EQ(mapped.mastering_display->maxLuma, 256000u);
  EXPECT_EQ(mapped.mastering_display->minLuma, 819u);
  EXPECT_FALSE(mapped.content_light_level);
}

TEST(NvencHdrMetadataTest, OmitsMetadataForH264OrMissingSource) {
  SS_HDR_METADATA source {};
  EXPECT_FALSE(nvenc::hdr_metadata_from_sunshine(source, 0).mastering_display);
  EXPECT_FALSE(
    nvenc::hdr_metadata_from_sunshine(std::nullopt, 1).mastering_display
  );
}

TEST(ColorVectorsTest, LimitedRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, false, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto scale = static_cast<float>(1u << (bit_depth - 8u));
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 16.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 235.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), 128.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), 128.0f * scale / max_value, 1e-6f);
  }
}

TEST(ColorVectorsTest, FullRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, true, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto neutral_chroma = static_cast<float>(1u << (bit_depth - 1u)) / max_value;
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), neutral_chroma, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), neutral_chroma, 1e-6f);
  }
}

TEST(ColorVectorsTest, UintOutputRoundsBt2020LimitedValues) {
  const auto *vectors = video::color_vectors_from_colorspace(
    {video::colorspace_e::bt2020, false, 10},
    false
  );
  ASSERT_NE(vectors, nullptr);

  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f)), 64u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f)), 940u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f)), 512u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_v, 1.0f, 0.0f, 0.0f)), 960u);
}

TEST(NvencConfigTest, UsesVerifiedStreamingDefaults) {
  nvenc::nvenc_config config;

  EXPECT_EQ(config.vbv_percentage_increase, 100);
  EXPECT_TRUE(config.hevc_unidirectional_b);
}

TEST(NvencConfigTest, GatesHevcUnidirectionalBFrames) {
  nvenc::nvenc_config config;

  config.hevc_unidirectional_b = false;
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 1, true));

  config.hevc_unidirectional_b = true;
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 0, true));
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 2, true));
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 1, false));
  EXPECT_TRUE(nvenc::should_enable_hevc_unidirectional_b(config, 1, true));
}

TEST(NvencConfigTest, ForcesSplitEncodingOnlyForWideModernCodecs) {
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(false, 1, 7680, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 0, 7680, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 1, 4096, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 1, 7680, 1));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 3, 7680, 2));
  EXPECT_TRUE(nvenc::should_force_split_frame_encoding(true, 1, 7680, 2));
  EXPECT_TRUE(nvenc::should_force_split_frame_encoding(true, 2, 8192, 3));
}

TEST(CaptureBackendFailoverTest, RepeatedEarlyDdupFailuresLatchWgc) {
  video::capture_backend_failover_t failover;

  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    0,
    std::chrono::milliseconds(100)
  );
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);

  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::error,
    1,
    std::chrono::milliseconds(100)
  );
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::wgc);
}

TEST(CaptureBackendFailoverTest, StableDdupTenureForgivesOneOffReinit) {
  video::capture_backend_failover_t failover;
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    0,
    std::chrono::milliseconds(100)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    1,
    std::chrono::seconds(3)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::error,
    0,
    std::chrono::milliseconds(100)
  );

  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);
}

TEST(CaptureBackendFailoverTest, WgcSelectionDoesNotOscillate) {
  video::capture_backend_failover_t failover;
  failover.note_backend_opened(platf::capture_backend_e::wgc);
  failover.note_capture_result(
    platf::capture_backend_e::wgc,
    platf::capture_e::error,
    0,
    std::chrono::milliseconds(10)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    240,
    std::chrono::seconds(3)
  );

  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::wgc);

  failover.reset();
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);
}

struct FramerateX100Test: testing::TestWithParam<std::tuple<std::int32_t, video::rational_t>> {};

TEST_P(FramerateX100Test, ConvertsToExpectedRational) {
  const auto &[x100, expected] = GetParam();
  const auto actual = video::framerate_x100_to_rational(x100);
  EXPECT_EQ(actual.num, expected.num);
  EXPECT_EQ(actual.den, expected.den);
}

INSTANTIATE_TEST_SUITE_P(
  FramerateX100Tests,
  FramerateX100Test,
  testing::Values(
    std::make_tuple(2397, video::rational_t {24000, 1001}),
    std::make_tuple(2398, video::rational_t {24000, 1001}),
    std::make_tuple(2500, video::rational_t {25, 1}),
    std::make_tuple(2997, video::rational_t {30000, 1001}),
    std::make_tuple(3000, video::rational_t {30, 1}),
    std::make_tuple(5994, video::rational_t {60000, 1001}),
    std::make_tuple(6000, video::rational_t {60, 1}),
    std::make_tuple(11988, video::rational_t {120000, 1001}),
    std::make_tuple(23976, video::rational_t {240000, 1001}),
    std::make_tuple(9498, video::rational_t {4749, 50})
  )
);

TEST(HostSbsTelemetryTest, WireVisibleBitAssignmentsAreFrozen) {
  // stream.cpp serializes these masks and the client mirrors them. New flags are append-only:
  // renumbering an existing bit silently changes a running client's interpretation.
  EXPECT_EQ(platf::platform_caps::sbs_telemetry, 0x40000000u);
  EXPECT_EQ(video::sbs_telemetry_valid_field::config, 1u << 0);
  EXPECT_EQ(video::sbs_telemetry_valid_field::effective_pop, 1u << 1);
  EXPECT_EQ(video::sbs_telemetry_valid_field::edge, 1u << 2);
  EXPECT_EQ(video::sbs_telemetry_valid_field::change, 1u << 3);
  EXPECT_EQ(video::sbs_telemetry_valid_field::anchor, 1u << 4);
  EXPECT_EQ(video::sbs_telemetry_valid_field::subject, 1u << 5);
  EXPECT_EQ(video::sbs_telemetry_valid_field::valid_fraction, 1u << 6);
  EXPECT_EQ(video::sbs_telemetry_valid_field::range, 1u << 7);
  EXPECT_EQ(video::sbs_telemetry_valid_field::scene, 1u << 8);
  EXPECT_EQ(video::sbs_telemetry_valid_field::cuts, 1u << 9);
  EXPECT_EQ(video::sbs_telemetry_valid_field::faults, 1u << 10);

  EXPECT_EQ(video::sbs_telemetry_runtime_flag::profile_initialized, 1u << 0);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::adaptive_enabled, 1u << 1);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::pop_classified, 1u << 2);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::anchor_valid, 1u << 3);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::geometry_armed, 1u << 4);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::appearance_armed, 1u << 5);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::range_collapsed, 1u << 6);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::depth_ready, 1u << 7);
  EXPECT_EQ(video::sbs_telemetry_runtime_flag::hard_cut_pulse, 1u << 8);
}

TEST(HostSbsTelemetryTest, RendererConfigUsesNeutralWirePlaneForV2) {
  config::video_t::sbs_t settings;
  settings.pop_strength = 1.5;

  video::sbs_telemetry_snapshot_t v2;
  v2.runtime_flags = video::sbs_telemetry_runtime_flag::adaptive_enabled;
  video::apply_sbs_telemetry_config(v2, settings);
  // Telemetry v1 cannot encode V2's scene-latched coordinate policy. Its neutral compatibility
  // value is a protocol placeholder and must not imply a configurable V2 plane.
  EXPECT_EQ(v2.zero_plane_mode, 2);
  EXPECT_FLOAT_EQ(v2.pop_floor, 1.5f);
  EXPECT_FLOAT_EQ(v2.pop_ceiling, 1.5f);
  EXPECT_FLOAT_EQ(v2.effective_pop, 1.5f);
  EXPECT_EQ(
    v2.runtime_flags & video::sbs_telemetry_runtime_flag::adaptive_enabled,
    0u
  );
  EXPECT_NE(v2.valid_fields & video::sbs_telemetry_valid_field::config, 0u);
}

TEST(HostSbsTelemetryTest, SubscriptionLatchIsDisabledUntilExplicitlyEnabled) {
  video::sbs_telemetry_subscription_t subscription;
  EXPECT_FALSE(subscription.enabled());
  EXPECT_FALSE(subscription.focused());
  EXPECT_EQ(subscription.interval_ms(), 500);

  subscription.update(true, true, 125);
  EXPECT_TRUE(subscription.enabled());
  EXPECT_TRUE(subscription.focused());
  EXPECT_EQ(subscription.interval_ms(), 125);

  subscription.update(false, false, 750);
  EXPECT_FALSE(subscription.enabled());
  EXPECT_FALSE(subscription.focused());
  EXPECT_EQ(subscription.interval_ms(), 750);
}

TEST(HostSbsTelemetryTest, RendererGenerationsAreNonzeroAndAdvance) {
  const auto first = video::next_sbs_telemetry_generation();
  const auto second = video::next_sbs_telemetry_generation();
  EXPECT_NE(first, 0u);
  EXPECT_NE(second, 0u);
  EXPECT_NE(first, second);
}

TEST(HostSbsDimensionsTest, KeepsFourKPerEyeForHevcAndAv1) {
  for (const int video_format : {1, 2}) {
    const auto dimensions = video::host_sbs_output_dimensions(
      3840,
      2160,
      video_format,
      8192,
      8192
    );
    EXPECT_EQ(dimensions.width, 7680);
    EXPECT_EQ(dimensions.height, 2160);
  }
}

TEST(HostSbsDimensionsTest, HonorsStricterConfiguredLimit) {
  const auto dimensions = video::host_sbs_output_dimensions(2560, 1440, 2, 3840, 8192);
  EXPECT_EQ(dimensions.width, 3840);
  EXPECT_EQ(dimensions.height, 1080);
}

TEST(HostSbsDimensionsTest, CapsFiveKPerEyeToCurrentNvencLimit) {
  const auto dimensions = video::host_sbs_output_dimensions(5120, 2160, 2, 8192, 8192);
  EXPECT_EQ(dimensions.width, 8192);
  EXPECT_EQ(dimensions.height, 1728);
}

TEST(HostSbsDimensionsTest, HonorsLowerRuntimeCodecCapability) {
  const auto dimensions = video::host_sbs_output_dimensions(3840, 2160, 1, 8192, 4096);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 1152);
}

TEST(HostSbsDimensionsTest, CapsPortraitAgainstBothH264Axes) {
  const auto dimensions = video::host_sbs_output_dimensions(
    2160,
    5120,
    0,
    8192,
    4096,
    4096
  );
  EXPECT_EQ(dimensions.width, 3456);
  EXPECT_EQ(dimensions.height, 4096);
}

TEST(HostSbsDimensionsTest, KeepsPortraitWithinHevcAndAv1Limits) {
  for (const int video_format : {1, 2}) {
    const auto dimensions = video::host_sbs_output_dimensions(
      2160,
      5120,
      video_format,
      8192,
      8192,
      8192
    );
    EXPECT_EQ(dimensions.width, 4320);
    EXPECT_EQ(dimensions.height, 5120);
  }
}

TEST(HostSbsDimensionsTest, UsesMeasuredH264Capability) {
  const auto dimensions = video::host_sbs_output_dimensions(3840, 2160, 0, 8192);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 1152);
}

TEST(ClampEncodeDimensionsTest, PassesThroughAnEncodableMode) {
  for (const int video_format : {0, 1, 2}) {
    const auto dimensions = video::clamp_encode_dimensions(1920, 1080, video_format, 8192);
    EXPECT_EQ(dimensions.width, 1920);
    EXPECT_EQ(dimensions.height, 1080);
  }
}

TEST(ClampEncodeDimensionsTest, CapsAnOversizedRequestPreservingAspect) {
  // A live 0x3007 request can name any width the wire format carries. H.264 tops out at 4096, so
  // the request must be capped rather than refused: a failed non-SBS encoder creation ends the
  // whole session, whereas capping always produces a creatable mode.
  const auto dimensions = video::clamp_encode_dimensions(8192, 4320, 0, 8192);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 2160);
}

TEST(ClampEncodeDimensionsTest, CapsTallRequestAgainstRuntimeHeight) {
  const auto dimensions = video::clamp_encode_dimensions(
    2160,
    5120,
    0,
    4096,
    4096
  );
  EXPECT_EQ(dimensions.width, 1728);
  EXPECT_EQ(dimensions.height, 4096);
}

TEST(ClampEncodeDimensionsTest, HonorsLowerRuntimeCodecCapability) {
  const auto dimensions = video::clamp_encode_dimensions(7680, 4320, 1, 4096);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 2304);
}

TEST(ClampEncodeDimensionsTest, IgnoresAnUnknownRuntimeCapability) {
  // A zero runtime capability means "not probed yet"; the conservative per-codec ceiling applies.
  const auto hevc = video::clamp_encode_dimensions(7680, 4320, 1, 0);
  EXPECT_EQ(hevc.width, 7680);
  EXPECT_EQ(hevc.height, 4320);

  const auto h264 = video::clamp_encode_dimensions(7680, 4320, 0, 0);
  EXPECT_EQ(h264.width, 4096);
  EXPECT_EQ(h264.height, 2304);
}

TEST(ClampEncodeDimensionsTest, AlwaysProducesAnEvenEncodableSize) {
  // 4:2:0 subsampling means an odd derived height would be unencodable, and a degenerate cap must
  // still leave a usable surface rather than a zero-sized one.
  const auto scaled = video::clamp_encode_dimensions(4098, 1081, 0, 4096);
  EXPECT_EQ(scaled.width, 4096);
  EXPECT_EQ(scaled.height % 2, 0);
  EXPECT_GT(scaled.height, 0);

  const auto unscaled_odd = video::clamp_encode_dimensions(1921, 1081, 0, 4096, 4096);
  EXPECT_EQ(unscaled_odd.width, 1920);
  EXPECT_EQ(unscaled_odd.height, 1080);

  const auto degenerate = video::clamp_encode_dimensions(4096, 2160, 0, 1);
  EXPECT_EQ(degenerate.width, 2);
  EXPECT_GE(degenerate.height, 2);
  EXPECT_EQ(degenerate.height % 2, 0);
}

TEST(EncodeDimensionLimitsTest, HostSbsAndOrdinaryClampShareRuntimeCeilings) {
  const auto packed = video::host_sbs_output_dimensions(
    5000,
    3000,
    1,
    8192,
    7000,
    2800
  );
  const auto ordinary = video::clamp_encode_dimensions(
    10000,
    3000,
    1,
    7000,
    2800
  );

  EXPECT_EQ(packed.width, ordinary.width);
  EXPECT_EQ(packed.height, ordinary.height);
  EXPECT_LE(packed.width, 7000);
  EXPECT_LE(packed.height, 2800);
}

TEST(VideoPacketLifetimeTest, RetainsBroadcastStateUntilPacketIsConsumed) {
  auto channel = std::make_shared<int>(42);
  std::weak_ptr<int> weak_channel = channel;
  video::packet_raw_generic packet {{0x01}, 1, true};
  packet.channel_data = channel;

  channel.reset();
  EXPECT_FALSE(weak_channel.expired());

  packet.channel_data.reset();
  EXPECT_TRUE(weak_channel.expired());
}

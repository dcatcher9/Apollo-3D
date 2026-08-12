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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <src/depth_coordinate_v2.h>
#include <src/generated/sbs_adaptive_state_contract.h>
#include <src/host_sbs_shader_cache.h>
#include <src/nvenc/nvenc_base.h>
#include <src/nvenc/nvenc_config.h>
#include <src/platform/windows/video_dom_client.h>
#include <src/video.h>
#include <src/video_colorspace.h>
#include <src/video_depth_estimator.h>
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

  // The one-root preprocess closure remains available only to compute the calibrated model-input
  // identity. Its production shader is owned exactly once by the full producer closure; this test
  // deliberately never requests bytecode from the smaller snapshot.
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
        return candidate.filename == wanted.filename;
      }
    );
  };
  EXPECT_TRUE(has_producer_shader(cache::depth_scene_cut_evidence));
  EXPECT_TRUE(has_producer_shader(cache::depth_scene_cut_resolve));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_preprocess));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_cells));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_ocr_resolve));
  EXPECT_TRUE(has_producer_shader(cache::host_sbs_subtitle_locator_resolve));
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
  EXPECT_EQ(v2::contract_schema, 46u);
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
  EXPECT_EQ(small_calibration.preprocess.profile, "apollo-dav2-area-hdr-srgb-imagenet-v1");
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

TEST(DepthInputRegionTest, RequiresCanonicalFullSourceOrAuthenticatedVideoIdentity) {
  const models::depth_input_region_t full_source {
    .source_width = 1920u,
    .source_height = 1080u,
    .left = 0u,
    .top = 0u,
    .right = 1920u,
    .bottom = 1080u,
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
    .analysis_generation = 7u,
    .video_region = true,
  };
  EXPECT_TRUE(video_region.valid());

  auto video_without_identity = video_region;
  video_without_identity.analysis_generation = 0u;
  EXPECT_FALSE(video_without_identity.valid());

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
    .analysis_generation = 7u,
    .video_region = true,
  };
  auto moved = original;
  moved.left += 100u;
  moved.right += 100u;
  EXPECT_TRUE(original.same_analysis_domain(moved));
  EXPECT_NE(original, moved);

  auto resized = original;
  resized.right += 1u;
  EXPECT_FALSE(original.same_analysis_domain(resized));

  auto new_video = original;
  new_video.analysis_generation += 1u;
  EXPECT_FALSE(original.same_analysis_domain(new_video));

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
  EXPECT_FALSE(original.same_analysis_domain(full_source));
}

TEST(DepthInputRegionTest, DomainTransitionDecisionResetsExactlyOncePerStableChange) {
  models::video_depth_analysis_generation_tracker_t generation_tracker;
  models::depth_input_domain_tracker_t domain_tracker;
  const models::depth_input_region_t full_source {
    .source_width = 3840u,
    .source_height = 2160u,
    .left = 0u,
    .top = 0u,
    .right = 3840u,
    .bottom = 2160u,
  };
  EXPECT_TRUE(domain_tracker.update(full_source, models::input_color_space::srgb));
  EXPECT_FALSE(domain_tracker.update(full_source, models::input_color_space::srgb));

  models::video_depth_domain_key_t key {
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
    .analysis_generation = first_generation,
    .video_region = true,
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

  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::linear_sdr));

  generation_tracker.select_full_source();
  EXPECT_TRUE(domain_tracker.update(full_source, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(full_source, models::input_color_space::linear_sdr));
  roi.analysis_generation = generation_tracker.select(new_identity);
  EXPECT_TRUE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
  EXPECT_FALSE(domain_tracker.update(roi, models::input_color_space::linear_sdr));
}

TEST(DepthInputRegionTest, DumpSnapshotsPreserveFullOrRoiAnalysisDomain) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp"
  );
  ASSERT_FALSE(source.empty());
  EXPECT_NE(
    source.find("if (snapshot_debug_inputs)"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("if (snapshot_raw_model_depth)"),
    std::string::npos
  );
  EXPECT_EQ(
    source.find("&& !pending_input_region.video_region"),
    std::string::npos
  );

  const auto display_source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(display_source.empty());
  EXPECT_EQ(
    display_source.find(".observer_generation = border.generation"),
    std::string::npos
  );
  const auto estimator_header = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.h"
  );
  ASSERT_FALSE(estimator_header.empty());
  EXPECT_NE(
    estimator_header.find(
      "Position and observer snapshot generation are deliberately absent"
    ),
    std::string::npos
  );
  EXPECT_NE(
    display_source.find("dump_frame.depth_input_region = est.input_region"),
    std::string::npos
  );
  EXPECT_NE(
    display_source.find("dump_frame.depth_input_source ="),
    std::string::npos
  );
  EXPECT_NE(
    display_source.find("matched_render_slot->depth_input_srv();"),
    std::string::npos
  );
  EXPECT_EQ(
    display_source.find("Dump 3D request rejected: window-video ROI rendering is active"),
    std::string::npos
  );
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
  };
  EXPECT_FALSE(result.shadow_coordinate);
  EXPECT_TRUE(models::parallax_v2_result_is_authenticated(result));

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
    .analysis_generation = 7u,
    .video_region = true,
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

TEST(ParallaxV2ContractTest, DumpDecodesExactCountersInsteadOfSubnormalFloats) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp"
  );
  ASSERT_FALSE(source.empty());
  EXPECT_NE(
    source.find("std::bit_cast<std::uint32_t>(state[calibration_revision])"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("std::bit_cast<std::uint32_t>(state[confirmed_cut_count])"),
    std::string::npos
  );
  EXPECT_NE(
    source.find("calibration_revision_is_valid(calibration_revision_value)"),
    std::string::npos
  );
  EXPECT_EQ(
    source.find("{\"calibration_revision\", state[calibration_revision]}"),
    std::string::npos
  );
  EXPECT_NE(source.find("{\"schema\", shadow_state_dump_schema}"), std::string::npos);
  EXPECT_NE(
    source.find("{\"schema\", shadow_frame_stats_dump_schema}"),
    std::string::npos
  );
  EXPECT_NE(source.find("parallax_v2_coordinate_binding("), std::string::npos);
  EXPECT_NE(source.find("source_closure_sha256"), std::string::npos);
  const auto manifest = source.find("nlohmann::json manifest {");
  ASSERT_NE(manifest, std::string::npos);
  EXPECT_NE(source.find("{\"schema\", 27}", manifest), std::string::npos);
  EXPECT_NE(source.find("depth_input_region.json"), std::string::npos);
  EXPECT_NE(source.find("depth_input_source.png"), std::string::npos);
  EXPECT_NE(source.find("\"shadow_final_parallax + depth_input_region embedding\""), std::string::npos);
  EXPECT_NE(source.find("completed.parallax_v2_render_selected"), std::string::npos);
  EXPECT_NE(
    source.find("mapping_artifacts_match_selected_renderer"),
    std::string::npos
  );
  EXPECT_NE(source.find("live_shader_source"), std::string::npos);
  EXPECT_NE(
    source.find("parallax_v2_live_renderer_source_closure_sha256"),
    std::string::npos
  );
  EXPECT_NE(source.find("{\"shader_source\", shadow_shader_source}"), std::string::npos);
  EXPECT_NE(
    source.find("!parallax_v2_shader_identity_matches_contract("),
    std::string::npos
  );
  EXPECT_NE(source.find("state_semantics_valid"), std::string::npos);
  EXPECT_NE(source.find("{\"shared_configured\", {"), std::string::npos);
  EXPECT_NE(source.find("{\"live_effective\", {"), std::string::npos);
  EXPECT_EQ(source.find("{\"offline_analysis_configured\", {"), std::string::npos);
}

TEST(ParallaxV2ContractTest, DebugDumpPublishesCpuSnapshotOffTheRenderThread) {
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
  const auto capture = source.find("bool dumper::maybe_dump(");
  ASSERT_NE(publisher, std::string::npos);
  ASSERT_NE(capture, std::string::npos);
  ASSERT_LT(publisher, capture);

  const auto publisher_body = source.substr(publisher, capture - publisher);
  const auto capture_body = source.substr(capture);
  EXPECT_NE(publisher_body.find("write_color_preview("), std::string::npos);
  EXPECT_NE(publisher_body.find("models::file_sha256_hex("), std::string::npos);
  EXPECT_NE(
    publisher_body.find("std::filesystem::create_directories("),
    std::string::npos
  );
  EXPECT_EQ(publisher_body.find("read_texture(device"), std::string::npos);
  EXPECT_EQ(publisher_body.find("ID3D11DeviceContext"), std::string::npos);

  const auto enqueue = capture_body.find("worker_state->enqueue(");
  ASSERT_NE(enqueue, std::string::npos);
  const auto render_thread_prefix = capture_body.substr(0, enqueue);
  EXPECT_NE(render_thread_prefix.find("read_texture(device"), std::string::npos);
  EXPECT_EQ(render_thread_prefix.find("write_png("), std::string::npos);
  EXPECT_EQ(
    render_thread_prefix.find("models::file_sha256_hex("),
    std::string::npos
  );
  EXPECT_EQ(
    render_thread_prefix.find("std::filesystem::create_directories("),
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
    capture_body.find("worker_state->record_publication_failure("),
    std::string::npos
  );
  EXPECT_NE(source.find("async_->cancel_retries(button_request_);"), std::string::npos);
}

TEST(ParallaxV2ContractTest, CalibrationRevisionRejectsReservedSentinel) {
  namespace v2 = models::depth_coordinate_v2;
  EXPECT_TRUE(v2::calibration_revision_is_valid(0u));
  EXPECT_TRUE(v2::calibration_revision_is_valid(1u));
  EXPECT_FALSE(v2::calibration_revision_is_valid(v2::reserved_calibration_revision));
}

TEST(ParallaxV2ContractTest, NativeReplayUsesTheAuthenticatedShaderCache) {
  const auto replay = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/sbs_bench_depth_coordinate_v2.cpp"
  );
  ASSERT_FALSE(replay.empty());
  EXPECT_NE(replay.find("shader_cache::snapshot_sources("), std::string::npos);
  EXPECT_NE(replay.find("shader_cache::source_closure_sha256("), std::string::npos);
  EXPECT_NE(replay.find("shader_cache::get("), std::string::npos);
  EXPECT_EQ(replay.find("D3DCompileFromFile("), std::string::npos);
}

TEST(ParallaxV2ContractTest, RawProvenanceResolvesTheExactModelIdentity) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp"
  );
  ASSERT_FALSE(source.empty());
  const auto lookup = source.find("depth_coordinate_v2::find_model_calibration(");
  ASSERT_NE(lookup, std::string::npos);
  const auto preprocess_snapshot = source.find(
    "host_sbs_shader_cache::snapshot_sources(",
    lookup
  );
  ASSERT_NE(preprocess_snapshot, std::string::npos);
  const auto exact_calibration = source.find(
    "const auto *coordinate_calibration =",
    preprocess_snapshot
  );
  ASSERT_NE(exact_calibration, std::string::npos);
  const auto provenance = source.find("raw_model_provenance =", exact_calibration);
  ASSERT_NE(provenance, std::string::npos);
  EXPECT_LT(lookup, preprocess_snapshot);
  EXPECT_LT(preprocess_snapshot, exact_calibration);
  EXPECT_LT(exact_calibration, provenance);
  EXPECT_NE(
    source.find(".preprocess_source_closure_sha256 =", provenance),
    std::string::npos
  );
  EXPECT_EQ(
    source.rfind("model_calibrations.front()", provenance),
    std::string::npos
  );
}

TEST(ParallaxV2ContractTest, EvaluationRawArtifactsCarryProducerAttestation) {
  const auto harness = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/sbs_bench_harness.cpp"
  );
  ASSERT_FALSE(harness.empty());

  const auto provenance_gate = harness.find(
    "ordinary evaluation completed without exact raw-model provenance"
  );
  const auto provenance_contract = harness.find("\\\"raw_model_provenance\\\"");
  ASSERT_NE(provenance_gate, std::string::npos);
  ASSERT_NE(provenance_contract, std::string::npos);
  EXPECT_LT(provenance_gate, provenance_contract);
  EXPECT_NE(harness.find("direct_geometry_contract_schema = 25u"), std::string::npos);
  for (const auto *field : {
         "\\\"depth_model_url\\\"",
         "\\\"onnx_sha256\\\"",
         "\\\"preprocess_profile\\\"",
         "\\\"preprocess_source_closure_sha256\\\"",
         "\\\"raw_width\\\"",
         "\\\"raw_height\\\"",
       }) {
    EXPECT_NE(harness.find(field, provenance_contract), std::string::npos);
  }

  const auto raw_shape = harness.find("fs::path(o.out) / \"raw_shape.json\"");
  ASSERT_NE(raw_shape, std::string::npos);
  EXPECT_NE(harness.find("\\\"schema\\\": 1", raw_shape), std::string::npos);
  EXPECT_NE(harness.find("\\\"layout\\\": \\\"row-major\\\"", raw_shape), std::string::npos);
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

TEST(DirectxShaderSourceTest, DumpGeometryCompilationStaysOffTheLivePath) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(source.empty());
  const auto ensure_begin = source.find(
    "bool ensure_sbs_debug_geometry_resources()"
  );
  const auto ensure_end = source.find(
    "bool render_sbs_debug_geometry(",
    ensure_begin
  );
  ASSERT_NE(ensure_begin, std::string::npos);
  ASSERT_NE(ensure_end, std::string::npos);
  const auto dump_initializer = source.substr(
    ensure_begin,
    ensure_end - ensure_begin
  );
  EXPECT_EQ(dump_initializer.find("compile_shader("), std::string::npos);
  EXPECT_NE(
    dump_initializer.find("parallax_v2_live_diagnostic_specs"),
    std::string::npos
  );
  EXPECT_NE(dump_initializer.find("cache::get("), std::string::npos);
  EXPECT_NE(
    dump_initializer.find("parallax_v2_diagnostic_source_closure_sha256"),
    std::string::npos
  );
}

TEST(ParallaxV2ContractTest, ExactGpuReplayHashesAndDecodesOneImmutableSourceSnapshot) {
  const auto harness = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/sbs_bench_harness.cpp"
  );
  ASSERT_FALSE(harness.empty());

  const auto snapshot = harness.find("read_file_snapshot(current_frame, exact_source_snapshot)");
  const auto digest = harness.find("exact_source_sha256 = sha256_hex(exact_source_snapshot)", snapshot);
  const auto memory_decode = harness.find(
    "load_png(std::string_view(exact_source_snapshot), img)",
    digest
  );
  const auto dispatch = harness.find("exact_source_sha256,", memory_decode);
  ASSERT_NE(snapshot, std::string::npos);
  ASSERT_NE(digest, std::string::npos);
  ASSERT_NE(memory_decode, std::string::npos);
  ASSERT_NE(dispatch, std::string::npos);
  EXPECT_LT(snapshot, digest);
  EXPECT_LT(digest, memory_decode);
  EXPECT_LT(memory_decode, dispatch);
  EXPECT_NE(harness.find("InitializeFromMemory"), std::string::npos);
  EXPECT_NE(harness.find("WICDecodeMetadataCacheOnLoad"), std::string::npos);
  EXPECT_EQ(harness.find("sha256_file_hex(current_frame)"), std::string::npos);
}

TEST(DirectxShaderSourceTest, DumpGeometryBindsMatchedAuthenticatedState) {
  const auto source = read_source_file(
    SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp"
  );
  ASSERT_FALSE(source.empty());
  const auto render_begin = source.find(
    "bool render_sbs_debug_geometry("
  );
  const auto render_end = source.find(
    "void reset_matched_stats(",
    render_begin
  );
  ASSERT_NE(render_begin, std::string::npos);
  ASSERT_NE(render_end, std::string::npos);
  const auto render = source.substr(render_begin, render_end - render_begin);

  // The diagnostic shaders include the production WarpAvailable() authentication check. Both
  // mapping and mask draws must therefore bind the matched completion's state at t2; a null t2
  // silently turns both artifacts into identity geometry.
  EXPECT_NE(render.find("!parallax_state"), std::string::npos);
  for (const auto declaration : {
         "ID3D11ShaderResourceView *mapping_inputs[]",
         "ID3D11ShaderResourceView *mask_inputs[]",
       }) {
    const auto inputs_begin = render.find(declaration);
    ASSERT_NE(inputs_begin, std::string::npos);
    const auto inputs_end = render.find("};", inputs_begin);
    ASSERT_NE(inputs_end, std::string::npos);
    const auto inputs = render.substr(inputs_begin, inputs_end - inputs_begin);
    std::string compact_inputs;
    for (const char value : inputs) {
      if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
        compact_inputs.push_back(value);
      }
    }
    EXPECT_NE(
      compact_inputs.find("[]={source,warp_depth,parallax_state,"),
      std::string::npos
    );
  }

  const auto call = source.find("geometry_available = render_sbs_debug_geometry(");
  ASSERT_NE(call, std::string::npos);
  const auto call_end = source.find(");", call);
  ASSERT_NE(call_end, std::string::npos);
  EXPECT_NE(
    source.substr(call, call_end - call).find("est.shadow_state.Get(),"),
    std::string::npos
  );
}

TEST(DirectxShaderTest, CompilesGeneratedAdaptiveStateConsumers) {
  using Microsoft::WRL::ComPtr;

  constexpr std::array shaders {
    std::tuple {"depth_ema_motion_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_minmax_cs.hlsl", "main", "cs_5_0"},
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
  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> shader_errors;
  const auto compile_status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0,
    &shader_blob,
    &shader_errors
  );
  ASSERT_TRUE(SUCCEEDED(compile_status))
    << (shader_errors ?
          static_cast<const char *>(shader_errors->GetBufferPointer()) :
          "no compiler diagnostics");

  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(
    shader_blob->GetBufferPointer(),
    shader_blob->GetBufferSize(),
    nullptr,
    &shader
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
                          std::uint32_t color_mode = 0u
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

    std::array<std::uint32_t, 12> constants {};
    constants[0] = target_width;
    constants[1] = target_height;
    constants[2] = color_mode;
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
      appearance_uav.Get()
    };
    ID3D11Buffer *constant_buffers[] = {constant_buffer.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, input_srvs);
    context->CSSetUnorderedAccessViews(0, 2, output_uavs, nullptr);
    context->CSSetConstantBuffers(0, 1, constant_buffers);
    context->Dispatch(
      (target_width + 15u) / 16u,
      (target_height + 15u) / 16u,
      1u
    );

    ID3D11ShaderResourceView *null_srvs[] = {nullptr};
    ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
    context->CSSetShaderResources(0, 1, null_srvs);
    context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);

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

    return read_buffer(
             model_buffer.Get(),
             target_texels * 3u,
             model_output
           ) &&
           read_buffer(
             appearance_buffer.Get(),
             target_texels,
             appearance_ordinal
           );
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
  EXPECT_NE(
    display.find("Use an explicit portrait resolution instead of Windows display"),
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

TEST(DirectxShaderSourceTest, EmaMotionMaskUsesResolvedReferenceGridGradients) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto ema_motion = read_source_file(shader_dir + "depth_ema_motion_cs.hlsl");
  const auto constants = read_source_file(shader_dir + "include/depth_constants.hlsl");

  ASSERT_FALSE(ema_motion.empty());
  ASSERT_FALSE(constants.empty());
  EXPECT_NE(
    constants.find("#define DEPTH_GRADIENT_REFERENCE_SHORT_SIDE 434.0f"),
    std::string::npos
  );
  EXPECT_NE(
    constants.find("return (float)min(target_w, target_h) /"),
    std::string::npos
  );
  EXPECT_NE(
    ema_motion.find("reference_gradient = gradient * DepthReferenceTexelScale()"),
    std::string::npos
  );
  EXPECT_NE(
    ema_motion.find("reference_gradient >= ema_edge_gradient"),
    std::string::npos
  );
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

  // Subscription-off may retire an already submitted slot but cannot enqueue another copy, and
  // the diagnostic work is ordered after the production output rather than ahead of it.
  const auto due_declaration = display.find("const bool due =");
  const auto enabled_gate = display.find("enabled &&", due_declaration);
  const auto poll_call =
    display.find("depth_estimator->poll_depth_telemetry(", enabled_gate);
  const auto due_argument = display.find("due && producer_active", poll_call);
  ASSERT_NE(due_declaration, std::string::npos);
  ASSERT_NE(enabled_gate, std::string::npos);
  ASSERT_NE(poll_call, std::string::npos);
  ASSERT_NE(due_argument, std::string::npos);
  EXPECT_LT(due_declaration, enabled_gate);
  EXPECT_LT(enabled_gate, poll_call);
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

/**
 * @file tests/unit/test_video.cpp
 * @brief Test src/video.*.
 */
#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <src/nvenc/nvenc_base.h>
#include <src/nvenc/nvenc_config.h>
#include <src/video.h>
#include <src/video_colorspace.h>
#include <tuple>
#include <vector>

#ifdef _WIN32
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

  float structural_change_fraction(
    const std::vector<float> &current,
    const std::vector<float> &previous,
    int width,
    int height
  ) {
    constexpr std::array<std::array<int, 2>, 5> offsets {{
      {{0, 0}}, {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}}
    }};
    const auto sample = [&](const std::vector<float> &values, int sx, int sy) {
      sx = std::clamp(sx, 0, width - 1);
      sy = std::clamp(sy, 0, height - 1);
      return values[static_cast<std::size_t>(sy * width + sx)];
    };
    std::size_t changed = 0;
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
        unsigned common_comparisons = 0;
        unsigned ordering_flips = 0;
        for (std::size_t first = 0; first < 4; ++first) {
          for (std::size_t second = first + 1; second < 5; ++second) {
            const float current_delta =
              current_samples[first] - current_samples[second];
            const float previous_delta =
              previous_samples[first] - previous_samples[second];
            if (std::abs(current_delta) >= 0.01f &&
                std::abs(previous_delta) >= 0.01f) {
              ++common_comparisons;
              if ((current_delta < 0.0f) != (previous_delta < 0.0f)) {
                ++ordering_flips;
              }
            }
          }
        }
        if (common_comparisons >= 4 && ordering_flips >= 2 &&
            ordering_flips * 2 >= common_comparisons) {
          ++changed;
        }
      }
    }
    return static_cast<float>(changed) / static_cast<float>(width * height);
  }

  constexpr unsigned cut_flag_geometry_armed = 1u;
  constexpr unsigned cut_flag_appearance_armed = 2u;
  constexpr unsigned cut_flag_geometry_low_once = 4u;
  constexpr unsigned cut_flag_appearance_quiet_once = 8u;
  constexpr unsigned cut_flag_latched = 16u;
  constexpr unsigned cut_flags_ready =
    cut_flag_geometry_armed | cut_flag_appearance_armed;

  struct shot_cut_state_t {
    unsigned cut_flags = cut_flags_ready;
    float scene_age = 8.0f;
    float depth_change_baseline = 0.0f;
    bool initialized = true;
  };

  bool advance_shot_cut(
    shot_cut_state_t &state,
    float depth_change_fraction,
    float raw_rgb_change_fraction,
    float structural_change_fraction,
    bool model_input_history_valid = true
  ) {
    state.scene_age = state.initialized ?
                        std::min(state.scene_age + 1.0f, 65535.0f) :
                        0.0f;
    const bool appearance_proposal =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f &&
      structural_change_fraction >= 0.03f;
    const bool exposure_like_transition =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f &&
      structural_change_fraction < 0.01f;
    const bool geometry_armed =
      (state.cut_flags & cut_flag_geometry_armed) != 0u;
    const bool appearance_armed =
      (state.cut_flags & cut_flag_appearance_armed) != 0u;
    const bool cut_latched = (state.cut_flags & cut_flag_latched) != 0u;
    state.depth_change_baseline =
      std::clamp(state.depth_change_baseline, 0.0f, 1.0f);
    const bool relative_geometry_spike =
      cut_latched && !geometry_armed &&
      !exposure_like_transition &&
      state.scene_age >= 8.0f &&
      depth_change_fraction >= 0.30f &&
      (depth_change_fraction >= state.depth_change_baseline + 0.20f ||
       depth_change_fraction >= state.depth_change_baseline * 2.0f);
    const bool shot_cut =
      state.initialized &&
      ((geometry_armed && !exposure_like_transition &&
        depth_change_fraction >= 0.60f) ||
       (appearance_armed && appearance_proposal &&
        depth_change_fraction >= 0.25f) ||
       relative_geometry_spike);

    if (!state.initialized) {
      state.scene_age = 0.0f;
      state.cut_flags = 0u;
      state.depth_change_baseline = depth_change_fraction;
    } else if (shot_cut) {
      state.scene_age = 0.0f;
      state.cut_flags = cut_flag_latched;
      state.depth_change_baseline = depth_change_fraction;
    } else {
      state.depth_change_baseline +=
        (depth_change_fraction - state.depth_change_baseline) * 0.125f;
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
    }
    state.initialized = true;
    return shot_cut;
  }

  float reference_grid_step_edge_risk(
    int width,
    int height,
    float depth_step,
    bool vertical
  ) {
    constexpr float reference_short_side = 434.0f;
    constexpr float edge_threshold = 0.02f;
    constexpr float edge_weight_max = 8.0f;
    constexpr float edge_weight_scale = 256.0f;
    const float reference_texel_scale =
      static_cast<float>(std::min(width, height)) / reference_short_side;
    const float reference_gradient = depth_step * reference_texel_scale;
    const float weight =
      reference_gradient >= edge_threshold ?
        std::min(
          reference_gradient / edge_threshold,
          edge_weight_max * reference_texel_scale
        ) :
        0.0f;
    // Match the producer's positive float-to-uint conversion after +0.5.
    const auto fixed_weight = static_cast<std::uint32_t>(
      weight * edge_weight_scale + 0.5f
    );
    const std::uint64_t edge_texels =
      static_cast<std::uint64_t>(vertical ? height : width);
    const std::uint64_t total_texels =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    return static_cast<float>(
      static_cast<double>(fixed_weight) * static_cast<double>(edge_texels) /
      (static_cast<double>(total_texels) * edge_weight_scale)
    );
  }

  float unnormalized_step_edge_risk(
    int width,
    int height,
    float depth_step,
    bool vertical
  ) {
    constexpr float edge_threshold = 0.02f;
    constexpr float edge_weight_max = 8.0f;
    constexpr float edge_weight_scale = 256.0f;
    const float weight =
      depth_step >= edge_threshold ?
        std::min(depth_step / edge_threshold, edge_weight_max) :
        0.0f;
    const auto fixed_weight = static_cast<std::uint32_t>(
      weight * edge_weight_scale + 0.5f
    );
    const std::uint64_t edge_texels =
      static_cast<std::uint64_t>(vertical ? height : width);
    const std::uint64_t total_texels =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    return static_cast<float>(
      static_cast<double>(fixed_weight) * static_cast<double>(edge_texels) /
      (static_cast<double>(total_texels) * edge_weight_scale)
    );
  }
}  // namespace

#ifdef _WIN32
TEST(DirectxShaderTest, CompilesAllColorShaderVariants) {
  // D3DCompileFromFile does not require a D3D device. This covers BGRA8, FP16 SDR, PQ,
  // planar luma, both chroma sitings, and the HDR cursor shader in one focused check.
  EXPECT_EQ(platf::dxgi::init(), 0);
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

TEST(DirectxShaderSourceTest, HostSbsKeepsFramePairingAndProbeCapsAndRejectsRotation) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto reprojection = read_source_file(shader_dir + "sbs_reprojection_ps.hlsl");
  const auto common = read_source_file(shader_dir + "include/sbs_warp_common.hlsl");
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");

  ASSERT_FALSE(reprojection.empty());
  ASSERT_FALSE(common.empty());
  ASSERT_FALSE(display.empty());

  EXPECT_NE(reprojection.find("depth_change_baseline, adaptive_pop_ratio"), std::string::npos);
  EXPECT_EQ(reprojection.find("convergence_ema"), std::string::npos);

  // Portrait is an explicit W < H mode. Host SBS rejects non-identity DXGI rotation instead of
  // adding a dynamic transform to every depth probe or rotating the packed frame into top/bottom.
  EXPECT_EQ(reprojection.find("SourceTextureUV"), std::string::npos);
  EXPECT_NE(
    display.find("display->display_rotation != DXGI_MODE_ROTATION_IDENTITY"),
    std::string::npos
  );
  EXPECT_NE(
    display.find("Use an explicit portrait resolution instead of Windows display"),
    std::string::npos
  );

  // An all-invalid completion preserves the old packed target only after real depth exists,
  // instead of pairing its new color slot with deliberately held older depth. With no history,
  // repeated invalid completions keep drawing their matched colors through the flat identity path.
  EXPECT_NE(reprojection.find("DepthFrameState[0].w < 0.5f"), std::string::npos);
  EXPECT_NE(reprojection.find("DepthFrameState[0].z >= 0.5f"), std::string::npos);
  EXPECT_NE(reprojection.find("discard;"), std::string::npos);
  const auto preserves_previous = [](float initialized, float frame_state) {
    return frame_state < 0.5f && initialized >= 0.5f;
  };
  EXPECT_FALSE(preserves_previous(0.0f, 0.0f));  // first/repeated invalid: live flat color
  EXPECT_FALSE(preserves_previous(1.0f, 1.0f));  // first valid: matched color + depth
  EXPECT_TRUE(preserves_previous(1.0f, 0.0f));  // later invalid: hold matched pair
  EXPECT_NE(display.find("est.depth_frame_state.Get()"), std::string::npos);
  EXPECT_NE(
    display.find("matched_render_slot ? est.depth_frame_state.Get() : nullptr"),
    std::string::npos
  );

  // The cap is in total samples: one initial lookup plus at most max_probes - 1 loop samples.
  EXPECT_NE(common.find("int max_intervals = max(max_probes - 1, 1);"), std::string::npos);
  // Meeting that cap must coarsen the lattice rather than truncate its right-edge bracket.
  EXPECT_NE(common.find("while (intervals > max_intervals)"), std::string::npos);
  EXPECT_NE(common.find("return intervals;"), std::string::npos);
  EXPECT_EQ(common.find("return min(intervals, max_intervals);"), std::string::npos);

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

  EXPECT_FLOAT_EQ(
    structural_change_fraction(exposed, original, width, height),
    0.0f
  );
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
  EXPECT_GT(
    structural_change_fraction(vertical_ramp, horizontal_ramp, width, height),
    0.75f
  );

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
  EXPECT_FALSE(advance_shot_cut(below_raw, 0.25f, 0.699f, 0.03f));
  shot_cut_state_t below_ordinal;
  EXPECT_FALSE(advance_shot_cut(below_ordinal, 0.25f, 0.70f, 0.029f));
  shot_cut_state_t below_corroboration;
  EXPECT_FALSE(advance_shot_cut(below_corroboration, 0.249f, 0.70f, 0.03f));

  shot_cut_state_t exact_appearance;
  EXPECT_TRUE(advance_shot_cut(exact_appearance, 0.25f, 0.70f, 0.03f));
  shot_cut_state_t exact_geometry;
  EXPECT_TRUE(advance_shot_cut(exact_geometry, 0.60f, 0.0f, 0.0f));
}

TEST(HostSbsSceneCutTest, ExposureLikeTransitionVetoesAbsoluteAndRelativeDepthAuthority) {
  shot_cut_state_t absolute;
  absolute.cut_flags = cut_flags_ready;
  EXPECT_FALSE(advance_shot_cut(absolute, 0.95f, 0.70f, 0.009f));
  EXPECT_EQ(absolute.cut_flags, cut_flags_ready);

  shot_cut_state_t relative;
  relative.cut_flags = cut_flag_latched;
  relative.scene_age = 100.0f;
  relative.depth_change_baseline = 0.20f;
  EXPECT_FALSE(advance_shot_cut(relative, 0.95f, 0.70f, 0.0f));
  EXPECT_EQ(relative.cut_flags & cut_flag_latched, cut_flag_latched);

  // The veto describes broad appearance replacement, not a blanket reduction in geometry
  // sensitivity. A depth-authoritative change with no frame-wide RGB transition still cuts.
  shot_cut_state_t geometry_only;
  EXPECT_TRUE(advance_shot_cut(geometry_only, 0.60f, 0.699f, 0.0f));

  // Some structure below the qualified appearance threshold is intentionally ambiguous, not
  // exposure-like. Preserve standalone geometry authority in that band.
  shot_cut_state_t ambiguous_structure;
  EXPECT_TRUE(advance_shot_cut(ambiguous_structure, 0.60f, 0.70f, 0.01f));

  // Structural evidence at the exact entry boundary is a qualified cut, not exposure-like.
  shot_cut_state_t structural_cut;
  EXPECT_TRUE(advance_shot_cut(structural_cut, 0.60f, 0.70f, 0.03f));
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

TEST(HostSbsSceneCutTest, PersistentEvidencePulsesOnceAndIndependentArmsRecover) {
  shot_cut_state_t state;
  EXPECT_TRUE(advance_shot_cut(state, 0.60f, 0.80f, 0.03f));
  EXPECT_EQ(state.cut_flags, cut_flag_latched);

  // Steady shot-level evidence never rearms or periodically resets scene state.
  for (int update = 0; update < 12; ++update) {
    EXPECT_FALSE(advance_shot_cut(state, 0.60f, 0.80f, 0.03f)) << update;
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

  // Arming affects the following update, so the second quiet update did not pulse.
  EXPECT_TRUE(advance_shot_cut(state, 0.60f, 0.80f, 0.03f));
}

TEST(HostSbsSceneCutTest, RelativeGeometrySpikeEscapesWithoutPeriodicCooldown) {
  shot_cut_state_t state;
  state.cut_flags = cut_flag_latched;
  state.depth_change_baseline = 0.35f;
  state.scene_age = 100.0f;

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
  EXPECT_TRUE(advance_shot_cut(exact_margin, 0.55f, 0.0f, 0.0f));

  shot_cut_state_t exact_multiplier;
  exact_multiplier.cut_flags = cut_flag_latched;
  exact_multiplier.depth_change_baseline = 0.16f;
  EXPECT_TRUE(advance_shot_cut(exact_multiplier, 0.32f, 0.0f, 0.0f));
}

TEST(HostSbsSceneCutTest, RelativeGeometryIgnoresPostCutNormalizationSettling) {
  shot_cut_state_t state;

  // An appearance-authoritative cut can arrive at the 0.25 corroboration floor.
  EXPECT_TRUE(advance_shot_cut(state, 0.25f, 0.70f, 0.03f));
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

TEST(DirectxShaderSourceTest, HostSceneCutUsesIndependentAppearanceAndGeometryArms) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto histogram = read_source_file(shader_dir + "depth_subject_hist_cs.hlsl");
  const auto resolve = read_source_file(shader_dir + "depth_subject_resolve_cs.hlsl");
  const auto constants = read_source_file(shader_dir + "include/depth_constants.hlsl");
  const auto preprocess = read_source_file(shader_dir + "rgb_to_nchw_cs.hlsl");
  const auto history = read_source_file(shader_dir + "depth_valid_history_cs.hlsl");

  ASSERT_FALSE(histogram.empty());
  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(constants.empty());
  ASSERT_FALSE(preprocess.empty());
  ASSERT_FALSE(history.empty());
  EXPECT_NE(histogram.find("CurrentModelColor"), std::string::npos);
  EXPECT_NE(
    histogram.find("groupshared float g_current_appearance_ordinal"),
    std::string::npos
  );
  EXPECT_NE(histogram.find("CurrentAppearanceOrdinal"), std::string::npos);
  EXPECT_NE(histogram.find("PreviousAppearanceOrdinal"), std::string::npos);
  const auto ordinal_write = preprocess.find("OutputAppearanceOrdinal[base_idx]");
  const auto tone_map = preprocess.find("DepthColorToSrgb(pixel.rgb, color_mode)");
  ASSERT_NE(ordinal_write, std::string::npos);
  ASSERT_NE(tone_map, std::string::npos);
  EXPECT_LT(ordinal_write, tone_map);
  EXPECT_NE(preprocess.find("InputTexture.Load"), std::string::npos);
  EXPECT_NE(history.find("PreviousAppearanceOrdinal[idx]"), std::string::npos);
  EXPECT_NE(histogram.find("RAW_RGB_PIXEL_DELTA"), std::string::npos);
  EXPECT_NE(histogram.find("PlainHist[NUM_BINS + 3]"), std::string::npos);
  EXPECT_NE(histogram.find("tile_idx += 256u"), std::string::npos);
  EXPECT_NE(
    histogram.find("for (int first = 0; first < 4; ++first)"),
    std::string::npos
  );
  EXPECT_NE(
    histogram.find("for (int second = first + 1; second < 5; ++second)"),
    std::string::npos
  );
  EXPECT_NE(
    histogram.find("abs(current_delta) >= STRUCTURAL_ORDINAL_CONTRAST_FLOOR"),
    std::string::npos
  );
  EXPECT_NE(histogram.find("ordering_flips * 2u >= common_comparisons"), std::string::npos);
  EXPECT_EQ(histogram.find("CurrentModelLuma"), std::string::npos);
  EXPECT_NE(resolve.find("appearance_proposal"), std::string::npos);
  EXPECT_NE(resolve.find("exposure_like_transition"), std::string::npos);
  EXPECT_NE(resolve.find("raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH"), std::string::npos);
  EXPECT_NE(
    resolve.find("structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET"),
    std::string::npos
  );
  EXPECT_NE(resolve.find("geometry_armed && !exposure_like_transition"), std::string::npos);
  EXPECT_NE(
    resolve.find("cut_latched && !geometry_armed &&\n            !exposure_like_transition"),
    std::string::npos
  );
  EXPECT_NE(resolve.find("change_fraction >= DEPTH_CUT_CORROBORATE"), std::string::npos);
  EXPECT_NE(resolve.find("relative_geometry_spike"), std::string::npos);
  EXPECT_NE(
    resolve.find("scene_age >= POP_CLASSIFY_SETTLE_FRAMES"),
    std::string::npos
  );
  EXPECT_NE(resolve.find("CUT_FLAG_GEOMETRY_ARMED"), std::string::npos);
  EXPECT_NE(resolve.find("CUT_FLAG_APPEARANCE_ARMED"), std::string::npos);
  EXPECT_NE(resolve.find("change_fraction < DEPTH_CUT_LOW"), std::string::npos);
  EXPECT_NE(
    resolve.find(
      "lerp(depth_change_baseline, change_fraction, DEPTH_CUT_BASELINE_ALPHA)"
    ),
    std::string::npos
  );
  EXPECT_EQ(resolve.find("cut_state = -2.0f"), std::string::npos);
  EXPECT_EQ(resolve.find("scene_age >= 2.0f"), std::string::npos);
  EXPECT_EQ(resolve.find("color_change_fraction"), std::string::npos);
  EXPECT_NE(constants.find("#define RAW_RGB_PIXEL_DELTA 0.20f"), std::string::npos);
  EXPECT_NE(constants.find("#define RAW_RGB_CUT_HIGH 0.70f"), std::string::npos);
  EXPECT_NE(constants.find("#define STRUCTURAL_COLOR_CUT_HIGH 0.03f"), std::string::npos);
  EXPECT_NE(
    constants.find("#define STRUCTURAL_COLOR_EXPOSURE_QUIET 0.01f"),
    std::string::npos
  );
  EXPECT_NE(constants.find("#define DEPTH_CUT_HIGH 0.60f"), std::string::npos);
  EXPECT_NE(constants.find("#define DEPTH_CUT_CORROBORATE 0.25f"), std::string::npos);
  EXPECT_NE(constants.find("#define DEPTH_CUT_RELATIVE_FLOOR 0.30f"), std::string::npos);
  EXPECT_NE(constants.find("#define DEPTH_CUT_RELATIVE_MARGIN 0.20f"), std::string::npos);
  EXPECT_NE(constants.find("#define DEPTH_CUT_RELATIVE_MULTIPLIER 2.0f"), std::string::npos);
}

TEST(HostAdaptivePopTest, WeightedStepEdgeRiskIsStableAcrossResolvedDepthGrids) {
  // These are the actual 14-pixel-aligned 16:9 grids produced for supported short-side requests.
  // A discontinuity occupies one column (or row), so the coarser grid has more edge texels as a
  // fraction of the frame. Scaling both the linear weight and its cap cancels that density change.
  constexpr std::array<std::array<int, 2>, 3> grids {{
    {{700, 392}},
    {{742, 420}},
    {{770, 434}},
  }};
  constexpr float max_patch_rounding_residual = 0.007f;

  for (const float depth_step : {0.10f, 1.0f}) {
    const float reference_vertical =
      reference_grid_step_edge_risk(770, 434, depth_step, true);
    const float reference_horizontal =
      reference_grid_step_edge_risk(770, 434, depth_step, false);
    for (const auto &grid : grids) {
      EXPECT_NEAR(
        reference_grid_step_edge_risk(grid[0], grid[1], depth_step, true),
        reference_vertical,
        reference_vertical * max_patch_rounding_residual
      ) << grid[0] << 'x' << grid[1] << " depth_step=" << depth_step;
      EXPECT_NEAR(
        reference_grid_step_edge_risk(grid[0], grid[1], depth_step, false),
        reference_horizontal,
        reference_horizontal * max_patch_rounding_residual
      ) << grid[0] << 'x' << grid[1] << " depth_step=" << depth_step;
    }

    // Guard the regression itself: without reference-grid normalization, the 392-short-side
    // vertical edge reports about 10% more risk solely because one column is a larger fraction.
    EXPECT_GT(
      unnormalized_step_edge_risk(700, 392, depth_step, true),
      unnormalized_step_edge_risk(770, 434, depth_step, true) * 1.09f
    );
  }
}

TEST(DirectxShaderSourceTest, HostAdaptivePopUsesResolvedReferenceGridGradients) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto ema_motion = read_source_file(shader_dir + "depth_ema_motion_cs.hlsl");
  const auto histogram = read_source_file(shader_dir + "depth_subject_hist_cs.hlsl");
  const auto resolve = read_source_file(shader_dir + "depth_subject_resolve_cs.hlsl");
  const auto constants = read_source_file(shader_dir + "include/depth_constants.hlsl");

  ASSERT_FALSE(ema_motion.empty());
  ASSERT_FALSE(histogram.empty());
  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(constants.empty());
  EXPECT_NE(
    constants.find("#define DEPTH_GRADIENT_REFERENCE_SHORT_SIDE 434.0f"),
    std::string::npos
  );
  EXPECT_NE(
    constants.find("return (float)min(target_w, target_h) /"),
    std::string::npos
  );
  EXPECT_NE(histogram.find("float reference_grad = grad * reference_texel_scale"), std::string::npos);
  EXPECT_NE(
    histogram.find("reference_grad >= EDGE_GRADIENT_THRESHOLD"),
    std::string::npos
  );
  EXPECT_NE(
    histogram.find("EDGE_WEIGHT_MAX * reference_texel_scale"),
    std::string::npos
  );
  EXPECT_NE(
    histogram.find("exp(-10.0f * (reference_grad - 0.025f))"),
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
  EXPECT_NE(resolve.find("434-reference-texel"), std::string::npos);
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

TEST(ColorTransferTest, SbsIntermediatePreservesPrecisionDuringDdupFormatDiscovery) {
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(true, false, false, false));
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(false, true, true, false));
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(false, true, false, true));
  EXPECT_FALSE(video::sbs_intermediate_requires_fp16(false, true, false, false));
  EXPECT_FALSE(video::sbs_intermediate_requires_fp16(false, false, true, true));
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

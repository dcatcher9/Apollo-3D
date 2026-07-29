/**
 * @file tests/unit/test_video.cpp
 * @brief Test src/video.*.
 */
#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <src/nvenc/nvenc_base.h>
#include <src/nvenc/nvenc_config.h>
#include <src/generated/sbs_adaptive_state_contract.h>
#include <src/generated/sbs_scene_controller_contract.h>
#include <src/sbs_frame_roi_transform.h>
#include <src/video.h>
#include <src/video_colorspace.h>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <d3d11.h>
#include <d3dcompiler.h>
#include <src/platform/windows/sbs_debug_dump.h>
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
    constexpr std::array<std::array<int, 2>, 5> offsets {{
      {{0, 0}}, {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}}
    }};
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
            const bool current_reliable = std::abs(current_delta) >= 0.01f;
            const bool previous_reliable = std::abs(previous_delta) >= 0.01f;
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
        if (common_comparisons >= 4 && ordering_flips >= 2 &&
            ordering_flips * 2 >= common_comparisons) {
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
  constexpr unsigned cut_flags_ready =
    cut_flag_geometry_armed | cut_flag_appearance_armed;

  struct shot_cut_state_t {
    unsigned cut_flags = cut_flags_ready;
    float scene_age = 8.0f;
    float depth_change_baseline = 0.0f;
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
    float common_structural_support_fraction = 1.0f
  ) {
    state.scene_age = state.initialized ?
                        std::min(state.scene_age + 1.0f, 65535.0f) :
                        0.0f;
    const bool model_input_history_valid =
      state.model_input_history_state > 0.5f;
    const bool model_input_history_gap =
      state.model_input_history_state > 1.5f &&
      state.model_input_history_state < 2.5f;
    const bool low_structure_scene =
      state.model_input_history_state > 2.5f;
    const bool appearance_proposal =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f &&
      structural_change_fraction >= 0.03f;
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
    const bool broad_rgb_transition =
      model_input_history_valid &&
      raw_rgb_change_fraction >= 0.70f;
    const bool exposure_like_transition =
      broad_rgb_transition &&
      current_structure_reliable &&
      previous_structure_reliable &&
      common_structure_representative &&
      structural_change_fraction < 0.01f;
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
    const bool quiet_supported_repeat =
      model_input_history_valid &&
      current_structure_reliable &&
      previous_structure_reliable &&
      common_structure_representative &&
      raw_rgb_change_fraction < 0.01f &&
      structural_change_fraction < 0.01f;
    const bool geometry_armed =
      (state.cut_flags & cut_flag_geometry_armed) != 0u;
    const bool appearance_armed =
      (state.cut_flags & cut_flag_appearance_armed) != 0u;
    const bool cut_latched = (state.cut_flags & cut_flag_latched) != 0u;
    const bool appearance_recovery =
      (state.cut_flags & cut_flag_appearance_recovery) != 0u;
    const bool appearance_recovery_tail =
      appearance_recovery && quiet_supported_repeat;
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
    const bool shot_cut =
      state.initialized &&
      ((geometry_armed && !appearance_veto &&
        depth_change_fraction >= 0.60f) ||
       (appearance_armed && appearance_proposal &&
        depth_change_fraction >= 0.25f) ||
       (low_structure_scene && current_structure_reliable &&
        depth_change_fraction >= 0.60f) ||
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
      if (!structureless_transition && !appearance_recovery_tail) {
        state.depth_change_baseline +=
          (depth_change_fraction - state.depth_change_baseline) * 0.125f;
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
    }
    state.model_input_history_state =
      structureless_transition ? 2.0f :
      (persistent_structureless_transition ||
       (low_structure_scene && !current_structure_reliable)) ? 3.0f :
      1.0f;
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

TEST(FrameRoiTransformTest, CandidateContainsOwnershipOnly) {
  const auto candidate = models::make_frame_roi_transform_identity(
    41,
    3840,
    2160,
    770,
    434,
    3,
    0
  );

  EXPECT_TRUE(candidate.is_valid_candidate());
  EXPECT_FALSE(candidate.is_committed());
  EXPECT_EQ(candidate.transform_version, 0u);
  EXPECT_EQ(candidate.source_frame_id, 41u);
  EXPECT_EQ(candidate.backend_generation, 3u);
  EXPECT_EQ(candidate.gpu_bank_index, 0u);

  auto invalid = candidate;
  invalid.source_width = 0;
  EXPECT_FALSE(invalid.is_valid_candidate());
  invalid = candidate;
  invalid.model_width = 768;
  EXPECT_FALSE(invalid.is_valid_candidate());
  invalid = candidate;
  invalid.gpu_bank_index = models::frame_roi_transform_bank_count;
  EXPECT_FALSE(invalid.is_valid_candidate());
}

TEST(FrameRoiTransformTest, TwoSlotsPreserveCompletedAndPendingFrameOwnership) {
  models::frame_roi_transform_buffer transforms;
  ASSERT_EQ(transforms.writable_bank(), 0u);
  const auto frame_0_candidate = models::make_frame_roi_transform_identity(
    0,
    1920,
    1080,
    770,
    434,
    1,
    *transforms.writable_bank()
  );

  // Frame ID zero is a valid identity. Reserving assigns its version before GPU dispatch but
  // changes neither completed nor pending ownership.
  const auto frame_0 = transforms.reserve(frame_0_candidate);
  ASSERT_TRUE(frame_0.has_value());
  EXPECT_TRUE(frame_0->is_committed());
  EXPECT_EQ(frame_0->source_frame_id, 0u);
  EXPECT_EQ(frame_0->transform_version, 1u);
  EXPECT_TRUE(transforms.is_reserved(*frame_0));
  EXPECT_TRUE(transforms.has_reserved());
  EXPECT_FALSE(transforms.has_pending());
  EXPECT_EQ(transforms.pending_for(0), nullptr);
  EXPECT_EQ(transforms.completed_for(0), nullptr);
  EXPECT_FALSE(transforms.writable_bank().has_value());

  ASSERT_TRUE(transforms.commit_reserved_enqueued(*frame_0));
  EXPECT_FALSE(transforms.has_reserved());
  ASSERT_TRUE(transforms.has_pending());
  const auto *pending_0 = transforms.pending_for(0);
  ASSERT_NE(pending_0, nullptr);
  EXPECT_EQ(pending_0->transform_version, 1u);
  EXPECT_EQ(transforms.pending_bank(), 0u);
  ASSERT_TRUE(transforms.complete(*pending_0));
  const auto *completed_0 = transforms.completed_for(0);
  ASSERT_NE(completed_0, nullptr);
  EXPECT_EQ(completed_0->source_frame_id, 0u);
  EXPECT_EQ(transforms.completed_bank(), 0u);
  EXPECT_FALSE(transforms.has_pending());

  ASSERT_EQ(transforms.writable_bank(), 1u);
  const auto frame_1 = transforms.reserve(
    models::make_frame_roi_transform_identity(
      1,
      1920,
      1080,
      756,
      448,
      2,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(frame_1.has_value());
  EXPECT_EQ(frame_1->transform_version, 2u);
  EXPECT_EQ(frame_1->backend_generation, 2u);
  EXPECT_EQ(frame_1->gpu_bank_index, 1u);
  EXPECT_EQ(frame_1->model_width, 756u);
  EXPECT_EQ(frame_1->model_height, 448u);
  // The prior completed slot survives both reservation and the next accepted inference.
  ASSERT_NE(transforms.completed_for(0), nullptr);
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*frame_1));
  ASSERT_NE(transforms.completed_for(0), nullptr);
  ASSERT_NE(transforms.pending_for(1), nullptr);
  ASSERT_TRUE(transforms.complete(*transforms.pending_for(1)));
  EXPECT_EQ(transforms.completed_for(0), nullptr);
  ASSERT_NE(transforms.completed_for(1), nullptr);
  EXPECT_EQ(transforms.completed_for(1)->transform_version, 2u);
}

TEST(FrameRoiTransformTest, RejectedSubmissionRollsBackWithoutReusingVersion) {
  models::frame_roi_transform_buffer transforms;
  auto rejected = transforms.reserve(
    models::make_frame_roi_transform_identity(
      7,
      1920,
      1080,
      770,
      434,
      1,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(rejected.has_value());
  ASSERT_EQ(rejected->transform_version, 1u);
  ASSERT_TRUE(transforms.rollback_reserved(*rejected));
  EXPECT_FALSE(transforms.has_reserved());
  EXPECT_FALSE(transforms.has_pending());
  ASSERT_EQ(transforms.writable_bank(), rejected->gpu_bank_index);

  const auto accepted = transforms.reserve(
    models::make_frame_roi_transform_identity(
      8,
      1920,
      1080,
      770,
      434,
      1,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(accepted.has_value());
  // A rejected version is never recycled onto possibly stale GPU contents.
  EXPECT_EQ(accepted->transform_version, 2u);
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*accepted));
  ASSERT_TRUE(transforms.complete(*transforms.pending_for(8)));
  EXPECT_NE(transforms.completed_for(8), nullptr);
}

TEST(FrameRoiTransformTest, AcceptedOwnershipFailureBecomesExplicitDroppedOrphan) {
  models::frame_roi_transform_buffer transforms;
  auto first = transforms.reserve(
    models::make_frame_roi_transform_identity(
      20,
      1920,
      1080,
      770,
      434,
      1,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*first));
  ASSERT_TRUE(transforms.complete(*transforms.pending_for(20)));

  const auto orphan = transforms.reserve(
    models::make_frame_roi_transform_identity(
      0,
      1920,
      1080,
      770,
      434,
      2,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(orphan.has_value());
  transforms.orphan_reserved_enqueued(0);
  EXPECT_FALSE(transforms.has_reserved());
  EXPECT_FALSE(transforms.has_pending());
  EXPECT_TRUE(transforms.has_orphaned());
  EXPECT_TRUE(transforms.orphaned_for(0));
  EXPECT_FALSE(transforms.writable_bank().has_value());
  // Orphaning never disturbs the last completed bank.
  EXPECT_NE(transforms.completed_for(20), nullptr);
  EXPECT_FALSE(transforms.drop_in_flight(1));
  ASSERT_TRUE(transforms.drop_in_flight(0));
  EXPECT_FALSE(transforms.has_orphaned());
  EXPECT_TRUE(transforms.writable_bank().has_value());
  EXPECT_NE(transforms.completed_for(20), nullptr);
}

TEST(FrameRoiTransformTest, MismatchedIdentityCannotTransitionOrComplete) {
  models::frame_roi_transform_buffer transforms;
  const auto reserved = transforms.reserve(
    models::make_frame_roi_transform_identity(
      100,
      1920,
      1080,
      770,
      434,
      4,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(reserved.has_value());
  auto wrong_generation = *reserved;
  ++wrong_generation.backend_generation;
  EXPECT_FALSE(transforms.commit_reserved_enqueued(wrong_generation));
  EXPECT_FALSE(transforms.rollback_reserved(wrong_generation));
  EXPECT_TRUE(transforms.is_reserved(*reserved));

  ASSERT_TRUE(transforms.commit_reserved_enqueued(*reserved));
  auto wrong_version = *reserved;
  ++wrong_version.transform_version;
  EXPECT_FALSE(transforms.complete(wrong_version));
  EXPECT_NE(transforms.pending_for(100), nullptr);
  ASSERT_TRUE(transforms.drop_in_flight(100));
  EXPECT_FALSE(transforms.has_pending());
}

TEST(FrameRoiTransformTest, TerminalAbandonPreservesOnlyLastCompletedBank) {
  models::frame_roi_transform_buffer transforms;
  const auto completed = transforms.reserve(
    models::make_frame_roi_transform_identity(
      30,
      1920,
      1080,
      770,
      434,
      1,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(completed.has_value());
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*completed));
  ASSERT_TRUE(transforms.complete(*transforms.pending_for(30)));

  const auto pending = transforms.reserve(
    models::make_frame_roi_transform_identity(
      31,
      1920,
      1080,
      770,
      434,
      2,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(pending.has_value());
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*pending));
  transforms.abandon_in_flight();
  EXPECT_FALSE(transforms.has_reserved());
  EXPECT_FALSE(transforms.has_pending());
  EXPECT_FALSE(transforms.has_orphaned());
  EXPECT_EQ(transforms.pending_for(31), nullptr);
  EXPECT_NE(transforms.completed_for(30), nullptr);
  EXPECT_TRUE(transforms.writable_bank().has_value());
}

TEST(FrameRoiTransformTest, PreparingAnotherCandidateCannotOverwriteCompleted) {
  models::frame_roi_transform_buffer transforms;
  const auto first = transforms.reserve(
    models::make_frame_roi_transform_identity(
      100,
      1920,
      1080,
      770,
      434,
      1,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(transforms.commit_reserved_enqueued(*first));
  ASSERT_TRUE(transforms.complete(*transforms.pending_for(100)));

  const auto next = transforms.reserve(
    models::make_frame_roi_transform_identity(
      101,
      1920,
      1080,
      770,
      434,
      2,
      *transforms.writable_bank()
    )
  );
  ASSERT_TRUE(next.has_value());
  EXPECT_NE(next->gpu_bank_index, first->gpu_bank_index);
  EXPECT_NE(transforms.completed_for(100), nullptr);
  EXPECT_FALSE(transforms.reserve(models::make_frame_roi_transform_identity(
    102,
    1920,
    1080,
    770,
    434,
    2,
    0
  )).has_value());
  EXPECT_NE(transforms.completed_for(100), nullptr);
}

#ifdef _WIN32
TEST(DirectxShaderTest, CompilesAllColorShaderVariants) {
  // D3DCompileFromFile does not require a D3D device. This covers BGRA8, FP16 SDR, PQ,
  // planar luma, both chroma sitings, and the HDR cursor shader in one focused check.
  EXPECT_EQ(platf::dxgi::init(), 0);
}

TEST(DirectxShaderTest, CompilesGeneratedAdaptiveStateConsumers) {
  using Microsoft::WRL::ComPtr;

  constexpr std::array shaders {
    std::tuple {"depth_minmax_ema_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_subject_resolve_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"depth_valid_history_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_prepare_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_features_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_evidence_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_columns_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_plan_columns_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_rows_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_plan_rows_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_candidates_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_rules_reduce_cs.hlsl", "main", "cs_5_0"},
    std::tuple {
      "sbs_scene_rules_reduce_serial_reference_cs.hlsl",
      "main",
      "cs_5_0"
    },
    std::tuple {"sbs_scene_rules_resolve_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_scene_history_commit_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_forward_coverage_cs.hlsl", "main", "cs_5_0"},
    std::tuple {"sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0"},
    std::tuple {"sbs_reprojection_ps.hlsl", "mapping_ps", "ps_5_0"},
    std::tuple {"sbs_reprojection_ps.hlsl", "mask_ps", "ps_5_0"},
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

TEST(
  DepthEstimatorTimingSourceTest,
  SceneControllerTelemetrySharesTheEstimatorDisjointScope
) {
  const auto estimator =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp");
  const auto controller =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/sbs_scene_controller_gpu.cpp");
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(controller.empty());

  const auto count_occurrences =
    [](const std::string &source, const std::string_view needle) {
      std::size_t count = 0;
      std::size_t cursor = 0;
      while ((cursor = source.find(needle, cursor)) != std::string::npos) {
        ++count;
        cursor += needle.size();
      }
      return count;
    };
  EXPECT_EQ(
    count_occurrences(
      estimator,
      "context->Begin(slot.disjoint.Get())"
    ),
    1u
  );
  EXPECT_EQ(
    count_occurrences(
      estimator,
      "context->End(slot->disjoint.Get())"
    ),
    1u
  );
  EXPECT_NE(estimator.find("\"scene_prepare_gpu\""), std::string::npos);
  EXPECT_NE(estimator.find("\"scene_rules_gpu\""), std::string::npos);

  const auto pre_end = estimator.find("mark_d3d_pre_end(d3d_timer);");
  const auto prepare_start =
    estimator.find("mark_d3d_scene_prepare_start(d3d_timer);", pre_end);
  const auto prepare_call =
    estimator.find("scene_controller->prepare_scene(", prepare_start);
  const auto prepare_end =
    estimator.find("mark_d3d_scene_prepare_end(", prepare_call);
  const auto scope_end = estimator.find("end_d3d_perf(d3d_timer);", prepare_end);
  ASSERT_NE(pre_end, std::string::npos);
  ASSERT_NE(prepare_start, std::string::npos);
  ASSERT_NE(prepare_call, std::string::npos);
  ASSERT_NE(prepare_end, std::string::npos);
  ASSERT_NE(scope_end, std::string::npos);
  EXPECT_LT(pre_end, prepare_start);
  EXPECT_LT(prepare_start, prepare_call);
  EXPECT_LT(prepare_call, prepare_end);
  EXPECT_LT(prepare_end, scope_end);

  const auto rules_start =
    estimator.find("mark_d3d_scene_rules_start(d3d_timer);");
  const auto resolve_call =
    estimator.find("scene_controller->resolve_completed(", rules_start);
  const auto rules_end =
    estimator.find("mark_d3d_scene_rules_end(", resolve_call);
  ASSERT_NE(rules_start, std::string::npos);
  ASSERT_NE(resolve_call, std::string::npos);
  ASSERT_NE(rules_end, std::string::npos);
  EXPECT_LT(rules_start, resolve_call);
  EXPECT_LT(resolve_call, rules_end);

  const auto test_guard = controller.rfind(
    "#ifdef SUNSHINE_TESTS",
    controller.find("D3D11_QUERY_TIMESTAMP_DISJOINT")
  );
  ASSERT_NE(test_guard, std::string::npos);
  EXPECT_LT(
    test_guard,
    controller.find("D3D11_QUERY_TIMESTAMP_DISJOINT")
  );
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

  D3D11_SAMPLER_DESC sampler_desc {};
  sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
  sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
  sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
  sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
  sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
  ComPtr<ID3D11SamplerState> sampler;
  ASSERT_TRUE(SUCCEEDED(device->CreateSamplerState(&sampler_desc, &sampler)));

  using rgba_pixel_t = std::array<float, 4>;
  const auto run_case = [&](
                          UINT source_width,
                          UINT source_height,
                          UINT target_width,
                          UINT target_height,
                          const std::vector<rgba_pixel_t> &source_pixels,
                          std::vector<float> &model_output,
                          std::vector<float> &appearance_ordinal
                        ) {
    if (source_pixels.size() !=
        static_cast<std::size_t>(source_width) * source_height) {
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
    if (FAILED(device->CreateTexture2D(
          &texture_desc,
          &texture_data,
          &input_texture
        )) ||
        FAILED(device->CreateShaderResourceView(
          input_texture.Get(),
          nullptr,
          &input_srv
        ))) {
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
    if (!create_output(target_texels * 3u, model_buffer, model_uav) ||
        !create_output(target_texels, appearance_buffer, appearance_uav)) {
      return false;
    }

    std::array<std::uint32_t, 16> constants {};
    constants[0] = target_width;
    constants[1] = target_height;
    constants[2] = 0u;  // display-referred sRGB input
    D3D11_BUFFER_DESC constant_desc {};
    constant_desc.ByteWidth = sizeof(constants);
    constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
    constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constant_data {};
    constant_data.pSysMem = constants.data();
    ComPtr<ID3D11Buffer> constant_buffer;
    if (FAILED(device->CreateBuffer(
          &constant_desc,
          &constant_data,
          &constant_buffer
        ))) {
      return false;
    }

    ID3D11ShaderResourceView *input_srvs[] = {input_srv.Get()};
    ID3D11UnorderedAccessView *output_uavs[] = {
      model_uav.Get(),
      appearance_uav.Get()
    };
    ID3D11Buffer *constant_buffers[] = {constant_buffer.Get()};
    ID3D11SamplerState *samplers[] = {sampler.Get()};
    context->CSSetShader(shader.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, input_srvs);
    context->CSSetUnorderedAccessViews(0, 2, output_uavs, nullptr);
    context->CSSetConstantBuffers(0, 1, constant_buffers);
    context->CSSetSamplers(0, 1, samplers);
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
      if (FAILED(context->Map(
            staging.Get(),
            0,
            D3D11_MAP_READ,
            0,
            &mapped
          ))) {
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
    EXPECT_FLOAT_EQ(appearance_ordinal[1], 0.6f);
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

#ifdef _WIN32
TEST(SbsDebugDumpContractTest, SceneControllerPackageIsMatchedAndOptional) {
  platf::sbs_debug::frame frame;
  EXPECT_EQ(frame.scene_controller_scene_rgb, nullptr);
  EXPECT_EQ(frame.scene_controller_analysis_grid, nullptr);
  EXPECT_EQ(frame.scene_controller_dense_output, nullptr);
  EXPECT_EQ(frame.scene_controller_global_output, nullptr);
  EXPECT_EQ(frame.scene_controller_layout_history, nullptr);
  EXPECT_EQ(frame.scene_controller_depth_history, nullptr);
  EXPECT_EQ(frame.scene_controller_hidden_output, nullptr);
  EXPECT_EQ(frame.scene_controller_meta, nullptr);
  EXPECT_EQ(frame.scene_controller_rule_state, nullptr);
  EXPECT_FALSE(frame.scene_controller_snapshot_available);

  EXPECT_EQ(sbs_scene_controller::schema_version, 1u);
  EXPECT_EQ(sbs_scene_controller::analysis_grid_channel_count, 10u);
  EXPECT_EQ(sbs_scene_controller::dense_out_channel_count, 14u);
  EXPECT_EQ(sbs_scene_controller::global_out_word_count, 41u);
  EXPECT_EQ(sbs_scene_controller::rule_state_word_count, 64u);

  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  const auto dumper =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/sbs_debug_dump.cpp");
  ASSERT_FALSE(display.empty());
  ASSERT_FALSE(dumper.empty());
  EXPECT_NE(
    display.find("est.scene_controller_snapshot_available &&"),
    std::string::npos
  );
  EXPECT_NE(
    display.find(
      "est.scene_controller_frame_id == est.completed_frame_id"
    ),
    std::string::npos
  );
  EXPECT_NE(
    dumper.find(
      "post-resolve promoted layout history; not the pre-resolve input bank"
    ),
    std::string::npos
  );
  EXPECT_NE(
    dumper.find(
      "post-resolve promoted depth history; not the pre-resolve input bank"
    ),
    std::string::npos
  );
  EXPECT_NE(
    dumper.find(
      "optional scene-controller package is unavailable"
    ),
    std::string::npos
  );
  EXPECT_NE(
    dumper.find("global_out_word_e::reserved_40"),
    std::string::npos
  );
}
#endif

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
  const auto unmatched_reset = display.find("if (!matched_render_slot) {");
  const auto estimator_clear = display.find("est = {};", unmatched_reset);
  const auto frame_state_bind =
    display.find("est.depth_frame_state.Get()", estimator_clear);
  ASSERT_NE(unmatched_reset, std::string::npos);
  ASSERT_NE(estimator_clear, std::string::npos);
  ASSERT_NE(frame_state_bind, std::string::npos);
  EXPECT_LT(estimator_clear, frame_state_bind);

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
  EXPECT_TRUE(advance_shot_cut(
    state,
    0.827988386f,
    0.998319566f,
    0.008665371f,
    0.160147399f,
    0.169116467f,
    0.020610625f
  ));

  // A disjoint-support real endpoint is never delayed by the recovery guard,
  // even when its aggregate structural flip fraction remains exposure-quiet.
  shot_cut_state_t immediate_cut;
  EXPECT_FALSE(advance_shot_cut(immediate_cut, 0.83f, 0.99f, 0.005f));
  EXPECT_TRUE(advance_shot_cut(
    immediate_cut,
    0.60f,
    0.99f,
    0.008f,
    0.16f,
    0.17f,
    0.02f
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
  // The first supported return consumes the event-scoped authority and can cut even though the
  // ordinary post-cut geometry arm is still refractory.
  EXPECT_TRUE(advance_shot_cut(
    persistent_flat,
    0.60f,
    0.01f,
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
  EXPECT_TRUE(advance_shot_cut(
    changed_return,
    0.60f,
    0.01f,
    0.0f,
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
  EXPECT_TRUE(advance_shot_cut(
    below_support,
    0.60f,
    0.70f,
    0.0f,
    0.90f,
    0.90f,
    0.009f
  ));

  shot_cut_state_t exact_support;
  EXPECT_FALSE(advance_shot_cut(
    exact_support,
    0.95f,
    0.70f,
    0.0f,
    0.01f,
    0.01f,
    0.01f
  ));
}

TEST(HostSbsSceneCutTest, ExposureClassificationRequiresRepresentativeCommonSupport) {
  shot_cut_state_t below_ratio;
  EXPECT_TRUE(advance_shot_cut(
    below_ratio,
    0.60f,
    0.70f,
    0.0f,
    0.90f,
    0.90f,
    0.449f
  ));

  shot_cut_state_t exact_ratio;
  EXPECT_FALSE(advance_shot_cut(
    exact_ratio,
    0.95f,
    0.70f,
    0.0f,
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
  EXPECT_NE(state.cut_flags & cut_flag_latched, 0u)
    << "independent proposal rearm must retain the permanent post-cut phase marker";

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
  const auto estimator =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp");

  ASSERT_FALSE(histogram.empty());
  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(constants.empty());
  ASSERT_FALSE(preprocess.empty());
  ASSERT_FALSE(history.empty());
  ASSERT_FALSE(estimator.empty());
  EXPECT_NE(histogram.find("CurrentModelColor"), std::string::npos);
  EXPECT_NE(
    histogram.find("groupshared float g_current_appearance_ordinal"),
    std::string::npos
  );
  EXPECT_NE(histogram.find("CurrentAppearanceOrdinal"), std::string::npos);
  EXPECT_NE(histogram.find("PreviousAppearanceOrdinal"), std::string::npos);
  const auto footprint_filter = preprocess.find("float4 SampleModelFootprint");
  const auto footprint_sample =
    preprocess.find("float4 pixel = SampleModelFootprint");
  const auto footprint_load =
    preprocess.find("InputTexture.Load(int3(source_x, source_y, 0))");
  const auto ordinal_point_load =
    preprocess.find(
      "float3 capture_rgb = InputTexture.Load(int3(source_point, 0)).rgb"
    );
  const auto ordinal_write = preprocess.find("OutputAppearanceOrdinal[base_idx]");
  const auto tone_map = preprocess.find("DepthColorToSrgb(pixel.rgb, color_mode)");
  ASSERT_NE(footprint_filter, std::string::npos);
  ASSERT_NE(footprint_sample, std::string::npos);
  ASSERT_NE(footprint_load, std::string::npos);
  ASSERT_NE(ordinal_point_load, std::string::npos);
  ASSERT_NE(ordinal_write, std::string::npos);
  ASSERT_NE(tone_map, std::string::npos);
  EXPECT_LT(footprint_filter, footprint_sample);
  EXPECT_LT(footprint_sample, ordinal_point_load);
  EXPECT_LT(ordinal_point_load, ordinal_write);
  EXPECT_LT(ordinal_write, tone_map);
  EXPECT_NE(preprocess.find("float2 source_scale"), std::string::npos);
  EXPECT_NE(preprocess.find("float2 source_lo"), std::string::npos);
  EXPECT_NE(preprocess.find("float2 source_hi"), std::string::npos);
  EXPECT_NE(preprocess.find("x_coverage * y_coverage"), std::string::npos);
  EXPECT_NE(preprocess.find("weighted_sum / footprint_area"), std::string::npos);
  EXPECT_NE(history.find("PreviousAppearanceOrdinal[idx]"), std::string::npos);
  EXPECT_NE(history.find("PreviousReliableDepth[dtid.xy]"), std::string::npos);
  EXPECT_NE(history.find("CurrentDepth[dtid.xy]"), std::string::npos);
  EXPECT_NE(
    history.find(
      "SBS_STATE_MODEL_INPUT_HISTORY_STATE("
    ),
    std::string::npos
  );
  EXPECT_NE(estimator.find("depth_cut_history_srv.Get()"), std::string::npos);
  EXPECT_NE(estimator.find("depth_cut_history_uav.Get()"), std::string::npos);
  EXPECT_NE(estimator.find("ID3D11ShaderResourceView *history_srvs[5]"), std::string::npos);
  EXPECT_NE(estimator.find("ID3D11UnorderedAccessView *history_uavs[3]"), std::string::npos);
  EXPECT_NE(histogram.find("RAW_RGB_PIXEL_DELTA"), std::string::npos);
  EXPECT_NE(histogram.find("PlainHist[NUM_BINS + 3]"), std::string::npos);
  EXPECT_NE(histogram.find("PlainHist[NUM_BINS + 4]"), std::string::npos);
  EXPECT_NE(histogram.find("PlainHist[NUM_BINS + 5]"), std::string::npos);
  EXPECT_NE(histogram.find("PlainHist[NUM_BINS + 6]"), std::string::npos);
  EXPECT_NE(histogram.find("current_comparisons"), std::string::npos);
  EXPECT_NE(histogram.find("previous_comparisons"), std::string::npos);
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
  EXPECT_NE(resolve.find("common_structure_representative"), std::string::npos);
  EXPECT_NE(resolve.find("appearance_recovery_tail"), std::string::npos);
  EXPECT_NE(resolve.find("CUT_FLAG_APPEARANCE_RECOVERY"), std::string::npos);
  EXPECT_NE(resolve.find("structureless_transition"), std::string::npos);
  EXPECT_NE(resolve.find("same_scene_gap_return"), std::string::npos);
  EXPECT_NE(
    resolve.find("raw_rgb_change_fraction < STRUCTURELESS_RETURN_RGB_SAME_MAX"),
    std::string::npos
  );
  EXPECT_NE(resolve.find("next_model_input_history_state"), std::string::npos);
  EXPECT_NE(resolve.find("model_input_history_gap"), std::string::npos);
  EXPECT_NE(resolve.find("low_structure_scene"), std::string::npos);
  EXPECT_NE(resolve.find("raw_rgb_change_fraction >= RAW_RGB_CUT_HIGH"), std::string::npos);
  EXPECT_NE(
    resolve.find("structural_change_fraction < STRUCTURAL_COLOR_EXPOSURE_QUIET"),
    std::string::npos
  );
  EXPECT_NE(resolve.find("geometry_armed && !appearance_veto"), std::string::npos);
  const auto relative_geometry_latch =
    resolve.find("cut_latched && !geometry_armed &&");
  ASSERT_NE(relative_geometry_latch, std::string::npos);
  EXPECT_NE(
    resolve.find("!appearance_veto", relative_geometry_latch),
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
  EXPECT_NE(
    resolve.find("low_structure_scene && current_structure_reliable"),
    std::string::npos
  );
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
  EXPECT_NE(constants.find("#define STRUCTURAL_COLOR_MIN_SUPPORT 0.01f"), std::string::npos);
  EXPECT_NE(
    constants.find(
      "#define STRUCTURAL_COLOR_EXPOSURE_MIN_COMMON_RATIO 0.50f"
    ),
    std::string::npos
  );
  EXPECT_NE(constants.find("#define STRUCTURAL_COLOR_CUT_HIGH 0.03f"), std::string::npos);
  EXPECT_NE(
    constants.find("#define STRUCTURELESS_RETURN_RGB_SAME_MAX 0.01f"),
    std::string::npos
  );
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

TEST(DirectxShaderSourceTest, HostTelemetryReadbackIsNonblockingAndPreservesTheBenchmarkPrefix) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto resolve = read_source_file(shader_dir + "depth_subject_resolve_cs.hlsl");
  const auto reprojection = read_source_file(shader_dir + "sbs_reprojection_ps.hlsl");
  const auto estimator =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/video_depth_estimator.cpp");
  const auto display =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/platform/windows/display_vram.cpp");
  const auto harness =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/sbs_bench_harness.cpp");

  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(reprojection.empty());
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(display.empty());
  ASSERT_FALSE(harness.empty());

  // The live path extends SubjectState append-only. The first three float4 values are the
  // production warp and benchmark contract, so a telemetry change must never insert fields ahead
  // of them or make the offline harness consume the diagnostic tail.
  EXPECT_EQ(sbs_adaptive_state::word_count, 32u);
  EXPECT_EQ(sbs_adaptive_state::render_prefix_word_count, 12u);
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::word_count"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("sbs_adaptive_state::initial_values"),
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
  EXPECT_NE(harness.find("std::array<float, 12> values"), std::string::npos);
  EXPECT_NE(
    harness.find(
      "std::memcpy(values.data(), mapped.pData, values.size() * sizeof(float))"
    ),
    std::string::npos
  );
  const auto prefix_end = resolve.find(
    "SubjectState[SBS_STATE_VECTOR_ZERO_ANCHOR_SHIFT_PX] = s2;"
  );
  const auto diagnostic_begin = resolve.find(
    "SubjectState[SBS_STATE_VECTOR_LATCHED_EDGE_FRACTION] = telemetry;"
  );
  const auto current_diagnostic_end =
    resolve.find(
      "SubjectState[SBS_STATE_VECTOR_CURRENT_EDGE_FRACTION] = current_diagnostics;"
    );
  const auto analysis_diagnostic_end =
    resolve.find(
      "SubjectState[SBS_STATE_VECTOR_CURRENT_STRUCTURAL_SUPPORT_FRACTION] ="
    );
  ASSERT_NE(prefix_end, std::string::npos);
  ASSERT_NE(diagnostic_begin, std::string::npos);
  ASSERT_NE(current_diagnostic_end, std::string::npos);
  ASSERT_NE(analysis_diagnostic_end, std::string::npos);
  EXPECT_LT(prefix_end, diagnostic_begin);
  EXPECT_LT(diagnostic_begin, current_diagnostic_end);
  EXPECT_LT(current_diagnostic_end, analysis_diagnostic_end);
  EXPECT_EQ(reprojection.find("SubjectState[3]"), std::string::npos);
  EXPECT_EQ(reprojection.find("SubjectState[6]"), std::string::npos);
  EXPECT_EQ(reprojection.find("SubjectState[7]"), std::string::npos);

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
  const auto due_gate = display.find("const bool due =");
  const auto enabled_gate = display.find("enabled &&", due_gate);
  const auto telemetry_call =
    display.find("depth_estimator->poll_depth_telemetry(", enabled_gate);
  const auto due_argument = display.find("due,", telemetry_call);
  ASSERT_NE(due_gate, std::string::npos);
  ASSERT_NE(enabled_gate, std::string::npos);
  ASSERT_NE(telemetry_call, std::string::npos);
  ASSERT_NE(due_argument, std::string::npos);
  EXPECT_LT(due_gate, enabled_gate);
  EXPECT_LT(enabled_gate, telemetry_call);
  EXPECT_LT(telemetry_call, due_argument);
  const auto output_end = display.find("end_sbs_gpu_timer(gpu_timer);");
  const auto telemetry_poll = display.find("poll_sbs_telemetry_after_output();", output_end);
  ASSERT_NE(output_end, std::string::npos);
  ASSERT_NE(telemetry_poll, std::string::npos);
  EXPECT_LT(output_end, telemetry_poll);

  // Readiness bits must agree with the runtime flags. Otherwise the shared client history records
  // sentinel/uninitialized values as real chart samples (notably anchor=0 and subject=0).
  const auto profile_ready = display.find("if (sample.profile_initialized)");
  const auto edge_ready = display.find("if (sample.edge_fraction >= 0.0f)", profile_ready);
  const auto anchor_ready = display.find("if (sample.anchor_valid)", edge_ready);
  const auto cut_flags = display.find(
    "sbs_adaptive_state::cut_flag_geometry_armed",
    anchor_ready
  );
  ASSERT_NE(profile_ready, std::string::npos);
  ASSERT_NE(edge_ready, std::string::npos);
  ASSERT_NE(anchor_ready, std::string::npos);
  ASSERT_NE(cut_flags, std::string::npos);
  EXPECT_LT(
    display.find("sbs_telemetry_valid_field::subject", profile_ready),
    edge_ready
  );
  EXPECT_LT(
    display.find("sbs_telemetry_valid_field::edge", edge_ready),
    anchor_ready
  );
  EXPECT_LT(
    display.find("sbs_telemetry_valid_field::anchor", anchor_ready),
    cut_flags
  );
  EXPECT_NE(
    display.find("sbs_adaptive_state::cut_flag_appearance_armed", cut_flags),
    std::string::npos
  );
  EXPECT_EQ(
    display.find("constexpr std::uint32_t cut_flag_geometry_armed"),
    std::string::npos
  );
}

TEST(DirectxShaderSourceTest, WholeClipReplayKeepsCanonicalStateAndUsesAnOfflineCameraCbuffer) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  const auto warp_common =
    read_source_file(shader_dir + "include/sbs_warp_common.hlsl");
  const auto resolve =
    read_source_file(shader_dir + "depth_subject_resolve_cs.hlsl");
  const auto adaptive_contract_hlsl = read_source_file(
    shader_dir + "include/sbs_adaptive_state_contract.generated.hlsl"
  );
  const auto harness =
    read_source_file(SUNSHINE_SOURCE_DIR "/src/sbs_bench_harness.cpp");

  ASSERT_FALSE(warp_common.empty());
  ASSERT_FALSE(resolve.empty());
  ASSERT_FALSE(adaptive_contract_hlsl.empty());
  ASSERT_FALSE(harness.empty());
  EXPECT_NE(
    warp_common.find("#ifdef SBS_SCENE_CAMERA_OVERRIDE"),
    std::string::npos
  );
  EXPECT_NE(
    warp_common.find("SceneCameraConstants : register(b3)"),
    std::string::npos
  );
  EXPECT_NE(
    warp_common.find("strength = scene_absolute_pop_strength;"),
    std::string::npos
  );
  EXPECT_NE(
    warp_common.find("p.anchor_shift_px = scene_zero_anchor_shift_px;"),
    std::string::npos
  );
  EXPECT_EQ(sbs_adaptive_state::render_prefix_word_count, 12u);
  EXPECT_NE(harness.find("render_prefix_word_count"), std::string::npos);
  EXPECT_NE(
    harness.find("--scene-cache and --render-cache are mutually exclusive"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("--render-cache requires conversion artifacts and --scene-plan"),
    std::string::npos
  );
  EXPECT_NE(harness.find("atomic_replace_attempts = 50"), std::string::npos);
  EXPECT_NE(harness.find("ERROR_SHARING_VIOLATION"), std::string::npos);
  EXPECT_NE(harness.find("ERROR_LOCK_VIOLATION"), std::string::npos);
  EXPECT_NE(
    harness.find("\"state\", {\n          {\"schema\", 1}"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("{\"word_count\", render_state_words_t {}.size()}"),
    std::string::npos
  );
  for (const auto *packed_field : {
         "packed_sbs",
         "output_sbs_width",
         "output_sbs_height",
         "output_frame_format",
         "output_file_extension",
       }) {
    EXPECT_NE(harness.find(packed_field), std::string::npos);
  }
  EXPECT_NE(
    harness.find("if (!replay_mode) {\n      estimator = std::make_unique"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find(
      "const auto *warp_macros = replay_mode ? scene_camera_macros : nullptr"
    ),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("\"frame_%010d.depth.r32f\""),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("\"frame_%010d.state.u32\""),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("\"absolute_pop_strength\""),
    std::string::npos
  );
  EXPECT_EQ(sbs_adaptive_state::schema_version, 3u);
  EXPECT_EQ(
    sbs_adaptive_state::fields[
      sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::current_structural_support_fraction
      )
    ].name,
    "current_structural_support_fraction"
  );
  EXPECT_EQ(
    sbs_adaptive_state::fields[
      sbs_adaptive_state::index(sbs_adaptive_state::word_e::analysis_flags)
    ].json_type,
    "uint32"
  );
  EXPECT_EQ(sbs_adaptive_state::analysis_flag_bits.size(), 6u);
  for (const auto *definition : {
         "#define ANALYSIS_FLAG_APPEARANCE_PROPOSAL 1u",
         "#define ANALYSIS_FLAG_EXPOSURE_LIKE 2u",
         "#define ANALYSIS_FLAG_STRUCTURELESS 4u",
         "#define ANALYSIS_FLAG_SAME_RETURN 8u",
         "#define ANALYSIS_FLAG_VETO 16u",
         "#define ANALYSIS_FLAG_RELATIVE_SPIKE 32u",
       }) {
    EXPECT_NE(adaptive_contract_hlsl.find(definition), std::string::npos);
  }
  EXPECT_NE(
    resolve.find(
      "SBS_STATE_ANALYSIS_FLAGS(analysis_diagnostics) = (float)analysis_flags"
    ),
    std::string::npos
  );
  EXPECT_EQ(
    resolve.find(
      "SBS_STATE_ANALYSIS_FLAGS(analysis_diagnostics) = asfloat(analysis_flags)"
    ),
    std::string::npos
  );
  EXPECT_NE(
    harness.find(
      "replay_mode ? replay_first_sequence + fi : fi + 1u"
    ),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("tensorrt_enqueue_count"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("depth_inference_enabled"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("scheduled_depth_update_count"),
    std::string::npos
  );
  EXPECT_NE(
    harness.find("\"scene-cache-contract-schema-1:R32_FLOAT\""),
    std::string::npos
  );
  EXPECT_EQ(
    harness.find("words[7] = std::bit_cast<std::uint32_t>"),
    std::string::npos
  );
  EXPECT_EQ(
    harness.find("words[8] = std::bit_cast<std::uint32_t>"),
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

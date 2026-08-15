/**
 * @file tests/unit/platform/test_windows_adaptive_motion_gate_fixture.cpp
 * @brief Deterministic small-object and phase witnesses for adaptive Host SBS gating.
 */
#include "../../tests_common.h"

#ifdef _WIN32
  #include <algorithm>
  #include <chrono>
  #include <cstdint>
  #include <fstream>
  #include <nlohmann/json.hpp>
  #include <optional>
  #include <stdexcept>
  #include <string>
  #include <vector>
  // display.h declares the small D3DKMT function table without depending on the WDK headers.
  typedef long NTSTATUS;
  #include <src/platform/windows/display.h>

namespace {
  using json = nlohmann::json;
  using namespace std::chrono_literals;
  namespace adaptive = platf::dxgi::detail;

  json load_adaptive_motion_fixture() {
    const std::string path =
      SUNSHINE_SOURCE_DIR "/tests/fixtures/host_sbs/adaptive_motion_gate_v1.json";
    std::ifstream input {path};
    if (!input) {
      throw std::runtime_error("cannot open adaptive-motion fixture: " + path);
    }
    return json::parse(input);
  }

  std::uint64_t field_area(const json &fixture) {
    const auto &field = fixture.at("model_field");
    return field.at("width").get<std::uint64_t>() *
           field.at("height").get<std::uint64_t>();
  }

  bool rects_intersect(const std::vector<int> &left, const std::vector<int> &right) {
    return left.size() == 4u && right.size() == 4u &&
           left[0] < right[2] && left[2] > right[0] &&
           left[1] < right[3] && left[3] > right[1];
  }

  TEST(WindowsHostSbsAdaptiveMotionFixtureTest,
       LocalizedTinyObjectsNeverEnterTheBroadAdaptiveGate) {
    const auto fixture = load_adaptive_motion_fixture();
    ASSERT_EQ(fixture.at("schema"), 1);
    ASSERT_EQ(
      fixture.at("object_widths").get<std::vector<unsigned>>(),
      (std::vector<unsigned> {1u, 2u, 4u, 8u, 16u})
    );
    const auto area = field_area(fixture);
    ASSERT_EQ(area, 334180u);
    const auto events = fixture.at("events").get<std::vector<std::string>>();
    ASSERT_EQ(events, (std::vector<std::string> {"appear", "move", "disappear"}));

    for (const auto &object : fixture.at("objects")) {
      const auto width = object.at("width").get<std::uint64_t>();
      const auto object_area = width * width;
      const auto start = object.at("start").get<std::vector<int>>();
      const auto moved = object.at("moved").get<std::vector<int>>();
      ASSERT_EQ(start.size(), 2u);
      ASSERT_EQ(moved.size(), 2u);
      EXPECT_GE(moved[0], start[0] + static_cast<int>(width) + 4);

      for (const auto &event : events) {
        const auto changed_area = event == "move" ? 2u * object_area : object_area;
        const adaptive::ddup_damage_coverage_t localized {
          changed_area,
          area,
          true,
          object_area,
        };
        EXPECT_FALSE(adaptive::host_sbs_adaptive_motion_broad_damage_candidate(localized))
          << "width=" << width << " event=" << event;
        EXPECT_FALSE(adaptive::host_sbs_adaptive_motion_damage_candidate(localized, true))
          << "width=" << width << " event=" << event;
        // Every requested object is intentionally within the separate legacy 0.25% experiment.
        // This fixture isolates the adaptive gate; that older gate must remain disabled.
        EXPECT_TRUE(adaptive::host_sbs_low_motion_damage_candidate(localized, true))
          << "width=" << width << " event=" << event;

        const auto fraction = static_cast<float>(changed_area) /
                              static_cast<float>(area);
        for (const auto cut_flags :
             fixture.at("cadence").at("stable_cut_flags").get<std::vector<std::uint32_t>>()) {
          EXPECT_EQ(
            adaptive::host_sbs_adaptive_motion_verdict(
              true,
              true,
              false,
              false,
              1u,
              0u,
              cut_flags,
              8u,
              fraction,
              fraction,
              fraction
            ),
            adaptive::host_sbs_adaptive_motion_verdict_e::quiet
          ) << "width=" << width << " event=" << event << " cut_flags=" << cut_flags;
        }
      }
    }
  }

  TEST(WindowsHostSbsAdaptiveMotionFixtureTest,
       BroadSingleRectSeparatesDepthOpportunityFromOcrWork) {
    const auto fixture = load_adaptive_motion_fixture();
    const auto area = field_area(fixture);
    const adaptive::ddup_damage_coverage_t full_surface {
      area,
      area,
      true,
      area,
    };
    EXPECT_TRUE(adaptive::host_sbs_adaptive_motion_broad_damage_candidate(full_surface));
    EXPECT_TRUE(adaptive::host_sbs_adaptive_motion_damage_candidate(full_surface, true));
    EXPECT_FALSE(adaptive::host_sbs_adaptive_motion_damage_candidate(full_surface, false));
    EXPECT_FALSE(adaptive::host_sbs_adaptive_motion_sum_only_broad(full_surface));

    const adaptive::ddup_damage_coverage_t overlapping_sum_only {
      area,
      area,
      true,
      area / 4u,
    };
    EXPECT_FALSE(adaptive::host_sbs_adaptive_motion_broad_damage_candidate(
      overlapping_sum_only
    ));
    EXPECT_TRUE(adaptive::host_sbs_adaptive_motion_sum_only_broad(overlapping_sum_only));

    const auto &probe = fixture.at("ocr_probe");
    const auto crop = probe.at("crop").get<std::vector<int>>();
    const auto before = probe.at("subtitle_before").get<std::vector<int>>();
    const auto after = probe.at("subtitle_after").get<std::vector<int>>();
    EXPECT_NE(before, after);
    ASSERT_EQ(before.size(), 4u);
    ASSERT_EQ(after.size(), 4u);
    EXPECT_EQ(before[2] - before[0], after[2] - after[0]);
    EXPECT_EQ(before[3] - before[1], after[3] - after[1]);
    EXPECT_TRUE(rects_intersect(crop, before));
    EXPECT_TRUE(rects_intersect(crop, after));
    EXPECT_FALSE(fixture.at("damage_modes").at("broad_single_rect")
                   .at("ocr_crop_unchanged").get<bool>());
  }

  TEST(WindowsHostSbsAdaptiveMotionFixtureTest,
       BothCadencePhasesHoldEachTransitionExactlyOnce) {
    using decision_e = adaptive::host_sbs_adaptive_hold_decision_e;
    const auto fixture = load_adaptive_motion_fixture();
    const auto event_count = fixture.at("objects").size() * fixture.at("events").size();
    ASSERT_EQ(event_count, 15u);
    const auto frame_interval = std::chrono::milliseconds {
      fixture.at("cadence").at("frame_interval_ms").get<int>()
    };
    const auto phase_offsets =
      fixture.at("cadence").at("phase_offsets").get<std::vector<unsigned>>();
    ASSERT_EQ(phase_offsets, (std::vector<unsigned> {0u, 1u}));

    std::vector<std::vector<bool>> held_by_phase;
    for (const auto phase : phase_offsets) {
      adaptive::host_sbs_adaptive_hold_cadence_t cadence;
      const auto base = std::chrono::steady_clock::time_point {1000ms};
      cadence.record_successful_enqueue(base, base);
      unsigned tick = 0u;
      if (phase == 1u) {
        ++tick;
        const auto lead = base + tick * frame_interval;
        ASSERT_EQ(cadence.observe_changed(lead, true, lead), decision_e::hold_candidate);
        ASSERT_EQ(
          cadence.observe_changed(lead, true, lead + 1ms),
          decision_e::hold_same_identity
        );
      }

      std::vector<bool> held;
      for (std::size_t event = 0; event < event_count; ++event) {
        ++tick;
        const auto identity = base + tick * frame_interval;
        const auto decision = cadence.observe_changed(identity, true, identity);
        const bool is_held = decision == decision_e::hold_candidate;
        held.push_back(is_held);
        if (is_held) {
          EXPECT_EQ(
            cadence.observe_changed(identity, true, identity + 1ms),
            decision_e::hold_same_identity
          );
          EXPECT_TRUE(cadence.refresh_required());
        } else {
          ASSERT_EQ(decision, decision_e::infer);
          cadence.record_successful_enqueue(identity, identity);
        }
      }
      held_by_phase.push_back(std::move(held));
    }

    ASSERT_EQ(held_by_phase.size(), 2u);
    EXPECT_EQ(std::count(held_by_phase[0].begin(), held_by_phase[0].end(), true), 8);
    EXPECT_EQ(std::count(held_by_phase[1].begin(), held_by_phase[1].end(), true), 7);
    for (std::size_t event = 0; event < event_count; ++event) {
      EXPECT_NE(held_by_phase[0][event], held_by_phase[1][event]) << "event=" << event;
      if (event > 0u) {
        EXPECT_FALSE(held_by_phase[0][event - 1u] && held_by_phase[0][event]);
        EXPECT_FALSE(held_by_phase[1][event - 1u] && held_by_phase[1][event]);
      }
    }

    const auto hard_cut_indices =
      fixture.at("cadence").at("hard_cut_event_indices").get<std::vector<unsigned>>();
    ASSERT_EQ(hard_cut_indices.size(), phase_offsets.size());
    for (std::size_t phase = 0; phase < phase_offsets.size(); ++phase) {
      const auto event = hard_cut_indices[phase];
      ASSERT_LT(event, event_count);
      EXPECT_TRUE(held_by_phase[phase][event])
        << "the synthetic cut must land on a would-hold delivery";
      EXPECT_EQ(
        adaptive::host_sbs_adaptive_motion_verdict(
          true,
          true,
          false,
          true,
          1u,
          0u,
          3u,
          8u,
          0.0f,
          0.0f,
          0.0f
        ),
        adaptive::host_sbs_adaptive_motion_verdict_e::hard_cut
      );
    }
  }
}  // namespace
#endif

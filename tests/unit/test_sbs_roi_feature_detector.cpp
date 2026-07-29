/**
 * @file tests/unit/test_sbs_roi_feature_detector.cpp
 * @brief Tests for deterministic Host SBS feature-grid candidate extraction.
 */
#include "../tests_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <src/sbs_roi_feature_detector.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
  struct cell_rect_t {
    std::uint16_t left;
    std::uint16_t top;
    std::uint16_t right;
    std::uint16_t bottom;
  };

  struct feature_fixture_t {
    explicit feature_fixture_t(
      std::uint16_t grid_width = sbs_roi::nominal_feature_grid_width,
      std::uint16_t grid_height = sbs_roi::nominal_feature_grid_height
    ):
        width(grid_width),
        height(grid_height),
        cells(static_cast<std::size_t>(width) * height) {
    }

    [[nodiscard]] sbs_roi::feature_grid_view_t view() const {
      return {
        width,
        height,
        std::span<const sbs_roi::feature_cell_t>(cells),
      };
    }

    void paint(
      cell_rect_t rect,
      std::uint8_t activity,
      std::uint8_t photo,
      std::uint8_t gutter = 0,
      std::int8_t vertical_shift = 0,
      std::uint8_t shift_confidence = 0
    ) {
      ASSERT_LE(rect.left, rect.right);
      ASSERT_LE(rect.top, rect.bottom);
      ASSERT_LE(rect.right, width);
      ASSERT_LE(rect.bottom, height);
      for (std::uint16_t y = rect.top; y < rect.bottom; ++y) {
        for (std::uint16_t x = rect.left; x < rect.right; ++x) {
          auto &cell = at(x, y);
          cell.temporal_occupancy_q8 = activity;
          cell.photographic_density_q8 = photo;
          cell.gutter_stability_q8 = gutter;
          cell.vertical_shift_rows = vertical_shift;
          cell.vertical_shift_confidence_q8 = shift_confidence;
        }
      }
    }

    void paint_gutter(cell_rect_t rect, std::uint8_t stability = 255) {
      ASSERT_LE(rect.right, width);
      ASSERT_LE(rect.bottom, height);
      for (std::uint16_t y = rect.top; y < rect.bottom; ++y) {
        for (std::uint16_t x = rect.left; x < rect.right; ++x) {
          at(x, y).gutter_stability_q8 = stability;
        }
      }
    }

    void paint_scroll(
      cell_rect_t rect,
      std::int8_t vertical_shift,
      std::uint8_t confidence = 255
    ) {
      ASSERT_LE(rect.right, width);
      ASSERT_LE(rect.bottom, height);
      for (std::uint16_t y = rect.top; y < rect.bottom; ++y) {
        for (std::uint16_t x = rect.left; x < rect.right; ++x) {
          auto &cell = at(x, y);
          cell.vertical_shift_rows = vertical_shift;
          cell.vertical_shift_confidence_q8 = confidence;
        }
      }
    }

    void paint_horizontal_bridge(
      std::uint16_t left,
      std::uint16_t right,
      std::uint16_t y,
      std::uint8_t photo
    ) {
      ASSERT_LT(y, height);
      ASSERT_LE(right, width);
      for (std::uint16_t x = left; x < right; ++x) {
        at(x, y).photographic_density_q8 = photo;
      }
    }

    std::uint16_t width;
    std::uint16_t height;
    std::vector<sbs_roi::feature_cell_t> cells;

  private:
    sbs_roi::feature_cell_t &at(std::uint16_t x, std::uint16_t y) {
      return cells[static_cast<std::size_t>(y) * width + x];
    }
  };

  [[nodiscard]] sbs_roi::normalized_rect_t normalized(
    cell_rect_t rect,
    std::uint16_t width,
    std::uint16_t height
  ) {
    return {
      static_cast<float>(rect.left) / width,
      static_cast<float>(rect.top) / height,
      static_cast<float>(rect.right) / width,
      static_cast<float>(rect.bottom) / height,
    };
  }

  [[nodiscard]] float area(const sbs_roi::normalized_rect_t &rect) {
    return std::max(0.0f, rect.right - rect.left) *
           std::max(0.0f, rect.bottom - rect.top);
  }

  [[nodiscard]] float iou(
    const sbs_roi::normalized_rect_t &left,
    const sbs_roi::normalized_rect_t &right
  ) {
    const sbs_roi::normalized_rect_t intersection {
      std::max(left.left, right.left),
      std::max(left.top, right.top),
      std::min(left.right, right.right),
      std::min(left.bottom, right.bottom),
    };
    const auto intersection_area = area(intersection);
    const auto union_area = area(left) + area(right) - intersection_area;
    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
  }

  [[nodiscard]] std::vector<const sbs_roi::candidate_t *> candidates_of_kind(
    const sbs_roi::feature_detection_result_t &result,
    sbs_roi::roi_kind_e kind
  ) {
    std::vector<const sbs_roi::candidate_t *> candidates;
    for (const auto &candidate : result.candidates) {
      if (candidate.kind == kind) {
        candidates.push_back(&candidate);
      }
    }
    return candidates;
  }

  [[nodiscard]] const sbs_roi::candidate_t *best_candidate(
    const sbs_roi::feature_detection_result_t &result,
    sbs_roi::roi_kind_e kind,
    sbs_roi::normalized_rect_t expected
  ) {
    const sbs_roi::candidate_t *best = nullptr;
    float best_iou = -1.0f;
    for (const auto &candidate : result.candidates) {
      if (candidate.kind != kind) {
        continue;
      }
      const auto candidate_iou = iou(candidate.rect, expected);
      if (candidate_iou > best_iou) {
        best = &candidate;
        best_iou = candidate_iou;
      }
    }
    return best;
  }

  void expect_rect_near(
    const sbs_roi::normalized_rect_t &actual,
    cell_rect_t expected,
    std::uint16_t width,
    std::uint16_t height,
    float tolerance_cells = 1.01f
  ) {
    const auto expected_rect = normalized(expected, width, height);
    EXPECT_NEAR(actual.left, expected_rect.left, tolerance_cells / width);
    EXPECT_NEAR(actual.top, expected_rect.top, tolerance_cells / height);
    EXPECT_NEAR(actual.right, expected_rect.right, tolerance_cells / width);
    EXPECT_NEAR(actual.bottom, expected_rect.bottom, tolerance_cells / height);
  }

  void expect_rect_exact(
    const sbs_roi::normalized_rect_t &left,
    const sbs_roi::normalized_rect_t &right
  ) {
    EXPECT_FLOAT_EQ(left.left, right.left);
    EXPECT_FLOAT_EQ(left.top, right.top);
    EXPECT_FLOAT_EQ(left.right, right.right);
    EXPECT_FLOAT_EQ(left.bottom, right.bottom);
  }

  void expect_same_result(
    const sbs_roi::feature_detection_result_t &left,
    const sbs_roi::feature_detection_result_t &right
  ) {
    ASSERT_EQ(left.status, right.status);
    ASSERT_EQ(left.candidates.size(), right.candidates.size());
    for (std::size_t index = 0; index < left.candidates.size(); ++index) {
      const auto &a = left.candidates[index];
      const auto &b = right.candidates[index];
      EXPECT_EQ(a.kind, b.kind);
      expect_rect_exact(a.rect, b.rect);
      EXPECT_FLOAT_EQ(a.temporal_occupancy, b.temporal_occupancy);
      EXPECT_FLOAT_EQ(a.photographic_density, b.photographic_density);
      EXPECT_FLOAT_EQ(a.primary_column_support, b.primary_column_support);
      EXPECT_FLOAT_EQ(a.gutter_confidence, b.gutter_confidence);
      EXPECT_EQ(a.inside_primary_column, b.inside_primary_column);
      EXPECT_EQ(a.crosses_stable_gutter, b.crosses_stable_gutter);
      EXPECT_EQ(a.recent_interaction, b.recent_interaction);
    }

    ASSERT_EQ(left.gutters.size(), right.gutters.size());
    for (std::size_t index = 0; index < left.gutters.size(); ++index) {
      expect_rect_exact(left.gutters[index].rect, right.gutters[index].rect);
      EXPECT_FLOAT_EQ(
        left.gutters[index].confidence,
        right.gutters[index].confidence
      );
    }

    ASSERT_EQ(left.columns.size(), right.columns.size());
    for (std::size_t index = 0; index < left.columns.size(); ++index) {
      expect_rect_exact(left.columns[index].rect, right.columns[index].rect);
      EXPECT_FLOAT_EQ(
        left.columns[index].evidence,
        right.columns[index].evidence
      );
      EXPECT_EQ(left.columns[index].primary, right.columns[index].primary);
    }

    ASSERT_EQ(
      left.primary_column.has_value(),
      right.primary_column.has_value()
    );
    if (left.primary_column && right.primary_column) {
      expect_rect_exact(*left.primary_column, *right.primary_column);
    }
    EXPECT_EQ(
      left.primary_column_ambiguous,
      right.primary_column_ambiguous
    );
    EXPECT_EQ(left.broad_page_scroll, right.broad_page_scroll);
    EXPECT_FLOAT_EQ(left.vertical_scroll_rows, right.vertical_scroll_rows);
    EXPECT_FLOAT_EQ(left.scroll_confidence, right.scroll_confidence);
  }

  [[nodiscard]] sbs_roi::tracker_config_t fast_tracker_config() {
    sbs_roi::tracker_config_t config;
    config.acquire_updates = 2;
    config.acquire_duration_us = 100000;
    config.max_continuity_gap_us = 250000;
    return config;
  }

  [[nodiscard]] sbs_roi::observation_t tracker_observation(
    std::uint64_t id,
    const sbs_roi::feature_detection_result_t &detection
  ) {
    sbs_roi::observation_t observation;
    observation.id = id;
    observation.timestamp_us = id * 100000;
    observation.arrival_timestamp_us = observation.timestamp_us;
    observation.candidates = detection.candidates;
    observation.broad_page_scroll = detection.broad_page_scroll;
    return observation;
  }

  [[nodiscard]] float annotated_coverage(
    const sbs_roi::normalized_rect_t &candidate,
    const std::vector<cell_rect_t> &annotations,
    std::uint16_t width,
    std::uint16_t height
  ) {
    std::size_t total = 0;
    std::size_t covered = 0;
    for (const auto &annotation : annotations) {
      for (auto y = annotation.top; y < annotation.bottom; ++y) {
        for (auto x = annotation.left; x < annotation.right; ++x) {
          ++total;
          const auto center_x =
            (static_cast<float>(x) + 0.5f) / width;
          const auto center_y =
            (static_cast<float>(y) + 0.5f) / height;
          covered +=
            center_x >= candidate.left &&
                center_x < candidate.right &&
                center_y >= candidate.top &&
                center_y < candidate.bottom ?
              1 :
              0;
        }
      }
    }
    return total == 0 ?
             0.0f :
             static_cast<float>(covered) / total;
  }
}  // namespace

TEST(SbsRoiFeatureDetector, PrimaryVideoBeatsFasterBrighterSidebarAd) {
  feature_fixture_t fixture;
  constexpr cell_rect_t main {8, 9, 92, 63};
  constexpr cell_rect_t ad {101, 12, 125, 30};
  fixture.paint(main, 96, 168);
  fixture.paint(ad, 255, 255);
  fixture.paint_gutter({94, 0, 98, 72});

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());

  ASSERT_EQ(result.status, sbs_roi::feature_detection_status_e::accepted);
  const auto *main_candidate = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized(main, fixture.width, fixture.height)
  );
  const auto *ad_candidate = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized(ad, fixture.width, fixture.height)
  );
  ASSERT_NE(main_candidate, nullptr);
  ASSERT_NE(ad_candidate, nullptr);
  EXPECT_GT(
    iou(
      main_candidate->rect,
      normalized(main, fixture.width, fixture.height)
    ),
    0.90f
  );
  EXPECT_TRUE(main_candidate->inside_primary_column);
  EXPECT_FALSE(ad_candidate->inside_primary_column);
  EXPECT_GT(
    main_candidate->primary_column_support,
    ad_candidate->primary_column_support
  );
  ASSERT_TRUE(result.primary_column.has_value());
  EXPECT_LT(result.primary_column->right, 98.01f / fixture.width);
}

TEST(SbsRoiFeatureDetector, LargeEligibleSidebarIsStructurallyExcluded) {
  feature_fixture_t fixture;
  constexpr cell_rect_t main {6, 8, 88, 66};
  constexpr cell_rect_t ad {100, 8, 127, 66};
  fixture.paint(main, 96, 168);
  fixture.paint(ad, 255, 255);
  fixture.paint_gutter({92, 0, 97, 72});

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  const auto *main_candidate = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized(main, fixture.width, fixture.height)
  );
  const auto *ad_candidate = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized(ad, fixture.width, fixture.height)
  );

  ASSERT_NE(main_candidate, nullptr);
  ASSERT_NE(ad_candidate, nullptr);
  EXPECT_GT(area(ad_candidate->rect), 0.12f);
  EXPECT_TRUE(main_candidate->inside_primary_column);
  EXPECT_FALSE(ad_candidate->inside_primary_column);
  EXPECT_LT(ad_candidate->primary_column_support, 0.50f);
}

TEST(SbsRoiFeatureDetector, EqualSplitVideosRemainMultipleAndAmbiguous) {
  feature_fixture_t fixture;
  constexpr cell_rect_t left {4, 10, 61, 64};
  constexpr cell_rect_t right {67, 10, 124, 64};
  fixture.paint(left, 128, 192);
  fixture.paint(right, 128, 192);
  fixture.paint_gutter({62, 0, 66, 72});

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  const auto videos =
    candidates_of_kind(result, sbs_roi::roi_kind_e::video);

  ASSERT_EQ(result.status, sbs_roi::feature_detection_status_e::accepted);
  EXPECT_GE(videos.size(), 2u);
  EXPECT_NE(
    best_candidate(
      result,
      sbs_roi::roi_kind_e::video,
      normalized(left, fixture.width, fixture.height)
    ),
    nullptr
  );
  EXPECT_NE(
    best_candidate(
      result,
      sbs_roi::roi_kind_e::video,
      normalized(right, fixture.width, fixture.height)
    ),
    nullptr
  );
  EXPECT_TRUE(result.primary_column_ambiguous);
  EXPECT_FALSE(result.primary_column.has_value());
}

TEST(SbsRoiFeatureDetector, AmbiguousColumnsRemainIneligibleThroughTracker) {
  feature_fixture_t fixture;
  fixture.paint({4, 10, 61, 64}, 64, 96);
  fixture.paint({67, 10, 124, 64}, 255, 255);
  fixture.paint_gutter({62, 0, 66, 72});

  const auto detection =
    sbs_roi::feature_detector_t().detect(fixture.view());
  ASSERT_TRUE(detection.primary_column_ambiguous);
  ASSERT_GE(
    candidates_of_kind(detection, sbs_roi::roi_kind_e::video).size(),
    2u
  );
  for (const auto *candidate :
       candidates_of_kind(detection, sbs_roi::roi_kind_e::video)) {
    EXPECT_FALSE(candidate->inside_primary_column);
    EXPECT_FLOAT_EQ(candidate->primary_column_support, 0.0f);
  }

  sbs_roi::tracker_t tracker(fast_tracker_config());
  for (std::uint64_t id = 1; id <= 5; ++id) {
    tracker.update(tracker_observation(id, detection));
  }
  EXPECT_EQ(tracker.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
  EXPECT_EQ(tracker.snapshot().generation, 0u);
}

TEST(SbsRoiFeatureDetector, MultipleVideosWithoutStableGutterFailClosed) {
  feature_fixture_t fixture;
  fixture.paint({5, 8, 80, 64}, 96, 168);
  fixture.paint({96, 14, 125, 48}, 255, 255);

  const auto detection =
    sbs_roi::feature_detector_t().detect(fixture.view());
  const auto videos =
    candidates_of_kind(detection, sbs_roi::roi_kind_e::video);
  ASSERT_GE(videos.size(), 2u);
  EXPECT_TRUE(detection.gutters.empty());
  for (const auto *candidate : videos) {
    EXPECT_FALSE(candidate->inside_primary_column);
    EXPECT_FLOAT_EQ(candidate->gutter_confidence, 0.0f);
  }

  sbs_roi::tracker_t tracker(fast_tracker_config());
  for (std::uint64_t id = 1; id <= 5; ++id) {
    tracker.update(tracker_observation(id, detection));
  }
  EXPECT_EQ(tracker.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
}

TEST(SbsRoiFeatureDetector, ThinPhotoBridgeCannotMergeMainAndAd) {
  feature_fixture_t fixture;
  fixture.paint({5, 8, 80, 64}, 96, 168);
  fixture.paint({96, 10, 127, 50}, 255, 255);
  fixture.paint({80, 20, 96, 21}, 0, 24);

  const auto detection =
    sbs_roi::feature_detector_t().detect(fixture.view());
  const auto videos =
    candidates_of_kind(detection, sbs_roi::roi_kind_e::video);

  ASSERT_GE(videos.size(), 2u);
  for (const auto *candidate : videos) {
    EXPECT_FALSE(
      candidate->rect.left < 0.10f &&
      candidate->rect.right > 0.90f
    );
    EXPECT_FALSE(candidate->inside_primary_column);
  }
}

TEST(SbsRoiFeatureDetector, AnimatedAdCannotReplaceLargerPausedContent) {
  feature_fixture_t fixture;
  constexpr cell_rect_t paused_main {5, 8, 80, 64};
  constexpr cell_rect_t animated_ad {96, 10, 127, 50};
  fixture.paint(paused_main, 0, 168);
  fixture.paint(animated_ad, 255, 255);

  const auto detection =
    sbs_roi::feature_detector_t().detect(fixture.view());
  const auto *ad = best_candidate(
    detection,
    sbs_roi::roi_kind_e::video,
    normalized(animated_ad, fixture.width, fixture.height)
  );

  ASSERT_NE(ad, nullptr);
  EXPECT_GT(area(ad->rect), 0.12f);
  EXPECT_FALSE(ad->inside_primary_column);
  EXPECT_FLOAT_EQ(ad->gutter_confidence, 0.0f);

  sbs_roi::tracker_t tracker(fast_tracker_config());
  for (std::uint64_t id = 1; id <= 5; ++id) {
    tracker.update(tracker_observation(id, detection));
  }
  EXPECT_EQ(tracker.snapshot().kind, sbs_roi::roi_kind_e::full_frame);
}

TEST(SbsRoiFeatureDetector, SparseMotionSeedsGrowToPhotographicPlayer) {
  feature_fixture_t fixture;
  constexpr cell_rect_t player {10, 10, 90, 62};
  fixture.paint(player, 0, 168);
  fixture.paint({38, 18, 54, 46}, 128, 168);
  fixture.paint({62, 28, 70, 52}, 96, 168);

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  const auto *video = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized(player, fixture.width, fixture.height)
  );

  ASSERT_NE(video, nullptr);
  EXPECT_GT(
    iou(video->rect, normalized(player, fixture.width, fixture.height)),
    0.95f
  );
  expect_rect_near(
    video->rect,
    player,
    fixture.width,
    fixture.height
  );
}

TEST(SbsRoiFeatureDetector, OneFrameCutKeepsRoiGenerationStable) {
  feature_fixture_t steady;
  feature_fixture_t cut;
  constexpr cell_rect_t player {10, 10, 90, 62};
  steady.paint(player, 96, 168);
  cut.paint(player, 255, 255);
  const sbs_roi::feature_detector_t detector;
  const auto steady_detection = detector.detect(steady.view());
  const auto cut_detection = detector.detect(cut.view());
  sbs_roi::tracker_t tracker(fast_tracker_config());

  tracker.update(tracker_observation(1, steady_detection));
  const auto acquired =
    tracker.update(tracker_observation(2, steady_detection));
  ASSERT_EQ(acquired.kind, sbs_roi::roi_kind_e::video);
  const auto generation = acquired.generation;

  const auto after_cut =
    tracker.update(tracker_observation(3, cut_detection));
  EXPECT_EQ(after_cut.kind, sbs_roi::roi_kind_e::video);
  EXPECT_EQ(after_cut.generation, generation);
  EXPECT_FALSE(after_cut.committed_this_update);
}

TEST(SbsRoiFeatureDetector, TwoByTwoCollageFormsOneContentEnvelope) {
  feature_fixture_t fixture;
  for (const auto x : {cell_rect_t {18, 14, 48, 34}, cell_rect_t {52, 14, 82, 34}, cell_rect_t {18, 38, 48, 58}, cell_rect_t {52, 38, 82, 58}}) {
    fixture.paint(x, 0, 200);
  }

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  constexpr cell_rect_t envelope {17, 13, 83, 59};
  const auto *content = best_candidate(
    result,
    sbs_roi::roi_kind_e::content,
    normalized(envelope, fixture.width, fixture.height)
  );

  ASSERT_NE(content, nullptr);
  EXPECT_GT(
    iou(
      content->rect,
      normalized(envelope, fixture.width, fixture.height)
    ),
    0.88f
  );
  expect_rect_near(
    content->rect,
    envelope,
    fixture.width,
    fixture.height,
    2.01f
  );
}

TEST(SbsRoiFeatureDetector, ThreeByThreeCollageFormsOneContentEnvelope) {
  feature_fixture_t fixture;
  constexpr std::uint16_t lefts[] {12, 40, 68};
  constexpr std::uint16_t tops[] {10, 30, 50};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      const auto photo =
        static_cast<std::uint8_t>(128 + 24 * ((row + column) % 3));
      fixture.paint(
        {
          lefts[column],
          tops[row],
          static_cast<std::uint16_t>(lefts[column] + 24),
          static_cast<std::uint16_t>(tops[row] + 16),
        },
        0,
        photo
      );
    }
  }

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  constexpr cell_rect_t envelope {11, 9, 93, 67};
  const auto *content = best_candidate(
    result,
    sbs_roi::roi_kind_e::content,
    normalized(envelope, fixture.width, fixture.height)
  );

  ASSERT_NE(content, nullptr);
  EXPECT_GT(
    iou(
      content->rect,
      normalized(envelope, fixture.width, fixture.height)
    ),
    0.84f
  );
  EXPECT_LT(content->rect.right, 0.80f);
}

TEST(SbsRoiFeatureDetector, ThirteenCardCollageRetainsAnnotatedContent) {
  feature_fixture_t fixture;
  std::vector<cell_rect_t> cards;
  for (std::uint16_t row = 0; row < 3; ++row) {
    for (std::uint16_t column = 0; column < 4; ++column) {
      cards.push_back({
        static_cast<std::uint16_t>(24 + column * 18),
        static_cast<std::uint16_t>(12 + row * 14),
        static_cast<std::uint16_t>(39 + column * 18),
        static_cast<std::uint16_t>(23 + row * 14),
      });
    }
  }
  cards.push_back({24, 54, 39, 65});
  for (const auto &card : cards) {
    fixture.paint(card, 0, 200);
  }
  fixture.paint({0, 0, 128, 9}, 0, 160);
  fixture.paint({110, 18, 128, 66}, 0, 180);

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  const auto *content = best_candidate(
    result,
    sbs_roi::roi_kind_e::content,
    normalized({23, 11, 94, 66}, fixture.width, fixture.height)
  );

  ASSERT_NE(content, nullptr);
  EXPECT_GE(
    annotated_coverage(
      content->rect,
      cards,
      fixture.width,
      fixture.height
    ),
    0.95f
  );
  EXPECT_LT(content->rect.right, 0.85f);
}

TEST(SbsRoiFeatureDetector, WeakClippedOutlierDoesNotStretchEnvelope) {
  feature_fixture_t fixture;
  constexpr std::uint16_t lefts[] {12, 40, 68};
  constexpr std::uint16_t tops[] {10, 30, 50};
  for (const auto top : tops) {
    for (const auto left : lefts) {
      fixture.paint(
        {
          left,
          top,
          static_cast<std::uint16_t>(left + 24),
          static_cast<std::uint16_t>(top + 16),
        },
        0,
        220
      );
    }
  }
  fixture.paint_horizontal_bridge(92, 124, 64, 65);
  fixture.paint({124, 64, 128, 72}, 0, 65);

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  constexpr cell_rect_t core {11, 9, 93, 67};
  const auto *content = best_candidate(
    result,
    sbs_roi::roi_kind_e::content,
    normalized(core, fixture.width, fixture.height)
  );

  ASSERT_NE(content, nullptr);
  EXPECT_GT(
    iou(content->rect, normalized(core, fixture.width, fixture.height)),
    0.82f
  );
  EXPECT_LT(content->rect.right, 0.80f);
}

TEST(SbsRoiFeatureDetector, StableGutterIsAHardComponentBarrier) {
  feature_fixture_t fixture;
  constexpr cell_rect_t main {8, 8, 94, 64};
  constexpr cell_rect_t sidebar {98, 16, 124, 56};
  fixture.paint(main, 96, 200);
  fixture.paint(sidebar, 96, 200);
  fixture.paint({94, 35, 98, 37}, 0, 200);
  fixture.paint_gutter({94, 0, 98, 72});

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  ASSERT_EQ(result.status, sbs_roi::feature_detection_status_e::accepted);
  ASSERT_FALSE(result.gutters.empty());
  ASSERT_NE(
    best_candidate(
      result,
      sbs_roi::roi_kind_e::content,
      normalized(main, fixture.width, fixture.height)
    ),
    nullptr
  );
  ASSERT_NE(
    best_candidate(
      result,
      sbs_roi::roi_kind_e::content,
      normalized(sidebar, fixture.width, fixture.height)
    ),
    nullptr
  );

  constexpr float gutter_left = 94.0f / 128.0f;
  constexpr float gutter_right = 98.0f / 128.0f;
  for (const auto &candidate : result.candidates) {
    const bool straddles =
      candidate.rect.left < gutter_left &&
      candidate.rect.right > gutter_right;
    EXPECT_FALSE(straddles);
    EXPECT_FALSE(candidate.crosses_stable_gutter);
  }
}

TEST(SbsRoiFeatureDetector, CandidateUsesAdjacentSeparatorConfidence) {
  feature_fixture_t fixture;
  fixture.paint({48, 10, 82, 62}, 128, 192);
  fixture.paint({4, 12, 17, 60}, 0, 96);
  fixture.paint({94, 12, 124, 60}, 0, 96);
  fixture.paint_gutter({20, 0, 24, 72}, 255);
  fixture.paint_gutter({42, 0, 46, 72}, 180);
  fixture.paint_gutter({86, 0, 90, 72}, 208);

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());
  const auto *candidate = best_candidate(
    result,
    sbs_roi::roi_kind_e::video,
    normalized({48, 10, 82, 62}, fixture.width, fixture.height)
  );

  ASSERT_NE(candidate, nullptr);
  EXPECT_NEAR(candidate->gutter_confidence, 180.0f / 255.0f, 0.01f);
}

TEST(SbsRoiFeatureDetector, ToolbarAndSidebarStripsAreNotContent) {
  feature_fixture_t toolbar;
  toolbar.paint({0, 0, 128, 11}, 0, 220);
  feature_fixture_t sidebar;
  sidebar.paint({104, 8, 128, 68}, 0, 220);
  const sbs_roi::feature_detector_t detector;

  const auto toolbar_result = detector.detect(toolbar.view());
  const auto sidebar_result = detector.detect(sidebar.view());

  EXPECT_TRUE(
    candidates_of_kind(
      toolbar_result,
      sbs_roi::roi_kind_e::content
    )
      .empty()
  );
  EXPECT_TRUE(
    candidates_of_kind(
      sidebar_result,
      sbs_roi::roi_kind_e::content
    )
      .empty()
  );
}

TEST(SbsRoiFeatureDetector, PageScrollExcludesCommittedVideoPanVotes) {
  feature_fixture_t page_scroll;
  page_scroll.paint({0, 0, 128, 72}, 0, 96);
  page_scroll.paint_scroll({0, 0, 128, 72}, 3);
  feature_fixture_t player_pan;
  constexpr cell_rect_t player_cells {13, 11, 90, 61};
  player_pan.paint(player_cells, 96, 168);
  player_pan.paint_scroll(player_cells, 3);

  sbs_roi::feature_detector_context_t context;
  context.video_motion_exclusion = normalized(
    player_cells,
    player_pan.width,
    player_pan.height
  );
  const sbs_roi::feature_detector_t detector;
  const auto page_result =
    detector.detect(page_scroll.view(), std::nullopt, context);
  const auto pan_result =
    detector.detect(player_pan.view(), std::nullopt, context);

  EXPECT_TRUE(page_result.broad_page_scroll);
  EXPECT_NEAR(page_result.vertical_scroll_rows, 3.0f, 0.01f);
  EXPECT_GT(page_result.scroll_confidence, 0.80f);
  EXPECT_FALSE(pan_result.broad_page_scroll);
}

TEST(SbsRoiFeatureDetector, InconsistentSameDirectionFlowIsNotScroll) {
  feature_fixture_t fixture;
  fixture.paint({0, 0, 128, 72}, 0, 96);
  fixture.paint_scroll({0, 0, 64, 72}, 1);
  fixture.paint_scroll({64, 0, 128, 72}, 10);

  const auto result = sbs_roi::feature_detector_t().detect(fixture.view());

  EXPECT_FALSE(result.broad_page_scroll);
  EXPECT_LT(result.scroll_confidence, 1.0f);
}

TEST(SbsRoiFeatureDetector, FullscreenVideoExclusionSuppressesCameraPan) {
  feature_fixture_t fixture;
  fixture.paint({0, 0, 128, 72}, 128, 192);
  fixture.paint_scroll({0, 0, 128, 72}, -4);
  sbs_roi::feature_detector_context_t context;
  context.video_motion_exclusion =
    sbs_roi::normalized_rect_t {0.0f, 0.0f, 1.0f, 1.0f};

  const auto result = sbs_roi::feature_detector_t().detect(
    fixture.view(),
    std::nullopt,
    context
  );

  EXPECT_FALSE(result.broad_page_scroll);
  EXPECT_FLOAT_EQ(result.vertical_scroll_rows, 0.0f);
}

TEST(SbsRoiFeatureDetector, EmptyAndUniformFeatureMapsFailSafe) {
  const feature_fixture_t empty;
  feature_fixture_t uniform;
  uniform.paint({0, 0, 128, 72}, 255, 255);
  const sbs_roi::feature_detector_t detector;

  const auto empty_result = detector.detect(empty.view());
  const auto uniform_result = detector.detect(uniform.view());

  EXPECT_EQ(
    empty_result.status,
    sbs_roi::feature_detection_status_e::accepted
  );
  EXPECT_TRUE(empty_result.candidates.empty());
  EXPECT_EQ(
    uniform_result.status,
    sbs_roi::feature_detection_status_e::accepted
  );
  ASSERT_FALSE(uniform_result.candidates.empty());
  const auto *full_frame_video = best_candidate(
    uniform_result,
    sbs_roi::roi_kind_e::video,
    {0.0f, 0.0f, 1.0f, 1.0f}
  );
  ASSERT_NE(full_frame_video, nullptr);
  expect_rect_near(
    full_frame_video->rect,
    {0, 0, 128, 72},
    uniform.width,
    uniform.height
  );
  for (const auto &candidate : uniform_result.candidates) {
    EXPECT_GE(candidate.rect.left, 0.0f);
    EXPECT_GE(candidate.rect.top, 0.0f);
    EXPECT_LE(candidate.rect.right, 1.0f);
    EXPECT_LE(candidate.rect.bottom, 1.0f);
    EXPECT_LT(candidate.rect.left, candidate.rect.right);
    EXPECT_LT(candidate.rect.top, candidate.rect.bottom);
    EXPECT_TRUE(std::isfinite(candidate.temporal_occupancy));
    EXPECT_TRUE(std::isfinite(candidate.photographic_density));
  }
}

TEST(SbsRoiFeatureDetector, InvalidShapeFailsClosed) {
  const sbs_roi::feature_detector_t detector;
  const std::vector<sbs_roi::feature_cell_t> short_cells(127 * 72);
  const auto invalid = detector.detect(
    {
      128,
      72,
      std::span<const sbs_roi::feature_cell_t>(short_cells),
    }
  );
  const auto over_budget_shape = detector.detect(
    {
      std::numeric_limits<std::uint16_t>::max(),
      std::numeric_limits<std::uint16_t>::max(),
      std::span<const sbs_roi::feature_cell_t> {},
    }
  );

  EXPECT_EQ(
    invalid.status,
    sbs_roi::feature_detection_status_e::invalid_shape
  );
  EXPECT_TRUE(invalid.candidates.empty());
  EXPECT_EQ(
    over_budget_shape.status,
    sbs_roi::feature_detection_status_e::invalid_shape
  );
  EXPECT_TRUE(over_budget_shape.candidates.empty());
  EXPECT_TRUE(over_budget_shape.gutters.empty());
  EXPECT_TRUE(over_budget_shape.columns.empty());
  EXPECT_FALSE(over_budget_shape.primary_column.has_value());
  EXPECT_FALSE(over_budget_shape.broad_page_scroll);
}

TEST(SbsRoiFeatureDetector, InvalidInteractionAndContextFailClosed) {
  feature_fixture_t fixture;
  const sbs_roi::feature_detector_t detector;
  sbs_roi::feature_detector_context_t invalid_context;
  invalid_context.video_motion_exclusion =
    sbs_roi::normalized_rect_t {
      0.5f,
      0.2f,
      0.4f,
      0.8f,
    };

  const auto endpoint = detector.detect(
    fixture.view(),
    sbs_roi::normalized_point_t {1.0f, 0.5f}
  );
  const auto context = detector.detect(
    fixture.view(),
    std::nullopt,
    invalid_context
  );

  EXPECT_EQ(
    endpoint.status,
    sbs_roi::feature_detection_status_e::invalid_interaction
  );
  EXPECT_TRUE(endpoint.candidates.empty());
  EXPECT_EQ(
    context.status,
    sbs_roi::feature_detection_status_e::invalid_context
  );
  EXPECT_TRUE(context.candidates.empty());
}

TEST(SbsRoiFeatureDetector, UnsafeConfigurationIsRejected) {
  auto config = sbs_roi::feature_detector_config_t {};
  config.content_bridge_x_cells = 64;
  EXPECT_THROW(
    (void) sbs_roi::feature_detector_t {config},
    std::invalid_argument
  );

  config = {};
  config.content_min_axis_occupancy =
    std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(
    (void) sbs_roi::feature_detector_t {config},
    std::invalid_argument
  );
}

TEST(SbsRoiFeatureDetector, ComponentOverflowFailsClosed) {
  feature_fixture_t fixture;
  for (const auto left : {2, 16, 30, 44, 58}) {
    fixture.paint(
      {
        static_cast<std::uint16_t>(left),
        10,
        static_cast<std::uint16_t>(left + 4),
        14,
      },
      0,
      200
    );
  }
  sbs_roi::feature_detector_config_t config;
  config.max_components = 4;

  const auto result = sbs_roi::feature_detector_t(config).detect(
    fixture.view()
  );

  EXPECT_EQ(result.status, sbs_roi::feature_detection_status_e::overflow);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_FALSE(result.broad_page_scroll);
}

TEST(SbsRoiFeatureDetector, CandidateOverflowFailsClosed) {
  feature_fixture_t fixture;
  fixture.paint({6, 8, 88, 66}, 96, 168);
  fixture.paint({100, 8, 127, 66}, 255, 255);
  fixture.paint_gutter({92, 0, 97, 72});
  sbs_roi::feature_detector_config_t config;
  config.max_candidates = 1;

  const auto result = sbs_roi::feature_detector_t(config).detect(
    fixture.view()
  );

  EXPECT_EQ(result.status, sbs_roi::feature_detection_status_e::overflow);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(SbsRoiFeatureDetector, PortraitAndUltrawideRectsNormalizeIdentically) {
  feature_fixture_t portrait(40, 80);
  feature_fixture_t ultrawide(120, 40);
  constexpr cell_rect_t portrait_rect {4, 8, 32, 64};
  constexpr cell_rect_t ultrawide_rect {12, 4, 96, 32};
  portrait.paint(portrait_rect, 128, 192);
  ultrawide.paint(ultrawide_rect, 128, 192);
  const sbs_roi::feature_detector_t detector;

  const auto portrait_result = detector.detect(portrait.view());
  const auto ultrawide_result = detector.detect(ultrawide.view());
  const auto *portrait_video = best_candidate(
    portrait_result,
    sbs_roi::roi_kind_e::video,
    normalized(portrait_rect, portrait.width, portrait.height)
  );
  const auto *ultrawide_video = best_candidate(
    ultrawide_result,
    sbs_roi::roi_kind_e::video,
    normalized(ultrawide_rect, ultrawide.width, ultrawide.height)
  );

  ASSERT_NE(portrait_video, nullptr);
  ASSERT_NE(ultrawide_video, nullptr);
  expect_rect_near(
    portrait_video->rect,
    portrait_rect,
    portrait.width,
    portrait.height
  );
  expect_rect_near(
    ultrawide_video->rect,
    ultrawide_rect,
    ultrawide.width,
    ultrawide.height
  );
  expect_rect_exact(portrait_video->rect, ultrawide_video->rect);
}

TEST(SbsRoiFeatureDetector, DetectionIsCanonicallyRepeatable) {
  feature_fixture_t fixture;
  fixture.paint({8, 9, 92, 63}, 96, 168);
  fixture.paint({101, 12, 125, 30}, 255, 255);
  fixture.paint({22, 65, 48, 70}, 0, 180);
  fixture.paint_gutter({94, 0, 98, 72});
  fixture.paint_scroll({0, 0, 128, 8}, -2, 200);
  const sbs_roi::feature_detector_t detector;
  const auto baseline = detector.detect(
    fixture.view(),
    sbs_roi::normalized_point_t {0.4f, 0.5f}
  );

  for (int iteration = 0; iteration < 8; ++iteration) {
    expect_same_result(
      baseline,
      detector.detect(
        fixture.view(),
        sbs_roi::normalized_point_t {0.4f, 0.5f}
      )
    );
  }
}

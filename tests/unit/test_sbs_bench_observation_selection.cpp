#include "src/sbs_bench_harness.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#ifdef _WIN32
namespace {
  using sbs_bench::detail::parse_subtitle_diagnostic_observation_ranges;
  using sbs_bench::detail::subtitle_diagnostic_observation_range;
  using sbs_bench::detail::subtitle_diagnostic_observation_selected;
  using sbs_bench::detail::subtitle_diagnostic_should_render;
  using sbs_bench::detail::validate_subtitle_diagnostic_observation_ranges;

  TEST(SbsBenchObservationSelectionTest, AcceptsSingletonsInclusiveAndAdjacentRanges) {
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges(
      "0001,2-4,5-5,8,10-12", error
    );

    ASSERT_TRUE(ranges) << error;
    ASSERT_EQ(ranges->size(), 5u);
    EXPECT_EQ((*ranges)[0].first, 1u);
    EXPECT_EQ((*ranges)[0].last, 1u);
    EXPECT_EQ((*ranges)[1].first, 2u);
    EXPECT_EQ((*ranges)[1].last, 4u);
    EXPECT_EQ((*ranges)[2].first, 5u);
    EXPECT_EQ((*ranges)[2].last, 5u);
    EXPECT_EQ((*ranges)[3].first, 8u);
    EXPECT_EQ((*ranges)[3].last, 8u);
    EXPECT_EQ((*ranges)[4].first, 10u);
    EXPECT_EQ((*ranges)[4].last, 12u);
    EXPECT_TRUE(validate_subtitle_diagnostic_observation_ranges(*ranges, 12u, error)) << error;
  }

  TEST(SbsBenchObservationSelectionTest, RejectsEmptySelectionAndEmptyCommaTerms) {
    for (const std::string_view text : {"", ",", ",1", "1,", "1,,2"}) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, RejectsZeroEndpoints) {
    for (const std::string_view text : {"0", "0000", "0-1", "1-0", "1,0", "1-2,0"}) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, RejectsDescendingRangeAndTermOrder) {
    for (const std::string_view text : {"4-3", "9,4", "5-8,1-2", "1-2,4,3"}) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, RejectsOverlappingOrRepeatedObservations) {
    for (const std::string_view text : {"1,1", "001,1", "1-3,3-4", "1-5,2-3", "1-5,5", "2,2-3"}) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, RejectsSignsWhitespaceAndTrailingGarbage) {
    for (const std::string_view text : {
           "-1", "+1", "1-+2", "1--2", "1-2-3", "1.0", "1x", "1-2x", "1,2junk",
           " ", " 1", "1 ", "\t1", "1\n", "1 -2", "1- 2", "1, 2"
         }) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, RejectsUnsignedEndpointOverflow) {
    for (const std::string_view text : {
           "18446744073709551616", "1-18446744073709551616",
           "18446744073709551616-18446744073709551617", "999999999999999999999999999999999999"
         }) {
      SCOPED_TRACE(std::string(text));
      std::string error;
      EXPECT_FALSE(parse_subtitle_diagnostic_observation_ranges(text, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, PreservesUint64BoundaryWithoutExpandingTheRange) {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges(
      "1-18446744073709551615", error
    );

    ASSERT_TRUE(ranges) << error;
    ASSERT_EQ(ranges->size(), 1u);
    EXPECT_EQ(ranges->front().first, 1u);
    EXPECT_EQ(ranges->front().last, maximum);
    EXPECT_TRUE(validate_subtitle_diagnostic_observation_ranges(*ranges, maximum, error)) << error;
    EXPECT_TRUE(subtitle_diagnostic_observation_selected(*ranges, maximum));
    EXPECT_FALSE(subtitle_diagnostic_observation_selected(*ranges, 0u));
    EXPECT_FALSE(validate_subtitle_diagnostic_observation_ranges(*ranges, maximum - 1u, error));
    EXPECT_FALSE(error.empty());
  }

  TEST(SbsBenchObservationSelectionTest, RejectsSelectionBeyondTheLimitedInputRun) {
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges("2-4,8", error);
    ASSERT_TRUE(ranges) << error;

    EXPECT_TRUE(validate_subtitle_diagnostic_observation_ranges(*ranges, 8u, error)) << error;
    EXPECT_FALSE(validate_subtitle_diagnostic_observation_ranges(*ranges, 7u, error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(validate_subtitle_diagnostic_observation_ranges(*ranges, 0u, error));
    EXPECT_FALSE(error.empty());
  }

  TEST(SbsBenchObservationSelectionTest, ValidatorRejectsMalformedDirectApiRanges) {
    const std::vector<std::vector<subtitle_diagnostic_observation_range>> invalid {
      {}, {{0u, 1u}}, {{1u, 0u}}, {{4u, 3u}}, {{2u, 2u}, {1u, 1u}},
      {{1u, 3u}, {3u, 4u}}, {{1u, 1u}, {1u, 1u}}
    };
    for (std::size_t index = 0; index < invalid.size(); ++index) {
      SCOPED_TRACE(index);
      std::string error;
      EXPECT_FALSE(validate_subtitle_diagnostic_observation_ranges(invalid[index], 10u, error));
      EXPECT_FALSE(error.empty());
    }
  }

  TEST(SbsBenchObservationSelectionTest, MembershipUsesSourceOrdinalWithoutOutputCadenceRounding) {
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges("2-4,8", error);
    ASSERT_TRUE(ranges) << error;
    ASSERT_TRUE(validate_subtitle_diagnostic_observation_ranges(*ranges, 10u, error)) << error;

    // Output stride 5 ordinarily renders observations 1 and 6. Selected readback must include
    // every consecutive observation 2..4, without rounding them to render samples or filenames.
    for (const std::uint64_t ordinal : {0u, 1u, 5u, 6u, 7u, 9u, 10u}) {
      SCOPED_TRACE(ordinal);
      EXPECT_FALSE(subtitle_diagnostic_observation_selected(*ranges, ordinal));
    }
    for (const std::uint64_t ordinal : {2u, 3u, 4u, 8u}) {
      SCOPED_TRACE(ordinal);
      EXPECT_TRUE(subtitle_diagnostic_observation_selected(*ranges, ordinal));
    }
  }

  TEST(SbsBenchObservationSelectionTest, DefaultRenderSelectionPreservesOrdinaryCadence) {
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges("2-4,8", error);
    ASSERT_TRUE(ranges) << error;
    for (std::uint64_t observation = 1u; observation <= 10u; ++observation) {
      SCOPED_TRACE(observation);
      const bool cadence = (observation - 1u) % 5u == 0u;
      EXPECT_EQ(subtitle_diagnostic_should_render(observation, 5u), cadence);
      EXPECT_EQ(subtitle_diagnostic_should_render(observation, 5u, &*ranges), cadence);
      EXPECT_EQ(subtitle_diagnostic_should_render(observation, 5u, &*ranges, false), cadence);
    }
  }

  TEST(SbsBenchObservationSelectionTest, OptInUnionsConsecutiveOnsetAndSparseCadenceOnce) {
    std::string error;
    const auto ranges = parse_subtitle_diagnostic_observation_ranges("64-68,113-118,146", error);
    ASSERT_TRUE(ranges) << error;
    ASSERT_TRUE(validate_subtitle_diagnostic_observation_ranges(*ranges, 150u, error)) << error;
    std::vector<std::uint64_t> rendered;
    for (std::uint64_t observation = 1u; observation <= 150u; ++observation) {
      if (subtitle_diagnostic_should_render(observation, 29u, &*ranges, true)) {
        rendered.push_back(observation);
      }
    }
    const std::vector<std::uint64_t> expected {
      1u, 30u, 59u, 64u, 65u, 66u, 67u, 68u, 88u,
      113u, 114u, 115u, 116u, 117u, 118u, 146u
    };
    EXPECT_EQ(rendered, expected);
  }

  TEST(SbsBenchObservationSelectionTest, RenderCadenceRejectsZeroAndPreservesUint64Boundary) {
    const std::vector<subtitle_diagnostic_observation_range> ranges {
      {1u, std::numeric_limits<std::uint64_t>::max()}
    };
    EXPECT_FALSE(subtitle_diagnostic_should_render(0u, 1u, &ranges, true));
    EXPECT_FALSE(subtitle_diagnostic_should_render(1u, 0u, &ranges, true));
    EXPECT_TRUE(subtitle_diagnostic_should_render(1u, 1u, nullptr, true));
    EXPECT_FALSE(subtitle_diagnostic_should_render(2u, 5u, nullptr, true));
    EXPECT_TRUE(subtitle_diagnostic_should_render(std::numeric_limits<std::uint64_t>::max(),
                                                5u, &ranges, true));
    EXPECT_EQ(subtitle_diagnostic_should_render(std::numeric_limits<std::uint64_t>::max(), 5u),
              (std::numeric_limits<std::uint64_t>::max() - 1u) % 5u == 0u);
  }
}  // namespace
#endif

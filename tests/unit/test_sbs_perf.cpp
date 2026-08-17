#include <gtest/gtest.h>

#include <src/sbs_perf.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {
  class sbs_perf_state_guard_t {
  public:
    sbs_perf_state_guard_t(): was_enabled_(sbs_perf::enabled()) {
      sbs_perf::set_enabled(true);
      sbs_perf::reset();
    }

    ~sbs_perf_state_guard_t() {
      sbs_perf::reset();
      sbs_perf::set_enabled(was_enabled_);
    }

  private:
    bool was_enabled_;
  };
}

TEST(SbsPerfTest, JsonReportsMinimumAndMeanWithoutChangingQuantiles) {
  sbs_perf_state_guard_t guard;
  sbs_perf::add_sample_ms("known", 1.0);
  sbs_perf::add_sample_ms("known", 2.0);
  sbs_perf::add_sample_ms("known", 9.0);

  const auto path = std::filesystem::temp_directory_path() /
                    "sunshine-sbs-perf-summary-test.json";
  ASSERT_TRUE(sbs_perf::dump_json(path.string()));
  std::ifstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.good());
  const std::string json {
    std::istreambuf_iterator<char> {stream},
    std::istreambuf_iterator<char> {}
  };
  stream.close();
  std::error_code ignored;
  std::filesystem::remove(path, ignored);

  EXPECT_NE(json.find("\"min_ms\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"mean_ms\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"p50_ms\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"p95_ms\": 9"), std::string::npos);
  EXPECT_NE(json.find("\"max_ms\": 9"), std::string::npos);
  EXPECT_NE(json.find("\"n\": 3"), std::string::npos);
}

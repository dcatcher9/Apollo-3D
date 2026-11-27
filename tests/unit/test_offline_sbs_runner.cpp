/** Admission failures must notify the native consumer before any GPU work. */
#include "src/offline_frame_transport.h"
#include "src/offline_sbs_runner.h"
#include "src/offline_sbs_wire_contract.h"

#include <chrono>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

TEST(NativeSbsRunnerTest, InvalidRequestsPublishAFailedTerminalWithoutAHeader) {
  using namespace offline_sbs;
  using namespace std::chrono_literals;
  native_pass_settings valid_shape {
    .source_identity_directory = "unused-source",
    .transient_output_directory = "unused-output",
    .observation_timeline = "unused-timeline",
    .source_format = raw_pixel_format::bgra8,
    .frame_count = 1,
    .operation = native_pass_operation::analyze,
    .transient_raster_hard_cap_bytes = 1024 * 1024,
  };
  std::vector<native_pass_settings> rejected;
  auto add = [&](auto mutate) {
    auto request = valid_shape;
    mutate(request);
    rejected.push_back(std::move(request));
  };
  add([](auto &request) {
    request.operation = static_cast<native_pass_operation>(42);
  });
  add([](auto &request) {
    request.source_format = raw_pixel_format::rgba16f;
  });
  add([](auto &request) {
    request.source_identity_directory.clear();
  });
  add([](auto &request) {
    request.transient_output_directory.clear();
  });
  add([](auto &request) {
    request.observation_timeline.clear();
  });
  add([](auto &request) {
    request.frame_count = 0;
  });
  add([](auto &request) {
    request.frame_count = 10000000000ull;
  });
  add([](auto &request) {
    request.transient_raster_hard_cap_bytes = 0;
  });
  add([](auto &request) {
    request.transient_raster_hard_cap_bytes *= 2;
  });

  for (std::size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    raw_frame_transport transport(valid_shape.transient_raster_hard_cap_bytes);
    wire::whole_clip_contract_t contract;
    EXPECT_EQ(run_native_pass(rejected[index], transport, contract), 2);
    EXPECT_TRUE(contract.model.empty()) << "a refused pass hands over no contract";
    const auto terminal = transport.completion.wait_finished(0ms);
    EXPECT_EQ(terminal.exit_code, 2);
    EXPECT_FALSE(terminal.error.empty());
    EXPECT_EQ(terminal.source_frame_count, 0u);
    EXPECT_EQ(terminal.adaptive_frame_count, 0u);
    EXPECT_EQ(terminal.sbs_frame_count, 0u);
    EXPECT_THROW(transport.completion.receive_header(0ms), std::runtime_error);
  }
}

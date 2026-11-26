/**
 * @file tests/integration/test_external_commands.cpp
 * @brief Integration test for running an external command through platf::run_command.
 */
#include "../tests_common.h"

// standard includes
#include <array>
#include <cstdio>
#include <format>
#include <string>

// lib includes
#include <boost/process/v1.hpp>

// local includes
#include "src/platform/common.h"

TEST(ExternalCommands, CommandThatCannotStartFails) {
  const auto env = boost::this_process::environment();
  boost::filesystem::path working_directory {SUNSHINE_SOURCE_DIR};
  const auto output_file = std::tmpfile();
  ASSERT_NE(output_file, nullptr);

  std::error_code ec;
  auto child = platf::run_command(
    false,  // not elevated
    false,  // not interactive
    "non_existent_command_12345",
    working_directory,
    env,
    output_file,
    ec,
    nullptr  // no process group
  );

  int exit_code = -1;
  if (!ec) {
    child.wait();
    exit_code = child.exit_code();
  }

  std::rewind(output_file);
  std::string output;
  std::array<char, 1024> buffer {};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), output_file)) {
    output += buffer.data();
  }
  std::fclose(output_file);

  EXPECT_TRUE(ec || exit_code != 0) << std::format("The command should have failed but succeeded\nOutput: {}", output);
}

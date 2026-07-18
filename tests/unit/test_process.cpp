/**
 * @file tests/unit/test_process.cpp
 * @brief Unit tests for application command handling.
 */
#include "../tests_common.h"

// standard includes
#include <string>

// lib includes
#include <boost/process/v1/environment.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/token_functions.hpp>

// local includes
#include "src/process.h"

TEST(ProcessTest, MalformedCommandDoesNotEscapeWorkingDirectoryResolution) {
  const std::string malformed_command {"command\\"};
  const auto env = boost::this_process::environment();

#ifdef _WIN32
  // split_winmain intentionally accepts a trailing backslash using Windows command-line rules.
  EXPECT_NO_THROW((void) boost::program_options::split_winmain(malformed_command));
  EXPECT_NO_THROW((void) proc::find_working_directory(malformed_command, env));
#else
  EXPECT_THROW(boost::program_options::split_unix(malformed_command), boost::escaped_list_error);
  EXPECT_NO_THROW({
    const auto working_directory = proc::find_working_directory(malformed_command, env);
    EXPECT_TRUE(working_directory.empty());
  });
#endif
}

#ifdef _WIN32
TEST(ProcessTest, PlatformLaunchDoesNotReuseStaleErrorCode) {
  const auto env = boost::this_process::environment();
  boost::filesystem::path working_directory;
  const auto stale_error = std::make_error_code(std::errc::permission_denied);
  std::error_code ec = stale_error;

  auto child = platf::run_command(
    false,
    false,
    "apollo_command_that_must_not_exist_7f42e31b",
    working_directory,
    env,
    nullptr,
    ec,
    nullptr
  );

  EXPECT_FALSE(child.valid());
  EXPECT_TRUE(ec);
  EXPECT_NE(ec, stale_error);
}
#endif

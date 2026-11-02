/**
 * @file tests/tests_main.cpp
 * @brief Entry point definition.
 */
#include "tests_common.h"
#include "tests_environment.h"
#include "tests_events.h"

#ifdef _WIN32
  #include "src/host_sbs_shader_cache.h"
  #include <cstdlib>
#endif

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new SunshineEnvironment);
  testing::UnitTest::GetInstance()->listeners().Append(new SunshineEventListener);
#ifdef _WIN32
  if (const auto *cache = std::getenv("SUNSHINE_TEST_HOST_SBS_SHADER_CACHE"); cache && *cache) {
    models::host_sbs_shader_cache::configure_persistent_cache(cache);
  }
#endif
  return RUN_ALL_TESTS();
}

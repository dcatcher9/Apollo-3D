/** Authenticated Host SBS shader groups and bounded bytecode preparation. */
#include <gtest/gtest.h>

#ifdef _WIN32
  #include "src/host_sbs_bytecode_groups.h"

  #include <atomic>
  #include <chrono>
  #include <condition_variable>
  #include <filesystem>
  #include <fstream>
  #include <mutex>
  #include <stdexcept>
  #include <string>
  #include <vector>
  #include <windows.h>

namespace {
  namespace groups = models::host_sbs_bytecode_groups;
  namespace cache = models::host_sbs_shader_cache;
  constexpr std::array<cache::shader_spec, 1u> fixture_specs {{{"fixture_pixel.hlsl", "main_ps", "ps_5_0"}}};

  class HostSbsBytecodeGroupsTest: public testing::Test {
  protected:
    std::filesystem::path directory;
    std::filesystem::path shader_root;
    std::string shader_identity;

    void SetUp() override {
      static std::atomic_uint64_t next_directory {0u};
      for (unsigned attempt = 0u; attempt < 100u && directory.empty(); ++attempt) {
        const auto candidate = std::filesystem::temp_directory_path() /
                               ("apollo-bytecode-groups-" + std::to_string(GetCurrentProcessId()) +
                                "-" + std::to_string(next_directory.fetch_add(1u)));
        if (std::filesystem::create_directory(candidate)) {
          directory = candidate;
        }
      }
      ASSERT_FALSE(directory.empty());
      shader_root = directory / "fixture-programs";
      std::filesystem::create_directories(shader_root);
      std::ofstream file(shader_root / "fixture_pixel.hlsl");
      file << "float4 main_ps(float4 position : SV_Position) : SV_Target { return float4(0.25, 0.5, 0.75, 1.0); }\n";
      file.close();
      ASSERT_TRUE(file.good());
      shader_identity = cache::source_closure_sha256(cache::snapshot_sources(shader_root, fixture_specs));
      ASSERT_EQ(shader_identity.size(), 64u);
    }

    void TearDown() override {
      if (!directory.empty()) {
        // Only the unique directory successfully reserved by this fixture is owned here.
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
      }
    }

    static std::string failure_text(const groups::bytecode_request_t &request) {
      try {
        groups::prepared_bytecode(request, "fixture");
      } catch (const std::runtime_error &error) {
        return error.what();
      }
      return {};
    }
  };

  TEST_F(HostSbsBytecodeGroupsTest, BytecodeWorkersAreBoundedAndPreserveRequestOrdering) {
    const auto sources = cache::snapshot_sources(shader_root, fixture_specs);
    ASSERT_TRUE(sources);
    constexpr std::size_t count = 24u;
    std::array<std::string, count> names;
    std::vector<groups::bytecode_request_t> requests;
    for (std::size_t i = 0u; i < count; ++i) {
      names[i] = std::to_string(i);
      requests.push_back({sources, {names[i], "main_ps", "ps_5_0"}, {}, {}});
    }
    std::atomic_size_t active {0u}, peak {0u}, entered {0u};
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic_bool barrier_completed {true};
    groups::prepare_bytecodes(std::span(requests), [&](const auto &, const auto &spec) -> cache::bytecode_t {
      const auto running = active.fetch_add(1u) + 1u;
      auto previous = peak.load();
      while (previous < running && !peak.compare_exchange_weak(previous, running)) {}
      const auto arrival = entered.fetch_add(1u) + 1u;
      if (arrival <= groups::maximum_bytecode_workers) {
        std::unique_lock lock(mutex);
        ready.notify_all();
        if (!ready.wait_for(lock, std::chrono::seconds {2}, [&]() {
              return entered.load() >= groups::maximum_bytecode_workers;
            })) {
          barrier_completed = false;
        }
      }
      const auto index = static_cast<unsigned char>(std::stoul(std::string(spec.filename)));
      active.fetch_sub(1u);
      return std::make_shared<const std::vector<unsigned char>>(std::initializer_list<unsigned char> {index});
    });
    EXPECT_TRUE(barrier_completed);
    EXPECT_EQ(peak.load(), groups::maximum_bytecode_workers);
    EXPECT_EQ(active.load(), 0u);
    EXPECT_EQ(entered.load(), count);
    for (std::size_t i = 0u; i < count; ++i) {
      SCOPED_TRACE(i);
      ASSERT_TRUE(requests[i].code);
      ASSERT_EQ(requests[i].code->size(), 1u);
      EXPECT_EQ(requests[i].code->front(), i);
      EXPECT_FALSE(requests[i].error);
    }
  }

  TEST_F(HostSbsBytecodeGroupsTest, BytecodeFailuresRemainPerRequestAndWorkersJoinBeforeReturning) {
    const auto sources = cache::snapshot_sources(shader_root, fixture_specs);
    ASSERT_TRUE(sources);
    std::array<std::string, 8u> names;
    std::array<std::atomic_uint, 8u> visits {};
    std::vector<groups::bytecode_request_t> requests;
    for (std::size_t i = 0u; i < names.size(); ++i) {
      names[i] = std::to_string(i);
      requests.push_back({sources, {names[i], "main_ps", "ps_5_0"}, {}, {}});
    }
    std::atomic_size_t active {0u};
    groups::prepare_bytecodes(std::span(requests), [&](const auto &, const auto &spec) -> cache::bytecode_t {
      struct active_scope_t {
        std::atomic_size_t &count;
        explicit active_scope_t(std::atomic_size_t &value):
            count(value) {
          count.fetch_add(1u);
        }
        ~active_scope_t() {
          count.fetch_sub(1u);
        }
      } scope(active);
      const auto index = std::stoul(std::string(spec.filename));
      visits[index].fetch_add(1u);
      if (index == 1u) {
        return {};
      }
      if (index == 2u) {
        return std::make_shared<const std::vector<unsigned char>>();
      }
      if (index == 3u) {
        throw std::runtime_error("deliberate resolver failure");
      }
      if (index == 4u) {
        throw 17;
      }
      return std::make_shared<const std::vector<unsigned char>>(1u, static_cast<unsigned char>(index));
    });
    EXPECT_EQ(active.load(), 0u);
    for (std::size_t i = 0u; i < visits.size(); ++i) {
      EXPECT_EQ(visits[i].load(), 1u) << i;
      if (i >= 1u && i <= 4u) {
        EXPECT_THROW(groups::prepared_bytecode(requests[i], "fixture"), std::runtime_error) << i;
      } else {
        EXPECT_NO_THROW(groups::prepared_bytecode(requests[i], "fixture")) << i;
      }
    }
    EXPECT_FALSE(requests[1u].error);
    EXPECT_FALSE(requests[2u].error);
    EXPECT_TRUE(requests[3u].error);
    EXPECT_TRUE(requests[4u].error);
    EXPECT_EQ(failure_text(requests[1u]), "Host SBS bytecode unavailable: fixture/main_ps");
    EXPECT_EQ(failure_text(requests[2u]), "Host SBS bytecode unavailable: fixture/main_ps");
    EXPECT_EQ(failure_text(requests[3u]), "Host SBS bytecode preparation failed: fixture/main_ps: deliberate resolver failure");
    EXPECT_EQ(failure_text(requests[4u]), "Host SBS bytecode preparation failed: fixture/main_ps");
    groups::prepare_bytecodes(std::span(requests), [](const auto &, const auto &) -> cache::bytecode_t {
      return std::make_shared<const std::vector<unsigned char>>(1u, 23u);
    });
    for (const auto &request : requests) {
      EXPECT_FALSE(request.error);
      ASSERT_TRUE(request.code);
      EXPECT_EQ(request.code->front(), 23u);
    }
  }

  TEST_F(HostSbsBytecodeGroupsTest, SourceAuthenticationAndMissingSnapshotsRejectBeforeResolution) {
    std::vector<groups::bytecode_request_t> requests;
    try {
      groups::authenticate_group("fixture", shader_root, fixture_specs, std::string(64u, '0'), requests);
      ADD_FAILURE() << "A different source closure was accepted";
    } catch (const std::runtime_error &error) {
      EXPECT_STREQ(error.what(), "Host SBS shader source closure mismatch: fixture");
    }
    EXPECT_TRUE(requests.empty());
    EXPECT_THROW(groups::authenticate_group("fixture", directory / "absent", fixture_specs, shader_identity, requests), std::runtime_error);
    EXPECT_TRUE(requests.empty());
    requests.push_back({{}, fixture_specs[0u], {}, {}});
    std::atomic_uint calls {0u};
    EXPECT_THROW(groups::prepare_bytecodes(std::span(requests), [&](const auto &, const auto &) -> cache::bytecode_t {
                   calls.fetch_add(1u);
                   return {};
                 }),
                 std::invalid_argument);
    EXPECT_EQ(calls.load(), 0u);
  }

  TEST_F(HostSbsBytecodeGroupsTest, AuthenticatedGroupAppendsItsSpecsAndResolvesTheCacheBytecode) {
    const auto root = directory / "compute";
    std::filesystem::create_directory(root);
    {
      std::ofstream file(root / "fixture_cs.hlsl");
      file << "RWStructuredBuffer<uint> output : register(u0);\n"
              "[numthreads(1,1,1)] void first_main() { output[0] = 17u; }\n"
              "[numthreads(1,1,1)] void second_main() { output[0] = 29u; }\n";
      ASSERT_TRUE(file.good());
    }
    const std::array specs {
      cache::shader_spec {"fixture_cs.hlsl", "first_main", "cs_5_0"},
      cache::shader_spec {"fixture_cs.hlsl", "second_main", "cs_5_0"},
    };
    const auto sources = cache::snapshot_sources(root, specs);
    ASSERT_TRUE(sources);
    std::vector<groups::bytecode_request_t> requests;
    requests.push_back({cache::snapshot_sources(shader_root, fixture_specs), fixture_specs[0u], {}, {}});
    const auto group = groups::authenticate_group("fixture", root, specs, cache::source_closure_sha256(sources), requests);
    EXPECT_EQ(group.name, "fixture");
    EXPECT_EQ(group.first, 1u);
    EXPECT_EQ(group.count, specs.size());
    ASSERT_EQ(requests.size(), 1u + specs.size());
    groups::prepare_bytecodes(std::span(requests));
    for (std::size_t i = 0u; i < specs.size(); ++i) {
      const auto &request = requests[group.first + i];
      EXPECT_EQ(request.spec.entrypoint, specs[i].entrypoint);
      EXPECT_EQ(groups::prepared_bytecode(request, group.name), cache::get(sources, specs[i]));
    }
  }
}  // namespace
#endif

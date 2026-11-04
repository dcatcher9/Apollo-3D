/** Authenticated renderer-role loading, independent of an estimator or queued probe. */
#include <gtest/gtest.h>

#ifdef _WIN32
  #include "src/host_sbs_current_outline_runtime_programs.h"

  #include <atomic>
  #include <condition_variable>
  #include <filesystem>
  #include <fstream>
  #include <future>
  #include <mutex>
  #include <stdexcept>
  #include <string>
  #include <thread>

namespace {
  namespace runtime = models::current_outline_runtime;
  namespace cache = models::host_sbs_shader_cache;
  using Microsoft::WRL::ComPtr;

  class HostSbsCurrentOutlineRuntimeTest: public testing::Test {
  protected:
    ComPtr<ID3D11Device> device;
    std::filesystem::path directory;
    std::filesystem::path completed;
    std::filesystem::path queued;
    std::string completed_identity;

    void SetUp() override {
      D3D_FEATURE_LEVEL level {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0u,
        nullptr,
        0u,
        D3D11_SDK_VERSION,
        &device,
        &level,
        nullptr
      )));
      ASSERT_GE(level, D3D_FEATURE_LEVEL_11_0);
      static std::atomic_uint64_t next_directory {0u};
      for (unsigned attempt = 0u; attempt < 100u && directory.empty(); ++attempt) {
        const auto candidate = std::filesystem::temp_directory_path() /
                               ("apollo-outline-runtime-" + std::to_string(GetCurrentProcessId()) +
                                "-" + std::to_string(next_directory.fetch_add(1u)));
        if (std::filesystem::create_directory(candidate)) {
          directory = candidate;
        }
      }
      ASSERT_FALSE(directory.empty());
      completed = directory / "completed";
      queued = directory / "queued";
      write_renderer(completed, "0.25");
      completed_identity = identity(completed);
      ASSERT_EQ(completed_identity.size(), 64u);
    }

    void TearDown() override {
      if (!directory.empty()) {
        // Only the unique directory successfully reserved by this fixture is owned here.
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
      }
    }

    static void write_renderer(const std::filesystem::path &root, const std::string &value) {
      std::filesystem::create_directories(root);
      std::ofstream file(root / "sbs_reprojection_v2_live_ps.hlsl");
      file << "float4 main_ps(float4 position : SV_Position) : SV_Target { return float4("
           << value << ", 0.5, 0.75, 1.0); }\n";
      ASSERT_TRUE(file.good());
    }

    static std::string identity(const std::filesystem::path &root) {
      const auto sources = cache::snapshot_sources(root, runtime::completed_specs);
      return cache::source_closure_sha256(sources);
    }
  };

  TEST_F(HostSbsCurrentOutlineRuntimeTest, CompletedRoleDoesNotRequireQueuedFiles) {
    runtime::pixel_programs_t programs;
    std::string error;
    ASSERT_FALSE(std::filesystem::exists(queued));
    ASSERT_TRUE(programs.load(device.Get(), completed, completed_identity, {}, {}, error)) << error;
    EXPECT_TRUE(programs.completed_ready());
    EXPECT_FALSE(programs.queued_ready());
    EXPECT_FALSE(std::filesystem::exists(queued));
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, ExplicitQueuedRoleRequiresAuthenticatedSource) {
    write_renderer(queued, "0.75");
    const auto queued_identity = identity(queued);
    runtime::pixel_programs_t programs;
    std::string error;
    ASSERT_FALSE(programs.load(device.Get(), completed, completed_identity, queued, std::string(64u, '0'), error));
    EXPECT_FALSE(programs.completed_ready());
    EXPECT_FALSE(programs.queued_ready());
    ASSERT_TRUE(programs.load(device.Get(), completed, completed_identity, queued, queued_identity, error)) << error;
    EXPECT_TRUE(programs.completed_ready());
    EXPECT_TRUE(programs.queued_ready());
    write_renderer(queued, "0.125");
    EXPECT_FALSE(programs.load(device.Get(), completed, completed_identity, queued, queued_identity, error));
    // A failed replacement does not discard the preceding fully prepared handle tuple.
    EXPECT_TRUE(programs.completed_ready());
    EXPECT_TRUE(programs.queued_ready());
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, PartialQueuedConfigurationAndInvalidCompletedStayClosed) {
    for (const bool provide_path : {false, true}) {
      runtime::pixel_programs_t programs;
      std::string error;
      EXPECT_FALSE(programs.load(device.Get(), completed, completed_identity, provide_path ? queued : std::filesystem::path {}, provide_path ? std::string {} : std::string(64u, '0'), error));
      EXPECT_FALSE(programs.completed_ready());
      EXPECT_FALSE(programs.queued_ready());
    }
    runtime::pixel_programs_t programs;
    std::string error;
    EXPECT_FALSE(programs.load(device.Get(), completed, std::string(64u, '0'), {}, {}, error));
    EXPECT_FALSE(programs.completed_ready());
    EXPECT_FALSE(programs.load(device.Get(), directory / "absent", completed_identity, {}, {}, error));
    EXPECT_FALSE(programs.completed_ready());
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, PreparedHandlesMoveFromWorkerAndRetireWithAbandonedResult) {
    const auto load = [owned_device = device, completed_path = completed, expected = completed_identity]() {
      runtime::pixel_programs_t programs;
      std::string error;
      if (!programs.load(owned_device.Get(), completed_path, expected, {}, {}, error)) {
        throw std::runtime_error(error);
      }
      return programs;
    };
    std::packaged_task<runtime::pixel_programs_t()> task(load);
    auto result = task.get_future();
    std::thread worker(std::move(task));
    worker.join();
    auto programs = result.get();
    ASSERT_TRUE(programs.completed_ready());
    ComPtr<ID3D11PixelShader> adopted;
    adopted.Attach(programs.take_completed());
    EXPECT_TRUE(adopted);
    EXPECT_FALSE(programs.completed_ready());
    EXPECT_FALSE(programs.queued_ready());

    // This is the unconsumed packaged-task result shape used by live pipeline teardown.
    std::packaged_task<runtime::pixel_programs_t()> abandoned_task(load);
    auto abandoned = abandoned_task.get_future();
    std::thread abandoned_worker(std::move(abandoned_task));
    abandoned_worker.join();
    std::exception_ptr retirement_error;
    std::thread retirement([&retirement_error, owned_result = std::move(abandoned)]() mutable {
      try {
        (void) owned_result.get();
      } catch (...) {
        retirement_error = std::current_exception();
      }
    });
    retirement.join();
    EXPECT_FALSE(retirement_error);
    // Releasing an unrelated future's result did not transfer or revoke the adopted COM handle.
    EXPECT_TRUE(adopted);
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, BytecodeWorkersAreBoundedAndPreserveRequestOrdering) {
    const auto sources = cache::snapshot_sources(completed, runtime::completed_specs);
    ASSERT_TRUE(sources);
    constexpr std::size_t count = 24u;
    std::array<std::string, count> names;
    std::vector<runtime::detail::bytecode_request_t> requests;
    for (std::size_t i = 0u; i < count; ++i) {
      names[i] = std::to_string(i);
      requests.push_back({sources, {names[i], "main_ps", "ps_5_0"}, {}, {}});
    }
    std::atomic_size_t active {0u}, peak {0u}, entered {0u};
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic_bool barrier_completed {true};
    runtime::detail::prepare_bytecodes(requests, [&](const auto &, const auto &spec) -> cache::bytecode_t {
      const auto running = active.fetch_add(1u) + 1u;
      auto previous = peak.load();
      while (previous < running && !peak.compare_exchange_weak(previous, running)) {}
      const auto arrival = entered.fetch_add(1u) + 1u;
      if (arrival <= 4u) {
        std::unique_lock lock(mutex);
        ready.notify_all();
        if (!ready.wait_for(lock, std::chrono::seconds {2}, [&]() {
              return entered.load() >= 4u;
            })) {
          barrier_completed = false;
        }
      }
      const auto index = static_cast<unsigned char>(std::stoul(std::string(spec.filename)));
      active.fetch_sub(1u);
      return std::make_shared<const std::vector<unsigned char>>(std::initializer_list<unsigned char> {index});
    });
    EXPECT_TRUE(barrier_completed);
    EXPECT_EQ(peak.load(), 4u);
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

  TEST_F(HostSbsCurrentOutlineRuntimeTest, BytecodeFailuresRemainPerRequestAndWorkersJoinBeforeReturning) {
    const auto sources = cache::snapshot_sources(completed, runtime::completed_specs);
    ASSERT_TRUE(sources);
    std::array<std::string, 8u> names;
    std::array<std::atomic_uint, 8u> visits {};
    std::vector<runtime::detail::bytecode_request_t> requests;
    for (std::size_t i = 0u; i < names.size(); ++i) {
      names[i] = std::to_string(i);
      requests.push_back({sources, {names[i], "main_ps", "ps_5_0"}, {}, {}});
    }
    std::atomic_size_t active {0u};
    runtime::detail::prepare_bytecodes(requests, [&](const auto &, const auto &spec) -> cache::bytecode_t {
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
        EXPECT_THROW(runtime::detail::prepared_bytecode(requests[i], "fixture"), std::runtime_error) << i;
      } else {
        EXPECT_NO_THROW(runtime::detail::prepared_bytecode(requests[i], "fixture")) << i;
      }
    }
    EXPECT_FALSE(requests[1u].error);
    EXPECT_FALSE(requests[2u].error);
    EXPECT_TRUE(requests[3u].error);
    EXPECT_TRUE(requests[4u].error);
    runtime::detail::prepare_bytecodes(requests, [](const auto &, const auto &) -> cache::bytecode_t {
      return std::make_shared<const std::vector<unsigned char>>(1u, 23u);
    });
    for (const auto &request : requests) {
      EXPECT_FALSE(request.error);
      ASSERT_TRUE(request.code);
      EXPECT_EQ(request.code->front(), 23u);
    }
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, SourceAuthenticationAndMissingSnapshotsRejectBeforeResolution) {
    std::vector<runtime::detail::bytecode_request_t> requests;
    EXPECT_THROW(runtime::detail::authenticate_group("fixture", completed, runtime::completed_specs, std::string(64u, '0'), requests), std::runtime_error);
    EXPECT_TRUE(requests.empty());
    EXPECT_THROW(runtime::detail::authenticate_group("fixture", directory / "absent", runtime::completed_specs, completed_identity, requests), std::runtime_error);
    EXPECT_TRUE(requests.empty());
    requests.push_back({{}, runtime::completed_specs[0u], {}, {}});
    std::atomic_uint calls {0u};
    EXPECT_THROW(runtime::detail::prepare_bytecodes(requests, [&](const auto &, const auto &) -> cache::bytecode_t {
                   calls.fetch_add(1u);
                   return {};
                 }),
                 std::invalid_argument);
    EXPECT_EQ(calls.load(), 0u);
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, ActualComputeBytecodeCreatesSequentialGroupsAndOptionalFailureKeepsMandatoryHandles) {
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
    std::vector<runtime::detail::bytecode_request_t> requests;
    const auto group = runtime::detail::authenticate_group("fixture", root, specs, cache::source_closure_sha256(sources), requests);
    runtime::detail::prepare_bytecodes(requests);
    for (std::size_t i = 0u; i < specs.size(); ++i) {
      EXPECT_EQ(requests[i].code, cache::get(sources, specs[i]));
    }
    std::array<ComPtr<ID3D11ComputeShader>, 3u> owned;
    std::size_t at = 0u;
    const auto pointers = runtime::detail::create_compute_group<2u>(device.Get(), group, requests, owned, at);
    EXPECT_EQ(at, 2u);
    EXPECT_EQ(pointers[0u], owned[0u].Get());
    EXPECT_EQ(pointers[1u], owned[1u].Get());
    EXPECT_NE(pointers[0u], nullptr);
    EXPECT_NE(pointers[1u], nullptr);
    requests.push_back({sources, specs[0u], {}, {}});
    const runtime::detail::bytecode_group_t optional {"optional-fixture", 2u, 1u};
    EXPECT_THROW(runtime::detail::create_compute_group<1u>(device.Get(), optional, requests, owned, at), std::runtime_error);
    EXPECT_EQ(at, 2u);
    EXPECT_EQ(pointers[0u], owned[0u].Get());
    EXPECT_EQ(pointers[1u], owned[1u].Get());
    EXPECT_FALSE(owned[2u]);
  }

  TEST_F(HostSbsCurrentOutlineRuntimeTest, OptionalCoverageHoleAndStabilityFailuresKeepDisjointOwnedSegments) {
    EXPECT_EQ(runtime::required_compute_shader_count, 51u);
    EXPECT_EQ(runtime::hole_compute_shader_offset, 54u);
    EXPECT_EQ(runtime::coverage_compute_shader_offset, 63u);
    EXPECT_EQ(runtime::unique_compute_shader_count, 66u);
    const auto root = directory / "optional-compute";
    std::filesystem::create_directory(root);
    {
      std::ofstream file(root / "fixture_cs.hlsl");
      file << "RWStructuredBuffer<uint> output : register(u0);\n"
              "[numthreads(1,1,1)] void main_cs() { output[0] = 41u; }\n";
      ASSERT_TRUE(file.good());
    }
    const std::array specs {cache::shader_spec {"fixture_cs.hlsl", "main_cs", "cs_5_0"}};
    const auto sources = cache::snapshot_sources(root, specs);
    ASSERT_TRUE(sources);
    const auto code = cache::get(sources, specs[0u]);
    ASSERT_TRUE(code);
    std::vector<runtime::detail::bytecode_request_t> requests(15u, {sources, specs[0u], code, {}});
    std::array<ComPtr<ID3D11ComputeShader>, runtime::unique_compute_shader_count> owned;
    const runtime::detail::bytecode_group_t core {"core", 0u, 1u};
    std::size_t at = 0u;
    const auto core_handle = runtime::detail::create_compute_group<1u>(device.Get(), core, requests, owned, at)[0u];
    const runtime::detail::bytecode_group_t stability {"stability", 0u, 3u}, hole {"hole", 3u, 9u}, coverage {"coverage", 12u, 3u};
    std::array<ID3D11ComputeShader *, 3u> stability_handles {};
    std::array<ID3D11ComputeShader *, 9u> hole_handles {};
    std::array<ID3D11ComputeShader *, 3u> coverage_handles {};
    std::string stability_error, hole_error, coverage_error;
    requests[1u].code.reset(); // Fail after one handle was created, exercising rollback.
    runtime::detail::create_optional_compute_group(device.Get(), stability, requests, owned,
      runtime::stability_compute_shader_offset, stability_handles, stability_error);
    ASSERT_FALSE(stability_error.empty());
    for (std::size_t i = 51u; i < 54u; ++i) EXPECT_FALSE(owned[i]);
    runtime::detail::create_optional_compute_group(device.Get(), hole, requests, owned,
      runtime::hole_compute_shader_offset, hole_handles, hole_error);
    ASSERT_TRUE(hole_error.empty()) << hole_error;
    for (auto *handle : hole_handles) EXPECT_NE(handle, nullptr);
    runtime::detail::create_optional_compute_group(device.Get(), coverage, requests, owned,
      runtime::coverage_compute_shader_offset, coverage_handles, coverage_error);
    ASSERT_TRUE(coverage_error.empty()) << coverage_error;
    const auto retained_coverage = coverage_handles;
    for (auto *handle : retained_coverage) EXPECT_NE(handle, nullptr);
    requests[1u].code = code;
    stability_error.clear();
    runtime::detail::create_optional_compute_group(device.Get(), stability, requests, owned,
      runtime::stability_compute_shader_offset, stability_handles, stability_error);
    ASSERT_TRUE(stability_error.empty()) << stability_error;
    const auto retained_stability = stability_handles;
    requests[5u].code.reset();
    runtime::detail::create_optional_compute_group(device.Get(), hole, requests, owned,
      runtime::hole_compute_shader_offset, hole_handles, hole_error);
    ASSERT_FALSE(hole_error.empty());
    for (std::size_t i = 54u; i < 63u; ++i) EXPECT_FALSE(owned[i]);
    for (std::size_t i = 0u; i < 3u; ++i) {
      EXPECT_EQ(owned[51u + i].Get(), retained_stability[i]);
      EXPECT_NE(retained_stability[i], nullptr);
      EXPECT_EQ(owned[63u + i].Get(), retained_coverage[i]);
    }
    EXPECT_EQ(owned[0u].Get(), core_handle);
    for (auto *handle : hole_handles) EXPECT_EQ(handle, nullptr);
    requests[5u].code = code;
    hole_error.clear();
    runtime::detail::create_optional_compute_group(device.Get(), hole, requests, owned,
      runtime::hole_compute_shader_offset, hole_handles, hole_error);
    ASSERT_TRUE(hole_error.empty()) << hole_error;
    const auto retained_hole = hole_handles;
    requests[13u].code.reset(); // A coverage rollback must retain both older optional groups.
    runtime::detail::create_optional_compute_group(device.Get(), coverage, requests, owned,
      runtime::coverage_compute_shader_offset, coverage_handles, coverage_error);
    ASSERT_FALSE(coverage_error.empty());
    for (std::size_t i = 63u; i < 66u; ++i) EXPECT_FALSE(owned[i]);
    for (std::size_t i = 0u; i < 3u; ++i) EXPECT_EQ(owned[51u + i].Get(), retained_stability[i]);
    for (std::size_t i = 0u; i < 9u; ++i) {
      EXPECT_EQ(owned[54u + i].Get(), retained_hole[i]);
      EXPECT_NE(retained_hole[i], nullptr);
    }
    for (auto *handle : coverage_handles) EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(owned[0u].Get(), core_handle);
  }
}  // namespace
#endif

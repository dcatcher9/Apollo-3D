/** @file Authenticated Host SBS shader groups and bounded parallel bytecode resolution. */
#pragma once
#ifdef _WIN32
#include "host_sbs_shader_cache.h"
#include <array>
#include <atomic>
#include <exception>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace models::host_sbs_bytecode_groups {
  namespace cache = host_sbs_shader_cache;
  inline constexpr std::size_t maximum_bytecode_workers = 4u;

  struct bytecode_request_t {
    cache::source_snapshot_t sources;
    cache::shader_spec spec;
    cache::bytecode_t code;
    std::exception_ptr error;
  };

  struct bytecode_group_t {
    std::string name;
    std::size_t first = 0u;
    std::size_t count = 0u;
  };

  inline bytecode_group_t authenticate_group(
    const std::string_view name,
    const std::filesystem::path &path,
    const std::span<const cache::shader_spec> specs,
    const std::string_view expected,
    std::vector<bytecode_request_t> &requests
  ) {
    const auto sources = cache::snapshot_sources(path, specs);
    if (!sources || expected.size() != 64u || cache::source_closure_sha256(sources) != expected) {
      throw std::runtime_error("Host SBS shader source closure mismatch: " + std::string(name));
    }
    const bytecode_group_t group {std::string(name), requests.size(), specs.size()};
    for (const auto &spec : specs) {
      requests.push_back({sources, spec, {}, {}});
    }
    return group;
  }

  // Every input snapshot is authenticated before this function is called. Only CPU bytecode
  // resolution runs in the bounded workers. Per-request errors keep one failed shader from
  // invalidating the other prepared bytecode.
  template<class Resolve>
  void prepare_bytecodes(const std::span<bytecode_request_t> requests, Resolve resolve) {
    for (const auto &request : requests) {
      if (!request.sources) {
        throw std::invalid_argument("Host SBS bytecode snapshot unavailable");
      }
    }
    for (auto &request : requests) {
      request.code.reset();
      request.error = {};
    }
    std::atomic_size_t next {0u};
    const auto run = [&]() noexcept {
      for (;;) {
        const auto index = next.fetch_add(1u, std::memory_order_relaxed);
        if (index >= requests.size()) {
          return;
        }
        auto &request = requests[index];
        try {
          request.code = resolve(request.sources, request.spec);
        } catch (...) {
          request.error = std::current_exception();
        }
      }
    };
    // The already tracked build worker participates as worker one. Scoped jthreads join on
    // success and exceptions, including a failure to create a later worker thread.
    std::array<std::jthread, maximum_bytecode_workers - 1u> workers;
    for (std::size_t i = 0u; i < workers.size() && i + 1u < requests.size(); ++i) {
      workers[i] = std::jthread(run);
    }
    run();
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  inline void prepare_bytecodes(const std::span<bytecode_request_t> requests) {
    prepare_bytecodes(requests, [](const auto &sources, const auto &spec) {
      return cache::get(sources, spec);
    });
  }

  inline const cache::bytecode_t &prepared_bytecode(const bytecode_request_t &request, const std::string_view group) {
    const auto role = std::string(group) + "/" + std::string(request.spec.entrypoint);
    if (request.error) {
      try {
        std::rethrow_exception(request.error);
      } catch (const std::exception &error) {
        throw std::runtime_error("Host SBS bytecode preparation failed: " + role + ": " + error.what());
      } catch (...) {
        throw std::runtime_error("Host SBS bytecode preparation failed: " + role);
      }
    }
    if (!request.code || request.code->empty()) {
      throw std::runtime_error("Host SBS bytecode unavailable: " + role);
    }
    return request.code;
  }
}  // namespace models::host_sbs_bytecode_groups
#endif

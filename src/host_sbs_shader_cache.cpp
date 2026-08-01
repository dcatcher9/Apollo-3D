/**
 * @file src/host_sbs_shader_cache.cpp
 * @brief Process-wide bytecode cache for the fixed-shape Host SBS depth shaders.
 */

#include "host_sbs_shader_cache.h"

#include "logging.h"

#include <algorithm>
#include <cstdint>
#include <d3dcompiler.h>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <wrl/client.h>

#ifdef _MSC_VER
  #pragma comment(lib, "d3dcompiler.lib")
#endif

namespace models::host_sbs_shader_cache {
  namespace {
    constexpr std::uint64_t fnv_offset = 14695981039346656037ull;
    constexpr std::uint64_t fnv_prime = 1099511628211ull;

    void hash_bytes(std::uint64_t &hash, const void *data, const std::size_t size) {
      const auto *bytes = static_cast<const unsigned char *>(data);
      for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= fnv_prime;
      }
    }

    void hash_string(std::uint64_t &hash, const std::string_view value) {
      const auto size = static_cast<std::uint64_t>(value.size());
      hash_bytes(hash, &size, sizeof(size));
      hash_bytes(hash, value.data(), value.size());
    }

    bool path_is_within(
      const std::filesystem::path &root,
      const std::filesystem::path &candidate
    ) {
      std::error_code ec;
      const auto relative = std::filesystem::relative(candidate, root, ec);
      if (ec || relative.empty()) {
        return !ec && candidate == root;
      }
      const auto first = *relative.begin();
      return first != ".." && !relative.is_absolute();
    }

    struct source_collector {
      std::filesystem::path root;
      std::set<std::filesystem::path> visited;
      std::uint64_t fingerprint = fnv_offset;

      bool collect(const std::filesystem::path &requested) {
        std::error_code ec;
        const auto path = std::filesystem::weakly_canonical(requested, ec);
        if (ec || !path_is_within(root, path)) {
          BOOST_LOG(error)
            << "Host SBS shader dependency is invalid or escapes the shader root: "
            << requested;
          return false;
        }
        if (!visited.insert(path).second) {
          return true;
        }

        std::ifstream input(path, std::ios::binary);
        if (!input.is_open()) {
          BOOST_LOG(error) << "Could not open Host SBS shader source " << path;
          return false;
        }
        const std::string source {
          std::istreambuf_iterator<char> {input},
          std::istreambuf_iterator<char> {}
        };
        hash_string(fingerprint, path.lexically_relative(root).generic_string());
        hash_string(fingerprint, source);

        static const std::regex quoted_include(
          R"(^\s*#\s*include\s*\"([^\"]+)\")"
        );
        std::istringstream lines(source);
        std::string line;
        while (std::getline(lines, line)) {
          std::smatch match;
          if (!std::regex_search(line, match, quoted_include)) {
            continue;
          }
          const std::filesystem::path relative_include = match[1].str();
          auto include_path = path.parent_path() / relative_include;
          if (!std::filesystem::exists(include_path)) {
            include_path = root / relative_include;
          }
          if (!collect(include_path)) {
            return false;
          }
        }
        return true;
      }
    };

    using cache_key = std::tuple<
      std::filesystem::path,
      std::uint64_t,
      std::string,
      std::string,
      std::string
    >;
    using cache_future = std::shared_future<bytecode_t>;

    struct owned_shader_spec {
      std::string filename;
      std::string entrypoint;
      std::string target;
    };

    std::mutex cache_mutex;
    std::map<cache_key, cache_future> cache;

    bool collect_fingerprint(
      const std::filesystem::path &root,
      const std::span<const owned_shader_spec> specs,
      std::uint64_t &fingerprint
    ) {
      source_collector collector;
      collector.root = root;
      for (const auto &spec : specs) {
        if (spec.filename.empty() || spec.entrypoint.empty() || spec.target.empty() ||
            !collector.collect(root / spec.filename)) {
          return false;
        }
        hash_string(collector.fingerprint, spec.filename);
        hash_string(collector.fingerprint, spec.entrypoint);
        hash_string(collector.fingerprint, spec.target);
      }
      fingerprint = collector.fingerprint;
      return true;
    }

    bytecode_t compile(
      const std::filesystem::path &path,
      const owned_shader_spec &spec
    ) {
      Microsoft::WRL::ComPtr<ID3DBlob> blob;
      Microsoft::WRL::ComPtr<ID3DBlob> errors;
      constexpr DWORD flags =
        D3DCOMPILE_ENABLE_STRICTNESS |
        D3DCOMPILE_OPTIMIZATION_LEVEL3;
      const auto status = D3DCompileFromFile(
        path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        spec.entrypoint.c_str(),
        spec.target.c_str(),
        flags,
        0,
        &blob,
        &errors
      );
      if (FAILED(status) || !blob) {
        BOOST_LOG(error)
          << "Host SBS shader compile failed (" << path << ", "
          << spec.entrypoint << "/" << spec.target << "): "
          << (errors ?
                static_cast<const char *>(errors->GetBufferPointer()) :
                "no compiler diagnostics");
        return {};
      }
      const auto *begin =
        static_cast<const unsigned char *>(blob->GetBufferPointer());
      return std::make_shared<const std::vector<unsigned char>>(
        begin,
        begin + blob->GetBufferSize()
      );
    }
  }  // namespace

  struct source_snapshot {
    std::filesystem::path root;
    std::uint64_t fingerprint = 0;
    std::vector<owned_shader_spec> specs;
  };

  source_snapshot_t snapshot_sources(
    const std::filesystem::path &shader_root,
    const std::span<const shader_spec> specs
  ) {
    std::error_code ec;
    const auto root = std::filesystem::weakly_canonical(shader_root, ec);
    if (ec || !std::filesystem::is_directory(root)) {
      BOOST_LOG(error) << "Host SBS shader root is unavailable: " << shader_root;
      return {};
    }

    std::vector<owned_shader_spec> owned_specs;
    owned_specs.reserve(specs.size());
    for (const auto &spec : specs) {
      if (spec.filename.empty() || spec.entrypoint.empty() || spec.target.empty()) {
        return {};
      }
      owned_specs.push_back({
        std::string(spec.filename),
        std::string(spec.entrypoint),
        std::string(spec.target),
      });
    }
    std::uint64_t fingerprint = 0;
    if (!collect_fingerprint(root, owned_specs, fingerprint)) {
      return {};
    }
    return std::make_shared<const source_snapshot>(source_snapshot {
      root,
      fingerprint,
      std::move(owned_specs),
    });
  }

  bytecode_t get(
    const source_snapshot_t &sources,
    const shader_spec &spec
  ) {
    if (!sources || spec.filename.empty() ||
        spec.entrypoint.empty() || spec.target.empty()) {
      return {};
    }
    const auto selected = std::find_if(
      sources->specs.begin(),
      sources->specs.end(),
      [&](const owned_shader_spec &candidate) {
        return candidate.filename == spec.filename &&
               candidate.entrypoint == spec.entrypoint &&
               candidate.target == spec.target;
      }
    );
    if (selected == sources->specs.end()) {
      BOOST_LOG(error)
        << "Host SBS shader was requested from a snapshot that does not own it: "
        << spec.filename;
      return {};
    }
    const cache_key key {
      sources->root,
      sources->fingerprint,
      selected->filename,
      selected->entrypoint,
      selected->target,
    };

    cache_future future;
    std::shared_ptr<std::promise<bytecode_t>> producer;
    {
      std::lock_guard lock(cache_mutex);
      if (const auto found = cache.find(key); found != cache.end()) {
        future = found->second;
      } else {
        producer = std::make_shared<std::promise<bytecode_t>>();
        future = producer->get_future().share();
        cache.emplace(key, future);
      }
    }

    if (producer) {
      auto bytecode = compile(sources->root / selected->filename, *selected);
      std::uint64_t current_fingerprint = 0;
      if (bytecode &&
          (!collect_fingerprint(
             sources->root,
             sources->specs,
             current_fingerprint
           ) ||
           current_fingerprint != sources->fingerprint)) {
        BOOST_LOG(warning)
          << "Host SBS shader sources changed during compilation; discarding the "
             "mixed-generation bytecode snapshot.";
        bytecode.reset();
      }
      producer->set_value(bytecode);
      if (!bytecode) {
        // A transient source/install problem may be repaired without restarting the process.
        std::lock_guard lock(cache_mutex);
        cache.erase(key);
      }
      return bytecode;
    }
    return future.get();
  }

  bool prewarm(const std::filesystem::path &assets_dir) {
    const auto sources = snapshot_sources(
      assets_dir / "shaders" / "directx",
      core_specs
    );
    if (!sources) {
      return false;
    }
    for (const auto &spec : core_specs) {
      if (!get(sources, spec)) {
        return false;
      }
    }
    BOOST_LOG(info)
      << "Prewarmed " << core_specs.size()
      << " fixed-shape Host SBS depth shaders.";
    return true;
  }
}  // namespace models::host_sbs_shader_cache

/**
 * @file src/host_sbs_shader_cache.cpp
 * @brief Process-wide bytecode cache for the fixed-shape Host SBS depth shaders.
 */

#include "host_sbs_shader_cache.h"

#include "crypto.h"
#include "logging.h"

#include <algorithm>
#include <cstdint>
#include <d3dcompiler.h>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
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
  struct owned_shader_spec {
    std::string filename;
    std::string entrypoint;
    std::string target;
  };

  struct source_snapshot {
    std::filesystem::path root;
    std::string closure_sha256;
    std::vector<owned_shader_spec> specs;
    std::map<std::string, std::string> sources;
    std::map<std::pair<std::string, std::string>, std::string> include_edges;
  };

  namespace {
    constexpr std::string_view source_closure_domain =
      "apollo-host-sbs-source-closure-v2\n";

    void append_u64_le(std::string &output, const std::uint64_t value) {
      for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xffu));
      }
    }

    void append_field(std::string &output, const std::string_view value) {
      append_u64_le(output, static_cast<std::uint64_t>(value.size()));
      output.append(value);
    }

    std::string standard_sha256_hex(const std::string_view bytes) {
      constexpr char digits[] = "0123456789abcdef";
      const auto digest = crypto::hash(bytes);
      std::string result;
      result.reserve(digest.size() * 2u);
      for (const auto byte : digest) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 0x0fu]);
      }
      return result;
    }

    std::string normalize_source_for_digest(const std::string_view source) {
      std::string normalized;
      normalized.reserve(source.size());
      for (std::size_t index = 0; index < source.size(); ++index) {
        if (source[index] != '\r') {
          normalized.push_back(source[index]);
          continue;
        }
        if (index + 1u < source.size() && source[index + 1u] == '\n') {
          ++index;
        }
        normalized.push_back('\n');
      }
      return normalized;
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
      std::map<std::string, std::string> sources;
      std::map<std::pair<std::string, std::string>, std::string> include_edges;

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
        const auto parent_key = path.lexically_relative(root).generic_string();
        sources.emplace(parent_key, source);

        static const std::regex any_include(R"(^\s*#\s*include\b)");
        static const std::regex quoted_include(
          R"(^\s*#\s*include\s*\"([^\"]+)\")"
        );
        // Digest and dependency discovery share one exact line-ending interpretation. Keep the
        // original owned bytes for compilation, but treat CRLF, bare CR, and LF identically here.
        std::istringstream lines(normalize_source_for_digest(source));
        std::string line;
        while (std::getline(lines, line)) {
          std::smatch match;
          if (!std::regex_search(line, match, quoted_include)) {
            if (std::regex_search(line, any_include)) {
              BOOST_LOG(error)
                << "Host SBS shader uses an unauthenticated non-quoted include in "
                << path << ": " << line;
              return false;
            }
            continue;
          }
          const std::filesystem::path relative_include = match[1].str();
          auto include_path = path.parent_path() / relative_include;
          if (!std::filesystem::exists(include_path)) {
            include_path = root / relative_include;
          }
          const auto resolved_include = std::filesystem::weakly_canonical(include_path, ec);
          if (ec || !path_is_within(root, resolved_include)) {
            BOOST_LOG(error)
              << "Host SBS shader include is invalid or escapes the shader root: "
              << include_path;
            return false;
          }
          const auto child_key = resolved_include.lexically_relative(root).generic_string();
          const auto [edge, inserted] = include_edges.emplace(
            std::pair {parent_key, match[1].str()}, child_key
          );
          if ((!inserted && edge->second != child_key) || !collect(resolved_include)) {
            return false;
          }
        }
        return true;
      }
    };

    using cache_key = std::tuple<
      std::filesystem::path,
      std::string,
      std::string,
      std::string,
      std::string
    >;
    using cache_future = std::shared_future<bytecode_t>;

    std::mutex cache_mutex;
    std::map<cache_key, cache_future> cache;

    bool collect_source_closure_sha256(
      const std::filesystem::path &root,
      const std::span<const owned_shader_spec> specs,
      std::string &sha256,
      std::map<std::string, std::string> &sources,
      std::map<std::pair<std::string, std::string>, std::string> &include_edges
    ) {
      source_collector collector;
      collector.root = root;
      for (const auto &spec : specs) {
        if (spec.filename.empty() || spec.entrypoint.empty() || spec.target.empty() ||
            !collector.collect(root / spec.filename)) {
          return false;
        }
      }
      std::string canonical {source_closure_domain};
      canonical.push_back('C');
      append_u64_le(canonical, shader_compile_flags);
      append_field(canonical, "macros:none");
      for (const auto &spec : specs) {
        canonical.push_back('S');
        append_field(canonical, spec.filename);
        append_field(canonical, spec.entrypoint);
        append_field(canonical, spec.target);
      }
      for (const auto &[edge, child] : collector.include_edges) {
        canonical.push_back('I');
        append_field(canonical, edge.first);
        append_field(canonical, edge.second);
        append_field(canonical, child);
      }
      for (const auto &[path, source] : collector.sources) {
        canonical.push_back('F');
        append_field(canonical, path);
        append_field(canonical, normalize_source_for_digest(source));
      }
      sha256 = standard_sha256_hex(canonical);
      sources = std::move(collector.sources);
      include_edges = std::move(collector.include_edges);
      return true;
    }

    class snapshot_include_t final: public ID3DInclude {
    public:
      snapshot_include_t(
        const source_snapshot &snapshot,
        const std::string_view root_filename
      ):
          snapshot_(snapshot),
          root_filename_(std::filesystem::path {root_filename}.lexically_normal()) {
        for (const auto &[path, source] : snapshot_.sources) {
          owners_.emplace(source.data(), path);
        }
      }

      HRESULT STDMETHODCALLTYPE Open(
        const D3D_INCLUDE_TYPE include_type,
        LPCSTR filename,
        LPCVOID parent_data,
        LPCVOID *data,
        UINT *bytes
      ) override {
        if (include_type != D3D_INCLUDE_LOCAL || !filename || !*filename ||
            !data || !bytes) {
          return E_FAIL;
        }
        std::filesystem::path parent_path =
          open_stack_.empty() ? root_filename_ : open_stack_.back().second;
        if (parent_data) {
          const auto parent = owners_.find(parent_data);
          if (parent == owners_.end()) {
            return E_FAIL;
          }
          parent_path = parent->second;
        }
        const std::filesystem::path requested {filename};
        if (requested.is_absolute()) {
          return E_FAIL;
        }
        const auto edge = snapshot_.include_edges.find({
          parent_path.lexically_normal().generic_string(),
          std::string {filename},
        });
        if (edge == snapshot_.include_edges.end()) {
          return E_FAIL;
        }
        const auto found = snapshot_.sources.find(edge->second);
        if (found == snapshot_.sources.end() ||
            found->second.size() > std::numeric_limits<UINT>::max()) {
          return E_FAIL;
        }
        *data = found->second.data();
        *bytes = static_cast<UINT>(found->second.size());
        open_stack_.emplace_back(*data, edge->second);
        return S_OK;
      }

      HRESULT STDMETHODCALLTYPE Close(const LPCVOID data) override {
        const auto opened = std::find_if(
          open_stack_.rbegin(),
          open_stack_.rend(),
          [data](const auto &entry) { return entry.first == data; }
        );
        if (opened == open_stack_.rend()) {
          return E_FAIL;
        }
        open_stack_.erase(std::next(opened).base());
        return S_OK;
      }

    private:
      const source_snapshot &snapshot_;
      std::filesystem::path root_filename_;
      std::map<const void *, std::string> owners_;
      std::vector<std::pair<const void *, std::string>> open_stack_;
    };

    bytecode_t compile(
      const source_snapshot &snapshot,
      const owned_shader_spec &spec
    ) {
      Microsoft::WRL::ComPtr<ID3DBlob> blob;
      Microsoft::WRL::ComPtr<ID3DBlob> errors;
      const auto root_source = snapshot.sources.find(
        std::filesystem::path {spec.filename}.lexically_normal().generic_string()
      );
      if (root_source == snapshot.sources.end()) {
        return {};
      }
      snapshot_include_t includes {snapshot, spec.filename};
      const auto status = D3DCompile(
        root_source->second.data(),
        root_source->second.size(),
        spec.filename.c_str(),
        nullptr,
        &includes,
        spec.entrypoint.c_str(),
        spec.target.c_str(),
        shader_compile_flags,
        0,
        &blob,
        &errors
      );
      if (FAILED(status) || !blob) {
        BOOST_LOG(error)
          << "Host SBS shader compile failed (" << spec.filename << ", "
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
    std::string closure_sha256;
    std::map<std::string, std::string> sources;
    std::map<std::pair<std::string, std::string>, std::string> include_edges;
    if (!collect_source_closure_sha256(
          root, owned_specs, closure_sha256, sources, include_edges)) {
      return {};
    }
    return std::make_shared<const source_snapshot>(source_snapshot {
      root,
      std::move(closure_sha256),
      std::move(owned_specs),
      std::move(sources),
      std::move(include_edges),
    });
  }

  std::string source_closure_sha256(const source_snapshot_t &sources) {
    return sources ? sources->closure_sha256 : std::string {};
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
      sources->closure_sha256,
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
      auto bytecode = compile(*sources, *selected);
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
    const auto shader_root = assets_dir / "shaders" / "directx";
    const auto preprocess_sources = snapshot_sources(
      shader_root,
      preprocess_specs
    );
    const auto sources = snapshot_sources(
      shader_root,
      core_specs
    );
    if (!preprocess_sources || !sources || !get(preprocess_sources, rgb_to_nchw)) {
      return false;
    }
    for (const auto &spec : core_specs) {
      if (spec.filename == rgb_to_nchw.filename) {
        continue;
      }
      if (!get(sources, spec)) {
        return false;
      }
    }

    const auto parallax_v2_sources = snapshot_sources(
      shader_root,
      parallax_v2_producer_specs
    );
    if (!parallax_v2_sources) {
      return false;
    }
    for (const auto &spec : parallax_v2_producer_specs) {
      if (!get(parallax_v2_sources, spec)) {
        return false;
      }
    }
    const auto parallax_v2_live_sources = snapshot_sources(
      shader_root,
      parallax_v2_live_renderer_specs
    );
    if (!parallax_v2_live_sources) {
      return false;
    }
    for (const auto &spec : parallax_v2_live_renderer_specs) {
      if (!get(parallax_v2_live_sources, spec)) {
        return false;
      }
    }
    BOOST_LOG(info)
      << "Prewarmed the complete Host SBS V2 shader set ("
      << core_specs.size() + parallax_v2_producer_specs.size() +
           parallax_v2_live_renderer_specs.size()
      << " production shaders; dump-only shaders remain lazy).";
    return true;
  }
}  // namespace models::host_sbs_shader_cache

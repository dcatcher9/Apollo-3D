// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <d3dcompiler.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <reshade.hpp>
#include "addon_lifetime.h"
#include "async_log.h"

// Compiled bytecode for the production Game 3D shader. D3DCompile of the whole
// source costs 100-300 ms per entry point, and the renderer used to run it on
// the game's Present at every (re)configuration: about 2 s at a 4K start and
// again after an HDR switch. Entry points now compile in parallel on the thread
// pool while the game stays 2D; blobs are kept for the process and persisted
// per user, so later launches only read them back. Explicit replay/test
// sources keep compiling synchronously and never touch this store.
namespace sunshine_game3d::shader_cache {
  using blob = std::vector<std::uint8_t>;
  struct entry_point {
    const char *name;
    const char *target;
  };
  // Everything that changes the bytecode besides the entry point.
  struct configuration {
    std::string_view source;
    unsigned width = 0, height = 0, color = 0;
  };
  inline constexpr UINT compile_flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;

  inline bool compile(const configuration &config, const entry_point &entry, blob &out) {
    const auto w = std::to_string(config.width), h = std::to_string(config.height), c = std::to_string(config.color);
    const D3D_SHADER_MACRO defines[]{{"BUFFER_WIDTH", w.c_str()}, {"BUFFER_HEIGHT", h.c_str()},
      {"BUFFER_COLOR_SPACE", c.c_str()}, {nullptr, nullptr}};
    ID3DBlob *code = nullptr, *errors = nullptr;
    const HRESULT result = D3DCompile(config.source.data(), config.source.size(), "Sunshine Game 3D", defines, nullptr,
      entry.name, entry.target, compile_flags, 0, &code, &errors);
    if (errors) {
      if (FAILED(result)) sunshine_log::message(reshade::log::level::error, static_cast<const char *>(errors->GetBufferPointer()));
      errors->Release();
    }
    if (FAILED(result) || !code) {
      if (code) code->Release();
      return false;
    }
    const auto *bytes = static_cast<const std::uint8_t *>(code->GetBufferPointer());
    out.assign(bytes, bytes + code->GetBufferSize());
    code->Release();
    return true;
  }

  namespace detail {
    struct job_t {
      configuration config;
      std::string source; // Owned copy; config.source points here.
      std::string key;
      std::vector<entry_point> entries;
      std::atomic<unsigned> remaining{0};
      std::atomic<bool> failed{false};
    };
    struct store_t {
      std::mutex mutex;
      std::map<std::string, blob> blobs;
      std::map<std::string, std::shared_ptr<job_t>> jobs;
      std::wstring directory;
      bool directory_checked = false;
    };
    // Deliberately leaked: a pool callback may finish during process exit.
    inline store_t &store() {
      static store_t *value = new store_t;
      return *value;
    }
    // Bytecode also depends on the compiler that produced it.
    inline std::string compiler_identity() {
      static const std::string value = [] {
        // By name: &D3DCompile is an import thunk inside this add-on.
        const HMODULE module = GetModuleHandleW(D3DCOMPILER_DLL_W);
        wchar_t path[MAX_PATH]{};
        WIN32_FILE_ATTRIBUTE_DATA data{};
        if (!module || !GetModuleFileNameW(module, path, MAX_PATH) || !GetFileAttributesExW(path, GetFileExInfoStandard, &data))
          return std::string("unknown-compiler");
        char text[96];
        std::snprintf(text, sizeof(text), "compiler-%lu-%lu-%lu", data.nFileSizeLow, data.ftLastWriteTime.dwHighDateTime,
          data.ftLastWriteTime.dwLowDateTime);
        return std::string(text);
      }();
      return value;
    }
    inline std::string config_key(const configuration &config) {
      char text[160];
      std::snprintf(text, sizeof(text), "%s|%s|%ux%u|c%u|f%u", SUNSHINE_GAME3D_SHADER_SHA256, compiler_identity().c_str(),
        config.width, config.height, config.color, static_cast<unsigned>(compile_flags));
      return text;
    }
    inline std::string blob_key(const std::string &config, const entry_point &entry) {
      return config + '|' + entry.name + '|' + entry.target;
    }
    inline std::uint64_t fnv1a(std::string_view text) {
      std::uint64_t hash = 14695981039346656037ull;
      for (unsigned char c : text) hash = (hash ^ c) * 1099511628211ull;
      return hash;
    }
    // %LOCALAPPDATA%\Sunshine3D\game3d-shaders; empty when unavailable.
    inline const std::wstring &directory(store_t &s) {
      if (s.directory_checked) return s.directory;
      s.directory_checked = true;
      wchar_t base[MAX_PATH]{};
      const DWORD length = GetEnvironmentVariableW(L"LOCALAPPDATA", base, MAX_PATH);
      if (!length || length >= MAX_PATH) return s.directory;
      std::wstring path = std::wstring(base) + L"\\Sunshine3D";
      CreateDirectoryW(path.c_str(), nullptr);
      path += L"\\game3d-shaders";
      CreateDirectoryW(path.c_str(), nullptr);
      const DWORD attributes = GetFileAttributesW(path.c_str());
      if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) return s.directory;
      s.directory = path;
      // Each shader build adds its own files. Drop those untouched for 30
      // days and any abandoned temporary files; a live entry is recompiled once.
      FILETIME now{};
      GetSystemTimeAsFileTime(&now);
      const auto ticks = [](const FILETIME &t) { return (std::uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
      const std::uint64_t cutoff = ticks(now) - 30ull * 24 * 3600 * 10000000ull;
      WIN32_FIND_DATAW found{};
      const HANDLE search = FindFirstFileW((path + L"\\*").c_str(), &found);
      if (search != INVALID_HANDLE_VALUE) {
        do {
          if (found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
          const std::wstring name = found.cFileName;
          if (name.find(L".tmp") != std::wstring::npos || ticks(found.ftLastWriteTime) < cutoff)
            DeleteFileW((path + L"\\" + name).c_str());
        } while (FindNextFileW(search, &found));
        FindClose(search);
      }
      return s.directory;
    }
    inline std::wstring file_path(store_t &s, const std::string &key) {
      const auto &dir = directory(s);
      if (dir.empty()) return {};
      wchar_t name[40];
      std::swprintf(name, 40, L"\\%016llx.dxbc", static_cast<unsigned long long>(fnv1a(key)));
      return dir + name;
    }
    // File: "S3DX", key length, key, blob length, blob. The stored key guards
    // against a file-name hash collision; D3D validates the DXBC checksum.
    inline bool read_file(const std::wstring &path, const std::string &key, blob &out) {
      if (path.empty()) return false;
      FILE *file = _wfopen(path.c_str(), L"rb");
      if (!file) return false;
      bool okay = false;
      char magic[4]{};
      std::uint32_t key_length = 0, size = 0;
      if (std::fread(magic, 1, 4, file) == 4 && !std::memcmp(magic, "S3DX", 4) &&
          std::fread(&key_length, 4, 1, file) == 1 && key_length == key.size()) {
        std::string stored(key_length, '\0');
        if (std::fread(stored.data(), 1, key_length, file) == key_length && stored == key &&
            std::fread(&size, 4, 1, file) == 1 && size > 0 && size <= 4u * 1024 * 1024) {
          out.resize(size);
          okay = std::fread(out.data(), 1, size, file) == size && std::fgetc(file) == EOF;
        }
      }
      std::fclose(file);
      if (!okay) out.clear();
      return okay;
    }
    inline void write_file(const std::wstring &path, const std::string &key, const blob &value) {
      if (path.empty()) return;
      const std::wstring temporary = path + L".tmp" + std::to_wstring(GetCurrentThreadId());
      FILE *file = _wfopen(temporary.c_str(), L"wb");
      if (!file) return;
      const std::uint32_t key_length = std::uint32_t(key.size()), size = std::uint32_t(value.size());
      const bool okay = std::fwrite("S3DX", 1, 4, file) == 4 && std::fwrite(&key_length, 4, 1, file) == 1 &&
        std::fwrite(key.data(), 1, key.size(), file) == key.size() && std::fwrite(&size, 4, 1, file) == 1 &&
        std::fwrite(value.data(), 1, value.size(), file) == value.size();
      if (std::fclose(file) != 0 || !okay || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        DeleteFileW(temporary.c_str());
    }
    struct task_t {
      std::shared_ptr<job_t> job;
      entry_point entry;
    };
    inline void CALLBACK compile_task(PTP_CALLBACK_INSTANCE, void *context) {
      std::unique_ptr<task_t> task(static_cast<task_t *>(context));
      auto &job = *task->job;
      blob code;
      const bool okay = !sunshine_addon_lifetime::stopping() && compile(job.config, task->entry, code);
      auto &s = store();
      std::wstring path;
      std::string key;
      if (okay) {
        key = blob_key(job.key, task->entry);
        std::lock_guard<std::mutex> lock(s.mutex);
        path = file_path(s, key);
        s.blobs[key] = code;
      } else {
        job.failed.store(true, std::memory_order_release);
      }
      if (okay) write_file(path, key, code);
      job.remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  enum class state { ready, pending, failed };

  // Makes every listed entry point available. Memory and disk hits are ready
  // at once; otherwise one pool task per missing entry compiles it and this
  // returns pending until all of them finish.
  inline state prepare(const configuration &config, const std::vector<entry_point> &entries) {
    auto &s = detail::store();
    const auto key = detail::config_key(config);
    std::shared_ptr<detail::job_t> job;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto running = s.jobs.find(key);
      if (running != s.jobs.end()) {
        auto &existing = *running->second;
        if (existing.remaining.load(std::memory_order_acquire)) return state::pending;
        // Kept: a failed source is not recompiled on every Present.
        if (existing.failed.load(std::memory_order_acquire)) return state::failed;
        s.jobs.erase(running);
      }
      std::vector<entry_point> missing;
      for (const auto &entry : entries) {
        const auto name = detail::blob_key(key, entry);
        if (s.blobs.count(name)) continue;
        blob code;
        if (detail::read_file(detail::file_path(s, name), name, code)) s.blobs.emplace(name, std::move(code));
        else missing.push_back(entry);
      }
      if (missing.empty()) return state::ready;
      job = std::make_shared<detail::job_t>();
      job->source.assign(config.source);
      job->config = config;
      job->config.source = job->source;
      job->key = key;
      job->entries = std::move(missing);
      job->remaining.store(unsigned(job->entries.size()), std::memory_order_release);
      s.jobs.emplace(key, job);
    }
    // Submitted outside the lock: each task takes it to publish its blob.
    for (const auto &entry : job->entries) {
      auto *task = new detail::task_t{job, entry};
      // Without a pool thread, compile here rather than never finish.
      if (!TrySubmitThreadpoolCallback(detail::compile_task, task, nullptr)) detail::compile_task(nullptr, task);
    }
    char text[128];
    std::snprintf(text, sizeof(text), "Sunshine Game 3D: compiling %u shader entries off the present thread (%ux%u, color %u)",
      unsigned(job->entries.size()), config.width, config.height, config.color);
    sunshine_log::message(reshade::log::level::info, text);
    return state::pending;
  }

  // A prepared entry's bytecode, or empty when it was never prepared.
  inline blob find(const configuration &config, const entry_point &entry) {
    auto &s = detail::store();
    const auto name = detail::blob_key(detail::config_key(config), entry);
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.blobs.find(name);
    return found == s.blobs.end() ? blob{} : found->second;
  }
}

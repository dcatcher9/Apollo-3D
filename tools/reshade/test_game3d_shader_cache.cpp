// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_shader_cache.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ReShade's log API resolves these exports from the loaded modules.
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t) { return false; }
extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {}
extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *) {}

namespace {
  namespace cache = sunshine_game3d::shader_cache;
  namespace fs = std::filesystem;

  void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }

  constexpr char valid_source[] = "RWTexture2D<float> Output : register(u0);\n"
                                  "[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) { Output[id.xy] = 1.0; }\n";
  const cache::entry_point entry{"main", "cs_5_0"};

  cache::state settle(const cache::configuration &config) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    auto state = cache::prepare(config, {entry});
    while (state == cache::state::pending && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      state = cache::prepare(config, {entry});
    }
    return state;
  }

  // Forgets what this process prepared, as a new launch would.
  void forget() {
    auto &s = cache::detail::store();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.blobs.clear();
    s.jobs.clear();
  }

  void age(const fs::path &path, unsigned hours) {
    const HANDLE file = CreateFileW(path.c_str(), FILE_WRITE_ATTRIBUTES, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    check(file != INVALID_HANDLE_VALUE, "Could not open a file to age it");
    FILETIME now{};
    GetSystemTimeAsFileTime(&now);
    ULARGE_INTEGER ticks{};
    ticks.LowPart = now.dwLowDateTime;
    ticks.HighPart = now.dwHighDateTime;
    ticks.QuadPart -= std::uint64_t(hours) * 3600ull * 10000000ull;
    const FILETIME old{ticks.LowPart, ticks.HighPart};
    const bool okay = SetFileTime(file, nullptr, nullptr, &old);
    CloseHandle(file);
    check(okay, "Could not age a file");
  }

  void touch(const fs::path &path) {
    std::ofstream(path, std::ios::binary) << 'x';
  }

  void test_file_format(const fs::path &root) {
    const fs::path file = root / "format.dxbc";
    const auto path = file.wstring();
    const std::string key = "config|main|cs_5_0";
    const cache::blob code{1, 2, 3, 4, 5};
    cache::blob out;
    check(!cache::detail::read_file(path, key, out), "A missing file must not load");
    cache::detail::write_file(path, key, code);
    check(cache::detail::read_file(path, key, out) && out == code, "A written entry must read back exactly");
    check(!cache::detail::read_file(path, "config|main|cs_5_1", out) && out.empty(), "A same-length key mismatch must be rejected");
    check(!cache::detail::read_file(path, key + "x", out), "A different-length key must be rejected");
    check(!cache::detail::read_file(L"", key, out), "An unavailable directory must not load");

    std::ifstream input(file, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    const auto rewrite = [&file](const std::string &content) {
      std::ofstream output(file, std::ios::binary | std::ios::trunc);
      output.write(content.data(), std::streamsize(content.size()));
    };
    rewrite(bytes.substr(0, bytes.size() - 1));
    check(!cache::detail::read_file(path, key, out) && out.empty(), "A truncated blob must be rejected");
    rewrite(bytes + '\0');
    check(!cache::detail::read_file(path, key, out), "Trailing bytes must be rejected");
    rewrite("S3DY" + bytes.substr(4));
    check(!cache::detail::read_file(path, key, out), "A wrong magic must be rejected");

    std::size_t leftovers = 0;
    for (const auto &item : fs::directory_iterator(root))
      if (item.path().filename().wstring().find(L".tmp") != std::wstring::npos) ++leftovers;
    check(leftovers == 0, "A completed write must not leave its temporary file");
    fs::remove(file);
  }

  void test_prepare(const fs::path &root) {
    const fs::path directory = cache::detail::directory();
    check(directory == root / "Sunshine3D" / "game3d-shaders", "The cache must live below LOCALAPPDATA");
    // Pruned by the first pool task: stale temporary and month-old entries.
    const auto stale_tmp = directory / "0000000000000001.dxbc.tmp42", fresh_tmp = directory / "0000000000000002.dxbc.tmp42";
    const auto stale_entry = directory / "0000000000000003.dxbc", kept_entry = directory / "0000000000000004.dxbc";
    for (const auto &path : {stale_tmp, fresh_tmp, stale_entry, kept_entry}) touch(path);
    age(stale_tmp, 2);
    age(stale_entry, 31 * 24);
    age(kept_entry, 29 * 24);

    const cache::configuration config{valid_source, 4, 4, 1};
    check(cache::prepare(config, {entry}) == cache::state::pending, "An unprepared entry must be pending, not compiled inline");
    check(settle(config) == cache::state::ready, "A valid entry must become ready");
    const auto compiled = cache::find(config, entry);
    check(!compiled.empty(), "A ready entry must be found");
    check(cache::prepare(config, {entry}) == cache::state::ready, "A prepared entry must stay ready without a new job");
    check(!fs::exists(stale_tmp) && !fs::exists(stale_entry), "Stale files must be pruned");
    check(fs::exists(fresh_tmp) && fs::exists(kept_entry), "Recent files must be kept");

    const fs::path file = cache::detail::file_path(cache::detail::blob_key(cache::detail::config_key(config), entry));
    check(fs::exists(file), "A compiled entry must be persisted");

    // A later launch reads the file; it never compiles this unusable source.
    forget();
    const cache::configuration broken{"not HLSL", 4, 4, 1};
    check(settle(broken) == cache::state::ready, "A persisted entry must load without compiling");
    check(cache::find(broken, entry) == compiled, "A loaded entry must equal the compiled one");

    // A damaged file falls back to compiling, and a failed compile stays failed.
    forget();
    std::ofstream(file, std::ios::binary | std::ios::app) << 'x';
    check(settle(broken) == cache::state::failed, "A damaged file with a broken source must fail");
    check(cache::prepare(broken, {entry}) == cache::state::failed, "A failed job must not restart on every Present");
    check(cache::find(broken, entry).empty(), "A failed entry must not be found");
    forget();
    check(settle(config) == cache::state::ready && cache::find(config, entry) == compiled,
      "A damaged file must be replaced by a fresh compile");
  }
}

int main() {
  try {
    const fs::path root = fs::temp_directory_path() / ("sunshine-game3d-shader-cache-" + std::to_string(GetCurrentProcessId()));
    fs::remove_all(root);
    fs::create_directories(root);
    check(SetEnvironmentVariableW(L"LOCALAPPDATA", root.c_str()), "Could not redirect LOCALAPPDATA");
    test_file_format(root);
    test_prepare(root);
    fs::remove_all(root);
    std::puts("PASS Game 3D shader cache: file format, pool preparation, persistence and pruning");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: GPL-3.0-only
// rename_when_free (test_fixture_files.h): a rename whose source another
// process briefly holds open succeeds once it is let go, and a real failure
// still fails with its own error.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "test_fixture_files.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {
  [[noreturn]] void fail(const std::string &text) {
    std::fprintf(stderr, "FAIL %s\n", text.c_str());
    std::exit(1);
  }

  void require(bool value, const std::string &text) {
    if (!value) {
      fail(text);
    }
  }

  // The file opened as a scanner opens it: readable, without delete sharing,
  // so a rename of it fails with a sharing violation while it is open.
  HANDLE hold(const fs::path &file) {
    const auto handle = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(handle != INVALID_HANDLE_VALUE, "cannot hold the staged file open");
    return handle;
  }
}  // namespace

int main() {
  wchar_t temp[MAX_PATH] {};
  require(GetTempPathW(MAX_PATH, temp) != 0, "no temporary folder");
  const auto directory = fs::path(temp) / ("sunshine_fixture_files_" + std::to_string(GetCurrentProcessId()));
  fs::remove_all(directory);
  fs::create_directories(directory);
  const auto from = directory / "SunshineGame3D.fx", to = directory / "SunshineGame3D.fx.disabled";
  std::ofstream(from) << "technique SunshineGame3D {}\n";

  // Held for 300 ms, as a scan of a freshly written file holds it: a plain
  // rename fails there, rename_when_free succeeds once it is let go.
  {
    const auto handle = hold(from);
    std::error_code error;
    fs::rename(from, to, error);
    require(error && fs::exists(from), "a held file was renamed: the fixture cannot show the busy source");
    std::thread release([handle] {
      Sleep(300);
      CloseHandle(handle);
    });
    try {
      sunshine_test_files::rename_when_free(from, to);
    } catch (const fs::filesystem_error &failure) {
      release.join();
      fail(std::string("a source released within the wait was not renamed: ") + failure.what());
    }
    release.join();
    require(!fs::exists(from) && fs::exists(to), "the rename did not move the staged file");
    std::puts("PASS a source held open for 300 ms is renamed once released");
  }

  // Held past the wait: the rename fails with the first error it met.
  {
    fs::rename(to, from);
    const auto handle = hold(from);
    const auto started = GetTickCount64();
    bool thrown = false;
    std::error_code first;
    fs::rename(from, to, first);
    try {
      sunshine_test_files::rename_when_free(from, to);
    } catch (const fs::filesystem_error &failure) {
      thrown = true;
      require(failure.code() == first, "a rename held past the wait failed with another error than its own: " + failure.code().message());
    }
    const auto waited = GetTickCount64() - started;
    CloseHandle(handle);
    require(thrown, "a source held past the wait was reported renamed");
    require(waited >= std::uint64_t(sunshine_test_files::busy_wait.count()) - 20, "a held source failed before the wait ended");
    require(fs::exists(from), "a failed rename moved the source");
    std::printf("PASS a source held past the wait fails with its own error after %llu ms\n", static_cast<unsigned long long>(waited));
  }

  // A missing source is no busy file: it fails at once.
  {
    const auto missing = directory / "missing.fx";
    const auto started = GetTickCount64();
    bool thrown = false;
    try {
      sunshine_test_files::rename_when_free(missing, to);
    } catch (const fs::filesystem_error &) {
      thrown = true;
    }
    require(thrown && GetTickCount64() - started < 500, "a missing source was retried or not reported");
    std::puts("PASS a missing source fails at once");
  }
  fs::remove_all(directory);
  std::puts("Fixture file staging passed");
}

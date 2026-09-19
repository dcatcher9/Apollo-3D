// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "scene_depth_source.h"
#include <Windows.h>
#include <cstdint>

namespace sunshine_ngx {
  using result = std::uint32_t;
  struct parameter_api {
    result (__cdecl *integer)(void *, const char *, int *){};
    result (__cdecl *unsigned_integer)(void *, const char *, unsigned *){};
    result (__cdecl *resource)(void *, const char *, void **){};
    result (__cdecl *void_pointer)(void *, const char *, void **){}; // Optional diagnostic only.
    result (__cdecl *floating)(void *, const char *, float *){}; // Optional per-evaluation jitter.
    explicit operator bool() const { return integer && unsigned_integer && resource; }
  };
  // Only exact named exports of the already loaded NGX SDK owner are used.
  // No assumed C++ parameter vtable and no parameter mutation.
  parameter_api resolve_parameter_api(HMODULE owner);
  void initialize(bool enabled, bool calibration_probe = false);
  void shutdown();
  bool enabled();
  std::uint64_t epoch();
  void poll();
  struct creation {
    std::uint64_t epoch{};
    std::uint32_t feature{}, width{}, height{};
    int flags{};
    bool valid{};
  };
  creation before_create(const parameter_api &api, std::uint32_t feature, const void *parameters);
  void after_create(HMODULE owner, const void *handle, const creation &value, bool successful);
  void after_release(HMODULE owner, const void *handle, std::uint64_t epoch, bool successful);
  struct evaluation {
    std::uint64_t epoch{}, ticket{}, source_id{};
    bool observed{}; // Confirmed feature, even when this frame's metadata fails.
  };
  evaluation before_evaluate(HMODULE owner, const parameter_api &api, std::uint64_t command,
    const void *handle, const void *parameters);
  void after_evaluate(const evaluation &value, bool successful);

#ifdef SUNSHINE_UPSCALER_TRACE_TEST
  namespace testing {
    struct callbacks {
      std::uint64_t (*record)(std::uint64_t command, const sunshine_scene_depth::frame &value){};
      void (*finish)(std::uint64_t ticket, bool successful){};
      void (*retire)(std::uint64_t epoch, std::uint64_t source_id){};
    };
    void set_callbacks(callbacks value);
  }
#endif
}

// SPDX-License-Identifier: GPL-3.0-only
// A game-side NGX SDK stand-in for the test executables: the D3D12 feature
// entry points and the named C parameter getters an NGX SDK owner exports
// (nvsdk_ngx.h, nvsdk_ngx_params.h), exported by the executable that includes
// this header once, so production module discovery hooks them as it hooks a
// game's. It reports metadata only: no capture, state, readiness or pixels
// are injected.
//
// Any feature ID is created. A parameter map answers only the keys a case
// set, each through the getter of the type it was set with, as an SDK's
// Set* stores it: Depth set through set_pointer (SetVoidPointer) is absent
// to GetD3d12Resource. Results are settable per entry point. Evaluation
// accepts any non-null command list, DIRECT or COMPUTE: the list's type is
// the fixture's choice.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_MSC_VER)
  #define NGX_STAND_IN_EXPORT extern "C" __declspec(dllexport) __declspec(noinline)
#else
  #define NGX_STAND_IN_EXPORT extern "C" __declspec(dllexport) __attribute__((noinline, noipa))
#endif

namespace ngx_fixture {
  // NVSDK_NGX_Result_Success and an NVSDK_NGX_Result_Fail code (nvsdk_ngx_defs.h).
  constexpr std::uint32_t success = 1, failure = 0xbad00001u;

  // One NVSDK_NGX_Parameter map.
  struct parameters {
    enum class kind {
      unsigned_integer,
      integer,
      floating,
      resource,
      pointer
    };

    struct entry {
      std::string key;
      kind type {};
      unsigned unsigned_value {};
      int int_value {};
      float float_value {};
      void *pointer {};
    };

    std::vector<entry> entries;

    void set_ui(const char *key, unsigned value) {
      put(key, kind::unsigned_integer).unsigned_value = value;
    }

    void set_i(const char *key, int value) {
      put(key, kind::integer).int_value = value;
    }

    void set_f(const char *key, float value) {
      put(key, kind::floating).float_value = value;
    }

    // SetD3d12Resource, and SetVoidPointer: the same pointer, another getter.
    void set_resource(const char *key, void *value) {
      put(key, kind::resource).pointer = value;
    }

    void set_pointer(const char *key, void *value) {
      put(key, kind::pointer).pointer = value;
    }

    void erase(const char *key) {
      for (auto at = entries.begin(); at != entries.end(); ++at) {
        if (at->key == key) {
          entries.erase(at);
          return;
        }
      }
    }

    // The entry a getter of this type answers; none for an absent key or one
    // set with another type.
    const entry *find(const char *key, kind type) const {
      for (const auto &value : entries) {
        if (value.key == key) {
          return value.type == type ? &value : nullptr;
        }
      }
      return nullptr;
    }

  private:
    entry &put(const char *key, kind type) {
      for (auto &value : entries) {
        if (value.key == key) {
          value = {key, type};
          return value;
        }
      }
      entries.push_back({key, type});
      return entries.back();
    }
  };

  // Each created feature's ID, while live (CreateFeature to ReleaseFeature).
  // A handle is the address of its record; records are never reused, so a
  // process creates at most this many features (the API contract matrix's
  // NGX family creates one or two a case).
  struct feature_record {
    std::uint32_t id {};
    bool live {};
  };

  inline std::array<feature_record, 64> features {};
  inline unsigned created {}, evaluated {}, released {}, getters {};

  // What a well-formed call returns (a null argument or an unknown or released
  // handle always fails); a failed create or release changes no feature.
  struct entry_results {
    std::uint32_t create {success}, evaluate {success}, release {success};
  };

  inline entry_results results;

  inline feature_record *live(const void *handle) {
    for (auto &value : features) {
      if (handle == &value) {
        return value.live ? &value : nullptr;
      }
    }
    return nullptr;
  }

  inline std::uint32_t get(const void *map, const char *key, parameters::kind type, const parameters::entry *&out) {
    ++getters;
    out = map && key ? static_cast<const parameters *>(map)->find(key, type) : nullptr;
    return out ? success : failure;
  }
}  // namespace ngx_fixture

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_CreateFeature(
  void *commands,
  std::uint32_t feature,
  void *parameters,
  void **handle
) {
  if (!commands || !parameters || !handle || ngx_fixture::created >= ngx_fixture::features.size()) {
    return ngx_fixture::failure;
  }
  if (ngx_fixture::results.create != ngx_fixture::success) {
    return ngx_fixture::results.create;
  }
  auto &record = ngx_fixture::features[ngx_fixture::created++];
  record = {feature, true};
  *handle = &record;
  return ngx_fixture::success;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_EvaluateFeature(
  void *commands,
  const void *handle,
  const void *parameters,
  void(__cdecl *)(float, bool &)
) {
  ++ngx_fixture::evaluated;
  if (!commands || !parameters || !ngx_fixture::live(handle)) {
    return ngx_fixture::failure;
  }
  return ngx_fixture::results.evaluate;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_ReleaseFeature(void *handle) {
  auto *record = ngx_fixture::live(handle);
  if (!record) {
    return ngx_fixture::failure;
  }
  if (ngx_fixture::results.release != ngx_fixture::success) {
    return ngx_fixture::results.release;
  }
  record->live = false;
  ++ngx_fixture::released;
  return ngx_fixture::success;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetI(void *parameters, const char *key, int *out) {
  const ngx_fixture::parameters::entry *value {};
  const auto result = ngx_fixture::get(parameters, key, ngx_fixture::parameters::kind::integer, value);
  if (value && out) {
    *out = value->int_value;
  }
  return out ? result : ngx_fixture::failure;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetUI(void *parameters, const char *key, unsigned *out) {
  const ngx_fixture::parameters::entry *value {};
  const auto result = ngx_fixture::get(parameters, key, ngx_fixture::parameters::kind::unsigned_integer, value);
  if (value && out) {
    *out = value->unsigned_value;
  }
  return out ? result : ngx_fixture::failure;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetF(void *parameters, const char *key, float *out) {
  const ngx_fixture::parameters::entry *value {};
  const auto result = ngx_fixture::get(parameters, key, ngx_fixture::parameters::kind::floating, value);
  if (value && out) {
    *out = value->float_value;
  }
  return out ? result : ngx_fixture::failure;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetD3d12Resource(void *parameters, const char *key, void **out) {
  const ngx_fixture::parameters::entry *value {};
  const auto result = ngx_fixture::get(parameters, key, ngx_fixture::parameters::kind::resource, value);
  if (value && out) {
    *out = value->pointer;
  }
  return out ? result : ngx_fixture::failure;
}

NGX_STAND_IN_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetVoidPointer(void *parameters, const char *key, void **out) {
  const ngx_fixture::parameters::entry *value {};
  const auto result = ngx_fixture::get(parameters, key, ngx_fixture::parameters::kind::pointer, value);
  if (value && out) {
    *out = value->pointer;
  }
  return out ? result : ngx_fixture::failure;
}

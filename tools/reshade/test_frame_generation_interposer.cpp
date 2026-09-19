// SPDX-License-Identifier: GPL-3.0-only
// Synthetic SDK metadata boundary for the opt-in GPU fixture. This DLL performs
// no rendering, capture or add-on state injection. The production observer must
// discover its versioned exports and detour the returned options function.
#include "streamline_camera_data.h"
#include <atomic>
#include <cstring>
#include <windows.h>

#if defined(_MSC_VER)
#define FG_FIXTURE_EXPORT extern "C" __declspec(dllexport) __declspec(noinline)
#else
#define FG_FIXTURE_EXPORT extern "C" __declspec(dllexport) __attribute__((noinline,noipa))
#endif

using namespace sunshine_streamline;
namespace {
  struct options_prefix { base_structure base; std::uint32_t mode{}, generated_frames{}; };
  static_assert(sizeof(options_prefix) == 40);
  std::atomic<std::int32_t> tag_result{};
  std::atomic<HANDLE> tag_entered{},tag_release{};
  alignas(void *) unsigned char token_storage[16]{};
  // Observable work gives every exported boundary a real, distinct prologue.
  // A three-byte xor/ret followed by multi-byte alignment NOPs is not a valid
  // MinHook target and would turn the regression into a hook-setup failure.
  std::atomic<std::uint64_t> calls[8]{};

  std::int32_t finish_tag() {
    if(const auto entered=tag_entered.exchange(nullptr,std::memory_order_acq_rel)) {
      const auto release=tag_release.load(std::memory_order_acquire);
      SetEvent(entered);
      if(!release || WaitForSingleObject(release,5000)!=WAIT_OBJECT_0)return 1;
    }
    return tag_result.load(std::memory_order_acquire);
  }
}

FG_FIXTURE_EXPORT std::int32_t slSetConstants(const abi_v2::constants &, const abi_v2::frame_token &, const abi_v2::viewport &) {
  calls[0].fetch_add(1,std::memory_order_relaxed);return 0;
}
FG_FIXTURE_EXPORT std::int32_t slSetTag(const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
  calls[1].fetch_add(1,std::memory_order_relaxed);return finish_tag();
}
FG_FIXTURE_EXPORT std::int32_t slSetTagForFrame(const abi_v2::frame_token &, const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
  calls[2].fetch_add(1,std::memory_order_relaxed);return finish_tag();
}
FG_FIXTURE_EXPORT void SunshineFixtureSetTagResult(std::int32_t result) {
  calls[3].fetch_add(1,std::memory_order_relaxed);tag_result.store(result,std::memory_order_release);
}
FG_FIXTURE_EXPORT void SunshineFixtureBlockNextTag(HANDLE entered,HANDLE release) {
  tag_release.store(release,std::memory_order_release);
  tag_entered.store(entered,std::memory_order_release);
}
FG_FIXTURE_EXPORT std::int32_t slGetNewFrameToken(abi_v2::frame_token *&token,const std::uint32_t *) {
  calls[4].fetch_add(1,std::memory_order_relaxed);
  // The production observer handles this as an opaque address plus observed
  // explicit index. Nothing in this fixture or observer invokes a token vtable.
  token=reinterpret_cast<abi_v2::frame_token *>(token_storage);return 0;
}
FG_FIXTURE_EXPORT std::int32_t slEvaluateFeature(std::uint32_t, const abi_v2::frame_token &, const base_structure **, std::uint32_t, void *) {
  calls[5].fetch_add(1,std::memory_order_relaxed);return 0;
}
FG_FIXTURE_EXPORT std::int32_t slDLSSGSetOptions(const abi_v2::viewport &, const options_prefix &options) {
  calls[6].fetch_add(1,std::memory_order_relaxed);
  return options.mode <= 2 && (!options.mode || options.generated_frames) ? 0 : 1;
}
FG_FIXTURE_EXPORT std::int32_t slGetFeatureFunction(std::uint32_t feature, const char *name, void *&function) {
  calls[7].fetch_add(1,std::memory_order_relaxed);
  if (feature != 1000 || !name || std::strcmp(name, "slDLSSGSetOptions")) return 1;
  function = reinterpret_cast<void *>(&slDLSSGSetOptions);
  return 0;
}

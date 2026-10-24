// SPDX-License-Identifier: GPL-3.0-only
// Synthetic SDK metadata boundary for the opt-in GPU fixtures. This DLL performs
// no rendering, capture or add-on state injection. The production observer must
// discover its versioned exports and detour the returned options function.
// CMake builds one DLL per synthetic release (README, interposer fixtures):
// SUNSHINE_SL_FIXTURE_VERSION is the release its version resource names, and
// SUNSHINE_SL_FIXTURE_NO_FRAME_TAG omits slSetTagForFrame, as before 2.7.
#include "streamline_camera_data.h"
#include "test_sl_survey_tables.generated.h"
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
  using options_prefix = abi_v2::fg_options_prefix;
  static_assert(sizeof(options_prefix) == 40);
  // sl_dlss_g.h DLSSGOptions: mode and numFramesToGenerate are its first fields
  // from version 1 on, and later versions only append (sl_struct.h), so a
  // version above this release's is read by its known prefix.
  constexpr guid options_type{0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
  // eErrorIO, what this fixture returns for options its release refuses.
  constexpr std::int32_t refused = 1;
  constexpr std::uint32_t version[]{SUNSHINE_SL_FIXTURE_VERSION};
  // What this release's header declares: that of the newest surveyed release
  // at or below it (test_sl_survey_tables.generated.h), never the add-on's.
  const game3d_contract::survey::release &declared() {
    const auto key = [](std::uint32_t major, std::uint32_t minor, std::uint32_t patch) {
      return (std::uint64_t(major) << 40) | (std::uint64_t(minor) << 20) | patch;
    };
    const auto own = key(version[0], version[1], version[2]);
    const game3d_contract::survey::release *newest = &game3d_contract::survey::releases[0];
    for (const auto &release : game3d_contract::survey::releases) {
      const auto at = key(release.major, release.minor, release.patch);
      if (at <= own && at >= key(newest->major, newest->minor, newest->patch)) newest = &release;
    }
    return *newest;
  }
  // DLSSGMode values before eCount, and numFramesToGenerate: it "Cannot exceed
  // DLSSGState::numFramesToGenerateMax", which a device reports. This one
  // reports the largest any surveyed header documents (5, for 6x) where the
  // release documents multi frame generation, else 1 ("Must be 1").
  bool accepted(const options_prefix &options) {
    const auto &release = declared();
    if (std::memcmp(&options.base.type, &options_type, sizeof(guid)) || !options.base.version ||
        options.mode >= release.dlssg_modes)
      return false;
    return !options.mode || (options.generated_frames && options.generated_frames <= (release.dlssg_multi_frame ? 5u : 1u));
  }

  // The sl::Result each boundary returns for a call it accepts
  // (SunshineFixtureSetResults).
  enum boundary { constants, tag, evaluate, options, feature, boundaries };
  std::atomic<std::int32_t> results[boundaries]{};
  std::int32_t result(boundary which) { return results[which].load(std::memory_order_acquire); }
  std::atomic<HANDLE> tag_entered{},tag_release{};
  alignas(void *) unsigned char token_storage[16]{};
  // Observable work gives every exported boundary a real, distinct prologue.
  // A three-byte xor/ret followed by multi-byte alignment NOPs is not a valid
  // MinHook target and would turn the regression into a hook-setup failure.
  std::atomic<std::uint64_t> calls[9]{};

  std::int32_t finish_tag() {
    if(const auto entered=tag_entered.exchange(nullptr,std::memory_order_acq_rel)) {
      const auto release=tag_release.load(std::memory_order_acquire);
      SetEvent(entered);
      if(!release || WaitForSingleObject(release,5000)!=WAIT_OBJECT_0)return 1;
    }
    return result(tag);
  }
}

FG_FIXTURE_EXPORT std::int32_t slSetConstants(const abi_v2::constants &, const abi_v2::frame_token &, const abi_v2::viewport &) {
  calls[0].fetch_add(1,std::memory_order_relaxed);return result(constants);
}
FG_FIXTURE_EXPORT std::int32_t slSetTag(const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
  calls[1].fetch_add(1,std::memory_order_relaxed);return finish_tag();
}
#ifndef SUNSHINE_SL_FIXTURE_NO_FRAME_TAG
FG_FIXTURE_EXPORT std::int32_t slSetTagForFrame(const abi_v2::frame_token &, const abi_v2::viewport &, const abi_v2::resource_tag *, std::uint32_t, void *) {
  calls[2].fetch_add(1,std::memory_order_relaxed);return finish_tag();
}
#endif
// The results slSetConstants, both tag entries, slEvaluateFeature, the
// options function and slGetFeatureFunction return for calls they accept
// (eOk until set). Options the release refuses and unknown feature
// functions still return their own error.
FG_FIXTURE_EXPORT void SunshineFixtureSetResults(std::int32_t constants_result, std::int32_t tag_result,
    std::int32_t evaluate_result, std::int32_t options_result, std::int32_t feature_result) {
  calls[8].fetch_add(1,std::memory_order_relaxed);
  const std::int32_t values[]{constants_result, tag_result, evaluate_result, options_result, feature_result};
  for (unsigned i = 0; i != boundaries; ++i) results[i].store(values[i], std::memory_order_release);
}
FG_FIXTURE_EXPORT void SunshineFixtureSetTagResult(std::int32_t value) {
  calls[3].fetch_add(1,std::memory_order_relaxed);results[tag].store(value,std::memory_order_release);
}
FG_FIXTURE_EXPORT void SunshineFixtureSetOptionsResult(std::int32_t value) {
  results[options].store(value,std::memory_order_release);
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
  calls[5].fetch_add(1,std::memory_order_relaxed);return result(evaluate);
}
FG_FIXTURE_EXPORT std::int32_t slDLSSGSetOptions(const abi_v2::viewport &, const options_prefix &value) {
  calls[6].fetch_add(1,std::memory_order_relaxed);
  return accepted(value) ? result(options) : refused;
}
FG_FIXTURE_EXPORT std::int32_t slGetFeatureFunction(std::uint32_t feature_id, const char *name, void *&function) {
  calls[7].fetch_add(1,std::memory_order_relaxed);
  if (feature_id != 1000 || !name || std::strcmp(name, "slDLSSGSetOptions")) return 1;
  function = reinterpret_cast<void *>(&slDLSSGSetOptions);
  return result(feature);
}

// SPDX-License-Identifier: GPL-3.0-only
// Real MinHook regression against isolated, noinline fixture entrypoints. No game,
// Streamline DLL, graphics device, or GPU work is involved.
#include "streamline_camera_data.h"
#include "addon_lifetime.h"
#include "streamline_camera_probe.h"
#include "streamline_depth_provider.h"
#include "streamline_camera_version.h"
#include "upscaler_call_trace.h"
#include "game3d_ui_mask.h"
#include <reshade.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <thread>

#if defined(__GNUC__) && !defined(__clang__)
  #define FIXTURE_NOINLINE __attribute__((noinline, noipa))
#else
  #define FIXTURE_NOINLINE __declspec(noinline)
#endif

// ReShade's header-only logging API locates these exports. Supply only a log
// sink/discovery identity, never a graphics runtime, for the real no-DLL poll.
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t) {
  return false;
}

extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {}

extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *) {}

// Diagnostic fixtures explicitly opt in through the same public config API as
// the game. nullptr models a missing setting for the default-off regression.
static const char *probe_setting = "1";
static const char *source_setting = "0"; // Existing diagnostic tests isolate their opt-in contract.
static const char *call_trace_setting = nullptr;
extern "C" __declspec(dllexport) bool ReShadeGetConfigValue(void *, reshade::api::effect_runtime *,
    const char *section, const char *key, char *value, size_t *size) {
  if (std::strcmp(section, "SUNSHINE_DEPTH") != 0 || !size) return false;
  const char *setting = std::strcmp(key, "StreamlineCameraProbe") == 0 ? probe_setting :
    std::strcmp(key, "StreamlineDepthSource") == 0 ? source_setting :
    std::strcmp(key, "NGXDepthSource") == 0 ? "0" :
    std::strcmp(key, "UpscalerCallTrace") == 0 ? call_trace_setting : nullptr;
  if (!setting) return false;
  const size_t length = std::strlen(setting);
  if (!value || *size <= length) { *size = length + 1; return value == nullptr; }
  std::memcpy(value, setting, length + 1);
  *size = length;
  return true;
}

namespace {
  using namespace sunshine_streamline;

  struct invocation {
    std::uint64_t calls {};
    const void *first {}, *second {}, *third {}, *fourth {};
    std::uint32_t a {}, b {}, c {};
    DWORD incoming_error {};
  } constants_call, tag_call, framed_tag_call, evaluate_call, new_token_call, feature_function_call, pcl_call, fg_call;

  constexpr DWORD incoming_error = 0x12345678, outgoing_error = 0xabcdef01;
  bool result_v1 = true;
  std::int32_t result_v2 = 0;
  std::atomic<bool> block_original {}, original_entered {}, release_original {};
  abi_v2::frame_token *returned_token{};
  void *returned_feature_function{};
  bool inspect_presentation_in_original{};
  presentation_snapshot presentation_inside_original;

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void block_if_requested() {
    if (!block_original.load(std::memory_order_acquire)) {
      return;
    }
    original_entered.store(true, std::memory_order_release);
    while (!release_original.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  FIXTURE_NOINLINE bool fake_v1_constants(const abi_v1::constants &value, std::uint32_t frame, std::uint32_t viewport) {
    constants_call.incoming_error = GetLastError();
    ++constants_call.calls;
    constants_call.first = &value;
    constants_call.a = frame;
    constants_call.b = viewport;
    block_if_requested();
    SetLastError(outgoing_error);
    return result_v1;
  }

  FIXTURE_NOINLINE bool fake_v1_tag(const abi_v1::resource *resource, std::uint32_t type, std::uint32_t viewport, const extent *area) {
    tag_call.incoming_error = GetLastError();
    ++tag_call.calls;
    tag_call.first = resource;
    tag_call.second = area;
    tag_call.a = type;
    tag_call.b = viewport;
    SetLastError(outgoing_error);
    return result_v1;
  }

  FIXTURE_NOINLINE bool fake_v1_evaluate(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport) {
    evaluate_call.incoming_error = GetLastError();
    ++evaluate_call.calls;
    evaluate_call.first = commands;
    evaluate_call.a = feature;
    evaluate_call.b = frame;
    evaluate_call.c = viewport;
    block_if_requested();
    SetLastError(outgoing_error);
    return result_v1;
  }

  FIXTURE_NOINLINE std::int32_t fake_v2_constants(const abi_v2::constants &value, const abi_v2::frame_token &frame, const abi_v2::viewport &viewport) {
    constants_call.incoming_error = GetLastError();
    ++constants_call.calls;
    constants_call.first = &value;
    constants_call.second = &frame;
    constants_call.third = &viewport;
    block_if_requested();
    SetLastError(outgoing_error);
    return result_v2;
  }

  FIXTURE_NOINLINE std::int32_t fake_v2_tag(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
    tag_call.incoming_error = GetLastError();
    ++tag_call.calls;
    tag_call.first = &viewport;
    tag_call.second = tags;
    tag_call.third = commands;
    tag_call.a = count;
    SetLastError(outgoing_error);
    return result_v2;
  }

  FIXTURE_NOINLINE std::int32_t fake_v2_framed_tag(const abi_v2::frame_token &frame, const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
    framed_tag_call.incoming_error = GetLastError();
    ++framed_tag_call.calls;
    framed_tag_call.first = &frame;
    framed_tag_call.second = &viewport;
    framed_tag_call.third = tags;
    framed_tag_call.fourth = commands;
    framed_tag_call.a = count;
    SetLastError(outgoing_error);
    return result_v2;
  }

  FIXTURE_NOINLINE std::int32_t fake_v2_evaluate(std::uint32_t feature, const abi_v2::frame_token &frame, const base_structure **inputs, std::uint32_t count, void *commands) {
    evaluate_call.incoming_error = GetLastError();
    ++evaluate_call.calls;
    evaluate_call.first = &frame;
    evaluate_call.second = inputs;
    evaluate_call.third = commands;
    evaluate_call.a = feature;
    evaluate_call.b = count;
    block_if_requested();
    SetLastError(outgoing_error);
    return result_v2;
  }
  FIXTURE_NOINLINE std::int32_t fake_v2_new_token(abi_v2::frame_token *&token, const std::uint32_t *index) {
    new_token_call.incoming_error = GetLastError();
    ++new_token_call.calls;
    new_token_call.first = &token;
    new_token_call.second = index;
    token = returned_token;
    block_if_requested();
    SetLastError(outgoing_error);
    return result_v2;
  }
  FIXTURE_NOINLINE std::int32_t fake_pcl_marker(std::uint32_t marker, const abi_v2::frame_token &frame) {
    pcl_call.incoming_error = GetLastError(); ++pcl_call.calls;
    pcl_call.first = &frame; pcl_call.a = marker;
    if (inspect_presentation_in_original) query_current_presentation(presentation_inside_original);
    block_if_requested();
    SetLastError(outgoing_error); return result_v2;
  }
  FIXTURE_NOINLINE std::int32_t fake_get_feature_function(std::uint32_t feature, const char *name, void *&function) {
    feature_function_call.incoming_error = GetLastError(); ++feature_function_call.calls;
    feature_function_call.first = name; feature_function_call.second = &function; feature_function_call.a = feature;
    block_if_requested();
    function = returned_feature_function;
    SetLastError(outgoing_error); return result_v2;
  }
  constexpr guid fg_options_type{0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
  constexpr guid precision_type{0x98f6e9ba,0x8d16,0x4831,{0xa8,0x02,0x4d,0x3b,0x52,0xff,0x26,0xbf}};
  struct fg_options { base_structure base{nullptr, fg_options_type, 3}; std::uint32_t mode{1}, generated_frames{1}; };
  struct tag_precision { base_structure base{nullptr, precision_type, 1}; std::uint32_t formula{1}; float bias{.25f}, scale{.5f}; };
  using fg_function = std::int32_t (*)(const abi_v2::viewport &, const fg_options &);
  FIXTURE_NOINLINE std::int32_t fake_fg_options(const abi_v2::viewport &view, const fg_options &options) {
    fg_call.incoming_error = GetLastError(); ++fg_call.calls;
    fg_call.first = &view; fg_call.second = &options;
    block_if_requested(); SetLastError(outgoing_error); return result_v2;
  }

  // Volatile dispatch prevents the compiler from substituting a direct call that
  // bypasses the function entry patched by MinHook, even under optimization/LTO.
  abi_v1::set_constants volatile call_v1_constants = &fake_v1_constants;
  abi_v1::set_tag volatile call_v1_tag = &fake_v1_tag;
  abi_v1::evaluate_feature volatile call_v1_evaluate = &fake_v1_evaluate;
  abi_v2::set_constants volatile call_v2_constants = &fake_v2_constants;
  abi_v2::set_tag volatile call_v2_tag = &fake_v2_tag;
  abi_v2::set_tag_for_frame volatile call_v2_framed_tag = &fake_v2_framed_tag;
  abi_v2::evaluate_feature volatile call_v2_evaluate = &fake_v2_evaluate;
  abi_v2::get_new_frame_token volatile call_v2_new_token = &fake_v2_new_token;
  abi_v2::get_feature_function volatile call_feature_function = &fake_get_feature_function;

  const testing::targets v1_targets {reinterpret_cast<void *>(&fake_v1_constants), reinterpret_cast<void *>(&fake_v1_tag), nullptr, reinterpret_cast<void *>(&fake_v1_evaluate)};
  const testing::targets v2_targets {reinterpret_cast<void *>(&fake_v2_constants), reinterpret_cast<void *>(&fake_v2_tag), reinterpret_cast<void *>(&fake_v2_framed_tag), reinterpret_cast<void *>(&fake_v2_evaluate)};
  const testing::targets tracked_v2_targets {v2_targets.constants, v2_targets.tag, v2_targets.tag_for_frame,
    v2_targets.evaluate, reinterpret_cast<void *>(&fake_v2_new_token)};
  const testing::targets presentation_v2_targets {v2_targets.constants, v2_targets.tag, v2_targets.tag_for_frame,
    v2_targets.evaluate, reinterpret_cast<void *>(&fake_v2_new_token), reinterpret_cast<void *>(&fake_get_feature_function)};

  struct fixture {
    fixture() {
      probe_setting = "1";
      source_setting = "0";
      testing::clear();
      constants_call = tag_call = framed_tag_call = evaluate_call = new_token_call = feature_function_call = pcl_call = fg_call = {};
      returned_token = nullptr;
      returned_feature_function = reinterpret_cast<void *>(&fake_pcl_marker);
      inspect_presentation_in_original = false; presentation_inside_original = {};
      result_v1 = true;
      result_v2 = 0;
      block_original = original_entered = release_original = false;
    }

    ~fixture() {
      testing::clear();
    }
  };

  common_constants camera() {
    common_constants value {};
    // Exact reversed infinite-far D3D perspective and its analytic inverse.
    value.camera_view_to_clip.m[0][0] = value.camera_view_to_clip.m[1][1] = 1;
    value.camera_view_to_clip.m[2][3] = 1;
    value.camera_view_to_clip.m[3][2] = .5f;
    value.clip_to_camera_view.m[0][0] = value.clip_to_camera_view.m[1][1] = 1;
    value.clip_to_camera_view.m[2][3] = 2;
    value.clip_to_camera_view.m[3][2] = 1;
    value.camera_near = .5f;
    value.camera_far = std::numeric_limits<float>::infinity();
    value.camera_fov = 1.5707963267948966f;
    value.camera_aspect = 1;
    return value;
  }

  void set_camera_near(common_constants &value, float near_plane) {
    value.camera_view_to_clip.m[3][2] = near_plane;
    value.clip_to_camera_view.m[2][3] = 1.f / near_plane;
    value.camera_near = near_plane;
  }

  abi_v1::constants old_camera() {
    abi_v1::constants value {};
    value.common = camera();
    value.depth_inverted = 1;
    return value;
  }

  abi_v2::constants modern_camera() {
    abi_v2::constants value {};
    value.base = {nullptr, constants_guid, 2};
    value.common = camera();
    value.depth_inverted = 1;
    return value;
  }

  abi_v2::viewport viewport(std::uint32_t id) {
    abi_v2::viewport value {};
    value.base = {nullptr, viewport_guid, 1};
    value.value = id;
    return value;
  }

  struct opaque_token {
    alignas(16) std::uint64_t identity[2] {};

    const abi_v2::frame_token &ref() const {
      // The observer must never invoke FrameToken's compiler-dependent virtual
      // conversion. Dummy storage deliberately has no valid vtable to invoke.
      return *reinterpret_cast<const abi_v2::frame_token *>(identity);
    }
  };

  abi_v2::resource modern_resource(void *native) {
    abi_v2::resource value {};
    value.base = {nullptr, resource_guid, 1};
    value.type = 0;
    value.native = native;
    value.width = 1920;
    value.height = 1080;
    return value;
  }

  abi_v2::resource_tag depth_tag_for(const abi_v2::resource &resource) {
    abi_v2::resource_tag value {};
    value.base = {nullptr, tag_guid, 1};
    value.resource_ptr = &resource;
    value.type = 0;
    value.lifecycle = 1;
    value.area = {0, 0, 1920, 1080};
    return value;
  }

  void unchanged(const testing::counters &a, const testing::counters &b, const char *message) {
    require(a.constants == b.constants && a.tags == b.tags && a.evaluations == b.evaluations, message);
  }

  struct observed {
    std::uint64_t resource {};
    bool found {}, camera {}, exact {}, same_token {};
  };

  observed latest(std::uint32_t id) {
    observed value;
    value.found = testing::latest(id, value.resource, value.camera, value.exact, value.same_token);
    return value;
  }

  void test_v1_forwarding() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 real MinHook install failed");
    auto constants = old_camera();
    const auto original_constants = constants;
    int native {}, commands {};
    abi_v1::resource resource {};
    resource.native = &native;
    const auto original_resource = resource;
    const extent area {7, 11, 1280, 720};
    const auto before = testing::counts();
    require(call_v1_constants(constants, 0x87654321u, 17), "v1 constants success changed");
    require(constants_call.calls == 1 && constants_call.first == &constants && constants_call.a == 0x87654321u && constants_call.b == 17, "v1 constants arguments changed");
    require(call_v1_tag(&resource, 0, 17, &area), "v1 tag success changed");
    require(tag_call.calls == 1 && tag_call.first == &resource && tag_call.second == &area && tag_call.a == 0 && tag_call.b == 17, "v1 tag arguments changed");
    require(call_v1_evaluate(&commands, 0x12345678u, 0x87654321u, 17), "v1 evaluate success changed");
    require(evaluate_call.calls == 1 && evaluate_call.first == &commands && evaluate_call.a == 0x12345678u && evaluate_call.b == 0x87654321u && evaluate_call.c == 17, "v1 evaluate arguments changed");
    require(std::memcmp(&constants, &original_constants, sizeof(constants)) == 0 && std::memcmp(&resource, &original_resource, sizeof(resource)) == 0, "v1 input mutated");
    const auto after = testing::counts();
    require(after.constants == before.constants + 1 && after.tags == before.tags + 1 && after.evaluations == before.evaluations + 1, "v1 successful calls not observed exactly once");

    result_v1 = false;
    require(!call_v1_constants(constants, 9, 18), "v1 constants failure changed");
    require(!call_v1_tag(&resource, 0, 18, nullptr), "v1 null-extent tag failure changed");
    require(tag_call.second == nullptr, "v1 null extent replaced");
    require(!call_v1_evaluate(nullptr, 3, 9, 18), "v1 evaluate failure changed");
    require(evaluate_call.first == nullptr, "v1 null command list replaced");
    const auto failed = latest(18);
    require(!failed.found && !failed.camera, "v1 failed calls published new viewport state");
    int replacement_native {};
    resource.native = &replacement_native;
    require(!call_v1_tag(&resource, 0, 17, &area), "v1 replacement failure changed");
    require(latest(17).resource == reinterpret_cast<std::uintptr_t>(&native), "v1 failed tag overwrote successful state");
  }

  void test_v2_forwarding() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, v2_targets), "v2 real MinHook install failed");
    auto constants = modern_camera();
    const auto original_constants = constants;
    auto view = viewport(27);
    opaque_token token;
    int native {}, commands {};
    auto resource = modern_resource(&native);
    abi_v2::resource_tag tags[] {depth_tag_for(resource), depth_tag_for(resource)};
    tags[1].type = 4;
    const auto original_resource = resource;
    const auto original_tag = tags[0];
    const base_structure *inputs[] {&view.base};
    const auto before = testing::counts();
    require(call_v2_constants(constants, token.ref(), view) == 0, "v2 constants success changed");
    require(constants_call.calls == 1 && constants_call.first == &constants && constants_call.second == &token.ref() && constants_call.third == &view, "v2 constants references changed");
    require(call_v2_tag(view, tags, 2, &commands) == 0, "v2 tag success changed");
    require(tag_call.calls == 1 && tag_call.first == &view && tag_call.second == tags && tag_call.third == &commands && tag_call.a == 2, "v2 tag arguments changed");
    require(call_v2_framed_tag(token.ref(), view, tags, 2, &commands) == 0, "v2 framed tag success changed");
    require(framed_tag_call.calls == 1 && framed_tag_call.first == &token.ref() && framed_tag_call.second == &view && framed_tag_call.third == tags && framed_tag_call.fourth == &commands && framed_tag_call.a == 2, "v2 framed tag arguments changed");
    require(call_v2_evaluate(0x87654321u, token.ref(), inputs, 1, &commands) == 0, "v2 evaluate success changed");
    require(evaluate_call.calls == 1 && evaluate_call.first == &token.ref() && evaluate_call.second == inputs && evaluate_call.third == &commands && evaluate_call.a == 0x87654321u && evaluate_call.b == 1, "v2 evaluate arguments changed");
    require(std::memcmp(&constants, &original_constants, sizeof(constants)) == 0 && std::memcmp(&resource, &original_resource, sizeof(resource)) == 0 && std::memcmp(&tags[0], &original_tag, sizeof(original_tag)) == 0, "v2 input mutated");
    const auto after = testing::counts();
    require(after.constants == before.constants + 1 && after.tags == before.tags + 2 && after.evaluations == before.evaluations + 1, "v2 successful calls not observed exactly once");

    opaque_token failed_token;
    int replacement_native {};
    auto replacement_resource = modern_resource(&replacement_native);
    auto replacement_tag = depth_tag_for(replacement_resource);
    for (const std::int32_t failure : {1, -7, 0x1234567}) {
      result_v2 = failure;
      require(call_v2_constants(constants, failed_token.ref(), view) == failure, "v2 constants failure changed");
      require(call_v2_tag(view, nullptr, 0, nullptr) == failure, "v2 tag failure changed");
      require(tag_call.second == nullptr && tag_call.third == nullptr && tag_call.a == 0, "v2 empty tag arguments replaced");
      require(call_v2_framed_tag(token.ref(), view, nullptr, 0, nullptr) == failure, "v2 framed tag failure changed");
      require(call_v2_evaluate(7, token.ref(), nullptr, 0, nullptr) == failure, "v2 evaluate failure changed");
      require(evaluate_call.second == nullptr && evaluate_call.third == nullptr && evaluate_call.b == 0, "v2 empty evaluate inputs replaced");
      require(call_v2_framed_tag(failed_token.ref(), view, &replacement_tag, 1, &commands) == failure, "v2 replacement tag failure changed");
      const base_structure *replacement_inputs[] {&view.base, &replacement_tag.base};
      require(call_v2_evaluate(7, failed_token.ref(), replacement_inputs, 2, &commands) == failure, "v2 replacement local tag failure changed");
      const auto retained = latest(view.value);
      require(retained.found && retained.camera && retained.same_token && retained.resource == reinterpret_cast<std::uintptr_t>(&native), "v2 failed constants/tag/evaluation overwrote successful state");
    }
  }

  void test_v1_pairing() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 pairing install failed");
    const auto constants = old_camera();
    int native_a {}, native_b {};
    abi_v1::resource a {}, b {};
    a.native = &native_a;
    b.native = &native_b;
    require(call_v1_constants(constants, 101, 1), "v1 pairing constants failed");
    require(call_v1_tag(&b, 0, 2, nullptr), "v1 pairing tag failed");
    auto view_b = latest(2);
    require(view_b.found && !view_b.camera && view_b.resource == reinterpret_cast<std::uintptr_t>(&native_b), "v1 unrelated viewport inherited a camera");
    require(call_v1_tag(&a, 0, 1, nullptr), "v1 first viewport tag failed");
    const auto view_a = latest(1);
    require(view_a.found && view_a.camera && !view_a.exact && !view_a.same_token && view_a.resource == reinterpret_cast<std::uintptr_t>(&native_a), "v1 unframed tag misreported as exact numeric-frame pairing");
    require(call_v1_constants(constants, 202, 2), "v1 second camera failed");
    view_b = latest(2);
    require(view_b.camera && !view_b.exact, "v1 global tag acquired unsupported exact frame proof");
    require(call_v1_tag(nullptr, 0, 1, nullptr), "v1 depth untag failed");
    require(latest(1).found && latest(1).resource == 0, "v1 successful untag retained stale native depth");
    require(latest(2).resource == reinterpret_cast<std::uintptr_t>(&native_b), "v1 untag crossed viewports");
  }

  void test_v2_pairing() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, v2_targets), "v2 pairing install failed");
    const auto constants = modern_camera();
    auto first = viewport(11), second = viewport(12);
    opaque_token token_a, token_b;
    int native_a {}, native_b {}, native_local {};
    auto resource_a = modern_resource(&native_a), resource_b = modern_resource(&native_b);
    auto resource_local = modern_resource(&native_local);
    auto tag_a = depth_tag_for(resource_a), tag_b = depth_tag_for(resource_b);
    auto tag_local = depth_tag_for(resource_local);
    require(call_v2_constants(constants, token_a.ref(), first) == 0, "v2 first camera failed");
    require(call_v2_framed_tag(token_b.ref(), first, &tag_a, 1, nullptr) == 0, "v2 different-token tag failed");
    auto a = latest(11);
    require(a.found && a.camera && !a.same_token && !a.exact, "v2 different tokens incorrectly paired");
    require(call_v2_framed_tag(token_a.ref(), first, &tag_a, 1, nullptr) == 0, "v2 same-token tag failed");
    a = latest(11);
    require(a.same_token && !a.exact, "v2 opaque token identity promoted to exact frame number");
    // Reusing storage can mean a later frame. It must never become numeric proof.
    token_a.identity[1] = 999;
    require(call_v2_constants(constants, token_a.ref(), first) == 0 && !latest(11).exact, "v2 reused token address promoted to exact frame number");
    require(call_v2_framed_tag(token_b.ref(), second, &tag_b, 1, nullptr) == 0, "v2 second viewport tag failed");
    require(latest(12).found && !latest(12).camera && !latest(12).same_token, "v2 second viewport inherited first camera/token");
    require(call_v2_constants(constants, token_b.ref(), second) == 0 && latest(12).same_token, "v2 second viewport failed to retain its own token pairing");

    const base_structure *inputs[] {&first.base, &tag_local.base};
    require(call_v2_evaluate(0, token_b.ref(), inputs, 2, nullptr) == 0, "v2 evaluation-local tag failed");
    a = latest(11);
    require(a.resource == reinterpret_cast<std::uintptr_t>(&native_local) && !a.same_token && !a.exact, "v2 local evaluation lost call-specific resource/token identity");
    require(latest(12).resource == reinterpret_cast<std::uintptr_t>(&native_b), "v2 local tag crossed viewports");
    require(call_v2_constants(constants, token_b.ref(), first) == 0 && latest(11).same_token, "v2 same local token not observable");
    auto untag = tag_local;
    untag.resource_ptr = nullptr;
    const base_structure *empty_inputs[] {&first.base, &untag.base};
    require(call_v2_evaluate(0, token_b.ref(), empty_inputs, 2, nullptr) == 0, "v2 local depth untag failed");
    require(latest(11).found && latest(11).resource == 0, "v2 local untag retained stale depth");
  }

  void test_absent_unsupported_and_delayed_install() {
    fixture cleanup;
    require(GetModuleHandleW(L"sl.interposer.dll") == nullptr, "isolated test unexpectedly contains Streamline DLL");
    initialize(nullptr);
    poll({});  // Real discovery path must remain inert when middleware is absent.
    require(GetModuleHandleW(L"sl.interposer.dll") == nullptr, "probe loaded middleware during discovery");
    require(!testing::install(testing::abi::unsupported, v2_targets), "unsupported ABI installed hooks");
    require(!testing::install(testing::abi::v2_7_30, {}), "empty export set installed hooks");
    auto incomplete = v2_targets;
    incomplete.tag_for_frame = nullptr;
    require(!testing::install(testing::abi::v2_7_30, incomplete), "incomplete v2 exports installed hooks");
    auto constants = modern_camera();
    auto view = viewport(41);
    opaque_token token;
    require(call_v2_constants(constants, token.ref(), view) == 0, "unhooked fixture call failed");
    require(testing::counts().constants == 0, "rejected install left a hook active");

    // Exercise actual MinHook rollback after creating one hook but encountering
    // a non-executable second target. This is a scoped data object, never called.
    int non_executable {};
    auto broken = v2_targets;
    broken.tag = &non_executable;
    require(!testing::install(testing::abi::v2_7_30, broken), "non-executable target accepted");
    require(call_v2_constants(constants, token.ref(), view) == 0 && testing::counts().constants == 0, "partial create failure left constants hook active");
    // The address injection seam models availability after absence; it does not
    // claim coverage of DLL version resources or module-enumeration races.
    require(testing::install(testing::abi::v2_7_30, v2_targets), "delayed install failed after prior absence/rejection");
    require(call_v2_constants(constants, token.ref(), view) == 0 && testing::counts().constants == 1, "delayed hook did not begin observing");
  }

  void test_diagnostic_opt_in() {
    fixture cleanup;
    for (const char *setting : {static_cast<const char *>(nullptr), "0", "invalid"}) {
      probe_setting = setting;
      initialize(nullptr);
      require(!enabled(), "Missing, disabled or invalid diagnostic setting enabled observation");
      require(!testing::install(testing::abi::v1_1_1, v1_targets), "Disabled probe installed middleware hooks");
      const auto before = testing::counts();
      const auto native_before = native_discard::counts();
      command_initialized(100, 10, 1);
      command_reset(100, 10, 1);
      queue_initialized(200, 10, 2);
      depth_resource_initialized(300, 3, 10, true);
      depth_content_write(100, 300, 3);
      depth_content_invalidate(100, 300, 3);
      const auto copy = record_depth_copy(100, {300, 3}, {400, 4});
      command_closed(100);
      command_executed(200, 100);
      observe_native_discard_command(1); // Disabled discovery must not even read this invalid pointer.
      poll({});
      require(!capture_command_marker(100) && !copy.recording, "Disabled probe retained command/copy tracking");
      evaluation_snapshot evaluation;
      require(query_evaluation({}, evaluation) == evidence_status::inactive, "Disabled probe exposed camera evidence");
      const auto after = testing::counts();
      unchanged(before, after, "Disabled probe observed middleware calls");
      require(before.dropped == after.dropped && before.invalid == after.invalid,
        "Disabled tracking changed loss counters");
      const auto native_after = native_discard::counts();
      require(native_after.targets == native_before.targets && native_after.installed == native_before.installed &&
        native_after.unreadable == native_before.unreadable && native_after.dropped == native_before.dropped,
        "Disabled probe discovered or installed native hooks");
    }
    probe_setting = "1";
    initialize(nullptr);
    require(enabled() && testing::install(testing::abi::v1_1_1, v1_targets), "Explicit diagnostic opt-in did not enable hooks");
    const auto constants = old_camera();
    require(call_v1_constants(constants, 1, 1) && testing::counts().constants == 1, "Opted-in middleware was not observed");
    command_initialized(100, 10, 1);
    command_reset(100, 10, 1);
    require(bool(capture_command_marker(100)), "Opted-in command tracking was unavailable");
    shutdown();
    probe_setting = "0";
    initialize(nullptr);
    const auto before = testing::counts();
    require(!enabled() && call_v1_constants(constants, 2, 1), "Disabling retained hooks broke pass-through");
    unchanged(before, testing::counts(), "Reinitialized disabled probe still observed retained hooks");
    require(!capture_command_marker(100), "Disabling probe retained old command evidence");
  }

  void test_shutdown(bool modern, bool restart = false) {
    fixture cleanup;
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1, modern ? v2_targets : v1_targets), "shutdown fixture install failed");
    auto old = old_camera();
    auto current = modern_camera();
    auto view = viewport(51);
    opaque_token token;
    block_original.store(true, std::memory_order_release);
    std::int32_t returned = -1;
    std::thread worker([&] {
      returned = modern ? call_v2_constants(current, token.ref(), view) :
                          (call_v1_constants(old, 123, view.value) ? 0 : -1);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool entered = original_entered.load(std::memory_order_acquire);
    const auto before = testing::counts();
    shutdown();  // Must return even while a game call is inside its original function.
    if (restart) {
      initialize(nullptr);  // A late old call must not enter this new observation epoch.
    }
    release_original.store(true, std::memory_order_release);
    worker.join();  // Only after this point may fixture cleanup remove trampolines.
    require(entered && returned == 0 && constants_call.calls == 1, "in-flight original call did not survive shutdown");
    require(!latest(view.value).camera, "in-flight callback published camera after shutdown");
    unchanged(before, testing::counts(), "shutdown callback altered observation counters");
    block_original = false;
    if (restart) {
      require(modern ? call_v2_constants(current, token.ref(), view) == 0 : call_v1_constants(old, 124, view.value), "new-epoch original call failed");
      require(latest(view.value).camera && testing::counts().constants == before.constants + 1, "new observation epoch failed to capture a fresh call");
      return;
    }
    result_v1 = false;
    result_v2 = -31;
    if (modern) {
      require(call_v2_constants(current, token.ref(), view) == -31, "v2 shutdown constants not pass-through");
      require(call_v2_tag(view, nullptr, 0, nullptr) == -31, "v2 shutdown tag not pass-through");
      require(call_v2_framed_tag(token.ref(), view, nullptr, 0, nullptr) == -31, "v2 shutdown framed tag not pass-through");
      require(call_v2_evaluate(0, token.ref(), nullptr, 0, nullptr) == -31, "v2 shutdown evaluate not pass-through");
    } else {
      require(!call_v1_constants(old, 124, view.value), "v1 shutdown constants not pass-through");
      require(!call_v1_tag(nullptr, 0, view.value, nullptr), "v1 shutdown tag not pass-through");
      require(!call_v1_evaluate(nullptr, 0, 124, view.value), "v1 shutdown evaluate not pass-through");
    }
    unchanged(before, testing::counts(), "pass-through calls observed after shutdown");
  }

  template<class Callback>
  void preserves_last_error(Callback call, const invocation &record) {
    SetLastError(incoming_error);
    const auto result = call();
    const auto after = GetLastError();
    require(result, "LastError fixture original result changed");
    require(record.incoming_error == incoming_error, "hook changed original incoming LastError");
    require(after == outgoing_error, "hook changed original outgoing LastError");
  }

  void test_last_error(bool modern) {
    fixture cleanup;
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1, modern ? v2_targets : v1_targets), "LastError fixture install failed");
    auto old = old_camera();
    auto current = modern_camera();
    auto view = viewport(61);
    opaque_token token;
    int native {};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    abi_v1::resource old_resource {};
    old_resource.native = &native;
    const base_structure *inputs[] {&view.base, &tag.base};
    // Test both success and failure: failed originals can also leave LastError.
    for (const bool success : {true, false}) {
      result_v1 = success;
      result_v2 = success ? 0 : -99;
      if (modern) {
        preserves_last_error([&] {
          return call_v2_constants(current, token.ref(), view) == result_v2;
        },
                             constants_call);
        preserves_last_error([&] {
          return call_v2_tag(view, &tag, 1, nullptr) == result_v2;
        },
                             tag_call);
        preserves_last_error([&] {
          return call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr) == result_v2;
        },
                             framed_tag_call);
        preserves_last_error([&] {
          return call_v2_evaluate(0, token.ref(), inputs, 2, nullptr) == result_v2;
        },
                             evaluate_call);
      } else {
        preserves_last_error([&] {
          return call_v1_constants(old, 72, view.value) == result_v1;
        },
                             constants_call);
        preserves_last_error([&] {
          return call_v1_tag(&old_resource, 0, view.value, nullptr) == result_v1;
        },
                             tag_call);
        preserves_last_error([&] {
          return call_v1_evaluate(nullptr, 0, 72, view.value) == result_v1;
        },
                             evaluate_call);
      }
    }
  }

  void test_v2_depth_kinds_and_chained_inputs() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, v2_targets), "v2 typed-tag install failed");
    auto constants = modern_camera();
    auto view = viewport(71);
    opaque_token token;
    int native_raw {}, native_hires {}, native_linear {};
    auto raw = modern_resource(&native_raw), high = modern_resource(&native_hires);
    auto linear = modern_resource(&native_linear);
    auto raw_tag = depth_tag_for(raw), high_tag = depth_tag_for(high), linear_tag = depth_tag_for(linear);
    high_tag.type = 48;
    linear_tag.type = 49;
    const abi_v2::resource_tag mixed[] {linear_tag, high_tag};
    require(call_v2_tag(view, mixed, 2, nullptr) == 0, "v2 mixed HiResDepth/LinearDepth call failed");
    require(latest(view.value).resource == reinterpret_cast<std::uintptr_t>(&native_hires), "v2 HiResDepth missing or confused with LinearDepth");
    require(call_v2_tag(view, &raw_tag, 1, nullptr) == 0, "v2 raw depth call failed");
    require(latest(view.value).resource == reinterpret_cast<std::uintptr_t>(&native_raw), "v2 raw depth did not retain its own typed record");
    auto linear_view = viewport(72);
    require(call_v2_tag(linear_view, &linear_tag, 1, nullptr) == 0 && latest(linear_view.value).resource == reinterpret_cast<std::uintptr_t>(&native_linear), "v2 linear-depth diagnostics were dropped");

    // Legal per-evaluation inputs may be one linked chain, rather than an array
    // of independent structures. Traversal must preserve the original links.
    auto local_view = viewport(73);
    local_view.base.next = &high_tag.base;
    const auto original_view = local_view;
    const base_structure *chain[] {&local_view.base};
    auto constants_view = viewport(local_view.value);
    require(call_v2_constants(constants, token.ref(), constants_view) == 0, "v2 chained-input camera failed");
    require(call_v2_evaluate(0, token.ref(), chain, 1, nullptr) == 0, "v2 chained evaluation failed");
    const auto local = latest(local_view.value);
    require(local.found && local.camera && local.same_token && !local.exact && local.resource == reinterpret_cast<std::uintptr_t>(&native_hires), "v2 legal chained inputs were not associated");
    require(std::memcmp(&local_view, &original_view, sizeof(local_view)) == 0, "v2 observer mutated input-chain links");
    require(evaluate_call.second == chain && evaluate_call.b == 1, "v2 chain array/count changed");

    // The original accepts an oversized array in this synthetic test. Observation
    // must bound its reads and invalidate its own cache, while forwarding as-is.
    require(call_v2_tag(view, &raw_tag, 33, nullptr) == 0, "v2 oversized-array result changed");
    require(tag_call.second == &raw_tag && tag_call.a == 33, "v2 oversized-array arguments changed");
    require(latest(view.value).found && latest(view.value).resource == 0 && testing::counts().invalid > 0, "v2 unsupported tag array left old resource looking current");
  }
  selected_depth selection(void *resource) {
    selected_depth value;
    value.resource = reinterpret_cast<std::uintptr_t>(resource);
    value.source_id = 7;
    value.layout_epoch = 3;
    value.frame_index = 999; // Deliberately unrelated to the Streamline frame.
    value.width = value.active_width = 1920;
    value.height = value.active_height = 1080;
    value.ready = true;
    return value;
  }
  void expect_evidence(const selected_depth &selected, evidence_status expected, const char *message) {
    evaluation_snapshot value;
    const auto actual = query_evaluation(selected, value);
    if (actual != expected) std::fprintf(stderr, "evidence expected %s, got %s\n", name(expected), name(actual));
    require(actual == expected && value.status == actual, message);
  }
  void mint(opaque_token &token, const std::uint32_t *index) {
    returned_token = const_cast<abi_v2::frame_token *>(&token.ref());
    abi_v2::frame_token *output{};
    SetLastError(incoming_error);
    require(call_v2_new_token(output, index) == 0, "frame token result changed");
    require(output == returned_token && new_token_call.first == &output && new_token_call.second == index,
      "frame token forwarding changed");
    require(new_token_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
      "frame token LastError changed");
  }
  struct token_source_fixture {
    fixture cleanup;
    abi_v2::viewport view{viewport(61)};
    abi_v2::constants constants{modern_camera()};
    opaque_token token;
    int native{};
    abi_v2::resource resource{modern_resource(&native)};
    abi_v2::resource_tag tag{depth_tag_for(resource)};
    const base_structure *inputs[1]{&view.base};
    std::uint32_t number{401};
    token_source_fixture() {
      probe_setting = "0"; source_setting = "1"; initialize(nullptr);
      require(!enabled() && source_enabled(), "Token isolation enabled the optional full probe");
      require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "Token isolation hook install failed");
      depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&native), 91, 10, true);
      mint(token, &number);
      publish(token);
    }
    evaluation_snapshot publish(opaque_token &current) {
      require(call_v2_constants(constants, current.ref(), view) == 0 &&
          call_v2_framed_tag(current.ref(), view, &tag, 1, nullptr) == 0 &&
          call_v2_evaluate(0, current.ref(), inputs, 1, nullptr) == 0,
        "Token fixture changed native API forwarding");
      sunshine_scene_depth::frame normalized;
      evaluation_snapshot snapshot;
      require(testing::normalized_source(view.value, 0, normalized) && normalized.projection.supplied &&
          normalized.projection.depth_scale == .5 && normalized.observation_revision == depth_observation_revision() &&
          testing::latest_snapshot(view.value, snapshot) && snapshot.frame.kind == frame_identity_kind::v2_observed_token,
        "Registered token did not establish its own fresh source tuple");
      return snapshot;
    }
    bool ready() {
      depth_source_snapshot snapshot;
      return query_depth_source(snapshot) == evidence_status::source_associated_evaluation;
    }
  };
  enum class fixture_lock { records_exclusive, records_shared, tokens };
  struct observation_guard {
    fixture_lock kind;
    bool locked{true};
    explicit observation_guard(fixture_lock value): kind(value) {
      if (kind == fixture_lock::records_exclusive) testing::lock_records();
      else if (kind == fixture_lock::records_shared) testing::lock_records_shared();
      else testing::lock_tokens();
    }
    ~observation_guard() { release(); }
    void release() {
      if (!locked) return;
      if (kind == fixture_lock::records_exclusive) testing::unlock_records();
      else if (kind == fixture_lock::records_shared) testing::unlock_records_shared();
      else testing::unlock_tokens();
      locked = false;
    }
  };
  bool bounded_finish(const std::atomic<bool> &done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done.load(std::memory_order_acquire);
  }
  void test_token_records_independence(bool shared) {
    token_source_fixture source;
    opaque_token next;
    const std::uint32_t number = 402;
    returned_token = const_cast<abi_v2::frame_token *>(&next.ref());
    abi_v2::frame_token *output{};
    const auto before = depth_observation_revision();
    const auto counters = testing::counts();
    const auto calls = new_token_call.calls;
    std::atomic<bool> done{};
    bool forwarded{};
    observation_guard held(shared ? fixture_lock::records_shared : fixture_lock::records_exclusive);
    std::thread worker([&] {
      SetLastError(incoming_error);
      const auto result = call_v2_new_token(output, &number);
      const auto error = GetLastError();
      forwarded = result == 0 && output == returned_token && error == outgoing_error &&
        new_token_call.incoming_error == incoming_error && new_token_call.first == &output && new_token_call.second == &number;
      done.store(true, std::memory_order_release);
    });
    const bool nonblocking = bounded_finish(done);
    held.release(); worker.join();
    require(nonblocking && forwarded && new_token_call.calls == calls + 1,
      "Token registration blocked on camera metadata or changed SDK forwarding");
    require(depth_observation_revision() == before && testing::counts().dropped == counters.dropped &&
        testing::counts().invalid == counters.invalid,
      shared ? "Shared camera-table reader invalidated independent token registration" :
        "Exclusive camera-table owner invalidated independent token registration");
    require(source.ready(), "Unrelated token registration revoked an existing camera/source tuple");
    const auto fresh = source.publish(next);
    require(fresh.frame.has_numeric && fresh.frame.numeric == number && fresh.frame.token == reinterpret_cast<std::uintptr_t>(returned_token),
      "Token registered during camera-table contention lost its exact identity");
  }
  void test_token_contention_and_remint() {
    token_source_fixture source;
    evaluation_snapshot initial;
    require(testing::latest_snapshot(source.view.value, initial), "Token fixture omitted initial identity");
    const auto before_remint = depth_observation_revision();
    const auto before_counts = testing::counts();
    mint(source.token, &source.number); // Same pointer and numeric frame, new generation.
    require(depth_observation_revision() == before_remint && testing::counts().dropped == before_counts.dropped &&
        !source.ready(), "Same-address token remint reused old identity or triggered broad observation loss");
    const auto reminted = source.publish(source.token);
    require(reminted.frame.generation > initial.frame.generation && reminted.frame.numeric == initial.frame.numeric,
      "Same-address token remint failed to establish a distinct exact generation");

    // An unfinished writer cannot hide the last immutable token publication.
    // Actual lost token registration below must still revoke that identity.
    std::atomic<bool> queried{};
    bool accepted = true;
    observation_guard held(fixture_lock::tokens);
    std::thread reader([&] { accepted = source.ready(); queried.store(true, std::memory_order_release); });
    const bool query_nonblocking = bounded_finish(queried);
    held.release(); reader.join();
    require(query_nonblocking && accepted && depth_observation_revision() == before_remint,
      "Unfinished token writer hid the last complete identity or changed observation history");
    require(source.ready(), "A read-only token lookup permanently revoked unchanged identity");

    returned_token = const_cast<abi_v2::frame_token *>(&source.token.ref());
    abi_v2::frame_token *output{};
    std::atomic<bool> done{};
    bool forwarded{};
    const auto before = depth_observation_revision();
    const auto counters = testing::counts();
    observation_guard busy(fixture_lock::tokens);
    std::thread writer([&] {
      SetLastError(incoming_error);
      const auto result = call_v2_new_token(output, &source.number);
      const auto error = GetLastError();
      forwarded = result == 0 && output == returned_token && error == outgoing_error &&
        new_token_call.incoming_error == incoming_error && new_token_call.first == &output && new_token_call.second == &source.number;
      done.store(true, std::memory_order_release);
    });
    const bool nonblocking = bounded_finish(done);
    busy.release(); writer.join();
    require(nonblocking && forwarded && depth_observation_revision() == before + 1 &&
        testing::counts().dropped == counters.dropped + 1 && testing::counts().invalid == counters.invalid,
      "Real token-table contention lost its bounded fail-closed registration behavior");
    loss_diagnostics::event loss;
    require(query_depth_observation_loss(before + 1, loss) && loss.site &&
        std::strcmp(loss.site, "hook_new_token_v2") == 0 && loss.details.sequence &&
        loss.details.has_sdk_result && loss.details.sdk_result == 0 && !source.ready() && !source.ready(),
      "Lost token registration was unlabeled or revived old source evidence after unlocking");
    mint(source.token, &source.number);
    require(!source.ready(), "Successful remint without new constants revived pre-loss source evidence");
    source.publish(source.token);
  }
  void test_pending_token_lifecycle() {
    token_source_fixture source;
    opaque_token pending;
    const std::uint32_t number = 403;
    returned_token = const_cast<abi_v2::frame_token *>(&pending.ref());
    abi_v2::frame_token *output{};
    bool forwarded{};
    block_original = true;
    std::thread worker([&] {
      SetLastError(incoming_error);
      const auto result = call_v2_new_token(output, &number);
      const auto error = GetLastError();
      forwarded = result == 0 && output == returned_token && error == outgoing_error &&
        new_token_call.incoming_error == incoming_error && new_token_call.first == &output && new_token_call.second == &number;
    });
    const bool entered = bounded_finish(original_entered);
    shutdown();
    initialize(nullptr); // The pending SDK call belongs to the retired epoch.
    const auto after_restart = depth_observation_revision();
    const auto counters = testing::counts();
    release_original = true;
    worker.join();
    block_original = false;
    require(entered && forwarded && depth_observation_revision() == after_restart,
      "Pending token call failed SDK forwarding or modified the restarted observation epoch");
    unchanged(counters, testing::counts(), "Retired token callback changed new-epoch observation counters");

    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&source.native), 92, 10, true);
    require(call_v2_constants(source.constants, pending.ref(), source.view) == 0 &&
        call_v2_framed_tag(pending.ref(), source.view, &source.tag, 1, nullptr) == 0 &&
        call_v2_evaluate(0, pending.ref(), source.inputs, 1, nullptr) == 0,
      "Post-restart token fixture changed native API forwarding");
    sunshine_scene_depth::frame normalized;
    require(!testing::normalized_source(source.view.value, 0, normalized) && !source.ready(),
      "Token returned from a retired callback populated the new epoch's source identity");
    mint(pending, &number);
    source.publish(pending);
  }
  void test_metadata_reader_isolation() {
    token_source_fixture source;
    evaluation_snapshot before;
    require(testing::latest_snapshot(source.view.value, before), "Metadata fixture missing initial source");
    const auto revision = depth_observation_revision();
    const auto counters = testing::counts();
    std::atomic<bool> done{};
    bool forwarded{};
    observation_guard held(fixture_lock::records_shared);
    std::thread worker([&] {
      SetLastError(incoming_error);
      const bool camera = call_v2_constants(source.constants, source.token.ref(), source.view) == 0 &&
        constants_call.incoming_error == incoming_error && GetLastError() == outgoing_error;
      SetLastError(incoming_error);
      const bool tag = call_v2_framed_tag(source.token.ref(), source.view, &source.tag, 1, nullptr) == 0 &&
        framed_tag_call.incoming_error == incoming_error && GetLastError() == outgoing_error;
      SetLastError(incoming_error);
      const bool evaluated = call_v2_evaluate(0, source.token.ref(), source.inputs, 1, nullptr) == 0 &&
        evaluate_call.incoming_error == incoming_error && GetLastError() == outgoing_error;
      forwarded = camera && tag && evaluated;
      done.store(true, std::memory_order_release);
    });
    const bool bounded = bounded_finish(done);
    held.release(); worker.join();
    require(bounded && forwarded, "Metadata callback blocked on a reader or changed native forwarding");
    require(depth_observation_revision() == revision && testing::counts().dropped == counters.dropped,
      "Published metadata reader invalidated camera/tag/evaluation observations");
    evaluation_snapshot after;
    require(testing::latest_snapshot(source.view.value, after) && source.ready() &&
        after.sequence > before.sequence && after.camera_sequence > before.camera_sequence &&
        after.loss_revision == revision && after.frame.generation == before.frame.generation,
      "Metadata reader prevented fresh exact camera/tag/evaluation publication");
  }
  void benchmark_metadata_callbacks() {
    token_source_fixture source;
    const auto update = [&] {
      call_v2_constants(source.constants, source.token.ref(), source.view);
      call_v2_framed_tag(source.token.ref(), source.view, &source.tag, 1, nullptr);
      call_v2_evaluate(0, source.token.ref(), source.inputs, 1, nullptr);
    };
    for (unsigned i = 0; i != 100; ++i) update();
    const auto revision = depth_observation_revision();
    constexpr unsigned iterations = 2000;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i != iterations; ++i) update();
    const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    require(source.ready() && depth_observation_revision() == revision,
      "Serial metadata benchmark lost valid source observations");
    std::printf("metadata_callback_bench iterations=%u total_us=%.3f camera_tag_evaluate_us=%.3f\n",
      iterations, elapsed, elapsed / iterations);
  }
#ifndef SUNSHINE_STREAMLINE_PROBE_LEGACY_CONTROL
  void test_token_reader_and_storage_pressure() {
    token_source_fixture source;
    struct pins_guard { ~pins_guard() { testing::unpin_tokens(); } } pins;
    const auto before = depth_observation_revision();
    const auto counters = testing::counts();
    opaque_token next;
    require(testing::pin_tokens(0), "Token publication could not be pinned");
    mint(next, &source.number);
    source.publish(next);
    require(depth_observation_revision() == before && testing::counts().dropped == counters.dropped,
      "Immutable token reader invalidated mint or camera/tag/evaluation callbacks");
    testing::unpin_tokens();
    // Eight distinct retained versions exhaust the fixed bank, not a mutable
    // metadata lock. Native API forwarding survives, but lost identity revokes.
    for (unsigned i = 0; i != 8; ++i) {
      require(testing::pin_tokens(i), "Token pressure fixture lost a published version");
      if (i != 7) mint(next, &source.number);
    }
    const auto exhausted = depth_observation_revision();
    mint(next, &source.number);
    loss_diagnostics::event loss;
    require(depth_observation_revision() == exhausted + 1 && !source.ready() &&
        query_depth_observation_loss(exhausted + 1, loss) && loss.cause == loss_diagnostics::reason::metadata_storage_busy,
      "Token version exhaustion did not preserve explicit fail-closed observation loss");
    testing::unpin_tokens();
    mint(next, &source.number);
    source.publish(next);
  }
  void test_pinned_metadata_lifecycle() {
    token_source_fixture source;
    observation_guard reader(fixture_lock::records_shared);
    struct pins_guard { ~pins_guard() { testing::unpin_tokens(); } } pins;
    require(testing::pin_tokens(0), "Lifecycle fixture omitted token version");
    shutdown();
    initialize(nullptr);
    require(!source.ready(), "Pinned retired version revived source after reinitialization");
    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&source.native), 92, 10, true);
    mint(source.token, &source.number);
    source.publish(source.token);
    require(source.ready(), "Held old metadata/token reader prevented fresh lifecycle publication");
  }
  void test_first_local_raw_entry() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "First-local-raw hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function,
      "First-local-raw fixture failed FG discovery"); // No options call or metadata publication.
    opaque_token token; mint(token, nullptr);
    auto view = viewport(73);
    int native{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&native), 94, 10, true);
    const base_structure *inputs[]{&view.base, &tag.base};
    testing::arm_entry_policy();
    require(call_v2_evaluate(0, token.ref(), inputs, 2, nullptr) == 0,
      "First-local-raw evaluation changed SDK result");
    bool selected{}, admitted{};
    require(testing::entry_policy(selected, admitted) && selected && admitted,
      "Initial local raw evaluation was rejected before SDK entry when metadata store was empty");
    sunshine_scene_depth::frame normalized;
    require(testing::normalized_source(view.value, 0, normalized) && !normalized.projection.supplied,
      "First local raw evaluation failed to publish its own camera-free source");
  }
#endif
  void test_source_nomination_v1() {
    fixture cleanup;
    probe_setting = nullptr;
    source_setting = nullptr; // Missing key means production nomination on, diagnostics off.
    initialize(nullptr);
    require(!enabled() && source_enabled(), "source default changed diagnostic opt-in");
    require(testing::install(testing::abi::v1_1_1, v1_targets), "production-only v1 install failed");
    int native{}, other{};
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    abi_v1::resource resource{}; resource.native = &native;
    auto constants = old_camera();
    depth_source_snapshot snapshot;
    const auto query = [&](evidence_status expected, const char *message) {
      const auto actual = query_depth_source(snapshot);
      if (actual != expected) std::fprintf(stderr, "source expected %s, got %s\n", name(expected), name(actual));
      require(actual == expected && snapshot.status == actual, message);
    };
    call_v1_constants(constants, 1, 7);
    call_v1_tag(&resource, 0, 7, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::missing_depth, "untracked resource was nominated");
    depth_resource_initialized(address, 41, 10, true);
    call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::missing_depth, "old untracked tag acquired a later lifetime");
    call_v1_tag(&resource, 0, 7, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "known v1 resource not nominated without selected-depth dependency");
    require(!snapshot.tags[0].present && snapshot.tags[1].present && snapshot.tags[1].value.native_resource == address &&
        snapshot.tags[1].resource_lifetime == 41 && snapshot.tags[1].device == 10 && snapshot.viewport == 7 &&
        snapshot.frame.numeric == 1 && snapshot.sequence && snapshot.epoch, "v1 nomination identity wrong");
    sunshine_scene_depth::frame normalized;
    require(testing::normalized_source(7, 0, normalized) && normalized.resource.native == address &&
        normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == .5 && normalized.projection.reversed &&
        normalized.proof == sunshine_scene_depth::state_proof::observed_nonzero &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_present,
      "v1 adapter did not normalize projection/resource/omitted-state authority");
    const auto first = snapshot;
    query(evidence_status::source_associated_evaluation, "query must be a value copy, consumer owns once/present admission");
    require(snapshot.sequence == first.sequence, "query fabricated another evaluation sequence");
    evaluation_snapshot diagnostic;
    require(query_evaluation(selection(&native), diagnostic) == evidence_status::inactive,
      "production nomination upgraded diagnostic/geometry evidence");
    command_initialized(100, 10, 1); command_reset(100, 10, 1); queue_initialized(200, 10, 2);
    const auto counters_before = testing::counts();
    const auto native_before = native_discard::counts();
    for (unsigned i = 0; i != 10000; ++i) { depth_content_write(100, address, 41); depth_content_invalidate(100, address, 41); }
    observe_native_discard_command(1);
    require(!capture_command_marker(100) && !record_depth_copy(100, {address, 41}, {600, 60}).recording,
      "production-only nomination enabled per-command/content tracking");
    require(testing::counts().dropped == counters_before.dropped && native_discard::counts().targets == native_before.targets,
      "production-only path observed expensive diagnostic events");
    query(evidence_status::source_associated_evaluation, "ignored draw diagnostics poisoned semantic nomination");
    depth_resource_destroyed(address, 41);
    depth_resource_initialized(address, 42, 10, true);
    query(evidence_status::missing_depth, "destroy/recreate accepted old evaluated lifetime");
    call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_mismatch, "stale global tag rebound at evaluation after ABA");
    call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "fresh tag did not admit new lifetime");
    require(snapshot.tags[1].resource_lifetime == 42, "new lifetime not frozen");
    // Lifetime changes DURING the original evaluation must not rebind at query.
    block_original = true;
    std::thread worker([&] { call_v1_evaluate(nullptr, 0, 1, 7); });
    while (!original_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    depth_resource_destroyed(address, 42); depth_resource_initialized(address, 43, 10, true);
    release_original = true; worker.join(); block_original = false;
    query(evidence_status::missing_depth, "in-flight evaluation acquired replacement lifetime");
    call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "post-ABA source did not recover");
    result_v1 = false; call_v1_evaluate(nullptr, 0, 1, 7); result_v1 = true;
    query(evidence_status::evaluation_failed, "failed evaluation reused last successful nomination");
    call_v1_constants(constants, 2, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 2, 7);
    testing::lose_observation();
    query(evidence_status::observation_lost, "lost semantic observation retained nomination");
    call_v1_constants(constants, 3, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 3, 7);
    query(evidence_status::source_associated_evaluation, "fresh source did not recover before age check");
    // The production clock may have a coarser tick than Sleep(2). Establish
    // actual elapsed clock time before testing the unchanged zero-ms limit.
    const auto age_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (GetTickCount64() <= snapshot.tick && std::chrono::steady_clock::now() < age_deadline) Sleep(1);
    require(GetTickCount64() > snapshot.tick, "source age clock did not advance within fixture deadline");
    require(query_depth_source(snapshot, 0) == evidence_status::stale, "age limit ignored");
    constants.reset = 1; call_v1_constants(constants, 4, 7); call_v1_evaluate(nullptr, 0, 4, 7);
    query(evidence_status::observation_lost, "reset camera nominated depth tags from the old scene");
    constants.reset = 0; call_v1_constants(constants, 5, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 5, 7);
    resource.native = &other;
    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&other), 50, 10, true);
    call_v1_constants(constants, 5, 8); call_v1_tag(&resource, 0, 8, nullptr); call_v1_evaluate(nullptr, 0, 5, 8);
    query(evidence_status::ambiguous_viewport, "multiple active viewports silently chose newest");
    source_setting = "0"; initialize(nullptr);
    require(!source_enabled() && !enabled(), "source rollback key ignored");
    query(evidence_status::inactive, "disabled source nomination remained live");
    const auto stopped = testing::counts(); call_v1_constants(constants, 6, 7);
    unchanged(stopped, testing::counts(), "disabled retained hooks did not pass through");
  }

  void test_source_lifecycle_contention() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v1_1_1, v1_targets), "source contention hook install failed");
    int native_a{}, native_b{};
    const auto a = reinterpret_cast<std::uintptr_t>(&native_a), b = reinterpret_cast<std::uintptr_t>(&native_b);
    depth_resource_initialized(a, 41, 10, true);
    auto constants = old_camera(); abi_v1::resource resource{}; resource.native = &native_a;
    call_v1_constants(constants, 1, 1); call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 1, 1);
    depth_source_snapshot snapshot;
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation, "contention fixture did not start ready");
    const auto original_epoch = snapshot.epoch;
    const auto wait_for = [](const auto &predicate) {
      const auto deadline = GetTickCount64() + 2000;
      while (!predicate() && GetTickCount64() < deadline) Sleep(1);
      return predicate();
    };
    testing::lock_source_resources();
    std::thread writer([&] { depth_resource_initialized(b, 50, 10, true); });
    const bool queued = wait_for([] { return testing::waiting_source_resources() == 1; });
    testing::unlock_source_resources(); writer.join();
    require(queued, "lifecycle writer did not reach deterministic lock contention");
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.tags[1].resource_lifetime == 41, "contended lifecycle update forgot existing live resource");
    resource.native = &native_b; call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 1, 1);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation && snapshot.tags[1].resource_lifetime == 50,
      "contended lifecycle update was dropped");
    testing::lock_source_resources();
    std::thread destroyer([&] { depth_resource_destroyed(b, 50); });
    const bool destroy_queued = wait_for([] { return testing::waiting_source_resources() == 1; });
    testing::unlock_source_resources(); destroyer.join();
    require(destroy_queued && query_depth_source(snapshot) == evidence_status::missing_depth,
      "contended destruction left dead lifetime nominatable");

    // An old callback already queued on the table cannot enter a new epoch,
    // regardless of which waiting thread obtains the table first.
    testing::lock_source_resources();
    std::thread old_writer([&] { depth_resource_initialized(b, 51, 10, true); });
    const bool old_queued = wait_for([] { return testing::waiting_source_resources() == 1; });
    std::thread restart([] { initialize(nullptr); });
    const bool disabled = wait_for([] { return !source_enabled(); });
    testing::unlock_source_resources(); old_writer.join(); restart.join();
    require(old_queued && disabled && source_enabled(), "reinitialization contention setup failed");
    call_v1_constants(constants, 2, 1); call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 2, 1);
    require(query_depth_source(snapshot) == evidence_status::missing_depth && snapshot.epoch != original_epoch,
      "queued old-epoch lifecycle event repopulated reinitialized registry");
    depth_resource_initialized(b, 52, 10, true);
    call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 2, 1);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation && snapshot.tags[1].resource_lifetime == 52,
      "fresh lifecycle event failed after reinitialization");
  }

  void test_source_nomination_v2() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "production-only v2 install failed");
    auto view = viewport(3); auto constants = modern_camera(); opaque_token token;
    int native_a{}, native_b{}, native_high{};
    auto resource_a = modern_resource(&native_a), resource_b = modern_resource(&native_b), resource_high = modern_resource(&native_high);
    auto raw = depth_tag_for(resource_a), local = depth_tag_for(resource_b), high = depth_tag_for(resource_high); high.type = 48;
    for (const auto *resource : {&resource_a, &resource_b, &resource_high}) {
      const auto native = reinterpret_cast<std::uintptr_t>(resource->native);
      depth_resource_initialized(native, native, 10, true);
    }
    const base_structure *inputs[]{&view.base};
    const base_structure *local_inputs[]{&view.base, &local.base};
    depth_source_snapshot snapshot;
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &raw, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(query_depth_source(snapshot) == evidence_status::untracked_frame, "opaque token address became source frame");
    mint(token, nullptr);
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &raw, 1, nullptr);
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), local_inputs, 2, nullptr);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation && snapshot.tags[0].present &&
        snapshot.tags[0].value.type == 48 && snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_high) &&
        snapshot.tags[1].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_b) && !snapshot.frame.has_numeric,
      "v2 preference/local override/opaque generation nomination wrong");
    sunshine_scene_depth::frame normalized;
    require(testing::normalized_source(view.value, 0, normalized) &&
        normalized.resource.native == reinterpret_cast<std::uintptr_t>(&native_b) && normalized.resource.width == 1920 &&
        normalized.resource.height == 1080 && normalized.resource.kind == sunshine_scene_depth::resource_kind::raw_depth &&
        normalized.proof == sunshine_scene_depth::state_proof::declared &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_present,
      "v2 adapter lost local source/extent/explicit state authority");
    require(testing::normalized_source(view.value, 1, normalized) &&
        normalized.resource.kind == sunshine_scene_depth::resource_kind::display_depth,
      "v2 high-resolution tag lost provider-neutral semantic kind");
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.tags[1].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_a), "local source leaked to later evaluation");
    high.lifecycle = UINT32_MAX;
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(query_depth_source(snapshot) != evidence_status::source_associated_evaluation,
      "malformed v2 lifecycle inherited the v1 unspecified sentinel");
    high.lifecycle = 1; call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation, "v2 lifecycle recovery failed");
    high.area.width = 1921; call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    require(query_depth_source(snapshot) != evidence_status::source_associated_evaluation, "malformed extent left old nomination active");
    high.area.width = 1920; call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(query_depth_source(snapshot) == evidence_status::source_associated_evaluation, "valid v2 source did not recover");
    mint(token, nullptr);
    require(query_depth_source(snapshot) == evidence_status::untracked_frame, "token reuse retained source nomination");
  }

  void test_jitter_frame_and_render_domain(bool modern) {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "jitter source hook install failed");
    auto view = viewport(9);
    int native{};
    abi_v1::resource old_resource{}; old_resource.native = &native;
    auto resource = modern_resource(&native); resource.width = 2228; resource.height = 1256;
    auto tag = depth_tag_for(resource); tag.area = {0, 0, 2228, 1253};
    const extent area = tag.area;
    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&native), 91, 10, true);
    opaque_token tokens[4];
    const base_structure *inputs[]{&view.base};
    const auto constants = [&](unsigned frame, float x, float y) {
      if (modern) {
        mint(tokens[frame-1], nullptr);
        auto camera = modern_camera(); camera.common.jitter_offset[0] = x; camera.common.jitter_offset[1] = y;
        call_v2_constants(camera, tokens[frame-1].ref(), view);
      } else {
        auto camera = old_camera(); camera.common.jitter_offset[0] = x; camera.common.jitter_offset[1] = y;
        call_v1_constants(camera, frame, view.value);
      }
    };
    const auto evaluate = [&](unsigned frame) {
      if (modern) {
        call_v2_framed_tag(tokens[frame-1].ref(), view, &tag, 1, nullptr);
        call_v2_evaluate(0, tokens[frame-1].ref(), inputs, 1, nullptr);
      } else {
        call_v1_tag(&old_resource, 0, view.value, &area);
        call_v1_evaluate(nullptr, 0, frame, view.value);
      }
    };
    constants(1, .25f, -.375f);
    constants(2, -.125f, .5f); // Newer constants must not contaminate frame 1.
    evaluate(1);
    sunshine_scene_depth::frame first, current;
    require(testing::normalized_source(view.value, 0, first) && first.jitter.supplied &&
        first.jitter.x == .25f && first.jitter.y == -.375f && first.jitter.width == 2228 && first.jitter.height == 1253,
      "Depth jitter borrowed newer camera offsets or allocation padding");
    evaluate(2);
    require(testing::normalized_source(view.value, 0, current) && current.jitter.supplied &&
        current.jitter.x == -.125f && current.jitter.y == .5f && first.jitter.x == .25f,
      "Per-frame jitter snapshot was not immutable");
    constants(3, std::numeric_limits<float>::quiet_NaN(), .5f); evaluate(3);
    require(testing::normalized_source(view.value, 0, current) && current.projection.supplied && !current.jitter.supplied,
      "Nonfinite jitter rejected usable depth or reused previous correction");
    if (modern) mint(tokens[3], nullptr); // No constants for this frame.
    evaluate(4);
    require(testing::normalized_source(view.value, 0, current) && !current.jitter.supplied,
      "Camera-free depth inherited another frame's jitter");
  }

  void test_high_resolution_jitter_domain() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "high-res jitter hook install failed");
    auto view = viewport(6); auto constants = modern_camera(); opaque_token token;
    constants.common.jitter_offset[0] = .25f; constants.common.jitter_offset[1] = -.375f;
    int raw_native{}, high_native{};
    auto raw_resource = modern_resource(&raw_native), high_resource = modern_resource(&high_native);
    raw_resource.width = 2228; raw_resource.height = 1256;
    high_resource.width = 3840; high_resource.height = 2160;
    auto raw = depth_tag_for(raw_resource), high = depth_tag_for(high_resource);
    raw.area = {0, 0, 2228, 1253}; high.area = {0, 0, 3840, 2160}; high.type = 48;
    for (auto *native : {&raw_native, &high_native})
      depth_resource_initialized(reinterpret_cast<std::uintptr_t>(native), reinterpret_cast<std::uintptr_t>(native), 10, true);
    mint(token, nullptr); call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); // An old global raw tag cannot establish the high-res frame's render size.
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    const base_structure *inputs[]{&view.base};
    const auto evaluate = [&] { call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); };
    sunshine_scene_depth::frame value;
    evaluate();
    require(testing::normalized_source(view.value, 1, value) && !value.jitter.supplied,
      "High-res jitter used display dimensions or an uncorrelated global render extent");
    call_v2_framed_tag(token.ref(), view, &raw, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 1, value) && value.jitter.supplied &&
        value.jitter.width == 2228 && value.jitter.height == 1253 && value.jitter.x == .25f && value.jitter.y == -.375f,
      "High-res depth did not retain same-frame raw render-pixel jitter units");
    // The next frame cannot reuse the previous frame's render-domain proof.
    mint(token, nullptr); call_v2_constants(constants, token.ref(), view);
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    evaluate();
    require(testing::normalized_source(view.value, 1, value) && !value.jitter.supplied,
      "High-res depth borrowed a previous frame's render domain");
  }

  void test_source_without_projection() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "projection-independent source hook install failed");
    auto view = viewport(9); auto constants = modern_camera(); opaque_token token;
    int native{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    depth_resource_initialized(address, 91, 10, true);
    const base_structure *inputs[]{&view.base};
    sunshine_scene_depth::frame normalized;
    evaluation_snapshot observed;
    const auto evaluate = [&] { call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); };

    mint(token, nullptr);
    constants.common.jitter_offset[0] = .25f; constants.common.jitter_offset[1] = -.375f;
    constants.common.clip_to_camera_view.m[0][0] *= 2; // Stale inverse must remain rejected.
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::latest_snapshot(view.value, observed) && observed.status == evidence_status::invalid_camera,
      "source normalization upgraded invalid camera evidence");
    require(testing::normalized_source(view.value, 0, normalized) && normalized.resource.native == address &&
        !normalized.projection.supplied && normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == 0.0 &&
        normalized.projection.direction_supplied && normalized.projection.reversed && normalized.jitter.supplied &&
        normalized.jitter.x == .25f && normalized.jitter.y == -.375f,
      "valid depth source was rejected with its invalid projection, or lost the independent decoded direction");

    // A new observed evaluation token has no camera. Never inherit the prior
    // token's otherwise well-formed depth direction or unusable projection.
    mint(token, nullptr); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.resource.native == address &&
        !normalized.projection.supplied && !normalized.projection.direction_supplied,
      "missing constants rejected an authoritative source or inherited unrelated camera metadata");
    depth_source_snapshot old_query;
    require(query_depth_source(old_query) == evidence_status::missing_constants,
      "direct source admission changed the existing stricter nomination query");

    result_v2 = -1; evaluate(); result_v2 = 0;
    require(!testing::normalized_source(view.value, 0, normalized), "failed evaluation exposed a normalized source");
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 0, normalized), "source did not recover after failed evaluation");
    testing::lose_observation();
    require(!testing::normalized_source(view.value, 0, normalized), "observation loss retained camera-free source authority");
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 0, normalized), "fresh camera-free source did not recover from observation loss");

    tag.lifecycle = 0; call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(!testing::normalized_source(view.value, 0, normalized), "OnlyValidNow source survived until a later evaluation");
    tag.lifecycle = 1; tag.area.width = resource.width + 1;
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(!testing::normalized_source(view.value, 0, normalized), "invalid depth extent acquired source authority");
    tag.area.width = resource.width;
    constants = modern_camera(); constants.reset = 1;
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied,
      "valid reset-frame depth lost its current projection/source");

    view = viewport(10); mint(token, nullptr); evaluate();
    require(!testing::normalized_source(view.value, 0, normalized), "missing depth tag acquired camera-free source authority");
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(testing::normalized_source(view.value, 0, normalized), "valid source without constants was not admitted");
    mint(token, nullptr);
    require(!testing::normalized_source(view.value, 0, normalized), "token reuse retained old camera-free source authority");
  }

  void test_until_evaluate_lifetime() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "UntilEvaluate hook install failed");
    auto view = viewport(8); auto constants = modern_camera(); opaque_token token;
    int native{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    tag.lifecycle = 2;
    depth_resource_initialized(reinterpret_cast<std::uintptr_t>(&native), 80, 10, true);
    const base_structure *inputs[]{&view.base};
    const base_structure *local_inputs[]{&view.base, &tag.base};
    evaluation_snapshot observed;
    depth_source_snapshot nomination;
    const auto evaluated = [&] {
      require(testing::latest_snapshot(view.value, observed) && observed.successful_evaluation &&
          observed.status == evidence_status::source_associated_evaluation && observed.tags[0].supported &&
          observed.tags[0].value.lifecycle == 2, "valid UntilEvaluate tag rejected at synchronous evaluation boundary");
      sunshine_scene_depth::frame normalized;
      require(testing::normalized_source(view.value, 0, normalized) &&
          normalized.valid_until == sunshine_scene_depth::lifetime::until_evaluation &&
          normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == .5,
        "UntilEvaluate adapter changed projection or lifetime while normalizing");
      require(query_depth_source(nomination) != evidence_status::source_associated_evaluation,
        "UntilEvaluate lifetime was upgraded to post-return nomination");
    };
    mint(token, nullptr);
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(testing::latest_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "global UntilEvaluate tag was reused after its evaluation returned");
    // A fresh exact-frame tag authorizes this evaluation only, independently of
    // the consumed global tag and without inventing native-present correlation.
    call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(testing::latest_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "frame UntilEvaluate tag was reused after return");
    call_v2_evaluate(0, token.ref(), local_inputs, 2, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(testing::latest_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "local UntilEvaluate tag leaked to a later call");
    tag.area.width = 1921;
    call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(testing::latest_snapshot(view.value, observed) && observed.status != evidence_status::source_associated_evaluation,
      "UntilEvaluate admission bypassed malformed extent rejection");
    tag.area.width = 1920;
    call_v2_constants(constants, token.ref(), view); call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); evaluated();
  }

  void test_scene_feedback(bool modern) {
    fixture cleanup;
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "scene feedback hook install failed");
    int native{};
    abi_v1::resource old_resource{}; old_resource.native = &native;
    auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    opaque_token tokens[2];
    const auto evaluate = [&](std::uint32_t frame, std::uint32_t view_id, bool reset = false) {
      if (modern) {
        auto view = viewport(view_id); auto constants = modern_camera(); constants.reset = reset;
        auto &token = tokens[view_id - 1]; mint(token, &frame);
        call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr);
        const base_structure *inputs[] = {&view.base};
        call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
      } else {
        auto constants = old_camera(); constants.reset = reset;
        call_v1_constants(constants, frame, view_id); call_v1_tag(&old_resource, 0, view_id, nullptr);
        call_v1_evaluate(nullptr, 0, frame, view_id);
      }
    };
    evaluate(1, 1);
    sunshine_scene_depth::frame initial;
    require(testing::normalized_source(1, 0, initial) && !initial.feedback.reset &&
        initial.feedback.revision != 0 && initial.feedback.revision == initial.observation_revision + 1,
      "source revision missing from normalized depth");
    evaluate(2, 1);
    sunshine_scene_depth::frame ordinary;
    require(testing::normalized_source(1, 0, ordinary) && !ordinary.feedback.reset &&
        ordinary.feedback.revision == initial.feedback.revision && ordinary.source_frame_numeric == 2 &&
        ordinary.source_id == initial.source_id && ordinary.viewport == initial.viewport,
      "ordinary input changed reset revision or source identity");
    evaluate(1, 2);
    sunshine_scene_depth::frame other;
    require(testing::normalized_source(2, 0, other) && other.viewport == 2 && !other.feedback.reset &&
        other.feedback.revision == initial.feedback.revision,
      "viewport feedback acquired an unrelated reset");
    evaluate(3, 1, true);
    evaluation_snapshot reset;
    require(testing::latest_snapshot(1, reset) && reset.feedback.reset &&
        reset.feedback.revision > ordinary.feedback.revision && testing::normalized_source(1, 0, other) &&
        other.feedback.reset && other.feedback.revision == reset.feedback.revision &&
        other.observation_revision == depth_observation_revision() && other.projection.supplied &&
        other.projection.depth_offset == initial.projection.depth_offset &&
        other.projection.depth_scale == initial.projection.depth_scale && other.source_frame_numeric == 3 &&
        other.source_id == initial.source_id,
      "reset failed to revoke history while admitting its fresh depth and projection");
    require(!testing::normalized_source(2, 0, other), "reset promoted a pre-reset viewport snapshot to the new revision");
    evaluate(4, 1);
    sunshine_scene_depth::frame recovered;
    require(testing::normalized_source(1, 0, recovered) && !recovered.feedback.reset &&
        recovered.feedback.revision == reset.feedback.revision && recovered.feedback.revision == depth_observation_revision() + 1 &&
        recovered.source_id == initial.source_id && recovered.viewport == initial.viewport,
      "fresh post-reset input lost revision or logical source identity");
    require(!initial.feedback.reset && initial.feedback.revision == ordinary.feedback.revision &&
        ordinary.feedback.revision < recovered.feedback.revision,
      "later reset mutated a frozen earlier frame");
  }

  void test_scene_feedback_inflight() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "in-flight feedback install failed");
    int native{}; abi_v1::resource resource{}; resource.native = &native;
    auto constants = old_camera();
    call_v1_constants(constants, 1, 1); call_v1_tag(&resource, 0, 1, nullptr);
    const auto original_revision = depth_observation_revision() + 1;
    block_original = true;
    std::thread pending([] { call_v1_evaluate(nullptr, 0, 1, 1); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    block_original = false;
    constants.reset = 1;
    if (entered) call_v1_constants(constants, 2, 1);
    release_original = true; pending.join();
    require(entered, "in-flight feedback evaluation did not reach original");
    evaluation_snapshot first;
    sunshine_scene_depth::frame normalized;
    require(testing::latest_snapshot(1, first) && first.frame.numeric == 1 && !first.feedback.reset &&
        first.feedback.revision == original_revision && first.feedback.revision < depth_observation_revision() + 1 &&
        first.status == evidence_status::observation_lost && !testing::normalized_source(1, 0, normalized),
      "pre-reset completion borrowed newer reset metadata or remained usable");
    call_v1_evaluate(nullptr, 0, 2, 1);
    evaluation_snapshot reset;
    require(testing::latest_snapshot(1, reset) && reset.frame.numeric == 2 && reset.feedback.reset &&
        reset.feedback.revision == depth_observation_revision() + 1 && !testing::normalized_source(1, 0, normalized),
      "reset evaluation accepted depth tags from before its temporal boundary");
    call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 2, 1);
    require(testing::normalized_source(1, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied &&
        normalized.source_frame_numeric == 2 && normalized.feedback.revision == reset.feedback.revision,
      "freshly retagged reset frame did not recover without waiting for a non-reset frame");
    constants.reset = 0;
    call_v1_constants(constants, 3, 1); call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 3, 1);
    require(testing::normalized_source(1, 0, normalized) && normalized.source_frame_numeric == 3 &&
        !normalized.feedback.reset && normalized.feedback.revision == reset.feedback.revision &&
        first.feedback.revision == original_revision,
      "post-reset source failed to recover or rewrote earlier feedback");
  }

  void test_reset_constants_observation_loss(bool modern) {
    fixture cleanup;
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "reset constants race install failed");
    int native{}; abi_v1::resource old_resource{}; old_resource.native = &native;
    auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    auto old = old_camera(); old.reset = 1;
    auto current = modern_camera(); current.reset = 1;
    auto view = viewport(1); opaque_token token; const std::uint32_t frame = 1;
    if (modern) mint(token, &frame);
    const auto constants = [&] {
      if (modern) call_v2_constants(current, token.ref(), view);
      else call_v1_constants(old, frame, view.value);
    };
    const auto evaluate = [&] {
      if (modern) {
        call_v2_tag(view, &tag, 1, nullptr);
        const base_structure *inputs[] = {&view.base};
        call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
      } else {
        call_v1_tag(&old_resource, 0, view.value, nullptr);
        call_v1_evaluate(nullptr, 0, frame, view.value);
      }
    };
    block_original = true;
    std::thread pending(constants);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    testing::lose_observation();
    release_original = true; pending.join(); block_original = false;
    require(entered, "reset constants race did not reach the original call");
    evaluate();
    sunshine_scene_depth::frame normalized;
    require(!testing::normalized_source(view.value, 0, normalized),
      "reset revision repaired unrelated observation loss during the constants call");
    constants(); evaluate();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied,
      "fresh reset constants did not recover after the interrupted reset call");
  }

  void test_depth_observation_loss_diagnostics(bool modern) {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(!enabled() && source_enabled(), "Loss journal enabled the optional command/content probe");
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "Loss journal hook install failed");
    const auto view = viewport(59);
    int native{};
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    depth_resource_initialized(address, 41, 10, true);
    abi_v1::resource old_resource{}; old_resource.native = &native;
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    auto old = old_camera();
    auto current = modern_camera();
    opaque_token token;
    std::uint32_t frame{};
    const base_structure *inputs[]{&view.base};
    const auto constants = [&] {
      return modern ? call_v2_constants(current, token.ref(), view) == result_v2 :
        call_v1_constants(old, frame, view.value) == result_v1;
    };
    const auto evaluate = [&] {
      return modern ? call_v2_evaluate(0, token.ref(), inputs, 1, nullptr) == result_v2 :
        call_v1_evaluate(nullptr, 0, frame, view.value) == result_v1;
    };
    const auto next_frame = [&] {
      ++frame;
      if (modern) mint(token, &frame);
    };
    const auto recover = [&] {
      old = old_camera(); current = modern_camera();
      next_frame();
      preserves_last_error(constants, constants_call);
      if (modern) call_v2_tag(view, &tag, 1, nullptr);
      else call_v1_tag(&old_resource, 0, view.value, nullptr);
      preserves_last_error(evaluate, evaluate_call);
      sunshine_scene_depth::frame normalized;
      require(testing::normalized_source(view.value, 0, normalized) &&
        normalized.resource.native == address && normalized.observation_revision == depth_observation_revision() &&
        normalized.projection.supplied && normalized.projection.depth_offset == 0. &&
        normalized.projection.depth_scale == .5 && normalized.projection.reversed && !normalized.feedback.reset,
        "Loss diagnostics changed fresh depth/projection recovery");
      return normalized;
    };
    const auto event_for = [&](std::uint64_t revision, loss_diagnostics::reason cause, DWORD thread) {
      loss_diagnostics::event event;
      require(query_depth_observation_loss(revision, event) && event.revision == revision && event.cause == cause &&
        event.tick && event.tick <= GetTickCount64() && event.thread_id == thread && event.line && event.site && *event.site,
        "Loss journal did not describe the exact real-hook invalidation");
      return event;
    };
    const auto still_recorded = [&](const loss_diagnostics::event &saved) {
      const auto actual = event_for(saved.revision, saved.cause, saved.thread_id);
      require(actual.tick == saved.tick && actual.line == saved.line && std::strcmp(actual.site, saved.site) == 0 &&
        actual.details.sequence == saved.details.sequence && actual.details.viewport == saved.details.viewport &&
        actual.details.feature == saved.details.feature && actual.details.sdk_result == saved.details.sdk_result &&
        actual.details.has_sdk_result == saved.details.has_sdk_result && actual.details.reset == saved.details.reset,
        "Recovery or a later loss overwrote an earlier exact-revision cause");
    };
    const auto no_old_source = [&] {
      sunshine_scene_depth::frame normalized;
      require(!testing::normalized_source(view.value, 0, normalized),
        "Diagnostic recording repaired the source invalidation it only describes");
    };

    const auto first = recover();
    old.reset = current.reset = 1;
    next_frame();
    const auto before_reset = depth_observation_revision();
    preserves_last_error(constants, constants_call);
    require(depth_observation_revision() == before_reset + 1,
      "Camera reset diagnostic changed the existing single revision increment");
    const auto reset = event_for(before_reset + 1, loss_diagnostics::reason::camera_reset, GetCurrentThreadId());
    require(reset.details.viewport == view.value && reset.details.sequence && reset.details.reset == 1,
      "Camera reset cause omitted its viewport/call/reset context");
    no_old_source();
    recover();
    still_recorded(reset);
    require(first.observation_revision == before_reset && !first.feedback.reset &&
      first.projection.depth_scale == .5, "Later diagnostic activity mutated the earlier normalized value");

    next_frame();
    old.common.clip_to_camera_view.m[0][0] = current.common.clip_to_camera_view.m[0][0] = 2.f;
    const auto before_invalid = depth_observation_revision();
    preserves_last_error(constants, constants_call);
    require(depth_observation_revision() == before_invalid + 1,
      "Invalid camera diagnostic changed the existing single revision increment");
    const auto invalid = event_for(before_invalid + 1, loss_diagnostics::reason::invalid_camera, GetCurrentThreadId());
    require(invalid.details.viewport == view.value && invalid.details.sequence && invalid.details.reset == 0,
      "Invalid projection was mislabeled as a game-requested camera reset");
    no_old_source();
    recover();

    const auto before_failure = depth_observation_revision();
    result_v1 = false; result_v2 = -73;
    preserves_last_error(evaluate, evaluate_call);
    result_v1 = true; result_v2 = 0;
    require(depth_observation_revision() == before_failure + 1,
      "Failed SDK evaluation diagnostic changed the existing revision increment");
    const auto failed = event_for(before_failure + 1, loss_diagnostics::reason::sdk_failure, GetCurrentThreadId());
    require(failed.details.viewport == view.value && failed.details.feature == 0 && failed.details.sequence &&
      failed.details.has_sdk_result && failed.details.sdk_result == (modern ? -73 : 0),
      "Failed SDK evaluation lost its original result or logical call identity");
    no_old_source();
    recover();

    const auto before_busy = depth_observation_revision();
    const auto counters_before_busy = testing::counts();
    std::atomic<bool> finished{};
    bool forwarded{};
    DWORD worker_thread{};
    struct records_guard {
      bool locked{true};
      records_guard() { testing::lock_records(); }
      ~records_guard() { release(); }
      void release() { if (locked) { testing::unlock_records(); locked = false; } }
    } guard;
    std::thread worker([&] {
      worker_thread = GetCurrentThreadId();
      SetLastError(incoming_error);
      const bool original_result = constants();
      const auto original_error = GetLastError();
      forwarded = original_result && constants_call.incoming_error == incoming_error && original_error == outgoing_error;
      finished.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool nonblocking = finished.load(std::memory_order_acquire);
    guard.release();
    worker.join();
    // V2 already observes the token and stores the camera under separate
    // try-locks; instrumentation must retain both original loss increments.
    // Token lookup has its own lock; only the camera-table update is lost.
    const auto expected_losses = 1u;
    require(nonblocking && forwarded && depth_observation_revision() == before_busy + expected_losses &&
      testing::counts().dropped == counters_before_busy.dropped + expected_losses,
      "Contended diagnostic changed callback latency, API forwarding or existing loss accounting");
    for (unsigned i = 1; i <= expected_losses; ++i)
      event_for(before_busy + i, loss_diagnostics::reason::records_busy, worker_thread);
    const auto busy = event_for(before_busy + expected_losses, loss_diagnostics::reason::records_busy, worker_thread);
    require(busy.details.viewport == view.value && busy.details.sequence,
      "Contended camera storage omitted its exact callback context");
    no_old_source();
    recover();

    const auto recovered_revision = depth_observation_revision();
    for (const auto &saved : {reset, invalid, failed, busy}) still_recorded(saved);
    loss_diagnostics::event missing;
    require(!query_depth_observation_loss(0, missing) &&
      !query_depth_observation_loss(recovered_revision + 1, missing) &&
      !query_depth_observation_loss(UINT64_MAX, missing) && depth_observation_revision() == recovered_revision,
      "Missing diagnostic revision borrowed another event or changed source authority");
    require(!enabled() && source_enabled(), "Loss diagnostics enabled the optional full probe during recovery");
  }

  void test_evaluation_v1() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 evidence install failed");
    int native{}, commands{};
    abi_v1::resource resource{};
    resource.native = &native;
    auto selected = selection(&native);
    auto constants = old_camera();
    call_v1_constants(constants, 10, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    expect_evidence(selected, evidence_status::missing_evaluation, "constants/tags alone fabricated evaluation evidence");
    auto next = constants;
    set_camera_near(next.common, 1.f);
    call_v1_constants(next, 11, 4);
    call_v1_evaluate(&commands, 0, 10, 4);
    evaluation_snapshot captured;
    require(query_evaluation(selected, captured) == evidence_status::source_associated_evaluation,
      "v1 exact constants frame not retained across frames in flight");
    require(captured.frame_correlated && captured.successful_evaluation && captured.frame.has_numeric &&
        captured.frame.numeric == 10 && captured.frame.kind == frame_identity_kind::v1_numeric &&
        captured.camera.near_plane == .5f &&
        captured.projection.valid() && captured.projection.depth_scale == .5 &&
        captured.command_buffer == reinterpret_cast<std::uintptr_t>(&commands) &&
        captured.tags[0].scope == tag_scope::active_global && captured.selection.frame_index == 999,
      "v1 snapshot did not preserve camera/tag/evaluation identities");
    captured.camera.near_plane = 999;
    evaluation_snapshot copied;
    query_evaluation(selected, copied);
    require(copied.camera.near_plane == .5f, "query exposed mutable internal storage");
    auto wrong = selected;
    wrong.active_width--;
    expect_evidence(wrong, evidence_status::extent_mismatch, "whole-resource tag accepted a different selected crop");
    wrong = selected;
    wrong.resource++;
    expect_evidence(wrong, evidence_status::source_mismatch, "wrong native source accepted");
    wrong = selected;
    wrong.ready = false;
    expect_evidence(wrong, evidence_status::depth_not_ready, "old evidence substituted for current depth readiness");

    call_v1_evaluate(nullptr, 0, 12, 4);
    expect_evidence(selected, evidence_status::missing_constants, "latest camera substituted for missing exact frame");
    call_v1_evaluate(nullptr, 0, 11, 4);
    require(query_evaluation(selected, copied) == evidence_status::source_associated_evaluation &&
        copied.camera.near_plane == 1.f && copied.projection.depth_scale == 1.,
      "older unsuccessful correlation poisoned a later complete tuple");
    result_v1 = false;
    call_v1_evaluate(nullptr, 0, 11, 4);
    expect_evidence(selected, evidence_status::evaluation_failed, "failed evaluation left previous success available");
    result_v1 = true;
    call_v1_constants(constants, 13, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 13, 4);
    expect_evidence(selected, evidence_status::source_associated_evaluation, "fresh complete observations did not recover");
    testing::lose_observation();
    expect_evidence(selected, evidence_status::observation_lost, "dropped callback left stale evidence accepted");
    call_v1_evaluate(nullptr, 0, 13, 4);
    expect_evidence(selected, evidence_status::observation_lost, "evaluation reused inputs from before observation loss");
    call_v1_constants(constants, 14, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 14, 4);
    sunshine_scene_depth::frame before_reset;
    require(testing::normalized_source(4, 0, before_reset) &&
        before_reset.observation_revision == depth_observation_revision(),
      "v1 normalized depth omitted its source observation revision");
    constants.reset = 1;
    call_v1_constants(constants, 15, 4);
    require(depth_observation_revision() != before_reset.observation_revision,
      "v1 camera reset failed to revoke retained depth before the next tag/evaluation");
    expect_evidence(selected, evidence_status::observation_lost, "camera reset did not revoke old evidence immediately");
    constants.reset = 0;
    call_v1_constants(constants, 16, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 16, 4);
    constants.common.clip_to_camera_view.m[0][0] = 2;
    call_v1_constants(constants, 17, 4);
    expect_evidence(selected, evidence_status::observation_lost, "invalid projection did not revoke old evidence");
    shutdown();
    expect_evidence(selected, evidence_status::inactive, "shutdown left evidence available");
  }
  void test_evaluation_immutable_and_ordered() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 ordered evidence install failed");
    int native{};
    abi_v1::resource resource{};
    resource.native = &native;
    const auto selected = selection(&native);
    auto first = old_camera(), second = old_camera();
    set_camera_near(second.common, 1.f);
    call_v1_constants(first, 1, 8);
    call_v1_constants(second, 2, 8);
    call_v1_tag(&resource, 0, 8, nullptr);
    block_original = true;
    std::thread older([] { call_v1_evaluate(nullptr, 0, 1, 8); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    block_original = false;
    if (entered) call_v1_evaluate(nullptr, 0, 2, 8);
    release_original = true;
    older.join();
    require(entered, "blocked evaluation never entered original");
    evaluation_snapshot snapshot;
    require(query_evaluation(selected, snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.frame.numeric == 2 && snapshot.camera.near_plane == 1.f && snapshot.projection.depth_scale == 1.,
      "out-of-order evaluation completion replaced newer immutable tuple");
    set_camera_near(second.common, 2.f);
    call_v1_constants(second, 2, 8);
    require(query_evaluation(selected, snapshot) == evidence_status::stale && snapshot.camera.near_plane == 1.f &&
        snapshot.projection.depth_scale == 1.,
      "later same-frame constants mutated or retained applicability of previously published evaluation");
    call_v1_evaluate(nullptr, 0, 2, 8);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    require(query_evaluation(selected, snapshot, 1) == evidence_status::stale, "explicit evidence age bound ignored");
    // Bounded frame storage evicts old frames instead of guessing from latest.
    for (unsigned frame = 3; frame != 38; ++frame) call_v1_constants(first, frame, 8);
    call_v1_evaluate(nullptr, 0, 1, 8);
    expect_evidence(selected, evidence_status::missing_constants, "evicted frame reused a different camera");
  }
  void test_evaluation_v2_tokens() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 tracked evidence install failed");
    auto view = viewport(3);
    auto constants = modern_camera();
    opaque_token first, second;
    int native_a{}, native_b{}, native_local{};
    auto resource_a = modern_resource(&native_a), resource_b = modern_resource(&native_b), resource_local = modern_resource(&native_local);
    auto tag_a = depth_tag_for(resource_a), tag_b = depth_tag_for(resource_b), tag_local = depth_tag_for(resource_local);
    const base_structure *inputs[] {&view.base};
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_evidence(selection(&native_a), evidence_status::untracked_frame, "opaque address alone became a frame identity");
    const std::uint32_t number = 101;
    mint(first, &number);
    mint(second, nullptr);
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    set_camera_near(constants.common, 1.f);
    call_v2_constants(constants, second.ref(), view);
    call_v2_framed_tag(second.ref(), view, &tag_b, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    evaluation_snapshot snapshot;
    require(query_evaluation(selection(&native_a), snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.frame.kind == frame_identity_kind::v2_observed_token && snapshot.frame.generation &&
        snapshot.frame.has_numeric && snapshot.frame.numeric == 101 && snapshot.camera.near_plane == .5f &&
        snapshot.tags[0].scope == tag_scope::explicit_frame,
      "v2 frame-specific constants/tags mixed frames in flight");
    const auto old_generation = snapshot.frame.generation;
    call_v2_evaluate(0, second.ref(), inputs, 1, nullptr);
    require(query_evaluation(selection(&native_b), snapshot) == evidence_status::source_associated_evaluation &&
        !snapshot.frame.has_numeric && snapshot.frame.generation != old_generation && snapshot.camera.near_plane == 1.f &&
        snapshot.projection.depth_scale == 1.,
      "internal token counter was invented as a game numeric frame");
    const base_structure *local_inputs[] {&view.base, &tag_local.base};
    call_v2_evaluate(0, first.ref(), local_inputs, 2, nullptr);
    require(query_evaluation(selection(&native_local), snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].scope == tag_scope::evaluation_local, "evaluation-local tag did not override frame tag");
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_evidence(selection(&native_a), evidence_status::source_associated_evaluation, "local override leaked into next evaluation");
    mint(first, &number); // Same address AND same optional numeric index, new generation.
    expect_evidence(selection(&native_a), evidence_status::untracked_frame, "token reuse left old frame evidence accepted");
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_evidence(selection(&native_a), evidence_status::missing_constants, "recycled token reused old constants/frame tags");
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    require(query_evaluation(selection(&native_a), snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.frame.generation != old_generation, "fresh recycled-token tuple failed recovery");
    resource_a.width = 3840;
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    expect_evidence(selection(&native_a), evidence_status::source_mismatch, "same-pointer frame tag dimension change left old tuple applicable");
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_evidence(selection(&native_a), evidence_status::extent_mismatch, "tagged resource size mismatch accepted");
    resource_a.width = 1920;
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    tag_a.lifecycle = 0;
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    require(query_evaluation(selection(&native_a), snapshot) != evidence_status::source_associated_evaluation,
      "same-pointer unsupported lifecycle left old frame tag applicable");
    tag_a.lifecycle = 1;
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    abi_v2::frame_token *output{};
    call_v2_new_token(output, reinterpret_cast<const std::uint32_t *>(1));
    expect_evidence(selection(&native_a), evidence_status::observation_lost, "unreadable token index left accepted evidence");
  }
  void test_evaluation_ambiguity_and_invalid_tags() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 ambiguity install failed");
    int native{};
    abi_v1::resource resource{};
    resource.native = &native;
    auto constants = old_camera();
    for (unsigned view : {1, 2}) {
      call_v1_constants(constants, 1, view);
      call_v1_tag(&resource, 0, view, nullptr);
      call_v1_evaluate(nullptr, 0, 1, view);
    }
    expect_evidence(selection(&native), evidence_status::ambiguous_viewport, "same native resource in two viewports guessed a camera");
    resource.ext = reinterpret_cast<void *>(1);
    call_v1_tag(&resource, 0, 1, nullptr);
    evaluation_snapshot value;
    require(query_evaluation(selection(&native), value) != evidence_status::source_associated_evaluation,
      "new unsupported tag retained accepted evidence");
  }

  void test_evaluation_global_contract() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 global contract install failed");
    opaque_token token;
    mint(token, nullptr);
    auto constants = modern_camera();
    auto view = viewport(21);
    int native{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    const auto selected = selection(&native);
    const base_structure *inputs[] {&view.base};
    call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    expect_evidence(selected, evidence_status::source_associated_evaluation, "global tag fixture did not start valid");
    resource.height = 2160;
    call_v2_tag(view, &tag, 1, nullptr);
    expect_evidence(selected, evidence_status::source_mismatch, "same-pointer global dimensions were not rechecked");
    resource.height = 1080;
    call_v2_tag(view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    tag.lifecycle = 0;
    call_v2_tag(view, &tag, 1, nullptr);
    evaluation_snapshot snapshot;
    require(query_evaluation(selected, snapshot) != evidence_status::source_associated_evaluation,
      "same-pointer global lifetime was not rechecked");
  }

  void test_evaluation_input_age() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 input-age install failed");
    int native{};
    auto constants = old_camera();
    abi_v1::resource resource{};
    resource.native = &native;
    block_original = true;
    std::thread delayed([&] { call_v1_constants(constants, 1, 1); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    block_original = false;
    release_original = true;
    delayed.join();
    require(entered, "delayed constants never entered original");
    call_v1_tag(&resource, 0, 1, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 1);
    evaluation_snapshot snapshot;
    require(query_evaluation(selection(&native), snapshot, 10) == evidence_status::stale &&
        snapshot.tick >= snapshot.camera_tick + 10,
      "blocked constants call completion made old input look newly observed");
  }

  void test_evaluation_command_markers() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "command-marker hook install failed");
    int native{}, copy_native{}, evaluation_native{};
    const auto copy_command = reinterpret_cast<std::uintptr_t>(&copy_native);
    const auto evaluation_command = reinterpret_cast<std::uintptr_t>(&evaluation_native);
    constexpr std::uint64_t queue = 100, device = 200;
    queue_initialized(queue, device);
    command_initialized(copy_command, device);
    command_initialized(evaluation_command, device);
    auto selected = selection(&native);
    selected.command_queue = queue;
    selected.capture_marker = capture_command_marker(copy_command);
    command_closed(copy_command);
    command_executed(queue, copy_command);
    auto constants = old_camera();
    abi_v1::resource resource{};
    resource.native = &native;
    call_v1_constants(constants, 1, 1);
    call_v1_tag(&resource, 0, 1, nullptr);
    call_v1_evaluate(&evaluation_native, 0, 1, 1);
    evaluation_snapshot snapshot;
    require(query_evaluation(selected, snapshot) == evidence_status::source_associated_evaluation && snapshot.recording_stable &&
        snapshot.evaluation_recording.command == evaluation_command && snapshot.command_association.state == commands::status::open_recording,
      "evaluation marker fabricated submission before close/execute observations");
    command_closed(evaluation_command);
    command_executed(queue, evaluation_command);
    require(query_evaluation(selected, snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.command_association.state == commands::status::same_queue_callbacks_order_unverified &&
        !snapshot.command_association.associated() && snapshot.command_association.ordering == commands::order::unknown &&
        !snapshot.command_association.content_registered && !snapshot.command_association.final_color_registered,
      "cross-recording callback order was promoted into execution association");
    command_executed(queue, copy_command);
    require(query_evaluation(selected, snapshot) == evidence_status::source_associated_evaluation &&
        snapshot.command_association.state == commands::status::repeated_submission,
      "repeated submission retained command-associated evidence");

    command_reset(evaluation_command);
    selected.capture_marker = capture_command_marker(evaluation_command);
    call_v1_constants(constants, 3, 1);
    call_v1_tag(&resource, 0, 1, nullptr);
    call_v1_evaluate(&evaluation_native, 0, 3, 1);
    command_closed(evaluation_command);
    command_executed(queue, evaluation_command);
    require(query_evaluation(selected, snapshot) == evidence_status::command_associated_evaluation &&
        snapshot.command_association.associated() && snapshot.command_association.same_recording &&
        snapshot.command_association.ordering == commands::order::copy_before_evaluation,
      "same-recording event order did not retain narrow command association");

    command_reset(copy_command);
    command_reset(evaluation_command);
    selected.capture_marker = capture_command_marker(copy_command);
    command_closed(copy_command);
    command_executed(queue, copy_command);
    call_v1_constants(constants, 2, 1);
    call_v1_tag(&resource, 0, 1, nullptr);
    block_original = true;
    std::thread delayed([&] { call_v1_evaluate(&evaluation_native, 0, 2, 1); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    command_reset(evaluation_command);
    block_original = false;
    release_original = true;
    delayed.join();
    require(entered, "delayed evaluation never entered original");
    require(query_evaluation(selected, snapshot, 5000) == evidence_status::source_associated_evaluation &&
        snapshot.evaluation_recording && !snapshot.recording_stable &&
        snapshot.command_association.state == commands::status::recording_changed,
      "reset inside original evaluation retained stable recording association");
  }

  void test_evaluation_content() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "content hook install failed");
    int native{}, command_native{}, backup_native{};
    const auto command = reinterpret_cast<std::uintptr_t>(&command_native);
    constexpr std::uint64_t queue = 501, device = 502;
    auto selected = selection(&native);
    selected.command_queue = queue;
    selected.backup_resource = reinterpret_cast<std::uintptr_t>(&backup_native);
    selected.backup_id = 8;
    const content::resource_key source{selected.resource, selected.source_id}, backup{selected.backup_resource, selected.backup_id};
    queue_initialized(queue, device, 1);
    command_initialized(command, device, 2);
    depth_resource_initialized(source.native, source.lifetime, device, true);
    depth_resource_initialized(backup.native, backup.lifetime, device, true);
    // Global unsupported operations can precede command destruction without a
    // Reset. Their retired bans must not saturate future content observations.
    for (unsigned i = 0; i != 600; ++i) {
      const std::uint64_t retired = 10000 + i;
      command_initialized(retired, device, retired);
      depth_content_invalidate(retired);
      command_destroyed(retired);
    }
    auto constants = old_camera();
    abi_v1::resource resource{};
    resource.native = &native;
    std::uint32_t frame{};
    auto setup = [&] {
      command_reset(command, device, 2);
      call_v1_constants(constants, ++frame, 1);
      call_v1_tag(&resource, 0, 1, nullptr);
    };
    auto copy = [&] {
      selected.depth_copy = record_depth_copy(command, source, backup);
      selected.capture_marker = selected.depth_copy.recording;
      require(selected.depth_copy.valid(), "copy observation unexpectedly invalid");
    };
    auto evaluate = [&] { call_v1_evaluate(&command_native, 0, frame, 1); };
    auto submit = [&] { command_closed(command); command_executed(queue, command); };
    evaluation_snapshot snapshot;
    setup(); copy(); evaluate(); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::tracked_content_evaluation &&
        snapshot.content_association.matched() && !snapshot.content_association.coverage_complete &&
        !snapshot.content_association.final_color_registered && !snapshot.command_association.content_registered,
      "real hook did not associate immutable observed content with incomplete coverage");
    command_initialized(50000, device, 50000);
    depth_content_invalidate(50000);
    require(query_evaluation(selected, snapshot) == evidence_status::command_associated_evaluation &&
        !snapshot.content_association.matched(), "another recording's unknown-resource mutation retained content association");
    command_destroyed(50000);
    setup(); copy(); evaluate(); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::tracked_content_evaluation,
      "fresh snapshots after cross-recording invalidation failed to recover");
    auto changed = selected;
    changed.backup_id++;
    require(query_evaluation(changed, snapshot) == evidence_status::command_associated_evaluation &&
        !snapshot.content_association.matched(), "selected backup generation mismatch retained content match");
    changed = selected; changed.capture_marker.event++;
    require(query_evaluation(changed, snapshot) != evidence_status::tracked_content_evaluation,
      "selected command marker substituted for actual copy event");

    setup(); copy(); depth_content_write(command, source.native, source.lifetime); evaluate(); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::command_associated_evaluation &&
        snapshot.content_association.state == content::status::different_source_content,
      "real hook accepted copy/clear/evaluate mismatch");
    setup(); evaluate(); depth_content_write(command, source.native, source.lifetime); copy(); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::command_associated_evaluation &&
        snapshot.content_association.state == content::status::different_source_content,
      "real hook accepted evaluate/write/copy mismatch");
    setup(); evaluate(); copy(); depth_content_write(command, source.native, source.lifetime); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::tracked_content_evaluation,
      "source clear after evaluation/copy incorrectly rejected preserved backup");
    command_reset(command, device, 2);
    depth_content_write(command, backup.native); // Native-only destination lookup covers game copies too.
    require(query_evaluation(selected, snapshot) != evidence_status::tracked_content_evaluation,
      "later backup overwrite retained old content association");

    setup(); copy();
    block_original = true;
    std::thread delayed([&] { evaluate(); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    depth_content_write(command, source.native, source.lifetime);
    block_original = false; release_original = true; delayed.join();
    require(entered, "blocked content evaluation never entered original");
    submit();
    require(query_evaluation(selected, snapshot, 5000) == evidence_status::command_associated_evaluation &&
        snapshot.tags[snapshot.matched_tag].content_use.state == content::status::different_source_content,
      "mutation inside original evaluation was published as stable content");

    setup(); copy(); evaluate(); submit();
    require(query_evaluation(selected, snapshot) == evidence_status::tracked_content_evaluation, "content baseline did not recover");
    depth_resource_destroyed(source.native, source.lifetime);
    depth_resource_initialized(source.native, source.lifetime, device, true);
    require(query_evaluation(selected, snapshot) != evidence_status::tracked_content_evaluation,
      "resource re-registration resurrected immutable evaluation/copy association");
  }

  void test_inactive_content_transition() {
    fixture cleanup;
    int native{}, backup_native{}, command_native{};
    const auto command = reinterpret_cast<std::uintptr_t>(&command_native);
    constexpr std::uint64_t device = 611, queue = 612;
    auto selected = selection(&native);
    selected.command_queue = queue;
    selected.backup_resource = reinterpret_cast<std::uintptr_t>(&backup_native);
    selected.backup_id = 8;
    const content::resource_key source{selected.resource, selected.source_id}, backup{selected.backup_resource, selected.backup_id};
    command_initialized(command, device, 1); queue_initialized(queue, device, 2);
    depth_resource_initialized(source.native, source.lifetime, device, true);
    depth_resource_initialized(backup.native, backup.lifetime, device, true);
    selected.depth_copy = record_depth_copy(command, source, backup);
    selected.capture_marker = selected.depth_copy.recording;
    require(selected.capture_marker && !selected.depth_copy.valid(), "inactive content observation lost command-only marker or fabricated content");
    depth_content_write(command, source.native, source.lifetime);
    depth_content_invalidate(command); // Ignored while no supported observer exists.
    require(testing::install(testing::abi::v1_1_1, v1_targets), "inactive transition hook install failed");
    auto constants = old_camera(); abi_v1::resource resource{}; resource.native = &native;
    call_v1_constants(constants, 1, 1); call_v1_tag(&resource, 0, 1, nullptr);
    call_v1_evaluate(&command_native, 0, 1, 1);
    command_closed(command); command_executed(queue, command);
    evaluation_snapshot snapshot;
    require(query_evaluation(selected, snapshot) == evidence_status::command_associated_evaluation &&
        !snapshot.content_association.matched(), "inactive copy was promoted after observation started");
    command_reset(command, device, 1);
    selected.depth_copy = record_depth_copy(command, source, backup);
    selected.capture_marker = selected.depth_copy.recording;
    require(selected.depth_copy.valid(), "observer transition did not preserve tracked resource lifetimes");
    call_v1_constants(constants, 2, 1); call_v1_evaluate(&command_native, 0, 2, 1);
    command_closed(command); command_executed(queue, command);
    require(query_evaluation(selected, snapshot) == evidence_status::tracked_content_evaluation,
      "fresh active copy/evaluation could not recover after inactive content interval");
  }

  void test_non_renderer_evaluations() {
    {
      fixture cleanup;
      require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 feature filter install failed");
      int native{}; abi_v1::resource resource{}; resource.native = &native;
      auto constants = old_camera();
      call_v1_constants(constants, 77, 0); call_v1_tag(&resource, 0, 0, nullptr); call_v1_evaluate(nullptr, 0, 77, 0);
      const auto selected = selection(&native);
      evaluation_snapshot baseline, next;
      require(query_evaluation(selected, baseline) == evidence_status::source_associated_evaluation, "missing DLSS filter baseline");
      for (const auto feature : {3u, 1u, 2u, 0xffffffffu}) for (const auto marker : {0u, 4u}) for (const auto success : {true, false}) {
        result_v1 = success;
        const auto before = evaluate_call.calls;
        SetLastError(incoming_error);
        require(call_v1_evaluate(nullptr, feature, 78, marker) == success && GetLastError() == outgoing_error,
          "non-renderer v1 call changed return/LastError");
        require(evaluate_call.calls == before + 1 && evaluate_call.first == nullptr && evaluate_call.a == feature &&
            evaluate_call.b == 78 && evaluate_call.c == marker && evaluate_call.incoming_error == incoming_error,
          "non-renderer v1 call changed count/arguments");
        require(query_evaluation(selected, next) == evidence_status::source_associated_evaluation &&
            next.sequence == baseline.sequence && next.frame.numeric == 77 && next.viewport == 0 && next.feature == 0,
          "Reflex/other feature overwrote DLSS viewport or failure revoked renderer evidence");
        require(!latest(4).found, "Reflex PresentStart marker fabricated viewport 4");
      }
    }
    {
      fixture cleanup;
      require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 feature filter install failed");
      opaque_token token; mint(token, nullptr);
      int native{}; auto view = viewport(0); auto constants = modern_camera();
      auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
      const base_structure *inputs[]{&view.base};
      call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
      const auto selected = selection(&native);
      evaluation_snapshot baseline, next;
      require(query_evaluation(selected, baseline) == evidence_status::source_associated_evaluation &&
          baseline.frame.kind == frame_identity_kind::v2_observed_token, "missing v2 DLSS filter baseline");
      for (const auto feature : {3u, 7u, 0xffffffffu}) for (const auto result : {0, -7}) {
        result_v2 = result;
        auto *opaque_inputs = reinterpret_cast<const base_structure **>(1);
        const auto before = evaluate_call.calls;
        SetLastError(incoming_error);
        require(call_v2_evaluate(feature, token.ref(), opaque_inputs, 0xffffffffu, nullptr) == result && GetLastError() == outgoing_error,
          "non-renderer v2 call changed result/LastError");
        require(evaluate_call.calls == before + 1 && evaluate_call.a == feature && evaluate_call.second == opaque_inputs &&
            evaluate_call.b == 0xffffffffu && evaluate_call.incoming_error == incoming_error,
          "non-renderer v2 opaque input forwarding changed");
        require(query_evaluation(selected, next) == evidence_status::source_associated_evaluation && next.sequence == baseline.sequence,
          "non-renderer v2 input was parsed or failure revoked DLSS evidence");
      }
    }
  }

  void test_color_tag_snapshots() {
    {
      fixture cleanup;
      require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 color fixture install failed");
      int depth{}, hudless{}, input{}, output{};
      abi_v1::resource d{}, h{}, i{}, o{}; d.native = &depth; h.native = &hudless; i.native = &input; o.native = &output;
      auto constants = old_camera();
      call_v1_constants(constants, 1, 0);
      call_v1_tag(&d, 0, 0, nullptr); call_v1_tag(&h, 2, 0, nullptr); call_v1_tag(&i, 3, 0, nullptr); call_v1_tag(&o, 4, 0, nullptr);
      call_v1_evaluate(nullptr, 0, 1, 0);
      evaluation_snapshot value;
      require(query_evaluation(selection(&depth), value) == evidence_status::source_associated_evaluation &&
          value.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&depth) &&
          value.colors[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&hudless) &&
          value.colors[1].value.native_resource == reinterpret_cast<std::uintptr_t>(&input) &&
          value.colors[2].value.native_resource == reinterpret_cast<std::uintptr_t>(&output) && !value.colors[3].present,
        "v1 colors were conflated with depth or each other");
      call_v1_tag(nullptr, 3, 0, nullptr);
      query_evaluation(selection(&depth), value);
      require(value.colors[1].identity_available, "later color untag mutated immutable old evaluation");
      call_v1_constants(constants, 2, 0); call_v1_evaluate(nullptr, 0, 2, 0);
      require(query_evaluation(selection(&depth), value) == evidence_status::source_associated_evaluation &&
          value.colors[1].present && !value.colors[1].identity_available && value.colors[2].identity_available,
        "v1 null color untag erased another color or poisoned depth");
    }
    {
      fixture cleanup;
      require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 color fixture install failed");
      opaque_token token; mint(token, nullptr);
      int depth{}, hudless{}, input{}, framed_input{}, output{}, local_output{};
      auto d = modern_resource(&depth), h = modern_resource(&hudless), i = modern_resource(&input), fi = modern_resource(&framed_input);
      auto o = modern_resource(&output), lo = modern_resource(&local_output);
      auto dt = depth_tag_for(d), ht = depth_tag_for(h), it = depth_tag_for(i), ot = depth_tag_for(o), bt = depth_tag_for(o);
      ht.type = 2; it.type = 3; ot.type = 4; bt.type = 53; bt.resource_ptr = nullptr;
      const abi_v2::resource_tag global[]{dt, ht, it, ot, bt};
      auto view = viewport(0); auto constants = modern_camera();
      call_v2_constants(constants, token.ref(), view); call_v2_tag(view, global, 5, nullptr);
      auto ft = depth_tag_for(fi); ft.type = 3;
      call_v2_framed_tag(token.ref(), view, &ft, 1, nullptr);
      auto lt = depth_tag_for(lo); lt.type = 4;
      const base_structure *inputs[]{&view.base, &lt.base};
      call_v2_evaluate(0, token.ref(), inputs, 2, nullptr);
      evaluation_snapshot value;
      require(query_evaluation(selection(&depth), value) == evidence_status::source_associated_evaluation &&
          value.colors[0].scope == tag_scope::active_global && value.colors[0].identity_available &&
          value.colors[1].scope == tag_scope::explicit_frame && value.colors[1].value.native_resource == reinterpret_cast<std::uintptr_t>(&framed_input) &&
          value.colors[2].scope == tag_scope::evaluation_local && value.colors[2].value.native_resource == reinterpret_cast<std::uintptr_t>(&local_output),
        "v2 independent color scope overrides lost");
      require(value.colors[3].present && value.colors[3].supported && !value.colors[3].identity_available &&
          value.colors[3].value.type == 53 && value.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&depth),
        "legal null v2 backbuffer fabricated identity or poisoned depth");
      value.colors[2].value.native_resource = 999;
      query_evaluation(selection(&depth), value);
      require(value.colors[2].value.native_resource == reinterpret_cast<std::uintptr_t>(&local_output), "color snapshot exposed mutable storage");
    }
  }
  void test_presentation_v1() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 presentation install failed");
    presentation_snapshot value;
    require(query_current_presentation(value) == presentation_status::missing_bracket, "presentation fabricated before markers");
    auto marker = [&](unsigned frame, unsigned type) {
      const auto before = evaluate_call.calls;
      SetLastError(incoming_error);
      require(call_v1_evaluate(nullptr, 3, frame, type) == result_v1 && GetLastError() == outgoing_error &&
          evaluate_call.calls == before + 1 && evaluate_call.a == 3 && evaluate_call.b == frame && evaluate_call.c == type &&
          evaluate_call.first == nullptr && evaluate_call.incoming_error == incoming_error, "v1 presentation marker forwarding changed");
    };
    marker(10, 0);
    require(query_current_presentation(value) == presentation_status::missing_bracket, "Reflex simulation marker opened presentation");
    marker(10, 4);
    require(query_current_presentation(value) == presentation_status::open_bracket && value.explicit_bracket && value.frame.numeric == 10,
      "v1 successful PresentStart did not open numeric bracket");
    const auto main_generation = value.thread_generation;
    std::atomic<bool> other_ok{};
    std::thread other([&] {
      presentation_snapshot local;
      const bool absent = query_current_presentation(local) == presentation_status::missing_bracket;
      call_v1_evaluate(nullptr, 3, 100, 4);
      other_ok = absent && query_current_presentation(local) == presentation_status::open_bracket && local.frame.numeric == 100 &&
        local.thread_generation != main_generation;
    });
    other.join();
    require(other_ok && query_current_presentation(value) == presentation_status::open_bracket && value.frame.numeric == 10,
      "presentation leaked across threads or another thread overwrote current bracket");
    std::thread replacement([&] { presentation_snapshot local; other_ok = query_current_presentation(local) == presentation_status::missing_bracket; });
    replacement.join(); require(other_ok, "terminated/reused thread inherited an open presentation");
    marker(11, 5);
    require(query_current_presentation(value) == presentation_status::out_of_order && !value.explicit_bracket, "mismatched PresentEnd retained open bracket");
    marker(11, 4); marker(11, 5);
    require(query_current_presentation(value) == presentation_status::ended_bracket && !value.explicit_bracket, "matching PresentEnd did not close bracket");
    marker(11, 4);
    require(query_current_presentation(value) == presentation_status::repeated_frame, "repeated presentation accepted");
    marker(9, 4); marker(10, 4);
    require(query_current_presentation(value) == presentation_status::out_of_order, "older frame rewound repetition guard");
    marker(12, 4);
    require(query_current_presentation(value) == presentation_status::open_bracket, "fresh presentation did not recover");
    // GetTickCount64 can have coarser resolution than Sleep(2). Wait for the
    // production age clock itself to advance, without assuming timer resolution.
    const auto age_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (GetTickCount64() <= value.start_tick && std::chrono::steady_clock::now() < age_deadline) Sleep(1);
    require(GetTickCount64() > value.start_tick, "presentation age clock did not advance within fixture deadline");
    require(query_current_presentation(value, 0) == presentation_status::stale, "stale presentation remained usable");
    marker(12, 5); result_v1 = false; marker(13, 4);
    require(query_current_presentation(value) != presentation_status::open_bracket && !value.explicit_bracket, "failed PresentStart opened bracket");
    result_v1 = true; marker(14, 4);
    require(query_current_presentation(value) == presentation_status::open_bracket, "presentation failed to recover after marker error");
    testing::lose_observation();
    require(query_current_presentation(value) == presentation_status::observation_lost && !value.explicit_bracket,
      "lifecycle observation loss retained presentation");
  }
  void test_presentation_v2() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "v2 PCL getter fixture install failed");
    presentation_snapshot value;
    require(query_current_presentation(value) == presentation_status::missing_marker_function, "v2 fabricated PCL function discovery");
    constexpr char name[] = "slPCLSetMarker";
    void *function{};
    auto retrieve = [&](unsigned feature, const char *requested_name) {
      const auto before = feature_function_call.calls;
      SetLastError(incoming_error);
      require(call_feature_function(feature, requested_name, function) == result_v2 && GetLastError() == outgoing_error &&
          function == returned_feature_function && feature_function_call.calls == before + 1 &&
          feature_function_call.a == feature && feature_function_call.first == requested_name && feature_function_call.second == &function &&
          feature_function_call.incoming_error == incoming_error, "PCL getter changed result/pointer/arguments/LastError");
    };
    retrieve(3, name); retrieve(4, "slPCLGetState");
    require(query_current_presentation(value) == presentation_status::missing_marker_function, "wrong PCL feature/name discovered marker");
    result_v2 = -7; retrieve(4, name);
    require(query_current_presentation(value) == presentation_status::missing_marker_function, "failed getter enabled markers");
    result_v2 = 0; retrieve(4, name);
    require(query_current_presentation(value) == presentation_status::missing_marker_function, "getter installed hook inside callback");
    require(testing::install_presentation_hooks(), "deferred PCL hook installation failed");
    auto call = reinterpret_cast<abi_v2::pcl_set_marker>(function);
    auto marker = [&](unsigned type, opaque_token &token) {
      const auto before = pcl_call.calls;
      SetLastError(incoming_error);
      require(call(type, token.ref()) == result_v2 && GetLastError() == outgoing_error && pcl_call.calls == before + 1 &&
          pcl_call.a == type && pcl_call.first == &token.ref() && pcl_call.incoming_error == incoming_error,
        "PCL marker changed return/arguments/LastError");
    };
    opaque_token token;
    marker(4, token);
    require(query_current_presentation(value) == presentation_status::untracked_frame, "opaque token address became present frame");
    mint(token, nullptr); marker(4, token);
    require(query_current_presentation(value) == presentation_status::open_bracket && value.explicit_bracket &&
        value.frame.kind == frame_identity_kind::v2_observed_token && !value.frame.has_numeric, "observed PCL token did not open bracket");
    const auto generation = value.frame.generation;
    retrieve(4, name);
    require(query_current_presentation(value) == presentation_status::open_bracket, "identical successful getter revoked current bracket");
    mint(token, nullptr);
    require(query_current_presentation(value) == presentation_status::untracked_frame, "reused token address retained older present frame");
    marker(5, token);
    require(query_current_presentation(value) == presentation_status::out_of_order, "different token generation closed wrong frame");
    inspect_presentation_in_original = true;
    marker(4, token);
    require(query_current_presentation(value) == presentation_status::open_bracket && value.frame.generation > generation,
      "new token generation did not recover presentation");
    require(presentation_inside_original.status == presentation_status::pending_marker && !presentation_inside_original.explicit_bracket,
      "PresentStart was usable before original success");
    marker(5, token);
    require(presentation_inside_original.status == presentation_status::pending_marker && !presentation_inside_original.explicit_bracket,
      "PresentEnd entry left bracket usable while original was pending");
    inspect_presentation_in_original = false;
    marker(4, token);
    require(query_current_presentation(value) == presentation_status::repeated_frame, "same token frame presented twice");
    mint(token, nullptr); marker(4, token);
    result_v2 = -7; retrieve(4, name);
    require(query_current_presentation(value) == presentation_status::missing_marker_function, "failed feature retrieval retained current PCL evidence");
    result_v2 = 0; retrieve(4, name);
    require(query_current_presentation(value) == presentation_status::observation_lost, "old bracket resurrected after getter recovered");
    marker(5, token); mint(token, nullptr); marker(4, token);
    result_v2 = -9; marker(5, token);
    require(query_current_presentation(value) != presentation_status::open_bracket && !value.explicit_bracket, "failed PCL End retained open bracket");
    result_v2 = 0;
    opaque_token delayed; mint(delayed, nullptr);
    block_original = true;
    std::atomic<bool> forwarded{};
    std::atomic<presentation_status> after_reinitialize{presentation_status::open_bracket};
    std::thread worker([&] {
      SetLastError(incoming_error);
      forwarded = call(4, delayed.ref()) == 0 && GetLastError() == outgoing_error;
      presentation_snapshot local; after_reinitialize = query_current_presentation(local);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    shutdown(); initialize(nullptr);
    block_original = false; release_original = true; worker.join();
    require(entered && forwarded && after_reinitialize != presentation_status::open_bracket,
      "in-flight PCL completion crossed shutdown/reinitialize epoch or changed forwarding");
    retrieve(4, name); mint(token, nullptr); marker(4, token);
    require(query_current_presentation(value) == presentation_status::open_bracket, "retained PCL trampoline did not recover after reinitialize");
    shutdown();
    const auto before = pcl_call.calls;
    marker(5, token);
    require(pcl_call.calls == before + 1 && query_current_presentation(value) == presentation_status::inactive,
      "shutdown PCL hook lost pure passthrough");
  }

  void test_frame_generation_color_only_alpha() {
    namespace mask = sunshine_game3d::ui_mask;
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG alpha hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function,
      "FG alpha options wrapper was unavailable");
    auto options_call = reinterpret_cast<fg_function>(function);
    auto view = viewport(11); fg_options options;
    require(options_call(view, options) == 0, "FG alpha options call failed");
    frame_generation_snapshot fg;
    require(query_frame_generation(view.value, fg) && fg.enabled, "FG alpha mode was not confirmed");
    constexpr std::uint64_t runtime = 0x100, device = 0x200;
    mask::request wanted{runtime, device, fg.epoch, depth_observation_revision(), view.value, 3840, 2160, true};
    mask::set_request(wanted);
    int native{}, commands{};
    auto resource = modern_resource(&native); resource.width = 3840; resource.height = 2160; resource.state = 8;
    auto tag = depth_tag_for(resource); tag.type = 53; tag.lifecycle = 0; tag.area = {0, 0, 3840, 2160};
    SetLastError(incoming_error);
    require(call_v2_tag(view, &tag, 1, &commands) == 0 && GetLastError() == outgoing_error &&
        tag_call.incoming_error == incoming_error, "color-only alpha hook changed SDK forwarding");
    mask::boundary observed;
    require(mask::testing::last_attempt(runtime, observed) && observed.source.epoch == fg.epoch &&
        observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native) &&
        observed.source.resource.width == 3840 && observed.source.resource.height == 2160 &&
        observed.source.resource.area.width == 3840 && observed.source.native_state == 8 &&
        observed.source.valid_until == sunshine_scene_depth::lifetime::at_call && !observed.tag_scope &&
        !observed.source.source_frame_explicit && observed.command == reinterpret_cast<std::uint64_t>(&commands),
      "color-only global tag53 was not captured independently of depth/camera/dump/diagnostic probe");
    evaluation_snapshot depth;
    require(!testing::latest_snapshot(view.value, depth), "alpha-only tag nominated a depth evaluation");
    const auto global_sequence = observed.source.sequence;
    opaque_token token; std::uint32_t numeric = 71; mint(token, &numeric);
    require(call_v2_framed_tag(token.ref(), view, &tag, 1, &commands) == 0 &&
        mask::testing::last_attempt(runtime, observed) && observed.source.sequence > global_sequence &&
        observed.tag_scope == 1 && observed.source.source_frame_explicit &&
        observed.source.source_frame_has_numeric && observed.source.source_frame_numeric == numeric &&
        observed.source.source_frame_token == reinterpret_cast<std::uint64_t>(&token.ref()),
      "explicit-frame tag53 lost its own frame provenance");
    tag.resource_ptr = nullptr;
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed) && !observed.source.resource.native,
      "null color tag did not reach alpha revocation owner");
    const auto null_sequence = observed.source.sequence;
    tag.resource_ptr = &resource;
    testing::lose_observation();
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed) && observed.source.sequence == null_sequence,
      "new observation revision captured into an old requested alpha scope");
    wanted.revision = depth_observation_revision(); mask::set_request(wanted);
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed), "renewed alpha scope failed to recover");
    options.mode = 0; options_call(view, options);
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport) &&
        !mask::testing::last_attempt(runtime, observed), "FG Off retained alpha owner scope");
    options.mode = 1; options_call(view, options);
    wanted.revision = depth_observation_revision(); mask::set_request(wanted);
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed), "re-enabled FG did not resume color-only alpha capture");
    result_v2 = -7; options_call(view, options); result_v2 = 0;
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport), "failed FG options retained alpha scope");
    mask::set_request(wanted); shutdown();
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport), "shutdown retained live alpha request");
  }

  void test_frame_generation_tags() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    SetLastError(incoming_error);
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function &&
        function != returned_feature_function && GetLastError() == outgoing_error,
      "FG getter did not return a first-call-safe forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(11);
    fg_options options;
    frame_generation_snapshot fg;
    require(!query_frame_generation(view.value, fg), "getter alone enabled FG");
    // The SDK immediately calls its cached function. No poll/deferred hook is
    // permitted between this getter and the first/only On setting.
    SetLastError(incoming_error);
    require(set_options(view, options) == 0 && fg_call.first == &view && fg_call.second == &options &&
        fg_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
      "FG wrapper changed arguments/result/LastError");
    require(query_frame_generation(UINT32_MAX, fg) && fg.viewport == view.value && fg.enabled && fg.generated_frames == 1,
      "first FG On was missed");
    {
      provider::frame_generation_policy policy;
      require(!policy.update(frame_generation_query_status::unavailable, {}).require_frame_generation,
        "Unobserved unavailable FG blocked ordinary depth routing");
      require(policy.update(frame_generation_query_status::busy, {}).require_frame_generation,
        "A first busy FG query was treated as confirmed Off");
      auto selected = policy.update(frame_generation_query_status::observed, fg);
      require(selected.require_frame_generation && selected.epoch == fg.epoch && selected.viewport == fg.viewport &&
          policy.generated_frames() == fg.generated_frames,
        "Confirmed FG did not select its exact scope");

      frame_generation_snapshot contended_snapshot;
      frame_generation_query_status contended_status{};
      bool queried = true;
      std::atomic<bool> completed{};
      struct records_guard {
        bool locked{true};
        records_guard() { testing::lock_records(); }
        ~records_guard() { release(); }
        void release() { if (locked) { testing::unlock_records(); locked = false; } }
      } guard;
      std::thread reader([&] {
        queried = query_frame_generation(UINT32_MAX, contended_snapshot, &contended_status);
        completed.store(true, std::memory_order_release);
      });
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
      while (!completed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      const bool nonblocking = completed.load(std::memory_order_acquire);
      guard.release();
      reader.join();
      require(nonblocking && queried && contended_status == frame_generation_query_status::observed &&
          contended_snapshot.known && contended_snapshot.epoch == fg.epoch && contended_snapshot.sequence == fg.sequence,
        "Metadata writer hid or mutated the last complete FG publication");
      selected = policy.update(contended_status, contended_snapshot);
      require(selected.require_frame_generation && selected.epoch == fg.epoch && selected.viewport == fg.viewport &&
          policy.generated_frames() == fg.generated_frames,
        "Contended FG query permitted an established NGX provider to take over");
      selected = policy.update(frame_generation_query_status::ambiguous, {});
      require(selected.require_frame_generation && !selected.epoch && !policy.generated_frames(),
        "Ambiguous FG reused old scope or selected ordinary depth");
      selected = policy.update(frame_generation_query_status::busy, {});
      require(selected.require_frame_generation && !selected.epoch,
        "Busy query resurrected scope revoked by observation loss");
      policy.update(frame_generation_query_status::observed, fg);
      selected = policy.update(frame_generation_query_status::unavailable, {});
      require(selected.require_frame_generation && !selected.epoch,
        "Lost FG evidence selected ordinary depth or preserved an obsolete scope");
      auto off = fg; off.enabled = false;
      require(!policy.update(frame_generation_query_status::observed, off).require_frame_generation &&
          !policy.update(frame_generation_query_status::busy, {}).require_frame_generation && !policy.generated_frames(),
        "Confirmed Off did not restore ordinary routing through a busy query");
      policy = {};
      require(!policy.update(frame_generation_query_status::unavailable, {}).require_frame_generation,
        "Runtime policy reset retained an old FG requirement");
    }
    const auto before_evaluations = testing::counts().evaluations;
    const auto before_reader_loss = depth_observation_revision();
    const auto before_reader_drops = testing::counts().dropped;
    observation_guard immutable_reader(fixture_lock::records_shared);
    require(set_options(view, options) == 0 && query_frame_generation(view.value, fg) && fg.enabled,
      "Pinned metadata reader blocked fresh FG options");
    opaque_token token; std::uint32_t numeric = 41; mint(token, &numeric);
    auto constants = modern_camera();
    constants.common.jitter_offset[0] = .25f; constants.common.jitter_offset[1] = -.375f;
    call_v2_constants(constants, token.ref(), view);
    int native{}, commands{};
    auto resource = modern_resource(&native);
    resource.type = 8; // Supported eUnknown crosses the shared owner's QI gate.
    auto tag = depth_tag_for(resource); tag.lifecycle = 0;
    tag_precision precision; tag.base.next = &precision;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    evaluation_snapshot snapshot; sunshine_scene_depth::frame normalized;
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.tag_boundary && snapshot.feature == 1000 &&
        snapshot.successful_evaluation && snapshot.tags[0].supported && snapshot.frame_correlated,
      "FG OnlyValidNow tag did not publish its synchronous frame/camera tuple");
    require(testing::normalized_source(view.value, 0, normalized) && normalized.frame_generation_input &&
        normalized.source_id == ((1ull << 63) | view.value) && normalized.source_frame_explicit &&
        normalized.source_frame_has_numeric && normalized.source_frame_numeric == numeric &&
        normalized.valid_until == sunshine_scene_depth::lifetime::at_call && normalized.projection.supplied &&
        normalized.projection.depth_offset == 0 && normalized.projection.depth_scale == .5 && normalized.projection.reversed &&
        normalized.projection.raw_scale == .5 && normalized.projection.raw_bias == .25 &&
        normalized.jitter.supplied && normalized.jitter.x == .25f && normalized.jitter.y == -.375f &&
        normalized.jitter.width == 1920 && normalized.jitter.height == 1080,
      "FG precision transform/frame identity/lifetime normalization is incorrect");
    immutable_reader.release();
    require(depth_observation_revision() == before_reader_loss && testing::counts().dropped == before_reader_drops,
      "Pinned metadata reader invalidated FG options/camera/synchronous tag capture");
    const auto inverse_distance = [](const sunshine_scene_depth::frame &value, double raw) {
      return (raw * value.projection.raw_scale + value.projection.raw_bias - value.projection.depth_offset) / value.projection.depth_scale;
    };
    require(inverse_distance(normalized, .5) == 1.0, "FG transformed sample lost its original matrix depth units");
    require(testing::counts().evaluations == before_evaluations, "FG tags invented Super Resolution evaluation calls");
    auto sr_tag = tag; sr_tag.lifecycle = 1;
    const base_structure *sr_inputs[]{&view.base, &sr_tag.base};
    call_v2_evaluate(0, token.ref(), sr_inputs, 2, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.feature == 0 && snapshot.tags[0].supported &&
        !testing::normalized_source(view.value, 0, normalized),
      "same-viewport SR evaluation competed with enabled FG source");
    auto unrelated_view = viewport(12);
    call_v2_constants(constants, token.ref(), unrelated_view);
    const base_structure *other_sr_inputs[]{&unrelated_view.base, &sr_tag.base};
    call_v2_evaluate(0, token.ref(), other_sr_inputs, 2, &commands);
    require(testing::normalized_source(unrelated_view.value, 0, normalized) && !normalized.frame_generation_input,
      "FG mode on one viewport suppressed another viewport's SR input");

    precision.scale = -.5f;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(testing::normalized_source(view.value, 0, normalized) && normalized.projection.depth_offset == 0 &&
        normalized.projection.depth_scale == .5 && normalized.projection.raw_scale == -.5 &&
        !normalized.projection.reversed && inverse_distance(normalized, 0.0) == .5,
      "negative precision scale failed to transform depth direction");
    precision.formula = 0; precision.scale = 0;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(testing::normalized_source(view.value, 0, normalized) && normalized.projection.depth_offset == 0 &&
        normalized.projection.depth_scale == .5 && normalized.projection.raw_scale == 1 && normalized.projection.raw_bias == 0,
      "NoTransform precision used ignored scale/bias fields");

    // Global FG tags are legal but have weaker frame association than an
    // explicit token. High-resolution-only tags must use the same contract.
    tag.type = 48;
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::normalized_source(view.value, 1, normalized) && !normalized.source_frame_explicit &&
        normalized.projection.supplied && normalized.resource.kind == sunshine_scene_depth::resource_kind::display_depth,
      "global high-resolution-only FG tag lost camera association");
    mint(token, &numeric);
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.tag_boundary &&
        snapshot.frame.kind == frame_identity_kind::unavailable && !testing::normalized_source(view.value, 1, normalized),
      "global FG tag borrowed camera constants after their opaque token was recycled");
    call_v2_constants(constants, token.ref(), view);

    precision.formula = 1; precision.scale = 0;
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.tag_boundary && !snapshot.tags[1].supported &&
        !testing::normalized_source(view.value, 1, normalized), "invalid precision inherited old usable FG depth");
    tag.base.next = nullptr;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::normalized_source(view.value, 1, normalized), "valid FG source did not recover after invalid precision");
    tag.resource_ptr = nullptr;
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.tag_boundary && !snapshot.tags[1].value.native_resource &&
        !testing::normalized_source(view.value, 1, normalized), "FG untag retained prior source");
    tag.resource_ptr = &resource;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    result_v2 = -7;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && !snapshot.successful_evaluation &&
        !testing::normalized_source(view.value, 1, normalized), "failed FG tag was promoted to a successful source");
    result_v2 = 0;

    options.mode = 0; require(set_options(view, options) == 0, "FG Off forwarding failed");
    require(query_frame_generation(UINT32_MAX, fg) && !fg.enabled, "FG Off retained enabled state");
    const auto last_fg_sequence = snapshot.sequence;
    call_v2_tag(view, &tag, 1, &commands);
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.sequence == last_fg_sequence,
      "FG Off still captured standalone FG tags");
    sr_tag.base.next = nullptr;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    call_v2_evaluate(0, token.ref(), sr_inputs, 2, &commands);
    require(testing::normalized_source(view.value, 0, normalized) && !normalized.frame_generation_input,
      "FG Off did not restore ordinary SR source admission");
    options.mode = 2; options.generated_frames = 3; set_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled && fg.automatic && fg.generated_frames == 3,
      "Auto/MFG options were lost");
    options.base.version = 4; set_options(view, options);
    require(!query_frame_generation(view.value, fg), "unknown FG options version retained a known enabled state");
    options.base.version = 3; set_options(view, options);
    auto second_view = viewport(12); set_options(second_view, options);
    frame_generation_query_status ambiguous_status;
    require(!query_frame_generation(UINT32_MAX, fg, &ambiguous_status) &&
        ambiguous_status == frame_generation_query_status::ambiguous && query_frame_generation(view.value, fg),
      "ambiguous FG viewports were silently combined");
    returned_feature_function = reinterpret_cast<void *>(&fake_pcl_marker);
    require(call_feature_function(abi_v2::feature_pcl, "slPCLSetMarker", function) == 0 &&
        testing::install_presentation_hooks(), "source-only PCL hook discovery failed");
    mint(token, &numeric);
    auto marker = reinterpret_cast<abi_v2::pcl_set_marker>(function);
    marker(4, token.ref());
    presentation_snapshot bracket;
    require(query_current_presentation(bracket) == presentation_status::open_bracket && bracket.explicit_bracket,
      "source-only FG lost its explicit present bracket when diagnostics were disabled");
    marker(5, token.ref());
    shutdown();
    const auto calls = fg_call.calls;
    require(set_options(view, options) == 0 && fg_call.calls == calls + 1 && !query_frame_generation(view.value, fg),
      "retained FG wrapper did not pass through after shutdown");
  }

  void test_frame_generation_cached_options_and_global_camera() {
    fixture cleanup;
    probe_setting = "0"; source_setting = "1"; initialize(nullptr);
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *cached_entry{};
    require(call_feature_function(1000, "slDLSSGSetOptions", cached_entry) == 0 && cached_entry == returned_feature_function,
      "pre-discovery SDK options cache did not retain the native entry");
    auto cached_options = reinterpret_cast<fg_function>(cached_entry);
    auto view = viewport(11); fg_options options;
    cached_options(view, options); // A past On cannot be invented by discovery.
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets) &&
        testing::discover_frame_generation_hooks(), "cached FG target discovery/detour failed");
    frame_generation_snapshot fg;
    require(!query_frame_generation(view.value, fg), "target discovery invented a prior FG options call");
    const auto calls = fg_call.calls;
    SetLastError(incoming_error);
    require(cached_options(view, options) == 0 && fg_call.calls == calls + 1 && fg_call.first == &view &&
        fg_call.second == &options && fg_call.incoming_error == incoming_error && GetLastError() == outgoing_error &&
        query_frame_generation(view.value, fg) && fg.enabled,
      "pre-cached FG pointer bypassed observation or changed SDK forwarding");
    const auto sequence = fg.sequence;
    void *wrapped_entry{};
    call_feature_function(1000, "slDLSSGSetOptions", wrapped_entry);
    auto wrapped_options = reinterpret_cast<fg_function>(wrapped_entry);
    require(wrapped_entry != cached_entry, "future getter lost its immediate forwarding wrapper");
    wrapped_options(view, options);
    require(fg_call.calls == calls + 2 && query_frame_generation(view.value, fg) && fg.sequence == sequence + 1,
      "wrapper plus native detour observed or invoked one options call twice");
    require(testing::discover_frame_generation_hooks(), "repeat FG discovery attempted a duplicate hook");
    result_v2 = -7; cached_options(view, options);
    require(!query_frame_generation(view.value, fg), "failed cached options call retained known enabled state");
    result_v2 = 0; options.mode = 0; cached_options(view, options);
    require(query_frame_generation(view.value, fg) && !fg.enabled, "cached FG Off was missed");
    options.mode = 1; cached_options(view, options);

    // A global OnlyValidNow tag can pair with the immutable successful
    // constants call even when the game's token mint predates observation.
    opaque_token token; auto constants = modern_camera();
    int native{}, commands{};
    auto resource = modern_resource(&native); resource.base = {};
    resource.width = resource.height = 0; // Shared native description owns size.
    auto tag = depth_tag_for(resource); tag.lifecycle = 0;
    evaluation_snapshot snapshot; sunshine_scene_depth::frame normalized;
    const auto tag_now = [&] { call_v2_tag(view, &tag, 1, &commands); };
    tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "global FG tag invented missing camera constants");
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.tags[0].supported && snapshot.frame_correlated &&
        snapshot.frame.kind == frame_identity_kind::v2_constants_call &&
        testing::normalized_source(view.value, 0, normalized) && normalized.projection.supplied &&
        normalized.proof == sunshine_scene_depth::state_proof::observed_nonzero &&
        !normalized.source_frame_explicit && !normalized.source_frame_has_numeric && normalized.source_frame_generation,
      "zero-base global FG tag lost its bounded constants-call association or claimed a numeric frame");
    require(normalized.observation_revision == snapshot.loss_revision &&
        normalized.observation_revision == depth_observation_revision(),
      "FG normalized depth omitted its immutable source observation revision");
    const auto first_camera_sequence = snapshot.camera_sequence;
    call_v2_constants(constants, token.ref(), view);
    require(!testing::normalized_source(view.value, 0, normalized), "new constants resurrected the previous camera/tag tuple");
    tag_now();
    require(testing::latest_snapshot(view.value, snapshot) && snapshot.camera_sequence > first_camera_sequence &&
        testing::normalized_source(view.value, 0, normalized), "new successful constants did not establish a new camera-call tuple");
    const auto valid_resource = resource;
    resource.state = 0x20; call_v2_constants(constants, token.ref(), view); tag_now();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.native_state == 0x20 &&
        normalized.proof == sunshine_scene_depth::state_proof::declared, "nonzero compatibility resource state was discarded");
    resource = valid_resource; resource.base = {nullptr, resource_guid, 1};
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.native_state == 0 &&
        normalized.proof == sunshine_scene_depth::state_proof::declared, "typed resource lost declared COMMON state");
    resource = valid_resource;
    const auto rejected_header = [&] {
      call_v2_constants(constants, token.ref(), view); tag_now();
      require(testing::latest_snapshot(view.value, snapshot) && !snapshot.tags[0].supported &&
          !testing::normalized_source(view.value, 0, normalized), "malformed nested resource header entered FG capture");
      resource = valid_resource;
    };
    resource.base.version = 1; rejected_header(); // Empty GUID with nonzero version.
    resource.base.type.a = 1; rejected_header();
    resource.base.type = resource_guid; resource.base.version = 2; rejected_header();
    resource.base.next = &native; rejected_header();
    resource.type = 3; rejected_header();
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(testing::normalized_source(view.value, 0, normalized), "valid zero-base resource failed to recover");
    auto high_resolution = tag; high_resolution.type = 48; high_resolution.lifecycle = 1;
    abi_v2::resource_tag mixed_tags[]{tag, high_resolution};
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(testing::normalized_source(view.value, 0, normalized) && !testing::normalized_source(view.value, 1, normalized),
      "mixed lifecycle batch lent synchronous camera evidence to an UntilPresent candidate");
    mixed_tags[1].lifecycle = 2;
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(testing::normalized_source(view.value, 0, normalized) && !testing::normalized_source(view.value, 1, normalized),
      "mixed lifecycle batch lent synchronous camera evidence to an UntilEvaluate candidate");
    mixed_tags[0].lifecycle = 1; mixed_tags[1].lifecycle = 0;
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(!testing::normalized_source(view.value, 0, normalized) && testing::normalized_source(view.value, 1, normalized),
      "mixed lifecycle batch rejected its valid high-resolution OnlyValidNow candidate");
    tag.lifecycle = 1; tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "untracked camera escaped synchronous OnlyValidNow scope");
    tag.lifecycle = 0;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(!testing::normalized_source(view.value, 0, normalized), "untracked explicit token borrowed global camera-call evidence");
    call_v2_constants(constants, token.ref(), view);
    std::this_thread::sleep_for(std::chrono::milliseconds(270)); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "stale global FG constants were accepted");
    call_v2_constants(constants, token.ref(), view); testing::lose_observation(); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "global FG constants crossed observation loss");
    const auto revision_before_reset = depth_observation_revision();
    constants.reset = 1; call_v2_constants(constants, token.ref(), view);
    require(depth_observation_revision() != revision_before_reset,
      "FG camera reset did not revoke retained depth before the next tag");
    tag_now();
    require(testing::normalized_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied &&
        normalized.observation_revision == depth_observation_revision() && normalized.frame_generation_input,
      "valid global FG reset frame lost current depth/projection after retiring older depth");
    constants.reset = 2; call_v2_constants(constants, token.ref(), view); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "malformed reset flag entered global FG capture");
    constants = modern_camera(); constants.reset = 1; constants.common.camera_view_to_clip.m[0][0] = 0;
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "reset admitted an invalid projection into global FG capture");
    constants = modern_camera(); constants.common.camera_view_to_clip.m[0][0] = 0;
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "invalid global camera projection entered FG capture");
    constants = modern_camera(); call_v2_constants(constants, token.ref(), view);
    result_v2 = -1; call_v2_constants(constants, token.ref(), view); result_v2 = 0; tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "failed constants call retained an applicable camera tuple");
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(testing::normalized_source(view.value, 0, normalized), "global FG tuple did not recover after invalid/reset/lost constants");
    require(normalized.observation_revision == depth_observation_revision(),
      "fresh FG depth retained the pre-reset observation revision");
    mint(token, nullptr); tag_now();
    require(!testing::normalized_source(view.value, 0, normalized), "observed token reuse retained an older untracked camera call");
    auto another_view = viewport(12); cached_options(another_view, options);
    call_v2_tag(another_view, &tag, 1, &commands);
    require(!testing::normalized_source(another_view.value, 0, normalized), "global FG borrowed another viewport's constants");

    block_original = true;
    std::atomic<bool> forwarded{};
    std::thread worker([&] { SetLastError(incoming_error); forwarded =
      cached_options(view, options) == 0 && GetLastError() == outgoing_error; });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    shutdown(); initialize(nullptr);
    block_original = false; release_original = true; worker.join();
    require(entered && forwarded && !query_frame_generation(view.value, fg),
      "in-flight cached options crossed shutdown/reinitialize or changed forwarding");
    cached_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled, "retained native FG detour did not recover after reinitialize");
    const auto before_detach_sequence = fg.sequence;
    const auto before_detach_calls = fg_call.calls;
    const auto before_detach_observation = testing::counts();
    sunshine_addon_lifetime::begin_detach();
    SetLastError(incoming_error);
    require(cached_options(view, options) == 0 && GetLastError() == outgoing_error &&
        fg_call.incoming_error == incoming_error && fg_call.calls == before_detach_calls + 1 &&
        !query_frame_generation(view.value, fg), "process-detach FG hook lost pure native forwarding");
    call_v2_constants(constants, token.ref(), view); tag_now();
    unchanged(before_detach_observation, testing::counts(), "process-detach SDK hook entered destroyed observation owners");
    sunshine_addon_lifetime::detaching.store(false, std::memory_order_release); // Isolated fixture only.
    require(query_frame_generation(view.value, fg) && fg.sequence == before_detach_sequence,
      "process-detach options call updated observation state");
    shutdown();
    const auto stopped_calls = fg_call.calls;
    cached_options(view, options); wrapped_options(view, options);
    require(fg_call.calls == stopped_calls + 2 && !query_frame_generation(view.value, fg),
      "shutdown cached/native FG hooks lost pure pass-through");
  }

  void test_version_identity() {
    using versioning::abi;
    auto make = [](unsigned major, unsigned minor, unsigned patch, const wchar_t *file, const wchar_t *product) {
      versioning::information value;
      value.fixed_valid = true;
      value.file_ms = value.product_ms = MAKELONG(minor, major);
      value.file_ls = value.product_ls = MAKELONG(0, patch);
      std::wcsncpy(value.file_text.data(), file, value.file_text.size() - 1);
      std::wcsncpy(value.product_text.data(), product, value.product_text.size() - 1);
      value.strings_complete = file[0] && product[0];
      return value;
    };
    auto v1 = make(1, 1, 1, L"1.1.1.0", L"1.1.1.0");
    require(versioning::classify(v1) == abi::v1_1_1, "validated v1 fixed version rejected");
    auto v2 = make(2, 7, 30, L"2,7,30,0", L"2,7,30,0");
    require(versioning::classify(v2) == abi::v2_7_30, "installed v2 comma-string version rejected");
    auto legacy = make(1, 0, 0, L"1.1.1.0", L"1.1.1.0");
    require(versioning::classify(legacy) == abi::v1_1_1, "pinned v1 SDK fixed-resource quirk rejected");
    require(std::strstr(legacy.reason, "quirk") != nullptr, "v1 quirk recognition not diagnosed");
    for (const auto *text : {L"", L"1.0.0.0", L"1.1.2.0", L"1.1.1.0-beta", L"2.7.30.0", L"1,1,1,0"}) {
      auto rejected = make(1, 0, 0, text, text);
      require(versioning::classify(rejected) == abi::unsupported, "unknown legacy string pair accepted");
    }
    auto missing_product = make(1, 0, 0, L"1.1.1.0", L"");
    require(versioning::classify(missing_product) == abi::unsupported, "one string alone enabled legacy ABI");
    auto conflict = make(1, 0, 0, L"1.1.1.0", L"1.1.2.0");
    require(versioning::classify(conflict) == abi::unsupported, "conflicting legacy versions accepted");
    auto unknown_fixed = make(1, 2, 0, L"1.1.1.0", L"1.1.1.0");
    require(versioning::classify(unknown_fixed) == abi::unsupported, "string fallback broadened to unknown fixed version");
    auto bad_product = legacy;
    bad_product.product_ls = MAKELONG(0, 1);
    require(versioning::classify(bad_product) == abi::unsupported, "legacy fixed product mismatch accepted");
    auto bad_fixed = v1;
    bad_fixed.fixed_valid = false;
    require(versioning::classify(bad_fixed) == abi::unsupported, "invalid fixed resource accepted");
    auto conflict_translation = legacy;
    conflict_translation.strings_conflict = true;
    require(versioning::classify(conflict_translation) == abi::unsupported, "conflicting translation versions accepted");
    auto incomplete_translation = legacy;
    incomplete_translation.strings_complete = false;
    require(versioning::classify(incomplete_translation) == abi::unsupported, "partial translation fallback accepted");
    auto bad_known = make(2, 7, 30, L"2.7.31.0", L"2.7.31.0");
    require(versioning::classify(bad_known) == abi::unsupported, "contradictory fixed/string identity accepted");
    auto missing = versioning::read_file(L"Z:\\__sunshine_streamline_no_such_file__.dll");
    require(versioning::classify(missing) == abi::unsupported && std::strstr(missing.reason, "resource"),
      "missing file did not produce a specific rejection reason");
  }
}  // namespace

// Exercise the trace through the actual SL detours with both ordinary source
// selection and expensive camera/content diagnostics disabled.
static void test_call_trace_only(bool modern) {
  using namespace sunshine_streamline;
  fixture cleanup;
  probe_setting = "0";
  source_setting = "0";
  call_trace_setting = "1";
  initialize(nullptr);
  require(sunshine_upscaler_trace::enabled() && !enabled() && !source_enabled(),
    "Call trace enabled source selection or full camera diagnostics");
  require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
    modern ? v2_targets : v1_targets), "Trace-only SL hook install failed");
  const auto before = sunshine_upscaler_trace::testing::counts();
  opaque_token token;
  for (bool success : {true, false}) {
    result_v1 = success;
    result_v2 = success ? 0 : -7;
    SetLastError(incoming_error);
    if (modern)
      require(call_v2_evaluate(0, token.ref(), nullptr, 0, nullptr) == result_v2, "Traced v2 result changed");
    else
      require(call_v1_evaluate(nullptr, 0, 1, 0) == result_v1, "Traced v1 result changed");
    require(GetLastError() == outgoing_error && evaluate_call.incoming_error == incoming_error,
      "Trace scope changed incoming or outgoing LastError");
  }
  const auto after = sunshine_upscaler_trace::testing::counts();
  require(after.streamline == before.streamline + 2 && after.streamline_covered,
    "Trace-only SL evaluations or coverage missing");
  require(testing::counts().evaluations == 0 && !capture_command_marker(1),
    "Trace-only calls entered source or content diagnostics");
  shutdown();
  const auto stopped = sunshine_upscaler_trace::testing::counts().streamline;
  if (modern) call_v2_evaluate(0, token.ref(), nullptr, 0, nullptr);
  else call_v1_evaluate(nullptr, 0, 1, 0);
  require(sunshine_upscaler_trace::testing::counts().streamline == stopped,
    "Disabled call trace observed retained pass-through hook");
  call_trace_setting = nullptr;
  probe_setting = "1";
}

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--metadata-callback-bench") == 0) {
      benchmark_metadata_callbacks();
      return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--metadata-reader-isolation") == 0) {
      test_metadata_reader_isolation();
      std::puts("PASS published reader cannot lose camera/tag/evaluation callbacks");
#ifndef SUNSHINE_STREAMLINE_PROBE_LEGACY_CONTROL
      test_first_local_raw_entry();
      test_pinned_metadata_lifecycle();
      test_token_reader_and_storage_pressure();
      std::puts("PASS first local raw entry, pinned lifecycle and token reader/storage boundaries");
#endif
      return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--token-lock-isolation") == 0) {
      bool passed = true;
      for (bool shared : {false, true}) {
        const auto name = shared ? "shared records lock" : "exclusive records lock";
        try {
          test_token_records_independence(shared);
          std::printf("PASS token registration independent of %s\n", name);
        }
        catch (const std::exception &error) {
          std::fprintf(stderr, "FAIL %s: %s\n", name, error.what());
          passed = false;
        }
      }
      try {
        test_token_contention_and_remint();
        std::puts("PASS genuine token contention and same-address remint remain fail closed");
      }
      catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL token contention/remint: %s\n", error.what());
        passed = false;
      }
      try {
        test_pending_token_lifecycle();
        std::puts("PASS pending token callback cannot publish across shutdown/reinitialization");
      }
      catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL pending token lifecycle: %s\n", error.what());
        passed = false;
      }
      return passed ? 0 : 1;
    }
    if (argc == 3 && std::strcmp(argv[1], "--inspect-version") == 0) {
      wchar_t path[32768]{};
      require(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, argv[2], -1, path, static_cast<int>(std::size(path))) > 0,
        "invalid UTF-8 inspection path");
      auto info = versioning::read_file(path);
      const auto version = versioning::classify(info);
      std::printf("fixed=%u.%u.%u.%u strings='%ls'/'%ls' ABI=%d reason=%s\n",
        HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        info.file_text.data(), info.product_text.data(), static_cast<int>(version), info.reason);
      return version == versioning::abi::unsupported ? 1 : 0;
    }
    require(depth_capture::testing::unsupported_com_boundary_regression(), "capture boundary used an unsupported COM interface");
    std::puts("PASS valid IUnknown-only object rejected before graphics/queue methods and provider ownership");
    require(depth_capture::testing::zero_cookie_submission_regression(), "unobserved submission matched a retired capture producer");
    std::puts("PASS zero-cookie submissions preserve retired producers while real producers/consumers/replay remain tracked");
    require(depth_capture::testing::submission_completion_regression(), "submission ordering invalidated pending completion or admitted an unfinished/failed capture");
    require(depth_capture::testing::provider_admission_regression(), "provider ownership admitted unusable or cross-provider depth");
    require(depth_capture::testing::crop_region_regression(), "native depth crop admitted an unsupported copy or rejected a valid extent");
    require(depth_capture::testing::record_diagnostic_regression(), "record diagnostics changed admission or were overwritten by another attempt");
    std::puts("PASS pending evaluation is unavailable until success in either submission order; failure, Signal loss and queue change remain terminal");
    require(depth_capture::testing::source_cookie_reentry_regression(), "retagged live resource lost its recorded state identity");
    std::puts("PASS live resource keeps its cookie across expired metadata wrappers; new resources and failed writes remain separate");
    require(depth_capture::testing::recording_recovery_regression(), "native command ownership, reset, capacity or retirement contract failed");
    std::puts("PASS 160 live native command owners, destruction/recreation, global invalidation and Reset preserve exact recording identity without assuming GPU completion");
    require(depth_capture::testing::recording_state_loss_regression(), "declared state could bypass alias/split history loss");
    std::puts("PASS 162 source states preserve unrelated depth and named alias/split blocks; Reset reuses storage; wildcard/identity/allocation loss rejects without retiring owned copies");
    test_version_identity();
    std::puts("PASS exact version identity, pinned v1 resource quirk and unknown/conflicting rejection");
    test_diagnostic_opt_in();
    std::puts("PASS default-off diagnostic config, absent command/native tracking, opt-in hooks and disabled retained pass-through");
    test_source_nomination_v1();
    test_source_nomination_v2();
    test_jitter_frame_and_render_domain(false);
    test_jitter_frame_and_render_domain(true);
    test_high_resolution_jitter_domain();
    std::puts("PASS v1/v2 frame-owned jitter, padded render extent, high-res domain proof and optional metadata rejection");
    test_source_without_projection();
    test_frame_generation_color_only_alpha();
    std::puts("PASS live pre-FG alpha observes color-only global/frame tag53 with diagnostic probe off; null, revision, Off and shutdown revoke scope");
    test_frame_generation_tags();
    std::puts("PASS FG first-call options wrapper, synchronous OnlyValidNow tags, precision transforms, global/explicit frames, Off and failure gates");
    test_frame_generation_cached_options_and_global_camera();
    std::puts("PASS pre-cached FG target detour, single forwarding, zero-base resource compatibility, bounded global camera-call pairing and lifecycle gates");
    test_until_evaluate_lifetime();
    std::puts("PASS UntilEvaluate global/frame/local pre-call admission, post-return expiry and unchanged extent rejection");
    std::puts("PASS default-on lightweight source nomination, no selected-depth dependency, lifetime ABA, freshness, loss and viewport gates");
    test_source_lifecycle_contention();
    std::puts("PASS bounded lifecycle lock contention preserves existing resources, records destruction and rejects queued old epochs");
    test_token_records_independence(false);
    test_token_records_independence(true);
    test_token_contention_and_remint();
    test_pending_token_lifecycle();
    test_metadata_reader_isolation();
#ifndef SUNSHINE_STREAMLINE_PROBE_LEGACY_CONTROL
    test_first_local_raw_entry();
    test_pinned_metadata_lifecycle();
    test_token_reader_and_storage_pressure();
#endif
    std::puts("PASS camera-table readers/writers cannot lose token registration; genuine token contention and remint still reject stale source identity");
    test_v1_forwarding();
    std::puts("PASS v1 real-hook forwarding and success-only capture");
    test_v2_forwarding();
    std::puts("PASS v2 real-hook forwarding and success-only capture");
    test_v1_pairing();
    std::puts("PASS v1 viewport isolation and conservative frame pairing");
    test_v2_pairing();
    std::puts("PASS v2 viewport/local-call pairing without opaque-token frame claims");
    test_v2_depth_kinds_and_chained_inputs();
    std::puts("PASS v2 raw/high-resolution/linear depth kinds, linked inputs and bounded arrays");
    test_absent_unsupported_and_delayed_install();
    std::puts("PASS absent DLL, rejected exports, partial rollback and delayed hook installation");
    test_shutdown(false);
    test_shutdown(true);
    std::puts("PASS v1/v2 in-flight shutdown and retained pass-through trampolines");
    test_shutdown(false, true);
    test_shutdown(true, true);
    std::puts("PASS v1/v2 stale in-flight completion rejected after reinitialization");
    test_last_error(false);
    test_last_error(true);
    std::puts("PASS v1/v2 incoming/outgoing LastError on success and failure");
    test_depth_observation_loss_diagnostics(false);
    test_depth_observation_loss_diagnostics(true);
    std::puts("PASS source-only v1/v2 exact-revision reset, invalid camera, SDK failure and lock contention causes preserve forwarding and source recovery");
    test_evaluation_v1();
    std::puts("PASS v1 evaluation evidence, exact frame history, copy isolation and fail-closed recovery");
    test_scene_feedback(false);
    test_scene_feedback(true);
    test_scene_feedback_inflight();
    test_reset_constants_observation_loss(false);
    test_reset_constants_observation_loss(true);
    std::puts("PASS v1/v2 fresh reset depth/projection, immutable feedback, and rejection of pre-reset tags, in-flight completion and interrupted reset constants");
    test_evaluation_immutable_and_ordered();
    std::puts("PASS immutable evaluation ordering, bounded frame eviction and age limits");
    test_evaluation_v2_tokens();
    std::puts("PASS v2 observed token generations, numeric provenance, frame tags and local overrides");
    test_evaluation_ambiguity_and_invalid_tags();
    std::puts("PASS ambiguous viewport and newer invalid tag rejection");
    test_evaluation_global_contract();
    std::puts("PASS complete active global/frame tag contract revalidation");
    test_evaluation_input_age();
    std::puts("PASS age uses observation entry rather than delayed original completion");
    test_evaluation_command_markers();
    std::puts("PASS real-hook recording markers, submit association, repeats and in-flight reset rejection");
    test_evaluation_content();
    std::puts("PASS real-hook observed content, copy/clear orders, backup identities and in-flight mutation rejection");
    test_inactive_content_transition();
    std::puts("PASS inactive content gating preserves command markers and restarts content evidence on observer activation");
    test_non_renderer_evaluations();
    std::puts("PASS Reflex/unknown feature passthrough preserves DLSS viewport/frame evidence and ignores non-renderer failures");
    test_color_tag_snapshots();
    std::puts("PASS separate immutable HUDless/input/output/backbuffer color tags and legal null-backbuffer metadata");
    test_presentation_v1();
    test_presentation_v2();
    std::puts("PASS v1 Reflex/v2 dynamic PCL presentation brackets, thread isolation, failures, token reuse and repeated-frame rejection");
    test_call_trace_only(false);
    test_call_trace_only(true);
    std::puts("PASS v1/v2 call-route trace remains independent of depth selection and full content diagnostics");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Streamline probe: %s\n", error.what());
    return 1;
  }
}

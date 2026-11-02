// SPDX-License-Identifier: GPL-3.0-only
// Real MinHook regression against isolated, noinline fixture entrypoints. No game,
// Streamline DLL, graphics device, or GPU work is involved. Game time is the
// probe view's test clock (test_game3d_clock.h), engaged in main and moved
// only by hitch(); --wall-clock runs it on the real clock.
#include "streamline_camera_data.h"
#include "addon_lifetime.h"
#include "streamline_camera_probe.h"
#include "hook_host.h"
#include "streamline_buffer_contract.h"
#include "streamline_depth_provider.h"
#include "streamline_camera_version.h"
#include "upscaler_call_trace.h"
#include "game3d_diagnostics.h"
#include "game3d_ui_mask.h"
#include "test_game3d_clock.h"
#include "test_minhook_ledger.h"
#include "test_sl_fixture_sdk.h"
#include "test_streamline_depth_capture_fixture.h"
#include "ngx_depth_source.h"
#include <reshade.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// ReShade's header-only logging API locates these exports. Supply only a log
// sink/discovery identity, never a graphics runtime, for the real no-DLL poll.
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t) {
  return false;
}

extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {}

// The target logs synchronously (SUNSHINE_SYNCHRONOUS_LOG): lines are kept
// for the tests that check what the observer reports.
static std::mutex logged_lock;
static std::vector<std::string> logged_lines;
extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *message) {
  std::lock_guard lock(logged_lock);
  if (message && logged_lines.size() < 4096) logged_lines.emplace_back(message);
}

// Fixtures set these through the same public config API as the game: the
// shipping configuration (StreamlineDepthSource=1). nullptr models a
// missing setting for the default-on regression.
static const char *source_setting = "1";
static const char *call_trace_setting = nullptr;
extern "C" __declspec(dllexport) bool ReShadeGetConfigValue(void *, reshade::api::effect_runtime *,
    const char *section, const char *key, char *value, size_t *size) {
  if (std::strcmp(section, "SUNSHINE_DEPTH") != 0 || !size) return false;
  const char *setting = std::strcmp(key, "StreamlineDepthSource") == 0 ? source_setting :
    std::strcmp(key, "NGXDepthSource") == 0 ? "0" :
    std::strcmp(key, "UpscalerCallTrace") == 0 ? call_trace_setting : nullptr;
  if (!setting) return false;
  const size_t length = std::strlen(setting);
  if (!value || *size <= length) { *size = length + 1; return value == nullptr; }
  std::memcpy(value, setting, length + 1);
  *size = length;
  return true;
}

// The interposer releases these fixtures stand in for (testing::install).
namespace sunshine_streamline::testing::abi {
  constexpr versioning::number unsupported {}, v1_0_0 {1, 0, 0, 0}, v1_1_1 {1, 1, 1, 0}, v2_7_30 {2, 7, 30, 0},
    v2_11_1 {2, 11, 1, 0}, v2_12_0 {2, 12, 0, 0};
}  // namespace sunshine_streamline::testing::abi

namespace {
  using namespace sunshine_streamline;
  using sunshine_game3d_fixture::fixture_clock;

  struct invocation {
    std::uint64_t calls {};
    const void *first {}, *second {}, *third {}, *fourth {};
    std::uint32_t a {}, b {}, c {};
    DWORD incoming_error {};
  } constants_call, tag_call, framed_tag_call, evaluate_call, new_token_call, feature_function_call, fg_call;

  constexpr DWORD incoming_error = 0x12345678, outgoing_error = 0xabcdef01;
  bool result_v1 = true;
  std::int32_t result_v2 = 0;
  std::atomic<bool> block_original {}, original_entered {}, release_original {};
  abi_v2::frame_token *returned_token{};
  void *returned_feature_function{};
  bool lose_during_tag_original{};

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }
  void forget_logged() { std::lock_guard lock(logged_lock); logged_lines.clear(); }
  // Lines containing every one of the given fragments.
  unsigned logged(std::initializer_list<const char *> fragments) {
    std::lock_guard lock(logged_lock);
    unsigned count{};
    for (const auto &line : logged_lines)
      count += std::all_of(fragments.begin(), fragments.end(), [&](const char *part) { return line.find(part) != std::string::npos; });
    return count;
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
    if (lose_during_tag_original) { lose_during_tag_original = false; testing::lose_observation(); }
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
    if (lose_during_tag_original) { lose_during_tag_original = false; testing::lose_observation(); }
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
  FIXTURE_NOINLINE std::int32_t fake_get_feature_function(std::uint32_t feature, const char *name, void *&function) {
    feature_function_call.incoming_error = GetLastError(); ++feature_function_call.calls;
    feature_function_call.first = name; feature_function_call.second = &function; feature_function_call.a = feature;
    block_if_requested();
    function = returned_feature_function;
    SetLastError(outgoing_error); return result_v2;
  }
  using fixture_sdk::fg_options_type;
  using fixture_sdk::precision_type;
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

  // A fixture initializes the probe in the shipping configuration.
  struct fixture {
    fixture() {
      source_setting = "1";
      testing::clear();
      // Nominations on this thread are recorded by the nomination tap
      // (testing::latest_nomination), which the checks read.
      testing::arm_nomination_tap(true);
      constants_call = tag_call = framed_tag_call = evaluate_call = new_token_call = feature_function_call = fg_call = {};
      returned_token = nullptr;
      returned_feature_function = nullptr;
      lose_during_tag_original = false;
      result_v1 = true;
      result_v2 = 0;
      block_original = original_entered = release_original = false;
    }

    ~fixture() {
      testing::clear();
    }
  };
  // The viewport's newest nomination the tap kept (testing::latest_nomination,
  // on a thread that armed it): the evaluation snapshot source nomination
  // received before the game's call, the nomination gates' answers as that
  // call began, and the call's result.
  bool nominated_snapshot(std::uint32_t viewport, evaluation_snapshot &out) {
    testing::nomination value;
    if (!testing::latest_nomination(viewport, value)) return false;
    out = value.snapshot;
    return true;
  }
  // Whether the game's call of that nomination succeeded.
  bool nomination_succeeded(std::uint32_t viewport) {
    testing::nomination value;
    return testing::latest_nomination(viewport, value) && value.successful;
  }
  // The depth that nomination normalized for snapshot.tags[index], where the
  // call succeeded and the gates admitted it.
  bool nominated_source(std::uint32_t viewport, unsigned index, sunshine_scene_depth::frame &out) {
    out = {};
    testing::nomination value;
    if (!testing::latest_nomination(viewport, value) || !value.nominates(index)) return false;
    out = value.frames[index];
    return true;
  }
  // The Diagnostics switch on for one scope.
  struct diagnostics_on {
    const bool previous = sunshine_game3d::diagnostics::enabled();
    diagnostics_on() { sunshine_game3d::diagnostics::set_enabled(true); }
    ~diagnostics_on() { sunshine_game3d::diagnostics::set_enabled(previous); }
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

  // The MinHook hooks one family created since a mark, in order: the
  // installed hook set the refactor (plan P4) must keep (test_minhook_ledger.h).
  void require_hooks(std::size_t from, std::initializer_list<void *> targets, const char *message) {
    if (minhook_ledger::available()) require(minhook_ledger::created_since(from) == std::vector<void *>(targets), message);
  }

  observed latest(std::uint32_t id) {
    observed value;
    value.found = testing::latest(id, value.resource, value.camera, value.exact, value.same_token);
    return value;
  }

  void test_v1_forwarding() {
    fixture cleanup;
    const auto hooks = minhook_ledger::mark();
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 real MinHook install failed");
    require_hooks(hooks, {v1_targets.constants, v1_targets.tag, v1_targets.evaluate},
      "v1 did not hook exactly its constants, tag and evaluate exports");
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

  void test_legacy_resource_without_state() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_0_0, v1_targets), "legacy resource-prefix MinHook install failed");
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    struct guarded_allocation {
      unsigned char *address{};
      ~guarded_allocation() { if (address) VirtualFree(address, 0, MEM_RELEASE); }
    } allocation{static_cast<unsigned char *>(VirtualAlloc(nullptr, system.dwPageSize * 2,
        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))};
    require(allocation.address != nullptr, "cannot allocate legacy resource guard pages");
    DWORD previous{};
    require(VirtualProtect(allocation.address + system.dwPageSize, system.dwPageSize, PAGE_NOACCESS, &previous),
      "cannot protect legacy resource guard page");
    auto *resource = new (allocation.address + system.dwPageSize - sizeof(abi_v1::resource_without_state))
      abi_v1::resource_without_state{};
    int native{};
    resource->native = &native;
    const auto *wire = reinterpret_cast<const abi_v1::resource *>(resource);
    const auto constants = old_camera();
    const extent area{0, 0, 1920, 1080};
    require(call_v1_constants(constants, 101, 17), "legacy prefix constants failed");
    SetLastError(incoming_error);
    require(call_v1_tag(wire, 0, 17, &area) && tag_call.first == wire &&
        tag_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
      "legacy prefix tag forwarding changed");
    const auto observed = latest(17);
    require(observed.found && observed.camera && observed.resource == reinterpret_cast<std::uintptr_t>(&native),
      "legacy resource read crossed its 40-byte allocation or lost native depth");
    resource->ext = &native;
    require(call_v1_tag(wire, 0, 17, &area) && latest(17).resource == 0,
      "legacy extension pointer was mistaken for declared resource state");
    resource->ext = nullptr;
    require(call_v1_tag(wire, 0, 17, &area) &&
        latest(17).resource == reinterpret_cast<std::uintptr_t>(&native),
      "legacy prefix failed to recover after unsupported extension");
  }

  void test_v2_forwarding(bool frame_tags = true) {
    fixture cleanup;
    auto exports = v2_targets;
    if (!frame_tags) exports.tag_for_frame = nullptr;
    const auto hooks = minhook_ledger::mark();
    require(testing::install(testing::abi::v2_7_30, exports), "v2 real MinHook install failed");
    if (frame_tags) require_hooks(hooks, {exports.constants, exports.tag, exports.tag_for_frame, exports.evaluate},
      "v2 did not hook exactly its constants, tag, frame tag and evaluate exports");
    else require_hooks(hooks, {exports.constants, exports.tag, exports.evaluate},
      "v2 without a frame tag export did not hook exactly its constants, tag and evaluate exports");
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
    if (frame_tags) {
      require(call_v2_framed_tag(token.ref(), view, tags, 2, &commands) == 0, "v2 framed tag success changed");
      require(framed_tag_call.calls == 1 && framed_tag_call.first == &token.ref() && framed_tag_call.second == &view && framed_tag_call.third == tags && framed_tag_call.fourth == &commands && framed_tag_call.a == 2, "v2 framed tag arguments changed");
    }
    require(call_v2_evaluate(0x87654321u, token.ref(), inputs, 1, &commands) == 0, "v2 evaluate success changed");
    require(evaluate_call.calls == 1 && evaluate_call.first == &token.ref() && evaluate_call.second == inputs && evaluate_call.third == &commands && evaluate_call.a == 0x87654321u && evaluate_call.b == 1, "v2 evaluate arguments changed");
    require(std::memcmp(&constants, &original_constants, sizeof(constants)) == 0 && std::memcmp(&resource, &original_resource, sizeof(resource)) == 0 && std::memcmp(&tags[0], &original_tag, sizeof(original_tag)) == 0, "v2 input mutated");
    const auto after = testing::counts();
    require(after.constants == before.constants + 1 && after.tags == before.tags + (frame_tags ? 2 : 1) && after.evaluations == before.evaluations + 1, "v2 successful calls not observed exactly once");

    opaque_token failed_token;
    int replacement_native {};
    auto replacement_resource = modern_resource(&replacement_native);
    auto replacement_tag = depth_tag_for(replacement_resource);
    for (const std::int32_t failure : {1, -7, 0x1234567}) {
      result_v2 = failure;
      require(call_v2_constants(constants, failed_token.ref(), view) == failure, "v2 constants failure changed");
      require(call_v2_tag(view, nullptr, 0, nullptr) == failure, "v2 tag failure changed");
      require(tag_call.second == nullptr && tag_call.third == nullptr && tag_call.a == 0, "v2 empty tag arguments replaced");
      if (frame_tags) require(call_v2_framed_tag(token.ref(), view, nullptr, 0, nullptr) == failure, "v2 framed tag failure changed");
      require(call_v2_evaluate(7, token.ref(), nullptr, 0, nullptr) == failure, "v2 evaluate failure changed");
      require(evaluate_call.second == nullptr && evaluate_call.third == nullptr && evaluate_call.b == 0, "v2 empty evaluate inputs replaced");
      if (frame_tags) require(call_v2_framed_tag(failed_token.ref(), view, &replacement_tag, 1, &commands) == failure, "v2 replacement tag failure changed");
      const base_structure *replacement_inputs[] {&view.base, &replacement_tag.base};
      require(call_v2_evaluate(7, failed_token.ref(), replacement_inputs, 2, &commands) == failure, "v2 replacement local tag failure changed");
      const auto retained = latest(view.value);
      require(retained.found && retained.camera && retained.same_token == frame_tags && retained.resource == reinterpret_cast<std::uintptr_t>(&native), "v2 failed constants/tag/evaluation overwrote successful state");
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
    sunshine_game3d::hook_host::poll();  // Real discovery path must remain inert when middleware is absent.
    require(GetModuleHandleW(L"sl.interposer.dll") == nullptr, "probe loaded middleware during discovery");
    require(!testing::install(testing::abi::unsupported, v2_targets), "unsupported ABI installed hooks");
    require(!testing::install(testing::abi::v2_7_30, {}), "empty export set installed hooks");
    for (unsigned missing = 0; missing < 3; ++missing) {
      auto incomplete = v2_targets;
      if (missing == 0) incomplete.constants = nullptr;
      if (missing == 1) incomplete.tag = nullptr;
      if (missing == 2) incomplete.evaluate = nullptr;
      require(!testing::install(testing::abi::v2_7_30, incomplete), "incomplete required v2 exports installed hooks");
    }
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

    // numTags and numInputs have no limit (S18): a longer tag array, input
    // list or input chain is read a chunk at a time, past its first 32
    // elements, and forwarded as-is.
    auto motion = raw_tag; motion.type = 1; // Motion vectors: unobserved tags.
    auto wide = viewport(74);
    std::vector<abi_v2::resource_tag> many(73, motion);
    many[40] = linear_tag;
    require(call_v2_tag(wide, many.data(), 73, nullptr) == 0 && tag_call.second == many.data() && tag_call.a == 73,
      "v2 73-tag array forwarding changed");
    require(latest(wide.value).found && latest(wide.value).resource == reinterpret_cast<std::uintptr_t>(&native_linear),
      "v2 depth tag past the first 32 of an array was not read");
    // An array unreadable past its first chunk leaves neither an older
    // resource nor a tag of the call read before it looking current.
    auto *pages = static_cast<unsigned char *>(VirtualAlloc(nullptr, 2 * 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    DWORD protection{};
    require(pages && VirtualProtect(pages + 4096, 4096, PAGE_NOACCESS, &protection), "guard page setup failed");
    auto *edge = reinterpret_cast<abi_v2::resource_tag *>(pages + 4096) - 32;
    for (unsigned i = 0; i < 32; ++i) edge[i] = motion;
    edge[0] = high_tag;
    const auto invalid_before = testing::counts().invalid;
    require(call_v2_tag(wide, edge, 33, nullptr) == 0 && tag_call.second == edge && tag_call.a == 33,
      "v2 partly unreadable array forwarding changed");
    require(latest(wide.value).found && latest(wide.value).resource == 0 && testing::counts().invalid > invalid_before,
      "v2 array unreadable past its first chunk left a tag looking current");
    VirtualFree(pages, 0, MEM_RELEASE);
    // 41 evaluate inputs, the viewport last: the depth input past the first
    // 32 is read once the viewport is known.
    auto inputs_view = viewport(75);
    std::vector<abi_v2::resource_tag> local_tags(40, motion);
    local_tags[35] = high_tag;
    std::vector<const base_structure *> local_inputs;
    for (const auto &value : local_tags) local_inputs.push_back(&value.base);
    local_inputs.push_back(&inputs_view.base);
    require(call_v2_constants(constants, token.ref(), inputs_view) == 0 &&
        call_v2_evaluate(0, token.ref(), local_inputs.data(), 41, nullptr) == 0 && evaluate_call.b == 41,
      "v2 41-input evaluation forwarding changed");
    require(latest(inputs_view.value).found && latest(inputs_view.value).resource == reinterpret_cast<std::uintptr_t>(&native_hires),
      "v2 evaluate input past the first 32 was not read");
    // A chain of 41 nodes: the viewport, 39 unobserved tags and the depth tag last.
    auto chain_view = viewport(76);
    std::vector<abi_v2::resource_tag> chained(40, motion);
    chained[39] = linear_tag;
    chain_view.base.next = &chained[0].base;
    for (unsigned i = 0; i + 1 < chained.size(); ++i) chained[i].base.next = &chained[i + 1].base;
    const base_structure *long_chain[]{&chain_view.base};
    auto chain_constants = viewport(chain_view.value);
    require(call_v2_constants(constants, token.ref(), chain_constants) == 0 &&
        call_v2_evaluate(0, token.ref(), long_chain, 1, nullptr) == 0, "v2 long-chain evaluation forwarding changed");
    require(latest(chain_view.value).found && latest(chain_view.value).resource == reinterpret_cast<std::uintptr_t>(&native_linear),
      "v2 depth tag at the end of a 41-node input chain was not read");
    // A chain that returns to a node it passed is malformed, whatever its length.
    chained[39].base.next = &chained[3].base;
    const auto cycle_before = testing::counts().invalid;
    require(call_v2_evaluate(0, token.ref(), long_chain, 1, nullptr) == 0 && testing::counts().invalid > cycle_before,
      "v2 cyclic input chain was read or changed forwarding");
    chained[39].base.next = nullptr;
    // There a tag's next may be the next input (the matrix's S15), as the
    // viewport's is: a depth tag that is not last is read, with a
    // PrecisionInfo after it as its own, whose next continues the chain. An
    // unknown structure after the tag or its PrecisionInfo is still an
    // extension (S14), and a tag call's tag is never part of a chain.
    auto order_view = viewport(77);
    tag_precision order_precision; order_precision.scale = -.5f; order_precision.bias = 8.f;
    auto first = raw_tag, second = motion;
    order_view.base.next = &first.base; first.base.next = &order_precision.base; order_precision.base.next = &second.base;
    const base_structure *ordered[]{&order_view.base};
    auto order_constants = viewport(order_view.value);
    evaluation_snapshot ordered_snapshot;
    const auto evaluate_ordered = [&] {
      return call_v2_constants(constants, token.ref(), order_constants) == 0 &&
        call_v2_evaluate(0, token.ref(), ordered, 1, nullptr) == 0 &&
        nominated_snapshot(order_view.value, ordered_snapshot) && ordered_snapshot.tags[0].present;
    };
    require(evaluate_ordered() && ordered_snapshot.tags[0].supported &&
        ordered_snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_raw) &&
        ordered_snapshot.tags[0].precision_scale == -.5 &&
        ordered_snapshot.tags[0].precision_bias == 8 && first.base.next == &order_precision.base &&
        order_precision.base.next == &second.base,
      "a chained depth input that is not last, or its PrecisionInfo, was not read");
    first.base.next = &second.base;
    require(evaluate_ordered() && ordered_snapshot.tags[0].supported && ordered_snapshot.tags[0].precision_scale == 1,
      "a chained depth input followed by another input was not read");
    base_structure chained_unknown{&second.base, {0x12345678, 1, 2, {3, 4, 5, 6, 7, 8, 9, 10}}, 1};
    first.base.next = &chained_unknown;
    require(evaluate_ordered() && !ordered_snapshot.tags[0].supported &&
        ordered_snapshot.tags[0].value.refusal == tag_refusal::extension,
      "an unknown structure after a chained depth input was not refused as its extension");
    first.base.next = &order_precision.base; order_precision.base.next = &chained_unknown;
    require(evaluate_ordered() && !ordered_snapshot.tags[0].supported &&
        ordered_snapshot.tags[0].value.refusal == tag_refusal::extension,
      "an unknown structure after a chained PrecisionInfo was not refused as its extension");
    first.base.next = &second.base; order_view.base.next = nullptr;
    require(call_v2_tag(order_view, &first, 1, nullptr) == 0 && evaluate_ordered() && !ordered_snapshot.tags[0].supported &&
        ordered_snapshot.tags[0].value.refusal == tag_refusal::extension,
      "a tag call's tag with another tag as its next was read as an input chain");
  }
  // The status of the viewport's newest tapped nomination's snapshot
  // (nominated_snapshot).
  bool nominated_is(std::uint32_t viewport, evidence_status expected) {
    evaluation_snapshot value;
    return nominated_snapshot(viewport, value) && value.status == expected;
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
      source_setting = "1"; initialize(nullptr);
      require(source_enabled(), "Token isolation enabled the optional full probe");
      require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "Token isolation hook install failed");
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
      require(nominated_source(view.value, 0, normalized) && normalized.projection.supplied &&
          normalized.projection.depth_scale == .5 && normalized.observation_revision == depth_observation_revision() &&
          nominated_snapshot(view.value, snapshot) && snapshot.frame.kind == frame_identity_kind::v2_observed_token,
        "Registered token did not establish its own fresh source tuple");
      return snapshot;
    }
    // The viewport's newest nomination normalized its depth (the gates
    // admitted it as the call began, and the call succeeded).
    bool ready() {
      sunshine_scene_depth::frame normalized;
      return nominated_source(view.value, 0, normalized);
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
  // How long a worker thread may take to finish while the fixture holds a
  // lock its call must not wait for. The verdict is that it finished while
  // the lock was still held; this bound only ends a call that waits for the
  // lock, so it is a hang bound, not a time limit (owner decision 11-02): a
  // starved machine schedules the worker late (40 busy processes failed the
  // token-table row at 250 ms, also on the real clock). --wall-clock keeps
  // the old 250 ms.
  std::chrono::milliseconds finish_bound() {
    return std::chrono::milliseconds(fixture_clock().wall_clock() ? 250 : 10000);
  }
  bool bounded_finish(const std::atomic<bool> &done) {
    const auto deadline = std::chrono::steady_clock::now() + finish_bound();
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done.load(std::memory_order_acquire);
  }
  // An engine may request the current frame's token again (same index, so Streamline returns the
  // same token) between tagging and evaluating. That is still one frame: its framed depth tag
  // must reach the evaluation. A different index at a recycled address is a new frame.
  void test_repeated_token_request_keeps_frame() {
    token_source_fixture source;
    opaque_token next;
    const std::uint32_t number = 402;
    mint(next, &number);
    require(call_v2_constants(source.constants, next.ref(), source.view) == 0 &&
        call_v2_framed_tag(next.ref(), source.view, &source.tag, 1, nullptr) == 0,
      "Repeated-token fixture changed native API forwarding");
    mint(next, &number);
    require(call_v2_evaluate(0, next.ref(), source.inputs, 1, nullptr) == 0,
      "Repeated-token evaluation changed native API forwarding");
    evaluation_snapshot snapshot;
    require(nominated_snapshot(source.view.value, snapshot) &&
        snapshot.frame.kind == frame_identity_kind::v2_observed_token && snapshot.frame.numeric == number &&
        snapshot.tags[0].present && snapshot.tags[0].scope == tag_scope::explicit_frame &&
        snapshot.status == evidence_status::source_associated_evaluation,
      "Requesting the same frame's token again orphaned that frame's tags and constants");
    const std::uint32_t following = 403;
    mint(next, &following);
    require(call_v2_evaluate(0, next.ref(), source.inputs, 1, nullptr) == 0 &&
        nominated_snapshot(source.view.value, snapshot) && snapshot.frame.numeric == following &&
        !snapshot.tags[0].present,
      "A recycled token address for the next frame index inherited the previous frame's tags");
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
    require(nominated_snapshot(source.view.value, initial), "Token fixture omitted initial identity");
    const auto before_remint = depth_observation_revision();
    const auto before_counts = testing::counts();
    // Same pointer and frame index: Streamline returned the current frame's token
    // again. It remains one frame with its identity and recorded evidence.
    mint(source.token, &source.number);
    require(depth_observation_revision() == before_remint && testing::counts().dropped == before_counts.dropped &&
        source.ready(), "Re-requesting the current frame's token revoked its evidence or caused observation loss");
    const auto reminted = source.publish(source.token);
    require(reminted.frame.generation == initial.frame.generation && reminted.frame.numeric == initial.frame.numeric,
      "Re-requesting the current frame's token changed that frame's identity");
    // A later index at the same address is a recycled token: a new frame.
    const std::uint32_t later = source.number + 3;
    mint(source.token, &later);
    require(depth_observation_revision() == before_remint && testing::counts().dropped == before_counts.dropped,
      "A recycled token address reused old identity or caused broad observation loss");
    const auto recycled = source.publish(source.token);
    require(recycled.frame.generation > initial.frame.generation && recycled.frame.numeric == later,
      "A recycled token address failed to establish a distinct exact generation");

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
        loss.details.has_sdk_result && loss.details.sdk_result == 0,
      "Lost token registration was unlabeled or revived old source evidence after unlocking");
    mint(source.token, &source.number);
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

    require(call_v2_constants(source.constants, pending.ref(), source.view) == 0 &&
        call_v2_framed_tag(pending.ref(), source.view, &source.tag, 1, nullptr) == 0 &&
        call_v2_evaluate(0, pending.ref(), source.inputs, 1, nullptr) == 0,
      "Post-restart token fixture changed native API forwarding");
    sunshine_scene_depth::frame normalized;
    require(!nominated_source(source.view.value, 0, normalized) && !source.ready(),
      "Token returned from a retired callback populated the new epoch's source identity");
    mint(pending, &number);
    source.publish(pending);
  }
  void test_metadata_reader_isolation() {
    token_source_fixture source;
    evaluation_snapshot before;
    require(nominated_snapshot(source.view.value, before), "Metadata fixture missing initial source");
    const auto revision = depth_observation_revision();
    const auto counters = testing::counts();
    std::atomic<bool> done{};
    bool forwarded{};
    observation_guard held(fixture_lock::records_shared);
    std::thread worker([&] {
      testing::arm_nomination_tap(true);
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
    require(nominated_snapshot(source.view.value, after) && source.ready() &&
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
    // Each version is a new frame: a repeated request for one frame writes nothing.
    std::uint32_t frame = source.number + 1000;
    for (unsigned i = 0; i != 8; ++i) {
      require(testing::pin_tokens(i), "Token pressure fixture lost a published version");
      if (i != 7) { ++frame; mint(next, &frame); }
    }
    const auto exhausted = depth_observation_revision();
    ++frame;
    mint(next, &frame);
    loss_diagnostics::event loss;
    require(depth_observation_revision() == exhausted + 1 &&
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
    mint(source.token, &source.number);
    source.publish(source.token);
    require(source.ready(), "Held old metadata/token reader prevented fresh lifecycle publication");
  }
  void test_first_local_raw_entry() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "First-local-raw hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function,
      "First-local-raw fixture failed FG discovery"); // No options call or metadata publication.
    opaque_token token; mint(token, nullptr);
    auto view = viewport(73);
    int native{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    const base_structure *inputs[]{&view.base, &tag.base};
    testing::arm_entry_policy();
    require(call_v2_evaluate(0, token.ref(), inputs, 2, nullptr) == 0,
      "First-local-raw evaluation changed SDK result");
    bool selected{}, admitted{};
    require(testing::entry_policy(selected, admitted) && selected && admitted,
      "Initial local raw evaluation was rejected before SDK entry when metadata store was empty");
    sunshine_scene_depth::frame normalized;
    require(nominated_source(view.value, 0, normalized) && !normalized.projection.supplied,
      "First local raw evaluation failed to publish its own camera-free source");
  }
#endif
  void test_source_nomination_v1() {
    fixture cleanup;
    source_setting = nullptr; // Missing key means production nomination on.
    initialize(nullptr);
    require(source_enabled(), "source default changed diagnostic opt-in");
    require(testing::install(testing::abi::v1_1_1, v1_targets), "production-only v1 install failed");
    int native{}, other{};
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    abi_v1::resource resource{}; resource.native = &native;
    auto constants = old_camera();
    evaluation_snapshot snapshot;
    sunshine_scene_depth::frame normalized;
    const auto query = [&](evidence_status expected, const char *message) { require(nominated_is(7, expected), message); };
    call_v1_constants(constants, 1, 7);
    call_v1_tag(&resource, 0, 7, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 7);
    call_v1_evaluate(nullptr, 0, 1, 7);
    call_v1_tag(&resource, 0, 7, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "known v1 resource not nominated without selected-depth dependency");
    require(nominated_snapshot(7, snapshot) && !snapshot.tags[1].present && snapshot.tags[0].present &&
        snapshot.tags[0].value.native_resource == address && snapshot.viewport == 7 &&
        snapshot.frame.numeric == 1 && snapshot.sequence && snapshot.epoch, "v1 nomination identity wrong");
    require(nominated_source(7, 0, normalized) && normalized.resource.native == address &&
        normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == .5 && normalized.projection.reversed &&
        normalized.proof == sunshine_scene_depth::state_proof::observed_nonzero &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_present,
      "v1 adapter did not normalize projection/resource/omitted-state authority");
    query(evidence_status::source_associated_evaluation, "ignored draw diagnostics poisoned semantic nomination");
    call_v1_evaluate(nullptr, 0, 1, 7);
    call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "fresh tag did not admit new lifetime");
    call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 1, 7);
    query(evidence_status::source_associated_evaluation, "post-ABA source did not recover");
    result_v1 = false; call_v1_evaluate(nullptr, 0, 1, 7); result_v1 = true;
    call_v1_constants(constants, 2, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 2, 7);
    testing::lose_observation();
    call_v1_constants(constants, 3, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 3, 7);
    query(evidence_status::source_associated_evaluation, "fresh source did not recover before age check");
    constants.reset = 1; call_v1_constants(constants, 4, 7); call_v1_evaluate(nullptr, 0, 4, 7);
    // A reset ends temporal history, not the still-tagged depth of this frame.
    query(evidence_status::camera_reset, "reset camera revoked its still-tagged depth");
    constants.reset = 0; call_v1_constants(constants, 5, 7); call_v1_tag(&resource, 0, 7, nullptr); call_v1_evaluate(nullptr, 0, 5, 7);
    resource.native = &other;
    call_v1_constants(constants, 5, 8); call_v1_tag(&resource, 0, 8, nullptr); call_v1_evaluate(nullptr, 0, 5, 8);
    source_setting = "0"; initialize(nullptr);
    require(!source_enabled(), "source rollback key ignored");
    const auto stopped = testing::counts(); call_v1_constants(constants, 6, 7);
    unchanged(stopped, testing::counts(), "disabled retained hooks did not pass through");
  }

  void test_source_nomination_v2() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "production-only v2 install failed");
    auto view = viewport(3); auto constants = modern_camera(); opaque_token token;
    int native_a{}, native_b{}, native_high{};
    auto resource_a = modern_resource(&native_a), resource_b = modern_resource(&native_b), resource_high = modern_resource(&native_high);
    auto raw = depth_tag_for(resource_a), local = depth_tag_for(resource_b), high = depth_tag_for(resource_high); high.type = 48;
    const base_structure *inputs[]{&view.base};
    const base_structure *local_inputs[]{&view.base, &local.base};
    evaluation_snapshot snapshot;
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &raw, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_is(view.value, evidence_status::untracked_frame), "opaque token address became source frame");
    mint(token, nullptr);
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &raw, 1, nullptr);
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), local_inputs, 2, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[1].present && snapshot.tags[1].value.type == 48 &&
        snapshot.tags[1].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_high) &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_b) && !snapshot.frame.has_numeric,
      "v2 preference/local override/opaque generation nomination wrong");
    sunshine_scene_depth::frame normalized;
    require(nominated_source(view.value, 0, normalized) &&
        normalized.resource.native == reinterpret_cast<std::uintptr_t>(&native_b) && normalized.resource.width == 1920 &&
        normalized.resource.height == 1080 && normalized.resource.kind == sunshine_scene_depth::resource_kind::raw_depth &&
        normalized.proof == sunshine_scene_depth::state_proof::declared &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_present,
      "v2 adapter lost local source/extent/explicit state authority");
    require(nominated_source(view.value, 1, normalized) &&
        normalized.resource.kind == sunshine_scene_depth::resource_kind::display_depth,
      "v2 high-resolution tag lost provider-neutral semantic kind");
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_a), "local source leaked to later evaluation");
    high.lifecycle = UINT32_MAX;
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    // The malformed kind is unusable; it does not revoke the valid raw tag.
    require(!nominated_source(view.value, 1, normalized) && nominated_source(view.value, 0, normalized) &&
        normalized.resource.native == reinterpret_cast<std::uintptr_t>(&native_a),
      "malformed v2 lifecycle inherited the v1 unspecified sentinel or revoked valid raw depth");
    high.lifecycle = 1; call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_is(view.value, evidence_status::source_associated_evaluation), "v2 lifecycle recovery failed");
    // An extent that wraps is malformed (the allocation bounds any other in
    // the capture, S16d).
    high.area.left = UINT32_MAX; call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    high.area.left = 0; call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_is(view.value, evidence_status::source_associated_evaluation), "valid v2 source did not recover");
    mint(token, nullptr);
  }

  // sl_struct.h Option 1: a later structure version only appends fields, so
  // a ResourceTag, Resource and PrecisionInfo of version 1 or later decode by
  // their version-1 fields; nativeFormat is read only from a typed Resource.
  void test_struct_versions() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "struct-version install failed");
    auto view = viewport(5); auto constants = modern_camera(); opaque_token token;
    int native{};
    constexpr std::uint64_t poison = ~0ull;
    struct resource_v2 { abi_v2::resource v1; std::uint64_t tail[4]{poison, poison, poison, poison}; } resource;
    resource.v1 = modern_resource(&native); resource.v1.native_format = 41; // DXGI_FORMAT_R32_FLOAT
    struct precision_v2 { tag_precision v1; std::uint64_t tail[2]{poison, poison}; } precision;
    precision.v1.scale = -.5f; precision.v1.bias = 8.f;
    struct tag_v2 { abi_v2::resource_tag v1; std::uint64_t tail[2]{poison, poison}; } tag;
    tag.v1 = depth_tag_for(resource.v1); tag.v1.base.next = &precision.v1;
    const base_structure *global[]{&view.base};
    sunshine_scene_depth::frame normalized;
    evaluation_snapshot snapshot;
    const auto evaluate = [&] {
      mint(token, nullptr);
      call_v2_constants(constants, token.ref(), view);
      call_v2_tag(view, &tag.v1, 1, nullptr);
      call_v2_evaluate(0, token.ref(), global, 1, nullptr);
      return nominated_snapshot(view.value, snapshot) && nominated_source(view.value, 0, normalized);
    };
    for (const std::uint64_t version : {1ull, 2ull, 7ull}) {
      resource.v1.base.version = precision.v1.base.version = tag.v1.base.version = version;
      require(evaluate() && normalized.resource.native == reinterpret_cast<std::uintptr_t>(&native) &&
          normalized.resource.native_format == 41 && normalized.resource.width == 1920 &&
          normalized.projection.raw_scale == -.5 && normalized.projection.raw_bias == 8 &&
          snapshot.tags[0].value.refusal == tag_refusal::none,
        "a later ResourceTag/Resource/PrecisionInfo version did not decode exactly as version 1");
    }
    resource.v1.base.version = precision.v1.base.version = tag.v1.base.version = 1;
    // The zero-header Resource declares no layout: its nativeFormat reads as 0.
    resource.v1.base = {nullptr, {}, 0};
    require(evaluate() && normalized.resource.native_format == 0, "a zero-header Resource's nativeFormat was read");
    resource.v1.base = {nullptr, resource_guid, 1};
    const auto refused = [&](tag_refusal expected, const char *message) {
      require(!evaluate() && snapshot.tags[0].present && !snapshot.tags[0].supported &&
        snapshot.tags[0].value.refusal == expected && snapshot.tags[0].value.native_format == 0, message);
    };
    resource.v1.base.version = 0; refused(tag_refusal::header_version, "a typed Resource of version 0 was accepted");
    resource.v1.base.version = 1; tag.v1.base.version = 0;
    refused(tag_refusal::header_version, "a ResourceTag of version 0 was accepted");
    tag.v1.base.version = 1; precision.v1.base.version = 0;
    refused(tag_refusal::header_version, "a PrecisionInfo of version 0 was accepted");
    precision.v1.base.version = 2; precision.v1.formula = 2;
    refused(tag_refusal::precision, "a PrecisionInfo v2 with an unknown formula was accepted");
    precision.v1.formula = 1;
    base_structure unknown{nullptr, {0x12345678, 1, 2, {3, 4, 5, 6, 7, 8, 9, 10}}, 1};
    tag.v1.base.next = &unknown; refused(tag_refusal::extension, "an unknown chained structure was accepted");
    tag.v1.base.next = &precision.v1; precision.v1.base.next = &unknown;
    refused(tag_refusal::extension, "a PrecisionInfo with its own extension was accepted");
    precision.v1.base.next = nullptr; resource.v1.base.next = &unknown;
    refused(tag_refusal::extension, "a Resource with an extension was accepted");
    resource.v1.base.next = nullptr; resource.v1.type = 3;
    refused(tag_refusal::resource_type, "a buffer resource was accepted as a texture tag");
    resource.v1.type = 0; tag.v1.lifecycle = 3;
    refused(tag_refusal::lifecycle, "an unknown lifecycle was accepted");
    tag.v1.lifecycle = 1;
    require(evaluate(), "valid versioned tags did not recover");
    // Past its first element, a later ResourceTag version's array stride is unknown.
    abi_v2::resource_tag pair[]{tag.v1, tag.v1};
    pair[1].type = 1; // Motion vectors: a second, unobserved tag.
    const auto tag_pair = [&](std::uint64_t version) {
      for (auto &value : pair) value.base.version = version;
      mint(token, nullptr);
      call_v2_constants(constants, token.ref(), view);
      call_v2_tag(view, pair, 2, nullptr);
      call_v2_evaluate(0, token.ref(), global, 1, nullptr);
      return nominated_snapshot(view.value, snapshot) && nominated_source(view.value, 0, normalized);
    };
    require(tag_pair(1), "a version-1 multi-tag array was refused");
    require(!tag_pair(2) && !snapshot.tags[0].supported,
      "a multi-tag array of a later ResourceTag version was read at the version-1 stride");
  }

  // S12: eUnknown (8) is a texture identity only where the loaded release's
  // header declares it, from 2.7.2; 2.4.x ends ResourceType at eHostFence
  // (eCount 8), and no surveyed header decides the releases between.
  void test_unknown_resource_type_by_release() {
    struct expectation { versioning::number version; bool accepted; };
    const expectation cases[]{{{2, 4, 0, 0}, false}, {{2, 6, 10, 0}, false}, {{2, 7, 2, 0}, true}, {testing::abi::v2_12_0, true}};
    for (const auto &c : cases) {
      fixture cleanup;
      source_setting = "1"; initialize(nullptr);
      require(testing::install(c.version, presentation_v2_targets), "eUnknown release install failed");
      auto view = viewport(5); auto constants = modern_camera(); opaque_token token;
      int native{};
      auto resource = modern_resource(&native); resource.type = 8;
      auto tag = depth_tag_for(resource);
      const base_structure *global[]{&view.base};
      mint(token, nullptr);
      call_v2_constants(constants, token.ref(), view);
      call_v2_tag(view, &tag, 1, nullptr);
      call_v2_evaluate(0, token.ref(), global, 1, nullptr);
      evaluation_snapshot snapshot; sunshine_scene_depth::frame normalized;
      require(nominated_snapshot(view.value, snapshot) && snapshot.tags[0].present, "eUnknown depth tag was not observed");
      const bool nominated = nominated_source(view.value, 0, normalized);
      require(snapshot.tags[0].supported == c.accepted && nominated == c.accepted &&
          (c.accepted || snapshot.tags[0].value.refusal == tag_refusal::resource_type),
        c.accepted ? "a release that declares eUnknown refused it" : "eUnknown was accepted where the release's header does not declare it");
    }
  }

  void test_rr_and_linear_depth_source() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "RR/linear source install failed");
    auto view = viewport(3); auto constants = modern_camera(); opaque_token token;
    constants.common.jitter_offset[0] = .25f;
    constants.common.jitter_offset[1] = -.375f;
    int native{}, commands{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource); tag.type = 49;
    const base_structure *local[]{&view.base, &tag.base};
    const base_structure *global[]{&view.base};
    mint(token, nullptr);
    call_v2_constants(constants, token.ref(), view);
    sunshine_scene_depth::frame normalized;
    evaluation_snapshot snapshot;
    for (const auto feature : {0u, 1001u}) {
      SetLastError(incoming_error);
      require(call_v2_evaluate(feature, token.ref(), local, 2, &commands) == 0 &&
          GetLastError() == outgoing_error && evaluate_call.incoming_error == incoming_error && evaluate_call.a == feature,
        "SR/RR linear evaluation changed native forwarding");
      require(nominated_snapshot(view.value, snapshot) && snapshot.feature == feature &&
          snapshot.status == evidence_status::source_associated_evaluation &&
          nominated_source(view.value, 2, normalized) && normalized.projection.supplied &&
          normalized.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
          normalized.projection.direction_supplied && !normalized.projection.reversed &&
          normalized.projection.raw_scale == 1 && normalized.projection.raw_bias == 0 &&
          normalized.jitter.supplied && normalized.jitter.width == 1920 && normalized.jitter.x == .25f,
        "SR/RR linear input was dropped or treated as reversed device depth");
    }
    tag_precision precision; precision.scale = -.5f; precision.bias = 8.f; tag.base.next = &precision;
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized) && normalized.projection.reversed &&
        normalized.projection.raw_scale == -.5 && normalized.projection.raw_bias == 8,
      "linear PrecisionInfo transform/direction was lost");
    tag.base.next = nullptr;
    auto no_camera = viewport(4); const base_structure *without_constants[]{&no_camera.base, &tag.base};
    call_v2_evaluate(1001, token.ref(), without_constants, 2, nullptr);
    require(nominated_source(no_camera.value, 2, normalized) && !normalized.projection.supplied &&
        normalized.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
        normalized.projection.direction_supplied && !normalized.projection.reversed,
      "explicit linear-distance input incorrectly required projection constants");
    // An UntilEvaluate tag needs its call's command list (sl_core_api.h; a
    // command-less one has no copy point, the matrix's S2).
    tag.lifecycle = 2;
    call_v2_tag(view, &tag, 1, nullptr); call_v2_evaluate(1001, token.ref(), global, 1, nullptr);
    require(!nominated_source(view.value, 2, normalized),
      "a command-less linear UntilEvaluate tag was copied at its evaluate");
    call_v2_tag(view, &tag, 1, &commands); call_v2_evaluate(1001, token.ref(), global, 1, nullptr);
    require(nominated_source(view.value, 2, normalized) &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_evaluation,
      "linear UntilEvaluate was not available at its synchronous boundary");
    call_v2_evaluate(1001, token.ref(), global, 1, nullptr);
    require(!nominated_source(view.value, 2, normalized), "linear UntilEvaluate survived its SDK return");
    // A local OnlyValidNow input is current at its own evaluate (sl_core_api.h
    // slEvaluateFeature, the matrix's S4); a tagged one is gone at a later
    // evaluate (S3).
    tag.lifecycle = 0; call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized) &&
        normalized.valid_until == sunshine_scene_depth::lifetime::at_call,
      "a linear OnlyValidNow local input was not current at its own evaluate");
    call_v2_tag(view, &tag, 1, &commands); call_v2_evaluate(1001, token.ref(), global, 1, nullptr);
    require(!nominated_source(view.value, 2, normalized), "linear OnlyValidNow was consumed at a later evaluation");
    tag.lifecycle = 1; tag.base.next = &precision; precision.scale = 0;
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(!nominated_source(view.value, 2, normalized), "malformed linear precision reached capture");
    tag.base.next = nullptr; call_v2_constants(constants, token.ref(), view);
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized), "RR did not recover with fresh valid inputs");
    result_v2 = -7; call_v2_evaluate(1001, token.ref(), local, 2, nullptr); result_v2 = 0;
    require(!nominated_source(view.value, 2, normalized), "failed RR evaluation retained source authority");
    call_v2_constants(constants, token.ref(), view);
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{}; call_feature_function(1000, "slDLSSGSetOptions", function);
    require(function != nullptr, "RR/linear FG options missing");
    auto set_options = reinterpret_cast<fg_function>(function); fg_options options; set_options(view, options);
    // Enabled FG owns the viewport only for the frames FG supplied
    // (depth_capture::frame_generation_scope_live): this fixture's capture
    // owner is inactive, so no FG copy is ever accepted, and RR of the same
    // viewport supplies depth right after FG On, a second later, with
    // repeated options and after FG Off then On (Assassin's Creed Shadows
    // 10-22 and 10-24).
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized) && !normalized.frame_generation_input,
      "FG On without an FG copy blocked same-viewport RR depth");
    constexpr std::uint64_t fg_scope_ms = 1000;
    testing::age_frame_generation(fg_scope_ms + 1);
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized) && !normalized.frame_generation_input,
      "A silent FG scope kept blocking same-viewport RR depth");
    set_options(view, options); call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized), "Repeated FG On options blocked same-viewport RR depth");
    options.mode = 0; set_options(view, options); options.mode = 1; set_options(view, options);
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized), "A new FG On without an FG copy owned its viewport");
    tag.lifecycle = 0;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(nominated_source(view.value, 2, normalized) && normalized.frame_generation_input &&
        normalized.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
        normalized.valid_until == sunshine_scene_depth::lifetime::at_call,
      "FG linear OnlyValidNow input did not use its own synchronous boundary");
    options.mode = 0; set_options(view, options);
    tag.lifecycle = 1; call_v2_constants(constants, token.ref(), view);
    call_v2_evaluate(1001, token.ref(), local, 2, nullptr);
    require(nominated_source(view.value, 2, normalized) && !normalized.frame_generation_input,
      "FG Off did not restore RR source admission");
    mint(token, nullptr);
  }

  void test_mixed_clear_and_linear_depth() {
    for (const bool framed : {false, true}) for (const bool reverse_order : {false, true}) {
      fixture cleanup;
      source_setting = "1"; initialize(nullptr);
      require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Mixed clear source install failed");
      auto view = viewport(3); auto constants = modern_camera(); opaque_token old_token, token;
      int native{}, linear_native{}, commands{};
      auto resource = modern_resource(&native), linear_resource = modern_resource(&linear_native);
      auto raw = depth_tag_for(resource), high = raw, linear = depth_tag_for(linear_resource);
      high.type = 48; linear.type = 49;
      auto clear_raw = raw; clear_raw.resource_ptr = nullptr;
      const base_structure *inputs[]{&view.base};
      mint(old_token, nullptr); call_v2_constants(constants, old_token.ref(), view);
      const abi_v2::resource_tag old_tags[]{raw, high};
      call_v2_tag(view, old_tags, 2, &commands);
      call_v2_evaluate(0, old_token.ref(), inputs, 1, &commands);
      sunshine_scene_depth::frame normalized;
      require(nominated_source(view.value, 0, normalized) && normalized.projection.supplied,
        "Mixed clear fixture did not establish its old camera/depth tuple");
      const auto old_revision = normalized.observation_revision;
      const auto publish = [&](const abi_v2::resource_tag *tags, unsigned count) {
        return framed ? call_v2_framed_tag(token.ref(), view, tags, count, &commands) :
          call_v2_tag(view, tags, count, &commands);
      };
      abi_v2::resource_tag mixed[]{clear_raw, linear};
      if (reverse_order) std::swap(mixed[0], mixed[1]);
      // A clear is observed, not lost: the earlier evaluation keeps its depth
      // and only later evaluations see the cleared kind.
      mint(token, nullptr);
      require(publish(mixed, 2) == 0 && depth_observation_revision() == old_revision &&
          nominated_source(view.value, 0, normalized) && normalized.observation_revision == old_revision,
        "Mixed clear revoked depth already evaluated for an earlier frame");
      const auto mixed_revision = depth_observation_revision();
      for (unsigned repeat = 0; repeat != 3; ++repeat) {
        if (repeat) { mint(token, nullptr); require(publish(mixed, 2) == 0, "Repeated mixed tag failed"); }
        require(call_v2_evaluate(1001, token.ref(), inputs, 1, &commands) == 0 &&
            nominated_source(view.value, 2, normalized) && !normalized.projection.supplied &&
            normalized.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
            normalized.observation_revision == mixed_revision && depth_observation_revision() == mixed_revision &&
            !nominated_source(view.value, 0, normalized),
          "Null alternative poisoned fresh linear depth, changed its revision, or kept the cleared raw depth");
      }
      auto clear_linear = linear; clear_linear.resource_ptr = nullptr;
      const abi_v2::resource_tag full_clear[]{clear_raw, clear_linear};
      require(publish(full_clear, 2) == 0 && depth_observation_revision() == mixed_revision,
        "Full depth clear was recorded as a lost observation");
      call_v2_evaluate(1001, token.ref(), inputs, 1, &commands);
      require(!nominated_source(view.value, 2, normalized), "Full clear retained a linear source");

      mint(token, nullptr); publish(mixed, 2); call_v2_evaluate(1001, token.ref(), inputs, 1, &commands);
      require(nominated_source(view.value, 2, normalized), "Fresh mixed tags did not recover after full clear");
      result_v2 = -7;
      require(publish(mixed, 2) == -7, "Failed mixed SDK call changed its result");
      result_v2 = 0;
      call_v2_evaluate(1001, token.ref(), inputs, 1, &commands);
      require(!nominated_source(view.value, 2, normalized), "Failed mixed SDK call published fresh depth");

      // The original SDK can trigger another observed loss before returning.
      // A subsequent clear may advance only its own revision, never heal that gap.
      mint(token, nullptr); publish(&raw, 1);
      const auto before_interruption = depth_observation_revision();
      lose_during_tag_original = true;
      require(publish(mixed, 2) == 0 && depth_observation_revision() == before_interruption + 1,
        "Mixed clear did not preserve the intervening loss");
      call_v2_evaluate(1001, token.ref(), inputs, 1, &commands);
      require(!nominated_source(view.value, 2, normalized), "Mixed clear healed an intervening observation loss");
    }
  }

  void test_source_without_frame_tag_export() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    auto exports = tracked_v2_targets;
    exports.tag_for_frame = nullptr;
    require(testing::install(testing::abi::v2_7_30, exports), "v2 global-only source install failed");
    auto view = viewport(3);
    auto constants = modern_camera();
    constants.base.version = 1; // Older typed SDKs do not supply the optional constants tail.
    opaque_token token;
    int native{}, commands{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    const base_structure *inputs[]{&view.base};

    const auto observe = [&] {
      SetLastError(incoming_error);
      require(call_v2_constants(constants, token.ref(), view) == 0 &&
          constants_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
        "v2 global-only constants changed return value or LastError");
      SetLastError(incoming_error);
      require(call_v2_tag(view, &tag, 1, &commands) == 0 &&
          tag_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
        "v2 global-only tag changed return value or LastError");
      SetLastError(incoming_error);
      require(call_v2_evaluate(0, token.ref(), inputs, 1, &commands) == 0 &&
          evaluate_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
        "v2 global-only evaluation changed return value or LastError");
    };
    observe();
    require(nominated_is(view.value, evidence_status::untracked_frame),
      "missing frame-tag export promoted an untracked token to a frame");
    mint(token, nullptr);
    observe();
    sunshine_scene_depth::frame normalized;
    require(nominated_is(view.value, evidence_status::source_associated_evaluation) &&
        nominated_source(view.value, 0, normalized) && normalized.resource.native == address &&
        normalized.resource.width == 1920 && normalized.resource.height == 1080 &&
        normalized.proof == sunshine_scene_depth::state_proof::declared &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_present,
      "global tags and observed tokens failed to supply production depth without frame-tag export");
    require(framed_tag_call.calls == 0 && feature_function_call.calls == 0,
      "global-only source depended on unavailable optional exports");
    result_v2 = -7;
    SetLastError(incoming_error);
    require(call_v2_evaluate(0, token.ref(), inputs, 1, &commands) == -7 &&
        evaluate_call.incoming_error == incoming_error && GetLastError() == outgoing_error &&
        !nominated_source(view.value, 0, normalized),
      "failed global-only evaluation changed forwarding or retained usable depth");
    result_v2 = 0;
    mint(token, nullptr);
    observe();
    require(nominated_source(view.value, 0, normalized),
      "global-only source did not recover after a failed evaluation");
    mint(token, nullptr);
  }

  void test_jitter_frame_and_render_domain(bool modern) {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "jitter source hook install failed");
    auto view = viewport(9);
    int native{};
    abi_v1::resource old_resource{}; old_resource.native = &native;
    auto resource = modern_resource(&native); resource.width = 2228; resource.height = 1256;
    auto tag = depth_tag_for(resource); tag.area = {0, 0, 2228, 1253};
    const extent area = tag.area;
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
    require(nominated_source(view.value, 0, first) && first.jitter.supplied &&
        first.jitter.x == .25f && first.jitter.y == -.375f && first.jitter.width == 2228 && first.jitter.height == 1253,
      "Depth jitter borrowed newer camera offsets or allocation padding");
    evaluate(2);
    require(nominated_source(view.value, 0, current) && current.jitter.supplied &&
        current.jitter.x == -.125f && current.jitter.y == .5f && first.jitter.x == .25f,
      "Per-frame jitter snapshot was not immutable");
    constants(3, std::numeric_limits<float>::quiet_NaN(), .5f); evaluate(3);
    require(nominated_source(view.value, 0, current) && current.projection.supplied && !current.jitter.supplied,
      "Nonfinite jitter rejected usable depth or reused previous correction");
    if (modern) mint(tokens[3], nullptr); // No constants for this frame.
    evaluate(4);
    require(nominated_source(view.value, 0, current) && !current.jitter.supplied,
      "Camera-free depth inherited another frame's jitter");
  }

  void test_high_resolution_jitter_domain() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "high-res jitter hook install failed");
    auto view = viewport(6); auto constants = modern_camera(); opaque_token token;
    constants.common.jitter_offset[0] = .25f; constants.common.jitter_offset[1] = -.375f;
    int raw_native{}, high_native{};
    auto raw_resource = modern_resource(&raw_native), high_resource = modern_resource(&high_native);
    raw_resource.width = 2228; raw_resource.height = 1256;
    high_resource.width = 3840; high_resource.height = 2160;
    auto raw = depth_tag_for(raw_resource), high = depth_tag_for(high_resource);
    raw.area = {0, 0, 2228, 1253}; high.area = {0, 0, 3840, 2160}; high.type = 48;
    mint(token, nullptr); call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &raw, 1, nullptr); // An old global raw tag cannot establish the high-res frame's render size.
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    const base_structure *inputs[]{&view.base};
    const auto evaluate = [&] { call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); };
    sunshine_scene_depth::frame value;
    evaluate();
    require(nominated_source(view.value, 1, value) && !value.jitter.supplied,
      "High-res jitter used display dimensions or an uncorrelated global render extent");
    call_v2_framed_tag(token.ref(), view, &raw, 1, nullptr); evaluate();
    require(nominated_source(view.value, 1, value) && value.jitter.supplied &&
        value.jitter.width == 2228 && value.jitter.height == 1253 && value.jitter.x == .25f && value.jitter.y == -.375f,
      "High-res depth did not retain same-frame raw render-pixel jitter units");
    // The next frame cannot reuse the previous frame's render-domain proof.
    mint(token, nullptr); call_v2_constants(constants, token.ref(), view);
    call_v2_framed_tag(token.ref(), view, &high, 1, nullptr);
    evaluate();
    require(nominated_source(view.value, 1, value) && !value.jitter.supplied,
      "High-res depth borrowed a previous frame's render domain");
  }

  void test_source_without_projection() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "projection-independent source hook install failed");
    auto view = viewport(9); auto constants = modern_camera(); opaque_token token;
    int native{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
    const base_structure *inputs[]{&view.base};
    sunshine_scene_depth::frame normalized;
    evaluation_snapshot observed;
    const auto evaluate = [&] { call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); };

    mint(token, nullptr);
    constants.common.jitter_offset[0] = .25f; constants.common.jitter_offset[1] = -.375f;
    constants.common.clip_to_camera_view.m[0][0] *= 2; // Stale inverse must remain rejected.
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_snapshot(view.value, observed) && observed.status == evidence_status::invalid_camera,
      "source normalization upgraded invalid camera evidence");
    require(nominated_source(view.value, 0, normalized) && normalized.resource.native == address &&
        !normalized.projection.supplied && normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == 0.0 &&
        normalized.projection.direction_supplied && normalized.projection.reversed && normalized.jitter.supplied &&
        normalized.jitter.x == .25f && normalized.jitter.y == -.375f,
      "valid depth source was rejected with its invalid projection, or lost the independent decoded direction");

    // A new observed evaluation token has no camera. Never inherit the prior
    // token's otherwise well-formed depth direction or unusable projection.
    mint(token, nullptr); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_source(view.value, 0, normalized) && normalized.resource.native == address &&
        !normalized.projection.supplied && !normalized.projection.direction_supplied,
      "missing constants rejected an authoritative source or inherited unrelated camera metadata");
    require(nominated_is(view.value, evidence_status::missing_constants),
      "direct source admission changed the existing stricter nomination query");

    // Direction is a fixed viewport convention: a token without its own camera
    // takes direction, never projection or jitter, from the viewport's recent
    // valid camera (the invalid one above supplied none).
    {
      auto valid = modern_camera(); valid.common.jitter_offset[0] = .25f;
      mint(token, nullptr); call_v2_constants(valid, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); evaluate();
      sunshine_scene_depth::frame framed;
      require(nominated_source(view.value, 0, framed) && framed.projection.supplied &&
          framed.projection.direction_supplied && framed.jitter.supplied, "valid camera did not calibrate its own frame");
      mint(token, nullptr); call_v2_tag(view, &tag, 1, nullptr); evaluate();
      require(nominated_source(view.value, 0, normalized) && !normalized.projection.supplied &&
          normalized.projection.direction_supplied && normalized.projection.reversed == framed.projection.reversed &&
          !normalized.jitter.supplied,
        "a camera-less frame lost the viewport's direction or borrowed an unsteady projection or jitter");
      // Coefficients the viewport's camera repeated in eight consecutive valid
      // frames are camera constants: camera-less depth takes them, never jitter.
      // Engines that rebuild the matrix every frame differ in the last bits;
      // that is still the same camera.
      auto rebuilt = valid;
      rebuilt.common.camera_view_to_clip.m[3][2] = std::nextafter(valid.common.camera_view_to_clip.m[3][2], 1.f);
      for (unsigned i = 0; i != 8; ++i) {
        mint(token, nullptr); call_v2_constants(i % 2 ? valid : rebuilt, token.ref(), view);
      }
      mint(token, nullptr); call_v2_tag(view, &tag, 1, nullptr); evaluate();
      require(nominated_source(view.value, 0, normalized) && normalized.projection.supplied &&
          normalized.projection.depth_offset == framed.projection.depth_offset &&
          normalized.projection.depth_scale == framed.projection.depth_scale &&
          normalized.projection.direction_supplied && !normalized.jitter.supplied,
        "a camera-less frame did not take the viewport's steady projection, or borrowed jitter");
      // A changed camera restarts the count.
      auto zoomed = valid; set_camera_near(zoomed.common, valid.common.camera_near * 2.f);
      mint(token, nullptr); call_v2_constants(zoomed, token.ref(), view);
      mint(token, nullptr); call_v2_tag(view, &tag, 1, nullptr); evaluate();
      require(nominated_source(view.value, 0, normalized) && !normalized.projection.supplied,
        "a changed camera kept the previous steady projection");
    }

    result_v2 = -1; evaluate(); result_v2 = 0;
    require(!nominated_source(view.value, 0, normalized), "failed evaluation exposed a normalized source");
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_source(view.value, 0, normalized), "source did not recover after failed evaluation");
    testing::lose_observation();
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_source(view.value, 0, normalized), "fresh camera-free source did not recover from observation loss");

    tag.lifecycle = 0; call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(!nominated_source(view.value, 0, normalized), "OnlyValidNow source survived until a later evaluation");
    tag.lifecycle = 1; tag.area.left = UINT32_MAX;
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(!nominated_source(view.value, 0, normalized), "invalid depth extent acquired source authority");
    tag.area.left = 0;
    constants = modern_camera(); constants.reset = 1;
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied,
      "valid reset-frame depth lost its current projection/source");

    view = viewport(10); mint(token, nullptr); evaluate();
    require(!nominated_source(view.value, 0, normalized), "missing depth tag acquired camera-free source authority");
    call_v2_tag(view, &tag, 1, nullptr); evaluate();
    require(nominated_source(view.value, 0, normalized), "valid source without constants was not admitted");
    mint(token, nullptr);
  }

  void test_until_evaluate_lifetime() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "UntilEvaluate hook install failed");
    auto view = viewport(8); auto constants = modern_camera(); opaque_token token;
    // An UntilEvaluate tag call needs a command list (sl_core_api.h; S2).
    int native{}, commands{}; auto resource = modern_resource(&native); auto tag = depth_tag_for(resource);
    tag.lifecycle = 2;
    const base_structure *inputs[]{&view.base};
    const base_structure *local_inputs[]{&view.base, &tag.base};
    evaluation_snapshot observed;
    const auto evaluated = [&] {
      require(nominated_snapshot(view.value, observed) && nomination_succeeded(view.value) &&
          observed.status == evidence_status::source_associated_evaluation && observed.tags[0].supported &&
          observed.tags[0].value.lifecycle == 2, "valid UntilEvaluate tag rejected at synchronous evaluation boundary");
      sunshine_scene_depth::frame normalized;
      require(nominated_source(view.value, 0, normalized) &&
          normalized.valid_until == sunshine_scene_depth::lifetime::until_evaluation &&
          normalized.projection.depth_offset == 0.0 && normalized.projection.depth_scale == .5,
        "UntilEvaluate adapter changed projection or lifetime while normalizing");
    };
    mint(token, nullptr);
    call_v2_constants(constants, token.ref(), view); call_v2_tag(view, &tag, 1, &commands);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "global UntilEvaluate tag was reused after its evaluation returned");
    // A fresh exact-frame tag authorizes this evaluation only, independently of
    // the consumed global tag and without inventing native-present correlation.
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "frame UntilEvaluate tag was reused after return");
    call_v2_evaluate(0, token.ref(), local_inputs, 2, nullptr); evaluated();
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, observed) && observed.status == evidence_status::missing_depth,
      "local UntilEvaluate tag leaked to a later call");
    tag.area.left = UINT32_MAX;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    {
      sunshine_scene_depth::frame normalized;
      require(!nominated_source(view.value, 0, normalized),
        "UntilEvaluate admission bypassed malformed extent rejection");
    }
    tag.area.left = 0;
    call_v2_constants(constants, token.ref(), view); call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
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
    require(nominated_source(1, 0, initial) && !initial.feedback.reset && initial.feedback.revision != 0,
      "source revision missing from normalized depth");
    evaluate(2, 1);
    sunshine_scene_depth::frame ordinary;
    require(nominated_source(1, 0, ordinary) && !ordinary.feedback.reset &&
        ordinary.feedback.revision == initial.feedback.revision && ordinary.source_frame_numeric == 2 &&
        ordinary.source_id == initial.source_id && ordinary.viewport == initial.viewport,
      "ordinary input changed reset revision or source identity");
    evaluate(1, 2);
    sunshine_scene_depth::frame other;
    require(nominated_source(2, 0, other) && other.viewport == 2 && !other.feedback.reset &&
        other.feedback.revision == initial.feedback.revision,
      "viewport feedback acquired an unrelated reset");
    evaluate(3, 1, true);
    evaluation_snapshot reset;
    require(nominated_snapshot(1, reset) && reset.feedback.reset &&
        reset.feedback.revision > ordinary.feedback.revision && nominated_source(1, 0, other) &&
        other.feedback.reset && other.feedback.revision == reset.feedback.revision &&
        other.observation_revision == initial.observation_revision &&
        other.observation_revision == depth_observation_revision() && other.projection.supplied &&
        other.projection.depth_offset == initial.projection.depth_offset &&
        other.projection.depth_scale == initial.projection.depth_scale && other.source_frame_numeric == 3 &&
        other.source_id == initial.source_id,
      "reset failed to start a new history while admitting its fresh depth and projection");
    require(nominated_source(2, 0, other) && !other.feedback.reset &&
        other.feedback.revision == initial.feedback.revision,
      "reset of one viewport revoked or restarted another viewport's history");
    evaluate(4, 1);
    sunshine_scene_depth::frame recovered;
    require(nominated_source(1, 0, recovered) && !recovered.feedback.reset &&
        recovered.feedback.revision == reset.feedback.revision && recovered.feedback.revision == initial.feedback.revision + 1 &&
        recovered.source_id == initial.source_id && recovered.viewport == initial.viewport,
      "fresh post-reset input lost revision or logical source identity");
    require(!initial.feedback.reset && initial.feedback.revision == ordinary.feedback.revision &&
        ordinary.feedback.revision < recovered.feedback.revision,
      "later reset mutated a frozen earlier frame");
    // A transient loss (a busy lock, an SDK failure) cannot hide a camera
    // reset: it moves the observation revision but keeps temporal history.
    testing::lose_observation();
    evaluate(5, 1);
    sunshine_scene_depth::frame after_loss;
    require(nominated_source(1, 0, after_loss) && !after_loss.feedback.reset &&
        after_loss.feedback.revision == recovered.feedback.revision &&
        after_loss.observation_revision == depth_observation_revision() &&
        after_loss.observation_revision != recovered.observation_revision,
      "a transient observation loss restarted temporal history");
  }

  void test_scene_feedback_inflight() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "in-flight feedback install failed");
    int native{}; abi_v1::resource resource{}; resource.native = &native;
    auto constants = old_camera();
    call_v1_constants(constants, 1, 1); call_v1_tag(&resource, 0, 1, nullptr);
    const auto loss_before = depth_observation_revision();
    block_original = true;
    std::thread pending([] { testing::arm_nomination_tap(true); call_v1_evaluate(nullptr, 0, 1, 1); });
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
    // The next frame's reset is observed metadata, not lost evidence: the
    // in-flight frame keeps its own depth and its pre-reset history.
    require(nominated_snapshot(1, first) && first.frame.numeric == 1 && !first.feedback.reset &&
        first.feedback.revision != 0 && depth_observation_revision() == loss_before &&
        first.status == evidence_status::source_associated_evaluation &&
        nominated_source(1, 0, normalized) && normalized.source_frame_numeric == 1 && !normalized.feedback.reset,
      "a later frame's reset revoked in-flight depth or leaked into its history");
    const auto original_revision = first.feedback.revision;
    call_v1_evaluate(nullptr, 0, 2, 1);
    evaluation_snapshot reset;
    require(nominated_snapshot(1, reset) && reset.frame.numeric == 2 && reset.feedback.reset &&
        reset.feedback.revision == original_revision + 1 && nominated_source(1, 0, normalized) &&
        normalized.feedback.reset && normalized.projection.supplied && normalized.source_frame_numeric == 2 &&
        normalized.feedback.revision == reset.feedback.revision,
      "reset frame lost its still-tagged depth or did not start a new history");
    constants.reset = 0;
    call_v1_constants(constants, 3, 1); call_v1_tag(&resource, 0, 1, nullptr); call_v1_evaluate(nullptr, 0, 3, 1);
    require(nominated_source(1, 0, normalized) && normalized.source_frame_numeric == 3 &&
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
    require(!nominated_source(view.value, 0, normalized),
      "reset revision repaired unrelated observation loss during the constants call");
    constants(); evaluate();
    require(nominated_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied,
      "fresh reset constants did not recover after the interrupted reset call");
  }

  void test_depth_observation_loss_diagnostics(bool modern) {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(source_enabled(), "Loss journal enabled the optional command/content probe");
    require(testing::install(modern ? testing::abi::v2_7_30 : testing::abi::v1_1_1,
      modern ? tracked_v2_targets : v1_targets), "Loss journal hook install failed");
    const auto view = viewport(59);
    int native{};
    const auto address = reinterpret_cast<std::uintptr_t>(&native);
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
      require(nominated_source(view.value, 0, normalized) &&
        normalized.resource.native == address && normalized.observation_revision == depth_observation_revision() &&
        normalized.projection.supplied && normalized.projection.depth_offset == 0. &&
        normalized.projection.depth_scale == .5 && normalized.projection.reversed && !normalized.feedback.reset,
        "Loss diagnostics changed fresh depth/projection recovery");
      return normalized;
    };
    const auto event_for = [&](std::uint64_t revision, loss_diagnostics::reason cause, DWORD thread) {
      loss_diagnostics::event event;
      require(query_depth_observation_loss(revision, event) && event.revision == revision && event.cause == cause &&
        event.tick && event.tick <= fixture_clock().now() && event.thread_id == thread && event.line && event.site && *event.site,
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
      require(!nominated_source(view.value, 0, normalized),
        "Diagnostic recording repaired the source invalidation it only describes");
    };

    const auto first = recover();
    old.reset = current.reset = 1;
    next_frame();
    const auto before_reset = depth_observation_revision();
    preserves_last_error(constants, constants_call);
    require(depth_observation_revision() == before_reset, "Camera reset was recorded as a lost observation");
    recover();
    require(first.observation_revision == before_reset && !first.feedback.reset &&
      first.projection.depth_scale == .5, "Later diagnostic activity mutated the earlier normalized value");

    next_frame();
    old.common.clip_to_camera_view.m[0][0] = current.common.clip_to_camera_view.m[0][0] = 2.f;
    const auto before_invalid = depth_observation_revision();
    preserves_last_error(constants, constants_call);
    require(depth_observation_revision() == before_invalid, "Invalid camera constants were recorded as a lost observation");
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
    const auto deadline = std::chrono::steady_clock::now() + finish_bound();
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
    recover();

    const auto recovered_revision = depth_observation_revision();
    for (const auto &saved : {failed, busy}) still_recorded(saved);
    loss_diagnostics::event missing;
    require(!query_depth_observation_loss(0, missing) &&
      !query_depth_observation_loss(recovered_revision + 1, missing) &&
      !query_depth_observation_loss(UINT64_MAX, missing) && depth_observation_revision() == recovered_revision,
      "Missing diagnostic revision borrowed another event or changed source authority");
    require(source_enabled(), "Loss diagnostics enabled the optional full probe during recovery");
  }

  // The checks of kept behaviour from the StreamlineCameraProbe research
  // ledger's tests, which ran in the ledger's configuration and read the
  // evaluation through query_evaluation, which only the ledger served. Each
  // test_*_shipping repeats its ledger test's calls in the shipping
  // configuration and reads the same outcome through the testing seams,
  // with the same expected values: the evaluation snapshot source
  // nomination received before the game's call, and the depth it normalized
  // there, through the nomination tap (nominated_snapshot, nominated_source,
  // nomination_succeeded), the forwarded results and arguments, the counters
  // and the observation revision.
  // Where a ledger test read an earlier frame's camera record through a
  // later query, the port evaluates that frame again. Checks of the
  // ledger's own logic (the selected-depth match, active-tag revalidation,
  // age limits, ambiguity, the value copy, command and content association,
  // the StreamlineCameraProbe opt-in and its scaling colour tags) are not
  // ported; they leave with the ledger.
  void expect_nominated(std::uint32_t viewport, evidence_status expected, const char *message) {
    evaluation_snapshot value;
    const bool found = nominated_snapshot(viewport, value);
    if (!found || value.status != expected)
      std::fprintf(stderr, "nominated evaluation expected %s, got %s\n", name(expected), found ? name(value.status) : "none");
    require(found && value.status == expected, message);
  }
  // test_diagnostic_opt_in: with source selection rolled back
  // (StreamlineDepthSource=0) and the probe off, nothing installs or
  // observes, and a reinitialized disabled probe's retained hooks pass
  // through without observing.
  void test_source_rollback_shipping() {
    fixture cleanup;
    source_setting = "0";
    initialize(nullptr);
    require(!testing::install(testing::abi::v1_1_1, v1_targets), "Disabled probe installed middleware hooks");
    const auto before = testing::counts();
    sunshine_game3d::hook_host::poll();
    const auto after = testing::counts();
    unchanged(before, after, "Disabled probe observed middleware calls");
    require(before.dropped == after.dropped && before.invalid == after.invalid,
      "Disabled tracking changed loss counters");
    source_setting = "1";
    initialize(nullptr);
    require(testing::install(testing::abi::v1_1_1, v1_targets), "Explicit diagnostic opt-in did not enable hooks");
    const auto constants = old_camera();
    require(call_v1_constants(constants, 1, 1) && testing::counts().constants == 1, "Opted-in middleware was not observed");
    shutdown();
    source_setting = "0";
    initialize(nullptr);
    const auto disabled = testing::counts();
    require(call_v1_constants(constants, 2, 1), "Disabling retained hooks broke pass-through");
    unchanged(disabled, testing::counts(), "Reinitialized disabled probe still observed retained hooks");
  }
  // test_evaluation_v1.
  void test_evaluation_v1_shipping() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 evidence install failed");
    int native{}, commands{};
    abi_v1::resource resource{};
    resource.native = &native;
    auto constants = old_camera();
    call_v1_constants(constants, 10, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    evaluation_snapshot captured;
    require(!nominated_snapshot(4, captured), "constants/tags alone fabricated evaluation evidence");
    auto next = constants;
    set_camera_near(next.common, 1.f);
    call_v1_constants(next, 11, 4);
    call_v1_evaluate(&commands, 0, 10, 4);
    require(nominated_snapshot(4, captured) && captured.status == evidence_status::source_associated_evaluation,
      "v1 exact constants frame not retained across frames in flight");
    require(captured.frame_correlated && nomination_succeeded(4) && captured.frame.has_numeric &&
        captured.frame.numeric == 10 && captured.frame.kind == frame_identity_kind::v1_numeric &&
        captured.camera.near_plane == .5f &&
        captured.projection.valid() && captured.projection.depth_scale == .5 &&
        captured.command_buffer == reinterpret_cast<std::uintptr_t>(&commands) &&
        captured.tags[0].scope == tag_scope::active_global &&
        captured.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native),
      "v1 snapshot did not preserve camera/tag/evaluation identities");
    call_v1_evaluate(nullptr, 0, 12, 4);
    expect_nominated(4, evidence_status::missing_constants, "latest camera substituted for missing exact frame");
    call_v1_evaluate(nullptr, 0, 11, 4);
    require(nominated_snapshot(4, captured) && captured.status == evidence_status::source_associated_evaluation &&
        captured.camera.near_plane == 1.f && captured.projection.depth_scale == 1.,
      "older unsuccessful correlation poisoned a later complete tuple");
    result_v1 = false;
    call_v1_evaluate(nullptr, 0, 11, 4);
    result_v1 = true;
    call_v1_constants(constants, 13, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 13, 4);
    expect_nominated(4, evidence_status::source_associated_evaluation, "fresh complete observations did not recover");
    testing::lose_observation();
    call_v1_evaluate(nullptr, 0, 13, 4);
    expect_nominated(4, evidence_status::observation_lost, "evaluation reused inputs from before observation loss");
    call_v1_constants(constants, 14, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 14, 4);
    sunshine_scene_depth::frame before_reset;
    require(nominated_source(4, 0, before_reset) &&
        before_reset.observation_revision == depth_observation_revision(),
      "v1 normalized depth omitted its source observation revision");
    // A later frame's reset or invalid camera is observed metadata; it never
    // revokes the depth already evaluated for an earlier frame.
    constants.reset = 1;
    call_v1_constants(constants, 15, 4);
    require(depth_observation_revision() == before_reset.observation_revision,
      "v1 camera reset was recorded as a lost observation");
    call_v1_evaluate(nullptr, 0, 14, 4);
    expect_nominated(4, evidence_status::source_associated_evaluation, "camera reset revoked earlier evidence");
    constants.reset = 0;
    call_v1_constants(constants, 16, 4);
    call_v1_tag(&resource, 0, 4, nullptr);
    call_v1_evaluate(nullptr, 0, 16, 4);
    constants.common.clip_to_camera_view.m[0][0] = 2;
    call_v1_constants(constants, 17, 4);
    call_v1_evaluate(nullptr, 0, 16, 4);
    expect_nominated(4, evidence_status::source_associated_evaluation, "invalid projection revoked earlier evidence");
  }
  // test_evaluation_immutable_and_ordered.
  void test_evaluation_order_shipping() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 ordered evidence install failed");
    int native{};
    abi_v1::resource resource{};
    resource.native = &native;
    auto first = old_camera(), second = old_camera();
    set_camera_near(second.common, 1.f);
    call_v1_constants(first, 1, 8);
    call_v1_constants(second, 2, 8);
    call_v1_tag(&resource, 0, 8, nullptr);
    block_original = true;
    std::thread older([] { testing::arm_nomination_tap(true); call_v1_evaluate(nullptr, 0, 1, 8); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    block_original = false;
    if (entered) call_v1_evaluate(nullptr, 0, 2, 8);
    release_original = true;
    older.join();
    require(entered, "blocked evaluation never entered original");
    evaluation_snapshot snapshot;
    require(nominated_snapshot(8, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.frame.numeric == 2 && snapshot.camera.near_plane == 1.f && snapshot.projection.depth_scale == 1.,
      "out-of-order evaluation completion replaced newer immutable tuple");
    set_camera_near(second.common, 2.f);
    call_v1_constants(second, 2, 8);
    require(nominated_snapshot(8, snapshot) && snapshot.camera.near_plane == 1.f &&
        snapshot.projection.depth_scale == 1.,
      "later same-frame constants mutated or retained applicability of previously published evaluation");
    call_v1_evaluate(nullptr, 0, 2, 8);
    // Bounded frame storage evicts old frames instead of guessing from latest.
    for (unsigned frame = 3; frame != 38; ++frame) call_v1_constants(first, frame, 8);
    call_v1_evaluate(nullptr, 0, 1, 8);
    expect_nominated(8, evidence_status::missing_constants, "evicted frame reused a different camera");
  }
  // test_evaluation_v2_tokens.
  void test_evaluation_v2_tokens_shipping() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 tracked evidence install failed");
    auto view = viewport(3);
    auto constants = modern_camera();
    opaque_token first, second;
    int native_a{}, native_b{}, native_local{};
    auto resource_a = modern_resource(&native_a), resource_b = modern_resource(&native_b), resource_local = modern_resource(&native_local);
    auto tag_a = depth_tag_for(resource_a), tag_b = depth_tag_for(resource_b), tag_local = depth_tag_for(resource_local);
    const auto a = reinterpret_cast<std::uintptr_t>(&native_a), b = reinterpret_cast<std::uintptr_t>(&native_b),
      local = reinterpret_cast<std::uintptr_t>(&native_local);
    const base_structure *inputs[] {&view.base};
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_nominated(view.value, evidence_status::untracked_frame, "opaque address alone became a frame identity");
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
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == a &&
        snapshot.frame.kind == frame_identity_kind::v2_observed_token && snapshot.frame.generation &&
        snapshot.frame.has_numeric && snapshot.frame.numeric == 101 && snapshot.camera.near_plane == .5f &&
        snapshot.tags[0].scope == tag_scope::explicit_frame,
      "v2 frame-specific constants/tags mixed frames in flight");
    const auto old_generation = snapshot.frame.generation;
    call_v2_evaluate(0, second.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == b &&
        !snapshot.frame.has_numeric && snapshot.frame.generation != old_generation && snapshot.camera.near_plane == 1.f &&
        snapshot.projection.depth_scale == 1.,
      "internal token counter was invented as a game numeric frame");
    const base_structure *local_inputs[] {&view.base, &tag_local.base};
    call_v2_evaluate(0, first.ref(), local_inputs, 2, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == local && snapshot.tags[0].scope == tag_scope::evaluation_local,
      "evaluation-local tag did not override frame tag");
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == a, "local override leaked into next evaluation");
    // Streamline recycles the address for a later frame index: a new generation. (The
    // same index returns the same frame's token; see test_repeated_token_request_keeps_frame.)
    const std::uint32_t recycled_number = number + 3;
    mint(first, &recycled_number);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_nominated(view.value, evidence_status::missing_constants, "recycled token reused old constants/frame tags");
    call_v2_constants(constants, first.ref(), view);
    call_v2_framed_tag(first.ref(), view, &tag_a, 1, nullptr);
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == a &&
        snapshot.frame.generation != old_generation, "fresh recycled-token tuple failed recovery");
    abi_v2::frame_token *output{};
    call_v2_new_token(output, reinterpret_cast<const std::uint32_t *>(1));
    call_v2_evaluate(0, first.ref(), inputs, 1, nullptr);
    expect_nominated(view.value, evidence_status::observation_lost, "unreadable token index left accepted evidence");
  }
  // test_sdk_vram_warning (BF6 10-22): the warning accepts an evaluation's
  // inputs; every other non-zero result fails it.
  void test_sdk_vram_warning_shipping() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 warning install failed");
    auto view = viewport(3);
    auto constants = modern_camera();
    opaque_token token;
    std::uint32_t number = 200;
    int native{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    const base_structure *inputs[] {&view.base};
    const auto frame = [&](std::int32_t inputs_result, std::int32_t evaluate_result) {
      ++number; mint(token, &number);
      result_v2 = inputs_result;
      require(call_v2_constants(constants, token.ref(), view) == inputs_result &&
        call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr) == inputs_result, "input call result changed");
      result_v2 = evaluate_result;
      const auto returned = call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
      result_v2 = 0;
      require(returned == evaluate_result, "evaluation result changed");
    };
    const auto accepted = [&](const char *message) {
      evaluation_snapshot snapshot;
      require(nominated_snapshot(view.value, snapshot) &&
        snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native) &&
        nomination_succeeded(view.value) && snapshot.sequence, message);
    };
    frame(0, 0);
    accepted("warning fixture baseline was not accepted");
    const auto revision = depth_observation_revision();
    frame(0, abi_v2::result_warn_out_of_vram);
    accepted("an over-budget warning on a DLSS evaluation discarded its accepted inputs");
    require(depth_observation_revision() == revision && testing::counts().vram_warnings == 1,
      "an over-budget warning on a DLSS evaluation was counted as an observation loss");
    frame(abi_v2::result_warn_out_of_vram, abi_v2::result_warn_out_of_vram);
    accepted("over-budget warnings on constants and tags discarded them");
    require(depth_observation_revision() == revision && testing::counts().vram_warnings == 4,
      "over-budget warnings on constants and tags were counted as observation losses");
    // Errors next to the warning (eErrorInvalidState 38, the unknown 40, a
    // negative value) still fail the evaluation and lose its observation.
    for (const std::int32_t error : {38, 40, -39}) {
      const auto before = depth_observation_revision();
      frame(0, error);
      require(depth_observation_revision() > before && testing::counts().vram_warnings == 4,
        "an SDK error was not an observation loss, or was counted as a warning");
      frame(0, 0);
      accepted("a successful evaluation did not recover after an SDK error");
    }
  }
  // test_evaluation_global_contract.
  void test_evaluation_global_contract_shipping() {
    fixture cleanup;
    require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 global contract install failed");
    opaque_token token;
    mint(token, nullptr);
    auto constants = modern_camera();
    auto view = viewport(21);
    int native{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    const base_structure *inputs[] {&view.base};
    call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &tag, 1, nullptr);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    evaluation_snapshot snapshot;
    require(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::source_associated_evaluation &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native),
      "global tag fixture did not start valid");
  }
  // test_evaluation_input_age.
  void test_evaluation_input_age_shipping() {
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
    // 25 ms of game time pass while the call is blocked in the original.
    fixture_clock().hitch(25);
    block_original = false;
    release_original = true;
    delayed.join();
    require(entered, "delayed constants never entered original");
    call_v1_tag(&resource, 0, 1, nullptr);
    call_v1_evaluate(nullptr, 0, 1, 1);
    // The evaluation's time: read as its call returns (the deleted store
    // stamped it as the call finished).
    const auto evaluated = fixture_clock().now();
    evaluation_snapshot snapshot;
    require(nominated_snapshot(1, snapshot) && evaluated >= snapshot.camera_tick + 10,
      "blocked constants call completion made old input look newly observed");
  }
  // test_non_renderer_evaluations.
  void test_non_renderer_evaluations_shipping() {
    {
      fixture cleanup;
      require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 feature filter install failed");
      int native{}; abi_v1::resource resource{}; resource.native = &native;
      auto constants = old_camera();
      call_v1_constants(constants, 77, 0); call_v1_tag(&resource, 0, 0, nullptr); call_v1_evaluate(nullptr, 0, 77, 0);
      evaluation_snapshot baseline, next;
      require(nominated_snapshot(0, baseline) && baseline.status == evidence_status::source_associated_evaluation,
        "missing DLSS filter baseline");
      for (const auto feature : {3u, 1u, 2u, 0xffffffffu}) for (const auto marker : {0u, 4u}) for (const auto success : {true, false}) {
        result_v1 = success;
        const auto before = evaluate_call.calls;
        SetLastError(incoming_error);
        require(call_v1_evaluate(nullptr, feature, 78, marker) == success && GetLastError() == outgoing_error,
          "non-renderer v1 call changed return/LastError");
        require(evaluate_call.calls == before + 1 && evaluate_call.first == nullptr && evaluate_call.a == feature &&
            evaluate_call.b == 78 && evaluate_call.c == marker && evaluate_call.incoming_error == incoming_error,
          "non-renderer v1 call changed count/arguments");
        require(nominated_snapshot(0, next) && next.status == evidence_status::source_associated_evaluation &&
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
      evaluation_snapshot baseline, next;
      require(nominated_snapshot(view.value, baseline) && baseline.status == evidence_status::source_associated_evaluation &&
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
        require(nominated_snapshot(view.value, next) && next.status == evidence_status::source_associated_evaluation &&
            next.sequence == baseline.sequence,
          "non-renderer v2 input was parsed or failure revoked DLSS evidence");
      }
    }
  }
  // test_color_tag_snapshots: colour tags and colour untags beside the depth
  // tag never conflate with or poison the evaluated depth.
  void test_color_tag_snapshots_shipping() {
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
      require(nominated_snapshot(0, value) && value.status == evidence_status::source_associated_evaluation &&
          value.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&depth),
        "v1 colors were conflated with depth or each other");
      call_v1_tag(nullptr, 3, 0, nullptr);
      call_v1_constants(constants, 2, 0); call_v1_evaluate(nullptr, 0, 2, 0);
      expect_nominated(0, evidence_status::source_associated_evaluation, "v1 null color untag erased another color or poisoned depth");
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
      require(nominated_snapshot(view.value, value) && value.status == evidence_status::source_associated_evaluation,
        "v2 independent color scope overrides lost");
      require(value.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&depth),
        "legal null v2 backbuffer fabricated identity or poisoned depth");
    }
  }
  // BF6 10-22: over its video-memory budget, Streamline answered every DLSS
  // evaluation with eWarnOutOfVRAM (39) while SR and DLSS-G kept running. The
  // warning accepts the call's inputs; every other non-zero result fails it.
  // The same warning on DLSS-G options (BF6 10-22: success=0 retired the FG
  // source while frame generation kept running) keeps FG knowledge and the FG
  // depth source; a real error still revokes them.
  void test_frame_generation_vram_warning() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG warning hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function &&
        function != returned_feature_function, "FG getter did not return the forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(11);
    fg_options options;
    frame_generation_snapshot fg;
    result_v2 = abi_v2::result_warn_out_of_vram;
    require(set_options(view, options) == abi_v2::result_warn_out_of_vram && query_frame_generation(UINT32_MAX, fg) &&
        fg.known && fg.viewport == view.value && fg.enabled && fg.generated_frames == 1, "a warned FG On was not accepted");
    result_v2 = 0;
    opaque_token token; std::uint32_t numeric = 61;
    auto constants = modern_camera();
    int native{}, commands{};
    auto resource = modern_resource(&native);
    resource.type = 8;
    auto tag = depth_tag_for(resource); tag.lifecycle = 0;
    sunshine_scene_depth::frame normalized;
    const auto fg_source = [&] {
      mint(token, &++numeric); call_v2_constants(constants, token.ref(), view);
      call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
      return nominated_source(view.value, 0, normalized) && normalized.frame_generation_input &&
        normalized.source_id == ((1ull << 63) | view.value);
    };
    require(fg_source(), "a warned FG On did not route FG depth");
    result_v2 = abi_v2::result_warn_out_of_vram;
    require(set_options(view, options) == abi_v2::result_warn_out_of_vram, "warned FG options result changed");
    result_v2 = 0;
    require(query_frame_generation(view.value, fg) && fg.known && fg.enabled && fg_source(),
      "a repeated warned FG On revoked FG knowledge or its depth source");
    options.mode = 0; result_v2 = abi_v2::result_warn_out_of_vram;
    set_options(view, options);
    result_v2 = 0;
    require(query_frame_generation(UINT32_MAX, fg) && fg.known && !fg.enabled, "a warned FG Off was not accepted");
    options.mode = 1; set_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled && fg_source(), "FG On after a warned Off failed");
    result_v2 = 38;
    set_options(view, options);
    result_v2 = 0;
    frame_generation_query_status status{};
    require(!query_frame_generation(UINT32_MAX, fg, &status) && !fg.known && !fg_source(),
      "a failed FG On kept FG knowledge or its FG depth source");
    // A warned getter still hands out the wrapper that observes the options.
    function = nullptr; result_v2 = abi_v2::result_warn_out_of_vram;
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == abi_v2::result_warn_out_of_vram && function &&
        function != returned_feature_function, "a warned FG getter did not return the forwarding wrapper");
    result_v2 = 0;
  }
  // The add-on's own discovery lookup accepts the same warning, so a pointer the
  // engine cached before observation is still detoured; the lookup is not a
  // game call and does not count as one.
  void test_frame_generation_warned_discovery() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *cached_entry{};
    require(call_feature_function(1000, "slDLSSGSetOptions", cached_entry) == 0 && cached_entry == returned_feature_function,
      "pre-observation SDK options cache did not keep the native entry");
    auto cached_options = reinterpret_cast<fg_function>(cached_entry);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "warned FG discovery install failed");
    const auto warnings = testing::counts().vram_warnings;
    result_v2 = abi_v2::result_warn_out_of_vram;
    const bool discovered = testing::discover_frame_generation_hooks();
    result_v2 = 0;
    require(discovered && testing::counts().vram_warnings == warnings,
      "a warned FG discovery lookup failed or was counted as a game call");
    auto view = viewport(11); fg_options options; frame_generation_snapshot fg;
    const auto calls = fg_call.calls;
    require(cached_options(view, options) == 0 && fg_call.calls == calls + 1 && query_frame_generation(view.value, fg) &&
        fg.enabled, "a warned FG discovery lookup left the cached options pointer unobserved");
  }

  void test_feature_independent_depth_contracts() {
    {
      fixture cleanup;
      source_setting = "1"; initialize(nullptr);
      require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 feature-contract install failed");
      int native{}, commands{}; abi_v1::resource resource{}; resource.native = &native;
      auto constants = old_camera();
      call_v1_constants(constants, 77, 0); call_v1_tag(&resource, 0, 0, nullptr);
      sunshine_scene_depth::frame normalized; evaluation_snapshot snapshot;
      for (const auto feature : {1u, 2u, 0xffffffffu}) {
        require(call_v1_evaluate(&commands, feature, 77, 0) && nominated_source(0, 0, normalized) &&
            nominated_snapshot(0, snapshot) && snapshot.feature == feature && snapshot.frame.numeric == 77,
          "legacy feature ID blocked an exact numeric camera/depth contract");
      }
      const auto sequence = snapshot.sequence, revision = depth_observation_revision();
      for (const auto feature : {3u, 17u}) {
        // Reflex remains a timing marker even with matching numbers. The other
        // feature has no exact numeric camera tuple for frame78.
        call_v1_evaluate(&commands, feature, feature == 3 ? 77 : 78, 0);
        require(nominated_snapshot(0, snapshot) && snapshot.sequence == sequence &&
            depth_observation_revision() == revision,
          "unproven legacy argument semantics changed scene-depth authority");
      }
      result_v1 = false; call_v1_evaluate(&commands, 17, 77, 0); result_v1 = true;
      require(!nominated_source(0, 0, normalized), "failed feature with proven depth contract retained source");
    }
    {
      fixture cleanup;
      source_setting = "1"; initialize(nullptr);
      require(testing::install(testing::abi::v2_7_30, tracked_v2_targets), "v2 feature-contract install failed");
      opaque_token token; mint(token, nullptr);
      int native{}, commands{}; auto view = viewport(0);
      auto resource = modern_resource(&native); auto tag = depth_tag_for(resource); tag.type = 49;
      const base_structure *inputs[]{&view.base, &tag.base};
      sunshine_scene_depth::frame normalized; evaluation_snapshot snapshot;
      for (const auto feature : {3u, 17u, 1005u, 0xffffffffu}) {
        SetLastError(incoming_error);
        require(call_v2_evaluate(feature, token.ref(), inputs, 2, &commands) == 0 &&
            GetLastError() == outgoing_error && evaluate_call.incoming_error == incoming_error &&
            nominated_source(view.value, 2, normalized) && !normalized.projection.supplied &&
            normalized.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance &&
            nominated_snapshot(view.value, snapshot) && snapshot.feature == feature,
          "arbitrary v2 feature ID blocked a valid typed depth contract or changed forwarding");
      }
      const auto sequence = snapshot.sequence, revision = depth_observation_revision();
      tag.resource_ptr = reinterpret_cast<const abi_v2::resource *>(1);
      result_v2 = -7;
      call_v2_evaluate(17, token.ref(), inputs, 2, &commands);
      call_v2_evaluate(17, token.ref(), reinterpret_cast<const base_structure **>(1), UINT32_MAX, &commands);
      result_v2 = 0;
      require(nominated_snapshot(view.value, snapshot) && snapshot.sequence == sequence &&
          depth_observation_revision() == revision,
        "speculative opaque/invalid feature input revoked valid source evidence");
      tag.resource_ptr = &resource;
      call_v2_evaluate(17, token.ref(), inputs, 2, nullptr);
      require(nominated_snapshot(view.value, snapshot) && snapshot.sequence == sequence,
        "unknown feature without a scene command acquired depth authority");
      result_v2 = -7; call_v2_evaluate(17, token.ref(), inputs, 2, &commands); result_v2 = 0;
      require(!nominated_source(view.value, 2, normalized), "failed typed feature evaluation retained source");
    }
  }

  // A v2 slGetFeatureFunction lookup other than the FG options entry passes
  // through unchanged (result, returned pointer, arguments and LastError),
  // with the Diagnostics switch on as well.
  void test_feature_function_passthrough() {
    fixture cleanup;
    diagnostics_on diagnostics;
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "v2 PCL getter fixture install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options); // Any non-null entry.
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
    result_v2 = -7; retrieve(4, name);
    result_v2 = 0; retrieve(4, name);
  }
  // A v1 Reflex marker call (slEvaluateFeature feature 3; simulation start,
  // PresentStart and PresentEnd) passes through unchanged (result, LastError
  // and arguments) on success and failure, with the Diagnostics switch off and
  // on. These are test_presentation_v1's forwarding checks, kept when the
  // presentation brackets were deleted.
  void test_reflex_marker_passthrough() {
    fixture cleanup;
    require(testing::install(testing::abi::v1_1_1, v1_targets), "v1 presentation install failed");
    auto marker = [&](unsigned frame, unsigned type) {
      const auto before = evaluate_call.calls;
      SetLastError(incoming_error);
      require(call_v1_evaluate(nullptr, 3, frame, type) == result_v1 && GetLastError() == outgoing_error &&
          evaluate_call.calls == before + 1 && evaluate_call.a == 3 && evaluate_call.b == frame && evaluate_call.c == type &&
          evaluate_call.first == nullptr && evaluate_call.incoming_error == incoming_error, "v1 presentation marker forwarding changed");
    };
    auto markers = [&](unsigned frame) {
      for (const bool success : {true, false}) {
        result_v1 = success;
        marker(frame, 0); marker(frame, 4); marker(frame, 5);
      }
      result_v1 = true;
    };
    markers(10);
    diagnostics_on diagnostics;
    markers(11);
  }


  // Streamline 2.11.x declares UIAlpha as 68; 2.12.0 moved it to 69 and reused
  // 68 for ResponsivityMask. Earlier headers declare neither. The hook maps the
  // raw tag through the loaded interposer's header before offering UI capture.
  void test_versioned_ui_alpha_tags() {
    namespace mask = sunshine_game3d::ui_mask;
    namespace buffers = sunshine_streamline::buffers;
    struct expectation { versioning::number version; std::uint32_t raw; bool offered; };
    const expectation cases[]{
      {testing::abi::v2_11_1, 68, true}, {testing::abi::v2_11_1, 69, false},
      {testing::abi::v2_12_0, 69, true}, {testing::abi::v2_12_0, 68, false},
      {testing::abi::v2_7_30, 68, false}, {testing::abi::v2_7_30, 69, false}};
    for (const auto &c : cases) {
      fixture cleanup;
      source_setting = "1"; initialize(nullptr);
      require(testing::install(c.version, presentation_v2_targets), "versioned UI alpha hook install failed");
      const auto contract = buffers::active();
      require(contract.known && buffers::ui_alpha_authenticated(contract) == !(c.version == testing::abi::v2_7_30),
        "installed interposer did not publish its buffer contract");
      returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
      void *function{};
      require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function,
        "versioned UI alpha FG options wrapper was unavailable");
      auto view = viewport(12); fg_options options;
      require(reinterpret_cast<fg_function>(function)(view, options) == 0, "versioned UI alpha FG options call failed");
      frame_generation_snapshot fg;
      require(query_frame_generation(view.value, fg) && fg.enabled, "versioned UI alpha FG mode was not confirmed");
      constexpr std::uint64_t runtime = 0x101, device = 0x201;
      mask::set_request({runtime, device, fg.epoch, depth_observation_revision(), view.value, 1920, 1080, true}, fixture_clock().now());
      int native{}, commands{};
      auto resource = modern_resource(&native); resource.width = 1920; resource.height = 1080; resource.state = 8;
      auto tag = depth_tag_for(resource); tag.type = c.raw; tag.lifecycle = 0; tag.area = {0, 0, 1920, 1080};
      require(call_v2_tag(view, &tag, 1, &commands) == 0, "versioned UI alpha tag changed SDK forwarding");
      mask::boundary observed;
      const bool offered = mask::testing::last_attempt(runtime, observed) && observed.kind == mask::source_kind::alpha &&
        observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native);
      require(offered == c.offered, c.offered ? "header-declared UIAlpha tag was not offered as UI alpha" :
        "a tag number not assigned to UIAlpha by the loaded header was offered as UI alpha");
      mask::invalidate(runtime);
    }
    // The pure contract also rejects 1.x UIHint as UIColorAndAlpha and clamps
    // each release's declared range.
    const auto v1 = buffers::for_version({1, 1, 1, 0});
    require(buffers::canonical(v1, 23) == buffers::unknown && buffers::canonical(v1, 0) == 0 &&
        buffers::canonical(v1, 48) == buffers::unknown && buffers::canonical(v1, 2) == 2,
      "1.x buffer contract changed depth/HUD-less or accepted UIHint/unknown tags");
    // The 2.2.1 header already declares HiResDepth 48 and LinearDepth 49 and
    // ends at 52 (S17); 2.2.0 ends at 37.
    const auto v2_2 = buffers::for_version({2, 2, 1, 0});
    require(buffers::canonical(v2_2, 23) == 23 && buffers::canonical(v2_2, 48) == 48 && buffers::canonical(v2_2, 49) == 49 &&
        buffers::canonical(v2_2, 52) == 52 && buffers::canonical(v2_2, 53) == buffers::unknown,
      "2.2.1 contract lost its depth kinds or accepted post-2.4 tags");
    const auto v2_2_0 = buffers::for_version({2, 2, 0, 0});
    require(buffers::canonical(v2_2_0, 37) == 37 && buffers::canonical(v2_2_0, 38) == buffers::unknown &&
        buffers::canonical(v2_2_0, 48) == buffers::unknown, "2.2.0 contract accepted tags its header lacks");
    const auto v2_6 = buffers::for_version({2, 6, 10, 0});
    require(!v2_6.unknown_resource_type && buffers::for_version({2, 7, 2, 0}).unknown_resource_type &&
        !v2_2.unknown_resource_type && !v1.unknown_resource_type, "eUnknown declared outside 2.7.2 and later (S12)");
    require(buffers::canonical(v2_6, 48) == 48 && buffers::canonical(v2_6, 53) == 53 &&
        buffers::canonical(v2_6, 68) == buffers::unknown && !buffers::ui_alpha_authenticated(v2_6),
      "unpublished 2.6 build lost its 2.4 tags or gained UIAlpha");
    const auto future = buffers::for_version({2, 16, 0, 0});
    require(future.known && !future.surveyed && buffers::canonical(future, 69) == buffers::ui_alpha &&
        !buffers::ui_alpha_authenticated(future), "unsurveyed 2.x release was treated as authenticated");
    require(!buffers::for_version({3, 0, 0, 0}).known, "unknown Streamline major received a buffer contract");
    // An inexact 2.x family: a raw tag is known only where every candidate's
    // header agrees, and the contract is never surveyed (UI alpha stays manual).
    const auto family = [](std::initializer_list<versioning::number> values) {
      versioning::candidates out;
      for (const auto &v : values) out.values[out.count++] = v;
      return buffers::for_versions(out);
    };
    require(family({{2, 7, 2, 0}, {2, 12, 0, 0}}).unknown_resource_type && !family({{2, 6, 10, 0}, {2, 7, 2, 0}}).unknown_resource_type,
      "an inexact family declared eUnknown although a candidate does not (S12)");
    const auto uialpha = family({{2, 11, 1, 0}, {2, 12, 0, 0}});
    require(uialpha.known && !uialpha.surveyed && !buffers::ui_alpha_authenticated(uialpha) &&
        buffers::canonical(uialpha, 68) == buffers::unknown && buffers::canonical(uialpha, 69) == buffers::unknown &&
        buffers::canonical(uialpha, 0) == 0 && buffers::canonical(uialpha, 53) == 53 && uialpha.raw_ui_alpha == buffers::unknown,
      "candidates {2.11, 2.12} mapped a renumbered tag");
    const auto depth = family({{2, 2, 0, 0}, {2, 5, 0, 0}});
    require(depth.known && !depth.surveyed && buffers::canonical(depth, 48) == buffers::unknown &&
        buffers::canonical(depth, 49) == buffers::unknown && buffers::canonical(depth, 0) == 0 &&
        depth.version == versioning::number{2, 2, 0, 0} && depth.last_type == 37,
      "candidates {2.2.0, 2.5} mapped a tag one candidate lacks");
    const auto same = family({{2, 12, 0, 0}, {2, 13, 0, 0}});
    require(buffers::canonical(same, 69) == buffers::ui_alpha && !buffers::ui_alpha_authenticated(same),
      "agreeing candidates lost a shared tag or authenticated an inexact header");
    require(!family({{2, 12, 0, 0}, {3, 0, 0, 0}}).known, "a candidate of an unknown major left the family known");
    const auto single = family({{2, 12, 0, 0}});
    require(single.surveyed && single.candidate_count == 0 && buffers::ui_alpha_authenticated(single),
      "a single candidate was not its exact contract");
    buffers::set_active_contract(depth);
    const auto active = buffers::active();
    require(active.candidate_count == 2 && buffers::canonical(active, 48) == buffers::unknown && !active.surveyed,
      "the process-wide contract lost an inexact family's candidates");
    buffers::clear_active();
    require(!buffers::active().known, "a cleared contract stayed known");
    std::puts("PASS versioned Streamline buffer contract: 2.11.x tag68 and 2.12+ tag69 map to UIAlpha; undeclared/renumbered tags stay unavailable");
  }

  void test_frame_generation_color_only_alpha() {
    namespace mask = sunshine_game3d::ui_mask;
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    // UIAlpha is tag 69 only from Streamline 2.12.0 onward.
    require(testing::install(testing::abi::v2_12_0, presentation_v2_targets), "FG alpha hook install failed");
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
    mask::set_request(wanted, fixture_clock().now());
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
    require(!nominated_snapshot(view.value, depth), "alpha-only tag nominated a depth evaluation");
    const auto global_sequence = observed.source.sequence;
    opaque_token token; std::uint32_t numeric = 71; mint(token, &numeric);
    require(call_v2_framed_tag(token.ref(), view, &tag, 1, &commands) == 0 &&
        mask::testing::last_attempt(runtime, observed) && observed.source.sequence > global_sequence &&
        observed.tag_scope == 1 && observed.source.source_frame_explicit &&
        observed.source.source_frame_has_numeric && observed.source.source_frame_numeric == numeric &&
        observed.source.source_frame_token == reinterpret_cast<std::uint64_t>(&token.ref()),
      "explicit-frame tag53 lost its own frame provenance");
    for (const auto kind : {mask::source_kind::color_and_alpha, mask::source_kind::alpha, mask::source_kind::hudless}) {
      tag.type = static_cast<std::uint32_t>(kind);
      call_v2_tag(view, &tag, 1, &commands);
      require(mask::testing::last_attempt(runtime, observed) && observed.kind == kind &&
          observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native) &&
          observed.source.valid_until == sunshine_scene_depth::lifetime::at_call && !observed.tag_scope,
        "dedicated UI tag was not offered to the live owner with diagnostics off");
      call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
      require(mask::testing::last_attempt(runtime, observed) && observed.kind == kind && observed.tag_scope == 1 &&
          observed.source.source_frame_numeric == numeric,
        "dedicated UI tag lost its explicit frame provenance");
    }
    // A local UI input is offered under the same contract, copied on its
    // evaluate's list inside that evaluate (owner ruling R4, the matrix's U3).
    for (const auto kind : {mask::source_kind::color_and_alpha, mask::source_kind::alpha, mask::source_kind::hudless,
           mask::source_kind::backbuffer}) {
      tag.type = static_cast<std::uint32_t>(kind);
      const auto before = mask::testing::last_attempt(runtime, observed) ? observed.source.sequence : 0;
      const base_structure *inputs[]{&view.base, &tag.base};
      int evaluate_commands{};
      require(call_v2_evaluate(0, token.ref(), inputs, 2, &evaluate_commands) == 0 &&
          mask::testing::last_attempt(runtime, observed) && observed.kind == kind && observed.source.sequence > before &&
          observed.command == reinterpret_cast<std::uint64_t>(&evaluate_commands) &&
          observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native) &&
          observed.source.valid_until == sunshine_scene_depth::lifetime::at_call && observed.tag_scope == 3,
        "a local UI input was not offered at its evaluate");
    }
    // A local UI input names its viewport's UI scope, as a UI tag does,
    // also from an evaluate that supplies no depth (an unknown feature).
    {
      auto local_view = viewport(12);
      const base_structure *inputs[]{&local_view.base, &tag.base};
      int evaluate_commands{};
      ui_observation_scope scope;
      require(call_v2_evaluate(77, token.ref(), inputs, 2, &evaluate_commands) == 0 &&
          query_ui_scope(local_view.value, scope) && scope.viewport == local_view.value && scope.sequence,
        "a local UI input did not name its viewport's UI scope");
    }
    for (const auto kind : {53u, 23u, 69u, 2u}) {
      for (const auto lifecycle : {0u, 1u, 2u, 3u, UINT32_MAX}) {
        auto lifetime_tag = tag; lifetime_tag.type = kind; lifetime_tag.lifecycle = lifecycle;
        call_v2_tag(view, &lifetime_tag, 1, &commands);
        const auto expected = lifecycle == 0 ? sunshine_scene_depth::lifetime::at_call : lifecycle == 1 ?
          sunshine_scene_depth::lifetime::until_present : lifecycle == 2 ?
          sunshine_scene_depth::lifetime::until_evaluation : sunshine_scene_depth::lifetime::unsupported;
        require(mask::testing::last_attempt(runtime, observed) && observed.kind == static_cast<mask::source_kind>(kind) &&
            observed.source.valid_until == expected && observed.source.native_state == resource.state &&
            observed.source.resource.native == (lifecycle <= 2 ? reinterpret_cast<std::uint64_t>(&native) : 0),
          "UI tag hook rewrote the declared lifetime/state or accepted an unsupported lifecycle");
      }
    }
    auto absent_alpha = tag; absent_alpha.resource_ptr = nullptr;
    tag.type = 53;
    const abi_v2::resource_tag fallback[]{absent_alpha, tag};
    call_v2_tag(view, fallback, 2, &commands);
    require(mask::testing::last_attempt(runtime, observed) && observed.kind == mask::source_kind::backbuffer &&
        observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native),
      "null preferred UI source blocked a valid same-batch backbuffer fallback");
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
    wanted.revision = depth_observation_revision(); mask::set_request(wanted, fixture_clock().now());
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed), "renewed alpha scope failed to recover");
    options.mode = 0; options_call(view, options);
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport, fixture_clock().now()) &&
        !mask::testing::last_attempt(runtime, observed), "FG Off retained alpha owner scope");
    wanted.revision = depth_observation_revision(); mask::set_request(wanted, fixture_clock().now());
    tag.type = 2;
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed) && observed.kind == mask::source_kind::hudless &&
        !observed.source.frame_generation_input, "Fresh FG-off scope did not capture HUD-less color");
    options.mode = 1; options_call(view, options);
    wanted.revision = depth_observation_revision(); mask::set_request(wanted, fixture_clock().now());
    call_v2_tag(view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed), "re-enabled FG did not resume color-only alpha capture");
    result_v2 = -7; options_call(view, options); result_v2 = 0;
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport, fixture_clock().now()), "failed FG options retained alpha scope");
    mask::set_request(wanted, fixture_clock().now()); shutdown();
    require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport, fixture_clock().now()), "shutdown retained live alpha request");
  }

  void test_ui_gate_viewport_zero_mixed_batch() {
    namespace mask = sunshine_game3d::ui_mask;
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Mixed UI hook install failed");
    auto view = viewport(0);
    int depth{}, motion{}, hudless{}, commands{};
    abi_v2::resource resources[3]{};
    resources[0].native = &depth; resources[0].state = 96;
    resources[1].native = &motion; resources[1].state = 8;
    resources[2].native = &hudless; resources[2].state = 1024;
    // Match the live zero-base descriptors, omitted dimensions and mixed tag
    // batch. No FG options, explicit frame tag or SL evaluation is supplied.
    const abi_v2::resource_tag tags[]{
      {{nullptr, tag_guid, 1}, &resources[0], 0, 0, {0, 0, 2228, 1253}},
      {{nullptr, tag_guid, 1}, &resources[1], 1, 0, {0, 0, 2228, 1253}},
      {{nullptr, tag_guid, 1}, &resources[2], 2, 0, {0, 0, 3840, 2160}}};
    call_v2_tag(view, tags, 3, &commands);
    ui_observation_scope scope;
    require(query_ui_scope(UINT32_MAX, scope) && !scope.viewport,
      "Mixed viewport-zero tags did not establish live UI discovery scope");
    mask::request wanted{0x710, 0x720, scope.epoch, scope.revision, 0, 3840, 2160, true};
    mask::set_request(wanted, fixture_clock().now());
    SetLastError(incoming_error);
    require(call_v2_tag(view, tags, 3, &commands) == 0 && GetLastError() == outgoing_error,
      "Mixed UI capture gate changed public SDK forwarding");
    mask::diagnostic_snapshot observed;
    require(mask::query_diagnostic(wanted.runtime, observed) &&
        observed.hook_gate.state == mask::capture_gate::admitted && observed.hook_gate.matching_requests == 1 &&
        observed.hook_gate.seen_kinds == mask::source_mask(mask::source_kind::hudless) &&
        observed.hook_gate.epoch == wanted.epoch && observed.hook_gate.revision == wanted.revision &&
        observed.latest_boundary.kind == mask::source_kind::hudless && observed.latest_boundary.source.sequence &&
        !observed.latest_boundary.source.viewport && !observed.latest_boundary.source.frame_generation_input,
      "Mixed viewport-zero HUDless batch did not reach its matching live capture boundary");
    ++wanted.revision;
    mask::set_request(wanted, fixture_clock().now());
    call_v2_tag(view, tags, 3, &commands);
    require(mask::query_diagnostic(wanted.runtime, observed) && !observed.latest_boundary.source.sequence &&
        observed.hook_gate.state == mask::capture_gate::no_matching_request && !observed.hook_gate.matching_requests &&
        observed.hook_gate.seen_kinds == 8 && observed.hook_gate.revision != observed.wanted.revision,
      "Rejected UI request scope lacked exact hook gate evidence or fabricated capture");
  }

  void test_repeated_fg_off_preserves_independent_ui() {
    namespace mask = sunshine_game3d::ui_mask;
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Repeated FG Off hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function,
      "Repeated FG Off options wrapper was unavailable");
    const auto options_call = reinterpret_cast<fg_function>(function);
    const auto view = viewport(0);
    fg_options options; options.mode = options.generated_frames = 0;
    int depth{}, motion{}, hudless{}, commands{};
    abi_v2::resource resources[3]{};
    resources[0].native = &depth; resources[0].state = 96;
    resources[1].native = &motion; resources[1].state = 8;
    resources[2].native = &hudless; resources[2].state = 1024;
    const abi_v2::resource_tag tags[]{
      {{nullptr, tag_guid, 1}, &resources[0], 0, 0, {0, 0, 2228, 1253}},
      {{nullptr, tag_guid, 1}, &resources[1], 1, 0, {0, 0, 2228, 1253}},
      {{nullptr, tag_guid, 1}, &resources[2], 2, 0, {0, 0, 3840, 2160}}};
    const auto set_options = [&](std::int32_t result) {
      result_v2 = result;
      SetLastError(incoming_error);
      const auto actual = options_call(view, options);
      result_v2 = 0;
      require(actual == result && fg_call.incoming_error == incoming_error && GetLastError() == outgoing_error,
        "Repeated FG options changed SDK result or LastError");
    };
    // Reproduce successful independent tags followed by an unsupported Off
    // call every frame. A known, unchanged Off has the same ownership rule.
    for (const std::int32_t result : {-7, 0}) {
      set_options(result);
      call_v2_tag(view, tags, 3, &commands);
      ui_observation_scope scope;
      require(query_ui_scope(0, scope), "Repeated Off prevented independent UI scope discovery");
      const mask::request wanted{0x730, 0x740, scope.epoch, scope.revision, 0, 3840, 2160, true};
      mask::set_request(wanted, fixture_clock().now());
      mask::diagnostic_snapshot observed;
      require(mask::query_diagnostic(wanted.runtime, observed), "Repeated Off fixture lost initial request");
      const auto generation = observed.request_generation;
      std::uint64_t sequence{};
      for (unsigned frame = 0; frame != 4; ++frame) {
        call_v2_tag(view, tags, 3, &commands);
        require(mask::query_diagnostic(wanted.runtime, observed) &&
            observed.latest_boundary.source.sequence > sequence,
          "Mixed independent tag batch did not update its capture boundary");
        sequence = observed.latest_boundary.source.sequence;
        set_options(result);
        require(mask::query_diagnostic(wanted.runtime, observed) && observed.request_generation == generation &&
            observed.latest_boundary.source.sequence == sequence &&
            observed.hook_gate.state == mask::capture_gate::admitted && observed.hook_gate.matching_requests == 1 &&
            observed.hook_gate.seen_kinds == 8 && !observed.latest_boundary.source.frame_generation_input,
          "Unchanged failed/Off options cleared independent HUDless request or captured boundary");
      }
      // Changing the FG knowledge state still revokes once; the next fresh tag
      // may establish a new independent request under the resulting state.
      options.mode = options.generated_frames = 1;
      set_options(0);
      call_v2_tag(view, tags, 3, &commands);
      require(mask::testing::last_attempt(wanted.runtime, observed.latest_boundary) &&
          observed.latest_boundary.source.frame_generation_input,
        "Confirmed FG On did not retain its distinct input role");
      options.mode = options.generated_frames = 0;
      set_options(result);
      require(!mask::interested(wanted.epoch, wanted.revision, wanted.viewport, fixture_clock().now()),
        "Actual FG mode/knowledge loss retained the previous UI request");
    }
  }

  void test_ui_scope_without_frame_generation_options() {
    namespace mask = sunshine_game3d::ui_mask;
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    // Includes UIAlpha, which is tag 69 only from Streamline 2.12.0 onward.
    require(testing::install(testing::abi::v2_12_0, presentation_v2_targets), "Automatic UI hook install failed");
    auto view = viewport(11);
    int native{}, commands{};
    auto resource = modern_resource(&native); resource.width = 3840; resource.height = 2160; resource.state = 8;
    auto tag = depth_tag_for(resource); tag.type = 2; tag.lifecycle = 0; tag.area = {0, 0, 3840, 2160};
    ui_observation_scope scope;
    require(!query_ui_scope(UINT32_MAX, scope), "Unobserved SL viewport invented a UI capture scope");
    call_v2_tag(view, &tag, 1, &commands);
    require(query_ui_scope(UINT32_MAX, scope) && scope.epoch && scope.sequence && scope.viewport == view.value &&
        scope.revision == depth_observation_revision(), "Fresh HUD-less tag did not establish scope with FG/diagnostic probe absent");
    require(!query_ui_scope(view.value, scope, 0), "Expired UI tag admitted capture scope");
    require(query_ui_scope(view.value, scope), "Fresh UI scope could not be selected explicitly");
    constexpr std::uint64_t runtime = 0x100, device = 0x200;
    mask::request wanted{runtime, device, scope.epoch, scope.revision, scope.viewport, 3840, 2160, true};
    mask::set_request(wanted, fixture_clock().now());
    mask::boundary observed;
    for (const auto kind : {mask::source_kind::hudless, mask::source_kind::backbuffer,
        mask::source_kind::color_and_alpha, mask::source_kind::alpha}) {
      tag.type = static_cast<std::uint32_t>(kind);
      SetLastError(incoming_error);
      require(call_v2_tag(view, &tag, 1, &commands) == 0 && GetLastError() == outgoing_error &&
          tag_call.incoming_error == incoming_error && mask::testing::last_attempt(runtime, observed) &&
          observed.kind == kind && observed.source.epoch == scope.epoch && !observed.source.frame_generation_input &&
          observed.source.resource.native == reinterpret_cast<std::uint64_t>(&native) && !observed.tag_scope,
        "Automatic capture depended on FG options or changed source forwarding/provenance");
    }
    opaque_token token; std::uint32_t frame = 89; mint(token, &frame);
    tag.type = 2;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(mask::testing::last_attempt(runtime, observed) && observed.kind == mask::source_kind::hudless &&
        observed.tag_scope == 1 && observed.source.source_frame_explicit && observed.source.source_frame_has_numeric &&
        observed.source.source_frame_numeric == frame && !observed.source.frame_generation_input,
      "FG-independent HUD-less capture fabricated or lost explicit frame identity");
    auto other = viewport(12); call_v2_tag(other, &tag, 1, &commands);
    require(!query_ui_scope(UINT32_MAX, scope) && query_ui_scope(view.value, scope),
      "Multiple fresh UI viewports were guessed to belong to one output");
    testing::lose_observation();
    require(!query_ui_scope(view.value, scope), "UI scope survived observation loss");
    call_v2_tag(view, &tag, 1, &commands);
    require(query_ui_scope(view.value, scope) && scope.revision == depth_observation_revision(),
      "Fresh UI tag did not recover after observation loss");
    result_v2 = -7; call_v2_tag(view, &tag, 1, &commands); result_v2 = 0;
    require(!query_ui_scope(view.value, scope), "Failed SDK tag call retained an automatic UI scope");
  }

  void test_standalone_depth_tags() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Standalone depth hook install failed");
    auto view = viewport(0);
    int native{}, commands{};
    auto resource = modern_resource(&native);
    resource.base = {}; resource.width = resource.height = 0; resource.state = 0x60;
    auto depth = depth_tag_for(resource); depth.lifecycle = 0; depth.area = {0, 0, 2228, 1253};
    auto hudless = depth; hudless.type = 2; hudless.area = {0, 0, 3840, 2160};
    auto motion = depth; motion.type = 1;
    abi_v2::resource_tag tags[]{hudless, motion, depth};
    const auto before = testing::counts().evaluations;
    evaluation_snapshot snapshot;
    sunshine_scene_depth::frame normalized;
    const auto tag_now = [&] { return call_v2_tag(view, tags, 3, &commands); };
    SetLastError(incoming_error);
    require(tag_now() == 0 && tag_call.incoming_error == incoming_error && GetLastError() == outgoing_error &&
        tag_call.first == &view && tag_call.second == tags && tag_call.third == &commands,
      "Standalone mixed tags changed public SDK arguments or LastError");
    require(nominated_snapshot(0, snapshot) && snapshot.tag_boundary && !snapshot.frame_generation_input &&
        snapshot.command_buffer == reinterpret_cast<std::uintptr_t>(&commands) && nomination_succeeded(0) &&
        nominated_source(0, 0, normalized) && !normalized.frame_generation_input && !normalized.source_id &&
        !normalized.projection.supplied && !normalized.projection.direction_supplied && !normalized.jitter.supplied &&
        !normalized.source_frame_explicit && !normalized.source_frame_generation && !normalized.source_frame_token &&
        normalized.native_state == 0x60 && normalized.proof == sunshine_scene_depth::state_proof::declared &&
        normalized.valid_until == sunshine_scene_depth::lifetime::at_call,
      "FG-unobserved viewport-zero tag did not retain raw depth with its own exact state/boundary");
    auto constants = modern_camera(); opaque_token token;
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_snapshot(0, snapshot) && snapshot.frame.kind == frame_identity_kind::v2_constants_call &&
        snapshot.camera_sequence < snapshot.sequence && snapshot.frame_correlated &&
        nominated_source(0, 0, normalized) && normalized.projection.supplied &&
        !normalized.source_frame_explicit && !normalized.source_frame_has_numeric && !normalized.frame_generation_input,
      "Standalone global depth lost its fresh same-viewport camera-call association");
    call_v2_tag(view, tags, 3, nullptr);
    require(!nominated_source(0, 0, normalized), "Standalone depth accepted a missing copy command");
    tags[2].resource_ptr = nullptr; tag_now();
    require(!nominated_source(0, 0, normalized), "Standalone untag retained old depth");
    tags[2] = depth;
    result_v2 = -7; tag_now(); result_v2 = 0;
    require(!nominated_source(0, 0, normalized), "Failed standalone tag retained source authority");
    tag_now();
    require(nominated_source(0, 0, normalized) && !normalized.projection.supplied,
      "Fresh post-failure tag did not recover raw depth independently of revoked camera");
    call_v2_constants(constants, token.ref(), view); tag_now();
    testing::lose_observation();
    tag_now();
    require(nominated_source(0, 0, normalized) && !normalized.projection.supplied,
      "Fresh tag borrowed a camera from before observation loss");
    call_v2_constants(constants, token.ref(), view);
    fixture_clock().hitch(270); tag_now();
    require(nominated_source(0, 0, normalized) && !normalized.projection.supplied &&
        !normalized.projection.direction_supplied && !normalized.jitter.supplied,
      "Stale camera calibrated standalone raw depth");
    // A mixed batch must not expire or eagerly copy an UntilEvaluate input.
    std::uint32_t numeric = 99; mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    auto until = depth; until.type = 48; until.lifecycle = 2;
    abi_v2::resource_tag mixed[]{depth, until};
    call_v2_framed_tag(token.ref(), view, mixed, 2, &commands);
    require(nominated_source(0, 0, normalized) && normalized.source_frame_explicit &&
        normalized.source_frame_numeric == numeric && !nominated_source(0, 1, normalized),
      "Standalone mixed lifetimes claimed an UntilEvaluate input at the tag boundary");
    require(testing::counts().evaluations == before, "Standalone tags fabricated SL evaluation calls");
    const base_structure *inputs[]{&view.base};
    call_v2_evaluate(0, token.ref(), inputs, 1, &commands);
    require(nominated_source(0, 1, normalized) &&
        normalized.valid_until == sunshine_scene_depth::lifetime::until_evaluation,
      "Standalone OnlyValidNow capture prematurely expired a neighboring UntilEvaluate tag");
    call_v2_evaluate(0, token.ref(), inputs, 1, &commands);
    require(!nominated_source(0, 1, normalized), "UntilEvaluate escaped its real evaluation return");
    // The same tag call changes role only after observed successful FG options.
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{}; call_feature_function(1000, "slDLSSGSetOptions", function);
    require(function != nullptr, "Standalone role transition lacked FG options wrapper");
    auto set_options = reinterpret_cast<fg_function>(function); fg_options options;
    set_options(view, options); call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(0, 0, normalized) && normalized.frame_generation_input &&
        normalized.source_id == (1ull << 63), "Observed FG On lost its distinct mandatory source role");
    options.mode = 0; set_options(view, options); tag_now();
    require(nominated_source(0, 0, normalized) && !normalized.frame_generation_input && !normalized.source_id,
      "FG Off suppressed independent standalone depth or retained its FG namespace");
    require(nominated_snapshot(0, snapshot), "Standalone final observation missing");
    const auto only_valid_now_sequence = snapshot.sequence;
    tags[2].lifecycle = 1; tag_now();
    require(nominated_snapshot(0, snapshot) && snapshot.sequence == only_valid_now_sequence,
      "Ordinary UntilPresent tag invented a synchronous depth opportunity");
    tags[2].lifecycle = 2; tag_now();
    require(nominated_snapshot(0, snapshot) && snapshot.sequence == only_valid_now_sequence,
      "Ordinary longer-lived tag invented a synchronous depth opportunity");
    tags[2].lifecycle = 3; tag_now();
    // The earlier capture already copied its pixels; a later unusable tag is
    // not a new opportunity and does not revoke that finished capture.
    require(nominated_snapshot(0, snapshot) && snapshot.sequence == only_valid_now_sequence,
      "Unsupported tag lifetime invented a synchronous depth opportunity or revoked the earlier capture");
  }

  // Streamline's untag (sl_core_api.h: a null resource removes the tag) is
  // counted and reported as an untag, never as the zero-header Resource
  // compatibility layout, which only a real native can have (ACS 10-22 logged
  // its FG untags as "zero-base resource compatibility ... native=0x0").
  void test_depth_untag_reporting() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Untag hook install failed");
    auto view = viewport(0);
    int native{}, commands{};
    auto resource = modern_resource(&native);
    resource.base = {}; resource.state = 0x60; // The zero-header layout with a real native.
    auto depth = depth_tag_for(resource); depth.lifecycle = 0;
    auto untag = depth; untag.resource_ptr = nullptr;
    abi_v2::resource empty{};
    auto zero_native = depth; zero_native.resource_ptr = &empty;
    forget_logged();
    const auto before = testing::counts().depth_untags;
    require(call_v2_tag(view, &untag, 1, &commands) == 0 && testing::counts().depth_untags == before + 1 &&
        logged({"Sunshine Streamline depth untag:", "viewport=0", "scope=global", "lifecycle=0", "resource=null ", "untags=1"}) == 1 &&
        !logged({"zero-base resource compatibility"}),
      "A null depth Resource was not reported as an untag, or was reported as the zero-header layout");
    require(call_v2_tag(view, &zero_native, 1, nullptr) == 0 && testing::counts().depth_untags == before + 2 &&
        !logged({"zero-base resource compatibility"}),
      "A Resource with a null native was not counted as an untag, or was reported as the zero-header layout");
    evaluation_snapshot snapshot; sunshine_scene_depth::frame normalized;
    require(nominated_snapshot(0, snapshot) && snapshot.tag_boundary && snapshot.tags[0].present &&
        snapshot.tags[0].supported && !snapshot.tags[0].value.native_resource && !nominated_source(0, 0, normalized),
      "An untag changed the probe's no-current-depth metadata");
    require(call_v2_tag(view, &depth, 1, &commands) == 0 && testing::counts().depth_untags == before + 2 &&
        logged({"zero-base resource compatibility", "native=0x0 "}) == 0 &&
        logged({"zero-base resource compatibility", "native_retained=0"}) == 1 && nominated_source(0, 0, normalized),
      "The zero-header layout with a real native lost its compatibility report or its depth");
    // A malformed Resource is unsupported, not an untag, whatever its native.
    auto malformed = resource; malformed.base.version = 1; malformed.native = nullptr;
    auto rejected = depth; rejected.resource_ptr = &malformed;
    require(call_v2_tag(view, &rejected, 1, &commands) == 0 && testing::counts().depth_untags == before + 2 &&
        nominated_snapshot(0, snapshot) && !snapshot.tags[0].supported,
      "A malformed Resource with a null native was counted as an untag");
  }

  void test_frame_generation_tags() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
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
      presentation_scope::fg_view policy;
      require(!policy.update(frame_generation_query_status::unavailable, {}).require_frame_generation,
        "Unobserved unavailable FG blocked ordinary depth routing");
      auto selected = policy.update(frame_generation_query_status::observed, fg);
      require(selected.require_frame_generation && selected.epoch == fg.epoch && selected.viewport == fg.viewport &&
          !selected.exclude_unconfirmed_fg && policy.generated_frames() == fg.generated_frames,
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
      const auto deadline = std::chrono::steady_clock::now() + finish_bound();
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
      require(!selected.require_frame_generation && selected.exclude_unconfirmed_fg && !selected.epoch && !policy.generated_frames(),
        "Ambiguous FG retained a mandatory missing scope or authorized unconfirmed FG depth");
      selected = policy.update(frame_generation_query_status::observed, fg);
      require(selected.require_frame_generation && selected.epoch == fg.epoch && selected.viewport == fg.viewport &&
          !selected.exclude_unconfirmed_fg && policy.generated_frames() == fg.generated_frames,
        "A new confirmed FG On failed to restore its exact scope after ambiguity");
      selected = policy.update(frame_generation_query_status::unavailable, {});
      require(!selected.require_frame_generation && selected.exclude_unconfirmed_fg && !selected.epoch,
        "Lost FG evidence preserved an obsolete scope or blocked independent ordinary depth");
      auto off = fg; off.enabled = false;
      selected = policy.update(frame_generation_query_status::observed, off);
      require(!selected.require_frame_generation && !selected.exclude_unconfirmed_fg && !selected.epoch &&
          !policy.update(frame_generation_query_status::unavailable, {}).require_frame_generation && !policy.generated_frames(),
        "Confirmed Off did not restore ordinary routing through an unavailable query");
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
    require(nominated_snapshot(view.value, snapshot) && snapshot.tag_boundary && snapshot.feature == 1000 &&
        nomination_succeeded(view.value) && snapshot.tags[0].supported && snapshot.frame_correlated,
      "FG OnlyValidNow tag did not publish its synchronous frame/camera tuple");
    require(nominated_source(view.value, 0, normalized) && normalized.frame_generation_input &&
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
    // Enabled FG owns the viewport only for frames whose FG copy the capture
    // owner accepted (frame_generation_scope_live); this fixture's owner is
    // inactive, so the same viewport's SR evaluation supplies its own depth.
    auto sr_tag = tag; sr_tag.lifecycle = 1;
    const base_structure *sr_inputs[]{&view.base, &sr_tag.base};
    call_v2_evaluate(0, token.ref(), sr_inputs, 2, &commands);
    require(nominated_snapshot(view.value, snapshot) && snapshot.feature == 0 && snapshot.tags[0].supported &&
        nominated_source(view.value, 0, normalized) && !normalized.frame_generation_input,
      "same-viewport SR evaluation of an FG scope without an FG copy did not supply its own depth");
    auto unrelated_view = viewport(12);
    call_v2_constants(constants, token.ref(), unrelated_view);
    const base_structure *other_sr_inputs[]{&unrelated_view.base, &sr_tag.base};
    call_v2_evaluate(0, token.ref(), other_sr_inputs, 2, &commands);
    require(nominated_source(unrelated_view.value, 0, normalized) && !normalized.frame_generation_input,
      "FG mode on one viewport suppressed another viewport's SR input");

    precision.scale = -.5f;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(nominated_source(view.value, 0, normalized) && normalized.projection.depth_offset == 0 &&
        normalized.projection.depth_scale == .5 && normalized.projection.raw_scale == -.5 &&
        !normalized.projection.reversed && inverse_distance(normalized, 0.0) == .5,
      "negative precision scale failed to transform depth direction");
    precision.formula = 0; precision.scale = 0;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(nominated_source(view.value, 0, normalized) && normalized.projection.depth_offset == 0 &&
        normalized.projection.depth_scale == .5 && normalized.projection.raw_scale == 1 && normalized.projection.raw_bias == 0,
      "NoTransform precision used ignored scale/bias fields");

    // Global FG tags are legal but have weaker frame association than an
    // explicit token. High-resolution-only tags must use the same contract.
    tag.type = 48;
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_source(view.value, 1, normalized) && !normalized.source_frame_explicit &&
        normalized.projection.supplied && normalized.resource.kind == sunshine_scene_depth::resource_kind::display_depth,
      "global high-resolution-only FG tag lost camera association");
    // Streamline recycles a token address for a later frame index; the same
    // index returns the same frame's token and is not a recycle.
    ++numeric;
    mint(token, &numeric);
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_snapshot(view.value, snapshot) && snapshot.tag_boundary &&
        snapshot.frame.kind == frame_identity_kind::unavailable && !nominated_source(view.value, 1, normalized),
      "global FG tag borrowed camera constants after their opaque token was recycled");
    call_v2_constants(constants, token.ref(), view);

    precision.formula = 1; precision.scale = 0;
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_snapshot(view.value, snapshot) && snapshot.tag_boundary && !snapshot.tags[1].supported &&
        !nominated_source(view.value, 1, normalized), "invalid precision inherited old usable FG depth");
    tag.base.next = nullptr;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_source(view.value, 1, normalized), "valid FG source did not recover after invalid precision");
    tag.resource_ptr = nullptr;
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_snapshot(view.value, snapshot) && snapshot.tag_boundary && !snapshot.tags[1].value.native_resource &&
        !nominated_source(view.value, 1, normalized), "FG untag retained prior source");
    tag.resource_ptr = &resource;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    result_v2 = -7;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(nominated_snapshot(view.value, snapshot) && !nomination_succeeded(view.value) &&
        !nominated_source(view.value, 1, normalized), "failed FG tag was promoted to a successful source");
    result_v2 = 0;

    options.mode = 0; require(set_options(view, options) == 0, "FG Off forwarding failed");
    require(query_frame_generation(UINT32_MAX, fg) && !fg.enabled, "FG Off retained enabled state");
    const auto last_fg_sequence = snapshot.sequence;
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_snapshot(view.value, snapshot) && snapshot.sequence > last_fg_sequence &&
        nominated_source(view.value, 1, normalized) && !normalized.frame_generation_input && !normalized.source_id,
      "FG Off did not switch standalone depth to its independent ordinary source role");
    sr_tag.base.next = nullptr;
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    call_v2_evaluate(0, token.ref(), sr_inputs, 2, &commands);
    require(nominated_source(view.value, 0, normalized) && !normalized.frame_generation_input,
      "FG Off did not restore ordinary SR source admission");
    options.mode = 2; options.generated_frames = 3; set_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled && fg.automatic && fg.generated_frames == 3,
      "Auto/MFG options were lost");
    for (const auto version : {4u, 5u}) {
      options.base.version = version;
      SetLastError(incoming_error);
      require(set_options(view, options) == 0 && fg_call.first == &view && fg_call.second == &options &&
          fg_call.incoming_error == incoming_error && GetLastError() == outgoing_error &&
          query_frame_generation(view.value, fg) && fg.enabled && fg.automatic && fg.generated_frames == 3,
        "append-only modern FG options lost prefix compatibility or forwarding");
    }
    options.mode = 3; options.generated_frames = 0; set_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled && fg.automatic && fg.generated_frames == 0,
      "Dynamic FG incorrectly required a fixed generated-frame count");
    presentation_scope::fg_view dynamic_policy;
    require(dynamic_policy.update(frame_generation_query_status::observed, fg).require_frame_generation &&
        dynamic_policy.generated_frames() == 0,
      "Dynamic FG with an unknown count selected ordinary depth routing");
    mint(token, &numeric); call_v2_constants(constants, token.ref(), view);
    call_v2_tag(view, &tag, 1, &commands);
    require(nominated_source(view.value, 1, normalized) && normalized.frame_generation_input,
      "Dynamic FG with an unknown count blocked valid tag-boundary depth");
    // A failed Off request is not evidence that the SDK disabled FG. Revoke
    // its former scope while keeping independently valid fallback eligible.
    options.mode = 0; result_v2 = -7;
    SetLastError(incoming_error);
    const auto failed_off_result = set_options(view, options);
    const auto failed_off_error = GetLastError();
    result_v2 = 0;
    frame_generation_query_status failed_off_status{};
    frame_generation_snapshot failed_off;
    require(failed_off_result == -7 && failed_off_error == outgoing_error &&
        !query_frame_generation(UINT32_MAX, failed_off, &failed_off_status) &&
        failed_off_status == frame_generation_query_status::ambiguous && !failed_off.known,
      "Failed FG Off call was treated as confirmed Off or retained the old known mode");
    auto fallback_policy = dynamic_policy.update(failed_off_status, failed_off);
    require(!fallback_policy.require_frame_generation && fallback_policy.exclude_unconfirmed_fg &&
        !fallback_policy.epoch && !dynamic_policy.generated_frames(),
      "Failed FG Off left a mandatory missing scope or authorized old FG reuse");
    fallback_policy = dynamic_policy.update(frame_generation_query_status::unavailable, {});
    require(!fallback_policy.require_frame_generation && fallback_policy.exclude_unconfirmed_fg && !fallback_policy.epoch,
      "An unavailable query revived the failed-Off scope or blocked fresh independent fallback");
    options.mode = 1; options.generated_frames = 1;
    require(set_options(view, options) == 0 && query_frame_generation(UINT32_MAX, fg) && fg.enabled,
      "A successful FG On did not recover after the failed Off");
    const auto restored_policy = dynamic_policy.update(frame_generation_query_status::observed, fg);
    require(restored_policy.require_frame_generation && !restored_policy.exclude_unconfirmed_fg &&
        restored_policy.epoch == fg.epoch && restored_policy.viewport == view.value && dynamic_policy.generated_frames() == 1,
      "Successful FG On did not restore exact SL priority after fallback");
    options.generated_frames = 0;
    options.mode = 4; set_options(view, options);
    require(!query_frame_generation(view.value, fg), "unknown FG mode retained a known enabled state");
    options.mode = 1; set_options(view, options);
    require(!query_frame_generation(view.value, fg), "fixed FG On admitted a zero generated-frame count");
    options.mode = 2; options.generated_frames = 3;
    // S13: a newer options version keeps its known mode/count prefix; a
    // chained structure is still refused.
    options.base.version = 6; set_options(view, options);
    require(query_frame_generation(view.value, fg) && fg.enabled && fg.mode == 2 && fg.generated_frames == 3,
      "a newer FG options version lost its known prefix");
    options.base.next = &options.base; set_options(view, options);
    require(!query_frame_generation(view.value, fg), "a newer FG options version with a chained structure retained a known state");
    options.base.next = nullptr;
    options.base.version = 3; set_options(view, options);
    auto second_view = viewport(12); set_options(second_view, options);
    frame_generation_query_status ambiguous_status;
    require(!query_frame_generation(UINT32_MAX, fg, &ambiguous_status) &&
        ambiguous_status == frame_generation_query_status::ambiguous && query_frame_generation(view.value, fg),
      "ambiguous FG viewports were silently combined");
    shutdown();
    const auto calls = fg_call.calls;
    require(set_options(view, options) == 0 && fg_call.calls == calls + 1 && !query_frame_generation(view.value, fg),
      "retained FG wrapper did not pass through after shutdown");
  }

  void test_frame_generation_cached_options_and_global_camera() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *cached_entry{};
    require(call_feature_function(1000, "slDLSSGSetOptions", cached_entry) == 0 && cached_entry == returned_feature_function,
      "pre-discovery SDK options cache did not retain the native entry");
    auto cached_options = reinterpret_cast<fg_function>(cached_entry);
    auto view = viewport(11); fg_options options;
    cached_options(view, options); // A past On cannot be invented by discovery.
    const auto hooks = minhook_ledger::mark();
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets) &&
        testing::discover_frame_generation_hooks(), "cached FG target discovery/detour failed");
    require_hooks(hooks, {presentation_v2_targets.constants, presentation_v2_targets.tag,
        presentation_v2_targets.tag_for_frame, presentation_v2_targets.evaluate, presentation_v2_targets.new_frame_token,
        presentation_v2_targets.get_feature_function, cached_entry},
      "v2 with every optional export and its FG options entry did not hook exactly those seven entries");
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
    const auto repeat = minhook_ledger::mark();
    require(testing::discover_frame_generation_hooks(), "repeat FG discovery attempted a duplicate hook");
    require_hooks(repeat, {}, "repeat FG discovery created another hook");
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
    require(!nominated_source(view.value, 0, normalized), "global FG tag invented missing camera constants");
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_snapshot(view.value, snapshot) && snapshot.tags[0].supported && snapshot.frame_correlated &&
        snapshot.frame.kind == frame_identity_kind::v2_constants_call &&
        nominated_source(view.value, 0, normalized) && normalized.projection.supplied &&
        normalized.proof == sunshine_scene_depth::state_proof::observed_nonzero &&
        !normalized.source_frame_explicit && !normalized.source_frame_has_numeric && normalized.source_frame_generation,
      "zero-base global FG tag lost its bounded constants-call association or claimed a numeric frame");
    require(normalized.observation_revision == snapshot.loss_revision &&
        normalized.observation_revision == depth_observation_revision(),
      "FG normalized depth omitted its immutable source observation revision");
    const auto first_camera_sequence = snapshot.camera_sequence;
    call_v2_constants(constants, token.ref(), view);
    tag_now();
    require(nominated_snapshot(view.value, snapshot) && snapshot.camera_sequence > first_camera_sequence &&
        nominated_source(view.value, 0, normalized), "new successful constants did not establish a new camera-call tuple");
    const auto valid_resource = resource;
    // S14: the zero header's members are version 1's, so its nonzero state
    // is a declaration, as a typed Resource's; only its 0 needs observation.
    resource.state = 0x20; call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(view.value, 0, normalized) && normalized.native_state == 0x20 &&
        normalized.proof == sunshine_scene_depth::state_proof::declared, "nonzero compatibility resource state was discarded (S14)");
    resource = valid_resource; resource.base = {nullptr, resource_guid, 1};
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(view.value, 0, normalized) && normalized.native_state == 0 &&
        normalized.proof == sunshine_scene_depth::state_proof::declared, "typed resource lost declared COMMON state");
    resource = valid_resource;
    const auto rejected_header = [&] {
      call_v2_constants(constants, token.ref(), view); tag_now();
      require(nominated_snapshot(view.value, snapshot) && !snapshot.tags[0].supported &&
          !nominated_source(view.value, 0, normalized), "malformed nested resource header entered FG capture");
      resource = valid_resource;
    };
    resource.base.version = 1; rejected_header(); // Empty GUID with nonzero version.
    resource.base.type.a = 1; rejected_header();
    resource.base.type = resource_guid; resource.base.version = 0; rejected_header();
    resource.base.next = &native; rejected_header();
    resource.type = 3; rejected_header();
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(view.value, 0, normalized), "valid zero-base resource failed to recover");
    auto high_resolution = tag; high_resolution.type = 48; high_resolution.lifecycle = 1;
    abi_v2::resource_tag mixed_tags[]{tag, high_resolution};
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(nominated_source(view.value, 0, normalized) && !nominated_source(view.value, 1, normalized),
      "mixed lifecycle batch lent synchronous camera evidence to an UntilPresent candidate");
    mixed_tags[1].lifecycle = 2;
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(nominated_source(view.value, 0, normalized) && !nominated_source(view.value, 1, normalized),
      "mixed lifecycle batch lent synchronous camera evidence to an UntilEvaluate candidate");
    mixed_tags[0].lifecycle = 1; mixed_tags[1].lifecycle = 0;
    call_v2_tag(view, mixed_tags, 2, &commands);
    require(!nominated_source(view.value, 0, normalized) && nominated_source(view.value, 1, normalized),
      "mixed lifecycle batch rejected its valid high-resolution OnlyValidNow candidate");
    tag.lifecycle = 1; tag_now();
    require(!nominated_source(view.value, 0, normalized), "untracked camera escaped synchronous OnlyValidNow scope");
    tag.lifecycle = 0;
    call_v2_framed_tag(token.ref(), view, &tag, 1, &commands);
    require(!nominated_source(view.value, 0, normalized), "untracked explicit token borrowed global camera-call evidence");
    call_v2_constants(constants, token.ref(), view);
    fixture_clock().hitch(270); tag_now();
    require(!nominated_source(view.value, 0, normalized), "stale global FG constants were accepted");
    call_v2_constants(constants, token.ref(), view); testing::lose_observation(); tag_now();
    require(!nominated_source(view.value, 0, normalized), "global FG constants crossed observation loss");
    const auto revision_before_reset = depth_observation_revision();
    constants.reset = 1; call_v2_constants(constants, token.ref(), view);
    require(depth_observation_revision() == revision_before_reset, "FG camera reset was recorded as a lost observation");
    tag_now();
    require(nominated_source(view.value, 0, normalized) && normalized.feedback.reset && normalized.projection.supplied &&
        normalized.observation_revision == depth_observation_revision() && normalized.frame_generation_input,
      "valid global FG reset frame lost current depth/projection");
    constants.reset = 2; call_v2_constants(constants, token.ref(), view); tag_now();
    require(!nominated_source(view.value, 0, normalized), "malformed reset flag entered global FG capture");
    constants = modern_camera(); constants.reset = 1; constants.common.camera_view_to_clip.m[0][0] = 0;
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(!nominated_source(view.value, 0, normalized), "reset admitted an invalid projection into global FG capture");
    // The camera is optional for depth: an invalid projection supplies no
    // metric scale, but never withholds the tagged depth itself.
    constants = modern_camera(); constants.common.camera_view_to_clip.m[0][0] = 0;
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(view.value, 0, normalized) && !normalized.projection.supplied,
      "invalid global camera projection supplied FG metric scale or withheld its depth");
    constants = modern_camera(); call_v2_constants(constants, token.ref(), view);
    result_v2 = -1; call_v2_constants(constants, token.ref(), view); result_v2 = 0; tag_now();
    require(!nominated_source(view.value, 0, normalized), "failed constants call retained an applicable camera tuple");
    call_v2_constants(constants, token.ref(), view); tag_now();
    require(nominated_source(view.value, 0, normalized), "global FG tuple did not recover after invalid/reset/lost constants");
    require(normalized.observation_revision == depth_observation_revision(),
      "fresh FG depth retained the pre-reset observation revision");
    mint(token, nullptr); tag_now();
    require(!nominated_source(view.value, 0, normalized), "observed token reuse retained an older untracked camera call");
    auto another_view = viewport(12); cached_options(another_view, options);
    call_v2_tag(another_view, &tag, 1, &commands);
    require(!nominated_source(another_view.value, 0, normalized), "global FG borrowed another viewport's constants");

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

  // The FG mode under lost metadata reads and a held metadata writer (lane P,
  // PD1-PD5, 11-01). Until the FG options store (presentation_scope_mode.inl)
  // the mode lived in the per-viewport camera records, which every camera,
  // tag and token call republishes: a read that lost that race was "busy",
  // and each reader had its own rule for it (the query reported busy, the SL
  // route took the viewport as FG's, DLSS-G's inputs found no FG scope), an
  // options call lost its write to a held camera writer, which left the
  // mode ambiguous until the game's next options call, and a fifth
  // viewport's camera evicted the FG viewport's mode. Each check loses a
  // metadata read where an FG read used one (testing::fail_metadata_reads).
  void test_frame_generation_mode_store() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(31), off_view = viewport(32);
    opaque_token token; std::uint32_t numeric = 7; mint(token, &numeric);
    const auto constants = modern_camera();
    // A camera publication exists, so a lost read is a busy one.
    call_v2_constants(constants, token.ref(), view);
    // Every check runs, so a run names each FG read a lost metadata read changes.
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    frame_generation_snapshot fg;
    frame_generation_query_status status{};
    // PD1: before any options call a lost read left the provider requiring FG
    // it could not name (a mono Present where Generic would supply).
    {
      presentation_scope::fg_view policy;
      testing::fail_metadata_reads(1);
      query_frame_generation(UINT32_MAX, fg, &status);
      testing::fail_metadata_reads(0);
      check(status == frame_generation_query_status::unavailable && !policy.update(status, fg).require_frame_generation,
        "A lost metadata read before any FG options required FG");
    }
    fg_options on; on.generated_frames = 3;
    require(set_options(view, on) == 0, "FG On options failed");
    testing::fail_metadata_reads(1);
    const bool queried = query_frame_generation(UINT32_MAX, fg, &status);
    testing::fail_metadata_reads(0);
    check(queried && status == frame_generation_query_status::observed && fg.enabled && fg.viewport == view.value &&
        fg.generated_frames == 3,
      "A lost metadata read hid the committed FG mode from the query");
    // This fixture's capture owner is inactive, so the enabled scope is silent.
    testing::fail_metadata_reads(1);
    const auto silent_route = testing::route_of(view.value);
    testing::fail_metadata_reads(0);
    check(silent_route == presentation_scope::fg_route::silent, "A lost metadata read made a silent FG scope own its viewport");
    int native{}, commands{};
    frame_generation::inputs inputs;
    inputs.command = reinterpret_cast<std::uint64_t>(&commands);
    inputs.depth.state = frame_generation::parameter::present;
    inputs.depth.native = reinterpret_cast<std::uint64_t>(&native);
    testing::fail_metadata_reads(1);
    const auto attempt = observe_frame_generation_inputs(inputs);
    testing::fail_metadata_reads(0);
    finish_frame_generation_inputs(attempt, false);
    check(attempt.depth != frame_generation::depth_outcome::no_fg_scope, "A lost metadata read left DLSS-G's inputs without their FG scope");
    fg_options on2; on2.generated_frames = 2;
    testing::lock_records();
    const auto held = set_options(view, on2);
    testing::unlock_records();
    check(held == 0 && query_frame_generation(UINT32_MAX, fg, &status) && status == frame_generation_query_status::observed &&
        fg.enabled && fg.generated_frames == 2,
      "An options call under a held metadata writer lost its write, which left the FG mode ambiguous");
    // The game repeats its options, which commits them in any case.
    require(set_options(view, on2) == 0, "FG On options failed");
    // Four other viewports' cameras replace the FG viewport's camera record:
    // the game renders it no longer, and its FG mode goes with the record, as
    // before the store (a guard). A new FG viewport's mode is then the only
    // one: with the retired viewport's mode kept, the any-viewport query was
    // ambiguous and depth revoked FG input (the store's first form, 11-01,
    // where every session timeline after the first with FG read ambiguous).
    for (std::uint32_t i = 0; i != frame_generation::viewport_slots; ++i) {
      auto other = viewport(40 + i);
      call_v2_constants(constants, token.ref(), other);
    }
    check(!query_frame_generation(view.value, fg), "A viewport whose camera record four other viewports replaced kept its FG mode");
    auto next_view = viewport(45);
    require(set_options(next_view, on) == 0, "FG On options failed");
    check(query_frame_generation(UINT32_MAX, fg, &status) && status == frame_generation_query_status::observed &&
        fg.viewport == next_view.value,
      "A retired viewport's FG mode left the new FG viewport's mode ambiguous");
    fg_options off; off.mode = 0; off.generated_frames = 0;
    require(set_options(off_view, off) == 0, "FG Off options failed");
    testing::fail_metadata_reads(1);
    const auto off_route = testing::route_of(off_view.value);
    testing::fail_metadata_reads(0);
    check(off_route == presentation_scope::fg_route::none, "A lost metadata read made an FG Off viewport's route FG's");
    require(failed.empty(), failed.c_str());
  }

  // One FG snapshot per Present and one process-wide FG view (lane P, PD6,
  // PD11, 11-01). Every consumer of a Present reads the snapshot taken at its
  // start, so an options call during the Present is the next Present's for
  // UI and depth alike; under an ambiguous read depth excludes FG input while
  // the UI keeps the last confirmed mode (owner ruling Q1); and a runtime
  // whose first Present comes during that ambiguity reads the same modes as
  // an older one (until 11-01 each exporter runtime's UI policy began
  // unknown, test_game3d_controls pinned it: "A new runtime inherited the old
  // runtime's mode").
  void test_present_fg_snapshot() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(51);
    fg_options on; on.generated_frames = 3;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    require(set_options(view, on) == 0, "FG On options failed");
    constexpr std::uint64_t older = 0x1000, newer = 0x2000;
    take_present_fg(older);
    const auto first = present_frame_generation(older);
    end_present_fg();
    require(first.swapchain == older && first.read == frame_generation_query_status::observed &&
        first.depth.require_frame_generation && first.generated_frames == 3 && first.ui.known && first.ui.enabled &&
        first.mode(view.value).enabled && first.mode(view.value).generated_frames == 3,
      "A Present's FG snapshot did not carry the committed FG On to depth, UI and the viewport's mode");
    take_present_fg(older);
    require(set_options(view, off) == 0, "FG Off options failed");
    const auto during = present_frame_generation(older);
    end_present_fg();
    require(during.depth.require_frame_generation && during.ui.enabled && during.mode(view.value).enabled,
      "An options call during a Present changed that Present's FG snapshot");
    take_present_fg(older);
    const auto after = present_frame_generation(older);
    end_present_fg();
    require(!after.depth.require_frame_generation && !after.depth.exclude_unconfirmed_fg && after.ui.known && !after.ui.enabled,
      "The next Present did not read the FG Off committed during the one before");
    require(set_options(view, on) == 0, "FG On options failed");
    take_present_fg(older); end_present_fg();
    result_v2 = -7;
    set_options(view, off);
    result_v2 = 0;
    take_present_fg(older);
    const auto ambiguous = present_frame_generation(older);
    end_present_fg();
    require(ambiguous.read == frame_generation_query_status::ambiguous && !ambiguous.depth.require_frame_generation &&
        ambiguous.depth.exclude_unconfirmed_fg && ambiguous.ui.known && ambiguous.ui.enabled,
      "An ambiguous read did not revoke depth's FG input while the UI kept its confirmed FG mode");
    take_present_fg(newer);
    const auto created = present_frame_generation(newer);
    end_present_fg();
    require(created.ui.known == ambiguous.ui.known && created.ui.enabled == ambiguous.ui.enabled &&
        created.ui.epoch == ambiguous.ui.epoch && created.ui.sequence == ambiguous.ui.sequence &&
        created.depth.require_frame_generation == ambiguous.depth.require_frame_generation &&
        created.depth.exclude_unconfirmed_fg == ambiguous.depth.exclude_unconfirmed_fg,
      "A runtime whose first Present came during an ambiguous FG read read other FG modes than an older runtime");
    const auto &fallback = present_frame_generation(older);
    require(fallback.swapchain == older && fallback.ui.enabled && fallback.depth.exclude_unconfirmed_fg,
      "A consumer with no FG snapshot of its Present did not read its own");
    end_present_fg();
  }

  // One read of the FG options store per Present snapshot, applied to the
  // process's one FG view in store order (lane L-fg, F1 and F70, 11-01).
  // Until then take_fg read the store twice outside fg_view_lock: an options
  // call between the two reads gave present_deferred's per-viewport modes
  // another mode than depth's and the UI's in one Present (F1), and a present
  // thread could apply its older read to the view after another present
  // thread's newer one, which the next ambiguous read then kept (F70).
  void test_present_fg_one_store_read() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(71);
    fg_options on; on.generated_frames = 3;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    constexpr std::uint64_t first_chain = 0x3000, second_chain = 0x4000;
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    // F1: an options call right after the snapshot's store read.
    require(set_options(view, on) == 0, "FG On options failed");
    std::atomic<bool> fired{};
    testing::set_fg_store_read_seam([&] { if (!fired.exchange(true)) set_options(view, off); });
    take_present_fg(first_chain);
    testing::set_fg_store_read_seam({});
    const auto split = present_frame_generation(first_chain);
    end_present_fg();
    check(fired.load() && split.depth.require_frame_generation && split.mode(view.value).known && split.mode(view.value).enabled,
      "F1: an options call during a Present's FG snapshot gave present_deferred another mode than depth and the UI");
    // F70: another present thread reads the store after the Off below and
    // updates the view while this thread is between its read and its update.
    require(set_options(view, on) == 0, "FG On options failed");
    take_present_fg(first_chain); end_present_fg();
    fired = false;
    std::atomic<bool> other_done{};
    std::thread other;
    testing::set_fg_store_read_seam([&] {
      if (fired.exchange(true)) return;
      set_options(view, off);
      other = std::thread([&] { take_present_fg(second_chain); end_present_fg(); other_done = true; });
      // With the view applied in store order the other thread waits for this
      // one's update; give it time to finish first otherwise.
      for (unsigned i = 0; i != 100 && !other_done.load(); ++i) Sleep(2);
    });
    take_present_fg(first_chain); end_present_fg();
    other.join();
    testing::set_fg_store_read_seam({});
    // The next read is ambiguous (a failed options call), so the UI keeps the
    // mode of the read applied last: the newer one, FG Off.
    result_v2 = -7;
    set_options(view, off);
    result_v2 = 0;
    take_present_fg(first_chain);
    const auto kept = present_frame_generation(first_chain);
    end_present_fg();
    check(kept.read == frame_generation_query_status::ambiguous && kept.ui.known && !kept.ui.enabled,
      "F70: a present thread's older FG read was applied to the FG view after another thread's newer one");
    require(failed.empty(), failed.c_str());
  }

  // A consumer with no FG snapshot of its Present reads its own for that
  // call only (lane L-fg, F68, 11-01). Until then the read became the
  // thread's snapshot, so where depth's present events frame no Present
  // (depth not registered) every later Present of the swapchain on that
  // thread read the first one: FG turned on later stayed unknown to the UI,
  // and present_fg_fallbacks rose once.
  void test_present_fg_fallback_one_shot() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(81);
    fg_options on; on.generated_frames = 2;
    constexpr std::uint64_t unframed = 0x5000;
    const auto counted = present_fg_fallbacks();
    const auto first = present_frame_generation(unframed);
    require(set_options(view, on) == 0, "FG On options failed");
    const auto second = present_frame_generation(unframed);
    require(!first.ui.known && second.ui.known && second.ui.enabled && second.depth.require_frame_generation &&
        present_fg_fallbacks() == counted + 2,
      "F68: a consumer without an FG snapshot of its Present read an earlier Present's fallback snapshot");
  }

  // A restart (initialize after device recreation, which clears the FG
  // options store and starts a new epoch) resets the process's FG view
  // (lane L-fg, F26, 11-01). Until then the view kept its UI projection
  // through every later unavailable read, so FG On confirmed before the
  // restart stayed the UI's mode for the rest of the process in a game that
  // sets its options only when it enables FG.
  void test_fg_view_restart() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(91);
    fg_options on; on.generated_frames = 3;
    constexpr std::uint64_t chain = 0x6000;
    require(set_options(view, on) == 0, "FG On options failed");
    take_present_fg(chain);
    const auto before = present_frame_generation(chain);
    end_present_fg();
    require(before.ui.known && before.ui.enabled && before.depth.require_frame_generation, "FG On did not reach the FG view");
    initialize(nullptr);
    take_present_fg(chain);
    const auto after = present_frame_generation(chain);
    end_present_fg();
    require(after.read == frame_generation_query_status::unavailable && !after.ui.known &&
        !after.depth.require_frame_generation && !after.depth.exclude_unconfirmed_fg,
      "F26: the FG view kept a mode from before a restart");
  }

  // A viewport's confirmed Off binds no FG scope (lane L-fg, R5-3, 11-01):
  // beside another viewport's enabled options the any-viewport read names
  // that viewport, so its view breaks a tie of two active SR views (owner
  // ruling R5); two enabled viewports stay ambiguous; with none enabled the
  // read is the Off of the lowest viewport id (test_all_off_read_is_stable).
  // Until then any two viewports' known options made
  // the read ambiguous, so a game that sets Off on its second SR viewport
  // every frame had no confirmed FG scope while both rendered.
  void test_any_viewport_read_beside_off() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto main_view = viewport(101), second_view = viewport(102);
    fg_options on; on.generated_frames = 2;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    constexpr std::uint64_t chain = 0x7000;
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    frame_generation_snapshot fg;
    frame_generation_query_status status{};
    require(set_options(main_view, on) == 0 && set_options(second_view, off) == 0, "FG options failed");
    check(query_frame_generation(UINT32_MAX, fg, &status) && status == frame_generation_query_status::observed && fg.enabled &&
        fg.viewport == main_view.value,
      "R5-3: another viewport's FG Off made the enabled viewport's FG scope ambiguous");
    take_present_fg(chain);
    const auto bound = present_frame_generation(chain);
    end_present_fg();
    check(bound.depth.require_frame_generation && bound.depth.tie_epoch && bound.depth.tie_viewport == main_view.value,
      "R5-3: the FG viewport beside another viewport's Off broke no tie of two active views");
    require(set_options(second_view, on) == 0, "FG On options failed");
    check(!query_frame_generation(UINT32_MAX, fg, &status) && status == frame_generation_query_status::ambiguous,
      "Two viewports with FG enabled read as one FG scope");
    require(set_options(main_view, off) == 0, "FG Off options failed");
    check(query_frame_generation(UINT32_MAX, fg, &status) && fg.enabled && fg.viewport == second_view.value,
      "R5-3: an FG viewport switch with the old viewport's Off stayed ambiguous");
    require(set_options(second_view, off) == 0, "FG Off options failed");
    check(query_frame_generation(UINT32_MAX, fg, &status) && status == frame_generation_query_status::observed && fg.known &&
        !fg.enabled && fg.viewport == main_view.value,
      "R5-3: two viewports' FG Off did not read as the lowest viewport's FG Off");
    require(failed.empty(), failed.c_str());
  }

  // Two viewports' FG Off set every frame, in either order (R5-3's fix
  // review, 11-01): the any-viewport read names the same Off whatever the
  // calls' order around each Present's read, so the UI's FG viewport, the
  // key of its source qualification and detection scope where no SL UI
  // tag names one (game3d_ui_input_provider), stays. Until then the read was
  // the newest Off, whose sequence each call rewrites: it named either
  // viewport, and each flip reset the UI's qualification.
  void test_all_off_read_is_stable() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto first = viewport(111), second = viewport(112);
    fg_options off; off.mode = 0; off.generated_frames = 0;
    constexpr std::uint64_t chain = 0x7100;
    std::uint32_t named{};
    std::string failed;
    for (unsigned frame = 0; frame != 6 && failed.empty(); ++frame) {
      const bool swapped = frame % 2 != 0;
      require(set_options(swapped ? second : first, off) == 0 && set_options(swapped ? first : second, off) == 0,
        "FG Off options failed");
      take_present_fg(chain);
      const auto taken = present_frame_generation(chain);
      end_present_fg();
      if (taken.read != frame_generation_query_status::observed || !taken.ui.known || taken.ui.enabled) {
        failed = "two viewports' FG Off did not read as FG Off";
      } else if (!named) {
        named = taken.ui.viewport;
      } else if (taken.ui.viewport != named) {
        failed = "R5-3: the UI's FG viewport moved between two viewports' FG Off with the calls' order (" +
          std::to_string(named) + " then " + std::to_string(taken.ui.viewport) + " at frame " + std::to_string(frame) + ")";
      }
    }
    require(failed.empty(), failed.c_str());
  }

  // Two options calls' FG transitions follow their store commits (lane
  // L-fg, F81, 11-01). An Off committed, then, before its own transition
  // entries ran, an On on another thread committed and restored the FG role
  // (frame_generation_restored found no retirement yet); the Off's
  // retire_source then retired the role while the store said On, and only
  // the next FG Off-to-On toggle restored it: every later On call repeats
  // the mode, so none restores. The FG scope was then never live for the
  // whole FG-on period (no presenting window, generated Presents mono after
  // a hitch, DLSS-G's own copy and the deferred tag copy both made).
  void test_fg_transition_order() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(111);
    fg_options on; on.generated_frames = 1;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    // The capture owner keeps FG retirements while it is requested.
    depth_capture::fixture::reset_owner();
    require(set_options(view, on) == 0, "FG On options failed");
    std::atomic<bool> fired{}, other_done{};
    std::thread other;
    testing::set_fg_transition_seam([&] {
      if (fired.exchange(true)) return;
      other = std::thread([&] { set_options(view, on); other_done = true; });
      // With ordered transitions the other call waits for this one's.
      for (unsigned i = 0; i != 100 && !other_done.load(); ++i) Sleep(2);
    });
    set_options(view, off);
    other.join();
    testing::set_fg_transition_seam({});
    frame_generation_snapshot fg;
    const bool on_now = query_frame_generation(view.value, fg) && fg.enabled;
    const bool retired = on_now && depth_capture::frame_generation_retired(fg.epoch, view.value, 0);
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(on_now && !retired,
      "F81: an FG Off's retirement landed after a newer On's restore and left the FG role retired while FG is on");
  }

  // FG Off retires the viewport's FG role once (lane L-fg, F3, 11-01): an
  // Off that repeats the committed one, as games repeat their options every
  // frame, retires nothing new; a failed call after a confirmed Off still
  // does. Until then each repeated Off ran retire_source again under the
  // capture lock, which also moved the retirement's serial to that call.
  void test_repeated_fg_off_retires_once() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(121);
    fg_options on; on.generated_frames = 1;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    depth_capture::fixture::reset_owner();
    frame_generation_snapshot fg;
    const auto committed = [&] { return query_frame_generation(view.value, fg) ? fg.sequence : 0; };
    require(set_options(view, on) == 0 && set_options(view, off) == 0, "FG options failed");
    const auto first_off = committed();
    require(set_options(view, off) == 0, "FG Off options failed");
    const auto repeated_off = committed();
    require(set_options(view, on) == 0, "FG On options failed");
    const auto epoch = fg.epoch;
    const bool first_retired = depth_capture::frame_generation_retired(epoch, view.value, first_off);
    const bool repeat_retired = depth_capture::frame_generation_retired(epoch, view.value, repeated_off);
    // A failed call after a confirmed Off retires again.
    require(set_options(view, off) == 0, "FG Off options failed");
    result_v2 = -7;
    set_options(view, off);
    result_v2 = 0;
    require(set_options(view, on) == 0, "FG On options failed");
    const bool failed_retired = depth_capture::frame_generation_retired(epoch, view.value, committed() - 1);
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(first_off && repeated_off > first_off && first_retired && !repeat_retired && failed_retired,
      "F3: a repeated FG Off retired the FG role again, or the first Off or a failed call after it retired nothing");
  }

  // Dynamic FG without a generated count (mode 3, numFramesToGenerate 0) is
  // one generated Present per real frame for the deferral clock and for
  // DLSS-G's evaluation window (lane L-fg, F4, 11-01; the fix review's
  // F4 row). Until then present_deferred took the raw count (0) for a
  // deferral's window, so a frame's deferred FG copy lapsed at the earlier
  // frame's generated Present after a hitch of the source age, and DLSS-G's
  // evaluation kept no window, so after that hitch its evaluation of the
  // frame no longer made the copy due at the frame's generated Present.
  // The seam rows check only generated_presents(): reverting either call
  // site (present_deferred's mode, observe_frame_generation_inputs's fg) to
  // the raw count fails here.
  void test_dynamic_fg_deferral_window() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(141);
    fg_options dynamic; dynamic.mode = 3; dynamic.generated_frames = 0;
    require(set_options(view, dynamic) == 0, "Dynamic FG options failed");
    depth_capture::fixture::reset_owner();
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    depth_capture::input candidate;
    candidate.provider = sunshine_scene_depth::provider_kind::streamline;
    candidate.epoch = 1; candidate.sequence = 1; candidate.viewport = view.value;
    candidate.frame_generation_input = true; candidate.source_id = sunshine_scene_depth::fg_source_id(view.value);
    struct no_interface final : IUnknown {
      HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
      ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
      ULONG STDMETHODCALLTYPE Release() override { return 1; }
    } queue, list;
    constexpr std::uint64_t chain = 0x8100;
    const auto present = [&] {
      take_present_fg(chain);
      present_deferred(chain, reinterpret_cast<std::uint64_t>(&queue), reinterpret_cast<std::uint64_t>(&list));
      end_present_fg();
    };
    // The deferral clock: the frame's tag call deferred its copy a source
    // age ago, and one Present came since (the earlier frame's generated
    // one): its frame's Present is still to come.
    auto before = testing::deferrals_now();
    testing::defer_depth(view.value, candidate, sunshine_freshness::source_age_ms + 50);
    depth_capture::fixture::presents_since(1);
    present();
    auto after = testing::deferrals_now();
    check(after.pending == 1 && after.lapsed == before.lapsed,
      "F4: a Dynamic FG frame's deferred copy lapsed at the earlier frame's generated Present after a hitch (present_deferred read Dynamic's 0 generated frames)");
    // Past that one Present it lapses.
    depth_capture::fixture::presents_since(2);
    present();
    check(testing::deferrals_now().pending == 0, "F4: a Dynamic FG frame's deferred copy outlived its window");
    // DLSS-G's evaluation window: the frame's tag call defers its copy,
    // DLSS-G evaluates the frame at that Present, and a hitch of the source
    // age passes before the frame's generated Present, at which the copy
    // is due (its frame's Present: no source here, so it is refused).
    before = testing::deferrals_now();
    testing::defer_depth(view.value, candidate);
    const auto attempt = observe_frame_generation_inputs({});
    finish_frame_generation_inputs(attempt, true);
    fixture_clock().hitch(sunshine_freshness::source_age_ms + 30);
    depth_capture::fixture::presents_since(1);
    present();
    after = testing::deferrals_now();
    check(after.pending == 0 && after.depth_refused == before.depth_refused + 1 && after.lapsed == before.lapsed,
      "F4: after a hitch DLSS-G's evaluation of a Dynamic FG frame did not make its deferred copy due at its generated Present (its window read Dynamic's 0 generated frames)");
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(failed.empty(), failed.c_str());
  }

  // A frame call that makes no copy still ends an earlier frame's
  // presenting window (frame_presenting) at its frame's Present or call
  // (lane L-fg, F2, 11-01): a deferred FG tag copy refused at its frame's
  // Present (DLSS-G's evaluation of the frame stood aside for it and began
  // nothing), and DLSS-G's evaluation that makes no copy attempt (a
  // malformed subrect). Until then neither did, so where DLSS-G presented
  // fewer of the earlier frame's generated Presents than declared, a hitch
  // of the source age before the frame's Present held the earlier frame's
  // copy past the source age as the frame's own.
  void test_refused_frame_calls_end_window() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(131);
    depth_capture::fixture::reset_owner();
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    // A frame's deferred FG depth copy with no source to copy: refused at
    // the first Present after a game list executed following its tag call.
    depth_capture::input candidate;
    candidate.provider = sunshine_scene_depth::provider_kind::streamline;
    candidate.epoch = 1; candidate.sequence = 1; candidate.viewport = view.value;
    candidate.frame_generation_input = true; candidate.source_id = sunshine_scene_depth::fg_source_id(view.value);
    testing::defer_depth(view.value, candidate);
    depth_capture::capture_queues.game_execution_count.fetch_add(1);
    depth_capture::fixture::presents_since(3);
    // A queue and list that implement no interface: nothing reads them.
    struct no_interface final : IUnknown {
      HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
      ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
      ULONG STDMETHODCALLTYPE Release() override { return 1; }
    } queue, list;
    constexpr std::uint64_t chain = 0x8000;
    take_present_fg(chain);
    present_deferred(chain, reinterpret_cast<std::uint64_t>(&queue), reinterpret_cast<std::uint64_t>(&list));
    end_present_fg();
    check(depth_capture::scope.newest_evaluation_present == depth_capture::present_ordinal(),
      "F2: a deferred FG tag copy refused at its frame's Present ended no earlier frame's presenting window");
    // DLSS-G's evaluation with a malformed depth subrect.
    fg_options on; on.generated_frames = 3;
    require(set_options(view, on) == 0, "FG On options failed");
    depth_capture::fixture::presents_since(1);
    int native{}, commands{};
    frame_generation::inputs inputs;
    inputs.command = reinterpret_cast<std::uint64_t>(&commands);
    inputs.depth.state = frame_generation::parameter::present;
    inputs.depth.native = reinterpret_cast<std::uint64_t>(&native);
    inputs.depth.area_valid = false;
    const auto attempt = observe_frame_generation_inputs(inputs);
    finish_frame_generation_inputs(attempt, true);
    check(attempt.depth == frame_generation::depth_outcome::subrect &&
        depth_capture::scope.newest_evaluation_present == depth_capture::present_ordinal(),
      "F2: DLSS-G's evaluation without a copy attempt ended no earlier frame's presenting window");
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(failed.empty(), failed.c_str());
  }

  // DLSS-G's evaluation that stands aside for its frame's FG tag copy
  // recorded at the tag call anchors that frame's presenting window at the
  // next Present (92b98375's anchor (b): frame_generation_evaluated through
  // tags_supply_frame's recorded answer), with no submission of the copy on
  // the reading queue to anchor it (anchor (a), which PD9 limits to the
  // reading queue: FG moved Present to DLSS-G's queue). No session row
  // reaches that case: the fixture presents on the queue that runs the
  // tag's list, where anchor (a) fires as well (lane L-fg, F88, 11-01). The
  // earlier frame's generated Presents came after this frame's first stamp.
  void test_dlssg_anchor_for_recorded_tag_copy() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(141);
    opaque_token token; std::uint32_t numeric = 7; mint(token, &numeric);
    call_v2_constants(modern_camera(), token.ref(), view);
    fg_options on; on.generated_frames = 3;
    require(set_options(view, on) == 0, "FG On options failed");
    frame_generation_snapshot fg;
    require(query_frame_generation(view.value, fg) && fg.enabled, "FG On was not committed");
    depth_capture::fixture::reset_owner();
    depth_capture::capture_queues.queues[0] = std::make_unique<depth_capture::queue_state>();
    // The frame's first FG stamp, then two of the earlier frame's generated
    // Presents.
    depth_capture::scope.frame_generation_present = depth_capture::present_ordinal();
    depth_capture::scope.frame_generation_anchor = 0;
    depth_capture::scope.presenting_generated = 3;
    depth_capture::fixture::presents_since(2);
    testing::remember_fg_tag_copy(fg.epoch, view.value, UINT64_MAX / 2);
    int native{}, commands{};
    frame_generation::inputs inputs;
    inputs.command = reinterpret_cast<std::uint64_t>(&commands);
    inputs.depth.state = frame_generation::parameter::present;
    inputs.depth.native = reinterpret_cast<std::uint64_t>(&native);
    const auto attempt = observe_frame_generation_inputs(inputs);
    finish_frame_generation_inputs(attempt, true);
    const bool anchored = depth_capture::scope.frame_generation_anchor == depth_capture::present_ordinal();
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(attempt.depth == frame_generation::depth_outcome::tag_copied && anchored,
      "F88: DLSS-G's evaluation that stood aside for its frame's recorded FG tag copy did not anchor the frame's presenting window");
  }

  // Direct NGX's FG ownership against the real FG options store (lane L-fg,
  // F96, 11-01): enabled options whose scope is live own a direct NGX
  // evaluation's depth (no copy), a silent scope and FG Off do not, and
  // another viewport's confirmed Off leaves the FG viewport's ownership
  // (R5-3). Until then only a stub (the call-trace build) and a direct
  // predicate call (the seam rows) were tested; the store read at the
  // evaluation was compiled out of every test build.
  void test_ngx_fg_ownership() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(151), other = viewport(152);
    fg_options on; on.generated_frames = 1;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    require(set_options(view, on) == 0, "FG On options failed");
    frame_generation_snapshot fg;
    require(query_frame_generation(view.value, fg) && fg.enabled, "FG On was not committed");
    depth_capture::fixture::reset_owner();
    check(!sunshine_ngx::frame_generation_owns_evaluation(), "A silent FG scope owned NGX's depth");
    // The scope's FG copy accepted now: the scope is live.
    auto &scope_view = depth_capture::evaluations[depth_capture::provider_index(sunshine_scene_depth::provider_kind::streamline)].views[0];
    scope_view = {fg.epoch, 1, fixture_clock().now(), sunshine_scene_depth::fg_source_id(view.value), view.value};
    scope_view.accepted_tick = fixture_clock().now(); scope_view.accepted_sequence = 1;
    check(sunshine_ngx::frame_generation_owns_evaluation(), "A live FG scope did not own NGX's depth");
    require(set_options(other, off) == 0, "FG Off options failed");
    check(sunshine_ngx::frame_generation_owns_evaluation(), "Another viewport's FG Off took NGX's depth from the live FG scope");
    require(set_options(view, off) == 0, "FG Off options failed");
    check(!sunshine_ngx::frame_generation_owns_evaluation(), "FG Off still owned NGX's depth");
    depth_capture::fixture::reset_owner();
    depth_capture::requested = false;
    require(failed.empty(), ("F96: " + failed).c_str());
  }

  // DLSS-G's evaluation supply (lane P, PD10 and round 1's C1, 11-01). A
  // refused FG tag boundary reads it once for both of its decisions: whether
  // a command-less boundary withdraws nothing, and whether its frame is missed
  // (nominate_depth_source). Until 11-01 the first read was a try-lock, so a
  // supply lock held by another thread (DLSS-G's evaluation remembering its
  // copy, a deferral, a UI tag) withdrew the FG nomination while the second,
  // blocking read found the supply and left the frame unmissed. And a supply
  // from before an FG Off and On supplied the new FG period for up to
  // fg_silent_ms (soak seed 4547e4874aef62bb's frames 20-22 against 79).
  void test_evaluation_supply_read() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "FG source hook install failed");
    returned_feature_function = reinterpret_cast<void *>(&fake_fg_options);
    void *function{};
    require(call_feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, "FG getter returned no forwarding wrapper");
    auto set_options = reinterpret_cast<fg_function>(function);
    auto view = viewport(61);
    fg_options on; on.generated_frames = 1;
    fg_options off; off.mode = 0; off.generated_frames = 0;
    require(set_options(view, on) == 0, "FG On options failed");
    frame_generation_snapshot fg;
    require(query_frame_generation(view.value, fg) && fg.enabled, "FG On was not committed");
    // FG has been on for 500 ms; DLSS-G copied this scope 100 ms ago.
    testing::age_frame_generation(500);
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    testing::supply_evaluation(fg.epoch, view.value, fixture_clock().now() - 100);
    check(testing::evaluation_supplied(fg.epoch, view.value), "A fresh supply of the FG period did not supply its scope");
    {
      testing::hold_evaluation_supply();
      std::atomic<bool> started{};
      bool read{};
      std::thread boundary([&] {
        started.store(true, std::memory_order_release);
        read = testing::evaluation_supplied(fg.epoch, view.value);
      });
      while (!started.load(std::memory_order_acquire)) SwitchToThread();
      Sleep(20);
      testing::release_evaluation_supply();
      boundary.join();
      check(read, "PD10: an FG tag boundary read DLSS-G's fresh supply as absent while another thread held its lock");
    }
    require(set_options(view, off) == 0, "FG Off options failed");
    require(set_options(view, on) == 0, "FG On options failed");
    check(!testing::evaluation_supplied(fg.epoch, view.value),
      "C1: DLSS-G's supply from before an FG Off and On supplied the new FG period");
    testing::supply_evaluation(fg.epoch, view.value, fixture_clock().now());
    check(testing::evaluation_supplied(fg.epoch, view.value), "DLSS-G's supply of the new FG period did not supply it");
    require(failed.empty(), failed.c_str());
  }

  // A call's camera read that lost the race with the writers (lane P,
  // 11-01; the PD design's seen-not-fixed race). read_metadata fails only
  // when every attempt failed: the published camera and tag metadata changed
  // under it, or the slot's pin count changed between its load and its pin
  // (other readers, a writer's base pin; observer_snapshot.h), which a burst
  // of calls on other threads can do (testing::fail_metadata_reads). Until 11-01 such
  // a read made an SL evaluation's snapshot busy (capture_evaluation, which
  // DLSS-G's camera evidence also reads: scope_changed, no copy) and dropped a
  // whole depth tag boundary (capture_depth_tag). The call's read is now
  // retried a bounded number of times; one that keeps losing still refuses.
  void test_camera_read_retry() {
    fixture cleanup;
    source_setting = "1"; initialize(nullptr);
    require(testing::install(testing::abi::v2_7_30, presentation_v2_targets), "Camera read hook install failed");
    std::string failed;
    const auto check = [&](bool passed, const char *what) { if (!passed) failed += std::string(failed.empty() ? "" : "; ") + what; };
    auto view = viewport(3), standalone = viewport(0);
    auto constants = modern_camera();
    opaque_token token;
    const std::uint32_t number = 7;
    mint(token, &number);
    int native{}, native_standalone{}, commands{};
    auto resource = modern_resource(&native);
    auto tag = depth_tag_for(resource);
    const base_structure *inputs[] {&view.base};
    call_v2_constants(constants, token.ref(), view);
    call_v2_framed_tag(token.ref(), view, &tag, 1, nullptr);
    evaluation_snapshot snapshot;
    testing::fail_metadata_reads(1);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    testing::fail_metadata_reads(0);
    check(nominated_snapshot(view.value, snapshot) && snapshot.status != evidence_status::busy &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native),
      "An SL evaluation whose camera read lost the race once read no camera evidence (busy)");
    auto standalone_resource = modern_resource(&native_standalone);
    standalone_resource.base = {}; standalone_resource.width = standalone_resource.height = 0; standalone_resource.state = 0x60;
    auto depth = depth_tag_for(standalone_resource); depth.lifecycle = 0; depth.area = {0, 0, 2228, 1253};
    testing::fail_metadata_reads(1);
    call_v2_tag(standalone, &depth, 1, &commands);
    testing::fail_metadata_reads(0);
    check(nominated_snapshot(standalone.value, snapshot) && snapshot.tag_boundary &&
        snapshot.tags[0].value.native_resource == reinterpret_cast<std::uintptr_t>(&native_standalone),
      "A depth tag boundary whose camera read lost the race once was dropped");
    // Bounded: a read that keeps losing refuses as before.
    testing::fail_metadata_reads(1000);
    call_v2_evaluate(0, token.ref(), inputs, 1, nullptr);
    testing::fail_metadata_reads(0);
    check(nominated_snapshot(view.value, snapshot) && snapshot.status == evidence_status::busy,
      "An SL evaluation whose camera read kept losing was not refused as busy");
    require(failed.empty(), failed.c_str());
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
    for (const auto &entry : std::initializer_list<std::tuple<unsigned, unsigned, unsigned, abi>>{
        {1, 0, 0, abi::legacy_no_state}, {1, 0, 1, abi::legacy_no_state}, {1, 0, 2, abi::legacy_no_state},
        {1, 0, 3, abi::legacy},
        {1, 1, 0, abi::legacy}, {1, 1, 1, abi::legacy}, {1, 1, 2, abi::legacy},
        {2, 0, 0, abi::v2}, {2, 1, 0, abi::v2}, {2, 6, 10, abi::v2}, {2, 7, 30, abi::v2},
        {2, 9, 0, abi::v2}, {2, 12, 0, abi::v2}, {2, 14, 1, abi::v2}}) {
      const auto [major, minor, patch, expected] = entry;
      wchar_t dotted[64]{}, comma[64]{};
      std::swprintf(dotted, std::size(dotted), L"%u.%u.%u.0", major, minor, patch);
      std::swprintf(comma, std::size(comma), L"%u,%u,%u,0", major, minor, patch);
      auto value = make(major, minor, patch, dotted, comma);
      require(versioning::classify(value) == expected, "matching dotted/comma version rejected by ABI family");
      auto fixed_only = make(major, minor, patch, L"", L"");
      require(versioning::classify(fixed_only) == expected, "consistent fixed-only ABI family rejected");
    }
    auto v1 = make(1, 1, 1, L"1.1.1.0", L"1.1.1.0");
    require(versioning::classify(v1) == abi::legacy && versioning::private_state_v1_1_1(v1),
      "validated v1 fixed version or exact private-state adapter rejected");
    auto revision = make(2, 6, 10, L"2.6.10.7", L"2,6,10,7");
    revision.file_ls = revision.product_ls = MAKELONG(7, 10);
    require(versioning::classify(revision) == abi::v2, "compatible build revision rejected");
    auto legacy = make(1, 0, 0, L"1.1.1.0", L"1.1.1.0");
    require(versioning::classify(legacy) == abi::legacy && versioning::private_state_v1_1_1(legacy),
      "pinned v1 SDK fixed-resource quirk rejected");
    require(std::strstr(legacy.reason, "quirk") != nullptr, "v1 quirk recognition not diagnosed");
    for (const auto &entry : std::initializer_list<std::array<unsigned, 2>>{
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}}) {
      wchar_t label[64]{};
      std::swprintf(label, std::size(label), L"1.%u.%u.0", entry[0], entry[1]);
      auto quirk = make(1, 0, 0, label, label);
      require(versioning::classify(quirk) ==
          (entry[0] == 0 && entry[1] < 3 ? abi::legacy_no_state : abi::legacy),
        "surveyed legacy resource-version quirk selected the wrong resource layout");
      require(versioning::private_state_v1_1_1(quirk) == (entry[0] == 1 && entry[1] == 1),
        "legacy resource-version quirk broadened private-state compatibility");
    }
    for (const auto *text : {L"1.1.1.0-beta", L"2.7.30.0", L"1.1.1", L"1.1.1.0.0", L"1.1.1.-1"}) {
      auto rejected = make(1, 0, 0, text, text);
      require(versioning::classify(rejected) == abi::unsupported, "unknown legacy string pair accepted");
    }
    for (const auto *text : {L"2.0.1.0", L"2,1,0,0"}) {
      auto early_v2 = make(2, 0, 0, text, text);
      require(versioning::classify(early_v2) == abi::v2 && !versioning::private_state_v1_1_1(early_v2),
        "surveyed early typed SDK resource-version quirk rejected or enabled legacy private state");
      early_v2.strings_complete = false;
      require(versioning::classify(early_v2) == abi::unsupported,
        "partial translation enabled the early typed SDK resource-version quirk");
    }
    // Disagreeing identities of one typed family: the 2.x ABI, an inexact
    // version whose candidates are every identity it names.
    const auto inexact = [&](versioning::information value, std::initializer_list<versioning::number> expected) {
      const auto family = versioning::family(value);
      bool all = family.count == expected.size();
      for (const auto &v : expected) {
        bool found = false;
        for (unsigned i = 0; i != family.count; ++i) found = found || family.values[i] == v;
        all = all && found;
      }
      return versioning::classify(value) == abi::v2 && std::strcmp(value.reason, "v2 family, inexact version") == 0 &&
        all && !versioning::private_state_v1_1_1(value);
    };
    require(inexact(make(2, 0, 0, L"2.0.1.0", L"2.0.1.0"), {{2, 0, 0, 0}, {2, 0, 1, 0}}),
      "the fixed-2.0.0.0 release identity was not the family {2.0.0, strings}");
    for (const auto *text : {L"2.1.1.0", L"2.6.10.0", L"2.14.1.0", L"2.5.0.0"}) {
      versioning::number named{};
      require(versioning::parse(text, named) && inexact(make(2, 0, 0, text, text), {{2, 0, 0, 0}, named}),
        "a fixed 2.0.0.0 with other 2.x strings was not an inexact 2.x family");
    }
    require(inexact(make(2, 3, 0, L"2.4.1.0", L"2.4.1.0"), {{2, 3, 0, 0}, {2, 4, 1, 0}}),
      "fixed 2.3.0.0 with 2.4.1.0 strings was not an inexact 2.x family");
    auto fixed_pair = make(2, 4, 0, L"", L"");
    fixed_pair.product_ms = MAKELONG(5, 2);
    require(inexact(fixed_pair, {{2, 4, 0, 0}, {2, 5, 0, 0}}), "disagreeing 2.x fixed file/product were not a family");
    for (const auto *text : {L"3.0.0.0", L"1.1.1.0"}) {
      auto mixed = make(2, 0, 0, text, text);
      require(versioning::classify(mixed) == abi::unsupported && !versioning::family(mixed).count,
        "mixed Streamline ABI majors were accepted as one family");
    }
    auto mixed_fixed = make(1, 1, 1, L"2.0.0.0", L"2.0.0.0");
    require(versioning::classify(mixed_fixed) == abi::unsupported, "a 1.x fixed version with 2.x strings was accepted");
    auto fixed_major = make(2, 4, 0, L"", L"");
    fixed_major.product_ms = MAKELONG(0, 3);
    require(versioning::classify(fixed_major) == abi::unsupported, "a 2.x/3.x fixed pair was accepted");
    for (const auto major : {0u, 3u, 65535u}) {
      auto future = make(major, 0, 0, L"", L"");
      require(versioning::classify(future) == abi::unsupported,
        "unknown ABI major inferred from version metadata alone");
    }
    for (const auto &entry : std::initializer_list<std::array<unsigned, 3>>{
        {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 2}, {2, 6, 10}}) {
      auto public_only = make(entry[0], entry[1], entry[2], L"", L"");
      require(!versioning::private_state_v1_1_1(public_only),
        "public ABI compatibility enabled an unverified private resource-state encoding");
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
    require(inexact(bad_known, {{2, 7, 30, 0}, {2, 7, 31, 0}}), "contradictory 2.x fixed/string identity was not inexact");
    for (const auto *text : {L"2.6.10", L"2,6.10,0", L"2.6.10.65536", L"2.6.10.4294967296",
        L"2.6.10.-1", L"2.6.10.0.0", L"2.6.10.0-beta"}) {
      auto malformed = make(2, 6, 10, text, text);
      require(versioning::classify(malformed) == abi::unsupported, "malformed typed ABI version accepted");
    }
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
  source_setting = "0";
  call_trace_setting = "1";
  initialize(nullptr);
  require(sunshine_upscaler_trace::enabled() && !source_enabled(),
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
  require(testing::counts().evaluations == 0,
    "Trace-only calls entered source or content diagnostics");
  shutdown();
  const auto stopped = sunshine_upscaler_trace::testing::counts().streamline;
  if (modern) call_v2_evaluate(0, token.ref(), nullptr, 0, nullptr);
  else call_v1_evaluate(nullptr, 0, 1, 0);
  require(sunshine_upscaler_trace::testing::counts().streamline == stopped,
    "Disabled call trace observed retained pass-through hook");
  call_trace_setting = nullptr;
  source_setting = "1";
}

int main(int argc, char **argv) {
  try {
    auto &clock = sunshine_game3d_fixture::fixture_clock();
    sunshine_game3d_fixture::take_wall_clock_option(argc, argv);
    require(clock.bind(GetModuleHandleW(nullptr)), "the probe view's test clock exports are missing");
    clock.engage();
    if (argc == 2 && std::strcmp(argv[1], "--standalone-depth-tags") == 0) {
      test_standalone_depth_tags();
      std::puts("PASS standalone viewport-zero mixed depth tags capture independently of FG and SL evaluation with bounded camera/raw-only evidence");
      return 0;
    }
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
    require(depth_capture::testing::initial_recording_regression(), "first native operation lost state/lifetime evidence or revived a stale callback");
    std::puts("PASS newly created command recording keeps first barrier/close/pass/opaque evidence without Reset; stale callbacks remain rejected");
    require(depth_capture::testing::submission_completion_regression(), "submission ordering invalidated pending completion or admitted an unfinished/failed capture");
    require(depth_capture::testing::provider_admission_regression(), "provider ownership admitted unusable or cross-provider depth");
    require(depth_capture::testing::live_source_admission_regression(),
      "a pipelined live source was locked out by an expired, absent or replaced owner, or a delivering owner alternated or lost selection to a pixel-less capture");
    std::puts("PASS live pipelined sources use their own completed snapshots after owner expiry, reset or source change; delivering owners are kept through rejected attempts and stay revocable");
    require(depth_capture::testing::fg_off_retained_lineage_regression(),
      "FG Off released the queue while the display held its retired FG role's viewport's ordinary copy, or kept it for an expired, other or revoked copy, or after FG On");
    std::puts("PASS FG Off keeps the queue associated while the display holds any copy of the retired FG role's viewport within the source age");
    require(depth_capture::testing::crop_region_regression(), "native depth crop admitted an unsupported copy or rejected a valid extent");
    require(depth_capture::testing::record_diagnostic_regression(), "record diagnostics changed admission or were overwritten by another attempt");
    require(depth_capture::testing::snapshot_basis_regression(), "Snapshot state-basis provenance changed admission");
    std::puts("PASS snapshot state basis: enhanced and legacy state provenance without admission changes");
    std::puts("PASS pending evaluation is unavailable until success in either submission order; failure, Signal loss and queue change remain terminal");
    require(depth_capture::testing::source_cookie_reentry_regression(), "retagged live resource lost its recorded state identity");
    std::puts("PASS live resource keeps its cookie across expired metadata wrappers; new resources and failed writes remain separate");
    require(depth_capture::testing::recording_recovery_regression(), "native command ownership, reset, capacity or retirement contract failed");
    std::puts("PASS 160 live native command owners, destruction/recreation, global invalidation and Reset preserve exact recording identity without assuming GPU completion");
    require(depth_capture::testing::recording_state_loss_regression(), "declared state could bypass alias/split history loss");
    require(depth_capture::testing::alias_lookup_regression(), "sorted aliasing marks answered a lookup differently");
    std::puts("PASS 162 source states preserve unrelated depth and named alias/split blocks; Reset reuses storage; wildcard/identity/allocation loss rejects without retiring owned copies");
    test_version_identity();
    std::puts("PASS ABI families, optional version strings, exact private-state scope and unknown/conflicting rejection");
    test_source_nomination_v1();
    test_source_nomination_v2();
    test_rr_and_linear_depth_source();
    test_mixed_clear_and_linear_depth();
    std::puts("PASS SR/RR linear depth encoding, precision, camera-independent input, lifecycle, FG priority and token reuse");
    test_struct_versions();
    test_unknown_resource_type_by_release();
    std::puts("PASS ResourceTag/Resource/PrecisionInfo of version 1 or later decode by their v1 fields with nativeFormat; refusals name header_version, extension, precision, resource_type and lifecycle; a later tag version's array stride is refused");
    test_source_without_frame_tag_export();
    std::puts("PASS optional frame-tag export preserves global-only production depth, token safety, failure and recovery");
    test_jitter_frame_and_render_domain(false);
    test_jitter_frame_and_render_domain(true);
    test_high_resolution_jitter_domain();
    std::puts("PASS v1/v2 frame-owned jitter, padded render extent, high-res domain proof and optional metadata rejection");
    test_source_without_projection();
    test_standalone_depth_tags();
    test_depth_untag_reporting();
    test_frame_generation_color_only_alpha();
    test_versioned_ui_alpha_tags();
    test_ui_gate_viewport_zero_mixed_batch();
    test_repeated_fg_off_preserves_independent_ui();
    test_ui_scope_without_frame_generation_options();
    std::puts("PASS live pre-FG alpha observes color-only global/frame tags53/23/69 and null-preferred fallback with diagnostic probe off; null, revision, Off and shutdown revoke scope");
    test_frame_generation_tags();
    std::puts("PASS FG first-call options wrapper, synchronous OnlyValidNow tags, precision transforms, global/explicit frames, Off and failure gates");
    test_frame_generation_cached_options_and_global_camera();
    std::puts("PASS pre-cached FG target detour, single forwarding, zero-base resource compatibility, bounded global camera-call pairing and lifecycle gates");
    test_frame_generation_mode_store();
    std::puts("PASS the FG mode survives lost metadata reads, a held metadata writer and a fifth viewport's camera");
    test_present_fg_snapshot();
    std::puts("PASS one FG snapshot per Present from one FG view: options during a Present are the next one's, Q1's two projections, equal modes for a runtime created during ambiguity");
    test_present_fg_one_store_read();
    std::puts("PASS one FG options store read per Present snapshot, applied to the FG view in store order");
    test_present_fg_fallback_one_shot();
    std::puts("PASS a consumer without an FG snapshot of its Present reads its own for that call only");
    test_fg_view_restart();
    std::puts("PASS a restart resets the process's FG view");
    test_any_viewport_read_beside_off();
    test_all_off_read_is_stable();
    std::puts("PASS another viewport's FG Off binds no FG scope: the enabled viewport breaks the tie; two enabled stay ambiguous");
    test_fg_transition_order();
    std::puts("PASS two options calls' FG transitions follow their store commits");
    test_repeated_fg_off_retires_once();
    std::puts("PASS FG Off retires the FG role once; a repeated Off retires nothing new");
    test_refused_frame_calls_end_window();
    test_dynamic_fg_deferral_window();
    std::puts("PASS a refused deferred FG copy and DLSS-G's evaluation without a copy attempt end an earlier frame's presenting window");
    test_dlssg_anchor_for_recorded_tag_copy();
    std::puts("PASS DLSS-G's evaluation standing aside for its frame's recorded FG tag copy anchors that frame's presenting window");
    test_ngx_fg_ownership();
    std::puts("PASS direct NGX's FG ownership reads the FG options store and the scope's liveness");
    test_evaluation_supply_read();
    std::puts("PASS an FG tag boundary reads DLSS-G's supply once, also under a held lock, and FG On revokes the supply of the FG period before it");
    test_camera_read_retry();
    std::puts("PASS a call's camera read that lost the race with the writers is retried briefly: no busy SL evaluation, no dropped tag boundary; one that keeps losing still refuses");
    test_until_evaluate_lifetime();
    std::puts("PASS UntilEvaluate global/frame/local pre-call admission, post-return expiry and unchanged extent rejection");
    std::puts("PASS default-on lightweight source nomination, no selected-depth dependency, lifetime ABA, freshness, loss and viewport gates");
    test_token_records_independence(false);
    test_token_records_independence(true);
    test_repeated_token_request_keeps_frame();
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
    test_legacy_resource_without_state();
    std::puts("PASS legacy no-state resource reads stop at the 40-byte ABI boundary and preserve extension rejection");
    test_v2_forwarding();
    test_v2_forwarding(false);
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
    test_scene_feedback(false);
    test_scene_feedback(true);
    test_scene_feedback_inflight();
    test_reset_constants_observation_loss(false);
    test_reset_constants_observation_loss(true);
    std::puts("PASS v1/v2 fresh reset depth/projection, immutable feedback, and rejection of pre-reset tags, in-flight completion and interrupted reset constants");
    test_frame_generation_vram_warning();
    test_frame_generation_warned_discovery();
    std::puts("PASS eWarnOutOfVRAM keeps FG knowledge and the FG depth source; other SDK results still fail");
    test_feature_independent_depth_contracts();
    std::puts("PASS arbitrary feature IDs with proven depth contracts; opaque v2 and unproven legacy/Reflex inputs remain pass-through");
    test_feature_function_passthrough();
    std::puts("PASS v2 feature-function lookups other than FG options pass through unchanged");
    test_reflex_marker_passthrough();
    std::puts("PASS v1 Reflex marker calls pass through unchanged on success and failure");
    test_call_trace_only(false);
    test_call_trace_only(true);
    std::puts("PASS v1/v2 call-route trace remains independent of depth selection and full content diagnostics");
    // The research ledger tests' checks of kept behaviour, in the shipping
    // configuration.
    test_source_rollback_shipping();
    std::puts("PASS source rollback installs and observes nothing; disabled retained hooks pass through");
    test_evaluation_v1_shipping();
    test_evaluation_order_shipping();
    test_evaluation_v2_tokens_shipping();
    std::puts("PASS stored v1/v2 evaluations keep exact frame cameras, tag scopes, completion order and fail-closed recovery");
    test_sdk_vram_warning_shipping();
    std::puts("PASS eWarnOutOfVRAM keeps stored evaluations and inputs; other SDK results still fail");
    test_evaluation_global_contract_shipping();
    test_evaluation_input_age_shipping();
    std::puts("PASS stored global-tag evaluation and observation-entry camera age");
    test_non_renderer_evaluations_shipping();
    std::puts("PASS Reflex/unknown feature passthrough preserves the stored DLSS evaluation and ignores non-renderer failures");
    test_color_tag_snapshots_shipping();
    std::puts("PASS stored HUD-less and legal null-backbuffer colour tags stay separate from depth");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Streamline probe: %s\n", error.what());
    return 1;
  }
}

// SPDX-License-Identifier: GPL-3.0-only
// Real executable exports + MinHook. No game, graphics runtime or GPU needed.
#include "upscaler_call_trace.h"
#include "ngx_depth_source.h"
#include "addon_lifetime.h"
#include "game3d_diagnostic_metadata.h"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#define FIXTURE_NOINLINE __attribute__((noinline, noipa))
#else
#define FIXTURE_NOINLINE __declspec(noinline)
#endif

namespace {
  using namespace sunshine_upscaler_trace;
  using result = std::uint32_t;
  using callback_type = void (__cdecl *)(float, bool &);
  using create_type = result (__cdecl *)(void *, std::uint32_t, void *, void **);
  using evaluate_type = result (__cdecl *)(void *, const void *, const void *, callback_type);
  using release_type = result (__cdecl *)(void *);
  constexpr DWORD input_error = 0x12345678, output_error = 0x87654321;
  constexpr result ngx_success = 1, ngx_failure = 0xbad00005;
  constexpr std::uintptr_t handle_bits = 0x123456780;
  std::vector<std::string> messages;
  std::vector<std::string> feedback_requests;
  std::vector<std::string> probe_requests;
  std::vector<std::string> jitter_requests;
  enum class probe_reply { failed, null_value, present };
  struct observed_call {
    void *commands{};
    const void *handle{}, *parameters{};
    void **output{};
    callback_type callback{};
    std::uint32_t feature{};
    DWORD incoming_error{};
  } observed;
  std::atomic<unsigned> actual_creates{}, actual_evaluations{}, actual_releases{};
  std::atomic<bool> block_original{}, original_entered{}, release_original{};
  std::atomic<bool> feedback_getter_entered{}, release_feedback_getter{};
  std::atomic<result> return_result{ngx_success};
  std::atomic<bool> write_output{true};
  bool nested_evaluation{};
  bool release_in_original{};
  bool release_in_probe{};
  struct fixture_parameters {
    int flags{8};
    unsigned width{1920}, height{1080}, render_width{1280}, render_height{720}, x{3}, y{2};
    void *depth{reinterpret_cast<void *>(0xface0000)};
    void *ui_hint{};
    bool ui_hint_available{};
    bool missing_flags{}, missing_render{}, missing_y{}, missing_depth{};
    int reset_value{};
    bool reset_unavailable{true};
    float jitter_x{}, jitter_y{};
    bool missing_jitter_x{true}, missing_jitter_y{true};
    std::array<probe_reply, 5> probe_values{};
  };
  sunshine_scene_depth::frame captured_frame;
  std::uint64_t captured_command{};
  unsigned captures{}, finishes{};
  unsigned retirements{};
  std::uint64_t retired_id{};
  bool last_finish{};
  std::mutex feedback_capture_mutex;
  std::vector<sunshine_scene_depth::frame> feedback_captures;
  sunshine_game3d::diagnostic::resource_observation diagnostic_resource;
  sunshine_game3d::diagnostic::stamp diagnostic_finish;
  std::uint64_t diagnostic_finish_source{};
  unsigned diagnostic_resources{}, diagnostic_finishes{}, diagnostic_originals_before{}, diagnostic_originals_after{};
  bool diagnostic_success{}, diagnostic_provider_valid{};
  const sunshine_game3d::diagnostic::resource_callbacks diagnostic_callbacks {
    [](const sunshine_game3d::diagnostic::resource_observation &value) noexcept {
      if (value.artifact_id != 20) return;
      diagnostic_resource = value;
      ++diagnostic_resources;
      diagnostic_originals_before = actual_evaluations.load();
    },
    [](sunshine_game3d::ui_resources::provider provider, const sunshine_game3d::diagnostic::stamp &at,
        std::uint64_t source_id, bool successful) noexcept {
      diagnostic_finish = at;
      diagnostic_finish_source = source_id;
      diagnostic_provider_valid = provider == sunshine_game3d::ui_resources::provider::ngx;
      diagnostic_success = successful;
      ++diagnostic_finishes;
      diagnostic_originals_after = actual_evaluations.load();
    }
  };
  create_type create_entry{};
  evaluate_type evaluate_entry{}, evaluate_c_entry{};
  release_type release_entry{};
  void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
  bool logged(const char *text) {
    for (const auto &value : messages) if (value.find(text) != std::string::npos) return true;
    return false;
  }
  void callback(float, bool &) {}
  result optional_pointer_reply(probe_reply reply, void **output) {
    // Neither successful nor failed getters authorize reading this address.
    *output = reply == probe_reply::null_value ? nullptr : reinterpret_cast<void *>(1);
    return reply == probe_reply::failed ? ngx_failure : ngx_success;
  }
  template<class Predicate> void await(Predicate predicate, const char *message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    require(predicate(), message);
  }
}

extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void *, std::uint32_t) { return false; }
extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void *) {}
extern "C" __declspec(dllexport) void ReShadeLogMessage(void *, int, const char *message) { messages.emplace_back(message); }

extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_D3D12_CreateFeature(
    void *commands, std::uint32_t feature, void *parameters, void **output) {
  observed = {commands, nullptr, parameters, output, nullptr, feature, GetLastError()};
  ++actual_creates;
  if (write_output.load() && output) *output = reinterpret_cast<void *>(handle_bits);
  SetLastError(output_error);
  return return_result.load();
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_D3D12_EvaluateFeature(
    void *commands, const void *handle, const void *parameters, callback_type progress) {
  observed = {commands, handle, parameters, nullptr, progress, 0, GetLastError()};
  ++actual_evaluations;
  if (nested_evaluation) evaluate_c_entry(commands, handle, parameters, progress);
  if (release_in_original) release_entry(const_cast<void *>(handle));
  if (block_original.load()) {
    original_entered.store(true);
    while (!release_original.load()) std::this_thread::yield();
  }
  SetLastError(output_error);
  return return_result.load();
}
// Distinct address proves the C spelling is discovered too. Its callback remains
// an opaque pointer to the tracer; the fixture deliberately does not invoke it.
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_D3D12_EvaluateFeature_C(
    void *commands, const void *handle, const void *parameters, callback_type progress) {
  observed = {commands, handle, parameters, nullptr, progress, 1, GetLastError()};
  ++actual_evaluations;
  SetLastError(output_error);
  return return_result.load();
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_D3D12_ReleaseFeature(void *handle) {
  observed = {nullptr, handle, nullptr, nullptr, nullptr, 0, GetLastError()};
  ++actual_releases;
  SetLastError(output_error);
  return return_result.load();
}

extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_Parameter_GetI(
    void *opaque, const char *key, int *output) {
  auto &value = *static_cast<fixture_parameters *>(opaque);
  SetLastError(0xfade);
  if (std::strcmp(key, "Reset") == 0) {
    feedback_requests.emplace_back("I:Reset");
    if (value.reset_unavailable) return ngx_failure;
    *output = value.reset_value;
    return ngx_success;
  }
  if (value.missing_flags || std::strcmp(key, "DLSS.Feature.Create.Flags") != 0) return ngx_failure;
  *output = value.flags;
  return ngx_success;
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_Parameter_GetUI(
    void *opaque, const char *key, unsigned *output) {
  auto &value = *static_cast<fixture_parameters *>(opaque);
  SetLastError(0xfade);
  if (std::strcmp(key, "Width") == 0) *output = value.width;
  else if (std::strcmp(key, "Height") == 0) *output = value.height;
  else if (std::strcmp(key, "DLSS.Render.Subrect.Dimensions.Width") == 0 && !value.missing_render) *output = value.render_width;
  else if (std::strcmp(key, "DLSS.Render.Subrect.Dimensions.Height") == 0 && !value.missing_render) *output = value.render_height;
  else if (std::strcmp(key, "DLSS.Input.Depth.Subrect.Base.X") == 0) *output = value.x;
  else if (std::strcmp(key, "DLSS.Input.Depth.Subrect.Base.Y") == 0 && !value.missing_y) *output = value.y;
  else return ngx_failure;
  return ngx_success;
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_Parameter_GetD3d12Resource(
    void *opaque, const char *key, void **output) {
  auto &value = *static_cast<fixture_parameters *>(opaque);
  SetLastError(0xfade);
  feedback_requests.emplace_back(std::string("Resource:") + key);
  if (std::strcmp(key, "Position.ViewSpace") == 0) {
    probe_requests.emplace_back(std::string("Resource:") + key);
    return optional_pointer_reply(value.probe_values[0], output);
  }
  if (std::strcmp(key, "TransparencyMask") == 0 && value.ui_hint_available) {
    *output = value.ui_hint;
    return ngx_success;
  }
  if (value.missing_depth || std::strcmp(key, "Depth") != 0) return ngx_failure;
  *output = value.depth;
  return ngx_success;
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_Parameter_GetF(
    void *opaque, const char *key, float *output) {
  const auto &value = *static_cast<fixture_parameters *>(opaque);
  SetLastError(0xfade);
  jitter_requests.emplace_back(key);
  // Failed getters deliberately leave their payload behind to catch stale/poisoned metadata.
  if (std::strcmp(key, "Jitter.Offset.X") == 0) {
    *output = value.jitter_x;
    return value.missing_jitter_x ? ngx_failure : ngx_success;
  }
  if (std::strcmp(key, "Jitter.Offset.Y") == 0) {
    *output = value.jitter_y;
    return value.missing_jitter_y ? ngx_failure : ngx_success;
  }
  return ngx_failure;
}
extern "C" __declspec(dllexport) FIXTURE_NOINLINE result __cdecl NVSDK_NGX_Parameter_GetVoidPointer(
    void *opaque, const char *key, void **output) {
  const auto &value = *static_cast<fixture_parameters *>(opaque);
  SetLastError(0xfade);
  probe_requests.emplace_back(std::string("Pointer:") + key);
  if (release_in_probe) {
    release_in_probe = false;
    sunshine_ngx::after_release(GetModuleHandleW(nullptr), reinterpret_cast<void *>(handle_bits), sunshine_ngx::epoch(), true);
  }
  constexpr const char *names[] = {"WorldToViewMatrix", "ViewToClipMatrix", "InvViewProjectionMatrix", "ClipToPrevClipMatrix"};
  for (unsigned i = 0; i != 4; ++i)
    if (std::strcmp(key, names[i]) == 0) return optional_pointer_reply(value.probe_values[i + 1], output);
  *output = reinterpret_cast<void *>(1);
  return ngx_failure;
}
namespace {
  void evaluate(bool c = false) {
    const auto before = actual_evaluations.load();
    auto *commands = reinterpret_cast<void *>(0x10203040);
    const auto *handle = reinterpret_cast<void *>(handle_bits);
    const auto *parameters = reinterpret_cast<void *>(0x8090a0b0);
    SetLastError(input_error);
    const auto value = (c ? evaluate_c_entry : evaluate_entry)(commands, handle, parameters, &callback);
    const DWORD error = GetLastError();
    require(value == return_result.load() && error == output_error, "evaluate result/LastError changed");
    require(actual_evaluations.load() == before + 1, "evaluate original was not called exactly once");
    require(observed.commands == commands && observed.handle == handle && observed.parameters == parameters &&
      observed.callback == &callback && observed.incoming_error == input_error, "evaluate argument/incoming LastError changed");
  }
  void create(std::uint32_t feature = 1) {
    void *handle{};
    const auto before = actual_creates.load();
    auto *commands = reinterpret_cast<void *>(0x1000);
    auto *parameters = reinterpret_cast<void *>(0x2000);
    SetLastError(input_error);
    const auto value = create_entry(commands, feature, parameters, &handle);
    const DWORD error = GetLastError();
    require(value == return_result.load() && error == output_error, "create result/LastError changed");
    require(actual_creates.load() == before + 1 && observed.output == &handle && observed.commands == commands &&
      observed.parameters == parameters && observed.feature == feature && observed.incoming_error == input_error,
      "create argument/count/incoming LastError changed");
    require(handle == reinterpret_cast<void *>(handle_bits), "create output handle changed");
  }
  void release() {
    const auto before = actual_releases.load();
    auto *handle = reinterpret_cast<void *>(handle_bits);
    SetLastError(input_error);
    require(release_entry(handle) == return_result.load() && GetLastError() == output_error, "release result/LastError changed");
    require(actual_releases.load() == before + 1 && observed.handle == handle && observed.incoming_error == input_error,
      "release argument/count/incoming LastError changed");
  }
  void basic_test() {
    const HMODULE self = GetModuleHandleW(nullptr);
    create_entry = reinterpret_cast<create_type>(GetProcAddress(self, "NVSDK_NGX_D3D12_CreateFeature"));
    evaluate_entry = reinterpret_cast<evaluate_type>(GetProcAddress(self, "NVSDK_NGX_D3D12_EvaluateFeature"));
    evaluate_c_entry = reinterpret_cast<evaluate_type>(GetProcAddress(self, "NVSDK_NGX_D3D12_EvaluateFeature_C"));
    release_entry = reinterpret_cast<release_type>(GetProcAddress(self, "NVSDK_NGX_D3D12_ReleaseFeature"));
    require(create_entry && evaluate_entry && evaluate_c_entry && release_entry, "fixture exports unavailable");
    initialize(self, false);
    poll();
    create(); evaluate(); release();
    require(!enabled() && testing::counts().installed == 0 && messages.empty(), "default-off installed hooks or emitted logs");
    initialize(self, true);
    SetLastError(input_error);
    poll();
    require(GetLastError() == input_error, "poll changed LastError");
    auto counts = testing::counts();
    require(counts.installed == 4 && counts.ngx_evaluate == 0 && counts.reports == 1,
      "real export discovery or no-call coverage failed");
    require(logged("NGX_evaluate=0") && logged("not API absence") && logged("SL_hook=0"),
      "zero-call report omitted coverage limits");
    create(); evaluate(); evaluate(true);
    counts = testing::counts();
    require(counts.ngx_create == 1 && counts.ngx_evaluate == 2 && counts.outside_streamline == 3 &&
      counts.inside_streamline == 0 && counts.unknown_feature == 0, "direct observation or create-handle feature association failed");
    set_streamline_coverage(true);
    SetLastError(input_error);
    {
      streamline_scope outer(0, reinterpret_cast<std::uintptr_t>(&basic_test));
      evaluate();
      {
        streamline_scope inner(11, reinterpret_cast<std::uintptr_t>(&basic_test));
        evaluate(true);
        SetLastError(output_error);
        inner.finish(false);
        inner.finish(true); // A second finish must not double count.
      }
      require(GetLastError() == output_error, "nested SL scope changed LastError");
      evaluate();
      outer.finish(true);
    }
    require(GetLastError() == output_error, "SL scope destruction changed LastError");
    evaluate();
    counts = testing::counts();
    require(counts.inside_streamline == 3 && counts.outside_streamline == 4 && counts.streamline == 2 &&
      counts.failed == 1 && counts.streamline_covered, "SL nesting restoration/result observation failed");
    return_result.store(ngx_failure);
    evaluate();
    require(testing::counts().failed == 2, "NGX failure treated as success");
    return_result.store(ngx_success);
    release();
    evaluate();
    require(testing::counts().unknown_feature == 1, "released handle retained its feature identity");
    const auto before = messages.size();
    testing::report_now();
    require(logged("inside-observed-SL") && logged("outside-observed-SL") && logged("feature=1") &&
      logged("unknown(create not observed)") && logged("first-call stack"), "call provenance report missing");
    const auto after = messages.size();
    require(after > before, "report omitted observed calls");
    testing::report_now();
    require(messages.size() == after + 1, "unchanged report repeated coverage/callsite/stacks");
    shutdown();
    const auto stopped = testing::counts();
    create(); evaluate(); release();
    { streamline_scope scope(0, 0xabc); scope.finish(true); }
    poll();
    counts = testing::counts();
    require(!enabled() && stopped.ngx_create == counts.ngx_create && stopped.ngx_evaluate == counts.ngx_evaluate &&
      stopped.ngx_release == counts.ngx_release && stopped.streamline == counts.streamline,
      "shutdown did not become pass-through");
  }
  void epoch_test() {
    initialize(GetModuleHandleW(nullptr), true);
    create();
    block_original.store(true); original_entered.store(false); release_original.store(false);
    std::atomic<bool> passed{false};
    std::thread in_flight([&] {
      SetLastError(input_error);
      const auto value = evaluate_entry(nullptr, reinterpret_cast<void *>(handle_bits), nullptr, nullptr);
      passed.store(value == ngx_success && GetLastError() == output_error);
    });
    await([] { return original_entered.load(); }, "delayed original did not enter");
    const auto previous_epoch = testing::counts().epoch;
    shutdown();
    initialize(GetModuleHandleW(nullptr), true);
    require(testing::counts().epoch != previous_epoch, "reinitialize reused old epoch");
    release_original.store(true);
    in_flight.join();
    block_original.store(false);
    require(passed.load() && testing::counts().ngx_evaluate == 0 && testing::counts().stale == 1,
      "in-flight old call corrupted new epoch or lost pass-through");
    evaluate();
    require(testing::counts().unknown_feature == 1, "new epoch reused stale feature handle");
    create(); evaluate();
    require(testing::counts().ngx_evaluate == 2 && testing::counts().unknown_feature == 1,
      "retained hook failed after reinitialize");
    // Invalid output observation must never dereference or change successful
    // original results, even if a third-party API violates its output contract.
    write_output.store(false);
    SetLastError(input_error);
    const auto result = create_entry(nullptr, 1, nullptr, reinterpret_cast<void **>(1));
    require(result == ngx_success && GetLastError() == output_error, "unreadable output changed pass-through");
    write_output.store(true);
    evaluate();
    require(testing::counts().unknown_feature == 2, "unreadable create retained old feature association");
    shutdown();
  }
  void capture_test() {
    initialize(GetModuleHandleW(nullptr), false, true);
    require(!enabled() && capture_enabled(), "production capture depends on diagnostics");
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t command, const sunshine_scene_depth::frame &value) -> std::uint64_t {
        captured_command = command;
        captured_frame = value;
        ++captures;
        SetLastError(0xfaded);
        return 77;
      },
      [](std::uint64_t ticket, bool success) {
        require(ticket == 77, "wrong production capture ticket");
        ++finishes;
        last_finish = success;
        SetLastError(0xfaded);
      },
      [](std::uint64_t epoch, std::uint64_t source_id) {
        require(epoch != 0 && source_id != 0, "release retirement omitted source identity");
        ++retirements;
        retired_id = source_id;
        SetLastError(0xfaded);
      }});
    fixture_parameters parameters;
    void *handle{};
    auto create_feature = [&](unsigned feature = 1) {
      SetLastError(input_error);
      const auto result = create_entry(reinterpret_cast<void *>(0x123), feature, &parameters, &handle);
      require(result == ngx_success && GetLastError() == output_error && observed.incoming_error == input_error,
        "capture metadata changed creation LastError/result");
    };
    auto evaluate_feature = [&] {
      SetLastError(input_error);
      const auto result = evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback);
      require(result == return_result.load() && GetLastError() == output_error,
        "capture changed evaluation result/LastError");
    };
    handle = reinterpret_cast<void *>(handle_bits);
    evaluate_feature();
    require(captures == 0, "late unknown feature was guessed");
    create_feature();
    nested_evaluation = true;
    evaluate_feature();
    nested_evaluation = false;
    require(captures == 1 && finishes == 1 && last_finish && captured_command == 0x123,
      "nested NGX evaluation did not capture exactly once");
    require(captured_frame.provider == sunshine_scene_depth::provider_kind::ngx && captured_frame.source_id &&
      !captured_frame.projection.supplied && captured_frame.projection.direction_supplied && captured_frame.projection.reversed &&
      captured_frame.proof == sunshine_scene_depth::state_proof::unavailable && captured_frame.native_state == UINT32_MAX &&
      captured_frame.valid_until == sunshine_scene_depth::lifetime::until_evaluation &&
      captured_frame.resource.native == reinterpret_cast<std::uint64_t>(parameters.depth) &&
      captured_frame.resource.area.left == 3 && captured_frame.resource.area.top == 2 &&
      captured_frame.resource.area.width == 1280 && captured_frame.resource.area.height == 720,
      "named getters did not produce exact raw NGX depth metadata");
    const auto first_id = captured_frame.source_id;
    {
      streamline_scope scope(0, 0x123);
      evaluate_feature();
      scope.finish(true);
    }
    require(captures == 1, "SL-nested NGX capture was not suppressed with tracing disabled");
    parameters.missing_depth = true;
    evaluate_feature();
    require(captures == 1, "missing depth captured stale parameters");
    parameters.missing_depth = false;
    parameters.missing_y = true;
    evaluate_feature();
    require(captures == 1, "partial crop metadata was accepted");
    parameters.missing_y = false;
    parameters.missing_render = true;
    evaluate_feature();
    require(captures == 2 && captured_frame.resource.area.width == 1920 && captured_frame.resource.area.height == 1080 &&
      captured_frame.source_id == first_id, "older dimensions or stable feature ID failed");
    return_result.store(ngx_failure);
    evaluate_feature();
    require(captures == 3 && finishes == 3 && !last_finish, "failed NGX evaluation accepted its capture");
    release();
    require(retirements == 0, "failed release retired a live feature");
    return_result.store(ngx_success);
    release();
    require(retirements == 1 && retired_id == first_id, "successful release did not retire the exact feature");
    release();
    require(retirements == 1, "unknown/already-released handle retired a source");
    evaluate_feature();
    require(captures == 3, "released feature kept capturing");
    create_feature(11);
    evaluate_feature();
    require(captures == 3, "frame generation feature was treated as DLSS depth");
    parameters.flags = 0;
    create_feature();
    evaluate_feature();
    require(captures == 4 && !captured_frame.projection.reversed && captured_frame.source_id != first_id,
      "feature replacement did not update orientation/identity");
    release_in_original = true;
    evaluate_feature();
    release_in_original = false;
    require(captures == 5 && !last_finish && retirements == 2,
      "in-flight evaluation revived a successfully released feature");
    require(testing::counts().ngx_evaluate == 0, "capture-only emitted diagnostic call records");
    shutdown();
    evaluate_feature();
    require(captures == 5, "shutdown did not stop production capture");
    sunshine_ngx::testing::set_callbacks({});
  }
  void reset_feedback_test() {
    initialize(GetModuleHandleW(nullptr), false, true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &value) -> std::uint64_t {
        captured_frame = value;
        ++captures;
        return 77;
      }, nullptr, nullptr});
    fixture_parameters parameters;
    parameters.reset_unavailable = false;
    parameters.reset_value = 1;
    feedback_requests.clear();
    probe_requests.clear();
    messages.clear();
    const auto before = captures;
    void *handle{};
    require(create_entry(reinterpret_cast<void *>(0x123), 1, &parameters, &handle) == ngx_success,
      "reset feedback fixture create failed");
    auto evaluate_feature = [&] {
      const auto originals = actual_evaluations.load();
      SetLastError(input_error);
      const auto result = evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback);
      require(result == ngx_success && GetLastError() == output_error && observed.incoming_error == input_error &&
        actual_evaluations == originals + 1 && observed.parameters == &parameters && observed.handle == handle &&
        observed.callback == &callback, "reset feedback changed original evaluation/LastError");
    };
    evaluate_feature();
    const std::vector<std::string> expected = {"I:Reset", "Resource:Depth"};
    require(feedback_requests == expected && captures == before + 1 && !captured_frame.projection.supplied &&
      captured_frame.projection.reversed && captured_frame.resource.native == reinterpret_cast<std::uint64_t>(parameters.depth),
      "reset feedback used incorrect getter types/keys or changed depth capture");
    require(probe_requests.empty(), "default capture requested optional calibration metadata");
    require(messages.empty(), "reset feedback logged inside an evaluation callback");
    SetLastError(input_error);
    sunshine_ngx::poll();
    require(GetLastError() == input_error && logged("Sunshine NGX depth: confirmed_features=1"),
      "depth report changed LastError or lost the confirmed feature");
    require(captured_frame.feedback.revision != 0 && captured_frame.feedback.reset,
      "NGX reset did not reach immutable feedback");
    const auto initial_feedback = captured_frame.feedback;
    const auto initial_source = captured_frame.source_id;
    parameters.reset_value = 0;
    for (unsigned i = 0; i != 100; ++i) evaluate_feature();
    require(feedback_requests.size() == expected.size() * 101, "reset and depth were not sampled every evaluation");
    for (std::size_t i = 0; i < feedback_requests.size(); ++i)
      require(feedback_requests[i] == expected[i % expected.size()], "evaluation requested unused metadata");
    require(captured_frame.feedback.revision == initial_feedback.revision && !captured_frame.feedback.reset &&
      captured_frame.source_id == initial_source, "ordinary evaluations changed scene revision or source identity");
    // A one-frame reset may fall between readbacks or have no usable depth.
    // Its revision must still invalidate older scene evidence on a later frame.
    parameters.reset_value = 1;
    parameters.missing_depth = true;
    const auto before_reset_capture = captures;
    evaluate_feature();
    require(captures == before_reset_capture, "reset without depth unexpectedly captured pixels");
    parameters.reset_value = 0;
    parameters.missing_depth = false;
    evaluate_feature();
    require(captured_frame.feedback.revision == initial_feedback.revision + 1 && !captured_frame.feedback.reset &&
      captured_frame.source_id == initial_source, "reset pulse was lost between usable captures or replaced the logical source");
    parameters.reset_value = -1;
    evaluate_feature();
    const auto reset_revision = captured_frame.feedback.revision;
    require(reset_revision == initial_feedback.revision + 2 && captured_frame.feedback.reset,
      "nonzero NGX reset was ignored");
    parameters.reset_unavailable = true;
    evaluate_feature();
    require(captured_frame.feedback.revision == reset_revision && !captured_frame.feedback.reset &&
      captured_frame.source_id == initial_source,
      "absent optional reset changed the scene or source identity");
    parameters.reset_unavailable = false;
    parameters.reset_value = 0;
    require(parameters.flags == 8 && parameters.width == 1920 && parameters.height == 1080 && parameters.x == 3 &&
      parameters.y == 2 && parameters.render_width == 1280 && parameters.render_height == 720 &&
      parameters.depth == reinterpret_cast<void *>(0xface0000) && !parameters.reset_unavailable,
      "reset feedback mutated original parameters");
    shutdown();
    sunshine_ngx::testing::set_callbacks({});
  }
  void calibration_probe_test() {
    const auto owner = GetModuleHandleW(nullptr);
    initialize(owner, false, true);
    sunshine_ngx::initialize(true, true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &value) -> std::uint64_t {
        captured_frame = value;
        ++captures;
        SetLastError(0xfaded);
        return 77;
      }, nullptr, nullptr});
    const auto api = sunshine_ngx::resolve_parameter_api(owner);
    require(api && api.void_pointer, "optional named pointer getter export was not discovered");
    fixture_parameters parameters;
    parameters.probe_values = {probe_reply::present, probe_reply::present, probe_reply::failed,
      probe_reply::null_value, probe_reply::present};
    void *handle{};
    require(create_entry(reinterpret_cast<void *>(0x123), 1, &parameters, &handle) == ngx_success,
      "calibration probe fixture create failed");
    const std::vector<std::string> expected = {"Resource:Position.ViewSpace", "Pointer:WorldToViewMatrix",
      "Pointer:ViewToClipMatrix", "Pointer:InvViewProjectionMatrix", "Pointer:ClipToPrevClipMatrix"};
    probe_requests.clear();
    messages.clear();
    parameters.missing_depth = true;
    const auto missing_depth_captures = captures, missing_depth_originals = actual_evaluations.load();
    SetLastError(input_error);
    require(evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback) == ngx_success &&
      GetLastError() == output_error && actual_evaluations == missing_depth_originals + 1 &&
      captures == missing_depth_captures && probe_requests.empty(),
      "missing required depth queried optional calibration fields or changed original evaluation");
    parameters.missing_depth = false;
    auto evaluate_feature = [&] {
      const auto originals = actual_evaluations.load(), before = captures;
      SetLastError(input_error);
      const auto result = evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback);
      require(result == ngx_success && GetLastError() == output_error && observed.incoming_error == input_error &&
        actual_evaluations == originals + 1 && observed.parameters == &parameters && observed.handle == handle &&
        observed.callback == &callback && captures == before + 1 &&
        captured_frame.resource.native == reinterpret_cast<std::uint64_t>(parameters.depth) &&
        !captured_frame.projection.supplied && captured_frame.projection.reversed,
        "calibration presence probe changed evaluation arguments/result/LastError or depth capture");
    };
    evaluate_feature();
    require(probe_requests == expected, "calibration probe used incorrect keys or getter types");
    require(messages.empty(), "calibration probe logged inside the evaluation callback");
    SetLastError(input_error);
    sunshine_ngx::poll();
    require(GetLastError() == input_error && logged("Sunshine NGX calibration probe:") &&
      logged("Position.ViewSpace=present") && logged("WorldToViewMatrix=present") &&
      logged("ViewToClipMatrix=failed") && logged("InvViewProjectionMatrix=null") &&
      logged("ClipToPrevClipMatrix=present"), "calibration report lost typed presence or treated failed poisoned pointer as present");
    for (unsigned i = 0; i != 100; ++i) evaluate_feature();
    require(probe_requests == expected, "calibration probe queried optional getters for every evaluation");

    // The time gate belongs to the confirmed feature, not a global timer which
    // could hide a second viewport/feature's first observation indefinitely.
    const auto other = reinterpret_cast<void *>(handle_bits + 16);
    sunshine_ngx::after_create(owner, other, sunshine_ngx::before_create(api, 1, &parameters), true);
    auto attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, other, &parameters);
    sunshine_ngx::after_evaluate(attempt, true);
    require(attempt.ticket == 77 && probe_requests.size() == 2 * expected.size(),
      "one feature's calibration throttle suppressed another confirmed feature");

    // Reinitialization isolates availability scenarios without waiting or
    // making assumptions about scheduler timing during a test run.
    for (const auto state : {probe_reply::null_value, probe_reply::failed}) {
      sunshine_ngx::initialize(true, true);
      parameters.probe_values.fill(state);
      sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, 1, &parameters), true);
      messages.clear();
      probe_requests.clear();
      SetLastError(input_error);
      attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, handle, &parameters);
      sunshine_ngx::after_evaluate(attempt, true);
      sunshine_ngx::poll();
      require(GetLastError() == input_error && attempt.ticket == 77 && probe_requests == expected,
        "unavailable optional metadata changed LastError/capture or probe coverage");
      const char *label = state == probe_reply::null_value ? "=null" : "=failed";
      for (const auto *name : {"Position.ViewSpace", "WorldToViewMatrix", "ViewToClipMatrix",
          "InvViewProjectionMatrix", "ClipToPrevClipMatrix"})
        require(logged((std::string(name) + label).c_str()), "null/failed optional metadata state was misreported");
      if (state == probe_reply::failed)
        require(logged("Position.ViewSpace=failed(0x0)"), "failed getter's poisoned address leaked into the identity report");
    }

    auto missing = api;
    missing.void_pointer = nullptr;
    require(static_cast<bool>(missing), "missing optional getter disabled required depth API");
    sunshine_ngx::initialize(true, true);
    parameters.probe_values.fill(probe_reply::present);
    sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(missing, 1, &parameters), true);
    messages.clear();
    probe_requests.clear();
    SetLastError(input_error);
    attempt = sunshine_ngx::before_evaluate(owner, missing, 0x123, handle, &parameters);
    sunshine_ngx::after_evaluate(attempt, true);
    sunshine_ngx::poll();
    require(GetLastError() == input_error && attempt.ticket == 77 &&
      probe_requests == std::vector<std::string>{"Resource:Position.ViewSpace"} &&
      logged("Position.ViewSpace=present"), "missing optional getter changed capture or called an unavailable function");
    for (const auto *name : {"WorldToViewMatrix", "ViewToClipMatrix", "InvViewProjectionMatrix", "ClipToPrevClipMatrix"})
      require(logged((std::string(name) + "=missing-getter").c_str()), "missing optional getter was reported as field absence");
    require(parameters.flags == 8 && parameters.width == 1920 && parameters.height == 1080 && parameters.x == 3 &&
      parameters.y == 2 && parameters.render_width == 1280 && parameters.render_height == 720 &&
      parameters.depth == reinterpret_cast<void *>(0xface0000), "calibration probe mutated game parameters");

    sunshine_ngx::initialize(true, true);
    sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, 1, &parameters), true);
    messages.clear();
    probe_requests.clear();
    release_in_probe = true;
    SetLastError(input_error);
    attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, handle, &parameters);
    sunshine_ngx::after_evaluate(attempt, true);
    sunshine_ngx::poll();
    require(!release_in_probe && GetLastError() == input_error && probe_requests == expected &&
      !logged("Sunshine NGX calibration probe:") && logged("confirmed_features=0"),
      "reentrant release inside an optional getter revived retired calibration metadata");
    sunshine_ngx::shutdown();
    sunshine_ngx::testing::set_callbacks({});
    shutdown();
  }
  void jitter_metadata_test() {
    sunshine_ngx::initialize(true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &value) -> std::uint64_t {
        captured_frame = value;
        ++captures;
        return 77;
      }, nullptr, nullptr});
    const auto owner = GetModuleHandleW(nullptr);
    const auto handle = reinterpret_cast<void *>(handle_bits);
    auto api = sunshine_ngx::resolve_parameter_api(owner);
    require(api && api.floating == NVSDK_NGX_Parameter_GetF, "optional named NGX float getter was not resolved");
    fixture_parameters parameters;
    parameters.missing_jitter_x = parameters.missing_jitter_y = false;
    parameters.jitter_x = 0.375f;
    parameters.jitter_y = -0.25f;
    sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, 1, &parameters), true);
    const auto evaluate_metadata = [&](const sunshine_ngx::parameter_api &getters) {
      const auto before = captures;
      const auto previous_sequence = captured_frame.sequence;
      SetLastError(input_error);
      const auto evaluation = sunshine_ngx::before_evaluate(owner, getters, 0x123, handle, &parameters);
      require(GetLastError() == input_error && evaluation.observed && evaluation.ticket == 77 && captures == before + 1,
        "optional jitter changed usable depth admission or LastError");
      require(captured_frame.sequence > previous_sequence &&
        captured_frame.resource.native == reinterpret_cast<std::uint64_t>(parameters.depth),
        "jitter was not associated with this evaluation's nominated depth");
      sunshine_ngx::after_evaluate(evaluation, true);
    };
    jitter_requests.clear();
    evaluate_metadata(api);
    require(jitter_requests == std::vector<std::string>{"Jitter.Offset.X", "Jitter.Offset.Y"},
      "jitter did not use the two documented float keys");
    require(captured_frame.jitter.supplied && captured_frame.jitter.x == 0.375f && captured_frame.jitter.y == -0.25f &&
      captured_frame.jitter.width == 1280 && captured_frame.jitter.height == 720 &&
      captured_frame.resource.area.left == 3 && captured_frame.resource.area.top == 2,
      "cropped render-domain jitter used output/allocation dimensions or included crop base");
    parameters.jitter_x = -0.125f;
    parameters.jitter_y = 0.4375f;
    parameters.render_width = 960;
    parameters.render_height = 540;
    parameters.x = 37;
    parameters.y = 19;
    evaluate_metadata(api);
    require(captured_frame.jitter.x == -0.125f && captured_frame.jitter.y == 0.4375f &&
      captured_frame.jitter.width == 960 && captured_frame.jitter.height == 540 &&
      captured_frame.resource.area.left == 37 && captured_frame.resource.area.top == 19,
      "next evaluation retained stale jitter or render subrect metadata");
    parameters.missing_render = true;
    parameters.jitter_x = parameters.jitter_y = 0.0f;
    evaluate_metadata(api);
    require(captured_frame.jitter.supplied && captured_frame.jitter.x == 0.0f && captured_frame.jitter.y == 0.0f &&
      captured_frame.jitter.width == 1920 && captured_frame.jitter.height == 1080,
      "zero jitter or legacy creation-resolution domain was lost");

    const auto expect_unsupplied = [&] {
      require(!captured_frame.jitter.supplied && captured_frame.jitter.x == 0.0f && captured_frame.jitter.y == 0.0f &&
        captured_frame.jitter.width == 0 && captured_frame.jitter.height == 0,
        "missing/invalid jitter inherited stale or poisoned metadata");
    };
    struct jitter_case { float x, y; bool missing_x, missing_y; };
    const std::array<jitter_case, 7> invalid {{
      {0.25f, -0.25f, true, false},
      {0.25f, -0.25f, false, true},
      {0.25f, -0.25f, true, true},
      {std::numeric_limits<float>::quiet_NaN(), 0.25f, false, false},
      {0.25f, std::numeric_limits<float>::quiet_NaN(), false, false},
      {std::numeric_limits<float>::infinity(), 0.25f, false, false},
      {0.25f, -std::numeric_limits<float>::infinity(), false, false},
    }};
    for (const auto &candidate : invalid) {
      parameters.jitter_x = candidate.x;
      parameters.jitter_y = candidate.y;
      parameters.missing_jitter_x = candidate.missing_x;
      parameters.missing_jitter_y = candidate.missing_y;
      evaluate_metadata(api);
      expect_unsupplied();
    }
    parameters.jitter_x = 0.25f;
    parameters.jitter_y = -0.25f;
    parameters.missing_jitter_x = parameters.missing_jitter_y = false;
    evaluate_metadata(api);
    require(captured_frame.jitter.supplied, "valid jitter did not recover after malformed input");
    const auto getter_calls = jitter_requests.size();
    api.floating = nullptr;
    require(static_cast<bool>(api), "missing optional float export disabled mandatory NGX getters");
    evaluate_metadata(api);
    expect_unsupplied();
    require(jitter_requests.size() == getter_calls, "missing float getter was called");
    sunshine_ngx::testing::set_callbacks({});
    sunshine_ngx::shutdown();
  }
  void concurrent_feedback_test() {
    sunshine_ngx::initialize(true);
    feedback_captures.clear();
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &value) -> std::uint64_t {
        std::lock_guard lock(feedback_capture_mutex);
        feedback_captures.push_back(value);
        return 77;
      }, nullptr, nullptr});
    // Private getters avoid the real-hook fixture's deliberately unsynchronized
    // original-call diagnostics. Delay only one typed Reset getter, never a lock.
    sunshine_ngx::parameter_api api;
    api.integer = [](void *opaque, const char *key, int *output) -> result {
      const auto &value = *static_cast<fixture_parameters *>(opaque);
      SetLastError(0xfade);
      if (std::strcmp(key, "Reset") == 0) {
        if (value.reset_value != 0) {
          feedback_getter_entered = true;
          while (!release_feedback_getter.load()) std::this_thread::yield();
        }
        *output = value.reset_value;
        return ngx_success;
      }
      if (std::strcmp(key, "DLSS.Feature.Create.Flags") != 0) return ngx_failure;
      *output = value.flags;
      return ngx_success;
    };
    api.unsigned_integer = NVSDK_NGX_Parameter_GetUI;
    api.resource = [](void *opaque, const char *key, void **output) -> result {
      SetLastError(0xfade);
      if (std::strcmp(key, "Depth") != 0) return ngx_failure;
      *output = static_cast<fixture_parameters *>(opaque)->depth;
      return ngx_success;
    };
    fixture_parameters reset, ordinary;
    reset.reset_value = 1;
    const auto owner = GetModuleHandleW(nullptr);
    const auto handle = reinterpret_cast<void *>(handle_bits);
    sunshine_ngx::after_create(owner, handle, sunshine_ngx::before_create(api, 1, &ordinary), true);
    feedback_getter_entered = release_feedback_getter = false;
    bool delayed_error_preserved = false;
    std::thread delayed([&] {
      SetLastError(input_error);
      const auto attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, handle, &reset);
      sunshine_ngx::after_evaluate(attempt, true);
      delayed_error_preserved = attempt.ticket == 77 && GetLastError() == input_error;
    });
    try {
      await([] { return feedback_getter_entered.load(); }, "delayed reset getter was not entered");
      SetLastError(input_error);
      const auto attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, handle, &ordinary);
      sunshine_ngx::after_evaluate(attempt, true);
      require(attempt.ticket == 77 && GetLastError() == input_error, "concurrent normal evaluation changed LastError or capture");
    } catch (...) {
      release_feedback_getter = true;
      delayed.join();
      throw;
    }
    release_feedback_getter = true;
    delayed.join();
    require(delayed_error_preserved && feedback_captures.size() == 2 &&
      feedback_captures[0].source_id == feedback_captures[1].source_id &&
      feedback_captures[0].sequence < feedback_captures[1].sequence &&
      feedback_captures[0].feedback.revision + 1 == feedback_captures[1].feedback.revision &&
      !feedback_captures[0].feedback.reset && feedback_captures[1].feedback.reset,
      "concurrent getter completion reordered feedback revisions or changed logical source");
    const auto attempt = sunshine_ngx::before_evaluate(owner, api, 0x123, handle, &ordinary);
    sunshine_ngx::after_evaluate(attempt, true);
    require(feedback_captures.size() == 3 && feedback_captures[2].sequence > feedback_captures[1].sequence &&
      feedback_captures[2].feedback.revision == feedback_captures[1].feedback.revision && !feedback_captures[2].feedback.reset,
      "normal frame failed to retain the concurrently observed scene reset");
    sunshine_ngx::shutdown();
    sunshine_ngx::testing::set_callbacks({});
  }
  void unknown_feature_dump_test() {
    initialize(GetModuleHandleW(nullptr), false, true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &) -> std::uint64_t { ++captures; return 91; },
      [](std::uint64_t, bool) { ++finishes; },
      [](std::uint64_t, std::uint64_t) { ++retirements; }});
    fixture_parameters parameters;
    auto *first = reinterpret_cast<void *>(handle_bits);
    auto *second = reinterpret_cast<void *>(handle_bits + 16);
    const auto captures_before = captures, finishes_before = finishes, retirements_before = retirements;
    const auto originals_before = actual_evaluations.load();
    sunshine_game3d::arm_diagnostic_metadata(false);
    sunshine_game3d::arm_diagnostic_metadata(true);
    const auto evaluate_unknown = [&](void *handle, result expected) {
      return_result = expected;
      SetLastError(input_error);
      require(evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback) == expected &&
        GetLastError() == output_error && observed.incoming_error == input_error,
        "unknown-feature dump altered original evaluation/result/LastError");
    };
    evaluate_unknown(first, ngx_success);
    auto dumped = nlohmann::json::parse(sunshine_game3d::diagnostic_metadata_json());
    require(dumped["ngx"].size() == 1, "unknown NGX evaluation omitted diagnostic evidence");
    const auto &entry = dumped["ngx"][0];
    require(entry["feature_known"] == false && entry["feature"].is_null() && entry["source_id"] == 0 &&
      entry["create_width"].is_null() && entry["create_flags"].is_null() && entry["result_known"] == true && entry["successful"] == true &&
      entry["sequence_domain"] == "diagnostic-unknown-feature", "unknown feature fabricated capture/creation metadata or lost original result");
    require(entry["owner_identity"] != "0x0" && entry["feature_handle_identity"] == "0x123456780", "unknown evaluation lost owner/handle identity");
    bool found_depth = false;
    for (const auto &value : entry["parameters"]) if (value["name"] == "Depth")
      found_depth = value["successful"] == true && value["value"] == "0xface0000";
    require(found_depth, "unknown evaluation lost available named parameters");
    evaluate_unknown(first, ngx_failure);
    evaluate_unknown(second, ngx_success);
    dumped = nlohmann::json::parse(sunshine_game3d::diagnostic_metadata_json());
    require(dumped["ngx"].size() == 2 && dumped["ngx"][0]["successful"] == false && dumped["ngx"][1]["successful"] == true,
      "source ID zero misassociated results across unknown handles");
    require(captures == captures_before && finishes == finishes_before && retirements == retirements_before &&
      actual_evaluations == originals_before + 3, "unknown diagnostic observations changed capture admission or original call count");
    sunshine_game3d::arm_diagnostic_metadata(false);
    const auto resource_queries = feedback_requests.size(), pointer_queries = probe_requests.size(), float_queries = jitter_requests.size();
    evaluate_unknown(first, ngx_success);
    require(feedback_requests.size() == resource_queries && probe_requests.size() == pointer_queries && jitter_requests.size() == float_queries,
      "disarmed unknown-feature evaluation performed diagnostic getters");
    require(nlohmann::json::parse(sunshine_game3d::diagnostic_metadata_json())["ngx"] == dumped["ngx"],
      "disarmed unknown evaluation mutated retained diagnostic evidence");
    shutdown();
    sunshine_ngx::testing::set_callbacks({});
  }
  void diagnostic_completion_identity_test() {
    initialize(GetModuleHandleW(nullptr), true, true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &) -> std::uint64_t { return 91; },
      [](std::uint64_t, bool) {},
      [](std::uint64_t, std::uint64_t) {}});
    fixture_parameters parameters;
    parameters.ui_hint_available = true;
    parameters.ui_hint = reinterpret_cast<void *>(0xcafebeef);
    void *handle {};
    return_result = ngx_success;
    require(create_entry(reinterpret_cast<void *>(0x123), 1, &parameters, &handle) == ngx_success,
      "diagnostic identity fixture could not create known DLSS feature");
    sunshine_game3d::diagnostic::set_resource_callbacks(&diagnostic_callbacks);
    for (unsigned known = 1; ; --known) {
      for (const auto expected : {ngx_success, ngx_failure}) {
        sunshine_game3d::arm_diagnostic_metadata(false);
        sunshine_game3d::arm_diagnostic_metadata(true);
        diagnostic_resources = diagnostic_finishes = 0;
        diagnostic_provider_valid = false;
        const auto originals = actual_evaluations.load();
        const auto command = std::uint64_t(0x432100 + known);
        return_result = expected;
        SetLastError(input_error);
        require(evaluate_entry(reinterpret_cast<void *>(command), handle, &parameters, &callback) == expected &&
            GetLastError() == output_error, "diagnostic observer changed SDK result/LastError");
        require(diagnostic_resources == 1 && diagnostic_finishes == 1 && diagnostic_provider_valid &&
            diagnostic_originals_before == originals && diagnostic_originals_after == originals + 1,
            "resource observation/finish did not bracket the real vendor evaluation exactly once");
        const auto &before = diagnostic_resource.observation;
        const auto &after = diagnostic_finish;
        require(diagnostic_resource.readable && diagnostic_resource.descriptor_supported && diagnostic_resource.native == 0xcafebeef,
            "NGX diagnostic observer lost available optional resource");
        require(before.session && before.epoch && before.sequence && before.command == command &&
            after.session == before.session && after.epoch == before.epoch && after.sequence == before.sequence &&
            after.tick == before.tick && after.command == before.command && after.viewport == before.viewport &&
            after.frame_token == before.frame_token && after.frame_numeric == before.frame_numeric && after.numeric_frame == before.numeric_frame,
            "NGX after_evaluate lost the original diagnostic command/stamp needed to finish optional capture");
        require(diagnostic_finish_source == diagnostic_resource.source_id &&
            (known ? diagnostic_finish_source != 0 : diagnostic_finish_source == 0) &&
            diagnostic_success == (expected == ngx_success), "known/unknown diagnostic completion lost source or result");
      }
      if (!known) break;
      return_result = ngx_success;
      require(release_entry(handle) == ngx_success, "diagnostic identity fixture could not remove known feature");
    }
    sunshine_game3d::diagnostic::set_resource_callbacks(nullptr);
    sunshine_game3d::arm_diagnostic_metadata(false);
    return_result = ngx_success;
    shutdown();
    sunshine_ngx::testing::set_callbacks({});
  }
  void detach_test() {
    initialize(GetModuleHandleW(nullptr), true, true);
    sunshine_ngx::testing::set_callbacks({
      [](std::uint64_t, const sunshine_scene_depth::frame &) -> std::uint64_t { ++captures; return 91; },
      [](std::uint64_t, bool) { ++finishes; },
      [](std::uint64_t, std::uint64_t) { ++retirements; }});
    fixture_parameters parameters;
    void *handle{};
    require(create_entry(reinterpret_cast<void *>(0x123), 1, &parameters, &handle) == ngx_success,
      "detach fixture feature create failed");
    const auto capture_before = captures, finish_before = finishes, retire_before = retirements;
    const auto original_before = actual_evaluations.load();
    const auto trace_before = testing::counts().ngx_evaluate;
    block_original = true; original_entered = false; release_original = false;
    std::atomic<bool> original_ok{};
    std::thread detaching_call([&] {
      SetLastError(input_error);
      original_ok = evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback) == ngx_success &&
        GetLastError() == output_error;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!original_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool entered = original_entered.load(std::memory_order_acquire);
    sunshine_addon_lifetime::begin_detach();
    release_original = true; detaching_call.join(); block_original = false;
    SetLastError(input_error);
    const auto evaluated = evaluate_entry(reinterpret_cast<void *>(0x123), handle, &parameters, &callback);
    const bool evaluate_error = GetLastError() == output_error;
    const auto releases_before = actual_releases.load();
    SetLastError(input_error);
    const auto released = release_entry(handle);
    const bool detach_ok = entered && original_ok && evaluated == ngx_success && evaluate_error &&
      released == ngx_success && GetLastError() == output_error && actual_releases == releases_before + 1 &&
      actual_evaluations == original_before + 2 && captures == capture_before + 1 &&
      finishes == finish_before && retirements == retire_before && testing::counts().ngx_evaluate == trace_before &&
      !enabled() && !capture_enabled() && sunshine_ngx::epoch() == 0;
    // Test-only reset: production has no reverse transition after DllMain.
    sunshine_addon_lifetime::detaching.store(false, std::memory_order_release);
    shutdown();
    sunshine_ngx::testing::set_callbacks({});
    require(detach_ok, "process detach changed cached NGX forwarding or completed/retired an old capture");
  }
}

int main() {
  try {
    basic_test();
    std::puts("PASS actual export discovery, disabled/no-call coverage, C/C++ evaluate, args/results/LastError, nested SL and feature lifetime");
    epoch_test();
    std::puts("PASS delayed-original shutdown/reinitialize, retained hooks and unreadable output safety");
    capture_test();
    std::puts("PASS default-off trace with independent NGX capture, named getters, crop/orientation, feature lifetime, nested and SL deduplication");
    reset_feedback_test();
    std::puts("PASS per-evaluation reset feedback, persistent scene revisions, no unused metadata queries and unchanged calls/LastError");
    calibration_probe_test();
    std::puts("PASS opt-in typed NGX calibration presence, poisoned pointers, per-feature throttle, missing getters and unchanged calls/LastError/capture");
    jitter_metadata_test();
    std::puts("PASS per-evaluation NGX jitter, signed/zero offsets, cropped/legacy render domains, missing/nonfinite inputs and optional float export");
    concurrent_feedback_test();
    std::puts("PASS concurrent NGX reset getter completion preserves scene revision and source sequence ordering");
    unknown_feature_dump_test();
    std::puts("PASS armed unknown-feature NGX diagnostic parameters/results with unchanged capture rejection, original calls and disarmed getter silence");
    diagnostic_completion_identity_test();
    std::puts("PASS real NGX before/after evaluation preserves full diagnostic stamp and source across known/unknown feature success/failure");
    detach_test();
    std::puts("PASS cached NGX forwarding across process detach suppresses capture, late completion, release and tracing");
    return 0;
  } catch (const std::exception &error) {
    release_original.store(true);
    std::fprintf(stderr, "FAIL upscaler call trace: %s\n", error.what());
    return 1;
  }
}

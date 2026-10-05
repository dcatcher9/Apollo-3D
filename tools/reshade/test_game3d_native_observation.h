// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Current native Game 3D observations for runtime fixtures with no installed
// FX. Include after test_raw_runtime_fixture.h. Every present may query the
// test add-on's passive Automatic/scale state and, with a streaming consumer,
// read its exact exported SBS. A checkpoint requests one production Dump 3D
// capture: the depth, constants and SBS consumed by one native render.
// Nothing here injects depth, readiness, pixels or completion.
#include "game3d_controls_model.h"
#include "game3d_test_render.h"
#include "../../src/game3d_debug_protocol.h"
#include <nlohmann/json.hpp>

namespace {
  // Call before initialize(). A native fixture boots the official runtime with
  // the frozen SunshineGame3D effect of its supplied shader directory and the
  // test add-on, then removes the effect (native_game3d_observer::start). It
  // selects that boot effect, its compile gate and the test add-on itself, as
  // reshade_game3d_native_provider_runtime_test does, so no caller environment
  // is required; preserve2 also selects DepthCopyBeforeClears=2.
  inline void select_native_boot(bool preserve2 = false) {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 &&
      (!preserve2 || _putenv_s("SUNSHINE_DEPTH_BIND_SWITCH_TEST", "1") == 0),
      "Could not select the native fixture's boot effect and test add-on");
  }

  struct automatic_status {
    unsigned flags{}, basis{}, scale_state{};
    float scale{};
    bool ready() const { return flags & 4u; }
    bool active_scale() const {
      return scale_state == unsigned(sunshine_game3d::automatic_scale_state::active) && std::isfinite(scale) && scale > 0.f;
    }
  };

  // One native render, as recorded by the production diagnostic owner.
  struct native_render {
    bool depth_ready{}, reused_depth{}, camera_ready{}, projection_supplied{};
    std::uint64_t source_resource{}, captured_resource{}, source_id{}, frame_index{}, provider_source_id{};
    unsigned width{}, height{}, x{}, y{}, active_width{}, active_height{}, allocation_format{}, detected_orientation{};
    std::string provider;
    float depth_scale{}, strength_blend{};
    int coordinate_basis{};
    std::array<float, 2> convergence{};
    std::array<float, 4> depth_rect{};
    // Full consumed R32F allocation when depth was ready; packed SBS on request.
    std::vector<std::uint8_t> raw_depth, sbs;
    float depth(unsigned column, unsigned row) const {
      float value{};
      std::memcpy(&value, raw_depth.data() + (size_t(row) * width + column) * sizeof(float), sizeof(float));
      return value;
    }
  };

  class native_game3d_observer {
  public:
    explicit native_game3d_observer(raw_runtime_fixture &fixture): fixture_(fixture) {}
    native_game3d_observer(const native_game3d_observer &) = delete;
    native_game3d_observer &operator=(const native_game3d_observer &) = delete;
    ~native_game3d_observer() {
      if (state_) {
        if (request_) InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state_->released_id), request_);
        UnmapViewOfFile(state_);
      }
      if (mapping_) CloseHandle(mapping_);
      if (export_state_) UnmapViewOfFile(export_state_);
      if (export_mapping_) CloseHandle(export_mapping_);
      if (set_foreground_) set_foreground_(nullptr);
    }

    // Call after the runtime has initialized. Like an ordinary installation,
    // no FX remains: native Game 3D is the only renderer and depth consumer.
    void start() {
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      query_automatic_ = reinterpret_cast<query_automatic_t>(GetProcAddress(module, "SunshineGame3DTestQueryAutomatic"));
      query_scale_ = reinterpret_cast<query_scale_t>(GetProcAddress(module, "SunshineGame3DTestQueryScale"));
      edit_float_ = reinterpret_cast<edit_float_t>(GetProcAddress(module, "SunshineGame3DTestEditFloat"));
      recalibrate_ = reinterpret_cast<action_t>(GetProcAddress(module, "SunshineGame3DTestRecalibrate"));
      set_foreground_ = reinterpret_cast<set_foreground_t>(GetProcAddress(module, "SunshineSbsTestSetForeground"));
      last_render_ = reinterpret_cast<last_render_t>(GetProcAddress(module, "SunshineGame3DTestLastRender"));
      require(module && query_automatic_ && query_scale_ && edit_float_ && recalibrate_ && set_foreground_,
        "Native Game 3D fixture requires the test add-on's passive UI queries, controls and foreground observer");
      const auto effects = fixture_.runtime_directory / "effects";
      fs::rename(effects / effect_file, effects / (std::string(effect_file) + ".disabled"));
      const auto reloads = observed.reloads;
      observed.runtime->reload_effect_next_frame(nullptr);
      const auto until = GetTickCount64() + 30000;
      unsigned count = 1;
      do {
        fixture_.step();
        count = techniques();
      } while ((observed.reloads == reloads || count) && GetTickCount64() < until);
      require(observed.reloads > reloads && !count, "FX remained loaded before native Game 3D validation");
      for (const auto &entry : fs::recursive_directory_iterator(effects))
        require(entry.path().extension() != ".fx", "Native fixture still has an installed FX");
      fx_renders_ = observed.renders;
      no_effects();
      // Select the hidden fixture window for diagnostics and export without
      // moving desktop focus; this is the foreground game's ordinary path.
      set_foreground_(fixture_.window);
      const auto name = std::wstring(game3d_debug::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
      require(mapping_ != nullptr, "Loaded add-on did not publish its production Dump3D mailbox");
      state_ = static_cast<game3d_debug::shared_state_t *>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(game3d_debug::shared_state_t)));
      require(state_ && state_->signature == game3d_debug::magic && state_->protocol_version == game3d_debug::version,
        "Production Dump3D mailbox is incompatible");
      FILETIME created{}, exited{}, kernel{}, user{};
      require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "Identify native render observer");
      state_->consumer_pid = GetCurrentProcessId();
      state_->consumer_creation_time = (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state_->consumer_nonce), 0x4e41544956453344LL);
      evidence_.open(fixture_.runtime_directory / "native-renders.jsonl");
      require(evidence_.good(), "Cannot record native render evidence");
    }

    void no_effects() const {
      require(!techniques() && observed.renders == fx_renders_ && !observed.inject && observed.ready_uniforms.empty(),
        "Native Game 3D depended on an FX technique or injected effect state");
    }

    automatic_status automatic() const {
      automatic_status result;
      require(query_automatic_(observed.runtime, &result.flags) &&
        query_scale_(observed.runtime, &result.basis, &result.scale_state, &result.scale), "Native Game 3D UI observation failed");
      require(result.flags & 1u, "Native Game 3D became disabled");
      return result;
    }

    void set_strength(float value) {
      require(edit_float_(observed.runtime, unsigned(sunshine_game3d::control::strength), value), "Native Game 3D rejected a strength edit");
    }

    bool recalibrate() { return recalibrate_(observed.runtime) != FALSE; }

    // What the last native render consumed: its depth identity, allocation
    // and readiness, and its constants. Per Present, without a dump; the
    // caller compares sequence to know a render followed its Present.
    sunshine_game3d::test::last_render last_render() const {
      sunshine_game3d::test::last_render result;
      require(last_render_ && last_render_(observed.runtime, &result), "Native Game 3D last-render observation failed");
      return result;
    }

    // Presents through the caller until native Game 3D has rendered: the game
    // stays 2D while the renderer compiles its shaders on the thread pool.
    // Returns the last render's sequence; every later Present renders.
    std::uint64_t await_render(const std::function<void()> &present) const {
      const auto until = GetTickCount64() + 30000;
      auto render = last_render();
      while (!render.sequence && GetTickCount64() < until) { present(); render = last_render(); }
      require(render.sequence != 0, "Native Game 3D never rendered; its shaders did not finish compiling");
      return render.sequence;
    }

    // Become the production streaming consumer. Every later native present
    // publishes its packed SBS to the shared export ring.
    void attach_export() {
      const auto name = std::wstring(reshade_bridge::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      export_mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
      require(export_mapping_ != nullptr, "Native fixture cannot open the production exporter mapping");
      export_state_ = static_cast<reshade_bridge::shared_state_t *>(MapViewOfFile(export_mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(reshade_bridge::shared_state_t)));
      require(export_state_ != nullptr, "Native fixture cannot map the production exporter state");
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&export_state_->consumer_nonce), 0x4e41544956455850LL);
    }

    // The newest completed publication, which belongs to the latest present.
    std::vector<std::uint8_t> exported() {
      namespace wire = reshade_bridge;
      require(export_state_ != nullptr, "Native export was read without a streaming consumer");
      const auto before = InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&export_state_->metadata_sequence), 0, 0);
      require(!(before & 1), "Native export metadata was changing after a completed Present");
      const auto metadata = export_state_->metadata;
      MemoryBarrier();
      require(before == InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&export_state_->metadata_sequence), 0, 0) &&
        wire::valid_metadata(metadata) && metadata.source_width == width && metadata.source_height == height,
        "Native export lost coherent source metadata");
      if (metadata.generation != export_generation_) {
        export_generation_ = metadata.generation;
        export_sequence_ = 0;
      }
      unsigned index = wire::slot_count;
      std::uint64_t sequence{};
      for (unsigned i = 0; i < wire::slot_count; ++i) {
        const auto control = load(export_state_->slots[i].control), candidate = load(export_state_->slots[i].sequence);
        if (wire::control_generation(control) == metadata.generation && wire::control_state(control) == wire::slot_state::ready && candidate > sequence) {
          index = i;
          sequence = candidate;
        }
      }
      require(index < wire::slot_count && sequence > export_sequence_, "Native export did not publish the current presentation");
      auto &slot = export_state_->slots[index];
      const auto ready = wire::slot_control(metadata.generation, wire::slot_state::ready);
      const auto reading = wire::slot_control(metadata.generation, wire::slot_state::reading);
      require(std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), reading, ready)) == ready &&
        load(slot.sequence) == sequence, "Native export changed while claiming its published slot");
      const auto release = [&] { InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), ready, reading); };
      try {
        com_ptr<ID3D12Fence> fence;
        com_ptr<ID3D12Resource> texture;
        checked(fixture_.game->OpenSharedHandle(reinterpret_cast<HANDLE>(metadata.ready_fence_handle), IID_PPV_ARGS(fence.put())), "Open native export fence");
        require(fence->GetCompletedValue() != UINT64_MAX && fence->GetCompletedValue() >= sequence, "Native export preceded GPU completion");
        checked(fixture_.game->OpenSharedHandle(reinterpret_cast<HANDLE>(metadata.texture_handles[index]), IID_PPV_ARGS(texture.put())), "Open native export texture");
        auto pixels = fixture_.read(texture.p, D3D12_RESOURCE_STATE_COMMON);
        release();
        export_sequence_ = sequence;
        return pixels;
      } catch (...) {
        release();
        throw;
      }
    }

    // Presents through the caller until the production owner completes one
    // native render capture. The caller's present also checks each frame.
    native_render capture(const char *label, bool sbs, const std::function<void()> &present) {
      request_ = state_->request_id + 1;
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state_->request_id), request_);
      const auto until = GetTickCount64() + 10000;
      while (load(state_->response_id) != request_ && GetTickCount64() < until) present();
      require(load(state_->response_id) == request_, "Production Dump3D did not capture a native render");
      const auto &response = state_->response;
      require(response.result == game3d_debug::status::complete && response.json_bytes <= game3d_debug::max_json_bytes &&
        response.texture_count <= game3d_debug::max_textures, "Native render capture did not complete");
      const auto metadata = nlohmann::json::parse(std::string(state_->json, response.json_bytes));
      native_render frame;
      const auto &depth = metadata.at("consumed_depth");
      frame.depth_ready = depth.at("ready");
      frame.reused_depth = depth.at("reused_depth");
      frame.source_resource = depth.at("source_resource");
      frame.captured_resource = depth.at("captured_resource");
      frame.source_id = depth.at("source_id");
      frame.frame_index = depth.at("frame_index");
      frame.width = depth.at("width");
      frame.height = depth.at("height");
      const auto &rect = depth.at("active_rect");
      frame.x = rect.at(0); frame.y = rect.at(1); frame.active_width = rect.at(2); frame.active_height = rect.at(3);
      frame.provider = depth.at("provider");
      frame.provider_source_id = depth.at("provider_source_id");
      frame.projection_supplied = depth.at("projection").at("supplied");
      frame.detected_orientation = depth.at("detected_orientation");
      if (depth.contains("allocation")) frame.allocation_format = depth.at("allocation").at("api_format");
      const auto &parameters = metadata.at("render_parameters");
      frame.camera_ready = parameters.at("camera_ready").get<unsigned>() != 0;
      frame.coordinate_basis = parameters.at("coordinate_basis");
      frame.depth_scale = parameters.at("depth_scale");
      frame.strength_blend = parameters.at("strength_blend");
      frame.convergence = {parameters.at("convergence").at(0), parameters.at("convergence").at(1)};
      for (unsigned i = 0; i < 4; ++i) frame.depth_rect[i] = parameters.at("depth_rect").at(i);
      require((parameters.at("depth_ready").get<unsigned>() != 0) == frame.depth_ready, "Native constants disagree with the consumed depth");
      for (unsigned i = 0; i < response.texture_count; ++i) {
        const auto &item = response.textures[i];
        const bool depth_artifact = item.kind == game3d_debug::artifact::raw_depth;
        if (!depth_artifact && !(sbs && item.kind == game3d_debug::artifact::sbs)) continue;
        com_ptr<ID3D12Resource> texture;
        checked(fixture_.game->OpenSharedHandle(reinterpret_cast<HANDLE>(item.handle), IID_PPV_ARGS(texture.put())), "Open native render artifact");
        if (depth_artifact) {
          require(item.dxgi_format == DXGI_FORMAT_R32_FLOAT && item.width == frame.width && item.height == frame.height,
            "Consumed native depth artifact differs from its allocation");
          frame.raw_depth = fixture_.read(texture.p, D3D12_RESOURCE_STATE_COMMON);
        } else {
          require(item.width == 2 * width && item.height == height, "Native SBS artifact has the wrong geometry");
          frame.sbs = fixture_.read(texture.p, D3D12_RESOURCE_STATE_COMMON);
        }
      }
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&state_->released_id), request_);
      request_ = 0;
      require(frame.depth_ready == !frame.raw_depth.empty(), "Native render capture lost its consumed depth artifact");
      require(!sbs || !frame.sbs.empty(), "Native render capture omitted its SBS");
      const auto &policy = metadata.at("render_scene_policy");
      evidence_ << nlohmann::json{{"label", label}, {"wall_ms", GetTickCount64()}, {"consumed_depth", depth},
        {"render_parameters", parameters}, {"render_scene_policy", {{"phase", policy.at("phase")}, {"basis", policy.at("basis")},
        {"active", policy.at("active")}, {"current_scale", policy.at("current_scale")}}}}.dump() << '\n';
      require(evidence_.good(), "Cannot record native render evidence");
      return frame;
    }

  private:
    using query_automatic_t = BOOL (*)(api::effect_runtime *, unsigned *);
    using query_scale_t = BOOL (*)(api::effect_runtime *, unsigned *, unsigned *, float *);
    using edit_float_t = BOOL (*)(api::effect_runtime *, unsigned, float);
    using action_t = BOOL (*)(api::effect_runtime *);
    using set_foreground_t = void (*)(HWND);
    using last_render_t = BOOL (*)(api::effect_runtime *, sunshine_game3d::test::last_render *);

    static std::uint64_t load(std::uint64_t &value) {
      return std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&value), 0, 0));
    }
    static unsigned techniques() {
      unsigned count{};
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++count; });
      return count;
    }

    raw_runtime_fixture &fixture_;
    query_automatic_t query_automatic_{};
    query_scale_t query_scale_{};
    edit_float_t edit_float_{};
    action_t recalibrate_{};
    set_foreground_t set_foreground_{};
    last_render_t last_render_{};
    unsigned fx_renders_{};
    HANDLE mapping_{}, export_mapping_{};
    game3d_debug::shared_state_t *state_{};
    reshade_bridge::shared_state_t *export_state_{};
    std::uint64_t request_{}, export_generation_{}, export_sequence_{};
    std::ofstream evidence_;
  };
}

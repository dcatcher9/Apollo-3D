// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Current native Game 3D observations for runtime fixtures with no installed
// FX. Include after test_raw_runtime_fixture.h. Every present may query the
// test add-on's passive Automatic/scale state and, with a streaming consumer,
// read its exact exported SBS. A checkpoint requests one production Dump 3D
// capture: the depth, constants and SBS consumed by one native render.
// Nothing here injects depth, readiness, pixels or completion.
#include "game3d_controls_model.h"
#include "game3d_frame_record.h"
#include "game3d_test_depth_events.h"
#include "game3d_test_render.h"
#include "../../src/game3d_debug_protocol.h"
#include "test_fixture_files.h"
#include "test_game3d_present_trace.h"
#include <cstring>
#include <nlohmann/json.hpp>

namespace {
  // The per-Present trace of a test add-on fixture (test_game3d_present_trace.h):
  // one line per Present, read from the test add-on after that Present (its
  // frame record, last native render, Generic hold and presenting queue's
  // depth choice). It starts at the first Present with a native render, so
  // the shader compilation before it adds no line. A fixture reads it where
  // it observes each Present: native_game3d_observer::last_render, or the
  // native provider fixture's reshade_present event (record_present), which
  // also sees the Presents a fixture makes generated. Each line ends with the
  // add-on's DEPTH binding and readiness writes since the line before (T8,
  // game3d_test_depth_events.h); every call collects them, and the first
  // line of a trace has only its own Present's.
  class native_present_trace {
  public:
    // Appends this Present's line to the open trace once: when its frame
    // record's Present or its native render's sequence moved since the
    // trace's last line. record and render: what the caller already read
    // after this Present, if anything.
    void present(api::effect_runtime *runtime, const sunshine_game3d::frame_record *record = nullptr,
        const sunshine_game3d::test::last_render *render = nullptr) {
      auto &out = game3d_present_trace::out;
      if (!runtime || !bind()) return;
      // A new trace takes no event from before it opened.
      if (!out.active() || generation_ != out.generation()) drop_events();
      collect_events();
      if (!out.active()) {
        drop_events();
        return;
      }
      if (generation_ != out.generation()) {
        generation_ = out.generation();
        traced_ = false;
      }
      sunshine_game3d::frame_record value {};
      if (record) value = *record;
      else if (!frame_record_(runtime, &value)) value = {};
      sunshine_game3d::test::last_render rendered {};
      if (render) rendered = *render;
      else if (!last_render_(runtime, &rendered)) rendered = {};
      if (!rendered.sequence) {
        // Before the trace's first line its events are dropped: how many
        // Presents precede the first native render is timing.
        if (!traced_) drop_events();
        return;
      }
      // An unchanged Present keeps its events for the next line.
      if (traced_ && value.present == last_present_ && rendered.sequence == last_sequence_) return;
      const bool first = !traced_;
      traced_ = true;
      last_present_ = value.present;
      last_sequence_ = rendered.sequence;
      line(runtime, value, rendered, first);
    }

  private:
    using frame_record_t = BOOL (*)(api::effect_runtime *, sunshine_game3d::frame_record *);
    using last_render_t = BOOL (*)(api::effect_runtime *, sunshine_game3d::test::last_render *);
    using generic_hold_t = BOOL (*)(api::effect_runtime *, sunshine_game3d::test::generic_hold *);
    using queue_choice_t = BOOL (*)(api::effect_runtime *, sunshine_streamline::depth_capture::depth_choice *);
    using depth_event = sunshine_game3d::test::depth_events::event;
    using binding_events_t = unsigned (*)(std::uint64_t *, depth_event *, unsigned, std::uint64_t *);

    bool bind() {
      if (frame_record_ && last_render_) return true;
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      if (!module) return false;
      frame_record_ = reinterpret_cast<frame_record_t>(GetProcAddress(module, "SunshineGame3DTestFrameRecord"));
      last_render_ = reinterpret_cast<last_render_t>(GetProcAddress(module, "SunshineGame3DTestLastRender"));
      generic_hold_ = reinterpret_cast<generic_hold_t>(GetProcAddress(module, "SunshineDepthTestGenericHold"));
      queue_choice_ = reinterpret_cast<queue_choice_t>(GetProcAddress(module, "SunshineDepthTestQueueChoice"));
      binding_events_ = reinterpret_cast<binding_events_t>(GetProcAddress(module, "SunshineDepthTestBindingEvents"));
      return frame_record_ && last_render_;
    }

    // T8: the binding events after the cursor, into the fixed pending list
    // (no allocation). A full list drops its older half, so it keeps the
    // newest events (the first call after a trace opens can read the whole
    // ring); those dropped, and those the add-on's ring overwrote first, are
    // counted.
    void collect_events() {
      if (!binding_events_) return;
      for (;;) {
        depth_event batch[64];
        std::uint64_t skipped {};
        const unsigned count = binding_events_(&event_cursor_, batch, unsigned(std::size(batch)), &skipped);
        events_skipped_ += skipped;
        for (unsigned i = 0; i != count; ++i) {
          if (pending_count_ == pending_.size()) {
            constexpr std::size_t half = pending_capacity / 2;
            std::copy(pending_.begin() + half, pending_.end(), pending_.begin());
            pending_count_ -= half;
            events_skipped_ += half;
          }
          pending_[pending_count_++] = batch[i];
        }
        if (count < std::size(batch)) break;
      }
    }
    void drop_events() {
      pending_count_ = 0;
      events_skipped_ = 0;
    }

    // t8=: each pending event as runtime.present.phase.site.value, comma
    // separated, oldest first. runtime: its rank; present: the newest
    // event's Present ordinal minus its own (0: the newest Present's);
    // phase and site: game3d_test_depth_events.h's ids; value: b1 or b0 for
    // a binding (a view or null; also site 10's display eviction record in
    // traces before 11-01), r1 or r0 for a readiness write, x for a reset that writes neither (provider_runtime_reload). t8skip=n
    // (only when n is not 0): events missing from the list. A trace's first
    // line lists only its newest Present's events: the older ones came with
    // Presents before the trace's first line, whose number is timing (a
    // fixture may present many times between opening the trace and its
    // first observation), and so does any count of missing ones.
    void events(game3d_present_trace::writer &out, bool first) {
      using sunshine_game3d::test::depth_events::not_applicable;
      std::uint64_t newest = 0;
      for (std::size_t i = 0; i != pending_count_; ++i) newest = std::max(newest, pending_[i].present);
      out.add(" t8=");
      bool listed = false;
      for (std::size_t i = 0; i != pending_count_; ++i) {
        const auto &event = pending_[i];
        if (first && event.present != newest) continue;
        out.add("%s%u.%llu.%u.%u.", listed ? "," : "", out.rank(game3d_present_trace::runtimes, event.runtime),
          static_cast<unsigned long long>(newest - event.present), unsigned(event.phase), unsigned(event.site));
        if (event.view != not_applicable) out.add("b%u", unsigned(event.view));
        else if (event.ready != not_applicable) out.add("r%u", unsigned(event.ready));
        else out.add("x");
        listed = true;
      }
      if (events_skipped_ && !first) out.add(" t8skip=%llu", static_cast<unsigned long long>(events_skipped_));
      drop_events();
    }

    // A Present ordinal relative to this line's Present ("-" without one).
    static void delta(game3d_present_trace::writer &out, std::uint64_t present, std::uint64_t value) {
      if (!value) out.add("-");
      else if (!present) out.add("?");
      else out.add("%lld", static_cast<long long>(present - value));
    }

    static bool default_row(const sunshine_game3d::source_status &row) {
      return row.state == sunshine_game3d::source_state::inactive && row.key.api == sunshine_game3d::source_api::none &&
        !row.key.viewport && !row.key.id && !row.epoch && row.fallback_to == sunshine_game3d::source_slot::none &&
        row.reason && !std::strcmp(row.reason, "not_recorded");
    }

    // The line (test_game3d_present_trace.h): Present and render (p, n, r),
    // the frame record's owner, read order, shown capture and non-default
    // rows; the consumed depth (rdy..ori), the render constants' integer
    // fields (pdr..pdv), the frame decision (gap, ss, did), the reentry ramp
    // (reh, rph, rb), the queue's depth choice (ch, cd), Generic's hold (gh)
    // and the DEPTH binding events (t8).
    void line(api::effect_runtime *runtime, const sunshine_game3d::frame_record &record,
        const sunshine_game3d::test::last_render &render, bool first) {
      using namespace game3d_present_trace;
      using sunshine_game3d::name;
      auto &out = game3d_present_trace::out;
      out.begin();
      if (record.present) out.add("p=%lld", out.present(record.present));
      else out.add("p=?");
      out.add(" n=%lld r=%u own=%s ro=%s cap=%u", out.sequence(render.sequence), unsigned(render.rendered),
        name(record.depth_owner), name(record.depth_read_order), out.rank(capture, record.depth_capture));
      for (std::size_t slot = 0; slot != sunshine_game3d::source_slot_count; ++slot) {
        const auto &row = record.rows[slot];
        if (default_row(row)) continue;
        out.add(" %s=%s,", name(static_cast<sunshine_game3d::source_slot>(slot)), name(row.state));
        out.token(row.reason);
        out.add(",%s,%s.%u.%u,%u", name(row.fallback_to), name(row.key.api), unsigned(row.key.viewport),
          out.rank(identity, row.key.id), out.rank(epoch, row.epoch));
      }
      const auto &depth = render.depth;
      out.add(" rdy=%u reu=%u src=%u geo=%ux%u+%u+%u/%ux%u fga=%u cal=%u ori=%u", unsigned(depth.ready),
        unsigned(depth.reused_depth), out.rank(source, depth.source_id), unsigned(depth.width), unsigned(depth.height),
        unsigned(depth.x), unsigned(depth.y), unsigned(depth.active_width), unsigned(depth.active_height),
        unsigned(depth.frame_generation_active), unsigned(depth.calibrated), unsigned(depth.orientation));
      const auto &parameters = render.parameters;
      out.add(" pdr=%u pcr=%u pcb=%d pdv=%d", unsigned(parameters.depth_ready), unsigned(parameters.camera_ready),
        int(parameters.coordinate_basis), int(parameters.depth_view));
      // A Generic capture's scene source is runtime epoch, source, frame and
      // layout epoch | bit 63 (exporter.cpp generic_scene_source); an API
      // capture's ends with its viewport.
      const auto &scene = render.scene_source;
      constexpr std::uint64_t generic_bit = std::uint64_t {1} << 63;
      out.add(" gap=%u ss=%u.%u.%u.", unsigned(render.gap), out.rank(epoch, scene[0]), out.rank(source, scene[1]),
        out.rank(sequence, scene[2]));
      if (scene[3] & generic_bit) out.add("g%u", out.rank(epoch, scene[3] & ~generic_bit));
      else out.add("v%llu", static_cast<unsigned long long>(scene[3]));
      out.add(" did=%u reh=%u rph=%u rb=%u", out.rank(identity, render.depth_identity), unsigned(render.reentry_heard),
        unsigned(render.reentry_phase), unsigned(render.reentry_basis));
      sunshine_streamline::depth_capture::depth_choice choice {};
      if (queue_choice_ && queue_choice_(runtime, &choice)) {
        const auto &decision = choice.decision;
        out.add(" ch=");
        delta(out, record.present, choice.present);
        out.add(",%s,%u,%u,%u,%u,%u%u,%u,%u,%u", name(choice.rung), unsigned(choice.result), unsigned(choice.selection),
          unsigned(choice.failure), unsigned(choice.pixel_ready), unsigned(choice.fg_required), unsigned(choice.fg_excluded),
          unsigned(choice.owner_established), unsigned(choice.owner), out.rank(source, choice.owner_source));
        out.add(" cd=%u,%u,%u,%u,%u,%u,%u,%u%u%u,", unsigned(decision.source_selected), unsigned(decision.provider),
          out.rank(epoch, decision.epoch), out.rank(sequence, decision.sequence), out.rank(source, decision.source_id),
          out.rank(capture, decision.capture_id), unsigned(decision.viewport), unsigned(decision.source_valid),
          unsigned(decision.repeated_frame), unsigned(decision.pending_frame));
        delta(out, record.present, decision.presenting_floor);
      } else out.add(" ch=- cd=-");
      sunshine_game3d::test::generic_hold hold {};
      if (generic_hold_ && generic_hold_(runtime, &hold))
        out.add(" gh=%u%u%u,%u,%u", unsigned(hold.held), unsigned(hold.hold_only), unsigned(hold.capture_ready),
          unsigned(hold.verdict), out.rank(identity, hold.shown_identity));
      else out.add(" gh=-");
      // T8 (refactor plan 3.4): the DEPTH binding events since the line before.
      events(out, first);
      out.end();
    }

    frame_record_t frame_record_ {};
    last_render_t last_render_ {};
    generic_hold_t generic_hold_ {};
    queue_choice_t queue_choice_ {};
    binding_events_t binding_events_ {};
    std::uint64_t generation_ {}, last_present_ {}, last_sequence_ {};
    bool traced_ {};
    // T8: the events collected since the last line, and the ring's cursor.
    static constexpr std::size_t pending_capacity = 512;
    std::array<depth_event, pending_capacity> pending_ {};
    std::size_t pending_count_ {};
    std::uint64_t event_cursor_ {}, events_skipped_ {};
  } native_trace;

  // Call before initialize(). A native fixture boots the official runtime with
  // the frozen SunshineGame3D effect of its supplied shader directory and the
  // test add-on, then removes the effect (native_game3d_observer::start). It
  // selects that boot effect, its compile gate and the test add-on itself, as
  // reshade_game3d_native_provider_runtime_test does, so no caller environment
  // is required.
  inline void select_native_boot() {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 &&
      _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0,
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
    std::array<float, 2> convergence{}, projection{};
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
      if (traced_) game3d_present_trace::out.close();
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
      set_enabled_ = reinterpret_cast<set_enabled_t>(GetProcAddress(module, "SunshineGame3DTestSetEnabled"));
      require(module && query_automatic_ && query_scale_ && edit_float_ && recalibrate_ && set_foreground_,
        "Native Game 3D fixture requires the test add-on's passive UI queries, controls and foreground observer");
      const auto effects = fixture_.runtime_directory / "effects";
      sunshine_test_files::rename_when_free(effects / effect_file, effects / (std::string(effect_file) + ".disabled"));
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
      // The fixture's per-Present trace, written once when this observer ends.
      game3d_present_trace::out.open(fixture_.runtime_directory / "trace.txt", "native");
      traced_ = true;
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

    // The overlay's Enable Game 3D, as the panel edits it.
    void set_enabled(bool value) {
      require(set_enabled_ && set_enabled_(observed.runtime, value ? TRUE : FALSE), "Native Game 3D rejected an Enable Game 3D edit");
    }

    // What the last native render consumed: its depth identity, allocation
    // and readiness, and its constants. Per Present, without a dump; the
    // caller compares sequence to know a render followed its Present. It
    // also traces that Present (native_present_trace).
    sunshine_game3d::test::last_render last_render() const {
      sunshine_game3d::test::last_render result;
      require(last_render_ && last_render_(observed.runtime, &result), "Native Game 3D last-render observation failed");
      native_trace.present(observed.runtime, nullptr, &result);
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
      // A consumer of this protocol without capabilities: its declaration, then its request.
      InterlockedExchange(reinterpret_cast<volatile LONG *>(&export_state_->consumer_protocol), static_cast<LONG>(reshade_bridge::version));
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&export_state_->capability_nonce), 0x4e41544956455850LL);
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&export_state_->consumer_nonce), 0x4e41544956455850LL);
    }

    // The newest completed publication, which belongs to the latest present.
    std::vector<std::uint8_t> exported() {
      std::vector<std::uint8_t> pixels;
      read_exported([&](ID3D12Resource *texture) { pixels = fixture_.read(texture, D3D12_RESOURCE_STATE_COMMON); });
      return pixels;
    }

    // Claims the newest completed publication, which belongs to the latest
    // present, and lends its shared texture (COMMON state) to read, for
    // fixtures that copy only a few texels per Present.
    void read_exported(const std::function<void(ID3D12Resource *)> &read) {
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
        read(texture.p);
        release();
        export_sequence_ = sequence;
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
      frame.projection = {parameters.at("projection").at(0), parameters.at("projection").at(1)};
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
    using set_enabled_t = BOOL (*)(api::effect_runtime *, BOOL);

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
    set_enabled_t set_enabled_{};
    unsigned fx_renders_{};
    HANDLE mapping_{}, export_mapping_{};
    game3d_debug::shared_state_t *state_{};
    reshade_bridge::shared_state_t *export_state_{};
    std::uint64_t request_{}, export_generation_{}, export_sequence_{};
    std::ofstream evidence_;
    bool traced_{};
  };
}

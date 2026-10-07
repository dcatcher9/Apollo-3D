// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_renderer.h"
#include "game3d_shader_source.h"
#include <reshade.hpp>
#include "async_log.h"
#include "game3d_diagnostics.h"
#include "game3d_shader_cache.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_temporal.h"
#include <d3d11_1.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sunshine_game3d {
  namespace api = reshade::api;
  namespace {
    template<class T> struct com {
      T *p = nullptr;
      ~com() { if (p) p->Release(); }
      T **put() { return &p; }
      T *operator->() const { return p; }
    };
    api::format typed(api::format f) { return api::format_to_default_typed(f, 0); }
  }
  struct renderer::impl {
    api::device *device = nullptr;
    api::command_queue *queue = nullptr;
    uint32_t width = 0, height = 0, color = 0;
    api::format source_format = api::format::unknown;
    // Only offline replay supplies an override. Live rendering keeps a view of
    // the immutable embedded source and does not copy/compare it each frame.
    std::string source_override;
    std::string_view shader_source() const {
      return source_override.empty() ? renderer::shader_source() : std::string_view(source_override);
    }
    api::pipeline_layout layout{};
    enum pass { pq, candidate, vertical, ui_tiles, ui_reduce, horizontal, eyes, pack, ui_conflict, ui_apply,
      detection_tiles, detection_reduce, detection_mask, scene_cells, scene_compare, scene_evidence, pack_pq, vertical_live,
      pass_count };
    std::array<api::pipeline, pass_count> pipelines{};
    std::array<api::sampler, 3> samplers{};
    api::resource_view null_srv{}, null_uav{};
    struct texture { api::resource resource{}; api::resource_view srv{}, uav{}, rtv{}; };
    enum texture_id { source, empty_depth, linear, raw, vertical_majorant, vertical_field, field,
      ui_plane_tiles, ui_plane_resolved, left, right, packed, ui_conflict_statistics, detection_statistics, detection_decision, detected_mask, scene_cell_sums, ui_counter_words,
      detection_hold, retained_color,
      retained_color_last = retained_color + ui_detection_inputs::max_retained_presents - 1, packed_pq, texture_count };
    std::array<texture, texture_count> textures{};
    std::vector<std::pair<api::resource, api::resource_view>> backbuffers;
    // Render-target views of the current export generation's slot textures.
    std::vector<std::pair<api::resource, api::resource_view>> export_views;
    std::uint64_t export_generation = 0;
    // The final pack of this presentation is recorded exactly once, after the
    // caller knows whether an export slot can receive it directly.
    bool pack_owed = false;
    render_parameters pack_parameters;
    // Packed-eye shaders render both eyes when the side-by-side target is
    // recorded; the borrowed depth stays leased until the present ends.
    api::resource_view pack_depth{};
    api::fence completion{};
    uint64_t sequence = 0;
    // Timestamps per presentation: begin, render, source, detection, the start
    // of the recorded conditioning (in render, or owed with the pack), its
    // stages, eyes, pack. A small ring is read after its fence completes.
    // Recorded only while the Diagnostics switch is on: a Present's first
    // mark decides, so a toggle never splits one Present's marks.
    enum profile_mark { mark_begin, mark_render, mark_source, mark_detection, mark_owed, mark_linearize, mark_candidate,
      mark_vertical, mark_conditioning, mark_eyes, mark_pack, mark_count };
    static constexpr uint32_t profile_frames = 4;
    struct profile_frame { uint64_t fence{}; uint32_t written{}; };
    api::query_heap profile_heap{};
    // Where the device can copy query results, each timestamp is resolved into
    // this readback buffer on the command list that recorded it, and the
    // frame's completion fence alone decides readiness. ReShade 6.8 reads D3D12
    // results through ID3D12Device15::ResolveQueryData whenever the runtime
    // offers it (Agility SDK 1.619 in The Witcher 3) but creates its heaps
    // without D3D12_QUERY_HEAP_FLAG_CPU_RESOLVE, so that read always fails.
    // D3D11 reads ReShade's results.
    api::resource profile_readback{};
    // Each slot's last resolved render tick: an unchanged value is a copy that
    // has not executed yet, never a new frame.
    std::array<uint64_t, profile_frames> profile_resolved_render{};
    bool profile_attempted{}, profile_ready{}, profile_open{};
    gpu_timing::profile_state profile_state = gpu_timing::profile_state::not_started;
    double profile_ticks_per_ms{};
    uint32_t profile_slot{};
    std::array<profile_frame, profile_frames> profile_ring{};
    gpu_timing profile_window{};
    std::array<double, gpu_timing::stage_count> profile_sum_ms{};
    // Conditioning (C2): everything after detection that only the pack, a
    // dump or the adaptive probe read (PQ linearization, depth candidate,
    // vertical, nearest-UI tiles and reduce, horizontal and UI pinning), with
    // the inputs and b1 words render() resolved for it. render() records it
    // for a probe frame, an older two-pass replay shader or an immediate pack;
    // otherwise it is owed and record_pack records it first. A Present whose
    // pack is never recorded records none; the next render or begin_present
    // drops it.
    struct conditioning_input {
      render_parameters p;
      api::resource_view depth{}, ui_alpha{};
      ui_plane_parameters plane;
      ui_mask_channel channel = ui_mask_channel::alpha;
      bool apply_ui{}, probe{}, armed{};
      uint64_t depth_identity{};
    };
    conditioning_input owed;
    bool conditioning_owed = false;
    // The shader's mono pack reads no conditioning (SUNSHINE_MONO_SKIPS_CONDITIONING):
    // shows_mono() may then skip it. Older replay shaders always record it.
    bool mono_skip_supported = false;
    // The candidate and vertical field of the last recorded conditioning, kept
    // while its depth view, nonzero depth identity and parameters repeat.
    bool memo_valid = false;
    api::resource_view memo_depth{};
    uint64_t memo_identity{};
    render_parameters memo_parameters;
    // The final field of that conditioning (the horizontal and UI pin
    // passes), kept with the vertical memo while the b1 words, the mask view
    // and the detected mask's epoch repeat: a held generated Present, whose
    // mask no detection rewrote, records neither pass. mask_epoch advances
    // whenever detect_ui dispatches the mask pass.
    bool field_memo_valid = false;
    std::array<uint32_t, 4> field_memo_words{};
    uint64_t field_memo_view{}, field_memo_epoch{}, mask_epoch{};
    renderer::conditioning_counters conditioning_counts;
    // pending: this presentation recorded work awaiting finish_present.
    // unsignaled: an earlier presentation's work awaits the next signal.
    bool pending = false, unsignaled = false, failed = false;
    com<ID3D11DeviceContext1> context11;
    com<ID3DDeviceContextState> isolated11;
    ID3DDeviceContextState *previous11 = nullptr;
    bool frame_state = false;
    render_parameters consumed;
    ui_plane_parameters consumed_plane;
    bool source_alpha_ui = false;
    api::resource consumed_ui_source{};
    // Candidate slots: 0 UIAlpha, 1 UI color tag, 2 Backbuffer, 3 HUD-less, 4
    // the offscreen UI layer (UI framework E1: the tag and the layer never
    // share a slot).
    std::array<texture, 5> ui_candidates{};
    std::array<api::format, 5> ui_candidate_formats{};
    ui_mask_channel consumed_channel = ui_mask_channel::alpha;
    bool mask_channel_supported = false;
    // Lines per limiter group, from SUNSHINE_LIMITER_LINE_GROUPS. Zero for older
    // embedded replay shaders, which scan one line per 32-thread group and pin
    // UI inside the horizontal pass.
    uint32_t limiter_lines = 0;
    // Older embedded replay shaders render two FP16 eye textures and pack them
    // in a second pass.
    bool packed_eyes = false;
    // The shader decodes an HDR10 (PQ) source per tap in the pack
    // (SUNSHINE_PQ_PER_TAP): no linearization pass and no FP16 linear
    // texture. Older replay shaders keep both.
    bool pq_per_tap = false;
    // The pack writes the 10-bit PQ export (renderer::set_pq_output) into
    // textures[packed_pq] or an R10G10B10A2 export slot instead of FP16 scRGB.
    bool pq_output = false;
    // The shader packs a native scRGB source as 10-bit PQ too
    // (SUNSHINE_SCRGB_PQ_PACK); older replay shaders pack only HDR10 as PQ.
    bool scrgb_pq_pack = false;
    // The shader's live vertical pass skips the majorant write-back only a
    // dump reads (SUNSHINE_VERTICAL_LIVE_ENTRY); older replay shaders always
    // write it.
    bool vertical_live_supported = false;
    // Candidates bound directly this Present (renderer::bind_ui_candidate):
    // the caller's view and its texture, which device::get_resource_from_view
    // cannot resolve for a leased snapshot's raw descriptor.
    std::array<std::pair<api::resource_view, api::resource>, 5> direct_views{};
    // D3D12: one original-device descriptor per slot for leased snapshots
    // (renderer::bind_ui_snapshot), created on first use.
    com<ID3D12DescriptorHeap> snapshot_views;
    UINT snapshot_stride{};
    api::resource resource_of(api::resource_view view) const {
      for (const auto &[bound, resource] : direct_views)
        if (bound.handle && bound.handle == view.handle) return resource;
      return device->get_resource_from_view(view);
    }
    // Rows per UI pinning group, from SUNSHINE_UI_PIN_LINE_GROUPS. Zero for
    // older embedded replay shaders, which pin one row per 32-thread group.
    uint32_t pin_lines = 0;
    bool nearest_ui_supported = false, nearest_ui_rendered = false;
    bool nearest_ui_attempted = false, nearest_ui_ready = false;
    bool front_limit_ui_supported = false, shallow_front_ui_supported = false;
    bool display_fraction_ui_supported = false;
    alpha_auto_decision consumed_auto;
    bool detection_attempted{}, detection_ready{}, detection_active{}, detection_pending{}, detection_awaiting_signal{};
    // Decision, then candidate bits with HUD-less counts, alpha coverage,
    // invalid alpha, lit HUD-less pixels with the accepted candidates, the
    // offscreen UI layer's counts with the valid candidates (texel 7), and the
    // one-way judgment counts with the refused candidate and the frame reason
    // (texels 8 and 9, and the declared alphas' texel 16), the H1 texel 10
    // (the opaque Backbuffer and current counts, the claims and the h1 word)
    // and texel 11 (the layer's pre-UI pixel counts, H1 d). Detection runs
    // only with the current revision's markers: its decision texels and both
    // scene-evidence images (docs/reshade-sbs.md, UI detection flags and
    // decision texels); zero when the shader has no marker.
    uint32_t detection_decision_texels = 0;
    uint32_t detection_statistics_images = 0;
    // Parts per statistics tile (ui_detection::tile_parts_marker; zero for a
    // shader without the marker, which counts each tile in one group).
    uint32_t detection_tile_parts = 0;
    // Hidden-scene evidence (docs/reshade-sbs.md, hidden-scene evidence): the
    // shader writes decision texels 5 and 6 with its evidence passes. They
    // run on sample frames only, and only when the scene guard measures
    // (scene_guard::state::measure: an acting-capable claim in the latest
    // sample, a held verdict, or a proven layer that is the offer's pre-UI
    // image), so everything they measure is actionable.
    bool scene_evidence_supported() const {
      return detection_statistics_images == ui_detection::max_scene_evidence_images &&
        detection_decision_texels == ui_detection::decision_texels;
    }
    uint64_t scene_evidence_runs{};
    // This render's scene guard bits (per_frame_scene_hidden,
    // per_frame_pre_ui_visible, the refuted candidates and
    // per_frame_pre_ui_proven), pushed with its detection, and whether its
    // offered layer's signature is proven the pre-UI scene image (H1 d, the
    // session ledger's pre-UI proof).
    uint32_t scene_bits{};
    bool scene_layer_proven{};
    // The CPU-side temporal state (game3d_ui_temporal.h): the adopted inputs
    // (the offered and accepted candidates), which Presents detect and which
    // show a real frame's decision (T1) and the latest status sample. The GPU
    // owns the T1 grace of a real frame without a decision of its own, in the
    // hold store (textures[detection_hold], u5 of the reduce only).
    ui_temporal::detection_state temporal;
    // The hidden-scene guard (M5, game3d_scene_guard.h), owned by the depth
    // path: the held D verdicts and refuted signatures behind H1. Only an
    // identity change clears it; the layer's pre-UI proof lives in the
    // session's acceptance ledger, which no identity change clears.
    scene_guard::state guard;
    // The pending sample ran the evidence passes (measure: actionable).
    bool detection_pending_actionable{};
    // The constants of the last detection run, and the run this render's mask
    // came from (fresh, or held on a generated Present).
    ui_detection_snapshot detection_run, consumed_detection;
    // Presented colors kept for late HUD-less captures, created on first need.
    // Each slot records the Present number it holds; zero is empty.
    bool retention_wanted{}, retention_attempted{}, retention_ready{};
    // Retention stops this many Presents after the last late capture needed it.
    static constexpr uint64_t retention_linger_presents = 120;
    uint64_t retention_requested_present{};
    std::array<uint64_t, ui_detection_inputs::max_retained_presents> retained_present{};
    uint64_t present_number{};
    // The status key (ui_temporal::detection_state::status_key) when the
    // pending sample was submitted; the sample describes frames of that winner.
    uint32_t detection_pending_status_key{};
    // The acceptance signatures the pending sample was submitted with; its
    // read earns or revokes acceptance for exactly these (A1).
    candidate_signatures detection_pending_signatures;
    float difference_threshold = 4.f / 1023.f;
    // b2 word 4: the offscreen UI layer's pair threshold with the presented
    // color (H1 d, the pre-UI pixel counts); zero without a layer or when the
    // two are not comparable.
    float pre_ui_threshold = 0.f;
    uint64_t detection_fence{}, detection_last_submit{}, detection_submitted{}, detection_mapped{};
    alpha_auto_source detection_pending_source;
    com<ID3D11Texture2D> detection_readback11;
    com<ID3D12Resource> detection_readback12;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT detection_footprint{};
    uint64_t detection_readback_bytes{};
    // Exact UI counters (game3d_ui_counters.h). The reduce adds every detection
    // frame to the ui_counter_words texture; a sample frame copies it beside the
    // decision texels and snapshots this renderer's CPU counts, and the sample's
    // read commits both deltas to the session. Zero words: a shader without
    // counters, or resources that could not be made (counting is diagnostic).
    uint32_t counter_words{};
    bool counters_cleared{}, counters_pending{};
    com<ID3D11Texture2D> counters_readback11;
    com<ID3D12Resource> counters_readback12;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT counters_footprint{};
    uint64_t counters_readback_bytes{};
    ui_counters cpu_counts, pending_counts, committed_counts;
    std::array<uint32_t, ui_counter_word::count> committed_words{};
    // Adaptive placement has its own observation slot and lifecycle. It never
    // changes UI qualification, depth admission or capture ownership.
    ui_adaptive::policy adaptive_policy;
    ui_adaptive::decision consumed_adaptive;
    ui_adaptive::source adaptive_scope, adaptive_pending_source;
    bool adaptive_scope_known = false, adaptive_probe_attempted = false, adaptive_probe_ready = false;
    bool adaptive_readback_pending = false, adaptive_awaiting_signal = false;
    uint64_t adaptive_scope_generation = 0, adaptive_pending_generation = 0;
    uint64_t adaptive_fence = 0, adaptive_last_submit = 0, adaptive_last_sequence = 0,
      adaptive_last_mask_sequence = 0, adaptive_last_tick = 0;
    com<ID3D11Texture2D> adaptive_readback11;
    com<ID3D12Resource> adaptive_readback12;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT adaptive_footprint{};
    uint64_t adaptive_readback_bytes = 0;
    uint64_t adaptive_submitted = 0;

    // A cached renderer becomes current again (a colour toggle back): it keeps
    // its GPU objects and cumulative counters, and starts every per-scene
    // decision state as a new renderer would, so that a toggle cannot carry a
    // minute-old decision, hold or sample into the new session.
    void resume() {
      pending = unsignaled = false;
      conditioning_owed = pack_owed = false;
      profile_open = false;
      direct_views = {};
      memo_valid = field_memo_valid = false;
      temporal = {};
      guard = {};
      detection_pending = detection_awaiting_signal = detection_pending_actionable = false;
      detection_fence = detection_last_submit = 0;
      counters_pending = false;
      scene_bits = 0;
      scene_layer_proven = false;
      detection_run = consumed_detection = {};
      consumed_auto = {};
      retention_wanted = false;
      retained_present.fill(0);
      adaptive_policy.reset();
      consumed_adaptive = {};
      adaptive_scope = adaptive_pending_source = {};
      adaptive_scope_known = adaptive_readback_pending = adaptive_awaiting_signal = false;
      ++adaptive_scope_generation;
      adaptive_fence = adaptive_last_submit = adaptive_last_sequence = adaptive_last_mask_sequence = adaptive_last_tick = 0;
    }
    bool idle() const {
      return !pending && !unsignaled && !failed &&
        (!completion.handle || device->get_completed_fence_value(completion) >= sequence);
    }
    ~impl() {
      if (profile_heap.handle) device->destroy_query_heap(profile_heap);
      if (profile_readback.handle) device->destroy_resource(profile_readback);
      for (auto &[resource, view] : backbuffers) device->destroy_resource_view(view);
      for (auto &[resource, view] : export_views) device->destroy_resource_view(view);
      for (auto &t : textures) {
        if (t.srv.handle) device->destroy_resource_view(t.srv);
        if (t.uav.handle) device->destroy_resource_view(t.uav);
        if (t.rtv.handle) device->destroy_resource_view(t.rtv);
        if (t.resource.handle) device->destroy_resource(t.resource);
      }
      for (auto &t : ui_candidates) {
        if (t.srv.handle) device->destroy_resource_view(t.srv);
        if (t.resource.handle) device->destroy_resource(t.resource);
      }
      for (auto p : pipelines) if (p.handle) device->destroy_pipeline(p);
      for (auto s : samplers) if (s.handle) device->destroy_sampler(s);
      if (null_srv.handle) device->destroy_resource_view(null_srv);
      if (null_uav.handle) device->destroy_resource_view(null_uav);
      if (layout.handle) device->destroy_pipeline_layout(layout);
      if (completion.handle) device->destroy_fence(completion);
    }
    bool texture_create(texture_id id, uint32_t w, uint32_t h, api::format format, api::resource_usage extra) {
      auto &t = textures[id];
      const auto usage = api::resource_usage::shader_resource | extra;
      const api::resource_desc desc(w, h, 1, 1, format, 1, api::memory_heap::default_, usage);
      if (!device->create_resource(desc, nullptr, api::resource_usage::shader_resource, &t.resource) ||
          !device->create_resource_view(t.resource, api::resource_usage::shader_resource, api::resource_view_desc(format), &t.srv)) return false;
      if ((extra & api::resource_usage::unordered_access) != api::resource_usage::undefined &&
          !device->create_resource_view(t.resource, api::resource_usage::unordered_access, api::resource_view_desc(format), &t.uav)) return false;
      return (extra & api::resource_usage::render_target) == api::resource_usage::undefined ||
        device->create_resource_view(t.resource, api::resource_usage::render_target, api::resource_view_desc(format), &t.rtv);
    }
    // Prepared production bytecode comes from the shared store; anything else
    // (explicit replay/test sources, an entry missing from the store) compiles here.
    bool precompiled = false;
    bool compile(const char *entry, const char *target, shader_cache::blob &code) {
      const shader_cache::configuration config{shader_source(), width, height, color};
      if (precompiled && !(code = shader_cache::find(config, {entry, target})).empty()) return true;
      return shader_cache::compile(config, {entry, target}, code);
    }
    bool pipeline_create(pass id, const char *entry, bool compute, api::shader_desc vs) {
      shader_cache::blob code;
      if (!compile(entry, compute ? "cs_5_0" : "ps_5_0", code)) return false;
      api::shader_desc shader{code.data(), code.size()};
      if (compute) {
        const api::pipeline_subobject part{api::pipeline_subobject_type::compute_shader, 1, &shader};
        return device->create_pipeline(layout, 1, &part, &pipelines[id]);
      }
      api::rasterizer_desc raster; raster.cull_mode = api::cull_mode::none; raster.scissor_enable = true;
      api::depth_stencil_desc depth; depth.depth_enable = false; depth.depth_write_mask = false; depth.stencil_enable = false;
      api::blend_desc blend;
      auto topology = api::primitive_topology::triangle_list;
      api::format formats[]{(id == pack && color == 1) || id == pack_pq ? api::format::r10g10b10a2_unorm :
        api::format::r16g16b16a16_float, api::format::r16g16b16a16_float};
      uint32_t count = 1;
      const api::pipeline_subobject parts[]{
        {api::pipeline_subobject_type::vertex_shader, 1, &vs},
        {api::pipeline_subobject_type::pixel_shader, 1, &shader},
        {api::pipeline_subobject_type::rasterizer_state, 1, &raster},
        {api::pipeline_subobject_type::depth_stencil_state, 1, &depth},
        {api::pipeline_subobject_type::blend_state, 1, &blend},
        {api::pipeline_subobject_type::primitive_topology, 1, &topology},
        {api::pipeline_subobject_type::render_target_formats, id == eyes ? 2u : 1u, formats},
        {api::pipeline_subobject_type::sample_count, 1, &count},
        {api::pipeline_subobject_type::viewport_count, 1, &count}};
      return device->create_pipeline(layout, uint32_t(std::size(parts)), parts, &pipelines[id]);
    }
    bool initialize(api::effect_runtime *runtime, const api::resource_desc &desc, uint32_t input_color) {
      device = runtime->get_device(); queue = runtime->get_command_queue();
      width = desc.texture.width; height = desc.texture.height; source_format = typed(desc.texture.format); color = input_color;
      nearest_ui_supported = shader_source().find("#define SUNSHINE_UI_NEAREST_PLANE 1") != std::string_view::npos;
      front_limit_ui_supported = shader_source().find("#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1") != std::string_view::npos;
      shallow_front_ui_supported = shader_source().find("#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1") != std::string_view::npos;
      display_fraction_ui_supported = shader_source().find("#define SUNSHINE_UI_DISPLAY_FRACTION_PLANE 1") != std::string_view::npos;
      mask_channel_supported = shader_source().find("#define SUNSHINE_UI_MASK_CHANNEL 1") != std::string_view::npos;
      limiter_lines = shader_marker(shader_source(), "SUNSHINE_LIMITER_LINE_GROUPS");
      packed_eyes = shader_source().find("#define SUNSHINE_PACKED_EYES 1") != std::string_view::npos;
      mono_skip_supported = packed_eyes &&
        shader_source().find("#define SUNSHINE_MONO_SKIPS_CONDITIONING 1") != std::string_view::npos;
      pq_per_tap = packed_eyes && shader_source().find("#define SUNSHINE_PQ_PER_TAP 1") != std::string_view::npos;
      scrgb_pq_pack = packed_eyes && shader_source().find("#define SUNSHINE_SCRGB_PQ_PACK 1") != std::string_view::npos;
      vertical_live_supported = shader_source().find("#define SUNSHINE_VERTICAL_LIVE_ENTRY 1") != std::string_view::npos;
      pin_lines = shader_marker(shader_source(), "SUNSHINE_UI_PIN_LINE_GROUPS");
      const auto texels = shader_marker(shader_source(), ui_detection::decision_texels_marker);
      const auto images = shader_marker(shader_source(), ui_detection::scene_evidence_images_marker);
      detection_decision_texels = texels;
      // An out-of-range marker leaves automatic detection unavailable (prepare_detection).
      detection_statistics_images = images;
      detection_tile_parts = shader_marker(shader_source(), ui_detection::tile_parts_marker);
      counter_words = shader_marker(shader_source(), "SUNSHINE_UI_COUNTER_WORDS") == ui_counter_word::count ?
        uint32_t(ui_counter_word::count) : 0u;
      if (!device->create_fence(0, api::fence_flags::none, &completion)) return false;
      if (device->get_api() == api::device_api::d3d12 &&
          (!device->create_resource_view({}, api::resource_usage::shader_resource, api::resource_view_desc(api::format::r32_float), &null_srv) ||
           !device->create_resource_view({}, api::resource_usage::unordered_access, api::resource_view_desc(api::format::r32_float), &null_uav))) return false;
      if (device->get_api() == api::device_api::d3d11) {
        com<ID3D11Device1> native;
        if (FAILED(reinterpret_cast<ID3D11Device *>(device->get_native())->QueryInterface(IID_PPV_ARGS(native.put())))) return false;
        native->GetImmediateContext1(context11.put());
        const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        if (FAILED(native->CreateDeviceContextState(0, levels, 2, D3D11_SDK_VERSION, __uuidof(ID3D11Device), nullptr, isolated11.put()))) return false;
      }
      const api::pipeline_layout_param params[]{
        api::constant_range{0, 0, 0, sizeof(render_parameters) / 4, api::shader_stage::all},
        api::descriptor_range{0, 0, 0, 3, api::shader_stage::all, 1, api::descriptor_type::sampler},
        api::descriptor_range{0, 0, 0, 15, api::shader_stage::all, 1, api::descriptor_type::shader_resource_view},
        api::descriptor_range{0, 0, 0, 8, api::shader_stage::compute, 1, api::descriptor_type::unordered_access_view},
        api::constant_range{0, 1, 0, 4, api::shader_stage::compute},
        api::constant_range{0, 2, 0, ui_detection::b2_words, api::shader_stage::compute}};
      if (!device->create_pipeline_layout(uint32_t(std::size(params)), params, &layout)) return false;
      for (unsigned i = 0; i < samplers.size(); ++i) {
        api::sampler_desc sampler;
        sampler.filter = i == 1 ? api::filter_mode::min_mag_linear_mip_point : api::filter_mode::min_mag_mip_point;
        if (i == 2) { sampler.address_u = sampler.address_v = sampler.address_w = api::texture_address_mode::border; std::memset(sampler.border_color, 0, sizeof(sampler.border_color)); }
        if (!device->create_sampler(sampler, &samplers[i])) return false;
      }
      if (!texture_create(source, width, height, source_format, api::resource_usage::copy_dest) ||
          !texture_create(empty_depth, 1, 1, api::format::r32_float, api::resource_usage::undefined)) return false;
      if (color == 3 && !pq_per_tap &&
          !texture_create(linear, width, height, api::format::r16g16b16a16_float, api::resource_usage::render_target)) return false;
      if (width <= 3840 && height <= 3840)
        for (auto id : {raw, vertical_majorant, vertical_field, field})
          if (!texture_create(id, width, height, api::format::r32_float, api::resource_usage::unordered_access)) return false;
      if (!packed_eyes)
        for (auto id : {left, right})
          if (!texture_create(id, width, height, api::format::r16g16b16a16_float, api::resource_usage::render_target)) return false;
      if (!texture_create(packed, width * 2, height, packed_format(),
          api::resource_usage::render_target | api::resource_usage::copy_source)) return false;
      shader_cache::blob vs_code;
      if (!compile("PostProcessVS", "vs_5_0", vs_code)) return false;
      const api::shader_desc vs{vs_code.data(), vs_code.size()};
      if (color == 3 && !pq_per_tap && !pipeline_create(pq, "SunshinePreparePQPS", false, vs)) return false;
      if (((color == 3 && pq_per_tap) || (color == 2 && scrgb_pq_pack)) &&
          !pipeline_create(pack_pq, "SunshineRenderPackedPQPS", false, vs)) return false;
      if (width <= 3840 && height <= 3840)
        if (!pipeline_create(candidate, "SunshineHostCandidateCS", true, vs) || !pipeline_create(vertical, "SunshineHostVerticalCS", true, vs) ||
            !pipeline_create(horizontal, "SunshineHostHorizontalCS", true, vs) ||
            (limiter_lines && !pipeline_create(ui_apply, "SunshineApplyUICS", true, vs)) ||
            (vertical_live_supported && !pipeline_create(vertical_live, "SunshineHostVerticalLiveCS", true, vs))) return false;
      if (packed_eyes) return pipeline_create(pack, "SunshineRenderPackedPS", false, vs);
      return pipeline_create(eyes, "SunshineRenderEyesPS", false, vs) && pipeline_create(pack, "SunshinePackEyesPS", false, vs);
    }
    bool prepare_nearest_ui() {
      if (nearest_ui_attempted) return nearest_ui_ready;
      nearest_ui_attempted = true;
      // Live mode 5 does not use this historical reduction. Create it only for
      // a requested mode-2 render, before that render records any commands.
      // Partially created objects remain owned by impl; a failure is not retried.
      if (!texture_create(ui_plane_tiles, (width + 15) / 16, (height + 15) / 16, api::format::r32_float,
            api::resource_usage::unordered_access) ||
          !texture_create(ui_plane_resolved, 1, 1, api::format::r32_float, api::resource_usage::unordered_access) ||
          !pipeline_create(ui_tiles, "SunshineUINearestTilesCS", true, {}) ||
          !pipeline_create(ui_reduce, "SunshineUINearestReduceCS", true, {})) return false;
      nearest_ui_ready = true;
      return true;
    }
    void bindings(api::command_list *cmd, api::shader_stage stage, const render_parameters &p,
        std::array<api::resource_view, 15> srvs, std::array<api::resource_view, 8> uavs = {}) {
      // D3D12 needs actual null descriptors; a zero CPU descriptor handle is
      // not a valid CopyDescriptors source. D3D11 uses zero COM views normally.
      for (auto &view : srvs) if (!view.handle) view = null_srv;
      for (auto &view : uavs) if (!view.handle) view = null_uav;
      cmd->push_constants(stage, layout, 0, 0, sizeof(p) / 4, &p);
      if (stage == api::shader_stage::compute) {
        const auto ui = ui_parameter_words(source_alpha_ui, consumed_plane, consumed_channel);
        cmd->push_constants(stage, layout, 4, 0, uint32_t(ui.size()), ui.data());
      }
      cmd->push_descriptors(stage, layout, 1, {{}, 0, 0, uint32_t(samplers.size()), api::descriptor_type::sampler, samplers.data()});
      cmd->push_descriptors(stage, layout, 2, {{}, 0, 0, uint32_t(srvs.size()), api::descriptor_type::shader_resource_view, srvs.data()});
      if (stage == api::shader_stage::compute)
        cmd->push_descriptors(stage, layout, 3, {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
    }
    api::format packed_format() const {
      return color == 1 || pq_output ? api::format::r10g10b10a2_unorm : api::format::r16g16b16a16_float;
    }
    // The internal side-by-side image of the current export transfer.
    texture_id packed_texture() const { return pq_output ? packed_pq : packed; }
    bool prepare_profile() {
      if (profile_attempted) return profile_ready;
      profile_attempted = true;
      const auto frequency = queue->get_timestamp_frequency();
      if (!frequency) {
        profile_state = gpu_timing::profile_state::no_timestamp_frequency;
        return false;
      }
      if (!device->create_query_heap(api::query_type::timestamp, profile_frames * mark_count, &profile_heap)) {
        profile_state = gpu_timing::profile_state::no_query_heap;
        return false;
      }
      if (device->check_capability(api::device_caps::copy_query_heap_results)) {
        const api::resource_desc readback(profile_frames * mark_count * sizeof(uint64_t), api::memory_heap::gpu_to_cpu,
          api::resource_usage::copy_dest);
        if (!device->create_resource(readback, nullptr, api::resource_usage::copy_dest, &profile_readback)) profile_readback = {};
      }
      profile_ticks_per_ms = double(frequency) / 1000.0;
      profile_state = gpu_timing::profile_state::ready;
      return profile_ready = true;
    }
    // Reads completed frames into the window; never waits on the GPU.
    void collect_profile() {
      if (!profile_ready) return;
      const auto completed = device->get_completed_fence_value(completion);
      void *resolved = nullptr;
      for (uint32_t i = 0; i != profile_frames; ++i) {
        auto &frame = profile_ring[i];
        if (!frame.written || completed == UINT64_MAX || completed < frame.fence) continue;
        const auto written = frame.written;
        const auto has = [written](unsigned mark) { return (written >> mark & 1u) != 0; };
        std::array<uint64_t, mark_count> ticks{};
        bool read = true;
        if (profile_readback.handle) {
          if (!resolved && !device->map_buffer_region(profile_readback, 0, UINT64_MAX, api::map_access::read_only, &resolved))
            resolved = nullptr;
          read = resolved != nullptr;
          if (read) {
            std::memcpy(ticks.data(), static_cast<const uint64_t *>(resolved) + i * mark_count, sizeof(ticks));
            read = !has(mark_render) || ticks[mark_render] > profile_resolved_render[i];
          }
        } else {
          // ReShade refuses a range containing a query this frame never wrote,
          // so read each written timestamp on its own.
          for (unsigned m = 0; m != mark_count; ++m)
            if (has(m))
              read = read && device->get_query_heap_results(profile_heap, api::query_type::timestamp,
                i * mark_count + m, 1, &ticks[m], sizeof(uint64_t));
        }
        if (!read) continue; // Not resolved yet; retry on a later collection.
        if (has(mark_render)) profile_resolved_render[i] = ticks[mark_render];
        frame.written = 0;
        // Packed-eye frames mark their conditioning and eyes only when the
        // side-by-side target is recorded; without a consumer the frame ends
        // after detection, and a mono pack marks no conditioning stage.
        if (!has(mark_render)) { ++profile_window.incomplete; continue; }
        const auto span = [&](unsigned from, unsigned to) {
          return has(from) && has(to) && ticks[to] >= ticks[from] ? double(ticks[to] - ticks[from]) / profile_ticks_per_ms : 0.0;
        };
        const unsigned first = has(mark_begin) ? mark_begin : mark_render,
          last = has(mark_pack) ? mark_pack : has(mark_eyes) ? mark_eyes : has(mark_conditioning) ? mark_conditioning :
            mark_detection;
        const std::array<double, gpu_timing::stage_count> ms{span(mark_begin, mark_render), span(mark_render, mark_source),
          span(mark_source, mark_detection), span(mark_owed, mark_linearize), span(mark_linearize, mark_candidate),
          span(mark_candidate, mark_vertical), span(mark_vertical, mark_conditioning), span(mark_conditioning, mark_eyes),
          span(mark_eyes, mark_pack), span(first, last)};
        ++profile_window.frames;
        for (unsigned s = 0; s != gpu_timing::stage_count; ++s) {
          profile_sum_ms[s] += ms[s];
          profile_window.max_ms[s] = std::max(profile_window.max_ms[s], ms[s]);
        }
      }
      if (resolved) device->unmap_buffer_region(profile_readback);
    }
    void mark(api::command_list *cmd, profile_mark which) {
      if (!profile_open) {
        // Diagnostics only (G1): with the switch off no timestamp is written.
        if (!diagnostics::enabled() || !prepare_profile()) return;
        // The first mark of a presentation claims the next slot, dropping an
        // unread frame; this presentation's completion signal retires it.
        profile_slot = (profile_slot + 1) % profile_frames;
        if (const auto &unread = profile_ring[profile_slot]; unread.written) {
          const auto completed = device->get_completed_fence_value(completion);
          ++(completed == UINT64_MAX || completed < unread.fence ?
            profile_window.dropped_fence_pending : profile_window.dropped_unresolved);
        }
        profile_ring[profile_slot] = {sequence + 1, 0};
        profile_open = true;
      }
      profile_ring[profile_slot].written |= 1u << which;
      const uint32_t index = profile_slot * mark_count + which;
      cmd->end_query(profile_heap, api::query_type::timestamp, index);
      if (profile_readback.handle)
        cmd->copy_query_heap_results(profile_heap, api::query_type::timestamp, index, 1, profile_readback,
          uint64_t(index) * sizeof(uint64_t), sizeof(uint64_t));
    }
    bool prepare_retention() {
      if (retention_attempted) return retention_ready;
      retention_attempted = true;
      for (unsigned i = 0; i != ui_detection_inputs::max_retained_presents; ++i)
        if (!texture_create(texture_id(retained_color + i), width, height, source_format, api::resource_usage::copy_dest)) return false;
      return retention_ready = true;
    }
    api::resource_view retained_view(uint32_t presents_ago) const {
      if (!retention_ready || !presents_ago || presents_ago > present_number) return {};
      const auto wanted = present_number - presents_ago;
      for (unsigned i = 0; i != retained_present.size(); ++i)
        if (retained_present[i] && retained_present[i] == wanted) return textures[retained_color + i].srv;
      return {};
    }
    // Keep this Present's source color, replacing the oldest retained one.
    void retain_color(api::command_list *cmd) {
      if (!retention_wanted || present_number - retention_requested_present > retention_linger_presents ||
          !prepare_retention() || !present_number) return;
      const auto slot = unsigned(present_number % retained_present.size());
      const auto target = textures[retained_color + slot].resource, from = textures[source].resource;
      cmd->barrier(from, api::resource_usage::shader_resource, api::resource_usage::copy_source);
      cmd->barrier(target, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      cmd->copy_resource(from, target);
      cmd->barrier(target, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
      cmd->barrier(from, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      retained_present[slot] = present_number;
    }
    // A pack that shows the source mono (SunshineAutomaticMono) or the depth
    // preview (Depth_Map_View 2, which reads depth only) reads no conditioning.
    // A conservative CPU subset of that predicate: when in doubt (a camera
    // only SunshineCameraActive rejects) the passes run.
    bool shows_mono(const render_parameters &p) const {
      return width > 3840 || height > 3840 || p.depth_view == 2 || !p.depth_ready || !p.camera_ready ||
        !(std::isfinite(p.strength) && p.strength > 0.f) || !(std::isfinite(p.strength_blend) && p.strength_blend > 0.f);
    }
    // owed: recorded by the pack it was owed to. Only such a pack may skip it
    // as mono; render() records it in full for a probe, an older two-pass
    // shader or an immediate pack (replay and fixtures that read every
    // diagnostic texture after the render).
    void record_conditioning(api::command_list *cmd, const conditioning_input &c, bool owed_pack = false) {
      conditioning_owed = false;
      // The passes push the b1 words of the render that resolved them.
      struct restore_words {
        impl &d;
        bool ui;
        ui_plane_parameters plane;
        ui_mask_channel channel;
        ~restore_words() { d.source_alpha_ui = ui; d.consumed_plane = plane; d.consumed_channel = channel; }
      } restore{*this, source_alpha_ui, consumed_plane, consumed_channel};
      source_alpha_ui = c.apply_ui;
      consumed_plane = c.plane;
      consumed_channel = c.channel;
      mark(cmd, mark_owed);
      // An adaptive probe reads the unpinned field and a dump every texture.
      if (owed_pack && mono_skip_supported && !c.probe && !c.armed && shows_mono(c.p)) {
        ++conditioning_counts.mono;
        mark(cmd, mark_conditioning);
        return;
      }
      ++conditioning_counts.recorded;
      nearest_ui_rendered = false;
      const auto &t = textures;
      const auto &p = c.p;
      if (color == 3 && !pq_per_tap) draw(cmd, pq, width, {linear}, p, {t[source].srv});
      mark(cmd, mark_linearize);
      if (width <= 3840 && height <= 3840) {
        // The candidate reads only the depth and b0, the vertical scan only
        // the candidate; an exact repeat keeps both. A dump records both, so
        // its vertical majorant is this render's.
        const bool reuse = !c.armed && memo_valid && c.depth_identity && c.depth_identity == memo_identity &&
          c.depth.handle == memo_depth.handle && !std::memcmp(&c.p, &memo_parameters, sizeof(render_parameters));
        // Only a dump reads the vertical majorant: an owed pack that no dump
        // reads runs the live pass, which leaves the majorant's intermediate
        // and writes the same field. render() records the full pass (replay,
        // fixtures and a probe read every diagnostic texture).
        const bool live_vertical = owed_pack && !c.armed && pipelines[vertical_live].handle;
        const uint32_t group_lines = std::max(limiter_lines, 1u);
        if (reuse) ++conditioning_counts.memo;
        else {
          memo_valid = field_memo_valid = false;
          dispatch(cmd, candidate, (width + 7) / 8, (height + 7) / 8, p, {api::resource_view{}, c.depth}, {raw});
        }
        mark(cmd, mark_candidate);
        if (!reuse) {
          dispatch(cmd, live_vertical ? vertical_live : vertical, (width + group_lines - 1) / group_lines, 1, p,
            {api::resource_view{}, {}, {}, t[raw].srv}, {vertical_majorant, vertical_field});
          memo_valid = c.depth_identity != 0;
          memo_depth = c.depth;
          memo_identity = c.depth_identity;
          memo_parameters = c.p;
        }
        mark(cmd, mark_vertical);
        const auto ui_words = ui_parameter_words(c.apply_ui, c.plane, c.channel);
        if (c.apply_ui && c.plane.mode == ui_plane_mode::depth_midpoint_nearest_ui) {
          dispatch(cmd, ui_tiles, (width + 15) / 16, (height + 15) / 16, p, {c.ui_alpha, c.depth}, {ui_plane_tiles});
          dispatch(cmd, ui_reduce, 1, 1, p,
            {api::resource_view{}, {}, {}, {}, {}, {}, {}, {}, t[ui_plane_tiles].srv}, {ui_plane_resolved});
          nearest_ui_rendered = true;
        }
        // UI passes read the explicit selected mask channel. Eye RGB stays current.
        // UI pinning follows the complete scene field as its own pass, and a
        // probe observes the unpinned field in between. Older embedded replay
        // shaders pin inside the horizontal pass unless a probe needs it apart.
        const std::array<api::resource_view, 15> field_inputs{c.ui_alpha, {}, {}, {}, t[vertical_field].srv,
          {}, {}, {}, {}, nearest_ui_rendered ? t[ui_plane_resolved].srv : api::resource_view{}};
        // An exact repeat of the vertical field, b1 and an unchanged applied
        // mask keeps the final field too: both passes read nothing else, and
        // nothing else writes it (a probe reads it in between, and a dump or
        // the nearest-UI plane needs this render's passes).
        const bool keep_field = reuse && field_memo_valid && !c.probe && !nearest_ui_rendered &&
          ui_words == field_memo_words && (!c.apply_ui || (c.ui_alpha.handle == t[detected_mask].srv.handle &&
            c.ui_alpha.handle == field_memo_view && mask_epoch == field_memo_epoch));
        if (keep_field) ++conditioning_counts.field_memo;
        else {
          const bool pin_apart = limiter_lines || c.probe;
          if (pin_apart) source_alpha_ui = false;
          dispatch(cmd, horizontal, (height + group_lines - 1) / group_lines, 1, p, field_inputs, {field});
          source_alpha_ui = c.apply_ui;
          if (c.probe) submit_adaptive_probe(cmd, c.ui_alpha, p);
          const uint32_t pin_groups = std::max(pin_lines, 1u);
          if (pin_apart && c.apply_ui)
            dispatch(cmd, ui_apply, (height + pin_groups - 1) / pin_groups, 1, p, field_inputs, {field});
        }
        field_memo_valid = memo_valid && !c.armed && !c.probe && !nearest_ui_rendered;
        field_memo_words = ui_words;
        field_memo_view = c.ui_alpha.handle;
        field_memo_epoch = mask_epoch;
      }
      mark(cmd, mark_conditioning);
    }
    // The final side-by-side pass. Its target rests in `resting` between owners.
    void record_pack(api::command_list *cmd, api::resource target, api::resource_view view, api::resource_usage resting) {
      if (conditioning_owed) record_conditioning(cmd, owed, true);
      cmd->barrier(target, resting, api::resource_usage::render_target);
      cmd->bind_pipeline(api::pipeline_stage::all_graphics, pipelines[pq_output ? pack_pq : pack]);
      if (packed_eyes)
        bindings(cmd, api::shader_stage::all_graphics, pack_parameters,
          {textures[source].srv, pack_depth, textures[linear].srv, {}, {}, textures[field].srv});
      else
        bindings(cmd, api::shader_stage::all_graphics, pack_parameters,
          {textures[source].srv, {}, {}, {}, {}, {}, textures[left].srv, textures[right].srv});
      const api::viewport viewport{0, 0, float(width * 2), float(height), 0, 1};
      const api::rect scissor{0, 0, int32_t(width * 2), int32_t(height)};
      cmd->bind_viewports(0, 1, &viewport); cmd->bind_scissor_rects(0, 1, &scissor);
      cmd->bind_render_targets_and_depth_stencil(1, &view);
      cmd->draw(3, 1, 0, 0);
      cmd->bind_render_targets_and_depth_stencil(0, nullptr);
      cmd->barrier(target, api::resource_usage::render_target, resting);
      mark(cmd, packed_eyes ? mark_eyes : mark_pack);
      pack_owed = false;
      pack_depth = {};
    }
    void draw(api::command_list *cmd, pass id, uint32_t w, std::initializer_list<texture_id> targets,
        const render_parameters &p, const std::array<api::resource_view, 15> &srvs) {
      std::array<api::resource_view, 2> rtvs{}; unsigned index = 0;
      for (auto target : targets) {
        auto &t = textures[target]; rtvs[index++] = t.rtv;
        cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::render_target);
      }
      cmd->bind_pipeline(api::pipeline_stage::all_graphics, pipelines[id]);
      bindings(cmd, api::shader_stage::all_graphics, p, srvs);
      const api::viewport viewport{0, 0, float(w), float(height), 0, 1};
      const api::rect scissor{0, 0, int32_t(w), int32_t(height)};
      cmd->bind_viewports(0, 1, &viewport); cmd->bind_scissor_rects(0, 1, &scissor);
      cmd->bind_render_targets_and_depth_stencil(index, rtvs.data());
      cmd->draw(3, 1, 0, 0);
      cmd->bind_render_targets_and_depth_stencil(0, nullptr);
      for (auto target : targets) cmd->barrier(textures[target].resource, api::resource_usage::render_target, api::resource_usage::shader_resource);
    }
    void dispatch(api::command_list *cmd, pass id, uint32_t x, uint32_t y,
        const render_parameters &p, const std::array<api::resource_view, 15> &srvs, std::initializer_list<texture_id> targets) {
      std::array<api::resource_view, 8> uavs{};
      for (auto target : targets) {
        cmd->barrier(textures[target].resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        uavs[unsigned(target) - unsigned(raw)] = textures[target].uav;
      }
      cmd->bind_pipeline(api::pipeline_stage::compute_shader, pipelines[id]);
      bindings(cmd, api::shader_stage::compute, p, srvs, uavs);
      cmd->dispatch(x, y, 1);
      uavs.fill(null_uav);
      cmd->push_descriptors(api::shader_stage::compute, layout, 3, {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
      for (auto target : targets) cmd->barrier(textures[target].resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
    }
    void update_alpha_auto(bool eligible, const alpha_auto_source *automatic) {
      consumed_auto = {};
      if (!automatic || !automatic->session) {
        // Explicit offline/replay input is already a resolved choice. An
        // incomplete live observation never acquires that authority.
        source_alpha_ui = !automatic && eligible;
        consumed_auto.enabled = source_alpha_ui;
        consumed_auto.state = source_alpha_ui ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
        return;
      }
      const auto requested = automatic->session->decision().state;
      const bool manual_on = requested == alpha_auto_state::manual_on;
      if ((manual_on && !detection_active) || requested == alpha_auto_state::manual_off) {
        // Manual Off, or manual On through the explicit first filtered
        // candidate where detection cannot run.
        consumed_auto.state = requested;
        source_alpha_ui = eligible && manual_on;
        consumed_auto.enabled = source_alpha_ui;
        return;
      }
      // Current-frame GPU validation chooses the mask. CPU readback is only
      // a bounded status sample, never authority for a later input image;
      // poll_detection only updates which sources the session accepts.
      // Manual On through detection reports the same sample as manual_on.
      source_alpha_ui = eligible;
      consumed_auto = temporal.latest;
      // F1: the latest sample describes this frame while it is fresh, in this
      // scope and taken under the current winner.
      if (!temporal.status_fresh(*automatic)) {
        consumed_auto = {};
        consumed_auto.state = eligible ? alpha_auto_state::collecting : alpha_auto_state::waiting_for_source;
      }
      if (manual_on) {
        consumed_auto.state = alpha_auto_state::manual_on;
        consumed_auto.enabled = source_alpha_ui;
      }
      consumed_auto.scene_guard = {(scene_bits & ui_detection::per_frame_scene_hidden) != 0,
        (scene_bits & ui_detection::per_frame_pre_ui_visible) != 0, uint32_t(guard.refuted_count), scene_layer_proven};
    }
    bool prepare_detection() {
      if (detection_attempted) return detection_ready;
      detection_attempted = true;
      // Live detection binds candidate layout 2 (the offscreen UI layer at t7
      // and the accepted mask in b2 word 2); a shader of another layout, such
      // as an older embedded replay shader, would misread both, so it gets
      // none. Selection revision 10 with its 17 decision texels and both
      // scene-evidence images (the T1 grace with its hold store at u5 and the
      // HUD-less re-offer bit, the one-way judgment of every alpha but the
      // layer on sample frames, the F1 reason words, H1 with its texel 10,
      // the layer's pre-UI pixel counts in texel 11 with b2 word 4, the empty
      // change set, and the layer's premultiplied bound counted apart) is the
      // only one this renderer drives; ui_detection_replay alone reads older
      // shaders.
      if (shader_source().find("#define SUNSHINE_UI_AUTOMATIC_DETECTION 1") == std::string_view::npos ||
          shader_marker(shader_source(), ui_detection::candidate_layout_marker) != ui_detection::candidate_layout ||
          shader_marker(shader_source(), ui_selection::revision_marker) != ui_selection::revision ||
          !scene_evidence_supported() ||
          detection_tile_parts > ui_detection::max_tile_parts ||
          !texture_create(detection_statistics, ui_detection::statistics_columns(detection_tile_parts),
            ui_detection::statistics_row_count, api::format::r32g32b32a32_uint, api::resource_usage::unordered_access) ||
          !texture_create(detection_decision, detection_decision_texels, 1, api::format::r32g32b32a32_uint,
            api::resource_usage::unordered_access | api::resource_usage::copy_source) ||
          !texture_create(detected_mask, width, height, api::format::r32_float, api::resource_usage::unordered_access) ||
          !texture_create(detection_hold, ui_detection::hold::store_texels, 1, api::format::r32_uint,
            api::resource_usage::unordered_access) ||
          !pipeline_create(detection_tiles, "SunshineUIDetectionTilesCS", true, {}) ||
          !pipeline_create(detection_reduce, "SunshineUIDetectionReduceCS", true, {}) ||
          !pipeline_create(detection_mask, "SunshineUIDetectionMaskCS", true, {}) ||
          !texture_create(scene_cell_sums, ui_detection::scene::cells_x, ui_detection::scene::cells_y,
            api::format::r32g32b32a32_uint, api::resource_usage::unordered_access) ||
          !pipeline_create(scene_cells, "SunshineSceneCellsCS", true, {}) ||
          !pipeline_create(scene_compare, "SunshineSceneCompareCS", true, {}) ||
          !pipeline_create(scene_evidence, "SunshineSceneEvidenceCS", true, {})) return false;
      if (context11.p) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = desc.Height = desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Width = detection_decision_texels;
        desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(reinterpret_cast<ID3D11Device *>(device->get_native())->CreateTexture2D(&desc, nullptr,
            detection_readback11.put()))) return false;
      } else {
        auto *native = reinterpret_cast<ID3D12Device *>(device->get_native());
        const auto desc = reinterpret_cast<ID3D12Resource *>(textures[detection_decision].resource.handle)->GetDesc();
        native->GetCopyableFootprints(&desc, 0, 1, 0, &detection_footprint, nullptr, nullptr, &detection_readback_bytes);
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = detection_readback_bytes; buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(detection_readback12.put())))) return false;
      }
      if (counter_words && !prepare_counters()) counter_words = 0;
      detection_ready = true;
      return true;
    }
    bool prepare_counters() {
      if (!texture_create(ui_counter_words, counter_words, 1, api::format::r32_uint,
          api::resource_usage::unordered_access | api::resource_usage::copy_source)) return false;
      if (context11.p) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = counter_words;
        desc.Height = desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R32_UINT;
        desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        return SUCCEEDED(reinterpret_cast<ID3D11Device *>(device->get_native())->CreateTexture2D(&desc, nullptr,
          counters_readback11.put()));
      }
      auto *native = reinterpret_cast<ID3D12Device *>(device->get_native());
      const auto desc = reinterpret_cast<ID3D12Resource *>(textures[ui_counter_words].resource.handle)->GetDesc();
      native->GetCopyableFootprints(&desc, 0, 1, 0, &counters_footprint, nullptr, nullptr, &counters_readback_bytes);
      D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
      D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      buffer.Width = counters_readback_bytes; buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
      buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      return SUCCEEDED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(counters_readback12.put())));
    }
    // The GPU words of the sample just completed (counter_words of them);
    // false when not readable now (the totals are cumulative, so a later
    // sample catches up).
    bool read_counters(std::array<uint32_t, ui_counter_word::count> &words) {
      const auto bytes = std::min<size_t>(words.size(), counter_words) * sizeof(uint32_t);
      if (context11.p) {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(context11->Map(counters_readback11.p, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped))) return false;
        std::memcpy(words.data(), mapped.pData, bytes);
        context11->Unmap(counters_readback11.p, 0);
        return true;
      }
      void *mapped{}; const D3D12_RANGE range{0, SIZE_T(counters_readback_bytes)};
      if (FAILED(counters_readback12->Map(0, &range, &mapped))) return false;
      std::memcpy(words.data(), static_cast<const unsigned char *>(mapped) + counters_footprint.Offset, bytes);
      const D3D12_RANGE written{0, 0}; counters_readback12->Unmap(0, &written);
      return true;
    }
    // Commits a completed sample's counts (ui_temporal::sample_counters).
    void commit_counters(const alpha_auto_decision &sample, alpha_auto_policy *session,
        const scene_guard::observation &observed) {
      std::array<uint32_t, ui_counter_word::count> words{};
      if (!counters_pending || !read_counters(words)) return;
      const auto delta = ui_temporal::sample_counters(sample, pending_counts, committed_counts, words, committed_words,
        detection_pending_source.now_ms, observed);
      committed_counts = pending_counts;
      committed_words = words;
      if (session) session->add_counters(delta);
    }
    void poll_detection(const alpha_auto_source &input) {
      if (!detection_pending || detection_awaiting_signal || !detection_fence) return;
      const auto completed = device->get_completed_fence_value(completion);
      if (completed == UINT64_MAX) { failed = true; return; }
      if (completed < detection_fence) return;
      if (ui_temporal::sample_discarded(input, detection_pending_source)) {
        // A sample from another scope, or stale on arrival, is no evidence for
        // this one (F1: a change of offered or accepted candidates does not
        // discard it). Its counts are cumulative: the next committed sample
        // includes them.
        detection_pending = counters_pending = false;
        return;
      }
      std::vector<uint32_t> counts(4 * size_t(detection_decision_texels));
      const auto bytes = counts.size() * sizeof(uint32_t);
      bool read = false;
      ++detection_mapped;
      if (context11.p) {
        D3D11_MAPPED_SUBRESOURCE mapped{};
        const auto result = context11->Map(detection_readback11.p, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING) return;
        if (SUCCEEDED(result)) {
          std::memcpy(counts.data(), mapped.pData, bytes);
          context11->Unmap(detection_readback11.p, 0); read = true;
        }
      } else {
        void *mapped{}; const D3D12_RANGE range{0, SIZE_T(detection_readback_bytes)};
        if (SUCCEEDED(detection_readback12->Map(0, &range, &mapped))) {
          std::memcpy(counts.data(), static_cast<const unsigned char *>(mapped) + detection_footprint.Offset, bytes);
          const D3D12_RANGE written{0, 0}; detection_readback12->Unmap(0, &written); read = true;
        }
      }
      detection_pending = false;
      auto &latest = temporal.latest;
      latest = {};
      if (!read) { counters_pending = false; return; }
      latest = ui_temporal::decode_detection_sample(counts.data(), counts.size(), detection_pending_source.now_ms,
        detection_submitted);
      temporal.latest_source = detection_pending_source;
      temporal.latest_key = detection_pending_status_key;
      // The scene guard reads the decoded sample before the acceptance
      // ledger observes it (game3d_scene_guard.h).
      const auto observed = guard.observe(ui_temporal::guard_sample(latest), detection_pending_actionable,
        detection_pending_signatures.by_kind());
      // Earns or revokes acceptance of the signatures this sample was taken
      // for; a session in a manual mode ignores it (S2).
      if (input.session)
        input.session->observe(latest.evidence, latest.pixels, detection_pending_source.now_ms, detection_pending_signatures);
      commit_counters(latest, input.session, observed);
      counters_pending = false;
    }
    // One real frame's detection, from its own offered candidates, accepted
    // candidates and stored layer flags (b2), with per_frame joining the
    // pushed flags only. A frame that offers nothing (bits zero, pushed for
    // the T1 grace with per_frame_accepted_missing) runs the reduce and mask
    // passes only and never submits a sample: its statistics rows are the
    // previous frame's, which the reduce masks off with offered zero.
    void detect_ui(api::command_list *cmd, const render_parameters &p, const ui_detection_inputs &input,
        const alpha_auto_source &observation, api::resource_view paired_color, api::resource_view depth,
        uint32_t bits, uint32_t accepted, uint32_t flags, uint32_t per_frame) {
      std::array<api::resource_view, 15> views{};
      // A HUD-less image is compared with the color of the frame it belongs to
      // (t0): its batch's tagged Backbuffer or a retained Present; eye
      // rendering stays current.
      views[0] = paired_color.handle ? paired_color : textures[source].srv;
      // The presented color (t6): its alpha is the current-alpha candidate
      // (tiles and mask passes), the one-way test reads it beside the pair
      // (A2), the tiles pass compares the offscreen UI layer with it (H1 d)
      // and the evidence passes measure it.
      views[6] = textures[source].srv;
      // Candidate layout 2: the offscreen UI layer in its own slot (t7).
      views[7] = input.layer;
      for (unsigned i = 0; i != 3; ++i) views[11+i] = input.masks[i];
      views[14] = input.hudless;
      // A sample frame: the CPU reads the decision texels of at most one real
      // frame every 100 ms. Only it pushes per_frame_sample, so that the
      // tiles pass and the reduce count the one-way judgment (A2) and the
      // pre-UI pixels (texel 11) in their sample phase, and the pre-UI
      // threshold; every other frame pushes neither, which skips the
      // judgment's per-pixel work, the presented-colour loads and the second
      // sums (its one-way and pre-UI counts are zero).
      const bool sample = bits && !detection_pending && !(detection_last_submit &&
        observation.now_ms >= detection_last_submit && observation.now_ms - detection_last_submit < 100);
      const float pushed_pre_ui_threshold = sample ? pre_ui_threshold : 0.f;
      // The b2 constants of every detection pass. Per-frame bits join the
      // pushed flags only, never the stored flags.
      uint32_t threshold_bits, pre_ui_threshold_bits;
      std::memcpy(&threshold_bits, &difference_threshold, sizeof(threshold_bits));
      std::memcpy(&pre_ui_threshold_bits, &pushed_pre_ui_threshold, sizeof(pre_ui_threshold_bits));
      detection_run = {ui_detection_snapshot::run_state::ran, bits, threshold_bits, accepted,
        flags | per_frame | (sample ? ui_detection::per_frame_sample : 0u), flags};
      detection_run.pre_ui_threshold_bits = pre_ui_threshold_bits;
      // The reduce also adds this frame to the exact counters at u7.
      auto &counter_texture = textures[ui_counter_words];
      if (counter_words && !counters_cleared) {
        const uint32_t zero[4]{};
        cmd->barrier(counter_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        cmd->clear_unordered_access_view_uint(counter_texture.uav, zero);
        cmd->barrier(counter_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        counters_cleared = true;
      }
      // The T1 hold store needs no initial clear: every chain's first
      // detection (no decision in scope, a new renderer or a resume included)
      // pushes per_frame_hold_reset, so the reduce ignores the store's old
      // contents and overwrites all of it; later detections of the chain read
      // what the chain wrote.
      auto &hold_texture = textures[detection_hold];
      // The reduce (reduce true) also binds the hold store at u5; no other
      // detection pass reads it.
      const auto dispatch_stage = [&](pass stage, texture_id target, unsigned output, unsigned x, unsigned y,
          bool count = false, bool reduce = false, unsigned z = 1) {
        auto &t = textures[target];
        count = count && counter_words;
        cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        if (count) cmd->barrier(counter_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        if (reduce) cmd->barrier(hold_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        std::array<api::resource_view, 8> uavs{}; uavs[output] = t.uav;
        if (count) uavs[7] = counter_texture.uav;
        if (reduce) uavs[5] = hold_texture.uav;
        cmd->bind_pipeline(api::pipeline_stage::compute_shader, pipelines[stage]);
        bindings(cmd, api::shader_stage::compute, p, views, uavs);
        // b2 word 5 is reserved and pushed as zero.
        struct constants {
          uint32_t bits;
          float threshold;
          uint32_t accepted, flags;
          float pre_ui_threshold;
          uint32_t reserved;
        } values{detection_run.candidates, difference_threshold, detection_run.accepted, detection_run.flags,
          pushed_pre_ui_threshold, 0u};
        static_assert(sizeof(values) == ui_detection::b2_words * sizeof(uint32_t));
        cmd->push_constants(api::shader_stage::compute, layout, 5, 0, ui_detection::b2_words, &values);
        cmd->dispatch(x, y, z);
        uavs.fill(null_uav);
        cmd->push_descriptors(api::shader_stage::compute, layout, 3,
          {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
        cmd->barrier(t.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        if (count) cmd->barrier(counter_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        if (reduce) cmd->barrier(hold_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
      };
      // Each statistics tile in its parts (group z), which the reduce adds.
      if (bits) dispatch_stage(detection_tiles, detection_statistics, 6, 16, 16, false, false,
        std::max(detection_tile_parts, 1u));
      views[10] = textures[detection_statistics].srv;
      dispatch_stage(detection_reduce, detection_decision, 6, 1, 1, true, true);
      views[10] = textures[detection_decision].srv;
      // A reused decision (texel 9 bit 16) leaves detected_mask as the
      // previous real frame made it.
      dispatch_stage(detection_mask, detected_mask, 0, (width+7)/8, (height+7)/8);
      ++mask_epoch;
      if (!sample) return;
      // A sample frame. Hidden-scene evidence only measures this frame for the
      // CPU and writes decision texels 5 and 6 after the decision and mask;
      // without an acting-capable claim in the latest sample or a held
      // verdict nothing runs (scene_guard::state::measure; a proven layer that
      // is the offer's pre-UI image always measures), and neither does it over
      // depth that is not this frame's, which is no evidence (the sample stays
      // not actionable, as the evidence sum would read it invalid). The cells pass reads
      // the pre-UI scene image the b2 constants name: the HUD-less image
      // (t14) when offered, else the offscreen UI layer's colour (t7), both
      // still bound from the detection passes above, beside the presented
      // color (t6).
      const bool proven_image = scene_layer_proven && ui_selection::pre_ui_image_of(bits) == ui_detection::pre_ui_image::layer;
      detection_pending_actionable = false;
      if (scene_evidence_supported() && !(per_frame & ui_detection::per_frame_depth_not_current) &&
          guard.measure(observation.now_ms, proven_image)) {
        namespace scene = ui_detection::scene;
        detection_pending_actionable = true;
        views[1] = depth;
        views[10] = {};
        dispatch_stage(scene_cells, scene_cell_sums, 6, scene::cells_x / 16, scene::cells_y);
        views[10] = textures[scene_cell_sums].srv;
        dispatch_stage(scene_compare, detection_statistics, 6, scene::cells_x / 16, scene::cells_y / 16);
        views[10] = textures[detection_statistics].srv;
        dispatch_stage(scene_evidence, detection_decision, 6, 1, 1);
        ++scene_evidence_runs;
      }
      // The single small diagnostic slot cannot accumulate GPU work. Its sample
      // may lag; the mask above always uses this frame's completed GPU decision.
      views.fill(null_srv);
      cmd->push_descriptors(api::shader_stage::compute, layout, 2,
        {{}, 0, 0, uint32_t(views.size()), api::descriptor_type::shader_resource_view, views.data()});
      auto &t = textures[detection_decision];
      cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::copy_source);
      if (context11.p) context11->CopyResource(detection_readback11.p, reinterpret_cast<ID3D11Resource *>(t.resource.handle));
      else {
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = reinterpret_cast<ID3D12Resource *>(t.resource.handle);
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = detection_readback12.p; destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = detection_footprint;
        reinterpret_cast<ID3D12GraphicsCommandList *>(cmd->get_native())->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      }
      cmd->barrier(t.resource, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      // The counters through this frame, under the same fence, with the CPU
      // counts of every render recorded so far.
      counters_pending = counter_words != 0;
      if (counters_pending) {
        cmd->barrier(counter_texture.resource, api::resource_usage::shader_resource, api::resource_usage::copy_source);
        if (context11.p) context11->CopyResource(counters_readback11.p, reinterpret_cast<ID3D11Resource *>(counter_texture.resource.handle));
        else {
          D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
          source.pResource = reinterpret_cast<ID3D12Resource *>(counter_texture.resource.handle);
          source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          destination.pResource = counters_readback12.p; destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
          destination.PlacedFootprint = counters_footprint;
          reinterpret_cast<ID3D12GraphicsCommandList *>(cmd->get_native())->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        }
        cmd->barrier(counter_texture.resource, api::resource_usage::copy_source, api::resource_usage::shader_resource);
        pending_counts = cpu_counts;
      }
      detection_pending_source = observation; detection_pending_status_key = temporal.status_key();
      detection_pending_signatures = input.signatures;
      detection_pending = detection_awaiting_signal = true;
      detection_last_submit = observation.now_ms; ++detection_submitted;
    }
    bool prepare_adaptive_probe() {
      if (adaptive_probe_attempted) return adaptive_probe_ready;
      adaptive_probe_attempted = true;
      if (!display_fraction_ui_supported ||
          shader_source().find("#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1") == std::string_view::npos ||
          !texture_create(ui_conflict_statistics, 16, 32, api::format::r32g32b32a32_uint,
            api::resource_usage::unordered_access | api::resource_usage::copy_source) ||
          !pipeline_create(ui_conflict, "SunshineUIConflictCS", true, {}) ||
          (!pipelines[ui_apply].handle && !pipeline_create(ui_apply, "SunshineApplyUICS", true, {}))) return false;
      if (context11.p) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = 16; desc.Height = 32;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        auto *native = reinterpret_cast<ID3D11Device *>(device->get_native());
        if (FAILED(native->CreateTexture2D(&desc, nullptr, adaptive_readback11.put()))) return false;
      } else {
        auto *native = reinterpret_cast<ID3D12Device *>(device->get_native());
        const auto desc = reinterpret_cast<ID3D12Resource *>(textures[ui_conflict_statistics].resource.handle)->GetDesc();
        native->GetCopyableFootprints(&desc, 0, 1, 0, &adaptive_footprint, nullptr, nullptr, &adaptive_readback_bytes);
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = adaptive_readback_bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(adaptive_readback12.put())))) return false;
      }
      adaptive_probe_ready = true;
      return true;
    }
    void adaptive_unavailable(const ui_adaptive::source &input, const char *reason) {
      if (adaptive_scope_known && ui_adaptive::fresh(adaptive_scope)) ++adaptive_scope_generation;
      adaptive_scope.eligible = false;
      auto unavailable = input;
      unavailable.eligible = false;
      adaptive_policy.prepare(unavailable);
      consumed_adaptive = adaptive_policy.current();
      consumed_adaptive.status = reason;
    }
    void poll_adaptive_probe(const ui_adaptive::source *input) {
      if (!adaptive_readback_pending || adaptive_awaiting_signal || !adaptive_fence) return;
      const auto completed = device->get_completed_fence_value(completion);
      if (completed == UINT64_MAX) {
        adaptive_probe_ready = false;
        failed = true;
        if (input) adaptive_unavailable(*input, "device_lost");
        return;
      }
      if (completed < adaptive_fence) return;
      if (!input || !ui_adaptive::fresh(*input) || adaptive_pending_generation != adaptive_scope_generation ||
          !adaptive_pending_source.tick_ms || adaptive_pending_source.tick_ms > input->now_ms ||
          input->now_ms - adaptive_pending_source.tick_ms > ui_adaptive::max_age_ms) {
        adaptive_readback_pending = false;
        return;
      }
      const unsigned char *pixels = nullptr;
      uint32_t pitch = 0;
      D3D11_MAPPED_SUBRESOURCE mapped11{};
      void *mapped12 = nullptr;
      HRESULT result;
      if (context11.p) {
        result = context11->Map(adaptive_readback11.p, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped11);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING) return;
        if (SUCCEEDED(result)) { pixels = static_cast<const unsigned char *>(mapped11.pData); pitch = mapped11.RowPitch; }
      } else {
        const D3D12_RANGE range{0, SIZE_T(adaptive_readback_bytes)};
        result = adaptive_readback12->Map(0, &range, &mapped12);
        if (SUCCEEDED(result)) {
          pixels = static_cast<const unsigned char *>(mapped12) + adaptive_footprint.Offset;
          pitch = adaptive_footprint.Footprint.RowPitch;
        }
      }
      ui_adaptive::sample sample;
      sample.provenance = adaptive_pending_source;
      sample.width = width; sample.height = height;
      const bool readable = pixels && pitch >= 16 * 4 * sizeof(uint32_t);
      if (readable) for (unsigned y = 0; y < 16; ++y) for (unsigned x = 0; x < 16; ++x) {
        std::array<uint32_t, 4> first{}, second{};
        const size_t row = size_t(y) * 2;
        std::memcpy(first.data(), pixels + row * pitch + x * sizeof(first), sizeof(first));
        std::memcpy(second.data(), pixels + (row + 1) * pitch + x * sizeof(second), sizeof(second));
        auto &tile = sample.tiles[y * 16 + x];
        tile.covered = first[0]; tile.invalid = first[1];
        tile.bad = {first[2], first[3], second[0], second[1], second[2]};
        tile.pixels = second[3];
      }
      if (mapped11.pData) context11->Unmap(adaptive_readback11.p, 0);
      if (mapped12) { const D3D12_RANGE written{0, 0}; adaptive_readback12->Unmap(0, &written); }
      adaptive_readback_pending = false;
      if (!readable) {
        adaptive_probe_ready = false;
        if (result == DXGI_ERROR_DEVICE_REMOVED || result == DXGI_ERROR_DEVICE_RESET) failed = true;
        adaptive_unavailable(*input, failed ? "device_lost" : "readback_failed");
        return;
      }
      const auto previous = adaptive_policy.current();
      const bool accepted = adaptive_policy.observe(sample, input->now_ms);
      consumed_adaptive = adaptive_policy.current();
      if (accepted && (!previous.accepted_sequence || previous.target_uv != consumed_adaptive.target_uv ||
          previous.capped_conflict != consumed_adaptive.capped_conflict)) {
        char message[512];
        std::snprintf(message, sizeof(message),
          "Sunshine UI plane: target_uv=%.7f applied_uv=%.7f required_uv=%.7f limit_uv=%.7f applied=%.3f%% covered=%llu conflicts=%llu capped_conflict=%u depth_sequence=%llu mask_sequence=%llu age_ms=%llu region=center75 criterion=ui_ratio_or_area entry=%u%% area_entry=%.1f%%",
          double(consumed_adaptive.target_uv), double(consumed_adaptive.applied_uv),
          double(consumed_adaptive.required_uv), double(consumed_adaptive.limit_uv),
          double(consumed_adaptive.applied_fraction) * 100.,
          static_cast<unsigned long long>(consumed_adaptive.covered_pixels),
          static_cast<unsigned long long>(consumed_adaptive.conflict_pixels), unsigned(consumed_adaptive.capped_conflict),
          static_cast<unsigned long long>(consumed_adaptive.accepted_sequence),
          static_cast<unsigned long long>(consumed_adaptive.accepted_mask_sequence),
          static_cast<unsigned long long>(consumed_adaptive.probe_age_ms), unsigned(ui_adaptive::entry_conflict_percent),
          double(ui_adaptive::entry_area_per_mille) / 10.);
        sunshine_log::message(reshade::log::level::info, message);
      }
    }
    bool prepare_adaptive_frame(const render_parameters &p, const ui_adaptive::source *source) {
      consumed_adaptive = {};
      if (!source || consumed_plane.mode != ui_plane_mode::display_fraction) {
        if (adaptive_scope_known) {
          ++adaptive_scope_generation;
          adaptive_scope_known = false;
          adaptive_policy.reset();
        }
        poll_adaptive_probe(nullptr);
        consumed_adaptive.applied_fraction = consumed_plane.inverse_depth;
        if (consumed_plane.mode == ui_plane_mode::display_fraction && std::isfinite(consumed_plane.inverse_depth) &&
            consumed_plane.inverse_depth >= 0.f && consumed_plane.inverse_depth <= .75f) {
          consumed_adaptive.applied_uv = consumed_plane.inverse_depth * display_parallax_cap_uv(p);
          consumed_adaptive.target_uv = consumed_adaptive.applied_uv;
        }
        consumed_adaptive.status = "frozen";
        return false;
      }
      auto input = *source;
      input.front_cap_uv = display_parallax_cap_uv(p);
      input.eligible = input.eligible && source_alpha_ui && display_fraction_ui_supported &&
        width <= 3840 && height <= 3840 && p.depth_ready && p.camera_ready &&
        input.front_cap_uv > 0.f;
      const bool unavailable = ui_adaptive::fresh(input) && !prepare_adaptive_probe();
      if (unavailable) input.eligible = false;
      if (!adaptive_scope_known || !ui_adaptive::same_scope(adaptive_scope, input) ||
          input.now_ms < adaptive_scope.now_ms) {
        ++adaptive_scope_generation;
        adaptive_last_sequence = adaptive_last_mask_sequence = adaptive_last_tick = 0;
        adaptive_scope_known = true;
      } else if (adaptive_scope.front_cap_uv != input.front_cap_uv ||
          (ui_adaptive::fresh(adaptive_scope) && !ui_adaptive::fresh(input))) {
        // An unfinished probe cannot bridge a cap change or ineligible frame.
        // Retain its GPU slot until the fence retires, but reject its evidence
        // even if this same source recovers before readback completion.
        // Keep the last submission time so a strength ramp stays throttled.
        ++adaptive_scope_generation;
      }
      // Scope changes invalidate evidence, not the renderer's work budget.
      // Only a regressed clock needs a new cadence origin.
      if (input.now_ms < adaptive_last_submit) adaptive_last_submit = 0;
      adaptive_scope = input;
      adaptive_policy.prepare(input);
      consumed_adaptive = adaptive_policy.current();
      poll_adaptive_probe(&input);
      if (unavailable) consumed_adaptive.status = "probe_unavailable";
      consumed_plane.inverse_depth = consumed_adaptive.applied_fraction;
      return !failed && !unavailable && ui_adaptive::fresh(input) && adaptive_probe_ready && !adaptive_readback_pending &&
        input.sequence > adaptive_last_sequence && (!input.mask_sequence || input.mask_sequence > adaptive_last_mask_sequence) &&
        input.tick_ms >= adaptive_last_tick &&
        (!adaptive_last_submit || input.now_ms - adaptive_last_submit >= ui_adaptive::probe_interval_ms);
    }
    void submit_adaptive_probe(api::command_list *cmd, api::resource_view selected, const render_parameters &p) {
      auto &t = textures[ui_conflict_statistics];
      cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
      std::array<api::resource_view, 8> uavs{}; uavs[7] = t.uav;
      cmd->bind_pipeline(api::pipeline_stage::compute_shader, pipelines[ui_conflict]);
      bindings(cmd, api::shader_stage::compute, p, {selected, {}, {}, {}, {}, textures[field].srv}, uavs);
      cmd->dispatch(16, 16, 1);
      uavs.fill(null_uav);
      cmd->push_descriptors(api::shader_stage::compute, layout, 3,
        {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
      // Unbind the pre-UI field SRV before the following UI application writes it.
      std::array<api::resource_view, 15> srvs{}; srvs.fill(null_srv);
      cmd->push_descriptors(api::shader_stage::compute, layout, 2,
        {{}, 0, 0, uint32_t(srvs.size()), api::descriptor_type::shader_resource_view, srvs.data()});
      cmd->barrier(t.resource, api::resource_usage::unordered_access, api::resource_usage::copy_source);
      if (context11.p) {
        context11->CopyResource(adaptive_readback11.p, reinterpret_cast<ID3D11Resource *>(t.resource.handle));
      } else {
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = reinterpret_cast<ID3D12Resource *>(t.resource.handle);
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = adaptive_readback12.p;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = adaptive_footprint;
        reinterpret_cast<ID3D12GraphicsCommandList *>(cmd->get_native())->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      }
      cmd->barrier(t.resource, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      adaptive_pending_source = adaptive_scope;
      adaptive_pending_generation = adaptive_scope_generation;
      adaptive_last_submit = adaptive_scope.now_ms;
      ++adaptive_submitted;
      adaptive_last_sequence = adaptive_scope.sequence;
      if (adaptive_scope.mask_sequence) adaptive_last_mask_sequence = adaptive_scope.mask_sequence;
      adaptive_last_tick = adaptive_scope.tick_ms;
      adaptive_fence = 0;
      adaptive_readback_pending = adaptive_awaiting_signal = true;
    }
  };
  renderer::renderer() = default;
  renderer::~renderer() {
    // Explicit hot-unload is rare. Never free a recorded frame or block the game.
    if (data_ && !data_->idle()) data_.release();
    if (cached_ && !cached_->idle()) cached_.release();
  }
  std::string_view renderer::shader_source() { return {game3d_shader_source, sizeof(game3d_shader_source) - 1}; }
  std::string_view renderer::active_shader_source() const { return data_ ? data_->shader_source() : shader_source(); }
  // Every entry point a renderer for this configuration may create, eagerly
  // or on first use, so background preparation leaves no compile on Present.
  static std::vector<shader_cache::entry_point> shader_entries(std::string_view source, uint32_t w, uint32_t h, uint32_t c) {
    const auto has = [source](const char *marker) { return source.find(marker) != std::string_view::npos; };
    std::vector<shader_cache::entry_point> entries{{"PostProcessVS", "vs_5_0"}};
    if (c == 3)
      entries.push_back(has("#define SUNSHINE_PACKED_EYES 1") && has("#define SUNSHINE_PQ_PER_TAP 1") ?
        shader_cache::entry_point{"SunshineRenderPackedPQPS", "ps_5_0"} : shader_cache::entry_point{"SunshinePreparePQPS", "ps_5_0"});
    if (c == 2 && has("#define SUNSHINE_PACKED_EYES 1") && has("#define SUNSHINE_SCRGB_PQ_PACK 1"))
      entries.push_back({"SunshineRenderPackedPQPS", "ps_5_0"});
    if (w <= 3840 && h <= 3840) {
      entries.insert(entries.end(), {{"SunshineHostCandidateCS", "cs_5_0"}, {"SunshineHostVerticalCS", "cs_5_0"},
        {"SunshineHostHorizontalCS", "cs_5_0"}});
      if (shader_marker(source, "SUNSHINE_LIMITER_LINE_GROUPS") || has("#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1"))
        entries.push_back({"SunshineApplyUICS", "cs_5_0"});
      if (has("#define SUNSHINE_VERTICAL_LIVE_ENTRY 1")) entries.push_back({"SunshineHostVerticalLiveCS", "cs_5_0"});
      if (has("#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1")) entries.push_back({"SunshineUIConflictCS", "cs_5_0"});
      if (has("#define SUNSHINE_UI_NEAREST_PLANE 1"))
        entries.insert(entries.end(), {{"SunshineUINearestTilesCS", "cs_5_0"}, {"SunshineUINearestReduceCS", "cs_5_0"}});
      if (has("#define SUNSHINE_UI_AUTOMATIC_DETECTION 1"))
        entries.insert(entries.end(), {{"SunshineUIDetectionTilesCS", "cs_5_0"}, {"SunshineUIDetectionReduceCS", "cs_5_0"},
          {"SunshineUIDetectionMaskCS", "cs_5_0"}});
      if (has("#define SUNSHINE_UI_AUTOMATIC_DETECTION 1") &&
          shader_marker(source, ui_detection::scene_evidence_images_marker) == ui_detection::max_scene_evidence_images)
        entries.insert(entries.end(), {{"SunshineSceneCellsCS", "cs_5_0"}, {"SunshineSceneCompareCS", "cs_5_0"},
          {"SunshineSceneEvidenceCS", "cs_5_0"}});
    }
    if (has("#define SUNSHINE_PACKED_EYES 1")) entries.push_back({"SunshineRenderPackedPS", "ps_5_0"});
    else entries.insert(entries.end(), {{"SunshineRenderEyesPS", "ps_5_0"}, {"SunshinePackEyesPS", "ps_5_0"}});
    return entries;
  }
  bool renderer::configure(api::effect_runtime *runtime, api::resource backbuffer, api::color_space color, std::string_view source_override,
      bool prepare_in_background) {
    preparing_ = false;
    auto *device = runtime->get_device();
    const auto desc = device->get_resource_desc(backbuffer);
    const auto format = typed(desc.texture.format);
    const uint32_t c = color == api::color_space::srgb ? 1 : color == api::color_space::scrgb ? 2 : color == api::color_space::hdr10_pq ? 3 : 0;
    if (!c || desc.type != api::resource_type::texture_2d || !desc.texture.width || !desc.texture.height ||
        desc.texture.width > 8192 || desc.texture.height > 8192 || (desc.texture.width % 2) || (desc.texture.height % 2) ||
        desc.texture.samples != 1 || desc.texture.depth_or_layers != 1 ||
        // ReShade draws its GUI to an internal alpha-bearing resolve target
        // for X8 swapchains. Our native overlay target must be the backbuffer.
        format == api::format::r8g8b8x8_unorm || format == api::format::b8g8r8x8_unorm) return false;
    const auto matches = [&](const impl &value) {
      return value.device == device && value.width == desc.texture.width && value.height == desc.texture.height &&
        value.source_format == typed(desc.texture.format) && value.color == c && value.source_override == source_override;
    };
    const auto now = GetTickCount64();
    // The cached renderer of the other colour transfer is released once idle
    // this long (item 6: bounded to one cached entry).
    if (cached_ && now - cached_since_ >= cached_idle_ms && cached_->idle()) cached_.reset();
    if (data_ && matches(*data_)) {
      // The first configure after a parked runtime reset starts the current
      // renderer's decision state as a new renderer would.
      if (resume_pending_ && !data_->failed) {
        if (data_->pending || data_->unsignaled) return false;
        resume_pending_ = false;
        ui_candidate_captures_.fill(0);
        data_->resume();
      }
      return !data_->failed;
    }
    // A toggle back (SDR and HDR) reuses the cached renderer: its pipelines,
    // textures and detection state, without a wait or recompilation. The
    // outgoing one has no unsignaled work, so its own fence retires it.
    if (cached_ && matches(*cached_) && !cached_->failed && (!data_ || (!data_->pending && !data_->unsignaled))) {
      std::swap(data_, cached_);
      cached_since_ = now;
      resume_pending_ = false;
      ui_candidate_captures_.fill(0);
      if (data_) data_->resume();
      sunshine_log::message(reshade::log::level::info, "Sunshine Game 3D: reusing the cached add-on GPU renderer of this colour transfer");
      return true;
    }
    if (data_ && !data_->idle()) return false;
    const bool background = prepare_in_background && source_override.empty();
    if (background) {
      const auto state = shader_cache::prepare({shader_source(), desc.texture.width, desc.texture.height, c},
        shader_entries(shader_source(), desc.texture.width, desc.texture.height, c));
      if (state != shader_cache::state::ready) {
        preparing_ = state == shader_cache::state::pending;
        return false;
      }
    }
    ui_candidate_captures_.fill(0);
    // The outgoing renderer (idle) stays cached for a toggle back, replacing
    // an older cached one only once that is idle too.
    if (data_ && !data_->failed && (!cached_ || cached_->idle())) {
      cached_ = std::move(data_);
      cached_since_ = now;
    }
    data_.reset();
    resume_pending_ = false;
    auto next = std::make_unique<impl>();
    next->source_override.assign(source_override);
    next->precompiled = background;
    if (!next->initialize(runtime, desc, c)) return false;
    data_ = std::move(next);
    sunshine_log::message(reshade::log::level::info, "Sunshine Game 3D: add-on GPU renderer ready (no FX file required)");
    return true;
  }
  bool renderer::render(api::command_list *cmd, const render_frame_input &input, bool defer_pack) {
    const auto backbuffer = input.color;
    auto depth = input.depth;
    const auto &parameters = input.scene;
    const auto &ui = input.ui;
    const bool source_alpha_ui = ui.available();
    auto alpha_source = !source_alpha_ui || ui.kind == ui_input_kind::current_color_alpha ? api::resource_view{} : ui.view;
    const auto &plane = ui.plane;
    auto observation = ui.automatic ? *ui.automatic : alpha_auto_source{};
    observation.retained = ui.kind == ui_input_kind::captured_color_alpha || ui.kind == ui_input_kind::dedicated_mask;
    const auto *automatic = ui.automatic ? &observation : nullptr;
    const auto *adaptive = ui.adaptive;
    const auto channel = source_alpha_ui ? ui.channel : ui_mask_channel::alpha;
    if (!data_ || data_->failed || data_->pending) return false;
    auto &d = *data_;
    // The previous render's owed conditioning, if its pack never came, is dropped.
    d.conditioning_owed = false;
    const auto mode = automatic && automatic->session ? automatic->session->decision().state : alpha_auto_state::manual_off;
    const bool auto_mode = automatic && automatic->session && mode != alpha_auto_state::manual_on && mode != alpha_auto_state::manual_off;
    // Manual On validates the provider's filtered candidates through detection,
    // each accepted for this session only (S2); where detection cannot run it
    // keeps the explicit first filtered candidate.
    const bool manual_detection = automatic && automatic->session && mode == alpha_auto_state::manual_on && ui.detection;
    namespace candidate = ui_detection::candidate;
    using ui_selection::kind;
    ui_detection_inputs candidates;
    candidates.current_color = ui.kind == ui_input_kind::current_color_alpha;
    if (ui.detection) candidates = *ui.detection;
    else if (ui.kind == ui_input_kind::hudless_difference) candidates.hudless = ui.view;
    else if (ui.kind == ui_input_kind::captured_color_alpha) candidates.masks[2] = ui.view;
    else if (ui.kind == ui_input_kind::dedicated_mask) candidates.masks[channel == ui_mask_channel::red ? 0 : 1] = ui.view;
    const auto compatible = [&](api::resource_view view) {
      if (!view.handle) return false;
      const auto desc = d.device->get_resource_desc(d.resource_of(view));
      return desc.type == api::resource_type::texture_2d && desc.texture.width == d.width && desc.texture.height == d.height &&
        desc.texture.samples == 1 && desc.texture.depth_or_layers == 1;
    };
    const auto format_of = [&](api::resource_view view) {
      return uint32_t(typed(d.device->get_resource_desc(d.resource_of(view)).texture.format));
    };
    uint32_t bits = candidates.current_color ? candidate::current : 0u;
    constexpr std::array<uint32_t, 3> mask_bits{candidate::ui_alpha, candidate::ui_color, candidate::backbuffer};
    for (unsigned i = 0; i < candidates.masks.size(); ++i) {
      if (compatible(candidates.masks[i])) bits |= mask_bits[i];
      else candidates.masks[i] = {};
    }
    if (compatible(candidates.layer)) bits |= candidate::layer;
    else { candidates.layer = {}; candidates.layer_flags = 0; }
    api::resource_view hudless_color{};
    if (candidates.hudless.handle && candidates.hudless_pair.handle) {
      hudless_color = candidates.hudless_pair; // Same tag batch: an exact pair.
    } else if (candidates.hudless.handle && candidates.hudless_presents_ago) {
      d.retention_wanted = true; // Retain colors from now on; this frame has none yet.
      d.retention_requested_present = d.present_number;
      hudless_color = d.retained_view(candidates.hudless_presents_ago);
      if (!hudless_color.handle) candidates.hudless = {};
    }
    // V2: a HUD-less image pairs only with color of the same transfer, from
    // the two snapshots' own encodings; the pair's difference threshold
    // follows from them. Without a HUD-less image the threshold is the
    // presented color's own, as the replay computes it.
    const ui_selection::encoding presented{uint32_t(d.source_format), d.color};
    std::optional<float> pair_threshold;
    if (hudless_color.handle && !compatible(hudless_color)) { candidates.hudless = {}; hudless_color = {}; }
    if (compatible(candidates.hudless))
      pair_threshold = ui_selection::comparable({format_of(candidates.hudless), d.color},
        hudless_color.handle ? ui_selection::encoding{format_of(hudless_color), d.color} : presented);
    if (pair_threshold) bits |= candidates.hudless_exact ? candidate::hudless | candidate::exact : candidate::hudless;
    else { candidates.hudless = {}; hudless_color = {}; }
    // The acceptance signature of each offered kind (A1): the provider's, or
    // the bound view's typed format and the swapchain color space.
    auto &signatures = candidates.signatures;
    if (!signatures.color_space) signatures.color_space = d.color;
    const auto sign = [&](kind k, api::resource_view view) {
      if ((bits & ui_selection::bit(k)) && !signatures.format[std::size_t(k)]) signatures.set(k, format_of(view));
    };
    sign(kind::ui_alpha, candidates.masks[0]);
    sign(kind::ui_color, candidates.masks[1]);
    sign(kind::backbuffer, candidates.masks[2]);
    sign(kind::ui_layer, candidates.layer);
    sign(kind::hudless, candidates.hudless);
    if ((bits & candidate::current) && !signatures.format[std::size_t(kind::current)])
      signatures.set(kind::current, uint32_t(d.source_format));
    // Only accepted candidates decide (S1): the session's ledger in Auto,
    // every offered candidate in manual On, none in manual Off. Explicit
    // offline input without a session is already a resolved choice.
    uint32_t accepted = automatic && automatic->session ? automatic->session->accepted(bits, signatures) :
      bits & ui_selection::candidate_bits;
    // T1 (ui_temporal::detection_state::arbitrate), in the call order
    // game3d_ui_temporal.h documents: a real Present detects from its own
    // inputs; a generated one shows the decision of the real frame it shows,
    // or has no mask (unavailable). Only a real Present whose accepted inputs
    // are all offered adopts them.
    const auto arbitration = d.temporal.arbitrate({candidates.hold_previous}, observation, bits);
    // The layer's stored flags are pushed with its detection; nothing on the
    // CPU reads them back.
    const uint32_t flags = (bits & candidate::layer) ? candidates.layer_flags : 0u;
    if (arbitration.adopt) d.temporal.adopt(bits, accepted);
    d.difference_threshold = pair_threshold ? *pair_threshold : ui_selection::comparable(presented, presented).value_or(2.f / 255.f);
    // H1 (d): the offscreen UI layer's pair threshold with the presented
    // color, from the layer signature's format; zero (no pre-UI pixel counts)
    // without a layer or when the two are not comparable. detect_ui pushes it
    // on sample frames only.
    d.pre_ui_threshold = (bits & candidate::layer) ?
      ui_selection::comparable({signatures.format[std::size_t(kind::ui_layer)], d.color}, presented).value_or(0.f) : 0.f;
    const bool needs_detection = auto_mode || manual_detection || ui.kind == ui_input_kind::hudless_difference;
    const bool detection_requested = needs_detection && (!automatic || mode != alpha_auto_state::manual_off);
    // A real Present that offers nothing runs detection only for the T1 grace
    // of a missing accepted candidate (the reduce and mask passes only).
    const bool grace_only = arbitration.detect && !bits &&
      (arbitration.per_frame & ui_detection::per_frame_accepted_missing);
    const bool runs = arbitration.hold || (arbitration.detect && (bits || grace_only));
    const bool fits = d.width <= 3840 && d.height <= 3840;
    d.detection_active = detection_requested && runs && fits && d.prepare_detection();
    // A generated hold and a zero-offer grace frame apply the detected mask
    // even while no UI input is available; where detection can run, a
    // generated Present without a decision to show has no mask at all.
    const bool unavailable = arbitration.kind == ui_temporal::hold_kind::unavailable && detection_requested && fits &&
      d.prepare_detection();
    const bool ui_eligible = (source_alpha_ui || d.detection_active) && !unavailable;
    if (unavailable) alpha_source = {};
    // Manual On without detection (a frame larger than 3840, or resources that
    // could not be prepared) keeps the explicit first filtered candidate.
    const bool explicit_fallback = manual_detection && !d.detection_active && !unavailable &&
      ui.kind != ui_input_kind::unavailable && ui.kind != ui_input_kind::hudless_difference;
    // Why a render that requested detection has none (ui_counter::inactive_*).
    const std::size_t inactive_reason = !runs ? ui_counter::inactive_no_candidates :
      d.width > 3840 || d.height > 3840 ? ui_counter::inactive_size : ui_counter::inactive_unprepared;
    if (channel != ui_mask_channel::alpha && channel != ui_mask_channel::red) return false;
    if (channel == ui_mask_channel::red && (!alpha_source.handle || !d.mask_channel_supported)) return false;
    d.consumed_channel = d.detection_active ? ui_mask_channel::red : channel;
    const bool ui_mode_supported = (plane.mode != ui_plane_mode::depth_midpoint_nearest_ui || d.nearest_ui_supported) &&
      (plane.mode != ui_plane_mode::front_limit || d.front_limit_ui_supported) &&
      (plane.mode != ui_plane_mode::shallow_front || d.shallow_front_ui_supported) &&
      (plane.mode != ui_plane_mode::display_fraction || d.display_fraction_ui_supported);
    if (ui_eligible && !ui_mode_supported && !automatic) return false;
    if (ui_eligible && ui_mode_supported && plane.mode == ui_plane_mode::depth_midpoint_nearest_ui &&
        d.width <= 3840 && d.height <= 3840 && !d.prepare_nearest_ui()) return false;
    com<ID3DDeviceContextState> previous;
    const bool isolated = d.context11.p && !d.frame_state;
    if (isolated) d.context11->SwapDeviceContextState(d.isolated11.p, previous.put());
    struct restore { impl &d; ID3DDeviceContextState *previous; bool isolated; ~restore() { if (isolated) d.context11->SwapDeviceContextState(previous, nullptr); } } restore_state{d, previous.p, isolated};
    d.pending = true;
    const auto &t = d.textures;
    if (diagnostics::enabled()) d.collect_profile();
    d.mark(cmd, impl::mark_render);
    cmd->barrier(backbuffer, api::resource_usage::present, api::resource_usage::copy_source);
    cmd->barrier(t[impl::source].resource, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
    cmd->copy_resource(backbuffer, t[impl::source].resource);
    cmd->barrier(t[impl::source].resource, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
    cmd->barrier(backbuffer, api::resource_usage::copy_source, api::resource_usage::present);
    d.mark(cmd, impl::mark_source);
    render_parameters p = parameters;
    if (!depth.handle) { depth = t[impl::empty_depth].srv; p.depth_ready = p.camera_ready = 0; }
    d.consumed = p;
    d.consumed_plane = plane;
    d.consumed_detection = {};
    d.scene_bits = 0;
    d.scene_layer_proven = false;
    if (detection_requested) ++d.cpu_counts[ui_counter::auto_frames];
    if (d.detection_active) {
      if (!observation.now_ms) observation.now_ms = observation.tick_ms = GetTickCount64();
      // H1 (d): the offered layer's signature is proven the pre-UI scene image
      // by the session's acceptance ledger (never cleared by an identity
      // change, FG toggles or acceptance changes); read after any poll, so a
      // sample that just earned it counts at once.
      const auto layer_proven = [&] {
        return (bits & candidate::layer) && automatic && automatic->session &&
          automatic->session->pre_ui_proven(candidates.signatures.of(kind::ui_layer));
      };
      if (arbitration.hold) {
        // A generated Present: the detected mask as the real frame it shows
        // left it; no detection, poll or sample. Its scene guard bits are
        // reported, never pushed.
        d.scene_layer_proven = layer_proven();
        d.scene_bits = d.guard.per_frame(observation.now_ms, bits, candidates.signatures.by_kind(), d.scene_layer_proven);
        ++d.cpu_counts[ui_counter::held_generated];
        d.temporal.held();
        d.consumed_detection = d.detection_run;
        d.consumed_detection.state = ui_detection_snapshot::run_state::held;
        d.consumed_detection.held_presents = d.temporal.holds;
      } else {
        d.temporal.enter_scope(observation);
        d.guard.enter_scope(observation.epoch, observation.viewport);
        d.poll_detection(observation);
        // Per-frame bits: the scene guard's held verdicts and refuted
        // candidates (H1), T1's hold reset or accepted-missing bit, and depth
        // that is not this frame's. The sample just read may hold or release.
        d.scene_layer_proven = layer_proven();
        d.scene_bits = d.guard.per_frame(observation.now_ms, bits, candidates.signatures.by_kind(), d.scene_layer_proven);
        // T1's re-offer bit: the provider offered the previous render's
        // inexact HUD-less snapshot again (ui_detection::per_frame_reoffer).
        // A tag beside it never clears the bit: an accepted, valid tag
        // decides the frame on its own, and an unaccepted or invalid one
        // cannot, so the held decision serves that frame better than the
        // one-shot grace.
        const bool reoffer = candidates.hudless_reoffer && (bits & candidate::hudless) && !(bits & candidate::exact);
        // A2: the offered declared tags outside the exact pair's tag batch,
        // which the one-way test does not judge
        // (ui_detection::per_frame_unaligned_shift).
        const uint32_t unaligned = (bits & candidate::exact) ?
          candidates.unaligned_declared & bits & (candidate::ui_alpha | candidate::ui_color) : 0u;
        const uint32_t per_frame = d.scene_bits | arbitration.per_frame |
          (!input.depth_current ? ui_detection::per_frame_depth_not_current : 0u) |
          (reoffer ? ui_detection::per_frame_reoffer : 0u) | (unaligned << ui_detection::per_frame_unaligned_shift);
        d.detect_ui(cmd, p, candidates, observation, hudless_color, depth, bits, accepted, flags, per_frame);
        d.consumed_detection = d.detection_run;
        d.temporal.detected(observation);
      }
      alpha_source = t[impl::detected_mask].srv;
    } else if (unavailable) {
      // A generated Present without a real frame's decision to show ends the
      // chain without a mask.
      ++d.cpu_counts[ui_counter::held_none];
      d.temporal.unavailable();
    } else {
      if (detection_requested) ++d.cpu_counts[inactive_reason];
      d.temporal.inactive();
    }
    d.retain_color(cmd);
    d.mark(cmd, impl::mark_detection);
    // A captured input's RGB never replaces current eye color. Auto consumes a
    // freshly derived mask; explicit manual/replay inputs retain their meaning.
    d.update_alpha_auto(ui_eligible && ui_mode_supported && (!needs_detection || d.detection_active || explicit_fallback) &&
      (!automatic || !automatic->retained || alpha_source.handle), automatic);
    const bool probe_ui = d.prepare_adaptive_frame(p, adaptive) && input.adaptive_probe;
    d.consumed_ui_source = d.source_alpha_ui && alpha_source.handle ? d.resource_of(alpha_source) : api::resource{};
    // The conditioning inputs, resolved now (C2). The views stay valid for
    // this Present: the source copy and the detected mask until the next
    // render, captured candidates until the next acquisition, the borrowed
    // depth for the render lease.
    impl::conditioning_input conditioning;
    conditioning.p = p;
    conditioning.depth = depth;
    conditioning.ui_alpha = d.consumed_ui_source.handle ? alpha_source : t[impl::source].srv;
    conditioning.plane = d.consumed_plane;
    conditioning.channel = d.consumed_channel;
    conditioning.apply_ui = d.source_alpha_ui;
    conditioning.probe = probe_ui;
    conditioning.armed = input.diagnostic_armed;
    conditioning.depth_identity = input.depth_identity;
    // A probe frame reads the unpinned field now, an older two-pass shader
    // renders its eyes now, and an immediate pack follows at once.
    if (probe_ui || !d.packed_eyes || !defer_pack) d.record_conditioning(cmd, conditioning);
    else {
      d.owed = conditioning;
      d.conditioning_owed = true;
    }
    if (!d.packed_eyes) {
      d.draw(cmd, impl::eyes, d.width, {impl::left, impl::right}, p, {t[impl::source].srv, depth, t[impl::linear].srv, {}, {}, t[impl::field].srv});
      d.mark(cmd, impl::mark_eyes);
    }
    // Without a consumer, a free slot or a dump, no eye is rendered at all.
    d.pack_depth = depth;
    d.pack_parameters = p;
    d.pack_owed = true;
    if (!defer_pack)
      d.record_pack(cmd, t[d.packed_texture()].resource, t[d.packed_texture()].rtv, api::resource_usage::shader_resource);
    return true;
  }
  bool renderer::pack(api::command_list *cmd, api::resource export_target, std::uint64_t export_generation) {
    if (!data_ || !data_->pack_owed) return false;
    auto &d = *data_;
    if (d.pq_output && !d.textures[impl::packed_pq].resource.handle) return false;
    auto target = d.textures[d.packed_texture()].resource;
    auto view = d.textures[d.packed_texture()].rtv;
    auto resting = api::resource_usage::shader_resource;
    if (!export_target.handle && !d.export_views.empty()) {
      // Views keep D3D11 slot textures alive; hold them only while exporting.
      for (auto &[resource, old] : d.export_views) d.device->destroy_resource_view(old);
      d.export_views.clear();
      d.export_generation = 0;
    }
    if (export_target.handle) {
      if (d.export_generation != export_generation) {
        // RTV descriptors are consumed when commands are recorded, so another
        // generation's views can be destroyed without waiting for the GPU.
        for (auto &[resource, old] : d.export_views) d.device->destroy_resource_view(old);
        d.export_views.clear();
        d.export_generation = export_generation;
      }
      view = {};
      for (const auto &[resource, cached] : d.export_views) if (resource == export_target) view = cached;
      if (!view.handle) {
        if (!d.device->create_resource_view(export_target, api::resource_usage::render_target,
            api::resource_view_desc(d.packed_format()), &view)) return false;
        d.export_views.emplace_back(export_target, view);
      }
      target = export_target;
      resting = api::resource_usage::general; // Shared slots rest in COMMON between owners.
    }
    com<ID3DDeviceContextState> previous;
    const bool isolated = d.context11.p && !d.frame_state;
    if (isolated) d.context11->SwapDeviceContextState(d.isolated11.p, previous.put());
    d.record_pack(cmd, target, view, resting);
    if (isolated) d.context11->SwapDeviceContextState(previous.p, nullptr);
    return true;
  }
  void renderer::release_export_view(api::resource export_target) {
    for (auto *value : {data_.get(), cached_.get()}) {
      if (!value) continue;
      auto &views = value->export_views;
      for (auto it = views.begin(); it != views.end();) {
        if (it->first == export_target) {
          value->device->destroy_resource_view(it->second);
          it = views.erase(it);
        } else ++it;
      }
    }
  }
  api::resource renderer::output() const { return data_ ? data_->textures[data_->packed_texture()].resource : api::resource{}; }
  bool renderer::set_pq_output(bool pq) {
    if (!data_) return !pq;
    auto &d = *data_;
    if (pq && !pq_output_supported()) pq = false;
    if (pq && !d.textures[impl::packed_pq].resource.handle &&
        !d.texture_create(impl::packed_pq, d.width * 2, d.height, api::format::r10g10b10a2_unorm,
          api::resource_usage::render_target | api::resource_usage::copy_source)) {
      auto &t = d.textures[impl::packed_pq];
      if (t.srv.handle) d.device->destroy_resource_view(t.srv);
      if (t.rtv.handle) d.device->destroy_resource_view(t.rtv);
      if (t.resource.handle) d.device->destroy_resource(t.resource);
      t = {};
      pq = false;
    }
    if (d.pq_output != pq && !d.export_views.empty()) {
      // Export views carry the transfer's format; RTV descriptors are consumed
      // at record time, so they go without waiting.
      for (auto &[resource, old] : d.export_views) d.device->destroy_resource_view(old);
      d.export_views.clear();
      d.export_generation = 0;
    }
    d.pq_output = pq;
    return pq;
  }
  bool renderer::pq_output() const { return data_ && data_->pq_output; }
  bool renderer::pq_output_supported() const {
    return data_ && ((data_->color == 3 && data_->pq_per_tap) || (data_->color == 2 && data_->scrgb_pq_pack)) &&
      data_->pipelines[impl::pack_pq].handle;
  }
  api::fence renderer::completion_fence() const { return data_ ? data_->completion : api::fence{}; }
  std::uint64_t renderer::completion_value() const { return data_ ? data_->sequence + 1 : 0; }
  api::resource_view renderer::bind_ui_candidate(unsigned slot, std::uint64_t capture_id, api::resource_view view,
      api::resource resource) {
    if (!data_ || data_->failed || slot >= data_->direct_views.size() || !capture_id || !view.handle || !resource.handle)
      return {};
    data_->direct_views[slot] = {view, resource};
    return view;
  }
  api::resource_view renderer::bind_ui_snapshot(unsigned slot, std::uint64_t capture_id, api::resource resource,
      std::uint32_t view_format) {
    if (!data_ || data_->failed || slot >= data_->direct_views.size() || !capture_id || !resource.handle || !view_format ||
        data_->device->get_api() != api::device_api::d3d12) return {};
    auto &d = *data_;
    auto *native = reinterpret_cast<ID3D12Device *>(d.device->get_native());
    if (!d.snapshot_views.p) {
      // Original-device CPU descriptors: push_descriptors copies them as they
      // are, which a descriptor of a heap ReShade wraps would not survive.
      D3D12_DESCRIPTOR_HEAP_DESC heap{};
      heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      heap.NumDescriptors = UINT(d.direct_views.size());
      if (FAILED(native->CreateDescriptorHeap(&heap, IID_PPV_ARGS(d.snapshot_views.put())))) return {};
      d.snapshot_stride = native->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    // Rewritten per Present: descriptors are consumed when commands record.
    auto handle = d.snapshot_views->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += SIZE_T(slot) * d.snapshot_stride;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = static_cast<DXGI_FORMAT>(view_format);
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MipLevels = 1;
    native->CreateShaderResourceView(reinterpret_cast<ID3D12Resource *>(resource.handle), &desc, handle);
    const api::resource_view view{static_cast<std::uint64_t>(handle.ptr)};
    d.direct_views[slot] = {view, resource};
    return view;
  }
  api::resource renderer::ui_candidate(unsigned slot, api::format format) {
    if (!data_ || slot >= data_->ui_candidates.size()) return {};
    auto &d = *data_;
    auto &t = d.ui_candidates[slot];
    format = typed(format);
    if (t.resource.handle && d.ui_candidate_formats[slot] != format) {
      if (!d.idle()) return {};
      d.device->destroy_resource_view(t.srv); d.device->destroy_resource(t.resource);
      t = {}; ui_candidate_captures_[slot] = 0;
    }
    if (!t.resource.handle) {
      const api::resource_desc desc(d.width, d.height, 1, 1, format, 1, api::memory_heap::default_,
        api::resource_usage::shader_resource | api::resource_usage::copy_dest);
      if (!d.device->create_resource(desc, nullptr, api::resource_usage::shader_resource, &t.resource)) return {};
      if (!d.device->create_resource_view(t.resource, api::resource_usage::shader_resource, api::resource_view_desc(format), &t.srv)) {
        d.device->destroy_resource(t.resource); t = {}; return {};
      }
      d.ui_candidate_formats[slot] = format;
    }
    return t.resource;
  }
  api::resource_view renderer::ui_candidate_view(unsigned slot) const {
    return data_ && slot < data_->ui_candidates.size() ? data_->ui_candidates[slot].srv : api::resource_view{};
  }
  render_parameters renderer::consumed_parameters() const { return data_ ? data_->consumed : render_parameters{}; }
  bool renderer::consumed_source_alpha_ui() const { return data_ && data_->source_alpha_ui; }
  ui_mask_channel renderer::consumed_ui_channel() const { return data_ ? data_->consumed_channel : ui_mask_channel::alpha; }
  alpha_auto_decision renderer::consumed_alpha_auto() const { return data_ ? data_->consumed_auto : alpha_auto_decision{}; }
  ui_detection_snapshot renderer::consumed_detection() const { return data_ ? data_->consumed_detection : ui_detection_snapshot{}; }
  alpha_probe_counters renderer::alpha_probe_activity() const {
    return data_ ? alpha_probe_counters{data_->detection_submitted, data_->detection_mapped, data_->scene_evidence_runs} :
      alpha_probe_counters{};
  }
  ui_adaptive::decision renderer::consumed_ui_adaptive() const { return data_ ? data_->consumed_adaptive : ui_adaptive::decision{}; }
  std::uint64_t renderer::ui_probe_submissions() const { return data_ ? data_->adaptive_submitted : 0; }
  ui_plane_parameters renderer::consumed_ui_plane() const { return data_ ? data_->consumed_plane : ui_plane_parameters{}; }
  renderer::conditioning_counters renderer::conditioning_activity() const {
    return data_ ? data_->conditioning_counts : conditioning_counters{};
  }
  diagnostic_resources renderer::diagnostics() const {
    if (!data_) return {};
    const auto &t = data_->textures;
    diagnostic_resources result{t[impl::source].resource, t[impl::linear].resource, t[impl::raw].resource,
      t[impl::vertical_majorant].resource, t[impl::vertical_field].resource,
      t[impl::field].resource, t[data_->packed_texture()].resource, data_->consumed_ui_source,
      data_->nearest_ui_rendered ? t[impl::ui_plane_tiles].resource : api::resource{},
      data_->nearest_ui_rendered ? t[impl::ui_plane_resolved].resource : api::resource{}};
    result.sbs_pq = data_->pq_output;
    return result;
  }
  api::resource_view renderer::native_rtv(api::resource backbuffer) {
    if (!data_) return {};
    for (const auto &entry : data_->backbuffers) if (entry.first == backbuffer) return entry.second;
    api::resource_view view{};
    if (!data_->device->create_resource_view(backbuffer, api::resource_usage::render_target, api::resource_view_desc(data_->source_format), &view)) return {};
    data_->backbuffers.emplace_back(backbuffer, view);
    return view;
  }
  void renderer::begin_present() {
    if (!data_) return;
    ++data_->present_number;
    data_->profile_open = false;
    // Direct bindings last one Present (their leases end with it).
    data_->direct_views = {};
    // A pack owed by an earlier Present is never recorded: its depth lease
    // ended with that Present.
    data_->conditioning_owed = data_->pack_owed = false;
    if (!data_->pending) return;
    // The missed presentation's commands still execute in queue order before
    // any later signal, so that signal conservatively retires them too.
    data_->pending = false;
    data_->unsignaled = true;
  }
  void renderer::begin_gpu_profile(api::command_list *cmd) {
    if (!data_ || data_->failed || !diagnostics::enabled() || !data_->prepare_profile()) return;
    data_->collect_profile();
    data_->mark(cmd, impl::mark_begin);
  }
  bool renderer::take_gpu_timing(gpu_timing &out) {
    if (!data_) return false;
    auto &d = *data_;
    if (!diagnostics::enabled()) {
      // G1: no timestamps while the Diagnostics switch is off. Frames marked
      // before it turned off are read once it is on again.
      out = {};
      out.state = gpu_timing::profile_state::disabled;
      d.profile_window = {};
      d.profile_sum_ms = {};
      return false;
    }
    d.collect_profile();
    out = d.profile_window;
    out.state = d.profile_state;
    for (unsigned s = 0; s != gpu_timing::stage_count; ++s)
      out.mean_ms[s] = out.frames ? d.profile_sum_ms[s] / out.frames : 0.0;
    d.profile_window = {};
    d.profile_sum_ms = {};
    return out.frames != 0;
  }
  void renderer::finish_present() {
    if (data_) data_->profile_open = false;
    if (!data_ || (!data_->pending && !data_->unsignaled)) return;
    auto &d = *data_;
    if (!d.queue->signal(d.completion, ++d.sequence)) d.failed = true;
    else {
      if (d.adaptive_awaiting_signal) {
        d.adaptive_fence = d.sequence;
        d.adaptive_awaiting_signal = false;
      }
      if (d.detection_awaiting_signal) {
        d.detection_fence = d.sequence;
        d.detection_awaiting_signal = false;
      }
    }
    d.pending = d.unsignaled = false;
  }
  void renderer::begin_frame_state() {
    if (!data_ || data_->frame_state) return;
    data_->frame_state = true;
    if (data_->context11.p) data_->context11->SwapDeviceContextState(data_->isolated11.p, &data_->previous11);
  }
  void renderer::end_frame_state() {
    if (!data_ || !data_->frame_state) return;
    if (data_->context11.p) {
      data_->context11->SwapDeviceContextState(data_->previous11, nullptr);
      if (data_->previous11) data_->previous11->Release();
      data_->previous11 = nullptr;
    }
    data_->frame_state = false;
  }
  void renderer::reset_after_runtime_drain() {
    ui_candidate_captures_.fill(0);
    resume_pending_ = false;
    data_.reset();
    cached_.reset();
  }
  api::device *renderer::device() const { return data_ ? data_->device : cached_ ? cached_->device : nullptr; }
  void renderer::park_after_runtime_drain() {
    ui_candidate_captures_.fill(0);
    preparing_ = false;
    // The swapchain's back buffers are released or recreated by the reset:
    // drop the views of them (a D3D11 view would keep a buffer referenced,
    // failing ResizeBuffers).
    for (auto *value : {data_.get(), cached_.get()}) {
      if (!value) continue;
      for (auto &[resource, view] : value->backbuffers) value->device->destroy_resource_view(view);
      value->backbuffers.clear();
    }
    resume_pending_ = data_ != nullptr;
  }
}

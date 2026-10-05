// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_renderer.h"
#include "game3d_shader_source.h"
#include <reshade.hpp>
#include "async_log.h"
#include "game3d_shader_cache.h"
#include "game3d_still_screen.h"
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
      detection_tiles, detection_reduce, detection_mask, scene_cells, scene_compare, scene_evidence, pass_count };
    std::array<api::pipeline, pass_count> pipelines{};
    std::array<api::sampler, 3> samplers{};
    api::resource_view null_srv{}, null_uav{};
    struct texture { api::resource resource{}; api::resource_view srv{}, uav{}, rtv{}; };
    enum texture_id { source, empty_depth, linear, raw, vertical_majorant, vertical_field, field,
      ui_plane_tiles, ui_plane_resolved, left, right, packed, ui_source, ui_source_second, ui_source_third,
      ui_conflict_statistics, detection_statistics, detection_decision, detected_mask, scene_cell_sums, ui_counter_words,
      detection_hold, scene_previous_luma, retained_color,
      retained_color_last = retained_color + ui_detection_inputs::max_retained_presents - 1, texture_count };
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
    // Timestamps per presentation: begin, render, source, detection,
    // conditioning, eyes, pack. A small ring is read after its fence completes.
    enum profile_mark { mark_begin, mark_render, mark_source, mark_detection, mark_linearize, mark_candidate, mark_vertical,
      mark_conditioning, mark_eyes, mark_pack, mark_count };
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
    std::array<api::format, 3> ui_source_formats{};
    std::array<bool, 3> ui_source_failed{};
    unsigned ui_source_active{};
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
    // (texels 8 and 9), the H1 texel 10 (the opaque Backbuffer and current
    // counts, the claims and the h1 word), texel 11 (the layer's pre-UI
    // pixel counts, H1 d) and texel 12 (H2's still and compared cells). The
    // shader's markers size the decision texels and statistics rows
    // (docs/reshade-sbs.md, UI detection flags and decision texels).
    uint32_t detection_decision_texels = ui_detection::default_decision_texels;
    uint32_t detection_statistics_images = ui_detection::default_scene_evidence_images;
    // Hidden-scene evidence (docs/reshade-sbs.md, hidden-scene evidence): the
    // shader writes decision texels 5, 6 and 12 with its evidence passes.
    // They run on sample frames only, and only when the scene guard measures
    // (scene_guard::state::measure: an acting-capable claim in the latest
    // sample, a held verdict, the first-run shadow, the whole-frame
    // diagnostic, or H2 in its scope while no source other than 11
    // decided).
    bool scene_evidence_supported() const {
      return detection_statistics_images == ui_detection::max_scene_evidence_images &&
        detection_decision_texels >= ui_detection::scene_decision_texels;
    }
    uint64_t scene_evidence_runs{};
    // This render's scene guard bits (per_frame_scene_hidden,
    // per_frame_pre_ui_visible, the refuted candidates and
    // per_frame_pre_ui_proven), pushed with its detection, and whether its
    // offered layer's signature is proven the pre-UI scene image (H1 d, the
    // session ledger's pre-UI proof).
    uint32_t scene_bits{};
    bool scene_shadow{}, scene_layer_proven{};
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
    // The pending sample ran the evidence passes for the guard (measure
    // actionable), not for the first-run shadow or diagnostic alone.
    bool detection_pending_actionable{};
    bool hold_cleared{};
    // H2 (game3d_still_screen.h): this render is in its scope (SDR Auto) and
    // the session enables flattening (UIFlattenStillScreens); the pending
    // sample was submitted in scope, and its evidence passes ran for H2 alone
    // (kept from the acceptance ledger, ui_temporal::ledger_evidence); the
    // previous presented-luma cell means (textures[scene_previous_luma], u5
    // of the compare pass only) were cleared to none; and the current
    // episode pushed still::flatten.
    bool still_scope{}, still_enabled{}, detection_pending_still_scope{}, detection_pending_h2_only{},
      previous_luma_cleared{}, still_flattened{};
    // H2: the longest run that ended before it entered since the last
    // status (update_alpha_auto publishes it once as the status's
    // evidence.still_short_ms), whatever ended it: a sample, a scope or
    // identity change or an unmeasured render, so that the UI line's
    // short_max_ms covers every run still.short counts.
    uint64_t still_short_unpublished_ms{};
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
    // S3: the stamp buffer (renderer::ui_stamps), created on first need, and
    // its D3D12 state within the current submission: COMMON at its start (a
    // buffer decays after every ExecuteCommandLists, and finish_present's
    // signal flushes the immediate list), copy_dest once a candidate copy
    // wrote an entry (note_ui_stamp_write) and after every detection that
    // read it (detect_ui returns it there, so later copies need no barrier).
    texture stamps{};
    bool stamps_attempted{}, stamps_copy_dest{};
    // S3: the shader verifies identity (ui_detection::identity::marker): it
    // reads the stamps at t9, b2 words 6-9 and writes decision texel 12 .z/.w
    // and the identity counter words.
    bool identity_supported{};
    // S3: the GPU identity verdict totals the committed samples carry
    // (ui_ticket::identity_counter::gpu_*, renderer::identity_counts), and
    // the identity counter words at the last commit.
    ui_ticket::identity_counters identity_gpu;
    std::array<uint32_t, ui_counter_word::identity_words> committed_identity_words{};
    // The identity verdict of the last committed sample (decision texel 12).
    ui_selection::identity_verdict sampled_identity{};
    // The status key (ui_temporal::detection_state::status_key) when the
    // pending sample was submitted; the sample describes frames of that winner.
    uint32_t detection_pending_status_key{};
    // The acceptance signatures the pending sample was submitted with; its
    // read earns or revokes acceptance for exactly these (A1).
    candidate_signatures detection_pending_signatures;
    // The Sunshine_UIDetectionFlags the pending sample was pushed with (its
    // offered layer's late-copy bit, A2/E2).
    uint32_t detection_pending_flags{};
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
      if (stamps.srv.handle) device->destroy_resource_view(stamps.srv);
      if (stamps.resource.handle) device->destroy_resource(stamps.resource);
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
      api::format formats[]{id == pack && color == 1 ? api::format::r10g10b10a2_unorm : api::format::r16g16b16a16_float, api::format::r16g16b16a16_float};
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
      pin_lines = shader_marker(shader_source(), "SUNSHINE_UI_PIN_LINE_GROUPS");
      const auto texels = shader_marker(shader_source(), ui_detection::decision_texels_marker);
      const auto images = shader_marker(shader_source(), ui_detection::scene_evidence_images_marker);
      detection_decision_texels = texels ? texels : ui_detection::default_decision_texels;
      // An out-of-range marker leaves automatic detection unavailable (prepare_detection).
      detection_statistics_images = images;
      counter_words = shader_marker(shader_source(), "SUNSHINE_UI_COUNTER_WORDS") == ui_counter_word::count ?
        uint32_t(ui_counter_word::count) : 0u;
      // S3: a shader that verifies identity appends its identity counter words.
      identity_supported = shader_marker(shader_source(), ui_detection::identity::marker) == ui_detection::identity::version &&
        shader_marker(shader_source(), "SUNSHINE_UI_COUNTER_IDENTITY_WORDS") == ui_counter_word::identity_words;
      if (counter_words && identity_supported) counter_words = uint32_t(ui_counter_word::with_identity);
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
        api::constant_range{0, 2, 0, ui_detection::identity::b2_words, api::shader_stage::compute}};
      if (!device->create_pipeline_layout(uint32_t(std::size(params)), params, &layout)) return false;
      for (unsigned i = 0; i < samplers.size(); ++i) {
        api::sampler_desc sampler;
        sampler.filter = i == 1 ? api::filter_mode::min_mag_linear_mip_point : api::filter_mode::min_mag_mip_point;
        if (i == 2) { sampler.address_u = sampler.address_v = sampler.address_w = api::texture_address_mode::border; std::memset(sampler.border_color, 0, sizeof(sampler.border_color)); }
        if (!device->create_sampler(sampler, &samplers[i])) return false;
      }
      if (!texture_create(source, width, height, source_format, api::resource_usage::copy_dest) ||
          !texture_create(empty_depth, 1, 1, api::format::r32_float, api::resource_usage::undefined)) return false;
      if (color == 3 && !texture_create(linear, width, height, api::format::r16g16b16a16_float, api::resource_usage::render_target)) return false;
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
      if (color == 3 && !pipeline_create(pq, "SunshinePreparePQPS", false, vs)) return false;
      if (width <= 3840 && height <= 3840)
        if (!pipeline_create(candidate, "SunshineHostCandidateCS", true, vs) || !pipeline_create(vertical, "SunshineHostVerticalCS", true, vs) ||
            !pipeline_create(horizontal, "SunshineHostHorizontalCS", true, vs) ||
            (limiter_lines && !pipeline_create(ui_apply, "SunshineApplyUICS", true, vs))) return false;
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
    api::format packed_format() const { return color == 1 ? api::format::r10g10b10a2_unorm : api::format::r16g16b16a16_float; }
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
        // Packed-eye frames mark their eyes only when the side-by-side target
        // is recorded; without a consumer the frame ends after conditioning.
        if (!has(mark_render) || !has(mark_conditioning)) { ++profile_window.incomplete; continue; }
        const auto span = [&](unsigned from, unsigned to) {
          return has(from) && has(to) && ticks[to] >= ticks[from] ? double(ticks[to] - ticks[from]) / profile_ticks_per_ms : 0.0;
        };
        const unsigned first = has(mark_begin) ? mark_begin : mark_render,
          last = has(mark_pack) ? mark_pack : has(mark_eyes) ? mark_eyes : mark_conditioning;
        const std::array<double, gpu_timing::stage_count> ms{span(mark_begin, mark_render), span(mark_render, mark_source),
          span(mark_source, mark_detection), span(mark_detection, mark_linearize), span(mark_linearize, mark_candidate),
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
      if (!profile_ready) return;
      if (!profile_open) {
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
    // The final side-by-side pass. Its target rests in `resting` between owners.
    void record_pack(api::command_list *cmd, api::resource target, api::resource_view view, api::resource_usage resting) {
      cmd->barrier(target, resting, api::resource_usage::render_target);
      cmd->bind_pipeline(api::pipeline_stage::all_graphics, pipelines[pack]);
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
      const auto still_short_ms = std::exchange(still_short_unpublished_ms, uint64_t{});
      if (!automatic || !automatic->session) {
        // Explicit offline/replay input is already a resolved choice. An
        // incomplete live observation never acquires that authority.
        source_alpha_ui = !automatic && eligible;
        consumed_auto.enabled = source_alpha_ui;
        consumed_auto.state = source_alpha_ui ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
        consumed_auto.evidence.still_short_ms = still_short_ms;
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
        consumed_auto.evidence.still_short_ms = still_short_ms;
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
      consumed_auto.scene_shadow = scene_shadow;
      consumed_auto.still = {still_scope, still_enabled, guard.still.state(), guard.still.run_ms()};
      consumed_auto.evidence.still_short_ms = std::max(consumed_auto.evidence.still_short_ms, still_short_ms);
    }
    // H2: logs what a run's observation or end did, records a run that ended
    // before it entered for the status, and counts the ends the guard's
    // sample observations do not (a scope or identity change, an unmeasured
    // render): counted says whether a committed sample counts this one.
    void still_log(const still_screen::observation &o, bool counted) {
      char text[256];
      if (o.entered) {
        const auto &run = guard.still.current();
        std::snprintf(text, sizeof(text),
          "Sunshine UI still screen: %s (UIFlattenStillScreens=%d) after run_ms=%llu samples=%u d=[%.3f,%.3f] "
          "still_min=%.3f", still_enabled ? "flattening" : "would flatten", still_enabled ? 1 : 0,
          static_cast<unsigned long long>(run.duration_ms()), run.samples, run.d_min, run.d_max, run.share_min);
        sunshine_log::message(reshade::log::level::info, text);
        still_flattened = false;
      }
      if (o.ended) {
        const auto &run = o.ended_episode;
        std::snprintf(text, sizeof(text),
          "Sunshine UI still screen: episode ended reason=%s duration_ms=%llu samples=%u d=[%.3f,%.3f] still_min=%.3f "
          "flattened=%d", std::string(still_screen::name(o.reason)).c_str(), static_cast<unsigned long long>(run.duration_ms()),
          run.samples, run.d_min, run.d_max, run.share_min, still_flattened ? 1 : 0);
        sunshine_log::message(reshade::log::level::info, text);
        still_flattened = false;
        if (!counted) ++cpu_counts[ui_counter::still_released];
      }
      if (o.short_run) {
        still_short_unpublished_ms = std::max(still_short_unpublished_ms, o.short_ms);
        if (!counted) ++cpu_counts[ui_counter::still_short];
      }
    }
    // H2: a render in scope that cannot be sampled (no offered candidate)
    // ends the run, so that no frame is flattened by a run nothing measures.
    void still_unmeasured() {
      if (still_scope) still_log(guard.still.leave(still_screen::end_reason::unmeasured), false);
    }
    bool prepare_detection() {
      if (detection_attempted) return detection_ready;
      detection_attempted = true;
      // A shader either measures both images' scene evidence or none. Live
      // detection binds candidate layout 2 (the offscreen UI layer at t7 and the
      // accepted mask in b2 word 2); a shader of another layout, such as an
      // older embedded replay shader, would misread both, so it gets none.
      // Selection revision 7 (decides as revision 5: the T1 grace with its hold store at u5, the
      // one-way judgment and the F1 reason words in texels 8 and 9, H1 with
      // its texel 10, the layer's pre-UI pixel counts in texel 11 from the
      // statistics rows at pre_ui_statistics_row with b2 word 4, and H2 with
      // b2 word 5, the stillness counts of texel 12 from the rows at
      // still_statistics_row and the previous cell means at u5 of the
      // compare pass) is the only one this renderer drives.
      if (shader_source().find("#define SUNSHINE_UI_AUTOMATIC_DETECTION 1") == std::string_view::npos ||
          shader_marker(shader_source(), ui_detection::candidate_layout_marker) != ui_detection::candidate_layout ||
          shader_marker(shader_source(), ui_selection::revision_marker) != ui_selection::revision ||
          detection_decision_texels < ui_detection::pre_ui_decision_texels ||
          detection_decision_texels > ui_detection::max_decision_texels ||
          (detection_statistics_images && !scene_evidence_supported()) ||
          !texture_create(detection_statistics, 16,
            ui_detection::statistics_rows(detection_statistics_images, detection_decision_texels),
            api::format::r32g32b32a32_uint, api::resource_usage::unordered_access) ||
          !texture_create(detection_decision, detection_decision_texels, 1, api::format::r32g32b32a32_uint,
            api::resource_usage::unordered_access | api::resource_usage::copy_source) ||
          !texture_create(detected_mask, width, height, api::format::r32_float, api::resource_usage::unordered_access) ||
          !texture_create(detection_hold, ui_detection::hold::store_texels, 1, api::format::r32_uint,
            api::resource_usage::unordered_access) ||
          detection_decision_texels < ui_detection::still_decision_texels ||
          !pipeline_create(detection_tiles, "SunshineUIDetectionTilesCS", true, {}) ||
          !pipeline_create(detection_reduce, "SunshineUIDetectionReduceCS", true, {}) ||
          !pipeline_create(detection_mask, "SunshineUIDetectionMaskCS", true, {}) ||
          (scene_evidence_supported() && (!texture_create(scene_cell_sums, ui_detection::scene::cells_x,
              ui_detection::scene::cells_y, api::format::r32g32b32a32_uint, api::resource_usage::unordered_access) ||
            !texture_create(scene_previous_luma, ui_detection::scene::cells_x, ui_detection::scene::cells_y,
              api::format::r32_uint, api::resource_usage::unordered_access) ||
            !pipeline_create(scene_cells, "SunshineSceneCellsCS", true, {}) ||
            !pipeline_create(scene_compare, "SunshineSceneCompareCS", true, {}) ||
            !pipeline_create(scene_evidence, "SunshineSceneEvidenceCS", true, {})))) return false;
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
    // The GPU words of the sample just completed (counter_words of them, S3's
    // identity words after count when the shader writes them); false when not
    // readable now (the totals are cumulative, so a later sample catches up).
    bool read_counters(std::array<uint32_t, ui_counter_word::with_identity> &words) {
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
      std::array<uint32_t, ui_counter_word::with_identity> all{};
      if (!counters_pending || !read_counters(all)) return;
      std::array<uint32_t, ui_counter_word::count> words{};
      std::copy_n(all.begin(), words.size(), words.begin());
      // S3: the identity verdict words' change since the last commit joins the
      // gpu group of the "Sunshine UI identity" totals, never ui_counters.
      if (counter_words == ui_counter_word::with_identity) {
        std::array<uint32_t, ui_counter_word::identity_words> identity{};
        std::copy_n(all.begin() + ui_counter_word::count, identity.size(), identity.begin());
        identity_gpu += ui_temporal::identity_gpu_delta(identity, committed_identity_words);
        committed_identity_words = identity;
      }
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
        detection_submitted, scene_evidence_supported(), detection_pending_flags);
      if (identity_supported && counts.size() >= 4u * ui_detection::still_decision_texels)
        sampled_identity = ui_selection::identity_of_words(counts[ui_detection::decision_word::id_verdicts],
          counts[ui_detection::decision_word::id_deltas]);
      temporal.latest_source = detection_pending_source;
      temporal.latest_key = detection_pending_status_key;
      // The scene guard reads the sample's decision words before the
      // acceptance ledger observes it (game3d_scene_guard.h).
      // H2 reads it as in scope only while this render still is: a sample in
      // flight when the render left scope must not start a run.
      const auto observed = guard.observe(scene_guard::sample_of(counts.data(), counts.size(), detection_pending_source.now_ms,
        scene_evidence_supported()), detection_pending_actionable, detection_pending_signatures.by_kind(),
        detection_pending_still_scope && still_scope);
      latest.evidence.shadow_hidden_ms = observed.shadow_hidden_ms;
      latest.evidence.still_short_ms = observed.still.short_ms;
      // H2: the episode log; the committed sample counts its observation.
      still_log(observed.still, true);
      // Earns or revokes acceptance of the signatures this sample was taken
      // for; a session in a manual mode ignores it (S2). Evidence measured for
      // H2 alone never reaches it.
      if (input.session)
        input.session->observe(ui_temporal::ledger_evidence(latest.evidence, detection_pending_h2_only), latest.pixels,
          detection_pending_source.now_ms, detection_pending_signatures);
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
        uint32_t bits, uint32_t accepted, uint32_t flags, uint32_t per_frame, uint32_t still_bits, uint32_t identity_bits) {
      std::array<api::resource_view, 15> views{};
      // A HUD-less image is compared with the color of the frame it belongs to:
      // its batch's tagged Backbuffer or a retained Present. Detection then also
      // reads that color's alpha for the present-alpha candidate; eye rendering
      // stays current.
      views[0] = paired_color.handle ? paired_color : textures[source].srv;
      // The presented color (t6): the tiles pass compares the offscreen UI
      // layer with it (H1 d), the evidence passes measure it.
      views[6] = textures[source].srv;
      // Candidate layout 2: the offscreen UI layer in its own slot (t7).
      views[7] = input.layer;
      for (unsigned i = 0; i != 3; ++i) views[11+i] = input.masks[i];
      views[14] = input.hudless;
      // A sample frame: the CPU reads the decision texels of at most one real
      // frame every 100 ms. Only it pushes the pre-UI threshold, so that the
      // tiles pass and the reduce count the pre-UI pixels (texel 11, read by
      // the CPU only) on sample frames; every other frame pushes zero, which
      // skips the presented-colour loads and the second sums.
      const bool sample = bits && !detection_pending && !(detection_last_submit &&
        observation.now_ms >= detection_last_submit && observation.now_ms - detection_last_submit < 100);
      const float pushed_pre_ui_threshold = sample ? pre_ui_threshold : 0.f;
      // The b2 constants of every detection pass. Per-frame bits join the
      // pushed flags only, never the stored flags.
      uint32_t threshold_bits, pre_ui_threshold_bits;
      std::memcpy(&threshold_bits, &difference_threshold, sizeof(threshold_bits));
      std::memcpy(&pre_ui_threshold_bits, &pushed_pre_ui_threshold, sizeof(pre_ui_threshold_bits));
      detection_run = {ui_detection_snapshot::run_state::ran, bits, threshold_bits, accepted, flags | per_frame, flags};
      detection_run.pre_ui_threshold_bits = pre_ui_threshold_bits;
      detection_run.still_bits = still_bits;
      // S3: b2 words 6-9, the proposals the provider made and the identity
      // bits; the stamps the GPU verifies them against are bound at t9.
      detection_run.expected_layer_present = input.expected_layer_present;
      detection_run.expected_layer_token = input.expected_layer_token;
      detection_run.expected_hudless_present = input.expected_hudless_present;
      detection_run.identity_bits = identity_bits;
      if (still_bits) still_flattened = true;
      // S3: every detection pass reads the stamps at t9 (the shader's
      // SunshineUIStamps). A candidate copy left the buffer in copy_dest this
      // submission, else it is still COMMON; after the passes it returns to
      // copy_dest for the copies that follow in the same submission.
      struct stamp_lease {
        api::command_list *cmd;
        impl *d;
        ~stamp_lease() {
          if (!d) return;
          cmd->barrier(d->stamps.resource, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
          d->stamps_copy_dest = true;
        }
      } stamps_lease{cmd, nullptr};
      if (identity_supported && stamps.srv.handle) {
        cmd->barrier(stamps.resource, stamps_copy_dest ? api::resource_usage::copy_dest : api::resource_usage::general,
          api::resource_usage::shader_resource);
        stamps_lease.d = this;
        views[9] = stamps.srv;
      }
      // The reduce also adds this frame to the exact counters at u7.
      auto &counter_texture = textures[ui_counter_words];
      if (counter_words && !counters_cleared) {
        const uint32_t zero[4]{};
        cmd->barrier(counter_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        cmd->clear_unordered_access_view_uint(counter_texture.uav, zero);
        cmd->barrier(counter_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        counters_cleared = true;
      }
      // The T1 hold store starts at none; every chain's first detection also
      // pushes per_frame_hold_reset.
      auto &hold_texture = textures[detection_hold];
      if (!hold_cleared) {
        const uint32_t zero[4]{};
        cmd->barrier(hold_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        cmd->clear_unordered_access_view_uint(hold_texture.uav, zero);
        cmd->barrier(hold_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        hold_cleared = true;
      }
      // H2's previous cell means start at none (nothing compared).
      auto &previous_texture = textures[scene_previous_luma];
      if (previous_texture.uav.handle && !previous_luma_cleared) {
        const uint32_t zero[4]{};
        cmd->barrier(previous_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        cmd->clear_unordered_access_view_uint(previous_texture.uav, zero);
        cmd->barrier(previous_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        previous_luma_cleared = true;
      }
      // The reduce (reduce true) also binds the hold store at u5, and the
      // scene compare pass (previous true) H2's previous cell means; no other
      // detection pass reads either.
      const auto dispatch_stage = [&](pass stage, texture_id target, unsigned output, unsigned x, unsigned y,
          bool count = false, bool reduce = false, bool previous = false) {
        auto &t = textures[target];
        count = count && counter_words;
        cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        if (count) cmd->barrier(counter_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        if (reduce) cmd->barrier(hold_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        if (previous)
          cmd->barrier(previous_texture.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
        std::array<api::resource_view, 8> uavs{}; uavs[output] = t.uav;
        if (count) uavs[7] = counter_texture.uav;
        if (reduce) uavs[5] = hold_texture.uav;
        if (previous) uavs[5] = previous_texture.uav;
        cmd->bind_pipeline(api::pipeline_stage::compute_shader, pipelines[stage]);
        bindings(cmd, api::shader_stage::compute, p, views, uavs);
        struct constants {
          uint32_t bits;
          float threshold;
          uint32_t accepted, flags;
          float pre_ui_threshold;
          uint32_t still;
          uint32_t expected_layer_present, expected_layer_token, expected_hudless_present, identity;
        } values{detection_run.candidates, difference_threshold, detection_run.accepted, detection_run.flags,
          pushed_pre_ui_threshold, detection_run.still_bits, detection_run.expected_layer_present,
          detection_run.expected_layer_token, detection_run.expected_hudless_present, detection_run.identity_bits};
        static_assert(sizeof(values) == ui_detection::identity::b2_words * sizeof(uint32_t));
        cmd->push_constants(api::shader_stage::compute, layout, 5, 0, ui_detection::identity::b2_words, &values);
        cmd->dispatch(x, y, 1);
        uavs.fill(null_uav);
        cmd->push_descriptors(api::shader_stage::compute, layout, 3,
          {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
        cmd->barrier(t.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        if (count) cmd->barrier(counter_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        if (reduce) cmd->barrier(hold_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
        if (previous)
          cmd->barrier(previous_texture.resource, api::resource_usage::unordered_access, api::resource_usage::shader_resource);
      };
      if (bits) dispatch_stage(detection_tiles, detection_statistics, 6, 16, 16);
      views[10] = textures[detection_statistics].srv;
      dispatch_stage(detection_reduce, detection_decision, 6, 1, 1, true, true);
      views[10] = textures[detection_decision].srv;
      // A reused decision (texel 9 bit 16) leaves detected_mask as the
      // previous real frame made it.
      dispatch_stage(detection_mask, detected_mask, 0, (width+7)/8, (height+7)/8);
      if (!sample) return;
      // A sample frame. Hidden-scene evidence only measures this frame for the
      // CPU and writes decision texels 5 and 6 after the decision and mask;
      // without an acting-capable claim in the latest sample, a held verdict,
      // the first-run shadow or an accepted whole-frame decision in the latest
      // sample nothing runs (scene_guard::state::measure; a proven layer that
      // is the offer's pre-UI image always measures, and H2 measures in its
      // scope unless the latest sample decided a source other than 11). The
      // compare pass also counts H2's still cells. The cells pass reads
      // the pre-UI scene image the b2 constants name: the HUD-less image
      // (t14) when offered, else the offscreen UI layer's colour (t7), both
      // still bound from the detection passes above, beside the presented
      // color (t6).
      const bool proven_image = scene_layer_proven && ui_selection::pre_ui_image_of(bits) == ui_detection::pre_ui_image::layer;
      const bool whole_frame = ui_temporal::whole_frame(temporal.latest);
      const auto measure = guard.measure(observation.now_ms, scene_shadow, whole_frame, proven_image,
        still_scope && guard.still.wants_measure());
      detection_pending_actionable = false;
      // Whether the passes run for H2 alone, whose measurements the
      // acceptance ledger never observes (ui_temporal::ledger_evidence).
      detection_pending_h2_only = measure.run && !guard.measure(observation.now_ms, scene_shadow, whole_frame, proven_image).run;
      if (scene_evidence_supported() && measure.run) {
        namespace scene = ui_detection::scene;
        detection_pending_actionable = measure.actionable;
        views[1] = depth;
        views[10] = {};
        dispatch_stage(scene_cells, scene_cell_sums, 6, scene::cells_x / 16, scene::cells_y);
        views[10] = textures[scene_cell_sums].srv;
        dispatch_stage(scene_compare, detection_statistics, 6, scene::cells_x / 16, scene::cells_y / 16, false, false, true);
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
      detection_pending_flags = detection_run.flags;
      detection_pending_still_scope = still_scope;
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
  }
  std::string_view renderer::shader_source() { return {game3d_shader_source, sizeof(game3d_shader_source) - 1}; }
  std::string_view renderer::active_shader_source() const { return data_ ? data_->shader_source() : shader_source(); }
  // Every entry point a renderer for this configuration may create, eagerly
  // or on first use, so background preparation leaves no compile on Present.
  static std::vector<shader_cache::entry_point> shader_entries(std::string_view source, uint32_t w, uint32_t h, uint32_t c) {
    const auto has = [source](const char *marker) { return source.find(marker) != std::string_view::npos; };
    std::vector<shader_cache::entry_point> entries{{"PostProcessVS", "vs_5_0"}};
    if (c == 3) entries.push_back({"SunshinePreparePQPS", "ps_5_0"});
    if (w <= 3840 && h <= 3840) {
      entries.insert(entries.end(), {{"SunshineHostCandidateCS", "cs_5_0"}, {"SunshineHostVerticalCS", "cs_5_0"},
        {"SunshineHostHorizontalCS", "cs_5_0"}});
      if (shader_marker(source, "SUNSHINE_LIMITER_LINE_GROUPS") || has("#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1"))
        entries.push_back({"SunshineApplyUICS", "cs_5_0"});
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
    if (data_ && data_->device == device && data_->width == desc.texture.width && data_->height == desc.texture.height &&
        data_->source_format == typed(desc.texture.format) && data_->color == c &&
        data_->source_override == source_override) return !data_->failed;
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
    ui_source_capture_ = 0;
    ui_candidate_captures_.fill(0);
    data_.reset();
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
    observation.dedicated_mask = ui.kind == ui_input_kind::dedicated_mask;
    const auto *automatic = ui.automatic ? &observation : nullptr;
    const auto *adaptive = ui.adaptive;
    const auto channel = source_alpha_ui ? ui.channel : ui_mask_channel::alpha;
    if (!data_ || data_->failed || data_->pending) return false;
    auto &d = *data_;
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
      const auto desc = d.device->get_resource_desc(d.device->get_resource_from_view(view));
      return desc.type == api::resource_type::texture_2d && desc.texture.width == d.width && desc.texture.height == d.height &&
        desc.texture.samples == 1 && desc.texture.depth_or_layers == 1;
    };
    const auto format_of = [&](api::resource_view view) {
      return uint32_t(typed(d.device->get_resource_desc(d.device->get_resource_from_view(view)).texture.format));
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
    // S3 (shadow): b2 word 9. The CPU proves a HUD-less pair one tag batch
    // when its image and the bound Backbuffer reference are snapshots of one
    // token (ui_ticket::same_frame); the GPU counts that pair exact in token
    // space. The gate exists only while identity is authoritative (the
    // compile-time switch; the sequence harness's override never reaches a
    // live render): the HUD-less change set then needs its exact pair.
    const bool identity_authoritative = ui_ticket::authoritative(d.temporal.identity_override);
    const bool hudless_offered = (bits & candidate::hudless) != 0;
    const auto &tickets = candidates.tickets;
    const bool token_batch = hudless_offered && candidates.hudless_pair.handle &&
      tickets[ui_ticket::slot::hudless].token.valid && tickets[ui_ticket::slot::backbuffer].token.valid &&
      ui_ticket::same_frame(tickets[ui_ticket::slot::hudless], tickets[ui_ticket::slot::backbuffer]);
    namespace identity_bit = ui_detection::identity;
    const uint32_t identity_bits = (token_batch ? identity_bit::token_batch : 0u) |
      (identity_authoritative ? identity_bit::gate_hudless : 0u);
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
    // S3 shadow: the identity also carries the newest offered token and
    // whether it is new to the last decision (ui_temporal::ticket_identity);
    // arbitrate reads them only while ui_ticket::identity_authoritative.
    const auto identity = ui_temporal::ticket_identity({candidates.hold_previous, candidates.real_frame},
      ui_ticket::newest_token(candidates.tickets), d.temporal.decision_token, bits != 0);
    const auto arbitration = d.temporal.arbitrate(identity, observation, bits);
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
    d.collect_profile();
    if (d.prepare_profile()) d.mark(cmd, impl::mark_render);
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
    d.scene_shadow = d.scene_layer_proven = false;
    // H2 (game3d_still_screen.h): Auto on SDR output only; a render outside
    // that scope ends the run at once.
    d.still_scope = auto_mode && still_screen::sdr_output(d.color, uint32_t(d.source_format));
    d.still_enabled = d.still_scope && automatic->session->still_flatten();
    if (!d.still_scope) d.still_log(d.guard.still.leave(still_screen::end_reason::scope), false);
    if (detection_requested) ++d.cpu_counts[ui_counter::auto_frames];
    if (d.detection_active) {
      if (!observation.now_ms) observation.now_ms = observation.tick_ms = GetTickCount64();
      d.scene_shadow = automatic && automatic->session && automatic->session->first_run();
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
        d.temporal.held(identity);
        d.consumed_detection = d.detection_run;
        d.consumed_detection.state = ui_detection_snapshot::run_state::held;
        d.consumed_detection.held_presents = d.temporal.holds;
      } else {
        d.temporal.enter_scope(observation);
        d.still_log(d.guard.enter_scope(observation.epoch, observation.viewport), false);
        d.poll_detection(observation);
        // Per-frame bits: the scene guard's held verdicts and refuted
        // candidates (H1), T1's hold reset or accepted-missing bit, and depth
        // that is not this frame's. The sample just read may hold or release.
        d.scene_layer_proven = layer_proven();
        d.scene_bits = d.guard.per_frame(observation.now_ms, bits, candidates.signatures.by_kind(), d.scene_layer_proven);
        const uint32_t per_frame = d.scene_bits | arbitration.per_frame |
          (!input.depth_current ? ui_detection::per_frame_depth_not_current : 0u);
        // H2: a still screen without a UI source flattens only while its run
        // is active and the session enables it; the shadow pushes nothing. A
        // zero-offer frame (the T1 grace) is never sampled, so it ends the
        // run instead.
        if (!bits) d.still_unmeasured();
        const uint32_t still_bits = d.still_enabled && d.guard.still_flatten() ? ui_detection::still::flatten : 0u;
        d.detect_ui(cmd, p, candidates, observation, hudless_color, depth, bits, accepted, flags, per_frame, still_bits,
          identity_bits);
        d.consumed_detection = d.detection_run;
        d.temporal.detected(observation, identity);
      }
      alpha_source = t[impl::detected_mask].srv;
    } else if (unavailable) {
      // A generated Present without a real frame's decision to show ends the
      // chain without a mask.
      ++d.cpu_counts[ui_counter::held_none];
      d.temporal.unavailable();
      d.still_unmeasured();
    } else {
      if (detection_requested) ++d.cpu_counts[inactive_reason];
      d.temporal.inactive();
      d.still_unmeasured();
    }
    d.retain_color(cmd);
    d.mark(cmd, impl::mark_detection);
    // A captured input's RGB never replaces current eye color. Auto consumes a
    // freshly derived mask; explicit manual/replay inputs retain their meaning.
    d.update_alpha_auto(ui_eligible && ui_mode_supported && (!needs_detection || d.detection_active || explicit_fallback) &&
      (!automatic || !automatic->retained || alpha_source.handle), automatic);
    const bool probe_ui = d.prepare_adaptive_frame(p, adaptive);
    d.nearest_ui_rendered = false;
    d.consumed_ui_source = d.source_alpha_ui && alpha_source.handle ? d.device->get_resource_from_view(alpha_source) : api::resource{};
    if (d.color == 3) d.draw(cmd, impl::pq, d.width, {impl::linear}, p, {t[impl::source].srv});
    d.mark(cmd, impl::mark_linearize);
    if (d.width <= 3840 && d.height <= 3840) {
      d.dispatch(cmd, impl::candidate, (d.width + 7) / 8, (d.height + 7) / 8, p, {api::resource_view{}, depth}, {impl::raw});
      d.mark(cmd, impl::mark_candidate);
      const uint32_t group_lines = std::max(d.limiter_lines, 1u);
      d.dispatch(cmd, impl::vertical, (d.width + group_lines - 1) / group_lines, 1, p, {api::resource_view{}, {}, {}, t[impl::raw].srv}, {impl::vertical_majorant, impl::vertical_field});
      d.mark(cmd, impl::mark_vertical);
      const auto ui_alpha = d.consumed_ui_source.handle ? alpha_source : t[impl::source].srv;
      if (d.source_alpha_ui && plane.mode == ui_plane_mode::depth_midpoint_nearest_ui) {
        d.dispatch(cmd, impl::ui_tiles, (d.width + 15) / 16, (d.height + 15) / 16, p,
          {ui_alpha, depth}, {impl::ui_plane_tiles});
        d.dispatch(cmd, impl::ui_reduce, 1, 1, p,
          {api::resource_view{}, {}, {}, {}, {}, {}, {}, {}, t[impl::ui_plane_tiles].srv}, {impl::ui_plane_resolved});
        d.nearest_ui_rendered = true;
      }
      // UI passes read the explicit selected mask channel. Eye RGB stays current.
      // UI pinning follows the complete scene field as its own pass, and a
      // probe observes the unpinned field in between. Older embedded replay
      // shaders pin inside the horizontal pass unless a probe needs it apart.
      const std::array<api::resource_view, 15> field_inputs{ui_alpha, {}, {}, {}, t[impl::vertical_field].srv,
        {}, {}, {}, {}, d.nearest_ui_rendered ? t[impl::ui_plane_resolved].srv : api::resource_view{}};
      const bool apply_ui = d.source_alpha_ui, pin_apart = d.limiter_lines || probe_ui;
      if (pin_apart) d.source_alpha_ui = false;
      d.dispatch(cmd, impl::horizontal, (d.height + group_lines - 1) / group_lines, 1, p, field_inputs, {impl::field});
      d.source_alpha_ui = apply_ui;
      if (probe_ui) d.submit_adaptive_probe(cmd, ui_alpha, p);
      const uint32_t pin_lines = std::max(d.pin_lines, 1u);
      if (pin_apart && apply_ui)
        d.dispatch(cmd, impl::ui_apply, (d.height + pin_lines - 1) / pin_lines, 1, p, field_inputs, {impl::field});
    }
    d.mark(cmd, impl::mark_conditioning);
    if (!d.packed_eyes) {
      d.draw(cmd, impl::eyes, d.width, {impl::left, impl::right}, p, {t[impl::source].srv, depth, t[impl::linear].srv, {}, {}, t[impl::field].srv});
      d.mark(cmd, impl::mark_eyes);
    }
    // Without a consumer, a free slot or a dump, no eye is rendered at all.
    d.pack_depth = depth;
    d.pack_parameters = p;
    d.pack_owed = true;
    if (!defer_pack) d.record_pack(cmd, t[impl::packed].resource, t[impl::packed].rtv, api::resource_usage::shader_resource);
    return true;
  }
  bool renderer::pack(api::command_list *cmd, api::resource export_target, std::uint64_t export_generation) {
    if (!data_ || !data_->pack_owed) return false;
    auto &d = *data_;
    auto target = d.textures[impl::packed].resource;
    auto view = d.textures[impl::packed].rtv;
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
  api::resource renderer::output() const { return data_ ? data_->textures[impl::packed].resource : api::resource{}; }
  api::resource renderer::ui_source(api::format format) {
    if (!data_ || data_->failed) return {};
    auto &d = *data_;
    if (format == api::format::unknown) format = d.source_format;
    unsigned selected = 3;
    for (unsigned i = 0; i != 3; ++i)
      if (d.ui_source_formats[i] == format) { selected = i; break; }
    if (selected == 3) {
      for (unsigned i = 0; i != 3; ++i)
        if (d.ui_source_formats[i] == api::format::unknown) { selected = i; break; }
      if (selected == 3) {
        if (!d.idle()) return {}; // Bounded storage; never destroy an in-flight view.
        selected = d.ui_source_active;
        auto &old = d.textures[static_cast<impl::texture_id>(impl::ui_source + selected)];
        if (old.srv.handle) d.device->destroy_resource_view(old.srv);
        if (old.resource.handle) d.device->destroy_resource(old.resource);
        old = {};
      }
      d.ui_source_formats[selected] = format;
      d.ui_source_failed[selected] = false;
      ui_source_capture_ = 0;
    }
    if (d.ui_source_active != selected) ui_source_capture_ = 0;
    d.ui_source_active = selected;
    if (d.ui_source_failed[selected]) return {};
    const auto id = static_cast<impl::texture_id>(impl::ui_source + selected);
    auto &texture = d.textures[id];
    if (!texture.resource.handle && !d.texture_create(id, d.width, d.height, format,
        api::resource_usage::copy_dest | api::resource_usage::copy_source)) {
      d.ui_source_failed[selected] = true;
      return {};
    }
    return texture.resource;
  }
  api::resource_view renderer::ui_source_view() const {
    return data_ && !data_->ui_source_failed[data_->ui_source_active] ?
      data_->textures[static_cast<impl::texture_id>(impl::ui_source + data_->ui_source_active)].srv : api::resource_view{};
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
  api::resource renderer::ui_stamps() {
    if (!data_) return {};
    auto &d = *data_;
    if (d.stamps_attempted) return d.stamps.resource;
    d.stamps_attempted = true;
    // D3D12 buffers live in COMMON: the copies into the entries promote it
    // implicitly, and it decays after each submission.
    constexpr uint64_t entries = ui_ticket::slot::count;
    const api::resource_desc desc(entries * ui_ticket::stamp_bytes, api::memory_heap::default_,
      api::resource_usage::copy_dest | api::resource_usage::shader_resource);
    if (!d.device->create_resource(desc, nullptr, api::resource_usage::general, &d.stamps.resource)) {
      d.stamps.resource = {};
      return {};
    }
    if (!d.device->create_resource_view(d.stamps.resource, api::resource_usage::shader_resource,
          api::resource_view_desc(api::format::r32g32b32a32_uint, 0, entries), &d.stamps.srv)) {
      d.device->destroy_resource(d.stamps.resource);
      d.stamps = {};
    }
    return d.stamps.resource;
  }
  api::resource_view renderer::ui_stamps_view() const { return data_ ? data_->stamps.srv : api::resource_view{}; }
  void renderer::note_ui_stamp_write() {
    if (data_ && data_->stamps.resource.handle) data_->stamps_copy_dest = true;
  }
  ui_ticket::identity_counters renderer::identity_counts() const {
    return data_ ? data_->identity_gpu : ui_ticket::identity_counters{};
  }
  ui_selection::identity_verdict renderer::sampled_identity() const {
    return data_ ? data_->sampled_identity : ui_selection::identity_verdict{};
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
  diagnostic_resources renderer::diagnostics() const {
    if (!data_) return {};
    const auto &t = data_->textures;
    diagnostic_resources result{t[impl::source].resource, t[impl::linear].resource, t[impl::raw].resource,
      t[impl::vertical_majorant].resource, t[impl::vertical_field].resource,
      t[impl::field].resource, t[impl::packed].resource, data_->consumed_ui_source,
      data_->nearest_ui_rendered ? t[impl::ui_plane_tiles].resource : api::resource{},
      data_->nearest_ui_rendered ? t[impl::ui_plane_resolved].resource : api::resource{}};
    result.ui_stamps = data_->stamps.resource;
    result.ui_stamps_copy_dest = data_->stamps_copy_dest;
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
    // S3: the previous Present's submission ended (ReShade flushes its
    // immediate list at every Present), so the stamp buffer is COMMON again
    // before this Present's candidate copies write it.
    data_->stamps_copy_dest = false;
    if (!data_->pending) return;
    // The missed presentation's commands still execute in queue order before
    // any later signal, so that signal conservatively retires them too.
    data_->pending = false;
    data_->unsignaled = true;
  }
  void renderer::begin_gpu_profile(api::command_list *cmd) {
    if (!data_ || data_->failed || !data_->prepare_profile()) return;
    data_->collect_profile();
    data_->mark(cmd, impl::mark_begin);
  }
  bool renderer::take_gpu_timing(gpu_timing &out) {
    if (!data_) return false;
    auto &d = *data_;
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
      // The signal flushed the immediate list: the stamp buffer decays to COMMON.
      d.stamps_copy_dest = false;
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
  void renderer::reset_after_runtime_drain() { ui_source_capture_ = 0; ui_candidate_captures_.fill(0); data_.reset(); }
}

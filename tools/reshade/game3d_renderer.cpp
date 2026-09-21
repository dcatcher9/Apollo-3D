// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_renderer.h"
#include "game3d_shader_source.h"
#include <reshade.hpp>
#include <d3d11_1.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <atomic>
#include <cstring>
#include <string>
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
    std::atomic<uint64_t> next_alpha_observation_source{1};
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
    enum pass { pq, candidate, vertical, ui_tiles, ui_reduce, horizontal, eyes, pack, alpha_coverage, pass_count };
    std::array<api::pipeline, pass_count> pipelines{};
    std::array<api::sampler, 3> samplers{};
    api::resource_view null_srv{}, null_uav{};
    struct texture { api::resource resource{}; api::resource_view srv{}, uav{}, rtv{}; };
    enum texture_id { source, empty_depth, linear, raw, vertical_majorant, vertical_field, field,
      ui_plane_tiles, ui_plane_resolved, left, right, packed, ui_source, alpha_statistics, texture_count };
    std::array<texture, texture_count> textures{};
    std::vector<std::pair<api::resource, api::resource_view>> backbuffers;
    api::fence completion{};
    uint64_t sequence = 0;
    bool pending = false, failed = false;
    com<ID3D11DeviceContext1> context11;
    com<ID3DDeviceContextState> isolated11;
    ID3DDeviceContextState *previous11 = nullptr;
    bool frame_state = false;
    render_parameters consumed;
    ui_plane_parameters consumed_plane;
    bool source_alpha_ui = false;
    api::resource consumed_ui_source{};
    bool ui_source_failed = false;
    bool nearest_ui_supported = false, nearest_ui_rendered = false;
    bool front_limit_ui_supported = false;
    alpha_auto_decision consumed_auto;
    alpha_probe_counters alpha_activity;
    // A process can own several runtimes/renderers, each with its own source
    // sequence. Their continuity and watermarks must never reset each other.
    const uint64_t alpha_observation_source = next_alpha_observation_source.fetch_add(1, std::memory_order_relaxed);
    alpha_auto_source alpha_scope;
    bool alpha_scope_active = false, alpha_probe_attempted = false, alpha_probe_ready = false;
    bool alpha_readback_pending = false, alpha_awaiting_signal = false;
    uint64_t alpha_scope_generation = 0, alpha_pending_generation = 0;
    uint64_t alpha_fence = 0, alpha_last_submit = 0, alpha_last_sequence = 0, alpha_last_tick = 0, alpha_present_serial = 0;
    alpha_coverage_sample alpha_pending_sample;
    com<ID3D11Texture2D> alpha_readback11;
    com<ID3D12Resource> alpha_readback12;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT alpha_footprint{};
    uint64_t alpha_readback_bytes = 0;

    bool idle() const {
      return !pending && !failed && (!completion.handle || device->get_completed_fence_value(completion) >= sequence);
    }
    ~impl() {
      for (auto &[resource, view] : backbuffers) device->destroy_resource_view(view);
      for (auto &t : textures) {
        if (t.srv.handle) device->destroy_resource_view(t.srv);
        if (t.uav.handle) device->destroy_resource_view(t.uav);
        if (t.rtv.handle) device->destroy_resource_view(t.rtv);
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
    bool compile(const char *entry, const char *target, com<ID3DBlob> &code) {
      const auto w = std::to_string(width), h = std::to_string(height), c = std::to_string(color);
      const D3D_SHADER_MACRO defines[]{{"BUFFER_WIDTH", w.c_str()}, {"BUFFER_HEIGHT", h.c_str()}, {"BUFFER_COLOR_SPACE", c.c_str()}, {nullptr, nullptr}};
      com<ID3DBlob> errors;
      const auto source = shader_source();
      const HRESULT result = D3DCompile(source.data(), source.size(),
        "Sunshine Game 3D", defines, nullptr, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, code.put(), errors.put());
      if (FAILED(result) && errors.p) reshade::log::message(reshade::log::level::error, static_cast<const char *>(errors->GetBufferPointer()));
      return SUCCEEDED(result);
    }
    bool pipeline_create(pass id, const char *entry, bool compute, api::shader_desc vs) {
      com<ID3DBlob> code;
      if (!compile(entry, compute ? "cs_5_0" : "ps_5_0", code)) return false;
      api::shader_desc shader{code->GetBufferPointer(), code->GetBufferSize()};
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
        api::descriptor_range{0, 0, 0, 10, api::shader_stage::all, 1, api::descriptor_type::shader_resource_view},
        api::descriptor_range{0, 0, 0, 7, api::shader_stage::compute, 1, api::descriptor_type::unordered_access_view},
        api::constant_range{0, 1, 0, 4, api::shader_stage::compute}};
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
      if (width <= 3840 && height <= 3840 && nearest_ui_supported &&
          (!texture_create(ui_plane_tiles, (width + 15) / 16, (height + 15) / 16, api::format::r32_float, api::resource_usage::unordered_access) ||
           !texture_create(ui_plane_resolved, 1, 1, api::format::r32_float, api::resource_usage::unordered_access))) return false;
      for (auto id : {left, right})
        if (!texture_create(id, width, height, api::format::r16g16b16a16_float, api::resource_usage::render_target)) return false;
      if (!texture_create(packed, width * 2, height, color == 1 ? api::format::r10g10b10a2_unorm : api::format::r16g16b16a16_float,
          api::resource_usage::render_target | api::resource_usage::copy_source)) return false;
      com<ID3DBlob> vs_code;
      if (!compile("PostProcessVS", "vs_5_0", vs_code)) return false;
      const api::shader_desc vs{vs_code->GetBufferPointer(), vs_code->GetBufferSize()};
      if (color == 3 && !pipeline_create(pq, "SunshinePreparePQPS", false, vs)) return false;
      if (width <= 3840 && height <= 3840)
        if (!pipeline_create(candidate, "SunshineHostCandidateCS", true, vs) || !pipeline_create(vertical, "SunshineHostVerticalCS", true, vs) ||
            !pipeline_create(horizontal, "SunshineHostHorizontalCS", true, vs)) return false;
      if (width <= 3840 && height <= 3840 && nearest_ui_supported &&
          (!pipeline_create(ui_tiles, "SunshineUINearestTilesCS", true, vs) ||
           !pipeline_create(ui_reduce, "SunshineUINearestReduceCS", true, vs))) return false;
      return pipeline_create(eyes, "SunshineRenderEyesPS", false, vs) && pipeline_create(pack, "SunshinePackEyesPS", false, vs);
    }
    void bindings(api::command_list *cmd, api::shader_stage stage, const render_parameters &p,
        std::array<api::resource_view, 10> srvs, std::array<api::resource_view, 7> uavs = {}) {
      // D3D12 needs actual null descriptors; a zero CPU descriptor handle is
      // not a valid CopyDescriptors source. D3D11 uses zero COM views normally.
      for (auto &view : srvs) if (!view.handle) view = null_srv;
      for (auto &view : uavs) if (!view.handle) view = null_uav;
      cmd->push_constants(stage, layout, 0, 0, sizeof(p) / 4, &p);
      if (stage == api::shader_stage::compute) {
        const auto ui = ui_parameter_words(source_alpha_ui, consumed_plane);
        cmd->push_constants(stage, layout, 4, 0, uint32_t(ui.size()), ui.data());
      }
      cmd->push_descriptors(stage, layout, 1, {{}, 0, 0, uint32_t(samplers.size()), api::descriptor_type::sampler, samplers.data()});
      cmd->push_descriptors(stage, layout, 2, {{}, 0, 0, uint32_t(srvs.size()), api::descriptor_type::shader_resource_view, srvs.data()});
      if (stage == api::shader_stage::compute)
        cmd->push_descriptors(stage, layout, 3, {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
    }
    void draw(api::command_list *cmd, pass id, uint32_t w, std::initializer_list<texture_id> targets,
        const render_parameters &p, const std::array<api::resource_view, 10> &srvs) {
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
        const render_parameters &p, const std::array<api::resource_view, 10> &srvs, std::initializer_list<texture_id> targets) {
      std::array<api::resource_view, 7> uavs{};
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
    bool prepare_alpha_probe() {
      if (alpha_probe_attempted) return alpha_probe_ready;
      alpha_probe_attempted = true;
      if (shader_source().find("#define SUNSHINE_UI_ALPHA_COVERAGE 1") == std::string_view::npos) return false;
      if (!texture_create(alpha_statistics, 16, 16, api::format::r32g32b32a32_uint,
            api::resource_usage::unordered_access | api::resource_usage::copy_source) ||
          !pipeline_create(alpha_coverage, "SunshineAlphaCoverageCS", true, {})) return false;
      if (context11.p) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = desc.Height = 16;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        auto *native = reinterpret_cast<ID3D11Device *>(device->get_native());
        if (FAILED(native->CreateTexture2D(&desc, nullptr, alpha_readback11.put()))) return false;
      } else {
        auto *native = reinterpret_cast<ID3D12Device *>(device->get_native());
        const auto desc = reinterpret_cast<ID3D12Resource *>(textures[alpha_statistics].resource.handle)->GetDesc();
        native->GetCopyableFootprints(&desc, 0, 1, 0, &alpha_footprint, nullptr, nullptr, &alpha_readback_bytes);
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = alpha_readback_bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(alpha_readback12.put())))) return false;
      }
      alpha_probe_ready = true;
      return true;
    }
    void reset_alpha_scope() {
      alpha_scope_active = false;
      ++alpha_scope_generation;
      alpha_last_submit = alpha_last_sequence = alpha_last_tick = 0;
      // An old GPU copy still owns its storage until its recorded fence retires.
    }
    void poll_alpha_probe(uint64_t now, alpha_auto_policy *session) {
      if (!alpha_readback_pending || alpha_awaiting_signal || !alpha_fence) return;
      const auto completed = device->get_completed_fence_value(completion);
      if (completed == UINT64_MAX) {
        alpha_probe_ready = false;
        if (session) session->reset_continuity(alpha_observation_source);
        failed = true;
        return;
      }
      if (completed < alpha_fence) return;
      if (!session || !alpha_scope_active || alpha_pending_generation != alpha_scope_generation) {
        alpha_readback_pending = false;
        return;
      }
      const unsigned char *pixels = nullptr;
      uint32_t pitch = 0;
      D3D11_MAPPED_SUBRESOURCE mapped11{};
      void *mapped12 = nullptr;
      ++alpha_activity.mapped;
      if (context11.p) {
        const auto result = context11->Map(alpha_readback11.p, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped11);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING) return;
        if (SUCCEEDED(result)) { pixels = static_cast<const unsigned char *>(mapped11.pData); pitch = mapped11.RowPitch; }
      } else {
        const D3D12_RANGE range{0, SIZE_T(alpha_readback_bytes)};
        if (SUCCEEDED(alpha_readback12->Map(0, &range, &mapped12))) {
          pixels = static_cast<const unsigned char *>(mapped12) + alpha_footprint.Offset;
          pitch = alpha_footprint.Footprint.RowPitch;
        }
      }
      uint64_t covered = 0, count = 0, invalid = 0;
      const bool readable = pixels && pitch >= 16 * 4 * sizeof(uint32_t);
      if (readable) for (unsigned y = 0; y < 16; ++y) for (unsigned x = 0; x < 16; ++x) {
        std::array<uint32_t, 4> tile{};
        std::memcpy(tile.data(), pixels + size_t(y) * pitch + x * sizeof(tile), sizeof(tile));
        covered += tile[0]; count += tile[1]; invalid += tile[2];
      }
      if (mapped11.pData) context11->Unmap(alpha_readback11.p, 0);
      if (mapped12) { const D3D12_RANGE written{0, 0}; alpha_readback12->Unmap(0, &written); }
      alpha_readback_pending = false;
      if (!readable || count != uint64_t(width) * height || covered > count || invalid > count - covered) {
        // A failed observation disables protection, never stereo rendering.
        alpha_probe_ready = false;
        session->reset_continuity(alpha_observation_source);
        return;
      }
      auto sample = alpha_pending_sample;
      sample.covered = uint32_t(covered); sample.pixels = uint32_t(count); sample.invalid = uint32_t(invalid);
      session->observe(sample, now, alpha_observation_source);
    }
    bool submit_alpha_probe(api::command_list *cmd, api::resource_view selected,
        const render_parameters &p, const alpha_auto_source &input) {
      if (alpha_readback_pending || !consumed_auto.probe_interval_ms || !input.sequence || !input.tick_ms || input.tick_ms > input.now_ms ||
          input.now_ms - input.tick_ms > alpha_observation_max_age_ms ||
          input.sequence <= alpha_last_sequence || input.tick_ms < alpha_last_tick ||
          (alpha_last_submit && input.now_ms >= alpha_last_submit &&
            input.now_ms - alpha_last_submit < consumed_auto.probe_interval_ms)) return false;
      // A game can spend its first seconds creating graphics resources. Start
      // the one-shot window only when an eligible observation can be recorded.
      input.session->begin_observation(input.now_ms);
      consumed_auto = input.session->decision(input.now_ms);
      if (!consumed_auto.monitoring) return false;
      auto &t = textures[alpha_statistics];
      cmd->barrier(t.resource, api::resource_usage::shader_resource, api::resource_usage::unordered_access);
      std::array<api::resource_view, 7> uavs{}; uavs[6] = t.uav;
      cmd->bind_pipeline(api::pipeline_stage::compute_shader, pipelines[alpha_coverage]);
      bindings(cmd, api::shader_stage::compute, p, {selected}, uavs);
      cmd->dispatch(16, 16, 1);
      uavs.fill(null_uav);
      cmd->push_descriptors(api::shader_stage::compute, layout, 3,
        {{}, 0, 0, uint32_t(uavs.size()), api::descriptor_type::unordered_access_view, uavs.data()});
      cmd->barrier(t.resource, api::resource_usage::unordered_access, api::resource_usage::copy_source);
      if (context11.p) {
        context11->CopyResource(alpha_readback11.p, reinterpret_cast<ID3D11Resource *>(t.resource.handle));
      } else {
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = reinterpret_cast<ID3D12Resource *>(t.resource.handle);
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = alpha_readback12.p;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = alpha_footprint;
        reinterpret_cast<ID3D12GraphicsCommandList *>(cmd->get_native())->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      }
      cmd->barrier(t.resource, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      alpha_pending_sample = {};
      alpha_pending_sample.sequence = input.sequence;
      alpha_pending_sample.tick_ms = input.tick_ms;
      alpha_pending_generation = alpha_scope_generation;
      alpha_last_sequence = input.sequence; alpha_last_tick = input.tick_ms; alpha_last_submit = input.now_ms;
      alpha_fence = 0; alpha_readback_pending = alpha_awaiting_signal = true;
      ++alpha_activity.submitted;
      return true;
    }
    void update_alpha_auto(api::command_list *cmd, api::resource_view selected,
        const render_parameters &p, bool eligible, const alpha_auto_source *automatic) {
      consumed_auto = {};
      if (!automatic || !automatic->session) {
        if (alpha_scope_active) reset_alpha_scope();
        poll_alpha_probe(automatic ? automatic->now_ms : 0, nullptr);
        source_alpha_ui = !automatic && eligible;
        consumed_auto.enabled = source_alpha_ui;
        consumed_auto.state = source_alpha_ui ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
        consumed_auto.monitoring = false;
        return;
      }
      auto input = *automatic;
      auto &session = *input.session;
      // The process-owned session waits for its first eligible observation,
      // then keeps that original deadline across renderer recreation.
      consumed_auto = session.decision(input.now_ms);
      source_alpha_ui = eligible && consumed_auto.enabled;
      if (!consumed_auto.monitoring || !eligible) {
        if (alpha_scope_active) {
          if (consumed_auto.monitoring) session.reset_continuity(alpha_observation_source);
          reset_alpha_scope();
        }
        // Retire an already recorded copy without mapping it. No observation,
        // new resource allocation or dispatch occurs after detection/manual.
        poll_alpha_probe(input.now_ms, nullptr);
        return;
      }
      if (!input.retained) {
        if (!input.sequence) input.sequence = ++alpha_present_serial;
        else if (input.sequence > alpha_present_serial) alpha_present_serial = input.sequence;
        if (!input.tick_ms) input.tick_ms = input.now_ms;
      }
      if (!alpha_scope_active || alpha_scope.session != input.session ||
          alpha_scope.retained != input.retained || alpha_scope.epoch != input.epoch ||
          alpha_scope.revision != input.revision || alpha_scope.viewport != input.viewport ||
          input.now_ms < alpha_last_submit) {
        reset_alpha_scope();
        session.reset_continuity(alpha_observation_source);
        alpha_scope = input;
        alpha_scope_active = true;
      }
      poll_alpha_probe(input.now_ms, &session);
      consumed_auto = session.decision(input.now_ms);
      if (consumed_auto.monitoring && prepare_alpha_probe() && submit_alpha_probe(cmd, selected, p, input))
        consumed_auto = session.decision(input.now_ms);
      source_alpha_ui = eligible && consumed_auto.enabled;
    }
  };
  renderer::renderer() = default;
  renderer::~renderer() {
    // Explicit hot-unload is rare. Never free a recorded frame or block the game.
    if (data_ && !data_->idle()) data_.release();
  }
  std::string_view renderer::shader_source() { return {game3d_shader_source, sizeof(game3d_shader_source) - 1}; }
  std::string_view renderer::active_shader_source() const { return data_ ? data_->shader_source() : shader_source(); }
  bool renderer::configure(api::effect_runtime *runtime, api::resource backbuffer, api::color_space color, std::string_view source_override) {
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
    ui_source_capture_ = 0;
    data_.reset();
    auto next = std::make_unique<impl>();
    next->source_override.assign(source_override);
    if (!next->initialize(runtime, desc, c)) return false;
    data_ = std::move(next);
    reshade::log::message(reshade::log::level::info, "Sunshine Game 3D: add-on GPU renderer ready (no FX file required)");
    return true;
  }
  bool renderer::render(api::command_list *cmd, api::resource backbuffer, api::resource_view depth, const render_parameters &parameters, bool source_alpha_ui, api::resource_view alpha_source, const ui_plane_parameters &plane, const alpha_auto_source *automatic) {
    if (!data_ || data_->failed || data_->pending) return false;
    auto &d = *data_;
    const bool ui_mode_supported = (plane.mode != ui_plane_mode::depth_midpoint_nearest_ui || d.nearest_ui_supported) &&
      (plane.mode != ui_plane_mode::front_limit || d.front_limit_ui_supported);
    if (source_alpha_ui && !ui_mode_supported && !automatic) return false;
    com<ID3DDeviceContextState> previous;
    const bool isolated = d.context11.p && !d.frame_state;
    if (isolated) d.context11->SwapDeviceContextState(d.isolated11.p, previous.put());
    struct restore { impl &d; ID3DDeviceContextState *previous; bool isolated; ~restore() { if (isolated) d.context11->SwapDeviceContextState(previous, nullptr); } } restore_state{d, previous.p, isolated};
    d.pending = true;
    const auto &t = d.textures;
    cmd->barrier(backbuffer, api::resource_usage::present, api::resource_usage::copy_source);
    cmd->barrier(t[impl::source].resource, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
    cmd->copy_resource(backbuffer, t[impl::source].resource);
    cmd->barrier(t[impl::source].resource, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
    cmd->barrier(backbuffer, api::resource_usage::copy_source, api::resource_usage::present);
    render_parameters p = parameters;
    if (!depth.handle) { depth = t[impl::empty_depth].srv; p.depth_ready = p.camera_ready = 0; }
    d.consumed = p;
    d.consumed_plane = plane;
    const auto selected_alpha = alpha_source.handle ? alpha_source : t[impl::source].srv;
    // Probe the eligible selected input even while the automatic policy has UI
    // protection disabled. A retained input's RGB never enters the eye passes.
    d.update_alpha_auto(cmd, selected_alpha, p, source_alpha_ui && ui_mode_supported &&
      (!automatic || !automatic->retained || alpha_source.handle), automatic);
    d.nearest_ui_rendered = false;
    d.consumed_ui_source = d.source_alpha_ui && alpha_source.handle ? d.device->get_resource_from_view(alpha_source) : api::resource{};
    if (d.color == 3) d.draw(cmd, impl::pq, d.width, {impl::linear}, p, {t[impl::source].srv});
    if (d.width <= 3840 && d.height <= 3840) {
      d.dispatch(cmd, impl::candidate, (d.width + 7) / 8, (d.height + 7) / 8, p, {api::resource_view{}, depth}, {impl::raw});
      d.dispatch(cmd, impl::vertical, d.width, 1, p, {api::resource_view{}, {}, {}, t[impl::raw].srv}, {impl::vertical_majorant, impl::vertical_field});
      const auto ui_alpha = d.consumed_ui_source.handle ? alpha_source : t[impl::source].srv;
      if (d.source_alpha_ui && plane.mode == ui_plane_mode::depth_midpoint_nearest_ui) {
        d.dispatch(cmd, impl::ui_tiles, (d.width + 15) / 16, (d.height + 15) / 16, p,
          {ui_alpha, depth}, {impl::ui_plane_tiles});
        d.dispatch(cmd, impl::ui_reduce, 1, 1, p,
          {api::resource_view{}, {}, {}, {}, {}, {}, {}, {}, t[impl::ui_plane_tiles].srv}, {impl::ui_plane_resolved});
        d.nearest_ui_rendered = true;
      }
      // UI passes read only the selected t0.a. Eye RGB stays the current frame.
      d.dispatch(cmd, impl::horizontal, d.height, 1, p,
        {ui_alpha, {}, {}, {}, t[impl::vertical_field].srv, {}, {}, {}, {},
          d.nearest_ui_rendered ? t[impl::ui_plane_resolved].srv : api::resource_view{}}, {impl::field});
    }
    d.draw(cmd, impl::eyes, d.width, {impl::left, impl::right}, p, {t[impl::source].srv, depth, t[impl::linear].srv, {}, {}, t[impl::field].srv});
    d.draw(cmd, impl::pack, d.width * 2, {impl::packed}, p, {t[impl::source].srv, {}, {}, {}, {}, {}, t[impl::left].srv, t[impl::right].srv});
    return true;
  }
  api::resource renderer::output() const { return data_ ? data_->textures[impl::packed].resource : api::resource{}; }
  api::resource renderer::ui_source() {
    if (!data_ || data_->failed || data_->ui_source_failed) return {};
    auto &d = *data_;
    auto &texture = d.textures[impl::ui_source];
    if (!texture.resource.handle && !d.texture_create(impl::ui_source, d.width, d.height, d.source_format,
        api::resource_usage::copy_dest | api::resource_usage::copy_source)) {
      d.ui_source_failed = true;
      return {};
    }
    return texture.resource;
  }
  api::resource_view renderer::ui_source_view() const {
    return data_ && !data_->ui_source_failed ? data_->textures[impl::ui_source].srv : api::resource_view{};
  }
  render_parameters renderer::consumed_parameters() const { return data_ ? data_->consumed : render_parameters{}; }
  bool renderer::consumed_source_alpha_ui() const { return data_ && data_->source_alpha_ui; }
  alpha_auto_decision renderer::consumed_alpha_auto() const { return data_ ? data_->consumed_auto : alpha_auto_decision{}; }
  alpha_probe_counters renderer::alpha_probe_activity() const { return data_ ? data_->alpha_activity : alpha_probe_counters{}; }
  ui_plane_parameters renderer::consumed_ui_plane() const { return data_ ? data_->consumed_plane : ui_plane_parameters{}; }
  diagnostic_resources renderer::diagnostics() const {
    if (!data_) return {};
    const auto &t = data_->textures;
    return {t[impl::source].resource, t[impl::linear].resource, t[impl::raw].resource,
      t[impl::vertical_majorant].resource, t[impl::vertical_field].resource,
      t[impl::field].resource, t[impl::packed].resource, data_->consumed_ui_source,
      data_->nearest_ui_rendered ? t[impl::ui_plane_tiles].resource : api::resource{},
      data_->nearest_ui_rendered ? t[impl::ui_plane_resolved].resource : api::resource{}};
  }
  api::resource_view renderer::native_rtv(api::resource backbuffer) {
    if (!data_) return {};
    for (const auto &entry : data_->backbuffers) if (entry.first == backbuffer) return entry.second;
    api::resource_view view{};
    if (!data_->device->create_resource_view(backbuffer, api::resource_usage::render_target, api::resource_view_desc(data_->source_format), &view)) return {};
    data_->backbuffers.emplace_back(backbuffer, view);
    return view;
  }
  void renderer::finish_present() {
    if (!data_ || !data_->pending) return;
    auto &d = *data_;
    if (!d.queue->signal(d.completion, ++d.sequence)) d.failed = true;
    else if (d.alpha_awaiting_signal) {
      d.alpha_fence = d.sequence;
      d.alpha_awaiting_signal = false;
    }
    d.pending = false;
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
  void renderer::reset_after_runtime_drain() { ui_source_capture_ = 0; data_.reset(); }
}

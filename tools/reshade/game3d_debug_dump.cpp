// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_debug_dump.h"

#include "game3d_diagnostic_metadata.h"
#include "game3d_ui_capture.h"
#include "scene_gain.h"
#include "src/game3d_debug_protocol.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <reshade.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <wrl/client.h>

#ifndef SUNSHINE_GAME3D_SHADER_SHA256
  #define SUNSHINE_GAME3D_SHADER_SHA256 "unknown"
#endif
#ifndef SUNSHINE_GAME3D_BUILD_REVISION
  #define SUNSHINE_GAME3D_BUILD_REVISION "unknown"
#endif

namespace sunshine_game3d {
  namespace api = reshade::api;
  namespace wire = ::game3d_debug;
  using Microsoft::WRL::ComPtr;

  namespace {
    std::uint64_t load(const std::uint64_t &value) {
      return InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(const_cast<std::uint64_t *>(&value)), 0, 0);
    }

    void store(std::uint64_t &target, std::uint64_t value) {
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&target), static_cast<LONG64>(value));
    }

    struct handle {
      HANDLE value = nullptr;

      ~handle() {
        if (value) {
          CloseHandle(value);
        }
      }
    };

    unsigned pixel_bytes(DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R16G16_FLOAT:
          return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R32G32_FLOAT:
          return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
          return 16;
        default:
          return 0;
      }
    }

    template<class T> std::string parameter_hex(const T &parameters) {
      constexpr char digits[] = "0123456789abcdef";
      const auto *bytes = reinterpret_cast<const unsigned char *>(&parameters);
      std::string encoded(sizeof(parameters) * 2, '0');
      for (size_t i = 0; i < sizeof(parameters); ++i) {
        encoded[i * 2] = digits[bytes[i] >> 4];
        encoded[i * 2 + 1] = digits[bytes[i] & 15];
      }
      return encoded;
    }

    nlohmann::json allocation_json(const api::resource_desc &desc) {
      return {{"api_type", static_cast<unsigned>(desc.type)}, {"width", desc.texture.width}, {"height", desc.texture.height}, {"depth_or_layers", desc.texture.depth_or_layers}, {"levels", desc.texture.levels}, {"samples", desc.texture.samples}, {"api_format", static_cast<unsigned>(desc.texture.format)}, {"heap", static_cast<unsigned>(desc.heap)}, {"usage", static_cast<unsigned>(desc.usage)}, {"flags", static_cast<unsigned>(desc.flags)}};
    }

    nlohmann::json replay_json(const diagnostic_frame &f, const api::resource_desc &source) {
      using json = nlohmann::json;
      const auto layout = [](const char *name, size_t offset, unsigned count, const char *type) {
        return json {{"name", name}, {"byte_offset", offset}, {"count", count}, {"scalar_type", type}};
      };
      const unsigned color = f.color == api::color_space::srgb ? 1 : f.color == api::color_space::scrgb ? 2 :
                                                                                                          3;
      const bool conditioning = source.texture.width <= 3840 && source.texture.height <= 3840;
      json result {{"schema", "sunshine.game3d.replay.v1"}, {"parameter_abi", "sunshine_game3d.render_parameters.v2"}, {"parameter_bytes", sizeof(render_parameters)}, {"parameter_hex", parameter_hex(f.parameters)}, {"parameter_encoding", "little-endian; IEEE754 float32 and 32-bit integers; exact b0 bytes"}, {"parameter_layout", json::array({layout("strength", offsetof(render_parameters, strength), 1, "float32"), layout("depth_view", offsetof(render_parameters, depth_view), 1, "int32"), layout("depth_ready", offsetof(render_parameters, depth_ready), 1, "uint32"), layout("camera_ready", offsetof(render_parameters, camera_ready), 1, "uint32"), layout("coordinate_basis", offsetof(render_parameters, coordinate_basis), 1, "int32"), layout("depth_scale", offsetof(render_parameters, depth_scale), 1, "float32"), layout("strength_blend", offsetof(render_parameters, strength_blend), 1, "float32"), layout("disparity_limit_uv", offsetof(render_parameters, disparity_limit_uv), 1, "float32"), layout("projection", offsetof(render_parameters, projection), 2, "float32"), layout("raw_depth_range", offsetof(render_parameters, raw_depth_range), 2, "float32"), layout("convergence", offsetof(render_parameters, convergence), 2, "float32"), layout("jitter", offsetof(render_parameters, jitter), 2, "float32"), layout("depth_rect", offsetof(render_parameters, depth_rect), 4, "float32")})}, {"shader_source", std::string(f.shader_source.empty() ? renderer::shader_source() : f.shader_source)}, {"compiler", "D3DCompile"}, {"compile_flags", D3DCOMPILE_OPTIMIZATION_LEVEL3}, {"compile_flags2", 0}, {"defines", {{"BUFFER_WIDTH", source.texture.width}, {"BUFFER_HEIGHT", source.texture.height}, {"BUFFER_COLOR_SPACE", color}}}, {"constant_binding", "b0; 80 exact bytes shared by all passes; v2 assigns byte28 to disparity_limit_uv"}, {"samplers", json::array({{{"register", "s0"}, {"filter", "min_mag_mip_point"}, {"address_uvw", "clamp"}}, {{"register", "s1"}, {"filter", "min_mag_linear_mip_point"}, {"address_uvw", "clamp"}}, {{"register", "s2"}, {"filter", "min_mag_mip_point"}, {"address_uvw", "border"}, {"border_rgba", {0, 0, 0, 0}}}})}, {"vertex_shader", {{"entry", "PostProcessVS"}, {"target", "vs_5_0"}, {"draw", "triangle list, 3 vertices, no vertex buffer"}}}, {"passes", json::array({{{"entry", "SunshinePreparePQPS"}, {"target", "ps_5_0"}, {"enabled", color == 3}, {"srvs", {{"t0", "source_color"}}}, {"rtvs", {"linear_color:RGBA16_FLOAT"}}}, {{"entry", "SunshineHostCandidateCS"}, {"target", "cs_5_0"}, {"enabled", conditioning}, {"srvs", {{"t1", "raw_depth"}}}, {"uavs", {{"u0", "candidate:R32_FLOAT"}}}}, {{"entry", "SunshineHostVerticalCS"}, {"target", "cs_5_0"}, {"enabled", conditioning}, {"srvs", {{"t3", "candidate"}}}, {"uavs", {{"u1", "vertical_majorant:R32_FLOAT"}, {"u2", "vertical_field:R32_FLOAT"}}}}, {{"entry", "SunshineHostHorizontalCS"}, {"target", "cs_5_0"}, {"enabled", conditioning}, {"srvs", {{"t0", "source_color"}, {"t4", "vertical_field"}}}, {"uavs", {{"u3", "final_field:R32_FLOAT"}}}}, {{"entry", "SunshineRenderEyesPS"}, {"target", "ps_5_0"}, {"enabled", true}, {"srvs", {{"t0", "source_color"}, {"t1", "raw_depth"}, {"t2", "linear_color"}, {"t5", "final_field"}}}, {"rtvs", {"left:RGBA16_FLOAT", "right:RGBA16_FLOAT"}}}, {{"entry", "SunshinePackEyesPS"}, {"target", "ps_5_0"}, {"enabled", true}, {"srvs", {{"t0", "source_color"}, {"t6", "left"}, {"t7", "right"}}}, {"rtvs", {color == 1 ? "sbs:R10G10B10A2_UNORM" : "sbs:RGBA16_FLOAT"}}}})}, {"binding_notes", "Absent textures bind null; missing depth uses a 1x1 placeholder with depth_ready=camera_ready=0. Raw depth artifact is the full consumed Texture2D SRV at mip0, layer0. Replay uploads it unchanged and retains depth_rect/jitter constants."}};
      result["source_alpha_ui"] = f.source_alpha_ui;
      const bool external_ui = f.source_alpha_ui && f.resources.ui_source.handle;
      result["ui_alpha_source"] = !f.source_alpha_ui ? "none" : external_ui ? "ui_source_color" : "source_color";
      const bool nearest_mode = f.ui_plane.mode == ui_plane_mode::depth_midpoint_nearest_ui;
      const bool front_mode = f.ui_plane.mode == ui_plane_mode::front_limit;
      const auto captured_shader = f.shader_source.empty() ? renderer::shader_source() : f.shader_source;
      if (captured_shader.find("#define SUNSHINE_UI_NEAREST_PLANE 1") != std::string_view::npos) {
        const bool reduced = f.resources.ui_plane_resolved.handle != 0;
        auto &passes = result["passes"];
        passes.insert(passes.begin() + 3, json {{"entry", "SunshineUINearestTilesCS"}, {"target", "cs_5_0"},
          {"enabled", reduced}, {"srvs", {{"t0", external_ui ? "ui_source_color" : "source_color"}, {"t1", "raw_depth"}}},
          {"uavs", {{"u4", "ui_plane_tiles:R32_FLOAT"}}}});
        passes.insert(passes.begin() + 4, json {{"entry", "SunshineUINearestReduceCS"}, {"target", "cs_5_0"},
          {"enabled", reduced}, {"srvs", {{"t8", "ui_plane_tiles"}}}, {"uavs", {{"u5", "ui_plane_resolved:R32_FLOAT"}}}});
        if (reduced) passes[5]["srvs"]["t9"] = "ui_plane_resolved";
        result["ui_plane_resolution"] = {{"reduction_ran", reduced},
          {"policy", front_mode ? "fixed_current_front_limit" : nearest_mode ? "max(submitted midpoint floor, maximum valid decoded q under finite positive selected alpha); one scalar for the entire UI in this render" : f.ui_plane.mode == ui_plane_mode::depth_midpoint ? "explicit_inverse_depth_plane" : "screen_plane"},
          {"inverse_depth_role", front_mode || f.ui_plane.mode == ui_plane_mode::screen ? "unused" : nearest_mode ? "midpoint_floor" : "explicit_plane"},
          {"resolved_value", nullptr},
          {"resolved_value_note", front_mode ? "No depth reduction. UI parallax is the current positive b0 display bound times strength and stereo blend when warp is active; depth/gain/zero do not place this plane." : nearest_mode ? "GPU result; no CPU readback in the live producer. Replay recomputes it from the exact captured depth, alpha, crop/jitter and constants." : "No depth reduction. Replay uses the exact captured UI mode and inverse-depth word."}};
      }
      if (external_ui) {
        for (auto &pass : result["passes"]) {
          if (pass["entry"] == "SunshineHostHorizontalCS") pass["srvs"]["t0"] = "ui_source_color";
        }
      }
      result["source_alpha_ui_requested"] = f.source_alpha_decision.requested;
      result["source_alpha_input_state"] = name(f.source_alpha_decision.input_state);
      result["source_alpha_capture_attempt"] = f.ui_capture_attempt_metadata.empty() ? json(nullptr) :
        json::parse(f.ui_capture_attempt_metadata);
      result["source_alpha_ui_status"] = f.source_alpha_ui ? (external_ui ? "captured_pre_fg_alpha" : "present_alpha") :
        f.source_alpha_decision.blocked_by_fg() ? "unavailable_fg_output_alpha" :
        f.source_alpha_decision.automatic ? name(f.source_alpha_decision.coverage.state) : "disabled";
      if (f.source_alpha_decision.automatic) {
        const auto &coverage = f.source_alpha_decision.coverage;
        result["source_alpha_auto"] = {{"state", name(coverage.state)}, {"enabled", coverage.enabled}, {"monitoring", coverage.monitoring},
          {"window_started", coverage.window_started}, {"window_start_ms", coverage.window_start_ms}, {"probe_interval_ms", coverage.probe_interval_ms}, {"accepted_samples", coverage.accepted_samples},
          {"covered_pixels", coverage.covered}, {"total_pixels", coverage.pixels},
          {"sample_sequence", coverage.sample_sequence}, {"sample_tick_ms", coverage.sample_tick_ms},
          {"meaning", "Detection starts with the first eligible alpha probe, not process creation, or follows an explicit manual choice. It probes frequently for one minute, then once per second until five minutes total. Selective alpha enables protection provisionally and resumes frequent probing; 500 ms of sustained evidence confirms On and stops monitoring. Without confirmation, the five-minute deadline fixes Off. Manual choices also stop monitoring. A retained choice is not ongoing semantic validation. Replay uses source_alpha_ui without rerunning detection."}};
      }
      const auto &fg = f.source_alpha_decision.fg;
      result["source_alpha_ui_fg_mode"] = {{"known", fg.known}, {"enabled", fg.enabled},
        {"automatic", fg.automatic}, {"generated_frames", fg.generated_frames}, {"epoch", fg.epoch},
        {"sequence", fg.sequence}, {"viewport", fg.viewport},
        {"meaning", "Last confirmed game-requested FG mode, retained during transient observation loss. Independent of selected depth; does not identify this presentation as real or generated."}};
      const auto ui = ui_parameter_words(f.source_alpha_ui, f.ui_plane);
      result["ui_parameter_abi"] = front_mode ? "sunshine_game3d.ui_parameters.v4" : nearest_mode ? "sunshine_game3d.ui_parameters.v3" : "sunshine_game3d.ui_parameters.v2";
      result["ui_parameter_bytes"] = sizeof(ui);
      result["ui_parameter_hex"] = parameter_hex(ui);
      result["ui_parameter_encoding"] = "little-endian exact b1 bytes: uint32 enabled, uint32 mode, IEEE754 float32 inverse depth, uint32 reserved zero";
      result["ui_constant_binding"] = {{"register", "b1"}, {"uint32", ui},
        {"mode", static_cast<std::uint32_t>(f.ui_plane.mode)},
        {"mode_name", f.ui_plane.mode == ui_plane_mode::screen ? "screen" : f.ui_plane.mode == ui_plane_mode::depth_midpoint ? "depth_midpoint" :
          nearest_mode ? "depth_midpoint_nearest_ui" : front_mode ? "front_limit" : "unknown"},
        {"inverse_depth", std::isfinite(f.ui_plane.inverse_depth) ? json(f.ui_plane.inverse_depth) : json(nullptr)},
        {"inverse_depth_bits", ui[2]},
        {"meaning", "When source_alpha_ui is enabled, alpha from ui_alpha_source supplies UI coverage. Horizontal-pass t0 uses that input; eye RGB remains current source_color. Mode 3 pins UI at the current positive display bound scaled by strength and stereo blend, independent of scene depth/gain/zero; its inverse-depth word is unused. Screen mode pins at zero disparity. Mode 1 uses the submitted independent depth. Mode 2 reduces max(submitted midpoint floor, nearest valid decoded depth under finite positive alpha). Both depth modes use b0 geometry. The horizontal protection includes one bilinear-support pixel. With protection enabled all-white masks are entirely UI; all-black masks have no UI constraints."}};
      return result;
    }

    nlohmann::json frame_json(api::effect_runtime *runtime, const diagnostic_frame &f) {
      const auto &p = f.parameters;
      const auto &d = f.depth;
      const auto &provided = d.provided;
      nlohmann::json result {
        {"schema", "sunshine.game3d.gpu-dump.v1"},
        {"build", {{"revision", SUNSHINE_GAME3D_BUILD_REVISION}, {"native_shader_sha256", SUNSHINE_GAME3D_SHADER_SHA256}, {"date", __DATE__}, {"time", __TIME__}, {"reshade_sdk", "6.8.0"}, {"reshade_api", RESHADE_API_VERSION},
#ifdef __VERSION__
                   {"compiler", __VERSION__},
#else
                   {"compiler", "MSVC"},
#endif
                   {"runtime_version", "see latest_observations.modules entry for the loaded ReShade file/product version; SDK version alone is not runtime identity"}}},
        {"pairing", "exact color and depth inputs consumed by this stereo render; reused depth may belong to an earlier real frame"},
        {"pairing_evidence", {{"exact_consumed_inputs", true}, {"same_game_frame", "unverified"}, {"reused_depth", d.reused_depth}, {"reason", "No common identity proves that presented color and the consumed depth originated in the same game-rendered frame. Export sequence and presentation ordinal are transport/presentation identities, not game frame IDs."}}},
        {"render_identity", {{"native_swapchain", runtime->get_native()}, {"native_queue", runtime->get_command_queue()->get_native()}, {"backbuffer_index", f.backbuffer_index == UINT32_MAX ? nlohmann::json(nullptr) : nlohmann::json(f.backbuffer_index)}, {"presentation_ordinal", f.presentation_ordinal ? nlohmann::json(f.presentation_ordinal) : nlohmann::json(nullptr)}, {"capture_qpc", f.qpc}, {"export_generation", f.export_generation}, {"export_sequence", f.export_sequence}}},
        {"overlay_included", false},
        {"color_space", static_cast<unsigned>(f.color)},
        {"artifact_semantics", {{"source_color", f.color == api::color_space::hdr10_pq ? "Unmodified game color: ST.2084 PQ, Rec.2020 primaries, encoded absolute luminance." : f.color == api::color_space::scrgb ? "Unmodified game color: linear scRGB, Rec.709 primaries, 1.0 = 80 nits; negative and greater-than-one values preserved." :
                                                                                                                                                                                                                     "Unmodified game color: encoded sRGB SDR, Rec.709 primaries."},
                                {"linear_color", "Optional PQ-decoded game color: linear scRGB, Rec.709 primaries, 1.0 = 80 nits."},
                                {"raw_depth", "Full allocation of unfiltered float32 hardware depth. Active rectangle and encoding belong to consumed_depth; no normalization or clamp applied."},
                                {"candidate", "Signed full-source-U parallax before spatial conditioning."},
                                {"vertical_majorant", "Signed full-source-U vertical upper envelope of candidate parallax."},
                                {"vertical_field", "Signed full-source-U field after fixed vertical envelope blending."},
                                {"final_field", "Signed full-source-U final horizontal envelope consumed by inverse stereo warping."},
                                {"sbs", f.color == api::color_space::srgb ? "Full SBS, left eye first, encoded sRGB SDR; before overlay and Windows cursor composition." : "Full SBS, left eye first, linear scRGB, Rec.709 primaries, 1.0 = 80 nits; before overlay and Windows cursor composition."}}},
        {"render_parameters", {{"strength", p.strength}, {"depth_view", p.depth_view}, {"depth_ready", p.depth_ready}, {"camera_ready", p.camera_ready}, {"coordinate_basis", p.coordinate_basis}, {"depth_scale", p.depth_scale}, {"strength_blend", p.strength_blend}, {"disparity_limit_uv", p.disparity_limit_uv}, {"projection", p.projection}, {"raw_depth_range", p.raw_depth_range}, {"convergence", p.convergence}, {"jitter_uv", p.jitter}, {"depth_rect", p.depth_rect}}},
        {"consumed_depth", {{"ready", d.ready}, {"reused_depth", d.reused_depth}, {"frame_generation_active", d.frame_generation_active}, {"source_resource", d.source_resource.handle}, {"captured_resource", d.resource.handle}, {"source_id", d.source_id}, {"layout_epoch", d.layout_epoch}, {"runtime_epoch", d.runtime_epoch}, {"frame_index", d.frame_index}, {"backup_id", d.backup_id}, {"command_queue", d.command_queue}, {"orientation", static_cast<unsigned>(d.orientation)}, {"detected_orientation", static_cast<unsigned>(d.detected_orientation)}, {"aligned_viewport_assumed", d.aligned_viewport_assumed}, {"capture_recording", {{"command", d.capture_marker.command}, {"object_generation", d.capture_marker.object_generation}, {"recording_generation", d.capture_marker.recording_generation}, {"event", d.capture_marker.event}, {"epoch", d.capture_marker.epoch}, {"loss", d.capture_marker.loss}}}, {"preserved_copy", {{"id", d.depth_copy.copy_id}, {"state", static_cast<unsigned>(d.depth_copy.state)}, {"source_content_version", d.depth_copy.source.version}, {"source_lifetime", d.depth_copy.source.resource.lifetime}, {"backup_lifetime", d.depth_copy.backup.lifetime}}}, {"width", d.width}, {"height", d.height}, {"active_rect", {d.x, d.y, d.active_width, d.active_height}}, {"provider", !d.ready ? "none" : !provided.resource.native                                   ? "generic" :
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           provided.provider == sunshine_scene_depth::provider_kind::ngx ? "ngx" :
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "streamline"},
                            {"nominated_provider", !provided.resource.native ? "unavailable" : provided.provider == sunshine_scene_depth::provider_kind::ngx ? "ngx" :
                                                                                                                                                               "streamline"},
                            {"provider_epoch", provided.epoch},
                            {"provider_sequence", provided.sequence},
                            {"provider_tick", provided.tick},
                            {"provider_viewport", provided.viewport},
                            {"provider_source_id", provided.source_id},
                            {"source_frame_generation", provided.source_frame_generation},
                            {"source_frame_token", provided.source_frame_token},
                            {"source_frame_numeric", provided.source_frame_numeric},
                            {"source_frame_has_numeric", provided.source_frame_has_numeric},
                            {"source_frame_explicit", provided.source_frame_explicit},
                            {"projection", {{"supplied", d.projection.supplied}, {"epoch", d.projection.epoch}, {"viewport", d.projection.viewport}, {"A", d.projection.A}, {"B", d.projection.B}, {"raw_scale", d.projection.raw_scale}, {"raw_bias", d.projection.raw_bias}}},
                            {"jitter", {{"supplied", provided.jitter.supplied}, {"x", provided.jitter.x}, {"y", provided.jitter.y}, {"width", provided.jitter.width}, {"height", provided.jitter.height}}},
                            {"raw_depth_encoding", "unfiltered Texture2D.Load of the consumed depth SRV, float32; no calibration, crop, jitter compensation or clamp"}}}
      };
      auto *device = runtime->get_device();
      const auto source_desc = device->get_resource_desc(f.resources.source);
      result["replay"] = replay_json(f, source_desc);
      result["source_color_allocation"] = allocation_json(source_desc);
      if (f.source_alpha_ui && f.resources.ui_source.handle) {
        result["ui_source_allocation"] = allocation_json(device->get_resource_desc(f.resources.ui_source));
        result["ui_source"] = f.ui_source_metadata.empty() ? nlohmann::json::object() : nlohmann::json::parse(f.ui_source_metadata);
        result["artifact_semantics"]["ui_source_color"] = "Exact full RGBA texture consumed as horizontal-pass UI alpha; RGB is diagnostic only and never replaces current source_color for eye rendering. This does not prove same-game-frame pairing.";
      }
      if (d.ready && d.shader_resource.handle) {
        const auto resource = device->get_resource_from_view(d.shader_resource);
        const auto allocation = device->get_resource_desc(resource);
        const auto view = device->get_resource_view_desc(d.shader_resource);
        // The shared capture owner supplies a single non-array 2D level. Do
        // not silently read another subresource if that contract ever changes.
        if (allocation.type != api::resource_type::texture_2d || allocation.texture.samples != 1 || allocation.texture.width != d.width || allocation.texture.height != d.height || view.type != api::resource_view_type::texture_2d || view.texture.first_level || view.texture.first_layer) {
          throw std::runtime_error("Consumed depth view is outside the diagnostic mip0/layer0 2D capture contract");
        }
        result["consumed_depth"]["allocation"] = allocation_json(allocation);
        result["consumed_depth"]["srv"] = {{"api_type", static_cast<unsigned>(view.type)}, {"api_format", static_cast<unsigned>(view.format)}, {"first_level", view.texture.first_level}, {"levels", view.texture.levels}, {"first_layer", view.texture.first_layer}, {"layers", view.texture.layers}};
      }
      const auto tick = GetTickCount64();
      result["consumed_depth"]["provider_observation_age_ms"] = provided.tick && provided.tick <= tick ? nlohmann::json(tick - provided.tick) : nlohmann::json(nullptr);
      result["consumed_depth"]["age_note"] = "Elapsed GetTickCount64 milliseconds since the consumed provider observation, when available; not a measured color/depth timestamp difference.";
      if (!d.ready || !d.shader_resource.handle) {
        result["missing_raw_depth"] = "The stereo render had no safe depth read lease; source color and SBS are still captured.";
      }
      if (!f.resources.final_field.handle) {
        result["missing_conditioning_fields"] = "Conditioning passes did not run for this render extent; retained fields are not exported.";
      }
      if (!f.resources.linear_color.handle) {
        result["missing_linear_color"] = "This input does not require the separate HDR10-PQ decoding pass.";
      }
      const auto &scale = f.scene_status.scale;
      const bool camera = scale.basis == automatic_scale_basis::camera_matrix;
      auto &state = result["render_scene_policy"];
      state = {{"phase", static_cast<unsigned>(f.scene_status.phase)}, {"basis", camera ? "camera_matrix" : scale.basis == automatic_scale_basis::relative_depth ? "relative_depth" :
                                                                                                                                                                   "unknown"},
               {"active", scale.active},
               {"zero_plane_units", camera ? "game units, not necessarily meters" : "relative raw depth, not distance"},
               {"applied_zero_inverse", p.convergence[1]},
               {"reference_zpd", p.convergence[0]}};
      state["current_scale"] = scale.has_value() ? nlohmann::json(scale.value) : nlohmann::json(nullptr);
      state["target_scale"] = scale.has_target() ? nlohmann::json(scale.target_value) : nlohmann::json(nullptr);
      state["disparity_budget"] = {{"full_strength_per_eye_source_uv", p.disparity_limit_uv},
        {"applied_per_eye_source_uv", p.disparity_limit_uv * std::clamp(p.strength, 0.f, 100.f) * .01f * std::clamp(p.strength_blend, 0.f, 1.f)},
        {"note", "Current-pixel source-U constraint, enforced before and after spatial conditioning. It does not clip source depth or imply measured color/depth alignment."}};
      state["gain_reference"] = "Ktarget=L/Q: nearest decoded depth Q over every active pixel; L is the captured output multiplier; current-pixel displacement is bounded in the GPU independently";
      state["target_inverse_gain"] = scale.has_target() && scale.target_value > 0 ?
        nlohmann::json(1.0 / double(scale.target_value)) : nlohmann::json(nullptr);
      state["depth_statistics"] = scale.depth_statistics_valid() ? nlohmann::json{
        {"reference_Q", scale.reference_inverse}, {"nearest_q", scale.maximum_inverse}, {"farthest_q", scale.minimum_inverse}, {"mean_q", scale.mean_inverse}, {"normalization_L", scale.normalization},
        {"mean_square_q", scale.depth_pixel_count ? nlohmann::json(scale.mean_square_inverse) : nlohmann::json(nullptr)},
        {"centered_moments_supplied", scale.has_centered_depth_statistics},
        {"mean_q_minus_min", scale.has_centered_depth_statistics ? nlohmann::json(scale.mean_contrast_inverse) : nlohmann::json(nullptr)},
        {"mean_square_q_minus_min", scale.has_centered_depth_statistics ? nlohmann::json(scale.mean_contrast_square_inverse) : nlohmann::json(nullptr)},
        {"pixel_count", scale.depth_pixel_count},
        {"tiles", {scale.depth_tiles_x, scale.depth_tiles_y}},
        {"sampling", scale.depth_pixel_count ? "all_active_pixels" : "legacy_point_fixture"},
        {"units", camera ? "inverse game units" : "relative inverse depth"},
        {"note", "Accepted scene-policy measurement for this captured rendering decision, not a new measurement of its pixels. Live Q, mean, second moment and bounds include every active-depth pixel. Larger q is nearer; these are not camera clipping planes. The uncentered mean_square_q is diagnostic and does not set gain. Centered moments determine the trial zero target. The UI may retain older values during a missing-depth hold when this decision has no statistics."}
      } : nlohmann::json(nullptr);
      state["zero_plane_policy"] = "trial contrast midpoint: b=qmin, d=q-b, q0 target=b+0.5*sum(d*d)/sum(d); stable centered full-pixel moments, fixtures without centered moments retain midpoint; exponential tracking bounded by abs(Knew*delta_q0/L)<=zero_budget_per_second*elapsed_seconds; no immediate applied-zero range clamp; positive flat depth targets b while holding gain, all-zero depth holds both";
      state["zero_tracking"] = {{"time_constant_seconds", sunshine_camera_scene::time_constant_seconds},
        {"zero_budget_per_second", sunshine_scene_gain::zero_budget_per_second}};
      state["gain_below_target"] = scale.gain_below_target;
      state["zero_inverse"] = scale.has_zero ? nlohmann::json(scale.zero_inverse) : nlohmann::json(nullptr);
      state["ui_midpoint_inverse"] = scale.has_ui_midpoint ? nlohmann::json(scale.ui_midpoint_inverse) : nlohmann::json(nullptr);
      state["target_ui_midpoint_inverse"] = scale.has_ui_midpoint_target() ? nlohmann::json(scale.target_ui_midpoint_inverse) : nlohmann::json(nullptr);
      state["ui_midpoint_policy"] = "Live mode 1 consumes the independently smoothed inverse-depth midpoint target=(qmin+qmax)/2, separate from scene zero. Current gain, zero, strength and blend map it to display parallax. Historical mode 2 uses the submitted midpoint as a floor for covered-depth reduction; mode 3 uses the full positive display bound. Exact consumed mode and words are in replay b1.";
      state["zero_plane_at_infinity"] = scale.has_zero && camera && scale.zero_inverse == 0.f;
      state["current_zero_plane"] = scale.has_zero && (!camera || scale.zero_inverse > 0.f) ?
        nlohmann::json(camera ? 1.0 / scale.zero_inverse : double(scale.zero_inverse)) : nlohmann::json(nullptr);
      state["target_zero_inverse"] = scale.has_zero_target() ? nlohmann::json(scale.target_zero_inverse) : nlohmann::json(nullptr);
      state["target_zero_plane_at_infinity"] = scale.has_zero_target() && camera && scale.target_zero_inverse == 0.;
      state["target_zero_plane"] = scale.has_zero_target() && (!camera || scale.target_zero_inverse > 0.) ?
        nlohmann::json(camera ? 1.0 / scale.target_zero_inverse : scale.target_zero_inverse) : nlohmann::json(nullptr);
      state["conversion_scale"] = scale.projection_conversion_valid() ? nlohmann::json(scale.conversion_multiplier) : nlohmann::json(nullptr);
      state["conversion_offset"] = scale.projection_conversion_valid() ? nlohmann::json(scale.conversion_offset) : nlohmann::json(nullptr);
      // These observations are useful diagnostics, but are not the authority for
      // the consumed frame above. Their own adapter identity remains explicit.
      result["latest_observations"] = nlohmann::json::parse(diagnostic_metadata_json());
      return result;
    }

    struct texture {
      wire::texture_t description;
      ComPtr<ID3D11Texture2D> d11;
      ComPtr<ID3D12Resource> d12;
      handle shared;
      api::resource source {};

      api::resource resource() const {
        return {reinterpret_cast<std::uint64_t>(d11 ? static_cast<void *>(d11.Get()) : static_cast<void *>(d12.Get()))};
      }
    };

    struct capture_batch {
      api::effect_runtime *runtime = nullptr;
      api::device *device = nullptr;
      std::uint64_t swapchain = 0, native_queue = 0, request = 0, nonce = 0;
      wire::response_t response;
      std::string json;
      std::vector<std::unique_ptr<texture>> textures;
      std::vector<ComPtr<IUnknown>> sources;
      std::unique_ptr<ui_capture_batch> optional;
      ComPtr<ID3D11Device5> device11;
      ComPtr<ID3D11DeviceContext4> context11;
      ComPtr<ID3D11Fence> fence11;
      ComPtr<ID3D12Device> device12;
      ComPtr<ID3D12CommandQueue> queue12;
      ComPtr<ID3D12Fence> fence12;
      api::pipeline_layout layout {};
      api::pipeline pipeline {};
      api::resource_view depth_uav {}, null_uav {};
      bool recorded = false, signalled = false, signal_failed = false, published = false;
      std::uint64_t bytes = 0;

      void release_program() {
        if (depth_uav.handle) {
          device->destroy_resource_view(depth_uav);
        }
        if (null_uav.handle) {
          device->destroy_resource_view(null_uav);
        }
        if (pipeline.handle) {
          device->destroy_pipeline(pipeline);
        }
        if (layout.handle) {
          device->destroy_pipeline_layout(layout);
        }
        depth_uav = null_uav = {};
        pipeline = {};
        layout = {};
        sources.clear();
      }

      ~capture_batch() {
        release_program();
      }

      bool complete() const {
        if (!recorded) {
          return true;
        }
        if (!signalled || signal_failed) {
          return false;
        }
        const auto value = fence11 ? fence11->GetCompletedValue() : fence12->GetCompletedValue();
        return value != UINT64_MAX && value >= 1;
      }

      bool initialize(api::effect_runtime *owner) {
        runtime = owner;
        device = owner->get_device();
        swapchain = owner->get_native();
        native_queue = owner->get_command_queue()->get_native();
        if (device->get_api() == api::device_api::d3d11) {
          return SUCCEEDED(reinterpret_cast<ID3D11Device *>(device->get_native())->QueryInterface(IID_PPV_ARGS(&device11))) &&
                 SUCCEEDED(reinterpret_cast<ID3D11DeviceContext *>(native_queue)->QueryInterface(IID_PPV_ARGS(&context11))) &&
                 SUCCEEDED(device11->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence11)));
        }
        if (device->get_api() != api::device_api::d3d12) {
          return false;
        }
        return SUCCEEDED(reinterpret_cast<ID3D12Device *>(device->get_native())->QueryInterface(IID_PPV_ARGS(&device12))) &&
               SUCCEEDED(reinterpret_cast<ID3D12CommandQueue *>(native_queue)->QueryInterface(IID_PPV_ARGS(&queue12))) &&
               SUCCEEDED(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence12)));
      }

      texture *create(wire::artifact kind, unsigned width, unsigned height, DXGI_FORMAT format, bool uav = false) {
        const auto bpp = pixel_bytes(format);
        const std::uint64_t added = std::uint64_t(width) * height * bpp;
        if (!width || !height || width > wire::max_dimension || height > wire::max_dimension || !bpp || added > wire::max_capture_bytes - bytes || textures.size() >= wire::max_textures) {
          return nullptr;
        }
        auto t = std::make_unique<texture>();
        if (device11) {
          D3D11_TEXTURE2D_DESC d {};
          d.Width = width;
          d.Height = height;
          d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1;
          d.Format = format;
          d.Usage = D3D11_USAGE_DEFAULT;
          d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0);
          d.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
          ComPtr<IDXGIResource1> shared;
          if (FAILED(device11->CreateTexture2D(&d, nullptr, &t->d11)) || FAILED(t->d11.As(&shared)) || FAILED(shared->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &t->shared.value))) {
            return nullptr;
          }
        } else {
          D3D12_HEAP_PROPERTIES heap {};
          heap.Type = D3D12_HEAP_TYPE_DEFAULT;
          D3D12_RESOURCE_DESC d {};
          d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
          d.Width = width;
          d.Height = height;
          d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
          d.Format = format;
          d.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
          if (uav) {
            d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
          }
          if (FAILED(device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t->d12))) || FAILED(device12->CreateSharedHandle(t->d12.Get(), nullptr, GENERIC_ALL, nullptr, &t->shared.value))) {
            return nullptr;
          }
        }
        t->description = {kind, width, height, static_cast<unsigned>(format), reinterpret_cast<std::uint64_t>(t->shared.value)};
        bytes += added;
        auto *result = t.get();
        textures.emplace_back(std::move(t));
        return result;
      }

      bool add(wire::artifact kind, api::resource resource) {
        if (!resource.handle) {
          return true;
        }
        const auto desc = device->get_resource_desc(resource);
        if (desc.type != api::resource_type::texture_2d || desc.texture.samples != 1 || desc.texture.levels != 1 || desc.texture.depth_or_layers != 1) {
          return false;
        }
        auto *t = create(kind, desc.texture.width, desc.texture.height, static_cast<DXGI_FORMAT>(api::format_to_default_typed(desc.texture.format, 0)));
        if (!t) {
          return false;
        }
        t->source = resource;
        sources.emplace_back(reinterpret_cast<IUnknown *>(resource.handle));
        return true;
      }

      bool depth_program(texture &depth) {
        static constexpr char shader[] =
          "Texture2D<float> Input : register(t0); RWTexture2D<float> Output : register(u0);"
          "[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID){uint w,h;Output.GetDimensions(w,h);"
          "if(id.x<w&&id.y<h)Output[id.xy]=Input.Load(int3(id.xy,0));}";
        ComPtr<ID3DBlob> code, errors;
        if (FAILED(D3DCompile(shader, sizeof(shader) - 1, "Game3D diagnostic raw depth", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
          return false;
        }
        const api::pipeline_layout_param ranges[] {
          api::descriptor_range {0, 0, 0, 1, api::shader_stage::compute, 1, api::descriptor_type::shader_resource_view},
          api::descriptor_range {0, 0, 0, 1, api::shader_stage::compute, 1, api::descriptor_type::unordered_access_view}
        };
        api::shader_desc s {code->GetBufferPointer(), code->GetBufferSize()};
        const api::pipeline_subobject part {api::pipeline_subobject_type::compute_shader, 1, &s};
        return device->create_pipeline_layout(2, ranges, &layout) && device->create_pipeline(layout, 1, &part, &pipeline) &&
               device->create_resource_view(depth.resource(), api::resource_usage::unordered_access, api::resource_view_desc(api::format::r32_float), &depth_uav) &&
               (!device12 || device->create_resource_view({}, api::resource_usage::unordered_access, api::resource_view_desc(api::format::r32_float), &null_uav));
      }

      void record(api::command_list *commands, api::resource_view depth) {
        recorded = true;
        for (const auto &t : textures) {
          const auto target = t->resource();
          if (t->description.kind == wire::artifact::raw_depth) {
            commands->barrier(target, api::resource_usage::general, api::resource_usage::unordered_access);
            commands->bind_pipeline(api::pipeline_stage::compute_shader, pipeline);
            commands->push_descriptors(api::shader_stage::compute, layout, 0, {{}, 0, 0, 1, api::descriptor_type::shader_resource_view, &depth});
            commands->push_descriptors(api::shader_stage::compute, layout, 1, {{}, 0, 0, 1, api::descriptor_type::unordered_access_view, &depth_uav});
            commands->dispatch((t->description.width + 7) / 8, (t->description.height + 7) / 8, 1);
            commands->push_descriptors(api::shader_stage::compute, layout, 1, {{}, 0, 0, 1, api::descriptor_type::unordered_access_view, &null_uav});
            commands->barrier(target, api::resource_usage::unordered_access, api::resource_usage::general);
          } else {
            commands->barrier(t->source, api::resource_usage::shader_resource, api::resource_usage::copy_source);
            commands->barrier(target, api::resource_usage::general, api::resource_usage::copy_dest);
            commands->copy_resource(t->source, target);
            commands->barrier(target, api::resource_usage::copy_dest, api::resource_usage::general);
            commands->barrier(t->source, api::resource_usage::copy_source, api::resource_usage::shader_resource);
          }
        }
      }
    };
  }  // namespace

  struct debug_dump::impl {
    mutable std::mutex mutex;
    handle mapping;
    wire::shared_state_t *shared = nullptr;
    std::unique_ptr<capture_batch> pending;
    std::unique_ptr<ui_capture_batch> armed_optional;
    std::uint64_t last_request = 0, last_nonce = 0, capture_id = 0;
    std::uint64_t armed_request = 0, armed_nonce = 0;
    std::uint64_t armed_tick = 0;
    bool armed_ready = false;

    ~impl() {
      arm_diagnostic_metadata(false);
      armed_optional.reset();
      // An explicit hot-unload cannot leave callbacks to reap GPU objects.
      if (pending && !pending->complete()) {
        pending.release();
      }
      if (shared) {
        UnmapViewOfFile(shared);
      }
    }

    bool current(const capture_batch &p) const {
      return shared && load(shared->request_id) == p.request && load(shared->consumer_nonce) == p.nonce &&
             load(shared->released_id) != p.request;
    }

    bool waiting() const {
      if (!shared || pending) {
        return false;
      }
      const auto id = load(shared->request_id);
      const auto nonce = load(shared->consumer_nonce);
      return id && (id != last_request || nonce != last_nonce) && id != load(shared->released_id) && nonce &&
             shared->consumer_pid && shared->consumer_creation_time;
    }

    bool requested() const {
      return waiting() && armed_ready && armed_request == load(shared->request_id) &&
             armed_nonce == load(shared->consumer_nonce);
    }

    void publish(capture_batch &p) {
      if (!current(p)) {
        return;
      }
      if (p.json.size() >= wire::max_json_bytes) {
        p.response.result = wire::status::failed;
        p.response.texture_count = 0;
        p.json = R"({"error":"Game3D diagnostic metadata exceeds the bounded mailbox."})";
      }
      p.response.json_bytes = static_cast<unsigned>(p.json.size());
      shared->response = p.response;
      std::memcpy(shared->json, p.json.data(), p.json.size());
      shared->json[p.json.size()] = 0;
      store(shared->response_id, p.request);
      p.published = true;
    }

    void poll(bool foreground_frame = false) {
      if (pending && pending->complete()) {
        pending->release_program();
        if (current(*pending)) {
          if (!pending->published) {
            if (!pending->optional || pending->optional->append(pending->response, pending->bytes, pending->json,
                  pending->response.result == wire::status::complete)) {
              publish(*pending);
            }
          }
        } else {
          pending.reset();
        }
      }
      if (!waiting()) {
        armed_ready = false;
        armed_optional.reset();
        // Finish callbacks for already recorded optional snapshots must still
        // arrive while their native producer work retires. New copies were
        // stopped by freeze at the main frame boundary.
        if (!pending || pending->published) arm_diagnostic_metadata(false);
      } else if (foreground_frame) {
        const auto id = load(shared->request_id), nonce = load(shared->consumer_nonce);
        if (armed_request != id || armed_nonce != nonce) {
          armed_request = id;
          armed_nonce = nonce;
          armed_ready = false;
          armed_tick = GetTickCount64();
          arm_diagnostic_metadata(false);
          arm_diagnostic_metadata(true);
          armed_optional = std::make_unique<ui_capture_batch>(diagnostic_metadata_generation());
        } else {
          // FG can present several times without a game/vendor evaluation.
          // Observe a real SDK call when possible, while keeping generic/no-API
          // captures bounded and never blocking rendering for diagnostics.
          armed_ready = has_diagnostic_frame_observation() || GetTickCount64() - armed_tick >= 100;
        }
      }
    }
  };

  debug_dump::debug_dump():
      data_(std::make_unique<impl>()) {}

  debug_dump::~debug_dump() = default;

  bool debug_dump::initialize(std::uint32_t pid, std::uint64_t creation_time) {
    auto &d = *data_;
    std::lock_guard<std::mutex> lock(d.mutex);
    const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(pid);
    d.mapping.value = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(wire::shared_state_t), name.c_str());
    if (!d.mapping.value || GetLastError() == ERROR_ALREADY_EXISTS) {
      return false;
    }
    d.shared = static_cast<wire::shared_state_t *>(MapViewOfFile(d.mapping.value, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
    if (!d.shared) {
      return false;
    }
    new (d.shared) wire::shared_state_t {};
    d.shared->producer_pid = pid;
    d.shared->producer_creation_time = creation_time;
    diagnostic::observe_module(reshade::internal::get_reshade_module_handle(), "ReShade", "SDK6.8.0/API20");
    return true;
  }

  void debug_dump::poll(bool foreground_frame) {
    std::lock_guard<std::mutex> lock(data_->mutex);
    data_->poll(foreground_frame);
  }

  bool debug_dump::awaiting_capture() const {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return data_->waiting();
  }

  bool debug_dump::requested() const {
    std::lock_guard<std::mutex> lock(data_->mutex);
    return data_->requested();
  }

  void debug_dump::unavailable(const char *reason) {
    auto &d = *data_;
    std::lock_guard<std::mutex> lock(d.mutex);
    if (!d.requested()) {
      return;
    }
    auto next = std::make_unique<capture_batch>();
    next->request = load(d.shared->request_id);
    next->nonce = load(d.shared->consumer_nonce);
    next->response.consumer_nonce = next->nonce;
    next->response.capture_id = ++d.capture_id;
    LARGE_INTEGER qpc {};
    QueryPerformanceCounter(&qpc);
    next->response.capture_qpc = static_cast<std::uint64_t>(qpc.QuadPart);
    next->response.result = wire::status::unavailable;
    next->optional = std::move(d.armed_optional);
    if (next->optional) next->optional->freeze();
    next->json = nlohmann::json {{"schema", "sunshine.game3d.gpu-dump.v1"}, {"unavailable", reason}, {"latest_observations", nlohmann::json::parse(diagnostic_metadata_json())}}.dump();
    d.last_request = next->request;
    d.last_nonce = next->nonce;
    d.pending = std::move(next);
    d.poll();
  }

  void debug_dump::capture(api::effect_runtime *runtime, api::command_list *commands, const diagnostic_frame &frame) {
    auto &d = *data_;
    std::lock_guard<std::mutex> lock(d.mutex);
    if (!d.requested()) {
      return;
    }
    auto next = std::make_unique<capture_batch>();
    next->request = load(d.shared->request_id);
    next->nonce = load(d.shared->consumer_nonce);
    next->response.consumer_nonce = next->nonce;
    next->response.capture_id = ++d.capture_id;
    next->response.capture_qpc = frame.qpc;
    next->response.runtime_epoch = frame.depth.runtime_epoch;
    next->response.export_generation = frame.export_generation;
    next->response.export_sequence = frame.export_sequence;
    next->optional = std::move(d.armed_optional);
    if (next->optional) next->optional->freeze();
    d.last_request = next->request;
    d.last_nonce = next->nonce;
    try {
      if (!next->initialize(runtime)) {
        throw std::runtime_error("Diagnostic GPU device/fence unavailable");
      }
      next->json = frame_json(runtime, frame).dump();
      const auto &r = frame.resources;
      if (frame.source_alpha_ui && r.ui_source.handle && !next->add(wire::artifact::ui_source_color, r.ui_source)) {
        throw std::runtime_error("Cannot snapshot the exact consumed UI-alpha source");
      }
      if (!r.source.handle || !r.sbs.handle || !next->add(wire::artifact::source_color, r.source) || !next->add(wire::artifact::linear_color, r.linear_color) || !next->add(wire::artifact::candidate, r.candidate) || !next->add(wire::artifact::vertical_majorant, r.vertical_majorant) || !next->add(wire::artifact::vertical_field, r.vertical_field) || !next->add(wire::artifact::final_field, r.final_field) || !next->add(wire::artifact::sbs, r.sbs)) {
        throw std::runtime_error("Cannot allocate the bounded shared diagnostic textures");
      }
      const auto &depth = frame.depth;
      if (depth.ready && depth.shader_resource.handle) {
        auto *t = next->create(wire::artifact::raw_depth, depth.width, depth.height, DXGI_FORMAT_R32_FLOAT, true);
        if (!t || !next->depth_program(*t)) {
          throw std::runtime_error("Cannot create the full-resolution raw-depth snapshot");
        }
        const auto resource = runtime->get_device()->get_resource_from_view(depth.shader_resource);
        if (!resource.handle) {
          throw std::runtime_error("Consumed depth view has no resource");
        }
        next->sources.emplace_back(reinterpret_cast<IUnknown *>(resource.handle));
      }
      next->response.texture_count = static_cast<unsigned>(next->textures.size());
      for (unsigned i = 0; i < next->response.texture_count; ++i) {
        next->response.textures[i] = next->textures[i]->description;
      }
      next->response.result = wire::status::complete;
      next->record(commands, depth.shader_resource);
    } catch (const std::exception &error) {
      next->response.result = wire::status::failed;
      next->response.texture_count = 0;
      auto metadata = nlohmann::json::parse(next->json, nullptr, false);
      if (!metadata.is_object()) metadata = {{"latest_observations", nlohmann::json::parse(diagnostic_metadata_json())}};
      metadata["error"] = error.what();
      next->json = metadata.dump();
    }
    d.pending = std::move(next);
    d.poll();
  }

  void debug_dump::finish_present(std::uint64_t queue, std::uint64_t swapchain) {
    std::lock_guard<std::mutex> lock(data_->mutex);
    auto &p = data_->pending;
    if (!p || !p->recorded || p->signalled || p->signal_failed || p->native_queue != queue || p->swapchain != swapchain) {
      return;
    }
    const auto hr = p->fence11 ? p->context11->Signal(p->fence11.Get(), 1) : p->queue12->Signal(p->fence12.Get(), 1);
    p->signal_failed = FAILED(hr);
    p->signalled = SUCCEEDED(hr);
    if (p->context11 && p->signalled) {
      p->context11->Flush();
    }
  }

  void debug_dump::retire_runtime(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(data_->mutex);
    auto &p = data_->pending;
    if (!p || p->runtime != runtime) {
      return;
    }
    // The caller owns ReShade's terminal drain. Preserve native shared snapshot
    // objects until acknowledgment, but never retain API wrappers across teardown.
    if (!p->complete()) {
      p->response.result = wire::status::unavailable;
      p->response.texture_count = 0;
      auto metadata = nlohmann::json::parse(p->json);
      metadata["unavailable"] = "Runtime destroyed before snapshot GPU completion was confirmed; no pixel artifacts published.";
      p->json = metadata.dump();
    }
    p->recorded = false;
    p->release_program();
    p->device = nullptr;
    p->runtime = nullptr;
    data_->poll();
  }
}  // namespace sunshine_game3d

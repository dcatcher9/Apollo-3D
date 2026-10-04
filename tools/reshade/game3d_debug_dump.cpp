// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_debug_dump.h"

#include "game3d_diagnostic_metadata.h"
#include "game3d_ui_capture.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_layer.h"
#include "game3d_ui_selection.h"
#include "scene_gain.h"
#include "src/game3d_debug_protocol.h"
#include "src/game3d_debug_formats.h"

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
      return wire::pixel_bytes(static_cast<unsigned>(format));
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

    // One image's hidden-scene evidence (docs/reshade-sbs.md, hidden-scene evidence).
    nlohmann::json scene_json(const alpha_auto_decision::scene_evidence &scene, bool verdict) {
      nlohmann::json result{{"n", scene.n}, {"d", scene.d}, {"valid", scene.valid}, {"ran", scene.ran}};
      if (verdict) {
        result["verdict"] = ui_detection::name(scene.verdict);
        result["decided"] = scene.decided;
      }
      return result;
    }
    // The pre-UI scene image's evidence (texel 6) and the image it measured;
    // its verdict is read from D.
    nlohmann::json pre_ui_scene_json(const alpha_auto_decision::scene_evidence &scene, std::uint32_t image) {
      auto result = scene_json(scene, false);
      result["image"] = image == ui_detection::pre_ui_image::hudless ? "hudless" :
        image == ui_detection::pre_ui_image::layer ? "layer" : "none";
      result["verdict"] = ui_detection::name(scene.verdict);
      return result;
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
      const bool shallow_mode = f.ui_plane.mode == ui_plane_mode::shallow_front;
      const bool fraction_mode = f.ui_plane.mode == ui_plane_mode::display_fraction;
      const auto fraction = fraction_mode ? (std::isfinite(f.ui_plane.inverse_depth) ? json(f.ui_plane.inverse_depth) : json(nullptr)) :
        shallow_mode ? json(0.25) : front_mode ? json(1.0) : json(nullptr);
      const auto captured_shader = f.shader_source.empty() ? renderer::shader_source() : f.shader_source;
      if (captured_shader.find("#define SUNSHINE_PACKED_EYES 1") != std::string_view::npos) {
        // Both eyes are rendered straight into the side-by-side target.
        auto &passes = result["passes"];
        passes.erase(passes.size() - 1);
        passes.erase(passes.size() - 1);
        passes.push_back({{"entry", "SunshineRenderPackedPS"}, {"target", "ps_5_0"}, {"enabled", true},
          {"srvs", {{"t0", "source_color"}, {"t1", "raw_depth"}, {"t2", "linear_color"}, {"t5", "final_field"}}},
          {"rtvs", {color == 1 ? "sbs:R10G10B10A2_UNORM" : "sbs:RGBA16_FLOAT"}}});
      }
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
          {"policy", fraction_mode ? "resolved_display_fraction" : shallow_mode ? "fixed_shallow_front" : front_mode ? "fixed_current_front_limit" : nearest_mode ? "max(submitted midpoint floor, maximum valid decoded q under finite positive selected alpha); one scalar for the entire UI in this render" : f.ui_plane.mode == ui_plane_mode::depth_midpoint ? "explicit_inverse_depth_plane" : "screen_plane"},
          {"inverse_depth_role", fraction_mode || shallow_mode || front_mode || f.ui_plane.mode == ui_plane_mode::screen ? "unused" : nearest_mode ? "midpoint_floor" : "explicit_plane"},
          {"word2_role", fraction_mode ? "front_limit_fraction" : "inverse_depth"},
          {"front_limit_fraction", fraction},
          {"resolved_value", nullptr},
          {"resolved_value_note", fraction_mode ? "Replay freezes the exact consumed front-limit fraction, including ramp values, and does not rerun the temporal adaptive classifier or its observational probe. The shader multiplies a finite fraction in [0,0.75] by the authoritative current positive display bound; invalid values fall back to screen disparity." : shallow_mode ? "No depth reduction. UI parallax is one quarter of the current positive b0 display bound after strength, stereo blend and warp-readiness guards; depth/gain/zero do not place this plane." : front_mode ? "No depth reduction. UI parallax is the current positive b0 display bound times strength and stereo blend when warp is active; depth/gain/zero do not place this plane." : nearest_mode ? "GPU result; no CPU readback in the live producer. Replay recomputes it from the exact captured depth, alpha, crop/jitter and constants." : "No depth reduction. Replay uses the exact captured UI mode and inverse-depth word."}};
      }
      // Shaders with limiter line groups pin UI in SunshineApplyUICS after the
      // complete scene field, so the horizontal limiter reads no UI input; older
      // ones pin inside the horizontal pass. A mode-5 probe frame observes the
      // unpinned field in between.
      const auto ui_input = external_ui ? "ui_source_color" : "source_color";
      if (shader_marker(captured_shader, "SUNSHINE_LIMITER_LINE_GROUPS")) {
        auto &passes = result["passes"];
        for (auto pass = passes.begin(); pass != passes.end(); ++pass) {
          if ((*pass)["entry"] != "SunshineHostHorizontalCS") continue;
          auto apply = json {{"entry", "SunshineApplyUICS"}, {"target", "cs_5_0"}, {"enabled", conditioning && f.source_alpha_ui},
            {"srvs", {{"t0", ui_input}}}, {"uavs", {{"u3", "final_field:R32_FLOAT (pinned in place)"}}}};
          if ((*pass)["srvs"].contains("t9")) apply["srvs"]["t9"] = (*pass)["srvs"]["t9"];
          (*pass)["srvs"].erase("t0");
          (*pass)["srvs"].erase("t9");
          auto next = passes.insert(pass + 1, apply);
          if (captured_shader.find("#define SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE 1") != std::string_view::npos)
            passes.insert(next, json {{"entry", "SunshineUIConflictCS"}, {"target", "cs_5_0"},
              {"enabled", conditioning && f.source_alpha_ui && fraction_mode},
              {"frequency", "mode-5 probe frames only; replay freezes the fraction and does not run it"},
              {"srvs", {{"t0", ui_input}, {"t5", "final_field before UI pinning"}}},
              {"uavs", {{"u7", "ui_conflict_statistics:R32G32B32A32_UINT 16x32, live readback only"}}}});
          break;
        }
      } else if (external_ui) {
        for (auto &pass : result["passes"]) {
          if (pass["entry"] == "SunshineHostHorizontalCS") pass["srvs"]["t0"] = "ui_source_color";
        }
      }
      const auto &detection = f.ui_detection;
      result["ui_detection"] = {{"ran_or_held", name(detection.state)}, {"candidates", detection.candidates},
        {"threshold_bits", detection.threshold_bits}, {"accepted", detection.accepted}, {"flags", detection.flags},
        {"pre_ui_threshold_bits", detection.pre_ui_threshold_bits}, {"still_bits", detection.still_bits},
        {"rules_bits", detection.rules_bits}, {"layer_presents_ago", detection.layer_presents_ago},
        {"layer_pairing", std::string(change_set::name(detection.layer_pairing))},
        {"retained_present_offsets", {{"retained_present_1", f.resources.retained_offsets[0]},
          {"retained_present_2", f.resources.retained_offsets[1]}}},
        {"held_presents", detection.held_presents}, {"candidate_layout", ui_detection::candidate_layout},
        {"meaning", "b2 constants of the automatic UI detection run whose mask this render consumed: it ran in this render from this real frame's own offered candidates, was held (a generated Present showing the decision of the real frame it shows, held_presents generated Presents in a row; UI framework T1), or was inactive (all zero). candidates and accepted are candidate bits of candidate_layout 2 (0x1 UIAlpha, 0x2 UI color tag, 0x4 Backbuffer, 0x8 current, 0x10 HUD-less, 0x20 exact pair, 0x40 offscreen UI layer, 0x100 the pre-UI change set: the layer proven the pre-UI scene image, paired with a retained Present, fix 3); accepted is the session's accepted candidates pushed in b2 word 2. flags is the full pushed Sunshine_UIDetectionFlags word: its stored bits describe the offscreen UI layer slot, and per-frame bits are included (0x40000: the depth was not this frame's; 0x100000: an accepted candidate the previous adopting real frame offered is missing; 0x200000: no previous real decision exists in this chain, so the T1 grace cannot reuse one; 0x400000: the hidden-scene guard holds a hidden verdict of D on the presented frame; 0x800000: its samples read the pre-UI scene image visible; bits 24-30: the candidate bits shifted left by 24 whose source signature a visible verdict refuted, so their full claims do not act (H1); 0x80000000: the offered layer's signature is proven the pre-UI scene image (the session ledger's key pre_ui:<format>:<space>), so its claim (d) may act (H1 d); 0x10000 and 0x80000, the layer and HUD-less route holds before S2b, and 0x20000000, the layer measured beside a HUD-less image for its D proof before fix 1, are retired and never reused); threshold_bits is the float32 difference threshold (b2 word 1) and pre_ui_threshold_bits the float32 pair threshold of the offscreen UI layer and the presented color (b2 word 4, zero without a layer, when they are not comparable or on a frame that is not a detection sample, since the CPU reads texel 11 from samples only), at 8 times which the tiles pass counts the layer's pre-UI pixels. rules_bits is b2 word 5 (Sunshine_UIRules) and still_bits its bit 0x1, set while rule H2's run is active in SDR Auto and the session enables it (UIFlattenStillScreens=1), so a frame applying no source of its own that the T1 grace did not reuse is shown flat as source 11 (a still screen without a UI source); zero otherwise, the default shadow included. Fix 3's rule bits: 0x2 pin only UI (Auto with UIPinOnlyUI=1: refine, a valid exact selective change set decides where S1's winner is an accepted alpha opaque on every pixel, and with 0x100 rule P2: the mask pass leaves the decided source's pure darkening unpinned), 0x100 that the darkening passes ran (fix 4, rule P2: every Auto detection frame with UIPinOnlyUI=1, sample frames in its shadow), 0x200 that the UI color tag has a float format (linear colour, rule P2's linear tolerance), bits 4-5 the Present offset the offscreen layer copy is compared with (0 the current Present, 1 and 2 the retained Presents one and two back at t2 and t3), 0x40 and 0x80 that those are bound. layer_presents_ago is the Presents since the offered layer copy was taken (it holds the frame of that Present), and layer_pairing its pairing: retained (exact by Present counting, frame generation off), late (frame generation active, against the current Present), unavailable (that Present not retained) or none (no layer). The retained Presents and the consumed layer copy are the optional artifacts retained_present_1, retained_present_2 and ui_layer_detected, at the offsets retained_present_offsets names; ui_detection_replay binds them by its case labels (layer_pair). ui_detection_replay pushes still from this field (rules_bits when present) with its own change-set bits and reruns detection from the package's candidates as a single frame without a previous decision: a render whose mask the T1 grace reused replays as its own decision (no mask, with its reason)."}};
      result["ui_pin"] = {{"soft_pin_gain", shader_marker(captured_shader, "SUNSHINE_UI_SOFT_PIN_GAIN")},
        {"decision_texels", shader_marker(captured_shader, ui_detection::decision_texels_marker)},
        {"evidence_images", shader_marker(captured_shader, ui_detection::scene_evidence_images_marker)},
        {"meaning", "Markers of the captured shader; 0 means absent: binary pinning and the 5-texel detection decision without scene-evidence images. A shader with 7 decision texels and 2 evidence images writes the presented and pre-UI scene image's hidden-scene evidence in texels 5 and 6 (before selection revision 3 the HUD-less image's); with 8 (candidate layout 2) texel 7 holds the offscreen UI layer's counts and the valid candidates; with 10 (selection revision 2) texel 8 holds the one-way strong counts and the refused candidate and texel 9 the one-way contradicted counts and the frame reason; with 11 (selection revision 3) texel 10 holds the opaque Backbuffer and current pixel counts, the informative full claims and the h1 word (H1); with 12 (selection revision 4) texel 11 holds the offscreen UI layer against the presented frame (matching, lit layer, lit presented, and lit presented but different pixels; H1 d); with 13 (selection revision 5) texel 12 holds rule H2's still and compared cells of the D grid (zero unless the evidence passes ran); with 16 (selection revision 6, fix 3) texels 13-15 hold the change-set shadow of the offscreen layer against its pair's Present on sample frames with an offered layer (changed, unchanged, non-finite, matching tiles; changed pixels the 3x3 rule keeps, changed against the Presents one and two back, the judge's pixels; judge pixels the 3x3 rule keeps and the judge kind); and texel 0 is the applied decision, which the T1 grace may have reused from the previous real frame (docs/reshade-sbs.md)."}};
      result["source_alpha_ui_requested"] = f.source_alpha_decision.requested;
      result["source_alpha_input_state"] = name(f.source_alpha_decision.input_state);
      const auto &review = f.source_alpha_decision.qualification;
      const auto &scope = review.candidate;
      result["ui_source_detection"] = {
        {"selected", ui_qualification::name(review.selected)}, {"source", ui_qualification::name(scope.source)},
        {"revision", review.token}, {"available", review.available}, {"automatic", true},
        {"scope", {{"runtime", scope.runtime}, {"device", scope.device}, {"provider", scope.provider},
          {"source_id", scope.source_id}, {"epoch", scope.epoch}, {"revision", scope.revision}, {"viewport", scope.viewport},
          {"fg_known", scope.fg_known}, {"fg_enabled", scope.fg_enabled}, {"fg_automatic", scope.fg_automatic},
          {"fg_generated_frames", scope.fg_generated_frames}, {"semantic_contract", scope.semantic_contract},
          {"output_width", scope.output_width}, {"output_height", scope.output_height}, {"output_format", scope.output_format},
          {"color_space", scope.color_space}, {"mask_width", scope.mask_width}, {"mask_height", scope.mask_height},
          {"mask_format", scope.mask_format}, {"channel", scope.channel},
          {"left", scope.left}, {"top", scope.top}, {"width", scope.width}, {"height", scope.height}}},
        {"meaning", "Automatic source discovery; no human approval. Availability and current GPU selection are separate. Candidate metadata identifies inspected inputs; asynchronous status does not label the exact current-frame winner. Replay uses the frozen resolved mask."}};
      result["source_alpha_capture_attempt"] = f.ui_capture_attempt_metadata.empty() ? json(nullptr) :
        json::parse(f.ui_capture_attempt_metadata);
      result["source_alpha_ui_status"] = f.source_alpha_ui ?
        (f.source_alpha_decision.mode == source_alpha_mode::automatic ? "automatic_gpu_mask" : external_ui ? "captured_alpha" : "present_alpha") :
        f.source_alpha_decision.blocked_by_fg() ? "unavailable_fg_output_alpha" :
        f.source_alpha_decision.automatic ? name(f.source_alpha_decision.coverage.state) : "disabled";
      if (f.source_alpha_decision.automatic) {
        const auto &coverage = f.source_alpha_decision.coverage;
        result["source_alpha_auto"] = {{"state", name(coverage.state)}, {"enabled", coverage.enabled}, {"accepted_samples", coverage.accepted_samples},
          {"covered_pixels", coverage.covered}, {"total_pixels", coverage.pixels},
          {"sample_sequence", coverage.sample_sequence}, {"sample_tick_ms", coverage.sample_tick_ms},
          {"sampled_source", coverage.source_kind},
          {"sampled_evidence", {{"candidates", coverage.evidence.candidates},
            {"alpha_covered", coverage.evidence.alpha_covered}, {"alpha_invalid", coverage.evidence.alpha_invalid},
            {"layer", {{"covered", coverage.evidence.layer_covered}, {"invalid", coverage.evidence.layer_invalid},
              {"opaque", coverage.evidence.layer_opaque}}},
            {"accepted", coverage.evidence.accepted}, {"valid_bits", coverage.evidence.valid_bits},
            {"hudless_changed", coverage.evidence.hudless_changed}, {"hudless_unchanged", coverage.evidence.hudless_unchanged},
            {"hudless_invalid", coverage.evidence.hudless_invalid}, {"matching_tiles", coverage.evidence.matching_tiles},
            {"hudless_lit", coverage.evidence.hudless_lit}, {"alpha_opaque", coverage.evidence.alpha_opaque},
            {"inferred_opaque", coverage.evidence.inferred_opaque}, {"claims", coverage.evidence.claims},
            {"h1", {{"applied", coverage.evidence.h1_applied}, {"winner", coverage.evidence.s1_source},
              {"refined", coverage.evidence.refined}}},
            {"scene", scene_json(coverage.evidence.scene, true)},
            {"pre_ui_scene", pre_ui_scene_json(coverage.evidence.pre_ui_scene, coverage.evidence.pre_ui_image)},
            {"shadow_hidden_ms", coverage.evidence.shadow_hidden_ms},
            {"still_short_ms", coverage.evidence.still_short_ms},
            {"pre_ui_pixels", {{"match", coverage.evidence.pre_ui_match}, {"image_lit", coverage.evidence.pre_ui_image_lit},
              {"presented_lit", coverage.evidence.presented_lit},
              {"presented_lit_differs", coverage.evidence.presented_lit_differs}}},
            {"one_way", {{"strong", {{"ui_layer", coverage.evidence.strong[0]}, {"backbuffer", coverage.evidence.strong[1]},
                {"current", coverage.evidence.strong[2]}}},
              {"contradicted", {{"ui_layer", coverage.evidence.contradicted[0]},
                {"backbuffer", coverage.evidence.contradicted[1]}, {"current", coverage.evidence.contradicted[2]}}},
              {"late_layer", coverage.evidence.late_layer}}},
            {"reason", std::string(ui_selection::frame_reason_name(coverage.evidence.frame_reason))},
            {"refused", std::string(ui_selection::candidate_name(coverage.evidence.refused))},
            {"reused", coverage.evidence.reused},
            {"change_set", {{"changed", coverage.evidence.change_set.changed},
              {"unchanged", coverage.evidence.change_set.unchanged}, {"nonfinite", coverage.evidence.change_set.nonfinite},
              {"matching_tiles", coverage.evidence.change_set.matching_tiles},
              {"filtered", coverage.evidence.change_set.filtered}, {"changed_1", coverage.evidence.change_set.changed_1},
              {"changed_2", coverage.evidence.change_set.changed_2},
              {"judge", std::string(change_set::judge_name(coverage.evidence.change_set.judge_kind))},
              {"judge_pixels", coverage.evidence.change_set.judge_pixels},
              {"judge_tp", coverage.evidence.change_set.judge_tp}}}}},
          {"scene_guard", {{"hidden", coverage.scene_guard.hidden}, {"pre_ui", coverage.scene_guard.pre_ui},
            {"refuted", coverage.scene_guard.refuted}, {"proven", coverage.scene_guard.proven}}}, {"scene_shadow", coverage.scene_shadow},
          {"still_screen", {{"scope", coverage.still.scope}, {"enabled", coverage.still.enabled},
            {"phase", still_screen::name(coverage.still.phase)}, {"run_ms", coverage.still.run_ms},
            {"sampled_still", coverage.evidence.still_cells}, {"sampled_compared", coverage.evidence.still_compared}}},
          {"change_set", {{"measured", coverage.change_set.measured},
            {"pairing", std::string(change_set::name(coverage.change_set.pairing))},
            {"offset", coverage.change_set.offset}, {"valid", coverage.change_set.valid},
            {"would_refine", coverage.change_set.would_refine}, {"enabled", coverage.change_set.enabled}}},
          {"darkening", {{"measured", coverage.darkening.measured}, {"applied", coverage.darkening.applied},
            {"unpinned", coverage.darkening.unpinned}, {"kept", coverage.darkening.kept},
            {"colourless", coverage.darkening.colourless}}},
          {"meaning", "Auto selects and validates current-frame candidates on the GPU without review. These bounded asynchronous statistics describe a completed earlier detection sample and never authorize current pixels. ui_source_color contains the actual resolved red-channel mask for this frame, possibly empty. Replay uses that frozen mask without rerunning detection. sampled_source 2 is the UI color tag, 10 the offscreen UI layer, 8 a full-frame UI over a hidden scene (H1: an informative full claim while the hidden-scene guard holds a hidden verdict), 11 a still screen without a UI source shown flat (H2: no source of its own, not reused by T1, while rule H2's run is active and UIFlattenStillScreens=1), and 12 the changed pixels of the proven pre-UI layer against the retained Present it shows (fix 3, UIPinOnlyUI=1; h1.refined when it replaced a shapeless whole-frame alpha winner, as may 5); 7 and 9 (the HUD-less route before S2b) are retired. sampled_evidence.alpha_covered and alpha_invalid hold UIAlpha, the UI color tag, Backbuffer and current; layer holds the offscreen UI layer's covered, invalid (out of range or beyond the premultiplied bound) and opaque pixels; accepted is the accepted candidates pushed with that detection and valid_bits the offered candidates that passed V1/V2, both in candidate bits; sampled_evidence.alpha_opaque counts alpha of at least 254/255 in UIAlpha and the UI color tag and inferred_opaque in Backbuffer and current alpha; claims are the raw informative full claims before refutation (candidate bits, 0x80 the pre-UI scene image: a HUD-less image changed on 90% of pixels, or an offered layer without coverage whose signature is proven) and h1 the S1 winner's source and whether H1 overrode it with 8; scene and pre_ui_scene are that sample's hidden-scene evidence of the presented frame and of the pre-UI scene image (image none, hudless or layer) (n edge cells, D, valid, ran, the presented verdict with its decided, untied comparisons, and the pre-UI image's verdict read from D), and shadow_hidden_ms how long consecutive samples up to it read the presented frame hidden while no source decided and the frame was not blank (at least 128 decided comparisons), and still_short_ms the length of a run of still hidden samples without a decided source that this sample ended before it reached 2 s (rule H2's gameplay-safety evidence; zero otherwise). pre_ui_pixels holds the offscreen UI layer against the presented frame at 8 times their pair threshold (decision texel 11; all zero without a layer or a comparable pair): match, the pixels whose colours match, and image_lit, the lit layer pixels, from which the session ledger proves the layer's signature the pre-UI scene image (3 samples over 2 s with match on at least 90% and image_lit on at least half of the pixels while the layer has no coverage); presented_lit, the lit presented pixels, and presented_lit_differs, those of them that differ from the layer, are shadow statistics nothing acts on. one_way holds the A2 one-way judgment counts of the inferred alpha kinds: strong pixels (finite alpha of at least 1/2) and those of them where an offered exact HUD-less pair is lit and unchanged (contradicted); neither is counted for the one-frame-late layer copy (late_layer), which no A2 judge reads. sampled_source and covered are the applied decision; reason names the sample's own decision (decided, or the ui_no_mask reason without a mask), refused the highest-ranked candidate that reason refused (none when it decided), and reused whether the T1 grace applied the previous real frame's decision and mask instead. scene_guard and scene_shadow describe this render: the hidden-scene guard's pushed verdicts (hidden: a held hidden verdict; pre_ui: its samples read the pre-UI image visible) and how many source signatures it holds refuted, proven whether the offered layer's signature is proven the pre-UI scene image (the session ledger's key pre_ui:<format>:<space>, so that its image may act as H1 (d)), and the first-run shadow that measures without an acting claim (docs/reshade-sbs.md, hidden-scene evidence). still_screen describes rule H2 for this render (docs/reshade-sbs.md, still screens without a UI source): scope (SDR output in Auto), enabled (UIFlattenStillScreens=1; otherwise a shadow that only logs), the run's phase (none, pending, or active: the screen would flatten, or flattens when enabled) and run_ms, its length; sampled_still and sampled_compared are the sample's texel 12, the D grid cells whose presented-luma mean stayed within 1/255 of the previous measured sample's and the cells compared. sampled_evidence.change_set is that sample's change-set shadow (texels 13-15, fix 3): the offscreen layer against the Present of its pairing at 8 times their pair threshold (changed beyond it, unchanged within half), non-finite pixels, tiles at least 99% unchanged, changed pixels the 3x3 rule keeps (at least 3 changed in the window), changed pixels against the Presents one and two back when retained, and the judge (the first offered and accepted declared alpha) with its pixels and those the 3x3 rule keeps; change_set describes it: whether the sample measured the layer, its pairing and offset, whether the pair passed the pre-UI change set's validity and would refine a shapeless alpha with UIPinOnlyUI=1, and enabled, this render's switch. darkening is that sample's rule P2 (fix 4, pin only UI; decision words 62-63): whether its darkening passes measured its eligible decided source (2, 5, 10 or 12; not on a frame T1 reused), the darkening pixels unpinned (or that would be, in the shadow) and those kept pinned (sharp structure, a change set's darkening region without a proven tile or with a colour verdict, or every one of an opacity source without colour on the frame), whether the mask pass applied it (UIPinOnlyUI=1), and colourless, an opacity source without colour on the frame. Nothing decides from the shadow and the acceptance ledger never reads it."}};
      }
      const auto &fg = f.source_alpha_decision.fg;
      result["source_alpha_ui_fg_mode"] = {{"known", fg.known}, {"enabled", fg.enabled},
        {"automatic", fg.automatic}, {"generated_frames", fg.generated_frames}, {"epoch", fg.epoch},
        {"sequence", fg.sequence}, {"viewport", fg.viewport},
        {"meaning", "Last confirmed game-requested FG mode, retained during transient observation loss. Independent of selected depth; does not identify this presentation as real or generated."}};
      const auto ui = ui_parameter_words(f.source_alpha_ui, f.ui_plane, f.ui_channel);
      result["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v7";
      result["ui_parameter_bytes"] = sizeof(ui);
      result["ui_parameter_hex"] = parameter_hex(ui);
      result["ui_parameter_encoding"] = fraction_mode ? "little-endian exact b1 bytes: uint32 enabled, uint32 mode, IEEE754 float32 resolved front-limit fraction, uint32 mask channel (0=alpha,1=red)" :
        "little-endian exact b1 bytes: uint32 enabled, uint32 mode, IEEE754 float32 inverse depth, uint32 mask channel (0=alpha,1=red)";
      result["ui_constant_binding"] = {{"register", "b1"}, {"uint32", ui},
        {"mask_channel", f.ui_channel == ui_mask_channel::red ? "red" : "alpha"},
        {"mode", static_cast<std::uint32_t>(f.ui_plane.mode)},
        {"mode_name", f.ui_plane.mode == ui_plane_mode::screen ? "screen" : f.ui_plane.mode == ui_plane_mode::depth_midpoint ? "depth_midpoint" :
          nearest_mode ? "depth_midpoint_nearest_ui" : front_mode ? "front_limit" : shallow_mode ? "shallow_front" : fraction_mode ? "display_fraction" : "unknown"},
        {"inverse_depth", std::isfinite(f.ui_plane.inverse_depth) ? json(f.ui_plane.inverse_depth) : json(nullptr)},
        {"inverse_depth_bits", ui[2]},
        {"inverse_depth_role", fraction_mode || shallow_mode || front_mode || f.ui_plane.mode == ui_plane_mode::screen ? "unused" : nearest_mode ? "midpoint_floor" : "explicit_plane"},
        {"word2_role", fraction_mode ? "front_limit_fraction" : "inverse_depth"},
        {"front_limit_fraction", fraction},
        {"meaning", "When source_alpha_ui is enabled, alpha from ui_alpha_source supplies UI coverage. The UI pinning pass (SunshineApplyUICS, or the horizontal pass of older shaders) reads that input at t0; eye RGB remains current source_color. Mode 4 pins UI at one quarter of the current positive display bound after strength, stereo blend and warp-readiness guards. Mode 3 uses that full bound. Both fixed modes ignore scene depth/gain/zero and their inverse-depth word is unused. Screen mode pins at zero disparity. Mode 1 uses the submitted independent depth. Mode 2 reduces max(submitted midpoint floor, nearest valid decoded depth under finite positive alpha). Both depth modes use b0 geometry. Pinning is a band around the UI plane: mask alpha at or above 1/replay.ui_pin.soft_pin_gain pins exactly, fainter alpha in proportion, with a one-texel collar for the bilinear color footprint and a 0.5/source_width ramp beyond it; a shader without that marker pins every finite positive alpha exactly. With protection enabled all-white masks are entirely UI; all-black masks have no UI constraints."}};
      if (fraction_mode) {
        auto &binding = result["ui_constant_binding"];
        binding.erase("inverse_depth");
        binding.erase("inverse_depth_bits");
        binding["front_limit_fraction_bits"] = ui[2];
        binding["meaning"] = "Mode 5 consumes the exact resolved display fraction from b1 word2, multiplied by the authoritative current positive display bound. Finite fractions in [0,0.75] are valid; invalid values fall back to screen disparity. The word is not an inverse depth. Replay freezes it and does not run the temporal adaptive classifier. UI coverage and RGB input semantics are unchanged.";
      }
      if (f.ui_adaptive) {
        const auto &a = *f.ui_adaptive;
        result["ui_adaptive"] = {{"levels_uv", ui_adaptive::levels_uv}, {"target_index", a.target_index},
          {"count_scope", "center_75_percent_width_and_height"}, {"entry_ratio", double(ui_adaptive::entry_conflict_percent) / 100.},
          {"conflict_aggregation", "ui_ratio_or_center_area"}, {"center_pixels", a.center_pixels},
          {"entry_area_ratio", double(ui_adaptive::entry_area_per_mille) / 1000.},
          {"release_area_ratio", double(ui_adaptive::release_area_per_mille) / 1000.},
          {"entry_comparison", "strictly_greater"}, {"release_ratio", double(ui_adaptive::release_conflict_percent) / 100.},
          {"required_index", a.required_index}, {"applied_fraction", std::isfinite(a.applied_fraction) ? json(a.applied_fraction) : json(nullptr)},
          {"placement", "absolute_uv_levels"}, {"applied_uv", a.applied_uv}, {"target_uv", a.target_uv},
          {"required_uv", a.required_uv}, {"limit_uv", a.limit_uv},
          {"observed_front_cap_uv", a.observed_front_cap_uv},
          {"comfort_cap_uv", ui_adaptive::comfort_cap_uv}, {"max_live_fraction", ui_adaptive::max_live_fraction},
          {"capped_conflict", a.capped_conflict}, {"probe_age_ms", a.probe_age_ms},
          {"accepted_sequence", a.accepted_sequence}, {"accepted_tick_ms", a.accepted_tick},
          {"accepted_mask_sequence", a.accepted_mask_sequence}, {"covered_pixels", a.covered_pixels},
          {"conflict_pixels", a.conflict_pixels}, {"invalid_pixels", a.invalid_pixels},
          {"conflict_counts", a.conflict_counts}, {"status", a.status ? a.status : "unknown"},
          {"meaning", "Optional snapshot of live absolute-disparity placement. Counts aggregate central 75% width and height. Entry occurs above 20% of covered UI OR 2% of central image pixels; retreat requires below 15% of covered UI AND 1.5% of central image pixels. Each absolute level is clipped to half observed_front_cap_uv. Counts belong to the accepted sample, not necessarily this render. Exact consumed mode5 fraction is frozen in b1; policy history is not a replay input."}};
      }
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
                                {"raw_depth", "Full allocation of unfiltered float32 depth samples. Device or linear-distance encoding and active rectangle belong to consumed_depth; no normalization or clamp applied."},
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
                            {"projection", {{"supplied", d.projection.supplied}, {"encoding", d.projection.encoding == sunshine_scene_depth::depth_encoding::linear_distance ? "linear_distance" : "device"}, {"epoch", d.projection.epoch}, {"viewport", d.projection.viewport}, {"A", d.projection.A}, {"B", d.projection.B}, {"raw_scale", d.projection.raw_scale}, {"raw_bias", d.projection.raw_bias}}},
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
        result["artifact_semantics"]["ui_source_color"] = "Exact typed texture consumed for UI coverage. b1 word3 selects alpha (RGBA) or red (single-channel mask). An automatic mask is the selected source's raw alpha, the one-frame-late offscreen UI layer's (replay.ui_detection.flags bit 0x4) included (docs/reshade-sbs.md, offscreen UI layer). A full-frame decision (sources 6, 8 and 11) is all 1.0; 9 is retired. The pre-UI change set (source 12) is binary: the layer copy changed beyond 8 times its pair threshold against the retained Present it shows, kept where at least 3 pixels of the 3x3 window changed. Other channels never replace current source_color for eye rendering. This does not prove same-game-frame pairing.";
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
      const bool distance = camera || scale.basis == automatic_scale_basis::linear_distance;
      auto &state = result["render_scene_policy"];
      state = {{"phase", static_cast<unsigned>(f.scene_status.phase)}, {"basis", camera ? "camera_matrix" : distance ? "linear_distance" : scale.basis == automatic_scale_basis::relative_depth ? "relative_depth" :
                                                                                                                                                                   "unknown"},
               {"active", scale.active},
               {"zero_plane_units", distance ? "game units, not necessarily meters" : "relative raw depth, not distance"},
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
        {"units", distance ? "inverse game units" : "relative inverse depth"},
        {"note", "Accepted scene-policy measurement for this captured rendering decision, not a new measurement of its pixels. Live Q, mean, second moment and bounds include every active-depth pixel. Larger q is nearer; these are not camera clipping planes. The uncentered mean_square_q is diagnostic and does not set gain. Centered moments determine the trial zero target. The UI may retain older values during a missing-depth hold when this decision has no statistics."}
      } : nlohmann::json(nullptr);
      state["zero_plane_policy"] = "trial contrast midpoint: b=qmin, d=q-b, q0 target=b+0.5*sum(d*d)/sum(d); stable centered full-pixel moments, fixtures without centered moments retain midpoint; exponential tracking bounded by abs(Knew*delta_q0/L)<=zero_budget_per_second*elapsed_seconds; no immediate applied-zero range clamp; positive flat depth targets b while holding gain, all-zero depth holds both";
      state["zero_tracking"] = {{"time_constant_seconds", sunshine_camera_scene::time_constant_seconds},
        {"zero_budget_per_second", sunshine_scene_gain::zero_budget_per_second}};
      state["gain_below_target"] = scale.gain_below_target;
      state["zero_inverse"] = scale.has_zero ? nlohmann::json(scale.zero_inverse) : nlohmann::json(nullptr);
      state["ui_midpoint_inverse"] = scale.has_ui_midpoint ? nlohmann::json(scale.ui_midpoint_inverse) : nlohmann::json(nullptr);
      state["target_ui_midpoint_inverse"] = scale.has_ui_midpoint_target() ? nlohmann::json(scale.target_ui_midpoint_inverse) : nlohmann::json(nullptr);
      state["ui_midpoint_policy"] = "Live mode 5 consumes an independently resolved display fraction; midpoint, gain and zero do not directly place it. The independently smoothed inverse-depth midpoint target=(qmin+qmax)/2 remains diagnostic state. Historical mode 1 maps the submitted midpoint through current geometry; mode 2 uses it as a covered-depth floor; mode 3 uses the full positive display bound; mode 4 uses one quarter. Exact consumed mode and words are in replay b1; replay does not rerun adaptive history.";
      state["zero_plane_at_infinity"] = scale.has_zero && distance && scale.zero_inverse == 0.f;
      state["current_zero_plane"] = scale.has_zero && (!distance || scale.zero_inverse > 0.f) ?
        nlohmann::json(distance ? 1.0 / scale.zero_inverse : double(scale.zero_inverse)) : nlohmann::json(nullptr);
      state["target_zero_inverse"] = scale.has_zero_target() ? nlohmann::json(scale.target_zero_inverse) : nlohmann::json(nullptr);
      state["target_zero_plane_at_infinity"] = scale.has_zero_target() && distance && scale.target_zero_inverse == 0.;
      state["target_zero_plane"] = scale.has_zero_target() && (!distance || scale.target_zero_inverse > 0.) ?
        nlohmann::json(distance ? 1.0 / scale.target_zero_inverse : scale.target_zero_inverse) : nlohmann::json(nullptr);
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

    // Adds each copied candidate UI layer as an optional artifact and describes
    // every qualifying target the armed census saw (docs/reshade-sbs.md).
    nlohmann::json ui_layer_census(capture_batch &batch, api::device *device) {
      api::device *owner = nullptr;
      auto layers = ui_layer::take(owner);
      auto rows = nlohmann::json::array();
      for (unsigned i = 0; i < layers.size() && i < wire::ui_layer_count; ++i) {
        auto &c = layers[i];
        const auto kind = static_cast<wire::artifact>(static_cast<unsigned>(wire::artifact::ui_layer_0) + i);
        std::string status = c.status;
        bool added = false;
        if (c.copy.handle) {
          added = owner == device && batch.add(kind, c.copy);
          if (!added) status = owner == device ? "transport_budget_exceeded" : "other_device";
        }
        char source[24];
        std::snprintf(source, sizeof(source), "0x%llx", static_cast<unsigned long long>(c.source));
        rows.push_back({{"artifact_id", static_cast<unsigned>(kind)}, {"kind", wire::ui_layer_names[i]}, {"captured", added},
          {"status", status}, {"source", source}, {"width", c.width}, {"height", c.height}, {"dxgi_format", c.format},
          {"clears_while_armed", c.clears}, {"active", c.active}});
        // add() holds its own reference; the add-on's handle is released once
        // any game command list that wrote the copy has executed.
        ui_layer::retire(owner, c.copy);
      }
      return {{"meaning", "Output-resolution color targets the game cleared to transparent black while this request was armed, "
        "the signature of an offscreen UI layer. Each copy was taken before a clear, so it shows the previous frame's content. "
        "active marks the target the live tracker chose; without a tagged UI buffer its copy is UI detection's color+alpha "
        "candidate, admitted only while premultiplied. The others are candidates only, and none is verified UI."},
        {"candidates", std::move(rows)}};
    }
    // Fix 3: the retained Presents the pre-UI change set pairs with and the
    // layer copy detection consumed, best effort after the replay inputs
    // (the optional catalog later reports its own budget errors).
    nlohmann::json change_set_artifacts(capture_batch &batch, const diagnostic_resources &r) {
      const std::array<std::pair<wire::artifact, api::resource>, 3> artifacts{{
        {wire::artifact::retained_present_1, r.retained_presents[0]},
        {wire::artifact::retained_present_2, r.retained_presents[1]},
        {wire::artifact::ui_layer_detected, r.ui_layer_detected},
      }};
      auto rows = nlohmann::json::array();
      for (std::size_t i = 0; i != artifacts.size(); ++i) {
        const auto [kind, resource] = artifacts[i];
        const bool captured = resource.handle && batch.add(kind, resource);
        nlohmann::json row{{"artifact_id", static_cast<unsigned>(kind)},
          {"kind", wire::change_set_name(static_cast<unsigned>(kind))}, {"captured", captured},
          {"status", !resource.handle ? "not_retained" : captured ? "captured" : "transport_budget_exceeded"}};
        if (i < r.retained_offsets.size()) row["presents_ago"] = r.retained_offsets[i];
        rows.push_back(std::move(row));
      }
      return {{"meaning", "Fix 3 (docs/reshade-sbs.md, UI decision framework): retained_present_1 and retained_present_2 are the "
        "presented colors the renderer retained one and two Presents before this render (the pairs of the offscreen layer "
        "copy, which holds the frame of the Present replay.ui_detection.layer_presents_ago back), ui_layer_detected the "
        "layer copy this render's detection consumed (t7). An armed Dump 3D retains every Present, and the dumped render "
        "retains its own only after these copies. not_retained: that Present was not retained, or no layer was offered."},
        {"artifacts", std::move(rows)}};
    }
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
        ui_layer::cancel();
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
          ui_layer::arm();
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
    ui_layer::cancel();
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
      auto metadata = frame_json(runtime, frame);
      const auto &r = frame.resources;
      if (frame.source_alpha_ui && r.ui_source.handle && !next->add(wire::artifact::ui_source_color, r.ui_source)) {
        throw std::runtime_error("Cannot snapshot the exact consumed UI-alpha source");
      }
      if (!r.source.handle || !r.sbs.handle || !next->add(wire::artifact::source_color, r.source) || !next->add(wire::artifact::linear_color, r.linear_color) || !next->add(wire::artifact::candidate, r.candidate) || !next->add(wire::artifact::vertical_majorant, r.vertical_majorant) || !next->add(wire::artifact::vertical_field, r.vertical_field) || !next->add(wire::artifact::final_field, r.final_field) || !next->add(wire::artifact::sbs, r.sbs)) {
        throw std::runtime_error("Cannot allocate the bounded shared diagnostic textures");
      }
      metadata["ui_layer_census"] = ui_layer_census(*next, runtime->get_device());
      metadata["change_set_artifacts"] = change_set_artifacts(*next, frame.resources);
      next->json = metadata.dump();
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

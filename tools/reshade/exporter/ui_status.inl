// SPDX-License-Identifier: GPL-3.0-only
// Part of exporter.cpp, included at the end of the same translation unit.
// UI and scene status publication. Session types and dynamically initialized
// automatic_ui references stay in the shell at their original positions.
namespace {
  // Returns the session whose UI counters the runtime had logged, if any.
  sunshine_game3d::alpha_auto_policy *clear_automatic_ui(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    if (found == automatic_ui.end()) return nullptr;
    const auto session = found->second.counters_logged ? found->second.ui_session : nullptr;
    automatic_ui.erase(found);
    return session;
  }

  // The game session's cumulative exact UI counters (docs/reshade-sbs.md, UI
  // counters), written with each UI protection line and once more when a
  // runtime that wrote them resets. Callers hold no lock: the snapshot takes
  // only the session's own mutex.
  void log_ui_counters(api::effect_runtime *runtime, sunshine_game3d::alpha_auto_policy &session) {
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "Sunshine UI counters: runtime=%p ", static_cast<void *>(runtime));
    const auto message = prefix + sunshine_game3d::format_ui_counters(session.counters());
    log(reshade::log::level::info, message.c_str());
  }

  void publish_automatic_ui(api::effect_runtime *runtime, sunshine_game3d::automatic_status status) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    auto &entry = automatic_ui[runtime];
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
    if (entry.recalibrate) return;
#endif
    if (status.phase != sunshine_game3d::automatic_phase::ready) status.scale.active = false;
    status.scale = sunshine_game3d::retain_automatic_scale(entry.status.scale, status.scale);
    entry.status = status;
  }

  void publish_source_alpha_ui(api::effect_runtime *runtime, sunshine_game3d::source_alpha_ui_decision value,
      bool game3d_enabled, sunshine_game3d::alpha_auto_policy *ui_session) {
    const auto now = GetTickCount64();
    bool write_log = false;
    sunshine_game3d::alpha_auto_policy *counters_session = nullptr;
    {
      std::lock_guard<std::mutex> lock(automatic_ui_mutex);
      auto &entry = automatic_ui[runtime];
      if (ui_session) entry.ui_session = ui_session;
      value.unprotected_since_ms =
        sunshine_game3d::next_unprotected_since(value, game3d_enabled, entry.source_alpha.unprotected_since_ms, now);
      entry.source_alpha = value;
      const auto &last = entry.logged_source_alpha;
      // Of the hidden-scene verdict only entering or leaving hidden counts:
      // matched scenes near the visible bound turn between ambiguous and
      // visible.
      const auto hidden = [](const sunshine_game3d::source_alpha_ui_decision &decision) {
        return decision.coverage.evidence.scene.verdict == sunshine_game3d::ui_detection::scene_verdict::hidden;
      };
      const bool changed = value.mode != last.mode || value.rendered != last.rendered || value.applied != last.applied ||
        value.input != last.input || value.input_state != last.input_state || value.fg.known != last.fg.known ||
        value.fg.enabled != last.fg.enabled || value.qualification.selected != last.qualification.selected ||
        value.qualification.available != last.qualification.available ||
        value.coverage.state != last.coverage.state || value.qualification.token != last.qualification.token ||
        value.coverage.source_kind != last.coverage.source_kind ||
        hidden(value) != hidden(last);
      write_log = entry.source_alpha_log.due(now, changed, false, true);
      if (write_log) {
        entry.logged_source_alpha = value;
        counters_session = entry.ui_session;
        entry.counters_logged |= counters_session != nullptr;
      }
    }
    if (!write_log) return;
    const auto &evidence = value.coverage.evidence;
    const auto &scene = evidence.scene, &pre_ui_scene = evidence.pre_ui_scene;
    // F1 and A2 (texels 8, 9 and 11): the one-way judgment counts of
    // Backbuffer and current alpha (lines through be7788bf led with the
    // layer's, 0 on every line: every layer was the never-judged late copy) and
    // of the declared alphas (sampled_declared_one_way: UIAlpha, UI color
    // tag), the sample's own-decision reason, the candidate it names and
    // whether the T1 grace reused a decision.
    const auto reason = sunshine_game3d::ui_selection::frame_reason_name(evidence.frame_reason);
    const auto refused = sunshine_game3d::ui_selection::candidate_name(evidence.refused);
    // H1 (texels 6 and 10): the pre-UI scene image texel 6 measured. H1 (d),
    // texel 11: the offscreen UI layer against the presented frame (matching
    // and lit layer pixels prove it the pre-UI scene image).
    namespace pre_ui_image = sunshine_game3d::ui_detection::pre_ui_image;
    const char *const pre_ui_name = evidence.pre_ui_image == pre_ui_image::hudless ? "hudless" :
      evidence.pre_ui_image == pre_ui_image::layer ? "layer" : "none";
    const auto &guard = value.coverage.scene_guard;
    char message[3072];
    std::snprintf(message, sizeof(message),
      "Sunshine UI protection: runtime=%p mode=%s rendered=%d mask_path=%d input=%s retained=%d fg=%d fg_known=%d fg_enabled=%d input_state=%s detection=%s selected=%s source=%s source_availability=%s sampled_source=%u sampled_covered=%u sampled_pixels=%u sampled_candidates=0x%x sampled_alpha_covered=%u/%u/%u/%u sampled_alpha_invalid=%u/%u/%u/%u accepted=0x%x sampled_layer={covered=%u invalid=%u opaque=%u} sampled_one_way={strong=%u/%u contradicted=%u/%u} sampled_reason=%.*s sampled_refused=%.*s sampled_reused=%d sampled_hudless={changed=%u unchanged=%u invalid=%u matching_tiles=%u lit=%u} sampled_alpha_opaque=%u/%u sampled_inferred_opaque=%u/%u sampled_claims=0x%x sampled_h1={applied=%d winner=%u} sampled_scene={n=%u d=%.3f valid=%d ran=%d verdict=%s} sampled_pre_ui_scene={image=%s n=%u d=%.3f valid=%d} scene_guard={hidden=%d pre_ui=%d refuted=%u proven=%d} sampled_pre_ui_pixels={match=%u image_lit=%u} sampled_declared_one_way={strong=%u/%u contradicted=%u/%u} status_revision=%llu",
      static_cast<void *>(runtime), value.mode == sunshine_game3d::source_alpha_mode::automatic ? "auto" :
        value.mode == sunshine_game3d::source_alpha_mode::on ? "on" : "off",
      int(value.rendered), int(value.applied), sunshine_game3d::name(value.input), int(value.retained_alpha_ready),
      int(value.fg_active()), int(value.fg.known), int(value.fg.enabled), sunshine_game3d::name(value.input_state),
      sunshine_game3d::name(value.coverage.state), sunshine_game3d::ui_qualification::name(value.qualification.selected),
      sunshine_game3d::ui_qualification::name(value.qualification.candidate.source), value.source_availability(),
      value.coverage.source_kind, value.coverage.covered, value.coverage.pixels, evidence.candidates,
      evidence.alpha_covered[0], evidence.alpha_covered[1], evidence.alpha_covered[2], evidence.alpha_covered[3],
      evidence.alpha_invalid[0], evidence.alpha_invalid[1], evidence.alpha_invalid[2], evidence.alpha_invalid[3],
      evidence.accepted, evidence.layer_covered, evidence.layer_invalid, evidence.layer_opaque,
      evidence.strong[2], evidence.strong[3], evidence.contradicted[2], evidence.contradicted[3],
      int(reason.size()), reason.data(), int(refused.size()), refused.data(), int(evidence.reused),
      evidence.hudless_changed, evidence.hudless_unchanged, evidence.hudless_invalid, evidence.matching_tiles, evidence.hudless_lit,
      evidence.alpha_opaque[0], evidence.alpha_opaque[1], evidence.inferred_opaque[0], evidence.inferred_opaque[1],
      evidence.claims, int(evidence.h1_applied), evidence.s1_source, scene.n, double(scene.d), int(scene.valid), int(scene.ran),
      sunshine_game3d::ui_detection::name(scene.verdict), pre_ui_name, pre_ui_scene.n, double(pre_ui_scene.d),
      int(pre_ui_scene.valid), int(guard.hidden), int(guard.pre_ui), guard.refuted, int(guard.proven),
      evidence.pre_ui_match, evidence.pre_ui_image_lit, evidence.strong[0], evidence.strong[1], evidence.contradicted[0],
      evidence.contradicted[1], static_cast<unsigned long long>(value.qualification.token));
    log(reshade::log::level::info, message);
    if (counters_session) log_ui_counters(runtime, *counters_session);
  }

#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
  bool take_recalibration(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    if (found == automatic_ui.end() || !found->second.recalibrate) return false;
    found->second.recalibrate = false;
    return true;
  }
#endif

}  // namespace

namespace sunshine_game3d {
  source_alpha_ui_decision query_source_alpha_ui(reshade::api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    return found == automatic_ui.end() ? source_alpha_ui_decision{} : found->second.source_alpha;
  }

  automatic_status query_automatic(reshade::api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    return found == automatic_ui.end() ? automatic_status{} : found->second.status;
  }

#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
  bool recalibrate_automatic(reshade::api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(automatic_ui_mutex);
    const auto found = automatic_ui.find(runtime);
    if (found == automatic_ui.end() || !found->second.status.can_recalibrate || found->second.recalibrate)
      return false;
    found->second.recalibrate = true;
    found->second.status.phase = automatic_phase::calibrating;
    found->second.status.can_recalibrate = false;
    found->second.status.scale.active = false;
    return true;
  }
#endif
}

// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_capture.h"
#include "game3d_diagnostic_metadata.h"
#include "streamline_depth_capture.h"
#include "../../src/game3d_debug_formats.h"

#include <array>
#include <atomic>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>

namespace sunshine_game3d {
  namespace capture = sunshine_streamline::depth_capture;
  namespace scene = sunshine_scene_depth;
  namespace wire = game3d_debug;
  namespace {
    constexpr unsigned resource_count = sizeof(ui_resources::catalog) / sizeof(ui_resources::catalog[0]);
    struct entry {
      diagnostic::resource_observation observed;
      capture::diagnostic_ticket ticket;
      capture::diagnostic_texture pixels;
      capture::record_diagnostic record;
      const char *status = "not_observed";
      unsigned attempts = 0;
      bool finished = false, successful = false;
    };
  }
  struct ui_capture_state {
    std::mutex mutex;
    std::uint64_t session, frozen_tick = 0;
    bool accepting = true, appended = false;
    std::array<entry, resource_count> entries;
    explicit ui_capture_state(std::uint64_t value): session(value) {}
    ~ui_capture_state() {
      // Native ownership independently retains all outstanding GPU work. These
      // releases never wait, even when the request was canceled before submit.
      for (auto &e : entries) capture::release_diagnostic_texture(e.ticket);
    }
  };
  namespace {
    std::shared_ptr<ui_capture_state> active;

    void resource(const diagnostic::resource_observation &observation) noexcept {
      try {
        auto state = std::atomic_load(&active);
        if (!state || state->session != observation.observation.session) return;
        std::lock_guard lock(state->mutex);
        if (!state->accepting) return;
        const auto *description = ui_resources::find(observation.artifact_id);
        if (!description) return;
        auto &e = state->entries[description - ui_resources::catalog];
        // One frozen capture per kind. Later API observations remain separately
        // available in ui_resources; never relabel older pixels as a newer tag.
        if (e.ticket) return;
        e.observed = observation;
        if (!observation.readable) { e.status = "unreadable_or_query_failed"; return; }
        if (!observation.descriptor_supported) { e.status = "unsupported_descriptor"; return; }
        if (!observation.native) { e.status = "explicit_null"; return; }
        if (!observation.observation.command) { e.status = "no_command_at_observation"; return; }
        if (e.attempts >= 4) { e.status = "attempt_limit"; return; }
        ++e.attempts;
        capture::input input;
        input.source = capture::retain_source(observation.native, &e.record);
        if (!input.source) { e.status = "source_unavailable"; return; }
        const auto &at = observation.observation;
        input.epoch = at.epoch; input.sequence = at.sequence; input.tick = at.tick;
        input.viewport = at.viewport;
        input.provider = description->source == ui_resources::provider::ngx ? scene::provider_kind::ngx : scene::provider_kind::streamline;
        input.source_id = observation.source_id;
        input.source_frame_token = at.frame_token; input.source_frame_numeric = at.frame_numeric;
        input.source_frame_has_numeric = at.numeric_frame;
        input.resource.native = observation.native;
        input.resource.area = {observation.area.left, observation.area.top, observation.area.width, observation.area.height};
        input.native_state = observation.state_declared ? observation.native_state : UINT32_MAX;
        input.proof = observation.state_declared ? scene::state_proof::declared : scene::state_proof::observed_nonzero;
        input.valid_until = scene::lifetime::at_call;
        input.force_snapshot = true;
        input.source_present_generation = capture::source_present_generation(input.source);
        e.ticket = capture::record_diagnostic_texture(at.command, input, &e.record);
        e.status = e.ticket ? "pending_gpu" : "capture_rejected";
      } catch (...) {
        // Diagnostic failure cannot propagate across a game's vendor API hook.
      }
    }

    void finish(ui_resources::provider provider, const diagnostic::stamp &at, std::uint64_t source_id, bool successful) noexcept {
      try {
        auto state = std::atomic_load(&active);
        if (!state || state->session != at.session) return;
        std::lock_guard lock(state->mutex);
        for (unsigned i = 0; i != resource_count; ++i) {
          auto &e = state->entries[i];
          const auto &before = e.observed.observation;
          if (ui_resources::catalog[i].source != provider || (provider == ui_resources::provider::ngx && e.observed.source_id != source_id) || !e.ticket || e.finished ||
              before.session != at.session || before.epoch != at.epoch || before.sequence != at.sequence || before.command != at.command) continue;
          e.finished = true; e.successful = successful;
          capture::finish_diagnostic_texture(e.ticket, successful);
        }
      } catch (...) {}
    }
    const diagnostic::resource_callbacks callbacks {resource, finish};

    std::string hex(std::uint64_t value) {
      std::ostringstream out;
      out << "0x" << std::hex << value;
      return out.str();
    }
    nlohmann::json rectangle(const scene::extent &value) {
      return {{"left", value.left}, {"top", value.top}, {"width", value.width}, {"height", value.height}};
    }
  }

  ui_capture_batch::ui_capture_batch(std::uint64_t session): state_(std::make_shared<ui_capture_state>(session)) {
    diagnostic::set_resource_callbacks(&callbacks);
    std::atomic_store(&active, state_);
  }
  ui_capture_batch::~ui_capture_batch() {
    auto expected = state_;
    std::atomic_compare_exchange_strong(&active, &expected, std::shared_ptr<ui_capture_state>{});
  }
  void ui_capture_batch::freeze() {
    std::lock_guard lock(state_->mutex);
    state_->accepting = false;
    if (!state_->frozen_tick) state_->frozen_tick = GetTickCount64();
  }
  bool ui_capture_batch::append(wire::response_t &response, std::uint64_t &bytes, std::string &json_text, bool allow_pixels) {
    std::lock_guard lock(state_->mutex);
    if (state_->appended) return true;
    try {
      auto candidate = response;
      auto candidate_bytes = bytes;
      auto candidate_json = json_text;
      if (!append_transaction(candidate, candidate_bytes, candidate_json, allow_pixels)) return false;
      if (candidate_json.size() >= wire::max_json_bytes) throw std::length_error("Optional metadata exceeds mailbox capacity");
      response = candidate;
      bytes = candidate_bytes;
      json_text.swap(candidate_json);
    } catch (...) {
      // Leave the complete primary response untouched. Even when no JSON byte
      // is free, the receiver can explain why optional pixels were not shared.
      response.flags |= wire::optional_metadata_omitted;
    }
    state_->appended = true;
    return true;
  }
  bool ui_capture_batch::append_transaction(wire::response_t &response, std::uint64_t &bytes, std::string &json_text, bool allow_pixels) {
    const auto now = GetTickCount64();
    const bool deadline = state_->frozen_tick && now - state_->frozen_tick >= 250;
    bool pending = false;
    for (auto &e : state_->entries) {
      if (!e.ticket) continue;
      const auto result = capture::acquire_diagnostic_texture(e.ticket, e.pixels);
      if (result == capture::status::ready) e.status = "captured";
      else if (result == capture::status::recorded || result == capture::status::submitted) {
        pending = true;
        e.status = "pending_gpu";
      } else e.status = "capture_failed";
    }
    if (pending && !deadline && allow_pixels) return false;
    auto metadata = nlohmann::json::parse(json_text);
    auto inventory = nlohmann::json::array();
    for (unsigned i = 0; i != resource_count; ++i) {
      auto &e = state_->entries[i];
      const auto &kind = ui_resources::catalog[i];
      const auto &p = e.pixels;
      if (e.ticket && p.result != capture::status::ready && std::strcmp(e.status, "pending_gpu") == 0)
        e.status = allow_pixels ? "gpu_completion_timeout" : "primary_capture_unavailable";
      if (p.result == capture::status::ready) {
        const auto bpp = wire::pixel_bytes(p.format);
        const auto size = std::uint64_t(p.width) * p.height * bpp;
        if (!allow_pixels) e.status = "primary_capture_unavailable";
        else if (!bpp) e.status = "unsupported_transport_format";
        else if (response.texture_count >= wire::max_textures || bytes + size > wire::max_capture_bytes) e.status = "transport_budget_exceeded";
        else {
          response.textures[response.texture_count++] = {static_cast<wire::artifact>(kind.artifact_id), p.width, p.height, p.format, p.shared_handle};
          bytes += size;
        }
      }
      const auto &o = e.observed;
      const auto &at = o.observation;
      nlohmann::json row {{"artifact_id", kind.artifact_id}, {"file_stem", kind.file_stem}, {"provider", kind.source == ui_resources::provider::ngx ? "ngx" : "streamline"},
        {"name", kind.name}, {"semantic", kind.semantic}, {"role", ui_resources::role_name(kind.content)}, {"status", e.status}, {"attempts", e.attempts}, {"transfer_status", "unknown"},
        {"pairing", "Independent API-call snapshot; association with the main color/depth frame is not verified."},
        {"capture_stage", capture::name(e.record.stage)}, {"capture_result", capture::name(e.ticket ? p.result : e.record.result)}, {"failure", capture::name(p.failure)}};
      if (at.session) {
        row["observation"] = {{"session", at.session}, {"epoch", at.epoch}, {"sequence", at.sequence}, {"tick_ms", at.tick}, {"frame_token", hex(at.frame_token)},
          {"frame_numeric", at.numeric_frame ? nlohmann::json(at.frame_numeric) : nlohmann::json(nullptr)}, {"viewport", at.viewport}, {"command", hex(at.command)},
          {"source_native", hex(o.native)}, {"owner", hex(o.owner)}, {"source_id", o.source_id}, {"feature", o.feature}, {"native_state", o.native_state},
          {"state_declared", o.state_declared}, {"tag_type", o.tag_type}, {"lifecycle", o.lifecycle}, {"sdk_success", e.finished ? nlohmann::json(e.successful) : nlohmann::json(nullptr)}};
        row["declared_area"] = {{"left", o.area.left}, {"top", o.area.top}, {"width", o.area.width}, {"height", o.area.height}};
      } else row["observation"] = nullptr;
      if (e.ticket) row["gpu"] = {{"capture_id", p.capture_id}, {"producer_queue", hex(p.producer_queue)}, {"producer_fence", p.producer_fence}, {"producer_completed", p.producer_completed}, {"recording_retired", p.producer_recording_retired}};
      if (p.result == capture::status::ready) {
        row["allocation"] = {{"width", p.width}, {"height", p.height}, {"dxgi_format", p.format}};
        row["area"] = rectangle(p.area);
        row["resource_id"] = p.resource_id;
        row["area_source"] = o.area.width && o.area.height ? "declared_for_this_resource" : "full_allocation_no_active_rect_supplied";
      }
      inventory.push_back(std::move(row));
    }
    metadata["optional_captures"] = std::move(inventory);
    metadata["optional_capture_policy"] = {{"selection", "first recorded observation per resource kind in this request; later observations are separate metadata"}, {"frozen_tick_ms", state_->frozen_tick}, {"completion_deadline_ms", 250}};
    json_text = metadata.dump();
    return true;
  }
}

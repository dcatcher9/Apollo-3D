// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_capture.h"
#include "game3d_diagnostic_metadata.h"
#include "streamline_depth_capture.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

// This test exercises the real API-observation -> frozen dump bridge. The
// native owner is a controllable asynchronous boundary, not a GPU simulation;
// native fixture tests separately verify copies, states and fence retirement.
namespace sunshine_streamline::depth_capture {
  struct source_reference {};
  struct texture_reference {};
}
namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace sl = sunshine_streamline;
  namespace d = sunshine_game3d::diagnostic;
  namespace ui = sunshine_game3d::ui_resources;
  namespace wire = game3d_debug;
  using sunshine_game3d::ui_capture_batch;
  using json = nlohmann::json;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

  struct operation {
    capture::input input;
    std::shared_ptr<const capture::texture_reference> gpu_owner;
    capture::diagnostic_texture texture;
    unsigned finish_count {}, release_count {};
    bool finished {}, successful {}, completed {};
  };
  std::vector<operation> operations;
  unsigned record_calls {};
  bool reject_record {};
  capture::record_diagnostic rejection;
  std::uint32_t next_width = 128, next_height = 64, next_format = 41;

  operation &op(std::uint64_t id) { return operations.at(static_cast<std::size_t>(id - 1)); }
  void reset_native() {
    for (const auto &value : operations) require(value.release_count == 1, "batch did not release exactly one native ticket lease");
    operations.clear();
    record_calls = 0;
    reject_record = false;
    rejection = {};
    rejection.stage = capture::record_stage::missing_state;
    rejection.result = capture::status::missing_state;
    next_width = 128; next_height = 64; next_format = 41;
  }
  std::uint64_t begin_window() {
    sunshine_game3d::arm_diagnostic_metadata(false);
    sunshine_game3d::arm_diagnostic_metadata(true);
    return sunshine_game3d::diagnostic_metadata_generation();
  }
  d::stamp stamp(std::uint64_t sequence = 1) {
    return {sunshine_game3d::diagnostic_metadata_generation(), 71, sequence, GetTickCount64(), 19, 24, 0x4321, 7, true};
  }
  void tag(unsigned type, std::uint64_t native, const d::stamp &at, std::uint32_t state = 128) {
    sl::abi_v2::resource resource {};
    resource.base.type = sl::resource_guid; resource.base.version = 1;
    resource.native = reinterpret_cast<void *>(native); resource.state = state;
    sl::abi_v2::resource_tag value {};
    value.base.type = sl::tag_guid; value.base.version = 1;
    value.type = type; value.resource_ptr = native ? &resource : nullptr;
    value.area = {3, 5, 64, 32};
    sl::abi_v2::viewport viewport {};
    viewport.base.type = sl::viewport_guid; viewport.base.version = 1; viewport.value = 7;
    d::observe_sl_tags_v2(at, viewport, &value, 1, d::tag_scope::frame);
  }
  void ngx(unsigned artifact_id, std::uint64_t native, const d::stamp &at, std::uint64_t source) {
    d::ngx_evaluation value;
    value.observation = at; value.source_id = source; value.owner = 66; value.handle = 77;
    value.feature_known = source != 0; value.feature = 1;
    value.parameter_count = 1;
    auto &parameter = value.parameters[0];
    std::snprintf(parameter.name, sizeof(parameter.name), "%s", ui::find(artifact_id)->parameter_key);
    parameter.type = d::parameter_type::resource;
    parameter.getter_available = parameter.successful = true; parameter.integer = native;
    d::observe_ngx(value);
  }
  json row(const std::string &text, unsigned id) {
    const auto parsed = json::parse(text);
    for (const auto &item : parsed.at("optional_captures")) if (item.at("artifact_id") == id) return item;
    throw std::runtime_error("fixed optional capture row missing");
  }
  wire::response_t primary() {
    wire::response_t out;
    out.result = wire::status::complete; out.texture_count = 1;
    out.textures[0] = {wire::artifact::source_color, 4, 4, 28, 987};
    return out;
  }
}

namespace sunshine_streamline::depth_capture {
  source_ref retain_source(std::uint64_t native, record_diagnostic *diagnostic) {
    if (diagnostic) { diagnostic->stage = record_stage::retained; diagnostic->result = status::ready; }
    return native ? std::make_shared<source_reference>() : source_ref {};
  }
  std::uint64_t source_present_generation(const source_ref &) { return 44; }
  diagnostic_ticket record_diagnostic_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    ++record_calls;
    if (diagnostic) {
      *diagnostic = reject_record ? rejection : record_diagnostic {};
      diagnostic->command = command;
      diagnostic->resource = value.resource.native;
      diagnostic->native_state = value.native_state;
      diagnostic->expected_generation = value.source_present_generation;
      if (!reject_record) {
        diagnostic->stage = record_stage::recorded;
        diagnostic->result = status::recorded;
      }
    }
    if (reject_record) return {};
    operation value_copy;
    value_copy.input = value;
    value_copy.gpu_owner = std::make_shared<texture_reference>();
    value_copy.texture.width = next_width; value_copy.texture.height = next_height; value_copy.texture.format = next_format;
    value_copy.texture.shared_handle = 0x10000 + operations.size();
    value_copy.texture.texture = 0x20000 + operations.size();
    value_copy.texture.capture_id = operations.size() + 1;
    value_copy.texture.resource_id = value.resource.native;
    value_copy.texture.area = value.resource.area.width ? value.resource.area : sunshine_scene_depth::extent {0, 0, next_width, next_height};
    value_copy.texture.producer_queue = 45;
    value_copy.texture.producer_fence = 46;
    operations.push_back(std::move(value_copy));
    return {operations.size(), operations.back().gpu_owner};
  }
  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful) {
    auto &value = op(ticket.id);
    ++value.finish_count; value.finished = true; value.successful = successful;
  }
  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out) {
    auto &value = op(ticket.id);
    out = {};
    if (value.release_count) out.result = status::stale;
    else if (value.finished && !value.successful) {
      out.result = status::failed; out.failure = capture_failure::evaluation_failed;
    } else if (!value.finished || !value.completed) out.result = status::submitted;
    else {
      out = value.texture;
      out.ownership = value.gpu_owner;
      out.producer_completed = out.producer_fence;
      out.producer_recording_retired = true;
      out.result = status::ready;
    }
    return out.result;
  }
  void release_diagnostic_texture(const diagnostic_ticket &ticket) { if (ticket) ++op(ticket.id).release_count; }
  const char *name(status value) {
    switch (value) {
      case status::ready: return "ready";
      case status::submitted: return "submitted";
      case status::failed: return "failed";
      case status::recorded: return "recorded";
      case status::missing_state: return "missing_state";
      case status::conflicting_state: return "conflicting_state";
      default: return "unavailable";
    }
  }
  const char *name(capture_failure value) { return value == capture_failure::evaluation_failed ? "evaluation_failed" : "none"; }
  const char *name(record_stage value) {
    switch (value) {
      case record_stage::recorded: return "recorded";
      case record_stage::missing_state: return "missing_state";
      case record_stage::conflicting_state: return "conflicting_state";
      default: return "not_recorded";
    }
  }
  const char *name(recording_loss value) { return value == recording_loss::global_observation_loss ? "global_observation_loss" : "none"; }
}

int main() try {
  {
    ui_capture_batch batch(begin_window());
    tag(23, 0, stamp());
    batch.freeze();
    auto response = primary();
    std::uint64_t bytes = 64;
    std::string metadata = R"({"sentinel":42})";
    require(batch.append(response, bytes, metadata, true), "empty/null resources blocked primary dump");
    require(row(metadata, 10)["status"] == "explicit_null" && row(metadata, 9)["status"] == "not_observed",
            "null resource was conflated with an unobserved resource");
    require(response.texture_count == 1 && bytes == 64 && record_calls == 0, "null resource scheduled native capture");
    require(!row(metadata, 10).contains("capture_diagnostic") && !row(metadata, 9).contains("capture_diagnostic"),
            "unattempted resources fabricated a native capture diagnostic");
    require(json::parse(metadata)["sentinel"] == 42, "bridge replaced primary metadata");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    reject_record = true;
    rejection.result = capture::status::conflicting_state;
    rejection.stage = capture::record_stage::conflicting_state;
    rejection.recording_cookie = 123;
    rejection.device_identity = rejection.expected_device_identity = 456;
    rejection.current_generation = 45;
    rejection.observed_state = 4;
    rejection.observed = true;
    rejection.command_type = 0;
    rejection.width = 3840; rejection.height = 2160; rejection.format = 24; rejection.flags = 5;
    rejection.dimension = 3; rejection.mip_levels = 1; rejection.array_size = 1; rejection.samples = 1;
    const auto at = stamp(19);
    tag(53, 0x5353, at, 8);
    batch.freeze();
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(batch.append(response, bytes, metadata, true), "state conflict blocked primary capture");
    const auto captured = row(metadata, 32);
    const json expected {
      {"result", "conflicting_state"}, {"stage", "conflicting_state"}, {"loss", "none"},
      {"command", "0x4321"}, {"resource", "0x5353"}, {"recording_cookie", 123},
      {"device_identity", 456}, {"expected_device_identity", 456},
      {"expected_present_generation", 44}, {"current_present_generation", 45},
      {"native_state", 8}, {"observed_state", 4}, {"observed", true}, {"blocked", false},
      {"copy_state", nullptr}, {"copy_state_known", false}, {"used_observed_state", false},
      {"recording_closed", false}, {"recording_invalid", false}, {"render_pass", false},
      {"command_type", 0}, {"width", 3840}, {"height", 2160}, {"format", 24}, {"flags", 5},
      {"dimension", 3}, {"mip_levels", 1}, {"array_size", 1}, {"samples", 1}, {"attempt", 1},
      {"observation", {{"session", at.session}, {"epoch", 71}, {"sequence", 19}, {"tick_ms", at.tick}, {"viewport", 7}}}
    };
    require(captured.at("capture_diagnostic") == expected, "state conflict evidence was omitted, relabeled or altered");
    require(captured["status"] == "capture_rejected" && captured["capture_stage"] == "conflicting_state" &&
            captured["capture_result"] == "conflicting_state" && !captured.contains("gpu") &&
            response.texture_count == 1 && bytes == 64 && operations.empty(),
            "state conflict diagnostics changed native admission or published pixels");
    require(json::parse(metadata)["optional_captures"].size() == std::size(ui::catalog) && metadata.size() < wire::max_json_bytes,
            "capture diagnostics exceeded the fixed inventory or metadata budget");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    const auto at = stamp();
    next_width = 3840; next_height = 2160; next_format = 24;
    tag(53, 0x5353, at, 8);
    require(operations.size() == 1 && op(1).input.resource.native == 0x5353 &&
            op(1).input.native_state == 8 && op(1).input.valid_until == sunshine_scene_depth::lifetime::at_call &&
            op(1).input.force_snapshot, "Backbuffer was not copied synchronously at its actual tag boundary");
    batch.freeze();
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(!batch.append(response, bytes, metadata, true), "Backbuffer exposed an unfinished GPU snapshot");
    d::finish_sl_call(at, true);
    op(1).completed = true;
    require(batch.append(response, bytes, metadata, true), "completed Backbuffer snapshot stayed unavailable");
    const auto captured = row(metadata, 32);
    require(captured["status"] == "captured" && captured["file_stem"] == "sl_backbuffer" &&
            captured["role"] == "color_alpha" && captured["transfer_status"] == "unknown" &&
            captured["semantic"] == "final_game_color_before_fg_candidate_not_verified_ui_mask" &&
            captured["observation"]["tag_type"] == 53 && captured["observation"]["source_native"] == "0x5353",
            "Backbuffer evidence lost provenance or was promoted to an authoritative UI mask");
    require(response.texture_count == 2 && static_cast<unsigned>(response.textures[1].kind) == 32 &&
            response.textures[1].dxgi_format == 24 && response.textures[1].width == 3840 && response.textures[1].height == 2160 &&
            bytes == 64 + std::uint64_t(3840) * 2160 * 4,
            "Backbuffer was converted, cropped, or charged an incorrect native byte size");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    const auto at = stamp();
    tag(23, 0x1111, at);
    require(operations.size() == 1, "nonnull resource was not captured at API observation");
    const auto &input = op(1).input;
    require(input.resource.native == 0x1111 && input.resource.area.left == 5 && input.resource.area.top == 3 &&
            input.resource.area.width == 64 && input.native_state == 128 && input.force_snapshot &&
            input.source_present_generation == 44, "capture did not freeze actual source, state and own rectangle");
    tag(23, 0x2222, stamp(2));
    require(operations.size() == 1, "later resource replaced already-recorded immutable snapshot");
    batch.freeze();
    tag(2, 0x3333, stamp(3));
    require(operations.size() == 1, "freeze still admitted new resource kind");
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(!batch.append(response, bytes, metadata, true) && response.texture_count == 1 && metadata == "{}", "pending GPU snapshot leaked pixels or partial metadata");
    d::finish_sl_call(at, true);
    require(op(1).finish_count == 1, "freeze suppressed a recorded invocation's finish callback");
    op(1).completed = true;
    require(batch.append(response, bytes, metadata, true), "completed immutable snapshot remained pending");
    const auto captured = row(metadata, 10);
    require(captured["status"] == "captured" && captured["observation"]["source_native"] == "0x1111" &&
            captured["observation"]["sequence"] == 1 && captured["observation"]["sdk_success"] == true,
            "captured pixels were relabeled using later resource metadata");
    require(captured["role"] == "color_alpha" && captured["transfer_status"] == "unknown", "preview metadata lost role or invented transfer function");
    require(response.texture_count == 2 && response.textures[1].handle == op(1).texture.shared_handle &&
            bytes == 64 + 128 * 64 * 4, "ready texture was not appended with exact size");
    const auto fixed = metadata;
    require(batch.append(response, bytes, metadata, true) && metadata == fixed && response.texture_count == 2,
            "repeated append changed already-published response");
    require(op(1).release_count == 0, "native shared handle lease ended before batch acknowledgement");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    const auto at = stamp();
    tag(23, 0x1111, at);
    ngx(20, 0x2222, at, 12);
    ngx(21, 0x3333, at, 0); // Unknown-feature counter may equal known-source counter.
    require(operations.size() == 3 && op(2).input.native_state == UINT32_MAX &&
            op(2).input.proof == sunshine_scene_depth::state_proof::observed_nonzero && op(2).input.resource.area.width == 0,
            "NGX bridge fabricated state or borrowed the depth rectangle");
    d::finish_sl_call(at, true);
    require(op(1).finished && !op(2).finished && !op(3).finished, "SL completion finished colliding NGX stamp");
    d::finish_ngx(at, 12, false);
    require(op(2).finished && !op(2).successful && !op(3).finished, "known NGX completion finished unknown-source stamp");
    d::finish_ngx(at, 0, true);
    require(op(3).finished && op(3).successful && op(2).finish_count == 1, "NGX source-scoped finish was lost or duplicated");
    op(1).completed = op(2).completed = op(3).completed = true;
    batch.freeze();
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(batch.append(response, bytes, metadata, true), "failed vendor invocation blocked valid optional resources");
    require(row(metadata, 20)["status"] == "capture_failed" && row(metadata, 21)["status"] == "captured" && response.texture_count == 3,
            "failed SDK invocation published optional pixels or hid successful peers");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    reject_record = true;
    for (unsigned i = 1; i <= 7; ++i) tag(23, 0x1111, stamp(i));
    batch.freeze();
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(batch.append(response, bytes, metadata, true) && record_calls == 4 &&
            row(metadata, 10)["status"] == "attempt_limit", "failed diagnostic allocation retries were unbounded");
    require(row(metadata, 10)["observation"]["sequence"] == 7 &&
            row(metadata, 10)["capture_diagnostic"]["attempt"] == 4 &&
            row(metadata, 10)["capture_diagnostic"]["observation"]["sequence"] == 4,
            "later retry-limit observation relabeled the last attempted capture diagnostic");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    tag(23, 0x1111, stamp());
    batch.freeze();
    auto response = primary(); response.result = wire::status::unavailable; response.texture_count = 0;
    std::uint64_t bytes = 0; std::string metadata = "{}";
    require(batch.append(response, bytes, metadata, false), "unavailable primary capture waited for optional GPU work");
    require(row(metadata, 10)["status"] == "primary_capture_unavailable" && response.texture_count == 0,
            "unavailable primary invented an optional GPU timeout or published pixels");
  }
  reset_native();
  {
    ui_capture_batch batch(begin_window());
    tag(23, 0x1111, stamp());
    batch.freeze();
    auto response = primary(); std::uint64_t bytes = 64; std::string metadata = "{}";
    require(!batch.append(response, bytes, metadata, true), "pending native capture did not honor bounded deadline");
    Sleep(270);
    require(batch.append(response, bytes, metadata, true) && row(metadata, 10)["status"] == "gpu_completion_timeout" &&
            response.texture_count == 1, "expired optional capture blocked or failed the primary dump");
  }
  reset_native();
  for (unsigned budget_case = 0; budget_case != 3; ++budget_case) {
    {
      ui_capture_batch batch(begin_window());
      next_format = 26; // R11G11B10_FLOAT is admitted by the native owner.
      const auto at = stamp(); tag(2, 0x1111, at); d::finish_sl_call(at, true); op(1).completed = true;
      batch.freeze();
      auto response = primary();
      if (budget_case == 1) response.texture_count = wire::max_textures;
      std::uint64_t bytes = budget_case == 2 ? wire::max_capture_bytes - 1 : 64;
      const auto before_bytes = bytes; const auto before_count = response.texture_count;
      std::string metadata = "{}";
      require(batch.append(response, bytes, metadata, true), "transport admission remained pending");
      if (!budget_case) {
        require(row(metadata, 9)["status"] == "captured" && bytes == before_bytes + 128 * 64 * 4 &&
                response.texture_count == before_count + 1, "native-supported packed float color was rejected by transport");
      } else {
        require(row(metadata, 9)["status"] == "transport_budget_exceeded" && bytes == before_bytes &&
                response.texture_count == before_count && response.result == wire::status::complete,
                "optional transport overflow altered primary capture or exceeded response bounds");
      }
    }
    reset_native();
  }
  // Optional provenance must fit together with its pixels. A valid primary
  // dump near the mailbox limit must survive unchanged when that is impossible.
  // Invalid JSON exercises the same transactional exception boundary directly.
  for (unsigned metadata_case = 0; metadata_case != 3; ++metadata_case) {
    {
      ui_capture_batch batch(begin_window());
      const auto at = stamp(); tag(23, 0x1111, at); d::finish_sl_call(at, true); op(1).completed = true;
      batch.freeze();
      auto response = primary();
      response.capture_id = 55; response.consumer_nonce = 66;
      std::uint64_t bytes = 64;
      std::string metadata;
      if (!metadata_case) {
        json document {{"latest_observations", json::parse(sunshine_game3d::diagnostic_metadata_json())}, {"padding", ""}};
        const auto current_size = document.dump().size();
        require(current_size + 64 < wire::max_json_bytes, "near-limit fixture already exceeded mailbox");
        document["padding"] = std::string(wire::max_json_bytes - current_size - 64, 'x');
        metadata = document.dump();
        require(metadata.size() == wire::max_json_bytes - 64, "near-limit fixture did not leave the intended small headroom");
      } else metadata = metadata_case == 1 ? "{invalid" : "[]";
      const auto original_json = metadata;
      const auto original_response = response;
      require(batch.append(response, bytes, metadata, true), "optional metadata overflow/exception did not complete nonblocking");
      require(metadata == original_json && bytes == 64 && response.texture_count == original_response.texture_count &&
              response.result == original_response.result && response.capture_id == 55 && response.consumer_nonce == 66 &&
              std::memcmp(response.textures, original_response.textures, sizeof(response.textures)) == 0,
              "optional metadata failure partially committed textures/bytes or changed primary dump");
      require((response.flags & wire::optional_metadata_omitted) != 0, "optional metadata omission was not explicitly reported out of band");
      if (!metadata_case) require(json::parse(metadata)["latest_observations"]["ui_resources"].size() == std::size(ui::catalog),
                                 "optional omission discarded primary fixed resource inventory");
      const auto flags = response.flags;
      require(batch.append(response, bytes, metadata, true) && metadata == original_json && response.flags == flags &&
              response.texture_count == original_response.texture_count && bytes == 64 &&
              std::memcmp(response.textures, original_response.textures, sizeof(response.textures)) == 0,
              "repeated append after optional omission duplicated descriptors or changed published state");
      require(op(1).release_count == 0, "transaction fallback prematurely released the native request lease");
    }
    reset_native();
  }
  {
    auto first = std::make_unique<ui_capture_batch>(begin_window());
    tag(23, 0x1111, stamp());
    require(operations.size() == 1, "cancellation fixture did not record pending resource");
    auto second = std::make_unique<ui_capture_batch>(begin_window());
    first.reset();
    require(op(1).release_count == 1 && op(1).gpu_owner, "cancellation did not revoke ticket or prematurely freed native GPU ownership");
    tag(23, 0x2222, stamp());
    require(operations.size() == 2, "old batch destruction canceled replacement request's observer");
    second.reset();
    require(op(2).release_count == 1, "pending replacement request was not canceled");
    tag(23, 0x3333, stamp(2));
    require(operations.size() == 2, "destroyed request accepted later resource observation");
  }
  reset_native();
  sunshine_game3d::arm_diagnostic_metadata(false);
  d::set_resource_callbacks(nullptr);
  std::puts("PASS optional UI capture bridge: availability, frozen ownership, asynchronous readiness, scoped SDK completion, bounds and cancellation");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL optional UI capture bridge: %s\n", error.what());
  return 1;
}

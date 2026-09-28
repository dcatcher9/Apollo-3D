// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_diagnostic_metadata.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
  using namespace sunshine_game3d;
  namespace d = sunshine_game3d::diagnostic;
  namespace sl = sunshine_streamline;
  using json = nlohmann::json;

  void require(bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  json snapshot() {
    return json::parse(diagnostic_metadata_json());
  }

  d::stamp stamp(unsigned viewport = 7, std::uint64_t sequence = 1) {
    return {diagnostic_metadata_generation(), 100, sequence, 1234, 0x123, 99, 0x456, viewport, true};
  }

  sl::abi_v2::viewport viewport(unsigned id) {
    sl::abi_v2::viewport out {};
    out.base.type = sl::viewport_guid;
    out.base.version = 1;
    out.value = id;
    return out;
  }

  json ui_row(const json &value, unsigned id) {
    for (const auto &entry : value.at("ui_resources")) if (entry.at("artifact_id") == id) return entry;
    throw std::runtime_error("fixed UI resource missing from inventory");
  }

  unsigned resource_notifications {}, finish_notifications {};
  d::resource_observation last_resource;
  bool callback_outside_lock {true}, last_sdk_success {};
  ui_resources::provider last_finished_provider {};
  std::uint64_t last_finished_source {};
  void observe_resource(const d::resource_observation &value) noexcept {
    ++resource_notifications;
    last_resource = value;
    try { callback_outside_lock &= snapshot().at("status") != "busy"; }
    catch (...) { callback_outside_lock = false; }
  }
  void observe_finish(ui_resources::provider source, const d::stamp &, std::uint64_t source_id, bool successful) noexcept {
    ++finish_notifications;
    last_sdk_success = successful;
    last_finished_provider = source;
    last_finished_source = source_id;
  }
  const d::resource_callbacks callbacks {observe_resource, observe_finish};

  void test_sl_resource_type_admission() {
    d::set_resource_callbacks(&callbacks);
    for (const bool version_one : {false, true}) {
      for (const auto type : {0u, 8u, 1u, UINT32_MAX}) {
        arm_diagnostic_metadata(false); arm_diagnostic_metadata(true);
        const auto at = stamp();
        constexpr std::uint64_t identity = 0xabcdef;
        if (version_one) {
          sl::abi_v1::resource resource {};
          resource.type = type; resource.native = reinterpret_cast<void *>(identity); resource.state = 128;
          d::observe_sl_tag_v1(at, &resource, 23, nullptr, true);
        } else {
          sl::abi_v2::resource resource {};
          resource.base.type = sl::resource_guid; resource.base.version = 1;
          resource.type = type; resource.native = reinterpret_cast<void *>(identity); resource.state = 128;
          sl::abi_v2::resource_tag tag {};
          tag.base.type = sl::tag_guid; tag.base.version = 1;
          tag.resource_ptr = &resource; tag.type = 23; tag.lifecycle = 0;
          const auto view = viewport(7);
          d::observe_sl_tags_v2(at, view, &tag, 1, d::tag_scope::frame);
        }
        const bool accepted = type == 0 || (!version_one && type == 8);
        const auto row = ui_row(snapshot(), 10);
        require(last_resource.readable && last_resource.version_one == version_one &&
            last_resource.descriptor_supported == accepted && last_resource.native == (accepted ? identity : 0) &&
            row["state"] == (accepted ? "non_null" : "unsupported"),
          "SL resource enum admission diverged between callback/inventory or accepted a buffer/invalid type");
        // The identity is deliberately not a COM object. Metadata reports its
        // declaration only; the native capture owner must still validate it.
        require(row["capture_status"].is_null() && row["runtime_support"] == "unknown",
          "SL resource enum was promoted to a validated texture or successful capture");
      }
    }
    d::set_resource_callbacks(nullptr);
    arm_diagnostic_metadata(false); arm_diagnostic_metadata(true);
  }

  void test_ngx_depth_owned_observation() {
    arm_diagnostic_metadata(false); arm_diagnostic_metadata(true);
    d::set_resource_callbacks(&callbacks);
    const auto resources_before = resource_notifications;
    const auto finishes_before = finish_notifications;
    d::ngx_evaluation value;
    value.observation = stamp(7, 50);
    value.owner = 4; value.handle = 5;
    value.feature_known = true; value.feature = 11;
    value.create_width = 1920; value.create_height = 1080;
    value.capture_suppressed_by_depth_owner = true;
    value.parameter_count = 1;
    auto &p = value.parameters[0];
    std::snprintf(p.name, sizeof(p.name), "DLSSG.UIAlpha");
    p.type = d::parameter_type::resource;
    p.getter_available = p.successful = true; p.integer = 0xfeed;
    d::observe_ngx(value);
    d::finish_ngx(value.observation, value.source_id, true, true);
    const auto data = snapshot();
    const auto &feature = data.at("ngx").at(0);
    const auto row = ui_row(data, 36);
    require(row["state"] == "non_null" && row["observations"][0]["sdk_successful"] == true &&
        row["observations"][0]["resource_capture_status"] == "not_attempted_depth_owned",
      "Depth ownership hid a successful NGX UI query or its capture suppression");
    require(feature["feature"] == 11 && feature["create_width"] == 1920 &&
        feature["capture_metadata_available"] == false && feature["scene_revision"].is_null() &&
        feature["sequence_domain"] == "diagnostic-depth-owned",
      "Metadata-only NGX observation lost known feature metadata or claimed depth capture identity");
    require(resource_notifications == resources_before && finish_notifications == finishes_before,
      "Metadata-only NGX observation dispatched native capture callbacks");
    // A dropped/stale metadata entry must not regain native callback authority.
    arm_diagnostic_metadata(false); arm_diagnostic_metadata(true);
    d::finish_ngx(value.observation, value.source_id, true, true);
    require(finish_notifications == finishes_before, "Suppressed stale finish dispatched a capture callback");
    value.observation = stamp(7, 51);
    value.capture_suppressed_by_depth_owner = false;
    value.source_id = 6;
    d::observe_ngx(value);
    d::finish_ngx(value.observation, value.source_id, true);
    require(resource_notifications == resources_before + 1 && finish_notifications == finishes_before + 1,
      "Suppression leaked into a subsequent independent NGX capture");
    d::set_resource_callbacks(nullptr);
    arm_diagnostic_metadata(false); arm_diagnostic_metadata(true);
  }
}  // namespace

int main() try {
  require(snapshot()["status"] == "not-armed", "initial snapshot must identify missing observation window");
  require(snapshot()["ui_resources"].size() == std::size(ui_resources::catalog), "unarmed snapshot omitted fixed UI inventory");
  const auto initial = snapshot();
  for (const auto &row : initial["ui_resources"]) {
    require(row["state"] == "unobserved" && row["observations"].empty(), "catalog fabricated resource availability");
    require(row["runtime_support"] == "unknown" && row["capture_status"].is_null(), "catalog fabricated runtime support or capture");
  }
  require(!has_diagnostic_frame_observation(), "unarmed diagnostic window reported a frame observation");
  arm_diagnostic_metadata(true);
  const auto old = stamp();
  require(!has_diagnostic_frame_observation(), "fresh window inherited a frame observation");
  d::observe_sl_fg(old, 1, 3, true, true);
  d::finish_sl_call(old, true);
  require(!has_diagnostic_frame_observation(), "cached FG options or completion-only call counted as a fresh frame");
  sl::abi_v1::constants raw {};
  raw.common.camera_position[1] = 123.f;
  raw.common.clip_to_prev_clip.m[2][3] = 42.f;
  raw.common.motion_vector_scale[0] = 0.5f;
  raw.common.invalid_motion_vector = std::numeric_limits<float>::infinity();
  raw.motion_vectors_jittered = 1;
  d::observe_sl_constants(old, raw, true, sl::decode_status::ok);
  d::finish_sl_call(old, true);
  auto first = snapshot();
  require(has_diagnostic_frame_observation() && first["frame_observation_in_window"] == true, "SL constants did not mark the current observation window");
  const auto &camera = first["streamline"][0]["constants"];
  require(camera["common"]["camera_position"][1] == 123.f, "camera position was discarded");
  require(camera["common"]["clip_to_prev_clip"][2][3] == 42.f, "motion matrix was discarded");
  require(camera["common"]["invalid_motion_vector"].is_null(), "non-finite values must remain valid JSON");
  require(camera["flags"]["motion_vectors_jittered"] == 1 && camera["successful"] == true, "flags/result lost");

  sl::abi_v2::resource resource {};
  resource.base.type = sl::resource_guid;
  resource.base.version = 1;
  resource.native = reinterpret_cast<void *>(0xabcdef);
  resource.width = 2228;
  resource.height = 1256;
  resource.native_format = 41;
  sl::abi_v2::resource_tag tag {};
  tag.base.type = sl::tag_guid;
  tag.base.version = 1;
  tag.type = 9876;
  tag.resource_ptr = &resource;
  tag.area = {8, 16, 2196, 1240};
  tag.lifecycle = 2;
  auto view = viewport(7);
  d::observe_sl_tags_v2(stamp(7, 2), view, &tag, 1, d::tag_scope::frame);
  d::finish_sl_call(stamp(7, 2), false);
  const auto tags = snapshot()["streamline"][0]["tags"];
  require(tags.size() == 1 && tags[0]["type"] == 9876, "unrecognized SL tag types must be observed too");
  require(tags[0]["native_identity"] == "0xabcdef" && tags[0]["format"] == 41, "resource descriptor not copied");
  require(tags[0]["extent"]["left"] == 16 && tags[0]["extent"]["height"] == 1240, "active rectangle lost");
  require(tags[0]["successful"] == false, "failed SDK tag call must not be labeled accepted");

  // Malformed tag versions must not follow even an intentionally invalid
  // resource pointer. Unknown resource headers expose no guessed prefix data.
  tag.type = 9990;
  tag.base.version = 99;
  tag.resource_ptr = reinterpret_cast<sl::abi_v2::resource *>(1);
  d::observe_sl_tags_v2(stamp(7, 3), view, &tag, 1, d::tag_scope::global);
  tag.type = 9991;
  tag.base.version = 1;
  tag.resource_ptr = &resource;
  resource.base.version = 999;
  d::observe_sl_tags_v2(stamp(7, 4), view, &tag, 1, d::tag_scope::global);
  const auto malformed = snapshot()["streamline"][0]["tags"];
  require(!malformed[1]["resource_readable"].get<bool>() && malformed[1]["native_identity"] == "0x0", "malformed tag followed its pointer");
  require(!malformed[2]["full_descriptor_readable"].get<bool>() && malformed[2]["format"].is_null(), "unknown resource version was interpreted");

  d::ngx_evaluation ngx;
  ngx.observation = stamp();
  ngx.owner = 4;
  ngx.handle = 5;
  ngx.source_id = 6;
  ngx.feature_known = true;
  ngx.feature = 1;
  ngx.parameter_count = 3;
  std::snprintf(ngx.parameters[0].name, sizeof(ngx.parameters[0].name), "Depth");
  ngx.parameters[0].getter_available = true;
  ngx.parameters[0].successful = true;
  ngx.parameters[0].integer = 0;
  std::snprintf(ngx.parameters[1].name, sizeof(ngx.parameters[1].name), "MotionVectors");
  ngx.parameters[1].getter_available = true;
  ngx.parameters[1].result = 0xbad00005;
  std::snprintf(ngx.parameters[2].name, sizeof(ngx.parameters[2].name), "ViewToClipMatrix");
  d::observe_ngx(ngx);
  d::finish_ngx(ngx.observation, ngx.source_id, true);
  const auto parameters = snapshot()["ngx"][0]["parameters"];
  require(parameters[0]["value"] == "0x0", "successful null resource must remain distinguishable");
  require(parameters[1]["successful"] == false && parameters[1]["value"].is_null(), "failed getter fabricated a value");
  require(parameters[1]["result_detail"]["code"] == 0xbad00005 &&
          parameters[1]["result_detail"]["hex"] == "0xbad00005" &&
          parameters[1]["result_detail"]["name"] == "FAIL_InvalidParameter", "primary NGX result lost readable error details");
  require(parameters[2]["getter_available"] == false && parameters[2]["successful"].is_null(), "missing getter not identified");

  arm_diagnostic_metadata(false);
  require(!has_diagnostic_frame_observation() && snapshot()["frame_observation_in_window"] == true, "disarming must clear readiness without erasing captured evidence");
  d::observe_sl_evaluation(old, 999, true);
  require(snapshot()["streamline"][0]["evaluation"].is_null(), "disarmed write mutated retained snapshot");
  arm_diagnostic_metadata(true);
  d::observe_sl_constants(old, raw, true, sl::decode_status::ok);
  require(snapshot()["streamline"].empty(), "previous dump's in-flight hook contaminated new window");
  require(!has_diagnostic_frame_observation(), "stale session satisfied the current frame observation gate");
  ngx.observation = stamp();
  d::observe_ngx(ngx);
  require(has_diagnostic_frame_observation(), "NGX named parameters did not mark a fresh frame observation");

  // The optional resource inventory is fixed even when no SDK supplies these
  // hints. SDK presence, null, failed query and unsupported ABI remain distinct.
  arm_diagnostic_metadata(false);
  arm_diagnostic_metadata(true);
  d::set_resource_callbacks(&callbacks);
  tag.base.type = sl::tag_guid;
  tag.base.version = 1;
  tag.type = 23;
  tag.resource_ptr = nullptr;
  d::observe_sl_tags_v2(stamp(7, 1), view, &tag, 1, d::tag_scope::frame);
  require(resource_notifications == 1 && last_resource.artifact_id == 10 && !last_resource.native && last_resource.descriptor_supported,
          "explicit SL null resource was not synchronously identified");
  require(ui_row(snapshot(), 10)["state"] == "null", "explicit SL null became unobserved");
  resource.base.version = 1;
  resource.state = 128;
  tag.resource_ptr = &resource;
  d::observe_sl_tags_v2(stamp(7, 2), view, &tag, 1, d::tag_scope::frame);
  require(last_resource.native == 0xabcdef && last_resource.state_declared && last_resource.native_state == 128,
          "validated SL observation lost its invocation-scoped resource/state");
  require(last_resource.area.left == 16 && last_resource.area.top == 8 && last_resource.area.width == 2196,
          "SL optional resource borrowed a different crop");
  require(ui_row(snapshot(), 10)["state"] == "non_null" && ui_row(snapshot(), 10)["observations"][0]["sdk_successful"].is_null(),
          "SL pre-call presence was confused with SDK success");
  d::finish_sl_call(stamp(7, 2), false);
  require(finish_notifications == 1 && !last_sdk_success && last_finished_provider == ui_resources::provider::streamline && ui_row(snapshot(), 10)["observations"][0]["sdk_successful"] == false,
          "SL failure was not paired with its pre-call observation");
  resource.state = 0;
  d::observe_sl_tags_v2(stamp(7, 3), view, &tag, 1, d::tag_scope::frame);
  require(!last_resource.state_declared, "SL zero state invented declared GPU state");
  tag.base.version = 99;
  tag.resource_ptr = reinterpret_cast<sl::abi_v2::resource *>(1);
  d::observe_sl_tags_v2(stamp(7, 4), view, &tag, 1, d::tag_scope::frame);
  require(!last_resource.native && !last_resource.descriptor_supported && ui_row(snapshot(), 10)["state"] == "unsupported",
          "unsupported SL layout exposed a pointer or masqueraded as null");
  tag.base.version = 1;
  tag.resource_ptr = nullptr;
  tag.type = 69;
  d::observe_sl_tags_v2(stamp(7, 5), view, &tag, 1, d::tag_scope::frame);
  require(last_resource.artifact_id == 19 && ui_row(snapshot(), 19)["state"] == "null", "verified UIAlpha tag69 was not recognized");

  ngx = {};
  ngx.observation = stamp(7, 6);
  ngx.owner = 4;
  ngx.handle = 5;
  ngx.source_id = 6;
  ngx.feature_known = true;
  ngx.feature = 1;
  for (const auto &entry : ui_resources::catalog) {
    if (entry.source != ui_resources::provider::ngx) continue;
    auto &p = ngx.parameters[ngx.parameter_count++];
    std::snprintf(p.name, sizeof(p.name), "%s", entry.parameter_key);
    p.type = d::parameter_type::resource;
    p.getter_available = entry.artifact_id != 22;
    p.successful = entry.artifact_id != 21;
    p.result = p.successful ? 1 : 0xbad00005;
    p.integer = entry.artifact_id == 20 ? 0 : 0xfeed;
  }
  d::observe_ngx(ngx);
  const auto optional = snapshot();
  require(ui_row(optional, 20)["state"] == "null" && ui_row(optional, 21)["state"] == "query_failed" &&
          ui_row(optional, 22)["state"] == "getter_unavailable" && ui_row(optional, 23)["state"] == "non_null",
          "NGX resource availability states were collapsed");
  require(ui_row(optional, 21)["observations"][0]["native_identity"].is_null(), "failed NGX getter published poisoned pointer");
  require(last_resource.owner == 4 && last_resource.source_id == 6 && !last_resource.state_declared &&
          last_resource.area.width == 0 && last_resource.area.height == 0,
          "NGX resource lost ownership scope or fabricated resource state/crop");
  d::finish_ngx(ngx.observation, ngx.source_id, true);
  require(finish_notifications == 2 && last_sdk_success && callback_outside_lock && last_finished_provider == ui_resources::provider::ngx && last_finished_source == 6,
          "callbacks ran under metadata lock or lost SDK completion/provider scope");
  const auto delivered = resource_notifications;
  d::observe_ngx(d::ngx_evaluation {old});
  require(resource_notifications == delivered, "stale dump session reached resource observer");
  arm_diagnostic_metadata(false);
  d::observe_ngx(ngx);
  d::finish_ngx(ngx.observation, ngx.source_id, false);
  require(resource_notifications == delivered && finish_notifications == 2, "disarmed observations reached resource observer");
  d::set_resource_callbacks(nullptr);
  arm_diagnostic_metadata(true);
  require(ui_row(snapshot(), 10)["state"] == "unobserved" && ui_row(snapshot(), 20)["state"] == "unobserved",
          "new dump inherited old optional resource availability");
  test_sl_resource_type_admission();
  test_ngx_depth_owned_observation();

  // More types than the per-viewport bound must be explicitly marked. A
  // maximum-size snapshot must fit the outer IPC envelope as valid JSON.
  resource.base.version = 1;
  std::array<sl::abi_v2::resource_tag, 32> many {};
  for (unsigned v = 0; v != 4; ++v) {
    auto id = viewport(v);
    for (unsigned i = 0; i != many.size(); ++i) {
      many[i] = tag;
      many[i].resource_ptr = &resource;
      many[i].type = i + 100;
    }
    d::observe_sl_tags_v2(stamp(v, v + 10), id, many.data(), 40, d::tag_scope::global);
    d::observe_sl_constants(stamp(v, v + 20), raw, true, sl::decode_status::ok);
  }
  for (unsigned i = 0; i != 8; ++i) {
    ngx.observation = stamp(0, 100 + i);
    ngx.source_id = 100 + i;
    ngx.parameter_count = d::maximum_parameters;
    for (auto &p : ngx.parameters) {
      std::snprintf(p.name, sizeof(p.name), "%063u", i);
      p.getter_available = true;
      p.successful = true;
      p.integer = UINT64_MAX;
    }
    d::observe_ngx(ngx);
  }
  const auto encoded = diagnostic_metadata_json();
  const auto bounded = json::parse(encoded);
  require(encoded.size() <= 192 * 1024, "metadata exceeded reserved IPC budget");
  require(bounded["truncated"].get<bool>(), "bounded tags silently truncated");
  require(bounded.contains("serialization_omitted"), "serialization omissions not counted");
  require(bounded["serialization_omitted"]["sl_tags"].get<unsigned>() + bounded["serialization_omitted"]["ngx_parameters"].get<unsigned>() > 0, "maximum snapshot did not exercise JSON budget truncation");
  arm_diagnostic_metadata(false);
  std::puts("PASS diagnostic metadata: values, all tags, malformed inputs, availability, arming epochs and bounded JSON");
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "FAIL diagnostic metadata: %s\n", error.what());
  return 1;
}

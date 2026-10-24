// SPDX-License-Identifier: GPL-3.0-only
// The API depth provider's status lines, byte for byte (refactor plan P7:
// "provider log text is rendered from source_status and is byte-identical").
// Each Present below gives the provider's formatters
// (streamline_depth_provider_log.h) fixed values, and the lines they render
// must equal provider_log_goldens/status_lines.txt. A change of how the
// provider gathers the values keeps every line; only a commit that changes a
// line on purpose regenerates the goldens:
//   reshade_provider_log_lines_tests <goldens> --update
#include "streamline_depth_provider_log.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
  namespace depth_capture = sunshine_streamline::depth_capture;
  namespace provider = sunshine_streamline::provider;
  using sunshine_scene_depth::provider_kind;

  std::vector<std::string> lines;

  void emit(const char *present, const char *text) {
    lines.push_back(std::string("[") + present + "] " + text);
  }

  // A readable Streamline SR capture of a 2560x1440 allocation, 2400x1350 active.
  depth_capture::capture_diagnostic sl_capture() {
    depth_capture::capture_diagnostic c;
    c.result = depth_capture::status::ready;
    c.selection = depth_capture::selection_reason::current_capture;
    c.capture_id = 812;
    c.sequence = c.newest_sequence = 4410;
    c.epoch = 3;
    c.command = 0x2a10;
    c.queue = c.requested_queue = c.consumer_queue = 0x7f10;
    c.producer_fence = c.producer_completed = 9001;
    c.retire_fence = 9002;
    c.selection_tick_ms = 50000;
    c.newest_view_tick_ms = c.source_tick_ms = 49992;
    c.active_view_count = 1;
    c.producer_completion_valid = c.producer_recording_retired = true;
    c.finished = c.success = c.submitted = true;
    return c;
  }

  depth_capture::packet sl_frame() {
    depth_capture::packet frame;
    frame.metadata.provider = provider_kind::streamline;
    frame.metadata.epoch = 3;
    frame.metadata.sequence = 4410;
    frame.metadata.viewport = 0;
    frame.metadata.native_state = 0x40;
    frame.metadata.projection.supplied = true;
    frame.metadata.observation_revision = 17;
    frame.metadata.feedback.revision = 4;
    frame.capture_id = 812;
    frame.resource_id = 33;
    frame.pixel_ready = true;
    frame.width = 2560;
    frame.height = 1440;
    frame.area = {0, 0, 2400, 1350};
    return frame;
  }

  depth_capture::consumer_diagnostic consumed() {
    depth_capture::consumer_diagnostic consumer;
    consumer.result = depth_capture::consumer_status::ready;
    consumer.cookie = 71;
    consumer.device_identity = consumer.expected_device_identity = 5;
    consumer.tracked_commands = 3;
    return consumer;
  }

  sunshine_streamline::loss_diagnostics::event unpinned_loss() {
    sunshine_streamline::loss_diagnostics::event loss;
    loss.revision = 12;
    loss.tick = 48800;
    loss.line = 2210;
    loss.thread_id = 4242;
    loss.cause = sunshine_streamline::loss_diagnostics::reason::metadata_storage_busy;
    loss.site = "observe_tag";
    loss.details.sequence = 4409;
    loss.details.viewport = 0;
    loss.details.feature = 0;
    loss.details.has_sdk_result = true;
    loss.details.sdk_result = 0;
    return loss;
  }

  void settled_streamline() {
    char text[2560] {};
    provider::selected_line line;
    line.path = "Streamline";
    line.status = depth_capture::status::ready;
    line.output_ready = line.output_projection_supplied = true;
    line.display = provider::display_status::ready;
    line.present = 1200;
    line.command = 0x2b20;
    provider::format_selected(text, sizeof(text), line, sl_frame(), consumed(), sl_capture());
    depth_capture::display_decision fresh {depth_capture::display_action::copy_fresh, "fresh_copy_candidate"};
    sunshine_scene_depth::jitter_offset jitter {0.25f, -0.375f, 2400, 1350, true};
    provider::append_selection(text, sizeof(text), sl_capture(), fresh, 16, jitter);
    provider::append_loss(text, sizeof(text), false, {}, 0, 0);
    emit("settled Streamline SR fresh copy", text);
  }

  void pending_ngx() {
    auto c = sl_capture();
    c.provider = provider_kind::ngx;
    c.result = depth_capture::status::recorded;
    c.selection = depth_capture::selection_reason::no_completed_snapshot;
    c.capture_id = 813;
    c.sequence = c.newest_sequence = 21;
    c.epoch = 4;
    c.source_id = 9;
    c.finished = c.success = false;
    c.producer_completed = 9000;
    c.producer_recording_retired = false;
    sunshine_streamline::native_observer::counters observer;
    observer.dropped = 1;
    observer.unreadable = 2;
    observer.rejected = 3;
    char text[2560] {};
    provider::format_held(text, sizeof(text), "NGX", c.result, true, provider::display_status::reused_depth, c, observer);
    depth_capture::display_decision hold {depth_capture::display_action::hold, "retained_copy"};
    provider::append_selection(text, sizeof(text), c, hold, 31, {});
    provider::append_loss(text, sizeof(text), true, unpinned_loss(), 1200, 2);
    emit("NGX pending successor held", text);
  }

  void unassociated() {
    auto c = sl_capture();
    c.result = depth_capture::status::stale;
    c.selection = depth_capture::selection_reason::inactive_views;
    c.failure = depth_capture::capture_failure::none;
    char text[768] {};
    provider::format_unassociated(text, sizeof(text), c.result, c, false, 250);
    emit("released to Generic", text);
    provider::format_unassociated(text, sizeof(text), c.result, c, true, 250);
    emit("provider handoff hold", text);
  }

  void frame_generation_scope() {
    char text[384] {};
    provider::format_frame_generation_scope(text, sizeof(text), true, 0, 3);
    emit("FG scope silent", text);
    provider::format_frame_generation_scope(text, sizeof(text), false, 0, 3);
    emit("FG scope live again", text);
  }

  void readiness() {
    provider::readiness_evidence e;
    e.prior.provided = sl_frame().metadata;
    e.prior.provided.tick = 49700;
    e.prior.ready = true;
    e.prior.frame_index = 1199;
    e.prior_capture = 811;
    e.prior_provider_known = e.prior_output_ready = true;
    e.capture = sl_capture();
    e.capture.result = depth_capture::status::stale;
    e.capture.selection = depth_capture::selection_reason::fg_scope_mismatch;
    e.consumer.result = depth_capture::consumer_status::not_attempted;
    e.fg.known = e.fg.enabled = true;
    e.fg.mode = 1;
    e.fg.generated_frames = 1;
    e.fg.epoch = 3;
    e.selection = {true, 3, 0, false, 49000};
    e.fg_query = sunshine_streamline::frame_generation_query_status::observed;
    e.display = provider::display_status::waiting_capture;
    e.present = 1201;
    e.tick = 50000;
    e.action = depth_capture::display_action::invalidate;
    e.source_reason = "FG_scope_changed";
    e.cache_reason = "FG_scope_changed";
    provider::readiness_line line;
    line.ready = false;
    line.why = "FG_scope_changed";
    line.runtime = 0x1000;
    line.native_swapchain = 0x2000;
    line.now = 50000;
    line.loss_tick = 50000;
    line.current = sl_frame().metadata;
    line.retained_clear = "FG_scope_changed";
    line.memory = "; vram_mb=9000/12000";
    char text[3072] {};
    provider::format_readiness(text, sizeof(text), e, line);
    emit("readiness lost: FG took the viewport", text);
    provider::observation_loss_episode loss;
    loss.present = 1201;
    loss.sampled_tick = 50001;
    loss.sampled.requested_revision = 12;
    loss.sampled.found = true;
    loss.sampled.value = unpinned_loss();
    provider::format_observation_evidence(text, sizeof(text), false, 0x1000, 50000, loss);
    emit("observation evidence of the loss", text);

    e.capture = sl_capture();
    e.capture.provider = provider_kind::streamline;
    e.capture.source_id = (1ull << 63);
    e.capture.frame_generation_input = true;
    e.display = provider::display_status::ready;
    e.available = e.copied = e.snapshot_ready = true;
    e.action = depth_capture::display_action::copy_fresh;
    e.source_reason = "fresh_copy_candidate";
    e.cache_reason = "fresh_copy";
    e.present = 1209;
    e.tick = 50140;
    line.ready = true;
    line.why = "fresh_copy";
    line.now = 50140;
    line.suppressed_episodes = 2;
    line.output_reused = false;
    line.retained_clear = "none";
    line.memory = "";
    provider::format_readiness(text, sizeof(text), e, line);
    emit("readiness recovered: FG's own copy", text);
    loss.sampled = {};
    provider::format_observation_evidence(text, sizeof(text), true, 0x1000, 50000, loss);
    emit("observation evidence without a journal entry", text);
  }

  void render_size() {
    provider::render_size_state size;
    size.source = {static_cast<std::uint64_t>(provider_kind::streamline), 3, 0, 0};
    size.width = 2442;
    size.height = 1374;
    size.area = {0, 0, 2442, 1374};
    size.changes = 14;
    size.held = 2;
    size.total = 300;
    size.min_width = 2206;
    size.min_height = 1240;
    size.max_width = 2658;
    size.max_height = 1494;
    char text[640] {};
    provider::format_render_size(text, sizeof(text), size);
    emit("dynamic resolution", text);
  }

  std::string read(const char *path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string text = buffer.str();
    std::string out;
    for (char c : text) {
      if (c != '\r') {
        out.push_back(c);
      }
    }
    return out;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fputs("usage: reshade_provider_log_lines_tests <status_lines.txt> [--update]\n", stderr);
    return 2;
  }
  settled_streamline();
  pending_ngx();
  unassociated();
  frame_generation_scope();
  readiness();
  render_size();
  std::string rendered;
  for (const auto &line : lines) {
    rendered += line + "\n";
  }
  if (argc > 2 && std::strcmp(argv[2], "--update") == 0) {
    std::ofstream out(argv[1], std::ios::binary);
    out << rendered;
    std::printf("Wrote %zu provider status lines to %s\n", lines.size(), argv[1]);
    return out ? 0 : 1;
  }
  const auto golden = read(argv[1]);
  if (golden == rendered) {
    std::printf("PASS provider status lines: %zu lines byte-identical to the goldens\n", lines.size());
    return 0;
  }
  // Name the first differing line.
  std::istringstream want(golden), got(rendered);
  std::string a, b;
  for (unsigned index = 1;; ++index) {
    const bool more_a = static_cast<bool>(std::getline(want, a)), more_b = static_cast<bool>(std::getline(got, b));
    if (!more_a && !more_b) {
      break;
    }
    if (a != b || more_a != more_b) {
      std::fprintf(stderr, "FAIL status line %u differs\n golden:   %s\n rendered: %s\n", index, more_a ? a.c_str() : "(none)", more_b ? b.c_str() : "(none)");
      return 1;
    }
  }
  std::fputs("FAIL the rendered status lines differ from the goldens\n", stderr);
  return 1;
}

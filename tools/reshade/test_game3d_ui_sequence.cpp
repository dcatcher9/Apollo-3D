// SPDX-License-Identifier: GPL-3.0-only
// Sequence replay of live automatic UI detection's temporal rules
// (docs/reshade-sbs.md, UI protection). Per-frame decision streams drive
// today's production state machines without a GPU: the game session's trust
// (alpha_auto_policy), the renderer's holds, status samples and CPU-owned
// hidden-scene verdicts (ui_temporal::detection_state), the sample decode and
// counter commit (ui_temporal::decode_detection_sample, sample_counters),
// Present pairing under frame generation (ui_mask::pair_hudless_present) and
// the exact counters (ui_counters). GPU decisions are stream inputs: decision texels 0-6 that
// ui_detection_replay --verbose recorded on labelled dumps with the HEAD
// shader, or synthetic texels. This test never derives a shader outcome.
//
// It asserts TODAY's behaviour. An outcome that a later stage of the UI
// decision framework (docs/reshade-sbs.md, UI decision framework: stages
// S0-S6; rules E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1, F1) changes prints
// "KNOWN_TODAY <stage> <rule>: <text>" and does not fail; that stage turns it
// into a strict assertion. Outcomes the rules already call correct, such as
// an accepted source pinning a whole-frame alpha flat over a visible scene
// (P1, the opacity ruling), are asserted strictly.
//
// Informational, not in ctest: --log <ReShade.log>... replays the logged
// "Sunshine UI protection" samples through alpha_auto_policy and prints the
// predicted trust transitions beside the logged ones.
#include "game3d_alpha_auto.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_mask.h"
#include "game3d_ui_temporal.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
  using namespace sunshine_game3d;
  namespace word = ui_detection::decision_word;
  using ui_detection::scene_verdict;
  using ui_temporal::hold_kind;

  // ui_detection_inputs::max_held_presents (game3d_renderer.h, which needs
  // ReShade); game3d_ui_input_provider.cpp ties it to max_generated_frames.
  constexpr std::uint32_t max_held_presents = 3;
  static_assert(max_held_presents == ui_mask::max_generated_frames, "The hold bound moved");
  // The renderer's status sample cadence (game3d_renderer.cpp, detect_ui).
  constexpr std::uint64_t sample_interval_ms = 100;

  void require(bool value, const std::string &message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  std::uint32_t known_today_count = 0;

  void known_today(const char *stage, const char *rule, const std::string &text) {
    std::printf("KNOWN_TODAY %s %s: %s\n", stage, rule, text.c_str());
    ++known_today_count;
  }

  std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
  }

  // Decision texels 0-6 as the renderer reads them back.
  constexpr std::size_t texel_words = 4 * ui_detection::scene_decision_texels;
  using texels = std::array<std::uint32_t, texel_words>;
  constexpr std::size_t scene_texels_begin = 4 * 5;

  texels parse_texels(std::string_view csv) {
    texels result {};
    std::size_t count = 0;
    const char *at = csv.data(), *end = csv.data() + csv.size();
    while (at < end && count < result.size()) {
      const auto parsed = std::from_chars(at, end, result[count]);
      if (parsed.ec != std::errc()) {
        throw std::runtime_error("Recorded texels do not parse");
      }
      ++count;
      at = parsed.ptr;
      if (at < end && *at == ',') {
        ++at;
      }
    }
    if (count != result.size() || at != end) {
      throw std::runtime_error("Recorded texels are not 28 words");
    }
    return result;
  }

  std::uint32_t float_bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  // Texel 5's state word: valid, ran, and the presented image's verdict.
  constexpr std::uint32_t scene_state(bool valid, scene_verdict verdict) {
    return (valid ? 1u : 0u) | 2u | std::uint32_t(verdict) << 2;
  }

  // Recorded decision texels: ui_detection_replay --verbose with the HEAD
  // shader on E:/ApolloDev/sbs_dump (scratchpad s0/P4_replay_after.txt; the
  // "P3" variants of the same dumps are in s0/P3/p3_replay*.txt). Each names
  // its dump and replay case. Word 4 holds the candidate bits and word 17 the
  // trusted slots the frame was decided with.
  namespace recorded {
    // The Witcher 3 Remastered, FG off, graphics settings over a hidden scene
    // (game3d_46312_198402901901355): an opaque offscreen layer, no CPU hold
    // ("P3 W3 graphics settings, layer, no CPU hold") and held ("W3 graphics
    // settings, layer hidden-scene hold").
    constexpr std::string_view w3_settings =
      "0,0,3686400,59,10,2059821,1626579,0,0,3686400,0,3686400,0,0,0,0,0,0,0,"
      "3686400,1565,3165951781,7,3596,0,0,0,0";
    constexpr std::string_view w3_settings_held =
      "8,3686400,3686400,59,10,2059821,1626579,0,0,3686400,0,3686400,0,0,0,0,"
      "0,0,0,3686400,1565,3165951781,7,3596,0,0,0,0";
    // W3 FG off, notice board (game3d_46312_198402901901363): layer 12.48%,
    // untrusted ("W3 notice board, scene visible under holds") and trusted
    // ("W3 notice board FG off, trusted layer, hold").
    constexpr std::string_view w3_notice =
      "2,460125,3686400,0,10,3643197,43203,0,0,460125,0,3686400,0,0,0,0,0,0,0,100337,"
      "2029,1049969278,15,8116,0,0,0,0";
    constexpr std::string_view w3_notice_trusted =
      "2,460125,3686400,0,10,3643197,43203,0,0,460125,0,3686400,0,0,0,0,0,2,0,"
      "100337,2029,1049969278,15,8116,0,0,0,0";
    // W3 FG off, sign wheel over a visible scene (game3d_46312_198402901901353):
    // a full layer, untrusted ("P3 W3 sign wheel, layer, no hold") and trusted
    // ("W3 sign wheel FG off, trusted layer, hold", xfail S2b).
    constexpr std::string_view w3_wheel =
      "0,0,3686400,0,10,3686120,280,0,0,3686400,0,3686400,0,0,0,0,0,0,0,145726,564,"
      "1057143089,15,2256,0,0,0,0";
    constexpr std::string_view w3_wheel_trusted =
      "2,3686400,3686400,0,10,3686120,280,0,0,3686400,0,3686400,0,0,0,0,0,2,0,"
      "145726,564,1057143089,15,2256,0,0,0,0";
    // W3 FG on, HUD (game3d_46312_198402901901357): UIAlpha 5.23% with an
    // inexact HUD-less pair, untrusted and trusted ("W3 HUD FG on, ... UIAlpha, hold").
    constexpr std::string_view w3_hud_fg =
      "1,192930,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,0,"
      "103614,0,472,1056573614,15,1888,472,1056964608,3,0";
    constexpr std::string_view w3_hud_fg_trusted =
      "1,192930,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,"
      "3686400,1,103614,0,472,1056573614,15,1888,472,1056964608,3,0";
    // W3 FG on, sign wheel (game3d_46312_198402901901359): a full UIAlpha,
    // untrusted and trusted ("W3 sign wheel FG on, ... UIAlpha, hold", the
    // trusted one xfail S2b).
    constexpr std::string_view w3_wheel_fg =
      "0,0,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,0,149627,0,"
      "462,1056020436,15,1848,462,1057709052,3,0";
    constexpr std::string_view w3_wheel_fg_trusted =
      "1,3686400,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,"
      "1,149627,0,462,1056020436,15,1848,462,1057709052,3,0";
    // Clair Obscur: Expedition 33, FG on, title over a visible scene
    // (game3d_31636_135749029986373): Backbuffer alpha 2.38%, untrusted
    // ("E33 title FG on, untrusted backbuffer, hold", xfail S2a) and trusted.
    constexpr std::string_view e33_title =
      "3,197797,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,0,0,0,1680,"
      "1061468492,15,6720,0,0,0,0";
    constexpr std::string_view e33_title_trusted =
      "3,197797,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,4,0,"
      "0,1680,1061468492,15,6720,0,0,0,0";
    // E33 FG on, Load Game over a hidden scene (game3d_59540_257918763574026):
    // a full Backbuffer alpha, trusted (flat) and untrusted (empty).
    constexpr std::string_view e33_load_trusted =
      "3,8294400,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,4,0,0,"
      "1753,1000268529,7,6942,0,0,0,0";
    constexpr std::string_view e33_load =
      "0,0,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,0,0,0,1753,"
      "1000268529,7,6942,0,0,0,0";
    // Resident Evil Requiem, dark room (game3d_52696_225539427440975): an
    // offscreen layer 0.18% beside a full presented alpha and an inexact pair,
    // untrusted ("RE9 dark room, untrusted layer, inexact pair, hold", xfail
    // S2a) and trusted ("P3 RE9 dark room, trusted layer, inexact pair").
    constexpr std::string_view re9_room =
      "2,14611,8294400,31,26,2568733,4352259,0,0,14611,0,8294400,0,0,0,0,8294400,0,0,"
      "3540,386,1058855305,15,1544,386,1057529644,3,0";
    constexpr std::string_view re9_room_trusted =
      "2,14611,8294400,31,26,2568733,4352259,0,0,14611,0,8294400,0,0,0,0,8294400,"
      "2,0,3540,386,1058855305,15,1544,386,1057529644,3,0";
  }  // namespace recorded

  // What one frame's detection reads: the adopted inputs and the per-frame
  // bits (game3d_renderer.cpp, detect_ui's b2 constants).
  struct gpu_inputs {
    std::uint32_t bits {}, flags {}, trusted {}, per_frame {};
    std::uint64_t now_ms {};
  };

  using gpu_model = std::function<texels(const gpu_inputs &)>;

  // The recorded frame decided with exactly these candidate bits and trusted
  // slots; a stream that asks for any other combination fails rather than
  // guess the shader's answer.
  gpu_model recorded_frames(std::vector<std::string_view> frames) {
    std::vector<texels> parsed;
    for (const auto frame : frames) {
      parsed.push_back(parse_texels(frame));
    }
    return [parsed](const gpu_inputs &in) {
      for (const auto &t : parsed) {
        if (t[word::candidates] == in.bits && t[word::trusted] == in.trusted) {
          return t;
        }
      }
      throw std::runtime_error("No recorded decision for candidates " + hex(in.bits) + " trusted " + hex(in.trusted));
    };
  }

  // The first phase whose end the frame has not reached decides.
  gpu_model phased(std::vector<std::pair<std::uint64_t, gpu_model>> phases) {
    return [phases](const gpu_inputs &in) {
      for (const auto &phase : phases) {
        if (in.now_ms < phase.first) {
          return phase.second(in);
        }
      }
      return phases.back().second(in);
    };
  }

  // Synthetic decision texels over 1000 pixels. The fields are inputs; they
  // stand for what a stream's GPU decided, not for a shader rule.
  struct synthetic {
    std::uint32_t source {}, covered {}, pixels = 1000;
    std::array<std::uint32_t, 4> alpha_covered {}, alpha_invalid {};
    std::uint32_t hudless_changed {}, hudless_unchanged {};
    std::array<std::uint32_t, 2> opaque {};
    bool scene_valid {};
    scene_verdict verdict = scene_verdict::none;
    float d {};
    std::uint32_t n {}, decided {};

    texels words(const gpu_inputs &in) const {
      texels t {};
      t[word::source] = source;
      t[word::covered] = covered;
      t[word::pixels] = pixels;
      t[word::candidates] = in.bits;
      t[word::hudless_changed] = hudless_changed;
      t[word::hudless_unchanged] = hudless_unchanged;
      for (std::size_t slot = 0; slot != 4; ++slot) {
        t[word::alpha_covered + slot] = alpha_covered[slot];
        t[word::alpha_invalid + slot] = alpha_invalid[slot];
      }
      t[word::trusted] = in.trusted;
      t[word::alpha_opaque] = opaque[0];
      t[word::alpha_opaque + 1] = opaque[1];
      t[word::scene_n] = n;
      t[word::scene_d] = float_bits(d);
      t[word::scene_state] = scene_state(scene_valid, verdict);
      t[word::scene_decided] = decided;
      return t;
    }
  };

  // One Present as render() receives it.
  struct present {
    std::uint64_t now_ms {};
    // Candidate bits after the renderer's compatibility checks: 1, 2, 4, 8
    // alpha slots, 16 a paired HUD-less image, 32 an exact pair.
    std::uint32_t offered {};
    // The stored flags of the UI color slot's source (layer_detection_flags
    // for the offscreen layer, none for a tagged UIColorAndAlpha).
    std::uint32_t color_alpha_flags {};
    // A generated Present (ui_mask::pair_hudless_present, set by the input
    // provider only while a HUD-less tag pairs).
    bool hold_previous {};
    // The provider offered an Auto input (source_alpha_ui).
    bool available = true;
    bool depth_current = true;
    std::uint64_t epoch = 1, revision = 1;
    std::uint32_t viewport = 1;
  };

  struct frame_result {
    std::uint64_t now_ms {};
    bool active {}, held {}, detected {}, polled {}, discarded {}, submitted {};
    ui_temporal::hold_decision hold;
    gpu_inputs gpu;
    // The mask this frame applies: its own decision, the held one, or none.
    std::uint32_t source {}, covered {}, pixels {};
    std::uint32_t scene_hold_bits {};
    // update_alpha_auto's status for this frame.
    alpha_auto_decision consumed;

    bool flat() const {
      return active && pixels && covered == pixels;
    }
  };

  // One renderer's temporal bookkeeping, calling the production code in the
  // order game3d_renderer.cpp does:
  //   render():           layer_slot, trusted_slots, arbitrate, the held bits,
  //                       adopt, detection_active, enter_scope, scene_holds,
  //                       then held(), or poll_detection + scene_holds +
  //                       detect_ui + detected(), or inactive();
  //   end of render:      the detection fence is signaled (detection_awaiting_signal);
  //   update_alpha_auto:  the 500 ms staleness of the latest sample;
  //   poll_detection:     sample_discarded, decode_detection_sample,
  //                       observe_scene, observe_alpha_channels,
  //                       commit_counters (sample_counters);
  //   detect_ui:          the 100 ms sample cadence, scene evidence by
  //                       measure_scene, the counter and CPU snapshot.
  // The frame size, resource preparation and depth are not modelled.
  class sequence {
  public:
    sequence(alpha_auto_policy &session, gpu_model gpu, std::uint32_t lag = 1):
        session(session),
        gpu(std::move(gpu)),
        lag(lag),
        last_trusted(session.trusted_alpha()) {}

    alpha_auto_policy &session;
    gpu_model gpu;
    // Renders after the submitting one before a sample's fence completes.
    std::uint32_t lag;
    // Edits a submitted sample's texels: (texels, tick, scene evidence ran).
    std::function<void(texels &, std::uint64_t, bool)> sample_edit;
    ui_temporal::detection_state temporal;
    std::vector<frame_result> frames;
    std::vector<alpha_auto_decision> samples;  // Every sample read, after observe_scene.

    struct transition {
      std::uint64_t tick;
      std::uint32_t trusted;
    };

    std::vector<transition> trust;  // trusted_alpha() after a sample changed it.
    std::size_t discards {}, commits {}, committed_frames {};

    const frame_result &step(const present &p) {
      frame_result r;
      r.now_ms = p.now_ms;
      const auto index = frames.size();
      std::uint32_t bits = p.offered;
      const bool layer = temporal.layer_slot(bits, p.color_alpha_flags);
      const std::uint32_t trusted = session.trusted_slots(layer);
      r.hold = temporal.arbitrate(bits, trusted, layer, p.hold_previous, max_held_presents);
      std::uint32_t flags = (bits & 2u) ? p.color_alpha_flags : 0u;
      if (r.hold.hold) {
        bits = temporal.bits;
        flags = temporal.flags;
      }
      temporal.adopt(bits, flags, trusted);
      r.active = p.available && bits;
      ++cpu[ui_counter::auto_frames];
      alpha_auto_source observation;
      observation.now_ms = observation.tick_ms = p.now_ms;
      observation.epoch = p.epoch;
      observation.revision = p.revision;
      observation.viewport = p.viewport;
      observation.session = &session;
      if (r.active) {
        temporal.enter_scope(observation);
        const bool shadow = session.first_run();
        r.scene_hold_bits = temporal.scene_holds(p.now_ms);
        if (r.hold.hold) {
          ++cpu[r.hold.kind == hold_kind::generated ? ui_counter::held_generated : r.hold.kind == hold_kind::inexact_after_exact ? ui_counter::held_inexact_after_exact :
                                                                                                                                   ui_counter::held_trusted_missing];
          temporal.held();
          r.held = true;
        } else {
          if (r.hold.cap_reached) {
            ++cpu[ui_counter::held_cap];
          }
          poll(observation, r);
          r.scene_hold_bits = temporal.scene_holds(p.now_ms);
          const std::uint32_t per_frame = r.scene_hold_bits |
                                          (!p.depth_current || p.hold_previous ? ui_detection::per_frame_depth_not_current : 0u);
          detect(observation, per_frame, shadow, index, r);
          temporal.detected();
          r.detected = true;
        }
      } else {
        ++cpu[ui_counter::inactive_no_candidates];
        temporal.inactive();
        applied_source = applied_covered = applied_pixels = 0;
      }
      r.source = applied_source;
      r.covered = applied_covered;
      r.pixels = applied_pixels;
      if (awaiting) {
        awaiting = false;
        ready_frame = index + lag;
      }
      r.consumed = temporal.latest;
      if (ui_temporal::sample_stale(r.consumed.sample_tick_ms, p.now_ms)) {
        r.consumed = {};
        r.consumed.state = alpha_auto_state::collecting;
      }
      frames.push_back(r);
      return frames.back();
    }

  private:
    void poll(const alpha_auto_source &input, frame_result &r) {
      if (!pending || awaiting || frames.size() < ready_frame) {
        return;
      }
      if (ui_temporal::sample_discarded(input, pending_source, temporal.key(), pending_key)) {
        pending = counters_pending = false;
        temporal.latest = {};
        r.discarded = true;
        ++discards;
        return;
      }
      pending = false;
      auto &latest = temporal.latest;
      latest = ui_temporal::decode_detection_sample(pending_texels.data(), pending_texels.size(), pending_source.now_ms, submitted, pending_layer, true);
      temporal.latest_source = pending_source;
      temporal.observe_scene(latest, temporal.pending_scene_actionable);
      session.observe_alpha_channels(latest.evidence, latest.pixels, pending_source.now_ms);
      commit(latest);
      counters_pending = false;
      r.polled = true;
      samples.push_back(latest);
      const auto now_trusted = session.trusted_alpha();
      if (now_trusted != last_trusted) {
        trust.push_back({latest.sample_tick_ms, now_trusted});
      }
      last_trusted = now_trusted;
    }

    void detect(const alpha_auto_source &observation, std::uint32_t per_frame, bool shadow, std::size_t index, frame_result &r) {
      r.gpu = {temporal.bits, temporal.flags, temporal.trusted, per_frame, observation.now_ms};
      texels t = gpu(r.gpu);
      // The reduce's counter words (counting only, SunshineUIDetectionReduceCS).
      ++gpu_words[ui_counter_word::detection_frames];
      if (t[word::source] < ui_counter_word::decided_count) {
        ++gpu_words[ui_counter_word::decided + t[word::source]];
      }
      if (per_frame & ui_detection::per_frame_depth_not_current) {
        ++gpu_words[ui_counter_word::depth_not_current];
      }
      // As the reduce counts them from its decision (the stream's input).
      const auto source = t[word::source];
      if (source >= 1 && source <= 4 && std::uint64_t(t[word::covered]) * 100 >= std::uint64_t(t[word::pixels]) * 99) {
        ++gpu_words[ui_counter_word::full_alpha];
        const auto slot = 1u << (source - 1);
        if ((r.gpu.trusted & r.gpu.bits & slot) && (r.gpu.bits & 48u) == 48u && !t[word::hudless_invalid] && std::uint64_t(t[word::hudless_unchanged]) * 2 >= t[word::pixels]) {
          ++gpu_words[ui_counter_word::trusted_full];
        }
      }
      applied_source = t[word::source];
      applied_covered = t[word::covered];
      applied_pixels = t[word::pixels];
      const auto now = observation.now_ms;
      if (pending || (last_submit && now >= last_submit && now - last_submit < sample_interval_ms)) {
        return;
      }
      const auto measure = temporal.measure_scene(now, shadow);
      temporal.pending_scene_actionable = false;
      const bool evidence = measure.run;
      if (evidence) {
        temporal.pending_scene_actionable = measure.actionable;
      }
      // The reduce zeroes texels 5 and 6; only the evidence pass writes them.
      else {
        std::fill(t.begin() + scene_texels_begin, t.end(), 0u);
      }
      if (sample_edit) {
        sample_edit(t, now, evidence);
      }
      pending_texels = t;
      counters_pending = true;
      pending_words = gpu_words;
      pending_cpu = cpu;
      pending_frame = index;
      pending_source = observation;
      pending_key = temporal.key();
      pending_layer = (temporal.flags & ui_detection::stored_late_layer) != 0;
      pending = awaiting = true;
      last_submit = now;
      ++submitted;
      r.submitted = true;
    }

    // commit_counters: ui_temporal::sample_counters of the CPU counts
    // snapshotted at submission and the GPU words copied under its fence.
    void commit(const alpha_auto_decision &sample) {
      if (!counters_pending) {
        return;
      }
      const auto delta = ui_temporal::sample_counters(sample, pending_cpu, committed_cpu, pending_words, committed_words, pending_source.now_ms);
      committed_cpu = pending_cpu;
      committed_words = pending_words;
      committed_frames = pending_frame + 1;
      ++commits;
      session.add_counters(delta);
    }

    ui_counters cpu, pending_cpu, committed_cpu;
    std::array<std::uint32_t, ui_counter_word::count> gpu_words {}, pending_words {}, committed_words {};
    bool pending {}, awaiting {}, counters_pending {};
    std::size_t ready_frame {}, pending_frame {};
    std::uint64_t last_submit {}, submitted {}, pending_key {};
    alpha_auto_source pending_source;
    bool pending_layer {};
    texels pending_texels {};
    std::uint32_t applied_source {}, applied_covered {}, applied_pixels {};
    std::uint32_t last_trusted {};
  };

  // The session's committed totals equal this stream's frames through the
  // last committed sample, and reconcile.
  void check_counters(sequence &s, const std::string &what) {
    const auto totals = s.session.counters();
    require(totals.reconciled(), what + ": the committed counters do not reconcile");
    ui_counters expected;
    for (std::size_t i = 0; i != s.committed_frames; ++i) {
      const auto &f = s.frames[i];
      ++expected[ui_counter::auto_frames];
      if (!f.active) {
        ++expected[ui_counter::inactive_no_candidates];
      } else if (f.held) {
        ++expected[f.hold.kind == hold_kind::generated ? ui_counter::held_generated : f.hold.kind == hold_kind::inexact_after_exact ? ui_counter::held_inexact_after_exact :
                                                                                                                                      ui_counter::held_trusted_missing];
      } else {
        ++expected[ui_counter::detection_frames];
        ++expected[ui_counter::decided + f.source];
        if (f.hold.cap_reached) {
          ++expected[ui_counter::held_cap];
        }
      }
    }
    std::uint64_t full_alpha = 0;
    for (std::size_t i = 0; i != s.committed_frames; ++i) {
      const auto &f = s.frames[i];
      full_alpha += f.detected && f.source >= 1 && f.source <= 4 && f.pixels && std::uint64_t(f.covered) * 100 >= std::uint64_t(f.pixels) * 99 ? 1 : 0;
    }
    require(totals[ui_counter::full_alpha] == full_alpha, what + ": full_alpha is " + std::to_string(totals[ui_counter::full_alpha]) + ", the stream says " + std::to_string(full_alpha));
    expected[ui_counter::samples] = s.commits;
    for (const auto index : {ui_counter::auto_frames, ui_counter::detection_frames, ui_counter::held_generated, ui_counter::held_inexact_after_exact, ui_counter::held_trusted_missing, ui_counter::held_cap, ui_counter::inactive_no_candidates, ui_counter::samples}) {
      require(totals[index] == expected[index], what + ": counter " + std::to_string(index) + " is " + std::to_string(totals[index]) + ", the stream says " + std::to_string(expected[index]));
    }
    for (std::uint32_t source = 0; source != 10; ++source) {
      require(totals.decided(source) == expected.decided(source), what + ": decided[" + std::to_string(source) + "] differs");
    }
  }

  void run(sequence &s, present p, std::uint64_t from, std::uint64_t to, std::uint64_t interval = 16) {
    for (auto now = from; now < to; now += interval) {
      p.now_ms = now;
      s.step(p);
    }
  }

  // Presents under frame generation. The game's actual cadence decides which
  // Presents are real: each real frame's tags are made right after the
  // previous real Present, then `actual` generated frames are presented,
  // then the real one; a cadence change applies from the next Present. The
  // production pairing classifies each Present against the latest tag from
  // the multiplier the provider reports, which can lag or lead the cadence
  // (dynamic or automatic frame generation, a multiplier change).
  struct fg_pacer {
    // The first Present is a real frame's.
    explicit fg_pacer(std::uint32_t actual):
        presented(1 + actual) {}

    std::uint64_t presented, tagged = 1;

    struct step {
      ui_mask::hudless_present kind;  // As the pairing classified it.
      bool real;  // As the game presented it.
    };

    step present(bool fg_active, std::uint32_t actual, std::uint32_t reported) {
      const auto pairing = ui_mask::pair_hudless_present(tagged, ++presented, fg_active, reported);
      const bool real = presented - tagged >= std::uint64_t(actual) + 1;
      if (real) {
        tagged = presented;
      }
      return {pairing.kind, real};
    }

    // A provider whose reported multiplier matches the cadence.
    ui_mask::hudless_present next(bool fg_active, std::uint32_t generated_frames) {
      const auto result = present(fg_active, generated_frames, generated_frames);
      require(result.real == (result.kind != ui_mask::hudless_present::generated_frame), "The pairing misread a matching cadence");
      return result.kind;
    }
  };

  // A Present the pairing reads as real (or late) offers `real`; a generated
  // one offers `generated_bits` and holds (hold_previous) when the stream
  // pairs a HUD-less image; an unpaired one offers `generated_bits` without
  // its HUD-less image (game3d_ui_input_provider.cpp admits only a pair).
  present fg_present(ui_mask::hudless_present kind, present real, std::uint32_t generated_bits) {
    if (kind == ui_mask::hudless_present::real_frame || kind == ui_mask::hudless_present::earlier_real_frame) {
      return real;
    }
    real.hold_previous = kind == ui_mask::hudless_present::generated_frame && (real.offered & 16u) != 0;
    real.offered = generated_bits;
    return real;
  }

  // The tick of the first sample at least `span` after `from`.
  std::uint64_t first_sample_from(const sequence &s, std::uint64_t from) {
    for (const auto &sample : s.samples) {
      if (sample.sample_tick_ms >= from) {
        return sample.sample_tick_ms;
      }
    }
    return 0;
  }

  // ---------------------------------------------------------------- A1-A3 acceptance (trust)

  synthetic backbuffer_alpha(std::uint32_t covered) {
    synthetic f;
    f.source = 3;  // As decided; an input, not a rule.
    f.covered = covered;
    f.alpha_covered = {0, 0, covered, 0};
    return f;
  }

  void trust_is_earned_by_steady_selective_samples() {
    alpha_auto_policy session;
    std::vector<std::uint32_t> heard;
    session.on_trust_change([&](std::uint32_t bits) {
      heard.push_back(bits);
    });
    sequence s(session, [](const gpu_inputs &in) {
      return backbuffer_alpha(200).words(in);
    });
    present p;
    p.offered = 4;
    run(s, p, 10000, 14000);
    require(s.samples.size() > 3, "Too few samples were read");
    for (std::size_t i = 1; i < s.samples.size(); ++i) {
      require(s.samples[i].sample_tick_ms - s.samples[i - 1].sample_tick_ms >= sample_interval_ms, "Samples came faster than the 100 ms cadence");
    }
    const auto first = s.samples.front().sample_tick_ms;
    const auto earned = first_sample_from(s, first + alpha_trust_span_ms);
    require(s.trust.size() == 1 && s.trust[0].tick == earned && s.trust[0].trusted == 4u, "Backbuffer trust was not earned by the first sample 2 s into a steady selective run");
    require(heard == std::vector<std::uint32_t> {4u}, "The trust listener did not hear the earn once");
    require(session.counters()[ui_counter::trust_earned] == 1, "The earn was not counted once");
    require(s.frames.front().gpu.trusted == 0 && s.frames.back().gpu.trusted == 4u, "Detection did not read the earned trust");
    check_counters(s, "steady selective earn");
  }

  void trust_restarts_on_a_full_sample_or_a_factor_two_break() {
    // A run of selective samples restarts at a full sample, or at a value
    // outside a factor of two of the run; within it the run continues.
    struct variant {
      const char *name;
      std::uint32_t before, after;
      std::uint64_t until;  // The second value lasts until this tick; then the first again.
      bool restarts;
    };

    for (const auto &v : {variant {"full sample", 200, 950, 11150, true}, variant {"factor-two break", 200, 450, 20000, true}, variant {"within a factor of two", 200, 390, 20000, false}}) {
      alpha_auto_policy session;
      sequence s(session, [&v](const gpu_inputs &in) {
        const bool second = in.now_ms >= 11000 && in.now_ms < v.until;
        return backbuffer_alpha(second ? v.after : v.before).words(in);
      });
      present p;
      p.offered = 4;
      run(s, p, 10000, 16000);
      const auto first = s.samples.front().sample_tick_ms;
      std::uint64_t run_start = first;
      if (v.restarts) {
        // The new run starts with the first sample that is selective and
        // after the break (the break value itself when it is selective).
        for (const auto &sample : s.samples) {
          const auto covered = sample.evidence.alpha_covered[2];
          if (sample.sample_tick_ms < 11000) {
            continue;
          }
          if (covered * 10 >= sample.pixels * 9) {
            continue;
          }
          run_start = sample.sample_tick_ms;
          break;
        }
      }
      const auto earned = first_sample_from(s, run_start + alpha_trust_span_ms);
      require(!s.trust.empty() && s.trust[0].tick == earned && s.trust[0].trusted == 4u, std::string("The ") + v.name + " did not earn at the expected sample");
      require(v.restarts == (earned > first_sample_from(s, first + alpha_trust_span_ms)), std::string("The ") + v.name + " restart rule moved");
    }
  }

  void trust_is_revoked_by_contradiction() {
    // A trusted channel covering the whole frame over an exact pair showing at
    // least 75% unchanged: revoked by the first such sample 2 s into the run.
    const auto full_over_pair = [](std::uint32_t unchanged) {
      return [unchanged](const gpu_inputs &in) {
        auto f = backbuffer_alpha(in.now_ms < 13000 ? 200 : 1000);
        f.hudless_unchanged = in.now_ms < 13000 ? 900 : unchanged;
        f.hudless_changed = 1000 - f.hudless_unchanged;
        return f.words(in);
      };
    };
    {
      alpha_auto_policy session;
      std::vector<std::uint32_t> heard;
      session.on_trust_change([&](std::uint32_t bits) {
        heard.push_back(bits);
      });
      sequence s(session, full_over_pair(800));
      present p;
      p.offered = 4 | 16 | 32;
      run(s, p, 10000, 17000);
      const auto first_full = first_sample_from(s, 13000);
      const auto revoked = first_sample_from(s, first_full + alpha_trust_span_ms);
      require(s.trust.size() == 2 && s.trust[0].trusted == 4u && s.trust[1].tick == revoked && !s.trust[1].trusted, "A full claim over a visible exact pair was not revoked 2 s into the run");
      require(heard == std::vector<std::uint32_t> {4u, 0u} && session.counters()[ui_counter::trust_revoked_full] == 1, "The contradiction revocation was not heard or counted once");
      std::uint64_t flat_ms = 0;
      for (const auto &f : s.frames) {
        if (f.now_ms >= 13000 && f.gpu.trusted == 4u) {
          flat_ms = f.now_ms - 13000;
        }
      }
      // An accepted source pins its whole-frame alpha flat (P1, the opacity
      // ruling) until the exact pair's repeated contradictions revoke it (A2):
      // it decides up to the poll of the revoking sample, and not after.
      require(flat_ms + 13000 >= revoked && flat_ms + 13000 <= revoked + sample_interval_ms, "The trusted full claim did not decide until its revocation (" + std::to_string(flat_ms) + " ms)");
      // Its frames before the revocation count as trusted_full.
      const auto c = session.counters();
      require(c[ui_counter::trusted_full] > 0 && c[ui_counter::trusted_full] <= c[ui_counter::full_alpha], "The contradicted full claim was not counted as trusted_full");
      check_counters(s, "full-claim revocation");
    }
    for (const auto &[bits, unchanged, what] : {std::tuple {4u | 16u | 32u, 700u, "a pair under 75% unchanged"}, std::tuple {4u | 16u, 800u, "an inexact pair"}}) {
      alpha_auto_policy session;
      sequence s(session, full_over_pair(unchanged));
      present p;
      p.offered = bits;
      run(s, p, 10000, 20000);
      // A2: only valid same-sample evidence of stronger provenance revokes.
      // A middle-band pair is invalid (V2) and an inexact one never judges
      // (E2); ambiguous or invalid self-doubt is not a revocation.
      require(s.trust.size() == 1 && session.trusted_alpha() == 4u, std::string("A full claim over ") + what + " revoked trust");
    }

    // Presented alpha differing from a trusted dedicated UIAlpha by at least
    // 10% of the frame: revoked by the first such sample 2 s into the run,
    // and not earned again while it disagrees.
    alpha_auto_policy session;
    sequence s(session, [](const gpu_inputs &in) {
      synthetic f;
      f.source = 1;
      f.covered = 100;
      const bool disagree = in.now_ms >= 13000 && in.now_ms < 20000;
      f.alpha_covered = {100, 0, disagree ? 300u : 100u, 0};
      return f.words(in);
    });
    present p;
    p.offered = 1 | 4;
    run(s, p, 10000, 25000);
    require(!s.trust.empty() && s.trust[0].trusted == 5u, "UIAlpha and Backbuffer did not earn together");
    const auto first_disagreement = first_sample_from(s, 13000);
    const auto revoked = first_sample_from(s, first_disagreement + alpha_trust_span_ms);
    require(s.trust.size() >= 2 && s.trust[1].tick == revoked && s.trust[1].trusted == 1u, "Disagreeing presented alpha was not revoked 2 s into the disagreement");
    const auto agreed = first_sample_from(s, 20000);
    require(s.trust.size() == 3 && s.trust[2].tick == first_sample_from(s, agreed + alpha_trust_span_ms) && s.trust[2].trusted == 5u, "Presented alpha earned during disagreement, or not after it");
    const auto c = session.counters();
    require(c[ui_counter::trust_revoked_presented] == 1 && c[ui_counter::trust_earned] == 3, "Presented-alpha revocation or the re-earn was not counted");
    check_counters(s, "presented disagreement");
  }

  void restored_trust_is_provisional() {
    for (const bool reconfirm : {false, true}) {
      alpha_auto_policy session;
      std::vector<std::uint32_t> heard;
      session.on_trust_change([&](std::uint32_t bits) {
        heard.push_back(bits);
      });
      session.restore_trusted_alpha(4u);
      require(heard.empty() && session.counters()[ui_counter::trust_restored] == 1, "A restore called the listener");
      sequence s(session, [reconfirm](const gpu_inputs &in) {
        return backbuffer_alpha(reconfirm ? 200 : 1000).words(in);
      });
      present p;
      p.offered = 4;
      run(s, p, 10000, 75000, 50);
      require(s.frames.front().gpu.trusted == 4u, "Restored trust did not decide from the first frame");
      const auto c = session.counters();
      if (reconfirm) {
        require(s.trust.empty() && heard.empty() && session.trusted_alpha() == 4u && c[ui_counter::trust_earned] == 1 && !c[ui_counter::trust_lapsed], "Restored trust that this session earned again lapsed, or was heard");
      } else {
        // Lapses at the first sample 60 s after the channel was first offered.
        const auto first = s.samples.front().sample_tick_ms;
        const auto lapsed = first_sample_from(s, first + alpha_trust_reconfirm_ms);
        require(s.trust.size() == 1 && s.trust[0].tick == lapsed && !s.trust[0].trusted && heard == std::vector<std::uint32_t> {0u}, "Unconfirmed restored trust did not lapse 60 s after first being offered");
        require(c[ui_counter::trust_lapsed] == 1 && !c[ui_counter::trust_earned], "The lapse was not counted once");
      }
      check_counters(s, reconfirm ? "reconfirmed restore" : "lapsed restore");
    }
  }

  void opaque_tag_proof_prefers_the_layer_and_trust_is_per_source() {
    // A tagged UIColorAndAlpha that is the opaque final image over an exact
    // pair showing most of the scene: once proven, the input provider fills
    // the UI color slot with the offscreen layer instead
    // (game3d_ui_input_provider.cpp, layer_first), which earns trust of its
    // own source (4). The tag never inherits it.
    alpha_auto_policy session;
    const auto layer = ui_detection::layer_detection_flags(false);
    sequence s(session, [](const gpu_inputs &in) {
      synthetic f;
      const bool is_layer = (in.flags & ui_detection::stored_late_layer) != 0;
      f.source = is_layer ? 2 : 0;
      f.covered = is_layer ? 150 : 0;
      f.alpha_covered = {0, is_layer ? 150u : 1000u, 0, 0};
      f.hudless_unchanged = 600;
      f.hudless_changed = 400;
      return f.words(in);
    });
    std::uint64_t switched = 0;
    for (std::uint64_t now = 10000; now < 16000; now += 16) {
      present p;
      p.now_ms = now;
      p.offered = 2 | 16 | 32;
      if (session.prefer_ui_layer()) {
        p.color_alpha_flags = layer;
        if (!switched) {
          switched = now;
        }
      }
      s.step(p);
    }
    const auto first = s.samples.front().sample_tick_ms;
    require(switched && session.counters()[ui_counter::trust_opaque_set] == 1, "The opaque final-image tag was not proven once");
    // The poll that proves it is the first sample 2 s into the run; the next
    // render offers the layer.
    std::uint64_t proof = 0;
    for (const auto &f : s.frames) {
      if (f.polled && f.now_ms < switched) {
        proof = f.now_ms;
      }
    }
    require(proof >= first + alpha_trust_span_ms, "The opaque proof came before 3 samples over 2 s");
    known_today("S1", "E1/A1",
                "the offscreen layer is offered only after the tagged UI color is proven opaque (from +" + std::to_string(switched - 10000) +
                  " ms): the two share the UI color slot; with one slot per signature both are offered and the "
                  "never-selective tag is simply never accepted");
    require(session.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source && session.trusted_slots(true) == 2u && !session.trusted_slots(false), "The layer's trust was not its own source's");
    present tag;
    tag.now_ms = 16000;
    tag.offered = 2 | 16 | 32;
    require(!s.step(tag).gpu.trusted, "The tagged UI color inherited the layer's trust");
    check_counters(s, "opaque proof");
  }

  void trust_is_not_context_keyed() {
    alpha_auto_policy session;
    sequence s(session, [](const gpu_inputs &in) {
      return backbuffer_alpha(200).words(in);
    });
    present hdr;
    hdr.offered = 4;
    run(s, hdr, 10000, 13000);
    require(session.trusted_alpha() == 4u, "Backbuffer trust was not earned in the first context");
    // An HDR to SDR switch recreates the swapchain: a new epoch and scope.
    present sdr = hdr;
    sdr.epoch = 2;
    sdr.now_ms = 13000;
    const auto &first = s.step(sdr);
    require(first.consumed.state == alpha_auto_state::collecting, "The scope change kept the previous context's sample");
    require(first.gpu.trusted == 4u, "Trust did not carry into the new context");
    known_today("S1", "A1",
                "trust earned in one context (HDR) decides from the first frame of another (SDR): it is keyed by game "
                "and source kind, not by source signature with the swapchain colour space");
  }

  // ---------------------------------------------------------------- holds

  // A HUD-less difference decides on exact pairs; other frames decide none.
  texels difference_frame(const gpu_inputs &in) {
    synthetic f;
    const bool exact = (in.bits & 48u) == 48u;
    f.source = exact ? 5 : 0;
    f.covered = exact ? 50 : 0;
    f.hudless_changed = 50;
    f.hudless_unchanged = 950;
    return f.words(in);
  }

  void generated_presents_hold_per_multiplier() {
    alpha_auto_policy session;
    sequence s(session, difference_frame);
    fg_pacer pacer(1);
    present real;
    real.offered = 8 | 16 | 32;  // Current alpha and a same-batch HUD-less pair.
    std::uint64_t now = 10000;
    for (const std::uint32_t generated : {1u, 2u, 3u, 1u}) {
      std::size_t generated_presents = 0, real_presents = 0, since_real = 0;
      const auto until = now + 1000;
      auto kind = ui_mask::hudless_present::unpaired;
      // Change the multiplier only after a real Present.
      while (now < until || kind == ui_mask::hudless_present::generated_frame) {
        kind = pacer.next(true, generated);
        auto p = fg_present(kind, real, 8u);
        p.now_ms = now;
        now += 8;
        const auto &r = s.step(p);
        if (kind == ui_mask::hudless_present::generated_frame) {
          ++generated_presents;
          ++since_real;
          require(r.held && r.hold.kind == hold_kind::generated && r.source == 5 && s.temporal.bits == real.offered && s.temporal.holds == since_real, "A generated Present did not hold the real frame's inputs and mask");
        } else {
          require(kind == ui_mask::hudless_present::real_frame && r.detected && r.source == 5 && !s.temporal.holds, "A real Present did not detect afresh");
          require(since_real == generated || s.frames.size() == 1, "The pairing did not give " + std::to_string(generated) + " generated Presents per real one");
          since_real = 0;
          ++real_presents;
        }
        require(!r.hold.cap_reached, "A hold within one real frame reached the cap");
      }
      require(generated_presents && real_presents, "A multiplier segment had no Presents");
    }
    require(!s.discards, "A generated Present's held inputs discarded a sample");
    check_counters(s, "frame generation holds");
    require(session.counters()[ui_counter::held_generated] > 0, "No generated hold was committed");
  }

  // What a frame-generation stream did to the Presents the game actually
  // generated and to its real ones.
  struct fg_tally {
    std::size_t presents {}, real {}, held {}, misread {}, lost {}, extra_holds {}, paired_real {}, cap {};
  };

  // Present-paired HUD-less difference under frame generation (inexact: no
  // tag batch). cadence(index) gives the game's actual generated frames and
  // the provider's reported ones for the Present with that index; tallies
  // count from Present `from`.
  fg_tally fg_drift(sequence &s, std::size_t count, std::size_t from, const std::function<std::pair<std::uint32_t, std::uint32_t>(std::size_t)> &cadence) {
    fg_pacer pacer(cadence(0).first);
    present real;
    real.offered = 8 | 16;
    fg_tally t;
    std::uint64_t now = 10000;
    for (std::size_t index = 0; index != count; ++index) {
      const auto [actual, reported] = cadence(index);
      const auto step = pacer.present(true, actual, reported);
      auto p = fg_present(step.kind, real, 8u);
      p.now_ms = now;
      now += 8;
      const auto &r = s.step(p);
      if (index < from) {
        continue;
      }
      ++t.presents;
      t.cap += r.hold.cap_reached ? 1 : 0;
      const bool paired = r.detected && (r.gpu.bits & 16u);
      if (step.real) {
        ++t.real;
        t.extra_holds += r.held ? 1 : 0;
        t.paired_real += paired ? 1 : 0;
      } else if (r.held) {
        ++t.held;
      } else {
        ++t.lost;
        t.misread += paired ? 1 : 0;
      }
    }
    return t;
  }

  // Decides a HUD-less difference wherever a pair is offered.
  texels pair_difference_frame(const gpu_inputs &in) {
    synthetic f;
    const bool pair = (in.bits & 16u) != 0;
    f.source = pair ? 5 : 0;
    f.covered = pair ? 50 : 0;
    f.hudless_changed = 50;
    f.hudless_unchanged = 950;
    return f.words(in);
  }

  void generated_presents_under_multiplier_drift() {
    {
      // 4x, reported as presented: every generated Present holds.
      alpha_auto_policy session;
      sequence s(session, pair_difference_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 3u};
      });
      require(t.real == 101 && t.held == 300 && !t.lost && !t.extra_holds && t.paired_real == 101 && !t.cap, "A 4x cadence reported as 4x did not hold every generated Present");
      check_counters(s, "4x matching cadence");
    }
    {
      // Presented at 4x while the provider still reports 2x (dynamic frame
      // generation): Presents 2 and 3 of each real frame pair as real or late.
      alpha_auto_policy session;
      sequence s(session, pair_difference_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 1u};
      });
      require(t.real == 101 && t.held == 100 && t.misread == 200 && t.lost == 200 && !t.extra_holds && t.paired_real == 101 && !t.cap, "The lagging multiplier's pairing moved");
      known_today("S3", "T1/E2", "presented at 4x but reported as 2x, " + std::to_string(t.misread) + " of " + std::to_string(t.held + t.lost) + " generated Presents pair as a real frame (or a late one) and detection reads an interpolated image instead of holding");
      check_counters(s, "lagging multiplier");
    }
    {
      // Presented at 2x while the provider reports 4x, after a matching
      // start: a real Present is held as a generated one, the hold reaches
      // the cap and the mask is lost while the multiplier stays wrong.
      alpha_auto_policy session;
      sequence s(session, pair_difference_frame);
      const auto t = fg_drift(s, 300, 100, [](std::size_t index) {
        return std::pair {1u, index < 100 ? 1u : 3u};
      });
      for (std::size_t i = 104; i != s.frames.size(); ++i) {
        require(!s.frames[i].source && !s.frames[i].held, "A mask survived the capped hold of a leading multiplier");
      }
      require(t.real == 100 && t.extra_holds == 1 && t.cap == 1 && !t.paired_real && !t.misread, "The leading multiplier's holds moved");
      known_today("S2a", "T1",
                  "presented at 2x but reported as 4x, a real Present is held as generated, the hold reaches the cap of 3 and "
                  "the HUD-less mask is lost on all " +
                    std::to_string(t.real) + " later real Presents (T1 has no multiplier constant; S3 identity fixes the pairing)");
      check_counters(s, "leading multiplier");
    }
    {
      // 2x to 4x in the middle of a real frame (after its generated Present),
      // reported one real frame late: two generated Presents pair as real or
      // late, then every generated Present holds again.
      alpha_auto_policy session;
      sequence s(session, pair_difference_frame);
      constexpr std::size_t change = 100, reported_from = 103;
      const auto t = fg_drift(s, 300, 0, [](std::size_t index) {
        return std::pair {index < change ? 1u : 3u, index < reported_from ? 1u : 3u};
      });
      // Present 102 is the first real one after the change: Presents 100 and 101 are generated.
      require(t.misread == 2 && !s.frames[change].held && !s.frames[change + 1].held && s.frames[change + 3].held && !t.extra_holds && !t.cap && t.paired_real == t.real, "The mid-frame multiplier change's pairing moved");
      for (std::size_t i = reported_from; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.held || (f.detected && (f.gpu.bits & 16u)), "A Present after the reported multiplier caught up neither held nor paired");
      }
      // Clearing nothing is right: the FG multiplier is not part of the scope.
      known_today("S3", "T1/E2", "a multiplier change in the middle of a real frame, reported a real frame late, pairs " + std::to_string(t.misread) + " generated Presents as real or late (Present counting, not real-frame identity)");
      check_counters(s, "mid-frame multiplier change");
    }
  }

  void holds_are_capped() {
    alpha_auto_policy session;
    sequence s(session, difference_frame);
    present exact;
    exact.offered = 8 | 16 | 32;
    exact.now_ms = 10000;
    require(s.step(exact).source == 5, "The exact pair did not decide");
    present generated;
    generated.offered = 8;
    generated.hold_previous = true;
    std::vector<frame_result> run_of;
    for (std::uint32_t i = 0; i != 5; ++i) {
      generated.now_ms = 10008 + 8 * i;
      run_of.push_back(s.step(generated));
    }
    for (std::uint32_t i = 0; i != max_held_presents; ++i) {
      require(run_of[i].held && run_of[i].source == 5, "A generated Present within the cap did not hold");
    }
    require(!run_of[3].held && run_of[3].hold.cap_reached && run_of[3].hold.kind == hold_kind::generated && run_of[3].detected && run_of[3].gpu.bits == 8u && !run_of[3].source, "The Present past the cap did not detect alone");
    require(!run_of[4].held && !run_of[4].hold.cap_reached && run_of[4].hold.kind == hold_kind::none, "A mask detected without a HUD-less pair was holdable");
    known_today("S2a", "T1",
                "a run of more than max_held_presents (3) generated Presents detects alone on the 4th and loses "
                "the mask; T1 holds by real-frame identity with no multiplier constant");
    // DLSS multi-frame generation beyond 4x: the pairing counts at most 3
    // generated Presents, so the 4th generated Present pairs as the real one.
    using ui_mask::hudless_present;
    const hudless_present expected[] {hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::real_frame, hudless_present::earlier_real_frame, hudless_present::earlier_real_frame};
    for (std::uint64_t after = 1; after <= 6; ++after) {
      const auto pairing = ui_mask::pair_hudless_present(100, 100 + after, true, 5);
      require(pairing.kind == expected[after - 1] && pairing.presents_ago == (after > 4 ? after - 4 : 0), "The 6x frame generation pairing moved");
    }
    known_today("S2a", "T1",
                "with 6x frame generation (5 generated) Present 4 of a real frame pairs as real and Presents 5-6 "
                "pair late: detection reads an interpolated image instead of holding (S3 stamps fix the pairing)");
    check_counters(s, "capped holds");
  }

  void inexact_after_exact_and_trusted_missing_holds() {
    {
      alpha_auto_policy session;
      sequence s(session, difference_frame);
      present p;
      p.offered = 8 | 16 | 32;
      p.now_ms = 10000;
      s.step(p);
      p.offered = 8 | 16;
      std::vector<frame_result> inexact;
      for (std::uint32_t i = 0; i != 5; ++i) {
        p.now_ms = 10016 + 16 * i;
        inexact.push_back(s.step(p));
      }
      for (std::uint32_t i = 0; i != 3; ++i) {
        require(inexact[i].held && inexact[i].hold.kind == hold_kind::inexact_after_exact && inexact[i].source == 5, "An inexact pair after an exact decision did not hold");
      }
      require(inexact[3].detected && inexact[3].hold.cap_reached && !s.temporal.exact, "The inexact pair past the cap did not detect");
      require(inexact[4].detected && !inexact[4].hold.cap_reached && inexact[4].hold.kind == hold_kind::none, "An inexact pair after an inexact decision held");
      // A generated Present names the hold first.
      p.offered = 8 | 16 | 32;
      p.now_ms = 10200;
      s.step(p);
      p.offered = 8 | 16;
      p.hold_previous = true;
      p.now_ms = 10216;
      require(s.step(p).hold.kind == hold_kind::generated, "Hold kinds lost their priority");
      check_counters(s, "inexact after exact");
    }
    {
      // A trusted Backbuffer that decided and is missing holds; a trusted,
      // admitted channel in the frame decides by itself and nothing is held.
      alpha_auto_policy session;
      session.restore_trusted_alpha(1u | 4u);
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(in.bits & 4u ? 200 : 0).words(in);
      });
      present p;
      p.offered = 4;
      p.now_ms = 10000;
      require(s.step(p).detected && s.temporal.mask_ready, "A trusted Backbuffer decision was not holdable");
      p.offered = 8;
      std::vector<frame_result> missing;
      for (std::uint32_t i = 0; i != 4; ++i) {
        p.now_ms = 10016 + 16 * i;
        missing.push_back(s.step(p));
      }
      for (std::uint32_t i = 0; i != 3; ++i) {
        require(missing[i].held && missing[i].hold.kind == hold_kind::trusted_missing && missing[i].source == 3, "A missing trusted channel did not hold");
      }
      require(missing[3].hold.cap_reached && missing[3].detected, "The trusted-missing hold was not capped");
      p.offered = 4;
      p.now_ms = 10100;
      s.step(p);
      p.offered = 1 | 8;
      p.hold_previous = true;
      p.now_ms = 10116;
      const auto &own = s.step(p);
      require(own.detected && own.hold.kind == hold_kind::none && own.gpu.bits == (1u | 8u), "A trusted channel in the frame did not decide by itself");
      check_counters(s, "trusted missing");
    }
  }

  void untrusted_inferred_masks_and_e33_streams() {
    // E33 FG on 2x: the Backbuffer tag comes with real Presents only and no
    // HUD-less image pairs, so generated Presents offer nothing.
    const auto e33 = phased({{14000, recorded_frames({recorded::e33_title, recorded::e33_title_trusted})}, {UINT64_MAX, recorded_frames({recorded::e33_load, recorded::e33_load_trusted})}});
    alpha_auto_policy session;
    sequence s(session, e33);
    fg_pacer pacer(1);
    present real;
    real.offered = 4;
    std::size_t untrusted_real = 0, untrusted_generated_unmasked = 0;
    // The real frame that reads the earning sample adopted its trust before
    // the poll, so its own mask is not yet holdable.
    bool previous_real_trusted = false;
    for (std::uint64_t now = 10000; now < 17000; now += 8) {
      const auto kind = pacer.next(true, 1);
      auto p = fg_present(kind, real, 0u);
      p.now_ms = now;
      const bool trusted = session.trusted_alpha() & 4u;
      const auto &r = s.step(p);
      const bool generated = kind == ui_mask::hudless_present::generated_frame;
      if (now < 14000 && !trusted) {
        if (!generated) {
          require(r.detected && r.source == 3 && r.covered == 197797 && !r.gpu.trusted, "The title's untrusted alpha moved");
          ++untrusted_real;
        } else {
          require(!r.active && !r.source && r.hold.kind == hold_kind::none, "An untrusted mask was held");
          ++untrusted_generated_unmasked;
        }
      } else if (trusted) {
        if (!generated) {
          require(r.detected && r.source == 3 && r.gpu.trusted == 4u, "The trusted Backbuffer did not decide");
        } else if (previous_real_trusted) {
          require(r.held && r.hold.kind == hold_kind::trusted_missing, "A generated Present lost the trusted mask");
        } else {
          require(!r.active, "A generated Present held a mask decided before trust");
        }
        if (now >= 14100) {
          require(r.flat(), "Load Game over a hidden scene was not flat with trusted Backbuffer alpha");
        }
      }
      if (!generated) {
        previous_real_trusted = r.gpu.trusted == 4u;
      }
    }
    require(untrusted_real && untrusted_generated_unmasked && session.trusted_alpha() == 4u, "The E33 title did not earn Backbuffer trust");
    known_today("S2a", "A1/S1", "untrusted inferred Backbuffer alpha decides the E33 title at once (" + std::to_string(untrusted_real) + " real Presents before acceptance)");
    known_today("S2a", "T1", "an untrusted inferred mask is not holdable: " + std::to_string(untrusted_generated_unmasked) + " generated Presents of the E33 title had no mask between partial real ones (T1 holds carry no trust bits)");
    const auto counted = session.counters();
    require(!counted[ui_counter::trust_revoked_full], "Load Game revoked the Backbuffer without an exact pair");
    // Load Game is a whole-frame alpha from the accepted Backbuffer, pinned
    // flat (P1) over a hidden scene: the counters measure it hidden, never
    // visible.
    require(counted[ui_counter::full_alpha] > 0 && counted[ui_counter::full_alpha_d_hidden] > 0 && !counted[ui_counter::full_alpha_d_visible], "The counters did not record Load Game's whole-frame alpha over a hidden scene");
    check_counters(s, "E33 title and Load Game");

    // Booted straight into Load Game: untrusted full alpha decides nothing (A1:
    // a source that has only been opaque is never accepted).
    alpha_auto_policy fresh;
    sequence boot(fresh, recorded_frames({recorded::e33_load, recorded::e33_load_trusted}));
    present load;
    load.offered = 4;
    run(boot, load, 10000, 13000);
    for (const auto &f : boot.frames) {
      require(!f.source, "Load Game without trust was not empty");
    }
  }

  // ---------------------------------------------------------------- staleness

  void samples_go_stale_and_are_discarded() {
    {
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      });
      present p;
      p.offered = 4;
      run(s, p, 10000, 11000);
      s.lag = 1000000;  // The next sample never completes.
      run(s, p, 11000, 12000);
      const auto tick = s.samples.back().sample_tick_ms;
      std::size_t last_poll = 0;
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        if (s.frames[i].polled) {
          last_poll = i;
        }
      }
      require(s.frames.back().now_ms > tick + ui_temporal::sample_fresh_ms, "The stream ended before the sample went stale");
      for (std::size_t i = last_poll; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        if (f.now_ms <= tick + ui_temporal::sample_fresh_ms) {
          require(f.consumed.sample_tick_ms == tick, "A sample stopped describing frames within 500 ms");
        } else {
          require(f.consumed.state == alpha_auto_state::collecting && !f.consumed.sample_tick_ms, "A sample described frames more than 500 ms after its tick");
        }
      }
    }
    {
      // Completing 40 frames (640 ms) later, every sample is stale on arrival.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      },
                 40);
      present p;
      p.offered = 4;
      run(s, p, 10000, 20000);
      require(s.samples.empty() && s.discards > 5 && !session.trusted_alpha() && !session.counters()[ui_counter::samples], "A sample older than 500 ms on arrival was read");
    }
    {
      // A change of the deciding inputs discards the pending sample.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      },
                 3);
      present p;
      p.offered = 4;
      p.now_ms = 10000;
      require(s.step(p).submitted, "The first frame did not submit a sample");
      session.restore_trusted_alpha(4u);
      for (std::uint32_t i = 1; i != 5; ++i) {
        p.now_ms = 10000 + 16 * i;
        s.step(p);
      }
      require(s.discards == 1 && s.samples.empty(), "A sample from other deciding inputs was read");
    }
  }

  // ---------------------------------------------------------------- H1 hidden scene

  void hidden_scene_layer_route() {
    // W3 graphics settings: an opaque untrusted layer over a hidden scene.
    // Phases by sample tick: hidden, invalid, hidden, visible, hidden while
    // refuted, an overlay sample (slot 1 below opaque), hidden again.
    const auto open = parse_texels(recorded::w3_settings), held = parse_texels(recorded::w3_settings_held);
    alpha_auto_policy session;
    sequence s(session, [&](const gpu_inputs &in) {
      require(in.bits == 10u && !in.trusted, "The W3 settings stream offered other inputs");
      return (in.per_frame & ui_detection::per_frame_scene_hold) ? held : open;
    });
    // Synthetic edits of the recorded sample: invalid evidence, one visible
    // sample (D 0.5), and overlay samples (slot 1 below opaque).
    bool visible_sent = false;
    s.sample_edit = [&visible_sent](texels &t, std::uint64_t tick, bool evidence) {
      if (tick >= 16000 && tick < 16200) {
        t[word::alpha_opaque + 1] = 0;
      }
      if (!evidence) {
        return;
      }
      if (tick >= 13000 && tick < 14000) {
        t[word::scene_state] = scene_state(false, scene_verdict::none);
      }
      if (tick >= 14500 && !visible_sent) {
        visible_sent = true;
        t[word::scene_state] = scene_state(true, scene_verdict::visible);
        t[word::scene_d] = float_bits(.5f);
      }
    };
    present p;
    p.offered = 2 | 8;
    p.color_alpha_flags = ui_detection::layer_detection_flags(false);
    std::uint64_t entry_poll = 0, first_hidden = 0;
    std::size_t visible_releases = 0;
    for (std::uint64_t now = 10000; now < 17500; now += 16) {
      p.now_ms = now;
      const auto before = s.temporal.scene_hold_until;
      const auto &r = s.step(p);
      if (r.polled) {
        const auto &sample = s.samples.back();
        const auto &scene = sample.evidence.scene;
        const bool hidden = scene.valid && scene.verdict == scene_verdict::hidden;
        const bool refuted = s.temporal.scene_refuted_slots != 0;
        if (hidden && !refuted && sample.evidence.alpha_opaque[1]) {
          require(s.temporal.scene_hold_until[0] == sample.sample_tick_ms + ui_detection::scene::hold_ms, "A valid hidden sample did not hold the layer route for 500 ms from its tick");
          if (!first_hidden) {
            first_hidden = sample.sample_tick_ms;
            entry_poll = now;
          }
        }
        if (scene.valid && scene.verdict == scene_verdict::visible) {
          require(!s.temporal.scene_hold_until[0] && !s.temporal.scene_hold_until[1] && s.temporal.scene_refuted_slots == 2u, "A visible sample did not release the routes and refute the layer");
          ++visible_releases;
        }
        if (!scene.valid) {
          require(s.temporal.scene_hold_until == before, "Invalid evidence changed a hold");
        }
      }
      // The mask follows the held verdict exactly.
      const bool hold = s.temporal.scene_hold_until[0] && now <= s.temporal.scene_hold_until[0];
      require(r.source == (hold ? 8u : 0u), "The frame's mask did not follow the CPU-held verdict at " + std::to_string(now));
      if (now >= 15000 && now < 16000) {
        require(!r.source, "A refuted layer held its route");
      }
    }
    require(first_hidden && s.samples.front().evidence.scene.ran == false && s.samples[1].evidence.scene.ran, "The first sample ran scene evidence before a gate was seen open");
    std::size_t flat_before = 0;
    for (const auto &f : s.frames) {
      if (f.now_ms < entry_poll && f.source) {
        ++flat_before;
      }
    }
    require(!flat_before && s.frames[(entry_poll - 10000) / 16].source == 8u, "The route did not enter on the first hidden read");
    known_today("S2b", "H1",
                "one valid hidden sample enters the full-frame layer route (W3 graphics settings, flat from the "
                "poll of the sample taken at +" +
                  std::to_string(first_hidden - 10000) + " ms); H1 wants 2");
    bool rearmed = false;
    for (const auto &f : s.frames) {
      if (f.now_ms >= 16200 && f.source == 8u) {
        rearmed = true;
      }
    }
    require(visible_releases == 1 && rearmed, "A refuted slot did not re-arm after an overlay sample");
    const auto c = session.counters();
    require(c[ui_counter::full_d_visible] == 1, "The releasing sample was not counted as full while visible");
    check_counters(s, "W3 layer route");
  }

  void hidden_scene_shadow_and_clears() {
    // A gate-open, hidden synthetic stream (an opaque untrusted layer).
    const auto hidden_layer = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha_covered = {0, 1000, 0, 0};
      f.opaque = {0, 1000};
      f.scene_valid = true;
      f.verdict = scene_verdict::hidden;
      f.d = -.02f;
      f.n = 300;
      f.decided = 600;
      return f.words(in);
    };
    {
      // The first-run shadow measures every sample, never holds on a sample
      // it alone measured, and tracks the uncovered hidden run.
      alpha_auto_policy session;
      session.set_first_run(true);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.scene_valid = true;
        f.verdict = scene_verdict::hidden;
        f.n = 300;
        f.decided = 600;
        return f.words(in);
      });
      present p;
      p.offered = 8;
      run(s, p, 10000, 11000);
      const auto first = s.samples.front().sample_tick_ms;
      for (const auto &sample : s.samples) {
        require(sample.evidence.scene.ran && sample.evidence.shadow_hidden_ms == sample.sample_tick_ms - first, "The first-run shadow did not measure every sample");
      }
      require(!s.temporal.scene_hold_until[0] && !s.temporal.scene_hold_until[1], "The shadow held a route");
      alpha_auto_policy gated;
      gated.set_first_run(true);
      sequence g(gated, hidden_layer);
      p.offered = 2 | 8;
      p.color_alpha_flags = ui_detection::layer_detection_flags(false);
      std::size_t polls = 0;
      for (std::uint64_t now = 10000; now < 11000 && polls < 2; now += 16) {
        p.now_ms = now;
        if (!g.step(p).polled) {
          continue;
        }
        ++polls;
        require(polls == 1 ? !g.temporal.scene_hold_until[0] : g.temporal.scene_hold_until[0] != 0, "A shadow-only sample held, or the next actionable one did not");
      }
    }
    alpha_auto_policy session;
    sequence s(session, hidden_layer);
    present p;
    p.offered = 2 | 8;
    p.color_alpha_flags = ui_detection::layer_detection_flags(false);
    std::uint64_t now = 10000;
    const auto establish = [&] {
      for (const auto until = now + 1000; now < until; now += 16) {
        p.now_ms = now;
        if (s.step(p).scene_hold_bits) {
          return;
        }
      }
      throw std::runtime_error("The layer route did not hold");
    };
    const auto cleared = [&](const char *what) {
      require(!s.temporal.scene_hold_until[0] && !s.temporal.scene_hold_until[1] && !s.temporal.scene_holds(now), std::string(what) + " kept a hidden-scene hold");
    };
    const auto step = [&](present q) {
      q.now_ms = now += 16;
      return s.step(q);
    };
    establish();
    auto route = p;
    route.offered = 1 | 2 | 8;  // UIAlpha appears: the routes' inputs change.
    step(route);
    cleared("A route-key change");
    for (const auto &[field, what] : {std::pair {0, "An epoch change"}, std::pair {1, "A revision change"}, std::pair {2, "A viewport change"}}) {
      establish();
      if (field == 0) {
        ++p.epoch;
      } else if (field == 1) {
        ++p.revision;
      } else {
        ++p.viewport;
      }
      const auto &r = step(p);
      cleared(what);
      require(r.consumed.state == alpha_auto_state::collecting, std::string(what) + " kept the previous scope's sample");
    }
    establish();
    auto inactive = p;
    inactive.available = false;
    step(inactive);
    cleared("An inactive frame");
    require(!s.temporal.mask_ready && !s.temporal.exact, "An inactive frame kept a holdable mask");
    // A visible sample refutes the slot; a route change clears the refutation.
    s.sample_edit = [](texels &t, std::uint64_t, bool evidence) {
      if (evidence) {
        t[word::scene_state] = scene_state(true, scene_verdict::visible);
      }
    };
    for (const auto until = now + 500; now < until;) {
      step(p);
    }
    require(s.temporal.scene_refuted_slots == 2u, "A visible sample did not refute the layer slot");
    step(route);
    require(!s.temporal.scene_refuted_slots, "A route change kept a refutation");
    known_today("S2b", "H1",
                "a route-key change (UIAlpha offered) clears the held hidden-scene verdict and the slot's refutation; "
                "H1 clears D state only on a scope change and refutes per signature until it shows below 99% opaque");
  }

  // ---------------------------------------------------------------- recorded trust streams

  void recorded_full_menus_after_earned_trust() {
    const auto layer = ui_detection::layer_detection_flags(false);
    {
      // W3 FG off: the notice board's layer earns trust; the sign wheel's
      // full layer (a 65% backdrop under a solid wheel) then pins flat over a
      // visible scene and is never revoked. Correct under the opacity ruling:
      // an accepted source pins saturate(8 alpha) at any coverage (P1), and
      // nothing of stronger provenance contradicts it (A2).
      const auto w3 = phased({{14000, recorded_frames({recorded::w3_notice, recorded::w3_notice_trusted})}, {UINT64_MAX, recorded_frames({recorded::w3_wheel, recorded::w3_wheel_trusted})}});
      alpha_auto_policy session;
      sequence s(session, w3);
      present p;
      p.offered = 2 | 8;
      p.color_alpha_flags = layer;
      run(s, p, 10000, 17000);
      require(s.frames.front().source == 2 && !s.frames.front().gpu.trusted, "The untrusted notice board layer did not decide");
      require(session.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source && s.trust.size() == 1, "The notice board layer did not earn trust once");
      for (const auto &f : s.frames) {
        if (f.now_ms >= 14000) {
          require(f.flat() && f.source == 2, "The trusted sign wheel layer was not flat");
        }
      }
      known_today("S2a", "A1/S1", "the untrusted W3 notice board layer decides at once, before acceptance");
      // The counters see it: whole-frame alpha frames, and the samples after
      // the first one measure the visible scene beneath (a diagnostic only).
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_alpha_d_visible] > 0 && c[ui_counter::full_alpha_d_invalid] == 1 && !c[ui_counter::full_alpha_d_hidden] && !c[ui_counter::trusted_full], "The counters did not record the trusted full layer over a visible scene");
      require(!c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented] && session.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source, "The trusted sign wheel layer was revoked");
      check_counters(s, "W3 FG off");
      // A session that never saw the HUD keeps the sign wheel 3D: a source
      // that has only been opaque or ambiguous is never accepted (A1).
      alpha_auto_policy fresh;
      sequence boot(fresh, w3);
      run(boot, p, 14000, 17000);
      for (const auto &f : boot.frames) {
        require(!f.source, "The untrusted sign wheel was not empty");
      }
    }
    {
      // W3 FG on 2x: the HUD's UIAlpha earns trust; the sign wheel's full
      // UIAlpha then pins flat on real Presents (P1) and is held on generated
      // ones (T1). The inexact pair never judges it (E2, A2).
      const auto w3 = phased({{14000, recorded_frames({recorded::w3_hud_fg, recorded::w3_hud_fg_trusted})}, {UINT64_MAX, recorded_frames({recorded::w3_wheel_fg, recorded::w3_wheel_fg_trusted})}});
      alpha_auto_policy session;
      sequence s(session, w3);
      fg_pacer pacer(1);
      present real;
      real.offered = 1 | 8 | 16;  // UIAlpha, current and an inexact HUD-less pair.
      for (std::uint64_t now = 10000; now < 17000; now += 8) {
        const auto kind = pacer.next(true, 1);
        auto p = fg_present(kind, real, 8u);
        p.now_ms = now;
        const auto &r = s.step(p);
        require(r.held == (kind == ui_mask::hudless_present::generated_frame), "A W3 FG Present was held wrongly");
        if (now < 14000) {
          require(r.source == 1 && r.covered == 192930, "The W3 HUD UIAlpha did not decide");
        } else if (now >= 14100) {
          require(r.flat() && r.source == 1, "The trusted W3 FG on sign wheel was not flat");
        }
      }
      require(session.trusted_alpha() == 1u && s.trust.size() == 1, "The W3 HUD UIAlpha did not earn trust once");
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_alpha_d_visible] > 0 && !c[ui_counter::full_alpha_d_hidden], "The counters did not record the trusted full UIAlpha over a visible scene");
      require(!c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented], "The trusted sign wheel UIAlpha was revoked");
      check_counters(s, "W3 FG on");
    }
    {
      // RE9 dark room: the offscreen layer (0.18%) decides untrusted at once
      // and earns trust; the full presented alpha never does.
      alpha_auto_policy session;
      sequence s(session, recorded_frames({recorded::re9_room, recorded::re9_room_trusted}));
      present p;
      p.offered = 2 | 8 | 16;
      p.color_alpha_flags = layer;
      run(s, p, 10000, 14000);
      for (const auto &f : s.frames) {
        require(f.source == 2 && f.covered == 14611, "The RE9 layer did not decide");
      }
      require(session.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source, "The RE9 layer did not earn trust, or the full presented alpha did");
      known_today("S2a", "A1/S1", "the untrusted RE9 offscreen layer decides from the first frame (matches the replay xfail)");
      check_counters(s, "RE9 dark room");
    }
  }

  // ---------------------------------------------------------------- adversaries

  struct lcg {
    std::uint32_t state;

    std::uint32_t next(std::uint32_t range) {
      state = state * 1664525u + 1013904223u;
      return (state >> 8) % range;
    }
  };

  void adversaries() {
    {
      // RE9-like garbage presented alpha. Fluctuating, it never earns trust.
      alpha_auto_policy session;
      lcg random {12345};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha_covered[3] = 20 + random.next(961);
        f.source = 4;
        f.covered = f.alpha_covered[3];
        return f.words(in);
      });
      present p;
      p.offered = 8;
      run(s, p, 10000, 40000);
      require(!session.trusted_alpha() && !session.counters()[ui_counter::trust_earned], "Fluctuating presented alpha earned trust");
    }
    {
      // Stable and selective, garbage presented alpha earns trust. A1 earns
      // it the same way when nothing declared is offered: only the earning
      // void (a declared alpha offered but invalid, below), a judge of
      // stronger provenance (A2) or Forget stops it.
      alpha_auto_policy session;
      lcg random {777};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha_covered[3] = 140 + random.next(21);
        f.source = 4;
        f.covered = f.alpha_covered[3];
        return f.words(in);
      });
      present p;
      p.offered = 8;
      run(s, p, 10000, 14000);
      require(session.trusted_alpha() == 8u, "Stable selective presented alpha did not earn trust");
      require(s.trust[0].tick == first_sample_from(s, s.samples.front().sample_tick_ms + alpha_trust_span_ms), "Stable selective presented alpha did not earn 2 s into its run");
    }
    {
      // RE9 FG off on rejected-tag frames: the tagged UIColorAndAlpha is
      // offered but invalid (more than 1% invalid pixels) in every sample,
      // while stable, selective presented alpha beside it earns trust.
      alpha_auto_policy session;
      lcg random {909};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha_invalid[1] = 50;
        f.alpha_covered[3] = 140 + random.next(21);
        f.source = 4;
        f.covered = f.alpha_covered[3];
        return f.words(in);
      });
      present p;
      p.offered = 2 | 8;
      run(s, p, 10000, 14000);
      require(session.trusted_alpha() == 8u, "Presented alpha beside an invalid tag did not earn trust");
      known_today("S2a", "A1",
                  "presented alpha earns trust while a declared tag is offered but invalid in every sample; A1 voids "
                  "those samples, so it would never be accepted");
    }
    {
      // A premultiplied bloom-like target cleared to transparent, offered as
      // the offscreen layer: selective and stable, it earns and is never
      // revoked. The reviewed rules keep this: with no declared or exact
      // judge, only Forget or a failed re-earn after a restore (A3) removes a
      // wrongly accepted inferred source (an open question of the framework).
      alpha_auto_policy session;
      lcg random {4242};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha_covered[1] = 215 + random.next(71);
        f.source = 2;
        f.covered = f.alpha_covered[1];
        return f.words(in);
      });
      present p;
      p.offered = 2 | 8;
      p.color_alpha_flags = ui_detection::layer_detection_flags(true);
      run(s, p, 10000, 40000);
      const auto c = session.counters();
      require(session.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source && s.trust.size() == 1 && !c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented], "The bloom-like layer did not earn trust once and keep it");
      check_counters(s, "bloom");
    }
    // A dark grainy scene under a full untrusted layer claim (the W3 settings
    // frames): one valid hidden sample, then valid but ambiguous ones.
    for (const auto verdict : {scene_verdict::hidden, scene_verdict::visible}) {
      const auto open = parse_texels(recorded::w3_settings), held = parse_texels(recorded::w3_settings_held);
      alpha_auto_policy session;
      sequence s(session, [&](const gpu_inputs &in) {
        return (in.per_frame & ui_detection::per_frame_scene_hold) ? held : open;
      });
      std::uint32_t evidence_samples = 0;
      std::uint64_t first_tick = 0;
      s.sample_edit = [&](texels &t, std::uint64_t tick, bool evidence) {
        if (!evidence) {
          return;
        }
        const bool first = !evidence_samples++;
        if (first) {
          first_tick = tick;
        }
        t[word::scene_state] = scene_state(true, first ? verdict : scene_verdict::ambiguous);
        t[word::scene_d] = float_bits(first ? (verdict == scene_verdict::hidden ? -.02f : .4f) : .2f);
      };
      present p;
      p.offered = 2 | 8;
      p.color_alpha_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 13000);
      std::size_t flat = 0;
      for (const auto &f : s.frames) {
        if (f.source != 8) {
          continue;
        }
        ++flat;
        require(f.now_ms <= first_tick + ui_detection::scene::hold_ms, "Ambiguous evidence renewed the hold");
      }
      if (verdict == scene_verdict::visible) {
        require(!flat, "A visible dark scene flattened");
      } else {
        require(flat > 0, "One hidden sample did not flatten");
        known_today("S2b", "H1", "a dark grainy scene under a full untrusted layer claim flattens " + std::to_string(flat) + " frames on one valid hidden sample; H1 wants 2");
      }
    }
  }

  // ---------------------------------------------------------------- --log

  // Informational: replays logged status samples through alpha_auto_policy.
  // Logged samples are gated and incomplete (about one per second, not every
  // 100 ms one), so the prediction is a sketch, never a gate.
  std::string field_text(const std::string &line, const std::string &key) {
    const auto at = line.find(" " + key + "=");
    if (at == std::string::npos) {
      return {};
    }
    const auto begin = at + key.size() + 2;
    return line.substr(begin, line.find(' ', begin) - begin);
  }

  std::uint32_t field_number(const std::string &text, int base = 10) {
    std::uint32_t value = 0;
    const char *begin = text.data();
    if (base == 16 && text.rfind("0x", 0) == 0) {
      begin += 2;
    }
    std::from_chars(begin, text.data() + text.size(), value, base);
    return value;
  }

  std::array<std::uint32_t, 4> field_quad(const std::string &text) {
    std::array<std::uint32_t, 4> result {};
    std::size_t at = 0;
    for (auto &value : result) {
      const auto end = text.find('/', at);
      value = field_number(text.substr(at, end - at));
      if (end == std::string::npos) {
        break;
      }
      at = end + 1;
    }
    return result;
  }

  int replay_logs(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
      std::ifstream file(path);
      if (!file) {
        std::fprintf(stderr, "Cannot read %s\n", path.c_str());
        return 2;
      }
      alpha_auto_policy session;
      std::uint32_t predicted = 0, fed = 0, logged = 0;
      std::uint64_t day = 0, last_ms = 0;
      std::map<std::string, std::string> last_sample;  // Per runtime.
      std::printf("== %s\n", path.c_str());
      std::string line;
      while (std::getline(file, line)) {
        if (line.size() < 12 || line.find("Sunshine UI protection") == std::string::npos) {
          continue;
        }
        const auto stamp = line.substr(0, 12);
        std::uint64_t ms = (field_number(stamp.substr(0, 2)) * 3600000ull) + field_number(stamp.substr(3, 2)) * 60000ull +
                           field_number(stamp.substr(6, 2)) * 1000ull + field_number(stamp.substr(9, 3));
        if (ms + day < last_ms) {
          day += 86400000ull;
        }
        ms += day;
        last_ms = ms;
        if (const auto at = line.find("restored alpha trust "); at != std::string::npos) {
          const auto bits = field_number(line.substr(at + 21, line.find(' ', at + 21) - at - 21), 16);
          session.restore_trusted_alpha(bits);
          predicted = session.trusted_alpha();
          std::printf("%s restored  0x%x\n", stamp.c_str(), bits);
          continue;
        }
        if (const auto at = line.find("alpha trust is now "); at != std::string::npos) {
          const auto bits = field_number(line.substr(at + 19, line.find(';', at) - at - 19), 16);
          std::printf("%s logged    0x%x\n", stamp.c_str(), bits);
          ++logged;
          continue;
        }
        if (line.find("Sunshine UI protection: runtime=") == std::string::npos) {
          continue;
        }
        const auto pixels = field_number(field_text(line, "sampled_pixels"));
        if (!pixels) {
          continue;
        }
        const auto runtime = field_text(line, "runtime");
        const auto sample_at = line.find(" sampled_source=");
        const auto sample_end = line.find(" status_revision=");
        const auto sample = line.substr(sample_at, sample_end - sample_at);
        if (last_sample[runtime] == sample) {
          continue;  // The same sample logged again.
        }
        last_sample[runtime] = sample;
        alpha_auto_decision::detection_evidence evidence;
        evidence.candidates = field_number(field_text(line, "sampled_candidates"), 16);
        evidence.alpha_covered = field_quad(field_text(line, "sampled_alpha_covered"));
        evidence.alpha_invalid = field_quad(field_text(line, "sampled_alpha_invalid"));
        evidence.ui_layer = field_text(line, "sampled_ui_layer") == "1";
        const auto hudless_at = line.find("sampled_hudless={");
        if (hudless_at != std::string::npos) {
          const auto group = line.substr(hudless_at, line.find('}', hudless_at) - hudless_at);
          evidence.hudless_changed = field_number(field_text(group, "changed"));
          evidence.hudless_unchanged = field_number(field_text(group, "unchanged"));
          evidence.hudless_invalid = field_number(field_text(group, "invalid"));
        }
        session.observe_alpha_channels(evidence, pixels, ms);
        ++fed;
        if (session.trusted_alpha() != predicted) {
          predicted = session.trusted_alpha();
          std::printf("%s predicted 0x%x\n", stamp.c_str(), predicted);
        }
      }
      std::printf("%u logged samples fed, %u logged trust changes, predicted trust at the end 0x%x\n", fed, logged, predicted);
    }
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    if (std::string_view(argv[1]) != "--log" || argc < 3) {
      std::fprintf(stderr, "Usage: %s [--log ReShade.log...]\n", argv[0]);
      return 2;
    }
    return replay_logs(std::vector<std::string>(argv + 2, argv + argc));
  }
  const std::pair<const char *, void (*)()> groups[] {
    {"A1 earn", trust_is_earned_by_steady_selective_samples},
    {"A1 restart", trust_restarts_on_a_full_sample_or_a_factor_two_break},
    {"A2/P1 revoke", trust_is_revoked_by_contradiction},
    {"A3 provisional restore", restored_trust_is_provisional},
    {"E1/A1 opaque tag proof and per-source trust", opaque_tag_proof_prefers_the_layer_and_trust_is_per_source},
    {"A1 context", trust_is_not_context_keyed},
    {"T1 frame generation holds", generated_presents_hold_per_multiplier},
    {"T1/E2 frame generation multiplier drift", generated_presents_under_multiplier_drift},
    {"T1 hold cap", holds_are_capped},
    {"T1 inexact-after-exact and trusted-missing holds", inexact_after_exact_and_trusted_missing_holds},
    {"A1/S1/T1/P1 E33 title and Load Game", untrusted_inferred_masks_and_e33_streams},
    {"sample staleness", samples_go_stale_and_are_discarded},
    {"H1 W3 layer route", hidden_scene_layer_route},
    {"H1 shadow and scope clears", hidden_scene_shadow_and_clears},
    {"P1/A1 recorded menus", recorded_full_menus_after_earned_trust},
    {"adversaries", adversaries},
  };
  try {
    for (const auto &[name, group] : groups) {
      group();
      std::printf("PASS %s\n", name);
    }
    std::printf("PASS UI sequence replay: %zu groups, %u KNOWN_TODAY\n", std::size(groups), known_today_count);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL UI sequence replay: %s\n", error.what());
    return 1;
  }
}

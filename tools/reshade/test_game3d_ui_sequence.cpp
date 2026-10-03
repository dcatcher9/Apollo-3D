// SPDX-License-Identifier: GPL-3.0-only
// Sequence replay of live automatic UI detection's temporal rules
// (docs/reshade-sbs.md, UI protection and UI decision framework). Per-frame
// decision streams drive the production state machines without a GPU: the
// game session's acceptance ledger (alpha_auto_policy), the renderer's T1
// arbitration, status samples and CPU-owned hidden-scene verdicts
// (ui_temporal::detection_state), the sample decode and counter commit
// (ui_temporal::decode_detection_sample, sample_counters), Present pairing
// and counting under frame generation (ui_mask::pair_hudless_present,
// generated_without_input) and the exact
// counters (ui_counters). A stream's GPU input is what one detection counts:
// decision texels 0-9 that ui_detection_replay --verbose recorded on labelled
// dumps with the S2a shader, or synthetic counts. Every decision is
// ui_selection::decide, the C++ mirror of SunshineUIDetectionReduceCS (the T1
// hold store included) that test_game3d_ui_selection_contract proves equal to
// the shader's reduce; every recorded decision must equal it.
//
// It asserts the rules through stage S2a strictly. An outcome that a later
// stage of the UI decision framework (docs/reshade-sbs.md, UI decision
// framework: stages S0-S6; rules E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1,
// F1) changes prints "KNOWN_TODAY <stage> <rule>: <text>" and does not fail;
// that stage turns it into a strict assertion. Five remain after S2a: S2b H1
// (three) and S3 T1/E2 (two). Outcomes the rules already call correct, such
// as an accepted source pinning a whole-frame alpha flat over a visible scene
// (P1, the opacity ruling), are asserted strictly.
//
// Informational, not in ctest: --log <ReShade.log>... replays the logged
// "Sunshine UI protection" samples through alpha_auto_policy and prints the
// predicted acceptance transitions beside the logged ones.
#include "game3d_alpha_auto.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_mask.h"
#include "game3d_ui_selection.h"
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
  namespace candidate = ui_detection::candidate;
  using ui_detection::scene_verdict;
  using ui_selection::kind;
  using ui_temporal::hold_kind;

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

  // Invariants every stream's detections keep (S1): no unaccepted inferred
  // alpha decides, and no inferred alpha decides beside an accepted declared
  // one; and (T1) no generated Present runs detection.
  struct {
    std::uint64_t detections {}, untrusted_inferred {}, presented_over_dedicated {}, generated_detections {};
  } invariants;

  // ---------------------------------------------------------------- acceptance signatures (A1)

  constexpr std::uint32_t srgb = 1, pq = 3;

  // Each kind's typed DXGI format and the swapchain color space. The formats
  // stand for a stream's sources; a signature only has to be stable.
  candidate_signatures signatures_in(std::uint32_t color_space) {
    candidate_signatures s;
    s.color_space = color_space;
    s.set(kind::ui_alpha, 61).set(kind::ui_color, 87).set(kind::ui_layer, 28).set(kind::backbuffer, 24).set(kind::current, 24).set(kind::hudless, 24);
    return s;
  }

  // Stellar Blade in HDR: a float offscreen UI layer under PQ.
  candidate_signatures sb_hdr_signatures() {
    auto s = signatures_in(pq);
    s.set(kind::ui_layer, 10);
    return s;
  }

  std::string key(kind k, std::uint32_t color_space = srgb) {
    return signatures_in(color_space).of(k).key();
  }

  // alpha_auto_policy::stored() of exactly these signatures.
  std::string stored_of(std::vector<ui_selection::signature> accepted) {
    std::sort(accepted.begin(), accepted.end());
    std::string text;
    for (const auto &signature : accepted) {
      text += (text.empty() ? "" : ",") + signature.key();
    }
    return text;
  }

  // Acceptance an earlier session of the game remembered (provisional).
  void restore(alpha_auto_policy &session, const std::string &stored, std::size_t expected) {
    const auto result = session.restore(stored);
    require(result.restored == expected && !result.discarded, "Restoring " + stored + " did not restore " + std::to_string(expected) + " entries");
  }

  // ---------------------------------------------------------------- decision texels

  // Decision texels 0-9 as the renderer reads them back.
  constexpr std::size_t texel_words = 4 * ui_detection::judgment_decision_texels;
  using texels = std::array<std::uint32_t, texel_words>;
  // Texels 5 and 6: the hidden-scene evidence, written only by the evidence pass.
  constexpr std::size_t scene_texels_begin = 4 * 5, scene_texels_end = 4 * ui_detection::scene_decision_texels;

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
      throw std::runtime_error("Recorded texels are not 40 words");
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

  // What one frame's detection reads: the frame's own inputs, the per-frame
  // bits and the hold store (game3d_renderer.cpp, detect_ui's b2 constants:
  // the candidates, the layer's stored flags with the per-frame bits, the
  // accepted candidates; the T1 hold store at u5, as the previous detection
  // wrote it).
  struct gpu_inputs {
    std::uint32_t bits {}, flags {}, accepted {}, per_frame {};
    std::uint64_t now_ms {};
    ui_selection::hold_state hold {};
  };

  using gpu_model = std::function<texels(const gpu_inputs &)>;

  // The reduce's decision of these counts and inputs (ui_selection::decide).
  ui_selection::decision decide(const texels &t, const gpu_inputs &in) {
    return ui_selection::decide(ui_selection::counts_from_words(t.data(), t.size()), in.bits, in.accepted, in.flags | in.per_frame, in.hold);
  }

  // The counts of `t` with the words the reduce writes for these inputs:
  // texel 0 the applied decision, the refused candidate and the frame reason
  // (F1, with the T1 reused bit).
  texels decision_words(texels t, const gpu_inputs &in) {
    const auto d = decide(t, in);
    t[word::source] = d.source;
    t[word::covered] = d.covered;
    t[word::candidates] = in.bits;
    t[word::accepted] = in.accepted;
    t[word::valid_bits] = d.valid_bits;
    t[word::refused] = d.refused;
    t[word::frame_reason] = ui_selection::frame_reason_word(d);
    return t;
  }

  // One recorded detection and the per-frame hold bits it was replayed with.
  struct recording {
    std::string_view words;
    std::uint32_t per_frame {};
  };

  // Recorded decision texels: ui_detection_replay --verbose with the S2a
  // shader on E:/ApolloDev/sbs_dump (selection revision 2, nothing bound at
  // the hold store). Words 4 and 17 hold the candidate bits and the accepted
  // candidates the frame was decided with; words 32-39 the one-way judgment
  // counts, the refused candidate and the frame reason. Each names its dump
  // and the replay case of the same inputs in ui_detection_cases.json
  // (scratchpad s2a/p1/replay_s2a_verbose.txt); the acceptance combinations
  // those cases do not hold were replayed alone (scratchpad
  // s1/p4/cases_p4.json, replayed by the S2a binary into
  // s2a/p1/replay_p4_cases_verbose.txt).
  namespace recorded {
    constexpr std::uint32_t scene_hold = ui_detection::per_frame_scene_hold;
    constexpr std::uint32_t scene_hold_hudless = ui_detection::per_frame_scene_hold_hudless;
    // The Witcher 3 Remastered, FG off, graphics settings over a hidden scene
    // (game3d_46312_198402901901355): a full opaque offscreen layer (0x40)
    // beside current alpha, unaccepted, without a CPU hold and held ("W3
    // graphics settings, layer hidden-scene hold").
    constexpr recording w3_settings {"0,0,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,0,0,0,0,3686400,0,3686400,72,3686400,0,3686400,64,0,0,0,5"};
    constexpr recording w3_settings_held {"8,3686400,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,0,0,0,0,3686400,0,3686400,72,3686400,0,3686400,0,0,0,0,255", scene_hold};
    // W3 FG off, notice board (game3d_46312_198402901901363): layer 12.48%,
    // unaccepted and accepted ("W3 notice board FG off, trusted layer, hold").
    constexpr recording w3_notice {"0,0,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,0,0,0,2029,1049969278,15,8116,0,0,0,0,460125,0,100337,72,281087,0,3686400,64,0,0,0,8"};
    constexpr recording w3_notice_accepted {"10,460125,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,64,0,0,2029,1049969278,15,8116,0,0,0,0,460125,0,100337,72,281087,0,3686400,0,0,0,0,255"};
    // W3 FG off, sign wheel over a visible scene (game3d_46312_198402901901353):
    // a full layer, unaccepted and accepted ("W3 sign wheel FG off, trusted
    // layer, hold").
    constexpr recording w3_wheel {"0,0,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,0,0,0,564,1057143089,15,2256,0,0,0,0,3686400,0,145726,72,3686400,0,3686400,64,0,0,0,3"};
    constexpr recording w3_wheel_accepted {"10,3686400,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,64,0,0,564,1057143089,15,2256,0,0,0,0,3686400,0,145726,72,3686400,0,3686400,0,0,0,0,255"};
    // W3 FG on, HUD (game3d_46312_198402901901357): UIAlpha 5.23% with an
    // inexact HUD-less pair, unaccepted ("W3 HUD FG on, not yet accepted
    // UIAlpha, hold") and accepted ("W3 HUD FG on, trusted UIAlpha, hold").
    constexpr recording w3_hud_fg {"0,0,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,0,103614,0,472,1056573614,15,1888,472,1056964608,3,0,0,0,0,9,0,0,3686400,1,0,0,0,8"};
    constexpr recording w3_hud_fg_accepted {"1,192930,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,1,103614,0,472,1056573614,15,1888,472,1056964608,3,0,0,0,0,9,0,0,3686400,0,0,0,0,255"};
    // W3 FG on, sign wheel (game3d_46312_198402901901359): a full UIAlpha,
    // unaccepted and accepted ("W3 sign wheel FG on, ... UIAlpha, hold").
    constexpr recording w3_wheel_fg {"0,0,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,0,149627,0,462,1056020436,15,1848,462,1057709052,3,0,0,0,0,9,0,0,3686400,16,0,0,0,5"};
    constexpr recording w3_wheel_fg_accepted {"1,3686400,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,1,149627,0,462,1056020436,15,1848,462,1057709052,3,0,0,0,0,9,0,0,3686400,0,0,0,0,255"};
    // Clair Obscur: Expedition 33, FG on, title over a visible scene
    // (game3d_31636_135749029986373): Backbuffer alpha 2.38%, unaccepted and
    // accepted ("E33 title FG on, untrusted/trusted backbuffer, hold").
    constexpr recording e33_title {"0,0,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,0,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4,0,178867,178867,4,0,0,0,8"};
    constexpr recording e33_title_accepted {"3,197797,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,4,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4,0,178867,178867,0,0,0,0,255"};
    // E33 FG on, Load Game over a hidden scene (game3d_59540_257918763574026):
    // a full Backbuffer alpha ("E33 load game FG on, untrusted / trusted
    // backbuffer").
    constexpr recording e33_load {"0,0,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,0,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4,0,8294400,8294400,4,0,0,0,3"};
    constexpr recording e33_load_accepted {"3,8294400,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,4,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4,0,8294400,8294400,0,0,0,0,255"};
    // Resident Evil Requiem, dark room (game3d_52696_225539427440975): the
    // offscreen layer 0.18% beside a full presented alpha and an inexact pair,
    // unaccepted ("RE9 dark room, untrusted layer, inexact pair, hold") and
    // accepted.
    constexpr recording re9_room {"0,0,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,386,1058855305,15,1544,386,1057529644,3,0,14611,0,3540,72,8475,0,8294400,64,0,0,0,8"};
    constexpr recording re9_room_accepted {"10,14611,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,64,0,0,386,1058855305,15,1544,386,1057529644,3,0,14611,0,3540,72,8475,0,8294400,0,0,0,0,255"};
    // Stellar Blade in SDR, FG on (game3d_69460_296226962143478): the tagged
    // UIColorAndAlpha (0.26%), the scene image in the cleared UI layer (V1
    // invalid) and the Backbuffer alpha; nothing accepted, the tag, the tag
    // and the Backbuffer ("SB SDR FG on, accepted tag beside the scene layer
    // and the Backbuffer"), and the Backbuffer of a Present without the tag.
    constexpr recording sb_sdr {"0,0,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,0,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6,0,15296,8294400,64,0,0,0,0"};
    constexpr recording sb_sdr_tag {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,2,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6,0,15296,8294400,0,0,0,0,255"};
    constexpr recording sb_sdr_tag_backbuffer {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,6,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6,0,15296,8294400,0,0,0,0,255"};
    constexpr recording sb_sdr_backbuffer {"3,21283,8294400,0,68,8294319,41,0,0,0,21283,8294400,0,0,0,0,0,4,0,0,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,4,0,15296,8294400,0,0,0,0,255"};
    // Stellar Blade in HDR, FG on (game3d_69460_296226962143474): the opaque
    // final-image tag, the offscreen layer 0.18%, a full Backbuffer alpha and
    // an exact pair; nothing accepted, the layer accepted ("SB HDR FG on HUD,
    // unaccepted opaque tag beside an accepted layer, exact pair"), and the
    // tag without the layer ("SB HDR FG on HUD, opaque tag, backbuffer and
    // exact pair"). Manual On accepts every offered candidate, with and without
    // a source filter that leaves the tag out.
    constexpr recording sb_hdr {"0,0,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70,6267,8294400,8294400,64,0,5561801,5561801,8"};
    constexpr recording sb_hdr_layer {"10,14519,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,64,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70,6267,8294400,8294400,0,0,5561801,5561801,255"};
    constexpr recording sb_hdr_tag_only {"0,0,8294400,3,54,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,0,0,0,6,0,8294400,8294400,16,0,5561801,5561801,4"};
    constexpr recording sb_hdr_manual {"2,8294400,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,86,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70,6267,8294400,8294400,0,0,5561801,5561801,255"};
    constexpr recording sb_hdr_manual_filtered {"10,14519,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,84,0,0,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,68,6267,8294400,8294400,0,0,5561801,5561801,255"};
    constexpr recording sb_hdr_filtered {"0,0,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,0,0,0,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,68,6267,8294400,8294400,64,0,5561801,5561801,8"};
    // Hogwarts Legacy, FG off, title screen (game3d_50196_216992971069363): an
    // exact HUD-less pair differing almost everywhere over a hidden scene;
    // unaccepted ("HL title screen FG off, HUD-less not yet accepted"), under
    // the HUD-less route's hold, and accepted ("HL title screen FG off").
    constexpr recording hl_title {"0,0,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24,0,0,8294400,16,0,0,1,5"};
    constexpr recording hl_title_held {"9,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24,0,0,8294400,0,0,0,1,255", scene_hold_hudless};
    constexpr recording hl_title_accepted {"6,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,16,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24,0,0,8294400,0,0,0,1,255"};
    // Hogwarts Legacy, FG off, gameplay HUD (game3d_40404_173061335442886):
    // an exact pair whose change set is the HUD (3.98%); unaccepted ("HL
    // gameplay HUD FG off (b), HUD-less not yet accepted") and accepted ("HL
    // gameplay HUD FG off (b)").
    constexpr recording hl_hud {"0,0,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,0,0,0,397,1061190608,15,1588,397,1061190608,3,0,0,0,0,24,0,0,8294400,16,0,0,7961885,8"};
    constexpr recording hl_hud_accepted {"5,329969,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,16,0,0,397,1061190608,15,1588,397,1061190608,3,0,0,0,0,24,0,0,8294400,0,0,0,7961885,255"};
  }  // namespace recorded

  // The recorded counts with exactly these candidate bits and accepted
  // candidates decide by ui_selection::decide under this frame's hold bits; a
  // stream that asks for any other combination fails rather than guess the
  // dump's counts. Every recording must equal decide() under its own hold bits.
  gpu_model recorded_frames(std::vector<recording> frames) {
    constexpr std::uint32_t holds = ui_detection::per_frame_scene_hold | ui_detection::per_frame_scene_hold_hudless;
    std::vector<std::pair<texels, std::uint32_t>> parsed;
    for (const auto &frame : frames) {
      const auto t = parse_texels(frame.words);
      const gpu_inputs own {t[word::candidates], 0, t[word::accepted], frame.per_frame};
      const auto d = decide(t, own);
      require(d.source == t[word::source] && d.covered == t[word::covered] && d.valid_bits == t[word::valid_bits] && d.refused == t[word::refused] && ui_selection::frame_reason_word(d) == t[word::frame_reason], "The recording for candidates " + hex(own.bits) + " accepted " + hex(own.accepted) + " differs from ui_selection::decide");
      parsed.emplace_back(t, frame.per_frame & holds);
    }
    return [parsed](const gpu_inputs &in) {
      const std::pair<texels, std::uint32_t> *match = nullptr;
      for (const auto &frame : parsed) {
        if (frame.first[word::candidates] != in.bits || frame.first[word::accepted] != in.accepted) {
          continue;
        }
        if (!match || (frame.second == (in.per_frame & holds) && match->second != (in.per_frame & holds))) {
          match = &frame;
        }
      }
      if (!match) {
        throw std::runtime_error("No recorded decision for candidates " + hex(in.bits) + " accepted " + hex(in.accepted));
      }
      return decision_words(match->first, in);
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

  // Synthetic counts over 1000 pixels. The fields are inputs; they stand for
  // what a stream's GPU counted, and decide() decides from them.
  struct synthetic {
    ui_selection::counts c;
    bool scene_valid {};
    scene_verdict verdict = scene_verdict::none;
    float d {};
    std::uint32_t n {}, decided_edges {};

    synthetic() {
      c.pixels = 1000;
    }

    synthetic &alpha(kind k, std::uint32_t covered, std::uint32_t invalid = 0) {
      c.covered[ui_selection::alpha_index(k)] = covered;
      c.invalid[ui_selection::alpha_index(k)] = invalid;
      return *this;
    }

    // A HUD-less pair's change set (V2): partial and valid under a quarter of
    // the frame, full (source 6) from 98% with lit() from an exact pair,
    // invalid in between (the middle band).
    synthetic &change_set(std::uint32_t changed) {
      c.changed = changed;
      c.unchanged = c.pixels - changed;
      c.matching_tiles = 200;
      return *this;
    }

    synthetic &lit(std::uint32_t pixels) {
      c.lit = pixels;
      return *this;
    }

    // A2, the one-way counts of a judged kind (the layer, Backbuffer or
    // current alpha): pixels with alpha of at least 1/2, and those of them
    // where an offered exact pair's HUD-less image is lit and unchanged.
    synthetic &strong(kind k, std::uint32_t pixels) {
      c.strong[judged_index(k)] = pixels;
      return *this;
    }

    synthetic &contradicted(kind k, std::uint32_t pixels) {
      c.contradicted[judged_index(k)] = pixels;
      return *this;
    }

    static std::size_t judged_index(kind k) {
      const auto &judged = ui_selection::judged_kinds;
      const auto at = std::find(judged.begin(), judged.end(), k);
      require(at != judged.end(), "Only inferred alpha is judged");
      return std::size_t(at - judged.begin());
    }

    texels words(const gpu_inputs &in) const {
      texels t {};
      t[word::pixels] = c.pixels;
      t[word::matching_tiles] = c.matching_tiles;
      t[word::hudless_changed] = c.changed;
      t[word::hudless_unchanged] = c.unchanged;
      t[word::hudless_invalid] = c.nonfinite;
      t[word::hudless_lit] = c.lit;
      const kind presented[] {kind::ui_alpha, kind::ui_color, kind::backbuffer, kind::current};
      for (std::size_t i = 0; i != 4; ++i) {
        t[word::alpha_covered + i] = c.covered[ui_selection::alpha_index(presented[i])];
        t[word::alpha_invalid + i] = c.invalid[ui_selection::alpha_index(presented[i])];
      }
      t[word::layer_covered] = c.covered[ui_selection::alpha_index(kind::ui_layer)];
      t[word::layer_invalid] = c.invalid[ui_selection::alpha_index(kind::ui_layer)];
      t[word::alpha_opaque] = c.opaque_ui_alpha;
      t[word::alpha_opaque + 1] = c.opaque_ui_color;
      t[word::layer_opaque] = c.opaque_layer;
      t[word::scene_n] = n;
      t[word::scene_d] = float_bits(d);
      t[word::scene_state] = scene_state(scene_valid, verdict);
      t[word::scene_decided] = decided_edges;
      for (std::size_t i = 0; i != ui_selection::judged_kinds.size(); ++i) {
        t[word::strong + i] = c.strong[i];
        t[word::contradicted + i] = c.contradicted[i];
      }
      // The tiles pass counts neither strong nor contradicted pixels of the
      // one-frame-late layer copy (stored flag 0x4): under E2 it is not
      // same-sample evidence (test_game3d_ui_selection_contract, check_tiles).
      if (in.flags & ui_detection::stored_late_layer) {
        t[word::strong + judged_index(kind::ui_layer)] = 0;
        t[word::contradicted + judged_index(kind::ui_layer)] = 0;
      }
      return decision_words(t, in);
    }
  };

  // One Present as render() receives it.
  struct present {
    std::uint64_t now_ms {};
    // Candidate bits after the renderer's compatibility checks
    // (ui_detection::candidate): 0x1 UIAlpha, 0x2 the UI color tag, 0x4
    // Backbuffer, 0x8 current, 0x10 a paired HUD-less image, 0x20 an exact
    // pair, 0x40 the offscreen UI layer.
    std::uint32_t offered {};
    // The offscreen layer's stored flags (layer_detection_flags).
    std::uint32_t layer_flags {};
    // The acceptance signatures of the offered kinds (A1).
    candidate_signatures signatures = signatures_in(srgb);
    // T1 identity by Present counting (ui_temporal::present_identity): a
    // generated Present (ui_mask::pair_hudless_present, set by the input
    // provider only while a HUD-less tag pairs), and the real frame it
    // shows, the HUD-less tag's present generation (0 without a HUD-less
    // capture).
    bool hold_previous {};
    std::uint64_t real_frame {};
    // The provider offered an Auto input (source_alpha_ui); without one it
    // offers no candidate.
    bool available = true;
    bool depth_current = true;
    std::uint64_t epoch = 1, revision = 1;
    std::uint32_t viewport = 1;
  };

  struct frame_result {
    std::uint64_t now_ms {};
    // active: the frame applied a mask from detection (detected) or held
    // one (held). held: a generated Present showing the decision of the real
    // frame it shows (held.generated); unavailable: a generated Present with
    // no such decision (held.none, no mask). grace: a zero-offer real frame
    // that ran the reduce and mask passes only (no tiles pass, no sample).
    bool active {}, held {}, unavailable {}, detected {}, grace {}, polled {}, discarded {}, submitted {};
    ui_temporal::hold_decision hold;
    gpu_inputs gpu;
    // The detection's decision: the applied source (source, covered), the
    // frame's own (own_source; none_reason when it decided nothing) and
    // whether the T1 grace reused the previous real frame's (reused).
    ui_selection::decision decision;
    // The mask this frame applies: the detected or the held one, or none.
    std::uint32_t source {}, covered {}, pixels {};
    std::uint32_t scene_hold_bits {};
    // update_alpha_auto's status for this frame.
    alpha_auto_decision consumed;

    bool flat() const {
      return active && pixels && covered == pixels;
    }
  };

  // One renderer's temporal bookkeeping, calling the production code in the
  // order game3d_renderer.cpp does (game3d_ui_temporal.h, its header
  // comment):
  //   render():           bits and signatures, accepted (session),
  //                       arbitrate(identity, observation, bits), the
  //                       layer's flags and adopt when t.adopt; then
  //                       held() (t.hold, held.generated), unavailable()
  //                       (held.none), enter_scope + poll_detection +
  //                       scene_holds + detect_ui + detected() (t.detect with
  //                       bits, or a zero-offer real frame flagged
  //                       accepted_missing), or inactive();
  //   end of render:      the detection fence is signaled (detection_awaiting_signal);
  //   update_alpha_auto:  the latest sample only while status_fresh;
  //   poll_detection:     sample_discarded (scope, stale on arrival),
  //                       decode_detection_sample, latest_key,
  //                       observe_scene, session.observe with the signatures
  //                       the sample was submitted with, commit_counters
  //                       (sample_counters);
  //   detect_ui:          the reduce with the hold store (u5) and, except on
  //                       a zero-offer frame, the tiles pass and the 100 ms
  //                       sample cadence, scene evidence by measure_scene,
  //                       the counter and CPU snapshot and the status key.
  // The frame size, resource preparation and depth are not modelled.
  class sequence {
  public:
    sequence(alpha_auto_policy &session, gpu_model gpu, std::uint32_t lag = 1):
        session(session),
        gpu(std::move(gpu)),
        lag(lag),
        last_accepted(session.stored()) {}

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
      std::string accepted;
    };

    std::vector<transition> trust;  // stored() after a sample changed it.
    std::size_t discards {}, commits {}, committed_frames {};

    // The overlay's Forget (A3): the session clears the game's entries; the
    // transitions continue from the cleared set.
    std::string forget() {
      auto cleared = session.forget();
      last_accepted = session.stored();
      return cleared;
    }

    const frame_result &step(const present &p) {
      frame_result r;
      r.now_ms = p.now_ms;
      const auto index = frames.size();
      const std::uint32_t bits = p.available ? p.offered : 0u;
      const std::uint32_t accepted = session.accepted(bits, p.signatures);
      alpha_auto_source observation;
      observation.now_ms = observation.tick_ms = p.now_ms;
      observation.epoch = p.epoch;
      observation.revision = p.revision;
      observation.viewport = p.viewport;
      observation.session = &session;
      const ui_temporal::present_identity identity {p.hold_previous, p.real_frame};
      r.hold = temporal.arbitrate(identity, observation, bits);
      const std::uint32_t flags = (bits & candidate::layer) ? p.layer_flags : 0u;
      if (r.hold.adopt) {
        temporal.adopt(bits, flags, accepted);
      }
      ++cpu[ui_counter::auto_frames];
      const bool missing = (r.hold.per_frame & ui_detection::per_frame_accepted_missing) != 0;
      if (r.hold.hold) {
        // The detected mask as it is: the decision of the real frame shown.
        ++cpu[ui_counter::held_generated];
        temporal.held(identity);
        r.active = r.held = true;
      } else if (r.hold.kind == hold_kind::unavailable) {
        ++cpu[ui_counter::held_none];
        temporal.unavailable();
        r.unavailable = true;
      } else if (r.hold.detect && (bits || missing)) {
        r.active = r.detected = true;
        r.grace = !bits;
        temporal.enter_scope(observation);
        const bool shadow = session.first_run();
        poll(observation, r);
        r.scene_hold_bits = temporal.scene_holds(p.now_ms);
        const std::uint32_t per_frame = r.scene_hold_bits | r.hold.per_frame | (!p.depth_current ? ui_detection::per_frame_depth_not_current : 0u);
        detect(observation, bits, flags, accepted, p.signatures, per_frame, shadow, index, r);
        temporal.detected(observation, identity);
        invariants.generated_detections += p.hold_previous ? 1 : 0;
      } else {
        ++cpu[ui_counter::inactive_no_candidates];
        temporal.inactive();
      }
      if (r.active) {
        r.source = mask_source;
        r.covered = mask_covered;
        r.pixels = mask_pixels;
      }
      if (awaiting) {
        awaiting = false;
        ready_frame = index + lag;
      }
      if (temporal.status_fresh(observation)) {
        r.consumed = temporal.latest;
      } else {
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
      if (ui_temporal::sample_discarded(input, pending_source)) {
        pending = counters_pending = false;
        r.discarded = true;
        ++discards;
        return;
      }
      pending = false;
      auto &latest = temporal.latest;
      latest = ui_temporal::decode_detection_sample(pending_texels.data(), pending_texels.size(), pending_source.now_ms, submitted, true, pending_flags);
      temporal.latest_source = pending_source;
      temporal.latest_key = pending_key;
      temporal.observe_scene(latest, temporal.pending_scene_actionable);
      session.observe(latest.evidence, latest.pixels, pending_source.now_ms, pending_signatures);
      commit(latest);
      counters_pending = false;
      r.polled = true;
      samples.push_back(latest);
      auto now_accepted = session.stored();
      if (now_accepted != last_accepted) {
        trust.push_back({latest.sample_tick_ms, now_accepted});
      }
      last_accepted = std::move(now_accepted);
    }

    void detect(const alpha_auto_source &observation, std::uint32_t bits, std::uint32_t flags, std::uint32_t accepted, const candidate_signatures &signatures, std::uint32_t per_frame, bool shadow, std::size_t index, frame_result &r) {
      r.gpu = {bits, flags, accepted, per_frame, observation.now_ms, gpu_hold};
      // A zero-offer frame runs no tiles pass: the reduce reads the
      // statistics rows the previous tiles pass left (with nothing offered,
      // they decide nothing of their own).
      texels t = r.grace ? decision_words(statistics, r.gpu) : gpu(r.gpu);
      // The reduce decides, writes the hold store and counts
      // (SunshineUIDetectionReduceCS, mirrored by ui_selection::decide and
      // counter_adds); the mask pass writes no mask when it reused one.
      r.decision = decide(t, r.gpu);
      require(r.decision.source == t[word::source] && r.decision.covered == t[word::covered], "A stream's GPU model did not decide as ui_selection::decide");
      gpu_hold = r.decision.next;
      if (!r.grace) {
        statistics = t;
      }
      const auto adds = ui_selection::counter_adds(r.decision, r.gpu.flags | per_frame);
      for (std::size_t i = 0; i != adds.size(); ++i) {
        gpu_words[i] += adds[i];
      }
      ++invariants.detections;
      invariants.untrusted_inferred += r.decision.untrusted_inferred ? 1 : 0;
      invariants.presented_over_dedicated += r.decision.presented_over_dedicated ? 1 : 0;
      mask_source = r.decision.source;
      mask_covered = r.decision.covered;
      mask_pixels = t[word::pixels];
      const auto now = observation.now_ms;
      if (r.grace || pending || (last_submit && now >= last_submit && now - last_submit < sample_interval_ms)) {
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
        std::fill(t.begin() + scene_texels_begin, t.begin() + scene_texels_end, 0u);
      }
      if (sample_edit) {
        sample_edit(t, now, evidence);
      }
      pending_texels = t;
      pending_signatures = signatures;
      pending_flags = r.gpu.flags | per_frame;
      counters_pending = true;
      pending_words = gpu_words;
      pending_cpu = cpu;
      pending_frame = index;
      pending_source = observation;
      pending_key = temporal.status_key();
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
    std::uint64_t last_submit {}, submitted {};
    std::uint32_t pending_key {};
    alpha_auto_source pending_source;
    candidate_signatures pending_signatures;
    std::uint32_t pending_flags {};
    texels pending_texels {};
    // The GPU's state across detections: the statistics rows of the last
    // tiles pass, the T1 hold store (u5) and the detected mask.
    texels statistics {};
    ui_selection::hold_state gpu_hold {};
    std::uint32_t mask_source {}, mask_covered {}, mask_pixels {};
    std::string last_accepted;
  };

  bool alpha_source(std::uint32_t source) {
    return (source >= 1 && source <= 4) || source == ui_detection::source_layer;
  }

  // The session's committed totals equal this stream's frames through the
  // last committed sample, and reconcile (auto_frames == detection_frames +
  // held.generated + held.none + inactive); no unaccepted inferred alpha
  // decided and none decided beside an accepted declared alpha (S1).
  void check_counters(sequence &s, const std::string &what) {
    const auto totals = s.session.counters();
    require(totals.reconciled(), what + ": the committed counters do not reconcile");
    ui_counters expected;
    std::uint64_t full_alpha = 0;
    for (std::size_t i = 0; i != s.committed_frames; ++i) {
      const auto &f = s.frames[i];
      ++expected[ui_counter::auto_frames];
      if (f.held) {
        ++expected[ui_counter::held_generated];
      } else if (f.unavailable) {
        ++expected[ui_counter::held_none];
      } else if (!f.detected) {
        ++expected[ui_counter::inactive_no_candidates];
      } else {
        ++expected[ui_counter::detection_frames];
        ++expected[ui_counter::decided + f.source];
        if (!f.source) {
          ++expected[ui_counter::none + f.decision.none_reason];
        }
        expected[ui_counter::reused] += f.decision.reused ? 1 : 0;
        expected[ui_counter::contradicted] += f.decision.contradicted ? 1 : 0;
        full_alpha += alpha_source(f.source) && f.pixels && std::uint64_t(f.covered) * 100 >= std::uint64_t(f.pixels) * 99 ? 1 : 0;
      }
    }
    require(totals[ui_counter::full_alpha] == full_alpha, what + ": full_alpha is " + std::to_string(totals[ui_counter::full_alpha]) + ", the stream says " + std::to_string(full_alpha));
    expected[ui_counter::samples] = s.commits;
    for (const auto index : {ui_counter::auto_frames, ui_counter::detection_frames, ui_counter::held_generated, ui_counter::held_none, ui_counter::reused, ui_counter::inactive_no_candidates, ui_counter::contradicted, ui_counter::samples}) {
      require(totals[index] == expected[index], what + ": counter " + std::to_string(index) + " is " + std::to_string(totals[index]) + ", the stream says " + std::to_string(expected[index]));
    }
    for (std::uint32_t source = 0; source != ui_counter_word::decided_count; ++source) {
      require(totals.decided(source) == expected.decided(source), what + ": decided[" + std::to_string(source) + "] differs");
    }
    for (std::size_t reason = 0; reason != ui_no_mask::count; ++reason) {
      require(totals[ui_counter::none + reason] == expected[ui_counter::none + reason], what + ": none." + std::string(ui_no_mask::names[reason]) + " differs");
    }
    require(!totals[ui_counter::untrusted_inferred] && !totals[ui_counter::presented_over_dedicated], what + ": an unaccepted inferred alpha decided, or an inferred alpha beside an accepted declared one");
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
    // The tag the last Present paired against: its T1 real-frame id (the
    // HUD-less tag's present generation).
    std::uint64_t tag {};

    struct step {
      ui_mask::hudless_present kind;  // As the pairing classified it.
      bool real;  // As the game presented it.
    };

    step present(bool fg_active, std::uint32_t actual, std::uint32_t reported) {
      tag = tagged;
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
  // one offers `generated_bits` and is generated (hold_previous) when the
  // stream pairs a HUD-less image; an unpaired one offers `generated_bits`
  // without its HUD-less image (game3d_ui_input_provider.cpp admits only a
  // pair). With a HUD-less capture every Present carries the tag it paired
  // against as its real-frame id.
  present fg_present(ui_mask::hudless_present kind, present real, std::uint32_t generated_bits, std::uint64_t tag) {
    real.real_frame = (real.offered & candidate::hudless) ? tag : 0u;
    if (kind == ui_mask::hudless_present::real_frame || kind == ui_mask::hudless_present::earlier_real_frame) {
      return real;
    }
    real.hold_previous = kind == ui_mask::hudless_present::generated_frame && (real.offered & candidate::hudless) != 0;
    real.offered = generated_bits;
    return real;
  }

  // The tick of the first sample at or after `from`.
  std::uint64_t first_sample_from(const sequence &s, std::uint64_t from) {
    for (const auto &sample : s.samples) {
      if (sample.sample_tick_ms >= from) {
        return sample.sample_tick_ms;
      }
    }
    return 0;
  }

  // The tick of the third sample at or after `from`: A2 revokes an accepted
  // source contradicted in every sample by its third contradiction, within
  // 2 s at the 100 ms cadence.
  std::uint64_t third_sample_from(const sequence &s, std::uint64_t from) {
    const auto first = first_sample_from(s, from);
    return first_sample_from(s, first_sample_from(s, first + 1) + 1);
  }

  // The index of the frame that polled the sample taken at `tick`.
  std::size_t poll_of(const sequence &s, std::uint64_t tick) {
    std::size_t polls = 0;
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      if (s.frames[i].polled && s.samples[polls++].sample_tick_ms == tick) {
        return i;
      }
    }
    throw std::runtime_error("No frame polled the sample at " + std::to_string(tick));
  }

  // ---------------------------------------------------------------- A1-A3 acceptance

  synthetic backbuffer_alpha(std::uint32_t covered) {
    synthetic f;
    f.alpha(kind::backbuffer, covered);
    return f;
  }

  void inferred_source_is_accepted_by_steady_selective_samples() {
    alpha_auto_policy session;
    std::vector<std::string> heard;
    session.on_change([&](const std::string &accepted) {
      heard.push_back(accepted);
    });
    sequence s(session, [](const gpu_inputs &in) {
      return backbuffer_alpha(200).words(in);
    });
    present p;
    p.offered = candidate::backbuffer;
    run(s, p, 10000, 14000);
    require(s.samples.size() > 3, "Too few samples were read");
    for (std::size_t i = 1; i < s.samples.size(); ++i) {
      require(s.samples[i].sample_tick_ms - s.samples[i - 1].sample_tick_ms >= sample_interval_ms, "Samples came faster than the 100 ms cadence");
    }
    const auto first = s.samples.front().sample_tick_ms;
    const auto earned = first_sample_from(s, first + alpha_trust_span_ms);
    const auto backbuffer = key(kind::backbuffer);
    require(s.trust.size() == 1 && s.trust[0].tick == earned && s.trust[0].accepted == backbuffer, "The Backbuffer was not accepted by the first sample 2 s into a steady selective run");
    require(heard == std::vector<std::string> {backbuffer}, "The acceptance listener did not hear the earn once");
    require(session.counters()[ui_counter::trust_earned] == 1, "The earn was not counted once");
    require(s.frames.front().gpu.accepted == 0 && s.frames.back().gpu.accepted == candidate::backbuffer, "Detection did not read the earned acceptance");
    // S1: before acceptance an inferred source decides nothing, counted as
    // unaccepted; afterwards it decides.
    for (const auto &f : s.frames) {
      if (!f.detected) {
        continue;
      }
      if (f.gpu.accepted) {
        require(f.source == 3 && f.covered == 200, "The accepted Backbuffer did not decide");
      } else {
        require(!f.source && f.decision.none_reason == ui_no_mask::unaccepted, "An unaccepted Backbuffer decided, or was not counted as unaccepted");
      }
    }
    check_counters(s, "steady selective earn");
    require(session.counters()[ui_counter::none + ui_no_mask::unaccepted] > 0, "No frame counted none.unaccepted");
  }

  void acceptance_restarts_on_a_full_sample_or_a_factor_two_break() {
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
      p.offered = candidate::backbuffer;
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
      require(!s.trust.empty() && s.trust[0].tick == earned && s.trust[0].accepted == key(kind::backbuffer), std::string("The ") + v.name + " did not earn at the expected sample");
      require(v.restarts == (earned > first_sample_from(s, first + alpha_trust_span_ms)), std::string("The ") + v.name + " restart rule moved");
    }
  }

  void acceptance_is_revoked_by_contradiction() {
    const auto backbuffer = signatures_in(srgb).of(kind::backbuffer), hudless = signatures_in(srgb).of(kind::hudless);
    // A Backbuffer selective and agreeing (200 strong pixels, none
    // contradicted) until 13000, then claiming the whole frame: 1000 strong
    // pixels, `contradicted` of them where the HUD-less image is lit and
    // unchanged. The pair changes `changed` pixels: 50 is a valid partial
    // change set (V2), 300 the invalid middle band.
    const auto full_over_pair = [](std::uint32_t contradicted, std::uint32_t changed) {
      return [contradicted, changed](const gpu_inputs &in) {
        const bool full = in.now_ms >= 13000;
        auto f = backbuffer_alpha(full ? 1000 : 200);
        f.strong(kind::backbuffer, full ? 1000 : 200).contradicted(kind::backbuffer, full ? contradicted : 0).change_set(changed);
        return f.words(in);
      };
    };
    {
      // Over a valid exact pair, 80% of the claim's strong pixels lie where
      // the HUD-less image is lit and unchanged: the one-way test contradicts
      // it in every sample (A2), and the accepted Backbuffer is revoked by
      // the third contradiction within 2 s (revoked_exact). The exact pair is
      // a declared source, accepted by its first selective sample.
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      sequence s(session, full_over_pair(800, 50));
      present p;
      p.offered = candidate::backbuffer | candidate::hudless | candidate::exact;
      run(s, p, 10000, 17000);
      const auto first = s.samples.front().sample_tick_ms;
      const auto first_full = first_sample_from(s, 13000);
      const auto revoked = third_sample_from(s, first_full);
      const auto both = stored_of({backbuffer, hudless});
      require(s.trust.size() == 3 && s.trust[0].tick == first && s.trust[0].accepted == hudless.key() && s.trust[1].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[1].accepted == both, "The exact pair and the agreeing Backbuffer were not accepted by their own rules");
      require(s.trust[2].tick == revoked && s.trust[2].accepted == hudless.key(), "A full claim the exact pair contradicts was not revoked by its third contradiction");
      require(heard == std::vector<std::string> {hudless.key(), both, hudless.key()} && session.counters()[ui_counter::trust_revoked_exact] == 1 && !session.counters()[ui_counter::trust_revoked_declared], "The one-way revocation was not heard or counted once as revoked_exact");
      std::uint64_t flat_ms = 0;
      for (const auto &f : s.frames) {
        if (f.now_ms >= 13000 && (f.gpu.accepted & candidate::backbuffer)) {
          // An accepted source pins its whole-frame alpha flat (P1, the
          // opacity ruling) until the exact pair's contradictions revoke it.
          require(f.flat() && f.source == 3 && f.decision.contradicted, "The accepted full claim did not decide flat, or was not counted contradicted");
          flat_ms = f.now_ms - 13000;
        }
        if (f.now_ms >= 13000 && !(f.gpu.accepted & candidate::backbuffer)) {
          require(f.source == 5 && f.covered == 50, "The accepted exact pair did not decide after the revocation");
        }
      }
      // It decides up to the poll of the revoking sample, and not after.
      require(flat_ms + 13000 >= revoked && flat_ms + 13000 <= revoked + sample_interval_ms, "The accepted full claim did not decide until its revocation (" + std::to_string(flat_ms) + " ms)");
      const auto c = session.counters();
      require(c[ui_counter::contradicted] > 0 && c[ui_counter::contradicted] <= c[ui_counter::full_alpha], "The contradicted full claim was not counted as contradicted");
      check_counters(s, "one-way revocation");
    }

    struct variant {
      std::uint32_t bits, contradicted, changed;
      const char *what;
    };

    for (const auto &v : {variant {candidate::backbuffer | candidate::hudless | candidate::exact, 0, 50, "a dim over unlit or changed HUD-less pixels"}, variant {candidate::backbuffer | candidate::hudless | candidate::exact, 800, 300, "a middle-band pair"}, variant {candidate::backbuffer | candidate::hudless, 800, 50, "an inexact pair"}}) {
      alpha_auto_policy session;
      sequence s(session, full_over_pair(v.contradicted, v.changed));
      present p;
      p.offered = v.bits;
      run(s, p, 10000, 20000);
      // A2: only valid same-sample evidence of stronger provenance revokes.
      // A dim or tint over dark or changed pixels never meets the one-way
      // test (E33 pause, the W3 sign wheel); a middle-band pair is invalid
      // (V2) and an inexact one never judges (E2). Ambiguous or invalid
      // self-doubt is not a revocation.
      const auto c = session.counters();
      require(session.accepts(backbuffer) && !c[ui_counter::trust_revoked_exact] && !c[ui_counter::trust_revoked_declared] && !c[ui_counter::contradicted], std::string("A full claim over ") + v.what + " revoked acceptance");
      for (const auto &f : s.frames) {
        require(f.now_ms < 13000 || (f.flat() && f.source == 3), std::string("The accepted full claim over ") + v.what + " did not pin flat");
      }
      check_counters(s, v.what);
    }

    // Presented alpha differing from an accepted dedicated UIAlpha by at
    // least 10% of the frame: the declared alpha's coverage contradicts it
    // (A2), revoked by the third contradiction within 2 s
    // (revoked_declared), and not earned again while it disagrees. The
    // declared UIAlpha is accepted by its first sample (A1), the Backbuffer
    // by its run, and while the UIAlpha is accepted and offered it alone
    // decides (S1).
    {
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        const bool disagree = in.now_ms >= 13000 && in.now_ms < 20000;
        synthetic f;
        f.alpha(kind::ui_alpha, 100).alpha(kind::backbuffer, disagree ? 300u : 100u);
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_alpha | candidate::backbuffer;
      run(s, p, 10000, 25000);
      const auto ui_alpha = signatures_in(srgb).of(kind::ui_alpha);
      const auto both = stored_of({ui_alpha, backbuffer});
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() >= 2 && s.trust[0].tick == first && s.trust[0].accepted == ui_alpha.key(), "The declared UIAlpha was not accepted by its first sample");
      require(s.trust[1].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[1].accepted == both, "The Backbuffer was not accepted 2 s into its run");
      const auto first_disagreement = first_sample_from(s, 13000);
      const auto revoked = third_sample_from(s, first_disagreement);
      require(s.trust.size() >= 3 && s.trust[2].tick == revoked && s.trust[2].accepted == ui_alpha.key(), "Disagreeing presented alpha was not revoked by its third contradiction");
      const auto agreed = first_sample_from(s, 20000);
      require(s.trust.size() == 4 && s.trust[3].tick == first_sample_from(s, agreed + alpha_trust_span_ms) && s.trust[3].accepted == both, "Presented alpha earned during disagreement, or not after it");
      for (const auto &f : s.frames) {
        require(!f.detected || f.source == (f.gpu.accepted & candidate::ui_alpha ? 1u : 0u), "The Backbuffer decided beside the accepted UIAlpha");
      }
      const auto c = session.counters();
      require(c[ui_counter::trust_revoked_declared] == 1 && !c[ui_counter::trust_revoked_exact] && c[ui_counter::trust_earned] == 3, "Presented-alpha revocation or the re-earn was not counted");
      check_counters(s, "presented disagreement");
    }
    for (const bool late : {false, true}) {
      // The offscreen layer is inferred: an accepted UIAlpha's coverage
      // judges a same-frame layer (S4, no stored flag 0x4) like presented
      // alpha (revoked_declared, the same timing). Today's one-frame-late
      // copy is not same-sample evidence (E2): no judge reads it.
      const auto layer = signatures_in(srgb).of(kind::ui_layer), ui_alpha = signatures_in(srgb).of(kind::ui_alpha);
      alpha_auto_policy session;
      restore(session, layer.key(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_alpha, 100).alpha(kind::ui_layer, in.now_ms >= 13000 ? 300u : 100u);
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_alpha | candidate::layer;
      p.layer_flags = late ? ui_detection::layer_detection_flags(false) : ui_detection::stored_premultiplied;
      run(s, p, 10000, 17000);
      const auto first = s.samples.front().sample_tick_ms;
      if (late) {
        require(s.trust.size() == 1 && s.trust[0].tick == first && s.trust[0].accepted == stored_of({ui_alpha, layer}) && !session.counters()[ui_counter::trust_revoked_declared], "An accepted UIAlpha judged the one-frame-late layer");
        check_counters(s, "late layer beside UIAlpha");
        continue;
      }
      const auto revoked = third_sample_from(s, 13000);
      require(s.trust.size() == 2 && s.trust[0].tick == first && s.trust[0].accepted == stored_of({ui_alpha, layer}) && s.trust[1].tick == revoked && s.trust[1].accepted == ui_alpha.key(), "An accepted UIAlpha did not revoke the disagreeing layer by its third contradiction");
      require(session.counters()[ui_counter::trust_revoked_declared] == 1, "The layer's revocation was not counted as revoked_declared");
      check_counters(s, "layer judged by UIAlpha");
    }
    {
      // The one-frame-late layer copy is never one-way judged before S4
      // (E2): its tiles count neither strong nor contradicted pixels, so an
      // accepted full layer over a valid exact pair is not revoked by it.
      const auto layer = signatures_in(srgb).of(kind::ui_layer);
      alpha_auto_policy session;
      restore(session, layer.key(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        const bool full = in.now_ms >= 13000;
        synthetic f;
        f.alpha(kind::ui_layer, full ? 1000 : 200).strong(kind::ui_layer, full ? 1000 : 200).contradicted(kind::ui_layer, full ? 800 : 0).change_set(50);
        return f.words(in);
      });
      present p;
      p.offered = candidate::layer | candidate::hudless | candidate::exact;
      p.layer_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 17000);
      for (const auto &sample : s.samples) {
        require(!sample.evidence.contradicted[0] && !sample.evidence.strong[0] && sample.evidence.late_layer, "The late layer was counted strong or contradicted");
      }
      for (const auto &f : s.frames) {
        require(f.source == ui_detection::source_layer && (f.now_ms < 13000 || f.flat()), "The accepted late layer did not decide");
      }
      require(session.accepts(layer) && !session.counters()[ui_counter::trust_revoked_exact], "The late layer was one-way judged");
      check_counters(s, "late layer");
    }
  }

  void restored_acceptance_is_provisional() {
    const auto backbuffer = key(kind::backbuffer);
    for (const bool reconfirm : {false, true}) {
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      restore(session, backbuffer, 1);
      require(heard.empty() && session.counters()[ui_counter::trust_restored] == 1, "A restore called the listener");
      sequence s(session, [reconfirm](const gpu_inputs &in) {
        return backbuffer_alpha(reconfirm ? 200 : 1000).words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
      run(s, p, 10000, 75000, 50);
      require(s.frames.front().gpu.accepted == candidate::backbuffer && s.frames.front().source == 3, "Restored acceptance did not decide from the first frame");
      const auto c = session.counters();
      if (reconfirm) {
        require(s.trust.empty() && heard.empty() && session.stored() == backbuffer && c[ui_counter::trust_earned] == 1 && !c[ui_counter::trust_lapsed], "Restored acceptance that this session earned again lapsed, or was heard");
      } else {
        // Lapses at the first sample 60 s after the source was first offered.
        const auto first = s.samples.front().sample_tick_ms;
        const auto lapsed = first_sample_from(s, first + alpha_trust_reconfirm_ms);
        require(s.trust.size() == 1 && s.trust[0].tick == lapsed && s.trust[0].accepted.empty() && heard == std::vector<std::string> {""}, "Unconfirmed restored acceptance did not lapse 60 s after first being offered");
        require(c[ui_counter::trust_lapsed] == 1 && !c[ui_counter::trust_earned], "The lapse was not counted once");
      }
      check_counters(s, reconfirm ? "reconfirmed restore" : "lapsed restore");
    }
    // A legacy TrustedUISources bitmask is discarded and counted; keys of
    // this format beside it are restored (A1).
    alpha_auto_policy session;
    const auto result = session.restore("4,0x1f," + backbuffer);
    require(result.restored == 1 && result.discarded == 2 && session.counters()[ui_counter::trust_discarded] == 2 && session.stored() == backbuffer, "Legacy acceptance entries were not discarded beside a key");
  }

  void invalid_declared_source_never_lapses_and_forget_clears() {
    {
      // A3: a restored declared tag offered but invalid (more than 1%
      // invalid pixels, V1) re-arms its reconfirm clock, so it never lapses
      // during an invalid run however long; it confirms at its first valid
      // selective sample afterwards. Meanwhile it decides nothing: no
      // decision of its own, and the T1 grace reuses only a decision of its
      // own, which no frame of the run had.
      const auto tag = key(kind::ui_color);
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      restore(session, tag, 1);
      constexpr std::uint64_t valid_from = 10000 + alpha_trust_reconfirm_ms + 10000;
      sequence s(session, [](const gpu_inputs &in) {
        const bool invalid = in.now_ms < valid_from;
        synthetic f;
        f.alpha(kind::ui_color, invalid ? 0 : 15, invalid ? 50 : 0);
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_color;
      run(s, p, 10000, valid_from + 1000, 50);
      const auto c = session.counters();
      require(s.trust.empty() && heard.empty() && session.stored() == tag && !c[ui_counter::trust_lapsed] && c[ui_counter::trust_earned] == 1, "A restored tag offered invalid for over 60 s lapsed, or did not confirm afterwards");
      for (const auto &f : s.frames) {
        if (f.now_ms < valid_from) {
          require(f.detected && !f.source && !f.decision.reused && f.decision.none_reason == ui_no_mask::trusted_invalid && f.decision.refused == candidate::ui_color, "The invalid accepted tag decided, or was not named trusted_invalid");
        } else {
          require(f.source == 2 && f.covered == 15, "The valid accepted tag did not decide");
        }
      }
      check_counters(s, "invalid declared tag");
    }
    {
      // Forget (the overlay's 'Forget learned UI sources'): the session
      // clears every entry of the game, the listener hears "" (persisting an
      // empty TrustedUISources) and trust.forgotten counts the cleared
      // signature. The next detection pushes accepted 0 and nothing is reused
      // (nothing accepted is missing); the source is accepted again by its
      // own rule, a new run of selective samples over 2 s.
      const auto backbuffer = key(kind::backbuffer);
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
      run(s, p, 10000, 13000);
      require(session.stored() == backbuffer && heard == std::vector<std::string> {backbuffer}, "The Backbuffer was not accepted before Forget");
      require(s.forget() == backbuffer && session.stored().empty() && heard == std::vector<std::string> {backbuffer, ""} && session.counters()[ui_counter::trust_forgotten] == 1, "Forget did not clear the accepted Backbuffer, persist none, or count it");
      require(s.forget().empty() && heard.size() == 2 && session.counters()[ui_counter::trust_forgotten] == 1, "Forgetting nothing was heard or counted");
      const auto read_before = s.samples.size();
      const auto forgot_at = s.frames.size();
      run(s, p, 13000, 16000);
      const auto &after = s.frames[forgot_at];
      require(after.detected && !after.gpu.accepted && !after.source && !after.decision.reused && after.decision.none_reason == ui_no_mask::unaccepted, "The detection after Forget kept the acceptance");
      const auto run_start = s.samples.at(read_before).sample_tick_ms;
      require(s.trust.size() == 2 && s.trust[1].tick == first_sample_from(s, run_start + alpha_trust_span_ms) && s.trust[1].accepted == backbuffer, "The forgotten Backbuffer was not accepted again by a new run");
      check_counters(s, "Forget");
    }
  }

  void separate_slots_never_accept_the_opaque_tag() {
    // Stellar Blade in HDR, FG on: the tagged UIColorAndAlpha is the opaque
    // final image and the HUD is in the offscreen UI layer. Both are offered in
    // their own candidate bits from the first frame (E1). The tag is never
    // selective, so it is never accepted and never blocks (S1, A1); the layer
    // is accepted by its own run and decides. Nothing is ever flat.
    alpha_auto_policy session;
    sequence s(session, recorded_frames({recorded::sb_hdr, recorded::sb_hdr_layer, recorded::sb_hdr_tag_only}));
    present p;
    p.offered = candidate::ui_color | candidate::layer | candidate::backbuffer | candidate::hudless | candidate::exact;
    p.layer_flags = ui_detection::layer_detection_flags(true);
    p.signatures = sb_hdr_signatures();
    run(s, p, 10000, 14000);
    const auto layer = p.signatures.of(kind::ui_layer), tag = p.signatures.of(kind::ui_color);
    require(tag.key() == "ui_color:87:pq" && layer.key() == "ui_layer:10:pq", "The SB HDR signatures moved");
    const auto first = s.samples.front().sample_tick_ms;
    require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[0].accepted == layer.key(), "The layer was not accepted alone 2 s into its run");
    require(!session.accepts(tag), "The never-selective opaque tag was accepted");
    for (const auto &f : s.frames) {
      require(!f.flat(), "A frame was flat");
      require(!f.detected || f.gpu.bits == p.offered, "The tag and the layer were not both offered");
      if (!f.detected) {
        continue;
      }
      if (f.gpu.accepted & candidate::layer) {
        require(f.source == ui_detection::source_layer && f.covered == 14519, "The accepted layer did not decide beside the unaccepted opaque tag");
      } else {
        require(!f.source && f.decision.none_reason == ui_no_mask::unaccepted, "A frame decided before the layer was accepted");
      }
    }
    // The tag never inherits the layer's acceptance: the first real frame
    // without the accepted layer has no decision of its own and reuses the
    // previous real frame's once (T1, the GPU grace); then no mask.
    auto tag_only = p;
    tag_only.offered = candidate::ui_color | candidate::backbuffer | candidate::hudless | candidate::exact;
    for (std::uint32_t i = 0; i != 3; ++i) {
      tag_only.now_ms = 14000 + 16 * i;
      const auto &r = s.step(tag_only);
      require(r.detected && !r.grace && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && !r.gpu.accepted && !r.decision.own_source, "The tagged UI color inherited the layer's acceptance");
      if (!i) {
        require(r.decision.reused && r.source == ui_detection::source_layer && r.covered == 14519, "The missing accepted layer's decision was not reused once");
      } else {
        require(!r.decision.reused && !r.source, "The missing accepted layer's decision was reused twice");
      }
    }
    check_counters(s, "separate slots");
  }

  void acceptance_is_keyed_by_signature() {
    alpha_auto_policy session;
    sequence s(session, [](const gpu_inputs &in) {
      return backbuffer_alpha(200).words(in);
    });
    const auto in_pq = signatures_in(pq).of(kind::backbuffer), in_srgb = signatures_in(srgb).of(kind::backbuffer);
    present hdr;
    hdr.offered = candidate::backbuffer;
    hdr.signatures = signatures_in(pq);
    run(s, hdr, 10000, 13000);
    require(session.accepts(in_pq) && !session.accepts(in_srgb), "The Backbuffer was not accepted in PQ alone");
    // An HDR to SDR switch recreates the swapchain: a new epoch, scope and
    // color space, so another signature that must earn its own acceptance.
    present sdr = hdr;
    sdr.epoch = 2;
    sdr.signatures = signatures_in(srgb);
    sdr.now_ms = 13000;
    const auto &first = s.step(sdr);
    require(first.consumed.state == alpha_auto_state::collecting, "The scope change kept the previous context's sample");
    require(!first.gpu.accepted && !first.source, "Acceptance crossed the color space");
    run(s, sdr, 13016, 16000);
    const auto srgb_first = first_sample_from(s, 13000);
    require(session.accepts(in_srgb) && s.trust.size() == 2 && s.trust[1].tick == first_sample_from(s, srgb_first + alpha_trust_span_ms) && s.trust[1].accepted == stored_of({in_srgb, in_pq}), "The Backbuffer did not earn again in sRGB");
    // Back in HDR the PQ acceptance decides from the first frame.
    present back = hdr;
    back.epoch = 3;
    back.now_ms = 16000;
    const auto &returned = s.step(back);
    require(returned.gpu.accepted == candidate::backbuffer && returned.source == 3, "The PQ acceptance did not decide after switching back");
    // Frame generation is not part of the signature: toggled on, real
    // Presents keep deciding with the same acceptance. Without a HUD-less
    // pairing the generated Presents offer nothing and count as real: each
    // reuses the decision of the real frame before it (T1).
    fg_pacer pacer(1);
    for (std::uint64_t now = 16016; now < 17000; now += 8) {
      const auto kind = pacer.next(true, 1);
      auto q = fg_present(kind, back, 0u, pacer.tag);
      q.now_ms = now;
      const auto &r = s.step(q);
      if (kind == ui_mask::hudless_present::generated_frame) {
        require(r.grace && r.decision.reused && r.source == 3 && !r.submitted, "A zero-offer Present did not reuse the accepted Backbuffer's decision");
      } else {
        require(r.gpu.accepted == candidate::backbuffer && r.source == 3, "An FG toggle changed the acceptance");
      }
    }
    require(s.trust.size() == 2, "Acceptance changed after the color-space round trip or the FG toggle");
    check_counters(s, "signature key");
  }

  void declared_source_is_accepted_by_one_sample() {
    // Stellar Blade in SDR, FG on: the tagged UIColorAndAlpha (0.26%) beside
    // the scene image in the cleared UI layer and the Backbuffer alpha. The
    // declared tag is accepted by its first valid selective sample and decides
    // from the render after that sample's poll; the Backbuffer earns by its run
    // but never decides while the accepted tag is offered (S1, the declared
    // block); the invalid layer (V1) never earns or blocks.
    alpha_auto_policy session;
    sequence s(session, recorded_frames({recorded::sb_sdr, recorded::sb_sdr_tag, recorded::sb_sdr_tag_backbuffer, recorded::sb_sdr_backbuffer}));
    present p;
    p.offered = candidate::ui_color | candidate::layer | candidate::backbuffer;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    run(s, p, 10000, 13000);
    const auto tag = p.signatures.of(kind::ui_color), backbuffer = p.signatures.of(kind::backbuffer);
    const auto first = s.samples.front().sample_tick_ms;
    require(s.trust.size() == 2 && s.trust[0].tick == first && s.trust[0].accepted == tag.key(), "The tag was not accepted by its first sample");
    require(s.trust[1].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[1].accepted == stored_of({tag, backbuffer}), "The Backbuffer was not accepted 2 s into its run");
    require(!session.accepts(p.signatures.of(kind::ui_layer)), "The scene image in the UI layer was accepted");
    const auto poll = poll_of(s, first);
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      const auto &f = s.frames[i];
      if (i <= poll) {
        require(!f.source && !f.gpu.accepted, "A frame decided before the tag's first sample was read");
      } else {
        require(f.source == 2 && f.covered == 21283, "The accepted tag did not decide 21283 pixels from the render after its first sample's poll");
      }
    }
    // A Present without the tag: the accepted tag is missing (T1), but the
    // accepted Backbuffer decides by itself, so nothing is reused.
    auto untagged = p;
    untagged.offered = candidate::layer | candidate::backbuffer;
    untagged.now_ms = 13000;
    const auto &r = s.step(untagged);
    require(r.detected && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && r.gpu.accepted == candidate::backbuffer && r.source == 3 && r.covered == 21283 && !r.decision.reused, "The accepted Backbuffer did not decide without the tag");
    check_counters(s, "declared one-sample acceptance");
  }

  void hudless_pair_is_accepted_by_one_exact_sample() {
    const auto hudless = key(kind::hudless);
    present p;
    p.offered = candidate::current | candidate::hudless | candidate::exact;
    {
      // Hogwarts Legacy FG off, gameplay HUD: the exact pair's change set is
      // accepted by its first selective sample (A1) and decides afterwards.
      alpha_auto_policy session;
      sequence s(session, recorded_frames({recorded::hl_hud, recorded::hl_hud_accepted}));
      run(s, p, 10000, 11000);
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() == 1 && s.trust[0].tick == first && s.trust[0].accepted == hudless, "The exact HUD-less pair was not accepted by its first sample");
      const auto poll = poll_of(s, first);
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(i <= poll ? !f.source && f.decision.none_reason == ui_no_mask::unaccepted : f.source == 5 && f.covered == 329969, "The HUD-less difference did not start right after its first selective exact sample");
      }
      check_counters(s, "Hogwarts HUD");
    }
    {
      // Title screen before acceptance: the full change set never decides
      // (route 6 needs an accepted pair); the HUD-less route (9) flattens only
      // under its CPU-held verdict (H1). Then the gameplay HUD earns the pair,
      // and back on the title the accepted pair decides it flat (6).
      alpha_auto_policy session;
      const auto title = recorded_frames({recorded::hl_title, recorded::hl_title_held, recorded::hl_title_accepted});
      sequence s(session, phased({{12000, title}, {13000, recorded_frames({recorded::hl_hud, recorded::hl_hud_accepted})}, {UINT64_MAX, title}}));
      run(s, p, 10000, 14000);
      std::size_t route = 0;
      for (const auto &f : s.frames) {
        const bool held = (f.gpu.per_frame & ui_detection::per_frame_scene_hold_hudless) != 0;
        if (f.now_ms < 12000) {
          require(f.source != 6, "The unaccepted title pair decided route 6");
          require(f.source == (held ? 9u : 0u), "The title's route 9 did not follow its CPU-held verdict");
          route += f.source == 9 ? 1 : 0;
        } else if (f.now_ms >= 13000) {
          require(f.flat() && f.source == 6, "The accepted pair did not decide the title flat");
        }
      }
      require(route > 0, "The title never took the HUD-less route");
      require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, 12000) && s.trust[0].accepted == hudless, "The gameplay HUD did not accept the pair by its first sample");
      check_counters(s, "Hogwarts title");
    }
  }

  void manual_on_is_a_session_override() {
    // S2: manual On accepts every offered candidate for this session only. The
    // first valid candidate in draw order decides at any coverage (here the
    // opaque Stellar Blade HDR tag pins flat), a source filter changes what is
    // offered, and the ledger never learns, revokes or persists anything.
    alpha_auto_policy session;
    std::vector<std::string> heard;
    session.on_change([&](const std::string &accepted) {
      heard.push_back(accepted);
    });
    session.set_manual(true);
    sequence s(session, recorded_frames({recorded::sb_hdr_manual, recorded::sb_hdr_manual_filtered, recorded::sb_hdr_filtered}));
    present p;
    p.offered = candidate::ui_color | candidate::layer | candidate::backbuffer | candidate::hudless | candidate::exact;
    p.layer_flags = ui_detection::layer_detection_flags(true);
    p.signatures = sb_hdr_signatures();
    run(s, p, 10000, 11000);
    for (const auto &f : s.frames) {
      require(f.detected && f.gpu.accepted == (p.offered & ui_selection::candidate_bits) && f.flat() && f.source == 2, "Manual On did not decide the first valid candidate (the opaque tag) flat");
    }
    // A source filter without the tag: the layer is the first valid candidate.
    auto filtered = p;
    filtered.offered = candidate::layer | candidate::backbuffer | candidate::hudless | candidate::exact;
    run(s, filtered, 11000, 12000);
    for (std::size_t i = 1; i != s.frames.size(); ++i) {
      const auto &f = s.frames[i];
      require(f.now_ms < 11000 || (f.source == ui_detection::source_layer && f.covered == 14519), "Manual On with the filter did not decide the layer");
    }
    const auto c = session.counters();
    require(heard.empty() && session.stored().empty() && s.trust.empty() && session.decision().state == alpha_auto_state::manual_on, "Manual On touched the ledger");
    for (const auto event : {ui_counter::trust_earned, ui_counter::trust_revoked_exact, ui_counter::trust_revoked_declared, ui_counter::trust_lapsed, ui_counter::trust_restored, ui_counter::trust_discarded, ui_counter::trust_forgotten}) {
      require(!c[event], "Manual On counted an acceptance event");
    }
    check_counters(s, "manual On");
    // Back in Auto, nothing was learned: detection pushes accepted 0 and
    // the frame decides nothing of its own. The tag that manual On accepted
    // is missing from the filtered Presents (T1 accepted_missing: the
    // reference stays the last frame that offered it, since flagged frames
    // adopt nothing), so the first Auto frame reuses the previous real
    // frame's decision once, and the next has no mask.
    require(s.temporal.bits == p.offered && s.temporal.accepted == (p.offered & ui_selection::candidate_bits), "A frame missing the accepted tag adopted its inputs");
    session.set_automatic();
    for (std::uint32_t i = 0; i != 2; ++i) {
      filtered.now_ms = 12000 + 16 * i;
      const auto &r = s.step(filtered);
      require(r.detected && !r.gpu.accepted && !r.decision.own_source && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing), "Auto inherited manual On's acceptance");
      require(i ? !r.decision.reused && !r.source : r.decision.reused && r.source == ui_detection::source_layer, "The first Auto frame did not reuse manual On's last decision exactly once");
    }
  }

  // ---------------------------------------------------------------- T1 holds

  // A HUD-less change set of 5% that is valid wherever a pair is offered; it
  // decides where the pair is offered and accepted (source 5).
  texels difference_frame(const gpu_inputs &in) {
    synthetic f;
    f.change_set(50);
    return f.words(in);
  }

  // A session that accepted the stream's HUD-less pair in an earlier session.
  void accept_hudless(alpha_auto_policy &session) {
    restore(session, key(kind::hudless), 1);
  }

  void generated_presents_hold_per_multiplier() {
    // FG 2x, 3x and 4x with a same-batch (exact) HUD-less pair: a generated
    // Present never detects; it shows the decision of the real frame it
    // shows (T1), held.generated once per generated Present.
    alpha_auto_policy session;
    accept_hudless(session);
    sequence s(session, difference_frame);
    fg_pacer pacer(1);
    present real;
    real.offered = candidate::current | candidate::hudless | candidate::exact;  // Current alpha and a same-batch HUD-less pair.
    std::uint64_t now = 10000;
    std::size_t all_generated = 0;
    for (const std::uint32_t generated : {1u, 2u, 3u, 1u}) {
      std::size_t generated_presents = 0, real_presents = 0, since_real = 0;
      const auto until = now + 1000;
      auto kind = ui_mask::hudless_present::unpaired;
      // Change the multiplier only after a real Present.
      while (now < until || kind == ui_mask::hudless_present::generated_frame) {
        kind = pacer.next(true, generated);
        auto p = fg_present(kind, real, candidate::current, pacer.tag);
        p.now_ms = now;
        now += 8;
        const auto &r = s.step(p);
        if (kind == ui_mask::hudless_present::generated_frame) {
          ++generated_presents;
          ++since_real;
          require(r.held && !r.detected && r.hold.kind == hold_kind::generated && r.source == 5 && s.temporal.bits == real.offered && s.temporal.holds == since_real, "A generated Present did not hold the real frame's mask");
        } else {
          require(kind == ui_mask::hudless_present::real_frame && r.detected && r.source == 5 && !s.temporal.holds, "A real Present did not detect afresh");
          require(since_real == generated || s.frames.size() == 1, "The pairing did not give " + std::to_string(generated) + " generated Presents per real one");
          since_real = 0;
          ++real_presents;
        }
      }
      require(generated_presents && real_presents, "A multiplier segment had no Presents");
      all_generated += generated_presents;
    }
    require(!s.discards, "A generated Present discarded a sample");
    check_counters(s, "frame generation holds");
    std::size_t committed_generated = 0;
    for (std::size_t i = 0; i != s.committed_frames; ++i) {
      committed_generated += s.frames[i].held ? 1 : 0;
    }
    require(all_generated > 100 && session.counters()[ui_counter::held_generated] == committed_generated && !session.counters()[ui_counter::held_none], "held.generated did not count the generated Presents");
  }

  // What a frame-generation stream did to the Presents the game actually
  // generated and to its real ones.
  struct fg_tally {
    std::size_t presents {}, real {}, held {}, misread {}, lost {}, extra_holds {}, paired_real {};
  };

  // Present-paired HUD-less difference under frame generation (inexact: no
  // tag batch). cadence(index) gives the game's actual generated frames and
  // the provider's reported ones for the Present with that index; tallies
  // count from Present `from`.
  fg_tally fg_drift(sequence &s, std::size_t count, std::size_t from, const std::function<std::pair<std::uint32_t, std::uint32_t>(std::size_t)> &cadence) {
    fg_pacer pacer(cadence(0).first);
    present real;
    real.offered = candidate::current | candidate::hudless;
    fg_tally t;
    std::uint64_t now = 10000;
    for (std::size_t index = 0; index != count; ++index) {
      const auto [actual, reported] = cadence(index);
      const auto step = pacer.present(true, actual, reported);
      auto p = fg_present(step.kind, real, candidate::current, pacer.tag);
      p.now_ms = now;
      now += 8;
      const auto &r = s.step(p);
      if (index < from) {
        continue;
      }
      ++t.presents;
      const bool paired = r.detected && (r.gpu.bits & candidate::hudless);
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

  void generated_presents_under_multiplier_drift() {
    {
      // 4x, reported as presented: every generated Present holds.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 3u};
      });
      require(t.real == 101 && t.held == 300 && !t.lost && !t.extra_holds && t.paired_real == 101, "A 4x cadence reported as 4x did not hold every generated Present");
      check_counters(s, "4x matching cadence");
    }
    {
      // Presented at 4x while the provider still reports 2x (dynamic frame
      // generation): Presents 2 and 3 of each real frame pair as real or late.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 1u};
      });
      require(t.real == 101 && t.held == 100 && t.misread == 200 && t.lost == 200 && !t.extra_holds && t.paired_real == 101, "The lagging multiplier's pairing moved");
      known_today("S3", "T1/E2", "presented at 4x but reported as 2x, " + std::to_string(t.misread) + " of " + std::to_string(t.held + t.lost) + " generated Presents pair as a real frame (or a late one) and detection reads an interpolated image instead of holding");
      check_counters(s, "lagging multiplier");
    }
    {
      // Presented at 2x while the provider reports 4x, after a matching
      // start: from Present 100 every Present pairs as generated, so none
      // detects from an interpolated image (T1). Present 99 (generated) and
      // Present 100 (real, read as generated) show the next real frame (tag
      // of Present 98) and hold the decision of Present 98; Present 101 shows
      // a third tag, has no mask and ends the chain; no Present pairs as
      // real again, so every later one has no mask (held.none). Fail-safe
      // under the wrong multiplier; S3 identity fixes the pairing.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
      const auto t = fg_drift(s, 300, 100, [](std::size_t index) {
        return std::pair {1u, index < 100 ? 1u : 3u};
      });
      require(s.frames[98].detected && s.frames[98].source == 5 && s.frames[99].held && s.frames[100].held && s.frames[100].source == 5, "The leading multiplier's last holds moved");
      for (std::size_t i = 101; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.unavailable && f.hold.kind == hold_kind::unavailable && !f.detected && !f.source, "A Present read as generated detected, or held a decision of a real frame it does not show");
      }
      require(t.real == 100 && t.extra_holds == 1 && !t.paired_real && !t.misread && !t.held && t.lost == 100, "The leading multiplier's holds moved");
      // No real Present detects, so no sample commits these frames' counts
      // (held.none) until one does.
      check_counters(s, "leading multiplier");
    }
    {
      // 2x to 4x in the middle of a real frame (after its generated Present),
      // reported one real frame late: two generated Presents pair as real or
      // late, then every generated Present holds again.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
      constexpr std::size_t change = 100, reported_from = 103;
      const auto t = fg_drift(s, 300, 0, [](std::size_t index) {
        return std::pair {index < change ? 1u : 3u, index < reported_from ? 1u : 3u};
      });
      // Present 102 is the first real one after the change: Presents 100 and 101 are generated.
      require(t.misread == 2 && !s.frames[change].held && !s.frames[change + 1].held && s.frames[change + 3].held && !t.extra_holds && t.paired_real == t.real, "The mid-frame multiplier change's pairing moved");
      for (std::size_t i = reported_from; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.held || (f.detected && (f.gpu.bits & candidate::hudless)), "A Present after the reported multiplier caught up neither held nor paired");
      }
      // Clearing nothing is right: the FG multiplier is not part of the scope.
      known_today("S3", "T1/E2", "a multiplier change in the middle of a real frame, reported a real frame late, pairs " + std::to_string(t.misread) + " generated Presents as real or late (Present counting, not real-frame identity)");
      check_counters(s, "mid-frame multiplier change");
    }
  }

  void generated_presents_hold_within_the_tag_bound() {
    // No multiplier constant: any number of generated Presents showing the
    // decided real frame or the next one holds its decision (T1).
    alpha_auto_policy session;
    accept_hudless(session);
    sequence s(session, difference_frame);
    present exact;
    exact.offered = candidate::current | candidate::hudless | candidate::exact;
    exact.real_frame = 100;
    exact.now_ms = 10000;
    require(s.step(exact).source == 5, "The exact pair did not decide");
    present generated;
    generated.offered = candidate::current;
    generated.hold_previous = true;
    generated.real_frame = 101;
    for (std::uint32_t i = 0; i != 5; ++i) {
      generated.now_ms = 10008 + 8 * i;
      const auto &r = s.step(generated);
      require(r.held && r.hold.kind == hold_kind::generated && r.source == 5 && s.temporal.holds == i + 1, "A run of generated Presents of one tag did not hold");
    }
    // A Present showing a third real frame has no decision of a frame it
    // shows: no mask, and the chain ends.
    generated.real_frame = 102;
    generated.now_ms = 10048;
    const auto &third = s.step(generated);
    require(third.unavailable && !third.active && !third.source && !s.temporal.have_decision, "A Present beyond the tag bound held a mask");
    generated.real_frame = 101;
    generated.now_ms = 10056;
    require(s.step(generated).unavailable, "A generated Present held after its chain ended");
    // The next real Present starts a new chain.
    exact.real_frame = 102;
    exact.now_ms = 10064;
    const auto &real = s.step(exact);
    require(real.detected && (real.gpu.per_frame & ui_detection::per_frame_hold_reset) && real.source == 5, "The real Present after a chain ended did not detect with hold_reset");
    // DLSS multi-frame generation at 6x (5 generated): generated after 1-5
    // Presents, real at 6, late at 7-8.
    using ui_mask::hudless_present;
    const hudless_present expected[] {hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::generated_frame, hudless_present::real_frame, hudless_present::earlier_real_frame, hudless_present::earlier_real_frame};
    for (std::uint64_t after = 1; after <= 8; ++after) {
      const auto pairing = ui_mask::pair_hudless_present(100, 100 + after, true, 5);
      require(pairing.kind == expected[after - 1] && pairing.presents_ago == (after > 6 ? after - 6 : 0), "The 6x frame generation pairing moved");
    }
    check_counters(s, "tag bound");
    // A 6x stream holds every generated Present.
    alpha_auto_policy six;
    accept_hudless(six);
    sequence stream(six, difference_frame);
    const auto t = fg_drift(stream, 601, 0, [](std::size_t) {
      return std::pair {5u, 5u};
    });
    require(t.real == 101 && t.held == 500 && !t.lost && !t.extra_holds && t.paired_real == 101, "A 6x stream did not hold every generated Present");
    check_counters(stream, "6x");
  }

  void real_frames_without_their_own_decision() {
    // T1, the GPU grace: a real frame that decided nothing while an accepted
    // candidate is missing or offered but invalid reuses the previous real
    // frame's own decision once (reused, the mask pass keeps the mask); the
    // next has no mask.
    struct variant {
      std::uint32_t changed;
      std::size_t reason;
      const char *what;
    };

    for (const auto &v : {variant {990, ui_no_mask::gate_no_hold, "an inexact full change set"}, variant {500, ui_no_mask::difference_failed, "an inexact middle-band change set"}}) {
      // An exact full change set decides 6; an inexact pair of the same
      // frames is invalid (V2), so the accepted HUD-less pair has no decision
      // of its own. From 10200 inexact partial pairs decide 5 themselves.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, [&v](const gpu_inputs &in) {
        synthetic f;
        f.change_set(in.now_ms >= 10200 ? 50 : (in.bits & candidate::exact) ? 990 :
                                                                              v.changed)
          .lit(1000);
        return f.words(in);
      });
      present p;
      p.offered = candidate::current | candidate::hudless | candidate::exact;
      p.now_ms = 10000;
      require(s.step(p).source == 6, "The exact full change set did not decide 6");
      p.offered = candidate::current | candidate::hudless;
      for (std::uint32_t i = 0; i != 3; ++i) {
        p.now_ms = 10016 + 16 * i;
        const auto &r = s.step(p);
        require(r.detected && !r.decision.own_source && r.decision.none_reason == v.reason && r.decision.refused == candidate::hudless, std::string("The real frame of ") + v.what + " did not name its own reason");
        if (!i) {
          require(r.decision.reused && r.flat() && r.source == 6, std::string("The first real frame of ") + v.what + " did not reuse the full change set once");
        } else {
          require(!r.decision.reused && !r.source, std::string("A real frame of ") + v.what + " reused twice");
        }
      }
      for (std::uint32_t i = 0; i != 3; ++i) {
        p.now_ms = 10200 + 16 * i;
        const auto &r = s.step(p);
        require(r.detected && !r.decision.reused && r.source == 5 && r.covered == 50 && r.decision.inexact_difference, "An inexact partial pair did not decide by itself");
      }
      check_counters(s, v.what);
      require(session.counters()[ui_counter::reused] == 1, "The grace was not counted reused once");
    }
    {
      // An accepted Backbuffer that decided and is missing: the next real
      // frame reuses its decision once, then no mask; current alpha never
      // inherits its acceptance. Such frames adopt nothing.
      alpha_auto_policy session;
      restore(session, stored_of({signatures_in(srgb).of(kind::ui_alpha), signatures_in(srgb).of(kind::backbuffer)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(in.bits & candidate::backbuffer ? 200 : 0).words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
      p.real_frame = 20;
      p.now_ms = 10000;
      require(s.step(p).source == 3, "The accepted Backbuffer did not decide");
      p.offered = candidate::current;
      for (std::uint32_t i = 0; i != 4; ++i) {
        p.now_ms = 10016 + 16 * i;
        const auto &r = s.step(p);
        require(r.detected && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && !r.gpu.accepted && !r.decision.own_source && s.temporal.bits == candidate::backbuffer, "Current alpha inherited the missing Backbuffer's acceptance, or the frame adopted");
        require(i ? !r.decision.reused && !r.source : r.decision.reused && r.source == 3 && r.covered == 200, "The missing accepted Backbuffer's decision was not reused exactly once");
      }
      // A generated Present offering an accepted UIAlpha shows the real
      // frame's decision; it never decides from Present-time evidence.
      p.offered = candidate::backbuffer;
      p.now_ms = 10100;
      require(s.step(p).source == 3, "The returning Backbuffer did not decide");
      auto generated = p;
      generated.offered = candidate::ui_alpha | candidate::current;
      generated.hold_previous = true;
      generated.real_frame = 21;
      generated.now_ms = 10116;
      const auto &held = s.step(generated);
      require(held.held && !held.detected && held.hold.kind == hold_kind::generated && held.source == 3, "A generated Present offering an accepted UIAlpha did not hold the real frame's decision");
      check_counters(s, "accepted missing");
    }
  }

  void holds_never_cross_a_scope() {
    // A generated Present after an epoch or viewport change (identity, M1)
    // has no mask; the next real detection pushes hold_reset and never
    // reuses. An observation revision alone (a depth observation loss) is a
    // missing input, not an identity change: generated Presents hold and a
    // real frame without a decision of its own reuses the previous one once
    // (T1). Exact pairs decide 5; an inexact full change set gives the
    // accepted pair no decision of its own.
    for (const int field : {0, 1, 2}) {
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set((in.bits & candidate::exact) ? 50 : 990).lit(1000);
        return f.words(in);
      });
      present exact;
      exact.offered = candidate::current | candidate::hudless | candidate::exact;
      exact.real_frame = 10;
      exact.now_ms = 10000;
      require(s.step(exact).source == 5, "The exact pair did not decide");
      const auto change = [field](present q) {
        if (field == 0) {
          ++q.epoch;
        } else if (field == 1) {
          ++q.revision;
        } else {
          ++q.viewport;
        }
        return q;
      };
      auto generated = change(exact);
      generated.offered = candidate::current;
      generated.hold_previous = true;
      generated.real_frame = 11;
      generated.now_ms = 10008;
      const bool identity = field != 1;
      const auto &g = s.step(generated);
      if (identity) {
        require(g.unavailable && !g.source && !g.detected, "A generated Present held a mask across a scope change");
      } else {
        require(g.held && g.source == 5 && !g.detected, "A generated Present did not hold across an observation revision");
      }
      auto inexact = change(exact);
      inexact.offered = candidate::current | candidate::hudless;
      inexact.real_frame = 11;
      inexact.now_ms = 10016;
      const auto &r = s.step(inexact);
      if (identity) {
        require(r.detected && (r.gpu.per_frame & ui_detection::per_frame_hold_reset) && !r.decision.reused && !r.source, "The first real frame in a new scope reused a decision");
      } else {
        require(r.detected && !(r.gpu.per_frame & ui_detection::per_frame_hold_reset) && r.decision.reused && r.source == 5, "The first real frame after an observation revision did not reuse the decision once");
      }
      // A real Present straight after a scope change: no reuse either.
      auto again = change(exact);
      again.now_ms = 10032;
      require(s.step(again).source == 5, "The exact pair did not decide in the new scope");
      auto other = change(inexact);
      other.now_ms = 10048;
      const auto &o = s.step(other);
      if (identity) {
        require(o.detected && (o.gpu.per_frame & ui_detection::per_frame_hold_reset) && !o.decision.reused && !o.source, "A real frame after a scope change reused a decision");
      } else {
        require(o.detected && !(o.gpu.per_frame & ui_detection::per_frame_hold_reset) && o.decision.reused && o.source == 5, "A real frame after a revision change did not reuse its own scope's decision");
      }
      check_counters(s, "scope");
    }
  }

  void holds_survive_key_churn() {
    // An accepted, V1-invalid offscreen layer beside an accepted Backbuffer
    // that frame generation tags on real Presents only, with a HUD-less
    // pairing: the Backbuffer decides every real frame and every generated
    // Present (offering the layer alone) holds it. The S1 status key, which
    // churned between them, no longer gates a hold.
    alpha_auto_policy session;
    const auto signatures = signatures_in(srgb);
    restore(session, stored_of({signatures.of(kind::ui_layer), signatures.of(kind::backbuffer)}), 2);
    sequence s(session, [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 40, 50).alpha(kind::backbuffer, 40);
      return f.words(in);
    });
    fg_pacer pacer(1);
    present real;
    real.offered = candidate::layer | candidate::backbuffer | candidate::hudless;
    real.layer_flags = ui_detection::layer_detection_flags(false);
    std::size_t generated = 0;
    for (std::uint64_t now = 10000; now < 13000; now += 8) {
      const auto kind = pacer.next(true, 1);
      auto p = fg_present(kind, real, candidate::layer, pacer.tag);
      p.now_ms = now;
      const auto &r = s.step(p);
      if (kind == ui_mask::hudless_present::generated_frame) {
        ++generated;
        require(r.held && r.source == 3 && r.covered == 40, "A generated Present beside the invalid accepted layer did not hold");
      } else {
        require(r.detected && r.source == 3 && r.covered == 40, "The accepted Backbuffer did not decide beside the invalid accepted layer");
      }
    }
    require(generated > 100 && !s.discards && s.frames.back().consumed.state != alpha_auto_state::collecting, "Key churn discarded samples or kept the status collecting");
    check_counters(s, "key churn");
  }

  struct e33_cadence {
    std::uint32_t actual, reported;
  };

  void e33_title_and_load_game_at(const e33_cadence c) {
    // E33 FG on at 2x, 3x and 4x: the Backbuffer tag comes with real Presents
    // only and no HUD-less image pairs, so generated Presents offer nothing.
    // The input provider counts Presents from the last one that offered a
    // tag (ui_mask::generated_without_input): one within the reported
    // generated count is generated and shows its real frame's decision
    // (T1, held.generated), before acceptance (3 samples over 2 s, A1: the
    // inferred Backbuffer decides nothing, S1) and after it. A count reported
    // too low leaves the later generated Presents real: the first reuses the
    // accepted decision once (the GPU grace: reduce and mask passes only, no
    // sample), the next ones have no mask, and before acceptance they are
    // inactive.
    const std::string name = std::to_string(c.actual + 1) + "x reported " + std::to_string(c.reported + 1) + "x";
    const auto e33 = phased({{14000, recorded_frames({recorded::e33_title, recorded::e33_title_accepted})}, {UINT64_MAX, recorded_frames({recorded::e33_load, recorded::e33_load_accepted})}});
    alpha_auto_policy session;
    sequence s(session, e33);
    fg_pacer pacer(c.actual);
    present real;
    real.offered = candidate::backbuffer;
    std::size_t unaccepted_real = 0, held_unaccepted = 0, held_accepted = 0, reused = 0, lost = 0;
    std::uint64_t presents = 0, input_present = 0;
    std::uint32_t real_source = 0, real_covered = 0;
    for (std::uint64_t now = 10000; now < 17000; now += 8) {
      const auto step = pacer.present(true, c.actual, c.reported);
      auto p = fg_present(step.real ? ui_mask::hudless_present::real_frame : ui_mask::hudless_present::generated_frame, real, 0u, pacer.tag);
      p.now_ms = now;
      // The input provider's Present counting without a HUD-less pairing.
      ++presents;
      if (p.offered) {
        input_present = presents;
      } else {
        p.hold_previous = ui_mask::generated_without_input(presents, input_present, true, c.reported);
      }
      const auto &r = s.step(p);
      require(!r.unavailable, "An E33 generated Present had no real decision to show (" + name + ")");
      if (step.real) {
        require(r.detected && !r.grace, "A real E33 Present did not detect (" + name + ")");
        real_source = r.source;
        real_covered = r.covered;
        if (!(r.gpu.accepted & candidate::backbuffer)) {
          require(now < 14000 && !r.source && r.decision.none_reason == ui_no_mask::unaccepted && r.decision.refused == candidate::backbuffer, "The unaccepted title Backbuffer decided (" + name + ")");
          ++unaccepted_real;
        } else if (now < 14000) {
          require(r.source == 3 && r.covered == 197797, "The accepted title Backbuffer did not decide (" + name + ")");
        }
      } else if (p.hold_previous) {
        require(r.held && !r.detected && r.hold.kind == hold_kind::generated && r.source == real_source && r.covered == real_covered, "A generated E33 Present did not show its real frame's decision (" + name + ")");
        ++(real_source ? held_accepted : held_unaccepted);
      } else if (r.detected) {
        require(r.grace && !r.submitted && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && (r.decision.reused ? r.source == 3 : !r.source), "A zero-offer Present beyond the reported count did not reuse the accepted mask at most once (" + name + ")");
        ++(r.decision.reused ? reused : lost);
      } else {
        require(!r.active && !r.source && !real_source, "A zero-offer Present beyond the reported count held a mask decided before acceptance (" + name + ")");
      }
      if (now >= 14100 && c.actual == c.reported) {
        require(r.flat() && r.source == 3, "Load Game over a hidden scene was not flat with the accepted Backbuffer alpha (" + name + ")");
      }
    }
    const auto first = s.samples.front().sample_tick_ms;
    require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[0].accepted == key(kind::backbuffer), "The E33 title did not accept the Backbuffer 2 s into its run (" + name + ")");
    require(unaccepted_real && held_unaccepted && held_accepted, "The E33 stream did not cover the Presents before and after acceptance (" + name + ")");
    const auto counted = session.counters();
    require(!counted[ui_counter::trust_revoked_exact] && !counted[ui_counter::trust_revoked_declared], "Load Game revoked the Backbuffer without a judge (" + name + ")");
    require(counted[ui_counter::held_generated] > 0 && !counted[ui_counter::held_none] && (counted[ui_counter::reused] > 0) == (reused > 0), "The generated Presents were not counted as held (" + name + ")");
    if (c.actual == c.reported) {
      require(!reused && !lost, "A generated Present within the reported count was read as real (" + name + ")");
    } else {
      // Under the reported count of 1 at 4x: the second generated Present of
      // each real frame reuses once and the third has no mask.
      require(reused && lost && reused >= lost && reused <= lost + 1, "A count reported too low did not fall back to the T1 grace (" + name + ")");
    }
    // Load Game is a whole-frame alpha from the accepted Backbuffer, pinned
    // flat (P1) over a hidden scene: the counters measure it hidden, never
    // visible.
    require(counted[ui_counter::full_alpha] > 0 && counted[ui_counter::full_alpha_d_hidden] > 0 && !counted[ui_counter::full_alpha_d_visible], "The counters did not record Load Game's whole-frame alpha over a hidden scene (" + name + ")");
    check_counters(s, "E33 title and Load Game " + name);
  }

  void e33_title_and_load_game() {
    for (const auto c : {e33_cadence {1, 1}, e33_cadence {2, 2}, e33_cadence {3, 3}, e33_cadence {3, 1}}) {
      e33_title_and_load_game_at(c);
    }

    // Booted straight into Load Game: unaccepted full alpha decides nothing
    // (A1: a source that has only been opaque is never accepted).
    alpha_auto_policy fresh;
    sequence boot(fresh, recorded_frames({recorded::e33_load, recorded::e33_load_accepted}));
    present load;
    load.offered = candidate::backbuffer;
    run(boot, load, 10000, 13000);
    for (const auto &f : boot.frames) {
      require(!f.source, "Load Game without acceptance was not empty");
    }
    require(fresh.stored().empty(), "Load Game's opaque Backbuffer was accepted");
  }

  // ---------------------------------------------------------------- F1 staleness and status

  void samples_go_stale_and_are_discarded() {
    {
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
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
      p.offered = candidate::backbuffer;
      run(s, p, 10000, 20000);
      require(s.samples.empty() && s.discards > 5 && session.stored().empty() && !session.counters()[ui_counter::samples], "A sample older than 500 ms on arrival was read");
    }
    {
      // F1: a change of the accepted candidates no longer discards the
      // pending sample; it is evidence for the ledger. The status shows
      // collecting until a sample taken under the new winner arrives.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(200).words(in);
      },
                 3);
      present p;
      p.offered = candidate::backbuffer;
      p.now_ms = 10000;
      require(s.step(p).submitted, "The first frame did not submit a sample");
      restore(session, key(kind::backbuffer), 1);
      for (std::uint32_t i = 1; i != 12; ++i) {
        p.now_ms = 10000 + 16 * i;
        s.step(p);
      }
      require(!s.discards && s.samples.size() == 2 && s.samples[0].sample_tick_ms == 10000, "A sample from other accepted candidates was not read");
      const auto second = poll_of(s, s.samples[1].sample_tick_ms);
      for (std::size_t i = 1; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(i < second ? f.consumed.state == alpha_auto_state::collecting : f.consumed.sample_tick_ms == s.samples[1].sample_tick_ms && f.consumed.state == alpha_auto_state::automatic_on, "The status did not wait for a sample under the new winner");
      }
    }
  }

  // F1: the status is keyed on the scope and the winner (the first adopted,
  // offered and accepted candidate in draw order); no sample is discarded
  // for a change of offered or accepted candidates.
  void accepted_alpha_keys_samples_alone() {
    const auto signatures = signatures_in(srgb);
    {
      // The Witcher 3 or Resident Evil Requiem with FG 2x after an FG-off
      // session accepted the exact HUD-less pair: real Presents offer the
      // accepted UIAlpha and the (inexact) pair, generated ones UIAlpha alone
      // and hold the real frame's decision (T1).
      alpha_auto_policy session;
      restore(session, stored_of({signatures.of(kind::ui_alpha), signatures.of(kind::hudless)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_alpha, 50).change_set(50);
        return f.words(in);
      });
      fg_pacer pacer(1);
      present real;
      real.offered = candidate::ui_alpha | candidate::hudless;
      std::size_t generated = 0;
      for (std::uint64_t now = 10000; now < 13000; now += 8) {
        const auto kind = pacer.next(true, 1);
        auto p = fg_present(kind, real, candidate::ui_alpha, pacer.tag);
        p.now_ms = now;
        const auto &r = s.step(p);
        const bool is_generated = kind == ui_mask::hudless_present::generated_frame;
        generated += is_generated ? 1 : 0;
        require((is_generated ? r.held : r.detected) && r.source == 1 && r.covered == 50, "The accepted UIAlpha did not decide every real Present, or a generated Present did not hold it");
      }
      require(generated > 100 && !s.discards && s.samples.size() >= 25, "An intermittent accepted HUD-less pair discarded samples beside the accepted UIAlpha (" + std::to_string(s.discards) + " discarded, " + std::to_string(s.samples.size()) + " read)");
      require(s.frames.back().consumed.state != alpha_auto_state::collecting, "The status stayed checking");
      check_counters(s, "accepted UIAlpha beside an accepted HUD-less pair");
    }
    {
      // An accepted offscreen layer beside an accepted Backbuffer that frame
      // generation tags on real Presents only (no HUD-less pair, so every
      // Present is real): the layer decides every Present; Presents without
      // the Backbuffer flag it missing but decide by themselves.
      alpha_auto_policy session;
      restore(session, stored_of({signatures.of(kind::ui_layer), signatures.of(kind::backbuffer)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 40).alpha(kind::backbuffer, 40);
        return f.words(in);
      });
      fg_pacer pacer(1);
      present real;
      real.offered = candidate::layer | candidate::backbuffer;
      real.layer_flags = ui_detection::layer_detection_flags(false);
      for (std::uint64_t now = 10000; now < 13000; now += 8) {
        auto p = fg_present(pacer.next(true, 1), real, candidate::layer, pacer.tag);
        p.now_ms = now;
        const auto &r = s.step(p);
        require(r.detected && r.source == ui_detection::source_layer && !r.decision.reused, "The accepted layer did not decide every Present");
      }
      require(!s.discards && s.samples.size() >= 25 && s.frames.back().consumed.state != alpha_auto_state::collecting, "An intermittent accepted Backbuffer discarded samples beside the accepted layer");
      check_counters(s, "accepted layer beside an accepted Backbuffer");
    }
  }

  // ---------------------------------------------------------------- H1 hidden scene

  void hidden_scene_layer_route() {
    // W3 graphics settings: an opaque unaccepted layer over a hidden scene.
    // Phases by sample tick: hidden, invalid, hidden, visible, hidden while
    // refuted, an overlay sample (the layer below opaque), hidden again.
    const auto w3 = recorded_frames({recorded::w3_settings, recorded::w3_settings_held});
    alpha_auto_policy session;
    sequence s(session, [&](const gpu_inputs &in) {
      require(in.bits == (candidate::layer | candidate::current) && !in.accepted, "The W3 settings stream offered other inputs");
      return w3(in);
    });
    // Synthetic edits of the recorded sample: invalid evidence, one visible
    // sample (D 0.5), and overlay samples (the layer below opaque).
    bool visible_sent = false;
    s.sample_edit = [&visible_sent](texels &t, std::uint64_t tick, bool evidence) {
      if (tick >= 16000 && tick < 16200) {
        t[word::layer_opaque] = 0;
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
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = ui_detection::layer_detection_flags(false);
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
        if (hidden && !refuted && sample.evidence.layer_opaque) {
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
    require(session.stored().empty(), "The full opaque layer was accepted");
    check_counters(s, "W3 layer route");
  }

  void hidden_scene_shadow_and_clears() {
    // A gate-open, hidden synthetic stream (an opaque unaccepted layer).
    const auto hidden_layer = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 1000);
      f.c.opaque_layer = 1000;
      f.scene_valid = true;
      f.verdict = scene_verdict::hidden;
      f.d = -.02f;
      f.n = 300;
      f.decided_edges = 600;
      return f.words(in);
    };
    const auto layer_flags = ui_detection::layer_detection_flags(false);
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
        f.decided_edges = 600;
        return f.words(in);
      });
      present p;
      p.offered = candidate::current;
      run(s, p, 10000, 11000);
      const auto first = s.samples.front().sample_tick_ms;
      for (const auto &sample : s.samples) {
        require(sample.evidence.scene.ran && sample.evidence.shadow_hidden_ms == sample.sample_tick_ms - first, "The first-run shadow did not measure every sample");
      }
      require(!s.temporal.scene_hold_until[0] && !s.temporal.scene_hold_until[1], "The shadow held a route");
      alpha_auto_policy gated;
      gated.set_first_run(true);
      sequence g(gated, hidden_layer);
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = layer_flags;
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
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = layer_flags;
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
    route.offered = candidate::ui_alpha | candidate::layer | candidate::current;  // UIAlpha appears: the routes' inputs change.
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
    require(!s.temporal.have_decision && s.temporal.reset_pending, "An inactive frame kept a decision to hold or reuse");
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

  // ---------------------------------------------------------------- recorded acceptance streams

  void recorded_menus_after_acceptance() {
    const auto layer_flags = ui_detection::layer_detection_flags(false);
    {
      // W3 FG off: the notice board's layer (12.48%) decides nothing until it
      // is accepted by its run (A1, S1), then decides; the sign wheel's full
      // layer (a 65% backdrop under a solid wheel) then pins flat over a
      // visible scene and is never revoked. Correct under the opacity ruling:
      // an accepted source pins saturate(8 alpha) at any coverage (P1), and
      // nothing of stronger provenance contradicts it (A2).
      const auto w3 = phased({{14000, recorded_frames({recorded::w3_notice, recorded::w3_notice_accepted})}, {UINT64_MAX, recorded_frames({recorded::w3_wheel, recorded::w3_wheel_accepted})}});
      alpha_auto_policy session;
      sequence s(session, w3);
      present p;
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = layer_flags;
      run(s, p, 10000, 17000);
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[0].accepted == key(kind::ui_layer), "The notice board layer was not accepted once, 2 s into its run");
      for (const auto &f : s.frames) {
        if (!(f.gpu.accepted & candidate::layer)) {
          require(f.now_ms < 14000 && !f.source && f.decision.none_reason == ui_no_mask::unaccepted, "The unaccepted notice board layer decided");
        } else if (f.now_ms < 14000) {
          require(f.source == ui_detection::source_layer && f.covered == 460125, "The accepted notice board layer did not decide");
        } else {
          require(f.flat() && f.source == ui_detection::source_layer, "The accepted sign wheel layer was not flat");
        }
      }
      // The counters see it: whole-frame alpha frames, and the samples after
      // the first one measure the visible scene beneath (a diagnostic only).
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_alpha_d_visible] > 0 && c[ui_counter::full_alpha_d_invalid] == 1 && !c[ui_counter::full_alpha_d_hidden] && !c[ui_counter::contradicted], "The counters did not record the accepted full layer over a visible scene");
      require(!c[ui_counter::trust_revoked_exact] && !c[ui_counter::trust_revoked_declared] && session.stored() == key(kind::ui_layer), "The accepted sign wheel layer was revoked");
      check_counters(s, "W3 FG off");
      // A session that never saw the HUD keeps the sign wheel 3D: a source
      // that has only been opaque or ambiguous is never accepted (A1).
      alpha_auto_policy fresh;
      sequence boot(fresh, w3);
      run(boot, p, 14000, 17000);
      for (const auto &f : boot.frames) {
        require(!f.source, "The unaccepted sign wheel was not empty");
      }
    }
    {
      // W3 FG on 2x: the HUD's declared UIAlpha is accepted by its first
      // sample (A1) and decides afterwards; the sign wheel's full UIAlpha then
      // pins flat on real Presents (P1) and is held on generated ones (T1).
      // The inexact pair never judges it (E2, A2).
      const auto w3 = phased({{14000, recorded_frames({recorded::w3_hud_fg, recorded::w3_hud_fg_accepted})}, {UINT64_MAX, recorded_frames({recorded::w3_wheel_fg, recorded::w3_wheel_fg_accepted})}});
      alpha_auto_policy session;
      sequence s(session, w3);
      fg_pacer pacer(1);
      present real;
      real.offered = candidate::ui_alpha | candidate::current | candidate::hudless;  // UIAlpha, current and an inexact HUD-less pair.
      std::uint32_t last_real = 0;
      for (std::uint64_t now = 10000; now < 17000; now += 8) {
        const auto kind = pacer.next(true, 1);
        auto p = fg_present(kind, real, candidate::current, pacer.tag);
        p.now_ms = now;
        const auto &r = s.step(p);
        require(r.held == (kind == ui_mask::hudless_present::generated_frame) && r.detected == !r.held, "A W3 FG Present was held wrongly");
        if (r.held) {
          require(r.source == last_real, "A generated Present did not hold the real frame's mask");
        } else if (!(r.gpu.accepted & candidate::ui_alpha)) {
          require(!r.source && r.decision.none_reason == ui_no_mask::unaccepted, "The W3 HUD UIAlpha decided before its first sample was read");
        } else if (now < 14000) {
          require(r.source == 1 && r.covered == 192930, "The accepted W3 HUD UIAlpha did not decide");
        }
        if (now >= 14100) {
          require(r.flat() && r.source == 1, "The accepted W3 FG on sign wheel was not flat");
        }
        if (!r.held) {
          last_real = r.source;
        }
      }
      require(s.trust.size() == 1 && s.trust[0].tick == s.samples.front().sample_tick_ms && s.trust[0].accepted == key(kind::ui_alpha), "The W3 HUD UIAlpha was not accepted once by its first sample");
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_alpha_d_visible] > 0 && !c[ui_counter::full_alpha_d_hidden], "The counters did not record the accepted full UIAlpha over a visible scene");
      require(!c[ui_counter::trust_revoked_exact] && !c[ui_counter::trust_revoked_declared], "The accepted sign wheel UIAlpha was revoked");
      check_counters(s, "W3 FG on");
    }
    {
      // RE9 dark room: the offscreen layer (0.18%) decides nothing until its
      // run accepts it, then decides; the full presented alpha never earns.
      alpha_auto_policy session;
      sequence s(session, recorded_frames({recorded::re9_room, recorded::re9_room_accepted}));
      present p;
      p.offered = candidate::layer | candidate::current | candidate::hudless;
      p.layer_flags = layer_flags;
      run(s, p, 10000, 14000);
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[0].accepted == key(kind::ui_layer), "The RE9 layer was not accepted alone, 2 s into its run");
      for (const auto &f : s.frames) {
        require(f.gpu.accepted & candidate::layer ? f.source == ui_detection::source_layer && f.covered == 14611 : !f.source, "The RE9 layer decided before acceptance, or not after it");
      }
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
    const auto current = signatures_in(srgb).of(kind::current), tag = signatures_in(srgb).of(kind::ui_color);
    {
      // RE9-like garbage presented alpha. Fluctuating, it is never accepted.
      alpha_auto_policy session;
      lcg random {12345};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::current, 20 + random.next(961));
        return f.words(in);
      });
      present p;
      p.offered = candidate::current;
      run(s, p, 10000, 40000);
      require(session.stored().empty() && !session.counters()[ui_counter::trust_earned], "Fluctuating presented alpha was accepted");
      for (const auto &f : s.frames) {
        require(!f.source, "Unaccepted presented alpha decided");
      }
    }
    {
      // Stable and selective, garbage presented alpha is accepted. A1 earns
      // it the same way when nothing declared is offered: only the earning
      // void (a declared alpha offered but invalid, below), a judge of
      // stronger provenance (A2) or Forget stops it.
      alpha_auto_policy session;
      lcg random {777};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::current, 140 + random.next(21));
        return f.words(in);
      });
      present p;
      p.offered = candidate::current;
      run(s, p, 10000, 14000);
      require(session.stored() == current.key(), "Stable selective presented alpha was not accepted");
      require(s.trust[0].tick == first_sample_from(s, s.samples.front().sample_tick_ms + alpha_trust_span_ms), "Stable selective presented alpha was not accepted 2 s into its run");
    }
    {
      // RE9 FG off on rejected-tag frames: the tagged UIColorAndAlpha is
      // offered but invalid (more than 1% invalid pixels) in every sample, so
      // every sample is void for earning (A1) and the stable, selective
      // presented alpha beside it is never accepted.
      alpha_auto_policy session;
      lcg random {909};
      sequence s(session, [&random](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_color, 0, 50).alpha(kind::current, 140 + random.next(21));
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_color | candidate::current;
      run(s, p, 10000, 20000);
      require(session.stored().empty() && !session.counters()[ui_counter::trust_earned], "Presented alpha beside an invalid tag was accepted");
      for (const auto &f : s.frames) {
        require(!f.source, "A source decided beside an invalid unaccepted tag");
      }
      check_counters(s, "void earning");
    }
    {
      // A restored tag and presented alpha, the tag invalid in every other
      // half second: the accepted declared tag blocks the presented alpha
      // (S1), so it never decides, whether the tag is valid (the tag decides)
      // or invalid (no decision of its own, counted as presented_blocked
      // while the presented alpha is accepted). The first rejected-tag frame
      // after a valid one reuses the tag's decision once (T1); the valid
      // tag's coverage judges the presented alpha (A2).
      alpha_auto_policy session;
      restore(session, stored_of({tag, current}), 2);
      lcg random {31};
      sequence s(session, [&random](const gpu_inputs &in) {
        const bool rejected = (in.now_ms / 500) % 2 == 0;
        synthetic f;
        f.alpha(kind::ui_color, rejected ? 0 : 15, rejected ? 50 : 0).alpha(kind::current, 140 + random.next(21));
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_color | candidate::current;
      run(s, p, 10000, 16000);
      std::size_t rejected_frames = 0, reused = 0;
      bool previous_valid = false;
      for (const auto &f : s.frames) {
        const bool rejected = (f.now_ms / 500) % 2 == 0;
        require(f.detected && f.source != 4 && f.decision.own_source != 4, "Presented alpha decided beside an accepted tag");
        if (rejected) {
          // presented_blocked while the presented alpha is accepted; once
          // disagreement revokes it (A2), trusted_invalid.
          require(!f.decision.own_source && f.decision.none_reason == ((f.gpu.accepted & candidate::current) ? ui_no_mask::presented_blocked : ui_no_mask::trusted_invalid), "A rejected-tag frame decided, or not for its reason");
          require(previous_valid ? f.decision.reused && f.source == 2 && f.covered == 15 : !f.decision.reused && !f.source, "A rejected-tag frame did not reuse the valid tag's decision exactly once");
          reused += f.decision.reused ? 1 : 0;
          ++rejected_frames;
        } else {
          require(f.source == 2 && f.covered == 15, "The valid accepted tag did not decide");
        }
        previous_valid = !rejected;
      }
      const auto c = session.counters();
      require(rejected_frames && reused && c[ui_counter::none + ui_no_mask::presented_blocked] > 0 && !c[ui_counter::presented_over_dedicated], "The declared block was not counted");
      require(!session.accepts(current) && c[ui_counter::trust_revoked_declared] == 1, "The tag's coverage did not revoke the disagreeing presented alpha");
      check_counters(s, "declared block");
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
        f.alpha(kind::ui_layer, 215 + random.next(71));
        return f.words(in);
      });
      present p;
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = ui_detection::layer_detection_flags(true);
      run(s, p, 10000, 40000);
      const auto c = session.counters();
      require(session.stored() == key(kind::ui_layer) && s.trust.size() == 1 && !c[ui_counter::trust_revoked_exact] && !c[ui_counter::trust_revoked_declared], "The bloom-like layer was not accepted once and kept");
      check_counters(s, "bloom");
    }
    // A dark grainy scene under a full unaccepted layer claim (the W3
    // settings frames): one valid hidden sample, then valid but ambiguous ones.
    for (const auto verdict : {scene_verdict::hidden, scene_verdict::visible}) {
      alpha_auto_policy session;
      sequence s(session, recorded_frames({recorded::w3_settings, recorded::w3_settings_held}));
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
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = ui_detection::layer_detection_flags(false);
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
        known_today("S2b", "H1", "a dark grainy scene under a full unaccepted layer claim flattens " + std::to_string(flat) + " frames on one valid hidden sample; H1 wants 2");
      }
    }
  }

  void s1_invariants() {
    // Every detection of every stream above: no unaccepted inferred alpha
    // decided (untrusted_inferred), and no inferred alpha decided beside an
    // offered, accepted declared alpha (presented_over_dedicated).
    require(invariants.detections > 10000, "Too few detections were checked");
    require(!invariants.untrusted_inferred && !invariants.presented_over_dedicated, "untrusted_inferred " + std::to_string(invariants.untrusted_inferred) + ", presented_over_dedicated " + std::to_string(invariants.presented_over_dedicated));
    // T1: a generated Present never runs detection.
    require(!invariants.generated_detections, std::to_string(invariants.generated_detections) + " generated Presents ran detection");
  }

  // ---------------------------------------------------------------- --log

  // Informational: replays logged status samples through alpha_auto_policy.
  // Logged samples are gated and incomplete (about one per second, not every
  // 100 ms one), and a log names no candidate format, so the signatures are
  // those of the restored keys or placeholders (format 0, sRGB); the
  // prediction is a sketch, never a gate. Lines of earlier versions (trusted
  // slots, the legacy trust bitmask) are read as far as they map.
  std::string field_text(const std::string &line, const std::string &key) {
    for (const char before : {' ', '{'}) {
      const auto at = line.find(std::string(1, before) + key + "=");
      if (at != std::string::npos) {
        const auto begin = at + key.size() + 2;
        const auto end = line.find_first_of(" }", begin);
        return line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
      }
    }
    return {};
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

  // The "{...}" group after `key=`.
  std::string field_group(const std::string &line, const std::string &key) {
    const auto at = line.find(" " + key + "={");
    if (at == std::string::npos) {
      return {};
    }
    const auto begin = at + key.size() + 2;
    return line.substr(begin, line.find('}', begin) + 1 - begin);
  }

  int replay_logs(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
      std::ifstream file(path);
      if (!file) {
        std::fprintf(stderr, "Cannot read %s\n", path.c_str());
        return 2;
      }
      alpha_auto_policy session;
      candidate_signatures signatures;
      signatures.color_space = srgb;
      std::string predicted;
      std::uint32_t fed = 0, logged = 0;
      std::uint64_t day = 0, last_ms = 0;
      std::map<std::string, std::string> last_sample;  // Per runtime.
      std::printf("== %s\n", path.c_str());
      const auto text_after = [](const std::string &line, std::size_t at, const char *end) {
        return line.substr(at, line.find(end, at) - at);
      };
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
        if (const auto at = line.find("restored accepted UI sources "); at != std::string::npos) {
          const auto keys = text_after(line, at + 29, " from an earlier");
          session.restore(keys);
          // The restored keys name their kinds' formats and the color space.
          std::size_t from = 0;
          while (from <= keys.size()) {
            const auto end = std::min(keys.find(',', from), keys.size());
            if (const auto parsed = ui_selection::signature::parse(std::string_view(keys).substr(from, end - from))) {
              signatures.set(parsed->source_kind, parsed->format);
              signatures.color_space = parsed->color_space;
            }
            from = end + 1;
          }
          predicted = session.stored();
          std::printf("%s restored  %s\n", stamp.c_str(), keys.c_str());
          continue;
        }
        if (const auto at = line.find("accepted UI sources are now "); at != std::string::npos) {
          std::printf("%s logged    %s\n", stamp.c_str(), text_after(line, at + 28, ";").c_str());
          ++logged;
          continue;
        }
        if (const auto at = line.find("discarded "); at != std::string::npos && line.find("legacy UI trust entries") != std::string::npos) {
          std::printf("%s discarded %s\n", stamp.c_str(), text_after(line, at + 10, ";").c_str());
          continue;
        }
        // Earlier versions: a trust bitmask, discarded by this one.
        if (const auto at = line.find("restored alpha trust "); at != std::string::npos) {
          std::printf("%s legacy    restored %s (discarded since S1)\n", stamp.c_str(), text_after(line, at + 21, " ").c_str());
          continue;
        }
        if (const auto at = line.find("alpha trust is now "); at != std::string::npos) {
          std::printf("%s legacy    logged %s\n", stamp.c_str(), text_after(line, at + 19, ";").c_str());
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
        const auto opaque = field_quad(field_text(line, "sampled_alpha_opaque"));
        evidence.alpha_opaque = {opaque[0], opaque[1]};
        if (const auto layer = field_group(line, "sampled_layer"); !layer.empty()) {
          evidence.layer_covered = field_number(field_text(layer, "covered"));
          evidence.layer_invalid = field_number(field_text(layer, "invalid"));
          evidence.layer_opaque = field_number(field_text(layer, "opaque"));
          evidence.accepted = field_number(field_text(line, "accepted"), 16);
        } else if (field_text(line, "sampled_ui_layer") == "1" && (evidence.candidates & candidate::ui_color)) {
          // Layout 1: the layer shared the UI color slot (bit 0x2).
          evidence.candidates = (evidence.candidates & ~candidate::ui_color) | candidate::layer;
          evidence.layer_covered = std::exchange(evidence.alpha_covered[1], 0u);
          evidence.layer_invalid = std::exchange(evidence.alpha_invalid[1], 0u);
          evidence.alpha_opaque[1] = 0;
        }
        if (const auto hudless = field_group(line, "sampled_hudless"); !hudless.empty()) {
          evidence.hudless_changed = field_number(field_text(hudless, "changed"));
          evidence.hudless_unchanged = field_number(field_text(hudless, "unchanged"));
          evidence.hudless_invalid = field_number(field_text(hudless, "invalid"));
          evidence.matching_tiles = field_number(field_text(hudless, "matching_tiles"));
          evidence.hudless_lit = field_number(field_text(hudless, "lit"));
        }
        // S2a: the one-way judgment counts (A2), the frame's own reason, its
        // refused candidate and the T1 grace (F1).
        if (const auto one_way = field_group(line, "sampled_one_way"); !one_way.empty()) {
          const auto strong = field_quad(field_text(one_way, "strong")), contradicted = field_quad(field_text(one_way, "contradicted"));
          for (std::size_t i = 0; i != evidence.strong.size(); ++i) {
            evidence.strong[i] = strong[i];
            evidence.contradicted[i] = contradicted[i];
          }
        }
        if (const auto reason = field_text(line, "sampled_reason"); !reason.empty() && reason != "decided") {
          for (std::size_t i = 0; i != ui_no_mask::count; ++i) {
            if (ui_no_mask::names[i] == reason) {
              evidence.frame_reason = std::uint32_t(i);
            }
          }
        }
        if (const auto refused = ui_selection::kind_named(field_text(line, "sampled_refused"))) {
          evidence.refused = ui_selection::bit(*refused);
        }
        evidence.reused = field_text(line, "sampled_reused") == "1";
        evidence.late_layer = field_text(line, "sampled_late_layer") == "1";
        // A log names no validity bits: V1 and V2 follow from the logged counts.
        {
          ui_selection::counts c;
          c.pixels = pixels;
          c.changed = evidence.hudless_changed;
          c.unchanged = evidence.hudless_unchanged;
          c.nonfinite = evidence.hudless_invalid;
          c.lit = evidence.hudless_lit;
          c.matching_tiles = evidence.matching_tiles;
          std::uint32_t valid = 0;
          if (ui_selection::change_set_selective(c) || ui_selection::change_set_full(c, (evidence.candidates & candidate::exact) != 0)) {
            valid |= candidate::hudless;
          }
          for (const auto k : ui_selection::draw_order) {
            if (ui_selection::alpha_kind(k) && ui_selection::alpha_valid(alpha_counts_of(evidence, k).invalid, pixels)) {
              valid |= ui_selection::bit(k);
            }
          }
          evidence.valid_bits = valid & evidence.candidates & ui_selection::candidate_bits;
        }
        session.observe(evidence, pixels, ms, signatures);
        ++fed;
        if (auto now = session.stored(); now != predicted) {
          predicted = std::move(now);
          std::printf("%s predicted %s\n", stamp.c_str(), predicted.empty() ? "none" : predicted.c_str());
        }
      }
      std::printf("%u logged samples fed, %u logged acceptance changes, predicted acceptance at the end %s\n", fed, logged, predicted.empty() ? "none" : predicted.c_str());
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
    {"A1 inferred earn", inferred_source_is_accepted_by_steady_selective_samples},
    {"A1 restart", acceptance_restarts_on_a_full_sample_or_a_factor_two_break},
    {"A2/P1 revoke by stronger provenance", acceptance_is_revoked_by_contradiction},
    {"A3 provisional restore and legacy discard", restored_acceptance_is_provisional},
    {"A3 invalid declared re-arm and Forget", invalid_declared_source_never_lapses_and_forget_clears},
    {"E1/S1/A1 separate slots", separate_slots_never_accept_the_opaque_tag},
    {"A1 signature key", acceptance_is_keyed_by_signature},
    {"A1/S1 declared one-sample acceptance", declared_source_is_accepted_by_one_sample},
    {"A1/S1/H1 Hogwarts exact HUD-less", hudless_pair_is_accepted_by_one_exact_sample},
    {"S2 manual On", manual_on_is_a_session_override},
    {"T1 frame generation holds", generated_presents_hold_per_multiplier},
    {"T1/E2 frame generation multiplier drift", generated_presents_under_multiplier_drift},
    {"T1 tag bound and 6x", generated_presents_hold_within_the_tag_bound},
    {"T1 grace of real frames without their own decision", real_frames_without_their_own_decision},
    {"T1 no hold across a scope change", holds_never_cross_a_scope},
    {"T1 holds survive key churn", holds_survive_key_churn},
    {"A1/S1/T1/P1 E33 title and Load Game", e33_title_and_load_game},
    {"F1 sample staleness and status key", samples_go_stale_and_are_discarded},
    {"F1 winner keys samples", accepted_alpha_keys_samples_alone},
    {"H1 W3 layer route", hidden_scene_layer_route},
    {"H1 shadow and scope clears", hidden_scene_shadow_and_clears},
    {"P1/A1/S1 recorded menus", recorded_menus_after_acceptance},
    {"A1/S1 adversaries", adversaries},
    {"S1/T1 invariants", s1_invariants},
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

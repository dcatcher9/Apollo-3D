// SPDX-License-Identifier: GPL-3.0-only
// Sequence replay of live automatic UI detection's temporal rules
// (docs/reshade-sbs.md, UI protection and UI decision framework). Per-frame
// decision streams drive the production state machines without a GPU: the
// game session's acceptance ledger (alpha_auto_policy), the renderer's holds,
// status samples and CPU-owned hidden-scene verdicts
// (ui_temporal::detection_state), the sample decode and counter commit
// (ui_temporal::decode_detection_sample, sample_counters), Present pairing
// under frame generation (ui_mask::pair_hudless_present) and the exact
// counters (ui_counters). A stream's GPU input is what one detection counts:
// decision texels 0-7 that ui_detection_replay --verbose recorded on labelled
// dumps with the S1 shader, or synthetic counts. Every decision is
// ui_selection::decide, the C++ mirror of SunshineUIDetectionReduceCS that
// test_game3d_ui_selection_contract proves equal to the shader's reduce; every
// recorded decision must equal it.
//
// It asserts the rules through stage S1 strictly. An outcome that a later
// stage of the UI decision framework (docs/reshade-sbs.md, UI decision
// framework: stages S0-S6; rules E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1,
// F1) changes prints "KNOWN_TODAY <stage> <rule>: <text>" and does not fail;
// that stage turns it into a strict assertion. Eight remain after S1: S2a T1
// (three), S2b H1 (three) and S3 T1/E2 (two). Outcomes the rules already call
// correct, such as an accepted source pinning a whole-frame alpha flat over a
// visible scene (P1, the opacity ruling), are asserted strictly.
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

  // Invariants every stream's detections keep (S1): no unaccepted inferred
  // alpha decides, and no inferred alpha decides beside an accepted declared one.
  struct {
    std::uint64_t detections {}, untrusted_inferred {}, presented_over_dedicated {};
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

  // Decision texels 0-7 as the renderer reads them back.
  constexpr std::size_t texel_words = 4 * ui_detection::layer_decision_texels;
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
      throw std::runtime_error("Recorded texels are not 32 words");
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

  // What one frame's detection reads: the adopted inputs and the per-frame
  // bits (game3d_renderer.cpp, detect_ui's b2 constants: the candidates, the
  // layer's stored flags with the per-frame bits, the accepted candidates).
  struct gpu_inputs {
    std::uint32_t bits {}, flags {}, accepted {}, per_frame {};
    std::uint64_t now_ms {};
  };

  using gpu_model = std::function<texels(const gpu_inputs &)>;

  // The reduce's decision of these counts and inputs (ui_selection::decide).
  ui_selection::decision decide(const texels &t, const gpu_inputs &in) {
    return ui_selection::decide(ui_selection::counts_from_words(t.data(), t.size()), in.bits, in.accepted, in.flags | in.per_frame);
  }

  // The counts of `t` with the words the reduce writes for these inputs.
  texels decision_words(texels t, const gpu_inputs &in) {
    const auto d = decide(t, in);
    t[word::source] = d.source;
    t[word::covered] = d.covered;
    t[word::candidates] = in.bits;
    t[word::accepted] = in.accepted;
    t[word::valid_bits] = d.valid_bits;
    return t;
  }

  // One recorded detection and the per-frame hold bits it was replayed with.
  struct recording {
    std::string_view words;
    std::uint32_t per_frame {};
  };

  // Recorded decision texels: ui_detection_replay --verbose with the S1 shader
  // on E:/ApolloDev/sbs_dump. Words 4 and 17 hold the candidate bits and the
  // accepted candidates the frame was decided with. Each names its dump and
  // the replay case of the same inputs in ui_detection_cases.json
  // (scratchpad s1/replay_s1_verbose.txt); the acceptance combinations those
  // cases do not hold were replayed alone (scratchpad s1/p4/cases_p4.json).
  namespace recorded {
    constexpr std::uint32_t scene_hold = ui_detection::per_frame_scene_hold;
    constexpr std::uint32_t scene_hold_hudless = ui_detection::per_frame_scene_hold_hudless;
    // The Witcher 3 Remastered, FG off, graphics settings over a hidden scene
    // (game3d_46312_198402901901355): a full opaque offscreen layer (0x40)
    // beside current alpha, unaccepted, without a CPU hold and held ("W3
    // graphics settings, layer hidden-scene hold").
    constexpr recording w3_settings {"0,0,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,0,0,0,0,3686400,0,3686400,72"};
    constexpr recording w3_settings_held {"8,3686400,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,0,0,0,0,3686400,0,3686400,72", scene_hold};
    // W3 FG off, notice board (game3d_46312_198402901901363): layer 12.48%,
    // unaccepted and accepted ("W3 notice board FG off, trusted layer, hold").
    constexpr recording w3_notice {"0,0,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,0,0,0,2029,1049969278,15,8116,0,0,0,0,460125,0,100337,72"};
    constexpr recording w3_notice_accepted {"10,460125,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,64,0,0,2029,1049969278,15,8116,0,0,0,0,460125,0,100337,72"};
    // W3 FG off, sign wheel over a visible scene (game3d_46312_198402901901353):
    // a full layer, unaccepted and accepted ("W3 sign wheel FG off, trusted
    // layer, hold").
    constexpr recording w3_wheel {"0,0,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,0,0,0,564,1057143089,15,2256,0,0,0,0,3686400,0,145726,72"};
    constexpr recording w3_wheel_accepted {"10,3686400,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,64,0,0,564,1057143089,15,2256,0,0,0,0,3686400,0,145726,72"};
    // W3 FG on, HUD (game3d_46312_198402901901357): UIAlpha 5.23% with an
    // inexact HUD-less pair, unaccepted ("W3 HUD FG on, not yet accepted
    // UIAlpha, hold") and accepted ("W3 HUD FG on, trusted UIAlpha, hold").
    constexpr recording w3_hud_fg {"0,0,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,0,103614,0,472,1056573614,15,1888,472,1056964608,3,0,0,0,0,9"};
    constexpr recording w3_hud_fg_accepted {"1,192930,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,1,103614,0,472,1056573614,15,1888,472,1056964608,3,0,0,0,0,9"};
    // W3 FG on, sign wheel (game3d_46312_198402901901359): a full UIAlpha,
    // unaccepted and accepted ("W3 sign wheel FG on, ... UIAlpha, hold").
    constexpr recording w3_wheel_fg {"0,0,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,0,149627,0,462,1056020436,15,1848,462,1057709052,3,0,0,0,0,9"};
    constexpr recording w3_wheel_fg_accepted {"1,3686400,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,1,149627,0,462,1056020436,15,1848,462,1057709052,3,0,0,0,0,9"};
    // Clair Obscur: Expedition 33, FG on, title over a visible scene
    // (game3d_31636_135749029986373): Backbuffer alpha 2.38%, unaccepted and
    // accepted ("E33 title FG on, untrusted/trusted backbuffer, hold").
    constexpr recording e33_title {"0,0,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,0,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4"};
    constexpr recording e33_title_accepted {"3,197797,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,4,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4"};
    // E33 FG on, Load Game over a hidden scene (game3d_59540_257918763574026):
    // a full Backbuffer alpha ("E33 load game FG on, untrusted / trusted
    // backbuffer").
    constexpr recording e33_load {"0,0,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,0,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4"};
    constexpr recording e33_load_accepted {"3,8294400,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,4,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4"};
    // Resident Evil Requiem, dark room (game3d_52696_225539427440975): the
    // offscreen layer 0.18% beside a full presented alpha and an inexact pair,
    // unaccepted ("RE9 dark room, untrusted layer, inexact pair, hold") and
    // accepted.
    constexpr recording re9_room {"0,0,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,386,1058855305,15,1544,386,1057529644,3,0,14611,0,3540,72"};
    constexpr recording re9_room_accepted {"10,14611,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,64,0,0,386,1058855305,15,1544,386,1057529644,3,0,14611,0,3540,72"};
    // Stellar Blade in SDR, FG on (game3d_69460_296226962143478): the tagged
    // UIColorAndAlpha (0.26%), the scene image in the cleared UI layer (V1
    // invalid) and the Backbuffer alpha; nothing accepted, the tag, the tag
    // and the Backbuffer ("SB SDR FG on, accepted tag beside the scene layer
    // and the Backbuffer"), and the Backbuffer of a Present without the tag.
    constexpr recording sb_sdr {"0,0,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,0,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6"};
    constexpr recording sb_sdr_tag {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,2,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6"};
    constexpr recording sb_sdr_tag_backbuffer {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,6,0,6885,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,6"};
    constexpr recording sb_sdr_backbuffer {"3,21283,8294400,0,68,8294319,41,0,0,0,21283,8294400,0,0,0,0,0,4,0,0,1088,1052816565,15,4352,0,0,0,0,0,8261496,0,4"};
    // Stellar Blade in HDR, FG on (game3d_69460_296226962143474): the opaque
    // final-image tag, the offscreen layer 0.18%, a full Backbuffer alpha and
    // an exact pair; nothing accepted, the layer accepted ("SB HDR FG on HUD,
    // unaccepted opaque tag beside an accepted layer, exact pair"), and the
    // tag without the layer ("SB HDR FG on HUD, opaque tag, backbuffer and
    // exact pair"). Manual On accepts every offered candidate, with and without
    // a source filter that leaves the tag out.
    constexpr recording sb_hdr {"0,0,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70"};
    constexpr recording sb_hdr_layer {"10,14519,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,64,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70"};
    constexpr recording sb_hdr_tag_only {"0,0,8294400,3,54,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,0,0,0,6"};
    constexpr recording sb_hdr_manual {"2,8294400,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,86,0,8294400,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,70"};
    constexpr recording sb_hdr_manual_filtered {"10,14519,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,84,0,0,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,68"};
    constexpr recording sb_hdr_filtered {"0,0,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,0,0,0,538,1057510335,15,2152,538,1057588296,3,0,14519,0,206,68"};
    // Hogwarts Legacy, FG off, title screen (game3d_50196_216992971069363): an
    // exact HUD-less pair differing almost everywhere over a hidden scene;
    // unaccepted ("HL title screen FG off, HUD-less not yet accepted"), under
    // the HUD-less route's hold, and accepted ("HL title screen FG off").
    constexpr recording hl_title {"0,0,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24"};
    constexpr recording hl_title_held {"9,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24", scene_hold_hudless};
    constexpr recording hl_title_accepted {"6,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,16,0,0,511,1025396520,7,2041,511,1057900324,3,0,0,0,0,24"};
    // Hogwarts Legacy, FG off, gameplay HUD (game3d_40404_173061335442886):
    // an exact pair whose change set is the HUD (3.98%); unaccepted ("HL
    // gameplay HUD FG off (b), HUD-less not yet accepted") and accepted ("HL
    // gameplay HUD FG off (b)").
    constexpr recording hl_hud {"0,0,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,0,0,0,397,1061190608,15,1588,397,1061190608,3,0,0,0,0,24"};
    constexpr recording hl_hud_accepted {"5,329969,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,16,0,0,397,1061190608,15,1588,397,1061190608,3,0,0,0,0,24"};
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
      require(d.source == t[word::source] && d.covered == t[word::covered] && d.valid_bits == t[word::valid_bits], "The recording for candidates " + hex(own.bits) + " accepted " + hex(own.accepted) + " differs from ui_selection::decide");
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

    // A HUD-less pair whose change set is valid where offered (V2).
    synthetic &change_set(std::uint32_t changed) {
      c.changed = changed;
      c.unchanged = c.pixels - changed;
      c.matching_tiles = 200;
      return *this;
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
    // The detection's decision (none_reason when it decided nothing).
    ui_selection::decision decision;
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
  // order game3d_renderer.cpp does (game3d_ui_temporal.h, its header comment):
  //   render():           accepted (session), arbitrate, the held inputs,
  //                       adopt, detection_active, enter_scope, scene_holds,
  //                       then held(), or poll_detection + scene_holds +
  //                       detect_ui + detected(), or inactive();
  //   end of render:      the detection fence is signaled (detection_awaiting_signal);
  //   update_alpha_auto:  the 500 ms staleness of the latest sample;
  //   poll_detection:     sample_discarded, decode_detection_sample,
  //                       observe_scene, session.observe with the signatures
  //                       the sample was submitted with, commit_counters
  //                       (sample_counters);
  //   detect_ui:          the 100 ms sample cadence, scene evidence by
  //                       measure_scene, the counter and CPU snapshot.
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

    const frame_result &step(const present &p) {
      frame_result r;
      r.now_ms = p.now_ms;
      const auto index = frames.size();
      std::uint32_t bits = p.offered;
      std::uint32_t accepted = session.accepted(bits, p.signatures);
      r.hold = temporal.arbitrate(bits, accepted, p.hold_previous, max_held_presents);
      std::uint32_t flags = (bits & candidate::layer) ? p.layer_flags : 0u;
      if (r.hold.hold) {
        bits = temporal.bits;
        flags = temporal.flags;
        accepted = temporal.accepted;
      }
      temporal.adopt(bits, flags, accepted);
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
          detect(observation, p.signatures, per_frame, shadow, index, r);
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
      latest = ui_temporal::decode_detection_sample(pending_texels.data(), pending_texels.size(), pending_source.now_ms, submitted, true);
      temporal.latest_source = pending_source;
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

    void detect(const alpha_auto_source &observation, const candidate_signatures &signatures, std::uint32_t per_frame, bool shadow, std::size_t index, frame_result &r) {
      r.gpu = {temporal.bits, temporal.flags, temporal.accepted, per_frame, observation.now_ms};
      texels t = gpu(r.gpu);
      // The reduce decides and counts (SunshineUIDetectionReduceCS, mirrored
      // by ui_selection::decide and counter_adds).
      r.decision = decide(t, r.gpu);
      require(r.decision.source == t[word::source] && r.decision.covered == t[word::covered], "A stream's GPU model did not decide as ui_selection::decide");
      const auto adds = ui_selection::counter_adds(r.decision, r.gpu.flags | per_frame);
      for (std::size_t i = 0; i != adds.size(); ++i) {
        gpu_words[i] += adds[i];
      }
      ++invariants.detections;
      invariants.untrusted_inferred += r.decision.untrusted_inferred ? 1 : 0;
      invariants.presented_over_dedicated += r.decision.presented_over_dedicated ? 1 : 0;
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
        std::fill(t.begin() + scene_texels_begin, t.begin() + scene_texels_end, 0u);
      }
      if (sample_edit) {
        sample_edit(t, now, evidence);
      }
      pending_texels = t;
      pending_signatures = signatures;
      counters_pending = true;
      pending_words = gpu_words;
      pending_cpu = cpu;
      pending_frame = index;
      pending_source = observation;
      pending_key = temporal.key();
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
    candidate_signatures pending_signatures;
    texels pending_texels {};
    std::uint32_t applied_source {}, applied_covered {}, applied_pixels {};
    std::string last_accepted;
  };

  bool alpha_source(std::uint32_t source) {
    return (source >= 1 && source <= 4) || source == ui_detection::source_layer;
  }

  // The session's committed totals equal this stream's frames through the
  // last committed sample, and reconcile; no unaccepted inferred alpha decided
  // and none decided beside an accepted declared alpha (S1).
  void check_counters(sequence &s, const std::string &what) {
    const auto totals = s.session.counters();
    require(totals.reconciled(), what + ": the committed counters do not reconcile");
    ui_counters expected;
    std::uint64_t full_alpha = 0;
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
        if (!f.source) {
          ++expected[ui_counter::none + f.decision.none_reason];
        }
        if (f.hold.cap_reached) {
          ++expected[ui_counter::held_cap];
        }
        full_alpha += alpha_source(f.source) && f.pixels && std::uint64_t(f.covered) * 100 >= std::uint64_t(f.pixels) * 99 ? 1 : 0;
      }
    }
    require(totals[ui_counter::full_alpha] == full_alpha, what + ": full_alpha is " + std::to_string(totals[ui_counter::full_alpha]) + ", the stream says " + std::to_string(full_alpha));
    expected[ui_counter::samples] = s.commits;
    for (const auto index : {ui_counter::auto_frames, ui_counter::detection_frames, ui_counter::held_generated, ui_counter::held_inexact_after_exact, ui_counter::held_trusted_missing, ui_counter::held_cap, ui_counter::inactive_no_candidates, ui_counter::samples}) {
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
    const auto backbuffer = signatures_in(srgb).of(kind::backbuffer);
    // An accepted channel covering the whole frame over an exact pair showing
    // at least 75% unchanged: revoked by the first such sample 2 s into the run.
    const auto full_over_pair = [](std::uint32_t unchanged) {
      return [unchanged](const gpu_inputs &in) {
        auto f = backbuffer_alpha(in.now_ms < 13000 ? 200 : 1000);
        f.c.unchanged = in.now_ms < 13000 ? 900 : unchanged;
        f.c.changed = 1000 - f.c.unchanged;
        return f.words(in);
      };
    };
    {
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      sequence s(session, full_over_pair(800));
      present p;
      p.offered = candidate::backbuffer | candidate::hudless | candidate::exact;
      run(s, p, 10000, 17000);
      const auto first_full = first_sample_from(s, 13000);
      const auto revoked = first_sample_from(s, first_full + alpha_trust_span_ms);
      require(s.trust.size() == 2 && s.trust[0].accepted == backbuffer.key() && s.trust[1].tick == revoked && s.trust[1].accepted.empty(), "A full claim over a visible exact pair was not revoked 2 s into the run");
      require(heard == std::vector<std::string> {backbuffer.key(), ""} && session.counters()[ui_counter::trust_revoked_full] == 1, "The contradiction revocation was not heard or counted once");
      std::uint64_t flat_ms = 0;
      for (const auto &f : s.frames) {
        if (f.now_ms >= 13000 && f.gpu.accepted == candidate::backbuffer) {
          flat_ms = f.now_ms - 13000;
        }
      }
      // An accepted source pins its whole-frame alpha flat (P1, the opacity
      // ruling) until the exact pair's repeated contradictions revoke it (A2):
      // it decides up to the poll of the revoking sample, and not after.
      require(flat_ms + 13000 >= revoked && flat_ms + 13000 <= revoked + sample_interval_ms, "The accepted full claim did not decide until its revocation (" + std::to_string(flat_ms) + " ms)");
      // Its frames before the revocation count as trusted_full.
      const auto c = session.counters();
      require(c[ui_counter::trusted_full] > 0 && c[ui_counter::trusted_full] <= c[ui_counter::full_alpha], "The contradicted full claim was not counted as trusted_full");
      check_counters(s, "full-claim revocation");
    }
    for (const auto &[bits, unchanged, what] : {std::tuple {candidate::backbuffer | candidate::hudless | candidate::exact, 700u, "a pair under 75% unchanged"}, std::tuple {candidate::backbuffer | candidate::hudless, 800u, "an inexact pair"}}) {
      alpha_auto_policy session;
      sequence s(session, full_over_pair(unchanged));
      present p;
      p.offered = bits;
      run(s, p, 10000, 20000);
      // A2: only valid same-sample evidence of stronger provenance revokes.
      // A middle-band pair is invalid (V2) and an inexact one never judges
      // (E2); ambiguous or invalid self-doubt is not a revocation.
      require(s.trust.size() == 1 && session.accepts(backbuffer), std::string("A full claim over ") + what + " revoked acceptance");
    }

    // Presented alpha differing from an accepted dedicated UIAlpha by at
    // least 10% of the frame: revoked by the first such sample 2 s into the
    // run, and not earned again while it disagrees. The declared UIAlpha is
    // accepted by its first sample (A1), the Backbuffer by its run, and while
    // the UIAlpha is accepted and offered it alone decides (S1).
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
    const auto revoked = first_sample_from(s, first_disagreement + alpha_trust_span_ms);
    require(s.trust.size() >= 3 && s.trust[2].tick == revoked && s.trust[2].accepted == ui_alpha.key(), "Disagreeing presented alpha was not revoked 2 s into the disagreement");
    const auto agreed = first_sample_from(s, 20000);
    require(s.trust.size() == 4 && s.trust[3].tick == first_sample_from(s, agreed + alpha_trust_span_ms) && s.trust[3].accepted == both, "Presented alpha earned during disagreement, or not after it");
    for (const auto &f : s.frames) {
      require(!f.detected || f.source == (f.gpu.accepted & candidate::ui_alpha ? 1u : 0u), "The Backbuffer decided beside the accepted UIAlpha");
    }
    const auto c = session.counters();
    require(c[ui_counter::trust_revoked_presented] == 1 && c[ui_counter::trust_earned] == 3, "Presented-alpha revocation or the re-earn was not counted");
    check_counters(s, "presented disagreement");
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
    // The tag never inherits the layer's acceptance: without the layer the
    // missing accepted layer's mask is held (T1), then the tag detects alone.
    auto tag_only = p;
    tag_only.offered = candidate::ui_color | candidate::backbuffer | candidate::hudless | candidate::exact;
    for (std::uint32_t i = 0; i != max_held_presents + 1; ++i) {
      tag_only.now_ms = 14000 + 16 * i;
      const auto &r = s.step(tag_only);
      if (i < max_held_presents) {
        require(r.held && r.hold.kind == hold_kind::trusted_missing && r.source == ui_detection::source_layer, "The missing accepted layer's mask was not held");
      } else {
        require(r.detected && r.hold.cap_reached && !r.gpu.accepted && !r.source, "The tagged UI color inherited the layer's acceptance");
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
    // Presents keep deciding with the same acceptance.
    fg_pacer pacer(1);
    for (std::uint64_t now = 16016; now < 17000; now += 8) {
      const auto kind = pacer.next(true, 1);
      auto q = fg_present(kind, back, 0u);
      q.now_ms = now;
      const auto &r = s.step(q);
      require(kind == ui_mask::hudless_present::generated_frame || (r.gpu.accepted == candidate::backbuffer && r.source == 3), "An FG toggle changed the acceptance");
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
    // A Present without the tag: the accepted Backbuffer decides by itself.
    auto untagged = p;
    untagged.offered = candidate::layer | candidate::backbuffer;
    untagged.now_ms = 13000;
    const auto &r = s.step(untagged);
    require(r.detected && r.gpu.accepted == candidate::backbuffer && r.source == 3 && r.covered == 21283, "The accepted Backbuffer did not decide without the tag");
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
    for (const auto event : {ui_counter::trust_earned, ui_counter::trust_revoked_full, ui_counter::trust_revoked_presented, ui_counter::trust_lapsed, ui_counter::trust_restored, ui_counter::trust_discarded}) {
      require(!c[event], "Manual On counted an acceptance event");
    }
    check_counters(s, "manual On");
    // Back in Auto, nothing was learned.
    session.set_automatic();
    filtered.now_ms = 12000;
    const auto &r = s.step(filtered);
    require(r.detected && !r.gpu.accepted && !r.source, "Auto inherited manual On's acceptance");
  }

  // ---------------------------------------------------------------- holds

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
    alpha_auto_policy session;
    accept_hudless(session);
    sequence s(session, difference_frame);
    fg_pacer pacer(1);
    present real;
    real.offered = candidate::current | candidate::hudless | candidate::exact;  // Current alpha and a same-batch HUD-less pair.
    std::uint64_t now = 10000;
    for (const std::uint32_t generated : {1u, 2u, 3u, 1u}) {
      std::size_t generated_presents = 0, real_presents = 0, since_real = 0;
      const auto until = now + 1000;
      auto kind = ui_mask::hudless_present::unpaired;
      // Change the multiplier only after a real Present.
      while (now < until || kind == ui_mask::hudless_present::generated_frame) {
        kind = pacer.next(true, generated);
        auto p = fg_present(kind, real, candidate::current);
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
    real.offered = candidate::current | candidate::hudless;
    fg_tally t;
    std::uint64_t now = 10000;
    for (std::size_t index = 0; index != count; ++index) {
      const auto [actual, reported] = cadence(index);
      const auto step = pacer.present(true, actual, reported);
      auto p = fg_present(step.kind, real, candidate::current);
      p.now_ms = now;
      now += 8;
      const auto &r = s.step(p);
      if (index < from) {
        continue;
      }
      ++t.presents;
      t.cap += r.hold.cap_reached ? 1 : 0;
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
      require(t.real == 101 && t.held == 300 && !t.lost && !t.extra_holds && t.paired_real == 101 && !t.cap, "A 4x cadence reported as 4x did not hold every generated Present");
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
      require(t.real == 101 && t.held == 100 && t.misread == 200 && t.lost == 200 && !t.extra_holds && t.paired_real == 101 && !t.cap, "The lagging multiplier's pairing moved");
      known_today("S3", "T1/E2", "presented at 4x but reported as 2x, " + std::to_string(t.misread) + " of " + std::to_string(t.held + t.lost) + " generated Presents pair as a real frame (or a late one) and detection reads an interpolated image instead of holding");
      check_counters(s, "lagging multiplier");
    }
    {
      // Presented at 2x while the provider reports 4x, after a matching
      // start: a real Present is held as a generated one, the hold reaches
      // the cap and the mask is lost while the multiplier stays wrong.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
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
      accept_hudless(session);
      sequence s(session, difference_frame);
      constexpr std::size_t change = 100, reported_from = 103;
      const auto t = fg_drift(s, 300, 0, [](std::size_t index) {
        return std::pair {index < change ? 1u : 3u, index < reported_from ? 1u : 3u};
      });
      // Present 102 is the first real one after the change: Presents 100 and 101 are generated.
      require(t.misread == 2 && !s.frames[change].held && !s.frames[change + 1].held && s.frames[change + 3].held && !t.extra_holds && !t.cap && t.paired_real == t.real, "The mid-frame multiplier change's pairing moved");
      for (std::size_t i = reported_from; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.held || (f.detected && (f.gpu.bits & candidate::hudless)), "A Present after the reported multiplier caught up neither held nor paired");
      }
      // Clearing nothing is right: the FG multiplier is not part of the scope.
      known_today("S3", "T1/E2", "a multiplier change in the middle of a real frame, reported a real frame late, pairs " + std::to_string(t.misread) + " generated Presents as real or late (Present counting, not real-frame identity)");
      check_counters(s, "mid-frame multiplier change");
    }
  }

  void holds_are_capped() {
    alpha_auto_policy session;
    accept_hudless(session);
    sequence s(session, difference_frame);
    present exact;
    exact.offered = candidate::current | candidate::hudless | candidate::exact;
    exact.now_ms = 10000;
    require(s.step(exact).source == 5, "The exact pair did not decide");
    present generated;
    generated.offered = candidate::current;
    generated.hold_previous = true;
    std::vector<frame_result> run_of;
    for (std::uint32_t i = 0; i != 5; ++i) {
      generated.now_ms = 10008 + 8 * i;
      run_of.push_back(s.step(generated));
    }
    for (std::uint32_t i = 0; i != max_held_presents; ++i) {
      require(run_of[i].held && run_of[i].source == 5, "A generated Present within the cap did not hold");
    }
    require(!run_of[3].held && run_of[3].hold.cap_reached && run_of[3].hold.kind == hold_kind::generated && run_of[3].detected && run_of[3].gpu.bits == candidate::current && !run_of[3].source, "The Present past the cap did not detect alone");
    require(!run_of[4].held && !run_of[4].hold.cap_reached && run_of[4].hold.kind == hold_kind::none, "A mask detected without a HUD-less pair or an accepted alpha was holdable");
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
      accept_hudless(session);
      sequence s(session, difference_frame);
      present p;
      p.offered = candidate::current | candidate::hudless | candidate::exact;
      p.now_ms = 10000;
      s.step(p);
      p.offered = candidate::current | candidate::hudless;
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
      p.offered = candidate::current | candidate::hudless | candidate::exact;
      p.now_ms = 10200;
      s.step(p);
      p.offered = candidate::current | candidate::hudless;
      p.hold_previous = true;
      p.now_ms = 10216;
      require(s.step(p).hold.kind == hold_kind::generated, "Hold kinds lost their priority");
      check_counters(s, "inexact after exact");
    }
    {
      // An accepted Backbuffer that decided and is missing holds; an
      // accepted, valid channel in the frame decides by itself and nothing is
      // held. Acceptance is never inherited by another candidate.
      alpha_auto_policy session;
      restore(session, stored_of({signatures_in(srgb).of(kind::ui_alpha), signatures_in(srgb).of(kind::backbuffer)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        return backbuffer_alpha(in.bits & candidate::backbuffer ? 200 : 0).words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
      p.now_ms = 10000;
      require(s.step(p).detected && s.temporal.mask_ready, "An accepted Backbuffer decision was not holdable");
      p.offered = candidate::current;
      std::vector<frame_result> missing;
      for (std::uint32_t i = 0; i != 4; ++i) {
        p.now_ms = 10016 + 16 * i;
        missing.push_back(s.step(p));
      }
      for (std::uint32_t i = 0; i != 3; ++i) {
        require(missing[i].held && missing[i].hold.kind == hold_kind::trusted_missing && missing[i].source == 3, "A missing accepted channel did not hold");
      }
      require(missing[3].hold.cap_reached && missing[3].detected && !missing[3].gpu.accepted && !missing[3].source, "The accepted-missing hold was not capped, or the current alpha inherited acceptance");
      p.offered = candidate::backbuffer;
      p.now_ms = 10100;
      s.step(p);
      p.offered = candidate::ui_alpha | candidate::current;
      p.hold_previous = true;
      p.now_ms = 10116;
      const auto &own = s.step(p);
      require(own.detected && own.hold.kind == hold_kind::none && own.gpu.bits == (candidate::ui_alpha | candidate::current) && own.source == 1, "An accepted channel in the frame did not decide by itself");
      check_counters(s, "accepted missing");
    }
  }

  void e33_title_and_load_game() {
    // E33 FG on 2x: the Backbuffer tag comes with real Presents only and no
    // HUD-less image pairs, so generated Presents offer nothing. Before its
    // acceptance (3 samples over 2 s, A1) the inferred Backbuffer decides
    // nothing (S1); afterwards it decides and generated Presents hold its mask
    // (T1, the accepted alpha is missing from them).
    const auto e33 = phased({{14000, recorded_frames({recorded::e33_title, recorded::e33_title_accepted})}, {UINT64_MAX, recorded_frames({recorded::e33_load, recorded::e33_load_accepted})}});
    alpha_auto_policy session;
    sequence s(session, e33);
    fg_pacer pacer(1);
    present real;
    real.offered = candidate::backbuffer;
    std::size_t unaccepted_real = 0, unaccepted_generated = 0, held_generated = 0;
    // The real frame that reads the earning sample adopted its acceptance
    // before the poll, so its own mask decided nothing and is not holdable.
    bool previous_real_accepted = false;
    for (std::uint64_t now = 10000; now < 17000; now += 8) {
      const auto kind = pacer.next(true, 1);
      auto p = fg_present(kind, real, 0u);
      p.now_ms = now;
      const auto &r = s.step(p);
      if (kind != ui_mask::hudless_present::generated_frame) {
        require(r.detected, "A real E33 Present did not detect");
        previous_real_accepted = (r.gpu.accepted & candidate::backbuffer) != 0;
        if (!previous_real_accepted) {
          require(now < 14000 && !r.source && r.decision.none_reason == ui_no_mask::unaccepted, "The unaccepted title Backbuffer decided");
          ++unaccepted_real;
        } else if (now < 14000) {
          require(r.source == 3 && r.covered == 197797, "The accepted title Backbuffer did not decide");
        }
      } else if (previous_real_accepted) {
        require(r.held && r.hold.kind == hold_kind::trusted_missing && r.source == 3, "A generated Present lost the accepted mask");
        ++held_generated;
      } else {
        require(!r.active && !r.source, "A generated Present held a mask decided before acceptance");
        ++unaccepted_generated;
      }
      if (now >= 14100) {
        require(r.flat() && r.source == 3, "Load Game over a hidden scene was not flat with the accepted Backbuffer alpha");
      }
    }
    const auto first = s.samples.front().sample_tick_ms;
    require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[0].accepted == key(kind::backbuffer), "The E33 title did not accept the Backbuffer 2 s into its run");
    require(unaccepted_real && unaccepted_generated && held_generated, "The E33 stream did not cover the Presents before and after acceptance");
    const auto counted = session.counters();
    require(!counted[ui_counter::trust_revoked_full], "Load Game revoked the Backbuffer without an exact pair");
    // Load Game is a whole-frame alpha from the accepted Backbuffer, pinned
    // flat (P1) over a hidden scene: the counters measure it hidden, never
    // visible.
    require(counted[ui_counter::full_alpha] > 0 && counted[ui_counter::full_alpha_d_hidden] > 0 && !counted[ui_counter::full_alpha_d_visible], "The counters did not record Load Game's whole-frame alpha over a hidden scene");
    check_counters(s, "E33 title and Load Game");

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

  // ---------------------------------------------------------------- staleness

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
      // A change of the deciding inputs (here the accepted candidates)
      // discards the pending sample.
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
      for (std::uint32_t i = 1; i != 5; ++i) {
        p.now_ms = 10000 + 16 * i;
        s.step(p);
      }
      require(s.discards == 1 && s.samples.empty(), "A sample from other deciding inputs was read");
    }
  }

  // An accepted alpha that decides first in draw order keys the status sample
  // alone (detection_decision_key): candidates that frame generation offers on
  // real Presents only, accepted ones included, never discard a sample.
  void accepted_alpha_keys_samples_alone() {
    const auto signatures = signatures_in(srgb);
    {
      // The Witcher 3 or Resident Evil Requiem with FG 2x after an FG-off
      // session accepted the exact HUD-less pair: real Presents offer the
      // accepted UIAlpha and the (inexact) pair, generated ones UIAlpha alone.
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
        auto p = fg_present(kind, real, candidate::ui_alpha);
        p.now_ms = now;
        const auto &r = s.step(p);
        generated += kind == ui_mask::hudless_present::generated_frame ? 1 : 0;
        require(r.detected && r.source == 1 && r.covered == 50, "The accepted UIAlpha did not decide every Present");
      }
      require(generated > 100 && !s.discards && s.samples.size() >= 25, "An intermittent accepted HUD-less pair discarded samples beside the accepted UIAlpha (" + std::to_string(s.discards) + " discarded, " + std::to_string(s.samples.size()) + " read)");
      require(s.frames.back().consumed.state != alpha_auto_state::collecting, "The status stayed checking");
      check_counters(s, "accepted UIAlpha beside an accepted HUD-less pair");
    }
    {
      // An accepted offscreen layer beside an accepted Backbuffer that frame
      // generation tags on real Presents only (no HUD-less pair, so nothing
      // holds).
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
        auto p = fg_present(pacer.next(true, 1), real, candidate::layer);
        p.now_ms = now;
        const auto &r = s.step(p);
        require(r.detected && r.source == ui_detection::source_layer, "The accepted layer did not decide every Present");
      }
      require(!s.discards && s.samples.size() >= 25, "An intermittent accepted Backbuffer discarded samples beside the accepted layer");
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
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_alpha_d_visible] > 0 && c[ui_counter::full_alpha_d_invalid] == 1 && !c[ui_counter::full_alpha_d_hidden] && !c[ui_counter::trusted_full], "The counters did not record the accepted full layer over a visible scene");
      require(!c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented] && session.stored() == key(kind::ui_layer), "The accepted sign wheel layer was revoked");
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
        auto p = fg_present(kind, real, candidate::current);
        p.now_ms = now;
        const auto &r = s.step(p);
        require(r.held == (kind == ui_mask::hudless_present::generated_frame), "A W3 FG Present was held wrongly");
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
      require(!c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented], "The accepted sign wheel UIAlpha was revoked");
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
      // or invalid (nothing decides, counted as presented_blocked while the
      // presented alpha is accepted).
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
      std::size_t rejected_frames = 0;
      for (const auto &f : s.frames) {
        const bool rejected = (f.now_ms / 500) % 2 == 0;
        require(f.detected && f.source != 4, "Presented alpha decided beside an accepted tag");
        if (rejected) {
          // presented_blocked while the presented alpha is accepted; once
          // disagreement revokes it (A2), trusted_invalid.
          require(!f.source && f.decision.none_reason == ((f.gpu.accepted & candidate::current) ? ui_no_mask::presented_blocked : ui_no_mask::trusted_invalid), "A rejected-tag frame did not decide nothing for its reason");
          ++rejected_frames;
        } else {
          require(f.source == 2 && f.covered == 15, "The valid accepted tag did not decide");
        }
      }
      const auto c = session.counters();
      require(rejected_frames && c[ui_counter::none + ui_no_mask::presented_blocked] > 0 && !c[ui_counter::presented_over_dedicated], "The declared block was not counted");
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
      require(session.stored() == key(kind::ui_layer) && s.trust.size() == 1 && !c[ui_counter::trust_revoked_full] && !c[ui_counter::trust_revoked_presented], "The bloom-like layer was not accepted once and kept");
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
    {"A2/P1 revoke", acceptance_is_revoked_by_contradiction},
    {"A3 provisional restore and legacy discard", restored_acceptance_is_provisional},
    {"E1/S1/A1 separate slots", separate_slots_never_accept_the_opaque_tag},
    {"A1 signature key", acceptance_is_keyed_by_signature},
    {"A1/S1 declared one-sample acceptance", declared_source_is_accepted_by_one_sample},
    {"A1/S1/H1 Hogwarts exact HUD-less", hudless_pair_is_accepted_by_one_exact_sample},
    {"S2 manual On", manual_on_is_a_session_override},
    {"T1 frame generation holds", generated_presents_hold_per_multiplier},
    {"T1/E2 frame generation multiplier drift", generated_presents_under_multiplier_drift},
    {"T1 hold cap", holds_are_capped},
    {"T1 inexact-after-exact and accepted-missing holds", inexact_after_exact_and_trusted_missing_holds},
    {"A1/S1/T1/P1 E33 title and Load Game", e33_title_and_load_game},
    {"sample staleness", samples_go_stale_and_are_discarded},
    {"T1 accepted alpha keys samples alone", accepted_alpha_keys_samples_alone},
    {"H1 W3 layer route", hidden_scene_layer_route},
    {"H1 shadow and scope clears", hidden_scene_shadow_and_clears},
    {"P1/A1/S1 recorded menus", recorded_menus_after_acceptance},
    {"A1/S1 adversaries", adversaries},
    {"S1 invariants", s1_invariants},
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

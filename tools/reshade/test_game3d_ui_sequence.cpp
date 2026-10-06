// SPDX-License-Identifier: GPL-3.0-only
// Sequence replay of live automatic UI detection's temporal rules
// (docs/reshade-sbs.md, UI protection and UI decision framework). Per-frame
// decision streams drive the production state machines without a GPU: the
// game session's acceptance ledger (alpha_auto_policy, the offscreen
// layer's pre-UI proof of H1 (d) included), the renderer's T1
// arbitration and status samples (ui_temporal::detection_state), the
// hidden-scene guard that holds the CPU's verdicts of D and refutes claims
// per signature (scene_guard::state, M5), the sample decode and counter
// commit (ui_temporal::decode_detection_sample, sample_counters), Present
// counting under frame generation (ui_mask::generated_without_input: inputs
// tagged on real frames only, a HUD-less image offered on every Present it
// pairs, inexact by ui_mask::pair_hudless_present) and the exact counters
// (ui_counters). A stream's GPU input is what one detection counts:
// decision texels 0-11 that ui_detection_replay --verbose recorded on
// labelled dumps with the fix-1 shader, or synthetic counts. Every decision
// is ui_selection::decide, the C++ mirror of SunshineUIDetectionReduceCS
// (the T1 hold store and the H1 override included) that
// test_game3d_ui_selection_contract proves equal to the shader's reduce;
// every recorded decision must equal it.
//
// It asserts the rules through stage S2b and fix 1 (the pre-UI proof by
// pixels) strictly. An outcome that a later stage of the UI decision
// framework (docs/reshade-sbs.md, UI decision framework: stages S0-S6;
// rules E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1, F1) changes prints
// "KNOWN_TODAY <stage> <rule>: <text>" and does not fail; that stage turns
// it into a strict assertion. None remains: S3 (frame identity), which would
// have changed the last two, was removed with fix 2 (H2, still screens). Those
// two are known limits of Present counting under a drifting frame-generation
// multiplier and print "KNOWN_LIMIT <rule>: <text>", which does not fail
// either. Fix 3 and fix 4 (the pre-UI change set and pin only UI) were
// removed by user decision with their groups. Outcomes the rules already call correct, such as an accepted source
// pinning a whole-frame alpha flat over a visible scene (P1, the opacity
// ruling), are asserted strictly, and so is the risk the approved pre-UI
// proof accepts (h1_dark_gameplay_never_flat, a proven pre-fog buffer).
//
// Informational, not in ctest: --log <ReShade.log>... replays the logged
// "Sunshine UI protection" samples through alpha_auto_policy and prints the
// predicted acceptance transitions beside the logged ones.
#include "game3d_alpha_auto.h"
#include "game3d_scene_guard.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_mask.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_temporal.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

  // A known limit of the design that no roadmap stage changes (Present
  // counting under a drifting frame-generation multiplier).
  std::uint32_t known_limit_count = 0;

  void known_limit(const char *rule, const std::string &text) {
    std::printf("KNOWN_LIMIT %s: %s\n", rule, text.c_str());
    ++known_limit_count;
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

  // Decision texels 0-16 as the renderer reads them back (selection revision
  // 10; texels 12-15 are reserved zeros).
  constexpr std::size_t texel_words = 4 * ui_detection::decision_texels;
  using texels = std::array<std::uint32_t, texel_words>;
  // Texels 5 and 6: the hidden-scene evidence of the presented frame and of
  // the pre-UI scene image, written only by the evidence passes.
  constexpr std::size_t scene_texels_begin = 4 * 5, scene_texels_end = 4 * ui_detection::scene_decision_texels;

  // A recording of 48 words, or of 44 (texels 0-10, recorded before
  // selection revision 4: texel 11 reads zero).
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
    if ((count != 4 * ui_detection::pre_ui_decision_texels && count != 4 * ui_detection::h1_decision_texels) || at != end) {
      throw std::runtime_error("Recorded texels are not 44 or 48 words");
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

  // Texel 6, the pre-UI scene image's evidence as the evidence pass writes
  // it: n, D, valid | ran, and the image (the HUD-less image when offered,
  // else the offscreen UI layer's colour); all zero without an image. The
  // CPU reads the image's verdict from D.
  void pre_ui_evidence(texels &t, std::uint32_t offered, std::uint32_t n, float d, bool valid) {
    const auto image = ui_selection::pre_ui_image_of(offered);
    t[word::pre_ui_scene_n] = image ? n : 0u;
    t[word::pre_ui_scene_d] = image ? float_bits(d) : 0u;
    t[word::pre_ui_scene_state] = image ? (valid ? 1u : 0u) | 2u : 0u;
    t[word::pre_ui_scene_image] = image;
  }

  // What one frame's detection reads: the frame's own inputs, the per-frame
  // bits and the hold store (game3d_renderer.cpp, detect_ui's b2 constants:
  // the candidates, the layer's stored flags with the per-frame bits of T1,
  // the scene guard and the depth, the accepted candidates; the T1 hold store
  // at u5, as the previous detection wrote it).
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
  // the one-way judgment and pre-UI counts only on a status sample
  // (per_frame_sample; words 32 and 36, the layer's before revision 10, are
  // reserved zeros), none of a declared tag pushed unaligned with the exact
  // pair (per_frame_unaligned_shift), texel 0 the applied decision, the
  // refused candidate and the frame reason (F1, with the T1 reused bit), and
  // texel 10's informative claims and h1 word (the S1 winner, and whether H1
  // overrode it).
  texels decision_words(texels t, const gpu_inputs &in) {
    t[word::strong] = t[word::contradicted] = 0;
    const std::uint32_t unaligned = in.per_frame >> ui_detection::per_frame_unaligned_shift;
    if (unaligned & candidate::ui_alpha) {
      t[word::strong_ui_alpha] = t[word::contradicted_ui_alpha] = 0;
    }
    if (unaligned & candidate::ui_color) {
      t[word::strong_ui_color] = t[word::contradicted_ui_color] = 0;
    }
    if (!(in.per_frame & ui_detection::per_frame_sample)) {
      for (const auto w : {word::strong_ui_alpha, word::strong_ui_color, word::strong_backbuffer, word::strong_current,
             word::contradicted_ui_alpha, word::contradicted_ui_color, word::contradicted_backbuffer,
             word::contradicted_current, word::pre_ui_match, word::pre_ui_image_lit}) {
        t[w] = 0;
      }
    }
    const auto d = decide(t, in);
    t[word::source] = d.source;
    t[word::covered] = d.covered;
    t[word::candidates] = in.bits;
    t[word::accepted] = in.accepted;
    t[word::valid_bits] = d.valid_bits;
    t[word::refused] = d.refused;
    t[word::frame_reason] = ui_selection::frame_reason_word(d);
    t[word::claims] = d.claims;
    t[word::h1] = ui_selection::h1_word(d);
    return t;
  }

  // One recorded detection and the per-frame hold bits it was replayed with.
  struct recording {
    std::string_view words;
    std::uint32_t per_frame {};
  };

  // Recorded decision texels: ui_detection_replay --verbose with the fix-1
  // shader on E:/ApolloDev/sbs_dump (selection revision 4, nothing bound at
  // the hold store). Words 4 and 17 hold the candidate bits and the accepted
  // candidates the frame was decided with; words 20-27 the hidden-scene
  // evidence of the presented frame and of the pre-UI scene image (texel 6:
  // n, D, valid | ran, image); words 32-39 the one-way judgment counts, the
  // refused candidate and the frame reason; words 40-43 the opaque Backbuffer
  // and current pixels, the informative claims and the h1 word; words 44-47
  // the offscreen UI layer against the presented frame (texel 11: matching
  // pixels, lit layer pixels, lit presented pixels and lit presented pixels
  // that differ; zero without a layer). Each names its dump and the replay
  // case of the same inputs in ui_detection_cases.json; every recording was
  // replayed alone from scratchpad fix1/p2/cases_p2.json (the S2b recordings,
  // plus the proven and unproven Stellar Blade SDR variants and FG-on
  // gameplay) by the fix-1 binary into fix1/p2/replay_p2_verbose.txt. Against
  // the S2b recordings only the claims word changed, where an unproven
  // V1-invalid layer had claimed the pre-UI image (0x80).
  namespace recorded {
    // The scene guard's per-frame bits (H1): a held hidden verdict, its
    // samples reading the pre-UI scene image visible, and the offered
    // layer's signature proven the pre-UI scene image (the ledger's answer).
    constexpr std::uint32_t scene_hidden = ui_detection::per_frame_scene_hidden;
    constexpr std::uint32_t pre_ui_visible = ui_detection::per_frame_pre_ui_visible;
    constexpr std::uint32_t pre_ui_proven = ui_detection::per_frame_pre_ui_proven;
    // The Witcher 3 Remastered, FG off, graphics settings over a hidden scene
    // (game3d_46312_198402901901355): a full opaque offscreen layer (0x40)
    // beside current alpha, unaccepted, without a CPU hold and held ("W3
    // graphics settings, layer hidden-scene hold": H1 claim (b), the cleared
    // layer opaque-full).
    constexpr recording w3_settings {"0,0,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,1565,3166123306,3,2,3686400,0,3686400,72,0,0,3686400,64,0,0,0,5,0,3686400,64,0,3038675,145526,1280396,647725"};
    constexpr recording w3_settings_held {"8,3686400,3686400,59,72,2059821,1626579,0,0,0,0,3686400,0,0,0,0,0,0,0,0,1565,3165951781,7,3596,1565,3166123306,3,2,3686400,0,3686400,72,0,0,3686400,0,0,0,0,255,0,3686400,64,256,3038675,145526,1280396,647725", scene_hidden};
    // W3 FG off, notice board (game3d_46312_198402901901363): layer 12.48%,
    // unaccepted and accepted ("W3 notice board FG off, trusted layer, hold").
    constexpr recording w3_notice {"0,0,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,0,0,0,2029,1049969278,15,8116,2029,977390662,3,2,460125,0,100337,72,0,0,3686400,64,0,0,0,8,0,3507790,0,0,61929,50899,3631720,3624471"};
    constexpr recording w3_notice_accepted {"10,460125,3686400,0,72,3643197,43203,0,0,0,0,3686400,0,0,0,0,0,64,0,0,2029,1049969278,15,8116,2029,977390662,3,2,460125,0,100337,72,0,0,3686400,0,0,0,0,255,0,3507790,0,10,61929,50899,3631720,3624471"};
    // W3 FG off, sign wheel over a visible scene (game3d_46312_198402901901353):
    // a full layer, unaccepted and accepted ("W3 sign wheel FG off, trusted
    // layer, hold").
    constexpr recording w3_wheel {"0,0,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,0,0,0,564,1057143089,15,2256,564,3173571040,3,2,3686400,0,145726,72,0,0,3686400,64,0,0,0,3,0,236987,0,0,91823,121543,3626905,3594577"};
    constexpr recording w3_wheel_accepted {"10,3686400,3686400,0,72,3686120,280,0,0,0,0,3686400,0,0,0,0,0,64,0,0,564,1057143089,15,2256,564,3173571040,3,2,3686400,0,145726,72,0,0,3686400,0,0,0,0,255,0,236987,0,10,91823,121543,3626905,3594577"};
    // W3 FG on, HUD (game3d_46312_198402901901357): UIAlpha 5.23% with an
    // inexact HUD-less pair, unaccepted ("W3 HUD FG on, not yet accepted
    // UIAlpha, hold") and accepted ("W3 HUD FG on, trusted UIAlpha, hold").
    constexpr recording w3_hud_fg {"0,0,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,0,103614,0,472,1056573614,15,1888,472,1056964608,3,1,0,0,0,9,0,0,3686400,1,0,0,0,8,0,3686400,0,0,0,0,0,0"};
    constexpr recording w3_hud_fg_accepted {"1,192930,3686400,1,25,2124039,1186939,0,192930,0,0,3686400,0,0,0,0,3686400,1,103614,0,472,1056573614,15,1888,472,1056964608,3,1,0,0,0,9,0,0,3686400,0,0,0,0,255,0,3686400,0,1,0,0,0,0"};
    // W3 FG on, sign wheel (game3d_46312_198402901901359): a full UIAlpha,
    // unaccepted and accepted ("W3 sign wheel FG on, ... UIAlpha, hold").
    constexpr recording w3_wheel_fg {"0,0,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,0,149627,0,462,1056020436,15,1848,462,1057709052,3,1,0,0,0,9,0,0,3686400,16,0,0,0,4,0,3686400,128,0,0,0,0,0"};
    constexpr recording w3_wheel_fg_accepted {"1,3686400,3686400,0,25,3686398,0,0,3686400,0,0,3686400,0,0,0,0,3686400,1,149627,0,462,1056020436,15,1848,462,1057709052,3,1,0,0,0,9,0,0,3686400,0,0,0,0,255,0,3686400,128,1,0,0,0,0"};
    // Clair Obscur: Expedition 33, FG on, title over a visible scene
    // (game3d_31636_135749029986373): Backbuffer alpha 2.38%, unaccepted and
    // accepted ("E33 title FG on, untrusted/trusted backbuffer, hold").
    constexpr recording e33_title {"0,0,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,0,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4,0,178867,178867,4,0,0,0,8,155808,155808,0,0,0,0,0,0"};
    constexpr recording e33_title_accepted {"3,197797,8294400,19,4,4139412,3333766,0,0,0,197797,197797,0,0,0,0,0,4,0,0,1680,1061468492,15,6720,0,0,0,0,0,0,0,4,0,178867,178867,0,0,0,0,255,155808,155808,0,3,0,0,0,0"};
    // E33 FG on, Load Game over a hidden scene (game3d_59540_257918763574026):
    // a full Backbuffer alpha ("E33 load game FG on, untrusted / trusted
    // backbuffer").
    constexpr recording e33_load {"0,0,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,0,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4,0,8294400,8294400,4,0,0,0,3,8294382,8294382,0,0,0,0,0,0"};
    constexpr recording e33_load_accepted {"3,8294400,8294400,0,4,8138474,84306,0,0,0,8294400,8294400,0,0,0,0,0,4,0,0,1753,1000268529,7,6942,0,0,0,0,0,0,0,4,0,8294400,8294400,0,0,0,0,255,8294382,8294382,4,3,0,0,0,0"};
    // Resident Evil Requiem, dark room (game3d_52696_225539427440975): the
    // offscreen layer 0.18% beside a full presented alpha and an inexact pair,
    // unaccepted ("RE9 dark room, untrusted layer, inexact pair, hold") and
    // accepted.
    constexpr recording re9_room {"0,0,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,386,1058855305,15,1544,386,1057529644,3,1,14611,0,3540,72,0,0,8294400,64,0,0,0,8,0,8294400,0,0,2911709,7417,5385198,5382691"};
    constexpr recording re9_room_accepted {"10,14611,8294400,31,88,2568733,4352259,0,0,0,0,8294400,0,0,0,0,8294400,64,0,0,386,1058855305,15,1544,386,1057529644,3,1,14611,0,3540,72,0,0,8294400,0,0,0,0,255,0,8294400,0,10,2911709,7417,5385198,5382691"};
    // Stellar Blade in SDR, FG on (game3d_69460_296226962143478): the tagged
    // UIColorAndAlpha (0.26%), the scene image in the cleared UI layer (V1
    // invalid; it equals the presented frame on 99.32% of pixels and is lit
    // on 90.4%) and the Backbuffer alpha; nothing accepted, the tag, the tag
    // and the Backbuffer ("SB SDR FG on, accepted tag beside the scene layer
    // and the Backbuffer"), and the Backbuffer of a Present without the tag.
    constexpr recording sb_sdr {"0,0,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,0,0,6885,1088,1052816565,15,4352,1088,1052754884,3,2,0,8261496,0,6,0,15296,8294400,64,0,0,0,0,6885,8294400,0,0,8238176,7496745,7474537,54650"};
    constexpr recording sb_sdr_tag {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,2,0,6885,1088,1052816565,15,4352,1088,1052754884,3,2,0,8261496,0,6,0,15296,8294400,0,0,0,0,255,6885,8294400,0,2,8238176,7496745,7474537,54650"};
    constexpr recording sb_sdr_tag_backbuffer {"2,21283,8294400,0,70,8294319,41,0,0,21283,21283,8294400,0,0,0,0,0,6,0,6885,1088,1052816565,15,4352,1088,1052754884,3,2,0,8261496,0,6,0,15296,8294400,0,0,0,0,255,6885,8294400,0,2,8238176,7496745,7474537,54650"};
    constexpr recording sb_sdr_backbuffer {"3,21283,8294400,0,68,8294319,41,0,0,0,21283,8294400,0,0,0,0,0,4,0,0,1088,1052816565,15,4352,1088,1052754884,3,2,0,8261496,0,4,0,15296,8294400,0,0,0,0,255,6885,8294400,0,3,8238176,7496745,7474537,54650"};
    // Stellar Blade in HDR, FG on (game3d_69460_296226962143474): the opaque
    // final-image tag, the offscreen layer 0.18%, a full Backbuffer alpha and
    // an exact pair; nothing accepted, the layer accepted ("SB HDR FG on HUD,
    // unaccepted opaque tag beside an accepted layer, exact pair"), and the
    // tag without the layer ("SB HDR FG on HUD, opaque tag, backbuffer and
    // exact pair"). Manual On accepts every offered candidate, with and without
    // a source filter that leaves the tag out.
    constexpr recording sb_hdr {"0,0,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,1,14519,0,206,70,0,8294400,8294400,64,0,5561801,5561801,8,8294400,8294400,0,0,129958,6435,8164495,8164442"};
    constexpr recording sb_hdr_layer {"10,14519,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,64,0,8294400,538,1057510335,15,2152,538,1057588296,3,1,14519,0,206,70,0,8294400,8294400,0,0,5561801,5561801,255,8294400,8294400,0,10,129958,6435,8164495,8164442"};
    constexpr recording sb_hdr_tag_only {"0,0,8294400,3,54,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,0,0,8294400,538,1057510335,15,2152,538,1057588296,3,1,0,0,0,6,0,8294400,8294400,16,0,5561801,5561801,4,8294400,8294400,0,0,0,0,0,0"};
    constexpr recording sb_hdr_manual {"2,8294400,8294400,3,118,1996036,5562279,0,0,8294400,8294400,8294400,0,0,0,0,8258352,86,0,8294400,538,1057510335,15,2152,538,1057588296,3,1,14519,0,206,70,0,8294400,8294400,0,0,5561801,5561801,255,8294400,8294400,2,2,129958,6435,8164495,8164442"};
    constexpr recording sb_hdr_manual_filtered {"10,14519,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,84,0,0,538,1057510335,15,2152,538,1057588296,3,1,14519,0,206,68,0,8294400,8294400,0,0,5561801,5561801,255,8294400,8294400,4,10,129958,6435,8164495,8164442"};
    constexpr recording sb_hdr_filtered {"0,0,8294400,3,116,1996036,5562279,0,0,0,8294400,8294400,0,0,0,0,8258352,0,0,0,538,1057510335,15,2152,538,1057588296,3,1,14519,0,206,68,0,8294400,8294400,64,0,5561801,5561801,8,8294400,8294400,0,0,129958,6435,8164495,8164442"};
    // Hogwarts Legacy, FG off, title screen (game3d_50196_216992971069363): an
    // exact HUD-less pair differing almost everywhere over a hidden scene;
    // unaccepted ("HL title screen FG off, HUD-less not yet accepted"), under
    // H1 (the held hidden verdict and pre-UI image; the unaccepted exact full
    // change set is claim (c)), and accepted ("HL title screen FG off").
    constexpr recording hl_title {"0,0,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,1,0,0,0,24,0,0,8294400,16,0,0,1,5,0,8294400,144,0,0,0,0,0"};
    constexpr recording hl_title_held {"8,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,0,0,0,511,1025396520,7,2041,511,1057900324,3,1,0,0,0,24,0,0,8294400,0,0,0,1,255,0,8294400,144,256,0,0,0,0", scene_hidden | pre_ui_visible};
    constexpr recording hl_title_accepted {"6,8294400,8294400,0,56,8294395,1,0,0,0,0,8294400,0,0,0,0,8294400,16,0,0,511,1025396520,7,2041,511,1057900324,3,1,0,0,0,24,0,0,8294400,0,0,0,1,255,0,8294400,144,6,0,0,0,0"};
    // Hogwarts Legacy, FG off, gameplay HUD (game3d_40404_173061335442886):
    // an exact pair whose change set is the HUD (3.98%); unaccepted ("HL
    // gameplay HUD FG off (b), HUD-less not yet accepted") and accepted ("HL
    // gameplay HUD FG off (b)").
    constexpr recording hl_hud {"0,0,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,0,0,0,397,1061190608,15,1588,397,1061190608,3,1,0,0,0,24,0,0,8294400,16,0,0,7961885,8,0,8294400,0,0,0,0,0,0"};
    constexpr recording hl_hud_accepted {"5,329969,8294400,224,56,329969,7962406,0,0,0,0,8294400,0,0,0,0,8293872,16,0,0,397,1061190608,15,1588,397,1061190608,3,1,0,0,0,24,0,0,8294400,0,0,0,7961885,255,0,8294400,0,5,0,0,0,0"};
    // Stellar Blade in SDR, the settings menu with frame generation set to 2x
    // but suspended by the game while a menu is open, so no Streamline tag
    // (game3d_50264_218658377782962, "SB SDR settings FG suspended, no hold"
    // and "..., H1 pre-UI layer, holds"): the cleared output target holds the
    // pre-UI scene image (BGRA8, alpha 0: V1-invalid; claim (d) once its
    // signature is proven) beside an opaque current alpha. The presented
    // frame reads hidden (D 0.031) and the pre-UI image visible (D 0.588);
    // the layer equals the presented frame on 35.28% of pixels (the menu).
    // Unbound, and under H1 (both held bits with the proof).
    constexpr recording sb_sdr_menu {"0,0,8294400,0,72,8238793,1209,0,0,0,0,8294400,0,0,0,0,0,0,0,0,914,1023465244,7,3655,914,1058442251,3,2,0,4684873,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,2925933,2157057,5638262,4930313"};
    constexpr recording sb_sdr_menu_held {"8,8294400,8294400,0,72,8238793,1209,0,0,0,0,8294400,0,0,0,0,0,0,0,0,914,1023465244,7,3655,914,1058442251,3,2,0,4684873,0,8,0,0,8294400,0,0,0,0,255,0,8294400,128,256,2925933,2157057,5638262,4930313", scene_hidden | pre_ui_visible | pre_ui_proven};
    // The same menu with the layer proven but no hold ("..., H1 measured,
    // layer proven" before its sample): claim (d), which acts only under the
    // held verdicts; and both held bits without the proof ("..., holds, layer
    // unproven"): no claim, so nothing acts.
    constexpr recording sb_sdr_menu_proven {"0,0,8294400,0,72,8238793,1209,0,0,0,0,8294400,0,0,0,0,0,0,0,0,914,1023465244,7,3655,914,1058442251,3,2,0,4684873,0,8,0,0,8294400,64,0,0,0,0,0,8294400,128,0,2925933,2157057,5638262,4930313", pre_ui_proven};
    constexpr recording sb_sdr_menu_held_unproven {"0,0,8294400,0,72,8238793,1209,0,0,0,0,8294400,0,0,0,0,0,0,0,0,914,1023465244,7,3655,914,1058442251,3,2,0,4684873,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,2925933,2157057,5638262,4930313", scene_hidden | pre_ui_visible};
    // Stellar Blade SDR gameplay (game3d_69460_296226962143470, "SB SDR FG
    // off gameplay, both images visible, measured"): the same inputs, both
    // images visible (presented 0.515, pre-UI layer 0.518); the layer equals
    // the presented frame on 99.79% of pixels (lit 88.9%) despite its copy
    // being one frame late. Unproven and proven ("..., layer proven,
    // measured").
    constexpr recording sb_sdr_play {"0,0,8294400,0,72,8289048,500,0,0,0,0,8294400,0,0,0,0,0,0,0,0,546,1057210428,15,2184,546,1057271883,3,2,0,8221955,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,8277322,7375372,7357573,17048"};
    constexpr recording sb_sdr_play_proven {"0,0,8294400,0,72,8289048,500,0,0,0,0,8294400,0,0,0,0,0,0,0,0,546,1057210428,15,2184,546,1057271883,3,2,0,8221955,0,8,0,0,8294400,64,0,0,0,0,0,8294400,128,0,8277322,7375372,7357573,17048", pre_ui_proven};
    // Stellar Blade SDR, FG on without the tag (game3d_69460_296226962143476,
    // "SB SDR FG on gameplay with HUD-less, measured"): the cleared target,
    // current alpha and the HUD-less tag (an inexact pair whose change set is
    // the HUD). The HUD-less image is the offer's pre-UI image, so nothing
    // claims, proven or not; the layer equals the presented frame on 99.82%
    // of pixels (lit 88.8%). The same offer in dump 296226962143478: 99.32%
    // (lit 90.4%).
    constexpr recording sb_sdr_fg_play {"0,0,8294400,0,88,710096,5413855,0,0,0,0,8294400,0,0,0,0,7349439,0,0,0,527,1057028279,15,2108,527,1057155620,3,1,0,8224538,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,8279361,7369429,7374379,15014"};
    constexpr recording sb_sdr_fg_play_proven {"0,0,8294400,0,88,710096,5413855,0,0,0,0,8294400,0,0,0,0,7349439,0,0,0,527,1057028279,15,2108,527,1057155620,3,1,0,8224538,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,8279361,7369429,7374379,15014", pre_ui_proven};
    constexpr recording sb_sdr_fg_play_478 {"0,0,8294400,0,88,1227340,4761669,0,0,0,0,8294400,0,0,0,0,7461445,0,0,0,1088,1052816565,15,4352,1088,1052724043,3,1,0,8261496,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,8238176,7496745,7474537,54650"};
    constexpr recording sb_sdr_fg_play_478_proven {"0,0,8294400,0,88,1227340,4761669,0,0,0,0,8294400,0,0,0,0,7461445,0,0,0,1088,1052816565,15,4352,1088,1052724043,3,1,0,8261496,0,8,0,0,8294400,64,0,0,0,0,0,8294400,0,0,8238176,7496745,7474537,54650", pre_ui_proven};
  }  // namespace recorded

  // The recorded counts with exactly these candidate bits and accepted
  // candidates decide by ui_selection::decide under this frame's hold bits; a
  // stream that asks for any other combination fails rather than guess the
  // dump's counts. Every recording must equal decide() under its own hold bits
  // (the scene guard's held verdicts, refuted candidates and the layer's
  // pre-UI proof).
  gpu_model recorded_frames(std::vector<recording> frames) {
    constexpr std::uint32_t holds = ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_refuted_mask | ui_detection::per_frame_pre_ui_proven;
    std::vector<std::pair<texels, std::uint32_t>> parsed;
    for (const auto &frame : frames) {
      const auto t = parse_texels(frame.words);
      const gpu_inputs own {t[word::candidates], 0, t[word::accepted], frame.per_frame};
      const auto d = decide(t, own);
      require(d.source == t[word::source] && d.covered == t[word::covered] && d.valid_bits == t[word::valid_bits] && d.refused == t[word::refused] && ui_selection::frame_reason_word(d) == t[word::frame_reason] && d.claims == t[word::claims] && ui_selection::h1_word(d) == t[word::h1], "The recording for candidates " + hex(own.bits) + " accepted " + hex(own.accepted) + " differs from ui_selection::decide");
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
    // The presented frame's hidden-scene evidence (texel 5).
    bool scene_valid {};
    scene_verdict verdict = scene_verdict::none;
    float d {};
    std::uint32_t n {}, decided_edges {};
    // The pre-UI scene image's (texel 6), written when the offer has one.
    bool pre_ui_valid {};
    float pre_ui_d {};
    // Texel 11: the offscreen UI layer against the presented frame (matching
    // pixels, lit layer pixels), written when a layer is offered (the reduce
    // writes zeros otherwise); .z and .w are reserved zeros.
    std::array<std::uint32_t, 2> pre_ui_counts {};

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

    // Pixels of an alpha kind with alpha of at least 254/255 (H1's
    // opaque-full claims need 99%).
    synthetic &opaque(kind k, std::uint32_t pixels) {
      switch (k) {
        case kind::ui_alpha:
          c.opaque_ui_alpha = pixels;
          break;
        case kind::ui_color:
          c.opaque_ui_color = pixels;
          break;
        case kind::ui_layer:
          c.opaque_layer = pixels;
          break;
        case kind::backbuffer:
          c.opaque_backbuffer = pixels;
          break;
        default:
          c.opaque_current = pixels;
          break;
      }
      return *this;
    }

    // Valid hidden-scene evidence of the presented frame and, when valid_pre_ui,
    // of the pre-UI scene image; each verdict follows from its D.
    synthetic &scene(float presented, float pre_ui, bool valid_pre_ui = true) {
      scene_valid = true;
      verdict = ui_detection::scene_verdict_of(presented);
      d = presented;
      n = 300;
      decided_edges = 600;
      pre_ui_valid = valid_pre_ui;
      pre_ui_d = pre_ui;
      return *this;
    }

    // The layer's pre-UI pixel counts (texel 11, fix 1): the ledger proves
    // its signature from samples matching on 90% of pixels while lit on half.
    synthetic &pre_ui_pixels(std::uint32_t matched, std::uint32_t image_lit) {
      pre_ui_counts = {matched, image_lit};
      return *this;
    }

    // A2, the one-way counts of a judged kind (UIAlpha, the UI color tag,
    // Backbuffer or current alpha; never the layer copy): pixels with alpha of
    // at least 1/2, and those of them where an offered exact pair's HUD-less
    // image is lit and unchanged. The reduce keeps them on status samples only.
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
      require(at != judged.end(), "The layer copy is never judged");
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
      t[word::opaque_backbuffer] = c.opaque_backbuffer;
      t[word::opaque_current] = c.opaque_current;
      t[word::scene_n] = n;
      t[word::scene_d] = float_bits(d);
      t[word::scene_state] = scene_state(scene_valid, verdict);
      t[word::scene_decided] = decided_edges;
      pre_ui_evidence(t, in.bits, n, pre_ui_d, pre_ui_valid);
      if (in.bits & candidate::layer) {
        t[word::pre_ui_match] = pre_ui_counts[0];
        t[word::pre_ui_image_lit] = pre_ui_counts[1];
      }
      t[word::strong_ui_alpha] = c.strong[0];
      t[word::strong_ui_color] = c.strong[1];
      t[word::strong_backbuffer] = c.strong[2];
      t[word::strong_current] = c.strong[3];
      t[word::contradicted_ui_alpha] = c.contradicted[0];
      t[word::contradicted_ui_color] = c.contradicted[1];
      t[word::contradicted_backbuffer] = c.contradicted[2];
      t[word::contradicted_current] = c.contradicted[3];
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
    // generated Present, one that offers nothing within the reported
    // generated count of the last Present that offered a UI tag
    // (ui_mask::generated_without_input; fg_pacer computes it).
    bool hold_previous {};
    // The provider offered the previous render's inexact HUD-less snapshot
    // again (ui_detection_inputs::hudless_reoffer); the renderer pushes
    // per_frame_reoffer when no UIAlpha, UI color or Backbuffer tag comes
    // with it.
    bool hudless_reoffer {};
    // With an exact pair, the offered declared tags (UIAlpha, UI color) not
    // captured in the pair's tag batch (ui_detection_inputs::
    // unaligned_declared); the renderer pushes them from
    // per_frame_unaligned_shift.
    std::uint32_t unaligned_declared {};
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
    // The scene guard's bits pushed with this detection (H1): the held hidden
    // verdict, the held pre-UI image, the refuted offered candidates and the
    // offered layer's pre-UI proof; layer_proven: the session ledger's answer
    // for the offered layer's signature, read after the poll.
    std::uint32_t scene_bits {};
    bool layer_proven {};
    // update_alpha_auto's status for this frame.
    alpha_auto_decision consumed;

    bool flat() const {
      return active && pixels && covered == pixels;
    }
  };

  // One renderer's temporal bookkeeping, calling the production code in the
  // order game3d_renderer.cpp does (game3d_ui_temporal.h, its header
  // comment; game3d_scene_guard.h):
  //   render():           bits and signatures, accepted (session),
  //                       arbitrate(identity, observation, bits), the
  //                       layer's flags and adopt(bits, accepted) when
  //                       t.adopt; then held() (t.hold, held.generated),
  //                       unavailable() (held.none), enter_scope and the
  //                       scene guard's enter_scope(epoch, viewport) +
  //                       poll_detection + the session's pre-UI proof of
  //                       the offered layer's signature (layer_proven) + the
  //                       guard's per_frame(..., layer_proven) + detect_ui +
  //                       detected() (t.detect with bits, or a zero-offer
  //                       real frame flagged accepted_missing; with
  //                       per_frame_reoffer for a tagless HUD-less re-offer
  //                       and the declared tags outside an exact pair's
  //                       batch from per_frame_unaligned_shift), or
  //                       inactive() (the guard keeps its state);
  //   end of render:      the detection fence is signaled (detection_awaiting_signal);
  //   update_alpha_auto:  the latest sample only while status_fresh;
  //   poll_detection:     sample_discarded (scope, stale on arrival),
  //                       decode_detection_sample, latest_key, the guard's
  //                       observe(guard_sample(the decoded sample), the pending actionable,
  //                       the submitted signatures by kind), session.observe
  //                       of the sample's evidence
  //                       with the signatures the sample was submitted
  //                       with, commit_counters (sample_counters with the
  //                       guard's observation);
  //   detect_ui:          the reduce with the hold store (u5) and, except on
  //                       a zero-offer frame, the tiles pass and the 100 ms
  //                       sample cadence (per_frame_sample: only samples
  //                       count the one-way judgment and the pre-UI
  //                       pixels), scene evidence when the guard's
  //                       measure(now, a proven layer that is the offer's
  //                       pre-UI image) says it is actionable, the counter
  //                       and CPU snapshot and the status key.
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
    // The hidden-scene guard (M5), owned by the depth path.
    scene_guard::state guard;
    std::vector<frame_result> frames;
    std::vector<alpha_auto_decision> samples;  // Every sample read, after the guard observed it.
    std::vector<scene_guard::observation> observed;  // The guard's observation of each sample read.
    // The guard's observations of committed samples (the scene counters).
    std::uint64_t scene_entered {}, scene_released {}, scene_refuted {};

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
      r.hold = temporal.arbitrate({p.hold_previous}, observation, bits);
      const std::uint32_t flags = (bits & candidate::layer) ? p.layer_flags : 0u;
      if (r.hold.adopt) {
        temporal.adopt(bits, accepted);
      }
      ++cpu[ui_counter::auto_frames];
      const bool missing = (r.hold.per_frame & ui_detection::per_frame_accepted_missing) != 0;
      if (r.hold.hold) {
        // The detected mask as it is: the last real decision.
        ++cpu[ui_counter::held_generated];
        temporal.held();
        r.active = r.held = true;
      } else if (r.hold.kind == hold_kind::unavailable) {
        ++cpu[ui_counter::held_none];
        temporal.unavailable();
        r.unavailable = true;
      } else if (r.hold.detect && (bits || missing)) {
        r.active = r.detected = true;
        r.grace = !bits;
        temporal.enter_scope(observation);
        guard.enter_scope(observation.epoch, observation.viewport);
        poll(observation, r);
        // H1 (d): the ledger's pre-UI proof of the offered layer's signature,
        // read after the poll, so a sample that just earned it counts at once.
        r.layer_proven = (bits & candidate::layer) && session.pre_ui_proven(p.signatures.of(kind::ui_layer));
        r.scene_bits = guard.per_frame(p.now_ms, bits, p.signatures.by_kind(), r.layer_proven);
        constexpr std::uint32_t tags = candidate::ui_alpha | candidate::ui_color | candidate::backbuffer | candidate::exact;
        const bool reoffer = p.hudless_reoffer && (bits & candidate::hudless) && !(bits & tags);
        const std::uint32_t unaligned = (bits & candidate::exact) ? p.unaligned_declared & bits & (candidate::ui_alpha | candidate::ui_color) : 0u;
        const std::uint32_t per_frame = r.scene_bits | r.hold.per_frame | (!p.depth_current ? ui_detection::per_frame_depth_not_current : 0u) |
          (reoffer ? ui_detection::per_frame_reoffer : 0u) | (unaligned << ui_detection::per_frame_unaligned_shift);
        detect(observation, bits, flags, accepted, p.signatures, per_frame, index, r);
        // T1's invariant: no generated Present detects.
        invariants.generated_detections += p.hold_previous ? 1 : 0;
        temporal.detected(observation);
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
      latest = ui_temporal::decode_detection_sample(pending_texels.data(), pending_texels.size(), pending_source.now_ms, submitted);
      temporal.latest_source = pending_source;
      temporal.latest_key = pending_key;
      const auto observation = guard.observe(ui_temporal::guard_sample(latest), pending_actionable, pending_signatures.by_kind());
      session.observe(latest.evidence, latest.pixels, pending_source.now_ms, pending_signatures);
      commit(latest, observation);
      counters_pending = false;
      r.polled = true;
      samples.push_back(latest);
      observed.push_back(observation);
      auto now_accepted = session.stored();
      if (now_accepted != last_accepted) {
        trust.push_back({latest.sample_tick_ms, now_accepted});
      }
      last_accepted = std::move(now_accepted);
    }

    void detect(const alpha_auto_source &observation, std::uint32_t bits, std::uint32_t flags, std::uint32_t accepted, const candidate_signatures &signatures, std::uint32_t per_frame, std::size_t index, frame_result &r) {
      // A status sample (the renderer's detect_ui): at most one real frame
      // with candidates every 100 ms while no sample is pending pushes
      // per_frame_sample, so that only it counts the one-way judgment and
      // the pre-UI pixels.
      const auto now = observation.now_ms;
      const bool sample = !r.grace && !pending && !(last_submit && now >= last_submit && now - last_submit < sample_interval_ms);
      per_frame |= sample ? ui_detection::per_frame_sample : 0u;
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
      invariants.untrusted_inferred += ui_selection::untrusted_inferred(r.decision, accepted) ? 1 : 0;
      invariants.presented_over_dedicated += ui_selection::inferred_over_declared(r.decision, bits, accepted) ? 1 : 0;
      mask_source = r.decision.source;
      mask_covered = r.decision.covered;
      mask_pixels = t[word::pixels];
      if (!sample) {
        return;
      }
      // Depth that is not this frame's measures no evidence.
      const bool proven_image = r.layer_proven && ui_selection::pre_ui_image_of(bits) == ui_detection::pre_ui_image::layer;
      const bool evidence = !(per_frame & ui_detection::per_frame_depth_not_current) && guard.measure(now, proven_image);
      pending_actionable = evidence;
      // The reduce zeroes texels 5 and 6; only the evidence passes write them.
      if (!evidence) {
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
      pending_key = temporal.status_key();
      pending = awaiting = true;
      last_submit = now;
      ++submitted;
      r.submitted = true;
    }

    // commit_counters: ui_temporal::sample_counters of the CPU counts
    // snapshotted at submission, the GPU words copied under its fence and the
    // scene guard's observation of the sample.
    void commit(const alpha_auto_decision &sample, const scene_guard::observation &observation) {
      if (!counters_pending) {
        return;
      }
      const auto delta = ui_temporal::sample_counters(sample, pending_cpu, committed_cpu, pending_words, committed_words, pending_source.now_ms, observation);
      scene_entered += observation.entered ? 1 : 0;
      scene_released += observation.released ? 1 : 0;
      scene_refuted += observation.refuted;
      committed_cpu = pending_cpu;
      committed_words = pending_words;
      committed_frames = pending_frame + 1;
      ++commits;
      session.add_counters(delta);
    }

    ui_counters cpu, pending_cpu, committed_cpu;
    std::array<std::uint32_t, ui_counter_word::count> gpu_words {}, pending_words {}, committed_words {};
    bool pending {}, awaiting {}, counters_pending {};
    // Whether the pending sample's scene evidence is actionable (the guard's
    // measure when it was submitted).
    bool pending_actionable {};
    std::size_t ready_frame {}, pending_frame {};
    std::uint64_t last_submit {}, submitted {};
    std::uint32_t pending_key {};
    alpha_auto_source pending_source;
    candidate_signatures pending_signatures;
    texels pending_texels {};
    // The GPU's state across detections: the statistics rows of the last
    // tiles pass, the T1 hold store (u5) and the detected mask.
    texels statistics {};
    ui_selection::hold_state gpu_hold {};
    std::uint32_t mask_source {}, mask_covered {}, mask_pixels {};
    std::string last_accepted;
  };

  // Whether a sample's evidence passes ran.
  bool measured(const alpha_auto_decision &sample) {
    return sample.evidence.scene.ran;
  }

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
    // The scene group counts the guard's observations of committed samples,
    // and every visible sample that decided H1 released a hold.
    require(totals[ui_counter::scene_entered] == s.scene_entered && totals[ui_counter::scene_released] == s.scene_released && totals[ui_counter::scene_refuted] == s.scene_refuted, what + ": the scene counters differ from the guard's observations");
    require(totals[ui_counter::full_d_visible] <= totals[ui_counter::scene_released], what + ": an H1 decision over a visible scene released no hold");
    require(!totals.decided(7) && !totals.decided(9) && !totals.decided(11) && !totals.decided(12), what + ": a retired source decided");
    // full_alpha_d is reserved: nothing measures an accepted whole-frame
    // decision for itself since the whole-frame diagnostic was removed.
    require(!totals[ui_counter::full_alpha_d_hidden] && !totals[ui_counter::full_alpha_d_ambiguous] &&
      !totals[ui_counter::full_alpha_d_visible] && !totals[ui_counter::full_alpha_d_invalid], what + ": full_alpha_d counted");
  }

  void run(sequence &s, present p, std::uint64_t from, std::uint64_t to, std::uint64_t interval = 16) {
    for (auto now = from; now < to; now += interval) {
      p.now_ms = now;
      s.step(p);
    }
  }

  // Presents under frame generation, identified the production way. The
  // game's actual cadence decides which Presents are real: `actual`
  // generated frames, then the real one; a cadence change applies from the
  // next Present. A real Present offers the stream's inputs; a generated one
  // offers `generated_bits` only: the game tags its UI inputs (UIAlpha, the
  // UI color tag, the Backbuffer) for real frames only, while an image
  // offered on every Present (a HUD-less image, paired inexactly by Present
  // counting, or the offscreen UI layer copy) is offered there too. As the
  // input provider does (ui_mask::generated_without_input), a Present that
  // offers nothing within the multiplier the provider reports, which can lag
  // or lead the cadence (dynamic or automatic frame generation, a multiplier
  // change), of the last Present that offered a UI tag is generated
  // (hold_previous).
  struct fg_pacer {
    // The first Present is a real frame's.
    explicit fg_pacer(std::uint32_t actual):
        since_real(actual) {}

    std::uint64_t presented {}, input_present {};
    // Generated Presents since the last real one.
    std::uint32_t since_real;

    struct step {
      bool real;  // As the game presented it.
      present p;  // As the provider offers it.
    };

    step next(present real, std::uint32_t generated_bits, std::uint32_t actual, std::uint32_t reported) {
      ++presented;
      const bool is_real = since_real >= actual;
      since_real = is_real ? 0u : since_real + 1;
      if (!is_real) {
        real.offered = generated_bits;
      }
      constexpr std::uint32_t tags = candidate::ui_alpha | candidate::ui_color | candidate::backbuffer;
      if (real.offered & tags) {
        input_present = presented;
      }
      real.hold_previous = !real.offered && ui_mask::generated_without_input(presented, input_present, true, reported);
      return {is_real, real};
    }
  };

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
          // The one-way counts exist on status samples only (selection
          // revision 10), so only those count it contradicted.
          require(f.flat() && f.source == 3 && f.decision.contradicted == f.submitted, "The accepted full claim did not decide flat, or a sample was not counted contradicted");
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
    {
      // The offscreen layer is inferred, but every layer copy is one frame
      // late, not same-sample evidence (E2): it is never a judged kind
      // (selection revision 10 dropped the unused judgment of a same-frame
      // layer), so an accepted UIAlpha's disagreeing coverage leaves it
      // accepted.
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
      p.layer_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 17000);
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() == 1 && s.trust[0].tick == first && s.trust[0].accepted == stored_of({ui_alpha, layer}) && !session.counters()[ui_counter::trust_revoked_declared], "An accepted UIAlpha judged the one-frame-late layer");
      check_counters(s, "late layer beside UIAlpha");
    }
    {
      // Nor is it one-way judged: an accepted full layer over a valid exact
      // pair is not revoked, and no sample counts a strong pixel for it.
      const auto layer = signatures_in(srgb).of(kind::ui_layer);
      alpha_auto_policy session;
      restore(session, layer.key(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        const bool full = in.now_ms >= 13000;
        synthetic f;
        f.alpha(kind::ui_layer, full ? 1000 : 200).change_set(50);
        return f.words(in);
      });
      present p;
      p.offered = candidate::layer | candidate::hudless | candidate::exact;
      p.layer_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 17000);
      for (const auto &sample : s.samples) {
        require(sample.evidence.strong == std::array<std::uint32_t, 4> {} && sample.evidence.contradicted == std::array<std::uint32_t, 4> {}, "The late layer was counted strong or contradicted");
      }
      for (const auto &f : s.frames) {
        require(f.source == ui_detection::source_layer && (f.now_ms < 13000 || f.flat()), "The accepted late layer did not decide");
      }
      require(session.accepts(layer) && !session.counters()[ui_counter::trust_revoked_exact], "The late layer was one-way judged");
      check_counters(s, "late layer");
    }
    {
      // D1 (selection revision 10): a declared alpha is judged one way too.
      // An opaque final image tagged as UI color (Hogwarts Legacy's and
      // Stellar Blade HDR's UIColorAndAlpha) read selective once during a
      // fade is accepted by that first sample (A1); afterwards it covers the
      // frame over a valid exact pair whose HUD-less image shows the scene
      // unchanged under it, so the third contradiction within 2 s revokes it
      // and the pair decides again. Before it was revoked it decided flat.
      const auto tag = signatures_in(srgb).of(kind::ui_color), hudless = signatures_in(srgb).of(kind::hudless);
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        const bool fade = in.now_ms < 10200;
        synthetic f;
        f.alpha(kind::ui_color, fade ? 300u : 1000u).change_set(50).lit(1000);
        if (!fade) {
          f.strong(kind::ui_color, 1000).contradicted(kind::ui_color, 900);
        }
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_color | candidate::hudless | candidate::exact;
      run(s, p, 10000, 13000);
      const auto first = s.samples.front().sample_tick_ms;
      require(s.trust.size() >= 2 && s.trust[0].tick == first && s.trust[0].accepted == stored_of({tag, hudless}), "The fading tag and the exact pair were not accepted by their first sample");
      const auto revoked = third_sample_from(s, 10200);
      require(s.trust[1].tick == revoked && s.trust[1].accepted == hudless.key() && session.counters()[ui_counter::trust_revoked_exact] == 1, "The opaque final image was not revoked by its third one-way contradiction");
      const auto revoked_at = poll_of(s, revoked);
      for (std::size_t i = revoked_at + 1; i != s.frames.size(); ++i) {
        require(s.frames[i].source == 5 && s.frames[i].covered == 50, "The exact pair did not decide after the tag was revoked");
      }
      require(session.counters()[ui_counter::contradicted] > 0, "The contradicted declared winner was not counted");
      check_counters(s, "declared one-way revocation");
    }
    {
      // A declared tag captured in another tag batch than the exact pair
      // (UIAlpha tagged in a list that completes a frame later): moved UI
      // would read as contradicted against the pair's unchanged scene, so
      // the renderer pushes it unaligned and the tiles pass counts no strong
      // pixel of it (per_frame_unaligned_shift). The accepted tag is never
      // judged by such samples, never revoked, and keeps deciding; the same
      // counts from an aligned tag revoke it (the group above).
      const auto alpha = signatures_in(srgb).of(kind::ui_alpha);
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_alpha, 60).change_set(50).lit(1000).strong(kind::ui_alpha, 60).contradicted(kind::ui_alpha, 30);
        return f.words(in);
      });
      present p;
      p.offered = candidate::ui_alpha | candidate::hudless | candidate::exact;
      p.unaligned_declared = candidate::ui_alpha;
      run(s, p, 10000, 13000);
      for (const auto &sample : s.samples) {
        require(!sample.evidence.strong[0] && !sample.evidence.contradicted[0], "An unaligned UIAlpha was counted strong or contradicted");
      }
      for (const auto &f : s.frames) {
        require((f.gpu.per_frame & ui_detection::per_frame_unaligned_mask) == (candidate::ui_alpha << ui_detection::per_frame_unaligned_shift), "The unaligned UIAlpha was not pushed with every detection");
      }
      require(session.accepts(alpha) && !session.counters()[ui_counter::trust_revoked_exact] && !session.counters()[ui_counter::contradicted] && s.frames.back().source == 1 && s.frames.back().covered == 60, "An unaligned UIAlpha was judged one way against the exact pair");
      check_counters(s, "unaligned declared tag");
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
        return backbuffer_alpha(reconfirm ? 200 : 0).words(in);
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
      // invalid pixels, V1) is not testable and its reconfirm clock does not
      // count, so it never lapses during an invalid run however long; it
      // confirms at its first valid
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
    // Presents keep deciding with the same acceptance. The generated
    // Presents offer nothing, so Present counting reads them as generated
    // and each shows the decision of the real frame before it (T1).
    fg_pacer pacer(1);
    for (std::uint64_t now = 16016; now < 17000; now += 8) {
      auto step = pacer.next(back, 0u, 1, 1);
      step.p.now_ms = now;
      const auto &r = s.step(step.p);
      if (!step.real) {
        require(r.held && !r.detected && r.source == 3 && !r.submitted, "A generated Present did not show the accepted Backbuffer's decision");
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
    // block); the invalid layer (V1) never earns or blocks. Its colour equals
    // the presented frame (99.32% of pixels, lit 90.4%), so by the same
    // sample its signature earns the pre-UI proof (H1 (d), fix 1), which is
    // no UI coverage.
    alpha_auto_policy session;
    sequence s(session, recorded_frames({recorded::sb_sdr, recorded::sb_sdr_tag, recorded::sb_sdr_tag_backbuffer, recorded::sb_sdr_backbuffer}));
    present p;
    p.offered = candidate::ui_color | candidate::layer | candidate::backbuffer;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    run(s, p, 10000, 13000);
    const auto tag = p.signatures.of(kind::ui_color), backbuffer = p.signatures.of(kind::backbuffer);
    const auto first = s.samples.front().sample_tick_ms;
    require(s.trust.size() == 2 && s.trust[0].tick == first && s.trust[0].accepted == tag.key(), "The tag was not accepted by its first sample");
    const auto proof = ui_selection::pre_ui_key(p.signatures.of(kind::ui_layer));
    require(s.trust[1].tick == first_sample_from(s, first + alpha_trust_span_ms) && s.trust[1].accepted == stored_of({tag, backbuffer, proof}), "The Backbuffer and the layer's pre-UI proof were not earned 2 s into their runs");
    require(!session.accepts(p.signatures.of(kind::ui_layer)) && session.pre_ui_proven(p.signatures.of(kind::ui_layer)), "The scene image in the UI layer was accepted as UI coverage, or not proven the pre-UI image");
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
      // Title screen before acceptance: the unaccepted full change set never
      // decides 6 (S1 needs an accepted pair), but it is an informative full
      // claim (H1 (c)), as is the HUD-less pre-UI image (d). The first sample
      // shows the claim; the next two measure the presented frame hidden (D
      // 0.039) and the pre-UI image visible, and from the poll of the second
      // the held verdict flattens the title as source 8. Then the gameplay HUD
      // earns the pair and its first sample, visible, releases the hold
      // without refuting anything (the HUD claims nothing); back on the title
      // the accepted pair decides it flat (6), and once the guard holds the
      // hidden verdict again H1 relabels it 8 (selection revision 10: a flat
      // winner pins at weight 1 either way, P1).
      alpha_auto_policy session;
      const auto title = recorded_frames({recorded::hl_title, recorded::hl_title_held, recorded::hl_title_accepted});
      sequence s(session, phased({{12000, title}, {13000, recorded_frames({recorded::hl_hud, recorded::hl_hud_accepted})}, {UINT64_MAX, title}}));
      run(s, p, 10000, 14000);
      std::size_t relabelled = 0;
      require(s.samples.size() > 3 && !measured(s.samples[0]) && s.samples[1].evidence.scene.ran && s.samples[2].sample_tick_ms - s.samples[1].sample_tick_ms <= ui_detection::scene::hold_ms, "The title's first samples did not open the gate and measure");
      const auto entry = poll_of(s, s.samples[2].sample_tick_ms);
      require(s.observed[2].entered && s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible), "The title did not enter both holds at its second measured hidden sample");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        if (f.now_ms < 12000) {
          require(f.source != 6, "The unaccepted title pair decided 6");
          require(f.source == (i >= entry ? 8u : 0u) && f.decision.h1 == (i >= entry), "The title did not go flat (H1) exactly from the poll of its second measured hidden sample");
          require(i >= entry || (f.decision.none_reason == ui_no_mask::gate_no_hold && f.decision.refused == candidate::hudless), "The title before the hold did not name its acting claim");
        } else if (f.now_ms >= 13000) {
          // The label follows the frame's own pushed bits: 8 exactly where
          // the held hidden verdict acts, reused depth included, else 6.
          const bool hidden = (f.gpu.per_frame & ui_detection::per_frame_scene_hidden) != 0u;
          relabelled += hidden ? 1 : 0;
          require(f.flat() && f.decision.s1_source == 6 && f.source == (hidden ? 8u : 6u) && f.decision.h1 == hidden, "The accepted pair did not decide the title flat, or H1 did not relabel it exactly under the held verdict");
        }
      }
      require(relabelled > 0, "The title's return never reached a frame that H1 relabelled 8");
      require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, 12000) && s.trust[0].accepted == hudless, "The gameplay HUD did not accept the pair by its first sample");
      const auto c = session.counters();
      require(c[ui_counter::scene_entered] == 2 && c[ui_counter::scene_released] == 1 && !c[ui_counter::scene_refuted] && c[ui_counter::full_d_hidden] > 0 && !c[ui_counter::full_d_visible], "The title's H1 hold was not counted entered twice and released once");
      check_counters(s, "Hogwarts title");
    }
    {
      // H1 under frame generation with reused depth: each generated Present
      // re-offers the real frame's exact pair, so it detects, and consumes
      // the real frame's depth (per_frame_depth_not_current). Its samples
      // measure no evidence; the hold that the real frames' samples enter
      // acts on every Present, so none of the held window falls back to its
      // own S1 decision (none: the pair is not accepted).
      alpha_auto_policy session;
      sequence s(session, recorded_frames({recorded::hl_title, recorded::hl_title_held, recorded::hl_title_accepted}));
      bool real = true;
      for (std::uint64_t now = 10000; now < 12000; now += 8, real = !real) {
        auto q = p;
        q.now_ms = now;
        q.depth_current = real;
        s.step(q);
      }
      std::size_t entry = s.frames.size(), reused = 0;
      for (std::size_t i = 0; i != s.frames.size() && entry == s.frames.size(); ++i) {
        if (s.frames[i].gpu.per_frame & ui_detection::per_frame_scene_hidden) {
          entry = i;
        }
      }
      require(entry < s.frames.size(), "The title under FG never entered the hidden-scene hold");
      for (std::size_t i = entry; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        reused += (f.gpu.per_frame & ui_detection::per_frame_depth_not_current) ? 1 : 0;
        require((f.gpu.per_frame & ui_detection::per_frame_scene_hidden) && f.flat() && f.source == 8 && f.decision.h1, "A Present of the held window, reused depth included, was not flat as 8");
      }
      require(reused * 3 > (s.frames.size() - entry), "The held window had too few Presents over reused depth");
      std::size_t unmeasured = 0;
      for (const auto &f : s.frames) {
        if (!f.submitted || !(f.gpu.per_frame & ui_detection::per_frame_depth_not_current)) {
          continue;
        }
        for (const auto &sample : s.samples) {
          if (sample.sample_tick_ms == f.now_ms) {
            require(!measured(sample), "A sample over reused depth measured hidden-scene evidence");
            ++unmeasured;
          }
        }
      }
      require(unmeasured > 0, "No sample was taken over reused depth");
      check_counters(s, "Hogwarts title FG reused depth");
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

  // A Backbuffer tagged on real Presents only (Expedition 33 under frame
  // generation), covering 20% and accepted in an earlier session.
  texels backbuffer_frame(const gpu_inputs &in) {
    return backbuffer_alpha(200).words(in);
  }

  void accept_backbuffer(alpha_auto_policy &session) {
    restore(session, key(kind::backbuffer), 1);
  }

  void generated_presents_hold_per_multiplier() {
    // FG 2x, 3x and 4x with the Backbuffer tagged on real Presents only: a
    // generated Present offers nothing, Present counting reads it as
    // generated, and it never detects; it shows the last real decision (T1),
    // held.generated once per generated Present.
    alpha_auto_policy session;
    accept_backbuffer(session);
    sequence s(session, backbuffer_frame);
    fg_pacer pacer(1);
    present real;
    real.offered = candidate::backbuffer;
    std::uint64_t now = 10000;
    std::size_t all_generated = 0;
    for (const std::uint32_t generated : {1u, 2u, 3u, 1u}) {
      std::size_t generated_presents = 0, real_presents = 0, since_real = 0;
      const auto until = now + 1000;
      bool real_last = false;
      // Change the multiplier only after a real Present.
      while (now < until || !real_last) {
        auto step = pacer.next(real, 0u, generated, generated);
        step.p.now_ms = now;
        now += 8;
        real_last = step.real;
        const auto &r = s.step(step.p);
        if (!step.real) {
          ++generated_presents;
          ++since_real;
          require(step.p.hold_previous && r.held && !r.detected && r.hold.kind == hold_kind::generated && r.source == 3 && s.temporal.bits == real.offered && s.temporal.holds == since_real, "A generated Present did not hold the real frame's mask");
        } else {
          require(!step.p.hold_previous && r.detected && r.source == 3 && !s.temporal.holds, "A real Present did not detect afresh");
          require(since_real == generated || s.frames.size() == 1, "The cadence did not give " + std::to_string(generated) + " generated Presents per real one");
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
  // generated and to its real ones: generated Presents that held, that Present
  // counting read as real (beyond the reported count) and that then reused
  // the decision once or had no mask (the T1 grace), and real Presents that
  // held.
  struct fg_tally {
    std::size_t presents {}, real {}, held {}, beyond {}, reused {}, lost {}, extra_holds {};
  };

  // An accepted Backbuffer tagged on real Presents only under frame
  // generation (Expedition 33). cadence(index) gives the game's actual
  // generated frames and the provider's reported ones for the Present with
  // that index; tallies count from Present `from`.
  fg_tally fg_drift(sequence &s, std::size_t count, std::size_t from, const std::function<std::pair<std::uint32_t, std::uint32_t>(std::size_t)> &cadence) {
    fg_pacer pacer(cadence(0).first);
    present real;
    real.offered = candidate::backbuffer;
    fg_tally t;
    std::uint64_t now = 10000;
    for (std::size_t index = 0; index != count; ++index) {
      const auto [actual, reported] = cadence(index);
      auto step = pacer.next(real, 0u, actual, reported);
      step.p.now_ms = now;
      now += 8;
      const auto &r = s.step(step.p);
      if (index < from) {
        continue;
      }
      ++t.presents;
      if (step.real) {
        ++t.real;
        t.extra_holds += r.held ? 1 : 0;
      } else if (r.held) {
        ++t.held;
      } else {
        require(r.detected && r.grace && !r.submitted, "A generated Present beyond the reported count did not run the T1 grace alone");
        ++t.beyond;
        t.reused += r.decision.reused ? 1 : 0;
        t.lost += r.source ? 0 : 1;
      }
    }
    return t;
  }

  void generated_presents_under_multiplier_drift() {
    {
      // 4x, reported as presented: every generated Present holds.
      alpha_auto_policy session;
      accept_backbuffer(session);
      sequence s(session, backbuffer_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 3u};
      });
      require(t.real == 101 && t.held == 300 && !t.beyond && !t.extra_holds, "A 4x cadence reported as 4x did not hold every generated Present");
      check_counters(s, "4x matching cadence");
    }
    {
      // Presented at 4x while the provider still reports 2x (dynamic frame
      // generation): Presents 2 and 3 of each real frame lie beyond the
      // reported count, so counting reads them as real. They offer nothing
      // while the accepted Backbuffer is missing: the first reuses the
      // decision once, the second has no mask (the T1 grace). No Present
      // detects from an interpolated image.
      alpha_auto_policy session;
      accept_backbuffer(session);
      sequence s(session, backbuffer_frame);
      const auto t = fg_drift(s, 401, 0, [](std::size_t) {
        return std::pair {3u, 1u};
      });
      require(t.real == 101 && t.held == 100 && t.beyond == 200 && t.reused == 100 && t.lost == 100 && !t.extra_holds, "The lagging multiplier's grace moved");
      known_limit("T1/E2", "presented at 4x but reported as 2x, " + std::to_string(t.beyond) + " of " + std::to_string(t.held + t.beyond) + " generated Presents lie beyond the reported count: Present counting reads them as real, and the T1 grace reuses the decision for " + std::to_string(t.reused) + " and leaves " + std::to_string(t.lost) + " without a mask");
      check_counters(s, "lagging multiplier");
    }
    {
      // Presented at 2x while the provider reports 4x, after a matching
      // start: a count reported too high holds only Presents without an
      // input of their own, so every generated Present still holds and every
      // real one, offering its tag, detects.
      alpha_auto_policy session;
      accept_backbuffer(session);
      sequence s(session, backbuffer_frame);
      const auto t = fg_drift(s, 300, 100, [](std::size_t index) {
        return std::pair {1u, index < 100 ? 1u : 3u};
      });
      require(t.real == 100 && t.held == 100 && !t.beyond && !t.extra_holds, "The leading multiplier's holds moved");
      check_counters(s, "leading multiplier");
    }
    {
      // 2x to 4x in the middle of a real frame (after its generated Present),
      // reported one real frame late: the two generated Presents beyond the
      // old count fall to the grace (Present 100 reuses once, 101 has no
      // mask), then every generated Present holds again.
      alpha_auto_policy session;
      accept_backbuffer(session);
      sequence s(session, backbuffer_frame);
      constexpr std::size_t change = 100, reported_from = 103;
      const auto t = fg_drift(s, 300, 0, [](std::size_t index) {
        return std::pair {index < change ? 1u : 3u, index < reported_from ? 1u : 3u};
      });
      // Present 102 is the first real one after the change: Presents 100 and 101 are generated.
      require(t.beyond == 2 && s.frames[change].decision.reused && s.frames[change].source == 3 && !s.frames[change + 1].source && s.frames[change + 3].held && !t.extra_holds, "The mid-frame multiplier change's grace moved");
      for (std::size_t i = reported_from; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.held || (f.detected && (f.gpu.bits & candidate::backbuffer)), "A Present after the reported multiplier caught up neither held nor detected its tag");
      }
      // Clearing nothing is right: the FG multiplier is not part of the scope.
      known_limit("T1/E2", "a multiplier change in the middle of a real frame, reported a real frame late, leaves " + std::to_string(t.beyond) + " generated Presents beyond the reported count to the T1 grace (Present counting, not real-frame identity)");
      check_counters(s, "mid-frame multiplier change");
    }
  }

  void generated_presents_hold_without_a_bound() {
    // No multiplier constant and no real-frame id: any number of generated
    // Presents (offering nothing within the reported count) hold the last
    // real decision (T1). Only another identity scope, or a chain ended by a
    // Present without a mask, leaves a generated Present without one.
    alpha_auto_policy session;
    accept_backbuffer(session);
    sequence s(session, backbuffer_frame);
    present real;
    real.offered = candidate::backbuffer;
    real.now_ms = 10000;
    require(s.step(real).source == 3, "The accepted Backbuffer did not decide");
    present generated;
    generated.hold_previous = true;
    for (std::uint32_t i = 0; i != 5; ++i) {
      generated.now_ms = 10008 + 8 * i;
      const auto &r = s.step(generated);
      require(r.held && r.hold.kind == hold_kind::generated && r.source == 3 && s.temporal.holds == i + 1, "A run of generated Presents did not hold");
    }
    // A generated Present in another identity scope (a recreated swapchain)
    // has no decision to show: no mask, and the chain ends.
    auto other = generated;
    other.epoch = 2;
    other.now_ms = 10048;
    const auto &third = s.step(other);
    require(third.unavailable && !third.active && !third.source && !s.temporal.have_decision, "A generated Present held a mask across a scope change");
    generated.now_ms = 10056;
    require(s.step(generated).unavailable, "A generated Present held after its chain ended");
    // The next real Present starts a new chain.
    real.now_ms = 10064;
    const auto &next = s.step(real);
    require(next.detected && (next.gpu.per_frame & ui_detection::per_frame_hold_reset) && next.source == 3, "The real Present after a chain ended did not detect with hold_reset");
    check_counters(s, "no bound");
    // DLSS multi-frame generation at 6x (5 generated Presents per real one)
    // holds every generated Present.
    alpha_auto_policy six;
    accept_backbuffer(six);
    sequence stream(six, backbuffer_frame);
    const auto t = fg_drift(stream, 601, 0, [](std::size_t) {
      return std::pair {5u, 5u};
    });
    require(t.real == 101 && t.held == 500 && !t.beyond && !t.extra_holds, "A 6x stream did not hold every generated Present");
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

    // Both name difference_failed: an inexact pair that changed 90% or more
    // is a pre-UI image claim (H1 (d)), which acts only while the scene
    // guard holds the pre-UI image visible, so it is not gate_no_hold
    // without that hold (F1 under H1).
    for (const auto &v : {variant {990, ui_no_mask::difference_failed, "an inexact full change set"}, variant {500, ui_no_mask::difference_failed, "an inexact middle-band change set"}}) {
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
      p.now_ms = 10000;
      require(s.step(p).source == 3, "The accepted Backbuffer did not decide");
      p.offered = candidate::current;
      for (std::uint32_t i = 0; i != 4; ++i) {
        p.now_ms = 10016 + 16 * i;
        const auto &r = s.step(p);
        require(r.detected && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && !r.gpu.accepted && !r.decision.own_source && s.temporal.bits == candidate::backbuffer, "Current alpha inherited the missing Backbuffer's acceptance, or the frame adopted");
        require(i ? !r.decision.reused && !r.source : r.decision.reused && r.source == 3 && r.covered == 200, "The missing accepted Backbuffer's decision was not reused exactly once");
      }
      // A generated Present (it offers nothing) shows the real frame's
      // decision; it never decides from Present-time evidence.
      p.offered = candidate::backbuffer;
      p.now_ms = 10100;
      require(s.step(p).source == 3, "The returning Backbuffer did not decide");
      auto generated = p;
      generated.offered = 0;
      generated.hold_previous = true;
      generated.now_ms = 10116;
      const auto &held = s.step(generated);
      require(held.held && !held.detected && held.hold.kind == hold_kind::generated && held.source == 3, "A generated Present did not hold the real frame's decision");
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
      generated.offered = 0;
      generated.hold_previous = true;
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
    // that frame generation tags on real Presents only: the Backbuffer
    // decides every real frame. A generated Present offers the layer (copied
    // at every Present), so Present counting reads it as real: it misses the
    // accepted Backbuffer and, the accepted layer being invalid, reuses the
    // real frame's decision once (T1). The S1 status key, which churned
    // between them, gates nothing and discards no sample.
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
    real.offered = candidate::layer | candidate::backbuffer;
    real.layer_flags = ui_detection::layer_detection_flags(false);
    std::size_t generated = 0;
    for (std::uint64_t now = 10000; now < 13000; now += 8) {
      auto step = pacer.next(real, candidate::layer, 1, 1);
      step.p.now_ms = now;
      const auto &r = s.step(step.p);
      if (!step.real) {
        ++generated;
        require(r.detected && r.decision.reused && r.source == 3 && r.covered == 40, "A generated Present beside the invalid accepted layer did not reuse the real frame's decision");
      } else {
        require(r.detected && !r.decision.reused && r.source == 3 && r.covered == 40, "The accepted Backbuffer did not decide beside the invalid accepted layer");
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
    std::uint32_t real_source = 0, real_covered = 0;
    for (std::uint64_t now = 10000; now < 17000; now += 8) {
      // The input provider's Present counting (fg_pacer).
      auto step = pacer.next(real, 0u, c.actual, c.reported);
      auto &p = step.p;
      p.now_ms = now;
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
        require(r.grace && !r.submitted && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && (r.decision.reused ? r.source == 3 || r.source == 8 : !r.source), "A zero-offer Present beyond the reported count did not reuse the accepted mask at most once (" + name + ")");
        ++(r.decision.reused ? reused : lost);
      } else {
        require(!r.active && !r.source && !real_source, "A zero-offer Present beyond the reported count held a mask decided before acceptance (" + name + ")");
      }
      // Load Game is flat from the accepted Backbuffer (3), and from H1 (8)
      // once the guard holds its hidden verdict: the Backbuffer is opaque on
      // 99.9998% of pixels, not on every one.
      if (now >= 14100 && c.actual == c.reported) {
        require(r.flat() && (r.source == 3 || r.source == 8), "Load Game over a hidden scene was not flat with the accepted Backbuffer alpha (" + name + ")");
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
    // flat (P1) over a hidden scene until H1 takes it over: the counters
    // measure H1's samples hidden, never visible.
    require(counted[ui_counter::full_alpha] > 0 && counted[ui_counter::full_d_hidden] > 0 && !counted[ui_counter::full_d_visible], "The counters did not record Load Game's whole-frame alpha over a hidden scene (" + name + ")");
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

  // ---------------------------------------------------------------- E2/V2 HUD-less pairs

  // Hogwarts Legacy 10-05 (Epic launch, DLSS-G 4x multi-frame generation)
  // tagged only HUDLessColor: no Backbuffer and no UI color. Without a
  // same-batch Backbuffer, counting proposes the presented colour after the
  // tag (ui_mask::pair_hudless_present), which is never exact (E2): only V2
  // validates the pair, and only a same-batch Backbuffer makes it exact.

  // The input provider's HUD-less offers under a frame-generation cadence:
  // the game tags the HUD-less image of each real frame once, and every
  // Present of that frame's window (its `generated` generated Presents and
  // its real one, presented first, or last as DLSS-G does) offers that
  // snapshot, from `delay` Presents into the window (a capture that
  // completes late, on another queue); until then the previous window's
  // snapshot is offered again. Each offer pairs as the provider pairs it
  // under frame generation (ui_mask::pair_hudless_offer: a first offer on
  // the Present after the tag with that Present, a later first offer and
  // every later offer with its own Present's colour), and the pair compares
  // the snapshot's own real frame only when the compared Present is that
  // frame's real one; any other shows interpolated colour or another frame,
  // a mispaired pair. A later offer is a re-offer of the render before
  // (hudless_reoffer).
  struct hudless_window {
    std::uint32_t generated;
    bool real_first;
    std::uint32_t delay = 0;

    struct offer {
      bool real, first, own_frame;
    };

    offer at(std::size_t present_index) const {
      const std::size_t window = generated + 1, j = present_index % window, real_at = real_first ? 0 : generated;
      const std::uint64_t tagged = 1000 + window * (present_index / window), current = tagged + j + 1;
      const std::uint64_t snapshot = j < delay ? tagged - window : tagged;
      const auto pairing = ui_mask::pair_hudless_offer(snapshot, current, j != delay, true);
      require(pairing.kind != ui_mask::hudless_present::unpaired && pairing.presents_ago <= j, "An on-time HUD-less offer did not pair");
      return {j == real_at, j == delay, current - pairing.presents_ago == snapshot + 1 + real_at};
    }
  };

  void hudless_pairs_are_exact_only_by_batch() {
    // The game tags its next frame one to three counted Presents before each
    // Present, so every Present pairs (where counting by the reported
    // multiplier had read every one as generated and held 49 s without a
    // decision).
    for (std::uint64_t after = 1; after <= 3; ++after) {
      require(ui_mask::pair_hudless_present(3725, 3725 + after).kind != ui_mask::hudless_present::unpaired, "A pipelined HUD-less tag did not pair");
    }
    for (const auto [generated, real_first, delay] : {std::tuple {3u, false, 0u}, std::tuple {3u, true, 0u}, std::tuple {1u, false, 0u},
           std::tuple {1u, true, 0u}, std::tuple {1u, false, 1u}, std::tuple {3u, false, 1u}}) {
      // (i) FG 4x (three generated Presents per real one; Hogwarts Legacy
      // 10-05) and 2x, every Present offering its real frame's inexact
      // HUD-less snapshot, with the real Present last (DLSS-G) or first in
      // the window; and with the real Present last, a capture one Present
      // late, whose first offer under frame generation pairs with its own
      // Present (at 2x the real one; before, the retained generated colour
      // left every pair mispaired). Each Present's change
      // set follows from the pair the provider makes (hudless_window): the
      // real Present compares its own colour, a valid partial set (the HUD),
      // and every generated Present interpolated colour (a middle-band set,
      // V2-invalid). The first V2-valid selective sample accepts the pair
      // (A1, exact or not); from then on the real Present decides 5 and every
      // mispaired Present shows that held decision (T1: a snapshot's first
      // offer without a decision of its own reuses it once, each re-offer
      // keeps it, per_frame_reoffer). Then a menu-sized change set over a
      // visible scene: full but inexact, so never 6, and with claim (d) but
      // no hidden verdict never 8: the re-offers of the snapshot that spent
      // the grace keep the held HUD, and from the next snapshot's first offer
      // there is no mask.
      const hudless_window cadence {generated, real_first, delay};
      const std::string name = std::to_string(generated + 1) + "x, " + (real_first ? "real Present first" : "real Present last") +
        (delay ? ", capture one Present late" : "");
      alpha_auto_policy session;
      sequence s(session, [cadence](const gpu_inputs &in) {
        synthetic f;
        f.change_set(in.now_ms >= 12000 ? 990 : cadence.at((in.now_ms - 10000) / 8).own_frame ? 50 : 500).lit(1000).scene(.5f, .5f);
        return f.words(in);
      });
      for (std::uint64_t now = 10000; now < 14000; now += 8) {
        present p;
        p.now_ms = now;
        p.offered = candidate::hudless;
        p.hudless_reoffer = !cadence.at((now - 10000) / 8).first;
        s.step(p);
      }
      std::size_t accepted_at = s.frames.size(), menu_first_offers = 0, held_menu = 0;
      bool real_decided = false;
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &r = s.frames[i];
        const auto offer = cadence.at(i);
        require(r.detected && !r.held && (r.gpu.bits & candidate::hudless) && !(r.gpu.bits & candidate::exact), "A Hogwarts Present did not detect from an inexact HUD-less pair (" + name + ")");
        require(r.source != 6 && r.source != 8 && !r.decision.h1, "A Hogwarts inexact pair decided a whole-frame source (" + name + ")");
        require(((r.gpu.per_frame & ui_detection::per_frame_reoffer) != 0) == !offer.first, "A re-offer was not flagged, or a first offer was (" + name + ")");
        if (accepted_at == s.frames.size() && (r.gpu.accepted & candidate::hudless)) {
          accepted_at = i;
        }
        if (i < accepted_at) {
          require(!r.source, "The pair decided before it was accepted (" + name + ")");
        } else if (r.now_ms < 12000) {
          // Until the first real Present after acceptance, the held decision
          // is that of a real frame decided before it: no mask.
          if (offer.real) {
            real_decided = true;
          }
          // A held decision of no mask (source 0) is no decision to keep, so
          // a re-offer may leave it unused.
          const bool shows = real_decided ? r.source == 5 && r.covered == 50 : !r.source;
          require(shows && (offer.real ? !r.decision.reused && r.decision.own_source == 5 : !r.decision.own_source && (r.decision.reused || !real_decided)), "A Hogwarts gameplay Present did not decide its own valid change set, or a mispaired one did not show the held decision (" + name + ", Present " + std::to_string(i) + ")");
        } else {
          menu_first_offers += offer.first ? 1 : 0;
          held_menu += r.source ? 1 : 0;
          require(!r.decision.own_source && r.decision.none_reason == ui_no_mask::difference_failed && (!r.source || (r.decision.reused && r.source == 5)) && (menu_first_offers < 2 || !r.source), "The inexact full change set over a visible scene decided something, or the held HUD outlived the snapshot that spent the grace (" + name + ")");
        }
      }
      require(accepted_at < s.frames.size() && s.frames[accepted_at].now_ms < 11000 && s.trust.size() == 1 && s.trust[0].accepted == key(kind::hudless), "The inexact HUD-less pair was not accepted by a valid selective sample (" + name + ")");
      require(held_menu && held_menu <= 2 * (cadence.generated + 1), "The menu's held HUD did not end within two real frames (" + name + ")");
      require(!s.scene_entered && !s.guard.hidden.held(14000), "A visible scene entered the hidden-scene hold (" + name + ")");
      check_counters(s, "Hogwarts HUD-less only, FG " + name);
    }
    {
      // (ii) FG off, the Present after each tag counted, not a batch: an
      // accepted pair's full change set over a visible scene (a pause menu
      // over the live scene, or a pair that missed its frame) is inexact, so
      // it never decides 6; the T1 grace reuses the partial decision once.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set(in.now_ms < 11000 ? 50 : 990).lit(1000).scene(.5f, .5f);
        return f.words(in);
      });
      present p;
      p.offered = candidate::hudless;
      run(s, p, 10000, 12000);
      std::size_t reused = 0;
      for (const auto &r : s.frames) {
        require(r.detected && r.source != 6 && r.source != 8, "A counted full change set decided a whole-frame source");
        if (r.now_ms < 11000) {
          require(r.source == 5 && r.covered == 50, "A counted partial change set did not decide 5");
        } else {
          reused += r.decision.reused ? 1 : 0;
          require(!r.decision.own_source && (r.decision.reused ? r.source == 5 : !r.source), "A counted full change set had a decision of its own");
        }
      }
      require(reused == 1 && !s.scene_entered, "The grace did not reuse once, or a visible scene entered the hidden-scene hold");
      check_counters(s, "FG off counted full change set");
    }
    {
      // (iii) A title or full-screen menu of a HUD-less-only game over a hidden
      // scene (presented D 0.039, the HUD-less image 0.556): the inexact full
      // change set never decides 6, but the HUD-less image is the pre-UI
      // scene image (claim (d), changed on 90% or more), so the frame is flat
      // through H1 (8) from the poll of the second hidden sample, once the
      // guard holds the hidden and the pre-UI verdicts; a first session that
      // never accepted the pair.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set(999).lit(1000).scene(.039f, .556f);
        return f.words(in);
      });
      present p;
      p.offered = candidate::hudless;
      run(s, p, 10000, 11000);
      require(s.samples.size() > 3 && !measured(s.samples[0]) && s.samples[1].evidence.scene.ran && s.observed[2].entered, "The HUD-less-only title did not measure and enter at its second hidden sample");
      const auto entry = poll_of(s, s.samples[2].sample_tick_ms);
      require(s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible), "The title did not hold both verdicts from its entry");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &r = s.frames[i];
        require(r.source != 6 && (i >= entry ? r.flat() && r.source == 8 && r.decision.h1 : !r.source && r.decision.none_reason == ui_no_mask::difference_failed), "The HUD-less-only title was not flat exactly from the hold's entry through H1 (d)");
      }
      require(session.stored().empty(), "The title's full change set earned acceptance");
      check_counters(s, "HUD-less-only title over a hidden scene");
    }
    {
      // (iv) A same-batch Backbuffer makes the pair exact (Hogwarts 09-30 with
      // its Backbuffer tagged): an accepted full change set over a lit scene
      // is a full-screen menu and decides 6, flat.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set(990).lit(1000).scene(.5f, .5f);
        return f.words(in);
      });
      present p;
      p.offered = candidate::backbuffer | candidate::hudless | candidate::exact;
      run(s, p, 10000, 11000);
      for (const auto &r : s.frames) {
        require(r.detected && r.flat() && r.source == 6 && !r.decision.h1, "A same-batch exact full change set did not decide 6");
      }
      check_counters(s, "same-batch exact full change set");
    }
  }

  // T1 across HUD-less re-offers (selection revision 10, per_frame_reoffer).
  void hudless_reoffers_keep_the_held_decision() {
    // A game that tags UIAlpha for real frames only and a HUD-less image
    // that every Present of the real frame's window offers (The Witcher 3 at
    // 3x and 4x): the real Present decides its accepted UIAlpha; a generated
    // Present misses it (accepted missing) and pairs its HUD-less snapshot
    // with interpolated colour (V2-invalid), so it has no decision of its
    // own. The snapshot's first offer reuses the real decision once, and
    // every re-offer keeps the held decision, so every generated Present
    // shows its last real frame's UIAlpha, with the real Present first or
    // last in the window. Before revision 10 the grace was spent by the
    // first generated Present and the rest of the window had no mask.
    const auto ui_alpha = key(kind::ui_alpha);
    for (const std::uint32_t generated : {2u, 3u}) {
      for (const bool real_first : {false, true}) {
        const hudless_window cadence {generated, real_first};
        const std::string name = std::to_string(generated + 1) + "x, " + (real_first ? "real Present first" : "real Present last");
        alpha_auto_policy session;
        restore(session, ui_alpha, 1);
        sequence s(session, [cadence](const gpu_inputs &in) {
          synthetic f;
          f.alpha(kind::ui_alpha, (in.bits & candidate::ui_alpha) ? 60u : 0u).change_set(cadence.at((in.now_ms - 10000) / 8).own_frame ? 60 : 500).lit(1000);
          return f.words(in);
        });
        std::uint32_t real_source = 0;
        std::size_t generated_presents = 0;
        for (std::uint64_t now = 10000; now < 12000; now += 8) {
          const auto offer = cadence.at((now - 10000) / 8);
          present p;
          p.now_ms = now;
          p.offered = candidate::hudless | (offer.real ? candidate::ui_alpha : 0u);
          p.hudless_reoffer = !offer.first;
          const auto &r = s.step(p);
          require(r.detected && !r.held, "A W3 Present did not detect (" + name + ")");
          require(((r.gpu.per_frame & ui_detection::per_frame_reoffer) != 0) == (!offer.first && !offer.real), "The re-offer bit was not pushed exactly on tagless re-offers (" + name + ")");
          if (offer.real) {
            require(r.source == 1 && r.covered == 60 && !r.decision.reused, "The real W3 Present did not decide its accepted UIAlpha (" + name + ")");
            real_source = r.source;
          } else if (real_source) {
            ++generated_presents;
            require(!r.decision.own_source && r.decision.reused && r.source == real_source && r.covered == 60 && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing), "A generated W3 Present did not show its last real frame's UIAlpha (" + name + ")");
          }
        }
        require(generated_presents > 100, "The W3 stream had too few generated Presents (" + name + ")");
        check_counters(s, "W3 HUD-less re-offers " + name);
      }
    }
    {
      // FG off, a stale re-offer: the game misses a few HUD-less tags (a busy
      // capture reservation), so the provider offers the last snapshot again.
      // Without frame generation a re-offer pairs as its first offer did,
      // with the snapshot's own frame retained (ui_mask::pair_hudless_offer:
      // earlier_present, a valid pair that decides its own change set), and
      // more than max_late_presents Presents after its tag it is unpaired
      // and not offered at all: the accepted pair is missing, so the T1
      // grace reuses the decision once and the next renders have no mask
      // until a new snapshot pairs again. Before, the re-offers were compared
      // with each later Present's colour, a pair of two frames that V2 could
      // pass at low motion.
      alpha_auto_policy session;
      accept_hudless(session);
      struct offer_t {
        bool offered, reoffer, own_frame;
      };
      std::vector<offer_t> offers;
      for (std::uint64_t i = 0; i != 40; ++i) {
        // Snapshot 19 (tagged 1019) is offered again on renders 20-29.
        const bool stale = i >= 20 && i < 30;
        const std::uint64_t tagged = stale ? 1019 : 1000 + i, current = 1001 + i;
        const auto pairing = ui_mask::pair_hudless_offer(tagged, current, stale, false);
        const bool offered = pairing.kind != ui_mask::hudless_present::unpaired;
        // The compared color is the snapshot's own frame, the Present after its tag.
        require(!offered || current - pairing.presents_ago == tagged + 1, "An FG-off offer did not pair with its own frame");
        offers.push_back({offered, stale, offered});
      }
      require(offers[20].offered && offers[21].offered && !offers[22].offered && !offers[29].offered && offers[30].offered, "An FG-off re-offer did not pair up to max_late_presents after its tag, and not after");
      sequence s(session, [offers](const gpu_inputs &in) {
        synthetic f;
        f.change_set(offers[(in.now_ms - 10000) / 16].own_frame ? 50 : 500).lit(1000);
        return f.words(in);
      });
      for (std::size_t i = 0; i != offers.size(); ++i) {
        present p;
        p.now_ms = 10000 + 16 * i;
        p.offered = offers[i].offered ? candidate::hudless : 0u;
        p.hudless_reoffer = offers[i].reoffer;
        const auto &r = s.step(p);
        if (offers[i].offered) {
          require(r.detected && r.source == 5 && r.covered == 50 && !r.decision.reused, "An FG-off pair of the snapshot's own frame, re-offered or not, did not decide");
        } else if (i == 22) {
          require(r.detected && (r.gpu.per_frame & ui_detection::per_frame_accepted_missing) && r.decision.reused && r.source == 5 && r.covered == 50, "The first render without the unpaired snapshot did not reuse the decision once");
        } else {
          require(!r.source, "The held decision outlived the grace after the snapshot became unpaired");
        }
      }
      check_counters(s, "FG-off stale re-offer");
    }
    {
      // A re-offer beside a UIAlpha, UI color or Backbuffer tag, or paired
      // exactly with its batch's Backbuffer, is a frame of its own: the
      // renderer pushes no re-offer bit (an accepted tag decides it or the
      // grace applies once).
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set(500).lit(1000);
        return f.words(in);
      });
      for (const std::uint32_t with : {unsigned(candidate::ui_alpha), unsigned(candidate::ui_color), unsigned(candidate::backbuffer), unsigned(candidate::exact)}) {
        present p;
        p.now_ms = 10000 + 16 * s.frames.size();
        p.offered = candidate::hudless | with;
        p.hudless_reoffer = true;
        require(!(s.step(p).gpu.per_frame & ui_detection::per_frame_reoffer), "A re-offer beside a tag or an exact pair pushed the re-offer bit");
      }
    }
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
      // Resident Evil Requiem with FG after a session accepted the HUD-less
      // pair: every Present offers the accepted UIAlpha, and its HUD-less
      // image pairs (inexactly, by Present counting) on some Presents only.
      // The accepted UIAlpha decides every Present; the Presents without the
      // pair miss an accepted candidate, adopt nothing and discard nothing.
      alpha_auto_policy session;
      restore(session, stored_of({signatures.of(kind::ui_alpha), signatures.of(kind::hudless)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_alpha, 50).change_set(50);
        return f.words(in);
      });
      present p;
      std::size_t unpaired = 0, index = 0;
      for (std::uint64_t now = 10000; now < 13000; now += 8, ++index) {
        const bool paired = index % 2 == 0;
        p.offered = paired ? candidate::ui_alpha | candidate::hudless : candidate::ui_alpha;
        p.now_ms = now;
        const auto &r = s.step(p);
        unpaired += paired ? 0 : 1;
        require(r.detected && !r.decision.reused && r.source == 1 && r.covered == 50, "The accepted UIAlpha did not decide every Present");
      }
      require(unpaired > 100 && !s.discards && s.samples.size() >= 25, "An intermittent accepted HUD-less pair discarded samples beside the accepted UIAlpha (" + std::to_string(s.discards) + " discarded, " + std::to_string(s.samples.size()) + " read)");
      require(s.frames.back().consumed.state != alpha_auto_state::collecting, "The status stayed checking");
      check_counters(s, "accepted UIAlpha beside an accepted HUD-less pair");
    }
    {
      // An accepted offscreen layer beside an accepted Backbuffer that frame
      // generation tags on real Presents only: the layer copy is offered on
      // every Present, so every Present is real to Present counting; the
      // layer decides every Present, and Presents without the Backbuffer flag
      // it missing but decide by themselves.
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
        auto step = pacer.next(real, candidate::layer, 1, 1);
        step.p.now_ms = now;
        const auto &r = s.step(step.p);
        require(r.detected && r.source == ui_detection::source_layer && !r.decision.reused, "The accepted layer did not decide every Present");
      }
      require(!s.discards && s.samples.size() >= 25 && s.frames.back().consumed.state != alpha_auto_state::collecting, "An intermittent accepted Backbuffer discarded samples beside the accepted layer");
      check_counters(s, "accepted layer beside an accepted Backbuffer");
    }
  }

  // ---------------------------------------------------------------- H1 hidden scene

  void hidden_scene_layer_route() {
    // The Witcher 3 graphics settings: an opaque, unaccepted cleared layer
    // over a hidden scene, an informative full claim (H1 (b)). Phases by
    // sample tick: hidden, invalid, hidden, one visible sample, hidden while
    // the layer's signature is refuted, overlay samples (the layer below
    // opaque, which restore it), hidden again.
    const auto w3 = recorded_frames({recorded::w3_settings, recorded::w3_settings_held});
    alpha_auto_policy session;
    sequence s(session, [&](const gpu_inputs &in) {
      require(in.bits == (candidate::layer | candidate::current) && !in.accepted, "The W3 settings stream offered other inputs");
      auto t = w3(in);
      // Overlay frames: the GPU counts the layer below opaque, so no claim.
      if (in.now_ms >= 16000 && in.now_ms < 16200) {
        t[word::layer_opaque] = 0;
        t = decision_words(t, in);
      }
      return t;
    });
    // Synthetic edits of the recorded evidence: invalid samples, and one
    // visible sample (D 0.5).
    bool visible_sent = false;
    s.sample_edit = [&visible_sent](texels &t, std::uint64_t tick, bool evidence) {
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
    const auto layer = p.signatures.of(kind::ui_layer);
    require(layer.key() == "ui_layer:28:srgb", "The W3 layer signature moved");
    std::vector<std::size_t> entries;
    std::size_t visible_releases = 0, visible_poll = 0, restore_poll = 0;
    for (std::uint64_t now = 10000; now < 17500; now += 16) {
      p.now_ms = now;
      const auto before = s.guard.hidden;
      const auto &r = s.step(p);
      const auto index = s.frames.size() - 1;
      if (r.polled) {
        const auto &sample = s.samples.back();
        const auto &scene = sample.evidence.scene;
        const auto &observed = s.observed.back();
        const auto tick = sample.sample_tick_ms;
        if (!scene.valid) {
          require(s.guard.hidden.until == before.until && s.guard.hidden.last == before.last, "Invalid evidence changed a hold");
        } else if (scene.verdict == scene_verdict::visible) {
          require(!s.guard.hidden.until && observed.released && observed.refuted == 1 && s.guard.refuted(layer), "A visible sample did not release the hold and refute the layer's signature");
          ++visible_releases;
          visible_poll = index;
        } else if (scene.verdict == scene_verdict::hidden && scene.ran && !s.guard.refuted(layer)) {
          // H1: each valid hidden sample renews a held verdict to its tick
          // plus 500 ms; one that follows another within 500 ms enters it.
          if (before.held(tick) || observed.entered) {
            require(s.guard.hidden.until == tick + ui_detection::scene::hold_ms, "A valid hidden sample did not hold for 500 ms from its tick");
          }
          require(observed.entered == (!before.held(tick) && before.last && tick - before.last <= ui_detection::scene::hold_ms), "A hidden sample entered without a second sample within 500 ms, or did not enter with one");
        }
        if (observed.entered) {
          entries.push_back(index);
        }
        if (tick >= 16000 && tick < 16200 && !restore_poll) {
          require(!s.guard.refuted(layer), "A sample showing the layer below opaque did not restore its signature");
          restore_poll = index;
        }
      }
      // The mask follows the held verdict and the refutation exactly.
      require(bool(r.scene_bits & ui_detection::per_frame_scene_hidden) == s.guard.hidden.held(now) && bool(r.scene_bits & (candidate::layer << ui_detection::per_frame_refuted_shift)) == s.guard.refuted(layer), "The pushed bits did not follow the guard at " + std::to_string(now));
      const bool overlay = now >= 16000 && now < 16200;
      const bool flat = s.guard.hidden.held(now) && !s.guard.refuted(layer) && !overlay;
      require(r.source == (flat ? 8u : 0u) && r.decision.h1 == flat, "The frame's mask did not follow the CPU-held verdict at " + std::to_string(now));
      if (!flat && !overlay && !s.guard.refuted(layer)) {
        require(r.decision.none_reason == ui_no_mask::gate_no_hold && r.decision.refused == candidate::layer, "A frame with the layer's acting claim and no hold did not name gate_no_hold");
      }
    }
    // Entry: the first sample opens the gate unmeasured, and the poll of the
    // second measured hidden sample within 500 ms is the first flat frame.
    require(!measured(s.samples[0]) && s.samples[1].evidence.scene.ran && s.samples[2].evidence.scene.ran && s.samples[2].sample_tick_ms - s.samples[1].sample_tick_ms <= ui_detection::scene::hold_ms, "The first samples did not open the gate and measure");
    require(!entries.empty() && entries[0] == poll_of(s, s.samples[2].sample_tick_ms), "The hold did not enter at the poll of the second measured hidden sample");
    for (std::size_t i = 0; i != entries[0]; ++i) {
      require(!s.frames[i].source, "A frame was flat before the hold entered");
    }
    // A refuted layer never flattens until a sample shows it below opaque;
    // then it re-arms (the gate, then two hidden samples).
    require(visible_releases == 1 && visible_poll && restore_poll > visible_poll, "The visible release or the restore did not happen once, in order");
    for (std::size_t i = visible_poll; i != restore_poll; ++i) {
      require(!s.frames[i].source, "A refuted layer flattened");
    }
    require(entries.size() == 3 && entries[1] < visible_poll && entries[2] > restore_poll && s.frames.back().source == 8u, "The hold did not re-enter after the invalid run and re-arm after the restore");
    const auto c = session.counters();
    require(c[ui_counter::full_d_visible] == 1 && c[ui_counter::scene_released] == 1 && c[ui_counter::scene_refuted] == 1 && c[ui_counter::scene_entered] == 3, "The visible release was not counted once as full_d.visible, released and refuted");
    require(session.stored().empty(), "The full opaque layer was accepted");
    check_counters(s, "W3 layer route");
  }

  void hidden_scene_gate_and_clears() {
    // A hidden synthetic stream with an opaque, unaccepted cleared layer: an
    // informative full claim (H1 (b)).
    const auto hidden_layer = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 1000).opaque(kind::ui_layer, 1000).scene(-.02f, -.02f);
      return f.words(in);
    };
    const auto layer_flags = ui_detection::layer_detection_flags(false);
    {
      // Without a claim nothing measures. With one, the sample that first
      // shows it measures nothing (the gate opens at its poll), and the next
      // two, actionable, enter the hold.
      alpha_auto_policy session;
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
      for (const auto &sample : s.samples) {
        require(!sample.evidence.scene.ran, "A sample without a claim measured");
      }
      require(!s.guard.hidden.until && !s.guard.hidden.last, "A sample without a claim held a verdict");
      alpha_auto_policy gated;
      sequence g(gated, hidden_layer);
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = layer_flags;
      std::size_t polls = 0;
      for (std::uint64_t now = 10000; now < 11000 && polls < 3; now += 16) {
        p.now_ms = now;
        if (!g.step(p).polled) {
          continue;
        }
        ++polls;
        require(g.samples.back().evidence.scene.ran == (polls != 1) && (polls == 3) == g.guard.hidden.held(now) && (polls == 1) == !g.guard.hidden.last, "The sample that first showed the claim measured, or the next two actionable ones did not enter");
      }
    }
    alpha_auto_policy session;
    sequence s(session, hidden_layer);
    present p;
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = layer_flags;
    const auto layer = p.signatures.of(kind::ui_layer);
    std::uint64_t now = 10000;
    const auto establish = [&] {
      for (const auto until = now + 1000; now < until; now += 16) {
        p.now_ms = now;
        if (s.step(p).scene_bits & ui_detection::per_frame_scene_hidden) {
          require(s.frames.back().source == 8u, "The held verdict did not flatten the layer's claim");
          return;
        }
      }
      throw std::runtime_error("The hidden verdict did not hold");
    };
    const auto held = [&] {
      return s.guard.hidden.held(now);
    };
    // H1: D state clears only on an identity change (epoch or viewport).
    const auto cleared = [&](const char *what) {
      require(!s.guard.hidden.until && !s.guard.hidden.last && !s.guard.pre_ui.until && !s.guard.refuted_count && !s.guard.per_frame(now, p.offered, p.signatures.by_kind(), session.pre_ui_proven(p.signatures.of(kind::ui_layer))), std::string(what) + " kept hidden-scene state");
    };
    const auto step = [&](present q) -> const frame_result & {
      q.now_ms = now += 16;
      return s.step(q);
    };
    establish();
    // UIAlpha appearing changes the offered inputs, not the identity: the
    // held verdict stays and the layer's claim keeps the frame flat.
    const auto with_alpha = [&p] {
      auto q = p;
      q.offered = candidate::ui_alpha | candidate::layer | candidate::current;
      return q;
    };
    for (const auto until = now + 300; now < until;) {
      const auto &r = step(with_alpha());
      require(held() && r.source == 8u, "UIAlpha appearing cleared the held verdict");
    }
    for (const auto &[field, what] : {std::pair {0, "An epoch change"}, std::pair {2, "A viewport change"}}) {
      establish();
      if (field == 0) {
        ++p.epoch;
      } else {
        ++p.viewport;
      }
      const auto &r = step(p);
      cleared(what);
      require(!r.source && r.consumed.state == alpha_auto_state::collecting, std::string(what) + " kept the previous scope's mask or sample");
    }
    {
      // A revision change (a depth observation loss) discards the pending
      // sample unread but keeps the guard's holds.
      establish();
      while (!step(p).submitted) {}
      ++p.revision;
      const auto &r = step(p);
      require(r.discarded && held() && r.source == 8u && r.consumed.state == alpha_auto_state::collecting, "A revision change did not discard the pending sample, or cleared the held verdict");
    }
    {
      // An inactive frame clears T1's chain, not the guard: the next frame is
      // flat at once.
      establish();
      auto inactive = p;
      inactive.available = false;
      const auto &off = step(inactive);
      require(!off.active && !s.temporal.have_decision && s.temporal.reset_pending && held(), "An inactive frame kept a T1 decision, or cleared the held verdict");
      require(step(p).source == 8u, "The frame after an inactive one was not flat under the held verdict");
    }
    {
      // Acceptance changes never clear D state: the layer accepted mid-hold
      // is the S1 winner (opaque on every pixel), which H1 relabels 8 under
      // the held verdict (selection revision 10: flat either way, P1), and
      // stays flat; forgotten again, H1 still flattens it.
      establish();
      restore(session, layer.key(), 1);
      for (const auto until = now + 300; now < until;) {
        const auto &r = step(p);
        require(held() && r.flat() && r.source == 8u && r.decision.h1 && r.decision.s1_source == ui_detection::source_layer && (r.scene_bits & ui_detection::per_frame_scene_hidden), "The layer accepted mid-hold did not stay flat, or cleared the held verdict");
      }
      require(s.forget() == layer.key(), "Forget did not clear the accepted layer");
      const auto &r = step(p);
      require(held() && r.flat() && r.source == 8u && r.decision.h1, "Forgetting the layer mid-hold did not return to H1 at once");
    }
    // A visible sample releases the hold and refutes the layer's signature;
    // UIAlpha appearing keeps the refutation, and only an identity change
    // clears it.
    s.sample_edit = [](texels &t, std::uint64_t, bool evidence) {
      if (evidence) {
        t[word::scene_state] = scene_state(true, scene_verdict::visible);
        t[word::scene_d] = float_bits(.5f);
      }
    };
    for (const auto until = now + 500; now < until;) {
      step(p);
    }
    require(s.guard.refuted(layer) && !held() && !s.frames.back().source, "A visible sample did not release the hold and refute the layer");
    s.sample_edit = nullptr;
    for (const auto until = now + 1000; now < until;) {
      const auto &r = step(with_alpha());
      require(!r.source && (r.scene_bits & (candidate::layer << ui_detection::per_frame_refuted_shift)), "UIAlpha appearing cleared the layer's refutation");
    }
    require(s.guard.refuted(layer), "The refutation did not last");
    ++p.epoch;
    step(p);
    cleared("An epoch change after a refutation");
    check_counters(s, "scope clears");
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
      // The counters see it: whole-frame alpha frames.
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && !c[ui_counter::contradicted], "The counters did not record the accepted full layer over a visible scene");
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
      // pins flat on real Presents (P1). A generated Present offers only its
      // HUD-less image, paired by Present counting with interpolated colour
      // (a middle-band change set, V2-invalid), so it is real to counting:
      // without the accepted UIAlpha it reuses the real frame's decision once
      // (T1). The inexact pair never judges it (E2, A2).
      const auto recorded = phased({{14000, recorded_frames({recorded::w3_hud_fg, recorded::w3_hud_fg_accepted})}, {UINT64_MAX, recorded_frames({recorded::w3_wheel_fg, recorded::w3_wheel_fg_accepted})}});
      const auto w3 = [recorded](const gpu_inputs &in) {
        if (in.bits & candidate::ui_alpha) {
          return recorded(in);
        }
        synthetic f;
        f.c.pixels = 3686400;
        f.change_set(3686400 / 2);
        return f.words(in);
      };
      alpha_auto_policy session;
      sequence s(session, w3);
      fg_pacer pacer(1);
      present real;
      real.offered = candidate::ui_alpha | candidate::current | candidate::hudless;  // UIAlpha, current and an inexact HUD-less pair.
      std::uint32_t last_real = 0;
      for (std::uint64_t now = 10000; now < 17000; now += 8) {
        auto step = pacer.next(real, candidate::hudless, 1, 1);
        step.p.now_ms = now;
        const auto &r = s.step(step.p);
        require(r.detected && !r.held, "A W3 FG Present did not detect");
        if (!step.real) {
          require(r.source == last_real && !r.decision.own_source, "A generated Present did not show the real frame's mask");
        } else if (!(r.gpu.accepted & candidate::ui_alpha)) {
          require(!r.source && r.decision.none_reason == ui_no_mask::unaccepted, "The W3 HUD UIAlpha decided before its first sample was read");
        } else if (now < 14000) {
          require(r.source == 1 && r.covered == 192930, "The accepted W3 HUD UIAlpha did not decide");
        }
        if (now >= 14100) {
          require(r.flat() && r.source == 1, "The accepted W3 FG on sign wheel was not flat");
        }
        if (step.real) {
          last_real = r.source;
        }
      }
      require(s.trust.size() == 1 && s.trust[0].tick == s.samples.front().sample_tick_ms && s.trust[0].accepted == key(kind::ui_alpha), "The W3 HUD UIAlpha was not accepted once by its first sample");
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0, "The counters did not record the accepted full UIAlpha over a visible scene");
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
    // settings frames): one valid hidden (or visible) sample, then valid but
    // ambiguous ones. H1 enters only on two valid hidden samples within
    // 500 ms, and an ambiguous sample breaks the run: nothing flattens.
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
        flat += f.source ? 1 : 0;
      }
      require(first_tick && evidence_samples > (verdict == scene_verdict::hidden ? 2u : 0u) && !flat && !s.guard.hidden.until && !s.scene_entered, std::string("A dark grainy scene flattened on one valid ") + (verdict == scene_verdict::hidden ? "hidden" : "visible") + " sample");
      check_counters(s, "dark grainy");
    }
  }

  // ---------------------------------------------------------------- H1 (S2b) streams

  // Stellar Blade in SDR (fix 1): the cleared output target is BGRA8
  // (DXGI_FORMAT_B8G8R8A8_UNORM); its pre-UI proof's ledger key.
  candidate_signatures sb_sdr_signatures() {
    auto s = signatures_in(srgb);
    s.set(kind::ui_layer, 87);
    return s;
  }

  std::string sb_sdr_proof() {
    const auto proof = ui_selection::pre_ui_key(sb_sdr_signatures().of(kind::ui_layer)).key();
    require(proof == "pre_ui:87:srgb", "The Stellar Blade SDR pre-UI proof key moved");
    return proof;
  }

  // A Stellar Blade SDR Present offering these candidates on this viewport
  // (1: depth from Streamline with frame generation on; 0: from NGX while a
  // menu suspends it).
  present sb_sdr_present(std::uint32_t offered, std::uint32_t viewport = 1) {
    present p;
    p.offered = offered;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    p.signatures = sb_sdr_signatures();
    p.viewport = viewport;
    return p;
  }

  // Whether a sample counts toward the offered layer's pre-UI proof: a layer
  // without coverage that equals the presented frame (texel 11).
  bool pre_ui_matching(const alpha_auto_decision &sample) {
    const auto &e = sample.evidence;
    return (e.candidates & candidate::layer) && !e.layer_covered && ui_selection::pre_ui_match(e.pre_ui_match, e.pre_ui_image_lit, sample.pixels);
  }

  void h1_stellar_blade_sdr_menu_visits() {
    // Stellar Blade in SDR with frame generation off: no Streamline tag, so
    // the offer is the cleared output target and current alpha. The target
    // holds the pre-UI scene image (colour, alpha 0: V1-invalid). Gameplay
    // shows that image: the layer equals the presented frame on 99.79% of
    // pixels and is lit on 88.9% (dump 470, texel 11), so the session ledger
    // proves its signature (pre_ui:87:srgb) by the first sample 2 s into
    // gameplay, three matching samples (H1 (d), fix 1), without any D. Before
    // the proof nothing claims and no sample measures; from it every sample
    // measures (a proven layer is the offer's pre-UI image). Gameplay reads
    // both images visible; the settings menu (dump 962) reads the presented
    // frame hidden and the pre-UI image visible: the depth describes the
    // pre-UI image, not the menu shown. Gameplay -> settings -> gameplay,
    // three visits: each goes flat (8) from the poll of its second measured
    // sample and is released by the first gameplay sample, which decided 8
    // and reads visible. The menu's samples mismatch (35.28%) and never
    // withdraw the proof; the session ends with it as its only key.
    const auto play = recorded_frames({recorded::sb_sdr_play, recorded::sb_sdr_play_proven});
    const auto menu = recorded_frames({recorded::sb_sdr_menu, recorded::sb_sdr_menu_proven, recorded::sb_sdr_menu_held, recorded::sb_sdr_menu_held_unproven});
    constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 3> visits {{{13000, 14500}, {16000, 17000}, {18500, 20000}}};
    const auto in_menu = [&visits](std::uint64_t tick) {
      return std::any_of(visits.begin(), visits.end(), [tick](const auto &visit) {
        return tick >= visit.first && tick < visit.second;
      });
    };
    alpha_auto_policy session;
    std::vector<std::string> heard;
    session.on_change([&](const std::string &accepted) {
      heard.push_back(accepted);
    });
    sequence s(session, [&](const gpu_inputs &in) {
      return in_menu(in.now_ms) ? menu(in) : play(in);
    });
    const auto p = sb_sdr_present(candidate::layer | candidate::current);
    run(s, p, 10000, 21500);
    const auto proof = sb_sdr_proof();
    const auto first = s.samples.front().sample_tick_ms;
    const auto earned = first_sample_from(s, first + alpha_trust_span_ms);
    require(!s.discards && s.trust.size() == 1 && s.trust[0].tick == earned && s.trust[0].accepted == proof && heard == std::vector<std::string> {proof} && earned < visits[0].first, "Gameplay did not prove the layer by its pixels 2 s in, before the first visit");
    for (const auto &sample : s.samples) {
      const auto &pre_ui = sample.evidence.pre_ui_scene;
      const bool menu_sample = in_menu(sample.sample_tick_ms);
      require(pre_ui_matching(sample) == !menu_sample, "A sample's pixel match does not follow its recording");
      if (sample.sample_tick_ms <= earned) {
        require(!measured(sample) && !sample.evidence.claims, "A sample before the proof claimed or measured");
        continue;
      }
      require(sample.evidence.scene.ran && sample.evidence.scene.valid && pre_ui.valid && sample.evidence.pre_ui_image == ui_detection::pre_ui_image::layer && sample.evidence.claims == ui_detection::claim_pre_ui, "A sample after the proof did not measure both images");
      require(menu_sample ? sample.evidence.scene.verdict == scene_verdict::hidden && pre_ui.verdict == scene_verdict::visible : sample.evidence.scene.verdict == scene_verdict::visible && pre_ui.verdict == scene_verdict::visible, "A sample's verdicts do not match its recording");
    }
    const auto proven_from = poll_of(s, earned);
    // Each visit's flat run: from the poll of its second sample to the poll
    // of the first gameplay sample after it.
    std::vector<std::pair<std::size_t, std::size_t>> flat_runs;
    for (const auto &[open, close] : visits) {
      std::vector<std::size_t> menu_samples;
      std::size_t release = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        const auto tick = s.samples[i].sample_tick_ms;
        if (tick >= open && tick < close) {
          menu_samples.push_back(i);
        } else if (tick >= close && !release) {
          release = i;
        }
      }
      require(menu_samples.size() > 2 && release, "A visit had too few samples");
      const auto entered = menu_samples[1];
      require(!s.observed[menu_samples[0]].entered && s.observed[entered].entered && s.samples[entered].sample_tick_ms - s.samples[menu_samples[0]].sample_tick_ms <= ui_detection::scene::hold_ms, "A visit did not enter at its second hidden sample");
      const auto entry = poll_of(s, s.samples[entered].sample_tick_ms);
      require(s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven), "A visit did not hold both verdicts with the proof from its entry");
      const auto exit = poll_of(s, s.samples[release].sample_tick_ms);
      require(s.observed[release].released && s.samples[release].source_kind == 8u && s.frames[exit].scene_bits == ui_detection::per_frame_pre_ui_proven, "The first gameplay sample after a visit did not release it");
      // The exit latency the pre-UI layer claim implies (it stays true in
      // gameplay): at most one sample interval plus the readback frame.
      require(s.frames[exit].now_ms - close <= sample_interval_ms + 2 * 16, "A visit's release came later than one sample interval after the menu closed");
      flat_runs.emplace_back(entry, exit);
    }
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      const auto &f = s.frames[i];
      const bool flat = std::any_of(flat_runs.begin(), flat_runs.end(), [i](const auto &run) {
        return i >= run.first && i < run.second;
      });
      require(f.layer_proven == (i >= proven_from), "The frame at " + std::to_string(f.now_ms) + " did not read the proof as the ledger held it");
      require(f.source == (flat ? 8u : 0u) && f.flat() == flat && f.decision.h1 == flat, "A Stellar Blade SDR frame at " + std::to_string(f.now_ms) + " was flat outside a visit's held verdict, or not flat inside it");
      require(flat || (f.decision.none_reason == ui_no_mask::layer_aside && f.decision.refused == candidate::layer), "A frame without the held verdict did not set the pre-UI layer aside");
    }
    const auto c = session.counters();
    require(c[ui_counter::scene_entered] == 3 && c[ui_counter::scene_released] == 3 && c[ui_counter::full_d_visible] == 3 && c[ui_counter::full_d_visible] <= c[ui_counter::scene_released] && !c[ui_counter::scene_refuted], "The three visits were not counted entered and released three times");
    require(session.stored() == proof && session.pre_ui_proven(p.signatures.of(kind::ui_layer)) && !session.accepts(p.signatures.of(kind::ui_layer)) && c[ui_counter::trust_earned] == 1, "The session did not end with the layer's proof as its only key");
    for (const auto event : {ui_counter::trust_revoked_exact, ui_counter::trust_revoked_declared, ui_counter::trust_lapsed, ui_counter::trust_restored, ui_counter::trust_discarded, ui_counter::trust_forgotten}) {
      require(!c[event], "The Stellar Blade SDR stream counted an acceptance event other than the proof");
    }
    check_counters(s, "Stellar Blade SDR menu visits");
  }

  void h1_stellar_blade_sdr_fg_suspended() {
    // Stellar Blade in SDR with FG set to 2x, on one identity (the FG session
    // below adds the viewport change): gameplay offers the HUD-less tag (an
    // inexact pair whose change set is the HUD) beside the cleared output
    // target (the scene image, V1-invalid) and current alpha (dump 476). The
    // HUD-less image is the offer's pre-UI image, so nothing claims and no
    // sample measures D; the layer equals the presented frame on 99.82% of
    // pixels (texel 11), which proves its signature 2 s into gameplay. The
    // settings menu suspends FG, so it offers the layer and current alpha
    // only (the recorded game3d_50264_218658377782962 counts): flat from the
    // poll of its second hidden sample; FG-on gameplay is 3D from its first
    // frame (no claim), and its first measured sample, visible, releases the
    // holds. A session that opens the menu before any gameplay stays 3D: the
    // layer is unproven, so nothing claims and nothing measures (S2b held the
    // hidden verdict there from the V1-invalid layer's own claim).
    const auto menu = recorded_frames({recorded::sb_sdr_menu, recorded::sb_sdr_menu_proven, recorded::sb_sdr_menu_held, recorded::sb_sdr_menu_held_unproven});
    const auto play = recorded_frames({recorded::sb_sdr_fg_play, recorded::sb_sdr_fg_play_proven});
    const auto fg_on = sb_sdr_present(candidate::layer | candidate::current | candidate::hudless), suspended = sb_sdr_present(candidate::layer | candidate::current);
    const auto layer = fg_on.signatures.of(kind::ui_layer);
    {
      alpha_auto_policy session;
      sequence s(session, [&](const gpu_inputs &in) {
        return (in.bits & candidate::hudless) ? play(in) : menu(in);
      });
      run(s, fg_on, 10000, 12500);
      require(session.pre_ui_proven(layer) && session.stored() == sb_sdr_proof(), "FG-on gameplay did not prove the layer by its pixels");
      for (const auto &sample : s.samples) {
        require(!measured(sample) && !sample.evidence.claims && !sample.source_kind && pre_ui_matching(sample), "FG-on gameplay claimed, measured or decided, or its layer did not match");
      }
      const std::size_t menu_from = s.frames.size();
      run(s, suspended, 12500, 14000);
      const std::size_t play_from = s.frames.size();
      run(s, fg_on, 14000, 15500);
      std::size_t entry = 0, exit = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        if (s.observed[i].entered && !entry) {
          entry = poll_of(s, s.samples[i].sample_tick_ms);
        }
        if (s.observed[i].released && !exit) {
          exit = poll_of(s, s.samples[i].sample_tick_ms);
        }
      }
      require(entry > menu_from && exit > play_from && s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven), "The FG-suspended menu did not hold both verdicts with the proof");
      require(s.frames[entry].now_ms - 12500 <= 3 * sample_interval_ms + 2 * 16, "The FG-suspended menu entered later than its second hidden sample");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        // Back in FG-on gameplay the offer's pre-UI image is the HUD-less
        // image, which changed on far fewer than 90% of pixels: no claim, so
        // 3D from the first gameplay frame although the release comes later.
        const bool flat = i >= entry && i < play_from;
        require(f.source == (flat ? 8u : 0u) && f.flat() == flat, "An FG-suspended menu frame at " + std::to_string(f.now_ms) + " was flat outside its held verdict, or not flat inside it");
      }
      require(session.pre_ui_proven(layer) && s.trust.size() == 1 && !s.discards, "The menu visit withdrew the proof, or a sample was discarded");
      check_counters(s, "Stellar Blade SDR FG suspended");
    }
    {
      // Booted into the settings menu: no gameplay sample proved the layer.
      alpha_auto_policy session;
      sequence s(session, menu);
      run(s, suspended, 10000, 13000);
      for (const auto &f : s.frames) {
        require(!f.source && !f.layer_proven && !(f.scene_bits & (ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven)), "An unproven layer flattened the menu, or was pushed as proven");
      }
      for (const auto &sample : s.samples) {
        require(!measured(sample) && !sample.evidence.claims && !pre_ui_matching(sample), "The unproven menu claimed, measured or matched");
      }
      require(!s.scene_entered && !session.pre_ui_proven(layer) && session.stored().empty() && !s.guard.pre_ui.until, "The unproven menu held a verdict or earned the proof");
      check_counters(s, "Stellar Blade SDR menu before gameplay");
    }
  }

  void h1_stellar_blade_sdr_fg_session() {
    // The live failure fix 1 answers (S2b build, Stellar Blade SDR, FG set to
    // 2x, 07:19-07:21): the game suspends frame generation while a menu is
    // open. FG-on gameplay offers the layer, current alpha and the HUD-less
    // image (paired by Present counting) on every Present (viewport 1: depth
    // from Streamline), so every Present detects. Nothing claims, so no
    // sample measures D (the log's
    // evidence was invalid on its samples), yet the layer equals the
    // presented frame on 99.82% of pixels (dump 476, texel 11): the ledger
    // proves pre_ui:87:srgb by the first sample 2 s in. A menu suspends FG:
    // the offer is the layer and current alpha, and the depth moves to NGX
    // (viewport 0), an identity change that clears the scene guard but not
    // the proof. Each visit's first sample is a fade-in that reads the
    // presented frame visible (D 0.453) beside the layer (0.538) while their
    // pixels mismatch (35.28%): nothing changes. The next two read the
    // presented frame hidden and the layer visible: flat (8) from the poll of
    // the second hidden sample. Visits 1 and 2 end with FG on again (viewport
    // 1): 3D from the first gameplay frame, whose pre-UI image is the
    // HUD-less image again. Visit 3 ends with a moment of FG-off gameplay on
    // viewport 0 (dump 470), released by its first visible sample. Under S2b
    // the identity change cleared the layer's D proof and the fade-in
    // withdrew it, so no visit went flat.
    const auto fg_play = recorded_frames({recorded::sb_sdr_fg_play, recorded::sb_sdr_fg_play_proven});
    const auto play = recorded_frames({recorded::sb_sdr_play, recorded::sb_sdr_play_proven});
    const auto menu = recorded_frames({recorded::sb_sdr_menu, recorded::sb_sdr_menu_proven, recorded::sb_sdr_menu_held, recorded::sb_sdr_menu_held_unproven});

    struct visit {
      std::uint64_t open, close;
      bool fg_off_exit;
    };

    constexpr std::array<visit, 3> visits {{{12600, 14000, false}, {15500, 16800, false}, {18300, 19600, true}}};
    constexpr std::uint64_t fg_off_until = 19900, end = 21000;
    const auto visit_at = [&visits](std::uint64_t tick) -> const visit * {
      for (const auto &v : visits) {
        if (tick >= v.open && tick < v.close) {
          return &v;
        }
      }
      return nullptr;
    };
    alpha_auto_policy session;
    std::vector<std::string> heard;
    session.on_change([&](const std::string &accepted) {
      heard.push_back(accepted);
    });
    const auto proof = sb_sdr_proof();
    sequence s(session, [&](const gpu_inputs &in) {
      if (in.bits & candidate::hudless) {
        return fg_play(in);
      }
      return visit_at(in.now_ms) ? menu(in) : play(in);
    });
    // The fade-in: each visit's first measured sample reads the presented
    // frame visible beside the layer.
    std::vector<std::uint64_t> fade_ins;
    s.sample_edit = [&](texels &t, std::uint64_t tick, bool evidence) {
      const auto *v = visit_at(tick);
      if (!v || !evidence || (!fade_ins.empty() && fade_ins.back() >= v->open)) {
        return;
      }
      fade_ins.push_back(tick);
      t[word::scene_state] = scene_state(true, scene_verdict::visible);
      t[word::scene_d] = float_bits(.453f);
      t[word::pre_ui_scene_d] = float_bits(.538f);
    };
    const auto fg_on = sb_sdr_present(candidate::layer | candidate::current | candidate::hudless, 1);
    const auto suspended = sb_sdr_present(candidate::layer | candidate::current, 0);
    fg_pacer pacer(1);
    std::size_t generated = 0;
    for (std::uint64_t now = 10000; now < end;) {
      if (visit_at(now) || (now >= visits[2].close && now < fg_off_until)) {
        auto p = suspended;
        p.now_ms = now;
        s.step(p);
        now += 16;
        continue;
      }
      auto step = pacer.next(fg_on, fg_on.offered, 1, 1);
      step.p.now_ms = now;
      const auto &r = s.step(step.p);
      if (!step.real) {
        ++generated;
        require(r.detected && !r.flat(), "A generated FG-on Present did not detect, or was flat");
      }
      now += 8;
    }
    const auto first = s.samples.front().sample_tick_ms;
    const auto earned = first_sample_from(s, first + alpha_trust_span_ms);
    require(generated > 100 && s.trust.size() == 1 && s.trust[0].tick == earned && s.trust[0].accepted == proof && heard == std::vector<std::string> {proof} && earned < visits[0].open, "FG-on gameplay did not prove the layer by its pixels 2 s in, before the first menu");
    for (const auto &sample : s.samples) {
      if (sample.evidence.candidates & candidate::hudless) {
        require(!measured(sample) && !sample.evidence.claims && pre_ui_matching(sample), "FG-on gameplay claimed, measured D or did not match");
      }
    }
    const auto proven_from = poll_of(s, earned);
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      require(!s.frames[i].detected || s.frames[i].layer_proven == (i >= proven_from), "A detection did not read the proof as the ledger held it");
    }
    require(fade_ins.size() == visits.size(), "A visit had no fade-in sample");
    std::vector<std::pair<std::size_t, std::size_t>> flat_runs;
    for (std::size_t k = 0; k != visits.size(); ++k) {
      const auto &v = visits[k];
      std::vector<std::size_t> menu_samples;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        const auto tick = s.samples[i].sample_tick_ms;
        if (tick >= v.open && tick < v.close) {
          menu_samples.push_back(i);
        }
      }
      require(menu_samples.size() > 3 && s.samples[menu_samples[0]].sample_tick_ms == fade_ins[k], "A visit's first sample was not its fade-in");
      const auto &fade = s.samples[menu_samples[0]];
      const auto &fade_observed = s.observed[menu_samples[0]];
      require(fade.evidence.scene.valid && fade.evidence.scene.verdict == scene_verdict::visible && fade.evidence.pre_ui_scene.valid && fade.evidence.claims == ui_detection::claim_pre_ui && !pre_ui_matching(fade) && !fade_observed.entered && !fade_observed.released && !fade_observed.refuted, "A visit's fade-in sample changed something, or matched");
      require(s.samples[menu_samples[1]].evidence.scene.verdict == scene_verdict::hidden && !s.observed[menu_samples[1]].entered && s.samples[menu_samples[2]].evidence.scene.verdict == scene_verdict::hidden && s.observed[menu_samples[2]].entered, "A visit did not enter at its second hidden sample");
      const auto entry = poll_of(s, s.samples[menu_samples[2]].sample_tick_ms);
      require(s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven), "A visit did not hold both verdicts with the proof from its entry");
      std::size_t exit = 0;
      if (!v.fg_off_exit) {
        // FG on again, viewport 1: the guard clears and the HUD-less image is
        // the pre-UI image, so 3D from the first gameplay frame.
        for (std::size_t i = entry; i != s.frames.size() && !exit; ++i) {
          exit = s.frames[i].now_ms >= v.close ? i : 0;
        }
        require(exit, "FG-on gameplay did not follow a visit");
      } else {
        std::size_t release = 0;
        for (std::size_t i = 0; i != s.samples.size() && !release; ++i) {
          release = s.samples[i].sample_tick_ms >= v.close ? i : 0;
        }
        require(release && s.samples[release].sample_tick_ms < fg_off_until && s.observed[release].released && s.samples[release].source_kind == 8u && s.samples[release].evidence.scene.verdict == scene_verdict::visible, "The first FG-off gameplay sample did not release the visit");
        exit = poll_of(s, s.samples[release].sample_tick_ms);
        require(s.frames[exit].now_ms - v.close <= sample_interval_ms + 2 * 16 && s.frames[exit].scene_bits == ui_detection::per_frame_pre_ui_proven, "The FG-off release came later than one sample interval after the menu closed");
      }
      flat_runs.emplace_back(entry, exit);
    }
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      const auto &f = s.frames[i];
      const bool flat = std::any_of(flat_runs.begin(), flat_runs.end(), [i](const auto &run) {
        return i >= run.first && i < run.second;
      });
      require(f.flat() == flat && (!flat || (f.source == 8u && f.decision.h1)), "A frame at " + std::to_string(f.now_ms) + " was flat outside a visit's held verdicts, or not flat inside them");
    }
    const auto c = session.counters();
    require(c[ui_counter::scene_entered] == 3 && c[ui_counter::scene_released] == 1 && c[ui_counter::full_d_visible] == 1 && !c[ui_counter::scene_refuted], "The visits were not counted entered three times and released once by a sample");
    require(session.stored() == proof && c[ui_counter::trust_earned] == 1 && !c[ui_counter::trust_lapsed] && !c[ui_counter::trust_forgotten], "The proof was withdrawn, or earned more than once");
    check_counters(s, "Stellar Blade SDR FG session");
  }

  // Synthetic Stellar Blade SDR gameplay the layer does not show (its pixels
  // equal the presented frame on half the frame): the layer without
  // coverage (V1-invalid) beside opaque current alpha, the presented frame
  // visible with valid evidence (testable for a restored proof's clock) or
  // with invalid evidence.
  texels sb_mismatching_play(const gpu_inputs &in, bool valid) {
    synthetic f;
    f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(500, 1000);
    if (valid) {
      f.scene(.5f, .5f);
    }
    return f.words(in);
  }

  // The tick of the sample at which a restored pre-UI proof lapses (A3, fix
  // 1), zero when none does: its reconfirm clock counts the gaps between
  // consecutive testable samples (the layer offered without coverage while
  // the presented frame's evidence is valid and visible), each capped at
  // alpha_trust_reconfirm_gap_ms, and the proof lapses at the testable sample
  // by which it has counted alpha_trust_reconfirm_ms. The stream must not
  // re-earn it.
  std::uint64_t pre_ui_lapse_tick(const sequence &s, std::uint64_t from) {
    std::uint64_t counted = 0, last = 0;
    for (const auto &sample : s.samples) {
      if (sample.sample_tick_ms < from) {
        continue;
      }
      const auto &e = sample.evidence;
      const bool testable = (e.candidates & candidate::layer) && !e.layer_covered && measured(sample) && e.scene.valid && e.scene.verdict == scene_verdict::visible;
      if (!testable) {
        continue;
      }
      if (last && sample.sample_tick_ms > last) {
        counted += std::min(sample.sample_tick_ms - last, alpha_trust_reconfirm_gap_ms);
      }
      last = sample.sample_tick_ms;
      if (counted >= alpha_trust_reconfirm_ms) {
        return sample.sample_tick_ms;
      }
    }
    return 0;
  }

  void h1_pre_ui_proof_across_sessions() {
    // The proof is a ledger key (pre_ui:87:srgb) that TrustedUISources keeps
    // with the other keys. A second session restores it provisionally: a
    // menu opened before any gameplay goes flat at its second hidden sample.
    // It lapses after 60 s of testable time (the layer offered without
    // coverage while the presented frame reads visible with valid evidence)
    // without a matching sample, never while only menus or samples with
    // invalid evidence arrive; 3 matching samples over 2 s confirm it, after
    // which it no longer lapses. Forget clears it, the listener hears the
    // change, and the next menu stays 3D until gameplay earns it again.
    const auto proof = sb_sdr_proof();
    const auto layer = sb_sdr_signatures().of(kind::ui_layer);
    const auto menu = recorded_frames({recorded::sb_sdr_menu, recorded::sb_sdr_menu_proven, recorded::sb_sdr_menu_held, recorded::sb_sdr_menu_held_unproven});
    const auto play = recorded_frames({recorded::sb_sdr_play, recorded::sb_sdr_play_proven});
    enum class showing {
      menu,
      play,
      mismatch,
      mismatch_invalid
    };
    auto shown = showing::menu;
    const gpu_model model = [&](const gpu_inputs &in) {
      switch (shown) {
        case showing::menu:
          return menu(in);
        case showing::play:
          return play(in);
        default:
          return sb_mismatching_play(in, shown == showing::mismatch);
      }
    };
    // Menus on viewport 0 (FG suspended), gameplay on viewport 1: each change
    // is an identity change. Returns the index of the phase's first frame.
    const auto show = [&](sequence &s, showing what, std::uint64_t from, std::uint64_t to) {
      shown = what;
      const auto first = s.frames.size();
      run(s, sb_sdr_present(candidate::layer | candidate::current, what == showing::menu ? 0u : 1u), from, to);
      return first;
    };
    // A menu from frame `first`: 3D until the poll of its second sample, flat
    // (8) from there.
    const auto menu_goes_flat = [](const sequence &s, std::size_t first, const char *what) {
      std::vector<std::uint64_t> ticks;
      for (const auto &sample : s.samples) {
        if (sample.sample_tick_ms >= s.frames[first].now_ms) {
          ticks.push_back(sample.sample_tick_ms);
        }
      }
      require(ticks.size() > 2, std::string(what) + ": too few samples");
      const auto entry = poll_of(s, ticks[1]);
      for (std::size_t i = first; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.layer_proven && f.flat() == (i >= entry) && f.source == (i >= entry ? 8u : 0u), std::string(what) + ": not flat exactly from the poll of its second hidden sample");
      }
    };
    const auto menu_stays_3d = [](const sequence &s, std::size_t first, const char *what) {
      for (std::size_t i = first; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(!f.flat() && !f.source && !f.layer_proven && !(f.scene_bits & (ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven)), std::string(what) + ": flattened, or pushed the proof");
      }
    };
    {
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      restore(session, proof, 1);
      require(session.pre_ui_proven(layer) && !session.accepts(layer) && heard.empty(), "The restored proof did not read as proven, or was heard");
      sequence s(session, model);
      menu_goes_flat(s, show(s, showing::menu, 10000, 11000), "A menu before any gameplay under the restored proof");
      // 70 s of gameplay whose evidence is invalid, then a menu: no testable
      // time, so no lapse.
      show(s, showing::mismatch_invalid, 11000, 81000);
      menu_goes_flat(s, show(s, showing::menu, 81000, 82000), "A menu after 70 s without testable time");
      require(session.stored() == proof && !session.counters()[ui_counter::trust_lapsed], "The restored proof lapsed without testable time");
      // Testable gameplay that never matches: 30 s, a 10 s menu (paused),
      // then until the clock has run 60 s.
      show(s, showing::mismatch, 82000, 112000);
      menu_goes_flat(s, show(s, showing::menu, 112000, 122000), "A menu after 30 s of testable time");
      require(session.stored() == proof, "The restored proof lapsed after 30 s of testable time");
      show(s, showing::mismatch, 122000, 160000);
      const auto lapsed = pre_ui_lapse_tick(s, 82000);
      require(lapsed > 150000 && lapsed < 154000, "The testable time did not reach 60 s about 30 s into the second testable run");
      const auto c = session.counters();
      require(s.trust.size() == 1 && s.trust[0].tick == lapsed && s.trust[0].accepted.empty() && heard == std::vector<std::string> {""} && c[ui_counter::trust_lapsed] == 1 && c[ui_counter::trust_restored] == 1 && !c[ui_counter::trust_earned], "The restored proof did not lapse at 60 s of testable time, or was not heard and counted once");
      // Lapsed, the next menu stays 3D: nothing claims, nothing measures.
      menu_stays_3d(s, show(s, showing::menu, 160000, 161000), "The menu after the lapse");
      check_counters(s, "restored pre-UI proof lapsing");
    }
    {
      alpha_auto_policy session;
      std::vector<std::string> heard;
      session.on_change([&](const std::string &accepted) {
        heard.push_back(accepted);
      });
      restore(session, proof, 1);
      sequence s(session, model);
      menu_goes_flat(s, show(s, showing::menu, 10000, 11000), "A menu before any gameplay under the restored proof");
      // Gameplay showing the layer confirms the proof: the provisional entry
      // becomes earned (heard by nobody: stored() is unchanged).
      const auto before = s.samples.size();
      show(s, showing::play, 11000, 14000);
      const auto confirmed = first_sample_from(s, s.samples.at(before).sample_tick_ms + alpha_trust_span_ms);
      require(confirmed && confirmed < 14000 && s.trust.empty() && heard.empty() && session.stored() == proof && session.counters()[ui_counter::trust_earned] == 1, "Matching gameplay did not confirm the restored proof, or was heard");
      // Confirmed, it no longer lapses: 70 s of testable gameplay that never
      // matches.
      show(s, showing::mismatch, 14000, 84000);
      require(session.stored() == proof && !session.counters()[ui_counter::trust_lapsed], "The confirmed proof lapsed");
      // Forget clears it, persists none and counts it once.
      require(s.forget() == proof && session.stored().empty() && !session.pre_ui_proven(layer) && heard == std::vector<std::string> {""} && session.counters()[ui_counter::trust_forgotten] == 1, "Forget did not clear the proof, persist none or count it");
      menu_stays_3d(s, show(s, showing::menu, 84000, 85000), "The menu after Forget");
      const auto again = s.samples.size();
      show(s, showing::play, 85000, 88000);
      const auto reearned = first_sample_from(s, s.samples.at(again).sample_tick_ms + alpha_trust_span_ms);
      require(s.trust.size() == 1 && s.trust[0].tick == reearned && s.trust[0].accepted == proof && heard == std::vector<std::string> {"", proof} && session.counters()[ui_counter::trust_earned] == 2, "Gameplay after Forget did not earn the proof again 2 s in");
      menu_goes_flat(s, show(s, showing::menu, 88000, 89000), "A menu after the proof was earned again");
      check_counters(s, "confirmed pre-UI proof and Forget");
    }
  }

  void h1_effects_target_is_never_proven() {
    // A bloom or effects target the census took for the layer: cleared,
    // colour without alpha (no coverage, V1-invalid), reading D visible and
    // within 0.03 of the presented frame in gameplay (0.50 and 0.51: the S2b
    // D proof accepted such a target), but equal to the presented frame on
    // only 60% of pixels in every sample. It is never proven, so it never
    // claims, nothing measures, and a hidden menu over it (presented D -0.01,
    // the target 0.55) stays 3D.
    alpha_auto_policy session;
    sequence s(session, [](const gpu_inputs &in) {
      const bool menu = in.now_ms >= 20000 && in.now_ms < 25000;
      synthetic f;
      f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000);
      if (menu) {
        f.scene(-.01f, .55f).pre_ui_pixels(300, 1000);
      } else {
        f.scene(.5f, .51f).pre_ui_pixels(600, 1000);
      }
      return f.words(in);
    });
    const auto p = sb_sdr_present(candidate::layer | candidate::current);
    run(s, p, 10000, 30000);
    for (const auto &sample : s.samples) {
      const auto &e = sample.evidence;
      require(!e.scene.ran && !e.claims && !pre_ui_matching(sample), "The effects target claimed, matched or measured");
    }
    for (const auto &f : s.frames) {
      require(!f.flat() && !f.source && !f.layer_proven && !(f.scene_bits & (ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven)), "The effects target was proven or flattened");
    }
    require(session.stored().empty() && !session.pre_ui_proven(p.signatures.of(kind::ui_layer)) && !s.guard.pre_ui.until && !s.scene_entered, "The effects target earned the proof, or a verdict held");
    check_counters(s, "effects target");
  }

  void h1_pre_ui_proof_keys() {
    // Only a lit layer without coverage that equals the presented frame
    // proves its signature, and each format and colour space is a key of its
    // own.
    const auto layer = sb_sdr_signatures().of(kind::ui_layer);
    {
      // Black frames over a transparent real UI layer (V1-valid, alpha 0
      // everywhere): equal on every pixel, lit on none. Never proven.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(1000, 0);
        return f.words(in);
      });
      run(s, sb_sdr_present(candidate::layer | candidate::current), 10000, 20000);
      for (const auto &sample : s.samples) {
        require((sample.evidence.valid_bits & candidate::layer) && !sample.evidence.layer_covered && sample.evidence.pre_ui_match == sample.pixels && !sample.evidence.pre_ui_image_lit && !pre_ui_matching(sample), "The black stream's layer is not a valid transparent layer equal to the frame");
      }
      require(session.stored().empty() && !session.pre_ui_proven(layer), "Black frames over a transparent UI layer proved it");
      check_counters(s, "black frames over a transparent layer");
    }
    {
      // A layer with coverage (a selective HUD), equal to the presented frame
      // elsewhere: its own run accepts it as UI coverage (A1); it is never
      // the pre-UI scene image.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 50).alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(990, 900);
        return f.words(in);
      });
      run(s, sb_sdr_present(candidate::layer | candidate::current), 10000, 14000);
      require(session.stored() == layer.key() && !session.pre_ui_proven(layer), "A layer with coverage earned a pre-UI proof, or its own run was not accepted");
      check_counters(s, "layer with coverage");
    }
    {
      // The same lit, matching layer in SDR and then in PQ (an HDR switch:
      // a new epoch and colour space): each colour space earns its own key,
      // and another format is another key again.
      alpha_auto_policy session;
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(.5f, .5f).pre_ui_pixels(990, 900);
        return f.words(in);
      });
      const auto sdr = sb_sdr_present(candidate::layer | candidate::current);
      run(s, sdr, 10000, 13000);
      const auto srgb_key = ui_selection::pre_ui_key(layer);
      require(session.stored() == srgb_key.key() && session.pre_ui_proven(layer), "SDR gameplay did not earn the SDR key");
      auto hdr = sdr;
      hdr.epoch = 2;
      hdr.signatures.color_space = pq;
      const auto hdr_layer = hdr.signatures.of(kind::ui_layer);
      hdr.now_ms = 13000;
      const auto &crossed = s.step(hdr);
      require(crossed.detected && !crossed.layer_proven && !(crossed.gpu.per_frame & ui_detection::per_frame_pre_ui_proven), "The SDR proof crossed into PQ");
      run(s, hdr, 13016, 16000);
      require(session.pre_ui_proven(hdr_layer) && session.stored() == stored_of({srgb_key, ui_selection::pre_ui_key(hdr_layer)}) && session.stored() == "pre_ui:87:srgb,pre_ui:87:pq", "PQ gameplay did not earn its own key beside the SDR one");
      auto other = sdr;
      other.epoch = 3;
      other.signatures.set(kind::ui_layer, 28);
      other.now_ms = 16000;
      require(!s.step(other).layer_proven, "Another layer format read the proof of format 87");
      auto back = sdr;
      back.epoch = 4;
      back.now_ms = 16016;
      const auto &returned = s.step(back);
      require(returned.layer_proven && (returned.gpu.per_frame & ui_detection::per_frame_pre_ui_proven), "Back in SDR the SDR proof did not apply from the first frame");
      check_counters(s, "pre-UI proof keys");
    }
  }

  void h1_dark_gameplay_never_flat() {
    present p;
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    // The layer's pre-UI proof as an earlier session left it (provisional).
    const auto proof = ui_selection::pre_ui_key(p.signatures.of(kind::ui_layer)).key();
    const auto never_flat = [](const sequence &s, const char *what) {
      for (const auto &f : s.frames) {
        require(!f.flat() && !f.source && !(f.scene_bits & ui_detection::per_frame_pre_ui_visible), std::string(what) + " flattened, or held the pre-UI image visible");
      }
      require(!s.guard.pre_ui.until, std::string(what) + " entered the pre-UI hold");
    };
    {
      // A dark scene behind a V1-invalid cleared layer whose signature an
      // earlier session proved (the Stellar Blade SDR inputs): the presented
      // frame and the pre-UI image both read hidden for 10 s. The layer
      // equals the presented frame but is lit on no pixel, so it never
      // confirms the proof, and no sample reads visible, so its clock never
      // runs. The hidden hold enters, but the pre-UI claim never acts.
      alpha_auto_policy session;
      restore(session, proof, 1);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(-.01f, .05f).pre_ui_pixels(1000, 0);
        return f.words(in);
      });
      run(s, p, 10000, 20000);
      never_flat(s, "A dark scene with both images hidden");
      const auto c = session.counters();
      require(s.scene_entered == 1 && s.guard.hidden.held(s.frames.back().now_ms) && s.frames.back().layer_proven, "The dark scene's presented frame did not hold hidden beside the proven layer");
      require(session.stored() == proof && !c[ui_counter::trust_earned] && !c[ui_counter::trust_lapsed], "The dark image confirmed the proof, or its clock ran");
      check_counters(s, "dark gameplay");
    }
    {
      // Correlated grain behind the proven layer, which equals the presented
      // frame (and confirms the proof): the pre-UI image reads within 0.05
      // of the presented frame, which reads between 0.10 and 0.30. A pre-UI
      // image visible (0.25 or more) beside a hidden presented frame (below
      // 0.15) needs a contradiction of 0.10, so the pre-UI claim never acts.
      alpha_auto_policy session;
      restore(session, proof, 1);
      lcg random {2024};
      std::uint32_t hidden_samples = 0;
      sequence s(session, [&random](const gpu_inputs &in) {
        const float presented = .10f + float(random.next(2001)) / 10000.f;
        const float pre_ui = presented + (float(random.next(1001)) - 500.f) / 10000.f;
        synthetic f;
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(presented, pre_ui).pre_ui_pixels(990, 900);
        return f.words(in);
      });
      run(s, p, 10000, 40000);
      for (const auto &sample : s.samples) {
        hidden_samples += sample.evidence.scene.valid && sample.evidence.scene.verdict == scene_verdict::hidden ? 1 : 0;
      }
      never_flat(s, "Correlated grain");
      require(hidden_samples > 20 && s.scene_entered > 0 && session.counters()[ui_counter::trust_earned] == 1, "The grain stream never read the presented frame hidden, or did not confirm the proof");
      check_counters(s, "correlated grain");
    }
    {
      // A scene buffer from before fog or grain (V1-invalid, colour without
      // alpha) while fog drifts the presented frame down to 0.05 over 3 s and
      // lifts after 5 s, the buffer staying at 0.5. Behind grain that
      // keeps its pixels from the presented frame (60% equal) it is never
      // proven, so it never claims and nothing measures or flattens.
      // ACCEPTED RISK of the approved pixel proof (fix 1): the same buffer
      // before the fog sets in equals the presented frame (99%), so it is
      // proven 2 s in, and a mismatch never withdraws the proof; once the fog
      // reads the presented frame hidden beside the visible buffer, H1 (d)
      // flattens from the poll of the second hidden sample until the first
      // visible sample after the fog lifts. Its mitigations are the
      // two-sample entry, the release on a visible sample, a restored proof's
      // 60 s lapse and Forget.
      for (const bool grain : {true, false}) {
        alpha_auto_policy session;
        sequence s(session, [grain](const gpu_inputs &in) {
          const float in_fog = in.now_ms < 12000 ? 0.f : std::min(1.f, float(in.now_ms - 12000) / 3000.f);
          const float fade = in.now_ms < 20000 ? in_fog : std::max(0.f, 1.f - float(in.now_ms - 20000) / 1000.f);
          const float buffer = .5f, offset = grain ? .05f : 0.f, presented = buffer - offset - fade * (.45f - offset);
          const auto matched = grain ? 600u : std::uint32_t(990.f - 700.f * fade);
          synthetic f;
          f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(presented, buffer).pre_ui_pixels(matched, 1000);
          return f.words(in);
        });
        run(s, p, 10000, 23000);
        if (grain) {
          never_flat(s, "A scene buffer behind grain");
          for (const auto &sample : s.samples) {
            require(!measured(sample) && !sample.evidence.claims, "The unproven buffer behind grain claimed or measured");
          }
          require(!s.scene_entered && session.stored().empty(), "The buffer behind grain was proven, or a verdict held");
          check_counters(s, "scene buffer behind grain");
          continue;
        }
        require(session.stored() == proof && s.trust.size() == 1 && s.trust[0].tick < 12500, "The pre-fog buffer was not proven by its pixels 2 s in");
        std::size_t entered = 0, released = 0, hidden_seen = 0;
        for (std::size_t i = 0; i != s.samples.size(); ++i) {
          const auto &e = s.samples[i].evidence;
          hidden_seen += e.scene.valid && e.scene.verdict == scene_verdict::hidden ? 1 : 0;
          if (s.observed[i].entered && !entered) {
            entered = i;
            require(hidden_seen == 2, "The fog entered on other than its second hidden sample");
          }
          if (s.observed[i].released && !released) {
            released = i;
          }
        }
        require(entered && released > entered && s.samples[released].sample_tick_ms >= 20000, "The fog did not enter H1 (d), or the lifting fog did not release it");
        const auto entry = poll_of(s, s.samples[entered].sample_tick_ms), exit = poll_of(s, s.samples[released].sample_tick_ms);
        for (std::size_t i = 0; i != s.frames.size(); ++i) {
          const bool flat = i >= entry && i < exit;
          require(s.frames[i].flat() == flat && (!flat || s.frames[i].source == 8u), "The proven pre-fog buffer flattened other than from the second hidden sample to the release");
        }
        check_counters(s, "fog after a proven scene buffer (accepted risk)");
      }
    }
    {
      // Resident Evil Requiem: an accepted UI color tag deciding a partial
      // HUD beside accepted presented alpha that is opaque everywhere. The
      // declared-alpha block keeps the presented alpha out of S1 and of claim
      // (a), so a dark room where every sample reads hidden for 10 s holds
      // the hidden verdict and stays 3D with the tag's mask.
      alpha_auto_policy session;
      const auto signatures = signatures_in(srgb);
      restore(session, stored_of({signatures.of(kind::ui_color), signatures.of(kind::current)}), 2);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_color, 15).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(.05f, .05f);
        return f.words(in);
      });
      present q;
      q.offered = candidate::ui_color | candidate::current;
      run(s, q, 10000, 20000);
      for (const auto &f : s.frames) {
        require(f.source == 2 && f.covered == 15 && !f.decision.h1 && !f.decision.claims, "Dark gameplay with an accepted tag and opaque presented alpha flattened or claimed");
      }
      for (const auto &sample : s.samples) {
        require(!measured(sample), "Dark gameplay without a claim measured");
      }
      check_counters(s, "accepted tag beside opaque presented alpha");
    }
    {
      // Dead Space-like: diegetic UI only, so no informative claim (an
      // unaccepted current alpha over the whole frame, not opaque), with D
      // near the verdict bounds: nothing measures.
      alpha_auto_policy session;
      lcg random {77};
      sequence s(session, [&random](const gpu_inputs &in) {
        constexpr float near_bounds[] {.14f, .16f, .24f, .26f};
        const float d = near_bounds[random.next(4)];
        synthetic f;
        f.alpha(kind::current, 1000).opaque(kind::current, 500).scene(d, d);
        return f.words(in);
      });
      present q;
      q.offered = candidate::current;
      run(s, q, 10000, 20000);
      for (const auto &sample : s.samples) {
        require(sample.evidence.claims == 0 && !measured(sample), "The Dead Space-like stream claimed or measured");
      }
      never_flat(s, "The Dead Space-like stream");
      require(!s.guard.hidden.until && !s.guard.hidden.last && !s.scene_entered, "The Dead Space-like stream acted on evidence");
      check_counters(s, "Dead Space-like");
    }
  }

  void h1_overrides_a_partial_winner() {
    const auto signatures = signatures_in(srgb);
    const auto layer = signatures.of(kind::ui_layer);
    {
      // An accepted UIAlpha decides 0.25% (the HUD) while an unaccepted,
      // V1-valid cleared layer covers the frame opaque (a full menu, claim
      // (b)). Over a hidden scene the held verdict overrides the partial
      // winner (H1 has no '!source' gate): 1 -> 8 from the poll of the second
      // measured hidden sample. One visible sample releases it and refutes
      // the layer's signature: back to 1. Samples showing the layer at 50%
      // opaque restore it, and the next hidden samples override again.
      alpha_auto_policy session;
      restore(session, signatures.of(kind::ui_alpha).key(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.c.pixels = 4000;
        const bool overlay = in.now_ms >= 13000 && in.now_ms < 13300;
        f.alpha(kind::ui_alpha, 10).alpha(kind::ui_layer, 4000).opaque(kind::ui_layer, overlay ? 2000 : 4000).scene(-.02f, -.02f);
        return f.words(in);
      });
      bool visible_sent = false;
      s.sample_edit = [&visible_sent](texels &t, std::uint64_t tick, bool evidence) {
        if (evidence && tick >= 11500 && !visible_sent) {
          visible_sent = true;
          t[word::scene_state] = scene_state(true, scene_verdict::visible);
          t[word::scene_d] = float_bits(.5f);
        }
      };
      present p;
      p.offered = candidate::ui_alpha | candidate::layer | candidate::current;
      p.layer_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 15000);
      std::size_t visible = 0, restored = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        if (s.samples[i].evidence.scene.valid && s.samples[i].evidence.scene.verdict == scene_verdict::visible) {
          visible = i;
        }
        if (!restored && s.samples[i].sample_tick_ms >= 13000) {
          restored = i;
        }
      }
      const auto entry = poll_of(s, s.samples[2].sample_tick_ms), release = poll_of(s, s.samples[visible].sample_tick_ms), restore_poll = poll_of(s, s.samples[restored].sample_tick_ms);
      require(s.observed[2].entered && visible > 2 && s.observed[visible].released && s.observed[visible].refuted == 1 && restored > visible, "The override did not enter, release and restore in order");
      std::size_t rearmed = 0;
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.decision.s1_source == 1 && f.gpu.accepted == candidate::ui_alpha, "The accepted UIAlpha was not the S1 winner");
        if (i < entry || (i >= release && i < restore_poll)) {
          require(f.source == 1 && f.covered == 10 && !f.decision.h1, "The partial winner was overridden without an acting held verdict");
        } else if (i < release) {
          require(f.flat() && f.source == 8 && f.decision.h1, "H1 did not override the accepted partial winner");
        } else if (f.source == 8) {
          rearmed = rearmed ? rearmed : i;
        }
        if (i >= release && i < restore_poll) {
          require(f.scene_bits & (candidate::layer << ui_detection::per_frame_refuted_shift), "The refuted layer was not pushed as refuted");
        }
      }
      require(rearmed > restore_poll && s.frames.back().source == 8 && !s.guard.refuted(layer), "The restored layer did not override again");
      const auto c = session.counters();
      require(c[ui_counter::full_d_visible] == 1 && c[ui_counter::scene_released] == 1 && c[ui_counter::scene_refuted] == 1 && c[ui_counter::scene_entered] == 2 && !session.accepts(layer), "The override's release and refutation were not counted once, or the layer was accepted");
      check_counters(s, "partial winner overridden");
    }
    {
      // An accepted opaque-full winner that is not opaque on every pixel:
      // E33 Load Game's accepted Backbuffer (99.9998% opaque) over a hidden
      // scene is flat from P1, then from H1 (8) from the poll of the second
      // hidden sample, so no pixel of it is warped by the hidden depth.
      alpha_auto_policy session;
      restore(session, signatures.of(kind::backbuffer).key(), 1);
      sequence s(session, recorded_frames({recorded::e33_load_accepted}));
      present p;
      p.offered = candidate::backbuffer;
      run(s, p, 10000, 13000);
      require(s.scene_entered == 1 && (s.frames.back().scene_bits & ui_detection::per_frame_scene_hidden), "The accepted Load Game Backbuffer's hidden scene was not held");
      std::size_t entry = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        if (s.observed[i].entered) {
          entry = poll_of(s, s.samples[i].sample_tick_ms);
        }
      }
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.flat() && f.decision.s1_source == 3 && f.decision.claims == candidate::backbuffer && f.source == (i < entry ? 3u : 8u) && f.decision.h1 == (i >= entry), "H1 did not take over an accepted winner with transparent pixels from its entry");
      }
      const auto c = session.counters();
      require(c[ui_counter::full_alpha] > 0 && c[ui_counter::full_d_hidden] > 0 && !c[ui_counter::full_d_visible], "Load Game's frames were not counted as whole-frame alpha, then as H1");
      check_counters(s, "accepted opaque-full winner");
    }
    {
      // An accepted alpha winner opaque on every pixel is flat already; H1
      // relabels it 8 from the hold's entry like any other winner (selection
      // revision 10: it pins at weight 1 either way, P1, so only the label
      // and the counters move from source 3 to 8).
      alpha_auto_policy session;
      restore(session, signatures.of(kind::backbuffer).key(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::backbuffer, 1000).opaque(kind::backbuffer, 1000).scene(.03f, .03f);
        return f.words(in);
      });
      present p;
      p.offered = candidate::backbuffer;
      run(s, p, 10000, 13000);
      require(s.scene_entered == 1, "The opaque winner's hidden scene was not held");
      std::size_t entry = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        if (s.observed[i].entered) {
          entry = poll_of(s, s.samples[i].sample_tick_ms);
        }
      }
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(f.flat() && f.decision.s1_source == 3 && f.decision.claims == candidate::backbuffer && f.source == (i < entry ? 3u : 8u) && f.decision.h1 == (i >= entry), "H1 did not relabel an accepted winner opaque on every pixel from its entry");
      }
      check_counters(s, "accepted winner opaque everywhere");
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
  // slots, the legacy trust bitmask, the HUD-less route's evidence and
  // scene_hold before S2b, no pre-UI pixel counts before fix 1) are read as
  // far as they map.
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
      std::uint32_t fed = 0, logged = 0, held_hidden = 0, h1 = 0;
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
              // A layer's pre-UI proof names the layer's format.
              if (parsed->source_kind == kind::pre_ui && !signatures.format[std::size_t(kind::ui_layer)]) {
                signatures.set(kind::ui_layer, parsed->format);
              }
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
        // The line's columns are the layer (never judged since selection
        // revision 10), Backbuffer and current alpha; sampled_declared_one_way
        // (revision 10) holds UIAlpha and the UI color tag. Evidence keeps
        // judged_kinds order.
        if (const auto one_way = field_group(line, "sampled_one_way"); !one_way.empty()) {
          const auto strong = field_quad(field_text(one_way, "strong")), contradicted = field_quad(field_text(one_way, "contradicted"));
          for (std::size_t i = 0; i != 2; ++i) {
            evidence.strong[2 + i] = strong[1 + i];
            evidence.contradicted[2 + i] = contradicted[1 + i];
          }
        }
        if (const auto declared = field_group(line, "sampled_declared_one_way"); !declared.empty()) {
          const auto strong = field_quad(field_text(declared, "strong")), contradicted = field_quad(field_text(declared, "contradicted"));
          for (std::size_t i = 0; i != 2; ++i) {
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
        // S2b: texel 10 (the opaque Backbuffer and current pixels, the
        // informative claims and the h1 word) and the pre-UI scene image's
        // evidence; before S2b, the HUD-less image's (sampled_hudless_scene).
        if (const auto inferred = field_text(line, "sampled_inferred_opaque"); !inferred.empty()) {
          const auto opaque_counts = field_quad(inferred);
          evidence.inferred_opaque = {opaque_counts[0], opaque_counts[1]};
        }
        evidence.claims = field_number(field_text(line, "sampled_claims"), 16);
        if (const auto h1_word = field_group(line, "sampled_h1"); !h1_word.empty()) {
          evidence.s1_source = field_number(field_text(h1_word, "winner"));
          evidence.h1_applied = field_text(h1_word, "applied") == "1";
        }
        const auto pre_ui = field_group(line, "sampled_pre_ui_scene"), hudless_scene = field_group(line, "sampled_hudless_scene");
        if (const auto &group = pre_ui.empty() ? hudless_scene : pre_ui; !group.empty()) {
          evidence.pre_ui_scene.n = field_number(field_text(group, "n"));
          evidence.pre_ui_scene.d = std::strtof(field_text(group, "d").c_str(), nullptr);
          evidence.pre_ui_scene.valid = field_text(group, "valid") == "1";
          const auto image = field_text(group, "image");
          evidence.pre_ui_image = ui_detection::pre_ui_image::none;
          if (pre_ui.empty() || image == "hudless") {
            evidence.pre_ui_image = ui_detection::pre_ui_image::hudless;
          } else if (image == "layer") {
            evidence.pre_ui_image = ui_detection::pre_ui_image::layer;
          }
        }
        // Fix 1: the presented frame's verdict (the pre-UI proof's reconfirm
        // clock) and texel 11 (the proof's pixel counts).
        if (const auto scene = field_group(line, "sampled_scene"); !scene.empty()) {
          evidence.scene.valid = field_text(scene, "valid") == "1";
          evidence.scene.ran = field_text(scene, "ran") == "1";
          for (const auto verdict : {scene_verdict::hidden, scene_verdict::ambiguous, scene_verdict::visible}) {
            if (field_text(scene, "verdict") == ui_detection::name(verdict)) {
              evidence.scene.verdict = verdict;
            }
          }
        }
        if (const auto pre_ui_pixels = field_group(line, "sampled_pre_ui_pixels"); !pre_ui_pixels.empty()) {
          evidence.pre_ui_match = field_number(field_text(pre_ui_pixels, "match"));
          evidence.pre_ui_image_lit = field_number(field_text(pre_ui_pixels, "image_lit"));
        }
        // This render's guard state, not the sample's: scene_guard since S2b,
        // scene_hold (the held routes) before.
        if (const auto guard = field_group(line, "scene_guard"); !guard.empty()) {
          held_hidden += field_text(guard, "hidden") == "1" ? 1 : 0;
        } else {
          held_hidden += field_number(field_text(line, "scene_hold")) ? 1 : 0;
        }
        h1 += evidence.h1_applied ? 1 : 0;
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
      std::printf("%u fed samples logged under a held hidden verdict, %u decided H1 (source 8)\n", held_hidden, h1);
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
    {"T1 no bound and 6x", generated_presents_hold_without_a_bound},
    {"T1 grace of real frames without their own decision", real_frames_without_their_own_decision},
    {"T1 no hold across a scope change", holds_never_cross_a_scope},
    {"T1 holds survive key churn", holds_survive_key_churn},
    {"A1/S1/T1/P1 E33 title and Load Game", e33_title_and_load_game},
    {"E2/V2/A1/H1 Hogwarts HUD-less pairs exact only by batch", hudless_pairs_are_exact_only_by_batch},
    {"T1 HUD-less re-offers keep the held decision", hudless_reoffers_keep_the_held_decision},
    {"F1 sample staleness and status key", samples_go_stale_and_are_discarded},
    {"F1 winner keys samples", accepted_alpha_keys_samples_alone},
    {"H1 W3 layer route", hidden_scene_layer_route},
    {"H1 claim gate and scope clears", hidden_scene_gate_and_clears},
    {"P1/A1/S1 recorded menus", recorded_menus_after_acceptance},
    {"A1/S1 adversaries", adversaries},
    {"H1 Stellar Blade SDR menu visits", h1_stellar_blade_sdr_menu_visits},
    {"H1 Stellar Blade SDR with FG suspended", h1_stellar_blade_sdr_fg_suspended},
    {"H1 Stellar Blade SDR FG session (fix 1)", h1_stellar_blade_sdr_fg_session},
    {"H1/A3 pre-UI proof across sessions", h1_pre_ui_proof_across_sessions},
    {"H1 effects target never proven", h1_effects_target_is_never_proven},
    {"H1/A1 pre-UI proof keys", h1_pre_ui_proof_keys},
    {"H1 dark gameplay never flat", h1_dark_gameplay_never_flat},
    {"H1 overrides a partial winner", h1_overrides_a_partial_winner},
    {"S1/T1 invariants", s1_invariants},
  };
  try {
    for (const auto &[name, group] : groups) {
      group();
      std::printf("PASS %s\n", name);
    }
    std::printf("PASS UI sequence replay: %zu groups, %u KNOWN_TODAY, %u KNOWN_LIMIT\n", std::size(groups), known_today_count, known_limit_count);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL UI sequence replay: %s\n", error.what());
    return 1;
  }
}

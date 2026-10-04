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
// pairing and counting under frame generation (ui_mask::pair_hudless_present,
// generated_without_input) and the exact counters (ui_counters), and rule
// H2's run of still screens without a UI source (still_screen::run, fix 2)
// with the renderer's scope and switch, and fix 3's pre-UI change set: the
// layer copy's pairing with a retained Present (change_set::pair_layer), its
// offer as candidate::pre_ui under UIPinOnlyUI (change_set::offered), b2
// word 5's rule bits (change_set::rule_bits, the refine rule included) and
// the change-set shadow of every sample that measured the layer
// (change_set::measure_shadow, its log throttle and counters), and fix 4's
// rule P2 (pin only UI, game3d_ui_darkening.h): when the darkening passes
// run (rules::darkening_measured), whether the mask pass unpins an eligible
// source's darkening (ui_detection::darkening_applied, the mask a T1 reuse
// keeps included), and every sample's darkening counters and its throttled
// line from its decision words dk_unpinned and dk_kept. A stream's
// GPU input is what one detection counts: decision texels 0-11 that
// ui_detection_replay --verbose recorded on labelled dumps with the fix-1
// shader, or synthetic counts; texel 12, the compare pass's still and
// compared cells, is modelled from each Present's still share (a
// recording's texel 12 is zero); texels 13-15 and the change-set slot of the
// layer pair are synthetic (a recording's are zero), and so are words 62-63
// (fix 4's darkening counts, which test_game3d_ui_selection_contract and
// ui_detection_replay prove per pixel against the CPU reference). Every
// decision is ui_selection::decide, the C++ mirror of
// SunshineUIDetectionReduceCS (the T1 hold store, the refine rule and the H1
// and H2 overrides included) that test_game3d_ui_selection_contract proves
// equal to the shader's reduce; every recorded decision must equal it.
//
// It asserts the rules through stage S2b, fix 1 (the pre-UI proof by
// pixels), fix 2 (H2), fix 3 (pre-UI change sets and the refine rule,
// thirteen groups) and fix 4 (rule P2, pin only UI, six groups) strictly. An outcome that a later
// stage of the UI decision framework (docs/reshade-sbs.md, UI decision
// framework: stages S0-S6; rules E1, E2, V1, V2, A1-A3, S1, S2, H1, P1, T1,
// F1) changes prints "KNOWN_TODAY <stage> <rule>: <text>" and does not fail;
// that stage turns it into a strict assertion. Two remain after S2b, both S3
// T1/E2: S3 ships in shadow, and its group (S3 identity shadow) prints
// strict "SHADOW S3" lines for what the snapshot ticket would decide and
// asserts both cases strictly under the test-only identity override
// (ui_temporal::detection_state::identity_override); its GPU-verdict group
// (S3 GPU identity verdicts) feeds stamps and proposals (b2 words 6-9,
// ui_selection::identity_input) into every detection, proves that by
// default no verdict changes a decision while the GPU identity words are
// counted, and under the override asserts the gates: a late, unstamped or
// mismatched pair is refused (T1 reuses once), a token-space pre-UI pair
// decides with frame generation suspended, and an inexact HUD-less pair is
// absent. Outcomes the rules already call correct, such as an accepted source
// pinning a whole-frame alpha flat over a visible scene (P1, the opacity
// ruling), are asserted strictly, and so is the risk the approved pre-UI
// proof accepts (h1_dark_gameplay_never_flat, a proven pre-fog buffer).
//
// Informational, not in ctest: --log <ReShade.log>... replays the logged
// "Sunshine UI protection" samples through alpha_auto_policy and prints the
// predicted acceptance transitions beside the logged ones.
#include "game3d_alpha_auto.h"
#include "game3d_scene_guard.h"
#include "game3d_ui_change_set.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_darkening.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_mask.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_temporal.h"
#include "game3d_ui_ticket.h"

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
#include <optional>
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

  // Decision texels 0-15 as the renderer reads them back (selection revision 6).
  constexpr std::size_t texel_words = 4 * ui_detection::change_set_decision_texels;
  using texels = std::array<std::uint32_t, texel_words>;
  // Texels 5 and 6: the hidden-scene evidence of the presented frame and of
  // the pre-UI scene image, and texel 12, H2's still and compared cells,
  // written only by the evidence passes. Texels 13-15, the change-set shadow
  // (fix 3), are the tiles pass's on every sample frame with an offered layer.
  constexpr std::size_t scene_texels_begin = 4 * 5, scene_texels_end = 4 * ui_detection::scene_decision_texels;
  constexpr std::size_t still_texel_begin = 4 * ui_detection::pre_ui_decision_texels, still_texel_end = 4 * ui_detection::still_decision_texels;
  constexpr std::size_t change_set_texel_begin = still_texel_end;
  // The cells of the D grid the compare pass compares (H2).
  constexpr std::uint32_t still_cells = ui_detection::scene::cells_x * ui_detection::scene::cells_y;

  // A recording of 48 words, or of 44 (texels 0-10, recorded before
  // selection revision 4: texel 11 reads zero); texel 12 (selection revision
  // 5) reads zero.
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
  // at u5, as the previous detection wrote it; b2 word 5's rule bits: H2's
  // still::flatten and fix 3's refine, layer pair offset and bound retained
  // Presents, change_set::rule_bits).
  struct gpu_inputs {
    std::uint32_t bits {}, flags {}, accepted {}, per_frame {};
    std::uint64_t now_ms {};
    ui_selection::hold_state hold {};
    std::uint32_t rules {};
    // S3: the stamps at t9 and b2 words 6-9 (the renderer's detect_ui).
    ui_selection::identity_input identity {};
  };

  using gpu_model = std::function<texels(const gpu_inputs &)>;

  // The reduce's decision of these counts and inputs (ui_selection::decide).
  ui_selection::decision decide(const texels &t, const gpu_inputs &in) {
    return ui_selection::decide(ui_selection::counts_from_words(t.data(), t.size()), in.bits, in.accepted, in.flags | in.per_frame, in.hold, in.rules, in.identity);
  }

  // The counts of `t` with the words the reduce writes for these inputs:
  // texel 0 the applied decision, the refused candidate and the frame reason
  // (F1, with the T1 reused bit), and texel 10's informative claims and h1
  // word (the S1 winner, and whether H1 overrode it).
  texels decision_words(texels t, const gpu_inputs &in) {
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
    // pixels, lit layer pixels, lit presented pixels, lit presented pixels
    // that differ), written when a layer is offered (the reduce writes zeros
    // otherwise).
    std::array<std::uint32_t, 4> pre_ui_counts {};
    // Fix 4: the decided source's darkening pixels unpinned (or that would
    // be) and kept, decision words dk_unpinned and dk_kept.
    std::uint32_t dark_unpinned {}, dark_kept {};

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
    // invalid in between (the middle band). Its HUD-less image is lit
    // everywhere unless lit() says otherwise (fix 3: a partial set needs it
    // lit on 1% of pixels).
    synthetic &change_set(std::uint32_t changed) {
      c.changed = changed;
      c.unchanged = c.pixels - changed;
      c.matching_tiles = 200;
      c.lit = c.pixels;
      return *this;
    }

    // Fix 3: the offscreen layer against its pair's Present at
    // change_set::inferred_scale times the layer's pair threshold (texels
    // 13-15, the tiles pass's shadow on sample frames with an offered layer;
    // and the change-set slot when candidate::pre_ui is offered without a
    // HUD-less image): changed and unchanged pixels, matching tiles, the
    // changed pixels the 3x3 rule keeps, and the changed pixels against the
    // Presents one and two back (measured when bound). The pair's lit pixels
    // are the layer's (texel 11, pre_ui_pixels).
    synthetic &layer_pair(std::uint32_t changed, std::uint32_t filtered, std::uint32_t tiles, std::uint32_t changed_1, std::uint32_t changed_2) {
      c.shadow.changed = changed;
      c.shadow.unchanged = c.pixels - changed;
      c.shadow.nonfinite = 0;
      c.shadow.matching_tiles = tiles;
      c.shadow.filtered = filtered;
      c.shadow.changed_1 = changed_1;
      c.shadow.changed_2 = changed_2;
      return *this;
    }

    // The shadow's judge: the first offered and accepted declared alpha's
    // pixels (ui_detection::change_set::judge kind) and those of them the
    // filtered set holds.
    synthetic &judge(std::uint32_t judge_kind, std::uint32_t pixels, std::uint32_t tp) {
      c.shadow.judge_kind = judge_kind;
      c.shadow.judge_pixels = pixels;
      c.shadow.judge_tp = tp;
      return *this;
    }

    synthetic &lit(std::uint32_t pixels) {
      c.lit = pixels;
      return *this;
    }

    // Fix 4: what the darkening count and finish passes sum for the decided
    // source (rule P2).
    synthetic &darkening(std::uint32_t unpinned, std::uint32_t kept) {
      dark_unpinned = unpinned;
      dark_kept = kept;
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
    synthetic &pre_ui_pixels(std::uint32_t matched, std::uint32_t image_lit, std::uint32_t presented_lit, std::uint32_t differs) {
      pre_ui_counts = {matched, image_lit, presented_lit, differs};
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
      // The change-set slot: the layer pair when the pre-UI change set holds
      // it (fix 3), else the HUD-less pair.
      if (ui_selection::layer_pair_slot(in.bits)) {
        t[word::matching_tiles] = c.shadow.matching_tiles;
        t[word::hudless_changed] = c.shadow.changed;
        t[word::hudless_unchanged] = c.shadow.unchanged;
        t[word::hudless_invalid] = c.shadow.nonfinite;
        t[word::hudless_lit] = pre_ui_counts[1];
      } else {
        t[word::matching_tiles] = c.matching_tiles;
        t[word::hudless_changed] = c.changed;
        t[word::hudless_unchanged] = c.unchanged;
        t[word::hudless_invalid] = c.nonfinite;
        t[word::hudless_lit] = c.lit;
      }
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
        t[word::presented_lit] = pre_ui_counts[2];
        t[word::presented_lit_differs] = pre_ui_counts[3];
        // Texels 13-15: the change-set shadow of a layer proven the pre-UI
        // scene image (b2 word 5's shadow bit; zero otherwise); the Presents
        // one and two back only when bound (its retained bits).
        if (in.rules & ui_detection::change_set::shadow) {
          const auto &s = c.shadow;
          const bool bound_1 = (in.rules & ui_detection::change_set::retained_1) != 0;
          const bool bound_2 = (in.rules & ui_detection::change_set::retained_2) != 0;
          t[word::cs_changed] = s.changed;
          t[word::cs_unchanged] = s.unchanged;
          t[word::cs_nonfinite] = s.nonfinite;
          t[word::cs_matching_tiles] = s.matching_tiles;
          t[word::cs_filtered] = s.filtered;
          t[word::cs_changed_1] = bound_1 ? s.changed_1 : 0u;
          t[word::cs_changed_2] = bound_2 ? s.changed_2 : 0u;
          t[word::cs_judge_pixels] = s.judge_pixels;
          t[word::cs_judge_tp] = s.judge_tp;
          t[word::cs_judge_kind] = s.judge_kind;
        }
      }
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
      t = decision_words(t, in);
      // Fix 4: words 62-63 hold the darkening only when its passes measured
      // an eligible decided source on a frame T1 did not reuse (the finish
      // pass writes zero otherwise; the stream reads them on sample frames).
      if (ui_detection::darkening_dispatched(in.rules, t[word::source], (t[word::frame_reason] & ui_detection::frame_reason_reused) != 0u)) {
        t[word::dk_unpinned] = dark_unpinned;
        t[word::dk_kept] = dark_kept;
      }
      return t;
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
    // S3 (game3d_ui_ticket.h, shadow): the newest token generation among the
    // Present's offered tag snapshots (ui_ticket::newest_token; 0 none).
    // ui_temporal::ticket_identity fills the identity's token fields from it;
    // T1 reads them only under the test-only identity_override.
    std::uint64_t token_label {};
    // The provider offered an Auto input (source_alpha_ui); without one it
    // offers no candidate.
    bool available = true;
    bool depth_current = true;
    std::uint64_t epoch = 1, revision = 1;
    std::uint32_t viewport = 1;
    // The swapchain's typed DXGI format; with signatures.color_space it says
    // whether the output is SDR (still_screen::sdr_output, H2's scope).
    std::uint32_t output_format = 24;
    // H2: the share of the D grid's cells whose presented-luma mean stays
    // within the tolerance of the previous measured sample's (texel 12).
    float still_share = 1.f;
    // Fix 3, the offscreen layer copy's pairing inputs
    // (change_set::pair_layer): the game's frame generation mode is known off
    // at this Present (the sequence counts the run of such real Presents,
    // change_set::next_fg_off_presents), the Presents observed since the
    // copy (ui_layer::live_capture::presents_since_copy: 1 when it was taken
    // at the first clear after the previous Present), and whether the
    // renderer's ring retained the Presents one and two back (the ring's
    // requesters and linger are the renderer's,
    // test_game3d_source_alpha_runtime's section 'pre-UI change set').
    bool fg_known_off = true;
    std::uint32_t layer_presents_ago = 1;
    bool retained_1 = true, retained_2 = true;
    // S3 (game3d_ui_ticket.h): the stamps the GPU reads and the labels the
    // provider proposes (ui_selection::identity_input, b2 words 6-8), and
    // in bits only identity::token_batch (the renderer's proof of a
    // same-token HUD-less pair); the renderer adds the gates while identity
    // is authoritative.
    ui_selection::identity_input identity {};
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
    // H2: this render's scope (SDR Auto) and the still word it pushed.
    bool still_scope {};
    std::uint32_t still_bits {};
    // Fix 3: the run of real Presents with frame generation known off, this
    // one included; the layer copy's pairing, whether the pre-UI change set
    // was offered (candidate::pre_ui), and b2 word 5's rule bits as pushed.
    std::uint32_t fg_off_presents {};
    change_set::layer_pairing pairing;
    bool pre_ui_offered {};
    std::uint32_t rules {};
    // Fix 4: whether this detection ran the darkening passes (b2 word 5 as
    // pushed is rules | rules::darkening_measured: every Auto frame that
    // offers candidates with UIPinOnlyUI=1, Auto sample frames in its shadow),
    // and whether the mask this frame applies has an eligible source's
    // darkening unpinned (darkening_applied; a frame T1 reused, or a held
    // Present, keeps the mask of the frame it shows).
    bool darkening {}, darkened {};
    // S3: the identity this detection pushed (stamps, proposals and bits).
    ui_selection::identity_input identity {};
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
  //                       t.adopt; H2's still_scope (Auto and SDR output)
  //                       and switch, a render outside scope ending its run
  //                       (still.leave(scope)); then held() (t.hold,
  //                       held.generated), unavailable() (held.none, and
  //                       in scope still.leave(unmeasured)),
  //                       enter_scope and the scene guard's
  //                       enter_scope(epoch, viewport) (what it ends of H2's
  //                       run counted as the renderer's own) +
  //                       poll_detection + the session's pre-UI proof of
  //                       the offered layer's signature (layer_proven) + the
  //                       guard's per_frame(..., layer_proven) + on a
  //                       zero-offer frame in scope still.leave(unmeasured)
  //                       + the still word (the switch and
  //                       still_flatten()) + detect_ui + detected()
  //                       (t.detect with bits, or a zero-offer real frame
  //                       flagged accepted_missing), or inactive() (the
  //                       guard keeps its state; in scope
  //                       still.leave(unmeasured));
  //   end of render:      the detection fence is signaled (detection_awaiting_signal);
  //   update_alpha_auto:  the latest sample only while status_fresh;
  //   poll_detection:     sample_discarded (scope, stale on arrival),
  //                       decode_detection_sample, latest_key, the guard's
  //                       observe(sample_of(words), the pending actionable,
  //                       the submitted signatures by kind, the pending
  //                       still_scope while this render is in scope) and
  //                       the shadow and H2 runs, session.observe of
  //                       ui_temporal::ledger_evidence (no scene evidence
  //                       measured for H2 alone) with the signatures the
  //                       sample was submitted with, commit_counters
  //                       (sample_counters with the guard's observation);
  //   detect_ui:          the reduce with the hold store (u5) and, except on
  //                       a zero-offer frame, the tiles pass and the 100 ms
  //                       sample cadence, scene evidence by the guard's
  //                       measure(now, shadow, whole_frame(latest), a
  //                       proven layer that is the offer's pre-UI image,
  //                       still_scope and still.wants_measure()) with the
  //                       compare pass's previous cell means (u5, texel
  //                       12), the counter and CPU snapshot and the status
  //                       key.
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
    // Fix 3: the run of real Presents with frame generation known off, the
    // last step's included. A stream starts within such a run (two Presents
    // before its first: the Present one and two back were known off), as a
    // menu entered with FG off; a stream that models the run's start sets it
    // to zero.
    std::uint32_t fg_off_presents = 2;
    // The hidden-scene guard (M5), owned by the depth path.
    scene_guard::state guard;
    std::vector<frame_result> frames;
    std::vector<alpha_auto_decision> samples;  // Every sample read, after the guard observed it.
    std::vector<scene_guard::observation> observed;  // The guard's observation of each sample read.
    // Whether each sample read ran the evidence passes for rule H2 alone (in
    // SDR Auto, where it measures without acting; fix 2).
    std::vector<bool> h2_only;
    // The guard's observations of committed samples (the scene counters).
    std::uint64_t scene_entered {}, scene_released {}, scene_refuted {};
    // H2: every observation of the run (a sample's, or an end by a scope or
    // identity change, counted by the renderer itself), as the renderer logs
    // them; and the still counters the committed samples carry.
    struct still_event {
      std::size_t frame;
      std::uint64_t now_ms;
      still_screen::observation what;
      // The run as it entered (entered only).
      still_screen::episode run;
      bool counted;
    };
    std::vector<still_event> still_events;
    std::uint64_t still_committed_entered {}, still_committed_released {}, still_committed_short {};

    struct transition {
      std::uint64_t tick;
      std::string accepted;
    };

    std::vector<transition> trust;  // stored() after a sample changed it.
    // Fix 3: each sample's change-set shadow (when it measured the layer),
    // the shadow lines the renderer would log, and the change_set counters
    // the committed samples carry. Fix 4: the darkening lines the renderer
    // would log (one per sample that measured an eligible source) and the
    // darkening counters the committed samples carry.
    std::vector<std::optional<change_set::shadow_sample>> shadows;
    std::vector<std::string> shadow_lines;
    ui_counters shadow_committed;
    std::vector<std::string> darkening_lines;
    ui_darkening::log_state darkening_log;
    std::size_t darkening_samples {};
    ui_counters darkening_committed;
    // What render() saves with a sample for its change-set shadow.
    struct change_state {
      change_set::layer_pairing pairing;
      bool layer_proven {}, hudless_offered {}, auto_mode {}, enabled {};
    };
    std::size_t discards {}, commits {}, committed_frames {};
    // S3: the GPU identity verdict totals the committed samples carry
    // (renderer::identity_counts).
    ui_ticket::identity_counters identity_gpu;

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
      current_frame = index;
      current_now = p.now_ms;
      std::uint32_t bits = p.available ? p.offered : 0u;
      const auto state = session.decision().state;
      const bool auto_mode = state != alpha_auto_state::manual_on && state != alpha_auto_state::manual_off;
      // Fix 3 (game3d_ui_change_set.h): the layer copy pairs with the
      // retained Present it shows; in Auto with UIPinOnlyUI=1 a layer
      // whose signature the ledger proved the pre-UI scene image is offered
      // as the pre-UI change set (signed with the layer's format, so its
      // acceptance is that proof) unless a HUD-less image holds the slot.
      const bool layer_offered = (bits & candidate::layer) != 0;
      auto signatures = p.signatures;
      fg_off_presents = change_set::next_fg_off_presents(fg_off_presents, p.fg_known_off, p.hold_previous);
      r.fg_off_presents = fg_off_presents;
      r.pairing = change_set::pair_layer(layer_offered, fg_off_presents, p.layer_presents_ago, p.retained_1, p.retained_2);
      const bool pin_only_ui = auto_mode && session.pin_only_ui();
      const bool pre_ui_proven = layer_offered && session.pre_ui_proven(signatures.of(kind::ui_layer));
      const bool hudless_offered = (bits & candidate::hudless) != 0;
      // S3 (game3d_renderer.cpp): only while identity is authoritative does a
      // layer without a Present-counted pairing pair in token space (frame
      // generation not known off, a Backbuffer tag offered, a token
      // proposed), and only then are the gates pushed.
      const bool authoritative = ui_ticket::authoritative(temporal.identity_override);
      if (authoritative && layer_offered && r.pairing.kind != change_set::pair_class::retained && (bits & candidate::backbuffer) && p.identity.expected_layer_token && !p.fg_known_off) {
        r.pairing = change_set::token_pairing();
      }
      r.identity = p.identity;
      r.identity.bits = (p.identity.bits & ui_detection::identity::token_batch) | (authoritative ? ui_detection::identity::gate_hudless : 0u) | (authoritative && pin_only_ui ? ui_detection::identity::gate_layer : 0u);
      if (change_set::offered(auto_mode, pin_only_ui, pre_ui_proven, hudless_offered, r.pairing)) {
        bits |= candidate::pre_ui;
        signatures.set(kind::pre_ui, signatures.format[std::size_t(kind::ui_layer)]);
        r.pre_ui_offered = true;
      }
      const std::uint32_t accepted = session.accepted(bits, signatures);
      alpha_auto_source observation;
      observation.now_ms = observation.tick_ms = p.now_ms;
      observation.epoch = p.epoch;
      observation.revision = p.revision;
      observation.viewport = p.viewport;
      observation.session = &session;
      const auto identity = ui_temporal::ticket_identity({p.hold_previous, p.real_frame}, p.token_label, temporal.decision_token, bits != 0);
      r.hold = temporal.arbitrate(identity, observation, bits);
      const std::uint32_t flags = (bits & candidate::layer) ? p.layer_flags : 0u;
      if (r.hold.adopt) {
        temporal.adopt(bits, accepted);
      }
      // H2 (game3d_still_screen.h): Auto on SDR output only; a render outside
      // that scope ends the run at once.
      still_scope = auto_mode && still_screen::sdr_output(p.signatures.color_space, p.output_format);
      const bool still_enabled = still_scope && session.still_flatten();
      r.still_scope = still_scope;
      if (!still_scope) {
        still_end(guard.still.leave(still_screen::end_reason::scope), false);
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
        still_unmeasured();
      } else if (r.hold.detect && (bits || missing)) {
        r.active = r.detected = true;
        r.grace = !bits;
        temporal.enter_scope(observation);
        still_end(guard.enter_scope(observation.epoch, observation.viewport), false);
        const bool shadow = session.first_run();
        poll(observation, r);
        // H1 (d): the ledger's pre-UI proof of the offered layer's signature,
        // read after the poll, so a sample that just earned it counts at once.
        r.layer_proven = (bits & candidate::layer) && session.pre_ui_proven(p.signatures.of(kind::ui_layer));
        r.scene_bits = guard.per_frame(p.now_ms, bits, p.signatures.by_kind(), r.layer_proven);
        const std::uint32_t per_frame = r.scene_bits | r.hold.per_frame | (!p.depth_current ? ui_detection::per_frame_depth_not_current : 0u);
        // H2: flat only while its run is active and the session enables it;
        // a zero-offer frame is never sampled and ends the run instead.
        if (r.grace) {
          still_unmeasured();
        }
        r.still_bits = still_enabled && guard.still_flatten() ? ui_detection::still::flatten : 0u;
        // b2 word 5: H2's flatten, refine (Auto and the switch), the layer
        // pair's offset and the retained Presents bound at t2 and t3.
        r.rules = change_set::rule_bits(r.still_bits != 0, pin_only_ui, pre_ui_proven, r.hold.change_set_gap, r.pairing, layer_offered && p.retained_1, layer_offered && p.retained_2);
        still_share = p.still_share;
        pending_change = {r.pairing, pre_ui_proven, hudless_offered, auto_mode, pin_only_ui};
        detect(observation, bits, flags, accepted, signatures, per_frame, shadow, index, r);
        // T1's invariant on the identity T1 applied (today's, or under the
        // test-only override the ticket's).
        invariants.generated_detections += temporal.applied(identity).generated ? 1 : 0;
        temporal.detected(observation, identity, bits);
      } else {
        ++cpu[ui_counter::inactive_no_candidates];
        temporal.inactive();
        still_unmeasured();
      }
      if (r.active) {
        r.source = mask_source;
        r.covered = mask_covered;
        r.pixels = mask_pixels;
        r.darkened = mask_darkened;
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
      r.consumed.still = {still_scope, still_enabled, guard.still.state(), guard.still.run_ms()};
      // update_alpha_auto: the longest run that ended before it entered since
      // the previous status, whatever ended it, published once.
      r.consumed.evidence.still_short_ms = std::max(r.consumed.evidence.still_short_ms, std::exchange(still_short_unpublished_ms, std::uint64_t {}));
      frames.push_back(r);
      return frames.back();
    }

    // The still counters the session's committed totals must carry: the
    // guard's observations of committed samples and the renderer's own ends
    // of a run counted before the last committed submission.
    std::array<std::uint64_t, 3> still_committed() const {
      return {still_committed_entered + committed_cpu[ui_counter::still_entered],
        still_committed_released + committed_cpu[ui_counter::still_released],
        still_committed_short + committed_cpu[ui_counter::still_short]};
    }

  private:
    // H2: the renderer's still_log. It records what a run's observation or
    // end did and counts the ends the guard's sample observations do not (a
    // scope or identity change): counted says whether a committed sample
    // counts this one.
    void still_end(const still_screen::observation &o, bool counted) {
      if (o.entered || o.ended || o.short_run) {
        still_events.push_back({current_frame, current_now, o, o.entered ? guard.still.current() : still_screen::episode {}, counted});
      }
      if (o.ended && !counted) {
        ++cpu[ui_counter::still_released];
      }
      if (o.short_run) {
        still_short_unpublished_ms = std::max(still_short_unpublished_ms, o.short_ms);
        if (!counted) {
          ++cpu[ui_counter::still_short];
        }
      }
    }

    // The renderer's still_unmeasured: a render in scope that cannot be
    // sampled ends H2's run.
    void still_unmeasured() {
      if (still_scope) {
        still_end(guard.still.leave(still_screen::end_reason::unmeasured), false);
      }
    }

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
      // H2 reads it as in scope only while this render still is.
      const auto observation = guard.observe(scene_guard::sample_of(pending_texels.data(), pending_texels.size(), pending_source.now_ms, true), pending_actionable, pending_signatures.by_kind(), pending_still_scope && still_scope);
      latest.evidence.shadow_hidden_ms = observation.shadow_hidden_ms;
      latest.evidence.still_short_ms = observation.still.short_ms;
      // H2: the episode log; the committed sample counts its observation.
      still_end(observation.still, true);
      session.observe(ui_temporal::ledger_evidence(latest.evidence, pending_h2_only), latest.pixels, pending_source.now_ms, pending_signatures);
      // Fix 3, the change-set shadow (game3d_renderer.cpp, change_set_shadow):
      // measure_shadow of the sample's counts with the state it was submitted
      // with, logged when due; its counters join this commit. It reaches no
      // ledger and changes no decision.
      std::optional<change_set::shadow_sample> shadow;
      if (pending_layer_measured) {
        const auto c = ui_selection::counts_from_words(pending_texels.data(), pending_texels.size());
        const auto &change = submitted_change;
        shadow = change_set::measure_shadow(c, latest.evidence.candidates, latest.evidence.accepted, pending_flags, pending_rules, change.pairing, change.layer_proven, change.hudless_offered, change.auto_mode, change.enabled);
        latest.change_set = {true, shadow->pairing.kind, shadow->pairing.offset, shadow->valid, shadow->would_refine, change.enabled};
        if (change_set::shadow_log_due(shadow_log, *shadow, pending_source.now_ms)) {
          shadow_lines.push_back(change_set::shadow_log_text(*shadow));
        }
      }
      // Fix 4, rule P2 (game3d_renderer.cpp, darkening_sample): a sample whose
      // darkening passes measured its eligible decided source on a frame T1
      // did not reuse reads its words dk_unpinned and dk_kept, its commit
      // counts them, and its line is logged when due (on a change, else once
      // a second).
      if (pending_darkening && ui_detection::darkening_dispatched(pending_rules | ui_detection::rules::darkening_measured, latest.source_kind, latest.evidence.reused)) {
        const bool applied = (pending_rules & ui_detection::rules::pin_only_ui) != 0u;
        const auto measured = ui_darkening::sample_of(latest.source_kind, applied, latest.covered, pending_texels[word::dk_unpinned], pending_texels[word::dk_kept]);
        latest.darkening = {true, applied, measured.unpinned, measured.kept, measured.colourless};
        ++darkening_samples;
        if (ui_darkening::log_due(darkening_log, measured, pending_source.now_ms)) darkening_lines.push_back(ui_darkening::log_text(measured));
      }
      commit(latest, observation, shadow ? &*shadow : nullptr);
      counters_pending = false;
      r.polled = true;
      samples.push_back(latest);
      observed.push_back(observation);
      h2_only.push_back(pending_h2_only);
      shadows.push_back(shadow);
      auto now_accepted = session.stored();
      if (now_accepted != last_accepted) {
        trust.push_back({latest.sample_tick_ms, now_accepted});
      }
      last_accepted = std::move(now_accepted);
    }

    void detect(const alpha_auto_source &observation, std::uint32_t bits, std::uint32_t flags, std::uint32_t accepted, const candidate_signatures &signatures, std::uint32_t per_frame, bool shadow, std::size_t index, frame_result &r) {
      const auto now = observation.now_ms;
      // A sample frame: the 100 ms cadence, one sample pending at a time,
      // never a zero-offer frame.
      const bool sample = !r.grace && !pending && !(last_submit && now >= last_submit && now - last_submit < sample_interval_ms);
      // Fix 4 (game3d_renderer.cpp, detect_ui): the darkening passes run on
      // every Auto frame that offers candidates with the switch on and on
      // Auto sample frames in its shadow; the pushed b2 word 5 says so.
      r.darkening = !r.grace && pending_change.auto_mode && ((r.rules & ui_detection::rules::pin_only_ui) || sample);
      r.gpu = {bits, flags, accepted, per_frame, now, gpu_hold, r.rules | (r.darkening ? ui_detection::rules::darkening_measured : 0u), r.identity};
      // A zero-offer frame runs no tiles pass: the reduce reads the
      // statistics rows the previous tiles pass left (with nothing offered,
      // they decide nothing of their own).
      texels t = r.grace ? decision_words(statistics, r.gpu) : gpu(r.gpu);
      // The reduce decides, writes the hold store and counts
      // (SunshineUIDetectionReduceCS, mirrored by ui_selection::decide and
      // counter_adds); the mask pass writes no mask when it reused one.
      r.decision = decide(t, r.gpu);
      require(r.decision.source == t[word::source] && r.decision.covered == t[word::covered], "A stream's GPU model did not decide as ui_selection::decide");
      // Fix 4: the darkening passes' bit never changes a decision (the reduce
      // ignores it; only the passes after it and the mask pass read it).
      if (r.darkening) {
        auto plain = r.gpu;
        plain.rules &= ~ui_detection::rules::darkening_measured;
        const auto d = decide(t, plain);
        require(d.source == r.decision.source && d.covered == r.decision.covered && d.reused == r.decision.reused && d.refined == r.decision.refined && d.none_reason == r.decision.none_reason && d.next.source == r.decision.next.source, "The darkening passes' bit changed a decision");
      }
      gpu_hold = r.decision.next;
      if (!r.grace) {
        statistics = t;
      }
      const auto adds = ui_selection::counter_adds(r.decision, r.gpu.flags | per_frame);
      for (std::size_t i = 0; i != adds.size(); ++i) {
        gpu_words[i] += adds[i];
      }
      // S3: the identity counter words after count.
      const auto identity_adds = ui_selection::identity_counter_adds(r.decision.identity);
      for (std::size_t i = 0; i != identity_adds.size(); ++i) {
        identity_words[i] += identity_adds[i];
      }
      ++invariants.detections;
      invariants.untrusted_inferred += r.decision.untrusted_inferred ? 1 : 0;
      invariants.presented_over_dedicated += r.decision.presented_over_dedicated ? 1 : 0;
      mask_source = r.decision.source;
      mask_covered = r.decision.covered;
      mask_pixels = t[word::pixels];
      // The mask pass returns early on a frame T1 reused, so its mask keeps
      // the previous real frame's darkening.
      if (!r.decision.reused) {
        mask_darkened = ui_detection::darkening_applied(r.gpu.rules, r.decision.source);
      }
      if (!sample) {
        return;
      }
      const bool proven_image = r.layer_proven && ui_selection::pre_ui_image_of(bits) == ui_detection::pre_ui_image::layer;
      const auto measure = guard.measure(now, shadow, ui_temporal::whole_frame(temporal.latest), proven_image, still_scope && guard.still.wants_measure());
      pending_h2_only = measure.run && !guard.measure(now, shadow, ui_temporal::whole_frame(temporal.latest), proven_image).run;
      pending_actionable = false;
      const bool evidence = measure.run;
      if (evidence) {
        pending_actionable = measure.actionable;
        // The compare pass compares every cell with the mean the previous
        // compare pass stored (none before the first), then stores its own.
        const std::uint32_t compared = previous_luma ? still_cells : 0u;
        t[word::still_cells] = std::uint32_t(std::lround(double(still_share) * compared));
        t[word::still_compared] = compared;
        previous_luma = true;
      }
      // The reduce zeroes texels 5, 6 and 12; only the evidence passes write
      // them.
      else {
        std::fill(t.begin() + scene_texels_begin, t.begin() + scene_texels_end, 0u);
        std::fill(t.begin() + still_texel_begin, t.begin() + still_texel_end, 0u);
      }
      if (sample_edit) {
        sample_edit(t, now, evidence);
      }
      pending_texels = t;
      pending_signatures = signatures;
      pending_flags = r.gpu.flags | per_frame;
      pending_still_scope = still_scope;
      // Fix 3: a sample with an offered layer proven the pre-UI scene image
      // measured its pair (b2 word 4 is pushed on sample frames, word 5
      // carries the shadow bit); the change-set state it was submitted with.
      pending_layer_measured = (bits & candidate::layer) && (r.rules & ui_detection::change_set::shadow);
      pending_rules = r.rules;
      pending_darkening = r.darkening;
      submitted_change = pending_change;
      counters_pending = true;
      pending_words = gpu_words;
      pending_identity_words = identity_words;
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
    void commit(const alpha_auto_decision &sample, const scene_guard::observation &observation, const change_set::shadow_sample *shadow) {
      if (!counters_pending) {
        return;
      }
      auto delta = ui_temporal::sample_counters(sample, pending_cpu, committed_cpu, pending_words, committed_words, pending_source.now_ms, observation);
      if (shadow) {
        change_set::add_shadow_counters(delta, *shadow);
        change_set::add_shadow_counters(shadow_committed, *shadow);
      }
      if (sample.darkening.measured) {
        delta.add_darkening(sample.darkening.unpinned, sample.darkening.kept);
        darkening_committed.add_darkening(sample.darkening.unpinned, sample.darkening.kept);
      }
      scene_entered += observation.entered ? 1 : 0;
      scene_released += observation.released ? 1 : 0;
      scene_refuted += observation.refuted;
      still_committed_entered += observation.still.entered ? 1 : 0;
      still_committed_released += observation.still.ended ? 1 : 0;
      still_committed_short += observation.still.short_run ? 1 : 0;
      committed_cpu = pending_cpu;
      committed_words = pending_words;
      // S3 (game3d_renderer.cpp, commit_counters): the identity words' change
      // joins the "Sunshine UI identity" gpu group, never ui_counters.
      identity_gpu += ui_temporal::identity_gpu_delta(pending_identity_words, committed_identity_words);
      committed_identity_words = pending_identity_words;
      committed_frames = pending_frame + 1;
      ++commits;
      session.add_counters(delta);
    }

    ui_counters cpu, pending_cpu, committed_cpu;
    std::array<std::uint32_t, ui_counter_word::count> gpu_words {}, pending_words {}, committed_words {};
    std::array<std::uint32_t, ui_counter_word::identity_words> identity_words {}, pending_identity_words {}, committed_identity_words {};
    bool pending {}, awaiting {}, counters_pending {};
    // Whether the pending sample's scene evidence is actionable (the guard's
    // measure when it was submitted).
    bool pending_actionable {};
    // H2: this render's scope, the pending sample's, the Present's still
    // share, and whether the compare pass stored previous cell means (u5).
    bool still_scope {}, pending_still_scope {}, previous_luma {}, pending_h2_only {};
    // The renderer's still_short_unpublished_ms.
    std::uint64_t still_short_unpublished_ms {};
    float still_share = 1.f;
    std::size_t current_frame {};
    std::uint64_t current_now {};
    std::size_t ready_frame {}, pending_frame {};
    std::uint64_t last_submit {}, submitted {};
    std::uint32_t pending_key {};
    alpha_auto_source pending_source;
    candidate_signatures pending_signatures;
    std::uint32_t pending_flags {};
    texels pending_texels {};
    // Fix 3: the change-set state of this render and of the pending sample,
    // and the shadow's log throttle.
    bool pending_layer_measured {}, pending_darkening {};
    std::uint32_t pending_rules {};
    change_state pending_change, submitted_change;
    change_set::shadow_log_state shadow_log;
    // The GPU's state across detections: the statistics rows of the last
    // tiles pass, the T1 hold store (u5) and the detected mask.
    texels statistics {};
    ui_selection::hold_state gpu_hold {};
    std::uint32_t mask_source {}, mask_covered {}, mask_pixels {};
    bool mask_darkened {};
    std::string last_accepted;
  };

  // Whether a sample's evidence passes ran for a rule other than H2, which
  // measures every SDR Auto sample frame while no source other than 11
  // decided and never acts on what it measures (fix 2).
  bool measured(const sequence &s, const alpha_auto_decision &sample) {
    const auto i = std::size_t(&sample - s.samples.data());
    require(i < s.samples.size(), "Not a sample of this stream");
    return sample.evidence.scene.ran && !s.h2_only[i];
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
        expected[ui_counter::refined] += f.decision.refined ? 1 : 0;
        expected[ui_counter::contradicted] += f.decision.contradicted ? 1 : 0;
        full_alpha += alpha_source(f.source) && f.pixels && std::uint64_t(f.covered) * 100 >= std::uint64_t(f.pixels) * 99 ? 1 : 0;
      }
    }
    require(totals[ui_counter::full_alpha] == full_alpha, what + ": full_alpha is " + std::to_string(totals[ui_counter::full_alpha]) + ", the stream says " + std::to_string(full_alpha));
    expected[ui_counter::samples] = s.commits;
    for (const auto index : {ui_counter::auto_frames, ui_counter::detection_frames, ui_counter::held_generated, ui_counter::held_none, ui_counter::reused, ui_counter::inactive_no_candidates, ui_counter::contradicted, ui_counter::samples, ui_counter::refined}) {
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
    require(!totals.decided(7) && !totals.decided(9), what + ": a retired source decided");
    // The still group counts the guard's observations of committed samples
    // and the renderer's own ends of a run (H2).
    const auto still = s.still_committed();
    require(totals[ui_counter::still_entered] == still[0] && totals[ui_counter::still_released] == still[1] && totals[ui_counter::still_short] == still[2], what + ": the still counters differ from the run's observations");
    // The change_set group counts the committed samples' shadows (fix 3).
    for (auto index = ui_counter::change_set_samples; index <= ui_counter::change_set_pair_contradicted; ++index) {
      require(totals[index] == s.shadow_committed[index], what + ": change_set counter " + std::to_string(index) + " differs from the committed shadows");
    }
    // The darkening group counts the committed samples' darkening (fix 4):
    // every measured sample, of which the throttled lines are a subset.
    for (auto index = ui_counter::darkening_samples; index <= ui_counter::darkening_kept_px; ++index) {
      require(totals[index] == s.darkening_committed[index], what + ": darkening counter " + std::to_string(index) + " differs from the committed samples");
    }
    require(totals[ui_counter::darkening_samples] == s.darkening_samples && s.darkening_lines.size() <= s.darkening_samples, what + ": the darkening samples counted differ from those measured, or more lines were logged than samples");
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
      // the accepted pair decides it flat (6), which H1 never overrides
      // although the guard holds the hidden verdict again.
      alpha_auto_policy session;
      const auto title = recorded_frames({recorded::hl_title, recorded::hl_title_held, recorded::hl_title_accepted});
      sequence s(session, phased({{12000, title}, {13000, recorded_frames({recorded::hl_hud, recorded::hl_hud_accepted})}, {UINT64_MAX, title}}));
      run(s, p, 10000, 14000);
      require(s.samples.size() > 3 && !measured(s, s.samples[0]) && s.samples[1].evidence.scene.ran && s.samples[2].sample_tick_ms - s.samples[1].sample_tick_ms <= ui_detection::scene::hold_ms, "The title's first samples did not open the gate and measure");
      const auto entry = poll_of(s, s.samples[2].sample_tick_ms);
      require(s.observed[2].entered && s.frames[entry].scene_bits == (ui_detection::per_frame_scene_hidden | ui_detection::per_frame_pre_ui_visible), "The title did not enter both holds at its second measured hidden sample");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        if (f.now_ms < 12000) {
          require(f.source != 6, "The unaccepted title pair decided 6");
          require(f.source == (i >= entry ? 8u : 0u) && f.decision.h1 == (i >= entry), "The title did not go flat (H1) exactly from the poll of its second measured hidden sample");
          require(i >= entry || (f.decision.none_reason == ui_no_mask::gate_no_hold && f.decision.refused == candidate::hudless), "The title before the hold did not name its acting claim");
        } else if (f.now_ms >= 13000) {
          require(f.flat() && f.source == 6 && !f.decision.h1, "The accepted pair did not decide the title flat");
        }
      }
      require(s.trust.size() == 1 && s.trust[0].tick == first_sample_from(s, 12000) && s.trust[0].accepted == hudless, "The gameplay HUD did not accept the pair by its first sample");
      const auto c = session.counters();
      require(c[ui_counter::scene_entered] == 2 && c[ui_counter::scene_released] == 1 && !c[ui_counter::scene_refuted] && c[ui_counter::full_d_hidden] > 0 && !c[ui_counter::full_d_visible], "The title's H1 hold was not counted entered twice and released once");
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

  // ---------------------------------------------------------------- S3 identity shadow

  // A frame-generation stream with S3 token labels (game3d_ui_ticket.h). The
  // game tags each real frame's HUD-less and Backbuffer images under one
  // token right after the previous real Present (the pacer's tag stands for
  // the token generation), and every Present until the next real one
  // re-offers that snapshot: under DLSS-G ordering a token's first Present is
  // a generated one. Today (authoritative false) the stream is fg_drift's:
  // the provider pairs the HUD-less image with the presented colour by
  // Present counting under the reported multiplier, inexact, and the
  // identity shadow compares that with the ticket per Present. Under the
  // test-only override T1 follows the tokens and the HUD-less image pairs
  // only with its same-token Backbuffer (exact), never with a presented
  // image.
  struct identity_stream {
    fg_tally tally;
    std::size_t tokens {}, detections {}, fresh_detections {}, holds {}, unavailable {}, inexact_detections {};
    ui_ticket::identity_counters counters;
  };

  identity_stream fg_identity(std::size_t count, const std::function<std::pair<std::uint32_t, std::uint32_t>(std::size_t)> &cadence, bool authoritative, const char *what) {
    alpha_auto_policy session;
    accept_hudless(session);
    sequence s(session, difference_frame);
    s.temporal.identity_override = authoritative;
    fg_pacer pacer(cadence(0).first);
    present real;
    real.offered = authoritative ? candidate::hudless | candidate::exact : candidate::current | candidate::hudless;
    identity_stream out;
    ui_ticket::identity_shadow shadow;
    std::uint64_t now = 10000, last_token = 0;
    for (std::size_t index = 0; index != count; ++index) {
      const auto [actual, reported] = cadence(index);
      const auto step = pacer.present(true, actual, reported);
      // Today's identity and offer by Present counting; under the override
      // every Present re-offers the token's exact pair.
      auto p = fg_present(step.kind, real, candidate::current, pacer.tag);
      if (authoritative) {
        p.offered = real.offered;
      }
      p.token_label = pacer.tag;
      p.now_ms = now;
      now += 8;
      const bool first = pacer.tag != last_token;
      last_token = pacer.tag;
      out.tokens += first ? 1 : 0;
      const auto &r = s.step(p);
      const bool paired = r.detected && (r.gpu.bits & candidate::hudless);
      ++out.tally.presents;
      if (step.real) {
        ++out.tally.real;
        out.tally.extra_holds += r.held ? 1 : 0;
        out.tally.paired_real += paired ? 1 : 0;
      } else if (r.held) {
        ++out.tally.held;
      } else {
        ++out.tally.lost;
        out.tally.misread += paired ? 1 : 0;
      }
      out.detections += r.detected ? 1 : 0;
      out.fresh_detections += r.detected && first ? 1 : 0;
      out.holds += r.held ? 1 : 0;
      out.unavailable += r.unavailable ? 1 : 0;
      out.inexact_detections += paired && !(r.gpu.bits & candidate::exact) ? 1 : 0;
      ui_ticket::today_view today;
      today.offered = true;  // The game tags a HUD-less image for every real frame.
      today.detects = paired;
      today.holds = r.held;
      today.exact = (p.offered & candidate::exact) != 0;
      today.real_frame = p.real_frame;
      // The ticket pairs the HUD-less image only with its same-token
      // Backbuffer, which only the override's stream offers.
      const auto view = shadow.step(today, pacer.tag, authoritative, authoritative, out.counters);
      require(view.fresh == first && view.hold_previous == !first && view.real_frame == pacer.tag, std::string("The ticket identity was not fresh once per token: ") + what);
    }
    shadow.end_scope(out.counters);
    check_counters(s, what);
    return out;
  }

  void s3_identity_shadow() {
    namespace idn = ui_ticket::identity_counter;
    using cadence_fn = std::function<std::pair<std::uint32_t, std::uint32_t>(std::size_t)>;
    struct drift {
      const char *what;
      std::size_t count;
      cadence_fn cadence;
      bool known_today;
    };

    const drift cases[] {
      {"4x matching cadence", 401, [](std::size_t) {
         return std::pair {3u, 3u};
       },
       false},
      {"4x reported as 2x", 401, [](std::size_t) {
         return std::pair {3u, 1u};
       },
       true},
      {"2x reported as 4x", 300, [](std::size_t index) {
         return std::pair {1u, index < 100 ? 1u : 3u};
       },
       false},
      {"2x to 4x in the middle of a real frame, reported late", 300, [](std::size_t index) {
         return std::pair {index < 100 ? 1u : 3u, index < 103 ? 1u : 3u};
       },
       true},
    };
    for (const auto &c : cases) {
      // Default: today's outcomes, identical to the multiplier drift group's
      // fg_drift, and the identity shadow of them.
      alpha_auto_policy session;
      accept_hudless(session);
      sequence reference(session, difference_frame);
      const auto expected = fg_drift(reference, c.count, 0, c.cadence);
      const auto today = fg_identity(c.count, c.cadence, false, c.what);
      require(today.tally.real == expected.real && today.tally.held == expected.held && today.tally.misread == expected.misread && today.tally.lost == expected.lost && today.tally.extra_holds == expected.extra_holds && today.tally.paired_real == expected.paired_real, std::string("The S3 shadow changed today's outcome: ") + c.what);
      const auto &k = today.counters;
      require(k[idn::frames_total] <= today.tokens && k[idn::frames_total] == k[idn::frames_once] + k[idn::frames_missed] + k[idn::frames_repeated], std::string("The identity shadow's frames do not add up: ") + c.what);
      // Every detection beyond a token's first is a misread generated Present.
      require(k[idn::frames_extra] == expected.misread && (!c.known_today || k[idn::frames_repeated]), std::string("The identity shadow's repeated detections are not today's misreads: ") + c.what);
      std::printf("SHADOW S3 T1/E2: %s: today detects %llu of %llu frames once, %llu repeatedly (%llu extra detections, %zu from an inexact presented pair), misses %llu; the ticket detects each of %zu tokens once and holds every Present that re-offers one\n", c.what, static_cast<unsigned long long>(k[idn::frames_once]), static_cast<unsigned long long>(k[idn::frames_total]), static_cast<unsigned long long>(k[idn::frames_repeated]), static_cast<unsigned long long>(k[idn::frames_extra]), today.inexact_detections, static_cast<unsigned long long>(k[idn::frames_missed]), today.tokens);
      // The test-only override (S3 enabled): T1 by token, the exact
      // same-token pair. Strict for every cadence, the two KNOWN_TODAY S3
      // T1/E2 cases included.
      const auto ticket = fg_identity(c.count, c.cadence, true, c.what);
      require(ticket.detections == ticket.tokens && ticket.fresh_detections == ticket.tokens, std::string("Under S3 a token did not detect exactly once, on its first Present: ") + c.what);
      require(ticket.holds + ticket.detections == ticket.tally.presents && !ticket.unavailable && !ticket.inexact_detections, std::string("Under S3 a Present re-offering a token did not hold its decision, or a pair was inexact: ") + c.what);
      require(!ticket.counters[idn::frames_repeated] && !ticket.counters[idn::frames_extra] && ticket.counters[idn::detect_agree] == ticket.tokens && !ticket.counters[idn::detect_today_only], std::string("Under S3 the shadow saw a repeated detection: ") + c.what);
    }

    // Labels (game3d_ui_ticket.h, game3d_ui_change_set.h): what the stamp
    // proves where Present counting cannot see.
    using ui_ticket::boundary;
    using ui_ticket::label_space;
    using ui_ticket::refusal;
    ui_ticket::identity_counters counters;
    const change_set::layer_pairing retained1 {change_set::pair_class::retained, 1u};
    const std::uint32_t present_label = 10;
    const auto proposed = ui_ticket::valid_label(label_space::present, change_set::proposed_layer_label(retained1, present_label));
    // A layer copy whose list executed as counted pairs; one that executed a
    // Present earlier than the count (or after the next Present) is a
    // mismatch: absent once identity decides, whatever the content verdict.
    require(ui_ticket::pair_exact(ui_ticket::present_label(boundary::before_clear, 9), proposed), "A copy executed as counted did not pair");
    for (const std::uint32_t read : {8u, 10u}) {
      const auto copy = ui_ticket::present_label(boundary::before_clear, read);
      require(!ui_ticket::pair_exact(copy, proposed) && ui_ticket::pair_refusal(copy, proposed) == refusal::mismatch, "A late-executed layer copy paired");
      ui_ticket::count_gpu_verdict(ui_ticket::gpu_verdict::mismatch, false, counters);
    }
    // A copy executed on another queue: its read races the clock.
    ui_ticket::ticket foreign;
    foreign.kind = ui_ticket::capture_kind::layer_copy;
    foreign.at = boundary::before_clear;
    foreign.present_queue = ui_ticket::queue_relation::foreign;
    foreign.present = ui_ticket::refuse(ui_ticket::present_label(boundary::before_clear, 9), refusal::foreign_queue);
    ui_ticket::refusal_inputs foreign_inputs;
    foreign_inputs.foreign_queue = foreign_inputs.mismatch = true;
    foreign.refused = ui_ticket::first_refusal(foreign_inputs);
    require(foreign.refused == refusal::foreign_queue && ui_ticket::pair_refusal(foreign.present, proposed) == refusal::foreign_queue, "A foreign-queue copy was not refused as foreign");
    ui_ticket::count_ticket(foreign, counters);
    // A scope change recreates the stamp entries at 0 and the tags' scope
    // differs: old stamps never pair.
    require(ui_ticket::pair_refusal(ui_ticket::present_label(boundary::before_clear, 0), proposed) == refusal::unstamped, "A recreated stamp entry paired");
    ui_ticket::ticket old_tag, new_tag;
    old_tag.token = new_tag.token = ui_ticket::token_label_of_tag(41);
    old_tag.token_generation = new_tag.token_generation = 41;
    old_tag.epoch = 1;
    new_tag.epoch = 2;
    require(!ui_ticket::same_frame(old_tag, new_tag), "Tags of two scopes were one frame");
    // FG suspended (known and requested): no Present is proven real, so no
    // present-space proposal; with Backbuffer tags the layer pairs in token
    // space through C_T, without them there is no pair.
    const bool span = ui_ticket::real_span(5, 1, ui_ticket::interposer::sl_interposer | ui_ticket::interposer::sl_dlss_g, true, true);
    const auto present_proposal = span ? change_set::proposed_layer_label(retained1, present_label) : 0u;
    require(!span && !present_proposal, "A present-space pair was proposed with FG suspended");
    const auto with_tags = ui_ticket::valid_label(label_space::token, change_set::token_proposal(true, 77));
    require(ui_ticket::pair_exact(ui_ticket::token_label_of_copy(77), with_tags), "The layer copy after a tagged frame did not pair by token");
    ui_ticket::count_gpu_verdict(ui_ticket::gpu_verdict::exact, true, counters);
    require(!change_set::token_proposal(false, 77), "A token was proposed without a ready Backbuffer tag");
    require(ui_ticket::pair_refusal(ui_ticket::token_label_of_copy(77), ui_ticket::label {}) == refusal::no_reference, "A layer copy without a Backbuffer tag had a reference");
    ui_ticket::count_gpu_verdict(ui_ticket::gpu_verdict::unproposed, false, counters);
    require(counters[idn::gpu_mismatch] == 2 && counters[idn::gpu_exact] == 1 && counters[idn::gpu_token_exact] == 1 && counters[idn::gpu_unproposed] == 1 && counters[idn::refused + std::size_t(refusal::foreign_queue)] == 1, "The label cases' counts moved");
    std::printf("SHADOW S3 E2: label cases: %s\n", ui_ticket::format_identity_counters(counters).c_str());
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
    // measure it hidden, never visible.
    require(counted[ui_counter::full_alpha] > 0 && counted[ui_counter::full_alpha_d_hidden] > 0 && !counted[ui_counter::full_alpha_d_visible] && counted[ui_counter::full_d_hidden] > 0 && !counted[ui_counter::full_d_visible], "The counters did not record Load Game's whole-frame alpha over a hidden scene (" + name + ")");
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
    require(!measured(s, s.samples[0]) && s.samples[1].evidence.scene.ran && s.samples[2].evidence.scene.ran && s.samples[2].sample_tick_ms - s.samples[1].sample_tick_ms <= ui_detection::scene::hold_ms, "The first samples did not open the gate and measure");
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

  void hidden_scene_shadow_and_clears() {
    // A hidden synthetic stream with an opaque, unaccepted cleared layer: an
    // informative full claim (H1 (b)).
    const auto hidden_layer = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 1000).opaque(kind::ui_layer, 1000).scene(-.02f, -.02f);
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
      require(!s.guard.hidden.until && !s.guard.hidden.last, "The shadow held a verdict");
      // With a claim, the shadow's own first sample holds nothing either:
      // the next two actionable samples enter the hold.
      alpha_auto_policy gated;
      gated.set_first_run(true);
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
        require(g.samples.back().evidence.scene.ran && (polls == 3) == g.guard.hidden.held(now) && (polls == 1) == !g.guard.hidden.last, "A shadow-only sample held, or the next two actionable ones did not enter");
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
      // decides itself (opaque on every pixel, never overridden) and stays
      // flat; forgotten again, H1 flattens it at once.
      establish();
      restore(session, layer.key(), 1);
      for (const auto until = now + 300; now < until;) {
        const auto &r = step(p);
        require(held() && r.flat() && r.source == ui_detection::source_layer && !r.decision.h1 && (r.scene_bits & ui_detection::per_frame_scene_hidden), "The layer accepted mid-hold did not stay flat as itself, or cleared the held verdict");
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
        require(!measured(s, sample) && !sample.evidence.claims, "A sample before the proof claimed or measured");
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
        require(!measured(s, sample) && !sample.evidence.claims && !sample.source_kind && pre_ui_matching(sample), "FG-on gameplay claimed, measured or decided, or its layer did not match");
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
        require(!measured(s, sample) && !sample.evidence.claims && !pre_ui_matching(sample), "The unproven menu claimed, measured or matched");
      }
      require(!s.scene_entered && !session.pre_ui_proven(layer) && session.stored().empty() && !s.guard.pre_ui.until, "The unproven menu held a verdict or earned the proof");
      check_counters(s, "Stellar Blade SDR menu before gameplay");
    }
  }

  void h1_stellar_blade_sdr_fg_session() {
    // The live failure fix 1 answers (S2b build, Stellar Blade SDR, FG set to
    // 2x, 07:19-07:21): the game suspends frame generation while a menu is
    // open. FG-on gameplay offers the layer, current alpha and the HUD-less
    // tag on real Presents (viewport 1: depth from Streamline), and generated
    // Presents hold. Nothing claims, so no sample measures D (the log's
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
      const auto kind = pacer.next(true, 1);
      auto p = fg_present(kind, fg_on, candidate::layer | candidate::current, pacer.tag);
      p.now_ms = now;
      const auto &r = s.step(p);
      if (kind == ui_mask::hudless_present::generated_frame) {
        ++generated;
        require(!r.detected && !r.flat(), "A generated FG-on Present detected or was flat");
      }
      now += 8;
    }
    const auto first = s.samples.front().sample_tick_ms;
    const auto earned = first_sample_from(s, first + alpha_trust_span_ms);
    require(generated > 100 && s.trust.size() == 1 && s.trust[0].tick == earned && s.trust[0].accepted == proof && heard == std::vector<std::string> {proof} && earned < visits[0].open, "FG-on gameplay did not prove the layer by its pixels 2 s in, before the first menu");
    for (const auto &sample : s.samples) {
      if (sample.evidence.candidates & candidate::hudless) {
        require(!measured(s, sample) && !sample.evidence.claims && pre_ui_matching(sample), "FG-on gameplay claimed, measured D or did not match");
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
    f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(500, 1000, 1000, 500);
    if (valid) {
      f.scene(.5f, .5f);
    }
    return f.words(in);
  }

  // The tick of the sample at which a restored pre-UI proof lapses (A3, fix
  // 1), zero when none does: its reconfirm clock runs from a testable sample
  // (the layer offered without coverage while the presented frame's evidence
  // is valid and visible) to the next sample that is not, and the proof
  // lapses at the testable sample by which the clock has run
  // alpha_trust_reconfirm_ms. The stream must not re-earn it.
  std::uint64_t pre_ui_lapse_tick(const sequence &s, std::uint64_t from) {
    std::uint64_t elapsed = 0, running = 0;
    for (const auto &sample : s.samples) {
      if (sample.sample_tick_ms < from) {
        continue;
      }
      const auto &e = sample.evidence;
      const bool testable = (e.candidates & candidate::layer) && !e.layer_covered && measured(s, sample) && e.scene.valid && e.scene.verdict == scene_verdict::visible;
      if (!testable) {
        elapsed += running ? sample.sample_tick_ms - running : 0;
        running = 0;
        continue;
      }
      running = running ? running : sample.sample_tick_ms;
      if (elapsed + (sample.sample_tick_ms - running) >= alpha_trust_reconfirm_ms) {
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
    // claims, and a hidden menu over it (presented D -0.01, the target 0.55)
    // stays 3D. The first-run shadow measures every sample to show the D
    // agreement, and acts on none.
    alpha_auto_policy session;
    session.set_first_run(true);
    sequence s(session, [](const gpu_inputs &in) {
      const bool menu = in.now_ms >= 20000 && in.now_ms < 25000;
      synthetic f;
      f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000);
      if (menu) {
        f.scene(-.01f, .55f).pre_ui_pixels(300, 1000, 1000, 700);
      } else {
        f.scene(.5f, .51f).pre_ui_pixels(600, 1000, 1000, 400);
      }
      return f.words(in);
    });
    const auto p = sb_sdr_present(candidate::layer | candidate::current);
    run(s, p, 10000, 30000);
    std::size_t agreeing = 0, hidden = 0;
    for (const auto &sample : s.samples) {
      const auto &e = sample.evidence;
      require(e.scene.ran && !e.claims && !pre_ui_matching(sample), "The effects target claimed, matched, or a sample was not measured by the shadow");
      agreeing += e.scene.valid && e.scene.verdict == scene_verdict::visible && std::fabs(e.scene.d - e.pre_ui_scene.d) <= .03f ? 1 : 0;
      hidden += e.scene.valid && e.scene.verdict == scene_verdict::hidden ? 1 : 0;
    }
    require(agreeing > 50 && hidden > 20, "The effects stream did not agree with the presented D in gameplay, or never read the menu hidden");
    for (const auto &f : s.frames) {
      require(!f.flat() && !f.source && !f.layer_proven && !(f.scene_bits & (ui_detection::per_frame_pre_ui_visible | ui_detection::per_frame_pre_ui_proven)), "The effects target was proven or flattened");
    }
    require(session.stored().empty() && !session.pre_ui_proven(p.signatures.of(kind::ui_layer)) && !s.guard.pre_ui.until && !s.scene_entered, "The effects target earned the proof, or a shadow-only sample held");
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
        f.alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(1000, 0, 0, 0);
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
        f.alpha(kind::ui_layer, 50).alpha(kind::current, 1000).opaque(kind::current, 1000).pre_ui_pixels(990, 900, 900, 5);
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
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(.5f, .5f).pre_ui_pixels(990, 900, 900, 5);
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
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(-.01f, .05f).pre_ui_pixels(1000, 0, 0, 0);
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
        f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(presented, pre_ui).pre_ui_pixels(990, 900, 900, 5);
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
          f.alpha(kind::ui_layer, 0, 500).alpha(kind::current, 1000).opaque(kind::current, 1000).scene(presented, buffer).pre_ui_pixels(matched, 1000, 1000, 1000 - matched);
          return f.words(in);
        });
        run(s, p, 10000, 23000);
        if (grain) {
          never_flat(s, "A scene buffer behind grain");
          for (const auto &sample : s.samples) {
            require(!measured(s, sample) && !sample.evidence.claims, "The unproven buffer behind grain claimed or measured");
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
        require(!measured(s, sample), "Dark gameplay without a claim measured");
      }
      check_counters(s, "accepted tag beside opaque presented alpha");
    }
    for (const bool shadow : {false, true}) {
      // Dead Space-like: diegetic UI only, so no informative claim (an
      // unaccepted current alpha over the whole frame, not opaque), with D
      // near the verdict bounds. Without the first-run shadow nothing
      // measures; with it every sample measures but none is actionable.
      alpha_auto_policy session;
      session.set_first_run(shadow);
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
        require(sample.evidence.claims == 0 && measured(s, sample) == shadow, "The Dead Space-like stream claimed, or measured without the shadow");
      }
      never_flat(s, "The Dead Space-like stream");
      require(!s.guard.hidden.until && !s.guard.hidden.last && !s.scene_entered, "The Dead Space-like stream acted on evidence");
      check_counters(s, shadow ? "Dead Space-like, shadow" : "Dead Space-like");
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
      require(c[ui_counter::full_alpha_d_hidden] > 0 && c[ui_counter::full_d_hidden] > 0 && !c[ui_counter::full_d_visible], "Load Game's samples were not counted as whole-frame alpha, then as H1");
      check_counters(s, "accepted opaque-full winner");
    }
    {
      // An accepted alpha winner opaque on every pixel is flat already: H1
      // holds the hidden verdict but never overrides it.
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
      for (const auto &f : s.frames) {
        require(f.flat() && f.source == 3 && !f.decision.h1 && f.decision.claims == candidate::backbuffer, "H1 overrode an accepted winner opaque on every pixel");
      }
      check_counters(s, "accepted winner opaque everywhere");
    }
  }

  // ---------------------------------------------------------------- H2 still screens (fix 2)

  // Rule H2 (docs/reshade-sbs.md, still screens without a UI source;
  // game3d_still_screen.h). The offer of Stellar Blade's SDR loading screen
  // (sb_s2b_0719.log, 07:19:57-07:20:06): the cleared output target almost
  // black without alpha (V1-invalid) and an opaque current alpha (0x48),
  // neither accepted, so no source decides and nothing claims; or E33's SDR
  // FG-off current alpha alone. Texel 5 carries the presented frame's D;
  // depth that is not the frame's own marks it invalid (H1 ignores it), but
  // the evidence pass still writes n and D, which H2 reads.
  gpu_model still_screen_model(std::function<float(std::uint64_t)> d_of, std::uint32_t current_covered = 1000) {
    return [d_of, current_covered](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 0, 1000).alpha(kind::current, current_covered).opaque(kind::current, current_covered).scene(d_of(in.now_ms), .3f);
      if (in.per_frame & ui_detection::per_frame_depth_not_current) {
        f.scene_valid = false;
        f.verdict = scene_verdict::none;
      }
      return f.words(in);
    };
  }

  present loading_screen_present() {
    present p;
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    return p;
  }

  // Presents every 16 ms from `from` to `to`; with reused_depth every eighth
  // Present's depth is not its own (about 12%, as on the live loading screen:
  // 93 of 827 publications by 07:20:01, 204 of 1767 by 07:20:06).
  void run_still(sequence &s, present p, std::uint64_t from, std::uint64_t to, bool reused_depth = false) {
    std::size_t n = 0;
    for (auto now = from; now < to; now += 16, ++n) {
      p.now_ms = now;
      p.depth_current = !reused_depth || n % 8 != 7;
      s.step(p);
    }
  }

  // The stream's H2 events that entered, ended an entered episode, or ended a
  // run before it entered.
  std::vector<sequence::still_event> still_events_of(const sequence &s, bool (*which)(const still_screen::observation &)) {
    std::vector<sequence::still_event> result;
    std::copy_if(s.still_events.begin(), s.still_events.end(), std::back_inserter(result), [which](const sequence::still_event &e) {
      return which(e.what);
    });
    return result;
  }

  bool still_entered(const still_screen::observation &o) {
    return o.entered;
  }

  bool still_ended(const still_screen::observation &o) {
    return o.ended;
  }

  bool still_short(const still_screen::observation &o) {
    return o.short_run;
  }

  void h2_still_screens() {
    // Stellar Blade's SDR loading screen: D -0.014..-0.068 on every sample,
    // every cell still.
    const auto sb_d = [](std::uint64_t now) {
      return -.014f - .054f * float((now / 100) % 10) / 9.f;
    };
    // The run enters at the first sample whose tick is at least run_ms after
    // its first passing sample, so within one sample interval of it.
    const auto entered_at_run_ms = [](const sequence::still_event &e) {
      return e.run.duration_ms() >= ui_detection::still::run_ms && e.run.duration_ms() <= ui_detection::still::run_ms + sample_interval_ms + 16;
    };
    {
      // The default shadow: the run enters at its 2 s sample and only
      // reports it; no decision changes. About 12% of the samples read depth
      // that is not the frame's own (invalid for H1) and keep the run.
      alpha_auto_policy session;
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      run_still(s, p, 10000, 19000, true);
      require(!s.samples.empty() && !s.samples[0].evidence.still_compared && s.samples[0].evidence.scene.ran, "The first sample compared cells without a previous mean, or was not measured");
      std::size_t stale = 0;
      for (const auto &sample : s.samples) {
        require(!sample.evidence.claims && !sample.source_kind, "The loading screen claimed or decided");
        stale += sample.evidence.scene.ran && !sample.evidence.scene.valid && sample.evidence.scene.n >= ui_detection::scene::min_edges ? 1 : 0;
      }
      const auto entered = still_events_of(s, still_entered);
      require(stale >= 5 && entered.size() == 1 && entered[0].counted && entered_at_run_ms(entered[0]) && entered[0].run.first == s.samples[1].sample_tick_ms && still_events_of(s, still_ended).empty() && still_events_of(s, still_short).empty(), "The SDR loading screen's shadow did not enter once at its 2 s sample, or reused depth ended its run");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        require(!f.source && !f.still_bits && f.still_scope && !f.decision.still && !f.decision.own_source, "The shadow changed a decision or pushed the still word");
        require((f.consumed.still.phase == still_screen::phase::active) == (i >= entered[0].frame) && !f.consumed.still.enabled, "The render's still phase did not follow the run");
      }
      // The screen starts to change (90% of the cells still): the first
      // moving sample read ends the episode.
      p.still_share = .9f;
      run_still(s, p, 19000, 19600, true);
      const auto ended = still_events_of(s, still_ended);
      require(ended.size() == 1 && ended[0].counted && ended[0].what.reason == still_screen::end_reason::moving && ended[0].frame == poll_of(s, first_sample_from(s, 19000)) && ended[0].what.ended_episode.duration_ms() >= 6500, "The first moving sample did not end the episode");
      require(session.counters()[ui_counter::still_entered] == 1 && session.counters()[ui_counter::still_released] == 1, "The shadow's episode was not counted");
      check_counters(s, "SB SDR loading screen, shadow");
    }
    {
      // Enabled (UIFlattenStillScreens=1): flat as source 11 from the frame
      // that reads the 2 s sample until the frame that reads the first
      // moving one; the T1 hold store keeps the decision before H2.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      run_still(s, p, 10000, 19000, true);
      p.still_share = .9f;
      run_still(s, p, 19000, 19600, true);
      const auto entered = still_events_of(s, still_entered), ended = still_events_of(s, still_ended);
      require(entered.size() == 1 && ended.size() == 1 && entered_at_run_ms(entered[0]) && ended[0].what.reason == still_screen::end_reason::moving, "The enabled loading screen did not enter once and end on its first moving sample");
      const auto release = poll_of(s, first_sample_from(s, 19000));
      require(ended[0].frame == release && s.frames[release].now_ms - 19000 <= sample_interval_ms + 2 * 16, "The release took longer than one sample interval and its readback");
      bool sampled_flat = false;
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        const bool flat = i >= entered[0].frame && i < release;
        require(f.source == (flat ? ui_detection::source_still : 0u) && f.still_bits == (flat ? ui_detection::still::flatten : 0u) && (!flat || (f.flat() && f.decision.still && !f.decision.next.source && !f.decision.own_source)), "The enabled loading screen was not flat as source 11 exactly from its entry to its release, at frame " + std::to_string(i));
        sampled_flat = sampled_flat || f.consumed.source_kind == ui_detection::source_still;
      }
      require(sampled_flat && session.counters().decided(ui_detection::source_still) > 0, "No sample or counter showed source 11");
      check_counters(s, "SB SDR loading screen, enabled");
    }
    {
      // E33 SDR FG-off gameplay (e33_s2b.log 07:43:02-07:43:23): the current
      // alpha alone, unaccepted; the presented frame reads visible except for
      // short dips whose hidden runs last at most 547 ms (shadow_hidden_ms).
      // Three of them read D <= 0.05 (0.025, 0.035, 0.040): even every cell
      // still and flattening enabled, they end as short runs and never enter.
      const std::array<std::pair<std::uint64_t, float>, 16> logged {{{20, .380f}, {1791, .025f}, {2796, .424f}, {5315, .035f}, {6330, .211f}, {7631, .112f}, {9047, .193f}, {10289, .074f}, {11294, .539f}, {12897, .099f}, {13894, .229f}, {15298, .072f}, {16291, .166f}, {17604, .040f}, {18608, .362f}, {20113, .111f}}};
      const auto e33_d = [logged](std::uint64_t now) {
        const auto t = now - 10000;
        float visible = .38f;
        for (const auto &[at, d] : logged) {
          if (d < .15f && t + 273 >= at && t < at + 274) {
            return d;
          }
          if (d >= .15f && at <= t) {
            visible = d;
          }
        }
        return visible;
      };
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(e33_d));
      present q;
      q.offered = candidate::current;
      run_still(s, q, 10000, 31000);
      const auto shorts = still_events_of(s, still_short);
      std::uint64_t longest = 0;
      for (const auto &e : shorts) {
        longest = std::max(longest, e.what.short_ms);
      }
      require(still_events_of(s, still_entered).empty() && shorts.size() == 3 && longest > 0 && longest <= 547, "E33's gameplay dips entered H2, or were not three short runs of at most 547 ms: " + std::to_string(shorts.size()) + " / " + std::to_string(longest));
      for (const auto &f : s.frames) {
        require(!f.source && !f.still_bits, "E33 gameplay flattened");
      }
      require(session.counters()[ui_counter::still_short] == 3 && !session.counters()[ui_counter::still_entered], "E33's short runs were not counted");
      check_counters(s, "E33 SDR FG-off gameplay dips");
    }
    {
      // Moving dark grainy gameplay: D at most 0.05 on every sample, but only
      // 80% of the cells still. It never starts a run.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model([](std::uint64_t now) { return (now / 100) % 2 ? .02f : .05f; }));
      present q;
      q.offered = candidate::current;
      q.still_share = .8f;
      run_still(s, q, 10000, 16000);
      require(s.still_events.empty(), "Moving dark gameplay started a still run");
      for (const auto &f : s.frames) {
        require(!f.source && f.consumed.still.phase == still_screen::phase::none, "Moving dark gameplay ran or flattened");
      }
      check_counters(s, "moving dark grainy gameplay");
    }
    {
      // The measured residual risk: an idle, still, very dark and finely
      // grained SDR scene. Grain collapses D (0.02) while the cell means stay
      // within 1/255 (99% still): it would flatten after 2 s and stay flat
      // until it moves. The default shadow only reports it; the shadow period
      // measures how often this happens in real gameplay.
      alpha_auto_policy session;
      sequence s(session, still_screen_model([](std::uint64_t) { return .02f; }));
      present q;
      q.offered = candidate::current;
      q.still_share = .99f;
      run_still(s, q, 10000, 13000);
      const auto entered = still_events_of(s, still_entered);
      require(entered.size() == 1 && entered_at_run_ms(entered[0]), "The still grainy dark scene (the measured residual risk) did not would-flatten after 2 s");
      for (const auto &f : s.frames) {
        require(!f.source && !f.still_bits, "The shadow flattened the still grainy dark scene");
      }
      check_counters(s, "still grainy dark scene (the measured residual risk)");
    }
    for (const auto &[color_space, format] : {std::pair<std::uint32_t, std::uint32_t> {pq, 24}, std::pair<std::uint32_t, std::uint32_t> {2, 10}}) {
      // HDR output (PQ or scRGB) is out of scope: the same loading screen
      // with flattening enabled never runs, and no evidence pass runs for it.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      p.signatures = signatures_in(color_space);
      p.output_format = format;
      run_still(s, p, 10000, 14000, true);
      require(s.still_events.empty(), "An HDR loading screen started a still run");
      for (const auto &f : s.frames) {
        require(!f.still_scope && !f.source && !f.still_bits && !f.consumed.still.scope, "An HDR frame was in H2's scope");
      }
      for (const auto &sample : s.samples) {
        require(!sample.evidence.scene.ran && !sample.evidence.still_compared, "An HDR sample ran the evidence passes for H2");
      }
      check_counters(s, "HDR loading screen");
    }
    {
      // E33 with its current alpha accepted: it decides an empty mask (source
      // 4 at 0% coverage), which H2 respects. Only the first sample, before
      // any source decided, runs the evidence passes.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      restore(session, key(kind::current), 1);
      sequence s(session, still_screen_model([](std::uint64_t) { return .02f; }, 0));
      present q;
      q.offered = candidate::current;
      run_still(s, q, 10000, 14000);
      std::size_t measured_samples = 0;
      for (const auto &sample : s.samples) {
        measured_samples += sample.evidence.scene.ran ? 1 : 0;
      }
      require(s.still_events.empty() && measured_samples <= 1, "An accepted empty decision started a still run, or kept measuring for H2");
      for (const auto &f : s.frames) {
        require(f.source == 4u && !f.covered && !f.still_bits && f.consumed.still.phase == still_screen::phase::none, "The accepted empty current alpha did not decide, or H2 overrode it");
      }
      check_counters(s, "E33 accepted current alpha at 0% coverage");
    }
    {
      // An identity change restarts the 2 s: FG toggled (another epoch)
      // during a pending run, then Stellar Blade suspending FG in a menu
      // (the depth moves to another provider: another viewport) during an
      // active one, which ends it at once (reason identity).
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      run_still(s, p, 10000, 11500);
      ++p.epoch;
      run_still(s, p, 11500, 15000);
      ++p.viewport;
      const std::size_t suspended = s.frames.size();
      run_still(s, p, 15000, 18000);
      const auto entered = still_events_of(s, still_entered), ended = still_events_of(s, still_ended), shorts = still_events_of(s, still_short);
      require(shorts.size() == 1 && !shorts[0].counted && shorts[0].what.reason == still_screen::end_reason::identity && shorts[0].now_ms == 11500, "An FG toggle did not end the pending run");
      // The run it ended reaches the status once (the UI line's short_max_ms).
      std::size_t published = 0;
      for (const auto &f : s.frames) {
        published += f.consumed.evidence.still_short_ms == shorts[0].what.short_ms && f.now_ms == 11500 ? 1 : 0;
      }
      require(shorts[0].what.short_ms >= 1000 && published == 1, "The run an identity change ended did not reach the status");
      require(entered.size() == 2 && entered[0].run.first >= 11500 && entered[0].now_ms >= 11500 + ui_detection::still::run_ms && entered[1].run.first >= 15000 && entered[1].now_ms >= 15000 + ui_detection::still::run_ms, "A run entered less than 2 s after an identity change");
      require(ended.size() == 1 && !ended[0].counted && ended[0].what.reason == still_screen::end_reason::identity && ended[0].frame == suspended, "The suspended FG's identity change did not end the active episode at once");
      for (std::size_t i = suspended; i != s.frames.size(); ++i) {
        require((s.frames[i].source == ui_detection::source_still) == (i >= entered[1].frame), "A frame after the identity change was flat before its new run entered");
      }
      check_counters(s, "identity changes during a still run");
    }
    {
      // Auto -> manual On during an active run leaves scope: the episode ends
      // at once (reason scope); back in Auto a new run needs its own 2 s. The
      // last Auto frame submits a sample, which the first manual frame reads.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      run_still(s, p, 10000, 13000);
      std::uint64_t switch_at = 13000;
      for (; !s.frames.back().submitted; switch_at += 16) {
        p.now_ms = switch_at;
        s.step(p);
      }
      require(still_events_of(s, still_entered).size() == 1 && s.frames.back().source == ui_detection::source_still, "The run did not enter before the mode change");
      session.set_manual(true);
      const std::size_t manual = s.frames.size();
      run_still(s, p, switch_at, 13500);
      require(s.frames[manual].polled, "The first manual frame did not read the Auto sample in flight");
      session.set_automatic();
      const std::size_t automatic = s.frames.size();
      run_still(s, p, 13500, 16000);
      const auto ended = still_events_of(s, still_ended), entered = still_events_of(s, still_entered);
      require(ended.size() == 1 && !ended[0].counted && ended[0].what.reason == still_screen::end_reason::scope && ended[0].frame == manual, "Manual On did not end the episode at once with reason scope");
      for (std::size_t i = manual; i != automatic; ++i) {
        require(!s.frames[i].still_scope && s.frames[i].source != ui_detection::source_still && !s.frames[i].still_bits, "A manual frame was in H2's scope or flat as 11");
      }
      require(entered.size() == 2 && entered[1].run.first >= 13500 && entered[1].now_ms >= 13500 + ui_detection::still::run_ms, "Back in Auto the run did not need its own 2 s");
      // The Auto sample still in flight when manual On began is read out of
      // scope: it starts no run, so no short run follows the scope end.
      require(still_events_of(s, still_short).empty() && !session.counters()[ui_counter::still_short], "A sample in flight across the scope end started a run");
      check_counters(s, "Auto to manual On during a still run");
    }
    {
      // Enabled, an active episode, then real frames that offer nothing in
      // the same epoch while an accepted candidate is missing (T1 grace,
      // accepted_missing): they are never sampled, so the first ends the run
      // (unmeasured) and none is flat; offers returning need a new 2 s run.
      // The accepted UI colour tag (empty, source 2) is offered first, so
      // the adopted bits keep it and the loading screen's 0x48 frames are
      // flagged accepted_missing too.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      restore(session, key(kind::ui_color), 1);
      sequence s(session, still_screen_model(sb_d));
      auto tagged = loading_screen_present();
      tagged.offered |= candidate::ui_color;
      run_still(s, tagged, 10000, 10500);
      auto p = loading_screen_present();
      run_still(s, p, 10500, 14000);
      const auto entered = still_events_of(s, still_entered);
      require(entered.size() == 1 && s.frames.back().source == ui_detection::source_still && (s.frames.back().hold.per_frame & ui_detection::per_frame_accepted_missing), "The loading screen with the tag missing did not enter and flatten");
      const std::size_t zero_offer = s.frames.size();
      present nothing = p;
      nothing.offered = 0;
      run_still(s, nothing, 14000, 15000);
      const std::size_t offers_return = s.frames.size();
      run_still(s, p, 15000, 16000);
      for (std::size_t i = zero_offer; i != offers_return; ++i) {
        require(s.frames[i].grace && !s.frames[i].submitted && s.frames[i].source != ui_detection::source_still && !s.frames[i].still_bits, "A zero-offer grace frame was flat as 11 or sampled");
      }
      for (std::size_t i = offers_return; i != s.frames.size(); ++i) {
        require(s.frames[i].source != ui_detection::source_still && !s.frames[i].still_bits, "A frame after the zero-offer stretch was flat before a new 2 s run");
      }
      const auto ended = still_events_of(s, still_ended);
      require(ended.size() == 1 && !ended[0].counted && ended[0].what.reason == still_screen::end_reason::unmeasured && ended[0].frame == zero_offer, "The first zero-offer frame did not end the episode as unmeasured");
      check_counters(s, "zero-offer grace frames after a still episode");
    }
    {
      // Enabled, an active episode, then an inactive gap (nothing offered,
      // nothing accepted missing): the first inactive render ends the run,
      // and the frames that offer again are not flat before a new 2 s run.
      alpha_auto_policy session;
      session.set_still_flatten(true);
      sequence s(session, still_screen_model(sb_d));
      auto p = loading_screen_present();
      run_still(s, p, 10000, 13000);
      require(still_events_of(s, still_entered).size() == 1 && s.frames.back().source == ui_detection::source_still, "The run did not enter before the gap");
      const std::size_t gap = s.frames.size();
      present nothing = p;
      nothing.offered = 0;
      run_still(s, nothing, 13000, 13500);
      const std::size_t back = s.frames.size();
      run_still(s, p, 13500, 15000);
      for (std::size_t i = gap; i != back; ++i) {
        require(!s.frames[i].active && !s.frames[i].still_bits, "An inactive frame ran detection or pushed the still word");
      }
      for (std::size_t i = back; i != s.frames.size(); ++i) {
        require(s.frames[i].source != ui_detection::source_still && !s.frames[i].still_bits, "A frame offering again after an inactive gap was flat before a new 2 s run");
      }
      const auto ended = still_events_of(s, still_ended);
      require(ended.size() == 1 && !ended[0].counted && ended[0].what.reason == still_screen::end_reason::unmeasured && ended[0].frame == gap, "The first inactive render did not end the episode as unmeasured");
      check_counters(s, "inactive gap after a still episode");
    }
    {
      // The default shadow leaves the acceptance ledger as it was before
      // fix 2. Stellar Blade SDR, second session: the layer's pre-UI proof
      // restored provisional; gameplay offers the layer (uncovered, never
      // matching) beside a HUD-less image that does not decide, so the
      // HUD-less image is the offer's pre-UI image and no other rule
      // measures. H2 measures every sample frame (visible, so it never
      // runs), but its measurements never reach the ledger: the proof's
      // reconfirm clock stays paused and it does not lapse.
      alpha_auto_policy session;
      restore(session, sb_sdr_proof(), 1);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 0).change_set(10).scene(.4f, .4f);
        return f.words(in);
      });
      run_still(s, sb_sdr_present(candidate::layer | candidate::hudless), 10000, 80000);
      std::uint64_t first = 0, last = 0;
      for (std::size_t i = 0; i != s.samples.size(); ++i) {
        const auto &e = s.samples[i].evidence;
        require(!e.claims && !s.samples[i].source_kind, "The gameplay stream claimed or decided");
        if (e.scene.ran) {
          require(s.h2_only[i] && e.scene.valid && e.scene.verdict == scene_verdict::visible, "A sample measured for a rule other than H2, or not visible");
          first = first ? first : s.samples[i].sample_tick_ms;
          last = s.samples[i].sample_tick_ms;
        }
      }
      require(last - first > alpha_trust_reconfirm_ms, "H2 did not measure visible samples for longer than the reconfirm time");
      require(session.stored() == sb_sdr_proof() && session.pre_ui_proven(sb_sdr_signatures().of(kind::ui_layer)) && !session.counters()[ui_counter::trust_lapsed], "H2's measurements ran the restored proof's reconfirm clock: it lapsed in the shadow");
      check_counters(s, "H2 measurements kept from the ledger");
    }
  }

  // ---------------------------------------------------------------- Pre-UI change sets (fix 3)

  // Fix 3 (docs/reshade-sbs.md, UI decision framework; game3d_ui_change_set.h):
  // Stellar Blade in SDR since 10-03 21:52 accepts current:24:srgb from FG-off
  // gameplay, where the presented alpha equals the UI colour tag; in menus
  // that alpha is 1.0 on every pixel (shapeless), so P1 pins the whole menu
  // flat. Its cleared output target is the pre-UI scene image without alpha
  // (V1-invalid), proven by gameplay (pre_ui:87:srgb), and the copy shows the
  // previous Present's frame, so it pairs exactly with the retained Present
  // one back while frame generation is off or suspended. A page's counts over
  // 1000 pixels: the exact pair (offset 1), the one-frame-late pair against
  // the current Present (offset 0: texel 11's mismatch; the dump's own pair),
  // the pair against the Present two back, and the layer's lit pixels.
  struct pair_counts {
    std::uint32_t changed, filtered, tiles;
  };

  struct sb_page {
    pair_counts exact, late;
    std::uint32_t two_back, lit;
  };

  // The Equipment tab (dump 033 scaled: tab row, icon panel, prompts and the
  // bottom dim band, about 5.4%; the late pair also holds the idling
  // character's motion and still passes the tile test, 138 tiles).
  constexpr sb_page sb_equipment {{52, 51, 150}, {56, 54, 138}, 60, 260};
  // The Settings page (dumps 035, 468, 962): changed almost everywhere, no
  // matching tile.
  constexpr sb_page sb_settings {{600, 590, 0}, {610, 600, 0}, 615, 300};
  // A loading screen (sb_fix1_b.log 21:51:36-52): the pre-UI image is black,
  // lit on 0.2% (0.04-0.46% logged), below the 1% a change set needs.
  constexpr sb_page sb_loading {{30, 28, 200}, {31, 29, 200}, 32, 2};

  std::string sb_sdr_current() {
    return sb_sdr_signatures().of(kind::current).key();
  }

  // A session that restored the layer's proof and, with current, the
  // accepted current alpha (provisional; no sample below is testable for
  // longer than A3's 60 s).
  void restore_sb_sdr(alpha_auto_policy &session, bool current = true) {
    restore(session, current ? sb_sdr_current() + "," + sb_sdr_proof() : sb_sdr_proof(), current ? 2 : 1);
  }

  // A page: the layer without alpha, the opaque current alpha (accepted or
  // not by the session), texel 11 against the current Present, and the layer
  // pair of the pushed offset (b2 word 5); edit adds a stream's evidence.
  gpu_model sb_sdr_page(const sb_page page, std::function<void(synthetic &, const gpu_inputs &)> edit = {}) {
    return [page, edit](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 0, 1000).alpha(kind::current, 1000).opaque(kind::current, 1000);
      f.pre_ui_pixels(1000 - page.late.changed, page.lit, 900, page.late.changed);
      const auto offset = (in.rules & ui_detection::change_set::pair_mask) >> ui_detection::change_set::pair_shift;
      const pair_counts at = offset == 1 ? page.exact : offset == 2 ? pair_counts {page.two_back, page.two_back, page.exact.tiles} : page.late;
      f.layer_pair(at.changed, at.filtered, at.tiles, page.exact.changed, page.two_back);
      if (edit) {
        edit(f, in);
      }
      return f.words(in);
    };
  }

  // A Stellar Blade SDR menu Present with frame generation off or suspended
  // (viewport 0: depth from NGX).
  present sb_sdr_menu_present() {
    return sb_sdr_present(candidate::layer | candidate::current, 0);
  }

  // Hidden presented frame, visible pre-UI image (dump 962: 0.031, 0.588).
  void hidden_menu(synthetic &f, const gpu_inputs &) {
    f.scene(.031f, .588f);
  }

  // The samples that measured the layer, with their shadows.
  std::vector<const change_set::shadow_sample *> shadows_of(const sequence &s) {
    std::vector<const change_set::shadow_sample *> result;
    for (const auto &shadow : s.shadows) {
      if (shadow) {
        result.push_back(&*shadow);
      }
    }
    return result;
  }

  bool starts_with(const std::string &text, const std::string &prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
  }

  void fix3_equipment_page() {
    // (a) The Equipment tab over a live backdrop, the exact retained pairing,
    // the current alpha accepted and uniformly opaque. Switch off (the
    // default shadow): flat as source 4 on every frame, and every sample's
    // shadow is retained, valid, verified by its offsets, and would refine to
    // 12. Switch on: source 12 from the first frame, the S1 winner refined
    // (h1 word), only the changed pixels pinned; the ledger never moves.
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(enabled);
      const auto stored = session.stored();
      sequence s(session, sb_sdr_page(sb_equipment));
      run(s, sb_sdr_menu_present(), 10000, 13000);
      for (const auto &f : s.frames) {
        require(f.detected && f.pairing == change_set::layer_pairing {change_set::pair_class::retained, 1u} && f.pre_ui_offered == enabled, "The Equipment page's layer did not pair with the Present one back, or the switch did not gate its offer");
        require(f.rules == ((enabled ? ui_detection::rules::pin_only_ui : 0u) | ui_detection::change_set::shadow | (1u << ui_detection::change_set::pair_shift) | ui_detection::change_set::retained_1 | ui_detection::change_set::retained_2), "The Equipment page pushed other rule bits");
        if (enabled) {
          require(f.source == ui_detection::source_pre_ui && f.covered == sb_equipment.exact.changed && f.covered < f.pixels && f.decision.refined && f.decision.shapeless && f.decision.s1_source == ui_detection::source_pre_ui && (ui_selection::h1_word(f.decision) & ui_detection::h1_refined) && !f.decision.reused, "The enabled Equipment page did not pin only its changed pixels as a refined source 12");
        } else {
          require(f.source == 4u && f.flat() && !f.decision.refined && f.decision.shapeless && !(f.rules & ui_detection::rules::pin_only_ui), "The shadow changed the Equipment page's flat current alpha");
        }
      }
      const auto shadows = shadows_of(s);
      require(shadows.size() == s.samples.size() && shadows.size() >= 20, "Not every sample measured the layer pair");
      for (const auto *shadow : shadows) {
        require(shadow->pairing.kind == change_set::pair_class::retained && shadow->pairing.offset == 1 && shadow->valid && shadow->would_refine && shadow->would_source == ui_detection::source_pre_ui && !shadow->would_decide && shadow->verdict == change_set::pair_verdict::verified && shadow->shapeless && shadow->enabled == enabled, "An Equipment sample's shadow was not retained, valid, verified and refining");
        require(shadow->winner == (enabled ? ui_detection::source_pre_ui : 4u) && shadow->applied_source == (enabled ? ui_detection::source_pre_ui : 4u) && shadow->counts.filtered == sb_equipment.exact.filtered, "An Equipment sample's own decision was not the frame's");
      }
      for (const auto &sample : s.samples) {
        require(sample.change_set.measured && sample.change_set.valid && sample.change_set.would_refine && sample.change_set.enabled == enabled && sample.evidence.refined == enabled, "A sample's change-set status or refined bit was wrong");
      }
      const std::string flag = enabled ? "1" : "0";
      require(!s.shadow_lines.empty() && starts_with(s.shadow_lines[0], "Sunshine UI change set: pairing=retained offset=1 fg=0 UIPinOnlyUI=" + flag + " changed=52 unchanged=948 nonfinite=0 matching_tiles=150 lit=260 layer_covered=0 valid=1 filtered=51 offsets={0=56 1=52 2=60} pair=verified judge={kind=none pixels=0 tp=0 precision=- recall=- iou=-} winner=" + (enabled ? "12" : "4") + " shapeless=1 would_refine=1 would_source=12 applied_source=" + (enabled ? "12" : "4") + " pixels=1000"), "The shadow's first line differs: " + (s.shadow_lines.empty() ? std::string("none") : s.shadow_lines[0]));
      require(s.shadow_lines.size() >= 3 && s.shadow_lines.size() <= 4, "The shadow lines were not throttled to one per second");
      require(session.stored() == stored && s.trust.empty(), "The Equipment page moved the ledger");
      const auto totals = session.counters();
      require(totals[ui_counter::change_set_samples] && totals[ui_counter::change_set_retained] == totals[ui_counter::change_set_samples] && totals[ui_counter::change_set_valid] == totals[ui_counter::change_set_samples] && totals[ui_counter::change_set_would_refine] == totals[ui_counter::change_set_samples] && totals[ui_counter::change_set_pair_verified] == totals[ui_counter::change_set_samples] && !totals[ui_counter::change_set_pair_contradicted] && !totals[ui_counter::change_set_would_decide], "The change_set counters do not count every retained, valid, verified sample");
      require((totals[ui_counter::refined] > 0) == enabled && (totals.decided(ui_detection::source_pre_ui) > 0) == enabled && (totals.decided(4) > 0) == !enabled, "The refined and decided counters do not follow the switch");
      check_counters(s, enabled ? "SB SDR Equipment page, UIPinOnlyUI=1" : "SB SDR Equipment page, shadow");
    }
  }

  // The sources of one stream's detected frames.
  std::vector<std::uint32_t> sources_of(const sequence &s) {
    std::vector<std::uint32_t> result;
    for (const auto &f : s.frames) {
      result.push_back(f.source);
    }
    return result;
  }

  void fix3_settings_page() {
    // (b) The Settings page: the layer pair changed almost everywhere and no
    // tile matches, so the set is invalid whatever the switch. With the
    // current alpha accepted it stays flat as 4; with nothing but the proof
    // (before 10-03) and a held hidden verdict, H1 (d) shows it flat as 8
    // exactly as with the switch off.
    for (const bool current : {true, false}) {
      std::array<std::vector<std::uint32_t>, 2> modes;
      for (const bool enabled : {false, true}) {
        alpha_auto_policy session;
        restore_sb_sdr(session, current);
        session.set_pin_only_ui(enabled);
        sequence s(session, sb_sdr_page(sb_settings, hidden_menu));
        run(s, sb_sdr_menu_present(), 10000, 12500);
        for (const auto &f : s.frames) {
          require(f.pre_ui_offered == enabled && !(f.decision.valid_bits & candidate::pre_ui) && !f.decision.refined && f.source != ui_detection::source_pre_ui, "The Settings page's change set was valid or refined");
          if (current) {
            require(f.source == 4u && f.flat(), "The Settings page with the accepted current alpha was not flat as 4");
          }
        }
        for (const auto *shadow : shadows_of(s)) {
          require(!shadow->valid && !shadow->counts.matching_tiles && !shadow->would_refine && shadow->would_source == shadow->applied_source, "The Settings page's shadow was valid or would refine");
        }
        if (!current) {
          std::size_t flat = 0;
          for (const auto &f : s.frames) {
            flat += f.source == 8u && f.flat() ? 1 : 0;
          }
          require(flat > 50 && s.frames.back().source == 8u && s.scene_entered == 1, "H1 did not show the Settings page flat as 8 under the held hidden verdict");
        }
        require(!session.counters()[ui_counter::refined] && !session.counters().decided(ui_detection::source_pre_ui), "The Settings page counted a refined or source-12 frame");
        check_counters(s, std::string("SB SDR Settings page") + (current ? ", current accepted" : ", H1") + (enabled ? ", UIPinOnlyUI=1" : ""));
        modes[enabled] = sources_of(s);
      }
      require(modes[0] == modes[1], "The switch changed a Settings page decision");
    }
  }

  void fix3_loading_screen() {
    // (c) A loading screen whose pre-UI image is black (lit on 0.2%): a black
    // image is no scene, so its small changed set is invalid by the lit rule
    // and the frame stays flat (4 by the accepted current alpha; 8 under a
    // held hidden verdict without it) in both modes.
    for (const bool current : {true, false}) {
      std::array<std::vector<std::uint32_t>, 2> modes;
      for (const bool enabled : {false, true}) {
        alpha_auto_policy session;
        restore_sb_sdr(session, current);
        session.set_pin_only_ui(enabled);
        sequence s(session, sb_sdr_page(sb_loading, hidden_menu));
        run(s, sb_sdr_menu_present(), 10000, 12500);
        for (const auto &f : s.frames) {
          require(!(f.decision.valid_bits & candidate::pre_ui) && !f.decision.refined && (!current || (f.source == 4u && f.flat())), "The black loading screen's change set was valid, or the frame was not flat");
        }
        for (const auto *shadow : shadows_of(s)) {
          require(!shadow->valid && shadow->lit == sb_loading.lit && shadow->counts.matching_tiles == 200 && !shadow->would_refine, "The loading screen's shadow passed the lit rule");
        }
        if (!current) {
          require(s.frames.back().source == 8u && s.frames.back().flat(), "The loading screen under a held hidden verdict was not flat as 8");
        }
        check_counters(s, std::string("SB SDR loading screen") + (enabled ? ", UIPinOnlyUI=1" : ""));
        modes[enabled] = sources_of(s);
      }
      require(modes[0] == modes[1], "The switch changed a loading screen decision");
    }
  }

  void fix3_fg_on() {
    // (d) Stellar Blade SDR gameplay with frame generation on: the UI colour
    // tag is accepted and selective (2), and decides with or without the
    // switch. The layer pairs late, so the pre-UI change set is never offered;
    // the shadow still measures the late pair against the current Present
    // and judges it against the tag (dump 478: precision 0.39, recall 0.84,
    // IoU 0.36, 125 tiles: invalid).
    const auto model = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_color, 30).alpha(kind::ui_layer, 0, 1000).alpha(kind::current, 1000).opaque(kind::current, 1000);
      f.pre_ui_pixels(920, 880, 900, 80).layer_pair(80, 70, 125, 85, 90).judge(ui_detection::change_set::judge::ui_color, 30, 26);
      return f.words(in);
    };
    std::array<std::vector<std::uint32_t>, 2> modes;
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      restore(session, sb_sdr_signatures().of(kind::ui_color).key() + "," + sb_sdr_current() + "," + sb_sdr_proof(), 3);
      session.set_pin_only_ui(enabled);
      sequence s(session, model);
      auto p = sb_sdr_present(candidate::ui_color | candidate::layer | candidate::current);
      p.fg_known_off = false;
      run(s, p, 10000, 12500);
      for (const auto &f : s.frames) {
        require(f.source == 2u && f.covered == 30 && !f.pre_ui_offered && f.pairing == change_set::layer_pairing {change_set::pair_class::late, 0u} && !f.decision.refined, "FG-on gameplay did not decide the tag, or offered the late pair");
      }
      const auto shadows = shadows_of(s);
      require(shadows.size() == s.samples.size() && !shadows.empty(), "FG-on samples did not measure the late pair");
      for (const auto *shadow : shadows) {
        require(shadow->fg && shadow->pairing.kind == change_set::pair_class::late && shadow->verdict == change_set::pair_verdict::none && !shadow->valid && !shadow->would_refine && shadow->would_source == 2u && std::fabs(shadow->precision - 26. / 70.) < 1e-9 && std::fabs(shadow->recall - 26. / 30.) < 1e-9 && std::fabs(shadow->iou - 26. / 74.) < 1e-9, "The late pair's shadow or judge agreement is wrong");
      }
      require(!s.shadow_lines.empty() && s.shadow_lines[0].find("pairing=late offset=0 fg=1") != std::string::npos && s.shadow_lines[0].find("judge={kind=ui_color pixels=30 tp=26 precision=0.371 recall=0.867 iou=0.351}") != std::string::npos, "The late pair's shadow line lacks its judge: " + (s.shadow_lines.empty() ? std::string("none") : s.shadow_lines[0]));
      const auto totals = session.counters();
      require(totals[ui_counter::change_set_late] == totals[ui_counter::change_set_samples] && totals[ui_counter::change_set_samples] && !totals[ui_counter::change_set_valid] && !totals[ui_counter::refined], "The late pair's counters are wrong");
      check_counters(s, std::string("SB SDR FG on") + (enabled ? ", UIPinOnlyUI=1" : ""));
      modes[enabled] = sources_of(s);
    }
    require(modes[0] == modes[1], "The switch changed an FG-on decision");
  }

  void fix3_late_pairing_never_acts() {
    // (e) With the switch on, the Equipment page never decides from a pair
    // that is not exact by Present counting: a copy of the current interval
    // (presents_since_copy 0), frame generation not known off (on, or an
    // unknown mode: no observation, no Streamline FG observer, another SDK's
    // FG), or the Present it shows not retained. Each stays flat by the
    // accepted current alpha.
    struct variant {
      const char *name;
      std::uint32_t presents_ago;
      bool fg_known_off, retained_1;
      change_set::pair_class kind;
    };

    for (const auto &v : {variant {"current interval", 0, true, true, change_set::pair_class::unavailable}, variant {"FG on or unknown", 1, false, true, change_set::pair_class::late}, variant {"ring missing", 1, true, false, change_set::pair_class::unavailable}}) {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(true);
      sequence s(session, sb_sdr_page(sb_equipment));
      auto p = sb_sdr_menu_present();
      p.layer_presents_ago = v.presents_ago;
      p.fg_known_off = v.fg_known_off;
      p.retained_1 = v.retained_1;
      run(s, p, 10000, 12000);
      for (const auto &f : s.frames) {
        require(f.pairing.kind == v.kind && !f.pre_ui_offered && f.source == 4u && f.flat() && !f.decision.refined && !f.decision.reused, std::string("A pair that is not exact acted: ") + v.name);
      }
      for (const auto *shadow : shadows_of(s)) {
        require(shadow->pairing.kind == v.kind && !shadow->would_refine && shadow->would_source == 4u, std::string("The shadow would act on a pair that is not exact: ") + v.name);
      }
      require(!session.counters()[ui_counter::refined] && !session.counters().decided(ui_detection::source_pre_ui), std::string("A late pairing was counted as refined: ") + v.name);
      check_counters(s, std::string("SB SDR late pairing, ") + v.name);
    }
  }

  void fix3_hdr_layer_unchanged() {
    // (f) Stellar Blade in HDR: the offscreen target is a real UI layer with
    // coverage (accepted, 1.5%), never proven a pre-UI image (a layer with
    // coverage earns no proof), so the change set is never offered and the
    // layer decides as source 10 in both modes.
    const auto model = [](const gpu_inputs &in) {
      synthetic f;
      f.alpha(kind::ui_layer, 15).opaque(kind::ui_layer, 10).alpha(kind::current, 1000).opaque(kind::current, 1000);
      f.pre_ui_pixels(985, 900, 900, 15).layer_pair(15, 14, 240, 15, 18);
      return f.words(in);
    };
    std::array<std::vector<std::uint32_t>, 2> modes;
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      const auto signatures = sb_hdr_signatures();
      restore(session, signatures.of(kind::ui_layer).key(), 1);
      session.set_pin_only_ui(enabled);
      sequence s(session, model);
      present p;
      p.offered = candidate::layer | candidate::current;
      p.layer_flags = ui_detection::layer_detection_flags(false);
      p.signatures = signatures;
      run(s, p, 10000, 13000);
      for (const auto &f : s.frames) {
        require(f.source == ui_detection::source_layer && f.covered == 15 && !f.pre_ui_offered && !f.layer_proven && !f.decision.refined, "The HDR UI layer did not decide as 10, or its target was offered as a pre-UI image");
      }
      // A real UI layer can never be the provider, so the shadow never
      // measures it (no shadow bit, no tiles-pass work, no retention).
      for (const auto &f : s.frames) {
        require(!(f.rules & ui_detection::change_set::shadow), "The HDR UI layer pushed the shadow bit");
      }
      require(shadows_of(s).empty() && !s.samples.empty() && !session.counters()[ui_counter::change_set_samples], "The change-set shadow measured the HDR UI layer");
      require(!session.pre_ui_proven(signatures.of(kind::ui_layer)), "A layer with coverage was proven a pre-UI image");
      check_counters(s, std::string("SB HDR layer") + (enabled ? ", UIPinOnlyUI=1" : ""));
      modes[enabled] = sources_of(s);
    }
    require(modes[0] == modes[1], "The switch changed an HDR decision");
  }

  void fix3_e33_hudless_refines() {
    // (g) Every game: an Expedition 33-like menu over a visible scene whose
    // accepted Backbuffer alpha is opaque on every pixel (no shape), beside
    // an exact accepted HUD-less pair whose change set is selective (6%).
    // The refine rule lets the change set decide (5) with the switch on; off,
    // the whole-frame alpha keeps it flat (3). The pause menu (76% changed)
    // fails the selective test and stays flat in both modes.
    for (const std::uint32_t changed : {60u, 760u}) {
      for (const bool enabled : {false, true}) {
        alpha_auto_policy session;
        restore(session, key(kind::backbuffer) + "," + key(kind::hudless), 2);
        session.set_pin_only_ui(enabled);
        sequence s(session, [changed](const gpu_inputs &in) {
          synthetic f;
          f.alpha(kind::backbuffer, 1000).opaque(kind::backbuffer, 1000).change_set(changed);
          return f.words(in);
        });
        present p;
        p.offered = candidate::backbuffer | candidate::hudless | candidate::exact;
        run(s, p, 10000, 12000);
        const bool refines = enabled && changed == 60u;
        for (const auto &f : s.frames) {
          require(f.source == (refines ? 5u : 3u) && f.covered == (refines ? 60u : 1000u) && f.decision.refined == refines && !f.decision.inexact_difference && !(f.rules & ~ui_detection::rules::pin_only_ui) && f.pairing.kind == change_set::pair_class::none, "An exact selective HUD-less set did not refine the shapeless Backbuffer alpha exactly when enabled");
        }
        require(s.shadows.size() == s.samples.size() && shadows_of(s).empty(), "A stream without a layer measured a layer pair");
        require((session.counters()[ui_counter::refined] > 0) == refines, "The refined counter does not follow the refine");
        check_counters(s, std::string("E33-like HUD-less refine, ") + std::to_string(changed) + (enabled ? ", UIPinOnlyUI=1" : ""));
      }
    }
  }

  void fix3_w3_semi_transparent() {
    // (h) A Witcher 3-like sign wheel: an accepted layer over the whole frame
    // but opaque on only 35% of it (the semi-transparent backdrop), beside an
    // exact selective HUD-less set. It has shape, so the refine rule never
    // replaces it: source 10 in both modes.
    std::array<std::vector<std::uint32_t>, 2> modes;
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      restore(session, key(kind::ui_layer) + "," + key(kind::hudless), 2);
      session.set_pin_only_ui(enabled);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.alpha(kind::ui_layer, 1000).opaque(kind::ui_layer, 350).change_set(60).pre_ui_pixels(400, 900, 900, 600).layer_pair(600, 590, 0, 600, 600);
        return f.words(in);
      });
      present p;
      p.offered = candidate::layer | candidate::hudless | candidate::exact;
      p.layer_flags = ui_detection::layer_detection_flags(false);
      run(s, p, 10000, 12000);
      for (const auto &f : s.frames) {
        require(f.source == ui_detection::source_layer && f.covered == 1000 && !f.decision.shapeless && !f.decision.refined && !f.pre_ui_offered, "The semi-transparent whole-frame layer was refined");
      }
      check_counters(s, std::string("W3-like sign wheel") + (enabled ? ", UIPinOnlyUI=1" : ""));
      modes[enabled] = sources_of(s);
    }
    require(modes[0] == modes[1], "The switch changed the sign wheel's decision");
  }

  void fix3_missing_pair_t1() {
    // (i) T1 with the switch on: a real frame whose pair is not retained
    // (the pre-UI change set the previous real frame offered is missing:
    // T1's change-set gap in b2 word 5, never accepted_missing, so the frame
    // still adopts) under the shapeless current alpha has no decision of its
    // own (refine_missing): one such frame reuses the refined decision; of
    // three, the first reuses it and the next two apply their own flat
    // alpha, until the pair returns.
    alpha_auto_policy session;
    restore_sb_sdr(session);
    session.set_pin_only_ui(true);
    sequence s(session, sb_sdr_page(sb_equipment));
    auto p = sb_sdr_menu_present();
    std::uint64_t now = 10000;
    const auto step = [&](bool retained) -> const frame_result & {
      p.now_ms = now;
      now += 16;
      p.retained_1 = retained;
      return s.step(p);
    };
    for (int i = 0; i != 60; ++i) {
      require(step(true).source == ui_detection::source_pre_ui, "The Equipment page did not decide 12 before the missing pair");
    }
    const auto &once = step(false);
    require(once.decision.reused && once.source == ui_detection::source_pre_ui && once.covered == sb_equipment.exact.changed && !once.pre_ui_offered && (once.rules & ui_detection::change_set::gap) && once.hold.change_set_gap && !(once.gpu.per_frame & ui_detection::per_frame_accepted_missing) && once.hold.adopt && once.decision.own_source == 4u, "One missing pair did not reuse the refined decision through the change-set gap");
    require(step(true).source == ui_detection::source_pre_ui && !s.frames.back().decision.reused, "The pair's return did not decide 12 on its own");
    for (int i = 0; i != 3; ++i) {
      const auto &f = step(false);
      require(f.decision.reused == (i == 0) && f.source == (i == 0 ? ui_detection::source_pre_ui : 4u) && (i == 0 || f.flat()) && ((f.rules & ui_detection::change_set::gap) != 0) == (i == 0) && !(f.gpu.per_frame & ui_detection::per_frame_accepted_missing) && f.hold.adopt, "Three missing pairs did not reuse once and then show flat, or a frame did not adopt");
    }
    require(step(true).source == ui_detection::source_pre_ui, "The pair's return after three missing frames did not decide 12");
    check_counters(s, "SB SDR Equipment page, missing pair");
  }

  void fix3_h1_over_refined() {
    // (j) H1 runs after refine: the Equipment page whose presented frame
    // reads hidden while the pre-UI image reads visible enters a held hidden
    // verdict; the shapeless alpha's claim (a) and the proven layer's (d)
    // act, so the refined frame is flat as source 8 from the entry (its S1
    // winner stays the refined 12). Switch off, the shapeless alpha is flat
    // already and H1 leaves it (4).
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(enabled);
      sequence s(session, sb_sdr_page(sb_equipment, hidden_menu));
      run(s, sb_sdr_menu_present(), 10000, 12500);
      require(s.scene_entered == 1, "The hidden Equipment page did not enter a held hidden verdict");
      std::size_t entry = s.frames.size();
      for (std::size_t i = 0; i != s.frames.size() && entry == s.frames.size(); ++i) {
        entry = s.frames[i].scene_bits & ui_detection::per_frame_scene_hidden ? i : entry;
      }
      require(entry > 0 && entry < s.frames.size(), "No frame held the hidden verdict");
      for (std::size_t i = 0; i != s.frames.size(); ++i) {
        const auto &f = s.frames[i];
        if (!enabled) {
          require(f.source == 4u && f.flat() && !f.decision.h1, "H1 overrode the flat shapeless alpha");
        } else if (i < entry) {
          require(f.source == ui_detection::source_pre_ui && f.decision.refined, "The refined frame before the hold was not 12");
        } else {
          require(f.source == 8u && f.flat() && f.decision.h1 && f.decision.refined && f.decision.s1_source == ui_detection::source_pre_ui && (f.decision.claims & candidate::current) && (f.decision.claims & ui_detection::claim_pre_ui), "H1 did not show the refined frame flat as 8 under the held hidden verdict");
        }
      }
      check_counters(s, std::string("SB SDR Equipment page under H1") + (enabled ? ", UIPinOnlyUI=1" : ""));
    }
  }

  void fix3_declared_block() {
    // (k) An offered, accepted declared alpha keeps the pre-UI change set out
    // of S1 even while it is invalid itself: UIAlpha decides (1) while valid;
    // invalid, no source decides (the current alpha and the valid pre-UI set
    // are blocked) and T1 reuses its decision once, then shows no mask.
    alpha_auto_policy session;
    restore(session, key(kind::ui_alpha) + "," + sb_sdr_current() + "," + sb_sdr_proof(), 3);
    session.set_pin_only_ui(true);
    sequence s(session, sb_sdr_page(sb_equipment, [](synthetic &f, const gpu_inputs &in) {
      f.alpha(kind::ui_alpha, 20, in.now_ms >= 11000 ? 500 : 0);
    }));
    run(s, sb_sdr_present(candidate::ui_alpha | candidate::layer | candidate::current, 0), 10000, 12000);
    bool reused = false;
    for (const auto &f : s.frames) {
      require(f.pre_ui_offered && (f.decision.valid_bits & candidate::pre_ui) && f.source != ui_detection::source_pre_ui && f.source != 4u && !f.decision.refined, "The valid pre-UI change set decided beside an accepted declared alpha");
      if (f.now_ms < 11000) {
        require(f.source == 1u && f.covered == 20, "The valid UIAlpha did not decide");
      } else {
        require(f.decision.reused ? (!reused && f.source == 1u) : (!f.source && f.decision.none_reason == ui_no_mask::presented_blocked && f.decision.refused == ((f.gpu.accepted & candidate::current) ? candidate::current : candidate::pre_ui)), "The invalid UIAlpha did not reuse once and then show no mask, blocking the presented alpha: source " + std::to_string(f.source) + " reused " + std::to_string(f.decision.reused) + " reason " + std::to_string(f.decision.none_reason) + " refused " + hex(f.decision.refused) + " at " + std::to_string(f.now_ms));
        reused = reused || f.decision.reused;
      }
    }
    require(reused, "The invalid UIAlpha's first frame did not reuse its decision");
    check_counters(s, "Declared alpha blocks the pre-UI change set");
  }

  void fix3_switch_toggled() {
    // (l) UIPinOnlyUI edited mid-session (the panel): the decision
    // follows from the next frame, and the ledger never moves.
    alpha_auto_policy session;
    restore_sb_sdr(session);
    const auto stored = session.stored();
    sequence s(session, sb_sdr_page(sb_equipment));
    const auto p = sb_sdr_menu_present();
    run(s, p, 10000, 11000);
    const auto on = s.frames.size();
    session.set_pin_only_ui(true);
    run(s, p, 11000, 12000);
    const auto off = s.frames.size();
    session.set_pin_only_ui(false);
    run(s, p, 12000, 13000);
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      const bool enabled = i >= on && i < off;
      require(s.frames[i].source == (enabled ? ui_detection::source_pre_ui : 4u) && s.frames[i].decision.refined == enabled && !s.frames[i].decision.reused, "A decision did not follow the switch at once, at frame " + std::to_string(i));
    }
    require(session.stored() == stored && s.trust.empty() && session.pin_only_ui() == false, "Editing the switch moved the ledger");
    check_counters(s, "SB SDR switch toggled");
  }

  void fix3_shadow_inert() {
    // (m) The shadow is diagnostics only: the same Stellar Blade SDR gameplay
    // stream (the current alpha selective, earned 2 s in; the restored proof
    // provisional while the layer mismatches on testable samples, so it
    // lapses after 60 s of testable time) with texels 13-15 as the tiles pass
    // writes them or zeroed gives the same decisions, ledger transitions,
    // stored keys and counters, the change_set group apart.
    const auto model = sb_sdr_page({{50, 48, 230}, {150, 140, 120}, 160, 900}, [](synthetic &f, const gpu_inputs &) {
      f.alpha(kind::current, 50).opaque(kind::current, 40).scene(.4f, .4f);
    });
    std::array<std::vector<frame_result>, 2> frames;
    std::array<std::vector<sequence::transition>, 2> trust;
    std::array<std::string, 2> stored;
    std::array<ui_counters, 2> totals;
    for (const bool zeroed : {false, true}) {
      alpha_auto_policy session;
      restore(session, sb_sdr_proof(), 1);
      sequence s(session, model);
      if (zeroed) {
        s.sample_edit = [](texels &t, std::uint64_t, bool) {
          std::fill(t.begin() + change_set_texel_begin, t.end(), 0u);
        };
      }
      run(s, sb_sdr_present(candidate::layer | candidate::current, 0), 10000, 75000, 33);
      check_counters(s, std::string("SB SDR shadow inert") + (zeroed ? ", zeroed" : ""));
      frames[zeroed] = s.frames;
      trust[zeroed] = s.trust;
      stored[zeroed] = session.stored();
      totals[zeroed] = session.counters();
    }
    require(frames[0].size() == frames[1].size(), "The streams differ in length");
    for (std::size_t i = 0; i != frames[0].size(); ++i) {
      const auto &a = frames[0][i], &b = frames[1][i];
      require(a.source == b.source && a.covered == b.covered && a.decision.reused == b.decision.reused && a.scene_bits == b.scene_bits && a.layer_proven == b.layer_proven && a.rules == b.rules && a.submitted == b.submitted, "The shadow texels changed a decision at frame " + std::to_string(i));
    }
    require(trust[0].size() == trust[1].size() && stored[0] == stored[1], "The shadow texels changed the ledger");
    for (std::size_t i = 0; i != trust[0].size(); ++i) {
      require(trust[0][i].tick == trust[1][i].tick && trust[0][i].accepted == trust[1][i].accepted, "The shadow texels moved a ledger clock");
    }
    // The ledger's clocks ran: the current alpha earned, the proof lapsed.
    const auto earned = sb_sdr_current();
    require(trust[0].size() == 2 && trust[0][0].accepted == stored_of({sb_sdr_signatures().of(kind::current), ui_selection::pre_ui_key(sb_sdr_signatures().of(kind::ui_layer))}) && trust[0][1].accepted == earned && trust[0][1].tick >= 70000, "The stream did not earn the current alpha and then lapse the proof");
    for (std::size_t i = 0; i != ui_counter::count; ++i) {
      if (i < ui_counter::change_set_samples || i > ui_counter::change_set_pair_contradicted) {
        require(totals[0][i] == totals[1][i], "The shadow texels changed counter " + std::to_string(i));
      }
    }
    require(totals[0][ui_counter::change_set_samples] == totals[1][ui_counter::change_set_samples] && totals[0][ui_counter::change_set_valid] == totals[0][ui_counter::change_set_samples] && !totals[1][ui_counter::change_set_valid], "The change_set group did not show the populated and zeroed shadows");
  }

  void fix3_fg_exactness() {
    // (n) Exactness is a property of the paired Presents (review fix): the
    // run of real Presents with frame generation known off must cover the
    // Present the copy shows. With the switch on, over the Equipment page:
    // (1) FG on or unknown for a while, then known off: the first known-off
    // Present still pairs late (the Present it shows was not known off), the
    // second pairs retained and refines; (2) a generated Present inside the
    // run (one an unobserved mode change let through) ends it: the next real
    // frame pairs late and decides nothing from the pair (T1's gap reuses
    // the refined decision once), the one after pairs retained again; (3) a
    // game whose FG mode is never known (no Streamline FG observer, or FG
    // another SDK generates) never pairs retained.
    alpha_auto_policy session;
    restore_sb_sdr(session);
    session.set_pin_only_ui(true);
    sequence s(session, sb_sdr_page(sb_equipment));
    s.fg_off_presents = 0;
    auto p = sb_sdr_menu_present();
    std::uint64_t now = 10000;
    const auto step = [&](bool known_off, bool generated = false) -> const frame_result & {
      p.now_ms = now;
      now += 16;
      p.fg_known_off = known_off;
      p.hold_previous = generated;
      return s.step(p);
    };
    for (int i = 0; i != 20; ++i) {
      const auto &f = step(false);
      require(f.pairing.kind == change_set::pair_class::late && !f.pre_ui_offered && f.source == 4u && f.flat() && !f.fg_off_presents, "A pair under FG on or unknown was not late, or acted");
    }
    const auto &first = step(true);
    require(first.fg_off_presents == 1 && first.pairing.kind == change_set::pair_class::late && !first.pre_ui_offered && first.source == 4u && !first.decision.reused, "The first known-off Present paired the Present before it, which was not known off");
    const auto &second = step(true);
    require(second.fg_off_presents == 2 && second.pairing == change_set::layer_pairing {change_set::pair_class::retained, 1u} && second.pre_ui_offered && second.source == ui_detection::source_pre_ui && second.decision.refined, "The second known-off Present did not pair retained and refine");
    for (int i = 0; i != 10; ++i) {
      require(step(true).source == ui_detection::source_pre_ui, "The known-off run did not keep refining");
    }
    const auto &generated = step(true, true);
    require(generated.held && !generated.detected && !generated.fg_off_presents, "A generated Present detected, or did not end the run");
    const auto &after = step(true);
    require(after.fg_off_presents == 1 && after.pairing.kind == change_set::pair_class::late && !after.pre_ui_offered && after.decision.own_source == 4u && after.decision.reused && after.source == ui_detection::source_pre_ui && (after.rules & ui_detection::change_set::gap), "The real frame after a generated Present paired retained, or did not reuse once through the gap");
    const auto &again = step(true);
    require(again.pairing.kind == change_set::pair_class::retained && again.source == ui_detection::source_pre_ui && !again.decision.reused, "The run after a generated Present did not pair retained again");
    check_counters(s, "SB SDR FG exactness");

    alpha_auto_policy unknown;
    restore_sb_sdr(unknown);
    unknown.set_pin_only_ui(true);
    sequence never(unknown, sb_sdr_page(sb_equipment));
    auto q = sb_sdr_menu_present();
    q.fg_known_off = false;
    run(never, q, 10000, 12000);
    for (const auto &f : never.frames) {
      require(f.pairing.kind == change_set::pair_class::late && !f.pre_ui_offered && f.source == 4u && f.flat() && !f.decision.refined, "A game whose FG mode is never known paired retained or refined");
    }
    require(!unknown.counters()[ui_counter::refined] && unknown.counters()[ui_counter::change_set_late] == unknown.counters()[ui_counter::change_set_samples], "An unknown FG mode counted a refined frame or a pairing other than late");
    check_counters(never, "SB SDR FG unknown");
  }

  void fix3_gap_keeps_adoption() {
    // (o) T1 adoption with the pre-UI change set adopted (review fix): the
    // menu decides 12 with the set offered; then FG-on gameplay offers a
    // full-page opaque UI colour tag (accepted) and never the set (late).
    // Only the first gameplay frame carries the change-set gap and reuses
    // the refined decision once; every later frame decides on its own (2,
    // flat) and adopts its inputs, so the status key follows the tag.
    alpha_auto_policy session;
    restore(session, sb_sdr_signatures().of(kind::ui_color).key() + "," + sb_sdr_current() + "," + sb_sdr_proof(), 3);
    session.set_pin_only_ui(true);
    sequence s(session, sb_sdr_page(sb_equipment, [](synthetic &f, const gpu_inputs &in) {
      if (in.bits & candidate::ui_color) {
        f.alpha(kind::ui_color, 1000).opaque(kind::ui_color, 1000);
      }
    }));
    run(s, sb_sdr_menu_present(), 10000, 11000);
    require(s.frames.back().source == ui_detection::source_pre_ui && (s.temporal.bits & candidate::pre_ui), "The menu did not decide and adopt the pre-UI change set");
    const auto menu = s.frames.size();
    // The same identity scope (viewport 0), so T1's chain continues.
    auto p = sb_sdr_present(candidate::ui_color | candidate::layer | candidate::current, 0);
    p.fg_known_off = false;
    run(s, p, 11000, 12000);
    for (std::size_t i = menu; i != s.frames.size(); ++i) {
      const auto &f = s.frames[i];
      const bool gap = i == menu;
      require(f.detected && f.pairing.kind == change_set::pair_class::late && !f.pre_ui_offered && f.hold.change_set_gap == gap && ((f.rules & ui_detection::change_set::gap) != 0) == gap && !(f.gpu.per_frame & ui_detection::per_frame_accepted_missing) && f.hold.adopt, "A gameplay frame carried the gap after the first, flagged accepted_missing or did not adopt, at frame " + std::to_string(i));
      if (gap) {
        require(f.decision.reused && f.source == ui_detection::source_pre_ui && f.decision.own_source == 2u, "The first gameplay frame did not reuse the refined decision once");
      } else {
        require(!f.decision.reused && f.source == 2u && f.flat(), "A later gameplay frame reused a decision or did not decide the full-page tag, at frame " + std::to_string(i));
      }
    }
    require(s.temporal.bits == p.offered && s.temporal.status_key() == candidate::ui_color, "The adopted inputs stopped updating after the gap");
    check_counters(s, "SB SDR gap keeps adoption");
  }

  // ---------------------------------------------------------------- fix 4: pin only UI (rule P2)

  // A Stellar Blade HDR-like stream: the accepted offscreen UI layer (1.5%
  // covered) decides 10; its darkening words say 4 of its pixels are a pure
  // darkening the rule unpins and 6 a darkening sharp structure keeps.
  // invalid_from..invalid_to makes the layer invalid (T1's grace reuses the
  // previous decision once).
  gpu_model sb_hdr_layer_model(std::uint64_t invalid_from = 0, std::uint64_t invalid_to = 0) {
    return [invalid_from, invalid_to](const gpu_inputs &in) {
      synthetic f;
      const bool invalid = in.now_ms >= invalid_from && in.now_ms < invalid_to;
      f.alpha(kind::ui_layer, 15, invalid ? 500 : 0).opaque(kind::ui_layer, 10).alpha(kind::current, 1000).opaque(kind::current, 1000);
      f.pre_ui_pixels(985, 900, 900, 15).darkening(4, 6);
      return f.words(in);
    };
  }

  present sb_hdr_layer_present() {
    present p;
    p.offered = candidate::layer | candidate::current;
    p.layer_flags = ui_detection::layer_detection_flags(false);
    p.signatures = sb_hdr_signatures();
    return p;
  }

  // The darkening lines a stream logged, by their switch flag.
  std::size_t darkening_lines_with(const sequence &s, const std::string &text) {
    return std::size_t(std::count_if(s.darkening_lines.begin(), s.darkening_lines.end(), [&](const std::string &line) {
      return line.find(text) != std::string::npos;
    }));
  }

  void fix4_shadow_changes_nothing() {
    // (a) The default shadow (UIPinOnlyUI=0) only measures: the darkening
    // passes run on Auto sample frames alone, every sample of the eligible
    // layer logs one line with UIPinOnlyUI=0 and is counted, and no mask is
    // darkened. The same stream with words 62-63 zeroed gives the same
    // decisions, masks, rules, ledger and counters, the darkening group
    // apart (and every detection of every stream decides the same without
    // the passes' bit: sequence::detect).
    std::array<std::vector<frame_result>, 2> frames;
    std::array<ui_counters, 2> totals;
    std::array<std::string, 2> stored;
    for (const bool zeroed : {false, true}) {
      alpha_auto_policy session;
      restore(session, sb_hdr_signatures().of(kind::ui_layer).key(), 1);
      sequence s(session, sb_hdr_layer_model());
      if (zeroed) {
        s.sample_edit = [](texels &t, std::uint64_t, bool) {
          t[word::dk_unpinned] = t[word::dk_kept] = 0u;
        };
      }
      run(s, sb_hdr_layer_present(), 10000, 13000);
      for (const auto &f : s.frames) {
        require(f.source == ui_detection::source_layer && f.covered == 15 && !f.darkened && f.darkening == f.submitted && !(f.rules & ui_detection::rules::pin_only_ui) && !(f.rules & ui_detection::rules::darkening_measured), "The shadow darkened a mask, ran the darkening passes on a frame that was not a sample, or pushed the switch's bit");
      }
      // Every sample is measured and counted; its unchanging line logs at
      // the first sample and then once a second (3 s: at most 4 lines).
      require(s.samples.size() >= 20 && s.darkening_samples == s.samples.size() && !s.darkening_lines.empty() && s.darkening_lines.size() <= 4u && darkening_lines_with(s, "UIPinOnlyUI=0") == s.darkening_lines.size(), "The shadow's darkening lines were not throttled to once a second, or a sample was not measured");
      require(s.darkening_lines[0] == (zeroed ? "Sunshine UI darkening: source=10 UIPinOnlyUI=0 covered=15 unpinned=0 kept=0 colourless=0" : "Sunshine UI darkening: source=10 UIPinOnlyUI=0 covered=15 unpinned=4 kept=6 colourless=0"), "The darkening line differs: " + s.darkening_lines[0]);
      for (const auto &sample : s.samples) {
        require(sample.darkening.measured && !sample.darkening.applied && sample.darkening.unpinned == (zeroed ? 0u : 4u), "A shadow sample's darkening status is wrong");
      }
      const auto c = session.counters();
      require(c[ui_counter::darkening_samples] && c[ui_counter::darkening_unpinned_samples] == (zeroed ? 0u : c[ui_counter::darkening_samples]) && c[ui_counter::darkening_unpinned_px] == (zeroed ? 0u : 4u * c[ui_counter::darkening_samples]) && c[ui_counter::darkening_kept_px] == (zeroed ? 0u : 6u * c[ui_counter::darkening_samples]), "The darkening counters do not sum the committed samples");
      check_counters(s, std::string("SB HDR layer, darkening shadow") + (zeroed ? ", zeroed" : ""));
      frames[zeroed] = s.frames;
      totals[zeroed] = c;
      stored[zeroed] = session.stored();
    }
    require(frames[0].size() == frames[1].size() && stored[0] == stored[1], "The darkening words changed the stream or the ledger");
    for (std::size_t i = 0; i != frames[0].size(); ++i) {
      const auto &a = frames[0][i], &b = frames[1][i];
      require(a.source == b.source && a.covered == b.covered && a.darkened == b.darkened && a.rules == b.rules && a.darkening == b.darkening && a.decision.reused == b.decision.reused && a.submitted == b.submitted, "The darkening words changed a decision or mask at frame " + std::to_string(i));
    }
    for (std::size_t i = 0; i != ui_counter::count; ++i) {
      if (i < ui_counter::darkening_samples || i > ui_counter::darkening_kept_px) {
        require(totals[0][i] == totals[1][i], "The darkening words changed counter " + std::to_string(i));
      }
    }
  }

  void fix4_switch_unpins_only_eligible_sources() {
    // (b) UIPinOnlyUI=1: the darkening passes run on every Auto detection
    // frame; the mask is darkened exactly when the frame's own decision is an
    // eligible source (2 the UI color tag, 10 the layer, 5 a HUD-less set,
    // 12 the pre-UI set), never for an alpha-only source (1 UIAlpha, 3
    // Backbuffer, 4 current). Every decision equals the shadow's, apart from
    // refine (its own groups).
    struct variant {
      const char *name;
      std::string stored;
      std::size_t restored;
      std::uint32_t offered;
      std::function<void(synthetic &)> counts;
      std::uint32_t source;
    };
    const variant variants[] {
      {"UI color tag", key(kind::ui_color), 1, candidate::ui_color | candidate::current, [](synthetic &f) { f.alpha(kind::ui_color, 30); }, 2u},
      {"offscreen UI layer", key(kind::ui_layer), 1, candidate::layer | candidate::current, [](synthetic &f) { f.alpha(kind::ui_layer, 15); }, ui_detection::source_layer},
      {"HUD-less set", key(kind::hudless), 1, candidate::hudless | candidate::exact | candidate::current, [](synthetic &f) { f.change_set(60); }, 5u},
      {"UIAlpha", key(kind::ui_alpha), 1, candidate::ui_alpha | candidate::current, [](synthetic &f) { f.alpha(kind::ui_alpha, 20); }, 1u},
      {"Backbuffer alpha", key(kind::backbuffer), 1, candidate::backbuffer, [](synthetic &f) { f.alpha(kind::backbuffer, 40); }, 3u},
      {"current alpha", key(kind::current), 1, candidate::current, [](synthetic &f) { f.alpha(kind::current, 40); }, 4u},
    };
    for (const auto &v : variants) {
      std::array<std::vector<std::uint32_t>, 2> modes;
      for (const bool enabled : {false, true}) {
        alpha_auto_policy session;
        restore(session, v.stored, v.restored);
        session.set_pin_only_ui(enabled);
        const auto counts = v.counts;
        sequence s(session, [counts](const gpu_inputs &in) {
          synthetic f;
          counts(f);
          f.darkening(3, 2);
          return f.words(in);
        });
        present p;
        p.offered = v.offered;
        p.layer_flags = (v.offered & candidate::layer) ? ui_detection::layer_detection_flags(false) : 0u;
        run(s, p, 10000, 12000);
        const bool eligible = ui_detection::darkening::eligible(v.source);
        for (const auto &f : s.frames) {
          require(f.source == v.source && f.darkening == (enabled || f.submitted) && f.darkened == (enabled && eligible) && ((f.rules & ui_detection::rules::pin_only_ui) != 0) == enabled, std::string(v.name) + ": the darkening passes or the darkened mask do not follow the switch and the source's eligibility");
        }
        require(s.darkening_samples == (eligible ? s.samples.size() : 0u) && (s.darkening_lines.empty() != eligible) && s.darkening_lines.size() <= 3u && darkening_lines_with(s, enabled ? "UIPinOnlyUI=1" : "UIPinOnlyUI=0") == s.darkening_lines.size(), std::string(v.name) + ": the darkening samples or their throttled lines do not follow eligibility");
        if (eligible) {
          require(starts_with(s.darkening_lines[0], "Sunshine UI darkening: source=" + std::to_string(v.source) + " UIPinOnlyUI=" + (enabled ? "1" : "0") + " covered="), std::string(v.name) + ": the darkening line names another source or switch: " + s.darkening_lines[0]);
          for (const auto &sample : s.samples) {
            require(sample.darkening.measured && sample.darkening.applied == enabled && sample.darkening.unpinned == 3u && sample.darkening.kept == 2u, std::string(v.name) + ": a sample's darkening is wrong");
          }
        }
        require((session.counters()[ui_counter::darkening_samples] > 0) == eligible, std::string(v.name) + ": the darkening samples were not counted exactly for an eligible source");
        check_counters(s, std::string("Pin only UI, ") + v.name + (enabled ? ", UIPinOnlyUI=1" : ""));
        modes[enabled] = sources_of(s);
      }
      require(modes[0] == modes[1], std::string(v.name) + ": the switch changed a decision");
    }
  }

  void fix4_t1_reuse_keeps_the_darkened_mask() {
    // (c) T1 with the switch on: a real frame whose accepted layer turns
    // invalid reuses the previous real frame's decision and mask once (the
    // mask pass returns early), so it stays darkened, and a sample of a
    // reused frame logs no darkening (texel 9's reused bit stops the passes);
    // the next frame has no mask. Generated Presents hold the darkened mask
    // of the real frame they show.
    alpha_auto_policy session;
    restore(session, sb_hdr_signatures().of(kind::ui_layer).key(), 1);
    session.set_pin_only_ui(true);
    sequence s(session, sb_hdr_layer_model(11000, 11040));
    auto p = sb_hdr_layer_present();
    run(s, p, 10000, 11100);
    std::size_t reused = 0, lost = 0;
    for (const auto &f : s.frames) {
      if (f.decision.reused) {
        ++reused;
        require(f.source == ui_detection::source_layer && f.darkened && f.darkening, "A frame T1 reused lost the darkened mask");
      } else if (f.now_ms >= 11000 && f.now_ms < 11040) {
        ++lost;
        require(!f.source && !f.darkened, "An invalid frame after the reuse kept a mask");
      } else {
        require(f.source == ui_detection::source_layer && f.darkened, "A valid layer frame was not darkened");
      }
    }
    require(reused == 1 && lost >= 1, "The invalid layer did not reuse its decision exactly once");
    for (const auto &sample : s.samples) {
      require(sample.darkening.measured == (sample.source_kind == ui_detection::source_layer && !sample.evidence.reused), "A sample of a reused or undecided frame measured darkening");
    }
    // A generated Present shows the real frame's darkened mask.
    p.hold_previous = true;
    p.now_ms = 11104;
    const auto &held = s.step(p);
    require(held.held && held.source == ui_detection::source_layer && held.darkened && !held.darkening, "A generated Present did not hold the darkened mask, or ran the passes");
    check_counters(s, "Pin only UI, T1 reuse");
  }

  void fix4_full_frame_sources_untouched() {
    // (d) Whole-frame decisions are never darkened: H1's flat 8 over a
    // refined Equipment page (its S1 winner 12 is eligible), H2's still
    // screen 11 and the exact full HUD-less set 6, each with the switch on.
    {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(true);
      sequence s(session, sb_sdr_page(sb_equipment, [](synthetic &f, const gpu_inputs &in) {
        hidden_menu(f, in);
        f.darkening(20, 10);
      }));
      run(s, sb_sdr_menu_present(), 10000, 12500);
      std::size_t flat = 0, refined = 0;
      for (const auto &f : s.frames) {
        require(f.darkening && f.darkened == (f.source == ui_detection::source_pre_ui), "H1 left an eligible frame undarkened or darkened its flat frame");
        flat += f.source == 8u ? 1 : 0;
        refined += f.source == ui_detection::source_pre_ui ? 1 : 0;
      }
      require(flat > 50 && refined > 0 && s.frames.back().source == 8u && s.frames.back().flat(), "H1 did not show the refined page flat as 8");
      require(darkening_lines_with(s, "source=12 ") == s.darkening_lines.size() && darkening_lines_with(s, "source=8 ") == 0u, "A flat H1 sample logged darkening");
      check_counters(s, "Pin only UI, H1 over a refined page");
    }
    {
      const auto sb_d = [](std::uint64_t now) {
        return -.014f - .054f * float((now / 100) % 10) / 9.f;
      };
      alpha_auto_policy session;
      session.set_still_flatten(true);
      session.set_pin_only_ui(true);
      sequence s(session, still_screen_model(sb_d));
      run_still(s, loading_screen_present(), 10000, 14000);
      std::size_t still = 0;
      for (const auto &f : s.frames) {
        require(!f.darkened, "A still screen was darkened");
        still += f.source == ui_detection::source_still ? 1 : 0;
      }
      require(still > 50 && s.darkening_lines.empty() && !session.counters()[ui_counter::darkening_samples], "H2 did not flatten, or its samples logged darkening");
      check_counters(s, "Pin only UI, H2 still screen");
    }
    {
      alpha_auto_policy session;
      restore(session, key(kind::hudless), 1);
      session.set_pin_only_ui(true);
      sequence s(session, [](const gpu_inputs &in) {
        synthetic f;
        f.change_set(990).darkening(5, 5);
        return f.words(in);
      });
      present p;
      p.offered = candidate::hudless | candidate::exact | candidate::current;
      run(s, p, 10000, 11500);
      for (const auto &f : s.frames) {
        require(f.source == 6u && f.flat() && f.darkening && !f.darkened, "The full change set was not flat as 6, or was darkened");
      }
      require(s.darkening_lines.empty(), "The full change set logged darkening");
      check_counters(s, "Pin only UI, full change set");
    }
  }

  void fix4_refine_to_12_with_darkening() {
    // (e) The Stellar Blade SDR Equipment page (dump 033: the bottom dim
    // band about 40% of the changed pixels): with the switch on, refine
    // decides the pre-UI set 12 and the darkening unpins its band on every
    // frame; off, the page is flat by the shapeless current alpha (4, not
    // eligible), and its samples log no darkening although the passes run.
    for (const bool enabled : {false, true}) {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(enabled);
      sequence s(session, sb_sdr_page(sb_equipment, [](synthetic &f, const gpu_inputs &) {
        f.darkening(20, 4);
      }));
      run(s, sb_sdr_menu_present(), 10000, 12000);
      for (const auto &f : s.frames) {
        require(f.source == (enabled ? ui_detection::source_pre_ui : 4u) && f.darkened == enabled && f.darkening == (enabled || f.submitted) && f.decision.refined == enabled, "The Equipment page did not refine and darken exactly with the switch");
      }
      require(s.darkening_samples == (enabled ? s.samples.size() : 0u) && s.darkening_lines.empty() != enabled, "The Equipment page's darkening samples or lines do not follow the switch");
      if (enabled) {
        require(s.darkening_lines[0] == "Sunshine UI darkening: source=12 UIPinOnlyUI=1 covered=52 unpinned=20 kept=4 colourless=0", "The refined page's darkening line differs: " + s.darkening_lines[0]);
        require(!s.shadow_lines.empty() && s.shadow_lines[0].find(" UIPinOnlyUI=1 ") != std::string::npos, "The change-set line does not name the shared switch");
      }
      check_counters(s, std::string("Pin only UI, SB SDR Equipment") + (enabled ? ", UIPinOnlyUI=1" : ""));
    }
  }

  void fix4_one_switch() {
    // (f) One switch, UIPinOnlyUI (fix 3's UIPinChangedPixels renamed): an
    // edit mid-session turns refine and the darkening on and off together
    // from the next frame, the change-set and darkening lines carry the same
    // flag, and the ledger never moves.
    alpha_auto_policy session;
    restore_sb_sdr(session);
    const auto stored = session.stored();
    sequence s(session, sb_sdr_page(sb_equipment, [](synthetic &f, const gpu_inputs &) {
      f.darkening(20, 4);
    }));
    const auto p = sb_sdr_menu_present();
    run(s, p, 10000, 11000);
    const auto on = s.frames.size();
    session.set_pin_only_ui(true);
    run(s, p, 11000, 12000);
    const auto off = s.frames.size();
    session.set_pin_only_ui(false);
    run(s, p, 12000, 13000);
    for (std::size_t i = 0; i != s.frames.size(); ++i) {
      const bool enabled = i >= on && i < off;
      const auto &f = s.frames[i];
      require(f.source == (enabled ? ui_detection::source_pre_ui : 4u) && f.decision.refined == enabled && f.darkened == enabled && ((f.rules & ui_detection::rules::pin_only_ui) != 0) == enabled, "Refine and the darkening did not follow the one switch at frame " + std::to_string(i));
    }
    for (const auto &line : s.darkening_lines) {
      require(starts_with(line, "Sunshine UI darkening: source=12 UIPinOnlyUI=1 "), "A darkening line outside the switched-on span: " + line);
    }
    for (const auto &line : s.shadow_lines) {
      require(line.find("UIPinChangedPixels") == std::string::npos && line.find(" UIPinOnlyUI=") != std::string::npos, "A change-set line names the old key: " + line);
    }
    require(!s.darkening_lines.empty() && session.stored() == stored && s.trust.empty() && !session.pin_only_ui(), "Editing the switch moved the ledger or logged no darkening");
    check_counters(s, "Pin only UI, switch toggled");
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
          evidence.presented_lit = field_number(field_text(pre_ui_pixels, "presented_lit"));
          evidence.presented_lit_differs = field_number(field_text(pre_ui_pixels, "presented_lit_differs"));
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
  // S3, the GPU's identity verdicts (game3d_native.hlsl SunshineIdentityWords,
  // ui_selection::verify_identity): every detection carries the stamps of
  // its candidate copies and the provider's proposals. By default no
  // verdict changes a decision and the identity words count each offered
  // pair; under the test-only override the renderer pushes the gates.
  void s3_gpu_identity() {
    namespace idn = ui_ticket::identity_counter;
    using ui_detection::identity::token_batch;
    // (a) The SB SDR Equipment page with UIPinOnlyUI=1: the layer copy's
    // stamp read equals the proposed label (its retained Present's), except
    // on late frames, where its list executed one Present late (read + 1).
    const auto equipment = [](bool authoritative, const std::function<bool(int)> &late, int frames) {
      auto session = std::make_unique<alpha_auto_policy>();
      restore_sb_sdr(*session);
      session->set_pin_only_ui(true);
      auto s = std::make_unique<sequence>(*session, sb_sdr_page(sb_equipment));
      s->temporal.identity_override = authoritative;
      auto p = sb_sdr_menu_present();
      for (int i = 0; i != frames; ++i) {
        const std::uint32_t label = 100u + std::uint32_t(i);
        p.now_ms = 10000u + std::uint64_t(i) * 16u;
        p.identity = {late(i) ? label + 1u : label, 0u, 0u, label, 0u, 0u};
        s->step(p);
      }
      return std::make_pair(std::move(session), std::move(s));
    };
    const auto late_frames = [](int i) {
      return i == 60 || (i >= 80 && i < 83);
    };
    {
      const auto [session, reference] = equipment(false, [](int) { return false; }, 120);
      const auto [late_session, today] = equipment(false, late_frames, 120);
      for (std::size_t i = 0; i != today->frames.size(); ++i) {
        const auto &f = today->frames[i];
        require(f.source == reference->frames[i].source && f.covered == reference->frames[i].covered && f.decision.reused == reference->frames[i].decision.reused && f.source == ui_detection::source_pre_ui, "By default a late layer copy's verdict changed the Equipment page's decision");
        require(f.decision.identity.layer == (late_frames(int(i)) ? ui_detection::identity::mismatch : ui_detection::identity::exact) && f.decision.identity.layer_space == ui_detection::identity::space_present && f.decision.identity.layer_delta == (late_frames(int(i)) ? 1 : 0) && !(f.identity.bits & (ui_detection::identity::gate_layer | ui_detection::identity::gate_hudless)), "A late layer copy was not a present-space mismatch, or the shadow pushed a gate");
      }
      // The committed identity words count every offered layer pair.
      std::uint64_t exact = 0, mismatch = 0;
      for (std::size_t i = 0; i != today->committed_frames; ++i) {
        (late_frames(int(i)) ? mismatch : exact) += 1;
      }
      const auto &k = today->identity_gpu;
      require(today->commits && k[idn::gpu_exact] == exact && k[idn::gpu_mismatch] == mismatch && !k[idn::gpu_unstamped] && !k[idn::gpu_unproposed] && !k[idn::gpu_token_exact], "The committed identity words do not count the frames through the last commit");
      check_counters(*today, "S3 shadow: SB SDR Equipment page, late copies");
      std::printf("SHADOW S3 E2: SB SDR Equipment page, UIPinOnlyUI=1, 4 of 120 copies late: decisions unchanged; committed %s\n", ui_ticket::format_identity_counters(k).c_str());
      // Under the override the gate refuses each late pair: refine finds no
      // set, so the first late frame reuses the refined 12 once (T1), and of
      // three in a row the next two show the flat alpha, until the exact pair
      // returns.
      const auto [gated_session, gated] = equipment(true, late_frames, 120);
      for (std::size_t i = 0; i != gated->frames.size(); ++i) {
        const auto &f = gated->frames[i];
        require((f.identity.bits & ui_detection::identity::gate_layer) && f.pre_ui_offered, "The override did not push the layer gate");
        if (i == 60 || i == 80) {
          require(f.decision.reused && f.source == ui_detection::source_pre_ui && f.decision.own_source == 4u, "Under S3 the first late copy did not reuse the refined 12 once");
        } else if (i == 81 || i == 82) {
          require(!f.decision.reused && f.source == 4u && f.flat(), "Under S3 a late copy after a reuse did not show the flat alpha");
        } else {
          require(f.source == ui_detection::source_pre_ui && !f.decision.reused, "Under S3 an exact layer copy did not decide 12");
        }
      }
      check_counters(*gated, "S3 enabled: SB SDR Equipment page, late copies");
    }
    // (b) Stellar Blade SDR with frame generation suspended (requested, not
    // known off): no Present-counted pairing. The game tags its Backbuffer,
    // and the layer copy after a tagged frame reads that token through C_T.
    // By default nothing pairs and the frame stays flat (4); under the
    // override the layer pairs in token space (offset 3, the same-token
    // Backbuffer) and decides 12, a token mismatch is refused, and without
    // Backbuffer tags there is no pair.
    {
      // The page's counts at the Backbuffer pair: the counted pair's.
      const auto page = sb_sdr_page(sb_equipment);
      const gpu_model token_page = [page](const gpu_inputs &in) {
        auto at = in;
        if (ui_detection::change_set::pair_offset(at.rules) == ui_detection::change_set::pair_backbuffer) {
          at.rules = (at.rules & ~ui_detection::change_set::pair_mask) | (1u << ui_detection::change_set::pair_shift);
        }
        return decision_words(page(at), in);
      };
      for (const bool authoritative : {false, true}) {
        for (const bool tagged : {true, false}) {
          alpha_auto_policy session;
          restore_sb_sdr(session);
          session.set_pin_only_ui(true);
          sequence s(session, token_page);
          s.temporal.identity_override = authoritative;
          auto p = sb_sdr_present(candidate::layer | candidate::current | (tagged ? candidate::backbuffer : 0u), 0);
          p.fg_known_off = false;
          for (int i = 0; i != 60; ++i) {
            const std::uint32_t token = 500u + std::uint32_t(i);
            p.now_ms = 10000u + std::uint64_t(i) * 16u;
            p.identity = {0u, i == 40 ? token - 1u : token, 0u, 0u, tagged ? token : 0u, 0u};
            const auto &f = s.step(p);
            const bool token_pair = authoritative && tagged;
            require(f.pairing.kind == (token_pair ? change_set::pair_class::token : change_set::pair_class::late) && f.pre_ui_offered == token_pair, "The token-space pairing did not follow the override and the Backbuffer tag");
            if (!token_pair) {
              require(f.source == 4u && f.flat() && f.decision.identity.layer == (tagged ? (i == 40 ? ui_detection::identity::mismatch : ui_detection::identity::exact) : ui_detection::identity::unproposed), "Without the token pair the FG-suspended page was not flat, or its verdict was wrong");
            } else if (i == 40) {
              require(f.decision.reused && f.source == ui_detection::source_pre_ui && f.decision.identity.layer == ui_detection::identity::mismatch && f.decision.identity.layer_delta == -1, "Under S3 a token mismatch was not refused with one T1 reuse");
            } else {
              require(f.source == ui_detection::source_pre_ui && f.decision.refined && f.decision.identity.layer_space == ui_detection::identity::space_token && ui_detection::change_set::pair_offset(f.rules) == ui_detection::change_set::pair_backbuffer, "Under S3 the token-space pair did not decide 12 against the Backbuffer");
            }
          }
          check_counters(s, "S3 SB SDR FG suspended, token pair");
          if (authoritative && tagged) {
            require(s.identity_gpu[idn::gpu_token_exact] && s.identity_gpu[idn::gpu_token_exact] == s.identity_gpu[idn::gpu_exact], "Token-space exact pairs were not counted as token_exact");
          }
        }
      }
    }
    // (c) A scope change recreates the stamp entries at 0: every read is
    // unstamped until a copy stamps again. By default nothing changes; under
    // the override the unstamped pair is refused, and after the scope change
    // (T1's hold reset) the frame shows the flat alpha, never an old pair.
    for (const bool authoritative : {false, true}) {
      alpha_auto_policy session;
      restore_sb_sdr(session);
      session.set_pin_only_ui(true);
      sequence s(session, sb_sdr_page(sb_equipment));
      s.temporal.identity_override = authoritative;
      auto p = sb_sdr_menu_present();
      for (int i = 0; i != 40; ++i) {
        const std::uint32_t label = 300u + std::uint32_t(i);
        p.now_ms = 10000u + std::uint64_t(i) * 16u;
        p.epoch = i < 20 ? 1u : 2u;
        p.identity = {i == 20 || i == 21 ? 0u : label, 0u, 0u, label, 0u, 0u};
        const auto &f = s.step(p);
        if (i == 20 || i == 21) {
          require(f.decision.identity.layer == ui_detection::identity::unstamped, "A recreated stamp entry was not unstamped");
          require(authoritative ? f.source == 4u && f.flat() && !f.decision.reused : f.source == ui_detection::source_pre_ui, "An unstamped pair after a scope change was not refused (enabled) or changed a decision (shadow)");
        } else {
          require(f.source == ui_detection::source_pre_ui, "An exact pair did not decide 12");
        }
      }
      check_counters(s, "S3 unstamped after a scope change");
    }
    // (d) A HUD-less image paired by Present counting under frame generation
    // (inexact: no same-token Backbuffer, no proposal). By default it decides
    // as today (5 from an inexact pair); under the override it is absent: the
    // gate refuses it, so the frame reuses the last exact decision once and
    // then has no mask, while a token batch (the CPU's proof) decides.
    for (const bool authoritative : {false, true}) {
      alpha_auto_policy session;
      accept_hudless(session);
      sequence s(session, difference_frame);
      s.temporal.identity_override = authoritative;
      present p;
      p.offered = candidate::hudless;
      for (int i = 0; i != 40; ++i) {
        p.now_ms = 10000u + std::uint64_t(i) * 16u;
        const bool batch = i < 20 || i >= 23;
        p.identity = {0u, 0u, 0u, 0u, 0u, 0u, batch ? token_batch : 0u};
        p.offered = candidate::hudless | (batch ? candidate::exact : 0u);
        const auto &f = s.step(p);
        if (batch) {
          require(f.source == 5u && f.decision.identity.hudless == ui_detection::identity::exact && f.decision.identity.hudless_space == ui_detection::identity::space_token, "A token batch did not decide 5 as an exact token-space pair");
        } else if (!authoritative) {
          require(f.source == 5u && f.decision.inexact_difference && f.decision.identity.hudless == ui_detection::identity::unproposed, "By default an inexact HUD-less pair did not decide as today");
        } else {
          require(f.decision.identity.hudless == ui_detection::identity::unproposed && !(f.decision.valid_bits & candidate::hudless) && (i == 20 ? f.decision.reused && f.source == 5u : !f.decision.reused && !f.source), "Under S3 an inexact HUD-less pair was not absent with one T1 reuse");
        }
      }
      check_counters(s, "S3 inexact HUD-less pair");
    }
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
    {"H1 Stellar Blade SDR menu visits", h1_stellar_blade_sdr_menu_visits},
    {"H1 Stellar Blade SDR with FG suspended", h1_stellar_blade_sdr_fg_suspended},
    {"H1 Stellar Blade SDR FG session (fix 1)", h1_stellar_blade_sdr_fg_session},
    {"H1/A3 pre-UI proof across sessions", h1_pre_ui_proof_across_sessions},
    {"H1 effects target never proven", h1_effects_target_is_never_proven},
    {"H1/A1 pre-UI proof keys", h1_pre_ui_proof_keys},
    {"H1 dark gameplay never flat", h1_dark_gameplay_never_flat},
    {"H1 overrides a partial winner", h1_overrides_a_partial_winner},
    {"H2 still screens (fix 2)", h2_still_screens},
    {"S1/P1 refine: SB SDR Equipment page (fix 3)", fix3_equipment_page},
    {"V2 SB SDR Settings page (fix 3)", fix3_settings_page},
    {"V2 black loading screen (fix 3)", fix3_loading_screen},
    {"E2 FG on: tags decide, late shadow (fix 3)", fix3_fg_on},
    {"E2 late pairing never acts (fix 3)", fix3_late_pairing_never_acts},
    {"A1 HDR UI layer unchanged (fix 3)", fix3_hdr_layer_unchanged},
    {"S1 refine: E33-like HUD-less set (fix 3)", fix3_e33_hudless_refines},
    {"S1 refine: semi-transparent layer unaffected (fix 3)", fix3_w3_semi_transparent},
    {"T1 missing pair under refine (fix 3)", fix3_missing_pair_t1},
    {"H1 over a refined frame (fix 3)", fix3_h1_over_refined},
    {"S1 declared alpha blocks the pre-UI set (fix 3)", fix3_declared_block},
    {"S1 switch toggled mid-session (fix 3)", fix3_switch_toggled},
    {"A1/F1 change-set shadow inert (fix 3)", fix3_shadow_inert},
    {"E2 FG exactness by paired Presents (fix 3)", fix3_fg_exactness},
    {"T1 change-set gap keeps adoption (fix 3)", fix3_gap_keeps_adoption},
    {"P2 shadow changes nothing (fix 4)", fix4_shadow_changes_nothing},
    {"P2 switch unpins only eligible sources (fix 4)", fix4_switch_unpins_only_eligible_sources},
    {"P2/T1 reuse keeps the darkened mask (fix 4)", fix4_t1_reuse_keeps_the_darkened_mask},
    {"P2/H1/H2 whole-frame decisions untouched (fix 4)", fix4_full_frame_sources_untouched},
    {"P2/S1 refine to 12 with darkening (fix 4)", fix4_refine_to_12_with_darkening},
    {"P2 one switch UIPinOnlyUI (fix 4)", fix4_one_switch},
    {"S3 identity shadow", s3_identity_shadow},
    {"S3 GPU identity verdicts", s3_gpu_identity},
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

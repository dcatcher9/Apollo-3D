// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
  namespace api = reshade::api;
  namespace layer = sunshine_game3d::ui_layer;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }

  api::resource_desc target(api::format format, std::uint32_t width = 3840, std::uint32_t height = 2160) {
    return api::resource_desc(width, height, 1, 1, format, 1, api::memory_heap::default_, api::resource_usage::render_target);
  }
}

int main() {
  try {
    const float transparent[4]{}, black[4]{0.f, 0.f, 0.f, 1.f}, tinted[4]{0.f, 0.f, 0.001f, 0.f};
    for (const auto format : {api::format::r8g8b8a8_unorm, api::format::r8g8b8a8_typeless, api::format::b8g8r8a8_unorm_srgb,
           api::format::r16g16b16a16_float, api::format::r32g32b32a32_float})
      require(layer::qualifies(target(format), transparent, 3840, 2160, 0, nullptr), "A transparent-cleared output-size alpha target did not qualify");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), black, 3840, 2160, 0, nullptr), "Opaque black is not a transparent clear");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), tinted, 3840, 2160, 0, nullptr), "A tinted clear qualified");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm, 2228, 1256), transparent, 3840, 2160, 0, nullptr),
      "A render-resolution scene target qualified as an output layer");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), transparent, 0, 0, 0, nullptr), "No output size must reject everything");
    for (const auto format : {api::format::r11g11b10_float, api::format::r8_unorm, api::format::r16g16_float, api::format::b8g8r8x8_unorm,
           api::format::r10g10b10a2_unorm, api::format::r10g10b10a2_typeless})
      require(!layer::qualifies(target(format), transparent, 3840, 2160, 0, nullptr), "A format without blended alpha qualified");
    auto msaa = target(api::format::r8g8b8a8_unorm); msaa.texture.samples = 4;
    auto mips = target(api::format::r8g8b8a8_unorm); mips.texture.levels = 2;
    auto array = target(api::format::r8g8b8a8_unorm); array.texture.depth_or_layers = 2;
    auto volume = target(api::format::r8g8b8a8_unorm); volume.type = api::resource_type::texture_3d;
    for (const auto &desc : {msaa, mips, array, volume})
      require(!layer::qualifies(desc, transparent, 3840, 2160, 0, nullptr), "A multisampled, mipped, layered or 3D target qualified");
    // Only a whole clear: no rectangles, or one that covers the target. A
    // partial clear (a viewport strip, a split-screen half) is not a layer clear.
    {
      const auto layer_target = target(api::format::r8g8b8a8_unorm);
      const api::rect whole{0, 0, 3840, 2160}, larger{-8, -8, 4096, 4096}, strip{0, 0, 3840, 200}, half{1920, 0, 3840, 2160};
      const api::rect pair[]{strip, whole}, halves[]{{0, 0, 1920, 2160}, half};
      require(layer::qualifies(layer_target, transparent, 3840, 2160, 1, &whole) &&
          layer::qualifies(layer_target, transparent, 3840, 2160, 1, &larger) &&
          layer::qualifies(layer_target, transparent, 3840, 2160, 2, pair), "A clear rectangle covering the target did not qualify");
      require(!layer::qualifies(layer_target, transparent, 3840, 2160, 1, &strip) &&
          !layer::qualifies(layer_target, transparent, 3840, 2160, 1, &half) &&
          !layer::qualifies(layer_target, transparent, 3840, 2160, 2, halves) &&
          !layer::qualifies(layer_target, transparent, 3840, 2160, 1, nullptr),
        "A partial clear qualified as a layer clear");
    }
    std::puts("PASS UI layer census qualification: output-size single-sample 2D alpha targets cleared whole to transparent "
      "black only");

    // A layer copy in its own slot (candidate layout 2) carries the
    // late-layer identity with the premultiplied check; float layers add HDR
    // headroom. Typeless allocations are named by their typeless format.
    namespace detection = sunshine_game3d::ui_detection;
    for (const auto format : {api::format::r8g8b8a8_unorm, api::format::r8g8b8a8_unorm_srgb, api::format::r8g8b8a8_typeless,
           api::format::b8g8r8a8_unorm, api::format::b8g8r8a8_unorm_srgb, api::format::b8g8r8a8_typeless})
      require(layer::detection_flags(format) == 5u &&
          layer::detection_flags(format) == (detection::stored_premultiplied | detection::stored_late_layer),
        "An 8-bit layer lost its late-layer identity or premultiplied check");
    for (const auto format : {api::format::r16g16b16a16_float, api::format::r16g16b16a16_typeless, api::format::r16g16b16a16_unorm,
           api::format::r16g16b16a16_snorm, api::format::r16g16b16a16_uint, api::format::r16g16b16a16_sint,
           api::format::r32g32b32a32_float, api::format::r32g32b32a32_typeless, api::format::r32g32b32a32_uint,
           api::format::r32g32b32a32_sint})
      require(layer::detection_flags(format) == 7u &&
          layer::detection_flags(format) == (detection::stored_premultiplied | detection::stored_hdr_headroom | detection::stored_late_layer),
        "A float layer lost its late-layer identity or HDR headroom");
    // The contract's DXGI test, which ui_detection_replay also uses, names the
    // same float layers as ReShade's typeless families; stored flags never
    // carry a per-frame or reserved bit.
    for (std::uint32_t value = 0; value != 256; ++value) {
      const auto format = static_cast<api::format>(value), typeless = api::format_to_typeless(format);
      if (layer::alpha_format(format))
        require(detection::float_layer_format(value) ==
            (typeless == api::format::r16g16b16a16_typeless || typeless == api::format::r32g32b32a32_typeless),
          "The contract's float layer formats differ from the renderer's typeless families");
      // 0x8u is the reserved stored bit (game3d_ui_detection_contract.h).
      require(!(layer::detection_flags(format) & (detection::per_frame_mask | 0x8u)),
        "Layer flags set a per-frame or reserved bit");
    }
    std::puts("PASS UI layer detection flags: 5 for 8-bit and 7 for float layers, typeless included; no per-frame bits");

    // game3d_native.hlsl mirrors every flag and sizes detection within the
    // contract's range.
    {
      std::ifstream input(SUNSHINE_GAME3D_NATIVE_HLSL, std::ios::binary);
      require(input.good(), "Cannot read game3d_native.hlsl");
      const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      const auto mirrored = [&source](const auto &defines) {
        for (const auto &[name, value] : defines) {
          const auto key = "#define " + std::string(name) + ' ';
          const auto at = source.find(key);
          const auto message = std::string(name) + " is missing from game3d_native.hlsl or differs from game3d_ui_detection_contract.h";
          require(at != std::string::npos && std::stoul(source.substr(at + key.size(), 16), nullptr, 0) == value, message.c_str());
        }
      };
      mirrored(detection::hlsl_flag_defines);
      mirrored(detection::hlsl_scene_defines);
      mirrored(sunshine_game3d::hlsl_counter_defines);
      mirrored(detection::hlsl_candidate_defines);
      mirrored(detection::hlsl_hold_defines);
      mirrored(detection::hlsl_h1_defines);
      // The hidden-scene routes' per-frame bits (0x10000, 0x80000) and the
      // layer's D proof bit (0x20000000) are retired; fix 1's proven bit and
      // the pre-UI statistics row are mirrored.
      require(source.find("SUNSHINE_UI_PER_FRAME_SCENE_HOLD") == std::string::npos &&
          source.find("SUNSHINE_UI_PER_FRAME_PRE_UI_LAYER") == std::string::npos,
        "game3d_native.hlsl still defines a retired hidden-scene route or pre-UI layer bit");
      require(source.find("#define SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN 0x80000000u") != std::string::npos &&
          source.find("#define SUNSHINE_UI_PRE_UI_ROW 128") != std::string::npos,
        "game3d_native.hlsl does not mirror the pre-UI proven bit and statistics row");
      require(sunshine_game3d::shader_marker(source, detection::candidate_layout_marker) == detection::candidate_layout &&
          sunshine_game3d::shader_marker(source, sunshine_game3d::ui_selection::revision_marker) ==
            sunshine_game3d::ui_selection::revision,
        "game3d_native.hlsl's candidate layout or selection revision differs from the contract");
      const auto texels = sunshine_game3d::shader_marker(source, detection::decision_texels_marker);
      const auto images = sunshine_game3d::shader_marker(source, detection::scene_evidence_images_marker);
      require(texels == detection::decision_texels && texels <= detection::max_decision_texels &&
          images == detection::max_scene_evidence_images,
        "game3d_native.hlsl's UI detection size markers are not the current revision's (the renderer requires them)");
      // Selection revision 10: the unaligned-tag shift and the sample and
      // re-offer bits, the layer-bound row, and the invariant counter words
      // 14 and 18 reserved (no define).
      require(source.find("#define SUNSHINE_UI_PER_FRAME_UNALIGNED_SHIFT 12") != std::string::npos &&
          source.find("#define SUNSHINE_UI_PER_FRAME_SAMPLE 0x4000u") != std::string::npos &&
          source.find("#define SUNSHINE_UI_PER_FRAME_REOFFER 0x8000u") != std::string::npos &&
          source.find("#define SUNSHINE_UI_LAYER_BOUND_ROW 208") != std::string::npos &&
          source.find("SUNSHINE_UI_COUNTER_UNTRUSTED_INFERRED") == std::string::npos &&
          source.find("SUNSHINE_UI_COUNTER_PRESENTED_OVER_DEDICATED") == std::string::npos,
        "game3d_native.hlsl does not mirror selection revision 10's bits, layer-bound row and reserved counter words");
    }
    // Selection revision 4 (fix 1) writes the pre-UI pixel counts in texel
    // 11 after the H1 texel 10. Revision 5 (fix 2) added H2's stillness
    // counts in texel 12 and statistics rows 144-159; revision 7 removed fix 3
    // and fix 4 (revision 6), keeping source 12's counter word and word 30
    // reserved (zero); revision 8 added the empty change set; and revision 9
    // removes H2 and S3: 12 texels and 144 statistics rows again, with source
    // 11, texel 12 (words 48-51), rows 144-159, counter words 31-35 and b2
    // words 5-9 reserved, b2 shrunk to 6 words, and the removed defines gone.
    // Revision 10 adds texel 16 (the declared alphas' one-way counts; texels
    // 12-15 stay reserved zeros and words 32 and 36, the layer's one-way
    // counts, become reserved zeros) and rows 208-223 (the layer's pixels
    // beyond the premultiplied bound; rows 144-207 stay reserved), 17 texels
    // and 224 rows, moves the stored/per-frame boundary to 0x1000 for the
    // unaligned declared tags (0x1000, 0x2000) and the sample and re-offer
    // bits, and reserves counter words 14 and 18.
    require(sunshine_game3d::ui_selection::revision == 10u && detection::h1_decision_texels == 11u &&
        detection::decision_word::h1 == 43u && detection::pre_ui_decision_texels == 12u &&
        detection::decision_word::pre_ui_match == 44u && detection::decision_word::pre_ui_image_lit == 45u &&
        detection::per_frame_pre_ui_proven == 0x80000000u && detection::pre_ui_statistics_row == 128u &&
        detection::decision_texels == 17u && detection::decision_word::strong_ui_alpha == 64u &&
        detection::decision_word::strong_backbuffer == 33u && detection::decision_word::contradicted_current == 38u &&
        detection::layer_bound_statistics_row == 208u && detection::statistics_row_count == 224u &&
        detection::per_frame_sample == 0x4000u && detection::per_frame_reoffer == 0x8000u &&
        detection::per_frame_mask == 0xfffff000u && detection::per_frame_unaligned_shift == 12u &&
        detection::per_frame_unaligned_mask == 0x3000u &&
        detection::source_count == 12u && detection::source_layer == 10u && detection::b2_words == 6u &&
        sunshine_game3d::ui_counter_word::decided_count == 13u && sunshine_game3d::ui_counter_word::count == 31u &&
        sunshine_game3d::ui_counter_word::reserved_refined == 30u &&
        sunshine_game3d::ui_counter_word::untrusted_inferred == 14u &&
        sunshine_game3d::ui_counter_word::presented_over_dedicated == 18u,
      "The decision layout is not selection revision 10 with 17 texels and 224 statistics rows");
    {
      std::ifstream input(SUNSHINE_GAME3D_NATIVE_HLSL, std::ios::binary);
      const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      for (const char *removed : {"SUNSHINE_UI_CANDIDATE_PRE_UI", "SUNSHINE_UI_SOURCE_PRE_UI", "SUNSHINE_UI_CHANGE_SET_",
             "SUNSHINE_UI_DARKENING_", "SUNSHINE_UI_PIN_ONLY_UI", "SUNSHINE_UI_COUNTER_REFINED", "SUNSHINE_UI_H1_REFINED",
             "SUNSHINE_UI_TAG_LINEAR", "SUNSHINE_UI_IDENTITY", "SUNSHINE_UI_COUNTER_IDENTITY_",
             "SUNSHINE_UI_SOURCE_STILL", "SUNSHINE_UI_STILL_"})
        require(source.find(std::string("#define ") + removed) == std::string::npos,
          (std::string("game3d_native.hlsl still defines the removed ") + removed).c_str());
    }
    std::puts("PASS UI detection contract: game3d_native.hlsl mirrors every flag, candidate bit, counter word, hold store value, "
      "H1 claim word, the pre-UI statistics rows and the selection revision, and sizes detection within range; removed "
      "defines stay gone");

    // One game frame: each target cleared in order, then Present.
    const auto frame = [](layer::layer_tracker &tracker, std::initializer_list<std::uint64_t> clears, std::uint64_t now) {
      unsigned captures = 0;
      for (const auto resource : clears) captures += tracker.clear(resource, now) ? 1u : 0u;
      tracker.present(now);
      return captures;
    };
    {
      layer::layer_tracker tracker;
      std::uint64_t now = 1000;
      for (unsigned i = 1; i < layer::confirm_clears; ++i, now += 16)
        require(!frame(tracker, {7}, now) && !tracker.active(), "A layer was trusted before enough clears");
      require(!frame(tracker, {7}, now) && tracker.active() == 7, "A layer cleared every frame was not confirmed");
      now += 16;
      require(frame(tracker, {7, 7, 7}, now) == 1, "The active layer was not copied exactly once per Present");
      layer::layer_tracker burst;
      require(!frame(burst, {8, 8, 8, 8}, now) && !burst.active(), "Clears within one frame confirmed a layer");
      // Of two confirmed layers, the one cleared last in the frame holds the UI.
      for (unsigned i = 0; i < layer::confirm_clears; ++i) frame(tracker, {9, 7}, now += 16);
      require(tracker.active() == 7, "The layer cleared last in the frame was not chosen");
      for (unsigned i = 0; i < layer::confirm_clears; ++i) frame(tracker, {7, 9}, now += 16);
      require(tracker.active() == 9, "The choice did not follow the clear order");
      // A layer that stops being cleared expires.
      tracker.present(now + layer::max_clear_gap_ms + 1);
      require(!tracker.active(), "A layer no longer cleared stayed active");
    }
    {
      layer::layer_tracker tracker;
      std::uint64_t now = 1000;
      // Cleared only now and then: never a per-frame layer.
      for (unsigned i = 0; i < 6; ++i, now += layer::max_clear_gap_ms + 1) frame(tracker, {5}, now);
      require(!tracker.active(), "A target with long gaps between clears was confirmed");
      for (unsigned i = 0; i < layer::confirm_clears; ++i, now += 16) frame(tracker, {1, 2, 3, 4, 5, 6}, now);
      require(tracker.active() == 6, "Several per-frame targets were not all tracked");
      // Many transient targets never displace a confirmed layer.
      std::uint64_t transient = 100;
      for (unsigned i = 0; i < 8; ++i, now += 16)
        frame(tracker, {transient++, transient++, transient++, transient++, transient++, transient++, transient++, transient++, 6}, now);
      require(tracker.active() == 6, "Transient targets displaced the confirmed layer");
      require(!tracker.clear(0, now), "A null resource was tracked");
      tracker.reset();
      require(!tracker.active(), "Reset kept an active layer");
    }
    std::puts("PASS UI layer tracking: confirmed after repeated clears, last-cleared layer active, one copy per Present, expiry");
    // The Present count at the live copy names the Present whose frame it
    // holds (presents_since_copy); the provider reads it after the Present's
    // observe_output and reports it with the layer's metadata.
    {
      layer::layer_tracker tracker;
      std::uint64_t now = 1000;
      for (unsigned i = 0; i < layer::confirm_clears; ++i, now += 16) frame(tracker, {7}, now);
      require(tracker.active() == 7 && !tracker.presents_since_copy(), "A tracker without a copy counted Presents");
      // Present k observed; the first clear of frame k+1 copies (the content
      // Present k showed), then Present k+1 is observed and its render reads 1.
      require(tracker.clear(7, now), "The active layer's first clear after a Present did not ask for its copy");
      tracker.copied();
      require(!tracker.presents_since_copy(), "A copy recorded in the current interval did not read 0");
      tracker.present(now += 16);
      require(tracker.presents_since_copy() == 1u, "A copy after Present k did not read 1 at render k+1");
      // An interval without a qualifying clear: the copy is two Presents old.
      tracker.present(now += 16);
      require(tracker.presents_since_copy() == 2u, "An interval without a copy did not read 2");
      // The next copy restarts the count.
      require(tracker.clear(7, now), "The next interval's first clear did not ask for a copy");
      tracker.copied();
      tracker.present(now += 16);
      require(tracker.presents_since_copy() == 1u, "A new copy did not restart the Present count");
      tracker.reset();
      require(!tracker.presents_since_copy(), "Reset kept the Present count of a copy");
    }
    std::puts("PASS UI layer Present count: a copy after Present k reads 1 at render k+1, 2 after an interval without "
      "one, 0 within its own interval");
    // Queue order: each live copy keeps the queue that executed it, and is in
    // order (bound directly) only when that is the presenting queue alone. The
    // verdict is per copy: the next copy run on the presenting queue is in
    // order again.
    {
      constexpr std::uint64_t presenting = 0x10, other = 0x20;
      std::uint64_t copy = 0;
      copy = layer::merge_queue(copy, presenting);
      require(copy == presenting && layer::presented_in_order(copy, presenting),
        "A copy run on the presenting queue was not in order");
      copy = layer::merge_queue(copy, presenting);
      require(layer::presented_in_order(copy, presenting), "A list executed twice on the presenting queue lost its order");
      copy = layer::merge_queue(copy, other);
      require(copy == layer::mixed_queue && !layer::presented_in_order(copy, presenting) &&
          !layer::presented_in_order(layer::merge_queue(copy, presenting), presenting),
        "A copy run on two queues was in order");
      require(!layer::presented_in_order(layer::merge_queue(0, other), presenting), "A copy run on another queue was in order");
      require(!layer::presented_in_order(layer::merge_queue(0, presenting), 0) && !layer::presented_in_order(0, presenting),
        "A copy was in order with an unknown presenting queue or before it executed");
      require(layer::presented_in_order(layer::merge_queue(0, presenting), presenting),
        "A new copy on the presenting queue was not in order after a copy on another queue");
    }
    std::puts("PASS UI layer queue order: a copy run on the presenting queue alone is in order (bound directly); one run on "
      "another queue, on two queues, before it executed or with an unknown presenting queue is not; the verdict is per copy");
    // Live-copy ring (direct binding): the oldest free entry other than the
    // newest takes the next copy; with every entry read by an unfinished
    // render the ring grows to ring_capacity, then the copy is skipped.
    {
      std::array<bool, layer::ring_capacity> free{};
      std::array<std::uint64_t, layer::ring_capacity> age{};
      auto choice = layer::choose_ring_entry(0, free, age, -1);
      require(choice.index == 0 && choice.allocate, "An empty ring did not allocate its first entry");
      free = {true, true, true, false}; age = {7, 5, 9, 0};
      choice = layer::choose_ring_entry(3, free, age, 2);
      require(choice.index == 1 && !choice.allocate, "The ring did not reuse its oldest free entry");
      choice = layer::choose_ring_entry(3, free, age, 1);
      require(choice.index == 0 && !choice.allocate, "The ring reused the newest copy");
      free = {false, false, true, false};
      choice = layer::choose_ring_entry(3, free, age, 2);
      require(choice.index == 3 && choice.allocate, "A busy ring did not grow");
      free = {false, false, false, true};
      choice = layer::choose_ring_entry(4, free, age, 3);
      require(choice.index < 0, "A full busy ring overwrote an entry a render still reads");
      choice = layer::choose_ring_entry(4, {true, false, false, true}, age, 3);
      require(choice.index == 0, "A full ring ignored a completed entry");
      // An entry whose allocation failed (no copy, age UINT64_MAX) is taken
      // last: a free entry with a copy wins, and it is allocated again only
      // when it is the only free one; an entry being allocated is not free.
      choice = layer::choose_ring_entry(3, {true, true, false, false}, {7, UINT64_MAX, 9, 0}, -1);
      require(choice.index == 0 && !choice.allocate, "An entry whose allocation failed beat a free entry with a copy");
      choice = layer::choose_ring_entry(3, {false, true, false, false}, {7, UINT64_MAX, 9, 0}, -1);
      require(choice.index == 1 && !choice.allocate, "An entry whose allocation failed was not retried as the only free one");
    }
    std::puts("PASS UI layer live-copy ring: oldest free non-newest entry, growth to the capacity, skip when every entry is read");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}

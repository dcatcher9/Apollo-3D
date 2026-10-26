// SPDX-License-Identifier: GPL-3.0-only
// The API contract mappers of api_contract.h (refactor plan P3): the state
// proof, lifetime and native state each API's declaration gives a capture
// input, the lifecycles the probe reads and match() judges at each copy point,
// which of those refusals keep a tag call's copy (nominate_depth_source),
// Dump 3D's own contract for its UI copies, and an absent extent as the whole
// allocation. The Streamline probe's depth and UI
// adapters, tag_rejection, the 1.x state resolver, the dump's UI capture, the
// NGX adapter, DLSS-G's own inputs and the Generic copy read these; the rows
// pin today's mapping for every declaration those callers pass.
#include "api_contract.h"

#include <cstdint>
#include <cstdio>
#include <iterator>

namespace {
  using namespace sunshine_api_contract;
  using streamline::copy_point;

  unsigned failures {};

  void require(bool condition, const char *pins) {
    if (!condition) {
      std::fprintf(stderr, "FAIL %s\n", pins);
      ++failures;
    }
  }

  // A tag's lifecycle at a copy point. A UI tag is always copied at its tag
  // call; depth also at a later evaluate.
  struct lifetime_row {
    const char *pins;
    bool version_one, supported;
    std::uint32_t lifecycle;
    copy_point at;
    lifetime expected;
  };

  constexpr lifetime_row lifetimes[] {
    {"A 1.x tag has no lifecycle and holds until Present", true, true, streamline::no_lifecycle, copy_point::evaluate,
     lifetime::until_present},
    {"ValidUntilPresent holds until Present at its tag call", false, true, streamline::valid_until_present,
     copy_point::tag_call, lifetime::until_present},
    {"ValidUntilPresent holds until Present at a later evaluate", false, true, streamline::valid_until_present,
     copy_point::evaluate, lifetime::until_present},
    {"ValidUntilEvaluate holds until its evaluation at its tag call", false, true, streamline::valid_until_evaluate,
     copy_point::tag_call, lifetime::until_evaluation},
    {"ValidUntilEvaluate holds until its evaluation at the evaluate", false, true, streamline::valid_until_evaluate,
     copy_point::evaluate, lifetime::until_evaluation},
    {"OnlyValidNow holds at its own tag call (depth at a tag boundary, every UI tag)", false, true,
     streamline::only_valid_now, copy_point::tag_call, lifetime::at_call},
    {"OnlyValidNow is gone at a later evaluate", false, true, streamline::only_valid_now, copy_point::evaluate,
     lifetime::unsupported},
    {"ValidUntilPresent holds until Present on the Present queue (R2)", false, true, streamline::valid_until_present,
     copy_point::present, lifetime::until_present},
    {"OnlyValidNow is gone at its frame's Present", false, true, streamline::only_valid_now, copy_point::present,
     lifetime::unsupported},
    {"A lifecycle above ValidUntilEvaluate is unsupported", false, true, 3, copy_point::tag_call, lifetime::unsupported},
    {"A 2.x tag without a lifecycle is unsupported", false, true, streamline::no_lifecycle, copy_point::tag_call,
     lifetime::unsupported},
    {"An unsupported tag (tag_refusal) has no lifetime", false, false, streamline::valid_until_present,
     copy_point::tag_call, lifetime::unsupported},
    {"An unsupported 1.x tag has no lifetime", true, false, streamline::no_lifecycle, copy_point::evaluate,
     lifetime::unsupported},
  };

  // Where an observation that reads a tag copies it: inside the call that
  // supplied it, which for a local input is its evaluate (S4).
  struct copied_at_row {
    const char *pins;
    bool tag_call, local;
    copy_point expected;
  };

  constexpr copied_at_row copied_ats[] {
    {"A tag call copies its tags inside the call", true, false, copy_point::tag_call},
    {"An evaluate copies a local input inside the call that supplied it (S4)", false, true, copy_point::tag_call},
    {"An evaluate copies a global or frame tag at a later evaluate", false, false, copy_point::evaluate},
  };

  // Resource::state and what proves it.
  struct proof_row {
    const char *pins;
    bool version_one, requires_observation;
    std::uint32_t state;
    state_proof expected;
  };

  constexpr proof_row proofs[] {
    {"A 2.x COMMON (0) is declared", false, false, 0, state_proof::declared},
    {"A 2.x DEPTH_READ is declared", false, false, 0x20, state_proof::declared},
    {"An undeclared 2.x state (UINT32_MAX) claims nothing", false, false, UINT32_MAX, state_proof::unavailable},
    {"The zero-header compatibility layout's 0 needs an observed state", false, true, 0, state_proof::observed_nonzero},
    {"A 1.x 0 is chi eUnknown and needs an observed state", true, false, 0, state_proof::observed_nonzero},
    {"A 1.x DEPTH_WRITE is declared", true, false, 0x10, state_proof::declared},
    {"An undeclared 1.x state claims nothing", true, false, UINT32_MAX, state_proof::unavailable},
  };

  // The lifecycle match() judges: one that holds at the copy point reads as
  // ValidUntilPresent; any other is unchanged.
  struct matched_row {
    const char *pins;
    std::uint32_t lifecycle;
    copy_point at;
    std::uint32_t expected;
  };

  constexpr matched_row matched[] {
    {"ValidUntilPresent is judged as itself at its tag call", streamline::valid_until_present, copy_point::tag_call,
     streamline::valid_until_present},
    {"ValidUntilPresent is judged as itself at an evaluate", streamline::valid_until_present, copy_point::evaluate,
     streamline::valid_until_present},
    {"ValidUntilEvaluate is judged as ValidUntilPresent at its tag call", streamline::valid_until_evaluate,
     copy_point::tag_call, streamline::valid_until_present},
    {"ValidUntilEvaluate is judged as ValidUntilPresent at the evaluate", streamline::valid_until_evaluate,
     copy_point::evaluate, streamline::valid_until_present},
    {"OnlyValidNow is judged as ValidUntilPresent at its tag call", streamline::only_valid_now, copy_point::tag_call,
     streamline::valid_until_present},
    {"OnlyValidNow stays OnlyValidNow at an evaluate (match refuses it)", streamline::only_valid_now,
     copy_point::evaluate, streamline::only_valid_now},
    {"A 1.x tag's missing lifecycle stays (match admits it)", streamline::no_lifecycle, copy_point::evaluate,
     streamline::no_lifecycle},
    {"An unknown lifecycle stays (match refuses it)", 3, copy_point::tag_call, 3},
  };

  // Dump 3D's optional UI copy: a Streamline 2.x tag's lifetime at its tag
  // call, a 1.x tag or an NGX input at the call, and only a declared state
  // other than 0 or UINT32_MAX (state_declared) declared.
  struct dump_row {
    const char *pins;
    bool streamline_v2;
    std::uint32_t lifecycle;
    bool state_declared;
    std::uint32_t state;
    std::uint32_t native_state;
    state_proof proof;
    lifetime expected;
  };

  constexpr dump_row dumps[] {
    {"A dumped 2.x OnlyValidNow tag is copied at its tag call", true, streamline::only_valid_now, true, 0x80, 0x80,
     state_proof::declared, lifetime::at_call},
    {"A dumped 2.x ValidUntilPresent tag holds until Present", true, streamline::valid_until_present, true, 0x80, 0x80,
     state_proof::declared, lifetime::until_present},
    {"A dumped 2.x ValidUntilEvaluate tag holds until its evaluation", true, streamline::valid_until_evaluate, true, 0x80,
     0x80, state_proof::declared, lifetime::until_evaluation},
    {"A dumped 2.x tag of an unknown lifecycle is unsupported", true, 3, true, 0x80, 0x80, state_proof::declared,
     lifetime::unsupported},
    {"A dumped 2.x COMMON (0) needs an observed state", true, streamline::valid_until_present, false, 0, UINT32_MAX,
     state_proof::observed_nonzero, lifetime::until_present},
    {"A dumped 1.x tag is copied at the call", false, streamline::no_lifecycle, true, 0x10, 0x10, state_proof::declared,
     lifetime::at_call},
    {"A dumped 1.x 0 needs an observed state", false, streamline::no_lifecycle, false, 0, UINT32_MAX,
     state_proof::observed_nonzero, lifetime::at_call},
    {"A dumped NGX input names no state and is copied at the call", false, streamline::no_lifecycle, false, 0, UINT32_MAX,
     state_proof::observed_nonzero, lifetime::at_call},
  };

  // Where a frame-generation or UI tag call copies a tag (R2): inside the
  // call from a state its command list records, a ValidUntilPresent tag else
  // (and where that state writes it, S11) on the Present queue before its
  // frame's Present, an UntilEvaluate tag else at its evaluate (none here).
  struct tag_call_copy_row {
    const char *pins;
    std::uint32_t lifecycle;
    bool command_list, recorded, writing;
    streamline::tag_call_copy expected;
  };

  constexpr tag_call_copy_row tag_call_copies[] {
    {"A ValidUntilPresent tag without a command list is copied before its Present (S6)", streamline::valid_until_present,
     false, false, false, streamline::tag_call_copy::present},
    {"A ValidUntilPresent tag whose list records no state is copied before its Present (S11b)",
     streamline::valid_until_present, true, false, false, streamline::tag_call_copy::present},
    {"A ValidUntilPresent tag whose list records its state is copied at the call (S5)", streamline::valid_until_present,
     true, true, false, streamline::tag_call_copy::at_call},
    {"A ValidUntilPresent tag whose list records a write state is copied before its Present (S11)",
     streamline::valid_until_present, true, true, true, streamline::tag_call_copy::present},
    {"An OnlyValidNow tag is copied at its call (S1)", streamline::only_valid_now, true, false, false,
     streamline::tag_call_copy::at_call},
    {"An OnlyValidNow tag in a write state is copied at its call (S1)", streamline::only_valid_now, true, true, true,
     streamline::tag_call_copy::at_call},
    {"An OnlyValidNow tag without a command list has no copy point (S2)", streamline::only_valid_now, false, false, false,
     streamline::tag_call_copy::none},
    {"An UntilEvaluate tag whose list records no state waits for its evaluate (S11c)", streamline::valid_until_evaluate,
     true, false, false, streamline::tag_call_copy::none},
    {"An UntilEvaluate tag whose list records its state is copied at the call (S5)", streamline::valid_until_evaluate,
     true, true, false, streamline::tag_call_copy::at_call},
    {"An UntilEvaluate tag whose list records a write state is copied at the call, which its evaluate repeats (S11)",
     streamline::valid_until_evaluate, true, true, true, streamline::tag_call_copy::at_call},
    {"An UntilEvaluate tag without a command list has no copy point (S2)", streamline::valid_until_evaluate, false, false,
     false, streamline::tag_call_copy::none},
  };

  // The proof behind a declaration where a frame-generation tag call copies
  // its depth tag (S11): a longer-lived tag's declaration describes its later
  // use, so the state the call's list records proves the copy.
  struct tag_call_proof_row {
    const char *pins;
    std::uint32_t lifecycle;
    state_proof supplied, expected;
  };

  constexpr tag_call_proof_row tag_call_proofs[] {
    {"A ValidUntilPresent declaration describes its frame's Present, not its tag call (S11)", streamline::valid_until_present,
     state_proof::declared, state_proof::declared_for_use},
    {"An UntilEvaluate declaration describes its evaluate, not its tag call (S11)", streamline::valid_until_evaluate,
     state_proof::declared, state_proof::declared_for_use},
    {"An OnlyValidNow declaration describes its tag call itself (S11a)", streamline::only_valid_now, state_proof::declared,
     state_proof::declared},
    {"An undeclared state stays a claim of nothing", streamline::valid_until_present, state_proof::unavailable,
     state_proof::unavailable},
    {"A state that needs an observation keeps needing it", streamline::valid_until_present, state_proof::observed_nonzero,
     state_proof::observed_nonzero},
  };

  // Whether an evaluate that refuses a tag for its lifetime keeps the copy (or
  // the refusal) its tag call made (S1, S3): only an OnlyValidNow global or
  // frame tag of the evaluate's own frame. Until 10-26 every such evaluate
  // withdrew it (the matrix's S1-evaluate row).
  struct kept_row {
    const char *pins;
    bool version_one, supported;
    std::uint32_t lifecycle;
    copy_point at;
    bool same_frame, kept;
  };

  constexpr kept_row kept_copies[] {
    {"An evaluate of its frame keeps an OnlyValidNow tag's tag-call copy (S1, S3)", false, true, streamline::only_valid_now,
     copy_point::evaluate, true, true},
    {"An evaluate of a later frame is an attempt of its own", false, true, streamline::only_valid_now, copy_point::evaluate,
     false, false},
    {"A tag call, or the evaluate a local input came with (S4), decides the tag itself", false, true,
     streamline::only_valid_now, copy_point::tag_call, true, false},
    {"A ValidUntilPresent tag is the evaluate's own to copy", false, true, streamline::valid_until_present,
     copy_point::evaluate, true, false},
    {"A ValidUntilEvaluate tag is the evaluate's own to copy", false, true, streamline::valid_until_evaluate,
     copy_point::evaluate, true, false},
    {"A 1.x tag holds until Present", true, true, streamline::no_lifecycle, copy_point::evaluate, true, false},
    {"An unsupported tag keeps nothing", false, false, streamline::only_valid_now, copy_point::evaluate, true, false},
    {"An undefined lifecycle keeps nothing", false, true, 3, copy_point::evaluate, true, false},
    {"An OnlyValidNow tag keeps nothing at its frame's Present", false, true, streamline::only_valid_now,
     copy_point::present, true, false},
  };

  // Whether a null untag cancels a copy deferred to its Present (R7).
  struct untag_row {
    const char *pins;
    bool untag_framed, tag_framed, same_frame, cancels;
  };

  constexpr untag_row untags[] {
    {"The same frame's untag cancels the deferred copy", true, true, true, true},
    {"Another frame's untag leaves it", true, true, false, false},
    {"An untag without a frame token cancels it", false, true, false, true},
    {"Any untag cancels a deferred tag that has no frame", true, false, false, true},
  };

  // Streamline's sl::Extent orders top before left.
  struct top_first_extent {
    std::uint32_t top {}, left {}, width {}, height {};
  };
}

int main() {
  for (const auto &row : lifetimes) {
    streamline::tag_declaration tag;
    tag.version_one = row.version_one;
    tag.supported = row.supported;
    tag.lifecycle = row.lifecycle;
    require(streamline::valid_until(tag, row.at) == row.expected, row.pins);
  }
  for (const auto &row : proofs) {
    streamline::tag_declaration tag;
    tag.version_one = row.version_one;
    tag.state_requires_observation = row.requires_observation;
    tag.state = row.state;
    require(streamline::proof(tag) == row.expected, row.pins);
  }
  for (const auto &row : matched) require(streamline::lifecycle_at(row.lifecycle, row.at) == row.expected, row.pins);
  for (const auto &row : copied_ats) require(streamline::copied_at(row.tag_call, row.local) == row.expected, row.pins);
  {
    streamline::tag_declaration tag;
    tag.lifecycle = streamline::only_valid_now;
    require(streamline::valid_until(tag, streamline::copied_at(false, true)) == lifetime::at_call &&
              streamline::lifecycle_at(tag.lifecycle, streamline::copied_at(false, true)) == streamline::valid_until_present,
      "An OnlyValidNow local input holds inside its own evaluate, where match() admits it (S4)");
  }
  for (const auto &row : tag_call_copies)
    require(streamline::deferred_copy(row.lifecycle, row.command_list, row.recorded, row.writing) == row.expected, row.pins);
  for (const auto &row : tag_call_proofs)
    require(streamline::tag_call_proof(row.lifecycle, row.supplied) == row.expected, row.pins);
  for (const auto &row : untags)
    require(streamline::untag_cancels(row.untag_framed, row.tag_framed, row.same_frame) == row.cancels, row.pins);
  for (const auto &row : kept_copies) {
    streamline::tag_declaration tag;
    tag.version_one = row.version_one;
    tag.supported = row.supported;
    tag.lifecycle = row.lifecycle;
    require(streamline::keeps_tag_call_copy(tag, row.at, row.same_frame) == row.kept, row.pins);
  }
  {
    // The rule keeps exactly what the lifetime refuses at that evaluate.
    streamline::tag_declaration tag;
    tag.lifecycle = streamline::only_valid_now;
    require(streamline::valid_until(tag, copy_point::evaluate) == lifetime::unsupported &&
              streamline::keeps_tag_call_copy(tag, streamline::copied_at(false, false), true) &&
              !streamline::keeps_tag_call_copy(tag, streamline::copied_at(false, true), true),
      "An OnlyValidNow global or frame tag refused at its frame's evaluate keeps its tag call's copy; a local input is "
      "its evaluate's own");
  }
  require(streamline::defined_lifecycle(streamline::only_valid_now) &&
            streamline::defined_lifecycle(streamline::valid_until_present) &&
            streamline::defined_lifecycle(streamline::valid_until_evaluate) && !streamline::defined_lifecycle(3) &&
            !streamline::defined_lifecycle(streamline::no_lifecycle),
    "The probe reads only a lifecycle sl::ResourceLifecycle defines");
  for (const auto &row : dumps) {
    dump::ui_declaration input;
    input.streamline_v2 = row.streamline_v2;
    input.lifecycle = row.lifecycle;
    input.state_declared = row.state_declared;
    input.state = row.state;
    const auto value = dump::ui_copy(input);
    require(value.native_state == row.native_state && value.proof == row.proof && value.valid_until == row.expected,
      row.pins);
  }
  {
    streamline::tag_declaration tag;
    tag.lifecycle = streamline::only_valid_now;
    tag.state = 0x40;
    const auto value = streamline::map(tag, copy_point::tag_call);
    require(value.native_state == 0x40 && value.proof == state_proof::declared && value.valid_until == lifetime::at_call,
      "A Streamline tag's contract names its declared state, its proof and its lifetime");
  }
  require(ngx::evaluation_input.native_state == 0x40 && ngx::evaluation_input.proof == state_proof::sdk_contract &&
            ngx::evaluation_input.valid_until == lifetime::until_evaluation,
    "An NGX evaluation's input is in NON_PIXEL_SHADER_RESOURCE by the SDK contract for the evaluation");
  require(ngx::observed_input.native_state == UINT32_MAX && ngx::observed_input.proof == state_proof::observed_nonzero &&
            ngx::observed_input.valid_until == lifetime::until_evaluation,
    "An NGX input no guide states names no state: only an observed state proves it, for the evaluation");
  // One mapper entry per proof: DLSS-G's own inputs and every feature but
  // Super Resolution take that same contract, never a copy of it.
  static_assert(&ngx::frame_generation_input == &ngx::observed_input, "DLSS-G's inputs have their own copy of observed_input");
  static_assert(&ngx::evaluation_input_of(13) == &ngx::observed_input && &ngx::evaluation_input_of(1) == &ngx::evaluation_input,
    "A feature's Depth contract is not the named mapper entry");
  {
    const auto sr = ngx::evaluation_input_of(1);
    require(sr.native_state == 0x40 && sr.proof == state_proof::sdk_contract && sr.valid_until == lifetime::until_evaluation,
      "A Super Resolution Depth has the SDK contract's state (N1)");
    for (const std::uint32_t feature : {0u, 13u, 2u, 0xfffffffeu}) {
      const auto value = ngx::evaluation_input_of(feature);
      require(value.native_state == ngx::observed_input.native_state && value.proof == ngx::observed_input.proof &&
                value.valid_until == ngx::observed_input.valid_until,
        "Ray Reconstruction's and any other feature's Depth is not observed_input (N1b)");
    }
  }
  {
    const auto value = generic::preserved(0x10);
    require(value.native_state == 0x10 && value.proof == state_proof::declared && value.valid_until == lifetime::until_present,
      "A Generic boundary other than a clear declares its caller's state, held until Present");
    require(generic::clear.native_state == 0x10 && generic::clear.proof == state_proof::clear_contract &&
              generic::clear.valid_until == lifetime::until_present,
      "A depth clear's DEPTH_WRITE is the D3D12 clear contract, held until Present");
    require(generic::end_of_frame.native_state == UINT32_MAX && generic::end_of_frame.proof == state_proof::last_proven &&
              generic::end_of_frame.valid_until == lifetime::until_present,
      "The end of the frame names no state: the source's last proven state supplies it (D10)");
    const auto writable = generic::bound_view(false), read_only = generic::bound_view(true);
    require(writable.native_state == 0x10 && writable.proof == state_proof::declared && writable.valid_until == lifetime::until_present,
      "A writable bound view's boundary declares DEPTH_WRITE, held until Present");
    require(read_only.native_state == UINT32_MAX && read_only.proof == state_proof::observed_nonzero &&
              read_only.valid_until == lifetime::until_present,
      "A read-only bound view's boundary names no state: only its recording's observation proves it (D11c)");
  }
  {
    sunshine_scene_depth::frame input;
    apply(ngx::evaluation_input, input);
    require(input.native_state == 0x40 && input.proof == state_proof::sdk_contract &&
              input.valid_until == lifetime::until_evaluation,
      "A contract is written into a capture input whole");
  }
  {
    const auto whole = normalized_extent(extent {}, 1920, 1080);
    require(!whole.left && !whole.top && whole.width == 1920 && whole.height == 1080,
      "An absent extent is the whole allocation");
    const auto active = normalized_extent(extent {8, 4, 1280, 720}, 1920, 1080);
    require(active.left == 8 && active.top == 4 && active.width == 1280 && active.height == 720,
      "A nonzero extent is its own rectangle");
    const auto tagged = normalized_extent(top_first_extent {}, 1920, 1080);
    require(!tagged.top && !tagged.left && tagged.width == 1920 && tagged.height == 1080,
      "Streamline's absent extent is the whole allocation in its own field order");
    for (const extent zero_sided : {extent {0, 0, 1280, 0}, extent {0, 0, 0, 720}, extent {8, 4, 0, 0}}) {
      const auto value = normalized_extent(zero_sided, 1920, 1080);
      require(null_extent(zero_sided) && !value.left && !value.top && value.width == 1920 && value.height == 1080,
        "An extent with a zero side is null, the whole allocation, whatever its top and left (S16)");
    }
    require(extent_in_allocation(extent {0, 0, 2228, 1256}, 3840, 2160, 1920, 1080),
      "A declared Resource size bounds no extent: the allocation does (S16d)");
    require(extent_in_allocation(extent {200, 120, 3640, 2040}, 3840, 2160, 0, 0) &&
              !extent_in_allocation(extent {201, 120, 3640, 2040}, 3840, 2160, 0, 0) &&
              !extent_in_allocation(extent {0, 0, 3968, 2160}, 3840, 2160, 3840, 2160) &&
              !extent_in_allocation(extent {UINT32_MAX, 0, 2, 2}, 3840, 2160, 0, 0),
      "An extent must lie inside the allocation (S16b, S16c)");
    require(extent_in_allocation(extent {}, 3840, 2160, 3840, 2160) && extent_in_allocation(extent {}, 3840, 2160, 0, 2160) &&
              !extent_in_allocation(extent {}, 3840, 2160, 1920, 1080) &&
              !extent_in_allocation(extent {0, 0, 1280, 0}, 3840, 2160, 3840, 1080),
      "A null extent is the whole allocation, which a declared size must then name");
  }
  if (failures) {
    std::fprintf(stderr, "API contract: %u cells changed\n", failures);
    return 1;
  }
  std::printf("PASS API contract: %zu lifetime, %zu proof, %zu matched-lifecycle, %zu copy-point, %zu tag-call copy, %zu "
    "untag, %zu kept tag-call copy and %zu Dump 3D rows, the defined lifecycles, each API's contract and the "
    "whole-allocation extent\n",
    std::size(lifetimes), std::size(proofs), std::size(matched), std::size(copied_ats), std::size(tag_call_copies),
    std::size(untags), std::size(kept_copies), std::size(dumps));
  return 0;
}

// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"

#include <cstdint>

// How an API's declaration of an input becomes the capture input's contract:
// the state it names and the proof behind it (state_proof), how long the
// resource stays valid (lifetime), its active extent and the typed format a
// colour snapshot of it is read as. One pure mapper per API, shared by depth
// and UI, and Dump 3D's own for its optional UI copies. Admission stays with
// the capture owner: copy_state (capture_state_policy.h) decides a copy from
// the proof and the recording's own state, and a depth snapshot's view
// format is depth_view_format.h's.
namespace sunshine_api_contract {
  using sunshine_scene_depth::extent;
  using sunshine_scene_depth::lifetime;
  using sunshine_scene_depth::state_proof;

  // What a declaration gives a capture input: the native state it names
  // (UINT32_MAX none), the proof behind that state, and its lifetime.
  struct contract {
    std::uint32_t native_state{UINT32_MAX};
    state_proof proof{state_proof::unavailable};
    lifetime valid_until{lifetime::unsupported};
  };
  inline void apply(const contract &value, sunshine_scene_depth::frame &out) {
    out.native_state = value.native_state;
    out.proof = value.proof;
    out.valid_until = value.valid_until;
  }

  // Streamline's sl::Extent is null when !(width && height) (sl_consts.h,
  // its operator bool): the whole resource, whatever its top and left ("if
  // using the entire resource leave as null"; the matrix's S16).
  // Extent: the capture's extent or Streamline's sl::Extent (top first).
  template<class Extent>
  constexpr bool null_extent(const Extent &area) {
    return !(area.width && area.height);
  }
  // A null extent is the whole allocation; any other is its own rectangle.
  template<class Extent>
  constexpr Extent normalized_extent(const Extent &area, std::uint32_t width, std::uint32_t height) {
    if (!null_extent(area)) return area;
    Extent whole{};
    whole.width = width; whole.height = height;
    return whole;
  }
  // Whether a tag's extent names a region of its width x height allocation:
  // a rectangle inside it (S16b; one outside it is malformed, S16c), whatever
  // the Resource's declared size, which is optional (SLT:334-336, 360-363;
  // S16d), or the null extent, the whole allocation, which a declared size (0
  // none) must then name.
  template<class Extent>
  constexpr bool extent_in_allocation(const Extent &area, std::uint64_t width, std::uint32_t height,
      std::uint32_t declared_width, std::uint32_t declared_height) {
    if (null_extent(area))
      return (!declared_width || declared_width == width) && (!declared_height || declared_height == height);
    return area.left <= width && area.top <= height && area.width <= width - area.left && area.height <= height - area.top;
  }

  // Streamline (sl_core_types.h): a tag's Resource and its lifecycle.
  namespace streamline {
    // sl::ResourceLifecycle, a 2.x tag's lifecycle.
    inline constexpr std::uint32_t only_valid_now = 0, valid_until_present = 1, valid_until_evaluate = 2;
    // A 1.x Resource has no lifecycle field: its tags carry this and hold
    // until Present.
    inline constexpr std::uint32_t no_lifecycle = UINT32_MAX;
    // A lifecycle sl::ResourceLifecycle defines. The probe refuses a 2.x tag
    // with any other as it reads it (tag_refusal::lifecycle); valid_until
    // judges a defined one at its copy point.
    constexpr bool defined_lifecycle(std::uint32_t lifecycle) {
      return lifecycle <= valid_until_evaluate;
    }
    // Where the copy is made. OnlyValidNow holds only inside the call that
    // supplied the tag (tag_call): its tag call, or for a local tag the
    // evaluate it is an input of. A UI tag is copied there, while depth is
    // also copied at a later evaluate (evaluate), where an OnlyValidNow
    // resource is gone. A ValidUntilPresent tag may also be copied on the
    // Present queue right before its frame's Present (present, deferred_copy
    // below), where its declared state is exact.
    enum class copy_point : std::uint8_t { tag_call, evaluate, present };
    // Where an observation that reads a tag copies it: inside a tag call, and
    // inside the evaluate a local tag came with, since local tags "do NOT
    // interact with" global ones and are current at that evaluate
    // (sl_core_api.h slEvaluateFeature, SLA:258-262; the matrix's S4); a
    // global tag at an evaluate is copied at a later evaluate.
    constexpr copy_point copied_at(bool tag_call, bool local) {
      return tag_call || local ? copy_point::tag_call : copy_point::evaluate;
    }

    // A tag as the probe read it.
    struct tag_declaration {
      std::uint32_t lifecycle{no_lifecycle};
      // Resource::state: a native D3D12 state, UINT32_MAX undeclared. A 1.x
      // Resource's 0 is chi eUnknown (streamline_v1_resource_state.h), not
      // COMMON.
      std::uint32_t state{UINT32_MAX};
      bool version_one{};
      // false: unreadable or of an unsupported layout (tag_refusal).
      bool supported{true};
      // The zero-header compatibility layout's 0 proves nothing: it may be
      // the zeroing (S14). Its nonzero state is declared.
      bool state_requires_observation{};
    };

    // The proof behind Resource::state. A 1.x 0, and the compatibility
    // layout's 0, need a state observed on the recording (or, for exactly
    // 1.1.1, its private property); UINT32_MAX claims nothing.
    constexpr state_proof proof(const tag_declaration &tag) {
      if ((tag.version_one && tag.state == 0) || tag.state_requires_observation) return state_proof::observed_nonzero;
      return tag.state != UINT32_MAX ? state_proof::declared : state_proof::unavailable;
    }
    // ValidUntilPresent (and a 1.x tag) until Present, ValidUntilEvaluate
    // until its evaluation, OnlyValidNow inside the call that supplied it
    // only (tag_call); anything else, and an unsupported tag, is unsupported.
    constexpr lifetime valid_until(const tag_declaration &tag, copy_point at) {
      if (!tag.supported) return lifetime::unsupported;
      if (tag.version_one || tag.lifecycle == valid_until_present) return lifetime::until_present;
      if (tag.lifecycle == valid_until_evaluate) return lifetime::until_evaluation;
      return tag.lifecycle == only_valid_now && at == copy_point::tag_call ? lifetime::at_call : lifetime::unsupported;
    }
    constexpr contract map(const tag_declaration &tag, copy_point at) {
      return {tag.state, proof(tag), valid_until(tag, at)};
    }
    // Where a 2.x tag call copies a tag it supplies with no later evaluate of
    // its own in view: a frame-generation input (DLSS-G reads it at its
    // frame's Present), or a UI input (copied only at its tag call or, here,
    // its Present). Owner ruling R2, the matrix's S5, S6, S11b, S11c and U2.
    // Resource::state "needs to be correct when tagged resources are actually
    // used" (sl_core_types.h): a ValidUntilPresent declaration describes the
    // state at its frame's Present, not at the call. So a ValidUntilPresent
    // tag is copied inside the call only when the call's command list
    // records its state there (recorded) and that state does not let the
    // list write it (writing: a write state or layout, after which the frame
    // may write it again before that Present, whose content the declaration
    // promises; the matrix's S11), else on the Present queue right before its
    // frame's Present (present); its command list may be null (sl_core_api.h:
    // only ValidUntilPresent tags may come without one). An OnlyValidNow tag
    // is copied inside the call, which needs a command list. A
    // ValidUntilEvaluate tag is copied inside the call only from a recorded
    // state, a write state included, since its evaluate copies it again;
    // without one its declaration describes its evaluate, which copies it
    // (none).
    enum class tag_call_copy : std::uint8_t { at_call, present, none };
    constexpr tag_call_copy deferred_copy(std::uint32_t lifecycle, bool command_list, bool recorded, bool writing) {
      if (lifecycle == valid_until_present)
        return command_list && recorded && !writing ? tag_call_copy::at_call : tag_call_copy::present;
      if (!command_list || (lifecycle != only_valid_now && !recorded)) return tag_call_copy::none;
      return tag_call_copy::at_call;
    }
    // The proof behind a depth tag's declaration where deferred_copy copies
    // it inside its frame-generation tag call (at_call). A ValidUntilPresent or
    // ValidUntilEvaluate declaration describes the tag's later use (its
    // frame's Present, its evaluate), not the call, so the state the call's
    // list records proves that copy, and a declaration that disagrees with it
    // does not refuse it (declared_for_use; the matrix's S11). An OnlyValidNow
    // declaration describes the call itself, and the copies on the Present
    // queue and at an evaluate are made at the use the declaration describes:
    // each keeps its proof, which an observation must match (S11a).
    constexpr state_proof tag_call_proof(std::uint32_t lifecycle, state_proof proof) {
      return proof == state_proof::declared && (lifecycle == valid_until_present || lifecycle == valid_until_evaluate) ?
        state_proof::declared_for_use : proof;
    }
    // Owner ruling R7: a null untag (a null Resource or native) of a viewport
    // cancels that viewport's copy deferred to its Present when it names the
    // deferred tag's frame, and whenever the two frames cannot be told apart
    // (the untag names no frame token, slSetTag, or the deferred tag has no
    // frame); an untag of another frame leaves it.
    constexpr bool untag_cancels(bool untag_framed, bool tag_framed, bool same_frame) {
      return !untag_framed || !tag_framed || same_frame;
    }
    // The lifecycle match() (streamline_camera_data.cpp) judges at a copy
    // point. It admits a lifetime that lasts until Present (and a 1.x tag's
    // none), so a 2.x lifetime that holds only at this copy point (UntilEvaluate,
    // or OnlyValidNow inside its call) reads as ValidUntilPresent; any other
    // is unchanged, and match() refuses it. The published lifetime is never
    // promoted.
    constexpr std::uint32_t lifecycle_at(std::uint32_t lifecycle, copy_point at) {
      tag_declaration tag;
      tag.lifecycle = lifecycle;
      const auto held = valid_until(tag, at);
      return held == lifetime::until_evaluation || held == lifetime::at_call ? valid_until_present : lifecycle;
    }
    // The matrix's S1 and S3 within one frame: a global or frame OnlyValidNow
    // tag is copied inside its tag call (S1), and a later evaluate of the
    // frame it was set for refuses it for its lifetime (S3, match():
    // not-valid-until-present). Such a refusal is no newer attempt at that
    // frame's depth: the evaluate offers only the tag its tag call already
    // copied, or refused, so that call's outcome stays the frame's, and the
    // evaluate keeps it rather than beginning an evaluation without a copy.
    // same_frame: the evaluate is of the frame the tag was set for (it names
    // that frame, or no Present came since the tag call: the capture's rule
    // for a later use of a tag, source_lifetime_current). A tag of an earlier
    // frame, one refused for anything else, a 1.x tag (it holds until
    // Present) and a local input (copied inside its own evaluate, tag_call
    // here) leave the evaluate an attempt of its own. History: docs/reshade-sbs.md
    // (Streamline depth sources) and the README's C5 paragraph.
    constexpr bool keeps_tag_call_copy(const tag_declaration &tag, copy_point at, bool same_frame) {
      return same_frame && at == copy_point::evaluate && tag.supported && !tag.version_one && tag.lifecycle == only_valid_now;
    }
  }

  // NGX (nvsdk_ngx_defs.h): an evaluation names its inputs but no state.
  namespace ngx {
    // D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE. NVIDIA's DLSS Programming
    // Guide, section 3.4 "Resource States": every D3D12 input must be in this
    // state at the evaluation call, and DLSS always restores it afterwards.
    inline constexpr std::uint32_t evaluation_input_state = 0x40;
    // NVSDK_NGX_Feature_SuperSampling (nvsdk_ngx_defs.h), the feature that
    // guide covers.
    inline constexpr std::uint32_t super_sampling = 1;
    // An accepted Super Resolution evaluation's input: the SDK contract fixes
    // its state at this boundary (a state observed on the same recording
    // still takes precedence), and it holds for the evaluation.
    inline constexpr contract evaluation_input{evaluation_input_state, state_proof::sdk_contract, lifetime::until_evaluation};
    // An input whose guide states no state at this call: only a state
    // observed on the evaluation's own recording proves it, for the
    // evaluation.
    inline constexpr contract observed_input{UINT32_MAX, state_proof::observed_nonzero, lifetime::until_evaluation};
    // DLSS-G's own inputs (frame_generation_inputs.h): no API states their
    // state at this call (observed_input).
    inline constexpr const contract &frame_generation_input = observed_input;
    // A Depth input by its evaluation's feature: Super Resolution's SDK
    // contract. No other feature's guide states one: the DLSS-RR Integration
    // Guide, an addendum to the DLSS guide, names no resource state and
    // extends only its evaluation call (section 5.4, not 3.4), and
    // nvsdk_ngx_helpers_dlssd_d3d.h sets none. So Ray Reconstruction (13) and
    // any other feature take observed_input (the matrix's N1b).
    constexpr const contract &evaluation_input_of(std::uint32_t feature) {
      return feature == super_sampling ? evaluation_input : observed_input;
    }
  }

  // Generic: the depth-stencil copy depth_addon.cpp records at the end of the
  // frame (depth_capture::record_preserved), held until Present. The end of
  // the frame names no state: the source's last proven state on the copy's
  // queue supplies it (the matrix's D10), for example a depth clear's
  // DEPTH_WRITE (its D3D12 contract, MS Learn D3D12_RESOURCE_STATES: the
  // state ClearDepthStencilView needs when its flags clear the subresource)
  // or the state its creation declared (D10b). preserved() is a copy whose
  // caller declares the source's state (DEPTH_WRITE is depth_write).
  namespace generic {
    inline constexpr std::uint32_t depth_write = 0x10; // D3D12_RESOURCE_STATE_DEPTH_WRITE
    inline constexpr contract end_of_frame{UINT32_MAX, state_proof::last_proven, lifetime::until_present};
    constexpr contract preserved(std::uint32_t declared_state) {
      return {declared_state, state_proof::declared, lifetime::until_present};
    }
  }

  // Dump 3D: the optional UI copy (game3d_ui_capture.cpp) of a Streamline tag
  // or an NGX input, made inside the call that supplied it. Its proof is
  // stricter than the live readings': a state is declared only when it is
  // neither 0 nor UINT32_MAX (game3d_diagnostic_metadata.cpp), so a 0 needs
  // an observed state for every Streamline version (a live 2.x tag's 0 is
  // COMMON), and so does an NGX input, which names none (a live one has its
  // SDK contract). A 2.x tag keeps its lifetime at its tag call; a 1.x tag
  // and an NGX input are copied at the call.
  namespace dump {
    struct ui_declaration {
      bool streamline_v2{};
      std::uint32_t lifecycle{streamline::no_lifecycle};
      bool state_declared{};
      std::uint32_t state{UINT32_MAX};
    };
    constexpr contract ui_copy(const ui_declaration &input) {
      streamline::tag_declaration tag;
      tag.lifecycle = input.lifecycle;
      return {input.state_declared ? input.state : UINT32_MAX,
        input.state_declared ? state_proof::declared : state_proof::observed_nonzero,
        input.streamline_v2 ? streamline::valid_until(tag, streamline::copy_point::tag_call) : lifetime::at_call};
    }
  }

  // The storage family of a colour format whose views are plain casts of one
  // another: its typeless DXGI format (R11G11B10_FLOAT and A8_UNORM are their
  // own), or 0 for none. Only UNORM, FLOAT and SRGB views belong; integer,
  // SNORM and depth views are never read as colour.
  constexpr std::uint32_t color_family(std::uint32_t format) {
    switch (format) {
      case 27: case 28: case 29: return 27; // R8G8B8A8
      case 90: case 87: case 91: return 90; // B8G8R8A8
      case 92: case 88: case 93: return 92; // B8G8R8X8
      case 23: case 24: return 23;          // R10G10B10A2
      case 9: case 10: case 11: return 9;   // R16G16B16A16
      case 1: case 2: return 1;             // R32G32B32A32
      case 26: return 26;                   // R11G11B10_FLOAT
      case 60: case 61: return 60;          // R8
      case 65: return 65;                   // A8_UNORM
      case 53: case 54: case 56: return 53; // R16
      case 39: case 41: return 39;          // R32
      default: return 0;
    }
  }
  constexpr bool typeless_color(std::uint32_t format) {
    return format == 27 || format == 90 || format == 92 || format == 23 || format == 9 || format == 1 || format == 60 ||
      format == 53 || format == 39;
  }
  // A legal view format of the storage without relaxed format casting: the
  // storage's own typed format, or a typed member of a typeless storage's
  // family. (An sRGB storage read through UNORM is not.)
  constexpr bool same_color_family(std::uint32_t storage, std::uint32_t view) {
    if (typeless_color(view) || !color_family(view)) return false;
    return view == storage || (typeless_color(storage) && color_family(view) == color_family(storage));
  }
  // The typed DXGI format an auxiliary (UI) snapshot of a source allocation
  // of this format is stored and read as, 0 for none. A typed allocation as
  // is. A typeless one as declared (Streamline Resource.nativeFormat) when
  // the declaration is a view of its family (same_color_family), an sRGB
  // declaration as its UNORM copy, since UI views read the stored bits and
  // an sRGB RGB view leaves alpha alone; else, for the families whose every
  // colour view reads the same bits as UNORM, that UNORM format (R10G10B10A2,
  // R8G8B8A8, B8G8R8A8, B8G8R8X8). R16G16B16A16 (FLOAT or UNORM),
  // R32G32B32A32 and the single-channel families are never guessed, and a
  // declaration outside the family is ignored.
  constexpr std::uint32_t auxiliary_snapshot_format(std::uint32_t format, std::uint32_t declared = 0) {
    if (!typeless_color(format)) return format;
    if (declared && same_color_family(format, declared))
      return declared == 29 ? 28u : declared == 91 ? 87u : declared == 93 ? 88u : declared;
    return format == 23 ? 24u : format == 27 ? 28u : format == 90 ? 87u : format == 92 ? 88u : 0u;
  }
  static_assert(auxiliary_snapshot_format(27) == 28 && auxiliary_snapshot_format(90) == 87 &&
    auxiliary_snapshot_format(23) == 24 && auxiliary_snapshot_format(92) == 88 && !auxiliary_snapshot_format(9) &&
    !auxiliary_snapshot_format(1) && !auxiliary_snapshot_format(39) && !auxiliary_snapshot_format(53) &&
    !auxiliary_snapshot_format(60) && auxiliary_snapshot_format(9, 10) == 10 && auxiliary_snapshot_format(9, 11) == 11 &&
    !auxiliary_snapshot_format(9, 13) && auxiliary_snapshot_format(1, 2) == 2 && auxiliary_snapshot_format(27, 29) == 28 &&
    auxiliary_snapshot_format(90, 91) == 87 && auxiliary_snapshot_format(92, 93) == 88 &&
    auxiliary_snapshot_format(23, 28) == 24 && auxiliary_snapshot_format(39, 41) == 41 &&
    !auxiliary_snapshot_format(39, 40) && auxiliary_snapshot_format(53, 54) == 54 &&
    auxiliary_snapshot_format(53, 56) == 56 && !auxiliary_snapshot_format(53, 55) &&
    auxiliary_snapshot_format(60, 61) == 61 && !auxiliary_snapshot_format(60, 65) &&
    auxiliary_snapshot_format(28, 87) == 28 && auxiliary_snapshot_format(65) == 65);
}

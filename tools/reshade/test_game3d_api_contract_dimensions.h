// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Dimensions of the Game 3D API contract matrix and their legal values. Each
// value is something Streamline, D3D12 or NGX permits a game to do, never a
// value chosen because a game was seen doing it. The oracle
// (test_game3d_api_contract_oracle.h) maps one assignment to its required
// outcome; test_game3d_api_contract_known_gaps.h lists where the add-on does
// not meet it yet.
//
// Independence: these headers may include only <d3d12.h>, <cstdint>, each
// other and generated SDK constant tables, never an add-on header
// (reshade_contract_oracle_independence enforces it).
//
// Citations: SLT = include/sl_core_types.h and SLA = include/sl_core_api.h of
// NVIDIA Streamline v2.14.1, other sl_*.h files from the same release, and an
// older release's header where named (the ABI survey keeps every release under
// cmake-build-relwithdebinfo/sl-abi-survey). nvsdk_ngx_*.h are the pinned NGX
// SDK headers and dlssg-guide.txt the DLSS-FG Programming Guide's text
// (cmake-build-relwithdebinfo/ngx-dlssg-contract-20260921). d3d12.h is the
// MinGW UCRT64 header; "MS Learn" is Microsoft's D3D12 documentation;
// "Enhanced Barriers" is the DirectX-Specs D3D12 Enhanced Barriers
// specification.
#include <cstdint>
#include <d3d12.h>

namespace game3d_contract {
  // Streamline.
  struct sl_version {
    std::uint32_t major {}, minor {}, patch {};
  };

  constexpr bool operator<(const sl_version &a, const sl_version &b) {
    return a.major != b.major ? a.major < b.major : a.minor != b.minor ? a.minor < b.minor :
                                                                         a.patch < b.patch;
  }

  constexpr bool operator==(const sl_version &a, const sl_version &b) {
    return a.major == b.major && a.minor == b.minor && a.patch == b.patch;
  }

  // The newest surveyed release (survey REPORT.md). The generated survey table
  // holds every surveyed release's own declarations; a newer 2.x release and an
  // unknown major version are outside the survey.
  inline constexpr sl_version newest_surveyed {2, 14, 1};
  // slSetTagForFrame first appears in 2.7.30 (SLA:136-152; absent through 2.7.2).
  inline constexpr sl_version first_frame_tag {2, 7, 30};

  // SLA:136-173 (slSetTag, slSetTagForFrame); SLA:250-262 (tags passed as
  // evaluate inputs, as an array or a chain of BaseStructures).
  enum class tag_entry {
    set_tag,
    set_tag_for_frame,
    local_array,
    local_chain_not_last
  };

  // SLT:397-405 ResourceLifecycle. Values above 2 are not declared.
  enum class lifecycle : std::uint32_t {
    only_valid_now,
    valid_until_present,
    valid_until_evaluate,
    undeclared
  };

  // SLA:145, 162: the command buffer on a tag call.
  enum class tag_commands {
    game_list,
    none
  };

  // Where the copy is attempted (SLT:400-404 bound which points are legal).
  enum class copy_point {
    tag_call,
    same_frame_evaluate,
    after_present
  };

  // Generated (frame-generation) Presents of earlier frames between the tag
  // and the copy point.
  inline constexpr std::uint32_t generated_presents[] {0, 1, 3};

  // SLA:143 (null removes a tag); SLT:280-292 (ResourceType); SLT:331-336
  // (Resource header and mandatory fields). Header versions 2 and 7 are the
  // struct_form dimension of the Resource.
  enum class resource_form {
    native,
    null_resource,
    null_native,
    zero_header,
    type_buffer,  // ResourceType::eBuffer (1)
    type_unknown  // ResourceType::eUnknown (8), castable to IUnknown only
  };

  // sl_consts.h:143-169 Extent; an extent is null when !(width && height).
  enum class extent_form {
    null,
    whole,
    padded,
    offset,
    one_dimension_zero,
    outside_allocation,
    changing,  // A new extent every frame inside the allocation (dynamic resolution).
    declared_size_differs  // Resource.width/height differ from the allocation; extent valid.
  };

  // SLT:437-475 PrecisionInfo chained on a tag.
  enum class precision_form {
    none,
    scale_bias,  // Formula 1 with a nonzero scale.
    zero_scale,  // Formula 1 with scale 0.
    unknown_formula  // Formula 2 or above.
  };

  // No limit is stated for numTags (SLA:143-173).
  inline constexpr std::uint32_t tags_per_call[] {1, 32, 33, 73};

  // sl_struct.h:43-106: versions are append-only; optional data is chained.
  enum class struct_form {
    first,
    newest_known,
    above_known,  // Newest known + 1.
    unknown_chained  // A chained struct with an unknown GUID.
  };

  // SLT:614-621 ViewportHandle; any number of viewports is legal.
  enum class viewports {
    one,
    two_active,
    five_rotating
  };

  // sl_dlss_g.h:34-40 DLSSGMode. Each release's header declares its own values
  // (survey tables: eOff and eOn through 2.2.0, eAuto from 2.2.1, eDynamic from
  // 2.11.1); 4 is never declared.
  enum class fg_mode : std::uint32_t {
    off,
    on,
    automatic,
    dynamic,
    undeclared
  };

  // sl_dlss_g.h:75-79 numFramesToGenerate: 1 for 2x, 3 for 4x, 0 generates
  // nothing; above 1 only where that release's header documents multi frame
  // generation (from 2.7.2; before, "Must be 1"). The default comes first.
  inline constexpr std::uint32_t frames_generated[] {1, 0, 3, 5};

  // sl_result.h:33-74: eOk = 0, 38 eError* values, eWarnOutOfVRAM = 39.
  enum class sl_result {
    ok,
    warn_out_of_vram,
    error
  };

  enum class untag_pattern {
    none,
    null_after_each,  // A null untag after every tag, in its frame through its entry (Assassin's Creed Shadows).
    before_tag
  };

  enum class render_pass {
    closed,
    open
  };

  // The meaning a tag value has in that release's own header (the generated
  // survey table supplies it).
  enum class buffer_meaning {
    depth,
    high_resolution_depth,
    linear_depth,
    hudless_color,
    ui_color_and_alpha,
    ui_alpha,
    ui_hint,  // 1.x value 23.
    backbuffer,
    scaling_input,
    scaling_output,
    other,  // Declared with another meaning (68 on 2.12+ is ResponsivityMask).
    undeclared  // Not declared by that release's header.
  };

  // A raw buffer-type value read against the release the game's interposer
  // identifies. The surveyed releases' headers are
  // test_sl_survey_tables.generated.h.
  struct buffer_case {
    sl_version version {newest_surveyed};
    std::uint32_t raw {};
  };

  // The structure ABI a release's headers define (survey REPORT.md): a 1.x
  // Resource without a state word (1.0.0-1.0.2) or with one (1.0.3-1.1.1),
  // or the typed, versioned 2.x structures (sl_struct.h).
  enum class sl_abi {
    none,
    legacy_no_state,
    legacy,
    typed
  };

  // D3D12.
  enum class list_type : std::uint32_t {
    direct = D3D12_COMMAND_LIST_TYPE_DIRECT,
    bundle = D3D12_COMMAND_LIST_TYPE_BUNDLE,
    compute = D3D12_COMMAND_LIST_TYPE_COMPUTE,
    copy = D3D12_COMMAND_LIST_TYPE_COPY
  };

  // The latest barrier naming subresource 0 (or all) of the source on the
  // recording that carries the copy (d3d12.h:1792-1799, 19273-19371).
  enum class evidence {
    none,
    transition,  // A plain transition into observed.
    end_without_begin,  // END_ONLY whose begin is on an earlier list.
    split_open,  // BEGIN_ONLY into observed, not yet ended.
    end_mismatched,  // END_ONLY whose StateAfter differs from its begin's on this recording.
    begin_and_end,  // BEGIN_ONLY | END_ONLY together (no defined flag).
    alias_named,  // An aliasing barrier naming the source.
    alias_wildcard,  // An aliasing barrier with NULL resources.
    enhanced,  // An enhanced texture barrier into layout with access.
    enhanced_split_open,  // SyncAfter SPLIT, not yet ended.
    mixed_split,  // A legacy barrier inside an enhanced split, or the reverse.
    earlier_list_enhanced  // Nothing here; an enhanced barrier set the layout on an earlier list.
  };

  enum class subresource_barrier {
    zero,
    all,
    other_mip_only,
    stencil_plane_only
  };

  // depth_view_format.h's families: storage (typeless) and the depth or
  // colour format a game binds.
  enum class depth_format {
    d32,  // D32_FLOAT / R32_TYPELESS
    d32s8,  // D32_FLOAT_S8X24_UINT / R32G8X24_TYPELESS
    d24s8,  // D24_UNORM_S8_UINT / R24G8_TYPELESS
    d16,  // D16_UNORM / R16_TYPELESS
    r32_float,  // Linear distance or a colour-format depth copy.
    r16_float
  };

  // A depth input's DXGI formats: the allocation's format, the typed format
  // Streamline's optional Resource.nativeFormat declares (DXGI_FORMAT_UNKNOWN
  // when absent; SLT:331-389), and whether the input holds hardware depth (a
  // depth tag or NGX depth) rather than linear depth.
  struct format_case {
    std::uint32_t allocation {}, declared {};
    bool device_depth {true};
  };

  enum class resource_shape {
    texture2d,
    texture2d_mips,
    msaa,
    array,
    texture3d
  };

  inline constexpr std::uint32_t texture_read_states = D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE |
                                                       D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
  inline constexpr std::uint32_t texture_write_states[] {D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RESOLVE_DEST};
  inline constexpr std::uint32_t shader_read_states = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  inline constexpr std::uint32_t allocation_flag_bits = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE |
                                                        D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

  // One write state alone (MS Learn, Read/write resource state restrictions:
  // "At most, only one read/write bit can be set ... If a write bit is set,
  // then no read-only bit may be set").
  constexpr bool single_write_state(std::uint32_t state) {
    for (const auto write : texture_write_states) {
      if (state == write) {
        return true;
      }
    }
    return false;
  }

  // Any nonempty combination of read-only texture states, or GENERIC_READ
  // itself (a fixed OR of read bits, MS Learn D3D12_RESOURCE_STATES). Its
  // VERTEX_AND_CONSTANT_BUFFER, INDEX_BUFFER and INDIRECT_ARGUMENT bits are
  // buffer states: the D3D12 runtime refuses a texture barrier into any other
  // state holding one, GENERIC_READ with DEPTH_READ or RESOLVE_SOURCE
  // included (ID3D12GraphicsCommandList::Close returns E_INVALIDARG; the
  // device layer, test_game3d_api_contract_device.cpp, found it).
  constexpr bool read_only_texture_state(std::uint32_t state) {
    return state == D3D12_RESOURCE_STATE_GENERIC_READ || (state && !(state & ~texture_read_states));
  }

  // The legacy states a texture can be in on a direct list: COMMON, one write
  // state, or a read-only combination (d3d12.h:1762-1788).
  constexpr bool texture_state(std::uint32_t state) {
    return state == D3D12_RESOURCE_STATE_COMMON || single_write_state(state) || read_only_texture_state(state);
  }

  // Visits each legacy state a texture can be in (the dimension's legal
  // values): COMMON, each write state alone, every read-only combination and
  // GENERIC_READ.
  template<class Visit>
  constexpr void each_texture_state(Visit &&visit) {
    visit(std::uint32_t(D3D12_RESOURCE_STATE_COMMON));
    for (const auto write : texture_write_states) {
      visit(write);
    }
    constexpr std::uint32_t read[] {D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
    for (std::uint32_t bits = 1; bits != 0x20; ++bits) {
      std::uint32_t state {};
      for (unsigned bit = 0; bit != sizeof(read) / sizeof(read[0]); ++bit) {
        if (bits & (1u << bit)) {
          state |= read[bit];
        }
      }
      visit(state);
    }
    visit(std::uint32_t(D3D12_RESOURCE_STATE_GENERIC_READ));
  }

  // D3D12_RESOURCE_FLAGS (MS Learn): ALLOW_DEPTH_STENCIL "Can't be used with"
  // ALLOW_RENDER_TARGET, ALLOW_UNORDERED_ACCESS or ALLOW_SIMULTANEOUS_ACCESS;
  // DENY_SHADER_RESOURCE "Must be used with" ALLOW_DEPTH_STENCIL.
  constexpr bool legal_flags(std::uint32_t flags) {
    if (flags & ~allocation_flag_bits) {
      return false;
    }
    if ((flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) && (flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS))) {
      return false;
    }
    return !(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) || (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
  }

  constexpr bool depth_stencil(std::uint32_t flags) {
    return (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
  }

  // D3D12_RESOURCE_FLAGS (MS Learn): ALLOW_RENDER_TARGET, ALLOW_DEPTH_STENCIL
  // and ALLOW_UNORDERED_ACCESS each "enables the resource to transition into"
  // its state; DENY_SHADER_RESOURCE "disables the resource from transitioning
  // into" NON_PIXEL_SHADER_RESOURCE or PIXEL_SHADER_RESOURCE.
  constexpr bool state_allowed(std::uint32_t state, std::uint32_t flags) {
    if (!texture_state(state)) {
      return false;
    }
    if ((state & D3D12_RESOURCE_STATE_RENDER_TARGET) && !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
      return false;
    }
    if ((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) && !(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
      return false;
    }
    if ((state & (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE)) && !depth_stencil(flags)) {
      return false;
    }
    return !(state & shader_read_states) || !(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
  }

  // Enhanced Barriers, Command Queue Layout Compatibility: the base layouts
  // through SHADING_RATE_SOURCE and the DIRECT_QUEUE_* layouts.
  constexpr bool direct_queue_layout(std::uint32_t layout) {
    return layout <= D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE ||
           (layout >= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST);
  }

  // Enhanced Barriers, Command Queue Layout Compatibility (COMPUTE).
  constexpr bool compute_queue_layout(std::uint32_t layout) {
    switch (layout) {
      case D3D12_BARRIER_LAYOUT_COMMON:
      case D3D12_BARRIER_LAYOUT_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_COPY_DEST:
        return true;
      default:
        return layout >= D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON && layout <= D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST;
    }
  }

  // One access the Enhanced Barriers Layout Access Compatibility table permits
  // in each layout (any one serves the matrix); NO_ACCESS is compatible with
  // every layout and is a separate dimension value.
  constexpr std::uint32_t compatible_access(std::uint32_t layout) {
    switch (layout) {
      case D3D12_BARRIER_LAYOUT_COMMON:
      case D3D12_BARRIER_LAYOUT_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE:
        return D3D12_BARRIER_ACCESS_COPY_SOURCE;
      case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
        return D3D12_BARRIER_ACCESS_RENDER_TARGET;
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
        return D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
        return D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
        return D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
        return D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
      case D3D12_BARRIER_LAYOUT_COPY_DEST:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST:
      case D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON:
        return D3D12_BARRIER_ACCESS_COPY_DEST;
      case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
        return D3D12_BARRIER_ACCESS_RESOLVE_SOURCE;
      case D3D12_BARRIER_LAYOUT_RESOLVE_DEST:
        return D3D12_BARRIER_ACCESS_RESOLVE_DEST;
      case D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE:
        return D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE;
      case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ:
        return D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ;
      case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE:
        return D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE;
      case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ:
        return D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ;
      case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE:
        return D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE;
      case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ:
        return D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ;
      case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE:
        return D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;
      default:  // UNDEFINED: a one-sided UNDEFINED requires NO_ACCESS.
        return D3D12_BARRIER_ACCESS_NO_ACCESS;
    }
  }

  // The resource flags a layout needs, as for the equivalent legacy states
  // (D3D12_RESOURCE_FLAGS). SHADING_RATE_SOURCE belongs to a variable-rate
  // shading image (MS Learn D3D12_RESOURCE_STATES), never to a depth or UI
  // texture.
  constexpr bool layout_allowed(std::uint32_t layout, std::uint32_t flags) {
    switch (layout) {
      case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
        return (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0;
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
        return (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
        return depth_stencil(flags);
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
        return !(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
      case D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE:
        return false;
      default:
        return layout <= D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON || layout == std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED);
    }
  }

  // Legacy states a COMPUTE list may record: no DEPTH_*, PIXEL_SHADER_RESOURCE,
  // RENDER_TARGET or RESOLVE_* (the legacy counterparts of the COMPUTE layouts
  // above, which hold no RESOLVE layout; MS Learn, Multi-engine
  // synchronization: ResolveSubresource is not a compute-list method).
  constexpr bool compute_list_state(std::uint32_t state) {
    return !(state & (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_RESOLVE_SOURCE | D3D12_RESOURCE_STATE_RESOLVE_DEST));
  }

  // Who states the resource's state, and for which point.
  enum class claim {
    none,  // The adapter claims nothing; only a recorded state proves it.
    sl_declared,  // Streamline Resource.state (SLT:357).
    sl_undeclared,  // Streamline Resource.state left at UINT_MAX.
    ngx_contract  // NGX's documented input state, NON_PIXEL_SHADER_RESOURCE.
  };

  enum class claim_scope {
    at_use,  // The copy is at the point the claim describes (SLT:357).
    before_use  // The claim describes a later use (a lifecycle 1 or 2 tag copied at its call).
  };

  // The pipeline input the copy feeds. A UI input is a colour buffer, so it
  // never has a depth-stencil allocation.
  enum class consumer {
    depth,
    ui
  };

  // The source allocation and the evidence on the recording that carries the
  // copy. The barrier names subresource 0 alone (zero), every subresource
  // (all), or only another mip or the stencil plane, which leaves subresource
  // 0 unnamed. format changes no outcome here; the device layer checks that
  // plane 0 of every family is copied byte-exact (D9).
  struct resource_state {
    std::uint32_t flags {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    evidence seen {evidence::none};
    std::uint32_t observed {};  // Legacy StateAfter.
    std::uint32_t layout {}, access {};  // Enhanced LayoutAfter and AccessAfter.
    subresource_barrier subresource {subresource_barrier::all};
    resource_shape shape {resource_shape::texture2d};
    depth_format format {depth_format::d32};
  };

  struct state_case {
    resource_state resource;
    consumer role {consumer::depth};
    claim source {claim::none};
    std::uint32_t declared {};  // sl_declared only.
    claim_scope scope {claim_scope::at_use};
    list_type list {list_type::direct};
  };

  struct sl_case {
    sl_version version {newest_surveyed};
    tag_entry entry {tag_entry::set_tag};
    lifecycle life {lifecycle::valid_until_present};
    tag_commands commands {tag_commands::game_list};
    copy_point at {copy_point::same_frame_evaluate};
    std::uint32_t generated_presents {};
    resource_form resource {resource_form::native};
    buffer_meaning meaning {buffer_meaning::depth};
    extent_form extent {extent_form::null};
    precision_form precision {precision_form::none};
    std::uint32_t tags_in_call {1};
    struct_form tag_header {struct_form::newest_known}, resource_header {struct_form::newest_known},
      constants {struct_form::newest_known}, viewport_handle {struct_form::newest_known},
      frame_generation_options {struct_form::newest_known};
    viewports views {viewports::one};
    bool fg_bound_view {};  // This viewport is the one frame generation is bound to.
    fg_mode fg {fg_mode::off};
    std::uint32_t frames_generated {1};
    sl_result tag_result {sl_result::ok}, evaluate_result {sl_result::ok}, options_result {sl_result::ok};
    untag_pattern untag {untag_pattern::none};
    list_type list {list_type::direct};
    render_pass pass {render_pass::closed};
    std::uint32_t declared_state {D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};  // UINT_MAX: undeclared.
    resource_state depth {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
  };

  // NGX (nvsdk_ngx_defs.h:190-214 feature ids).
  enum class ngx_feature : std::uint32_t {
    super_resolution = 1,
    frame_generation = 11,
    ray_reconstruction = 13,
    unknown = 60001
  };

  // DLSS.Use.HW.Depth, as RR's create helpers write it
  // (nvsdk_ngx_helpers_dlssd_d3d.h:242, 548).
  enum class ngx_hw_depth {
    absent,
    linear,  // 0
    device,  // 1
    other
  };

  enum class ngx_depth_set {
    d3d12_resource,  // SetD3d12Resource, as nvsdk_ngx_helpers_d3d.h:305 writes it.
    void_pointer,
    missing,
    null
  };

  // DLSS.Render.Subrect.Dimensions and DLSS.Input.Depth.Subrect.Base
  // (nvsdk_ngx_defs.h:803-812).
  enum class ngx_subrect {
    missing,
    zero,
    one_side,
    inside,
    outside
  };

  enum class ngx_result {
    success,
    failure  // (r & 0xFFF00000) == 0xBAD00000 (nvsdk_ngx_defs.h:99-177).
  };

  enum class ngx_lifetime {
    steady,
    release_recreate,
    nested_in_streamline
  };

  struct ngx_case {
    ngx_feature feature {ngx_feature::super_resolution};
    ngx_hw_depth hw_depth {ngx_hw_depth::absent};
    bool depth_inverted {};
    ngx_depth_set depth_set {ngx_depth_set::d3d12_resource};
    ngx_subrect subrect {ngx_subrect::missing};
    ngx_result result {ngx_result::success};
    list_type list {list_type::direct};
    ngx_lifetime lifetime {ngx_lifetime::steady};
    resource_state depth {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
  };

  // What a clear boundary clears. MS Learn D3D12_RESOURCE_STATES: DEPTH_WRITE
  // is the state for ClearDepthStencilView "when the flags (see
  // D3D12_CLEAR_FLAGS) indicate a given subresource should be cleared
  // (otherwise the subresource state doesn't matter)".
  enum class clear_target {
    depth_of_subresource_zero,  // D3D12_CLEAR_FLAG_DEPTH on a DSV of mip 0, or a render pass whose depth access is CLEAR.
    stencil_only,  // D3D12_CLEAR_FLAG_STENCIL alone, or a render pass that clears only stencil.
    other_subresource  // A DSV of another mip.
  };

  // Generic path: the depth-stencil boundary where a copy is attempted.
  enum class generic_boundary {
    clear_before_draws,
    clear_after_draws,
    render_pass_clear,
    unbind_writable_dsv,
    unbind_read_only_dsv,
    transition_away,
    end_of_frame
  };

  struct generic_case {
    generic_boundary boundary {generic_boundary::clear_before_draws};
    bool depth_written {};  // A draw with depth writes enabled since the last transition, on this recording.
    resource_state depth {D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    std::uint32_t transition_before {};  // transition_away: the barrier's StateBefore.
    clear_target cleared {clear_target::depth_of_subresource_zero};  // The clear boundaries only.
  };

  // UI inputs: tags 2, 23, 68 or 69 (by release) and 53.
  enum class ui_capture {
    tag_call,
    no_command_list,
    local
  };

  enum class ui_clear_boundary {
    none,
    legacy_render_target,
    enhanced_render_target,
    simultaneous_common
  };

  struct ui_case {
    sl_version version {newest_surveyed};
    buffer_meaning meaning {buffer_meaning::ui_color_and_alpha};
    lifecycle life {lifecycle::only_valid_now};
    ui_capture how {ui_capture::tag_call};
    bool fg_on {};
    bool full_size {true};
    ui_clear_boundary clear {ui_clear_boundary::none};
  };

  // The device layer's copy points (test_game3d_api_contract_device.cpp),
  // each recorded on a real list: an adapter's copy at its call, the Generic
  // path's depth-stencil boundaries and the UI layer's clear boundary.
  enum class device_point {
    api,  // A Streamline, NGX or UI adapter's copy at its call; the state rules decide.
    clear,  // The Generic path before ClearDepthStencilView of subresource 0's depth (D11).
    end_of_frame,  // The Generic path's end-of-frame copy, on the runtime's own list (D10).
    unbind_read_only_dsv,  // The Generic path where a read-only DSV is unbound (D11c).
    ui_clear  // The UI layer before a full-frame ClearRenderTargetView (D12).
  };

  // One device cell: the source and the evidence on the copy's recording
  // (state); the state an earlier, executed list left the source in (prior),
  // which a claim must describe when the recording names nothing about
  // subresource 0; and for the UI clear boundary, its target (clear).
  struct device_case {
    state_case state;
    device_point point {device_point::api};
    std::uint32_t prior {};
    ui_clear_boundary clear {ui_clear_boundary::none};
  };

  // The generic layout a queue-specific layout specializes (Enhanced
  // Barriers, Command Queue Layout Compatibility); itself otherwise.
  constexpr std::uint32_t generic_layout(std::uint32_t layout) {
    switch (layout) {
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON:
        return D3D12_BARRIER_LAYOUT_COMMON;
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ:
        return D3D12_BARRIER_LAYOUT_GENERIC_READ;
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
        return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
        return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE:
        return D3D12_BARRIER_LAYOUT_COPY_SOURCE;
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST:
        return D3D12_BARRIER_LAYOUT_COPY_DEST;
      default:
        return layout;
    }
  }

  // Whether a legacy state describes a texture in an enhanced layout
  // (Enhanced Barriers, the layout equivalent to each D3D12_RESOURCE_STATES
  // value): DEPTH_STENCIL_READ is DEPTH_READ with any shader-resource bits,
  // SHADER_RESOURCE is shader-resource bits alone, and every other layout has
  // its one state.
  constexpr bool describes_layout(std::uint32_t state, std::uint32_t layout) {
    switch (generic_layout(layout)) {
      case D3D12_BARRIER_LAYOUT_COMMON:
        return state == D3D12_RESOURCE_STATE_COMMON;
      case D3D12_BARRIER_LAYOUT_GENERIC_READ:
        return state == D3D12_RESOURCE_STATE_GENERIC_READ;
      case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
        return state == D3D12_RESOURCE_STATE_RENDER_TARGET;
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
        return state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
        return state == D3D12_RESOURCE_STATE_DEPTH_WRITE;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
        return (state & ~shader_read_states) == D3D12_RESOURCE_STATE_DEPTH_READ;
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
        return state && !(state & ~shader_read_states);
      case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
        return state == D3D12_RESOURCE_STATE_COPY_SOURCE;
      case D3D12_BARRIER_LAYOUT_COPY_DEST:
        return state == D3D12_RESOURCE_STATE_COPY_DEST;
      case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
        return state == D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
      case D3D12_BARRIER_LAYOUT_RESOLVE_DEST:
        return state == D3D12_RESOURCE_STATE_RESOLVE_DEST;
      default:
        return false;
    }
  }

  // Legality filter. The matrix enumerates only legal combinations; a
  // separate set of deliberately illegal rows must end in a safe refusal.
  constexpr bool legal_evidence(const resource_state &value, list_type list) {
    switch (value.seen) {
      case evidence::begin_and_end:
      case evidence::end_mismatched:
      case evidence::mixed_split:
        return false;
      case evidence::transition:
      case evidence::end_without_begin:
      case evidence::split_open:
        return state_allowed(value.observed, value.flags) && (list != list_type::compute || compute_list_state(value.observed));
      case evidence::enhanced:
      case evidence::enhanced_split_open:
        // Enhanced Barriers: a one-sided UNDEFINED layout needs NO_ACCESS on
        // that side; any other layout must suit the recording's queue type.
        if (value.layout == std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED)) {
          return value.access == D3D12_BARRIER_ACCESS_NO_ACCESS;
        }
        if (!layout_allowed(value.layout, value.flags) || !(list == list_type::compute ? compute_queue_layout(value.layout) : direct_queue_layout(value.layout))) {
          return false;
        }
        return value.access == D3D12_BARRIER_ACCESS_NO_ACCESS || value.access == compatible_access(value.layout);
      default:
        return true;
    }
  }

  constexpr bool legal(const state_case &value) {
    if (!legal_flags(value.resource.flags) || (value.role == consumer::ui && depth_stencil(value.resource.flags))) {
      return false;
    }
    if (value.list != list_type::direct && value.list != list_type::compute) {
      return false;
    }
    if (value.source == claim::sl_declared && (!state_allowed(value.declared, value.resource.flags) || (value.list == list_type::compute && !compute_list_state(value.declared)))) {
      return false;
    }
    return legal_evidence(value.resource, value.list);
  }

  constexpr bool legal(const sl_case &value) {
    if (value.entry == tag_entry::set_tag_for_frame && value.version < first_frame_tag) {
      return false;
    }
    // 1.x slSetTag takes no command buffer and no lifecycle (v1.1.1 sl.h:347);
    // its state need only be correct when the tag is used (sl.h:176), so a
    // 1.x tag is valid until its evaluate. 1.x has no frame tags, evaluate
    // inputs or DLSS-G, so no generated Presents either.
    if (value.version.major == 1 && (value.entry != tag_entry::set_tag || value.life != lifecycle::valid_until_evaluate || value.commands != tag_commands::none || value.fg != fg_mode::off || value.generated_presents)) {
      return false;
    }
    // A local tag is an evaluate input: it has no tag call of its own, and
    // the evaluate's command buffer is not optional (SLA:253).
    const bool local = value.entry == tag_entry::local_array || value.entry == tag_entry::local_chain_not_last;
    if (local && (value.at == copy_point::tag_call || value.commands == tag_commands::none)) {
      return false;
    }
    const bool untags = value.resource == resource_form::null_resource || value.resource == resource_form::null_native;
    if (value.version.major != 1 && value.commands == tag_commands::none && !untags && value.life != lifecycle::valid_until_present) {
      return false;
    }
    if (value.declared_state != UINT32_MAX && !state_allowed(value.declared_state, value.depth.flags)) {
      return false;
    }
    if (value.list == list_type::compute && value.declared_state != UINT32_MAX && !compute_list_state(value.declared_state)) {
      return false;
    }
    return legal_flags(value.depth.flags) && legal_evidence(value.depth, value.list);
  }

  constexpr bool legal(const ngx_case &value) {
    return legal_flags(value.depth.flags) && (value.list == list_type::direct || value.list == list_type::compute) &&
           legal_evidence(value.depth, value.list);
  }

  constexpr bool legal(const generic_case &value) {
    if (!legal_flags(value.depth.flags) || !depth_stencil(value.depth.flags)) {
      return false;
    }
    if (value.boundary == generic_boundary::transition_away && !state_allowed(value.transition_before, value.depth.flags)) {
      return false;
    }
    const bool clears = value.boundary == generic_boundary::clear_before_draws || value.boundary == generic_boundary::clear_after_draws ||
                        value.boundary == generic_boundary::render_pass_clear;
    if (value.cleared != clear_target::depth_of_subresource_zero && !clears) {
      return false;
    }
    if (value.cleared == clear_target::other_subresource && value.depth.shape != resource_shape::texture2d_mips) {
      return false;
    }
    // MS Learn, Split Barriers: while a split barrier is pending "the
    // (sub)resource cannot be read or written by the GPU", so no depth clear
    // and no other transition may meet it.
    const bool pending = value.depth.seen == evidence::split_open || value.depth.seen == evidence::enhanced_split_open;
    if (pending && ((clears && value.cleared == clear_target::depth_of_subresource_zero) || value.boundary == generic_boundary::transition_away)) {
      return false;
    }
    return legal_evidence(value.depth, list_type::direct);
  }

  // A device cell is legal when its state case is, and when what the game
  // did before and on the recording is legal D3D12 that its claims describe
  // truthfully.
  constexpr bool legal(const device_case &value) {
    const auto &state = value.state;
    const auto &source = state.resource;
    const auto flags = source.flags;
    const bool compute = state.list == list_type::compute;
    const bool simultaneous = (flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;
    if (!legal(state) || (value.point == device_point::ui_clear) != (value.clear != ui_clear_boundary::none)) {
      return false;
    }
    // The state the source starts the recording in: one the allocation and
    // the recording's list type allow. A simultaneous-access texture decays to
    // COMMON when the earlier list's ExecuteCommandLists completes (MS Learn,
    // Implicit state transitions), so no split begun there spans the boundary.
    if (!state_allowed(value.prior, flags) || (compute && !compute_list_state(value.prior)) || (simultaneous && (value.prior != D3D12_RESOURCE_STATE_COMMON || source.seen == evidence::end_without_begin))) {
      return false;
    }
    if (value.point != device_point::api) {
      // The boundaries copy on a direct list, with no claim, from a source
      // whose recording names nothing (the runtime's own list at the end of
      // frame holds no game barrier).
      if (compute || state.source != claim::none || source.seen != evidence::none || source.shape != resource_shape::texture2d) {
        return false;
      }
      switch (value.point) {
        case device_point::clear:
          // MS Learn D3D12_RESOURCE_STATES, DEPTH_WRITE: the state for
          // ClearDepthStencilView of a subresource it clears.
          return state.role == consumer::depth && depth_stencil(flags) && value.prior == D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case device_point::end_of_frame:
          return state.role == consumer::depth && depth_stencil(flags);
        case device_point::unbind_read_only_dsv:
          // MS Learn D3D12_RESOURCE_STATES, DEPTH_READ: a read-only DSV's state.
          return state.role == consumer::depth && depth_stencil(flags) && (value.prior & D3D12_RESOURCE_STATE_DEPTH_READ);
        default:
          // ClearRenderTargetView needs RENDER_TARGET (MS Learn
          // D3D12_RESOURCE_STATES), the RENDER_TARGET layout, or a
          // simultaneous-access target in COMMON that the clear promotes.
          if (state.role != consumer::ui || !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
            return false;
          }
          switch (value.clear) {
            case ui_clear_boundary::legacy_render_target:
              return !simultaneous && value.prior == D3D12_RESOURCE_STATE_RENDER_TARGET;
            case ui_clear_boundary::enhanced_render_target:
              return !simultaneous && value.prior == D3D12_RESOURCE_STATE_COMMON;
            default:
              return simultaneous;
          }
      }
    }
    // A barrier on the recording starts from prior: a legacy one leaves it
    // for another state, an enhanced one starts from COMMON (Enhanced
    // Barriers, legacy interop).
    const bool named_by_barrier = source.seen == evidence::transition || source.seen == evidence::end_without_begin ||
                                  source.seen == evidence::split_open;
    if (named_by_barrier && value.prior == source.observed) {
      return false;
    }
    if ((source.seen == evidence::enhanced || source.seen == evidence::enhanced_split_open) && value.prior != D3D12_RESOURCE_STATE_COMMON) {
      return false;
    }
    if (source.seen == evidence::earlier_list_enhanced) {
      const resource_state earlier {flags, evidence::enhanced, 0, source.layout, source.access};
      if (!legal_evidence(earlier, state.list) || source.layout == std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED)) {
        return false;
      }
      return (state.source != claim::sl_declared || describes_layout(state.declared, source.layout)) &&
             (state.source != claim::ngx_contract || describes_layout(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, source.layout));
    }
    // Where the recording names nothing about subresource 0, a claim names
    // the state it is in: SLT:357 ('needs to be correct when tagged resources
    // are actually used'); the DLSS guide's NON_PIXEL_SHADER_RESOURCE.
    const bool unnamed = source.subresource == subresource_barrier::other_mip_only || source.subresource == subresource_barrier::stencil_plane_only ||
                         source.seen == evidence::none || source.seen == evidence::alias_named || source.seen == evidence::alias_wildcard;
    if (unnamed) {
      return (state.source != claim::sl_declared || state.declared == value.prior) &&
             (state.source != claim::ngx_contract || value.prior == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    return true;
  }

  // legal(const ui_case &) reads each release's buffer-type declarations, so
  // it follows them in test_game3d_api_contract_oracle.h.
}  // namespace game3d_contract

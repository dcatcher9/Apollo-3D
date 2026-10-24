// SPDX-License-Identifier: GPL-3.0-only
// The API contract matrix's device layer (L2): the capture owner's copies
// recorded on real D3D12 lists of the first hardware adapter, judged by the
// Game 3D API contract oracle (test_game3d_api_contract_oracle.h).
//
// Each cell takes a fresh source and puts it in its prior state on an
// earlier, executed list. The copy's recording then holds the game's barrier
// (legacy, split, aliasing or enhanced; on subresource 0, all subresources,
// another mip or the stencil plane; a split it ends began on an earlier list
// of the same ExecuteCommandLists call, the only place D3D12 completes one),
// then the copy at its point:
// - an adapter's copy at its call: capture::record for a depth input with
//   the proof its claim selects, record_local_texture with the UI policy
//   for a UI tag;
// - a Generic boundary: record_preserved with the DEPTH_WRITE depth_addon.cpp
//   declares at a clear, a DSV unbind and the end of frame (on a recording
//   that names nothing about the source, as the runtime's own list does);
// - the UI clear boundary: the three barriers game3d_ui_layer.cpp
//   record_into records through ReShade's D3D12 backend. The layer itself
//   runs only through ReShade, in the runtime layer's UI family.
// Every cell is executed alone, so a D3D12 debug-layer error names it.
//
// A cell meets the oracle by its kind, refusal class (depth_refusal.h's class
// of the capture owner's stage) and copy basis: the state the copy leaves
// and restores, which must be the state the test put the source in. The bytes
// of subresource 0's plane 0 are read back through the capture owner's
// consumer copies: an admitted depth copy is repeated by record_preserved from
// the same basis, whose snapshot copy_preserved reads; a UI snapshot is read
// by copy_local_texture. A cell a known-gap row says copies from a state
// nothing proved is recorded but never executed.
//
// Exit 77 (skipped): no hardware adapter, or no D3D12 debug layer (Windows
// Graphics Tools) to prove the recorded barriers. Without the debug layer
// every other check still runs and a failure still fails the test.
#define SUNSHINE_DIAGNOSTIC_TEXTURE_FIXTURE_ONLY
#include "test_diagnostic_texture_capture.cpp"

#include "depth_refusal.h"
#include "test_game3d_api_contract_tally.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>

namespace {
  namespace matrix = game3d_contract;
  namespace gaps = matrix::known_gaps;
  namespace reasons = sunshine_depth_refusal;
  using namespace matrix::judging;
  using matrix::claim;
  using matrix::consumer;
  using matrix::device_point;
  using matrix::evidence;
  using matrix::list_type;
  using matrix::resource_shape;
  using matrix::subresource_barrier;
  using matrix::ui_clear_boundary;

  // The oracle's REFUSE classes and depth_refusal.h's share their order.
  static_assert(unsigned(reasons::reason_class::state_unprovable) == unsigned(matrix::refusal::state_unprovable) && unsigned(reasons::reason_class::no_copy_point) == unsigned(matrix::refusal::no_copy_point) && unsigned(reasons::reason_class::malformed) == unsigned(matrix::refusal::malformed) && unsigned(reasons::reason_class::fg_scope) == unsigned(matrix::refusal::fg_scope));

  constexpr std::uint32_t ds = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, deny = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
                          rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          simultaneous = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
  constexpr std::uint32_t common = D3D12_RESOURCE_STATE_COMMON, copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE,
                          copy_dest = D3D12_RESOURCE_STATE_COPY_DEST, nps = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          ps = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, depth_read = D3D12_RESOURCE_STATE_DEPTH_READ,
                          depth_write = D3D12_RESOURCE_STATE_DEPTH_WRITE, generic_read = D3D12_RESOURCE_STATE_GENERIC_READ,
                          render_target = D3D12_RESOURCE_STATE_RENDER_TARGET, unordered = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  constexpr unsigned source_width = 16, source_height = 8;

  // One cell: the oracle's dimensions and the allocation's DXGI format.
  struct cell {
    matrix::device_case value;
    DXGI_FORMAT format {};
    const char *block {};
  };

  // The texel values a source's subresource 0 holds, which its plane 0 keeps
  // exactly: depth in [0, 1), finite halves, 24-bit depth (the X8 byte is
  // not data), any bits otherwise.
  enum class texel_kind {
    float32,
    unorm24,
    word16,
    half16,
    bytes4,
    half64
  };

  texel_kind kind_of(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R32_TYPELESS:
      case DXGI_FORMAT_D32_FLOAT:
      case DXGI_FORMAT_R32_FLOAT:
      case DXGI_FORMAT_R32G8X24_TYPELESS:
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return texel_kind::float32;
      case DXGI_FORMAT_R24G8_TYPELESS:
      case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return texel_kind::unorm24;
      case DXGI_FORMAT_R16_TYPELESS:
      case DXGI_FORMAT_D16_UNORM:
        return texel_kind::word16;
      case DXGI_FORMAT_R16_FLOAT:
        return texel_kind::half16;
      case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return texel_kind::half64;
      default:
        return texel_kind::bytes4;
    }
  }

  unsigned texel_bytes(texel_kind kind) {
    return kind == texel_kind::word16 || kind == texel_kind::half16 ? 2u : kind == texel_kind::half64 ? 8u :
                                                                                                        4u;
  }

  void texel(texel_kind kind, unsigned index, unsigned char *out) {
    const auto mixed = index * 2654435761u >> 7;
    switch (kind) {
      case texel_kind::float32:
        {
          const float value = float(index % 1021 + 1) / 1024.0f;
          std::memcpy(out, &value, 4);
          break;
        }
      case texel_kind::unorm24:
        {
          const std::uint32_t value = mixed & 0xffffffu;
          std::memcpy(out, &value, 4);
          break;
        }
      case texel_kind::word16:
        {
          const auto value = std::uint16_t(mixed);
          std::memcpy(out, &value, 2);
          break;
        }
      case texel_kind::half16:
        {
          const auto value = std::uint16_t(0x3000u + mixed % 0x0c00u);
          std::memcpy(out, &value, 2);
          break;
        }
      case texel_kind::half64:
        for (unsigned channel = 0; channel != 4; ++channel) {
          const auto value = std::uint16_t(0x3000u + (mixed + channel * 977u) % 0x0c00u);
          std::memcpy(out + channel * 2, &value, 2);
        }
        break;
      default:
        std::memcpy(out, &mixed, 4);
        break;
    }
  }

  bool same_texel(texel_kind kind, const unsigned char *a, const unsigned char *b) {
    if (kind == texel_kind::unorm24) {
      std::uint32_t x {}, y {};
      std::memcpy(&x, a, 4);
      std::memcpy(&y, b, 4);
      return (x & 0xffffffu) == (y & 0xffffffu);
    }
    return std::memcmp(a, b, texel_bytes(kind)) == 0;
  }

  bool two_planes(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R32G8X24_TYPELESS:
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      case DXGI_FORMAT_R24G8_TYPELESS:
      case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return true;
      default:
        return false;
    }
  }

  // The DSV format of a depth-stencil allocation's family.
  DXGI_FORMAT depth_view_of(DXGI_FORMAT format) {
    switch (sunshine_depth_format::storage(format)) {
      case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_D32_FLOAT;
      case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
      case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
      default:
        return DXGI_FORMAT_D16_UNORM;
    }
  }

  matrix::depth_format family_of(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R32G8X24_TYPELESS:
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return matrix::depth_format::d32s8;
      case DXGI_FORMAT_R24G8_TYPELESS:
      case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return matrix::depth_format::d24s8;
      case DXGI_FORMAT_R16_TYPELESS:
      case DXGI_FORMAT_D16_UNORM:
        return matrix::depth_format::d16;
      case DXGI_FORMAT_R32_FLOAT:
        return matrix::depth_format::r32_float;
      case DXGI_FORMAT_R16_FLOAT:
        return matrix::depth_format::r16_float;
      default:
        return matrix::depth_format::d32;
    }
  }

  // The legacy state a declaration names for a copy basis: the state, or the
  // one each enhanced layout is equivalent to.
  std::uint32_t declaration_for(const matrix::copy_basis &basis) {
    if (!basis.enhanced) {
      return basis.state;
    }
    switch (matrix::generic_layout(basis.layout)) {
      case D3D12_BARRIER_LAYOUT_COMMON:
        return common;
      case D3D12_BARRIER_LAYOUT_GENERIC_READ:
        return generic_read;
      case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
        return render_target;
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
        return unordered;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
        return depth_write;
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
        return depth_read;
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
        return nps | ps;
      case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
        return copy_source;
      case D3D12_BARRIER_LAYOUT_COPY_DEST:
        return copy_dest;
      case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
        return D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
      default:
        return UINT32_MAX;
    }
  }

  // The cells.

  struct enumeration {
    std::vector<cell> cells;

    void add(const char *block, DXGI_FORMAT format, matrix::device_case value) {
      value.state.resource.format = family_of(format);
      if (matrix::legal(value)) {
        cells.push_back({value, format, block});
      }
    }

    // An adapter's copy at its call.
    void api(const char *block, DXGI_FORMAT format, matrix::resource_state source, std::uint32_t prior, claim made = claim::none, std::uint32_t declared = 0, consumer role = consumer::depth, list_type list = list_type::direct) {
      matrix::device_case value;
      value.state.resource = source;
      value.state.role = role;
      value.state.source = made;
      value.state.declared = declared;
      value.state.list = list;
      value.prior = prior;
      add(block, format, value);
    }

    void boundary(const char *block, DXGI_FORMAT format, std::uint32_t flags, device_point point, std::uint32_t prior, ui_clear_boundary clear = ui_clear_boundary::none) {
      matrix::device_case value;
      value.state.resource = {flags, evidence::none};
      value.state.role = point == device_point::ui_clear ? consumer::ui : consumer::depth;
      value.point = point;
      value.prior = prior;
      value.clear = clear;
      add(block, format, value);
    }
  };

  // A prior state a barrier into state starts from.
  std::uint32_t prior_for(std::uint32_t state, std::uint32_t flags) {
    if (flags & simultaneous) {
      return common;
    }
    return state != copy_dest ? copy_dest : copy_source;
  }

  template<class Visit>
  void each_allowed(std::uint32_t flags, Visit &&visit) {
    matrix::each_texture_state([&](std::uint32_t state) {
      if (matrix::state_allowed(state, flags)) {
        visit(state);
      }
    });
  }

  // The enhanced layouts a copy may meet on a direct list: the generic and
  // DIRECT_QUEUE_* layouts other than SHADING_RATE_SOURCE (no depth or UI
  // texture), the video layouts and UNDEFINED (whose contents are gone).
  template<class Visit>
  void each_direct_layout(Visit &&visit) {
    for (std::uint32_t layout = D3D12_BARRIER_LAYOUT_COMMON; layout <= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST; ++layout) {
      if (layout < D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE || layout >= D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON) {
        visit(layout);
      }
    }
  }

  std::vector<cell> enumerate(bool enhanced_barriers) {
    enumeration out;

    const struct {
      std::uint32_t flags;
      DXGI_FORMAT format;
    } allocations[] {
      {ds, DXGI_FORMAT_R32_TYPELESS},
      {ds | deny, DXGI_FORMAT_R32_TYPELESS},
      {0, DXGI_FORMAT_R32_FLOAT},
      {rt, DXGI_FORMAT_R32_FLOAT},
      {uav, DXGI_FORMAT_R32_FLOAT},
      {rt | uav, DXGI_FORMAT_R32_FLOAT},
      {simultaneous, DXGI_FORMAT_R32_FLOAT},
      {simultaneous | rt, DXGI_FORMAT_R32_FLOAT},
      {simultaneous | uav, DXGI_FORMAT_R32_FLOAT},
      {simultaneous | rt | uav, DXGI_FORMAT_R32_FLOAT},
    };

    // Every legal legacy state of every allocation, completed, ended (its
    // split begun on an earlier list of the same call) or begun on the
    // recording, and named only by a declaration.
    for (const auto &allocation : allocations) {
      each_allowed(allocation.flags, [&](std::uint32_t state) {
        for (const auto seen : {evidence::transition, evidence::end_without_begin, evidence::split_open}) {
          out.api("legacy state", allocation.format, {allocation.flags, seen, state}, prior_for(state, allocation.flags));
        }
        out.api("legacy state", allocation.format, {allocation.flags, evidence::none}, state, claim::sl_declared, state);
      });
    }

    // Each claim against a recorded state and alone (D0, N1, S10, S10a,
    // S11a), and a wildcard aliasing barrier, which names no state.
    for (const auto &allocation : {allocations[0], allocations[5]}) {
      each_allowed(allocation.flags, [&](std::uint32_t state) {
        const auto prior = prior_for(state, allocation.flags);
        auto other = state != copy_source ? copy_source : (allocation.flags & ds ? depth_read : nps);
        for (const auto made : {claim::sl_undeclared, claim::ngx_contract}) {
          out.api("claims", allocation.format, {allocation.flags, evidence::transition, state}, prior, made);
        }
        out.api("claims", allocation.format, {allocation.flags, evidence::transition, state}, prior, claim::sl_declared, state);
        out.api("claims", allocation.format, {allocation.flags, evidence::transition, state}, prior, claim::sl_declared, other);
        for (const auto made : {claim::none, claim::sl_undeclared, claim::ngx_contract}) {
          out.api("claims", allocation.format, {allocation.flags, evidence::none}, state, made);
        }
        out.api("claims", allocation.format, {allocation.flags, evidence::alias_wildcard}, state, claim::sl_declared, state);
      });
    }

    // Enhanced barriers on the recording: every direct-list layout with a
    // compatible access or NO_ACCESS, completed, or begun as a split with a
    // compatible access (D5, D5b), with no claim; and declarations and NGX's
    // contract against a few layouts.
    if (enhanced_barriers) {
      for (const auto &allocation : allocations) {
        if (allocation.flags & simultaneous) {
          continue;  // Left to legacy barriers here.
        }
        each_direct_layout([&](std::uint32_t layout) {
          for (const auto access : {matrix::compatible_access(layout), std::uint32_t(D3D12_BARRIER_ACCESS_NO_ACCESS)}) {
            out.api("enhanced layout", allocation.format, {allocation.flags, evidence::enhanced, 0, layout, access}, common);
          }
          // A split the runtime accepts (split_layout).
          out.api("enhanced layout", allocation.format, {allocation.flags, evidence::enhanced_split_open, 0, layout, matrix::compatible_access(layout)}, common);
        });
      }
      for (const auto &allocation : {allocations[0], allocations[5]}) {
        for (const std::uint32_t layout : {D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, D3D12_BARRIER_LAYOUT_GENERIC_READ, D3D12_BARRIER_LAYOUT_COPY_SOURCE}) {
          const matrix::resource_state source {allocation.flags, evidence::enhanced, 0, layout, matrix::compatible_access(layout)};
          for (const auto state : {nps, nps | ps, depth_read, depth_read | nps | ps, generic_read, copy_source, copy_dest}) {
            out.api("enhanced claims", allocation.format, source, common, claim::sl_declared, state);
          }
          out.api("enhanced claims", allocation.format, source, common, claim::ngx_contract);
          out.api("enhanced claims", allocation.format, source, common, claim::sl_undeclared);
        }
      }

      // A layout an enhanced barrier set on an earlier list, then nothing on
      // the recording (D6, N8).
      for (const auto &allocation : {allocations[0], allocations[2], allocations[5]}) {
        for (const std::uint32_t layout : {D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_GENERIC_READ, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE}) {
          const matrix::resource_state source {allocation.flags, evidence::earlier_list_enhanced, 0, layout, matrix::compatible_access(layout)};
          each_allowed(allocation.flags, [&](std::uint32_t state) {
            out.api("earlier enhanced layout", allocation.format, source, common, claim::sl_declared, state);
          });
          for (const auto made : {claim::none, claim::sl_undeclared, claim::ngx_contract}) {
            out.api("earlier enhanced layout", allocation.format, source, common, made);
          }
        }
      }
      for (const std::uint32_t layout : {D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE}) {
        const matrix::resource_state source {rt, evidence::earlier_list_enhanced, 0, layout, matrix::compatible_access(layout)};
        for (const auto state : {render_target, nps | ps, ps, copy_source}) {
          out.api("earlier enhanced layout", DXGI_FORMAT_R8G8B8A8_UNORM, source, common, claim::sl_declared, state, consumer::ui);
        }
      }
    }

    // Every depth family, typed and typeless, and the colour depth formats:
    // a whole texture or a mip chain, transitioned on all subresources, on
    // subresource 0 alone, or only on another mip or the stencil plane (D7,
    // D9); and the shapes no depth input names (D8).
    const struct {
      DXGI_FORMAT format;
      std::uint32_t flags;
    } formats[] {
      {DXGI_FORMAT_R32_TYPELESS, ds},
      {DXGI_FORMAT_D32_FLOAT, ds},
      {DXGI_FORMAT_R32G8X24_TYPELESS, ds},
      {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ds},
      {DXGI_FORMAT_R24G8_TYPELESS, ds},
      {DXGI_FORMAT_D24_UNORM_S8_UINT, ds},
      {DXGI_FORMAT_R16_TYPELESS, ds},
      {DXGI_FORMAT_D16_UNORM, ds},
      {DXGI_FORMAT_R32_FLOAT, 0},
      {DXGI_FORMAT_R16_FLOAT, 0},
    };

    for (const auto &allocation : formats) {
      const bool depth = allocation.flags & ds;
      const std::uint32_t depth_states[] {depth_write, depth_read, depth_read | nps | ps, nps, generic_read, copy_source};
      const std::uint32_t colour_states[] {nps, nps | ps, copy_source, generic_read};
      for (const auto shape : {resource_shape::texture2d, resource_shape::texture2d_mips}) {
        for (const auto subresource : {subresource_barrier::all, subresource_barrier::zero, subresource_barrier::other_mip_only, subresource_barrier::stencil_plane_only}) {
          if ((subresource == subresource_barrier::other_mip_only && shape != resource_shape::texture2d_mips) || (subresource == subresource_barrier::stencil_plane_only && !two_planes(allocation.format))) {
            continue;
          }
          const bool named = subresource == subresource_barrier::all || subresource == subresource_barrier::zero;
          const auto visit = [&](std::uint32_t state) {
            matrix::resource_state source {allocation.flags, evidence::transition, state};
            source.subresource = subresource;
            source.shape = shape;
            if (named) {
              out.api("formats and subresources", allocation.format, source, copy_dest);
            } else {
              // Subresource 0 stays where the earlier list left it; its
              // declaration names that state.
              const auto prior = depth ? (state != depth_write ? depth_write : depth_read) : (state != copy_dest ? copy_dest : nps);
              out.api("formats and subresources", allocation.format, source, prior, claim::sl_declared, prior);
            }
          };
          if (depth) {
            for (const auto state : depth_states) {
              visit(state);
            }
          } else {
            for (const auto state : colour_states) {
              visit(state);
            }
          }
        }
      }
    }
    {
      const auto shaped = [&](DXGI_FORMAT format, std::uint32_t flags, resource_shape shape, std::uint32_t prior, std::uint32_t state) {
        matrix::resource_state source {flags, evidence::transition, state};
        source.shape = shape;
        out.api("shapes", format, source, prior);
      };
      shaped(DXGI_FORMAT_D32_FLOAT, ds, resource_shape::msaa, depth_write, depth_read);
      shaped(DXGI_FORMAT_R32_FLOAT, rt, resource_shape::msaa, render_target, nps);
      shaped(DXGI_FORMAT_R32_TYPELESS, ds, resource_shape::array, copy_dest, depth_read);
      shaped(DXGI_FORMAT_R32_FLOAT, 0, resource_shape::array, copy_dest, nps);
      shaped(DXGI_FORMAT_R32_FLOAT, 0, resource_shape::texture3d, copy_dest, nps);
    }

    // COMPUTE lists: each state a compute list may record, completed there
    // or named by a claim, and the compute-queue layouts (S22, N7).
    for (const auto &allocation : {allocations[0], allocations[2], allocations[4]}) {
      each_allowed(allocation.flags, [&](std::uint32_t state) {
        if (!matrix::compute_list_state(state)) {
          return;
        }
        const auto prior = prior_for(state, allocation.flags);
        for (const auto made : {claim::none, claim::sl_declared, claim::ngx_contract}) {
          out.api("compute list", allocation.format, {allocation.flags, evidence::transition, state}, prior, made, state, consumer::depth, list_type::compute);
        }
        for (const auto made : {claim::sl_declared, claim::sl_undeclared, claim::ngx_contract}) {
          out.api("compute list", allocation.format, {allocation.flags, evidence::none}, state, made, state, consumer::depth, list_type::compute);
        }
        if (!(allocation.flags & ds)) {
          out.api("compute list", DXGI_FORMAT_R8G8B8A8_UNORM, {allocation.flags, evidence::transition, state}, prior, claim::sl_declared, state, consumer::ui, list_type::compute);
        }
      });
      if (enhanced_barriers) {
        for (const std::uint32_t layout : {D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE}) {
          out.api("compute list", allocation.format, {allocation.flags, evidence::enhanced, 0, layout, matrix::compatible_access(layout)}, common, claim::none, 0, consumer::depth, list_type::compute);
        }
      }
    }

    // The Generic boundaries of every depth family (D10, D11, D11c).
    for (const auto &allocation : formats) {
      if (!(allocation.flags & ds)) {
        continue;
      }
      for (const auto flags : {ds, ds | deny}) {
        if ((flags & deny) && allocation.format != DXGI_FORMAT_R32_TYPELESS) {
          continue;
        }
        out.boundary("generic boundaries", allocation.format, flags, device_point::clear, depth_write);
        for (const auto prior : {depth_write, depth_read, depth_read | nps | ps, nps | ps, generic_read, copy_source}) {
          out.boundary("generic boundaries", allocation.format, flags, device_point::end_of_frame, prior);
        }
        for (const auto prior : {depth_read, depth_read | nps, depth_read | ps, depth_read | nps | ps}) {
          out.boundary("generic boundaries", allocation.format, flags, device_point::unbind_read_only_dsv, prior);
        }
      }
    }

    // UI tags: every state of a UI target, completed under a stale or absent
    // declaration or named by its declaration alone.
    for (const auto flags : {0u, rt, uav, rt | uav, simultaneous | rt}) {
      each_allowed(flags, [&](std::uint32_t state) {
        const auto prior = prior_for(state, flags);
        const auto stale = state != unordered && (flags & uav) ? unordered : state != copy_dest ? copy_dest :
                                                                                                  copy_source;
        out.api("UI tags", DXGI_FORMAT_R8G8B8A8_UNORM, {flags, evidence::transition, state}, prior, claim::sl_declared, stale, consumer::ui);
        out.api("UI tags", DXGI_FORMAT_R8G8B8A8_UNORM, {flags, evidence::transition, state}, prior, claim::sl_undeclared, 0, consumer::ui);
        out.api("UI tags", DXGI_FORMAT_R8G8B8A8_UNORM, {flags, evidence::none}, state, claim::sl_declared, state, consumer::ui);
      });
    }
    for (const auto format : {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
      for (const auto state : {render_target, nps | ps, copy_source, unordered}) {
        out.api("UI tags", format, {rt | uav, evidence::transition, state}, prior_for(state, rt | uav), claim::sl_declared, state, consumer::ui);
        out.api("UI tags", format, {rt | uav, evidence::none}, state, claim::sl_declared, state, consumer::ui);
      }
    }

    // The UI layer's clear boundary (D12).
    for (const auto format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
      out.boundary("UI clear boundary", format, rt, device_point::ui_clear, render_target, ui_clear_boundary::legacy_render_target);
      if (enhanced_barriers) {
        out.boundary("UI clear boundary", format, rt, device_point::ui_clear, common, ui_clear_boundary::enhanced_render_target);
      }
      out.boundary("UI clear boundary", format, rt | simultaneous, device_point::ui_clear, common, ui_clear_boundary::simultaneous_common);
    }
    return std::move(out.cells);
  }

  // The device.

  void transition_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, UINT subresource, std::uint32_t before, std::uint32_t after, D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = flags;
    barrier.Transition = {resource, subresource, D3D12_RESOURCE_STATES(before), D3D12_RESOURCE_STATES(after)};
    list->ResourceBarrier(1, &barrier);
  }

  // One enhanced texture barrier on all subresources, or on the index form of
  // subresource 0.
  void texture_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, bool whole, D3D12_BARRIER_SYNC sync_before, D3D12_BARRIER_SYNC sync_after, std::uint32_t access_before, std::uint32_t access_after, std::uint32_t layout_before, std::uint32_t layout_after) {
    ComPtr<ID3D12GraphicsCommandList7> list7;
    check(list->QueryInterface(IID_PPV_ARGS(&list7)), "enhanced barrier list");
    D3D12_TEXTURE_BARRIER value {};
    value.SyncBefore = sync_before;
    value.SyncAfter = sync_after;
    value.AccessBefore = D3D12_BARRIER_ACCESS(access_before);
    value.AccessAfter = D3D12_BARRIER_ACCESS(access_after);
    value.LayoutBefore = D3D12_BARRIER_LAYOUT(layout_before);
    value.LayoutAfter = D3D12_BARRIER_LAYOUT(layout_after);
    value.pResource = resource;
    value.Subresources.IndexOrFirstMipLevel = whole ? UINT32_MAX : 0;
    D3D12_BARRIER_GROUP group {};
    group.Type = D3D12_BARRIER_TYPE_TEXTURE;
    group.NumBarriers = 1;
    group.pTextureBarriers = &value;
    list7->Barrier(1, &group);
  }

  // From the legacy COMMON state into layout with access (NO_ACCESS syncs
  // nothing). The legacy state is the COMMON layout with ACCESS_COMMON, and
  // a barrier on a list may follow other access in its ExecuteCommandLists
  // call, which SyncBefore NONE would deny (the debug layer's 1417).
  void enter_layout(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, bool whole, std::uint32_t layout, std::uint32_t access) {
    const auto sync = access == D3D12_BARRIER_ACCESS_NO_ACCESS ? D3D12_BARRIER_SYNC_NONE : D3D12_BARRIER_SYNC_ALL;
    texture_barrier(list, resource, whole, D3D12_BARRIER_SYNC_ALL, sync, D3D12_BARRIER_ACCESS_COMMON, access, D3D12_BARRIER_LAYOUT_COMMON, layout);
  }

  // A split into layout with access, begun from a copy layout: the runtime
  // refuses (Close: E_INVALIDARG) a split that begins with no access, ends
  // in NO_ACCESS or keeps its layout.
  struct split_layout {
    std::uint32_t via {}, via_access {};

    explicit split_layout(std::uint32_t layout):
        via(matrix::generic_layout(layout) == D3D12_BARRIER_LAYOUT_COPY_SOURCE ? D3D12_BARRIER_LAYOUT_COPY_DEST : D3D12_BARRIER_LAYOUT_COPY_SOURCE),
        via_access(via == D3D12_BARRIER_LAYOUT_COPY_DEST ? D3D12_BARRIER_ACCESS_COPY_DEST : D3D12_BARRIER_ACCESS_COPY_SOURCE) {}

    void begin(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, bool whole, std::uint32_t layout, std::uint32_t access) const {
      texture_barrier(list, resource, whole, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COMMON, via_access, D3D12_BARRIER_LAYOUT_COMMON, via);
      texture_barrier(list, resource, whole, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_SPLIT, via_access, access, via, layout);
    }

    void end(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, bool whole, std::uint32_t layout, std::uint32_t access) const {
      texture_barrier(list, resource, whole, D3D12_BARRIER_SYNC_SPLIT, D3D12_BARRIER_SYNC_ALL, via_access, access, via, layout);
    }
  };

  // A cell's source: subresource 0 holds the cell's pattern from the shared
  // upload buffer; its other subresources keep a committed resource's zeros.
  struct source_texture {
    ComPtr<ID3D12Resource> resource;
    capture::source_ref retained;
    std::uint32_t initial {copy_dest};
    bool uploaded {};
  };

  struct run {
    capture::record_diagnostic diagnostic;
    matrix::result actual;
    bool quarantined {}, debug_clean {true}, bytes_checked {}, bytes_exact {true};
    std::string trouble;
    capture::preservation_ticket preserved;
    capture::diagnostic_ticket local;
    ComPtr<ID3D12Resource> ui_copy;
  };

  // A COMPUTE queue and its list, in ReShade's list lifecycle like the
  // fixture's direct list.
  struct compute_lane {
    fixture &gpu;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator, spare;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    std::uint64_t completion {};

    explicit compute_lane(fixture &value):
        gpu(value) {
      D3D12_COMMAND_QUEUE_DESC desc {};
      desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
      check(gpu.device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "compute queue");
      check(gpu.device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&allocator)), "compute allocator");
      check(gpu.device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&spare)), "compute spare allocator");
      check(gpu.device->CreateCommandList(0, desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "compute list");
      check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "compute fence");
      capture::observe_queue(native(queue.Get()));
      created(list.Get());
      capture::observe_command(native(list.Get()));
      capture::poll();
      check(close_list(list.Get()), "compute initial close");
      reset();
    }

    ~compute_lane() {
      destroyed(list.Get());
      capture::command_destroyed(native(list.Get()));
    }

    void reset() {
      check(spare->Reset(), "compute allocator reset");
      check(reset_list(list.Get(), spare.Get()), "compute list reset");
      std::swap(allocator, spare);
      capture::poll();
    }

    void wait() {
      check(queue->Signal(fence.Get(), ++completion), "compute signal");
      HANDLE event = CreateEventW(nullptr, false, false, nullptr);
      require(event != nullptr, "compute event");
      const auto hr = fence->SetEventOnCompletion(completion, event);
      const auto outcome = SUCCEEDED(hr) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
      CloseHandle(event);
      require(outcome == WAIT_OBJECT_0, "compute completion timeout");
    }
  };

  // The earlier list of a cell's own ExecuteCommandLists call, on its queue
  // type: where the game begins the split barrier the cell's recording ends
  // (D4b). D3D12 completes a split only within one ExecuteCommandLists call
  // (the debug layer refuses a BEGIN_ONLY with no END_ONLY in the same call),
  // so this list executes with the cell's, in ReShade's list lifecycle.
  struct earlier_list {
    ComPtr<ID3D12CommandAllocator> allocator, spare;
    ComPtr<ID3D12GraphicsCommandList> list;

    earlier_list(fixture &gpu, D3D12_COMMAND_LIST_TYPE type) {
      check(gpu.device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)), "earlier allocator");
      check(gpu.device->CreateCommandAllocator(type, IID_PPV_ARGS(&spare)), "earlier spare allocator");
      check(gpu.device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "earlier list");
      created(list.Get());
      capture::observe_command(native(list.Get()));
      capture::poll();
      check(close_list(list.Get()), "earlier initial close");
      reset();
    }

    ~earlier_list() {
      destroyed(list.Get());
      capture::command_destroyed(native(list.Get()));
    }

    void reset() {
      check(spare->Reset(), "earlier allocator reset");
      check(reset_list(list.Get(), spare.Get()), "earlier list reset");
      std::swap(allocator, spare);
      capture::poll();
    }
  };

  // The D3D12 debug layer's errors since the last call, which it clears.
  std::string debug_errors(fixture &gpu) {
    std::string out;
    if (!gpu.debug_messages) {
      return out;
    }
    for (UINT64 i = 0; i != gpu.debug_messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
      SIZE_T size {};
      check(gpu.debug_messages->GetMessage(i, nullptr, &size), "debug message size");
      std::vector<unsigned char> storage(size);
      auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
      check(gpu.debug_messages->GetMessage(i, message, &size), "debug message");
      if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
        out += out.empty() ? "" : " | ";
        out += message->pDescription;
      }
    }
    gpu.debug_messages->ClearStoredMessages();
    return out;
  }

  struct device_matrix {
    fixture &gpu;
    compute_lane compute;
    consumer_fixture consumer;
    earlier_list earlier_direct, earlier_compute;
    ComPtr<ID3D12DescriptorHeap> depth_views;
    std::uint64_t sequence {};

    explicit device_matrix(fixture &value):
        gpu(value),
        compute(value),
        consumer(value),
        earlier_direct(value, D3D12_COMMAND_LIST_TYPE_DIRECT),
        earlier_compute(value, D3D12_COMMAND_LIST_TYPE_COMPUTE) {
      D3D12_DESCRIPTOR_HEAP_DESC desc {};
      desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
      desc.NumDescriptors = 1;
      check(gpu.device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&depth_views)), "DSV heap");
    }

    D3D12_RESOURCE_DESC description(const cell &value) const {
      const auto &source = value.value.state.resource;
      D3D12_RESOURCE_DESC desc {};
      desc.Dimension = source.shape == resource_shape::texture3d ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = source_width;
      desc.Height = source_height;
      desc.DepthOrArraySize = source.shape == resource_shape::array || source.shape == resource_shape::texture3d ? 2 : 1;
      desc.MipLevels = source.shape == resource_shape::texture2d_mips ? 3 : 1;
      desc.SampleDesc.Count = source.shape == resource_shape::msaa ? 4 : 1;
      desc.Format = value.format;
      desc.Flags = D3D12_RESOURCE_FLAGS(source.flags);
      return desc;
    }

    // Whether this device can create the cell's allocation (MSAA support).
    bool creatable(const cell &value) const {
      const auto desc = description(value);
      if (desc.SampleDesc.Count == 1) {
        return true;
      }
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels {};
      levels.Format = desc.Format;
      levels.SampleCount = desc.SampleDesc.Count;
      return SUCCEEDED(gpu.device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) && levels.NumQualityLevels;
    }

    UINT subresource_of(const cell &value, const D3D12_RESOURCE_DESC &desc) const {
      switch (value.value.state.resource.subresource) {
        case subresource_barrier::zero:
          return 0;
        case subresource_barrier::other_mip_only:
          return 1;
        case subresource_barrier::stencil_plane_only:
          return desc.MipLevels * desc.DepthOrArraySize;  // Plane 1, mip 0.
        default:
          return D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      }
    }

    // The pattern of a cell's subresource 0, written into its upload rows.
    static void fill(const cell &value, unsigned index, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint, unsigned char *base) {
      const auto kind = kind_of(value.format);
      for (unsigned y = 0; y != source_height; ++y) {
        for (unsigned x = 0; x != source_width; ++x) {
          texel(kind, index * 4099u + y * source_width + x, base + footprint.Offset + y * footprint.Footprint.RowPitch + x * texel_bytes(kind));
        }
      }
    }

    static bool matches(const cell &value, unsigned index, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint, const unsigned char *base) {
      const auto kind = kind_of(value.format);
      unsigned char expected[8] {};
      for (unsigned y = 0; y != source_height; ++y) {
        for (unsigned x = 0; x != source_width; ++x) {
          texel(kind, index * 4099u + y * source_width + x, expected);
          if (!same_texel(kind, expected, base + footprint.Offset + y * footprint.Footprint.RowPitch + x * texel_bytes(kind))) {
            return false;
          }
        }
      }
      return true;
    }

    // The earlier list: each source's upload and the state the cell starts in.
    void prepare(const std::vector<const cell *> &batch, std::vector<source_texture> &sources) {
      UINT64 total {};
      std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(batch.size());
      std::vector<D3D12_RESOURCE_DESC> descs(batch.size());
      for (std::size_t i = 0; i != batch.size(); ++i) {
        descs[i] = description(*batch[i]);
        auto &source = sources[i];
        const auto shape = batch[i]->value.state.resource.shape;
        source.uploaded = shape == resource_shape::texture2d || shape == resource_shape::texture2d_mips;
        source.initial = shape != resource_shape::msaa ? copy_dest : (descs[i].Flags & ds) ? depth_write :
                                                                                             render_target;
        const auto properties = heap(D3D12_HEAP_TYPE_DEFAULT);
        check(gpu.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &descs[i], D3D12_RESOURCE_STATES(source.initial), nullptr, IID_PPV_ARGS(&source.resource)), "cell source");
        source.retained = capture::retain_source(native(source.resource.Get()));
        if (source.uploaded) {
          UINT64 size {};
          total = (total + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~UINT64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
          gpu.device->GetCopyableFootprints(&descs[i], 0, 1, total, &footprints[i], nullptr, nullptr, &size);
          total += size;
        }
      }
      ComPtr<ID3D12Resource> upload;
      if (total) {
        D3D12_RESOURCE_DESC buffer {};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const auto properties = heap(D3D12_HEAP_TYPE_UPLOAD);
        check(gpu.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "cell upload");
        unsigned char *mapped {};
        D3D12_RANGE none {};
        check(upload->Map(0, &none, reinterpret_cast<void **>(&mapped)), "cell upload map");
        for (std::size_t i = 0; i != batch.size(); ++i) {
          if (sources[i].uploaded) {
            fill(*batch[i], unsigned(batch[i] - first), footprints[i], mapped);
          }
        }
        upload->Unmap(0, nullptr);
      }
      auto *list = gpu.list.Get();
      for (std::size_t i = 0; i != batch.size(); ++i) {
        const auto &value = batch[i]->value;
        const auto &source = value.state.resource;
        auto *resource = sources[i].resource.Get();
        if (sources[i].uploaded) {
          D3D12_TEXTURE_COPY_LOCATION from {}, to {};
          from.pResource = upload.Get();
          from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
          from.PlacedFootprint = footprints[i];
          to.pResource = resource;
          to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
        // A simultaneous-access texture decays to COMMON when this list's
        // execution completes.
        if (!(source.flags & simultaneous) && value.prior != sources[i].initial) {
          transition_barrier(list, resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, sources[i].initial, value.prior);
        }
        // A split the cell ends begins on its earlier_list (record).
        if (source.seen == evidence::earlier_list_enhanced) {
          enter_layout(list, resource, true, source.layout, source.access);
        }
      }
      gpu.submit();
      gpu.wait();
      gpu.reset();
      const auto errors = debug_errors(gpu);
      if (!errors.empty()) {
        std::fprintf(stderr, "D3D12 errors preparing a batch's sources: %s\n", errors.c_str());
        throw std::runtime_error("the device layer's own setup is invalid D3D12");
      }
    }

    capture::input api_input(const cell &value, const source_texture &source) {
      const auto &state = value.value.state;
      capture::input input;
      input.source = source.retained;
      input.resource.native = native(source.resource.Get());
      input.valid_until = sunshine_scene_depth::lifetime::at_call;
      input.force_snapshot = true;
      input.epoch = 1;
      input.sequence = ++sequence;
      input.tick = GetTickCount64();
      // The adapters' proofs (as the CPU layer's copy-state view maps them):
      // Streamline declares or leaves UINT_MAX, NGX's Super Resolution names
      // its contract, and an input without either needs an observation.
      const bool ngx = state.role == consumer::depth && (state.source == claim::ngx_contract || state.source == claim::none);
      input.provider = ngx ? sunshine_scene_depth::provider_kind::ngx : sunshine_scene_depth::provider_kind::streamline;
      switch (state.source) {
        case claim::sl_declared:
          input.proof = sunshine_scene_depth::state_proof::declared;
          input.native_state = state.declared;
          break;
        case claim::sl_undeclared:
          input.proof = sunshine_scene_depth::state_proof::unavailable;
          input.native_state = UINT32_MAX;
          break;
        case claim::ngx_contract:
          input.proof = sunshine_scene_depth::state_proof::sdk_contract;
          input.native_state = nps;
          break;
        default:
          input.proof = sunshine_scene_depth::state_proof::observed_nonzero;
          input.native_state = 0;
          break;
      }
      return input;
    }

    static matrix::result judged(bool admitted, const capture::record_diagnostic &diagnostic) {
      matrix::result out;
      if (admitted) {
        out.kind = matrix::outcome_kind::capture;
        out.basis = {diagnostic.enhanced_copy, diagnostic.enhanced_copy ? 0 : diagnostic.copy_state, diagnostic.enhanced_copy ? diagnostic.observed_layout : 0, diagnostic.enhanced_copy ? diagnostic.observed_access : 0};
        out.basis_known = true;
        return out;
      }
      const auto reason = reasons::classify(diagnostic.stage);
      if (reason == reasons::reason_class::hold) {
        out.kind = matrix::outcome_kind::hold;
        return out;
      }
      out.why = unsigned(reason) <= unsigned(reasons::reason_class::fg_scope) ? matrix::refusal(unsigned(reason)) : matrix::refusal::none;
      return out;
    }

    // A known-gap row says this cell copies from a state nothing proved.
    static bool unsafe(const cell &value) {
      for (const auto &row : gaps::device_rows) {
        if (row.status == gaps::kind::expected_failure && row.records == &gaps::behaviours::copy && row.matches(value.value)) {
          return true;
        }
      }
      return false;
    }

    // Records one cell on its list and executes it alone (or discards it).
    void record(const cell &value, source_texture &source, run &out) {
      const auto &state = value.value.state;
      const auto &evidence_on = state.resource;
      const auto desc = source.resource->GetDesc();
      const auto subresource = subresource_of(value, desc);
      const bool whole = evidence_on.subresource != subresource_barrier::zero;
      auto *resource = source.resource.Get();
      const bool on_compute = state.list == list_type::compute;
      // The end-of-frame copy runs on the runtime's own list, whose recording
      // ends only when it is submitted. The copy is decided by what its
      // recording names, which there is nothing, so it is recorded here on a
      // game list that names nothing either and can be discarded.
      auto *list = on_compute ? compute.list.Get() : gpu.list.Get();
      auto &earlier = on_compute ? earlier_compute : earlier_direct;
      const bool split_ended = evidence_on.seen == evidence::end_without_begin;
      out.quarantined = unsafe(value);
      debug_errors(gpu);  // Drop what an earlier discarded recording produced.

      // A split the copy's recording ends began on an earlier list of the
      // same ExecuteCommandLists call.
      if (split_ended) {
        transition_barrier(earlier.list.Get(), resource, subresource, value.value.prior, evidence_on.observed, D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY);
      }

      // The game's barrier on the copy's recording.
      switch (evidence_on.seen) {
        case evidence::transition:
          transition_barrier(list, resource, subresource, value.value.prior, evidence_on.observed);
          break;
        case evidence::end_without_begin:
          transition_barrier(list, resource, subresource, value.value.prior, evidence_on.observed, D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);
          break;
        case evidence::split_open:
          transition_barrier(list, resource, subresource, value.value.prior, evidence_on.observed, D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY);
          break;
        case evidence::alias_wildcard:
          {
            D3D12_RESOURCE_BARRIER alias {};
            alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
            list->ResourceBarrier(1, &alias);
            break;
          }
        case evidence::enhanced:
          enter_layout(list, resource, whole, evidence_on.layout, evidence_on.access);
          break;
        case evidence::enhanced_split_open:
          split_layout(evidence_on.layout).begin(list, resource, whole, evidence_on.layout, evidence_on.access);
          break;
        default:
          break;
      }

      // The copy at its point.
      bool admitted {};
      switch (value.value.point) {
        case device_point::api:
          {
            auto input = api_input(value, source);
            if (state.role == consumer::ui) {
              out.local = capture::record_local_texture(native(list), input, &out.diagnostic, capture::auxiliary_state_policy(input));
              admitted = bool(out.local) && out.diagnostic.result == capture::status::recorded;
              if (out.local) {
                capture::finish_diagnostic_texture(out.local, true);
              }
            } else {
              const auto ticket = capture::record(native(list), input, &out.diagnostic);
              admitted = ticket && out.diagnostic.result == capture::status::recorded;
              if (ticket) {
                capture::finish(ticket, true);
              }
            }
            break;
          }
        case device_point::ui_clear:
          {
            // game3d_ui_layer.cpp record_into through ReShade's D3D12
            // backend: the copy (kept shader-readable) to COPY_DEST, the
            // target from RENDER_TARGET to COPY_SOURCE, CopyResource, and
            // both back. The game first puts an enhanced target in the
            // RENDER_TARGET layout.
            if (value.value.clear == ui_clear_boundary::enhanced_render_target) {
              enter_layout(list, resource, true, D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_ACCESS_RENDER_TARGET);
            }
            out.ui_copy = destination(gpu.device.Get(), value.format, source_width, source_height);
            transition_barrier(list, out.ui_copy.Get(), 0, ps, copy_dest);
            transition_barrier(list, resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, render_target, copy_source);
            list->CopyResource(out.ui_copy.Get(), resource);
            transition_barrier(list, resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, copy_source, render_target);
            transition_barrier(list, out.ui_copy.Get(), 0, copy_dest, ps);
            out.diagnostic.result = capture::status::recorded;
            out.diagnostic.stage = capture::record_stage::recorded;
            out.diagnostic.copy_state = render_target;
            admitted = true;
            break;
          }
        default:
          // The DEPTH_WRITE depth_addon.cpp declares at a clear (on_clear_depth_impl),
          // where a DSV is unbound (on_bind_depth_stencil) and at the end of
          // the frame (preserve_d3d12_at_effects, from the allocation's usage).
          out.preserved = capture::record_preserved(native(list), native(resource), depth_write, &out.diagnostic);
          admitted = bool(out.preserved) && out.diagnostic.result == capture::status::recorded;
          break;
      }
      out.actual = judged(admitted, out.diagnostic);

      // A repeat by record_preserved from the same basis reads an admitted
      // depth copy's bytes.
      if (admitted && !out.quarantined && value.value.point == device_point::api && state.role == consumer::depth) {
        capture::record_diagnostic repeat;
        out.preserved = capture::record_preserved(native(list), native(resource), declaration_for(out.actual.basis), &repeat);
        if (!out.preserved || repeat.result != capture::status::recorded || repeat.enhanced_copy != out.actual.basis.enhanced) {
          out.bytes_exact = false;
          out.trouble += " the byte check's repeat copy was refused;";
        }
      }

      // The game continues: ends its split; clears depth after the copy.
      if (evidence_on.seen == evidence::split_open) {
        transition_barrier(list, resource, subresource, value.value.prior, evidence_on.observed, D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);
      } else if (evidence_on.seen == evidence::enhanced_split_open) {
        split_layout(evidence_on.layout).end(list, resource, whole, evidence_on.layout, evidence_on.access);
      } else if (value.value.point == device_point::clear) {
        D3D12_DEPTH_STENCIL_VIEW_DESC view {};
        view.Format = depth_view_of(value.format);
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        const auto handle = depth_views->GetCPUDescriptorHandleForHeapStart();
        gpu.device->CreateDepthStencilView(resource, &view, handle);
        list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
      }

      // Execute alone (after its earlier list, in one call), or discard a
      // copy from an unproven state.
      check(close_list(list), on_compute ? "compute close" : "copy close");
      if (split_ended) {
        check(close_list(earlier.list.Get()), "earlier close");
      }
      if (!out.quarantined) {
        ID3D12CommandList *lists[] {earlier.list.Get(), list};
        auto &queue = on_compute ? compute.queue : gpu.queue;
        queue->ExecuteCommandLists(split_ended ? 2 : 1, split_ended ? lists : lists + 1);
        if (on_compute) {
          compute.wait();
        } else {
          gpu.wait();
        }
      }
      if (on_compute) {
        compute.reset();
      } else {
        gpu.reset();
      }
      if (split_ended) {
        earlier.reset();
      }
      if (!out.quarantined) {
        const auto errors = debug_errors(gpu);
        if (!errors.empty()) {
          out.debug_clean = false;
          out.trouble += " D3D12: " + errors + ";";
        }
      } else {
        // A discarded copy never reads the source.
        out.preserved = {};
        if (out.local) {
          capture::release_diagnostic_texture(out.local);
        }
        out.local = {};
        out.ui_copy.Reset();
      }
    }

    // Reads every byte check of the batch through the capture owner's
    // consumer copies, in one submission.
    void read_back(const std::vector<const cell *> &batch, std::vector<run> &runs) {
      struct read {
        std::size_t index;
        ComPtr<ID3D12Resource> target;
        std::unique_ptr<readback> bytes;
      };

      std::vector<read> reads;
      const auto queue = native(gpu.foreign_queue.Get());
      for (std::size_t i = 0; i != batch.size(); ++i) {
        auto &out = runs[i];
        if (!out.preserved && !out.local && !out.ui_copy) {
          continue;
        }
        read entry {i};
        bool copied {};
        capture::consumer_diagnostic diagnostic;
        if (out.preserved) {
          entry.target = destination(gpu.device.Get(), sunshine_depth_format::storage(batch[i]->format), source_width, source_height);
          copied = capture::copy_preserved(native(consumer.list.Get()), queue, out.preserved, native(entry.target.Get()), ps, &diagnostic);
        } else if (out.local) {
          capture::diagnostic_texture pixels;
          if (capture::acquire_local_texture(out.local, queue, pixels) == capture::status::ready) {
            entry.target = destination(gpu.device.Get(), DXGI_FORMAT(pixels.format), source_width, source_height);
            copied = capture::copy_local_texture(native(consumer.list.Get()), queue, out.local, native(entry.target.Get()), ps, &diagnostic);
          }
        } else {
          entry.target = out.ui_copy;
          copied = true;
        }
        out.bytes_checked = true;
        if (!copied) {
          out.bytes_exact = false;
          out.trouble += std::string(" the snapshot was not readable (") + capture::name(diagnostic.result) + ");";
          continue;
        }
        entry.bytes = std::make_unique<readback>(gpu, entry.target.Get());
        entry.bytes->record(consumer.list.Get(), entry.target.Get());
        reads.push_back(std::move(entry));
      }
      consumer.submit();
      consumer.wait();
      consumer.reset();
      const auto errors = debug_errors(gpu);
      for (auto &entry : reads) {
        auto &out = runs[entry.index];
        unsigned char *mapped {};
        check(entry.bytes->bytes->Map(0, nullptr, reinterpret_cast<void **>(&mapped)), "cell readback map");
        const bool exact = matches(*batch[entry.index], unsigned(batch[entry.index] - first), entry.bytes->footprint, mapped);
        D3D12_RANGE none {};
        entry.bytes->bytes->Unmap(0, &none);
        if (!exact) {
          out.bytes_exact = false;
          out.trouble += " subresource 0's plane 0 changed in the snapshot;";
        }
        if (!errors.empty()) {
          out.debug_clean = false;
          out.trouble += " D3D12 reading back: " + errors + ";";
        }
      }
      for (auto &out : runs) {
        out.preserved = {};
        if (out.local) {
          capture::release_diagnostic_texture(out.local);
        }
        out.local = {};
        out.ui_copy.Reset();
      }
      capture::poll();
    }

    const cell *first {};
  };

  const char *point_name(device_point point) {
    constexpr const char *names[] {"api", "clear", "end_of_frame", "unbind_read_only_dsv", "ui_clear"};
    return names[unsigned(point)];
  }

  void describe(const char *what, const cell &value, const run &out, const bool *matched) {
    const auto &state = value.value.state;
    const auto &source = state.resource;
    const auto expected = matrix::expect(value.value);
    std::fprintf(stderr,
                 "%s [%s]: format=%u flags=0x%x shape=%u point=%s clear=%u list=%u role=%u claim=%u declared=0x%x prior=0x%x evidence=%u "
                 "observed=0x%x layout=0x%x access=0x%x subresource=%u\n",
                 what,
                 value.block,
                 unsigned(value.format),
                 source.flags,
                 unsigned(source.shape),
                 point_name(value.value.point),
                 unsigned(value.value.clear),
                 unsigned(state.list),
                 unsigned(state.role),
                 unsigned(state.source),
                 state.declared,
                 value.value.prior,
                 unsigned(source.seen),
                 source.observed,
                 source.layout,
                 source.access,
                 unsigned(source.subresource));
    describe_expected(expected);
    const char *ids[std::size(gaps::device_rows)] {};
    for (unsigned i = 0; i != std::size(gaps::device_rows); ++i) {
      ids[i] = gaps::device_rows[i].id;
    }
    std::string rows;
    const bool none[std::size(gaps::device_rows)] {};
    std::fprintf(stderr, "  capture owner: %s status=%s stage=%s basis=0x%x/0x%x/0x%x enhanced=%u refusal=%u; executed=%u debug_clean=%u bytes=%s; rows %s;%s\n", name(out.actual.kind), capture::name(out.diagnostic.result), capture::name(out.diagnostic.stage), out.actual.basis.state, out.actual.basis.layout, out.actual.basis.access, unsigned(out.actual.basis.enhanced), unsigned(out.actual.why), unsigned(!out.quarantined), unsigned(out.debug_clean), !out.bytes_checked ? "not read" : out.bytes_exact ? "exact" :
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "wrong",
                 row_names(matched ? matched : none, ids, unsigned(std::size(gaps::device_rows)), rows),
                 out.trouble.c_str());
  }

  // True when a hardware D3D12 adapter exists (the fixture requires one).
  bool hardware_adapter() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      return false;
    }
    ComPtr<IDXGIAdapter1> adapter;
    for (unsigned index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
      DXGI_ADAPTER_DESC1 desc {};
      adapter->GetDesc1(&desc);
      if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
        return true;
      }
      adapter.Reset();
    }
    return false;
  }
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    // --gpu-validation: the debug layer's GPU-based validation as well.
    // --block <name>: only that block's cells (a partial run cannot show a
    // known-gap row stale). --trace: each batch and cell before it records.
    bool gpu_validation {}, trace {};
    const char *only {};
    unsigned batch_size = 16;  // Sources prepared together; each cell still executes alone.
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--gpu-validation") == 0) {
        gpu_validation = true;
      } else if (std::strcmp(argv[i], "--trace") == 0) {
        trace = true;
      } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
        only = argv[++i];
      } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
        batch_size = unsigned(std::max(1, std::atoi(argv[++i])));
      } else {
        std::fputs("usage: reshade_contract_matrix_device_tests [--gpu-validation] [--trace] [--block <name>] [--batch <cells>]\n", stderr);
        return 2;
      }
    }
    if (!hardware_adapter()) {
      std::puts("SKIP API contract device matrix: no hardware D3D12 adapter");
      return 77;
    }
    {
      ComPtr<ID3D12Debug> available;
      if (gpu_validation && FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&available)))) {
        std::puts("SKIP API contract device matrix under GPU-based validation: the D3D12 debug layer is not installed");
        return 77;
      }
    }
    const auto start = std::chrono::steady_clock::now();
    fixture gpu(gpu_validation);
    const bool debug_layer = gpu.debug_messages != nullptr;
    if (debug_layer) {
      gpu.debug_messages->ClearStoredMessages();
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 options {};
    ComPtr<ID3D12GraphicsCommandList7> list7;
    const bool enhanced_barriers = SUCCEEDED(gpu.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options, sizeof(options))) &&
                                   options.EnhancedBarriersSupported && SUCCEEDED(gpu.list.As(&list7));
    list7.Reset();
    auto cells = enumerate(enhanced_barriers);
    require(cells.size() > 1500, "the device matrix enumerated too few legal cells");
    if (only) {
      cells.erase(std::remove_if(cells.begin(), cells.end(), [only](const cell &value) {
                    return std::strcmp(value.block, only) != 0;
                  }),
                  cells.end());
      require(!cells.empty(), "no cell belongs to that block");
    }
    device_matrix device(gpu);
    device.first = cells.data();

    each_tally<std::size(gaps::device_rows)> verdicts;
    std::vector<std::pair<const char *, unsigned>> blocks;
    unsigned executed {}, discarded {}, read {}, not_creatable {};
    for (std::size_t begin = 0; begin < cells.size();) {
      std::vector<const cell *> batch;
      for (; begin < cells.size() && batch.size() != batch_size; ++begin) {
        if (!device.creatable(cells[begin])) {
          ++not_creatable;
          continue;
        }
        batch.push_back(&cells[begin]);
      }
      if (batch.empty()) {
        continue;
      }
      std::vector<source_texture> sources(batch.size());
      std::vector<run> runs(batch.size());
      if (trace) {
        std::fprintf(stderr, "batch of %zu from cell %u\n", batch.size(), unsigned(batch.front() - cells.data()));
      }
      device.prepare(batch, sources);
      for (std::size_t i = 0; i != batch.size(); ++i) {
        if (trace) {
          describe("cell", *batch[i], runs[i], nullptr);
        }
        device.record(*batch[i], sources[i], runs[i]);
        ++(runs[i].quarantined ? discarded : executed);
      }
      device.read_back(batch, runs);
      for (std::size_t i = 0; i != batch.size(); ++i) {
        const auto &value = *batch[i];
        const auto &out = runs[i];
        read += out.bytes_checked;
        const auto expected = matrix::expect(value.value);
        const bool satisfied = matrix::satisfies(expected, out.actual) && out.debug_clean && out.bytes_exact;
        bool matched[std::size(gaps::device_rows)] {};
        const auto verdict = gaps::judge_each(gaps::device_rows, value.value, satisfied, {out.actual}, matched);
        if (const auto *what = verdicts.add(verdict, matched, satisfied)) {
          describe(what, value, out, matched);
        }
        if (blocks.empty() || std::strcmp(blocks.back().first, value.block) != 0) {
          blocks.push_back({value.block, 0});
        }
        ++blocks.back().second;
      }
      sources.clear();
      capture::poll();
    }
    verdicts.finish(gaps::device_rows, "device", !only);
    std::printf("PASS API contract device matrix: %zu legal cells (", cells.size());
    for (std::size_t i = 0; i != blocks.size(); ++i) {
      std::printf("%s%s %u", i ? ", " : "", blocks[i].first, blocks[i].second);
    }
    std::printf("); %u executed alone, %u recorded and discarded (an unproven copy), %u snapshots read back exact", executed, discarded, read);
    if (not_creatable) {
      std::printf(", %u not creatable on this device", not_creatable);
    }
    std::printf("%s; ", enhanced_barriers ? "" : "; no enhanced barriers on this device");
    verdicts.print(gaps::device_rows);
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("; %.1f s\n", seconds);
    if (!debug_layer) {
      std::puts("SKIP the D3D12 debug layer (Windows Graphics Tools) is not installed: every other check passed, but no recorded barrier was validated");
      return 77;
    }
    std::printf("PASS D3D12 validation clean%s\n", gpu_validation ? " under GPU-based validation" : "");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

#pragma once
// EXTERNAL, DEFAULT-OFF RESEARCH CONTRACT. No production producer/renderer wire.
// The caller owns actual capture/texture and Base publication lifetime. These tags
// are identity joins, not cryptographic proof that a detector ran or pixels are glyphs.
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace models::research_current_ui_v1 {
  inline constexpr std::uint32_t scope_tag=0x31554943u;
  inline constexpr std::uint32_t packet_magic=0x31505543u;
  inline constexpr std::uint32_t schema=1u;
  enum class decision_e:std::uint32_t {unknown=0u,positive=1u,current_empty=2u};
  enum class producer_e:std::uint32_t {fixture_preuploaded=1u,validated_current_rgb=2u};

  struct current_capture_t {
    std::uint64_t capture_epoch=0,conversion_serial=0,source_clock_us=0;
    std::uint64_t analysis_generation=0,subtitle_context_generation=0,domain_tag=0;
  };
  struct retained_base_t {
    std::uint64_t estimator_generation=0,publication_sequence=0,observation_clock_us=0;
    std::uint32_t cut_epoch=0;
  };
  struct layout_t {
    std::array<std::uint32_t,4> source_rect{}; // left,top,width,height in current t0
    std::array<std::uint32_t,4> native_field{}; // crop_top,crop_height,BaseW,BaseH
    std::array<std::uint32_t,4> content{}; // exact Base content LTRB
  };
  // b4:9uint4/144B, deliberately separate from legacy b2/b3 and SCT/SCP.
  struct alignas(16) constants_t {
    std::array<std::uint32_t,4> control{}; // enabled0=>literal legacy;tag,schema,color_mode
    std::array<std::uint32_t,4> current{}; // conversion serial64,source clock64
    std::array<std::uint32_t,4> scope{}; // capture epoch64,analysis generation64
    std::array<std::uint32_t,4> base{}; // retained Base clock64,cut epoch,current cutUNKNOWN0
    std::array<std::uint32_t,4> source_rect{};
    std::array<std::uint32_t,4> native_field{};
    std::array<std::uint32_t,4> content{};
    std::array<std::uint32_t,4> domain{}; // source-domain tag64,Base estimator generation64
    std::array<std::uint32_t,4> publication{}; // subtitle-context64,Base publication sequence64
    bool operator==(const constants_t&)const=default;
  };
  using packet_t=std::array<std::uint32_t,64>; //t9 StructuredBuffer<uint>,256B
  static_assert(sizeof(constants_t)==144u&&alignof(constants_t)==16u);
  static_assert(offsetof(constants_t,current)==16u&&offsetof(constants_t,base)==48u);
  static_assert(offsetof(constants_t,publication)==128u&&sizeof(packet_t)==256u);
  inline std::array<std::uint32_t,2> split(std::uint64_t x)noexcept{return {std::uint32_t(x),std::uint32_t(x>>32)};}
  // SERIALIZATION ONLY, not an authorizer. Before binding, the host must authenticate
  // retained_base_t against the actual still-current expected_t and Base/state SRVs,
  // own the exact current capture texture, and verify current domain/analysis/context
  // permission. GPU state cross-checks clock/cut/field; gen/pub/domain echo checks
  // cannot independently prove the provenance of arbitrary caller-supplied SRVs.
  inline constants_t make_constants(const current_capture_t &c,const retained_base_t &b,const layout_t &l,std::uint32_t color_mode=0)noexcept {
    constants_t out;
    const auto serial=split(c.conversion_serial),clock=split(c.source_clock_us),epoch=split(c.capture_epoch),analysis=split(c.analysis_generation),baseclock=split(b.observation_clock_us),domain=split(c.domain_tag),generation=split(b.estimator_generation),context=split(c.subtitle_context_generation),publication=split(b.publication_sequence);
    out.control={1u,scope_tag,schema,color_mode};out.current={serial[0],serial[1],clock[0],clock[1]};
    out.scope={epoch[0],epoch[1],analysis[0],analysis[1]};out.base={baseclock[0],baseclock[1],b.cut_epoch,0u};
    out.source_rect=l.source_rect;out.native_field=l.native_field;out.content=l.content;
    out.domain={domain[0],domain[1],generation[0],generation[1]};out.publication={context[0],context[1],publication[0],publication[1]};return out;
  }
  // CONTRACT FIXTURE AUTHORING ONLY. This does not produce current glyph evidence.
  // A future validated current-RGB producer must emit the same packet atomically with
  // M and its actual source snapshot under the exact current capture permission.
  inline packet_t make_fixture_packet(const constants_t &expected,decision_e decision,float analysis_source_u_target)noexcept {
    packet_t out{};out[0]=packet_magic;out[1]=schema;out[2]=static_cast<std::uint32_t>(decision);
    out[3]=static_cast<std::uint32_t>(producer_e::fixture_preuploaded);
    out[4]=expected.current[2];out[5]=expected.current[3];out[6]=std::bit_cast<std::uint32_t>(analysis_source_u_target);
    std::memcpy(out.data()+8,&expected,sizeof(expected));return out;
  }
}

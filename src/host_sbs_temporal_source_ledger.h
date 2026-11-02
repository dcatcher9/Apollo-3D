#pragma once
// Private capture/material metadata only. No glyph, EMPTY, completion or drawing authority.
#include <array>
#include <cstdint>
#include <limits>

namespace models::host_sbs_temporal_source_bridge {
  struct domain_t {
    std::uint32_t source_width=0,source_height=0,format=0,transfer=0;
    std::uint32_t mip_levels=0,array_size=0,samples=0,quality=0;
    std::array<std::uint32_t,4> analysis_ltrb{};
    std::uint64_t analysis_generation=0,context_generation=0;
    std::uint64_t root_authority_generation=0,region_authority_generation=0,browser_epoch=0;
    std::uint32_t authority=0;
    bool operator==(const domain_t &) const = default;
    bool valid() const noexcept {
      return source_width&&source_height&&context_generation&&authority<=2u&&
        analysis_ltrb[0]<analysis_ltrb[2]&&analysis_ltrb[1]<analysis_ltrb[3]&&
        analysis_ltrb[2]<=source_width&&analysis_ltrb[3]<=source_height&&
        (authority==0u ? analysis_generation==0u : analysis_generation!=0u);
    }
  };
  struct certificate_t {
    domain_t domain{};
    std::uint64_t source_epoch=0,material_serial=0,capture_clock_us=0;
    // A STICKY certificate for this material's FIRST transition, not the last observe() call.
    bool predecessor_contiguous=false;
    bool genuine_capture=false;
    bool valid() const noexcept {return domain.valid()&&source_epoch&&material_serial&&capture_clock_us&&genuine_capture;}
  };
  struct observation_t {
    domain_t domain{};
    std::uint64_t source_epoch=0,capture_clock_us=0,content_clock_us=0;
    // Provider identity is backed by a separately retained shared DDup-history owner.
    std::uintptr_t history_identity=0;
    std::uint64_t surface_token=0;
    bool genuine_capture=false;
    // These may be true ONLY for actual adjacent DDup single-present acquisitions and the
    // identical input/cursor domain. No token-gap, model-ID or ACK-count inference is allowed.
    bool immediate_previous_capture_observed=false;
    bool exact_analysis_pixels_unchanged=false;
  };
  struct counters_t {
    std::uint64_t captures=0,duplicates=0,clean_folds=0,materials=0,unknown=0,admitted=0;
  };
  class ledger_t {
    observation_t previous_{};
    certificate_t material_{};
    counters_t counters_{};
    bool have_previous_=false,gap_=true;
    std::uint64_t watermark_=0;
  public:
    // New authenticated domain/pipeline epoch only. The owner allocates a monotonically unique
    // epoch; a reset/finish/duplicate is not a new source pipeline.
    bool begin_domain(std::uint64_t epoch,const domain_t &domain) noexcept {
      if(!epoch||epoch<=material_.source_epoch||!domain.valid())return false;
      material_={};material_.source_epoch=epoch;material_.domain=domain;
      previous_={};have_previous_=false;gap_=true;watermark_=0;return true;
    }
    void uncertain_interval() noexcept {gap_=true;++counters_.unknown;}
    certificate_t observe(const observation_t &in) noexcept {
      if(!in.genuine_capture||!in.capture_clock_us||!in.source_epoch||!in.domain.valid()||
        in.source_epoch!=material_.source_epoch||in.domain!=material_.domain||
        in.capture_clock_us<watermark_) {uncertain_interval();return {};}
      const bool same_capture=have_previous_&&in.capture_clock_us==previous_.capture_clock_us;
      if(same_capture) {
        if(gap_){++counters_.unknown;return {};}
        if(in.content_clock_us!=previous_.content_clock_us||in.history_identity!=previous_.history_identity||
          in.surface_token!=previous_.surface_token) {uncertain_interval();return {};}
        ++counters_.duplicates;
        // Deliberately do NOT rewrite predecessor_contiguous or clear a pending gap.
        return material_;
      }
      if(have_previous_&&in.capture_clock_us<=previous_.capture_clock_us){uncertain_interval();return {};}
      const bool adjacent=have_previous_&&!gap_&&in.immediate_previous_capture_observed;
      const bool clean=adjacent&&in.exact_analysis_pixels_unchanged;
      ++counters_.captures;
      if(clean){++counters_.clean_folds;}
      else {
        if(material_.material_serial==std::numeric_limits<std::uint64_t>::max()){uncertain_interval();return {};}
        ++material_.material_serial;++counters_.materials;
        material_.capture_clock_us=in.capture_clock_us;
        material_.predecessor_contiguous=adjacent;
        material_.genuine_capture=true;
      }
      previous_=in;have_previous_=true;watermark_=in.capture_clock_us;gap_=false;
      return material_;
    }
    bool note_admitted(const certificate_t &c) noexcept {
      if(!c.valid()||c.source_epoch!=material_.source_epoch||c.domain!=material_.domain||
        c.material_serial!=material_.material_serial||c.capture_clock_us!=material_.capture_clock_us)return false;
      ++counters_.admitted;return true;
    }
    const certificate_t &current_material() const noexcept {return material_;}
    const counters_t &counters() const noexcept {return counters_;}
  };
} // namespace models::host_sbs_temporal_source_bridge

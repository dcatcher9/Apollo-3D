#pragma once
// PRIVATE source-material identity. Not model completion, frame admission, or UI authority.
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>

namespace models::research_material_capture_v1 {
  using clock_t = std::chrono::steady_clock;
  struct domain_t {
    std::uint32_t width=0, height=0, format=0, transfer=0, mip_levels=0, array_size=0, samples=0, quality=0;
    std::array<std::uint32_t,4> analysis_ltrb{}, content_ltrb{};
    std::uint32_t field_width=0, field_height=0, authority=0;
    std::uint64_t analysis_generation=0, context_generation=0;
    std::uint64_t root_authority_generation=0, region_authority_generation=0, browser_epoch=0;
    bool operator==(const domain_t &) const = default;
  };
  enum class status_e { unknown, first_material, unchanged_material, changed_material };
  struct input_t {
    std::uint64_t capture_epoch=0; // explicit converter/source namespace; never model generation
    domain_t domain{};
    std::optional<clock_t::time_point> source_stamp, content_stamp;
    // Caller must derive these from the existing SAME exact DDup history + analysis rectangle.
    // They are never inferred from conversion/model IDs or authored background/labels.
    bool exact_analysis_pixels_unchanged=false;
    bool complete_capture_interval_observed=false;
  };
  struct identity_t {
    status_e status=status_e::unknown;
    std::uint64_t epoch=0, material_serial=0, material_capture_clock_us=0;
    bool interval_contiguous=false;
    std::optional<clock_t::time_point> latest_capture_stamp, latest_content_stamp;
    bool usable() const noexcept { return status!=status_e::unknown; }
  };
  class tracker_t {
    std::uint64_t namespace_epoch_=0;
    std::optional<input_t> previous_;
    identity_t material_{};
    std::optional<clock_t::time_point> high_watermark_;
    static std::optional<std::uint64_t> clock(const clock_t::time_point &p) noexcept {
      const auto u=std::chrono::duration_cast<std::chrono::microseconds>(p.time_since_epoch()).count();
      if(u<0 || static_cast<std::uint64_t>(u)==std::numeric_limits<std::uint64_t>::max())return std::nullopt;
      return static_cast<std::uint64_t>(u)+1u; // zero remains invalid; actual stamp, no synthetic age
    }
  public:
    // Caller owns an explicitly NEW source pipeline/domain namespace, not a model/ACK retry.
    bool begin_source_namespace(std::uint64_t epoch) noexcept {
      if(epoch==0u || epoch<=namespace_epoch_)return false;
      namespace_epoch_=epoch;previous_.reset();material_={};high_watermark_.reset();return true;
    }
    identity_t observe(const input_t &in) noexcept {
      identity_t unknown{};
      if(!in.capture_epoch || !in.source_stamp || !in.domain.width || !in.domain.height || !in.domain.context_generation)return unknown;
      if(in.domain.authority>2u || (in.domain.authority==0u ? in.domain.analysis_generation!=0u : in.domain.analysis_generation==0u))return unknown;
      if(namespace_epoch_==0u)namespace_epoch_=in.capture_epoch;
      if(in.capture_epoch!=namespace_epoch_)return unknown;
      const auto own_clock=clock(*in.source_stamp);if(!own_clock)return unknown;
      if(high_watermark_ && *in.source_stamp<*high_watermark_)return unknown; // do not rewind even after conflict
      if(previous_ && (in.capture_epoch!=previous_->capture_epoch || in.domain!=previous_->domain))return unknown;
      if(previous_ && in.source_stamp==previous_->source_stamp && in.content_stamp!=previous_->content_stamp)return unknown;
      const bool same_capture=previous_ && in.source_stamp==previous_->source_stamp && in.content_stamp==previous_->content_stamp;
      const bool same_pixels=previous_ && in.exact_analysis_pixels_unchanged;
      if(previous_ && !same_capture && *own_clock<=material_.material_capture_clock_us)return unknown;
      if(!previous_) {
        material_.epoch=in.capture_epoch;material_.material_serial=1u;material_.material_capture_clock_us=*own_clock;
        material_.status=status_e::first_material;material_.interval_contiguous=false;
      } else if(same_capture || same_pixels) {
        // New desktop capture with a complete exact ROI-clean proof is the SAME material.
        // Its latest literal capture is exported separately, not relabelled as the old capture.
        material_.status=status_e::unchanged_material;
        material_.interval_contiguous=same_capture || in.complete_capture_interval_observed;
      } else {
        if(material_.material_serial==std::numeric_limits<std::uint64_t>::max())return unknown;
        ++material_.material_serial;material_.material_capture_clock_us=*own_clock;
        material_.status=status_e::changed_material;
        material_.interval_contiguous=in.complete_capture_interval_observed;
      }
      material_.latest_capture_stamp=in.source_stamp;material_.latest_content_stamp=in.content_stamp;
      previous_=in;high_watermark_=in.source_stamp;return material_;
    }
    const identity_t &last_material() const noexcept { return material_; }
  };
} // namespace models::research_material_capture_v1

#pragma once
// EXTERNAL C++20 RESEARCH POLICY V2. Not a depth/OCR reuse receipt or production path.
// The caller owns the newest current-colour texture (or already-drawn packed identity) and
// retains it across stale rejection until a newer capture is accepted or explicit pipeline
// reset retires it. Policy stores metadata only. reject_stale_colour NEVER permits drawing
// the rejected argument; keep/redraw the latest owned source named by the returned identity.
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>

namespace external_held_geometry_current_colour_v2 {
  using clock_t = std::chrono::steady_clock;
  using stamp_t = clock_t::time_point;
  inline constexpr auto max_owner_age = std::chrono::milliseconds {250};

  struct domain_t {
    std::uint32_t source_width {}, source_height {}, format {}, mip_levels {}, array_size {};
    std::uint32_t sample_count {}, sample_quality {}, transfer {};
    std::array<std::uint32_t, 4> analysis_ltrb {}, tensor_content_ltrb {};
    std::uint32_t field_width {}, field_height {}, analysis_authority {};
    std::uint64_t root_authority_generation {}, region_authority_generation {};
    std::uint64_t browser_authority_epoch {}, subtitle_context_generation {};
    bool operator==(const domain_t &) const = default;
  };
  struct geometry_owner_t {
    // Real expected_t::estimator_generation + publication_sequence namespace.
    std::uint64_t estimator_generation {}, publication_token {};
    std::uint64_t completed_colour_frame {}, actual_depth_owner_frame {};
    // Caller records this CPU conversion identity when the genuine input is submitted.
    // It is NOT the estimator frame or GPU depth-owner frame.
    std::uint64_t completed_colour_conversion_serial {};
    stamp_t completed_colour_source_stamp {}, accepted_capture_at {};
    domain_t domain {};
    std::uint32_t published_cut_epoch {};
    std::optional<std::uint64_t> GPU_base_observation_clock_us;
    bool authenticated_completion_and_renderer {};
    bool operator==(const geometry_owner_t &) const = default;
  };
  struct captured_colour_t {
    std::uint64_t conversion_serial {};
    stamp_t capture_source_stamp {}, observed_at {};
    domain_t domain {};
    bool source_stamp_present {}, current_texture_owned_until_draw_submission {};
  };
  enum class cut_evidence_e { unknown, no_new_published_epoch, known_changed };
  struct current_evidence_t {
    bool joined_root_busy {}, publication_still_current {}, structural_route_join_verified {};
    bool ordinary_exact_route_available {}, producer_terminal {}, move_size_or_reprocess {};
    cut_evidence_e current_cut = cut_evidence_e::unknown;
  };
  enum class route_e { ordinary_existing_policy, presentation_only_held_geometry_current_color, current_flat, reject_stale_colour };
  enum class alive_action_e { none, initialise_from_new_owner_then_withdraw, withdraw_only, hold_same_capture };
  struct presentation_receipt_t {
    route_e route = route_e::ordinary_existing_policy;
    alive_action_e alive_action = alive_action_e::none;
    std::uint64_t colour_conversion_serial {}, geometry_publication_token {}, estimator_generation {};
    std::uint64_t tuple_completed_colour_frame {}, actual_depth_owner_frame {}, geometry_completed_conversion_serial {};
    stamp_t colour_source_stamp {}, geometry_completed_colour_stamp {}, owner_deadline {};
    std::optional<std::uint64_t> GPU_base_observation_clock_us;
    cut_evidence_e current_cut = cut_evidence_e::unknown;
    std::uint64_t withdrawal_revision {};
    bool exact_depth_reuse = false, new_subtitle_observation = false;
  };

  class lane_t {
  public:
    // Only a genuinely newer real estimator generation can start another pipeline.
    // Caller must retire old owner/scratch/packed views and rebind current-colour ownership
    // before this reset. Never call it for domain changes, repeated receipts or revocation.
    // Numeric generation comes from the producer's monotonic process generation counter;
    // same/older/zero generations are rejected. No wrap/restart is silently accepted.
    bool reset_pipeline(std::uint64_t generation) noexcept {
      if (generation == 0 || generation <= pipeline_generation_) return false;
      pipeline_generation_ = generation;
      owner_.reset(); last_colour_.reset(); owner_available_ = false;
      active_ = false; initialise_ = false; revision_ = 0;
      return true;
    }

    // Retire this held epoch, but retain its one latest genuine publication and newest
    // source watermark. Re-delivering that retired publication cannot restore old M_alive.
    void revoke() noexcept { owner_available_ = false; active_ = false; initialise_ = false; }

    bool adopt_completed(const geometry_owner_t &owner) noexcept {
      if (!owner.authenticated_completion_and_renderer || pipeline_generation_ == 0 ||
          owner.estimator_generation != pipeline_generation_ || owner.publication_token == 0 ||
          owner.completed_colour_frame == 0 || owner.completed_colour_conversion_serial == 0 ||
          owner.accepted_capture_at == stamp_t {} || owner.completed_colour_source_stamp == stamp_t {}) return false;
      if (owner_) {
        if (owner.publication_token == owner_->publication_token) {
          if (owner == *owner_ && owner_available_) return true;
          if (!(owner == *owner_)) revoke();
          return false; // Identical retired owner is NOT a new completion.
        }
        if (owner.publication_token < owner_->publication_token ||
            owner.completed_colour_source_stamp < owner_->completed_colour_source_stamp ||
            owner.completed_colour_frame < owner_->completed_colour_frame ||
            owner.completed_colour_conversion_serial < owner_->completed_colour_conversion_serial) return false;
      }
      // Completion may lag the newest displayed colour even if source stamps coincide.
      // Keep that watermark; only a completion reaching both identities releases the latch.
      const bool reached_colour = !last_colour_ || (owner.domain == last_colour_->domain &&
        owner.completed_colour_source_stamp >= last_colour_->capture_source_stamp &&
        owner.completed_colour_conversion_serial >= last_colour_->conversion_serial);
      owner_ = owner; owner_available_ = true;
      active_ = !reached_colour; initialise_ = true;
      return true;
    }

    presentation_receipt_t choose(const captured_colour_t &colour,
                                  const current_evidence_t &evidence,
                                  bool explicitly_enabled_bounded_approximate_presentation) noexcept {
      presentation_receipt_t out;
      // MUST precede owner/ordinary early returns. Revocation does not make older B fresh.
      if (last_colour_ && (colour.conversion_serial < last_colour_->conversion_serial ||
          colour.capture_source_stamp < last_colour_->capture_source_stamp ||
          colour.observed_at < last_colour_->observed_at ||
          (colour.conversion_serial == last_colour_->conversion_serial &&
           colour.capture_source_stamp != last_colour_->capture_source_stamp))) {
        out.route = route_e::reject_stale_colour;
        out.colour_conversion_serial = last_colour_->conversion_serial;
        out.colour_source_stamp = last_colour_->capture_source_stamp;
        out.estimator_generation = pipeline_generation_;
        return out; // Keep/redraw latest owned colour; never draw the rejected argument.
      }
      if (!explicitly_enabled_bounded_approximate_presentation && active_) {
        remember(colour); revoke(); out.route = route_e::current_flat; return out;
      }
      if (!owner_available_ || !owner_) {
        if (last_colour_) { remember(colour); out.route = route_e::current_flat; }
        return out;
      }
      const auto owner = *owner_;
      if (evidence.ordinary_exact_route_available && !active_) { remember(colour); return out; }
      if (!evidence.joined_root_busy && !active_) { remember(colour); return out; }
      const bool domain_ok = evidence.publication_still_current && evidence.structural_route_join_verified &&
        colour.source_stamp_present && colour.current_texture_owned_until_draw_submission &&
        colour.conversion_serial != 0 && !evidence.producer_terminal && !evidence.move_size_or_reprocess &&
        evidence.current_cut != cut_evidence_e::known_changed && colour.domain == owner.domain &&
        colour.capture_source_stamp >= owner.completed_colour_source_stamp &&
        colour.conversion_serial >= owner.completed_colour_conversion_serial;
      const auto age = colour.observed_at - owner.accepted_capture_at;
      if (!domain_ok || age < clock_t::duration::zero() || age > max_owner_age) {
        remember(colour); revoke(); out.route = route_e::current_flat; return out;
      }
      const bool changed = colour.capture_source_stamp != owner.completed_colour_source_stamp ||
        colour.conversion_serial != owner.completed_colour_conversion_serial;
      if (!explicitly_enabled_bounded_approximate_presentation || (!changed && !active_)) { remember(colour); return out; }
      const bool same_capture = last_colour_ &&
        colour.capture_source_stamp == last_colour_->capture_source_stamp &&
        colour.conversion_serial == last_colour_->conversion_serial && colour.domain == last_colour_->domain;
      out.route = route_e::presentation_only_held_geometry_current_color;
      out.alive_action = initialise_ ? alive_action_e::initialise_from_new_owner_then_withdraw :
        same_capture ? alive_action_e::hold_same_capture : alive_action_e::withdraw_only;
      if (out.alive_action != alive_action_e::hold_same_capture) ++revision_;
      out.colour_conversion_serial = colour.conversion_serial;
      out.colour_source_stamp = colour.capture_source_stamp;
      out.geometry_publication_token = owner.publication_token;
      out.estimator_generation = owner.estimator_generation;
      out.tuple_completed_colour_frame = owner.completed_colour_frame;
      out.actual_depth_owner_frame = owner.actual_depth_owner_frame;
      out.geometry_completed_conversion_serial = owner.completed_colour_conversion_serial;
      out.geometry_completed_colour_stamp = owner.completed_colour_source_stamp;
      out.GPU_base_observation_clock_us = owner.GPU_base_observation_clock_us;
      out.owner_deadline = owner.accepted_capture_at + max_owner_age;
      out.current_cut = evidence.current_cut; out.withdrawal_revision = revision_;
      active_ = true; initialise_ = false; remember(colour);
      return out;
    }

    static bool same_presentation_pixels(const presentation_receipt_t &packed,
                                         const presentation_receipt_t &current) noexcept {
      return packed.route == route_e::presentation_only_held_geometry_current_color &&
        current.route == packed.route && packed.colour_source_stamp == current.colour_source_stamp &&
        packed.colour_conversion_serial == current.colour_conversion_serial &&
        packed.estimator_generation == current.estimator_generation &&
        packed.geometry_publication_token == current.geometry_publication_token &&
        packed.withdrawal_revision == current.withdrawal_revision && packed.owner_deadline == current.owner_deadline;
    }
  private:
    void remember(const captured_colour_t &colour) noexcept {
      if (colour.source_stamp_present && colour.current_texture_owned_until_draw_submission &&
          colour.conversion_serial != 0 && colour.capture_source_stamp != stamp_t {}) last_colour_ = colour;
    }
    std::uint64_t pipeline_generation_ = 0;
    std::optional<geometry_owner_t> owner_; // Latest genuine publication, even when retired.
    std::optional<captured_colour_t> last_colour_; // Latest accepted presentation source identity.
    bool owner_available_ = false, active_ = false, initialise_ = false;
    std::uint64_t revision_ = 0;
  };
}

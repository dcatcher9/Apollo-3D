/**
 * @file src/sbs_roi_tracker.cpp
 * @brief Deterministic candidate selection and temporal tracking for Host SBS content ROIs.
 */
#include "sbs_roi_tracker.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace sbs_roi {
  namespace {
    constexpr normalized_rect_t full_frame_rect {};
    constexpr std::size_t maximum_candidate_budget = 1024;

    bool finite(float value) {
      return std::isfinite(value);
    }

    float area(const normalized_rect_t &rect) {
      return (rect.right - rect.left) * (rect.bottom - rect.top);
    }

    bool valid_rect(const normalized_rect_t &rect) {
      return finite(rect.left) &&
             finite(rect.top) &&
             finite(rect.right) &&
             finite(rect.bottom) &&
             rect.left >= 0.0f &&
             rect.top >= 0.0f &&
             rect.right <= 1.0f &&
             rect.bottom <= 1.0f &&
             rect.left < rect.right &&
             rect.top < rect.bottom;
    }

    bool valid_fraction(float value) {
      return finite(value) && value >= 0.0f && value <= 1.0f;
    }

    float intersection_over_union(const normalized_rect_t &lhs, const normalized_rect_t &rhs) {
      const auto intersection_width =
        std::max(0.0f, std::min(lhs.right, rhs.right) - std::max(lhs.left, rhs.left));
      const auto intersection_height =
        std::max(0.0f, std::min(lhs.bottom, rhs.bottom) - std::max(lhs.top, rhs.top));
      const auto intersection = intersection_width * intersection_height;
      const auto union_area = area(lhs) + area(rhs) - intersection;
      return union_area > 0.0f ? intersection / union_area : 0.0f;
    }

    bool same_candidate(
      roi_kind_e lhs_kind,
      const normalized_rect_t &lhs,
      roi_kind_e rhs_kind,
      const normalized_rect_t &rhs,
      float association_iou
    ) {
      return lhs_kind == rhs_kind &&
             intersection_over_union(lhs, rhs) >= association_iou;
    }

    bool canonical_less(const candidate_t &lhs, const candidate_t &rhs) {
      return std::tie(
               lhs.kind,
               lhs.rect.left,
               lhs.rect.top,
               lhs.rect.right,
               lhs.rect.bottom
             ) <
             std::tie(
               rhs.kind,
               rhs.rect.left,
               rhs.rect.top,
               rhs.rect.right,
               rhs.rect.bottom
             );
    }

    bool interval_mature(
      std::uint32_t updates,
      std::uint32_t required_updates,
      std::uint64_t first_timestamp_us,
      std::uint64_t timestamp_us,
      std::uint64_t required_duration_us
    ) {
      return updates >= required_updates &&
             timestamp_us >= first_timestamp_us &&
             timestamp_us - first_timestamp_us >= required_duration_us;
    }

    void validate_config(const tracker_config_t &config) {
      if (config.max_candidates == 0 || config.max_candidates > maximum_candidate_budget || config.acquire_updates == 0 || config.challenger_updates == 0 || config.release_updates == 0 || config.scroll_enter_updates == 0 || config.scroll_exit_updates == 0 || config.scroll_reacquire_timeout_updates == 0 || config.invalid_input_timeout_updates == 0 || config.weak_retention_grace_updates == 0) {
        throw std::invalid_argument("SBS ROI tracker candidate and update budgets must be bounded and non-zero");
      }

      if (config.max_continuity_gap_us == 0 || config.acquire_duration_us == 0 || config.challenger_duration_us == 0 || config.release_duration_us == 0 || config.scroll_enter_duration_us == 0 || config.scroll_exit_duration_us == 0 || config.scroll_reacquire_timeout_us == 0 || config.invalid_input_timeout_us == 0 || config.weak_retention_grace_us == 0) {
        throw std::invalid_argument("SBS ROI tracker timing windows must be non-zero");
      }

      const float fractions[] {
        config.association_iou,
        config.stable_geometry_iou,
        config.min_video_area,
        config.min_content_area,
        config.min_video_temporal_occupancy,
        config.min_video_photographic_density,
        config.min_content_photographic_density,
        config.min_retention_evidence,
        config.min_primary_column_support,
        config.min_gutter_confidence,
        config.interaction_min_score_ratio,
        config.identity_area,
        config.identity_edge_margin,
      };
      if (std::any_of(std::begin(fractions), std::end(fractions), [](float value) {
            return !valid_fraction(value);
          }) ||
          config.association_iou <= 0.0f || config.stable_geometry_iou < config.association_iou || config.min_retention_evidence <= 0.0f || config.interaction_min_score_ratio <= 0.0f || config.identity_area < 0.50f || config.identity_edge_margin > 0.25f) {
        throw std::invalid_argument("SBS ROI tracker fractions are outside their safe ranges");
      }

      const float ratios[] {
        config.video_winner_ratio,
        config.content_winner_ratio,
        config.challenger_ratio,
      };
      if (std::any_of(std::begin(ratios), std::end(ratios), [](float value) {
            return !finite(value) || value <= 1.0f;
          })) {
        throw std::invalid_argument("SBS ROI tracker dominance ratios must be finite and greater than one");
      }
    }

    bool hard_eligible(const candidate_t &candidate, const tracker_config_t &config) {
      if (!candidate.inside_primary_column || candidate.crosses_stable_gutter || candidate.primary_column_support < config.min_primary_column_support || candidate.gutter_confidence < config.min_gutter_confidence) {
        return false;
      }

      const auto candidate_area = area(candidate.rect);
      if (candidate.kind == roi_kind_e::video) {
        return candidate_area >= config.min_video_area;
      }
      if (candidate.kind == roi_kind_e::content) {
        return candidate_area >= config.min_content_area;
      }
      return false;
    }

    bool acquisition_eligible(const candidate_t &candidate, const tracker_config_t &config) {
      if (!hard_eligible(candidate, config)) {
        return false;
      }

      // A lone, static photographic rectangle attached to the outer edge is
      // more likely to be an advertisement or navigation rail than the user's
      // intended content.  Keep this deliberately narrow: a clicked target,
      // moving media, a central collage, and near-fullscreen content all remain
      // eligible.
      const auto candidate_area = area(candidate.rect);
      const auto edge_limit = 0.40f;
      const auto identity_geometry =
        candidate_area >= config.identity_area &&
        candidate.rect.left <= config.identity_edge_margin &&
        candidate.rect.top <= config.identity_edge_margin &&
        candidate.rect.right >=
          1.0f - config.identity_edge_margin &&
        candidate.rect.bottom >=
          1.0f - config.identity_edge_margin;
      const auto static_edge_rail =
        candidate.kind == roi_kind_e::content &&
        !candidate.recent_interaction &&
        candidate.temporal_occupancy <
          config.min_video_temporal_occupancy &&
        candidate_area >= config.min_video_area &&
        !identity_geometry &&
        (
          (candidate.rect.left <= config.identity_edge_margin &&
           candidate.rect.right <= edge_limit) ||
          (candidate.rect.right >= 1.0f - config.identity_edge_margin &&
           candidate.rect.left >= 1.0f - edge_limit) ||
          (candidate.rect.top <= config.identity_edge_margin &&
           candidate.rect.bottom <= edge_limit) ||
          (candidate.rect.bottom >=
             1.0f - config.identity_edge_margin &&
           candidate.rect.top >= 1.0f - edge_limit)
        );
      if (static_edge_rail) {
        return false;
      }

      if (candidate.kind == roi_kind_e::video) {
        return candidate.temporal_occupancy >= config.min_video_temporal_occupancy &&
               candidate.photographic_density >= config.min_video_photographic_density;
      }
      return candidate.photographic_density >= config.min_content_photographic_density;
    }

    bool active_as(
      const candidate_t &candidate,
      roi_kind_e committed_kind,
      const tracker_config_t &config
    ) {
      if (committed_kind == roi_kind_e::video) {
        return candidate.temporal_occupancy >= config.min_video_temporal_occupancy &&
               candidate.photographic_density >= config.min_video_photographic_density;
      }
      return candidate.photographic_density >= config.min_content_photographic_density;
    }

    bool retention_eligible(const candidate_t &candidate, const tracker_config_t &config) {
      return std::max(candidate.temporal_occupancy, candidate.photographic_density) >=
             config.min_retention_evidence;
    }

    bool strong_retention(
      const candidate_t &candidate,
      roi_kind_e committed_kind,
      const tracker_config_t &config
    ) {
      return candidate.kind == committed_kind &&
             hard_eligible(candidate, config);
    }

    float candidate_score(const candidate_t &candidate) {
      const auto candidate_area = area(candidate.rect);
      if (candidate.kind == roi_kind_e::video) {
        const auto evidence =
          0.35f +
          0.30f * candidate.temporal_occupancy +
          0.15f * candidate.photographic_density +
          0.10f * candidate.primary_column_support +
          0.10f * candidate.gutter_confidence;
        return candidate_area * evidence;
      }

      const auto evidence =
        0.50f +
        0.30f * candidate.photographic_density +
        0.10f * candidate.primary_column_support +
        0.10f * candidate.gutter_confidence;
      return candidate_area * evidence;
    }

    bool near_identity(const candidate_t &candidate, const tracker_config_t &config) {
      return area(candidate.rect) >= config.identity_area &&
             candidate.rect.left <= config.identity_edge_margin &&
             candidate.rect.top <= config.identity_edge_margin &&
             candidate.rect.right >= 1.0f - config.identity_edge_margin &&
             candidate.rect.bottom >= 1.0f - config.identity_edge_margin;
    }
  }  // namespace

  tracker_t::tracker_t(tracker_config_t config):
      config_(std::move(config)) {
    validate_config(config_);
  }

  bool tracker_t::valid_candidates(const observation_t &observation) const {
    if (observation.candidates.size() > config_.max_candidates) {
      return false;
    }
    return std::all_of(
      observation.candidates.begin(),
      observation.candidates.end(),
      [](const candidate_t &candidate) {
        return (candidate.kind == roi_kind_e::video ||
                candidate.kind == roi_kind_e::content) &&
               valid_rect(candidate.rect) &&
               valid_fraction(candidate.temporal_occupancy) &&
               valid_fraction(candidate.photographic_density) &&
               valid_fraction(candidate.primary_column_support) &&
               valid_fraction(candidate.gutter_confidence);
      }
    );
  }

  std::optional<tracker_t::scored_candidate_t> tracker_t::select_candidate(
    const observation_t &observation
  ) const {
    const auto select_kind = [&](roi_kind_e kind, float winner_ratio) -> std::optional<scored_candidate_t> {
      std::vector<scored_candidate_t> candidates;
      for (const auto &candidate : observation.candidates) {
        if (candidate.kind == kind && acquisition_eligible(candidate, config_)) {
          candidates.push_back({candidate, candidate_score(candidate)});
        }
      }
      if (candidates.empty()) {
        return std::nullopt;
      }

      std::sort(
        candidates.begin(),
        candidates.end(),
        [](const scored_candidate_t &lhs, const scored_candidate_t &rhs) {
          if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
          }
          return canonical_less(lhs.candidate, rhs.candidate);
        }
      );

      // The feature extractor may emit several nearby bounds for one physical target. Collapse
      // direct spatial duplicates before applying the semantic winner ratio so they do not
      // compete with themselves. Representatives are constructed from a deterministic sort.
      std::vector<scored_candidate_t> representatives;
      for (const auto &candidate : candidates) {
        const auto duplicate = std::find_if(
          representatives.begin(),
          representatives.end(),
          [&](const scored_candidate_t &representative) {
            return same_candidate(
              candidate.candidate.kind,
              candidate.candidate.rect,
              representative.candidate.kind,
              representative.candidate.rect,
              config_.stable_geometry_iou
            );
          }
        );
        if (duplicate == representatives.end()) {
          representatives.push_back(candidate);
        } else {
          duplicate->candidate.recent_interaction =
            duplicate->candidate.recent_interaction ||
            candidate.candidate.recent_interaction;
        }
      }

      if (representatives.size() == 1) {
        return representatives.front();
      }

      const auto &best = representatives[0];
      const auto &runner_up = representatives[1];
      if (best.score >= runner_up.score * winner_ratio) {
        return best;
      }

      const auto clicked = std::find_if(
        representatives.begin(),
        representatives.end(),
        [](const scored_candidate_t &candidate) {
          return candidate.candidate.recent_interaction;
        }
      );
      if (clicked == representatives.end()) {
        return std::nullopt;
      }
      const auto another_clicked = std::find_if(
        std::next(clicked),
        representatives.end(),
        [](const scored_candidate_t &candidate) {
          return candidate.candidate.recent_interaction;
        }
      );
      if (another_clicked != representatives.end() || clicked->score < best.score * config_.interaction_min_score_ratio) {
        return std::nullopt;
      }
      return *clicked;
    };

    // Resolve ambiguity independently within each semantic class, then compare
    // the surviving class winners by their common score.  This preserves the
    // stricter multi-video ambiguity contract without granting every eligible
    // autoplay/video proposal unconditional priority over a stronger content
    // region.
    const auto video =
      select_kind(roi_kind_e::video, config_.video_winner_ratio);
    const auto content =
      select_kind(roi_kind_e::content, config_.content_winner_ratio);
    if (!video) {
      return content;
    }
    if (!content) {
      return video;
    }
    if (intersection_over_union(
          video->candidate.rect,
          content->candidate.rect
        ) >= config_.stable_geometry_iou) {
      // The detector intentionally emits both semantic views for some moving
      // photographic regions.  Equivalent geometry is one physical target,
      // not a cross-kind competition; retain the richer video identity.
      return video;
    }
    if (video->score != content->score) {
      return video->score > content->score ? video : content;
    }
    return std::nullopt;
  }

  std::optional<tracker_t::scored_candidate_t> tracker_t::find_incumbent(
    const observation_t &observation
  ) const {
    std::optional<scored_candidate_t> incumbent;
    for (const auto &candidate : observation.candidates) {
      if (intersection_over_union(candidate.rect, committed_rect_) <
          config_.association_iou) {
        continue;
      }

      scored_candidate_t scored {candidate, candidate_score(candidate)};
      const bool scored_is_strong =
        strong_retention(scored.candidate, committed_kind_, config_);
      const bool incumbent_is_strong =
        incumbent &&
        strong_retention(
          incumbent->candidate,
          committed_kind_,
          config_
        );
      if (!incumbent || (scored_is_strong && !incumbent_is_strong) || (scored_is_strong == incumbent_is_strong && (scored.score > incumbent->score || (scored.score == incumbent->score && canonical_less(scored.candidate, incumbent->candidate))))) {
        incumbent = std::move(scored);
      }
    }
    return incumbent;
  }

  void tracker_t::track_acquisition(
    const scored_candidate_t &candidate,
    std::uint64_t timestamp_us
  ) {
    if (acquisition_ && same_candidate(acquisition_->value.candidate.kind, acquisition_->anchor_rect, candidate.candidate.kind, candidate.candidate.rect, config_.stable_geometry_iou) && same_candidate(acquisition_->value.candidate.kind, acquisition_->value.candidate.rect, candidate.candidate.kind, candidate.candidate.rect, config_.stable_geometry_iou)) {
      acquisition_->value = candidate;
      if (acquisition_->updates < std::numeric_limits<std::uint32_t>::max()) {
        ++acquisition_->updates;
      }
    } else {
      acquisition_ = tracked_candidate_t {
        candidate,
        candidate.candidate.rect,
        1,
        timestamp_us,
      };
    }
  }

  void tracker_t::track_challenger(
    const scored_candidate_t &candidate,
    std::uint64_t timestamp_us
  ) {
    if (challenger_ && same_candidate(challenger_->value.candidate.kind, challenger_->anchor_rect, candidate.candidate.kind, candidate.candidate.rect, config_.stable_geometry_iou) && same_candidate(challenger_->value.candidate.kind, challenger_->value.candidate.rect, candidate.candidate.kind, candidate.candidate.rect, config_.stable_geometry_iou)) {
      challenger_->value = candidate;
      if (challenger_->updates < std::numeric_limits<std::uint32_t>::max()) {
        ++challenger_->updates;
      }
    } else {
      challenger_ = tracked_candidate_t {
        candidate,
        candidate.candidate.rect,
        1,
        timestamp_us,
      };
    }
  }

  bool tracker_t::acquisition_mature(std::uint64_t timestamp_us) const {
    return acquisition_ &&
           interval_mature(
             acquisition_->updates,
             config_.acquire_updates,
             acquisition_->first_timestamp_us,
             timestamp_us,
             config_.acquire_duration_us
           );
  }

  bool tracker_t::challenger_mature(std::uint64_t timestamp_us) const {
    return challenger_ &&
           interval_mature(
             challenger_->updates,
             config_.challenger_updates,
             challenger_->first_timestamp_us,
             timestamp_us,
             config_.challenger_duration_us
           );
  }

  void tracker_t::clear_search_state() {
    acquisition_.reset();
    challenger_.reset();
    missing_updates_ = 0;
    missing_since_us_.reset();
    weak_retention_updates_ = 0;
    weak_retention_since_us_.reset();
  }

  void tracker_t::clear_scroll_state() {
    scroll_updates_ = 0;
    rest_updates_ = 0;
    scroll_reacquire_updates_ = 0;
    scroll_since_us_.reset();
    rest_since_us_.reset();
    scroll_reacquire_since_us_.reset();
    scroll_hold_ = false;
    scroll_reacquiring_ = false;
  }

  bool tracker_t::commit(
    const scored_candidate_t &candidate,
    std::uint64_t observation_id
  ) {
    if (near_identity(candidate.candidate, config_)) {
      return commit_full_frame(observation_id, false);
    }

    committed_kind_ = candidate.candidate.kind;
    committed_rect_ = candidate.candidate.rect;
    committed_confidence_ = std::clamp(candidate.score, 0.0f, 1.0f);
    committed_observation_id_ = observation_id;
    committed_age_updates_ = 1;
    if (generation_ < std::numeric_limits<std::uint64_t>::max()) {
      ++generation_;
    }
    clear_search_state();
    return true;
  }

  bool tracker_t::commit_full_frame(
    std::uint64_t observation_id,
    bool force_generation
  ) {
    const bool changed = committed_kind_ != roi_kind_e::full_frame;
    if (!changed && !force_generation) {
      clear_search_state();
      return false;
    }

    committed_kind_ = roi_kind_e::full_frame;
    committed_rect_ = full_frame_rect;
    committed_confidence_ = 0.0f;
    committed_observation_id_ = observation_id;
    committed_age_updates_ = 1;
    if (generation_ < std::numeric_limits<std::uint64_t>::max()) {
      ++generation_;
    }
    clear_search_state();
    return true;
  }

  tracker_output_t tracker_t::output(
    update_status_e status,
    bool committed_this_update
  ) const {
    tracker_output_t result;
    result.kind = scroll_hold_ ? roi_kind_e::scroll_hold : committed_kind_;
    result.rect = committed_rect_;
    result.confidence = committed_confidence_;
    result.generation = generation_;
    result.observation_id = last_observation_id_;
    result.committed_observation_id = committed_observation_id_;
    result.age_updates = committed_age_updates_;
    result.committed_this_update = committed_this_update;
    result.status = status;
    return result;
  }

  tracker_output_t tracker_t::snapshot() const {
    return output();
  }

  tracker_output_t tracker_t::handle_invalid_input(
    std::uint64_t arrival_timestamp_us,
    update_status_e status
  ) {
    clear_search_state();
    rest_updates_ = 0;
    rest_since_us_.reset();
    scroll_reacquiring_ = false;
    scroll_reacquire_updates_ = 0;
    scroll_reacquire_since_us_.reset();
    if (!scroll_hold_) {
      scroll_updates_ = 0;
      scroll_since_us_.reset();
    }
    if (invalid_updates_ == 0) {
      invalid_since_us_ = arrival_timestamp_us;
    }
    if (invalid_updates_ < std::numeric_limits<std::uint32_t>::max()) {
      ++invalid_updates_;
    }
    if (!invalid_fail_safe_active_ && invalid_since_us_ && interval_mature(invalid_updates_, config_.invalid_input_timeout_updates, *invalid_since_us_, arrival_timestamp_us, config_.invalid_input_timeout_us)) {
      clear_scroll_state();
      const bool committed =
        commit_full_frame(last_observation_id_, true);
      invalid_fail_safe_active_ = true;
      return output(
        update_status_e::invalidated_invalid_input,
        committed
      );
    }
    return output(status);
  }

  tracker_output_t tracker_t::detector_unavailable(
    std::uint64_t arrival_timestamp_us
  ) {
    if (arrival_timestamp_us == 0) {
      return output(update_status_e::ignored_invalid);
    }
    if (arrival_timestamp_us <= last_arrival_timestamp_us_) {
      return output(update_status_e::ignored_stale);
    }
    last_arrival_timestamp_us_ = arrival_timestamp_us;
    return handle_invalid_input(
      arrival_timestamp_us,
      update_status_e::ignored_invalid
    );
  }

  tracker_output_t tracker_t::update(const observation_t &observation) {
    if (observation.arrival_timestamp_us == 0) {
      return output(update_status_e::ignored_invalid);
    }
    if (observation.arrival_timestamp_us <= last_arrival_timestamp_us_) {
      return output(update_status_e::ignored_stale);
    }
    const bool continuity_broken =
      last_arrival_timestamp_us_ != 0 &&
      observation.arrival_timestamp_us -
          last_arrival_timestamp_us_ >
        config_.max_continuity_gap_us;
    last_arrival_timestamp_us_ = observation.arrival_timestamp_us;

    const bool metadata_fresh =
      observation.id != 0 &&
      observation.timestamp_us != 0 &&
      observation.id > last_observation_id_ &&
      (last_timestamp_us_ == 0 ||
       observation.timestamp_us > last_timestamp_us_);

    // Display geometry, HDR, and page replacement carry a monotonic epoch.
    // The control event remains authoritative even when detector metadata or
    // candidates are malformed, and adjacent distinct events cannot collapse.
    if (observation.invalidation_epoch != 0) {
      if (metadata_fresh) {
        last_observation_id_ = observation.id;
        last_timestamp_us_ = observation.timestamp_us;
      }
      if (observation.invalidation_epoch > last_invalidation_epoch_) {
        last_invalidation_epoch_ = observation.invalidation_epoch;
        invalid_updates_ = 0;
        invalid_since_us_.reset();
        invalid_fail_safe_active_ = false;
        clear_scroll_state();
        const bool committed =
          commit_full_frame(last_observation_id_, true);
        return output(update_status_e::accepted, committed);
      }
      return output(
        observation.invalidation_epoch ==
            last_invalidation_epoch_ ?
          update_status_e::accepted :
          update_status_e::ignored_stale
      );
    }

    if (!metadata_fresh) {
      const auto status =
        observation.id == 0 || observation.timestamp_us == 0 ?
          update_status_e::ignored_invalid :
          update_status_e::ignored_stale;
      if (status == update_status_e::ignored_stale) {
        return output(status);
      }
      return handle_invalid_input(
        observation.arrival_timestamp_us,
        status
      );
    }

    if (!valid_candidates(observation)) {
      return handle_invalid_input(
        observation.arrival_timestamp_us,
        update_status_e::ignored_invalid
      );
    }

    last_observation_id_ = observation.id;
    last_timestamp_us_ = observation.timestamp_us;
    if (continuity_broken) {
      clear_search_state();
      scroll_updates_ = 0;
      scroll_since_us_.reset();
      rest_updates_ = 0;
      rest_since_us_.reset();
      scroll_reacquiring_ = false;
      scroll_reacquire_updates_ = 0;
      scroll_reacquire_since_us_.reset();
      invalid_updates_ = 0;
      invalid_since_us_.reset();
    }

    invalid_updates_ = 0;
    invalid_since_us_.reset();
    invalid_fail_safe_active_ = false;

    // Broad page motion is authoritative only after the complete observation
    // envelope has passed identity, timestamp, arrival-order, and candidate
    // validation.  A stale or malformed packet carrying this bit must never
    // advance scroll dwell or clear otherwise valid acquisition state.
    if (observation.broad_page_scroll) {
      clear_search_state();
      rest_updates_ = 0;
      rest_since_us_.reset();
      scroll_reacquiring_ = false;
      scroll_reacquire_updates_ = 0;
      scroll_reacquire_since_us_.reset();
      if (scroll_updates_ == 0) {
        scroll_since_us_ = observation.arrival_timestamp_us;
      }
      if (scroll_updates_ < std::numeric_limits<std::uint32_t>::max()) {
        ++scroll_updates_;
      }
      if (scroll_since_us_ && interval_mature(scroll_updates_, config_.scroll_enter_updates, *scroll_since_us_, observation.arrival_timestamp_us, config_.scroll_enter_duration_us)) {
        scroll_hold_ = true;
      }
      return output();
    }

    if (!scroll_hold_) {
      scroll_updates_ = 0;
      scroll_since_us_.reset();
    } else {
      if (!scroll_reacquiring_) {
        if (rest_updates_ == 0) {
          rest_since_us_ = observation.arrival_timestamp_us;
        }
        if (rest_updates_ < std::numeric_limits<std::uint32_t>::max()) {
          ++rest_updates_;
        }
        if (!rest_since_us_ || !interval_mature(rest_updates_, config_.scroll_exit_updates, *rest_since_us_, observation.arrival_timestamp_us, config_.scroll_exit_duration_us)) {
          return output();
        }
        scroll_reacquiring_ = true;
        scroll_reacquire_updates_ = 0;
        scroll_reacquire_since_us_ =
          observation.arrival_timestamp_us;
        acquisition_.reset();
      }

      if (scroll_reacquire_updates_ < std::numeric_limits<std::uint32_t>::max()) {
        ++scroll_reacquire_updates_;
      }
      const auto selected = select_candidate(observation);
      if (selected) {
        track_acquisition(
          *selected,
          observation.arrival_timestamp_us
        );
        if (acquisition_mature(observation.arrival_timestamp_us)) {
          const auto candidate = acquisition_->value;
          clear_scroll_state();
          const bool committed =
            near_identity(candidate.candidate, config_) ?
              commit_full_frame(observation.id, true) :
              commit(candidate, observation.id);
          return output(update_status_e::accepted, committed);
        }
      } else {
        acquisition_.reset();
      }

      if (scroll_reacquire_since_us_ && interval_mature(scroll_reacquire_updates_, config_.scroll_reacquire_timeout_updates, *scroll_reacquire_since_us_, observation.arrival_timestamp_us, config_.scroll_reacquire_timeout_us)) {
        clear_scroll_state();
        const bool committed =
          commit_full_frame(observation.id, true);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }
    rest_updates_ = 0;
    rest_since_us_.reset();

    if (committed_age_updates_ < std::numeric_limits<std::uint32_t>::max()) {
      ++committed_age_updates_;
    }

    const auto selected = select_candidate(observation);
    if (committed_kind_ == roi_kind_e::full_frame) {
      if (!selected || near_identity(selected->candidate, config_)) {
        acquisition_.reset();
        return output();
      }
      track_acquisition(
        *selected,
        observation.arrival_timestamp_us
      );
      if (acquisition_mature(observation.arrival_timestamp_us)) {
        const auto candidate = acquisition_->value;
        const bool committed = commit(candidate, observation.id);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }

    const auto incumbent = find_incumbent(observation);
    const bool incumbent_missing = !incumbent;
    bool weak_release_due = false;
    const bool weak_incumbent =
      incumbent &&
      (
        !retention_eligible(incumbent->candidate, config_) ||
        !strong_retention(
          incumbent->candidate,
          committed_kind_,
          config_
        ) ||
        !active_as(
          incumbent->candidate,
          committed_kind_,
          config_
        )
      );
    if (weak_incumbent) {
      if (weak_retention_updates_ == 0) {
        weak_retention_since_us_ =
          observation.arrival_timestamp_us;
      }
      if (weak_retention_updates_ < std::numeric_limits<std::uint32_t>::max()) {
        ++weak_retention_updates_;
      }
      const bool weak_grace_mature =
        weak_retention_since_us_ &&
        interval_mature(
          weak_retention_updates_,
          config_.weak_retention_grace_updates,
          *weak_retention_since_us_,
          observation.arrival_timestamp_us,
          config_.weak_retention_grace_us
        );
      if (weak_grace_mature) {
        if (missing_updates_ == 0) {
          missing_since_us_ = observation.arrival_timestamp_us;
        }
        if (missing_updates_ < std::numeric_limits<std::uint32_t>::max()) {
          ++missing_updates_;
        }
        weak_release_due =
          missing_since_us_ &&
          interval_mature(
            missing_updates_,
            config_.release_updates,
            *missing_since_us_,
            observation.arrival_timestamp_us,
            config_.release_duration_us
          );
      } else {
        missing_updates_ = 0;
        missing_since_us_.reset();
      }
    } else {
      weak_retention_updates_ = 0;
      weak_retention_since_us_.reset();
      if (!incumbent_missing) {
        missing_updates_ = 0;
        missing_since_us_.reset();
      }
    }

    if (incumbent_missing) {
      weak_retention_updates_ = 0;
      weak_retention_since_us_.reset();
      if (missing_updates_ == 0) {
        missing_since_us_ = observation.arrival_timestamp_us;
      }
      if (missing_updates_ < std::numeric_limits<std::uint32_t>::max()) {
        ++missing_updates_;
      }

      if (selected) {
        track_challenger(
          *selected,
          observation.arrival_timestamp_us
        );
        const bool absence_confirmed =
          missing_since_us_ &&
          interval_mature(
            missing_updates_,
            config_.acquire_updates,
            *missing_since_us_,
            observation.arrival_timestamp_us,
            config_.acquire_duration_us
          );
        if (absence_confirmed && challenger_mature(observation.arrival_timestamp_us)) {
          const auto candidate = challenger_->value;
          const bool committed = commit(candidate, observation.id);
          return output(update_status_e::accepted, committed);
        }
      } else {
        challenger_.reset();
      }

      if (missing_since_us_ && interval_mature(missing_updates_, config_.release_updates, *missing_since_us_, observation.arrival_timestamp_us, config_.release_duration_us)) {
        const bool committed =
          commit_full_frame(observation.id, false);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }

    if (!selected) {
      challenger_.reset();
      if (weak_release_due) {
        const bool committed =
          commit_full_frame(observation.id, false);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }

    const bool same_identity =
      same_candidate(
        selected->candidate.kind,
        selected->candidate.rect,
        committed_kind_,
        committed_rect_,
        config_.association_iou
      );
    const bool equivalent_geometry =
      intersection_over_union(
        selected->candidate.rect,
        committed_rect_
      ) >= config_.stable_geometry_iou;
    if (!same_identity && equivalent_geometry) {
      // Video/content proposals can describe the same physical rectangle as
      // motion comes and goes.  A semantic reclassification alone is not a
      // challenger and must not churn the committed ROI generation.
      challenger_.reset();
      if (weak_release_due) {
        const bool committed =
          commit_full_frame(observation.id, false);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }
    if (same_identity) {
      if (equivalent_geometry) {
        challenger_.reset();
        return output();
      }

      track_challenger(
        *selected,
        observation.arrival_timestamp_us
      );
      if (challenger_mature(observation.arrival_timestamp_us)) {
        const auto candidate = challenger_->value;
        const bool committed = commit(candidate, observation.id);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }

    const bool decisive =
      selected->score >=
      incumbent->score * config_.challenger_ratio;

    if (!decisive) {
      challenger_.reset();
      if (weak_release_due) {
        const bool committed =
          commit_full_frame(observation.id, false);
        return output(update_status_e::accepted, committed);
      }
      return output();
    }

    track_challenger(
      *selected,
      observation.arrival_timestamp_us
    );
    if (challenger_mature(observation.arrival_timestamp_us)) {
      const auto candidate = challenger_->value;
      const bool committed = commit(candidate, observation.id);
      return output(update_status_e::accepted, committed);
    }
    if (weak_release_due) {
      const bool committed =
        commit_full_frame(observation.id, false);
      return output(update_status_e::accepted, committed);
    }
    return output();
  }
}  // namespace sbs_roi

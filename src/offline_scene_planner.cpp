/**
 * @file src/offline_scene_planner.cpp
 * @brief Bounded-lookahead scene planning for native offline SBS conversion.
 */
#include "offline_scene_planner.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <sstream>
#include <utility>

namespace offline_sbs {
  namespace {
    // The production thresholds were calibrated in depth-update counts at
    // 30 Hz. Map timestamped offline sources onto that reference clock so the
    // same flash/lookahead/settling duration behaves consistently at 24, 60,
    // 120, and variable frame rates. Tests without timestamps retain the
    // original one-update-per-ordinal fallback.
    constexpr double scene_reference_updates_per_second = 30.0;

    constexpr float depth_cut_high = 0.60f;
    constexpr float depth_cut_corroborate = 0.25f;
    constexpr float raw_rgb_cut_high = 0.70f;
    constexpr float structural_color_cut_high = 0.03f;
    constexpr float structural_color_min_support = 0.01f;
    constexpr float zero_anchor_min_px = -1.39635933f;
    constexpr float zero_anchor_max_px = 8.58230571f;

    bool finite(float value) {
      return std::isfinite(value);
    }

    bool finite(double value) {
      return std::isfinite(value);
    }

    bool usable_metric(const std::optional<float> &value) {
      return value && finite(*value) && *value >= 0.0f;
    }

    float metric_or_zero(const std::optional<float> &value) {
      return usable_metric(value) ? *value : 0.0f;
    }

    bool has_flag(const scene_frame_t &frame, std::uint32_t flag) {
      return (frame.analysis_flags & flag) != 0;
    }

    bool appearance_qualified(const scene_frame_t &frame) {
      if (has_flag(frame, analysis_appearance_veto)) {
        return false;
      }
      const std::optional<float> supports[] {
        frame.current_structural_support_fraction,
        frame.previous_structural_support_fraction,
        frame.common_structural_support_fraction,
      };
      for (const auto &support : supports) {
        if (support && (!finite(*support) || *support < structural_color_min_support)) {
          return false;
        }
      }
      return usable_metric(frame.depth_change_fraction) &&
             *frame.depth_change_fraction >= depth_cut_corroborate &&
             usable_metric(frame.raw_rgb_change_fraction) &&
             *frame.raw_rgb_change_fraction >= raw_rgb_cut_high &&
             usable_metric(frame.structural_change_fraction) &&
             *frame.structural_change_fraction >= structural_color_cut_high;
    }

    bool geometry_qualified(const scene_frame_t &frame) {
      return !has_flag(frame, analysis_appearance_veto) &&
             usable_metric(frame.depth_change_fraction) &&
             *frame.depth_change_fraction >= depth_cut_high;
    }

    bool complete_evidence(const scene_frame_t &frame) {
      return usable_metric(frame.depth_change_fraction) &&
             usable_metric(frame.raw_rgb_change_fraction) &&
             usable_metric(frame.structural_change_fraction);
    }

    float evidence_score(const scene_frame_t &frame) {
      const auto depth = metric_or_zero(frame.depth_change_fraction);
      const auto raw = metric_or_zero(frame.raw_rgb_change_fraction);
      const auto structural = metric_or_zero(frame.structural_change_fraction);
      const auto geometry = depth / depth_cut_high;
      const auto appearance = std::min({
        depth / depth_cut_corroborate,
        raw / raw_rgb_cut_high,
        structural / structural_color_cut_high,
      });
      return std::max(geometry, appearance);
    }

    float quantile(std::vector<float> values, float q) {
      if (values.empty()) {
        throw scene_plan_error("cannot take a quantile of an empty sequence");
      }
      std::sort(values.begin(), values.end());
      if (values.size() == 1) {
        return values.front();
      }
      const auto position = static_cast<double>(values.size() - 1) * q;
      const auto lower = static_cast<std::size_t>(std::floor(position));
      const auto upper = static_cast<std::size_t>(std::ceil(position));
      if (lower == upper) {
        return values[lower];
      }
      const auto fraction = static_cast<float>(position - lower);
      return values[lower] + (values[upper] - values[lower]) * fraction;
    }

    float smoothstep(float low, float high, float value) {
      const auto t = std::clamp((value - low) / (high - low), 0.0f, 1.0f);
      return t * t * (3.0f - 2.0f * t);
    }

    bool valid_anchor(float value) {
      return finite(value) && value >= zero_anchor_min_px && value <= zero_anchor_max_px;
    }

    template<class T>
    bool add_overflows(T lhs, T rhs) {
      return rhs > std::numeric_limits<T>::max() - lhs;
    }
  }  // namespace

  struct scene_planner_t::tracked_frame_t {
    scene_frame_t sample;
    std::int64_t depth_update_ordinal = -1;
  };

  struct scene_planner_t::proposal_cluster_t {
    std::vector<std::uint64_t> proposal_sequences;
    std::vector<std::string> proposal_frame_ids;
    std::vector<std::int64_t> proposal_update_ordinals;
    std::int64_t first_update_ordinal = -1;
    std::int64_t last_update_ordinal = -1;
  };

  scene_cache_budget_error::scene_cache_budget_error(
    std::uint64_t limit,
    std::uint64_t live,
    std::uint64_t open_start,
    std::uint64_t current
  ):
      scene_plan_error([&] {
        std::ostringstream message;
        message
          << "scene cache budget exceeded before a semantic boundary was finalized: "
          << live << " > " << limit << " bytes for sequences ["
          << open_start << ',' << current << ']';
        return message.str();
      }()),
      limit_bytes(limit),
      live_bytes(live),
      open_start_sequence(open_start),
      current_sequence(current) {
  }

  scene_metadata_budget_error::scene_metadata_budget_error(
    const std::size_t limit,
    const std::size_t attempted,
    const std::uint64_t open_start,
    const std::uint64_t current
  ):
      scene_plan_error([&] {
        std::ostringstream message;
        message
          << "scene analysis metadata budget exceeded before a semantic "
             "boundary was finalized: "
          << attempted << " > " << limit << " frames for sequences ["
          << open_start << ',' << current << ']';
        return message.str();
      }()),
      limit_frames(limit),
      attempted_frames(attempted),
      open_start_sequence(open_start),
      current_sequence(current) {
  }

  scene_planner_t::scene_planner_t(scene_planner_config_t config):
      config_(std::move(config)) {
    if (!finite(config_.pop_strength) ||
        !finite(config_.adaptive_pop_max) ||
        !finite(config_.risk_quantile) ||
        !finite(config_.pop_risk_low) ||
        !finite(config_.pop_risk_high)) {
      throw scene_plan_error("scene planner numeric configuration must be finite");
    }
    if (config_.pop_strength < 0.25f || config_.pop_strength > 2.0f) {
      throw scene_plan_error("pop_strength must be in [0.25, 2.0]");
    }
    if (config_.adaptive_pop_max < config_.pop_strength ||
        config_.adaptive_pop_max > 2.0f) {
      throw scene_plan_error("adaptive_pop_max must be at least pop_strength and at most 2.0");
    }
    if (config_.zero_plane != "subject" &&
        config_.zero_plane != "median" &&
        config_.zero_plane != "background") {
      throw scene_plan_error("zero_plane must be subject, median, or background");
    }
    if (config_.minimum_scene_frames == 0) {
      throw scene_plan_error("minimum_scene_frames must be at least one");
    }
    if (config_.max_open_frames < config_.minimum_scene_frames) {
      throw scene_plan_error(
        "max_open_frames must be at least minimum_scene_frames"
      );
    }
    if (config_.risk_quantile <= 0.0f || config_.risk_quantile > 1.0f) {
      throw scene_plan_error("risk_quantile must be in (0, 1]");
    }
    if (config_.pop_risk_low < 0.0f ||
        config_.pop_risk_low >= config_.pop_risk_high) {
      throw scene_plan_error("pop-risk endpoints must be ordered and non-negative");
    }
    constexpr auto ordinal_max =
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    if (config_.lookbehind_depth_updates > ordinal_max ||
        config_.lookahead_depth_updates > ordinal_max ||
        config_.duplicate_pulse_distance_updates > ordinal_max ||
        config_.settle_depth_updates > ordinal_max ||
        add_overflows(
          config_.lookbehind_depth_updates,
          config_.lookahead_depth_updates
        ) ||
        config_.lookbehind_depth_updates +
            config_.lookahead_depth_updates >
          ordinal_max) {
      throw scene_plan_error("scene planner update windows exceed the signed ordinal range");
    }
  }

  scene_planner_t::~scene_planner_t() = default;

  std::vector<scene_plan_t> scene_planner_t::feed(scene_frame_t frame) {
    if (closed_) {
      throw scene_plan_error("cannot feed a finalized scene planner");
    }
    if (frame.sequence != next_sequence_) {
      std::ostringstream message;
      message << "scene planner requires contiguous one-based sequences: got "
              << frame.sequence << ", expected " << next_sequence_;
      throw scene_plan_error(message.str());
    }
    if (frame.sequence >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw scene_plan_error("source sequence exceeds the signed audit range");
    }
    if (frame.frame_id.empty()) {
      throw scene_plan_error("scene planner frame_id must not be empty");
    }
    if (frame.frame_id.size() > max_scene_frame_id_bytes) {
      throw scene_plan_error(
        "scene planner frame_id exceeds its bounded identifier contract"
      );
    }
    if (next_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      throw scene_plan_error("source sequence overflow");
    }
    const auto validate_optional = [](const std::optional<float> &value, const char *name) {
      if (value && !finite(*value)) {
        throw scene_plan_error(std::string(name) + " must be finite when present");
      }
    };
    const auto validate_fraction = [&](const std::optional<float> &value, const char *name) {
      validate_optional(value, name);
      if (value && (*value < 0.0f || *value > 1.0f)) {
        throw scene_plan_error(std::string(name) + " must be in [0, 1] when present");
      }
    };
    validate_fraction(frame.depth_change_fraction, "depth_change_fraction");
    validate_fraction(frame.raw_rgb_change_fraction, "raw_rgb_change_fraction");
    validate_fraction(frame.structural_change_fraction, "structural_change_fraction");
    validate_fraction(
      frame.current_structural_support_fraction,
      "current_structural_support_fraction"
    );
    validate_fraction(
      frame.previous_structural_support_fraction,
      "previous_structural_support_fraction"
    );
    validate_fraction(
      frame.common_structural_support_fraction,
      "common_structural_support_fraction"
    );
    validate_optional(frame.edge_fraction, "edge_fraction");
    validate_optional(frame.zero_anchor_candidate_shift_px, "zero_anchor_candidate_shift_px");
    validate_optional(frame.production_zero_anchor_shift_px, "production_zero_anchor_shift_px");
    if (!finite(frame.valid_depth_fraction) ||
        frame.valid_depth_fraction < 0.0f ||
        frame.valid_depth_fraction > 1.0f) {
      throw scene_plan_error("valid_depth_fraction must be in [0, 1]");
    }
    if (!finite(frame.scene_age) || frame.scene_age < 0.0f) {
      throw scene_plan_error("scene_age must be finite and non-negative");
    }
    if (frame.pts_seconds && !finite(*frame.pts_seconds)) {
      throw scene_plan_error("pts_seconds must be finite when present");
    }
    if (frame.duration_seconds &&
        (!finite(*frame.duration_seconds) || *frame.duration_seconds <= 0.0)) {
      throw scene_plan_error("duration_seconds must be positive and finite when present");
    }
    if (add_overflows(open_cache_bytes_, frame.cache_bytes)) {
      throw scene_plan_error("scene cache byte counter overflow");
    }
    if (config_.max_open_cache_bytes != 0 &&
        frame.cache_bytes > config_.max_open_cache_bytes) {
      throw scene_cache_budget_error(
        config_.max_open_cache_bytes,
        frame.cache_bytes,
        frame.sequence,
        frame.sequence
      );
    }
    if (frames_.size() >= config_.max_open_frames) {
      throw scene_metadata_budget_error(
        config_.max_open_frames,
        frames_.size() + 1,
        open_start_sequence(),
        frame.sequence
      );
    }

    tracked_frame_t tracked {
      .sample = std::move(frame),
      .depth_update_ordinal = depth_update_ordinal_,
    };
    if (tracked.sample.depth_updated) {
      if (tracked.sample.pts_seconds) {
        const auto pts = *tracked.sample.pts_seconds;
        if (
          last_depth_pts_seconds_ &&
          pts < *last_depth_pts_seconds_
        ) {
          throw scene_plan_error(
            "timestamped depth updates must have nondecreasing presentation time"
          );
        }
        if (!timeline_origin_seconds_) {
          timeline_origin_seconds_ = pts;
        }
        const auto elapsed = pts - *timeline_origin_seconds_;
        const auto reference_ordinal =
          std::floor(elapsed * scene_reference_updates_per_second + 1e-9);
        if (
          !finite(reference_ordinal) ||
          reference_ordinal < 0.0 ||
          reference_ordinal >
            static_cast<double>(std::numeric_limits<std::int64_t>::max())
        ) {
          throw scene_plan_error(
            "timestamped depth update exceeds the reference timeline range"
          );
        }
        const auto ordinal = static_cast<std::int64_t>(reference_ordinal);
        if (ordinal < depth_update_ordinal_) {
          throw scene_plan_error(
            "timestamped depth update reference ordinal moved backward"
          );
        }
        tracked.depth_update_ordinal = ordinal;
        depth_update_ordinal_ = ordinal;
        last_depth_pts_seconds_ = pts;
      } else {
        if (depth_update_ordinal_ == std::numeric_limits<std::int64_t>::max()) {
          throw scene_plan_error("depth update ordinal overflow");
        }
        tracked.depth_update_ordinal = ++depth_update_ordinal_;
      }
    }
    open_cache_bytes_ += tracked.sample.cache_bytes;
    frames_.push_back(std::move(tracked));
    ++next_sequence_;

    const auto &stored = frames_.back();
    // A held color frame carries the previous SubjectState pulse. It is not a
    // second causal proposal.
    if (stored.sample.depth_updated && stored.sample.hard_cut_pulse) {
      add_proposal(stored);
    }

    auto finalized = resolve_mature(false);
    auto budget = enforce_budget();
    finalized.insert(
      finalized.end(),
      std::make_move_iterator(budget.begin()),
      std::make_move_iterator(budget.end())
    );
    return finalized;
  }

  std::vector<scene_plan_t> scene_planner_t::finish() {
    if (closed_) {
      throw scene_plan_error("scene planner was already finalized");
    }
    if (frames_.empty() && next_sequence_ == 1) {
      throw scene_plan_error("cannot finalize an empty clip");
    }
    auto finalized = resolve_mature(true);
    if (!frames_.empty()) {
      boundary_audit_t boundary;
      boundary.accepted = true;
      boundary.semantic_cut = false;
      boundary.decision = boundary_decision_e::end_of_stream;
      boundary.reason = "the final scene ends at the source EOF";
      finalized.push_back(finalize_prefix(frames_.size(), std::move(boundary)));
    }
    pending_.clear();
    closed_ = true;
    return finalized;
  }

  std::uint64_t scene_planner_t::open_cache_bytes() const {
    return open_cache_bytes_;
  }

  std::size_t scene_planner_t::open_frame_count() const {
    return frames_.size();
  }

  std::uint64_t scene_planner_t::open_start_sequence() const {
    return frames_.empty() ? next_sequence_ : frames_.front().sample.sequence;
  }

  std::size_t scene_planner_t::pending_proposal_count() const {
    std::size_t result = 0;
    for (const auto &cluster : pending_) {
      result += cluster.proposal_sequences.size();
    }
    return result;
  }

  const std::vector<boundary_audit_t> &scene_planner_t::boundary_audit() const {
    return boundary_audit_;
  }

  void scene_planner_t::add_proposal(const tracked_frame_t &frame) {
    if (frame.depth_update_ordinal < 0) {
      throw scene_plan_error("a cut pulse cannot precede the first depth update");
    }
    const auto overlap_reach =
      config_.lookbehind_depth_updates + config_.lookahead_depth_updates;
    const auto merge_reach = std::max(
      config_.duplicate_pulse_distance_updates,
      overlap_reach
    );
    if (!pending_.empty()) {
      auto &cluster = pending_.back();
      const auto distance = frame.depth_update_ordinal - cluster.last_update_ordinal;
      const auto cluster_span =
        frame.depth_update_ordinal - cluster.first_update_ordinal;
      // Overlapping evidence windows are duplicates only within one bounded
      // window measured from the first proposal. Measuring solely from the
      // latest proposal lets a persistent pulse train extend one cluster
      // forever, so neither lookahead nor the scene-cache cap can close it.
      if (distance >= 0 &&
          cluster_span >= 0 &&
          static_cast<std::uint64_t>(distance) <= merge_reach &&
          static_cast<std::uint64_t>(cluster_span) <= merge_reach) {
        cluster.proposal_sequences.push_back(frame.sample.sequence);
        cluster.proposal_frame_ids.push_back(frame.sample.frame_id);
        cluster.proposal_update_ordinals.push_back(frame.depth_update_ordinal);
        cluster.last_update_ordinal = frame.depth_update_ordinal;
        return;
      }
    }
    proposal_cluster_t cluster;
    cluster.proposal_sequences.push_back(frame.sample.sequence);
    cluster.proposal_frame_ids.push_back(frame.sample.frame_id);
    cluster.proposal_update_ordinals.push_back(frame.depth_update_ordinal);
    cluster.first_update_ordinal = frame.depth_update_ordinal;
    cluster.last_update_ordinal = frame.depth_update_ordinal;
    pending_.push_back(std::move(cluster));
  }

  std::vector<scene_plan_t> scene_planner_t::resolve_mature(bool eof) {
    std::vector<scene_plan_t> finalized;
    while (!pending_.empty()) {
      const auto &front = pending_.front();
      const auto maturity_distance =
        config_.lookahead_depth_updates + config_.lookbehind_depth_updates;
      const auto mature =
        depth_update_ordinal_ >= front.last_update_ordinal &&
        static_cast<std::uint64_t>(
          depth_update_ordinal_ - front.last_update_ordinal
        ) >= maturity_distance;
      if (!eof && !mature) {
        break;
      }
      auto cluster = std::move(pending_.front());
      pending_.erase(pending_.begin());
      if (auto scene = resolve_cluster(cluster, eof && !mature)) {
        finalized.push_back(std::move(*scene));
      }
    }
    return finalized;
  }

  std::optional<scene_plan_t> scene_planner_t::resolve_cluster(
    const proposal_cluster_t &cluster,
    bool truncated,
    bool budget_forced
  ) {
    if (frames_.empty()) {
      return std::nullopt;
    }

    struct candidate_t {
      std::size_t position;
      std::uint64_t sequence;
      std::int64_t ordinal;
      float score;
      std::uint64_t nearest_distance;
      bool evidence_complete;
      bool appearance;
      bool geometry;
      bool relative;
    };

    const auto open_start = frames_.front().sample.sequence;
    if (config_.minimum_scene_frames >
        std::numeric_limits<std::uint64_t>::max() - open_start) {
      throw scene_plan_error("minimum scene length overflows the source sequence");
    }
    const auto minimum_boundary = open_start + config_.minimum_scene_frames;
    const auto left_ordinal =
      cluster.first_update_ordinal -
      static_cast<std::int64_t>(config_.lookbehind_depth_updates);
    if (static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max() -
          cluster.last_update_ordinal
        ) < config_.lookahead_depth_updates) {
      throw scene_plan_error("scene evidence window overflows the depth ordinal");
    }
    const auto right_ordinal =
      cluster.last_update_ordinal +
      static_cast<std::int64_t>(config_.lookahead_depth_updates);
    const auto next_cluster_start = pending_.empty() ?
      std::optional<std::uint64_t> {} :
      std::optional<std::uint64_t> {pending_.front().proposal_sequences.front()};

    const auto is_proposal = [&](std::uint64_t sequence) {
      return std::find(
               cluster.proposal_sequences.begin(),
               cluster.proposal_sequences.end(),
               sequence
             ) != cluster.proposal_sequences.end();
    };
    std::vector<candidate_t> candidates;
    for (std::size_t position = 0; position < frames_.size(); ++position) {
      const auto &tracked = frames_[position];
      const auto &frame = tracked.sample;
      if (!frame.depth_updated ||
          tracked.depth_update_ordinal < left_ordinal ||
          tracked.depth_update_ordinal > right_ordinal ||
          frame.sequence < minimum_boundary) {
        continue;
      }
      if (next_cluster_start &&
          (config_.minimum_scene_frames >
             std::numeric_limits<std::uint64_t>::max() - frame.sequence ||
           frame.sequence + config_.minimum_scene_frames > *next_cluster_start)) {
        continue;
      }
      const auto proposal = is_proposal(frame.sequence);
      const auto appearance = appearance_qualified(frame);
      const auto geometry = geometry_qualified(frame);
      const auto relative = has_flag(frame, analysis_relative_geometry_spike);
      if (!(appearance || geometry || relative) &&
          !(proposal && !complete_evidence(frame))) {
        continue;
      }
      if (!proposal && !(appearance || geometry)) {
        continue;
      }
      if (has_flag(frame, analysis_appearance_veto)) {
        continue;
      }
      auto nearest = std::numeric_limits<std::uint64_t>::max();
      for (const auto proposal_ordinal : cluster.proposal_update_ordinals) {
        const auto distance =
          tracked.depth_update_ordinal > proposal_ordinal ?
            static_cast<std::uint64_t>(
              tracked.depth_update_ordinal - proposal_ordinal
            ) :
            static_cast<std::uint64_t>(
              proposal_ordinal - tracked.depth_update_ordinal
            );
        nearest = std::min(nearest, distance);
      }
      candidates.push_back({
        .position = position,
        .sequence = frame.sequence,
        .ordinal = tracked.depth_update_ordinal,
        .score = evidence_score(frame),
        .nearest_distance = nearest,
        .evidence_complete = complete_evidence(frame),
        .appearance = appearance,
        .geometry = geometry,
        .relative = relative,
      });
    }

    const auto returned_to_same_scene = std::any_of(
      frames_.begin(),
      frames_.end(),
      [&](const tracked_frame_t &frame) {
        return frame.depth_update_ordinal >= left_ordinal &&
               frame.depth_update_ordinal <= right_ordinal &&
               frame.sample.sequence > cluster.proposal_sequences.back() &&
               has_flag(frame.sample, analysis_same_scene_return);
      }
    );
    const auto all_proposals_vetoed = std::all_of(
      cluster.proposal_sequences.begin(),
      cluster.proposal_sequences.end(),
      [&](std::uint64_t sequence) {
        const auto found = std::find_if(
          frames_.begin(),
          frames_.end(),
          [&](const tracked_frame_t &frame) {
            return frame.sample.sequence == sequence;
          }
        );
        return found != frames_.end() &&
               has_flag(found->sample, analysis_appearance_veto);
      }
    );
    const auto qualified_replacement_after_return = std::any_of(
      candidates.begin(),
      candidates.end(),
      [&](const candidate_t &candidate) {
        return candidate.sequence > cluster.proposal_sequences.back() &&
               (candidate.appearance || candidate.geometry);
      }
    );
    if (all_proposals_vetoed &&
        returned_to_same_scene &&
        !qualified_replacement_after_return) {
      boundary_audit_t boundary;
      boundary.proposal_sequences = cluster.proposal_sequences;
      boundary.proposal_frame_ids = cluster.proposal_frame_ids;
      boundary.accepted = false;
      boundary.decision = boundary_decision_e::rejected_supported_flash_return;
      boundary.reason =
        "every proposal was appearance-vetoed and future evidence returned "
        "to the supported left endpoint";
      boundary.truncated = truncated;
      boundary.budget_forced = budget_forced;
      boundary_audit_.push_back(std::move(boundary));
      return std::nullopt;
    }

    std::uint64_t boundary_sequence = 0;
    std::int64_t selected_ordinal = cluster.first_update_ordinal;
    std::optional<float> selected_score;
    std::size_t selected_position = 0;
    bool selected_appearance = false;
    bool selected_geometry = false;
    bool selected_relative = false;
    boundary_decision_e decision;
    std::string reason;

    if (candidates.empty()) {
      const auto legal_length_proposal = std::any_of(
        cluster.proposal_sequences.begin(),
        cluster.proposal_sequences.end(),
        [&](std::uint64_t sequence) {
          if (sequence < minimum_boundary || sequence < open_start) {
            return false;
          }
          const auto prefix =
            static_cast<std::size_t>(sequence - open_start);
          return prefix >= config_.minimum_scene_frames &&
                 frames_.size() - prefix >= config_.minimum_scene_frames;
        }
      );
      boundary_audit_t boundary;
      boundary.proposal_sequences = cluster.proposal_sequences;
      boundary.proposal_frame_ids = cluster.proposal_frame_ids;
      boundary.accepted = false;
      boundary.decision = legal_length_proposal ?
        boundary_decision_e::rejected_unsupported_proposal :
        boundary_decision_e::rejected_minimum_scene_length;
      boundary.reason = legal_length_proposal ?
        "every legal proposal was vetoed or its complete exported evidence "
        "supported neither geometry, appearance, nor the relative-geometry escape" :
        "the proposed boundary could not leave legal scene prefixes on both sides";
      boundary.truncated = truncated;
      boundary.budget_forced = budget_forced;
      boundary_audit_.push_back(std::move(boundary));
      return std::nullopt;
    }

    const auto best = std::min_element(
      candidates.begin(),
      candidates.end(),
      [](const candidate_t &lhs, const candidate_t &rhs) {
        if (lhs.score != rhs.score) {
          return lhs.score > rhs.score;
        }
        if (lhs.nearest_distance != rhs.nearest_distance) {
          return lhs.nearest_distance < rhs.nearest_distance;
        }
        return lhs.sequence < rhs.sequence;
      }
    );
    boundary_sequence = best->sequence;
    selected_ordinal = best->ordinal;
    selected_position = best->position;
    selected_appearance = best->appearance;
    selected_geometry = best->geometry;
    selected_relative = best->relative;
    if (best->evidence_complete) {
      selected_score = best->score;
    }
    if (!best->evidence_complete) {
      decision = boundary_decision_e::confirmed_causal_fallback;
      reason =
        "the specific non-vetoed production proposal lacked optional diagnostics "
        "and was retained conservatively";
    } else if (!is_proposal(boundary_sequence)) {
      decision = boundary_decision_e::moved_to_correlated_evidence;
      reason = "lookahead found a stronger depth-corroborated transition";
    } else if (cluster.proposal_sequences.size() > 1) {
      decision = boundary_decision_e::merged_duplicate_proposals;
      reason = "nearby production pulses resolved to one deterministic boundary";
    } else {
      decision = boundary_decision_e::confirmed;
      reason = "the production pulse remained the strongest legal transition";
    }

    if (boundary_sequence < open_start) {
      throw scene_plan_error("resolved boundary precedes the open scene");
    }
    const auto prefix_count =
      static_cast<std::size_t>(boundary_sequence - open_start);
    const auto right_count = frames_.size() - prefix_count;
    if (prefix_count < config_.minimum_scene_frames ||
        right_count < config_.minimum_scene_frames) {
      boundary_audit_t boundary;
      boundary.proposal_sequences = cluster.proposal_sequences;
      boundary.proposal_frame_ids = cluster.proposal_frame_ids;
      boundary.accepted = false;
      boundary.decision = boundary_decision_e::rejected_minimum_scene_length;
      boundary.reason =
        "the refined boundary could not leave legal scene prefixes on both sides";
      boundary.truncated = truncated;
      boundary.budget_forced = budget_forced;
      boundary_audit_.push_back(std::move(boundary));
      return std::nullopt;
    }

    boundary_audit_t boundary;
    boundary.proposal_sequences = cluster.proposal_sequences;
    boundary.proposal_frame_ids = cluster.proposal_frame_ids;
    boundary.final_sequence = boundary_sequence;
    boundary.accepted = true;
    boundary.semantic_cut = true;
    boundary.decision = decision;
    boundary.reason = std::move(reason);
    boundary.truncated = truncated;
    boundary.budget_forced = budget_forced;
    boundary.revision_depth_updates =
      selected_ordinal - cluster.first_update_ordinal;
    boundary.revision_source_frames =
      static_cast<std::int64_t>(boundary_sequence) -
      static_cast<std::int64_t>(cluster.proposal_sequences.front());
    boundary.candidate_count = candidates.size();
    boundary.selected_evidence_score = selected_score;
    const auto window_first = std::find_if(
      frames_.begin(),
      frames_.end(),
      [&](const tracked_frame_t &frame) {
        return frame.depth_update_ordinal >= left_ordinal &&
               frame.depth_update_ordinal <= right_ordinal;
      }
    );
    const auto window_last = std::find_if(
      frames_.rbegin(),
      frames_.rend(),
      [&](const tracked_frame_t &frame) {
        return frame.depth_update_ordinal >= left_ordinal &&
               frame.depth_update_ordinal <= right_ordinal;
      }
    );
    if (window_first != frames_.end()) {
      boundary.evidence_window_first_sequence = window_first->sample.sequence;
    }
    if (window_last != frames_.rend()) {
      boundary.evidence_window_last_sequence = window_last->sample.sequence;
    }
    const auto &selected = frames_.at(selected_position).sample;
    boundary.selected_sequence = selected.sequence;
    boundary.selected_frame_id = selected.frame_id;
    boundary.selected_depth_change_fraction = selected.depth_change_fraction;
    boundary.selected_raw_rgb_change_fraction = selected.raw_rgb_change_fraction;
    boundary.selected_structural_change_fraction =
      selected.structural_change_fraction;
    boundary.selected_appearance_qualified = selected_appearance;
    boundary.selected_geometry_qualified = selected_geometry;
    boundary.selected_relative_geometry_spike = selected_relative;
    boundary_audit_.push_back(boundary);
    return finalize_prefix(prefix_count, std::move(boundary));
  }

  std::vector<scene_plan_t> scene_planner_t::enforce_budget() {
    const auto limit = config_.max_open_cache_bytes;
    if (limit == 0 || open_cache_bytes_ <= limit) {
      return {};
    }
    if (!config_.allow_administrative_split) {
      throw scene_cache_budget_error(
        limit,
        open_cache_bytes_,
        open_start_sequence(),
        next_sequence_ - 1
      );
    }

    std::vector<scene_plan_t> result;
    // The explicit split policy may close bounded pending evidence early. This
    // preserves the normal semantic resolver and its audit trail, but records
    // that the available lookahead was truncated by the storage budget. A
    // rejected proposal is audited before the administrative fallback; later
    // clusters remain pending as soon as enough cache has been released.
    while (open_cache_bytes_ > limit && !pending_.empty()) {
      auto cluster = std::move(pending_.front());
      pending_.erase(pending_.begin());
      if (auto scene = resolve_cluster(cluster, true, true)) {
        result.push_back(std::move(*scene));
      }
    }
    if (open_cache_bytes_ <= limit) {
      return result;
    }
    if (frames_.size() <= config_.minimum_scene_frames) {
      throw scene_cache_budget_error(
        limit,
        open_cache_bytes_,
        open_start_sequence(),
        next_sequence_ - 1
      );
    }

    const auto prefix_count = frames_.size() - 1;
    boundary_audit_t boundary;
    boundary.final_sequence = frames_[prefix_count].sample.sequence;
    boundary.accepted = false;
    boundary.semantic_cut = false;
    boundary.truncated = true;
    boundary.budget_forced = true;
    boundary.decision = boundary_decision_e::administrative_cache_split;
    boundary.reason =
      "the opt-in split policy inserted an explicit camera boundary before a "
      "semantic cut; the next segment receives a new scene-wide camera plan";
    boundary_audit_.push_back(boundary);
    result.push_back(finalize_prefix(prefix_count, std::move(boundary)));
    if (open_cache_bytes_ > limit) {
      throw scene_cache_budget_error(
        limit,
        open_cache_bytes_,
        open_start_sequence(),
        next_sequence_ - 1
      );
    }
    return result;
  }

  scene_plan_t scene_planner_t::finalize_prefix(
    std::size_t prefix_count,
    boundary_audit_t boundary,
    bool increment_semantic_scene
  ) {
    if (prefix_count == 0 || prefix_count > frames_.size()) {
      throw scene_plan_error("finalized scene prefix is empty or out of range");
    }

    std::vector<tracked_frame_t> scene_frames;
    scene_frames.reserve(prefix_count);
    std::move(
      frames_.begin(),
      frames_.begin() + static_cast<std::ptrdiff_t>(prefix_count),
      std::back_inserter(scene_frames)
    );
    frames_.erase(
      frames_.begin(),
      frames_.begin() + static_cast<std::ptrdiff_t>(prefix_count)
    );

    std::uint64_t released_bytes = 0;
    for (const auto &frame : scene_frames) {
      released_bytes += frame.sample.cache_bytes;
    }
    if (released_bytes > open_cache_bytes_) {
      throw scene_plan_error("scene cache accounting underflow");
    }
    open_cache_bytes_ -= released_bytes;

    std::vector<const scene_frame_t *> settled;
    std::optional<std::int64_t> first_depth_ordinal;
    std::size_t depth_updates = 0;
    for (const auto &tracked : scene_frames) {
      if (!tracked.sample.depth_updated) {
        continue;
      }
      ++depth_updates;
      if (!first_depth_ordinal) {
        first_depth_ordinal = tracked.depth_update_ordinal;
      }
      if (
        tracked.depth_update_ordinal >= *first_depth_ordinal &&
        static_cast<std::uint64_t>(
          tracked.depth_update_ordinal - *first_depth_ordinal
        ) >= config_.settle_depth_updates
      ) {
        settled.push_back(&tracked.sample);
      }
    }
    const auto usable_depth = [&](const scene_frame_t &frame) {
      return frame.initialized &&
             frame.depth_ready &&
             frame.valid_depth_fraction > 0.0f &&
             !frame.range_collapsed;
    };

    std::vector<float> edge_values;
    std::vector<float> anchor_values;
    std::size_t usable_settled_count = 0;
    for (const auto *frame : settled) {
      if (!usable_depth(*frame)) {
        continue;
      }
      ++usable_settled_count;
      if (usable_metric(frame->edge_fraction)) {
        edge_values.push_back(*frame->edge_fraction);
      }
      if (frame->zero_anchor_valid &&
          frame->zero_anchor_candidate_shift_px &&
          valid_anchor(*frame->zero_anchor_candidate_shift_px)) {
        anchor_values.push_back(*frame->zero_anchor_candidate_shift_px);
      }
    }

    float absolute_pop = config_.pop_strength;
    std::string pop_origin;
    std::optional<std::string> pop_fallback;
    std::optional<float> risk_value;
    if (!config_.adaptive_pop) {
      pop_origin = "configured-fixed";
    } else if (!edge_values.empty()) {
      const auto risk = quantile(edge_values, config_.risk_quantile);
      risk_value = risk;
      const auto confidence =
        1.0f - smoothstep(config_.pop_risk_low, config_.pop_risk_high, risk);
      absolute_pop = std::clamp(
        config_.pop_strength +
          (config_.adaptive_pop_max - config_.pop_strength) * confidence,
        config_.pop_strength,
        config_.adaptive_pop_max
      );
      pop_origin = "whole-finalized-scene-edge-risk";
    } else {
      pop_origin = "conservative-floor-fallback";
      pop_fallback = "no settled valid instantaneous edge-risk samples";
    }

    float zero_anchor = 0.0f;
    std::string zero_origin;
    std::optional<std::string> zero_fallback;
    if (!anchor_values.empty()) {
      zero_anchor = quantile(anchor_values, 0.50f);
      zero_origin = "whole-finalized-scene-median-candidate";
    } else {
      bool found = false;
      for (const auto &tracked : scene_frames) {
        const auto &frame = tracked.sample;
        if (frame.zero_anchor_valid &&
            frame.production_zero_anchor_shift_px &&
            valid_anchor(*frame.production_zero_anchor_shift_px)) {
          zero_anchor = *frame.production_zero_anchor_shift_px;
          found = true;
        }
      }
      zero_origin = found ? "production-latched-fallback" : "neutral-fallback";
      zero_fallback = "no settled valid zero-anchor candidate samples";
    }

    std::vector<float> depth_changes;
    std::size_t appearance_veto_count = 0;
    for (const auto &tracked : scene_frames) {
      if (usable_metric(tracked.sample.depth_change_fraction)) {
        depth_changes.push_back(*tracked.sample.depth_change_fraction);
      }
      if (has_flag(tracked.sample, analysis_appearance_veto)) {
        ++appearance_veto_count;
      }
    }

    scene_plan_t result;
    result.scene_id = ++scene_number_;
    result.semantic_scene_id = semantic_scene_number_;
    result.start_sequence = scene_frames.front().sample.sequence;
    result.end_sequence_exclusive = scene_frames.back().sample.sequence + 1;
    result.frame_count = scene_frames.size();
    result.cache_bytes = released_bytes;
    result.absolute_pop_strength = absolute_pop;
    result.zero_anchor_shift_px = zero_anchor;
    result.pop_origin = std::move(pop_origin);
    result.pop_fallback = std::move(pop_fallback);
    result.zero_origin = std::move(zero_origin);
    result.zero_fallback = std::move(zero_fallback);
    result.evidence.source_frame_count = scene_frames.size();
    result.evidence.depth_update_count = depth_updates;
    result.evidence.settled_depth_update_count = settled.size();
    result.evidence.usable_settled_depth_update_count = usable_settled_count;
    result.evidence.valid_edge_sample_count = edge_values.size();
    result.evidence.valid_anchor_sample_count = anchor_values.size();
    result.evidence.excluded_edge_sample_count =
      settled.size() - edge_values.size();
    result.evidence.excluded_anchor_sample_count =
      settled.size() - anchor_values.size();
    result.evidence.appearance_veto_count = appearance_veto_count;
    result.evidence.risk_quantile = config_.risk_quantile;
    result.evidence.pop_risk_low = config_.pop_risk_low;
    result.evidence.pop_risk_high = config_.pop_risk_high;
    result.evidence.risk_value = risk_value;
    if (!edge_values.empty()) {
      result.evidence.edge_p50 = quantile(edge_values, 0.50f);
      result.evidence.edge_p90 = quantile(edge_values, 0.90f);
      result.evidence.edge_p95 = quantile(edge_values, 0.95f);
      result.evidence.edge_max =
        *std::max_element(edge_values.begin(), edge_values.end());
    }
    if (!anchor_values.empty()) {
      result.evidence.anchor_p10 = quantile(anchor_values, 0.10f);
      result.evidence.anchor_p50 = quantile(anchor_values, 0.50f);
      result.evidence.anchor_p90 = quantile(anchor_values, 0.90f);
    }
    if (!depth_changes.empty()) {
      result.evidence.depth_change_max =
        *std::max_element(depth_changes.begin(), depth_changes.end());
    }
    result.boundary = std::move(boundary);

    const auto &first = scene_frames.front().sample;
    const auto &last = scene_frames.back().sample;
    if (first.pts_seconds &&
        last.pts_seconds &&
        last.duration_seconds &&
        finite(*first.pts_seconds) &&
        finite(*last.pts_seconds) &&
        finite(*last.duration_seconds) &&
        *last.duration_seconds > 0.0 &&
        *last.pts_seconds >= *first.pts_seconds) {
      const auto end = *last.pts_seconds + *last.duration_seconds;
      if (finite(end)) {
        result.start_pts_seconds = *first.pts_seconds;
        result.end_pts_seconds_exclusive = end;
      }
    }
    if (increment_semantic_scene && result.boundary.semantic_cut) {
      ++semantic_scene_number_;
    }
    return result;
  }

  const char *boundary_decision_name(boundary_decision_e decision) {
    switch (decision) {
      case boundary_decision_e::confirmed:
        return "confirmed";
      case boundary_decision_e::moved_to_correlated_evidence:
        return "moved_to_correlated_evidence";
      case boundary_decision_e::merged_duplicate_proposals:
        return "merged_duplicate_proposals";
      case boundary_decision_e::confirmed_causal_fallback:
        return "confirmed_causal_fallback";
      case boundary_decision_e::rejected_supported_flash_return:
        return "rejected_supported_flash_return";
      case boundary_decision_e::rejected_unsupported_proposal:
        return "rejected_unsupported_proposal";
      case boundary_decision_e::rejected_minimum_scene_length:
        return "rejected_minimum_scene_length";
      case boundary_decision_e::administrative_cache_split:
        return "administrative_cache_split";
      case boundary_decision_e::end_of_stream:
        return "end_of_stream";
    }
    return "unknown";
  }
}  // namespace offline_sbs

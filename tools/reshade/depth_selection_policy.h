// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace sunshine_depth {
  enum class content_kind { unknown, flat, unreliable, useful };

  struct depth_quality {
    content_kind kind = content_kind::unknown;
    float score = 0;
    float clear_fraction = 0;
    float finite_fraction = 0;
    float relative_range = 0;
    float roughness = 0;
    // Endpoint occupancy estimates missing support; no exact per-copy clear
    // value is available, so these are relative completeness clues, not proof.
    float nonendpoint_fraction = 0, spatial_coverage = 0, completeness = 0;
    // Breadth of strict nonendpoint hardware depth in globally anchored log-odds
    // bins. This is a dimensionless distribution clue, not camera distance.
    float histogram_span = 0, histogram_breadth = 0;
    unsigned histogram_populated_bins = 0;
    std::size_t samples = 0;
  };

  // The input is raw hardware depth, not linearized scene distance. Never impose
  // an absolute variance floor: valid reversed-Z scenes may occupy tiny values.
  inline depth_quality analyze_depth(const float *values, std::size_t count, unsigned grid_width, unsigned grid_height) {
    depth_quality result;
    if (!values || !grid_width || !grid_height || count != std::size_t(grid_width) * grid_height || count < 16)
      return result;
    result.samples = count;
    std::size_t finite = 0, clear = 0, outside = 0;
    std::vector<float> sorted;
    sorted.reserve(count);
    std::vector<unsigned char> support(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      const float value = values[i];
      if (!std::isfinite(value))
        continue;
      ++finite;
      if (value < 0 || value > 1)
        ++outside;
      if (value == 0 || value == 1)
        ++clear;
      else if (value > 0 && value < 1) {
        sorted.push_back(value);
        support[i] = 1;
      }
    }
    result.finite_fraction = float(finite) / float(count);
    result.clear_fraction = float(clear) / float(count);
    result.nonendpoint_fraction = float(sorted.size()) / float(count);
    if (finite != count || outside) {
      result.kind = content_kind::unreliable;
      return result;
    }
    // Coherent terrain can occupy only a small part of a sky-heavy view. Do not
    // reject it merely because many samples are exact zero/one. A minimum amount
    // of two-dimensional support is still required below.
    if (sorted.size() < 16) {
      result.kind = content_kind::flat;
      return result;
    }
    std::array<unsigned, 12> tile_support {}, tile_size {};
    for (unsigned y = 0; y < grid_height; ++y)
      for (unsigned x = 0; x < grid_width; ++x) {
        const unsigned tile = std::min(y * 3 / grid_height, 2u) * 4 + std::min(x * 4 / grid_width, 3u);
        ++tile_size[tile];
        tile_support[tile] += support[std::size_t(y) * grid_width + x];
      }
    unsigned occupied = 0, tiles = 0;
    for (unsigned tile = 0; tile < tile_size.size(); ++tile) {
      if (!tile_size[tile]) continue;
      ++tiles;
      if (tile_support[tile] >= std::max(2u, (tile_size[tile] + 9) / 10)) ++occupied;
    }
    result.spatial_coverage = tiles ? float(occupied) / tiles : 0;
    result.completeness = .7f * result.nonendpoint_fraction + .3f * result.spatial_coverage;

    // Isolated speckles and alternating rows can cover every coarse tile while
    // containing no coherent surface. Require a substantial connected component
    // before spatial occupancy is allowed to contribute positive evidence.
    std::vector<unsigned char> visited(count, 0);
    std::vector<std::size_t> pending;
    std::size_t largest_component = 0;
    for (std::size_t start = 0; start < count; ++start) {
      if (!support[start] || visited[start]) continue;
      pending.clear();
      pending.push_back(start);
      visited[start] = 1;
      std::size_t component = 0;
      while (!pending.empty()) {
        const auto i = pending.back();
        pending.pop_back();
        ++component;
        const auto visit = [&](std::size_t next) {
          if (support[next] && !visited[next]) { visited[next] = 1; pending.push_back(next); }
        };
        if (i % grid_width) visit(i - 1);
        if (i % grid_width + 1 < grid_width) visit(i + 1);
        if (i >= grid_width) visit(i - grid_width);
        if (i + grid_width < count) visit(i + grid_width);
      }
      largest_component = std::max(largest_component, component);
    }
    if (largest_component * 4 < sorted.size()) {
      result.kind = content_kind::unreliable;
      return result;
    }
    std::sort(sorted.begin(), sorted.end());
    const auto supported_count = sorted.size();
    const float low = sorted[(supported_count - 1) / 10];
    const float high = sorted[((supported_count - 1) * 9) / 10];
    const auto log_odds = [](float value) {
      const double depth = value;
      return std::log2(depth) - std::log2(1.0 - depth);
    };
    result.histogram_span = float(log_odds(high) - log_odds(low));
    // Every representable strict nonendpoint float fits within these global
    // bounds. Explicit saturation also keeps indexing bounded. Half-octave bins
    // reflect around zero on normal/reversed depth; no candidate normalization
    // or epsilon near the endpoints can manufacture or erase histogram breadth.
    std::array<unsigned, 641> histogram {};
    for (const float value : sorted) {
      const int bin = int(std::clamp(std::round(log_odds(value) * 2.0), -320.0, 320.0));
      ++histogram[std::size_t(bin + 320)];
    }
    const std::size_t minimum_bin_samples = std::max(std::size_t(4), (supported_count + 49) / 50);
    for (const unsigned bin_samples : histogram)
      if (bin_samples >= minimum_bin_samples) ++result.histogram_populated_bins;
    result.histogram_breadth = .75f * std::min(result.histogram_span / 8.f, 1.f) +
                               .25f * std::min(float(result.histogram_populated_bins > 0 ? result.histogram_populated_bins - 1 : 0) / 15.f, 1.f);
    const float range = high - low;
    const float magnitude = std::max(std::abs(low), std::abs(high));
    const float precision = std::max(magnitude * std::numeric_limits<float>::epsilon() * 4,
                                     std::numeric_limits<float>::denorm_min() * 16);
    result.relative_range = magnitude > 0 ? range / magnitude : 0;
    // Robust quantiles suppress isolated speckles; the precision-relative bound
    // rejects float-rounding noise without penalizing near-zero reversed depth.
    if (!(range > precision)) {
      result.kind = content_kind::flat;
      return result;
    }
    std::vector<float> horizontal, vertical;
    horizontal.reserve(count);
    vertical.reserve(count);
    for (unsigned y = 0; y < grid_height; ++y) {
      for (unsigned x = 0; x < grid_width; ++x) {
        const auto i = std::size_t(y) * grid_width + x;
        if (!support[i]) continue;
        if (x + 1 < grid_width && support[i + 1])
          horizontal.push_back(std::abs(values[i] - values[i + 1]));
        if (y + 1 < grid_height && support[i + grid_width])
          vertical.push_back(std::abs(values[i] - values[i + grid_width]));
      }
    }
    const auto median = [](std::vector<float> &differences) {
      if (differences.empty())
        return 0.f;
      const auto middle = differences.begin() + differences.size() / 2;
      std::nth_element(differences.begin(), middle, differences.end());
      return *middle;
    };
    if (horizontal.size() < 4 || vertical.size() < 4) {
      result.kind = content_kind::unreliable;
      return result;
    }
    // Check each axis separately so rows/columns of garbage cannot hide behind
    // the many zero differences along the other axis of the rectangular grid.
    result.roughness = std::max(median(horizontal), median(vertical)) / range;
    // Rich-looking checkerboards/noise are not proof of scene depth. This only
    // withholds positive confidence; it does not force the selected source off.
    if (result.roughness > .35f) {
      result.kind = content_kind::unreliable;
      return result;
    }
    result.kind = content_kind::useful;
    result.score = .65f * (1 - std::min(result.roughness / .35f, 1.f)) + .35f * (1 - result.clear_fraction);
    return result;
  }

  // Default geometry eligibility; the caller adds only the original similar-aspect
  // term. Quarter resolution includes common upscaling modes; aspect agreement
  // excludes square and skinny shadow targets.
  inline bool auto_candidate_shape(unsigned width, unsigned height, unsigned output_width, unsigned output_height) {
    if (!width || !height || !output_width || !output_height)
      return false;
    const double x = double(width) / output_width, y = double(height) / output_height;
    return x >= .25 && x <= 2 && y >= .25 && y <= 2 &&
           std::abs(double(width) / height - double(output_width) / output_height) <= .1;
  }

  struct candidate_info {
    // This is a lifetime generation, never a resource pointer or reusable slot.
    std::uint64_t id = 0;
    unsigned width = 0, height = 0;
    float viewport_width = 0, viewport_height = 0;
    unsigned output_width = 0, output_height = 0;
    std::uint64_t last_seen_frame = 0;
    std::uint64_t draws = 0, vertices = 0;
  };

  struct candidate_evidence {
    depth_quality quality;
    // Conservative challenger score and protected incumbent score from the
    // recent positive captures. One unusually smooth/rough sample cannot alone
    // make a challenger win or make the incumbent cheap to replace.
    float candidate_quality_score = 0, retained_quality_score = 0;
    float candidate_completeness = 0, retained_completeness = 0;
    float candidate_histogram_span = 0, retained_histogram_span = 0;
    unsigned candidate_histogram_bins = 0, retained_histogram_bins = 0;
    float candidate_histogram_breadth = 0, retained_histogram_breadth = 0;
    unsigned good_samples = 0, bad_samples = 0;
    bool confirmed_good = false;
    std::uint64_t sampled_frame = 0;
  };

  struct selection_result {
    std::uint64_t id = 0;
    const char *reason = "no active candidate";
  };

  struct policy_options {
    std::uint64_t active_grace_frames = 3;
    // Sampling can rotate behind deferred GPU retirement. Keep this longer than
    // a round through candidates; callers may tune it to their bounded cadence.
    std::uint64_t evidence_stale_frames = 1800;
    unsigned good_samples_required = 2;
    unsigned bad_samples_required = 3;
    unsigned comparison_samples_required = 3;
    double comparison_margin = .04;
    float comparison_quality_tolerance = .25f;
    float completeness_margin = .20f;
    float histogram_completeness_tolerance = .10f;
    float histogram_span_margin = .75f;
    unsigned histogram_bin_margin = 2;
  };

  class selection_policy {
  public:
    explicit selection_policy(policy_options options = {}) : options_(options) {}
    // The depth frames after its last work during which a candidate still
    // counts as active (Generic's hold cut gate reads the same grace).
    std::uint64_t active_grace_frames() const { return options_.active_grace_frames; }

    void observe(const candidate_info &info) {
      if (info.id)
        candidates_[info.id].info = info;
    }

    void sample(std::uint64_t id, const depth_quality &quality, std::uint64_t frame) {
      const auto found = candidates_.find(id);
      // Delayed readbacks cannot recreate a destroyed resource or update its
      // replacement. An unavailable capture is not a negative depth sample.
      if (found == candidates_.end() || quality.kind == content_kind::unknown)
        return;
      auto &e = found->second.evidence;
      if (e.sampled && frame <= e.value.sampled_frame)
        return;
      if (e.sampled && frame - e.value.sampled_frame > options_.evidence_stale_frames) {
        e = {};
        if (pending_.incumbent == id || pending_.challenger == id)
          pending_ = {};
      }
      if (quality.kind != content_kind::useful && (pending_.incumbent == id || pending_.challenger == id))
        pending_ = {};
      e.sampled = true;
      e.value.quality = quality;
      e.value.sampled_frame = frame;
      if (quality.kind == content_kind::useful) {
        e.value.bad_samples = 0;
        const unsigned required = std::max({options_.good_samples_required, options_.comparison_samples_required, 1u});
        e.value.good_samples = std::min(e.value.good_samples + 1, required);
        for (std::size_t i = e.positive_scores.size() - 1; i > 0; --i) {
          e.positive_scores[i] = e.positive_scores[i - 1];
          e.positive_completeness[i] = e.positive_completeness[i - 1];
          e.positive_histogram_span[i] = e.positive_histogram_span[i - 1];
          e.positive_histogram_bins[i] = e.positive_histogram_bins[i - 1];
          e.positive_histogram_breadth[i] = e.positive_histogram_breadth[i - 1];
          e.positive_frames[i] = e.positive_frames[i - 1];
        }
        e.positive_scores[0] = quality.score;
        e.positive_completeness[0] = quality.completeness;
        e.positive_histogram_span[0] = quality.histogram_span;
        e.positive_histogram_bins[0] = quality.histogram_populated_bins;
        e.positive_histogram_breadth[0] = quality.histogram_breadth;
        e.positive_frames[0] = frame;
        e.positive_count = std::min(e.positive_count + 1, unsigned(e.positive_scores.size()));
        e.value.candidate_quality_score = *std::min_element(e.positive_scores.begin(), e.positive_scores.begin() + e.positive_count);
        e.value.retained_quality_score = *std::max_element(e.positive_scores.begin(), e.positive_scores.begin() + e.positive_count);
        e.value.candidate_completeness = *std::min_element(e.positive_completeness.begin(), e.positive_completeness.begin() + e.positive_count);
        e.value.retained_completeness = *std::max_element(e.positive_completeness.begin(), e.positive_completeness.begin() + e.positive_count);
        e.value.candidate_histogram_span = *std::min_element(e.positive_histogram_span.begin(), e.positive_histogram_span.begin() + e.positive_count);
        e.value.retained_histogram_span = *std::max_element(e.positive_histogram_span.begin(), e.positive_histogram_span.begin() + e.positive_count);
        e.value.candidate_histogram_bins = *std::min_element(e.positive_histogram_bins.begin(), e.positive_histogram_bins.begin() + e.positive_count);
        e.value.retained_histogram_bins = *std::max_element(e.positive_histogram_bins.begin(), e.positive_histogram_bins.begin() + e.positive_count);
        e.value.candidate_histogram_breadth = *std::min_element(e.positive_histogram_breadth.begin(), e.positive_histogram_breadth.begin() + e.positive_count);
        e.value.retained_histogram_breadth = *std::max_element(e.positive_histogram_breadth.begin(), e.positive_histogram_breadth.begin() + e.positive_count);
        if (e.value.good_samples >= std::max(options_.good_samples_required, 1u))
          e.value.confirmed_good = true;
      } else {
        e.value.good_samples = 0;
        e.value.bad_samples = std::min(e.value.bad_samples + 1, std::max(options_.bad_samples_required, 1u));
        if (e.value.bad_samples >= std::max(options_.bad_samples_required, 1u)) {
          e.value.confirmed_good = false;
          e.positive_count = 0;
          e.value.candidate_quality_score = e.value.retained_quality_score = 0;
          e.value.candidate_completeness = e.value.retained_completeness = 0;
          e.value.candidate_histogram_span = e.value.retained_histogram_span = 0;
          e.value.candidate_histogram_bins = e.value.retained_histogram_bins = 0;
          e.value.candidate_histogram_breadth = e.value.retained_histogram_breadth = 0;
        }
      }
    }

    void erase(std::uint64_t id) {
      candidates_.erase(id);
      if (pending_.incumbent == id || pending_.challenger == id)
        pending_ = {};
    }

    // What a capture set did on one Present: `active`, work that makes its
    // end-of-frame copy current (raster, mesh or a transfer that is depth of
    // its own); `drew`, raster or mesh work only.
    struct set_presence {
      bool active = false, drew = false;
    };

    // Present coverage, reported once per depth-active Present after current-copy
    // routing. `incumbent` is the routed anchor (0: none), `current` whether this
    // Present got a ready current copy from the anchor's capture set and `anchor`
    // what that set did. presence(id) answers the same for each candidate's
    // prospective capture set. Only the latest frame of each kind is kept; a
    // comparison judges the frames since its own start, so there is no window,
    // count or threshold here.
    template <class SetPresence>
    void observe_present(std::uint64_t frame, std::uint64_t incumbent, bool current, set_presence anchor, SetPresence &&presence) {
      // Incumbent-independent gaps for recovery and initial selection: an
      // active candidate whose set had no current-copy work on a Present some
      // confirmed-good candidate's set drew. Kept across incumbent changes.
      bool good_drew = false;
      std::vector<std::pair<std::uint64_t, set_presence>> sets;
      sets.reserve(candidates_.size());
      for (const auto &item : candidates_) {
        sets.emplace_back(item.first, presence(item.first));
        good_drew = good_drew || (sets.back().second.drew && good(item.second, frame));
      }
      const auto of = [&sets](std::uint64_t id) {
        for (const auto &entry : sets) if (entry.first == id) return entry.second;
        return set_presence {};
      };
      if (good_drew)
        for (auto &item : candidates_)
          if (active(item.second, frame) && !of(item.first).active)
            item.second.coverage.gap = frame;
      if (incumbent != coverage_incumbent_) {
        for (auto &item : candidates_)
          item.second.coverage.missed = item.second.coverage.filled = 0;
        coverage_incumbent_ = incumbent;
      }
      // A Present on which the anchor's set had current-copy work but no usable
      // copy is a capture fault, not a gap in the incumbent's rendering cadence;
      // it neither blocks nor favors others.
      const bool uncovered = !current && !anchor.active;
      if (!incumbent || (!current && !uncovered))
        return;
      for (auto &item : candidates_) {
        if (item.first == incumbent)
          continue;
        const auto set = of(item.first);
        // Missing is judged by current-copy work, so a set fed by a copy or
        // resolve every Present is not blocked; covering a gap needs a draw,
        // so a copy can never earn a coverage win.
        if (current && !set.active)
          item.second.coverage.missed = frame;
        else if (uncovered && set.drew)
          item.second.coverage.filled = frame;
      }
    }

    const candidate_evidence *evidence(std::uint64_t id) const {
      const auto found = candidates_.find(id);
      return found == candidates_.end() ? nullptr : &found->second.evidence.value;
    }

    double comparison_score(std::uint64_t id) const {
      const auto found = candidates_.find(id);
      return found == candidates_.end() ? 0 : score(found->second, false);
    }

    double retention_score(std::uint64_t id) const {
      const auto found = candidates_.find(id);
      return found == candidates_.end() ? 0 : score(found->second, true);
    }

    float source_coverage(std::uint64_t id) const {
      const auto found = candidates_.find(id);
      return found == candidates_.end() ? 0 : coverage(found->second.info);
    }

    // Lets the sampler finish a live comparison within its existing probe
    // deadline. Reading this never adds wins or changes round-robin order.
    std::uint64_t pending_challenger_id() const { return pending_.challenger; }
    // The challenger whose comparison the last choose() found won: it takes
    // over on that Present, or waits for its next rendered Present (below).
    // The wait ends by the takeover, or within the activity grace once it
    // stops rendering, since an inactive challenger drops the comparison.
    std::uint64_t won_challenger_id() const {
      return pending_.wins >= std::max(options_.comparison_samples_required, 1u) ? pending_.challenger : 0;
    }

    // Choosing tracks relative superiority across fresh captures. Repeated
    // presents with the same evidence never manufacture comparison rounds.
    // `depth_present`: this call is for the Present that advanced `frame` (one
    // that rendered depth). A won comparison takes over only on such a Present
    // on which the challenger itself rendered, so the first Present it anchors
    // has its own current copy. On any other Present (another member of its
    // A/B/C set rendered, or none) the new anchor would have none and could not
    // hold the old anchor's depth, so the Present would be mono: the incumbent
    // keeps it (current or held) and the won comparison waits for the
    // challenger's next rendered Present.
    selection_result choose(std::uint64_t frame, std::uint64_t current_id = 0, bool depth_present = true) {
      const auto current = candidates_.find(current_id);
      const bool current_active = current != candidates_.end() && active(current->second, frame);
      const bool current_good = current_active && good(current->second, frame);

      if (current_good) {
        const candidate_state *challenger = nullptr;
        unsigned challenger_priority = 0;
        bool coverage_only = false;
        for (const auto &item : candidates_) {
          const auto &candidate = item.second;
          if (item.first == current_id || !active(candidate, frame) || !comparison_ready(candidate, frame))
            continue;
          if (candidate.evidence.value.candidate_quality_score + options_.comparison_quality_tolerance <
              current->second.evidence.value.retained_quality_score)
            continue;
          const float completeness_gain = candidate.evidence.value.candidate_completeness - current->second.evidence.value.retained_completeness;
          const auto &candidate_evidence = candidate.evidence.value;
          const auto &current_evidence = current->second.evidence.value;
          const auto candidate_content = content_key(candidate, false);
          const auto current_content = content_key(current->second, true);
          // A protected, globally anchored completeness class prevents a chain
          // of individually small support losses from buying higher resolution.
          // Within one class, useful native coverage outranks histogram breadth.
          if (completeness_class(candidate, false) < completeness_class(current->second, true))
            continue;
          const auto candidate_coverage = coverage(candidate.info), current_coverage = coverage(current->second.info);
          const bool fuller = completeness_gain >= options_.completeness_margin && candidate_content > current_content;
          const bool broader = candidate_coverage == current_coverage &&
            completeness_gain >= -options_.histogram_completeness_tolerance &&
            candidate_content >= current_content + 100 &&
            histogram_broader(candidate_evidence.candidate_histogram_span, candidate_evidence.candidate_histogram_bins,
                              current_evidence.retained_histogram_span, current_evidence.retained_histogram_bins);
          // Within a completeness class, coverage over Presents outranks
          // resolution and histogram: a source that missed a Present the
          // incumbent showed would leave that Present flat, while one that drew
          // where the incumbent left a Present without depth covers more.
          const auto since = comparison_start(candidate, current_id);
          if (!fuller && candidate.coverage.missed > since)
            continue;
          const bool covers_more = candidate.coverage.filled > since;
          // Comparable useful support may reclaim native coverage even when
          // the smaller source has a broader histogram or one more occupied
          // sample. Lower resolution needs materially fuller support or more
          // Presents covered.
          const bool sharper = !fuller && !broader && completeness_gain > -options_.completeness_margin &&
            candidate_coverage > current_coverage &&
            resolution_score(candidate, false) >= resolution_score(current->second, true) + options_.comparison_margin;
          if (!fuller && !broader && !sharper && !covers_more)
            continue;
          // Material completeness takes precedence. Histogram differences can
          // replace peers at the same coverage, but cannot demote a useful native
          // source. Every healthy replacement also needs fresh paired wins.
          const unsigned priority = fuller ? 4 : covers_more ? 3 : broader ? 2 : 1;
          if (!challenger || priority > challenger_priority || (priority == challenger_priority && better(candidate, *challenger, frame))) {
            challenger = &candidate;
            challenger_priority = priority;
            coverage_only = !fuller && !broader && !sharper;
          }
        }
        if (challenger && current->second.evidence.value.quality.kind == content_kind::useful) {
          const auto incumbent_frame = current->second.evidence.value.sampled_frame;
          const auto challenger_frame = challenger->evidence.value.sampled_frame;
          if (pending_.incumbent != current_id || pending_.challenger != challenger->info.id)
            pending_ = {current_id, challenger->info.id, incumbent_frame, challenger_frame, 1, comparison_start(*challenger, current_id), frame};
          else if (incumbent_frame > pending_.incumbent_frame && challenger_frame > pending_.challenger_frame) {
            // Coverage alone must recur within every paired interval, so one
            // missed incumbent Present can supply at most one win.
            if (coverage_only && challenger->coverage.filled <= pending_.interval) {
              pending_ = {};
              return {current_id, "retained confirmed scene depth"};
            }
            pending_.incumbent_frame = incumbent_frame;
            pending_.challenger_frame = challenger_frame;
            pending_.interval = frame;
            pending_.wins = std::min(pending_.wins + 1, std::max(options_.comparison_samples_required, 1u));
          }
          if (pending_.wins >= std::max(options_.comparison_samples_required, 1u) &&
              !(depth_present && challenger->info.last_seen_frame == frame))
            return {current_id, "waiting for the challenger's next rendered Present"};
          if (pending_.wins >= std::max(options_.comparison_samples_required, 1u))
            return {challenger->info.id, challenger_priority == 4 ? "sustained fuller nonendpoint depth support" :
                                        challenger_priority == 3 ? "sustained comparable depth on more Presents" :
                                        challenger_priority == 2 ? "sustained broader nonendpoint depth histogram" :
                                        "sustained comparable depth with better source coverage"};
          return {current_id, "waiting for fresh paired depth comparisons"};
        }
        // Losing the advantage, or a currently inconclusive incumbent capture,
        // invalidates the pending comparison rather than banking old wins.
        pending_ = {};
        return {current_id, "retained confirmed scene depth"};
      }

      // Initial selection and recovery from invalid/stale/destroyed sources
      // remain immediate once another source has confirmed useful content.
      // Among confirmed sources the completeness class ranks first, as for a
      // challenger; within it a source whose set left a Present without depth
      // that another confirmed source drew, since its comparison start, ranks
      // after one that did not, so recovery does not pick a source drawn only
      // on some Presents over one drawn on all of them.
      pending_ = {};
      const candidate_state *best = nullptr;
      for (const auto &item : candidates_) {
        const auto &candidate = item.second;
        if (active(candidate, frame) && (!best || recovery_better(candidate, *best, frame)))
          best = &candidate;
      }
      if (best && good(*best, frame))
        return {best->info.id, "confirmed scene depth outranks workload"};
      // Flat menus and unknown readbacks must not walk through resources. Keep
      // the current live source until there is positive evidence for another.
      if (current_active)
        return {current_id, "retained source while content is inconclusive"};
      return best ? selection_result {best->info.id, "workload fallback pending content"} : selection_result {};
    }

    // Called only while the native sampler is idle. Until the current source
    // covers the output, alternate full-output discovery with an ordinary visit.
    // Each lane rotates by lifetime ID. Separate cursors keep even unusable
    // native candidates from starving smaller sources; probing is not evidence.
    std::uint64_t next_probe(std::uint64_t frame, std::uint64_t current_id = 0) {
      const auto next = [&](std::uint64_t &cursor, bool native_only) {
        std::uint64_t first = 0;
        for (const auto &item : candidates_) {
          if (!active(item.second, frame) || (native_only && coverage(item.second.info) != 1.f))
            continue;
          if (!first)
            first = item.first;
          if (item.first > cursor) {
            cursor = item.first;
            return item.first;
          }
        }
        cursor = first;
        return first;
      };
      if (native_probe_due_ && source_coverage(current_id) < 1.f) {
        if (const auto id = next(last_native_probed_, true)) {
          native_probe_due_ = false;
          return id;
        }
      }
      native_probe_due_ = true;
      return next(last_probed_, false);
    }

  private:
    struct pending_challenge {
      std::uint64_t incumbent = 0, challenger = 0;
      std::uint64_t incumbent_frame = 0, challenger_frame = 0;
      unsigned wins = 0;
      // First frame this comparison judges, and the end of its last paired interval.
      std::uint64_t since = 0, interval = 0;
    };

    // Latest depth-active Presents, relative to the routed incumbent, where this
    // candidate's prospective capture set had no current-copy work while the
    // incumbent showed current depth (missed), or drew while the incumbent's
    // set had neither current-copy work nor a current copy (filled); and,
    // independent of any incumbent, where it was active but its set had no
    // current-copy work while a confirmed-good candidate's set drew (gap).
    struct coverage_marks {
      std::uint64_t missed = 0, filled = 0, gap = 0;
    };

    struct evidence_state {
      candidate_evidence value;
      bool sampled = false;
      std::array<float, 3> positive_scores {};
      std::array<float, 3> positive_completeness {};
      std::array<float, 3> positive_histogram_span {}, positive_histogram_breadth {};
      std::array<unsigned, 3> positive_histogram_bins {};
      std::array<std::uint64_t, 3> positive_frames {};
      unsigned positive_count = 0;
    };
    struct candidate_state {
      candidate_info info;
      evidence_state evidence;
      coverage_marks coverage;
    };

    bool active(const candidate_state &candidate, std::uint64_t frame) const {
      // A clear-only present updates resource activity, but did not render scene depth.
      return candidate.info.draws != 0 && frame >= candidate.info.last_seen_frame &&
             frame - candidate.info.last_seen_frame <= options_.active_grace_frames;
    }

    bool good(const candidate_state &candidate, std::uint64_t frame) const {
      return candidate.evidence.sampled && candidate.evidence.value.confirmed_good &&
             frame >= candidate.evidence.value.sampled_frame && frame - candidate.evidence.value.sampled_frame <= options_.evidence_stale_frames;
    }

    bool comparison_ready(const candidate_state &candidate, std::uint64_t frame) const {
      if (!good(candidate, frame) || candidate.evidence.value.good_samples < std::max(options_.comparison_samples_required, 1u) ||
          candidate.evidence.positive_count < std::min(std::max(options_.comparison_samples_required, 1u), 3u))
        return false;
      const auto oldest = candidate.evidence.positive_frames[candidate.evidence.positive_count - 1];
      return frame >= oldest && frame - oldest <= options_.evidence_stale_frames;
    }

    // A comparison judges coverage from the challenger's oldest comparison
    // capture, or from the start of its pending comparison when that is older.
    // Callers have checked comparison_ready().
    std::uint64_t comparison_start(const candidate_state &candidate, std::uint64_t current_id) const {
      const auto oldest = candidate.evidence.positive_frames[std::min(std::max(options_.comparison_samples_required, 1u), 3u) - 1];
      return pending_.incumbent == current_id && pending_.challenger == candidate.info.id ? std::min(oldest, pending_.since) : oldest;
    }

    bool histogram_broader(float span, unsigned bins, float reference_span, unsigned reference_bins) const {
      return span >= reference_span + options_.histogram_span_margin && bins >= 3 &&
             bins >= reference_bins && bins - reference_bins >= options_.histogram_bin_margin;
    }

    // A gap since the candidate's oldest retained positive capture (its
    // comparison start once it has three).
    static bool gapped(const candidate_state &candidate) {
      const auto &e = candidate.evidence;
      const auto since = e.positive_count ? e.positive_frames[e.positive_count - 1] : 0;
      return candidate.coverage.gap > since;
    }

    bool recovery_better(const candidate_state &a, const candidate_state &b, std::uint64_t frame) const {
      if (good(a, frame) && good(b, frame)) {
        const auto ac = completeness_class(a, false), bc = completeness_class(b, false);
        if (ac != bc)
          return ac > bc;
        if (gapped(a) != gapped(b))
          return !gapped(a);
      }
      return better(a, b, frame);
    }

    bool better(const candidate_state &a, const candidate_state &b, std::uint64_t frame) const {
      const bool ag = good(a, frame), bg = good(b, frame);
      if (ag != bg)
        return ag;
      if (ag) {
        const double as = score(a, false);
        const double bs = score(b, false);
        if (as != bs)
          return as > bs;
      } else {
        if (a.info.vertices != b.info.vertices)
          return a.info.vertices > b.info.vertices;
        if (a.info.draws != b.info.draws)
          return a.info.draws > b.info.draws;
      }
      return a.info.id < b.info.id;
    }

    static double workload_bonus(const candidate_info &info) {
      return .005 * std::min(std::log2(1.0 + double(info.draws)) / 20.0, 1.0);
    }

    static float coverage(const candidate_info &info) {
      if (!info.width || !info.height || !info.output_width || !info.output_height)
        return 0;
      double width = info.width, height = info.height;
      // Padded/oversized allocations qualify as native only when the caller has
      // supplied a valid active viewport within that allocation.
      if (std::isfinite(info.viewport_width) && std::isfinite(info.viewport_height) &&
          info.viewport_width >= 1 && info.viewport_height >= 1 &&
          info.viewport_width <= info.width && info.viewport_height <= info.height) {
        width = info.viewport_width;
        height = info.viewport_height;
      }
      const double x = width / info.output_width, y = height / info.output_height;
      if (x > 1 || y > 1 || std::abs(width / height - double(info.output_width) / info.output_height) > .1)
        return 0;
      return float(std::min(x, y));
    }

    static double resolution_score(const candidate_state &candidate, bool incumbent) {
      const auto &evidence = candidate.evidence.value;
      // Content qualification is a prerequisite in every caller. Resolution is
      // a bounded preference among credible candidates, never proof of depth.
      // Smoothness and workload cannot overwhelm native-vs-half coverage.
      return .20 * coverage(candidate.info) +
             .05 * (incumbent ? evidence.retained_quality_score : evidence.candidate_quality_score) +
             workload_bonus(candidate.info);
    }

    double completeness_class(const candidate_state &candidate, bool incumbent) const {
      const auto &evidence = candidate.evidence.value;
      const double completeness = incumbent ? evidence.retained_completeness : evidence.candidate_completeness;
      // Reuse the material support margin; no game-specific or new threshold.
      // Upper-inclusive classes keep .99 and 1.0 together at the default .20,
      // while preventing the 1.0 -> .81 -> .62 resolution staircase.
      return std::isfinite(options_.completeness_margin) && options_.completeness_margin > 0 ?
        std::ceil(completeness / options_.completeness_margin) : completeness;
    }

    static std::int64_t content_key(const candidate_state &candidate, bool incumbent) {
      const auto &evidence = candidate.evidence.value;
      // Within equal completeness class and coverage, this quantized content
      // order prevents histogram/support trades from cycling through fixed
      // sources. Min/max histories conservatively bound actual capture keys.
      return std::llround(10000.0 *
        (.65 * (incumbent ? evidence.retained_completeness : evidence.candidate_completeness) +
         .10 * (incumbent ? evidence.retained_histogram_breadth : evidence.candidate_histogram_breadth)));
    }

    static double score(const candidate_state &candidate, bool incumbent) {
      const auto &evidence = candidate.evidence.value;
      return .65 * (incumbent ? evidence.retained_completeness : evidence.candidate_completeness) +
             .10 * (incumbent ? evidence.retained_histogram_breadth : evidence.candidate_histogram_breadth) +
             resolution_score(candidate, incumbent);
    }

    policy_options options_;
    std::map<std::uint64_t, candidate_state> candidates_;
    std::uint64_t last_probed_ = 0, last_native_probed_ = 0;
    bool native_probe_due_ = true;
    pending_challenge pending_;
    std::uint64_t coverage_incumbent_ = 0;
  };
}  // namespace sunshine_depth

/**
 * @file src/platform/windows/local_shutter_slot_timing.h
 * @brief Opt-in slot timing diagnostics of the local shutter presenter.
 */
#pragma once

// standard includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>

// platform includes
#include <windows.h>

// local includes
#include "open3d_emitter.h"
#include "shutter_presentation_clock.h"
#include "shutter_slot_timing.h"
#include "src/logging.h"
#include "src/utility.h"

namespace platf::dxgi {
  /** Internals of run_local_presenter(), for display_vram.cpp only. */
  namespace local_presenter {
    using namespace std::literals;

    // Opt-in diagnostics only. All armed-frame work uses fixed-size storage; formatting and
    // logging happen after the emitter closes. These host timestamps do not measure USB arrival,
    // IR reception, panel scanout or the glasses' optical response.
    struct local_shutter_slot_timing_t {
      enum class outcome_e : std::size_t {
        present_failed,
        present_id_failed,
        cold_unsignalled,
        feedback_cancelled,
        feedback_unconfirmed,
        tracker_rejected,
        eye_clock_rejected,
        expired_before_emit,
        cancelled_before_emit,
        signal_failed,
        signalled,
        count,
      };
      static constexpr std::array outcome_names {
        "present_failed",
        "present_id_failed",
        "cold_unsignalled",
        "feedback_cancelled",
        "feedback_unconfirmed",
        "tracker_rejected",
        "eye_clock_rejected",
        "expired_before_emit",
        "cancelled_before_emit",
        "signal_failed",
        "signalled",
      };

      enum metric_e : std::size_t {
        present_call,
        feedback_receipt_age,
        signal_entry_age,
        host_write_begin_age,
        signal_prepare,
        host_write_call,
        signal_call,
        boundary_to_next_submit,
        write_end_to_next_submit,
        metric_count,
      };

      static constexpr std::array metric_names {
        "present_call_us",
        "feedback_receipt_age_us",
        "signal_entry_age_us",
        "host_write_begin_age_us",
        "signal_prepare_us",
        "host_write_call_us",
        "signal_call_us",
        "boundary_to_next_submit_us",
        "write_end_to_next_submit_us",
      };

      struct previous_t {
        unsigned logical_slot = 0;
        std::uint64_t segment = 0;
        std::int64_t boundary = 0, write_end = 0;
      };

      struct sample_t {
        unsigned logical_slot = 0;
        bool acquiring = false, delta_known = false;
        std::uint64_t segment = 0;
        outcome_e outcome = outcome_e::present_failed;
        UINT requested = 0;
        std::uint32_t refresh_delta = 0;
        detail::shutter_feedback_t feedback {};
        std::int64_t present_begin = 0, present_end = 0, receipt = 0, boundary = 0;
        std::int64_t signal_begin = 0, signal_end = 0;
        platf::open3d::signal_timing_t write {};
        std::optional<previous_t> previous;
      };

      struct signed_summary_t {
        std::uint64_t count = 0, negative = 0, invalid = 0;
        double minimum = 0, mean = 0, maximum = 0;

        void observe(double value) noexcept {
          if (!std::isfinite(value)) {
            ++invalid;
            return;
          }
          if (!count) {
            minimum = maximum = value;
          } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
          }
          ++count;
          negative += value < 0;
          mean += (value - mean) / static_cast<double>(count);
        }
      };

      struct slot_t {
        std::array<detail::shutter_timing_aggregate_t, metric_count> metrics;
        std::array<std::uint64_t, static_cast<std::size_t>(outcome_e::count)> outcomes {};
        // Actual deltas 0..16 and one overflow bucket; supported repeats are 1..4.
        std::array<std::uint64_t, 18> refresh_deltas {};
        std::array<sample_t, 32> first_armed_samples {};
        signed_summary_t submission_to_boundary;
        signed_summary_t write_end_to_next_submit;
        std::uint64_t cold_presents = 0, armed_presents = 0, raw_omitted = 0;
        std::uint64_t invalid_write_metadata = 0, held_images = 0, extra_refreshes = 0;
        std::size_t raw_count = 0;
      };

      std::array<slot_t, 2> slots;
      std::optional<previous_t> previous;
      std::optional<std::uint64_t> last_signalled_segment;
      std::uint64_t segment = 0, recovery_attempts = 0, signalled_segments = 0;
      std::int64_t frequency;
      unsigned repeats;

      local_shutter_slot_timing_t(std::int64_t qpc_frequency, unsigned eye_repeats):
          frequency(qpc_frequency),
          repeats(eye_repeats) {}

      double microseconds(std::int64_t begin, std::int64_t end) const noexcept {
        return static_cast<double>((static_cast<long double>(end) - begin) * 1000000.0L / frequency);
      }

      void recovering() noexcept {
        previous.reset();
        ++segment;
        ++recovery_attempts;
      }

      void finish(const sample_t &sample) noexcept {
        auto &slot = slots[sample.logical_slot];
        if (sample.acquiring) {
          ++slot.cold_presents;
          return;
        }
        ++slot.armed_presents;
        ++slot.outcomes[static_cast<std::size_t>(sample.outcome)];
        if (slot.raw_count < slot.first_armed_samples.size()) {
          slot.first_armed_samples[slot.raw_count++] = sample;
        } else {
          ++slot.raw_omitted;
        }
        slot.metrics[present_call].observe(microseconds(sample.present_begin, sample.present_end));
        if (sample.previous && sample.previous->segment == sample.segment) {
          auto &prior = slots[sample.previous->logical_slot];
          prior.metrics[boundary_to_next_submit].observe(microseconds(sample.previous->boundary, sample.present_begin));
          if (sample.previous->write_end > 0) {
            const auto overlap = microseconds(sample.previous->write_end, sample.present_begin);
            prior.write_end_to_next_submit.observe(overlap);
            // The signed value is kept separately from the nonnegative histogram.
            if (overlap >= 0) {
              prior.metrics[write_end_to_next_submit].observe(overlap);
            }
          }
        }
        // A rejected tracker has no trusted current reconstructed boundary. Its raw evidence
        // remains in the bounded sample record and existing failure log, excluded from ages.
        if (sample.boundary > 0) {
          slot.metrics[feedback_receipt_age].observe(microseconds(sample.boundary, sample.receipt));
          slot.submission_to_boundary.observe(microseconds(sample.present_begin, sample.boundary));
        }
        const bool write_metadata_valid = sample.write.valid() && sample.write.qpc_frequency == frequency &&
                                          sample.signal_begin >= sample.receipt && sample.signal_end >= sample.signal_begin &&
                                          sample.write.write_begin_qpc >= sample.signal_begin && sample.write.write_end_qpc <= sample.signal_end;
        if (sample.signal_begin > 0) {
          slot.metrics[signal_call].observe(microseconds(sample.signal_begin, sample.signal_end));
          if (sample.boundary > 0) {
            slot.metrics[signal_entry_age].observe(microseconds(sample.boundary, sample.signal_begin));
          }
          if (write_metadata_valid) {
            slot.metrics[signal_prepare].observe(microseconds(sample.signal_begin, sample.write.write_begin_qpc));
            slot.metrics[host_write_call].observe(microseconds(sample.write.write_begin_qpc, sample.write.write_end_qpc));
            if (sample.boundary > 0) {
              slot.metrics[host_write_begin_age].observe(microseconds(sample.boundary, sample.write.write_begin_qpc));
            }
          } else {
            ++slot.invalid_write_metadata;
          }
        }
        if (sample.outcome == outcome_e::signalled) {
          if (sample.delta_known) {
            ++slot.refresh_deltas[std::min<std::size_t>(sample.refresh_delta, slot.refresh_deltas.size() - 1)];
            if (sample.refresh_delta > repeats) {
              ++slot.held_images;
              slot.extra_refreshes += sample.refresh_delta - repeats;
            }
          }
          if (!last_signalled_segment || *last_signalled_segment != sample.segment) {
            ++signalled_segments;
            last_signalled_segment = sample.segment;
          }
          previous = previous_t {sample.logical_slot, sample.segment, sample.boundary, write_metadata_valid ? sample.write.write_end_qpc : 0};
        }
      }

      void log_summary(bool flip_eyes, int age_budget_us) const {
        BOOST_LOG(info) << "Local shutter slot timing: enabled=true qpc_frequency="sv << frequency
                        << " repeats="sv << repeats << " host_age_budget_us="sv << age_budget_us
                        << " recovery_attempts="sv << recovery_attempts << " signalled_segments="sv << signalled_segments
                        << " histogram_bin_us=100 histogram_regular_limit_us=12800 raw_limit_per_slot=32; "sv
                           "armed samples only; boundary ages exclude rejected trackers; host WriteFile entry/end are not USB or optical timestamps."sv;
        for (unsigned index = 0; index < slots.size(); ++index) {
          const auto &slot = slots[index];
          const auto logical = index == 0 ? 'L' : 'R';
          const auto physical = ((index == 0) != flip_eyes) ? 'L' : 'R';
          std::ostringstream outcomes, deltas;
          for (std::size_t i = 0; i < outcome_names.size(); ++i) {
            outcomes << (i ? "," : "") << outcome_names[i] << ':' << slot.outcomes[i];
          }
          for (std::size_t i = 0; i < slot.refresh_deltas.size(); ++i) {
            deltas << (i ? "," : "") << i << ':' << slot.refresh_deltas[i];
          }
          BOOST_LOG(info) << "Local shutter slot timing counts: logical="sv << logical << " nominal_physical="sv << physical
                          << " cold_presents="sv << slot.cold_presents << " armed_presents="sv << slot.armed_presents
                          << " invalid_write_metadata="sv << slot.invalid_write_metadata << " held_images="sv << slot.held_images
                          << " extra_refreshes="sv << slot.extra_refreshes << " raw_saved="sv << slot.raw_count << " raw_omitted="sv << slot.raw_omitted
                          << " outcomes="sv << outcomes.str() << " refresh_delta_counts_17_is_overflow="sv << deltas.str() << '.';
          for (std::size_t metric = 0; metric < metric_names.size(); ++metric) {
            const auto &values = slot.metrics[metric];
            const auto p50 = values.percentile(50), p95 = values.percentile(95), p99 = values.percentile(99);
            BOOST_LOG(info) << "Local shutter slot timing metric: logical="sv << logical << " nominal_physical="sv << physical
                            << " metric="sv << metric_names[metric] << " count="sv << values.count() << " invalid="sv << values.invalid_count()
                            << " min="sv << values.minimum_us() << " mean="sv << values.mean_us() << " max="sv << values.maximum_us()
                            << " p50_hist_bound="sv << p50.upper_bound_us << " p50_overflow="sv << p50.overflow
                            << " p95_hist_bound="sv << p95.upper_bound_us << " p95_overflow="sv << p95.overflow
                            << " p99_hist_bound="sv << p99.upper_bound_us << " p99_overflow="sv << p99.overflow << '.';
          }
          const auto &signed_values = slot.submission_to_boundary;
          BOOST_LOG(info) << "Local shutter slot timing signed: logical="sv << logical << " metric=submission_to_boundary_us count="sv << signed_values.count
                          << " negative="sv << signed_values.negative << " invalid="sv << signed_values.invalid
                          << " min="sv << signed_values.minimum << " mean="sv << signed_values.mean << " max="sv << signed_values.maximum << '.';
          const auto &overlap = slot.write_end_to_next_submit;
          BOOST_LOG(info) << "Local shutter slot timing signed: logical="sv << logical << " metric=write_end_to_next_submit_us count="sv << overlap.count
                          << " negative="sv << overlap.negative << " invalid="sv << overlap.invalid
                          << " min="sv << overlap.minimum << " mean="sv << overlap.mean << " max="sv << overlap.maximum << '.';
          for (std::size_t i = 0; i < slot.raw_count; ++i) {
            const auto &raw = slot.first_armed_samples[i];
            BOOST_LOG(info) << "Local shutter slot timing raw: logical="sv << logical << " nominal_physical="sv << physical << " index="sv << i
                            << " segment="sv << raw.segment << " outcome="sv << outcome_names[static_cast<std::size_t>(raw.outcome)]
                            << " requested="sv << raw.requested << " observed="sv << raw.feedback.present_count
                            << " feedback_status="sv << static_cast<unsigned>(raw.feedback.status) << " feedback_hr="sv << util::log_hex(raw.feedback.result_code)
                            << " refresh="sv << raw.feedback.refresh_count << " sync_refresh="sv << raw.feedback.sync_refresh_count << " raw_sync_qpc="sv << raw.feedback.sync_qpc
                            << " delta_known="sv << raw.delta_known << " refresh_delta="sv << raw.refresh_delta
                            << " present_begin_qpc="sv << raw.present_begin << " present_end_qpc="sv << raw.present_end << " receipt_qpc="sv << raw.receipt
                            << " boundary_qpc="sv << raw.boundary << " signal_begin_qpc="sv << raw.signal_begin << " signal_end_qpc="sv << raw.signal_end
                            << " write_begin_qpc="sv << raw.write.write_begin_qpc << " write_end_qpc="sv << raw.write.write_end_qpc
                            << " write_qpc_frequency="sv << raw.write.qpc_frequency << " write_pending="sv << raw.write.write_pending << '.';
          }
        }
      }
    };
  }  // namespace local_presenter
}  // namespace platf::dxgi

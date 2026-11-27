/**
 * @file src/offline_sbs_worker_probe.h
 * @brief Worker-private FFprobe media contracts, exact timeline checks, and stream inventory.
 *
 * Probing is strict and bounded: frame and packet documents are parsed incrementally, and every
 * timestamp comparison uses exact rational arithmetic. The same contracts gate the source before
 * decoding and the intermediate and final outputs before publication.
 */
#pragma once

#include "offline_sbs_worker.h"

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace offline_sbs::worker_detail {
  // The largest source sequence an attested frame list or generated AVExpr may contain.
  inline constexpr std::uint64_t max_sequence = 9999999999ull;

  std::int64_t nonnegative_tick_difference(
    std::int64_t later,
    std::int64_t earlier,
    std::string_view description
  );

  bool rational_ticks_within_actual_ticks(
    std::int64_t source_ticks,
    rational_t source_time_base,
    std::int64_t actual_ticks,
    rational_t actual_time_base,
    std::uint64_t tolerance_ticks
  );

  long double rational_error_in_actual_ticks(
    std::int64_t source_ticks,
    rational_t source_time_base,
    std::int64_t actual_ticks,
    rational_t actual_time_base
  );

  void require_millisecond_or_finer(
    rational_t time_base,
    std::string_view description
  );

  void validate_timeline_contract(
    const media_contract_t &source,
    const media_contract_t &actual,
    std::uint64_t tolerance_actual_ticks
  );

  nlohmann::json timeline_error_report(
    const media_contract_t &source,
    const media_contract_t &actual
  );

  nlohmann::json media_contract_json(const media_contract_t &media);

  media_contract_t probe_media(
    const worker_spec_t &spec,
    const std::filesystem::path &media_path,
    const std::filesystem::path &json_path,
    const std::filesystem::path &log_path,
    bool source_packet_durations
  );

  // Exact intermediate-stream packet check; see validate_encoded_packet_timeline().
  void probe_encoded_packet_timeline(
    const worker_spec_t &spec,
    const media_contract_t &source,
    const std::filesystem::path &encoded_video,
    const std::filesystem::path &log_path
  );

  void validate_hdr_equivalence(
    const media_contract_t &source,
    const media_contract_t &output
  );

  void validate_sdr_color_equivalence(
    const media_contract_t &source,
    const media_contract_t &output
  );

  struct packet_timing_t {
    std::optional<std::int64_t> pts;
    std::optional<std::int64_t> dts;
    std::optional<std::int64_t> duration;
  };

  struct stream_inventory_entry_t {
    std::int64_t index = -1;
    std::string type;
    std::string codec;
    rational_t time_base;
    nlohmann::json tags = nlohmann::json::object();
    nlohmann::json disposition = nlohmann::json::object();
    std::vector<packet_timing_t> packets;
  };

  struct chapter_inventory_entry_t {
    rational_t time_base;
    std::int64_t start = 0;
    std::int64_t end = 0;
    nlohmann::json tags = nlohmann::json::object();
  };

  struct stream_inventory_t {
    std::vector<stream_inventory_entry_t> streams;
    std::vector<chapter_inventory_entry_t> chapters;
    nlohmann::json format_tags = nlohmann::json::object();
  };

  stream_inventory_t parse_stream_inventory(const nlohmann::json &value);

  stream_inventory_t probe_stream_inventory(
    const worker_spec_t &spec,
    const std::filesystem::path &media,
    const std::filesystem::path &json_path,
    const std::filesystem::path &log_path
  );
}  // namespace offline_sbs::worker_detail

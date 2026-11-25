/** Versioned, bounded raw-frame handoff within the isolated offline worker. */
#pragma once

#include "generated/sbs_adaptive_state_contract.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace offline_sbs {
  inline std::uint64_t raw_source_byte_bound(std::uint64_t source_bytes, bool hdr, bool overlap = false) {
    if (!source_bytes || source_bytes > 16384ull * 16384 * 12)
      throw std::runtime_error("invalid raw source byte bound");
    return source_bytes * (overlap ? 2u : 1u) + (hdr ? source_bytes / 12 * 8 : 0) + 1024 * 1024;
  }

  // Worker/harness CPU raster reservation. Source leases include the decoder
  // snapshot and (for HDR) one FP16 upload conversion. At most three packed
  // snapshots coexist during overlap: encoder, handoff, and GPU readback.
  inline std::uint64_t raw_raster_byte_bound(std::uint64_t source_bytes,
                                           std::uint32_t width, std::uint32_t height,
                                           bool hdr, bool overlap) {
    if (!source_bytes || !width || !height || width > 16384 || height > 16384 ||
        source_bytes > 16384ull * 16384 * 12)
      throw std::runtime_error("invalid raw raster bound dimensions");
    const auto source = raw_source_byte_bound(source_bytes, hdr, overlap);
    const auto packed = std::uint64_t(width) * height * (hdr ? 8u : 4u) * (overlap ? 3u : 1u);
    // Bounded decoder pipe, encoder pipe, and one HDR planar conversion chunk.
    return source + packed + 64 * 1024 + (hdr ? 64 * 1024 : 0);
  }

  enum class raw_pixel_format : std::uint32_t { bgra8 = 1, gbrpf32le = 2, rgba16f = 3 };

  // The exact float32 value of a finite half float (exponent field below 31), built from its
  // bits: every half is representable, so no rounding is involved. The caller rejects
  // exponent 31 (infinities and NaNs).
  [[nodiscard]] inline float finite_half_to_float(const std::uint16_t half) noexcept {
    const std::uint32_t sign = std::uint32_t(half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 31u;
    const std::uint32_t mantissa = half & 1023u;
    if (exponent != 0u) {
      return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
    }
    // Zero and subnormal halves are mantissa * 2^-24, a normal float32.
    return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(float(mantissa) * 0x1p-24f));
  }

  struct raw_frame_header {
    std::uint32_t schema = 1;
    std::uint64_t sequence = 0;
    std::int64_t pts = 0;
    std::int64_t time_base_numerator = 0;
    std::int64_t time_base_denominator = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    raw_pixel_format format = raw_pixel_format::bgra8;
  };

  // All rows are top-down. BGRA is sRGB; float formats are finite linear scRGB
  // at the canonical reference white. Moving a frame transfers pixel ownership.
  struct raw_frame {
    raw_frame_header header;
    std::vector<std::uint8_t> bgra;
    std::vector<float> gbr32;
    std::vector<std::uint16_t> rgba16;

    std::uint64_t bytes() const {
      return bgra.size() + gbr32.size() * sizeof(float) + rgba16.size() * sizeof(std::uint16_t);
    }

    void validate(std::uint64_t byte_limit) const {
      const auto pixels = std::uint64_t(header.width) * header.height;
      if (header.schema != 1 || !header.sequence || !pixels ||
          header.width > 16384 || header.height > 16384 ||
          header.time_base_numerator <= 0 || header.time_base_denominator <= 0 ||
          bytes() > byte_limit) {
        throw std::runtime_error("invalid bounded raw-frame header");
      }
      const bool valid =
        (header.format == raw_pixel_format::bgra8 && bgra.size() == pixels * 4 && gbr32.empty() && rgba16.empty()) ||
        (header.format == raw_pixel_format::gbrpf32le && gbr32.size() == pixels * 3 && bgra.empty() && rgba16.empty()) ||
        (header.format == raw_pixel_format::rgba16f && rgba16.size() == pixels * 4 && bgra.empty() && gbr32.empty());
      if (!valid) {
        throw std::runtime_error("raw-frame payload does not match its declared format/geometry");
      }
    }
  };

  class raw_frame_channel {
  public:
    raw_frame_channel(std::size_t capacity, std::uint64_t byte_limit):
        capacity_(capacity), byte_limit_(byte_limit) {
      if (!capacity || !byte_limit) throw std::invalid_argument("empty raw-frame channel bound");
    }

    void publish(raw_frame frame) {
      frame.validate(byte_limit_);
      std::unique_lock lock(mutex_);
      wait(lock, [&] { return frames_.size() < capacity_ || finished_; });
      if (finished_ || frame.header.sequence != next_publish_)
        throw std::runtime_error("raw-frame publication sequence mismatch");
      ++next_publish_;
      frames_.push_back(std::move(frame));
      changed_.notify_all();
    }

    raw_frame receive(std::uint64_t sequence) {
      std::unique_lock lock(mutex_);
      wait(lock, [&] { return !frames_.empty() || finished_; });
      if (frames_.empty() || frames_.front().header.sequence != sequence || sequence != next_receive_)
        throw std::runtime_error("raw-frame receive sequence mismatch or premature EOF");
      auto frame = std::move(frames_.front());
      frames_.pop_front();
      ++next_receive_;
      changed_.notify_all();
      return frame;
    }

    // ACK is separate from receiving pixels: the consumer must read matching
    // adaptive-state evidence before the producer can replace that evidence.
    void acknowledge(std::uint64_t sequence) {
      std::lock_guard lock(mutex_);
      if (sequence != acknowledged_ + 1 || sequence >= next_receive_)
        throw std::runtime_error("raw-frame acknowledgement sequence mismatch");
      acknowledged_ = sequence;
      changed_.notify_all();
    }
    void wait_acknowledged(std::uint64_t sequence) {
      std::unique_lock lock(mutex_);
      wait(lock, [&] { return acknowledged_ >= sequence; });
    }
    void finish() {
      std::lock_guard lock(mutex_);
      finished_ = true;
      changed_.notify_all();
    }
    void cancel() noexcept {
      std::lock_guard lock(mutex_);
      cancelled_ = true;
      frames_.clear();
      changed_.notify_all();
    }

  private:
    template<class Predicate> void wait(std::unique_lock<std::mutex> &lock, Predicate ready) {
      if (!changed_.wait_for(lock, std::chrono::seconds(120), [&] { return cancelled_ || ready(); }))
        throw std::runtime_error("bounded raw-frame handoff timed out");
      if (cancelled_) throw std::runtime_error("bounded raw-frame handoff cancelled");
    }
    const std::size_t capacity_;
    const std::uint64_t byte_limit_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<raw_frame> frames_;
    std::uint64_t next_publish_ = 1;
    std::uint64_t next_receive_ = 1;
    std::uint64_t acknowledged_ = 0;
    bool finished_ = false;
    bool cancelled_ = false;
  };

  struct completion_field_t {
    std::string name;
    std::string type;
  };

  struct completion_header_t {
    std::uint32_t schema = 1;
    std::uint32_t adaptive_schema = 0;
    std::string adaptive_contract_sha256;
    std::string source;
    std::string capture;
    std::array<completion_field_t, sbs_adaptive_state::word_count> fields;
    std::string model;
    double pop_strength = 0;
    int depth_reuse_interval = 0;
    raw_pixel_format source_format = raw_pixel_format::bgra8;
    std::uint64_t source_frame_count = 0;
    std::uint64_t first_sequence = 1;
    bool sbs_enabled = false;

    void validate() const {
      constexpr std::size_t max_owned_header_bytes = 1024 * 1024;
      std::size_t owned_bytes = model.size();
      const auto count_owned = [&](const std::string &text) {
        if (text.size() > max_owned_header_bytes || owned_bytes > max_owned_header_bytes - text.size()) {
          throw std::runtime_error("completion header exceeds its owned byte bound");
        }
        owned_bytes += text.size();
      };
      if (owned_bytes > max_owned_header_bytes) {
        throw std::runtime_error("completion header exceeds its owned byte bound");
      }
      count_owned(adaptive_contract_sha256);
      count_owned(source);
      count_owned(capture);
      for (const auto &field : fields) {
        count_owned(field.name);
        count_owned(field.type);
      }
      if (schema != 1 || adaptive_schema != sbs_adaptive_state::schema_version || adaptive_contract_sha256 != sbs_adaptive_state::contract_canonical_sha256 || source != sbs_adaptive_state::source || capture != sbs_adaptive_state::capture) {
        throw std::runtime_error("completion header adaptive identity differs");
      }
      for (std::size_t index = 0; index < fields.size(); ++index) {
        if (fields[index].name != sbs_adaptive_state::fields[index].name || fields[index].type != sbs_adaptive_state::fields[index].json_type) {
          throw std::runtime_error("completion header adaptive field layout differs");
        }
      }
      if (model.empty() || !std::isfinite(pop_strength) || pop_strength < 0.25 || pop_strength > 2.0 || depth_reuse_interval != 1 || !source_frame_count || source_frame_count == std::numeric_limits<std::uint64_t>::max() || first_sequence != 1 || (source_format != raw_pixel_format::bgra8 && source_format != raw_pixel_format::gbrpf32le)) {
        throw std::runtime_error("completion header resolved configuration is invalid");
      }
    }
  };

  struct completion_sbs_info_t {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    raw_pixel_format format = raw_pixel_format::bgra8;
  };

  struct completion_frame_t {
    raw_frame_header source;
    sbs_adaptive_state::words_t adaptive_state {};
    bool depth_updated = false;
    std::optional<completion_sbs_info_t> sbs;

    void validate() const {
      if (source.schema != 1 || !source.sequence || !source.width || !source.height || source.width > 16384 || source.height > 16384 || source.time_base_numerator <= 0 || source.time_base_denominator <= 0 || (source.format != raw_pixel_format::bgra8 && source.format != raw_pixel_format::gbrpf32le)) {
        throw std::runtime_error("completion source header is invalid");
      }
      if (!depth_updated) {
        throw std::runtime_error("completion did not update depth for its source");
      }
      for (const auto &field : sbs_adaptive_state::fields) {
        if (field.gpu_encoding != sbs_adaptive_state::gpu_encoding_e::uint_bits && !std::isfinite(std::bit_cast<float>(adaptive_state[sbs_adaptive_state::index(field.word)]))) {
          throw std::runtime_error("completion adaptive state contains a non-finite float32");
        }
      }
      for (const auto word : sbs_adaptive_state::reserved_words) {
        const auto index = sbs_adaptive_state::index(word);
        const auto &field = sbs_adaptive_state::fields[index];
        const bool matches = field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits ?
                               adaptive_state[index] == sbs_adaptive_state::initial_words[index] :
                               std::bit_cast<float>(adaptive_state[index]) == field.initial_value;
        if (!matches) {
          throw std::runtime_error("completion adaptive reserved field is non-default");
        }
      }
      for (const auto word : {sbs_adaptive_state::word_e::empty_raw_count, sbs_adaptive_state::word_e::collapsed_raw_count}) {
        if (adaptive_state[sbs_adaptive_state::index(word)] > sbs_adaptive_state::counter_max) {
          throw std::runtime_error("completion adaptive counter exceeds its bound");
        }
      }
      // The existing GPU readback validator remains the producer's contract-tag authority.
      // Native evidence preserves finite float32 bits, including +/-float::max. It does not
      // emulate the evaluator JSON path's decimal round-trip representation.
      if (sbs && (!sbs->width || !sbs->height || sbs->width > 16384 || sbs->height > 16384 || (sbs->format != raw_pixel_format::bgra8 && sbs->format != raw_pixel_format::rgba16f))) {
        throw std::runtime_error("completion SBS description is invalid");
      }
    }
  };

  struct completion_terminal_t {
    int exit_code = -1;
    std::uint64_t source_frame_count = 0;
    std::uint64_t adaptive_frame_count = 0;
    std::uint64_t sbs_frame_count = 0;
    std::string error;
  };

  // One owned header, one owned completion, and an independent sticky terminal record.
  // Dequeuing evidence never acknowledges source consumption or replaces that barrier.
  class completion_channel_t {
  public:
    static constexpr auto name = "bounded-completion-memory-v1";

    void publish_header(completion_header_t header) {
      header.validate();
      std::lock_guard lock(mutex_);
      throw_if_failed();
      if (terminal_ || header_published_) {
        throw std::runtime_error("completion header was already published or the pass finished");
      }
      declared_source_count_ = header.source_frame_count;
      source_format_ = header.source_format;
      sbs_enabled_ = header.sbs_enabled;
      header_ = std::move(header);
      header_published_ = true;
      changed_.notify_all();
    }

    completion_header_t receive_header(std::chrono::milliseconds timeout) {
      std::unique_lock lock(mutex_);
      if (header_received_) {
        throw std::runtime_error("completion header was already received");
      }
      wait(lock, timeout, [&] {
        return header_.has_value() || terminal_.has_value();
      });
      throw_if_failed();
      if (!header_) {
        throw std::runtime_error("completion pass finished before its header");
      }
      auto header = std::move(*header_);
      header_.reset();
      header_received_ = true;
      return header;
    }

    void publish_frame(completion_frame_t frame) {
      frame.validate();
      std::unique_lock lock(mutex_);
      throw_if_failed();
      validate_publication(frame);
      wait(lock, std::chrono::seconds {120}, [&] {
        return !frame_ || terminal_.has_value();
      });
      throw_if_failed();
      validate_publication(frame);
      frame_ = std::move(frame);
      ++next_publish_;
      changed_.notify_all();
    }

    completion_frame_t receive_frame(std::uint64_t sequence, std::chrono::milliseconds timeout) {
      std::unique_lock lock(mutex_);
      if (!header_received_ || sequence != next_receive_) {
        throw std::runtime_error("completion receive sequence or header order differs");
      }
      wait(lock, timeout, [&] {
        return frame_.has_value() || terminal_.has_value();
      });
      throw_if_failed();
      if (!frame_ || frame_->source.sequence != sequence) {
        throw std::runtime_error("completion receive sequence mismatch or premature EOF");
      }
      auto frame = std::move(*frame_);
      frame_.reset();
      ++next_receive_;
      changed_.notify_all();
      return frame;
    }

    // Never waits for a consumer. A failure must wake waits even if no header was published.
    // Success does not overwrite an outstanding final completion or synthesize its source ACK.
    void finish(completion_terminal_t terminal) {
      if (terminal.error.size() > 1024) {
        terminal.error.resize(1024);
      }
      std::lock_guard lock(mutex_);
      if (cancelled_ || terminal_) {
        return;
      }
      if (terminal.exit_code == 0 && (!header_published_ || terminal.source_frame_count != declared_source_count_ || terminal.adaptive_frame_count != declared_source_count_ || terminal.sbs_frame_count != (sbs_enabled_ ? declared_source_count_ : 0) || next_publish_ - 1 != declared_source_count_ || !terminal.error.empty())) {
        throw std::runtime_error("completion terminal success disagrees with published evidence");
      }
      const bool failed = terminal.exit_code != 0;
      terminal_ = std::move(terminal);
      if (failed) {
        header_.reset();
        frame_.reset();
      }
      changed_.notify_all();
    }

    completion_terminal_t wait_finished(std::chrono::milliseconds timeout) {
      std::unique_lock lock(mutex_);
      wait(lock, timeout, [&] {
        return terminal_.has_value();
      });
      // Unlike frame/header receive, observing a terminal failure returns its actual status.
      return *terminal_;
    }

    void cancel() noexcept {
      std::lock_guard lock(mutex_);
      cancelled_ = true;
      header_.reset();
      frame_.reset();
      changed_.notify_all();
    }

  private:
    void throw_if_failed() const {
      if (cancelled_) {
        throw std::runtime_error("bounded completion handoff cancelled");
      }
      if (terminal_ && terminal_->exit_code != 0) {
        throw std::runtime_error("native pass failed (exit " + std::to_string(terminal_->exit_code) + "): " + terminal_->error);
      }
    }

    void validate_publication(const completion_frame_t &frame) const {
      if (!header_published_ || terminal_ || frame.source.sequence != next_publish_ || frame.source.sequence > declared_source_count_ || frame.source.format != source_format_ || frame.sbs.has_value() != sbs_enabled_) {
        throw std::runtime_error("completion publication sequence or run metadata differs");
      }
      if (frame.sbs && frame.sbs->format != (source_format_ == raw_pixel_format::gbrpf32le ? raw_pixel_format::rgba16f : raw_pixel_format::bgra8)) {
        throw std::runtime_error("completion packed format differs from the source transfer");
      }
    }

    template<class Predicate>
    void wait(std::unique_lock<std::mutex> &lock, std::chrono::milliseconds timeout, Predicate ready) {
      if (timeout.count() < 0) {
        throw std::invalid_argument("negative completion receive deadline");
      }
      if (!changed_.wait_for(lock, timeout, [&] {
            return cancelled_ || ready();
          })) {
        throw std::runtime_error("bounded completion handoff timed out");
      }
      if (cancelled_) {
        throw std::runtime_error("bounded completion handoff cancelled");
      }
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    std::optional<completion_header_t> header_;
    std::optional<completion_frame_t> frame_;
    std::optional<completion_terminal_t> terminal_;
    std::uint64_t next_publish_ = 1;
    std::uint64_t next_receive_ = 1;
    std::uint64_t declared_source_count_ = 0;
    raw_pixel_format source_format_ = raw_pixel_format::bgra8;
    bool header_published_ = false;
    bool header_received_ = false;
    bool sbs_enabled_ = false;
    bool cancelled_ = false;
  };

  struct raw_frame_transport {
    static constexpr auto name = "bounded-raw-memory-v1";

    explicit raw_frame_transport(std::uint64_t limit):
        byte_limit(limit),
        source(2, limit),
        sbs(1, limit) {}

    void cancel() noexcept {
      source.cancel();
      sbs.cancel();
      completion.cancel();
    }

    const std::uint64_t byte_limit;
    raw_frame_channel source;
    raw_frame_channel sbs;
    completion_channel_t completion;
  };
}  // namespace offline_sbs

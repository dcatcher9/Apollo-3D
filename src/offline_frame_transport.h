/** Versioned, bounded raw-frame handoff within the isolated offline worker. */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
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

  struct raw_frame_transport {
    static constexpr auto name = "bounded-raw-memory-v1";
    explicit raw_frame_transport(std::uint64_t limit): byte_limit(limit), source(2, limit), sbs(1, limit) {}
    void cancel() noexcept { source.cancel(); sbs.cancel(); }
    const std::uint64_t byte_limit;
    raw_frame_channel source;
    raw_frame_channel sbs;
  };
}

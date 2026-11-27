/**
 * @file src/stream_media.cpp
 * @brief Definitions for the video FEC packetizer and the encrypted UDP media senders.
 */

// standard includes
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

// lib includes
#include <boost/align/aligned_allocator.hpp>
#include <boost/endian/arithmetic.hpp>
#include <rs.h>

extern "C" {
  // clang-format off
#include <moonlight-common-c/src/Limelight-internal.h>
  // clang-format on
}

// local includes
#include "config.h"
#include "crypto.h"
#include "logging.h"
#include "platform/common.h"
#include "stream.h"
#include "stream_internal.h"
#ifdef SUNSHINE_TESTS
  #include "stream_media_test.h"
#endif
#include "utility.h"
#include "video.h"

namespace asio = boost::asio;
namespace sys = boost::system;

using asio::ip::udp;

using namespace std::literals;

namespace stream {
#pragma pack(push, 1)

  struct video_short_frame_header_t {
    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    std::uint8_t headerType;  // Always 0x01 for short headers

    // Sunshine extension
    // Frame processing latency, in 1/10 ms units
    //     zero when the frame is repeated or there is no backend implementation
    boost::endian::little_uint16_at frame_processing_latency;

    // Currently known values:
    // 1 = Normal P-frame
    // 2 = IDR-frame
    // 4 = P-frame with intra-refresh blocks
    // 5 = P-frame after reference frame invalidation
    std::uint8_t frameType;

    // Length of the final packet payload for codecs that cannot handle
    // zero padding, such as AV1 (Sunshine extension).
    boost::endian::little_uint16_at lastPayloadLen;

    // SOURCE_FRAME_ID_V1: nonzero exact encoder-input token, only after negotiation.
    boost::endian::little_uint16_at source_frame_id;
  };

  static_assert(
    sizeof(video_short_frame_header_t) == 8,
    "Short frame header must be 8 bytes"
  );
  static_assert(offsetof(video_short_frame_header_t, source_frame_id) == 6);

  struct video_packet_raw_t {
    uint8_t *payload() {
      return (uint8_t *) (this + 1);
    }

    RTP_PACKET rtp;
    char reserved[4];

    NV_VIDEO_PACKET packet;
  };

  static_assert(sizeof(video_packet_raw_t) == MAX_RTP_HEADER_SIZE + sizeof(NV_VIDEO_PACKET));

  struct video_packet_enc_prefix_t {
    std::uint8_t iv[12];  // 12-byte IV is ideal for AES-GCM
    std::uint32_t frameNumber;
    std::uint8_t tag[16];
  };

  struct audio_packet_t {
    RTP_PACKET rtp;
  };

#pragma pack(pop)

  constexpr std::size_t FEC_PACKET_INDEX_MAX = 1023;
  constexpr std::size_t VIDEO_RS_CONTEXT_CACHE_LIMIT = 8;
  constexpr std::size_t VIDEO_SEND_WORKSPACE_RETAIN_LIMIT = 16 * 1024 * 1024;

  // Keep the data and parity shard bases aligned for the widest Reed–Solomon SIMD backend.
  // Shards retain their negotiated wire stride, so alignment of individual shards depends on it.
  constexpr std::size_t VIDEO_SHARD_ALIGNMENT = 64;
  using video_shard_storage_t = std::vector<
    std::uint8_t,
    boost::alignment::aligned_allocator<std::uint8_t, VIDEO_SHARD_ALIGNMENT>>;

  namespace fec {
    // The video packet header has two bits for the number of blocks minus one.
    constexpr std::size_t MAX_BLOCKS = 4;

    struct block_plan_t {
      std::size_t data_shards;
      std::size_t parity_shards;
      std::size_t percentage;
    };

    struct frame_plan_t {
      std::size_t data_shards;
      std::size_t wire_shards;
      std::size_t block_count;
      std::size_t final_payload_bytes;
      std::array<block_plan_t, MAX_BLOCKS> blocks;
    };

    static std::optional<frame_plan_t> make_frame_plan(
      std::size_t picture_size,
      std::size_t payload_blocksize,
      int fec_percentage,
      std::size_t min_required_fec_packets
    ) {
      const auto tail_size = picture_size % payload_blocksize;
      const auto data_shards = picture_size / payload_blocksize + (tail_size != 0);
      // D data shards at rate F need D + ceil(D * F / 100) slots.
      const auto max_data_per_block = (RS8_DATA_SHARDS_MAX * 100) / (100 + fec_percentage);
      auto block_count = static_cast<std::size_t>(ceil_div_u64(data_shards, max_data_per_block));
      if (block_count > MAX_BLOCKS) {
        BOOST_LOG(warning) << "Skipping FEC for abnormally large encoded frame (needed "sv << block_count << " FEC blocks)"sv;
        fec_percentage = 0;
        block_count = MAX_BLOCKS;
      }

      // Split on shard boundaries. Only the final shard can contain padding.
      const std::size_t data_per_block = ceil_div_u64(data_shards, block_count);
      if (data_per_block > FEC_PACKET_INDEX_MAX) {
        BOOST_LOG(error) << "Dropping encoded frame that exceeds the 10-bit FEC packet index (needed "sv << data_per_block << " packets per block)"sv;
        return std::nullopt;
      }

      frame_plan_t plan {data_shards, data_shards, block_count, tail_size == 0 ? payload_blocksize : tail_size, {}};
      std::size_t remaining = data_shards;
      for (std::size_t index = 0; index < block_count; ++index) {
        const auto count = std::min(data_per_block, remaining);
        auto parity = (count * fec_percentage + 99) / 100;
        auto percentage = static_cast<std::size_t>(fec_percentage);
        if (percentage != 0 && parity < min_required_fec_packets) {
          parity = min_required_fec_packets;
          percentage = (100 * parity) / count;
        }
        plan.blocks[index] = {count, parity, percentage};
        plan.wire_shards += parity;
        remaining -= count;
      }
      return plan;
    }

    using rs_t = util::safe_ptr<reed_solomon, [](reed_solomon *rs) {
      reed_solomon_release(rs);
    }>;

    class rs_cache_t {
    public:
      reed_solomon *acquire(std::size_t data_shards, std::size_t parity_shards) {
        ++use_generation_;
        for (auto &entry : entries_) {
          if (entry.context && entry.data_shards == data_shards && entry.parity_shards == parity_shards) {
            entry.last_use = use_generation_;
            return entry.context.get();
          }
        }

        auto *replacement = &entries_.front();
        for (auto &entry : entries_) {
          if (!entry.context) {
            replacement = &entry;
            break;
          }
          if (entry.last_use < replacement->last_use) {
            replacement = &entry;
          }
        }

        rs_t context {reed_solomon_new(data_shards, parity_shards)};
        if (!context) {
          return nullptr;
        }

        replacement->data_shards = data_shards;
        replacement->parity_shards = parity_shards;
        replacement->last_use = use_generation_;
        replacement->context = std::move(context);
        return replacement->context.get();
      }

      void discard(std::size_t data_shards, std::size_t parity_shards) {
        for (auto &entry : entries_) {
          if (entry.context && entry.data_shards == data_shards && entry.parity_shards == parity_shards) {
            entry.context.reset();
            return;
          }
        }
      }

    private:
      struct entry_t {
        std::size_t data_shards = 0;
        std::size_t parity_shards = 0;
        std::uint64_t last_use = 0;
        rs_t context;
      };

      std::array<entry_t, VIDEO_RS_CONTEXT_CACHE_LIMIT> entries_;
      std::uint64_t use_generation_ = 0;
    };

    struct workspace_t {
      video_shard_storage_t parity;
      std::array<std::uint8_t *, RS8_DATA_SHARDS_MAX> rs_pointers;

      void release_oversized() {
        if (parity.capacity() > VIDEO_SEND_WORKSPACE_RETAIN_LIMIT) {
          video_shard_storage_t().swap(parity);
        }
      }
    };

    struct encoded_block_t {
      platf::udp_payload_block_t payloads;
      std::size_t percentage;
    };

    static encoded_block_t encode(
      std::uint8_t *payload,
      std::size_t blocksize,
      const block_plan_t &plan,
      rs_cache_t &rs_cache,
      workspace_t &workspace
    ) {
      platf::udp_payload_block_t payloads {
        reinterpret_cast<char *>(payload),
        nullptr,
        plan.data_shards,
        0,
        blocksize,
      };

      if (plan.parity_shards == 0) {
        return {payloads, 0};
      }

      // The context validates the total Reed-Solomon shard count before using the fixed table.
      // Data-only blocks can exceed that count and never use the table.
      auto *rs = rs_cache.acquire(plan.data_shards, plan.parity_shards);
      if (!rs) {
        BOOST_LOG(error) << "Failed to allocate video Reed-Solomon context; sending this FEC block without parity"sv;
        return {payloads, 0};
      }

      const auto parity_bytes = plan.parity_shards * blocksize;
      if (workspace.parity.size() < parity_bytes) {
        workspace.parity.resize(parity_bytes);
      }
      payloads.parity = reinterpret_cast<char *>(workspace.parity.data());
      payloads.parity_count = plan.parity_shards;
      for (std::size_t index = 0; index < payloads.size(); ++index) {
        workspace.rs_pointers[index] = reinterpret_cast<std::uint8_t *>(payloads.payload_at(index));
      }

      if (reed_solomon_encode(rs, workspace.rs_pointers.data(), payloads.size(), blocksize) != 0) {
        rs_cache.discard(plan.data_shards, plan.parity_shards);
        BOOST_LOG(error) << "Failed to encode video Reed-Solomon shards; sending this FEC block without parity"sv;
        payloads.parity_count = 0;
        return {payloads, 0};
      }

      return {payloads, plan.percentage};
    }
  }  // namespace fec

  // Pack one block in wire order. FEC and the synchronous sends share these same data shards.
  static std::uint8_t *packetize_video_block(
    video_shard_storage_t &storage,
    std::size_t blocksize,
    std::size_t data_shards,
    std::string_view &payload,
    video_packet_raw_t header,
    std::uint32_t block_lowseq,
    const video_short_frame_header_t *frame_header
  ) {
    const auto size = data_shards * blocksize;
    if (storage.size() < size) {
      storage.resize(size);
    }

    auto *const first_shard = storage.data();
    auto *shard = first_shard;
    auto prefix_bytes = sizeof(header);
    if (frame_header) {
      std::memcpy(shard + prefix_bytes, frame_header, sizeof(*frame_header));
      prefix_bytes += sizeof(*frame_header);
    }
    header.packet.flags = FLAG_CONTAINS_PIC_DATA;
    for (std::size_t index = 0; index < data_shards; ++index, shard += blocksize) {
      std::memcpy(shard, &header, sizeof(header));
      reinterpret_cast<video_packet_raw_t *>(shard)->packet.streamPacketIndex = block_lowseq++ << 8;
      auto *picture = shard + prefix_bytes;
      const auto capacity = blocksize - prefix_bytes;
      const auto bytes = std::min(capacity, payload.size());
      if (bytes != 0) {
        std::memcpy(picture, payload.data(), bytes);
        payload.remove_prefix(bytes);
      }
      if (bytes < capacity) {
        std::memset(picture + bytes, 0, capacity - bytes);
      }
      prefix_bytes = sizeof(header);
    }
    reinterpret_cast<video_packet_raw_t *>(first_shard)->packet.flags |= FLAG_SOF;
    reinterpret_cast<video_packet_raw_t *>(shard - blocksize)->packet.flags |= FLAG_EOF;
    return first_shard;
  }

#ifdef SUNSHINE_TESTS
  struct media_sender_test_hooks_t {
    std::function<void()> ready;
    std::function<void()> before_admission;
    std::function<void()> processed;
    std::optional<int> video_fec_percentage;
    std::optional<std::uint64_t> video_fail_encryption_at_iv;
  };
#endif

  void run_video_sender(
    udp::socket &sock,
    video::packet_queue_t packets,
    std::shared_ptr<video_channel_t> channel,
    safe::mail_raw_t::event_t<bool> shutdown_event,
    std::chrono::steady_clock::time_point video_epoch
#ifdef SUNSHINE_TESTS
    ,
    media_sender_test_hooks_t *test_hooks
#endif
  ) {
    // Video traffic is sent on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    logging::min_max_avg_periodic_logger<double> frame_processing_latency_logger(info, "Frame processing latency", "ms");

    logging::time_delta_periodic_logger frame_send_batch_latency_logger(info, "Network: each send_batch() latency");
    logging::time_delta_periodic_logger frame_fec_latency_logger(info, "Network: each FEC block latency");
    logging::time_delta_periodic_logger frame_network_latency_logger(info, "Network: frame's overall network latency");

    struct video_stage_diagnostics_t {
      logging::min_max_avg_periodic_logger<double> encoded_queue {
        info,
        "Video packet: encoded queue residence",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> new_content_age {
        info,
        "Video output: new content age before packetization",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> repeated_content_age {
        info,
        "Video output: repeated content age before packetization",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> packetization {
        info,
        "Video packet: packetization and pacing-plan CPU",
        "ms",
      };
    };

    // Logger construction itself reads the clock and owns strings. Keep even that work outside
    // the disabled path, along with all new samples and packet classification.
    auto stage_diagnostics = video::detail::make_diagnostic_state<video_stage_diagnostics_t>(
      config::sunshine.diagnostics_enabled
    );

    crypto::aes_t iv(12);
    iv[11] = 'V';  // Video stream
    std::array<video_packet_enc_prefix_t, platf::MAX_UDP_BATCH_PACKETS> prefixes {};
    video_shard_storage_t data_shard_storage;
    fec::rs_cache_t video_rs_cache;
    fec::workspace_t video_fec_workspace;

    auto timer = platf::create_high_precision_timer();
    if (!timer || !*timer) {
      BOOST_LOG(error) << "Failed to create timer, aborting video broadcast thread";
      return;
    }

    auto ratecontrol_next_frame_start = std::chrono::steady_clock::now();

#ifdef SUNSHINE_TESTS
    if (test_hooks && test_hooks->ready) {
      test_hooks->ready();
    }
#endif
    while (auto packet = packets->pop()) {
#ifdef SUNSHINE_TESTS
      auto report_processed = util::fail_guard([test_hooks] {
        if (test_hooks && test_hooks->processed) {
          test_hooks->processed();
        }
      });
      if (test_hooks && test_hooks->before_admission) {
        test_hooks->before_admission();
      }
#endif
      auto release_oversized_workspaces = util::fail_guard([&]() {
        if (data_shard_storage.capacity() > VIDEO_SEND_WORKSPACE_RETAIN_LIMIT) {
          video_shard_storage_t().swap(data_shard_storage);
        }
        video_fec_workspace.release_oversized();
      });
      if (shutdown_event->peek()) {
        break;
      }

      frame_network_latency_logger.first_point_now();

      if (!channel || !channel->active.load(std::memory_order_acquire)) {
        continue;
      }
      const auto effective_pacing = channel->effective_mode ?
                                      channel->effective_mode->pacing() :
                                      video::effective_video_pacing_t {0, 0};

      auto request_recovery_idr = [&](std::string_view reason) {
        if (!channel->awaiting_recovery_idr) {
          BOOST_LOG(warning) << "Dropping encoded video until a recovery IDR ("sv << reason << ")."sv;
        }
        channel->awaiting_recovery_idr = true;
        channel->idr_events->try_raise(true);
        channel->wake_encoder();
      };

      const auto packet_dequeued_at = std::chrono::steady_clock::now();
      const auto encoded_packet_age = packet_dequeued_at - packet->encoded_timestamp;
      if (stage_diagnostics) {
        if (const auto elapsed = video::detail::diagnostic_elapsed_ms(packet->encoded_timestamp, packet_dequeued_at)) {
          stage_diagnostics->encoded_queue.collect_and_log(*elapsed);
        }
      }
      const auto max_encoded_packet_age = std::chrono::nanoseconds {
        video_packet_max_queue_age_ns(effective_pacing.framerate_millihz)
      };
      if (encoded_packet_age > max_encoded_packet_age) {
        request_recovery_idr("encoded packet exceeded the host backlog age limit"sv);
        continue;
      }
      if (channel->awaiting_recovery_idr) {
        if (!packet->is_idr()) {
          continue;
        }
        channel->awaiting_recovery_idr = false;
      }

      std::string_view payload {reinterpret_cast<const char *>(packet->data()), packet->data_size()};

      video_short_frame_header_t frame_header = {};
      frame_header.headerType = 0x01;  // Short header type
      frame_header.frameType = packet->is_idr()                     ? 2 :
                               packet->after_ref_frame_invalidation ? 5 :
                                                                      1;
      frame_header.source_frame_id = channel->source_frame_id.next_frame(
        channel->source_frame_id_negotiated,
        static_cast<std::uint32_t>(packet->frame_index()),
        packet->encoder_input_retained,
        packet->is_idr(),
        packet->after_ref_frame_invalidation
      );
      if (packet->processing_started || stage_diagnostics) {
        const auto measured_at = std::chrono::steady_clock::now();
        const auto latency = video::detail::frame_processing_latency_tenths_ms(
          packet->processing_started,
          measured_at
        );
        frame_header.frame_processing_latency = latency;
        if (latency != 0) {
          frame_processing_latency_logger.collect_and_log(latency / 10.);
        }
        if (stage_diagnostics && packet->diagnostic_content != video::detail::diagnostic_content_e::unknown) {
          const auto content_timestamp = video::detail::select_content_timestamp(
            packet->content_timestamp,
            packet->frame_timestamp
          );
          if (const auto elapsed = video::detail::diagnostic_elapsed_ms(content_timestamp, measured_at)) {
            auto &logger = packet->diagnostic_content == video::detail::diagnostic_content_e::new_content ?
                             stage_diagnostics->new_content_age :
                             stage_diagnostics->repeated_content_age;
            logger.collect_and_log(*elapsed);
            if (packet->sbs_telemetry_performance && packet->diagnostic_content == video::detail::diagnostic_content_e::new_content) {
              packet->sbs_telemetry_performance->record(host_sbs_telemetry::stage::new_content_age, *elapsed, measured_at);
            }
          }
        }
      } else {
        frame_header.frame_processing_latency = 0;
      }

      auto fec_percentage = config::stream.fec_percentage;
#ifdef SUNSHINE_TESTS
      if (test_hooks && test_hooks->video_fec_percentage) {
        fec_percentage = *test_hooks->video_fec_percentage;
      }
#endif

      const auto packetization_started = video::detail::diagnostic_timestamp(
        stage_diagnostics.has_value(),
        video::detail::diagnostic_clock_t::now
      );
      if (!is_valid_video_transport_config(channel->packet_size, channel->min_required_fec_packets) || fec_percentage < 0 || fec_percentage > 255 || payload.size() > std::numeric_limits<std::size_t>::max() - sizeof(frame_header)) {
        request_recovery_idr("invalid video packetization parameters"sv);
        continue;
      }

      const std::size_t blocksize = channel->packet_size + MAX_RTP_HEADER_SIZE;
      const auto payload_blocksize = blocksize - sizeof(video_packet_raw_t);
      const auto picture_size = payload.size() + sizeof(frame_header);
      const auto frame_plan = fec::make_frame_plan(
        picture_size,
        payload_blocksize,
        fec_percentage,
        channel->min_required_fec_packets
      );
      if (!frame_plan) {
        request_recovery_idr("encoded frame exceeded the FEC packet-index limit"sv);
        continue;
      }

      frame_header.lastPayloadLen = frame_plan->final_payload_bytes;
      const auto fec_blocks_needed = frame_plan->block_count;

      try {
        const auto wire_packet_bytes = blocksize + sizeof(video_packet_enc_prefix_t);
        const auto pacing_plan = make_video_pacing_plan(
          effective_pacing.bitrate,
          effective_pacing.framerate_millihz,
          frame_plan->data_shards,
          frame_plan->wire_shards,
          payload_blocksize,
          wire_packet_bytes
        );

        // Send less than 64K in a single batch.
        // On Windows, batches above 64K seem to bypass SO_SNDBUF regardless of its size,
        // appear in "Other I/O" and begin waiting for interrupts.
        // This gives inconsistent performance so we'd rather avoid it. Also cap each batch to
        // roughly one pacing quantum so a nominally paced frame is not emitted as a few 64K bursts.
        size_t send_batch_size = std::max<std::size_t>(1, 64 * 1024 / wire_packet_bytes);
        // Also don't exceed 64 packets, which can happen when Artemis requests
        // unusually small packet size.
        // Generic Segmentation Offload on Linux can't do more than 64.
        send_batch_size = std::min({
          platf::MAX_UDP_BATCH_PACKETS,
          send_batch_size,
          pacing_plan.packets_per_quantum,
        });
        auto packetization_time = video::detail::diagnostic_clock_t::duration::zero();
        if (packetization_started) {
          packetization_time = video::detail::diagnostic_clock_t::now() - *packetization_started;
        }

        const auto ceiling_label =
          pacing_plan.target_wire_bps == VIDEO_PACING_MAX_WIRE_BPS ? ", ceiling-limited"sv : ""sv;
        if (packet->is_idr()) {
          if (!channel->idr_pacing_plan_logged.exchange(true, std::memory_order_acq_rel)) {
            BOOST_LOG(debug) << "Video IDR packet pacing (transient): target="sv
                             << (pacing_plan.target_wire_bps / 1'000'000.0)
                             << " Mbps"sv << ceiling_label
                             << "; steady-state pacing will be logged on the first inter frame."sv;
          }
        } else if (!channel->pacing_plan_logged.exchange(true, std::memory_order_acq_rel)) {
          BOOST_LOG(info) << "Video packet pacing (steady state): target="sv
                          << (pacing_plan.target_wire_bps / 1'000'000.0)
                          << " Mbps"sv << ceiling_label
                          << " (negotiated bitrate + actual packet overhead/cadence), batch<="sv
                          << send_batch_size
                          << " packets, frame-span<="sv
                          << (pacing_plan.max_frame_span_ns / 1'000'000.0) << " ms."sv;
        }

        // A duplicate without a capture timestamp uses the same pacing timestamp for every block.
        using rtp_tick = std::chrono::duration<std::uint32_t, std::ratio<1, 90000>>;
        const auto capture_time = packet->frame_timestamp.value_or(ratecontrol_next_frame_start);
        const auto timestamp = std::chrono::round<rtp_tick>(capture_time - video_epoch).count();

        // Leave RTP and fecInfo zero until the parity shards have been generated.
        video_packet_raw_t block_header {};
        block_header.packet.frameIndex = packet->frame_index();
        block_header.packet.multiFecFlags = 0x10;
        auto ratecontrol_frame_start = ratecontrol_next_frame_start;
        auto peer_address = channel->peer.address();

        size_t ratecontrol_frame_packets_sent = 0;
        bool frame_encrypted = true;

        for (std::size_t block_index = 0; block_index < fec_blocks_needed && frame_encrypted; ++block_index) {
          const auto &block = frame_plan->blocks[block_index];
          const auto block_lowseq = channel->lowseq;
          const auto block_packing_started = video::detail::diagnostic_timestamp(
            stage_diagnostics.has_value(),
            video::detail::diagnostic_clock_t::now
          );
          block_header.packet.multiFecBlocks = (block_index << 4) | ((fec_blocks_needed - 1) << 6);
          auto *block_payload = packetize_video_block(
            data_shard_storage,
            blocksize,
            block.data_shards,
            payload,
            block_header,
            block_lowseq,
            block_index == 0 ? &frame_header : nullptr
          );
          if (block_packing_started) {
            packetization_time += video::detail::diagnostic_clock_t::now() - *block_packing_started;
          }
          if (block_index == 0) {
            // Start pacing when the first block is packed and the previous frame's slot has ended.
            ratecontrol_frame_start = std::max(ratecontrol_frame_start, std::chrono::steady_clock::now());
          }

          frame_fec_latency_logger.first_point_now();
          const auto [payloads, percentage] = fec::encode(block_payload, blocksize, block, video_rs_cache, video_fec_workspace);
          frame_fec_latency_logger.second_point_now_and_log();

          // Reserve this block's sequence range before the first send. A recovery IDR starts
          // after that range even when a failed send may have delivered part of the block.
          channel->lowseq += static_cast<std::uint32_t>(payloads.size());

          for (std::size_t first_packet = 0; first_packet < payloads.size(); first_packet += send_batch_size) {
            const auto count = std::min(send_batch_size, payloads.size() - first_packet);

            // Complete encryption of the whole batch before handing any of it to the socket.
            for (std::size_t index = 0; index < count; ++index) {
              const auto packet_index = first_packet + index;
              auto *payload = payloads.payload_at(packet_index);
              auto *inspect = reinterpret_cast<video_packet_raw_t *>(payload);

              inspect->packet.fecInfo =
                (packet_index << 12 |
                 payloads.data_count << 22 |
                 percentage << 4);
              inspect->rtp.header = 0x80 | FLAG_EXTENSION;
              inspect->rtp.sequenceNumber = util::endian::big<uint16_t>(
                static_cast<std::uint16_t>(block_lowseq + static_cast<std::uint32_t>(packet_index))
              );
              inspect->rtp.timestamp = util::endian::big<uint32_t>(timestamp);
              inspect->packet.multiFecBlocks = block_header.packet.multiFecBlocks;
              inspect->packet.frameIndex = block_header.packet.frameIndex;

              // NIST SP 800-38D, Section 8.2.1. The counter identifies each use of the client's unique key.
              const auto counter = channel->gcm_iv_counter++;
              std::memcpy(iv.data(), &counter, sizeof(counter));

              auto &prefix = prefixes[index];
              prefix.frameNumber = block_header.packet.frameIndex;
              std::copy(std::begin(iv), std::end(iv), prefix.iv);
#ifdef SUNSHINE_TESTS
              if (test_hooks && test_hooks->video_fail_encryption_at_iv == counter && channel->cipher.encrypt_ctx) {
                EVP_CIPHER_CTX_reset(channel->cipher.encrypt_ctx.get());
              }
#endif
              if (!channel->cipher.encrypt(std::string_view {payload, blocksize}, prefix.tag, reinterpret_cast<std::uint8_t *>(payload), &iv)) {
                frame_encrypted = false;
                break;
              }
            }
            if (!frame_encrypted) {
              request_recovery_idr("video datagram encryption failed"sv);
              break;
            }

            // Pace every bounded batch. The first batch is due immediately. Later batches
            // are distributed according to the negotiated bitrate, FEC, and cadence.
            const auto due = ratecontrol_frame_start +
                             video_pacing_offset(ratecontrol_frame_packets_sent, pacing_plan.packets_per_second);
            auto now = std::chrono::steady_clock::now();
            const auto schedule_rebase = std::chrono::nanoseconds {
              video_pacing_rebase_ns(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  due - ratecontrol_frame_start
                )
                  .count(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now - ratecontrol_frame_start
                )
                  .count()
              )
            };
            ratecontrol_frame_start += schedule_rebase;
            const auto bounded_due = due + schedule_rebase;
            if (now < bounded_due) {
              timer->sleep_for(bounded_due - now);
            }

            const platf::batched_send_info_t batch {
              payloads,
              reinterpret_cast<const char *>(prefixes.data()),
              sizeof(video_packet_enc_prefix_t),
              first_packet,
              count,
              (uintptr_t) sock.native_handle(),
              peer_address,
              channel->peer.port(),
              channel->local_address,
            };
            frame_send_batch_latency_logger.first_point_now();
            if (!platf::send_batch(batch)) {
              for (std::size_t index = 0; index < batch.packet_count; ++index) {
                auto send_info = platf::send_info_t {
                  batch.header_at(index),
                  batch.header_size,
                  batch.payload_at(index),
                  batch.payloads.payload_size,
                  batch.native_socket,
                  peer_address,
                  batch.target_port,
                  channel->local_address,
                };
                platf::send(send_info);
              }
            }
            frame_send_batch_latency_logger.second_point_now_and_log();

            ratecontrol_frame_packets_sent += count;
          }
        }
        if (stage_diagnostics) {
          stage_diagnostics->packetization.collect_and_log(std::chrono::duration<double, std::milli>(packetization_time).count());
        }

        // Remember the final scheduled packet time in case the next frame is already queued.
        ratecontrol_next_frame_start = ratecontrol_frame_start +
                                       video_pacing_offset(
                                         ratecontrol_frame_packets_sent,
                                         pacing_plan.packets_per_second
                                       );

        if (frame_encrypted) {
          frame_network_latency_logger.second_point_now_and_log();
        }
      } catch (const std::exception &e) {
        BOOST_LOG(error) << "Broadcast video failed "sv << e.what();
        request_recovery_idr("video packetization or send failed"sv);
        std::this_thread::sleep_for(100ms);
      }
    }
  }

  void run_audio_sender(
    udp::socket &sock,
    audio::packet_queue_t packets,
    std::shared_ptr<audio_channel_t> channel,
    safe::mail_raw_t::event_t<bool> shutdown_event
#ifdef SUNSHINE_TESTS
    ,
    media_sender_test_hooks_t *test_hooks
#endif
  ) {
    audio_packet_t audio_packet;
    fec::rs_t rs {reed_solomon_new(RTPA_DATA_SHARDS, RTPA_FEC_SHARDS)};
    crypto::aes_t iv(16);

    if (!rs) {
      BOOST_LOG(error) << "Failed to allocate audio Reed-Solomon context"sv;
      return;
    }

    // For unknown reasons, the RS parity matrix computed by our RS implementation
    // doesn't match the one Nvidia uses for audio data. I'm not exactly sure why,
    // but we can simply replace it with the matrix generated by OpenFEC which
    // works correctly. This is possible because the data and FEC shard count is
    // constant and known in advance.
    const unsigned char parity[] = {0x77, 0x40, 0x38, 0x0e, 0xc7, 0xa7, 0x0d, 0x6c};
    memcpy(rs.get()->p, parity, sizeof(parity));

    audio_packet.rtp.header = 0x80;
    audio_packet.rtp.packetType = 97;
    audio_packet.rtp.ssrc = 0;

    // Audio traffic is sent on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

#ifdef SUNSHINE_TESTS
    if (test_hooks && test_hooks->ready) {
      test_hooks->ready();
    }
#endif
    while (auto packet = packets->pop()) {
#ifdef SUNSHINE_TESTS
      auto report_processed = util::fail_guard([test_hooks] {
        if (test_hooks && test_hooks->processed) {
          test_hooks->processed();
        }
      });
      if (test_hooks && test_hooks->before_admission) {
        test_hooks->before_admission();
      }
#endif
      if (shutdown_event->peek()) {
        break;
      }

      auto &packet_data = *packet;
      if (!channel || !channel->active.load(std::memory_order_acquire)) {
        continue;
      }

      auto sequenceNumber = channel->sequence_number;
      auto timestamp = channel->timestamp;

      const auto packet_iv = util::endian::big<std::uint32_t>(channel->av_ri_key_id + sequenceNumber);
      std::memcpy(iv.data(), &packet_iv, sizeof(packet_iv));

      auto &shards_p = channel->shard_ptrs;

      const auto bytes = channel->cipher.encrypt(
        std::string_view {
          reinterpret_cast<const char *>(std::begin(packet_data)),
          packet_data.size()
        },
        shards_p[sequenceNumber % RTPA_DATA_SHARDS],
        &iv
      );
      if (bytes < 0) {
        BOOST_LOG(error) << "Couldn't encode audio packet"sv;
        break;
      }

      audio_packet.rtp.sequenceNumber = util::endian::big(sequenceNumber);
      audio_packet.rtp.timestamp = util::endian::big(timestamp);

      channel->sequence_number++;
      channel->timestamp += channel->packet_duration;

      auto peer_address = channel->peer.address();
      try {
        auto send_info = platf::send_info_t {
          (const char *) &audio_packet,
          sizeof(audio_packet),
          (const char *) shards_p[sequenceNumber % RTPA_DATA_SHARDS],
          (size_t) bytes,
          (uintptr_t) sock.native_handle(),
          peer_address,
          channel->peer.port(),
          channel->local_address,
        };
        platf::send(send_info);

        auto &fec_packet = channel->fec_packet;
        // initialize the FEC header at the beginning of the FEC block
        if (sequenceNumber % RTPA_DATA_SHARDS == 0) {
          fec_packet.fecHeader.baseSequenceNumber = util::endian::big(sequenceNumber);
          fec_packet.fecHeader.baseTimestamp = util::endian::big(timestamp);
        }

        // generate parity shards at the end of the FEC block
        if ((sequenceNumber + 1) % RTPA_DATA_SHARDS == 0) {
          if (reed_solomon_encode(rs.get(), shards_p.begin(), RTPA_TOTAL_SHARDS, bytes) != 0) {
            BOOST_LOG(error) << "Failed to encode audio Reed-Solomon shards"sv;
            break;
          }

          for (auto x = 0; x < RTPA_FEC_SHARDS; ++x) {
            fec_packet.rtp.sequenceNumber = util::endian::big<std::uint16_t>(sequenceNumber + x + 1);
            fec_packet.fecHeader.fecShardIndex = x;

            auto send_info = platf::send_info_t {
              (const char *) &fec_packet,
              sizeof(fec_packet),
              (const char *) shards_p[RTPA_DATA_SHARDS + x],
              (size_t) bytes,
              (uintptr_t) sock.native_handle(),
              peer_address,
              channel->peer.port(),
              channel->local_address,
            };
            platf::send(send_info);
          }
        }
      } catch (const std::exception &e) {
        BOOST_LOG(error) << "Broadcast audio failed "sv << e.what();
        std::this_thread::sleep_for(100ms);
      }
    }
  }

#ifdef SUNSHINE_TESTS
  #include "stream_media_test.inc"
#endif
}  // namespace stream

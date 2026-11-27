/**
 * @file src/offline_sbs_worker_ffmpeg.cpp
 * @brief FFmpeg children: the raw-pipe source decoder, the persistent NVENC encoder with its
 * exact-timeline filter graph, and the stream-preserving final mux.
 */

#include "offline_sbs_worker_ffmpeg.h"

#include "sbs_perf.h"

#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;
  using namespace worker_detail;

  namespace {
    class transport_timer_t {
    public:
      explicit transport_timer_t(const char *stage): stage_(stage), start_(std::chrono::steady_clock::now()) {}
      ~transport_timer_t() {
        sbs_perf::add_sample_ms(stage_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count());
      }
    private:
      const char *stage_;
      std::chrono::steady_clock::time_point start_;
    };
    constexpr std::size_t offline_encoder_pipeline_capacity = 1;

    std::string zscale_range(const std::string &range) {
      const auto value = lower(range);
      if (value == "tv" || value == "mpeg" || value == "limited") {
        return "limited";
      }
      if (value == "pc" || value == "jpeg" || value == "full") {
        return "full";
      }
      throw worker_error("unsupported HDR color range: " + range);
    }

    std::string hdr_decode_filter(const media_contract_t &media) {
      if (media.color == media_color_e::sdr) {
        throw worker_error("HDR decode filter requested for SDR");
      }
      return "zscale=rangein=" + zscale_range(media.color_range) +
             ":primariesin=" + media.color_primaries +
             ":transferin=" + media.color_transfer +
             ":matrixin=" + media.color_space +
             ":range=full:primaries=bt709:transfer=linear:matrix=gbr:npl=80,"
             "format=gbrpf32le";
    }

    std::string hdr_encode_filter(const media_contract_t &media) {
      if (media.color == media_color_e::sdr) {
        throw worker_error("HDR encode filter requested for SDR");
      }
      return "format=gbrpf32le,"
             "zscale=rangein=full:primariesin=bt709:transferin=linear:matrixin=gbr:"
             "range=" +
             zscale_range(media.color_range) +
             ":primaries=" + media.color_primaries +
             ":transfer=" + media.color_transfer +
             ":matrix=" + media.color_space +
             ":npl=80,format=p010le";
    }

    std::vector<std::string> codec_arguments_impl(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const bool hdr,
      const std::uint32_t encoded_width,
      const std::uint32_t encoded_height
    ) {
      if (spec.codec == "hevc_nvenc") {
        std::vector<std::string> result {
          "-c:v",
          "hevc_nvenc",
          "-preset",
          "p5",
          "-tune",
          "hq",
          "-rc",
          "vbr",
          "-cq",
          "18",
          "-b:v",
          "0",
          "-bf",
          "0",
          "-pix_fmt",
          hdr ? "p010le" : "yuv420p",
        };
        if (hdr) {
          result.insert(
            result.end(),
            {"-profile:v", "main10", "-extra_sei", "1"}
          );
        }
        return result;
      }
      if (spec.codec == "av1_nvenc") {
        std::vector<std::string> result {
          "-c:v",
          "av1_nvenc",
          "-preset",
          "p5",
          "-tune",
          "hq",
          "-rc",
          "vbr",
          "-cq",
          "20",
          "-b:v",
          "0",
          "-bf",
          "0",
          "-pix_fmt",
          hdr ? "p010le" : "yuv420p",
          "-level",
          select_av1_level(media, encoded_width, encoded_height),
        };
        if (hdr) {
          result.insert(result.end(), {"-extra_sei", "1"});
        }
        return result;
      }
      throw worker_error("unsupported offline codec");
    }

    struct pts_run_t {
      std::size_t begin = 0;
      std::size_t end = 0;
      std::int64_t first_pts = 0;
      std::int64_t step = 0;
    };

    std::vector<pts_run_t> pts_runs(const media_contract_t &media) {
      std::vector<pts_run_t> runs;
      if (media.frames.empty()) {
        return runs;
      }
      std::size_t begin = 0;
      while (begin < media.frames.size()) {
        const auto step = media.frames[begin].duration;
        std::size_t end = begin + 1;
        while (end < media.frames.size() &&
               media.frames[end].pts - media.frames[end - 1].pts == step) {
          ++end;
        }
        runs.push_back({
          .begin = begin,
          .end = end,
          .first_pts = media.frames[begin].pts,
          .step = step,
        });
        begin = end;
      }
      return runs;
    }

    void append_pts_lookup(
      std::ostringstream &output,
      const std::vector<pts_run_t> &runs,
      const std::size_t begin,
      const std::size_t end
    ) {
      if (end <= begin || end > runs.size()) {
        throw worker_error("cannot build an empty/out-of-range PTS lookup");
      }
      if (end - begin == 1) {
        const auto &run = runs[begin];
        if (run.end - run.begin == 1) {
          output << run.first_pts;
        } else {
          output << '(' << run.first_pts << "+(N-" << run.begin
                 << ")*" << run.step << ')';
        }
        return;
      }
      const auto middle = begin + (end - begin) / 2;
      output << "if(lt(N\\," << runs[middle].begin << ")\\,";
      append_pts_lookup(output, runs, begin, middle);
      output << "\\,";
      append_pts_lookup(output, runs, middle, end);
      output << ')';
    }

    void write_whole_clip_filter(
      const fs::path &path,
      const media_contract_t &media,
      const bool hdr,
      const std::uint32_t sbs_width,
      const std::uint32_t sbs_height
    ) {
      const auto runs = pts_runs(media);
      if (runs.empty()) {
        throw worker_error("cannot encode an empty source timeline");
      }
      constexpr std::size_t max_pts_runs = 1000000;
      if (runs.size() > max_pts_runs) {
        throw worker_error(
          "source VFR timeline exceeds one million exact timing runs"
        );
      }
      std::ostringstream lookup;
      lookup.imbue(std::locale::classic());
      append_pts_lookup(lookup, runs, 0, runs.size());
      const auto expression = lookup.str();
      constexpr std::size_t max_filter_script_bytes = 128ull * 1024ull * 1024ull;
      if (expression.size() > max_filter_script_bytes / (hdr ? 2 : 1)) {
        throw worker_error(
          "exact source timeline requires an unreasonably large PTS map"
        );
      }
      const auto timing_filter =
        "settb=expr=" + std::to_string(media.time_base.numerator) + "/" +
        std::to_string(media.time_base.denominator) + ",setpts='" +
        expression + "'";
      std::ostringstream graph;
      graph.imbue(std::locale::classic());
      if (hdr) {
        graph
          // HDR10 static side data cannot travel on the raw planar-float pipe. Keep the authenticated
          // source stream aligned one-for-one as a metadata donor; FFmpeg's frame-loop filters
          // do not preserve mastering-display/content-light side data on cloned frames.
          // Pixel values come exclusively from the direct SBS stream below, so the donor needs
          // only the raster and pixel format blend requires of both inputs: one nearest-neighbour
          // swscale pass, with no transfer or primaries conversion.
          << "[0:v:0]trim=end_frame=" << media.frames.size() << ','
          << timing_filter
          << ",scale=" << sbs_width << ':' << sbs_height
          // Scaling the source to the packed SBS width would otherwise preserve
          // its original display aspect by changing the donor SAR to 1:2.
          << ":flags=neighbor,format=gbrpf32le,setsar=1/1[donor];\n"
          << "[1:v:0]" << timing_filter
          << ",format=gbrpf32le[sbs];\n"
          // Normal mode at zero opacity copies every plane of the second (SBS) input unchanged;
          // the output frame keeps the first (donor) input's side data.
          << "[donor][sbs]blend=all_mode=normal:all_opacity=0:shortest=1,"
          << hdr_encode_filter(media) << "[vout]\n";
      } else {
        graph << "[0:v:0]" << timing_filter << "[vout]\n";
      }
      const auto contents = graph.str();
      if (contents.size() > max_filter_script_bytes) {
        throw worker_error(
          "exact source timeline filter exceeds the 128 MiB safety limit"
        );
      }
      write_text_atomic(path, contents);
    }

    std::vector<std::string> build_whole_clip_encode_command(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &filter_script,
      const fs::path &encoded_video,
      const std::uint32_t encoded_width,
      const std::uint32_t encoded_height
    ) {
      if (media.time_base.denominator > std::numeric_limits<std::int32_t>::max()) {
        throw worker_error(
          "source time-base denominator exceeds the MP4 timescale limit"
        );
      }
      const bool hdr = media.color != media_color_e::sdr;
      std::vector<std::string> command {
        path_utf8(spec.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-xerror",
        "-nostdin",
        "-y",
        "-copyts",
      };
      if (hdr) {
        // HDR static side data must accompany every donor frame. Pixel values always come from
        // the direct causal raw stream (the blend copies its planes).
        command.insert(command.end(), {
                                        "-protocol_whitelist",
                                        "file",
                                        "-i",
                                        path_utf8(spec.input_path),
                                      });
      }
      command.insert(command.end(), {
                                      "-protocol_whitelist",
                                      "pipe",
                                      "-f",
                                      "rawvideo",
                                      "-pixel_format",
                                      hdr ? "gbrpf32le" : "bgra",
                                      "-video_size",
                                      std::to_string(encoded_width) + "x" + std::to_string(encoded_height),
                                      "-framerate",
                                      "1",
                                      "-color_primaries",
                                      hdr || media.color_primaries.empty() ? "bt709" : media.color_primaries,
                                      "-color_trc",
                                      hdr ? "linear" : (media.color_transfer.empty() ? "iec61966-2-1" : media.color_transfer),
                                      "-i",
                                      "pipe:0",
                                    });
      command.insert(command.end(), {
                                      "-filter_complex_script",
                                      path_utf8(filter_script),
                                      "-map",
                                      "[vout]",
                                      "-an",
                                      "-sn",
                                      "-dn",
                                      "-fps_mode",
                                      "passthrough",
                                      "-enc_time_base",
                                      "filter",
                                    });
      const auto codec =
        build_codec_arguments(spec, media, hdr, encoded_width, encoded_height);
      command.insert(command.end(), codec.begin(), codec.end());
      command.insert(command.end(), {
                                      "-bsf:v",
                                      "setts=duration=if(eq(N\\," + std::to_string(media.frames.size() - 1) + ")\\," + std::to_string(media.frames.back().duration) + "\\,DURATION)",
                                      "-avoid_negative_ts",
                                      "disabled",
                                    });
      if (!media.color_range.empty()) {
        command.insert(command.end(), {
                                        "-color_range",
                                        media.color_range,
                                      });
      }
      if (!media.color_space.empty()) {
        command.insert(command.end(), {
                                        "-colorspace",
                                        media.color_space,
                                      });
      }
      if (!media.color_transfer.empty()) {
        command.insert(command.end(), {
                                        "-color_trc",
                                        media.color_transfer,
                                      });
      }
      if (!media.color_primaries.empty()) {
        command.insert(command.end(), {
                                        "-color_primaries",
                                        media.color_primaries,
                                      });
      }
      command.insert(command.end(), {
                                      "-video_track_timescale",
                                      std::to_string(media.time_base.denominator),
                                      "-movie_timescale",
                                      std::to_string(media.time_base.denominator),
                                      "-movflags",
                                      "+write_colr",
                                      "-f",
                                      "mp4",
                                      path_utf8(encoded_video),
                                    });
      return command;
    }

    void require_tags_preserved(
      const nlohmann::json &source,
      const nlohmann::json &output,
      const std::string_view description
    ) {
      for (const auto &[key, value] : source.items()) {
        const auto found = output.find(key);
        if (found == output.end() || *found != value) {
          throw worker_error(
            std::string(description) + " metadata was not preserved: " + key
          );
        }
      }
    }

    std::vector<const stream_inventory_entry_t *> streams_of_type(
      const stream_inventory_t &inventory,
      const std::string_view type
    ) {
      std::vector<const stream_inventory_entry_t *> result;
      for (const auto &stream : inventory.streams) {
        if (stream.type == type) {
          result.push_back(&stream);
        }
      }
      return result;
    }

  }  // namespace

  namespace worker_detail {
    streaming_decoder_t::streaming_decoder_t(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &log_path
    ):
        media_(media),
        format_(media.color == media_color_e::sdr ? "bgra8" : "gbrpf32le"),
        process_(child_process_t::launch(build_decoder_command(spec, media), spec.sunshine_executable.parent_path(), log_path, std::nullopt, true)) {
      const auto source_bytes = std::uint64_t(media.width) * media.height * (media.color == media_color_e::sdr ? 4 : 12);
      if (source_bytes > spec.transient_raster_hard_cap_bytes)
        throw worker_error("raw source frame exceeds the transient raster hard cap before allocation");
    }

    const std::string &streaming_decoder_t::format() const {
      return format_;
    }

    raw_frame streaming_decoder_t::read_next(const frame_timing_t &timing) {
      transport_timer_t timer("offline_decode_read");
      if (timing.sequence != next_sequence_ || timing.sequence > media_.frames.size())
        throw worker_error("streaming decoder sequence mismatch");
      raw_frame frame;
      frame.header = {1, timing.sequence, timing.pts, media_.time_base.numerator,
                      media_.time_base.denominator, media_.width, media_.height,
                      media_.color == media_color_e::sdr ? raw_pixel_format::bgra8 : raw_pixel_format::gbrpf32le};
      const auto pixels = std::size_t(media_.width) * media_.height;
      if (media_.color == media_color_e::sdr) {
        frame.bgra.resize(pixels * 4);
        process_.read_exact(frame.bgra.data(), frame.bgra.size());
      } else {
        frame.gbr32.resize(pixels * 3);
        process_.read_exact(frame.gbr32.data(), frame.gbr32.size() * sizeof(float));
        for (float value : frame.gbr32)
          if (!std::isfinite(value) || std::abs(value) > 65504.0f)
            throw worker_error("FFmpeg emitted non-finite/out-of-range raw HDR pixels");
      }
      ++next_sequence_;
      return frame;
    }

    void streaming_decoder_t::finish() {
      if (next_sequence_ != media_.frames.size() + 1) {
        throw worker_error("streaming decoder finished at the wrong frame count");
      }
      // Drain detection and process completion must be observed together. Waiting for exit
      // first deadlocks if FFmpeg has an unexpected extra frame blocked by the bounded pipe.
      const auto deadline = std::chrono::steady_clock::now() + 30s;
      while (process_.running() &&
             std::chrono::steady_clock::now() < deadline) {
        if (process_.read_one_if_available()) {
          process_.terminate();
          throw worker_error(
            "streaming decoder emitted more frames than FFprobe reported"
          );
        }
        std::this_thread::sleep_for(child_poll);
      }
      if (process_.running()) {
        process_.terminate();
        throw worker_error(
          "streaming decoder did not terminate at the probed frame count"
        );
      }
      const int code = process_.wait(1s);
      if (code != 0) {
        throw worker_error(
          "streaming decoder exited " + std::to_string(code)
        );
      }
      if (process_.read_one_if_available()) {
        throw worker_error(
          "streaming decoder emitted more frames than FFprobe reported"
        );
      }
      process_.terminate();
    }

    void streaming_decoder_t::abort() noexcept {
      process_.terminate();
    }

    std::string codec_name_for_request(const std::string &codec) {
      if (codec == "hevc_nvenc") {
        return "hevc";
      }
      if (codec == "av1_nvenc") {
        return "av1";
      }
      throw worker_error("unsupported offline codec");
    }

    whole_clip_encoder_t::whole_clip_encoder_t(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &work,
      const std::uint32_t sbs_width,
      const std::uint32_t sbs_height,
      const bool asynchronous
    ):
        encoded_video_(work / "encoded-video.mp4"),
        asynchronous_(asynchronous) {
      const auto filter_script = work / "encoded-video-filter.txt";
      write_whole_clip_filter(
        filter_script,
        media,
        media.color != media_color_e::sdr,
        sbs_width,
        sbs_height
      );
      process_ = child_process_t::launch(
        build_whole_clip_encode_command(
          spec,
          media,
          filter_script,
          encoded_video_,
          sbs_width,
          sbs_height
        ),
        spec.sunshine_executable.parent_path(),
        work / "logs" / "encode-whole-clip.log",
        std::nullopt, false, true
      );
      if (asynchronous_) {
        consumer_ = std::thread([this] {
          consume();
        });
      }
    }

    whole_clip_encoder_t::~whole_clip_encoder_t() {
      abort("whole-clip encoder destroyed");
    }

    void whole_clip_encoder_t::publish(
      const std::uint64_t sequence,
      raw_frame frame
    ) {
      if (sequence != next_sequence_++ || frame.header.sequence != sequence)
        throw worker_error("raw encoder frame sequence mismatch");
      if (finished_) {
        throw worker_error("cannot publish to a finished whole-clip encoder");
      }
      frame.validate(std::numeric_limits<std::uint64_t>::max());
      if (!asynchronous_) {
        write_frame(frame);
        return;
      }

      std::unique_lock lock(mutex_);
      const auto deadline =
        std::chrono::steady_clock::now() + frame_io_timeout;
      const auto outstanding = [&] {
        return static_cast<std::size_t>(pending_.has_value()) +
               static_cast<std::size_t>(active_);
      };
      while (outstanding() >= offline_encoder_pipeline_capacity &&
             error_.empty()) {
        if (!process_.running()) {
          error_ =
            "whole-clip encoder exited before accepting frame " +
            std::to_string(sequence) + " (exit " +
            std::to_string(process_.exit_code()) + ")";
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          error_ =
            "timed out waiting for the bounded encoder handoff slot";
          break;
        }
        condition_.wait_for(lock, child_poll);
      }
      if (!error_.empty()) {
        throw worker_error("whole-clip encoder pipeline failed: " + error_);
      }
      if (stopping_ || input_finished_ ||
          outstanding() >= offline_encoder_pipeline_capacity) {
        throw worker_error("whole-clip encoder pipeline is not accepting frames");
      }
      pending_ = std::move(frame);
      condition_.notify_all();
    }

    void whole_clip_encoder_t::finish() {
      if (finished_) {
        throw worker_error("whole-clip encoder was already finished");
      }
      if (asynchronous_) {
        {
          std::lock_guard lock(mutex_);
          input_finished_ = true;
          condition_.notify_all();
        }
        if (consumer_.joinable()) {
          consumer_.join();
        }
        std::lock_guard lock(mutex_);
        if (!error_.empty()) {
          throw worker_error("whole-clip encoder pipeline failed: " + error_);
        }
        if (pending_ || active_) {
          throw worker_error("whole-clip encoder pipeline did not drain");
        }
      }
      process_.close_stdin();
      const int code = process_.wait(
        std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
      );
      if (code != 0) {
        throw worker_error(
          "whole-clip encoder exited " + std::to_string(code)
        );
      }
      process_.terminate();
      finished_ = true;
    }

    void whole_clip_encoder_t::abort(const std::string &) noexcept {
      if (finished_) {
        return;
      }
      if (asynchronous_) {
        {
          std::lock_guard lock(mutex_);
          stopping_ = true;
          pending_.reset();
          condition_.notify_all();
        }
      }
      process_.cancel_writes();
      if (consumer_.joinable()) {
        consumer_.join();
      }
      process_.terminate();
      finished_ = true;
    }

    const fs::path &whole_clip_encoder_t::encoded_video() const {
      return encoded_video_;
    }

    void whole_clip_encoder_t::write_frame(const raw_frame &frame) {
      transport_timer_t timer("offline_encoder_write");
      const auto deadline = std::chrono::steady_clock::now() + frame_io_timeout;
      if (frame.header.format == raw_pixel_format::bgra8) {
        process_.write_exact(frame.bgra.data(), frame.bgra.size(), deadline);
        return;
      }
      if (frame.header.format != raw_pixel_format::rgba16f)
        throw worker_error("encoder received unsupported raw pixel format");
      // FFmpeg's planar float input has the same values as the former PFM
      // interchange. Expand bounded chunks, with no full-frame float copy.
      std::array<float, 16384> chunk;
      const auto pixels = std::size_t(frame.header.width) * frame.header.height;
      for (unsigned channel : {1u, 2u, 0u}) {
        for (std::size_t first = 0; first < pixels; first += chunk.size()) {
          const auto count = std::min(chunk.size(), pixels - first);
          for (std::size_t x = 0; x < count; ++x) {
            const auto h = frame.rgba16[(first + x) * 4 + channel];
            if (((h >> 10) & 31u) == 31u) throw worker_error("non-finite raw HDR SBS output");
            chunk[x] = finite_half_to_float(h);
          }
          process_.write_exact(chunk.data(), count * sizeof(float), deadline);
        }
      }
    }

    void whole_clip_encoder_t::consume() noexcept {
      for (;;) {
        raw_frame frame;
        {
          std::unique_lock lock(mutex_);
          condition_.wait(lock, [&] {
            return stopping_ || pending_ || input_finished_;
          });
          if (stopping_) {
            return;
          }
          if (!pending_) {
            return;
          }
          frame = std::move(*pending_);
          pending_.reset();
          active_ = true;
        }

        std::string failure;
        try {
          write_frame(frame);
        } catch (const std::exception &exception) {
          failure = exception.what();
        }

        {
          std::lock_guard lock(mutex_);
          active_ = false;
          if (!failure.empty() && error_.empty()) {
            error_ = std::move(failure);
            stopping_ = true;
          }
          condition_.notify_all();
        }
        if (!failure.empty()) {
          return;
        }
      }
    }

    std::string output_container(const fs::path &path) {
      const auto extension = lower(path.extension().string());
      if (extension == ".mkv") {
        return "matroska";
      }
      if (extension == ".mp4") {
        return "mp4";
      }
      throw worker_error("offline output must use .mkv or .mp4");
    }

    void validate_output_time_base(
      const worker_spec_t &spec,
      const media_contract_t &media
    ) {
      if (!spec.staging_output) {
        return;
      }
      validate_avexpr_timeline_exactness(media);
      (void) output_container(*spec.staging_output);
      if (media.time_base.denominator > std::numeric_limits<std::int32_t>::max()) {
        throw worker_error(
          "source time-base denominator exceeds the MP4 encoder limit"
        );
      }
    }

    void validate_stream_preflight(
      const worker_spec_t &spec,
      const stream_inventory_t &inventory
    ) {
      if (streams_of_type(inventory, "video").size() != 1) {
        throw worker_error(
          "offline conversion requires exactly one source video stream"
        );
      }
      static const std::set<std::string> known_types {
        "video",
        "audio",
        "subtitle",
        "data",
        "attachment",
      };
      for (const auto &stream : inventory.streams) {
        if (!known_types.contains(stream.type)) {
          throw worker_error(
            "unsupported source stream type would be dropped: " + stream.type
          );
        }
      }
      const auto container = output_container(*spec.staging_output);
      if (container == "mp4") {
        static const std::set<std::string> mp4_audio_codecs {
          "aac",
          "ac3",
          "alac",
          "eac3",
          "flac",
          "mp2",
          "mp3",
          "opus",
          "pcm_f32le",
          "pcm_s16le",
          "pcm_s24le",
          "pcm_s32le",
        };
        for (const auto *audio : streams_of_type(inventory, "audio")) {
          if (!mp4_audio_codecs.contains(audio->codec)) {
            throw worker_error(
              "MP4 cannot copy source audio codec '" + audio->codec +
              "' exactly; choose MKV or transcode the source audio first"
            );
          }
        }
        for (const auto *subtitle : streams_of_type(inventory, "subtitle")) {
          if (subtitle->codec != "mov_text") {
            throw worker_error(
              "MP4 cannot copy source subtitle codec '" + subtitle->codec +
              "' exactly; choose MKV or convert subtitles to mov_text"
            );
          }
        }
        if (!streams_of_type(inventory, "data").empty() || !streams_of_type(inventory, "attachment").empty()) {
          throw worker_error(
            "MP4 cannot preserve source data/attachment streams; choose MKV"
          );
        }
      } else if (!streams_of_type(inventory, "data").empty()) {
        throw worker_error(
          "FFmpeg Matroska cannot preserve this source data stream exactly"
        );
      }
    }

    nlohmann::json validate_stream_inventory(
      const stream_inventory_t &source,
      const stream_inventory_t &output,
      const std::uint64_t tolerance_actual_ticks
    ) {
      long double max_packet_pts_error = 0;
      long double max_packet_dts_error = 0;
      long double max_packet_duration_error = 0;
      long double max_stream_duration_error = 0;
      long double max_chapter_error = 0;
      std::uint64_t copied_packet_count = 0;
      if (tolerance_actual_ticks != 0) {
        for (const auto &stream : output.streams) {
          if (!stream.packets.empty()) {
            require_millisecond_or_finer(
              stream.time_base,
              "output auxiliary stream"
            );
          }
        }
        for (const auto &chapter : output.chapters) {
          require_millisecond_or_finer(
            chapter.time_base,
            "output chapter"
          );
        }
      }
      require_tags_preserved(
        source.format_tags,
        output.format_tags,
        "global"
      );
      for (const auto type : {
             std::string_view {"video"},
             std::string_view {"audio"},
             std::string_view {"subtitle"},
             std::string_view {"data"},
             std::string_view {"attachment"},
           }) {
        const auto wanted = streams_of_type(source, type);
        const auto actual = streams_of_type(output, type);
        if (wanted.size() != actual.size()) {
          throw worker_error(
            "output " + std::string(type) + " stream count differs from source"
          );
        }
        for (std::size_t index = 0; index < wanted.size(); ++index) {
          if (type != "video" && wanted[index]->codec != actual[index]->codec) {
            throw worker_error(
              "output " + std::string(type) +
              " codec differs from the copied source stream"
            );
          }
          require_tags_preserved(
            wanted[index]->tags,
            actual[index]->tags,
            std::string(type) + " stream"
          );
          const auto expected_disposition =
            type == "subtitle" ? nlohmann::json::object() : wanted[index]->disposition;
          if (expected_disposition != actual[index]->disposition) {
            throw worker_error(
              "output " + std::string(type) +
              (
                type == "subtitle" ?
                  " stream disposition was not cleared" :
                  " stream disposition differs from source"
              )
            );
          }
          if (type == "video") {
            continue;
          }
          if (wanted[index]->packets.size() != actual[index]->packets.size()) {
            throw worker_error(
              "output " + std::string(type) +
              " packet count differs from source"
            );
          }
          for (std::size_t packet_index = 0;
               packet_index < wanted[index]->packets.size();
               ++packet_index) {
            const auto &wanted_packet = wanted[index]->packets[packet_index];
            const auto &actual_packet = actual[index]->packets[packet_index];
            const auto compare = [&](const std::optional<std::int64_t> expected, const std::optional<std::int64_t> got, const std::string_view field, long double &max_error) {
              if (!expected) {
                return;
              }
              if (!got || !rational_ticks_within_actual_ticks(*expected, wanted[index]->time_base, *got, actual[index]->time_base, tolerance_actual_ticks)) {
                throw worker_error(
                  "output " + std::string(type) + " packet " +
                  std::string(field) + " differs from source"
                );
              }
              max_error = std::max(
                max_error,
                rational_error_in_actual_ticks(
                  *expected,
                  wanted[index]->time_base,
                  *got,
                  actual[index]->time_base
                )
              );
            };
            compare(
              wanted_packet.pts,
              actual_packet.pts,
              "PTS",
              max_packet_pts_error
            );
            compare(
              wanted_packet.dts,
              actual_packet.dts,
              "DTS",
              max_packet_dts_error
            );
            compare(
              wanted_packet.duration,
              actual_packet.duration,
              "duration",
              max_packet_duration_error
            );
            ++copied_packet_count;
          }
          const auto stream_span = [](const stream_inventory_entry_t &stream) {
            std::optional<std::int64_t> first;
            std::optional<std::int64_t> end;
            for (const auto &packet : stream.packets) {
              if (!packet.pts) {
                continue;
              }
              first = first ? std::min(*first, *packet.pts) : packet.pts;
              auto packet_end = *packet.pts;
              if (packet.duration && *packet.duration > 0) {
                if (packet_end > std::numeric_limits<std::int64_t>::max() - *packet.duration) {
                  throw worker_error(
                    "auxiliary packet end timestamp overflows"
                  );
                }
                packet_end += *packet.duration;
              }
              end = end ? std::max(*end, packet_end) :
                          std::optional<std::int64_t>(packet_end);
            }
            return std::pair {first, end};
          };
          const auto wanted_span = stream_span(*wanted[index]);
          const auto actual_span = stream_span(*actual[index]);
          if (wanted_span.first.has_value() != actual_span.first.has_value() || wanted_span.second.has_value() != actual_span.second.has_value()) {
            throw worker_error(
              "output auxiliary stream duration cannot be attested"
            );
          }
          if (wanted_span.first && wanted_span.second) {
            using boost::multiprecision::cpp_int;
            const cpp_int wanted_span_ticks =
              cpp_int(*wanted_span.second) - *wanted_span.first;
            const cpp_int actual_span_ticks =
              cpp_int(*actual_span.second) - *actual_span.first;
            if (wanted_span_ticks < 0 || actual_span_ticks < 0 || wanted_span_ticks > std::numeric_limits<std::int64_t>::max() || actual_span_ticks > std::numeric_limits<std::int64_t>::max()) {
              throw worker_error(
                "auxiliary stream duration is invalid"
              );
            }
            const auto wanted_duration =
              wanted_span_ticks.convert_to<std::int64_t>();
            const auto actual_duration =
              actual_span_ticks.convert_to<std::int64_t>();
            if (!rational_ticks_within_actual_ticks(
                  wanted_duration,
                  wanted[index]->time_base,
                  actual_duration,
                  actual[index]->time_base,
                  tolerance_actual_ticks
                )) {
              throw worker_error(
                "output auxiliary stream accumulated timeline drift"
              );
            }
            max_stream_duration_error = std::max(
              max_stream_duration_error,
              rational_error_in_actual_ticks(
                wanted_duration,
                wanted[index]->time_base,
                actual_duration,
                actual[index]->time_base
              )
            );
          }
        }
      }
      if (source.chapters.size() != output.chapters.size()) {
        throw worker_error("output chapter count differs from source");
      }
      for (std::size_t index = 0; index < source.chapters.size(); ++index) {
        const auto &wanted = source.chapters[index];
        const auto &actual = output.chapters[index];
        if (!rational_ticks_within_actual_ticks(wanted.start, wanted.time_base, actual.start, actual.time_base, tolerance_actual_ticks) || !rational_ticks_within_actual_ticks(wanted.end, wanted.time_base, actual.end, actual.time_base, tolerance_actual_ticks)) {
          throw worker_error("output chapter timing differs from source");
        }
        max_chapter_error = std::max({
          max_chapter_error,
          rational_error_in_actual_ticks(
            wanted.start,
            wanted.time_base,
            actual.start,
            actual.time_base
          ),
          rational_error_in_actual_ticks(
            wanted.end,
            wanted.time_base,
            actual.end,
            actual.time_base
          ),
        });
        require_tags_preserved(
          wanted.tags,
          actual.tags,
          "chapter"
        );
      }
      return {
        {"copied_packet_count", copied_packet_count},
        {"max_packet_pts_error_output_ticks",
         static_cast<double>(max_packet_pts_error)},
        {"max_packet_dts_error_output_ticks",
         static_cast<double>(max_packet_dts_error)},
        {"max_packet_duration_error_output_ticks",
         static_cast<double>(max_packet_duration_error)},
        {"max_stream_duration_error_output_ticks",
         static_cast<double>(max_stream_duration_error)},
        {"max_chapter_error_output_ticks",
         static_cast<double>(max_chapter_error)},
      };
    }

    std::vector<std::string> build_mux_command(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const stream_inventory_t &source_inventory,
      const fs::path &encoded_video
    ) {
      if (!spec.staging_output) {
        throw worker_error("conversion staging output is missing");
      }
      const auto container = output_container(*spec.staging_output);
      std::vector<std::string> command {
        path_utf8(spec.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-xerror",
        "-nostdin",
        // The worker atomically reserves and identity-pins this exact staging file
        // before launching any media tool. FFmpeg must truncate that owned placeholder.
        "-y",
        "-copyts",
        "-protocol_whitelist",
        "file",
        "-i",
        path_utf8(encoded_video),
        "-protocol_whitelist",
        "file",
        "-i",
        path_utf8(spec.input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
      };
      if (container == "matroska") {
        command.insert(command.end(), {
                                        "-map",
                                        "1:t?",
                                      });
      }
      command.insert(command.end(), {
                                      "-map_metadata",
                                      "-1",
                                      "-map_chapters",
                                      "1",
                                      "-c",
                                      "copy",
                                      "-avoid_negative_ts",
                                      "disabled",
                                    });
      const auto source_video = streams_of_type(source_inventory, "video");
      if (source_video.size() != 1) {
        throw worker_error("source video disposition contract is missing");
      }
      const auto append_metadata =
        [&](const std::string &specifier, const nlohmann::json &tags) {
          for (const auto &[name, value] : tags.items()) {
            if (!value.is_string()) {
              throw worker_error(
                "source metadata value is not textual: " + name
              );
            }
            command.insert(command.end(), {
                                            specifier,
                                            name + "=" + value.get<std::string>(),
                                          });
          }
        };
      const auto append_disposition =
        [&](const std::string &specifier, const nlohmann::json &disposition) {
          std::string enabled_dispositions;
          for (const auto &[name, enabled] : disposition.items()) {
            if (enabled.is_boolean() && enabled.get<bool>()) {
              if (!enabled_dispositions.empty()) {
                enabled_dispositions.push_back('+');
              }
              enabled_dispositions += name;
            }
          }
          command.insert(command.end(), {
                                          specifier,
                                          enabled_dispositions.empty() ? "0" : enabled_dispositions,
                                        });
        };
      const auto append_stream_contracts =
        [&](const std::string_view type, const std::string_view ffmpeg_type, const bool preserve_disposition = true) {
          const auto streams = streams_of_type(source_inventory, type);
          for (std::size_t index = 0; index < streams.size(); ++index) {
            const auto stream_index = std::to_string(index);
            append_metadata(
              "-metadata:s:" + std::string(ffmpeg_type) + ":" + stream_index,
              streams[index]->tags
            );
            if (preserve_disposition) {
              append_disposition(
                "-disposition:" + std::string(ffmpeg_type) + ":" + stream_index,
                streams[index]->disposition
              );
            }
          }
        };
      append_metadata("-metadata", source_inventory.format_tags);
      append_stream_contracts("video", "v");
      append_stream_contracts("audio", "a");
      const auto subtitle_streams = streams_of_type(source_inventory, "subtitle");
      append_stream_contracts("subtitle", "s", false);
      if (!subtitle_streams.empty()) {
        // A player cannot safely render one soft-subtitle plane over a packed SBS
        // raster. Preserve the streams and their metadata, but remove default/forced
        // disposition signals that can drive automatic selection. A player may still
        // select a stream through explicit language or user policy.
        command.insert(command.end(), {"-disposition:s", "0"});
      }
      if (container == "matroska") {
        append_stream_contracts("attachment", "t");
      }
      for (std::size_t index = 0; index < source_inventory.chapters.size(); ++index) {
        append_metadata(
          "-metadata:c:" + std::to_string(index),
          source_inventory.chapters[index].tags
        );
      }
      if (container == "mp4") {
        if (spec.codec == "hevc_nvenc") {
          command.insert(command.end(), {"-tag:v", "hvc1"});
        }
        command.insert(command.end(), {
                                        "-video_track_timescale",
                                        std::to_string(media.time_base.denominator),
                                        "-movie_timescale",
                                        std::to_string(media.time_base.denominator),
                                        "-movflags",
                                        "+faststart+write_colr",
                                        "-f",
                                        "mp4",
                                      });
      } else {
        command.insert(command.end(), {"-f", "matroska"});
      }
      command.push_back(path_utf8(*spec.staging_output));
      return command;
    }

    void mux_encoded_video(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const stream_inventory_t &source_inventory,
      const fs::path &encoded_video,
      const fs::path &log_path
    ) {
      std::error_code ec;
      if (!fs::is_regular_file(encoded_video, ec) || ec || fs::file_size(encoded_video, ec) == 0 || ec) {
        throw worker_error("conversion produced no compressed video");
      }
      run_logged(
        build_mux_command(spec, media, source_inventory, encoded_video),
        spec.sunshine_executable.parent_path(),
        log_path
      );
    }
  }  // namespace worker_detail

  std::vector<std::string> build_codec_arguments(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const bool hdr,
    const std::uint32_t encoded_width,
    const std::uint32_t encoded_height
  ) {
    return codec_arguments_impl(
      spec,
      media,
      hdr,
      encoded_width,
      encoded_height
    );
  }

#ifdef SUNSHINE_TESTS
  std::size_t offline_encoder_pipeline_capacity_for_test() {
    return offline_encoder_pipeline_capacity;
  }

  std::vector<std::string> build_mux_command_for_test(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const nlohmann::json &source_inventory,
    const fs::path &encoded_video
  ) {
    return build_mux_command(
      spec,
      media,
      parse_stream_inventory(source_inventory),
      encoded_video
    );
  }
#endif

  void validate_avexpr_timeline_exactness(const media_contract_t &media) {
    using boost::multiprecision::cpp_int;
    static const cpp_int max_exact_avexpr_integer {
      "9007199254740991"
    };
    if (media.frames.empty() || media.frames.size() > max_sequence) {
      throw worker_error("cannot generate AVExpr for an invalid frame sequence");
    }
    for (const auto &frame : media.frames) {
      const cpp_int pts = frame.pts;
      const cpp_int duration = frame.duration;
      if (frame.duration <= 0 || boost::multiprecision::abs(pts) > max_exact_avexpr_integer || duration > max_exact_avexpr_integer) {
        throw worker_error(
          "source timestamps exceed AVExpr's exact integer range"
        );
      }
    }
    for (const auto &run : pts_runs(media)) {
      if (run.end <= run.begin || run.step <= 0) {
        throw worker_error("source timing run is invalid");
      }
      const cpp_int index_span = run.end - run.begin - 1;
      const cpp_int product = index_span * run.step;
      if (product > max_exact_avexpr_integer) {
        throw worker_error(
          "source timing run exceeds AVExpr's exact multiplication range"
        );
      }
      const cpp_int last_pts = cpp_int(run.first_pts) + product;
      if (boost::multiprecision::abs(last_pts) > max_exact_avexpr_integer) {
        throw worker_error(
          "source timing run exceeds AVExpr's exact addition range"
        );
      }
    }
  }

  std::string select_av1_level(
    const media_contract_t &media,
    const std::uint32_t encoded_width,
    const std::uint32_t encoded_height
  ) {
    struct level_limit_t {
      const char *name;
      std::uint64_t max_picture_size;
      std::uint32_t max_width;
      std::uint32_t max_height;
      std::uint64_t max_display_rate;
      std::uint32_t max_header_rate;
    };

    // AV1 Bitstream & Decoding Process Specification, Annex A.  Reserved level
    // indices (including 7.x) deliberately do not appear here.
    static constexpr std::array levels {
      level_limit_t {"2.0", 147456, 2048, 1152, 4423680, 150},
      level_limit_t {"2.1", 278784, 2816, 1584, 8363520, 150},
      level_limit_t {"3.0", 665856, 4352, 2448, 19975680, 150},
      level_limit_t {"3.1", 1065024, 5504, 3096, 31950720, 150},
      level_limit_t {"4.0", 2359296, 6144, 3456, 70778880, 300},
      level_limit_t {"4.1", 2359296, 6144, 3456, 141557760, 300},
      level_limit_t {"5.0", 8912896, 8192, 4352, 267386880, 300},
      level_limit_t {"5.1", 8912896, 8192, 4352, 534773760, 300},
      level_limit_t {"5.2", 8912896, 8192, 4352, 1069547520, 300},
      level_limit_t {"5.3", 8912896, 8192, 4352, 1069547520, 300},
      level_limit_t {"6.0", 35651584, 16384, 8704, 1069547520, 300},
      level_limit_t {"6.1", 35651584, 16384, 8704, 2139095040, 300},
      level_limit_t {"6.2", 35651584, 16384, 8704, 4278190080, 300},
      level_limit_t {"6.3", 35651584, 16384, 8704, 4278190080, 300},
    };

    if (
      encoded_width == 0 || encoded_height == 0 || media.frames.empty() ||
      media.time_base.numerator <= 0 || media.time_base.denominator <= 0
    ) {
      throw worker_error("cannot select an AV1 level from an invalid media contract");
    }
    const auto shortest =
      std::min_element(
        media.frames.begin(),
        media.frames.end(),
        [](const frame_timing_t &left, const frame_timing_t &right) {
          return left.duration < right.duration;
        }
      )->duration;
    if (shortest <= 0) {
      throw worker_error("cannot select an AV1 level for a non-positive frame interval");
    }

    using boost::multiprecision::cpp_int;
    // NVENC validates AV1 levels against its coded superblock raster, not only
    // the visible frame dimensions. AV1 superblocks are at least 64x64, so a
    // visible height such as 180 is coded as 192 lines. Selecting against the
    // visible raster can therefore choose a boundary level that NVENC rejects.
    constexpr std::uint64_t av1_superblock_size = 64;
    const auto coded_width =
      (static_cast<std::uint64_t>(encoded_width) + av1_superblock_size - 1) /
      av1_superblock_size * av1_superblock_size;
    const auto coded_height =
      (static_cast<std::uint64_t>(encoded_height) + av1_superblock_size - 1) /
      av1_superblock_size * av1_superblock_size;
    const cpp_int picture_size = cpp_int(coded_width) * coded_height;
    // display_rate = picture_size * time_base.denominator /
    //                (shortest_duration * time_base.numerator).
    // Compare as exact integers so nanosecond and large-container time bases cannot
    // silently round a boundary down into the wrong level.
    const cpp_int display_rate_numerator =
      picture_size * media.time_base.denominator;
    const cpp_int interval_denominator =
      cpp_int(shortest) * media.time_base.numerator;

    for (const auto &level : levels) {
      if (
        picture_size > level.max_picture_size ||
        coded_width > level.max_width ||
        coded_height > level.max_height ||
        display_rate_numerator >
          cpp_int(level.max_display_rate) * interval_denominator ||
        media.time_base.denominator >
          cpp_int(level.max_header_rate) * interval_denominator
      ) {
        continue;
      }
      return level.name;
    }
    throw worker_error(
      "packed SBS coded raster/frame rate exceeds the defined AV1 level 6.3 limits"
    );
  }

  std::vector<std::string> build_decoder_command(
    const worker_spec_t &spec,
    const media_contract_t &media
  ) {
    std::vector<std::string> command {
      path_utf8(spec.ffmpeg_executable),
      "-hide_banner",
      "-loglevel",
      "warning",
      "-xerror",
      "-nostdin",
      "-noautorotate",
      "-protocol_whitelist",
      "file",
      "-i",
      path_utf8(spec.input_path),
      "-map",
      "0:v:0",
      "-an",
      "-sn",
      "-dn",
      "-fps_mode",
      "passthrough",
    };
    if (media.color == media_color_e::sdr) {
      command.insert(command.end(), {
                                      "-pix_fmt",
                                      "bgra",
                                      "-f",
                                      "rawvideo",
                                      "pipe:1",
                                    });
    } else {
      command.insert(command.end(), {
                                      "-vf",
                                      hdr_decode_filter(media),
                                      "-pix_fmt",
                                      "gbrpf32le",
                                      "-f",
                                      "rawvideo",
                                      "pipe:1",
                                    });
    }
    return command;
  }
}  // namespace offline_sbs

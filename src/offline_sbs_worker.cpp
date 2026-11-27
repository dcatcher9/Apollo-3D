/**
 * @file src/offline_sbs_worker.cpp
 * @brief Native causal SBS evaluation/conversion worker.
 *
 * This file owns the worker entry: the authenticated specification, the staging-output claim,
 * progress and result publication, and the source-order causal pass that joins the FFprobe
 * contract, the FFmpeg decoder/encoder and the in-process native pass.
 */

#include "offline_sbs_worker.h"

#include "config.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "host_sbs_observation_timeline.h"
#include "offline_frame_transport.h"
#include "offline_sbs_contract.h"
#include "offline_sbs_digest.h"
#include "offline_sbs_runner.h"
#include "offline_sbs_wire_contract.h"
#include "offline_sbs_worker_ffmpeg.h"
#include "offline_sbs_worker_io.h"
#include "offline_sbs_worker_probe.h"
#ifdef SUNSHINE_TESTS
  #include "offline_sbs_progress_test.h"
#endif
#include "sbs_perf.h"

#include <algorithm>
#include <atomic>
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <windows.h>

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;
  using namespace worker_detail;

  namespace {
    // The manager already owns an isolated worker process. Run its only GPU
    // pipeline here, on one thread, so pixels cross bounded memory channels.
    class analysis_process_t final {
    public:
      analysis_process_t(native_pass_settings settings, raw_frame_transport &transport):
          transport_(transport) {
        thread_ = std::thread([this, settings = std::move(settings)]() {
          try {
            code_.store(run_native_pass(settings, transport_, contract_));
          } catch (...) {
            // The typed entry normally publishes even a failed terminal. An exceptional escape
            // cannot leave a consumer waiting for evidence that will never be notified.
            code_.store(10);
            transport_.cancel();
          }
          if (code_.load() != 0) {
            // Preserve an ordinary failed terminal until the worker consumes its status.
            transport_.source.cancel();
            transport_.sbs.cancel();
          }
          running_.store(false, std::memory_order_release);
        });
      }

      ~analysis_process_t() {
        terminate();
      }

      bool running() const {
        return running_.load(std::memory_order_acquire);
      }
      int exit_code() const { return code_.load(); }
      // The successful pass's whole-clip contract; read only after wait() has joined its thread.
      const wire::whole_clip_contract_t &contract() const { return contract_; }
      int wait(std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (running()) {
          if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("causal GPU pass timed out");
          std::this_thread::sleep_for(10ms);
        }
        if (thread_.joinable()) thread_.join();
        return exit_code();
      }
      void terminate() noexcept {
        transport_.cancel();
        if (thread_.joinable()) thread_.join();
      }
    private:
      raw_frame_transport &transport_;
      wire::whole_clip_contract_t contract_;
      std::atomic_bool running_ {true};
      std::atomic_int code_ {-1};
      std::thread thread_;
    };

    constexpr std::uintmax_t max_spec_bytes = 4ull * 1024ull * 1024ull;
    constexpr std::size_t offline_source_pipeline_capacity = 2;

    std::vector<std::uint64_t> exact_observation_timestamps(
      const media_contract_t &media
    ) {
      using boost::multiprecision::cpp_int;
      if (media.frames.empty() || media.time_base.numerator <= 0 ||
          media.time_base.denominator <= 0) {
        throw worker_error("source observation timeline has no positive time base/frames");
      }
      const cpp_int first_pts = media.frames.front().pts;
      const cpp_int numerator = media.time_base.numerator;
      const cpp_int denominator = media.time_base.denominator;
      const cpp_int maximum = std::numeric_limits<std::uint64_t>::max();
      std::vector<std::uint64_t> timestamps;
      timestamps.reserve(media.frames.size());
      for (std::size_t index = 0u; index < media.frames.size(); ++index) {
        const auto &frame = media.frames[index];
        if (frame.sequence != index + 1u) {
          throw worker_error("source observation timeline sequence is not contiguous");
        }
        const cpp_int delta = cpp_int(frame.pts) - first_pts;
        const cpp_int elapsed_us = delta * numerator * 1000000 / denominator;
        if (elapsed_us < 0 || elapsed_us >= maximum) {
          throw worker_error("source observation timestamp is outside uint64 microseconds");
        }
        timestamps.push_back((elapsed_us + 1).convert_to<std::uint64_t>());
      }
      if (!models::host_sbs_observation_timeline::valid_timestamps(timestamps)) {
        throw worker_error("source observation timestamps are zero or regressed");
      }
      return timestamps;
    }

    void write_observation_timeline(
      const fs::path &path,
      const media_contract_t &media
    ) {
      const auto timestamps = exact_observation_timestamps(media);
      std::string error;
      if (!models::host_sbs_observation_timeline::write(path, timestamps, error)) {
        throw worker_error(error);
      }
    }

    bool constant_time_equal(
      const std::string_view left,
      const std::string_view right
    ) {
      if (left.size() != right.size()) {
        return false;
      }
      unsigned char difference = 0;
      for (std::size_t index = 0; index < left.size(); ++index) {
        difference |= static_cast<unsigned char>(left[index] ^ right[index]);
      }
      return difference == 0;
    }

    std::string read_bounded_bytes(
      const fs::path &path,
      const std::uintmax_t max_bytes,
      bool *acquisition_failed = nullptr
    ) {
      if (acquisition_failed) {
        *acquisition_failed = false;
      }
      std::ifstream stream(path, std::ios::binary | std::ios::ate);
      if (!stream) {
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
        throw worker_error("cannot open bounded file snapshot");
      }
      const auto end = static_cast<std::streamoff>(stream.tellg());
      if (end < 0) {
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
        throw worker_error("cannot measure bounded file snapshot");
      }
      if (end == 0 || static_cast<std::uintmax_t>(end) > max_bytes ||
          static_cast<std::uintmax_t>(end) >
            std::numeric_limits<std::size_t>::max() ||
          static_cast<std::uintmax_t>(end) >
            static_cast<std::uintmax_t>(
              std::numeric_limits<std::streamsize>::max()
            )) {
        throw worker_error("bounded file snapshot has an invalid size");
      }
      std::string bytes(static_cast<std::size_t>(end), '\0');
      stream.seekg(0, std::ios::beg);
      if (!stream ||
          !stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
        throw worker_error("cannot read complete bounded file snapshot");
      }
      char trailing = 0;
      if (stream.read(&trailing, 1) || !stream.eof()) {
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
        throw worker_error("bounded file snapshot changed while being read");
      }
      return bytes;
    }

    /**
     * Atomically reserves the manager-selected staging pathname and keeps the exact
     * file identity open without delete sharing while FFmpeg writes it. Failed output
     * is deliberately preserved: cleanup never performs a pathname deletion that
     * could target a user file after the child loses its identity lock.
     */
    class staging_output_claim_t {
    public:
      explicit staging_output_claim_t(fs::path path):
          path_(std::move(path)) {
        handle_ = CreateFileW(
          path_.c_str(),
          FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          nullptr,
          CREATE_NEW,
          FILE_ATTRIBUTE_NORMAL,
          nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
          throw worker_error(
            "cannot exclusively reserve conversion staging output"
          );
        }
        BY_HANDLE_FILE_INFORMATION information {};
        if (
          !GetFileInformationByHandle(handle_, &information) ||
          (information.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0
        ) {
          cleanup_owned();
          throw worker_error("cannot attest reserved staging output");
        }
        volume_serial_ = information.dwVolumeSerialNumber;
        file_index_high_ = information.nFileIndexHigh;
        file_index_low_ = information.nFileIndexLow;
      }

      ~staging_output_claim_t() {
        cleanup_owned();
      }

      staging_output_claim_t(const staging_output_claim_t &) = delete;
      staging_output_claim_t &operator=(const staging_output_claim_t &) = delete;

      staging_output_claim_t(staging_output_claim_t &&other) noexcept:
          path_(std::move(other.path_))
          ,
          handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
          volume_serial_(other.volume_serial_),
          file_index_high_(other.file_index_high_),
          file_index_low_(other.file_index_low_)
      {
      }

      staging_output_claim_t &operator=(
        staging_output_claim_t &&other
      ) noexcept {
        if (this == &other) {
          return *this;
        }
        cleanup_owned();
        path_ = std::move(other.path_);
        handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
        volume_serial_ = other.volume_serial_;
        file_index_high_ = other.file_index_high_;
        file_index_low_ = other.file_index_low_;
        return *this;
      }

      [[nodiscard]] nlohmann::json identity_json() const {
        const auto information = verified_path_information();
        const auto size =
          (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
          information.nFileSizeLow;
        if (size == 0) {
          throw worker_error("conversion staging output is empty");
        }
        return {
          {"volume_serial", information.dwVolumeSerialNumber},
          {"file_index_high", information.nFileIndexHigh},
          {"file_index_low", information.nFileIndexLow},
          {"size_bytes", size},
        };
      }

      void release_for_publish() {
        (void) identity_json();
        CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
      }

    private:
      [[nodiscard]] BY_HANDLE_FILE_INFORMATION
      verified_path_information() const {
        if (handle_ == INVALID_HANDLE_VALUE) {
          throw worker_error("conversion staging output claim is not active");
        }
        const HANDLE observed = CreateFileW(
          path_.c_str(),
          FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          nullptr,
          OPEN_EXISTING,
          FILE_FLAG_OPEN_REPARSE_POINT,
          nullptr
        );
        if (observed == INVALID_HANDLE_VALUE) {
          throw worker_error(
            "conversion staging output identity changed"
          );
        }
        BY_HANDLE_FILE_INFORMATION information {};
        const bool valid =
          GetFileInformationByHandle(observed, &information) &&
          (information.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0 &&
          information.dwVolumeSerialNumber == volume_serial_ &&
          information.nFileIndexHigh == file_index_high_ &&
          information.nFileIndexLow == file_index_low_;
        CloseHandle(observed);
        if (!valid) {
          throw worker_error(
            "conversion staging output identity changed"
          );
        }
        return information;
      }

      void cleanup_owned() noexcept {
        if (handle_ == INVALID_HANDLE_VALUE) {
          return;
        }
        // DELETE access is intentionally not held: the packaged FFmpeg does not share
        // delete access when opening an existing output. Keeping a no-delete read-
        // attribute handle locks the pathname throughout conversion. On failure, close
        // and preserve the exact claimed file; never fall back to a racy path deletion.
        CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
      }

      fs::path path_;
      HANDLE handle_ = INVALID_HANDLE_VALUE;
      DWORD volume_serial_ = 0;
      DWORD file_index_high_ = 0;
      DWORD file_index_low_ = 0;
    };

    bool path_is_within(
      const fs::path &candidate,
      const fs::path &parent
    ) {
      const auto normalized_candidate = candidate.lexically_normal();
      const auto normalized_parent = parent.lexically_normal();
      auto candidate_part = normalized_candidate.begin();
      for (auto parent_part = normalized_parent.begin();
           parent_part != normalized_parent.end();
           ++parent_part, ++candidate_part) {
        if (candidate_part == normalized_candidate.end()) {
          return false;
        }
        if (lower(path_utf8(*candidate_part)) != lower(path_utf8(*parent_part))) {
          return false;
        }
      }
      return true;
    }

    std::uint64_t checked_byte_sum(
      const std::uint64_t left,
      const std::uint64_t right,
      const std::string_view description
    ) {
      if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw worker_error(std::string(description) + " byte count overflows");
      }
      return left + right;
    }


    void publish_progress(
      const worker_spec_t &spec,
      const std::string_view phase,
      const std::uint64_t processed_frames,
      const media_contract_t *media
    ) {
      nlohmann::json value {
        {"schema", 1},
        {"job_id", spec.job_id},
        {"phase", phase},
        {"processed_frames", processed_frames},
        {"total_frames", media ? nlohmann::json(media->frames.size()) : nlohmann::json(nullptr)},
        {"source_time_seconds", media && processed_frames > 0 ? nlohmann::json(media->time_base.seconds(nonnegative_tick_difference(media->frames.at(std::min<std::size_t>(static_cast<std::size_t>(processed_frames), media->frames.size()) - 1).pts, media->first_pts(), "worker progress presentation offset"))) : nlohmann::json(nullptr)},
        {"source_duration_seconds", media ? nlohmann::json(media->duration_seconds()) : nlohmann::json(nullptr)},
      };
      // This file is a live UI handoff, not restart state. The manager atomically reads it,
      // coalesces durable job.json updates, and never resumes an interrupted worker from it.
      write_json_atomic(
        spec.progress_path,
        value,
        publication_e::transient
      );
    }

    class worker_progress_publisher_t {
    public:
      explicit worker_progress_publisher_t(const worker_spec_t &spec):
          spec_(spec) {
      }

      worker_progress_publisher_t(const worker_progress_publisher_t &) = delete;
      worker_progress_publisher_t &operator=(const worker_progress_publisher_t &) = delete;

      bool publish(
        const std::string_view phase,
        const std::uint64_t processed_frames,
        const media_contract_t *media,
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()
      ) {
        const bool final_source_frame = media && processed_frames == media->frames.size();
        if (next_publication_ && phase == last_phase_ && !final_source_frame && now < *next_publication_) {
          return false;
        }
        // Allocate the next phase before writing, then commit eligibility only after the
        // actual publication succeeds. Suppressed updates never construct progress JSON.
        std::string published_phase {phase};
        publish_progress(spec_, phase, processed_frames, media);
        last_phase_ = std::move(published_phase);
        next_publication_ = now + 200ms;
        return true;
      }

    private:
      const worker_spec_t &spec_;
      std::string last_phase_;
      std::optional<std::chrono::steady_clock::time_point> next_publication_;
    };

    void publish_failure(
      const worker_spec_t &spec,
      const std::string &error
    ) noexcept {
      try {
        write_json_atomic(
          spec.result_path,
          wire::worker_failure_json(spec.job_id, error)
        );
      } catch (...) {
      }
    }
  }  // namespace

#ifdef SUNSHINE_TESTS
  struct worker_progress_test_publisher_t::impl_t {
    worker_spec_t spec;
    worker_progress_publisher_t publisher {spec};
    std::size_t successful_publications = 0;
  };

  worker_progress_test_publisher_t::worker_progress_test_publisher_t(
    std::filesystem::path progress_path,
    std::string job_id
  ):
      impl_(std::make_unique<impl_t>()) {
    impl_->spec.progress_path = std::move(progress_path);
    impl_->spec.job_id = std::move(job_id);
  }

  worker_progress_test_publisher_t::~worker_progress_test_publisher_t() = default;

  void worker_progress_test_publisher_t::publish(
    const std::string_view phase,
    const std::uint64_t processed_frames,
    const media_contract_t *media,
    const std::chrono::steady_clock::time_point now
  ) {
    if (impl_->publisher.publish(phase, processed_frames, media, now)) {
      ++impl_->successful_publications;
    }
  }

  std::size_t worker_progress_test_publisher_t::successful_publications() const {
    return impl_->successful_publications;
  }

  std::size_t offline_source_pipeline_capacity_for_test() {
    return offline_source_pipeline_capacity;
  }
#endif

  worker_spec_t parse_worker_spec(const nlohmann::json &value) {
    wire::worker_spec_contract_t contract;
    try {
      contract = wire::parse_worker_spec_contract(value);
    } catch (const wire::contract_error &exception) {
      throw worker_error(exception.what());
    }
    worker_spec_t spec;
    spec.job_id = contract.job_id;
    if (spec.job_id.size() > 128) {
      throw worker_error("worker job identity is too long");
    }
    spec.operation = contract.operation;
    const auto absolute_path = [](const std::string &text, const std::string_view name) {
      auto path = path_from_utf8(text);
      if (!path.is_absolute()) {
        throw worker_error(
          "worker contract path '" + std::string {name} + "' must be absolute"
        );
      }
      return path.lexically_normal();
    };
    spec.input_path = absolute_path(contract.input_path, "input_path");
    spec.job_directory = absolute_path(contract.job_directory, "job_directory");
    spec.result_directory = absolute_path(contract.result_directory, "result_directory");
    spec.progress_path = absolute_path(contract.progress_path, "progress_path");
    spec.result_path = absolute_path(contract.result_path, "result_path");
    if (!path_is_within(spec.result_directory, spec.job_directory) || !path_is_within(spec.progress_path, spec.job_directory) || !path_is_within(spec.result_path, spec.job_directory)) {
      throw worker_error(
        "worker state/result paths must remain inside the job directory"
      );
    }
    if (contract.staging_output) {
      spec.staging_output = absolute_path(*contract.staging_output, "staging_output");
    }
    spec.sunshine_executable = absolute_path(
      contract.sunshine_executable, "sunshine.executable"
    );
    spec.sunshine_config = absolute_path(contract.sunshine_config, "sunshine.config");
    spec.ffmpeg_executable = absolute_path(contract.ffmpeg_path, "ffmpeg.path");
    spec.ffprobe_executable = absolute_path(contract.ffprobe_path, "ffprobe.path");
    spec.codec = contract.codec;
    spec.transient_raster_hard_cap_bytes =
      contract.transient_raster_hard_cap_bytes;
    return spec;
  }

  worker_spec_t read_authenticated_worker_spec(
    const fs::path &path,
    const std::string_view expected_sha256
  ) {
    if (!wire::valid_sha256_hex(expected_sha256)) {
      throw worker_error(
        "worker specification SHA-256 must be 64 lowercase hexadecimal characters"
      );
    }
    const auto bytes = read_bounded_bytes(path, max_spec_bytes);
    const auto actual_sha256 = sha256_hex(bytes);
    if (!constant_time_equal(actual_sha256, expected_sha256)) {
      throw worker_error(
        "worker specification authentication failed; exact bytes changed after admission"
      );
    }
    try {
      auto spec = parse_worker_spec(wire::parse_json_without_duplicate_keys(bytes));
      spec.authenticated_spec_sha256 = actual_sha256;
      return spec;
    } catch (const worker_error &) {
      throw;
    } catch (const std::exception &exception) {
      throw worker_error(
        std::string {"cannot parse authenticated worker specification: "} +
        exception.what()
      );
    }
  }

  int run(const int argc, char **argv) {
    std::optional<worker_spec_t> parsed_spec;
    std::optional<staging_output_claim_t> staging_claim;
    try {
      if (
        argc != 2 || !argv ||
        !argv[0] || !*argv[0] ||
        !argv[1] || !*argv[1]
      ) {
        throw worker_error(
          "--offline-sbs-worker requires a worker-spec path and its SHA-256"
        );
      }
      const auto spec_path = fs::absolute(path_from_utf8(argv[0])).lexically_normal();
      parsed_spec = read_authenticated_worker_spec(spec_path, argv[1]);
      const auto &spec = *parsed_spec;
      worker_progress_publisher_t progress {spec};
      std::error_code ec;
      if (!fs::is_regular_file(spec.input_path, ec) || ec || fs::file_size(spec.input_path, ec) == 0 || ec) {
        throw worker_error("offline input is missing or empty");
      }
      for (const auto &tool : {
             spec.sunshine_executable,
             spec.sunshine_config,
             spec.ffmpeg_executable,
             spec.ffprobe_executable,
           }) {
        if (!fs::is_regular_file(tool, ec) || ec) {
          throw worker_error("trusted worker executable/config is missing");
        }
      }
      if (spec.staging_output) {
        staging_claim.emplace(*spec.staging_output);
      }
      fs::create_directories(spec.result_directory, ec);
      if (ec) {
        throw worker_error("cannot create worker result directory");
      }
      const auto work = spec.job_directory / "native-work";
      if (fs::exists(work, ec) && !fs::is_empty(work, ec)) {
        throw worker_error("native worker directory must be new or empty");
      }
      fs::create_directories(work / "logs", ec);
      if (ec) {
        throw worker_error("cannot create native worker directory");
      }

      progress.publish("probe", 0, nullptr);
      const auto probe_json = work / "source-probe.json";
      const auto media = probe_media(
        spec,
        spec.input_path,
        probe_json,
        work / "logs" / "source-probe.log",
        true
      );
      validate_output_time_base(spec, media);
      const auto observation_timeline = work / "source-observation-timeline.sbsotl";
      write_observation_timeline(observation_timeline, media);
      std::optional<stream_inventory_t> source_inventory;
      if (spec.operation == "convert") {
        source_inventory = probe_stream_inventory(
          spec,
          spec.input_path,
          work / "source-stream-inventory.json",
          work / "logs" / "source-stream-inventory.log"
        );
        validate_stream_preflight(spec, *source_inventory);
        validate_windows_command_line_capacity(
          build_mux_command(
            spec,
            media,
            *source_inventory,
            work / "encoded-video.mp4"
          ),
          "final mux command; reduce source metadata"
        );
      }
      write_json_atomic(
        spec.result_directory / "source-contract.json",
        media_contract_json(media)
      );

      // The same build's causal estimator/renderer pass runs in this process. Offline conversion
      // removes wall-clock pacing; it does not replace the online camera, OCR, conditioning, or
      // renderer state machines with an offline replay implementation.
      const auto analysis_input = work / "analysis-input";
      // These directories retain source identity and the causal pass's bounded diagnostics.
      // Per-frame pixels, completion evidence and the pass's whole-clip contract travel only
      // through owned memory.
      // The manager's identity-pinned cleanup still owns all transient native-work artifacts.
      const auto analysis_output = work / "analysis-output";
      fs::create_directories(analysis_input, ec);
      fs::create_directories(analysis_output, ec);
      if (ec) {
        throw worker_error("cannot create analysis directories");
      }
      const auto source_raster_bytes = std::uint64_t(media.width) * media.height * (media.color == media_color_e::sdr ? 4 : 12);
      const auto source_raster_bound = raw_source_byte_bound(source_raster_bytes, media.color != media_color_e::sdr);
      if (source_raster_bound > spec.transient_raster_hard_cap_bytes)
        throw worker_error("raw source, HDR upload and decoder pipe exceed the transient raster hard cap before allocation");
      // Own the measurement window before either raw transport or the GPU thread starts.
      // Transfer timings are collected, and summarised in the log, only with diagnostics on.
      sbs_perf::set_enabled(config::sunshine.diagnostics_enabled);
      sbs_perf::reset();
      streaming_decoder_t analysis_decoder {
        spec,
        media,
        work / "logs" / "analysis-decoder.log",
      };
      const auto expected_source_format = media.color == media_color_e::sdr ? raw_pixel_format::bgra8 : raw_pixel_format::gbrpf32le;
      native_pass_settings analysis_settings {
        .source_identity_directory = path_utf8(analysis_input),
        .transient_output_directory = path_utf8(analysis_output),
        .observation_timeline = path_utf8(observation_timeline),
        .source_format = expected_source_format,
        .frame_count = media.frames.size(),
        .operation = spec.operation == "convert" ? native_pass_operation::convert : native_pass_operation::analyze,
        .transient_raster_hard_cap_bytes = spec.transient_raster_hard_cap_bytes,
      };
      raw_frame_transport transport(spec.transient_raster_hard_cap_bytes);
      analysis_process_t analysis(std::move(analysis_settings), transport);
      completion_header_t analysis_header;
      std::optional<completion_terminal_t> analysis_terminal;
      std::uint64_t peak_live_raster_bytes = 0;
      std::optional<std::uint64_t> analysis_source_raster_bytes;
      std::uint32_t sbs_width = 0;
      std::uint32_t sbs_height = 0;
      bool conversion_overlap_enabled = false;
      std::optional<std::uint64_t> conversion_single_slot_raster_bound;
      std::optional<std::uint64_t> conversion_overlap_raster_bound;
      std::optional<std::uint64_t> conversion_sbs_artifact_bound;
      std::unique_ptr<whole_clip_encoder_t> conversion_encoder;
      nlohmann::json timeline_contract =
        spec.operation != "convert" ?
          nlohmann::json {
            {"mode", "evaluation-only"},
            {"max_output_ticks", nullptr},
          } :
          (output_container(*spec.staging_output) == "mp4" ?
             nlohmann::json {
               {"mode", "exact-rational"},
               {"max_output_ticks", 0},
               {"container", "mp4"},
             } :
             nlohmann::json {
               {"mode", "bounded-output-timebase"},
               {"max_output_ticks", 1},
               {"container", "matroska"},
               {"absolute_timestamps", true},
               {"cumulative_drift_allowed", false},
             });

      try {
        progress.publish(
          spec.operation == "convert" ? "convert" : "evaluate",
          0,
          &media
        );
        struct queued_source_t {
          const frame_timing_t *timing = nullptr;
          std::uint64_t bytes = 0;
        };
        std::deque<queued_source_t> source_queue;
        std::size_t next_source_index = 0;
        std::size_t source_window_capacity = 1;
        std::uint64_t live_source_raster_bytes = 0;
        const auto fill_source_window = [&] {
          while (source_queue.size() < source_window_capacity &&
                 next_source_index < media.frames.size()) {
            const auto &timing = media.frames[next_source_index];
            auto source = analysis_decoder.read_next(timing);
            try {
              const auto bytes = source.bytes();
              if (!analysis_source_raster_bytes) {
                analysis_source_raster_bytes = bytes;
              } else if (*analysis_source_raster_bytes != bytes) {
                throw worker_error(
                  "fixed-resolution analysis source raster size changed mid-clip"
                );
              }
              live_source_raster_bytes = checked_byte_sum(
                live_source_raster_bytes,
                bytes,
                "live source pipeline"
              );
              if (live_source_raster_bytes >
                  spec.transient_raster_hard_cap_bytes) {
                throw worker_error(
                  "live source pipeline exceeds the transient raster hard cap"
                );
              }
              source_queue.push_back({
                .timing = &timing,
                .bytes = bytes,
              });
              transport.source.publish(std::move(source));
              ++next_source_index;
            } catch (...) {
              transport.source.cancel();
              transport.sbs.cancel();
              throw;
            }
          }
        };

        fill_source_window();
        const auto completion_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout);
        analysis_header = transport.completion.receive_header(completion_timeout);
        analysis_header.validate();
        if (analysis_header.source_format != expected_source_format ||
            analysis_header.source_frame_count != media.frames.size() ||
            analysis_header.first_sequence != 1 ||
            analysis_header.sbs_enabled != (spec.operation == "convert")) {
          throw worker_error("native completion header differs from the exact source/operation");
        }
        while (!source_queue.empty()) {
          const auto queued_source = source_queue.front();
          const auto &timing = *queued_source.timing;
          const auto current_source_raster_bytes = queued_source.bytes;
          const auto completion = transport.completion.receive_frame(timing.sequence, completion_timeout);
          completion.validate();
          const bool converting = spec.operation == "convert";
          const auto &completed_source = completion.source;
          if (
            completed_source.schema != 1 || completed_source.sequence != timing.sequence ||
            completed_source.pts != timing.pts ||
            completed_source.time_base_numerator != media.time_base.numerator ||
            completed_source.time_base_denominator != media.time_base.denominator ||
            completed_source.width != media.width || completed_source.height != media.height ||
            completed_source.format != expected_source_format ||
            completion.sbs.has_value() != converting
          ) {
            throw worker_error("native completion differs from the exact source timeline/raster/operation");
          }

          std::uint64_t current_live_raster_bytes = source_raster_bound;
          if (converting) {
            const auto &completed_sbs = *completion.sbs;
            const auto completed_sbs_width = completed_sbs.width;
            const auto completed_sbs_height = completed_sbs.height;
            if (completed_sbs_width == 0 || completed_sbs_height == 0 || completed_sbs.format != (media.color == media_color_e::sdr ? raw_pixel_format::bgra8 : raw_pixel_format::rgba16f)) {
              throw worker_error(
                "conversion completion lacks its direct SBS geometry/format"
              );
            }
            if (timing.sequence == 1) {
              sbs_width = completed_sbs_width;
              sbs_height = completed_sbs_height;
            } else if (sbs_width != completed_sbs_width ||
                       sbs_height != completed_sbs_height) {
              throw worker_error(
                "direct SBS geometry changed during conversion"
              );
            }
            auto sbs = transport.sbs.receive(timing.sequence);
            if (sbs.header.schema != 1 || sbs.header.sequence != timing.sequence || sbs.header.width != sbs_width || sbs.header.height != sbs_height || sbs.header.pts != timing.pts || sbs.header.time_base_numerator != media.time_base.numerator || sbs.header.time_base_denominator != media.time_base.denominator || sbs.header.format != (media.color == media_color_e::sdr ? raw_pixel_format::bgra8 : raw_pixel_format::rgba16f)) {
              throw worker_error("raw SBS output header differs from the exact source timeline/geometry/color");
            }
            const auto raw_sbs_size = sbs.bytes();
            if (!conversion_encoder) {
              conversion_sbs_artifact_bound = std::uint64_t(sbs_width) * sbs_height *
                (media.color == media_color_e::sdr ? 4u : 8u);
              if (raw_sbs_size > *conversion_sbs_artifact_bound) {
                throw worker_error(
                  "direct SBS artifact exceeded its conservative pipeline bound"
                );
              }
              conversion_single_slot_raster_bound =
                raw_raster_byte_bound(*analysis_source_raster_bytes, sbs_width, sbs_height,
                                      media.color != media_color_e::sdr, false);
              if (*conversion_single_slot_raster_bound >
                  spec.transient_raster_hard_cap_bytes) {
                throw worker_error(
                  "single-slot conversion pipeline exceeds the transient raster hard cap"
                );
              }
              conversion_overlap_raster_bound =
                raw_raster_byte_bound(*analysis_source_raster_bytes, sbs_width, sbs_height,
                                      media.color != media_color_e::sdr, true);
              conversion_overlap_enabled =
                media.frames.size() > 1u &&
                *conversion_overlap_raster_bound <=
                  spec.transient_raster_hard_cap_bytes;
              source_window_capacity = conversion_overlap_enabled ?
                offline_source_pipeline_capacity : 1u;
              conversion_encoder = std::make_unique<whole_clip_encoder_t>(
                spec,
                media,
                work,
                sbs_width,
                sbs_height,
                conversion_overlap_enabled
              );
            }
            if (!conversion_sbs_artifact_bound ||
                raw_sbs_size > *conversion_sbs_artifact_bound) {
              throw worker_error(
                "direct SBS artifact exceeded its conservative pipeline bound"
              );
            }
            current_live_raster_bytes = conversion_overlap_enabled ?
              *conversion_overlap_raster_bound :
              *conversion_single_slot_raster_bound;
            if (current_live_raster_bytes >
                spec.transient_raster_hard_cap_bytes) {
              throw worker_error(
                "single-slot/overlapped conversion pipeline exceeded the transient raster hard cap"
              );
            }
            try {
              conversion_encoder->publish(timing.sequence, std::move(sbs));
            } catch (...) {
              transport.source.cancel();
              transport.sbs.cancel();
              throw;
            }
          }
          peak_live_raster_bytes = std::max(
            peak_live_raster_bytes,
            current_live_raster_bytes
          );
          transport.source.acknowledge(timing.sequence);
          if (live_source_raster_bytes < current_source_raster_bytes) {
            throw worker_error("live source pipeline accounting underflow");
          }
          live_source_raster_bytes -= current_source_raster_bytes;
          source_queue.pop_front();
          fill_source_window();
          progress.publish(
            converting ? "convert" : "evaluate",
            timing.sequence,
            &media
          );
        }
        transport.source.finish();
        // The native pass waits for the final source ACK before publishing its terminal.
        // Record that status before terminate() performs full transport cancellation.
        analysis_terminal = transport.completion.wait_finished(completion_timeout);
        const int analysis_code = analysis.wait(
          std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
        );
        if (analysis_code != analysis_terminal->exit_code || analysis_code != 0) {
          throw worker_error(
            "native analysis exited " + std::to_string(analysis_code) +
            " (terminal " + std::to_string(analysis_terminal->exit_code) + "): " + analysis_terminal->error
          );
        }
        if (analysis_terminal->source_frame_count != media.frames.size() || analysis_terminal->adaptive_frame_count != media.frames.size() || analysis_terminal->sbs_frame_count != (spec.operation == "convert" ? media.frames.size() : 0)) {
          throw worker_error("native analysis terminal counts differ from the exact source/operation");
        }
        analysis.terminate();
        analysis_decoder.finish();
      } catch (const std::exception &exception) {
        if (!analysis_terminal) {
          // Ordinary runner failure publishes first, then cancels only its pixel channels.
          // Worker-initiated abort or an exceptional unnotified exit may have no terminal yet.
          try {
            analysis_terminal = transport.completion.wait_finished(0ms);
          } catch (...) {
          }
        }
        analysis.terminate();
        analysis_decoder.abort();
        if (conversion_encoder) {
          conversion_encoder->abort(exception.what());
        }
        throw;
      }

      // The in-process pass validated this contract before handing it over in memory.
      const auto &typed_analysis_contract = analysis.contract();
      const bool converting = spec.operation == "convert";
      if (typed_analysis_contract.artifact_mode !=
            (converting ? "conversion" : "adaptive") ||
          typed_analysis_contract.source_frame_count != media.frames.size() ||
          typed_analysis_contract.source_first_sequence != 1 ||
          typed_analysis_contract.inference_mode != "single-pass-tensorrt" ||
          !typed_analysis_contract.depth_inference_enabled ||
          typed_analysis_contract.scheduled_depth_update_count != media.frames.size() ||
          typed_analysis_contract.tensorrt_enqueue_count != media.frames.size()) {
        throw worker_error(
          "causal pass did not attest exactly one TensorRT enqueue per source frame"
        );
      }
      // Offline conversion runs the production pipeline directly: the harness must attest the
      // raw-memory causal pass this worker drove.
      const auto &analysis_runtime = typed_analysis_contract.resolved_runtime;
      const auto expected_input_frame_format =
        media.color == media_color_e::sdr ?
          std::string_view {"sRGB-BGRA8-raw"} :
          std::string_view {"linear-scRGB-f32-planar"};
      if (
        analysis_runtime.frame_transport != raw_frame_transport::name ||
        analysis_runtime.input_frame_format != expected_input_frame_format
      ) {
        throw worker_error("analysis did not attest the native raw-memory causal pass");
      }
      const auto &direct_sbs = typed_analysis_contract.sbs;
      if (converting) {
        if (!direct_sbs.enabled ||
            direct_sbs.frame_count != media.frames.size() ||
            direct_sbs.width != sbs_width ||
            direct_sbs.height != sbs_height ||
            direct_sbs.width != analysis_runtime.output_sbs_width ||
            direct_sbs.height != analysis_runtime.output_sbs_height) {
          throw worker_error(
            "causal conversion did not attest its complete direct SBS stream"
          );
        }
      } else if (direct_sbs.enabled || direct_sbs.frame_count != 0 || direct_sbs.width != 0 || direct_sbs.height != 0) {
        throw worker_error("analysis-only pass unexpectedly attested SBS output");
      }
      if (
        analysis_header.model != analysis_runtime.model ||
        analysis_header.model != typed_analysis_contract.model ||
        analysis_header.depth_reuse_interval != analysis_runtime.depth_reuse_interval ||
        analysis_header.pop_strength != analysis_runtime.pop_strength
      ) {
        throw worker_error(
          "completion header config disagrees with the authenticated analysis runtime"
        );
      }
      const auto &analysis_state = typed_analysis_contract.adaptive_state;
      if (
        analysis_state.transport != completion_channel_t::name ||
        analysis_state.retained_history ||
        analysis_state.schema != sbs_adaptive_state::schema_version ||
        analysis_state.capture != sbs_adaptive_state::capture ||
        analysis_state.frame_count != media.frames.size()
      ) {
        throw worker_error(
          "analysis did not attest its bounded adaptive-state transport"
        );
      }
      if (spec.operation == "convert") {
        if (!conversion_encoder) {
          throw worker_error("conversion produced no whole-clip encoder");
        }
        conversion_encoder->finish();
        progress.publish(
          "mux",
          media.frames.size(),
          &media
        );
        // The stream-copied final output is the gate for codec, raster, color, side data, and
        // timeline. Only Matroska's millisecond timestamps hide an inexact encoder timeline.
        const bool matroska_output =
          output_container(*spec.staging_output) == "matroska";
        if (matroska_output) {
          probe_encoded_packet_timeline(
            spec,
            media,
            conversion_encoder->encoded_video(),
            work / "logs" / "encoded-video-packets.log"
          );
        }
        mux_encoded_video(
          spec,
          media,
          *source_inventory,
          conversion_encoder->encoded_video(),
          work / "logs" / "final-mux.log"
        );
        {
          const auto output_media = probe_media(
            spec,
            *spec.staging_output,
            work / "output-probe.json",
            work / "logs" / "output-probe.log",
            false
          );
          if (matroska_output) {
            require_millisecond_or_finer(
              output_media.time_base,
              "output video"
            );
          }
          validate_timeline_contract(
            media,
            output_media,
            matroska_output ? 1 : 0
          );
          timeline_contract["observed_video"] =
            timeline_error_report(media, output_media);
          if (output_media.codec_name != codec_name_for_request(spec.codec) || output_media.width != sbs_width || output_media.height != sbs_height) {
            throw worker_error("final output codec/raster contract mismatch");
          }
          if (media.color != media_color_e::sdr) {
            validate_hdr_equivalence(media, output_media);
          } else {
            validate_sdr_color_equivalence(media, output_media);
          }
          write_json_atomic(
            spec.result_directory / "output-contract.json",
            media_contract_json(output_media)
          );
        }
        const auto output_inventory = probe_stream_inventory(
          spec,
          *spec.staging_output,
          work / "output-stream-inventory.json",
          work / "logs" / "output-stream-inventory.log"
        );
        timeline_contract["observed_auxiliary"] =
          validate_stream_inventory(
            *source_inventory,
            output_inventory,
            matroska_output ? 1 : 0
          );
        write_json_atomic(
          spec.result_directory / "timeline-contract.json",
          timeline_contract
        );
      }

      progress.publish(
        "complete",
        media.frames.size(),
        &media
      );
      const auto staging_identity = staging_claim ?
                                      staging_claim->identity_json() :
                                      nlohmann::json(nullptr);
      wire::worker_result_contract_t result_contract {
        .job_id = spec.job_id,
        .operation = spec.operation,
        .codec = spec.codec,
        .worker_spec_sha256 = spec.authenticated_spec_sha256,
        .source = {
          .width = media.width,
          .height = media.height,
          .frame_count = media.frames.size(),
          .duration_seconds = media.duration_seconds(),
          .variable_frame_rate = media.variable_frame_rate(),
          .color_mode = media.color == media_color_e::sdr ?
                          "sdr" :
                          (media.color == media_color_e::hdr_pq ? "hdr-pq" : "hdr-hlg"),
          .contract_path = path_utf8(spec.result_directory / "source-contract.json"),
        },
        .analysis_contract = typed_analysis_contract,
        .cache = {
          .hard_cap_bytes = spec.transient_raster_hard_cap_bytes,
          .analysis_source_raster_bytes = analysis_source_raster_bytes.value_or(0),
          .peak_live_raster_bytes = peak_live_raster_bytes,
        },
        .timeline_contract = timeline_contract,
        .staging_identity = staging_identity.is_null() ?
                              std::nullopt :
                              std::optional<nlohmann::json> {staging_identity},
        .output_path = spec.staging_output ?
                         std::optional<std::string> {path_utf8(*spec.staging_output)} :
                         std::nullopt,
      };
      const auto result = wire::to_json(result_contract);
      write_json_atomic_bounded(
        spec.result_path,
        result,
        worker_result_max_bytes,
        "worker result"
      );
      // The manager retains the job-root identity handle and removes native-work
      // only after this process (and its process group) has been reaped. A child
      // cannot safely re-resolve or delete that user-writable pathname, and on
      // Windows such an attempt also conflicts with the manager's delete pin.
      if (staging_claim) {
        staging_claim->release_for_publish();
      }
      return 0;
    } catch (const std::exception &exception) {
      if (parsed_spec) {
        publish_failure(*parsed_spec, exception.what());
      }
      return 2;
    }
  }
}  // namespace offline_sbs

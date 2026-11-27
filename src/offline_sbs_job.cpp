/**
 * @file src/offline_sbs_job.cpp
 * @brief Native offline SBS job lifecycle, persistence, and publication.
 */

#include "offline_sbs_job.h"
#include "offline_sbs_service_owner.h"
#ifdef SUNSHINE_TESTS
#include "offline_sbs_service_test.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cwchar>
#include <cwctype>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "crypto.h"
#include "entry_handler.h"
#include "gpu_workload_arbiter.h"
#include "offline_sbs_contract.h"
#include "offline_sbs_digest.h"
#include "offline_sbs_filesystem.h"
#include "offline_sbs_job_files.h"
#include "offline_sbs_job_worker.h"
#include "offline_sbs_wire_contract.h"
#include "platform/common.h"
#include "uuid.h"

#include <boost/filesystem/path.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/process/v1/group.hpp>

#include "logging.h"

  #include <fcntl.h>
  #include <io.h>
  #include <windows.h>

using namespace std::literals;

namespace offline_sbs {
  namespace fs = std::filesystem;
  namespace bp = boost::process::v1;
  using namespace job_detail;

  namespace {
    constexpr std::uint64_t transient_raster_hard_cap_bytes =
      4ull * 1024ull * 1024ull * 1024ull;

    // The operation's spelling is shared with the worker specification and result.
    using job_detail::to_string;

    const char *to_string(output_location_e value) {
      switch (value) {
        case output_location_e::input_directory:
          return "input-directory";
      }
      return "unknown";
    }

    const char *to_string(job_state_e value) {
      switch (value) {
        case job_state_e::queued:
          return "queued";
        case job_state_e::running:
          return "running";
        case job_state_e::publishing:
          return "publishing";
        case job_state_e::canceling:
          return "canceling";
        case job_state_e::canceled:
          return "canceled";
        case job_state_e::complete:
          return "complete";
        case job_state_e::failed:
          return "failed";
        case job_state_e::interrupted:
          return "interrupted";
      }
      return "unknown";
    }

    const char *to_string(error_code_e value) {
      switch (value) {
        case error_code_e::none:
          return "none";
        case error_code_e::not_initialized:
          return "not_initialized";
        case error_code_e::invalid_request:
          return "invalid_request";
        case error_code_e::busy:
          return "busy";
        case error_code_e::not_found:
          return "not_found";
        case error_code_e::conflict:
          return "conflict";
        case error_code_e::unavailable:
          return "unavailable";
        case error_code_e::io_error:
          return "io_error";
      }
      return "unknown";
    }

    std::optional<operation_e> parse_operation(const std::string_view value) {
      if (value == "evaluate") {
        return operation_e::evaluate;
      }
      if (value == "convert") {
        return operation_e::convert;
      }
      return std::nullopt;
    }

    std::optional<output_location_e> parse_output_location(
      const std::string_view value
    ) {
      if (value == "input-directory") {
        return output_location_e::input_directory;
      }
      return std::nullopt;
    }

    std::optional<job_state_e> parse_job_state(const std::string_view value) {
      constexpr std::array values {
        job_state_e::queued,
        job_state_e::running,
        job_state_e::publishing,
        job_state_e::canceling,
        job_state_e::canceled,
        job_state_e::complete,
        job_state_e::failed,
        job_state_e::interrupted,
      };
      for (const auto candidate : values) {
        if (value == to_string(candidate)) {
          return candidate;
        }
      }
      return std::nullopt;
    }

    bool is_terminal(const job_state_e state) {
      return state == job_state_e::canceled ||
             state == job_state_e::complete ||
             state == job_state_e::failed ||
             state == job_state_e::interrupted;
    }

    std::int64_t unix_time_ms() {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()
      ).count();
    }

    bool valid_job_id(const std::string_view id) {
      static const std::regex expression {
        R"(^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$)"
      };
      return std::regex_match(id.begin(), id.end(), expression);
    }

    bool is_managed_staging_filename(const fs::path &filename) {
      auto name = path_to_utf8(filename.filename());
      auto folded = name;
      std::ranges::transform(folded, folded.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      constexpr auto marker = ".sunshine3d-"sv;
      const auto marker_at = folded.rfind(marker);
      if (marker_at == std::string::npos || marker_at == 0) {
        return false;
      }
      const auto id_at = marker_at + marker.size();
      constexpr std::size_t job_id_size = 36;
      if (id_at + job_id_size > name.size()) {
        return false;
      }
      auto job_id = name.substr(id_at, job_id_size);
      std::ranges::transform(job_id, job_id.begin(), [](const unsigned char c) {
        return static_cast<char>(std::toupper(c));
      });
      if (!valid_job_id(job_id)) {
        return false;
      }
      const auto suffix = folded.substr(id_at + job_id_size);
      return suffix == ".part.mkv" || suffix == ".part.mp4";
    }

    bool same_path_component(
      const fs::path &left,
      const fs::path &right
    ) {
      auto left_native = left.native();
      auto right_native = right.native();
      std::ranges::transform(left_native, left_native.begin(), [](const wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
      });
      std::ranges::transform(right_native, right_native.begin(), [](const wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
      });
      return left_native == right_native;
    }

    bool path_is_same_or_ancestor(
      const fs::path &candidate,
      const fs::path &path
    ) {
      auto candidate_part = candidate.begin();
      auto path_part = path.begin();
      for (
        ;
        candidate_part != candidate.end() && path_part != path.end();
        ++candidate_part, ++path_part
      ) {
        if (!same_path_component(*candidate_part, *path_part)) {
          return false;
        }
      }
      return candidate_part == candidate.end();
    }

    bool managed_roots_overlap(
      const fs::path &left,
      const fs::path &right
    ) {
      return path_is_same_or_ancestor(left, right) ||
             path_is_same_or_ancestor(right, left);
    }

    bool valid_output_name(const std::string &name) {
      if (
        name.empty() || name.size() > 180 ||
        name.back() == ' ' || name.back() == '.'
      ) {
        return false;
      }
      const fs::path path = path_from_utf8(name);
      if (
        path.has_parent_path() || path.filename() != path ||
        name == "." || name == ".." ||
        is_managed_staging_filename(path)
      ) {
        return false;
      }
      if (std::any_of(name.begin(), name.end(), [](const unsigned char c) {
            return c < 0x20 || c == '<' || c == '>' || c == ':' ||
                   c == '"' || c == '/' || c == '\\' || c == '|' ||
                   c == '?' || c == '*';
          })) {
        return false;
      }

      // Win32 device names reserve the component before the first dot, regardless of later
      // extensions (for example, CON.preview.mkv is still the CON device).
      auto device_component = name.substr(0, name.find('.'));
      std::transform(
        device_component.begin(),
        device_component.end(),
        device_component.begin(),
        [](const unsigned char c) {
        return static_cast<char>(std::toupper(c));
      });
      static const std::array reserved {
        "CON"sv, "PRN"sv, "AUX"sv, "NUL"sv, "CLOCK$"sv,
        "CONIN$"sv, "CONOUT$"sv,
        "COM1"sv, "COM2"sv, "COM3"sv, "COM4"sv, "COM5"sv,
        "COM6"sv, "COM7"sv, "COM8"sv, "COM9"sv,
        "LPT1"sv, "LPT2"sv, "LPT3"sv, "LPT4"sv, "LPT5"sv,
        "LPT6"sv, "LPT7"sv, "LPT8"sv, "LPT9"sv,
      };
      if (
        std::find(
          reserved.begin(),
          reserved.end(),
          device_component
        ) != reserved.end()
      ) {
        return false;
      }
      auto extension = path.extension().string();
      std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](const unsigned char c) {
          return static_cast<char>(std::tolower(c));
        }
      );
      return extension == ".mkv" || extension == ".mp4";
    }

    bool valid_codec(const std::string_view codec) {
      return codec == "hevc_nvenc" ||
             codec == "av1_nvenc";
    }

    enum class staging_cleanup_e {
      pending,
      complete,
    };

    // Publishing the output and retiring its staging hard link are separate outcomes.
    struct publication_outcome_t {
      staging_cleanup_e staging_cleanup = staging_cleanup_e::pending;
    };

    enum class publication_origin_e {
      worker,
      recovery,
    };

    class publication_file_t {
    public:
      ~publication_file_t() {
        if (handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(handle_);
        }
      }

      publication_file_t(const publication_file_t &) = delete;
      publication_file_t &operator=(const publication_file_t &) = delete;

      publication_file_t(publication_file_t &&other) noexcept:
          source_(std::move(other.source_)),
          identity_(std::move(other.identity_))
          , handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE))
      {
      }

      publication_file_t &operator=(publication_file_t &&other) noexcept {
        if (this != &other) {
          if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
          }
          handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
          source_ = std::move(other.source_);
          identity_ = std::move(other.identity_);
        }
        return *this;
      }

      [[nodiscard]] const nlohmann::json &identity() const {
        return identity_;
      }

      static std::optional<publication_file_t> open(
        const fs::path &source,
        const std::optional<nlohmann::json> &expected_identity,
        std::string &error,
        bool *missing = nullptr
      ) {
        if (missing) {
          *missing = false;
        }
        const HANDLE handle = CreateFileW(
          source.c_str(),
          FILE_READ_ATTRIBUTES | DELETE,
          FILE_SHARE_READ,
          nullptr,
          OPEN_EXISTING,
          FILE_FLAG_OPEN_REPARSE_POINT,
          nullptr
        );
        if (handle == INVALID_HANDLE_VALUE) {
          const auto open_error = GetLastError();
          if (
            missing &&
            (
              open_error == ERROR_FILE_NOT_FOUND ||
              open_error == ERROR_PATH_NOT_FOUND
            )
          ) {
            *missing = true;
          }
          error =
            "cannot pin the staged output for publication (Windows error " +
            std::to_string(open_error) + ")";
          return std::nullopt;
        }
        BY_HANDLE_FILE_INFORMATION information {};
        const BOOL inspected = GetFileInformationByHandle(handle, &information);
        const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
        const auto size =
          (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
          information.nFileSizeLow;
        if (
          !inspected ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
          size == 0
        ) {
          CloseHandle(handle);
          error =
            "staged output is not a non-empty, non-reparse regular file "
            "(Windows error " +
            std::to_string(inspect_error) + ")";
          return std::nullopt;
        }
        nlohmann::json identity {
          {"volume_serial", information.dwVolumeSerialNumber},
          {"file_index_high", information.nFileIndexHigh},
          {"file_index_low", information.nFileIndexLow},
          {"size_bytes", size},
        };
        if (expected_identity && identity != *expected_identity) {
          CloseHandle(handle);
          error = "staged output identity changed after worker attestation";
          return std::nullopt;
        }
        return publication_file_t {
          source,
          std::move(identity),
          handle
        };
      }

      std::optional<publication_outcome_t> publish_no_replace(
        const fs::path &destination,
        std::string &error,
        const bool force_staging_disposition_failure = false
      ) {
        // The legacy Windows rename-by-handle API can replace an existing leaf
        // despite FILE_RENAME_INFO::ReplaceIfExists being false on supported
        // filesystems. Publish as a hard link instead: CreateHardLinkW is
        // atomically no-replace, while this open handle denies write/delete
        // sharing so source_ cannot be swapped between attestation and linking.
        if (!CreateHardLinkW(
              destination.c_str(),
              source_.c_str(),
              nullptr
            )) {
          const auto publish_error = GetLastError();
          if (
            publish_error == ERROR_FILE_EXISTS ||
            publish_error == ERROR_ALREADY_EXISTS
          ) {
            error =
              "final output appeared while the job was running; refusing to "
              "overwrite it";
          } else {
            error =
              "cannot atomically publish the pinned conversion output "
              "(Windows error " +
              std::to_string(publish_error) + ")";
          }
          return std::nullopt;
        }
        std::string retirement_error;
        const bool retired = retire_staging_link(
          retirement_error,
          force_staging_disposition_failure
        );
        if (!retired) {
          // The exact output is already safely published. Preserve success and
          // retain the attested staging hard link for identity-checked restart
          // recovery rather than re-resolving and deleting a user-writable
          // pathname.
          BOOST_LOG(warning)
            << "Published offline SBS output but could not retire its exact "
               "staging link: "sv
            << retirement_error;
        }
        return publication_outcome_t {
          .staging_cleanup = retired ? staging_cleanup_e::complete : staging_cleanup_e::pending,
        };
      }

      bool retire_staging_link(
        std::string &error,
        const bool force_failure = false
      ) {
        if (force_failure) {
          error = "injected staging disposition failure";
          return false;
        }
        FILE_DISPOSITION_INFO disposition {
          .DeleteFile = TRUE,
        };
        if (
          !SetFileInformationByHandle(
            handle_,
            FileDispositionInfo,
            &disposition,
            sizeof(disposition)
          )
        ) {
          error =
            "Windows error " + std::to_string(GetLastError());
          return false;
        }
        return true;
      }

    private:
      publication_file_t(
        fs::path source,
        nlohmann::json identity,
        const HANDLE handle
      ):
          source_(std::move(source)),
          identity_(std::move(identity)),
          handle_(handle) {
      }

      fs::path source_;
      nlohmann::json identity_;
      HANDLE handle_ = INVALID_HANDLE_VALUE;
    };

    fs::path staging_path_for(
      const fs::path &final_output,
      const std::string &job_id
    ) {
      return final_output.parent_path() /
             (
               final_output.stem().wstring() +
               L".sunshine3d-" +
               fs::path(path_from_utf8(job_id)).wstring() +
               L".part" +
               final_output.extension().wstring()
             );
    }

    // The protected worker summary remains the sole publication state. This view
    // centralizes its transitions without eagerly validating or normalizing retained
    // contracts; file identity is checked only at the existing filesystem proof points.
    class publication_transaction_t {
    public:
      explicit publication_transaction_t(job_snapshot_t &snapshot):
          snapshot_(snapshot) {
      }

      bool published() const {
        return snapshot_.worker_result.is_object() &&
               snapshot_.worker_result.value("published", false);
      }

      bool cleanup_pending() const {
        return snapshot_.worker_result.is_object() &&
               snapshot_.worker_result.value("staging_cleanup_pending", false);
      }

      const nlohmann::json &identity() const {
        return snapshot_.worker_result.at("publish_identity");
      }

      void prepare(nlohmann::json worker_summary, const nlohmann::json &identity) {
        snapshot_.state = job_state_e::publishing;
        snapshot_.progress.phase = "publishing";
        snapshot_.error.clear();
        snapshot_.worker_result = std::move(worker_summary);
        snapshot_.worker_result["publish_identity"] = identity;
      }

      void complete(const fs::path &output, const publication_outcome_t outcome) {
        snapshot_.worker_result["output"] = path_to_utf8(output);
        snapshot_.worker_result["published"] = true;
        set_staging_cleanup(outcome.staging_cleanup);
        snapshot_.state = job_state_e::complete;
        snapshot_.progress.phase = "complete";
        snapshot_.error.clear();
      }

      void set_staging_cleanup(const staging_cleanup_e cleanup) {
        snapshot_.worker_result["staging_cleanup_pending"] =
          cleanup == staging_cleanup_e::pending;
      }

    private:
      job_snapshot_t &snapshot_;
    };
  }  // namespace

  namespace {
    wire::job_snapshot_contract_t job_snapshot_wire_contract(
      const job_snapshot_t &snapshot
    ) {
      return {
      .id = snapshot.id,
      .state = to_string(snapshot.state),
      .operation = to_string(snapshot.operation),
      .input_path = path_to_utf8(snapshot.input_path),
      .output_path = snapshot.output_path ?
                       std::optional<std::string> {
                         path_to_utf8(*snapshot.output_path)
                       } :
                       std::nullopt,
      .output_location = snapshot.output_location ?
                           std::optional<std::string> {
                             to_string(*snapshot.output_location)
                           } :
                           std::nullopt,
      .codec = snapshot.codec,
      .transient_raster_hard_cap_bytes = snapshot.transient_raster_hard_cap_bytes,
      .progress = {
        .phase = snapshot.progress.phase,
        .processed_frames = snapshot.progress.processed_frames,
        .total_frames = snapshot.progress.total_frames,
        .source_time_seconds = snapshot.progress.source_time_seconds,
        .source_duration_seconds = snapshot.progress.source_duration_seconds,
      },
      .created_at_unix_ms = snapshot.created_at_unix_ms,
      .started_at_unix_ms = snapshot.started_at_unix_ms,
      .ended_at_unix_ms = snapshot.ended_at_unix_ms,
      .error = snapshot.error.empty() ?
                 std::nullopt : std::optional<std::string> {snapshot.error},
      .worker_result = snapshot.worker_result.is_null() ?
                         std::nullopt :
                         std::optional<nlohmann::json> {snapshot.worker_result},
      };
    }
  }  // namespace

  nlohmann::json job_snapshot_t::json() const {
    return wire::to_json(job_snapshot_wire_contract(*this));
  }

  nlohmann::json service_reply_t::json() const {
    return {
      {"status", ok},
      {"error_code", to_string(code)},
      {"error", error.empty() ? nlohmann::json(nullptr) : nlohmann::json(error)},
      {"job", job ? job->json() : nlohmann::json(nullptr)},
    };
  }

  nlohmann::json browse_reply_t::json() const {
    return {
      {"status", ok},
      {"error_code", to_string(code)},
      {"error", error.empty() ? nlohmann::json(nullptr) : nlohmann::json(error)},
      {"path", path ? nlohmann::json(path_to_utf8(*path)) : nlohmann::json(nullptr)},
      {"parent", parent ? nlohmann::json(path_to_utf8(*parent)) : nlohmann::json(nullptr)},
      {"entries", ok ? entries : nlohmann::json::array()},
      {"truncated", ok && truncated},
    };
  }

  struct job_service_t::impl_t {
    struct record_t {
      job_snapshot_t snapshot;
      std::uint64_t retention_sequence = 0;
      worker_context_t worker;
      fs::path state_path;
      std::chrono::steady_clock::time_point last_progress_persist {};
      std::optional<gpu_workload::lease_t> gpu_lease;
      // Retains DELETE-capable identity ownership for the exact user-writable job
      // root from admission/recovery until no-follow pruning or shutdown.
      std::optional<safe_filesystem::pinned_tree_t> worker_tree;
      // Pruning is two-phase: remove the identity-pinned user artifact tree
      // first, then erase the durable service-owned state. These flags keep a
      // failed pass retryable without re-resolving a released child pin.
      bool worker_artifacts_pruned = false;
      bool worker_result_pin_released_for_prune = false;
      mutable std::mutex artifact_mutex;
      std::vector<pinned_handle_t> input_pins;
      // The job root is owned by worker_tree. Retain an additional result-directory
      // identity pin until pruning/shutdown so the retained artifacts cannot be redirected.
      std::vector<pinned_handle_t> worker_path_pins;
    };

    service_config_t config;
    worker_runner_t runner;
    fs::path ffmpeg;
    std::string ffmpeg_version;
    fs::path ffprobe;
    std::string ffprobe_version;
    std::vector<std::string> codecs;
    mutable std::mutex mutex;
    mutable std::mutex browse_mutex;
    std::condition_variable_any changed;
    std::map<std::string, std::shared_ptr<record_t>> jobs;
    std::uint64_t next_retention_sequence = 0;
    std::optional<std::string> active_id;
    std::optional<std::string> pending_id;
    std::unique_ptr<std::stop_source> active_stop;
    std::jthread supervisor;
    bool started = false;
    bool stopping = false;
    // Every component is held without FILE_SHARE_DELETE for the whole manager
    // lifetime. Revalidation compares the current pathname to the pinned file IDs.
    std::vector<pinned_handle_t> managed_root_pins;

    explicit impl_t(service_config_t value, worker_runner_t worker):
        config(std::move(value)),
        runner(std::move(worker)) {
    }

    bool run_user_filesystem_action(
      const std::function<void()> &action,
      std::string &error
    ) const {
      return run_bound_user_filesystem_action(
        config.expected_user_id,
        action,
        error
      );
    }

    static void release_job_resources(record_t &record) {
      record.gpu_lease.reset();
      record.input_pins.clear();
    }

    void complete_publication_locked(
      record_t &record,
      const publication_outcome_t outcome,
      const publication_origin_e origin
    ) const {
      publication_transaction_t {record.snapshot}.complete(*record.worker.final_output, outcome);
      if (origin == publication_origin_e::recovery) {
        record.snapshot.ended_at_unix_ms =
          record.snapshot.ended_at_unix_ms.value_or(unix_time_ms());
      }
      std::string result_error;
      if (!run_user_filesystem_action([&]() {
            if (!write_json_atomically(
                  record.worker.result_directory / "publication-contract.json",
                  record.snapshot.worker_result,
                  result_error
                )) {
              throw std::runtime_error(result_error);
            }
          },
                                      result_error)) {
        const auto message = origin == publication_origin_e::worker ?
                               "Could not write published offline SBS contract ["sv :
                               "Could not rewrite recovered offline SBS publication ["sv;
        BOOST_LOG(warning) << message << record.snapshot.id << "]: "sv << result_error;
      }
    }

    bool persist_locked(const record_t &record, std::string &error) const {
      const auto value = wire::to_json(wire::persisted_job_contract_t {
        .snapshot = job_snapshot_wire_contract(record.snapshot),
        .retention_sequence = record.retention_sequence,
        .worker = wire::persisted_worker_contract_t {
          .spec_path = path_to_utf8(record.worker.worker_spec),
          .spec_sha256 = record.worker.worker_spec_sha256,
          .progress_path = path_to_utf8(record.worker.worker_progress),
          .result_path = path_to_utf8(record.worker.worker_result),
          .log_path = path_to_utf8(record.worker.worker_log),
          .result_directory = path_to_utf8(record.worker.result_directory),
          .staging_output = record.worker.staging_output ?
                              std::optional<std::string> {
                                path_to_utf8(*record.worker.staging_output)
                              } :
                              std::nullopt,
          .ffmpeg_path = path_to_utf8(record.worker.ffmpeg_executable),
          .ffmpeg_version = record.worker.ffmpeg_version,
          .ffprobe_path = path_to_utf8(record.worker.ffprobe_executable),
          .ffprobe_version = record.worker.ffprobe_version,
        },
      });
      return write_json_atomically(record.state_path, value, error);
    }

    std::optional<job_snapshot_t> parse_persisted_snapshot(
      const nlohmann::json &value,
      std::string &error
    ) const {
      try {
        const auto persisted = wire::parse_persisted_job_contract(value);
        const auto &contract = persisted.snapshot;
        job_snapshot_t snapshot;
        snapshot.id = contract.id;
        const auto state = parse_job_state(contract.state);
        const auto operation = parse_operation(contract.operation);
        if (!valid_job_id(snapshot.id) || !state || !operation) {
          error = "persisted job enum or identity is invalid";
          return std::nullopt;
        }
        snapshot.state = *state;
        snapshot.operation = *operation;
        snapshot.input_path = path_from_utf8(contract.input_path);
        std::string persisted_path_error;
        if (!validate_local_path_syntax(
              snapshot.input_path,
              "persisted input path",
              persisted_path_error,
              false
            )) {
          error = std::move(persisted_path_error);
          return std::nullopt;
        }
        if (contract.output_path) {
          const auto persisted_output = path_from_utf8(*contract.output_path);
          const auto output_name = path_to_utf8(persisted_output.filename());
          const auto output_location = contract.output_location ?
                                         parse_output_location(*contract.output_location) :
                                         std::nullopt;
          if (!output_location) {
            error = "persisted output location is unsupported";
            return std::nullopt;
          }
          const auto expected_output =
            (snapshot.input_path.parent_path() / path_from_utf8(output_name)).lexically_normal();
          if (
            !valid_output_name(output_name) ||
            persisted_output.lexically_normal() != expected_output
          ) {
            error = "persisted output escaped the input directory";
            return std::nullopt;
          }
          snapshot.output_path = expected_output;
          snapshot.output_location = output_location;
        } else if (contract.output_location) {
          error = "persisted evaluation job unexpectedly names an output location";
          return std::nullopt;
        }
        snapshot.codec = contract.codec;
        snapshot.transient_raster_hard_cap_bytes = contract.transient_raster_hard_cap_bytes;
        snapshot.created_at_unix_ms = contract.created_at_unix_ms;
        snapshot.started_at_unix_ms = contract.started_at_unix_ms;
        snapshot.ended_at_unix_ms = contract.ended_at_unix_ms;
        snapshot.error = contract.error.value_or("");
        if (contract.worker_result) {
          snapshot.worker_result = *contract.worker_result;
        }
        snapshot.progress.phase = contract.progress.phase;
        snapshot.progress.processed_frames = contract.progress.processed_frames;
        snapshot.progress.total_frames = contract.progress.total_frames;
        snapshot.progress.source_time_seconds = contract.progress.source_time_seconds;
        snapshot.progress.source_duration_seconds = contract.progress.source_duration_seconds;
        return snapshot;
      } catch (const std::exception &exception) {
        error = std::string {"invalid persisted job: "} + exception.what();
        return std::nullopt;
      }
    }

    worker_context_t worker_context_for(
      const job_snapshot_t &snapshot,
      const fs::path &job_directory,
      const std::optional<fs::path> &staging
    ) const {
      return {
        .job_id = snapshot.id,
        .operation = snapshot.operation,
        .input_path = snapshot.input_path,
        .final_output = snapshot.output_path,
        .staging_output = staging,
        .job_directory = job_directory,
        .result_directory = job_directory / "result",
        .worker_spec = job_directory / "worker-spec.json",
        .worker_spec_sha256 = {},
        .worker_progress = job_directory / "worker-progress.json",
        .worker_result = job_directory / "worker-result.json",
        .worker_log = job_directory / "worker.log",
        .sunshine_executable = config.sunshine_executable,
        .sunshine_config = config.sunshine_config,
        .ffmpeg_executable = ffmpeg,
        .ffmpeg_version = ffmpeg_version,
        .ffprobe_executable = ffprobe,
        .ffprobe_version = ffprobe_version,
        .codec = snapshot.codec,
        .transient_raster_hard_cap_bytes = snapshot.transient_raster_hard_cap_bytes,
      };
    }

    static bool accumulate_measured_bytes(
      const safe_filesystem::tree_measurement_t &measurement,
      std::uint64_t &total,
      std::string &error
    ) {
      if (
        total >
        std::numeric_limits<std::uint64_t>::max() -
          measurement.regular_file_bytes
      ) {
        error = "retained artifact byte count overflowed";
        return false;
      }
      total += measurement.regular_file_bytes;
      return true;
    }

    static bool accumulate_tree_bytes(
      const fs::path &root,
      std::uint64_t &total,
      std::string &error
    ) {
      auto measurement =
        safe_filesystem::measure_tree_no_follow(root, error);
      if (!measurement) {
        error =
          "cannot safely measure retained artifact tree [" +
          path_to_utf8(root) + "]: " + error;
        return false;
      }
      return accumulate_measured_bytes(*measurement, total, error);
    }

    std::optional<std::uint64_t> retained_artifact_bytes_locked(
      std::string &error
    ) const {
      std::uint64_t total = 0;
      if (!accumulate_tree_bytes(
            config.state_root / "jobs",
            total,
            error
          )) {
        return std::nullopt;
      }

      if (!run_user_filesystem_action([&]() {
            if (!accumulate_tree_bytes(
                  config.worker_root / "jobs",
                  total,
                  error
                )) {
              throw std::runtime_error(error);
            }

            for (const auto &[id, record] : jobs) {
              if (!record->worker.staging_output) {
                continue;
              }
              std::string staging_error;
              const auto staging_charge = retained_staging_charge(
                *record->worker.staging_output,
                staging_error
              );
              if (!staging_charge) {
                throw std::runtime_error(
                  "cannot safely account retained staging output for job [" +
                  id + "]: " + staging_error
                );
              }
              if (
                total >
                std::numeric_limits<std::uint64_t>::max() - *staging_charge
              ) {
                throw std::runtime_error(
                  "retained staging artifact quota charge overflowed"
                );
              }
              total += *staging_charge;
            }
          }, error)) {
        return std::nullopt;
      }
      return total;
    }

    bool retire_completed_staging_locked(
      const std::shared_ptr<record_t> &record,
      std::string &error
    ) const {
      publication_transaction_t transaction {record->snapshot};
      if (
        record->snapshot.state != job_state_e::complete ||
        record->snapshot.operation != operation_e::convert ||
        !record->worker.staging_output ||
        !record->worker.final_output ||
        !transaction.published()
      ) {
        error = "completed publication contract is incomplete";
        return false;
      }

      bool cleanup_complete = false;
      std::string cleanup_error;
      const bool cleanup_accessible = run_user_filesystem_action([&]() {
        std::optional<std::vector<pinned_handle_t>> output_directory_pins;
        do {
          // Check the staging leaf before the final output or its parent. If
          // the exact staging pathname is already absent, there is nothing
          // left for the protected record to retire even when the user has
          // since moved or deleted the whole media directory.
          bool staging_missing = false;
          auto staged_file = publication_file_t::open(
            *record->worker.staging_output,
            std::nullopt,
            cleanup_error,
            &staging_missing
          );
          if (staging_missing) {
            cleanup_error.clear();
            cleanup_complete = true;
            break;
          }
          if (!staged_file) {
            break;
          }

          output_directory_pins = pin_publication_directory(
            record->worker.final_output->parent_path(),
            cleanup_error
          );
          if (!output_directory_pins) {
            break;
          }
          if (!file_matches_publish_identity(
                *record->worker.staging_output,
                staged_file->identity()
              )) {
            cleanup_error =
              "retained staging output changed while its directory was "
              "being pinned";
            break;
          }

          try {
            const auto &identity = transaction.identity();
            if (!identity.is_object()) {
              cleanup_error =
                "completed publication identity is not an object";
              break;
            }
            if (staged_file->identity() != identity) {
              cleanup_error =
                "retained staging output identity changed after publication";
              break;
            }
            if (!file_matches_publish_identity(
                  *record->worker.final_output,
                  identity
                )) {
              // The staging link can be the last intact copy if the user moved,
              // deleted, or replaced the final path. Preserve it rather than
              // making Clear destructive.
              cleanup_error =
                "the converted output is missing or changed; the retained "
                "staging file may be its last intact copy";
              break;
            }
            cleanup_complete =
              staged_file->retire_staging_link(cleanup_error);
          } catch (const std::exception &exception) {
            cleanup_error =
              std::string {"invalid completed publication contract: "} +
              exception.what();
          }
        } while (false);

        if (
          output_directory_pins &&
          !revalidate_pins(
            *output_directory_pins,
            "offline output directory during staging cleanup",
            cleanup_error
          )
        ) {
          cleanup_complete = false;
        }
      }, cleanup_error);
      if (!cleanup_accessible || !cleanup_complete) {
        error = cleanup_error.empty() ?
                  "completed staging cleanup did not finish" :
                  cleanup_error;
        return false;
      }
      error.clear();
      return true;
    }

    bool clear_terminal_record_locked(
      const std::shared_ptr<record_t> &record,
      std::string &error
    ) {
      if (!is_terminal(record->snapshot.state)) {
        error = "only a finished offline SBS job can be cleared";
        return false;
      }
      publication_transaction_t transaction {record->snapshot};
      const bool staging_cleanup_pending = transaction.cleanup_pending();
      if (staging_cleanup_pending) {
        std::string cleanup_error;
        if (!retire_completed_staging_locked(record, cleanup_error)) {
          error = "retained staging cleanup is not safe: " + cleanup_error;
          return false;
        }
        transaction.set_staging_cleanup(staging_cleanup_e::complete);
        std::string persist_error;
        if (!persist_locked(*record, persist_error)) {
          error =
            "cannot persist completed staging cleanup: " + persist_error;
          return false;
        }
      }
      if (record->worker.staging_output) {
        std::optional<std::uint64_t> staging_charge;
        std::string staging_error;
        const bool staging_inspected = run_user_filesystem_action([&]() {
          staging_charge = retained_staging_charge(
            *record->worker.staging_output,
            staging_error
          );
          if (!staging_charge) {
            throw std::runtime_error(staging_error);
          }
        }, staging_error);
        if (!staging_inspected || !staging_charge) {
          error =
            "the retained staging path could not be inspected safely: " +
            staging_error;
          return false;
        }
        if (*staging_charge != 0) {
          // The protected record is the only durable map to a staging name
          // beside the input. Preserve it until the
          // exact path disappears; the manager never deletes an unattested
          // or possibly replaced user-writable leaf by pathname.
          error =
            "remove the retained .sunshine3d-*.part* file before clearing "
            "this job";
          return false;
        }
      }
      if (transaction.cleanup_pending()) {
        error =
          "staging cleanup remains pending because the protected publication "
          "contract is incomplete";
        return false;
      }

      std::lock_guard artifact_lock {record->artifact_mutex};
      if (!revalidate_pins(
            managed_root_pins,
            "offline managed roots",
            error
          )) {
        return false;
      }
      if (!record->worker_artifacts_pruned) {
        if (!record->worker_result_pin_released_for_prune) {
          if (record->worker_path_pins.empty()) {
            error = "worker result path is not identity-pinned";
          }
          if (
            record->worker_path_pins.empty() ||
            !revalidate_pins(
              record->worker_path_pins,
              "offline worker job paths",
              error
            )
          ) {
            return false;
          }
        }
        if (!record->worker_tree) {
          error = "offline SBS worker tree is not identity-pinned";
          return false;
        }
        // Release the child result pin before the retained tree removes that
        // exact child by handle. The job-root removal pin remains continuously
        // held, and a failed pass records the release so it can be retried.
        record->worker_path_pins.clear();
        record->worker_result_pin_released_for_prune = true;
        if (!run_user_filesystem_action([&]() {
              if (!record->worker_tree->remove(error)) {
                throw std::runtime_error(error);
              }
            }, error)) {
          // Keep durable state, the exact root pin, and the in-memory record so
          // a later pass can retry without resolving a user-writable pathname.
          return false;
        }
        record->worker_tree.reset();
        record->worker_artifacts_pruned = true;
      }

      if (
        !safe_filesystem::remove_tree_no_follow(
          record->state_path.parent_path(),
          error
        )
      ) {
        // The durable record remains recoverable. A restart recognizes a
        // missing worker tree as an already completed first clear phase.
        return false;
      }
      jobs.erase(record->snapshot.id);
      return true;
    }

    void prune_history_locked(const std::size_t reserved_job_slots = 0) {
      const auto retained_job_limit =
        reserved_job_slots >= config.max_retained_jobs ?
          0u :
          config.max_retained_jobs - reserved_job_slots;
      std::string quota_error;
      auto retained_bytes = retained_artifact_bytes_locked(quota_error);
      if (!retained_bytes) {
        BOOST_LOG(warning)
          << "Could not measure retained offline artifacts: "sv
          << quota_error;
      }
      const auto has_pending_state_prune = std::ranges::any_of(
        jobs,
        [](const auto &entry) {
          return
            is_terminal(entry.second->snapshot.state) &&
            entry.second->worker_artifacts_pruned;
        }
      );
      if (
        !has_pending_state_prune &&
        jobs.size() <= retained_job_limit &&
        (
          !retained_bytes ||
          *retained_bytes <= config.max_retained_artifact_bytes
        )
      ) {
        return;
      }
      std::vector<std::shared_ptr<record_t>> terminal;
      terminal.reserve(jobs.size());
      for (const auto &[id, record] : jobs) {
        if (is_terminal(record->snapshot.state)) {
          terminal.push_back(record);
        }
      }
      std::ranges::sort(terminal, [](const auto &left, const auto &right) {
        if (left->retention_sequence != right->retention_sequence) {
          return left->retention_sequence < right->retention_sequence;
        }
        return left->snapshot.id < right->snapshot.id;
      });
      for (const auto &record : terminal) {
        const auto pressure_requires_prune = [&]() {
          return
            jobs.size() > retained_job_limit ||
            (
              retained_bytes &&
              *retained_bytes > config.max_retained_artifact_bytes
            );
        };
        // Preserve one newest record so the UI can always explain the last outcome.
        // Admission may reserve the manager's entire one-record capacity, in which
        // case the prior terminal record must be pruned before the replacement can
        // be accepted. If that record cannot be pruned safely, admission fails closed.
        if (
          !record->worker_artifacts_pruned &&
          (
            (retained_job_limit != 0 && jobs.size() <= 1) ||
            !pressure_requires_prune()
          )
        ) {
          continue;
        }

        std::string prune_error;
        const bool pruned = clear_terminal_record_locked(record, prune_error);
        quota_error.clear();
        retained_bytes = retained_artifact_bytes_locked(quota_error);
        if (!retained_bytes) {
          BOOST_LOG(warning)
            << "Could not remeasure retained offline artifacts after pruning: "sv
            << quota_error;
        }
        if (!pruned) {
          BOOST_LOG(warning)
            << "Could not prune offline SBS job ["sv
            << record->snapshot.id << "]: "sv << prune_error;
          continue;
        }
      }
    }

    bool recover(std::string &error) {
      const auto jobs_root = config.state_root / "jobs";
      std::error_code ec;
      std::vector<fs::path> invalid_state_directories;
      for (const auto &entry : fs::directory_iterator(jobs_root, ec)) {
        if (ec) {
          error = "cannot enumerate persisted jobs: " + ec.message();
          return false;
        }
        if (entry.is_symlink(ec) || ec || !entry.is_directory(ec) || ec) {
          ec.clear();
          continue;
        }
        const auto id = entry.path().filename().string();
        if (!valid_job_id(id)) {
          invalid_state_directories.push_back(entry.path());
          continue;
        }
        std::string read_error;
        auto value = read_json_contract(entry.path() / "job.json", read_error);
        if (!value) {
          BOOST_LOG(warning) << "Ignoring invalid offline SBS job ["sv
                             << id << "]: "sv << read_error;
          invalid_state_directories.push_back(entry.path());
          continue;
        }
        auto snapshot = parse_persisted_snapshot(*value, read_error);
        if (!snapshot || snapshot->id != id) {
          BOOST_LOG(warning) << "Ignoring invalid offline SBS job ["sv
                             << id << "]: "sv << read_error;
          invalid_state_directories.push_back(entry.path());
          continue;
        }
        const bool migrate_retention_sequence =
          !value->contains("retention_sequence");
        std::uint64_t retention_sequence = 0;
        try {
          if (value->contains("retention_sequence")) {
            if (
              !value->at("retention_sequence").is_number_unsigned() ||
              value->at("retention_sequence").get<std::uint64_t>() == 0
            ) {
              throw std::runtime_error("invalid retention sequence");
            }
            retention_sequence =
              value->at("retention_sequence").get<std::uint64_t>();
          } else {
            if (
              next_retention_sequence ==
              std::numeric_limits<std::uint64_t>::max()
            ) {
              throw std::runtime_error("retention sequence exhausted");
            }
            retention_sequence = ++next_retention_sequence;
          }
        } catch (const std::exception &exception) {
          BOOST_LOG(warning) << "Ignoring invalid offline SBS job ["sv
                             << id << "]: "sv << exception.what();
          invalid_state_directories.push_back(entry.path());
          continue;
        }
        next_retention_sequence =
          std::max(next_retention_sequence, retention_sequence);
        std::optional<fs::path> staging;
        if (snapshot->output_path) {
          staging = staging_path_for(
            *snapshot->output_path,
            snapshot->id
          );
        }
        auto record = std::make_shared<record_t>();
        record->snapshot = std::move(*snapshot);
        record->retention_sequence = retention_sequence;
        record->worker = worker_context_for(
          record->snapshot,
          config.worker_root / "jobs" / id,
          staging
        );
        record->state_path = entry.path() / "job.json";
        publication_transaction_t transaction {record->snapshot};
        bool worker_paths_trusted = false;
        std::string worker_pin_error;
        const bool worker_paths_accessible =
          run_user_filesystem_action([&]() {
            std::error_code user_ec;
            const auto worker_status =
              fs::symlink_status(record->worker.job_directory, user_ec);
            if (
              !user_ec &&
              worker_status.type() == fs::file_type::not_found
            ) {
              if (is_terminal(record->snapshot.state)) {
                // A prior prune may have removed the user artifact tree and then
                // failed to remove this durable record. Resume at phase two.
                record->worker_artifacts_pruned = true;
                worker_paths_trusted = true;
              } else {
                worker_pin_error =
                  "retained worker job directory is missing";
              }
              return;
            }
            if (user_ec) {
              worker_pin_error =
                "cannot inspect retained worker job directory: " +
                user_ec.message();
              return;
            }
            if (
              !fs::is_directory(record->worker.job_directory, user_ec) ||
              user_ec
            ) {
              worker_pin_error =
                "retained worker job directory is not a plain directory";
              return;
            }
            auto worker_tree = safe_filesystem::pinned_tree_t::open(
              record->worker.job_directory,
              safe_filesystem::tree_access_e::remove,
              worker_pin_error
            );
            if (!worker_tree) {
              return;
            }
            auto measurement = worker_tree->measure(worker_pin_error);
            if (
              !measurement ||
              measurement->directory_count == 0
            ) {
              if (worker_pin_error.empty()) {
                worker_pin_error =
                  "retained offline worker root is not a plain directory tree";
                }
                return;
              }
            user_ec.clear();
            const auto result_status =
              fs::symlink_status(record->worker.result_directory, user_ec);
            const bool result_is_plain_directory =
              !user_ec &&
              result_status.type() == fs::file_type::directory;
            if (!result_is_plain_directory) {
              if (!is_terminal(record->snapshot.state)) {
                worker_pin_error =
                  "retained worker result directory is missing or unsafe";
                return;
              }
              // A prior exact-tree prune can remove result/ and then fail on a
              // later locked sibling. Keep the surviving root handle as a
              // cleanup-only terminal record so restart can finish phase one.
              record->worker_tree.emplace(std::move(*worker_tree));
              record->worker_result_pin_released_for_prune = true;
              worker_paths_trusted = true;
              return;
            }
            auto result_pin = pin_managed_directory(
              record->worker.result_directory,
              "retained offline worker result directory",
              worker_pin_error
            );
            if (!result_pin) {
              return;
            }
            record->worker_path_pins.emplace_back(std::move(*result_pin));
            record->worker_tree.emplace(std::move(*worker_tree));
            worker_paths_trusted = true;
          }, worker_pin_error);
        worker_paths_trusted =
          worker_paths_accessible && worker_paths_trusted;
        if (!worker_paths_trusted && !is_terminal(record->snapshot.state)) {
          record->snapshot.state = job_state_e::interrupted;
          record->snapshot.progress.phase = "interrupted";
          record->snapshot.ended_at_unix_ms = unix_time_ms();
          record->snapshot.error =
            "Retained worker paths could not be identity-pinned; no recovery "
            "or cleanup was attempted, and the job was not resumed";
          if (!persist_locked(*record, read_error)) {
            error = "cannot persist path-safe restart recovery: " + read_error;
            return false;
          }
        } else if (record->snapshot.state == job_state_e::publishing) {
          std::optional<publication_outcome_t> publication_outcome;
          std::string publish_error;
          if (!run_user_filesystem_action([&]() {
                if (!record->worker.final_output) {
                  publish_error =
                    "persisted publishing contract has no final output";
                  return;
                }
                const auto output_directory_pins = pin_publication_directory(
                  record->worker.final_output->parent_path(),
                  publish_error
                );
                if (!output_directory_pins) {
                  return;
                }
                try {
                  const auto &identity = transaction.identity();
                  if (
                    !record->worker.staging_output ||
                    !record->worker.final_output ||
                    !identity.is_object()
                  ) {
                    publish_error =
                      "persisted publishing contract is incomplete";
                  } else {
                    const bool final_matches = file_matches_publish_identity(
                      *record->worker.final_output,
                      identity
                    );
                    bool staging_missing = false;
                    auto staged_file = publication_file_t::open(
                      *record->worker.staging_output,
                      identity,
                      publish_error,
                      &staging_missing
                    );
                    if (final_matches) {
                      publication_outcome = publication_outcome_t {
                        .staging_cleanup = staging_cleanup_e::complete,
                      };
                      if (staged_file) {
                        std::string retirement_error;
                        if (!staged_file->retire_staging_link(
                              retirement_error
                            )) {
                          publication_outcome->staging_cleanup = staging_cleanup_e::pending;
                          BOOST_LOG(warning)
                            << "Recovered offline SBS publication but could "
                               "not retire its exact staging link ["sv
                            << id << "]: "sv << retirement_error;
                        }
                      } else if (!staging_missing) {
                        // The final identity proves publication succeeded, but
                        // the staging pathname now names something else or
                        // cannot be pinned. Never delete it by name.
                        publication_outcome->staging_cleanup = staging_cleanup_e::pending;
                        BOOST_LOG(warning)
                          << "Recovered offline SBS publication while "
                             "preserving an untrusted staging pathname ["sv
                          << id << "]: "sv << publish_error;
                      }
                    } else if (staged_file && !final_matches) {
                      publication_outcome = staged_file->publish_no_replace(
                        *record->worker.final_output,
                        publish_error
                      );
                    } else {
                      publish_error =
                        "publishing recovery found an ambiguous or mismatched "
                        "output identity";
                    }
                  }
                } catch (const std::exception &exception) {
                  publish_error =
                    std::string {"invalid persisted publishing contract: "} +
                    exception.what();
                }
                if (
                  output_directory_pins &&
                  !revalidate_pins(
                    *output_directory_pins,
                    "offline output directory during recovery",
                    publish_error
                  )
                ) {
                  publication_outcome.reset();
                }
              }, publish_error)) {
            publication_outcome.reset();
          }
          if (publication_outcome) {
            complete_publication_locked(*record, *publication_outcome, publication_origin_e::recovery);
          } else {
            record->snapshot.state = job_state_e::interrupted;
            record->snapshot.progress.phase = "interrupted";
            record->snapshot.ended_at_unix_ms = unix_time_ms();
            record->snapshot.error =
              "Could not reconcile interrupted output publication: " +
              publish_error;
          }
          std::string cleanup_error;
          if (!run_user_filesystem_action([&]() {
                std::string remove_error;
                if (
                  !record->worker_tree ||
                  !record->worker_tree->remove_child(
                    "native-work",
                    remove_error
                  )
                ) {
                  throw std::runtime_error(
                    remove_error.empty() ?
                      "retained offline worker tree pin is missing" :
                      remove_error
                  );
                }
              }, cleanup_error)) {
            BOOST_LOG(warning)
              << "Could not remove recovered offline SBS transient work ["sv
              << id << "]: "sv << cleanup_error;
          }
          if (!persist_locked(*record, read_error)) {
            error = "cannot persist publication recovery: " + read_error;
            return false;
          }
        } else if (
          record->snapshot.state == job_state_e::complete &&
          record->snapshot.operation == operation_e::convert &&
          record->worker.staging_output &&
          record->worker.final_output &&
          transaction.published()
        ) {
          // A disposition failure after publication can leave a completed
          // record with two hard links. Retry cleanup on every restart using
          // the protected publish identity and a DELETE-capable handle to the
          // exact staging file. The shared helper checks for an already-missing
          // staging leaf before it requires the final video to still exist.
          std::string cleanup_error;
          const bool cleanup_pending =
            !retire_completed_staging_locked(record, cleanup_error);
          const bool marker_changed = transaction.cleanup_pending() != cleanup_pending;
          transaction.set_staging_cleanup(cleanup_pending ? staging_cleanup_e::pending : staging_cleanup_e::complete);
          if (cleanup_pending) {
            BOOST_LOG(warning)
              << "Could not retire retained offline SBS staging link ["sv
              << id << "]: "sv << cleanup_error;
          }
          if (
            marker_changed &&
            !persist_locked(*record, read_error)
          ) {
            error =
              "cannot persist recovered staging cleanup state: " + read_error;
            return false;
          }
        } else if (!is_terminal(record->snapshot.state)) {
          record->snapshot.state = job_state_e::interrupted;
          record->snapshot.progress.phase = "interrupted";
          record->snapshot.ended_at_unix_ms = unix_time_ms();
          record->snapshot.error =
            "Sunshine stopped while this job was active; it was not resumed";
          std::string cleanup_error;
          if (!run_user_filesystem_action([&]() {
                std::string remove_error;
                if (
                  !record->worker_tree ||
                  !record->worker_tree->remove_child(
                    "native-work",
                    remove_error
                  )
                ) {
                  throw std::runtime_error(
                    remove_error.empty() ?
                      "retained offline worker tree pin is missing" :
                      remove_error
                  );
                }
              }, cleanup_error)) {
            BOOST_LOG(warning)
              << "Could not remove interrupted offline SBS transient work ["sv
              << id << "]: "sv << cleanup_error;
          }
          if (!persist_locked(*record, read_error)) {
            error = "cannot persist restart recovery: " + read_error;
            return false;
          }
        }
        if (
          migrate_retention_sequence &&
          !persist_locked(*record, read_error)
        ) {
          error =
            "cannot persist migrated retention order: " + read_error;
          return false;
        }
        jobs.emplace(id, std::move(record));
      }

      for (const auto &directory : invalid_state_directories) {
        const auto id = directory.filename().string();
        std::string state_remove_error;
        if (
          !safe_filesystem::remove_tree_no_follow(
            directory,
            state_remove_error
          )
        ) {
          BOOST_LOG(warning)
            << "Could not remove invalid offline SBS state directory ["sv
            << directory << "]: "sv << state_remove_error;
        }
        if (valid_job_id(id)) {
          // The protected state record was the only durable binding to the
          // user-writable worker tree. Once that record is invalid, resolving
          // the UUID pathname and deleting whatever currently occupies it
          // could erase a same-user replacement. Retain the untrusted tree;
          // quota accounting will fail closed until it is inspected manually.
          BOOST_LOG(warning)
            << "Retaining unowned offline SBS worker directory after invalid "
               "state ["sv
            << id << "]; automatic deletion has no surviving identity proof"sv;
        }
      }
      prune_history_locked();

      // Do not sweep UUID-looking paths that lack protected state. A pathname
      // alone is not ownership evidence after downtime; the current leaf may be
      // an unrelated same-user replacement. Such artifacts remain visible to
      // aggregate quota accounting and therefore block new admission safely.
      return true;
    }

    void publish_progress(
      const std::string &id,
      progress_t progress
    ) {
      std::lock_guard lock {mutex};
      const auto iterator = jobs.find(id);
      if (
        iterator == jobs.end() ||
        (iterator->second->snapshot.state != job_state_e::running &&
         iterator->second->snapshot.state != job_state_e::canceling)
      ) {
        return;
      }
      auto &record = *iterator->second;
      if (progress.processed_frames < record.snapshot.progress.processed_frames) {
        return;
      }
      if (
        record.snapshot.progress.total_frames &&
        progress.total_frames &&
        *record.snapshot.progress.total_frames != *progress.total_frames
      ) {
        return;
      }
      const bool phase_changed =
        record.snapshot.progress.phase != progress.phase;
      record.snapshot.progress = std::move(progress);
      const auto now = std::chrono::steady_clock::now();
      if (
        phase_changed ||
        now - record.last_progress_persist >= 1s
      ) {
        std::string persist_error;
        if (!persist_locked(record, persist_error)) {
          BOOST_LOG(error) << "Could not persist offline SBS progress ["sv
                           << id << "]: "sv << persist_error;
        } else {
          record.last_progress_persist = now;
        }
      }
    }

    void worker_loop(const std::stop_token supervisor_stop) {
      while (true) {
        std::shared_ptr<record_t> record;
        std::stop_token job_stop;
        {
          std::unique_lock lock {mutex};
          changed.wait(lock, supervisor_stop, [&]() {
            return stopping || pending_id.has_value();
          });
          if (!pending_id) {
            if (stopping || supervisor_stop.stop_requested()) {
              return;
            }
            continue;
          }
          const auto iterator = jobs.find(*pending_id);
          if (iterator == jobs.end()) {
            pending_id.reset();
            active_id.reset();
            continue;
          }
          record = iterator->second;
          pending_id.reset();
          if (stopping || supervisor_stop.stop_requested()) {
            record->snapshot.state = job_state_e::interrupted;
            record->snapshot.progress.phase = "interrupted";
            record->snapshot.ended_at_unix_ms = unix_time_ms();
            record->snapshot.error =
              "Sunshine stopped before the queued job started";
            std::string error;
            persist_locked(*record, error);
            release_job_resources(*record);
            active_id.reset();
            return;
          }
          record->snapshot.state = job_state_e::running;
          record->snapshot.progress.phase = "starting";
          record->snapshot.started_at_unix_ms = unix_time_ms();
          active_stop = std::make_unique<std::stop_source>();
          job_stop = active_stop->get_token();
          std::string persist_error;
          if (!persist_locked(*record, persist_error)) {
            record->snapshot.state = job_state_e::failed;
            record->snapshot.progress.phase = "failed";
            record->snapshot.ended_at_unix_ms = unix_time_ms();
            record->snapshot.error =
              "cannot persist running job state: " + persist_error;
            release_job_resources(*record);
            active_stop.reset();
            active_id.reset();
            continue;
          }
        }

        worker_outcome_t outcome;
        try {
          const auto callback = [this, id = record->snapshot.id](
                                  progress_t progress) {
            publish_progress(id, std::move(progress));
          };
          bool worker_identity_valid = record->worker_tree.has_value();
          std::string identity_error;
          if (worker_identity_valid) {
            bool pins_valid = false;
            const bool pins_accessible = run_user_filesystem_action([&]() {
              pins_valid =
              revalidate_pins(
                managed_root_pins,
                "offline managed roots",
                identity_error
              ) &&
              revalidate_pins(
                record->worker_path_pins,
                "offline worker job paths",
                identity_error
              ) &&
              !record->input_pins.empty() &&
              revalidate_pins(
                record->input_pins,
                "offline input/output paths",
                identity_error
              );
            }, identity_error);
            worker_identity_valid = pins_accessible && pins_valid;
          }
          if (!worker_identity_valid) {
            outcome.error =
              identity_error.empty() ?
                "offline worker tree identity pin is missing before worker launch" :
                "offline managed path attestation failed before worker launch: " +
                  identity_error;
          } else if (
            std::string codec_probe_error;
            // Hardware NVENC probing is itself GPU work. It runs only once a user has
            // started a conversion, under the same exclusive lease as the worker, so host
            // startup can never race a fast live-stream connection.
            record->snapshot.operation == operation_e::convert &&
            config.probe_media_tools &&
            !probe_ffmpeg_codec_runtime(
              config,
              ffmpeg,
              record->snapshot.codec,
              codec_probe_error
            )
          ) {
            outcome.error =
              "the requested 10-bit NVIDIA encoder failed its runtime preflight: " +
              codec_probe_error;
          } else if (job_stop.stop_requested()) {
            // Canceled or shut down during the preflight: do not launch the worker at all.
            outcome.canceled = true;
          } else
            outcome = runner ?
                        runner(record->worker, job_stop, callback) :
                        run_native_worker(config, record->worker, job_stop, callback);
        } catch (const std::exception &exception) {
          outcome.error =
            std::string {"native offline worker threw: "} + exception.what();
        } catch (...) {
          outcome.error = "native offline worker threw an unknown exception";
        }

        // The outer worker is now reaped. Remove frame caches/chunks regardless of outcome;
        // cancellation cannot rely on the force-killed child's exception cleanup.
        std::string transient_error;
        std::string transient_access_error;
        bool transient_removed = false;
        for (int attempt = 0; attempt < 3; ++attempt) {
          transient_error.clear();
          transient_access_error.clear();
          const bool accessed = run_user_filesystem_action([&]() {
            if (!record->worker_tree) {
              throw std::runtime_error(
                "offline worker tree identity pin is missing"
              );
            }
            if (
              !record->worker_tree->remove_child(
                "native-work",
                transient_error
              )
            ) {
              throw std::runtime_error(transient_error);
            }
          }, transient_access_error);
          if (accessed) {
            transient_removed = true;
            break;
          }
          std::this_thread::sleep_for(50ms);
        }
        if (!transient_removed) {
          BOOST_LOG(warning) << "Could not remove offline SBS transient work ["sv
                             << record->snapshot.id << "]: "sv
                             << (transient_access_error.empty() ?
                                   transient_error :
                                   transient_access_error);
        }

        {
          std::lock_guard lock {mutex};
          const bool shutdown_interruption = stopping;
          const bool user_canceled =
            record->snapshot.state == job_state_e::canceling;
          std::string completion_error;
          if (shutdown_interruption) {
            record->snapshot.state = job_state_e::interrupted;
            record->snapshot.progress.phase = "interrupted";
            record->snapshot.error =
              "Sunshine stopped while this job was active";
          } else if (user_canceled || outcome.canceled) {
            record->snapshot.state = job_state_e::canceled;
            record->snapshot.progress.phase = "canceled";
            record->snapshot.error.clear();
          } else if (!outcome.completed) {
            record->snapshot.state = job_state_e::failed;
            record->snapshot.progress.phase = "failed";
            record->snapshot.error =
              outcome.error.empty() ?
                "native offline worker did not complete" :
                std::move(outcome.error);
          } else if (record->snapshot.operation == operation_e::convert) {
            std::optional<publication_file_t> staged_output;
            if (
              record->worker.staging_output &&
              record->worker.final_output
            ) {
              run_user_filesystem_action([&]() {
                std::optional<nlohmann::json> expected_identity;
                const auto attested =
                  outcome.result.find("staging_identity");
                if (
                  attested != outcome.result.end() &&
                  attested->is_object()
                ) {
                  expected_identity = *attested;
                } else if (!runner) {
                  completion_error =
                    "native worker omitted its staged-output identity";
                  return;
                }
                staged_output = publication_file_t::open(
                  *record->worker.staging_output,
                  expected_identity,
                  completion_error
                );
              }, completion_error);
            }
            if (!staged_output) {
              record->snapshot.state = job_state_e::failed;
              record->snapshot.progress.phase = "failed";
              record->snapshot.error =
                completion_error.empty() ?
                  "native worker did not produce a publishable staged video" :
                  completion_error;
            } else {
              // Keep the exact attested file open without write/delete sharing
              // across persistence and publication, so the staging path cannot
              // be replaced before its no-replace hard link is created.
              publication_transaction_t transaction {record->snapshot};
              transaction.prepare(
                compact_worker_result(std::move(outcome.result), record->worker),
                staged_output->identity()
              );
              std::string publishing_error;
              if (!persist_locked(*record, publishing_error)) {
                record->snapshot.state = job_state_e::failed;
                record->snapshot.progress.phase = "failed";
                record->snapshot.error =
                  "cannot persist pre-publish state: " + publishing_error;
              } else {
                std::optional<publication_outcome_t> publication_outcome;
                if (!run_user_filesystem_action([&]() {
                      if (
                        record->input_pins.empty() ||
                        !revalidate_pins(
                          record->input_pins,
                          "offline input/output paths before publication",
                          completion_error
                        )
                      ) {
                        return;
                      }
                      publication_outcome = staged_output->publish_no_replace(
                        *record->worker.final_output,
                        completion_error
#ifdef SUNSHINE_TESTS
                        ,
                        config.test_force_staging_disposition_failure
#endif
                      );
                    }, completion_error)) {
                  publication_outcome.reset();
                }
                if (!publication_outcome) {
                  record->snapshot.state = job_state_e::failed;
                  record->snapshot.progress.phase = "failed";
                  record->snapshot.error = completion_error;
                } else {
                  complete_publication_locked(*record, *publication_outcome, publication_origin_e::worker);
                }
              }
            }
          } else {
            record->snapshot.state = job_state_e::complete;
            record->snapshot.progress.phase = "complete";
            record->snapshot.error.clear();
            record->snapshot.worker_result = compact_worker_result(
              std::move(outcome.result),
              record->worker
            );
          }
          // Never remove a failed/canceled staging pathname from the manager. The
          // native worker reserves its file identity and performs handle-based cleanup;
          // after a force kill there is no trustworthy proof that the current pathname
          // still names that file, so a safe leak is preferable to deleting user data.
          record->snapshot.ended_at_unix_ms = unix_time_ms();
          std::string persist_error;
          if (!persist_locked(*record, persist_error)) {
            BOOST_LOG(error) << "Could not persist terminal offline SBS job ["sv
                             << record->snapshot.id << "]: "sv << persist_error;
          }
          release_job_resources(*record);
          active_stop.reset();
          active_id.reset();
          prune_history_locked();
          changed.notify_all();
          if (stopping || supervisor_stop.stop_requested()) {
            return;
          }
        }
      }
    }
  };

  job_service_t::job_service_t(service_config_t config, worker_runner_t runner):
      impl_(std::make_unique<impl_t>(std::move(config), std::move(runner))) {
  }

  job_service_t::~job_service_t() {
    shutdown();
  }

  bool job_service_t::start(std::string &error) {
    std::lock_guard lock {impl_->mutex};
    if (impl_->started) {
      return true;
    }
    if (
      impl_->config.state_root.empty() ||
      impl_->config.worker_root.empty() ||
      impl_->config.sunshine_executable.empty() ||
      impl_->config.sunshine_config.empty() ||
      impl_->config.max_retained_jobs == 0 ||
      impl_->config.max_retained_artifact_bytes == 0 ||
      impl_->config.process_poll_interval <= 0ms
    ) {
      error = "offline SBS job manager configuration is incomplete";
      return false;
    }
    if (
      !validate_local_path_syntax(
        impl_->config.state_root,
        "offline SBS state root",
        error
      ) ||
      !validate_local_path_syntax(
        impl_->config.worker_root,
        "offline SBS worker root",
        error
      )
    ) {
      return false;
    }
    const auto lexical_state_root = impl_->config.state_root.lexically_normal();
    const auto lexical_worker_root = impl_->config.worker_root.lexically_normal();
    if (managed_roots_overlap(lexical_state_root, lexical_worker_root)) {
      error = "offline SBS state and worker roots must be disjoint";
      return false;
    }

    std::error_code ec;
    directory_creation_pins_t state_creation_pins;
    try {
      state_creation_pins = create_user_directory(
        impl_->config.state_root / "jobs",
        "offline SBS state root"
      );
    } catch (const std::exception &caught) {
      error = caught.what();
      return false;
    }
    std::optional<std::vector<pinned_handle_t>> state_root_pins;
    std::optional<std::vector<pinned_handle_t>> state_jobs_pins;
    std::optional<std::vector<pinned_handle_t>> worker_root_pins;
    std::optional<std::vector<pinned_handle_t>> worker_jobs_pins;
    std::string state_pin_error;
    state_root_pins = pin_managed_root_components(
      impl_->config.state_root,
      "offline state root",
      state_pin_error
    );
    if (!state_root_pins) {
      error = state_pin_error;
      return false;
    }
    state_jobs_pins = pin_managed_root_components(
      impl_->config.state_root / "jobs",
      "offline state jobs root",
      state_pin_error
    );
    if (!state_jobs_pins) {
      error = state_pin_error;
      return false;
    }
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          std::error_code user_ec;
          auto worker_creation_pins = create_user_directory(
            impl_->config.worker_root / "jobs",
            "offline SBS worker root"
          );
          std::string pin_error;
          worker_root_pins = pin_managed_root_components(
            impl_->config.worker_root,
            "offline worker root",
            pin_error
          );
          if (!worker_root_pins) {
            throw std::runtime_error(pin_error);
          }
          worker_jobs_pins = pin_managed_root_components(
            impl_->config.worker_root / "jobs",
            "offline worker jobs root",
            pin_error
          );
          if (!worker_jobs_pins) {
            throw std::runtime_error(pin_error);
          }
          impl_->config.worker_root =
            fs::canonical(impl_->config.worker_root, user_ec);
          if (user_ec) {
            throw std::runtime_error(
              "cannot canonicalize offline SBS worker root: " +
              user_ec.message()
            );
          }
        }, error)) {
      return false;
    }
    impl_->managed_root_pins.clear();
    impl_->managed_root_pins.reserve(
      state_root_pins->size() +
      state_jobs_pins->size() +
      worker_root_pins->size() +
      worker_jobs_pins->size()
    );
    std::ranges::move(
      *state_root_pins,
      std::back_inserter(impl_->managed_root_pins)
    );
    std::ranges::move(
      *state_jobs_pins,
      std::back_inserter(impl_->managed_root_pins)
    );
    std::ranges::move(
      *worker_root_pins,
      std::back_inserter(impl_->managed_root_pins)
    );
    std::ranges::move(
      *worker_jobs_pins,
      std::back_inserter(impl_->managed_root_pins)
    );
    impl_->config.state_root = fs::canonical(impl_->config.state_root, ec);
    if (ec) {
      error = "cannot canonicalize offline SBS state root: " + ec.message();
      return false;
    }
    if (managed_roots_overlap(impl_->config.state_root, impl_->config.worker_root)) {
      error = "offline SBS state and worker roots must be disjoint";
      return false;
    }
    auto executable = canonical_regular_file(
      impl_->config.sunshine_executable,
      error
    );
    if (!executable) {
      error = "invalid Sunshine worker executable: " + error;
      return false;
    }
    impl_->config.sunshine_executable = *executable;
    auto config_file = canonical_regular_file(
      impl_->config.sunshine_config,
      error
    );
    if (!config_file) {
      error = "invalid Sunshine configuration file: " + error;
      return false;
    }
    impl_->config.sunshine_config = *config_file;
    auto ffmpeg = discover_media_tool(
      impl_->config,
      impl_->config.ffmpeg_executable,
      "ffmpeg",
      error
    );
    if (!ffmpeg) {
      return false;
    }
    impl_->ffmpeg = *ffmpeg;
    auto version = probe_media_tool_version(
      impl_->config,
      impl_->ffmpeg,
      "ffmpeg",
      error
    );
    if (!version) {
      return false;
    }
    impl_->ffmpeg_version = *version;
    auto ffprobe = discover_media_tool(
      impl_->config,
      impl_->config.ffprobe_executable,
      "ffprobe",
      error
    );
    if (!ffprobe) {
      return false;
    }
    impl_->ffprobe = *ffprobe;
    version = probe_media_tool_version(
      impl_->config,
      impl_->ffprobe,
      "ffprobe",
      error
    );
    if (!version) {
      return false;
    }
    impl_->ffprobe_version = *version;
    if (impl_->config.probe_media_tools) {
      const auto ffmpeg_token = media_tool_version_token(
        impl_->ffmpeg_version,
        "ffmpeg"
      );
      const auto ffprobe_token = media_tool_version_token(
        impl_->ffprobe_version,
        "ffprobe"
      );
      if (!ffmpeg_token || !ffprobe_token || *ffmpeg_token != *ffprobe_token) {
        error =
          "approved ffmpeg.exe and ffprobe.exe do not report the same version";
        return false;
      }
    }
    auto codecs = probe_ffmpeg_codecs(
      impl_->config,
      impl_->ffmpeg,
      error
    );
    if (!codecs) {
      return false;
    }
    impl_->codecs = std::move(*codecs);
    if (!impl_->recover(error)) {
      return false;
    }
    impl_->started = true;
    impl_->stopping = false;
    impl_->supervisor = std::jthread([implementation = impl_.get()](
                                      const std::stop_token stop) {
      implementation->worker_loop(stop);
    });
    BOOST_LOG(info) << "Native offline SBS job manager ready; FFmpeg ["sv
                    << impl_->ffmpeg << "] "sv << impl_->ffmpeg_version
                    << ", FFprobe ["sv << impl_->ffprobe << "] "sv
                    << impl_->ffprobe_version;
    return true;
  }

  void job_service_t::shutdown() {
    std::jthread supervisor;
    {
      std::lock_guard lock {impl_->mutex};
      if (!impl_->started) {
        for (auto &entry : impl_->jobs) {
          entry.second->input_pins.clear();
          entry.second->worker_path_pins.clear();
          entry.second->worker_tree.reset();
        }
        impl_->managed_root_pins.clear();
        return;
      }
      impl_->stopping = true;
      if (impl_->active_id) {
        const auto iterator = impl_->jobs.find(*impl_->active_id);
        if (
          iterator != impl_->jobs.end() &&
          iterator->second->snapshot.state == job_state_e::queued
        ) {
          iterator->second->snapshot.state = job_state_e::interrupted;
          iterator->second->snapshot.progress.phase = "interrupted";
          iterator->second->snapshot.ended_at_unix_ms = unix_time_ms();
          iterator->second->snapshot.error =
            "Sunshine stopped before the queued job started";
          std::string ignored;
          impl_->persist_locked(*iterator->second, ignored);
          impl_t::release_job_resources(*iterator->second);
          impl_->pending_id.reset();
          impl_->active_id.reset();
          impl_->prune_history_locked();
        } else if (impl_->active_stop) {
          impl_->active_stop->request_stop();
        }
      }
      impl_->supervisor.request_stop();
      impl_->changed.notify_all();
      supervisor = std::move(impl_->supervisor);
    }
    if (supervisor.joinable()) {
      supervisor.join();
    }
    {
      std::lock_guard lock {impl_->mutex};
      impl_->started = false;
      impl_->stopping = false;
      impl_->active_stop.reset();
      impl_->pending_id.reset();
      impl_->active_id.reset();
      for (auto &entry : impl_->jobs) {
        auto &record = entry.second;
        std::lock_guard artifact_lock {record->artifact_mutex};
        record->input_pins.clear();
        record->worker_path_pins.clear();
        record->worker_tree.reset();
      }
      impl_->managed_root_pins.clear();
    }
  }

  service_reply_t job_service_t::create(const create_request_t &request) {
    {
      std::lock_guard lock {impl_->mutex};
      if (!impl_->started || impl_->stopping) {
        return {
          .code = error_code_e::not_initialized,
          .error = "offline SBS job manager is not running",
        };
      }
      if (impl_->active_id) {
        return {
          .code = error_code_e::busy,
          .error = "another offline SBS job is already active",
          .job = impl_->jobs.at(*impl_->active_id)->snapshot,
        };
      }
      std::string root_error;
      if (!revalidate_pins(
            impl_->managed_root_pins,
            "offline managed roots",
            root_error
          )) {
        return {
          .code = error_code_e::unavailable,
          .error = root_error,
        };
      }
    }
    if (
      request.operation == operation_e::convert &&
      !valid_codec(request.codec)
    ) {
      return {
        .code = error_code_e::invalid_request,
        .error = "unsupported offline NVENC codec",
      };
    }
    const std::string effective_codec =
      request.operation == operation_e::convert ?
        request.codec :
        "hevc_nvenc";
    std::string path_error;
    const auto normalized_input = request.input_path.lexically_normal();
    if (!validate_local_input_path_syntax(normalized_input, path_error)) {
      return {
        .code = error_code_e::invalid_request,
        .error = path_error,
      };
    }
    std::optional<fs::path> input;
    std::optional<std::vector<pinned_handle_t>> input_pins;
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          input_pins =
            pin_no_reparse_components(normalized_input, path_error);
          if (!input_pins) {
            throw std::runtime_error(path_error);
          }
          input = canonical_regular_file(
            normalized_input,
            path_error
          );
          if (!input || !regular_nonempty_file(*input)) {
            throw std::runtime_error(
              "input is not a non-empty regular file" +
              (path_error.empty() ? std::string {} : ": " + path_error)
            );
          }
        }, path_error)) {
      return {
        .code = error_code_e::invalid_request,
        .error = path_error,
      };
    }
    if (!validate_local_input_path_syntax(*input, path_error)) {
      return {
        .code = error_code_e::invalid_request,
        .error = "resolved input is not a local file: " + path_error,
      };
    }
    if (
      request.operation == operation_e::convert &&
      !valid_output_name(request.output_name)
    ) {
      return {
        .code = error_code_e::invalid_request,
        .error =
          "conversion output_name must be a safe .mkv or .mp4 basename",
      };
    }
    if (
      request.operation == operation_e::evaluate &&
      !request.output_name.empty()
    ) {
      return {
        .code = error_code_e::invalid_request,
        .error = "evaluation jobs do not accept output_name",
      };
    }

    // Filesystem canonicalization happens above without holding the lifecycle mutex. Recheck
    // admission after reacquiring it so a concurrent create/shutdown cannot race this job
    // into the queue.
    std::lock_guard lock {impl_->mutex};
    if (!impl_->started || impl_->stopping) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not running",
      };
    }
    if (impl_->active_id) {
      return {
        .code = error_code_e::busy,
        .error = "another offline SBS job is already active",
        .job = impl_->jobs.at(*impl_->active_id)->snapshot,
      };
    }
    if (
      request.operation == operation_e::convert &&
      std::find(
        impl_->codecs.begin(),
        impl_->codecs.end(),
        effective_codec
      ) == impl_->codecs.end()
    ) {
      return {
        .code = error_code_e::unavailable,
        .error = "the requested 10-bit NVIDIA encoder is unavailable",
      };
    }
    if (
      impl_->next_retention_sequence ==
      std::numeric_limits<std::uint64_t>::max()
    ) {
      return {
        .code = error_code_e::unavailable,
        .error = "offline job retention sequence is exhausted",
      };
    }
    // Retention is enforced before acquiring scarce GPU ownership. Terminal job records
    // are pruned first; unattested staging files are deliberately never deleted and can
    // therefore fail admission until the interactive user inspects/removes them.
    // Reserve room for this job before acquiring scarce GPU ownership. A retained
    // input-directory staging leaf can make an old terminal record intentionally
    // unprunable; do not let repeated failures grow the durable map beyond its
    // configured record bound while remaining below the much larger byte quota.
    impl_->prune_history_locked(1);
    const auto admission_history_limit =
      impl_->config.max_retained_jobs - 1;
    if (impl_->jobs.size() > admission_history_limit) {
      return {
        .code = error_code_e::unavailable,
        .error =
          "offline job history cannot make room under the configured " +
          std::to_string(impl_->config.max_retained_jobs) +
          "-record limit; inspect retained .sunshine3d-*.part* files and "
          "locked offline artifacts before starting another job",
      };
    }
    std::string quota_error;
    const auto retained_bytes =
      impl_->retained_artifact_bytes_locked(quota_error);
    if (!retained_bytes) {
      return {
        .code = error_code_e::unavailable,
        .error =
          "cannot verify the offline retained-artifact quota: " + quota_error,
      };
    }
    if (*retained_bytes > impl_->config.max_retained_artifact_bytes) {
      return {
        .code = error_code_e::unavailable,
        .error =
          "offline retained artifacts exceed the configured " +
          std::to_string(impl_->config.max_retained_artifact_bytes) +
          "-byte quota; inspect and remove retained .sunshine3d-*.part* "
          "files before starting another job",
      };
    }
    auto gpu_lease =
      gpu_workload::try_acquire(gpu_workload::kind_e::offline_sbs);
    if (!gpu_lease) {
      return {
        .code = error_code_e::busy,
        .error =
          "offline conversion cannot start while a live streaming session is active",
      };
    }
    // The hardware NVENC runtime preflight is GPU work that can take several seconds. It runs
    // on the job's supervisor thread under this lease (see worker_loop), not here: this call
    // is made on the Web UI's only request thread while holding the lifecycle mutex.

    job_snapshot_t snapshot;
    snapshot.state = job_state_e::queued;
    snapshot.operation = request.operation;
    snapshot.input_path = *input;
    // Analysis never invokes an encoder. Keep one canonical value in the
    // persisted worker schema without making NVENC availability an admission
    // requirement for evaluate-only jobs.
    snapshot.codec = effective_codec;
    snapshot.transient_raster_hard_cap_bytes = transient_raster_hard_cap_bytes;
    snapshot.progress.phase = "queued";
    snapshot.created_at_unix_ms = unix_time_ms();
    if (request.operation == operation_e::convert) {
      snapshot.output_path =
        input->parent_path() / path_from_utf8(request.output_name);
      snapshot.output_location = output_location_e::input_directory;
      bool output_exists = false;
      bool same_as_input = false;
      if (!run_bound_user_filesystem_action(
            impl_->config.expected_user_id,
            [&]() {
            std::error_code user_ec;
            output_exists = fs::exists(*snapshot.output_path, user_ec);
            if (user_ec) {
              throw std::runtime_error(
                "cannot inspect conversion output: " + user_ec.message()
              );
            }
            if (output_exists) {
              same_as_input = fs::equivalent(
                *input,
                *snapshot.output_path,
                user_ec
              );
              if (user_ec) {
                throw std::runtime_error(
                  "cannot compare conversion output: " + user_ec.message()
                );
              }
            }
          }, path_error)) {
        return {
          .code = error_code_e::io_error,
          .error = path_error,
        };
      }
      if (output_exists) {
        return {
          .code = error_code_e::conflict,
          .error = "conversion output already exists",
        };
      }
      if (same_as_input) {
        return {
          .code = error_code_e::conflict,
          .error = "input and output resolve to the same file",
        };
      }
    }

    fs::path state_job_directory;
    std::error_code ec;
    for (int attempt = 0; attempt < 32; ++attempt) {
      const auto candidate = uuid_util::uuid_t::generate().string();
      if (impl_->jobs.contains(candidate)) {
        continue;
      }
      const auto candidate_directory =
        impl_->config.state_root / "jobs" / candidate;
      ec.clear();
      if (fs::create_directory(candidate_directory, ec)) {
        snapshot.id = candidate;
        state_job_directory = candidate_directory;
        break;
      }
      if (ec && ec != std::errc::file_exists) {
        return {
          .code = error_code_e::io_error,
          .error = "cannot reserve job directory: " + ec.message(),
        };
      }
    }
    if (snapshot.id.empty()) {
      return {
        .code = error_code_e::io_error,
        .error = "could not allocate a unique offline SBS job identifier",
      };
    }

    const auto worker_job_directory =
      impl_->config.worker_root / "jobs" / snapshot.id;
    std::optional<fs::path> staging;
    if (snapshot.output_path) {
      staging = staging_path_for(
        *snapshot.output_path,
        snapshot.id
      );
    }

    std::optional<safe_filesystem::pinned_tree_t> worker_tree_pin;
    std::optional<pinned_handle_t> worker_result_pin;
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          std::error_code user_ec;
          if (staging && fs::exists(*staging, user_ec)) {
            throw std::runtime_error(
              "conversion staging output already exists"
            );
          }
          if (user_ec) {
            throw std::runtime_error(
              "cannot inspect conversion staging output: " +
              user_ec.message()
            );
          }
          if (!fs::create_directory(worker_job_directory, user_ec)) {
            throw std::runtime_error(
              user_ec ?
                "cannot reserve worker directory: " + user_ec.message() :
                "worker directory already exists"
            );
          }
          std::string pin_error;
          worker_tree_pin = safe_filesystem::pinned_tree_t::open(
            worker_job_directory,
            safe_filesystem::tree_access_e::remove,
            pin_error
          );
          if (!worker_tree_pin) {
            throw std::runtime_error(pin_error);
          }
          fs::create_directory(worker_job_directory / "result", user_ec);
          if (user_ec) {
            throw std::runtime_error(
              "cannot create worker result directory: " +
              user_ec.message()
            );
          }
          auto worker_measurement = worker_tree_pin->measure(pin_error);
          if (
            !worker_measurement ||
            worker_measurement->directory_count == 0
          ) {
            throw std::runtime_error(
              pin_error.empty() ?
                "offline worker root is not a plain directory tree" :
                pin_error
            );
          }
          worker_result_pin = pin_managed_directory(
            worker_job_directory / "result",
            "offline worker result directory",
            pin_error
          );
          if (!worker_result_pin) {
            throw std::runtime_error(pin_error);
          }
        }, path_error)) {
      worker_result_pin.reset();
      const auto admission_error = path_error;
      std::string cleanup_error;
      if (!run_bound_user_filesystem_action(
            impl_->config.expected_user_id,
            [&]() {
            if (worker_tree_pin) {
              if (!worker_tree_pin->remove(cleanup_error)) {
                throw std::runtime_error(cleanup_error);
              }
              worker_tree_pin.reset();
            } else {
              // Creation succeeded but exact identity pinning did not. The
              // current UUID pathname may already be a same-user replacement;
              // preserve it rather than deleting by name without ownership.
              cleanup_error =
                "worker directory was never identity-pinned; retained safely";
            }
          }, cleanup_error)) {
        BOOST_LOG(warning)
          << "Could not clean rejected offline SBS worker directory: "sv
          << cleanup_error;
      } else if (!cleanup_error.empty()) {
        BOOST_LOG(warning)
          << "Retaining rejected offline SBS worker directory: "sv
          << cleanup_error;
      }
      std::string state_cleanup_error;
      if (
        !safe_filesystem::remove_tree_no_follow(
          state_job_directory,
          state_cleanup_error
        )
      ) {
        BOOST_LOG(warning)
          << "Could not clean rejected offline SBS state directory: "sv
          << state_cleanup_error;
      }
      return {
        .code = admission_error == "conversion staging output already exists" ?
                  error_code_e::conflict :
                  error_code_e::io_error,
        .error = admission_error,
      };
    }

    auto record = std::make_shared<impl_t::record_t>();
    record->snapshot = snapshot;
    record->retention_sequence = ++impl_->next_retention_sequence;
    record->worker = impl_->worker_context_for(
      snapshot,
      worker_job_directory,
      staging
    );
    record->state_path = state_job_directory / "job.json";
    record->gpu_lease = std::move(gpu_lease);
    record->worker_tree.emplace(std::move(*worker_tree_pin));
    record->input_pins = std::move(*input_pins);
    record->worker_path_pins.emplace_back(std::move(*worker_result_pin));
    std::string write_error;
    const auto worker_spec_bytes =
      worker_spec_json(record->worker).dump(2) + "\n";
    record->worker.worker_spec_sha256 = sha256_hex(worker_spec_bytes);
    bool worker_spec_written = false;
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          worker_spec_written = write_text_atomically(
            record->worker.worker_spec,
            worker_spec_bytes,
            write_error
          );
        }, path_error)) {
      write_error = path_error;
    }
    if (!worker_spec_written || !impl_->persist_locked(*record, write_error)) {
      record->worker_path_pins.clear();
      std::string worker_cleanup_error;
      if (!run_bound_user_filesystem_action(
            impl_->config.expected_user_id,
            [&]() {
            if (
              !record->worker_tree ||
              !record->worker_tree->remove(worker_cleanup_error)
            ) {
              throw std::runtime_error(
                worker_cleanup_error.empty() ?
                  "offline worker tree pin is missing" :
                  worker_cleanup_error
              );
            }
            record->worker_tree.reset();
          }, worker_cleanup_error)) {
        BOOST_LOG(warning)
          << "Could not clean unpersisted offline SBS worker directory: "sv
          << worker_cleanup_error;
      }
      std::string state_cleanup_error;
      if (
        !safe_filesystem::remove_tree_no_follow(
          state_job_directory,
          state_cleanup_error
        )
      ) {
        BOOST_LOG(warning)
          << "Could not clean unpersisted offline SBS state directory: "sv
          << state_cleanup_error;
      }
      return {
        .code = error_code_e::io_error,
        .error = "cannot persist new offline SBS job: " + write_error,
      };
    }
    const auto [inserted, accepted] =
      impl_->jobs.emplace(snapshot.id, record);
    if (!accepted) {
      record->worker_path_pins.clear();
      std::string worker_cleanup_error;
      if (!run_bound_user_filesystem_action(
            impl_->config.expected_user_id,
            [&]() {
            if (
              !record->worker_tree ||
              !record->worker_tree->remove(worker_cleanup_error)
            ) {
              throw std::runtime_error(
                worker_cleanup_error.empty() ?
                  "offline worker tree pin is missing" :
                  worker_cleanup_error
              );
            }
            record->worker_tree.reset();
          }, worker_cleanup_error)) {
        BOOST_LOG(warning)
          << "Could not clean colliding offline SBS worker directory: "sv
          << worker_cleanup_error;
      }
      std::string state_cleanup_error;
      if (
        !safe_filesystem::remove_tree_no_follow(
          state_job_directory,
          state_cleanup_error
        )
      ) {
        BOOST_LOG(warning)
          << "Could not clean colliding offline SBS state directory: "sv
          << state_cleanup_error;
      }
      return {
        .code = error_code_e::io_error,
        .error = "offline SBS job identifier collided during admission",
      };
    }
    impl_->active_id = snapshot.id;
    impl_->pending_id = snapshot.id;
    impl_->changed.notify_all();
    return {
      .ok = true,
      .job = snapshot,
    };
  }

  service_reply_t job_service_t::cancel(const std::string_view id) {
    std::lock_guard lock {impl_->mutex};
    if (!impl_->started || impl_->stopping) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not running",
      };
    }
    const auto iterator = impl_->jobs.find(std::string {id});
    if (iterator == impl_->jobs.end()) {
      return {
        .code = error_code_e::not_found,
        .error = "offline SBS job was not found",
      };
    }
    // Keep an owning reference through persistence, pruning, and reply construction.
    // A queued cancellation becomes terminal immediately and may itself be selected by
    // retention when timestamps tie or the wall clock moves backwards.
    const auto record = iterator->second;
    if (is_terminal(record->snapshot.state)) {
      return {
        .code = error_code_e::conflict,
        .error = "offline SBS job is already terminal",
        .job = record->snapshot,
      };
    }
    const bool canceled_while_queued =
      record->snapshot.state == job_state_e::queued;
    if (canceled_while_queued) {
      record->snapshot.state = job_state_e::canceled;
      record->snapshot.progress.phase = "canceled";
      record->snapshot.ended_at_unix_ms = unix_time_ms();
      impl_->pending_id.reset();
      impl_->active_id.reset();
      impl_t::release_job_resources(*record);
    } else {
      record->snapshot.state = job_state_e::canceling;
      record->snapshot.progress.phase = "canceling";
      if (impl_->active_stop) {
        impl_->active_stop->request_stop();
      }
    }
    std::string persist_error;
    if (!impl_->persist_locked(*record, persist_error)) {
      return {
        .code = error_code_e::io_error,
        .error = "cancel accepted but state persistence failed: " +
                 persist_error,
        .job = record->snapshot,
      };
    }
    const auto reply_snapshot = record->snapshot;
    if (canceled_while_queued) {
      impl_->prune_history_locked();
    }
    impl_->changed.notify_all();
    return {
      .ok = true,
      .job = std::move(reply_snapshot),
    };
  }

  service_reply_t job_service_t::clear(const std::string_view id) {
    std::lock_guard lock {impl_->mutex};
    if (!impl_->started || impl_->stopping) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not running",
      };
    }
    const auto iterator = impl_->jobs.find(std::string {id});
    if (iterator == impl_->jobs.end()) {
      return {
        .code = error_code_e::not_found,
        .error = "offline SBS job was not found",
      };
    }
    const auto record = iterator->second;
    if (!is_terminal(record->snapshot.state)) {
      return {
        .code = error_code_e::conflict,
        .error = "an active offline SBS job must be canceled before it can be cleared",
        .job = record->snapshot,
      };
    }

    std::string clear_error;
    if (!impl_->clear_terminal_record_locked(record, clear_error)) {
      return {
        .code = error_code_e::io_error,
        .error = "offline SBS job could not be cleared safely: " + clear_error,
        .job = record->snapshot,
      };
    }
    impl_->changed.notify_all();
    return {
      .ok = true,
      .job = record->snapshot,
    };
  }

  service_reply_t job_service_t::get(const std::string_view id) const {
    std::lock_guard lock {impl_->mutex};
    if (!impl_->started) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not running",
      };
    }
    const auto iterator = impl_->jobs.find(std::string {id});
    if (iterator == impl_->jobs.end()) {
      return {
        .code = error_code_e::not_found,
        .error = "offline SBS job was not found",
      };
    }
    return {
      .ok = true,
      .job = iterator->second->snapshot,
    };
  }

  browse_reply_t job_service_t::browse(
    const browse_request_t &request
  ) const {
    std::unique_lock browse_lock {impl_->browse_mutex, std::try_to_lock};
    if (!browse_lock.owns_lock()) {
      return {
        .code = error_code_e::busy,
        .error = "another host filesystem browse is already in progress",
      };
    }
    std::optional<std::string> expected_user_id;
    {
      std::lock_guard lock {impl_->mutex};
      if (!impl_->started || impl_->stopping) {
        return {
          .code = error_code_e::not_initialized,
          .error = "offline SBS job manager is not running",
        };
      }
      expected_user_id = impl_->config.expected_user_id;
    }
    return browse_host_filesystem(request, expected_user_id);
  }

  std::vector<job_snapshot_t> job_service_t::list() const {
    std::lock_guard lock {impl_->mutex};
    std::vector<job_snapshot_t> values;
    values.reserve(impl_->jobs.size());
    for (const auto &[id, record] : impl_->jobs) {
      // The Web UI polls this collection. Keep it bounded per retained job; the
      // selected-job endpoint supplies its worker result.
      const auto &snapshot = record->snapshot;
      values.push_back({
        .id = snapshot.id,
        .state = snapshot.state,
        .operation = snapshot.operation,
        .input_path = snapshot.input_path,
        .output_path = snapshot.output_path,
        .output_location = snapshot.output_location,
        .codec = snapshot.codec,
        .transient_raster_hard_cap_bytes = snapshot.transient_raster_hard_cap_bytes,
        .progress = snapshot.progress,
        .created_at_unix_ms = snapshot.created_at_unix_ms,
        .started_at_unix_ms = snapshot.started_at_unix_ms,
        .ended_at_unix_ms = snapshot.ended_at_unix_ms,
        .error = snapshot.error,
        .worker_result = nullptr,
      });
    }
    std::ranges::sort(values, [](const auto &left, const auto &right) {
      return left.created_at_unix_ms > right.created_at_unix_ms;
    });
    return values;
  }

  bool job_service_t::has_active_job() const {
    std::lock_guard lock {impl_->mutex};
    return impl_->active_id.has_value();
  }

  std::optional<std::string> job_service_t::active_job_id() const {
    std::lock_guard lock {impl_->mutex};
    return impl_->active_id;
  }

  nlohmann::json job_service_t::capabilities() const {
    std::lock_guard lock {impl_->mutex};
    return {
      {"schema", 2},
      {"available", impl_->started && !impl_->stopping},
      {"implementation", "native-sunshine-child"},
      {"python_dependency", false},
      {"one_active_job", true},
      {"live_stream_active", gpu_workload::live_stream_active()},
      {"offline_gpu_active", gpu_workload::offline_sbs_active()},
      {"restart_resume", false},
      {"restart_recovery", "mark-interrupted"},
      {"operations", {"evaluate", "convert"}},
      {"codecs", impl_->codecs},
      {
        "codec_runtime_preflight",
        "at-conversion-admission-under-exclusive-gpu-lease",
      },
      {"containers", {"mkv", "mp4"}},
      {"pipeline", {
        {"causal_online_logic", true},
        {"single_estimator_renderer_pass", true},
        {"lookahead", false},
        {"scene_cache", false},
        {"replay", false},
        {"playback_pacing", false},
        {"source_order", true},
        {"persistent_encoder", true},
      }},
      {"retention", {
        {"max_jobs", impl_->config.max_retained_jobs},
        {
          "max_artifact_bytes",
          impl_->config.max_retained_artifact_bytes,
        },
        {"published_outputs_count_toward_quota", false},
        {"unattested_staging_is_never_deleted", true},
        {
          "unattested_staging_minimum_charge_bytes",
          retained_staging_minimum_charge_bytes,
        },
      }},
      {"ffmpeg", {
        {"available", !impl_->ffmpeg.empty()},
        {"path", impl_->ffmpeg.empty() ?
                   nlohmann::json(nullptr) :
                   nlohmann::json(path_to_utf8(impl_->ffmpeg))},
        {"version", impl_->ffmpeg_version.empty() ?
                      nlohmann::json(nullptr) :
                      nlohmann::json(impl_->ffmpeg_version)},
        {"path_discovery", "installation-local-or-trusted-host-override"},
      }},
      {"ffprobe", {
        {"available", !impl_->ffprobe.empty()},
        {"path", impl_->ffprobe.empty() ?
                   nlohmann::json(nullptr) :
                   nlohmann::json(path_to_utf8(impl_->ffprobe))},
        {"version", impl_->ffprobe_version.empty() ?
                      nlohmann::json(nullptr) :
                      nlohmann::json(impl_->ffprobe_version)},
        {"path_discovery", "installation-local-or-trusted-host-override"},
      }},
      {"output_security", {
        {"location", "input-directory"},
        {"destination_derived_from_input", true},
        {"request_accepts_basename_only", true},
        {"overwrite", false},
        {"atomic_publish", true},
      }},
      {"active_job_id", impl_->active_id ?
                          nlohmann::json(*impl_->active_id) :
                          nlohmann::json(nullptr)},
    };
  }

  service_config_t default_service_config() {
    const auto user_data = platf::user_local_appdata();
    const auto user_root =
      user_data.empty() ?
        fs::path {} :
        user_data / "Sunshine 3D" / "offline-sbs";
    const auto active_user = platf::active_user_id();
    const bool valid_user_id =
      active_user &&
      !active_user->empty() &&
      std::ranges::all_of(*active_user, [](const unsigned char character) {
        return std::isalnum(character) ||
               character == '-' ||
               character == '_';
      });
    const auto state_base = platf::appdata();
    const auto state_root =
      valid_user_id && !state_base.empty() ?
        state_base /
          "offline-sbs" /
          "users" /
          path_from_utf8(*active_user) :
        fs::path {};
    fs::path executable;
    if (
      lifetime::get_argv() &&
      lifetime::get_argv()[0] &&
      *lifetime::get_argv()[0]
    ) {
      executable = fs::absolute(path_from_utf8(lifetime::get_argv()[0]));
    }
    return {
      .state_root = state_root,
      .worker_root = user_root.empty() ? fs::path {} : user_root / "worker",
      .sunshine_executable = std::move(executable),
      .sunshine_config = path_from_utf8(config::sunshine.config_file),
      .expected_user_id = valid_user_id ? active_user : std::nullopt,
    };
  }

  namespace detail {
    service_owner_t::~service_owner_t() {
      shutdown();
    }

    void service_owner_t::configure(service_config_t config) {
      {
        std::lock_guard lock {mutex_};
        if (phase_ != phase_e::unconfigured) {
          return;
        }
        config_.emplace(std::move(config));
        phase_ = phase_e::dormant;
      }
    }

    void service_owner_t::configuration_failed(std::string error) {
      if (error.empty()) {
        error = "offline SBS job manager initialization failed";
      }
      BOOST_LOG(warning) << "Offline SBS conversion is unavailable: "sv << error;
      std::lock_guard lock {mutex_};
      if (phase_ == phase_e::ready || phase_ == phase_e::closing || phase_ == phase_e::closed) {
        return;
      }
      error_ = std::move(error);
      config_.reset();
      phase_ = phase_e::failed;
    }

    void service_owner_t::request() {
      std::string launch_error;
      {
        std::lock_guard lock {mutex_};
        if (phase_ != phase_e::dormant || !config_) {
          return;
        }
        phase_ = phase_e::initializing;
        try {
#ifdef SUNSHINE_TESTS
          if (test_hooks_ && test_hooks_->before_thread_launch) {
            test_hooks_->before_thread_launch();
          }
#endif
          initializer_ = std::jthread([this, config = std::move(*config_)]() mutable {
            initialize_worker(std::move(config));
          });
          config_.reset();
        } catch (const std::exception &exception) {
          launch_error = "offline SBS job manager initialization failed: "s + exception.what();
          error_ = launch_error;
          config_.reset();
          phase_ = phase_e::failed;
        } catch (...) {
          launch_error = "offline SBS job manager initialization failed unexpectedly";
          error_ = launch_error;
          config_.reset();
          phase_ = phase_e::failed;
        }
      }
      if (!launch_error.empty()) {
        BOOST_LOG(warning) << "Offline SBS conversion is unavailable: "sv << launch_error;
      }
    }

    void service_owner_t::initialize_worker(service_config_t config) {
      // Unlike the former std::async future, an owning thread must also catch construction
      // failures before job_service_t::start() is entered.
      try {
#ifdef SUNSHINE_TESTS
        if (test_hooks_ && test_hooks_->observe) {
          test_hooks_->observe(testing::service_initialization_stage_e::worker_enter);
        }
        if (test_hooks_ && test_hooks_->before_service_allocation) {
          test_hooks_->before_service_allocation();
        }
#endif
        auto candidate = std::make_shared<job_service_t>(std::move(config));
        std::string error;
        bool started = false;
        try {
          started = candidate->start(error);
        } catch (const std::exception &exception) {
          error = "offline SBS job manager initialization failed: "s + exception.what();
        } catch (...) {
          error = "offline SBS job manager initialization failed unexpectedly";
        }
#ifdef SUNSHINE_TESTS
        if (test_hooks_ && test_hooks_->observe) {
          test_hooks_->observe(testing::service_initialization_stage_e::service_start_finished);
        }
#endif
        if (!started) {
          configuration_failed(std::move(error));
        } else {
          bool shutdown_requested = false;
          {
            std::lock_guard lock {mutex_};
            shutdown_requested = phase_ == phase_e::closing || phase_ == phase_e::closed;
            if (!shutdown_requested) {
              service_ = candidate;
              error_.clear();
              phase_ = phase_e::ready;
            }
          }
          if (shutdown_requested) {
            candidate->shutdown();
          }
        }
      } catch (const std::exception &exception) {
        configuration_failed("offline SBS job manager initialization failed: "s + exception.what());
      } catch (...) {
        configuration_failed("offline SBS job manager initialization failed unexpectedly");
      }
#ifdef SUNSHINE_TESTS
      if (test_hooks_ && test_hooks_->observe) {
        // A faulty test observer cannot escape the owning thread.
        try {
          test_hooks_->observe(testing::service_initialization_stage_e::worker_exit);
        } catch (...) {
        }
      }
#endif
    }

    std::shared_ptr<job_service_t> service_owner_t::ready_service() const {
      std::lock_guard lock {mutex_};
      return phase_ == phase_e::ready ? service_ : nullptr;
    }

    nlohmann::json service_owner_t::capabilities() const {
      std::shared_ptr<job_service_t> service;
      std::string error;
      bool initializing = false;
      {
        std::lock_guard lock {mutex_};
        if (phase_ == phase_e::ready) {
          service = service_;
        } else {
          initializing = phase_ == phase_e::initializing;
          error = phase_ == phase_e::closing || phase_ == phase_e::closed ?
                    "offline SBS job manager is shutting down" :
                    error_;
        }
      }
      if (service) {
        return service->capabilities();
      }
      return {
        {"schema", 1},
        {"available", false},
        {"initializing", initializing},
        {"python_dependency", false},
        {"error", initializing ? "offline SBS job manager is initializing" : error.empty() ? "offline SBS job manager is not initialized" :
                                                                                             error},
      };
    }

    void service_owner_t::shutdown() {
      std::shared_ptr<job_service_t> service;
      std::jthread initializer;
      {
        std::lock_guard lock {mutex_};
        if (phase_ == phase_e::closed) {
          return;
        }
        phase_ = phase_e::closing;
        config_.reset();
        service = std::move(service_);
        initializer = std::move(initializer_);
      }
      if (service) {
        service->shutdown();
      }
      if (initializer.joinable()) {
        initializer.join();
      }
      std::lock_guard lock {mutex_};
      phase_ = phase_e::closed;
    }
  }  // namespace detail

  namespace {
    detail::service_owner_t global_service_owner;
#ifdef SUNSHINE_TESTS
    thread_local detail::service_owner_t *service_owner_override = nullptr;
#endif

    detail::service_owner_t &facade_owner() {
#ifdef SUNSHINE_TESTS
      if (service_owner_override) {
        return *service_owner_override;
      }
#endif
      return global_service_owner;
    }
  }  // namespace

  void configure_default_service() {
    // Freeze the interactive account and paths now. A late demand cannot rebind the service.
    // Configuration capture used to run inside an unobserved std::async future; keep its
    // failure local to offline availability rather than aborting host startup.
    try {
      facade_owner().configure(default_service_config());
    } catch (const std::exception &exception) {
      facade_owner().configuration_failed(
        "offline SBS job manager configuration failed: "s + exception.what()
      );
    } catch (...) {
      facade_owner().configuration_failed("offline SBS job manager configuration failed unexpectedly");
    }
  }

  void request_initialization() {
    facade_owner().request();
  }

  void shutdown() {
    facade_owner().shutdown();
  }

  service_reply_t create(const create_request_t &request) {
    auto service = facade_owner().ready_service();
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->create(request);
  }

  service_reply_t cancel(const std::string_view id) {
    auto service = facade_owner().ready_service();
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->cancel(id);
  }

  service_reply_t clear(const std::string_view id) {
    auto service = facade_owner().ready_service();
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->clear(id);
  }

  service_reply_t get(const std::string_view id) {
    auto service = facade_owner().ready_service();
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->get(id);
  }

  browse_reply_t browse(const browse_request_t &request) {
    auto service = facade_owner().ready_service();
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->browse(request);
  }

  std::vector<job_snapshot_t> list() {
    auto service = facade_owner().ready_service();
    return service ? service->list() : std::vector<job_snapshot_t> {};
  }

  bool has_active_job() {
    auto service = facade_owner().ready_service();
    return service && service->has_active_job();
  }

  nlohmann::json capabilities() {
    return facade_owner().capabilities();
  }

#ifdef SUNSHINE_TESTS
  namespace testing {
    scoped_service_owner_override_t::scoped_service_owner_override_t(detail::service_owner_t &owner):
        previous_(service_owner_override) {
      service_owner_override = &owner;
    }

    scoped_service_owner_override_t::~scoped_service_owner_override_t() {
      service_owner_override = previous_;
    }

    std::unique_ptr<detail::service_owner_t> service_owner_test_access::create(
      service_config_t config,
      service_initialization_hooks_t hooks
    ) {
      auto owner = std::make_unique<detail::service_owner_t>();
      owner->test_hooks_ = std::make_shared<const service_initialization_hooks_t>(std::move(hooks));
      owner->configure(std::move(config));
      return owner;
    }

    std::unique_ptr<detail::service_owner_t> service_owner_test_access::start(
      service_config_t config,
      service_initialization_hooks_t hooks
    ) {
      auto owner = create(std::move(config), std::move(hooks));
      owner->request();
      return owner;
    }

    void service_owner_test_access::join(detail::service_owner_t &owner) {
      // Tests serialize this observational join with request and shutdown.
      if (owner.initializer_.joinable()) {
        owner.initializer_.join();
      }
    }
  }  // namespace testing

#endif
}  // namespace offline_sbs

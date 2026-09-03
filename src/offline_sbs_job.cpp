/**
 * @file src/offline_sbs_job.cpp
 * @brief Native offline SBS job lifecycle, persistence, and child supervision.
 */

#include "offline_sbs_job.h"

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
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "crypto.h"
#include "entry_handler.h"
#include "gpu_workload_arbiter.h"
#include "offline_sbs_contract.h"
#include "offline_sbs_filesystem.h"
#include "offline_sbs_wire_contract.h"
#include "platform/common.h"
#include "uuid.h"

#include <boost/filesystem/path.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/process/v1/group.hpp>

#include "logging.h"

#ifdef _WIN32
  #include <fcntl.h>
  #include <io.h>
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>

  #ifndef AT_EMPTY_PATH
    #define AT_EMPTY_PATH 0x1000
  #endif
#endif

using namespace std::literals;

namespace offline_sbs {
  namespace fs = std::filesystem;
  namespace bp = boost::process::v1;

  namespace {
    constexpr std::uint64_t min_cache_bytes = 16ull * 1024ull * 1024ull;
    constexpr std::uint64_t max_cache_bytes = 64ull * 1024ull * 1024ull * 1024ull;
    // Unattested staging files are intentionally never deleted automatically.
    // Charge at least 1 MiB apiece so repeated zero-byte failures cannot grow
    // an unbounded retained staging set while consuming no byte quota.
    constexpr std::uint64_t retained_staging_minimum_charge_bytes =
      1ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_contract_bytes = 16ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_scene_audit_bytes = 32ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_progress_contract_bytes = 256ull * 1024ull;
    constexpr std::size_t max_current_scene_bytes = 16ull * 1024ull;
    constexpr std::size_t max_scene_count = max_serialized_scene_count;
    constexpr std::size_t max_progress_scene_decisions = 64;
    constexpr std::size_t max_browse_entries = 4096;
    constexpr std::size_t max_browse_path_components = 256;
    constexpr std::size_t max_browse_response_bytes = 512ull * 1024ull;
    constexpr auto max_browse_duration = std::chrono::seconds(2);
    constexpr auto media_probe_timeout = std::chrono::seconds(10);
    constexpr auto worker_termination_timeout = std::chrono::seconds(5);
    // The outer manager captures the native worker's merged stdout/stderr. Keep the
    // retained diagnostic strictly below the much larger artifact quota even when a
    // broken codec or logger emits continuously for the 12-hour worker allowance.
    constexpr std::size_t max_native_worker_log_bytes = 1ull * 1024ull * 1024ull;
    constexpr std::string_view native_worker_log_limit_marker =
      "\n[Sunshine 3D stopped this offline worker because its diagnostic "
      "output exceeded the bounded log limit.]\n";
    constexpr std::size_t max_native_worker_log_payload_bytes =
      max_native_worker_log_bytes - native_worker_log_limit_marker.size();

    static_assert(max_contract_bytes == worker_result_max_bytes);
    static_assert(max_scene_audit_bytes == scene_audit_max_bytes);
    static_assert(
      native_worker_log_limit_marker.size() < max_native_worker_log_bytes
    );

    const char *to_string(operation_e value) {
      switch (value) {
        case operation_e::evaluate:
          return "evaluate";
        case operation_e::convert:
          return "convert";
      }
      return "unknown";
    }

    const char *to_string(output_location_e value) {
      switch (value) {
        case output_location_e::legacy_managed_exports:
          return "legacy-managed-exports";
        case output_location_e::input_directory:
          return "input-directory";
      }
      return "unknown";
    }

    const char *to_string(cache_budget_policy_e value) {
      switch (value) {
        case cache_budget_policy_e::fail:
          return "fail";
        case cache_budget_policy_e::split:
          return "split";
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
      if (value == "legacy-managed-exports") {
        return output_location_e::legacy_managed_exports;
      }
      if (value == "input-directory") {
        return output_location_e::input_directory;
      }
      return std::nullopt;
    }

    std::optional<cache_budget_policy_e> parse_budget_policy(
      const std::string_view value
    ) {
      if (value == "fail") {
        return cache_budget_policy_e::fail;
      }
      if (value == "split") {
        return cache_budget_policy_e::split;
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

    std::string path_to_utf8(const fs::path &path) {
      const auto value = path.generic_u8string();
      return {
        reinterpret_cast<const char *>(value.data()),
        value.size(),
      };
    }

    fs::path path_from_utf8(const std::string &value) {
      std::u8string utf8;
      utf8.assign(
        reinterpret_cast<const char8_t *>(value.data()),
        reinterpret_cast<const char8_t *>(value.data() + value.size())
      );
      return fs::path(utf8);
    }

#ifndef _WIN32
    bool is_already_exists_error(const std::error_code &error) {
      return error == std::errc::file_exists;
    }
#endif

#ifdef _WIN32
    class directory_creation_pins_t {
    public:
      directory_creation_pins_t() = default;

      ~directory_creation_pins_t() {
        close();
      }

      directory_creation_pins_t(const directory_creation_pins_t &) = delete;
      directory_creation_pins_t &operator=(
        const directory_creation_pins_t &
      ) = delete;

      directory_creation_pins_t(
        directory_creation_pins_t &&other
      ) noexcept {
        handles_.swap(other.handles_);
      }

      directory_creation_pins_t &operator=(
        directory_creation_pins_t &&other
      ) noexcept {
        if (this != &other) {
          close();
          handles_.swap(other.handles_);
        }
        return *this;
      }

      void retain(const HANDLE handle) {
        try {
          handles_.push_back(handle);
        } catch (...) {
          CloseHandle(handle);
          throw;
        }
      }

    private:
      void close() noexcept {
        for (const auto handle : handles_) {
          CloseHandle(handle);
        }
        handles_.clear();
      }

      std::vector<HANDLE> handles_;
    };

    directory_creation_pins_t create_user_directory(
#else
    void create_user_directory(
#endif
      const fs::path &directory,
      const std::string_view description
    ) {
#ifdef _WIN32
      // MinGW's recursive create_directories() can incorrectly return the
      // native ERROR_ALREADY_EXISTS for an existing intermediate component
      // while leaving the requested leaf absent. Create and validate each
      // component explicitly so ALREADY_EXISTS is attributed to the object
      // that produced it. Every existing component must be a real directory,
      // never a symlink, junction, or other reparse point.
      directory_creation_pins_t pins;
      auto current = directory.root_path();
      for (const auto &component : directory.relative_path()) {
        if (component == L".") {
          continue;
        }
        if (component == L"..") {
          throw std::runtime_error(
            std::string {description} +
            " must not contain parent traversal"
          );
        }
        current /= component;

        bool ready = false;
        for (int attempt = 0; attempt < 8 && !ready; ++attempt) {
          const auto open_component = [&]() {
            const HANDLE handle = CreateFileW(
              current.c_str(),
              FILE_READ_ATTRIBUTES,
              FILE_SHARE_READ | FILE_SHARE_WRITE,
              nullptr,
              OPEN_EXISTING,
              FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
              nullptr
            );
            if (handle == INVALID_HANDLE_VALUE) {
              const auto open_error = GetLastError();
              if (
                open_error == ERROR_FILE_NOT_FOUND ||
                open_error == ERROR_PATH_NOT_FOUND
              ) {
                return INVALID_HANDLE_VALUE;
              }
              throw std::runtime_error(
                "cannot inspect " + std::string {description} +
                " component [" + path_to_utf8(current) +
                "]: " +
                std::error_code {
                  static_cast<int>(open_error),
                  std::system_category(),
                }.message()
              );
            }
            BY_HANDLE_FILE_INFORMATION information {};
            const BOOL inspected =
              GetFileInformationByHandle(handle, &information);
            const auto inspect_error =
              inspected ? ERROR_SUCCESS : GetLastError();
            if (!inspected) {
              CloseHandle(handle);
              throw std::runtime_error(
                "cannot inspect existing " + std::string {description} +
                " component [" + path_to_utf8(current) +
                "]: " +
                std::error_code {
                  static_cast<int>(inspect_error),
                  std::system_category(),
                }.message()
              );
            }
            if (
              (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
              (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
            ) {
              CloseHandle(handle);
              throw std::runtime_error(
                std::string {description} +
                " path already exists but is not a real directory [" +
                path_to_utf8(current) + "]"
              );
            }
            return handle;
          };

          if (const auto handle = open_component();
              handle != INVALID_HANDLE_VALUE) {
            // Denying delete sharing keeps this exact ancestor from being
            // renamed or replaced while descendants are created and while
            // the service's long-lived managed-root pins are acquired.
            pins.retain(handle);
            ready = true;
            continue;
          }
          if (CreateDirectoryW(current.c_str(), nullptr)) {
            if (const auto handle = open_component();
                handle != INVALID_HANDLE_VALUE) {
              pins.retain(handle);
              ready = true;
            }
            continue;
          }
          const auto create_error = GetLastError();
          if (
            create_error == ERROR_ALREADY_EXISTS ||
            create_error == ERROR_FILE_EXISTS
          ) {
            // A racing creator: inspect and retain the exact identity now.
            if (const auto handle = open_component();
                handle != INVALID_HANDLE_VALUE) {
              pins.retain(handle);
              ready = true;
            }
            continue;
          }
          throw std::runtime_error(
            "cannot create " + std::string {description} +
            " component [" + path_to_utf8(current) +
            "]: " +
            std::error_code {
              static_cast<int>(create_error),
              std::system_category(),
            }.message()
          );
        }
        if (!ready) {
          throw std::runtime_error(
            "cannot stabilize " + std::string {description} +
            " component after repeated create races [" +
            path_to_utf8(current) + "]"
          );
        }
      }
      return pins;
#else
      std::error_code create_error;
      fs::create_directories(directory, create_error);
      if (!create_error) {
        return;
      }

      // Re-stat while the caller is still impersonating the bound standard
      // user. ERROR_ALREADY_EXISTS is benign only when the exact requested
      // object is an actual directory; a file, symlink, or uninspectable
      // object is a path conflict and must fail closed.
      std::error_code status_error;
      const auto status = fs::symlink_status(directory, status_error);
      bool real_directory =
        !status_error && status.type() == fs::file_type::directory;
#ifdef _WIN32
      if (real_directory) {
        const auto attributes = GetFileAttributesW(directory.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES) {
          status_error = std::error_code {
            static_cast<int>(GetLastError()),
            std::system_category(),
          };
          real_directory = false;
        } else {
          real_directory =
            (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
        }
      }
#endif
      if (is_already_exists_error(create_error)) {
        if (real_directory) {
          return;
        }
        if (status_error) {
          throw std::runtime_error(
            "cannot verify existing " + std::string {description} + " [" +
            path_to_utf8(directory) + "]: " + status_error.message()
          );
        }
      }
      if (
        !real_directory &&
        !status_error &&
        status.type() != fs::file_type::not_found &&
        status.type() != fs::file_type::none
      ) {
        // MinGW may report not_a_directory instead of file_exists when the
        // target itself is a file. The observed object type is authoritative
        // for this conflict, but a real directory with an unrelated create
        // failure must retain the original access/I/O diagnostic.
        throw std::runtime_error(
          std::string {description} + " path already exists but is not a "
          "real directory [" + path_to_utf8(directory) + "]"
        );
      }
      throw std::runtime_error(
        "cannot create " + std::string {description} + " [" +
        path_to_utf8(directory) + "]: " + create_error.message()
      );
#endif
    }

    std::string sha256_hex(const std::string_view bytes) {
      static constexpr char digits[] = "0123456789abcdef";
      const auto digest = crypto::hash(bytes);
      std::string result;
      result.resize(digest.size() * 2);
      for (std::size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = digits[digest[index] >> 4u];
        result[index * 2 + 1] = digits[digest[index] & 0x0fu];
      }
      return result;
    }

    bool run_bound_user_filesystem_action(
      const std::optional<std::string> &expected_user_id,
      const std::function<void()> &action,
      std::string &error
    ) {
      std::exception_ptr exception;
      const auto impersonation_error = platf::run_as_active_user(
        [&]() {
          try {
            action();
          } catch (...) {
            exception = std::current_exception();
          }
        },
        expected_user_id
      );
      if (impersonation_error) {
        error =
          "cannot access the active user's offline workspace: " +
          impersonation_error.message();
        return false;
      }
      if (exception) {
        try {
          std::rethrow_exception(exception);
        } catch (const std::exception &caught) {
          error = caught.what();
        } catch (...) {
          error = "active-user filesystem operation failed";
        }
        return false;
      }
      return true;
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
#ifdef _WIN32
      auto left_native = left.native();
      auto right_native = right.native();
      std::ranges::transform(left_native, left_native.begin(), [](const wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
      });
      std::ranges::transform(right_native, right_native.begin(), [](const wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
      });
      return left_native == right_native;
#else
      return left == right;
#endif
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

    bool regular_nonempty_file(const fs::path &path) {
      std::error_code ec;
      return fs::is_regular_file(path, ec) && !ec &&
             fs::file_size(path, ec) > 0 && !ec;
    }

#ifdef _WIN32
    bool windows_publish_identity_attributes_accepted(
      const DWORD attributes,
      const std::uint64_t size
    ) {
      return size > 0 &&
             (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0 &&
             (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
    }
#endif

    std::optional<nlohmann::json> file_publish_identity(
      const fs::path &path,
      std::string &error
    ) {
#ifdef _WIN32
      const HANDLE handle = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        error =
          "cannot open staged output for identity attestation (Windows error " +
          std::to_string(GetLastError()) + ")";
        return std::nullopt;
      }
      BY_HANDLE_FILE_INFORMATION information {};
      const BOOL inspected = GetFileInformationByHandle(handle, &information);
      const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
      CloseHandle(handle);
      if (!inspected) {
        error =
          "cannot inspect output identity (Windows error " +
          std::to_string(inspect_error) + ")";
        return std::nullopt;
      }
      const auto size =
        (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
        information.nFileSizeLow;
      if (!windows_publish_identity_attributes_accepted(
            information.dwFileAttributes,
            size
          )) {
        error =
          size == 0 ?
            "staged output is empty" :
            "cannot attest a directory or reparse-point output";
        return std::nullopt;
      }
      return nlohmann::json {
        {"volume_serial", information.dwVolumeSerialNumber},
        {"file_index_high", information.nFileIndexHigh},
        {"file_index_low", information.nFileIndexLow},
        {"size_bytes", size},
      };
#else
      const int fd = ::open(
        path.c_str(),
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW
      );
      if (fd < 0) {
        error =
          "cannot open output for identity attestation: " +
          std::string {std::strerror(errno)};
        return std::nullopt;
      }
      struct stat information {};
      if (::fstat(fd, &information) != 0) {
        const auto inspect_error = errno;
        ::close(fd);
        error =
          "cannot attest output identity: " +
          std::string {std::strerror(inspect_error)};
        return std::nullopt;
      }
      if (!S_ISREG(information.st_mode) || information.st_size <= 0) {
        ::close(fd);
        error = "output is not a non-empty regular file";
        return std::nullopt;
      }
      const auto identity = nlohmann::json {
        {"device", static_cast<std::uint64_t>(information.st_dev)},
        {"inode", static_cast<std::uint64_t>(information.st_ino)},
        {"size_bytes", static_cast<std::uint64_t>(information.st_size)},
      };
      ::close(fd);
      return identity;
#endif
    }

    bool file_matches_publish_identity(
      const fs::path &path,
      const nlohmann::json &identity
    ) {
      std::string error;
      const auto actual = file_publish_identity(path, error);
      return actual && *actual == identity;
    }

    class publication_file_t {
    public:
      ~publication_file_t() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) {
          ::close(fd_);
        }
#endif
      }

      publication_file_t(const publication_file_t &) = delete;
      publication_file_t &operator=(const publication_file_t &) = delete;

      publication_file_t(publication_file_t &&other) noexcept:
          source_(std::move(other.source_)),
          identity_(std::move(other.identity_))
#ifdef _WIN32
          , handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE))
#else
          , fd_(std::exchange(other.fd_, -1))
#endif
      {
      }

      publication_file_t &operator=(publication_file_t &&other) noexcept {
        if (this != &other) {
#ifdef _WIN32
          if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
          }
          handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
#else
          if (fd_ >= 0) {
            ::close(fd_);
          }
          fd_ = std::exchange(other.fd_, -1);
#endif
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
#ifdef _WIN32
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
#else
        const int fd = ::open(
          source.c_str(),
          O_RDONLY | O_CLOEXEC | O_NOFOLLOW
        );
        if (fd < 0) {
          if (missing && errno == ENOENT) {
            *missing = true;
          }
          error =
            "cannot pin the staged output for publication: " +
            std::string {std::strerror(errno)};
          return std::nullopt;
        }
        struct stat information {};
        if (
          ::fstat(fd, &information) != 0 ||
          !S_ISREG(information.st_mode) ||
          information.st_size <= 0
        ) {
          const auto inspect_error = errno;
          ::close(fd);
          error =
            "staged output is not a non-empty, non-link regular file: " +
            std::string {std::strerror(inspect_error)};
          return std::nullopt;
        }
        nlohmann::json identity {
          {"device", static_cast<std::uint64_t>(information.st_dev)},
          {"inode", static_cast<std::uint64_t>(information.st_ino)},
          {"size_bytes", static_cast<std::uint64_t>(information.st_size)},
        };
        if (expected_identity && identity != *expected_identity) {
          ::close(fd);
          error = "staged output identity changed after worker attestation";
          return std::nullopt;
        }
        return publication_file_t {
          source,
          std::move(identity),
          fd
        };
#endif
      }

      bool publish_no_replace(
        const fs::path &destination,
        std::string &error,
        bool *staging_retired = nullptr,
        const bool force_staging_disposition_failure = false
      ) {
        if (staging_retired) {
          *staging_retired = false;
        }
#ifdef _WIN32
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
          return false;
        }
        std::string retirement_error;
        const bool retired = retire_staging_link(
          retirement_error,
          force_staging_disposition_failure
        );
        if (staging_retired) {
          *staging_retired = retired;
        }
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
        return true;
#else
        (void) force_staging_disposition_failure;
        // AT_EMPTY_PATH binds the new destination link to this exact open inode,
        // while linkat's normal EEXIST behavior provides atomic no-replace
        // publication. Do not unlink the old pathname by name afterward: a
        // same-user rename race could otherwise make cleanup delete an unrelated
        // file. Linux currently exposes this feature as unavailable, so the safe
        // retained staging link is preferable to an unsafe cleanup fallback.
        if (
          ::linkat(
            fd_,
            "",
            AT_FDCWD,
            destination.c_str(),
            AT_EMPTY_PATH
          ) != 0
        ) {
          const auto publish_error = errno;
          if (publish_error == EEXIST) {
            error =
              "final output appeared while the job was running; refusing to "
              "overwrite it";
          } else {
            error =
              "cannot atomically publish the pinned conversion output: " +
              std::string {std::strerror(publish_error)};
          }
          return false;
        }
        return true;
#endif
      }

      bool retire_staging_link(
        std::string &error,
        const bool force_failure = false
      ) {
#ifdef _WIN32
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
#else
        (void) force_failure;
        error =
          "exact staging-link retirement is unavailable on this platform";
        return false;
#endif
      }

    private:
#ifdef _WIN32
      publication_file_t(
        fs::path source,
        nlohmann::json identity,
        const HANDLE handle
      ):
          source_(std::move(source)),
          identity_(std::move(identity)),
          handle_(handle) {
      }
#else
      publication_file_t(
        fs::path source,
        nlohmann::json identity,
        const int fd
      ):
          source_(std::move(source)),
          identity_(std::move(identity)),
          fd_(fd) {
      }
#endif

      fs::path source_;
      nlohmann::json identity_;
#ifdef _WIN32
      HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
      int fd_ = -1;
#endif
    };

    bool validate_local_path_syntax(
      const fs::path &path,
      const std::string_view purpose,
      std::string &error,
      const bool check_drive_type = true
    ) {
      if (path.empty()) {
        error = std::string {purpose} + " must be absolute";
        return false;
      }
      const auto &native = path.native();
      if (native.find(fs::path::value_type {}) != fs::path::string_type::npos) {
        error = std::string {purpose} + " contains an embedded null";
        return false;
      }
#ifdef _WIN32
      // MinGW's std::filesystem::path::is_absolute() does not classify every
      // Windows UNC spelling consistently. Check the required drive-root form
      // directly so UNC/device/extended paths always reach the same fail-closed
      // contract and diagnostic.
      if (
        native.size() < 3 ||
        !std::iswalpha(native[0]) ||
        native[1] != L':' ||
        (native[2] != L'\\' && native[2] != L'/')
      ) {
        error =
          std::string {purpose} +
          " must use a local drive (UNC, device, and extended paths are rejected)";
        return false;
      }
      const std::array<wchar_t, 4> drive_root {
        static_cast<wchar_t>(std::towupper(native[0])),
        L':',
        L'\\',
        L'\0',
      };
      const auto drive_type = check_drive_type ?
                                GetDriveTypeW(drive_root.data()) :
                                DRIVE_FIXED;
      if (drive_type != DRIVE_FIXED && drive_type != DRIVE_REMOVABLE) {
        error =
          std::string {purpose} +
          " must be on a fixed or removable local drive";
        return false;
      }
#else
      if (!path.is_absolute()) {
        error = std::string {purpose} + " must be absolute";
        return false;
      }
      if (native.starts_with("//")) {
        error = "network " + std::string {purpose} + "s are not accepted";
        return false;
      }
#endif
      if (std::ranges::any_of(
            path.relative_path(),
            [](const fs::path &component) {
              return component == fs::path {".."};
            }
          )) {
        error = std::string {purpose} + " must not contain parent traversal";
        return false;
      }
      return true;
    }

    bool validate_local_input_path_syntax(
      const fs::path &path,
      std::string &error
    ) {
      return validate_local_path_syntax(path, "input path", error);
    }

#ifdef _WIN32
    struct pinned_identity_t {
      DWORD volume_serial = 0;
      DWORD file_index_high = 0;
      DWORD file_index_low = 0;
      bool directory = false;

      [[nodiscard]] bool matches(
        const BY_HANDLE_FILE_INFORMATION &information
      ) const {
        return
          volume_serial == information.dwVolumeSerialNumber &&
          file_index_high == information.nFileIndexHigh &&
          file_index_low == information.nFileIndexLow &&
          directory ==
            ((information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
      }
    };

    class pinned_handle_t {
    public:
      pinned_handle_t(
        HANDLE handle,
        fs::path path,
        const BY_HANDLE_FILE_INFORMATION &information
      ):
          handle_(handle),
          path_(std::move(path)),
          identity_ {
            .volume_serial = information.dwVolumeSerialNumber,
            .file_index_high = information.nFileIndexHigh,
            .file_index_low = information.nFileIndexLow,
            .directory =
              (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0,
          } {
      }

      ~pinned_handle_t() {
        if (handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(handle_);
        }
      }

      pinned_handle_t(const pinned_handle_t &) = delete;
      pinned_handle_t &operator=(const pinned_handle_t &) = delete;

      pinned_handle_t(pinned_handle_t &&other) noexcept:
          handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
          path_(std::move(other.path_)),
          identity_(other.identity_) {
      }

      pinned_handle_t &operator=(pinned_handle_t &&other) noexcept {
        if (this != &other) {
          if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
          }
          handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
          path_ = std::move(other.path_);
          identity_ = other.identity_;
        }
        return *this;
      }

      [[nodiscard]] bool revalidate(std::string &error) const {
        const HANDLE observed = CreateFileW(
          path_.c_str(),
          FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
          nullptr,
          OPEN_EXISTING,
          FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
          nullptr
        );
        if (observed == INVALID_HANDLE_VALUE) {
          error =
            "managed path identity changed or became inaccessible (Windows error " +
            std::to_string(GetLastError()) + ")";
          return false;
        }
        BY_HANDLE_FILE_INFORMATION information {};
        const BOOL inspected =
          GetFileInformationByHandle(observed, &information);
        const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
        CloseHandle(observed);
        if (
          !inspected ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
          !identity_.matches(information)
        ) {
          error =
            "managed path identity or reparse status changed (Windows error " +
            std::to_string(inspect_error) + ")";
          return false;
        }
        return true;
      }

    private:
      HANDLE handle_ = INVALID_HANDLE_VALUE;
      fs::path path_;
      pinned_identity_t identity_;
    };

    enum class pinned_leaf_kind_e {
      regular_file,
      directory,
    };

    std::optional<pinned_handle_t> pin_path_component(
      const fs::path &path,
      const pinned_leaf_kind_e kind,
      const bool lock_file_contents,
      const std::string_view purpose,
      std::string &error
    ) {
      const bool regular_file = kind == pinned_leaf_kind_e::regular_file;
      const HANDLE handle = CreateFileW(
        path.c_str(),
        regular_file && lock_file_contents ?
          GENERIC_READ :
          (
            !regular_file && lock_file_contents ?
              (FILE_READ_ATTRIBUTES | DELETE) :
              FILE_READ_ATTRIBUTES
          ),
        regular_file && lock_file_contents ?
          FILE_SHARE_READ :
          (FILE_SHARE_READ | FILE_SHARE_WRITE),
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        error =
          "cannot pin " + std::string {purpose} +
          " component (Windows error " +
          std::to_string(GetLastError()) + ")";
        return std::nullopt;
      }
      BY_HANDLE_FILE_INFORMATION information {};
      const BOOL inspected = GetFileInformationByHandle(handle, &information);
      const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
      const bool is_directory =
        (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
      if (
        !inspected ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
        is_directory == regular_file
      ) {
        CloseHandle(handle);
        error =
          std::string {purpose} +
          (
            inspected &&
                (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ?
              " must not traverse a symbolic link, junction, or other reparse point" :
              " has an invalid path-component type (Windows error " +
                std::to_string(inspect_error) + ")"
          );
        return std::nullopt;
      }
      return pinned_handle_t {handle, path, information};
    }

    std::optional<std::vector<pinned_handle_t>> pin_no_reparse_components_impl(
      const fs::path &path,
      const pinned_leaf_kind_e leaf_kind,
      const bool lock_leaf_contents,
      const std::string_view purpose,
      std::string &error
    ) {
      std::vector<pinned_handle_t> pins;
      const auto relative = path.relative_path();
      std::vector<fs::path> components;
      components.reserve(
        static_cast<std::size_t>(
          std::distance(relative.begin(), relative.end())
        )
      );
      for (const auto &component : relative) {
        if (component == L".") {
          continue;
        }
        if (component == L"..") {
          error = std::string {purpose} + " must not contain parent traversal";
          return std::nullopt;
        }
        components.push_back(component);
      }
      if (components.empty()) {
        error = std::string {purpose} + " contains no path component";
        return std::nullopt;
      }

      auto current = path.root_path();
      for (std::size_t index = 0; index < components.size(); ++index) {
        const auto &component = components[index];
        current /= component;
        const bool is_leaf = index + 1 == components.size();
        auto pin = pin_path_component(
          current,
          is_leaf ? leaf_kind : pinned_leaf_kind_e::directory,
          is_leaf && lock_leaf_contents,
          purpose,
          error
        );
        if (!pin) {
          return std::nullopt;
        }
        pins.emplace_back(std::move(*pin));
      }
      return pins;
    }

    std::optional<std::vector<pinned_handle_t>> pin_no_reparse_components(
      const fs::path &path,
      std::string &error
    ) {
      return pin_no_reparse_components_impl(
        path,
        pinned_leaf_kind_e::regular_file,
        true,
        "input path",
        error
      );
    }

    std::optional<std::vector<pinned_handle_t>> pin_managed_root_components(
      const fs::path &path,
      const std::string_view purpose,
      std::string &error
    ) {
      return pin_no_reparse_components_impl(
        path,
        pinned_leaf_kind_e::directory,
        false,
        purpose,
        error
      );
    }

    std::optional<pinned_handle_t> pin_ephemeral_root_child(
      const fs::path &root,
      const std::string_view purpose,
      std::string &error
    ) {
      for (int attempt = 0; attempt < 32; ++attempt) {
        const auto path =
          root /
          (
            ".sunshine3d-root-" +
            uuid_util::uuid_t::generate().string() +
            ".lock"
          );
        const HANDLE handle = CreateFileW(
          path.c_str(),
          GENERIC_READ | DELETE,
          FILE_SHARE_READ,
          nullptr,
          CREATE_NEW,
          FILE_ATTRIBUTE_HIDDEN |
            FILE_ATTRIBUTE_TEMPORARY |
            FILE_FLAG_DELETE_ON_CLOSE,
          nullptr
        );
        if (handle == INVALID_HANDLE_VALUE) {
          const auto open_error = GetLastError();
          if (
            open_error == ERROR_FILE_EXISTS ||
            open_error == ERROR_ALREADY_EXISTS
          ) {
            continue;
          }
          error =
            "cannot create " + std::string {purpose} +
            " child identity lock (Windows error " +
            std::to_string(open_error) + ")";
          return std::nullopt;
        }
        BY_HANDLE_FILE_INFORMATION information {};
        if (
          !GetFileInformationByHandle(handle, &information) ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
        ) {
          const auto inspect_error = GetLastError();
          CloseHandle(handle);
          error =
            "cannot inspect " + std::string {purpose} +
            " child identity lock (Windows error " +
            std::to_string(inspect_error) + ")";
          return std::nullopt;
        }
        return pinned_handle_t {handle, path, information};
      }
      error =
        "cannot reserve a unique " + std::string {purpose} +
        " child identity lock";
      return std::nullopt;
    }

    std::optional<pinned_handle_t> pin_managed_directory(
      const fs::path &path,
      const std::string_view purpose,
      std::string &error
    ) {
      return pin_path_component(
        path,
        pinned_leaf_kind_e::directory,
        true,
        purpose,
        error
      );
    }

    bool revalidate_pins(
      const std::vector<pinned_handle_t> &pins,
      const std::string_view purpose,
      std::string &error
    ) {
      for (const auto &pin : pins) {
        if (!pin.revalidate(error)) {
          error = std::string {purpose} + ": " + error;
          return false;
        }
      }
      return true;
    }

    std::optional<std::vector<pinned_handle_t>> pin_browse_directory(
      const fs::path &path,
      std::string &error
    ) {
      const auto normalized = path.lexically_normal();
      if (normalized == normalized.root_path()) {
        auto root = pin_path_component(
          normalized,
          pinned_leaf_kind_e::directory,
          false,
          "browse path",
          error
        );
        if (!root) {
          return std::nullopt;
        }
        std::vector<pinned_handle_t> pins;
        pins.emplace_back(std::move(*root));
        return pins;
      }
      return pin_no_reparse_components_impl(
        normalized,
        pinned_leaf_kind_e::directory,
        false,
        "browse path",
        error
      );
    }

    std::optional<std::vector<pinned_handle_t>> pin_publication_directory(
      const fs::path &path,
      std::string &error
    ) {
      const auto normalized = path.lexically_normal();
      if (normalized == normalized.root_path()) {
        auto root = pin_path_component(
          normalized,
          pinned_leaf_kind_e::directory,
          false,
          "offline output directory",
          error
        );
        if (!root) {
          return std::nullopt;
        }
        std::vector<pinned_handle_t> pins;
        pins.emplace_back(std::move(*root));
        return pins;
      }
      return pin_no_reparse_components_impl(
        normalized,
        pinned_leaf_kind_e::directory,
        false,
        "offline output directory",
        error
      );
    }
#endif

    std::optional<std::uint64_t> retained_staging_charge(
      const fs::path &path,
      std::string &error
    ) {
#ifdef _WIN32
      const HANDLE handle = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        const auto open_error = GetLastError();
        if (
          open_error == ERROR_FILE_NOT_FOUND ||
          open_error == ERROR_PATH_NOT_FOUND
        ) {
          return 0;
        }
        error =
          "cannot inspect retained input-directory staging output (Windows error " +
          std::to_string(open_error) + ")";
        return std::nullopt;
      }
      BY_HANDLE_FILE_INFORMATION information {};
      const BOOL inspected = GetFileInformationByHandle(handle, &information);
      const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
      if (
        !inspected ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
      ) {
        CloseHandle(handle);
        error =
          "retained input-directory staging output is not a plain file "
          "(Windows error " + std::to_string(inspect_error) + ")";
        return std::nullopt;
      }
      auto parent_pins = pin_publication_directory(path.parent_path(), error);
      if (!parent_pins) {
        CloseHandle(handle);
        return std::nullopt;
      }
      const HANDLE observed = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr
      );
      BY_HANDLE_FILE_INFORMATION observed_information {};
      const bool same_leaf =
        observed != INVALID_HANDLE_VALUE &&
        GetFileInformationByHandle(observed, &observed_information) &&
        (observed_information.dwFileAttributes &
           (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0 &&
        observed_information.dwVolumeSerialNumber ==
          information.dwVolumeSerialNumber &&
        observed_information.nFileIndexHigh == information.nFileIndexHigh &&
        observed_information.nFileIndexLow == information.nFileIndexLow;
      if (observed != INVALID_HANDLE_VALUE) {
        CloseHandle(observed);
      }
      if (!same_leaf) {
        CloseHandle(handle);
        error =
          "retained input-directory staging output changed during accounting";
        return std::nullopt;
      }
      if (!revalidate_pins(
            *parent_pins,
            "offline output directory during staging accounting",
            error
          )) {
        CloseHandle(handle);
        return std::nullopt;
      }
      CloseHandle(handle);
      const auto size =
        (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
        information.nFileSizeLow;
#else
      struct stat information {};
      if (::lstat(path.c_str(), &information) != 0) {
        if (errno == ENOENT) {
          return 0;
        }
        error =
          "cannot inspect retained input-directory staging output: " +
          std::string {std::strerror(errno)};
        return std::nullopt;
      }
      if (!S_ISREG(information.st_mode)) {
        error = "retained input-directory staging output is not a plain file";
        return std::nullopt;
      }
      const auto size = static_cast<std::uint64_t>(information.st_size);
#endif
      return std::max(size, retained_staging_minimum_charge_bytes);
    }

    struct browse_entry_record_t {
      std::string name;
      fs::path path;
      bool directory = false;
    };

    nlohmann::json browse_entry_json(
      const browse_entry_record_t &record,
      const browse_type_e type
    ) {
      const bool selectable =
        type == browse_type_e::any ||
        (type == browse_type_e::directory ? record.directory : !record.directory);
      return {
        {"name", record.name},
        {"path", path_to_utf8(record.path)},
        {"type", record.directory ? "directory" : "file"},
        {"selectable", selectable},
      };
    }

    std::optional<bool> inspect_browse_entry(const fs::path &path) {
#ifdef _WIN32
      const HANDLE handle = CreateFileW(
        path.c_str(),
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        return std::nullopt;
      }
      auto handle_guard = util::fail_guard([&]() {
        CloseHandle(handle);
      });
      BY_HANDLE_FILE_INFORMATION information {};
      if (
        !GetFileInformationByHandle(handle, &information) ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
      ) {
        return std::nullopt;
      }
      return
        (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
      std::error_code error;
      const auto status = fs::symlink_status(path, error);
      if (
        error || fs::is_symlink(status) ||
        (!fs::is_directory(status) && !fs::is_regular_file(status))
      ) {
        return std::nullopt;
      }
      return fs::is_directory(status);
#endif
    }

    void sort_browse_entries(std::vector<browse_entry_record_t> &entries) {
      std::ranges::sort(entries, [](const auto &left, const auto &right) {
        if (left.directory != right.directory) {
          return left.directory;
        }
#ifdef _WIN32
        auto left_folded = left.path.filename().native();
        auto right_folded = right.path.filename().native();
        std::ranges::transform(
          left_folded,
          left_folded.begin(),
          [](const wchar_t character) {
          return static_cast<wchar_t>(std::towlower(character));
        });
        std::ranges::transform(
          right_folded,
          right_folded.begin(),
          [](const wchar_t character) {
          return static_cast<wchar_t>(std::towlower(character));
        });
        if (left_folded != right_folded) {
          return left_folded < right_folded;
        }
#else
        auto left_folded = left.name;
        auto right_folded = right.name;
        std::ranges::transform(
          left_folded,
          left_folded.begin(),
          [](const unsigned char character) {
          return static_cast<char>(std::tolower(character));
        });
        std::ranges::transform(
          right_folded,
          right_folded.begin(),
          [](const unsigned char character) {
          return static_cast<char>(std::tolower(character));
        });
        if (left_folded != right_folded) {
          return left_folded < right_folded;
        }
#endif
        if (left.name != right.name) {
          return left.name < right.name;
        }
        return path_to_utf8(left.path) < path_to_utf8(right.path);
      });
    }

    std::optional<std::vector<browse_entry_record_t>> local_browse_roots(
      std::string &error
    ) {
      std::vector<browse_entry_record_t> roots;
#ifdef _WIN32
      const DWORD required = GetLogicalDriveStringsW(0, nullptr);
      if (required == 0) {
        error =
          "cannot enumerate local drives (Windows error " +
          std::to_string(GetLastError()) + ")";
        return std::nullopt;
      }
      std::vector<wchar_t> buffer(static_cast<std::size_t>(required) + 1, L'\0');
      const DWORD copied = GetLogicalDriveStringsW(
        static_cast<DWORD>(buffer.size()),
        buffer.data()
      );
      if (copied == 0 || copied >= buffer.size()) {
        error =
          "cannot enumerate local drives (Windows error " +
          std::to_string(GetLastError()) + ")";
        return std::nullopt;
      }
      for (
        const wchar_t *root = buffer.data();
        *root != L'\0';
        root += std::wcslen(root) + 1
      ) {
        const auto drive_type = GetDriveTypeW(root);
        if (drive_type != DRIVE_FIXED && drive_type != DRIVE_REMOVABLE) {
          continue;
        }
        const fs::path drive = fs::path {root}.lexically_normal();
        roots.push_back({
          .name = path_to_utf8(drive),
          .path = drive,
          .directory = true,
        });
      }
#else
      roots.push_back({
        .name = "/",
        .path = fs::path {"/"},
        .directory = true,
      });
#endif
      sort_browse_entries(roots);
      return roots;
    }

    std::optional<fs::path> canonical_regular_file(
      const fs::path &path,
      std::string &error
    ) {
      std::error_code ec;
      if (path.empty()) {
        error = "path is empty";
        return std::nullopt;
      }
      const auto absolute = fs::absolute(path, ec);
      if (ec) {
        error = "cannot make path absolute: " + ec.message();
        return std::nullopt;
      }
      const auto canonical = fs::canonical(absolute, ec);
      if (ec || !fs::is_regular_file(canonical, ec) || ec) {
        error = "path is not an existing regular file";
        return std::nullopt;
      }
      return canonical;
    }

    bool write_text_atomically(
      const fs::path &path,
      const std::string &contents,
      std::string &error
    ) {
      auto temporary = path;
      temporary += L".part";
      std::error_code ignored;
      fs::remove(temporary, ignored);
#ifdef _WIN32
      HANDLE handle = CreateFileW(
        temporary.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        error = "cannot create atomic state temporary file";
        return false;
      }
      bool ok = true;
      std::size_t offset = 0;
      while (offset < contents.size()) {
        const auto chunk = static_cast<DWORD>(std::min<std::size_t>(
          contents.size() - offset,
          std::numeric_limits<DWORD>::max()
        ));
        DWORD written = 0;
        if (!WriteFile(handle, contents.data() + offset, chunk, &written, nullptr) ||
            written != chunk) {
          ok = false;
          break;
        }
        offset += written;
      }
      ok = ok && FlushFileBuffers(handle);
      CloseHandle(handle);
      if (
        !ok ||
        !MoveFileExW(
          temporary.c_str(),
          path.c_str(),
          MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        )
      ) {
        fs::remove(temporary, ignored);
        error = "cannot atomically replace durable job state";
        return false;
      }
#else
      const int fd = ::open(
        temporary.c_str(),
        O_CREAT | O_TRUNC | O_WRONLY,
        0600
      );
      if (fd < 0) {
        error = "cannot create atomic state temporary file";
        return false;
      }
      bool ok = true;
      std::size_t offset = 0;
      while (offset < contents.size()) {
        const auto written = ::write(
          fd,
          contents.data() + offset,
          contents.size() - offset
        );
        if (written <= 0) {
          ok = false;
          break;
        }
        offset += static_cast<std::size_t>(written);
      }
      ok = ok && ::fsync(fd) == 0;
      ::close(fd);
      if (!ok) {
        fs::remove(temporary, ignored);
        error = "cannot durably write job state";
        return false;
      }
      fs::rename(temporary, path, ignored);
      if (ignored) {
        fs::remove(temporary, ignored);
        error = "cannot atomically replace durable job state";
        return false;
      }
#endif
      return true;
    }

    bool write_json_atomically(
      const fs::path &path,
      const nlohmann::json &value,
      std::string &error
    ) {
      return write_text_atomically(path, value.dump(2) + "\n", error);
    }

    std::optional<nlohmann::json> read_json_contract(
      const fs::path &path,
      std::string &error,
      const std::uintmax_t max_bytes = max_contract_bytes,
      std::string *content_sha256 = nullptr
    ) {
      std::string serialized;
#ifdef _WIN32
      const HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        error =
          "cannot open contract without following links (Windows error " +
          std::to_string(GetLastError()) + ")";
        return std::nullopt;
      }
      BY_HANDLE_FILE_INFORMATION information {};
      const BOOL inspected = GetFileInformationByHandle(handle, &information);
      const auto inspect_error = inspected ? ERROR_SUCCESS : GetLastError();
      const auto bytes =
        (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
        information.nFileSizeLow;
      if (
        !inspected ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
      ) {
        CloseHandle(handle);
        error =
          "contract is not a no-follow regular file (Windows error " +
          std::to_string(inspect_error) + ")";
        return std::nullopt;
      }
      if (bytes == 0 || bytes > max_bytes) {
        CloseHandle(handle);
        error = "contract size is invalid";
        return std::nullopt;
      }
      serialized.resize(static_cast<std::size_t>(bytes));
      std::size_t offset = 0;
      while (offset < serialized.size()) {
        const auto chunk = static_cast<DWORD>(std::min<std::size_t>(
          serialized.size() - offset,
          std::numeric_limits<DWORD>::max()
        ));
        DWORD read = 0;
        if (
          !ReadFile(handle, serialized.data() + offset, chunk, &read, nullptr) ||
          read == 0
        ) {
          const auto read_error = GetLastError();
          CloseHandle(handle);
          error =
            "cannot read the pinned contract (Windows error " +
            std::to_string(read_error) + ")";
          return std::nullopt;
        }
        offset += read;
      }
      BY_HANDLE_FILE_INFORMATION final_information {};
      if (!GetFileInformationByHandle(handle, &final_information)) {
        const auto inspect_error = GetLastError();
        CloseHandle(handle);
        error =
          "cannot re-inspect the pinned contract (Windows error " +
          std::to_string(inspect_error) + ")";
        return std::nullopt;
      }
      const auto final_bytes =
        (static_cast<std::uint64_t>(final_information.nFileSizeHigh) << 32u) |
        final_information.nFileSizeLow;
      if (final_bytes != bytes) {
        CloseHandle(handle);
        error = "contract size changed while it was being read";
        return std::nullopt;
      }
      CloseHandle(handle);
#else
      const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      if (fd < 0) {
        error =
          "cannot open contract without following links: " +
          std::string {std::strerror(errno)};
        return std::nullopt;
      }
      struct stat information {};
      if (
        ::fstat(fd, &information) != 0 ||
        !S_ISREG(information.st_mode)
      ) {
        const auto inspect_error = errno;
        ::close(fd);
        error =
          "contract is not a no-follow regular file: " +
          std::string {std::strerror(inspect_error)};
        return std::nullopt;
      }
      if (
        information.st_size <= 0 ||
        static_cast<std::uintmax_t>(information.st_size) > max_bytes
      ) {
        ::close(fd);
        error = "contract size is invalid";
        return std::nullopt;
      }
      serialized.resize(static_cast<std::size_t>(information.st_size));
      std::size_t offset = 0;
      while (offset < serialized.size()) {
        const auto read = ::read(
          fd,
          serialized.data() + offset,
          serialized.size() - offset
        );
        if (read <= 0) {
          const auto read_error = errno;
          ::close(fd);
          error =
            "cannot read the pinned contract: " +
            std::string {std::strerror(read_error)};
          return std::nullopt;
        }
        offset += static_cast<std::size_t>(read);
      }
      struct stat final_information {};
      if (::fstat(fd, &final_information) != 0) {
        const auto inspect_error = errno;
        ::close(fd);
        error =
          "cannot re-inspect the pinned contract: " +
          std::string {std::strerror(inspect_error)};
        return std::nullopt;
      }
      if (final_information.st_size != information.st_size) {
        ::close(fd);
        error = "contract size changed while it was being read";
        return std::nullopt;
      }
      ::close(fd);
#endif
      try {
        auto value = wire::parse_json_without_duplicate_keys(serialized);
        if (!value.is_object()) {
          error = "contract root is not an object";
          return std::nullopt;
        }
        if (content_sha256) {
          *content_sha256 = sha256_hex(serialized);
        }
        return value;
      } catch (const std::exception &exception) {
        error = std::string {"cannot parse contract: "} + exception.what();
        return std::nullopt;
      }
    }

    std::string quote_windows_process_argument(const std::string_view argument) {
      // The command is passed to CreateProcess, not cmd.exe. Implement the documented
      // CommandLineToArgvW-compatible backslash/quote convention and always quote.
      std::string result;
      result.push_back('"');
      std::size_t backslashes = 0;
      for (const char character : argument) {
        if (character == '\\') {
          ++backslashes;
          continue;
        }
        if (character == '"') {
          result.append(backslashes * 2 + 1, '\\');
          result.push_back('"');
          backslashes = 0;
          continue;
        }
        result.append(backslashes, '\\');
        backslashes = 0;
        result.push_back(character);
      }
      result.append(backslashes * 2, '\\');
      result.push_back('"');
      return result;
    }

    std::string build_worker_command(
      const fs::path &sunshine_executable,
      const fs::path &sunshine_config,
      const fs::path &worker_spec,
      const std::string_view worker_spec_sha256
    ) {
      return quote_windows_process_argument(path_to_utf8(sunshine_executable)) +
             " " +
             quote_windows_process_argument(path_to_utf8(sunshine_config)) +
             " --offline-sbs-worker " +
             quote_windows_process_argument(path_to_utf8(worker_spec)) +
             " " +
             quote_windows_process_argument(worker_spec_sha256);
    }

    FILE *open_binary_log(const fs::path &path) {
#ifdef _WIN32
      return _wfopen(path.c_str(), L"wb");
#else
      return std::fopen(path.c_str(), "wb");
#endif
    }

    /**
     * Retains the beginning of a worker diagnostic and appends a deterministic
     * explanation when the source exceeds its allowance. The marker's space is
     * reserved up front, so the sink can never write more than
     * max_native_worker_log_bytes even when one read returns a very large chunk.
     *
     * Exceeding the limit is fatal to the worker (the supervisor enforces that
     * policy). This is deliberately a bounded head capture rather than a rotating
     * file: startup diagnostics remain intact, and the child cannot hide a runaway
     * logger by continually replacing the evidence that caused termination.
     */
    class bounded_native_worker_log_writer_t {
    public:
      using sink_t = std::function<bool(std::string_view)>;

      explicit bounded_native_worker_log_writer_t(sink_t sink):
          sink_(std::move(sink)) {
      }

      void append(const std::string_view bytes) {
        if (bytes.empty() || write_failed_.load(std::memory_order_relaxed)) {
          return;
        }

        const auto available =
          max_native_worker_log_payload_bytes - payload_bytes_;
        const auto retained = std::min(available, bytes.size());
        if (
          retained != 0 &&
          !sink_(bytes.substr(0, retained))
        ) {
          write_failed_.store(true, std::memory_order_release);
          return;
        }
        payload_bytes_ += retained;

        if (
          retained != bytes.size() &&
          !limit_exceeded_.exchange(true, std::memory_order_acq_rel)
        ) {
          if (!sink_(native_worker_log_limit_marker)) {
            write_failed_.store(true, std::memory_order_release);
          }
        }
      }

      [[nodiscard]] bool limit_exceeded() const noexcept {
        return limit_exceeded_.load(std::memory_order_acquire);
      }

      [[nodiscard]] bool write_failed() const noexcept {
        return write_failed_.load(std::memory_order_acquire);
      }

    private:
      sink_t sink_;
      std::size_t payload_bytes_ = 0;
      std::atomic<bool> limit_exceeded_ {false};
      std::atomic<bool> write_failed_ {false};
    };

    /**
     * Pipe-backed capture for the native worker's merged stdout/stderr.
     *
     * The child receives the duplicated pipe writer as its STARTF_USESTDHANDLES
     * standard output/error. The elevated-tray token path uses plain STARTUPINFO;
     * no other handle is named by that launch contract. A manager-owned reader drains
     * it into the strictly bounded file above. The reader uses non-blocking availability
     * checks, so abort() always joins promptly even if a failed process-tree cleanup
     * leaves a writer handle alive; this keeps cancellation and Sunshine shutdown
     * deadlock-free.
     */
    class bounded_native_worker_log_pipe_t {
    public:
      explicit bounded_native_worker_log_pipe_t(FILE *output):
          output_(output, &std::fclose) {
      }

      ~bounded_native_worker_log_pipe_t() {
        abort();
      }

      bounded_native_worker_log_pipe_t(
        const bounded_native_worker_log_pipe_t &
      ) = delete;
      bounded_native_worker_log_pipe_t &operator=(
        const bounded_native_worker_log_pipe_t &
      ) = delete;

      bool initialize(std::string &error) {
#ifdef _WIN32
        HANDLE read_handle = INVALID_HANDLE_VALUE;
        HANDLE write_handle = INVALID_HANDLE_VALUE;
        if (
          !CreatePipe(
            &read_handle,
            &write_handle,
            nullptr,
            64u * 1024u
          )
        ) {
          error =
            "cannot create bounded native worker diagnostic pipe (Windows error " +
            std::to_string(GetLastError()) + ")";
          return false;
        }
        read_handle_ = read_handle;
        const auto descriptor = _open_osfhandle(
          reinterpret_cast<std::intptr_t>(write_handle),
          _O_BINARY | _O_WRONLY
        );
        if (descriptor == -1) {
          CloseHandle(write_handle);
          close_read_handle();
          error = "cannot bind bounded native worker diagnostic pipe";
          return false;
        }
        child_stream_ = _fdopen(descriptor, "wb");
        if (!child_stream_) {
          _close(descriptor);
          close_read_handle();
          error = "cannot open bounded native worker diagnostic stream";
          return false;
        }
#else
        int descriptors[2] {-1, -1};
        if (::pipe(descriptors) != 0) {
          error =
            "cannot create bounded native worker diagnostic pipe: " +
            std::generic_category().message(errno);
          return false;
        }
        read_descriptor_ = descriptors[0];
        const auto flags = ::fcntl(read_descriptor_, F_GETFL, 0);
        if (
          flags == -1 ||
          ::fcntl(read_descriptor_, F_SETFL, flags | O_NONBLOCK) == -1
        ) {
          ::close(descriptors[1]);
          close_read_handle();
          error =
            "cannot make native worker diagnostic pipe non-blocking: " +
            std::generic_category().message(errno);
          return false;
        }
        child_stream_ = ::fdopen(descriptors[1], "wb");
        if (!child_stream_) {
          ::close(descriptors[1]);
          close_read_handle();
          error =
            "cannot open bounded native worker diagnostic stream: " +
            std::generic_category().message(errno);
          return false;
        }
#endif

        writer_ = std::make_unique<bounded_native_worker_log_writer_t>(
          [this](const std::string_view bytes) {
            if (!output_) {
              return false;
            }
            const auto written =
              std::fwrite(bytes.data(), 1, bytes.size(), output_.get());
            return written == bytes.size() && std::fflush(output_.get()) == 0;
          }
        );
        try {
          reader_ = std::jthread([this](const std::stop_token stop) {
            drain(stop);
          });
        } catch (const std::exception &exception) {
          error =
            std::string {"cannot start native worker diagnostic reader: "} +
            exception.what();
          abort();
          return false;
        }
        return true;
      }

      [[nodiscard]] FILE *child_stream() const noexcept {
        return child_stream_;
      }

      void close_parent_writer() noexcept {
        if (child_stream_) {
          std::fclose(child_stream_);
          child_stream_ = nullptr;
        }
      }

      // Call only after the complete process group is known to be stopped. All pipe
      // writers are then closed and the reader drains the final buffered bytes to EOF.
      void finish() noexcept {
        close_parent_writer();
        if (reader_.joinable()) {
          reader_.join();
        }
        close_read_handle();
      }

      // Safe even when process-tree cleanup failed: the reader never blocks in a
      // consuming read, so its stop request is observed within one short poll.
      void abort() noexcept {
        close_parent_writer();
        if (reader_.joinable()) {
          reader_.request_stop();
          reader_.join();
        }
        close_read_handle();
      }

      [[nodiscard]] bool limit_exceeded() const noexcept {
        return writer_ && writer_->limit_exceeded();
      }

      [[nodiscard]] bool failed() const noexcept {
        return
          pipe_error_.load(std::memory_order_acquire) != 0 ||
          (writer_ && writer_->write_failed());
      }

      [[nodiscard]] std::string failure() const {
        if (writer_ && writer_->write_failed()) {
          return "cannot write bounded native worker diagnostic log";
        }
        const auto error = pipe_error_.load(std::memory_order_acquire);
        if (error == 0) {
          return {};
        }
#ifdef _WIN32
        return
          "native worker diagnostic pipe failed (Windows error " +
          std::to_string(error) + ")";
#else
        return
          "native worker diagnostic pipe failed: " +
          std::generic_category().message(error);
#endif
      }

    private:
      static constexpr auto reader_idle_poll = std::chrono::milliseconds(2);

      void drain(const std::stop_token stop) noexcept {
        std::array<char, 64u * 1024u> buffer {};
        while (!stop.stop_requested()) {
#ifdef _WIN32
          DWORD available = 0;
          if (
            !PeekNamedPipe(
              read_handle_,
              nullptr,
              0,
              nullptr,
              &available,
              nullptr
            )
          ) {
            const auto error = GetLastError();
            if (
              error != ERROR_BROKEN_PIPE &&
              error != ERROR_PIPE_NOT_CONNECTED &&
              error != ERROR_NO_DATA
            ) {
              pipe_error_.store(error, std::memory_order_release);
            }
            break;
          }
          if (available == 0) {
            std::this_thread::sleep_for(reader_idle_poll);
            continue;
          }
          DWORD bytes_read = 0;
          const auto requested = static_cast<DWORD>(
            std::min<std::size_t>(buffer.size(), available)
          );
          if (
            !ReadFile(
              read_handle_,
              buffer.data(),
              requested,
              &bytes_read,
              nullptr
            )
          ) {
            const auto error = GetLastError();
            if (
              error != ERROR_BROKEN_PIPE &&
              error != ERROR_PIPE_NOT_CONNECTED &&
              error != ERROR_NO_DATA
            ) {
              pipe_error_.store(error, std::memory_order_release);
            }
            break;
          }
          if (bytes_read == 0) {
            break;
          }
          writer_->append(
            std::string_view {buffer.data(), static_cast<std::size_t>(bytes_read)}
          );
#else
          const auto bytes_read =
            ::read(read_descriptor_, buffer.data(), buffer.size());
          if (bytes_read > 0) {
            writer_->append(
              std::string_view {
                buffer.data(),
                static_cast<std::size_t>(bytes_read),
              }
            );
            continue;
          }
          if (bytes_read == 0) {
            break;
          }
          if (errno == EINTR) {
            continue;
          }
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            std::this_thread::sleep_for(reader_idle_poll);
            continue;
          }
          pipe_error_.store(errno, std::memory_order_release);
          break;
#endif
        }
      }

      void close_read_handle() noexcept {
#ifdef _WIN32
        if (read_handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(read_handle_);
          read_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (read_descriptor_ != -1) {
          ::close(read_descriptor_);
          read_descriptor_ = -1;
        }
#endif
      }

      std::unique_ptr<FILE, decltype(&std::fclose)> output_ {
        nullptr,
        &std::fclose,
      };
      FILE *child_stream_ = nullptr;
      std::unique_ptr<bounded_native_worker_log_writer_t> writer_;
      std::jthread reader_;
      std::atomic<int> pipe_error_ {0};
#ifdef _WIN32
      HANDLE read_handle_ = INVALID_HANDLE_VALUE;
#else
      int read_descriptor_ = -1;
#endif
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

    std::optional<fs::path> discover_media_tool(
      const service_config_t &config,
      const std::optional<fs::path> &trusted_override,
      const std::string_view tool_name,
      std::string &error
    ) {
      std::vector<fs::path> candidates;
      if (trusted_override) {
        if (!trusted_override->is_absolute()) {
          error = "trusted media-tool override must be an absolute path";
          return std::nullopt;
        }
        candidates.push_back(*trusted_override);
      } else {
        const auto installation = config.sunshine_executable.parent_path();
#ifdef _WIN32
        candidates.push_back(
          installation / path_from_utf8(std::string {tool_name} + ".exe")
        );
        candidates.push_back(
          installation / "tools" /
          path_from_utf8(std::string {tool_name} + ".exe")
        );
#else
        candidates.push_back(installation / path_from_utf8(std::string {tool_name}));
        candidates.push_back(
          installation / "tools" / path_from_utf8(std::string {tool_name})
        );
#endif
      }

      for (const auto &candidate : candidates) {
        std::string ignored;
        auto resolved = canonical_regular_file(candidate, ignored);
        if (!resolved) {
          continue;
        }
        auto name = resolved->filename().string();
        std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
#ifdef _WIN32
        if (name != std::string {tool_name} + ".exe") {
#else
        if (name != tool_name) {
#endif
          error =
            "trusted media-tool executable must be named " +
            std::string {tool_name};
          return std::nullopt;
        }
        return resolved;
      }
      error =
        std::string {tool_name} +
        " was not found beside Sunshine (or in its tools directory); "
        "offline video jobs are unavailable until the trusted media tools are installed";
      return std::nullopt;
    }

    bool terminate_and_reap_process_tree(
      bp::child &child,
      bp::group &group,
      const std::chrono::milliseconds timeout,
      std::string &error
    ) {
      std::error_code terminate_error;
      const bool group_was_running =
        group.valid() &&
        platf::process_group_running(
          reinterpret_cast<std::uintptr_t>(group.native_handle())
        );
      const bool child_was_running = child.valid() && child.running();
      if (group_was_running) {
        group.terminate(terminate_error);
      } else if (child_was_running) {
        child.terminate(terminate_error);
      }

      const auto deadline = std::chrono::steady_clock::now() + timeout;
      while (std::chrono::steady_clock::now() < deadline) {
        const bool child_running = child.valid() && child.running();
        const bool group_running =
          group.valid() &&
          platf::process_group_running(
            reinterpret_cast<std::uintptr_t>(group.native_handle())
          );
        if (!child_running && !group_running) {
          break;
        }
        std::this_thread::sleep_for(20ms);
      }

      const bool child_running = child.valid() && child.running();
      const bool group_running =
        group.valid() &&
        platf::process_group_running(
          reinterpret_cast<std::uintptr_t>(group.native_handle())
        );
      if (child_running || group_running) {
        if (child_running) {
          std::error_code ignored;
          child.terminate(ignored);
          child.detach();
        }
        // Do not detach a still-running group. Its destructor closes the native job handle,
        // providing the final kill-on-close fallback without blocking Sunshine shutdown.
        error =
          "process tree did not terminate within " +
          std::to_string(timeout.count()) + " ms";
        if (terminate_error) {
          error += ": " + terminate_error.message();
        }
        return false;
      }

      if (child.valid()) {
        std::error_code wait_error;
        child.wait(wait_error);
        if (wait_error) {
          error = "cannot reap child process: " + wait_error.message();
          if (group.valid()) {
            group.detach();
          }
          return false;
        }
      }
      if (group.valid()) {
        group.detach();
      }
      if (terminate_error) {
        error = "process-tree termination reported: " + terminate_error.message();
        return false;
      }
      return true;
    }

    bool harden_worker_process_group(bp::group &group, std::string &error) {
#ifdef _WIN32
      if (!group.valid()) {
        error = "native worker process group is invalid";
        return false;
      }
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      const auto handle = reinterpret_cast<HANDLE>(group.native_handle());
      if (!QueryInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            &limits,
            sizeof(limits),
            nullptr
          )) {
        error =
          "cannot query native worker job object (Windows error " +
          std::to_string(GetLastError()) + ")";
        return false;
      }
      limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      limits.BasicLimitInformation.LimitFlags &=
        ~(JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK);
      if (!SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            &limits,
            sizeof(limits)
          )) {
        error =
          "cannot harden native worker job object (Windows error " +
          std::to_string(GetLastError()) + ")";
        return false;
      }
#else
      (void) group;
#endif
      return true;
    }

    std::optional<std::string> run_media_tool_probe(
      const service_config_t &config,
      const fs::path &tool,
      const std::string_view tool_name,
      const std::string_view probe_name,
      const std::string_view arguments,
      const bool allow_empty_output,
      std::string &error
    ) {
      const auto log_path =
        config.state_root /
        path_from_utf8(
          std::string {tool_name} + "-" + std::string {probe_name} + "-probe.log"
        );
      std::unique_ptr<FILE, decltype(&std::fclose)> log {
        open_binary_log(log_path),
        &std::fclose,
      };
      if (!log) {
        error = "cannot create FFmpeg probe log";
        return std::nullopt;
      }
      auto environment = boost::this_process::environment();
      boost::filesystem::path working_directory {
        config.sunshine_executable.parent_path().string()
      };
      bp::group group;
      std::string group_error;
      if (!harden_worker_process_group(group, group_error)) {
        error =
          std::string {tool_name} + " probe isolation failed before launch: " +
          group_error;
        return std::nullopt;
      }
      std::error_code launch_error;
      const auto command =
        quote_windows_process_argument(path_to_utf8(tool)) +
        " " + std::string {arguments};
      auto child = platf::run_command_unelevated(
        false,
        command,
        working_directory,
        environment,
        log.get(),
        launch_error,
        &group,
        config.expected_user_id
      );
      if (launch_error || !child.valid()) {
        error =
          "cannot launch " + std::string {tool_name} +
          " probe: " + launch_error.message();
        return std::nullopt;
      }
      const auto deadline = std::chrono::steady_clock::now() + media_probe_timeout;
      while (child.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(20ms);
      }
      if (child.running()) {
        std::string cleanup_error;
        terminate_and_reap_process_tree(
          child,
          group,
          worker_termination_timeout,
          cleanup_error
        );
        error =
          std::string {tool_name} + " probe timed out" +
          (cleanup_error.empty() ? "" : ": " + cleanup_error);
        return std::nullopt;
      }
      const auto exit_code = child.exit_code();
      std::string cleanup_error;
      if (!terminate_and_reap_process_tree(
            child,
            group,
            worker_termination_timeout,
            cleanup_error
          )) {
        error = std::string {tool_name} + " probe cleanup failed: " + cleanup_error;
        return std::nullopt;
      }
      if (exit_code != 0) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe failed";
        return std::nullopt;
      }
      log.reset();
      std::error_code size_error;
      const auto output_bytes = fs::file_size(log_path, size_error);
      if (size_error || output_bytes > 8ull * 1024ull * 1024ull) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe output is invalid";
        return std::nullopt;
      }
      std::ifstream input(log_path, std::ios::binary);
      std::string output {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
      };
      if (output.empty() && !allow_empty_output) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe produced no output";
        return std::nullopt;
      }
      return output;
    }

    std::optional<std::string> probe_media_tool_version(
      const service_config_t &config,
      const fs::path &tool,
      const std::string_view tool_name,
      std::string &error
    ) {
      if (!config.probe_media_tools) {
        return "not-probed";
      }
      auto output = run_media_tool_probe(
        config,
        tool,
        tool_name,
        "version",
        "-hide_banner -version",
        false,
        error
      );
      if (!output) {
        return std::nullopt;
      }
      std::istringstream lines {*output};
      std::string line;
      while (std::getline(lines, line)) {
        if (!line.empty()) {
          if (line.size() > 512) {
            line.resize(512);
          }
          return line;
        }
      }
      error = std::string {tool_name} + " probe produced no version line";
      return std::nullopt;
    }

    std::optional<std::string> media_tool_version_token(
      const std::string_view line,
      const std::string_view tool_name
    ) {
      const std::string prefix = std::string {tool_name} + " version ";
      if (!line.starts_with(prefix)) {
        return std::nullopt;
      }
      const auto token_start = prefix.size();
      const auto token_end = line.find_first_of(" \t\r\n", token_start);
      if (token_end == token_start) {
        return std::nullopt;
      }
      return std::string {
        line.substr(
          token_start,
          token_end == std::string_view::npos ?
            line.size() - token_start :
            token_end - token_start
        )
      };
    }

    std::optional<std::vector<std::string>> probe_ffmpeg_codecs(
      const service_config_t &config,
      const fs::path &ffmpeg,
      std::string &error
    ) {
#ifdef SUNSHINE_TESTS
      if (config.test_probed_codecs) {
        return *config.test_probed_codecs;
      }
#endif
      if (!config.probe_media_tools) {
        return std::vector<std::string> {"hevc_nvenc", "av1_nvenc"};
      }
      auto encoders = run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        "encoders",
        "-hide_banner -encoders",
        false,
        error
      );
      if (!encoders) {
        return std::nullopt;
      }
      auto filters = run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        "filters",
        "-hide_banner -filters",
        false,
        error
      );
      if (!filters) {
        return std::nullopt;
      }
      if (filters->find("zscale") == std::string::npos) {
        error =
          "approved FFmpeg lacks the zscale filter required for static HDR conversion";
        return std::nullopt;
      }

      std::vector<std::string> codecs;
      for (const auto codec : {"hevc_nvenc"sv, "av1_nvenc"sv}) {
        if (encoders->find(codec) == std::string::npos) {
          continue;
        }
        codecs.emplace_back(codec);
      }
      return codecs;
    }

    bool probe_ffmpeg_codec_runtime(
      const service_config_t &config,
      const fs::path &ffmpeg,
      const std::string_view codec,
      std::string &error
    ) {
      const auto arguments = codec == "hevc_nvenc" ?
        "-hide_banner -loglevel error -nostdin -f lavfi "
        "-i color=c=black:s=256x256:d=0.04 -frames:v 1 -an "
        "-vf format=p010le -c:v hevc_nvenc -profile:v main10 -f null -" :
        "-hide_banner -loglevel error -nostdin -f lavfi "
        "-i color=c=black:s=256x256:d=0.04 -frames:v 1 -an "
        "-vf format=p010le -c:v av1_nvenc -f null -";
      return run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        std::string {codec} + "-10bit",
        arguments,
        true,
        error
      ).has_value();
    }

    progress_t parse_worker_progress(
      const nlohmann::json &value,
      const std::string &job_id
    ) {
      if (
        value.value("schema", 0) != 1 ||
        value.value("job_id", "") != job_id
      ) {
        throw std::runtime_error("worker progress identity mismatch");
      }
      progress_t progress;
      progress.phase = value.value("phase", "");
      if (progress.phase.empty() || progress.phase.size() > 64) {
        throw std::runtime_error("worker progress phase is invalid");
      }
      progress.processed_frames = value.value("processed_frames", 0ull);
      if (value.contains("total_frames") && !value["total_frames"].is_null()) {
        progress.total_frames = value["total_frames"].get<std::uint64_t>();
        if (
          *progress.total_frames == 0 ||
          progress.processed_frames > *progress.total_frames
        ) {
          throw std::runtime_error("worker progress frame counts are invalid");
        }
      }
      if (
        value.contains("source_time_seconds") &&
        !value["source_time_seconds"].is_null()
      ) {
        const auto number = value["source_time_seconds"].get<double>();
        if (!std::isfinite(number) || number < 0) {
          throw std::runtime_error("worker source time is invalid");
        }
        progress.source_time_seconds = number;
      }
      if (
        value.contains("source_duration_seconds") &&
        !value["source_duration_seconds"].is_null()
      ) {
        const auto number = value["source_duration_seconds"].get<double>();
        if (!std::isfinite(number) || number <= 0) {
          throw std::runtime_error("worker source duration is invalid");
        }
        progress.source_duration_seconds = number;
      }
      if (value.contains("scene_count") && !value["scene_count"].is_null()) {
        progress.scene_count = value["scene_count"].get<std::uint64_t>();
        if (*progress.scene_count > max_scene_count) {
          throw std::runtime_error("worker progress scene count exceeds the serialized contract");
        }
      }
      if (value.contains("current_scene") && !value["current_scene"].is_null()) {
        if (
          !value["current_scene"].is_object() ||
          value["current_scene"].dump().size() > max_current_scene_bytes
        ) {
          throw std::runtime_error("worker current scene is invalid");
        }
        progress.current_scene = value["current_scene"];
      }
      if (value.contains("scene_decisions")) {
        const auto &decisions = value["scene_decisions"];
        if (
          !decisions.is_array() ||
          decisions.size() > max_progress_scene_decisions ||
          decisions.dump().size() > max_progress_contract_bytes ||
          !std::ranges::all_of(decisions, [](const auto &decision) {
            return decision.is_object();
          })
        ) {
          throw std::runtime_error("worker scene decisions are invalid");
        }
        progress.scene_decisions = decisions;
      }
      return progress;
    }

    nlohmann::json worker_spec_json(const worker_context_t &context) {
      return wire::to_json(wire::worker_spec_contract_t {
        .job_id = context.job_id,
        .operation = to_string(context.operation),
        .input_path = path_to_utf8(context.input_path),
        .job_directory = path_to_utf8(context.job_directory),
        .result_directory = path_to_utf8(context.result_directory),
        .progress_path = path_to_utf8(context.worker_progress),
        .result_path = path_to_utf8(context.worker_result),
        .staging_output = context.staging_output ?
                            std::optional<std::string> {
                              path_to_utf8(*context.staging_output)
                            } :
                            std::nullopt,
        .sunshine_executable = path_to_utf8(context.sunshine_executable),
        .sunshine_config = path_to_utf8(context.sunshine_config),
        .ffmpeg_path = path_to_utf8(context.ffmpeg_executable),
        .ffmpeg_version = context.ffmpeg_version,
        .ffprobe_path = path_to_utf8(context.ffprobe_executable),
        .ffprobe_version = context.ffprobe_version,
        .codec = context.codec,
        .transient_raster_hard_cap_bytes = context.scene_cache_max_bytes,
      });
    }

    bool validate_worker_result_contract(
      const nlohmann::json &result,
      const worker_context_t &context,
      std::string &error
    ) {
      try {
        const auto typed_result = wire::parse_worker_result_contract(result);
        if (
          typed_result.job_id != context.job_id ||
          typed_result.operation != to_string(context.operation) ||
          typed_result.codec != context.codec ||
          context.worker_spec_sha256.empty() ||
          typed_result.worker_spec_sha256 != context.worker_spec_sha256
        ) {
          error = "worker result identity/configuration attestation mismatch";
          return false;
        }
        const auto scene_count = typed_result.scenes.size();
        if (
          scene_count == 0 ||
          scene_count > max_scene_count
        ) {
          error = "worker result scene count is invalid";
          return false;
        }
        const auto &source = typed_result.source;
        if (
          source.width == 0 || source.height == 0 || source.frame_count == 0 ||
          !std::isfinite(source.duration_seconds) ||
          source.duration_seconds <= 0
        ) {
          error = "worker source summary is invalid";
          return false;
        }
        const auto expected_source_contract =
          (context.result_directory / "source-contract.json").lexically_normal();
        const auto expected_scene_audit =
          (context.result_directory / "scene-audit.json").lexically_normal();
        if (
          path_from_utf8(source.contract_path)
              .lexically_normal() != expected_source_contract ||
          path_from_utf8(typed_result.scene_audit_path)
              .lexically_normal() != expected_scene_audit ||
          !regular_nonempty_file(expected_source_contract) ||
          !regular_nonempty_file(expected_scene_audit)
        ) {
          error = "worker result artifact paths are invalid";
          return false;
        }
        const auto &cache = typed_result.cache;
        if (
          cache.hard_cap_bytes != context.scene_cache_max_bytes ||
          cache.remaining_bytes != 0 ||
          cache.peak_bytes > context.scene_cache_max_bytes
        ) {
          error = "worker transient-raster attestation is invalid";
          return false;
        }

        if (context.operation == operation_e::convert) {
          if (
            !context.staging_output ||
            !context.final_output ||
            !typed_result.output_path ||
            !typed_result.staging_identity ||
            path_from_utf8(*typed_result.output_path)
                .lexically_normal() != context.staging_output->lexically_normal() ||
            !file_matches_publish_identity(
              *context.staging_output,
              *typed_result.staging_identity
            ) ||
            !regular_nonempty_file(*context.staging_output) ||
            !regular_nonempty_file(
              context.result_directory / "output-contract.json"
            )
          ) {
            error = "worker direct conversion output attestation is invalid";
            return false;
          }
        } else if (typed_result.output_path || typed_result.staging_identity ||
                   !typed_result.replay_contracts.empty()) {
          error = "evaluation worker unexpectedly reported rendered output";
          return false;
        }
      } catch (const std::exception &exception) {
        error = std::string {"invalid worker result: "} + exception.what();
        return false;
      }
      return true;
    }

    nlohmann::json compact_worker_result(
      const nlohmann::json &result,
      const worker_context_t &context
    ) {
      // The child-owned worker-result.json and scene-audit.json retain the complete
      // per-scene causal evidence. Durable service state stays deliberately small so
      // listing jobs cannot deep-copy a multi-hour clip's full report under the service
      // mutex and job.json remains bounded and restart-safe.
      nlohmann::json compact = nlohmann::json::object();
      constexpr std::array scalar_fields {
        "schema"sv,
        "job_id"sv,
        "status"sv,
        "operation"sv,
        "codec"sv,
        "worker_spec_sha256"sv,
        "python_dependency"sv,
        "scene_count"sv,
        "scene_audit"sv,
        "output"sv,
      };
      for (const auto field : scalar_fields) {
        const auto iterator = result.find(field);
        if (
          iterator != result.end() &&
          (
            iterator->is_null() ||
            iterator->is_boolean() ||
            iterator->is_number() ||
            iterator->is_string()
          )
        ) {
          compact[std::string {field}] = *iterator;
        }
      }
      const auto copy_summary_object = [&](const std::string_view name,
                                           const auto &allowed_fields) {
        const auto source = result.find(name);
        if (source == result.end() || !source->is_object()) {
          return;
        }
        nlohmann::json summary = nlohmann::json::object();
        for (const auto field : allowed_fields) {
          const auto value = source->find(field);
          if (
            value != source->end() &&
            (
              value->is_null() ||
              value->is_boolean() ||
              value->is_number() ||
              value->is_string()
            )
          ) {
            summary[std::string {field}] = *value;
          }
        }
        compact[std::string {name}] = std::move(summary);
      };
      copy_summary_object(
        "source",
        std::array {
          "width"sv,
          "height"sv,
          "frame_count"sv,
          "duration_seconds"sv,
          "variable_frame_rate"sv,
          "color_mode"sv,
          "contract"sv,
        }
      );
      copy_summary_object(
        "cache",
        std::array {
          "hard_cap_bytes"sv,
          "peak_bytes"sv,
          "remaining_bytes"sv,
        }
      );
      compact["full_result_contract"] = path_to_utf8(context.worker_result);
      compact["scene_decisions_contract"] =
        path_to_utf8(context.result_directory / "scene-audit.json");
      compact["details_storage"] = "worker-artifacts";
      return compact;
    }
  }  // namespace

  namespace {
    wire::job_snapshot_contract_t job_snapshot_wire_contract(
      const job_snapshot_t &snapshot
    ) {
      std::vector<nlohmann::json> decisions;
      if (snapshot.progress.scene_decisions.is_array()) {
        decisions.assign(
          snapshot.progress.scene_decisions.begin(),
          snapshot.progress.scene_decisions.end()
        );
      }
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
      .scene_cache_max_bytes = snapshot.scene_cache_max_bytes,
      .cache_budget_policy = to_string(snapshot.cache_budget_policy),
      .progress = {
        .phase = snapshot.progress.phase,
        .processed_frames = snapshot.progress.processed_frames,
        .total_frames = snapshot.progress.total_frames,
        .source_time_seconds = snapshot.progress.source_time_seconds,
        .source_duration_seconds = snapshot.progress.source_duration_seconds,
        .scene_count = snapshot.progress.scene_count,
        .current_scene = snapshot.progress.current_scene.is_object() ?
                           std::optional<nlohmann::json> {
                             snapshot.progress.current_scene
                           } :
                           std::nullopt,
        .scene_decisions = std::move(decisions),
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

  nlohmann::json scene_audit_reply_t::json() const {
    return {
      {"status", ok},
      {"error_code", to_string(code)},
      {"error", error.empty() ? nlohmann::json(nullptr) : nlohmann::json(error)},
      {"audit", ok ? audit : nlohmann::json(nullptr)},
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
#ifdef _WIN32
      bool worker_result_pin_released_for_prune = false;
#endif
      mutable std::mutex artifact_mutex;
#ifdef _WIN32
      std::vector<pinned_handle_t> input_pins;
      // The job root is owned by worker_tree. Retain an additional result-directory
      // identity pin until pruning/shutdown so audit reads cannot be redirected.
      std::vector<pinned_handle_t> worker_path_pins;
#endif
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
#ifdef _WIN32
    // Every component is held without FILE_SHARE_DELETE for the whole manager
    // lifetime. Revalidation compares the current pathname to the pinned file IDs.
    std::vector<pinned_handle_t> managed_root_pins;
#endif

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
#ifdef _WIN32
      record.input_pins.clear();
#endif
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
        const auto policy = parse_budget_policy(
          contract.cache_budget_policy
        );
        if (!valid_job_id(snapshot.id) || !state || !operation || !policy) {
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
          auto output_location = output_location_e::legacy_managed_exports;
          if (contract.output_location) {
            const auto parsed_location = parse_output_location(*contract.output_location);
            if (!parsed_location) {
              error = "persisted output location is unsupported";
              return std::nullopt;
            }
            output_location = *parsed_location;
          }
          const auto expected_output =
            (
              output_location == output_location_e::input_directory ?
                snapshot.input_path.parent_path() / path_from_utf8(output_name) :
                config.exports_root / path_from_utf8(output_name)
            ).lexically_normal();
          if (
            !valid_output_name(output_name) ||
            persisted_output.lexically_normal() != expected_output
          ) {
            error =
              output_location == output_location_e::input_directory ?
                "persisted output escaped the input directory" :
                "persisted output escaped the legacy managed exports root";
            return std::nullopt;
          }
          snapshot.output_path = expected_output;
          snapshot.output_location = output_location;
        } else if (contract.output_location) {
          error = "persisted evaluation job unexpectedly names an output location";
          return std::nullopt;
        }
        snapshot.codec = contract.codec;
        snapshot.scene_cache_max_bytes = contract.scene_cache_max_bytes;
        snapshot.cache_budget_policy = *policy;
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
        snapshot.progress.scene_count = contract.progress.scene_count;
        if (contract.progress.current_scene) {
          if (contract.progress.current_scene->dump().size() > max_current_scene_bytes) {
            throw std::runtime_error("persisted current scene is too large");
          }
          snapshot.progress.current_scene = *contract.progress.current_scene;
        }
        snapshot.progress.scene_decisions = contract.progress.scene_decisions;
        if (snapshot.progress.scene_decisions.dump().size() >
            max_progress_contract_bytes) {
          throw std::runtime_error("persisted scene decisions are too large");
        }
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
        .scene_cache_max_bytes = snapshot.scene_cache_max_bytes,
        .cache_budget_policy = snapshot.cache_budget_policy,
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

            std::error_code ec;
            for (const auto &entry :
                 fs::directory_iterator(config.exports_root, ec)) {
              if (ec) {
                throw std::runtime_error(ec.message());
              }
              if (!is_managed_staging_filename(entry.path().filename())) {
                continue;
              }
              std::string staging_error;
              auto measurement =
                safe_filesystem::measure_tree_no_follow(
                  entry.path(),
                  staging_error
                );
              if (!measurement) {
                throw std::runtime_error(
                  "cannot safely measure retained staging artifact [" +
                  path_to_utf8(entry.path()) + "]: " + staging_error
                );
              }
              // A directory entry can disappear after enumeration. That is a
              // harmless concurrent cleanup; every other shape fails closed so a
              // junction or directory cannot evade or redirect quota accounting.
              if (!measurement->root_exists) {
                continue;
              }
              if (
                measurement->regular_file_count != 1 ||
                measurement->directory_count != 0 ||
                measurement->reparse_point_count != 0
              ) {
                throw std::runtime_error(
                  "retained staging artifact is not one regular file [" +
                  path_to_utf8(entry.path()) + "]"
                );
              }
              const auto staging_charge = std::max(
                measurement->regular_file_bytes,
                retained_staging_minimum_charge_bytes
              );
              if (
                total >
                std::numeric_limits<std::uint64_t>::max() - staging_charge
              ) {
                throw std::runtime_error(
                  "retained staging artifact quota charge overflowed"
                );
              }
              total += staging_charge;
            }
            if (ec) {
              throw std::runtime_error(ec.message());
            }

            for (const auto &[id, record] : jobs) {
              if (
                record->snapshot.output_location !=
                  output_location_e::input_directory ||
                !record->worker.staging_output
              ) {
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
        if (
          record->snapshot.output_location ==
            output_location_e::input_directory &&
          record->worker.staging_output
        ) {
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
            BOOST_LOG(warning)
              << "Retaining offline SBS record because its input-directory "
                 "staging path could not be inspected ["sv
              << record->snapshot.id << "]: "sv << staging_error;
            continue;
          }
          if (*staging_charge != 0) {
            // The protected record is the only durable map to a staging name
            // outside the legacy managed exports root. Preserve it until the
            // exact path disappears; the manager never deletes an unattested
            // or possibly replaced user-writable leaf by pathname.
            continue;
          }
        }
#ifdef _WIN32
        if (
          record->snapshot.worker_result.is_object() &&
          record->snapshot.worker_result.value(
            "staging_cleanup_pending",
            false
          )
        ) {
          // The protected record is the only durable authority that can prove
          // which user-writable staging identity is safe to retire. Keep it
          // until handle-based recovery succeeds; pruning it would turn a
          // transient disposition failure into a permanent quota leak.
          continue;
        }
#endif
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

        std::lock_guard artifact_lock {record->artifact_mutex};
        std::string worker_error;
#ifdef _WIN32
        if (!revalidate_pins(
              managed_root_pins,
              "offline managed roots",
              worker_error
            )) {
          BOOST_LOG(warning)
            << "Refusing to prune through changed offline managed roots ["sv
            << record->snapshot.id << "]: "sv << worker_error;
          continue;
        }
#endif
        if (!record->worker_artifacts_pruned) {
#ifdef _WIN32
          if (!record->worker_result_pin_released_for_prune) {
            if (record->worker_path_pins.empty()) {
              worker_error = "worker result path is not identity-pinned";
            }
            if (
              record->worker_path_pins.empty() ||
              !revalidate_pins(
                record->worker_path_pins,
                "offline worker job paths",
                worker_error
              )
            ) {
              BOOST_LOG(warning)
                << "Refusing to prune changed offline SBS worker paths ["sv
                << record->snapshot.id << "]: "sv << worker_error;
              continue;
            }
          }
#endif
          if (!record->worker_tree) {
            BOOST_LOG(warning)
              << "Refusing to prune unpinned offline SBS worker tree ["sv
              << record->snapshot.id << "]"sv;
            continue;
          }
#ifdef _WIN32
          // Release the child result pin before the retained tree removes that
          // exact child by handle. The job-root removal pin remains continuously
          // held, and a failed pass records the release so it can be retried.
          record->worker_path_pins.clear();
          record->worker_result_pin_released_for_prune = true;
#endif
          if (!run_user_filesystem_action([&]() {
                if (!record->worker_tree->remove(worker_error)) {
                  throw std::runtime_error(worker_error);
                }
              }, worker_error)) {
            BOOST_LOG(warning)
              << "Could not prune offline SBS worker artifacts ["sv
              << record->snapshot.id << "]: "sv << worker_error;
            // Keep durable state, the exact root pin, and the in-memory record so
            // a later pass can retry without resolving a user-writable pathname.
            continue;
          }
          record->worker_tree.reset();
          record->worker_artifacts_pruned = true;
          quota_error.clear();
          retained_bytes = retained_artifact_bytes_locked(quota_error);
          if (!retained_bytes) {
            BOOST_LOG(warning)
              << "Could not remeasure retained offline artifacts after worker "
                 "pruning: "sv
              << quota_error;
          }
        }

        std::string state_error;
        if (
          !safe_filesystem::remove_tree_no_follow(
            record->state_path.parent_path(),
            state_error
          )
        ) {
          BOOST_LOG(warning)
            << "Could not prune protected offline SBS state ["sv
            << record->snapshot.id << "]: "sv << state_error;
          // The durable record remains recoverable. A restart recognizes a
          // missing worker tree as an already completed first prune phase.
          continue;
        }
        jobs.erase(record->snapshot.id);
        quota_error.clear();
        retained_bytes = retained_artifact_bytes_locked(quota_error);
        if (!retained_bytes) {
          BOOST_LOG(warning)
            << "Could not remeasure retained offline artifacts after pruning: "sv
            << quota_error;
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
        const bool persisted_terminal_state =
          is_terminal(record->snapshot.state);
        record->retention_sequence = retention_sequence;
        record->worker = worker_context_for(
          record->snapshot,
          config.worker_root / "jobs" / id,
          staging
        );
        record->state_path = entry.path() / "job.json";
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
#ifdef _WIN32
              record->worker_result_pin_released_for_prune = true;
#endif
              worker_paths_trusted = true;
              return;
            }
#ifdef _WIN32
            auto result_pin = pin_managed_directory(
              record->worker.result_directory,
              "retained offline worker result directory",
              worker_pin_error
            );
            if (!result_pin) {
              return;
            }
            record->worker_path_pins.emplace_back(std::move(*result_pin));
#endif
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
          bool reconciled = false;
          bool staging_cleanup_pending = false;
          std::string publish_error;
          if (!run_user_filesystem_action([&]() {
#ifdef _WIN32
                std::optional<std::vector<pinned_handle_t>> output_directory_pins;
                if (
                  record->snapshot.output_location ==
                    output_location_e::input_directory
                ) {
                  if (!record->worker.final_output) {
                    publish_error =
                      "persisted publishing contract has no final output";
                    return;
                  }
                  output_directory_pins = pin_publication_directory(
                    record->worker.final_output->parent_path(),
                    publish_error
                  );
                  if (!output_directory_pins) {
                    return;
                  }
                }
#endif
                try {
                  const auto &identity =
                    record->snapshot.worker_result.at("publish_identity");
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
                      reconciled = true;
                      if (staged_file) {
                        std::string retirement_error;
#ifdef _WIN32
                        if (!staged_file->retire_staging_link(
                              retirement_error
                            )) {
                          staging_cleanup_pending = true;
                          BOOST_LOG(warning)
                            << "Recovered offline SBS publication but could "
                               "not retire its exact staging link ["sv
                            << id << "]: "sv << retirement_error;
                        }
#else
                        staging_cleanup_pending = true;
#endif
                      } else if (!staging_missing) {
                        // The final identity proves publication succeeded, but
                        // the staging pathname now names something else or
                        // cannot be pinned. Never delete it by name.
                        staging_cleanup_pending = true;
                        BOOST_LOG(warning)
                          << "Recovered offline SBS publication while "
                             "preserving an untrusted staging pathname ["sv
                          << id << "]: "sv << publish_error;
                      }
                    } else if (staged_file && !final_matches) {
                      bool staging_retired = false;
                      reconciled = staged_file->publish_no_replace(
                        *record->worker.final_output,
                        publish_error,
                        &staging_retired
                      );
                      staging_cleanup_pending =
                        reconciled && !staging_retired;
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
#ifdef _WIN32
                if (
                  output_directory_pins &&
                  !revalidate_pins(
                    *output_directory_pins,
                    "offline output directory during recovery",
                    publish_error
                  )
                ) {
                  reconciled = false;
                }
#endif
              }, publish_error)) {
            reconciled = false;
          }
          if (reconciled) {
            record->snapshot.state = job_state_e::complete;
            record->snapshot.progress.phase = "complete";
            record->snapshot.ended_at_unix_ms =
              record->snapshot.ended_at_unix_ms.value_or(unix_time_ms());
            record->snapshot.error.clear();
            record->snapshot.worker_result["output"] =
              path_to_utf8(*record->worker.final_output);
            record->snapshot.worker_result["published"] = true;
            record->snapshot.worker_result["staging_cleanup_pending"] =
              staging_cleanup_pending;
            std::string result_error;
            if (!run_user_filesystem_action([&]() {
                  if (!write_json_atomically(
                        record->worker.result_directory /
                          "publication-contract.json",
                        record->snapshot.worker_result,
                        result_error
                      )) {
                    throw std::runtime_error(result_error);
                  }
                }, result_error)) {
              BOOST_LOG(warning)
                << "Could not rewrite recovered offline SBS publication ["sv
                << id << "]: "sv << result_error;
            }
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
#ifdef _WIN32
        } else if (
          record->snapshot.state == job_state_e::complete &&
          record->snapshot.operation == operation_e::convert &&
          record->worker.staging_output &&
          record->worker.final_output &&
          record->snapshot.worker_result.is_object() &&
          record->snapshot.worker_result.value("published", false)
        ) {
          // A disposition failure after publication can leave a completed
          // record with two hard links. Retry cleanup on every restart using
          // the protected publish identity and a DELETE-capable handle to the
          // exact staging file; never unlink a re-resolved pathname.
          bool cleanup_pending = true;
          std::string cleanup_error;
          if (!run_user_filesystem_action([&]() {
                std::optional<std::vector<pinned_handle_t>> output_directory_pins;
                if (
                  record->snapshot.output_location ==
                    output_location_e::input_directory
                ) {
                  output_directory_pins = pin_publication_directory(
                    record->worker.final_output->parent_path(),
                    cleanup_error
                  );
                  if (!output_directory_pins) {
                    return;
                  }
                }
                try {
                  const auto &identity =
                    record->snapshot.worker_result.at("publish_identity");
                  if (
                    !identity.is_object() ||
                    !file_matches_publish_identity(
                      *record->worker.final_output,
                      identity
                    )
                  ) {
                    cleanup_error =
                      "completed publication identity no longer matches its "
                      "final output";
                    return;
                  }
                  bool staging_missing = false;
                  auto staged_file = publication_file_t::open(
                    *record->worker.staging_output,
                    identity,
                    cleanup_error,
                    &staging_missing
                  );
                  if (staging_missing) {
                    cleanup_pending = false;
                    return;
                  }
                  if (!staged_file) {
                    // A mismatched or inaccessible pathname is not ours to
                    // delete. Leave it quota-visible and retry on a later
                    // restart in case the failure was transient.
                    return;
                  }
                  cleanup_pending =
                    !staged_file->retire_staging_link(cleanup_error);
                } catch (const std::exception &exception) {
                  cleanup_error =
                    std::string {
                      "invalid completed publication contract: "
                    } + exception.what();
                }
                if (
                  output_directory_pins &&
                  !revalidate_pins(
                    *output_directory_pins,
                    "offline output directory during cleanup recovery",
                    cleanup_error
                  )
                ) {
                  cleanup_pending = true;
                }
              }, cleanup_error)) {
            cleanup_pending = true;
          }
          const bool marker_changed =
            record->snapshot.worker_result.value(
              "staging_cleanup_pending",
              false
            ) != cleanup_pending;
          record->snapshot.worker_result["staging_cleanup_pending"] =
            cleanup_pending;
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
#endif
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
          !persisted_terminal_state &&
          is_terminal(record->snapshot.state) &&
          worker_paths_trusted &&
          !record->worker_artifacts_pruned
        ) {
          std::string audit_hash;
          std::string audit_error;
          bool audit_attested = false;
          run_user_filesystem_action([&]() {
            auto audit = read_json_contract(
              record->worker.result_directory / "scene-audit.json",
              audit_error,
              max_scene_audit_bytes,
              &audit_hash
            );
            audit_attested = audit.has_value();
          }, audit_error);
          if (audit_attested) {
            if (!record->snapshot.worker_result.is_object()) {
              record->snapshot.worker_result = nlohmann::json::object();
            }
            record->snapshot.worker_result["scene_audit_sha256"] =
              std::move(audit_hash);
            if (!persist_locked(*record, read_error)) {
              error =
                "cannot persist recovered scene-audit identity: " +
                read_error;
              return false;
            }
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

    worker_outcome_t run_native_worker(
      const worker_context_t &context,
      const std::stop_token stop,
      const progress_callback_t &publish_progress
    ) {
      if (context.worker_spec_sha256.size() != 64) {
        return {.error = "native worker specification digest is missing"};
      }
      const auto command = build_worker_command(
        config.sunshine_executable,
        config.sunshine_config,
        context.worker_spec,
        context.worker_spec_sha256
      );
      FILE *raw_log = nullptr;
      std::string filesystem_error;
      if (!run_user_filesystem_action([&]() {
            raw_log = open_binary_log(context.worker_log);
          }, filesystem_error)) {
        if (raw_log) {
          std::fclose(raw_log);
        }
        return {.error = filesystem_error};
      }
      if (!raw_log) {
        return {.error = "cannot create native worker log"};
      }
      bounded_native_worker_log_pipe_t log {raw_log};
      std::string log_error;
      if (!log.initialize(log_error)) {
        return {.error = log_error};
      }
      auto environment = boost::this_process::environment();
      boost::filesystem::path working_directory {
        config.sunshine_executable.parent_path().string()
      };
      bp::group group;
      std::string group_error;
      if (!harden_worker_process_group(group, group_error)) {
        return {
          .error =
            "native worker isolation failed before launch: " + group_error,
        };
      }
      std::error_code process_error;
      auto child = platf::run_command_unelevated(
        false,
        command,
        working_directory,
        environment,
        log.child_stream(),
        process_error,
        &group,
        config.expected_user_id
      );
      // run_command_unelevated() duplicates this stream into inheritable standard-output/error
      // handles. Its elevated-tray token path supplies them through STARTF_USESTDHANDLES with
      // plain STARTUPINFO. Closing the manager's writer immediately guarantees EOF once the
      // isolated process group and its duplicated writer are gone.
      log.close_parent_writer();
      if (process_error || !child.valid()) {
        log.finish();
        return {
          .error = "cannot launch native offline worker: " +
                   process_error.message(),
        };
      }

      const auto terminate_worker = [&](std::string &cleanup_error) {
        const bool stopped = terminate_and_reap_process_tree(
          child,
          group,
          worker_termination_timeout,
          cleanup_error
        );
        if (stopped) {
          log.finish();
        } else {
          log.abort();
        }
        return stopped;
      };

      fs::file_time_type last_progress_write {};
      const auto consume_progress_update = [&](
        const bool force_read = false
      ) -> std::optional<std::string> {
        std::error_code time_error;
        fs::file_time_type write_time {};
        std::optional<nlohmann::json> value;
        std::string contract_error;
        filesystem_error.clear();
        const bool progress_read = run_user_filesystem_action([&]() {
          write_time = fs::last_write_time(
            context.worker_progress,
            time_error
          );
          if (
            !time_error &&
            (force_read || write_time != last_progress_write)
          ) {
            value = read_json_contract(
              context.worker_progress,
              contract_error,
              max_progress_contract_bytes
            );
          }
        }, filesystem_error);
        if (!progress_read) {
          return "cannot inspect native worker progress: " + filesystem_error;
        }
        if (
          time_error ||
          (!force_read && write_time == last_progress_write)
        ) {
          return std::nullopt;
        }
        if (!value) {
          if (contract_error == "contract does not exist") {
            return std::nullopt;
          }
          return "invalid native worker progress: " + contract_error;
        }
        try {
          publish_progress(parse_worker_progress(*value, context.job_id));
          last_progress_write = write_time;
        } catch (const std::exception &exception) {
          return std::string {"invalid native worker progress: "} +
                 exception.what();
        }
        return std::nullopt;
      };

      while (child.running()) {
        if (stop.stop_requested()) {
          std::string cleanup_error;
          if (!terminate_worker(cleanup_error)) {
            BOOST_LOG(error) << "Offline SBS worker cancellation cleanup failed: "sv
                             << cleanup_error;
          }
          return {.canceled = true};
        }
        if (log.limit_exceeded() || log.failed()) {
          std::string cleanup_error;
          terminate_worker(cleanup_error);
          return {
            .error =
              log.limit_exceeded() ?
                "native offline worker diagnostic output exceeded its " +
                  std::to_string(max_native_worker_log_bytes) +
                  "-byte retained limit" +
                  (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error) :
                log.failure() +
                  (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
          };
        }

        if (const auto progress_error = consume_progress_update()) {
          std::string cleanup_error;
          terminate_worker(cleanup_error);
          return {
            .error = *progress_error +
                     (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
          };
        }
        std::this_thread::sleep_for(config.process_poll_interval);
      }

      // The child can atomically publish its final progress and exit between two
      // polling ticks. Drain that last contract before terminalizing the job so
      // finalized scene decisions are not lost on fast success or failure exits.
      if (const auto progress_error = consume_progress_update(true)) {
        std::string cleanup_error;
        terminate_worker(cleanup_error);
        return {
          .error = *progress_error +
                   (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
        };
      }

      const auto exit_code = child.exit_code();
      std::string cleanup_error;
      if (!terminate_worker(cleanup_error)) {
        return {.error = "native worker cleanup failed: " + cleanup_error};
      }
      if (log.limit_exceeded()) {
        return {
          .error =
            "native offline worker diagnostic output exceeded its " +
            std::to_string(max_native_worker_log_bytes) +
            "-byte retained limit",
        };
      }
      if (log.failed()) {
        return {.error = log.failure()};
      }
      if (exit_code != 0) {
        return {
          .error = "native offline worker exited with code " +
                   std::to_string(exit_code),
        };
      }

      std::string contract_error;
      std::optional<nlohmann::json> result;
      bool contract_valid = false;
      if (!run_user_filesystem_action([&]() {
            result = read_json_contract(
              context.worker_result,
              contract_error
            );
            if (result) {
              contract_valid = validate_worker_result_contract(
                *result,
                context,
                contract_error
              );
            }
          }, filesystem_error)) {
        return {
          .error =
            "cannot inspect native worker result: " + filesystem_error,
        };
      }
      if (!result) {
        return {
          .error = "native worker did not publish a valid result: " +
                   contract_error,
        };
      }
      if (!contract_valid) {
        return {.error = "native worker result validation failed: " + contract_error};
      }
      return {
        .completed = true,
        .result = std::move(*result),
      };
    }

    void publish_progress(
      const std::string &id,
      progress_t progress
    ) {
      std::lock_guard lock {mutex};
      if (
        (
          progress.scene_count &&
          *progress.scene_count > max_scene_count
        ) ||
        !progress.scene_decisions.is_array() ||
        progress.scene_decisions.size() > max_progress_scene_decisions ||
        progress.scene_decisions.dump().size() >
          max_progress_contract_bytes ||
        (
          !progress.current_scene.is_null() &&
          (
            !progress.current_scene.is_object() ||
            progress.current_scene.dump().size() > max_current_scene_bytes
          )
        )
      ) {
        BOOST_LOG(error)
          << "Discarding invalid offline SBS progress decision window ["sv
          << id << ']';
        return;
      }
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
#ifdef _WIN32
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
#endif
          if (!worker_identity_valid) {
            outcome.error =
              identity_error.empty() ?
                "offline worker tree identity pin is missing before worker launch" :
                "offline managed path attestation failed before worker launch: " +
                  identity_error;
          } else
            outcome = runner ?
                        runner(record->worker, job_stop, callback) :
                        run_native_worker(record->worker, job_stop, callback);
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
              // across persistence and publication. The final rename is issued
              // against this handle, never by re-resolving the staging pathname.
              record->snapshot.state = job_state_e::publishing;
              record->snapshot.progress.phase = "publishing";
              record->snapshot.error.clear();
              record->snapshot.worker_result = compact_worker_result(
                std::move(outcome.result),
                record->worker
              );
              record->snapshot.worker_result["publish_identity"] =
                staged_output->identity();
              std::string publishing_error;
              if (!persist_locked(*record, publishing_error)) {
                record->snapshot.state = job_state_e::failed;
                record->snapshot.progress.phase = "failed";
                record->snapshot.error =
                  "cannot persist pre-publish state: " + publishing_error;
              } else {
                bool published = false;
                bool staging_retired = false;
                if (!run_user_filesystem_action([&]() {
#ifdef _WIN32
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
#endif
                      published = staged_output->publish_no_replace(
                        *record->worker.final_output,
                        completion_error,
                        &staging_retired
#ifdef SUNSHINE_TESTS
                        ,
                        config.test_force_staging_disposition_failure
#endif
                      );
                    }, completion_error)) {
                  published = false;
                }
                if (!published) {
                  record->snapshot.state = job_state_e::failed;
                  record->snapshot.progress.phase = "failed";
                  record->snapshot.error = completion_error;
                } else {
                  record->snapshot.worker_result["output"] =
                    path_to_utf8(*record->worker.final_output);
                  record->snapshot.worker_result["published"] = true;
                  record->snapshot.worker_result["staging_cleanup_pending"] =
                    !staging_retired;
                  record->snapshot.state = job_state_e::complete;
                  record->snapshot.progress.phase = "complete";
                  record->snapshot.error.clear();
                  std::string result_error;
                  if (!run_user_filesystem_action([&]() {
                        if (!write_json_atomically(
                              record->worker.result_directory /
                                "publication-contract.json",
                              record->snapshot.worker_result,
                              result_error
                            )) {
                          throw std::runtime_error(result_error);
                        }
                      }, result_error)) {
                    BOOST_LOG(warning)
                      << "Could not write published offline SBS contract ["sv
                      << record->snapshot.id << "]: "sv << result_error;
                  }
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
          // Bind any retained audit to the exact bounded bytes observed after
          // the worker process has exited. Later HTTP reads use a no-follow
          // handle and must reproduce this digest, so a same-user leaf swap or
          // in-place rewrite cannot masquerade as the worker's audit.
          std::string audit_hash;
          std::string audit_attestation_error;
          bool audit_attested = false;
          run_user_filesystem_action([&]() {
            auto audit = read_json_contract(
              record->worker.result_directory / "scene-audit.json",
              audit_attestation_error,
              max_scene_audit_bytes,
              &audit_hash
            );
            audit_attested = audit.has_value();
          }, audit_attestation_error);
          if (audit_attested) {
            if (!record->snapshot.worker_result.is_object()) {
              record->snapshot.worker_result = nlohmann::json::object();
            }
            record->snapshot.worker_result["scene_audit_sha256"] =
              std::move(audit_hash);
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
      impl_->config.exports_root.empty() ||
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
      ) ||
      !validate_local_path_syntax(
        impl_->config.exports_root,
        "offline SBS exports root",
        error
      )
    ) {
      return false;
    }
    const auto lexical_state_root = impl_->config.state_root.lexically_normal();
    const auto lexical_worker_root = impl_->config.worker_root.lexically_normal();
    const auto lexical_exports_root = impl_->config.exports_root.lexically_normal();
    if (
      managed_roots_overlap(lexical_state_root, lexical_worker_root) ||
      managed_roots_overlap(lexical_state_root, lexical_exports_root) ||
      managed_roots_overlap(lexical_worker_root, lexical_exports_root)
    ) {
      error =
        "offline SBS state, worker, and export roots must be disjoint";
      return false;
    }

    std::error_code ec;
#ifdef _WIN32
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
    std::optional<std::vector<pinned_handle_t>> exports_root_pins;
    std::optional<pinned_handle_t> exports_root_child_pin;
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
#else
    fs::create_directories(impl_->config.state_root / "jobs", ec);
    if (ec) {
      error = "cannot create offline SBS state root: " + ec.message();
      return false;
    }
#endif
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          std::error_code user_ec;
#ifdef _WIN32
          auto worker_creation_pins = create_user_directory(
            impl_->config.worker_root / "jobs",
            "offline SBS worker root"
          );
          auto exports_creation_pins = create_user_directory(
            impl_->config.exports_root,
            "offline SBS exports root"
          );
#else
          create_user_directory(
            impl_->config.worker_root / "jobs",
            "offline SBS worker root"
          );
          create_user_directory(
            impl_->config.exports_root,
            "offline SBS exports root"
          );
#endif
#ifdef _WIN32
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
          exports_root_pins = pin_managed_root_components(
            impl_->config.exports_root,
            "offline exports root",
            pin_error
          );
          if (!exports_root_pins) {
            throw std::runtime_error(pin_error);
          }
          exports_root_child_pin = pin_ephemeral_root_child(
            impl_->config.exports_root,
            "offline exports root",
            pin_error
          );
          if (!exports_root_child_pin) {
            throw std::runtime_error(pin_error);
          }
#endif
          impl_->config.worker_root =
            fs::canonical(impl_->config.worker_root, user_ec);
          if (user_ec) {
            throw std::runtime_error(
              "cannot canonicalize offline SBS worker root: " +
              user_ec.message()
            );
          }
          impl_->config.exports_root =
            fs::canonical(impl_->config.exports_root, user_ec);
          if (user_ec) {
            throw std::runtime_error(
              "cannot canonicalize offline SBS exports root: " +
              user_ec.message()
            );
          }
        }, error)) {
      return false;
    }
#ifdef _WIN32
    impl_->managed_root_pins.clear();
    impl_->managed_root_pins.reserve(
      state_root_pins->size() +
      state_jobs_pins->size() +
      worker_root_pins->size() +
      worker_jobs_pins->size() +
      exports_root_pins->size() +
      1
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
    std::ranges::move(
      *exports_root_pins,
      std::back_inserter(impl_->managed_root_pins)
    );
    impl_->managed_root_pins.emplace_back(
      std::move(*exports_root_child_pin)
    );
#endif
    impl_->config.state_root = fs::canonical(impl_->config.state_root, ec);
    if (ec) {
      error = "cannot canonicalize offline SBS state root: " + ec.message();
      return false;
    }
    if (
      managed_roots_overlap(
        impl_->config.state_root,
        impl_->config.worker_root
      ) ||
      managed_roots_overlap(
        impl_->config.state_root,
        impl_->config.exports_root
      ) ||
      managed_roots_overlap(
        impl_->config.worker_root,
        impl_->config.exports_root
      )
    ) {
      error =
        "offline SBS state, worker, and export roots must be disjoint";
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
#ifdef _WIN32
          entry.second->input_pins.clear();
          entry.second->worker_path_pins.clear();
#endif
          entry.second->worker_tree.reset();
        }
#ifdef _WIN32
        impl_->managed_root_pins.clear();
#endif
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
#ifdef _WIN32
        record->input_pins.clear();
        record->worker_path_pins.clear();
#endif
        record->worker_tree.reset();
      }
#ifdef _WIN32
      impl_->managed_root_pins.clear();
#endif
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
#ifdef _WIN32
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
#endif
    }
    if (
      request.scene_cache_max_bytes < min_cache_bytes ||
      request.scene_cache_max_bytes > max_cache_bytes
    ) {
      return {
        .code = error_code_e::invalid_request,
        .error =
          "transient raster limit must be between 16 MiB and 64 GiB",
      };
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
#ifdef _WIN32
    std::optional<std::vector<pinned_handle_t>> input_pins;
#endif
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
#ifdef _WIN32
          input_pins =
            pin_no_reparse_components(normalized_input, path_error);
          if (!input_pins) {
            throw std::runtime_error(path_error);
          }
#endif
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
    if (
      request.operation == operation_e::convert &&
      impl_->config.probe_media_tools
    ) {
      // Hardware NVENC probing is itself GPU work. Defer it until a user
      // explicitly starts conversion and hold the same exclusive lease as the
      // worker, so host startup can never race a fast live-stream connection.
      std::string codec_probe_error;
      if (!probe_ffmpeg_codec_runtime(
            impl_->config,
            impl_->ffmpeg,
            effective_codec,
            codec_probe_error
          )) {
        return {
          .code = error_code_e::unavailable,
          .error =
            "the requested 10-bit NVIDIA encoder failed its runtime preflight: " +
            codec_probe_error,
        };
      }
    }

    job_snapshot_t snapshot;
    snapshot.state = job_state_e::queued;
    snapshot.operation = request.operation;
    snapshot.input_path = *input;
    // Analysis never invokes an encoder. Keep one canonical value in the
    // persisted worker schema without making NVENC availability an admission
    // requirement for evaluate-only jobs.
    snapshot.codec = effective_codec;
    snapshot.scene_cache_max_bytes = request.scene_cache_max_bytes;
    // Legacy clients may still send the retired cache policy. Direct causal conversion has
    // no administrative scene split, so normalize persisted state to the only valid behavior.
    snapshot.cache_budget_policy = cache_budget_policy_e::fail;
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
#ifdef _WIN32
    std::optional<pinned_handle_t> worker_result_pin;
#endif
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
#ifdef _WIN32
          worker_result_pin = pin_managed_directory(
            worker_job_directory / "result",
            "offline worker result directory",
            pin_error
          );
          if (!worker_result_pin) {
            throw std::runtime_error(pin_error);
          }
#endif
        }, path_error)) {
#ifdef _WIN32
      worker_result_pin.reset();
#endif
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
#ifdef _WIN32
    record->input_pins = std::move(*input_pins);
    record->worker_path_pins.emplace_back(std::move(*worker_result_pin));
#endif
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
#ifdef _WIN32
      record->worker_path_pins.clear();
#endif
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
#ifdef _WIN32
      record->worker_path_pins.clear();
#endif
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

  scene_audit_reply_t job_service_t::scene_audit(
    const std::string_view id
  ) const {
    std::shared_ptr<impl_t::record_t> record;
    job_state_e state_at_lookup = job_state_e::queued;
    std::optional<std::uint64_t> expected_scene_count;
    {
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
      record = iterator->second;
      state_at_lookup = record->snapshot.state;
      expected_scene_count = record->snapshot.progress.scene_count;
    }
    if (!is_terminal(state_at_lookup)) {
      return {
        .code = error_code_e::conflict,
        .error = "scene audit is available only after the job becomes terminal",
      };
    }
    const bool completed_job = state_at_lookup == job_state_e::complete;
    std::optional<std::string> expected_audit_sha256;
    if (
      record->snapshot.worker_result.is_object() &&
      record->snapshot.worker_result.contains("scene_audit_sha256") &&
      record->snapshot.worker_result["scene_audit_sha256"].is_string()
    ) {
      expected_audit_sha256 =
        record->snapshot.worker_result["scene_audit_sha256"]
          .get<std::string>();
    }
    if (
      !expected_audit_sha256 ||
      expected_audit_sha256->size() != 64
    ) {
      return {
        .code = error_code_e::io_error,
        .error =
          "scene audit is unavailable because its terminal identity was not "
          "attested",
      };
    }

    // Never accept an artifact path from HTTP or from the child result document.
    // The only readable path is the manager-owned path fixed when this job was admitted.
    std::lock_guard artifact_lock {record->artifact_mutex};
    if (!record->worker_tree) {
      return {
        .code = error_code_e::io_error,
        .error = "scene audit worker tree is not identity-pinned",
      };
    }
#ifdef _WIN32
    std::string identity_error;
    if (
      !revalidate_pins(
        impl_->managed_root_pins,
        "offline managed roots",
        identity_error
      ) ||
      record->worker_path_pins.empty() ||
      !revalidate_pins(
        record->worker_path_pins,
        "offline worker job paths",
        identity_error
      )
    ) {
      return {
        .code = error_code_e::io_error,
        .error =
          identity_error.empty() ?
            "scene audit path is not identity-pinned" :
            identity_error,
      };
    }
#endif
    const auto audit_path =
      record->worker.result_directory / "scene-audit.json";
    std::optional<nlohmann::json> audit;
    std::string read_error;
    std::string observed_audit_sha256;
    if (!run_bound_user_filesystem_action(
          impl_->config.expected_user_id,
          [&]() {
          audit = read_json_contract(
            audit_path,
            read_error,
            max_scene_audit_bytes,
            &observed_audit_sha256
          );
        }, read_error)) {
      return {
        .code = error_code_e::io_error,
        .error = read_error,
      };
    }
    if (!audit) {
      return {
        .code = error_code_e::io_error,
        .error = "scene audit is unavailable: " + read_error,
      };
    }
    if (observed_audit_sha256 != *expected_audit_sha256) {
      return {
        .code = error_code_e::io_error,
        .error =
          "scene audit identity changed after the worker became terminal",
      };
    }
    try {
      const auto typed_audit = wire::parse_scene_audit_contract(*audit);
      const bool complete_audit = typed_audit.status == "complete";
      const bool partial_audit = typed_audit.status == "running";
      if (
        (
          completed_job ?
            !complete_audit :
            !(complete_audit || partial_audit)
        ) ||
        (
          complete_audit &&
          expected_scene_count &&
          typed_audit.scenes.size() != *expected_scene_count
        )
      ) {
        return {
          .code = error_code_e::io_error,
          .error = "scene audit contract is invalid",
        };
      }
      (*audit)["availability"] = complete_audit ? "complete" : "partial";
      (*audit)["job_terminal_state"] = to_string(state_at_lookup);
    } catch (const std::exception &exception) {
      return {
        .code = error_code_e::io_error,
        .error =
          std::string {"scene audit contract is invalid: "} +
          exception.what(),
      };
    }
    return {
      .ok = true,
      .audit = std::move(*audit),
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

    const bool roots_view = !request.path || request.path->empty();
    fs::path requested_path;
    if (!roots_view) {
      requested_path = request.path->lexically_normal();
      const auto serialized_path = path_to_utf8(requested_path);
      if (serialized_path.size() > 32ull * 1024ull) {
        return {
          .code = error_code_e::invalid_request,
          .error = "browse path exceeds the 32 KiB UTF-8 limit",
        };
      }
      std::string syntax_error;
      if (!validate_local_path_syntax(
            requested_path,
            "browse path",
            syntax_error,
            false
          )) {
        return {
          .code = error_code_e::invalid_request,
          .error = syntax_error,
        };
      }
      const auto relative_path = requested_path.relative_path();
      const auto component_count = static_cast<std::size_t>(std::distance(
        relative_path.begin(),
        relative_path.end()
      ));
      if (component_count > max_browse_path_components) {
        return {
          .code = error_code_e::invalid_request,
          .error = "browse path contains too many components",
        };
      }
    }

    std::vector<browse_entry_record_t> records;
    std::optional<fs::path> resolved_path;
    std::optional<fs::path> parent_path;
    bool truncated = false;
    bool action_invoked = false;
    error_code_e action_error_code = error_code_e::invalid_request;
    std::string action_error;
    const auto browse_deadline =
      std::chrono::steady_clock::now() + max_browse_duration;
    std::size_t serialized_response_bytes = 0;
    if (!run_bound_user_filesystem_action(
          expected_user_id,
          [&]() {
          action_invoked = true;
          if (roots_view) {
            action_error_code = error_code_e::io_error;
            auto roots = local_browse_roots(action_error);
            if (!roots) {
              throw std::runtime_error(action_error);
            }
            records = std::move(*roots);
            return;
          }

#ifdef _WIN32
          std::string bound_syntax_error;
          if (!validate_local_path_syntax(
                requested_path,
                "browse path",
                bound_syntax_error
              )) {
            action_error = std::move(bound_syntax_error);
            throw std::runtime_error(action_error);
          }
          bool requested_file = false;
          auto path_pins = pin_browse_directory(requested_path, action_error);
          if (!path_pins) {
            std::string file_error;
            path_pins = pin_no_reparse_components_impl(
              requested_path,
              pinned_leaf_kind_e::regular_file,
              false,
              "browse path",
              file_error
            );
            if (!path_pins) {
              action_error = std::move(file_error);
              throw std::runtime_error(action_error);
            }
            requested_file = true;
          }
#else
          bool requested_file = false;
          auto current = requested_path.root_path();
          for (const auto &component : requested_path.relative_path()) {
            if (component == fs::path {"."}) {
              continue;
            }
            current /= component;
            std::error_code status_error;
            const auto status = fs::symlink_status(current, status_error);
            if (status_error || fs::is_symlink(status)) {
              action_error =
                "browse path must exist without symbolic links";
              throw std::runtime_error(action_error);
            }
            if (current == requested_path) {
              if (fs::is_regular_file(status)) {
                requested_file = true;
              } else if (!fs::is_directory(status)) {
                action_error =
                  "browse path must be an existing file or directory";
                throw std::runtime_error(action_error);
              }
            } else if (!fs::is_directory(status)) {
              action_error =
                "browse path contains a non-directory component";
              throw std::runtime_error(action_error);
            }
          }
#endif

          std::error_code canonical_error;
          const auto resolved_request = fs::canonical(
            requested_path,
            canonical_error
          );
          if (canonical_error || resolved_request.empty()) {
            action_error =
              "browse path is not an existing file or directory" +
              (
                canonical_error ?
                  ": " + canonical_error.message() :
                  std::string {}
            );
            throw std::runtime_error(action_error);
          }
          resolved_path = requested_file ?
                            resolved_request.parent_path() :
                            resolved_request;
          std::string resolved_syntax_error;
          if (
            !validate_local_path_syntax(
              *resolved_path,
              "resolved browse path",
              resolved_syntax_error
            )
          ) {
            action_error = resolved_syntax_error;
            throw std::runtime_error(action_error);
          }

          if (*resolved_path != resolved_path->root_path()) {
            parent_path = resolved_path->parent_path();
          }
          serialized_response_bytes = browse_reply_t {
            .ok = true,
            .path = resolved_path,
            .parent = parent_path,
          }.json().dump().size();
          if (serialized_response_bytes > max_browse_response_bytes) {
            action_error = "browse path exceeds the response budget";
            throw std::runtime_error(action_error);
          }

          action_error_code = error_code_e::io_error;
          std::error_code iterator_error;
          fs::directory_iterator iterator {*resolved_path, iterator_error};
          const fs::directory_iterator end;
          if (iterator_error) {
            action_error =
              "cannot enumerate browse path: " + iterator_error.message();
            throw std::runtime_error(action_error);
          }
          std::size_t visited = 0;
          while (iterator != end) {
            if (
              visited >= max_browse_entries ||
              std::chrono::steady_clock::now() >= browse_deadline
            ) {
              truncated = true;
              break;
            }
            ++visited;
            const auto entry_path = iterator->path().lexically_normal();
            const auto directory = inspect_browse_entry(entry_path);
            if (
              directory &&
              (
                *directory ||
                request.type != browse_type_e::directory
              )
            ) {
              browse_entry_record_t record {
                .name = path_to_utf8(entry_path.filename()),
                .path = entry_path,
                .directory = *directory,
              };
              const auto entry = browse_entry_json(record, request.type);
              const auto entry_bytes =
                entry.dump().size() + (records.empty() ? 0ull : 1ull);
              if (
                serialized_response_bytes + entry_bytes >
                max_browse_response_bytes
              ) {
                truncated = true;
                break;
              }
              serialized_response_bytes += entry_bytes;
              records.push_back(std::move(record));
            }
            iterator.increment(iterator_error);
            if (iterator_error) {
              action_error =
                "cannot continue enumerating browse path: " +
                iterator_error.message();
              throw std::runtime_error(action_error);
            }
          }
#ifdef _WIN32
          if (!revalidate_pins(*path_pins, "browse path", action_error)) {
            throw std::runtime_error(action_error);
          }
#endif
        }, action_error)) {
      return {
        .code = action_invoked ? action_error_code : error_code_e::unavailable,
        .error = action_error,
      };
    }

    sort_browse_entries(records);
    auto entries = nlohmann::json::array();
    std::size_t final_response_bytes = browse_reply_t {
      .ok = true,
      .path = resolved_path,
      .parent = parent_path,
    }.json().dump().size();
    for (const auto &record : records) {
      auto entry = browse_entry_json(record, request.type);
      const auto entry_bytes =
        entry.dump().size() + (entries.empty() ? 0ull : 1ull);
      if (final_response_bytes + entry_bytes > max_browse_response_bytes) {
        truncated = true;
        break;
      }
      final_response_bytes += entry_bytes;
      entries.push_back(std::move(entry));
    }
    return {
      .ok = true,
      .path = std::move(resolved_path),
      .parent = std::move(parent_path),
      .entries = std::move(entries),
      .truncated = truncated,
    };
  }

  std::vector<job_snapshot_t> job_service_t::list() const {
    std::lock_guard lock {impl_->mutex};
    std::vector<job_snapshot_t> values;
    values.reserve(impl_->jobs.size());
    for (const auto &[id, record] : impl_->jobs) {
      // The Web UI polls this collection. Keep it bounded per retained job; the
      // selected-job endpoint supplies its recent decisions and worker result.
      auto summary = record->snapshot;
      summary.progress.scene_decisions = nlohmann::json::array();
      summary.worker_result = nullptr;
      values.push_back(std::move(summary));
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
        {"legacy_root", path_to_utf8(impl_->config.exports_root)},
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
      .exports_root = user_root.empty() ? fs::path {} : user_root / "exports",
      .sunshine_executable = std::move(executable),
      .sunshine_config = path_from_utf8(config::sunshine.config_file),
      .expected_user_id = valid_user_id ? active_user : std::nullopt,
    };
  }

  namespace {
    std::mutex global_mutex;
    std::shared_ptr<job_service_t> global_service;
    std::string global_initialization_error;
    bool global_initializing = false;
    bool global_shutdown_requested = false;
  }

  bool initialize(service_config_t config, std::string &error) {
    {
      std::lock_guard lock {global_mutex};
      if (global_service || global_initializing) {
        error = "offline SBS job manager is already initialized";
        return false;
      }
      if (global_shutdown_requested) {
        error =
          "offline SBS job manager cannot initialize after shutdown began";
        global_initialization_error = error;
        return false;
      }
      global_initializing = true;
      global_initialization_error.clear();
    }

    auto candidate = std::make_shared<job_service_t>(std::move(config));
    bool started = false;
    try {
      started = candidate->start(error);
    } catch (const std::exception &exception) {
      error =
        "offline SBS job manager initialization failed: "s +
        exception.what();
    } catch (...) {
      error = "offline SBS job manager initialization failed unexpectedly";
    }
    if (!started) {
      if (error.empty()) {
        error = "offline SBS job manager initialization failed";
      }
      std::lock_guard lock {global_mutex};
      global_initializing = false;
      global_initialization_error = error;
      return false;
    }

    bool shutdown_requested = false;
    {
      std::lock_guard lock {global_mutex};
      global_initializing = false;
      shutdown_requested = global_shutdown_requested;
      if (shutdown_requested) {
        error =
          "offline SBS job manager initialization was canceled by shutdown";
        global_initialization_error = error;
      } else {
        global_service = candidate;
        global_initialization_error.clear();
      }
    }
    if (shutdown_requested) {
      candidate->shutdown();
      return false;
    }
    return true;
  }

  void shutdown() {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      global_shutdown_requested = true;
      service = std::move(global_service);
    }
    if (service) {
      service->shutdown();
    }
  }

  service_reply_t create(const create_request_t &request) {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->create(request);
  }

  service_reply_t cancel(const std::string_view id) {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->cancel(id);
  }

  service_reply_t get(const std::string_view id) {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->get(id);
  }

  scene_audit_reply_t scene_audit(const std::string_view id) {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->scene_audit(id);
  }

  browse_reply_t browse(const browse_request_t &request) {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    if (!service) {
      return {
        .code = error_code_e::not_initialized,
        .error = "offline SBS job manager is not initialized",
      };
    }
    return service->browse(request);
  }

  std::vector<job_snapshot_t> list() {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    return service ? service->list() : std::vector<job_snapshot_t> {};
  }

  bool has_active_job() {
    std::shared_ptr<job_service_t> service;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
    }
    return service && service->has_active_job();
  }

  nlohmann::json capabilities() {
    std::shared_ptr<job_service_t> service;
    std::string initialization_error;
    bool initializing = false;
    {
      std::lock_guard lock {global_mutex};
      service = global_service;
      initialization_error = global_initialization_error;
      initializing = global_initializing;
    }
    if (!service) {
      return {
        {"schema", 1},
        {"available", false},
        {"initializing", initializing},
        {"python_dependency", false},
        {
          "error",
          initializing ?
            "offline SBS job manager is initializing" :
            initialization_error.empty() ?
              "offline SBS job manager is not initialized" :
              initialization_error,
        },
      };
    }
    return service->capabilities();
  }

#ifdef SUNSHINE_TESTS
  bounded_worker_log_test_result_t bound_worker_log_for_test(
    const std::vector<std::string> &chunks
  ) {
    bounded_worker_log_test_result_t result;
    bounded_native_worker_log_writer_t writer {
      [&result](const std::string_view bytes) {
        result.retained.append(bytes);
        return true;
      },
    };
    for (const auto &chunk : chunks) {
      writer.append(chunk);
    }
    result.limit_exceeded = writer.limit_exceeded();
    result.write_failed = writer.write_failed();
    return result;
  }

  bounded_worker_log_test_result_t capture_worker_log_pipe_for_test(
    const fs::path &output,
    const std::vector<std::string> &chunks,
    std::string &error
  ) {
    bounded_worker_log_test_result_t result;
    auto *raw_output = open_binary_log(output);
    if (!raw_output) {
      error = "cannot create test worker log";
      result.write_failed = true;
      return result;
    }
    {
      bounded_native_worker_log_pipe_t pipe {raw_output};
      if (!pipe.initialize(error)) {
        result.write_failed = true;
        return result;
      }
      for (const auto &chunk : chunks) {
        if (
          std::fwrite(
            chunk.data(),
            1,
            chunk.size(),
            pipe.child_stream()
          ) != chunk.size() ||
          std::fflush(pipe.child_stream()) != 0
        ) {
          error = "cannot write test worker diagnostic pipe";
          break;
        }
      }
      pipe.close_parent_writer();
      pipe.finish();
      result.limit_exceeded = pipe.limit_exceeded();
      result.write_failed = pipe.failed() || !error.empty();
      if (pipe.failed() && error.empty()) {
        error = pipe.failure();
      }
    }

    std::ifstream input(output, std::ios::binary);
    if (!input) {
      error = "cannot read retained test worker log";
      result.write_failed = true;
      return result;
    }
    result.retained.assign(
      std::istreambuf_iterator<char> {input},
      std::istreambuf_iterator<char> {}
    );
    return result;
  }

  bool abort_worker_log_pipe_with_open_writer_for_test(
    const fs::path &output,
    std::string &error
  ) {
    auto *raw_output = open_binary_log(output);
    if (!raw_output) {
      error = "cannot create test worker log";
      return false;
    }
    bounded_native_worker_log_pipe_t pipe {raw_output};
    if (!pipe.initialize(error)) {
      return false;
    }

#ifdef _WIN32
    const auto source_handle = reinterpret_cast<HANDLE>(
      _get_osfhandle(_fileno(pipe.child_stream()))
    );
    HANDLE duplicate_handle = INVALID_HANDLE_VALUE;
    if (
      source_handle == INVALID_HANDLE_VALUE ||
      !DuplicateHandle(
        GetCurrentProcess(),
        source_handle,
        GetCurrentProcess(),
        &duplicate_handle,
        0,
        FALSE,
        DUPLICATE_SAME_ACCESS
      )
    ) {
      error =
        "cannot duplicate test worker diagnostic writer (Windows error " +
        std::to_string(GetLastError()) + ")";
      pipe.abort();
      return false;
    }
    pipe.close_parent_writer();
    pipe.abort();
    CloseHandle(duplicate_handle);
#else
    const auto duplicate_descriptor = ::dup(::fileno(pipe.child_stream()));
    if (duplicate_descriptor == -1) {
      error =
        "cannot duplicate test worker diagnostic writer: " +
        std::generic_category().message(errno);
      pipe.abort();
      return false;
    }
    pipe.close_parent_writer();
    pipe.abort();
    ::close(duplicate_descriptor);
#endif

    if (pipe.failed()) {
      error = pipe.failure();
      return false;
    }
    return true;
  }

  std::size_t native_worker_log_max_bytes_for_test() {
    return max_native_worker_log_bytes;
  }

  std::string build_worker_command_for_test(
    const fs::path &sunshine_executable,
    const fs::path &sunshine_config,
    const fs::path &worker_spec,
    const std::string_view worker_spec_sha256
  ) {
    return build_worker_command(
      sunshine_executable,
      sunshine_config,
      worker_spec,
      worker_spec_sha256
    );
  }

#ifdef _WIN32
  bool windows_publish_identity_attributes_accepted_for_test(
    const std::uint32_t attributes,
    const std::uint64_t size
  ) {
    return windows_publish_identity_attributes_accepted(attributes, size);
  }
#endif
#endif
}  // namespace offline_sbs

/**
 * @file src/offline_sbs_job_files.cpp
 * @brief Identity-bound host filesystem access for the offline job manager.
 */

#include "offline_sbs_job_files.h"

#include "offline_sbs_filesystem.h"
#include "offline_sbs_wire_contract.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cwchar>
#include <cwctype>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "platform/common.h"

#include "logging.h"

using namespace std::literals;

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace job_detail;

  namespace {
    constexpr std::size_t max_browse_entries = 4096;
    constexpr std::size_t max_browse_path_components = 256;
    constexpr std::size_t max_browse_response_bytes = 512ull * 1024ull;
    constexpr auto max_browse_duration = std::chrono::seconds(2);

    bool windows_publish_identity_attributes_accepted(
      const DWORD attributes,
      const std::uint64_t size
    ) {
      return size > 0 &&
             (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0 &&
             (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
    }

    std::optional<nlohmann::json> file_publish_identity(
      const fs::path &path,
      std::string &error
    ) {
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
    }

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

    struct browse_entry_record_t {
      std::string name;
      fs::path path;
      bool directory = false;
      std::wstring folded_filename {};
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
    }

    void sort_browse_entries(std::vector<browse_entry_record_t> &entries) {
      if (entries.size() < 2) {
        return;
      }
      for (auto &entry : entries) {
        entry.folded_filename = entry.path.filename().native();
        std::ranges::transform(
          entry.folded_filename,
          entry.folded_filename.begin(),
          [](const wchar_t character) {
            return static_cast<wchar_t>(std::towlower(character));
          }
        );
      }
      std::ranges::sort(entries, [](const auto &left, const auto &right) {
        if (left.directory != right.directory) {
          return left.directory;
        }
        if (left.folded_filename != right.folded_filename) {
          return left.folded_filename < right.folded_filename;
        }
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
      return roots;
    }
  }  // namespace

  namespace job_detail {
    // Generic '/' separators, as job records store paths. The worker's path_utf8() keeps native
    // separators for FFmpeg command lines; do not merge them.
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

    directory_creation_pins_t::~directory_creation_pins_t() {
      close();
    }

    directory_creation_pins_t::directory_creation_pins_t(
      directory_creation_pins_t &&other
    ) noexcept {
      handles_.swap(other.handles_);
    }

    directory_creation_pins_t &directory_creation_pins_t::operator=(
      directory_creation_pins_t &&other
    ) noexcept {
      if (this != &other) {
        close();
        handles_.swap(other.handles_);
      }
      return *this;
    }

    void directory_creation_pins_t::retain(const HANDLE handle) {
      try {
        handles_.push_back(handle);
      } catch (...) {
        CloseHandle(handle);
        throw;
      }
    }

    void directory_creation_pins_t::close() noexcept {
      for (const auto handle : handles_) {
        CloseHandle(handle);
      }
      handles_.clear();
    }

    directory_creation_pins_t create_user_directory(
      const fs::path &directory,
      const std::string_view description
    ) {
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

    bool regular_nonempty_file(const fs::path &path) {
      std::error_code ec;
      return fs::is_regular_file(path, ec) && !ec &&
             fs::file_size(path, ec) > 0 && !ec;
    }

    bool file_matches_publish_identity(
      const fs::path &path,
      const nlohmann::json &identity
    ) {
      std::string error;
      const auto actual = file_publish_identity(path, error);
      return actual && *actual == identity;
    }

    bool validate_local_path_syntax(
      const fs::path &path,
      const std::string_view purpose,
      std::string &error,
      const bool check_drive_type
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

    bool pinned_identity_t::matches(
      const BY_HANDLE_FILE_INFORMATION &information
    ) const {
      return
        volume_serial == information.dwVolumeSerialNumber &&
        file_index_high == information.nFileIndexHigh &&
        file_index_low == information.nFileIndexLow &&
        directory ==
          ((information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
    }

    pinned_handle_t::pinned_handle_t(
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

    pinned_handle_t::~pinned_handle_t() {
      if (handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
      }
    }

    pinned_handle_t::pinned_handle_t(pinned_handle_t &&other) noexcept:
        handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
        path_(std::move(other.path_)),
        identity_(other.identity_) {
    }

    pinned_handle_t &pinned_handle_t::operator=(pinned_handle_t &&other) noexcept {
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

    bool pinned_handle_t::revalidate(std::string &error) const {
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

    std::optional<std::uint64_t> retained_staging_charge(
      const fs::path &path,
      std::string &error
    ) {
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
      return std::max(size, retained_staging_minimum_charge_bytes);
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
      return safe_filesystem::publish_bytes_atomically(
        path, contents, safe_filesystem::publication_e::durable, error
      );
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
      const std::uintmax_t max_bytes,
      bool *acquisition_failed
    ) {
      if (acquisition_failed) {
        *acquisition_failed = false;
      }
      std::string serialized;
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
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
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
        if (!inspected && acquisition_failed) {
          *acquisition_failed = true;
        }
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
          if (acquisition_failed) {
            *acquisition_failed = true;
          }
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
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
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
        if (acquisition_failed) {
          *acquisition_failed = true;
        }
        error = "contract size changed while it was being read";
        return std::nullopt;
      }
      CloseHandle(handle);
      try {
        auto value = wire::parse_json_without_duplicate_keys(serialized);
        if (!value.is_object()) {
          error = "contract root is not an object";
          return std::nullopt;
        }
        return value;
      } catch (const std::exception &exception) {
        error = std::string {"cannot parse contract: "} + exception.what();
        return std::nullopt;
      }
    }
  }  // namespace job_detail

  browse_reply_t job_detail::browse_host_filesystem(
    const browse_request_t &request,
    const std::optional<std::string> &expected_user_id
  ) {
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
          if (!revalidate_pins(*path_pins, "browse path", action_error)) {
            throw std::runtime_error(action_error);
          }
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

#ifdef SUNSHINE_TESTS
  bool windows_publish_identity_attributes_accepted_for_test(
    const std::uint32_t attributes,
    const std::uint64_t size
  ) {
    return windows_publish_identity_attributes_accepted(attributes, size);
  }
#endif
}  // namespace offline_sbs

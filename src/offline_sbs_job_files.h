/**
 * @file src/offline_sbs_job_files.h
 * @brief Job-manager-private, identity-bound host filesystem access.
 *
 * Every user-visible path the manager touches is resolved without following reparse points and
 * pinned by file identity. Filesystem actions run under the interactive account that owns the
 * offline workspaces, and the host browse view is bounded in entries, bytes and time.
 */
#pragma once

#include "offline_sbs_job.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace offline_sbs::job_detail {
  namespace fs = std::filesystem;

  // Unattested staging files are intentionally never deleted automatically.
  // Charge at least 1 MiB apiece so repeated zero-byte failures cannot grow
  // an unbounded retained staging set while consuming no byte quota.
  inline constexpr std::uint64_t retained_staging_minimum_charge_bytes =
    1ull * 1024ull * 1024ull;
  inline constexpr std::uintmax_t max_contract_bytes = 16ull * 1024ull * 1024ull;

  // Generic '/' separators, as job records store paths.
  std::string path_to_utf8(const fs::path &path);
  fs::path path_from_utf8(const std::string &value);

  class directory_creation_pins_t {
  public:
    directory_creation_pins_t() = default;
    ~directory_creation_pins_t();
    directory_creation_pins_t(const directory_creation_pins_t &) = delete;
    directory_creation_pins_t &operator=(
      const directory_creation_pins_t &
    ) = delete;
    directory_creation_pins_t(
      directory_creation_pins_t &&other
    ) noexcept;
    directory_creation_pins_t &operator=(
      directory_creation_pins_t &&other
    ) noexcept;
    void retain(const HANDLE handle);

  private:
    void close() noexcept;
    std::vector<HANDLE> handles_;
  };

  directory_creation_pins_t create_user_directory(
    const fs::path &directory,
    std::string_view description
  );

  bool run_bound_user_filesystem_action(
    const std::optional<std::string> &expected_user_id,
    const std::function<void()> &action,
    std::string &error
  );

  bool regular_nonempty_file(const fs::path &path);

  bool file_matches_publish_identity(
    const fs::path &path,
    const nlohmann::json &identity
  );

  bool validate_local_path_syntax(
    const fs::path &path,
    std::string_view purpose,
    std::string &error,
    bool check_drive_type = true
  );

  bool validate_local_input_path_syntax(
    const fs::path &path,
    std::string &error
  );

  struct pinned_identity_t {
    DWORD volume_serial = 0;
    DWORD file_index_high = 0;
    DWORD file_index_low = 0;
    bool directory = false;
    [[nodiscard]] bool matches(
      const BY_HANDLE_FILE_INFORMATION &information
    ) const;
  };

  class pinned_handle_t {
  public:
    pinned_handle_t(
      HANDLE handle,
      fs::path path,
      const BY_HANDLE_FILE_INFORMATION &information
    );
    ~pinned_handle_t();
    pinned_handle_t(const pinned_handle_t &) = delete;
    pinned_handle_t &operator=(const pinned_handle_t &) = delete;
    pinned_handle_t(pinned_handle_t &&other) noexcept;
    pinned_handle_t &operator=(pinned_handle_t &&other) noexcept;
    [[nodiscard]] bool revalidate(std::string &error) const;

  private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    fs::path path_;
    pinned_identity_t identity_;
  };

  std::optional<std::vector<pinned_handle_t>> pin_no_reparse_components(
    const fs::path &path,
    std::string &error
  );

  std::optional<std::vector<pinned_handle_t>> pin_managed_root_components(
    const fs::path &path,
    std::string_view purpose,
    std::string &error
  );

  std::optional<pinned_handle_t> pin_managed_directory(
    const fs::path &path,
    std::string_view purpose,
    std::string &error
  );

  bool revalidate_pins(
    const std::vector<pinned_handle_t> &pins,
    std::string_view purpose,
    std::string &error
  );

  std::optional<std::vector<pinned_handle_t>> pin_publication_directory(
    const fs::path &path,
    std::string &error
  );

  std::optional<std::uint64_t> retained_staging_charge(
    const fs::path &path,
    std::string &error
  );

  std::optional<fs::path> canonical_regular_file(
    const fs::path &path,
    std::string &error
  );

  bool write_text_atomically(
    const fs::path &path,
    const std::string &contents,
    std::string &error
  );

  bool write_json_atomically(
    const fs::path &path,
    const nlohmann::json &value,
    std::string &error
  );

  std::optional<nlohmann::json> read_json_contract(
    const fs::path &path,
    std::string &error,
    std::uintmax_t max_bytes = max_contract_bytes,
    bool *acquisition_failed = nullptr
  );

  // The bounded local-drive or directory view behind job_service_t::browse(), run under
  // expected_user_id. The caller holds the service's single browse slot.
  browse_reply_t browse_host_filesystem(
    const browse_request_t &request,
    const std::optional<std::string> &expected_user_id
  );
}  // namespace offline_sbs::job_detail

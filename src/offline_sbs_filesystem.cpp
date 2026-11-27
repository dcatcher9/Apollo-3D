/**
 * @file src/offline_sbs_filesystem.cpp
 * @brief No-follow filesystem traversal and atomic publication for offline SBS artifacts.
 */


#include "offline_sbs_filesystem.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

  #include <windows.h>

namespace offline_sbs::safe_filesystem {
  namespace fs = std::filesystem;

  namespace {
    constexpr std::string_view missing_tree_error =
      "no-follow tree root does not exist";
    constexpr std::size_t max_tree_depth = 256;
    constexpr std::size_t max_entries_per_directory = 1'000'000;
    constexpr std::size_t max_visited_tree_nodes = 1'000'000;
    constexpr int max_delete_passes = 32;

    std::string path_text(const fs::path &path) {
      try {
        const auto value = path.generic_u8string();
        return {
          reinterpret_cast<const char *>(value.data()),
          value.size(),
        };
      } catch (...) {
        return "<unprintable path>";
      }
    }

    bool add_measurement(
      std::uint64_t &value,
      const std::uint64_t increment,
      std::string &error
    ) {
      if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
        error = "no-follow tree measurement overflowed";
        return false;
      }
      value += increment;
      return true;
    }

    bool validate_direct_child_name(
      const fs::path &child_name,
      std::string &error
    ) {
      const auto &native = child_name.native();
      if (
        native.empty() ||
        child_name.has_root_path() ||
        child_name.parent_path() != fs::path {} ||
        child_name.filename() != child_name ||
        child_name == fs::path {"."} ||
        child_name == fs::path {".."} ||
        native.find(
          L'\0'
        ) != fs::path::string_type::npos
      ) {
        error =
          "no-follow child name must be one safe relative filename "
          "component";
        return false;
      }
      if (
        native.find(L'\\') != std::wstring::npos ||
        native.find(L'/') != std::wstring::npos ||
        native.find(L':') != std::wstring::npos
      ) {
        error =
          "no-follow child name must be one safe relative filename "
          "component";
        return false;
      }
      return true;
    }

    class unique_handle_t {
    public:
      unique_handle_t() = default;

      explicit unique_handle_t(const HANDLE handle):
          handle_(handle) {
      }

      ~unique_handle_t() {
        reset();
      }

      unique_handle_t(const unique_handle_t &) = delete;
      unique_handle_t &operator=(const unique_handle_t &) = delete;

      unique_handle_t(unique_handle_t &&other) noexcept:
          handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)) {
      }

      unique_handle_t &operator=(unique_handle_t &&other) noexcept {
        if (this != &other) {
          reset();
          handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
        }
        return *this;
      }

      [[nodiscard]] explicit operator bool() const noexcept {
        return handle_ != INVALID_HANDLE_VALUE;
      }

      [[nodiscard]] HANDLE get() const noexcept {
        return handle_;
      }

      void reset(const HANDLE replacement = INVALID_HANDLE_VALUE) noexcept {
        if (handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(handle_);
        }
        handle_ = replacement;
      }

    private:
      HANDLE handle_ = INVALID_HANDLE_VALUE;
    };

    struct opened_node_t {
      fs::path path;
      unique_handle_t pin;
      unique_handle_t enumeration;
      FILE_ATTRIBUTE_TAG_INFO attributes {};
      FILE_ID_INFO identity {};
    };

    struct enumerated_entry_t {
      std::wstring name;
      FILE_ID_INFO identity {};
    };

    enum class open_node_status_e {
      opened,
      missing,
      changed,
      failed,
    };

    enum class disposition_status_e {
      removed,
      not_empty,
      failed,
    };

    bool missing_windows_error(const DWORD error) {
      return error == ERROR_FILE_NOT_FOUND ||
             error == ERROR_PATH_NOT_FOUND ||
             error == ERROR_INVALID_NAME;
    }

    std::string windows_error(
      const std::string_view operation,
      const fs::path &path,
      const DWORD error
    ) {
      return std::string {operation} + " [" + path_text(path) +
             "] (Windows error " + std::to_string(error) + ")";
    }

    std::optional<fs::path> normalized_local_path(
      const fs::path &root,
      std::string &error
    ) {
      const auto &supplied = root.native();
      if (
        supplied.size() < 3 ||
        !(
          (supplied[0] >= L'A' && supplied[0] <= L'Z') ||
          (supplied[0] >= L'a' && supplied[0] <= L'z')
        ) ||
        supplied[1] != L':' ||
        (supplied[2] != L'\\' && supplied[2] != L'/')
      ) {
        error =
          "no-follow tree root must use an absolute local drive path";
        return std::nullopt;
      }
      std::error_code ec;
      auto absolute = fs::absolute(root, ec);
      if (ec) {
        error =
          "cannot make no-follow tree root absolute: " + ec.message();
        return std::nullopt;
      }
      absolute = absolute.lexically_normal();
      absolute.make_preferred();
      const auto &native = absolute.native();
      if (
        native.size() < 3 ||
        !(
          (native[0] >= L'A' && native[0] <= L'Z') ||
          (native[0] >= L'a' && native[0] <= L'z')
        ) ||
        native[1] != L':' ||
        (native[2] != L'\\' && native[2] != L'/')
      ) {
        error =
          "no-follow tree root must use an absolute local drive path";
        return std::nullopt;
      }
      const auto relative = absolute.relative_path();
      if (relative.empty()) {
        error = "refusing to open a volume root as a removable tree";
        return std::nullopt;
      }
      for (const auto &component : relative) {
        const auto &part = component.native();
        if (
          part.empty() || part == L"." || part == L".." ||
          part.find(L':') != std::wstring::npos
        ) {
          error = "no-follow tree root contains an unsafe path component";
          return std::nullopt;
        }
      }
      return absolute;
    }

    std::wstring extended_windows_path(const fs::path &path) {
      const auto &native = path.native();
      if (native.starts_with(LR"(\\?\)")) {
        return native;
      }
      return LR"(\\?\)" + native;
    }

    bool query_attribute_tag(
      const HANDLE handle,
      const fs::path &path,
      FILE_ATTRIBUTE_TAG_INFO &attributes,
      std::string &error
    ) {
      if (
        !GetFileInformationByHandleEx(
          handle,
          FileAttributeTagInfo,
          &attributes,
          sizeof(attributes)
        )
      ) {
        error = windows_error(
          "cannot inspect no-follow tree entry",
          path,
          GetLastError()
        );
        return false;
      }
      return true;
    }

    bool query_file_identity(
      const HANDLE handle,
      const fs::path &path,
      FILE_ID_INFO &identity,
      std::string &error
    ) {
      if (
        !GetFileInformationByHandleEx(
          handle,
          FileIdInfo,
          &identity,
          sizeof(identity)
        )
      ) {
        error = windows_error(
          "cannot query no-follow tree identity",
          path,
          GetLastError()
        );
        return false;
      }
      return true;
    }

    bool same_file_id(
      const FILE_ID_128 &left,
      const FILE_ID_128 &right
    ) {
      return std::memcmp(
               left.Identifier,
               right.Identifier,
               sizeof(left.Identifier)
             ) == 0;
    }

    bool open_enumeration_handle(
      opened_node_t &node,
      std::string &error
    ) {
      const auto path = extended_windows_path(node.path);
      const HANDLE handle = CreateFileW(
        path.c_str(),
        FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        error = windows_error(
          "cannot enumerate pinned no-follow directory",
          node.path,
          GetLastError()
        );
        return false;
      }
      unique_handle_t enumeration {handle};
      FILE_ATTRIBUTE_TAG_INFO attributes {};
      FILE_ID_INFO identity {};
      if (
        !query_attribute_tag(handle, node.path, attributes, error) ||
        !query_file_identity(handle, node.path, identity, error)
      ) {
        return false;
      }
      if (
        (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
        (attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
        identity.VolumeSerialNumber != node.identity.VolumeSerialNumber ||
        !same_file_id(identity.FileId, node.identity.FileId)
      ) {
        error =
          "no-follow directory identity changed while opening enumeration "
          "handle [" +
          path_text(node.path) + "]";
        return false;
      }
      node.enumeration = std::move(enumeration);
      return true;
    }

    open_node_status_e open_node(
      const fs::path &path,
      const tree_access_e access,
      const FILE_ID_INFO *expected_identity,
      opened_node_t &node,
      std::string &error
    ) {
      const DWORD desired_access =
        FILE_READ_ATTRIBUTES |
        (access == tree_access_e::remove ?
           (FILE_WRITE_ATTRIBUTES | DELETE) :
           0);
      const auto extended = extended_windows_path(path);
      const DWORD share_mode =
        FILE_SHARE_READ |
        FILE_SHARE_WRITE |
        (access == tree_access_e::measure_only ?
           FILE_SHARE_DELETE :
           0);
      const HANDLE handle = CreateFileW(
        extended.c_str(),
        desired_access,
        share_mode,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        const auto open_error = GetLastError();
        if (missing_windows_error(open_error)) {
          return open_node_status_e::missing;
        }
        error = windows_error(
          "cannot pin no-follow tree entry",
          path,
          open_error
        );
        return open_node_status_e::failed;
      }

      opened_node_t candidate;
      candidate.path = path;
      candidate.pin.reset(handle);
      if (
        !query_attribute_tag(
          candidate.pin.get(),
          path,
          candidate.attributes,
          error
        ) ||
        !query_file_identity(
          candidate.pin.get(),
          path,
          candidate.identity,
          error
        )
      ) {
        return open_node_status_e::failed;
      }
      if (
        expected_identity &&
        (
          expected_identity->VolumeSerialNumber !=
            candidate.identity.VolumeSerialNumber ||
          !same_file_id(
            expected_identity->FileId,
            candidate.identity.FileId
          )
        )
      ) {
        return open_node_status_e::changed;
      }

      if (
        (candidate.attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ==
          0 &&
        (candidate.attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
        !open_enumeration_handle(candidate, error)
      ) {
        return open_node_status_e::failed;
      }
      node = std::move(candidate);
      return open_node_status_e::opened;
    }

    bool pin_ancestor(
      const fs::path &path,
      unique_handle_t &pin,
      std::string &error
    ) {
      const auto extended = extended_windows_path(path);
      const DWORD share_mode =
        FILE_SHARE_READ |
        FILE_SHARE_WRITE;
      const HANDLE handle = CreateFileW(
        extended.c_str(),
        FILE_READ_ATTRIBUTES,
        share_mode,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        const auto open_error = GetLastError();
        if (missing_windows_error(open_error)) {
          error = std::string {missing_tree_error};
        } else {
          error = windows_error(
            "cannot pin no-follow tree ancestor",
            path,
            open_error
          );
        }
        return false;
      }
      unique_handle_t candidate {handle};
      FILE_ATTRIBUTE_TAG_INFO attributes {};
      if (!query_attribute_tag(handle, path, attributes, error)) {
        return false;
      }
      if (
        (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
      ) {
        error =
          "no-follow tree root traverses a reparse-point ancestor [" +
          path_text(path) + "]";
        return false;
      }
      if ((attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
        error =
          "no-follow tree ancestor is not a directory [" +
          path_text(path) + "]";
        return false;
      }
      pin = std::move(candidate);
      return true;
    }

    bool enumerate_directory(
      opened_node_t &node,
      std::vector<enumerated_entry_t> &entries,
      std::string &error
    ) {
      entries.clear();
      if (!node.enumeration && !open_enumeration_handle(node, error)) {
        return false;
      }

      constexpr std::size_t buffer_size = 64 * 1024;
      alignas(FILE_ID_EXTD_DIR_INFO)
        std::array<std::byte, buffer_size>
          buffer {};
      bool restart = true;
      while (true) {
        const auto information_class =
          restart ?
            FileIdExtdDirectoryRestartInfo :
            FileIdExtdDirectoryInfo;
        if (
          !GetFileInformationByHandleEx(
            node.enumeration.get(),
            information_class,
            buffer.data(),
            static_cast<DWORD>(buffer.size())
          )
        ) {
          const auto enumeration_error = GetLastError();
          if (enumeration_error == ERROR_NO_MORE_FILES) {
            return true;
          }
          error = windows_error(
            "cannot enumerate no-follow directory",
            node.path,
            enumeration_error
          );
          return false;
        }
        restart = false;

        std::size_t offset = 0;
        while (true) {
          if (
            offset >
            buffer.size() - offsetof(FILE_ID_EXTD_DIR_INFO, FileName)
          ) {
            error =
              "Windows returned a malformed no-follow directory entry";
            return false;
          }
          const auto *entry =
            reinterpret_cast<const FILE_ID_EXTD_DIR_INFO *>(
              buffer.data() + offset
            );
          const auto available = buffer.size() - offset;
          const auto required =
            offsetof(FILE_ID_EXTD_DIR_INFO, FileName) +
            static_cast<std::size_t>(entry->FileNameLength);
          if (
            entry->FileNameLength % sizeof(wchar_t) != 0 ||
            required > available
          ) {
            error =
              "Windows returned an invalid no-follow directory name";
            return false;
          }
          std::wstring name {
            entry->FileName,
            entry->FileNameLength / sizeof(wchar_t),
          };
          if (name != L"." && name != L"..") {
            if (
              name.empty() ||
              name.find(L'\\') != std::wstring::npos ||
              name.find(L'/') != std::wstring::npos ||
              entries.size() >= max_entries_per_directory
            ) {
              error =
                "no-follow directory contains too many or invalid entries [" +
                path_text(node.path) + "]";
              return false;
            }
            FILE_ID_INFO identity {};
            identity.VolumeSerialNumber =
              node.identity.VolumeSerialNumber;
            identity.FileId = entry->FileId;
            entries.push_back({
              .name = std::move(name),
              .identity = identity,
            });
          }

          if (entry->NextEntryOffset == 0) {
            break;
          }
          if (
            entry->NextEntryOffset < required ||
            entry->NextEntryOffset > available
          ) {
            error =
              "Windows returned a malformed no-follow directory offset";
            return false;
          }
          offset += entry->NextEntryOffset;
        }
      }
    }

    bool measure_node(
      opened_node_t &node,
      tree_measurement_t &measurement,
      const std::size_t depth,
      std::size_t &visited_nodes,
      std::string &error
    ) {
      if (depth > max_tree_depth) {
        error = "no-follow tree exceeds the supported directory depth";
        return false;
      }
      if (visited_nodes >= max_visited_tree_nodes) {
        error = "no-follow tree exceeds the supported entry count";
        return false;
      }
      ++visited_nodes;
      if (
        (node.attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0
      ) {
        return add_measurement(
          measurement.reparse_point_count,
          1,
          error
        );
      }
      if (
        (node.attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0
      ) {
        if (!add_measurement(measurement.directory_count, 1, error)) {
          return false;
        }
        std::vector<enumerated_entry_t> entries;
        if (!enumerate_directory(node, entries, error)) {
          return false;
        }
        for (const auto &entry : entries) {
          opened_node_t child;
          const auto status = open_node(
            node.path / entry.name,
            tree_access_e::measure_only,
            &entry.identity,
            child,
            error
          );
          if (
            status == open_node_status_e::missing ||
            status == open_node_status_e::changed
          ) {
            error =
              "no-follow tree changed during measurement [" +
              path_text(node.path / entry.name) + "]";
            return false;
          }
          if (
            status != open_node_status_e::opened ||
            !measure_node(
              child,
              measurement,
              depth + 1,
              visited_nodes,
              error
            )
          ) {
            return false;
          }
        }
        return true;
      }

      FILE_STANDARD_INFO standard {};
      if (
        !GetFileInformationByHandleEx(
          node.pin.get(),
          FileStandardInfo,
          &standard,
          sizeof(standard)
        )
      ) {
        error = windows_error(
          "cannot measure pinned no-follow file",
          node.path,
          GetLastError()
        );
        return false;
      }
      if (standard.EndOfFile.QuadPart < 0) {
        error =
          "pinned no-follow file has an invalid negative size [" +
          path_text(node.path) + "]";
        return false;
      }
      return add_measurement(measurement.regular_file_count, 1, error) &&
             add_measurement(
               measurement.regular_file_bytes,
               static_cast<std::uint64_t>(standard.EndOfFile.QuadPart),
               error
             );
    }

    disposition_status_e disposition_handle(
      opened_node_t &node,
      std::string &error
    ) {
      node.enumeration.reset();

      FILE_DISPOSITION_INFO_EX extended {
        .Flags =
          FILE_DISPOSITION_FLAG_DELETE |
          FILE_DISPOSITION_FLAG_POSIX_SEMANTICS |
          FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,
      };
      if (
        SetFileInformationByHandle(
          node.pin.get(),
          FileDispositionInfoEx,
          &extended,
          sizeof(extended)
        )
      ) {
        node.pin.reset();
        return disposition_status_e::removed;
      }

      auto disposition_error = GetLastError();
      if (disposition_error == ERROR_DIR_NOT_EMPTY) {
        return disposition_status_e::not_empty;
      }
      if (
        disposition_error != ERROR_INVALID_PARAMETER &&
        disposition_error != ERROR_INVALID_FUNCTION &&
        disposition_error != ERROR_NOT_SUPPORTED &&
        disposition_error != ERROR_ACCESS_DENIED
      ) {
        error = windows_error(
          "cannot disposition no-follow tree entry",
          node.path,
          disposition_error
        );
        return disposition_status_e::failed;
      }

      FILE_BASIC_INFO basic {};
      if (
        GetFileInformationByHandleEx(
          node.pin.get(),
          FileBasicInfo,
          &basic,
          sizeof(basic)
        ) &&
        (basic.FileAttributes & FILE_ATTRIBUTE_READONLY) != 0
      ) {
        basic.FileAttributes &= ~FILE_ATTRIBUTE_READONLY;
        if (
          !SetFileInformationByHandle(
            node.pin.get(),
            FileBasicInfo,
            &basic,
            sizeof(basic)
          )
        ) {
          error = windows_error(
            "cannot clear read-only no-follow tree entry",
            node.path,
            GetLastError()
          );
          return disposition_status_e::failed;
        }
      }

      FILE_DISPOSITION_INFO fallback {.DeleteFile = TRUE};
      if (
        SetFileInformationByHandle(
          node.pin.get(),
          FileDispositionInfo,
          &fallback,
          sizeof(fallback)
        )
      ) {
        node.pin.reset();
        return disposition_status_e::removed;
      }
      disposition_error = GetLastError();
      if (disposition_error == ERROR_DIR_NOT_EMPTY) {
        return disposition_status_e::not_empty;
      }
      error = windows_error(
        "cannot disposition no-follow tree entry",
        node.path,
        disposition_error
      );
      return disposition_status_e::failed;
    }

    bool remove_node(
      opened_node_t &node,
      const std::size_t depth,
      std::size_t &visited_nodes,
      std::string &error
    ) {
      if (depth > max_tree_depth) {
        error = "no-follow tree exceeds the supported directory depth";
        return false;
      }
      if (visited_nodes >= max_visited_tree_nodes) {
        error = "no-follow tree exceeds the supported entry count";
        return false;
      }
      ++visited_nodes;
      const bool reparse =
        (node.attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
      const bool directory =
        (node.attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
      if (reparse || !directory) {
        return disposition_handle(node, error) ==
               disposition_status_e::removed;
      }

      for (int pass = 0; pass < max_delete_passes; ++pass) {
        std::vector<enumerated_entry_t> entries;
        if (!enumerate_directory(node, entries, error)) {
          return false;
        }

        bool restart = false;
        for (const auto &entry : entries) {
          opened_node_t child;
          const auto status = open_node(
            node.path / entry.name,
            tree_access_e::remove,
            &entry.identity,
            child,
            error
          );
          if (
            status == open_node_status_e::missing ||
            status == open_node_status_e::changed
          ) {
            restart = true;
            break;
          }
          if (
            status != open_node_status_e::opened ||
            !remove_node(child, depth + 1, visited_nodes, error)
          ) {
            return false;
          }
        }
        if (restart) {
          continue;
        }

        const auto disposed = disposition_handle(node, error);
        if (disposed == disposition_status_e::removed) {
          return true;
        }
        if (disposed != disposition_status_e::not_empty) {
          return false;
        }
        if (!open_enumeration_handle(node, error)) {
          return false;
        }
      }
      error =
        "no-follow tree kept changing during bounded removal [" +
        path_text(node.path) + "]";
      return false;
    }

    bool windows_names_equal(
      const std::wstring &left,
      const std::wstring &right
    ) {
      if (left == right) {
        return true;
      }
      if (
        left.size() > static_cast<std::size_t>(MAXLONG) ||
        right.size() > static_cast<std::size_t>(MAXLONG)
      ) {
        return false;
      }
      return CompareStringOrdinal(
               left.data(),
               static_cast<int>(left.size()),
               right.data(),
               static_cast<int>(right.size()),
               TRUE
             ) == CSTR_EQUAL;
    }

    bool remove_windows_child(
      opened_node_t &root,
      const std::wstring &child_name,
      std::string &error
    ) {
      if (
        (root.attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
        (root.attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0
      ) {
        error =
          "no-follow child removal requires a pinned directory root [" +
          path_text(root.path) + "]";
        return false;
      }

      std::size_t visited_nodes = 0;
      for (int pass = 0; pass < max_delete_passes; ++pass) {
        std::vector<enumerated_entry_t> entries;
        if (!enumerate_directory(root, entries, error)) {
          return false;
        }

        const enumerated_entry_t *matched = nullptr;
        for (const auto &entry : entries) {
          if (!windows_names_equal(entry.name, child_name)) {
            continue;
          }
          if (matched) {
            error =
              "pinned no-follow directory has ambiguous child names [" +
              path_text(root.path / child_name) + "]";
            return false;
          }
          matched = &entry;
        }
        if (!matched) {
          return true;
        }

        opened_node_t child;
        const auto child_path = root.path / matched->name;
        const auto status = open_node(
          child_path,
          tree_access_e::remove,
          &matched->identity,
          child,
          error
        );
        if (
          status == open_node_status_e::missing ||
          status == open_node_status_e::changed
        ) {
          continue;
        }
        if (
          status != open_node_status_e::opened ||
          !remove_node(child, 1, visited_nodes, error)
        ) {
          return false;
        }
      }
      error =
        "no-follow child kept changing during bounded removal [" +
        path_text(root.path / child_name) + "]";
      return false;
    }
  }  // namespace

  struct pinned_tree_t::impl_t {
    fs::path root;
    tree_access_e access = tree_access_e::measure_only;
    bool removed = false;
    std::vector<unique_handle_t> ancestor_pins;
    opened_node_t root_node;
  };

  pinned_tree_t::pinned_tree_t(std::unique_ptr<impl_t> impl) noexcept:
      impl_(std::move(impl)) {
  }

  pinned_tree_t::pinned_tree_t(pinned_tree_t &&other) noexcept = default;

  pinned_tree_t &pinned_tree_t::operator=(pinned_tree_t &&other) noexcept =
    default;

  pinned_tree_t::~pinned_tree_t() = default;

  std::optional<pinned_tree_t> pinned_tree_t::open(
    const fs::path &root,
    const tree_access_e access,
    std::string &error
  ) {
    error.clear();
    auto normalized = normalized_local_path(root, error);
    if (!normalized) {
      return std::nullopt;
    }

    auto implementation = std::make_unique<impl_t>();
    implementation->root = *normalized;
    implementation->access = access;
    std::vector<fs::path> components;
    for (const auto &component : normalized->relative_path()) {
      if (component != L".") {
        components.push_back(component);
      }
    }
    auto current = normalized->root_path();
    implementation->ancestor_pins.reserve(components.size() - 1);
    for (std::size_t index = 0; index + 1 < components.size(); ++index) {
      current /= components[index];
      unique_handle_t ancestor;
      if (!pin_ancestor(current, ancestor, error)) {
        return std::nullopt;
      }
      implementation->ancestor_pins.emplace_back(std::move(ancestor));
    }

    const auto status = open_node(
      *normalized,
      access,
      nullptr,
      implementation->root_node,
      error
    );
    if (status == open_node_status_e::missing) {
      error = std::string {missing_tree_error};
      return std::nullopt;
    }
    if (status != open_node_status_e::opened) {
      if (error.empty()) {
        error =
          "cannot pin no-follow tree root [" +
          path_text(*normalized) + "]";
      }
      return std::nullopt;
    }
    return pinned_tree_t {std::move(implementation)};
  }

  std::optional<tree_measurement_t> pinned_tree_t::measure(
    std::string &error
  ) const {
    error.clear();
    if (!impl_ || impl_->removed) {
      error = "no-follow tree pin is empty or already removed";
      return std::nullopt;
    }
    if (
      (
        impl_->root_node.attributes.FileAttributes &
        FILE_ATTRIBUTE_REPARSE_POINT
      ) != 0
    ) {
      error =
        "no-follow tree root is a reparse point [" +
        path_text(impl_->root) + "]";
      return std::nullopt;
    }
    tree_measurement_t measurement;
    std::size_t visited_nodes = 0;
    if (
      !measure_node(
        impl_->root_node,
        measurement,
        0,
        visited_nodes,
        error
      )
    ) {
      return std::nullopt;
    }
    return measurement;
  }

  bool pinned_tree_t::remove(std::string &error) {
    error.clear();
    if (!impl_ || impl_->removed) {
      error = "no-follow tree pin is empty or already removed";
      return false;
    }
    if (impl_->access != tree_access_e::remove) {
      error =
        "no-follow tree was not opened with removal access";
      return false;
    }
    std::size_t visited_nodes = 0;
    if (!remove_node(impl_->root_node, 0, visited_nodes, error)) {
      return false;
    }
    impl_->removed = true;
    return true;
  }

  bool pinned_tree_t::remove_child(
    const fs::path &child_name,
    std::string &error
  ) {
    error.clear();
    if (!impl_ || impl_->removed) {
      error = "no-follow tree pin is empty or already removed";
      return false;
    }
    if (impl_->access != tree_access_e::remove) {
      error =
        "no-follow tree was not opened with removal access";
      return false;
    }
    if (!validate_direct_child_name(child_name, error)) {
      return false;
    }
    return remove_windows_child(
      impl_->root_node,
      child_name.native(),
      error
    );
  }

  const fs::path &pinned_tree_t::path() const noexcept {
    return impl_->root;
  }

  tree_access_e pinned_tree_t::access() const noexcept {
    return impl_->access;
  }

  std::optional<tree_measurement_t> measure_tree_no_follow(
    const fs::path &root,
    std::string &error
  ) {
    auto pin = pinned_tree_t::open(
      root,
      tree_access_e::measure_only,
      error
    );
    if (!pin) {
      if (error == missing_tree_error) {
        error.clear();
        return tree_measurement_t {.root_exists = false};
      }
      return std::nullopt;
    }
    return pin->measure(error);
  }

  bool remove_tree_no_follow(
    const fs::path &root,
    std::string &error
  ) {
    auto pin = pinned_tree_t::open(root, tree_access_e::remove, error);
    if (!pin) {
      if (error == missing_tree_error) {
        error.clear();
        return true;
      }
      return false;
    }
    return pin->remove(error);
  }

  bool publish_file_atomically(
    const fs::path &path,
    const std::function<bool(const fs::path &temporary)> &write,
    const publication_e publication,
    std::string &error
  ) {
    error.clear();
    auto temporary = path;
    temporary += L".part";
    std::error_code ignored;
    fs::remove(temporary, ignored);
    if (!write(temporary)) {
      fs::remove(temporary, ignored);
      error = "cannot write the atomic temporary file for " + path_text(path);
      return false;
    }
    // The temporary is a sibling of the final name, so the rename stays on one volume, and
    // MOVEFILE_REPLACE_EXISTING keeps a replacement atomic.
    const DWORD flags = MOVEFILE_REPLACE_EXISTING |
                        (publication == publication_e::durable ? MOVEFILE_WRITE_THROUGH : 0);
    constexpr int attempts = 50;
    for (int attempt = 0; attempt < attempts; ++attempt) {
      if (MoveFileExW(temporary.c_str(), path.c_str(), flags)) {
        return true;
      }
      const auto move_error = GetLastError();
      if (move_error != ERROR_SHARING_VIOLATION && move_error != ERROR_LOCK_VIOLATION &&
          move_error != ERROR_ACCESS_DENIED) {
        break;
      }
      if (attempt + 1 < attempts) {
        std::this_thread::sleep_for(std::chrono::milliseconds {2});
      }
    }
    fs::remove(temporary, ignored);
    error = "cannot atomically publish " + path_text(path);
    return false;
  }

  bool publish_bytes_atomically(
    const fs::path &path,
    const std::string_view bytes,
    const publication_e publication,
    std::string &error
  ) {
    const bool durable = publication == publication_e::durable;
    return publish_file_atomically(path, [&](const fs::path &temporary) {
      const HANDLE handle = CreateFileW(
        temporary.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | (durable ? FILE_FLAG_WRITE_THROUGH : 0),
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        return false;
      }
      bool written = true;
      for (std::size_t offset = 0; written && offset < bytes.size();) {
        const auto chunk = static_cast<DWORD>(
          std::min<std::size_t>(bytes.size() - offset, std::numeric_limits<DWORD>::max())
        );
        DWORD count = 0;
        written = WriteFile(handle, bytes.data() + offset, chunk, &count, nullptr) && count == chunk;
        offset += count;
      }
      written = written && (!durable || FlushFileBuffers(handle));
      return CloseHandle(handle) && written;
    }, publication, error);
  }
}  // namespace offline_sbs::safe_filesystem

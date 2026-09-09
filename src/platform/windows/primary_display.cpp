/**
 * @file src/platform/windows/primary_display.cpp
 * @brief Temporary primary-display transactions with crash recovery.
 */
#include "primary_display.h"

#include "display_config.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace platf::primary_display {
  namespace {
    constexpr size_t max_displays = 128;
    constexpr size_t max_journal_bytes = 256 * 1024;

    bool same_device(std::wstring_view left, std::wstring_view right) {
      return left.size() == right.size() && !left.empty() &&
             CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
    }

    bool same_adapter(const LUID &left, const LUID &right) {
      return left.LowPart == right.LowPart && left.HighPart == right.HighPart;
    }

    const detail::position_t *find_position(const detail::layout_t &layout, std::wstring_view identity) {
      const auto found = std::ranges::find_if(layout, [&](const auto &entry) {
        return same_device(entry.device_path, identity);
      });
      return found == layout.end() ? nullptr : &*found;
    }

    bool valid_layout(const detail::layout_t &layout) {
      if (layout.empty() || layout.size() > max_displays) {
        return false;
      }
      size_t origins = 0;
      for (size_t i = 0; i < layout.size(); ++i) {
        const auto &entry = layout[i];
        if (entry.device_path.empty() || entry.device_path.size() > 1024 || entry.device_path.find(L'\0') != std::wstring::npos) {
          return false;
        }
        origins += entry.x == 0 && entry.y == 0;
        for (size_t j = 0; j < i; ++j) {
          if (same_device(entry.device_path, layout[j].device_path)) {
            return false;
          }
        }
      }
      return origins == 1;
    }

    bool same_layout(const detail::layout_t &left, const detail::layout_t &right) {
      return left.size() == right.size() && std::ranges::all_of(left, [&](const auto &entry) {
               const auto *other = find_position(right, entry.device_path);
               return other && entry.x == other->x && entry.y == other->y;
             });
    }

    std::optional<LONG> subtract(LONG value, LONG offset) {
      const auto result = static_cast<int64_t>(value) - offset;
      if (result < std::numeric_limits<LONG>::min() || result > std::numeric_limits<LONG>::max()) {
        return std::nullopt;
      }
      return static_cast<LONG>(result);
    }

    std::optional<detail::layout_t> translated(const detail::layout_t &layout, LONG x, LONG y) {
      auto result = layout;
      for (auto &entry : result) {
        const auto new_x = subtract(entry.x, x);
        const auto new_y = subtract(entry.y, y);
        if (!new_x || !new_y) {
          return std::nullopt;
        }
        entry.x = *new_x;
        entry.y = *new_y;
      }
      return result;
    }

    UINT32 source_index(const DISPLAYCONFIG_PATH_INFO &path) {
      return path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.sourceInfo.sourceModeInfoIdx : path.sourceInfo.modeInfoIdx;
    }

    bool valid_journal(const detail::journal_t &journal) {
      if (!valid_layout(journal.original)) {
        return false;
      }
      const auto *original = find_position(journal.original, journal.original_primary);
      if (!original || original->x != 0 || original->y != 0) {
        return false;
      }
      if (journal.prepared) {
        return journal.promoted.empty() && journal.promoted_primary.empty();
      }
      if (!valid_layout(journal.promoted) || same_device(journal.original_primary, journal.promoted_primary)) {
        return false;
      }
      const auto *target = find_position(journal.original, journal.promoted_primary);
      if (!target) {
        return false;
      }
      const auto expected = translated(journal.original, target->x, target->y);
      return expected && same_layout(*expected, journal.promoted);
    }

    bool same_modes(const detail::snapshot_t &left, const detail::snapshot_t &right) {
      if (left.paths.size() != right.paths.size()) {
        return false;
      }
      for (size_t i = 0; i < left.paths.size(); ++i) {
        const auto match = std::ranges::find_if(right.device_paths, [&](const auto &identity) {
          return same_device(identity, left.device_paths[i]);
        });
        if (match == right.device_paths.end()) {
          return false;
        }
        const auto j = static_cast<size_t>(match - right.device_paths.begin());
        const auto &a = left.paths[i];
        const auto &b = right.paths[j];
        const auto &am = left.modes[source_index(a)].sourceMode;
        const auto &bm = right.modes[source_index(b)].sourceMode;
        const auto &at = a.targetInfo;
        const auto &bt = b.targetInfo;
        if (am.width != bm.width || am.height != bm.height || am.pixelFormat != bm.pixelFormat || at.rotation != bt.rotation || at.scaling != bt.scaling || at.outputTechnology != bt.outputTechnology || at.scanLineOrdering != bt.scanLineOrdering || static_cast<uint64_t>(at.refreshRate.Numerator) * bt.refreshRate.Denominator != static_cast<uint64_t>(bt.refreshRate.Numerator) * at.refreshRate.Denominator) {
          return false;
        }
      }
      return true;
    }

    std::optional<detail::layout_t> restrict_layout(const detail::layout_t &recorded, const detail::layout_t &current, std::wstring_view virtual_identity) {
      // A crashed host can lose its virtual display before recovery. No other missing or
      // newly attached monitor is silently adopted into an old transaction.
      const bool virtual_missing = !find_position(current, virtual_identity);
      if (recorded.size() != current.size() + (virtual_missing ? 1 : 0)) {
        return std::nullopt;
      }
      detail::layout_t result;
      for (const auto &entry : recorded) {
        if (virtual_missing && same_device(entry.device_path, virtual_identity)) {
          continue;
        }
        if (!find_position(current, entry.device_path)) {
          return std::nullopt;
        }
        result.push_back(entry);
      }
      return result;
    }

    std::filesystem::path journal_path() {
      auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
      path += L".apollo-primary-display.json";
      return path;
    }

    std::optional<detail::snapshot_t> query_snapshot() {
      detail::snapshot_t result;
      const auto status = display_config::query_display_config(QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_MODE_AWARE, result.paths, result.modes);
      if (!status || result.paths.size() > max_displays) {
        BOOST_LOG(warning) << "Could not query displays for temporary primary-display change: " << status.status;
        return std::nullopt;
      }
      for (const auto &path : result.paths) {
        DISPLAYCONFIG_TARGET_DEVICE_NAME name {};
        name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        name.header.size = sizeof(name);
        name.header.adapterId = path.targetInfo.adapterId;
        name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&name.header) != ERROR_SUCCESS || !name.monitorDevicePath[0]) {
          return std::nullopt;
        }
        const auto end = std::find(std::begin(name.monitorDevicePath), std::end(name.monitorDevicePath), L'\0');
        if (end == std::end(name.monitorDevicePath)) {
          return std::nullopt;
        }
        result.device_paths.emplace_back(name.monitorDevicePath, end);
      }
      return result;
    }

    detail::load_result_t load_journal() {
      const auto path = journal_path();
      std::error_code error;
      if (!std::filesystem::exists(path, error)) {
        return {!error, std::nullopt};
      }
      const auto size = std::filesystem::file_size(path, error);
      if (error || size == 0 || size > max_journal_bytes) {
        return {};
      }
      std::ifstream input(path, std::ios::binary);
      std::string contents(static_cast<size_t>(size), '\0');
      if (!input.read(contents.data(), static_cast<std::streamsize>(contents.size())) || input.peek() != std::char_traits<char>::eof()) {
        return {};
      }
      auto journal = detail::deserialize(contents);
      return {journal.has_value(), std::move(journal)};
    }

    bool save_journal(const detail::journal_t &journal) {
      const auto contents = detail::serialize(journal);
      if (contents.empty() || contents.size() > max_journal_bytes) {
        return false;
      }
      const auto path = journal_path();
      auto temporary = path;
      temporary += L".tmp";
      const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
      if (file == INVALID_HANDLE_VALUE) {
        return false;
      }
      DWORD written = 0;
      const bool durable = WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) &&
                           written == contents.size() && FlushFileBuffers(file);
      const bool closed = CloseHandle(file);
      if (!durable || !closed || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(temporary.c_str());
        return false;
      }
      return true;
    }

    bool clear_journal() {
      if (DeleteFileW(journal_path().c_str())) {
        return true;
      }
      return GetLastError() == ERROR_FILE_NOT_FOUND;
    }

    bool apply_snapshot(detail::snapshot_t snapshot) {
      // Never save the temporary primary/layout to the Windows topology database and never
      // permit CCD to choose replacement modes. Only source positions have been changed.
      constexpr UINT32 flags = SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_MODE_AWARE;
      const auto validate = SetDisplayConfig(static_cast<UINT32>(snapshot.paths.size()), snapshot.paths.data(), static_cast<UINT32>(snapshot.modes.size()), snapshot.modes.data(), flags | SDC_VALIDATE);
      if (validate != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Temporary primary-display configuration is not valid: " << validate;
        return false;
      }
      const auto applied = SetDisplayConfig(static_cast<UINT32>(snapshot.paths.size()), snapshot.paths.data(), static_cast<UINT32>(snapshot.modes.size()), snapshot.modes.data(), flags | SDC_APPLY);
      if (applied != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Temporary primary-display configuration failed: " << applied;
      }
      return applied == ERROR_SUCCESS;
    }

    std::mutex transaction_mutex;
    detail::ownership_t process_ownership;

    bool acquire_ownership() {
      auto path = journal_path();
      path.replace_extension(L".lock");
      if (!process_ownership.acquire(path)) {
        BOOST_LOG(warning) << "Cannot acquire primary-display recovery ownership; another host may be using this configuration.";
        return false;
      }
      return true;
    }

    detail::manager_t manager() {
      return detail::manager_t({query_snapshot, apply_snapshot, load_journal, save_journal, clear_journal});
    }
  }  // namespace

  namespace detail {
    ownership_t::~ownership_t() {
      if (handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(handle_);
      }
    }

    bool ownership_t::acquire(const std::filesystem::path &path) {
      std::error_code error;
      const auto absolute = std::filesystem::absolute(path, error);
      if (error) {
        return false;
      }
      const auto parent = std::filesystem::weakly_canonical(absolute.parent_path(), error);
      if (error) {
        return false;
      }
      const auto canonical = parent / absolute.filename();
      if (handle_ != INVALID_HANDLE_VALUE) {
        return same_device(path_.wstring(), canonical.wstring());
      }
      // File sharing is enforced across processes and is not tied to the calling thread.
      // Windows closes this handle after a crash; delete-on-close leaves no stale lock file.
      path_ = canonical;
      const auto handle = CreateFileW(canonical.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
      if (handle == INVALID_HANDLE_VALUE) {
        return false;
      }
      handle_ = handle;
      return true;
    }

    std::optional<layout_t> inspect(const snapshot_t &snapshot) {
      if (snapshot.paths.empty() || snapshot.paths.size() > max_displays || snapshot.modes.size() > max_displays * 3 || snapshot.paths.size() != snapshot.device_paths.size()) {
        return std::nullopt;
      }
      layout_t result;
      std::set<UINT32> source_indices;
      for (size_t i = 0; i < snapshot.paths.size(); ++i) {
        const auto &path = snapshot.paths[i];
        const auto index = source_index(path);
        const UINT32 target_index = path.flags & DISPLAYCONFIG_PATH_SUPPORT_VIRTUAL_MODE ? path.targetInfo.targetModeInfoIdx : path.targetInfo.modeInfoIdx;
        if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE) || !path.targetInfo.targetAvailable || index >= snapshot.modes.size() || target_index >= snapshot.modes.size() || !source_indices.insert(index).second) {
          return std::nullopt;
        }
        const auto &target_mode = snapshot.modes[target_index];
        if (target_mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET || target_mode.id != path.targetInfo.id || !same_adapter(target_mode.adapterId, path.targetInfo.adapterId)) {
          return std::nullopt;
        }
        const auto &mode = snapshot.modes[index];
        if (mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE || mode.id != path.sourceInfo.id || !same_adapter(mode.adapterId, path.sourceInfo.adapterId) || mode.sourceMode.width == 0 || mode.sourceMode.height == 0 || static_cast<int64_t>(mode.sourceMode.position.x) + mode.sourceMode.width > std::numeric_limits<LONG>::max() || static_cast<int64_t>(mode.sourceMode.position.y) + mode.sourceMode.height > std::numeric_limits<LONG>::max()) {
          return std::nullopt;
        }
        for (size_t j = 0; j < i; ++j) {
          const auto &other = snapshot.paths[j];
          if ((same_adapter(path.sourceInfo.adapterId, other.sourceInfo.adapterId) && path.sourceInfo.id == other.sourceInfo.id) || (same_adapter(path.targetInfo.adapterId, other.targetInfo.adapterId) && path.targetInfo.id == other.targetInfo.id)) {
            return std::nullopt;
          }
        }
        result.push_back({snapshot.device_paths[i], mode.sourceMode.position.x, mode.sourceMode.position.y});
      }
      return valid_layout(result) ? std::make_optional(std::move(result)) : std::nullopt;
    }

    std::string serialize(const journal_t &journal) {
      if (!valid_journal(journal)) {
        return {};
      }
      auto encode = [](const layout_t &layout) {
        auto result = nlohmann::json::array();
        for (const auto &entry : layout) {
          result.push_back({{"device_path", platf::to_utf8(entry.device_path)}, {"x", entry.x}, {"y", entry.y}});
        }
        return result;
      };
      return nlohmann::json({{"version", 1}, {"prepared", journal.prepared}, {"original_primary", platf::to_utf8(journal.original_primary)}, {"promoted_primary", platf::to_utf8(journal.promoted_primary)}, {"original", encode(journal.original)}, {"promoted", encode(journal.promoted)}}).dump();
    }

    std::optional<journal_t> deserialize(std::string_view contents) {
      if (contents.size() > max_journal_bytes) {
        return std::nullopt;
      }
      try {
        const auto value = nlohmann::json::parse(contents);
        if (value.at("version") != 1) {
          return std::nullopt;
        }
        auto decode = [](const nlohmann::json &entries) {
          layout_t result;
          if (!entries.is_array() || entries.size() > max_displays) {
            throw std::runtime_error("invalid primary-display layout");
          }
          for (const auto &entry : entries) {
            auto coordinate = [&](const char *name) {
              const auto &number = entry.at(name);
              if (!number.is_number_integer()) {
                throw std::runtime_error("invalid display coordinate");
              }
              if (number.is_number_unsigned() && number.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<LONG>::max())) {
                throw std::runtime_error("display coordinate out of range");
              }
              const auto parsed = number.get<int64_t>();
              if (parsed < std::numeric_limits<LONG>::min() || parsed > std::numeric_limits<LONG>::max()) {
                throw std::runtime_error("display coordinate out of range");
              }
              return static_cast<LONG>(parsed);
            };
            result.push_back({platf::from_utf8(entry.at("device_path").get<std::string>()), coordinate("x"), coordinate("y")});
          }
          return result;
        };
        journal_t result {platf::from_utf8(value.at("original_primary").get<std::string>()), platf::from_utf8(value.at("promoted_primary").get<std::string>()), decode(value.at("original")), decode(value.at("promoted")), value.at("prepared").get<bool>()};
        return valid_journal(result) ? std::make_optional(std::move(result)) : std::nullopt;
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    manager_t::manager_t(io_t io):
        io_(std::move(io)) {}

    bool manager_t::prepare() {
      const auto loaded = io_.load();
      if (!loaded.success || loaded.journal) {
        return false;
      }
      const auto snapshot = io_.query();
      if (!snapshot) {
        return false;
      }
      if (snapshot->paths.empty() && snapshot->device_paths.empty()) {
        // A headless machine has no original primary display to restore.
        return true;
      }
      const auto layout = inspect(*snapshot);
      if (!layout) {
        return false;
      }
      const auto primary = std::ranges::find_if(*layout, [](const auto &entry) {
        return entry.x == 0 && entry.y == 0;
      });
      return io_.save(journal_t {primary->device_path, {}, *layout, {}, true});
    }

    bool manager_t::bind_pending(std::wstring_view device_path) {
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (!loaded.journal) {
        return true;
      }
      const auto &prepared = *loaded.journal;
      if (!prepared.prepared) {
        return same_device(prepared.promoted_primary, device_path);
      }
      const auto snapshot = io_.query();
      const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!current) {
        return false;
      }
      const auto *target = find_position(*current, device_path);
      const auto *original_primary = find_position(*current, prepared.original_primary);
      const bool new_target = !find_position(prepared.original, device_path);
      if (!target || !original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0)) {
        return false;
      }
      const auto original = translated(*current, original_primary->x, original_primary->y);
      if (!original || !std::ranges::all_of(prepared.original, [&](const auto &saved) {
            const auto *observed = find_position(*original, saved.device_path);
            return observed && observed->x == saved.x && observed->y == saved.y;
          })) {
        return false;
      }
      if (same_device(prepared.original_primary, device_path)) {
        return io_.clear();
      }
      const auto promoted = translated(*current, target->x, target->y);
      return promoted && io_.save(journal_t {prepared.original_primary, target->device_path, *original, *promoted});
    }

    bool manager_t::apply_verified(const snapshot_t &before, const layout_t &desired) {
      const auto baseline = inspect(before);
      auto latest = io_.query();
      const auto current = latest ? inspect(*latest) : std::nullopt;
      if (!baseline || !current || !same_layout(*baseline, *current) || !same_modes(before, *latest)) {
        BOOST_LOG(warning) << "Display configuration changed during primary-display transaction; retaining recovery record.";
        return false;
      }
      for (size_t i = 0; i < latest->paths.size(); ++i) {
        const auto *position = find_position(desired, latest->device_paths[i]);
        if (!position) {
          return false;
        }
        latest->modes[source_index(latest->paths[i])].sourceMode.position = {position->x, position->y};
      }
      if (!inspect(*latest) || !io_.apply(*latest)) {
        return false;
      }
      const auto observed = io_.query();
      const auto layout = observed ? inspect(*observed) : std::nullopt;
      if (!layout || !same_layout(*layout, desired) || !same_modes(*latest, *observed)) {
        BOOST_LOG(warning) << "Could not verify primary-display change; retaining recovery record.";
        return false;
      }
      return true;
    }

    bool manager_t::promote(std::wstring_view device_path) {
      auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
      if (loaded.journal && loaded.journal->prepared) {
        if (!bind_pending(device_path)) {
          return false;
        }
        loaded = io_.load();
        if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
          return false;
        }
      }
      auto snapshot = io_.query();
      auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!layout) {
        return false;
      }
      if (loaded.journal) {
        // Keep the first baseline on repeated promotion (including a resized display).
        if (!same_device(loaded.journal->promoted_primary, device_path)) {
          return false;
        }
        if (same_layout(*layout, loaded.journal->promoted)) {
          return true;
        }
        if (!same_layout(*layout, loaded.journal->original)) {
          return false;
        }
        // A prepared baseline has already been bound. Keep that durable first baseline
        // instead of clearing it and taking a later snapshot after mode changes.
        return apply_verified(*snapshot, loaded.journal->promoted);
      }
      const auto *target = find_position(*layout, device_path);
      if (!target) {
        return false;
      }
      if (target->x == 0 && target->y == 0) {
        return true;
      }
      const auto original_primary = std::ranges::find_if(*layout, [](const auto &entry) {
        return entry.x == 0 && entry.y == 0;
      });
      auto desired = translated(*layout, target->x, target->y);
      if (!desired) {
        return false;
      }
      const journal_t journal {original_primary->device_path, target->device_path, *layout, *desired};
      if (!io_.save(journal)) {
        BOOST_LOG(warning) << "Could not durably save primary-display recovery record; display configuration was not changed.";
        return false;
      }
      if (!apply_verified(*snapshot, *desired)) {
        return false;
      }
      BOOST_LOG(info) << "Virtual display is temporarily the Windows primary display.";
      return true;
    }

    bool manager_t::restore(std::wstring_view expected_device_path) {
      const auto loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        BOOST_LOG(warning) << "Could not read primary-display recovery record; leaving displays unchanged.";
        return false;
      }
      if (!loaded.journal) {
        return true;
      }
      const auto &journal = *loaded.journal;
      if (journal.prepared) {
        const auto snapshot = io_.query();
        const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
        if (!expected_device_path.empty() && current && find_position(*current, expected_device_path)) {
          return bind_pending(expected_device_path) && restore(expected_device_path);
        }
        const auto *primary = current ? find_position(*current, journal.original_primary) : nullptr;
        if (!primary || current->size() != journal.original.size()) {
          return false;
        }
        const auto rebased = translated(*current, primary->x, primary->y);
        if (!rebased || !same_layout(*rebased, journal.original)) {
          return false;
        }
        // If a crash occurred before binding the target, wait for the driver's orphan
        // cleanup. Only the exact known physical set may be restored without that identity.
        if (!same_layout(*current, journal.original) && !apply_verified(*snapshot, journal.original)) {
          return false;
        }
        return io_.clear();
      }
      if (!expected_device_path.empty() && !same_device(expected_device_path, journal.promoted_primary)) {
        return false;
      }
      const auto snapshot = io_.query();
      const auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!layout) {
        return false;
      }
      const auto original = restrict_layout(journal.original, *layout, journal.promoted_primary);
      const auto promoted = restrict_layout(journal.promoted, *layout, journal.promoted_primary);
      if (!original || !promoted) {
        BOOST_LOG(warning) << "Monitors changed since primary-display promotion; retaining recovery record until the original displays are available.";
        return false;
      }
      if (same_layout(*layout, *original)) {
        return io_.clear();
      }
      if (!find_position(*layout, journal.promoted_primary)) {
        const auto *primary = find_position(*layout, journal.original_primary);
        const auto rebased = primary ? translated(*layout, primary->x, primary->y) : std::nullopt;
        if (rebased && same_layout(*rebased, *original)) {
          return apply_verified(*snapshot, *original) && io_.clear();
        }
      }
      if (!same_layout(*layout, *promoted)) {
        BOOST_LOG(warning) << "Monitor positions were changed outside the primary-display transaction; leaving that arrangement unchanged and retaining recovery record.";
        return false;
      }
      if (!apply_verified(*snapshot, *original)) {
        return false;
      }
      if (!io_.clear()) {
        return false;
      }
      BOOST_LOG(info) << "Restored the original Windows primary display and monitor positions.";
      return true;
    }
  }  // namespace detail

  bool promote(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().promote(device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display promotion failed: " << exception.what();
      return false;
    }
  }

  bool prepare() {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().prepare();
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not prepare primary-display recovery: " << exception.what();
      return false;
    }
  }

  bool bind_pending(std::wstring_view device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().bind_pending(device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Could not bind primary-display recovery identity: " << exception.what();
      return false;
    }
  }

  bool restore(std::wstring_view expected_device_path) {
    std::lock_guard lock(transaction_mutex);
    try {
      return acquire_ownership() && manager().restore(expected_device_path);
    } catch (const std::exception &exception) {
      BOOST_LOG(error) << "Temporary primary-display recovery failed: " << exception.what();
      return false;
    }
  }

  bool recover() {
    return restore();
  }
}  // namespace platf::primary_display

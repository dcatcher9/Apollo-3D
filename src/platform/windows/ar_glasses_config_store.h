/**
 * @file src/platform/windows/ar_glasses_config_store.h
 * @brief The configuration file as Local AR changes it: a strict read, a one-line replacement
 * that keeps every other line, and an atomic write.
 */
#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace ar_glasses {
  namespace internal {
    inline constexpr std::string_view managed_config_key = "ar_glass_devices";

    /** The single writer lock for the configuration file. The AR device list, which is saved
     * in that file, is guarded by the same lock.
     */
    extern std::mutex device_mutex;

    std::optional<std::string> read_config_file_strict();
    std::string replace_config_value(std::string contents, std::string_view key, std::string_view value);
    std::string replace_managed_config_value(std::string contents, std::string_view value);
    bool write_config_atomically(std::string_view contents);
  }  // namespace internal
}  // namespace ar_glasses

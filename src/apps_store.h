/**
 * @file src/apps_store.h
 * @brief Declarations for reading, validating and migrating the app library in apps.json.
 */
#pragma once

// standard includes
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

// lib includes
#include <nlohmann/json.hpp>

#define VIRTUAL_DISPLAY_UUID "8902CB19-674A-403D-A587-41B092E900BA"
#define FALLBACK_DESKTOP_UUID "EAAC6159-089A-46A9-9E24-6436885F6610"

namespace proc {
  class proc_t;

  /**
   * @brief Find the ')' that closes the '(' at `begin` in a `$(VAR)` reference.
   * @throws std::out_of_range when the bracket is not closed before `end`.
   */
  std::string_view::iterator find_match(std::string_view::iterator begin, std::string_view::iterator end);

  /**
   * @brief Calculate a stable id based on name and image data
   * @return Tuple of id calculated without index (for use if no collision) and one with.
   */
  std::tuple<std::string, std::string> calculate_app_id(const std::string &app_name, std::string app_image_path, int index);

  std::string validate_app_image_path(std::string app_image_path);
  void migrate_apps(nlohmann::json *fileTree_p, nlohmann::json *inputTree_p);
  std::optional<proc::proc_t> parse(const std::string &file_name);
}  // namespace proc

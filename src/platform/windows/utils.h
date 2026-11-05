#pragma once

#include <optional>
#include <string>

std::string utf8ToAcp(const std::string& utf8Str);
std::string currentCodePageToCharset();
// Query the active console session only: true means locked, false unlocked, and nullopt means
// Windows could not establish its state. A Default input-desktop name does not prove unlocked.
std::optional<bool> query_console_session_locked();
bool is_changing_settings_going_to_fail();

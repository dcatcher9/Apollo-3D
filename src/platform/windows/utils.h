#pragma once

#include <string>

std::string utf8ToAcp(const std::string& utf8Str);
std::string currentCodePageToCharset();
bool is_changing_settings_going_to_fail();

#include "ar_glasses_config_store.h"

#include "ar_glasses.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>
#include <windows.h>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace internal {
    std::mutex device_mutex;

    std::optional<std::string> read_config_file_strict() {
      const std::filesystem::path path = platf::from_utf8(config::sunshine.config_file);
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        return std::nullopt;
      }
      std::string contents {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
      };
      if (input.bad()) {
        return std::nullopt;
      }
      return contents;
    }

    std::string replace_config_value(std::string contents, std::string_view key, std::string_view value) {
      const std::string replacement = std::string(key) + " = " + std::string(value);
      size_t cursor = 0;
      while (cursor < contents.size()) {
        const auto newline = contents.find('\n', cursor);
        const auto line_end = newline == std::string::npos ? contents.size() : newline;
        std::string_view line(contents.data() + cursor, line_end - cursor);
        const bool had_carriage_return = !line.empty() && line.back() == '\r';
        if (!line.empty() && line.back() == '\r') {
          line.remove_suffix(1);
        }
        const auto first = line.find_first_not_of(" \t");
        const auto equals = line.find('=', first == std::string_view::npos ? 0 : first);
        const auto key_end = equals == std::string_view::npos ? equals : line.find_last_not_of(" \t", equals - 1);
        const bool managed_line = first != std::string_view::npos && equals != std::string_view::npos &&
                                  key_end != std::string_view::npos &&
                                  line.substr(first, key_end - first + 1) == key;
        if (managed_line) {
          contents.replace(cursor, line_end - cursor, replacement + (had_carriage_return ? "\r" : ""));
          return contents;
        }
        if (newline == std::string::npos) {
          break;
        }
        cursor = newline + 1;
      }

      const std::string_view newline = contents.find("\r\n") != std::string::npos ? "\r\n" : "\n";
      if (!contents.empty() && contents.back() != '\n') {
        contents.append(newline);
      }
      contents.append(replacement).append(newline);
      return contents;
    }

    std::string replace_managed_config_value(std::string contents, std::string_view value) {
      return replace_config_value(std::move(contents), managed_config_key, value);
    }

    bool write_config_atomically(std::string_view contents) {
      const std::filesystem::path path = platf::from_utf8(config::sunshine.config_file);
      auto temporary_path = path;
      temporary_path += L".apollo-ar.tmp";
      {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
          return false;
        }
        output.write(contents.data(), (std::streamsize) contents.size());
        output.flush();
        if (!output) {
          output.close();
          std::error_code ignored;
          std::filesystem::remove(temporary_path, ignored);
          return false;
        }
      }

      if (!MoveFileExW(
            temporary_path.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
          )) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
      }
      return true;
    }
  }  // namespace internal

  bool persist_config_value(std::string_view key, std::string_view value) {
    // device_mutex is the single writer lock for the configuration file.
    std::lock_guard lock(device_mutex);
    auto contents = read_config_file_strict();
    if (!contents) {
      BOOST_LOG(error) << "Could not read the complete configuration file; refusing to replace it while saving ["sv
                       << key << "]."sv;
      return false;
    }
    if (!write_config_atomically(replace_config_value(std::move(*contents), key, value))) {
      BOOST_LOG(error) << "Could not save ["sv << key << "] to "sv << config::sunshine.config_file << '.';
      return false;
    }
    return true;
  }
}  // namespace ar_glasses

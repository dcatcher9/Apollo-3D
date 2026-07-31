/**
 * @file src/sbs_bench_harness.cpp
 * @brief Headless frame-fed SBS benchmark harness (see sbs_bench_harness.h).
 *
 * Duplicates the minimal SBS composite from platform/windows/display_vram.cpp convert()
 * (which lives in an anonymous-namespace class and can't be called directly) but drives it
 * with the REAL video_depth_estimator on a fixed directory of frames. Output PNGs are scored
 * by tools/sbsbench/sbsbench.py. Windows-only (the estimator + shaders are D3D11/TensorRT).
 */
#include "sbs_bench_harness.h"

#ifdef _WIN32

  // standard includes
  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cctype>
  #include <chrono>
  #include <cmath>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <iomanip>
  #include <limits>
  #include <locale>
  #include <map>
  #include <memory>
  #include <new>
  #include <optional>
  #include <set>
  #include <sstream>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <vector>

  // platform includes
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <wincodec.h>
  #include <wrl/client.h>
  #include <nlohmann/json.hpp>
  #include <openssl/evp.h>

  // local includes
  #include "config.h"
  #include "generated/sbs_adaptive_state_contract.h"
  #include "generated/sbs_scene_controller_contract.h"
  #include "../src_assets/windows/assets/shaders/directx/include/sbs_scene_rules_summary_layout.shared.h"
  #include "logging.h"
  #include "sbs_perf.h"
  #include "sbs_scene_cache_contract.h"
  #include "video.h"
  #include "video_depth_estimator.h"

  #ifndef SUNSHINE_SHADERS_DIR
    #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
  #endif

using Microsoft::WRL::ComPtr;
using namespace std::literals;

namespace sbs_bench {

  namespace fs = std::filesystem;

  namespace {

    struct rgba_image {
      UINT w = 0, h = 0;
      std::vector<uint8_t> bgra;  // tightly packed B,G,R,A rows, top-to-bottom
    };

    struct scrgb_image {
      UINT w = 0, h = 0;
      // R16G16B16A16_FLOAT rows, top-to-bottom. PFM stores RGB float32 rows bottom-to-top;
      // loading into the live texture format once keeps the actual SBS path unchanged.
      std::vector<uint16_t> rgba16;
    };

    std::string json_string(std::string_view value) {
      static constexpr char hex[] = "0123456789abcdef";
      std::string escaped;
      escaped.reserve(value.size() + 2);
      escaped.push_back('"');
      for (unsigned char c : value) {
        switch (c) {
          case '"':
            escaped += "\\\"";
            break;
          case '\\':
            escaped += "\\\\";
            break;
          case '\b':
            escaped += "\\b";
            break;
          case '\f':
            escaped += "\\f";
            break;
          case '\n':
            escaped += "\\n";
            break;
          case '\r':
            escaped += "\\r";
            break;
          case '\t':
            escaped += "\\t";
            break;
          default:
            if (c < 0x20) {
              escaped += "\\u00";
              escaped.push_back(hex[c >> 4]);
              escaped.push_back(hex[c & 0x0f]);
            } else {
              escaped.push_back((char) c);
            }
            break;
        }
      }
      escaped.push_back('"');
      return escaped;
    }

    std::string json_finite_double(const double value) {
      if (!std::isfinite(value)) {
        return "null";
      }
      std::ostringstream text;
      text.imbue(std::locale::classic());
      text
        << std::showpoint
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value;
      return text.str();
    }

    uint16_t float_to_half(float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      const uint32_t sign = (bits >> 16) & 0x8000u;
      int exp = (int) ((bits >> 23) & 0xffu) - 127 + 15;
      uint32_t mant = bits & 0x7fffffu;
      if (exp <= 0) {
        if (exp < -10) {
          return (uint16_t) sign;
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t) (sign | ((mant + 0x1000u) >> 13));
      }
      if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
      }
      mant += 0x1000u;
      if (mant & 0x800000u) {
        mant = 0;
        if (++exp >= 31) {
          return (uint16_t) (sign | 0x7c00u);
        }
      }
      return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
    }

    float half_to_float(uint16_t h) {
      const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
      int exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ffu;
      uint32_t bits;
      if (exp == 0) {
        if (mant == 0) {
          bits = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x400u)) {
            mant <<= 1;
            --exp;
          }
          bits = sign | ((uint32_t) exp << 23) | ((mant & 0x3ffu) << 13);
        }
      } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
      } else {
        bits = sign | ((uint32_t) (exp - 15 + 127) << 23) | (mant << 13);
      }
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }

    // Strict HDR interchange for offline whole-clip conversion. FFmpeg's image2 PFM codec writes
    // this exact standard form reliably: RGB float32, little-endian (negative scale), with rows
    // stored bottom-to-top. Values are linear scRGB where 1.0 is the 80-nit reference white.
    //
    // Deliberately reject comments, grayscale Pf, alternate endianness/scales, trailing bytes,
    // non-finite values, and values outside the finite FP16 domain used by the live host path.
    bool load_pfm(const fs::path &path, scrgb_image &out) {
      static_assert(std::endian::native == std::endian::little,
                    "PFM interchange requires a little-endian host");
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return false;
      }

      auto read_header_line = [&](std::string &line) {
        if (!std::getline(stream, line) || line.size() > 128u) {
          return false;
        }
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        return true;
      };

      std::string magic;
      std::string dimensions;
      std::string scale_line;
      if (!read_header_line(magic) || magic != "PF" ||
          !read_header_line(dimensions) ||
          !read_header_line(scale_line)) {
        return false;
      }

      std::uint64_t width64 = 0;
      std::uint64_t height64 = 0;
      {
        std::istringstream values(dimensions);
        values.imbue(std::locale::classic());
        std::string extra;
        if (!(values >> width64 >> height64) || (values >> extra) ||
            width64 == 0u || height64 == 0u ||
            width64 > std::numeric_limits<UINT>::max() ||
            height64 > std::numeric_limits<UINT>::max() ||
            width64 > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
            height64 > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
          return false;
        }
      }
      {
        std::istringstream values(scale_line);
        values.imbue(std::locale::classic());
        double scale = 0.0;
        std::string extra;
        if (!(values >> scale) || (values >> extra) || scale != -1.0) {
          return false;
        }
      }

      const size_t width = static_cast<size_t>(width64);
      const size_t height = static_cast<size_t>(height64);
      if (width > std::numeric_limits<size_t>::max() / height ||
          width * height > std::numeric_limits<size_t>::max() / 4u ||
          width > std::numeric_limits<std::streamsize>::max() /
                    (3u * sizeof(float))) {
        return false;
      }

      scrgb_image decoded;
      decoded.w = static_cast<UINT>(width64);
      decoded.h = static_cast<UINT>(height64);
      decoded.rgba16.resize(width * height * 4u);
      std::vector<float> row(width * 3u);
      const auto row_bytes =
        static_cast<std::streamsize>(row.size() * sizeof(float));
      for (size_t stored_y = 0; stored_y < height; ++stored_y) {
        if (!stream.read(reinterpret_cast<char *>(row.data()), row_bytes)) {
          return false;
        }
        const size_t top_down_y = height - 1u - stored_y;
        uint16_t *dst = decoded.rgba16.data() + top_down_y * width * 4u;
        for (size_t x = 0; x < width; ++x) {
          for (size_t channel = 0; channel < 3u; ++channel) {
            const float value = row[x * 3u + channel];
            if (!std::isfinite(value) || std::abs(value) > 65504.0f) {
              return false;
            }
            dst[x * 4u + channel] = float_to_half(value);
          }
          dst[x * 4u + 3u] = float_to_half(1.0f);
        }
      }

      char trailing = 0;
      if (stream.read(&trailing, 1) || !stream.eof()) {
        return false;
      }
      out = std::move(decoded);
      return true;
    }

    bool save_pfm(const fs::path &path, UINT width, UINT height,
                  const D3D11_MAPPED_SUBRESOURCE &mapped) {
      static_assert(std::endian::native == std::endian::little,
                    "PFM interchange requires a little-endian host");
      if (!mapped.pData ||
          mapped.RowPitch < static_cast<size_t>(width) * 4u * sizeof(uint16_t)) {
        return false;
      }

      std::ofstream stream(path, std::ios::binary | std::ios::trunc);
      if (!stream) {
        return false;
      }
      stream.imbue(std::locale::classic());
      stream << "PF\n" << width << ' ' << height << "\n-1.000000\n";
      if (!stream.good()) {
        stream.close();
        fs::remove(path);
        return false;
      }

      std::vector<float> row(static_cast<size_t>(width) * 3u);
      for (UINT stored_y = 0; stored_y < height; ++stored_y) {
        const UINT top_down_y = height - 1u - stored_y;
        const auto *src = reinterpret_cast<const uint16_t *>(
          static_cast<const uint8_t *>(mapped.pData) +
          static_cast<size_t>(top_down_y) * mapped.RowPitch
        );
        for (UINT x = 0; x < width; ++x) {
          for (size_t channel = 0; channel < 3u; ++channel) {
            const float value = half_to_float(src[static_cast<size_t>(x) * 4u + channel]);
            if (!std::isfinite(value)) {
              stream.close();
              fs::remove(path);
              return false;
            }
            row[static_cast<size_t>(x) * 3u + channel] = value;
          }
        }
        stream.write(
          reinterpret_cast<const char *>(row.data()),
          static_cast<std::streamsize>(row.size() * sizeof(float))
        );
        if (!stream.good()) {
          stream.close();
          fs::remove(path);
          return false;
        }
      }
      stream.close();
      if (!stream.good()) {
        fs::remove(path);
        return false;
      }
      return true;
    }

    template<class Writer>
    bool publish_file_atomically(const fs::path &final_path, Writer &&writer) {
      fs::path temporary_path = final_path;
      temporary_path += ".part";
      std::error_code ec;
      fs::remove(temporary_path, ec);
      if (!writer(temporary_path)) {
        fs::remove(temporary_path, ec);
        return false;
      }
      // The temporary name is a sibling of the final name, guaranteeing the same volume.
      // MOVEFILE_REPLACE_EXISTING keeps progress replacement and reruns atomic on Windows.
      // A polling reader or filesystem filter can transiently deny replacement after the writer
      // has already closed the durable .part file. Retry only Windows sharing/locking failures;
      // permanent path, ACL, and volume errors still fail immediately.
      constexpr int atomic_replace_attempts = 50;
      for (int attempt = 0; attempt < atomic_replace_attempts; ++attempt) {
        if (MoveFileExW(
              temporary_path.wstring().c_str(),
              final_path.wstring().c_str(),
              MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
            )) {
          return true;
        }
        const auto move_error = GetLastError();
        if (
          move_error != ERROR_SHARING_VIOLATION &&
          move_error != ERROR_LOCK_VIOLATION &&
          move_error != ERROR_ACCESS_DENIED
        ) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      fs::remove(temporary_path, ec);
      return false;
    }

    bool write_bytes_durably(const fs::path &path, const void *data, std::size_t size) {
      HANDLE handle = CreateFileW(
        path.wstring().c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        return false;
      }
      const auto *bytes = static_cast<const std::uint8_t *>(data);
      std::size_t offset = 0;
      bool succeeded = true;
      while (offset < size) {
        const DWORD chunk = static_cast<DWORD>(
          std::min<std::size_t>(size - offset, std::numeric_limits<DWORD>::max())
        );
        DWORD written = 0;
        if (!WriteFile(handle, bytes + offset, chunk, &written, nullptr) ||
            written != chunk) {
          succeeded = false;
          break;
        }
        offset += written;
      }
      if (succeeded && !FlushFileBuffers(handle)) {
        succeeded = false;
      }
      if (!CloseHandle(handle)) {
        succeeded = false;
      }
      if (!succeeded) {
        std::error_code ec;
        fs::remove(path, ec);
      }
      return succeeded;
    }

    bool publish_json_atomically(const fs::path &path,
                                 const nlohmann::ordered_json &value) {
      const std::string serialized = value.dump(2) + "\n";
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        return write_bytes_durably(
          temporary_path,
          serialized.data(),
          serialized.size()
        );
      });
    }

    constexpr std::uint64_t follow_max_sequence = 9999999999ull;

    std::string follow_frame_id(std::size_t sequence) {
      char id[16];
      snprintf(id, sizeof(id), "%010zu", sequence);
      return id;
    }

    std::string follow_frame_filename(std::size_t sequence, std::string_view format) {
      return "frame_" + follow_frame_id(sequence) + "." + std::string(format);
    }

    bool parse_follow_frame_filename(std::string_view filename,
                                     std::string_view format,
                                     std::size_t &sequence) {
      const std::string suffix = "." + std::string(format);
      constexpr std::string_view prefix = "frame_";
      constexpr std::size_t digits = 10;
      if (filename.size() != prefix.size() + digits + suffix.size() ||
          !filename.starts_with(prefix) ||
          !filename.ends_with(suffix)) {
        return false;
      }
      std::size_t value = 0;
      for (std::size_t index = prefix.size();
           index < prefix.size() + digits;
           ++index) {
        const unsigned char c = static_cast<unsigned char>(filename[index]);
        if (!std::isdigit(c)) {
          return false;
        }
        value = value * 10u + static_cast<std::size_t>(c - '0');
      }
      if (value == 0u) {
        return false;
      }
      sequence = value;
      return true;
    }

    struct producer_done_t {
      std::size_t frame_count = 0;
    };

    bool read_producer_done(const fs::path &path, producer_done_t &result,
                            std::string &error) {
      try {
        std::ifstream stream(path);
        nlohmann::json value;
        if (!stream || !(stream >> value) || !value.is_object() ||
            value.value("schema", 0) != 1 ||
            value.value("status", std::string()) != "complete" ||
            !value.contains("frame_count") ||
            !value["frame_count"].is_number_integer()) {
          error = "malformed .producer-done.json";
          return false;
        }
        const auto count = value["frame_count"].get<std::int64_t>();
        if (count <= 0 ||
            static_cast<std::uint64_t>(count) > follow_max_sequence ||
            static_cast<std::uint64_t>(count) >
              static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
          error = ".producer-done.json frame_count must fit the positive 10-digit sequence";
          return false;
        }
        result.frame_count = static_cast<std::size_t>(count);
        return true;
      } catch (const std::exception &) {
        error = "malformed .producer-done.json";
        return false;
      }
    }

    bool read_producer_failure(const fs::path &path, std::string &error) {
      try {
        std::ifstream stream(path);
        nlohmann::json value;
        if (!stream || !(stream >> value) || !value.is_object() ||
            value.value("schema", 0) != 1 ||
            value.value("status", std::string()) != "failed" ||
            !value.contains("error") || !value["error"].is_string()) {
          error = "malformed .producer-failed.json";
          return false;
        }
        error = value["error"].get<std::string>();
        if (error.empty() || error.size() > 1024u) {
          error = ".producer-failed.json error must contain 1..1024 characters";
          return false;
        }
        return true;
      } catch (const std::exception &) {
        error = "malformed .producer-failed.json";
        return false;
      }
    }

    enum class follow_wait_status_e {
      ready,
      complete,
      error,
    };

    struct follow_wait_result_t {
      follow_wait_status_e status = follow_wait_status_e::error;
      fs::path frame;
      std::optional<std::size_t> producer_frame_count;
      std::string error;
    };

    follow_wait_result_t wait_for_follow_frame(const fs::path &directory,
                                               std::string_view format,
                                               std::size_t sequence,
                                               std::size_t first_sequence,
                                               std::size_t processed_count,
                                               std::size_t count_bound) {
      const fs::path expected =
        directory / follow_frame_filename(sequence, format);
      const fs::path done_path = directory / ".producer-done.json";
      const fs::path failed_path = directory / ".producer-failed.json";
      const std::string selected_extension = "." + std::string(format);

      for (;;) {
        std::error_code ec;
        const bool done_exists = fs::exists(done_path, ec);
        if (ec ||
            (done_exists && !fs::is_regular_file(done_path, ec)) ||
            ec) {
          return {follow_wait_status_e::error, {}, {},
                  "cannot inspect producer done sentinel"};
        }
        ec.clear();
        const bool failed_exists = fs::exists(failed_path, ec);
        if (ec ||
            (failed_exists && !fs::is_regular_file(failed_path, ec)) ||
            ec) {
          return {follow_wait_status_e::error, {}, {},
                  "cannot inspect producer failed sentinel"};
        }
        if (done_exists && failed_exists) {
          return {follow_wait_status_e::error, {}, {},
                  "conflicting producer done and failed sentinels"};
        }
        if (failed_exists) {
          std::string producer_error;
          if (!read_producer_failure(failed_path, producer_error)) {
            return {follow_wait_status_e::error, {}, {}, producer_error};
          }
          return {follow_wait_status_e::error, {}, {},
                  "producer failed: " + producer_error};
        }

        std::optional<std::size_t> done_count;
        if (done_exists) {
          producer_done_t done;
          std::string parse_error;
          if (!read_producer_done(done_path, done, parse_error)) {
            return {follow_wait_status_e::error, {}, {}, parse_error};
          }
          if (count_bound > 0u && done.frame_count > count_bound) {
            return {follow_wait_status_e::error, {}, {},
                    "producer frame_count exceeds --follow-count bound"};
          }
          if (done.frame_count < processed_count) {
            return {follow_wait_status_e::error, {}, {},
                    "producer frame_count is below already processed count"};
          }
          done_count = done.frame_count;
        }

        bool expected_exists = false;
        bool future_exists = false;
        std::error_code iter_ec;
        fs::directory_iterator iterator(directory, iter_ec);
        const fs::directory_iterator end;
        for (; !iter_ec && iterator != end; iterator.increment(iter_ec)) {
          const auto &entry = *iterator;
          std::error_code entry_ec;
          if (!entry.is_regular_file(entry_ec)) {
            if (entry_ec) {
              // The wrapper may delete an already acknowledged file while this scan is in
              // progress. It never deletes the current/future identity, so a vanished entry is
              // safe to ignore and the exact expected-name test below still protects ordering.
              continue;
            }
            continue;
          }
          const std::string filename = entry.path().filename().string();
          std::string extension = entry.path().extension().string();
          for (char &c : extension) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
          }
          const bool image_extension =
            extension == ".png" || extension == ".jpg" ||
            extension == ".jpeg" || extension == ".pfm" ||
            extension == ".bmp" || extension == ".tif" ||
            extension == ".tiff" || extension == ".exr" ||
            extension == ".webp";
          if (!image_extension) {
            const bool frame_like =
              filename.size() > 16u && filename.starts_with("frame_") &&
              std::all_of(
                filename.begin() + 6,
                filename.begin() + 16,
                [](unsigned char c) {
                  return std::isdigit(c);
                }
              );
            if (frame_like && extension != ".part" && extension != ".tmp") {
              return {follow_wait_status_e::error, {}, done_count,
                      "unexpected frame artifact in follow directory: " + filename};
            }
            continue;  // Producer temporary names and JSON sentinels are not ready frames.
          }
          if (extension != selected_extension) {
            return {follow_wait_status_e::error, {}, done_count,
                    "mixed image format in follow directory: " + filename};
          }
          std::size_t found_sequence = 0;
          if (!parse_follow_frame_filename(filename, format, found_sequence)) {
            return {follow_wait_status_e::error, {}, done_count,
                    "unexpected final image identity in follow directory: " + filename};
          }
          if (found_sequence < first_sequence ||
              (count_bound > 0u &&
               found_sequence - first_sequence >= count_bound)) {
            return {follow_wait_status_e::error, {}, done_count,
                    "frame identity is outside the requested follow range: " + filename};
          }
          if (done_count &&
              found_sequence - first_sequence >= *done_count) {
            return {follow_wait_status_e::error, {}, done_count,
                    "frame identity exceeds producer frame_count: " + filename};
          }
          expected_exists = expected_exists || found_sequence == sequence;
          future_exists = future_exists || found_sequence > sequence;
        }
        if (iter_ec) {
          return {follow_wait_status_e::error, {}, done_count,
                  "cannot enumerate follow directory"};
        }

        if (expected_exists) {
          return {follow_wait_status_e::ready, expected, done_count, {}};
        }
        if (future_exists) {
          return {follow_wait_status_e::error, {}, done_count,
                  "follow queue skipped expected " + expected.filename().string()};
        }
        if (done_count) {
          if (*done_count == processed_count) {
            return {follow_wait_status_e::complete, {}, done_count, {}};
          }
          return {follow_wait_status_e::error, {}, done_count,
                  "producer completed before publishing expected " +
                    expected.filename().string()};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    bool publish_follow_progress(const fs::path &path,
                                 std::string_view status,
                                 std::string_view input_format,
                                 std::string_view artifact_mode,
                                 std::size_t first_sequence,
                                 std::size_t processed_count,
                                 int sbs_frame_count,
                                 std::optional<std::size_t> producer_frame_count) {
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        std::ofstream stream(temporary_path);
        if (!stream) {
          return false;
        }
        stream << "{\n"
               << "  \"schema\": 1,\n"
               << "  \"status\": " << json_string(status) << ",\n"
               << "  \"input_format\": " << json_string(input_format) << ",\n"
               << "  \"artifact_mode\": " << json_string(artifact_mode) << ",\n"
               << "  \"processed_count\": " << processed_count << ",\n"
               << "  \"first_sequence\": " << first_sequence << ",\n"
               << "  \"last_completed_sequence\": "
               << (processed_count > 0u ?
                     std::to_string(first_sequence + processed_count - 1u) :
                     "null")
               << ",\n"
               << "  \"last_frame_id\": "
               << (processed_count > 0u ?
                     json_string(follow_frame_id(
                       first_sequence + processed_count - 1u
                     )) :
                     "null")
               << ",\n"
               << "  \"last_completed_frame_id\": "
               << (processed_count > 0u ?
                     json_string(follow_frame_id(
                       first_sequence + processed_count - 1u
                     )) :
                     "null")
               << ",\n"
               << "  \"source_frame_count\": " << processed_count << ",\n"
               << "  \"sbs_frame_count\": " << sbs_frame_count << ",\n"
               << "  \"producer_frame_count\": ";
        if (producer_frame_count) {
          stream << *producer_frame_count;
        } else {
          stream << "null";
        }
        stream << "\n}\n";
        stream.flush();
        return stream.good();
      });
    }

    float srgb_to_linear(float value) {
      return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
    }

    void hdr_preview_bgra(float r, float g, float b, uint8_t *out) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float y = std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      r /= 1.0f + y;
      g /= 1.0f + y;
      b /= 1.0f + y;
      const float peak = std::max(1.0f, std::max(r, std::max(g, b)));
      r /= peak;
      g /= peak;
      b /= peak;
      auto encode = [](float c) {
        c = std::clamp(c, 0.0f, 1.0f);
        c = c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        return (uint8_t) std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f);
      };
      out[0] = encode(b);
      out[1] = encode(g);
      out[2] = encode(r);
      out[3] = 255;
    }

    // ---- WIC PNG load/save (32bpp BGRA, matching the SDR B8G8R8A8_UNORM pipeline) ----

    ComPtr<IWICImagingFactory> g_wic;

    bool wic_init() {
      if (g_wic) {
        return true;
      }
      if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        // Already initialized on this thread with another mode is fine.
      }
      return SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&g_wic)));
    }

    bool load_png(const fs::path &path, rgba_image &out) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) {
        return false;
      }
      ComPtr<IWICBitmapFrameDecode> frame;
      if (FAILED(dec->GetFrame(0, &frame))) {
        return false;
      }
      ComPtr<IWICFormatConverter> conv;
      if (FAILED(g_wic->CreateFormatConverter(&conv))) {
        return false;
      }
      if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) {
        return false;
      }
      if (FAILED(conv->GetSize(&out.w, &out.h))) {
        return false;
      }
      out.bgra.resize((size_t) out.w * out.h * 4);
      return SUCCEEDED(conv->CopyPixels(nullptr, out.w * 4, (UINT) out.bgra.size(), out.bgra.data()));
    }

    bool load_depth_texture(ID3D11Device *dev, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &texture,
                            ComPtr<ID3D11ShaderResourceView> &srv) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(path.wstring().c_str(), nullptr, GENERIC_READ,
                                                   WICDecodeMetadataCacheOnDemand, &dec))) {
        return false;
      }
      ComPtr<IWICBitmapFrameDecode> frame;
      ComPtr<IWICFormatConverter> conv;
      if (FAILED(dec->GetFrame(0, &frame)) || FAILED(g_wic->CreateFormatConverter(&conv)) ||
          FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat16bppGray,
                                  WICBitmapDitherTypeNone, nullptr, 0.0,
                                  WICBitmapPaletteTypeCustom))) {
        return false;
      }
      UINT width = 0, height = 0;
      if (FAILED(conv->GetSize(&width, &height)) || !width || !height) {
        return false;
      }
      std::vector<uint16_t> gray((size_t) width * height);
      if (FAILED(conv->CopyPixels(nullptr, width * sizeof(uint16_t),
                                  (UINT) (gray.size() * sizeof(uint16_t)),
                                  (BYTE *) gray.data()))) {
        return false;
      }
      std::vector<float> depth(gray.size());
      std::transform(gray.begin(), gray.end(), depth.begin(), [](uint16_t value) {
        return value / 65535.0f;
      });
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA data = {depth.data(), (UINT) (width * sizeof(float)), 0};
      return SUCCEEDED(dev->CreateTexture2D(&desc, &data, &texture)) &&
             SUCCEEDED(dev->CreateShaderResourceView(texture.Get(), nullptr, &srv));
    }

    bool save_png(const fs::path &path, UINT w, UINT h, const std::vector<uint8_t> &bgra) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) {
        return false;
      }
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) {
        return false;
      }
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) {
        return false;
      }
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
      }
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) {
        return false;
      }
      if (FAILED(fe->Initialize(props.Get()))) {
        return false;
      }
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 4, (UINT) bgra.size(), const_cast<uint8_t *>(bgra.data())))) {
        return false;
      }
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    bool save_gray16_png(const fs::path &path, UINT w, UINT h, const std::vector<uint16_t> &gray) {
      ComPtr<IWICStream> stream;
      if (FAILED(g_wic->CreateStream(&stream))) {
        return false;
      }
      if (FAILED(stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE))) {
        return false;
      }
      ComPtr<IWICBitmapEncoder> enc;
      if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) {
        return false;
      }
      if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
      }
      ComPtr<IWICBitmapFrameEncode> fe;
      ComPtr<IPropertyBag2> props;
      if (FAILED(enc->CreateNewFrame(&fe, &props))) {
        return false;
      }
      if (FAILED(fe->Initialize(props.Get()))) {
        return false;
      }
      fe->SetSize(w, h);
      WICPixelFormatGUID fmt = GUID_WICPixelFormat16bppGray;
      fe->SetPixelFormat(&fmt);
      if (FAILED(fe->WritePixels(h, w * 2, (UINT) (gray.size() * 2), (BYTE *) const_cast<uint16_t *>(gray.data())))) {
        return false;
      }
      return SUCCEEDED(fe->Commit()) && SUCCEEDED(enc->Commit());
    }

    // Read back an R32_FLOAT depth SRV and save it as a 16-bit grayscale PNG (values clamped to
    // [0,1] scaled to 0-65535). 16-bit matters: the swim metric measures frame-to-frame depth
    // deltas that sit below 1/255. The staging texture is cached across frames (constant size).
    void dump_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, const fs::path &path, ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) {
        return;
      }
      D3D11_TEXTURE2D_DESC d = {};
      tex->GetDesc(&d);
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return;
      }
      std::vector<uint16_t> gray((size_t) d.Width * d.Height);
      for (UINT y = 0; y < d.Height; y++) {
        const float *row = (const float *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
        for (UINT x = 0; x < d.Width; x++) {
          float v = row[x];
          v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
          gray[(size_t) y * d.Width + x] = (uint16_t) (v * 65535.0f + 0.5f);
        }
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_gray16_png(path, d.Width, d.Height, gray);
    }

    void dump_uint_mask(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                        ID3D11ShaderResourceView *srv, const fs::path &path,
                        ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(resource.As(&texture))) {
        return;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (!stage_cache) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&stage_desc, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return;
      }
      std::vector<uint16_t> gray((size_t) desc.Width * desc.Height);
      for (UINT y = 0; y < desc.Height; ++y) {
        const auto *row = (const uint32_t *) ((const uint8_t *) mapped.pData +
                                              (size_t) y * mapped.RowPitch);
        for (UINT x = 0; x < desc.Width; ++x) {
          gray[(size_t) y * desc.Width + x] = row[x] ? 65535u : 0u;
        }
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_gray16_png(path, desc.Width, desc.Height, gray);
    }

    // Read back a harness-only B8G8R8A8 diagnostic target without coupling it to the stream's
    // SDR/HDR format. Disocclusion masks use R=pre-fill hole and G=still unresolved after fill.
    void dump_bgra8_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!texture) {
        return;
      }
      D3D11_TEXTURE2D_DESC d = {};
      texture->GetDesc(&d);
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return;
      }
      std::vector<uint8_t> pixels((size_t) d.Width * d.Height * 4);
      for (UINT y = 0; y < d.Height; ++y) {
        std::memcpy(pixels.data() + (size_t) y * d.Width * 4,
                    (const uint8_t *) mapped.pData + (size_t) y * mapped.RowPitch,
                    (size_t) d.Width * 4);
      }
      ctx->Unmap(stage_cache.Get(), 0);
      save_png(path, d.Width, d.Height, pixels);
    }

    // Preserve the harness-only R32_FLOAT mapping target without quantization. Rows are
    // written tightly packed even though D3D11 staging resources may have a padded RowPitch.
    // Windows/D3D11 targets are little-endian, matching the sidecar's declared float32-le dtype.
    bool dump_float_texture(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11Texture2D *texture, const fs::path &path,
                            ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return false;
      }
      if (!stage_cache) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::ofstream out(path, std::ios::binary);
      const std::streamsize row_bytes = (std::streamsize) desc.Width * sizeof(float);
      if (out) {
        for (UINT y = 0; y < desc.Height; ++y) {
          out.write((const char *) mapped.pData + (size_t) y * mapped.RowPitch, row_bytes);
        }
      }
      out.flush();
      const bool succeeded = out.good();
      ctx->Unmap(stage_cache.Get(), 0);
      return succeeded;
    }

    // Preserve the exact raw model output for stage-by-stage parity checks. Unlike the display
    // PNG, this is not clamped or normalized: it is row-major float32, width*height values.
    void dump_raw_model_depth(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, int width, int height, const fs::path &path, ComPtr<ID3D11Buffer> &stage_cache) {
      if (!srv || width <= 0 || height <= 0) {
        return;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Buffer> buf;
      if (FAILED(res.As(&buf))) {
        return;
      }
      D3D11_BUFFER_DESC d = {};
      buf->GetDesc(&d);
      if (!stage_cache) {
        D3D11_BUFFER_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateBuffer(&sd, nullptr, &stage_cache))) {
          return;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buf.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return;
      }
      std::ofstream out(path, std::ios::binary);
      if (out) {
        out.write((const char *) m.pData, (std::streamsize) width * height * sizeof(float));
      }
      ctx->Unmap(stage_cache.Get(), 0);
    }

    struct subject_state_record {
      std::string frame_id;
      std::array<float, 12> values {};
    };

    // Benchmark-only state trace. This readback is deliberately confined to the synchronous
    // offline harness; the live capture loop must remain free of staging copies and Map calls.
    bool read_subject_state(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                            ID3D11ShaderResourceView *srv,
                            ComPtr<ID3D11Buffer> &stage_cache,
                            std::array<float, 12> &values) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      if (desc.ByteWidth < values.size() * sizeof(float) ||
          desc.StructureByteStride != 4 * sizeof(float)) {
        return false;
      }
      bool recreate = !stage_cache;
      if (!recreate) {
        D3D11_BUFFER_DESC stage_desc {};
        stage_cache->GetDesc(&stage_desc);
        recreate = stage_desc.ByteWidth != desc.ByteWidth;
      }
      if (recreate) {
        D3D11_BUFFER_DESC stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        stage_cache.Reset();
        if (FAILED(dev->CreateBuffer(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::memcpy(values.data(), mapped.pData, values.size() * sizeof(float));
      ctx->Unmap(stage_cache.Get(), 0);
      return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
      });
    }

    using adaptive_state_words_t = sbs_adaptive_state::words_t;
    using scene_controller_global_words_t =
      std::array<
        std::uint32_t,
        sbs_scene_controller::global_out_word_count
      >;
    using scene_controller_rule_words_t =
      std::array<
        std::uint32_t,
        sbs_scene_controller::rule_state_word_count
      >;
    using scene_controller_rule_summary_words_t =
      std::array<std::uint32_t, SBS_RULE_SUMMARY_FLOAT_COUNT>;
    using render_state_words_t =
      std::array<std::uint32_t, sbs_adaptive_state::render_prefix_word_count>;
    using cached_state_words_t = sbs_scene_cache::cached_state_words_t;
    using frame_metadata_t = sbs_scene_cache::frame_metadata_t;
    using roi_transform_words_t = sbs_scene_cache::roi_transform_words_t;

    constexpr float scene_anchor_shift_min = -1.39635933f;
    constexpr float scene_anchor_shift_max = 8.58230571f;
    constexpr std::uint32_t scene_controller_trace_schema = 1u;
    constexpr std::string_view scene_controller_trace_source =
      "video_depth_estimator.scene_controller_snapshot";
    constexpr std::string_view scene_controller_trace_capture =
      "every-source-frame-after-estimator-update";
    constexpr std::string_view scene_controller_trace_name =
      "scene_controller.jsonl";
    constexpr std::uint32_t scene_controller_source_time_schema = 2u;
    constexpr std::string_view scene_controller_source_time_clock =
      "source-presentation-timeline-v2";
    constexpr std::size_t scene_controller_source_time_max_line_bytes =
      256u;

    struct scene_controller_source_time_t {
      std::ifstream stream;
      std::string file_sha256;
      std::size_t frame_count = 0;
      std::size_t consumed_frame_count = 0;
      std::int64_t time_base_num = 0;
      std::int64_t time_base_den = 0;
      std::int64_t previous_pts_ticks = 0;
      bool has_previous_pts = false;
      double total_elapsed_seconds = 0.0;
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest {
        nullptr,
        &EVP_MD_CTX_free,
      };
    };

    bool exact_json_object_keys(
      const nlohmann::json &value,
      const std::initializer_list<const char *> keys
    ) {
      return
        value.is_object() &&
        value.size() == keys.size() &&
        std::all_of(keys.begin(), keys.end(), [&](const char *key) {
          return value.contains(key);
        });
    }

    bool read_bounded_jsonl_line(
      std::istream &stream,
      std::string &line,
      std::string &error,
      EVP_MD_CTX *digest
    ) {
      line.clear();
      for (;;) {
        const int next = stream.get();
        if (next == std::char_traits<char>::eof()) {
          if (stream.bad()) {
            error = "cannot read source-time JSONL record";
          } else if (!line.empty()) {
            error = "source-time JSONL record is not LF-terminated";
          } else {
            error = "source-time JSONL ended before the next record";
          }
          return false;
        }
        if (next == '\n') {
          static constexpr char newline = '\n';
          if (
            digest &&
            (
              EVP_DigestUpdate(
                digest,
                line.data(),
                line.size()
              ) != 1 ||
              EVP_DigestUpdate(digest, &newline, 1u) != 1
            )
          ) {
            error = "cannot hash source-time JSONL record";
            return false;
          }
          if (!line.empty() && line.back() == '\r') {
            line.pop_back();
          }
          if (line.empty()) {
            error = "source-time JSONL contains an empty record";
            return false;
          }
          return true;
        }
        if (
          line.size() + 2u >
          scene_controller_source_time_max_line_bytes
        ) {
          error = "source-time JSONL record exceeds its bounded size";
          return false;
        }
        line.push_back(static_cast<char>(next));
      }
    }

    bool parse_json_line_strict(
      const std::string &line,
      nlohmann::json &value,
      std::string &error
    ) {
      std::map<int, std::set<std::string>> object_keys;
      bool invalid_callback_depth = false;
      std::optional<std::string> duplicate_key;
      const auto callback =
        [&](const int depth,
            const nlohmann::json::parse_event_t event,
            nlohmann::json &parsed) {
          switch (event) {
            case nlohmann::json::parse_event_t::object_start:
              object_keys[depth].clear();
              break;
            case nlohmann::json::parse_event_t::key: {
              if (
                depth <= 0 || !parsed.is_string() ||
                !object_keys.contains(depth - 1)
              ) {
                invalid_callback_depth = true;
                break;
              }
              const auto &key = parsed.get_ref<const std::string &>();
              if (
                !object_keys[depth - 1].insert(key).second &&
                !duplicate_key
              ) {
                duplicate_key = key;
              }
              break;
            }
            case nlohmann::json::parse_event_t::object_end:
              object_keys.erase(depth);
              break;
            default:
              break;
          }
          return true;
        };
      try {
        value = nlohmann::json::parse(line, callback);
        if (invalid_callback_depth) {
          error = "source-time JSONL has invalid object nesting";
          return false;
        }
        if (duplicate_key) {
          error =
            "source-time JSONL contains duplicate key '" +
            *duplicate_key + "'";
          return false;
        }
        return true;
      } catch (const std::exception &exception) {
        error =
          std::string {"malformed source-time JSONL record: "} +
          exception.what();
        return false;
      }
    }

    bool exact_signed_integer(
      const nlohmann::json &value,
      std::int64_t &result
    ) {
      try {
        if (!value.is_number_integer()) {
          return false;
        }
        if (value.is_number_unsigned()) {
          const auto unsigned_value = value.get<std::uint64_t>();
          if (
            unsigned_value >
            static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max()
            )
          ) {
            return false;
          }
          result = static_cast<std::int64_t>(unsigned_value);
          return true;
        }
        result = value.get<std::int64_t>();
        return true;
      } catch (...) {
        return false;
      }
    }

    bool read_scene_controller_source_time(
      const fs::path &path,
      const std::size_t expected_frame_count,
      scene_controller_source_time_t &result,
      std::string &error
    ) {
      try {
        scene_controller_source_time_t parsed;
        parsed.digest.reset(EVP_MD_CTX_new());
        if (
          !parsed.digest ||
          EVP_DigestInit_ex(
            parsed.digest.get(),
            EVP_sha256(),
            nullptr
          ) != 1
        ) {
          error = "cannot initialize source-time JSONL SHA-256";
          return false;
        }
        parsed.stream.open(path, std::ios::binary);
        if (!parsed.stream) {
          error = "cannot open source-time JSONL";
          return false;
        }
        std::string line;
        nlohmann::json value;
        if (
          !read_bounded_jsonl_line(
            parsed.stream,
            line,
            error,
            parsed.digest.get()
          ) ||
          !parse_json_line_strict(line, value, error)
        ) {
          return false;
        }
        if (
          !exact_json_object_keys(value, {
            "record",
            "schema",
            "clock",
            "frame_count",
            "time_base",
          }) ||
          !value.at("record").is_string() ||
          value.at("record").get<std::string>() != "header" ||
          !value.at("schema").is_number_unsigned() ||
          value.at("schema").get<std::uint32_t>() !=
            scene_controller_source_time_schema ||
          !value.at("clock").is_string() ||
          value.at("clock").get<std::string>() !=
            scene_controller_source_time_clock ||
          !value.at("frame_count").is_number_unsigned() ||
          !value.at("time_base").is_object()
        ) {
          error =
            "header is not an exact source-presentation-timeline-v2 contract";
          return false;
        }
        const auto frame_count_u64 =
          value.at("frame_count").get<std::uint64_t>();
        if (
          frame_count_u64 == 0u ||
          frame_count_u64 > follow_max_sequence ||
          frame_count_u64 >
            static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max()
            ) ||
          frame_count_u64 != expected_frame_count
        ) {
          error =
            "header frame_count must match the positive bounded source count";
          return false;
        }
        const auto &time_base = value.at("time_base");
        if (
          !exact_json_object_keys(time_base, {"num", "den"}) ||
          !exact_signed_integer(
            time_base.at("num"),
            parsed.time_base_num
          ) ||
          !exact_signed_integer(
            time_base.at("den"),
            parsed.time_base_den
          ) ||
          parsed.time_base_num <= 0 ||
          parsed.time_base_den <= 0
        ) {
          error = "time_base must contain exact positive integer num/den";
          return false;
        }
        parsed.frame_count = static_cast<std::size_t>(frame_count_u64);
        result = std::move(parsed);
        return true;
      } catch (const std::exception &exception) {
        error = std::string {"malformed source-time contract: "} +
                exception.what();
        return false;
      }
    }

    bool read_scene_controller_source_time_frame(
      scene_controller_source_time_t &source_time,
      const std::string_view expected_frame_id,
      double &elapsed_seconds,
      std::string &error
    ) {
      try {
        if (
          source_time.consumed_frame_count >= source_time.frame_count
        ) {
          error = "source-time JSONL has more source frames than declared";
          return false;
        }
        std::string line;
        nlohmann::json value;
        if (
          !read_bounded_jsonl_line(
            source_time.stream,
            line,
            error,
            source_time.digest.get()
          ) ||
          !parse_json_line_strict(line, value, error)
        ) {
          return false;
        }
        const auto index = source_time.consumed_frame_count;
        std::int64_t pts_ticks = 0;
        std::int64_t duration_ticks = 0;
        if (
          !exact_json_object_keys(value, {
            "record",
            "source_index",
            "frame_id",
            "pts_ticks",
            "duration_ticks",
          }) ||
          !value.at("record").is_string() ||
          value.at("record").get<std::string>() != "frame" ||
          !value.at("source_index").is_number_unsigned() ||
          value.at("source_index").get<std::uint64_t>() != index ||
          !value.at("frame_id").is_string() ||
          value.at("frame_id").get<std::string>() != expected_frame_id ||
          !exact_signed_integer(value.at("pts_ticks"), pts_ticks) ||
          !exact_signed_integer(
            value.at("duration_ticks"),
            duration_ticks
          ) ||
          duration_ticks <= 0
        ) {
          error =
            "source-time frame is not an exact contiguous timeline record";
          return false;
        }

        std::int64_t delta_ticks = 0;
        if (source_time.has_previous_pts) {
          if (pts_ticks <= source_time.previous_pts_ticks) {
            error = "source-time PTS must increase strictly";
            return false;
          }
          if (
            source_time.previous_pts_ticks < 0 &&
            pts_ticks >
              std::numeric_limits<std::int64_t>::max() +
                source_time.previous_pts_ticks
          ) {
            error = "source-time presentation delta exceeds int64";
            return false;
          }
          delta_ticks = pts_ticks - source_time.previous_pts_ticks;
        }
        elapsed_seconds =
          static_cast<double>(delta_ticks) *
          static_cast<double>(source_time.time_base_num) /
          static_cast<double>(source_time.time_base_den);
        if (
          !std::isfinite(elapsed_seconds) ||
          elapsed_seconds < 0.0
        ) {
          error = "source-time presentation delta is not finite";
          return false;
        }
        source_time.total_elapsed_seconds += elapsed_seconds;
        if (!std::isfinite(source_time.total_elapsed_seconds)) {
          error = "source-time presentation duration overflowed";
          return false;
        }
        source_time.previous_pts_ticks = pts_ticks;
        source_time.has_previous_pts = true;
        ++source_time.consumed_frame_count;
        return true;
      } catch (const std::exception &exception) {
        error = std::string {"malformed source-time frame: "} +
                exception.what();
        return false;
      }
    }

    bool finish_scene_controller_source_time(
      scene_controller_source_time_t &source_time,
      std::string &error
    ) {
      if (
        source_time.consumed_frame_count != source_time.frame_count
      ) {
        error = "source-time JSONL ended before every source frame";
        return false;
      }
      const int trailing = source_time.stream.get();
      if (trailing != std::char_traits<char>::eof()) {
        error = "source-time JSONL contains trailing records or bytes";
        return false;
      }
      if (source_time.stream.bad()) {
        error = "cannot read the source-time JSONL tail";
        return false;
      }
      std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
      unsigned int digest_size = 0;
      if (
        !source_time.digest ||
        EVP_DigestFinal_ex(
          source_time.digest.get(),
          digest.data(),
          &digest_size
        ) != 1 ||
        digest_size != 32u
      ) {
        error = "cannot finalize the consumed source-time JSONL SHA-256";
        return false;
      }
      static constexpr char digits[] = "0123456789abcdef";
      source_time.file_sha256.assign(digest_size * 2u, '\0');
      for (std::size_t index = 0; index < digest_size; ++index) {
        source_time.file_sha256[index * 2u] =
          digits[digest[index] >> 4u];
        source_time.file_sha256[index * 2u + 1u] =
          digits[digest[index] & 0x0fu];
      }
      return true;
    }

    bool valid_adaptive_state_words(const adaptive_state_words_t &words) {
      for (const auto &field : sbs_adaptive_state::fields) {
        if (field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits) {
          continue;
        }
        if (!std::isfinite(std::bit_cast<float>(
              words[sbs_adaptive_state::index(field.word)]
            ))) {
          return false;
        }
      }
      return true;
    }

    std::string scene_cache_frame_stem(std::size_t sequence) {
      return "frame_" + follow_frame_id(sequence);
    }

    struct scene_cache_metadata {
      UINT source_width = 0;
      UINT source_height = 0;
      std::string input_frame_format;
      std::string input_texture_format;
      std::string input_color_space;
      std::string model_name;
      std::string model_url;
      double pop_strength = 1.0;
      double adaptive_pop_max = 1.0;
      bool subject_stretch = true;
      bool literal_bestv2 = false;
      bool simulate_hdr = false;
      double hdr_scale = 1.0;
      int depth_reuse_interval = 1;
      int eye_width = 0;
      int eye_height = 0;
      double output_scale = 1.0;
      int max_output_width = 0;
      UINT output_eye_width = 0;
      UINT output_eye_height = 0;
      UINT output_sbs_width = 0;
      UINT output_sbs_height = 0;
      std::string packed_texture_format;
      std::string output_frame_format;
      std::string output_file_extension;
      std::string status;
      std::size_t processed_count = 0;
      std::size_t frame_count = 0;
    };

    struct resolved_sbs_geometry {
      UINT eye_width = 0;
      UINT eye_height = 0;
      UINT sbs_width = 0;
      UINT sbs_height = 0;
    };

    resolved_sbs_geometry resolve_sbs_geometry(
      UINT source_width,
      UINT source_height,
      int requested_eye_width,
      int requested_eye_height,
      double output_scale,
      int max_output_width
    ) {
      const int eye_height_target =
        requested_eye_height > 0 ?
          requested_eye_height :
          std::max(
            2,
            static_cast<int>(std::lround(source_height * output_scale))
          );
      const float aspect =
        static_cast<float>(source_width) / source_height;
      int eye_width =
        requested_eye_width > 0 ?
          requested_eye_width :
          (requested_eye_height > 0 ?
             std::max(
               1,
               static_cast<int>(std::lround(eye_height_target * aspect))
             ) :
             std::max(
               1,
               static_cast<int>(std::lround(source_width * output_scale))
             ));
      int eye_height = eye_height_target;
      if (requested_eye_width > 0 && requested_eye_height <= 0) {
        eye_height =
          std::max(1, static_cast<int>(std::lround(eye_width / aspect)));
      }
      if (2 * eye_width > max_output_width) {
        const double scale =
          static_cast<double>(max_output_width) / (2 * eye_width);
        eye_width = std::max(1, max_output_width / 2);
        eye_height = std::max(
          2,
          static_cast<int>(std::lround(eye_height_target * scale)) & ~1
        );
      }
      return {
        static_cast<UINT>(eye_width),
        static_cast<UINT>(eye_height),
        static_cast<UINT>(2 * eye_width),
        static_cast<UINT>(eye_height),
      };
    }

    bool publish_scene_cache_contract(const fs::path &directory,
                                      const scene_cache_metadata &metadata,
                                      std::string_view status,
                                      std::size_t processed_count) {
      nlohmann::ordered_json contract = {
        {"schema", sbs_scene_cache::contract_schema},
        {"status", status},
        {"source", {
          {"width", metadata.source_width},
          {"height", metadata.source_height},
          {"frame_format", metadata.input_frame_format},
          {"texture_format", metadata.input_texture_format},
          {"color_space", metadata.input_color_space},
        }},
        {"depth", {
          {"dimensions", "per-frame-metadata"},
          {"dxgi_format", "R32_FLOAT"},
          {"dtype", "float32-le"},
          {"layout", "row-major"},
          {"row_order", "top-down"},
          {"file_pattern", "frame_%010d.depth.r32f"},
          {"bytes_per_sample", sizeof(float)},
          {"bytes_per_frame", nullptr},
        }},
        {"frame_metadata", {
          {"schema", sbs_scene_cache::frame_metadata_schema},
          {"magic", sbs_scene_cache::frame_metadata_magic},
          {"word_count", sbs_scene_cache::frame_metadata_word_count},
          {"dtype", "uint32-le"},
          {"layout", "raw-word-order"},
          {"file_pattern", "frame_%010d.meta.u32"},
          {"bytes_per_frame", sizeof(frame_metadata_t)},
          {"depth_dimensions_words", {4, 5}},
          {"cache_sequence_words", {8, 9}},
          {"retained_source_frame_id_words", {10, 11}},
          {"roi_transform_word_offset",
           sbs_scene_cache::roi_transform_word_offset},
          {"roi_transform_word_count",
           sbs_scene_cache::roi_transform_word_count},
          {"roi_transform_contract_schema",
           models::frame_roi_transform_contract_version},
        }},
        {"state", {
          {"schema", sbs_scene_cache::cached_state_schema},
          {"source",
           "depth_subject_resolve_cs.SubjectState[0..2]+DepthFrameState[0]"},
          {"subject_word_count", render_state_words_t {}.size()},
          {"depth_frame_state_word_count",
           sbs_scene_cache::depth_frame_state_word_count},
          {"word_count", cached_state_words_t {}.size()},
          {"dtype", "uint32-le"},
          {"layout", "raw-word-order"},
          {"file_pattern", "frame_%010d.state.u32"},
          {"bytes_per_frame", sizeof(cached_state_words_t)},
        }},
        {"render_config", {
          {"model", metadata.model_name},
          {"model_url", metadata.model_url},
          {"pop_strength", metadata.pop_strength},
          {"adaptive_pop_max", metadata.adaptive_pop_max},
          {"subject_stretch", metadata.subject_stretch},
          {"literal_bestv2", metadata.literal_bestv2},
          {"simulate_hdr", metadata.simulate_hdr},
          {"hdr_scale", metadata.hdr_scale},
          {"depth_reuse_interval", metadata.depth_reuse_interval},
          {"requested_eye_width", metadata.eye_width},
          {"requested_eye_height", metadata.eye_height},
          {"output_scale", metadata.output_scale},
          {"resolved_max_output_width", metadata.max_output_width},
        }},
        {"packed_sbs", {
          {"eye_width", metadata.output_eye_width},
          {"eye_height", metadata.output_eye_height},
          {"width", metadata.output_sbs_width},
          {"height", metadata.output_sbs_height},
          {"texture_format", metadata.packed_texture_format},
          {"frame_format", metadata.output_frame_format},
          {"file_extension", metadata.output_file_extension},
          {"file_pattern",
           "sbs_%010d." + metadata.output_file_extension},
          {"atomic_replay_publication", true},
        }},
        {"first_sequence", 1},
        {"processed_count", processed_count},
        {"last_completed_sequence",
         processed_count > 0u ?
           nlohmann::ordered_json(processed_count) :
           nlohmann::ordered_json(nullptr)},
        {"frame_count",
         status == "complete" ?
           nlohmann::ordered_json(metadata.frame_count) :
           nlohmann::ordered_json(nullptr)},
        {"atomic_frame_publication", true},
        {"native_source_deletion", false},
        {"cache_budget", {
          {"enforced_by", "wrapper"},
          {"native_limit_bytes", nullptr},
          {"on_write_failure", "fail-closed"},
          {"native_eviction", false},
        }},
      };
      return publish_json_atomically(
        directory / "scene_cache_contract.json",
        contract
      );
    }

    bool read_exact_file(const fs::path &path, void *data, std::size_t size) {
      std::error_code ec;
      if (!fs::is_regular_file(path, ec) || ec ||
          fs::file_size(path, ec) != size || ec ||
          size > static_cast<std::size_t>(
                   std::numeric_limits<std::streamsize>::max())) {
        return false;
      }
      std::ifstream stream(path, std::ios::binary);
      return stream &&
             static_cast<bool>(stream.read(
               static_cast<char *>(data),
               static_cast<std::streamsize>(size)
             ));
    }

    template<std::size_t WordCount>
    bool read_structured_words(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      ComPtr<ID3D11Buffer> &stage_cache,
      std::array<std::uint32_t, WordCount> &words
    ) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      if (
        desc.ByteWidth < words.size() * sizeof(std::uint32_t) ||
        desc.StructureByteStride != 4u * sizeof(std::uint32_t)
      ) {
        return false;
      }
      bool recreate = !stage_cache;
      if (!recreate) {
        D3D11_BUFFER_DESC current {};
        stage_cache->GetDesc(&current);
        recreate = current.ByteWidth != desc.ByteWidth;
      }
      if (recreate) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        stage_cache.Reset();
        if (FAILED(dev->CreateBuffer(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::memcpy(words.data(), mapped.pData, words.size() * sizeof(words[0]));
      ctx->Unmap(stage_cache.Get(), 0);
      return true;
    }

    bool make_scene_cache_frame_metadata(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *transform_srv,
      ComPtr<ID3D11Buffer> &transform_stage,
      const std::uint64_t sequence,
      const UINT source_width,
      const UINT source_height,
      const UINT depth_width,
      const UINT depth_height,
      const std::uint32_t depth_reuse_interval,
      const bool allow_forced_current,
      frame_metadata_t &metadata
    ) {
      roi_transform_words_t transform {};
      if (
        !read_structured_words(
          dev,
          ctx,
          transform_srv,
          transform_stage,
          transform
        )
      ) {
        return false;
      }
      const std::uint64_t pixels =
        static_cast<std::uint64_t>(depth_width) * depth_height;
      if (
        sequence == 0u ||
        pixels == 0u ||
        pixels > std::numeric_limits<std::uint32_t>::max()
      ) {
        return false;
      }

      metadata = {};
      metadata.depth = {
        depth_width,
        depth_height,
        static_cast<std::uint32_t>(pixels),
        sizeof(float),
      };
      sbs_scene_cache::split_u64(
        sequence,
        metadata.sequence[0],
        metadata.sequence[1]
      );
      const auto retained_frame_id =
        sbs_scene_cache::transform_is_unbound_zero(transform) ?
          sbs_scene_cache::unbound_source_frame_id :
          sbs_scene_cache::join_u64(transform[2u], transform[3u]);
      sbs_scene_cache::split_u64(
        retained_frame_id,
        metadata.sequence[2],
        metadata.sequence[3]
      );
      metadata.identity = {
        source_width,
        source_height,
        depth_width,
        depth_height,
      };
      metadata.roi_transform = transform;
      return sbs_scene_cache::valid_frame_metadata(
        metadata,
        sequence,
        source_width,
        source_height,
        depth_reuse_interval,
        allow_forced_current
      );
    }

    bool cache_frame_metadata_atomically(
      const frame_metadata_t &metadata,
      const fs::path &path
    ) {
      static_assert(std::endian::native == std::endian::little);
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        return write_bytes_durably(
          temporary_path,
          &metadata,
          sizeof(metadata)
        );
      });
    }

    bool cache_depth_texture_atomically(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const fs::path &path,
      ComPtr<ID3D11Texture2D> &stage_cache,
      UINT expected_width,
      UINT expected_height
    ) {
      static_assert(std::endian::native == std::endian::little);
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(resource.As(&texture))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (desc.Format != DXGI_FORMAT_R32_FLOAT ||
          desc.Width != expected_width ||
          desc.Height != expected_height) {
        return false;
      }
      bool recreate = !stage_cache;
      if (!recreate) {
        D3D11_TEXTURE2D_DESC current {};
        stage_cache->GetDesc(&current);
        recreate = current.Width != desc.Width ||
                   current.Height != desc.Height ||
                   current.Format != desc.Format;
      }
      if (recreate) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        stage_cache.Reset();
        if (FAILED(dev->CreateTexture2D(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      const std::size_t row_bytes =
        static_cast<std::size_t>(desc.Width) * sizeof(float);
      std::vector<float> values(
        static_cast<std::size_t>(desc.Width) * desc.Height
      );
      bool valid = mapped.RowPitch >= row_bytes;
      for (UINT y = 0; valid && y < desc.Height; ++y) {
        std::memcpy(
          values.data() + static_cast<std::size_t>(y) * desc.Width,
          static_cast<const std::uint8_t *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          row_bytes
        );
      }
      ctx->Unmap(stage_cache.Get(), 0);
      valid = valid &&
              std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
              });
      return valid &&
             publish_file_atomically(path, [&](const fs::path &temporary_path) {
               return write_bytes_durably(
                 temporary_path,
                 values.data(),
                 values.size() * sizeof(float)
               );
             });
    }

    bool cache_render_state_atomically(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      const adaptive_state_words_t &words,
      ID3D11ShaderResourceView *depth_frame_state,
      ComPtr<ID3D11Buffer> &depth_frame_state_stage,
      const fs::path &path
    ) {
      static_assert(std::endian::native == std::endian::little);
      if (!valid_adaptive_state_words(words)) {
        return false;
      }
      cached_state_words_t render_words {};
      std::copy_n(
        words.begin(),
        sbs_adaptive_state::render_prefix_word_count,
        render_words.begin()
      );
      std::array<
        std::uint32_t,
        sbs_scene_cache::depth_frame_state_word_count
      > frame_state {};
      if (
        !read_structured_words(
          dev,
          ctx,
          depth_frame_state,
          depth_frame_state_stage,
          frame_state
        )
      ) {
        return false;
      }
      std::copy(
        frame_state.begin(),
        frame_state.end(),
        render_words.begin() +
          sbs_adaptive_state::render_prefix_word_count
      );
      if (!sbs_scene_cache::valid_cached_state(render_words)) {
        return false;
      }
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        return write_bytes_durably(
          temporary_path,
          render_words.data(),
          sizeof(render_words)
        );
      });
    }

    bool create_cached_depth_srv(ID3D11Device *dev,
                                 const fs::path &path,
                                 UINT width,
                                 UINT height,
                                 ComPtr<ID3D11Texture2D> &texture,
                                 ComPtr<ID3D11ShaderResourceView> &srv) {
      if (
        !width ||
        !height ||
        width > sbs_scene_cache::max_depth_dimension ||
        height > sbs_scene_cache::max_depth_dimension ||
        width % models::frame_roi_model_patch_size != 0u ||
        height % models::frame_roi_model_patch_size != 0u ||
        static_cast<std::uint64_t>(width) * height >
          std::numeric_limits<std::size_t>::max() / sizeof(float)
      ) {
        return false;
      }
      std::vector<float> values;
      try {
        values.resize(static_cast<std::size_t>(width) * height);
      } catch (const std::bad_alloc &) {
        return false;
      } catch (const std::length_error &) {
        return false;
      }
      if (!read_exact_file(
            path,
            values.data(),
            values.size() * sizeof(float)
          ) ||
          !std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
          })) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA initial = {
        values.data(),
        static_cast<UINT>(width * sizeof(float)),
        0
      };
      return SUCCEEDED(dev->CreateTexture2D(&desc, &initial, &texture)) &&
             SUCCEEDED(dev->CreateShaderResourceView(texture.Get(), nullptr, &srv));
    }

    bool create_structured_words_srv(
      ID3D11Device *dev,
      const std::uint32_t *words,
      const std::size_t word_count,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      if (
        !words ||
        word_count == 0u ||
        word_count % 4u != 0u ||
        word_count >
          std::numeric_limits<UINT>::max() / sizeof(std::uint32_t)
      ) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth =
        static_cast<UINT>(word_count * sizeof(std::uint32_t));
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = 4 * sizeof(float);
      D3D11_SUBRESOURCE_DATA initial = {words, 0, 0};
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      view.Format = DXGI_FORMAT_UNKNOWN;
      view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      view.Buffer.FirstElement = 0;
      view.Buffer.NumElements =
        static_cast<UINT>(word_count / 4u);
      return SUCCEEDED(dev->CreateBuffer(&desc, &initial, &buffer)) &&
             SUCCEEDED(dev->CreateShaderResourceView(buffer.Get(), &view, &srv));
    }

    bool create_cached_state_srvs(
      ID3D11Device *dev,
      const cached_state_words_t &words,
      ComPtr<ID3D11Buffer> &subject_buffer,
      ComPtr<ID3D11ShaderResourceView> &subject_srv,
      ComPtr<ID3D11Buffer> &depth_frame_state_buffer,
      ComPtr<ID3D11ShaderResourceView> &depth_frame_state_srv
    ) {
      if (!sbs_scene_cache::valid_cached_state(words)) {
        return false;
      }
      return create_structured_words_srv(
               dev,
               words.data(),
               sbs_adaptive_state::render_prefix_word_count,
               subject_buffer,
               subject_srv
             ) &&
             create_structured_words_srv(
               dev,
               words.data() +
                 sbs_adaptive_state::render_prefix_word_count,
               sbs_scene_cache::depth_frame_state_word_count,
               depth_frame_state_buffer,
               depth_frame_state_srv
             );
    }

    bool create_cached_roi_transform_srv(
      ID3D11Device *dev,
      const frame_metadata_t &metadata,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      return create_structured_words_srv(
        dev,
        metadata.roi_transform.data(),
        metadata.roi_transform.size(),
        buffer,
        srv
      );
    }

    struct scene_plan_entry {
      std::size_t start_sequence = 0;
      std::size_t end_sequence_exclusive = 0;
      float absolute_pop_strength = 1.0f;
      float zero_anchor_shift_px = 0.0f;
    };

    bool read_scene_cache_contract(const fs::path &directory,
                                   scene_cache_metadata &metadata,
                                   std::string &error) {
      try {
        std::ifstream stream(directory / "scene_cache_contract.json");
        nlohmann::json value;
        if (!stream || !(stream >> value) || !value.is_object() ||
            value.at("schema").get<std::uint32_t>() !=
              sbs_scene_cache::contract_schema ||
            (value.at("status").get<std::string>() != "running" &&
             value.at("status").get<std::string>() != "complete") ||
            !value.at("processed_count").is_number_integer()) {
          error = "scene cache contract is not a running/complete schema-2 contract";
          return false;
        }
        const std::string status = value.at("status").get<std::string>();
        const auto processed_count =
          value.at("processed_count").get<std::int64_t>();
        std::int64_t frame_count = 0;
        if (status == "complete") {
          if (!value.at("frame_count").is_number_integer()) {
            error = "complete scene cache contract has no frame_count";
            return false;
          }
          frame_count = value.at("frame_count").get<std::int64_t>();
        } else if (!value.at("frame_count").is_null()) {
          error = "running scene cache contract must not claim a terminal frame_count";
          return false;
        }
        if (processed_count < 0 ||
            static_cast<std::uint64_t>(processed_count) > follow_max_sequence ||
            (status == "complete" &&
             (frame_count <= 0 ||
              frame_count != processed_count ||
              static_cast<std::uint64_t>(frame_count) > follow_max_sequence)) ||
            value.at("first_sequence").get<int>() != 1 ||
            !value.at("atomic_frame_publication").get<bool>()) {
          error = "scene cache contract has an invalid durable sequence";
          return false;
        }
        const auto &source = value.at("source");
        const auto &depth = value.at("depth");
        const auto &frame_metadata = value.at("frame_metadata");
        const auto &state = value.at("state");
        const auto &render = value.at("render_config");
        const auto &packed = value.at("packed_sbs");
        metadata.source_width = source.at("width").get<UINT>();
        metadata.source_height = source.at("height").get<UINT>();
        metadata.input_frame_format = source.at("frame_format").get<std::string>();
        metadata.input_texture_format = source.at("texture_format").get<std::string>();
        metadata.input_color_space = source.at("color_space").get<std::string>();
        metadata.model_name = render.at("model").get<std::string>();
        metadata.model_url = render.at("model_url").get<std::string>();
        metadata.pop_strength = render.at("pop_strength").get<double>();
        metadata.adaptive_pop_max = render.at("adaptive_pop_max").get<double>();
        metadata.subject_stretch = render.at("subject_stretch").get<bool>();
        metadata.literal_bestv2 = render.at("literal_bestv2").get<bool>();
        metadata.simulate_hdr = render.at("simulate_hdr").get<bool>();
        metadata.hdr_scale = render.at("hdr_scale").get<double>();
        metadata.depth_reuse_interval =
          render.at("depth_reuse_interval").get<int>();
        metadata.eye_width = render.at("requested_eye_width").get<int>();
        metadata.eye_height = render.at("requested_eye_height").get<int>();
        metadata.output_scale = render.at("output_scale").get<double>();
        metadata.max_output_width =
          render.at("resolved_max_output_width").get<int>();
        metadata.output_eye_width = packed.at("eye_width").get<UINT>();
        metadata.output_eye_height = packed.at("eye_height").get<UINT>();
        metadata.output_sbs_width = packed.at("width").get<UINT>();
        metadata.output_sbs_height = packed.at("height").get<UINT>();
        metadata.packed_texture_format =
          packed.at("texture_format").get<std::string>();
        metadata.output_frame_format =
          packed.at("frame_format").get<std::string>();
        metadata.output_file_extension =
          packed.at("file_extension").get<std::string>();
        metadata.status = status;
        metadata.processed_count =
          static_cast<std::size_t>(processed_count);
        metadata.frame_count =
          frame_count > 0 ? static_cast<std::size_t>(frame_count) : 0u;

        if (!metadata.source_width || !metadata.source_height ||
            depth.at("dimensions").get<std::string>() !=
              "per-frame-metadata" ||
            depth.at("dxgi_format").get<std::string>() != "R32_FLOAT" ||
            depth.at("dtype").get<std::string>() != "float32-le" ||
            depth.at("layout").get<std::string>() != "row-major" ||
            depth.at("row_order").get<std::string>() != "top-down" ||
            depth.at("file_pattern").get<std::string>() !=
              "frame_%010d.depth.r32f" ||
            depth.at("bytes_per_sample").get<std::size_t>() !=
              sizeof(float) ||
            !depth.at("bytes_per_frame").is_null() ||
            frame_metadata.at("schema").get<std::uint32_t>() !=
              sbs_scene_cache::frame_metadata_schema ||
            frame_metadata.at("magic").get<std::uint32_t>() !=
              sbs_scene_cache::frame_metadata_magic ||
            frame_metadata.at("word_count").get<std::uint32_t>() !=
              sbs_scene_cache::frame_metadata_word_count ||
            frame_metadata.at("dtype").get<std::string>() != "uint32-le" ||
            frame_metadata.at("layout").get<std::string>() !=
              "raw-word-order" ||
            frame_metadata.at("file_pattern").get<std::string>() !=
              "frame_%010d.meta.u32" ||
            frame_metadata.at("bytes_per_frame").get<std::size_t>() !=
              sizeof(frame_metadata_t) ||
            frame_metadata.at("roi_transform_word_offset")
                .get<std::uint32_t>() !=
              sbs_scene_cache::roi_transform_word_offset ||
            frame_metadata.at("roi_transform_word_count")
                .get<std::uint32_t>() !=
              sbs_scene_cache::roi_transform_word_count ||
            frame_metadata.at("roi_transform_contract_schema")
                .get<std::uint32_t>() !=
              models::frame_roi_transform_contract_version ||
            state.at("schema").get<std::uint32_t>() !=
              sbs_scene_cache::cached_state_schema ||
            state.at("subject_word_count").get<std::size_t>() !=
              render_state_words_t {}.size() ||
            state.at("depth_frame_state_word_count").get<std::size_t>() !=
              sbs_scene_cache::depth_frame_state_word_count ||
            state.at("word_count").get<std::size_t>() !=
              cached_state_words_t {}.size() ||
            state.at("dtype").get<std::string>() != "uint32-le" ||
            state.at("layout").get<std::string>() != "raw-word-order" ||
            state.at("file_pattern").get<std::string>() !=
              "frame_%010d.state.u32" ||
            state.at("bytes_per_frame").get<std::size_t>() !=
              sizeof(cached_state_words_t) ||
            !(metadata.pop_strength >= 0.25 &&
              metadata.pop_strength <= 2.0) ||
            !(metadata.adaptive_pop_max >= metadata.pop_strength &&
              metadata.adaptive_pop_max <= 2.0) ||
            metadata.depth_reuse_interval < 1 ||
            metadata.depth_reuse_interval > 8 ||
            !(metadata.output_scale > 0.0 && metadata.output_scale <= 4.0) ||
            metadata.max_output_width <= 0 ||
            !metadata.output_eye_width ||
            !metadata.output_eye_height ||
            metadata.output_sbs_width !=
              2u * metadata.output_eye_width ||
            metadata.output_sbs_height !=
              metadata.output_eye_height ||
            (metadata.output_file_extension != "png" &&
             metadata.output_file_extension != "pfm") ||
            packed.at("file_pattern").get<std::string>() !=
              "sbs_%010d." + metadata.output_file_extension ||
            !packed.at("atomic_replay_publication").get<bool>() ||
            !std::isfinite(metadata.pop_strength) ||
            !std::isfinite(metadata.adaptive_pop_max) ||
            !std::isfinite(metadata.hdr_scale) ||
            !std::isfinite(metadata.output_scale)) {
          error = "scene cache contract contains an unsupported layout or render configuration";
          return false;
        }
        return true;
      } catch (const std::exception &) {
        error = "malformed scene cache contract";
        return false;
      }
    }

    bool validate_scene_cache_range(const fs::path &directory,
                                    const scene_cache_metadata &metadata,
                                    std::size_t start_sequence,
                                    std::size_t end_sequence_exclusive,
                                    std::string &error) {
      if (start_sequence == 0u ||
          end_sequence_exclusive <= start_sequence ||
          end_sequence_exclusive - 1u > metadata.processed_count) {
        error = "requested scene range is not durably available in the cache";
        return false;
      }
      std::error_code ec;
      for (std::size_t sequence = start_sequence;
           sequence < end_sequence_exclusive;
           ++sequence) {
        const std::string stem = scene_cache_frame_stem(sequence);
        const fs::path depth = directory / (stem + ".depth.r32f");
        const fs::path state = directory / (stem + ".state.u32");
        const fs::path frame_metadata_path =
          directory / (stem + ".meta.u32");
        frame_metadata_t frame_metadata {};
        cached_state_words_t render_words {};
        if (!read_exact_file(
              frame_metadata_path,
              &frame_metadata,
              sizeof(frame_metadata)
            ) ||
            !read_exact_file(
              state,
              render_words.data(),
              sizeof(render_words)
            ) ||
            !sbs_scene_cache::valid_scene_replay_frame(
              frame_metadata,
              render_words,
              sequence,
              start_sequence,
              metadata.source_width,
              metadata.source_height,
              static_cast<std::uint32_t>(metadata.depth_reuse_interval),
              sequence == metadata.processed_count
            ) ||
            !fs::is_regular_file(depth, ec) || ec ||
            fs::file_size(depth, ec) !=
              sbs_scene_cache::depth_payload_bytes(frame_metadata) || ec ||
            !fs::is_regular_file(state, ec) || ec ||
            fs::file_size(state, ec) != sizeof(cached_state_words_t) || ec) {
          error = "scene cache is missing or has a wrong-sized frame triplet for global sequence " +
                  std::to_string(sequence);
          return false;
        }
      }
      return true;
    }

    bool read_scene_plan(const fs::path &path,
                         std::vector<scene_plan_entry> &entries,
                         std::string &error) {
      try {
        std::ifstream stream(path);
        nlohmann::json value;
        if (!stream || !(stream >> value) || !value.is_object() ||
            value.at("schema").get<int>() != 1 ||
            value.at("version").get<std::string>() != "scene-plan-v1" ||
            value.at("cache_contract_schema").get<std::uint32_t>() !=
              sbs_scene_cache::contract_schema ||
            !value.at("scenes").is_array() ||
            value.at("scenes").size() != 1u) {
          error = "scene plan is not a scene-plan-v1 schema-1 document";
          return false;
        }
        for (const auto &scene : value.at("scenes")) {
          if (!scene.is_object() ||
              !scene.at("start_sequence").is_number_integer() ||
              !scene.at("end_sequence_exclusive").is_number_integer() ||
              !scene.at("absolute_pop_strength").is_number() ||
              !scene.at("zero_anchor_shift_px").is_number()) {
            error = "scene plan contains a malformed scene";
            return false;
          }
          const auto start = scene.at("start_sequence").get<std::int64_t>();
          const auto end =
            scene.at("end_sequence_exclusive").get<std::int64_t>();
          const double absolute_pop =
            scene.at("absolute_pop_strength").get<double>();
          const double anchor =
            scene.at("zero_anchor_shift_px").get<double>();
          if (start <= 0 || end <= start ||
              static_cast<std::uint64_t>(end - 1) >
                follow_max_sequence ||
              static_cast<std::uint64_t>(start) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
              static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
              !std::isfinite(absolute_pop) ||
              absolute_pop < 0.25 ||
              absolute_pop > 2.0 ||
              !std::isfinite(anchor) ||
              anchor < scene_anchor_shift_min ||
              anchor > scene_anchor_shift_max) {
            error = "scene plan has a gap, overlap, or out-of-range pop/zero value";
            return false;
          }
          entries.push_back({
            static_cast<std::size_t>(start),
            static_cast<std::size_t>(end),
            static_cast<float>(absolute_pop),
            static_cast<float>(anchor),
          });
        }
        return true;
      } catch (const std::exception &) {
        error = "malformed scene plan";
        return false;
      }
    }

    // Whole-clip modes consume the complete append-only state. Slots 16..19 are uint bit
    // patterns, not floats; retaining the raw words prevents counter precision loss and avoids
    // treating a large counter's bit pattern as a NaN.
    bool read_adaptive_state(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                             ID3D11ShaderResourceView *srv,
                             ComPtr<ID3D11Buffer> &stage_cache,
                             adaptive_state_words_t &words) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      if (desc.ByteWidth < words.size() * sizeof(std::uint32_t) ||
          desc.StructureByteStride != 4 * sizeof(float)) {
        return false;
      }
      bool recreate = !stage_cache;
      if (!recreate) {
        D3D11_BUFFER_DESC stage_desc {};
        stage_cache->GetDesc(&stage_desc);
        recreate = stage_desc.ByteWidth != desc.ByteWidth;
      }
      if (recreate) {
        D3D11_BUFFER_DESC stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        stage_cache.Reset();
        if (FAILED(dev->CreateBuffer(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::memcpy(words.data(), mapped.pData, words.size() * sizeof(std::uint32_t));
      ctx->Unmap(stage_cache.Get(), 0);
      return valid_adaptive_state_words(words);
    }

    template<std::size_t WordCount>
    bool read_scene_controller_buffer(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      UINT expected_structure_stride,
      ComPtr<ID3D11Buffer> &stage_cache,
      std::array<std::uint32_t, WordCount> &words
    ) {
      if (!dev || !ctx || !srv || expected_structure_stride == 0u) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      if (
        desc.ByteWidth != words.size() * sizeof(std::uint32_t) ||
        desc.StructureByteStride != expected_structure_stride ||
        (desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) == 0u
      ) {
        return false;
      }
      bool recreate = !stage_cache;
      if (!recreate) {
        D3D11_BUFFER_DESC stage_desc {};
        stage_cache->GetDesc(&stage_desc);
        recreate = stage_desc.ByteWidth != desc.ByteWidth;
      }
      if (recreate) {
        D3D11_BUFFER_DESC stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0u;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0u;
        stage_cache.Reset();
        if (FAILED(dev->CreateBuffer(&stage_desc, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      std::memcpy(
        words.data(),
        mapped.pData,
        words.size() * sizeof(std::uint32_t)
      );
      ctx->Unmap(stage_cache.Get(), 0);
      return true;
    }

    bool valid_scene_controller_global_words(
      const scene_controller_global_words_t &words
    ) {
      for (std::size_t index = 0; index < words.size(); ++index) {
        const float value = std::bit_cast<float>(words[index]);
        if (!std::isfinite(value)) {
          return false;
        }
        const auto &field =
          sbs_scene_controller::global_out_names[index];
        if (
          field.starts_with("reserved_") &&
          value != 0.0f
        ) {
          return false;
        }
      }
      return true;
    }

    bool valid_scene_controller_rule_words(
      const scene_controller_rule_words_t &words
    ) {
      for (const auto &field : sbs_scene_controller::rule_state_fields) {
        const auto index = sbs_scene_controller::index(field.word);
        if (
          field.gpu_encoding ==
          sbs_scene_controller::gpu_encoding_e::uint_bits
        ) {
          if (field.required_zero && words[index] != 0u) {
            return false;
          }
          continue;
        }
        const float value = std::bit_cast<float>(words[index]);
        if (!std::isfinite(value) || (field.required_zero && value != 0.0f)) {
          return false;
        }
        if (
          field.gpu_encoding ==
            sbs_scene_controller::gpu_encoding_e::uint_valued_float &&
          (
            value < 0.0f ||
            static_cast<double>(value) >
              static_cast<double>(
                std::numeric_limits<std::uint32_t>::max()
              ) ||
            std::trunc(value) != value
          )
        ) {
          return false;
        }
      }
      const float schema = std::bit_cast<float>(
        words[sbs_scene_controller::index(
          sbs_scene_controller::rule_state_word_e::schema_version
        )]
      );
      return schema ==
             static_cast<float>(sbs_scene_controller::schema_version);
    }

    std::string_view scene_controller_encoding_name(
      const sbs_scene_controller::gpu_encoding_e encoding
    ) {
      switch (encoding) {
        case sbs_scene_controller::gpu_encoding_e::float_value:
          return "float";
        case sbs_scene_controller::gpu_encoding_e::uint_bits:
          return "uint_bits";
        case sbs_scene_controller::gpu_encoding_e::uint_valued_float:
          return "uint_valued_float";
      }
      return {};
    }

    bool write_scene_controller_header(
      std::ostream &out,
      const std::string_view model_name,
      const int depth_reuse_interval
    ) {
      nlohmann::ordered_json global_out_fields =
        nlohmann::ordered_json::array();
      for (const auto name : sbs_scene_controller::global_out_names) {
        global_out_fields.push_back({
          {"name", name},
          {"type", "float32"},
        });
      }
      nlohmann::ordered_json rule_state_fields =
        nlohmann::ordered_json::array();
      for (const auto &field : sbs_scene_controller::rule_state_fields) {
        rule_state_fields.push_back({
          {"name", field.name},
          {"type", field.json_type},
          {"gpu_encoding",
           scene_controller_encoding_name(field.gpu_encoding)},
        });
      }
      const nlohmann::ordered_json header {
        {"record", "header"},
        {"trace_schema", scene_controller_trace_schema},
        {"source", scene_controller_trace_source},
        {"capture", scene_controller_trace_capture},
        {"backend", "shadow_rules"},
        {"controller_schema", sbs_scene_controller::schema_version},
        {"rule_revision", sbs_scene_controller::rule_revision},
        {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
        {"global_out_fields", std::move(global_out_fields)},
        {"rule_state_fields", std::move(rule_state_fields)},
        {"config", {
          {"model", model_name},
          {"depth_reuse_interval", depth_reuse_interval},
          {"active_roi_authority", false},
        }},
      };
      out << header.dump() << '\n';
      out.flush();
      return out.good();
    }

    bool write_scene_controller_frame(
      std::ostream &out,
      const std::string_view frame_id,
      const std::size_t source_index,
      const bool depth_updated,
      const models::estimate_result &estimate,
      const scene_controller_global_words_t *global_words,
      const scene_controller_rule_words_t *rule_words
    ) {
      const bool available = estimate.scene_controller_snapshot_available;
      if (!available) {
        return false;
      }
      if (available != (global_words != nullptr) ||
          available != (rule_words != nullptr)) {
        return false;
      }

      nlohmann::ordered_json global_out = nullptr;
      nlohmann::ordered_json rule_state = nullptr;
      if (available) {
        if (
          !estimate.completed_frame_valid ||
          estimate.scene_controller_frame_id != estimate.completed_frame_id ||
          estimate.scene_controller_backend_generation == 0u ||
          !estimate.scene_controller_shadow ||
          !valid_scene_controller_global_words(*global_words) ||
          !valid_scene_controller_rule_words(*rule_words) ||
          (*rule_words)[sbs_scene_controller::index(
            sbs_scene_controller::rule_state_word_e::backend_generation
          )] != estimate.scene_controller_backend_generation
        ) {
          return false;
        }
        global_out = nlohmann::ordered_json::array();
        for (const auto word : *global_words) {
          global_out.push_back(std::bit_cast<float>(word));
        }
        rule_state = nlohmann::ordered_json::array();
        for (const auto &field : sbs_scene_controller::rule_state_fields) {
          const auto index = sbs_scene_controller::index(field.word);
          if (
            field.gpu_encoding ==
            sbs_scene_controller::gpu_encoding_e::uint_bits
          ) {
            rule_state.push_back((*rule_words)[index]);
          } else {
            rule_state.push_back(std::bit_cast<float>((*rule_words)[index]));
          }
        }
      }

      nlohmann::ordered_json frame {
        {"record", "frame"},
        {"frame_id", frame_id},
        {"source_index", source_index},
        {"depth_updated", depth_updated},
        {"snapshot_available", available},
        {"controller_frame_id",
         available ?
           nlohmann::ordered_json(estimate.scene_controller_frame_id) :
           nlohmann::ordered_json(nullptr)},
        {"backend_generation",
         available ?
           nlohmann::ordered_json(
             estimate.scene_controller_backend_generation
           ) :
           nlohmann::ordered_json(nullptr)},
        {"shadow", estimate.scene_controller_shadow},
        {"global_out", std::move(global_out)},
        {"rule_state", std::move(rule_state)},
      };
      out << frame.dump() << '\n';
      out.flush();
      return out.good();
    }

    bool write_adaptive_state_header(
      std::ostream &out,
      const std::string_view model_name,
      const config::video_t::sbs_t &cfg,
      const int depth_reuse_interval
    ) {
      nlohmann::json fields = nlohmann::json::array();
      for (const auto &field : sbs_adaptive_state::fields) {
        fields.push_back({
          {"name", field.name},
          {"type", field.json_type},
        });
      }
      nlohmann::json analysis_flag_bits = nlohmann::json::object();
      for (const auto &flag : sbs_adaptive_state::analysis_flag_bits) {
        analysis_flag_bits[std::string {flag.name}] = flag.bit;
      }
      const nlohmann::json header {
        {"record", "header"},
        {"schema", sbs_adaptive_state::schema_version},
        {"source", sbs_adaptive_state::source},
        {"capture", sbs_adaptive_state::capture},
        {"fields", std::move(fields)},
        {"analysis_flag_bits", std::move(analysis_flag_bits)},
        {"config", {
          {"model", model_name},
          {"profile", cfg.profile},
          {"pop_strength", cfg.pop_strength},
          {"adaptive_pop", cfg.adaptive_pop},
          {"adaptive_pop_max", cfg.adaptive_pop_max},
          {"zero_plane", cfg.zero_plane},
          {"depth_reuse_interval", depth_reuse_interval},
        }},
      };
      out << header.dump() << '\n';
      out.flush();
      return out.good();
    }

    bool write_adaptive_state_frame(
      std::ostream &out,
      std::string_view frame_id,
      std::size_t source_index,
      bool depth_updated,
      const adaptive_state_words_t &words,
      const config::video_t::sbs_t &cfg,
      const scene_plan_entry *scene_camera = nullptr
    ) {
      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
      };
      const float flags_value = scalar(word_e::cut_flags);
      if (flags_value < 0.0f ||
          flags_value >
            static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) ||
          std::trunc(flags_value) != flags_value) {
        return false;
      }
      const auto cut_flags = static_cast<std::uint32_t>(flags_value);
      const float analysis_flags_value = scalar(word_e::analysis_flags);
      if (
        analysis_flags_value < 0.0f ||
        analysis_flags_value >
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) ||
        std::trunc(analysis_flags_value) != analysis_flags_value
      ) {
        return false;
      }
      const float effective_ratio = cfg.adaptive_pop ?
                                      std::max(
                                        scalar(word_e::adaptive_pop_ratio),
                                        1.0f
                                      ) :
                                      1.0f;
      const float absolute_effective_pop = scene_camera ?
        scene_camera->absolute_pop_strength :
        static_cast<float>(cfg.pop_strength) * effective_ratio;
      if (!std::isfinite(absolute_effective_pop)) {
        return false;
      }

      out << "{\"record\":\"frame\",\"frame_id\":" << json_string(frame_id)
          << ",\"source_index\":" << source_index
          << ",\"depth_updated\":" << (depth_updated ? "true" : "false")
          << ",\"absolute_effective_pop\":" << absolute_effective_pop
          << ",\"scene_camera_override\":"
          << (scene_camera ? "true" : "false")
          << ",\"resolved_zero_anchor_shift_px\":"
          << (scene_camera ?
                scene_camera->zero_anchor_shift_px :
                scalar(word_e::zero_anchor_shift_px))
          << ",\"geometry_armed\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_geometry_armed) != 0u ?
                "true" : "false")
          << ",\"appearance_armed\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_appearance_armed) != 0u ?
                "true" : "false")
          << ",\"geometry_low_once\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_geometry_low_once) != 0u ?
                "true" : "false")
          << ",\"appearance_quiet_once\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_appearance_quiet_once) != 0u ?
                "true" : "false")
          << ",\"cut_latched\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_latched) != 0u ?
                "true" : "false")
          << ",\"appearance_recovery\":"
          << ((cut_flags & sbs_adaptive_state::cut_flag_appearance_recovery) != 0u ?
                "true" : "false")
          << ",\"geometry_confirmation_pending\":"
          << ((cut_flags &
               sbs_adaptive_state::cut_flag_geometry_confirmation_pending) != 0u ?
                "true" : "false")
          << ",\"hard_cut_pulse\":"
          << (scalar(word_e::hard_cut_pulse) > 0.5f ? "true" : "false")
          << ",\"hard_cut_count\":"
          << words[sbs_adaptive_state::index(word_e::hard_cut_count)]
          << ",\"external_cut_count\":"
          << words[sbs_adaptive_state::index(word_e::external_cut_count)]
          << ",\"empty_raw_count\":"
          << words[sbs_adaptive_state::index(word_e::empty_raw_count)]
          << ",\"collapsed_raw_count\":"
          << words[sbs_adaptive_state::index(word_e::collapsed_raw_count)]
          << ",\"values\":[";
      for (const auto &field : sbs_adaptive_state::fields) {
        const auto index = sbs_adaptive_state::index(field.word);
        if (index != 0u) {
          out << ',';
        }
        if (field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits) {
          out << words[index];
        } else if (
          field.gpu_encoding ==
          sbs_adaptive_state::gpu_encoding_e::uint_valued_float
        ) {
          out << static_cast<std::uint32_t>(scalar(field.word));
        } else {
          out << scalar(field.word);
        }
      }
      out << "]}\n";
      out.flush();
      return out.good();
    }

    template<class Writer>
    bool publish_adaptive_state_snapshot(
      const fs::path &path,
      Writer &&writer
    ) {
      std::ostringstream text;
      text.imbue(std::locale::classic());
      text << std::setprecision(std::numeric_limits<float>::max_digits10);
      if (!writer(text) || !text.good()) {
        return false;
      }
      const auto bytes = text.str();
      return publish_file_atomically(path, [&](const fs::path &temporary) {
        return write_bytes_durably(
          temporary,
          bytes.data(),
          bytes.size()
        );
      });
    }

    bool write_subject_state_trace(const fs::path &path,
                                   const std::vector<subject_state_record> &records) {
      std::ofstream out(path);
      if (!out) {
        return false;
      }
      out.imbue(std::locale::classic());
      out << std::setprecision(std::numeric_limits<float>::max_digits10);
      out << "{\n"
          << "  \"schema\": 1,\n"
          << "  \"source\": \"depth_subject_resolve_cs.SubjectState\",\n"
          << "  \"capture\": \"every-source-frame-after-estimator-update\",\n"
          << "  \"fields\": [\"subject_recenter_delta\", \"scene_age\", "
             "\"subject_depth_ema\", \"initialized\", \"stretch_lo\", "
             "\"stretch_inv_range\", \"depth_change_baseline_ema\", "
             "\"adaptive_pop_ratio\", \"zero_anchor_shift_px\", "
             "\"zero_anchor_valid\", \"cut_flags\", "
             "\"model_input_history_valid\"],\n"
          << "  \"frames\": [\n";
      for (size_t index = 0; index < records.size(); ++index) {
        const auto &record = records[index];
        out << "    {\"frame_id\": " << json_string(record.frame_id) << ", \"values\": [";
        for (size_t value_index = 0; value_index < record.values.size(); ++value_index) {
          if (value_index != 0) {
            out << ", ";
          }
          out << record.values[value_index];
        }
        out << "]}" << (index + 1 == records.size() ? "\n" : ",\n");
      }
      out << "  ]\n}\n";
      out.flush();
      return out.good();
    }

    // Keep output identities tied to source identities. Positional renumbering made a dropped
    // source frame silently shift every depth/SBS/source comparison by one.
    std::string source_frame_id(const fs::path &path) {
      std::string stem = path.stem().string();
      size_t split = stem.find_last_of('_');
      std::string id = split == std::string::npos ? "" : stem.substr(split + 1);
      if (!id.empty() && std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return std::isdigit(c);
          })) {
        return id;
      }
      return {};
    }

    bool numeric_frame_less(const fs::path &left, const fs::path &right) {
      auto left_id = source_frame_id(left);
      auto right_id = source_frame_id(right);
      if (!left_id.empty() && !right_id.empty()) {
        auto trim_zeroes = [](const std::string &value) {
          const auto first = value.find_first_not_of('0');
          return first == std::string::npos ? std::string_view(value).substr(value.size() - 1) :
                                             std::string_view(value).substr(first);
        };
        const auto left_number = trim_zeroes(left_id);
        const auto right_number = trim_zeroes(right_id);
        if (left_number.size() != right_number.size()) {
          return left_number.size() < right_number.size();
        }
        if (left_number != right_number) {
          return left_number < right_number;
        }
      } else if (left_id.empty() != right_id.empty()) {
        return !left_id.empty();
      }
      return left.filename().string() < right.filename().string();
    }

    std::string frame_id(const fs::path &path, size_t fallback) {
      auto id = source_frame_id(path);
      if (!id.empty()) {
        return id;
      }
      char buf[16];
      snprintf(buf, sizeof(buf), "%05zu", fallback);
      return buf;
    }

    // ---- D3D helpers ----

    ComPtr<ID3DBlob> compile(const char *file, const char *entry, const char *model,
                            const D3D_SHADER_MACRO *macros = nullptr) {
      std::wstring wfile(file, file + strlen(file));
      ComPtr<ID3DBlob> blob, err;
      HRESULT hr = D3DCompileFromFile(wfile.c_str(), macros, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, model, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "sbs-bench: shader compile failed [" << file << "]: "
                         << (err ? (const char *) err->GetBufferPointer() : "?");
        return nullptr;
      }
      return blob;
    }

    template<int N>
    ComPtr<ID3D11Buffer> const_buffer(ID3D11Device *dev, const float (&params)[N]) {
      static_assert(N % 4 == 0, "cbuffer must be 16-byte aligned");
      D3D11_BUFFER_DESC bd = {};
      bd.ByteWidth = N * 4;  // 16-byte aligned
      bd.Usage = D3D11_USAGE_IMMUTABLE;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA sd = {params, 0, 0};
      ComPtr<ID3D11Buffer> b;
      dev->CreateBuffer(&bd, &sd, &b);
      return b;
    }

    // ---- argument parsing ----

    enum class artifact_mode_e {
      evaluation,
      adaptive,
      conversion,
    };

    const char *artifact_mode_name(artifact_mode_e mode) {
      switch (mode) {
        case artifact_mode_e::evaluation:
          return "evaluation";
        case artifact_mode_e::adaptive:
          return "adaptive";
        case artifact_mode_e::conversion:
          return "conversion";
      }
      return "evaluation";
    }

    struct opts {
      std::string frames, out, model, depth_override_root, capabilities;
      std::string scene_cache, render_cache, scene_plan;
      std::string scene_controller;  // empty = inherit; otherwise off or shadow_rules
      std::string scene_controller_source_time;
      artifact_mode_e artifacts = artifact_mode_e::evaluation;
      bool follow = false;
      // Offline worker transport: publish one atomic header and one replace-in-place frame
      // snapshot. The consumer ACKs each frame before the producer can advance, so retaining
      // an ever-growing JSONL history is unnecessary.
      bool bounded_adaptive_state = false;
      std::string follow_format;
      std::size_t follow_count = 0;  // optional producer frame-count upper bound
      int eye_w = 0;  // 0 -> derive from source aspect; set with eye_h to test letterboxing
      int eye_h = 0;  // 0 -> match/derive from the input frame
      double output_scale = 1.0;  // per-eye linear scale vs source; preserves source aspect
      double pop_strength = -1.0;  // final shared stereo-parallax multiplier; <0 = conf
      double adaptive_pop_max = -1.0;  // absolute ceiling; <0 = conf
      std::string zero_plane;  // empty = conf; subject, median, or background
      bool simulate_hdr = false;  // decode sRGB frames into linear scRGB FP16 and use HDR paths
      double hdr_scale = 4.0;  // scRGB multiplier after sRGB EOTF (4.0 = 320-nit diffuse white)
      int max_width = 0;  // 0 -> use config max_encode_width
      int limit = 0;  // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      int depth_every = 1;  // infer every Nth source frame; reuse depth between updates
      // Apollo depth-pipeline A/B levers; <0 / false -> use the conf's value.
      double subject_recenter = -1.0;  // global subject recenter override
      int depth_short_side = 0;  // depth inference short-side override (0 = conf)
      double ema = -1.0;  // per-pixel depth EMA override (1.0 = off)
      double ema_edge_change = -1.0;
      double ema_edge_gradient = -1.0;
      double ema_edge_strength = -1.0;
      int subject_stretch = -1;  // -1 = conf, 0 = off, 1 = on
      double minmax_ema = -1.0;  // range-bounds EMA new-weight; <0 = conf
      int cuda_graph = -1;  // -1 = conf, 0 = ordinary enqueue, 1 = CUDA graph replay
      int adaptive_pop = -1;  // -1 = conf, 0 = off, 1 = on
      bool literal_bestv2 = false;  // reference-only: disable production resolution/pop scaling
      bool depth_override_all = false;  // reference-only: replace every inferred depth frame
    };

    bool parse_opts(int argc, char **argv, opts &o) {
      for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *n) -> std::string {
          if (i + 1 >= argc) {
            BOOST_LOG(error) << "sbs-bench: " << n << " needs a value";
            return "";
          }
          return argv[++i];
        };
        if (a == "--frames") {
          o.frames = next("--frames");
        } else if (a == "--capabilities") {
          o.capabilities = next("--capabilities");
        } else if (a == "--follow") {
          o.follow = true;
        } else if (a == "--bounded-adaptive-state") {
          o.bounded_adaptive_state = true;
        } else if (a == "--follow-format") {
          o.follow_format = next("--follow-format");
        } else if (a == "--follow-count") {
          const auto value = std::stoull(next("--follow-count"));
          if (value == 0u ||
              value > follow_max_sequence ||
              value > static_cast<unsigned long long>(
                        std::numeric_limits<std::size_t>::max())) {
            BOOST_LOG(error) << "sbs-bench: --follow-count must fit the positive 10-digit sequence";
            return false;
          }
          o.follow_count = static_cast<std::size_t>(value);
        } else if (a == "--out") {
          o.out = next("--out");
        } else if (a == "--scene-cache") {
          o.scene_cache = next("--scene-cache");
        } else if (a == "--render-cache") {
          o.render_cache = next("--render-cache");
        } else if (a == "--scene-plan") {
          o.scene_plan = next("--scene-plan");
        } else if (a == "--artifacts") {
          const auto value = next("--artifacts");
          if (value == "evaluation") {
            o.artifacts = artifact_mode_e::evaluation;
          } else if (value == "adaptive") {
            o.artifacts = artifact_mode_e::adaptive;
          } else if (value == "conversion") {
            o.artifacts = artifact_mode_e::conversion;
          } else {
            BOOST_LOG(error) << "sbs-bench: --artifacts must be evaluation, adaptive, or conversion";
            return false;
          }
        } else if (a == "--model") {
          o.model = next("--model");
        } else if (a == "--scene-controller") {
          o.scene_controller = next("--scene-controller");
          if (
            o.scene_controller != "off" &&
            o.scene_controller != "shadow_rules"
          ) {
            BOOST_LOG(error)
              << "sbs-bench: --scene-controller must be off or shadow_rules";
            return false;
          }
        } else if (a == "--scene-controller-source-time") {
          o.scene_controller_source_time =
            next("--scene-controller-source-time");
        } else if (a == "--eye-w") {
          o.eye_w = std::stoi(next("--eye-w"));
        } else if (a == "--eye-h") {
          o.eye_h = std::stoi(next("--eye-h"));
        } else if (a == "--output-scale") {
          o.output_scale = std::stod(next("--output-scale"));
        } else if (a == "--pop-strength") {
          o.pop_strength = std::stod(next("--pop-strength"));
        } else if (a == "--adaptive-pop") {
          o.adaptive_pop = 1;
        } else if (a == "--no-adaptive-pop") {
          o.adaptive_pop = 0;
        } else if (a == "--adaptive-pop-max") {
          o.adaptive_pop_max = std::stod(next("--adaptive-pop-max"));
        } else if (a == "--zero-plane") {
          o.zero_plane = next("--zero-plane");
        } else if (a == "--simulate-hdr") {
          o.simulate_hdr = true;
        } else if (a == "--hdr-scale") {
          o.hdr_scale = std::stod(next("--hdr-scale"));
        } else if (a == "--max-width") {
          o.max_width = std::stoi(next("--max-width"));
        } else if (a == "--limit") {
          o.limit = std::stoi(next("--limit"));
        } else if (a == "--output-every") {
          o.output_every = std::max(1, std::stoi(next("--output-every")));
        } else if (a == "--depth-every") {
          o.depth_every = std::stoi(next("--depth-every"));
        } else if (a == "--depth-override-root") {
          o.depth_override_root = next("--depth-override-root");
        } else if (a == "--depth-override-all") {
          o.depth_override_all = true;
        } else if (a == "--subject-recenter") {
          o.subject_recenter = std::stod(next("--subject-recenter"));
        } else if (a == "--depth-short-side") {
          o.depth_short_side = std::stoi(next("--depth-short-side"));
        } else if (a == "--subject-stretch") {
          o.subject_stretch = 1;
        } else if (a == "--no-subject-stretch") {
          o.subject_stretch = 0;
        } else if (a == "--ema") {
          o.ema = std::stod(next("--ema"));
        } else if (a == "--ema-edge-change") {
          o.ema_edge_change = std::stod(next("--ema-edge-change"));
        } else if (a == "--ema-edge-gradient") {
          o.ema_edge_gradient = std::stod(next("--ema-edge-gradient"));
        } else if (a == "--ema-edge-strength") {
          o.ema_edge_strength = std::stod(next("--ema-edge-strength"));
        } else if (a == "--minmax-ema") {
          o.minmax_ema = std::stod(next("--minmax-ema"));
        } else if (a == "--cuda-graph") {
          std::string v = next("--cuda-graph");
          if (v == "on" || v == "1" || v == "true") {
            o.cuda_graph = 1;
          } else if (v == "off" || v == "0" || v == "false") {
            o.cuda_graph = 0;
          } else {
            BOOST_LOG(error) << "sbs-bench: --cuda-graph must be on or off";
            return false;
          }
        } else if (a == "--literal-bestv2") {
          o.literal_bestv2 = true;
        } else {
          BOOST_LOG(error) << "sbs-bench: unknown arg '" << a << "'";
          return false;
        }
      }
      if (!o.capabilities.empty()) {
        if (!o.frames.empty() || !o.out.empty()) {
          BOOST_LOG(error) << "sbs-bench: --capabilities is a standalone query";
          return false;
        }
        return true;
      }
      if (o.frames.empty() || o.out.empty()) {
        BOOST_LOG(error) << "sbs-bench: --frames DIR and --out DIR are required";
        return false;
      }
      if (!o.scene_cache.empty() && !o.render_cache.empty()) {
        BOOST_LOG(error) << "sbs-bench: --scene-cache and --render-cache are mutually exclusive";
        return false;
      }
      if (!o.render_cache.empty()) {
        if (o.artifacts != artifact_mode_e::conversion ||
            o.scene_plan.empty()) {
          BOOST_LOG(error)
            << "sbs-bench: --render-cache requires conversion artifacts and --scene-plan";
          return false;
        }
        if (!o.depth_override_root.empty() || o.depth_override_all) {
          BOOST_LOG(error) << "sbs-bench: cache replay forbids depth overrides";
          return false;
        }
      } else if (!o.scene_plan.empty()) {
        BOOST_LOG(error) << "sbs-bench: --scene-plan requires --render-cache";
        return false;
      }
      if (!o.scene_cache.empty() &&
          o.artifacts == artifact_mode_e::evaluation) {
        BOOST_LOG(error) << "sbs-bench: --scene-cache requires adaptive or conversion artifacts";
        return false;
      }
      if (o.follow) {
        if (o.artifacts == artifact_mode_e::evaluation) {
          BOOST_LOG(error) << "sbs-bench: --follow supports adaptive or conversion artifacts only";
          return false;
        }
        if (o.follow_format != "png" &&
            o.follow_format != "bmp" &&
            o.follow_format != "pfm") {
          BOOST_LOG(error) << "sbs-bench: --follow-format must be png, bmp, or pfm";
          return false;
        }
        if (o.limit != 0 || !o.depth_override_root.empty() ||
            o.depth_override_all || o.output_every != 1) {
          BOOST_LOG(error) << "sbs-bench: --follow forbids --limit, depth overrides, and --output-every != 1";
          return false;
        }
        if (o.follow_format == "pfm" && o.simulate_hdr) {
          BOOST_LOG(error) << "sbs-bench: --simulate-hdr cannot be combined with PFM follow input";
          return false;
        }
      } else if (!o.follow_format.empty() || o.follow_count != 0u) {
        BOOST_LOG(error) << "sbs-bench: --follow-format/--follow-count require --follow";
        return false;
      }
      if (
        o.bounded_adaptive_state &&
        (!o.follow || o.artifacts == artifact_mode_e::evaluation)
      ) {
        BOOST_LOG(error)
          << "sbs-bench: --bounded-adaptive-state requires adaptive/conversion --follow";
        return false;
      }
      if (!(o.output_scale > 0.0 && o.output_scale <= 4.0)) {
        BOOST_LOG(error) << "sbs-bench: --output-scale must be greater than 0 and at most 4";
        return false;
      }
      if (o.pop_strength >= 0.0 && !(o.pop_strength >= 0.25 && o.pop_strength <= 2.0)) {
        BOOST_LOG(error) << "sbs-bench: --pop-strength must be between 0.25 and 2";
        return false;
      }
      if (o.adaptive_pop_max >= 0.0 &&
          !(o.adaptive_pop_max >= 0.25 && o.adaptive_pop_max <= 2.0)) {
        BOOST_LOG(error) << "sbs-bench: --adaptive-pop-max must be between 0.25 and 2";
        return false;
      }
      if (!o.zero_plane.empty() &&
          o.zero_plane != "subject" && o.zero_plane != "median" &&
          o.zero_plane != "background") {
        BOOST_LOG(error) << "sbs-bench: --zero-plane must be subject, median, or background";
        return false;
      }
      if (!(o.hdr_scale > 0.0 && o.hdr_scale <= 64.0)) {
        BOOST_LOG(error) << "sbs-bench: --hdr-scale must be greater than 0 and at most 64";
        return false;
      }
      if (o.depth_every < 1 || o.depth_every > 8) {
        BOOST_LOG(error) << "sbs-bench: --depth-every must be between 1 and 8";
        return false;
      }
      if (o.artifacts == artifact_mode_e::conversion && o.output_every != 1) {
        BOOST_LOG(error) << "sbs-bench: conversion artifacts require --output-every 1";
        return false;
      }
      if (o.ema_edge_change > 1.0 || o.ema_edge_gradient > 1.0 || o.ema_edge_strength > 1.0) {
        BOOST_LOG(error) << "sbs-bench: EMA edge thresholds and strength must be <=1";
        return false;
      }
      if (!o.depth_override_root.empty() && !fs::is_directory(o.depth_override_root)) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-root is not a directory";
        return false;
      }
      if (o.depth_override_all && o.depth_override_root.empty()) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-all requires --depth-override-root";
        return false;
      }
      if (o.depth_override_all && o.depth_every != 1) {
        BOOST_LOG(error) << "sbs-bench: --depth-override-all requires --depth-every 1";
        return false;
      }
      return true;
    }

    config::depth_model_info pick_model(const opts &o) {
      const auto &reg = config::depth_model_registry();
      std::string want = o.model;
      if (!want.empty()) {
        for (const auto &m : reg) {
          if (m.name == want) {
            return m;
          }
        }
        BOOST_LOG(warning) << "sbs-bench: model '" << want << "' not in registry; using active model";
      }
      return video::active_depth_model();
    }

  }  // namespace

  int run(int argc, char **argv) {
    opts o;
    if (!parse_opts(argc, argv, o)) {
      return 2;
    }
    if (!o.capabilities.empty()) {
      std::error_code ec;
      const fs::path path = o.capabilities;
      if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path(), ec);
      }
      nlohmann::ordered_json adaptive_analysis_flag_bits =
        nlohmann::ordered_json::object();
      for (const auto &flag : sbs_adaptive_state::analysis_flag_bits) {
        adaptive_analysis_flag_bits[std::string {flag.name}] = flag.bit;
      }
      nlohmann::ordered_json capabilities = {
        {"schema", 1},
        {"native_whole_clip", {
          {"artifact_modes", {"adaptive", "conversion"}},
          {"source_formats", {"png", "bmp", "pfm"}},
          {"follow_protocol_schema", 1},
           {"follow_global_first_sequence", true},
           {"adaptive_state_schema", sbs_adaptive_state::schema_version},
           {"adaptive_analysis_flag_bits", std::move(adaptive_analysis_flag_bits)},
          {"scene_controller_trace", {
            {"trace_schema", scene_controller_trace_schema},
            {"controller_schema", sbs_scene_controller::schema_version},
            {"rule_revision", sbs_scene_controller::rule_revision},
            {"ordered_abi_hash", sbs_scene_controller::ordered_abi_hash},
            {"backends", {"off", "shadow_rules"}},
            {"active_roi_authority", false},
            {"source_time_override", scene_controller_source_time_clock},
            {"file", scene_controller_trace_name},
            {"transports", {"jsonl-v1", "atomic-latest-v1"}},
            {"atomic_header_file", "scene_controller_header.json"},
            {"atomic_frame_file", "scene_controller_frame.json"},
            {"global_out_word_count",
             sbs_scene_controller::global_out_word_count},
            {"rule_state_word_count",
             sbs_scene_controller::rule_state_word_count},
          }},
           {"scene_cache_contract_schema",
            sbs_scene_cache::contract_schema},
          {"scene_cache_packed_sbs_contract", true},
          {"scene_cache_depth", {
            {"dtype", "float32-le"},
            {"layout", "row-major"},
            {"dxgi_format", "R32_FLOAT"},
            {"dimensions", "per-frame-metadata"},
            {"bytes_per_frame", nullptr},
          }},
          {"scene_cache_frame_metadata", {
            {"schema", sbs_scene_cache::frame_metadata_schema},
            {"word_count", sbs_scene_cache::frame_metadata_word_count},
            {"roi_transform_word_offset",
             sbs_scene_cache::roi_transform_word_offset},
            {"roi_transform_word_count",
             sbs_scene_cache::roi_transform_word_count},
            {"roi_transform_contract_schema",
             models::frame_roi_transform_contract_version},
          }},
          {"scene_cache_state", {
            {"schema", sbs_scene_cache::cached_state_schema},
            {"subject_word_count", render_state_words_t {}.size()},
            {"depth_frame_state_word_count",
             sbs_scene_cache::depth_frame_state_word_count},
            {"word_count", cached_state_words_t {}.size()},
            {"dtype", "uint32-le"},
          }},
          {"scene_plan", {
            {"schema", 1},
            {"version", "scene-plan-v1"},
            {"one_scene_per_replay", true},
            {"absolute_pop_strength", true},
            {"source_pixel_zero_anchor", true},
          }},
          {"render_cache_follow", true},
          {"render_skips_tensorrt", true},
          {"whole_clip_inference_attestation", {
            {"depth_inference_enabled", true},
            {"scheduled_depth_update_count", true},
            {"tensorrt_enqueue_count", true},
          }},
          {"atomic_sbs_publication", true},
          {"cache_budget", {
            {"enforced_by", "wrapper"},
            {"native_eviction", false},
            {"on_write_failure", "fail-closed"},
          }},
        }},
      };
      if (ec || !publish_json_atomically(path, capabilities)) {
        BOOST_LOG(error) << "sbs-bench: cannot publish capabilities document";
        return 2;
      }
      return 0;
    }
    if (!wic_init()) {
      BOOST_LOG(error) << "sbs-bench: WIC init failed";
      return 3;
    }

    // Evaluation keeps its established WIC raster contract. Whole-clip adaptive/conversion modes
    // additionally accept strict linear-scRGB PFM frames. A clip is one format family throughout:
    // silently mixing SDR raster and HDR float interchange would corrupt both color semantics and
    // timeline identity.
    std::vector<fs::path> frames;
    bool found_sdr_raster = false;
    bool found_hdr_pfm = false;
    bool found_mixed_sdr_formats = false;
    std::string sdr_raster_format;
    std::error_code ec;
    if (!fs::is_directory(o.frames, ec) || ec) {
      BOOST_LOG(error) << "sbs-bench: --frames is not a readable directory: " << o.frames;
      return 4;
    }
    if (o.follow) {
      found_hdr_pfm = o.follow_format == "pfm";
      found_sdr_raster =
        o.follow_format == "png" || o.follow_format == "bmp";
      sdr_raster_format = found_sdr_raster ? o.follow_format : "";
    } else {
      for (auto &e : fs::directory_iterator(o.frames, ec)) {
        if (!e.is_regular_file()) {
          continue;
        }
        auto ext = e.path().extension().string();
        for (auto &ch : ext) {
          ch = (char) tolower((unsigned char) ch);
        }
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
          const std::string format = ext == ".png" ? "png" : "jpeg";
          if (!sdr_raster_format.empty() && sdr_raster_format != format) {
            found_mixed_sdr_formats = true;
          }
          sdr_raster_format = format;
          found_sdr_raster = true;
          frames.push_back(e.path());
        } else if (ext == ".pfm" &&
                   o.artifacts != artifact_mode_e::evaluation) {
          found_hdr_pfm = true;
          frames.push_back(e.path());
        }
      }
    }
    if (o.artifacts != artifact_mode_e::evaluation &&
        ((found_sdr_raster && found_hdr_pfm) || found_mixed_sdr_formats)) {
      BOOST_LOG(error) << "sbs-bench: mixed source image formats are not a valid whole clip";
      return 4;
    }
    const bool pfm_input = found_hdr_pfm;
    if (pfm_input && o.simulate_hdr) {
      BOOST_LOG(error) << "sbs-bench: --simulate-hdr cannot be combined with real linear-scRGB PFM input";
      return 2;
    }
    if (!o.follow) {
      std::sort(frames.begin(), frames.end(), numeric_frame_less);
      if (o.limit > 0 && (int) frames.size() > o.limit) {
        frames.resize(o.limit);
      }
    }
    if (!o.follow && frames.empty()) {
      BOOST_LOG(error) << "sbs-bench: no supported "
                       << (o.artifacts == artifact_mode_e::evaluation ?
                             "png/jpg" :
                             "png/jpg/PFM")
                       << " frames in " << o.frames;
      return 4;
    }
    fs::create_directories(o.out, ec);
    if (ec) {
      BOOST_LOG(error) << "sbs-bench: cannot create output directory " << o.out;
      return 4;
    }
    const auto normalized_path = [](const fs::path &path) {
      std::error_code path_ec;
      auto absolute = fs::absolute(path, path_ec);
      return (path_ec ? path : absolute).lexically_normal();
    };
    const fs::path normalized_frames = normalized_path(o.frames);
    const fs::path normalized_out = normalized_path(o.out);
    if (!o.scene_cache.empty()) {
      const fs::path normalized_cache = normalized_path(o.scene_cache);
      if (normalized_cache == normalized_frames ||
          normalized_cache == normalized_out) {
        BOOST_LOG(error) << "sbs-bench: --scene-cache must be distinct from --frames and --out";
        return 4;
      }
      fs::create_directories(o.scene_cache, ec);
      if (ec || !fs::is_empty(o.scene_cache, ec) || ec) {
        BOOST_LOG(error) << "sbs-bench: --scene-cache must be a new or empty directory";
        return 4;
      }
    }
    if (!o.render_cache.empty()) {
      const fs::path normalized_cache = normalized_path(o.render_cache);
      if (normalized_cache == normalized_frames ||
          normalized_cache == normalized_out ||
          !fs::is_directory(o.render_cache, ec) || ec ||
          !fs::is_regular_file(o.scene_plan, ec) || ec) {
        BOOST_LOG(error)
          << "sbs-bench: --render-cache/--scene-plan must be readable and distinct "
             "from --frames/--out";
        return 4;
      }
    }
    const fs::path follow_progress_path =
      fs::path(o.out) / "follow_progress.json";
    std::optional<std::size_t> producer_frame_count;

    // Inherit the loaded config, then pin one depth update per source frame. The benchmark is
    // frame-driven rather than wall-clock-driven, so cadence throttling would make the result
    // depend on machine speed.
    auto sbs_cfg = config::video.sbs;
    if (o.pop_strength >= 0.0) {
      sbs_cfg.pop_strength = o.pop_strength;
    }
    if (o.adaptive_pop >= 0) {
      sbs_cfg.adaptive_pop = (o.adaptive_pop != 0);
    }
    if (o.adaptive_pop_max >= 0.0) {
      sbs_cfg.adaptive_pop_max = o.adaptive_pop_max;
    }
    if (!o.zero_plane.empty()) {
      sbs_cfg.zero_plane = o.zero_plane;
    }
    sbs_cfg.adaptive_pop_max = std::max(sbs_cfg.adaptive_pop_max, sbs_cfg.pop_strength);
    if (o.subject_recenter >= 0.0) {
      sbs_cfg.subject_recenter = o.subject_recenter;
    }
    if (o.depth_short_side > 0) {
      sbs_cfg.depth_short_side = o.depth_short_side;
    }
    if (o.subject_stretch >= 0) {
      sbs_cfg.subject_stretch = (o.subject_stretch != 0);
    }
    if (o.ema > 0.0) {
      sbs_cfg.ema = o.ema;  // A/B lever: depth EMA (1.0 = off)
    }
    if (o.ema_edge_change >= 0.0) {
      sbs_cfg.ema_edge_change = o.ema_edge_change;
    }
    if (o.ema_edge_gradient >= 0.0) {
      sbs_cfg.ema_edge_gradient = o.ema_edge_gradient;
    }
    if (o.ema_edge_strength >= 0.0) {
      sbs_cfg.ema_edge_strength = o.ema_edge_strength;
    }
    if (o.minmax_ema >= 0.0) {
      sbs_cfg.minmax_ema = o.minmax_ema;
    }
    if (o.cuda_graph >= 0) {
      sbs_cfg.cuda_graph = (o.cuda_graph != 0);
    }
    if (!o.scene_controller.empty()) {
      const auto parsed =
        config::parse_sbs_scene_controller(o.scene_controller);
      if (!parsed) {
        BOOST_LOG(error)
          << "sbs-bench: invalid explicit scene-controller backend";
        return 2;
      }
      sbs_cfg.scene_controller = *parsed;
    }
    config::sunshine.diagnostics_enabled = true;  // benchmark processes always measure
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    auto model = pick_model(o);
    const bool replay_mode = !o.render_cache.empty();
    if (replay_mode) {
      // Scene-cache replay is an attested zero-inference render pass. It has no estimator and
      // therefore cannot produce a fresh controller snapshot, regardless of the loaded config.
      sbs_cfg.scene_controller = config::sbs_scene_controller_e::off;
    }
    const bool whole_clip_mode = o.artifacts != artifact_mode_e::evaluation;
    const bool hdr_texture_input = pfm_input || o.simulate_hdr;
    const std::string discovered_input_frame_format =
      pfm_input ?
        "linear-scRGB-f32-pfm" :
        (sdr_raster_format == "jpeg" ?
           "sRGB-JPEG-WIC" :
           (sdr_raster_format == "bmp" ? "sRGB-BMP-WIC" : "sRGB-PNG-WIC"));
    const std::string discovered_input_texture_format =
      hdr_texture_input ? "R16G16B16A16_FLOAT" : "B8G8R8A8_UNORM";
    const std::string discovered_input_color_space =
      hdr_texture_input ? "linear-scRGB" : "sRGB-SDR";
    scene_cache_metadata replay_cache_metadata;
    std::vector<scene_plan_entry> scene_plan;
    std::size_t replay_first_sequence = 1u;
    std::size_t replay_frame_count = 0u;
    int effective_depth_every = o.depth_every;
    if (replay_mode) {
      std::string cache_error;
      if (!read_scene_cache_contract(
            o.render_cache,
            replay_cache_metadata,
            cache_error
          )) {
        BOOST_LOG(error) << "sbs-bench: invalid render cache: " << cache_error;
        return 6;
      }
      if (replay_cache_metadata.input_frame_format !=
            discovered_input_frame_format ||
          replay_cache_metadata.input_texture_format !=
            discovered_input_texture_format ||
          replay_cache_metadata.input_color_space !=
            discovered_input_color_space ||
          replay_cache_metadata.packed_texture_format !=
            (hdr_texture_input ?
               "R16G16B16A16_FLOAT" :
               "B8G8R8A8_UNORM") ||
          replay_cache_metadata.output_file_extension !=
            (pfm_input ? "pfm" : "png") ||
          replay_cache_metadata.output_frame_format !=
            (pfm_input ?
               "linear-scRGB-f32-pfm" :
               (o.simulate_hdr ?
                  "tone-mapped-sRGB-BGRA8-PNG-preview" :
                  "sRGB-BGRA8-PNG")) ||
          replay_cache_metadata.simulate_hdr != o.simulate_hdr ||
          (o.simulate_hdr &&
           replay_cache_metadata.hdr_scale != o.hdr_scale)) {
        BOOST_LOG(error)
          << "sbs-bench: cache/source color contract or frame count does not match replay input";
        return 6;
      }
      if (!read_scene_plan(
            o.scene_plan,
            scene_plan,
            cache_error
          )) {
        BOOST_LOG(error) << "sbs-bench: invalid scene plan: " << cache_error;
        return 6;
      }
      replay_first_sequence = scene_plan.front().start_sequence;
      replay_frame_count =
        scene_plan.front().end_sequence_exclusive -
        scene_plan.front().start_sequence;
      if (!validate_scene_cache_range(
            o.render_cache,
            replay_cache_metadata,
            replay_first_sequence,
            scene_plan.front().end_sequence_exclusive,
            cache_error
          ) ||
          (!o.follow && frames.size() != replay_frame_count) ||
          (o.follow_count > 0u && o.follow_count != replay_frame_count)) {
        BOOST_LOG(error)
          << "sbs-bench: render scene does not match its durable cache range"
          << (cache_error.empty() ? "" : ": " + cache_error);
        return 6;
      }
      if (o.follow) {
        o.follow_count = replay_frame_count;
      }
      sbs_cfg.pop_strength = replay_cache_metadata.pop_strength;
      sbs_cfg.adaptive_pop = true;
      sbs_cfg.adaptive_pop_max = replay_cache_metadata.adaptive_pop_max;
      sbs_cfg.subject_stretch = replay_cache_metadata.subject_stretch;
      o.literal_bestv2 = replay_cache_metadata.literal_bestv2;
      o.eye_w = replay_cache_metadata.eye_width;
      o.eye_h = replay_cache_metadata.eye_height;
      o.output_scale = replay_cache_metadata.output_scale;
      o.max_width = replay_cache_metadata.max_output_width;
      effective_depth_every = replay_cache_metadata.depth_reuse_interval;
      model.name = replay_cache_metadata.model_name;
      model.url = replay_cache_metadata.model_url;
    }
    const int max_width =
      o.max_width > 0 ? o.max_width : config::video.sbs.max_encode_width;
    const std::size_t follow_first_sequence =
      replay_mode ? replay_first_sequence : 1u;
    if (o.follow &&
        !publish_follow_progress(
          follow_progress_path,
          "running",
          o.follow_format,
          artifact_mode_name(o.artifacts),
          follow_first_sequence,
          0u,
          0,
          {}
        )) {
      BOOST_LOG(error) << "sbs-bench: cannot initialize atomic follow progress";
      return 4;
    }
    std::ofstream adaptive_state_stream;
    std::size_t adaptive_state_frame_count = 0;
    std::ofstream scene_controller_stream;
    std::size_t scene_controller_frame_count = 0;
    const bool scene_controller_trace_enabled =
      whole_clip_mode &&
      !replay_mode &&
      sbs_cfg.scene_controller ==
        config::sbs_scene_controller_e::shadow_rules;
    const fs::path scene_controller_trace_path =
      fs::path(o.out) / scene_controller_trace_name;
    const fs::path scene_controller_header_path =
      fs::path(o.out) / "scene_controller_header.json";
    const fs::path scene_controller_frame_path =
      fs::path(o.out) / "scene_controller_frame.json";
    const fs::path adaptive_state_header_path =
      fs::path(o.out) / "adaptive_state_header.json";
    const fs::path adaptive_state_frame_path =
      fs::path(o.out) / "adaptive_state_frame.json";
    scene_controller_source_time_t scene_controller_source_time;
    if (scene_controller_trace_enabled) {
      if (o.scene_controller_source_time.empty()) {
        BOOST_LOG(error)
          << "sbs-bench: shadow scene-controller traces require "
             "--scene-controller-source-time so dwell uses source time";
        return 2;
      }
      const std::size_t expected_source_time_count =
        o.follow ? o.follow_count : frames.size();
      std::string source_time_error;
      if (
        expected_source_time_count == 0u ||
        !read_scene_controller_source_time(
          o.scene_controller_source_time,
          expected_source_time_count,
          scene_controller_source_time,
          source_time_error
        )
      ) {
        BOOST_LOG(error)
          << "sbs-bench: invalid scene-controller source-time contract"
          << (source_time_error.empty() ?
                "" :
                ": " + source_time_error);
        return 2;
      }
    } else if (!o.scene_controller_source_time.empty()) {
      BOOST_LOG(error)
        << "sbs-bench: --scene-controller-source-time requires a whole-clip "
           "shadow_rules analysis pass";
      return 2;
    }
    if (whole_clip_mode) {
      {
        std::error_code cleanup_error;
        fs::remove(scene_controller_trace_path, cleanup_error);
        if (!cleanup_error) {
          fs::remove(scene_controller_header_path, cleanup_error);
        }
        if (!cleanup_error) {
          fs::remove(scene_controller_frame_path, cleanup_error);
        }
        if (cleanup_error) {
          BOOST_LOG(error)
            << "sbs-bench: cannot remove stale scene-controller trace artifacts";
          return 6;
        }
      }
      if (o.bounded_adaptive_state) {
        std::error_code cleanup_error;
        fs::remove(adaptive_state_header_path, cleanup_error);
        fs::remove(adaptive_state_frame_path, cleanup_error);
        if (!publish_adaptive_state_snapshot(
              adaptive_state_header_path,
              [&](std::ostream &out) {
                return write_adaptive_state_header(
                  out,
                  model.name,
                  sbs_cfg,
                  effective_depth_every
                );
              }
            )) {
          BOOST_LOG(error)
            << "sbs-bench: failed publishing bounded adaptive-state header";
          return 6;
        }
      } else {
        adaptive_state_stream.open(fs::path(o.out) / "adaptive_state.jsonl");
        if (!adaptive_state_stream) {
          BOOST_LOG(error) << "sbs-bench: cannot create adaptive_state.jsonl";
          return 6;
        }
        adaptive_state_stream.imbue(std::locale::classic());
        adaptive_state_stream
          << std::setprecision(std::numeric_limits<float>::max_digits10);
        if (!write_adaptive_state_header(
              adaptive_state_stream,
              model.name,
              sbs_cfg,
              effective_depth_every
            )) {
          BOOST_LOG(error)
            << "sbs-bench: failed writing adaptive_state.jsonl header";
          return 6;
        }
      }
      if (scene_controller_trace_enabled) {
        if (o.bounded_adaptive_state) {
          if (!publish_adaptive_state_snapshot(
                scene_controller_header_path,
                [&](std::ostream &out) {
                  return write_scene_controller_header(
                    out,
                    model.name,
                    effective_depth_every
                  );
                }
              )) {
            BOOST_LOG(error)
              << "sbs-bench: failed publishing bounded scene-controller header";
            return 6;
          }
        } else {
          scene_controller_stream.open(scene_controller_trace_path);
          if (!scene_controller_stream) {
            BOOST_LOG(error)
              << "sbs-bench: cannot create " << scene_controller_trace_name;
            return 6;
          }
          scene_controller_stream.imbue(std::locale::classic());
          scene_controller_stream
            << std::setprecision(std::numeric_limits<float>::max_digits10);
          if (!write_scene_controller_header(
                scene_controller_stream,
                model.name,
                effective_depth_every
              )) {
            BOOST_LOG(error)
              << "sbs-bench: failed writing scene-controller trace header";
            return 6;
          }
        }
      }
    }

    BOOST_LOG(info) << "sbs-bench: "
                    << (o.follow ? "following producer frames" :
                                   std::to_string(frames.size()) + " frames")
                    << ", model '" << model.name
                    << "', eye " << (o.eye_w > 0 ? std::to_string(o.eye_w) : "auto") << 'x'
                    << (o.eye_h > 0 ? std::to_string(o.eye_h) : "auto")
                    << ", depth_step "
                    << (effective_depth_every == 1 ? std::string("current-once") :
                                                     "reuse-" + std::to_string(effective_depth_every))
                    << ", profile " << sbs_cfg.profile
                    << ", literal_bestv2 " << (o.literal_bestv2 ? "on" : "off")
                    << ", depth_every " << effective_depth_every
                    << (replay_mode ? ", cache replay (TensorRT skipped)" : "")
                    << " -> " << o.out;

    // ---- D3D device + shaders ----
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL want_fl[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, want_fl, 2, D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
      BOOST_LOG(error) << "sbs-bench: D3D11CreateDevice failed";
      return 5;
    }

    // Harness-only GPU timestamps. CPU submission time is not a useful warp-cost measurement.
    ComPtr<ID3D11Query> warp_disjoint, warp_start, warp_end;
    D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    dev->CreateQuery(&qd, &warp_disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    dev->CreateQuery(&qd, &warp_start);
    dev->CreateQuery(&qd, &warp_end);

    const D3D_SHADER_MACRO scene_camera_macros[] = {
      {"SBS_SCENE_CAMERA_OVERRIDE", "1"},
      {nullptr, nullptr},
    };
    const auto *warp_macros = replay_mode ? scene_camera_macros : nullptr;
    auto vs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0");
    auto ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "main_ps", "ps_5_0", warp_macros);
    auto mask_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "mask_ps", "ps_5_0", warp_macros);
    auto mapping_ps_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_reprojection_ps.hlsl", "mapping_ps", "ps_5_0", warp_macros);
    auto coverage_cs_blob = compile(SUNSHINE_SHADERS_DIR "/sbs_forward_coverage_cs.hlsl", "main", "cs_5_0", warp_macros);
    auto warp_prefilter_cs_blob = compile(SUNSHINE_SHADERS_DIR "/depth_warp_prefilter_cs.hlsl", "main", "cs_5_0");
    if (!vs_blob || !ps_blob || !mask_ps_blob || !mapping_ps_blob || !coverage_cs_blob ||
        !warp_prefilter_cs_blob) {
      return 6;
    }
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, mask_ps, mapping_ps;
    ComPtr<ID3D11ComputeShader> coverage_cs, warp_prefilter_cs;
    if (FAILED(dev->CreateVertexShader(
          vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs)) ||
        FAILED(dev->CreatePixelShader(
          ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps)) ||
        FAILED(dev->CreatePixelShader(
          mask_ps_blob->GetBufferPointer(), mask_ps_blob->GetBufferSize(), nullptr, &mask_ps)) ||
        FAILED(dev->CreatePixelShader(mapping_ps_blob->GetBufferPointer(),
          mapping_ps_blob->GetBufferSize(), nullptr, &mapping_ps)) ||
        FAILED(dev->CreateComputeShader(coverage_cs_blob->GetBufferPointer(),
          coverage_cs_blob->GetBufferSize(), nullptr, &coverage_cs)) ||
        FAILED(dev->CreateComputeShader(warp_prefilter_cs_blob->GetBufferPointer(),
          warp_prefilter_cs_blob->GetBufferSize(), nullptr, &warp_prefilter_cs)) ||
        !vs || !ps || !mask_ps || !mapping_ps || !coverage_cs || !warp_prefilter_cs) {
      BOOST_LOG(error) << "sbs-bench: D3D11 shader creation failed";
      return 6;
    }

    ComPtr<ID3D11SamplerState> sampler;
    {
      D3D11_SAMPLER_DESC sd = {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      dev->CreateSamplerState(&sd, &sampler);
    }

    // Built after the first source frame reveals the source/output aspect relationship.
    ComPtr<ID3D11Buffer> repro_cb, repro_completed_cb;

    // ---- estimator ----
    // Cache replay is deliberately a pure D3D render pass: do not even construct the estimator,
    // because its constructor initializes TensorRT/CUDA resources.
    std::unique_ptr<models::video_depth_estimator> estimator;
    if (!replay_mode) {
      estimator = std::make_unique<models::video_depth_estimator>(
        dev,
        ctx,
        fs::path(SUNSHINE_ASSETS_DIR),
        sbs_cfg,
        model
      );
    }

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    ComPtr<ID3D11Texture2D> warp_mask_tex, warp_mask_stage;
    ComPtr<ID3D11RenderTargetView> warp_mask_rtv;
    ComPtr<ID3D11Texture2D> warp_mapping_tex, warp_mapping_stage;
    ComPtr<ID3D11RenderTargetView> warp_mapping_rtv;
    ComPtr<ID3D11Texture2D> coverage_tex;
    ComPtr<ID3D11UnorderedAccessView> coverage_uav;
    ComPtr<ID3D11ShaderResourceView> coverage_srv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    ComPtr<ID3D11Texture2D> depth_stage;
    ComPtr<ID3D11Texture2D> warp_depth_tex;
    ComPtr<ID3D11UnorderedAccessView> warp_depth_uav;
    ComPtr<ID3D11ShaderResourceView> warp_depth_srv;
    ComPtr<ID3D11Texture2D> ema_mask_stage;
    ComPtr<ID3D11Buffer> raw_depth_stage;
    ComPtr<ID3D11Buffer> subject_state_stage;
    ComPtr<ID3D11Buffer> scene_controller_global_stage;
    ComPtr<ID3D11Buffer> scene_controller_rule_stage;
    ComPtr<ID3D11Buffer> scene_controller_rule_summary_stage;
    ComPtr<ID3D11Texture2D> scene_cache_depth_stage;
    ComPtr<ID3D11Buffer> scene_cache_transform_stage;
    ComPtr<ID3D11Buffer> scene_cache_depth_frame_state_stage;
    // Evaluation keeps the legacy subject_state.json contract used by the bounded clip
    // scorer. Whole-clip follow/conversion already emits the superset adaptive-state
    // transport (JSONL for standalone tooling, atomic latest-record snapshots for the native
    // worker), so retaining a second per-frame vector there would make memory scale with clip
    // duration for no additional evidence.
    std::vector<subject_state_record> subject_state_records;
    if (!whole_clip_mode) {
      subject_state_records.reserve(frames.size());
    }
    bool raw_shape_written = false;
    bool warp_mapping_shape_written = false;
    float hdr_output_min = std::numeric_limits<float>::infinity();
    float hdr_output_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_nonfinite = 0;

    int written = 0;
    models::estimate_result est;
    bool cuda_graph_captured = false;
    bool have_depth_result = false;
    bool packed_target_valid = false;
    std::size_t tensorrt_enqueue_count = 0;
    bool scene_cache_contract_started = false;
    scene_cache_metadata scene_cache_metadata_value;
    std::size_t scene_plan_index = 0;
    UINT source_width = 0;
    UINT source_height = 0;
    const fs::path depth_override_dir = o.depth_override_root.empty() ? fs::path() :
                                          fs::path(o.depth_override_root) / fs::path(o.frames).filename();
    size_t expected_depth_override_frames = 0;
    size_t applied_depth_override_frames = 0;
    if (!depth_override_dir.empty()) {
      if (!fs::is_directory(depth_override_dir)) {
        BOOST_LOG(error) << "sbs-bench: depth-override clip directory is missing: "
                         << depth_override_dir;
        return 7;
      }
      for (size_t fi = 0; fi < frames.size(); ++fi) {
        expected_depth_override_frames += o.depth_override_all ||
                                          (fi % (size_t) o.depth_every) != 0;
      }
      size_t actual_depth_override_frames = 0;
      for (const auto &entry : fs::directory_iterator(depth_override_dir)) {
        const auto filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.starts_with("depth_") &&
            entry.path().extension() == ".png") {
          ++actual_depth_override_frames;
        }
      }
      if (actual_depth_override_frames != expected_depth_override_frames) {
        BOOST_LOG(error) << "sbs-bench: expected " << expected_depth_override_frames
                         << " depth overrides in " << depth_override_dir << ", found "
                         << actual_depth_override_frames;
        return 7;
      }
    }
    std::size_t processed_frame_count = 0;
    double scene_controller_elapsed_accumulator = 0.0;
    for (size_t fi = 0;; fi++) {
      const std::size_t global_sequence =
        replay_mode ? replay_first_sequence + fi : fi + 1u;
      fs::path current_frame;
      if (o.follow) {
        const auto next = wait_for_follow_frame(
          o.frames,
          o.follow_format,
          global_sequence,
          replay_mode ? replay_first_sequence : 1u,
          fi,
          o.follow_count
        );
        if (next.status == follow_wait_status_e::error) {
          BOOST_LOG(error) << "sbs-bench: follow queue failed: " << next.error;
          return 10;
        }
        if (next.producer_frame_count) {
          producer_frame_count = next.producer_frame_count;
        }
        if (next.status == follow_wait_status_e::complete) {
          break;
        }
        current_frame = next.frame;
      } else if (fi >= frames.size()) {
        break;
      } else {
        current_frame = frames[fi];
      }

      rgba_image img;
      scrgb_image hdr_img;
      const bool loaded = pfm_input ?
                            load_pfm(current_frame, hdr_img) :
                            load_png(current_frame, img);
      if (!loaded) {
        if (whole_clip_mode) {
          BOOST_LOG(error) << "sbs-bench: invalid source frame " << current_frame
                           << (pfm_input ?
                                 " (expected strict finite FP16-range linear-scRGB PF little-endian PFM)" :
                                 "");
          return 9;
        }
        BOOST_LOG(warning) << "sbs-bench: skip " << current_frame;
        continue;
      }
      const UINT frame_width = pfm_input ? hdr_img.w : img.w;
      const UINT frame_height = pfm_input ? hdr_img.h : img.h;
      if (!frame_width || !frame_height ||
          frame_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
          frame_height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        BOOST_LOG(error) << "sbs-bench: unsupported source dimensions in " << current_frame
                         << ": " << frame_width << "x" << frame_height;
        return 9;
      }
      if (replay_mode &&
          (frame_width != replay_cache_metadata.source_width ||
           frame_height != replay_cache_metadata.source_height)) {
        BOOST_LOG(error)
          << "sbs-bench: replay source dimensions do not match scene cache contract: "
          << frame_width << 'x' << frame_height << " vs "
          << replay_cache_metadata.source_width << 'x'
          << replay_cache_metadata.source_height;
        return 9;
      }
      if (source_width == 0) {
        source_width = frame_width;
        source_height = frame_height;
      } else if (frame_width != source_width || frame_height != source_height) {
        BOOST_LOG(error) << "sbs-bench: mixed source dimensions are not a valid clip: first frame "
                         << source_width << "x" << source_height << ", " << current_frame
                         << " is " << frame_width << "x" << frame_height;
        return 9;
      }
      const std::string output_id =
        replay_mode ? follow_frame_id(global_sequence) :
                      frame_id(current_frame, fi);
      if (scene_controller_trace_enabled) {
        double source_elapsed_seconds = 0.0;
        std::string source_time_error;
        if (!read_scene_controller_source_time_frame(
              scene_controller_source_time,
              output_id,
              source_elapsed_seconds,
              source_time_error
            )) {
          BOOST_LOG(error)
            << "sbs-bench: source-time identity does not match source frame "
            << output_id
            << (source_time_error.empty() ?
                  "" :
                  ": " + source_time_error);
          return 6;
        }
        scene_controller_elapsed_accumulator +=
          source_elapsed_seconds;
        if (!std::isfinite(scene_controller_elapsed_accumulator)) {
          BOOST_LOG(error)
            << "sbs-bench: accumulated source presentation time overflowed";
          return 6;
        }
      }

      // Input texture + SRV.
      D3D11_TEXTURE2D_DESC id = {};
      id.Width = frame_width;
      id.Height = frame_height;
      id.MipLevels = 1;
      id.ArraySize = 1;
      id.Format = hdr_texture_input ?
                    DXGI_FORMAT_R16G16B16A16_FLOAT :
                    DXGI_FORMAT_B8G8R8A8_UNORM;
      id.SampleDesc.Count = 1;
      id.Usage = D3D11_USAGE_IMMUTABLE;
      id.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      std::vector<uint16_t> hdr_rgba;
      const void *input_pixels = pfm_input ?
                                   static_cast<const void *>(hdr_img.rgba16.data()) :
                                   static_cast<const void *>(img.bgra.data());
      UINT input_pitch = frame_width * (pfm_input ? 8u : 4u);
      if (o.simulate_hdr) {
        hdr_rgba.resize((size_t) frame_width * frame_height * 4);
        for (size_t p = 0; p < (size_t) frame_width * frame_height; ++p) {
          const float b = srgb_to_linear(img.bgra[p * 4 + 0] / 255.0f) * (float) o.hdr_scale;
          const float g = srgb_to_linear(img.bgra[p * 4 + 1] / 255.0f) * (float) o.hdr_scale;
          const float r = srgb_to_linear(img.bgra[p * 4 + 2] / 255.0f) * (float) o.hdr_scale;
          hdr_rgba[p * 4 + 0] = float_to_half(r);
          hdr_rgba[p * 4 + 1] = float_to_half(g);
          hdr_rgba[p * 4 + 2] = float_to_half(b);
          hdr_rgba[p * 4 + 3] = float_to_half(1.0f);
        }
        input_pixels = hdr_rgba.data();
        input_pitch = frame_width * 8;
      }
      D3D11_SUBRESOURCE_DATA isd = {input_pixels, input_pitch, 0};
      ComPtr<ID3D11Texture2D> in_tex;
      if (FAILED(dev->CreateTexture2D(&id, &isd, &in_tex))) {
        BOOST_LOG(error) << "sbs-bench: input tex fail";
        continue;
      }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv);

      // First frame: size the SBS target. Per eye = the input resolution by default (so the clip
      // size, not a fixed constant, drives eval cost); --eye-h pins a specific output height.
      // The width is still capped at max_encode_width like the live path.
      if (o.artifacts != artifact_mode_e::adaptive && !sbs_tex) {
        int eh_target = o.eye_h > 0 ? o.eye_h :
                                      std::max(2, (int) std::lround((double) frame_height * o.output_scale));
        float aspect = (float) frame_width / (float) frame_height;
        int eye_w = o.eye_w > 0 ? o.eye_w : (o.eye_h > 0 ? std::max(1, (int) std::lround(eh_target * aspect)) : std::max(1, (int) std::lround((double) frame_width * o.output_scale)));
        int eye_h = eh_target;
        if (o.eye_w > 0 && o.eye_h <= 0) {
          eye_h = std::max(1, (int) std::lround(eye_w / aspect));
        }
        if (2 * eye_w > max_width) {
          const double scale = (double) max_width / (double) (2 * eye_w);
          eye_w = std::max(1, max_width / 2);
          eye_h = std::max(2, ((int) std::lround(eh_target * scale)) & ~1);
        }
        sbs_w = (UINT) (2 * eye_w);
        sbs_h = (UINT) eye_h;
        if (replay_mode &&
            (sbs_w != replay_cache_metadata.output_sbs_width ||
             sbs_h != replay_cache_metadata.output_sbs_height ||
             static_cast<UINT>(eye_w) !=
               replay_cache_metadata.output_eye_width ||
             static_cast<UINT>(eye_h) !=
               replay_cache_metadata.output_eye_height)) {
          BOOST_LOG(error)
            << "sbs-bench: resolved replay SBS raster does not match cache contract: "
            << sbs_w << 'x' << sbs_h << " vs "
            << replay_cache_metadata.output_sbs_width << 'x'
            << replay_cache_metadata.output_sbs_height;
          return 6;
        }
        const float eye_aspect = (float) eye_w / (float) eye_h;
        const float content_scale_x = eye_aspect > aspect ? aspect / eye_aspect : 1.0f;
        const float content_scale_y = eye_aspect < aspect ? eye_aspect / aspect : 1.0f;
        // Slot-for-slot mirror of the b2 `Constants` cbuffer in sbs_warp_common.hlsl. Keep both
        // live variants: the first draw must establish the packed target, while every later
        // completed draw preserves that target when DepthFrameState marks an all-invalid model
        // completion.
        float repro_params[8] = {
          sbs_cfg.subject_stretch ? 1.0f : 0.0f,
          content_scale_x,
          content_scale_y,
          (float) sbs_cfg.pop_strength,
          o.literal_bestv2 ? 1.0f : 0.0f,
          sbs_cfg.adaptive_pop ? 1.0f : 0.0f,
          (float) sbs_cfg.adaptive_pop_max,
          0.0f  // preserve_previous_on_invalid
        };
        repro_cb = const_buffer(dev.Get(), repro_params);
        repro_params[7] = 1.0f;
        repro_completed_cb = const_buffer(dev.Get(), repro_params);
        if (!repro_cb || !repro_completed_cb) {
          BOOST_LOG(error)
            << "sbs-bench: cannot create SBS reprojection constants";
          return 6;
        }
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = sbs_w;
        td.Height = sbs_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = hdr_texture_input ?
                      DXGI_FORMAT_R16G16B16A16_FLOAT :
                      DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &sbs_tex);
        dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv);
        dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv);
        if (o.artifacts == artifact_mode_e::evaluation) {
          D3D11_TEXTURE2D_DESC mask_desc = td;
          mask_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
          mask_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
          if (FAILED(dev->CreateTexture2D(&mask_desc, nullptr, &warp_mask_tex)) ||
              FAILED(dev->CreateRenderTargetView(warp_mask_tex.Get(), nullptr, &warp_mask_rtv))) {
            BOOST_LOG(error) << "sbs-bench: warp-mask texture creation failed";
            return 6;
          }
          D3D11_TEXTURE2D_DESC mapping_desc = td;
          mapping_desc.Format = DXGI_FORMAT_R32_FLOAT;
          mapping_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
          if (FAILED(dev->CreateTexture2D(&mapping_desc, nullptr, &warp_mapping_tex)) ||
              FAILED(dev->CreateRenderTargetView(warp_mapping_tex.Get(), nullptr,
                                                 &warp_mapping_rtv))) {
            BOOST_LOG(error) << "sbs-bench: warp-mapping texture creation failed";
            return 6;
          }
          D3D11_TEXTURE2D_DESC wd = td;
          wd.Format = DXGI_FORMAT_R32_UINT;
          wd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
          if (FAILED(dev->CreateTexture2D(&wd, nullptr, &coverage_tex)) ||
              FAILED(dev->CreateUnorderedAccessView(coverage_tex.Get(), nullptr,
                                                    &coverage_uav)) ||
              FAILED(dev->CreateShaderResourceView(coverage_tex.Get(), nullptr,
                                                   &coverage_srv))) {
            BOOST_LOG(error) << "sbs-bench: forward-coverage texture creation failed";
            return 6;
          }
        }
        D3D11_TEXTURE2D_DESC sd2 = td;
        sd2.Usage = D3D11_USAGE_STAGING;
        sd2.BindFlags = 0;
        sd2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        dev->CreateTexture2D(&sd2, nullptr, &sbs_stage);
        vp = {0, 0, (float) sbs_w, (float) sbs_h, 0, 1};
        if (o.artifacts == artifact_mode_e::evaluation) {
          std::ofstream mapping_shape(fs::path(o.out) / "warp_map_shape.json");
          if (!mapping_shape) {
            BOOST_LOG(error) << "sbs-bench: cannot write warp_map_shape.json";
            return 6;
          }
          mapping_shape.imbue(std::locale::classic());
          mapping_shape << std::setprecision(std::numeric_limits<float>::max_digits10);
          mapping_shape
            << "{\n"
            << "  \"schema\": 1,\n"
            << "  \"width\": " << sbs_w << ",\n"
            << "  \"height\": " << sbs_h << ",\n"
            << "  \"eye_width\": " << eye_w << ",\n"
            << "  \"eye_height\": " << eye_h << ",\n"
            << "  \"source_width\": " << frame_width << ",\n"
            << "  \"source_height\": " << frame_height << ",\n"
            << "  \"content_scale_x\": " << content_scale_x << ",\n"
            << "  \"content_scale_y\": " << content_scale_y << ",\n"
            << "  \"dtype\": \"float32-le\",\n"
            << "  \"layout\": \"row-major\",\n"
            << "  \"channels\": [\n"
            << "    \"raw_reproject_source_u_normalized\"\n"
            << "  ],\n"
            << "  \"validity\": {\"content\": \"derive from content_scale_x/content_scale_y and packed output coordinate\", \"forward_coverage\": \"warp_mask_<frame-id>.png red == 0 inside content\"},\n"
            << "  \"live_sample_source_u_normalized\": \"clamp(raw_reproject_source_u_normalized, 0, 1)\",\n"
            << "  \"derived_inverse_displacement_output_eye_px\": \"(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width\",\n"
            << "  \"derived_signed_binocular_disparity_px\": \"invert both eye maps at common source-U samples; x_right - x_left\"\n"
            << "}\n";
          mapping_shape.flush();
          if (!mapping_shape.good()) {
            BOOST_LOG(error) << "sbs-bench: failed writing warp_map_shape.json";
            return 6;
          }
          warp_mapping_shape_written = true;
        }
        BOOST_LOG(info) << "sbs-bench: input " << frame_width << "x" << frame_height << " -> SBS "
                        << sbs_w << "x" << sbs_h
                        << (pfm_input ?
                              " (linear scRGB PFM -> FP16)" :
                              (o.simulate_hdr ?
                                 " (linear scRGB FP16 HDR simulation)" :
                                 " (sRGB SDR)"));
      }

      // Submit and consume exactly one inference for this source frame.
      const auto input_color = hdr_texture_input ?
                                 models::input_color_space::scrgb_hdr :
                                 models::input_color_space::srgb;
      const bool depth_updated = replay_mode ?
        ((global_sequence - 1u) %
          static_cast<std::size_t>(effective_depth_every)) == 0u :
        (!have_depth_result ||
         (fi % static_cast<std::size_t>(effective_depth_every)) == 0u ||
         (
           scene_controller_trace_enabled &&
           fi + 1u == scene_controller_source_time.frame_count
         ));
      adaptive_state_words_t words {};
      const scene_plan_entry *active_scene_camera = nullptr;
      ComPtr<ID3D11Texture2D> cached_depth_texture;
      ComPtr<ID3D11Buffer> cached_state_buffer;
      ComPtr<ID3D11Buffer> cached_depth_frame_state_buffer;
      ComPtr<ID3D11Buffer> cached_roi_transform_buffer;
      if (replay_mode) {
        const std::size_t sequence = global_sequence;
        const std::string cache_stem = scene_cache_frame_stem(sequence);
        cached_state_words_t render_words {};
        frame_metadata_t frame_metadata {};
        if (!read_exact_file(
              fs::path(o.render_cache) / (cache_stem + ".state.u32"),
              render_words.data(),
              sizeof(render_words)
            ) ||
            !read_exact_file(
              fs::path(o.render_cache) / (cache_stem + ".meta.u32"),
              &frame_metadata,
              sizeof(frame_metadata)
            ) ||
            !sbs_scene_cache::valid_frame_metadata_for_state(
              frame_metadata,
              render_words,
              sequence,
              replay_cache_metadata.source_width,
              replay_cache_metadata.source_height,
              static_cast<std::uint32_t>(
                replay_cache_metadata.depth_reuse_interval
              ),
              sequence == replay_cache_metadata.processed_count
            ) ||
            !create_cached_depth_srv(
              dev.Get(),
              fs::path(o.render_cache) / (cache_stem + ".depth.r32f"),
              frame_metadata.depth[0u],
              frame_metadata.depth[1u],
              cached_depth_texture,
              est.depth
            ) ||
            !create_cached_roi_transform_srv(
              dev.Get(),
              frame_metadata,
              cached_roi_transform_buffer,
              est.depth_roi_transform
            )) {
          BOOST_LOG(error) << "sbs-bench: invalid cached depth/state/metadata triplet for global sequence "
                           << sequence;
          return 6;
        }
        while (scene_plan_index < scene_plan.size() &&
               sequence >=
                 scene_plan[scene_plan_index].end_sequence_exclusive) {
          ++scene_plan_index;
        }
        if (scene_plan_index >= scene_plan.size() ||
            sequence < scene_plan[scene_plan_index].start_sequence ||
            sequence >=
              scene_plan[scene_plan_index].end_sequence_exclusive) {
          BOOST_LOG(error) << "sbs-bench: scene plan did not resolve global sequence "
                           << sequence;
          return 6;
        }
        const auto &scene = scene_plan[scene_plan_index];
        active_scene_camera = &scene;
        if (!create_cached_state_srvs(
              dev.Get(),
              render_words,
              cached_state_buffer,
              est.subject,
              cached_depth_frame_state_buffer,
              est.depth_frame_state
            )) {
          BOOST_LOG(error) << "sbs-bench: cannot create cached render state for sequence "
                           << sequence;
          return 6;
        }
        std::copy_n(
          render_words.begin(),
          sbs_adaptive_state::render_prefix_word_count,
          words.begin()
        );
        // Analysis diagnostics deliberately are not part of the render cache. Keep their
        // unavailable sentinels explicit so a replay trace cannot be mistaken for fresh detector
        // evidence; the separate analysis trace remains authoritative.
        for (
          std::size_t index = sbs_adaptive_state::render_prefix_word_count;
          index < words.size();
          ++index
        ) {
          words[index] = std::bit_cast<std::uint32_t>(
            sbs_adaptive_state::initial_values[index]
          );
        }
        constexpr std::array replay_unavailable_words {
          sbs_adaptive_state::word_e::current_depth_change_fraction,
          sbs_adaptive_state::word_e::valid_depth_fraction,
          sbs_adaptive_state::word_e::effective_raw_range_width,
        };
        for (const auto word : replay_unavailable_words) {
          words[sbs_adaptive_state::index(word)] =
            std::bit_cast<std::uint32_t>(-1.0f);
        }
        est.raw_width = static_cast<int>(frame_metadata.depth[0u]);
        est.raw_height = static_cast<int>(frame_metadata.depth[1u]);
        have_depth_result = true;
      } else {
        if (depth_updated) {
          if (
            scene_controller_trace_enabled &&
            !estimator->
              set_next_scene_controller_elapsed_seconds_for_evaluation(
                static_cast<float>(
                  scene_controller_elapsed_accumulator
                )
              )
          ) {
            BOOST_LOG(error)
              << "sbs-bench: cannot bind source presentation time to "
                 "scene-controller update "
              << output_id;
            return 6;
          }
          estimator->estimate_depth(
            in_srv.Get(),
            input_color,
            static_cast<std::uint64_t>(fi)
          );
          est = estimator->finish_pending_depth_for_evaluation(input_color);
          if (whole_clip_mode &&
              (!est.completed_frame_valid ||
               est.completed_frame_id != static_cast<std::uint64_t>(fi))) {
            BOOST_LOG(error) << "sbs-bench: scheduled depth update for source frame "
                             << output_id << " did not complete the expected current-frame oracle "
                             << "(valid=" << (est.completed_frame_valid ? "true" : "false")
                             << ", completed_frame_id=" << est.completed_frame_id
                             << ", expected=" << fi << ')';
            return 6;
          }
          cuda_graph_captured = cuda_graph_captured || est.cuda_graph_active;
          ++tensorrt_enqueue_count;
          have_depth_result = true;
          scene_controller_elapsed_accumulator = 0.0;
        } else {
          // Match the live stream between depth ticks: color advances while all depth-derived
          // geometry remains the last completed result. The views remain owned by the estimator
          // and are valid until the next inference overwrites their backing resources.
        }
      }

      // Offline motion-compensation reference: replace only explicitly supplied held-depth
      // frames, while retaining the estimator's real subject/convergence state and production
      // warp. This keeps the experiment on the actual shader path without pretending the Python
      // flow prototype is production code.
      ComPtr<ID3D11Texture2D> override_depth_texture;
      ComPtr<ID3D11ShaderResourceView> override_depth_srv;
      if (!depth_override_dir.empty() &&
          (o.depth_override_all ||
           (fi % static_cast<std::size_t>(effective_depth_every)) != 0)) {
        const fs::path override_path = depth_override_dir / ("depth_" + output_id + ".png");
        if (!fs::exists(override_path) ||
            !load_depth_texture(dev.Get(), override_path, override_depth_texture,
                                override_depth_srv)) {
          BOOST_LOG(error) << "sbs-bench: missing or invalid depth override " << override_path;
          return 7;
        }
        est.depth = override_depth_srv;
        ++applied_depth_override_frames;
      }

      // Export one state sample for every source frame, even when --output-every skips composite
      // artifacts. This makes accepted shot pulses and the zero/pop latches directly testable
      // without adding any synchronization or readback to production.
      subject_state_record state_record;
      state_record.frame_id = output_id;
      if (whole_clip_mode) {
        if (!replay_mode &&
            !read_adaptive_state(dev.Get(), ctx.Get(), est.subject.Get(),
                                 subject_state_stage, words)) {
          BOOST_LOG(error) << "sbs-bench: cannot read adaptive state for frame " << output_id;
          return 6;
        }
        if (!o.scene_cache.empty()) {
          ComPtr<ID3D11Resource> depth_resource;
          ComPtr<ID3D11Texture2D> depth_texture;
          if (!est.depth ||
              (est.depth->GetResource(&depth_resource),
               FAILED(depth_resource.As(&depth_texture)))) {
            BOOST_LOG(error) << "sbs-bench: scene cache depth is not an R32 texture";
            return 6;
          }
          D3D11_TEXTURE2D_DESC depth_desc {};
          depth_texture->GetDesc(&depth_desc);
          if (!scene_cache_contract_started) {
            if (depth_desc.Format != DXGI_FORMAT_R32_FLOAT ||
                !depth_desc.Width || !depth_desc.Height) {
              BOOST_LOG(error) << "sbs-bench: scene cache requires finite R32_FLOAT depth";
              return 6;
            }
            scene_cache_metadata_value.source_width = source_width;
            scene_cache_metadata_value.source_height = source_height;
            scene_cache_metadata_value.input_frame_format =
              discovered_input_frame_format;
            scene_cache_metadata_value.input_texture_format =
              discovered_input_texture_format;
            scene_cache_metadata_value.input_color_space =
              discovered_input_color_space;
            scene_cache_metadata_value.model_name = model.name;
            scene_cache_metadata_value.model_url = model.url;
            scene_cache_metadata_value.pop_strength = sbs_cfg.pop_strength;
            scene_cache_metadata_value.adaptive_pop_max =
              sbs_cfg.adaptive_pop_max;
            scene_cache_metadata_value.subject_stretch =
              sbs_cfg.subject_stretch;
            scene_cache_metadata_value.literal_bestv2 =
              o.literal_bestv2;
            scene_cache_metadata_value.simulate_hdr = o.simulate_hdr;
            scene_cache_metadata_value.hdr_scale = o.hdr_scale;
            scene_cache_metadata_value.depth_reuse_interval =
              effective_depth_every;
            scene_cache_metadata_value.eye_width = o.eye_w;
            scene_cache_metadata_value.eye_height = o.eye_h;
            scene_cache_metadata_value.output_scale = o.output_scale;
            scene_cache_metadata_value.max_output_width = max_width;
            const auto cache_geometry = resolve_sbs_geometry(
              source_width,
              source_height,
              o.eye_w,
              o.eye_h,
              o.output_scale,
              max_width
            );
            scene_cache_metadata_value.output_eye_width =
              cache_geometry.eye_width;
            scene_cache_metadata_value.output_eye_height =
              cache_geometry.eye_height;
            scene_cache_metadata_value.output_sbs_width =
              cache_geometry.sbs_width;
            scene_cache_metadata_value.output_sbs_height =
              cache_geometry.sbs_height;
            scene_cache_metadata_value.packed_texture_format =
              hdr_texture_input ?
                "R16G16B16A16_FLOAT" :
                "B8G8R8A8_UNORM";
            scene_cache_metadata_value.output_frame_format =
              pfm_input ?
                "linear-scRGB-f32-pfm" :
                (o.simulate_hdr ?
                   "tone-mapped-sRGB-BGRA8-PNG-preview" :
                   "sRGB-BGRA8-PNG");
            scene_cache_metadata_value.output_file_extension =
              pfm_input ? "pfm" : "png";
            if (!publish_scene_cache_contract(
                  o.scene_cache,
                  scene_cache_metadata_value,
                  "running",
                  0u
                )) {
              BOOST_LOG(error) << "sbs-bench: cannot publish initial running scene cache contract";
              return 6;
            }
            scene_cache_contract_started = true;
          }
          const std::size_t sequence = fi + 1u;
          const std::string cache_stem = scene_cache_frame_stem(sequence);
          frame_metadata_t frame_metadata {};
          if (!make_scene_cache_frame_metadata(
                dev.Get(),
                ctx.Get(),
                est.depth_roi_transform.Get(),
                scene_cache_transform_stage,
                sequence,
                source_width,
                source_height,
                depth_desc.Width,
                depth_desc.Height,
                static_cast<std::uint32_t>(effective_depth_every),
                scene_controller_trace_enabled &&
                  sequence == scene_controller_source_time.frame_count,
                frame_metadata
              ) ||
              !cache_depth_texture_atomically(
                dev.Get(),
                ctx.Get(),
                est.depth.Get(),
                fs::path(o.scene_cache) / (cache_stem + ".depth.r32f"),
                scene_cache_depth_stage,
                depth_desc.Width,
                depth_desc.Height
              ) ||
              !cache_render_state_atomically(
                dev.Get(),
                ctx.Get(),
                words,
                est.depth_frame_state.Get(),
                scene_cache_depth_frame_state_stage,
                fs::path(o.scene_cache) / (cache_stem + ".state.u32")
              ) ||
              !cache_frame_metadata_atomically(
                frame_metadata,
                fs::path(o.scene_cache) / (cache_stem + ".meta.u32")
              )) {
            BOOST_LOG(error) << "sbs-bench: cannot durably publish scene cache triplet "
                             << sequence;
            return 6;
          }
        }
        for (std::size_t index = 0; index < state_record.values.size(); ++index) {
          state_record.values[index] = std::bit_cast<float>(words[index]);
        }
        const auto write_frame = [&](std::ostream &out) {
          return write_adaptive_state_frame(
            out,
            output_id,
            replay_mode ? global_sequence - 1u : fi,
            depth_updated,
            words,
            sbs_cfg,
            active_scene_camera
          );
        };
        const bool adaptive_state_written =
          o.bounded_adaptive_state ?
            publish_adaptive_state_snapshot(
              adaptive_state_frame_path,
              write_frame
            ) :
            write_frame(adaptive_state_stream);
        if (!adaptive_state_written) {
          BOOST_LOG(error) << "sbs-bench: cannot write adaptive state for frame " << output_id;
          return 6;
        }
        ++adaptive_state_frame_count;
        if (scene_controller_trace_enabled) {
          scene_controller_global_words_t global_words {};
          scene_controller_rule_words_t rule_words {};
          scene_controller_rule_summary_words_t rule_summary_words {};
          const scene_controller_global_words_t *global_words_ptr = nullptr;
          const scene_controller_rule_words_t *rule_words_ptr = nullptr;
          if (est.scene_controller_snapshot_available) {
            if (
              !read_scene_controller_buffer(
                dev.Get(),
                ctx.Get(),
                est.scene_controller_global_output.Get(),
                sizeof(float),
                scene_controller_global_stage,
                global_words
              ) ||
              !read_scene_controller_buffer(
                dev.Get(),
                ctx.Get(),
                est.scene_controller_rule_state.Get(),
                4u * sizeof(float),
                scene_controller_rule_stage,
                rule_words
              ) ||
              !read_scene_controller_buffer(
                dev.Get(),
                ctx.Get(),
                est.scene_controller_rule_summary.Get(),
                sizeof(float),
                scene_controller_rule_summary_stage,
                rule_summary_words
              )
            ) {
              BOOST_LOG(error)
                << "sbs-bench: cannot read scene-controller snapshot for frame "
                << output_id;
              return 6;
            }
            global_words_ptr = &global_words;
            rule_words_ptr = &rule_words;

            const auto summary_float =
              [&](const std::size_t index) {
                return std::bit_cast<float>(rule_summary_words[index]);
              };
            std::array<float, 3> scroll_votes_y {};
            std::array<float, 3> scroll_votes_x {};
            for (std::size_t group = 0u;
                 group < SBS_RULE_SUMMARY_EVENT_GROUP_COUNT;
                 ++group) {
              for (std::size_t field = 0u; field < 3u; ++field) {
                scroll_votes_y[field] += summary_float(
                  SBS_RULE_SUMMARY_EVENT_GROUP_BASE +
                  group * SBS_RULE_SUMMARY_EVENT_GROUP_STRIDE +
                  SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_Y_WEIGHT +
                  field);
                scroll_votes_x[field] += summary_float(
                  SBS_RULE_SUMMARY_EVENT_GROUP_BASE +
                  group * SBS_RULE_SUMMARY_EVENT_GROUP_STRIDE +
                  SBS_RULE_SUMMARY_EVENT_GROUP_SCROLL_X_WEIGHT +
                  field);
              }
            }
            std::ostringstream debug;
            debug.imbue(std::locale::classic());
            debug << "sbs-bench: rule-debug frame=" << (fi + 1u)
                  << " plan_fresh="
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_FRESH_ACTIVITY_MASS)
                  << " plan_prev_inside="
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_FRESH_ACTIVITY_INSIDE_PREVIOUS_TARGET)
                  << " scroll=["
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_WEIGHT)
                  << ','
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_SIGNED)
                  << ','
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_SCROLL_SUPPORT)
                  << ','
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_WEIGHT)
                  << ','
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_SIGNED)
                  << ','
                  << summary_float(
                       SBS_RULE_SUMMARY_PLAN_BASE +
                       SBS_RULE_SUMMARY_PLAN_GLOBAL_HORIZONTAL_SCROLL_SUPPORT)
                  << "] scroll_votes_y=[";
            for (std::size_t field = 0u; field < scroll_votes_y.size(); ++field) {
              if (field != 0u) {
                debug << ',';
              }
              debug << scroll_votes_y[field];
            }
            debug << "] scroll_votes_x=[";
            for (std::size_t field = 0u; field < scroll_votes_x.size(); ++field) {
              if (field != 0u) {
                debug << ',';
              }
              debug << scroll_votes_x[field];
            }
            debug << ']'
                  << " attr=[";
            for (std::size_t index = 0u;
                 index < SBS_RULE_SUMMARY_ATTRIBUTION_META_FLOAT_COUNT;
                 ++index) {
              if (index != 0u) {
                debug << ',';
              }
              debug << summary_float(
                SBS_RULE_SUMMARY_ATTRIBUTION_META_BASE + index);
            }
            debug << "] probe=[";
            for (std::size_t index = 0u;
                 index < SBS_RULE_SUMMARY_TEMPORAL_PROBE_FLOAT_COUNT;
                 ++index) {
              if (index != 0u) {
                debug << ',';
              }
              debug << summary_float(
                SBS_RULE_SUMMARY_TEMPORAL_PROBE_BASE + index);
            }
            debug << ']';
            if (fi == 0u) {
              constexpr std::size_t column_gutter_mass_field = 3u;
              constexpr std::size_t temporal_row_gutter_mass_field = 2u;
              debug << " media_columns=[";
              for (std::size_t column = 0u;
                   column < SBS_RULE_SUMMARY_COLUMN_COUNT;
                   ++column) {
                if (column != 0u) {
                  debug << ',';
                }
                debug << summary_float(
                  SBS_RULE_SUMMARY_COLUMN_BASE +
                  column * SBS_RULE_SUMMARY_COLUMN_STRIDE);
              }
              debug << "] media_rows=[";
              for (std::size_t row = 0u;
                   row < SBS_RULE_SUMMARY_MEDIA_ROW_COUNT;
                   ++row) {
                if (row != 0u) {
                  debug << ',';
                }
                debug << summary_float(
                  SBS_RULE_SUMMARY_MEDIA_ROW_BASE + row);
              }
              debug << "] gutter_columns=[";
              for (std::size_t column = 0u;
                   column < SBS_RULE_SUMMARY_COLUMN_COUNT;
                   ++column) {
                if (column != 0u) {
                  debug << ',';
                }
                debug << summary_float(
                  SBS_RULE_SUMMARY_COLUMN_BASE +
                  column * SBS_RULE_SUMMARY_COLUMN_STRIDE +
                  column_gutter_mass_field);
              }
              debug << "] gutter_rows=[";
              for (std::size_t row = 0u;
                   row < SBS_RULE_SUMMARY_TEMPORAL_ROW_COUNT;
                   ++row) {
                if (row != 0u) {
                  debug << ',';
                }
                debug << summary_float(
                  SBS_RULE_SUMMARY_TEMPORAL_ROW_BASE +
                  row * SBS_RULE_SUMMARY_TEMPORAL_ROW_STRIDE +
                  temporal_row_gutter_mass_field);
              }
              debug << ']';
            }
            BOOST_LOG(info) << debug.str();
          }
          const auto write_controller_frame = [&](std::ostream &out) {
            return write_scene_controller_frame(
              out,
              output_id,
              replay_mode ? global_sequence - 1u : fi,
              depth_updated,
              est,
              global_words_ptr,
              rule_words_ptr
            );
          };
          const bool controller_frame_written =
            o.bounded_adaptive_state ?
              publish_adaptive_state_snapshot(
                scene_controller_frame_path,
                write_controller_frame
              ) :
              write_controller_frame(scene_controller_stream);
          if (!controller_frame_written) {
            BOOST_LOG(error)
              << "sbs-bench: invalid or unwritable scene-controller snapshot for frame "
              << output_id;
            return 6;
          }
          ++scene_controller_frame_count;
        }
        if (!o.scene_cache.empty() &&
            !publish_scene_cache_contract(
              o.scene_cache,
              scene_cache_metadata_value,
              "running",
              fi + 1u
            )) {
          BOOST_LOG(error)
            << "sbs-bench: cannot acknowledge traced scene cache triplet "
            << (fi + 1u);
          return 6;
        }
      } else {
        if (!read_subject_state(dev.Get(), ctx.Get(), est.subject.Get(),
                                subject_state_stage, state_record.values)) {
          BOOST_LOG(error) << "sbs-bench: cannot read subject state for frame " << output_id;
          return 6;
        }
      }
      if (!whole_clip_mode) {
        subject_state_records.push_back(std::move(state_record));
      }

      if (o.artifacts == artifact_mode_e::adaptive) {
        sbs_perf::tick();
        if (o.follow &&
            !publish_follow_progress(
              follow_progress_path,
              "running",
              o.follow_format,
              artifact_mode_name(o.artifacts),
              follow_first_sequence,
              fi + 1u,
              written,
              producer_frame_count
            )) {
          BOOST_LOG(error) << "sbs-bench: cannot acknowledge adaptive follow frame "
                           << output_id;
          return 10;
        }
        processed_frame_count = fi + 1u;
        if (((fi + 1) % 20) == 0) {
          BOOST_LOG(info) << "sbs-bench: processed " << (fi + 1)
                          << (o.follow ? " follow frames" :
                                         "/" + std::to_string(frames.size()));
        }
        continue;
      }

      // Sampling output must never sample the depth/EMA/subject pipeline itself. Every source
      // frame above was inferred and consumed; only expensive composite/readback is skipped.
      if ((fi % (size_t) o.output_every) != 0) {
        continue;
      }

      ComPtr<ID3D11Buffer> scene_camera_cb;
      if (active_scene_camera) {
        float scene_camera_params[4] = {
          active_scene_camera->absolute_pop_strength,
          active_scene_camera->zero_anchor_shift_px,
          1.0f,
          0.0f,
        };
        scene_camera_cb = const_buffer(dev.Get(), scene_camera_params);
        if (!scene_camera_cb) {
          BOOST_LOG(error) << "sbs-bench: cannot create immutable scene camera constants";
          return 6;
        }
      }

      // Composite (mirrors display_vram::convert()'s SBS block): probe reprojection.
      const auto comp_t0 = std::chrono::steady_clock::now();
      ID3D11Buffer *active_repro_cb =
        packed_target_valid && est.depth_frame_state ?
          repro_completed_cb.Get() :
          repro_cb.Get();
      const bool time_warp = warp_disjoint && warp_start && warp_end;
      if (time_warp) {
        ctx->Begin(warp_disjoint.Get());
        ctx->End(warp_start.Get());
      }
      ID3D11ShaderResourceView *warp_depth = est.depth.Get();
      if (est.depth) {
        ComPtr<ID3D11Resource> depth_resource;
        est.depth->GetResource(&depth_resource);
        ComPtr<ID3D11Texture2D> depth_texture;
        if (FAILED(depth_resource.As(&depth_texture))) {
          BOOST_LOG(error) << "sbs-bench: warp prefilter input is not a texture";
          return 6;
        }
        D3D11_TEXTURE2D_DESC depth_desc {};
        depth_texture->GetDesc(&depth_desc);
        bool recreate_warp_depth = !warp_depth_tex || !warp_depth_uav || !warp_depth_srv;
        if (!recreate_warp_depth) {
          D3D11_TEXTURE2D_DESC current_desc {};
          warp_depth_tex->GetDesc(&current_desc);
          recreate_warp_depth = current_desc.Width != depth_desc.Width ||
                                current_desc.Height != depth_desc.Height ||
                                current_desc.Format != depth_desc.Format;
        }
        if (recreate_warp_depth) {
          depth_desc.Usage = D3D11_USAGE_DEFAULT;
          depth_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
          depth_desc.CPUAccessFlags = 0;
          depth_desc.MiscFlags = 0;
          warp_depth_tex.Reset();
          warp_depth_uav.Reset();
          warp_depth_srv.Reset();
          if (FAILED(dev->CreateTexture2D(&depth_desc, nullptr, &warp_depth_tex)) || FAILED(dev->CreateUnorderedAccessView(warp_depth_tex.Get(), nullptr, &warp_depth_uav)) || FAILED(dev->CreateShaderResourceView(warp_depth_tex.Get(), nullptr, &warp_depth_srv))) {
            BOOST_LOG(error) << "sbs-bench: warp prefilter resource creation failed";
            return 6;
          }
        }
        ctx->CSSetShader(warp_prefilter_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 1, est.depth.GetAddressOf());
        ctx->CSSetUnorderedAccessViews(0, 1, warp_depth_uav.GetAddressOf(), nullptr);
        ctx->Dispatch((depth_desc.Width + 15u) / 16u, (depth_desc.Height + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_prefilter_uav = nullptr;
        ID3D11ShaderResourceView *null_prefilter_srv = nullptr;
        ctx->CSSetUnorderedAccessViews(0, 1, &null_prefilter_uav, nullptr);
        ctx->CSSetShaderResources(0, 1, &null_prefilter_srv);
        warp_depth = warp_depth_srv.Get();
      }
      auto dispatch_coverage = [&](ID3D11ComputeShader *shader,
                                   ID3D11UnorderedAccessView *coverage_view) {
        const UINT clear_winner[4] = {0, 0, 0, 0};
        ctx->ClearUnorderedAccessViewUint(coverage_view, clear_winner);
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *cs_srvs[] = {
          in_srv.Get(),
          warp_depth,
          est.subject.Get(),
          nullptr,
          nullptr,
          est.depth_roi_transform.Get(),
        };
        ctx->CSSetShaderResources(0, (UINT) std::size(cs_srvs), cs_srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, &coverage_view, nullptr);
        ctx->CSSetConstantBuffers(2, 1, &active_repro_cb);
        if (scene_camera_cb) {
          ctx->CSSetConstantBuffers(3, 1, scene_camera_cb.GetAddressOf());
        }
        ctx->Dispatch(((sbs_w / 2u) + 15u) / 16u, (sbs_h + 15u) / 16u, 1u);
        ID3D11UnorderedAccessView *null_uav[] = {nullptr};
        ID3D11ShaderResourceView *null_cs_srvs[] = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
        };
        ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
        ctx->CSSetShaderResources(
          0,
          (UINT) std::size(null_cs_srvs),
          null_cs_srvs
        );
      };
      ctx->OMSetRenderTargets(1, sbs_rtv.GetAddressOf(), nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());

      ID3D11ShaderResourceView *srvs[] = {
        in_srv.Get(),
        warp_depth,
        est.subject.Get(),
        nullptr,
        est.depth_frame_state.Get(),
        est.depth_roi_transform.Get(),
      };
      ctx->PSSetShaderResources(0, (UINT) std::size(srvs), srvs);
      ID3D11Buffer *cb = active_repro_cb;
      ctx->PSSetConstantBuffers(2, 1, &cb);
      if (scene_camera_cb) {
        ctx->PSSetConstantBuffers(3, 1, scene_camera_cb.GetAddressOf());
      }
      ctx->Draw(3, 0);
      packed_target_valid = true;

      ID3D11RenderTargetView *null_rtv[] = {nullptr};
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ID3D11ShaderResourceView *null_srv[] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
      };
      ctx->PSSetShaderResources(0, (UINT) std::size(null_srv), null_srv);
      ID3D11Texture2D *final_sbs_tex = sbs_tex.Get();
      if (time_warp) {
        ctx->End(warp_end.Get());
        ctx->End(warp_disjoint.Get());
      }

      // Real composite-submission CPU cost. GPU warp time is captured separately below with D3D
      // timestamp queries; tick() advances the perf window.
      sbs_perf::add_sample_ms("sbs_composite_cpu", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - comp_t0).count());
      if (time_warp) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing = {};
        UINT64 start_tick = 0, end_tick = 0;
        ctx->Flush();
        while (ctx->GetData(warp_disjoint.Get(), &timing, sizeof(timing), 0) == S_FALSE) {
          std::this_thread::yield();
        }
        HRESULT hs = ctx->GetData(warp_start.Get(), &start_tick, sizeof(start_tick), 0);
        HRESULT he = ctx->GetData(warp_end.Get(), &end_tick, sizeof(end_tick), 0);
        if (SUCCEEDED(hs) && SUCCEEDED(he) && !timing.Disjoint && timing.Frequency > 0 && end_tick >= start_tick) {
          sbs_perf::add_sample_ms("warp_infer", (double) (end_tick - start_tick) * 1000.0 / (double) timing.Frequency);
        }
      }
      sbs_perf::tick();

      if (o.artifacts == artifact_mode_e::evaluation) {
        // Offline-only mask pass, deliberately outside the production warp timestamp/CPU sample.
        // It exports R=pre-fill disocclusion. This evidence must not perturb perf conclusions.
        if (est.depth) {
          dispatch_coverage(coverage_cs.Get(), coverage_uav.Get());
        }
        ctx->OMSetRenderTargets(1, warp_mask_rtv.GetAddressOf(), nullptr);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(mask_ps.Get(), nullptr, 0);
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *mask_srvs[] = {
          in_srv.Get(),
          warp_depth,
          est.subject.Get(),
          est.depth ? coverage_srv.Get() : nullptr,
          est.depth_frame_state.Get(),
          est.depth_roi_transform.Get(),
        };
        ctx->PSSetShaderResources(
          0,
          (UINT) std::size(mask_srvs),
          mask_srvs
        );
        ctx->PSSetConstantBuffers(2, 1, &cb);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(
          0,
          (UINT) std::size(null_srv),
          null_srv
        );

        // Offline-only exact inverse-warp mapping. This deliberately repeats the production
        // Reproject path after timing has ended: metric labels receive the shader's actual sampled
        // source coordinate instead of estimating correspondence again from the rendered colors.
        ctx->OMSetRenderTargets(1, warp_mapping_rtv.GetAddressOf(), nullptr);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(mapping_ps.Get(), nullptr, 0);
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *mapping_srvs[] = {
          in_srv.Get(),
          warp_depth,
          est.subject.Get(),
          nullptr,
          est.depth_frame_state.Get(),
          est.depth_roi_transform.Get(),
        };
        ctx->PSSetShaderResources(
          0,
          (UINT) std::size(mapping_srvs),
          mapping_srvs
        );
        ctx->PSSetConstantBuffers(2, 1, &cb);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(
          0,
          (UINT) std::size(null_srv),
          null_srv
        );

        char mapping_name[64];
        snprintf(mapping_name, sizeof(mapping_name), "warp_map_%s.f32", output_id.c_str());
        if (!dump_float_texture(dev.Get(), ctx.Get(), warp_mapping_tex.Get(),
                                fs::path(o.out) / mapping_name, warp_mapping_stage)) {
          BOOST_LOG(error) << "sbs-bench: failed writing " << mapping_name;
          return 6;
        }

        char mask_name[64];
        snprintf(mask_name, sizeof(mask_name), "warp_mask_%s.png", output_id.c_str());
        dump_bgra8_texture(dev.Get(), ctx.Get(), warp_mask_tex.Get(),
                           fs::path(o.out) / mask_name, warp_mask_stage);
      }

      // Preserve real HDR conversion as linear scRGB float interchange. SDR and the legacy
      // synthetic-HDR evaluator retain their existing PNG artifact behavior.
      ctx->CopyResource(sbs_stage.Get(), final_sbs_tex);
      D3D11_MAPPED_SUBRESOURCE m = {};
      bool output_completed = false;
      if (SUCCEEDED(ctx->Map(sbs_stage.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        bool frame_written = false;
        if (pfm_input) {
          char name[64];
          snprintf(name, sizeof(name), "sbs_%s.pfm", output_id.c_str());
          const fs::path output_path = fs::path(o.out) / name;
          frame_written = (o.follow || replay_mode) ?
                            publish_file_atomically(output_path, [&](const fs::path &temporary_path) {
                              return save_pfm(temporary_path, sbs_w, sbs_h, m);
                            }) :
                            save_pfm(output_path, sbs_w, sbs_h, m);
          ctx->Unmap(sbs_stage.Get(), 0);
          if (!frame_written) {
            BOOST_LOG(error) << "sbs-bench: non-finite or unwritable HDR SBS frame "
                             << output_path;
            return 6;
          }
        } else {
          std::vector<uint8_t> buf((size_t) sbs_w * sbs_h * 4);
          if (o.simulate_hdr) {
            for (UINT y = 0; y < sbs_h; ++y) {
              const uint16_t *row = (const uint16_t *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
              for (UINT x = 0; x < sbs_w; ++x) {
                const float r = half_to_float(row[x * 4 + 0]);
                const float g = half_to_float(row[x * 4 + 1]);
                const float b = half_to_float(row[x * 4 + 2]);
                for (float value : {r, g, b}) {
                  if (std::isfinite(value)) {
                    hdr_output_min = std::min(hdr_output_min, value);
                    hdr_output_max = std::max(hdr_output_max, value);
                  } else {
                    ++hdr_nonfinite;
                  }
                }
                hdr_preview_bgra(r, g, b, &buf[((size_t) y * sbs_w + x) * 4]);
              }
            }
          } else {
            for (UINT y = 0; y < sbs_h; y++) {
              memcpy(&buf[(size_t) y * sbs_w * 4], (uint8_t *) m.pData + (size_t) y * m.RowPitch, sbs_w * 4);
            }
          }
          ctx->Unmap(sbs_stage.Get(), 0);
          char name[64];
          snprintf(name, sizeof(name), "sbs_%s.png", output_id.c_str());
          const fs::path output_path = fs::path(o.out) / name;
          frame_written = (o.follow || replay_mode) ?
                            publish_file_atomically(output_path, [&](const fs::path &temporary_path) {
                              return save_png(temporary_path, sbs_w, sbs_h, buf);
                            }) :
                            save_png(output_path, sbs_w, sbs_h, buf);
          if (whole_clip_mode && !frame_written) {
            BOOST_LOG(error) << "sbs-bench: failed writing required SBS frame "
                             << output_path;
            return 6;
          }
        }
        if (frame_written) {
          written++;
          output_completed = true;
        }
        if (o.artifacts == artifact_mode_e::evaluation) {
          char dname[64];
          snprintf(dname, sizeof(dname), "depth_%s.png", output_id.c_str());
          dump_depth(dev.Get(), ctx.Get(), est.depth.Get(), fs::path(o.out) / dname, depth_stage);
          if (sbs_cfg.ema_edge_change > 0.0) {
            char mname[64];
            snprintf(mname, sizeof(mname), "ema_mask_%s.png", output_id.c_str());
            dump_uint_mask(dev.Get(), ctx.Get(), est.ema_motion_mask.Get(),
                           fs::path(o.out) / mname, ema_mask_stage);
          }
          char rname[64];
          snprintf(rname, sizeof(rname), "raw_%s.f32", output_id.c_str());
          dump_raw_model_depth(dev.Get(), ctx.Get(), est.raw_model_depth.Get(), est.raw_width,
                               est.raw_height, fs::path(o.out) / rname, raw_depth_stage);
          if (!raw_shape_written && est.raw_width > 0 && est.raw_height > 0) {
            std::ofstream shape(fs::path(o.out) / "raw_shape.json");
            if (shape) {
              shape << "{\n  \"width\": " << est.raw_width << ",\n  \"height\": "
                    << est.raw_height << ",\n  \"dtype\": \"float32-le\",\n"
                                         "  \"stage\": \"raw model output before transform/normalization/EMA/curvature\"\n}\n";
              raw_shape_written = true;
            }
          }
        }
      }
      if (o.follow) {
        if (!output_completed) {
          BOOST_LOG(error) << "sbs-bench: follow conversion did not finalize SBS frame "
                           << output_id;
          return 10;
        }
        if (!publish_follow_progress(
              follow_progress_path,
              "running",
              o.follow_format,
              artifact_mode_name(o.artifacts),
              follow_first_sequence,
              fi + 1u,
              written,
              producer_frame_count
            )) {
          BOOST_LOG(error) << "sbs-bench: cannot acknowledge conversion follow frame "
                           << output_id;
          return 10;
        }
      }
      processed_frame_count = fi + 1u;
      if (((fi + 1) % 20) == 0) {
        BOOST_LOG(info) << "sbs-bench: processed " << (fi + 1)
                        << (o.follow ? " follow frames" :
                                       "/" + std::to_string(frames.size()));
      }
    }

    if (o.follow &&
        (!producer_frame_count || processed_frame_count == 0u ||
         processed_frame_count != *producer_frame_count)) {
      BOOST_LOG(error) << "sbs-bench: follow sequence did not finish at the producer frame_count";
      return 10;
    }
    const std::size_t completed_frame_count =
      o.follow ? processed_frame_count : frames.size();
    if (scene_controller_trace_enabled) {
      std::string source_time_error;
      if (
        scene_controller_elapsed_accumulator != 0.0 ||
        !finish_scene_controller_source_time(
          scene_controller_source_time,
          source_time_error
        )
      ) {
        BOOST_LOG(error)
          << "sbs-bench: source-time contract was not consumed exactly"
          << (source_time_error.empty() ?
                "" :
                ": " + source_time_error);
        return 6;
      }
    }
    if (!o.scene_cache.empty()) {
      if (!scene_cache_contract_started ||
          completed_frame_count == 0u) {
        BOOST_LOG(error) << "sbs-bench: scene cache never published a frame";
        return 8;
      }
      scene_cache_metadata_value.frame_count = completed_frame_count;
      if (!publish_scene_cache_contract(
            o.scene_cache,
            scene_cache_metadata_value,
            "complete",
            completed_frame_count
          )) {
        BOOST_LOG(error) << "sbs-bench: cannot publish terminal scene cache contract";
        return 8;
      }
    }
    if (applied_depth_override_frames != expected_depth_override_frames) {
      BOOST_LOG(error) << "sbs-bench: applied " << applied_depth_override_frames
                       << " of " << expected_depth_override_frames << " expected depth overrides";
      return 7;
    }
    if (o.artifacts == artifact_mode_e::evaluation && !warp_mapping_shape_written) {
      BOOST_LOG(error) << "sbs-bench: no warp mapping shape contract was written";
      return 8;
    }
    if (!whole_clip_mode) {
      if (subject_state_records.size() != completed_frame_count ||
          !write_subject_state_trace(
            fs::path(o.out) / "subject_state.json", subject_state_records)) {
        BOOST_LOG(error) << "sbs-bench: failed writing complete subject_state.json";
        return 8;
      }
    }

    sbs_perf::dump_json((fs::path(o.out) / "sbs_perf.json").string());
    if (o.artifacts == artifact_mode_e::evaluation) {
      // Machine-readable execution contract. Evaluation must not scrape human log prose: custom
      // profile names are case-sensitive and fidelity runs must prove literal Bestv2 was active.
      std::ofstream contract(fs::path(o.out) / "contract.json");
      if (contract) {
        contract << "{\n"
                 << "  \"schema\": 17,\n"
                 << "  \"model\": " << json_string(model.name) << ",\n"
                 << "  \"profile\": " << json_string(sbs_cfg.profile) << ",\n"
                 << "  \"depth_step\": "
                 << json_string(o.depth_every == 1 ? std::string("current-once") : "reuse-" + std::to_string(o.depth_every))
                 << ",\n"
                 << "  \"depth_reuse_interval\": " << o.depth_every << ",\n"
                 << "  \"depth_compensation\": "
                 << json_string(o.depth_override_root.empty() ? "none" :
                                (o.depth_override_all ? "external-treatment" :
                                                        "external-reference"))
                 << ",\n"
                 << "  \"depth_override_frames\": " << applied_depth_override_frames << ",\n"
                 << "  \"ema\": " << sbs_cfg.ema << ",\n"
                 << "  \"ema_edge_change\": " << sbs_cfg.ema_edge_change << ",\n"
                 << "  \"ema_edge_gradient\": " << sbs_cfg.ema_edge_gradient << ",\n"
                 << "  \"ema_edge_strength\": " << sbs_cfg.ema_edge_strength << ",\n"
                 << "  \"adaptive_pop\": " << (sbs_cfg.adaptive_pop ? "true" : "false") << ",\n"
                 << "  \"adaptive_pop_max\": " << sbs_cfg.adaptive_pop_max << ",\n"
                 << "  \"zero_plane\": " << json_string(sbs_cfg.zero_plane) << ",\n"
                 << "  \"literal_bestv2\": " << (o.literal_bestv2 ? "true" : "false") << ",\n"
                 << "  \"cuda_graph\": " << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
                 << "  \"cuda_graph_captured\": " << (cuda_graph_captured ? "true" : "false") << ",\n"
                 << "  \"subject_state\": {\"file\": \"subject_state.json\", "
                    "\"schema\": 1, \"capture\": "
                    "\"every-source-frame-after-estimator-update\"},\n"
                 << "  \"warp_mask\": {\"red\": \"forward_disocclusion_before_fill\"},\n"
                 << "  \"warp_mapping\": {\n"
                 << "    \"file_pattern\": \"warp_map_<frame-id>.f32\",\n"
                 << "    \"shape_contract\": \"warp_map_shape.json\",\n"
                 << "    \"dtype\": \"float32-le\",\n"
                 << "    \"layout\": \"row-major\",\n"
                 << "    \"channels\": [\"raw_reproject_source_u_normalized\"],\n"
                 << "    \"live_sample_transform\": \"clamp(raw_reproject_source_u_normalized, 0, 1)\",\n"
                 << "    \"validity_companion\": \"warp_mask_<frame-id>.png:red=forward_disocclusion_before_fill; content validity derives from warp_map_shape.json\"\n"
                 << "  }\n"
                 << "}\n";
      }
      if (o.simulate_hdr) {
        std::ofstream stats(fs::path(o.out) / "hdr_output_stats.json");
        if (stats) {
          stats << "{\n  \"format\": \"linear-scRGB-fp16\",\n"
                << "  \"input_scale\": " << o.hdr_scale << ",\n"
                << "  \"output_min\": " << hdr_output_min << ",\n"
                << "  \"output_max\": " << hdr_output_max << ",\n"
                << "  \"nonfinite_components\": " << hdr_nonfinite << "\n}\n";
        }
      }
    } else {
      if (!o.bounded_adaptive_state) {
        adaptive_state_stream.flush();
      }
      if (
        (!o.bounded_adaptive_state && !adaptive_state_stream.good()) ||
        adaptive_state_frame_count != completed_frame_count
      ) {
        BOOST_LOG(error) << "sbs-bench: adaptive-state trace is incomplete";
        return 8;
      }
      if (!o.bounded_adaptive_state) {
        adaptive_state_stream.close();
      }
      if (scene_controller_trace_enabled) {
        if (scene_controller_frame_count != completed_frame_count) {
          BOOST_LOG(error)
            << "sbs-bench: scene-controller trace is incomplete";
          return 8;
        }
        if (o.bounded_adaptive_state) {
          if (
            !fs::is_regular_file(scene_controller_header_path) ||
            !fs::is_regular_file(scene_controller_frame_path) ||
            fs::exists(scene_controller_trace_path)
          ) {
            BOOST_LOG(error)
              << "sbs-bench: bounded scene-controller snapshots are incomplete";
            return 8;
          }
        } else {
          scene_controller_stream.flush();
          if (!scene_controller_stream.good()) {
            BOOST_LOG(error)
              << "sbs-bench: failed flushing scene-controller trace";
            return 8;
          }
          scene_controller_stream.close();
          if (!scene_controller_stream.good()) {
            BOOST_LOG(error)
              << "sbs-bench: failed closing scene-controller trace";
            return 8;
          }
        }
      } else if (
        scene_controller_frame_count != 0u ||
        fs::exists(scene_controller_trace_path) ||
        fs::exists(scene_controller_header_path) ||
        fs::exists(scene_controller_frame_path)
      ) {
        BOOST_LOG(error)
          << "sbs-bench: disabled scene controller produced a trace";
        return 8;
      }
      if (o.artifacts == artifact_mode_e::conversion &&
          written != static_cast<int>(completed_frame_count)) {
        BOOST_LOG(error) << "sbs-bench: conversion wrote " << written << " of "
                         << completed_frame_count << " required SBS frames";
        return 8;
      }

      // Adaptive-only runs deliberately skip allocating/compositing the packed render target, but
      // their contract still needs to freeze the geometry that the same resolved configuration
      // would render. Conversion runs attest the dimensions actually allocated above.
      UINT contract_eye_width = sbs_w / 2u;
      UINT contract_eye_height = sbs_h;
      UINT contract_sbs_width = sbs_w;
      UINT contract_sbs_height = sbs_h;
      if (o.artifacts == artifact_mode_e::adaptive &&
          source_width > 0u && source_height > 0u) {
        const int requested_eye_height =
          o.eye_h > 0 ?
            o.eye_h :
            std::max(2, (int) std::lround((double) source_height * o.output_scale));
        const float source_aspect = (float) source_width / (float) source_height;
        int resolved_eye_width =
          o.eye_w > 0 ?
            o.eye_w :
            (o.eye_h > 0 ?
               std::max(1, (int) std::lround(requested_eye_height * source_aspect)) :
               std::max(1, (int) std::lround((double) source_width * o.output_scale)));
        int resolved_eye_height = requested_eye_height;
        if (o.eye_w > 0 && o.eye_h <= 0) {
          resolved_eye_height =
            std::max(1, (int) std::lround(resolved_eye_width / source_aspect));
        }
        if (2 * resolved_eye_width > max_width) {
          const double scale = (double) max_width / (double) (2 * resolved_eye_width);
          resolved_eye_width = std::max(1, max_width / 2);
          resolved_eye_height =
            std::max(2, ((int) std::lround(requested_eye_height * scale)) & ~1);
        }
        contract_eye_width = (UINT) resolved_eye_width;
        contract_eye_height = (UINT) resolved_eye_height;
        contract_sbs_width = 2u * contract_eye_width;
        contract_sbs_height = contract_eye_height;
      }
      float contract_content_scale_x = 0.0f;
      float contract_content_scale_y = 0.0f;
      if (source_width > 0u && source_height > 0u &&
          contract_eye_width > 0u && contract_eye_height > 0u) {
        const float source_aspect = (float) source_width / (float) source_height;
        const float eye_aspect =
          (float) contract_eye_width / (float) contract_eye_height;
        contract_content_scale_x =
          eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
        contract_content_scale_y =
          eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;
      }
      const std::string depth_override_mode =
        o.depth_override_root.empty() ?
          "none" :
          (o.depth_override_all ? "external-treatment" : "external-reference");
      const std::string input_frame_format =
        pfm_input ?
          "linear-scRGB-f32-pfm" :
          (sdr_raster_format == "jpeg" ? "sRGB-JPEG-WIC" : "sRGB-PNG-WIC");
      const std::string input_texture_format =
        hdr_texture_input ? "R16G16B16A16_FLOAT" : "B8G8R8A8_UNORM";
      const std::string pipeline_color_space =
        hdr_texture_input ? "linear-scRGB" : "sRGB-SDR";
      const std::string packed_texture_format =
        hdr_texture_input ? "R16G16B16A16_FLOAT" : "B8G8R8A8_UNORM";
      const std::string output_frame_format =
        pfm_input ?
          "linear-scRGB-f32-pfm" :
          (o.simulate_hdr ? "tone-mapped-sRGB-BGRA8-PNG-preview" :
                            "sRGB-BGRA8-PNG");
      const std::string output_transfer = pfm_input ? "linear" : "sRGB";
      const std::string output_row_order = pfm_input ? "bottom-up" : "top-down";

      std::ofstream contract(fs::path(o.out) / "whole_clip_contract.json");
      if (!contract) {
        BOOST_LOG(error) << "sbs-bench: cannot create whole_clip_contract.json";
        return 8;
      }
      contract.imbue(std::locale::classic());
      contract << std::setprecision(std::numeric_limits<float>::max_digits10);
      contract
        << "{\n"
        << "  \"schema\": 1,\n"
        << "  \"artifact_mode\": " << json_string(artifact_mode_name(o.artifacts)) << ",\n"
        << "  \"inference_mode\": "
        << json_string(replay_mode ? "scene-cache-replay" :
                                    "single-pass-tensorrt")
        << ",\n"
        << "  \"depth_inference_enabled\": "
        << (replay_mode ? "false" : "true") << ",\n"
        << "  \"scheduled_depth_update_count\": "
        << (replay_mode ? 0u : tensorrt_enqueue_count) << ",\n"
        << "  \"tensorrt_enqueue_count\": " << tensorrt_enqueue_count << ",\n"
        << "  \"depth_provenance\": "
        << json_string(replay_mode ?
                         "scene-cache-contract-schema-2:per-frame-R32_FLOAT" :
                         "video_depth_estimator")
        << ",\n"
        << "  \"subject_state_provenance\": "
        << json_string(replay_mode ?
                         "scene-cache-contract-schema-2:subject-plus-depth-frame-state" :
                         "depth_subject_resolve_cs")
        << ",\n"
        << "  \"model\": " << json_string(model.name) << ",\n"
        << "  \"profile\": " << json_string(sbs_cfg.profile) << ",\n"
        << "  \"source_frame_count\": " << completed_frame_count << ",\n"
        << "  \"source_width\": " << source_width << ",\n"
        << "  \"source_height\": " << source_height << ",\n"
        << "  \"source_first_sequence\": " << follow_first_sequence << ",\n"
        << "  \"depth_reuse_interval\": " << effective_depth_every << ",\n"
        << "  \"resolved_runtime\": {\n"
        << "    \"model\": " << json_string(model.name) << ",\n"
        << "    \"model_url\": " << json_string(model.url) << ",\n"
        << "    \"profile\": " << json_string(sbs_cfg.profile) << ",\n"
        << "    \"pop_strength\": " << sbs_cfg.pop_strength << ",\n"
        << "    \"adaptive_pop\": "
        << (sbs_cfg.adaptive_pop ? "true" : "false") << ",\n"
        << "    \"adaptive_pop_max\": " << sbs_cfg.adaptive_pop_max << ",\n"
        << "    \"zero_plane\": " << json_string(sbs_cfg.zero_plane) << ",\n"
        << "    \"depth_short_side\": " << sbs_cfg.depth_short_side << ",\n"
        << "    \"depth_max_aspect\": " << sbs_cfg.depth_max_aspect << ",\n"
        << "    \"depth_width\": " << est.raw_width << ",\n"
        << "    \"depth_height\": " << est.raw_height << ",\n"
        << "    \"depth_reuse_interval\": " << effective_depth_every << ",\n"
        << "    \"subject_recenter\": " << sbs_cfg.subject_recenter << ",\n"
        << "    \"subject_stretch\": "
        << (sbs_cfg.subject_stretch ? "true" : "false") << ",\n"
        << "    \"ema\": " << sbs_cfg.ema << ",\n"
        << "    \"minmax_ema\": " << sbs_cfg.minmax_ema << ",\n"
        << "    \"ema_edge_change\": " << sbs_cfg.ema_edge_change << ",\n"
        << "    \"ema_edge_gradient\": " << sbs_cfg.ema_edge_gradient << ",\n"
        << "    \"ema_edge_strength\": " << sbs_cfg.ema_edge_strength << ",\n"
        << "    \"cuda_graph\": "
        << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
        << "    \"cuda_graph_captured\": "
        << (cuda_graph_captured ? "true" : "false") << ",\n"
        << "    \"literal_bestv2\": "
        << (o.literal_bestv2 ? "true" : "false") << ",\n"
        << "    \"scene_controller\": "
        << json_string(config::to_string(sbs_cfg.scene_controller)) << ",\n"
        << "    \"scene_controller_active_roi_authority\": false,\n"
        << "    \"simulate_hdr\": " << (o.simulate_hdr ? "true" : "false") << ",\n"
        << "    \"hdr_scale\": " << o.hdr_scale << ",\n"
        << "    \"input_color_space\": "
        << json_string(pipeline_color_space) << ",\n"
        << "    \"input_frame_format\": " << json_string(input_frame_format) << ",\n"
        << "    \"input_texture_format\": " << json_string(input_texture_format) << ",\n"
        << "    \"input_transfer\": "
        << json_string(hdr_texture_input ? "linear" : "sRGB") << ",\n"
        << "    \"input_primaries\": "
        << json_string(hdr_texture_input ? "scRGB/BT.709" : "sRGB/BT.709") << ",\n"
        << "    \"input_reference_white_nits\": "
        << (hdr_texture_input ? "80" : "null") << ",\n"
        << "    \"packed_texture_format\": "
        << json_string(packed_texture_format) << ",\n"
        << "    \"packed_color_space\": " << json_string(pipeline_color_space) << ",\n"
        << "    \"packed_transfer\": "
        << json_string(hdr_texture_input ? "linear" : "sRGB") << ",\n"
        << "    \"packed_primaries\": "
        << json_string(hdr_texture_input ? "scRGB/BT.709" : "sRGB/BT.709") << ",\n"
        << "    \"packed_reference_white_nits\": "
        << (hdr_texture_input ? "80" : "null") << ",\n"
        << "    \"output_frame_format\": " << json_string(output_frame_format) << ",\n"
        << "    \"output_transfer\": " << json_string(output_transfer) << ",\n"
        << "    \"output_primaries\": "
        << json_string(pfm_input ? "scRGB/BT.709" : "sRGB/BT.709") << ",\n"
        << "    \"output_reference_white_nits\": "
        << (pfm_input ? "80" : "null") << ",\n"
        << "    \"output_row_order\": " << json_string(output_row_order) << ",\n"
        << "    \"output_ffmpeg_pixel_format\": "
        << (pfm_input ? json_string("gbrpf32le") : "null") << ",\n"
        << "    \"output_scale\": " << o.output_scale << ",\n"
        << "    \"requested_eye_width\": " << o.eye_w << ",\n"
        << "    \"requested_eye_height\": " << o.eye_h << ",\n"
        << "    \"configured_max_encode_width\": "
        << sbs_cfg.max_encode_width << ",\n"
        << "    \"resolved_max_output_width\": " << max_width << ",\n"
        << "    \"input_limit\": " << o.limit << ",\n"
        << "    \"output_every\": " << o.output_every << ",\n"
        << "    \"follow_mode\": " << (o.follow ? "true" : "false") << ",\n"
        << "    \"follow_format\": "
        << (o.follow ? json_string(o.follow_format) : "null") << ",\n"
        << "    \"follow_count_bound\": "
        << (o.follow_count > 0u ? std::to_string(o.follow_count) : "null") << ",\n"
        << "    \"follow_producer_frame_count\": "
        << (producer_frame_count ?
              std::to_string(*producer_frame_count) :
              "null")
        << ",\n"
        << "    \"follow_frame_pattern\": "
        << (o.follow ?
              json_string("frame_%010d." + o.follow_format) :
              "null")
        << ",\n"
        << "    \"follow_first_sequence\": "
        << (o.follow ? std::to_string(follow_first_sequence) : "null") << ",\n"
        << "    \"follow_poll_interval_ms\": "
        << (o.follow ? "10" : "null") << ",\n"
        << "    \"follow_done_sentinel\": "
        << (o.follow ? json_string(".producer-done.json") : "null") << ",\n"
        << "    \"follow_failed_sentinel\": "
        << (o.follow ? json_string(".producer-failed.json") : "null") << ",\n"
        << "    \"follow_progress_file\": "
        << (o.follow ? json_string("follow_progress.json") : "null") << ",\n"
        << "    \"follow_native_input_deletion\": false,\n"
        << "    \"follow_atomic_sbs_publication\": "
        << ((o.follow || replay_mode) &&
            o.artifacts == artifact_mode_e::conversion ?
              "true" :
              "false")
        << ",\n"
        << "    \"output_eye_width\": " << contract_eye_width << ",\n"
        << "    \"output_eye_height\": " << contract_eye_height << ",\n"
        << "    \"output_sbs_width\": " << contract_sbs_width << ",\n"
        << "    \"output_sbs_height\": " << contract_sbs_height << ",\n"
        << "    \"content_scale_x\": " << contract_content_scale_x << ",\n"
        << "    \"content_scale_y\": " << contract_content_scale_y << ",\n"
        << "    \"depth_override_mode\": " << json_string(depth_override_mode) << ",\n"
        << "    \"depth_override_all\": "
        << (o.depth_override_all ? "true" : "false") << ",\n"
        << "    \"expected_depth_override_frames\": "
        << expected_depth_override_frames << ",\n"
        << "    \"applied_depth_override_frames\": "
        << applied_depth_override_frames << ",\n"
        << "    \"scene_cache_write\": "
        << (!o.scene_cache.empty() ? "true" : "false") << ",\n"
        << "    \"scene_cache_replay\": "
        << (replay_mode ? "true" : "false") << ",\n"
        << "    \"scene_cache_contract_schema\": "
        << ((!o.scene_cache.empty() || replay_mode) ?
              std::to_string(sbs_scene_cache::contract_schema) :
              "null")
        << ",\n"
        << "    \"scene_plan_schema\": "
        << (replay_mode ? "1" : "null") << ",\n"
        << "    \"scene_plan_version\": "
        << (replay_mode ? json_string("scene-plan-v1") : "null") << ",\n"
        << "    \"scene_start_sequence\": "
        << (replay_mode ?
              std::to_string(scene_plan.front().start_sequence) :
              "null")
        << ",\n"
        << "    \"scene_end_sequence_exclusive\": "
        << (replay_mode ?
              std::to_string(scene_plan.front().end_sequence_exclusive) :
              "null")
        << ",\n"
        << "    \"scene_cache_status_at_replay_start\": "
        << (replay_mode ?
              json_string(replay_cache_metadata.status) :
              "null")
        << ",\n"
        << "    \"scene_cache_processed_count_at_replay_start\": "
        << (replay_mode ?
              std::to_string(replay_cache_metadata.processed_count) :
              "null")
        << "\n"
        << "  },\n"
        << "  \"adaptive_state\": {"
        << (
             o.bounded_adaptive_state ?
               "\"transport\":\"atomic-latest-v1\","
               "\"header_file\":\"adaptive_state_header.json\","
               "\"frame_file\":\"adaptive_state_frame.json\","
               "\"retained_history\":false," :
               "\"transport\":\"jsonl-v1\","
               "\"file\":\"adaptive_state.jsonl\","
               "\"retained_history\":true,"
           )
        << "\"schema\":" << sbs_adaptive_state::schema_version
        << ",\"capture\":" << json_string(sbs_adaptive_state::capture)
        << ",\"frame_count\":"
        << adaptive_state_frame_count << "},\n"
        << "  \"scene_controller_source_time\": {"
        << "\"enabled\":"
        << (scene_controller_trace_enabled ? "true" : "false")
        << ",\"schema\":"
        << (scene_controller_trace_enabled ?
              std::to_string(scene_controller_source_time_schema) :
              "null")
        << ",\"clock\":"
        << (scene_controller_trace_enabled ?
              json_string(scene_controller_source_time_clock) :
              "null")
        << ",\"file_sha256\":"
        << (scene_controller_trace_enabled ?
              json_string(scene_controller_source_time.file_sha256) :
              "null")
        << ",\"frame_count\":"
        << (scene_controller_trace_enabled ?
              scene_controller_source_time.consumed_frame_count :
              0u)
        << ",\"total_elapsed_seconds\":"
        << (scene_controller_trace_enabled ?
              json_finite_double(
                scene_controller_source_time.total_elapsed_seconds
              ) :
              "0.0")
        << "},\n"
        << "  \"scene_controller\": {"
        << "\"enabled\":"
        << (scene_controller_trace_enabled ? "true" : "false")
        << ",\"backend\":"
        << json_string(config::to_string(sbs_cfg.scene_controller))
        << ",\"active_roi_authority\":false"
        << ",\"transport\":"
        << (scene_controller_trace_enabled ?
              json_string(
                o.bounded_adaptive_state ?
                  "atomic-latest-v1" :
                  "jsonl-v1"
              ) :
              "null")
        << ",\"file\":"
        << (scene_controller_trace_enabled &&
              !o.bounded_adaptive_state ?
              json_string(scene_controller_trace_name) :
              "null")
        << ",\"header_file\":"
        << (scene_controller_trace_enabled &&
              o.bounded_adaptive_state ?
              json_string("scene_controller_header.json") :
              "null")
        << ",\"frame_file\":"
        << (scene_controller_trace_enabled &&
              o.bounded_adaptive_state ?
              json_string("scene_controller_frame.json") :
              "null")
        << ",\"retained_history\":"
        << (scene_controller_trace_enabled &&
              !o.bounded_adaptive_state ?
              "true" :
              "false")
        << ",\"trace_schema\":"
        << (scene_controller_trace_enabled ?
              std::to_string(scene_controller_trace_schema) :
              "null")
        << ",\"controller_schema\":"
        << sbs_scene_controller::schema_version
        << ",\"rule_revision\":"
        << json_string(sbs_scene_controller::rule_revision)
        << ",\"ordered_abi_hash\":"
        << json_string(sbs_scene_controller::ordered_abi_hash)
        << ",\"frame_count\":" << scene_controller_frame_count
        << "},\n"
        << "  \"sbs\": {\"enabled\": "
        << (o.artifacts == artifact_mode_e::conversion ? "true" : "false")
        << ", \"file_pattern\": "
        << (o.artifacts == artifact_mode_e::conversion ?
              json_string(pfm_input ?
                            "sbs_<frame-id>.pfm" :
                            "sbs_<frame-id>.png") :
              "null")
        << ", \"frame_count\": " << written
        << ", \"width\": " << sbs_w
        << ", \"height\": " << sbs_h
        << ", \"frame_format\": " << json_string(output_frame_format)
        << ", \"transfer\": " << json_string(output_transfer)
        << ", \"primaries\": "
        << json_string(pfm_input ? "scRGB/BT.709" : "sRGB/BT.709")
        << ", \"reference_white_nits\": "
        << (pfm_input ? "80" : "null")
        << ", \"row_order\": " << json_string(output_row_order)
        << ", \"ffmpeg_pixel_format\": "
        << (pfm_input ? json_string("gbrpf32le") : "null")
        << ", \"atomic_publication\": "
        << ((o.follow || replay_mode) &&
            o.artifacts == artifact_mode_e::conversion ?
              "true" :
              "false")
        << "}\n"
        << "}\n";
      contract.flush();
      if (!contract.good()) {
        BOOST_LOG(error) << "sbs-bench: failed writing whole_clip_contract.json";
        return 8;
      }
      contract.close();
      if (!contract.good()) {
        BOOST_LOG(error) << "sbs-bench: failed closing whole_clip_contract.json";
        return 8;
      }
      if (o.follow &&
          !publish_follow_progress(
            follow_progress_path,
            "complete",
            o.follow_format,
            artifact_mode_name(o.artifacts),
            follow_first_sequence,
            completed_frame_count,
            written,
            producer_frame_count
          )) {
        BOOST_LOG(error) << "sbs-bench: cannot publish terminal follow progress";
        return 10;
      }
    }
    BOOST_LOG(info) << "sbs-bench: wrote " << written << " SBS frames + sbs_perf.json to " << o.out;
    return o.artifacts == artifact_mode_e::adaptive || written > 0 ? 0 : 8;
  }

#ifdef SUNSHINE_TESTS
  bool validate_scene_controller_source_time_for_test(
    const fs::path &path,
    const std::vector<std::string> &expected_frame_ids,
    source_time_validation_result &result,
    std::string &error
  ) {
    scene_controller_source_time_t source_time;
    if (
      expected_frame_ids.empty() ||
      !read_scene_controller_source_time(
        path,
        expected_frame_ids.size(),
        source_time,
        error
      )
    ) {
      return false;
    }
    for (const auto &frame_id : expected_frame_ids) {
      double elapsed_seconds = 0.0;
      if (!read_scene_controller_source_time_frame(
            source_time,
            frame_id,
            elapsed_seconds,
            error
          )) {
        return false;
      }
    }
    if (!finish_scene_controller_source_time(source_time, error)) {
      return false;
    }
    result = {
      .frame_count = source_time.consumed_frame_count,
      .total_elapsed_seconds = source_time.total_elapsed_seconds,
      .file_sha256 = source_time.file_sha256,
    };
    return true;
  }
#endif

}  // namespace sbs_bench

#else  // !_WIN32
namespace sbs_bench {
  int run(int, char **) {
    return 1;
  }
}  // namespace sbs_bench
#endif

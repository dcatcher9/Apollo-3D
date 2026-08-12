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
  #include <iterator>
  #include <limits>
  #include <locale>
  #include <memory>
  #include <optional>
  #include <sstream>
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

  // local includes
  #include "config.h"
  #include "crypto.h"
  #include "depth_coordinate_v2.h"
  #include "generated/sbs_adaptive_state_contract.h"
  #include "host_sbs_shader_cache.h"
  #include "logging.h"
  #include "offline_sbs_contract.h"
  #include "sbs_perf.h"
  #include "sbs_bench_depth_coordinate_v2.h"
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

    constexpr float direct_parallax_source_u_limit =
      models::depth_coordinate_v2::direct_container_limit;
    constexpr float direct_parallax_max_horizontal_slope =
      models::depth_coordinate_v2::max_horizontal_slope;
    constexpr float direct_parallax_slope_tolerance = 2.0e-5f;
    constexpr unsigned direct_geometry_contract_schema = 25u;
    constexpr unsigned direct_geometry_manifest_schema = 6u;
    constexpr std::string_view direct_geometry_warp_input =
      "external-final-parallax-with-diagnostic-order-v6";
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

    std::string sha256_hex(std::string_view bytes) {
      static constexpr char hex[] = "0123456789abcdef";
      const auto digest = crypto::hash(bytes);
      std::string encoded;
      encoded.reserve(digest.size() * 2u);
      for (const std::uint8_t byte : digest) {
        encoded.push_back(hex[byte >> 4u]);
        encoded.push_back(hex[byte & 0x0fu]);
      }
      return encoded;
    }

    std::string sha256_file_hex(const fs::path &path) {
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return {};
      }
      std::ostringstream bytes;
      bytes << stream.rdbuf();
      if (!stream.good() && !stream.eof()) {
        return {};
      }
      return sha256_hex(bytes.str());
    }

    bool read_file_snapshot(const fs::path &path, std::string &bytes) {
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return false;
      }
      bytes.assign(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
      );
      return !bytes.empty() && !stream.bad();
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
    bool load_pfm_stream(std::istream &stream, scrgb_image &out) {
      static_assert(std::endian::native == std::endian::little,
                    "PFM interchange requires a little-endian host");
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

    bool load_pfm(const fs::path &path, scrgb_image &out) {
      std::ifstream stream(path, std::ios::binary);
      return stream && load_pfm_stream(stream, out);
    }

    bool load_pfm(std::string_view bytes, scrgb_image &out) {
      std::istringstream stream(
        std::string(bytes),
        std::ios::in | std::ios::binary
      );
      return load_pfm_stream(stream, out);
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

    bool decode_wic_bgra(IWICBitmapDecoder *dec, rgba_image &out) {
      if (!dec) {
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
      rgba_image decoded;
      if (FAILED(conv->GetSize(&decoded.w, &decoded.h)) ||
          decoded.w == 0u || decoded.h == 0u ||
          decoded.w > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
          decoded.h > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        return false;
      }
      const std::uint64_t byte_count =
        static_cast<std::uint64_t>(decoded.w) * decoded.h * 4u;
      if (byte_count > std::numeric_limits<UINT>::max()) {
        return false;
      }
      decoded.bgra.resize(static_cast<std::size_t>(byte_count));
      if (FAILED(conv->CopyPixels(
            nullptr,
            decoded.w * 4u,
            static_cast<UINT>(decoded.bgra.size()),
            decoded.bgra.data()
          ))) {
        return false;
      }
      out = std::move(decoded);
      return true;
    }

    bool load_png(const fs::path &path, rgba_image &out) {
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateDecoderFromFilename(
            path.wstring().c_str(),
            nullptr,
            GENERIC_READ,
            WICDecodeMetadataCacheOnDemand,
            &dec
          ))) {
        return false;
      }
      return decode_wic_bgra(dec.Get(), out);
    }

    bool load_png(std::string_view bytes, rgba_image &out) {
      if (bytes.empty() || bytes.size() > std::numeric_limits<DWORD>::max()) {
        return false;
      }
      ComPtr<IWICStream> stream;
      ComPtr<IWICBitmapDecoder> dec;
      if (FAILED(g_wic->CreateStream(&stream)) ||
          FAILED(stream->InitializeFromMemory(
            reinterpret_cast<BYTE *>(const_cast<char *>(bytes.data())),
            static_cast<DWORD>(bytes.size())
          )) ||
          FAILED(g_wic->CreateDecoderFromStream(
            stream.Get(),
            nullptr,
            WICDecodeMetadataCacheOnLoad,
            &dec
          ))) {
        return false;
      }
      return decode_wic_bgra(dec.Get(), out);
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

    bool dump_float_texture_srv(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const fs::path &path,
      ComPtr<ID3D11Texture2D> &stage_cache
    ) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      return resource && SUCCEEDED(resource.As(&texture)) && texture &&
             dump_float_texture(dev, ctx, texture.Get(), path, stage_cache);
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

    // V2 evaluation depth artifact. Under host_sbs_v2 the estimator's normalized depth SRV is a
    // private cut field with no geometry authority, so the scored depth_<frame>.png is produced
    // from the authenticated raw model output instead. The container matches the historical
    // depth dump (16-bit grayscale PNG, values clamped to [0,1] scaled to 0-65535); the [0,1] normalization
    // est.depth received upstream is replicated here as an explicit per-frame finite min/max
    // normalization of the raw float buffer. Non-finite and degenerate-range values map to 0.
    bool dump_raw_model_depth_png(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, int width, int height, const fs::path &path, ComPtr<ID3D11Buffer> &stage_cache) {
      if (!srv || width <= 0 || height <= 0) {
        return false;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Buffer> buf;
      if (FAILED(res.As(&buf))) {
        return false;
      }
      D3D11_BUFFER_DESC d = {};
      buf->GetDesc(&d);
      if (d.ByteWidth < (UINT) ((size_t) width * height * sizeof(float))) {
        return false;
      }
      if (!stage_cache) {
        D3D11_BUFFER_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateBuffer(&sd, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), buf.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return false;
      }
      const auto *values = (const float *) m.pData;
      const size_t count = (size_t) width * height;
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < count; i++) {
        const float v = values[i];
        if (std::isfinite(v)) {
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      const float range = hi - lo;
      std::vector<uint16_t> gray(count);
      for (size_t i = 0; i < count; i++) {
        const float raw = values[i];
        float v = (std::isfinite(raw) && range > 0.0f) ? (raw - lo) / range : 0.0f;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        gray[i] = (uint16_t) (v * 65535.0f + 0.5f);
      }
      ctx->Unmap(stage_cache.Get(), 0);
      return save_gray16_png(path, (UINT) width, (UINT) height, gray);
    }

    // V2 evaluation structure artifact. Structure-consistency metrics (polarity/plateau) must
    // score the geometry the shipped V2 renderer intends, not raw model ordering: on thin
    // structure the raw model and the V2-limited field legitimately disagree. The canonical
    // coordinate SRV (est.shadow_coordinate) is Dump-3D-gated and never published in evaluation
    // mode, so this dumps est.shadow_candidate_parallax instead -- the always-published signed
    // candidate parallax, monotone in the canonical coordinate. Container matches
    // dump_raw_model_depth_png exactly: 16-bit grayscale PNG under an explicit per-frame finite
    // min/max normalization; non-finite and degenerate-range values map to 0.
    bool dump_structure_field_png(ID3D11Device *dev, ID3D11DeviceContext *ctx, ID3D11ShaderResourceView *srv, const fs::path &path, ComPtr<ID3D11Texture2D> &stage_cache) {
      if (!srv) {
        return false;
      }
      ComPtr<ID3D11Resource> res;
      srv->GetResource(&res);
      ComPtr<ID3D11Texture2D> tex;
      if (FAILED(res.As(&tex))) {
        return false;
      }
      D3D11_TEXTURE2D_DESC d = {};
      tex->GetDesc(&d);
      if (d.Format != DXGI_FORMAT_R32_FLOAT || !d.Width || !d.Height) {
        return false;
      }
      if (!stage_cache) {
        D3D11_TEXTURE2D_DESC sd = d;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        sd.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stage_cache))) {
          return false;
        }
      }
      ctx->CopyResource(stage_cache.Get(), tex.Get());
      D3D11_MAPPED_SUBRESOURCE m = {};
      if (FAILED(ctx->Map(stage_cache.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        return false;
      }
      const size_t count = (size_t) d.Width * d.Height;
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (UINT y = 0; y < d.Height; y++) {
        const float *row = (const float *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
        for (UINT x = 0; x < d.Width; x++) {
          const float v = row[x];
          if (std::isfinite(v)) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
          }
        }
      }
      const float range = hi - lo;
      std::vector<uint16_t> gray(count);
      for (UINT y = 0; y < d.Height; y++) {
        const float *row = (const float *) ((const uint8_t *) m.pData + (size_t) y * m.RowPitch);
        for (UINT x = 0; x < d.Width; x++) {
          const float raw = row[x];
          float v = (std::isfinite(raw) && range > 0.0f) ? (raw - lo) / range : 0.0f;
          v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
          gray[(size_t) y * d.Width + x] = (uint16_t) (v * 65535.0f + 0.5f);
        }
      }
      ctx->Unmap(stage_cache.Get(), 0);
      return save_gray16_png(path, d.Width, d.Height, gray);
    }

    using adaptive_state_words_t = sbs_adaptive_state::words_t;
    using render_state_words_t = models::depth_coordinate_v2::state_words_t;

    bool valid_adaptive_state_words(const adaptive_state_words_t &words) {
      using sbs_adaptive_state::word_e;
      if (words[sbs_adaptive_state::index(word_e::cut_contract_tag_bits)] !=
          sbs_adaptive_state::cut_contract_tag) {
        return false;
      }
      for (const auto &field : sbs_adaptive_state::fields) {
        const auto word_index = sbs_adaptive_state::index(field.word);
        if (field.name.starts_with("reserved_") &&
            words[word_index] != sbs_adaptive_state::initial_words[word_index]) {
          return false;
        }
        if (field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits) {
          continue;
        }
        if (!std::isfinite(std::bit_cast<float>(
              words[sbs_adaptive_state::index(field.word)]
            ))) {
          return false;
        }
      }
      const auto scalar = [&](const word_e word) {
        return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
      };
      const float hard_cut_pulse = scalar(word_e::hard_cut_pulse);
      const float cut_flags = scalar(word_e::cut_flags);
      const float analysis_flags = scalar(word_e::analysis_flags);
      return
        (hard_cut_pulse == 0.0f || hard_cut_pulse == 1.0f) &&
        cut_flags >= 0.0f &&
        cut_flags <= static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) &&
        std::trunc(cut_flags) == cut_flags &&
        analysis_flags >= 0.0f &&
        analysis_flags <=
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) &&
        std::trunc(analysis_flags) == analysis_flags &&
        words[sbs_adaptive_state::index(word_e::hard_cut_count)] <=
          sbs_adaptive_state::counter_max &&
        words[sbs_adaptive_state::index(word_e::empty_raw_count)] <=
          sbs_adaptive_state::counter_max &&
        words[sbs_adaptive_state::index(word_e::collapsed_raw_count)] <=
          sbs_adaptive_state::counter_max;
    }

    struct cut_state_record {
      std::string frame_id;
      adaptive_state_words_t words {};
    };

    // Benchmark-only state trace. This readback is deliberately confined to the synchronous
    // offline harness; the live capture loop must remain free of staging copies and Map calls.
    bool read_cut_state(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                        ID3D11ShaderResourceView *srv,
                        ComPtr<ID3D11Buffer> &stage_cache,
                        adaptive_state_words_t &state_words) {
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
      if (desc.ByteWidth < sbs_adaptive_state::word_count * sizeof(std::uint32_t) ||
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
      const auto *words = static_cast<const std::uint32_t *>(mapped.pData);
      std::memcpy(
        state_words.data(),
        words,
        state_words.size() * sizeof(std::uint32_t)
      );
      ctx->Unmap(stage_cache.Get(), 0);
      return valid_adaptive_state_words(state_words);
    }

    std::string scene_cache_frame_stem(std::size_t sequence) {
      return "frame_" + follow_frame_id(sequence);
    }

    struct scene_cache_metadata {
      UINT source_width = 0;
      UINT source_height = 0;
      UINT depth_width = 0;
      UINT depth_height = 0;
      std::string input_frame_format;
      std::string input_texture_format;
      std::string input_color_space;
      std::string model_name;
      std::string model_url;
      double pop_strength = 1.0;
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
      // Schema 3: offline conversion caches the authenticated Depth Coordinate V2 geometry and
      // pins both halves of the path: the producer closure and the exact live renderer closure.
      // The "depth" section carries the SIGNED one-eye final-parallax field (source-U units,
      // |value| <= the pointwise soft-container limit), and "state" carries the 12-word ParallaxState the
      // live renderer authenticates per pixel. Legacy pop/subject/adaptive knobs are gone; the
      // renderer identity and producer closure make the geometry provenance explicit.
      nlohmann::ordered_json contract = {
        {"schema", offline_sbs::scene_cache_contract_schema},
        {"status", status},
        {"source", {
          {"width", metadata.source_width},
          {"height", metadata.source_height},
          {"frame_format", metadata.input_frame_format},
          {"texture_format", metadata.input_texture_format},
          {"color_space", metadata.input_color_space},
        }},
        {"depth", {
          {"width", metadata.depth_width},
          {"height", metadata.depth_height},
          {"dxgi_format", "R32_FLOAT"},
          {"dtype", "float32-le"},
          {"layout", "row-major"},
          {"row_order", "top-down"},
          {"semantics", "depth-coordinate-v2-signed-final-parallax-source-u"},
          {"source_u_limit",
           static_cast<double>(models::depth_coordinate_v2::direct_container_limit)},
          {"file_pattern", "frame_%010d.depth.r32f"},
          {"bytes_per_frame",
           static_cast<std::uint64_t>(metadata.depth_width) *
             metadata.depth_height * sizeof(float)},
        }},
        {"state", {
          {"schema", 2},
          {"source", "depth_coordinate_v2.ParallaxState[0..2]"},
          {"word_count", render_state_words_t {}.size()},
          {"dtype", "uint32-le"},
          {"layout", "raw-word-order"},
          {"contract_schema", models::depth_coordinate_v2::contract_schema},
          {"contract_tag", models::depth_coordinate_v2::contract_tag},
          {"file_pattern", "frame_%010d.state.u32"},
          {"bytes_per_frame", sizeof(render_state_words_t)},
        }},
        {"render_config", {
          {"model", metadata.model_name},
          {"model_url", metadata.model_url},
          {"renderer", "depth-coordinate-v2-live-signed-parallax"},
          {"producer_source_closure_sha256",
           std::string(models::depth_coordinate_v2::shader_source_closure_sha256)},
          {"renderer_source_closure_sha256",
           std::string(models::host_sbs_shader_cache::
                         parallax_v2_live_renderer_source_closure_sha256)},
          {"pop_strength", metadata.pop_strength},
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

    bool cache_depth_texture_atomically(
      ID3D11Device *dev,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const fs::path &path,
      ComPtr<ID3D11Texture2D> &stage_cache,
      UINT expected_width,
      UINT expected_height,
      bool signed_container = false
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
              std::all_of(values.begin(), values.end(), [signed_container](float value) {
                if (!std::isfinite(value)) {
                  return false;
                }
                if (signed_container) {
                  return std::fabs(value) <=
                         models::depth_coordinate_v2::direct_container_limit;
                }
                return value >= 0.0f && value <= 1.0f;
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

    // Read the live renderer's 12-word ParallaxState back from the estimator's structured
    // buffer for durable caching. Validation matches create_cached_v2_state_srv: the replay
    // child re-authenticates the exact bytes written here.
    bool read_v2_state_words(ID3D11Device *dev, ID3D11DeviceContext *ctx,
                             ID3D11ShaderResourceView *srv,
                             ComPtr<ID3D11Buffer> &stage_cache,
                             render_state_words_t &words) {
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
      if (desc.ByteWidth < sizeof(render_state_words_t) ||
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
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      ctx->Unmap(stage_cache.Get(), 0);
      return true;
    }

    bool cache_v2_state_atomically(const render_state_words_t &words,
                                   const fs::path &path) {
      static_assert(std::endian::native == std::endian::little);
      namespace v2 = models::depth_coordinate_v2;
      if (words[7] != v2::contract_tag) {  // V2_STATE_WORD_CONTRACT_TAG_BITS
        return false;
      }
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        return write_bytes_durably(temporary_path, words.data(), sizeof(words));
      });
    }

    bool create_cached_float_srv(ID3D11Device *dev,
                                 const fs::path &path,
                                 UINT width,
                                 UINT height,
                                 ComPtr<ID3D11Texture2D> &texture,
                                 ComPtr<ID3D11ShaderResourceView> &srv,
                                 bool require_unit_range = true,
                                 float *field_minimum = nullptr,
                                 float *field_maximum = nullptr,
                                 std::string *content_sha256 = nullptr,
                                 std::vector<float> *uploaded_values = nullptr) {
      static_assert(std::endian::native == std::endian::little);
      if (!width || !height ||
          width > std::numeric_limits<UINT>::max() / sizeof(float) ||
          static_cast<std::uint64_t>(width) * height >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
      }
      std::vector<float> values(static_cast<std::size_t>(width) * height);
      if (!read_exact_file(
            path,
            values.data(),
            values.size() * sizeof(float)
          ) ||
          !std::all_of(values.begin(), values.end(), [require_unit_range](float value) {
            return std::isfinite(value) &&
                   (!require_unit_range || (value >= 0.0f && value <= 1.0f));
          })) {
        return false;
      }
      const auto [minimum_it, maximum_it] = std::minmax_element(values.begin(), values.end());
      if (field_minimum) {
        *field_minimum = *minimum_it;
      }
      if (field_maximum) {
        *field_maximum = *maximum_it;
      }
      if (content_sha256) {
        *content_sha256 = sha256_hex(std::string_view {
          reinterpret_cast<const char *>(values.data()),
          values.size() * sizeof(float)
        });
      }
      if (uploaded_values) {
        *uploaded_values = values;
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

    float measured_horizontal_source_u_slope(
      const std::vector<float> &values,
      const UINT width,
      const UINT height,
      const double source_u_per_stored_unit
    ) {
      float maximum = 0.0f;
      for (UINT y = 0; y < height; ++y) {
        const auto row = static_cast<std::size_t>(y) * width;
        for (UINT x = 1; x < width; ++x) {
          const auto index = row + x;
          const double slope = std::abs(
            static_cast<double>(values[index]) - values[index - 1u]
          ) * source_u_per_stored_unit * width;
          maximum = std::max(maximum, static_cast<float>(slope));
        }
      }
      return maximum;
    }

    bool publish_float_field(const fs::path &path, const std::vector<float> &values) {
      if (values.empty()) {
        return false;
      }
      return publish_file_atomically(path, [&](const fs::path &temporary_path) {
        return write_bytes_durably(
          temporary_path,
          values.data(),
          values.size() * sizeof(float)
        );
      });
    }

    // Re-authenticate the complete cached ParallaxState camera, checksum, frame authorization,
    // and reserved-word contract before exposing its renderer token. Integer words remain raw
    // bit patterns because an integrity checksum can legitimately encode non-finite float bits.
    bool create_cached_v2_state_srv(ID3D11Device *dev,
                                    const render_state_words_t &words,
                                    ComPtr<ID3D11Buffer> &buffer,
                                    ComPtr<ID3D11ShaderResourceView> &srv) {
      namespace v2 = models::depth_coordinate_v2;
      static_assert(!v2::model_calibrations.empty());
      if (!v2::parallax_state_words_are_authenticated(
            words,
            v2::model_calibrations.front().raw_coordinate_scale
          )) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(words);
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = 4 * sizeof(float);
      D3D11_SUBRESOURCE_DATA initial = {words.data(), 0, 0};
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      view.Format = DXGI_FORMAT_UNKNOWN;
      view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      view.Buffer.FirstElement = 0;
      view.Buffer.NumElements = static_cast<UINT>(words.size() / 4u);
      return SUCCEEDED(dev->CreateBuffer(&desc, &initial, &buffer)) &&
             SUCCEEDED(dev->CreateShaderResourceView(buffer.Get(), &view, &srv));
    }

    struct scene_plan_entry {
      std::size_t start_sequence = 0;
      std::size_t end_sequence_exclusive = 0;
    };

    bool read_scene_cache_contract(const fs::path &directory,
                                   scene_cache_metadata &metadata,
                                   std::string &error) {
      try {
        std::ifstream stream(directory / "scene_cache_contract.json");
        nlohmann::json value;
        if (!stream || !(stream >> value) || !value.is_object() ||
            value.at("schema").get<unsigned>() != offline_sbs::scene_cache_contract_schema ||
            (value.at("status").get<std::string>() != "running" &&
             value.at("status").get<std::string>() != "complete") ||
            !value.at("processed_count").is_number_integer()) {
          error = "scene cache contract is not a running/complete schema-3 contract";
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
        const auto &state = value.at("state");
        const auto &render = value.at("render_config");
        const auto &packed = value.at("packed_sbs");
        metadata.source_width = source.at("width").get<UINT>();
        metadata.source_height = source.at("height").get<UINT>();
        metadata.input_frame_format = source.at("frame_format").get<std::string>();
        metadata.input_texture_format = source.at("texture_format").get<std::string>();
        metadata.input_color_space = source.at("color_space").get<std::string>();
        metadata.depth_width = depth.at("width").get<UINT>();
        metadata.depth_height = depth.at("height").get<UINT>();
        metadata.model_name = render.at("model").get<std::string>();
        metadata.model_url = render.at("model_url").get<std::string>();
        metadata.pop_strength = render.at("pop_strength").get<double>();
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

        const std::uint64_t depth_bytes =
          static_cast<std::uint64_t>(metadata.depth_width) *
          metadata.depth_height * sizeof(float);
        if (!metadata.source_width || !metadata.source_height ||
            !metadata.depth_width || !metadata.depth_height ||
            depth.at("dxgi_format").get<std::string>() != "R32_FLOAT" ||
            depth.at("dtype").get<std::string>() != "float32-le" ||
            depth.at("layout").get<std::string>() != "row-major" ||
            depth.at("row_order").get<std::string>() != "top-down" ||
            depth.at("file_pattern").get<std::string>() !=
              "frame_%010d.depth.r32f" ||
            depth.at("bytes_per_frame").get<std::uint64_t>() != depth_bytes ||
            depth.at("semantics").get<std::string>() !=
              "depth-coordinate-v2-signed-final-parallax-source-u" ||
            depth.at("source_u_limit").get<double>() !=
              static_cast<double>(
                models::depth_coordinate_v2::direct_container_limit) ||
            state.at("schema").get<int>() != 2 ||
            state.at("source").get<std::string>() !=
              "depth_coordinate_v2.ParallaxState[0..2]" ||
            state.at("contract_schema").get<std::uint32_t>() !=
              models::depth_coordinate_v2::contract_schema ||
            state.at("contract_tag").get<std::uint32_t>() !=
              models::depth_coordinate_v2::contract_tag ||
            state.at("word_count").get<std::size_t>() !=
              render_state_words_t {}.size() ||
            state.at("dtype").get<std::string>() != "uint32-le" ||
            state.at("layout").get<std::string>() != "raw-word-order" ||
            state.at("file_pattern").get<std::string>() !=
              "frame_%010d.state.u32" ||
            state.at("bytes_per_frame").get<std::size_t>() !=
              sizeof(render_state_words_t) ||
            render.at("renderer").get<std::string>() !=
              "depth-coordinate-v2-live-signed-parallax" ||
            render.at("producer_source_closure_sha256").get<std::string>() !=
              models::depth_coordinate_v2::shader_source_closure_sha256 ||
            render.at("renderer_source_closure_sha256").get<std::string>() !=
              models::host_sbs_shader_cache::
                parallax_v2_live_renderer_source_closure_sha256 ||
            !(metadata.pop_strength >= 0.25 &&
              metadata.pop_strength <= 2.0) ||
            metadata.depth_reuse_interval != 1 ||
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
      const std::uint64_t depth_size =
        static_cast<std::uint64_t>(metadata.depth_width) *
        metadata.depth_height * sizeof(float);
      std::error_code ec;
      for (std::size_t sequence = start_sequence;
           sequence < end_sequence_exclusive;
           ++sequence) {
        const std::string stem = scene_cache_frame_stem(sequence);
        const fs::path depth = directory / (stem + ".depth.r32f");
        const fs::path state = directory / (stem + ".state.u32");
        if (!fs::is_regular_file(depth, ec) || ec ||
            fs::file_size(depth, ec) != depth_size || ec ||
            !fs::is_regular_file(state, ec) || ec ||
            fs::file_size(state, ec) != sizeof(render_state_words_t) || ec) {
          error = "scene cache is missing or has a wrong-sized pair for global sequence " +
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
            value.at("schema").get<int>() != 2 ||
            value.at("version").get<std::string>() != "scene-plan-v2" ||
            value.at("cache_contract_schema").get<unsigned>() !=
              offline_sbs::scene_cache_contract_schema ||
            !value.at("scenes").is_array() ||
            value.at("scenes").size() != 1u) {
          error = "scene plan is not a scene-plan-v2 schema-2 document";
          return false;
        }
        for (const auto &scene : value.at("scenes")) {
          if (!scene.is_object() ||
              !scene.at("start_sequence").is_number_integer() ||
              !scene.at("end_sequence_exclusive").is_number_integer()) {
            error = "scene plan contains a malformed scene";
            return false;
          }
          const auto start = scene.at("start_sequence").get<std::int64_t>();
          const auto end =
            scene.at("end_sequence_exclusive").get<std::int64_t>();
          if (start <= 0 || end <= start ||
              static_cast<std::uint64_t>(end - 1) >
                follow_max_sequence ||
              static_cast<std::uint64_t>(start) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
              static_cast<std::uint64_t>(end) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            error = "scene plan has a gap, overlap, or out-of-range sequence";
            return false;
          }
          entries.push_back({
            static_cast<std::size_t>(start),
            static_cast<std::size_t>(end),
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

    bool write_adaptive_state_header(
      std::ostream &out,
      const std::string_view model_name,
      const double pop_strength,
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
          {"pop_strength", pop_strength},
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
      const adaptive_state_words_t &words
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
      const float hard_cut_pulse = scalar(word_e::hard_cut_pulse);
      if (hard_cut_pulse != 0.0f && hard_cut_pulse != 1.0f) {
        return false;
      }
      out << "{\"record\":\"frame\",\"frame_id\":" << json_string(frame_id)
          << ",\"source_index\":" << source_index
          << ",\"depth_updated\":" << (depth_updated ? "true" : "false")
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
          << (hard_cut_pulse > 0.5f ? "true" : "false")
          << ",\"hard_cut_count\":"
          << words[sbs_adaptive_state::index(word_e::hard_cut_count)]
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

    bool write_cut_state_trace(
      const fs::path &path,
      const std::vector<cut_state_record> &records,
      const sbs_adaptive_state::compact_cut_trace_contract_t &trace_contract
    ) {
      std::ofstream out(path);
      if (!out) {
        return false;
      }
      out.imbue(std::locale::classic());
      out << std::setprecision(std::numeric_limits<float>::max_digits10);
      out << "{\n"
          << "  \"schema\": " << trace_contract.schema << ",\n"
          << "  \"source\": " << json_string(trace_contract.source) << ",\n"
          << "  \"capture\": " << json_string(trace_contract.capture) << ",\n"
          << "  \"fields\": [";
      for (std::size_t index = 0;
           index < sbs_adaptive_state::compact_cut_trace_fields.size();
           ++index) {
        if (index != 0u) {
          out << ", ";
        }
        out << json_string(sbs_adaptive_state::compact_cut_trace_fields[index]);
      }
      out << "],\n"
          << "  \"frames\": [\n";
      for (size_t index = 0; index < records.size(); ++index) {
        const auto &record = records[index];
        out << "    {\"frame_id\": " << json_string(record.frame_id) << ", \"values\": [";
        for (std::size_t value_index = 0;
             value_index < sbs_adaptive_state::compact_cut_trace_words.size();
             ++value_index) {
          if (value_index != 0) {
            out << ", ";
          }
          const auto word = sbs_adaptive_state::compact_cut_trace_words[value_index];
          const auto word_index = sbs_adaptive_state::index(word);
          const auto &descriptor = sbs_adaptive_state::fields[word_index];
          if (descriptor.gpu_encoding ==
              sbs_adaptive_state::gpu_encoding_e::uint_bits) {
            out << record.words[word_index];
          } else {
            out << std::bit_cast<float>(record.words[word_index]);
          }
        }
        out << "]}"
            << (index + 1 == records.size() ? "\n" : ",\n");
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
      std::string frames, out, direct_parallax_root;
      std::string depth_coordinate_v2_manifest, capabilities;
      std::string scene_cache, render_cache, scene_plan;
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
      bool simulate_hdr = false;  // decode sRGB frames into linear scRGB FP16 and use HDR paths
      double hdr_scale = 4.0;  // scRGB multiplier after sRGB EOTF (4.0 = 320-nit diffuse white)
      int max_width = 0;  // 0 -> use config max_encode_width
      int limit = 0;  // 0 -> all
      int output_every = 1;  // process every input for temporal state; dump only every Nth
      int cuda_graph = -1;  // -1 = conf, 0 = ordinary enqueue, 1 = CUDA graph replay
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
        } else if (a == "--eye-w") {
          o.eye_w = std::stoi(next("--eye-w"));
        } else if (a == "--eye-h") {
          o.eye_h = std::stoi(next("--eye-h"));
        } else if (a == "--output-scale") {
          o.output_scale = std::stod(next("--output-scale"));
        } else if (a == "--pop-strength") {
          o.pop_strength = std::stod(next("--pop-strength"));
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
        } else if (a == "--direct-parallax-root") {
          o.direct_parallax_root = next("--direct-parallax-root");
        } else if (a == "--depth-coordinate-v2-manifest") {
          o.depth_coordinate_v2_manifest = next("--depth-coordinate-v2-manifest");
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
      const unsigned direct_geometry_input_count =
        (!o.direct_parallax_root.empty() ? 1u : 0u) +
        (!o.depth_coordinate_v2_manifest.empty() ? 1u : 0u);
      if (direct_geometry_input_count > 0u) {
        if (o.artifacts != artifact_mode_e::evaluation || o.follow ||
            !o.scene_cache.empty() || !o.render_cache.empty() ||
            o.output_every != 1) {
          BOOST_LOG(error)
            << "sbs-bench: direct/v2 GPU replay requires evaluation artifacts, output "
               "cadence 1, and forbids follow and caches";
          return false;
        }
        if (direct_geometry_input_count > 1u) {
          BOOST_LOG(error)
            << "sbs-bench: --direct-parallax-root and --depth-coordinate-v2-manifest "
               "are mutually exclusive";
          return false;
        }
      }
      if (!o.render_cache.empty()) {
        if (o.artifacts != artifact_mode_e::conversion ||
            o.scene_plan.empty()) {
          BOOST_LOG(error)
            << "sbs-bench: --render-cache requires conversion artifacts and --scene-plan";
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
        if (o.limit != 0 || o.output_every != 1) {
          BOOST_LOG(error) << "sbs-bench: --follow forbids --limit and --output-every != 1";
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
      if (!(o.hdr_scale > 0.0 && o.hdr_scale <= 64.0)) {
        BOOST_LOG(error) << "sbs-bench: --hdr-scale must be greater than 0 and at most 64";
        return false;
      }
      if (o.artifacts == artifact_mode_e::conversion && o.output_every != 1) {
        BOOST_LOG(error) << "sbs-bench: conversion artifacts require --output-every 1";
        return false;
      }
      if (!o.direct_parallax_root.empty() && !fs::is_directory(o.direct_parallax_root)) {
        BOOST_LOG(error) << "sbs-bench: --direct-parallax-root is not a directory";
        return false;
      }
      if (!o.depth_coordinate_v2_manifest.empty() &&
          !fs::is_regular_file(o.depth_coordinate_v2_manifest)) {
        BOOST_LOG(error) << "sbs-bench: --depth-coordinate-v2-manifest is not a file";
        return false;
      }
      return true;
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
          {"adaptive_state_contract_tag", sbs_adaptive_state::cut_contract_tag},
          {"adaptive_state_contract_canonical_sha256",
           sbs_adaptive_state::contract_canonical_sha256},
          {"adaptive_analysis_flag_bits", std::move(adaptive_analysis_flag_bits)},
          {"scene_cache_contract_schema", offline_sbs::scene_cache_contract_schema},
          {"scene_cache_packed_sbs_contract", true},
          {"renderer", "depth-coordinate-v2-live-signed-parallax"},
          {"parallax_v2_contract_schema",
           models::depth_coordinate_v2::contract_schema},
          {"scene_cache_depth", {
            {"dtype", "float32-le"},
            {"layout", "row-major"},
            {"dxgi_format", "R32_FLOAT"},
            {"semantics", "depth-coordinate-v2-signed-final-parallax-source-u"},
          }},
          {"scene_cache_state", {
            {"schema", 2},
            {"word_count", render_state_words_t {}.size()},
            {"dtype", "uint32-le"},
            {"source", "depth_coordinate_v2.ParallaxState[0..2]"},
          }},
          {"scene_plan", {
            {"schema", 2},
            {"version", "scene-plan-v2"},
            {"one_scene_per_replay", true},
            {"boundary_only", true},
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
    if (o.cuda_graph >= 0) {
      sbs_cfg.cuda_graph = (o.cuda_graph != 0);
    }
    config::sunshine.diagnostics_enabled = true;  // benchmark processes always measure
    sbs_perf::set_enabled(true);
    sbs_perf::reset();
    const auto model = video::host_sbs_v2_depth_model();
    const bool replay_mode = !o.render_cache.empty();
    // Direct replay supplies an authenticated, horizontally contractive final field plus an
    // independent canonical order. Native V2 sequence replay runs
    // the production seven-shader coordinate pipeline and feeds its exact final field through the
    // same fixed-point renderer contract.
    const bool depth_coordinate_v2_gpu_mode =
      !o.depth_coordinate_v2_manifest.empty();
    const bool external_direct_parallax_mode = !o.direct_parallax_root.empty();
    const bool direct_parallax_mode =
      external_direct_parallax_mode || depth_coordinate_v2_gpu_mode;
    const auto &cut_state_trace_contract = depth_coordinate_v2_gpu_mode ?
      sbs_adaptive_state::gpu_replay_cut_trace_contract :
      sbs_adaptive_state::production_cut_trace_contract;
    const bool whole_clip_mode = o.artifacts != artifact_mode_e::evaluation;
    const bool writes_adaptive_state = whole_clip_mode && !replay_mode;
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
    // The V2 camera and cut history are stateful per frame, so depth cadence is always one
    // inference per source frame; the replay cache contract validates the same interval.
    int effective_depth_every = 1;
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
      // Schema-3 caches carry the complete V2 geometry and authenticated renderer identity. The
      // replay child only needs the recorded pop for provenance plus the output geometry. No
      // per-scene geometry knobs exist
      // to restore, and the cached field is the sole position authority.
      if (replay_cache_metadata.model_name != model.name ||
          replay_cache_metadata.model_url != model.url) {
        BOOST_LOG(error)
          << "sbs-bench: render cache model is not the authenticated Host SBS V2 model";
        return 6;
      }
      sbs_cfg.pop_strength = replay_cache_metadata.pop_strength;
      o.eye_w = replay_cache_metadata.eye_width;
      o.eye_h = replay_cache_metadata.eye_height;
      o.output_scale = replay_cache_metadata.output_scale;
      o.max_width = replay_cache_metadata.max_output_width;
      effective_depth_every = replay_cache_metadata.depth_reuse_interval;
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
    const fs::path adaptive_state_header_path =
      fs::path(o.out) / "adaptive_state_header.json";
    const fs::path adaptive_state_frame_path =
      fs::path(o.out) / "adaptive_state_frame.json";
    if (writes_adaptive_state) {
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
                  sbs_cfg.pop_strength,
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
              sbs_cfg.pop_strength,
              effective_depth_every
            )) {
          BOOST_LOG(error)
            << "sbs-bench: failed writing adaptive_state.jsonl header";
          return 6;
        }
      }
    }

    BOOST_LOG(info) << "sbs-bench: "
                    << (o.follow ? "following producer frames" :
                                   std::to_string(frames.size()) + " frames")
                    << ", model '" << model.name
                    << "', eye " << (o.eye_w > 0 ? std::to_string(o.eye_w) : "auto") << 'x'
                    << (o.eye_h > 0 ? std::to_string(o.eye_h) : "auto")
                    << ", depth_step current-once"
                    << (replay_mode ? ", cache replay (TensorRT skipped)" : "")
                    << (depth_coordinate_v2_gpu_mode ?
                          ", native depth-coordinate-v2 final-field GPU sequence replay" :
                          (external_direct_parallax_mode ?
                             ", external direct final-parallax replay" : ""))
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

    // GPU timing is an evaluation diagnostic. Normal conversion must not Flush and wait on a
    // timestamp fence every frame; its staging/encoder pipeline owns the required synchronization.
    ComPtr<ID3D11Query> warp_disjoint, warp_start, warp_end;
    if (o.artifacts == artifact_mode_e::evaluation) {
      D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
      if (FAILED(dev->CreateQuery(&qd, &warp_disjoint))) {
        warp_disjoint.Reset();
      }
      qd.Query = D3D11_QUERY_TIMESTAMP;
      if (FAILED(dev->CreateQuery(&qd, &warp_start))) {
        warp_start.Reset();
      }
      if (FAILED(dev->CreateQuery(&qd, &warp_end))) {
        warp_end.Reset();
      }
    }

    // External final-field replay has a deliberately separate renderer. All native analysis and
    // scene-cache replay use the exact production V2 renderer; diagnostic entry points remain in
    // their dump-only shader so they cannot grow the live shader root.
    const auto *diagnostic_shader = direct_parallax_mode ?
      SUNSHINE_SHADERS_DIR "/sbs_direct_replay_ps.hlsl" :
      SUNSHINE_SHADERS_DIR "/sbs_reprojection_v2_diagnostics_ps.hlsl";
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, mask_ps, mapping_ps;
    if (o.artifacts != artifact_mode_e::adaptive) {
      if (direct_parallax_mode) {
        auto vs_blob = compile(
          SUNSHINE_SHADERS_DIR "/sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0"
        );
        auto ps_blob = compile(
          SUNSHINE_SHADERS_DIR "/sbs_direct_replay_ps.hlsl", "main_ps", "ps_5_0"
        );
        if (!vs_blob || !ps_blob ||
            FAILED(dev->CreateVertexShader(
              vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs)) ||
            FAILED(dev->CreatePixelShader(
              ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &ps))) {
          BOOST_LOG(error) << "sbs-bench: direct replay shader creation failed";
          return 6;
        }
      } else {
        const auto live_sources =
          models::host_sbs_shader_cache::snapshot_sources(
            fs::path {SUNSHINE_SHADERS_DIR},
            models::host_sbs_shader_cache::parallax_v2_live_renderer_specs
          );
        const auto closure =
          models::host_sbs_shader_cache::source_closure_sha256(live_sources);
        if (!live_sources ||
            closure != models::host_sbs_shader_cache::
                         parallax_v2_live_renderer_source_closure_sha256) {
          BOOST_LOG(error)
            << "sbs-bench: production V2 renderer source closure is unauthenticated";
          return 6;
        }
        const auto vs_bytecode = models::host_sbs_shader_cache::get(
          live_sources, models::host_sbs_shader_cache::sbs_reprojection_vertex
        );
        const auto ps_bytecode = models::host_sbs_shader_cache::get(
          live_sources, models::host_sbs_shader_cache::parallax_v2_live_renderer
        );
        if (!vs_bytecode || !ps_bytecode ||
            FAILED(dev->CreateVertexShader(
              vs_bytecode->data(), vs_bytecode->size(), nullptr, &vs)) ||
            FAILED(dev->CreatePixelShader(
              ps_bytecode->data(), ps_bytecode->size(), nullptr, &ps))) {
          BOOST_LOG(error) << "sbs-bench: authenticated V2 renderer creation failed";
          return 6;
        }
      }
    }
    if (o.artifacts == artifact_mode_e::evaluation) {
      auto mask_ps_blob = compile(diagnostic_shader, "mask_ps", "ps_5_0");
      auto mapping_ps_blob = compile(diagnostic_shader, "mapping_ps", "ps_5_0");
      if (!mask_ps_blob || !mapping_ps_blob ||
          FAILED(dev->CreatePixelShader(
            mask_ps_blob->GetBufferPointer(), mask_ps_blob->GetBufferSize(), nullptr, &mask_ps)) ||
          FAILED(dev->CreatePixelShader(
            mapping_ps_blob->GetBufferPointer(), mapping_ps_blob->GetBufferSize(), nullptr,
            &mapping_ps)) ||
          !mask_ps || !mapping_ps) {
        BOOST_LOG(error) << "sbs-bench: diagnostic shader creation failed";
        return 6;
      }
    }
    if (o.artifacts != artifact_mode_e::adaptive && (!vs || !ps)) {
      BOOST_LOG(error) << "sbs-bench: render shaders are unavailable";
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
    ComPtr<ID3D11Buffer> repro_cb;

    std::unique_ptr<depth_coordinate_v2_gpu_replay> depth_coordinate_v2_gpu;
    if (depth_coordinate_v2_gpu_mode) {
      std::string gpu_replay_error;
      depth_coordinate_v2_gpu = depth_coordinate_v2_gpu_replay::create(
        dev.Get(),
        ctx.Get(),
        o.depth_coordinate_v2_manifest,
        gpu_replay_error
      );
      if (!depth_coordinate_v2_gpu ||
          depth_coordinate_v2_gpu->frame_count() != frames.size() ||
          depth_coordinate_v2_gpu->model_name() != model.name) {
        BOOST_LOG(error)
          << "sbs-bench: cannot initialize native v2 GPU sequence replay"
          << (gpu_replay_error.empty() ? "" : ": " + gpu_replay_error);
        return 6;
      }
    }

    // ---- estimator ----
    // Cache and raw-v2 replay are deliberately pure D3D render passes: do not construct the
    // TensorRT estimator. The GPU replay path uploads authenticated model output directly into
    // the production V2 compute shaders. External final-parallax replay still runs the fixed
    // production estimator so its cut-state and resolved-shape evidence remain representative.
    std::unique_ptr<models::video_depth_estimator> estimator;
    if (!replay_mode && !depth_coordinate_v2_gpu_mode) {
      estimator = std::make_unique<models::video_depth_estimator>(
        dev,
        ctx,
        fs::path(SUNSHINE_ASSETS_DIR),
        sbs_cfg,
        model
      );
      if (estimator->has_terminal_failure()) {
        BOOST_LOG(error)
          << "sbs-bench: the production V2 estimator failed to initialize; refusing to "
             "emit output without authenticated cut-state and resolved-shape evidence";
        return 5;
      }
    }

    // Per-run state built lazily on the first frame (once we know the input size).
    ComPtr<ID3D11Texture2D> sbs_tex, sbs_stage;
    ComPtr<ID3D11RenderTargetView> sbs_rtv;
    ComPtr<ID3D11ShaderResourceView> sbs_srv;
    ComPtr<ID3D11Texture2D> warp_mask_tex, warp_mask_stage;
    ComPtr<ID3D11RenderTargetView> warp_mask_rtv;
    ComPtr<ID3D11Texture2D> warp_mapping_tex, warp_mapping_stage;
    ComPtr<ID3D11RenderTargetView> warp_mapping_rtv;
    D3D11_VIEWPORT vp = {};
    UINT sbs_w = 0, sbs_h = 0;
    ComPtr<ID3D11Texture2D> ema_mask_stage;
    ComPtr<ID3D11Buffer> raw_depth_stage;
    ComPtr<ID3D11Texture2D> structure_field_stage;
    ComPtr<ID3D11Buffer> cut_state_stage;
    ComPtr<ID3D11Texture2D> scene_cache_depth_stage;
    ComPtr<ID3D11Buffer> scene_cache_state_stage;
    // Evaluation keeps the compact cut-state trace used by the bounded clip scorer.
    // Whole-clip follow/conversion already emits the superset adaptive-state
    // transport (JSONL for standalone tooling, atomic latest-record snapshots for the native
    // worker), so retaining a second per-frame vector there would make memory scale with clip
    // duration for no additional evidence.
    std::vector<cut_state_record> cut_state_records;
    if (!whole_clip_mode) {
      cut_state_records.reserve(frames.size());
    }
    bool raw_shape_written = false;
    bool warp_mapping_shape_written = false;
    float hdr_output_min = std::numeric_limits<float>::infinity();
    float hdr_output_max = -std::numeric_limits<float>::infinity();
    uint64_t hdr_nonfinite = 0;

    int written = 0;
    models::estimate_result est;
    bool cuda_graph_captured = false;
    std::size_t tensorrt_enqueue_count = 0;
    bool scene_cache_contract_started = false;
    scene_cache_metadata scene_cache_metadata_value;
    std::size_t scene_plan_index = 0;
    UINT source_width = 0;
    UINT source_height = 0;
    // Match the depth-override clip layout so a root may hold several clips without filename
    // collisions. External direct replay reads parallax_<id> plus order_<id>.
    const fs::path direct_geometry_root = fs::path(o.direct_parallax_root);
    const fs::path direct_parallax_dir = direct_geometry_root.empty() ? fs::path() :
                                           direct_geometry_root /
                                             fs::path(o.frames).filename();
    size_t applied_direct_parallax_frames = 0;
    nlohmann::ordered_json direct_parallax_fields = nlohmann::ordered_json::array();
    if (!direct_parallax_dir.empty()) {
      if (!fs::is_directory(direct_parallax_dir)) {
        BOOST_LOG(error) << "sbs-bench: direct-parallax clip directory is missing: "
                         << direct_parallax_dir;
        return 7;
      }
      size_t actual_direct_parallax_frames = 0;
      size_t actual_direct_order_frames = 0;
      for (const auto &entry : fs::directory_iterator(direct_parallax_dir)) {
        const auto filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.starts_with("parallax_") &&
            entry.path().extension() == ".f32") {
          ++actual_direct_parallax_frames;
        }
        if (entry.is_regular_file() && filename.starts_with("order_") &&
            entry.path().extension() == ".f32") {
          ++actual_direct_order_frames;
        }
      }
        if (actual_direct_parallax_frames != frames.size() ||
            actual_direct_order_frames != frames.size()) {
          BOOST_LOG(error) << "sbs-bench: expected " << frames.size()
                         << " direct-parallax and order fields in "
                         << direct_parallax_dir
                         << ", found displacement=" << actual_direct_parallax_frames
                         << " order=" << actual_direct_order_frames;
        return 7;
      }
    }
    std::size_t processed_frame_count = 0;
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
      std::string exact_source_snapshot;
      std::string exact_source_sha256;
      bool loaded = false;
      if (depth_coordinate_v2_gpu_mode) {
        // Exact replay hashes and decodes one immutable byte snapshot. Reopening the path after
        // WIC upload would let an atomic replacement authenticate different pixels than the SRV.
        loaded = read_file_snapshot(current_frame, exact_source_snapshot);
        if (loaded) {
          exact_source_sha256 = sha256_hex(exact_source_snapshot);
          loaded = pfm_input ?
                     load_pfm(std::string_view(exact_source_snapshot), hdr_img) :
                     load_png(std::string_view(exact_source_snapshot), img);
        }
      } else {
        loaded = pfm_input ?
                   load_pfm(current_frame, hdr_img) :
                   load_png(current_frame, img);
      }
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
        return 6;
      }
      ComPtr<ID3D11ShaderResourceView> in_srv;
      if (FAILED(dev->CreateShaderResourceView(in_tex.Get(), nullptr, &in_srv))) {
        BOOST_LOG(error) << "sbs-bench: input SRV creation failed";
        return 6;
      }

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
        // Exact production b2 layout. Bench/replay stays full-frame, so ROI is disabled and the
        // second register carries the canonical full-source rectangle.
        float repro_params[8] = {
          content_scale_x,
          content_scale_y,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          1.0f,
          1.0f,
        };
        repro_cb = const_buffer(dev.Get(), repro_params);
        if (!repro_cb) {
          BOOST_LOG(error) << "sbs-bench: reprojection constant-buffer creation failed";
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
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &sbs_tex)) ||
            FAILED(dev->CreateRenderTargetView(sbs_tex.Get(), nullptr, &sbs_rtv)) ||
            FAILED(dev->CreateShaderResourceView(sbs_tex.Get(), nullptr, &sbs_srv))) {
          BOOST_LOG(error) << "sbs-bench: packed SBS target creation failed";
          return 6;
        }
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
        }
        D3D11_TEXTURE2D_DESC sd2 = td;
        sd2.Usage = D3D11_USAGE_STAGING;
        sd2.BindFlags = 0;
        sd2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev->CreateTexture2D(&sd2, nullptr, &sbs_stage))) {
          BOOST_LOG(error) << "sbs-bench: packed SBS staging texture creation failed";
          return 6;
        }
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
            << "  \"validity\": {\"content\": \"derive from content_scale_x/content_scale_y and packed output coordinate\", \"boundary_extrapolation\": \"warp_mask_<frame-id>.png red == 1 where the inverse sample left [0,1] source U\"},\n"
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
      // Depth cadence is fixed at one inference per source frame (the replay cache contract
      // validates the same interval), so every frame is a depth update.
      const bool depth_updated = true;
      adaptive_state_words_t words {};
      ComPtr<ID3D11Texture2D> cached_depth_texture;
      ComPtr<ID3D11Buffer> cached_state_buffer;
      depth_coordinate_v2_gpu_frame v2_gpu_frame;
      if (replay_mode) {
        const std::size_t sequence = global_sequence;
        const std::string cache_stem = scene_cache_frame_stem(sequence);
        render_state_words_t render_words {};
        std::vector<float> cached_parallax_values;
        if (!read_exact_file(
              fs::path(o.render_cache) / (cache_stem + ".state.u32"),
              render_words.data(),
              sizeof(render_words)
            ) ||
            !create_cached_float_srv(
              dev.Get(),
              fs::path(o.render_cache) / (cache_stem + ".depth.r32f"),
              replay_cache_metadata.depth_width,
              replay_cache_metadata.depth_height,
              cached_depth_texture,
              est.shadow_final_parallax,
              /* require_unit_range */ false,
              nullptr,
              nullptr,
              nullptr,
              &cached_parallax_values
            )) {
          BOOST_LOG(error) << "sbs-bench: invalid cached parallax/state pair for global sequence "
                           << sequence;
          return 6;
        }
        // The cached field is the signed one-eye source-U displacement; enforce the hard
        // container bound the authenticated producer guarantees before rendering with it.
        if (!std::all_of(
              cached_parallax_values.begin(),
              cached_parallax_values.end(),
              [](float value) {
                return std::fabs(value) <=
                       models::depth_coordinate_v2::direct_container_limit;
              }
            )) {
          BOOST_LOG(error)
            << "sbs-bench: cached V2 parallax exceeds the source-U container for sequence "
            << sequence;
          return 6;
        }
        const float cached_parallax_measured_slope =
          measured_horizontal_source_u_slope(
            cached_parallax_values,
            replay_cache_metadata.depth_width,
            replay_cache_metadata.depth_height,
            1.0
          );
        if (cached_parallax_measured_slope >
            direct_parallax_max_horizontal_slope +
              direct_parallax_slope_tolerance) {
          BOOST_LOG(error)
            << "sbs-bench: cached V2 parallax violates the horizontal slope contract for "
               "sequence "
            << sequence << " (" << cached_parallax_measured_slope << " > "
            << direct_parallax_max_horizontal_slope << ')';
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
        if (!create_cached_v2_state_srv(
              dev.Get(),
              render_words,
              cached_state_buffer,
              est.shadow_state
            )) {
          BOOST_LOG(error) << "sbs-bench: cannot create cached ParallaxState for sequence "
                           << sequence;
          return 6;
        }
        est.raw_width = static_cast<int>(replay_cache_metadata.depth_width);
        est.raw_height = static_cast<int>(replay_cache_metadata.depth_height);
      } else if (depth_coordinate_v2_gpu_mode) {
        std::string gpu_replay_error;
        if (!depth_coordinate_v2_gpu->dispatch(
              fi,
              output_id,
              in_srv.Get(),
              static_cast<std::uint32_t>(input_color),
              o.simulate_hdr ? static_cast<float>(o.hdr_scale) : 1.0f,
              exact_source_sha256,
              v2_gpu_frame,
              gpu_replay_error
            )) {
          BOOST_LOG(error) << "sbs-bench: native v2 GPU replay failed for frame "
                           << output_id << ": " << gpu_replay_error;
          return 7;
        }
        est.depth = v2_gpu_frame.encoded_final_parallax;
        est.cut_state = v2_gpu_frame.cut_state;
        est.raw_model_depth = v2_gpu_frame.raw_depth;
        est.raw_width = static_cast<int>(depth_coordinate_v2_gpu->width());
        est.raw_height = static_cast<int>(depth_coordinate_v2_gpu->height());
      } else {
        // The V2 authentication contract rejects frame id 0, so the estimator is always fed
        // the 1-based global sequence.
        const auto estimator_frame_id =
          static_cast<std::uint64_t>(global_sequence);
        estimator->estimate_depth(
          in_srv.Get(),
          input_color,
          estimator_frame_id
        );
        est = estimator->finish_pending_depth_for_evaluation(input_color);
        if (whole_clip_mode &&
            (!est.completed_frame_valid ||
             est.completed_frame_id != estimator_frame_id)) {
          BOOST_LOG(error) << "sbs-bench: scheduled depth update for source frame "
                           << output_id << " did not complete the expected current-frame oracle "
                           << "(valid=" << (est.completed_frame_valid ? "true" : "false")
                           << ", completed_frame_id=" << est.completed_frame_id
                           << ", expected=" << estimator_frame_id << ')';
          return 6;
        }
        // Never degrade to silently flat output: every frame must publish the complete
        // authenticated V2 geometry chain.
        if (estimator->has_terminal_failure() ||
            !models::parallax_v2_result_is_authenticated(est)) {
          BOOST_LOG(error)
            << "sbs-bench: frame " << output_id
            << " did not publish authenticated V2 geometry (terminal="
            << (estimator->has_terminal_failure() ? "true" : "false")
            << "); aborting the run";
          return 6;
        }
        cuda_graph_captured = cuda_graph_captured || est.cuda_graph_active;
        ++tensorrt_enqueue_count;
      }

      // Direct geometry replay consumes only the encoded slope-limited final field. Canonical
      // order remains independent finite, unbounded semantic evidence; it does not control the
      // production fixed-point renderer.
      ComPtr<ID3D11Texture2D> direct_parallax_texture;
      ComPtr<ID3D11ShaderResourceView> direct_parallax_srv;
      ComPtr<ID3D11Texture2D> direct_order_texture;
      ComPtr<ID3D11ShaderResourceView> direct_order_srv;
      ComPtr<ID3D11Buffer> direct_parallax_cb;
      float direct_parallax_encoded_min = 0.5f;
      float direct_parallax_encoded_max = 0.5f;
      float direct_parallax_max_abs = 0.0f;
      std::string direct_parallax_sha256;
      float direct_order_min = 0.0f;
      float direct_order_max = 0.0f;
      std::string direct_order_sha256;
      std::vector<float> direct_parallax_values;
      std::vector<float> direct_order_values;
      if (direct_parallax_mode) {
        D3D11_TEXTURE2D_DESC depth_desc {};
        fs::path field_path;
        if (depth_coordinate_v2_gpu_mode) {
          depth_desc.Width = depth_coordinate_v2_gpu->width();
          depth_desc.Height = depth_coordinate_v2_gpu->height();
          depth_desc.Format = DXGI_FORMAT_R32_FLOAT;
          direct_parallax_srv = v2_gpu_frame.encoded_final_parallax;
          direct_order_srv = v2_gpu_frame.canonical_order;
          direct_parallax_encoded_min = v2_gpu_frame.encoded_minimum;
          direct_parallax_encoded_max = v2_gpu_frame.encoded_maximum;
          direct_parallax_sha256 = v2_gpu_frame.parallax_sha256;
          direct_order_min = v2_gpu_frame.order_minimum;
          direct_order_max = v2_gpu_frame.order_maximum;
          direct_order_sha256 = v2_gpu_frame.order_sha256;
          direct_parallax_values = v2_gpu_frame.encoded_parallax_values;
          direct_order_values = v2_gpu_frame.canonical_values;
          field_path = fs::path(o.depth_coordinate_v2_manifest);
        } else {
          ComPtr<ID3D11Resource> depth_resource;
          ComPtr<ID3D11Texture2D> depth_texture;
          if (!est.depth ||
              (est.depth->GetResource(&depth_resource),
               FAILED(depth_resource.As(&depth_texture)))) {
            BOOST_LOG(error) << "sbs-bench: direct-parallax replay needs a valid depth texture";
            return 7;
          }
          depth_texture->GetDesc(&depth_desc);
          field_path = direct_parallax_dir / ("parallax_" + output_id + ".f32");
          const fs::path order_path =
            direct_parallax_dir / ("order_" + output_id + ".f32");
          if (depth_desc.Format != DXGI_FORMAT_R32_FLOAT ||
              !create_cached_float_srv(
                dev.Get(), field_path, depth_desc.Width, depth_desc.Height,
                direct_parallax_texture, direct_parallax_srv,
                true,
                &direct_parallax_encoded_min, &direct_parallax_encoded_max,
                &direct_parallax_sha256, &direct_parallax_values
              ) ||
              !create_cached_float_srv(
                dev.Get(), order_path, depth_desc.Width, depth_desc.Height,
                direct_order_texture, direct_order_srv,
                false, &direct_order_min, &direct_order_max, &direct_order_sha256,
                &direct_order_values
              )) {
            BOOST_LOG(error)
              << "sbs-bench: missing or invalid direct geometry fields " << field_path
              << " and " << order_path << " (expected finite float32-le encoded parallax [0,1]"
              << " and unbounded order at " << depth_desc.Width << 'x'
              << depth_desc.Height << ')';
            return 7;
          }
        }
        direct_parallax_max_abs = std::max(
            std::abs(
              (direct_parallax_encoded_min * 2.0f - 1.0f) *
              direct_parallax_source_u_limit
            ),
            std::abs(
              (direct_parallax_encoded_max * 2.0f - 1.0f) *
              direct_parallax_source_u_limit
            )
          );
        if (direct_parallax_max_abs > direct_parallax_source_u_limit + 2.0e-7f) {
          BOOST_LOG(error) << "sbs-bench: direct geometry field " << field_path
                           << " exceeds the generated source-U container ("
                           << direct_parallax_max_abs << " > "
                           << direct_parallax_source_u_limit << ')';
          return 7;
        }
        const float direct_parallax_measured_slope =
          measured_horizontal_source_u_slope(
            direct_parallax_values,
            depth_desc.Width,
            depth_desc.Height,
            2.0 * direct_parallax_source_u_limit
          );
        if (direct_parallax_measured_slope >
            direct_parallax_max_horizontal_slope + direct_parallax_slope_tolerance) {
          BOOST_LOG(error)
            << "sbs-bench: direct-parallax field " << field_path
            << " violates the generated horizontal slope contract ("
            << direct_parallax_measured_slope << " > "
            << direct_parallax_max_horizontal_slope << ')';
          return 7;
        }
        // b4 is harness-only. The final field's authenticated sub-unit source-U slope makes its
        // inverse contractive, so the renderer needs only the encoded container limit; the other
        // words remain reserved to preserve the 16-byte constant-buffer ABI.
        const float direct_params[4] = {
          direct_parallax_source_u_limit,
          0.0f,
          0.0f,
          0.0f,
        };
        direct_parallax_cb = const_buffer(dev.Get(), direct_params);
        if (!direct_parallax_cb) {
          BOOST_LOG(error) << "sbs-bench: cannot create direct-parallax constants";
          return 7;
        }
        direct_parallax_fields.push_back({
          {"frame_id", output_id},
          {"width", depth_desc.Width},
          {"height", depth_desc.Height},
          {"parallax_sha256", direct_parallax_sha256},
          {"maximum_absolute_source_u", direct_parallax_max_abs},
          {"order_sha256", direct_order_sha256},
          {"order_minimum", direct_order_min},
          {"order_maximum", direct_order_max},
        });
        ++applied_direct_parallax_frames;
      }

      // Export one state sample for every source frame, even when --output-every skips composite
      // artifacts. This makes accepted shot pulses directly testable without adding any
      // synchronization or readback to production.
      if (whole_clip_mode) {
        if (!replay_mode &&
            !read_adaptive_state(dev.Get(), ctx.Get(), est.cut_state.Get(),
                                 cut_state_stage, words)) {
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
            scene_cache_metadata_value.depth_width = depth_desc.Width;
            scene_cache_metadata_value.depth_height = depth_desc.Height;
            scene_cache_metadata_value.input_frame_format =
              discovered_input_frame_format;
            scene_cache_metadata_value.input_texture_format =
              discovered_input_texture_format;
            scene_cache_metadata_value.input_color_space =
              discovered_input_color_space;
            scene_cache_metadata_value.model_name = model.name;
            scene_cache_metadata_value.model_url = model.url;
            scene_cache_metadata_value.pop_strength = sbs_cfg.pop_strength;
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
          // Schema-3 cache pair: the signed V2 final field plus its ParallaxState words,
          // exactly what the replay child re-authenticates and binds to the warp.
          render_state_words_t v2_state_words {};
          if (!read_v2_state_words(
                dev.Get(),
                ctx.Get(),
                est.shadow_state.Get(),
                scene_cache_state_stage,
                v2_state_words
              ) ||
              !cache_depth_texture_atomically(
                dev.Get(),
                ctx.Get(),
                est.shadow_final_parallax.Get(),
                fs::path(o.scene_cache) / (cache_stem + ".depth.r32f"),
                scene_cache_depth_stage,
                scene_cache_metadata_value.depth_width,
                scene_cache_metadata_value.depth_height,
                /* signed_container */ true
              ) ||
              !cache_v2_state_atomically(
                v2_state_words,
                fs::path(o.scene_cache) / (cache_stem + ".state.u32")
              )) {
            BOOST_LOG(error) << "sbs-bench: cannot durably publish scene cache pair "
                             << sequence;
            return 6;
          }
        }
        if (!replay_mode) {
          const auto write_frame = [&](std::ostream &out) {
            return write_adaptive_state_frame(
              out,
              output_id,
              fi,
              depth_updated,
              words
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
        }
        if (!o.scene_cache.empty() &&
            !publish_scene_cache_contract(
              o.scene_cache,
              scene_cache_metadata_value,
              "running",
              fi + 1u
            )) {
          BOOST_LOG(error)
            << "sbs-bench: cannot acknowledge traced scene cache pair "
            << (fi + 1u);
          return 6;
        }
      } else {
        cut_state_record state_record;
        state_record.frame_id = output_id;
        if (!read_cut_state(dev.Get(), ctx.Get(), est.cut_state.Get(),
                            cut_state_stage, state_record.words)) {
          BOOST_LOG(error) << "sbs-bench: cannot read cut state for frame " << output_id;
          return 6;
        }
        cut_state_records.push_back(std::move(state_record));
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

      // Sampling output must never sample the depth/EMA/cut pipeline itself. Every source
      // frame above was inferred and consumed; only expensive composite/readback is skipped.
      if ((fi % (size_t) o.output_every) != 0) {
        continue;
      }

      // Composite (mirrors display_vram::convert()'s SBS block): probe reprojection.
      const auto comp_t0 = std::chrono::steady_clock::now();
      const bool time_warp = warp_disjoint && warp_start && warp_end;
      if (time_warp) {
        ctx->Begin(warp_disjoint.Get());
        ctx->End(warp_start.Get());
      }
      // V2-live analysis/conversion (and cache replay) bind the authenticated signed final
      // field plus ParallaxState; direct replay binds its uploaded encoded field with the
      // estimator's cut-compat state.
      ID3D11ShaderResourceView *warp_depth =
        direct_parallax_mode ? direct_parallax_srv.Get() :
                               est.shadow_final_parallax.Get();
      ID3D11ShaderResourceView *warp_state =
        direct_parallax_mode ? est.cut_state.Get() : est.shadow_state.Get();
      ctx->OMSetRenderTargets(1, sbs_rtv.GetAddressOf(), nullptr);
      ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      ctx->VSSetShader(vs.Get(), nullptr, 0);
      ctx->PSSetShader(ps.Get(), nullptr, 0);
      ctx->RSSetViewports(1, &vp);
      ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());

      ID3D11ShaderResourceView *srvs[] = {in_srv.Get(), warp_depth, warp_state};
      ctx->PSSetShaderResources(0, 3, srvs);
      ID3D11Buffer *cb = repro_cb.Get();
      ctx->PSSetConstantBuffers(2, 1, &cb);
      if (direct_parallax_cb) {
        ctx->PSSetConstantBuffers(4, 1, direct_parallax_cb.GetAddressOf());
      }
      ctx->Draw(3, 0);

      ID3D11RenderTargetView *null_rtv[] = {nullptr};
      ctx->OMSetRenderTargets(1, null_rtv, nullptr);
      ID3D11ShaderResourceView *null_srv[] = {nullptr, nullptr, nullptr, nullptr};
      ctx->PSSetShaderResources(0, 3, null_srv);
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
        // A contractive map has no internal holes; R marks finite-boundary extrapolation samples
        // that main_ps clamps to the nearest source column.
        ctx->OMSetRenderTargets(1, warp_mask_rtv.GetAddressOf(), nullptr);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(mask_ps.Get(), nullptr, 0);
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
        ID3D11ShaderResourceView *mask_srvs[] = {
          in_srv.Get(),
          warp_depth,
          warp_state
        };
        ctx->PSSetShaderResources(0, 3, mask_srvs);
        ctx->PSSetConstantBuffers(2, 1, &cb);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(0, 4, null_srv);

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
          warp_state
        };
        ctx->PSSetShaderResources(0, 3, mapping_srvs);
        ctx->PSSetConstantBuffers(2, 1, &cb);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(1, null_rtv, nullptr);
        ctx->PSSetShaderResources(0, 3, null_srv);

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

      // Preserve real HDR conversion as linear scRGB float interchange. SDR and the
      // synthetic-HDR compatibility mode retain their PNG artifact behavior.
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
          if (direct_parallax_mode) {
            // Displacement and semantic depth intentionally remain distinct. The limiter may lower
            // a foreground texel's final parallax without changing the canonical semantic order,
            // so scoring the conditioned displacement as "depth" would let safety conditioning
            // redefine the evidence used to judge itself.
            // Republish the exact bytes uploaded to both SRVs: canonical order is the evaluator's
            // unbounded depth authority, while encoded final parallax remains a separate geometry
            // artifact. Their hashes are authenticated by the manifest written below.
            const fs::path depth_path =
              fs::path(o.out) / ("depth_" + output_id + ".f32");
            const fs::path parallax_path =
              fs::path(o.out) / ("parallax_" + output_id + ".f32");
            if (!publish_float_field(depth_path, direct_order_values) ||
                !publish_float_field(parallax_path, direct_parallax_values)) {
              BOOST_LOG(error)
                << "sbs-bench: cannot publish authenticated direct geometry artifacts for frame "
                << output_id;
              return 7;
            }
          } else {
            // The estimator's normalized depth is a private cut field with no geometry
            // authority; scoring it against GT depth would be meaningless. The scored depth
            // artifact is the authenticated raw model output, normalized per frame into the
            // same 16-bit grayscale container dump_depth historically wrote.
            char dname[64];
            snprintf(dname, sizeof(dname), "depth_%s.png", output_id.c_str());
            if (!est.raw_model_depth) {
              BOOST_LOG(error) << "sbs-bench: v2 evaluation frame " << output_id
                               << " published no authenticated raw model depth";
              return 6;
            }
            if (!dump_raw_model_depth_png(dev.Get(), ctx.Get(), est.raw_model_depth.Get(),
                                          est.raw_width, est.raw_height,
                                          fs::path(o.out) / dname, raw_depth_stage)) {
              BOOST_LOG(error) << "sbs-bench: failed writing " << dname;
              return 6;
            }
            // Structure-consistency evidence: the V2 candidate parallax the renderer is actually
            // shaped by, in the same per-frame min/max 16-bit container as the raw depth PNG.
            char sname[64];
            snprintf(sname, sizeof(sname), "structure_%s.png", output_id.c_str());
            if (!est.shadow_candidate_parallax) {
              BOOST_LOG(error) << "sbs-bench: v2 evaluation frame " << output_id
                               << " published no candidate-parallax structure field";
              return 6;
            }
            if (!dump_structure_field_png(dev.Get(), ctx.Get(),
                                          est.shadow_candidate_parallax.Get(),
                                          fs::path(o.out) / sname, structure_field_stage)) {
              BOOST_LOG(error) << "sbs-bench: failed writing " << sname;
              return 6;
            }

          }
          // The fixed live calibration keeps the edge-selective EMA enabled, so the moving-edge
          // snap mask remains a per-frame evaluation artifact.
          static_assert(config::host_sbs_v2_live_calibration::edge_change > 0.0);
          {
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
              shape << "{\n  \"schema\": 1,\n  \"width\": " << est.raw_width
                    << ",\n  \"height\": " << est.raw_height
                    << ",\n  \"dtype\": \"float32-le\",\n"
                       "  \"layout\": \"row-major\",\n"
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
    std::string direct_parallax_manifest_sha256;
    std::string depth_coordinate_v2_state_trace_sha256;
    if (depth_coordinate_v2_gpu_mode) {
      const fs::path state_trace_path =
        fs::path(o.out) / "depth_coordinate_v2_state_trace.json";
      std::string gpu_replay_error;
      if (!depth_coordinate_v2_gpu->write_state_trace(
            state_trace_path,
            gpu_replay_error
          ) ||
          (depth_coordinate_v2_state_trace_sha256 =
             sha256_file_hex(state_trace_path)).empty()) {
        BOOST_LOG(error) << "sbs-bench: cannot publish native v2 GPU state trace"
                         << (gpu_replay_error.empty() ? "" : ": " + gpu_replay_error);
        return 7;
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
    if (direct_parallax_mode &&
        applied_direct_parallax_frames != completed_frame_count) {
      BOOST_LOG(error) << "sbs-bench: applied " << applied_direct_parallax_frames
                       << " of " << completed_frame_count
                       << " expected direct-parallax fields";
      return 7;
    }
    if (direct_parallax_mode) {
      nlohmann::ordered_json manifest = {
        {"schema", direct_geometry_manifest_schema},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"stored_range", {0.0, 1.0}},
        {"zero_encoding", 0.5},
        {"decode_source_u_one_eye", std::string {
                                            models::depth_coordinate_v2::
                                              direct_parallax_decode_expression
        }},
        {"displacement_semantics", "signed-source-u-positive-nearward-not-order"},
        {"maximum_horizontal_source_u_slope", direct_parallax_max_horizontal_slope},
        {"renderer_inverse", "contractive-fixed-point-11-iterations-v1"},
        {"renderer_uses_order", false},
        {"order_dtype", "float32-le"},
        {"order_range", "finite-unbounded"},
        {"order_high_is_near", true},
        {"order_role", "diagnostic-semantic-depth-only-v1"},
        {"depth_artifact_semantics", "canonical-pre-limiter-order-float32-v1"},
        {"parallax_artifact_semantics", "encoded-conditioned-final-parallax-float32-v1"},
        {"fields", direct_parallax_fields},
      };
      const std::string serialized = manifest.dump(2) + "\n";
      direct_parallax_manifest_sha256 = sha256_hex(serialized);
      std::ofstream manifest_stream(
        fs::path(o.out) / "direct_parallax_manifest.json",
        std::ios::binary | std::ios::trunc
      );
      manifest_stream.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
      manifest_stream.flush();
      if (!manifest_stream.good()) {
        BOOST_LOG(error) << "sbs-bench: cannot write direct_parallax_manifest.json";
        return 7;
      }
    }
    if (o.artifacts == artifact_mode_e::evaluation && !warp_mapping_shape_written) {
      BOOST_LOG(error) << "sbs-bench: no warp mapping shape contract was written";
      return 8;
    }
    if (!whole_clip_mode) {
      if (cut_state_records.size() != completed_frame_count ||
          !write_cut_state_trace(
            fs::path(o.out) / "cut_state.json",
            cut_state_records,
            cut_state_trace_contract
          )) {
        BOOST_LOG(error) << "sbs-bench: failed writing complete cut_state.json";
        return 8;
      }
    }

    sbs_perf::dump_json((fs::path(o.out) / "sbs_perf.json").string());
    if (o.artifacts == artifact_mode_e::evaluation) {
      if (!direct_parallax_mode &&
          (!est.raw_model_provenance || est.raw_width <= 0 || est.raw_height <= 0)) {
        BOOST_LOG(error)
          << "sbs-bench: ordinary evaluation completed without exact raw-model provenance";
        return 8;
      }
      // Machine-readable execution contract. Evaluation must not scrape human log prose. The
      // Independent evaluation-harness schema 22 attests the V2-only configuration surface;
      // it is unrelated to Dump 3D schema 27 and DVC2 schema 46 (the direct-replay schema stays
      // pinned by its own validator).
      std::ofstream contract(fs::path(o.out) / "contract.json");
      if (contract) {
        contract << "{\n"
                 << "  \"schema\": "
                 << (direct_parallax_mode ? direct_geometry_contract_schema : 22u)
                 << ",\n"
                 << "  \"model\": " << json_string(model.name) << ",\n"
                 << "  \"depth_step\": \"current-once\",\n"
                 << "  \"depth_reuse_interval\": 1,\n"
                 << "  \"pop_strength\": " << sbs_cfg.pop_strength << ",\n";
        if (!direct_parallax_mode) {
          const auto &provenance = *est.raw_model_provenance;
          contract
            << "  \"raw_model_provenance\": {"
               "\"schema\": 1, \"model\": "
            << json_string(provenance.depth_model)
            << ", \"depth_model_url\": " << json_string(provenance.depth_model_url)
            << ", \"onnx_sha256\": " << json_string(provenance.onnx_sha256)
            << ", \"preprocess_profile\": "
            << json_string(provenance.preprocess_profile)
            << ", \"preprocess_source_closure_sha256\": "
            << json_string(provenance.preprocess_source_closure_sha256)
            << ", \"raw_width\": " << est.raw_width
            << ", \"raw_height\": " << est.raw_height << "},\n";
        }
        if (direct_parallax_mode) {
          contract << "  \"warp_input\": "
                   << json_string(direct_geometry_warp_input)
                   << ",\n"
                   << "  \"direct_parallax_frames\": "
                   << applied_direct_parallax_frames << ",\n"
                   << "  \"direct_parallax_manifest\": {\"file\": "
                   << "\"direct_parallax_manifest.json\", \"schema\": "
                   << direct_geometry_manifest_schema
                   << ", \"sha256\": "
                   << json_string(direct_parallax_manifest_sha256) << "},\n"
                   << "  \"direct_parallax\": {\"enabled\": true, "
                      "\"file_pattern\": \"parallax_<frame-id>.f32\", "
                      "\"dtype\": \"float32-le\", \"stored_range\": [0, 1], "
                      "\"zero_encoding\": 0.5, "
                      "\"decode_source_u_one_eye\": \"(encoded * 2 - 1) * 0.04\", "
                   << "\"displacement_semantics\": "
                      "\"signed-source-u-positive-nearward-not-order\", "
                      "\"maximum_horizontal_source_u_slope\": "
                   << direct_parallax_max_horizontal_slope << ", "
                      "\"renderer_inverse\": "
                      "\"contractive-fixed-point-11-iterations-v1\", "
                      "\"renderer_uses_order\": false, ";
          contract <<
                      "\"order_file_pattern\": \"order_<frame-id>.f32\", "
                      "\"order_dtype\": \"float32-le\", "
                      "\"order_range\": \"finite-unbounded\", "
                       "\"order_high_is_near\": true, "
                       "\"order_role\": "
                   << json_string("diagnostic-semantic-depth-only-v1")
                   << ", "
                       "\"depth_artifact_file_pattern\": \"depth_<frame-id>.f32\", "
                       "\"depth_artifact_semantics\": "
                       "\"canonical-pre-limiter-order-float32-v1\", "
                      "\"parallax_artifact_file_pattern\": \"parallax_<frame-id>.f32\", "
                      "\"parallax_artifact_semantics\": "
                      "\"encoded-conditioned-final-parallax-float32-v1\", "
                   <<
                       "\"timing_scope\": "
                   << json_string(
                        depth_coordinate_v2_gpu_mode ?
                          "native-depth-coordinate-v2-gpu-replay; tensorrt-not-executed" :
                          "quality-only; production-v2-estimator-executes-for-cut-state-and-shape-evidence"
                      )
                   << "},\n";
        }
        if (depth_coordinate_v2_gpu_mode) {
          contract
            << "  \"depth_coordinate_v2_gpu\": {"
               "\"enabled\": true, "
               "\"execution\": "
            << json_string(sbs_bench::depth_coordinate_v2_gpu_execution)
            << ", "
               "\"tensorrt_executed\": false, "
               "\"input_manifest_sha256\": "
            << json_string(depth_coordinate_v2_gpu->manifest_sha256())
            << ", \"state_trace\": {\"file\": "
               "\"depth_coordinate_v2_state_trace.json\", \"schema\": "
            << sbs_bench::depth_coordinate_v2_state_trace_schema << ", \"sha256\": "
             << json_string(depth_coordinate_v2_state_trace_sha256)
             << "}, \"render_authority\": \"gpu-canonical-and-final-fields\", "
                "\"numpy_role\": \"comparison-only\", "
                "\"calibration_contract_canonical_sha256\": "
             << json_string(models::depth_coordinate_v2::contract_canonical_sha256)
             << ", "
                "\"pop_strength_authority\": \"contract.pop_strength-only\", "
                "\"adaptive_pop_applied\": false},\n";
        }
        if (!direct_parallax_mode) {
          contract
            << "  \"parallax_v2_live\": {\"enabled\": true, "
               "\"renderer\": \"depth-coordinate-v2-live-signed-parallax\", "
               "\"producer_source_closure_sha256\": "
            << json_string(std::string(
                 models::depth_coordinate_v2::shader_source_closure_sha256))
            << ", \"renderer_source_closure_sha256\": "
            << json_string(std::string(
                 models::host_sbs_shader_cache::
                   parallax_v2_live_renderer_source_closure_sha256))
            << ", \"contract_schema\": "
            << models::depth_coordinate_v2::contract_schema
            << ", \"legacy_levers_applied\": false, "
               "\"structure_source\": \"shadow_candidate_parallax\", "
               "\"structure_file_pattern\": \"structure_<frame-id>.png\", "
               "\"structure_normalization\": \"per-frame-finite-minmax-16bit\"},\n";
        }
        contract << "  \"parallax_v2_shadow\": false,\n"
                 << "  \"parallax_v2_render\": "
                 << (external_direct_parallax_mode ? "false" : "true")
                 << ",\n"
                 << "  \"cuda_graph\": " << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
                 << "  \"cuda_graph_captured\": " << (cuda_graph_captured ? "true" : "false") << ",\n"
                 << "  \"cut_state\": {\"file\": \"cut_state.json\", "
                    "\"schema\": " << cut_state_trace_contract.schema
                 << ", \"capture\": "
                 << json_string(cut_state_trace_contract.capture)
                 << "},\n"
                 << "  \"warp_mask\": {\"red\": \"finite_boundary_extrapolation\"},\n"
                 << "  \"warp_mapping\": {\n"
                 << "    \"file_pattern\": \"warp_map_<frame-id>.f32\",\n"
                 << "    \"shape_contract\": \"warp_map_shape.json\",\n"
                 << "    \"dtype\": \"float32-le\",\n"
                 << "    \"layout\": \"row-major\",\n"
                 << "    \"channels\": [\"raw_reproject_source_u_normalized\"],\n"
                 << "    \"live_sample_transform\": \"clamp(raw_reproject_source_u_normalized, 0, 1)\",\n"
                 << "    \"validity_companion\": \"warp_mask_<frame-id>.png:red=finite_boundary_extrapolation; content validity derives from warp_map_shape.json\"\n"
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
      if (writes_adaptive_state) {
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
      } else if (adaptive_state_frame_count != 0u) {
        BOOST_LOG(error) << "sbs-bench: cache replay fabricated adaptive-state evidence";
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
                         "scene-cache-contract-schema-" +
                           std::to_string(offline_sbs::scene_cache_contract_schema) +
                           ":signed-final-parallax-R32_FLOAT" :
                         std::string {"video_depth_estimator"})
        << ",\n"
        << "  \"pipeline_state_provenance\": "
        << json_string(replay_mode ?
                         "scene-cache-contract-schema-" +
                           std::to_string(offline_sbs::scene_cache_contract_schema) +
                           ":parallax-state-12-words" :
                         std::string {
                           "video_depth_estimator:cut-and-health-state-32-words"
                         })
        << ",\n"
        << "  \"model\": " << json_string(model.name) << ",\n"
        << "  \"source_frame_count\": " << completed_frame_count << ",\n"
        << "  \"source_width\": " << source_width << ",\n"
        << "  \"source_height\": " << source_height << ",\n"
        << "  \"source_first_sequence\": " << follow_first_sequence << ",\n"
        << "  \"depth_reuse_interval\": " << effective_depth_every << ",\n"
        << "  \"resolved_runtime\": {\n"
        << "    \"model\": " << json_string(model.name) << ",\n"
        << "    \"model_url\": " << json_string(model.url) << ",\n"
        << "    \"pop_strength\": " << sbs_cfg.pop_strength << ",\n"
        << "    \"depth_width\": " << est.raw_width << ",\n"
        << "    \"depth_height\": " << est.raw_height << ",\n"
        << "    \"depth_reuse_interval\": " << effective_depth_every << ",\n"
        << "    \"cuda_graph\": "
        << (sbs_cfg.cuda_graph ? "true" : "false") << ",\n"
        << "    \"parallax_v2_shadow\": false,\n"
        << "    \"parallax_v2_render\": "
        << "true,\n"
        << "    \"parallax_v2_live\": true,\n"
        << "    \"cuda_graph_captured\": "
        << (cuda_graph_captured ? "true" : "false") << ",\n"
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
        << "    \"scene_cache_write\": "
        << (!o.scene_cache.empty() ? "true" : "false") << ",\n"
        << "    \"scene_cache_replay\": "
        << (replay_mode ? "true" : "false") << ",\n"
        << "    \"scene_cache_contract_schema\": "
        << ((!o.scene_cache.empty() || replay_mode) ?
              std::to_string(offline_sbs::scene_cache_contract_schema) :
              "null")
        << ",\n"
        << "    \"scene_plan_schema\": "
        << (replay_mode ? "2" : "null") << ",\n"
        << "    \"scene_plan_version\": "
        << (replay_mode ? json_string("scene-plan-v2") : "null") << ",\n"
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
        << "  \"adaptive_state\": {";
      if (replay_mode) {
        // The analysis child already emitted the authoritative cut trace. Replaying cached V2
        // geometry runs no cut resolver, so publishing a synthetic CutBridgeState trace here
        // would be false attribution and unnecessary durable I/O.
        contract
          << "\"transport\":\"none\","
             "\"retained_history\":false,"
             "\"frame_count\":0";
      } else {
        contract
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
          << ",\"frame_count\":" << adaptive_state_frame_count;
      }
      contract
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

}  // namespace sbs_bench

#else  // !_WIN32
namespace sbs_bench {
  int run(int, char **) {
    return 1;
  }
}  // namespace sbs_bench
#endif
